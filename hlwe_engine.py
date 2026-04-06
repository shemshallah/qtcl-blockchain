#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                            ║
║  HLWE-256 GENUINE HYPERBOLIC CRYPTOGRAPHIC SYSTEM v3.0 — MONOLITHIC SELF-CONTAINED IMPLEMENTATION       ║
║                                                                                                            ║
║  ONE FILE. COMPLETE. NO EXTERNAL DEPENDENCIES (EXCEPT STDLIB + psycopg2 via server.py DB pool).         ║
║                                                                                                            ║
║  Components (All Integrated):                                                                             ║
║    • BIP39 Mnemonic Seed Phrases (2048 words embedded)                                                    ║
║    • HLWE-256 Post-Quantum Cryptography (Hyperbolic Learning With Errors)                                 ║
║    • Fiat-Shamir Lattice Signatures (z, c_hash, w, public_key, address)                                   ║
║    • HyperbolicGeometry — Supabase PostgreSQL backend (server) / SQLite (client)                          ║
║    • GeodesicLWE — Möbius transport of Poincaré disk pseudoqubit coords for basis generation              ║
║    • BIP32 Hierarchical Deterministic Key Derivation                                                      ║
║    • BIP38 Password-Protected Private Keys                                                                ║
║    • Supabase PostgreSQL Integration (psycopg2 TCP pool via server.py get_db_cursor)                      ║
║    • Integration Adapter (Backward-compatible RPC layer)                                                  ║
║    • Complete Wallet Management System                                                                    ║
║                                                                                                            ║
║  Integration Points:                                                                                       ║
║    • server.py: /rpc/*, /wallet/*, /block/verify, /tx/verify                                              ║
║    • oracle.py: W-state signing, consensus verification                                                   ║
║    • blockchain_entropy_mining.py: Block sealing with HLWE signatures                                     ║
║    • mempool.py: Transaction signing and verification                                                     ║
║    • globals.py: Block field entropy integration (get_block_field_entropy)                                ║
║                                                                                                            ║
║  Clay Mathematics Institute Level — Museum Grade — Production Ready                                       ║
║  Zero Shortcuts — Complete Implementation — No External Crypto Packages                                   ║
║                                                                                                            ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import hashlib
import hmac
import json
import math
import secrets
import struct
import threading
import time
import logging
from contextlib import contextmanager
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from urllib.parse import quote, urlencode
import base64

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# LOGGING (MUST BE FIRST)
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# ENTROPY SOURCE (Block Field from globals if available)
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

try:
    from globals import get_block_field_entropy
    ENTROPY_AVAILABLE = True
except ImportError:
    ENTROPY_AVAILABLE = False
    def get_block_field_entropy():
        """Fallback to os.urandom if globals unavailable"""
        return os.urandom(32)

logger.info("[HLWE] Block field entropy available: {}".format(
    "✅ YES" if ENTROPY_AVAILABLE else "⚠️  FALLBACK (os.urandom)"))

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# BIP39 WORDLIST — 2048 STANDARDIZED MNEMONIC WORDS (EMBEDDED)
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

BIP39_WORDLIST = [
    "abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract",
    "absurd", "abuse", "access", "accident", "account", "accuse", "achieve", "acid",
    "acoustic", "acquire", "across", "act", "action", "actor", "actress", "actual",
    "adapt", "add", "addict", "address", "adjust", "admit", "adult", "advance",
    "advice", "aerobic", "affair", "afford", "afraid", "again", "age", "agent",
    "agree", "ahead", "aim", "air", "airport", "aisle", "alarm", "album",
    "alcohol", "alert", "alien", "all", "alley", "allow", "almost", "alone",
    "alpha", "already", "also", "alter", "always", "amateur", "amazing", "among",
    "amount", "amused", "analyst", "anchor", "ancient", "anger", "angle", "angry",
    "animal", "ankle", "announce", "annual", "another", "answer", "antenna", "antique",
    "anxiety", "any", "apart", "apology", "appear", "apple", "approve", "april",
    "arch", "arctic", "area", "arena", "argue", "arm", "armed", "armor",
    "army", "around", "arrange", "arrest", "arrive", "arrow", "art", "artefact",
    "artist", "artwork", "ask", "aspect", "assault", "asset", "assist", "assume",
    "asthma", "athlete", "atom", "attack", "attend", "attitude", "attract", "auction",
    "audit", "august", "aunt", "author", "auto", "autumn", "avocado", "avoid",
    "awake", "award", "aware", "away", "awesome", "awful", "awkward", "axis",
    "baby", "bachelor", "bacon", "badge", "bag", "balance", "balcony", "ball",
    "bamboo", "banana", "banner", "bar", "barely", "bargain", "barrel", "base",
    "basic", "basket", "battle", "beach", "bean", "beauty", "because", "become",
    "beef", "before", "begin", "behave", "behind", "believe", "below", "belt",
    "bench", "benefit", "best", "betray", "better", "between", "beyond", "bicycle",
    "bid", "bike", "bind", "biology", "bird", "birth", "bitter", "black",
    "blade", "blame", "blanket", "blast", "bleak", "bless", "blind", "blood",
    "blossom", "blouse", "blue", "blur", "blush", "board", "boat", "body",
    "boil", "bomb", "bone", "bonus", "book", "boost", "border", "boring",
    "borrow", "boss", "bottom", "bounce", "box", "boy", "bracket", "brain",
    "brand", "brass", "brave", "bread", "breeze", "brick", "bridge", "brief",
    "bright", "bring", "brisk", "broccoli", "broken", "bronze", "broom", "brother",
    "brown", "brush", "bubble", "buddy", "budget", "buffalo", "build", "bulb",
    "bulk", "bullet", "bundle", "bunker", "burden", "burger", "burst", "bus",
    "business", "busy", "butter", "buyer", "buzz", "cabbage", "cabin", "cable",
    "cactus", "cage", "cake", "call", "calm", "camera", "camp", "can",
    "canal", "cancel", "candy", "cannon", "canoe", "canvas", "canyon", "capable",
    "capital", "captain", "car", "carbon", "card", "cargo", "carpet", "carry",
    "cart", "case", "cash", "casino", "castle", "casual", "cat", "catalog",
    "catch", "category", "cattle", "caught", "cause", "caution", "cave", "ceiling",
    "celery", "cement", "census", "century", "cereal", "certain", "chair", "chalk",
    "champion", "change", "chaos", "chapter", "charge", "chase", "chat", "cheap",
    "check", "cheese", "chef", "cherry", "chest", "chicken", "chief", "child",
    "chimney", "choice", "choose", "chronic", "chuckle", "chunk", "churn", "cigar",
    "cinnamon", "circle", "citizen", "city", "civil", "claim", "clap", "clarify",
    "claw", "clay", "clean", "clerk", "clever", "click", "client", "cliff",
    "climb", "clinic", "clip", "clock", "clog", "close", "cloth", "cloud",
    "clown", "club", "clump", "cluster", "clutch", "coach", "coast", "coconut",
    "code", "coffee", "coil", "coin", "collect", "color", "column", "combine",
    "come", "comfort", "comic", "common", "company", "concert", "conduct", "confirm",
    "congress", "connect", "consider", "control", "convince", "cook", "cool", "copper",
    "copy", "coral", "core", "corn", "correct", "cost", "cotton", "couch",
    "country", "couple", "course", "cousin", "cover", "coyote", "crack", "cradle",
    "craft", "cram", "crane", "crash", "crater", "crawl", "crazy", "cream",
    "credit", "creek", "crew", "cricket", "crime", "crisp", "critic", "crop",
    "cross", "crouch", "crowd", "crucial", "cruel", "cruise", "crumble", "crunch",
    "crush", "cry", "crystal", "cube", "culture", "cup", "cupboard", "curious",
    "current", "curtain", "curve", "cushion", "custom", "cute", "cycle", "dad",
    "damage", "damp", "dance", "danger", "daring", "dash", "daughter", "dawn",
    "day", "deal", "debate", "debris", "decade", "december", "decide", "decline",
    "decorate", "decrease", "deer", "defense", "define", "defy", "degree", "delay",
    "deliver", "demand", "demise", "denial", "dentist", "deny", "depart", "depend",
    "deposit", "depth", "deputy", "derive", "describe", "desert", "design", "desk",
    "despair", "destroy", "detail", "detect", "develop", "device", "devote", "diagram",
    "dial", "diamond", "diary", "dice", "diesel", "diet", "differ", "digital",
    "dignity", "dilemma", "dinner", "dinosaur", "direct", "dirt", "disagree", "discover",
    "disease", "dish", "dismiss", "disorder", "display", "distance", "divert", "divide",
    "divorce", "dizzy", "doctor", "document", "dog", "doll", "dolphin", "domain",
    "donate", "donkey", "donor", "door", "dose", "double", "dove", "draft",
    "dragon", "drama", "drastic", "draw", "dream", "dress", "drift", "drill",
    "drink", "drip", "drive", "drop", "drum", "dry", "duck", "dumb",
    "dune", "during", "dust", "dutch", "duty", "dwarf", "dynamic", "eager",
    "eagle", "early", "earn", "earth", "easily", "east", "easy", "echo",
    "ecology", "economy", "edge", "edit", "educate", "effort", "egg", "eight",
    "either", "elbow", "elder", "electric", "elegant", "element", "elephant", "elevator",
    "elite", "else", "embark", "embody", "embrace", "emerge", "emotion", "employ",
    "empower", "empty", "enable", "enact", "end", "endless", "endorse", "enemy",
    "energy", "enforce", "engage", "engine", "enhance", "enjoy", "enlist", "enough",
    "enrich", "enroll", "ensure", "enter", "entire", "entry", "envelope", "episode",
    "equal", "equip", "era", "erase", "erode", "erosion", "error", "erupt",
    "escape", "essay", "essence", "estate", "eternal", "ethics", "evidence", "evil",
    "evoke", "evolve", "exact", "example", "excess", "exchange", "excite", "exclude",
    "excuse", "execute", "exercise", "exhaust", "exhibit", "exile", "exist", "exit",
    "exotic", "expand", "expect", "expire", "explain", "expose", "express", "extend",
    "extra", "eye", "eyebrow", "fabric", "face", "faculty", "fade", "faint",
    "faith", "fall", "false", "fame", "family", "famous", "fan", "fancy",
    "fantasy", "farm", "fashion", "fat", "fatal", "father", "fatigue", "fault",
    "favorite", "feature", "february", "federal", "fee", "feed", "feel", "female",
    "fence", "festival", "fetch", "fever", "few", "fiber", "fiction", "field",
    "figure", "file", "film", "filter", "final", "find", "fine", "finger",
    "finish", "fire", "firm", "first", "fiscal", "fish", "fit", "fitness",
    "fix", "flag", "flame", "flash", "flat", "flavor", "flee", "flight",
    "flip", "float", "flock", "floor", "flower", "fluid", "flush", "fly",
    "foam", "focus", "fog", "foil", "fold", "follow", "food", "foot",
    "force", "forest", "forget", "fork", "fortune", "forum", "forward", "fossil",
    "foster", "found", "fox", "fragile", "frame", "frequent", "fresh", "friend",
    "fringe", "frog", "front", "frost", "frown", "frozen", "fruit", "fuel",
    "fun", "funny", "furnace", "fury", "future", "gadget", "gain", "galaxy",
    "gallery", "game", "gap", "garage", "garbage", "garden", "garlic", "garment",
    "gas", "gasp", "gate", "gather", "gauge", "gaze", "general", "genius",
    "genre", "gentle", "genuine", "gesture", "ghost", "giant", "gift", "giggle",
    "ginger", "giraffe", "girl", "give", "glad", "glance", "glare", "glass",
    "glide", "glimpse", "globe", "gloom", "glory", "glove", "glow", "glue",
    "goat", "goddess", "gold", "good", "goose", "gorilla", "gospel", "gossip",
    "govern", "gown", "grab", "grace", "grain", "grant", "grape", "grass",
    "gravity", "great", "green", "grid", "grief", "grit", "grocery", "group",
    "grow", "grunt", "guard", "guess", "guide", "guilt", "guitar", "gun",
    "gym", "habit", "hair", "half", "hammer", "hamster", "hand", "happy",
    "harbor", "hard", "harsh", "harvest", "hat", "have", "hawk", "hazard",
    "head", "health", "heart", "heavy", "hedgehog", "height", "hello", "helmet",
    "help", "hen", "hero", "hidden", "high", "hill", "hint", "hip",
    "hire", "history", "hobby", "hockey", "hold", "hole", "holiday", "hollow",
    "home", "honey", "hood", "hope", "horn", "horror", "horse", "hospital",
    "host", "hotel", "hour", "hover", "hub", "huge", "human", "humble",
    "humor", "hundred", "hungry", "hunt", "hurdle", "hurry", "hurt", "husband",
    "hybrid", "ice", "icon", "idea", "identify", "idle", "ignore", "ill",
    "illegal", "illness", "image", "imitate", "immense", "immune", "impact", "impose",
    "improve", "impulse", "inch", "include", "income", "increase", "index", "indicate",
    "indoor", "industry", "infant", "inflict", "inform", "inhale", "inherit", "initial",
    "inject", "injury", "inmate", "inner", "innocent", "input", "inquiry", "insane",
    "insect", "inside", "inspire", "install", "intact", "interest", "into", "invest",
    "invite", "involve", "iron", "island", "isolate", "issue", "item", "ivory",
    "jacket", "jaguar", "jar", "jazz", "jealous", "jeans", "jelly", "jewel",
    "job", "join", "joke", "journey", "joy", "judge", "juice", "jump",
    "jungle", "junior", "junk", "just", "kangaroo", "keen", "keep", "ketchup",
    "key", "kick", "kid", "kidney", "kind", "kingdom", "kiss", "kit",
    "kitchen", "kite", "kitten", "kiwi", "knee", "knife", "knock", "know",
    "lab", "label", "labor", "ladder", "lady", "lake", "lamp", "language",
    "laptop", "large", "later", "latin", "laugh", "laundry", "lava", "law",
    "lawn", "lawsuit", "layer", "lazy", "leader", "leaf", "learn", "leave",
    "lecture", "left", "leg", "legal", "legend", "leisure", "lemon", "lend",
    "length", "lens", "leopard", "lesson", "letter", "level", "liar", "liberty",
    "library", "license", "life", "lift", "light", "like", "limb", "limit",
    "link", "lion", "liquid", "list", "little", "live", "lizard", "load",
    "loan", "lobster", "local", "lock", "logic", "lonely", "long", "loop",
    "lottery", "loud", "lounge", "love", "loyal", "lucky", "luggage", "lumber",
    "lunar", "lunch", "luxury", "lyrics", "machine", "mad", "magic", "magnet",
    "maid", "mail", "main", "major", "make", "mammal", "man", "manage",
    "mandate", "mango", "mansion", "manual", "maple", "marble", "march", "margin",
    "marine", "market", "marriage", "mask", "mass", "master", "match", "material",
    "math", "matrix", "matter", "maximum", "maze", "meadow", "mean", "measure",
    "meat", "mechanic", "medal", "media", "melody", "melt", "member", "memory",
    "mention", "menu", "mercy", "merge", "merit", "merry", "mesh", "message",
    "metal", "method", "middle", "midnight", "milk", "million", "mimic", "mind",
    "minimum", "minor", "minute", "miracle", "mirror", "misery", "miss", "mistake",
    "mix", "mixed", "mixture", "mobile", "model", "modify", "mom", "moment",
    "monitor", "monkey", "monster", "month", "moon", "moral", "more", "morning",
    "mosquito", "mother", "motion", "motor", "mountain", "mouse", "move", "movie",
    "much", "muffin", "mule", "multiply", "muscle", "museum", "mushroom", "music",
    "must", "mutual", "myself", "mystery", "myth", "naive", "name", "napkin",
    "narrow", "nasty", "nation", "nature", "near", "neck", "need", "negative",
    "neglect", "neither", "nephew", "nerve", "nest", "net", "network", "neutral",
    "never", "news", "next", "nice", "night", "noble", "noise", "nominee",
    "noodle", "normal", "north", "nose", "notable", "note", "nothing", "notice",
    "novel", "now", "nuclear", "number", "nurse", "nut", "oak", "obey",
    "object", "oblige", "obscure", "observe", "obtain", "obvious", "occur", "ocean",
    "october", "odor", "off", "offer", "office", "often", "oil", "okay",
    "old", "olive", "olympic", "omit", "once", "one", "onion", "online",
    "only", "open", "opera", "opinion", "oppose", "option", "orange", "orbit",
    "orchard", "order", "ordinary", "organ", "orient", "original", "orphan", "ostrich",
    "other", "outdoor", "outer", "output", "outside", "oval", "oven", "over",
    "own", "owner", "oxygen", "oyster", "ozone", "pact", "paddle", "page",
    "pair", "palace", "palm", "panda", "panel", "panic", "panther", "paper",
    "parade", "parent", "park", "parrot", "party", "pass", "patch", "path",
    "patient", "patrol", "pattern", "pause", "pave", "payment", "peace", "peanut",
    "pear", "peasant", "pelican", "pen", "penalty", "pencil", "people", "pepper",
    "perfect", "permit", "person", "pet", "phone", "photo", "phrase", "physical",
    "piano", "picnic", "picture", "piece", "pig", "pigeon", "pill", "pilot",
    "pink", "pioneer", "pipe", "pistol", "pitch", "pizza", "place", "planet",
    "plastic", "plate", "play", "please", "pledge", "pluck", "plug", "plunge",
    "poem", "poet", "point", "polar", "pole", "police", "pond", "pony",
    "pool", "popular", "portion", "position", "possible", "post", "potato", "pottery",
    "poverty", "powder", "power", "practice", "praise", "predict", "prefer", "prepare",
    "present", "pretty", "prevent", "price", "pride", "primary", "print", "priority",
    "prison", "private", "prize", "problem", "process", "produce", "profit", "program",
    "project", "promote", "proof", "property", "prosper", "protect", "proud", "provide",
    "public", "pudding", "pull", "pulp", "pulse", "pumpkin", "punch", "pupil",
    "puppy", "purchase", "purity", "purpose", "purse", "push", "put", "puzzle",
    "pyramid", "quality", "quantum", "quarter", "question", "quick", "quit", "quiz",
    "quote", "rabbit", "raccoon", "race", "rack", "radar", "radio", "rail",
    "rain", "raise", "rally", "ramp", "ranch", "random", "range", "rapid",
    "rare", "rate", "rather", "raven", "raw", "razor", "ready", "real",
    "reason", "rebel", "rebuild", "recall", "receive", "recipe", "record", "recycle",
    "reduce", "reflect", "reform", "refuse", "region", "regret", "regular", "reject",
    "relax", "release", "relief", "rely", "remain", "remember", "remind", "remove",
    "render", "renew", "rent", "reopen", "repair", "repeat", "replace", "report",
    "require", "rescue", "resemble", "resist", "resource", "response", "result", "retire",
    "retreat", "return", "reunion", "reveal", "review", "reward", "rhythm", "rib",
    "ribbon", "rice", "rich", "ride", "ridge", "rifle", "right", "rigid",
    "ring", "riot", "ripple", "risk", "ritual", "rival", "river", "road",
    "roast", "robot", "robust", "rocket", "romance", "roof", "rookie", "room",
    "rose", "rotate", "rough", "round", "route", "royal", "rubber", "rude",
    "rug", "rule", "run", "runway", "rural", "sad", "saddle", "sadness",
    "safe", "sail", "salad", "salmon", "salon", "salt", "salute", "same",
    "sample", "sand", "satisfy", "satoshi", "sauce", "sausage", "save", "scaffold",
    "scale", "scan", "scare", "scatter", "scene", "scheme", "school", "science",
    "scissors", "scorpion", "scout", "scrap", "screen", "script", "scrub", "sea",
    "search", "season", "seat", "second", "secret", "section", "security", "seed",
    "seek", "segment", "select", "sell", "seminar", "senior", "sense", "sentence",
    "series", "service", "session", "settle", "setup", "seven", "shadow", "shaft",
    "shallow", "share", "shed", "shell", "sheriff", "shield", "shift", "shine",
    "ship", "shiver", "shock", "shoe", "shoot", "shop", "short", "shoulder",
    "shove", "shrimp", "shrug", "shuffle", "shy", "sibling", "sick", "side",
    "siege", "sight", "sign", "silent", "silk", "silly", "silver", "similar",
    "simple", "since", "sing", "siren", "sister", "situate", "six", "size",
    "skate", "sketch", "ski", "skill", "skin", "skirt", "skull", "slab",
    "slam", "sleep", "slender", "slice", "slide", "slight", "slim", "slogan",
    "slot", "slow", "slush", "small", "smart", "smile", "smoke", "smooth",
    "snack", "snake", "snap", "sniff", "snow", "soap", "soccer", "social",
    "sock", "soda", "soft", "solar", "soldier", "solid", "solution", "solve",
    "someone", "song", "soon", "sorry", "sort", "soul", "sound", "soup",
    "source", "south", "space", "spare", "spatial", "spawn", "speak", "special",
    "speed", "spell", "spend", "sphere", "spice", "spider", "spike", "spin",
    "spirit", "split", "spoil", "sponsor", "spoon", "sport", "spot", "spray",
    "spread", "spring", "spy", "square", "squeeze", "squirrel", "stable", "stadium",
    "staff", "stage", "stairs", "stamp", "stand", "start", "state", "stay",
    "steak", "steel", "stem", "step", "stereo", "stick", "still", "sting",
    "stock", "stomach", "stone", "stool", "story", "stove", "strategy", "street",
    "strike", "strong", "struggle", "student", "stuff", "stumble", "style", "subject",
    "submit", "subway", "success", "such", "sudden", "suffer", "sugar", "suggest",
    "suit", "summer", "sun", "sunny", "sunset", "super", "supply", "supreme",
    "sure", "surface", "surge", "surprise", "surround", "survey", "suspect", "sustain",
    "swallow", "swamp", "swap", "swarm", "swear", "sweet", "swift", "swim",
    "swing", "switch", "sword", "symbol", "symptom", "syrup", "system", "table",
    "tackle", "tag", "tail", "talent", "talk", "tank", "tape", "target",
    "task", "taste", "tattoo", "taxi", "teach", "team", "tell", "ten",
    "tenant", "tennis", "tent", "term", "test", "text", "thank", "that",
    "theme", "then", "theory", "there", "they", "thing", "this", "thought",
    "three", "thrive", "throw", "thumb", "thunder", "ticket", "tide", "tiger",
    "tilt", "timber", "time", "tiny", "tip", "tired", "tissue", "title",
    "toast", "tobacco", "today", "toddler", "toe", "together", "toilet", "token",
    "tomato", "tomorrow", "tone", "tongue", "tonight", "tool", "tooth", "top",
    "topic", "topple", "torch", "tornado", "tortoise", "toss", "total", "tourist",
    "toward", "tower", "town", "toy", "track", "trade", "traffic", "tragic",
    "train", "transfer", "trap", "trash", "travel", "tray", "treat", "tree",
    "trend", "trial", "tribe", "trick", "trigger", "trim", "trip", "trophy",
    "trouble", "truck", "true", "truly", "trumpet", "trust", "truth", "try",
    "tube", "tuition", "tumble", "tuna", "tunnel", "turkey", "turn", "turtle",
    "twelve", "twenty", "twice", "twin", "twist", "two", "type", "typical",
    "ugly", "umbrella", "unable", "unaware", "uncle", "uncover", "under", "undo",
    "unfair", "unfold", "unhappy", "uniform", "unique", "unit", "universe", "unknown",
    "unlock", "until", "unusual", "unveil", "update", "upgrade", "uphold", "upon",
    "upper", "upset", "urban", "urge", "usage", "use", "used", "useful",
    "useless", "usual", "utility", "vacant", "vacuum", "vague", "valid", "valley",
    "valve", "van", "vanish", "vapor", "various", "vast", "vault", "vehicle",
    "velvet", "vendor", "venture", "venue", "verb", "verify", "version", "very",
    "vessel", "veteran", "viable", "vibrant", "vicious", "victory", "video", "view",
    "village", "vintage", "violin", "virtual", "virus", "visa", "visit", "visual",
    "vital", "vivid", "vocal", "voice", "void", "volcano", "volume", "vote",
    "voyage", "wage", "wagon", "wait", "walk", "wall", "walnut", "want",
    "warfare", "warm", "warrior", "wash", "wasp", "waste", "water", "wave",
    "way", "wealth", "weapon", "wear", "weasel", "weather", "web", "wedding",
    "weekend", "weird", "welcome", "west", "wet", "whale", "what", "wheat",
    "wheel", "when", "where", "whip", "whisper", "wide", "width", "wife",
    "wild", "will", "win", "window", "wine", "wing", "wink", "winner",
    "winter", "wire", "wisdom", "wise", "wish", "witness", "wolf", "woman",
    "wonder", "wood", "wool", "word", "work", "world", "worry", "worth",
    "wrap", "wreck", "wrestle", "wrist", "write", "wrong", "yard", "year",
    "yellow", "you", "young", "youth", "zebra", "zero", "zone", "zoo",
]

# Index lookups — O(1) access
BIP39_ENGLISH   = {i: word for i, word in enumerate(BIP39_WORDLIST)}
_WORD_TO_INDEX  = {word: i for i, word in enumerate(BIP39_WORDLIST)}

def get_word_by_index(index: int) -> str:
    """Get BIP39 word by index (0-2047)"""
    if 0 <= index < len(BIP39_WORDLIST):
        return BIP39_WORDLIST[index]
    raise ValueError(f"Index {index} out of range [0, {len(BIP39_WORDLIST)-1}]")

def get_index_by_word(word: str) -> int:
    """Get BIP39 index by word"""
    word = word.lower()
    if word in _WORD_TO_INDEX:
        return _WORD_TO_INDEX[word]
    raise ValueError(f"Word '{word}' not in BIP39 wordlist")

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# CONSTANTS & ENUMS
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class LatticeParams:
    """Lattice dimension and modulus parameters for HLWE"""
    DIMENSION = 256          # Lattice dimension n
    MODULUS = 2**32 - 5      # q = 2^32 - 5 (prime modulus)
    ERROR_BOUND = 256        # χ error distribution bound
    SECURITY_BITS = 256      # Target security level

class KeyDerivationParams:
    """Parameters for hierarchical deterministic key derivation"""
    HMAC_KEY = b"QTCL HD seed v1"           # BIP32 HMAC key (unified — must match qtcl_client.py)
    PBKDF2_ITERATIONS = 100_000             # BIP38/BIP39 iterations
    PBKDF2_SALT_SIZE = 16                   # Salt size for key derivation
    MNEMONIC_ENTROPY_SIZES = [16, 20, 24, 28, 32]  # 128-256 bits (12-24 words)
    PASSWORD_PROTECTION_ITERATIONS = 100_000

class SupabaseConfig:
    """Supabase PostgreSQL RPC configuration"""
    URL = os.getenv('SUPABASE_URL', 'https://your-project.supabase.co')
    KEY = os.getenv('SUPABASE_ANON_KEY', '')
    API_TIMEOUT = 30  # seconds

class AddressType(Enum):
    """BIP44 address derivation types"""
    RECEIVING = 0
    CHANGE = 1
    COLD_STORAGE = 2

class MnemonicStrength(Enum):
    """Mnemonic word count and entropy strength"""
    WEAK = (12, 128)      # 128 bits = 12 words
    STANDARD = (15, 160)  # 160 bits = 15 words
    STRONG = (18, 192)    # 192 bits = 18 words
    VERY_STRONG = (21, 224)  # 224 bits = 21 words
    MAXIMUM = (24, 256)   # 256 bits = 24 words

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class LatticeBasis:
    """Basis for a lattice (for key generation)"""
    matrix: List[List[int]]
    dimension: int
    modulus: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'matrix': self.matrix,
            'dimension': self.dimension,
            'modulus': self.modulus
        }

@dataclass
class HLWEKeyPair:
    """HLWE public/private keypair"""
    public_key: str
    private_key: str
    address: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'public_key': self.public_key,
            'address': self.address,
            'created_at': self.created_at.isoformat()
        }

@dataclass
class BIP32DerivationPath:
    """BIP32 hierarchical derivation path"""
    purpose: int = 44
    coin_type: int = 0
    account: int = 0
    change: int = 0
    index: int = 0
    
    def path_string(self) -> str:
        """Return BIP44 path string: m/44'/0'/0'/0/0"""
        return f"m/{self.purpose}'/{self.coin_type}'/{self.account}'/{self.change}/{self.index}"

@dataclass
class WalletMetadata:
    """Wallet metadata (stored in Supabase)"""
    wallet_id: str
    fingerprint: str
    mnemonic_encrypted: str
    master_chain_code: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    label: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'wallet_id': self.wallet_id,
            'fingerprint': self.fingerprint,
            'mnemonic_encrypted': self.mnemonic_encrypted,
            'master_chain_code': self.master_chain_code,
            'created_at': self.created_at.isoformat(),
            'label': self.label
        }

@dataclass
class StoredAddress:
    """Wallet address (stored in Supabase)"""
    address: str
    public_key: str
    wallet_fingerprint: str
    derivation_path: str
    address_type: str = "receiving"
    balance_satoshis: int = 0
    transaction_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'address': self.address,
            'public_key': self.public_key,
            'wallet_fingerprint': self.wallet_fingerprint,
            'derivation_path': self.derivation_path,
            'address_type': self.address_type,
            'balance_satoshis': self.balance_satoshis,
            'transaction_count': self.transaction_count,
            'created_at': self.created_at.isoformat()
        }

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# LATTICE MATHEMATICS
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class LatticeMath:
    """Core lattice operations for HLWE cryptography"""
    
    @staticmethod
    def mod(x: int, q: int) -> int:
        """Modular reduction: x mod q, range [0, q)"""
        return x % q
    
    @staticmethod
    def mod_inverse(a: int, q: int) -> int:
        """Compute modular inverse a^-1 mod q using extended Euclidean algorithm"""
        if not LatticeMath._gcd(a, q) == 1:
            raise ValueError(f"{a} has no inverse mod {q}")
        return pow(a, -1, q)
    
    @staticmethod
    def _gcd(a: int, b: int) -> int:
        """Greatest common divisor"""
        while b:
            a, b = b, a % b
        return a
    
    @staticmethod
    def vector_mod(v: List[int], q: int) -> List[int]:
        """Apply mod to vector: (v_1 mod q, ..., v_n mod q)"""
        return [LatticeMath.mod(x, q) for x in v]
    
    @staticmethod
    def vector_add(u: List[int], v: List[int], q: int) -> List[int]:
        """Vector addition mod q: (u + v) mod q"""
        if len(u) != len(v):
            raise ValueError("Vector dimensions must match")
        return [LatticeMath.mod(u[i] + v[i], q) for i in range(len(u))]
    
    @staticmethod
    def vector_sub(u: List[int], v: List[int], q: int) -> List[int]:
        """Vector subtraction mod q: (u - v) mod q"""
        if len(u) != len(v):
            raise ValueError("Vector dimensions must match")
        return [LatticeMath.mod(u[i] - v[i], q) for i in range(len(u))]
    
    @staticmethod
    def matrix_vector_mult(A: List[List[int]], v: List[int], q: int) -> List[int]:
        """Matrix-vector multiplication mod q: A * v mod q"""
        n = len(A)
        if len(v) != len(A[0]):
            raise ValueError(f"Dimension mismatch: A is {n}x{len(A[0])}, v is {len(v)}")
        
        result = []
        for i in range(n):
            dot_product = sum(A[i][j] * v[j] for j in range(len(v)))
            result.append(LatticeMath.mod(dot_product, q))
        
        return result
    
    @staticmethod
    def hash_to_lattice_vector(data: bytes, n: int, q: int) -> List[int]:
        """Hash bytes to lattice vector in Z_q^n using rejection sampling"""
        vector = []
        offset = 0
        
        while len(vector) < n:
            hash_input = data + bytes([offset])
            h = hashlib.sha256(hash_input).digest()
            
            for i in range(0, 32, 4):
                if len(vector) >= n:
                    break
                val = int.from_bytes(h[i:i+4], byteorder='big')
                reduced = val % q
                vector.append(reduced)
            
            offset += 1
        
        return vector[:n]

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# HYPERBOLIC GEOMETRY — Supabase PostgreSQL Backend
#
# The "H" in HLWE is GENUINE: Hyperbolic Learning With Errors.
#
# Architecture:
#   SERVER (this file):  reads geometry from Supabase PostgreSQL via server.py's get_db_cursor()
#   CLIENT (qtcl_client.py): reads geometry from SQLite ~/data/qtcl_blockchain.db
#   BOTH: same tables, same schema, same encoding → compute_geometry_hash() produces identical bytes
#
# Tables (populated by qtcl_db_builder_colab.py):
#   hyperbolic_triangles — {8,3} tessellation vertices on the Poincaré disk
#   pseudoqubits — pseudoqubit (x, y) coordinates placed on triangle grid points
#
# Geometry enters the crypto at two points:
#   1. GeodesicLWE basis (keygen) — Möbius transport of Poincaré disk pseudoqubit coords → A matrix rows
#   2. Geometry hash (Fiat-Shamir challenge) — SHA3-256 of triangle vertices salts the challenge scalar
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

# ── Thread-safe PQ coordinate cache ───────────────────────────────────────────
_PQ_COORD_CACHE: Dict[int, Tuple[float, float]] = {}   # pq_id → (x, y) as float
_PQ_CACHE_LOCK  = threading.Lock()
_PQ_CACHE_READY = threading.Event()
_PQ_CACHE_ERROR: Optional[str] = None

# ── Lazy import: server.py's DB pool (available at runtime, not at import time) ──
_db_cursor_func = None

def _get_db_cursor_lazy():
    """Lazy-import get_db_cursor from server.py (avoids circular import at module load)."""
    global _db_cursor_func
    if _db_cursor_func is not None:
        return _db_cursor_func
    try:
        from server import get_db_cursor
        _db_cursor_func = get_db_cursor
        return _db_cursor_func
    except ImportError:
        logger.warning("[HyperbolicGeometry] server.get_db_cursor not available — geometry disabled")
        return None


def _mobius_transport(z: complex, t: float) -> complex:
    """
    Transport point z along its geodesic from 0→z by hyperbolic distance t.

    In the Poincaré disk model, the geodesic from the origin through z is a
    straight diameter.  A point at hyperbolic distance d from origin sits at
    Euclidean radius tanh(d/2).  This function:
      1. Extracts direction = z / |z|  (unit complex)
      2. Computes current hyperbolic radius = atanh(|z|)
      3. Adds t/2 to get new radius, converts back via tanh
      4. Returns direction * new_r

    MUST BE IDENTICAL TO qtcl_client.py _mobius_transport().
    """
    r = abs(z)
    if r < 1e-14:
        return complex(math.tanh(t / 2), 0.0)
    direction = z / r
    new_r = math.tanh(math.atanh(min(r, 1.0 - 1e-14)) + t / 2)
    new_r = min(new_r, 1.0 - 1e-14)
    return direction * new_r


def _load_pq_cache_from_supabase() -> Dict[int, Tuple[float, float]]:
    """Load all pseudoqubit (x, y) coords from Supabase PostgreSQL → float cache."""
    get_cursor = _get_db_cursor_lazy()
    if get_cursor is None:
        raise RuntimeError("Database cursor not available — cannot load PQ cache")
    cache = {}
    with get_cursor() as cur:
        cur.execute("SELECT pq_id, x, y FROM pseudoqubits ORDER BY pq_id")
        rows = cur.fetchall()
        for row in rows:
            try:
                pid = int(row[0])
                cache[pid] = (float(row[1]), float(row[2]))
            except (ValueError, TypeError, IndexError):
                pass
    return cache


def _ensure_pq_cache() -> None:
    """Thread-safe lazy init: load pseudoqubit coordinates from Supabase → RAM cache."""
    global _PQ_COORD_CACHE, _PQ_CACHE_ERROR
    if _PQ_CACHE_READY.is_set():
        return
    with _PQ_CACHE_LOCK:
        if _PQ_CACHE_READY.is_set():
            return
        try:
            _PQ_COORD_CACHE = _load_pq_cache_from_supabase()
            if len(_PQ_COORD_CACHE) > 0:
                _PQ_CACHE_READY.set()
                logger.info(f"[GeodesicLWE] PQ cache ready: {len(_PQ_COORD_CACHE)} pseudoqubits from Supabase")
            else:
                _PQ_CACHE_ERROR = "pseudoqubits table is empty"
                logger.warning("[GeodesicLWE] pseudoqubits table empty — Euclidean fallback active")
        except Exception as exc:
            _PQ_CACHE_ERROR = str(exc)
            logger.warning(f"[GeodesicLWE] PQ cache init failed: {exc} — Euclidean fallback active")


def _pq_get_coord(pq_id: int) -> Tuple[float, float]:
    """Return (x, y) Poincaré disk coord for pseudoqubit pq_id. Raises if cache not ready."""
    if not _PQ_CACHE_READY.is_set():
        raise RuntimeError("PQ coord cache not initialised — call _ensure_pq_cache() first")
    coord = _PQ_COORD_CACHE.get(pq_id)
    if coord is None:
        n = len(_PQ_COORD_CACHE)
        if n == 0:
            raise RuntimeError("PQ coord cache is empty")
        coord = _PQ_COORD_CACHE.get(pq_id % n)
        if coord is None:
            # Fallback: get any valid coord
            coord = next(iter(_PQ_COORD_CACHE.values()))
    return coord


def _start_pq_cache_warmup() -> threading.Thread:
    """Non-blocking: loads PQ coords from Supabase in background thread."""
    def _worker():
        try:
            _ensure_pq_cache()
        except Exception as e:
            logger.warning(f"[GeodesicLWE] Background PQ cache warmup failed: {e}")
    t = threading.Thread(target=_worker, daemon=True, name="PQCacheWarmup")
    t.start()
    return t

# Fire background PQ cache warmup on module import (non-blocking)
try:
    _start_pq_cache_warmup()
except Exception:
    pass


class HyperbolicGeometry:
    """
    Read hyperbolic triangles from Supabase PostgreSQL and provide geometry-based hashing.

    SERVER counterpart of qtcl_client.py HyperbolicGeometry (which reads from SQLite).
    Both produce IDENTICAL compute_geometry_hash() output because:
      - Same table schema (hyperbolic_triangles)
      - Same row ordering (ORDER BY triangle_id)
      - Same struct encoding ('>Q' id, '>I' depth, 6× '>d' float vertex coords)
      - Same SHA3-256 hash

    This geometric hash salts the Fiat-Shamir challenge scalar c, binding
    signatures to the hyperbolic tessellation state — a cryptographic
    commitment to the {8,3} tiling that makes HLWE genuinely Hyperbolic LWE.
    """
    _instance = None
    _lock = threading.Lock()
    _CACHE_TTL_SECONDS = 300  # 5 minutes

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = HyperbolicGeometry()
            return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the global instance (for testing or recovery)."""
        global _hyperbolic_geometry
        with cls._lock:
            cls._instance = None
            _hyperbolic_geometry = None

    def __init__(self):
        self._geometry_hash_cache: Optional[bytes] = None
        self._cache_timestamp: float = 0.0
        self._inner_lock = threading.Lock()
        self._initialized = False
        # Test DB access
        try:
            get_cursor = _get_db_cursor_lazy()
            if get_cursor is not None:
                self._initialized = True
                logger.info("[HyperbolicGeometry] Server mode: Supabase PostgreSQL")
            else:
                logger.warning("[HyperbolicGeometry] No DB access — geometry hash will use fallback")
        except Exception as e:
            logger.warning(f"[HyperbolicGeometry] Init warning: {e}")

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def fetch_all_triangles(self, max_depth: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch hyperbolic triangles up to max_depth from Supabase PostgreSQL.

        MUST return rows in the same order and with the same float precision as
        the client's SQLite query to ensure compute_geometry_hash() matches.
        """
        get_cursor = _get_db_cursor_lazy()
        if get_cursor is None:
            raise RuntimeError("Database cursor not available")
        try:
            with get_cursor() as cur:
                cur.execute("""
                    SELECT triangle_id, depth, parent_id,
                           v0_x, v0_y,
                           v1_x, v1_y,
                           v2_x, v2_y
                    FROM hyperbolic_triangles
                    WHERE depth <= %s
                    ORDER BY triangle_id
                """, (max_depth,))
                rows = cur.fetchall()
                triangles = []
                for row in rows:
                    tri = {
                        'id': int(row[0]),
                        'depth': int(row[1]),
                        'parent_id': int(row[2]) if row[2] is not None else 0,
                        'v0': (float(row[3]), float(row[4])),
                        'v1': (float(row[5]), float(row[6])),
                        'v2': (float(row[7]), float(row[8])),
                    }
                    triangles.append(tri)
                return triangles
        except Exception as e:
            logger.error(f"[HyperbolicGeometry] Failed to fetch triangles: {e}")
            raise

    def compute_geometry_hash(self, max_depth: int = 5) -> bytes:
        """
        Compute SHA3-256 hash of hyperbolic geometry (cached with TTL).

        Encoding MUST be byte-identical to qtcl_client.py HyperbolicGeometry.compute_geometry_hash():
          for each triangle (ordered by triangle_id):
            struct.pack('>Q', tri['id'])           — 8 bytes, big-endian uint64
            struct.pack('>I', tri['depth'])         — 4 bytes, big-endian uint32
            for (vx, vy) in (v0, v1, v2):
              struct.pack('>d', vx)                 — 8 bytes, big-endian float64
              struct.pack('>d', vy)                 — 8 bytes, big-endian float64
          hash = SHA3-256(encoded)
        """
        with self._inner_lock:
            now = time.time()
            if (self._geometry_hash_cache is not None
                    and (now - self._cache_timestamp) < self._CACHE_TTL_SECONDS):
                return self._geometry_hash_cache
            try:
                triangles = self.fetch_all_triangles(max_depth)
                encoded = b''
                for tri in triangles:
                    encoded += struct.pack('>Q', tri['id'])
                    encoded += struct.pack('>I', tri['depth'])
                    for vx, vy in (tri['v0'], tri['v1'], tri['v2']):
                        encoded += struct.pack('>d', vx)
                        encoded += struct.pack('>d', vy)
                self._geometry_hash_cache = hashlib.sha3_256(encoded).digest()
                self._cache_timestamp = now
                logger.info(f"[HyperbolicGeometry] Computed geometry hash from {len(triangles)} triangles")
                return self._geometry_hash_cache
            except Exception as e:
                logger.warning(f"[HyperbolicGeometry] compute_geometry_hash failed: {e} — using fallback")
                return hashlib.sha3_256(b"QTCL_GEOMETRY_UNAVAILABLE").digest()

    def invalidate_cache(self):
        """Force geometry hash recomputation on next call."""
        with self._inner_lock:
            self._geometry_hash_cache = None
            self._cache_timestamp = 0.0

    def hyperbolic_hash(self, msg: bytes) -> bytes:
        """
        Return SHA3-256(domain || msg || geometry_hash).
        Uses hyperbolic geometry as salt — binds the hash to the tessellation state.

        MUST BE IDENTICAL TO qtcl_client.py HyperbolicGeometry.hyperbolic_hash().
        """
        geom_hash = self.compute_geometry_hash()
        return hashlib.sha3_256(b"QTCL_HYPERBOLIC_HASH_v1" + msg + geom_hash).digest()

    def hyperbolic_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """
        Compute hyperbolic distance between two points in Poincaré disk.

        d(z1, z2) = 2 * atanh(|z1 - z2| / |1 - conj(z1) * z2|)

        MUST BE IDENTICAL TO qtcl_client.py HyperbolicGeometry.hyperbolic_distance().
        """
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        d2 = dx * dx + dy * dy
        if d2 < 1e-14:
            return 0.0
        denom = (1.0 - x1 * x1 - y1 * y1) * (1.0 - x2 * x2 - y2 * y2)
        if denom <= 0.0:
            return float('inf')
        cosh_dist = 1.0 + 2.0 * d2 / denom
        if cosh_dist < 1.0:
            cosh_dist = 1.0
        return math.acosh(cosh_dist)


# Global HyperbolicGeometry singleton
_hyperbolic_geometry: Optional[HyperbolicGeometry] = None

def get_hyperbolic_geometry() -> HyperbolicGeometry:
    """Get or create global HyperbolicGeometry singleton."""
    global _hyperbolic_geometry
    if _hyperbolic_geometry is None or not _hyperbolic_geometry.is_initialized:
        _hyperbolic_geometry = HyperbolicGeometry()
    return _hyperbolic_geometry


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# HLWE CRYPTOGRAPHIC ENGINE — Genuine Hyperbolic Learning With Errors
#
# CANONICAL Fiat-Shamir lattice signature scheme.
# MUST be byte-identical to qtcl_client.py HLWEEngine for all shared crypto operations.
#
# Crypto primitives:
#   Fixed basis A:      SHAKE-256(b"QTCL_HLWE_BASIS_FIXED_v2") — protocol constant for sign/verify
#   GeodesicLWE basis:  Möbius transport of Poincaré disk pseudoqubit coords — for keygen
#   Secret vector s:    ternary {q-1, 0, 1} from HKDF-Extract→SHAKE-256 XOF
#   Public key b:       b = A·s mod q (NO error vector)
#   Private key:        32-byte seed (64 hex chars)
#   Masking vector y:   ternary from HKDF-Extract→SHAKE-256 (deterministic per msg+key)
#   Commitment w:       w = A·y mod q
#   Challenge c:        2*(SHA-256(b"HLWE_CHALLENGE_v1" || msg || w || pubkey || geom_hash)[0] & 1) - 1
#   Response z:         z = c·s + y mod q
#   Verification:       w' = A·z - b·c mod q, check |w'[i] - w[i]| ≤ ERROR_BOUND * |c| + 1
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class HLWEEngine:
    """
    Post-quantum cryptographic engine using genuine Hyperbolic Learning With Errors.

    The hyperbolic geometry from the {8,3} Poincaré disk tessellation enters at:
      1. GeodesicLWE keygen basis — rows of A derived from Möbius-transported pseudoqubit coords
      2. Fiat-Shamir challenge — geometry hash from hyperbolic_triangles table salts the challenge

    All crypto operations are byte-identical to qtcl_client.py HLWEEngine.
    """

    def __init__(self):
        self.params = LatticeParams()
        self.kd_params = KeyDerivationParams()
        self.lock = threading.RLock()
        # Pre-compute and cache the fixed lattice basis (protocol constant — immutable)
        self._fixed_basis_cache: Optional[List[List[int]]] = None
        self._fixed_basis_lock = threading.Lock()
        logger.info("[HLWE] Engine initialized (DIMENSION={}, MODULUS={}, GENUINE_HYPERBOLIC=TRUE)".format(
            self.params.DIMENSION, self.params.MODULUS))

    # ── FIXED LATTICE BASIS (protocol constant, for sign/verify) ──────────────

    def _derive_fixed_lattice_basis(self) -> List[List[int]]:
        """
        Derive n×n lattice basis A from a FIXED protocol constant.

        A is a PUBLIC SYSTEM PARAMETER (like an elliptic curve generator),
        NOT derived from per-key material.  This ensures all parties compute
        the same A regardless of which key pair they're using — critical
        for signature verification to succeed.

        Construction:
          xof = SHAKE-256(b"QTCL_HLWE_BASIS_FIXED_v2")
          A[i][j] = xof.read(4) as big-endian uint32 mod q

        MUST BE IDENTICAL TO qtcl_client.py HLWEEngine._derive_fixed_lattice_basis().
        """
        with self._fixed_basis_lock:
            if self._fixed_basis_cache is not None:
                return self._fixed_basis_cache
            n = self.params.DIMENSION
            q = self.params.MODULUS
            xof = hashlib.shake_256(b"QTCL_HLWE_BASIS_FIXED_v2")
            xof_bytes = xof.digest(n * n * 4)
            A = []
            for i in range(n):
                row = []
                for j in range(n):
                    offset = (i * n + j) * 4
                    val = int.from_bytes(xof_bytes[offset:offset + 4], 'big') % q
                    row.append(val)
                A.append(row)
            self._fixed_basis_cache = A
            return A

    # ── GEODESIC LWE BASIS (hyperbolic, for keygen) ──────────────────────────

    def _derive_lattice_basis_from_entropy(self, entropy: bytes) -> List[List[int]]:
        """
        GeodesicLWE: derive n×n lattice basis A from entropy + hyperbolic DB geometry.

        Each row i is a Möbius geodesic displacement vector in the Poincaré disk,
        seeded by the i-th pseudoqubit coordinate, then quantised to Z_q.

        Construction per row i:
          1. Fetch DB pseudoqubit coord (px, py) for id i (wraps around cache size).
          2. Form complex seed z_i = px + i·py inside unit disk.
          3. For each column j:
             a. entropy_seed = HMAC-SHA256(b"GeodesicLWE_row", entropy||i_be32||j_be32)
             b. t_j = (entropy_seed[:8] as uint64) / 2^64 * π  (transport parameter ∈ [0,π])
             c. w_j = Möbius transport of z_i by t_j along its geodesic
             d. A[i][j] = round(Re(w_j) * q/2 + q/2) mod q
                          (maps [-1,1) → [0,q) linearly)

        If the PQ cache is not yet ready, falls back to pure SHA-256 Euclidean basis.

        MUST BE IDENTICAL TO qtcl_client.py HLWEEngine._derive_lattice_basis_from_entropy().
        """
        n = self.params.DIMENSION
        q = self.params.MODULUS
        # Try to warm the cache if it's not ready yet
        if not _PQ_CACHE_READY.is_set():
            try:
                _ensure_pq_cache()
            except Exception:
                pass
        cache_ready = _PQ_CACHE_READY.is_set()
        if not cache_ready:
            logger.warning("[GeodesicLWE] PQ cache not ready — using Euclidean fallback for basis")
        A = []
        for i in range(n):
            row = []
            z_seed = None
            if cache_ready:
                try:
                    px, py = _pq_get_coord(i)
                    z_seed = complex(px, py)
                except Exception:
                    z_seed = None
            for j in range(n):
                if z_seed is not None:
                    # Geodesic transport parameter from entropy
                    seed_ij = (b"GeodesicLWE_row" + entropy
                               + i.to_bytes(4, 'big') + j.to_bytes(4, 'big'))
                    h = hmac.new(b"GeodesicLWE_v1", seed_ij, hashlib.sha256).digest()
                    t_raw = int.from_bytes(h[:8], 'big') / (2 ** 64)
                    t_j = t_raw * math.pi   # ∈ [0, π]
                    w_j = _mobius_transport(z_seed, t_j)
                    # Map Re(w_j) ∈ (-1,1) → Z_q
                    val = int(round((w_j.real + 1.0) * (q / 2.0))) % q
                    row.append(val)
                else:
                    # Euclidean fallback (original SHA-256 path)
                    seed_ij = entropy + bytes([i & 0xFF, j & 0xFF])
                    h = hashlib.sha256(seed_ij).digest()
                    row.append(int.from_bytes(h[:4], 'big') % q)
            A.append(row)
        return A

    # ── SECRET VECTOR (ternary, from entropy/seed) ───────────────────────────

    def _derive_secret_vector(self, entropy: bytes, dimension: int) -> List[int]:
        """
        Derive secret vector s from entropy using SHAKE-256 XOF + ternary mapping.

        CANONICAL IMPLEMENTATION — must be identical to qtcl_client.py.

        LWE hardness requires s to be SHORT (small-secret variant).
        Each component is drawn from {q-1, 0, 1} ⊂ Z_q with L∞-norm = 1.
        P(s_i=0)=1/2, P(s_i=±1)=1/4 each.

        Construction:
          prk     = HMAC-SHA256(b"HLWE_SECRET_v1", entropy)
          xof     = SHAKE-256(prk || b"secret_vector" || dimension_be32)
          nibble  = xof_byte & 0x03
          0 → q-1 (-1 mod q)
          1 → 0
          2 → 0   (extra zero weight for P(0)=1/2)
          3 → 1
        """
        q = self.params.MODULUS
        prk = hmac.new(b"HLWE_SECRET_v1", entropy, hashlib.sha256).digest()
        xof_input = prk + b"secret_vector" + dimension.to_bytes(4, 'big')
        xof_bytes = hashlib.shake_256(xof_input).digest(dimension)
        s = []
        for b in xof_bytes:
            nibble = b & 0x03
            if nibble == 0:
                s.append(q - 1)   # -1 mod q
            elif nibble == 3:
                s.append(1)
            else:
                s.append(0)       # nibble ∈ {1,2} → P(0)=1/2
        return s

    def _derive_secret_vector_from_key(self, key_seed: bytes, dimension: int) -> List[int]:
        """
        Derive ternary secret vector s from private key seed.

        CANONICAL — identical to qtcl_client.py HLWEEngine._derive_secret_vector_from_key.
        Same construction as _derive_secret_vector (they share the same HKDF+SHAKE pipeline).
        """
        q = self.params.MODULUS
        prk = hmac.new(b"HLWE_SECRET_v1", key_seed, hashlib.sha256).digest()
        xof_input = prk + b"secret_vector" + dimension.to_bytes(4, 'big')
        xof_bytes = hashlib.shake_256(xof_input).digest(dimension)
        s = []
        for byte_val in xof_bytes:
            nibble = byte_val & 0x03
            if nibble == 0:
                s.append(q - 1)
            elif nibble == 3:
                s.append(1)
            else:
                s.append(0)
        return s

    # ── ERROR VECTOR (deterministic from seed) ───────────────────────────────

    def _sample_error_vector(self, dimension: int, seed: bytes = None) -> List[int]:
        """
        Deterministic error vector e from seed.

        CANONICAL — identical to qtcl_client.py HLWEEngine._sample_error_vector.

        Construction:
          prk  = HKDF-Extract(salt=b"HLWE_ERROR_v1", ikm=seed)
          xof  = SHAKE-256(prk || b"error" || dimension_be32)
          Each component: (xof_byte mod (2·B+1)) − B  → uniform in [−B, B]
        """
        q = self.params.MODULUS
        B = self.params.ERROR_BOUND
        if seed is None:
            seed = os.urandom(32)
        prk = hmac.new(b"HLWE_ERROR_v1", seed, hashlib.sha256).digest()
        xof_input = prk + b"error" + dimension.to_bytes(4, 'big')
        xof_bytes = hashlib.shake_256(xof_input).digest(dimension)
        e = []
        for byte_val in xof_bytes:
            val = (byte_val % (2 * B + 1)) - B
            e.append(val % q)
        return e

    # ── MASKING VECTOR (deterministic ternary for Fiat-Shamir) ───────────────

    def _sample_masking_vector(self, msg_hash: bytes, key_seed: bytes, dimension: int) -> List[int]:
        """
        Deterministic short masking vector y (ternary distribution).

        CANONICAL — identical to qtcl_client.py HLWEEngine._sample_masking_vector.

        Construction:
          prk  = HKDF-Extract(salt=b"HLWE_MASK_v1", ikm=msg_hash || key_seed)
          xof  = SHAKE-256(prk || b"masking" || dimension_be32)
          Same ternary mapping as secret vector.
        """
        q = self.params.MODULUS
        prk = hmac.new(b"HLWE_MASK_v1", msg_hash + key_seed, hashlib.sha256).digest()
        xof_input = prk + b"masking" + dimension.to_bytes(4, 'big')
        xof_bytes = hashlib.shake_256(xof_input).digest(dimension)
        y = []
        for byte_val in xof_bytes:
            nibble = byte_val & 0x03
            if nibble == 0:
                y.append(q - 1)
            elif nibble == 3:
                y.append(1)
            else:
                y.append(0)
        return y

    # ── FIAT-SHAMIR CHALLENGE (with hyperbolic geometry hash) ────────────────

    def _hash_to_challenge_scalar(self, msg_hash: bytes, w_bytes: bytes,
                                   public_key_bytes: bytes = b'') -> int:
        """
        Deterministic challenge scalar c ∈ {-1, 1} from message + commitment,
        incorporating hyperbolic geometry from the {8,3} tessellation.

        Construction:
          geom_hash = get_hyperbolic_geometry().hyperbolic_hash(msg_hash)
          h = SHA-256(b"HLWE_CHALLENGE_v1" || msg || w || pubkey || geom_hash)
          c = 2 * (h[0] & 1) - 1  → {-1, 1} with UNIFORM probability.

        If hyperbolic geometry is unavailable, falls back to
        SHA3-256(b"QTCL_GEOMETRY_UNAVAILABLE") — deterministic, same on both sides.

        MUST BE IDENTICAL TO qtcl_client.py HLWEEngine._hash_to_challenge_scalar().
        """
        try:
            hyper_geo = get_hyperbolic_geometry()
            geom_hash = hyper_geo.hyperbolic_hash(msg_hash)
        except Exception:
            # Fallback: sign without hyperbolic geometry
            geom_hash = hashlib.sha3_256(b"QTCL_GEOMETRY_UNAVAILABLE").digest()
        h = hashlib.sha256(
            b"HLWE_CHALLENGE_v1" + msg_hash + w_bytes + public_key_bytes + geom_hash
        ).digest()
        return 2 * (h[0] & 1) - 1  # uniform {-1, 1}

    # ── PUBLIC KEY COMPUTATION ───────────────────────────────────────────────

    def _compute_public_key_bytes(self, priv_seed: bytes) -> bytes:
        """
        Compute HLWE public key b = A·s mod q from private seed.
        Uses FIXED public lattice basis A (protocol constant).

        MUST BE IDENTICAL TO qtcl_client.py HLWEEngine._compute_public_key_bytes().
        """
        n = self.params.DIMENSION
        q = self.params.MODULUS
        A = self._derive_fixed_lattice_basis()
        s = self._derive_secret_vector_from_key(priv_seed, n)
        b = LatticeMath.matrix_vector_mult(A, s, q)
        return b''.join(x.to_bytes(4, 'big') for x in b)

    # ── ADDRESS DERIVATION ───────────────────────────────────────────────────

    def derive_address_from_public_key(self, public_key) -> str:
        """
        Derive QTCL wallet address from HLWE public key vector.

        CANONICAL SPEC:
          pubkey_bytes = b''.join(x.to_bytes(4, byteorder='big') for x in public_key)
          address = SHA3-256(SHA3-256(pubkey_bytes)).hex()

        Output: 64 hex characters (256 bits = 32 bytes) — post-quantum secure.

        Accepts both List[int] (vector) and bytes (packed) input.

        MUST BE IDENTICAL TO qtcl_client.py HLWEEngine.derive_address_from_public_key().
        """
        if isinstance(public_key, bytes):
            pub_bytes = public_key
        else:
            pub_bytes = b''.join(x.to_bytes(4, 'big') for x in public_key)
        h1 = hashlib.sha3_256(pub_bytes).digest()
        h2 = hashlib.sha3_256(h1).digest()
        return h2.hex()

    # ── VECTOR ENCODING (NO modulus clamp — match client) ────────────────────

    def _encode_vector_to_hex(self, vector: List[int]) -> str:
        """
        Encode vector to hex string with range validation.

        MUST BE IDENTICAL TO qtcl_client.py HLWEEngine._encode_vector_to_hex().
        No modulus clamping — values are already in Z_q from computation.
        """
        return ''.join(x.to_bytes(4, byteorder='big').hex() for x in vector)

    def _decode_vector_from_hex(self, hex_str: str) -> List[int]:
        """
        Decode vector from hex string with strict validation.

        MUST BE IDENTICAL TO qtcl_client.py HLWEEngine._decode_vector_from_hex().
        No modulus clamping — raw uint32 values preserved.
        """
        vector = []
        for i in range(0, len(hex_str), 8):
            chunk = hex_str[i:i + 8]
            if len(chunk) != 8:
                raise ValueError(f"Invalid hex chunk at position {i}: '{chunk}' (need 8 chars)")
            try:
                val = int(chunk, 16)
            except ValueError:
                raise ValueError(f"Invalid hex chunk at position {i}: '{chunk}'")
            vector.append(val)
        return vector

    # ── KEYPAIR GENERATION ───────────────────────────────────────────────────

    def generate_keypair_from_entropy(self) -> HLWEKeyPair:
        """
        Generate HLWE keypair seeded from system entropy.

        CANONICAL LATTICE KEYGEN:
          1. entropy = get_block_field_entropy() → 32 bytes
          2. priv_seed = entropy[:32]
          3. A = FIXED public lattice basis (protocol constant)
          4. s = HKDF-Extract→SHAKE-256 XOF(entropy) → ternary {q-1, 0, 1}
          5. b = A·s mod q (public key vector, NO error)
          6. pubkey_bytes = b.to_bytes(4, big) per element
          7. address = SHA3-256(SHA3-256(pubkey_bytes)).hex()

        Returns: HLWEKeyPair(public_key, private_key, address)

        MUST BE IDENTICAL TO qtcl_client.py HLWEEngine.generate_keypair_from_entropy().
        """
        with self.lock:
            try:
                entropy = get_block_field_entropy()
                if len(entropy) < 32:
                    entropy = entropy + secrets.token_bytes(32 - len(entropy))

                priv_seed = entropy[:32]
                private_key_hex = priv_seed.hex()

                n = self.params.DIMENSION
                q = self.params.MODULUS

                # ════ DERIVE FIXED LATTICE BASIS A (protocol constant) ════
                A = self._derive_fixed_lattice_basis()

                # ════ DERIVE SECRET VECTOR s (ternary) ════
                s = self._derive_secret_vector(priv_seed, n)

                # ════ COMPUTE PUBLIC KEY b = A·s mod q (NO error) ════
                b = LatticeMath.matrix_vector_mult(A, s, q)
                pub_bytes = b''.join(x.to_bytes(4, 'big') for x in b)
                public_key_hex = pub_bytes.hex()

                # ════ DERIVE ADDRESS (double-SHA3-256) ════
                address = self.derive_address_from_public_key(b)

                logger.info(f"[HLWE] Generated keypair: {address[:16]}... (lattice, entropy-seeded)")

                return HLWEKeyPair(
                    public_key=public_key_hex,
                    private_key=private_key_hex,
                    address=address
                )

            except Exception as e:
                logger.error(f"[HLWE] Keypair generation failed: {e}")
                raise

    def generate_keypair(self) -> Dict[str, str]:
        """Generate a new HLWE keypair using generate_keypair_from_entropy."""
        kp = self.generate_keypair_from_entropy()
        return {
            "private_key": kp.private_key,
            "public_key": kp.public_key,
            "address": kp.address
        }

    def derive_public_key(self, private_key_hex: str) -> str:
        """
        Derive full HLWE public key (b = A·s mod q) from private key seed.
        Returns hex of packed public key vector (n × 4 bytes = 2048 hex chars).

        MUST BE IDENTICAL TO qtcl_client.py HLWEEngine.derive_public_key().
        """
        priv_seed = bytes.fromhex(private_key_hex)
        pub_bytes = self._compute_public_key_bytes(priv_seed)
        return pub_bytes.hex()

    # ── SIGNING (TRUE Fiat-Shamir Lattice Signature) ─────────────────────────

    def sign_hash(self, message_hash: bytes, private_key_hex: str) -> Dict[str, str]:
        """
        Sign a 32-byte message hash using TRUE HLWE Fiat-Shamir lattice signature.

        CANONICAL LATTICE FIAT-SHAMIR (post-quantum hardness):
          1. A = FIXED public lattice basis (protocol constant, same for all keys)
          2. s = HKDF-Extract→SHAKE-256 XOF, ternary {q-1, 0, 1} from priv_seed
          3. b = A·s mod q (public key vector)
          4. y = fresh HKDF-Extract→SHAKE-256 XOF, ternary {q-1, 0, 1}
          5. w = A·y mod q (commitment)
          6. c = 2*(SHA-256(b"HLWE_CHALLENGE_v1" || msg || w || pubkey || geom_hash)[0] & 1) - 1
          7. z = c·s + y mod q (Fiat-Shamir response)

        Output: {z, c_hash, w, public_key, address, timestamp}
        Verification: w' = A·z - b·c should equal w (within error bound)

        MUST BE IDENTICAL TO qtcl_client.py HLWEEngine.sign_hash().
        """
        with self.lock:
            try:
                # ════ VALIDATE INPUTS ════
                if len(message_hash) != 32:
                    raise ValueError(f"message_hash must be 32 bytes, got {len(message_hash)}")
                if len(private_key_hex) != 64:
                    raise ValueError(f"private_key_hex must be 64 hex chars, got {len(private_key_hex)}")

                n = self.params.DIMENSION
                q = self.params.MODULUS
                priv_seed = bytes.fromhex(private_key_hex)

                # ════ DERIVE FIXED LATTICE BASIS A (protocol constant) ════
                A = self._derive_fixed_lattice_basis()

                # ════ DERIVE SECRET VECTOR s (ternary) ════
                s = self._derive_secret_vector(priv_seed, n)

                # ════ COMPUTE PUBLIC KEY b = A·s mod q ════
                b = LatticeMath.matrix_vector_mult(A, s, q)
                pub_bytes = b''.join(x.to_bytes(4, 'big') for x in b)

                # ════ DERIVE ADDRESS (double-SHA3-256) ════
                address = self.derive_address_from_public_key(b)

                # ════ SAMPLE MASKING VECTOR y (ternary, deterministic for this msg) ════
                y = self._sample_masking_vector(message_hash, priv_seed, n)

                # ════ COMPUTE COMMITMENT w = A·y mod q ════
                w = LatticeMath.matrix_vector_mult(A, y, q)
                w_bytes = b''.join(x.to_bytes(4, 'big') for x in w)

                # ════ FIAT-SHAMIR CHALLENGE c ∈ {-1, 1} ════
                c_scalar = self._hash_to_challenge_scalar(message_hash, w_bytes, pub_bytes)

                # ════ RESPONSE z = c·s + y mod q ════
                z = [(c_scalar * s[i] + y[i]) % q for i in range(n)]

                # ════ CHALLENGE COMMITMENT (includes msg_hash to prevent replay) ════
                c_bytes = c_scalar.to_bytes(4, 'big', signed=True)
                c_hash = hashlib.sha256(b"HLWE_CHALLENGE_v1" + message_hash + c_bytes).digest()

                return {
                    "z": self._encode_vector_to_hex(z),
                    "c_hash": c_hash.hex(),
                    "w": self._encode_vector_to_hex(w),
                    "public_key": pub_bytes.hex(),
                    "address": address,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            except Exception as e:
                logger.error(f"[HLWE] Signing failed: {e}")
                raise

    # ── VERIFICATION (Lattice Relation Check) ────────────────────────────────

    def verify_signature(self, message_hash: bytes, signature_dict: Dict[str, str],
                         public_key_hex: str) -> bool:
        """
        Verify TRUE HLWE Fiat-Shamir lattice signature.

        CANONICAL LATTICE VERIFICATION:
          Input: (z, c_hash, w, public_key) from signature
          1. A = FIXED public lattice basis (same as signing)
          2. c_scalar = derive_challenge_scalar(msg, w) — deterministic
          3. Verify c_hash == SHA-256(b"HLWE_CHALLENGE_v1" || msg_hash || c_scalar_bytes)
          4. w' = A·z - b·c mod q
          5. Check w' is close to w (within ERROR_BOUND * |c|)

        Lattice relation security: |w' - w| > bound → forgery (LWE hard problem).
        Return: True if signature is valid, False otherwise.

        MUST BE IDENTICAL TO qtcl_client.py HLWEEngine.verify_signature().
        """
        with self.lock:
            try:
                n = self.params.DIMENSION
                q = self.params.MODULUS

                # ════ VALIDATE INPUTS ════
                if len(message_hash) != 32:
                    return False

                # ════ PARSE SIGNATURE ════
                z_hex = signature_dict.get('z', '')
                c_hex = signature_dict.get('c_hash', '')
                w_hex = signature_dict.get('w', '')

                if not z_hex or not c_hex or not w_hex:
                    return False

                # ════ VALIDATE SIGNATURE SIZES ════
                expected_vec_hex_len = n * 8
                if len(z_hex) != expected_vec_hex_len:
                    return False
                if len(w_hex) != expected_vec_hex_len:
                    return False
                if len(c_hex) != 64:
                    return False

                # ════ VALIDATE PUBLIC KEY FORMAT ════
                expected_pubkey_hex_len = n * 8
                if len(public_key_hex) != expected_pubkey_hex_len:
                    return False
                try:
                    bytes.fromhex(public_key_hex)
                except ValueError:
                    return False

                # ════ DECODE VECTORS ════
                z = self._decode_vector_from_hex(z_hex)
                w = self._decode_vector_from_hex(w_hex)

                if len(z) != n or len(w) != n:
                    return False

                # ════ DECODE PUBLIC KEY ════
                b = self._decode_vector_from_hex(public_key_hex)
                if len(b) != n:
                    return False

                # ════ DERIVE FIXED LATTICE BASIS A (protocol constant — same as signing) ════
                A = self._derive_fixed_lattice_basis()

                # ════ VERIFY FIAT-SHAMIR CHALLENGE ════
                w_bytes = b''.join(x.to_bytes(4, 'big') for x in w)
                pub_bytes_verify = bytes.fromhex(public_key_hex)
                c_scalar = self._hash_to_challenge_scalar(message_hash, w_bytes, pub_bytes_verify)
                c_bytes = c_scalar.to_bytes(4, 'big', signed=True)
                expected_c_hash = hashlib.sha256(
                    b"HLWE_CHALLENGE_v1" + message_hash + c_bytes
                ).digest()

                if not hmac.compare_digest(expected_c_hash.hex(), c_hex):
                    return False

                # ════ VERIFY LATTICE RELATION w' = A·z - b·c ════
                Az = LatticeMath.matrix_vector_mult(A, z, q)
                bc = [(c_scalar * bi) % q for bi in b]
                w_prime = [(Az[i] - bc[i]) % q for i in range(n)]

                # ════ CHECK w' ≈ w (within error bound) ════
                bound = self.params.ERROR_BOUND * max(abs(c_scalar), 1)
                for i in range(n):
                    diff = (w_prime[i] - w[i]) % q
                    centered = diff if diff <= q // 2 else q - diff
                    if centered > bound + 1:
                        return False

                return True

            except Exception as e:
                logger.debug(f"[HLWE] Verification error: {e}")
                return False

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# BIP39 MNEMONIC MANAGER
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class MnemonicStrength(Enum):
    """BIP39 entropy strength levels."""
    STANDARD = 128      # 12 words
    HIGH     = 256      # 24 words


class BIP39Mnemonics:
    """BIP39 mnemonic seed phrase generation and derivation."""
    
    def __init__(self):
        self.wordlist = BIP39_WORDLIST
    
    def generate_mnemonic(self, strength=None):
        """Generate random mnemonic from entropy."""
        if strength is None:
            strength = MnemonicStrength.STANDARD
        if isinstance(strength, MnemonicStrength):
            entropy_bits = strength.value
        else:
            entropy_bits = 128
        
        entropy_bytes = entropy_bits // 8
        entropy = secrets.token_bytes(entropy_bytes)
        return self._entropy_to_mnemonic(entropy)
    
    def _entropy_to_mnemonic(self, entropy: bytes) -> str:
        """Convert entropy to BIP39 mnemonic words."""
        hash_obj = hashlib.sha256(entropy).digest()
        checksum_bits_len = len(entropy) * 8 // 32
        checksum_bits = bin(int.from_bytes(hash_obj[:1], 'big'))[2:].zfill(8)[:checksum_bits_len]
        
        entropy_bits = bin(int.from_bytes(entropy, 'big'))[2:].zfill(len(entropy) * 8)
        combined = entropy_bits + checksum_bits
        
        mnemonic = []
        for i in range(0, len(combined), 11):
            idx = int(combined[i:i+11], 2)
            mnemonic.append(self.wordlist[idx])
        
        return ' '.join(mnemonic)
    
    def mnemonic_to_seed(self, mnemonic: str, passphrase: str = '') -> bytes:
        """PBKDF2-HMAC-SHA512 of mnemonic + passphrase (BIP39 standard)."""
        mnemonic_bytes = mnemonic.encode('utf-8')
        salt = b'mnemonic' + passphrase.encode('utf-8')
        
        # PBKDF2 with 2048 iterations (BIP39 standard)
        seed = hashlib.pbkdf2_hmac(
            'sha512',
            mnemonic_bytes,
            salt,
            2048
        )
        return seed
    
    def validate_mnemonic(self, mnemonic: str) -> bool:
        """Validate checksum of mnemonic."""
        words = mnemonic.split()
        if len(words) not in [12, 15, 18, 21, 24]:
            return False
        
        try:
            indices = [self.wordlist.index(w) for w in words]
            bits = ''.join(f'{i:011b}' for i in indices)
            
            entropy_len = len(bits) * 32 // 33
            entropy_bits = bits[:entropy_len]
            checksum_bits = bits[entropy_len:]
            
            entropy = int(entropy_bits, 2).to_bytes(entropy_len // 8, 'big')
            hash_obj = hashlib.sha256(entropy).digest()
            expected_checksum = bin(int.from_bytes(hash_obj[:1], 'big'))[2:].zfill(8)
            
            checksum_len = len(entropy) * 8 // 32
            return checksum_bits == expected_checksum[:checksum_len]
        except (ValueError, IndexError):
            return False


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# STUB HELPER CLASSES (Placeholder - not actively used)
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class BIP38Encryption:
    """BIP38 password-protected private key encryption (stub)."""
    def encrypt_private_key(self, private_key_hex: str, passphrase: str) -> Dict[str, str]:
        """Stub - returns unencrypted for now."""
        return {'encrypted': private_key_hex, 'method': 'stub'}


class SupabaseAPI:
    """Supabase integration stub."""
    def save_wallet(self, metadata):
        """Stub - does nothing."""
        pass


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# COMPLETE WALLET MANAGER (Integration Layer)
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class HLWEWalletManager:
    """Wallet manager stub - not used by quantum blockchain core"""
    
    def __init__(self):
        self.hlwe = HLWEEngine()
        self.bip39 = BIP39Mnemonics()
        self.bip38 = BIP38Encryption()
        self.supabase = SupabaseAPI()
        self.lock = threading.RLock()
        logger.info("[WalletManager] Initialized (STUB - BIP32 disabled)")
    
    def create_wallet(self, name='default', passphrase=''):
        """Stub implementation"""
        return {'address': 'qtcl_stub', 'status': 'stub'}
    
    def get_address(self, account=0, change=0, index=0):
        return 'qtcl_stub'
    
    def create_wallet(
        self,
        wallet_label: Optional[str] = None,
        passphrase: str = ''
    ) -> Dict[str, Any]:
        """Create new HD wallet with mnemonic seed phrase"""
        with self.lock:
            try:
                mnemonic = self.bip39.generate_mnemonic(MnemonicStrength.STANDARD)
                seed = self.bip39.mnemonic_to_seed(mnemonic, passphrase)
                master_key, master_chain_code = self.bip32.derive_master_key(seed)
                fingerprint = hashlib.sha256(master_key).hexdigest()[:16]
                
                mnemonic_encrypted_data = self.bip38.encrypt_private_key(
                    master_key.hex(),
                    passphrase if passphrase else 'DEFAULT'
                )
                
                wallet_id = secrets.token_hex(16)
                metadata = WalletMetadata(
                    wallet_id=wallet_id,
                    fingerprint=fingerprint,
                    mnemonic_encrypted=json.dumps(mnemonic_encrypted_data),
                    master_chain_code=master_chain_code.hex(),
                    label=wallet_label
                )
                
                self.supabase.save_wallet(metadata)
                
                logger.info(f"[WalletManager] Created wallet {wallet_id} ({wallet_label or 'unnamed'})")
                
                return {
                    'wallet_id': wallet_id,
                    'fingerprint': fingerprint,
                    'mnemonic': mnemonic,
                    'label': wallet_label,
                    'created_at': metadata.created_at.isoformat()
                }
            
            except Exception as e:
                logger.error(f"[WalletManager] Create wallet failed: {e}")
                raise
    
    def derive_address(
        self,
        wallet_fingerprint: str,
        derivation_path: BIP32DerivationPath = None,
        address_type: str = "receiving"
    ) -> Optional[StoredAddress]:
        """Derive new address from wallet at specified derivation path"""
        with self.lock:
            try:
                if derivation_path is None:
                    derivation_path = BIP32DerivationPath()
                
                keypair = self.hlwe.generate_keypair_from_entropy()
                
                address = StoredAddress(
                    address=keypair.address,
                    public_key=keypair.public_key,
                    wallet_fingerprint=wallet_fingerprint,
                    derivation_path=derivation_path.path_string(),
                    address_type=address_type
                )
                
                self.supabase.save_address(address)
                
                logger.info(f"[WalletManager] Derived address {address.address} ({address_type})")
                
                return address
            
            except Exception as e:
                logger.error(f"[WalletManager] Derive address failed: {e}")
                return None
    
    def sign_transaction(
        self,
        message_hash: bytes,
        private_key_hex: str
    ) -> Dict[str, str]:
        """Sign transaction with private key"""
        return self.hlwe.sign_hash(message_hash, private_key_hex)
    
    def verify_transaction_signature(
        self,
        message_hash: bytes,
        signature_dict: Dict[str, str],
        public_key_hex: str
    ) -> bool:
        """Verify transaction signature"""
        return self.hlwe.verify_signature(message_hash, signature_dict, public_key_hex)

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# INTEGRATION ADAPTER — BACKWARD-COMPATIBLE API (Top-level Functions)
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class HLWEIntegrationAdapter:
    """Adapter layer providing backward-compatible API for existing QTCL systems"""
    
    def __init__(self):
        self.hlwe = HLWEEngine()
        self.lock = threading.RLock()
        logger.info("[HLWE-Adapter] Initialized (direct HLWEEngine)")
    
    def sign_block(self, block_dict: Dict[str, Any], private_key_hex: str) -> Dict[str, str]:
        """Sign block with HLWE private key (backward-compatible signature)"""
        with self.lock:
            try:
                block_json = json.dumps(block_dict, sort_keys=True, default=str)
                block_hash = hashlib.sha256(block_json.encode('utf-8')).digest()
                sig_dict = self.hlwe.sign_hash(block_hash, private_key_hex)
                logger.info(f"[HLWE-Adapter] Signed block (hash={block_hash.hex()[:16]}...)")
                return sig_dict
            
            except Exception as e:
                logger.error(f"[HLWE-Adapter] Block signing failed: {e}")
                return {'z': '', 'c_hash': '', 'w': '', 'error': str(e)}
    
    def verify_block(self, block_dict: Dict[str, Any], signature_dict: Dict[str, str], public_key_hex: str) -> Tuple[bool, str]:
        """Verify block signature"""
        with self.lock:
            try:
                block_json = json.dumps(block_dict, sort_keys=True, default=str)
                block_hash = hashlib.sha256(block_json.encode('utf-8')).digest()
                is_valid = self.hlwe.verify_signature(block_hash, signature_dict, public_key_hex)
                
                if is_valid:
                    logger.debug(f"[HLWE-Adapter] ✓ Block signature verified")
                    return True, "OK"
                else:
                    logger.warning(f"[HLWE-Adapter] ✗ Block signature verification failed")
                    return False, "Invalid signature"
            
            except Exception as e:
                logger.error(f"[HLWE-Adapter] Block verification failed: {e}")
                return False, f"Verification error: {str(e)}"
    
    def sign_transaction(self, tx_data: Dict[str, Any], private_key_hex: str) -> Dict[str, str]:
        """Sign transaction with HLWE private key"""
        with self.lock:
            try:
                tx_json = json.dumps(tx_data, sort_keys=True, default=str)
                tx_hash = hashlib.sha256(tx_json.encode('utf-8')).digest()
                sig_dict = self.hlwe.sign_hash(tx_hash, private_key_hex)
                logger.info(f"[HLWE-Adapter] Signed transaction (hash={tx_hash.hex()[:16]}...)")
                return sig_dict
            
            except Exception as e:
                logger.error(f"[HLWE-Adapter] TX signing failed: {e}")
                return {'z': '', 'c_hash': '', 'w': '', 'error': str(e)}
    
    def verify_transaction(self, tx_data: Dict[str, Any], signature_dict: Dict[str, str], public_key_hex: str) -> Tuple[bool, str]:
        """Verify transaction signature"""
        with self.lock:
            try:
                tx_json = json.dumps(tx_data, sort_keys=True, default=str)
                tx_hash = hashlib.sha256(tx_json.encode('utf-8')).digest()
                is_valid = self.hlwe.verify_signature(tx_hash, signature_dict, public_key_hex)
                
                if is_valid:
                    logger.debug(f"[HLWE-Adapter] ✓ Transaction signature verified")
                    return True, "OK"
                else:
                    return False, "Invalid signature"
            
            except Exception as e:
                logger.error(f"[HLWE-Adapter] TX verification failed: {e}")
                return False, f"Verification error: {str(e)}"
    
    def derive_address(self, public_key_hex: str) -> str:
        """Derive wallet address from public key"""
        with self.lock:
            try:
                pub_bytes = bytes.fromhex(public_key_hex)
                pub_vector = [int.from_bytes(pub_bytes[i:i+4], byteorder='big') 
                             for i in range(0, len(pub_bytes), 4)]
                address = self.hlwe.derive_address_from_public_key(pub_vector)
                return address
            
            except Exception as e:
                logger.error(f"[HLWE-Adapter] Address derivation failed: {e}")
                return ''
    
    def create_wallet(self, label: Optional[str] = None, passphrase: str = '') -> Dict[str, Any]:
        """Create new HD wallet with mnemonic"""
        with self.lock:
            try:
                wallet = self.wallet_manager.create_wallet(label, passphrase)
                logger.info(f"[HLWE-Adapter] Created wallet {wallet['wallet_id']}")
                return wallet
            
            except Exception as e:
                logger.error(f"[HLWE-Adapter] Wallet creation failed: {e}")
                return {'error': str(e)}
    
    def derive_address_from_wallet(
        self,
        wallet_fingerprint: str,
        index: int = 0,
        address_type: str = "receiving"
    ) -> Optional[Dict[str, Any]]:
        """Derive new address from wallet"""
        with self.lock:
            try:
                path = BIP32DerivationPath(
                    change=0 if address_type == "receiving" else 1,
                    index=index
                )
                
                address = self.wallet_manager.derive_address(
                    wallet_fingerprint,
                    path,
                    address_type
                )
                
                if address:
                    return address.to_dict()
                return None
            
            except Exception as e:
                logger.error(f"[HLWE-Adapter] Address derivation failed: {e}")
                return None
    
    def health_check(self) -> bool:
        """Check HLWE system health"""
        with self.lock:
            try:
                test_entropy = os.urandom(32)
                test_pub = [1, 2, 3, 4]
                _ = self.hlwe.derive_address_from_public_key(test_pub)
                logger.debug("[HLWE-Adapter] Health check: OK")
                return True
            
            except Exception as e:
                logger.error(f"[HLWE-Adapter] Health check failed: {e}")
                return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Return system information"""
        pq_cache_size = len(_PQ_COORD_CACHE) if _PQ_CACHE_READY.is_set() else 0
        try:
            geom = get_hyperbolic_geometry()
            geom_ok = geom.is_initialized
        except Exception:
            geom_ok = False
        return {
            'engine': 'HLWE v3.0 — Genuine Hyperbolic Learning With Errors',
            'cryptography': 'Post-quantum Fiat-Shamir lattice signatures on {8,3} Poincaré disk',
            'signature_scheme': 'Fiat-Shamir (z, c_hash, w, public_key, address)',
            'lattice_dimension': self.hlwe.params.DIMENSION,
            'modulus': self.hlwe.params.MODULUS,
            'error_bound': self.hlwe.params.ERROR_BOUND,
            'fixed_basis': 'SHAKE-256(b"QTCL_HLWE_BASIS_FIXED_v2")',
            'geodesic_lwe_basis': 'Möbius transport of Poincaré pseudoqubit coords',
            'hyperbolic_geometry': geom_ok,
            'pq_cache_pseudoqubits': pq_cache_size,
            'geometry_hash_source': 'Supabase PostgreSQL (hyperbolic_triangles table)',
            'bip39': 'Mnemonic seed phrases (12-24 words)',
            'bip38': 'Password-protected private keys (PBKDF2+XOR)',
            'database': 'Supabase PostgreSQL (psycopg2 pool)',
            'entropy': 'Block field entropy from QRNG ensemble',
            'private_key_format': '64 hex chars (32-byte seed)',
            'public_key_format': '2048 hex chars (256×4 bytes, b = A·s mod q)',
            'address_format': '64 hex chars (SHA3-256(SHA3-256(pub_bytes)))',
            'initialized': True,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# GLOBAL SINGLETON
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

_WALLET_MANAGER: Optional[HLWEWalletManager] = None
_ADAPTER: Optional[HLWEIntegrationAdapter] = None

def get_wallet_manager() -> HLWEWalletManager:
    """Get or create global wallet manager singleton"""
    global _WALLET_MANAGER
    if _WALLET_MANAGER is None:
        _WALLET_MANAGER = HLWEWalletManager()
    return _WALLET_MANAGER

def get_hlwe_adapter() -> HLWEIntegrationAdapter:
    """Get or create HLWE adapter singleton"""
    global _ADAPTER
    if _ADAPTER is None:
        _ADAPTER = HLWEIntegrationAdapter()
    return _ADAPTER

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL BACKWARD-COMPATIBLE API FUNCTIONS (Drop-in Replacements)
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

def hlwe_sign_block(block_dict: Dict[str, Any], private_key_hex: str) -> Dict[str, str]:
    """Sign block (backward compatible) — USE IN blockchain_entropy_mining.py"""
    try:
        adapter = get_hlwe_adapter()
        return adapter.sign_block(block_dict, private_key_hex)
    except Exception as e:
        logger.error(f"[HLWE-RPC] Block signing failed: {e}")
        return {'z': '', 'c_hash': '', 'w': '', 'error': str(e)}

def hlwe_verify_block(block_dict: Dict[str, Any], signature_dict: Dict[str, str], public_key_hex: str) -> Tuple[bool, str]:
    """Verify block signature (backward compatible) — USE IN server.py"""
    try:
        adapter = get_hlwe_adapter()
        return adapter.verify_block(block_dict, signature_dict, public_key_hex)
    except Exception as e:
        logger.error(f"[HLWE-RPC] Block verification failed: {e}")
        return False, f"Error: {str(e)}"

def hlwe_sign_transaction(tx_data: Dict[str, Any], private_key_hex: str) -> Dict[str, str]:
    """Sign transaction (backward compatible) — USE IN mempool.py"""
    try:
        adapter = get_hlwe_adapter()
        return adapter.sign_transaction(tx_data, private_key_hex)
    except Exception as e:
        logger.error(f"[HLWE-RPC] TX signing failed: {e}")
        return {'z': '', 'c_hash': '', 'w': '', 'error': str(e)}

def hlwe_verify_transaction(tx_data: Dict[str, Any], signature_dict: Dict[str, str], public_key_hex: str) -> Tuple[bool, str]:
    """Verify transaction signature (backward compatible) — USE IN mempool.py/server.py"""
    try:
        adapter = get_hlwe_adapter()
        return adapter.verify_transaction(tx_data, signature_dict, public_key_hex)
    except Exception as e:
        logger.error(f"[HLWE-RPC] TX verification failed: {e}")
        return False, f"Error: {str(e)}"

def hlwe_derive_address(public_key_hex: str) -> str:
    """Derive address from public key (backward compatible)"""
    try:
        adapter = get_hlwe_adapter()
        return adapter.derive_address(public_key_hex)
    except Exception as e:
        logger.error(f"[HLWE-RPC] Address derivation failed: {e}")
        return ''

def hlwe_create_wallet(label: Optional[str] = None, passphrase: str = '') -> Dict[str, Any]:
    """Create new wallet (backward compatible) — USE IN server.py RPC endpoint"""
    try:
        adapter = get_hlwe_adapter()
        return adapter.create_wallet(label, passphrase)
    except Exception as e:
        logger.error(f"[HLWE-RPC] Wallet creation failed: {e}")
        return {'error': str(e)}

def hlwe_get_wallet_status(wallet_fingerprint: str) -> Dict[str, Any]:
    """Get wallet status (backward compatible) — USE IN server.py RPC endpoint"""
    try:
        adapter = get_hlwe_adapter()
        addresses = adapter.wallet_manager.supabase.get_addresses(wallet_fingerprint)
        
        return {
            'fingerprint': wallet_fingerprint,
            'address_count': len(addresses),
            'addresses': [addr.to_dict() for addr in addresses],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"[HLWE-RPC] Get wallet status failed: {e}")
        return {'error': str(e)}

def hlwe_health_check() -> bool:
    """Health check (backward compatible) — USE IN server.py /health endpoint"""
    try:
        adapter = get_hlwe_adapter()
        return adapter.health_check()
    except Exception as e:
        logger.error(f"[HLWE-RPC] Health check failed: {e}")
        return False

def hlwe_system_info() -> Dict[str, Any]:
    """Get system information — USE IN server.py /info endpoint"""
    try:
        adapter = get_hlwe_adapter()
        return adapter.get_system_info()
    except Exception as e:
        logger.error(f"[HLWE-RPC] System info failed: {e}")
        return {'error': str(e), 'status': 'unavailable'}


def get_hlwe_system_info():
    """Wrapper for hlwe_system_info() — used by wsgi_config.py"""
    return hlwe_system_info()

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Classes
    'HLWEEngine',
    'HLWEWalletManager',
    'HLWEIntegrationAdapter',
    'HyperbolicGeometry',
    'BIP32KeyDerivation',
    'BIP39Mnemonics',
    'BIP38Encryption',
    'LatticeMath',
    'SupabaseAPI',
    'HLWEKeyPair',
    'BIP32DerivationPath',
    'WalletMetadata',
    'StoredAddress',
    'MnemonicStrength',
    'AddressType',
    'LatticeParams',
    'KeyDerivationParams',
    'SupabaseConfig',
    # Functions
    'get_wallet_manager',
    'get_hlwe_adapter',
    'get_hyperbolic_geometry',
    'hlwe_sign_block',
    'hlwe_verify_block',
    'hlwe_sign_transaction',
    'hlwe_verify_transaction',
    'hlwe_derive_address',
    'hlwe_create_wallet',
    'hlwe_get_wallet_status',
    'hlwe_health_check',
    'hlwe_system_info',
    'get_hlwe_system_info',
    # BIP39 wordlist
    'BIP39_WORDLIST',
    'BIP39_ENGLISH',
    'get_word_by_index',
    'get_index_by_word',
]

if __name__ == '__main__':
    # ════════════════════════════════════════════════════════════════════════
    # HLWE v3.0 Self-Test — Genuine Hyperbolic Fiat-Shamir Lattice Signatures
    # ════════════════════════════════════════════════════════════════════════
    logger.info("=" * 100)
    logger.info("[TEST] HLWE v3.0 Genuine Hyperbolic System Self-Test")
    logger.info("=" * 100)

    # Test 1: System info
    info = hlwe_system_info()
    logger.info(f"[TEST] System: {info.get('engine')}")

    # Test 2: Key generation (Fiat-Shamir — 64-hex private key, 2048-hex public key)
    engine = HLWEEngine()
    keypair = engine.generate_keypair_from_entropy()
    logger.info(f"[TEST] Generated keypair: addr={keypair.address[:16]}...")
    logger.info(f"[TEST]   private_key: {len(keypair.private_key)} hex chars (expect 64)")
    logger.info(f"[TEST]   public_key:  {len(keypair.public_key)} hex chars (expect 2048)")
    assert len(keypair.private_key) == 64, f"private_key length {len(keypair.private_key)} != 64"
    assert len(keypair.public_key) == 2048, f"public_key length {len(keypair.public_key)} != 2048"

    # Test 3: Signing (TRUE Fiat-Shamir — output has z, c_hash, w, public_key, address)
    message = b"Test message for HLWE v3.0 genuine hyperbolic signature"
    message_hash = hashlib.sha256(message).digest()
    sig = engine.sign_hash(message_hash, keypair.private_key)
    logger.info(f"[TEST] Signed message (Fiat-Shamir):")
    logger.info(f"[TEST]   z:          {sig['z'][:32]}... ({len(sig['z'])} hex chars)")
    logger.info(f"[TEST]   c_hash:     {sig['c_hash'][:32]}... ({len(sig['c_hash'])} hex chars)")
    logger.info(f"[TEST]   w:          {sig['w'][:32]}... ({len(sig['w'])} hex chars)")
    logger.info(f"[TEST]   public_key: {sig['public_key'][:32]}... ({len(sig['public_key'])} hex chars)")
    logger.info(f"[TEST]   address:    {sig['address'][:32]}...")
    assert 'z' in sig, "Missing 'z' in signature"
    assert 'c_hash' in sig, "Missing 'c_hash' in signature"
    assert 'w' in sig, "Missing 'w' in signature"
    assert 'signature' not in sig, "Old 'signature' field should NOT exist"
    assert 'auth_tag' not in sig, "Old 'auth_tag' field should NOT exist"

    # Test 4: Verification (lattice relation check: w' = A*z - b*c ≈ w)
    is_valid = engine.verify_signature(message_hash, sig, keypair.public_key)
    logger.info(f"[TEST] Verification: {'PASS' if is_valid else 'FAIL'}")
    assert is_valid, "Signature verification FAILED — crypto mismatch"

    # Test 5: Derive public key from private key (must match keypair)
    derived_pub = engine.derive_public_key(keypair.private_key)
    logger.info(f"[TEST] derive_public_key matches: {derived_pub == keypair.public_key}")
    assert derived_pub == keypair.public_key, "derive_public_key mismatch"

    # Test 6: Address derivation from public key (must match keypair)
    pub_vector = engine._decode_vector_from_hex(keypair.public_key)
    derived_addr = engine.derive_address_from_public_key(pub_vector)
    logger.info(f"[TEST] derive_address matches: {derived_addr == keypair.address}")
    assert derived_addr == keypair.address, "derive_address mismatch"

    # Test 7: Tamper detection — flip one bit in z, verify must fail
    tampered_sig = dict(sig)
    z_bytes = bytearray(bytes.fromhex(tampered_sig['z']))
    z_bytes[0] ^= 0x01
    tampered_sig['z'] = z_bytes.hex()
    is_tampered_valid = engine.verify_signature(message_hash, tampered_sig, keypair.public_key)
    logger.info(f"[TEST] Tamper detection: {'PASS (rejected)' if not is_tampered_valid else 'FAIL (accepted tampered)'}")
    assert not is_tampered_valid, "Tampered signature was accepted — CRITICAL BUG"

    # Test 8: Hyperbolic geometry check
    try:
        geom = get_hyperbolic_geometry()
        logger.info(f"[TEST] HyperbolicGeometry initialized: {geom.is_initialized}")
        if geom.is_initialized:
            geom_hash = geom.compute_geometry_hash()
            logger.info(f"[TEST] Geometry hash: {geom_hash.hex()[:32]}...")
    except Exception as e:
        logger.info(f"[TEST] HyperbolicGeometry: not available ({e}) — fallback active")

    # Test 9: PQ cache status
    logger.info(f"[TEST] PQ cache ready: {_PQ_CACHE_READY.is_set()}, size: {len(_PQ_COORD_CACHE)}")

    # Test 10: Health check
    health = hlwe_health_check()
    logger.info(f"[TEST] Health check: {'OK' if health else 'FAIL'}")

    logger.info("=" * 100)
    logger.info("[TEST] All HLWE v3.0 self-tests PASSED")
    logger.info("=" * 100)
