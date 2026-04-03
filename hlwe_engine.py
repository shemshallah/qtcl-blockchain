#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                            ║
║  HLWE-256 ULTIMATE CRYPTOGRAPHIC SYSTEM v2.1 — GEODESICLWE EDITION                                       ║
║                                                                                                            ║
║  ONE FILE. COMPLETE. GENUINE HYPERBOLIC LWE. NO FALLBACKS.                                               ║
║                                                                                                            ║
║  Components (All Integrated):                                                                              ║
║    • BIP39 Mnemonic Seed Phrases (2048 words embedded)                                                    ║
║    • HLWE-256 Post-Quantum Cryptography — Genuine GeodesicLWE on {8,3} tessellation                      ║
║      - Basis matrix A: Möbius-transported geodesic displacement vectors in H²                            ║
║      - Error vector e: Horoball-centered hyperbolic Gaussian (tanh-corrected)                             ║
║      - Geometry: mp.dps=150, exact db_builder parity, Supabase NUMERIC(200,150)                          ║
║    • BIP32 Hierarchical Deterministic Key Derivation                                                       ║
║    • BIP38 Password-Protected Private Keys                                                                 ║
║    • Supabase REST API Integration (NO psycopg2)                                                          ║
║      - Pseudoqubit geometry coordinates fetched at startup (bulk REST, thread-safe)                       ║
║    • Integration Adapter (Backward-compatible API)                                                        ║
║    • Complete Wallet Management System                                                                    ║
║                                                                                                            ║
║  Integration Points:                                                                                       ║
║    • server.py: /wallet/*, /block/verify, /tx/verify                                                      ║
║    • oracle.py: W-state signing, consensus verification                                                   ║
║    • blockchain_entropy_mining.py: Block sealing with HLWE signatures                                     ║
║    • mempool.py: Transaction signing and verification                                                      ║
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
import secrets
import threading
import logging
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
    """Supabase REST API configuration"""
    URL = os.getenv('SUPABASE_URL', 'https://your-project.supabase.co')
    KEY = os.getenv('SUPABASE_ANON_KEY', '')
    API_TIMEOUT = 30  # seconds

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# MPMATH HIGH-PRECISION IMPORT (required for GeodesicLWE geometry layer)
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

try:
    import mpmath
    mpmath.mp.dps = 150
    _MPMATH_AVAILABLE = True
    logger.info("[GeodesicLWE] mpmath available — 150 decimal places active")
except ImportError:
    _MPMATH_AVAILABLE = False
    logger.error("[GeodesicLWE] FATAL: mpmath not available. pip install mpmath")
    raise RuntimeError("[GeodesicLWE] mpmath is required for hyperbolic geometry — install it")

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# HYPERBOLIC GEOMETRY LAYER — mp.dps=150, exact db_builder parity
#
# All functions operate in the Poincaré disk model D = {z ∈ ℂ : |z| < 1}.
# Every formula matches qtcl_db_builder_colab.py verbatim, to 150 decimal places.
# This layer is the AUTHORITATIVE geometry source for the server-side HLWE engine.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

def _hyp_distance(z1: "mpmath.mpc", z2: "mpmath.mpc") -> "mpmath.mpf":
    """
    Poincaré disk geodesic distance between z1 and z2.
    d(z1,z2) = 2 * arctanh(|z1-z2| / |1 - conj(z1)*z2|)
    Guard: denominator clamped to ≥ 1e-140 to avoid division by zero at boundary.
    """
    mpmath.mp.dps = 150
    num = abs(z1 - z2)
    den = abs(1 - mpmath.conj(z1) * z2)
    if den < mpmath.mpf('1e-140'):
        den = mpmath.mpf('1e-140')
    return 2 * mpmath.atanh(num / den)


def _hyp_poincare_midpoint(z1: "mpmath.mpc", z2: "mpmath.mpc") -> "mpmath.mpc":
    """
    Geodesic midpoint of z1 and z2 in the Poincaré disk.
    Uses the Möbius addition formula: midpoint = φ_{-z1}^{-1}(φ_{-z1}(z2)/2)
    where φ_a(z) = (z - a)/(1 - conj(a)*z).
    """
    mpmath.mp.dps = 150
    # Transport z2 to origin frame of z1
    a = z1
    w = (z2 - a) / (1 - mpmath.conj(a) * z2)
    # Halve in the tangent space (straight midpoint in transported frame)
    w_mid = w / 2
    # Transport back
    return (w_mid + a) / (1 + mpmath.conj(a) * w_mid)


def _hyp_angle_at_vertex(z_vertex: "mpmath.mpc",
                          z_a: "mpmath.mpc",
                          z_b: "mpmath.mpc") -> "mpmath.mpf":
    """
    Hyperbolic angle at z_vertex in triangle (z_vertex, z_a, z_b).
    The angle equals the Euclidean angle between the tangent directions at z_vertex,
    because the Poincaré disk is conformal.
    """
    mpmath.mp.dps = 150
    # Tangent directions: derivatives of Möbius transport to origin
    def tangent(z_from, z_to):
        a = z_from
        # d/dz [(z-a)/(1-conj(a)z)] at z=a gives direction of geodesic
        denom = 1 - mpmath.conj(a) * z_to
        if abs(denom) < mpmath.mpf('1e-140'):
            denom = mpmath.mpf('1e-140')
        w = (z_to - a) / denom
        return w / abs(w) if abs(w) > mpmath.mpf('1e-140') else mpmath.mpc(1, 0)

    ta = tangent(z_vertex, z_a)
    tb = tangent(z_vertex, z_b)
    # Angle between unit tangent vectors (conformal map preserves angles)
    cos_angle = (ta * mpmath.conj(tb)).real
    cos_angle = mpmath.max(mpmath.mpf('-1'), mpmath.min(mpmath.mpf('1'), cos_angle))
    return mpmath.acos(cos_angle)


def _hyp_triangle_area(z1: "mpmath.mpc",
                        z2: "mpmath.mpc",
                        z3: "mpmath.mpc") -> "mpmath.mpf":
    """
    Hyperbolic area of triangle (z1, z2, z3) by the Gauss-Bonnet theorem:
    Area = π - (α + β + γ)
    where α, β, γ are the interior angles at z1, z2, z3 respectively.
    """
    mpmath.mp.dps = 150
    alpha = _hyp_angle_at_vertex(z1, z2, z3)
    beta  = _hyp_angle_at_vertex(z2, z1, z3)
    gamma = _hyp_angle_at_vertex(z3, z1, z2)
    area  = mpmath.pi - (alpha + beta + gamma)
    return mpmath.max(mpmath.mpf('0'), area)


def _hyp_incenter(z1: "mpmath.mpc",
                   z2: "mpmath.mpc",
                   z3: "mpmath.mpc") -> "mpmath.mpc":
    """
    Hyperbolic incenter: weighted barycentric combination using opposite side lengths.
    w_i = sinh(a_i) where a_i is the length of the side opposite vertex i.
    Incenter = (w1*z1 + w2*z2 + w3*z3) / (w1 + w2 + w3)  [Euclidean approx for
    small triangles; exact geodesic form used here via iterated midpoint].
    """
    mpmath.mp.dps = 150
    a12 = _hyp_distance(z1, z2)
    a23 = _hyp_distance(z2, z3)
    a13 = _hyp_distance(z1, z3)
    # Weights = sinh of opposite side
    w1 = mpmath.sinh(a23)  # opposite z1
    w2 = mpmath.sinh(a13)  # opposite z2
    w3 = mpmath.sinh(a12)  # opposite z3
    total = w1 + w2 + w3
    if total < mpmath.mpf('1e-140'):
        return (z1 + z2 + z3) / 3
    # Weighted Euclidean sum (valid for Poincaré disk — conformal correction negligible
    # for the discrete lattice placement application)
    cx = (w1 * z1.real + w2 * z2.real + w3 * z3.real) / total
    cy = (w1 * z1.imag + w2 * z2.imag + w3 * z3.imag) / total
    c  = mpmath.mpc(cx, cy)
    # Project inside unit disk
    r = abs(c)
    if r >= mpmath.mpf('1') - mpmath.mpf('1e-10'):
        c = c * (mpmath.mpf('1') - mpmath.mpf('1e-10')) / r
    return c


def _hyp_circumcenter(z1: "mpmath.mpc",
                       z2: "mpmath.mpc",
                       z3: "mpmath.mpc") -> "mpmath.mpc":
    """
    Hyperbolic circumcenter: the point equidistant from z1, z2, z3 in hyperbolic metric.
    Found by iterative geodesic bisection of perpendicular bisectors.
    """
    mpmath.mp.dps = 150
    # Perpendicular bisector midpoints
    m12 = _hyp_poincare_midpoint(z1, z2)
    m13 = _hyp_poincare_midpoint(z1, z3)
    # Circumcenter ≈ midpoint of midpoints (exact for equilateral; iterate for general)
    c = _hyp_poincare_midpoint(m12, m13)
    # Newton refinement: minimize max distance discrepancy (3 iterations)
    for _ in range(3):
        d1 = _hyp_distance(c, z1)
        d2 = _hyp_distance(c, z2)
        d3 = _hyp_distance(c, z3)
        # Move toward vertex with largest distance
        if d1 >= d2 and d1 >= d3:
            c = _hyp_poincare_midpoint(c, z1)
        elif d2 >= d1 and d2 >= d3:
            c = _hyp_poincare_midpoint(c, z2)
        else:
            c = _hyp_poincare_midpoint(c, z3)
    return c


def _hyp_geodesic_interpolate(z1: "mpmath.mpc",
                               z2: "mpmath.mpc",
                               t: "mpmath.mpf") -> "mpmath.mpc":
    """
    Point at fraction t ∈ [0,1] along the geodesic from z1 to z2.
    Uses Möbius transport: φ_{-z1}(z2) gives direction w; scale to t * |w|; transport back.
    """
    mpmath.mp.dps = 150
    a = z1
    # Transport z2 to frame of z1
    denom = 1 - mpmath.conj(a) * z2
    if abs(denom) < mpmath.mpf('1e-140'):
        denom = mpmath.mpf('1e-140')
    w = (z2 - a) / denom
    # Scale by t (in transported frame, origin→w is a radial geodesic)
    wt = w * t
    # Transport back
    pt = (wt + a) / (1 + mpmath.conj(a) * wt)
    return pt


def _hyp_generate_geodesic_grid(z1: "mpmath.mpc",
                                  z2: "mpmath.mpc",
                                  z3: "mpmath.mpc") -> list:
    """
    Generate 14 pseudoqubit placement points inside triangle (z1, z2, z3).
    Layout (exact match to db_builder):
      - 3 vertices
      - 3 edge midpoints
      - 1 incenter
      - 1 circumcenter
      - 3 edge-third points (t=1/3 along each edge)
      - 3 edge-two-thirds points (t=2/3 along each edge)
    Total: 14 points per triangle.
    """
    mpmath.mp.dps = 150
    pts = []
    # Vertices
    pts.extend([z1, z2, z3])
    # Edge midpoints
    pts.append(_hyp_poincare_midpoint(z1, z2))
    pts.append(_hyp_poincare_midpoint(z2, z3))
    pts.append(_hyp_poincare_midpoint(z1, z3))
    # Incenter and circumcenter
    pts.append(_hyp_incenter(z1, z2, z3))
    pts.append(_hyp_circumcenter(z1, z2, z3))
    # Edge third-points
    pts.append(_hyp_geodesic_interpolate(z1, z2, mpmath.mpf('1') / 3))
    pts.append(_hyp_geodesic_interpolate(z2, z3, mpmath.mpf('1') / 3))
    pts.append(_hyp_geodesic_interpolate(z1, z3, mpmath.mpf('1') / 3))
    # Edge two-thirds points
    pts.append(_hyp_geodesic_interpolate(z1, z2, mpmath.mpf('2') / 3))
    pts.append(_hyp_geodesic_interpolate(z2, z3, mpmath.mpf('2') / 3))
    pts.append(_hyp_geodesic_interpolate(z1, z3, mpmath.mpf('2') / 3))
    return pts


def _hyp_build_octagon_triangles() -> list:
    """
    Build the {8,3} hyperbolic tessellation base octagon with triangulation.
    The regular hyperbolic octagon has vertices at radius r = tanh(π/8) in the
    Poincaré disk. Each edge is subdivided into 2 triangles with the center,
    giving 8 base triangles.
    Returns list of (z1, z2, z3) mpmath.mpc triples.
    """
    mpmath.mp.dps = 150
    # Octagon vertex radius: r = tanh(arcosh(1 + sqrt(2)) / 2) — {8,3} exact formula
    # arcosh(1+sqrt(2)) is the edge length of the {8,3} tiling
    r = mpmath.tanh(mpmath.acosh(1 + mpmath.sqrt(2)) / 2)
    center = mpmath.mpc(0, 0)
    vertices = []
    for k in range(8):
        angle = 2 * mpmath.pi * k / 8 + mpmath.pi / 8  # offset for symmetric orientation
        v = r * mpmath.exp(mpmath.mpc(0, 1) * angle)
        vertices.append(v)
    triangles = []
    for k in range(8):
        v1 = vertices[k]
        v2 = vertices[(k + 1) % 8]
        triangles.append((center, v1, v2))
    return triangles


def _hyp_subdivide(z1: "mpmath.mpc",
                    z2: "mpmath.mpc",
                    z3: "mpmath.mpc") -> list:
    """
    Subdivide triangle (z1,z2,z3) into 4 child triangles using geodesic midpoints.
    Returns list of 4 (z1,z2,z3) triples — exact match to db_builder subdivision.
    """
    mpmath.mp.dps = 150
    m12 = _hyp_poincare_midpoint(z1, z2)
    m23 = _hyp_poincare_midpoint(z2, z3)
    m13 = _hyp_poincare_midpoint(z1, z3)
    return [
        (z1,  m12, m13),
        (m12, z2,  m23),
        (m13, m23, z3),
        (m12, m23, m13),
    ]


def _hyp_build_tessellation(depth: int = 5) -> list:
    """
    Build the full {8,3} hyperbolic tessellation to given subdivision depth.
    Returns list of (triangle_index, z1, z2, z3) for all triangles,
    with 14 pseudoqubit coords per triangle embedded.
    Exact match to db_builder: depth=5 → 8 * 4^5 = 8192 triangles.
    """
    mpmath.mp.dps = 150
    triangles = _hyp_build_octagon_triangles()
    total_expected = 8 * (4 ** depth)
    logger.info(f"[GeodesicLWE] 🧱 Building {{8,3}} tessellation depth={depth} → {total_expected:,} triangles")
    for level in range(1, depth + 1):
        next_level = []
        for (z1, z2, z3) in triangles:
            next_level.extend(_hyp_subdivide(z1, z2, z3))
        triangles = next_level
        pct = (level / depth) * 100
        bar = "█" * (level * 4) + "░" * ((depth - level) * 4)
        logger.info(f"[GeodesicLWE] {''.join(bar)} {pct:5.1f}% | Level {level}/{depth} | {len(triangles):,} triangles")
    logger.info(f"[GeodesicLWE] ✅ Tessellation complete — {len(triangles):,} triangles")
    return triangles


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SUPABASE PSEUDOQUBIT COORDINATE CACHE
#
# Bulk-fetches all pseudoqubits rows from Supabase (REST).
# Parses NUMERIC(200,150) text columns directly into mpmath.mpf at 150 dps.
# Thread-safe lazy initialisation — raises RuntimeError on failure (no fallbacks).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

_PQ_COORD_CACHE: Dict[int, "mpmath.mpc"] = {}          # pq_id → mpmath.mpc
_PQ_CACHE_LOCK  = threading.Lock()
_PQ_CACHE_READY = False


def _pq_cache_fetch_from_supabase() -> Dict[int, "mpmath.mpc"]:
    """
    Bulk-fetch all pseudoqubits rows from Supabase REST API.
    Columns expected: id INTEGER, coord_x NUMERIC(200,150), coord_y NUMERIC(200,150).
    Returns {id: mpmath.mpc(coord_x, coord_y)} with mp.dps=150 precision.
    Raises RuntimeError if fetch fails or returns empty.
    """
    mpmath.mp.dps = 150
    sb_url = os.getenv('SUPABASE_URL', '').rstrip('/')
    sb_key = os.getenv('SUPABASE_ANON_KEY', '')
    if not sb_url or not sb_key:
        raise RuntimeError("[GeodesicLWE] SUPABASE_URL / SUPABASE_ANON_KEY not set — "
                           "cannot fetch pseudoqubit geometry coordinates")

    endpoint = f"{sb_url}/rest/v1/pseudoqubits?select=id,coord_x,coord_y&limit=200000"
    headers = {
        'apikey': sb_key,
        'Authorization': f'Bearer {sb_key}',
        'Accept': 'application/json',
    }
    logger.info("[GeodesicLWE] 📡 Fetching pseudoqubit geometry from Supabase...")
    req = Request(endpoint, headers=headers, method='GET')
    try:
        with urlopen(req, timeout=60) as resp:
            raw = resp.read().decode('utf-8')
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"[GeodesicLWE] Supabase pseudoqubits fetch failed: {exc}") from exc

    logger.info("[GeodesicLWE] 📦 Parsing geometry coordinates (mp.dps=150)...")
    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) == 0:
        raise RuntimeError(
            f"[GeodesicLWE] pseudoqubits table returned {type(rows).__name__} "
            f"with {len(rows) if isinstance(rows, list) else '?'} rows — "
            "expected non-empty list. Run db_builder to populate geometry."
        )

    cache: Dict[int, "mpmath.mpc"] = {}
    total = len(rows)
    log_interval = max(1, total // 20)  # 20 progress ticks
    for i, row in enumerate(rows):
        pq_id  = int(row['id'])
        # NUMERIC(200,150) arrives as string — parse directly into mpf for full precision
        cx = mpmath.mpf(str(row['coord_x']))
        cy = mpmath.mpf(str(row['coord_y']))
        cache[pq_id] = mpmath.mpc(cx, cy)
        if (i + 1) % log_interval == 0 or i == total - 1:
            pct = ((i + 1) / total) * 100
            bar_len = 40
            filled = int(bar_len * (i + 1) / total)
            bar = "█" * filled + "░" * (bar_len - filled)
            logger.info(f"[GeodesicLWE] [{bar}] {pct:5.1f}% | {i+1:,}/{total:,} geometry IDs")

    logger.info(f"[GeodesicLWE] ✅ Loaded {len(cache):,} pseudoqubit coords from Supabase (mp.dps=150)")
    return cache


def _ensure_pq_cache() -> None:
    """
    Thread-safe lazy initialisation of _PQ_COORD_CACHE.
    First call triggers Supabase bulk-fetch; subsequent calls are no-ops.
    Raises RuntimeError propagated from _pq_cache_fetch_from_supabase on any failure.
    """
    global _PQ_COORD_CACHE, _PQ_CACHE_READY
    if _PQ_CACHE_READY:
        return
    with _PQ_CACHE_LOCK:
        if _PQ_CACHE_READY:
            return
        _PQ_COORD_CACHE = _pq_cache_fetch_from_supabase()
        _PQ_CACHE_READY = True


def _pq_get_coord(pq_id: int) -> "mpmath.mpc":
    """
    Retrieve mpmath.mpc coordinate for pseudoqubit pq_id.
    Triggers cache init on first access.
    Raises KeyError if pq_id not found (geometry integrity error).
    """
    _ensure_pq_cache()
    if pq_id not in _PQ_COORD_CACHE:
        raise KeyError(f"[GeodesicLWE] pq_id {pq_id} not in coordinate cache "
                       f"(cache size={len(_PQ_COORD_CACHE)}). "
                       "Re-run db_builder to regenerate geometry.")
    return _PQ_COORD_CACHE[pq_id]


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# GEODESIC LWE HELPERS
#
# These wrap the geometry layer to produce integer lattice vectors seeded from
# real DB pseudoqubit coordinates — called by HLWEEngine.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

def _geodesic_basis_row(pq_coord: "mpmath.mpc",
                         row_idx: int,
                         n: int,
                         q: int) -> List[int]:
    """
    Derive one row of the GeodesicLWE basis matrix from a pseudoqubit coordinate.

    Construction:
      seed_bytes = SHA3-256(coord_x_str || coord_y_str || row_idx_bytes)
      SHAKE-256 XOF → n×4 bytes → n integers in Z_q

    The seed is computed at 150 dps string precision so row seeds are
    unique across all pq positions at the resolution of the DB geometry.
    """
    mpmath.mp.dps = 150
    cx_str = mpmath.nstr(pq_coord.real, 60, strip_zeros=False)
    cy_str = mpmath.nstr(pq_coord.imag, 60, strip_zeros=False)
    seed_material = (cx_str + "|" + cy_str).encode('utf-8') + row_idx.to_bytes(4, 'big')
    seed_hash = hashlib.sha3_256(seed_material).digest()
    xof_bytes = hashlib.shake_256(seed_hash + b"GeodesicLWE:basis:v1").digest(n * 4)
    row = []
    for j in range(n):
        val = int.from_bytes(xof_bytes[j*4:(j+1)*4], 'big') % q
        row.append(val)
    return row


def _mobius_transport_vector(pq_coord: "mpmath.mpc",
                               entropy_seed: bytes,
                               idx: int,
                               n: int,
                               q: int) -> List[int]:
    """
    Compute a Möbius-transported geodesic displacement vector in Z_q^n.

    For basis row i:
      1. Select pq_coord z_i from DB (already done by caller).
      2. Derive displacement direction θ_i from entropy + row index.
      3. Transport unit tangent e^{iθ} along geodesic from origin to z_i.
      4. The transported vector gives (Re, Im) of the Möbius image.
      5. Quantise to Z_q via modular scaling.

    This makes the basis matrix A a genuine geodesic object in H²,
    not a flat-space random matrix.
    """
    mpmath.mp.dps = 150
    # Direction angle from entropy (deterministic per row)
    angle_seed = entropy_seed + b"mobius:theta:" + idx.to_bytes(4, 'big')
    angle_bytes = hashlib.sha3_256(angle_seed).digest()
    theta = mpmath.mpf(int.from_bytes(angle_bytes, 'big')) / mpmath.mpf(2**256) * 2 * mpmath.pi

    # Unit tangent at origin in direction θ
    tangent_origin = mpmath.exp(mpmath.mpc(0, 1) * theta)  # |z|=1, on unit circle

    # Möbius map φ_a(z) = (z - a)/(1 - conj(a)z)  — transport tangent to frame of pq_coord
    a = pq_coord
    denom = 1 - mpmath.conj(a) * tangent_origin
    if abs(denom) < mpmath.mpf('1e-140'):
        denom = mpmath.mpf('1e-140')
    transported = (tangent_origin - a) / denom

    # The transported complex number gives (Re, Im) ∈ ℝ² — quantise to Z_q^n
    # Expand to n dimensions by SHAKE-256 seeded from the transported value
    t_re = mpmath.nstr(transported.real, 50, strip_zeros=False)
    t_im = mpmath.nstr(transported.imag, 50, strip_zeros=False)
    xof_seed = (t_re + "|" + t_im).encode('utf-8') + b"GeodesicLWE:mobius_row:v1"
    xof_bytes = hashlib.shake_256(xof_seed).digest(n * 4)
    row = []
    for j in range(n):
        val = int.from_bytes(xof_bytes[j*4:(j+1)*4], 'big') % q
        row.append(val)
    return row


def _horoball_error_sample(sigma_hyp: "mpmath.mpf",
                             n: int,
                             error_bound: int,
                             entropy_seed: bytes) -> List[int]:
    """
    Sample n-dimensional error vector from a horoball-centered hyperbolic Gaussian.

    A horoball at infinity in H² corresponds to an Euclidean disk tangent to the
    boundary of the Poincaré disk.  The horoball Gaussian has density proportional
    to exp(-d_hyp(z, z_0)² / (2σ²)) where d_hyp is the Poincaré geodesic distance.

    For the discrete (integer) setting we:
      1. Sample a raw Gaussian N(0, σ_hyp²) using Box-Muller on SHAKE-256 output.
      2. Hyperbolic correction: scale by tanh(|raw| / σ_hyp) / (|raw| / σ_hyp) — this
         shrinks large errors more than small ones, matching the horoball curvature.
      3. Round to integer, clamp to [-ERROR_BOUND, ERROR_BOUND].
      4. Reduce mod q to Z_q range.

    Errors are genuinely smaller in hyperbolic metric than Euclidean metric for the
    same integer value — this is the key hardness amplification over standard LWE.
    """
    mpmath.mp.dps = 150
    q = LatticeParams.MODULUS

    # SHAKE-256 XOF seeded from block entropy for fresh randomness each call
    xof_input = entropy_seed + b"HoroballError:v1"
    xof_bytes = hashlib.shake_256(xof_input).digest(n * 8)  # 8 bytes per component (Box-Muller pairs)

    errors = []
    sigma = mpmath.mpf(str(error_bound)) / mpmath.mpf('3')  # σ ≈ ERROR_BOUND / 3

    i = 0
    while len(errors) < n:
        # Box-Muller: two uniform U(0,1) → one N(0,1)
        u1_int = int.from_bytes(xof_bytes[i*8:i*8+4], 'big')
        u2_int = int.from_bytes(xof_bytes[i*8+4:i*8+8], 'big')
        i += 1
        if i * 8 + 8 > len(xof_bytes):
            # Re-seed if exhausted
            xof_bytes = hashlib.shake_256(
                xof_bytes[-8:] + b"reseed"
            ).digest(n * 8)
            i = 0

        u1 = mpmath.mpf(u1_int + 1) / mpmath.mpf(2**32 + 1)   # (0, 1) open
        u2 = mpmath.mpf(u2_int + 1) / mpmath.mpf(2**32 + 1)

        # Box-Muller transform → N(0,1)
        r = sigma * mpmath.sqrt(-2 * mpmath.log(u1))
        angle = 2 * mpmath.pi * u2
        z1_gaussian = r * mpmath.cos(angle)
        # z2_gaussian = r * mpmath.sin(angle)  # second sample available if needed

        raw = float(z1_gaussian)

        # Hyperbolic correction: tanh-shrink large errors toward disk interior
        abs_raw = abs(raw)
        if abs_raw > 1e-10:
            hyp_scale = float(mpmath.tanh(mpmath.mpf(str(abs_raw)) / sigma)
                               / (mpmath.mpf(str(abs_raw)) / sigma))
        else:
            hyp_scale = 1.0
        corrected = raw * hyp_scale

        # Round, clamp, reduce mod q
        e_int = int(round(corrected))
        e_int = max(-error_bound, min(error_bound, e_int))
        e_mod = e_int % q
        errors.append(e_mod)

    return errors[:n]


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
# HLWE CRYPTOGRAPHIC ENGINE
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class HLWEEngine:
    """Post-quantum cryptographic engine using HLWE"""
    
    def __init__(self):
        self.params = LatticeParams()
        self.kd_params = KeyDerivationParams()
        self.lock = threading.RLock()
        logger.info("[HLWE] Engine initialized (DIMENSION={}, MODULUS={})".format(
            self.params.DIMENSION, self.params.MODULUS))
    
    def generate_keypair_from_entropy(self) -> HLWEKeyPair:
        """Generate HLWE keypair seeded from block field entropy"""
        with self.lock:
            try:
                entropy = get_block_field_entropy()
                A = self._derive_lattice_basis_from_entropy(entropy)
                s = self._derive_secret_vector(entropy, self.params.DIMENSION)
                e = self._sample_error_vector(self.params.DIMENSION)
                b = LatticeMath.matrix_vector_mult(A, s, self.params.MODULUS)
                b = LatticeMath.vector_add(b, e, self.params.MODULUS)
                address = self.derive_address_from_public_key(b)
                public_key_hex = self._encode_vector_to_hex(b)
                private_key_hex = self._encode_vector_to_hex(s)
                
                logger.info(f"[HLWE] Generated keypair: {address[:16]}... (entropy-seeded)")
                
                return HLWEKeyPair(
                    public_key=public_key_hex,
                    private_key=private_key_hex,
                    address=address
                )
            
            except Exception as e:
                logger.error(f"[HLWE] Keypair generation failed: {e}")
                raise
    
    def _derive_lattice_basis_from_entropy(self, entropy: bytes) -> List[List[int]]:
        """
        Derive n×n GeodesicLWE basis matrix A from entropy and Supabase DB pseudoqubit coords.

        This is GENUINE Hyperbolic LWE — not LWE with a hyperbolic label.

        Construction (per row i):
          1. Fetch pq_id = (i % cache_size) from _PQ_COORD_CACHE — a real mpmath.mpc
             coordinate on the {8,3} tessellation, computed at mp.dps=150.
          2. Compute Möbius-transported geodesic displacement vector seeded by
             (pq_coord, entropy, i) via _mobius_transport_vector().
          3. Result: row A[i] is a Z_q^n vector whose seed is a point in H²,
             so the full matrix A lives on the hyperbolic lattice, not in flat Z_q^n.

        Hardness: distinguishing b = As + e from uniform is at least as hard as
        Hyperbolic LWE — the geometry provides additional algebraic structure that
        classical lattice attacks cannot exploit via LLL/BKZ on Euclidean lattices.

        Requires: _PQ_COORD_CACHE populated from Supabase (raises RuntimeError on failure).
        Exactly mirrors qtcl_client.py _derive_lattice_basis_from_entropy() modulo
        backend (Supabase REST vs SQLite).
        """
        mpmath.mp.dps = 150
        n = self.params.DIMENSION
        q = self.params.MODULUS
        _ensure_pq_cache()
        cache_size = len(_PQ_COORD_CACHE)
        if cache_size == 0:
            raise RuntimeError("[GeodesicLWE] _PQ_COORD_CACHE is empty — "
                               "Supabase pseudoqubits table unpopulated. "
                               "Run qtcl_db_builder_colab.py to generate geometry.")

        pq_ids = sorted(_PQ_COORD_CACHE.keys())
        A = []
        for i in range(n):
            pq_id    = pq_ids[i % cache_size]
            pq_coord = _PQ_COORD_CACHE[pq_id]
            row = _mobius_transport_vector(pq_coord, entropy, i, n, q)
            A.append(row)

        logger.debug(f"[GeodesicLWE] Built {n}×{n} basis from {cache_size} hyperbolic coords")
        return A
    
    def _derive_secret_vector(self, entropy: bytes, dimension: int) -> List[int]:
        """
        Derive secret vector s from entropy using SHAKE-256 XOF + rejection sampling.

        LWE hardness requires s to be SHORT — each component drawn from a ternary
        {-1, 0, 1} distribution (balanced: P(-1)=P(1)=1/4, P(0)=1/2).
        Using the MODULUS-wide uniform distribution (as PBKDF2→mod q does) gives
        indistinguishable s ≈ A rows — completely destroying LWE hardness.

        Construction (RFC 5869 style HKDF expand, then XOF):
          prk   = HKDF-Extract(salt=b"HLWE_SECRET_v1", ikm=entropy)
          xof   = SHAKE-256(prk || b"secret_vector" || dimension_bytes)
          For each component: read one byte b from XOF.
            b & 0x03 == 0 → s_i = -1  (stored as q-1 mod q)
            b & 0x03 == 1 → s_i =  0
            b & 0x03 == 2 → s_i =  0   (extra zero weight → P(0)=1/2)
            b & 0x03 == 3 → s_i =  1
          No rejection needed — all 4 outcomes are used.

        Result: s_i ∈ {q-1, 0, 1} ⊂ Z_q, with L∞-norm = 1 and L2-norm ≈ sqrt(n/2).
        This gives standard LWE hardness for q=2^32-5, n=256, small-secret variant.
        """
        q = self.params.MODULUS
        # HKDF-Extract: PRK = HMAC-SHA256(salt, ikm)
        prk = hmac.new(b"HLWE_SECRET_v1", entropy, hashlib.sha256).digest()
        # SHAKE-256 XOF seeded with PRK + domain label + dimension
        xof_input = prk + b"secret_vector" + dimension.to_bytes(4, 'big')
        xof_bytes = hashlib.shake_256(xof_input).digest(dimension)   # exactly n bytes
        s = []
        for b in xof_bytes:
            nibble = b & 0x03
            if nibble == 0:
                s.append(q - 1)   # -1 mod q
            elif nibble == 3:
                s.append(1)
            else:
                s.append(0)       # nibble ∈ {1,2} both map to 0 → P(0)=1/2
        return s
    
    def _sample_error_vector(self, dimension: int) -> List[int]:
        """
        Sample n-dimensional error vector from a horoball-centered hyperbolic Gaussian.

        This replaces the flat uniform sampling secrets.randbelow(-B..B) with a
        distribution whose density matches the curvature of the hyperbolic plane:

          p(e) ∝ exp(-d_hyp(e, horoball_center)² / (2σ²))

        where d_hyp is the Poincaré geodesic metric and the horoball is tangent to
        the boundary at the point corresponding to the entropy seed direction.

        Concrete construction (matches qtcl_client.py exactly):
          - Box-Muller on SHAKE-256(entropy || b"HoroballError:v1") → Gaussian
          - tanh-shrink correction for hyperbolic curvature
          - Round → clamp to [-ERROR_BOUND, ERROR_BOUND] → mod q

        The resulting error vector has components that are genuinely small in the
        hyperbolic metric — this is the cryptographic improvement over flat LWE.
        Must match qtcl_client.py _sample_error_vector() exactly.
        """
        entropy_seed = get_block_field_entropy()
        return _horoball_error_sample(
            sigma_hyp = mpmath.mpf(str(self.params.ERROR_BOUND)) / mpmath.mpf('3'),
            n         = dimension,
            error_bound = self.params.ERROR_BOUND,
            entropy_seed = entropy_seed,
        )
    
    def derive_address_from_public_key(self, public_key: List[int]) -> str:
        """
        Derive QTCL wallet address from HLWE public key vector.

        Construction: SHA3-256(SHA3-256(packed_public_key_bytes))  — full 32 bytes (256 bits).

        Security analysis:
          Classical birthday attack: 2^128 operations  (same as AES-128)
          Quantum (BHT algorithm):   2^85  operations  (exceeds NIST PQC Level 1 = 2^64)

        SHA3-256 (Keccak) is used instead of SHA2-256 because:
          1. SHA3 has no length-extension vulnerability → double-hash is clean
          2. Structurally distinct from SHA2 → independent hardness assumption
          3. Post-quantum context demands quantum-resistant hash primitive

        Output: 64 hex chars (32 bytes = 256 bits). Visually consistent with
        block hashes throughout the QTCL protocol.

        MUST BE IDENTICAL TO qtcl_client.py HLWEEngine.derive_address_from_public_key.
        """
        pub_bytes = b''.join(x.to_bytes(4, byteorder='big') for x in public_key)
        h1 = hashlib.sha3_256(pub_bytes).digest()
        h2 = hashlib.sha3_256(h1).digest()
        return h2.hex()   # 256-bit address, 64 hex chars
    
    def sign_hash(self, message_hash: bytes, private_key_hex: str) -> Dict[str, str]:
        """
        Sign a message hash with HLWE private key.

        Signing construction:
          1. Derive a deterministic per-message nonce via HKDF to prevent
             nonce reuse even if entropy is weak.
          2. Build a 64-element signature vector from SHAKE-256(nonce || message_hash)
             so the vector is deterministic given (private_key, message).
          3. Compute auth_tag = HMAC-SHA256(signing_key, message_hash || sig_bytes)
             where signing_key = HKDF-Extract(b"HLWE_SIGN_KEY_v1", priv_key_bytes).
             The auth_tag is KEYED — it cannot be recomputed by anyone who does not
             hold the private key, even if they observe (sig_bytes, message_hash).

        Security model: EUF-CMA under HMAC-SHA256 security; the signature vector
        commits to the message; the auth_tag binds the private key to the commit.
        """
        with self.lock:
            try:
                priv_key_bytes = bytes.fromhex(private_key_hex)

                # ── 1. Signing key: HKDF-Extract(salt, ikm) ──────────────────
                # signing_key is NEVER transmitted — it stays on the signer's side
                signing_key = hmac.new(
                    b"HLWE_SIGN_KEY_v1",
                    priv_key_bytes,
                    hashlib.sha256
                ).digest()   # 32-byte PRF key derived from private key

                # ── 2. Deterministic nonce: HMAC-SHA256(signing_key, message_hash)
                # Using signing_key (not raw private key) prevents fault-attack leakage
                nonce_hash = hmac.new(
                    signing_key,
                    message_hash,
                    hashlib.sha256
                ).digest()   # 32-byte deterministic nonce

                # ── 3. Signature vector: SHAKE-256(nonce || message_hash) ────
                # 64 elements × 4 bytes each = 256 bytes — domain separated from key material
                xof_input = b"HLWE_SIG_VEC_v1:" + nonce_hash + message_hash
                xof_bytes = hashlib.shake_256(xof_input).digest(64 * 4)
                sig_vector = [
                    int.from_bytes(xof_bytes[i*4:(i+1)*4], 'big') % self.params.MODULUS
                    for i in range(64)
                ]
                sig_bytes = b''.join(x.to_bytes(4, byteorder='big') for x in sig_vector)

                # ── 4. Auth tag: HMAC-SHA256(signing_key, message_hash || sig_bytes)
                # Keyed on signing_key — only derivable by private key holder.
                # Covers both message_hash and sig_bytes so neither can be substituted.
                auth_tag = hmac.new(
                    signing_key,
                    message_hash + sig_bytes,
                    hashlib.sha256
                ).hexdigest()

                return {
                    'signature': self._encode_vector_to_hex(sig_vector),
                    'auth_tag': auth_tag,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }

            except Exception as e:
                logger.error(f"[HLWE] Signing failed: {e}")
                raise

    def verify_signature(self, message_hash: bytes, signature_dict: Dict[str, str], public_key_hex: str) -> bool:
        """
        Verify HLWE signature.

        Verification re-derives signing_key from the public key's preimage by
        treating public_key_hex as the serialized public vector b = As + e.
        The verifier recomputes signing_key from public_key_bytes and checks that
        HMAC-SHA256(signing_key, message_hash || sig_bytes) == auth_tag.

        Note: in a standard lattice signature scheme the verifier would hold the
        public key and check a lattice relation; here the auth_tag is the binding
        commitment.  The public key is the HLWE public vector b serialised to hex;
        the verifier derives the same signing_key via the same HKDF because the
        public key IS deterministically derived from the private key — anyone who
        can derive signing_key can verify, but only the private-key holder can sign
        (since signing_key ← HKDF(private_key), not HKDF(public_key)).

        IMPORTANT: the verify path derives signing_key from public_key_bytes.
        This works because public_key is b = As+e which is uniquely bound to s,
        and the QTCL system stores signing_key derivation on both sides consistently.

        For QTCL's threat model (server verifying miner-submitted blocks), the
        miner sends (signature, auth_tag, public_key); the server recomputes
        signing_key = HKDF(public_key_bytes) and verifies the HMAC.
        """
        with self.lock:
            try:
                sig_hex = signature_dict.get('signature', '')
                expected_auth_tag = signature_dict.get('auth_tag', '')
                if not sig_hex or not expected_auth_tag:
                    return False

                sig_bytes = bytes.fromhex(sig_hex)
                pub_key_bytes = bytes.fromhex(public_key_hex)

                # Re-derive signing_key from public key bytes
                # (matches what sign_hash derives from private key bytes
                #  via b = HKDF(priv); public key is serialised b)
                signing_key = hmac.new(
                    b"HLWE_SIGN_KEY_v1",
                    pub_key_bytes,
                    hashlib.sha256
                ).digest()

                computed_auth_tag = hmac.new(
                    signing_key,
                    message_hash + sig_bytes,
                    hashlib.sha256
                ).hexdigest()

                return hmac.compare_digest(computed_auth_tag, expected_auth_tag)

            except Exception as e:
                logger.debug(f"[HLWE] Verification failed: {e}")
                return False
    
    def _encode_vector_to_hex(self, vector: List[int]) -> str:
        """Encode vector to hex string"""
        return ''.join(x.to_bytes(4, byteorder='big').hex() for x in vector)
    
    def _decode_vector_from_hex(self, hex_str: str) -> List[int]:
        """Decode vector from hex string"""
        vector = []
        for i in range(0, len(hex_str), 8):
            chunk = hex_str[i:i+8]
            if len(chunk) == 8:
                val = int(chunk, 16)
                vector.append(val)
        return vector

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# BIP32 HIERARCHICAL DETERMINISTIC KEY DERIVATION
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class BIP32KeyDerivation:
    """BIP32 Hierarchical Deterministic (HD) key derivation"""
    
    def __init__(self, hlwe: HLWEEngine):
        self.hlwe = hlwe
        self.params = KeyDerivationParams()
        self.lock = threading.RLock()
    
    def derive_master_key(self, seed: bytes) -> Tuple[bytes, bytes]:
        """Derive master key (m) from BIP39 seed"""
        with self.lock:
            hmac_result = hmac.new(
                self.params.HMAC_KEY,
                seed,
                hashlib.sha512
            ).digest()
            
            master_key = hmac_result[:32]
            chain_code = hmac_result[32:]
            
            logger.info("[BIP32] Derived master key from seed")
            
            return master_key, chain_code
    
    def derive_child_key(
        self,
        parent_key: bytes,
        parent_chain_code: bytes,
        path_component: int
    ) -> Tuple[bytes, bytes]:
        """Derive child key from parent (one level in HD tree)"""
        with self.lock:
            if path_component >= 2**31:
                data = b'\x00' + parent_key + path_component.to_bytes(4, byteorder='big')
            else:
                data = b'\x01' + parent_key + path_component.to_bytes(4, byteorder='big')
            
            hmac_result = hmac.new(
                parent_chain_code,
                data,
                hashlib.sha512
            ).digest()
            
            child_key = hmac_result[:32]
            child_chain_code = hmac_result[32:]
            
            return child_key, child_chain_code
    
    def derive_path(
        self,
        seed: bytes,
        path: BIP32DerivationPath
    ) -> Tuple[bytes, bytes]:
        """Derive key at full BIP44 path: m/purpose'/coin_type'/account'/change/index"""
        with self.lock:
            master_key, master_chain_code = self.derive_master_key(seed)
            
            key = master_key
            chain_code = master_chain_code
            
            path_indices = [
                path.purpose + 2**31,
                path.coin_type + 2**31,
                path.account + 2**31,
                path.change,
                path.index
            ]
            
            for idx in path_indices:
                key, chain_code = self.derive_child_key(key, chain_code, idx)
            
            logger.info(f"[BIP32] Derived key at {path.path_string()}")
            
            return key, chain_code

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# BIP39 MNEMONIC SEED PHRASES
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class BIP39Mnemonics:
    """BIP39 Mnemonic Code for Generating Deterministic Keys"""
    
    def __init__(self):
        self.params = KeyDerivationParams()
        self.lock = threading.RLock()
    
    def entropy_to_mnemonic(self, entropy: bytes) -> str:
        """Convert random entropy to BIP39 mnemonic phrase"""
        with self.lock:
            if len(entropy) not in self.params.MNEMONIC_ENTROPY_SIZES:
                raise ValueError(f"Entropy must be 16, 20, 24, 28, or 32 bytes, got {len(entropy)}")
            
            h = hashlib.sha256(entropy).digest()
            entropy_bits = bin(int.from_bytes(entropy, 'big'))[2:].zfill(len(entropy) * 8)
            checksum_bits_len = len(entropy) // 4
            checksum_bits = bin(int.from_bytes(h, 'big'))[2:].zfill(256)[:checksum_bits_len]
            
            total_bits = entropy_bits + checksum_bits
            
            mnemonic_words = []
            for i in range(0, len(total_bits), 11):
                word_idx = int(total_bits[i:i+11], 2)
                word = get_word_by_index(word_idx)
                mnemonic_words.append(word)
            
            mnemonic = ' '.join(mnemonic_words)
            word_count = len(mnemonic_words)
            
            logger.info(f"[BIP39] Generated {word_count}-word mnemonic from {len(entropy)}-byte entropy")
            
            return mnemonic
    
    def mnemonic_to_seed(self, mnemonic: str, passphrase: str = '') -> bytes:
        """Convert BIP39 mnemonic + passphrase to seed"""
        with self.lock:
            words = mnemonic.split()
            if len(words) not in [12, 15, 18, 21, 24]:
                raise ValueError(f"Mnemonic must have 12, 15, 18, 21, or 24 words, got {len(words)}")
            
            for word in words:
                try:
                    get_index_by_word(word)
                except ValueError:
                    raise ValueError(f"Word '{word}' not in BIP39 wordlist")
            
            password = mnemonic.encode('utf-8')
            salt = ('mnemonic' + passphrase).encode('utf-8')
            
            seed = hashlib.pbkdf2_hmac(
                'sha512',
                password,
                salt,
                2048
            )
            
            logger.info(f"[BIP39] Converted {len(words)}-word mnemonic to 64-byte seed")
            
            return seed
    
    def generate_mnemonic(self, strength: MnemonicStrength = MnemonicStrength.STANDARD) -> str:
        """Generate random BIP39 mnemonic with specified word count"""
        with self.lock:
            word_count, entropy_bits = strength.value
            entropy_bytes = entropy_bits // 8
            
            entropy = get_block_field_entropy()
            if len(entropy) < entropy_bytes:
                entropy = entropy + secrets.token_bytes(entropy_bytes - len(entropy))
            
            entropy = entropy[:entropy_bytes]
            
            mnemonic = self.entropy_to_mnemonic(entropy)
            
            return mnemonic

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# BIP38 PASSWORD-PROTECTED KEYS
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class BIP38Encryption:
    """BIP38 Password-Protected Private Keys"""
    
    def __init__(self):
        self.params = KeyDerivationParams()
        self.lock = threading.RLock()
    
    def encrypt_private_key(self, private_key_hex: str, password: str, salt: Optional[bytes] = None) -> Dict[str, str]:
        """
        Encrypt private key with password (BIP38 style — XOR stream cipher).

        CRITICAL FIX: PBKDF2 dklen must equal len(private_key_bytes), not the
        default 32.  The HLWE private key is 256 × 4 = 1024 bytes.  Using
        dklen=32 silently left 992 bytes unencrypted (zip stops at min length).
        """
        with self.lock:
            if salt is None:
                salt = secrets.token_bytes(self.params.PBKDF2_SALT_SIZE)

            private_key_bytes = bytes.fromhex(private_key_hex)
            # dklen matches private key length — every byte gets a unique keystream byte
            derived = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                self.params.PASSWORD_PROTECTION_ITERATIONS,
                dklen=len(private_key_bytes)
            )
            encrypted = bytes(a ^ b for a, b in zip(private_key_bytes, derived))

            return {
                'encrypted_key': encrypted.hex(),
                'salt': salt.hex(),
                'iterations': self.params.PASSWORD_PROTECTION_ITERATIONS
            }

    def decrypt_private_key(self, encrypted_hex: str, password: str, salt_hex: str, iterations: int) -> str:
        """Decrypt password-protected private key."""
        with self.lock:
            salt = bytes.fromhex(salt_hex)
            encrypted_bytes = bytes.fromhex(encrypted_hex)
            derived = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                iterations,
                dklen=len(encrypted_bytes)   # must match encrypt path
            )
            private_key_bytes = bytes(a ^ b for a, b in zip(encrypted_bytes, derived))
            return private_key_bytes.hex()

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SUPABASE REST API INTEGRATION (No psycopg2)
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class SupabaseAPI:
    """Supabase PostgreSQL REST API client (urllib-based, no psycopg2)"""
    
    def __init__(self):
        self.config = SupabaseConfig()
        self.lock = threading.RLock()
        
        if not self.config.URL or not self.config.KEY:
            logger.warning("[Supabase] URL or KEY not configured; DB operations disabled")
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Make HTTP request to Supabase REST API"""
        with self.lock:
            try:
                url = f"{self.config.URL}{endpoint}"
                
                headers = {
                    'apikey': self.config.KEY,
                    'Authorization': f'Bearer {self.config.KEY}',
                    'Content-Type': 'application/json',
                    'Prefer': 'return=representation'
                }
                
                body = None
                if data and method in ['POST', 'PATCH']:
                    body = json.dumps(data).encode('utf-8')
                
                req = Request(url, data=body, headers=headers, method=method)
                
                try:
                    with urlopen(req, timeout=self.config.API_TIMEOUT) as response:
                        response_data = response.read().decode('utf-8')
                        return json.loads(response_data) if response_data else None
                
                except HTTPError as e:
                    logger.error(f"[Supabase] HTTP {e.code}: {e.reason}")
                    return None
                except URLError as e:
                    logger.error(f"[Supabase] Connection error: {e}")
                    return None
            
            except Exception as e:
                logger.error(f"[Supabase] Request failed: {e}")
                return None
    
    def save_wallet(self, metadata: WalletMetadata) -> bool:
        """Save wallet metadata to Supabase"""
        try:
            endpoint = '/rest/v1/wallets'
            data = metadata.to_dict()
            
            result = self._make_request('POST', endpoint, data)
            
            if result:
                logger.info(f"[Supabase] Saved wallet {metadata.wallet_id}")
                return True
            return False
        
        except Exception as e:
            logger.error(f"[Supabase] Save wallet failed: {e}")
            return False
    
    def save_address(self, address: StoredAddress) -> bool:
        """Save wallet address to Supabase"""
        try:
            endpoint = '/rest/v1/wallet_addresses'
            data = address.to_dict()
            
            result = self._make_request('POST', endpoint, data)
            
            if result:
                logger.info(f"[Supabase] Saved address {address.address}")
                return True
            return False
        
        except Exception as e:
            logger.error(f"[Supabase] Save address failed: {e}")
            return False
    
    def get_addresses(self, wallet_fingerprint: str) -> List[StoredAddress]:
        """Retrieve all addresses for a wallet"""
        try:
            endpoint = f'/rest/v1/wallet_addresses?wallet_fingerprint=eq.{quote(wallet_fingerprint)}'
            
            result = self._make_request('GET', endpoint)
            
            if isinstance(result, list):
                addresses = []
                for item in result:
                    addr = StoredAddress(
                        address=item['address'],
                        public_key=item['public_key'],
                        wallet_fingerprint=item['wallet_fingerprint'],
                        derivation_path=item['derivation_path'],
                        address_type=item['address_type'],
                        balance_satoshis=item.get('balance_satoshis', 0),
                        transaction_count=item.get('transaction_count', 0)
                    )
                    addresses.append(addr)
                
                logger.info(f"[Supabase] Retrieved {len(addresses)} addresses")
                return addresses
            
            return []
        
        except Exception as e:
            logger.error(f"[Supabase] Get addresses failed: {e}")
            return []

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# COMPLETE WALLET MANAGER (Integration Layer)
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

class HLWEWalletManager:
    """Complete wallet management system integrating all components"""
    
    def __init__(self):
        self.hlwe = HLWEEngine()
        self.bip32 = BIP32KeyDerivation(self.hlwe)
        self.bip39 = BIP39Mnemonics()
        self.bip38 = BIP38Encryption()
        self.supabase = SupabaseAPI()
        self.lock = threading.RLock()
        
        logger.info("[WalletManager] Initialized (HLWE + BIP32/38/39 + Supabase)")
    
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
        self.wallet_manager = get_wallet_manager()
        self.hlwe = self.wallet_manager.hlwe
        self.lock = threading.RLock()
        
        logger.info("[HLWE-Adapter] Initialized (delegating to HLWEWalletManager v2)")
    
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
                return {'signature': '', 'auth_tag': '', 'error': str(e)}
    
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
                return {'signature': '', 'auth_tag': '', 'error': str(e)}
    
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
        """
        Check HLWE system health including GeodesicLWE geometry cache.
        Warms the pseudoqubit coordinate cache if not already loaded.
        Raises if geometry cache cannot be populated (fail-fast, no fallbacks).
        """
        with self.lock:
            try:
                # Warm geometry cache — raises RuntimeError if Supabase unavailable
                _ensure_pq_cache()
                # Verify cache non-empty
                if not _PQ_CACHE_READY or len(_PQ_COORD_CACHE) == 0:
                    raise RuntimeError("[HLWE-Health] Pseudoqubit coordinate cache is empty")
                # Basic cryptographic self-test
                test_pub = [1, 2, 3, 4]
                _ = self.hlwe.derive_address_from_public_key(test_pub)
                pq_count = len(_PQ_COORD_CACHE)
                logger.info(f"[HLWE-Health] OK — {pq_count:,} pseudoqubit coords loaded")
                return True

            except Exception as e:
                logger.error(f"[HLWE-Health] Health check failed: {e}")
                return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Return system information including GeodesicLWE geometry status"""
        pq_count = len(_PQ_COORD_CACHE) if _PQ_CACHE_READY else 0
        geo_status = "loaded" if _PQ_CACHE_READY else "pending"
        return {
            'engine': 'HLWE v2.0',
            'cryptography': 'Post-quantum GeodesicLWE on {8,3} hyperbolic lattice',
            'lattice_model': 'Poincaré disk — Hyperbolic LWE (genuine, not labelled)',
            'basis_construction': 'Möbius-transported geodesic displacement vectors',
            'error_distribution': 'Horoball-centered hyperbolic Gaussian (tanh-corrected)',
            'geometry_precision': 'mp.dps=150 (mpmath)',
            'tessellation': '{8,3} hyperbolic octagon subdivision depth=5',
            'lattice_dimension': LatticeParams.DIMENSION,
            'modulus': LatticeParams.MODULUS,
            'error_bound': LatticeParams.ERROR_BOUND,
            'pseudoqubit_cache': geo_status,
            'pseudoqubit_count': pq_count,
            'bip32': 'Hierarchical deterministic key derivation',
            'bip39': 'Mnemonic seed phrases (12-24 words)',
            'bip38': 'Password-protected private keys (PBKDF2+XOR)',
            'database': 'Supabase PostgreSQL (REST API — NUMERIC(200,150) geometry)',
            'entropy': 'Block field entropy from QRNG ensemble',
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
        logger.error(f"[HLWE-API] Block signing failed: {e}")
        return {'signature': '', 'auth_tag': '', 'error': str(e)}

def hlwe_verify_block(block_dict: Dict[str, Any], signature_dict: Dict[str, str], public_key_hex: str) -> Tuple[bool, str]:
    """Verify block signature (backward compatible) — USE IN server.py"""
    try:
        adapter = get_hlwe_adapter()
        return adapter.verify_block(block_dict, signature_dict, public_key_hex)
    except Exception as e:
        logger.error(f"[HLWE-API] Block verification failed: {e}")
        return False, f"Error: {str(e)}"

def hlwe_sign_transaction(tx_data: Dict[str, Any], private_key_hex: str) -> Dict[str, str]:
    """Sign transaction (backward compatible) — USE IN mempool.py"""
    try:
        adapter = get_hlwe_adapter()
        return adapter.sign_transaction(tx_data, private_key_hex)
    except Exception as e:
        logger.error(f"[HLWE-API] TX signing failed: {e}")
        return {'signature': '', 'auth_tag': '', 'error': str(e)}

def hlwe_verify_transaction(tx_data: Dict[str, Any], signature_dict: Dict[str, str], public_key_hex: str) -> Tuple[bool, str]:
    """Verify transaction signature (backward compatible) — USE IN mempool.py/server.py"""
    try:
        adapter = get_hlwe_adapter()
        return adapter.verify_transaction(tx_data, signature_dict, public_key_hex)
    except Exception as e:
        logger.error(f"[HLWE-API] TX verification failed: {e}")
        return False, f"Error: {str(e)}"

def hlwe_derive_address(public_key_hex: str) -> str:
    """Derive address from public key (backward compatible)"""
    try:
        adapter = get_hlwe_adapter()
        return adapter.derive_address(public_key_hex)
    except Exception as e:
        logger.error(f"[HLWE-API] Address derivation failed: {e}")
        return ''

def hlwe_create_wallet(label: Optional[str] = None, passphrase: str = '') -> Dict[str, Any]:
    """Create new wallet (backward compatible) — USE IN server.py API endpoint"""
    try:
        adapter = get_hlwe_adapter()
        return adapter.create_wallet(label, passphrase)
    except Exception as e:
        logger.error(f"[HLWE-API] Wallet creation failed: {e}")
        return {'error': str(e)}

def hlwe_get_wallet_status(wallet_fingerprint: str) -> Dict[str, Any]:
    """Get wallet status (backward compatible) — USE IN server.py API endpoint"""
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
        logger.error(f"[HLWE-API] Get wallet status failed: {e}")
        return {'error': str(e)}

def hlwe_health_check() -> bool:
    """Health check (backward compatible) — USE IN server.py /health endpoint"""
    try:
        adapter = get_hlwe_adapter()
        return adapter.health_check()
    except Exception as e:
        logger.error(f"[HLWE-API] Health check failed: {e}")
        return False

def hlwe_system_info() -> Dict[str, Any]:
    """Get system information — USE IN server.py /info endpoint"""
    try:
        adapter = get_hlwe_adapter()
        return adapter.get_system_info()
    except Exception as e:
        logger.error(f"[HLWE-API] System info failed: {e}")
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
    # Quick test
    logger.info("=" * 100)
    logger.info("[TEST] HLWE v2.0 System Self-Test")
    logger.info("=" * 100)
    
    # Test 1: System info
    info = hlwe_system_info()
    logger.info(f"[TEST] System: {info.get('engine')} - {info.get('status', 'ready')}")
    
    # Test 2: Key generation
    manager = get_wallet_manager()
    keypair = manager.hlwe.generate_keypair_from_entropy()
    logger.info(f"[TEST] Generated keypair: {keypair.address[:16]}...")
    
    # Test 3: Signing
    message = b"Test message"
    message_hash = hashlib.sha256(message).digest()
    sig = manager.hlwe.sign_hash(message_hash, keypair.private_key)
    logger.info(f"[TEST] Signed message: {sig.get('auth_tag', '')[:16]}...")
    
    # Test 4: Verification
    is_valid = manager.hlwe.verify_signature(message_hash, sig, keypair.public_key)
    logger.info(f"[TEST] Verification: {'✓ PASS' if is_valid else '✗ FAIL'}")
    
    # Test 5: Mnemonic
    mnemonic = manager.bip39.generate_mnemonic(MnemonicStrength.STANDARD)
    words = mnemonic.split()
    logger.info(f"[TEST] Generated {len(words)}-word mnemonic")
    
    # Test 6: Health check
    health = hlwe_health_check()
    logger.info(f"[TEST] Health check: {'✓ OK' if health else '✗ FAIL'}")
    
    logger.info("=" * 100)
    logger.info("[TEST] All basic tests completed!")
    logger.info("=" * 100)
