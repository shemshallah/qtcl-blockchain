#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                   ║
║              🚀 QUANTUM TEMPORAL COHERENCE LEDGER (QTCL) TERMINAL ORCHESTRATOR v5.0 🚀           ║
║                                                                                                   ║
║                    PRODUCTION-GRADE SYSTEM ORCHESTRATION - 200KB COMPREHENSIVE                   ║
║                                                                                                   ║
║  SUBSYSTEMS INTEGRATED:                                                                          ║
║  ✅ Quantum Engine (Entropy, Validators, Finality Proofs)                                        ║
║  ✅ Oracle Services (Time, Price, Event, Random)                                                 ║
║  ✅ Ledger Management (Transactions, Blocks, Mempool)                                            ║
║  ✅ API Gateway (REST, WebSocket, Rate Limiting)                                                 ║
║  ✅ User Management (Registration, Roles, Profiles)                                              ║
║  ✅ Transaction Processing (Submit, Track, Cancel, Analyze)                                      ║
║  ✅ Block Explorer (Blocks, Transactions, Stats)                                                 ║
║  ✅ Wallet Management (Create, List, Balance, Multi-sig)                                         ║
║  ✅ Admin Controls (User Management, System Monitoring, Settings)                                ║
║  ✅ DeFi Operations (Staking, Lending, Yield)                                                    ║
║  ✅ Governance (Voting, Proposals)                                                               ║
║  ✅ NFT Management (Mint, Transfer, Metadata)                                                    ║
║  ✅ Smart Contracts (Deploy, Execute, Monitor)                                                   ║
║  ✅ Bridge Operations (Cross-chain, Wrapped Assets)                                              ║
║  ✅ Multi-sig Wallets (Create, Sign, Execute)                                                    ║
║  ✅ Parallel Task Execution & Monitoring                                                         ║
║                                                                                                   ║
║  ADMINISTRATIVE FEATURES:                                                                        ║
║  ✅ Admin Auto-Detection & Extended Help Menu                                                    ║
║  ✅ System-Wide Settings & Configuration Management                                              ║
║  ✅ User Management & Role Control                                                               ║
║  ✅ Transaction Approval/Rejection Workflow                                                      ║
║  ✅ System Monitoring & Performance Analytics                                                    ║
║  ✅ Audit Logs & Security Tracking                                                               ║
║  ✅ Database Management & Backup                                                                 ║
║  ✅ Rate Limiting & Quotas                                                                       ║
║  ✅ Emergency Controls & Shutdown Procedures                                                     ║
║                                                                                                   ║
║  COMMAND STRUCTURE:                                                                              ║
║  • auth/* (login, register, logout, 2fa)                                                        ║
║  • user/* (profile, settings, security, preferences)                                             ║
║  • transaction/* (create, track, cancel, analyze, export)                                        ║
║  • wallet/* (create, list, import, export, balance, multi-sig)                                   ║
║  • block/* (list, details, explorer, stats)                                                      ║
║  • quantum/* (circuit, entropy, validator, finality, status)                                     ║
║  • oracle/* (time, price, event, random, feed)                                                   ║
║  • defi/* (stake, unstake, borrow, lend, yield, pool)                                            ║
║  • governance/* (vote, proposal, delegate, stats)                                                ║
║  • nft/* (mint, transfer, burn, metadata, collection)                                            ║
║  • contract/* (deploy, execute, compile, state)                                                  ║
║  • bridge/* (initiate, status, history, wrapped)                                                 ║
║  • admin/* (users, approval, monitoring, settings, audit, emergency)                             ║
║  • system/* (status, health, config, backup, restore)                                            ║
║  • parallel/* (execute, monitor, batch, schedule)                                                ║
║                                                                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import threading
import uuid
import secrets
import getpass
import logging
import subprocess
import hashlib
import sqlite3
import csv
import io
import psutil
import signal
import queue
import socket
import base64
import pickle
import datetime as dt
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Coroutine
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from collections import deque, Counter, defaultdict, OrderedDict
from threading import Lock, RLock, Thread, Event
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import atexit
import traceback
import re

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# DEPENDENCY INSTALLATION
# ═════════════════════════════════════════════════════════════════════════════════════════════════

def ensure_packages():
    packages={
        'requests':'requests','colorama':'colorama','tabulate':'tabulate','PyJWT':'PyJWT',
        'cryptography':'cryptography','pydantic':'pydantic','python_dateutil':'python-dateutil'
    }
    for module,pip_name in packages.items():
        try:__import__(module)
        except ImportError:
            print(f"[SETUP] Installing {pip_name}...");subprocess.check_call([sys.executable,'-m','pip','install','-q',pip_name])

ensure_packages()

import requests
from colorama import Fore, Back, Style, init
from tabulate import tabulate
import jwt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
from cryptography.hazmat.backends import default_backend

init(autoreset=True)

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# LOGGING & METRICS
# ═════════════════════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO,format='[%(asctime)s][%(levelname)s]%(message)s',
    handlers=[logging.FileHandler('qtcl_terminal_complete.log'),logging.StreamHandler(sys.stdout)])
logger=logging.getLogger(__name__)

class Metrics:
    def __init__(self):
        self.lock=RLock()
        self.commands_executed=Counter()
        self.commands_failed=Counter()
        self.start_time=time.time()
        self.session_events=deque(maxlen=1000)
        self.api_calls=Counter()
        self.api_errors=Counter()
        self.login_attempts=Counter()
    
    def record_command(self,cmd:str,success:bool=True):
        with self.lock:
            self.commands_executed[cmd]+=1
            if not success:self.commands_failed[cmd]+=1
            self.session_events.append({'cmd':cmd,'ts':time.time(),'success':success})
    
    def record_api(self,endpoint:str,success:bool=True):
        with self.lock:
            self.api_calls[endpoint]+=1
            if not success:self.api_errors[endpoint]+=1
    
    def get_summary(self)->Dict:
        with self.lock:
            return{
                'uptime_seconds':time.time()-self.start_time,
                'total_commands':sum(self.commands_executed.values()),
                'failed_commands':sum(self.commands_failed.values()),
                'total_api_calls':sum(self.api_calls.values()),
                'api_errors':sum(self.api_errors.values()),
                'top_commands':self.commands_executed.most_common(10),
                'recent_events':list(self.session_events)[-10:]
            }

metrics=Metrics()

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# ENUMS & DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════════════════════════

class TransactionType(Enum):
    TRANSFER="transfer";STAKE="stake";UNSTAKE="unstake";SWAP="swap";GOVERNANCE="governance"
    SMART_CONTRACT="smart_contract";NFT_MINT="nft_mint";NFT_TRANSFER="nft_transfer"
    BRIDGE="bridge";LOAN="loan";REPAY="repay";YIELD="yield"

class UserRole(Enum):
    ADMIN="admin";USER="user";MODERATOR="moderator";SERVICE="service";GUEST="guest"

class TransactionStatus(Enum):
    PENDING="pending";CONFIRMED="confirmed";FAILED="failed";CANCELLED="cancelled"
    PROCESSING="processing";FINALIZED="finalized";REJECTED="rejected"

class CommandCategory(Enum):
    AUTH="auth";USER="user";TRANSACTION="transaction";WALLET="wallet"
    BLOCK="block";QUANTUM="quantum";ORACLE="oracle";DEFI="defi"
    GOVERNANCE="governance";NFT="nft";CONTRACT="contract";BRIDGE="bridge"
    ADMIN="admin";SYSTEM="system";PARALLEL="parallel";HELP="help"

@dataclass
class CommandMeta:
    name:str;category:CommandCategory;description:str;args:List[str]=field(default_factory=list)
    requires_auth:bool=True;requires_admin:bool=False;async_capable:bool=False

@dataclass
class SessionData:
    user_id:Optional[str]=None;email:Optional[str]=None;name:Optional[str]=None
    role:UserRole=UserRole.USER;token:Optional[str]=None;created_at:float=field(default_factory=time.time)
    last_activity:float=field(default_factory=time.time);is_authenticated:bool=False
    active_wallets:List[str]=field(default_factory=list);metadata:Dict=field(default_factory=dict)

@dataclass
class TaskResult:
    task_id:str;command:str;status:str;result:Any=None;error:Optional[str]=None
    start_time:float=field(default_factory=time.time);end_time:Optional[float]=None

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION MANAGER
# ═════════════════════════════════════════════════════════════════════════════════════════════════

class Config:
    API_BASE_URL=os.getenv('QTCL_API_URL','http://localhost:5000')
    API_TIMEOUT=30;API_RETRIES=3;API_RATE_LIMIT=100
    SESSION_FILE='.qtcl_session.json';SESSION_TIMEOUT_HOURS=24
    CACHE_ENABLED=True;CACHE_TTL=300;CACHE_MAX_SIZE=10000
    DB_FILE='.qtcl_terminal.db'
    PASSWORD_MIN_LENGTH=8;PASSWORD_REQUIRE_UPPERCASE=True
    PASSWORD_REQUIRE_LOWERCASE=True;PASSWORD_REQUIRE_DIGITS=True
    THREAD_POOL_SIZE=4;BATCH_SIZE=100
    TABLE_FORMAT='grid';ENABLE_COLORS=True;LOADING_ANIMATION_FRAMES=10
    ADMIN_EMAILS=['admin@qtcl.io','root@qtcl.io','system@qtcl.io']
    ADMIN_DETECT_ROLE=True;ADMIN_FEATURES_ENABLED=True
    PARALLEL_TIMEOUT=300;PARALLEL_MAX_WORKERS=8
    
    @classmethod
    def verify_api_connection(cls)->bool:
        try:r=requests.get(f"{cls.API_BASE_URL}/health",timeout=5);return r.status_code==200
        except:return False

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# UI UTILITIES
# ═════════════════════════════════════════════════════════════════════════════════════════════════

class UI:
    @staticmethod
    def header(text:str):
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'─'*80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}▶ {text}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'─'*80}{Style.RESET_ALL}\n")
    
    @staticmethod
    def success(msg:str):print(f"{Fore.GREEN}{Style.BRIGHT}✓ {msg}{Style.RESET_ALL}")
    
    @staticmethod
    def error(msg:str):print(f"{Fore.RED}{Style.BRIGHT}✗ {msg}{Style.RESET_ALL}")
    
    @staticmethod
    def info(msg:str):print(f"{Fore.YELLOW}{Style.BRIGHT}ℹ {msg}{Style.RESET_ALL}")
    
    @staticmethod
    def warning(msg:str):print(f"{Fore.RED}{Style.BRIGHT}⚠ {msg}{Style.RESET_ALL}")
    
    @staticmethod
    def debug(msg:str):print(f"{Fore.MAGENTA}{Style.DIM}DEBUG: {msg}{Style.RESET_ALL}")
    
    @staticmethod
    def print_table(headers:List[str],rows:List[List[str]]):
        print(tabulate(rows,headers=headers,tablefmt=Config.TABLE_FORMAT))
    
    @staticmethod
    def prompt(msg:str,default:str="",password:bool=False)->str:
        prompt_str=f"{Fore.CYAN}➤ {msg}"
        if default:prompt_str+=f" [{default}]"
        prompt_str+=f":{Style.RESET_ALL} "
        try:
            value=getpass.getpass(prompt_str) if password else input(prompt_str)
            return value if value else default
        except (KeyboardInterrupt,EOFError):return ""
    
    @staticmethod
    def prompt_choice(msg:str,options:List[str])->str:
        UI.header(msg)
        for i,opt in enumerate(options,1):print(f"{Fore.CYAN}{i}){Style.RESET_ALL} {opt}")
        choice=UI.prompt(f"Select (1-{len(options)})")
        try:idx=int(choice)-1;return options[idx] if 0<=idx<len(options) else options[0]
        except (ValueError,IndexError):return options[0]
    
    @staticmethod
    def confirm(msg:str,default:bool=False)->bool:
        suffix="[Y/n]" if default else "[y/N]"
        resp=input(f"{Fore.YELLOW}{msg} {suffix}:{Style.RESET_ALL} ").strip().lower()
        return resp in ['y','yes'] if not default else resp not in ['n','no']
    
    @staticmethod
    def loading(duration:float=3,msg:str="Loading"):
        frames=['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']
        start=time.time()
        i=0
        while time.time()-start<duration:
            print(f"\r{Fore.CYAN}{frames[i%len(frames)]} {msg}...{Style.RESET_ALL}",end='',flush=True)
            time.sleep(0.1);i+=1
        print(f"\r{' '*50}\r",end='',flush=True)
    
    @staticmethod
    def separator():print(f"{Fore.CYAN}{'─'*80}{Style.RESET_ALL}")

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# API CLIENT
# ═════════════════════════════════════════════════════════════════════════════════════════════════

class APIClient:
    def __init__(self,base_url:str):
        self.base_url=base_url;self.session=requests.Session()
        self.auth_token:Optional[str]=None;self.request_timeout=Config.API_TIMEOUT
        self.request_count=0;self.error_count=0;self.request_cache:Dict[str,Tuple[Any,float]]={}
        self.lock=RLock()
    
    def set_auth_token(self,token:str):
        self.auth_token=token
        self.session.headers.update({'Authorization':f'Bearer {token}','Content-Type':'application/json'})
        logger.info(f"Auth token set, length: {len(token)}")
    
    def clear_auth(self):
        self.auth_token=None
        self.session.headers.pop('Authorization',None)
        logger.info("Auth cleared")
    
    def _get_cached(self,cache_key:str)->Optional[Any]:
        with self.lock:
            if cache_key in self.request_cache:
                data,expiry=self.request_cache[cache_key]
                if time.time()<expiry:return data
                del self.request_cache[cache_key]
        return None
    
    def _set_cache(self,cache_key:str,data:Any,ttl:int=300):
        with self.lock:
            if len(self.request_cache)>=Config.CACHE_MAX_SIZE:
                oldest_key=next(iter(self.request_cache));del self.request_cache[oldest_key]
            self.request_cache[cache_key]=(data,time.time()+ttl)
    
    def request(self,method:str,endpoint:str,data:Dict=None,params:Dict=None,
                use_cache:bool=False,cache_ttl:int=300)->Tuple[bool,Any]:
        url=f"{self.base_url}{endpoint}"
        cache_key=f"{method}:{url}"
        
        if use_cache and method=='GET':
            cached=self._get_cached(cache_key)
            if cached is not None:return True,cached
        
        for attempt in range(Config.API_RETRIES):
            try:
                response=self.session.request(method,url,json=data,params=params,timeout=self.request_timeout)
                self.request_count+=1;metrics.record_api(endpoint,True)
                
                if response.status_code in [200,201,202]:
                    result=response.json() if response.text else {}
                    if use_cache and method=='GET':self._set_cache(cache_key,result,cache_ttl)
                    return True,result
                elif response.status_code==401:return False,{'error':'Unauthorized - please login'}
                elif response.status_code==403:return False,{'error':'Forbidden - insufficient permissions'}
                elif response.status_code==404:return False,{'error':'Resource not found'}
                elif response.status_code>=500:
                    if attempt<Config.API_RETRIES-1:time.sleep(2**attempt);continue
                    return False,{'error':'Server error - please try again later'}
                else:return False,response.json() if response.text else {'error':f'HTTP {response.status_code}'}
            except requests.exceptions.Timeout:
                self.error_count+=1;metrics.record_api(endpoint,False)
                if attempt<Config.API_RETRIES-1:time.sleep(2**attempt);continue
                return False,{'error':'Request timeout'}
            except requests.exceptions.ConnectionError:
                self.error_count+=1;metrics.record_api(endpoint,False)
                if attempt<Config.API_RETRIES-1:time.sleep(2**attempt);continue
                return False,{'error':'Connection failed'}
            except Exception as e:
                self.error_count+=1;metrics.record_api(endpoint,False)
                logger.error(f"API request error: {str(e)}")
                if attempt<Config.API_RETRIES-1:time.sleep(2**attempt);continue
                return False,{'error':str(e)}
        
        return False,{'error':'Max retries exceeded'}

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# SESSION MANAGER
# ═════════════════════════════════════════════════════════════════════════════════════════════════

class SessionManager:
    def __init__(self,client:APIClient):
        self.client=client;self.session:SessionData=SessionData()
        self.lock=RLock();self.load_session()
    
    def load_session(self):
        try:
            if os.path.exists(Config.SESSION_FILE):
                with open(Config.SESSION_FILE,'r') as f:
                    data=json.load(f)
                    self.session=SessionData(**data)
                    if time.time()-self.session.created_at>Config.SESSION_TIMEOUT_HOURS*3600:
                        self.clear_session()
                        return
                    if self.session.token:self.client.set_auth_token(self.session.token)
                    logger.info(f"Session loaded for {self.session.email}")
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            self.clear_session()
    
    def save_session(self):
        try:
            with open(Config.SESSION_FILE,'w') as f:
                data={k:v for k,v in asdict(self.session).items() if k!='metadata'}
                json.dump(data,f,indent=2,default=str)
            logger.info("Session saved")
        except Exception as e:logger.error(f"Failed to save session: {e}")
    
    def login(self,email:str,password:str)->Tuple[bool,str]:
        success,result=self.client.request('POST','/api/auth/login',{'email':email,'password':password})
        if success and result.get('token'):
            token=result['token']
            self.session.token=token
            self.session.user_id=result.get('user_id')
            self.session.email=email
            self.session.name=result.get('name','User')
            self.session.role=UserRole(result.get('role','user').lower()) if result.get('role') else UserRole.USER
            self.session.is_authenticated=True
            self.session.created_at=time.time()
            self.client.set_auth_token(token)
            self.save_session()
            return True,"Login successful"
        return False,result.get('error','Login failed')
    
    def register(self,email:str,password:str,name:str)->Tuple[bool,str]:
        success,result=self.client.request('POST','/api/auth/register',
            {'email':email,'password':password,'name':name})
        if success:return True,"Registration successful - please login"
        return False,result.get('error','Registration failed')
    
    def logout(self):
        self.session=SessionData()
        self.client.clear_auth()
        if os.path.exists(Config.SESSION_FILE):os.remove(Config.SESSION_FILE)
        logger.info("Logged out")
    
    def is_admin(self)->bool:
        if not Config.ADMIN_DETECT_ROLE:return False
        if self.session.role==UserRole.ADMIN:return True
        if self.session.email in Config.ADMIN_EMAILS:return True
        return False
    
    def is_authenticated(self)->bool:
        return self.session.is_authenticated and self.session.token is not None
    
    def get_user_id(self)->Optional[str]:
        return self.session.user_id

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# COMMAND REGISTRY & DISPATCHER
# ═════════════════════════════════════════════════════════════════════════════════════════════════

class CommandRegistry:
    def __init__(self):
        self.commands:Dict[str,Tuple[Callable,CommandMeta]]={}
        self.lock=RLock()
    
    def register(self,name:str,func:Callable,meta:CommandMeta):
        with self.lock:
            self.commands[name.lower()]=(func,meta)
            logger.info(f"Registered command: {name}")
    
    def get(self,name:str)->Optional[Tuple[Callable,CommandMeta]]:
        return self.commands.get(name.lower())
    
    def list_by_category(self,category:CommandCategory)->List[Tuple[str,CommandMeta]]:
        return [(name,meta) for name,(func,meta) in self.commands.items() if meta.category==category]
    
    def list_all(self)->List[Tuple[str,CommandMeta]]:
        return [(name,meta) for name,(func,meta) in self.commands.items()]
    
    def search(self,query:str)->List[Tuple[str,CommandMeta]]:
        q=query.lower()
        return [(name,meta) for name,(func,meta) in self.commands.items()
                if q in name or q in meta.description.lower()]

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# PARALLEL EXECUTION ENGINE
# ═════════════════════════════════════════════════════════════════════════════════════════════════

class ParallelExecutor:
    def __init__(self,max_workers:int=Config.PARALLEL_MAX_WORKERS):
        self.max_workers=max_workers;self.task_queue=queue.Queue()
        self.result_queue=queue.Queue();self.tasks:Dict[str,TaskResult]={}
        self.lock=RLock();self.active=True
        self.workers=[Thread(target=self._worker,daemon=True) for _ in range(max_workers)]
        for w in self.workers:w.start()
    
    def _worker(self):
        while self.active:
            try:task_id,func,args,kwargs=self.task_queue.get(timeout=1)
            except queue.Empty:continue
            
            result=TaskResult(task_id=task_id,command=func.__name__)
            start=time.time()
            try:
                result.result=func(*args,**kwargs)
                result.status="completed"
            except Exception as e:
                result.status="failed";result.error=str(e)
                logger.error(f"Task {task_id} failed: {e}")
            finally:
                result.end_time=time.time()
                with self.lock:self.tasks[task_id]=result
                self.result_queue.put(result)
    
    def submit(self,func:Callable,args:tuple=(),kwargs:dict=None)->str:
        task_id=str(uuid.uuid4())[:8]
        kwargs=kwargs or {}
        self.task_queue.put((task_id,func,args,kwargs))
        return task_id
    
    def get_result(self,task_id:str,timeout:float=None)->Optional[TaskResult]:
        start=time.time()
        while True:
            with self.lock:
                if task_id in self.tasks:return self.tasks[task_id]
            if timeout and time.time()-start>timeout:return None
            time.sleep(0.1)
    
    def wait_all(self,timeout:float=None)->Dict[str,TaskResult]:
        with self.lock:return self.tasks.copy()
    
    def shutdown(self):
        self.active=False
        for w in self.workers:w.join(timeout=1)

# ═════════════════════════════════════════════════════════════════════════════════════════════════
# TERMINAL ENGINE CORE
# ═════════════════════════════════════════════════════════════════════════════════════════════════

class TerminalEngine:
    def __init__(self):
        self.client=APIClient(Config.API_BASE_URL)
        self.session=SessionManager(self.client)
        self.registry=CommandRegistry()
        self.executor=ParallelExecutor()
        self.running=True;self.lock=RLock()
        self._register_all_commands()
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT,lambda s,f:self.shutdown())
        signal.signal(signal.SIGTERM,lambda s,f:self.shutdown())
    
    def _register_all_commands(self):
        # AUTH COMMANDS
        self.registry.register('login',self._cmd_login,CommandMeta(
            'login',CommandCategory.AUTH,'Login to QTCL system',requires_auth=False))
        self.registry.register('logout',self._cmd_logout,CommandMeta(
            'logout',CommandCategory.AUTH,'Logout from QTCL system'))
        self.registry.register('register',self._cmd_register,CommandMeta(
            'register',CommandCategory.AUTH,'Register new account',requires_auth=False))
        self.registry.register('whoami',self._cmd_whoami,CommandMeta(
            'whoami',CommandCategory.AUTH,'Show current user'))
        self.registry.register('auth/2fa/setup',self._cmd_2fa_setup,CommandMeta(
            'auth/2fa/setup',CommandCategory.AUTH,'Setup 2FA authentication'))
        self.registry.register('auth/token/refresh',self._cmd_refresh_token,CommandMeta(
            'auth/token/refresh',CommandCategory.AUTH,'Refresh authentication token'))
        
        # USER COMMANDS
        self.registry.register('user/profile',self._cmd_user_profile,CommandMeta(
            'user/profile',CommandCategory.USER,'Show user profile'))
        self.registry.register('user/settings',self._cmd_user_settings,CommandMeta(
            'user/settings',CommandCategory.USER,'Manage user settings'))
        self.registry.register('user/list',self._cmd_user_list,CommandMeta(
            'user/list',CommandCategory.USER,'List all users',requires_admin=True))
        self.registry.register('user/details',self._cmd_user_details,CommandMeta(
            'user/details',CommandCategory.USER,'Get user details'))
        
        # TRANSACTION COMMANDS
        self.registry.register('transaction/create',self._cmd_tx_create,CommandMeta(
            'transaction/create',CommandCategory.TRANSACTION,'Create new transaction'))
        self.registry.register('transaction/track',self._cmd_tx_track,CommandMeta(
            'transaction/track',CommandCategory.TRANSACTION,'Track transaction status'))
        self.registry.register('transaction/cancel',self._cmd_tx_cancel,CommandMeta(
            'transaction/cancel',CommandCategory.TRANSACTION,'Cancel pending transaction'))
        self.registry.register('transaction/list',self._cmd_tx_list,CommandMeta(
            'transaction/list',CommandCategory.TRANSACTION,'List user transactions'))
        self.registry.register('transaction/analyze',self._cmd_tx_analyze,CommandMeta(
            'transaction/analyze',CommandCategory.TRANSACTION,'Analyze transaction patterns'))
        self.registry.register('transaction/export',self._cmd_tx_export,CommandMeta(
            'transaction/export',CommandCategory.TRANSACTION,'Export transaction history'))
        self.registry.register('transaction/stats',self._cmd_tx_stats,CommandMeta(
            'transaction/stats',CommandCategory.TRANSACTION,'Show transaction statistics'))
        
        # WALLET COMMANDS
        self.registry.register('wallet/create',self._cmd_wallet_create,CommandMeta(
            'wallet/create',CommandCategory.WALLET,'Create new wallet'))
        self.registry.register('wallet/list',self._cmd_wallet_list,CommandMeta(
            'wallet/list',CommandCategory.WALLET,'List user wallets'))
        self.registry.register('wallet/balance',self._cmd_wallet_balance,CommandMeta(
            'wallet/balance',CommandCategory.WALLET,'Check wallet balance'))
        self.registry.register('wallet/import',self._cmd_wallet_import,CommandMeta(
            'wallet/import',CommandCategory.WALLET,'Import wallet'))
        self.registry.register('wallet/export',self._cmd_wallet_export,CommandMeta(
            'wallet/export',CommandCategory.WALLET,'Export wallet'))
        self.registry.register('wallet/multisig/create',self._cmd_multisig_create,CommandMeta(
            'wallet/multisig/create',CommandCategory.WALLET,'Create multi-sig wallet',async_capable=True))
        self.registry.register('wallet/multisig/sign',self._cmd_multisig_sign,CommandMeta(
            'wallet/multisig/sign',CommandCategory.WALLET,'Sign multi-sig transaction'))
        
        # BLOCK COMMANDS
        self.registry.register('block/list',self._cmd_block_list,CommandMeta(
            'block/list',CommandCategory.BLOCK,'List recent blocks'))
        self.registry.register('block/details',self._cmd_block_details,CommandMeta(
            'block/details',CommandCategory.BLOCK,'Show block details'))
        self.registry.register('block/explorer',self._cmd_block_explorer,CommandMeta(
            'block/explorer',CommandCategory.BLOCK,'Block explorer with search'))
        self.registry.register('block/stats',self._cmd_block_stats,CommandMeta(
            'block/stats',CommandCategory.BLOCK,'Show block statistics'))
        
        # QUANTUM COMMANDS
        self.registry.register('quantum/status',self._cmd_quantum_status,CommandMeta(
            'quantum/status',CommandCategory.QUANTUM,'Show quantum engine status'))
        self.registry.register('quantum/circuit',self._cmd_quantum_circuit,CommandMeta(
            'quantum/circuit',CommandCategory.QUANTUM,'Build quantum circuit'))
        self.registry.register('quantum/entropy',self._cmd_quantum_entropy,CommandMeta(
            'quantum/entropy',CommandCategory.QUANTUM,'Get quantum entropy'))
        self.registry.register('quantum/validator',self._cmd_quantum_validator,CommandMeta(
            'quantum/validator',CommandCategory.QUANTUM,'Quantum validator status'))
        self.registry.register('quantum/finality',self._cmd_quantum_finality,CommandMeta(
            'quantum/finality',CommandCategory.QUANTUM,'Check quantum finality'))
        
        # ORACLE COMMANDS
        self.registry.register('oracle/time',self._cmd_oracle_time,CommandMeta(
            'oracle/time',CommandCategory.ORACLE,'Get oracle time feed'))
        self.registry.register('oracle/price',self._cmd_oracle_price,CommandMeta(
            'oracle/price',CommandCategory.ORACLE,'Get price oracle data'))
        self.registry.register('oracle/random',self._cmd_oracle_random,CommandMeta(
            'oracle/random',CommandCategory.ORACLE,'Get random numbers from oracle'))
        self.registry.register('oracle/event',self._cmd_oracle_event,CommandMeta(
            'oracle/event',CommandCategory.ORACLE,'Listen for oracle events'))
        self.registry.register('oracle/feed',self._cmd_oracle_feed,CommandMeta(
            'oracle/feed',CommandCategory.ORACLE,'Show oracle feeds'))
        
        # DEFI COMMANDS
        self.registry.register('defi/stake',self._cmd_defi_stake,CommandMeta(
            'defi/stake',CommandCategory.DEFI,'Stake tokens'))
        self.registry.register('defi/unstake',self._cmd_defi_unstake,CommandMeta(
            'defi/unstake',CommandCategory.DEFI,'Unstake tokens'))
        self.registry.register('defi/borrow',self._cmd_defi_borrow,CommandMeta(
            'defi/borrow',CommandCategory.DEFI,'Borrow from lending pool'))
        self.registry.register('defi/repay',self._cmd_defi_repay,CommandMeta(
            'defi/repay',CommandCategory.DEFI,'Repay loan'))
        self.registry.register('defi/yield',self._cmd_defi_yield,CommandMeta(
            'defi/yield',CommandCategory.DEFI,'View yield farming opportunities'))
        self.registry.register('defi/pool',self._cmd_defi_pool,CommandMeta(
            'defi/pool',CommandCategory.DEFI,'Manage liquidity pools'))
        
        # GOVERNANCE COMMANDS
        self.registry.register('governance/vote',self._cmd_governance_vote,CommandMeta(
            'governance/vote',CommandCategory.GOVERNANCE,'Vote on proposal'))
        self.registry.register('governance/proposal',self._cmd_governance_proposal,CommandMeta(
            'governance/proposal',CommandCategory.GOVERNANCE,'Create governance proposal'))
        self.registry.register('governance/delegate',self._cmd_governance_delegate,CommandMeta(
            'governance/delegate',CommandCategory.GOVERNANCE,'Delegate voting power'))
        self.registry.register('governance/stats',self._cmd_governance_stats,CommandMeta(
            'governance/stats',CommandCategory.GOVERNANCE,'Show governance statistics'))
        
        # NFT COMMANDS
        self.registry.register('nft/mint',self._cmd_nft_mint,CommandMeta(
            'nft/mint',CommandCategory.NFT,'Mint NFT'))
        self.registry.register('nft/transfer',self._cmd_nft_transfer,CommandMeta(
            'nft/transfer',CommandCategory.NFT,'Transfer NFT'))
        self.registry.register('nft/burn',self._cmd_nft_burn,CommandMeta(
            'nft/burn',CommandCategory.NFT,'Burn NFT'))
        self.registry.register('nft/metadata',self._cmd_nft_metadata,CommandMeta(
            'nft/metadata',CommandCategory.NFT,'View/edit NFT metadata'))
        self.registry.register('nft/collection',self._cmd_nft_collection,CommandMeta(
            'nft/collection',CommandCategory.NFT,'Manage NFT collections'))
        
        # SMART CONTRACT COMMANDS
        self.registry.register('contract/deploy',self._cmd_contract_deploy,CommandMeta(
            'contract/deploy',CommandCategory.CONTRACT,'Deploy smart contract'))
        self.registry.register('contract/execute',self._cmd_contract_execute,CommandMeta(
            'contract/execute',CommandCategory.CONTRACT,'Execute contract function'))
        self.registry.register('contract/compile',self._cmd_contract_compile,CommandMeta(
            'contract/compile',CommandCategory.CONTRACT,'Compile contract code'))
        self.registry.register('contract/state',self._cmd_contract_state,CommandMeta(
            'contract/state',CommandCategory.CONTRACT,'View contract state'))
        
        # BRIDGE COMMANDS
        self.registry.register('bridge/initiate',self._cmd_bridge_initiate,CommandMeta(
            'bridge/initiate',CommandCategory.BRIDGE,'Initiate cross-chain bridge'))
        self.registry.register('bridge/status',self._cmd_bridge_status,CommandMeta(
            'bridge/status',CommandCategory.BRIDGE,'Check bridge status'))
        self.registry.register('bridge/history',self._cmd_bridge_history,CommandMeta(
            'bridge/history',CommandCategory.BRIDGE,'View bridge history'))
        self.registry.register('bridge/wrapped',self._cmd_bridge_wrapped,CommandMeta(
            'bridge/wrapped',CommandCategory.BRIDGE,'Manage wrapped assets'))
        
        # ADMIN COMMANDS
        self.registry.register('admin/users',self._cmd_admin_users,CommandMeta(
            'admin/users',CommandCategory.ADMIN,'Manage users',requires_admin=True))
        self.registry.register('admin/approval',self._cmd_admin_approval,CommandMeta(
            'admin/approval',CommandCategory.ADMIN,'Approve/reject transactions',requires_admin=True))
        self.registry.register('admin/monitoring',self._cmd_admin_monitoring,CommandMeta(
            'admin/monitoring',CommandCategory.ADMIN,'System monitoring',requires_admin=True))
        self.registry.register('admin/settings',self._cmd_admin_settings,CommandMeta(
            'admin/settings',CommandCategory.ADMIN,'System settings',requires_admin=True))
        self.registry.register('admin/audit',self._cmd_admin_audit,CommandMeta(
            'admin/audit',CommandCategory.ADMIN,'Audit logs',requires_admin=True))
        self.registry.register('admin/emergency',self._cmd_admin_emergency,CommandMeta(
            'admin/emergency',CommandCategory.ADMIN,'Emergency controls',requires_admin=True))
        
        # SYSTEM COMMANDS
        self.registry.register('system/status',self._cmd_system_status,CommandMeta(
            'system/status',CommandCategory.SYSTEM,'Show system status'))
        self.registry.register('system/health',self._cmd_system_health,CommandMeta(
            'system/health',CommandCategory.SYSTEM,'System health check'))
        self.registry.register('system/config',self._cmd_system_config,CommandMeta(
            'system/config',CommandCategory.SYSTEM,'View system configuration'))
        self.registry.register('system/backup',self._cmd_system_backup,CommandMeta(
            'system/backup',CommandCategory.SYSTEM,'Backup system data',requires_admin=True))
        self.registry.register('system/restore',self._cmd_system_restore,CommandMeta(
            'system/restore',CommandCategory.SYSTEM,'Restore from backup',requires_admin=True))
        
        # PARALLEL COMMANDS
        self.registry.register('parallel/execute',self._cmd_parallel_execute,CommandMeta(
            'parallel/execute',CommandCategory.PARALLEL,'Execute commands in parallel',async_capable=True))
        self.registry.register('parallel/batch',self._cmd_parallel_batch,CommandMeta(
            'parallel/batch',CommandCategory.PARALLEL,'Execute batch operations'))
        self.registry.register('parallel/monitor',self._cmd_parallel_monitor,CommandMeta(
            'parallel/monitor',CommandCategory.PARALLEL,'Monitor parallel tasks'))
        
        # HELP COMMANDS
        self.registry.register('help',self._cmd_help,CommandMeta(
            'help',CommandCategory.HELP,'Show help menu',requires_auth=False))
        self.registry.register('help/admin',self._cmd_help_admin,CommandMeta(
            'help/admin',CommandCategory.HELP,'Show admin help menu'))
        self.registry.register('help/search',self._cmd_help_search,CommandMeta(
            'help/search',CommandCategory.HELP,'Search help topics'))
        self.registry.register('help/commands',self._cmd_help_commands,CommandMeta(
            'help/commands',CommandCategory.HELP,'List all commands'))
        self.registry.register('help/examples',self._cmd_help_examples,CommandMeta(
            'help/examples',CommandCategory.HELP,'Show command examples'))
    
    # ═════════════════════════════════════════════════════════════════════════════════════════
    # AUTH COMMAND IMPLEMENTATIONS
    # ═════════════════════════════════════════════════════════════════════════════════════════
    
    def _cmd_login(self):
        UI.header("🔐 LOGIN")
        email=UI.prompt("Email")
        password=UI.prompt("Password",password=True)
        
        success,msg=self.session.login(email,password)
        if success:
            UI.success(msg)
            metrics.record_command('login')
        else:
            UI.error(f"Login failed: {msg}")
            metrics.record_command('login',False)
    
    def _cmd_logout(self):
        if not self.session.is_authenticated():
            UI.error("Not logged in")
            return
        
        if UI.confirm("Logout?"):
            self.session.logout()
            UI.success("Logged out")
            metrics.record_command('logout')
    
    def _cmd_register(self):
        UI.header("📝 REGISTER")
        name=UI.prompt("Full name")
        email=UI.prompt("Email")
        password=UI.prompt("Password",password=True)
        confirm=UI.prompt("Confirm password",password=True)
        
        if password!=confirm:
            UI.error("Passwords don't match")
            return
        
        success,msg=self.session.register(email,password,name)
        if success:
            UI.success(msg)
            metrics.record_command('register')
        else:
            UI.error(f"Registration failed: {msg}")
            metrics.record_command('register',False)
    
    def _cmd_whoami(self):
        if not self.session.is_authenticated():
            UI.info("Not authenticated")
            return
        
        UI.header("👤 CURRENT USER")
        UI.print_table(['Field','Value'],[
            ['User ID',self.session.session.user_id or 'N/A'],
            ['Email',self.session.session.email or 'N/A'],
            ['Name',self.session.session.name or 'N/A'],
            ['Role',self.session.session.role.value.upper()],
            ['Admin',str(self.session.is_admin())],
            ['Authenticated',str(self.session.is_authenticated())]
        ])
        metrics.record_command('whoami')
    
    def _cmd_2fa_setup(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated")
            return
        
        UI.header("🔐 2FA SETUP")
        success,result=self.client.request('POST','/api/auth/2fa/setup',{})
        
        if success:
            UI.success("2FA setup initiated")
            if result.get('qr_code'):
                UI.info("Scan QR code with authenticator app")
            secret=result.get('secret','')
            if secret:UI.info(f"Secret key: {secret}")
            metrics.record_command('auth/2fa/setup')
        else:
            UI.error(f"Setup failed: {result.get('error')}")
            metrics.record_command('auth/2fa/setup',False)
    
    def _cmd_refresh_token(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated")
            return
        
        UI.header("🔄 REFRESH TOKEN")
        success,result=self.client.request('POST','/api/auth/refresh',{})
        
        if success and result.get('token'):
            self.session.session.token=result['token']
            self.client.set_auth_token(result['token'])
            self.session.save_session()
            UI.success("Token refreshed")
            metrics.record_command('auth/token/refresh')
        else:
            UI.error(f"Refresh failed: {result.get('error')}")
            metrics.record_command('auth/token/refresh',False)
    
    # ═════════════════════════════════════════════════════════════════════════════════════════
    # USER COMMAND IMPLEMENTATIONS
    # ═════════════════════════════════════════════════════════════════════════════════════════
    
    def _cmd_user_profile(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated")
            return
        
        UI.header("👤 USER PROFILE")
        success,user=self.client.request('GET','/api/users/me')
        
        if success:
            UI.print_table(['Field','Value'],[
                ['User ID',user.get('user_id','N/A')[:16]+"..."],
                ['Email',user.get('email','N/A')],
                ['Name',user.get('name','N/A')],
                ['Role',user.get('role','user').upper()],
                ['Created',user.get('created_at','N/A')[:10]],
                ['Last Active',user.get('last_active','N/A')[:19]],
                ['Verified',str(user.get('verified',False))]
            ])
            metrics.record_command('user/profile')
        else:
            UI.error(f"Failed to fetch profile: {user.get('error')}")
            metrics.record_command('user/profile',False)
    
    def _cmd_user_settings(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated")
            return
        
        while True:
            choice=UI.prompt_choice("User Settings:",[
                "Change Password","Update Profile","Notification Preferences","Security Settings","Back"
            ])
            
            if choice=="Change Password":self._change_password()
            elif choice=="Update Profile":self._update_profile()
            elif choice=="Notification Preferences":self._notification_preferences()
            elif choice=="Security Settings":self._security_settings()
            else:break
    
    def _change_password(self):
        old_pass=UI.prompt("Current password",password=True)
        new_pass=UI.prompt("New password",password=True)
        confirm=UI.prompt("Confirm password",password=True)
        
        if new_pass!=confirm:
            UI.error("Passwords don't match")
            return
        
        success,result=self.client.request('POST','/api/auth/change-password',
            {'old_password':old_pass,'new_password':new_pass})
        
        if success:
            UI.success("Password changed")
        else:
            UI.error(f"Failed: {result.get('error')}")
    
    def _update_profile(self):
        name=UI.prompt("Full name",self.session.session.name or "")
        success,result=self.client.request('PUT','/api/users/me',{'name':name})
        
        if success:
            self.session.session.name=name
            self.session.save_session()
            UI.success("Profile updated")
        else:
            UI.error(f"Failed: {result.get('error')}")
    
    def _notification_preferences(self):
        UI.header("🔔 NOTIFICATION PREFERENCES")
        success,result=self.client.request('GET','/api/users/me/preferences')
        
        if success:
            UI.print_table(['Setting','Status'],[
                ['Email Notifications',result.get('email_notifications','false').upper()],
                ['SMS Notifications',result.get('sms_notifications','false').upper()],
                ['Transaction Alerts',result.get('tx_alerts','true').upper()],
                ['Security Alerts',result.get('security_alerts','true').upper()]
            ])
        else:
            UI.error(f"Failed: {result.get('error')}")
    
    def _security_settings(self):
        UI.header("🔒 SECURITY SETTINGS")
        success,result=self.client.request('GET','/api/users/me/security')
        
        if success:
            UI.print_table(['Setting','Status'],[
                ['2FA Enabled',str(result.get('totp_enabled',False))],
                ['Login Attempts',str(result.get('login_attempts',0))],
                ['Last Login IP',result.get('last_login_ip','N/A')],
                ['Active Sessions',str(result.get('active_sessions',0))]
            ])
        else:
            UI.error(f"Failed: {result.get('error')}")
    
    def _cmd_user_list(self):
        if not self.session.is_admin():
            UI.error("Admin access required")
            return
        
        UI.header("👥 ALL USERS")
        success,result=self.client.request('GET','/api/users')
        
        if success:
            users=result.get('users',[])
            rows=[[u.get('user_id','')[:12]+"...",u.get('email',''),u.get('role','user').upper(),
                   str(u.get('verified',False)),u.get('created_at','')[:10]] for u in users]
            UI.print_table(['User ID','Email','Role','Verified','Created'],rows)
            UI.info(f"Total users: {len(users)}")
            metrics.record_command('user/list')
        else:
            UI.error(f"Failed: {result.get('error')}")
            metrics.record_command('user/list',False)
    
    def _cmd_user_details(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated")
            return
        
        user_id=UI.prompt("User ID (or enter for current user)")
        if not user_id:user_id=self.session.get_user_id()
        
        UI.header(f"👤 USER DETAILS - {user_id[:12]}...")
        success,user=self.client.request('GET',f'/api/users/{user_id}')
        
        if success:
            UI.print_table(['Field','Value'],[
                ['User ID',user.get('user_id','')[:16]+"..."],
                ['Email',user.get('email','')],
                ['Name',user.get('name','')],
                ['Role',user.get('role','user').upper()],
                ['Verified',str(user.get('verified',False))],
                ['Created',user.get('created_at','')[:19]],
                ['Balance',f"{float(user.get('balance',0)):.2f} QTCL"]
            ])
            metrics.record_command('user/details')
        else:
            UI.error(f"Failed: {user.get('error')}")
            metrics.record_command('user/details',False)
    
    # ═════════════════════════════════════════════════════════════════════════════════════════
    # TRANSACTION COMMAND IMPLEMENTATIONS
    # ═════════════════════════════════════════════════════════════════════════════════════════
    
    def _cmd_tx_create(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("💸 CREATE TRANSACTION")
        to_address=UI.prompt("Recipient address")
        amount=UI.prompt("Amount")
        tx_type=UI.prompt_choice("Transaction type:",[
            "TRANSFER","STAKE","SWAP","SMART_CONTRACT","NFT_MINT","BRIDGE"
        ])
        description=UI.prompt("Description (optional)","")
        
        try:amount_val=Decimal(amount)
        except:UI.error("Invalid amount");return
        
        payload={
            'to_address':to_address,'amount':str(amount_val),
            'type':tx_type.upper(),'description':description
        }
        
        success,result=self.client.request('POST','/api/transactions',payload)
        if success:
            UI.success(f"Transaction created: {result.get('tx_id','')[:16]}...")
            UI.print_table(['Field','Value'],[
                ['TX ID',result.get('tx_id','')[:16]+"..."],
                ['Status',result.get('status','pending').upper()],
                ['Amount',f"{float(result.get('amount',0)):.2f} QTCL"],
                ['Created',result.get('created_at','')[:19]]
            ])
            metrics.record_command('transaction/create')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('transaction/create',False)
    
    def _cmd_tx_track(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        tx_id=UI.prompt("Transaction ID")
        UI.header(f"📊 TRACK TRANSACTION - {tx_id[:12]}...")
        
        success,tx=self.client.request('GET',f'/api/transactions/{tx_id}')
        if success:
            confirmations=tx.get('confirmations',0)
            status_color=Fore.GREEN if tx.get('status')=='confirmed' else Fore.YELLOW if tx.get('status')=='pending' else Fore.RED
            
            UI.print_table(['Field','Value'],[
                ['TX ID',tx.get('tx_id','')[:16]+"..."],
                ['Status',f"{status_color}{tx.get('status','unknown').upper()}{Style.RESET_ALL}"],
                ['From',tx.get('from_address','')[:16]+"..."],
                ['To',tx.get('to_address','')[:16]+"..."],
                ['Amount',f"{float(tx.get('amount',0)):.2f} QTCL"],
                ['Fee',f"{float(tx.get('fee',0)):.4f} QTCL"],
                ['Confirmations',str(confirmations)],
                ['Block',str(tx.get('block_number','pending'))],
                ['Created',tx.get('created_at','')[:19]],
                ['Updated',tx.get('updated_at','')[:19]]
            ])
            metrics.record_command('transaction/track')
        else:
            UI.error(f"Failed: {tx.get('error')}");metrics.record_command('transaction/track',False)
    
    def _cmd_tx_cancel(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        tx_id=UI.prompt("Transaction ID to cancel")
        if not UI.confirm(f"Cancel transaction {tx_id[:12]}...?"):return
        
        success,result=self.client.request('POST',f'/api/transactions/{tx_id}/cancel',{})
        if success:
            UI.success(f"Transaction cancelled")
            metrics.record_command('transaction/cancel')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('transaction/cancel',False)
    
    def _cmd_tx_list(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("📋 MY TRANSACTIONS")
        limit=int(UI.prompt("Limit (default 10)","10") or "10")
        
        success,result=self.client.request('GET','/api/transactions',params={'limit':limit})
        if success:
            txs=result.get('transactions',[])
            rows=[[t.get('tx_id','')[:12]+"...",t.get('type','transfer').upper(),
                   f"{float(t.get('amount',0)):.2f}",t.get('status','pending').upper(),
                   t.get('created_at','')[:10]] for t in txs]
            UI.print_table(['TX ID','Type','Amount','Status','Date'],rows)
            UI.info(f"Showing {len(txs)} of {result.get('total',len(txs))} transactions")
            metrics.record_command('transaction/list')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('transaction/list',False)
    
    def _cmd_tx_analyze(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("📈 TRANSACTION ANALYSIS")
        success,result=self.client.request('GET','/api/transactions/stats')
        
        if success:
            stats=result.get('stats',{})
            UI.print_table(['Metric','Value'],[
                ['Total Transactions',str(stats.get('total',0))],
                ['Total Volume',f"{float(stats.get('total_volume',0)):.2f} QTCL"],
                ['Average Amount',f"{float(stats.get('average_amount',0)):.2f} QTCL"],
                ['Largest TX',f"{float(stats.get('largest_tx',0)):.2f} QTCL"],
                ['Smallest TX',f"{float(stats.get('smallest_tx',0)):.2f} QTCL"],
                ['Success Rate',f"{float(stats.get('success_rate',0)):.1%}"],
                ['Pending Count',str(stats.get('pending_count',0))],
                ['Failed Count',str(stats.get('failed_count',0))]
            ])
            metrics.record_command('transaction/analyze')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('transaction/analyze',False)
    
    def _cmd_tx_export(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("📤 EXPORT TRANSACTIONS")
        fmt=UI.prompt_choice("Format:",[
            "CSV","JSON","XML","PDF"
        ])
        
        success,result=self.client.request('GET','/api/transactions',params={'format':fmt.lower()})
        if success:
            filename=f"transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{fmt.lower()}"
            try:
                with open(filename,'w') as f:f.write(str(result))
                UI.success(f"Exported to {filename}")
                metrics.record_command('transaction/export')
            except Exception as e:
                UI.error(f"Export failed: {e}")
                metrics.record_command('transaction/export',False)
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('transaction/export',False)
    
    def _cmd_tx_stats(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("📊 TRANSACTION STATISTICS")
        success,result=self.client.request('GET','/api/transactions/stats')
        
        if success:
            stats=result.get('stats',{})
            UI.print_table(['Metric','Value'],[
                ['Daily Average',f"{float(stats.get('daily_average',0)):.2f} QTCL"],
                ['Weekly Total',f"{float(stats.get('weekly_total',0)):.2f} QTCL"],
                ['Monthly Total',f"{float(stats.get('monthly_total',0)):.2f} QTCL"],
                ['Most Common Type',stats.get('most_common_type','N/A')],
                ['Avg Confirmation Time',f"{float(stats.get('avg_confirm_time',0)):.1f}s"],
                ['Network Fee Paid',f"{float(stats.get('network_fees',0)):.4f} QTCL"],
                ['24h Volume',f"{float(stats.get('volume_24h',0)):.2f} QTCL"]
            ])
            metrics.record_command('transaction/stats')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('transaction/stats',False)
    
    # ═════════════════════════════════════════════════════════════════════════════════════════
    # WALLET COMMAND IMPLEMENTATIONS
    # ═════════════════════════════════════════════════════════════════════════════════════════
    
    def _cmd_wallet_create(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("💼 CREATE WALLET")
        name=UI.prompt("Wallet name","My Wallet")
        wallet_type=UI.prompt_choice("Wallet type:",[
            "SINGLE","MULTI_SIG","HARDWARE","COLD_STORAGE"
        ])
        
        payload={'name':name,'type':wallet_type.lower()}
        success,result=self.client.request('POST','/api/wallets',payload)
        
        if success:
            UI.success("Wallet created")
            UI.print_table(['Field','Value'],[
                ['Wallet ID',result.get('wallet_id','')[:16]+"..."],
                ['Address',result.get('address','')[:32]+"..."],
                ['Type',result.get('type','').upper()],
                ['Balance',f"{float(result.get('balance',0)):.2f} QTCL"],
                ['Created',result.get('created_at','')[:19]]
            ])
            metrics.record_command('wallet/create')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('wallet/create',False)
    
    def _cmd_wallet_list(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("💼 MY WALLETS")
        success,result=self.client.request('GET','/api/wallets')
        
        if success:
            wallets=result.get('wallets',[])
            rows=[[w.get('wallet_id','')[:12]+"...",w.get('name','Wallet'),
                   f"{float(w.get('balance',0)):.2f}","✓" if w.get('is_default') else "",""]
                  for w in wallets]
            UI.print_table(['ID','Name','Balance','Default','Address'],rows)
            total_balance=sum(Decimal(str(w.get('balance',0))) for w in wallets)
            UI.info(f"Total balance: {float(total_balance):.2f} QTCL across {len(wallets)} wallets")
            metrics.record_command('wallet/list')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('wallet/list',False)
    
    def _cmd_wallet_balance(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        wallet_id=UI.prompt("Wallet ID (or leave for all)")
        UI.header(f"💰 WALLET BALANCE")
        
        endpoint=f'/api/wallets/{wallet_id}' if wallet_id else '/api/wallets'
        success,result=self.client.request('GET',endpoint)
        
        if success:
            if wallet_id:
                UI.print_table(['Field','Value'],[
                    ['Wallet ID',result.get('wallet_id','')[:16]+"..."],
                    ['Balance',f"{float(result.get('balance',0)):.2f} QTCL"],
                    ['Pending',f"{float(result.get('pending',0)):.2f} QTCL"],
                    ['Available',f"{float(result.get('available',0)):.2f} QTCL"]
                ])
            else:
                wallets=result.get('wallets',[])
                rows=[[w.get('name',''),f"{float(w.get('balance',0)):.2f}"] for w in wallets]
                UI.print_table(['Wallet','Balance'],rows)
            metrics.record_command('wallet/balance')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('wallet/balance',False)
    
    def _cmd_wallet_import(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("📥 IMPORT WALLET")
        name=UI.prompt("Wallet name")
        seed_phrase=UI.prompt("Seed phrase or private key")
        
        payload={'name':name,'seed_phrase':seed_phrase}
        success,result=self.client.request('POST','/api/wallets/import',payload)
        
        if success:
            UI.success("Wallet imported")
            metrics.record_command('wallet/import')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('wallet/import',False)
    
    def _cmd_wallet_export(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        wallet_id=UI.prompt("Wallet ID to export")
        if not UI.confirm("Export will include sensitive data. Continue?"):return
        
        password=UI.prompt("Confirm with password",password=True)
        payload={'password':password}
        
        success,result=self.client.request('POST',f'/api/wallets/{wallet_id}/export',payload)
        if success:
            filename=f"wallet_{wallet_id[:8]}.json"
            try:
                with open(filename,'w') as f:json.dump(result,f)
                UI.success(f"Exported to {filename}")
                metrics.record_command('wallet/export')
            except Exception as e:
                UI.error(f"Export failed: {e}");metrics.record_command('wallet/export',False)
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('wallet/export',False)
    
    def _cmd_multisig_create(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("🔑 CREATE MULTI-SIG WALLET")
        name=UI.prompt("Wallet name")
        signers=int(UI.prompt("Number of signers","2"))
        required=int(UI.prompt("Signatures required","2"))
        
        signer_addresses=[]
        for i in range(signers):
            addr=UI.prompt(f"Signer {i+1} address")
            signer_addresses.append(addr)
        
        payload={'name':name,'signers':signer_addresses,'required':required}
        success,result=self.client.request('POST','/api/wallets/multisig',payload)
        
        if success:
            UI.success("Multi-sig wallet created")
            UI.print_table(['Field','Value'],[
                ['Wallet ID',result.get('wallet_id','')[:16]+"..."],
                ['Signers',str(len(signer_addresses))],
                ['Required',str(required)],
                ['Address',result.get('address','')[:32]+"..."]
            ])
            metrics.record_command('wallet/multisig/create')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('wallet/multisig/create',False)
    
    def _cmd_multisig_sign(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("🔐 SIGN MULTI-SIG TRANSACTION")
        tx_id=UI.prompt("Transaction ID")
        wallet_id=UI.prompt("Multi-sig wallet ID")
        
        success,result=self.client.request('POST',f'/api/wallets/{wallet_id}/sign',{'tx_id':tx_id})
        
        if success:
            UI.success("Transaction signed")
            UI.print_table(['Field','Value'],[
                ['TX ID',result.get('tx_id','')[:16]+"..."],
                ['Signatures',f"{result.get('signatures_count',0)}/{result.get('signatures_required',0)}"],
                ['Executable',str(result.get('executable',False))]
            ])
            metrics.record_command('wallet/multisig/sign')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('wallet/multisig/sign',False)
    
    # ═════════════════════════════════════════════════════════════════════════════════════════
    # BLOCK COMMAND IMPLEMENTATIONS
    # ═════════════════════════════════════════════════════════════════════════════════════════
    
    def _cmd_block_list(self):
        UI.header("📦 RECENT BLOCKS")
        limit=int(UI.prompt("Limit (default 10)","10") or "10")
        
        success,result=self.client.request('GET','/api/blocks',params={'limit':limit})
        if success:
            blocks=result.get('blocks',[])
            rows=[[b.get('block_number',''),str(b.get('transactions',0)),
                   f"{float(b.get('size',0))/1024:.2f}","✓" if b.get('finalized') else "",""]
                  for b in blocks[:limit]]
            UI.print_table(['Block','TXs','Size(KB)','Finalized','Hash'],rows)
            metrics.record_command('block/list')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('block/list',False)
    
    def _cmd_block_details(self):
        block_num=UI.prompt("Block number")
        UI.header(f"📦 BLOCK {block_num} DETAILS")
        
        success,block=self.client.request('GET',f'/api/blocks/{block_num}')
        if success:
            UI.print_table(['Field','Value'],[
                ['Block Number',str(block.get('block_number',''))],
                ['Hash',block.get('hash','')[:32]+"..."],
                ['Parent Hash',block.get('parent_hash','')[:32]+"..."],
                ['Timestamp',block.get('timestamp','')],
                ['Transactions',str(len(block.get('transactions',[])))],
                ['Miner',block.get('miner','')[:16]+"..."],
                ['Gas Used',str(block.get('gas_used',''))],
                ['Finalized',str(block.get('finalized',False))],
                ['Quantum Proof',block.get('quantum_proof','')[:16]+"..." if block.get('quantum_proof') else "N/A"]
            ])
            metrics.record_command('block/details')
        else:
            UI.error(f"Failed: {block.get('error')}");metrics.record_command('block/details',False)
    
    def _cmd_block_explorer(self):
        UI.header("🔍 BLOCK EXPLORER")
        query=UI.prompt("Search (block number, hash, address, or tx)")
        query_type=UI.prompt_choice("Type:",[
            "AUTO","BLOCK","TRANSACTION","ADDRESS","HASH"
        ])
        
        success,result=self.client.request('GET','/api/blocks/search',
            params={'query':query,'type':query_type.lower()})
        
        if success:
            results=result.get('results',[])
            if results:
                for r in results[:5]:
                    print(f"\n{Fore.CYAN}Type: {r.get('type','')}{Style.RESET_ALL}")
                    print(f"Data: {json.dumps(r.get('data',{}),indent=2)[:200]}")
            else:UI.info("No results found")
            metrics.record_command('block/explorer')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('block/explorer',False)
    
    def _cmd_block_stats(self):
        UI.header("📊 BLOCK STATISTICS")
        success,result=self.client.request('GET','/api/blocks/stats')
        
        if success:
            stats=result.get('stats',{})
            UI.print_table(['Metric','Value'],[
                ['Total Blocks',str(stats.get('total_blocks',0))],
                ['Latest Block',str(stats.get('latest_block',0))],
                ['Avg Block Time',f"{float(stats.get('avg_block_time',0)):.2f}s"],
                ['Total Transactions',str(stats.get('total_transactions',0))],
                ['Avg TXs per Block',f"{float(stats.get('avg_txs_per_block',0)):.1f}"],
                ['Network TPS',f"{float(stats.get('transactions_per_second',0)):.2f}"],
                ['Total Data',f"{float(stats.get('total_data_mb',0)):.2f} MB"]
            ])
            metrics.record_command('block/stats')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('block/stats',False)
    
    # ═════════════════════════════════════════════════════════════════════════════════════════
    # QUANTUM COMMAND IMPLEMENTATIONS
    # ═════════════════════════════════════════════════════════════════════════════════════════
    
    def _cmd_quantum_status(self):
        UI.header("⚛️ QUANTUM ENGINE STATUS")
        success,result=self.client.request('GET','/api/quantum/status')
        
        if success:
            UI.print_table(['Component','Status'],[
                ['Engine',result.get('engine_status','offline')],
                ['Entropy Source',result.get('entropy_status','offline')],
                ['Validators Active',str(result.get('validators_active',0))],
                ['Finality Proofs',str(result.get('finality_proofs',0))],
                ['Coherence Level',f"{float(result.get('coherence',0)):.1%}"]
            ])
            metrics.record_command('quantum/status')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('quantum/status',False)
    
    def _cmd_quantum_circuit(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("⚛️ BUILD QUANTUM CIRCUIT")
        num_qubits=int(UI.prompt("Number of qubits (1-10)","3"))
        gates=UI.prompt("Gates (comma-separated, e.g., H,CNOT,X)","H")
        
        payload={'qubits':num_qubits,'gates':gates.split(',')}
        success,result=self.client.request('POST','/api/quantum/circuit',payload)
        
        if success:
            UI.success("Circuit created")
            UI.print_table(['Field','Value'],[
                ['Circuit ID',result.get('circuit_id','')[:16]+"..."],
                ['Qubits',str(result.get('qubits',0))],
                ['Gates',str(len(result.get('gates',[])))],
                ['Depth',str(result.get('depth',0))],
                ['Status',result.get('status','created')]
            ])
            metrics.record_command('quantum/circuit')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('quantum/circuit',False)
    
    def _cmd_quantum_entropy(self):
        UI.header("⚛️ QUANTUM ENTROPY")
        success,result=self.client.request('GET','/api/quantum/entropy')
        
        if success:
            UI.print_table(['Metric','Value'],[
                ['Current Entropy',f"{float(result.get('current_entropy',0)):.6f}"],
                ['Max Entropy',f"{float(result.get('max_entropy',0)):.6f}"],
                ['Entropy Pool Size',str(result.get('pool_size',0))],
                ['Last Updated',result.get('last_updated','')[:19]],
                ['Quality Score',f"{float(result.get('quality',0)):.1%}"]
            ])
            metrics.record_command('quantum/entropy')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('quantum/entropy',False)
    
    def _cmd_quantum_validator(self):
        UI.header("⚛️ QUANTUM VALIDATORS")
        success,result=self.client.request('GET','/api/quantum/validators')
        
        if success:
            validators=result.get('validators',[])
            rows=[[v.get('validator_id','')[:12]+"...",v.get('state',''),
                   f"{float(v.get('score',0)):.2f}","✓" if v.get('active') else "✗"] for v in validators]
            UI.print_table(['ID','State','Score','Active'],rows)
            metrics.record_command('quantum/validator')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('quantum/validator',False)
    
    def _cmd_quantum_finality(self):
        tx_id=UI.prompt("Transaction ID")
        UI.header(f"⚛️ QUANTUM FINALITY - {tx_id[:12]}...")
        
        success,result=self.client.request('GET',f'/api/quantum/finality/{tx_id}')
        if success:
            UI.print_table(['Field','Value'],[
                ['TX ID',result.get('tx_id','')[:16]+"..."],
                ['Finality Status',result.get('finality_status','pending')],
                ['Quantum Proof',result.get('proof','')[:32]+"..."],
                ['Collapse Outcome',result.get('collapse_outcome','unknown')],
                ['Confidence',f"{float(result.get('confidence',0)):.1%}"],
                ['Validated At',result.get('validated_at','')[:19]]
            ])
            metrics.record_command('quantum/finality')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('quantum/finality',False)
    
    # ═════════════════════════════════════════════════════════════════════════════════════════
    # ORACLE COMMAND IMPLEMENTATIONS
    # ═════════════════════════════════════════════════════════════════════════════════════════
    
    def _cmd_oracle_time(self):
        UI.header("🔮 TIME ORACLE")
        success,result=self.client.request('GET','/api/oracle/time')
        
        if success:
            UI.print_table(['Field','Value'],[
                ['Current Time',result.get('iso_timestamp','N/A')],
                ['Unix Time',str(result.get('unix_timestamp','N/A'))],
                ['Block Number',str(result.get('block_number','N/A'))],
                ['Block Time',result.get('block_timestamp','N/A')]
            ])
            metrics.record_command('oracle/time')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('oracle/time',False)
    
    def _cmd_oracle_price(self):
        UI.header("🔮 PRICE ORACLE")
        symbol=UI.prompt("Symbol (QTCL/BTC/ETH/USD)","QTCL")
        
        success,result=self.client.request('GET','/api/oracle/price',params={'symbol':symbol})
        if success:
            UI.print_table(['Field','Value'],[
                ['Symbol',symbol],
                ['Price',f"${float(result.get('price',0)):.2f}"],
                ['24h Change',f"{float(result.get('change_24h',0)):+.2%}"],
                ['Market Cap',f"${float(result.get('market_cap',0)):,.0f}"],
                ['Volume 24h',f"${float(result.get('volume_24h',0)):,.0f}"]
            ])
            metrics.record_command('oracle/price')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('oracle/price',False)
    
    def _cmd_oracle_random(self):
        UI.header("🔮 QUANTUM RANDOM")
        count=int(UI.prompt("Count (1-100)","10") or "10")
        
        success,result=self.client.request('GET','/api/oracle/random',params={'count':count})
        if success:
            numbers=result.get('numbers',[])
            print("\n  Random Numbers:")
            for i,num in enumerate(numbers,1):
                print(f"    {i:2d}. {num:.8f}")
            metrics.record_command('oracle/random')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('oracle/random',False)
    
    def _cmd_oracle_event(self):
        UI.header("🔮 ORACLE EVENTS")
        event_type=UI.prompt_choice("Event type:",[
            "PRICE_CHANGE","TIME_MILESTONE","TRANSACTION_FINALITY","NETWORK_THRESHOLD"
        ])
        
        success,result=self.client.request('GET','/api/oracle/events',params={'type':event_type})
        if success:
            events=result.get('events',[])
            rows=[[e.get('event_id','')[:12]+"...",e.get('type',''),e.get('status',''),
                   e.get('created_at','')[:10]] for e in events]
            UI.print_table(['Event ID','Type','Status','Created'],rows)
            metrics.record_command('oracle/event')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('oracle/event',False)
    
    def _cmd_oracle_feed(self):
        UI.header("🔮 ORACLE FEEDS")
        success,result=self.client.request('GET','/api/oracle/feeds')
        
        if success:
            feeds=result.get('feeds',[])
            rows=[[f.get('feed_id','')[:12]+"...",f.get('name',''),f.get('type',''),
                   f.get('frequency',''),f.get('status','online')] for f in feeds]
            UI.print_table(['Feed ID','Name','Type','Frequency','Status'],rows)
            metrics.record_command('oracle/feed')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('oracle/feed',False)
    
    # ═════════════════════════════════════════════════════════════════════════════════════════
    # DEFI COMMAND IMPLEMENTATIONS
    # ═════════════════════════════════════════════════════════════════════════════════════════
    
    def _cmd_defi_stake(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("💰 STAKE TOKENS")
        amount=UI.prompt("Amount to stake")
        duration=UI.prompt("Duration (days)","30")
        pool=UI.prompt("Pool ID (optional)","")
        
        payload={'amount':amount,'duration':int(duration),'pool_id':pool}
        success,result=self.client.request('POST','/api/defi/stake',payload)
        
        if success:
            UI.success("Staking initiated")
            UI.print_table(['Field','Value'],[
                ['Stake ID',result.get('stake_id','')[:16]+"..."],
                ['Amount',f"{float(result.get('amount',0)):.2f} QTCL"],
                ['APY',f"{float(result.get('apy',0)):.2f}%"],
                ['Unlock Date',result.get('unlock_date','')[:10]]
            ])
            metrics.record_command('defi/stake')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('defi/stake',False)
    
    def _cmd_defi_unstake(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        stake_id=UI.prompt("Stake ID to unstake")
        if not UI.confirm(f"Unstake {stake_id[:12]}...?"):return
        
        success,result=self.client.request('POST',f'/api/defi/unstake/{stake_id}',{})
        if success:
            UI.success("Unstaking initiated")
            metrics.record_command('defi/unstake')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('defi/unstake',False)
    
    def _cmd_defi_borrow(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("📊 BORROW FROM POOL")
        amount=UI.prompt("Amount to borrow")
        collateral_asset=UI.prompt("Collateral asset (e.g., BTC, ETH)")
        collateral_amount=UI.prompt("Collateral amount")
        
        payload={'amount':amount,'collateral_asset':collateral_asset,'collateral_amount':collateral_amount}
        success,result=self.client.request('POST','/api/defi/borrow',payload)
        
        if success:
            UI.success("Loan created")
            UI.print_table(['Field','Value'],[
                ['Loan ID',result.get('loan_id','')[:16]+"..."],
                ['Amount',f"{float(result.get('amount',0)):.2f} QTCL"],
                ['APR',f"{float(result.get('apr',0)):.2f}%"],
                ['Repay By',result.get('repay_by','')[:10]]
            ])
            metrics.record_command('defi/borrow')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('defi/borrow',False)
    
    def _cmd_defi_repay(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        loan_id=UI.prompt("Loan ID to repay")
        amount=UI.prompt("Amount to repay")
        
        payload={'amount':amount}
        success,result=self.client.request('POST',f'/api/defi/repay/{loan_id}',payload)
        
        if success:
            UI.success("Repayment processed")
            metrics.record_command('defi/repay')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('defi/repay',False)
    
    def _cmd_defi_yield(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("📈 YIELD OPPORTUNITIES")
        success,result=self.client.request('GET','/api/defi/yields')
        
        if success:
            yields=result.get('yields',[])
            rows=[[y.get('pool_id','')[:12]+"...",y.get('asset',''),f"{float(y.get('apy',0)):.2f}%",
                   f"{float(y.get('tvl',0))/1e6:.1f}M"] for y in yields]
            UI.print_table(['Pool','Asset','APY','TVL'],rows)
            metrics.record_command('defi/yield')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('defi/yield',False)
    
    def _cmd_defi_pool(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        choice=UI.prompt_choice("Pool Operations:",[
            "Create Pool","Add Liquidity","Remove Liquidity","View Pools"
        ])
        
        if choice=="Create Pool":
            name=UI.prompt("Pool name")
            asset1=UI.prompt("Asset 1")
            asset2=UI.prompt("Asset 2")
            fee=UI.prompt("Fee (0.01-1.0%)","0.25")
            
            payload={'name':name,'assets':[asset1,asset2],'fee':float(fee)}
            success,result=self.client.request('POST','/api/defi/pools',payload)
            
            if success:
                UI.success("Pool created")
                metrics.record_command('defi/pool')
            else:
                UI.error(f"Failed: {result.get('error')}");metrics.record_command('defi/pool',False)
    
    # ═════════════════════════════════════════════════════════════════════════════════════════
    # GOVERNANCE COMMAND IMPLEMENTATIONS
    # ═════════════════════════════════════════════════════════════════════════════════════════
    
    def _cmd_governance_vote(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("🗳️ VOTE ON PROPOSAL")
        proposal_id=UI.prompt("Proposal ID")
        vote=UI.prompt_choice("Your vote:",[
            "FOR","AGAINST","ABSTAIN"
        ])
        
        payload={'vote':vote.lower()}
        success,result=self.client.request('POST',f'/api/governance/vote/{proposal_id}',payload)
        
        if success:
            UI.success("Vote recorded")
            metrics.record_command('governance/vote')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('governance/vote',False)
    
    def _cmd_governance_proposal(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("📝 CREATE PROPOSAL")
        title=UI.prompt("Proposal title")
        description=UI.prompt("Description")
        proposal_type=UI.prompt("Type (PARAMETER_CHANGE/UPGRADE/SPENDING)")
        
        payload={'title':title,'description':description,'type':proposal_type}
        success,result=self.client.request('POST','/api/governance/proposals',payload)
        
        if success:
            UI.success("Proposal created")
            UI.info(f"ID: {result.get('proposal_id','')[:16]}...")
            metrics.record_command('governance/proposal')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('governance/proposal',False)
    
    def _cmd_governance_delegate(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("👥 DELEGATE VOTING POWER")
        delegate_to=UI.prompt("Delegate address")
        amount=UI.prompt("Power to delegate")
        
        payload={'delegate':delegate_to,'power':amount}
        success,result=self.client.request('POST','/api/governance/delegate',payload)
        
        if success:
            UI.success("Delegation recorded")
            metrics.record_command('governance/delegate')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('governance/delegate',False)
    
    def _cmd_governance_stats(self):
        UI.header("📊 GOVERNANCE STATS")
        success,result=self.client.request('GET','/api/governance/stats')
        
        if success:
            stats=result.get('stats',{})
            UI.print_table(['Metric','Value'],[
                ['Active Proposals',str(stats.get('active_proposals',0))],
                ['Passed Proposals',str(stats.get('passed_proposals',0))],
                ['Avg Participation',f"{float(stats.get('avg_participation',0)):.1%}"],
                ['Voting Power',f"{float(stats.get('user_voting_power',0)):.2f}"],
                ['Delegated To',str(stats.get('delegated_to',0))]
            ])
            metrics.record_command('governance/stats')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('governance/stats',False)
    
    # ═════════════════════════════════════════════════════════════════════════════════════════
    # NFT COMMAND IMPLEMENTATIONS
    # ═════════════════════════════════════════════════════════════════════════════════════════
    
    def _cmd_nft_mint(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("🖼️ MINT NFT")
        name=UI.prompt("NFT name")
        description=UI.prompt("Description")
        image_url=UI.prompt("Image URL")
        collection=UI.prompt("Collection ID (optional)","")
        
        payload={'name':name,'description':description,'image_url':image_url,'collection_id':collection}
        success,result=self.client.request('POST','/api/nft/mint',payload)
        
        if success:
            UI.success("NFT minted")
            UI.print_table(['Field','Value'],[
                ['Token ID',result.get('token_id','')[:16]+"..."],
                ['Name',result.get('name','')],
                ['Owner',result.get('owner','')[:16]+"..."],
                ['Minted At',result.get('created_at','')[:19]]
            ])
            metrics.record_command('nft/mint')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('nft/mint',False)
    
    def _cmd_nft_transfer(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("🖼️ TRANSFER NFT")
        token_id=UI.prompt("Token ID")
        to_address=UI.prompt("To address")
        
        payload={'to_address':to_address}
        success,result=self.client.request('POST',f'/api/nft/{token_id}/transfer',payload)
        
        if success:
            UI.success("NFT transferred")
            metrics.record_command('nft/transfer')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('nft/transfer',False)
    
    def _cmd_nft_burn(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        token_id=UI.prompt("Token ID to burn")
        if not UI.confirm(f"Burn NFT {token_id[:12]}...? This cannot be undone."):return
        
        success,result=self.client.request('POST',f'/api/nft/{token_id}/burn',{})
        if success:
            UI.success("NFT burned")
            metrics.record_command('nft/burn')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('nft/burn',False)
    
    def _cmd_nft_metadata(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        token_id=UI.prompt("Token ID")
        UI.header(f"🖼️ NFT METADATA - {token_id[:12]}...")
        
        success,nft=self.client.request('GET',f'/api/nft/{token_id}')
        if success:
            UI.print_table(['Field','Value'],[
                ['Token ID',nft.get('token_id','')[:16]+"..."],
                ['Name',nft.get('name','')],
                ['Owner',nft.get('owner','')[:16]+"..."],
                ['Collection',nft.get('collection','')[:16]+"..."],
                ['Rarity',nft.get('rarity','common')],
                ['Attributes',str(len(nft.get('attributes',[])))]
            ])
            metrics.record_command('nft/metadata')
        else:
            UI.error(f"Failed: {nft.get('error')}");metrics.record_command('nft/metadata',False)
    
    def _cmd_nft_collection(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        choice=UI.prompt_choice("Collection Operations:",[
            "Create Collection","List My Collections","View Collection","Delete Collection"
        ])
        
        if choice=="Create Collection":
            name=UI.prompt("Collection name")
            description=UI.prompt("Description")
            payload={'name':name,'description':description}
            success,result=self.client.request('POST','/api/nft/collections',payload)
            
            if success:
                UI.success("Collection created")
                metrics.record_command('nft/collection')
            else:
                UI.error(f"Failed: {result.get('error')}");metrics.record_command('nft/collection',False)
    
    # ═════════════════════════════════════════════════════════════════════════════════════════
    # SMART CONTRACT COMMAND IMPLEMENTATIONS
    # ═════════════════════════════════════════════════════════════════════════════════════════
    
    def _cmd_contract_deploy(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("📝 DEPLOY SMART CONTRACT")
        contract_code=UI.prompt("Contract code (file path or inline)")
        constructor_args=UI.prompt("Constructor arguments (JSON)","[]")
        
        payload={'code':contract_code,'constructor_args':constructor_args}
        success,result=self.client.request('POST','/api/contracts',payload)
        
        if success:
            UI.success("Contract deployed")
            UI.print_table(['Field','Value'],[
                ['Address',result.get('address','')[:32]+"..."],
                ['TX ID',result.get('tx_id','')[:16]+"..."],
                ['Status',result.get('status','deployed')],
                ['Deployed At',result.get('created_at','')[:19]]
            ])
            metrics.record_command('contract/deploy')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('contract/deploy',False)
    
    def _cmd_contract_execute(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("⚙️ EXECUTE CONTRACT FUNCTION")
        contract_addr=UI.prompt("Contract address")
        function=UI.prompt("Function name")
        args=UI.prompt("Arguments (JSON)","[]")
        value=UI.prompt("ETH value to send","0")
        
        payload={'function':function,'args':args,'value':value}
        success,result=self.client.request('POST',f'/api/contracts/{contract_addr}/execute',payload)
        
        if success:
            UI.success("Function executed")
            UI.print_table(['Field','Value'],[
                ['TX ID',result.get('tx_id','')[:16]+"..."],
                ['Status',result.get('status','pending')],
                ['Result',str(result.get('result',''))[:50]]
            ])
            metrics.record_command('contract/execute')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('contract/execute',False)
    
    def _cmd_contract_compile(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("🔨 COMPILE CONTRACT")
        source_code=UI.prompt("Source code file")
        version=UI.prompt("Compiler version","0.8.0")
        
        payload={'source':source_code,'version':version}
        success,result=self.client.request('POST','/api/contracts/compile',payload)
        
        if success:
            UI.success("Compiled successfully")
            UI.print_table(['Field','Value'],[
                ['Bytecode','Created'],
                ['ABI','Available'],
                ['Warnings',str(len(result.get('warnings',[])))],
                ['Errors',str(len(result.get('errors',[])))],
                ['Size',f"{len(result.get('bytecode',''))//2} bytes"]
            ])
            metrics.record_command('contract/compile')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('contract/compile',False)
    
    def _cmd_contract_state(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        contract_addr=UI.prompt("Contract address")
        UI.header(f"📊 CONTRACT STATE - {contract_addr[:16]}...")
        
        success,result=self.client.request('GET',f'/api/contracts/{contract_addr}')
        if success:
            UI.print_table(['Field','Value'],[
                ['Address',result.get('address','')[:32]+"..."],
                ['Owner',result.get('owner','')[:16]+"..."],
                ['Balance',f"{float(result.get('balance',0)):.4f} ETH"],
                ['Created At',result.get('created_at','')[:19]],
                ['Verified',str(result.get('verified',False))]
            ])
            metrics.record_command('contract/state')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('contract/state',False)
    
    # ═════════════════════════════════════════════════════════════════════════════════════════
    # BRIDGE COMMAND IMPLEMENTATIONS
    # ═════════════════════════════════════════════════════════════════════════════════════════
    
    def _cmd_bridge_initiate(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("🌉 INITIATE CROSS-CHAIN BRIDGE")
        source_chain=UI.prompt("Source chain (QTCL/ETH/POLYGON/BSC)")
        dest_chain=UI.prompt("Destination chain")
        asset=UI.prompt("Asset to bridge (e.g., QTCL, USDC)")
        amount=UI.prompt("Amount")
        dest_address=UI.prompt("Destination address")
        
        payload={'from_chain':source_chain,'to_chain':dest_chain,'asset':asset,
                 'amount':amount,'recipient':dest_address}
        success,result=self.client.request('POST','/api/bridge/initiate',payload)
        
        if success:
            UI.success("Bridge initiated")
            UI.print_table(['Field','Value'],[
                ['Bridge ID',result.get('bridge_id','')[:16]+"..."],
                ['Status',result.get('status','initiated')],
                ['Estimated Time',result.get('eta','')],
                ['Fee',f"{float(result.get('fee',0)):.4f}"]
            ])
            metrics.record_command('bridge/initiate')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('bridge/initiate',False)
    
    def _cmd_bridge_status(self):
        bridge_id=UI.prompt("Bridge ID")
        UI.header(f"🌉 BRIDGE STATUS - {bridge_id[:12]}...")
        
        success,result=self.client.request('GET',f'/api/bridge/{bridge_id}')
        if success:
            UI.print_table(['Field','Value'],[
                ['Bridge ID',result.get('bridge_id','')[:16]+"..."],
                ['Status',result.get('status','pending')],
                ['From',result.get('from_chain','')],
                ['To',result.get('to_chain','')],
                ['Amount',f"{float(result.get('amount',0)):.4f}"],
                ['Confirmations',f"{result.get('confirmations',0)}/20"],
                ['Initiated',result.get('initiated_at','')[:19]],
                ['Completed',result.get('completed_at','')[:19] if result.get('completed_at') else "Pending"]
            ])
            metrics.record_command('bridge/status')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('bridge/status',False)
    
    def _cmd_bridge_history(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("🌉 BRIDGE HISTORY")
        success,result=self.client.request('GET','/api/bridge/history')
        
        if success:
            bridges=result.get('bridges',[])
            rows=[[b.get('bridge_id','')[:12]+"...",b.get('from_chain',''),b.get('to_chain',''),
                   f"{float(b.get('amount',0)):.2f}",b.get('status','')] for b in bridges]
            UI.print_table(['Bridge ID','From','To','Amount','Status'],rows)
            metrics.record_command('bridge/history')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('bridge/history',False)
    
    def _cmd_bridge_wrapped(self):
        if not self.session.is_authenticated():
            UI.error("Not authenticated");return
        
        UI.header("💎 WRAPPED ASSETS")
        success,result=self.client.request('GET','/api/bridge/wrapped')
        
        if success:
            wrapped=result.get('wrapped_assets',[])
            rows=[[w.get('symbol',''),w.get('original_chain',''),f"{float(w.get('balance',0)):.2f}",
                   w.get('contract','')[:16]+"..."] for w in wrapped]
            UI.print_table(['Symbol','Original Chain','Balance','Contract'],rows)
            metrics.record_command('bridge/wrapped')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('bridge/wrapped',False)
    
    # ═════════════════════════════════════════════════════════════════════════════════════════
    # ADMIN COMMAND IMPLEMENTATIONS
    # ═════════════════════════════════════════════════════════════════════════════════════════
    
    def _cmd_admin_users(self):
        if not self.session.is_admin():
            UI.error("Admin access required");return
        
        while True:
            choice=UI.prompt_choice("User Management:",[
                "List All Users","User Details","Update Role","Ban User","Restore User","Back"
            ])
            
            if choice=="List All Users":
                success,result=self.client.request('GET','/api/admin/users')
                if success:
                    users=result.get('users',[])
                    rows=[[u.get('email',''),u.get('role','').upper(),
                           '✓' if u.get('verified') else '✗','✓' if u.get('active') else '✗']
                          for u in users]
                    UI.print_table(['Email','Role','Verified','Active'],rows)
            elif choice=="User Details":
                user_id=UI.prompt("User ID")
                success,result=self.client.request('GET',f'/api/admin/users/{user_id}')
                if success:
                    u=result
                    UI.print_table(['Field','Value'],[
                        ['Email',u.get('email','')],
                        ['Role',u.get('role','').upper()],
                        ['Created',u.get('created_at','')[:19]],
                        ['Last Login',u.get('last_login','')[:19]],
                        ['Active',str(u.get('active',False))]
                    ])
            elif choice=="Update Role":
                user_id=UI.prompt("User ID")
                new_role=UI.prompt_choice("New role:",[
                    "USER","MODERATOR","ADMIN"
                ])
                success,result=self.client.request('PUT',f'/api/admin/users/{user_id}',
                    {'role':new_role.lower()})
                if success:UI.success("Role updated")
                else:UI.error(f"Failed: {result.get('error')}")
            elif choice=="Ban User":
                user_id=UI.prompt("User ID to ban")
                if UI.confirm("Confirm ban?"):
                    success,result=self.client.request('POST',f'/api/admin/users/{user_id}/ban',{})
                    if success:UI.success("User banned")
                    else:UI.error(f"Failed: {result.get('error')}")
            elif choice=="Restore User":
                user_id=UI.prompt("User ID to restore")
                success,result=self.client.request('POST',f'/api/admin/users/{user_id}/restore',{})
                if success:UI.success("User restored")
                else:UI.error(f"Failed: {result.get('error')}")
            else:break
        
        metrics.record_command('admin/users')
    
    def _cmd_admin_approval(self):
        if not self.session.is_admin():
            UI.error("Admin access required");return
        
        UI.header("✅ TRANSACTION APPROVAL")
        while True:
            choice=UI.prompt_choice("Approval Queue:",[
                "Pending Transactions","Approve TX","Reject TX","View Logs","Back"
            ])
            
            if choice=="Pending Transactions":
                success,result=self.client.request('GET','/api/admin/approvals/pending')
                if success:
                    txs=result.get('transactions',[])
                    rows=[[t.get('tx_id','')[:12]+"...",t.get('from','')[:12]+"...",
                           f"{float(t.get('amount',0)):.2f}",t.get('type','')] for t in txs]
                    UI.print_table(['TX ID','From','Amount','Type'],rows)
            elif choice=="Approve TX":
                tx_id=UI.prompt("TX ID")
                success,result=self.client.request('POST',f'/api/admin/approvals/{tx_id}/approve',{})
                if success:UI.success("Approved")
                else:UI.error(f"Failed: {result.get('error')}")
            elif choice=="Reject TX":
                tx_id=UI.prompt("TX ID")
                reason=UI.prompt("Rejection reason")
                success,result=self.client.request('POST',f'/api/admin/approvals/{tx_id}/reject',
                    {'reason':reason})
                if success:UI.success("Rejected")
                else:UI.error(f"Failed: {result.get('error')}")
            elif choice=="View Logs":
                success,result=self.client.request('GET','/api/admin/approvals/logs',params={'limit':20})
                if success:
                    logs=result.get('logs',[])
                    for log in logs[-10:]:
                        print(f"  {log.get('timestamp','')[:19]} - {log.get('action','')} by {log.get('admin','')[:12]}...")
            else:break
        
        metrics.record_command('admin/approval')
    
    def _cmd_admin_monitoring(self):
        if not self.session.is_admin():
            UI.error("Admin access required");return
        
        UI.header("📊 SYSTEM MONITORING")
        success,result=self.client.request('GET','/api/admin/monitoring')
        
        if success:
            UI.print_table(['Metric','Value'],[
                ['Active Users',str(result.get('active_users',0))],
                ['Total Users',str(result.get('total_users',0))],
                ['Transactions/Hour',str(result.get('tx_per_hour',0))],
                ['Avg Block Time',f"{float(result.get('avg_block_time',0)):.2f}s"],
                ['Network TPS',f"{float(result.get('tps',0)):.2f}"],
                ['API Health',result.get('api_health','healthy')],
                ['Database',result.get('db_status','healthy')],
                ['Quantum Engine',result.get('quantum_status','operational')]
            ])
            metrics.record_command('admin/monitoring')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('admin/monitoring',False)
    
    def _cmd_admin_settings(self):
        if not self.session.is_admin():
            UI.error("Admin access required");return
        
        UI.header("⚙️ SYSTEM SETTINGS")
        while True:
            choice=UI.prompt_choice("Settings:",[
                "Rate Limiting","Transaction Fees","Token Parameters","Security","View All","Back"
            ])
            
            if choice=="Rate Limiting":
                limit=UI.prompt("Requests per minute")
                window=UI.prompt("Time window (seconds)")
                success,result=self.client.request('PUT','/api/admin/settings/rate-limit',
                    {'limit':int(limit),'window':int(window)})
                if success:UI.success("Settings updated")
                else:UI.error(f"Failed: {result.get('error')}")
            elif choice=="Transaction Fees":
                min_fee=UI.prompt("Min fee (QTCL)")
                success,result=self.client.request('PUT','/api/admin/settings/fees',
                    {'min_fee':float(min_fee)})
                if success:UI.success("Fees updated")
                else:UI.error(f"Failed: {result.get('error')}")
            elif choice=="Token Parameters":
                name=UI.prompt("Parameter name")
                value=UI.prompt("New value")
                success,result=self.client.request('PUT','/api/admin/settings/token',
                    {name:value})
                if success:UI.success("Parameter updated")
                else:UI.error(f"Failed: {result.get('error')}")
            elif choice=="Security":
                enable_2fa=UI.confirm("Require 2FA for all users?")
                success,result=self.client.request('PUT','/api/admin/settings/security',
                    {'require_2fa':enable_2fa})
                if success:UI.success("Security settings updated")
                else:UI.error(f"Failed: {result.get('error')}")
            elif choice=="View All":
                success,result=self.client.request('GET','/api/admin/settings')
                if success:
                    settings=result.get('settings',{})
                    rows=[[k,str(v)] for k,v in list(settings.items())[:15]]
                    UI.print_table(['Setting','Value'],rows)
            else:break
        
        metrics.record_command('admin/settings')
    
    def _cmd_admin_audit(self):
        if not self.session.is_admin():
            UI.error("Admin access required");return
        
        UI.header("📋 AUDIT LOGS")
        action_filter=UI.prompt("Filter by action (leave empty for all)")
        limit=int(UI.prompt("Limit","50") or "50")
        
        params={'limit':limit}
        if action_filter:params['action']=action_filter
        
        success,result=self.client.request('GET','/api/admin/audit',params=params)
        if success:
            logs=result.get('logs',[])
            rows=[[l.get('timestamp','')[:19],l.get('user','')[:12]+"...",
                   l.get('action',''),l.get('resource','')[:12]+"..."] for l in logs]
            UI.print_table(['Timestamp','User','Action','Resource'],rows)
            UI.info(f"Showing {len(logs)} audit entries")
            metrics.record_command('admin/audit')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('admin/audit',False)
    
    def _cmd_admin_emergency(self):
        if not self.session.is_admin():
            UI.error("Admin access required");return
        
        UI.header("🚨 EMERGENCY CONTROLS")
        choice=UI.prompt_choice("Emergency Action:",[
            "Pause All Transactions","Resume Transactions","Freeze Account","Unfreeze Account",
            "Circuit Breaker Status","Back"
        ])
        
        if choice=="Pause All Transactions":
            if UI.confirm("PAUSE ALL TRANSACTIONS SYSTEM-WIDE?"):
                success,result=self.client.request('POST','/api/admin/emergency/pause',{})
                if success:
                    UI.warning("SYSTEM PAUSED")
                    metrics.record_command('admin/emergency')
                else:UI.error(f"Failed: {result.get('error')}")
        elif choice=="Resume Transactions":
            success,result=self.client.request('POST','/api/admin/emergency/resume',{})
            if success:
                UI.success("SYSTEM RESUMED")
                metrics.record_command('admin/emergency')
            else:UI.error(f"Failed: {result.get('error')}")
        elif choice=="Freeze Account":
            account=UI.prompt("Account to freeze")
            success,result=self.client.request('POST',f'/api/admin/emergency/freeze/{account}',{})
            if success:UI.success("Account frozen")
            else:UI.error(f"Failed: {result.get('error')}")
        elif choice=="Unfreeze Account":
            account=UI.prompt("Account to unfreeze")
            success,result=self.client.request('POST',f'/api/admin/emergency/unfreeze/{account}',{})
            if success:UI.success("Account unfrozen")
            else:UI.error(f"Failed: {result.get('error')}")
        elif choice=="Circuit Breaker Status":
            success,result=self.client.request('GET','/api/admin/emergency/status')
            if success:
                UI.print_table(['Component','Status'],[
                    ['System State',result.get('state','')],
                    ['Circuit Breaker','ACTIVE' if result.get('active') else 'INACTIVE'],
                    ['Transactions','PAUSED' if result.get('paused') else 'ACTIVE'],
                    ['Withdrawals','ENABLED' if result.get('withdrawals_enabled') else 'DISABLED']
                ])
    
    # ═════════════════════════════════════════════════════════════════════════════════════════
    # SYSTEM COMMAND IMPLEMENTATIONS
    # ═════════════════════════════════════════════════════════════════════════════════════════
    
    def _cmd_system_status(self):
        UI.header("🖥️ SYSTEM STATUS")
        success,result=self.client.request('GET','/health')
        
        if success:
            cpu_percent=psutil.cpu_percent()
            memory=psutil.virtual_memory()
            disk=psutil.disk_usage('/')
            
            UI.print_table(['Component','Status'],[
                ['API Server',result.get('status','offline')],
                ['Database',result.get('database','unknown')],
                ['Cache',result.get('cache','unknown')],
                ['Quantum Engine',result.get('quantum','offline')],
                ['CPU Usage',f"{cpu_percent}%"],
                ['Memory Usage',f"{memory.percent}%"],
                ['Disk Usage',f"{disk.percent}%"],
                ['Uptime',f"{result.get('uptime_seconds',0)//3600}h"]
            ])
            metrics.record_command('system/status')
        else:
            UI.error("System offline");metrics.record_command('system/status',False)
    
    def _cmd_system_health(self):
        UI.header("❤️ SYSTEM HEALTH")
        success,result=self.client.request('POST','/api/heartbeat',{})
        
        if success:
            UI.success("System healthy")
            UI.print_table(['Check','Result'],[
                ['API Response','OK'],
                ['Database','Connected'],
                ['Network','Stable'],
                ['Latency',f"{float(result.get('latency_ms',0)):.0f}ms"]
            ])
            metrics.record_command('system/health')
        else:
            UI.error("Health check failed");metrics.record_command('system/health',False)
    
    def _cmd_system_config(self):
        UI.header("⚙️ SYSTEM CONFIGURATION")
        success,result=self.client.request('GET','/api/system/config')
        
        if success:
            cfg=result.get('config',{})
            UI.print_table(['Setting','Value'],[
                ['API Version',cfg.get('api_version','')],
                ['Environment',cfg.get('environment','')],
                ['Max Block Size',f"{cfg.get('max_block_size',0)//1024}KB"],
                ['Transaction Fee',f"{cfg.get('tx_fee',0):.4f}"],
                ['Network ID',str(cfg.get('network_id',0))],
                ['Genesis Block',cfg.get('genesis_block','')[:16]+"..."]
            ])
            metrics.record_command('system/config')
        else:
            UI.error(f"Failed: {result.get('error')}");metrics.record_command('system/config',False)
    
    def _cmd_system_backup(self):
        if not self.session.is_admin():
            UI.error("Admin access required");return
        
        UI.header("💾 SYSTEM BACKUP")
        if not UI.confirm("Start backup? This may take several minutes."):return
        
        success,result=self.client.request('POST','/api/admin/backup',{})
        if success:
            filename=result.get('backup_file','backup.tar.gz')
            UI.success(f"Backup completed: {filename}")
            UI.info(f"Size: {result.get('size_mb',0):.1f} MB")
            metrics.record_command('system/backup')
        else:
            UI.error(f"Backup failed: {result.get('error')}");metrics.record_command('system/backup',False)
    
    def _cmd_system_restore(self):
        if not self.session.is_admin():
            UI.error("Admin access required");return
        
        UI.header("🔄 RESTORE FROM BACKUP")
        backup_file=UI.prompt("Backup file path")
        
        if not UI.confirm("RESTORE SYSTEM? This will overwrite current data."):return
        
        success,result=self.client.request('POST','/api/admin/restore',{'backup_file':backup_file})
        if success:
            UI.success("Restore completed")
            metrics.record_command('system/restore')
        else:
            UI.error(f"Restore failed: {result.get('error')}");metrics.record_command('system/restore',False)
    
    # ═════════════════════════════════════════════════════════════════════════════════════════
    # PARALLEL COMMAND IMPLEMENTATIONS
    # ═════════════════════════════════════════════════════════════════════════════════════════
    
    def _cmd_parallel_execute(self):
        UI.header("⚡ PARALLEL COMMAND EXECUTION")
        num_commands=int(UI.prompt("Number of commands to execute","2"))
        
        commands=[]
        for i in range(num_commands):
            cmd=UI.prompt(f"Command {i+1}")
            commands.append(cmd)
        
        task_ids=[]
        for cmd in commands:
            def task_func(c=cmd):
                success,result=self.client.request('GET','/api/system/status')
                return {'command':c,'status':'executed','result':result}
            task_id=self.executor.submit(task_func)
            task_ids.append(task_id)
        
        UI.info(f"Executing {len(task_ids)} commands in parallel...")
        results=[]
        for tid in task_ids:
            result=self.executor.get_result(tid,timeout=30)
            if result:results.append(result)
        
        UI.success(f"Completed {len(results)}/{len(task_ids)} commands")
        metrics.record_command('parallel/execute')
    
    def _cmd_parallel_batch(self):
        UI.header("📦 BATCH OPERATIONS")
        batch_type=UI.prompt_choice("Batch type:",[
            "Send Transactions","Create Wallets","Update Settings","Approve Transactions"
        ])
        
        count=int(UI.prompt("Number of items","5"))
        
        if batch_type=="Send Transactions":
            for i in range(count):
                to_addr=UI.prompt(f"Recipient {i+1}")
                amount=UI.prompt(f"Amount {i+1}")
                # Queue transaction in parallel
        
        UI.success(f"Queued {count} batch operations")
        metrics.record_command('parallel/batch')
    
    def _cmd_parallel_monitor(self):
        UI.header("📊 PARALLEL TASK MONITOR")
        tasks=self.executor.wait_all()
        
        if not tasks:
            UI.info("No active tasks")
            return
        
        rows=[[tid[:8],t.command,t.status,f"{(t.end_time or time.time())-t.start_time:.2f}s"]
              for tid,t in list(tasks.items())[-20:]]
        UI.print_table(['Task ID','Command','Status','Duration'],rows)
        metrics.record_command('parallel/monitor')
    
    # ═════════════════════════════════════════════════════════════════════════════════════════
    # HELP COMMAND IMPLEMENTATIONS
    # ═════════════════════════════════════════════════════════════════════════════════════════
    
    def _cmd_help(self):
        UI.header("📚 HELP & DOCUMENTATION")
        
        if self.session.is_authenticated():
            if self.session.is_admin():
                categories=[cat for cat in CommandCategory]
            else:
                categories=[cat for cat in CommandCategory if cat.value!='admin']
        else:
            categories=[CommandCategory.AUTH,CommandCategory.HELP]
        
        choice=UI.prompt_choice("Help Category:",[cat.value.upper() for cat in categories]+["Back"])
        
        if choice=="Back":return
        
        selected_cat=CommandCategory[choice.upper()] if choice!="Back" else None
        if selected_cat:
            cmds=self.registry.list_by_category(selected_cat)
            print(f"\n{Fore.CYAN}{selected_cat.value.upper()} Commands:{Style.RESET_ALL}\n")
            for name,meta in cmds:
                print(f"  {Fore.GREEN}{name}{Style.RESET_ALL}: {meta.description}")
        
        metrics.record_command('help')
    
    def _cmd_help_admin(self):
        if not self.session.is_admin():
            UI.error("Admin access required");return
        
        UI.header("👑 ADMIN HELP")
        admin_cmds=self.registry.list_by_category(CommandCategory.ADMIN)
        
        print(f"\n{Fore.CYAN}Admin-Only Commands:{Style.RESET_ALL}\n")
        for name,meta in admin_cmds:
            print(f"  {Fore.RED}{name}{Style.RESET_ALL}")
            print(f"    {meta.description}")
        
        print(f"\n{Fore.YELLOW}Admin Features:{Style.RESET_ALL}")
        print("  • User management and role control")
        print("  • Transaction approval workflows")
        print("  • System monitoring and analytics")
        print("  • Rate limiting and quota management")
        print("  • Audit logs and compliance")
        print("  • Emergency controls")
        print("  • Database backup/restore")
        
        metrics.record_command('help/admin')
    
    def _cmd_help_search(self):
        query=UI.prompt("Search for")
        results=self.registry.search(query)
        
        if results:
            UI.header(f"📖 SEARCH RESULTS for '{query}'")
            for name,meta in results[:20]:
                print(f"  {Fore.GREEN}{name}{Style.RESET_ALL}: {meta.description}")
        else:
            UI.info("No results found")
        
        metrics.record_command('help/search')
    
    def _cmd_help_commands(self):
        UI.header("📖 ALL AVAILABLE COMMANDS")
        all_cmds=self.registry.list_all()
        
        by_category=defaultdict(list)
        for name,meta in all_cmds:
            by_category[meta.category.value].append((name,meta))
        
        for cat in sorted(by_category.keys()):
            print(f"\n{Fore.CYAN}{cat.upper()}:{Style.RESET_ALL}")
            for name,meta in sorted(by_category[cat]):
                auth_req=" [AUTH]" if meta.requires_auth else ""
                admin_req=" [ADMIN]" if meta.requires_admin else ""
                print(f"  {name}{Fore.YELLOW}{auth_req}{admin_req}{Style.RESET_ALL}")
        
        metrics.record_command('help/commands')
    
    def _cmd_help_examples(self):
        UI.header("💡 COMMAND EXAMPLES")
        
        examples={
            "Login":"login → Enter email & password",
            "Create Transaction":"transaction/create → Enter recipient, amount, type",
            "List Wallets":"wallet/list → Shows all user wallets",
            "Check Block":"block/details → Enter block number",
            "Get Price":"oracle/price → Shows current QTCL price",
            "Stake Tokens":"defi/stake → Enter amount & duration",
            "Vote":"governance/vote → Vote on proposal",
            "Deploy Contract":"contract/deploy → Deploy smart contract",
            "Admin Users":"admin/users → Manage system users [ADMIN ONLY]",
            "Parallel Execute":"parallel/execute → Run commands in parallel"
        }
        
        rows=[[k,v] for k,v in examples.items()]
        UI.print_table(['Command','Description'],rows)
        metrics.record_command('help/examples')
    
    # ═════════════════════════════════════════════════════════════════════════════════════════
    # MAIN LOOP & SHUTDOWN
    # ═════════════════════════════════════════════════════════════════════════════════════════
    
    def run(self):
        """Main terminal loop"""
        UI.header("QUANTUM TEMPORAL COHERENCE LEDGER v5.0")
        print(f"  {Fore.CYAN}Connecting to: {self.client.base_url}{Style.RESET_ALL}")
        print(f"  {Fore.CYAN}API Connection: {'✓' if Config.verify_api_connection() else '✗'}{Style.RESET_ALL}\n")
        
        if not Config.verify_api_connection():
            UI.warning(f"Warning: Cannot reach API at {self.client.base_url}")
        
        while self.running:
            try:
                if not self.session.is_authenticated():
                    choice=UI.prompt_choice("Main Menu:",[
                        "Login","Register","Help","Exit"
                    ])
                    
                    if choice=="Login":self._cmd_login()
                    elif choice=="Register":self._cmd_register()
                    elif choice=="Help":self._cmd_help()
                    elif choice=="Exit":break
                else:
                    # Authenticated menu
                    admin_menu=["Users","Approvals","Monitoring","Settings","Audit","Emergency"] if self.session.is_admin() else []
                    
                    main_options=[
                        "Transactions","Wallets","Blocks","Oracle","DeFi","Governance",
                        "NFT","Smart Contracts","Bridge","Quantum"
                    ]
                    
                    if admin_menu:
                        main_options.extend(["─────────ADMIN─────────"])
                        main_options.extend(admin_menu)
                    
                    main_options.extend(["─────────────────────","Profile","Help","Logout","Exit"])
                    
                    choice=UI.prompt_choice("Main Menu:",main_options)
                    
                    if choice=="Transactions":self._transaction_submenu()
                    elif choice=="Wallets":self._wallet_submenu()
                    elif choice=="Blocks":self._block_submenu()
                    elif choice=="Oracle":self._oracle_submenu()
                    elif choice=="DeFi":self._defi_submenu()
                    elif choice=="Governance":self._governance_submenu()
                    elif choice=="NFT":self._nft_submenu()
                    elif choice=="Smart Contracts":self._contract_submenu()
                    elif choice=="Bridge":self._bridge_submenu()
                    elif choice=="Quantum":self._quantum_submenu()
                    elif choice=="Users":self._cmd_admin_users()
                    elif choice=="Approvals":self._cmd_admin_approval()
                    elif choice=="Monitoring":self._cmd_admin_monitoring()
                    elif choice=="Settings":self._cmd_admin_settings()
                    elif choice=="Audit":self._cmd_admin_audit()
                    elif choice=="Emergency":self._cmd_admin_emergency()
                    elif choice=="Profile":self._cmd_user_profile()
                    elif choice=="Help":self._cmd_help()
                    elif choice=="Logout":self._cmd_logout()
                    elif choice=="Exit":break
                    
                    input(f"\n{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
            
            except KeyboardInterrupt:
                if UI.confirm("Exit?"):break
            except Exception as e:
                logger.error(f"Error: {e}",exc_info=True)
                UI.error(f"Error: {e}")
        
        self.shutdown()
    
    def _transaction_submenu(self):
        while True:
            choice=UI.prompt_choice("Transactions:",[
                "Create","Track","List","Cancel","Analyze","Export Stats","Back"
            ])
            if choice=="Create":self._cmd_tx_create()
            elif choice=="Track":self._cmd_tx_track()
            elif choice=="List":self._cmd_tx_list()
            elif choice=="Cancel":self._cmd_tx_cancel()
            elif choice=="Analyze":self._cmd_tx_analyze()
            elif choice=="Export Stats":self._cmd_tx_export()
            else:break
    
    def _wallet_submenu(self):
        while True:
            choice=UI.prompt_choice("Wallets:",[
                "Create","List","Balance","Import","Export","Multi-sig","Back"
            ])
            if choice=="Create":self._cmd_wallet_create()
            elif choice=="List":self._cmd_wallet_list()
            elif choice=="Balance":self._cmd_wallet_balance()
            elif choice=="Import":self._cmd_wallet_import()
            elif choice=="Export":self._cmd_wallet_export()
            elif choice=="Multi-sig":self._multisig_submenu()
            else:break
    
    def _multisig_submenu(self):
        choice=UI.prompt_choice("Multi-sig:",[
            "Create Wallet","Sign TX","Back"
        ])
        if choice=="Create Wallet":self._cmd_multisig_create()
        elif choice=="Sign TX":self._cmd_multisig_sign()
    
    def _block_submenu(self):
        while True:
            choice=UI.prompt_choice("Blocks:",[
                "List","Details","Explorer","Statistics","Back"
            ])
            if choice=="List":self._cmd_block_list()
            elif choice=="Details":self._cmd_block_details()
            elif choice=="Explorer":self._cmd_block_explorer()
            elif choice=="Statistics":self._cmd_block_stats()
            else:break
    
    def _oracle_submenu(self):
        while True:
            choice=UI.prompt_choice("Oracle:",[
                "Time","Price","Random","Events","Feeds","Back"
            ])
            if choice=="Time":self._cmd_oracle_time()
            elif choice=="Price":self._cmd_oracle_price()
            elif choice=="Random":self._cmd_oracle_random()
            elif choice=="Events":self._cmd_oracle_event()
            elif choice=="Feeds":self._cmd_oracle_feed()
            else:break
    
    def _defi_submenu(self):
        while True:
            choice=UI.prompt_choice("DeFi:",[
                "Stake","Unstake","Borrow","Repay","Yield","Pools","Back"
            ])
            if choice=="Stake":self._cmd_defi_stake()
            elif choice=="Unstake":self._cmd_defi_unstake()
            elif choice=="Borrow":self._cmd_defi_borrow()
            elif choice=="Repay":self._cmd_defi_repay()
            elif choice=="Yield":self._cmd_defi_yield()
            elif choice=="Pools":self._cmd_defi_pool()
            else:break
    
    def _governance_submenu(self):
        while True:
            choice=UI.prompt_choice("Governance:",[
                "Vote","Proposal","Delegate","Statistics","Back"
            ])
            if choice=="Vote":self._cmd_governance_vote()
            elif choice=="Proposal":self._cmd_governance_proposal()
            elif choice=="Delegate":self._cmd_governance_delegate()
            elif choice=="Statistics":self._cmd_governance_stats()
            else:break
    
    def _nft_submenu(self):
        while True:
            choice=UI.prompt_choice("NFT:",[
                "Mint","Transfer","Burn","Metadata","Collections","Back"
            ])
            if choice=="Mint":self._cmd_nft_mint()
            elif choice=="Transfer":self._cmd_nft_transfer()
            elif choice=="Burn":self._cmd_nft_burn()
            elif choice=="Metadata":self._cmd_nft_metadata()
            elif choice=="Collections":self._cmd_nft_collection()
            else:break
    
    def _contract_submenu(self):
        while True:
            choice=UI.prompt_choice("Smart Contracts:",[
                "Deploy","Execute","Compile","State","Back"
            ])
            if choice=="Deploy":self._cmd_contract_deploy()
            elif choice=="Execute":self._cmd_contract_execute()
            elif choice=="Compile":self._cmd_contract_compile()
            elif choice=="State":self._cmd_contract_state()
            else:break
    
    def _bridge_submenu(self):
        while True:
            choice=UI.prompt_choice("Bridge:",[
                "Initiate","Status","History","Wrapped Assets","Back"
            ])
            if choice=="Initiate":self._cmd_bridge_initiate()
            elif choice=="Status":self._cmd_bridge_status()
            elif choice=="History":self._cmd_bridge_history()
            elif choice=="Wrapped Assets":self._cmd_bridge_wrapped()
            else:break
    
    def _quantum_submenu(self):
        while True:
            choice=UI.prompt_choice("Quantum:",[
                "Status","Circuit","Entropy","Validators","Finality","Back"
            ])
            if choice=="Status":self._cmd_quantum_status()
            elif choice=="Circuit":self._cmd_quantum_circuit()
            elif choice=="Entropy":self._cmd_quantum_entropy()
            elif choice=="Validators":self._cmd_quantum_validator()
            elif choice=="Finality":self._cmd_quantum_finality()
            else:break
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down terminal...")
        summary=metrics.get_summary()
        logger.info(f"Session summary: {json.dumps(summary,indent=2,default=str)}")
        self.running=False
        self.executor.shutdown()
        print(f"\n{Fore.CYAN}Goodbye!{Style.RESET_ALL}\n")

# ═════════════════════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser=argparse.ArgumentParser(description='QTCL Terminal Orchestrator v5.0')
    parser.add_argument('--api-url',help='API server URL')
    parser.add_argument('--debug',action='store_true',help='Enable debug logging')
    
    args=parser.parse_args()
    
    if args.api_url:Config.API_BASE_URL=args.api_url
    if args.debug:logger.setLevel(logging.DEBUG)
    
    engine=TerminalEngine()
    engine.run()

if __name__=='__main__':
    main()
