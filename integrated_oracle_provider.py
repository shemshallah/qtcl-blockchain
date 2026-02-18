#!/usr/bin/env python3
"""
INTEGRATED ORACLE PRICE PROVIDER - Unified price source with subsystem broadcasting
"""
import threading
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class UnifiedOraclePriceProvider:
    """Unified oracle price provider with subsystem integration"""
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.price_cache = {
            'QTCL-USD': {'price': 1.0, 'timestamp': datetime.now(timezone.utc).isoformat(), 'source': 'internal'},
            'BTC-USD': {'price': 45000.0, 'timestamp': datetime.now(timezone.utc).isoformat(), 'source': 'chainlink'},
            'ETH-USD': {'price': 3000.0, 'timestamp': datetime.now(timezone.utc).isoformat(), 'source': 'chainlink'},
            'USDC-USD': {'price': 1.0, 'timestamp': datetime.now(timezone.utc).isoformat(), 'source': 'chainlink'},
        }
        self.blockchain = None
        self.defi = None
        self.ledger = None
        self.price_updates = 0
        self.broadcast_count = 0
        self.error_count = 0
        self._init_subsystems()
        self._initialized = True
        logger.info('[OraclePriceProvider] Initialized')
    
    def _init_subsystems(self):
        try:
            from blockchain_api import get_blockchain_integration
            self.blockchain = get_blockchain_integration()
        except: pass
        try:
            from defi_api import get_defi_integration
            self.defi = get_defi_integration()
        except: pass
        try:
            from ledger_manager import get_ledger_integration
            self.ledger = get_ledger_integration()
        except: pass
    
    def get_price(self, symbol: str) -> Dict[str, Any]:
        symbol = symbol.upper().replace('/', '-')
        if symbol in self.price_cache:
            entry = self.price_cache[symbol]
            return {
                'symbol': symbol, 'price': entry['price'],
                'timestamp': entry['timestamp'], 'source': entry['source'],
                'available': True, 'cached': True
            }
        return {'symbol': symbol, 'price': None, 'available': False, 'cached': False}
    
    def update_price(self, symbol: str, price: float, source: str = 'manual'):
        symbol = symbol.upper().replace('/', '-')
        self.price_cache[symbol] = {
            'price': price, 'timestamp': datetime.now(timezone.utc).isoformat(), 'source': source
        }
        self.price_updates += 1
        self._broadcast(symbol, price)
    
    def _broadcast(self, symbol: str, price: float):
        try:
            if self.blockchain and hasattr(self.blockchain, 'oracle_price_cache'):
                self.blockchain.oracle_price_cache[symbol] = price
                self.broadcast_count += 1
        except: pass
        try:
            if self.defi and hasattr(self.defi, 'oracle_prices'):
                self.defi.oracle_prices[symbol] = price
                self.broadcast_count += 1
        except: pass
        try:
            if self.ledger and hasattr(self.ledger, 'record_price_feed'):
                self.ledger.record_price_feed(symbol, price)
                self.broadcast_count += 1
        except: pass
    
    def get_all_prices(self) -> Dict[str, Dict[str, Any]]:
        return self.price_cache.copy()
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'module': 'oracle_price_provider', 'cached_symbols': len(self.price_cache),
            'price_updates': self.price_updates, 'broadcast_count': self.broadcast_count,
            'error_count': self.error_count, 'subsystems': {
                'blockchain': self.blockchain is not None,
                'defi': self.defi is not None, 'ledger': self.ledger is not None
            }
        }

ORACLE_PRICE_PROVIDER = UnifiedOraclePriceProvider()

def get_oracle_price_provider():
    return ORACLE_PRICE_PROVIDER


class ResponseWrapper:
    """Ensure all responses are valid JSON with proper structure"""
    @staticmethod
    def success(data=None, message='OK', metadata=None):
        resp = {'status': 'success', 'result': data or {}, 'message': message}
        if metadata: resp['metadata'] = metadata
        return resp
    
    @staticmethod
    def error(error: str, error_code: str = 'ERROR', suggestions=None, metadata=None):
        resp = {'status': 'error', 'error': error, 'error_code': error_code}
        if suggestions: resp['suggestions'] = suggestions
        if metadata: resp['metadata'] = metadata
        return resp
    
    @staticmethod
    def ensure_json(obj):
        if isinstance(obj, dict) and 'status' in obj:
            return obj
        elif obj is None:
            return ResponseWrapper.success()
        else:
            return ResponseWrapper.success(obj)
