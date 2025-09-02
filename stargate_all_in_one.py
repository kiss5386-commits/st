# === GPT PATCH HELPER: BEGIN ===
def _as_bool(v, default=True):
    if v is None: return default
    if isinstance(v, bool): return v
    return str(v).strip().lower() in ("1","true","yes","y","on")

def trailing_enabled_from_cfg(cfg):
    try:
        if isinstance(cfg, dict):
            if "enable_trailing" in cfg:
                return _as_bool(cfg.get("enable_trailing"), True)
            if "trailing_enabled" in cfg:
                return _as_bool(cfg.get("trailing_enabled"), True)
            t = cfg.get("trailing") or {}
            if isinstance(t, dict) and "enabled" in t:
                return _as_bool(t.get("enabled"), True)
        return True
    except Exception:
        return True
# === GPT PATCH HELPER: END ===

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stargate Multi-Exchange Trading Server v4.2 - Enhanced Risk Protection System
=============================================================================

주요 개선사항:
1. 재시도 기반 보호장치 시스템 구축 (enforce_protective_exits)
2. 손절/트레일링 실패시 강제청산 안전장치
3. 거래소별 어댑터 인터페이스 통일
4. 구버전 단일 시도 코드 삭제 및 최적화
5. 판매용 수준 안정성 및 리스크 관리 강화

지원 거래소:
- Bybit (기존 심볼 정규화 유지)
- Bitget (실시간 API 기반 심볼 매핑)

Author: Enhanced by Claude (Risk Protection Enhanced Version)
Version: 4.2
License: Commercial Use
"""

import sys
import json
import time
import hmac
import hashlib
import base64
import logging
import logging.handlers
import multiprocessing as mp
import webbrowser
import queue
import threading
import socket
import math
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Iterable
from abc import ABC, abstractmethod

# GUI (optional in headless environments)
GUI_AVAILABLE = False
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    GUI_AVAILABLE = True
except Exception:
    tk = ttk = messagebox = None  # headless-safe

# Web server
from flask import Flask, request, jsonify
import argparse

# HTTP
import requests

# ============================================================================
# 공통 유틸리티 함수
# ============================================================================


def round_to_tick(price: float, tick: float, side: str) -> float:
    """가격을 틱 단위로 반올림"""
    if side.lower() == "long":
        return math.floor(price / tick) * tick
    return math.ceil(price / tick) * tick


# ============================================================================
# 가격 및 VWAP 유틸리티 클래스 (중복 로직 통합)
# ============================================================================


class _PriceUtils:
    """가격 관련 유틸리티 함수들"""

    @staticmethod
    def within_deviation(a: float, b: float, max_pct: float = 10.0) -> bool:
        """
        두 가격 간의 편차가 허용 범위 내인지 확인

        Args:
            a: 첫 번째 가격
            b: 두 번째 가격 (기준값)
            max_pct: 최대 허용 편차 (%)

        Returns:
            편차가 허용 범위 내이면 True
        """
        if a is None or b is None or b == 0:
            return False
        return abs(a - b) / abs(b) * 100 <= max_pct


class _VWAPUtils:
    """
    공용 VWAP 계산 유틸.
    - 거래소 무관/부작용 없음
    - 입력: fills = Iterable[Dict], 각 항목에 price/qty(size) 키 존재
    """

    @staticmethod
    def calc_vwap_from_fills(fills: Optional[Iterable[Dict]]) -> Optional[float]:
        if not fills:
            return None
        notional, qty = 0.0, 0.0
        for f in fills:
            # 다양한 키 호환 (price/avgPrice, size/qty/filledQty/baseVolume/execQty)
            p = float(f.get("price") or f.get("avgPrice") or f.get("execPrice") or 0)
            q = float(
                f.get("size")
                or f.get("qty")
                or f.get("filledQty")
                or f.get("baseVolume")
                or f.get("execQty")
                or 0
            )
            if p > 0 and q > 0:
                notional += p * q
                qty += q
        return (notional / qty) if qty > 0 else None


def log_payload_safe(logger, tag: str, payload: dict):
    """안전한 payload 로깅 (API 키 제외)"""
    try:
        safe_payload = {
            k: v
            for k, v in payload.items()
            if k not in ["ACCESS-KEY", "ACCESS-SIGN", "ACCESS-PASSPHRASE"]
        }
        logger.info(f"{tag} {json.dumps(safe_payload, ensure_ascii=False)}")
    except Exception as e:
        logger.error(f"{tag} payload 로깅 오류: {e}")


# ============================================================================
# 상수 정의
# ============================================================================

SUPPORTED_EXCHANGES = ["bybit", "bitget"]
DEFAULT_VALUES = {
    "tick_size": 0.01,
    "step_size": 0.001,
    "min_qty": 0.001,
    "min_notional": 5.0,
}

# Bitget 진입가 확보 전용 상수 (GPT 요청사항)
BITGET_POSITION_RETRY_COUNT = 35
BITGET_POSITION_RETRY_INTERVAL = 0.5  # 초
BITGET_FILLS_LOOKBACK_COUNT = 50
BITGET_VWAP_MIN_FILLS = 3  # 최소 체결 건수
BITGET_MAX_PRICE_DEVIATION = 10.0  # ±10% 편차 임계값
BITGET_ENTRY_CACHE_TTL = 300  # 5분
BITGET_ENTRY_CACHE_FILE = "entry_cache.json"
BITGET_PRODUCT_TYPE = "USDT-FUTURES"
BITGET_MARGIN_COIN = "USDT"

LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ============================================================================
# Bybit 전용 심볼 정규화 시스템
# ============================================================================


class BybitSymbolNormalizer:
    """Bybit 전용 심볼 정규화"""

    def __init__(self):
        # 주요 베이스 코인들
        self.base_coins = [
            "BTC",
            "ETH",
            "BNB",
            "XRP",
            "ADA",
            "SOL",
            "DOT",
            "DOGE",
            "AVAX",
            "MATIC",
            "LINK",
            "ATOM",
            "UNI",
            "LTC",
            "BCH",
            "XLM",
            "ALGO",
            "VET",
            "ICP",
            "FIL",
            "TRX",
            "ETC",
            "THETA",
            "XMR",
            "NEAR",
            "FLOW",
            "MANA",
            "SAND",
            "AXS",
            "CHZ",
            "ENJ",
            "BAT",
            "ZIL",
            "HOT",
            "ONT",
            "ZRX",
            "REN",
            "KNC",
            "LRC",
            "CRV",
            "COMP",
            "AAVE",
            "SNX",
            "MKR",
            "YFI",
            "SUSHI",
            "BAL",
            "RLC",
            "NMR",
            "ANT",
            "OP",
            "ARB",
            "SHIB",
            "PEPE",
            "WIF",
            "BONK",
            "FLOKI",
            "DEGEN",
            "NOT",
            "TON",
        ]

        # 쿼트 화폐들
        self.quote_currencies = ["USDT", "USDC", "BUSD", "USD", "BTC", "ETH", "BNB"]

    def normalize_symbol(self, symbol: str) -> str:
        """Bybit용 심볼 정규화"""
        if not symbol:
            return ""

        original_symbol = symbol
        symbol = symbol.upper().strip()

        # Perpetual 접미사 우선 제거
        perpetual_suffixes = [".P", ".PERP", "_PERP", "-PERP", "PERP"]
        for suffix in perpetual_suffixes:
            if symbol.endswith(suffix):
                symbol = symbol[: -len(suffix)].rstrip(".-_")
                break

        # 특수 문자 정리
        symbol = symbol.replace("/", "").replace(":", "").replace(" ", "")
        symbol = symbol.replace("-", "").replace("_", "").replace(".", "")

        try:
            base, quote = self._parse_symbol(symbol)

            if not base:
                return symbol

            if not quote:
                quote = "USDT"  # 기본 쿼트

            result = f"{base}{quote}"
            return result

        except Exception:
            return symbol

    def _parse_symbol(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """심볼을 베이스와 쿼트로 분리"""

        # Quote 화폐로 끝나는지 확인
        for quote in sorted(self.quote_currencies, key=len, reverse=True):
            if symbol.endswith(quote):
                base = symbol[: -len(quote)]

                if base in self.base_coins or (2 <= len(base) <= 10 and base.isalpha()):
                    return base, quote

        # 베이스 코인으로 시작하는지 확인
        for base in sorted(self.base_coins, key=len, reverse=True):
            if symbol.startswith(base):
                remaining = symbol[len(base) :]

                if remaining in self.quote_currencies:
                    return base, remaining

        # 길이 기반 추정
        if 6 <= len(symbol) <= 10:
            for split_point in [3, 4, 5]:
                if split_point < len(symbol):
                    base = symbol[:split_point]
                    quote = symbol[split_point:]

                    if quote in self.quote_currencies:
                        return base, quote

        return None, None


# Bybit용 전역 인스턴스
bybit_normalizer = BybitSymbolNormalizer()

# ============================================================================
# 로깅 시스템
# ============================================================================


class SafeQueueHandler(logging.handlers.QueueHandler):
    """안전한 큐핸들러"""

    def prepare(self, record):
        try:
            msg = self.format(record)
            safe_record = logging.LogRecord(
                name=record.name,
                level=record.levelno,
                pathname="",
                lineno=0,
                msg=msg,
                args=(),
                exc_info=None,
            )
            return safe_record
        except Exception:
            try:
                basic_msg = f"{record.levelname}: {record.getMessage()}"
            except:
                basic_msg = f"{record.levelname}: [포맷팅 오류]"

            return logging.LogRecord(
                name=record.name,
                level=record.levelno,
                pathname="",
                lineno=0,
                msg=basic_msg,
                args=(),
                exc_info=None,
            )


def setup_logging(log_level: str = "INFO", gui_queue: Optional[queue.Queue] = None):
    """통합 로깅 시스템 설정"""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 기존 핸들러 정리
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # 파일 핸들러
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / "stargate.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
    except Exception as e:
        logging.error(f"파일 로깅 설정 실패: {e}")

    # GUI 큐핸들러
    if gui_queue:
        try:
            queue_handler = SafeQueueHandler(gui_queue)
            queue_handler.setFormatter(formatter)
            queue_handler.setLevel(logging.INFO)
            root_logger.addHandler(queue_handler)
        except Exception as e:
            logging.error(f"큐핸들러 설정 실패: {e}")

    logger = logging.getLogger("stargate.main")
    logger.info("로깅 시스템 초기화 완료")
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """네임스페이스별 로거 반환"""
    return logging.getLogger(f"stargate.{name}")


def mask_sensitive_data(data: str, show_last: int = 4) -> str:
    """민감한 데이터 마스킹"""
    if not data or len(data) <= show_last:
        return "***"
    return "***" + data[-show_last:]


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """재시도 데코레이터"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger("retry")
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (backoff**attempt)
                        logger.warning(
                            f"[{func.__name__}] 시도 {attempt + 1}/{max_retries} 실패, {wait_time:.1f}초 후 재시도: {e}"
                        )
                        time.sleep(wait_time)

            raise last_exception

        return wrapper

    return decorator


# ============================================================================
# 설정 관리자
# ============================================================================


@dataclass
class ServerConfig:
    """서버 설정"""

    host: str = "0.0.0.0"
    port: int = 5000
    webhook_secret: str = ""
    ip_whitelist: List[str] = field(default_factory=list)


@dataclass
class ExchangeKeys:
    """거래소 API 키"""

    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""


@dataclass
class TradingConfig:
    """멀티거래소 트레이딩 설정"""

    SELECTED_EXCHANGES: List[str] = field(default_factory=lambda: ["bybit"])
    QUOTE_PCT: float = 30.0
    LEVERAGE: float = 5.0
    MIN_SIGNAL_INTERVAL_SEC: int = 45
    USE_FIXED_STOP: bool = True
    FIXED_STOP_PCT: float = 1.5
    USE_TRAILING_STOP: bool = True
    TRAIL_TRIGGER_PCT: float = 0.8
    TRAIL_CALLBACK_PCT: float = 0.3
    COOLDOWN_AFTER_CLOSE_SEC: int = 30
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    # 새로운 설정 추가
    ENABLE_MIN_QTY_FALLBACK: bool = True  # 최소수량 진입 허용
    MAX_FALLBACK_RISK_PCT: float = 50.0  # 최대 허용 리스크
    # 리스크 보호장치 설정 추가
    ALLOW_SL_ONLY: bool = True  # 손절만 성공시 허용 여부
    MAX_PROTECTION_RETRIES: int = 5  # 보호장치 재시도 횟수
    PROTECTION_RETRY_INTERVAL: float = 5.0  # 보호장치 재시도 간격(초)
    SERVER: ServerConfig = field(default_factory=ServerConfig)
    EXCHANGE_KEYS: Dict[str, ExchangeKeys] = field(default_factory=dict)

    def __post_init__(self):
        # 지원되는 모든 거래소에 대해 키 구조 초기화
        for exchange in SUPPORTED_EXCHANGES:
            if exchange not in self.EXCHANGE_KEYS:
                self.EXCHANGE_KEYS[exchange] = ExchangeKeys()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingConfig":
        if "SERVER" in data and isinstance(data["SERVER"], dict):
            data["SERVER"] = ServerConfig(**data["SERVER"])
        if "EXCHANGE_KEYS" in data and isinstance(data["EXCHANGE_KEYS"], dict):
            exchange_keys = {}
            for exchange, keys in data["EXCHANGE_KEYS"].items():
                if isinstance(keys, dict):
                    # passphrase 안전 처리
                    passphrase = keys.get("passphrase", "")
                    if not isinstance(passphrase, str) or passphrase is None:
                        passphrase = ""
                    keys["passphrase"] = passphrase
                    exchange_keys[exchange] = ExchangeKeys(**keys)
                else:
                    exchange_keys[exchange] = keys
            data["EXCHANGE_KEYS"] = exchange_keys
        return cls(**data)

    def get_safe_dict(self) -> Dict[str, Any]:
        """민감한 정보가 마스킹된 딕셔너리 반환"""
        data = self.to_dict()
        if "EXCHANGE_KEYS" in data:
            for exchange, keys in data["EXCHANGE_KEYS"].items():
                if isinstance(keys, dict):
                    for key_name in ["api_key", "api_secret"]:
                        if key_name in keys and keys[key_name]:
                            keys[key_name] = mask_sensitive_data(keys[key_name])
        if "SERVER" in data and isinstance(data["SERVER"], dict):
            if data["SERVER"].get("webhook_secret"):
                data["SERVER"]["webhook_secret"] = mask_sensitive_data(
                    data["SERVER"]["webhook_secret"]
                )
        return data


class ConfigManager:
    """설정 관리자"""

    def __init__(self):
        self.config_dir = Path(__file__).parent / "config"
        self.config_file = self.config_dir / "config.json"
        self.backup_file = self.config_dir / "config.backup.json"
        self.config: TradingConfig = self._load_config()
        self.logger = get_logger("config")

    def _load_config(self) -> TradingConfig:
        """설정 파일 로드"""
        try:
            if self.config_file.exists():
                with open(self.config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # EXCHANGE_KEYS 안전 처리
                if "EXCHANGE_KEYS" in data:
                    for exchange, keys_data in data["EXCHANGE_KEYS"].items():
                        if isinstance(keys_data, dict):
                            if (
                                "passphrase" not in keys_data
                                or not keys_data["passphrase"]
                            ):
                                keys_data["passphrase"] = ""

                config = TradingConfig.from_dict(data)
                logging.info(f"설정 로드 완료: {self.config_file}")
                return config
            else:
                logging.info("기본 설정으로 시작")
                return TradingConfig()
        except Exception as e:
            logging.error(f"설정 로드 실패: {e}")
            if self.backup_file.exists():
                try:
                    with open(self.backup_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # 백업에서도 passphrase 안전 처리
                    if "EXCHANGE_KEYS" in data:
                        for exchange, keys_data in data["EXCHANGE_KEYS"].items():
                            if isinstance(keys_data, dict):
                                if (
                                    "passphrase" not in keys_data
                                    or not keys_data["passphrase"]
                                ):
                                    keys_data["passphrase"] = ""

                    config = TradingConfig.from_dict(data)
                    logging.warning("백업 설정으로 복구됨")
                    return config
                except Exception as backup_e:
                    logging.error(f"백업 설정 로드도 실패: {backup_e}")

            logging.warning("기본 설정으로 폴백")
            return TradingConfig()

    def save_config(self) -> bool:
        """설정 파일 저장"""
        try:
            if self.config_file.exists():
                import shutil

                shutil.copy2(self.config_file, self.backup_file)

            self.config_dir.mkdir(exist_ok=True)
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)

            self.logger.info(f"설정 저장 완료: {self.config_file}")
            return True
        except Exception as e:
            self.logger.error(f"설정 저장 실패: {e}")
            return False

    def get_exchange_keys(self, exchange: str) -> Optional[ExchangeKeys]:
        """거래소 API 키 조회"""
        return self.config.EXCHANGE_KEYS.get(exchange)

    def set_exchange_keys(
        self, exchange: str, api_key: str, api_secret: str, passphrase: str = ""
    ):
        """거래소 API 키 설정"""
        self.config.EXCHANGE_KEYS[exchange] = ExchangeKeys(
            api_key=api_key, api_secret=api_secret, passphrase=passphrase
        )


# 전역 설정 관리자
config_manager = ConfigManager()


def get_config() -> TradingConfig:
    return config_manager.config


def save_config() -> bool:
    return config_manager.save_config()


# ============================================================================
# 보안 및 유틸리티 함수
# ============================================================================


def check_ip_whitelist(remote_ip: str, whitelist: list) -> bool:
    """IP 화이트리스트 검증"""
    if not whitelist:
        return True

    logger = get_logger("security")
    try:
        import ipaddress

        remote = ipaddress.ip_address(remote_ip)
        for allowed in whitelist:
            try:
                if "/" in allowed:
                    if remote in ipaddress.ip_network(allowed, strict=False):
                        return True
                else:
                    if remote == ipaddress.ip_address(allowed):
                        return True
            except Exception as e:
                logger.warning(f"IP 파싱 오류 {allowed}: {e}")
                continue

        logger.warning(f"IP 차단됨: {remote_ip}")
        return False
    except Exception as e:
        logger.error(f"IP 화이트리스트 검증 오류: {e}")
        return False


@retry_on_failure(max_retries=3, delay=1.0)
def detect_ngrok_url() -> Optional[str]:
    """ngrok URL 자동 감지"""
    try:
        resp = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            tunnels = data.get("tunnels", [])
            for tunnel in tunnels:
                if tunnel.get("proto") == "https":
                    url = tunnel.get("public_url")
                    if url:
                        logging.info(f"ngrok HTTPS URL 감지: {url}")
                        return url
            for tunnel in tunnels:
                if tunnel.get("proto") == "http":
                    url = tunnel.get("public_url")
                    if url:
                        logging.info(f"ngrok HTTP URL 감지: {url}")
                        return url
        return None
    except Exception as e:
        logging.warning(f"ngrok 감지 실패: {e}")
        return None


def verify_webhook_signature(
    raw_body: bytes, signature: str, secret: str, ts: str, window_ms: int = 30000
) -> bool:
    """HMAC 웹훅 서명 검증"""
    if not secret:
        return True

    logger = get_logger("security")
    try:
        if not signature or not ts:
            logger.warning("서명 또는 타임스탬프 누락")
            return False

        now_ms = int(time.time() * 1000)
        ts_ms = int(ts)
        time_diff = abs(now_ms - ts_ms)

        if time_diff > window_ms:
            logger.warning(f"타임스탬프 윈도우 초과: {time_diff}ms")
            return False

        payload = f"{ts}:{raw_body.decode('utf-8', 'ignore')}"
        expected = hmac.new(
            secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        if "=" in signature:
            signature = signature.split("=", 1)[1]

        is_valid = hmac.compare_digest(expected, signature)
        if not is_valid:
            logger.warning("웹훅 서명 검증 실패")

        return is_valid
    except Exception as e:
        logger.error(f"서명 검증 오류: {e}")
        return False


# ============================================================================
# 거래소 추상화 및 구현체
# ============================================================================


@dataclass
class Position:
    """포지션 정보"""

    symbol: str
    side: str  # "long", "short", "flat"
    qty: float
    entry_price: float
    mark_price: float = 0.0
    unrealized_pnl: float = 0.0
    margin: float = 0.0
    leverage: float = 1.0
    category: str = "linear"


@dataclass
class OrderResult:
    """주문 결과"""

    success: bool
    order_id: str = ""
    message: str = ""
    filled_qty: float = 0.0
    avg_price: float = 0.0
    data: Dict[str, Any] = None


@dataclass
class Balance:
    """잔고 정보"""

    total: float
    available: float
    currency: str = "USDT"


@dataclass
class ContractInfo:
    """계약 정보"""

    symbol: str
    base_currency: str
    quote_currency: str
    tick_size: float
    step_size: float
    min_qty: float
    max_qty: float = float("inf")
    min_notional: float = 0.0


class ExchangeBase(ABC):
    """거래소 추상화 기본 클래스 - 보호장치 인터페이스 통일"""

    def __init__(self, name: str, api_key: str, api_secret: str, testnet: bool = False):
        self.name = name
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._contract_cache: Dict[str, ContractInfo] = {}
        self.logger = get_logger(f"exchange.{name}")
        self.logger.info(f"거래소 클라이언트 초기화: {name}")

    def normalize_symbol(self, symbol: str) -> str:
        """심볼 정규화 - 각 거래소에서 오버라이드"""
        return symbol

    @abstractmethod
    def validate_credentials(self) -> Tuple[bool, str]:
        pass

    @abstractmethod
    def get_account_balance(self) -> Balance:
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Position:
        pass

    @abstractmethod
    def get_all_positions(self) -> List[Position]:
        pass

    @abstractmethod
    def get_ticker_price(self, symbol: str) -> Tuple[float, float]:
        pass

    @abstractmethod
    def get_contract_info(self, symbol: str) -> ContractInfo:
        pass

    @abstractmethod
    def set_leverage(self, symbol: str, leverage: float) -> bool:
        pass

    @abstractmethod
    def place_market_order(
        self, symbol: str, side: str, qty: float, reduce_only: bool = False
    ) -> OrderResult:
        pass

    @abstractmethod
    def close_all_positions(self, symbol: str = None) -> Dict[str, OrderResult]:
        pass

    # ============================================================================
    # 보호장치 표준 인터페이스 - 모든 거래소 구현 필수
    # ============================================================================

    @abstractmethod
    def set_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        sl_pct: float,
        tick_size: float,
    ) -> bool:
        """표준 손절 설정 인터페이스"""
        pass

    @abstractmethod
    def set_trailing_stop(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        ts_trigger_pct: float,
        ts_callback_pct: float,
        tick_size: float,
    ) -> bool:
        """표준 트레일링 스톱 설정 인터페이스"""
        pass

    @abstractmethod
    def close_position(self, symbol: str) -> bool:
        """단일 포지션 강제 청산 인터페이스"""
        pass

    # ============================================================================
    # 공통 유틸리티 메소드 (기존 유지)
    # ============================================================================

    def adjust_quantity(
        self,
        symbol: str,
        qty: float,
        enable_fallback: bool = True,
        max_risk_pct: float = 50.0,
        total_balance: float = 0.0,
    ) -> float:
        """수량 조정 - Bybit min_notional 요구사항 포함"""
        try:
            contract = self.get_contract_info(symbol)

            # 현재가 조회 (notional 계산용)
            bid, ask = self.get_ticker_price(symbol)
            current_price = (bid + ask) / 2 if bid > 0 and ask > 0 else ask or bid
            if current_price <= 0:
                self.logger.error(f"현재가 조회 실패: {symbol}")
                return 0.0

            # 디버깅 로그
            original_qty = qty
            notional_value = qty * current_price
            self.logger.info(
                f"수량 조정: {symbol} qty={qty:.6f} price={current_price:.4f} notional=${notional_value:.2f}"
            )
            self.logger.info(
                f"요구사항: min_qty={contract.min_qty} min_notional=${contract.min_notional}"
            )

            # 1차: 최소 수량 체크
            if qty < contract.min_qty:
                if not enable_fallback:
                    self.logger.warning(f"수량 부족: {qty:.6f} < {contract.min_qty}")
                    return 0.0
                qty = contract.min_qty
                self.logger.info(f"최소수량 적용: {original_qty:.6f} -> {qty:.6f}")

            # 2차: 최소 거래금액 체크
            notional_value = qty * current_price
            if notional_value < contract.min_notional:
                required_qty = contract.min_notional / current_price

                if not enable_fallback:
                    self.logger.warning(
                        f"거래금액 부족: ${notional_value:.2f} < ${contract.min_notional}"
                    )
                    return 0.0

                qty = max(qty, required_qty)
                self.logger.info(
                    f"최소거래금액 적용: notional=${notional_value:.2f} -> qty={qty:.6f}"
                )

            # 3차: Step size 조정
            if contract.step_size > 0:
                step_decimal = Decimal(str(contract.step_size))
                qty_decimal = Decimal(str(qty))
                # 올림으로 조정 (최소 요구사항을 만족하기 위해)
                multiplier = (qty_decimal / step_decimal).quantize(
                    Decimal("1"), rounding=ROUND_UP
                )
                qty = float(multiplier * step_decimal)

            # 최종 검증
            final_notional = qty * current_price
            if qty < contract.min_qty or final_notional < contract.min_notional:
                self.logger.error(
                    f"최종 검증 실패: qty={qty:.6f} notional=${final_notional:.2f}"
                )
                return 0.0

            if qty != original_qty:
                self.logger.info(
                    f"수량 조정 완료: {original_qty:.6f} -> {qty:.6f} (${final_notional:.2f})"
                )

            return qty

        except Exception as e:
            self.logger.error(f"수량 조정 실패 {symbol}: {e}")
            return 0.0

    def adjust_price(self, symbol: str, price: float) -> float:
        """가격 조정"""
        try:
            contract = self.get_contract_info(symbol)
            if contract.tick_size > 0:
                tick_decimal = Decimal(str(contract.tick_size))
                price_decimal = Decimal(str(price))
                adjusted_price = float(
                    price_decimal.quantize(tick_decimal, rounding=ROUND_DOWN)
                )
                return adjusted_price
            return price
        except Exception as e:
            self.logger.error(f"가격 조정 실패 {symbol}: {e}")
            return price

    def get_position_side_enum(self, side: str) -> str:
        """포지션 사이드 변환"""
        side_lower = side.lower().strip()

        if side_lower in ["long", "buy", "1"]:
            return "Buy"
        elif side_lower in ["short", "sell", "2"]:
            return "Sell"
        else:
            raise ValueError(f"지원되지 않는 side 값: '{side}'")

    def get_opposite_side(self, side: str) -> str:
        """반대 사이드 반환"""
        if side.lower() in ["long", "buy"]:
            return "short"
        elif side.lower() in ["short", "sell"]:
            return "long"
        else:
            return "flat"


class BybitExchange(ExchangeBase):
    """Bybit V5 API 구현체 - 보호장치 인터페이스 표준화"""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        super().__init__("bybit", api_key, api_secret, testnet)
        self.base_url = (
            "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
        )
        self.recv_window = 20000
        self.categories = [("linear", "USDT"), ("linear", "USDC")]

    def normalize_symbol(self, symbol: str) -> str:
        """Bybit 전용 심볼 정규화"""
        original_symbol = symbol
        normalized_symbol = bybit_normalizer.normalize_symbol(symbol)

        # 변환 로깅 (원본과 다를 때만)
        if original_symbol != normalized_symbol:
            self.logger.info(f"심볼 변환: {original_symbol} -> {normalized_symbol}")

        return normalized_symbol

    def _sign_request(
        self, method: str, endpoint: str, params: str = "", body: str = ""
    ) -> Dict[str, str]:
        """API 요청 서명 생성"""
        timestamp = str(int(time.time() * 1000))
        param_str = params or ""
        payload = f"{timestamp}{self.api_key}{self.recv_window}{param_str}{body}"
        signature = hmac.new(
            self.api_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        return {
            "Content-Type": "application/json",
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": str(self.recv_window),
            "X-BAPI-SIGN": signature,
        }

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        body: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """API 요청 실행"""
        query_str = ""
        if params:
            query_str = "&".join(
                [f"{k}={v}" for k, v in params.items() if v is not None]
            )

        body_str = ""
        if body:
            body_str = json.dumps(body)

        url = f"{self.base_url}{endpoint}"
        if query_str:
            url += f"?{query_str}"

        headers = self._sign_request(method, endpoint, query_str, body_str)

        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=15)
        else:
            response = requests.post(url, headers=headers, json=body, timeout=15)

        if response.status_code == 200:
            data = response.json()
            if data.get("retCode") == 0:
                return data
            else:
                raise Exception(
                    f"Bybit API Error {data.get('retCode')}: {data.get('retMsg')}"
                )
        else:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

    def validate_credentials(self) -> Tuple[bool, str]:
        """API 자격증명 검증"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                time.sleep(0.1)

                data = self._make_request(
                    "GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"}
                )
                if data.get("retCode") == 0:
                    mode = "testnet" if self.testnet else "live"
                    return True, f"OK (mode={mode})"
                else:
                    return False, f"API Error: {data.get('retMsg')}"
            except Exception as e:
                if (
                    "invalid request" in str(e).lower()
                    and "timestamp" in str(e).lower()
                ):
                    if attempt < max_retries - 1:
                        wait_time = 1.0 * (attempt + 1)
                        time.sleep(wait_time)
                        continue
                return False, f"Connection error: {e}"

        return False, "Max retries exceeded"

    def get_account_balance(self) -> Balance:
        """계정 잔고 조회"""
        try:
            data = self._make_request(
                "GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"}
            )
            if data.get("retCode") != 0:
                return Balance(0.0, 0.0, "USDT")

            for coin_info in data["result"]["list"][0]["coin"]:
                if coin_info["coin"] in ("USDT", "USD"):
                    available = float(coin_info.get("availableToWithdraw", "0") or "0")
                    total = float(coin_info.get("walletBalance", "0") or "0")
                    currency = coin_info["coin"]

                    if available > 0:
                        return Balance(total, available, currency)
                    elif total > 0:
                        return Balance(total, total, currency)

            return Balance(0.0, 0.0, "USDT")
        except Exception as e:
            self.logger.error(f"잔고 조회 오류: {e}")
            return Balance(0.0, 0.0, "USDT")

    def get_position(self, symbol: str) -> Position:
        """단일 포지션 조회"""
        symbol = self.normalize_symbol(symbol)

        for category, settle_coin in self.categories:
            try:
                params = {"category": category, "symbol": symbol}
                if settle_coin:
                    params["settleCoin"] = settle_coin

                data = self._make_request("GET", "/v5/position/list", params)
                if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                    for pos_data in data["result"]["list"]:
                        size = float(pos_data.get("size", "0") or "0")
                        if size != 0:
                            side = "long" if pos_data.get("side") == "Buy" else "short"
                            return Position(
                                symbol=symbol,
                                side=side,
                                qty=abs(size),
                                entry_price=float(pos_data.get("avgPrice", "0") or "0"),
                                mark_price=float(pos_data.get("markPrice", "0") or "0"),
                                unrealized_pnl=float(
                                    pos_data.get("unrealisedPnl", "0") or "0"
                                ),
                                leverage=float(pos_data.get("leverage", "1") or "1"),
                                category=category,
                            )
            except Exception as e:
                self.logger.warning(f"포지션 조회 오류 {category} for {symbol}: {e}")
                continue

        return Position(symbol, "flat", 0.0, 0.0, category="linear")

    def get_all_positions(self) -> List[Position]:
        """모든 활성 포지션 조회"""
        positions = []
        seen = set()

        for category, settle_coin in self.categories:
            try:
                params = {"category": category}
                if settle_coin:
                    params["settleCoin"] = settle_coin

                data = self._make_request("GET", "/v5/position/list", params)
                if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                    for pos_data in data["result"]["list"]:
                        symbol = self.normalize_symbol(pos_data.get("symbol", ""))
                        size = float(pos_data.get("size", "0") or "0")

                        if size == 0:
                            continue

                        side = "long" if pos_data.get("side") == "Buy" else "short"
                        key = (symbol, side)

                        if key in seen:
                            continue

                        position = Position(
                            symbol=symbol,
                            side=side,
                            qty=abs(size),
                            entry_price=float(pos_data.get("avgPrice", "0") or "0"),
                            mark_price=float(pos_data.get("markPrice", "0") or "0"),
                            unrealized_pnl=float(
                                pos_data.get("unrealisedPnl", "0") or "0"
                            ),
                            leverage=float(pos_data.get("leverage", "1") or "1"),
                            category=category,
                        )
                        positions.append(position)
                        seen.add(key)
            except Exception as e:
                self.logger.error(f"포지션 조회 오류 {category}: {e}")
                continue

        return positions

    def get_ticker_price(self, symbol: str) -> Tuple[float, float]:
        """현재가 조회"""
        symbol = self.normalize_symbol(symbol)
        try:
            data = self._make_request(
                "GET",
                "/v5/market/orderbook",
                {"category": "linear", "symbol": symbol, "limit": 1},
            )
            if data.get("retCode") == 0:
                result = data["result"]
                bid = float(result["b"][0][0]) if result.get("b") else 0.0
                ask = float(result["a"][0][0]) if result.get("a") else 0.0
                return bid, ask
            else:
                return 0.0, 0.0
        except Exception as e:
            self.logger.error(f"현재가 조회 오류 {symbol}: {e}")
            return 0.0, 0.0

    def get_contract_info(self, symbol: str) -> ContractInfo:
        """계약 정보 조회"""
        symbol = self.normalize_symbol(symbol)
        if symbol in self._contract_cache:
            return self._contract_cache[symbol]

        try:
            data = self._make_request(
                "GET",
                "/v5/market/instruments-info",
                {"category": "linear", "symbol": symbol},
            )
            if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                info = data["result"]["list"][0]
                lot_filter = info.get("lotSizeFilter", {}) or {}
                price_filter = info.get("priceFilter", {}) or {}

                # Normalize filter fields with safe fallbacks across product types
                qty_step = lot_filter.get("qtyStep") or lot_filter.get("stepSize") or "0.001"
                min_qty  = lot_filter.get("minOrderQty") or lot_filter.get("minQty") or "0.001"
                min_notional = (lot_filter.get("minNotional")
                                or lot_filter.get("minOrderAmt")
                                or lot_filter.get("minValue")
                                or "5")

                contract = ContractInfo(
                    symbol=symbol,
                    base_currency=info.get("baseCoin", ""),
                    quote_currency=info.get("quoteCoin", "USDT"),
                    tick_size=float(price_filter.get("tickSize") or "0.01"),
                    step_size=float(qty_step),
                    min_qty=float(min_qty),
                    min_notional=float(min_notional),
                )
                self._contract_cache[symbol] = contract
                return contract
        except Exception as e:
            self.logger.error(f"계약 정보 조회 오류 {symbol}: {e}")

        # 기본값 반환
        return ContractInfo(
            symbol,
            "",
            "USDT",
            DEFAULT_VALUES["tick_size"],
            DEFAULT_VALUES["step_size"],
            DEFAULT_VALUES["min_qty"],
        )

    def set_leverage(self, symbol: str, leverage: float) -> bool:
        """레버리지 설정 - 110043 오류 허용 처리"""
        symbol = self.normalize_symbol(symbol)
        try:
            body = {
                "category": "linear",
                "symbol": symbol,
                "buyLeverage": str(leverage),
                "sellLeverage": str(leverage),
            }
            data = self._make_request("POST", "/v5/position/set-leverage", body=body)

            # 110043 (leverage not modified) 오류 허용
            if data.get("retCode") == 0 or data.get("retCode") == 110043:
                if data.get("retCode") == 110043:
                    self.logger.debug(f"레버리지 이미 설정됨: {symbol} x{leverage}")
                return True
            else:
                self.logger.warning(f"레버리지 설정 실패: {data}")
                return False
        except Exception as e:
            self.logger.error(f"레버리지 설정 오류 {symbol}: {e}")
            return False

    def place_market_order(
        self, symbol: str, side: str, qty: float, reduce_only: bool = False
    ) -> OrderResult:
        """마켓 주문 실행 - 중복 수량 조정 제거"""
        symbol = self.normalize_symbol(symbol)

        # 중복 수량 조정 제거: 상위에서 이미 처리됨을 가정
        # reduce_only 주문만 최소 검증
        if reduce_only and qty <= 0:
            return OrderResult(False, "", "청산 수량이 유효하지 않음")
        elif not reduce_only and qty <= 0:
            return OrderResult(False, "", "신규 주문 수량이 유효하지 않음")

        try:
            order_side = self.get_position_side_enum(side)

            body = {
                "category": "linear",
                "symbol": symbol,
                "side": order_side,
                "orderType": "Market",
                "qty": str(qty),
                "timeInForce": "ImmediateOrCancel",
                "reduceOnly": reduce_only,
            }

            data = self._make_request("POST", "/v5/order/create", body=body)

            if data.get("retCode") == 0:
                order_id = data["result"]["orderId"]
                return OrderResult(True, order_id, "주문 성공", qty, 0.0, data)
            else:
                error_msg = data.get("retMsg", "알 수 없는 오류")
                self.logger.error(f"마켓 주문 실패: {error_msg}")
                return OrderResult(False, "", error_msg)

        except Exception as e:
            self.logger.error(f"마켓 주문 예외: {e}")
            return OrderResult(False, "", str(e))

    def close_all_positions(self, symbol: str = None) -> Dict[str, OrderResult]:
        """포지션 일괄 청산"""
        results = {}

        if symbol:
            try:
                position = self.get_position(symbol)
                if position.side == "flat":
                    results[symbol] = OrderResult(False, "", "청산할 포지션 없음")
                    return results

                result = self._close_single_position(position)
                results[symbol] = result
            except Exception as e:
                results[symbol] = OrderResult(False, "", str(e))
        else:
            positions = self.get_all_positions()
            for position in positions:
                try:
                    result = self._close_single_position(position)
                    results[position.symbol] = result
                except Exception as e:
                    results[position.symbol] = OrderResult(False, "", str(e))

        return results

    def _close_single_position(self, position: Position) -> OrderResult:
        """단일 포지션 청산"""
        if position.side == "flat" or position.qty <= 0:
            return OrderResult(False, "", "청산할 포지션 없음")

        opposite_side = self.get_opposite_side(position.side)
        close_side = self.get_position_side_enum(opposite_side)
        category = position.category or "linear"

        try:
            position_idx = 0
            adj_qty = self.adjust_quantity(
                position.symbol, position.qty, enable_fallback=False
            )
            body = {
                "category": category,
                "symbol": position.symbol,
                "side": close_side,
                "orderType": "Market",
                "qty": str(adj_qty),
                "timeInForce": "ImmediateOrCancel",
                "reduceOnly": True,
                "positionIdx": position_idx,
            }

            data = self._make_request("POST", "/v5/order/create", body=body)

            if data.get("retCode") == 0:
                order_id = data["result"]["orderId"]
                return OrderResult(
                    True, order_id, "포지션 청산 성공", position.qty, 0.0, data
                )

            # 폴백 모드
            body_fb = {
                "category": category,
                "symbol": position.symbol,
                "side": close_side,
                "orderType": "Market",
                "qty": "0",
                "timeInForce": "ImmediateOrCancel",
                "reduceOnly": True,
                "closeOnTrigger": True,
                "positionIdx": position_idx,
            }

            data_fb = self._make_request("POST", "/v5/order/create", body=body_fb)
            if data_fb.get("retCode") == 0:
                order_id = data_fb["result"]["orderId"]
                return OrderResult(
                    True, order_id, "포지션 청산 성공", position.qty, 0.0, data_fb
                )

            error_msg = data_fb.get("retMsg", "알 수 없는 오류")
            return OrderResult(False, "", error_msg)
        except Exception as e:
            return OrderResult(False, "", str(e))

    # ============================================================================
    # 보호장치 표준 인터페이스 구현 - Bybit 특화
    # ============================================================================

    def set_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        sl_pct: float,
        tick_size: float,
    ) -> bool:
        """표준 손절 설정 - Bybit V5 API (멱등성 강화)"""
        symbol = self.normalize_symbol(symbol)

        try:
            # 손절가 계산
            if side.lower() == "long":
                stop_price = entry_price * (1 - sl_pct / 100.0)
            else:
                stop_price = entry_price * (1 + sl_pct / 100.0)

            # 틱 조정
            stop_price = round_to_tick(stop_price, tick_size, side)

            # 포지션 확인
            pos = self.get_position(symbol)
            if pos.side == "flat":
                return True

            position_idx = 0
            body = {
                "category": pos.category or "linear",
                "symbol": symbol,
                "stopLoss": str(stop_price),
                "slTriggerBy": "MarkPrice",
                "positionIdx": position_idx,
            }

            data = self._make_request("POST", "/v5/position/trading-stop", body=body)

            # 멱등성 강화: retCode in (0, 34040) → 성공 처리
            if data.get("retCode") == 0:
                self.logger.info(f"[BYBIT] 손절 설정 성공: {symbol} @ {stop_price}")
                return True
            elif data.get("retCode") == 34040:
                # 34040 = not modified (이미 설정됨)
                self.logger.info(f"[BYBIT] 손절 이미 설정됨 (retCode=34040): {symbol}")
                return True
            else:
                self.logger.error(f"[BYBIT] 손절 설정 실패: {data}")
                return False

        except Exception as e:
            self.logger.error(f"[BYBIT] 손절 설정 오류: {e}")
            return False

    def set_trailing_stop(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        ts_trigger_pct: float,
        ts_callback_pct: float,
        tick_size: float,
    ) -> bool:
        """표준 트레일링 스톱 설정 - Bybit V5 API (멱등성 강화)"""
        symbol = self.normalize_symbol(symbol)

        try:
            pos = self.get_position(symbol)
            if pos.side == "flat":
                return True

            category = pos.category or "linear"
            position_idx = 0

            # 현재가 조회
            bid, ask = self.get_ticker_price(symbol)
            ref_px = (bid + ask) / 2 if bid > 0 and ask > 0 else ask or bid
            if ref_px <= 0:
                return False

            # 활성화 가격 계산
            if side.lower() == "long":
                active_price = entry_price * (1.0 + float(ts_trigger_pct) / 100.0)
            else:
                active_price = entry_price * (1.0 - float(ts_trigger_pct) / 100.0)

            # 거리 계산
            distance = ref_px * (float(ts_callback_pct) / 100.0)

            # 틱 조정
            active_price = round_to_tick(active_price, tick_size, side)
            distance = round_to_tick(distance, tick_size, side)
            distance = max(distance, tick_size)  # 최소 1틱

            body = {
                "category": category,
                "symbol": symbol,
                "trailingStop": str(distance),
                "activePrice": str(active_price),
                "slTriggerBy": "MarkPrice",
                "positionIdx": position_idx,
            }

            data = self._make_request("POST", "/v5/position/trading-stop", body=body)

            # 멱등성 강화: retCode in (0, 34040) → 성공 처리
            if data.get("retCode") == 0:
                self.logger.info(
                    f"[BYBIT] 트레일링 설정 성공: {symbol} active={active_price} distance={distance}"
                )
                return True
            elif data.get("retCode") == 34040:
                # 34040 = not modified (이미 설정됨)
                self.logger.info(
                    f"[BYBIT] 트레일링 이미 설정됨 (retCode=34040): {symbol}"
                )
                return True
            else:
                self.logger.error(f"[BYBIT] 트레일링 설정 실패: {data}")
                return False

        except Exception as e:
            self.logger.error(f"[BYBIT] 트레일링 설정 오류: {e}")
            return False

    def close_position(self, symbol: str) -> bool:
        """단일 포지션 강제 청산"""
        try:
            results = self.close_all_positions(symbol)
            result = results.get(symbol)
            if result and result.success:
                self.logger.info(f"[BYBIT] 포지션 강제 청산 성공: {symbol}")
                return True
            else:
                error_msg = result.message if result else "알 수 없는 오류"
                self.logger.error(
                    f"[BYBIT] 포지션 강제 청산 실패: {symbol} - {error_msg}"
                )
                return False
        except Exception as e:
            self.logger.error(f"[BYBIT] 포지션 강제 청산 예외: {symbol} - {e}")
            return False


class BitgetExchange(ExchangeBase):
    """Bitget REST API 네이티브 구현체 - 보호장치 인터페이스 표준화"""

    def __init__(
        self, api_key: str, api_secret: str, passphrase: str = "", testnet: bool = False
    ):
        super().__init__("bitget", api_key, api_secret, testnet)
        self.passphrase = str(passphrase) if passphrase else ""
        self.base_url = "https://api.bitget.com"
        self._symbol_cache = {}  # 실제 심볼 캐시

        if not self.passphrase:
            raise ValueError("Bitget requires passphrase for API access")

        # 시작시 실제 심볼 목록 로드
        self._load_available_symbols()

        self.logger.info("Bitget 네이티브 클라이언트 초기화 완료")

    def _load_available_symbols(self):
        """Bitget의 실제 사용가능한 심볼 목록을 로드"""
        try:
            data = self._make_request(
                "GET", "/api/v2/mix/market/contracts", {"productType": "USDT-FUTURES"}
            )
            if data.get("code") == "00000" and data.get("data"):
                for contract in data["data"]:
                    symbol = contract.get("symbol", "")
                    if symbol:
                        # 여러 형태의 키로 저장
                        self._symbol_cache[symbol] = symbol
                        # .P 접미사 버전도 매핑
                        if not symbol.endswith(".P"):
                            self._symbol_cache[f"{symbol}.P"] = symbol
                        # 기타 변형들
                        base_coin = contract.get("baseCoin", "")
                        if base_coin:
                            self._symbol_cache[f"{base_coin}USDT"] = symbol
                            self._symbol_cache[f"{base_coin}USDT.P"] = symbol
                            self._symbol_cache[f"{base_coin}-USDT"] = symbol

                self.logger.info(
                    f"Bitget 심볼 캐시 로드 완료: {len(self._symbol_cache)}개"
                )

                # 주요 심볼들 로그 출력
                main_symbols = ["XRPUSDT", "BTCUSDT", "ETHUSDT"]
                for sym in main_symbols:
                    if sym in self._symbol_cache:
                        self.logger.info(f"  {sym} -> {self._symbol_cache[sym]}")

        except Exception as e:
            self.logger.error(f"심볼 목록 로드 실패: {e}")
            # 기본 매핑으로 fallback
            self._symbol_cache = {
                "XRPUSDT": "XRPUSDT",
                "XRPUSDT.P": "XRPUSDT",
                "BTCUSDT": "BTCUSDT",
                "BTCUSDT.P": "BTCUSDT",
                "ETHUSDT": "ETHUSDT",
                "ETHUSDT.P": "ETHUSDT",
            }

    def normalize_symbol(self, symbol: str) -> str:
        """Bitget 전용 실시간 심볼 정규화 - API 기반 정확한 매핑"""
        if not symbol:
            return ""

        original_symbol = symbol
        symbol = symbol.upper().strip()

        # 1차: 캐시에서 직접 찾기
        if symbol in self._symbol_cache:
            result = self._symbol_cache[symbol]
            if original_symbol != result:
                self.logger.info(f"심볼 변환: {original_symbol} -> {result}")
            return result

        # 2차: Perpetual 접미사 제거 후 찾기
        clean_symbol = symbol
        for suffix in [".P", ".PERP", "_PERP", "-PERP", "PERP"]:
            if clean_symbol.endswith(suffix):
                clean_symbol = clean_symbol[: -len(suffix)].rstrip(".-_")
                break

        if clean_symbol in self._symbol_cache:
            result = self._symbol_cache[clean_symbol]
            if original_symbol != result:
                self.logger.info(f"심볼 변환: {original_symbol} -> {result}")
            return result

        # 3차: 특수문자 제거 후 찾기
        clean_symbol = (
            clean_symbol.replace("/", "")
            .replace(":", "")
            .replace("-", "")
            .replace("_", "")
            .replace(".", "")
        )
        if clean_symbol in self._symbol_cache:
            result = self._symbol_cache[clean_symbol]
            if original_symbol != result:
                self.logger.info(f"심볼 변환: {original_symbol} -> {result}")
            return result

        # 4차: USDT 추가해서 찾기
        if not clean_symbol.endswith("USDT") and len(clean_symbol) <= 8:
            test_symbol = f"{clean_symbol}USDT"
            if test_symbol in self._symbol_cache:
                result = self._symbol_cache[test_symbol]
                if original_symbol != result:
                    self.logger.info(f"심볼 변환: {original_symbol} -> {result}")
                return result

        # 5차: 유사 매칭 시도
        for cached_symbol in self._symbol_cache.keys():
            if cached_symbol.replace("USDT", "").replace("BUSD", "").replace(
                "USD", ""
            ) == clean_symbol.replace("USDT", "").replace("BUSD", "").replace(
                "USD", ""
            ):
                result = self._symbol_cache[cached_symbol]
                if original_symbol != result:
                    self.logger.info(
                        f"심볼 변환 (유사매칭): {original_symbol} -> {result}"
                    )
                return result

        # 매핑 실패시 경고하고 원본 반환
        available_symbols = [s for s in self._symbol_cache.keys() if "XRP" in s][
            :5
        ]  # XRP 관련 심볼 5개 예시
        self.logger.error(f"심볼 매핑 실패: {original_symbol}")
        self.logger.error(f"사용가능한 XRP 관련 심볼 예시: {available_symbols}")
        return original_symbol

    def _sign_request(
        self, method: str, endpoint: str, params: str = "", body: str = ""
    ) -> Dict[str, str]:
        """Bitget API 요청 서명 생성"""
        timestamp = str(int(time.time() * 1000))

        if params and not params.startswith("?"):
            params = "?" + params

        # 서명용 문자열 생성
        message = f"{timestamp}{method.upper()}{endpoint}{params}{body}"

        # HMAC-SHA256 서명
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
            ).digest()
        ).decode("utf-8")

        return {
            "Content-Type": "application/json",
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.passphrase,
            "locale": "en-US",
        }

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        body: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Bitget API 요청 실행"""
        query_str = ""
        if params:
            query_str = "&".join(
                [f"{k}={v}" for k, v in params.items() if v is not None]
            )

        body_str = ""
        if body:
            body_str = json.dumps(body, separators=(",", ":"))

        url = f"{self.base_url}{endpoint}"

        headers = self._sign_request(method, endpoint, query_str, body_str)

        if query_str:
            url += f"?{query_str}"

        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=15)
        else:
            response = requests.post(url, headers=headers, data=body_str, timeout=15)

        if response.status_code == 200:
            data = response.json()
            if data.get("code") == "00000":
                return data
            else:
                raise Exception(
                    f"Bitget API Error {data.get('code')}: {data.get('msg')}"
                )
        else:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

    def validate_credentials(self) -> Tuple[bool, str]:
        """API 자격증명 검증"""
        try:
            data = self._make_request(
                "GET", "/api/v2/mix/account/accounts", {"productType": "USDT-FUTURES"}
            )
            if data.get("code") == "00000":
                return True, "OK (Bitget - Native)"
            else:
                return False, f"API Error: {data.get('msg', 'Unknown error')}"
        except Exception as e:
            return False, f"Connection error: {e}"

    def get_account_balance(self) -> Balance:
        """계정 잔고 조회"""
        try:
            data = self._make_request(
                "GET", "/api/v2/mix/account/accounts", {"productType": "USDT-FUTURES"}
            )
            if data.get("code") == "00000" and data.get("data"):
                for account in data["data"]:
                    if account.get("marginCoin") == "USDT":
                        available = float(account.get("available", "0"))
                        equity = float(account.get("equity", "0"))
                        return Balance(equity, available, "USDT")
            return Balance(0.0, 0.0, "USDT")
        except Exception as e:
            self.logger.error(f"잔고 조회 오류: {e}")
            return Balance(0.0, 0.0, "USDT")

    def get_position(self, symbol: str) -> Position:
        """단일 포지션 조회 - 35회 재조회로 avgPrice=0.0 문제 해결"""
        symbol = self.normalize_symbol(symbol)

        # 35회 재조회 (GPT 요청사항)
        for attempt in range(BITGET_POSITION_RETRY_COUNT):
            try:
                data = self._make_request(
                    "GET",
                    "/api/v2/mix/position/single-position",
                    {
                        "symbol": symbol,
                        "productType": BITGET_PRODUCT_TYPE,
                        "marginCoin": BITGET_MARGIN_COIN,
                    },
                )

                if data.get("code") == "00000" and data.get("data"):
                    for pos_data in data["data"]:
                        size = float(pos_data.get("total", "0"))
                        if size != 0:
                            side = (
                                "long"
                                if pos_data.get("holdSide") == "long"
                                else "short"
                            )
                            avg_price = float(pos_data.get("openPriceAvg", "0") or "0")

                            # avgPrice=0.0 대응
                            if avg_price <= 0:
                                if attempt < BITGET_POSITION_RETRY_COUNT - 1:
                                    self.logger.info(
                                        f"[BITGET] avgPrice=0.0 재조회 {attempt + 1}/{BITGET_POSITION_RETRY_COUNT}: {symbol}"
                                    )
                                    time.sleep(BITGET_POSITION_RETRY_INTERVAL)
                                    break  # 재시도
                                else:
                                    self.logger.warning(
                                        f"[BITGET] 최종 시도에서도 avgPrice=0.0: {symbol}"
                                    )
                                    # 포지션 없음으로 처리
                                    break
                            else:
                                # 정상 avgPrice 확보됨
                                self.logger.info(
                                    f"[BITGET] Entry price confirmed via position_api (attempt {attempt + 1})"
                                )

                                # 캐시에 저장
                                _update_bitget_entry_cache(symbol, avg_price)

                                return Position(
                                    symbol=symbol,
                                    side=side,
                                    qty=abs(size),
                                    entry_price=avg_price,
                                    mark_price=float(pos_data.get("markPrice", "0")),
                                    unrealized_pnl=float(
                                        pos_data.get("unrealizedPL", "0")
                                    ),
                                    leverage=float(pos_data.get("leverage", "1")),
                                    category="linear",
                                )

            except Exception as e:
                if attempt < BITGET_POSITION_RETRY_COUNT - 1:
                    self.logger.warning(
                        f"[BITGET] 포지션 조회 예외, 재시도 {attempt + 1}: {e}"
                    )
                    time.sleep(BITGET_POSITION_RETRY_INTERVAL)
                else:
                    self.logger.error(f"[BITGET] 포지션 조회 최종 실패: {e}")

        # 포지션이 없거나 avgPrice=0.0 문제로 실패
        return Position(symbol, "flat", 0.0, 0.0, category="linear")

    def _has_active_trailing_plan(self, symbol: str) -> bool:
        """기존 트레일링 플랜 존재 여부 확인 - 최신 엔드포인트로 교체"""
        symbol = self.normalize_symbol(symbol)

        try:
            # 최신 트랙플랜 조회 엔드포인트 사용
            data = self._make_request(
                "GET",
                "/api/v2/mix/order/plan-orders-pending",
                {
                    "symbol": symbol,
                    "productType": "USDT-FUTURES",
                    "planType": "track_plan",
                },
            )
            if data.get("code") == "00000" and data.get("data"):
                for order in data["data"]:
                    if (
                        order.get("planType") == "track_plan"
                        and order.get("symbol") == symbol
                    ):
                        self.logger.info(f"[BITGET] 기존 트레일링 발견: {symbol}")
                        return True
            return False
        except Exception as e:
            error_msg = str(e)
            if "40404" in error_msg:
                logger.info(f"[BITGET] 트레일링 주문 없음 (40404): {symbol}")
                return False
            else:
                logger.warning(f"[BITGET] 기존 트레일링 확인 실패, 계속 진행: {e}")
                return False

    def get_all_positions(self) -> List[Position]:
        """모든 활성 포지션 조회"""
        positions = []

        try:
            data = self._make_request(
                "GET",
                "/api/v2/mix/position/all-position",
                {"productType": "USDT-FUTURES"},
            )
            if data.get("code") == "00000" and data.get("data"):
                for pos_data in data["data"]:
                    size = float(pos_data.get("total", "0"))
                    if size != 0:
                        symbol = pos_data.get("symbol", "")
                        side = "long" if pos_data.get("holdSide") == "long" else "short"

                        position = Position(
                            symbol=symbol,
                            side=side,
                            qty=abs(size),
                            entry_price=float(pos_data.get("averageOpenPrice", "0")),
                            mark_price=float(pos_data.get("markPrice", "0")),
                            unrealized_pnl=float(pos_data.get("upl", "0")),
                            leverage=float(pos_data.get("leverage", "1")),
                            category="linear",
                        )
                        positions.append(position)
        except Exception as e:
            self.logger.error(f"전체 포지션 조회 오류: {e}")

        return positions

    def get_ticker_price(self, symbol: str) -> Tuple[float, float]:
        """현재가 조회 - 다층 fallback 시스템"""
        symbol = self.normalize_symbol(symbol)

        # 메인 API 시도
        try:
            data = self._make_request(
                "GET",
                "/api/v2/mix/market/ticker",
                {"symbol": symbol, "productType": "USDT-FUTURES"},
            )
            if data.get("code") == "00000" and data.get("data"):
                ticker_list = data["data"]
                if isinstance(ticker_list, list) and len(ticker_list) > 0:
                    ticker_data = ticker_list[0]
                    bid = float(ticker_data.get("bidPx", "0"))
                    ask = float(ticker_data.get("askPx", "0"))
                    if bid > 0 and ask > 0:
                        self.logger.info(
                            f"{symbol} 현재가 조회 성공: Bid={bid}, Ask={ask}"
                        )
                        return bid, ask
                elif isinstance(ticker_list, dict):
                    bid = float(ticker_list.get("bidPx", "0"))
                    ask = float(ticker_list.get("askPx", "0"))
                    if bid > 0 and ask > 0:
                        self.logger.info(
                            f"{symbol} 현재가 조회 성공: Bid={bid}, Ask={ask}"
                        )
                        return bid, ask
        except Exception as e:
            self.logger.error(f"메인 현재가 조회 오류 {symbol}: {e}")

        # Fallback: 24시간 통계 API 시도
        try:
            self.logger.warning(f"{symbol} fallback API 시도중...")
            data = self._make_request(
                "GET", "/api/v2/mix/market/tickers", {"productType": "USDT-FUTURES"}
            )
            if data.get("code") == "00000" and data.get("data"):
                tickers = data["data"]
                for ticker in tickers:
                    if ticker.get("symbol") == symbol:
                        # 24시간 통계에서 현재가 추출
                        last_price = float(ticker.get("lastPr", "0"))
                        if last_price > 0:
                            # bid/ask를 lastPrice 기준으로 추정
                            spread = last_price * 0.0001  # 0.01% 스프레드 가정
                            bid = last_price - spread / 2
                            ask = last_price + spread / 2
                            self.logger.info(
                                f"{symbol} fallback 현재가 성공: Last={last_price}, Bid={bid}, Ask={ask}"
                            )
                            return bid, ask
                        break
        except Exception as e:
            self.logger.error(f"Fallback 현재가 조회 오류 {symbol}: {e}")

        # 최종 Fallback: 인덱스 가격 API 시도
        try:
            self.logger.warning(f"{symbol} 인덱스 가격 API 시도중...")
            # 심볼에서 기본 통화 추출
            if symbol.endswith("USDT"):
                index_symbol = symbol.replace("USDT", "")
                data = self._make_request("GET", "/api/v2/spot/market/tickers")
                if data.get("code") == "00000" and data.get("data"):
                    for ticker in data["data"]:
                        if ticker.get("symbol") == f"{index_symbol}USDT":
                            price = float(ticker.get("lastPr", "0"))
                            if price > 0:
                                spread = price * 0.0001
                                bid = price - spread / 2
                                ask = price + spread / 2
                                self.logger.info(
                                    f"{symbol} 인덱스 가격 성공: Price={price}, Bid={bid}, Ask={ask}"
                                )
                                return bid, ask
                            break
        except Exception as e:
            self.logger.error(f"인덱스 가격 조회 오류 {symbol}: {e}")

        self.logger.error(f"{symbol} 모든 현재가 조회 실패 - 거래 불가능")
        return 0.0, 0.0

    def get_contract_info(self, symbol: str) -> ContractInfo:
        """계약 정보 조회 - 배열 응답 처리 수정"""
        symbol = self.normalize_symbol(symbol)
        if symbol in self._contract_cache:
            return self._contract_cache[symbol]

        try:
            data = self._make_request(
                "GET", "/api/v2/mix/market/contracts", {"productType": "USDT-FUTURES"}
            )
            if data.get("code") == "00000" and data.get("data"):
                # 배열 처리 - data["data"]는 계약 정보 리스트
                contracts_list = data["data"]
                if isinstance(contracts_list, list):
                    for contract_data in contracts_list:
                        if contract_data.get("symbol") == symbol:
                            contract = ContractInfo(
                                symbol=symbol,
                                base_currency=contract_data.get("baseCoin", ""),
                                quote_currency=contract_data.get("quoteCoin", "USDT"),
                                tick_size=float(
                                    contract_data.get("priceEndStep", "0.01")
                                ),
                                step_size=float(
                                    contract_data.get("minTradeNum", "0.001")
                                ),
                                min_qty=float(
                                    contract_data.get("minTradeNum", "0.001")
                                ),
                                min_notional=float(
                                    contract_data.get("minTradeUSDT", "5")
                                ),
                            )
                            self._contract_cache[symbol] = contract
                            return contract
        except Exception as e:
            self.logger.error(f"계약 정보 조회 오류 {symbol}: {e}")

        # 기본값 반환
        return ContractInfo(
            symbol,
            "",
            "USDT",
            DEFAULT_VALUES["tick_size"],
            DEFAULT_VALUES["step_size"],
            DEFAULT_VALUES["min_qty"],
        )

    def set_leverage(self, symbol: str, leverage: float) -> bool:
        """레버리지 설정"""
        symbol = self.normalize_symbol(symbol)

        try:
            body = {
                "symbol": symbol,
                "productType": "USDT-FUTURES",
                "marginCoin": "USDT",
                "leverage": str(int(leverage)),
            }

            data = self._make_request(
                "POST", "/api/v2/mix/account/set-leverage", body=body
            )

            if data.get("code") == "00000":
                return True
            else:
                self.logger.warning(f"레버리지 설정 실패: {data}")
                return False
        except Exception as e:
            self.logger.error(f"레버리지 설정 오류 {symbol}: {e}")
            return False

    def place_market_order(
        self, symbol: str, side: str, qty: float, reduce_only: bool = False
    ) -> OrderResult:
        """마켓 주문 실행 - Bitget API 기준 완전 재구현"""
        symbol = self.normalize_symbol(symbol)

        # 중복 수량 조정 제거: 상위에서 이미 처리됨을 가정
        # reduce_only 주문만 최소 검증
        if reduce_only and qty <= 0:
            return OrderResult(False, "", "청산 수량이 유효하지 않음")
        elif not reduce_only and qty <= 0:
            return OrderResult(False, "", "신규 주문 수량이 유효하지 않음")

        try:
            # Bitget side 변환 수정
            if reduce_only:
                pos = self.get_position(symbol)
                if pos.side == "long":
                    bitget_side = "sell"
                    bitget_trade_side = "close"
                elif pos.side == "short":
                    bitget_side = "buy"
                    bitget_trade_side = "close"
                else:
                    return OrderResult(False, "", "청산할 포지션이 없음")
            else:
                if side.lower() in ["long", "buy"]:
                    bitget_side = "buy"
                    bitget_trade_side = "open"
                elif side.lower() in ["short", "sell"]:
                    bitget_side = "sell"
                    bitget_trade_side = "open"
                else:
                    return OrderResult(False, "", f"잘못된 side 값: {side}")

            body = {
                "symbol": symbol,
                "productType": "USDT-FUTURES",
                "marginCoin": "USDT",
                "marginMode": "isolated",
                "size": str(qty),
                "side": bitget_side,
                "orderType": "market",
            }

            if reduce_only:
                body["reduceOnly"] = True

            self.logger.info(
                f"Bitget 주문: symbol={symbol}, side={bitget_side}, tradeSide={bitget_trade_side}, qty={qty}"
            )

            data = self._make_request(
                "POST", "/api/v2/mix/order/place-order", body=body
            )

            if data.get("code") == "00000":
                order_id = data["data"]["orderId"]
                return OrderResult(True, order_id, "주문 성공", qty, 0.0, data)
            else:
                error_msg = data.get("msg", "알 수 없는 오류")
                self.logger.error(f"마켓 주문 실패: {error_msg}")
                return OrderResult(False, "", error_msg)

        except Exception as e:
            self.logger.error(f"마켓 주문 예외: {e}")
            return OrderResult(False, "", str(e))

    def close_all_positions(self, symbol: str = None) -> Dict[str, OrderResult]:
        """포지션 일괄 청산"""
        results = {}

        if symbol:
            try:
                position = self.get_position(symbol)
                if position.side == "flat":
                    results[symbol] = OrderResult(False, "", "청산할 포지션 없음")
                    return results

                result = self._close_single_position(position)
                results[symbol] = result
            except Exception as e:
                results[symbol] = OrderResult(False, "", str(e))
        else:
            positions = self.get_all_positions()
            for position in positions:
                try:
                    result = self._close_single_position(position)
                    results[position.symbol] = result
                except Exception as e:
                    results[position.symbol] = OrderResult(False, "", str(e))

        return results

    def _cancel_all_orders(self, symbol: str) -> bool:
        """내부 헬퍼: 해당 심볼의 모든 주문(트레일링 포함) 취소"""
        symbol = self.normalize_symbol(symbol)

        endpoints = [
            "/api/v2/mix/order/cancel-all-orders",  # 일반 주문
            "/api/v2/mix/order/cancel-all-plan-orders",  # 트레일링
            "/api/v2/mix/order/cancel-all-tpsl-orders",  # TP/SL
        ]

        for endpoint in endpoints:
            try:
                body = {"symbol": symbol, "productType": "USDT-FUTURES"}
                if "cancel-all-orders" in endpoint:
                    body["marginCoin"] = "USDT"
                self._make_request("POST", endpoint, body=body)
            except Exception:
                pass  # 실패해도 계속 진행

        return True

    def _close_single_position(self, position: Position) -> OrderResult:
        """단일 포지션 청산"""
        if position.side == "flat" or position.qty <= 0:
            return OrderResult(False, "", "청산할 포지션 없음")

        try:
            # 모든 주문 취소 후 포지션 청산
            self.cancel_all_orders(position.symbol)
            time.sleep(0.3)

            close_side = "short" if position.side == "long" else "long"
            return self.place_market_order(
                position.symbol, close_side, position.qty, reduce_only=True
            )

        except Exception as e:
            return OrderResult(False, "", str(e))

    # ============================================================================
    # 보호장치 표준 인터페이스 구현 - Bitget 특화
    # ============================================================================

    def set_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        sl_pct: float,
        tick_size: float,
    ) -> bool:
        """표준 손절 설정 - Bitget V2 API"""
        symbol = self.normalize_symbol(symbol)

        try:
            # 손절가 계산
            if side.lower() == "long":
                stop_price = entry_price * (1 - sl_pct / 100.0)
            else:
                stop_price = entry_price * (1 + sl_pct / 100.0)

            # 틱 조정
            stop_price = round_to_tick(stop_price, tick_size, side)

            if stop_price <= 0:
                self.logger.error(f"[BITGET] 잘못된 손절가: {stop_price}")
                return False

            # 포지션 확인
            pos = self.get_position(symbol)
            if pos.side == "flat":
                return True

            # holdSide 매핑 (원웨이 모드)
            hold_side = "buy" if pos.side == "long" else "sell"

            # Bitget V2 TPSL payload
            body = {
                "symbol": symbol,
                "productType": "USDT-FUTURES",
                "marginMode": "isolated",
                "marginCoin": "USDT",
                "planType": "pos_loss",
                "holdSide": hold_side,
                "triggerPrice": f"{stop_price}",
                "triggerType": "mark_price",
                "executePrice": "0",
            }

            log_payload_safe(self.logger, "[BITGET TPSL REQ]", body)

            data = self._make_request(
                "POST", "/api/v2/mix/order/place-tpsl-order", body=body
            )

            self.logger.info(
                f"[BITGET TPSL RESP] code={data.get('code')} msg={data.get('msg')}"
            )

            if data.get("code") == "00000":
                self.logger.info(f"[BITGET] 손절 설정 성공: {symbol} @ {stop_price}")
                return True
            else:
                self.logger.error(f"[BITGET] 손절 설정 실패: {data}")
                return False

        except Exception as e:
            self.logger.error(f"[BITGET] 손절 설정 오류: {e}")
            return False

    def set_trailing_stop(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        ts_trigger_pct: float,
        ts_callback_pct: float,
        tick_size: float,
    ) -> bool:
        """표준 트레일링 스톱 설정 - Bitget V2 API (중복 방지 강화)"""
        symbol = self.normalize_symbol(symbol)

        # 기존 트레일링 주문 확인 (중복 방지)
        if self._has_active_trailing_plan(symbol):
            self.logger.info(f"[BITGET] 기존 트레일링 발견, 중복 생성 금지: {symbol}")
            return True  # 이미 설정됨 = 성공

        try:
            pos = self.get_position(symbol)
            if pos.side == "flat":
                return True

            # entry_price 검증 및 폴백
            if entry_price is None or entry_price <= 0:
                self.logger.warning(
                    f"[BITGET] entry_price 무효 ({entry_price}), 현재가로 대체: {symbol}"
                )
                bid, ask = self.get_ticker_price(symbol)
                entry_price = ask if side.lower() == "long" else bid
                if entry_price <= 0:
                    self.logger.error(f"[BITGET] 현재가 조회 실패: {symbol}")
                    return False
                self.logger.info(
                    f"[BITGET] entry_price 폴백 완료: {symbol} -> {entry_price}"
                )

            # 현재가 조회
            bid, ask = self.get_ticker_price(symbol)
            mark_price = (bid + ask) / 2 if bid > 0 and ask > 0 else ask or bid

            if mark_price <= 0:
                self.logger.error(f"[BITGET] 현재가 조회 실패: {symbol}")
                return False

            # 트리거 가격 계산
            if side.lower() == "long":
                trigger_price = entry_price * (1.0 + float(ts_trigger_pct) / 100.0)
                trail_side = "sell"  # 롱 포지션 보호용
                if trigger_price <= mark_price:
                    trigger_price = mark_price * (
                        1.0 + max(0.1, float(ts_trigger_pct)) / 100.0
                    )
            else:
                trigger_price = entry_price * (1.0 - float(ts_trigger_pct) / 100.0)
                trail_side = "buy"  # 숏 포지션 보호용
                if trigger_price >= mark_price:
                    trigger_price = mark_price * (
                        1.0 - max(0.1, float(ts_trigger_pct)) / 100.0
                    )

            # 틱 조정
            trigger_price = round_to_tick(trigger_price, tick_size, side)
            callback_ratio = float(ts_callback_pct)

            # 콜백 비율 검증
            if callback_ratio <= 0:
                self.logger.error(f"[BITGET] 콜백 비율이 0 이하: {callback_ratio}%")
                return False

            # Bitget V2 track_plan payload
            body = {
                "planType": "track_plan",
                "symbol": symbol,
                "productType": "USDT-FUTURES",
                "marginMode": "isolated",
                "marginCoin": "USDT",
                "size": f"{pos.qty}",
                "triggerPrice": f"{trigger_price}",
                "triggerType": "mark_price",
                "side": trail_side,
                "orderType": "market",
                "callbackRatio": f"{callback_ratio}",
                "reduceOnly": "YES",
            }

            log_payload_safe(self.logger, "[BITGET TRAIL REQ]", body)

            data = self._make_request(
                "POST", "/api/v2/mix/order/place-plan-order", body=body
            )

            self.logger.info(
                f"[BITGET TRAIL RESP] code={data.get('code')} msg={data.get('msg')}"
            )

            if data.get("code") == "00000":
                self.logger.info(
                    f"[BITGET] 트레일링 설정 성공: {symbol} trigger={trigger_price} callback={callback_ratio}"
                )
                return True
            else:
                self.logger.error(f"[BITGET] 트레일링 설정 실패: {data}")
                return False

        except Exception as e:
            self.logger.error(f"[BITGET] 트레일링 설정 오류: {e}")
            return False

    def _calculate_fills_vwap_bitget(
        self, symbol: str, side: str, max_lookback: int = 50
    ) -> Optional[float]:
        """
        Bitget 전용: 최근 체결(fills) 조회 → 방향 일치 체결만 추출 → 공용 유틸로 VWAP 계산
        - 계산은 _VWAPUtils.calc_vwap_from_fills() 사용(단일화)
        - 여기서는 캐시/로깅/편차 검증만 수행
        """
        try:
            # 1) 최근 체결 조회
            data = self._make_request(
                "GET",
                "/api/v2/mix/order/fills",
                {
                    "symbol": symbol,
                    "productType": BITGET_PRODUCT_TYPE,  # 상수 사용
                    "limit": str(max_lookback),  # 상수 사용
                },
            )

            if data.get("code") != "00000" or not data.get("data"):
                return None

            fills = data["data"].get("fillList", [])
            if not fills:
                return None

            # 2) 방향 필터링 (GPT 요청: 포지션 방향과 일치)
            target_side = "buy" if side.lower() == "long" else "sell"
            target_trade_side = "open"  # 신규 진입만

            matched_fills = []
            for fill in fills:
                fill_side = fill.get("side", "").lower()
                fill_trade_side = fill.get("tradeSide", "").lower()

                if fill_side == target_side and fill_trade_side == target_trade_side:
                    matched_fills.append(fill)

            # 최소 체결 건수 확인 (상수 사용)
            if len(matched_fills) < BITGET_VWAP_MIN_FILLS:
                return None

            # 3) 공용 유틸로 VWAP 계산
            vwap = _VWAPUtils.calc_vwap_from_fills(matched_fills)
            if not vwap or vwap <= 0:
                self.logger.info(f"[{symbol}] bitget fills_vwap unavailable")
                return None

            # 4) 현재가 대비 편차 검증 (기본 10%)
            try:
                bid, ask = self.get_ticker_price(symbol)
                last = (bid + ask) / 2 if bid > 0 and ask > 0 else ask or bid
            except Exception:
                last = 0.0

            if last > 0:
                dev = abs(vwap - last) / last * 100.0
                if dev > BITGET_MAX_PRICE_DEVIATION:  # 상수 사용
                    self.logger.warning(
                        f"[{symbol}] fills_vwap dev={dev:.2f}% > {BITGET_MAX_PRICE_DEVIATION}% → discard"
                    )
                    return None

            # 5) (선택) 캐시 업데이트
            try:
                _update_bitget_entry_cache(symbol, float(vwap))
            except Exception:
                pass

            self.logger.info(
                f"[{symbol}] entry source=fills_vwap, final=False, price={float(vwap):.8f}, fills={len(matched_fills)}"
            )
            return float(vwap)

        except Exception as e:
            self.logger.warning(f"[{symbol}] fills_vwap exception: {e}")
            return None

    def _get_recent_order_avg_price_bitget(
        self, symbol: str, side: str, max_lookback: int = 20
    ) -> Optional[float]:
        """
        Bitget 최근 주문/체결 내역에서 평균 체결가 추정

        Args:
            symbol: 심볼
            side: 포지션 방향 ("long" or "short")
            max_lookback: 조회할 최대 체결 건수

        Returns:
            평균 체결가 또는 None
        """
        try:
            symbol = self.normalize_symbol(symbol)

            # side를 Bitget API 형식으로 변환
            bitget_side = "buy" if side.lower() == "long" else "sell"

            # 최근 체결 내역 조회
            data = self._make_request(
                "GET",
                "/api/v2/mix/order/fills",
                {
                    "symbol": symbol,
                    "productType": BITGET_PRODUCT_TYPE,
                    "limit": str(max_lookback),
                },
            )

            if data.get("code") != "00000" or not data.get("data"):
                self.logger.warning(f"[BITGET] order fills 조회 실패: {symbol}")
                return None

            fills = data["data"].get("fillList", [])
            if not fills:
                self.logger.info(f"[BITGET] 체결 내역 없음: {symbol}")
                return None

            # side 방향과 일치하는 체결만 필터링
            target_fills = []
            for fill in fills:
                fill_side = fill.get("side", "").lower()
                fill_trade_side = fill.get("tradeSide", "").lower()

                # 신규 진입(open) 체결만 대상
                if fill_side == bitget_side and fill_trade_side == "open":
                    price = float(fill.get("price", 0) or 0)
                    if price > 0:
                        target_fills.append(price)

            if not target_fills:
                self.logger.info(f"[BITGET] {side} 방향 신규 체결 없음: {symbol}")
                return None

            # 단순 평균 계산
            avg_price = sum(target_fills) / len(target_fills)

            # 현재가 대비 편차 검증 (10% 이내)
            try:
                bid, ask = self.get_ticker_price(symbol)
                current_price = (bid + ask) / 2 if bid > 0 and ask > 0 else ask or bid
                if current_price > 0:
                    deviation = abs(avg_price - current_price) / current_price * 100
                    if deviation > BITGET_MAX_PRICE_DEVIATION:
                        self.logger.warning(
                            f"[BITGET] order_history 편차 초과 {deviation:.2f}% > {BITGET_MAX_PRICE_DEVIATION}%: {symbol}"
                        )
                        return None
            except Exception:
                pass

            self.logger.info(
                f"[BITGET] order_history 평균가 조회 성공: {symbol} price={avg_price:.8f} fills={len(target_fills)}"
            )
            return avg_price

        except Exception as e:
            self.logger.warning(
                f"[BITGET] order_history avg lookup 실패: {symbol} - {e}"
            )
            return None

    def close_position(self, symbol: str) -> bool:
        """단일 포지션 강제 청산"""
        try:
            results = self.close_all_positions(symbol)
            result = results.get(symbol)
            if result and result.success:
                self.logger.info(f"[BITGET] 포지션 강제 청산 성공: {symbol}")
                return True
            else:
                error_msg = result.message if result else "알 수 없는 오류"
                self.logger.error(
                    f"[BITGET] 포지션 강제 청산 실패: {symbol} - {error_msg}"
                )
                return False
        except Exception as e:
            self.logger.error(f"[BITGET] 포지션 강제 청산 예외: {symbol} - {e}")
            return False


def create_exchange_client(
    exchange_name: str,
    api_key: str,
    api_secret: str,
    passphrase: str = "",
    testnet: bool = False,
) -> Optional[ExchangeBase]:
    """거래소 클라이언트 생성 팩토리"""
    logger = get_logger("factory")

    if not api_key or not api_secret:
        logger.error(f"API 키 누락: {exchange_name}")
        return None

    try:
        if exchange_name.lower() == "bybit":
            client = BybitExchange(api_key, api_secret, testnet)
        elif exchange_name.lower() == "bitget":
            client = BitgetExchange(api_key, api_secret, passphrase, testnet)
        else:
            logger.error(f"지원되지 않는 거래소: {exchange_name}")
            return None

        logger.info(f"거래소 클라이언트 생성 완료: {exchange_name}")
        return client
    except Exception as e:
        logger.error(f"거래소 클라이언트 생성 실패 {exchange_name}: {e}")
        return None


# ============================================================================
# 상태 관리자
# ============================================================================


@dataclass
class TrailState:
    """트레일링 상태 정보"""

    active: bool = False
    side: str = ""
    activation_price: float = 0.0
    high_watermark: float = 0.0
    low_watermark: float = 0.0
    last_sl_price: float = 0.0
    callback_ratio: float = 0.0
    trigger_ratio: float = 0.0
    last_update_ts: float = 0.0


@dataclass
class SymbolState:
    """심볼별 상태 정보 - 보호장치 상태 추가"""

    position: Optional[Position] = None
    last_signal_ts: float = 0.0
    last_signal_side: str = ""
    cooldown_until: float = 0.0
    # 보호장치 상태 추가
    stop_loss_set: bool = False
    trail_active: bool = False
    protection_retries: int = 0
    last_protection_attempt: float = 0.0
    cooldown_active: bool = False
    # 기존 상태 유지
    fixed_stop_price: Optional[float] = None
    trail: TrailState = field(default_factory=TrailState)


@dataclass
class ExchangeState:
    """거래소별 상태 정보"""

    name: str
    connected: bool = False
    symbols: Dict[str, SymbolState] = field(default_factory=dict)


class StateManager:
    """전역 상태 관리자"""

    def __init__(self):
        self.exchanges: Dict[str, ExchangeState] = {}
        self._lock = threading.RLock()
        self._state_file = Path(__file__).parent / "state.json"
        self.logger = get_logger("state")
        self._load_state()

    def _load_state(self):
        """상태 파일 로드"""
        try:
            if self._state_file.exists():
                with open(self._state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for exchange_name, exchange_data in data.get("exchanges", {}).items():
                    symbols = {}
                    for symbol, symbol_data in exchange_data.get("symbols", {}).items():
                        position = None
                        if symbol_data.get("position"):
                            pos_data = symbol_data["position"]
                            position = Position(
                                symbol=pos_data["symbol"],
                                side=pos_data["side"],
                                qty=pos_data["qty"],
                                entry_price=pos_data["entry_price"],
                                mark_price=pos_data.get("mark_price", 0.0),
                                unrealized_pnl=pos_data.get("unrealized_pnl", 0.0),
                                category=pos_data.get("category", "linear"),
                            )

                        # TrailState 로딩
                        trail_data = symbol_data.get("trail", {})
                        trail = TrailState(
                            active=trail_data.get("active", False),
                            side=trail_data.get("side", ""),
                            activation_price=trail_data.get("activation_price", 0.0),
                            high_watermark=trail_data.get("high_watermark", 0.0),
                            low_watermark=trail_data.get("low_watermark", 0.0),
                            last_sl_price=trail_data.get("last_sl_price", 0.0),
                            callback_ratio=trail_data.get("callback_ratio", 0.0),
                            trigger_ratio=trail_data.get("trigger_ratio", 0.0),
                            last_update_ts=trail_data.get("last_update_ts", 0.0),
                        )

                        symbols[symbol] = SymbolState(
                            position=position,
                            last_signal_ts=symbol_data.get("last_signal_ts", 0.0),
                            last_signal_side=symbol_data.get("last_signal_side", ""),
                            cooldown_until=symbol_data.get("cooldown_until", 0.0),
                            # 보호장치 상태 로딩 (기본값 제공)
                            stop_loss_set=symbol_data.get("stop_loss_set", False),
                            trail_active=symbol_data.get("trail_active", False),
                            protection_retries=symbol_data.get("protection_retries", 0),
                            last_protection_attempt=symbol_data.get(
                                "last_protection_attempt", 0.0
                            ),
                            cooldown_active=symbol_data.get("cooldown_active", False),
                            # 미사용 필드 제거됨 - 핵심 보호장치 상태만 유지
                        )

                    self.exchanges[exchange_name] = ExchangeState(
                        name=exchange_name,
                        connected=exchange_data.get("connected", False),
                        symbols=symbols,
                    )

                self.logger.info(f"상태 파일 로드 완료: {len(self.exchanges)}개 거래소")
            else:
                self.logger.info("새로운 상태 파일 시작")
        except Exception as e:
            self.logger.error(f"상태 파일 로드 실패: {e}")

    def save_state(self):
        """상태 파일 저장"""
        try:
            with self._lock:
                data = {"exchanges": {}}
                for exchange_name, exchange_state in self.exchanges.items():
                    symbols_data = {}
                    for symbol, symbol_state in exchange_state.symbols.items():
                        position_data = None
                        if symbol_state.position:
                            position_data = asdict(symbol_state.position)

                        symbols_data[symbol] = {
                            "position": position_data,
                            "last_signal_ts": symbol_state.last_signal_ts,
                            "last_signal_side": symbol_state.last_signal_side,
                            "cooldown_until": symbol_state.cooldown_until,
                            # 보호장치 상태 저장
                            "stop_loss_set": symbol_state.stop_loss_set,
                            "trail_active": symbol_state.trail_active,
                            "protection_retries": symbol_state.protection_retries,
                            "last_protection_attempt": symbol_state.last_protection_attempt,
                            "cooldown_active": symbol_state.cooldown_active,
                        }

                    data["exchanges"][exchange_name] = {
                        "name": exchange_state.name,
                        "connected": exchange_state.connected,
                        "symbols": symbols_data,
                    }

                with open(self._state_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                self.logger.info("상태 저장 완료")
        except Exception as e:
            self.logger.error(f"상태 저장 실패: {e}")

    def get_exchange(self, exchange_name: str) -> ExchangeState:
        """거래소 상태 조회/생성"""
        with self._lock:
            if exchange_name not in self.exchanges:
                self.exchanges[exchange_name] = ExchangeState(name=exchange_name)
            return self.exchanges[exchange_name]

    def get_symbol_state(self, exchange_name: str, symbol: str) -> SymbolState:
        """심볼 상태 조회/생성"""
        with self._lock:
            exchange = self.get_exchange(exchange_name)
            if symbol not in exchange.symbols:
                exchange.symbols[symbol] = SymbolState()
            return exchange.symbols[symbol]

    def update_position(self, exchange_name: str, symbol: str, position: Position):
        """포지션 상태 업데이트"""
        with self._lock:
            symbol_state = self.get_symbol_state(exchange_name, symbol)
            symbol_state.position = position
            if position.side == "flat":
                # 포지션 청산시 보호장치 상태 초기화
                symbol_state.fixed_stop_price = None
                symbol_state.stop_loss_set = False
                symbol_state.trail_active = False
                symbol_state.protection_retries = 0

    def sync_positions(self, exchange_name: str, positions: List[Position]):
        """거래소 포지션 동기화"""
        with self._lock:
            exchange = self.get_exchange(exchange_name)

            # 기존 포지션을 모두 플랫으로 초기화
            for symbol_state in exchange.symbols.values():
                if symbol_state.position:
                    symbol_state.position.side = "flat"
                    symbol_state.position.qty = 0.0

            # 새 포지션으로 업데이트
            for position in positions:
                self.update_position(exchange_name, position.symbol, position)

            self.logger.info(
                f"포지션 동기화 완료 [{exchange_name}]: {len(positions)}개"
            )

    def set_signal_filter(self, exchange_name: str, symbol: str, side: str):
        """신호 필터 설정"""
        with self._lock:
            symbol_state = self.get_symbol_state(exchange_name, symbol)
            symbol_state.last_signal_ts = time.time()
            symbol_state.last_signal_side = side

    def can_process_signal(
        self, exchange_name: str, symbol: str, side: str, min_interval: int
    ) -> tuple[bool, str]:
        """신호 처리 가능 여부 확인"""
        with self._lock:
            now = time.time()
            symbol_state = self.get_symbol_state(exchange_name, symbol)

            # 쿨다운 체크 (보호장치 쿨다운 포함)
            if symbol_state.cooldown_until > now:
                remaining = int(symbol_state.cooldown_until - now)
                return False, f"쿨다운 중 (남은시간: {remaining}초)"

            if symbol_state.cooldown_active:
                return False, "보호장치 실패로 인한 쿨다운 활성"

            # 신호 간격 체크
            if now - symbol_state.last_signal_ts < min_interval:
                remaining = int(min_interval - (now - symbol_state.last_signal_ts))
                return False, f"신호 간격 부족 (남은시간: {remaining}초)"

            return True, "OK"

    def set_cooldown(self, exchange_name: str, symbol: str, cooldown_sec: int):
        """쿨다운 설정"""
        with self._lock:
            symbol_state = self.get_symbol_state(exchange_name, symbol)
            symbol_state.cooldown_until = time.time() + cooldown_sec

    def set_protection_cooldown(self, exchange_name: str, symbol: str):
        """보호장치 실패 쿨다운 설정"""
        with self._lock:
            symbol_state = self.get_symbol_state(exchange_name, symbol)
            symbol_state.cooldown_active = True
            symbol_state.cooldown_until = time.time() + 300  # 5분 쿨다운

    def get_all_positions(self) -> Dict[str, Dict[str, Position]]:
        """모든 활성 포지션 반환"""
        with self._lock:
            result = {}
            for exchange_name, exchange_state in self.exchanges.items():
                positions = {}
                for symbol, symbol_state in exchange_state.symbols.items():
                    if symbol_state.position and symbol_state.position.side != "flat":
                        positions[symbol] = symbol_state.position
                result[exchange_name] = positions
            return result

    def get_exchange_status(self) -> Dict[str, Dict[str, Any]]:
        """거래소 상태 요약"""
        with self._lock:
            result = {}
            for exchange_name, exchange_state in self.exchanges.items():
                active_positions = sum(
                    1
                    for s in exchange_state.symbols.values()
                    if s.position and s.position.side != "flat"
                )
                result[exchange_name] = {
                    "connected": exchange_state.connected,
                    "active_positions": active_positions,
                    "total_symbols": len(exchange_state.symbols),
                }
            return result


# 전역 상태 관리자
state_manager = StateManager()

# ============================================================================
# 보호장치 지원 함수들 - avgPrice 및 멱등성 문제 해결
# ============================================================================


def _check_entry_cache(symbol: str) -> Optional[float]:
    """진입가 캐시 확인"""
    try:
        cache_file = Path(__file__).parent / "entry_cache.json"
        if cache_file.exists():
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
                entry_data = cache_data.get(symbol, {})
                if (
                    entry_data and time.time() - entry_data.get("timestamp", 0) < 300
                ):  # 5분 유효
                    return entry_data.get("price")
    except Exception:
        pass
    return None


def _update_entry_cache(symbol: str, price: float):
    """진입가 캐시 업데이트"""
    try:
        cache_file = Path(__file__).parent / "entry_cache.json"
        cache_data = {}
        if cache_file.exists():
            with open(cache_file, "r") as f:
                cache_data = json.load(f)

        cache_data[symbol] = {"price": price, "timestamp": time.time()}

        with open(cache_file, "w") as f:
            json.dump(cache_data, f)
    except Exception:
        pass


def _check_bitget_entry_cache(symbol: str) -> Optional[float]:
    """Bitget 진입가 캐시 확인 (GPT 요청: 단순 캐시)"""
    try:
        cache_file = Path(__file__).parent / BITGET_ENTRY_CACHE_FILE
        if cache_file.exists():
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
                entry_data = cache_data.get(symbol, {})
                if (
                    entry_data
                    and time.time() - entry_data.get("timestamp", 0)
                    < BITGET_ENTRY_CACHE_TTL
                ):
                    return entry_data.get("price")
    except Exception:
        pass
    return None


def _update_bitget_entry_cache(symbol: str, price: float):
    """Bitget 진입가 캐시 업데이트 (GPT 요청: entry_cache.json)"""
    try:
        cache_file = Path(__file__).parent / BITGET_ENTRY_CACHE_FILE
        cache_data = {}
        if cache_file.exists():
            with open(cache_file, "r") as f:
                cache_data = json.load(f)

        cache_data[symbol] = {"price": price, "timestamp": time.time()}

        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        logger = get_logger("bitget_cache")
        logger.info(f"[BITGET] 진입가 캐시 업데이트: {symbol} @ {price}")
    except Exception:
        pass


def _validate_bitget_price_deviation(
    exchange: ExchangeBase, symbol: str, price: float
) -> bool:
    """
    Bitget 가격 편차 검증 - GPT 요청사항

    Args:
        exchange: Bitget 거래소 클라이언트
        symbol: 심볼
        price: 검증할 가격

    Returns:
        편차가 허용 범위 내이면 True
    """
    try:
        bid, ask = exchange.get_ticker_price(symbol)
        current_price = (bid + ask) / 2 if bid > 0 and ask > 0 else ask or bid

        if current_price <= 0:
            return False

        return _PriceUtils.within_deviation(
            price, current_price, BITGET_MAX_PRICE_DEVIATION
        )

    except Exception:
        return False


def get_valid_entry_price(
    exchange: ExchangeBase,
    symbol: str,
    fallback_price: float = 0.0,
    max_retries: int = 5,
    retry_interval: float = 5.0,
) -> float:
    """
    유효한 진입가격 조회 - avgPrice=0.0 문제 해결

    Args:
        exchange: 거래소 클라이언트
        symbol: 심볼
        fallback_price: 폴백 가격 (markPrice 등)
        max_retries: 최대 재시도 횟수
        retry_interval: 재시도 간격(초)

    Returns:
        유효한 진입가격 (0.0이면 실패)
    """
    logger = get_logger("entry_price")

    for attempt in range(max_retries):
        try:
            position = exchange.get_position(symbol)
            if position.side != "flat" and position.entry_price > 0:
                if fallback_price > 0 and position.entry_price != fallback_price:
                    logger.info(
                        f"[{exchange.name.upper()}] avgPrice 정상화: {symbol} {fallback_price} -> {position.entry_price}"
                    )
                return position.entry_price

            if attempt < max_retries - 1:
                logger.warning(
                    f"[{exchange.name.upper()}] avgPrice=0.0 재조회 시도 {attempt + 1}/{max_retries}: {symbol}"
                )
                time.sleep(retry_interval)
        except Exception as e:
            logger.error(
                f"[{exchange.name.upper()}] avgPrice 재조회 오류 {attempt + 1}: {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(retry_interval)

    # 최종 폴백
    if fallback_price > 0:
        logger.warning(
            f"[{exchange.name.upper()}] avgPrice 최종 폴백 사용: {symbol} -> {fallback_price}"
        )
        return fallback_price

    logger.error(f"[{exchange.name.upper()}] 유효한 진입가격 조회 실패: {symbol}")
    return 0.0


def check_existing_risk_orders(exchange: ExchangeBase, symbol: str) -> Dict[str, bool]:
    """
    기존 손절/트레일링 주문 상태 확인 - 중복 주문 방지

    Args:
        exchange: 거래소 클라이언트
        symbol: 심볼

    Returns:
        {"stop_loss": bool, "trailing_stop": bool}
    """
    logger = get_logger("risk_check")
    result = {"stop_loss": False, "trailing_stop": False}

    try:
        if exchange.name.lower() == "bybit":
            # Bybit: trading-stop 조회
            data = exchange._make_request(
                "GET", "/v5/position/list", {"category": "linear", "symbol": symbol}
            )
            if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                for pos_data in data["result"]["list"]:
                    stop_loss = pos_data.get("stopLoss", "")
                    trailing_stop = pos_data.get("trailingStop", "")

                    if stop_loss and float(stop_loss) > 0:
                        result["stop_loss"] = True
                        logger.info(f"[BYBIT] 기존 손절 감지: {symbol} @ {stop_loss}")

                    if trailing_stop and float(trailing_stop) > 0:
                        result["trailing_stop"] = True
                        logger.info(
                            f"[BYBIT] 기존 트레일링 감지: {symbol} distance={trailing_stop}"
                        )

                    break

        elif exchange.name.lower() == "bitget":
            # Bitget: plan orders 조회
            try:
                data = exchange._make_request(
                    "GET",
                    "/api/v2/mix/order/plan-orders-pending",
                    {"symbol": symbol, "productType": "USDT-FUTURES"},
                )
                if data.get("code") == "00000" and data.get("data"):
                    for order in data["data"]:
                        plan_type = order.get("planType", "")
                        if plan_type == "pos_loss":
                            result["stop_loss"] = True
                            logger.info(f"[BITGET] 기존 손절 감지: {symbol}")
                        elif plan_type == "track_plan":
                            result["trailing_stop"] = True
                            logger.info(f"[BITGET] 기존 트레일링 감지: {symbol}")
            except Exception as e:
                logger.warning(f"[BITGET] plan orders 조회 실패: {e}")

        return result

    except Exception as e:
        logger.error(f"[{exchange.name.upper()}] 기존 주문 확인 실패: {symbol} - {e}")
        return result


def get_entry_price_fast(
    exchange: ExchangeBase, symbol: str, side: str, timeout_sec: float = 2.0
) -> Tuple[Optional[float], str, bool]:
    """
    진입가(avg entry price)를 1~2초 내 가능한 빠르게 확보

    우선순위:
    (A) 캐시 → (B) REST 포지션 (Bitget 35회 재조회) → (C) 최근 체결 VWAP → (D) 주문내역 → (E) 최후 현재가 폴백

    Args:
        exchange: 거래소 클라이언트
        symbol: 심볼
        side: 포지션 방향 (long/short)
        timeout_sec: 타임아웃 (초)

    Returns:
        (entry_price, source, final)
        - final=True: 정확 값(평균진입가 확정)
        - final=False: 임시 폴백(현재가/체결기반)
    """
    logger = get_logger("entry_price_fast")
    start_time = time.time()

    # (A) 캐시 우선 확인 - Bitget 전용 캐시 사용
    try:
        if exchange.name.lower() == "bitget":
            cache_price = _check_bitget_entry_cache(symbol)
        else:
            cache_price = _check_entry_cache(symbol)

        if cache_price and cache_price > 0:
            # 현재가 대비 편차 확인 (0.4% 이내)
            bid, ask = exchange.get_ticker_price(symbol)
            current_price = (bid + ask) / 2 if bid > 0 and ask > 0 else ask or bid
            if current_price > 0:
                if _PriceUtils.within_deviation(cache_price, current_price, 0.4):
                    logger.info(
                        f"[{symbol}] Entry price confirmed via cache, final=True, price={cache_price:.8f}"
                    )
                    return cache_price, "cache", True
                else:
                    deviation = abs(cache_price - current_price) / current_price * 100
                    logger.warning(f"캐시 편차 초과 {deviation:.2f}%, 무효화: {symbol}")
    except Exception as e:
        logger.warning(f"캐시 확인 실패: {e}")

    # (B) REST 포지션 조회 (Bitget 35회 재조회 적용)
    try:
        if exchange.name.lower() == "bitget":
            # Bitget 전용: 35회 재조회
            for attempt in range(BITGET_POSITION_RETRY_COUNT):
                if time.time() - start_time > timeout_sec:
                    logger.warning(f"[BITGET] 포지션 조회 타임아웃: {symbol}")
                    break

                position = exchange.get_position(symbol)
                if position.side != "flat" and position.entry_price > 0:
                    # 현재가 대비 편차 10% 초과 시 비정상
                    bid, ask = exchange.get_ticker_price(symbol)
                    current_price = (
                        (bid + ask) / 2 if bid > 0 and ask > 0 else ask or bid
                    )
                    if current_price > 0:
                        if _PriceUtils.within_deviation(
                            position.entry_price, current_price, 10.0
                        ):
                            _update_bitget_entry_cache(symbol, position.entry_price)
                            logger.info(
                                f"[BITGET] Entry price confirmed via rest_position (attempt {attempt + 1})"
                            )
                            return position.entry_price, "rest_position", True
                        else:
                            deviation = (
                                abs(position.entry_price - current_price)
                                / current_price
                                * 100
                            )
                            logger.warning(
                                f"REST 포지션 편차 초과 {deviation:.2f}%, 재시도: {symbol}"
                            )

                if attempt < BITGET_POSITION_RETRY_COUNT - 1:
                    time.sleep(BITGET_POSITION_RETRY_INTERVAL)
        else:
            # 기타 거래소: 기존 방식
            position = exchange.get_position(symbol)
            if position.side != "flat" and position.entry_price > 0:
                bid, ask = exchange.get_ticker_price(symbol)
                current_price = (bid + ask) / 2 if bid > 0 and ask > 0 else ask or bid
                if current_price > 0:
                    if _PriceUtils.within_deviation(
                        position.entry_price, current_price, 10.0
                    ):
                        _update_entry_cache(symbol, position.entry_price)
                        logger.info(
                            f"[{symbol}] Entry price confirmed via rest_position"
                        )
                        return position.entry_price, "rest_position", True

    except Exception as e:
        logger.error(f"REST 포지션 조회 실패: {e}")

    # (C) 최근 체결 기반 VWAP
    try:
        if time.time() - start_time < timeout_sec:
            vwap_price = None

            # Bitget 전용 VWAP 계산
            if exchange.name.lower() == "bitget":
                vwap_price = exchange._calculate_fills_vwap_bitget(symbol, side)
                if vwap_price:
                    logger.info(f"[BITGET] Entry price confirmed via fills_vwap")
            else:
                # 기타 거래소는 범용 VWAP
                vwap_price = _get_fills_vwap(exchange, symbol, side)
                if vwap_price:
                    logger.info(f"[{symbol}] Entry price confirmed via fills_vwap")

            if vwap_price and vwap_price > 0:
                # 현재가 대비 편차 확인
                bid, ask = exchange.get_ticker_price(symbol)
                current_price = (bid + ask) / 2 if bid > 0 and ask > 0 else ask or bid
                if current_price > 0:
                    if _PriceUtils.within_deviation(vwap_price, current_price, 5.0):
                        return vwap_price, "fills_vwap", False
                    else:
                        deviation = (
                            abs(vwap_price - current_price) / current_price * 100
                        )
                        logger.warning(f"VWAP 편차 초과 {deviation:.2f}%: {symbol}")
    except Exception as e:
        logger.warning(f"VWAP 계산 실패: {e}")

    # (D) 주문내역 기반 평균가 (새로 추가)
    try:
        if time.time() - start_time < timeout_sec:
            logger.warning(f"[{symbol}] VWAP 실패, 주문내역 평균가 시도")
            order_avg_price = None

            # 거래소별 주문내역 조회
            if exchange.name.lower() == "bitget":
                # Bitget 전용: 새로 추가한 함수 사용
                try:
                    order_avg_price = exchange._get_recent_order_avg_price_bitget(
                        symbol, side
                    )
                    if order_avg_price:
                        logger.info(f"[BITGET] Entry price confirmed via order_history")
                except Exception as e:
                    logger.warning(f"[BITGET] order_history 함수 호출 실패: {e}")
            else:
                # 기타 거래소: 기존 범용 로직 (필요시 확장)
                try:
                    # 향후 Bybit 등 다른 거래소 지원 시 여기에 추가
                    logger.info(f"[{exchange.name.upper()}] order_history 지원 예정")
                except Exception:
                    pass

            if order_avg_price and order_avg_price > 0:
                # 현재가 대비 편차 재검증
                bid, ask = exchange.get_ticker_price(symbol)
                current_price = (bid + ask) / 2 if bid > 0 and ask > 0 else ask or bid
                if current_price > 0:
                    deviation = (
                        abs(order_avg_price - current_price) / current_price * 100
                    )
                    if deviation <= 8.0:  # 8% 이내 허용
                        return order_avg_price, "order_history", False
                    else:
                        logger.warning(
                            f"주문내역 평균가 편차 초과 {deviation:.2f}%: {symbol}"
                        )
                else:
                    logger.warning(f"현재가 조회 실패로 편차 검증 불가: {symbol}")
    except Exception as e:
        logger.warning(f"주문내역 평균가 계산 실패: {e}")

    # (E) 최후 수단: 현재가 폴백
    try:
        bid, ask = exchange.get_ticker_price(symbol)
        fallback_price = ask if side.lower() == "long" else bid
        if fallback_price > 0:
            logger.warning(
                f"[{exchange.name.upper()}] Entry price fallback to mark_price due to repeated failures: {symbol}"
            )
            return fallback_price, "fallback_current", False
    except Exception as e:
        logger.error(f"현재가 폴백도 실패: {e}")

    logger.critical(f"[{symbol}] Entry price failed completely")
    return None, "failed", False


# ============================================================================
# 보호장치 재시도 시스템 - 새로 추가
# ============================================================================


def check_existing_risk_orders(exchange: ExchangeBase, symbol: str) -> Dict[str, bool]:
    """
    기존 손절/트레일링 주문 상태 확인 - 중복 주문 방지

    Args:
        exchange: 거래소 클라이언트
        symbol: 심볼

    Returns:
        {"stop_loss": bool, "trailing_stop": bool}
    """
    logger = get_logger("risk_check")
    result = {"stop_loss": False, "trailing_stop": False}

    try:
        if exchange.name.lower() == "bybit":
            # Bybit: trading-stop 조회
            data = exchange._make_request(
                "GET", "/v5/position/list", {"category": "linear", "symbol": symbol}
            )
            if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                for pos_data in data["result"]["list"]:
                    stop_loss = pos_data.get("stopLoss", "")
                    trailing_stop = pos_data.get("trailingStop", "")

                    if stop_loss and float(stop_loss) > 0:
                        result["stop_loss"] = True
                        logger.info(f"[BYBIT] 기존 손절 감지: {symbol} @ {stop_loss}")

                    if trailing_stop and float(trailing_stop) > 0:
                        result["trailing_stop"] = True
                        logger.info(
                            f"[BYBIT] 기존 트레일링 감지: {symbol} distance={trailing_stop}"
                        )

                    break

        elif exchange.name.lower() == "bitget":
            # Bitget: plan orders 조회
            try:
                data = exchange._make_request(
                    "GET",
                    "/api/v2/mix/order/plan-orders-pending",
                    {"symbol": symbol, "productType": "USDT-FUTURES"},
                )
                if data.get("code") == "00000" and data.get("data"):
                    for order in data["data"]:
                        plan_type = order.get("planType", "")
                        if plan_type == "pos_loss":
                            result["stop_loss"] = True
                            logger.info(f"[BITGET] 기존 손절 감지: {symbol}")
                        elif plan_type == "track_plan":
                            result["trailing_stop"] = True
                            logger.info(f"[BITGET] 기존 트레일링 감지: {symbol}")
            except Exception as e:
                logger.warning(f"[BITGET] plan orders 조회 실패: {e}")

        return result

    except Exception as e:
        logger.error(f"[{exchange.name.upper()}] 기존 주문 확인 실패: {symbol} - {e}")
        return result


def enforce_protective_exits(
    exchange: ExchangeBase,
    symbol: str,
    side: str,
    entry_price: float,
    use_fixed_stop: bool,
    sl_pct: float,
    use_trailing: bool,
    ts_trigger_pct: float,
    ts_callback_pct: float,
    allow_sl_only: bool = True,
    max_retries: int = 5,
    retry_interval: float = 5.0,
) -> bool:
    """
    통합 보호장치 재시도 시스템 - 멱등성 및 avgPrice 문제 해결
    - SL/TS 설정 실패시 재시도
    - 기존 주문 상태 확인으로 중복 방지
    - avgPrice=0.0 문제 자동 해결
    - 최종 실패시 강제 청산
    - 상태 관리 통합
    """
    logger = get_logger("protection")
    symbol_state = state_manager.get_symbol_state(exchange.name, symbol)

    logger.info(f"[{exchange.name.upper()}] 보호장치 적용 시작: {symbol} {side}")

    # 기존 주문 상태 확인 (멱등성)
    existing_orders = check_existing_risk_orders(exchange, symbol)

    if existing_orders["stop_loss"] and existing_orders["trailing_stop"]:
        logger.info(
            f"[{exchange.name.upper()}] 손절/트레일링 이미 적용됨, skip: {symbol}"
        )
        symbol_state.stop_loss_set = True
        symbol_state.trail_active = True
        symbol_state.protection_retries = 0
        return True
    elif existing_orders["stop_loss"] and not use_trailing:
        logger.info(f"[{exchange.name.upper()}] 손절만 이미 적용됨, skip: {symbol}")
        symbol_state.stop_loss_set = True
        symbol_state.protection_retries = 0
        return True
    elif existing_orders["trailing_stop"] and not use_fixed_stop:
        logger.info(f"[{exchange.name.upper()}] 트레일링만 이미 적용됨, skip: {symbol}")
        symbol_state.trail_active = True
        symbol_state.protection_retries = 0
        return True

    # avgPrice=0.0 문제 해결 - 상위에서 이미 정확한 진입가를 전달받았으므로 최종 검증만
    if entry_price <= 0:
        logger.error(
            f"[{exchange.name.upper()}] entry_price가 여전히 무효함 ({entry_price}): {symbol}"
        )
        return False

    # 계약 정보 조회 (틱 사이즈)
    try:
        contract = exchange.get_contract_info(symbol)
        tick_size = contract.tick_size
    except Exception as e:
        logger.warning(f"계약 정보 조회 실패, 기본값 사용: {e}")
        tick_size = 0.01

    sl_success = existing_orders["stop_loss"]  # 기존에 있으면 성공으로 간주
    ts_success = existing_orders["trailing_stop"]  # 기존에 있으면 성공으로 간주

    # 재시도 루프
    for attempt in range(max_retries):
        try:
            logger.info(
                f"[{exchange.name.upper()}] 보호장치 시도 {attempt + 1}/{max_retries}"
            )

            # 현재 포지션 재조회 및 avgPrice 정상화 시도
            current_position = exchange.get_position(symbol)
            if current_position.side == "flat":
                logger.warning(f"포지션이 없음, 보호장치 설정 중단: {symbol}")
                return True  # 포지션 없으면 성공으로 간주

            # avgPrice 정상화 시도
            valid_entry_price = get_valid_entry_price(
                exchange, symbol, entry_price, max_retries=3, retry_interval=2.0
            )
            if valid_entry_price > 0 and valid_entry_price != entry_price:
                logger.info(
                    f"[{exchange.name.upper()}] entry_price 업데이트: {entry_price} -> {valid_entry_price}"
                )
                entry_price = valid_entry_price

            # 손절 설정 시도
            if use_fixed_stop and not sl_success:
                try:
                    logger.info(
                        f"[{exchange.name.upper()}] 손절 설정 시도: {symbol} {sl_pct}%"
                    )
                    sl_success = exchange.set_stop_loss(
                        symbol=symbol,
                        entry_price=entry_price,
                        side=side,
                        sl_pct=sl_pct,
                        tick_size=tick_size,
                    )
                    if sl_success:
                        symbol_state.stop_loss_set = True
                        logger.info(
                            f"[{exchange.name.upper()}] 손절 설정 성공: {symbol}"
                        )
                    else:
                        logger.warning(
                            f"[{exchange.name.upper()}] 손절 설정 실패: {symbol}"
                        )
                except Exception as e:
                    logger.error(
                        f"[{exchange.name.upper()}] 손절 설정 예외: {symbol} - {e}"
                    )
                    sl_success = False

            # 트레일링 설정 시도
            if use_trailing and not ts_success:
                try:
                    logger.info(
                        f"[{exchange.name.upper()}] 트레일링 설정 시도: {symbol} {ts_trigger_pct}%/{ts_callback_pct}%"
                    )
                    ts_success = exchange.set_trailing_stop(
                        symbol=symbol,
                        entry_price=entry_price,
                        side=side,
                        ts_trigger_pct=ts_trigger_pct,
                        ts_callback_pct=ts_callback_pct,
                        tick_size=tick_size,
                    )
                    if ts_success:
                        symbol_state.trail_active = True
                        logger.info(
                            f"[{exchange.name.upper()}] 트레일링 설정 성공: {symbol}"
                        )
                    else:
                        logger.warning(
                            f"[{exchange.name.upper()}] 트레일링 설정 실패: {symbol}"
                        )
                except Exception as e:
                    logger.error(
                        f"[{exchange.name.upper()}] 트레일링 설정 예외: {symbol} - {e}"
                    )
                    ts_success = False

            # 성공 조건 체크
            sl_ok = not use_fixed_stop or sl_success
            ts_ok = not use_trailing or ts_success

            if sl_ok and ts_ok:
                logger.info(
                    f"[{exchange.name.upper()}] 보호장치 최종 적용 완료: {symbol}"
                )
                symbol_state.protection_retries = 0
                return True

            # 부분 성공 정책
            if allow_sl_only and sl_success and use_fixed_stop:
                if not use_trailing:
                    logger.info(f"[{exchange.name.upper()}] 손절만 적용 완료: {symbol}")
                    symbol_state.protection_retries = 0
                    return True
                elif not ts_success:
                    logger.warning(
                        f"[{exchange.name.upper()}] 손절만 유지 허용: {symbol}"
                    )
                    symbol_state.protection_retries = 0
                    return True

            # 재시도 대기
            if attempt < max_retries - 1:
                logger.info(
                    f"[{exchange.name.upper()}] {retry_interval}초 후 재시도: {symbol}"
                )
                time.sleep(retry_interval)

        except Exception as e:
            logger.error(
                f"[{exchange.name.upper()}] 보호장치 시도 중 예외: {symbol} - {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(retry_interval)

    # 최종 실패 처리
    symbol_state.protection_retries = max_retries
    symbol_state.last_protection_attempt = time.time()

    logger.critical(
        f"[{exchange.name.upper()}] 보호장치 최종 실패 → 강제 청산: {symbol}"
    )

    # 강제 청산 시도
    try:
        close_success = exchange.close_position(symbol)
        if close_success:
            logger.critical(f"[{exchange.name.upper()}] 강제 청산 완료: {symbol}")
            # 포지션 상태 업데이트
            flat_position = Position(symbol, "flat", 0.0, 0.0)
            state_manager.update_position(exchange.name, symbol, flat_position)
            # 보호장치 쿨다운 설정
            state_manager.set_protection_cooldown(exchange.name, symbol)
            return False  # 보호장치 실패이지만 안전하게 청산됨
        else:
            logger.critical(
                f"[{exchange.name.upper()}] 강제 청산도 실패: {symbol} - 수동 개입 필요!"
            )
            state_manager.set_protection_cooldown(exchange.name, symbol)
            return False
    except Exception as e:
        logger.critical(f"[{exchange.name.upper()}] 강제 청산 예외: {symbol} - {e}")
        state_manager.set_protection_cooldown(exchange.name, symbol)
        return False


# ============================================================================
# 리스크 & 주문 관리자 - 보호장치 시스템 교체
# ============================================================================


class RiskManager:
    """개선된 리스크 관리자 - 보호장치 재시도 시스템 적용"""

    def __init__(self):
        self.logger = get_logger("risk")

    def calculate_position_size(
        self,
        exchange: ExchangeBase,
        symbol: str,
        side: str,
        balance: float,
        current_price: float,
        quote_pct: float,
        leverage: float,
        enable_fallback: bool = True,
        max_fallback_risk_pct: float = 50.0,
    ) -> float:
        """개선된 포지션 크기 계산 - 리스크 체크 간소화"""
        try:
            if balance <= 0 or current_price <= 0:
                return 0.0

            # 기본 계산
            risk_amount = balance * (quote_pct / 100.0)
            position_value = risk_amount * leverage
            quantity = position_value / current_price

            # 수량 조정 (여기서 fallback 로직 포함)
            adjusted_qty = exchange.adjust_quantity(
                symbol,
                quantity,
                enable_fallback=enable_fallback,
                max_risk_pct=max_fallback_risk_pct,
                total_balance=balance,
            )

            # fallback 적용 로그
            if adjusted_qty != quantity and adjusted_qty > 0:
                contract = exchange.get_contract_info(symbol)
                if quantity < contract.min_qty <= adjusted_qty:
                    actual_risk_pct = (adjusted_qty * current_price / balance) * 100
                    self.logger.info(
                        f"최소수량 진입 적용: {symbol} "
                        f"계획수량={quantity:.6f} -> 실제수량={adjusted_qty:.6f} "
                        f"(실제리스크: {actual_risk_pct:.1f}%)"
                    )

            return adjusted_qty
        except Exception as e:
            self.logger.error(f"포지션 크기 계산 실패: {e}")
            return 0.0

    def apply_risk_controls(
        self,
        exchange: ExchangeBase,
        symbol: str,
        side: str,
        entry_price: float,
        use_fixed_stop: bool,
        fixed_stop_pct: float,
        use_trailing: bool,
        trail_trigger_pct: float,
        trail_callback_pct: float,
        config: TradingConfig,
    ) -> Dict[str, bool]:
        """통합된 리스크 컨트롤 적용 - 빠른 진입가 확보 시스템으로 교체"""

        self.logger.info(f"[{exchange.name.upper()}] 리스크 컨트롤 적용 시작: {symbol}")

        try:
            # 정확한 진입가 확보 (빠른 조회 시스템 + 향상된 로깅)
            entry_price_actual, entry_src, is_final = get_entry_price_fast(
                exchange, symbol, side, timeout_sec=3.0
            )

            if entry_price_actual is None or entry_price_actual <= 0:
                self.logger.critical(
                    f"[{exchange.name.upper()}] Entry price failed completely → 보호장치 중단: {symbol}"
                )
                return {"protection_system": False, "forced_close": False}

            # 진입가 확보 성공 로깅
            if entry_src == "rest_position":
                if exchange.name.lower() == "bitget":
                    self.logger.info(
                        f"[BITGET] Entry price confirmed via rest_position"
                    )
                else:
                    self.logger.info(
                        f"[{exchange.name.upper()}] Entry price confirmed via rest_position"
                    )
            elif entry_src == "fills_vwap":
                self.logger.info(
                    f"[{exchange.name.upper()}] Entry price confirmed via fills_vwap"
                )
            elif entry_src == "order_history":
                self.logger.info(
                    f"[{exchange.name.upper()}] Entry price confirmed via order_history"
                )
            elif entry_src == "cache":
                self.logger.info(
                    f"[{exchange.name.upper()}] Entry price confirmed via cache"
                )
            elif entry_src == "fallback_current":
                self.logger.warning(
                    f"[{exchange.name.upper()}] Entry price fallback to mark_price due to repeated failures"
                )

            # 캐시 업데이트 (거래소별)
            if is_final:
                if exchange.name.lower() == "bitget":
                    _update_bitget_entry_cache(symbol, entry_price_actual)
                else:
                    _update_entry_cache(symbol, entry_price_actual)

            # 정확한 진입가로 보호장치 재시도 시스템 호출
            success = enforce_protective_exits(
                exchange=exchange,
                symbol=symbol,
                side=side,
                entry_price=entry_price_actual,  # 새로 확보한 정확한 진입가 사용
                use_fixed_stop=use_fixed_stop,
                sl_pct=fixed_stop_pct,
                use_trailing=use_trailing,
                ts_trigger_pct=trail_trigger_pct,
                ts_callback_pct=trail_callback_pct,
                allow_sl_only=config.ALLOW_SL_ONLY,
                max_retries=config.MAX_PROTECTION_RETRIES,
                retry_interval=config.PROTECTION_RETRY_INTERVAL,
            )

            if success:
                self.logger.info(
                    f"[{exchange.name.upper()}] 리스크 컨트롤 최종 성공: {symbol}"
                )
                return {"protection_system": True, "forced_close": False}
            else:
                self.logger.critical(
                    f"[{exchange.name.upper()}] 리스크 컨트롤 실패 → 안전 조치 완료: {symbol}"
                )
                return {"protection_system": False, "forced_close": True}

        except Exception as e:
            self.logger.error(
                f"[{exchange.name.upper()}] 리스크 컨트롤 시스템 오류: {symbol} - {e}"
            )
            return {"protection_system": False, "forced_close": False}


# 전역 리스크 관리자
risk_manager = RiskManager()


# ============================================================================
# 핵심 신호 처리 로직 - 기존 유지
# ============================================================================


def create_flat_position(symbol: str) -> Position:
    """플랫 포지션 생성 헬퍼"""
    return Position(symbol, "flat", 0.0, 0.0)


def handle_signal(
    exchange: ExchangeBase, symbol: str, side: str, config: TradingConfig
) -> Dict[str, Any]:
    """핵심 신호 처리 함수"""
    logger = get_logger("signal")
    logger.info(f"신호 처리 시작 [{exchange.name}] {symbol} {side}")

    try:
        # 현재 포지션 확인
        current_position = exchange.get_position(symbol)

        # 상황별 처리 분기
        if current_position.side == "flat":
            # 무포지션 -> 신규 진입
            return open_position(exchange, symbol, side, config)

        elif current_position.side == side:
            # 동일 방향 신호 무시
            return {
                "success": False,
                "action": "ignored",
                "reason": f"이미 {side} 포지션",
            }

        else:
            # 반대 신호 처리
            return manage_position(exchange, symbol, side, current_position, config)

    except Exception as e:
        logger.error(f"신호 처리 실패 [{exchange.name}] {symbol}: {e}")
        return {"success": False, "error": str(e)}


def manage_position(
    exchange: ExchangeBase,
    symbol: str,
    new_side: str,
    current_position: Position,
    config: TradingConfig,
) -> Dict[str, Any]:
    """포지션 관리 함수 - 올바른 리버스 로직"""
    logger = get_logger("position")

    # 손절/트레일링 활성화 상태 확인
    has_risk_controls = config.USE_FIXED_STOP or config.USE_TRAILING_STOP

    if has_risk_controls:
        # 손절/트레일링이 ON -> 반대 신호 무시 (손절/트레일링이 알아서 청산함)
        return {
            "success": False,
            "action": "ignored_due_to_risk_controls",
            "reason": f"손절/트레일링이 활성화되어 반대 신호 무시. 현재 포지션: {current_position.side}",
        }
    else:
        # 손절/트레일링이 OFF -> 반대 신호로 즉시 청산 후 반전 (리버스)
        logger.info(
            f"리버스 신호 감지 (손절/트레일링 OFF): {current_position.side} -> {new_side}"
        )
        return close_position(
            exchange, symbol, current_position, config, reverse_to=new_side
        )


def close_position(
    exchange: ExchangeBase,
    symbol: str,
    current_position: Position,
    config: TradingConfig,
    reverse_to: str = None,
) -> Dict[str, Any]:
    """포지션 청산 함수"""
    logger = get_logger("close")

    # 청산 사유 결정
    if reverse_to:
        close_reason = "opposite signal"
    else:
        close_reason = "manual close"

    try:
        # 기존 포지션 청산
        close_results = exchange.close_all_positions(symbol)
        close_result = close_results.get(symbol)

        if not close_result or not close_result.success:
            error_msg = close_result.message if close_result else "청산 실패"
            return {"success": False, "error": error_msg}

        # 상태 업데이트
        flat_position = create_flat_position(symbol)
        state_manager.update_position(exchange.name, symbol, flat_position)

        # 청산 사유 로깅
        logger.info(
            f"Position closed by {close_reason}: {symbol} {current_position.side} {current_position.qty}"
        )

        result = {
            "success": True,
            "action": "position_closed",
            "close_reason": close_reason,
            "closed_side": current_position.side,
            "closed_qty": current_position.qty,
            "order_id": close_result.order_id,
        }

        # 반전 진입 처리
        if reverse_to:
            time.sleep(1.0)  # 청산 완료 확인

            # 반대 포지션 진입
            open_result = open_position(exchange, symbol, reverse_to, config)
            if open_result.get("success"):
                result.update(
                    {
                        "action": "position_reversed",
                        "new_side": reverse_to,
                        "new_qty": open_result.get("qty", 0),
                        "new_entry_price": open_result.get("entry_price", 0),
                        "new_order_id": open_result.get("order_id", ""),
                    }
                )
            else:
                result["reverse_error"] = open_result.get("error", "반전 진입 실패")
        else:
            # 쿨다운 설정
            state_manager.set_cooldown(
                exchange.name, symbol, config.COOLDOWN_AFTER_CLOSE_SEC
            )

        return result

    except Exception as e:
        logger.error(f"포지션 청산 오류 [{exchange.name}] {symbol}: {e}")
        return {"success": False, "error": str(e)}


def open_position(
    exchange: ExchangeBase, symbol: str, side: str, config: TradingConfig
) -> Dict[str, Any]:
    """개선된 신규 포지션 진입 함수 - 보호장치 재시도 시스템 적용"""
    logger = get_logger("open")

    try:
        # 가격 조회
        bid, ask = exchange.get_ticker_price(symbol)
        entry_price = ask if side.lower() == "long" else bid
        if entry_price <= 0:
            return {"success": False, "error": "가격 조회 실패"}

        # 잔고 조회
        balance = exchange.get_account_balance()
        if balance.available <= 0:
            return {"success": False, "error": "잔고 부족"}

        # 포지션 크기 계산 (여기서 이미 adjust_quantity 포함됨)
        qty = risk_manager.calculate_position_size(
            exchange,
            symbol,
            side,
            balance.available,
            entry_price,
            config.QUOTE_PCT,
            config.LEVERAGE,
            enable_fallback=config.ENABLE_MIN_QTY_FALLBACK,
            max_fallback_risk_pct=config.MAX_FALLBACK_RISK_PCT,
        )

        if qty <= 0:
            return {"success": False, "error": "계산된 수량이 유효하지 않음"}

        # 레버리지 설정
        leverage_result = exchange.set_leverage(symbol, config.LEVERAGE)
        if not leverage_result:
            logger.warning(
                f"레버리지 설정 실패 (계속 진행): {symbol} x{config.LEVERAGE}"
            )

        # 주문 실행
        order_result = exchange.place_market_order(symbol, side, qty)
        if not order_result.success:
            return {"success": False, "error": f"주문 실패: {order_result.message}"}

        # 포지션 상태 업데이트 - 새로운 빠른 진입가 확보 시스템 사용
        time.sleep(0.3)  # 주문 체결 완료 대기

        # 빠른 진입가 확보 시스템으로 정확한 avgPrice 취득
        actual_entry_price, entry_src, is_final = get_entry_price_fast(
            exchange, symbol, side, timeout_sec=3.0
        )

        if actual_entry_price and actual_entry_price > 0:
            if actual_entry_price != entry_price:
                logger.info(
                    f"[{exchange.name.upper()}] 진입가 업데이트: {entry_price} -> {actual_entry_price} (source={entry_src})"
                )
            entry_price = actual_entry_price

            # 캐시 업데이트 (정확한 값만)
            if is_final:
                _update_entry_cache(symbol, actual_entry_price)
        else:
            logger.warning(
                f"[{exchange.name.upper()}] 진입가 확보 실패, 계산값 유지: {symbol} @ {entry_price}"
            )

        # 실제 포지션으로 상태 업데이트
        actual_position = exchange.get_position(symbol)
        if actual_position.side != "flat":
            # 확보한 진입가로 포지션 정보 보정
            actual_position.entry_price = entry_price
            new_position = actual_position
        else:
            # 폴백: 계산된 포지션 생성
            new_position = Position(
                symbol=symbol,
                side=side,
                qty=qty,
                entry_price=entry_price,
                mark_price=entry_price,
            )

        state_manager.update_position(exchange.name, symbol, new_position)

        logger.info(
            f"[{exchange.name.upper()}] 신규 진입 완료: {symbol} {side} {qty:.6f} @ {entry_price}"
        )

        # ========================================
        # 보호장치 재시도 시스템 적용 (기존 코드 교체)
        # ========================================

        risk_results = risk_manager.apply_risk_controls(
            exchange=exchange,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            use_fixed_stop=config.USE_FIXED_STOP,
            fixed_stop_pct=config.FIXED_STOP_PCT,
            use_trailing=config.USE_TRAILING_STOP,
            trail_trigger_pct=config.TRAIL_TRIGGER_PCT,
            trail_callback_pct=config.TRAIL_CALLBACK_PCT,
            config=config,
        )

        # 결과에 따른 처리
        if risk_results.get("forced_close"):
            logger.critical(
                f"[{exchange.name.upper()}] 보호장치 실패로 진입 직후 강제 청산됨: {symbol}"
            )
            return {
                "success": False,
                "error": "보호장치 설정 실패로 안전을 위해 포지션이 강제 청산되었습니다",
                "action": "forced_close_after_entry",
            }
        elif not risk_results.get("protection_system"):
            logger.warning(
                f"[{exchange.name.upper()}] 보호장치 부분 실패, 포지션 유지: {symbol}"
            )

        return {
            "success": True,
            "action": "position_opened",
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "entry_price": entry_price,
            "order_id": order_result.order_id,
            "risk_controls": risk_results,
        }

    except Exception as e:
        logger.error(f"신규 진입 실패 [{exchange.name}] {symbol}: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# 서버 메인 함수
# ============================================================================


def server_main(config_dict: Dict[str, Any], stop_event: mp.Event, log_queue: mp.Queue):
    """서버 메인 함수"""
    setup_logging("INFO", log_queue)
    logger = get_logger("server")

    try:
        config = TradingConfig.from_dict(config_dict)

        logger.info("=" * 60)
        logger.info("Stargate Multi-Exchange Trading Server v4.2")
        logger.info("Enhanced Risk Protection System")
        logger.info(f"거래소: {config.SELECTED_EXCHANGES}")
        logger.info(
            f"손절/트레일링: {config.USE_FIXED_STOP}/{config.USE_TRAILING_STOP}"
        )
        logger.info(
            f"최소수량 진입: {config.ENABLE_MIN_QTY_FALLBACK} (최대리스크: {config.MAX_FALLBACK_RISK_PCT}%)"
        )
        logger.info(
            f"보호장치 재시도: {config.MAX_PROTECTION_RETRIES}회 / {config.PROTECTION_RETRY_INTERVAL}초 간격"
        )
        logger.info(f"손절만 허용: {config.ALLOW_SL_ONLY}")
        logger.info("실시간 API 기반 Bitget 심볼 매핑 시스템")
        logger.info("재시도 기반 보호장치 안전 시스템")
        logger.info("=" * 60)

        # 거래소 연결 및 포지션 동기화
        exchange_clients = {}
        for exchange_name in config.SELECTED_EXCHANGES:
            try:
                keys = config.EXCHANGE_KEYS.get(exchange_name)
                if not keys or not keys.api_key or not keys.api_secret:
                    logger.error(f"API 키 누락 [{exchange_name}]")
                    continue

                passphrase = getattr(keys, "passphrase", "") or ""
                client = create_exchange_client(
                    exchange_name, keys.api_key, keys.api_secret, passphrase, False
                )

                if client:
                    ok, msg = client.validate_credentials()
                    if ok:
                        exchange_clients[exchange_name] = client
                        logger.info(f"연결 성공 [{exchange_name}]: {msg}")

                        # 포지션 동기화
                        try:
                            positions = client.get_all_positions()
                            state_manager.sync_positions(exchange_name, positions)
                            state_manager.exchanges[exchange_name].connected = True

                            # 서버 재시작 시 기존 포지션에 대한 스마트 리스크 재적용
                            if positions:
                                logger.info(
                                    f"[{exchange_name.upper()}] 기존 포지션 발견, 스마트 리스크 점검 시작"
                                )
                                for position in positions:
                                    if position.side != "flat":
                                        try:
                                            # 기존 주문 상태 확인 (멱등성)
                                            existing_orders = (
                                                check_existing_risk_orders(
                                                    client, position.symbol
                                                )
                                            )

                                            sl_needed = (
                                                config.USE_FIXED_STOP
                                                and not existing_orders["stop_loss"]
                                            )
                                            ts_needed = (
                                                config.USE_TRAILING_STOP
                                                and not existing_orders["trailing_stop"]
                                            )

                                            if sl_needed or ts_needed:
                                                logger.info(
                                                    f"[{exchange_name.upper()}] 리스크 누락 감지, 재적용: {position.symbol}"
                                                )
                                                enforce_protective_exits(
                                                    exchange=client,
                                                    symbol=position.symbol,
                                                    side=position.side,
                                                    entry_price=position.entry_price,
                                                    use_fixed_stop=sl_needed,
                                                    sl_pct=config.FIXED_STOP_PCT,
                                                    use_trailing=ts_needed,
                                                    ts_trigger_pct=config.TRAIL_TRIGGER_PCT,
                                                    ts_callback_pct=config.TRAIL_CALLBACK_PCT,
                                                    allow_sl_only=config.ALLOW_SL_ONLY,
                                                    max_retries=3,  # 서버 시작 시에는 빠르게
                                                    retry_interval=2.0,
                                                )
                                            else:
                                                # 기존 SL/TS 존재시 상태만 동기화
                                                symbol_state = (
                                                    state_manager.get_symbol_state(
                                                        exchange_name, position.symbol
                                                    )
                                                )
                                                symbol_state.stop_loss_set = (
                                                    existing_orders["stop_loss"]
                                                )
                                                symbol_state.trail_active = (
                                                    existing_orders["trailing_stop"]
                                                )
                                                logger.info(
                                                    f"[{exchange_name.upper()}] 리스크 이미 적용됨, skip: {position.symbol}"
                                                )
                                        except Exception as e:
                                            logger.error(
                                                f"[{exchange_name.upper()}] 기존 포지션 리스크 점검 실패: {position.symbol} - {e}"
                                            )

                        except Exception as e:
                            logger.error(f"포지션 동기화 실패 [{exchange_name}]: {e}")
                    else:
                        logger.error(f"연결 실패 [{exchange_name}]: {msg}")
                else:
                    logger.error(f"클라이언트 생성 실패 [{exchange_name}]")
            except Exception as e:
                logger.error(f"거래소 초기화 실패 [{exchange_name}]: {e}")

        if not exchange_clients:
            logger.error("사용 가능한 거래소가 없습니다!")
            return

        # ngrok URL 감지
        public_url = detect_ngrok_url()
        if public_url:
            logger.info(f"ngrok 감지: {public_url}")
            log_queue.put(f"PUBLIC_URL:{public_url}")

        # Flask 앱 생성
        app = Flask(__name__)

        @app.route("/healthz", methods=["GET"])
        def healthz():
            """헬스체크 엔드포인트"""
            try:
                return jsonify(
                    {
                        "status": "ok",
                        "timestamp": int(time.time() * 1000),
                        "exchanges": list(exchange_clients.keys()),
                        "positions": sum(
                            len(positions)
                            for positions in state_manager.get_all_positions().values()
                        ),
                        "version": "4.2",
                        "features": {
                            "bitget_symbol_mapping": True,
                            "min_qty_fallback": config.ENABLE_MIN_QTY_FALLBACK,
                            "max_fallback_risk_pct": config.MAX_FALLBACK_RISK_PCT,
                            "enhanced_risk_protection": True,
                            "protection_retries": config.MAX_PROTECTION_RETRIES,
                            "allow_sl_only": config.ALLOW_SL_ONLY,
                        },
                    }
                )
            except Exception as e:
                logger.error(f"헬스체크 오류: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

        @app.route("/status", methods=["GET"])
        def status():
            """상태 조회 엔드포인트"""
            try:
                balances = {}
                auth_status = {}

                # 각 거래소 상태 확인
                for name, client in exchange_clients.items():
                    try:
                        ok, msg = client.validate_credentials()
                        auth_status[name] = {"ok": ok, "message": msg}
                        if ok:
                            balance = client.get_account_balance()
                            balances[name] = {balance.currency: balance.available}
                        else:
                            balances[name] = {}
                    except Exception as e:
                        auth_status[name] = {"ok": False, "message": str(e)}
                        balances[name] = {}

                # 포지션 정보 수집
                all_positions = state_manager.get_all_positions()
                positions_with_pnl = {}

                for exchange_name, positions in all_positions.items():
                    positions_with_pnl[exchange_name] = {}
                    for symbol, position in positions.items():
                        positions_with_pnl[exchange_name][symbol] = {
                            "side": position.side,
                            "qty": position.qty,
                            "entry_price": position.entry_price,
                            "mark_price": position.mark_price,
                            "uPNL": position.unrealized_pnl,
                            "category": position.category,
                        }

                return jsonify(
                    {
                        "server_running": True,
                        "public_url": public_url,
                        "exchanges": list(exchange_clients.keys()),
                        "auth_status": auth_status,
                        "balances": balances,
                        "positions": positions_with_pnl,
                        "exchange_status": state_manager.get_exchange_status(),
                        "config": config.get_safe_dict(),
                        "timestamp": int(time.time() * 1000),
                        "version": "4.2",
                    }
                )
            except Exception as e:
                logger.error(f"Status 조회 실패: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route("/webhook", methods=["POST"])
        def webhook():
    reverse_on_opposite = (not trailing_enabled_from_cfg(locals().get('cfg', locals().get('settings', {}))))  # GPT PATCH: trailing OFF => reverse
            """웹훅 엔드포인트"""
            try:
                # IP 화이트리스트 검증
                if config.SERVER.ip_whitelist:
                    if not check_ip_whitelist(
                        request.remote_addr, config.SERVER.ip_whitelist
                    ):
                        return jsonify({"error": "IP not allowed"}), 403

                # 웹훅 서명 검증
                if config.SERVER.webhook_secret:
                    signature = request.headers.get("X-Signature", "")
                    ts = request.headers.get("X-Timestamp", "")
                    if not verify_webhook_signature(
                        request.get_data(), signature, config.SERVER.webhook_secret, ts
                    ):
                        return jsonify({"error": "Invalid signature"}), 401

                # 데이터 파싱
                data = request.get_json(force=True, silent=True) or {}

                symbol = (data.get("symbol", "") or "").strip().upper()
                side = (data.get("side", "") or "").strip().lower()
                signal_type = (data.get("type", "strong") or "").strip().lower()
                exchange_name = (
                    (data.get("exchange", config.SELECTED_EXCHANGES[0]) or "")
                    .strip()
                    .lower()
                )

                # 입력 검증
                if not symbol:
                    return jsonify({"error": "symbol 필드 필수"}), 400

                valid_sides = ["buy", "sell", "long", "short"]
                if side not in valid_sides:
                    return (
                        jsonify({"error": f"side는 {valid_sides} 중 하나여야 합니다"}),
                        400,
                    )

                # Side 정규화
                if side in ("buy", "long"):
                    normalized_side = "long"
                elif side in ("sell", "short"):
                    normalized_side = "short"
                else:
                    return jsonify({"error": f"예상치 못한 side 값: '{side}'"}), 400

                if exchange_name not in exchange_clients:
                    return (
                        jsonify({"error": f"거래소 '{exchange_name}' 사용 불가"}),
                        400,
                    )

                exchange_client = exchange_clients[exchange_name]

                # 원본 심볼 저장 및 정규화
                original_symbol = symbol
                normalized_symbol = exchange_client.normalize_symbol(symbol)

                # 변환된 경우 추가 로깅
                if original_symbol != normalized_symbol:
                    logger.info(
                        f"[{exchange_name.upper()}] 심볼 자동변환: {original_symbol} -> {normalized_symbol}"
                    )

                # 신호 검증
                can_process, reason = state_manager.can_process_signal(
                    exchange_name,
                    normalized_symbol,
                    normalized_side,
                    config.MIN_SIGNAL_INTERVAL_SEC,
                )
                if not can_process:
                    return jsonify({"success": False, "error": reason}), 202

                # 신호 세기 검증
                if signal_type.lower() != "strong":
                    return (
                        jsonify(
                            {
                                "success": False,
                                "error": f"약한 신호 무시: {signal_type}",
                            }
                        ),
                        202,
                    )

                # 신호 필터 적용
                state_manager.set_signal_filter(
                    exchange_name, normalized_symbol, normalized_side
                )

                # 핵심 신호 처리
                result = handle_signal(
                    exchange_client, normalized_symbol, normalized_side, config
                )

                # 상태 저장
                try:
                    state_manager.save_state()
                except Exception as e:
                    logger.error(f"상태 저장 실패: {e}")

                status_code = 200 if result.get("success") else 202
                return jsonify(result), status_code

            except Exception as e:
                logger.error(f"웹훅 처리 실패: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route("/sync", methods=["POST"])
        def sync_positions():
            """포지션 동기화 엔드포인트"""
            try:
                total_synced = 0
                errors = []

                for exchange_name, client in exchange_clients.items():
                    try:
                        positions = client.get_all_positions()
                        state_manager.sync_positions(exchange_name, positions)
                        total_synced += len(positions)
                    except Exception as e:
                        errors.append(f"동기화 실패 [{exchange_name}]: {e}")

                try:
                    state_manager.save_state()
                except Exception as e:
                    logger.error(f"상태 저장 실패: {e}")

                result = {
                    "success": True,
                    "synced_positions": total_synced,
                    "positions": state_manager.get_all_positions(),
                }

                if errors:
                    result["warnings"] = errors

                return jsonify(result)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @app.route("/apply_risk", methods=["POST"])
        def apply_risk():
            """리스크 컨트롤 재적용 엔드포인트"""
            try:
                applied_count = 0
                errors = []

                for exchange_name, client in exchange_clients.items():
                    positions = state_manager.get_all_positions().get(exchange_name, {})
                    for symbol, position in positions.items():
                        if position.side != "flat":
                            try:
                                # 새로운 보호장치 재시도 시스템 사용
                                success = enforce_protective_exits(
                                    exchange=client,
                                    symbol=symbol,
                                    side=position.side,
                                    entry_price=position.entry_price,
                                    use_fixed_stop=config.USE_FIXED_STOP,
                                    sl_pct=config.FIXED_STOP_PCT,
                                    use_trailing=config.USE_TRAILING_STOP,
                                    ts_trigger_pct=config.TRAIL_TRIGGER_PCT,
                                    ts_callback_pct=config.TRAIL_CALLBACK_PCT,
                                    allow_sl_only=config.ALLOW_SL_ONLY,
                                    max_retries=config.MAX_PROTECTION_RETRIES,
                                    retry_interval=config.PROTECTION_RETRY_INTERVAL,
                                )
                                if success:
                                    applied_count += 1
                                else:
                                    errors.append(
                                        f"보호장치 적용 실패 [{exchange_name}] {symbol}"
                                    )
                            except Exception as e:
                                errors.append(
                                    f"리스크 적용 실패 [{exchange_name}] {symbol}: {e}"
                                )

                result = {"success": True, "applied_positions": applied_count}
                if errors:
                    result["warnings"] = errors

                return jsonify(result)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @app.route("/close_all", methods=["POST"])
        def close_all_positions():
            """모든 포지션 강제 청산 엔드포인트"""
            try:
                data = request.get_json(force=True, silent=True) or {}
                target_symbol = (
                    data.get("symbol", "").strip().upper()
                    if data.get("symbol")
                    else None
                )

                total_closed = 0
                results = {}
                errors = []

                for exchange_name, client in exchange_clients.items():
                    try:
                        close_results = client.close_all_positions(target_symbol)
                        for symbol, result in close_results.items():
                            if result.success:
                                total_closed += 1
                                # 상태 업데이트
                                flat_position = create_flat_position(symbol)
                                state_manager.update_position(
                                    exchange_name, symbol, flat_position
                                )
                                logger.info(
                                    f"Position closed by manual close: {symbol}"
                                )
                            else:
                                errors.append(
                                    f"포지션 청산 실패 [{exchange_name}] {symbol}: {result.message}"
                                )
                        results[exchange_name] = close_results
                    except Exception as e:
                        results[exchange_name] = {"error": str(e)}
                        errors.append(f"포지션 청산 오류 [{exchange_name}]: {e}")

                try:
                    state_manager.save_state()
                except Exception as e:
                    logger.error(f"상태 저장 실패: {e}")

                result = {
                    "success": True,
                    "closed_positions": total_closed,
                    "results": results,
                }

                if errors:
                    result["warnings"] = errors

                return jsonify(result)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        # 서버 시작 알림
        log_queue.put("SERVER_STARTED")

        logger.info(f"서버 시작: http://{config.SERVER.host}:{config.SERVER.port}")
        if public_url:
            logger.info(f"Public URL: {public_url}")

        logger.info("Enhanced Risk Protection System 준비 완료")

        # Flask 서버 실행
        app.run(
            host=config.SERVER.host,
            port=config.SERVER.port,
            threaded=True,
            use_reloader=False,
            debug=False,
        )

    except KeyboardInterrupt:
        logger.info("서버 중지 (Ctrl+C)")
    except Exception as e:
        logger.error(f"서버 실행 오류: {e}")
        log_queue.put(f"ERROR:{e}")
    finally:
        try:
            # Bitget trailing workers 정리
            for exchange_name, client in exchange_clients.items():
                if hasattr(client, "stop_trailing_workers"):
                    client.stop_trailing_workers()

            state_manager.save_state()
            logger.info("서버 종료 시 상태 저장 완료")
        except Exception as e:
            logger.error(f"종료 시 상태 저장 실패: {e}")
        logger.info("서버 종료")


# ============================================================================
# GUI 인터페이스 - 새로운 설정 항목 추가
# ============================================================================


class StargateMultiExchangeGUI:
    """Stargate 멀티거래소 지원 GUI v4.2 - Enhanced Risk Protection"""

    def __init__(self):
        self.config = get_config()
        self.log_queue = mp.Queue()
        setup_logging("INFO", self.log_queue)
        self.logger = get_logger("gui")

        self.root = tk.Tk()
        self.root.title(
            "Stargate Multi-Exchange Trading Server v4.2 - Enhanced Risk Protection"
        )
        self.root.geometry("1000x850")
        self.root.minsize(900, 750)

        # 스타일 설정
        self.style = ttk.Style()
        try:
            self.style.theme_use("vista")
        except:
            self.style.theme_use("clam")

        self.server_process: Optional[mp.Process] = None
        self.stop_event = mp.Event()

        # 상태 변수
        self.server_status = tk.StringVar(value="서버 중지됨")
        self.public_url = tk.StringVar(value="Not detected")

        # 거래소별 API 키 변수
        self.exchange_vars = {}
        self._init_exchange_vars()

        self._create_widgets()
        self._start_log_monitor()
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.logger.info("Multi-Exchange GUI v4.2 초기화 완료")

    def _init_exchange_vars(self):
        """거래소별 API 키 변수 초기화"""
        for exchange in SUPPORTED_EXCHANGES:
            self.exchange_vars[exchange] = {
                "api_key": tk.StringVar(),
                "api_secret": tk.StringVar(),
                "passphrase": tk.StringVar(),
            }

            # 기존 키 로드
            keys = config_manager.get_exchange_keys(exchange)
            if keys:
                self.exchange_vars[exchange]["api_key"].set(keys.api_key)
                self.exchange_vars[exchange]["api_secret"].set(keys.api_secret)
                passphrase = getattr(keys, "passphrase", "") or ""
                self.exchange_vars[exchange]["passphrase"].set(passphrase)

    def _create_widgets(self):
        """GUI 위젯 생성"""
        main_container = ttk.Frame(self.root)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        self._create_exchange_config_section(main_container)
        self._create_trading_config_section(main_container)
        self._create_status_section(main_container)
        self._create_control_section(main_container)

    def _create_exchange_config_section(self, parent):
        """거래소 설정 섹션"""
        config_frame = ttk.LabelFrame(parent, text="거래소 API 설정", padding=10)
        config_frame.pack(fill="x", pady=(0, 10))

        # 거래소 선택
        exchange_select_frame = ttk.Frame(config_frame)
        exchange_select_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(
            exchange_select_frame, text="활성 거래소:", font=("Arial", 9, "bold")
        ).pack(side="left")

        self.selected_exchanges = {}
        for exchange in SUPPORTED_EXCHANGES:
            var = tk.BooleanVar()
            if exchange in self.config.SELECTED_EXCHANGES:
                var.set(True)
            self.selected_exchanges[exchange] = var

            cb = ttk.Checkbutton(
                exchange_select_frame, text=exchange.capitalize(), variable=var
            )
            cb.pack(side="left", padx=(10, 0))

        # 거래소별 API 키 설정
        notebook = ttk.Notebook(config_frame)
        notebook.pack(fill="both", expand=True, pady=(10, 0))

        for exchange in SUPPORTED_EXCHANGES:
            tab_frame = ttk.Frame(notebook)
            notebook.add(tab_frame, text=f"{exchange.capitalize()} API")

            # API 키
            api_key_frame = ttk.Frame(tab_frame)
            api_key_frame.pack(fill="x", pady=(10, 5))

            ttk.Label(api_key_frame, text="API Key:", width=12).pack(side="left")
            api_key_entry = ttk.Entry(
                api_key_frame,
                textvariable=self.exchange_vars[exchange]["api_key"],
                width=50,
                font=("Consolas", 9),
            )
            api_key_entry.pack(side="left", padx=(5, 0), fill="x", expand=True)

            # API 시크릿
            api_secret_frame = ttk.Frame(tab_frame)
            api_secret_frame.pack(fill="x", pady=(5, 10))

            ttk.Label(api_secret_frame, text="API Secret:", width=12).pack(side="left")
            api_secret_entry = ttk.Entry(
                api_secret_frame,
                textvariable=self.exchange_vars[exchange]["api_secret"],
                width=50,
                show="*",
                font=("Consolas", 9),
            )
            api_secret_entry.pack(side="left", padx=(5, 0), fill="x", expand=True)

            # 연결 테스트 버튼
            test_btn = ttk.Button(
                api_secret_frame,
                text="연결 테스트",
                command=lambda ex=exchange: self._test_exchange_connection(ex),
            )
            test_btn.pack(side="right", padx=(10, 0))

            # Passphrase
            passphrase_frame = ttk.Frame(tab_frame)
            passphrase_frame.pack(fill="x", pady=(5, 10))

            ttk.Label(passphrase_frame, text="Passphrase:", width=12).pack(side="left")
            passphrase_entry = ttk.Entry(
                passphrase_frame,
                textvariable=self.exchange_vars[exchange]["passphrase"],
                width=50,
                show="*",
                font=("Consolas", 9),
            )
            passphrase_entry.pack(side="left", padx=(5, 0), fill="x", expand=True)

            # Bitget만 Passphrase 필수
            if exchange == "bitget":
                hint_label = ttk.Label(
                    passphrase_frame, text="(필수)", foreground="red", font=("Arial", 8)
                )
                hint_label.pack(side="left", padx=(5, 0))
            else:
                hint_label = ttk.Label(
                    passphrase_frame,
                    text="(선택)",
                    foreground="gray",
                    font=("Arial", 8),
                )
                hint_label.pack(side="left", padx=(5, 0))

    def _create_trading_config_section(self, parent):
        """트레이딩 설정 섹션 - 보호장치 설정 추가"""
        trading_frame = ttk.LabelFrame(parent, text="트레이딩 설정", padding=10)
        trading_frame.pack(fill="x", pady=(0, 10))

        # 첫 번째 행: 기본 설정
        row1 = ttk.Frame(trading_frame)
        row1.pack(fill="x", pady=(0, 8))

        ttk.Label(row1, text="레버리지:").pack(side="left")
        self.leverage_var = tk.StringVar(value=str(self.config.LEVERAGE))
        leverage_entry = ttk.Entry(row1, textvariable=self.leverage_var, width=8)
        leverage_entry.pack(side="left", padx=(5, 20))

        ttk.Label(row1, text="증거금(%):").pack(side="left")
        self.quote_pct_var = tk.StringVar(value=str(self.config.QUOTE_PCT))
        quote_entry = ttk.Entry(row1, textvariable=self.quote_pct_var, width=8)
        quote_entry.pack(side="left", padx=(5, 0))

        # 두 번째 행: 리스크 관리
        row2 = ttk.Frame(trading_frame)
        row2.pack(fill="x", pady=(0, 8))

        ttk.Label(row2, text="고정손절(%):").pack(side="left")
        self.fixed_stop_pct_var = tk.StringVar(value=str(self.config.FIXED_STOP_PCT))
        fixed_stop_entry = ttk.Entry(
            row2, textvariable=self.fixed_stop_pct_var, width=8
        )
        fixed_stop_entry.pack(side="left", padx=(5, 20))

        ttk.Label(row2, text="트레일트리거(%):").pack(side="left")
        self.trail_trigger_pct_var = tk.StringVar(
            value=str(self.config.TRAIL_TRIGGER_PCT)
        )
        trail_trigger_entry = ttk.Entry(
            row2, textvariable=self.trail_trigger_pct_var, width=8
        )
        trail_trigger_entry.pack(side="left", padx=(5, 20))

        ttk.Label(row2, text="트레일콜백(%):").pack(side="left")
        self.trail_callback_pct_var = tk.StringVar(
            value=str(self.config.TRAIL_CALLBACK_PCT)
        )
        trail_callback_entry = ttk.Entry(
            row2, textvariable=self.trail_callback_pct_var, width=8
        )
        trail_callback_entry.pack(side="left", padx=(5, 0))

        # 세 번째 행: 폴백 설정
        row3 = ttk.Frame(trading_frame)
        row3.pack(fill="x", pady=(0, 8))

        ttk.Label(row3, text="최대폴백리스크(%):").pack(side="left")
        self.max_fallback_risk_var = tk.StringVar(
            value=str(self.config.MAX_FALLBACK_RISK_PCT)
        )
        fallback_entry = ttk.Entry(
            row3, textvariable=self.max_fallback_risk_var, width=8
        )
        fallback_entry.pack(side="left", padx=(5, 20))

        # 새로 추가: 보호장치 재시도 설정
        ttk.Label(row3, text="보호재시도횟수:").pack(side="left")
        self.max_protection_retries_var = tk.StringVar(
            value=str(self.config.MAX_PROTECTION_RETRIES)
        )
        protection_retries_entry = ttk.Entry(
            row3, textvariable=self.max_protection_retries_var, width=8
        )
        protection_retries_entry.pack(side="left", padx=(5, 20))

        ttk.Label(row3, text="재시도간격(초):").pack(side="left")
        self.protection_retry_interval_var = tk.StringVar(
            value=str(self.config.PROTECTION_RETRY_INTERVAL)
        )
        retry_interval_entry = ttk.Entry(
            row3, textvariable=self.protection_retry_interval_var, width=8
        )
        retry_interval_entry.pack(side="left", padx=(5, 0))

        # 네 번째 행: 옵션들
        row4 = ttk.Frame(trading_frame)
        row4.pack(fill="x")

        self.fixed_stop_var = tk.BooleanVar(value=self.config.USE_FIXED_STOP)
        fixed_stop_cb = ttk.Checkbutton(
            row4, text="고정 손절", variable=self.fixed_stop_var
        )
        fixed_stop_cb.pack(side="left", padx=(0, 15))

        self.trailing_var = tk.BooleanVar(value=self.config.USE_TRAILING_STOP)
        trailing_cb = ttk.Checkbutton(
            row4, text="트레일링 스톱", variable=self.trailing_var
        )
        trailing_cb.pack(side="left", padx=(0, 15))

        # 최소수량 진입 옵션
        self.min_qty_fallback_var = tk.BooleanVar(
            value=self.config.ENABLE_MIN_QTY_FALLBACK
        )
        min_qty_cb = ttk.Checkbutton(
            row4, text="최소수량 진입", variable=self.min_qty_fallback_var
        )
        min_qty_cb.pack(side="left", padx=(0, 15))

        # 새로 추가: 손절만 허용 옵션
        self.allow_sl_only_var = tk.BooleanVar(value=self.config.ALLOW_SL_ONLY)
        allow_sl_only_cb = ttk.Checkbutton(
            row4, text="손절만 허용", variable=self.allow_sl_only_var
        )
        allow_sl_only_cb.pack(side="left", padx=(0, 15))

        # 저장 버튼
        save_btn = ttk.Button(row4, text="설정 저장", command=self._save_config)
        save_btn.pack(side="right")

    def _create_status_section(self, parent):
        """상태 및 포지션 섹션"""
        status_frame = ttk.LabelFrame(parent, text="상태 및 포지션", padding=10)
        status_frame.pack(fill="both", expand=True, pady=(0, 10))

        # 상태 정보
        info_frame = ttk.Frame(status_frame)
        info_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(info_frame, text="서버 상태:").pack(side="left")
        status_label = ttk.Label(
            info_frame, textvariable=self.server_status, font=("Arial", 10, "bold")
        )
        status_label.pack(side="left", padx=(5, 20))

        ttk.Label(info_frame, text="Public URL:").pack(side="left")
        url_label = ttk.Label(
            info_frame, textvariable=self.public_url, foreground="blue", cursor="hand2"
        )
        url_label.pack(side="left", padx=(5, 10))
        url_label.bind("<Button-1>", self._open_public_url)

        copy_btn = ttk.Button(info_frame, text="복사", command=self._copy_url)
        copy_btn.pack(side="left", padx=(5, 0))

        # 포지션 테이블 컨테이너
        table_container = ttk.Frame(status_frame)
        table_container.pack(fill="both", expand=True)

        # 테이블과 스크롤바를 위한 프레임
        table_frame = ttk.Frame(table_container)
        table_frame.pack(side="left", fill="both", expand=True)

        # 포지션 테이블
        columns = ("거래소", "심볼", "방향", "수량", "진입가", "현재가", "PnL")
        self.position_tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", height=6
        )

        for col in columns:
            self.position_tree.heading(col, text=col)
            width = 80 if col in ["거래소", "방향"] else 100
            self.position_tree.column(col, width=width, anchor="center")

        # 수직 스크롤바
        v_scrollbar = ttk.Scrollbar(
            table_frame, orient="vertical", command=self.position_tree.yview
        )
        self.position_tree.configure(yscrollcommand=v_scrollbar.set)

        # 수평 스크롤바
        h_scrollbar = ttk.Scrollbar(
            table_frame, orient="horizontal", command=self.position_tree.xview
        )
        self.position_tree.configure(xscrollcommand=h_scrollbar.set)

        # 테이블과 스크롤바 배치
        self.position_tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        # grid 가중치 설정
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        # 버튼 영역 - 세로 배치
        btn_frame = ttk.Frame(table_container)
        btn_frame.pack(side="right", fill="y", padx=(10, 0))

        refresh_btn = ttk.Button(
            btn_frame, text="새로고침", command=self._refresh_positions, width=12
        )
        refresh_btn.pack(fill="x", pady=(0, 5))

        close_all_btn = ttk.Button(
            btn_frame,
            text="모든 포지션\n청산",
            command=self._close_all_positions,
            width=12,
        )
        close_all_btn.pack(fill="x", pady=(0, 5))

        status_btn = ttk.Button(
            btn_frame, text="상태 페이지", command=self._open_status, width=12
        )
        status_btn.pack(fill="x")

    def _create_control_section(self, parent):
        """제어 섹션"""
        control_frame = ttk.LabelFrame(parent, text="서버 제어", padding=10)
        control_frame.pack(fill="x")

        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill="x")

        self.start_btn = ttk.Button(
            btn_frame, text="서버 시작", command=self._start_server
        )
        self.start_btn.pack(side="left", padx=(0, 5))

        self.stop_btn = ttk.Button(
            btn_frame, text="서버 중지", command=self._stop_server, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=(0, 5))

        sync_btn = ttk.Button(
            btn_frame, text="포지션 동기화", command=self._sync_positions
        )
        sync_btn.pack(side="left", padx=(0, 5))

        risk_btn = ttk.Button(btn_frame, text="리스크 재적용", command=self._apply_risk)
        risk_btn.pack(side="left")

    def _test_exchange_connection(self, exchange_name: str):
        """특정 거래소 연결 테스트"""
        try:
            api_key = self.exchange_vars[exchange_name]["api_key"].get().strip()
            api_secret = self.exchange_vars[exchange_name]["api_secret"].get().strip()
            passphrase = self.exchange_vars[exchange_name]["passphrase"].get().strip()

            if not api_key or not api_secret:
                messagebox.showerror(
                    "오류", f"{exchange_name} API 키와 시크릿을 입력하세요!"
                )
                return

            if exchange_name.lower() == "bitget" and not passphrase:
                messagebox.showerror(
                    "오류", f"{exchange_name}은 Passphrase가 필수입니다!"
                )
                return

            client = create_exchange_client(
                exchange_name, api_key, api_secret, passphrase, False
            )

            if not client:
                messagebox.showerror("실패", f"{exchange_name} 클라이언트 생성 실패!")
                return

            ok, msg = client.validate_credentials()

            if ok:
                messagebox.showinfo("성공", f"{exchange_name} 연결 성공!\n{msg}")
            else:
                messagebox.showerror("실패", f"{exchange_name} 연결 실패!\n{msg}")
        except Exception as e:
            self.logger.error(f"{exchange_name} 연결 테스트 오류: {e}")
            messagebox.showerror("오류", f"{exchange_name} 연결 테스트 오류: {e}")

    def _save_config(self):
        """설정 저장 - 새로운 보호장치 설정 포함"""
        try:
            # 선택된 거래소 업데이트
            selected_exchanges = []
            for exchange, var in self.selected_exchanges.items():
                if var.get():
                    selected_exchanges.append(exchange)

            if not selected_exchanges:
                messagebox.showerror("오류", "최소 하나의 거래소를 선택해야 합니다!")
                return

            self.config.SELECTED_EXCHANGES = selected_exchanges

            # 기본 설정 저장
            self.config.LEVERAGE = float(self.leverage_var.get())
            self.config.QUOTE_PCT = float(self.quote_pct_var.get())
            self.config.USE_FIXED_STOP = self.fixed_stop_var.get()
            self.config.USE_TRAILING_STOP = self.trailing_var.get()

            # 리스크 값 저장
            self.config.FIXED_STOP_PCT = float(self.fixed_stop_pct_var.get())
            self.config.TRAIL_TRIGGER_PCT = float(self.trail_trigger_pct_var.get())
            self.config.TRAIL_CALLBACK_PCT = float(self.trail_callback_pct_var.get())

            # 기존 설정 저장
            self.config.ENABLE_MIN_QTY_FALLBACK = self.min_qty_fallback_var.get()
            self.config.MAX_FALLBACK_RISK_PCT = float(self.max_fallback_risk_var.get())

            # 새로운 보호장치 설정 저장
            self.config.ALLOW_SL_ONLY = self.allow_sl_only_var.get()
            self.config.MAX_PROTECTION_RETRIES = int(
                self.max_protection_retries_var.get()
            )
            self.config.PROTECTION_RETRY_INTERVAL = float(
                self.protection_retry_interval_var.get()
            )

            # 거래소별 API 키 저장
            for exchange, vars_dict in self.exchange_vars.items():
                api_key = vars_dict["api_key"].get().strip()
                api_secret = vars_dict["api_secret"].get().strip()
                passphrase = vars_dict["passphrase"].get().strip()

                config_manager.set_exchange_keys(
                    exchange, api_key, api_secret, passphrase
                )

            if save_config():
                messagebox.showinfo("성공", "설정이 저장되었습니다!")
            else:
                messagebox.showerror("오류", "설정 저장에 실패했습니다.")
        except Exception as e:
            messagebox.showerror("오류", f"설정 저장 오류: {e}")

    def _start_log_monitor(self):
        """로그 모니터링 시작"""

        def check_logs():
            try:
                while True:
                    try:
                        message = self.log_queue.get_nowait()
                        self._handle_log_message(message)
                    except queue.Empty:
                        break
            except Exception as e:
                self.logger.error(f"로그 모니터 오류: {e}")
            self.root.after(100, check_logs)

        check_logs()

    def _handle_log_message(self, message):
        """로그 메시지 처리"""
        try:
            if isinstance(message, str):
                if message.startswith("PUBLIC_URL:"):
                    url = message[11:]
                    self.public_url.set(url)
                elif message.startswith("ERROR:"):
                    error = message[6:]
                    pass  # 에러 로깅은 콘솔에만
                elif message == "SERVER_STARTED":
                    self.server_status.set("서버 실행중")
                    self.start_btn.config(state="disabled")
                    self.stop_btn.config(state="normal")
        except Exception as e:
            pass  # 로그 처리 오류 무시

    def _start_server(self):
        """서버 시작"""
        try:
            if self.server_process and self.server_process.is_alive():
                messagebox.showwarning("경고", "서버가 이미 실행 중입니다!")
                return

            # 설정 저장
            self._save_config()

            # 포트 사용 여부 확인
            if self._is_port_in_use(self.config.SERVER.port):
                messagebox.showerror(
                    "오류", f"포트 {self.config.SERVER.port}가 이미 사용 중입니다!"
                )
                return

            self.stop_event.clear()
            self.server_process = mp.Process(
                target=server_main,
                args=(self.config.to_dict(), self.stop_event, self.log_queue),
                daemon=True,
            )
            self.server_process.start()
            self.logger.info("Enhanced Risk Protection 서버 프로세스 시작됨")
        except Exception as e:
            self.logger.error(f"서버 시작 실패: {e}")
            messagebox.showerror("오류", f"서버 시작 실패: {e}")

    def _stop_server(self):
        """서버 중지"""
        try:
            if self.server_process and self.server_process.is_alive():
                self.stop_event.set()
                self.server_process.terminate()
                self.server_process.join(timeout=5)

                if self.server_process.is_alive():
                    self.server_process.kill()

                self.server_process = None
                self.server_status.set("서버 중지됨")
                self.start_btn.config(state="normal")
                self.stop_btn.config(state="disabled")
                self.public_url.set("Not detected")
            else:
                messagebox.showinfo("정보", "서버가 실행 중이 아닙니다!")
        except Exception as e:
            messagebox.showerror("오류", f"서버 중지 실패: {e}")

    def _sync_positions(self):
        """포지션 동기화 요청"""
        try:
            self._make_request("POST", "/sync")
        except Exception as e:
            self.logger.error(f"동기화 요청 실패: {e}")

    def _apply_risk(self):
        """리스크 재적용 요청"""
        try:
            self._make_request("POST", "/apply_risk")
        except Exception as e:
            self.logger.error(f"리스크 적용 요청 실패: {e}")

    def _close_all_positions(self):
        """모든 포지션 청산 요청"""
        try:
            if messagebox.askyesno("확인", "모든 포지션을 청산하시겠습니까?"):
                self._make_request("POST", "/close_all")
        except Exception as e:
            self.logger.error(f"포지션 청산 요청 실패: {e}")

    def _make_request(self, method: str, endpoint: str, data: dict = None):
        """HTTP 요청 통합 함수"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                url = f"http://127.0.0.1:{self.config.SERVER.port}{endpoint}"

                if method == "GET":
                    response = requests.get(url, timeout=10)
                else:
                    response = requests.post(url, json=data, timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        if "synced_positions" in result:
                            count = result.get("synced_positions", 0)
                            self._refresh_positions()
                        elif "applied_positions" in result:
                            count = result.get("applied_positions", 0)
                        elif "closed_positions" in result:
                            count = result.get("closed_positions", 0)
                            self._refresh_positions()

                        # 경고 메시지 처리
                        if result.get("warnings"):
                            for warning in result["warnings"]:
                                pass  # 경고는 콘솔에만 로깅
                    return
                else:
                    if attempt == max_retries - 1:
                        raise Exception(f"HTTP {response.status_code}")

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 1.0 * (2**attempt)
                    time.sleep(wait_time)
                else:
                    pass  # 오류는 콘솔에만 로깅

    def _refresh_positions(self):
        """포지션 테이블 새로고침"""
        try:
            url = f"http://127.0.0.1:{self.config.SERVER.port}/status"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                positions = data.get("positions", {})

                # 기존 항목 삭제
                for item in self.position_tree.get_children():
                    self.position_tree.delete(item)

                # 새 포지션 추가
                for exchange, symbols in positions.items():
                    for symbol, pos in symbols.items():
                        side = pos.get("side", "")
                        qty = pos.get("qty", 0)
                        entry = pos.get("entry_price", 0)
                        mark = pos.get("mark_price", 0)
                        pnl = pos.get("uPNL", 0)

                        tag = "profit" if pnl >= 0 else "loss"

                        self.position_tree.insert(
                            "",
                            "end",
                            values=(
                                exchange,
                                symbol,
                                side,
                                f"{qty:.6f}",
                                f"{entry:.4f}",
                                f"{mark:.4f}",
                                f"{pnl:+.2f}",
                            ),
                            tags=(tag,),
                        )

                # 색상 설정
                self.position_tree.tag_configure("profit", foreground="green")
                self.position_tree.tag_configure("loss", foreground="red")

            else:
                messagebox.showerror(
                    "오류", f"상태 조회 실패: HTTP {response.status_code}"
                )
        except Exception as e:
            messagebox.showerror("오류", f"포지션 새로고침 실패: {e}")

    def _open_status(self):
        """상태 페이지 열기"""
        try:
            public_url = self.public_url.get()
            if public_url and public_url not in ("Not detected", "Detecting..."):
                url = f"{public_url}/status"
            else:
                url = f"http://127.0.0.1:{self.config.SERVER.port}/status"

            webbrowser.open(url)
        except Exception as e:
            self.logger.error(f"상태 페이지 열기 실패: {e}")
            messagebox.showerror("오류", f"상태 페이지 열기 실패: {e}")

    def _open_public_url(self, event=None):
        """Public URL 열기"""
        try:
            url = self.public_url.get()
            if url and url not in ("Not detected", "Detecting..."):
                webbrowser.open(url)
        except Exception as e:
            self.logger.error(f"URL 열기 실패: {e}")

    def _copy_url(self):
        """URL 클립보드 복사"""
        try:
            url = self.public_url.get()
            if url and url not in ("Not detected", "Detecting..."):
                self.root.clipboard_clear()
                self.root.clipboard_append(url)
                self.root.update()
            else:
                messagebox.showinfo("정보", "복사할 URL이 없습니다")
        except Exception as e:
            self.logger.error(f"URL 복사 실패: {e}")

    def _is_port_in_use(self, port: int) -> bool:
        """포트 사용 여부 확인"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(("127.0.0.1", port))
                return result == 0
        except Exception:
            return False

    def _on_closing(self):
        """GUI 종료 처리"""
        try:
            self.logger.info("Enhanced Risk Protection GUI 종료 시작")
            if self.server_process and self.server_process.is_alive():
                self.stop_event.set()
                self.server_process.terminate()
                self.server_process.join(timeout=3)
                if self.server_process.is_alive():
                    self.server_process.kill()
        except Exception as e:
            self.logger.error(f"종료 시 정리 오류: {e}")
        finally:
            self.root.destroy()

    def run(self):
        """GUI 실행"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()

    def _cleanup(self):
        """정리 작업"""
        try:
            if self.server_process and self.server_process.is_alive():
                self.stop_event.set()
                self.server_process.terminate()
                self.server_process.join(timeout=3)
                if self.server_process.is_alive():
                    self.server_process.kill()
        except Exception as e:
            logging.error(f"정리 작업 오류: {e}")


# ============================================================================
# 메인 진입점
# ============================================================================



def main():
    """메인 진입점"""
    mp.set_start_method("spawn", force=True)

    try:
        # 기본 로깅 설정
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
        logger = logging.getLogger("stargate.main")

        # ---- CLI ----
        parser = argparse.ArgumentParser(
            prog="stargate",
            description="Stargate Multi-Exchange Trading Server v4.2.1"
        )
        parser.add_argument("--server-only", action="store_true",
                            help="GUI 없이 Flask 서버만 실행")
        parser.add_argument("--host", default=None, help="서버 바인딩 호스트 (기본: 설정값)")
        parser.add_argument("--port", type=int, default=None, help="서버 포트 (기본: 설정값)")
        parser.add_argument("--exchanges", default=None,
                            help="쉼표로 구분된 거래소 목록 예) bybit,bitget")
        args = parser.parse_args()

        logger.info("Stargate Multi-Exchange Trading Server v4.2.1")
        logger.info("Enhanced Risk Protection System")
        logger.info("=" * 60)

        # 설정 로드
        cfg = get_config()
        cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else asdict(cfg)

        # CLI 덮어쓰기
        if args.exchanges:
            cfg_dict["SELECTED_EXCHANGES"] = [x.strip().lower() for x in args.exchanges.split(",") if x.strip()]
        if args.host:
            cfg_dict.setdefault("SERVER", {}).update({"host": args.host})
        if args.port:
            cfg_dict.setdefault("SERVER", {}).update({"port": args.port})

        # 서버 전용 모드 또는 헤드리스 환경
        if args.server_only or not GUI_AVAILABLE:
            stop_event = mp.Event()
            log_queue = mp.Queue()
            server_main(cfg_dict, stop_event, log_queue)
            return

        # GUI 실행 (로컬 환경)
        gui = StargateMultiExchangeGUI()
        gui.run()

    except KeyboardInterrupt:
        logging.info("사용자 중단")
    except Exception as e:
        logging.error(f"치명적 오류: {e}", exc_info=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        logging.info("애플리케이션 종료")



if __name__ == "__main__":
    main()
