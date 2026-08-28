"""Symbol normalization and market-data error types for vendor calls.

Yahoo Finance (the default vendor) uses specific ticker conventions that
differ from the broker / TradingView / MT5 style symbols users often type:

    user types        Yahoo wants       why
    ---------------   ---------------   -----------------------------------
    XAUUSD, XAUUSD+   GC=F              gold has no forex pair on Yahoo;
                                        it is quoted as a COMEX future
    EURUSD            EURUSD=X          spot forex pairs take a ``=X`` suffix
    BTCUSD            BTC-USD           crypto pairs use a ``-`` separator
    SPX500, US500     ^GSPC             index CFDs map to Yahoo index symbols

Passing the raw broker symbol to Yahoo returns an empty result, which the
agents previously received as free text and could hallucinate a price
around (see issue #781). Centralizing the mapping here means every yfinance
entry point resolves symbols the same way, and new instruments are added by
appending a table row rather than editing call sites.
"""

from __future__ import annotations

import logging
import re

# NoMarketDataError lives in the vendor-error taxonomy (errors.py); re-exported
# here for the many call sites that import it alongside normalize_symbol.
from .errors import NoMarketDataError as NoMarketDataError

logger = logging.getLogger(__name__)


# ISO-4217 codes common enough to appear in retail forex pairs. A bare
# six-letter symbol whose halves are BOTH in this set is treated as a spot
# forex pair and given Yahoo's ``=X`` suffix.
_FOREX_CURRENCIES = frozenset(
    {
        "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
        "CNY", "CNH", "HKD", "SGD", "SEK", "NOK", "DKK", "PLN",
        "MXN", "ZAR", "TRY", "INR", "KRW", "BRL", "RUB", "THB",
    }
)

# Crypto bases that brokers quote against USD without a separator.
_CRYPTO_BASES = frozenset(
    {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "LTC", "BCH", "DOT", "AVAX", "LINK"}
)

# Explicit aliases for instruments whose broker symbol does not map to a
# Yahoo symbol by rule. Metals/energy resolve to their front-month future;
# index CFD names resolve to the underlying Yahoo index symbol. Extend by
# adding rows — no call site changes required.
_ALIASES = {
    # Precious metals (spot names -> COMEX/NYMEX futures)
    "XAUUSD": "GC=F", "XAU": "GC=F", "GOLD": "GC=F",
    "XAGUSD": "SI=F", "XAG": "SI=F", "SILVER": "SI=F",
    "XPTUSD": "PL=F", "XPDUSD": "PA=F",
    # Energy
    "WTICOUSD": "CL=F", "USOIL": "CL=F", "WTI": "CL=F",
    "BCOUSD": "BZ=F", "UKOIL": "BZ=F", "BRENT": "BZ=F",
    "NATGAS": "NG=F", "XNGUSD": "NG=F",
    "COPPER": "HG=F", "XCUUSD": "HG=F",
    # Index CFDs -> Yahoo index symbols
    "SPX500": "^GSPC", "US500": "^GSPC", "SPX": "^GSPC",
    "NAS100": "^NDX", "US100": "^NDX", "USTEC": "^NDX",
    "US30": "^DJI", "DJI30": "^DJI", "WS30": "^DJI",
    "GER40": "^GDAXI", "GER30": "^GDAXI", "DE40": "^GDAXI",
    "UK100": "^FTSE", "JP225": "^N225", "JPN225": "^N225",
    "FRA40": "^FCHI", "EU50": "^STOXX50E", "HK50": "^HSI",
}

# Yahoo symbols may contain letters, digits, and these structural characters.
_YAHOO_SAFE = re.compile(r"^[A-Za-z0-9._\-\^=]+$")


# Crypto quote currencies that all map to Yahoo's USD pair. Yahoo lists only
# ``<BASE>-USD`` (not the USDT/USDC stablecoin pairs), so a broker symbol quoted
# in any of these resolves to ``-USD`` (#982). Longest first so ``USDT``/``USDC``
# match before the ``USD`` substring.
_CRYPTO_QUOTES = ("USDT", "USDC", "USD")

# China A-share auto-suffix ranges (SSE/SZSE cash equities). Bare six-digit
# codes are suffixed for the whole pipeline so users can type either form;
# B-shares and BSE codes have no reliable Yahoo coverage and are left alone.
_ASHARE_SSE_PREFIXES = ("600", "601", "603", "605", "688", "689")
_ASHARE_SZSE_PREFIXES = ("000", "001", "002", "003", "300", "301")
_ASHARE_SUFFIX_MAP = {"SS": ".SS", "SH": ".SS", "SZ": ".SZ"}  # .SH is an alias of .SS

# China in-market exchange-traded funds (ETF/LOF). Bare six-digit codes with
# these prefixes resolve to Yahoo-style suffixed symbols as well; the akshare
# vendor routes them to the fund history endpoint instead of stock history.
_FUND_SSE_PREFIXES = ("51", "56", "58")
_FUND_SZSE_PREFIXES = ("15",)
_FUND_PREFIXES = _FUND_SSE_PREFIXES + _FUND_SZSE_PREFIXES


def is_fund_symbol(s: str) -> bool:
    """True when ``s`` (bare or suffixed six-digit A-share code) is an ETF/LOF."""
    s = s.strip().upper().rstrip("+")
    body = s.split(".", 1)[0]
    return len(body) == 6 and body.isdigit() and (
        body.startswith(_FUND_SSE_PREFIXES) or body.startswith(_FUND_SZSE_PREFIXES)
    )


def crypto_base(raw: str) -> str | None:
    """Return the crypto base (e.g. ``BTC``) for a known USD/USDT/USDC-quoted
    crypto symbol in any form the pipeline may hold — ``BTC-USD``, ``BTCUSD``,
    ``BTC-USDT`` — or None for non-crypto symbols. Purely syntactic.
    """
    if not isinstance(raw, str):
        return None
    compact = raw.strip().upper().rstrip("+").replace("-", "")
    for quote in _CRYPTO_QUOTES:
        if compact.endswith(quote):
            base = compact[: -len(quote)]
            return base if base in _CRYPTO_BASES else None
    return None


def _normalize_crypto(s: str) -> str | None:
    """Return ``<BASE>-USD`` for a known USD/USDT/USDC-quoted crypto, else None."""
    base = crypto_base(s)
    return f"{base}-USD" if base else None


def _normalize_ashare(s: str) -> str | None:
    """Return the canonical suffixed form for an A-share code, else None.

    ``600519`` -> ``600519.SS``, ``000001`` -> ``000001.SZ``,
    ``600519.SH`` -> ``600519.SS`` (east-money spelling of the SSE suffix).
    In-market funds resolve too: ``510300`` -> ``510300.SS``,
    ``159994`` -> ``159994.SZ``. Already-suffixed ``.SS``/``.SZ`` pass through
    upper-cased. Bare six-digit codes outside the auto-supported ranges return
    None unchanged.
    """
    body, dot, suffix = s.partition(".")
    if len(body) == 6 and body.isdigit() and (dot == "" or suffix in _ASHARE_SUFFIX_MAP):
        if dot:
            canonical = body + _ASHARE_SUFFIX_MAP[suffix]
        elif body.startswith(_FUND_PREFIXES):
            # ETFs/LOFs before the stock ranges: SSE 51*/56*/58*, SZSE 15*.
            canonical = f"{body}.SS" if body.startswith(_FUND_SSE_PREFIXES) else f"{body}.SZ"
        elif body.startswith(_ASHARE_SSE_PREFIXES):
            canonical = f"{body}.SS"
        elif body.startswith(_ASHARE_SZSE_PREFIXES):
            canonical = f"{body}.SZ"
        else:
            # B-shares / BSE: no auto-suffix — leave for explicit handling.
            return None
        return canonical if canonical != s else None
    return None


def normalize_symbol(raw: str) -> str:
    """Map a user/broker symbol to its canonical Yahoo Finance symbol.

    Resolution order (first match wins):
      1. Explicit alias table (metals, energy, index CFDs).
      2. Crypto rule: a known crypto base quoted in USD/USDT/USDC (dashed or
         not) -> ``BASE-USD``.
      3. Forex rule: six letters that are two ISO currency codes -> ``PAIR=X``.
      4. A-share rule: bare or suffixed SSE/SZSE codes -> Yahoo's exchange
         suffix (``600519``/``600519.SH`` -> ``600519.SS``,
         ``000001`` -> ``000001.SZ``).
      5. Otherwise the upper-cased symbol is returned unchanged (plain
         equities, ETFs, Yahoo-native symbols like ``GC=F`` or ``^GSPC``).

    A trailing ``+`` (broker CFD marker, e.g. ``XAUUSD+``) is stripped before
    matching. The function is purely syntactic — it performs no network
    calls — so it is safe to apply on every request.
    """
    if not isinstance(raw, str) or not raw.strip():
        return raw

    s = raw.strip().upper()
    # Broker CFD/qualifier suffixes Yahoo never uses.
    s = s.rstrip("+")

    crypto = _normalize_crypto(s)
    ashare = _normalize_ashare(s)
    if s in _ALIASES:
        canonical = _ALIASES[s]
    elif crypto is not None:
        canonical = crypto
    elif len(s) == 6 and s[:3] in _FOREX_CURRENCIES and s[3:] in _FOREX_CURRENCIES:
        canonical = f"{s}=X"
    elif ashare is not None:
        canonical = ashare
    else:
        canonical = s

    if canonical != raw.strip().upper():
        logger.info("Resolved symbol %r to Yahoo symbol %r", raw, canonical)
    return canonical


def is_yahoo_safe(symbol: str) -> bool:
    """True when ``symbol`` only contains characters Yahoo symbols use."""
    return bool(symbol) and _YAHOO_SAFE.fullmatch(symbol) is not None
