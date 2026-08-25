from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import io
import math
import re
import hashlib
from datetime import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =============================================================================
# PAGE
# =============================================================================
st.set_page_config(page_title="G. Balance Stock screener", page_icon="🎯", layout="wide")
st.title("🎯 G. Balance Stock screener")

LOOKBACK = 500
LAST_TOUCH_BARS = 5
DEFAULT_MIN_TOUCHES = 2


# =============================================================================
# MODEL — stesso data model Balance usato nel motore COT Smart Money
# =============================================================================
@dataclass(frozen=True)
class BalanceZone:
    center: float
    half: float
    pct_range: float
    strength: float
    hits: int
    support_hits: int
    resistance_hits: int
    dwell: int
    last_hit_age: int
    independent_tests: int
    independent_successes: int
    independent_support_success: int
    independent_resistance_success: int
    independent_breaks: int
    reliability: float


# =============================================================================
# TICKERS
# =============================================================================
def normalize_stock_ticker(ticker: str, market: str) -> str:
    t = ticker.strip().upper()
    if not t:
        return t
    if market == "Italia":
        if "." not in t and "=" not in t and ":" not in t:
            t = f"{t}.MI"
    elif market == "USA":
        if re.fullmatch(r"[A-Z0-9]+\.[A-Z]", t):
            t = t.replace(".", "-")
    return t


def infer_market_from_text(text: str, source_name: str = "") -> str:
    """Riconosce Italia/USA dai file ticker senza modificare la logica Balance."""
    name = (source_name or "").upper()
    if any(tag in name for tag in ("AZIONI_ITA", "ITALIA", "ITALY")):
        return "Italia"
    if any(tag in name for tag in ("STOCK USA", "AZIONI_USA", "_USA", " US ")):
        return "USA"

    raw_tokens = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        if ";" in line or "\t" in line:
            parts = [p.strip() for p in re.split(r"[;\t]", line) if p.strip()]
            if parts:
                raw_tokens.append(parts[-1].upper())
        else:
            raw_tokens.extend(t.strip().upper() for t in line.split(",") if t.strip())

    if not raw_tokens:
        return "USA"
    mi_count = sum(t.endswith(".MI") or t.startswith("MIL:") for t in raw_tokens)
    explicit_other_suffix = sum(bool(re.search(r"\.[A-Z]{1,4}$", t)) and not t.endswith(".MI") for t in raw_tokens)
    if mi_count >= max(1, len(raw_tokens) // 2):
        return "Italia"
    if mi_count and explicit_other_suffix:
        return "Misto / ticker Yahoo completi"
    return "USA"


def parse_tickers(text: str, market: str) -> list[tuple[str, str]]:
    """
    Formati accettati:
    - lista separata da virgole, anche su una sola riga: ENI.MI, UCG.MI, ISP.MI
    - un ticker per riga
    - Nome;Ticker (oppure Nome<TAB>Ticker)

    La virgola viene sempre interpretata come separatore di una LISTA di ticker,
    non come coppia Nome,Ticker. Questo mantiene compatibilita con i file usati
    dagli altri screener Python dell'utente.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    ignored_headers = {"TICKER", "TICKERS", "SYMBOL", "SYMBOLS", "SIMBOLO", "SIMBOLI"}

    def add_item(label: str, raw_ticker: str) -> None:
        raw_ticker = raw_ticker.strip().strip('"').strip("'")
        label_clean = label.strip().strip('"').strip("'")
        if not raw_ticker or raw_ticker.upper() in ignored_headers:
            return
        ticker = normalize_stock_ticker(raw_ticker, market)
        if ticker and ticker not in seen:
            out.append((label_clean or raw_ticker, ticker))
            seen.add(ticker)

    for raw in text.splitlines():
        line = raw.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue

        # Formato descrittivo: Nome;Ticker oppure Nome<TAB>Ticker.
        # Il punto e virgola NON viene usato per dividere una lista di ticker.
        if ";" in line or "\t" in line:
            parts = [p.strip() for p in re.split(r"[;\t]", line) if p.strip()]
            if len(parts) >= 2:
                add_item(parts[0], parts[-1])
            elif parts:
                add_item(parts[0], parts[0])
            continue

        # Formato lista degli screener esistenti: ticker separati da virgole.
        if "," in line:
            for token in line.split(","):
                token = token.strip()
                if token:
                    add_item(token, token)
            continue

        # Un ticker per riga.
        add_item(line, line)

    return out


def italy_example() -> str:
    return """# Nome;Ticker
A2A;A2A
Amplifon;AMP
Azimut;AZM
Banco BPM;BAMI
Brunello Cucinelli;BC
Banca Generali;BGN
MPS;BMPS
BPER Banca;BPE
Buzzi;BZU
Campari;CPR
DiaSorin;DIA
Enel;ENEL
Eni;ENI
ERG;ERG
Fineco;FBK
Generali;G
Hera;HER
Inwit;INW
Intesa Sanpaolo;ISP
Iveco;IVG
Leonardo;LDO
Mediobanca;MB
Moncler;MONC
Nexi;NEXI
Pirelli;PIRC
Prysmian;PRY
Poste Italiane;PST
Ferrari;RACE
Recordati;REC
Saipem;SPM
Snam;SRG
Stellantis;STLAM
STM;STMMI
Tenaris;TEN
Telecom Italia;TIT
Terna;TRN
UniCredit;UCG
Unipol;UNI
"""


def usa_example() -> str:
    return """# Nome;Ticker
Apple;AAPL
Microsoft;MSFT
Nvidia;NVDA
Amazon;AMZN
Alphabet;GOOGL
Meta;META
Tesla;TSLA
Broadcom;AVGO
Berkshire Hathaway;BRK-B
JPMorgan;JPM
Visa;V
Mastercard;MA
Eli Lilly;LLY
UnitedHealth;UNH
Exxon Mobil;XOM
Costco;COST
Walmart;WMT
Netflix;NFLX
AMD;AMD
Salesforce;CRM
"""


# =============================================================================
# DATA
# =============================================================================
def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    x = df.copy()
    x.columns = [str(c).lower().replace(" ", "_") for c in x.columns]
    required = ["open", "high", "low", "close"]
    if any(c not in x.columns for c in required):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if "volume" not in x.columns:
        x["volume"] = np.nan
    x = x[["open", "high", "low", "close", "volume"]].copy()
    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=required)
    x = x[~x.index.duplicated(keep="last")].sort_index()
    return x


def _extract_from_batch(raw: pd.DataFrame, ticker: str, single: bool) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if single and not isinstance(raw.columns, pd.MultiIndex):
        return _normalize_ohlc(raw)
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = set(map(str, raw.columns.get_level_values(0)))
        lvl1 = set(map(str, raw.columns.get_level_values(1)))
        try:
            if ticker in lvl0:
                return _normalize_ohlc(raw[ticker])
            if ticker in lvl1:
                return _normalize_ohlc(raw.xs(ticker, axis=1, level=1))
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def download_daily_chunk(tickers: tuple[str, ...], adjusted: bool) -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}
    raw = yf.download(
        tickers=list(tickers),
        period="5y",
        interval="1d",
        auto_adjust=bool(adjusted),
        actions=False,
        group_by="ticker",
        progress=False,
        threads=8,
        timeout=15,
    )
    single = len(tickers) == 1
    return {t: _extract_from_batch(raw, t, single) for t in tickers}


def load_universe_data(
    tickers: list[str], adjusted: bool, progress: Any | None = None
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    data_map: dict[str, pd.DataFrame] = {}
    notes: dict[str, str] = {}
    # Blocchi più ampi: meno overhead Streamlit/yfinance. yfinance limita comunque
    # il parallelismo interno tramite threads.
    chunk_size = 80
    total_chunks = max(1, math.ceil(len(tickers) / chunk_size))
    for chunk_no, start in enumerate(range(0, len(tickers), chunk_size), start=1):
        chunk = tuple(tickers[start:start + chunk_size])
        if progress is not None:
            frac = 0.05 + 0.30 * ((chunk_no - 1) / total_chunks)
            progress.progress(min(frac, 0.34), text=f"Download Yahoo {chunk_no}/{total_chunks} · {len(chunk)} ticker")
        try:
            result = download_daily_chunk(chunk, adjusted)
        except Exception as exc:
            for t in chunk:
                notes[t] = f"Download batch: {type(exc).__name__}: {exc}"
            continue
        for t in chunk:
            df = result.get(t, pd.DataFrame())
            if df.empty:
                notes[t] = "Nessun dato Yahoo Finance."
                continue
            data_map[t] = df
            notes[t] = "Dati Daily acquisiti."
    if progress is not None:
        progress.progress(0.35, text=f"Download completato · {len(data_map)}/{len(tickers)} ticker con dati")
    return data_map, notes

def _market_clock_for_ticker(ticker: str, market: str) -> tuple[str, time]:
    """Restituisce timezone e chiusura regolare per decidere se la Daily odierna è ancora aperta."""
    if market == "Italia" or (market == "Misto / ticker Yahoo completi" and ticker.upper().endswith(".MI")):
        return "Europe/Rome", time(17, 30)
    return "America/New_York", time(16, 0)


def only_closed_daily(data: pd.DataFrame, ticker: str, market: str) -> tuple[pd.DataFrame, bool]:
    """Esclude soltanto la Daily odierna se la seduta regolare non è ancora conclusa."""
    if data.empty:
        return data.copy(), False
    out = data.copy()
    tz_name, regular_close = _market_clock_for_ticker(ticker, market)
    now_local = pd.Timestamp.now(tz=ZoneInfo(tz_name))
    last_date = pd.Timestamp(out.index[-1]).date()
    if last_date == now_local.date() and now_local.time().replace(tzinfo=None) < regular_close:
        out = out.iloc[:-1].copy()
        return out, True
    return out, False


# =============================================================================
# BALANCE ENGINE — stesse definizioni del motore Python COT Smart Money
# lookback richiesto: 500
# =============================================================================
def wilder_atr(daily: pd.DataFrame, length: int = 14) -> pd.Series:
    if daily.empty:
        return pd.Series(dtype=float)
    high = daily["high"].astype(float)
    low = daily["low"].astype(float)
    close = daily["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = pd.Series(index=tr.index, dtype=float)
    if len(tr) < length:
        return atr
    atr.iloc[length - 1] = float(tr.iloc[:length].mean())
    for i in range(length, len(tr)):
        atr.iloc[i] = (float(atr.iloc[i - 1]) * (length - 1) + float(tr.iloc[i])) / length
    return atr


def _future_close_extremes(close: np.ndarray, validation_bars: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(close)
    fmin = np.full(n, np.nan, dtype=float)
    fmax = np.full(n, np.nan, dtype=float)
    for i in range(n):
        a = i + 1
        b = min(n, i + validation_bars + 1)
        if a < b:
            vals = close[a:b]
            fmin[i] = float(np.min(vals))
            fmax[i] = float(np.max(vals))
    return fmin, fmax


def _balance_eval_compatible(
    daily: pd.DataFrame, center: float, half: float, scan_limit: int, validation_bars: int = 10,
    future_min: np.ndarray | None = None, future_max: np.ndarray | None = None,
) -> tuple[int, int, int, int, int, float]:
    # Stesse definizioni del motore originale. Le finestre future vengono
    # precalcolate una sola volta per simbolo per velocizzare universi ampi.
    lows = daily["low"].to_numpy(float)
    highs = daily["high"].to_numpy(float)
    closes = daily["close"].to_numpy(float)
    if future_min is None or future_max is None:
        future_min, future_max = _future_close_extremes(closes, validation_bars)

    current = len(closes) - 1
    first_offset = validation_bars + 1
    last_offset = min(scan_limit - 1, current)
    if last_offset < first_offset:
        return 0, 0, 0, 0, 99999, 0.0

    start_pos = current - last_offset
    end_pos = current - first_offset
    pos = np.arange(start_pos, end_pos + 1)
    top, bottom = center + half, center - half
    overlap = (lows[pos] <= top) & (highs[pos] >= bottom)
    if not np.any(overlap):
        return 0, 0, 0, 0, 99999, 0.0

    pp = pos[overlap]
    is_support = closes[pp] >= center
    broken_support = future_min[pp] < bottom
    broken_resistance = future_max[pp] > top
    valid_support = is_support & (~broken_support)
    valid_resistance = (~is_support) & (~broken_resistance)

    support_hits = int(np.sum(valid_support))
    resistance_hits = int(np.sum(valid_resistance))
    dwell = int(len(pp))
    hits = support_hits + resistance_hits
    valid_any = valid_support | valid_resistance
    nearest_age = int(np.min(current - pp[valid_any])) if np.any(valid_any) else 99999

    hit_score = 1.0 - math.exp(-hits / 18.0)
    dwell_score = 1.0 - math.exp(-dwell / 45.0)
    role_mix = min(support_hits, resistance_hits) / max(1.0, float(max(support_hits, resistance_hits)))
    freshness = 0.0 if nearest_age == 99999 else math.exp(-nearest_age / 200.0)
    density = min(1.0, hits / max(1.0, float(dwell)))
    strength = 100.0 * (0.50 * hit_score + 0.15 * dwell_score + 0.12 * role_mix + 0.13 * freshness + 0.10 * density)
    strength = max(0.0, min(100.0, strength))
    return support_hits, resistance_hits, hits, dwell, nearest_age, strength

def _balance_eval_independent(
    daily: pd.DataFrame, atr: pd.Series, center: float, half: float, scan_limit: int,
    validation_bars: int = 10, cooldown_bars: int = 6, min_reaction_atr: float = 0.20,
) -> tuple[int, int, int, int, int, float]:
    top, bottom = center + half, center - half
    tests = successes = sup_success = res_success = breaks = 0
    last_accepted: int | None = None
    current = len(daily) - 1
    first_offset = validation_bars + 1
    last_offset = min(scan_limit - 1, current)
    if last_offset < first_offset:
        return 0, 0, 0, 0, 0, math.nan

    lows = daily["low"].to_numpy(float)
    highs = daily["high"].to_numpy(float)
    closes = daily["close"].to_numpy(float)
    atr_arr = atr.to_numpy(float)

    for offset in range(first_offset, last_offset + 1):
        pos = current - offset
        if pos - 1 < 0:
            continue
        overlaps = lows[pos] <= top and highs[pos] >= bottom
        previous_above = closes[pos - 1] > top
        previous_below = closes[pos - 1] < bottom
        independent_entry = overlaps and (previous_above or previous_below)
        cooldown_ok = last_accepted is None or abs(offset - last_accepted) >= cooldown_bars
        if not (independent_entry and cooldown_ok):
            continue

        is_support_test = previous_above
        broken = False
        max_away = 0.0
        atr_at_test = atr_arr[pos] if pos < len(atr_arr) and math.isfinite(atr_arr[pos]) else max(1e-12, abs(center) * 1e-6)
        for k in range(1, validation_bars + 1):
            later = pos + k
            if later <= current:
                c = closes[later]
                if is_support_test:
                    if c < bottom:
                        broken = True
                    max_away = max(max_away, highs[later] - center)
                else:
                    if c > top:
                        broken = True
                    max_away = max(max_away, center - lows[later])
        reacted = max_away >= atr_at_test * min_reaction_atr
        success = (not broken) and reacted
        tests += 1
        if success:
            successes += 1
            if is_support_test:
                sup_success += 1
            else:
                res_success += 1
        if broken:
            breaks += 1
        last_accepted = offset

    reliability = 100.0 * successes / tests if tests else math.nan
    return tests, successes, sup_success, res_success, breaks, reliability


def analyze_balance_zones(
    daily: pd.DataFrame,
    lookback: int = LOOKBACK,
    scan_step_pct: float = 1.0,
    max_zones: int = 9,
    zone_half_atr: float = 0.12,
    min_spacing_range_pct: float = 8.0,
) -> dict[str, Any]:
    unavailable = {"available": False, "zones": [], "detail": "Dati Daily insufficienti."}
    if daily.empty or len(daily) < max(60, min(lookback, 120)):
        return unavailable
    data = daily.copy().dropna(subset=["high", "low", "close"])
    if len(data) < 60:
        return unavailable

    atr = wilder_atr(data, 14)
    atr_now = float(atr.iloc[-1]) if not atr.empty and pd.notna(atr.iloc[-1]) else math.nan
    if not math.isfinite(atr_now) or atr_now <= 0:
        return unavailable

    scan_limit = min(int(lookback), len(data) - 1)
    window = data.iloc[-min(int(lookback), len(data)):]
    range_high = float(window["high"].max())
    range_low = float(window["low"].min())
    active_range = range_high - range_low
    if not math.isfinite(active_range) or active_range <= 0:
        return unavailable

    half = max(1e-12, atr_now * float(zone_half_atr))
    closes_arr = data["close"].to_numpy(float)
    future_min, future_max = _future_close_extremes(closes_arr, 10)
    candidates: list[tuple[float, float, int, int, int, int, int]] = []
    steps = int(round(100.0 / float(scan_step_pct)))
    for step_idx in range(steps + 1):
        pct = min(100.0, step_idx * float(scan_step_pct))
        center = range_low + active_range * pct / 100.0
        sup, res, hits, dwell, age, strength = _balance_eval_compatible(
            data, center, half, scan_limit, future_min=future_min, future_max=future_max
        )
        if hits >= 1:
            candidates.append((center, strength, hits, sup, res, dwell, age))

    if not candidates:
        return {**unavailable, "detail": "Nessuna Balance qualificata nel lookback corrente."}

    spacing = active_range * float(min_spacing_range_pct) / 100.0
    selected: list[BalanceZone] = []
    for center, strength, hits, sup, res, dwell, age in sorted(candidates, key=lambda x: x[1], reverse=True):
        if len(selected) >= int(max_zones):
            break
        if any(abs(center - z.center) < spacing for z in selected):
            continue
        tests, succ, ind_sup, ind_res, brk, rel = _balance_eval_independent(data, atr, center, half, scan_limit)
        selected.append(BalanceZone(
            center=center,
            half=half,
            pct_range=100.0 * (center - range_low) / active_range,
            strength=strength,
            hits=hits,
            support_hits=sup,
            resistance_hits=res,
            dwell=dwell,
            last_hit_age=age,
            independent_tests=tests,
            independent_successes=succ,
            independent_support_success=ind_sup,
            independent_resistance_success=ind_res,
            independent_breaks=brk,
            reliability=rel,
        ))
    selected.sort(key=lambda z: z.center)
    if not selected:
        return {**unavailable, "detail": "Nessuna Balance selezionata."}

    return {
        "available": True,
        "zones": selected,
        "atr14": atr_now,
        "range_high": range_high,
        "range_low": range_low,
        "detail": f"{len(selected)} Balance selezionate | Lookback {scan_limit} Daily | ATR14 {atr_now:.4f}",
    }


# =============================================================================
# ACTIVE AREA — nessuna tolleranza aggiuntiva
# =============================================================================
def candle_touches_zone(row: pd.Series, z: BalanceZone) -> bool:
    bottom = z.center - z.half
    top = z.center + z.half
    return float(row["low"]) <= top and float(row["high"]) >= bottom


def price_inside_zone(price: float, z: BalanceZone) -> bool:
    bottom = z.center - z.half
    top = z.center + z.half
    return bottom <= float(price) <= top


def active_zone_rows(
    label: str, ticker: str, data: pd.DataFrame, balance: dict[str, Any], require_last_close_inside: bool, min_touches: int
) -> list[dict[str, Any]]:
    if data.empty or len(data) < LAST_TOUCH_BARS:
        return []

    recent = data.iloc[-LAST_TOUCH_BARS:]
    last_closed = data.iloc[-1]
    last_close = float(last_closed["close"])
    last_date = pd.Timestamp(data.index[-1])
    rows: list[dict[str, Any]] = []

    for idx, z in enumerate(balance.get("zones", []), start=1):
        touches = [candle_touches_zone(recent.iloc[i], z) for i in range(LAST_TOUCH_BARS)]
        touch_count = int(sum(touches))
        close_inside = price_inside_zone(last_close, z)
        active = touch_count >= int(min_touches) and (close_inside if require_last_close_inside else True)
        if not active:
            continue

        rows.append({
            "Strumento": label,
            "Ticker": ticker,
            "Area": "AREA ATTIVA",
            "Ultimo Close": last_close,
            "Balance": z.center,
            "Zona min": z.center - z.half,
            "Zona max": z.center + z.half,
            "Tocchi ultime 5 chiuse": f"{touch_count}/5",
            "Tocco -4": "SI" if touches[0] else "NO",
            "Tocco -3": "SI" if touches[1] else "NO",
            "Tocco -2": "SI" if touches[2] else "NO",
            "Tocco -1": "SI" if touches[3] else "NO",
            "Tocco 0": "SI" if touches[4] else "NO",
            "Ultimo Close dentro": "SI" if close_inside else "NO",
            "H": z.hits,
            "Strength": z.strength,
            "Test indipendenti": z.independent_tests,
            "Successi indipendenti": z.independent_successes,
            "Break indipendenti": z.independent_breaks,
            "Reliability %": z.reliability,
            "Support H": z.support_hits,
            "Resistance H": z.resistance_hits,
            "Dwell": z.dwell,
            "Last Hit Age": z.last_hit_age,
            "Data ultima Daily chiusa": last_date.strftime("%Y-%m-%d"),
            "_zone_index": idx,
        })
    return rows


def all_balance_rows(
    label: str, ticker: str, data: pd.DataFrame, balance: dict[str, Any], require_last_close_inside: bool, min_touches: int
) -> list[dict[str, Any]]:
    if data.empty:
        return []
    last_close = float(data.iloc[-1]["close"])
    recent = data.iloc[-min(LAST_TOUCH_BARS, len(data)):]
    rows: list[dict[str, Any]] = []
    for idx, z in enumerate(balance.get("zones", []), start=1):
        touches = [candle_touches_zone(recent.iloc[i], z) for i in range(len(recent))]
        touch_count = int(sum(touches))
        close_inside = price_inside_zone(last_close, z)
        active = len(recent) == LAST_TOUCH_BARS and touch_count >= int(min_touches) and (close_inside if require_last_close_inside else True)
        rows.append({
            "Strumento": label,
            "Ticker": ticker,
            "Zona #": idx,
            "Area attiva": "SI" if active else "NO",
            "Ultimo Close": last_close,
            "Balance": z.center,
            "Zona min": z.center - z.half,
            "Zona max": z.center + z.half,
            "Tocchi ultime 5 chiuse": f"{touch_count}/5" if len(recent) == 5 else str(touch_count),
            "Ultimo Close dentro": "SI" if close_inside else "NO",
            "H": z.hits,
            "Strength": z.strength,
            "Test indipendenti": z.independent_tests,
            "Successi indipendenti": z.independent_successes,
            "Break indipendenti": z.independent_breaks,
            "Reliability %": z.reliability,
            "Support H": z.support_hits,
            "Resistance H": z.resistance_hits,
            "Dwell": z.dwell,
            "Last Hit Age": z.last_hit_age,
        })
    return rows


# =============================================================================
# CHART / EXCEL
# =============================================================================
def plot_balance(data: pd.DataFrame, balance: dict[str, Any], ticker: str, active_center: float) -> go.Figure:
    chart = data.tail(220)
    fig = go.Figure(data=[go.Candlestick(
        x=chart.index,
        open=chart["open"], high=chart["high"], low=chart["low"], close=chart["close"],
        name=ticker,
    )])
    for z in balance.get("zones", []):
        selected = abs(z.center - active_center) <= max(1e-12, z.half * 0.05)
        fill = "rgba(255,215,0,0.28)" if selected else "rgba(120,120,120,0.10)"
        line = "rgba(220,170,0,1.0)" if selected else "rgba(120,120,120,0.65)"
        width = 2 if selected else 1
        fig.add_hrect(y0=z.center - z.half, y1=z.center + z.half, fillcolor=fill, line_width=0)
        fig.add_hline(y=z.center, line_width=width, line_color=line)
    fig.update_layout(
        title=f"{ticker} · Daily chiusa · Balance attiva evidenziata",
        height=650,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig


def excel_bytes(active: pd.DataFrame, all_zones: pd.DataFrame, errors: list[dict[str, str]]) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        active.to_excel(writer, sheet_name="Aree attive", index=False)
        all_zones.to_excel(writer, sheet_name="Tutte le Balance", index=False)
        if errors:
            pd.DataFrame(errors).to_excel(writer, sheet_name="Errori", index=False)

        wb = writer.book
        header = wb.add_format({"bold": True, "bg_color": "#1F4E78", "font_color": "#FFFFFF", "border": 1, "align": "center"})
        n4 = wb.add_format({"num_format": "0.0000"})
        n2 = wb.add_format({"num_format": "0.00"})
        active_fmt = wb.add_format({"bg_color": "#FFF2CC", "bold": True})

        for name, df in [("Aree attive", active), ("Tutte le Balance", all_zones)]:
            ws = writer.sheets[name]
            ws.freeze_panes(1, 0)
            if len(df.columns):
                ws.autofilter(0, 0, max(0, len(df)), len(df.columns) - 1)
            ws.set_row(0, 24, header)
            for j, col in enumerate(df.columns):
                width = min(28, max(11, len(str(col)) + 2))
                if col in {"Strumento", "Ticker", "Area", "Tocchi ultime 5 chiuse"}:
                    width = max(width, 16)
                fmt = None
                if col in {"Ultimo Close", "Balance", "Zona min", "Zona max"}:
                    fmt = n4
                elif col in {"Strength", "Reliability %"}:
                    fmt = n2
                ws.set_column(j, j, width, fmt)
            if len(df) and "Area attiva" in df.columns:
                c = df.columns.get_loc("Area attiva")
                ws.conditional_format(1, c, len(df), c, {"type": "text", "criteria": "containing", "value": "SI", "format": active_fmt})
    return out.getvalue()


def scan_signature(universe: list[tuple[str, str]], market: str, adjusted: bool, require_last_close_inside: bool, min_touches: int) -> str:
    raw = "|".join([market, str(bool(adjusted)), str(bool(require_last_close_inside)), str(int(min_touches))] + [t for _, t in universe])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# =============================================================================
# UI
# =============================================================================
with st.sidebar:
    st.header("Impostazioni")
    market_choice = st.selectbox(
        "Mercato lista",
        ["Automatico", "Italia", "USA", "Misto / ticker Yahoo completi"],
        index=0,
        help="Automatico: riconosce i file italiani con ticker .MI e i file USA con ticker standard.",
    )
    adjusted = st.checkbox(
        "Prezzi Yahoo adjusted",
        value=False,
        help="È una scelta della serie dati, non un filtro dello screener. Per confronti con TradingView usa la stessa impostazione di aggiustamento del grafico.",
    )
    require_last_close_inside = st.checkbox(
        "Richiedi ultimo Close Daily dentro la Balance",
        value=False,
        help="ON: oltre al numero minimo di tocchi reali nelle ultime 5 Daily chiuse, l'ultimo Close deve essere dentro la stessa fascia.",
    )
    min_touches = st.number_input(
        "Tocchi minimi",
        min_value=1,
        max_value=5,
        value=DEFAULT_MIN_TOUCHES,
        step=1,
        help="Numero minimo di Daily chiuse, tra le ultime 5, che devono intersecare realmente la stessa Balance.",
    )
    uploaded = st.file_uploader("File ticker .txt", type=["txt", "csv"])
    use_manual = st.checkbox("Modifica/incolla ticker manualmente", value=uploaded is None)
    if use_manual:
        if market_choice == "Italia":
            default_text = italy_example()
        elif market_choice == "USA":
            default_text = usa_example()
        else:
            default_text = "# Ticker oppure Nome;Ticker Yahoo\n"
        manual_text = st.text_area("Ticker", default_text, height=320)
    else:
        manual_text = ""
    run = st.button("🔎 Cerca aree Balance attive", type="primary", use_container_width=True)

source_name = ""
if uploaded is not None and not use_manual:
    source_name = getattr(uploaded, "name", "")
    try:
        text = uploaded.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError:
        text = uploaded.getvalue().decode("latin-1")
else:
    text = manual_text

market = infer_market_from_text(text, source_name) if market_choice == "Automatico" else market_choice
universe = parse_tickers(text, market)
current_signature = scan_signature(universe, market, adjusted, require_last_close_inside, int(min_touches))

with st.sidebar:
    if text.strip():
        if market_choice == "Automatico":
            st.caption(f"Mercato rilevato: **{market}**")
        st.caption(f"Ticker riconosciuti: **{len(universe)}**")
        if universe:
            preview = ", ".join(t for _, t in universe[:8])
            if len(universe) > 8:
                preview += ", …"
            st.caption(f"Anteprima: {preview}")

if run:
    if not universe:
        st.error("Nessun ticker valido nel file/elenco.")
        st.stop()

    # Elimina subito il risultato precedente: durante una scansione USA non deve
    # restare visibile la vecchia tabella italiana.
    st.session_state.pop("balance_stock_screener_v2_6", None)

    labels = {ticker: label for label, ticker in universe}
    tickers = [ticker for _, ticker in universe]
    progress = st.progress(0.0, text="Avvio scansione…")
    data_map, notes = load_universe_data(tickers, adjusted=adjusted, progress=progress)

    active_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    details: dict[str, tuple[pd.DataFrame, dict[str, Any], str]] = {}
    errors: list[dict[str, str]] = []

    for i, ticker in enumerate(tickers, start=1):
        progress.progress(0.35 + 0.65 * ((i - 1) / len(tickers)), text=f"Balance {i}/{len(tickers)} · {ticker}")
        data = data_map.get(ticker, pd.DataFrame())
        data, _open_bar_removed = only_closed_daily(data, ticker, market)
        if data.empty or len(data) < 120:
            errors.append({"Ticker": ticker, "Errore": notes.get(ticker, "Dati insufficienti")})
            continue
        try:
            balance = analyze_balance_zones(data, lookback=LOOKBACK)
            if not balance.get("available"):
                errors.append({"Ticker": ticker, "Errore": str(balance.get("detail", "Balance non disponibili"))})
                continue
            active_rows.extend(active_zone_rows(labels[ticker], ticker, data, balance, require_last_close_inside, int(min_touches)))
            all_rows.extend(all_balance_rows(labels[ticker], ticker, data, balance, require_last_close_inside, int(min_touches)))
            details[ticker] = (data, balance, labels[ticker])
        except Exception as exc:
            errors.append({"Ticker": ticker, "Errore": f"{type(exc).__name__}: {exc}"})

    progress.progress(1.0, text="Completato")
    st.session_state["balance_stock_screener_v2_6"] = {
        "signature": current_signature,
        "active_rows": active_rows,
        "all_rows": all_rows,
        "details": details,
        "errors": errors,
        "adjusted": bool(adjusted),
        "require_last_close_inside": bool(require_last_close_inside),
        "min_touches": int(min_touches),
        "total_tickers": len(tickers),
        "downloaded_tickers": len(data_map),
    }

payload = st.session_state.get("balance_stock_screener_v2_6")
if payload and payload.get("signature") == current_signature:
    active_rows = payload["active_rows"]
    all_rows = payload["all_rows"]
    details = payload["details"]
    errors = payload["errors"]

    active = pd.DataFrame(active_rows)
    all_zones = pd.DataFrame(all_rows)

    c1, c2, c3 = st.columns(3)
    c1.metric("Titoli con Balance calcolate", len(details))
    c2.metric("Aree attive · Daily chiuse", len(active))
    c3.metric("Ticker senza risultato", len(errors))

    total_tickers = int(payload.get("total_tickers", len(details) + len(errors)))
    downloaded_tickers = int(payload.get("downloaded_tickers", len(details)))
    if total_tickers >= 100 and downloaded_tickers < total_tickers * 0.80:
        st.error(
            f"Scansione incompleta: Yahoo Finance ha restituito dati per {downloaded_tickers}/{total_tickers} ticker. "
            "I risultati sotto non rappresentano l'intero universo; controlla la sezione Ticker senza risultato e riprova dopo alcuni minuti."
        )

    st.subheader("Aree Balance attive")
    if active.empty:
        rule = f"almeno {int(payload.get('min_touches', DEFAULT_MIN_TOUCHES))} tocchi reali nelle ultime 5 Daily chiuse" + (" + ultimo Close dentro la Balance" if payload.get("require_last_close_inside", False) else "")
        st.info(f"Nessuna AREA ATTIVA con la regola selezionata: {rule}.")
    else:
        active["_touch_num"] = active["Tocchi ultime 5 chiuse"].str.extract(r"(\d+)")[0].astype(int)
        active = active.sort_values(["_touch_num", "Strength", "Strumento"], ascending=[False, False, True]).reset_index(drop=True)
        visible = [
            "Strumento", "Ticker", "Area", "Ultimo Close", "Balance", "Zona min", "Zona max",
            "Tocchi ultime 5 chiuse", "Tocco -4", "Tocco -3", "Tocco -2", "Tocco -1", "Tocco 0", "Ultimo Close dentro",
            "H", "Strength", "Test indipendenti", "Successi indipendenti", "Break indipendenti", "Reliability %",
            "Data ultima Daily chiusa",
        ]
        st.dataframe(
            active[visible],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Ultimo Close": st.column_config.NumberColumn("Ultimo Close", format="%.4f"),
                "Balance": st.column_config.NumberColumn("Balance", format="%.4f"),
                "Zona min": st.column_config.NumberColumn("Zona min", format="%.4f"),
                "Zona max": st.column_config.NumberColumn("Zona max", format="%.4f"),
                "Strength": st.column_config.NumberColumn("Strength", format="%.1f"),
                "Reliability %": st.column_config.NumberColumn("Reliability %", format="%.1f"),
            },
        )

    active_export = active.drop(columns=["_zone_index", "_touch_num", "_select"], errors="ignore") if not active.empty else pd.DataFrame(columns=[
        "Strumento", "Ticker", "Area", "Ultimo Close", "Balance", "Zona min", "Zona max", "Tocchi ultime 5 chiuse",
        "Ultimo Close dentro", "H", "Strength", "Test indipendenti", "Successi indipendenti", "Break indipendenti", "Reliability %"
    ])
    all_export = all_zones.copy()
    xlsx = excel_bytes(active_export, all_export, errors)
    st.download_button(
        "⬇️ Esporta Excel",
        data=xlsx,
        file_name="balance_stock_active_daily_closed.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    if not active.empty:
        st.subheader("Verifica grafica")
        active["_select"] = active["Ticker"].astype(str) + " | " + active["Balance"].map(lambda x: f"{x:.4f}")
        selected_key = st.selectbox("Area attiva", active["_select"].tolist(), key="balance_stock_active_chart")
        selected_row = active[active["_select"] == selected_key].iloc[0]
        ticker = str(selected_row["Ticker"])
        data, balance, _ = details[ticker]
        st.plotly_chart(plot_balance(data, balance, ticker, float(selected_row["Balance"])), use_container_width=True)
        a, b, c, d = st.columns(4)
        a.metric("Ultimo Close", f"{float(selected_row['Ultimo Close']):.4f}")
        b.metric("Balance", f"{float(selected_row['Balance']):.4f}")
        c.metric("Tocchi ultime 5 chiuse", str(selected_row["Tocchi ultime 5 chiuse"]))
        d.metric("Reliability", "n/d" if pd.isna(selected_row["Reliability %"]) else f"{float(selected_row['Reliability %']):.0f}%")

    with st.expander("Tutte le Balance calcolate"):
        if all_zones.empty:
            st.info("Nessuna Balance disponibile.")
        else:
            st.dataframe(all_zones, hide_index=True, use_container_width=True)

    if errors:
        with st.expander(f"Ticker senza risultato ({len(errors)})"):
            st.dataframe(pd.DataFrame(errors), hide_index=True, use_container_width=True)
