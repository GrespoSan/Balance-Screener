from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any
import io
import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =============================================================================
# PAGE
# =============================================================================
st.set_page_config(page_title="G. Balance Reversal Screener V1", page_icon="⚖️", layout="wide")
st.title("⚖️ G. Balance Reversal Screener V1")
st.caption(
    "Screener indipendente dal COT. Cerca test, reazioni e attraversamenti delle Balance su Daily, H4 e H1. "
    "Il motore Balance deriva dal COT Smart Money V6.49/V6.50; H4 viene ricostruito da dati H1."
)


# =============================================================================
# MODELS
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
# INPUT PARSING
# =============================================================================
def parse_tickers(text: str) -> list[tuple[str, str]]:
    """One instrument per line. Accepted: TICKER or LABEL;TICKER or LABEL,TICKER."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in re.split(r"[;,\t]", line) if p.strip()]
        if len(parts) >= 2:
            label, ticker = parts[0], parts[-1]
        else:
            ticker = parts[0]
            label = ticker
        ticker = ticker.upper()
        if ticker not in seen:
            out.append((label, ticker))
            seen.add(ticker)
    return out


# =============================================================================
# DATA
# =============================================================================
def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        # yfinance may return a ticker level even for a single symbol.
        x.columns = [str(c[0]).lower() for c in x.columns]
    else:
        x.columns = [str(c).lower() for c in x.columns]
    rename = {"adj close": "adj_close"}
    x = x.rename(columns=rename)
    need = ["open", "high", "low", "close"]
    if any(c not in x.columns for c in need):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if "volume" not in x.columns:
        x["volume"] = np.nan
    x = x[["open", "high", "low", "close", "volume"]].copy()
    for c in ["open", "high", "low", "close", "volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["open", "high", "low", "close"])
    x = x[~x.index.duplicated(keep="last")].sort_index()
    return x


def _drop_open_bar(df: pd.DataFrame, timeframe: str) -> tuple[pd.DataFrame, bool]:
    """Conservative closed-bar rule. Returns (data, last_bar_removed)."""
    if df.empty or len(df) < 2:
        return df, False
    x = df.copy()
    idx = x.index
    last = pd.Timestamp(idx[-1])

    try:
        now = pd.Timestamp.now(tz=last.tz) if last.tz is not None else pd.Timestamp.now()
    except Exception:
        now = pd.Timestamp.now()

    remove = False
    if timeframe == "Daily":
        # Daily quote for today's session is considered open.
        last_day = last.tz_localize(None).date() if last.tz is not None else last.date()
        now_day = now.tz_localize(None).date() if getattr(now, "tz", None) is not None else now.date()
        remove = last_day >= now_day
    else:
        bar_delta = pd.Timedelta(hours=1 if timeframe == "H1" else 4)
        # For resampled H4, timestamp marks the start of the bar.
        try:
            remove = last + bar_delta > now
        except TypeError:
            remove = last.tz_localize(None) + bar_delta > now.tz_localize(None)

    return (x.iloc[:-1].copy(), True) if remove else (x, False)


def _resample_h4(h1: pd.DataFrame) -> pd.DataFrame:
    if h1.empty:
        return h1
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    # Clock-time H4 bars. This is intentionally transparent: TradingView H4 can
    # differ when its exchange/session aggregation uses another anchor.
    h4 = h1.resample("4h", origin="start_day").agg(agg)
    return h4.dropna(subset=["open", "high", "low", "close"])


@st.cache_data(ttl=900, show_spinner=False)
def load_price_data(ticker: str, timeframe: str) -> tuple[pd.DataFrame, str]:
    ticker = ticker.strip().upper()
    try:
        if timeframe == "Daily":
            raw = yf.download(ticker, period="5y", interval="1d", auto_adjust=False, progress=False, threads=False)
            data = _normalize_ohlc(raw)
        else:
            # H1 is also the base for H4. 700d is requested; Yahoo may return less
            # depending on the instrument, but 500 bars are usually sufficient.
            raw = yf.download(ticker, period="700d", interval="1h", auto_adjust=False, progress=False, threads=False)
            h1 = _normalize_ohlc(raw)
            data = h1 if timeframe == "H1" else _resample_h4(h1)
        if data.empty:
            return data, "Nessun dato restituito da Yahoo Finance."
        data, removed = _drop_open_bar(data, timeframe)
        note = "Ultima barra aperta esclusa." if removed else "Ultima barra disponibile considerata chiusa."
        return data, note
    except Exception as exc:
        return pd.DataFrame(), f"{type(exc).__name__}: {exc}"


# =============================================================================
# BALANCE ENGINE — porting standalone from COT Smart Money V6.49/V6.50
# =============================================================================
def wilder_atr(data: pd.DataFrame, length: int = 14) -> pd.Series:
    if data.empty:
        return pd.Series(dtype=float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    close = data["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = pd.Series(index=tr.index, dtype=float)
    if len(tr) < length:
        return atr
    seed = float(tr.iloc[:length].mean())
    atr.iloc[length - 1] = seed
    for i in range(length, len(tr)):
        atr.iloc[i] = (float(atr.iloc[i - 1]) * (length - 1) + float(tr.iloc[i])) / length
    return atr


def _balance_eval_compatible(
    data: pd.DataFrame, center: float, half: float, scan_limit: int, validation_bars: int = 10,
) -> tuple[int, int, int, int, int, float]:
    top, bottom = center + half, center - half
    support_hits = resistance_hits = dwell = 0
    nearest_age: int | None = None
    current = len(data) - 1
    first_offset = validation_bars + 1
    last_offset = min(scan_limit - 1, current)
    if last_offset < first_offset:
        return 0, 0, 0, 0, 99999, 0.0
    for offset in range(first_offset, last_offset + 1):
        pos = current - offset
        row = data.iloc[pos]
        overlaps = float(row["low"]) <= top and float(row["high"]) >= bottom
        if not overlaps:
            continue
        dwell += 1
        is_support = float(row["close"]) >= center
        broken = False
        for k in range(1, validation_bars + 1):
            later = pos + k
            if later <= current:
                c = float(data.iloc[later]["close"])
                if is_support and c < bottom:
                    broken = True
                if (not is_support) and c > top:
                    broken = True
        if not broken:
            if is_support:
                support_hits += 1
            else:
                resistance_hits += 1
            nearest_age = offset if nearest_age is None else min(nearest_age, offset)
    hits = support_hits + resistance_hits
    hit_score = 1.0 - math.exp(-hits / 18.0)
    dwell_score = 1.0 - math.exp(-dwell / 45.0)
    role_mix = min(support_hits, resistance_hits) / max(1.0, float(max(support_hits, resistance_hits)))
    freshness = 0.0 if nearest_age is None else math.exp(-nearest_age / 200.0)
    density = min(1.0, hits / max(1.0, float(dwell)))
    strength = 100.0 * (0.50 * hit_score + 0.15 * dwell_score + 0.12 * role_mix + 0.13 * freshness + 0.10 * density)
    strength = max(0.0, min(100.0, strength))
    return support_hits, resistance_hits, hits, dwell, nearest_age if nearest_age is not None else 99999, strength


def _balance_eval_independent(
    data: pd.DataFrame, atr: pd.Series, center: float, half: float, scan_limit: int,
    validation_bars: int = 10, cooldown_bars: int = 6, min_reaction_atr: float = 0.20,
) -> tuple[int, int, int, int, int, float]:
    top, bottom = center + half, center - half
    tests = successes = sup_success = res_success = breaks = 0
    last_accepted: int | None = None
    current = len(data) - 1
    first_offset = validation_bars + 1
    last_offset = min(scan_limit - 1, current)
    if last_offset < first_offset:
        return 0, 0, 0, 0, 0, math.nan
    for offset in range(first_offset, last_offset + 1):
        pos = current - offset
        row = data.iloc[pos]
        prev_pos = pos - 1
        if prev_pos < 0:
            continue
        overlaps = float(row["low"]) <= top and float(row["high"]) >= bottom
        prev_close = float(data.iloc[prev_pos]["close"])
        previous_above = prev_close > top
        previous_below = prev_close < bottom
        independent_entry = overlaps and (previous_above or previous_below)
        cooldown_ok = last_accepted is None or abs(offset - last_accepted) >= cooldown_bars
        if not (independent_entry and cooldown_ok):
            continue
        is_support_test = previous_above
        broken = False
        max_away = 0.0
        atr_at_test = float(atr.iloc[pos]) if pos < len(atr) and not pd.isna(atr.iloc[pos]) else max(1e-12, abs(center) * 1e-6)
        for k in range(1, validation_bars + 1):
            later = pos + k
            if later <= current:
                rr = data.iloc[later]
                c = float(rr["close"])
                if is_support_test:
                    if c < bottom:
                        broken = True
                    max_away = max(max_away, float(rr["high"]) - center)
                else:
                    if c > top:
                        broken = True
                    max_away = max(max_away, center - float(rr["low"]))
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
    data: pd.DataFrame,
    timeframe: str,
    lookback: int = 500,
    scan_step_pct: float = 1.0,
    max_zones: int = 9,
    zone_half_atr: float = 0.12,
    min_spacing_range_pct: float = 8.0,
) -> dict[str, Any]:
    """Generic-timeframe adaptation of Balance First v0.5.1.9 / Bridge V1.5.57."""
    unavailable = {
        "available": False, "state": 0, "state_text": "BALANCE NON DISPONIBILI",
        "zones": [], "origin": math.nan, "next_zone": math.nan, "detail": "Dati insufficienti.",
    }
    if data.empty or len(data) < max(60, min(lookback, 120)):
        return unavailable
    data = data.copy().dropna(subset=["high", "low", "close"])
    if len(data) < 60:
        return unavailable
    atr = wilder_atr(data, 14)
    atr_now = float(atr.iloc[-1]) if not atr.empty and not pd.isna(atr.iloc[-1]) else math.nan
    if pd.isna(atr_now) or atr_now <= 0:
        return unavailable

    scan_limit = min(int(lookback), len(data) - 1)
    window = data.iloc[-min(int(lookback), len(data)):]
    range_high = float(window["high"].max())
    range_low = float(window["low"].min())
    active_range = range_high - range_low
    if not math.isfinite(active_range) or active_range <= 0:
        return unavailable
    half = max(1e-12, atr_now * float(zone_half_atr))

    candidates: list[tuple[float, float, int]] = []
    steps = int(round(100.0 / float(scan_step_pct)))
    for s in range(steps + 1):
        pct = min(100.0, s * float(scan_step_pct))
        center = range_low + active_range * pct / 100.0
        sup, res, hits, dwell, age, strength = _balance_eval_compatible(data, center, half, scan_limit)
        if hits >= 1:
            candidates.append((center, strength, hits))

    if not candidates:
        return {**unavailable, "detail": "Nessuna Balance qualificata nel lookback corrente."}

    spacing = active_range * float(min_spacing_range_pct) / 100.0
    selected: list[BalanceZone] = []
    for center, strength, _ in sorted(candidates, key=lambda x: x[1], reverse=True):
        if len(selected) >= int(max_zones):
            break
        if any(abs(center - z.center) < spacing for z in selected):
            continue
        sup, res, hits, dwell, age, strength2 = _balance_eval_compatible(data, center, half, scan_limit)
        tests, succ, ind_sup, ind_res, brk, rel = _balance_eval_independent(data, atr, center, half, scan_limit)
        selected.append(BalanceZone(
            center=center, half=half, pct_range=100.0 * (center - range_low) / active_range,
            strength=strength2, hits=hits, support_hits=sup, resistance_hits=res,
            dwell=dwell, last_hit_age=age, independent_tests=tests,
            independent_successes=succ, independent_support_success=ind_sup,
            independent_resistance_success=ind_res, independent_breaks=brk, reliability=rel,
        ))
    selected.sort(key=lambda z: z.center)
    if not selected:
        return {**unavailable, "detail": "Nessuna Balance selezionata dopo il filtro di distanza."}

    # Input is already closed-bar only; therefore the last bar is the causal reference bar.
    ref_pos = len(data) - 1
    if ref_pos < 1:
        return unavailable
    ref_close = float(data.iloc[ref_pos]["close"])
    prev_close = float(data.iloc[ref_pos - 1]["close"])
    low_ref = float(data.iloc[ref_pos]["low"])
    high_ref = float(data.iloc[ref_pos]["high"])
    recent_start = max(0, ref_pos - 5)
    recent_slice = data.iloc[recent_start:ref_pos + 1]
    recent_low = float(recent_slice["low"].min())
    recent_high = float(recent_slice["high"].max())
    cross_up_dist = cross_dn_dist = react_up_dist = react_dn_dist = 1e20
    bull_origin = bear_origin = math.nan
    bull_age = bear_age = 100000
    inside_any = False
    current = ref_pos

    for z in selected:
        top, bottom = z.center + z.half, z.center - z.half
        d = abs(ref_close - z.center)
        inside_now = bottom <= ref_close <= top
        crossed_up = ref_close > top and prev_close <= top
        crossed_dn = ref_close < bottom and prev_close >= bottom
        reacted_up = ref_close > top and recent_low <= top and recent_low >= bottom - z.half
        reacted_dn = ref_close < bottom and recent_high >= bottom and recent_high <= top + z.half
        touch_age = 100000
        touch_stop = min(current + 1, 60)
        for offset in range(0, touch_stop):
            rr = data.iloc[current - offset]
            if float(rr["low"]) <= top and float(rr["high"]) >= bottom:
                touch_age = offset
                break
        inside_any = inside_any or inside_now
        if crossed_up:
            cross_up_dist = min(cross_up_dist, d)
        if crossed_dn:
            cross_dn_dist = min(cross_dn_dist, d)
        if reacted_up:
            react_up_dist = min(react_up_dist, d)
        if reacted_dn:
            react_dn_dist = min(react_dn_dist, d)
        if touch_age < 100000 and ref_close > top and touch_age < bull_age:
            bull_age, bull_origin = touch_age, z.center
        if touch_age < 100000 and ref_close < bottom and touch_age < bear_age:
            bear_age, bear_origin = touch_age, z.center

    path_ref_pos = ref_pos - 10
    path_ref = float(data.iloc[path_ref_pos]["close"]) if path_ref_pos >= 0 else math.nan
    path_up = not pd.isna(path_ref) and ref_close > path_ref
    path_down = not pd.isna(path_ref) and ref_close < path_ref
    bull_path = not pd.isna(bull_origin) and (pd.isna(bear_origin) or path_up or ((not path_down) and bull_age < bear_age))
    bear_path = not pd.isna(bear_origin) and (pd.isna(bull_origin) or path_down or ((not path_up) and bear_age < bull_age))

    state = 0
    next_center = next_bottom = next_top = math.nan
    origin = origin_age = math.nan
    path_direction = "NESSUN PERCORSO"
    if bull_path:
        path_direction = "RIALZISTA"
        origin = bull_origin
        origin_age = float(bull_age)
        above = [z for z in selected if z.center > bull_origin]
        if above:
            nxt = min(above, key=lambda z: z.center)
            next_center = nxt.center
            next_bottom, next_top = nxt.center - nxt.half, nxt.center + nxt.half
            state = 5 if ref_close > next_top else 4 if (high_ref >= next_bottom and low_ref <= next_top) else 2
        else:
            state = 2
    elif bear_path:
        path_direction = "RIBASSISTA"
        origin = bear_origin
        origin_age = float(bear_age)
        below = [z for z in selected if z.center < bear_origin]
        if below:
            nxt = max(below, key=lambda z: z.center)
            next_center = nxt.center
            next_bottom, next_top = nxt.center - nxt.half, nxt.center + nxt.half
            state = -5 if ref_close < next_bottom else -4 if (low_ref <= next_top and high_ref >= next_bottom) else -2
        else:
            state = -2
    elif cross_up_dist < 1e20 or cross_dn_dist < 1e20:
        state = 3 if cross_up_dist <= cross_dn_dist else -3
    elif react_up_dist < 1e20 or react_dn_dist < 1e20:
        state = 2 if react_up_dist <= react_dn_dist else -2
    elif inside_any:
        state = 1

    labels = {
        5: "BALANCE SUPERIORE RECUPERATA DOPO LA REAZIONE",
        4: "PROGRESSIONE RIALZISTA FINO ALLA BALANCE SUPERIORE",
        3: "BALANCE RECUPERATA AL RIALZO",
        2: "REAZIONE RIALZISTA DA UNA BALANCE",
        1: "PREZZO IN TEST DI UNA BALANCE",
        0: "NESSUNA CONFERMA RECENTE DALLE BALANCE",
        -2: "REAZIONE RIBASSISTA DA UNA BALANCE",
        -3: "BALANCE PERSA AL RIBASSO",
        -4: "PROGRESSIONE RIBASSISTA FINO ALLA BALANCE INFERIORE",
        -5: "BALANCE INFERIORE PERSA DOPO LA REAZIONE",
    }
    return {
        "available": True, "state": state, "state_text": labels.get(state, labels[0]),
        "zones": selected, "origin": origin, "next_zone": next_center,
        "origin_touch_age": origin_age, "next_bottom": next_bottom, "next_top": next_top,
        "path_direction": path_direction,
        "reference_date": str(data.index[ref_pos]),
        "reference_open": float(data.iloc[ref_pos]["open"]),
        "reference_high": high_ref, "reference_low": low_ref,
        "close": ref_close, "atr14": atr_now, "range_high": range_high, "range_low": range_low,
        "timeframe": timeframe,
        "detail": f"{len(selected)} Balance selezionate | Lookback {scan_limit} {timeframe} | ATR14 {atr_now:.4f}",
    }


def balance_role(z: BalanceZone, ref_close: float) -> str:
    inside = z.center - z.half <= ref_close <= z.center + z.half
    if inside:
        return "BALANCE"
    enough_independent = z.independent_tests >= 2 and not pd.isna(z.reliability) and z.reliability >= 35.0
    if enough_independent:
        support_dom = z.independent_support_success >= 2 and z.independent_support_success >= max(1.0, z.independent_resistance_success * 1.35)
        resistance_dom = z.independent_resistance_success >= 2 and z.independent_resistance_success >= max(1.0, z.independent_support_success * 1.35)
        if z.center < ref_close and support_dom:
            return "SUPPORTO"
        if z.center > ref_close and resistance_dom:
            return "RESISTENZA"
    compatible_support = z.support_hits >= 2 and z.support_hits >= max(1.0, z.resistance_hits * 1.35)
    compatible_resistance = z.resistance_hits >= 2 and z.resistance_hits >= max(1.0, z.support_hits * 1.35)
    if z.center < ref_close and (compatible_support or z.hits >= 2):
        return "SUPPORTO"
    if z.center > ref_close and (compatible_resistance or z.hits >= 2):
        return "RESISTENZA"
    return "BALANCE"


# =============================================================================
# SCREENER LOGIC
# =============================================================================
def _nearest_zone(balance: dict[str, Any]) -> BalanceZone | None:
    zones: list[BalanceZone] = list(balance.get("zones", []))
    if not zones:
        return None
    ref = float(balance.get("close", math.nan))
    origin = float(balance.get("origin", math.nan))
    if math.isfinite(origin):
        return min(zones, key=lambda z: abs(z.center - origin))
    return min(zones, key=lambda z: abs(z.center - ref))


def setup_label(state: int) -> str:
    return {
        5: "RECUPERO COMPLETO LONG",
        4: "TEST BALANCE SUPERIORE",
        3: "RECUPERO LONG",
        2: "REAZIONE LONG",
        1: "TEST BALANCE",
        0: "NESSUN SETUP",
        -2: "REAZIONE SHORT",
        -3: "PERDITA SHORT",
        -4: "TEST BALANCE INFERIORE",
        -5: "PERDITA COMPLETA SHORT",
    }.get(state, "NESSUN SETUP")


def priority_score(balance: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Priority ranking only; NOT a probability or backtested expectancy."""
    state = int(balance.get("state", 0))
    event_map = {2: 100, -2: 100, 1: 85, 3: 75, -3: 75, 4: 65, -4: 65, 5: 55, -5: 55, 0: 0}
    event = float(event_map.get(state, 0))
    z = _nearest_zone(balance)
    if z is None:
        return 0.0, {"event": event, "strength": 0.0, "reliability": 0.0, "freshness": 0.0, "proximity": 0.0}
    strength = float(z.strength)
    reliability = float(z.reliability) if not pd.isna(z.reliability) else 0.0
    age = float(balance.get("origin_touch_age", math.nan))
    if not math.isfinite(age):
        age = float(z.last_hit_age) if z.last_hit_age < 99999 else 60.0
    freshness = 100.0 * math.exp(-max(0.0, age) / 8.0)
    atr = float(balance.get("atr14", math.nan))
    close = float(balance.get("close", math.nan))
    dist_atr = abs(close - z.center) / atr if math.isfinite(atr) and atr > 0 else 10.0
    proximity = max(0.0, 100.0 * (1.0 - min(dist_atr, 2.0) / 2.0))
    score = 0.50 * event + 0.20 * strength + 0.15 * reliability + 0.10 * freshness + 0.05 * proximity
    return round(max(0.0, min(100.0, score)), 1), {
        "event": event, "strength": round(strength, 1), "reliability": round(reliability, 1),
        "freshness": round(freshness, 1), "proximity": round(proximity, 1),
    }


def result_row(label: str, ticker: str, timeframe: str, data: pd.DataFrame, balance: dict[str, Any], data_note: str) -> dict[str, Any]:
    state = int(balance.get("state", 0))
    zone = _nearest_zone(balance)
    close = float(balance.get("close", math.nan))
    atr = float(balance.get("atr14", math.nan))
    score, parts = priority_score(balance)
    if zone is not None:
        dist_atr = abs(close - zone.center) / atr if math.isfinite(atr) and atr > 0 else math.nan
        role = balance_role(zone, close)
        zone_center = zone.center
        zone_low = zone.center - zone.half
        zone_high = zone.center + zone.half
        strength = zone.strength
        reliability = zone.reliability
        tests = zone.independent_tests
    else:
        dist_atr = zone_center = zone_low = zone_high = strength = reliability = math.nan
        tests = 0
        role = "N/D"
    return {
        "Strumento": label,
        "Ticker": ticker,
        "TF": timeframe,
        "Setup": setup_label(state),
        "State": state,
        "Score priorità": score,
        "Prezzo": close,
        "Balance": zone_center,
        "Zona min": zone_low,
        "Zona max": zone_high,
        "Ruolo": role,
        "Distanza ATR": dist_atr,
        "Strength": strength,
        "Reliability %": reliability,
        "Test indipendenti": tests,
        "Balance origine": balance.get("origin", math.nan),
        "Balance successiva": balance.get("next_zone", math.nan),
        "Percorso": balance.get("path_direction", ""),
        "Barra riferimento": balance.get("reference_date", ""),
        "Nota dati": data_note,
        "_score_event": parts["event"],
        "_score_strength": parts["strength"],
        "_score_reliability": parts["reliability"],
        "_score_freshness": parts["freshness"],
        "_score_proximity": parts["proximity"],
    }


def plot_balance(data: pd.DataFrame, balance: dict[str, Any], ticker: str, timeframe: str) -> go.Figure:
    bars = 180 if timeframe == "Daily" else 260
    chart = data.tail(bars)
    fig = go.Figure(data=[go.Candlestick(
        x=chart.index, open=chart["open"], high=chart["high"], low=chart["low"], close=chart["close"], name=ticker
    )])
    ref_close = float(balance.get("close", math.nan))
    origin = float(balance.get("origin", math.nan))
    next_zone = float(balance.get("next_zone", math.nan))
    for z in balance.get("zones", []):
        role = balance_role(z, ref_close)
        annotation = f"{role} {z.center:.4f}"
        if math.isfinite(origin) and abs(z.center - origin) <= max(1e-12, z.half):
            annotation += " · ORIGINE"
        if math.isfinite(next_zone) and abs(z.center - next_zone) <= max(1e-12, z.half):
            annotation += " · SUCCESSIVA"
        fig.add_hrect(y0=z.center-z.half, y1=z.center+z.half, opacity=0.10, line_width=0,
                      annotation_text=annotation, annotation_position="top right")
        fig.add_hline(y=z.center, line_dash="dot", line_width=1)
    fig.update_layout(
        title=f"{ticker} · {timeframe} · {balance.get('state_text', '')}", height=620,
        margin=dict(l=20, r=20, t=60, b=20), xaxis_rangeslider_visible=False,
    )
    return fig


# =============================================================================
# UI
# =============================================================================
def default_universe() -> str:
    return """# Un ticker Yahoo Finance per riga\nES;ES=F\nNQ;NQ=F\nYM;YM=F\nRTY;RTY=F\nGold;GC=F\nSilver;SI=F\nCopper;HG=F\nWTI;CL=F\nNatGas;NG=F\nEURUSD;EURUSD=X\nGBPUSD;GBPUSD=X\nUSDJPY;JPY=X\n"""

with st.sidebar:
    st.header("Impostazioni")
    timeframe = st.selectbox("Timeframe", ["Daily", "H4", "H1"], index=0)
    lookback = st.number_input("Lookback Balance (barre)", min_value=120, max_value=1000, value=500, step=50)
    show_only_active = st.checkbox("Mostra solo setup attivi", value=True)
    min_score = st.slider("Score priorità minimo", 0, 100, 45, 1)
    st.divider()
    uploaded = st.file_uploader("File ticker .txt", type=["txt", "csv"])
    use_manual = st.checkbox("Modifica/incolla ticker manualmente", value=uploaded is None)
    manual_text = st.text_area("Ticker", default_universe(), height=260, disabled=not use_manual)
    run = st.button("🔎 Avvia screener", type="primary", use_container_width=True)

st.info(
    "V1 metodologica: nessun COT, EMA, RSI o altro filtro. Lo Score serve solo a ordinare i casi più interessanti; "
    "non rappresenta una probabilità di successo. H4 è ricostruito da H1 e può non coincidere perfettamente con le barre H4 di TradingView."
)

if uploaded is not None and not use_manual:
    try:
        text = uploaded.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError:
        text = uploaded.getvalue().decode("latin-1")
else:
    text = manual_text

universe = parse_tickers(text)

if run:
    if not universe:
        st.error("Nessun ticker valido nel file/elenco.")
        st.stop()

    rows: list[dict[str, Any]] = []
    details: dict[str, tuple[pd.DataFrame, dict[str, Any], str]] = {}
    errors: list[dict[str, str]] = []
    bar = st.progress(0.0, text="Avvio screener…")

    for i, (label, ticker) in enumerate(universe, start=1):
        bar.progress((i - 1) / len(universe), text=f"{ticker} · download e analisi Balance")
        data, note = load_price_data(ticker, timeframe)
        if data.empty or len(data) < 60:
            errors.append({"Ticker": ticker, "Errore": note if note else "Dati insufficienti"})
            continue
        try:
            balance = analyze_balance_zones(data, timeframe=timeframe, lookback=int(lookback))
            if not balance.get("available"):
                errors.append({"Ticker": ticker, "Errore": str(balance.get("detail", "Balance non disponibile"))})
                continue
            row = result_row(label, ticker, timeframe, data, balance, note)
            rows.append(row)
            details[ticker] = (data, balance, label)
        except Exception as exc:
            errors.append({"Ticker": ticker, "Errore": f"{type(exc).__name__}: {exc}"})

    bar.progress(1.0, text="Analisi completata")
    st.session_state["balance_scan"] = {
        "rows": rows, "details": details, "errors": errors, "timeframe": timeframe, "lookback": int(lookback)
    }

payload = st.session_state.get("balance_scan")

if payload:
    rows = payload["rows"]
    details = payload["details"]
    errors = payload["errors"]
    scan_timeframe = payload["timeframe"]

    if not rows:
        st.warning("Nessun risultato disponibile.")
        if errors:
            st.dataframe(pd.DataFrame(errors), hide_index=True, use_container_width=True)
        st.stop()

    full = pd.DataFrame(rows).sort_values(["Score priorità", "Distanza ATR"], ascending=[False, True]).reset_index(drop=True)
    filtered = full.copy()
    if show_only_active:
        filtered = filtered[filtered["State"] != 0]
    filtered = filtered[filtered["Score priorità"] >= min_score].reset_index(drop=True)
    filtered.insert(0, "Rank", range(1, len(filtered) + 1))

    if scan_timeframe != timeframe:
        st.warning(f"I risultati visualizzati sono del timeframe {scan_timeframe}. Premi ‘Avvia screener’ per ricalcolare {timeframe}.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strumenti analizzati", len(rows))
    c2.metric("Setup attivi", int((full["State"] != 0).sum()))
    c3.metric("Reazioni", int(full["State"].isin([2, -2]).sum()))
    c4.metric("In test", int((full["State"] == 1).sum()))

    st.subheader("Classifica attuale")
    visible_cols = [
        "Rank", "Strumento", "Ticker", "TF", "Setup", "State", "Score priorità", "Prezzo", "Balance",
        "Ruolo", "Distanza ATR", "Strength", "Reliability %", "Test indipendenti", "Balance origine",
        "Balance successiva", "Percorso", "Barra riferimento",
    ]
    st.dataframe(
        filtered[visible_cols], hide_index=True, use_container_width=True,
        column_config={
            "Score priorità": st.column_config.ProgressColumn("Score priorità", min_value=0, max_value=100, format="%.1f"),
            "Distanza ATR": st.column_config.NumberColumn("Dist. ATR", format="%.2f"),
            "Strength": st.column_config.NumberColumn("Strength", format="%.1f"),
            "Reliability %": st.column_config.NumberColumn("Reliability %", format="%.1f"),
            "Prezzo": st.column_config.NumberColumn("Prezzo", format="%.4f"),
            "Balance": st.column_config.NumberColumn("Balance", format="%.4f"),
            "Balance origine": st.column_config.NumberColumn("Balance origine", format="%.4f"),
            "Balance successiva": st.column_config.NumberColumn("Balance successiva", format="%.4f"),
        },
    )

    csv = filtered[visible_cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Esporta CSV", csv, file_name=f"balance_screener_{scan_timeframe}.csv", mime="text/csv")

    if not filtered.empty:
        st.subheader("Verifica grafica")
        selected_ticker = st.selectbox("Strumento da visualizzare", filtered["Ticker"].tolist(), key="balance_chart_ticker")
        data, balance, label = details[selected_ticker]
        st.plotly_chart(plot_balance(data, balance, selected_ticker, scan_timeframe), use_container_width=True)

        z = _nearest_zone(balance)
        if z is not None:
            score, parts = priority_score(balance)
            a, b, c, d = st.columns(4)
            a.metric("State", f"{int(balance['state']):+d}" if int(balance['state']) else "0")
            b.metric("Score priorità", f"{score:.1f}")
            c.metric("Strength zona", f"{z.strength:.1f}")
            d.metric("Reliability", "—" if pd.isna(z.reliability) else f"{z.reliability:.1f}%")
            with st.expander("Composizione Score priorità"):
                st.write(parts)
                st.caption("Ranking euristico della V1: non è una probabilità e non è ancora un risultato di backtest.")

    if errors:
        with st.expander(f"Ticker senza risultato ({len(errors)})"):
            st.dataframe(pd.DataFrame(errors), hide_index=True, use_container_width=True)

else:
    st.subheader("Obiettivo della V1")
    st.markdown(
        """
- **TEST BALANCE**: prezzo dentro una zona; nessuna direzione ancora confermata.
- **REAZIONE LONG / SHORT**: il prezzo ha toccato una Balance e si è allontanato nella direzione opposta.
- **RECUPERO / PERDITA**: attraversamento confermato della fascia.
- **PROGRESSIONE**: movimento dalla Balance di origine verso la Balance successiva.

La priorità iniziale è trovare **reazioni fresche** e **test in corso**. Prima di aggiungere filtri esterni va verificato, con replay/backtest, se questi eventi hanno follow-through sufficiente.
        """
    )
