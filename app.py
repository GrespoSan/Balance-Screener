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
st.set_page_config(page_title="G. Balance Opportunity Screener V2", page_icon="⚖️", layout="wide")
st.title("⚖️ G. Balance Opportunity Screener V2")

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
# SCREENER LOGIC — V2: opportunità in formazione sulle Balance
# =============================================================================
def _zone_historical_role(z: BalanceZone) -> tuple[str, float]:
    """Historical bias of the zone. Used as context, never as a hard filter."""
    sup = float(z.support_hits) + 2.0 * float(z.independent_support_success)
    res = float(z.resistance_hits) + 2.0 * float(z.independent_resistance_success)
    total = sup + res
    if total <= 0:
        return "MISTA", 50.0
    sup_share = 100.0 * sup / total
    res_share = 100.0 * res / total
    if sup >= 2 and sup >= res * 1.25:
        return "SUPPORTO", sup_share
    if res >= 2 and res >= sup * 1.25:
        return "RESISTENZA", res_share
    return "MISTA", max(sup_share, res_share)


def _overlaps_zone(row: pd.Series, z: BalanceZone) -> bool:
    return float(row["low"]) <= z.center + z.half and float(row["high"]) >= z.center - z.half


def _distance_to_zone_atr(close: float, z: BalanceZone, atr: float) -> float:
    if not math.isfinite(atr) or atr <= 0:
        return math.nan
    bottom, top = z.center - z.half, z.center + z.half
    if bottom <= close <= top:
        return 0.0
    if close > top:
        return (close - top) / atr
    return (bottom - close) / atr


def _recent_touch_episode(data: pd.DataFrame, z: BalanceZone, max_age: int = 14, allowed_gap: int = 1) -> dict[str, int] | None:
    current = len(data) - 1
    start_scan = max(0, current - max_age)
    touch_positions = [p for p in range(start_scan, current + 1) if _overlaps_zone(data.iloc[p], z)]
    if not touch_positions:
        return None
    latest = touch_positions[-1]
    start = latest
    gap = 0
    for p in range(latest - 1, start_scan - 1, -1):
        if _overlaps_zone(data.iloc[p], z):
            start = p
            gap = 0
        else:
            gap += 1
            if gap > allowed_gap:
                break
    touches = sum(1 for p in range(start, current + 1) if _overlaps_zone(data.iloc[p], z))
    return {
        "first": start,
        "latest": latest,
        "bars_from_first": current - start,
        "bars_since_last": current - latest,
        "touch_bars": touches,
    }


def _arrival_direction(data: pd.DataFrame, z: BalanceZone, atr: float, first_touch: int | None = None) -> tuple[str, float]:
    """Direction into the Balance using price only. LONG means price arrived from above/downward."""
    current = len(data) - 1
    anchor = current if first_touch is None else first_touch
    ref = max(0, anchor - 6)
    if anchor <= ref or not math.isfinite(atr) or atr <= 0:
        return "NESSUNA", 0.0
    pre_close = float(data.iloc[ref]["close"])
    anchor_close = float(data.iloc[anchor]["close"])
    top, bottom = z.center + z.half, z.center - z.half
    move_atr = (anchor_close - pre_close) / atr

    long_score = max(0.0, (pre_close - top) / atr) + max(0.0, -move_atr)
    short_score = max(0.0, (bottom - pre_close) / atr) + max(0.0, move_atr)

    if long_score >= max(0.25, short_score * 1.15):
        return "LONG", abs(move_atr)
    if short_score >= max(0.25, long_score * 1.15):
        return "SHORT", abs(move_atr)
    return "NESSUNA", abs(move_atr)


def _role_alignment_score(z: BalanceZone, direction: str) -> tuple[str, float]:
    role, dominance = _zone_historical_role(z)
    if role == "MISTA":
        return role, 60.0
    aligned = (direction == "LONG" and role == "SUPPORTO") or (direction == "SHORT" and role == "RESISTENZA")
    return role, min(100.0, dominance) if aligned else 35.0


def analyze_zone_opportunity(
    data: pd.DataFrame,
    balance: dict[str, Any],
    z: BalanceZone,
    approach_max_atr: float = 0.45,
    already_moved_atr: float = 1.00,
) -> dict[str, Any]:
    atr = float(balance.get("atr14", math.nan))
    close = float(balance.get("close", math.nan))
    current = len(data) - 1
    bottom, top = z.center - z.half, z.center + z.half
    dist_zone = _distance_to_zone_atr(close, z, atr)
    episode = _recent_touch_episode(data, z)

    if episode is None:
        direction, arrival_atr = _arrival_direction(data, z, atr, None)
        moving_toward = (
            direction == "LONG" and close > top
        ) or (
            direction == "SHORT" and close < bottom
        )
        if direction == "NESSUNA" or not moving_toward or not math.isfinite(dist_zone) or dist_zone > approach_max_atr:
            return {"active": False, "phase": "NESSUNA", "direction": "", "score": 0.0}
        phase = "IN AVVICINAMENTO"
        bars_from_first = math.nan
        bars_since_last = math.nan
        touch_bars = 0
        move_from_zone_atr = 0.0
        local_confirm = False
    else:
        direction, arrival_atr = _arrival_direction(data, z, atr, episode["first"])
        if direction == "NESSUNA":
            # Fallback: position before the current touch episode tells us from which side price arrived.
            pre_idx = max(0, episode["first"] - 4)
            pre_close = float(data.iloc[pre_idx]["close"])
            if pre_close > top:
                direction = "LONG"
            elif pre_close < bottom:
                direction = "SHORT"
        if direction == "NESSUNA":
            return {"active": False, "phase": "NESSUNA", "direction": "", "score": 0.0}

        bars_from_first = episode["bars_from_first"]
        bars_since_last = episode["bars_since_last"]
        touch_bars = episode["touch_bars"]
        current_overlap = _overlaps_zone(data.iloc[current], z)
        current_away = max(0.0, (close - top) / atr) if direction == "LONG" else max(0.0, (bottom - close) / atr)
        move_from_zone_atr = current_away

        if direction == "LONG" and close < bottom - 0.20 * atr:
            return {"active": False, "phase": "INVALIDATA", "direction": direction, "score": 0.0}
        if direction == "SHORT" and close > top + 0.20 * atr:
            return {"active": False, "phase": "INVALIDATA", "direction": direction, "score": 0.0}

        prior_slice = data.iloc[max(episode["first"], current - 3):current]
        if direction == "LONG":
            prior_level = float(prior_slice["high"].max()) if not prior_slice.empty else math.inf
            local_confirm = close > prior_level
            two_closes = current >= 1 and close > top and float(data.iloc[current - 1]["close"]) > top
        else:
            prior_level = float(prior_slice["low"].min()) if not prior_slice.empty else -math.inf
            local_confirm = close < prior_level
            two_closes = current >= 1 and close < bottom and float(data.iloc[current - 1]["close"]) < bottom

        if current_away > already_moved_atr:
            phase = "GIÀ PARTITA"
        elif current_overlap or (math.isfinite(dist_zone) and dist_zone <= 0.08):
            phase = "IN TEST"
        elif current_away > 0:
            if local_confirm or (two_closes and current_away >= 0.10):
                phase = "CONFERMATA"
            else:
                phase = "REAZIONE"
        else:
            phase = "IN TEST"

    role, role_score = _role_alignment_score(z, direction)
    reliability = float(z.reliability) if not pd.isna(z.reliability) else 50.0
    strength = float(z.strength)
    stage_base = {
        "CONFERMATA": 100.0,
        "REAZIONE": 94.0,
        "IN TEST": 90.0,
        "IN AVVICINAMENTO": 78.0,
        "GIÀ PARTITA": 35.0,
    }.get(phase, 0.0)
    freshness = 100.0 if episode is None else 100.0 * math.exp(-float(bars_since_last) / 5.0)
    arrival_score = min(100.0, 100.0 * float(arrival_atr) / 1.5)
    score = (
        0.40 * stage_base
        + 0.20 * strength
        + 0.15 * reliability
        + 0.10 * arrival_score
        + 0.10 * freshness
        + 0.05 * role_score
    )
    score = round(max(0.0, min(100.0, score)), 1)

    signal = {
        ("LONG", "IN AVVICINAMENTO"): "WATCH LONG",
        ("LONG", "IN TEST"): "WATCH LONG",
        ("LONG", "REAZIONE"): "REAZIONE LONG",
        ("LONG", "CONFERMATA"): "TRIGGER LONG",
        ("LONG", "GIÀ PARTITA"): "GIÀ PARTITA LONG",
        ("SHORT", "IN AVVICINAMENTO"): "WATCH SHORT",
        ("SHORT", "IN TEST"): "WATCH SHORT",
        ("SHORT", "REAZIONE"): "REAZIONE SHORT",
        ("SHORT", "CONFERMATA"): "TRIGGER SHORT",
        ("SHORT", "GIÀ PARTITA"): "GIÀ PARTITA SHORT",
    }.get((direction, phase), "NESSUNA OCCASIONE")

    return {
        "active": phase in {"IN AVVICINAMENTO", "IN TEST", "REAZIONE", "CONFERMATA"},
        "phase": phase,
        "direction": direction,
        "signal": signal,
        "score": score,
        "zone": z,
        "historical_role": role,
        "distance_zone_atr": dist_zone,
        "arrival_atr": float(arrival_atr),
        "bars_from_first": bars_from_first,
        "bars_since_last": bars_since_last,
        "touch_bars": touch_bars,
        "move_from_zone_atr": move_from_zone_atr,
        "local_confirm": bool(local_confirm),
    }


def select_opportunity(data: pd.DataFrame, balance: dict[str, Any]) -> dict[str, Any]:
    zones: list[BalanceZone] = list(balance.get("zones", []))
    candidates: list[dict[str, Any]] = []
    for z in zones:
        candidate = analyze_zone_opportunity(data, balance, z)
        if candidate.get("phase") not in {"NESSUNA", "INVALIDATA"}:
            candidates.append(candidate)
    if candidates:
        phase_rank = {"CONFERMATA": 5, "REAZIONE": 4, "IN TEST": 3, "IN AVVICINAMENTO": 2, "GIÀ PARTITA": 1}
        return max(candidates, key=lambda x: (phase_rank.get(str(x.get("phase")), 0), float(x.get("score", 0.0))))

    # No setup: retain the nearest zone so the full-market sheet remains informative.
    close = float(balance.get("close", math.nan))
    atr = float(balance.get("atr14", math.nan))
    if zones:
        z = min(zones, key=lambda q: _distance_to_zone_atr(close, q, atr))
        role, _ = _zone_historical_role(z)
        return {
            "active": False, "phase": "NESSUNA", "direction": "", "signal": "NESSUNA OCCASIONE", "score": 0.0,
            "zone": z, "historical_role": role, "distance_zone_atr": _distance_to_zone_atr(close, z, atr),
            "arrival_atr": math.nan, "bars_from_first": math.nan, "bars_since_last": math.nan,
            "touch_bars": 0, "move_from_zone_atr": math.nan, "local_confirm": False,
        }
    return {"active": False, "phase": "NESSUNA", "direction": "", "signal": "NESSUNA OCCASIONE", "score": 0.0}


def result_row(label: str, ticker: str, timeframe: str, data: pd.DataFrame, balance: dict[str, Any], data_note: str) -> dict[str, Any]:
    opp = select_opportunity(data, balance)
    z: BalanceZone | None = opp.get("zone")
    close = float(balance.get("close", math.nan))
    if z is None:
        zone_center = zone_low = zone_high = strength = reliability = math.nan
        tests = 0
        role = "N/D"
    else:
        zone_center = z.center
        zone_low = z.center - z.half
        zone_high = z.center + z.half
        strength = z.strength
        reliability = z.reliability
        tests = z.independent_tests
        role = str(opp.get("historical_role", "MISTA"))

    arrival_text = "RIBASSISTA" if opp.get("direction") == "LONG" else "RIALZISTA" if opp.get("direction") == "SHORT" else "—"
    return {
        "Strumento": label,
        "Ticker": ticker,
        "TF": timeframe,
        "Segnale": opp.get("signal", "NESSUNA OCCASIONE"),
        "Fase": opp.get("phase", "NESSUNA"),
        "Direzione": opp.get("direction", ""),
        "Score priorità": float(opp.get("score", 0.0)),
        "Prezzo": close,
        "Balance": zone_center,
        "Zona min": zone_low,
        "Zona max": zone_high,
        "Ruolo storico": role,
        "Distanza zona ATR": opp.get("distance_zone_atr", math.nan),
        "Arrivo": arrival_text,
        "Forza arrivo ATR": opp.get("arrival_atr", math.nan),
        "Barre dal primo test": opp.get("bars_from_first", math.nan),
        "Barre in zona": opp.get("touch_bars", 0),
        "Ultimo test barre fa": opp.get("bars_since_last", math.nan),
        "Movimento dalla zona ATR": opp.get("move_from_zone_atr", math.nan),
        "Conferma locale": "SI" if opp.get("local_confirm") else "NO",
        "Strength": strength,
        "Reliability %": reliability,
        "Test indipendenti": tests,
        "Barra riferimento": balance.get("reference_date", ""),
        "Nota dati": data_note,
        "_active": bool(opp.get("active", False)),
    }


def plot_balance(data: pd.DataFrame, balance: dict[str, Any], ticker: str, timeframe: str, row: pd.Series | None = None) -> go.Figure:
    bars = 180 if timeframe == "Daily" else 260
    chart = data.tail(bars)
    fig = go.Figure(data=[go.Candlestick(
        x=chart.index, open=chart["open"], high=chart["high"], low=chart["low"], close=chart["close"], name=ticker
    )])
    selected_balance = float(row.get("Balance", math.nan)) if row is not None else math.nan
    for z in balance.get("zones", []):
        role, _ = _zone_historical_role(z)
        annotation = f"{role} {z.center:.4f}"
        opacity = 0.20 if math.isfinite(selected_balance) and abs(z.center - selected_balance) <= max(1e-12, z.half) else 0.08
        if opacity > 0.10:
            annotation += " · SETUP"
        fig.add_hrect(y0=z.center-z.half, y1=z.center+z.half, opacity=opacity, line_width=0,
                      annotation_text=annotation, annotation_position="top right")
        fig.add_hline(y=z.center, line_dash="dot", line_width=1)
    title_suffix = ""
    if row is not None:
        title_suffix = f" · {row.get('Segnale', '')} · {row.get('Fase', '')}"
    fig.update_layout(
        title=f"{ticker} · {timeframe}{title_suffix}", height=620,
        margin=dict(l=20, r=20, t=60, b=20), xaxis_rangeslider_visible=False,
    )
    return fig


def _excel_bytes(opportunities: pd.DataFrame, full: pd.DataFrame, errors: list[dict[str, str]]) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        opportunities.to_excel(writer, sheet_name="Opportunità", index=False)
        full.to_excel(writer, sheet_name="Tutti i mercati", index=False)
        if errors:
            pd.DataFrame(errors).to_excel(writer, sheet_name="Errori", index=False)

        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#1F4E78", "font_color": "#FFFFFF", "border": 1, "align": "center", "valign": "vcenter"})
        long_fmt = workbook.add_format({"bg_color": "#E2F0D9"})
        short_fmt = workbook.add_format({"bg_color": "#FCE4D6"})
        test_fmt = workbook.add_format({"bg_color": "#FFF2CC"})
        num4_fmt = workbook.add_format({"num_format": "0.0000"})
        num2_fmt = workbook.add_format({"num_format": "0.00"})

        for sheet_name, df in [("Opportunità", opportunities), ("Tutti i mercati", full)]:
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, max(0, len(df)), max(0, len(df.columns) - 1))
            ws.set_row(0, 24, header_fmt)
            for col_idx, col in enumerate(df.columns):
                width = min(34, max(11, len(str(col)) + 2))
                if col in {"Segnale", "Fase", "Strumento", "Ticker", "Ruolo storico"}:
                    width = max(width, 17)
                fmt = None
                if col in {"Prezzo", "Balance", "Zona min", "Zona max"}:
                    fmt = num4_fmt
                elif "ATR" in col or col in {"Score priorità", "Strength", "Reliability %"}:
                    fmt = num2_fmt
                ws.set_column(col_idx, col_idx, width, fmt)
            if len(df) > 0 and "Direzione" in df.columns:
                dcol = df.columns.get_loc("Direzione")
                fcol = df.columns.get_loc("Fase") if "Fase" in df.columns else dcol
                ws.conditional_format(1, 0, len(df), len(df.columns)-1, {"type": "formula", "criteria": f'=${chr(65+dcol)}2="LONG"', "format": long_fmt}) if dcol < 26 else None
                ws.conditional_format(1, 0, len(df), len(df.columns)-1, {"type": "formula", "criteria": f'=${chr(65+dcol)}2="SHORT"', "format": short_fmt}) if dcol < 26 else None
                ws.conditional_format(1, 0, len(df), len(df.columns)-1, {"type": "formula", "criteria": f'=${chr(65+fcol)}2="IN TEST"', "format": test_fmt}) if fcol < 26 else None
    return out.getvalue()


# =============================================================================
# UI
# =============================================================================
def default_universe() -> str:
    return """# LABEL;TICKER Yahoo Finance\nES;ES=F\nNQ;NQ=F\nYM;YM=F\nRTY;RTY=F\nGold;GC=F\nSilver;SI=F\nCopper;HG=F\nWTI;CL=F\nNatGas;NG=F\nEuro FX;6E=F\nBritish Pound;6B=F\nJapanese Yen;6J=F\n"""


with st.sidebar:
    st.header("Impostazioni")
    timeframe = st.selectbox("Timeframe", ["Daily", "H4", "H1"], index=0)
    lookback = st.number_input("Lookback Balance (barre)", min_value=120, max_value=1000, value=500, step=50)
    view_mode = st.selectbox("Visualizza", ["Solo opportunità", "Opportunità + già partite", "Tutti i mercati"], index=0)
    min_score = st.slider("Score priorità minimo", 0, 100, 45, 1)
    st.divider()
    uploaded = st.file_uploader("File ticker .txt", type=["txt", "csv"])
    use_manual = st.checkbox("Modifica/incolla ticker manualmente", value=uploaded is None)
    manual_text = st.text_area("Ticker", default_universe(), height=260, disabled=not use_manual)
    run = st.button("🔎 Avvia screener", type="primary", use_container_width=True)

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
        bar.progress((i - 1) / len(universe), text=f"{ticker} · analisi Balance")
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
    st.session_state["balance_scan_v2"] = {
        "rows": rows, "details": details, "errors": errors, "timeframe": timeframe, "lookback": int(lookback)
    }

payload = st.session_state.get("balance_scan_v2")

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

    full = pd.DataFrame(rows).sort_values(["Score priorità", "Distanza zona ATR"], ascending=[False, True], na_position="last").reset_index(drop=True)
    opportunities = full[full["_active"]].copy().reset_index(drop=True)
    started = full[full["Fase"] == "GIÀ PARTITA"].copy().reset_index(drop=True)

    if view_mode == "Solo opportunità":
        filtered = opportunities.copy()
    elif view_mode == "Opportunità + già partite":
        filtered = pd.concat([opportunities, started], ignore_index=True)
    else:
        filtered = full.copy()

    filtered = filtered[filtered["Score priorità"] >= min_score].copy().reset_index(drop=True)
    filtered.insert(0, "Rank", range(1, len(filtered) + 1))

    if scan_timeframe != timeframe:
        st.warning(f"I risultati visualizzati sono del timeframe {scan_timeframe}. Premi ‘Avvia screener’ per ricalcolare {timeframe}.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strumenti analizzati", len(full))
    c2.metric("Opportunità", len(opportunities))
    c3.metric("In test", int((full["Fase"] == "IN TEST").sum()))
    c4.metric("Trigger", int((full["Fase"] == "CONFERMATA").sum()))

    st.subheader("Opportunità Balance")
    visible_cols = [
        "Rank", "Strumento", "Ticker", "TF", "Segnale", "Fase", "Score priorità", "Prezzo", "Balance",
        "Zona min", "Zona max", "Distanza zona ATR", "Arrivo", "Forza arrivo ATR", "Barre dal primo test",
        "Barre in zona", "Ultimo test barre fa", "Movimento dalla zona ATR", "Conferma locale",
        "Ruolo storico", "Strength", "Reliability %", "Test indipendenti", "Barra riferimento",
    ]
    st.dataframe(
        filtered[visible_cols], hide_index=True, use_container_width=True,
        column_config={
            "Score priorità": st.column_config.ProgressColumn("Score priorità", min_value=0, max_value=100, format="%.1f"),
            "Distanza zona ATR": st.column_config.NumberColumn("Dist. zona ATR", format="%.2f"),
            "Forza arrivo ATR": st.column_config.NumberColumn("Forza arrivo ATR", format="%.2f"),
            "Movimento dalla zona ATR": st.column_config.NumberColumn("Mov. zona ATR", format="%.2f"),
            "Strength": st.column_config.NumberColumn("Strength", format="%.1f"),
            "Reliability %": st.column_config.NumberColumn("Reliability %", format="%.1f"),
            "Prezzo": st.column_config.NumberColumn("Prezzo", format="%.4f"),
            "Balance": st.column_config.NumberColumn("Balance", format="%.4f"),
            "Zona min": st.column_config.NumberColumn("Zona min", format="%.4f"),
            "Zona max": st.column_config.NumberColumn("Zona max", format="%.4f"),
        },
    )

    export_cols = [c for c in visible_cols if c != "Rank"] + ["Nota dati"]
    opp_export = opportunities[export_cols].copy()
    full_export = full[export_cols].copy()
    xlsx = _excel_bytes(opp_export, full_export, errors)
    st.download_button(
        "⬇️ Esporta Excel",
        data=xlsx,
        file_name=f"balance_opportunity_screener_{scan_timeframe}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    if not filtered.empty:
        st.subheader("Verifica grafica")
        selected_ticker = st.selectbox("Strumento da visualizzare", filtered["Ticker"].tolist(), key="balance_chart_ticker_v2")
        data, balance, label = details[selected_ticker]
        selected_row = filtered[filtered["Ticker"] == selected_ticker].iloc[0]
        st.plotly_chart(plot_balance(data, balance, selected_ticker, scan_timeframe, selected_row), use_container_width=True)

        a, b, c, d, e = st.columns(5)
        a.metric("Segnale", str(selected_row["Segnale"]))
        b.metric("Fase", str(selected_row["Fase"]))
        c.metric("Score", f"{float(selected_row['Score priorità']):.1f}")
        d.metric("Dist. zona", "—" if pd.isna(selected_row["Distanza zona ATR"]) else f"{float(selected_row['Distanza zona ATR']):.2f} ATR")
        e.metric("Barre in zona", int(selected_row["Barre in zona"]) if not pd.isna(selected_row["Barre in zona"]) else 0)

    if errors:
        with st.expander(f"Ticker senza risultato ({len(errors)})"):
            st.dataframe(pd.DataFrame(errors), hide_index=True, use_container_width=True)
