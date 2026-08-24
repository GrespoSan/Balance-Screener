from __future__ import annotations

from dataclasses import dataclass
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
st.set_page_config(page_title="G. Balance Alignment Screener V3", page_icon="⚖️", layout="wide")
st.title("⚖️ G. Balance Alignment Screener V3")


# =============================================================================
# MODEL
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
# INPUT / DATA
# =============================================================================
def parse_tickers(text: str) -> list[tuple[str, str]]:
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


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = [str(c[0]).lower() for c in x.columns]
    else:
        x.columns = [str(c).lower() for c in x.columns]
    if "adj close" in x.columns:
        x = x.rename(columns={"adj close": "adj_close"})
    need = ["open", "high", "low", "close"]
    if any(c not in x.columns for c in need):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if "volume" not in x.columns:
        x["volume"] = np.nan
    x = x[["open", "high", "low", "close", "volume"]].copy()
    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["open", "high", "low", "close"])
    x = x[~x.index.duplicated(keep="last")].sort_index()
    return x


def _drop_open_bar(df: pd.DataFrame, timeframe: str) -> tuple[pd.DataFrame, bool]:
    if df.empty or len(df) < 2:
        return df, False
    x = df.copy()
    last = pd.Timestamp(x.index[-1])
    try:
        now = pd.Timestamp.now(tz=last.tz) if last.tz is not None else pd.Timestamp.now()
    except Exception:
        now = pd.Timestamp.now()

    if timeframe == "Daily":
        last_day = last.tz_localize(None).date() if last.tz is not None else last.date()
        now_day = now.tz_localize(None).date() if getattr(now, "tz", None) is not None else now.date()
        remove = last_day >= now_day
    else:
        delta = pd.Timedelta(hours=1 if timeframe == "H1" else 4)
        try:
            remove = last + delta > now
        except TypeError:
            remove = last.tz_localize(None) + delta > now.tz_localize(None)
    return (x.iloc[:-1].copy(), True) if remove else (x, False)


def _resample_h4(h1: pd.DataFrame) -> pd.DataFrame:
    if h1.empty:
        return h1
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
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
# BALANCE ENGINE V3 — Pine defaults v0.5.1.9
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
    atr.iloc[length - 1] = float(tr.iloc[:length].mean())
    for i in range(length, len(tr)):
        atr.iloc[i] = (float(atr.iloc[i - 1]) * (length - 1) + float(tr.iloc[i])) / length
    return atr


def _eval_compatible(
    data: pd.DataFrame,
    center: float,
    half: float,
    scan_limit: int,
    validation_bars: int = 10,
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
        if not (float(row["low"]) <= top and float(row["high"]) >= bottom):
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


def _eval_independent(
    data: pd.DataFrame,
    atr: pd.Series,
    center: float,
    half: float,
    scan_limit: int,
    validation_bars: int = 10,
    cooldown_bars: int = 6,
    min_reaction_atr: float = 0.20,
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
        prev_pos = pos - 1
        if prev_pos < 0:
            continue
        row = data.iloc[pos]
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


def _make_zone(
    data: pd.DataFrame,
    atr: pd.Series,
    center: float,
    half: float,
    scan_limit: int,
    range_low: float,
    active_range: float,
) -> BalanceZone:
    sup, res, hits, dwell, age, strength = _eval_compatible(data, center, half, scan_limit)
    tests, succ, ind_sup, ind_res, brk, rel = _eval_independent(data, atr, center, half, scan_limit)
    return BalanceZone(
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
    )


def _calculate_snapshot(
    data: pd.DataFrame,
    previous_zones: list[BalanceZone] | None = None,
    lookback: int = 400,
    scan_step_pct: float = 1.0,
    max_zones: int = 9,
    zone_half_atr: float = 0.12,
    min_spacing_range_pct: float = 8.0,
    retention_strength: float = 25.0,
    retention_tolerance_steps: float = 1.5,
    use_retention: bool = True,
) -> dict[str, Any]:
    unavailable = {"available": False, "zones": [], "detail": "Dati insufficienti."}
    if data.empty or len(data) < max(60, min(lookback, 120)):
        return unavailable

    d = data.copy().dropna(subset=["high", "low", "close"])
    atr = wilder_atr(d, 14)
    atr_now = float(atr.iloc[-1]) if len(atr) and not pd.isna(atr.iloc[-1]) else math.nan
    if not math.isfinite(atr_now) or atr_now <= 0:
        return unavailable

    scan_limit = min(int(lookback), len(d) - 1)
    window = d.iloc[-min(int(lookback), len(d)):]
    range_high = float(window["high"].max())
    range_low = float(window["low"].min())
    active_range = range_high - range_low
    if not math.isfinite(active_range) or active_range <= 0:
        return unavailable

    half = max(1e-12, atr_now * float(zone_half_atr))
    candidates: list[tuple[float, float, int]] = []
    steps = int(math.floor(100.0 / float(scan_step_pct)))
    for s in range(steps + 1):
        pct = min(100.0, s * float(scan_step_pct))
        center = range_low + active_range * pct / 100.0
        sup, res, hits, dwell, age, strength = _eval_compatible(d, center, half, scan_limit)
        if hits >= 1:
            candidates.append((center, strength, hits))
    if not candidates:
        return {**unavailable, "detail": "Nessuna Balance qualificata."}

    spacing = max(1e-12, active_range * float(min_spacing_range_pct) / 100.0)
    retention_tolerance = active_range * float(scan_step_pct) / 100.0 * float(retention_tolerance_steps)
    selected: list[BalanceZone] = []
    used: set[int] = set()

    # Pine v0.5.1.9: first retain previous zones when a current candidate remains
    # sufficiently close and strong, then fill the remaining slots by strength.
    if use_retention and previous_zones:
        for old in previous_zones:
            if len(selected) >= int(max_zones):
                break
            best_idx = None
            best_dist = math.inf
            for idx, (center, strength, _) in enumerate(candidates):
                if idx in used:
                    continue
                dist = abs(center - old.center)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx is None:
                continue
            center, strength, _ = candidates[best_idx]
            if best_dist <= retention_tolerance and strength >= float(retention_strength):
                if not any(abs(center - z.center) < spacing for z in selected):
                    z = _make_zone(d, atr, center, half, scan_limit, range_low, active_range)
                    if z.hits >= 1:
                        selected.append(z)
                        used.add(best_idx)

    order = sorted(range(len(candidates)), key=lambda i: candidates[i][1], reverse=True)
    for idx in order:
        if len(selected) >= int(max_zones):
            break
        if idx in used:
            continue
        center, _, _ = candidates[idx]
        if any(abs(center - z.center) < spacing for z in selected):
            continue
        z = _make_zone(d, atr, center, half, scan_limit, range_low, active_range)
        if z.hits >= 1:
            selected.append(z)
            used.add(idx)

    selected.sort(key=lambda z: z.center)
    return {
        "available": bool(selected),
        "zones": selected,
        "atr14_at_calculation": atr_now,
        "range_high": range_high,
        "range_low": range_low,
        "active_range": active_range,
        "calculation_date": d.index[-1],
        "detail": f"{len(selected)} Balance | lookback {scan_limit}",
    }


def analyze_balance_v3(
    data: pd.DataFrame,
    mode: str,
    lookback: int = 400,
    update_frequency: int = 50,
    freeze_age: int = 0,
) -> dict[str, Any]:
    """Return current Balance snapshot.

    Snapshot corrente = current recalculation, deterministic and comparable after a Pine reload.
    Retention simulata = replay periodic recalculations every update_frequency bars ending
    freeze_age bars before the latest closed bar, preserving previous zones between updates.
    """
    unavailable = {"available": False, "zones": [], "detail": "Dati insufficienti."}
    if data.empty or len(data) < 120:
        return unavailable

    latest_pos = len(data) - 1
    calc_pos = latest_pos - int(freeze_age)
    if calc_pos < 119:
        return unavailable

    if mode == "Snapshot corrente":
        snap = _calculate_snapshot(data.iloc[:calc_pos + 1], previous_zones=None, lookback=lookback, use_retention=False)
    else:
        # Deterministic replay aligned to the chosen final calculation bar. Starting one
        # lookback before the first usable snapshot is enough to build a stable history
        # without processing the entire 5-year series on every scan.
        first_pos = max(119, calc_pos - max(lookback * 2, update_frequency * 10))
        positions = list(range(first_pos, calc_pos + 1, max(1, int(update_frequency))))
        if not positions or positions[-1] != calc_pos:
            positions.append(calc_pos)
        prev: list[BalanceZone] = []
        snap = unavailable
        for pos in positions:
            snap = _calculate_snapshot(
                data.iloc[:pos + 1],
                previous_zones=prev,
                lookback=lookback,
                use_retention=True,
            )
            if snap.get("available"):
                prev = list(snap.get("zones", []))

    if not snap.get("available"):
        return snap

    current_close = float(data.iloc[-1]["close"])
    current_atr = wilder_atr(data, 14)
    atr_now = float(current_atr.iloc[-1]) if len(current_atr) and not pd.isna(current_atr.iloc[-1]) else math.nan
    snap.update({
        "close": current_close,
        "atr14": atr_now,
        "reference_date": pd.Timestamp(data.index[-1]),
        "calculation_age_bars": latest_pos - calc_pos,
        "mode": mode,
    })
    return snap


def balance_role(z: BalanceZone, ref_close: float) -> str:
    inside = z.center - z.half <= ref_close <= z.center + z.half
    if inside:
        return "BALANCE"
    enough_ind = z.independent_tests >= 2 and not pd.isna(z.reliability) and z.reliability >= 35.0
    if enough_ind:
        support_dom = z.independent_support_success >= 2 and z.independent_support_success >= max(1.0, z.independent_resistance_success * 1.35)
        resistance_dom = z.independent_resistance_success >= 2 and z.independent_resistance_success >= max(1.0, z.independent_support_success * 1.35)
        if z.center < ref_close and support_dom:
            return "SUPPORTO"
        if z.center > ref_close and resistance_dom:
            return "RESISTENZA"
    comp_sup = z.support_hits >= 2 and z.support_hits >= max(1.0, z.resistance_hits * 1.35)
    comp_res = z.resistance_hits >= 2 and z.resistance_hits >= max(1.0, z.support_hits * 1.35)
    if z.center < ref_close and (comp_sup or z.hits >= 2):
        return "SUPPORTO"
    if z.center > ref_close and (comp_res or z.hits >= 2):
        return "RESISTENZA"
    return "BALANCE"


# =============================================================================
# OUTPUT TABLES / CHART
# =============================================================================
def _nearest_levels(zones: list[BalanceZone], close: float, count: int = 3) -> tuple[list[float], list[float]]:
    below = sorted([z.center for z in zones if z.center < close], reverse=True)[:count]
    above = sorted([z.center for z in zones if z.center > close])[:count]
    below += [math.nan] * (count - len(below))
    above += [math.nan] * (count - len(above))
    return below, above


def summary_row(label: str, ticker: str, timeframe: str, balance: dict[str, Any], note: str) -> dict[str, Any]:
    zones: list[BalanceZone] = list(balance.get("zones", []))
    close = float(balance.get("close", math.nan))
    below, above = _nearest_levels(zones, close, 3)
    inside = next((z for z in zones if z.center - z.half <= close <= z.center + z.half), None)
    nearest = min(zones, key=lambda z: abs(z.center - close)) if zones else None
    atr = float(balance.get("atr14", math.nan))
    dist_atr = abs(close - nearest.center) / atr if nearest is not None and math.isfinite(atr) and atr > 0 else math.nan
    calc_date = balance.get("calculation_date")
    ref_date = balance.get("reference_date")
    return {
        "Strumento": label,
        "Ticker": ticker,
        "TF": timeframe,
        "Prezzo": close,
        "Supporto 1": below[0],
        "Supporto 2": below[1],
        "Supporto 3": below[2],
        "Resistenza 1": above[0],
        "Resistenza 2": above[1],
        "Resistenza 3": above[2],
        "Dentro Balance": "SI" if inside is not None else "NO",
        "Balance corrente": inside.center if inside is not None else math.nan,
        "Balance più vicina": nearest.center if nearest is not None else math.nan,
        "Dist. centro ATR": dist_atr,
        "N. Balance": len(zones),
        "Data livelli": pd.Timestamp(calc_date).strftime("%Y-%m-%d %H:%M") if calc_date is not None else "",
        "Età livelli barre": int(balance.get("calculation_age_bars", 0)),
        "Barra prezzo": pd.Timestamp(ref_date).strftime("%Y-%m-%d %H:%M") if ref_date is not None else "",
        "Modalità": balance.get("mode", ""),
        "Nota dati": note,
    }


def zone_rows(label: str, ticker: str, timeframe: str, balance: dict[str, Any]) -> list[dict[str, Any]]:
    close = float(balance.get("close", math.nan))
    atr = float(balance.get("atr14", math.nan))
    rows: list[dict[str, Any]] = []
    for i, z in enumerate(balance.get("zones", []), start=1):
        role = balance_role(z, close)
        dist = abs(close - z.center) / atr if math.isfinite(atr) and atr > 0 else math.nan
        rows.append({
            "Strumento": label,
            "Ticker": ticker,
            "TF": timeframe,
            "Ordine prezzo": i,
            "Ruolo": role,
            "Centro": z.center,
            "Zona min": z.center - z.half,
            "Zona max": z.center + z.half,
            "Distanza ATR": dist,
            "Strength": z.strength,
            "Hits": z.hits,
            "Support hits": z.support_hits,
            "Resistance hits": z.resistance_hits,
            "Dwell": z.dwell,
            "Ultimo hit barre fa": z.last_hit_age,
            "Test indipendenti": z.independent_tests,
            "Successi indipendenti": z.independent_successes,
            "Break indipendenti": z.independent_breaks,
            "Reliability %": z.reliability,
        })
    return rows


def plot_balance(data: pd.DataFrame, balance: dict[str, Any], ticker: str, timeframe: str) -> go.Figure:
    bars = 260 if timeframe == "Daily" else 360
    chart = data.tail(bars)
    close = float(balance.get("close", math.nan))
    fig = go.Figure(data=[go.Candlestick(
        x=chart.index,
        open=chart["open"], high=chart["high"], low=chart["low"], close=chart["close"],
        name=ticker,
    )])
    for z in balance.get("zones", []):
        role = balance_role(z, close)
        if role == "SUPPORTO":
            fill = "rgba(0,170,110,0.16)"
            line = "rgba(0,150,95,0.95)"
        elif role == "RESISTENZA":
            fill = "rgba(255,120,40,0.15)"
            line = "rgba(255,105,35,0.95)"
        else:
            fill = "rgba(80,150,255,0.13)"
            line = "rgba(70,135,235,0.90)"
        fig.add_hrect(y0=z.center - z.half, y1=z.center + z.half, fillcolor=fill, line_width=0)
        fig.add_hline(y=z.center, line_width=1, line_color=line)
    fig.update_layout(
        title=f"{ticker} · {timeframe} · Balance V3",
        height=650,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig


def excel_bytes(summary: pd.DataFrame, zones: pd.DataFrame, errors: list[dict[str, str]]) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        summary.to_excel(writer, sheet_name="Riepilogo", index=False)
        zones.to_excel(writer, sheet_name="Zone Balance", index=False)
        if errors:
            pd.DataFrame(errors).to_excel(writer, sheet_name="Errori", index=False)
        wb = writer.book
        head = wb.add_format({"bold": True, "bg_color": "#1F4E78", "font_color": "#FFFFFF", "border": 1, "align": "center"})
        n4 = wb.add_format({"num_format": "0.0000"})
        n2 = wb.add_format({"num_format": "0.00"})
        for name, df in [("Riepilogo", summary), ("Zone Balance", zones)]:
            ws = writer.sheets[name]
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, max(0, len(df)), max(0, len(df.columns) - 1))
            ws.set_row(0, 24, head)
            for j, col in enumerate(df.columns):
                width = min(30, max(12, len(str(col)) + 2))
                fmt = None
                if col in {"Prezzo", "Supporto 1", "Supporto 2", "Supporto 3", "Resistenza 1", "Resistenza 2", "Resistenza 3", "Balance corrente", "Balance più vicina", "Centro", "Zona min", "Zona max"}:
                    fmt = n4
                elif "ATR" in col or col in {"Strength", "Reliability %"}:
                    fmt = n2
                ws.set_column(j, j, width, fmt)
    return out.getvalue()


def default_universe() -> str:
    return """# LABEL;TICKER Yahoo Finance
British Pound;6B=F
RTY;RTY=F
YM;YM=F
WTI;CL=F
Silver;SI=F
NQ;NQ=F
ES;ES=F
Gold;GC=F
Copper;HG=F
Japanese Yen;6J=F
Euro FX;6E=F
NatGas;NG=F
"""


# =============================================================================
# UI
# =============================================================================
with st.sidebar:
    st.header("Impostazioni")
    timeframe = st.selectbox("Timeframe", ["Daily", "H4", "H1"], index=0)
    mode = st.selectbox("Motore zone", ["Snapshot corrente", "Retention simulata"], index=0)
    freeze_age = 0
    if mode == "Retention simulata":
        freeze_age = st.slider("Età ultimo aggiornamento zone (barre)", 0, 49, 0, 1)
    st.divider()
    uploaded = st.file_uploader("File ticker .txt", type=["txt", "csv"])
    use_manual = st.checkbox("Modifica/incolla ticker manualmente", value=uploaded is None)
    manual_text = st.text_area("Ticker", default_universe(), height=260, disabled=not use_manual)
    run = st.button("🔎 Calcola Balance", type="primary", use_container_width=True)

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
        st.error("Nessun ticker valido.")
        st.stop()

    summary_rows: list[dict[str, Any]] = []
    all_zone_rows: list[dict[str, Any]] = []
    details: dict[str, tuple[pd.DataFrame, dict[str, Any], str]] = {}
    errors: list[dict[str, str]] = []
    progress = st.progress(0.0, text="Avvio…")

    for i, (label, ticker) in enumerate(universe, start=1):
        progress.progress((i - 1) / len(universe), text=f"{ticker} · Balance")
        data, note = load_price_data(ticker, timeframe)
        if data.empty or len(data) < 120:
            errors.append({"Ticker": ticker, "Errore": note or "Dati insufficienti"})
            continue
        try:
            bal = analyze_balance_v3(data, mode=mode, lookback=400, update_frequency=50, freeze_age=int(freeze_age))
            if not bal.get("available"):
                errors.append({"Ticker": ticker, "Errore": str(bal.get("detail", "Balance non disponibili"))})
                continue
            summary_rows.append(summary_row(label, ticker, timeframe, bal, note))
            all_zone_rows.extend(zone_rows(label, ticker, timeframe, bal))
            details[ticker] = (data, bal, label)
        except Exception as exc:
            errors.append({"Ticker": ticker, "Errore": f"{type(exc).__name__}: {exc}"})

    progress.progress(1.0, text="Completato")
    st.session_state["balance_alignment_v3"] = {
        "summary": summary_rows,
        "zones": all_zone_rows,
        "details": details,
        "errors": errors,
        "timeframe": timeframe,
        "mode": mode,
        "freeze_age": int(freeze_age),
    }

payload = st.session_state.get("balance_alignment_v3")
if payload:
    summary = pd.DataFrame(payload["summary"])
    zones = pd.DataFrame(payload["zones"])
    details = payload["details"]
    errors = payload["errors"]
    scan_tf = payload["timeframe"]
    scan_mode = payload["mode"]

    if summary.empty:
        st.warning("Nessun risultato disponibile.")
        if errors:
            st.dataframe(pd.DataFrame(errors), hide_index=True, use_container_width=True)
        st.stop()

    if scan_tf != timeframe or scan_mode != mode:
        st.warning("Le impostazioni sono cambiate. Premi ‘Calcola Balance’ per aggiornare i risultati.")

    st.subheader("Allineamento Balance")
    display_cols = [
        "Strumento", "Ticker", "TF", "Prezzo",
        "Supporto 3", "Supporto 2", "Supporto 1",
        "Balance corrente",
        "Resistenza 1", "Resistenza 2", "Resistenza 3",
        "Dentro Balance", "Balance più vicina", "Dist. centro ATR",
        "N. Balance", "Data livelli", "Età livelli barre", "Modalità",
    ]
    st.dataframe(
        summary[display_cols],
        hide_index=True,
        use_container_width=True,
        column_config={
            "Prezzo": st.column_config.NumberColumn("Prezzo", format="%.4f"),
            "Supporto 1": st.column_config.NumberColumn("Supporto 1", format="%.4f"),
            "Supporto 2": st.column_config.NumberColumn("Supporto 2", format="%.4f"),
            "Supporto 3": st.column_config.NumberColumn("Supporto 3", format="%.4f"),
            "Resistenza 1": st.column_config.NumberColumn("Resistenza 1", format="%.4f"),
            "Resistenza 2": st.column_config.NumberColumn("Resistenza 2", format="%.4f"),
            "Resistenza 3": st.column_config.NumberColumn("Resistenza 3", format="%.4f"),
            "Balance corrente": st.column_config.NumberColumn("Balance corrente", format="%.4f"),
            "Balance più vicina": st.column_config.NumberColumn("Balance più vicina", format="%.4f"),
            "Dist. centro ATR": st.column_config.NumberColumn("Dist. centro ATR", format="%.2f"),
        },
    )

    xlsx = excel_bytes(summary, zones, errors)
    st.download_button(
        "⬇️ Esporta Excel",
        data=xlsx,
        file_name=f"balance_alignment_v3_{scan_tf}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.subheader("Verifica grafica")
    selected = st.selectbox("Strumento", summary["Ticker"].tolist(), key="balance_alignment_chart_v3")
    data, bal, label = details[selected]
    st.plotly_chart(plot_balance(data, bal, selected, scan_tf), use_container_width=True)

    zdf = zones[zones["Ticker"] == selected].copy().reset_index(drop=True)
    st.dataframe(
        zdf[["Ordine prezzo", "Ruolo", "Centro", "Zona min", "Zona max", "Distanza ATR", "Strength", "Hits", "Reliability %", "Test indipendenti"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "Centro": st.column_config.NumberColumn("Centro", format="%.4f"),
            "Zona min": st.column_config.NumberColumn("Zona min", format="%.4f"),
            "Zona max": st.column_config.NumberColumn("Zona max", format="%.4f"),
            "Distanza ATR": st.column_config.NumberColumn("Distanza ATR", format="%.2f"),
            "Strength": st.column_config.NumberColumn("Strength", format="%.1f"),
            "Reliability %": st.column_config.NumberColumn("Reliability %", format="%.1f"),
        },
    )

    if errors:
        with st.expander(f"Ticker senza risultato ({len(errors)})"):
            st.dataframe(pd.DataFrame(errors), hide_index=True, use_container_width=True)
