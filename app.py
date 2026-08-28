from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import io
import math
import re
import hashlib
import json
from datetime import time
from zoneinfo import ZoneInfo
from html.parser import HTMLParser
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =============================================================================
# PAGE
# =============================================================================
st.set_page_config(page_title="G. Balance Stock Screener", page_icon="🎯", layout="wide")
st.title("🎯 G. Balance Stock Screener")

LOOKBACK = 400
DEFAULT_INTERACTION_WINDOW = 3
DEFAULT_MIN_INTERACTION_BARS = 2

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
    return """# Ticker Yahoo Italia\n1HOOD.MI
1RHM.MI
1TSLA.MI
A2A.MI
ACE.MI
AMP.MI
ARIS.MI
AVIO.MI
AZM.MI
BAMI.MI
BC.MI
BDB.MI
BMED.MI
BMPS.MI
BPE.MI
BRE.MI
BST.MI
BZU.MI
CE.MI
CEM.MI
CIRC.MI
CPR.MI
DBA.MI
DIA.MI
DIB.MI
ELE.MI
ELN.MI
ENEL.MI
ENI.MI
ERG.MI
FBK.MI
FCT.MI
G.MI
GEO.MI
HER.MI
IF.MI
IG.MI
IGV.MI
INRG.MI
INW.MI
IP.MI
ISP.MI
ITW.MI
IVG.MI
LDO.MI
MB.MI
MFEB.MI
MONC.MI
NEXI.MI
PIRC.MI
PRY.MI
PST.MI
PWS.MI
RACE.MI
RDUE.MI
REC.MI
REY.MI
SES.MI
SFER.MI
SFL.MI
SGF.MI
SL.MI
SPM.MI
SRG.MI
STLAM.MI
STMMI.MI
TEN.MI
TES.MI
TGYM.MI
TIT.MI
TPRO.MI
TRN.MI
UCG.MI
UNI.MI
WBD.MI
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
        progress.progress(0.30, text=f"Download completato · {len(data_map)}/{len(tickers)} ticker con dati")
    return data_map, notes


def download_daily_individual(ticker: str, adjusted: bool) -> pd.DataFrame:
    """Retry individuale non cached, usato quando la Daily è arretrata rispetto alla seduta attesa."""
    raw = yf.download(
        tickers=ticker,
        period="5y",
        interval="1d",
        auto_adjust=bool(adjusted),
        actions=False,
        group_by="ticker",
        progress=False,
        threads=False,
        timeout=20,
    )
    return _extract_from_batch(raw, ticker, True)


def _freshness_bucket(ticker: str, market: str) -> str:
    """Separa Italia/USA per confrontare solo ticker con lo stesso calendario di mercato."""
    t = ticker.upper()
    if market == "Italia" or t.endswith(".MI"):
        return "Italia"
    if market == "USA":
        return "USA"
    return "Italia" if t.endswith(".MI") else "USA"


def _expected_last_closed_daily(ticker: str, market: str) -> pd.Timestamp:
    """Ultima seduta che dovrebbe essere già chiusa, usando calendario lun-ven.
    Le festività possono produrre un retry innocuo, ma non un falso dato più recente.
    """
    tz_name, regular_close = _market_clock_for_ticker(ticker, market)
    now_local = pd.Timestamp.now(tz=ZoneInfo(tz_name))
    d = now_local.date()
    if now_local.time().replace(tzinfo=None) < regular_close:
        d = d - pd.Timedelta(days=1)
    else:
        # Dopo la chiusura la Daily odierna può essere disponibile; se Yahoo non l'ha
        # ancora pubblicata il retry non sostituisce comunque dati più vecchi con dati peggiori.
        d = d
    d = pd.Timestamp(d)
    while d.weekday() >= 5:
        d = d - pd.Timedelta(days=1)
    return d.normalize()


def refresh_stale_daily_data(
    data_map: dict[str, pd.DataFrame],
    tickers: list[str],
    adjusted: bool,
    market: str,
    notes: dict[str, str],
    progress: Any | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], int]:
    """
    Controllo freschezza senza modificare la logica Balance.
    Ogni ticker viene confrontato con l'ultima seduta che dovrebbe essere già chiusa
    per il suo mercato, così viene rilevato anche un intero batch Yahoo arretrato.
    """
    closed_preview: dict[str, pd.DataFrame] = {}

    for t in tickers:
        raw = data_map.get(t, pd.DataFrame())
        if raw.empty:
            continue
        closed, _ = only_closed_daily(raw, t, market)
        if closed.empty:
            continue
        closed_preview[t] = closed

    stale: list[str] = []
    for t, closed in closed_preview.items():
        expected = _expected_last_closed_daily(t, market)
        d = pd.Timestamp(closed.index[-1]).normalize()
        if d < expected:
            stale.append(t)

    refreshed = 0
    if progress is not None:
        progress.progress(0.31, text=f"Controllo freschezza Daily · {len(stale)} ticker da verificare")

    for i, t in enumerate(stale, start=1):
        if progress is not None:
            frac = 0.31 + 0.09 * (i / max(1, len(stale)))
            progress.progress(min(frac, 0.40), text=f"Retry freschezza {i}/{len(stale)} · {t}")
        try:
            retry_raw = download_daily_individual(t, adjusted)
            retry_closed, _ = only_closed_daily(retry_raw, t, market)
            old_closed = closed_preview.get(t, pd.DataFrame())
            if retry_closed.empty:
                notes[t] = notes.get(t, "") + " | Retry freschezza senza dati."
                continue
            old_date = pd.Timestamp(old_closed.index[-1]).normalize() if not old_closed.empty else pd.Timestamp.min
            new_date = pd.Timestamp(retry_closed.index[-1]).normalize()
            if new_date > old_date:
                data_map[t] = retry_raw
                closed_preview[t] = retry_closed
                refreshed += 1
                notes[t] = f"Retry freschezza OK: ultima Daily {new_date.date().isoformat()}."
            else:
                notes[t] = notes.get(t, "") + f" | Retry freschezza: ultima Daily {new_date.date().isoformat()}."
        except Exception as exc:
            notes[t] = notes.get(t, "") + f" | Retry freschezza fallito: {type(exc).__name__}: {exc}"

    if progress is not None:
        progress.progress(0.40, text=f"Controllo freschezza completato · {refreshed} ticker aggiornati")
    return data_map, notes, refreshed



class _SimpleHTMLTableParser(HTMLParser):
    """Parser minimale per tabelle HTML pubbliche."""
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            value = " ".join("".join(self._cell).replace("\xa0", " ").split())
            self._row.append(value)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def _http_text(url: str, *, data: bytes | None = None, timeout: int = 12, accept: str = "text/html,*/*") -> str:
    req = Request(
        url,
        data=data,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
            "Accept": accept,
            "Accept-Language": "en-US,en;q=0.9,it;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
        method="POST" if data is not None else "GET",
    )
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_public_number(value: Any) -> float:
    s = str(value).replace("\xa0", " ").strip()
    if not s or s in {"-", "--", "N/A", "n/a"}:
        return math.nan
    s = s.replace(" ", "")
    # Formato internazionale: 1,234.56. Formato europeo: 1.234,56.
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if len(parts[-1]) <= 4:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    s = re.sub(r"[^0-9eE+\-.]", "", s)
    try:
        return float(s)
    except Exception:
        return math.nan


def _yahoo_chart_recent_daily(ticker: str, adjusted: bool, expected: pd.Timestamp, market: str) -> pd.DataFrame:
    """Fallback diretto all'endpoint Chart Yahoo, separato da yfinance."""
    start = int((expected - pd.Timedelta(days=35)).tz_localize("UTC").timestamp())
    end = int((expected + pd.Timedelta(days=3)).tz_localize("UTC").timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker)}"
        f"?period1={start}&period2={end}&interval=1d&includePrePost=false"
        f"&events=div%2Csplits&includeAdjustedClose=true"
    )
    try:
        payload = json.loads(_http_text(url, timeout=12, accept="application/json,*/*"))
        result = ((payload.get("chart") or {}).get("result") or [None])[0]
        if not result:
            return pd.DataFrame()
        ts = result.get("timestamp") or []
        quote_block = (((result.get("indicators") or {}).get("quote") or [{}])[0])
        adj_block = (((result.get("indicators") or {}).get("adjclose") or [{}])[0])
        if not ts:
            return pd.DataFrame()
        tz_name, _ = _market_clock_for_ticker(ticker, market)
        idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(ZoneInfo(tz_name)).tz_localize(None).normalize()
        out = pd.DataFrame(index=idx)
        for src, dst in (("open","open"),("high","high"),("low","low"),("close","close"),("volume","volume")):
            vals = quote_block.get(src) or [math.nan] * len(idx)
            out[dst] = pd.to_numeric(pd.Series(vals, index=idx), errors="coerce")
        if adjusted:
            adj_vals = adj_block.get("adjclose") or [math.nan] * len(idx)
            adj = pd.to_numeric(pd.Series(adj_vals, index=idx), errors="coerce")
            raw_close = out["close"].replace(0, np.nan)
            factor = (adj / raw_close).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            for c in ("open", "high", "low", "close"):
                out[c] = out[c] * factor
        out = out.dropna(subset=["open", "high", "low", "close"])
        return out[~out.index.duplicated(keep="last")].sort_index()
    except Exception:
        return pd.DataFrame()


def _stockanalysis_recent_daily(ticker: str, adjusted: bool, expected: pd.Timestamp) -> pd.DataFrame:
    """Fallback HTML per titoli di Borsa Italiana. Usa solo le righe recenti mancanti."""
    if not ticker.upper().endswith(".MI"):
        return pd.DataFrame()
    root = ticker.upper().removesuffix(".MI")
    url = f"https://stockanalysis.com/quote/bit/{quote(root)}/history/"
    try:
        html = _http_text(url, timeout=12)
    except Exception:
        return pd.DataFrame()
    parser = _SimpleHTMLTableParser()
    try:
        parser.feed(html)
    except Exception:
        return pd.DataFrame()
    rows = parser.rows
    if len(rows) < 2:
        return pd.DataFrame()
    header_idx = next((i for i, r in enumerate(rows) if r and str(r[0]).strip().lower() == "date" and any(str(x).strip().lower() == "open" for x in r)), None)
    if header_idx is None:
        return pd.DataFrame()
    header = [str(x).strip() for x in rows[header_idx]]
    body = [r for r in rows[header_idx + 1:] if len(r) >= len(header)]
    if not body:
        return pd.DataFrame()
    frame = pd.DataFrame([r[:len(header)] for r in body], columns=header)
    cmap = {str(c).strip().lower(): c for c in frame.columns}
    def col(*names: str) -> str | None:
        for n in names:
            if n.lower() in cmap:
                return cmap[n.lower()]
        return None
    c_date, c_open, c_high, c_low, c_close = col("Date"), col("Open"), col("High"), col("Low"), col("Close")
    c_adj, c_vol = col("Adj. Close", "Adj Close"), col("Volume")
    if not all((c_date, c_open, c_high, c_low, c_close)):
        return pd.DataFrame()
    idx = pd.to_datetime(frame[c_date], errors="coerce")
    out = pd.DataFrame(index=idx)
    out["open"] = frame[c_open].map(_parse_public_number).to_numpy()
    out["high"] = frame[c_high].map(_parse_public_number).to_numpy()
    out["low"] = frame[c_low].map(_parse_public_number).to_numpy()
    out["close"] = frame[c_close].map(_parse_public_number).to_numpy()
    out["volume"] = frame[c_vol].map(_parse_public_number).to_numpy() if c_vol else 0.0
    if adjusted and c_adj:
        adj = frame[c_adj].map(_parse_public_number).to_numpy()
        raw = out["close"].to_numpy(dtype=float)
        factor = np.where(np.isfinite(adj) & np.isfinite(raw) & (raw != 0), adj / raw, 1.0)
        for c in ("open", "high", "low", "close"):
            out[c] = out[c].to_numpy(dtype=float) * factor
    out = out[~out.index.isna()].dropna(subset=["open", "high", "low", "close"])
    if out.empty:
        return out
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


def _stooq_recent_daily(ticker: str, expected: pd.Timestamp) -> pd.DataFrame:
    """Ultimo fallback CSV gratuito. Per Milano prova il simbolo root.it."""
    if not ticker.upper().endswith(".MI"):
        return pd.DataFrame()
    symbol = ticker.upper().removesuffix(".MI").lower() + ".it"
    d1 = (expected - pd.Timedelta(days=35)).strftime("%Y%m%d")
    d2 = (expected + pd.Timedelta(days=1)).strftime("%Y%m%d")
    url = f"https://stooq.com/q/d/l/?s={quote(symbol)}&d1={d1}&d2={d2}&i=d"
    try:
        csv_text = _http_text(url, timeout=12, accept="text/csv,text/plain,*/*")
        if "Date,Open,High,Low,Close" not in csv_text:
            return pd.DataFrame()
        frame = pd.read_csv(io.StringIO(csv_text))
    except Exception:
        return pd.DataFrame()
    if frame.empty or "Date" not in frame.columns:
        return pd.DataFrame()
    idx = pd.to_datetime(frame["Date"], errors="coerce")
    out = pd.DataFrame(index=idx)
    for src, dst in (("Open","open"),("High","high"),("Low","low"),("Close","close"),("Volume","volume")):
        if src in frame.columns:
            out[dst] = pd.to_numeric(frame[src], errors="coerce").to_numpy()
        elif dst == "volume":
            out[dst] = 0.0
        else:
            return pd.DataFrame()
    out = out[~out.index.isna()].dropna(subset=["open", "high", "low", "close"])
    if out.empty:
        return out
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


def _merge_recent_rows(old: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    if recent.empty:
        return old
    if old.empty:
        return recent.copy()
    base = _normalize_ohlc(old)
    base.index = pd.DatetimeIndex(base.index).tz_localize(None).normalize()
    merged = pd.concat([base, recent], axis=0).sort_index()
    return merged[~merged.index.duplicated(keep="last")]


def refresh_stale_italy_from_public_fallbacks(
    data_map: dict[str, pd.DataFrame],
    tickers: list[str],
    adjusted: bool,
    market: str,
    notes: dict[str, str],
    progress: Any | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], int, list[str]]:
    """Integra solo Daily mancanti. Ordine: Yahoo Chart diretto -> StockAnalysis -> Stooq."""
    italy_tickers = [t for t in tickers if market == "Italia" or t.upper().endswith(".MI")]
    stale: list[str] = []
    for t in italy_tickers:
        raw = data_map.get(t, pd.DataFrame())
        if raw.empty:
            continue
        closed, _ = only_closed_daily(raw, t, market)
        if closed.empty:
            continue
        if pd.Timestamp(closed.index[-1]).normalize() < _expected_last_closed_daily(t, market):
            stale.append(t)

    refreshed = 0
    still_stale: list[str] = []
    for i, t in enumerate(stale, start=1):
        expected = _expected_last_closed_daily(t, market)
        if progress is not None:
            frac = 0.40 + 0.08 * (i / max(1, len(stale)))
            progress.progress(min(frac, 0.48), text=f"Fallback Daily {i}/{len(stale)} · {t}")
        providers = (
            ("Yahoo Chart", lambda: _yahoo_chart_recent_daily(t, adjusted, expected, market)),
            ("StockAnalysis", lambda: _stockanalysis_recent_daily(t, adjusted, expected)),
            ("Stooq", lambda: _stooq_recent_daily(t, expected)),
        )
        updated = False
        provider_notes: list[str] = []
        for provider_name, loader in providers:
            try:
                recent = loader()
            except Exception as exc:
                provider_notes.append(f"{provider_name}: {type(exc).__name__}")
                continue
            if recent.empty:
                provider_notes.append(f"{provider_name}: no data")
                continue
            merged = _merge_recent_rows(data_map.get(t, pd.DataFrame()), recent)
            closed, _ = only_closed_daily(merged, t, market)
            if closed.empty:
                provider_notes.append(f"{provider_name}: no closed")
                continue
            new_date = pd.Timestamp(closed.index[-1]).normalize()
            if new_date >= expected:
                data_map[t] = merged
                refreshed += 1
                notes[t] = notes.get(t, "") + f" | {provider_name} OK: ultima Daily {new_date.date().isoformat()}."
                updated = True
                break
            provider_notes.append(f"{provider_name}: {new_date.date().isoformat()}")
        if not updated:
            still_stale.append(t)
            notes[t] = notes.get(t, "") + " | Fallback recenti: " + "; ".join(provider_notes)

    if progress is not None:
        progress.progress(0.48, text=f"Freschezza Italia completata · {refreshed} aggiornati · {len(still_stale)} ancora arretrati")
    return data_map, notes, refreshed, still_stale


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
# BALANCE ENGINE — PORTING DIRETTO G. Balance Zones Pro v0.5.3.6
# =============================================================================
# Principio di questa rebuild:
# - il motore Balance NON viene "reinterpretato" dallo screener;
# - A e B sono due istanze indipendenti dello stesso algoritmo;
# - il layer screener (tocchi / area attiva / ranking operativo) viene applicato
#   solo DOPO che le Balance sono state selezionate.
#
# Nota tecnica Python vs Pine:
# - lo screener Python lavora su Daily; quindi Target TF = Daily e stepBars = 1;
# - la serie dati arriva da Yahoo/fallback pubblici e può differire da TradingView.
#   Il motore è portato 1:1 nella logica, ma nessun provider esterno può garantire
#   OHLC identici a TradingView in ogni simbolo/giorno.
# =============================================================================

BUILD = "V4.6 REBUILD"
ENGINE_REFERENCE = "G. Balance Zones Pro v0.5.3.6"

# Defaults identici al riferimento Pine v0.5.3.6.
BAL_SCAN_STEP_PCT = 1.0
BAL_MAX_ZONES = 9
BAL_MIN_COMPATIBLE_HITS = 1
BAL_VALIDATION_A = 10
BAL_VALIDATION_B = 5
BAL_BREAK_SOURCE = "Close"
BAL_MIN_STRENGTH = 0.0
BAL_ATR_LENGTH = 14

BAL_ENABLE_INDEPENDENT_VALIDATION = True
BAL_INDEPENDENT_COOLDOWN_BARS = 6
BAL_MIN_REACTION_ATR = 0.20

BAL_ZONE_WIDTH_MODE = "ATR"
BAL_ZONE_HALF_ATR = 0.20
BAL_ZONE_HEIGHT_PCT = 0.07
BAL_ZONE_HALF_TICKS = 12
BAL_ZONE_HEIGHT_RANGE_PCT = 1.0

BAL_SPACING_MODE = "Range %"
BAL_MIN_SPACING_RANGE_PCT = 8.0
BAL_MIN_SPACING_ATR = 0.80
BAL_MIN_SPACING_PRICE_PCT = 0.20

BAL_CLUSTER_ADJACENT_LEVELS = False
BAL_CLUSTER_RADIUS_STEPS = 1.5
BAL_CLUSTER_MIN_STRENGTH_RATIO = 0.70

BAL_UPDATE_FREQUENCY_BARS = 1
BAL_RETAIN_PREVIOUS_ZONES = True
BAL_RETENTION_STRENGTH = 25.0
BAL_RETENTION_TOLERANCE_STEPS = 1.5


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
    engine: str = "A"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def wilder_atr(data: pd.DataFrame, length: int = BAL_ATR_LENGTH) -> pd.Series:
    """Equivalente operativo di ta.atr(): True Range + Wilder RMA."""
    if data.empty:
        return pd.Series(dtype=float)

    high = data["high"].astype(float)
    low = data["low"].astype(float)
    close = data["close"].astype(float)
    prev_close = close.shift(1)

    # ta.tr(true): sulla prima barra disponibile usa High-Low se prev close è na.
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=True)

    atr = pd.Series(np.nan, index=tr.index, dtype=float)
    if len(tr) < int(length):
        return atr

    atr.iloc[length - 1] = float(tr.iloc[:length].mean())
    for i in range(length, len(tr)):
        atr.iloc[i] = (float(atr.iloc[i - 1]) * (length - 1) + float(tr.iloc[i])) / float(length)
    return atr


def _mintick_floor(data: pd.DataFrame) -> float:
    """
    Yahoo non espone syminfo.mintick.
    Con la geometria default ATR questa soglia non entra praticamente mai nel calcolo;
    manteniamo un epsilon numerico solo come protezione da zero.
    """
    if data.empty:
        return 1e-12
    ref = abs(float(data["close"].iloc[-1]))
    return max(1e-12, ref * 1e-12)


def _zone_half(
    center: float,
    rng: float,
    atr_now: float,
    mintick: float,
    *,
    zone_width_mode: str = BAL_ZONE_WIDTH_MODE,
    zone_half_atr: float = BAL_ZONE_HALF_ATR,
    zone_height_pct: float = BAL_ZONE_HEIGHT_PCT,
    zone_half_ticks: int = BAL_ZONE_HALF_TICKS,
    zone_height_range_pct: float = BAL_ZONE_HEIGHT_RANGE_PCT,
) -> float:
    """Porting diretto di f_zoneHalf()."""
    half = mintick
    if zone_width_mode == "Percent Price":
        half = center * float(zone_height_pct) / 200.0
    elif zone_width_mode == "ATR":
        half = max(mintick, float(atr_now) * float(zone_half_atr))
    elif zone_width_mode == "Ticks":
        half = mintick * int(zone_half_ticks)
    else:
        half = float(rng) * float(zone_height_range_pct) / 200.0
    return max(mintick, float(half))


def _min_spacing(
    ref_price: float,
    rng: float,
    atr_now: float,
    mintick: float,
    *,
    spacing_mode: str = BAL_SPACING_MODE,
    min_spacing_range_pct: float = BAL_MIN_SPACING_RANGE_PCT,
    min_spacing_atr: float = BAL_MIN_SPACING_ATR,
    min_spacing_price_pct: float = BAL_MIN_SPACING_PRICE_PCT,
) -> float:
    """Porting diretto di f_minSpacing()."""
    if spacing_mode == "ATR":
        return max(mintick, float(atr_now) * float(min_spacing_atr))
    if spacing_mode == "Percent Price":
        return max(mintick, float(ref_price) * float(min_spacing_price_pct) / 100.0)
    return max(mintick, float(rng) * float(min_spacing_range_pct) / 100.0)


def _future_extremes(values: np.ndarray, validation_bars: int) -> tuple[np.ndarray, np.ndarray]:
    """Min/max delle successive validation_bars barre, in ordine cronologico."""
    n = len(values)
    fmin = np.full(n, np.nan, dtype=float)
    fmax = np.full(n, np.nan, dtype=float)
    vb = int(validation_bars)
    for pos in range(n):
        a = pos + 1
        b = min(n, pos + vb + 1)
        if a < b:
            window = values[a:b]
            fmin[pos] = float(np.nanmin(window))
            fmax[pos] = float(np.nanmax(window))
    return fmin, fmax


def _evaluate_compatible(
    data: pd.DataFrame,
    center: float,
    half: float,
    scan_limit: int,
    step_bars: int,
    valid_bars: int,
    *,
    break_source: str = BAL_BREAK_SOURCE,
    future_min: np.ndarray | None = None,
    future_max: np.ndarray | None = None,
) -> tuple[int, int, int, int, int, float]:
    """
    Porting 1:1 della logica f_evaluateCompatibleV().
    In Python Daily step_bars=1, ma il parametro resta esplicito come nel Pine.
    """
    lows = data["low"].to_numpy(float)
    highs = data["high"].to_numpy(float)
    closes = data["close"].to_numpy(float)

    top = float(center + half)
    bottom = float(center - half)
    support_hits = 0
    resistance_hits = 0
    dwell_bars = 0
    nearest_hit_age: int | None = None

    hold_span = int(valid_bars) * int(step_bars)
    first_offset = hold_span + int(step_bars)
    last_offset = int(scan_limit) - int(step_bars)
    current = len(data) - 1

    # Ottimizzazione equivalente: le rotture successive vengono precalcolate.
    # Non cambia la finestra né il verso temporale della logica Pine.
    if break_source == "Close":
        break_low_arr = closes
        break_high_arr = closes
    else:
        break_low_arr = lows
        break_high_arr = highs

    if future_min is None or future_max is None:
        future_min, future_max = _future_extremes(break_low_arr, int(valid_bars))
        _, future_max = _future_extremes(break_high_arr, int(valid_bars))

    if last_offset >= first_offset:
        for i in range(first_offset, last_offset + 1, int(step_bars)):
            pos = current - i
            if pos < 0 or pos >= len(data):
                continue

            overlaps = lows[pos] <= top and highs[pos] >= bottom
            if not overlaps:
                continue

            dwell_bars += 1
            is_support = closes[pos] >= center

            # Pine:
            # for k=1..validBars, idx=i-k*stepBars.
            # In ordine cronologico Python significa pos+k*stepBars.
            end_pos = pos + int(valid_bars) * int(step_bars)
            if end_pos >= len(data):
                continue

            broken = False
            if is_support:
                vals = break_low_arr[pos + int(step_bars): end_pos + 1: int(step_bars)]
                if vals.size and float(np.nanmin(vals)) < bottom:
                    broken = True
            else:
                vals = break_high_arr[pos + int(step_bars): end_pos + 1: int(step_bars)]
                if vals.size and float(np.nanmax(vals)) > top:
                    broken = True

            if not broken:
                if is_support:
                    support_hits += 1
                else:
                    resistance_hits += 1
                age = int(round(i / int(step_bars)))
                nearest_hit_age = age if nearest_hit_age is None else min(nearest_hit_age, age)

    hits = support_hits + resistance_hits
    hit_score = 1.0 - math.exp(-hits / 18.0)
    dwell_score = 1.0 - math.exp(-dwell_bars / 45.0)
    role_mix = min(support_hits, resistance_hits) / max(1.0, float(max(support_hits, resistance_hits)))
    freshness_score = 0.0 if nearest_hit_age is None else math.exp(-nearest_hit_age / 200.0)
    density_score = hits / max(1.0, float(dwell_bars))
    density_score = min(1.0, density_score)

    strength = 100.0 * (
        0.50 * hit_score
        + 0.15 * dwell_score
        + 0.12 * role_mix
        + 0.13 * freshness_score
        + 0.10 * density_score
    )
    strength = _clamp(strength, 0.0, 100.0)
    return support_hits, resistance_hits, hits, dwell_bars, (nearest_hit_age if nearest_hit_age is not None else 99999), strength


def _evaluate_independent(
    data: pd.DataFrame,
    atr: pd.Series,
    center: float,
    half: float,
    scan_limit: int,
    step_bars: int,
    valid_bars: int,
    mintick: float,
    *,
    enable_independent_validation: bool = BAL_ENABLE_INDEPENDENT_VALIDATION,
    independent_cooldown_bars: int = BAL_INDEPENDENT_COOLDOWN_BARS,
    min_reaction_atr: float = BAL_MIN_REACTION_ATR,
    break_source: str = BAL_BREAK_SOURCE,
) -> tuple[int, int, int, int, int, float]:
    """Porting diretto di f_evaluateIndependentV()."""
    top = float(center + half)
    bottom = float(center - half)
    tests = 0
    successes = 0
    support_success = 0
    resistance_success = 0
    breaks = 0
    last_accepted_offset: int | None = None

    hold_span = int(valid_bars) * int(step_bars)
    cooldown_span = int(independent_cooldown_bars) * int(step_bars)
    first_offset = hold_span + int(step_bars)
    last_offset = int(scan_limit) - int(step_bars)

    if not enable_independent_validation or last_offset < first_offset:
        return tests, successes, support_success, resistance_success, breaks, math.nan

    lows = data["low"].to_numpy(float)
    highs = data["high"].to_numpy(float)
    closes = data["close"].to_numpy(float)
    atr_arr = atr.to_numpy(float)
    current = len(data) - 1

    for i in range(first_offset, last_offset + 1, int(step_bars)):
        pos = current - i
        prev_pos = current - (i + int(step_bars))
        if pos < 0 or prev_pos < 0:
            continue

        overlaps = lows[pos] <= top and highs[pos] >= bottom
        previous_above = closes[prev_pos] > top
        previous_below = closes[prev_pos] < bottom
        independent_entry = overlaps and (previous_above or previous_below)
        cooldown_ok = last_accepted_offset is None or abs(i - last_accepted_offset) >= cooldown_span

        if not (independent_entry and cooldown_ok):
            continue

        is_support_test = previous_above
        broken = False
        max_away = 0.0
        atr_at_test = atr_arr[pos] if pos < len(atr_arr) and math.isfinite(atr_arr[pos]) else mintick
        atr_at_test = max(float(atr_at_test), mintick)

        for k in range(1, int(valid_bars) + 1):
            target_offset = i - k * int(step_bars)
            if target_offset < 0:
                continue
            later_pos = current - target_offset
            if later_pos < 0 or later_pos >= len(data):
                continue

            break_low = closes[later_pos] if break_source == "Close" else lows[later_pos]
            break_high = closes[later_pos] if break_source == "Close" else highs[later_pos]

            if is_support_test:
                if break_low < bottom:
                    broken = True
                max_away = max(max_away, highs[later_pos] - center)
            else:
                if break_high > top:
                    broken = True
                max_away = max(max_away, center - lows[later_pos])

        reacted_enough = max_away >= atr_at_test * float(min_reaction_atr)
        success = (not broken) and reacted_enough
        tests += 1
        if success:
            successes += 1
            if is_support_test:
                support_success += 1
            else:
                resistance_success += 1
        if broken:
            breaks += 1
        last_accepted_offset = i

    reliability = 100.0 * successes / tests if tests > 0 else math.nan
    return tests, successes, support_success, resistance_success, breaks, reliability


def _too_close_zone(candidate: float, zones: list[BalanceZone], spacing: float) -> bool:
    """Porting diretto di f_tooCloseZone()."""
    for z in zones:
        if abs(float(candidate) - float(z.center)) < float(spacing):
            return True
    return False


def _cluster_center(
    peak_idx: int,
    centers: list[float],
    strengths: list[float],
    rng: float,
    *,
    cluster_adjacent_levels: bool = BAL_CLUSTER_ADJACENT_LEVELS,
    scan_step_pct: float = BAL_SCAN_STEP_PCT,
    cluster_radius_steps: float = BAL_CLUSTER_RADIUS_STEPS,
    cluster_min_strength_ratio: float = BAL_CLUSTER_MIN_STRENGTH_RATIO,
) -> float:
    """Porting diretto di f_clusterCenter()."""
    peak_center = float(centers[peak_idx])
    peak_strength = float(strengths[peak_idx])
    result = peak_center

    if cluster_adjacent_levels:
        radius_price = float(rng) * float(scan_step_pct) / 100.0 * float(cluster_radius_steps)
        weighted_sum = 0.0
        weight_total = 0.0
        for c, s in zip(centers, strengths):
            if abs(float(c) - peak_center) <= radius_price and float(s) >= peak_strength * float(cluster_min_strength_ratio):
                w = max(float(s), 0.01)
                weighted_sum += float(c) * w
                weight_total += w
        if weight_total > 0:
            result = weighted_sum / weight_total

    return result


def _build_zone(
    data: pd.DataFrame,
    atr: pd.Series,
    center: float,
    range_low: float,
    rng: float,
    atr_now: float,
    scan_limit: int,
    step_bars: int,
    valid_bars: int,
    mintick: float,
    engine: str,
    *,
    scan_step_pct: float = BAL_SCAN_STEP_PCT,
    zone_width_mode: str = BAL_ZONE_WIDTH_MODE,
    zone_half_atr: float = BAL_ZONE_HALF_ATR,
    zone_height_pct: float = BAL_ZONE_HEIGHT_PCT,
    zone_half_ticks: int = BAL_ZONE_HALF_TICKS,
    zone_height_range_pct: float = BAL_ZONE_HEIGHT_RANGE_PCT,
    break_source: str = BAL_BREAK_SOURCE,
) -> BalanceZone:
    """Porting diretto di f_buildZoneV()."""
    half = _zone_half(
        center,
        rng,
        atr_now,
        mintick,
        zone_width_mode=zone_width_mode,
        zone_half_atr=zone_half_atr,
        zone_height_pct=zone_height_pct,
        zone_half_ticks=zone_half_ticks,
        zone_height_range_pct=zone_height_range_pct,
    )
    sup, res, hits, dwell, last_age, strength = _evaluate_compatible(
        data,
        center,
        half,
        scan_limit,
        step_bars,
        valid_bars,
        break_source=break_source,
    )
    ind_tests, ind_success, ind_sup, ind_res, ind_breaks, reliability = _evaluate_independent(
        data,
        atr,
        center,
        half,
        scan_limit,
        step_bars,
        valid_bars,
        mintick,
        break_source=break_source,
    )
    pct = 100.0 * (float(center) - float(range_low)) / float(rng) if rng > 0 else 0.0

    return BalanceZone(
        center=float(center),
        half=float(half),
        pct_range=float(pct),
        strength=float(strength),
        hits=int(hits),
        support_hits=int(sup),
        resistance_hits=int(res),
        dwell=int(dwell),
        last_hit_age=int(last_age),
        independent_tests=int(ind_tests),
        independent_successes=int(ind_success),
        independent_support_success=int(ind_sup),
        independent_resistance_success=int(ind_res),
        independent_breaks=int(ind_breaks),
        reliability=float(reliability),
        engine=str(engine),
    )


def _scan_centers(scan_step_pct: float) -> list[float]:
    """Replica il for Pine: for pct = 0.0 to 100.0 by scanStepPct."""
    step = float(scan_step_pct)
    vals: list[float] = []
    pct = 0.0
    # Guard contro errori floating point.
    while pct <= 100.0 + 1e-10:
        vals.append(min(100.0, pct))
        pct += step
    # Pine non aggiunge un 100 artificiale se il passo non ci arriva esattamente.
    # Se il clamp ha creato un duplicato finale lo eliminiamo.
    out: list[float] = []
    for v in vals:
        if not out or abs(v - out[-1]) > 1e-12:
            out.append(v)
    return out


def _select_engine_snapshot(
    data: pd.DataFrame,
    atr: pd.Series,
    *,
    range_low: float,
    active_range: float,
    atr_now: float,
    scan_limit: int,
    step_bars: int,
    valid_bars: int,
    engine: str,
    previous_zones: list[BalanceZone] | None = None,
    scan_step_pct: float = BAL_SCAN_STEP_PCT,
    max_zones: int = BAL_MAX_ZONES,
    min_compatible_hits: int = BAL_MIN_COMPATIBLE_HITS,
    min_strength: float = BAL_MIN_STRENGTH,
    break_source: str = BAL_BREAK_SOURCE,
    zone_width_mode: str = BAL_ZONE_WIDTH_MODE,
    zone_half_atr: float = BAL_ZONE_HALF_ATR,
    zone_height_pct: float = BAL_ZONE_HEIGHT_PCT,
    zone_half_ticks: int = BAL_ZONE_HALF_TICKS,
    zone_height_range_pct: float = BAL_ZONE_HEIGHT_RANGE_PCT,
    spacing_mode: str = BAL_SPACING_MODE,
    min_spacing_range_pct: float = BAL_MIN_SPACING_RANGE_PCT,
    min_spacing_atr: float = BAL_MIN_SPACING_ATR,
    min_spacing_price_pct: float = BAL_MIN_SPACING_PRICE_PCT,
    cluster_adjacent_levels: bool = BAL_CLUSTER_ADJACENT_LEVELS,
    cluster_radius_steps: float = BAL_CLUSTER_RADIUS_STEPS,
    cluster_min_strength_ratio: float = BAL_CLUSTER_MIN_STRENGTH_RATIO,
    retain_previous_zones: bool = BAL_RETAIN_PREVIOUS_ZONES,
    retention_strength: float = BAL_RETENTION_STRENGTH,
    retention_tolerance_steps: float = BAL_RETENTION_TOLERANCE_STEPS,
) -> list[BalanceZone]:
    """
    Blocco A/B del riferimento Pine.
    La funzione è la stessa per A e B: cambia soltanto valid_bars e l'etichetta engine.
    """
    mintick = _mintick_floor(data)
    ref_close = float(data["close"].iloc[-1])

    cand_centers: list[float] = []
    cand_strengths: list[float] = []
    cand_hits: list[int] = []

    for pct in _scan_centers(scan_step_pct):
        center = float(range_low) + float(active_range) * float(pct) / 100.0
        half = _zone_half(
            center,
            active_range,
            atr_now,
            mintick,
            zone_width_mode=zone_width_mode,
            zone_half_atr=zone_half_atr,
            zone_height_pct=zone_height_pct,
            zone_half_ticks=zone_half_ticks,
            zone_height_range_pct=zone_height_range_pct,
        )
        _sup, _res, hits, _dwell, _age, strength = _evaluate_compatible(
            data,
            center,
            half,
            scan_limit,
            step_bars,
            valid_bars,
            break_source=break_source,
        )
        if hits >= int(min_compatible_hits) and strength >= float(min_strength):
            cand_centers.append(center)
            cand_strengths.append(float(strength))
            cand_hits.append(int(hits))

    new_zones: list[BalanceZone] = []
    used = [False] * len(cand_centers)
    spacing = _min_spacing(
        ref_close,
        active_range,
        atr_now,
        mintick,
        spacing_mode=spacing_mode,
        min_spacing_range_pct=min_spacing_range_pct,
        min_spacing_atr=min_spacing_atr,
        min_spacing_price_pct=min_spacing_price_pct,
    )
    retention_tolerance = float(active_range) * float(scan_step_pct) / 100.0 * float(retention_tolerance_steps)

    # Retention identica al Pine. In una nuova scansione Python previous_zones è
    # normalmente vuoto: equivale ad aggiungere/ricalcolare da zero l'indicatore.
    old_zones = list(previous_zones or [])
    if retain_previous_zones and old_zones and cand_centers:
        for old_zone in old_zones:
            if len(new_zones) >= int(max_zones):
                break

            best_idx = -1
            best_dist = 1e20
            for c in range(len(cand_centers)):
                if not used[c]:
                    d = abs(cand_centers[c] - old_zone.center)
                    if d < best_dist:
                        best_dist = d
                        best_idx = c

            if (
                best_idx >= 0
                and best_dist <= retention_tolerance
                and cand_strengths[best_idx] >= float(retention_strength)
            ):
                selected_center = _cluster_center(
                    best_idx,
                    cand_centers,
                    cand_strengths,
                    active_range,
                    cluster_adjacent_levels=cluster_adjacent_levels,
                    scan_step_pct=scan_step_pct,
                    cluster_radius_steps=cluster_radius_steps,
                    cluster_min_strength_ratio=cluster_min_strength_ratio,
                )
                if not _too_close_zone(selected_center, new_zones, spacing):
                    z = _build_zone(
                        data,
                        atr,
                        selected_center,
                        range_low,
                        active_range,
                        atr_now,
                        scan_limit,
                        step_bars,
                        valid_bars,
                        mintick,
                        engine,
                        scan_step_pct=scan_step_pct,
                        zone_width_mode=zone_width_mode,
                        zone_half_atr=zone_half_atr,
                        zone_height_pct=zone_height_pct,
                        zone_half_ticks=zone_half_ticks,
                        zone_height_range_pct=zone_height_range_pct,
                        break_source=break_source,
                    )
                    if z.hits >= int(min_compatible_hits) and z.strength >= float(min_strength):
                        new_zones.append(z)
                        used[best_idx] = True

    # array.sort_indices(candStrength, order.descending)
    sorted_idx = sorted(range(len(cand_strengths)), key=lambda i: (-cand_strengths[i], i))
    for idx in sorted_idx:
        if len(new_zones) >= int(max_zones):
            break
        if used[idx]:
            continue

        selected_center = _cluster_center(
            idx,
            cand_centers,
            cand_strengths,
            active_range,
            cluster_adjacent_levels=cluster_adjacent_levels,
            scan_step_pct=scan_step_pct,
            cluster_radius_steps=cluster_radius_steps,
            cluster_min_strength_ratio=cluster_min_strength_ratio,
        )
        if _too_close_zone(selected_center, new_zones, spacing):
            continue

        z = _build_zone(
            data,
            atr,
            selected_center,
            range_low,
            active_range,
            atr_now,
            scan_limit,
            step_bars,
            valid_bars,
            mintick,
            engine,
            scan_step_pct=scan_step_pct,
            zone_width_mode=zone_width_mode,
            zone_half_atr=zone_half_atr,
            zone_height_pct=zone_height_pct,
            zone_half_ticks=zone_half_ticks,
            zone_height_range_pct=zone_height_range_pct,
            break_source=break_source,
        )
        if z.hits >= int(min_compatible_hits) and z.strength >= float(min_strength):
            new_zones.append(z)
            used[idx] = True

    new_zones.sort(key=lambda z: z.center)
    return new_zones


def _reaction_covered_by_structural(reaction: BalanceZone, structural: list[BalanceZone]) -> bool:
    """Deduplicazione identica al render/bridge del riferimento: prevale A."""
    r_top = reaction.center + reaction.half
    r_bottom = reaction.center - reaction.half
    for z in structural:
        a_top = z.center + z.half
        a_bottom = z.center - z.half
        if r_top >= a_bottom and r_bottom <= a_top:
            return True
    return False


def analyze_balance_zones(
    engine_data: pd.DataFrame,
    *,
    engine_mode: str = "A+B",
    lookback: int = LOOKBACK,
    previous_structural: list[BalanceZone] | None = None,
    previous_reaction: list[BalanceZone] | None = None,
) -> dict[str, Any]:
    """
    Snapshot corrente del Balance Zones Pro v0.5.3.6 su Daily.

    IMPORTANTE:
    - A e B vengono SEMPRE calcolati indipendentemente, come nel riferimento.
    - engine_mode appartiene al layer screener e decide soltanto quali zone
      vengono considerate DOPO la selezione.
    """
    unavailable = {
        "available": False,
        "zones": [],
        "structural_zones": [],
        "reaction_zones": [],
        "reaction_all": [],
        "detail": "Dati Daily insufficienti.",
    }

    if engine_data is None or engine_data.empty:
        return unavailable

    data = _normalize_ohlc(engine_data).dropna(subset=["high", "low", "close"])
    if len(data) < 60:
        return unavailable

    atr = wilder_atr(data, BAL_ATR_LENGTH)
    atr_now = float(atr.iloc[-1]) if len(atr) and pd.notna(atr.iloc[-1]) else math.nan
    if not math.isfinite(atr_now) or atr_now <= 0:
        return unavailable

    # Target TF = Daily = timeframe dei dati Python => stepBars = 1.
    step_bars = 1
    target_bars_in_window = max(1, min(int(lookback), 5000))
    current_index = len(data) - 1
    scan_limit = max(10, min(4800, min(target_bars_in_window, current_index)))

    # f_targetRange("Lookback Bars", ..., 400) = ta.highest/lowest sulle ultime 400 barre.
    window_len = min(target_bars_in_window, len(data))
    range_window = data.iloc[-window_len:]
    range_high = float(range_window["high"].max())
    range_low = float(range_window["low"].min())
    mintick = _mintick_floor(data)
    active_range = max(range_high - range_low, mintick * 10.0)

    if not math.isfinite(active_range) or active_range <= 0:
        return unavailable

    structural: list[BalanceZone] = []
    reaction_all: list[BalanceZone] = []

    # MOTORE A — stessa condizione del Pine.
    if scan_limit > BAL_VALIDATION_A * step_bars + step_bars * 2:
        structural = _select_engine_snapshot(
            data,
            atr,
            range_low=range_low,
            active_range=active_range,
            atr_now=atr_now,
            scan_limit=scan_limit,
            step_bars=step_bars,
            valid_bars=BAL_VALIDATION_A,
            engine="A",
            previous_zones=previous_structural,
        )

    # MOTORE B — calcolato autonomamente anche se lo screener è in Solo A.
    if scan_limit > BAL_VALIDATION_B * step_bars + step_bars * 2:
        reaction_all = _select_engine_snapshot(
            data,
            atr,
            range_low=range_low,
            active_range=active_range,
            atr_now=atr_now,
            scan_limit=scan_limit,
            step_bars=step_bars,
            valid_bars=BAL_VALIDATION_B,
            engine="B",
            previous_zones=previous_reaction,
        )

    reaction_visible = [z for z in reaction_all if not _reaction_covered_by_structural(z, structural)]

    mode = str(engine_mode or "A+B")
    if mode == "Solo A":
        combined = list(structural)
    else:
        combined = sorted(structural + reaction_visible, key=lambda z: z.center)

    if not combined:
        return {
            **unavailable,
            "structural_zones": structural,
            "reaction_zones": reaction_visible,
            "reaction_all": reaction_all,
            "detail": "Nessuna Balance selezionata.",
        }

    return {
        "available": True,
        "zones": combined,
        "structural_zones": structural,
        "reaction_zones": reaction_visible,
        "reaction_all": reaction_all,
        "atr14": atr_now,
        "range_high": range_high,
        "range_low": range_low,
        "scan_limit": scan_limit,
        "engine_mode": mode,
        "detail": (
            f"{len(structural)} Structural A + {len(reaction_visible)} Reaction B visibili "
            f"({len(reaction_all)} B selezionate prima della dedup) | "
            f"Lookback {target_bars_in_window} Daily | ATR14 {atr_now:.4f}"
        ),
    }


# =============================================================================
# ACTIVE AREA — LAYER SCREENING, NON PARTE DEL MOTORE BALANCE
# =============================================================================
def candle_touches_zone(row: pd.Series, z: BalanceZone) -> bool:
    bottom = z.center - z.half
    top = z.center + z.half
    return float(row["low"]) <= top and float(row["high"]) >= bottom


def price_inside_zone(price: float, z: BalanceZone) -> bool:
    bottom = z.center - z.half
    top = z.center + z.half
    return bottom <= float(price) <= top


def active_metrics(
    closed_data: pd.DataFrame,
    z: BalanceZone,
    require_last_close_inside: bool,
    interaction_window: int,
    min_interaction_bars: int,
) -> dict[str, Any]:
    """
    Conteggio SOLO su Daily chiuse.
    Il motore Balance resta separato e può usare anche l'ultima barra disponibile
    del feed, come il riferimento Pine.
    """
    interaction_window = int(max(2, min(10, interaction_window)))
    min_interaction_bars = int(max(1, min(interaction_window, min_interaction_bars)))

    if closed_data.empty or len(closed_data) < interaction_window:
        return {
            "active": False,
            "touches": 0,
            "touch_flags": [False] * interaction_window,
            "inside": False,
            "score": math.nan,
        }

    recent = closed_data.iloc[-interaction_window:]
    flags = [candle_touches_zone(recent.iloc[i], z) for i in range(interaction_window)]
    touches = int(sum(flags))
    last_close = float(closed_data.iloc[-1]["close"])
    inside = price_inside_zone(last_close, z)
    active = touches >= min_interaction_bars and (inside if require_last_close_inside else True)

    rel_score = 50.0 if pd.isna(z.reliability) else float(z.reliability)
    touch_score = 100.0 * float(touches) / float(interaction_window)
    score = _clamp(0.45 * touch_score + 0.35 * float(z.strength) + 0.20 * rel_score, 0.0, 100.0)

    return {
        "active": bool(active),
        "touches": int(touches),
        "touch_flags": flags,
        "inside": bool(inside),
        "score": float(score),
    }


def active_zone_row(
    label: str,
    ticker: str,
    closed_data: pd.DataFrame,
    balance: dict[str, Any],
    require_last_close_inside: bool,
    interaction_window: int,
    min_interaction_bars: int,
) -> dict[str, Any] | None:
    """Una sola Balance per ticker: la zona eleggibile con Score più alto."""
    if closed_data.empty or len(closed_data) < int(interaction_window):
        return None

    last_close = float(closed_data.iloc[-1]["close"])
    last_date = pd.Timestamp(closed_data.index[-1])

    best_zone: BalanceZone | None = None
    best_metrics: dict[str, Any] | None = None
    best_score = -1.0

    for z in balance.get("zones", []):
        m = active_metrics(
            closed_data,
            z,
            require_last_close_inside,
            interaction_window,
            min_interaction_bars,
        )
        if not m["active"]:
            continue
        if float(m["score"]) > best_score:
            best_score = float(m["score"])
            best_zone = z
            best_metrics = m

    if best_zone is None or best_metrics is None:
        return None

    z = best_zone
    flags = best_metrics["touch_flags"]
    engine_label = "Structural A" if str(z.engine).upper() == "A" else "Reaction B"

    return {
        "Strumento": label,
        "Ticker": ticker,
        "Motore": engine_label,
        "Area": "AREA ATTIVA",
        "Ultimo Close": last_close,
        "Balance": z.center,
        "Zona min": z.center - z.half,
        "Zona max": z.center + z.half,
        "Tocchi": f"{int(best_metrics['touches'])}/{int(interaction_window)}",
        "Sequenza tocchi": " ".join("SI" if x else "NO" for x in flags),
        "Ultimo Close dentro": "SI" if best_metrics["inside"] else "NO",
        "Score": best_score,
        "ST": z.strength,
        "H": z.hits,
        "T": z.independent_tests,
        "R %": z.reliability,
        "Successi indipendenti": z.independent_successes,
        "Break indipendenti": z.independent_breaks,
        "Support H": z.support_hits,
        "Resistance H": z.resistance_hits,
        "Dwell": z.dwell,
        "Last Hit Age": z.last_hit_age,
        "Data ultima Daily chiusa": last_date.strftime("%Y-%m-%d"),
    }


def all_balance_rows(
    label: str,
    ticker: str,
    closed_data: pd.DataFrame,
    balance: dict[str, Any],
    require_last_close_inside: bool,
    interaction_window: int,
    min_interaction_bars: int,
) -> list[dict[str, Any]]:
    """Diagnostica: tutte le zone considerate dal layer screener."""
    if closed_data.empty:
        return []

    last_close = float(closed_data.iloc[-1]["close"])
    rows: list[dict[str, Any]] = []

    for idx, z in enumerate(balance.get("zones", []), start=1):
        m = active_metrics(
            closed_data,
            z,
            require_last_close_inside,
            interaction_window,
            min_interaction_bars,
        )
        rows.append({
            "Strumento": label,
            "Ticker": ticker,
            "Zona #": idx,
            "Motore": "Structural A" if str(z.engine).upper() == "A" else "Reaction B",
            "Area attiva": "SI" if m["active"] else "NO",
            "Ultimo Close": last_close,
            "Balance": z.center,
            "Zona min": z.center - z.half,
            "Zona max": z.center + z.half,
            "Tocchi": f"{int(m['touches'])}/{int(interaction_window)}",
            "Sequenza tocchi": " ".join("SI" if x else "NO" for x in m["touch_flags"]),
            "Ultimo Close dentro": "SI" if m["inside"] else "NO",
            "Score": m["score"],
            "ST": z.strength,
            "H": z.hits,
            "T": z.independent_tests,
            "R %": z.reliability,
            "Successi indipendenti": z.independent_successes,
            "Break indipendenti": z.independent_breaks,
            "Support H": z.support_hits,
            "Resistance H": z.resistance_hits,
            "Dwell": z.dwell,
            "Last Hit Age": z.last_hit_age,
        })
    return rows


# =============================================================================
# CHART / EXCEL
# =============================================================================
def plot_balance(
    closed_data: pd.DataFrame,
    ticker: str,
    active_center: float,
    active_bottom: float,
    active_top: float,
    engine_label: str,
) -> go.Figure:
    chart = closed_data.tail(220)
    fig = go.Figure(data=[go.Candlestick(
        x=chart.index,
        open=chart["open"],
        high=chart["high"],
        low=chart["low"],
        close=chart["close"],
        name=ticker,
    )])

    last_close = float(closed_data.iloc[-1]["close"])
    if str(engine_label).upper().startswith("REACTION"):
        current_role = "REACTION B"
        line_color = "rgba(135, 135, 135, 1.0)"
        fill_color = "rgba(135, 135, 135, 0.14)"
    elif float(active_bottom) <= last_close <= float(active_top):
        current_role = "IN"
        line_color = "rgba(205, 165, 45, 1.0)"
        fill_color = "rgba(205, 165, 45, 0.20)"
    elif float(active_center) < last_close:
        current_role = "SUPPORTO"
        line_color = "rgba(0, 150, 90, 1.0)"
        fill_color = "rgba(0, 150, 90, 0.20)"
    else:
        current_role = "RESISTENZA"
        line_color = "rgba(235, 105, 25, 1.0)"
        fill_color = "rgba(235, 105, 25, 0.20)"

    fig.add_hrect(
        y0=float(active_bottom),
        y1=float(active_top),
        fillcolor=fill_color,
        line_width=0,
        annotation_text=f"AREA ATTIVA · {engine_label} · {current_role}",
        annotation_position="top left",
    )
    fig.add_hline(y=float(active_center), line_width=3, line_color=line_color)

    fig.update_layout(
        title=f"{ticker} · Daily chiusa · {engine_label} · {current_role}",
        height=650,
        xaxis_rangeslider_visible=False,
        showlegend=False,
        margin=dict(l=20, r=20, t=65, b=20),
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
        header = wb.add_format({
            "bold": True,
            "bg_color": "#1F4E78",
            "font_color": "#FFFFFF",
            "border": 1,
            "align": "center",
        })
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
                width = min(30, max(11, len(str(col)) + 2))
                if col in {"Strumento", "Ticker", "Motore", "Area", "Tocchi"}:
                    width = max(width, 16)
                fmt = None
                if col in {"Ultimo Close", "Balance", "Zona min", "Zona max"}:
                    fmt = n4
                elif col in {"ST", "R %", "Score"}:
                    fmt = n2
                ws.set_column(j, j, width, fmt)
            if len(df) and "Area attiva" in df.columns:
                c = df.columns.get_loc("Area attiva")
                ws.conditional_format(
                    1, c, len(df), c,
                    {"type": "text", "criteria": "containing", "value": "SI", "format": active_fmt},
                )
    return out.getvalue()


def scan_signature(
    universe: list[tuple[str, str]],
    market: str,
    adjusted: bool,
    require_last_close_inside: bool,
    interaction_window: int,
    min_interaction_bars: int,
    engine_mode: str,
) -> str:
    raw = "|".join([
        BUILD,
        market,
        str(bool(adjusted)),
        str(bool(require_last_close_inside)),
        str(int(interaction_window)),
        str(int(min_interaction_bars)),
        str(engine_mode),
    ] + [t for _, t in universe])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# =============================================================================
# UI — REBUILD
# =============================================================================
with st.sidebar:
    st.header("Impostazioni")
    st.caption(f"Build {BUILD}")
    st.caption(f"Motore: {ENGINE_REFERENCE}")

    market_choice = st.selectbox(
        "Mercato lista",
        ["Automatico", "Italia", "USA", "Misto / ticker Yahoo completi"],
        index=1,
        key="market_choice_v46_rebuild",
        help="Automatico: riconosce i file italiani con ticker .MI e i file USA con ticker standard.",
    )

    engine_mode = st.selectbox(
        "Motore",
        ["A+B", "Solo A"],
        index=0,
        help=(
            "A+B: il motore calcola A e B separatamente e lo screener considera A + le Reaction B "
            "non sovrapposte ad A. Solo A: il motore B viene comunque calcolato come nel riferimento, "
            "ma il layer screener considera soltanto Structural A."
        ),
    )

    adjusted = st.checkbox(
        "Prezzi Yahoo adjusted",
        value=False,
        help="Per confronti con TradingView usa la stessa impostazione di aggiustamento del grafico.",
    )

    require_last_close_inside = st.checkbox(
        "Richiedi ultimo Close Daily dentro la Balance",
        value=False,
        help=(
            "OFF: bastano i tocchi richiesti sulle Daily chiuse. "
            "ON: richiede anche che l'ultimo Close Daily chiuso sia dentro la stessa Balance."
        ),
    )

    min_interaction_bars = st.number_input(
        "Tocchi minimi",
        min_value=1,
        max_value=10,
        value=DEFAULT_MIN_INTERACTION_BARS,
        step=1,
    )

    interaction_window = st.number_input(
        "Finestra candele",
        min_value=2,
        max_value=10,
        value=DEFAULT_INTERACTION_WINDOW,
        step=1,
        help="I tocchi vengono contati soltanto sulle Daily completamente chiuse.",
    )

    uploaded = st.file_uploader("File ticker .txt", type=["txt", "csv"])
    use_manual = st.checkbox(
        "Modifica/incolla ticker manualmente",
        value=False,
        key="use_manual_v46_rebuild",
    )

    if use_manual:
        if market_choice in {"Automatico", "Italia"}:
            default_text = italy_example()
        elif market_choice == "USA":
            default_text = usa_example()
        else:
            default_text = "# Ticker oppure Nome;Ticker Yahoo\n"
        manual_text = st.text_area(
            "Ticker",
            default_text,
            height=320,
            key="ticker_manual_v46_rebuild",
        )
    else:
        manual_text = ""

    run = st.button(
        "🔎 Cerca aree Balance attive",
        type="primary",
        use_container_width=True,
    )


source_name = ""
if uploaded is not None and not use_manual:
    source_name = getattr(uploaded, "name", "")
    try:
        text = uploaded.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError:
        text = uploaded.getvalue().decode("latin-1")
elif use_manual:
    text = manual_text
else:
    if market_choice in {"Automatico", "Italia"}:
        text = italy_example()
    elif market_choice == "USA":
        text = usa_example()
    else:
        text = ""

market = infer_market_from_text(text, source_name) if market_choice == "Automatico" else market_choice
universe = parse_tickers(text, market)

current_signature = scan_signature(
    universe,
    market,
    adjusted,
    require_last_close_inside,
    int(interaction_window),
    int(min_interaction_bars),
    engine_mode,
)

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


SESSION_KEY = "balance_stock_screener_v4_6_rebuild"

if run:
    if not universe:
        st.error("Nessun ticker valido nel file/elenco.")
        st.stop()

    st.session_state.pop(SESSION_KEY, None)

    labels = {ticker: label for label, ticker in universe}
    tickers = [ticker for _, ticker in universe]

    progress = st.progress(0.0, text="Avvio scansione…")
    data_map, notes = load_universe_data(tickers, adjusted=adjusted, progress=progress)

    data_map, notes, refreshed_stale = refresh_stale_daily_data(
        data_map,
        tickers,
        adjusted=adjusted,
        market=market,
        notes=notes,
        progress=progress,
    )

    data_map, notes, refreshed_fallback, still_stale = refresh_stale_italy_from_public_fallbacks(
        data_map,
        tickers,
        adjusted=adjusted,
        market=market,
        notes=notes,
        progress=progress,
    )
    stale_set = set(still_stale)

    active_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    details: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], str]] = {}
    errors: list[dict[str, str]] = []

    for i, ticker in enumerate(tickers, start=1):
        progress.progress(
            0.48 + 0.52 * ((i - 1) / max(1, len(tickers))),
            text=f"Balance {i}/{len(tickers)} · {ticker}",
        )

        if ticker in stale_set:
            errors.append({
                "Ticker": ticker,
                "Errore": "Daily arretrata: le sorgenti pubbliche non hanno fornito l'ultima seduta chiusa attesa.",
            })
            continue

        engine_data = _normalize_ohlc(data_map.get(ticker, pd.DataFrame()))
        if engine_data.empty or len(engine_data) < 120:
            errors.append({"Ticker": ticker, "Errore": notes.get(ticker, "Dati insufficienti")})
            continue

        # Il layer Active Area usa SOLO barre chiuse.
        closed_data, _open_bar_removed = only_closed_daily(engine_data, ticker, market)
        if closed_data.empty or len(closed_data) < 120:
            errors.append({"Ticker": ticker, "Errore": "Daily chiuse insufficienti."})
            continue

        try:
            # Il motore riceve la serie completa disponibile, non la serie già
            # tagliata dal layer screener. Questo replica la separazione Pine
            # motore Balance / classificazione stabile.
            balance = analyze_balance_zones(
                engine_data,
                engine_mode=engine_mode,
                lookback=LOOKBACK,
            )

            if not balance.get("available"):
                errors.append({
                    "Ticker": ticker,
                    "Errore": str(balance.get("detail", "Balance non disponibili")),
                })
                continue

            row = active_zone_row(
                labels[ticker],
                ticker,
                closed_data,
                balance,
                require_last_close_inside,
                int(interaction_window),
                int(min_interaction_bars),
            )
            if row is not None:
                active_rows.append(row)

            all_rows.extend(all_balance_rows(
                labels[ticker],
                ticker,
                closed_data,
                balance,
                require_last_close_inside,
                int(interaction_window),
                int(min_interaction_bars),
            ))

            details[ticker] = (engine_data, closed_data, balance, labels[ticker])

        except Exception as exc:
            errors.append({"Ticker": ticker, "Errore": f"{type(exc).__name__}: {exc}"})

    progress.progress(1.0, text="Completato")

    st.session_state[SESSION_KEY] = {
        "signature": current_signature,
        "active_rows": active_rows,
        "all_rows": all_rows,
        "details": details,
        "errors": errors,
        "adjusted": bool(adjusted),
        "require_last_close_inside": bool(require_last_close_inside),
        "interaction_window": int(interaction_window),
        "min_interaction_bars": int(min_interaction_bars),
        "engine_mode": engine_mode,
        "total_tickers": len(tickers),
        "downloaded_tickers": len(data_map),
        "refreshed_stale": int(refreshed_stale),
        "refreshed_fallback": int(refreshed_fallback),
        "still_stale": list(still_stale),
    }


payload = st.session_state.get(SESSION_KEY)

if payload and payload.get("signature") == current_signature:
    active = pd.DataFrame(payload["active_rows"])
    all_zones = pd.DataFrame(payload["all_rows"])
    details = payload["details"]
    errors = payload["errors"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Titoli con Balance calcolate", len(details))
    c2.metric("Aree attive · Daily chiuse", len(active))
    c3.metric("Ticker senza risultato", len(errors))

    refreshed_stale = int(payload.get("refreshed_stale", 0))
    refreshed_fallback = int(payload.get("refreshed_fallback", 0))
    still_stale = list(payload.get("still_stale", []))

    if refreshed_stale > 0:
        st.caption(
            f"Freschezza dati: {refreshed_stale} ticker arretrati nel batch "
            "sono stati aggiornati con retry individuale Yahoo."
        )
    if refreshed_fallback > 0:
        st.caption(
            f"Freschezza Italia: {refreshed_fallback} ticker sono stati completati "
            "con una sorgente Daily alternativa."
        )
    if still_stale:
        st.warning(
            f"Dati Daily ancora arretrati per {len(still_stale)} ticker: "
            "esclusi dallo screening per evitare segnali su dati vecchi."
        )

    st.subheader("Aree Balance attive")

    if active.empty:
        suffix = " e l'ultimo Close deve essere dentro la stessa Balance" if require_last_close_inside else ""
        st.info(
            f"Nessuna AREA ATTIVA: servono almeno {int(min_interaction_bars)} tocchi reali "
            f"nelle ultime {int(interaction_window)} Daily chiuse{suffix}."
        )
        active_view = active.copy()
    else:
        active_view = active.sort_values(
            ["Score", "ST", "Strumento"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        visible = [
            "Strumento",
            "Ticker",
            "Motore",
            "Data ultima Daily chiusa",
            "Area",
            "Ultimo Close",
            "Balance",
            "Zona min",
            "Zona max",
            "Tocchi",
            "Sequenza tocchi",
            "Score",
            "ST",
            "H",
            "T",
            "R %",
        ]

        st.dataframe(
            active_view[visible],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Data ultima Daily chiusa": st.column_config.TextColumn("Ultima Daily"),
                "Ultimo Close": st.column_config.NumberColumn("Ultimo Close", format="%.4f"),
                "Balance": st.column_config.NumberColumn("Balance", format="%.4f"),
                "Zona min": st.column_config.NumberColumn("Zona min", format="%.4f"),
                "Zona max": st.column_config.NumberColumn("Zona max", format="%.4f"),
                "Score": st.column_config.NumberColumn("Score", format="%.1f"),
                "ST": st.column_config.NumberColumn("ST", format="%.1f"),
                "R %": st.column_config.NumberColumn("R %", format="%.1f"),
            },
        )

    active_export = active.copy()
    all_export = all_zones.copy()
    xlsx = excel_bytes(active_export, all_export, errors)

    st.download_button(
        "⬇️ Esporta Excel",
        data=xlsx,
        file_name="balance_stock_screener_v4_6.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    if not active_view.empty:
        st.subheader("Verifica grafica")
        active_view = active_view.copy()
        active_view["_select"] = (
            active_view["Ticker"].astype(str)
            + " | "
            + active_view["Motore"].astype(str)
            + " | "
            + active_view["Balance"].map(lambda x: f"{x:.4f}")
        )

        selected_key = st.selectbox(
            "Area attiva",
            active_view["_select"].tolist(),
            key="balance_stock_active_chart_v46",
        )
        selected_row = active_view[active_view["_select"] == selected_key].iloc[0]
        ticker = str(selected_row["Ticker"])
        _engine_data, closed_data, _balance, _label = details[ticker]

        st.plotly_chart(
            plot_balance(
                closed_data,
                ticker,
                float(selected_row["Balance"]),
                float(selected_row["Zona min"]),
                float(selected_row["Zona max"]),
                str(selected_row["Motore"]),
            ),
            use_container_width=True,
        )

        a, b, c, d, e, f = st.columns(6)
        a.metric("Motore", str(selected_row["Motore"]))
        b.metric("Ultima Daily", str(selected_row["Data ultima Daily chiusa"]))
        c.metric("Ultimo Close", f"{float(selected_row['Ultimo Close']):.4f}")
        d.metric("Balance", f"{float(selected_row['Balance']):.4f}")
        e.metric("Tocchi", str(selected_row["Tocchi"]))
        f.metric("Score", f"{float(selected_row['Score']):.1f}")

    with st.expander("Tutte le Balance calcolate · diagnostica"):
        if all_zones.empty:
            st.info("Nessuna Balance disponibile.")
        else:
            st.dataframe(all_zones, hide_index=True, use_container_width=True)

    if errors:
        with st.expander(f"Ticker senza risultato ({len(errors)})"):
            st.dataframe(pd.DataFrame(errors), hide_index=True, use_container_width=True)
