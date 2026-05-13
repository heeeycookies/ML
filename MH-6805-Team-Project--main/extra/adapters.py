# adapters.py
from __future__ import annotations
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from tqdm import tqdm  

from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data import DataFeed


def fetch_assets(
    trading_client,
    *,
    exchanges: Tuple[str, ...],
    only_tradable: bool,
    include_etfs: bool,
    max_symbols: int,
) -> pd.DataFrame:
    """
    Fetch active US equities; filter by exchanges/tradability; exclude ETFs by name heuristic.
    Returns [symbol, name, exchange] up to `max_symbols`.
    """
    req = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
    assets = list(trading_client.get_all_assets(req))

    def looks_like_etf(name: str) -> bool:
        nm = (name or "").upper()
        return any(tag in nm for tag in (" ETF", " ETN", " TRUST", " FUND ", " INDEX "))

    rows: List[Dict] = []
    for a in assets:
        if only_tradable and not a.tradable:
            continue
        if a.exchange not in exchanges:
            continue
        sym = a.symbol.upper()
        nm = (a.name or "").strip()
        if not include_etfs and looks_like_etf(nm):
            continue
        rows.append({"symbol": sym, "name": nm, "exchange": a.exchange})

    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["symbol"])
        .sort_values("symbol")
        .head(max_symbols)
    )


def fetch_bars_iex_chunked(
    data_client,
    symbols: Sequence[str],
    *,
    lookback_days: int,
    chunk_size: int,
    chunk_pause_sec: float,
) -> pd.DataFrame:
    """
    Download daily bars via IEX feed (free tier friendly), chunked to respect rate limits.
    Input: list of symbols. Output: MultiIndex DataFrame indexed by (symbol, ts).
    """
    if not symbols:
        return pd.DataFrame()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(lookback_days * 1.1))  # buffer for holidays
    frames: List[pd.DataFrame] = []
    chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]

    # ✅ tqdm progress bar for chunks
    for idx, chunk in enumerate(tqdm(chunks, desc="Downloading bars (IEX)", unit="chunk")):
        req = StockBarsRequest(
            symbol_or_symbols=list(chunk),
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        try:
            resp = data_client.get_stock_bars(req)
        except Exception:
            time.sleep(chunk_pause_sec)
            continue

        if not getattr(resp, "data", None):
            time.sleep(chunk_pause_sec)
            continue

        for sym, bars in resp.data.items():
            if not bars:
                continue
            frames.append(pd.DataFrame([{
                "symbol": sym,
                "ts": b.timestamp,
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": int(b.volume or 0),
            } for b in bars]))

        time.sleep(chunk_pause_sec)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["symbol", "ts"]).set_index(["symbol", "ts"])