
# bluechip_200_universe.py
from __future__ import annotations

import os
import sys
import time
import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv  # pip install python-dotenv

# Alpaca SDK
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data import DataFeed


class BlueChip200Universe:
    """
    Build a realistic 'blue-chip-like' US equities universe using Alpaca IEX data.
    - Core selection: liquidity (20D avg $ volume), price floor, continuity (>=18/20 traded days)
    - Challenge set: inject ~10% low-momentum, high-volatility names to avoid selection bias
    - Outputs:
        Parquet: full feature table (diagnostics included)
        CSV:     compact labels (good vs bad), derived from `selected_bucket`
    """

    # --- Output schemas (kept identical) ---
    _FEATURE_COLS = [
        "symbol", "name", "exchange",
        "last_close", "avg_volume", "avg_dollar_vol",
        "ret_trail", "vol_trail",
        "traded_days_lookback", "last20_traded",
        "adv_ok", "continuity_ok",
        "selected_bucket", "as_of_utc",
    ]
    _LABEL_COLS = ["symbol", "name", "exchange", "selected_bucket", "label"]

    def __init__(
        self,
        *,
        exchanges: Tuple[str, ...] = ("NYSE", "NASDAQ"),
        only_tradable: bool = True,
        include_etfs: bool = False,
        min_price: float = 10.0,
        avg_window: int = 20,
        lookback_days: int = 252,
        max_symbols: int = 4000,
        top_n: int = 500,                 # default universe size (override via CLI --top-n or run(..., top_n_override))
        chunk_size: int = 200,
        chunk_pause_sec: float = 0.25,
        paper: bool = True,
        verbose: bool = True,
        # Realism knobs
        min_adv_usd: float = 1_000_000.0,  # 20D avg $ volume floor
        continuity_min_traded: int = 18,   # >= this many traded days in the last 20
        trail_return_days: int = 126,      # momentum lookback (~6M)
        vol_window_days: int = 20,         # realized vol window
        inject_frac_bad: float = 0.10,     # ~10% injected "bad" names
        bad_return_pctile: float = 0.25,   # bottom 25% momentum
        bad_vol_pctile: float = 0.75,      # top 25% realized vol
        random_seed: int = 42,
    ) -> None:
        load_dotenv()
        api_key = os.getenv("APCA_API_KEY_ID")
        api_secret = os.getenv("APCA_API_SECRET_KEY")
        if not api_key or not api_secret:
            raise RuntimeError("Missing Alpaca keys. Put APCA_API_KEY_ID and APCA_API_SECRET_KEY in a .env file next to this script.")

        # Stored params
        self.exchanges = exchanges
        self.only_tradable = only_tradable
        self.include_etfs = include_etfs
        self.min_price = float(min_price)
        self.avg_window = int(avg_window)
        self.lookback_days = int(lookback_days)
        self.max_symbols = int(max_symbols)
        self.top_n = int(top_n)
        self.chunk_size = int(chunk_size)
        self.chunk_pause_sec = float(chunk_pause_sec)
        self.paper = paper
        self.verbose = verbose

        self.min_adv_usd = float(min_adv_usd)
        self.continuity_min_traded = int(continuity_min_traded)
        self.trail_return_days = int(trail_return_days)
        self.vol_window_days = int(vol_window_days)
        self.inject_frac_bad = float(inject_frac_bad)
        self.bad_return_pctile = float(bad_return_pctile)
        self.bad_vol_pctile = float(bad_vol_pctile)
        self.random_seed = int(random_seed)
        np.random.seed(self.random_seed)

        # Alpaca clients
        self.trading = TradingClient(api_key, api_secret, paper=self.paper)
        self.data = StockHistoricalDataClient(api_key, api_secret)

    # ---------------- Public API ----------------

    def run(self, parquet_path: str, csv_path: Optional[str] = None, top_n_override: Optional[int] = None) -> pd.DataFrame:
        """
        Build universe and write outputs.
        Args:
            parquet_path: Required. Full features (Parquet).
            csv_path:     Optional. Labels CSV path. Defaults to '<parquet_base>_labels.csv'.
            top_n_override: Optional. Override universe size at runtime.
        """
        if top_n_override is not None:
            self._say(f"Overriding top_n: {self.top_n} -> {top_n_override}")
            self.top_n = int(top_n_override)

        if csv_path is None:
            root, _ = os.path.splitext(parquet_path)
            csv_path = f"{root}_labels.csv"

        self._say("Starting universe build…")

        # 1) Candidate assets
        assets_df = self._fetch_assets()
        self._say(f"Fetched {len(assets_df):,} candidate assets from Alpaca.")
        if assets_df.empty:
            return self._write_empty_both(parquet_path, csv_path)

        # 2) Historical bars (IEX feed; chunked for free-plan limits)
        bars = self._fetch_bars_iex_chunked(assets_df["symbol"].tolist())
        fetched = bars.index.get_level_values(0).nunique() if not bars.empty else 0
        self._say(f"Downloaded bars for {fetched:,} symbols.")
        if bars.empty:
            return self._write_empty_both(parquet_path, csv_path)

        # 3) Rank/select, inject challengers
        final_df = self._rank_select_and_inject(assets_df, bars)
        self._say(f"Selected {len(final_df):,} names (core + injected_bad + topups).")

        # 4) Write outputs
        os.makedirs(os.path.dirname(parquet_path) or ".", exist_ok=True)
        final_df.to_parquet(parquet_path, index=False)
        self._say(f"Saved features to {parquet_path}")

        labels_df = self._labels_view(final_df)
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        labels_df.to_csv(csv_path, index=False)
        self._say(f"Saved labels to {csv_path} | Label counts: {labels_df['label'].value_counts(dropna=False).to_dict()}")

        return final_df

    # ---------------- Internals ----------------

    def _fetch_assets(self) -> pd.DataFrame:
        """
        Pull active US equities; filter by exchange/tradability; exclude ETFs/ETNs/trusts/funds by name heuristic.
        Returns a de-duplicated DataFrame: [symbol, name, exchange] capped at `max_symbols`.
        """
        req = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
        assets = list(self.trading.get_all_assets(req))

        def looks_like_etf(name: str) -> bool:
            nm = (name or "").upper()
            return any(tag in nm for tag in (" ETF", " ETN", " TRUST", " FUND ", " INDEX "))

        rows: List[Dict] = []
        for a in assets:
            if self.only_tradable and not a.tradable:
                continue
            if a.exchange not in self.exchanges:
                continue
            sym = a.symbol.upper()
            nm = (a.name or "").strip()
            if not self.include_etfs and looks_like_etf(nm):
                continue
            rows.append({"symbol": sym, "name": nm, "exchange": a.exchange})

        return (
            pd.DataFrame(rows)
            .drop_duplicates(subset=["symbol"])
            .sort_values("symbol")
            .head(self.max_symbols)
        )

    def _fetch_bars_iex_chunked(self, symbols: Sequence[str]) -> pd.DataFrame:
        """
        Download daily bars via IEX feed (free tier friendly), chunked to respect rate limits.
        Returns a MultiIndex DataFrame indexed by (symbol, ts).
        """
        if not symbols:
            return pd.DataFrame()

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=int(self.lookback_days * 1.1))  # buffer for holidays
        frames: List[pd.DataFrame] = []
        chunks = [symbols[i:i + self.chunk_size] for i in range(0, len(symbols), self.chunk_size)]

        for idx, chunk in enumerate(chunks, 1):
            self._say(f"Fetching bars chunk {idx}/{len(chunks)} (IEX)…")
            req = StockBarsRequest(
                symbol_or_symbols=list(chunk),
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                feed=DataFeed.IEX,
            )
            try:
                resp = self.data.get_stock_bars(req)
            except Exception as e:
                self._say(f"Warning: bars request failed for chunk {idx}: {e}")
                time.sleep(self.chunk_pause_sec)
                continue

            if not getattr(resp, "data", None):
                time.sleep(self.chunk_pause_sec)
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
            time.sleep(self.chunk_pause_sec)

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True)
        return out.sort_values(["symbol", "ts"]).set_index(["symbol", "ts"])

    # ---- Feature helpers ----

    @staticmethod
    def _pct_change(s: pd.Series, periods: int) -> float:
        """Simple trailing return over `periods` days; NaN if insufficient history or invalid start."""
        if periods <= 0 or len(s) <= periods:
            return np.nan
        start, end = s.iloc[-periods - 1], s.iloc[-1]
        return float(end / start - 1.0) if start > 0 else np.nan

    @staticmethod
    def _realized_vol(s: pd.Series, window: int) -> float:
        """Annualized realized volatility from daily log-returns over `window` days."""
        if len(s) < window + 1:
            return np.nan
        lr = np.log(s).diff().dropna()
        std = lr.tail(window).std(ddof=0)
        return float(std * np.sqrt(252)) if pd.notna(std) else np.nan

    # ---- Core selection + "bad" injection ----

    def _rank_select_and_inject(self, assets_df: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
        """
        1) Compute features & realism gates
        2) Take top by liquidity as core
        3) Inject ~10% 'bad' names (low momentum, high vol)
        4) Top up by liquidity if short
        """
        g = bars.groupby(level=0, sort=False)

        # --- Per-symbol features (last price, liquidity, continuity, momentum, risk) ---
        last_close = g["close"].last().rename("last_close")
        traded_days = g["close"].count().rename("traded_days_lookback")

        def tail_mean(s: pd.Series, n: int) -> float:
            return float(s.tail(n).mean())

        avg_volume = g["volume"].apply(lambda s: tail_mean(s, self.avg_window)).rename("avg_volume")
        avg_dollar_vol = g.apply(lambda df: float((df["close"] * df["volume"]).tail(self.avg_window).mean())).rename("avg_dollar_vol")
        last20_traded = g["volume"].apply(lambda s: int((s.tail(self.avg_window) > 0).sum())).rename("last20_traded")
        ret_trail = g["close"].apply(lambda s: self._pct_change(s, self.trail_return_days)).rename("ret_trail")
        vol_trail = g["close"].apply(lambda s: self._realized_vol(s, self.vol_window_days)).rename("vol_trail")

        metrics = pd.concat(
            [last_close, avg_volume, avg_dollar_vol, traded_days, last20_traded, ret_trail, vol_trail],
            axis=1
        ).reset_index()

        # --- Realism gates ---
        metrics = metrics.loc[metrics["last_close"] >= self.min_price].copy()
        metrics["adv_ok"] = metrics["avg_dollar_vol"] >= self.min_adv_usd
        metrics["continuity_ok"] = metrics["last20_traded"] >= self.continuity_min_traded

        # Join static attributes
        merged = metrics.merge(assets_df, how="left", on="symbol")

        # --- Core selection: rank by liquidity among passers ---
        core_pool = merged[(merged["adv_ok"]) & (merged["continuity_ok"])].copy()
        core_pool = core_pool.sort_values("avg_dollar_vol", ascending=False)

        core_target = max(int(round(self.top_n * (1.0 - self.inject_frac_bad))), 1)
        core_sel = core_pool.head(core_target).copy()
        core_sel["selected_bucket"] = "core"

        # --- Challenger pool: from remainder, filter low-mom & high-vol ---
        remainder = merged[~merged["symbol"].isin(core_sel["symbol"])].copy()
        ret_thresh = remainder["ret_trail"].quantile(self.bad_return_pctile, interpolation="linear")
        vol_thresh = remainder["vol_trail"].quantile(self.bad_vol_pctile, interpolation="linear")

        bad_pool = remainder[
            remainder["ret_trail"].notna() & remainder["vol_trail"].notna()
            & (remainder["ret_trail"] <= ret_thresh) & (remainder["vol_trail"] >= vol_thresh)
        ].copy()

        inject_n = max(min(int(round(self.top_n * self.inject_frac_bad)), self.top_n - len(core_sel)), 0)
        if inject_n > 0 and not bad_pool.empty:
            # Prefer continuity; allow lower ADV to keep challengers realistic
            bad_pool = bad_pool.sort_values(["continuity_ok", "avg_dollar_vol"], ascending=[False, True])
            injected = bad_pool.sample(n=min(inject_n, len(bad_pool)), replace=False, random_state=self.random_seed).copy()
            injected["selected_bucket"] = "injected_bad"
        else:
            injected = pd.DataFrame(columns=merged.columns.tolist() + ["selected_bucket"])

        final = pd.concat([core_sel, injected], ignore_index=True)

        # --- Top up by liquidity if still short ---
        if len(final) < self.top_n:
            need = self.top_n - len(final)
            topup_pool = remainder[~remainder["symbol"].isin(final["symbol"])].copy()
            topup = topup_pool.sort_values("avg_dollar_vol", ascending=False).head(need).copy()
            if not topup.empty:
                topup["selected_bucket"] = np.where(
                    topup["adv_ok"] & topup["continuity_ok"], "core_topup", "lenient_topup"
                )
                final = pd.concat([final, topup], ignore_index=True)

        # --- Final formatting ---
        final["as_of_utc"] = datetime.now(timezone.utc)
        final = final.sort_values(["selected_bucket", "avg_dollar_vol"], ascending=[True, False]).reset_index(drop=True)

        # Ensure all required columns exist (preserves exact output schema)
        for c in self._FEATURE_COLS:
            if c not in final.columns:
                final[c] = np.nan

        return final[self._FEATURE_COLS].head(self.top_n).reset_index(drop=True)

    # ---- Labels (good/bad) view ----

    @staticmethod
    def _labels_view(final_df: pd.DataFrame) -> pd.DataFrame:
        """
        Map `selected_bucket` to label:
            injected_bad -> "bad"
            otherwise    -> "good"
        """
        out = final_df[["symbol", "name", "exchange", "selected_bucket"]].copy()
        out["label"] = np.where(out["selected_bucket"] == "injected_bad", "bad", "good")
        return out.sort_values(["label", "symbol"]).reset_index(drop=True)

    # ---- Empty writers & logging ----

    def _write_empty(self, parquet_path: str) -> None:
        os.makedirs(os.path.dirname(parquet_path) or ".", exist_ok=True)
        pd.DataFrame(columns=self._FEATURE_COLS).to_parquet(parquet_path, index=False)

    def _write_empty_csv(self, csv_path: str) -> None:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        pd.DataFrame(columns=self._LABEL_COLS).to_csv(csv_path, index=False)

    def _write_empty_both(self, parquet_path: str, csv_path: str) -> pd.DataFrame:
        """Helper when no data is available after filters."""
        self._write_empty(parquet_path)
        self._write_empty_csv(csv_path)
        self._say("No assets/bars available. Wrote empty outputs.")
        return pd.DataFrame()

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(f"[BlueChip200Universe] {msg}")


# ---------------- CLI ----------------

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a realistic blue-chip universe and labels.")
    p.add_argument("--out-parquet", type=str, default="data/universe.parquet",
                   help="Path to the output Parquet file (features).")
    p.add_argument("--out-csv", type=str, default=None,
                   help="Path to the output CSV file (labels). Defaults to '<parquet_base>_labels.csv'.")
    p.add_argument("--top-n", type=int, default=None,
                   help="Override the universe size (e.g., 150, 300).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli()

    u = BlueChip200Universe(
        exchanges=("NYSE", "NASDAQ"),
        only_tradable=True,
        include_etfs=False,
        min_price=10.0,
        avg_window=20,
        lookback_days=252,
        max_symbols=4000,
        top_n=500,                 # default; override with --top-n
        chunk_size=200,
        chunk_pause_sec=0.3,
        verbose=True,
        
        # Realism knobs 
        min_adv_usd=1_000_000.0,
        continuity_min_traded=18,
        trail_return_days=126,
        vol_window_days=20,
        inject_frac_bad=0.10,
        bad_return_pctile=0.25,
        bad_vol_pctile=0.75,
        random_seed=42,
    )

    u.run(
        parquet_path=args.out_parquet,
        csv_path=args.out_csv,
        top_n_override=args.top_n,
    )