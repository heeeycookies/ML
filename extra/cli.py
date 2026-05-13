# cli.py
from __future__ import annotations

import os
import argparse
import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient

from adapters import fetch_assets, fetch_bars_iex_chunked
from core import (
    compute_features, apply_gates,
    select_universe_with_injection, labels_view,
    FEATURE_COLS, LABEL_COLS
)


# cli.py (top)
DEFAULTS = dict(
    exchanges=("NYSE", "NASDAQ"),
    only_tradable=True,
    include_etfs=False,
    min_price=10.0,
    avg_window=20,
    lookback_days=252 * 5,
    max_symbols=4000,
    top_n=20,
    chunk_size=200,
    chunk_pause_sec=0.3,
    # Realism knobs
    min_adv_usd=1_000_000.0,
    continuity_min_traded=18,
    trail_return_days=126,
    vol_window_days=20,
    # randomized bad fraction per snapshot
    inject_frac_bad_min=0.20,
    inject_frac_bad_max=0.25,
    bad_return_pctile=0.25,   # <-- add
    bad_vol_pctile=0.75,      # <-- add
    random_seed=42,
    # Snapshot cadence
    rebalance_every_days=30,
    # Bars cache
    cache_path="data/cached_bars.parquet",
)


def _write_empty(parquet_path: str, csv_path: str) -> None:
    os.makedirs(os.path.dirname(parquet_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    pd.DataFrame(columns=FEATURE_COLS).to_parquet(parquet_path, index=False)
    pd.DataFrame(columns=LABEL_COLS).to_csv(csv_path, index=False)
    print("[BlueChipUniverse] No assets/bars available. Wrote empty outputs.")


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a realistic blue-chip universe and labels.")
    p.add_argument("--out-parquet", type=str, default="data/universe.parquet",
                   help="Path to the output Parquet file (features).")
    p.add_argument("--out-csv", type=str, default=None,
                   help="Path to the output CSV file (labels). Defaults to '<parquet_base>_labels.csv'.")
    p.add_argument("--top-n", type=int, default=None,
                   help="Override the universe size (e.g., 150, 300).")
    p.add_argument("--no-cache", action="store_true",
                   help="Force fresh download of bars (ignore cache).")
    return p.parse_args()


def main() -> None:
    args = _parse_cli()

    parquet_path = args.out_parquet
    csv_path = args.out_csv or os.path.splitext(parquet_path)[0] + "_labels.csv"

    # Load API keys
    load_dotenv()
    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError("Missing Alpaca keys. Put APCA_API_KEY_ID and APCA_API_SECRET_KEY in a .env file.")

    # Build clients
    trading = TradingClient(api_key, api_secret, paper=True)
    data = StockHistoricalDataClient(api_key, api_secret)

    # ---------------- Caching logic ----------------
    cache_path = DEFAULTS["cache_path"]
    use_cache = os.path.exists(cache_path) and not args.no_cache

    if use_cache:
        print(f"[BlueChipUniverse] Using cached bars from {cache_path}")
        bars = pd.read_parquet(cache_path)
        bars = bars.set_index(["symbol", "ts"]).sort_index()
    else:
        # Fetch fresh assets
        assets_df = fetch_assets(
            trading_client=trading,
            exchanges=DEFAULTS["exchanges"],
            only_tradable=DEFAULTS["only_tradable"],
            include_etfs=DEFAULTS["include_etfs"],
            max_symbols=DEFAULTS["max_symbols"],
        )
        print(f"[BlueChipUniverse] Fetched {len(assets_df):,} candidate assets from Alpaca.")
        if assets_df.empty:
            _write_empty(parquet_path, csv_path)
            return

        # Fetch fresh bars
        bars = fetch_bars_iex_chunked(
            data_client=data,
            symbols=assets_df["symbol"].tolist(),
            lookback_days=DEFAULTS["lookback_days"],
            chunk_size=DEFAULTS["chunk_size"],
            chunk_pause_sec=DEFAULTS["chunk_pause_sec"],
        )
        fetched = bars.index.get_level_values(0).nunique() if not bars.empty else 0
        print(f"[BlueChipUniverse] Downloaded bars for {fetched:,} symbols.")

        if bars.empty:
            _write_empty(parquet_path, csv_path)
            return

        # Save cache
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        bars.reset_index().to_parquet(cache_path, index=False)
        print(f"[BlueChipUniverse] Cached bars saved to {cache_path}")

    # ---------------- Feature Engineering ----------------
    top_n = int(args.top_n) if args.top_n is not None else int(DEFAULTS["top_n"])

    feats = compute_features(
        bars,
        avg_window=DEFAULTS["avg_window"],
        trail_return_days=DEFAULTS["trail_return_days"],
        vol_window_days=DEFAULTS["vol_window_days"],
    )
    gated = apply_gates(
        feats,
        min_price=DEFAULTS["min_price"],
        min_adv_usd=DEFAULTS["min_adv_usd"],
        continuity_min_traded=DEFAULTS["continuity_min_traded"],
    )

    # We still need assets for metadata (name, exchange)
    assets_df = fetch_assets(
        trading_client=trading,
        exchanges=DEFAULTS["exchanges"],
        only_tradable=DEFAULTS["only_tradable"],
        include_etfs=DEFAULTS["include_etfs"],
        max_symbols=DEFAULTS["max_symbols"],
    )

    final_df = select_universe_with_injection(
        assets_df, gated,
        top_n=top_n,
        inject_frac_bad_min=DEFAULTS["inject_frac_bad_min"],
        inject_frac_bad_max=DEFAULTS["inject_frac_bad_max"],
        bad_return_pctile=DEFAULTS["bad_return_pctile"],   # <-- exists now
        bad_vol_pctile=DEFAULTS["bad_vol_pctile"],         # <-- exists now
        random_seed=DEFAULTS["random_seed"],
)
    
    print(f"[BlueChipUniverse] Selected {len(final_df):,} names across snapshots.")

    # ---------------- Save Outputs ----------------
    os.makedirs(os.path.dirname(parquet_path) or ".", exist_ok=True)
    final_df.to_parquet(parquet_path, index=False)
    print(f"[BlueChipUniverse] Saved features to {parquet_path}")

    labels_df = labels_view(final_df)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    labels_df.to_csv(csv_path, index=False)
    print(f"[BlueChipUniverse] Saved labels to {csv_path} | Label counts: {labels_df['label'].value_counts(dropna=False).to_dict()}")


if __name__ == "__main__":
    main()