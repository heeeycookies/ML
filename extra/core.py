# core.py
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import pandas_ta as ta  # <— NEW: use pandas-ta for indicators


# ---------------- Schema (final outputs) ----------------

FEATURE_COLS = [
    "as_of_date", "symbol", "name", "exchange",
    # snapshot-level hygiene / liquidity
    "last_close", "adv_20d", "cont_last20",
    # momentum / trend
    "ret_5d", "ret_21d", "ret_63d",
    "macd", "macd_signal", "macd_hist",
    # mean reversion
    "bb_percent_b", "bb_bandwidth",
    # risk / volume context
    "vol_20d", "vol_63d", "atr_14", "vol_z_20d",
    # selection flags
    "adv_ok", "continuity_ok",
    "selected_bucket",
    "as_of_utc",
]

LABEL_COLS = ["as_of_date", "symbol", "name", "exchange", "selected_bucket", "label"]


# ---------------- Helpers ----------------

def _rebalance_calendar(all_ts: pd.DatetimeIndex, every_days: int = 30) -> pd.DatetimeIndex:
    """Calendar in calendar days, snapped forward to next trading day in `all_ts`."""
    cal = pd.DatetimeIndex(all_ts).sort_values().unique()
    if len(cal) == 0:
        return cal
    t = cal[0]
    snaps = []
    while t <= cal[-1]:
        idx = cal.get_indexer([t], method="backfill")
        if idx[0] != -1:
            snaps.append(cal[idx[0]])
        t = t + pd.DateOffset(days=every_days)
    return pd.DatetimeIndex(sorted(set(snaps)))


# ---------------- Daily indicators (with tqdm progress) ----------------

def _compute_daily_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Input: MultiIndex (symbol, ts) with columns: open, high, low, close, volume
    Output: same index with added daily indicator columns.
    Uses pandas-ta for indicators and tqdm over symbols to show progress & ETA.
    """
    if bars.empty:
        return bars.copy()

    df = bars.copy()
    symbols = df.index.get_level_values(0).unique()
    results = []

    for sym in tqdm(symbols, desc="Computing daily indicators (pandas-ta)", unit="symbol"):
        sub = df.xs(sym, level=0).copy()  # index = ts

        # --- Basic returns (keep as before) ---
        sub["ret_1d"]  = sub["close"].pct_change()
        sub["ret_5d"]  = sub["close"].pct_change(5)
        sub["ret_21d"] = sub["close"].pct_change(21)
        sub["ret_63d"] = sub["close"].pct_change(63)

        # --- Realized vol (annualized) ---
        sub["vol_20d"] = sub["ret_1d"].rolling(20).std(ddof=0) * np.sqrt(252)
        sub["vol_63d"] = sub["ret_1d"].rolling(63).std(ddof=0) * np.sqrt(252)

        # --- MACD (12, 26, 9) via pandas-ta ---
        macd_df = ta.macd(sub["close"], fast=12, slow=26, signal=9)
        # Columns typically: MACD_12_26_9, MACDh_12_26_9 (hist), MACDs_12_26_9 (signal)
        if macd_df is not None and not macd_df.empty:
            sub["macd"]        = macd_df.iloc[:, 0]
            sub["macd_hist"]   = macd_df.iloc[:, 1]
            sub["macd_signal"] = macd_df.iloc[:, 2]
        else:
            sub["macd"] = sub["macd_hist"] = sub["macd_signal"] = np.nan

        # --- Bollinger Bands (20, 2) via pandas-ta ---
        bb_df = ta.bbands(sub["close"], length=20, std=2)
        # Columns typically include: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0 (bandwidth), BBP_20_2.0 (%B)
        if bb_df is not None and not bb_df.empty:
            # %B and bandwidth names can vary; look up by suffix 'BBP' and 'BBB'
            bbp_col = [c for c in bb_df.columns if "BBP" in c]
            bbb_col = [c for c in bb_df.columns if "BBB" in c]
            sub["bb_percent_b"] = bb_df[bbp_col[0]] if bbp_col else np.nan
            sub["bb_bandwidth"] = bb_df[bbb_col[0]] if bbb_col else np.nan
        else:
            sub["bb_percent_b"] = sub["bb_bandwidth"] = np.nan

        # --- ATR(14) via pandas-ta ---
        atr_ser = ta.atr(high=sub["high"], low=sub["low"], close=sub["close"], length=14)
        # ta.atr returns a Series named like "ATRr_14" (rma smoothing). Use values regardless of name.
        sub["atr_14"] = atr_ser if atr_ser is not None else np.nan

        # --- Volume z-score (20d) via pandas-ta (zscore) ---
        vz = ta.zscore(sub["volume"], length=20)
        sub["vol_z_20d"] = vz if vz is not None else np.nan

        # --- ADV & continuity (custom, simple) ---
        sub["adv_20d"] = (sub["close"] * sub["volume"]).rolling(20).mean()
        sub["cont_last20"] = sub["volume"].rolling(20).apply(lambda s: float((s > 0).sum()), raw=False)

        # carry symbol for reconstruction
        sub["symbol"] = sym
        results.append(sub.reset_index())  # columns: ts,..., symbol

    full = pd.concat(results, axis=0, ignore_index=True)
    full = full.set_index(["symbol", "ts"]).sort_index()
    return full


# ---------------- Monthly snapshot features ----------------

def compute_features(
    bars: pd.DataFrame,
    *,
    avg_window: int,            # kept for CLI compatibility; not used directly
    trail_return_days: int,     # kept for CLI compatibility; not used directly
    vol_window_days: int,       # kept for CLI compatibility; not used directly
    rebalance_every_days: int = 30,
) -> pd.DataFrame:
    """
    DAILY bars -> DAILY indicators -> SNAPSHOT every `rebalance_every_days`.
    Returns snapshot rows with one entry per (as_of_date, symbol).
    """
    if bars.empty:
        return pd.DataFrame(columns=FEATURE_COLS)

    daily = _compute_daily_indicators(bars)

    # Build snapshot calendar on union of trading days
    all_ts = daily.index.get_level_values(1).unique()
    snaps = _rebalance_calendar(all_ts, every_days=rebalance_every_days)
    if len(snaps) == 0:
        return pd.DataFrame(columns=FEATURE_COLS)

    # For each snapshot, take the indicator row at that exact date (no forward-fill across days)
    frames = []
    for snap in snaps:
        if (daily.index.get_level_values(1) == snap).any():
            sub = daily.xs(snap, level=1, drop_level=False).copy()  # index: (symbol, ts)
            sub = sub.reset_index()  # columns: symbol, ts, ...
            sub.rename(columns={"ts": "as_of_date"}, inplace=True)
            sub["as_of_date"] = pd.to_datetime(snap)
            sub["as_of_utc"] = datetime.now(timezone.utc)
            frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=FEATURE_COLS)

    snap = pd.concat(frames, axis=0, ignore_index=True)

    # Final column selection/rename; name/exchange will be merged in selection step
    out = snap[[
        "as_of_date", "symbol", "close", "adv_20d", "cont_last20",
        "ret_5d", "ret_21d", "ret_63d",
        "macd", "macd_signal", "macd_hist",
        "bb_percent_b", "bb_bandwidth",
        "vol_20d", "vol_63d", "atr_14", "vol_z_20d",
        "as_of_utc",
    ]].rename(columns={"close": "last_close"})

    return out


# ---------------- Gates at snapshot ----------------

def apply_gates(
    snapshot_features: pd.DataFrame,
    *,
    min_price: float,
    min_adv_usd: float,
    continuity_min_traded: int,
) -> pd.DataFrame:
    """Apply price/ADV/continuity gates at each snapshot date."""
    if snapshot_features.empty:
        return snapshot_features.assign(adv_ok=[], continuity_ok=[])
    out = snapshot_features.loc[snapshot_features["last_close"] >= min_price].copy()
    out["adv_ok"] = out["adv_20d"] >= float(min_adv_usd)
    out["continuity_ok"] = out["cont_last20"] >= float(continuity_min_traded)
    return out


# ---------------- Per-snapshot selection with randomized injection ----------------

def select_universe_with_injection(
    assets: pd.DataFrame,
    gated_features: pd.DataFrame,
    *,
    top_n: int,
    inject_frac_bad_min: float,   # e.g., 0.20
    inject_frac_bad_max: float,   # e.g., 0.25
    bad_return_pctile: float,
    bad_vol_pctile: float,
    random_seed: int,
) -> pd.DataFrame:
    """
    Per snapshot date:
      1) Draw a random bad fraction U[min,max] for THIS snapshot.
      2) Among passers (adv_ok & continuity_ok), rank by liquidity (adv_20d) and take core = top_n - inject_n.
      3) Inject 'bad' names (low ret_21d, high vol_20d) up to inject_n.
         If the strict pool is too small, progressively relax and finally force-fill by a composite rank:
            - bad_score = rank(low ret_21d) + rank(high vol_20d)
      4) Top-up by liquidity if still short (should be rare after force-fill).
    """
    if gated_features.empty:
        return pd.DataFrame(columns=FEATURE_COLS)

    rng = np.random.default_rng(seed=random_seed)
    gf = gated_features.merge(assets, how="left", on="symbol")

    out_frames = []
    for as_of_date, grp in gf.groupby("as_of_date", sort=True):
        # 1) Random bad fraction for THIS snapshot
        frac_bad = float(rng.uniform(inject_frac_bad_min, inject_frac_bad_max))
        inject_n = int(np.ceil(frac_bad * top_n))
        inject_n = max(0, min(inject_n, top_n))  # safety

        # 2) Core by liquidity among passers
        core_pool = grp[(grp["adv_ok"]) & (grp["continuity_ok"])].copy()
        core_pool = core_pool.sort_values("adv_20d", ascending=False)

        core_target = max(top_n - inject_n, 0)
        core_sel = core_pool.head(core_target).copy()
        core_sel["selected_bucket"] = "core"

        remainder = grp[~grp["symbol"].isin(core_sel["symbol"])].copy()
        final = core_sel

        # 3) Build bad pool and sample up to inject_n (with robust fallbacks)
        need_bad = inject_n
        injected = pd.DataFrame(columns=grp.columns)

        if need_bad > 0 and not remainder.empty:
            cand = remainder.copy()
            cand = cand[cand["ret_21d"].notna() & cand["vol_20d"].notna()]

            def try_pick(pool: pd.DataFrame, k: int) -> pd.DataFrame:
                if pool.empty or k <= 0:
                    return pd.DataFrame(columns=pool.columns)
                pool = pool.sort_values(["continuity_ok", "adv_20d"], ascending=[False, True])
                k = min(k, len(pool))
                idx = rng.choice(len(pool), size=k, replace=False)
                return pool.iloc[idx].copy()

            picked = pd.DataFrame(columns=cand.columns)

            # Stage A: strict
            if not cand.empty:
                ret_thresh_A = cand["ret_21d"].quantile(bad_return_pctile, interpolation="linear")
                vol_thresh_A = cand["vol_20d"].quantile(bad_vol_pctile, interpolation="linear")
                pool_A = cand[(cand["ret_21d"] <= ret_thresh_A) & (cand["vol_20d"] >= vol_thresh_A)]
                take = min(need_bad - len(picked), len(pool_A))
                if take > 0:
                    picked = pd.concat([picked, try_pick(pool_A, take)], ignore_index=True)

            # Stage B: relaxed
            if len(picked) < need_bad and not cand.empty:
                ret_thresh_B = cand["ret_21d"].quantile(min(0.35, max(0.0, bad_return_pctile + 0.10)), interpolation="linear")
                vol_thresh_B = cand["vol_20d"].quantile(max(0.65, min(1.0, bad_vol_pctile - 0.10)), interpolation="linear")
                pool_B = cand[(cand["ret_21d"] <= ret_thresh_B) & (cand["vol_20d"] >= vol_thresh_B)]
                pool_B = pool_B[~pool_B["symbol"].isin(picked["symbol"])]
                take = min(need_bad - len(picked), len(pool_B))
                if take > 0:
                    picked = pd.concat([picked, try_pick(pool_B, take)], ignore_index=True)

            # Stage C: force-fill by composite badness score
            if len(picked) < need_bad:
                pool_C = remainder[~remainder["symbol"].isin(picked["symbol"])].copy()
                pool_C["rank_lowret"] = pool_C["ret_21d"].rank(method="first", ascending=True)
                pool_C["rank_highvol"] = pool_C["vol_20d"].rank(method="first", ascending=False)
                pool_C["bad_score"] = pool_C["rank_lowret"] + pool_C["rank_highvol"]
                pool_C = pool_C.sort_values(["continuity_ok", "bad_score"], ascending=[False, True])
                take = min(need_bad - len(picked), len(pool_C))
                if take > 0:
                    picked = pd.concat([picked, pool_C.head(take)], ignore_index=True)

            if not picked.empty:
                injected = picked.copy()
                injected["selected_bucket"] = "injected_bad"
                final = pd.concat([final, injected], ignore_index=True)

        # 4) Top-up if still short
        if len(final) < top_n:
            need = top_n - len(final)
            topup_pool = remainder[~remainder["symbol"].isin(final["symbol"])].copy()
            topup = topup_pool.sort_values("adv_20d", ascending=False).head(need).copy()
            if not topup.empty:
                topup["selected_bucket"] = np.where(
                    topup["adv_ok"] & topup["continuity_ok"], "core_topup", "lenient_topup"
                )
                final = pd.concat([final, topup], ignore_index=True)

        # finalize columns
        final = final.copy()
        final["as_of_date"] = pd.to_datetime(as_of_date)
        final["as_of_utc"] = datetime.now(timezone.utc)

        for c in FEATURE_COLS:
            if c not in final.columns:
                final[c] = np.nan

        out_frames.append(final[FEATURE_COLS].head(top_n))

    if not out_frames:
        return pd.DataFrame(columns=FEATURE_COLS)

    return pd.concat(out_frames, ignore_index=True).sort_values(
        ["as_of_date", "selected_bucket", "adv_20d"], ascending=[True, True, False]
    ).reset_index(drop=True)


# ---------------- Labels view ----------------

def labels_view(final_df: pd.DataFrame) -> pd.DataFrame:
    """Per-snapshot labels (good/bad) from selected_bucket."""
    if final_df.empty:
        return pd.DataFrame(columns=LABEL_COLS)
    out = final_df[["as_of_date", "symbol", "name", "exchange", "selected_bucket"]].copy()
    out["label"] = np.where(out["selected_bucket"] == "injected_bad", "bad", "good")
    return out.sort_values(["as_of_date", "label", "symbol"]).reset_index(drop=True)