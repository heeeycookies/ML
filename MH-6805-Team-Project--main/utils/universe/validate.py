# validate_cross_lib.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional deps
try:
    import pandas_ta as ta
except Exception as e:
    raise RuntimeError("pandas_ta is required: pip install pandas_ta") from e

try:
    import talib
    TALIB_AVAILABLE = True
except Exception:
    TALIB_AVAILABLE = False

RAW_PARQ = "data/raw/ohlcv/ohlcv_daily.parquet"
IND_PARQ = "data/processed/indicators_daily.parquet"
OUT_CSV  = "data/processed/crosslib_validation_summary.csv"

# -------------------------- helpers --------------------------
def compare_series_idx(name, a_vals, a_idx, b_vals, b_idx, tol=1e-6):
    """
    Align two time series on Date index and compute metrics.
    Returns (name, MAD, within_tol, N) or None if no overlap.
    """
    A = pd.Series(a_vals, index=pd.to_datetime(a_idx)).astype(float)
    B = pd.Series(b_vals, index=pd.to_datetime(b_idx)).astype(float)
    both = pd.concat([A.rename("A"), B.rename("B")], axis=1, join="inner").dropna()
    if both.empty:
        return (name, np.nan, np.nan, 0)
    diff = (both["A"] - both["B"]).abs()
    mad = float(diff.mean())
    within = float((diff < tol).mean())
    return (name, mad, within, int(len(both)))

def compute_pandas_ta(ohlcv_df):
    """Compute key indicators with pandas_ta on OHLCV for a single ticker (sorted by Date)."""
    close = pd.Series(ohlcv_df["Close"].astype(float).values, index=ohlcv_df["Date"])
    high  = pd.Series(ohlcv_df["High"].astype(float).values,  index=ohlcv_df["Date"])
    low   = pd.Series(ohlcv_df["Low"].astype(float).values,   index=ohlcv_df["Date"])

    out = {}
    # RSI(14)
    out["RSI_14"] = ta.rsi(close, length=14)
    # EMA
    out["EMA_12"] = ta.ema(close, length=12)
    out["EMA_26"] = ta.ema(close, length=26)
    # MACD (12,26,9)
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        out["MACD"]        = macd[macd.columns[0]]
        out["MACD_Signal"] = macd[macd.columns[1]]
        out["MACD_Hist"]   = macd[macd.columns[2]]
    # ATR(14)
    out["ATR_14"] = ta.atr(high, low, close, length=14)

    return {k: v.to_numpy() for k, v in out.items()}

def compute_talib(ohlcv_df):
    """Compute indicators with TA-Lib (if available)."""
    close = ohlcv_df["Close"].astype(float).values
    high  = ohlcv_df["High"].astype(float).values
    low   = ohlcv_df["Low"].astype(float).values

    out = {}
    out["RSI_14"] = talib.RSI(close, timeperiod=14)
    out["EMA_12"] = talib.EMA(close, timeperiod=12)
    out["EMA_26"] = talib.EMA(close, timeperiod=26)
    macd, macds, macdh = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    out["MACD"] = macd
    out["MACD_Signal"] = macds
    out["MACD_Hist"] = macdh
    out["ATR_14"] = talib.ATR(high, low, close, timeperiod=14)
    return out

# -------------------------- main --------------------------
def main():
    # Load once
    raw = pd.read_parquet(RAW_PARQ)
    ind = pd.read_parquet(IND_PARQ)

    raw["Date"] = pd.to_datetime(raw["Date"])
    ind["Date"] = pd.to_datetime(ind["Date"])

    tickers = sorted(set(raw["Ticker"]).intersection(set(ind["Ticker"])))
    print(f"Found {len(tickers)} tickers present in both OHLCV and indicators.")

    rows = []
    metrics = ["RSI_14","EMA_12","EMA_26","MACD","MACD_Signal","MACD_Hist","ATR_14"]

    for ticker in tqdm(tickers, desc="Cross-lib validating"):
        o = raw[raw["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        y = ind[ind["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        if o.empty or y.empty:
            continue

        # pandas_ta reference
        pta = compute_pandas_ta(o)

        # (1) Your vs pandas_ta
        for m in metrics:
            if m in y.columns and m in pta:
                name = f"{ticker} | YOUR vs pTA | {m}"
                nm, mad, within, N = compare_series_idx(
                    name,
                    y[m].values, y["Date"],
                    pta[m],      o["Date"],
                    tol=1e-6
                )
                rows.append({"Ticker": ticker, "Pair": "YOUR_vs_pTA", "Metric": m,
                             "MAD": mad, "WithinTol": within, "N": N})

        # (2) pandas_ta vs TA-Lib (if available)
        if TALIB_AVAILABLE:
            tla = compute_talib(o)
            for m in metrics:
                if (m in pta) and (m in tla):
                    name = f"{ticker} | pTA vs TA-Lib | {m}"
                    nm, mad, within, N = compare_series_idx(
                        name,
                        pta[m], o["Date"],
                        tla[m], o["Date"],
                        tol=1e-6
                    )
                    rows.append({"Ticker": ticker, "Pair": "pTA_vs_TALib", "Metric": m,
                                 "MAD": mad, "WithinTol": within, "N": N})

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    # Print concise summaries
    def summarize(pair):
        sub = out[out["Pair"] == pair]
        if sub.empty:
            return
        print(f"\n=== Summary: {pair} ===")
        print(
            sub.groupby("Metric")
               .agg(MAD_mean=("MAD","mean"),
                    WithinTol_mean=("WithinTol","mean"),
                    N_total=("N","sum"))
               .reset_index()
               .sort_values("Metric")
               .to_string(index=False)
        )

    summarize("YOUR_vs_pTA")
    if TALIB_AVAILABLE:
        summarize("pTA_vs_TALib")

    print(f"\nSaved detailed results → {OUT_CSV}")

if __name__ == "__main__":
    main()