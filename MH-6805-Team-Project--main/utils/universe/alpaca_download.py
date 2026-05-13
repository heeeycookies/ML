import os
import math
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
INPUT_CSV = "data/raw/sector_20stocks.csv"   # CSV with 'Sector Name' and 'Symbol'

RAW_DIR = "data/raw/ohlcv"
PROC_DIR = "data/processed"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

START_DOWNLOAD_DATE = "2022-06-01"   # warmup start (for SMA_50, etc.)
END_DATE_EXCLUSIVE  = "2025-10-02"   # exclusive
INDICATOR_OUTPUT_START = "2022-10-01"  # indicators output starts here

BATCH_SIZE = 100

RAW_PARQUET = os.path.join(RAW_DIR, "ohlcv_daily.parquet")
RAW_CSV     = os.path.join(RAW_DIR, "ohlcv_daily.csv")
IND_PARQUET = os.path.join(PROC_DIR, "indicators_daily.parquet")
IND_CSV     = os.path.join(PROC_DIR, "indicators_daily.csv")

# ----------------------------------------------------------------------
# INIT ALPACA CLIENT
# ----------------------------------------------------------------------
load_dotenv()
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
if not ALPACA_KEY or not ALPACA_SECRET:
    raise RuntimeError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY")
client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def load_ticker_sector_mapping(path: str) -> pd.DataFrame:
    """
    Loads mapping from your CSV:
      - Sector column: 'Sector Name'
      - Ticker column: 'Symbol'
    Returns DataFrame with columns ['Ticker', 'Sector'].
    """
    df = pd.read_csv(path)
    for col in ("Sector Name", "Symbol"):
        if col not in df.columns:
            raise RuntimeError(f"Required column '{col}' not found in {path}")
    mapping = df[["Symbol", "Sector Name"]].copy()
    mapping.columns = ["Ticker", "Sector"]
    mapping["Ticker"] = (
        mapping["Ticker"]
        .astype(str).str.strip().str.upper()
        .replace({"", "NA", "NAN", "NONE"}, pd.NA)
    )
    mapping = mapping.dropna().drop_duplicates()
    return mapping

def download_ohlcv_alpaca(tickers: list[str]) -> pd.DataFrame:
    """
    Download daily OHLCV bars for tickers from Alpaca in batches.
    Returns tidy DataFrame: Date, Ticker, Open, High, Low, Close, Volume
    """
    all_batches = []
    num_batches = math.ceil(len(tickers) / BATCH_SIZE)

    for i in tqdm(range(num_batches), desc="Alpaca bars"):
        batch = tickers[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        try:
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=START_DOWNLOAD_DATE,
                end=END_DATE_EXCLUSIVE,
            )
            bars = client.get_stock_bars(req)
            df_batch = bars.df
            if df_batch.empty:
                continue

            df_batch = df_batch.reset_index().rename(
                columns={
                    "symbol": "Ticker",
                    "timestamp": "Date",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            )

            keep_cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
            df_batch = df_batch[keep_cols]
            df_batch["Date"] = pd.to_datetime(df_batch["Date"], utc=True).dt.tz_convert(None)
            df_batch["Ticker"] = df_batch["Ticker"].astype(str).str.upper()
            # enforce float64 to avoid tiny dtype diffs in TA calcs
            for c in ["Open","High","Low","Close","Volume"]:
                df_batch[c] = pd.to_numeric(df_batch[c], errors="coerce")
            df_batch = df_batch.sort_values(["Ticker", "Date"])
            all_batches.append(df_batch)
        except Exception as e:
            tqdm.write(f"⚠️  Batch {i} failed: {e}")

    if not all_batches:
        return pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"])

    return pd.concat(all_batches, ignore_index=True)

# ----------------------------------------------------------------------
# TA-compatible indicator implementations
#   * EMAs seeded with SMA(span)
#   * RSI/ATR using Wilder's RMA and seeding by SMA of initial window
#   * MACD derived from TA-style EMAs
# ----------------------------------------------------------------------
def _ema_ta_seed(close: pd.Series, span: int) -> pd.Series:
    """EMA seeded by SMA(span) at first full window; recursive thereafter."""
    close = close.astype(float)
    ema = pd.Series(np.nan, index=close.index, dtype=float)
    sma = close.rolling(span, min_periods=span).mean()
    i0 = sma.first_valid_index()
    if i0 is None:
        return ema
    alpha = 2.0 / (span + 1.0)
    ema.loc[i0] = sma.loc[i0]
    pos = close.index.get_loc(i0)
    for i in range(pos + 1, len(close)):
        ema.iloc[i] = alpha * close.iloc[i] + (1 - alpha) * ema.iloc[i - 1]
    return ema

def _rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI using Wilder's RMA for avg gain/loss; seeded by SMA at first window."""
    close = close.astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = pd.Series(np.nan, index=close.index, dtype=float)
    avg_loss = pd.Series(np.nan, index=close.index, dtype=float)

    g_sma = gain.rolling(period, min_periods=period).mean()
    l_sma = loss.rolling(period, min_periods=period).mean()
    i0 = g_sma.first_valid_index()
    if i0 is None:
        return pd.Series(np.nan, index=close.index)

    avg_gain.loc[i0] = g_sma.loc[i0]
    avg_loss.loc[i0] = l_sma.loc[i0]

    pos = close.index.get_loc(i0)
    for i in range(pos + 1, len(close)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def _atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR using Wilder's TR + RMA; seed with mean(TR[1..period])."""
    high = high.astype(float); low = low.astype(float); close = close.astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = pd.Series(np.nan, index=close.index, dtype=float)
    seed = tr.rolling(period, min_periods=period).mean()
    i0 = seed.first_valid_index()
    if i0 is None:
        return atr

    atr.loc[i0] = seed.loc[i0]
    pos = close.index.get_loc(i0)
    for i in range(pos + 1, len(tr)):
        atr.iloc[i] = (atr.iloc[i - 1] * (period - 1) + tr.iloc[i]) / period
    return atr

def _macd_ta(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """MACD built from TA-style EMAs above."""
    ema_fast = _ema_ta_seed(close, fast)
    ema_slow = _ema_ta_seed(close, slow)
    macd = ema_fast - ema_slow
    # signal EMA should also be SMA-seeded; compute on contiguous macd segment
    sig = _ema_ta_seed(macd.dropna(), signal).reindex(macd.index)
    hist = macd - sig
    return pd.DataFrame({"MACD": macd, "MACD_Signal": sig, "MACD_Hist": hist})

def _bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    ma = close.rolling(window=period, min_periods=period).mean()
    sd = close.rolling(window=period, min_periods=period).std()
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    return pd.DataFrame({"BB_Mid": ma, "BB_Upper": upper, "BB_Lower": lower})

# ----------------------------------------------------------------------
# INDICATOR PIPE
# ----------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: Date, Ticker, Sector, Open, High, Low, Close, Volume
    Returns df with indicators by Ticker (Sector column is preserved).
    """
    def _per_ticker(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()

        # Ensure correct dtype
        for c in ["Open","High","Low","Close","Volume"]:
            g[c] = pd.to_numeric(g[c], errors="coerce")

        # Returns
        g["Return"] = g["Close"].pct_change()
        g["LogReturn"] = np.log(g["Close"]).diff()

        # Trend (SMAs exact already)
        g["SMA_20"] = g["Close"].rolling(20, min_periods=20).mean()
        g["SMA_50"] = g["Close"].rolling(50, min_periods=50).mean()

        # TA-accurate EMAs
        g["EMA_12"] = _ema_ta_seed(g["Close"], 12)
        g["EMA_26"] = _ema_ta_seed(g["Close"], 26)

        # RSI (Wilder)
        g["RSI_14"] = _rsi_wilder(g["Close"], 14)

        # MACD from TA-EMAs
        macd_df = _macd_ta(g["Close"], 12, 26, 9)
        g = pd.concat([g, macd_df], axis=1)

        # Bollinger (20, 2σ)
        bb_df = _bollinger(g["Close"], 20, 2.0)
        g = pd.concat([g, bb_df], axis=1)

        # ATR (Wilder)
        g["ATR_14"] = _atr_wilder(g["High"], g["Low"], g["Close"], 14)

        return g

    return (
        df.sort_values(["Ticker", "Date"])
          .groupby("Ticker", group_keys=False)
          .apply(_per_ticker)
    )

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    print("=" * 80)
    print("Alpaca Daily OHLCV Downloader + Indicators (+Sector, TA-accurate)")
    print("=" * 80)
    print(f"Download window: {START_DOWNLOAD_DATE} → {END_DATE_EXCLUSIVE} (exclusive)")
    print(f"Indicator outputs trimmed from: {INDICATOR_OUTPUT_START}\n")

    # Load ticker-sector mapping from your CSV
    ticker_sector_df = load_ticker_sector_mapping(INPUT_CSV)
    tickers = ticker_sector_df["Ticker"].tolist()
    print(f"Loaded {len(tickers)} tickers with sectors from CSV")

    # Download OHLCV
    ohlcv_df = download_ohlcv_alpaca(tickers)
    print(f"\nOHLCV rows: {len(ohlcv_df):,}   tickers: {ohlcv_df['Ticker'].nunique():,}")

    # Merge Sector into OHLCV rows
    ohlcv_df = ohlcv_df.merge(ticker_sector_df, on="Ticker", how="left")

    # Save raw OHLCV
    ohlcv_df.to_parquet(RAW_PARQUET, index=False)
    ohlcv_df.to_csv(RAW_CSV, index=False)
    print(f"Saved raw OHLCV →\n  {RAW_PARQUET}\n  {RAW_CSV}")

    # Compute indicators (TA-compatible)
    ind_df_full = add_indicators(ohlcv_df)

    # Trim to output start date (after warmup)
    cutoff = pd.to_datetime(INDICATOR_OUTPUT_START)
    ind_df = ind_df_full[ind_df_full["Date"] >= cutoff].copy()

    # Save indicators (includes Sector)
    ind_df.to_parquet(IND_PARQUET, index=False)
    ind_df.to_csv(IND_CSV, index=False)
    print(f"Saved indicators →\n  {IND_PARQUET}\n  {IND_CSV}")

    print("\n✅ Done.")

if __name__ == "__main__":
    main()