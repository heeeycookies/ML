# learning.py
# -----------------------------------------------------------------------------
# Purpose
# -------
# Demo of a simple *production-ish* workflow that:
#   1) Downloads real prices (yfinance)
#   2) Rebalances monthly using PyPortfolioOpt (mean-variance, long-only, capped)
#   3) Backtests with costs + T+1 execution
#   4) Emits a QuantStats tear sheet (HTML)
#
# This file is deliberately verbose and commented for learning.
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

# ----- Local import helper ----------------------------------------------------
# Make the project root importable so we can do: from backtest.backtester import ...
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest.backtester import PortfolioBacktester, BacktestConfig  # noqa: E402

# ----- Third-party (install first) -------------------------------------------
# pip install yfinance quantstats PyPortfolioOpt scipy
import yfinance as yf
from pypfopt import EfficientFrontier, expected_returns, risk_models, objective_functions


# =============================================================================
# Configuration knobs (change these freely)
# =============================================================================
TICKERS: List[str] = [
    # Large, liquid names (feel free to change/expand)
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "BRK-B", "JPM",
    "XOM", "UNH", "V", "LLY", "WMT", "MA", "PG", "COST", "HD", "JNJ",
]
BENCHMARK = "SPY"

START_DATE = "2018-01-01"
END_DATE = None  # None = today

# Rebalancing / estimation params
LOOKBACK_DAYS = 252          # use last ~1Y of daily data to estimate mean & cov
REBAL_FREQ = "M"             # "M" = calendar month-end; you can also do business-day based rebal (see note below)
MAX_WEIGHT = 0.10            # per-name cap (10%)
WEIGHT_BOUNDS = (0.0, MAX_WEIGHT)  # long-only; use (-1, 1) to allow shorts

# Optimizer flavor (choose one)
OPTIMIZE_MODE = "max_sharpe"     # "max_sharpe" or "min_vol" or "efficient_risk"
TARGET_VOL = 0.18                # only used if OPTIMIZE_MODE == "efficient_risk"

# Small L2 regularization on weights to reduce concentration
L2_REG = 0.001

# Backtest (costs & QS report)
TX_COST_BPS = 10
SLIP_BPS = 2
EXECUTION_LAG_DAYS = 1
QS_HTML_PATH = "reports/pypfopt_tearsheet.html"
QS_TITLE = "PyPortfolioOpt Monthly (Long-only, 10% cap) vs SPY"


# =============================================================================
# Utilities
# =============================================================================
def load_adj_close(tickers: List[str], start: str, end: str | None = None) -> pd.DataFrame:
    """
    Download adjusted close prices from Yahoo Finance for given tickers.
    Returns a (date x ticker) DataFrame.
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):  # happens if only one ticker
        df = df.to_frame()
    df = df.dropna(how="all").sort_index()
    return df


def pypfopt_weights_from_window(price_window: pd.DataFrame) -> pd.Series:
    """
    Given a *price* window (date x tickers) of length LOOKBACK_DAYS,
    compute long-only mean-variance weights using PyPortfolioOpt.
    - Mean estimator: mean historical return (annualized)
    - Covariance estimator: Ledoit-Wolf shrinkage (annualized)
    - Objective: maximize Sharpe (default) with small L2 regularization
    - Constraints: sum(w) = 1, 0 <= w <= MAX_WEIGHT
    Returns a pd.Series of weights indexed by tickers (sums to ~1).
    """
    # 1) Expected returns & covariance (annualized)
    mu = expected_returns.mean_historical_return(price_window, frequency=252)  # pd.Series
    S = risk_models.CovarianceShrinkage(price_window, frequency=252).ledoit_wolf()

    # 2) Efficient frontier object with weight bounds
    ef = EfficientFrontier(mu, S, weight_bounds=WEIGHT_BOUNDS)

    # Optional: add L2 weight regularization to reduce concentration
    if L2_REG and L2_REG > 0:
        ef.add_objective(objective_functions.L2_reg, gamma=L2_REG)

    # 3) Choose optimization target
    if OPTIMIZE_MODE == "max_sharpe":
        ef.max_sharpe()
    elif OPTIMIZE_MODE == "min_vol":
        ef.min_volatility()
    elif OPTIMIZE_MODE == "efficient_risk":
        ef.efficient_risk(target_volatility=TARGET_VOL)
    else:
        raise ValueError("OPTIMIZE_MODE must be one of: 'max_sharpe', 'min_vol', 'efficient_risk'")

    # 4) Get cleaned weights (drops tiny dust; still sum ~ 1)
    w = ef.clean_weights()  # dict: {ticker: weight}
    w = pd.Series(w, dtype=float)

    # Numerical clean-ups
    w = w.clip(lower=0)  # ensure long-only even after clean_weights rounding
    s = w.sum()
    if s > 0:
        w = w / s

    return w


def build_rebalance_schedule(prices: pd.DataFrame, freq: str = "M") -> pd.DatetimeIndex:
    """
    Build a sequence of rebalance dates.
    - Default is calendar month-end ("M"), snapped automatically by resample().last()
    - If you want *every N business days*, you can do:
        pd.date_range(prices.index[0], prices.index[-1], freq="B") and slice every N
    """
    # Month-end dates present in prices index
    rebal_dates = prices.resample(freq).last().index
    # Keep only dates that are in the actual trading calendar index
    rebal_dates = pd.Index(rebal_dates).intersection(prices.index)
    return rebal_dates


def build_monthly_weights_pypfopt(prices: pd.DataFrame, lookback_days: int = 252) -> pd.DataFrame:
    """
    Construct a (rebalance-date x ticker) weight table using PyPortfolioOpt.
    On each rebalance date, we:
      - take the previous LOOKBACK_DAYS of *prices*,
      - run pypfopt to compute long-only capped weights,
      - store the target weights for that date.
    """
    rebal_dates = build_rebalance_schedule(prices, freq=REBAL_FREQ)
    rows = []
    prev = pd.Series(0.0, index=prices.columns)

    for t in rebal_dates:
        # Rolling window of prices up to the *day before* rebalance date (prevents trivial leakage)
        # If you prefer exact day t window, you can use .loc[:t] — the execution lag in backtester
        # will still enforce T+1 trading.
        win = prices.loc[:t].tail(lookback_days).dropna(how="all", axis=1)
        if len(win) < max(60, lookback_days // 6) or win.shape[1] < 2:
            # Not enough data or too few names → carry previous weights
            rows.append(prev.rename(t))
            continue

        try:
            w = pypfopt_weights_from_window(win)
        except Exception as e:
            # Optimizer can fail occasionally (singular cov, etc.) → carry previous weights
            print(f"[Warn] Optimizer failed at {t.date()}: {e}")
            rows.append(prev.rename(t))
            continue

        # Align to full universe with zeros for names not in window
        full = pd.Series(0.0, index=prices.columns)
        full.loc[w.index] = w
        rows.append(full.rename(t))
        prev = full

    weights = pd.DataFrame(rows).sort_index()
    return weights


# =============================================================================
# Main script
# =============================================================================
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 1) Download data (tickers + benchmark)
    # -------------------------------------------------------------------------
    tickers = list(dict.fromkeys(TICKERS))  # de-dup while preserving order
    all_symbols = tickers + [BENCHMARK]

    print("[Info] Downloading prices from Yahoo Finance…")
    px = load_adj_close(all_symbols, START_DATE, END_DATE)
    px = px.dropna(how="all")

    # Split strategy universe vs benchmark
    prices = px[tickers].dropna(how="all")
    bench_prices = px[BENCHMARK].dropna()

    # -------------------------------------------------------------------------
    # 2) Build monthly PyPortfolioOpt weights
    # -------------------------------------------------------------------------
    print("[Info] Building monthly PyPortfolioOpt weights…")
    weights = build_monthly_weights_pypfopt(prices, lookback_days=LOOKBACK_DAYS)

    # (Optional) Forward-fill weights to daily is not required here—
    # the backtester reindexes & ffills to the trading calendar and applies T+1 execution.

    # -------------------------------------------------------------------------
    # 3) Backtest with costs + T+1 execution + QuantStats tear sheet
    # -------------------------------------------------------------------------
    print("[Info] Running backtest…")
    os.makedirs("reports", exist_ok=True)
    cfg = BacktestConfig(
        tx_cost_bps=TX_COST_BPS,
        slip_bps=SLIP_BPS,
        execution_lag_days=EXECUTION_LAG_DAYS,
        allow_cash=True,
        qs_html_path=QS_HTML_PATH,         # turn ON QuantStats auto report
        qs_title=QS_TITLE,
    )

    bt = PortfolioBacktester(prices=prices, config=cfg)
    result = bt.run(weights=weights, benchmarks={BENCHMARK: bench_prices})

    # Simple console summary
    print("\n=== Strategy Metrics ===")
    for k, v in result["metrics"].items():
        print(f"{k:>18}: {v: .4f}")

    if "benchmarks" in result and BENCHMARK in result["benchmarks"]:
        print("\n=== Benchmark Metrics (SPY) ===")
        for k, v in result["benchmarks"][BENCHMARK].items():
            print(f"{k:>18}: {v: .4f}")

    print(f"\n[Done] Tear sheet saved to: {QS_HTML_PATH}")

    # -------------------------------------------------------------------------
    # Notes
    # -------------------------------------------------------------------------
    # • Change OPTIMIZE_MODE to "min_vol" for minimum-variance portfolios,
    #   or "efficient_risk" to target a specific annualized volatility.
    # • To rebalance every N business days instead of month-end:
    #       cal = prices.index
    #       rebal_dates = cal[::N]
    #   and use that in build_monthly_weights_pypfopt.
    # • You can plug this into your universe (e.g., filter columns to your BlueChip universe).
    # • For sector caps or turnover constraints, PyPortfolioOpt supports custom objectives
    #   and constraints—you can add them in pypfopt_weights_from_window().
