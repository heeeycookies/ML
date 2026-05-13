# backtester.py
from __future__ import annotations
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

# QuantStats (optional HTML report)
import quantstats as qs


@dataclass
class BacktestConfig:
    tx_cost_bps: float = 10.0
    slip_bps: float = 0.0
    execution_lag_days: int = 1
    allow_cash: bool = True
    # QuantStats options
    qs_html_path: Optional[str] = None   # e.g., "reports/tearsheet.html"
    qs_title: str = "Strategy Tear Sheet"


class PortfolioBacktester:
    """
    Portfolio backtester with:
      - daily execution of target weights (with T+1 lag, costs)
      - QuantStats tear sheet (HTML) emitted from run() if qs_html_path is set
    """

    def __init__(self, prices: pd.DataFrame, config: Optional[BacktestConfig] = None):
        self.prices = prices.sort_index()
        self.returns = (
            self.prices.pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        self.cfg = config or BacktestConfig()

    # ------------------ Core Backtest ------------------

    def run(
        self,
        weights: pd.DataFrame,                              # date x symbol; rebalance dates, will be ffilled
        benchmarks: Optional[Dict[str, pd.Series]] = None,  # dict of benchmark *price* series
    ) -> Dict[str, any]:
        # 1) Align & apply execution lag
        weights = weights.reindex(self.prices.index).ffill().fillna(0.0)
        if self.cfg.execution_lag_days > 0:
            weights = weights.shift(self.cfg.execution_lag_days).fillna(0.0)

        # 2) Portfolio daily returns (pre-cost)
        port_rets = (weights * self.returns).sum(axis=1)

        # 3) Costs via turnover
        w_prev = weights.shift(1).fillna(0.0)
        turnover = (weights - w_prev).abs().sum(axis=1)
        cost_perc = (self.cfg.tx_cost_bps + self.cfg.slip_bps) / 10_000.0
        port_rets = port_rets - turnover * cost_perc

        # 4) NAV & metrics
        cum_nav = (1.0 + port_rets).cumprod()
        metrics = self._compute_metrics(port_rets)

        # 5) Benchmarks (prices → returns)
        bench_perf: Dict[str, Dict[str, float]] = {}
        bench_nav: Dict[str, pd.Series] = {}
        if benchmarks:
            for name, ser in benchmarks.items():
                aligned_px = ser.reindex(self.prices.index).ffill()
                b_rets = (
                    aligned_px.pct_change()
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )
                bench_perf[name] = self._compute_metrics(b_rets)
                bench_nav[name] = (1.0 + b_rets).cumprod()

        # 6) QuantStats tear sheet (if path set)
        if self.cfg.qs_html_path:
            self._write_tearsheet(
                strat_rets=port_rets,
                benchmarks=benchmarks,
                html_path=self.cfg.qs_html_path,
                title=self.cfg.qs_title,
            )

        return {
            "daily_returns": port_rets,
            "cum_nav": cum_nav,
            "metrics": metrics,
            "benchmarks": bench_perf,
            "benchmarks_nav": bench_nav,
            "turnover": turnover,
            "weights": weights,
        }

    # ------------------ QuantStats integration ------------------

    def _write_tearsheet(
        self,
        strat_rets: pd.Series,
        benchmarks: Optional[Dict[str, pd.Series]],
        html_path: str,
        title: str,
    ) -> None:
        # Ensure output dir exists
        out_dir = os.path.dirname(html_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Strategy returns (QS expects a datetime-indexed Series of returns)
        rets = strat_rets.dropna().copy()
        rets.index = pd.to_datetime(rets.index)

        # Use the first benchmark if provided (QS takes *returns* as benchmark)
        bench_rets = None
        if benchmarks:
            first_name = next(iter(benchmarks))
            bp = benchmarks[first_name].reindex(rets.index).ffill()
            bench_rets = bp.pct_change().dropna()
            bench_rets = bench_rets.reindex(rets.index).fillna(0.0)

        # Generate the HTML report
        qs.reports.html(
            rets,
            benchmark=bench_rets,
            output=html_path,
            title=title,
        )

    # ------------------ Metrics ------------------

    @staticmethod
    def _compute_metrics(rets: pd.Series) -> Dict[str, float]:
        ann = 252
        r = rets.dropna()
        mu = r.mean() * ann
        vol = r.std(ddof=0) * np.sqrt(ann)
        sharpe = mu / vol if vol > 0 else np.nan
        nav = (1.0 + r).cumprod()
        cumret = nav.iloc[-1] - 1.0 if len(nav) else 0.0
        peak = nav.cummax()
        maxdd = ((nav / peak) - 1.0).min() if len(nav) else 0.0
        return {
            "AnnReturn": float(mu),
            "AnnVol": float(vol),
            "Sharpe": float(sharpe),
            "CumulativeReturn": float(cumret),
            "MaxDrawdown": float(maxdd),
        }