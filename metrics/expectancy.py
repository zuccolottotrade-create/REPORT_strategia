# metrics/expectancy.py
from __future__ import annotations

import pandas as pd
from .base import register_metric


@register_metric(name="Expectancy (per Trade)", unit="€")
def compute(equity_df, trades_df: pd.DataFrame, params) -> float:
    """
    Expectancy (per Trade) = Net Profit / Number of Round-Trip Trades

    Riusa ESCLUSIVAMENTE metriche già validate nel report:
    - Net Profit
    - Number of Round-Trip Trades
    """

    report_df: pd.DataFrame | None = params.get("report_df") if params else None
    if report_df is None or report_df.empty:
        return 0.0

    def _get_metric_value(name: str) -> float | None:
        row = report_df.loc[report_df["Indicatore"] == name, "Valore"]
        if row.empty:
            return None
        try:
            return float(row.iloc[0])
        except Exception:
            return None

    net_profit = _get_metric_value("Net Profit")
    n_trades = _get_metric_value("Number of Round-Trip Trades")

    if net_profit is None or n_trades in (None, 0):
        return 0.0

    return round(net_profit / n_trades, 6)
