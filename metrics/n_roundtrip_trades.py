from __future__ import annotations
from .base import register_metric

@register_metric(name="Number of Round-Trip Trades", unit="count")
def compute(equity_df, trades_df, params) -> float:
    if trades_df is None:
        return 0.0
    return float(len(trades_df))
