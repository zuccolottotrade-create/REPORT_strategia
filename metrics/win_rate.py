from __future__ import annotations
from .base import register_metric

@register_metric(name="Win Rate (Round-Trip)", unit="%")
def compute(equity_df, trades_df, params) -> float:
    if trades_df is None or trades_df.empty:
        return 0.0
    return float(trades_df["win"].mean() * 100.0)
