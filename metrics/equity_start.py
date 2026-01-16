from __future__ import annotations
from .base import register_metric

@register_metric(name="Equity Start", unit="EUR")
def compute(equity_df, trades_df, params) -> float:
    return float(equity_df["equity"].iloc[0])
