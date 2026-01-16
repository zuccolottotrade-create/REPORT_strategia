from __future__ import annotations
from .base import register_metric

@register_metric(name="Time IN Position", unit="%")
def compute(equity_df, trades_df, params) -> float:
    return float(equity_df["position"].mean() * 100.0)
