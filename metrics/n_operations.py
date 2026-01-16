from __future__ import annotations
from .base import register_metric

@register_metric(name="Number of Operations (IN+OUT)", unit="count")
def compute(equity_df, trades_df, params) -> float:
    ops = int((equity_df["position"].diff().fillna(equity_df["position"]).abs() > 0).sum())
    return float(ops)
