from __future__ import annotations
from .base import register_metric

@register_metric(name="Exits (IN->OUT)", unit="count")
def compute(equity_df, trades_df, params) -> float:
    diff = equity_df["position"].diff().fillna(0)
    return float((diff == -1).sum())
