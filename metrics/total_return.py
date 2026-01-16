# metrics/total_return.py
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .base import register_metric


@register_metric(name="Total Return", unit="%")
def compute(equity_df: pd.DataFrame, trades_df: pd.DataFrame, params: Any = None) -> float:
    """
    Versione A (coerente con Additive / Realized):
    Total Return % = (Net Profit / Equity Start) * 100

    Dove:
      - Equity Start = equity_df["equity"].iloc[0]
      - Net Profit = somma dei PnL realizzati in € (trades_df["pnl_eur"])
    """
    if equity_df is None or getattr(equity_df, "empty", True):
        return 0.0
    if "equity" not in equity_df.columns:
        return 0.0

    e0 = float(pd.to_numeric(equity_df["equity"].iloc[0], errors="coerce"))
    if not math.isfinite(e0) or e0 == 0.0:
        return 0.0

    if trades_df is None or getattr(trades_df, "empty", True):
        return 0.0
    if "pnl_eur" not in trades_df.columns:
        return 0.0

    net_profit = float(pd.to_numeric(trades_df["pnl_eur"], errors="coerce").fillna(0.0).sum())

    tr_pct = (net_profit / e0) * 100.0
    if not math.isfinite(tr_pct):
        return 0.0

    return float(tr_pct)
