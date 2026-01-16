# metrics/volatility_per_trade.py
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .base import register_metric

print("[LOADED] volatility_per_trade imported")

@register_metric(name="Volatility per Trade", unit="%")
def compute(equity_df: pd.DataFrame, trades_df: pd.DataFrame, params: Any = None) -> float:
    """
    Volatility per Trade (coerente Additive/Realized):

      r_k = pnl_eur(k) / EquityStart
      Volatility per Trade = std(r_k) * 100

    Nota:
    - ddof=0 (come le altre metriche)
    - NON annualizzata (perché è "per trade", non per periodo fisso)
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

    pnl = pd.to_numeric(trades_df["pnl_eur"], errors="coerce").dropna()
    if len(pnl) < 2:
        return 0.0

    r_trade = pnl / e0
    vol = float(r_trade.std(ddof=0))
    if not math.isfinite(vol) or vol <= 0.0:
        return 0.0

    return float(vol * 100.0)
