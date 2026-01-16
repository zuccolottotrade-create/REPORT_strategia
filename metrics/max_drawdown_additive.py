# metrics/max_drawdown_additive.py
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .base import register_metric


@register_metric(name="Max Drawdown (Additive)", unit="€")
def compute(equity_df: pd.DataFrame, trades_df: pd.DataFrame, params: Any = None) -> float:
    """
    STRADA A (REALIZED):
    Max Drawdown in € calcolato sulla cumulata dei PnL realizzati dei trade.

    Usa trades_df["pnl_eur"] (come da struttura del tuo trades_df).
    Definizione:
      eq = cumsum(pnl_eur)
      MDD = min_t( eq_t - cummax(eq)_t )   # <= 0
    """
    if trades_df is None or getattr(trades_df, "empty", True):
        return 0.0

    if "pnl_eur" not in trades_df.columns:
        return 0.0

    # Ordina per exit_dt se presente (sequenza trade corretta)
    df = trades_df.copy()
    if "exit_dt" in df.columns:
        df["exit_dt"] = pd.to_datetime(df["exit_dt"], errors="coerce")
        df = df.sort_values("exit_dt", kind="mergesort")

    pnl = pd.to_numeric(df["pnl_eur"], errors="coerce").fillna(0.0)

    eq = pnl.cumsum()
    if len(eq) < 2:
        return 0.0

    dd = eq - eq.cummax()
    mdd = float(dd.min())

    if not math.isfinite(mdd) or mdd > 0:
        return 0.0

    return mdd
