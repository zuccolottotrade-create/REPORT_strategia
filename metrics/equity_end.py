# metrics/equity_end.py
from __future__ import annotations

import pandas as pd

from .base import register_metric


# PnL columns (prefer the new single source of truth)
_PNL_COL_CANDIDATES = [
    "pnl_eur",  # ✅ preferred
    "pnl",
    "PnL",
    "P&L",
    "pl",
    "profit",
    "net_pnl",
    "pnl_abs",
]


def _get_equity_start(equity_df: pd.DataFrame, params) -> float:
    """
    Equity Start source order:
      1) equity_df['initial_capital'] (if injected upstream)
      2) params['equity_start'] (if provided)
      3) fallback: 0.0
    """
    if isinstance(equity_df, pd.DataFrame) and not equity_df.empty and "initial_capital" in equity_df.columns:
        s = pd.to_numeric(equity_df["initial_capital"], errors="coerce").dropna()
        if not s.empty:
            return float(s.iloc[0])

    if isinstance(params, dict):
        v = params.get("equity_start")
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass

    return 0.0


def _sum_pnl(df: pd.DataFrame) -> float | None:
    """
    Returns sum of PnL from the first matching column, or None if not available.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None

    for c in _PNL_COL_CANDIDATES:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            return float(s.sum())

    return None


@register_metric(name="Equity End", unit="€")
def compute(equity_df: pd.DataFrame, trades_df: pd.DataFrame, params) -> float:
    """
    Equity End (ABSOLUTE) = Equity Start + Net Profit

    IMPORTANT:
    - Non usa colonne tipo 'equity' / 'equity_norm' / fattori 1.xx (che sono normalizzazioni)
    - Usa esclusivamente:
        * Equity Start (initial_capital o params)
        * somma PnL (preferendo trades_df['pnl_eur'])
    """
    equity_start = _get_equity_start(equity_df, params)

    pnl_sum = _sum_pnl(trades_df)
    if pnl_sum is None:
        pnl_sum = _sum_pnl(equity_df)

    if pnl_sum is None:
        pnl_sum = 0.0

    return float(equity_start + pnl_sum)

