# metrics/equity_end_additive.py

from __future__ import annotations

import pandas as pd

from .base import register_metric


_PNL_COL_CANDIDATES = [
    "pnl_eur",      # ✅ nuova source of truth (qty=1)
    "pnl", "PnL", "P&L", "pl", "profit", "net_pnl", "pnl_abs"
]


def _sum_pnl_from_df(df: pd.DataFrame) -> float | None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for c in _PNL_COL_CANDIDATES:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            return float(s.sum())
    return None


@register_metric(name="Equity End (Additive)", unit="€")
def compute(equity_df: pd.DataFrame, trades_df: pd.DataFrame, params) -> float:
    """
    Equity End (Additive) = Net Profit

    - preferisce la somma PnL dai trade (trades_df), privilegiando 'pnl_eur'
    - fallback: somma PnL da equity_df (se presente)
    - NON somma mai Equity Start (quello renderebbe il valore "assoluto", non "additive")
    """
    pnl_sum = _sum_pnl_from_df(trades_df)
    if pnl_sum is not None:
        return pnl_sum

    pnl_sum = _sum_pnl_from_df(equity_df)
    if pnl_sum is not None:
        return pnl_sum

    return 0.0
