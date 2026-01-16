# metrics/net_profit.py
from __future__ import annotations

import pandas as pd
from .base import register_metric


def _side_sign(side_series: pd.Series | None) -> pd.Series:
    """
    LONG/BUY -> +1
    SHORT/SELL -> -1
    default -> +1
    """
    if side_series is None:
        return pd.Series(1.0)

    s = side_series.astype(str).str.strip().str.upper()
    sign = pd.Series(1.0, index=s.index)
    sign[s.isin({"SHORT", "SELL", "-1", "S"})] = -1.0
    sign[s.isin({"LONG", "BUY", "1", "+1", "L"})] = 1.0
    return sign


@register_metric(name="Net Profit", unit="€")
def compute(equity_df: pd.DataFrame, trades_df: pd.DataFrame, params) -> float:
    """
    Net Profit (qty=1) = somma (exit_price - entry_price) * sign
    Se esiste già trades_df['pnl_eur'], usa quello (source of truth).
    """
    if trades_df is None or trades_df.empty:
        return 0.0

    # 1) se pnl_eur è già calcolato a monte (report_strategia.py), usalo
    if "pnl_eur" in trades_df.columns:
        s = pd.to_numeric(trades_df["pnl_eur"], errors="coerce").fillna(0.0)
        return float(s.sum())

    # 2) altrimenti calcolalo da entry/exit (qty=1)
    if "entry_price" not in trades_df.columns or "exit_price" not in trades_df.columns:
        return 0.0

    entry = pd.to_numeric(trades_df["entry_price"], errors="coerce")
    exit_ = pd.to_numeric(trades_df["exit_price"], errors="coerce")
    sign = _side_sign(trades_df["side"] if "side" in trades_df.columns else None)

    pnl = (exit_ - entry) * sign
    pnl = pd.to_numeric(pnl, errors="coerce").fillna(0.0)
    return float(pnl.sum())


