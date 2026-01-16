from __future__ import annotations

from .base import register_metric


_PNL_COL_CANDIDATES = ["pnl", "PnL", "P&L", "pl", "profit", "net_pnl", "pnl_abs"]


def _pick_pnl_col(trades_df):
    for c in _PNL_COL_CANDIDATES:
        if c in trades_df.columns:
            return c
    return None


@register_metric(name="Profit Factor", unit="ratio")
def compute(equity_df, trades_df, params) -> float:
    if trades_df is None or trades_df.empty:
        return 0.0

    pnl_col = _pick_pnl_col(trades_df)

    # Se non ho pnl assoluto, come fallback uso pnl_pct (ma NON sarà coerente con Gross Profit/Loss assoluti)
    if pnl_col is None:
        if "pnl_pct" not in trades_df.columns:
            return
