# metrics/gross_pnl_trades.py
from __future__ import annotations

import pandas as pd

from .base import register_metric


def _get_notional(params: dict | None, equity_df: pd.DataFrame | None) -> float:
    """
    Notional fisso per trade. Priorità:
      1) params['notional_per_trade']
      2) params['initial_capital']
      3) equity_df['initial_capital'].iloc[0]
      4) 0.0 (fallback)
    """
    notional = 0.0
    if isinstance(params, dict):
        notional = float(params.get("notional_per_trade") or params.get("initial_capital") or 0.0)

    if notional == 0.0 and equity_df is not None and "initial_capital" in equity_df.columns:
        try:
            notional = float(pd.to_numeric(equity_df["initial_capital"], errors="coerce").dropna().iloc[0])
        except Exception:
            notional = 0.0

    return notional


def _pnl_eur_series(equity_df: pd.DataFrame, trades_df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Serie PnL per trade in € (SENZA costi).
    Source of truth:
      1) trades_df['pnl_eur'] se presente
      2) fallback: notional * pnl_pct  (pnl_pct è DECIMALE: 0.03=+3%)
    """
    if "pnl_eur" in trades_df.columns:
        return pd.to_numeric(trades_df["pnl_eur"], errors="coerce").fillna(0.0)

    if "pnl_pct" not in trades_df.columns:
        # nessuna base coerente -> non possiamo calcolare in €
        return pd.Series([0.0] * len(trades_df), index=trades_df.index, dtype=float)

    notional = _get_notional(params, equity_df)
    pnl_pct = pd.to_numeric(trades_df["pnl_pct"], errors="coerce").fillna(0.0)

    # pnl_pct è DECIMALE -> non dividere per 100
    if notional != 0.0:
        return notional * pnl_pct

    return pd.Series([0.0] * len(trades_df), index=trades_df.index, dtype=float)


@register_metric("Gross Profit (Winning Trades Only)", "€")
def gross_profit_winning_trades(equity_df: pd.DataFrame, trades_df: pd.DataFrame, params: dict) -> float:
    """
    Somma dei soli trade profittevoli (> 0) in €.
    """
    if trades_df is None or trades_df.empty:
        return 0.0

    pnl = _pnl_eur_series(equity_df, trades_df, params)
    return float(pnl[pnl > 0].sum())


@register_metric("Gross Loss (Losing Trades Only)", "€")
def gross_loss_losing_trades(equity_df: pd.DataFrame, trades_df: pd.DataFrame, params: dict) -> float:
    """
    Somma dei soli trade in perdita (< 0) in €.
    Ritorna un numero NEGATIVO.
    """
    if trades_df is None or trades_df.empty:
        return 0.0

    pnl = _pnl_eur_series(equity_df, trades_df, params)
    return float(pnl[pnl < 0].sum())
