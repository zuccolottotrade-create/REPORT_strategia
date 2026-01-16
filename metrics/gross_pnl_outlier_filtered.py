# metrics/gross_pnl_outlier_filtered.py
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import pandas as pd

from .base import register_metric


# Nel tuo report_strategia.py la colonna canonica e' pnl_eur
_PNL_COL_CANDIDATES: Sequence[str] = (
    "pnl_eur",        # PRIMARY (canonico)
    "pnl_trade_eur",  # fallback se un giorno la usi
    "pnl", "PnL", "P&L", "pl", "profit", "net_pnl",
)


def _pick_pnl_col(trades_df: pd.DataFrame) -> Optional[str]:
    for c in _PNL_COL_CANDIDATES:
        if c in trades_df.columns:
            return c
    return None


def _pnl_all_and_inliers(trades_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Ritorna (pnl_all, pnl_inliers) dove:
      - pnl_all    = pnl numerico (NaN rimossi)
      - pnl_inliers= pnl filtrato: scarta outlier con |pnl - mean| > 3*std

    std come popolazione (ddof=0) => coerente con Excel STDEV.P
    Se std <= 0 o non finita => non filtra nulla (inliers = all).
    """
    col = _pick_pnl_col(trades_df)
    if col is None:
        empty = pd.Series(dtype="float64")
        return empty, empty

    pnl_all = pd.to_numeric(trades_df[col], errors="coerce").dropna().astype(float)
    if pnl_all.empty:
        return pnl_all, pnl_all

    mu = float(pnl_all.mean())
    sigma = float(pnl_all.std(ddof=0))  # STDEV.P

    if (not math.isfinite(mu)) or (not math.isfinite(sigma)) or sigma <= 0.0:
        return pnl_all, pnl_all

    thr = 3.0 * sigma
    mask_inlier = (pnl_all - mu).abs() <= thr
    pnl_inliers = pnl_all.loc[mask_inlier]

    return pnl_all, pnl_inliers


# =========================
# KPI (-Outlier)
# =========================

@register_metric(name="Gross Profit (-Outlier)", unit="€")
def gross_profit_outlier_filtered(equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> float:
    _, pnl = _pnl_all_and_inliers(trades_df)
    if pnl.empty:
        return 0.0
    return float(pnl.loc[pnl > 0].sum())


@register_metric(name="Gross Loss (-Outlier)", unit="€")
def gross_loss_outlier_filtered(equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> float:
    _, pnl = _pnl_all_and_inliers(trades_df)
    if pnl.empty:
        return 0.0
    # Loss deve restare NEGATIVO
    return float(pnl.loc[pnl < 0].sum())


@register_metric(name="Net Profit (-Outlier)", unit="€")
def net_profit_outlier_filtered(equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> float:
    _, pnl = _pnl_all_and_inliers(trades_df)
    if pnl.empty:
        return 0.0
    # = GP + GL (GL negativo)
    return float(pnl.sum())


# =========================
# AUDIT outlier
# =========================

@register_metric(name="Outliers Removed (count)", unit="count")
def outliers_removed_count(equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> float:
    pnl_all, pnl_inliers = _pnl_all_and_inliers(trades_df)
    if pnl_all.empty:
        return 0.0
    removed = int(len(pnl_all) - len(pnl_inliers))
    return float(removed)


@register_metric(name="Outliers Removed (%)", unit="%")
def outliers_removed_pct(equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> float:
    pnl_all, pnl_inliers = _pnl_all_and_inliers(trades_df)
    n_all = int(len(pnl_all))
    if n_all == 0:
        return 0.0
    removed = float(n_all - len(pnl_inliers))
    return float(100.0 * removed / n_all)
