from __future__ import annotations

import pandas as pd

from .base import register_metric


@register_metric(name="Buy & Hold Return (First IN -> Last Close)", unit="%")
def compute(equity_df: pd.DataFrame, trades_df, params) -> float:
    """
    Buy & Hold Return operativo (mark-to-market):
      - entry = close della prima riga con evento IN
        (usa HOLD se presente, altrimenti SIGNAL)
      - exit  = close dell'ultima riga disponibile
    Formula:
      (last_close / entry_close - 1) * 100
    """
    if equity_df is None or equity_df.empty:
        return 0.0

    # Colonne necessarie
    if "close" not in equity_df.columns:
        return 0.0

    # Determina colonna eventi
    event_col = "HOLD" if "HOLD" in equity_df.columns else (
        "SIGNAL" if "SIGNAL" in equity_df.columns else None
    )
    if event_col is None:
        return 0.0

    # Serie prezzi ed eventi
    close = pd.to_numeric(equity_df["close"], errors="coerce")
    events = equity_df[event_col].astype(str).str.upper().str.strip()

    # Trova il primo IN
    in_mask = events == "IN"
    if not in_mask.any():
        return 0.0

    first_in_idx = in_mask.idxmax()  # primo True
    entry = close.loc[first_in_idx]

    # Ultimo close valido
    last = close.dropna().iloc[-1]

    if pd.isna(entry) or pd.isna(last) or float(entry) == 0.0:
        return 0.0

    return (float(last) / float(entry) - 1.0) * 100.0

