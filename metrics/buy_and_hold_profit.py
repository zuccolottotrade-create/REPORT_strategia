from __future__ import annotations

import pandas as pd
from .base import register_metric


@register_metric(name="Buy & Hold Profit", unit="€")
def compute(equity_df: pd.DataFrame, trades_df, params) -> float:
    """
    Profitto assoluto Buy&Hold (per 1 unità):
      last_close - first_close

    Nota: unit="€" è una convenzione di report; in realtà è in "unità prezzo"
    (es. punti) a meno che 1 unità corrisponda a 1€.
    """
    if equity_df is None or equity_df.empty:
        return 0.0
    if "close" not in equity_df.columns:
        return 0.0

    s = equity_df["close"].dropna()
    if len(s) < 2:
        return 0.0

    first = float(s.iloc[0])
    last = float(s.iloc[-1])
    return last - first
