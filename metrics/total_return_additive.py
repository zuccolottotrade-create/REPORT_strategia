# metrics/total_return_additive.py
from __future__ import annotations

import pandas as pd
from .base import register_metric


@register_metric(name="Total Return (Additive)", unit="%")
def compute(equity_df: pd.DataFrame, trades_df, params) -> float:
    if equity_df is None or equity_df.empty:
        return 0.0

    if "equity_additive" not in equity_df.columns:
        return 0.0

    s = pd.to_numeric(equity_df["equity_additive"], errors="coerce").dropna()
    if s.empty:
        return 0.0

    # Start = initial_capital (preferito), altrimenti primo valore della serie
    start = None
    if "initial_capital" in equity_df.columns:
        s0 = pd.to_numeric(equity_df["initial_capital"], errors="coerce").dropna()
        if not s0.empty:
            start = float(s0.iloc[0])

    if start is None:
        start = float(s.iloc[0])

    end = float(s.iloc[-1])

    if start == 0.0:
        return 0.0

    # percentuale
    return ((end - start) / start) * 100.0


    net_profit = end - start
    return (net_profit / start) * 100.0
