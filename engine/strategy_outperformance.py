from __future__ import annotations

import pandas as pd
from .base import register_metric


@register_metric(name="Strategy Outperformance", unit="%")
def compute(equity_df: pd.DataFrame, trades_df, params) -> float:
    """
    Outperformance in punti percentuali:
      Strategy Total Return (%) - Buy&Hold Return (%)

    Strategy Total Return: (last_equity / first_equity - 1) * 100
    Buy&Hold Return: (last_close / first_close - 1) * 100
    """
    if equity_df is None or equity_df.empty:
        return 0.0

    # Strategy total return da equity curve
    if "equity" not in equity_df.columns:
        return 0.0
    eq = equity_df["equity"].dropna()
    if len(eq) < 2:
        return 0.0
    eq0 = float(eq.iloc[0])
    eq1 = float(eq.iloc[-1])
    if eq0 == 0:
        return 0.0
    strat_ret = (eq1 / eq0 - 1.0) * 100.0

    # Buy&Hold return da close
    if "close" not in equity_df.columns:
        return 0.0
    cl = equity_df["close"].dropna()
    if len(cl) < 2:
        return 0.0
    c0 = float(cl.iloc[0])
    c1 = float(cl.iloc[-1])
    if c0 == 0:
        return 0.0
    bh_ret = (c1 / c0 - 1.0) * 100.0

    return strat_ret - bh_ret
