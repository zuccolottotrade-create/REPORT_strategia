from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import pandas as pd

from .trades import extract_trades_from_hold


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 10_000.0
    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    position_size: float = 1.0


def backtest_from_hold(df: pd.DataFrame, cfg: BacktestConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtest semplice:
    - position = 1 se HOLD==IN al tempo t
    - ret_strategy usa position lag (no look-ahead)
    - costi applicati su cambio posizione (entry/exit)
    """
    dfx = df.copy()
    if "HOLD" not in dfx.columns:
        raise ValueError("Missing HOLD column")

    dfx["position"] = (dfx["HOLD"] == "IN").astype(int)
    dfx["ret_close"] = dfx["close"].pct_change().fillna(0.0)

    dfx["pos_change"] = dfx["position"].diff().fillna(dfx["position"]).abs().astype(int)

    total_cost_bps = float(cfg.fee_bps + cfg.slippage_bps)
    dfx["cost"] = (dfx["pos_change"] > 0).astype(int) * (total_cost_bps / 10_000.0)

    dfx["position_lag"] = dfx["position"].shift(1).fillna(0).astype(int)
    dfx["ret_strategy"] = (cfg.position_size * dfx["position_lag"] * dfx["ret_close"]) - dfx["cost"]

    equity = [float(cfg.initial_capital)]
    for r in dfx["ret_strategy"].iloc[1:]:
        equity.append(equity[-1] * (1.0 + float(r)))
    dfx["equity"] = equity

    dfx["equity_peak"] = dfx["equity"].cummax()
    dfx["drawdown"] = (dfx["equity"] / dfx["equity_peak"]) - 1.0

    equity_df = dfx[["datetime", "close", "HOLD", "position", "ret_close", "ret_strategy", "equity", "drawdown"]].copy()
    trades_df = extract_trades_from_hold(equity_df)
    return equity_df, trades_df
