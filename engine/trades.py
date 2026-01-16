from __future__ import annotations

from typing import Optional
import pandas as pd


def extract_trades_from_hold(equity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estrae round-trip LONG basati su position 0/1 (derivata da HOLD).
    """
    df = equity_df.copy()
    df["pos_lag"] = df["position"].shift(1).fillna(0).astype(int)
    df["change"] = df["position"] - df["pos_lag"]

    trades = []
    open_trade: Optional[dict] = None

    for _, row in df.iterrows():
        dt = row["datetime"]
        price = float(row["close"])
        change = int(row["change"])

        if change == 1 and open_trade is None:
            open_trade = {"entry_dt": dt, "entry_price": price}
        elif change == -1 and open_trade is not None:
            exit_dt = dt
            exit_price = price
            pnl_pct = (exit_price / open_trade["entry_price"]) - 1.0
            trades.append(
                {
                    "side": "LONG",
                    "entry_dt": open_trade["entry_dt"],
                    "exit_dt": exit_dt,
                    "entry_price": float(open_trade["entry_price"]),
                    "exit_price": float(exit_price),
                    "pnl_pct": float(pnl_pct),
                }
            )
            open_trade = None

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["win"] = trades_df["pnl_pct"] > 0
    return trades_df
