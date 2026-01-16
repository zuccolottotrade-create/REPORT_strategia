# engine/equity_additive.py
from __future__ import annotations

import pandas as pd


def equity_curve_from_trades_additive(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    initial_capital: float,
    notional_per_trade: float | None = None,  # mantenuto per compatibilità, non usato qui
    cost_per_transaction: float = 0.0,
    transactions_per_trade: int = 2,
) -> pd.DataFrame:
    """
    Equity ADDITIVA (NO compounding) con qty=1:

      pnl_eur (trade) = (exit_price - entry_price) * sign
      equity_additive = initial_capital + cum_pnl_eur - cum_costs

    Costi:
      cum_costs = cost_per_transaction * transactions_per_trade * n_trade_chiusi_cumulati
    """
    if equity_df is None or equity_df.empty:
        raise ValueError("equity_df vuoto: serve la timeline (datetime).")
    if "datetime" not in equity_df.columns:
        raise ValueError("equity_df deve contenere 'datetime'.")

    out = equity_df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    if out["datetime"].isna().all():
        raise ValueError("equity_df['datetime'] non parsabile (tutti NaT).")

    init_cap = float(initial_capital)

    cpt = float(cost_per_transaction or 0.0)
    tpt = int(transactions_per_trade or 0)
    if tpt < 0:
        tpt = 0

    if trades_df is None or (isinstance(trades_df, pd.DataFrame) and trades_df.empty):
        out = out.sort_values("datetime")
        out["cum_pnl_additive"] = 0.0
        out["cum_costs"] = 0.0
        out["equity_additive"] = init_cap
        return out

    if not isinstance(trades_df, pd.DataFrame):
        raise ValueError("trades_df deve essere un DataFrame pandas.")
    if "exit_dt" not in trades_df.columns:
        raise ValueError("trades_df deve contenere 'exit_dt'.")

    t = trades_df.copy()
    t["exit_dt"] = pd.to_datetime(t["exit_dt"], errors="coerce")
    t = t.dropna(subset=["exit_dt"]).sort_values("exit_dt")

    if t.empty:
        out = out.sort_values("datetime")
        out["cum_pnl_additive"] = 0.0
        out["cum_costs"] = 0.0
        out["equity_additive"] = init_cap
        return out

    # 1) Source of truth: pnl_eur già calcolato (qty=1)
    if "pnl_eur" in t.columns:
        t["pnl_trade_eur"] = pd.to_numeric(t["pnl_eur"], errors="coerce").fillna(0.0)
    else:
        # 2) fallback: calcolalo da entry/exit (qty=1)
        if "entry_price" not in t.columns or "exit_price" not in t.columns:
            raise ValueError("trades_df deve contenere 'pnl_eur' oppure entry_price/exit_price.")
        entry = pd.to_numeric(t["entry_price"], errors="coerce")
        exit_ = pd.to_numeric(t["exit_price"], errors="coerce")
        side = t["side"].astype(str).str.upper().str.strip() if "side" in t.columns else pd.Series("LONG", index=t.index)
        sign = pd.Series(1.0, index=t.index)
        sign[side.isin({"SHORT", "SELL", "-1", "S"})] = -1.0
        t["pnl_trade_eur"] = ((exit_ - entry) * sign).fillna(0.0)

    t["cum_pnl"] = t["pnl_trade_eur"].cumsum()

    # Costi cumulati per trade chiuso
    cost_per_trade = cpt * tpt
    t["n_trades_closed"] = range(1, len(t) + 1)
    t["cum_costs"] = t["n_trades_closed"].astype(float) * float(cost_per_trade)

    pnl_ts = t.set_index("exit_dt")["cum_pnl"]
    cost_ts = t.set_index("exit_dt")["cum_costs"]

    out = out.sort_values("datetime")
    cum_pnl_on_bars = pnl_ts.reindex(out["datetime"], method="ffill").fillna(0.0).to_numpy()
    cum_cost_on_bars = cost_ts.reindex(out["datetime"], method="ffill").fillna(0.0).to_numpy()

    out["cum_pnl_additive"] = cum_pnl_on_bars
    out["cum_costs"] = cum_cost_on_bars
    out["equity_additive"] = init_cap + out["cum_pnl_additive"] - out["cum_costs"]
    return out
