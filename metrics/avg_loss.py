from __future__ import annotations

import pandas as pd

from .base import register_metric

# Preferiamo PnL assoluto in € se disponibile
_PNL_COL_CANDIDATES = ["pnl", "PnL", "P&L", "pl", "profit", "net_pnl", "pnl_abs"]
_PNL_PCT_COL = "pnl_pct"

# Fallback: ricostruzione trade da equity_df (IN/OUT + prezzo)
_EVENT_COL_CANDIDATES = ["HOLD", "SIGNAL"]
_PRICE_COL_CANDIDATES = ["close", "Close", "price", "last", "settle"]


def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_events(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _to_numeric_series(s: pd.Series) -> pd.Series:
    """
    Conversione robusta a float:
    - virgole decimali ("0,025")
    - separatori migliaia ("1.234,56")
    - pulizia spazi/NBSP e simbolo €
    """
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        s = (
            s.astype(str)
            .str.replace("\u00a0", "", regex=False)  # NBSP
            .str.replace("€", "", regex=False)
            .str.strip()
            .str.replace(".", "", regex=False)       # separatore migliaia
            .str.replace(",", ".", regex=False)      # virgola decimale -> punto
        )
    return pd.to_numeric(s, errors="coerce")


def _pick_pnl_col(trades_df: pd.DataFrame) -> str | None:
    for c in _PNL_COL_CANDIDATES:
        if c in trades_df.columns:
            return c
    if _PNL_PCT_COL in trades_df.columns:
        return _PNL_PCT_COL
    return None


def _build_trade_pnl_from_equity(equity_df: pd.DataFrame) -> pd.Series:
    """
    Ricostruisce PnL per trade (1 unità) da equity_df usando:
    - eventi/stato IN/OUT su HOLD o SIGNAL
    - prezzo (close di default)

    Usa SOLO le transizioni (fronti), non tutte le righe "IN".
    """
    event_col = _pick_first_existing(equity_df, _EVENT_COL_CANDIDATES)
    price_col = _pick_first_existing(equity_df, _PRICE_COL_CANDIDATES)
    if event_col is None or price_col is None:
        return pd.Series(dtype="float64")

    ev = _normalize_events(equity_df[event_col])
    px = _to_numeric_series(equity_df[price_col])

    if px.isna().all():
        return pd.Series(dtype="float64")

    prev = ev.shift(1)

    # Entry: fronte NON-IN -> IN
    entry_mask = ev.eq("IN") & ~prev.eq("IN")
    # Exit: fronte NON-OUT -> OUT
    exit_mask = ev.eq("OUT") & ~prev.eq("OUT")

    entry_idx = equity_df.index[entry_mask & px.notna()]
    exit_idx = equity_df.index[exit_mask & px.notna()]

    if len(entry_idx) == 0 or len(exit_idx) == 0:
        return pd.Series(dtype="float64")

    pnls: list[float] = []
    exits = exit_idx.tolist()
    j = 0

    for i in entry_idx.tolist():
        while j < len(exits) and exits[j] <= i:
            j += 1
        if j >= len(exits):
            break

        entry = float(px.loc[i])
        exit_ = float(px.loc[exits[j]])
        pnls.append(exit_ - entry)
        j += 1

    return pd.Series(pnls, dtype="float64")


@register_metric(name="AVG Loss", unit="€")
def compute(equity_df, trades_df: pd.DataFrame, params) -> float:
    """
    AVG Loss = media dei PnL negativi per trade.

    Regola:
    - Se trades_df contiene PnL assoluto (non pnl_pct), usa trades_df.
    - Altrimenti fallback: ricostruzione trade da equity_df (fronti IN/OUT).
    """

    # 1) Usa trades_df se contiene PnL assoluto in €
    if trades_df is not None and not trades_df.empty:
        col = _pick_pnl_col(trades_df)
        if col is not None and col != _PNL_PCT_COL:
            s = _to_numeric_series(trades_df[col])
            losses = s[(s < 0) & s.notna()]
            if not losses.empty:
                return float(losses.mean())

    # 2) Fallback equity_df (fronti IN/OUT)
    pnls = _build_trade_pnl_from_equity(equity_df)
    if pnls.empty:
        return 0.0

    losses = pnls[pnls < 0]
    if losses.empty:
        return 0.0

    return float(losses.mean())

