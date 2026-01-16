from __future__ import annotations

import pandas as pd
from .base import register_metric


def _get_event_series(equity_df: pd.DataFrame) -> pd.Series:
    candidates = []
    if "HOLD" in equity_df.columns:
        candidates.append("HOLD")
    if "SIGNAL" in equity_df.columns:
        candidates.append("SIGNAL")
    if not candidates:
        raise ValueError("Non trovo colonne eventi trade: attese 'HOLD' o 'SIGNAL'.")

    def norm(col: str) -> pd.Series:
        return equity_df[col].astype(str).str.upper().str.strip()

    def has_in_out(s: pd.Series) -> bool:
        vals = set(s.dropna().unique().tolist())
        return ("IN" in vals) or ("OUT" in vals)

    for col in candidates:
        s = norm(col)
        if has_in_out(s):
            return s

    raise ValueError("Né HOLD né SIGNAL contengono IN/OUT.")


@register_metric(name="Buy & Hold Profit (FILO)", unit="€")
def compute(equity_df: pd.DataFrame, trades_df, params) -> float:
    """
    FILO = First IN, Last OUT-run (exit al primo OUT dell'ultima serie di OUT)

    Entry:
      - close al primo IN

    Exit:
      - se esiste almeno un OUT dopo il primo IN:
          prendi l'ULTIMA sequenza contigua di OUT e usa il PRIMO OUT di quella sequenza
      - se NON esiste alcun OUT:
          close dell'ultima riga disponibile (posizione ancora aperta)

    Nota "POSIZIONE ANCORA APERTA" gestita a livello report.
    """
    if equity_df is None or equity_df.empty:
        return 0.0
    if "close" not in equity_df.columns:
        return 0.0

    close = equity_df["close"].dropna()
    if close.empty:
        return 0.0

    events = _get_event_series(equity_df)

    # Primo IN
    in_mask = events == "IN"
    if not in_mask.any():
        return 0.0

    first_in_idx = equity_df.index[in_mask][0]
    entry_price = float(equity_df.loc[first_in_idx, "close"])

    # Considera solo dal primo IN in avanti
    events_after = events.loc[first_in_idx:]
    equity_after = equity_df.loc[first_in_idx:]

    out_mask = events_after == "OUT"
    if out_mask.any():
        # indice dell'ultimo OUT (fine dell'ultima run)
        out_indices = equity_after.index[out_mask]
        last_out_idx = out_indices[-1]

        # risali finché sei dentro una run contigua di OUT
        # per trovare il PRIMO OUT di questa ultima run
        pos = equity_after.index.get_loc(last_out_idx)
        start_pos = pos
        while start_pos - 1 >= 0 and str(events_after.iat[start_pos - 1]).upper().strip() == "OUT":
            start_pos -= 1

        first_out_of_last_run_idx = equity_after.index[start_pos]
        exit_price = float(equity_df.loc[first_out_of_last_run_idx, "close"])
        return exit_price - entry_price

    # Nessun OUT → ultima riga disponibile
    exit_price = float(equity_df["close"].dropna().iloc[-1])
    return exit_price - entry_price
