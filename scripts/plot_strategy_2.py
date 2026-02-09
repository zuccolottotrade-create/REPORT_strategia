#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _to_float_comma(s: pd.Series) -> pd.Series:
    """Convert EU decimal-comma strings to float, coercing errors to NaN."""
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def pick_signal_file_interactive(data_dir: str, recursive: bool = True) -> Path:
    """
    Interactive picker for files that START WITH 'SIGNAL_' (case-insensitive).
    Searches in data_dir; if recursive=True scans subfolders too.
    """
    d = Path(data_dir).expanduser().resolve()
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"[ERR] Directory non trovata: {d}")

    it = d.rglob("*.csv") if recursive else d.glob("*.csv")

    files = sorted(
        [p for p in it if p.is_file() and p.name.upper().startswith("SIGNAL_")],
        key=lambda p: str(p).lower(),
    )

    if not files:
        raise FileNotFoundError(
            f"[ERR] Nessun file trovato in {d} che inizi con 'SIGNAL_' e finisca con '.csv' "
            f"(recursive={recursive})"
        )

    print("\nSeleziona un file SIGNAL_*.csv:\n")
    for i, p in enumerate(files, start=1):
        print(f"  {i:2d}) {p.relative_to(d)}")

    while True:
        s = input("\nInserisci numero (invio = 1): ").strip()
        if s == "":
            idx = 1
            break
        if s.isdigit() and 1 <= int(s) <= len(files):
            idx = int(s)
            break
        print(f"[ERR] Scelta non valida. Inserisci un numero 1..{len(files)}")

    chosen = files[idx - 1]
    print(f"\n[OK] Selezionato: {chosen}")
    return chosen


# ------------------------------------------------------------
# Plot (Plotly)
# ------------------------------------------------------------
def plot_signal_csv_hold_switch_plotly(csv_path: str, out_base_path: str, fmt: str = "png") -> None:
    """
    Plotly strategy plot:
      - Row 1: Candles + ENTRY/EXIT markers from HOLD switch (OUT->IN / IN->OUT)
      - Row 2: Cum PnL from 'Sum Profit/Trade' (if present) else cumsum('Profit/Trade')
      - Row 3: Bar chart Profit/Trade per Trade_ID (one bar per trade, last value per Trade_ID)

    out_base_path: path WITHOUT extension (e.g. /.../PLOT_SIGNAL_xxx)
    fmt: png | html | both
    """
    csv_path = str(Path(csv_path).expanduser().resolve())
    df = pd.read_csv(csv_path, sep=";")

    # datetime
    if "datetime" not in df.columns:
        raise ValueError("[ERR] Colonna 'datetime' mancante nel CSV.")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")

    # OHLC
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"[ERR] Colonna OHLC mancante: '{c}'")
        df[c] = _to_float_comma(df[c])

    # CumPnL
    if "Sum Profit/Trade" in df.columns:
        df["CumPnL"] = _to_float_comma(df["Sum Profit/Trade"])
    else:
        if "Profit/Trade" not in df.columns:
            raise ValueError("[ERR] Mancano sia 'Sum Profit/Trade' sia 'Profit/Trade'.")
        df["ProfitTrade_num_tmp"] = _to_float_comma(df["Profit/Trade"]).fillna(0.0)
        df["CumPnL"] = df["ProfitTrade_num_tmp"].cumsum()

    # Profit per trade (Trade_ID)
    if "Trade_ID" in df.columns and "Profit/Trade" in df.columns:
        df["TradeID"] = pd.to_numeric(df["Trade_ID"], errors="coerce")
        df["ProfitTrade"] = _to_float_comma(df["Profit/Trade"])

        trades = (
            df.dropna(subset=["TradeID"])
              .sort_values(["TradeID", "datetime"])
              .groupby("TradeID", as_index=False)["ProfitTrade"]
              .last()
        )
        trades = trades.dropna(subset=["ProfitTrade"])
    else:
        trades = pd.DataFrame(columns=["TradeID", "ProfitTrade"])


    # CumPnL per trade (Trade_ID) -> per allineare asse con l'analisi trade
    # Prendiamo l'ULTIMO CumPnL del trade (bar di uscita o ultimo bar con quel Trade_ID)
    if "Trade_ID" in df.columns:
        if "TradeID" not in df.columns:
            df["TradeID"] = pd.to_numeric(df["Trade_ID"], errors="coerce")
        trade_cum = (
            df.dropna(subset=["TradeID"])
              .sort_values(["TradeID", "datetime"])
              .groupby("TradeID", as_index=False)["CumPnL"]
              .last()
        )
        trade_cum = trade_cum.dropna(subset=["CumPnL"])
    else:
        trade_cum = pd.DataFrame(columns=["TradeID", "CumPnL"])


    # HOLD switch (ENTRY/EXIT)
    if "HOLD" not in df.columns:
        raise ValueError("[ERR] Colonna 'HOLD' mancante nel CSV.")
    h = df["HOLD"].astype(str).str.upper().str.strip()
    prev = h.shift(1)
    entry = (prev == "OUT") & (h == "IN")
    exit_ = (prev == "IN") & (h == "OUT")

    # --- Figure: 3 rows (datetime rows 1-2; Trade_ID row 3) ---
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=False,
        vertical_spacing=0.06,
        row_heights=[0.62, 0.23, 0.15],
        subplot_titles=(Path(csv_path).name, "Cum PnL", "Profit/Trade per Trade_ID"),
    )

    # Row 1: candles
    fig.add_trace(
        go.Candlestick(
            x=df["datetime"],
            open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="Price",
        ),
        row=1, col=1
    )

    # Row 1: entry/exit markers
    fig.add_trace(
        go.Scatter(
            x=df.loc[entry, "datetime"],
            y=df.loc[entry, "close"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=10),
            name="ENTRY (OUT→IN)",
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.loc[exit_, "datetime"],
            y=df.loc[exit_, "close"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=10),
            name="EXIT (IN→OUT)",
        ),
        row=1, col=1
    )

    # Row 2: CumPnL (asse X allineato con Trade_ID quando disponibile)
    if len(trade_cum) > 0:
        fig.add_trace(
            go.Scatter(
                x=trade_cum["TradeID"].astype(int).astype(str),
                y=trade_cum["CumPnL"],
                mode="lines+markers",
                name="CumPnL (per Trade_ID)",
            ),
            row=2, col=1
        )
        fig.update_xaxes(title_text="Trade_ID", row=2, col=1, tickangle=-90, type="category")
    else:
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["CumPnL"],
                mode="lines",
                name="CumPnL",
            ),
            row=2, col=1
        )

    # Row 3: Profit/Trade per Trade_ID
    if len(trades) > 0:
        fig.add_trace(
            go.Bar(
                x=trades["TradeID"].astype(int).astype(str),
                y=trades["ProfitTrade"],
                name="Profit/Trade",
            ),
            row=3, col=1
        )
        fig.update_xaxes(title_text="Trade_ID", row=3, col=1, tickangle=-90)

    fig.update_yaxes(zeroline=True, row=3, col=1)

    # Layout (wide)
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        width=1800,
        height=950,
        margin=dict(l=30, r=30, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    out_base = Path(out_base_path).expanduser().resolve()
    out_base.parent.mkdir(parents=True, exist_ok=True)

    fmt = fmt.lower().strip()
    if fmt == "png":
        fig.write_image(str(out_base.with_suffix(".png")))  # requires kaleido
    elif fmt == "html":
        fig.write_html(str(out_base.with_suffix(".html")), include_plotlyjs="cdn")
    elif fmt == "both":
        fig.write_image(str(out_base.with_suffix(".png")))
        fig.write_html(str(out_base.with_suffix(".html")), include_plotlyjs="cdn")
    else:
        raise ValueError("fmt deve essere: png | html | both")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--format", default="png", help="png | html | both (default=png)")
    p.add_argument("--input", default="", help="Path al SIGNAL_*.csv (se vuoto => selezione interattiva)")
    p.add_argument("--outdir", required=True, help="Directory output (base filename = PLOT_<input.stem>)")
    p.add_argument("--datadir", default="", help="Directory dove cercare i SIGNAL_*.csv (default = --outdir)")
    p.add_argument("--no-recursive", action="store_true", help="Disabilita ricerca ricorsiva in sottocartelle")
    args = p.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()

    if args.input.strip():
        in_path = Path(args.input).expanduser().resolve()
    else:
        datadir = args.datadir.strip() or str(outdir)
        in_path = pick_signal_file_interactive(datadir, recursive=(not args.no_recursive))

    out_base = outdir / f"PLOT_{in_path.stem}"
    plot_signal_csv_hold_switch_plotly(str(in_path), str(out_base), fmt=args.format)
    print(f"[OK] Plotly plot salvato: {out_base}.[png/html]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
