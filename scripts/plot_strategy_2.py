#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import webbrowser

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------
# Colori standard (coerenti gated / no_regime)
# ------------------------------------------------------------
COLOR_GATED = "#9467bd"     # viola
COLOR_NO_REGIME = "#ff7f0e" # arancio


# ------------------------------------------------------------
# Defaults (Py_SUITE_TRADING)
# ------------------------------------------------------------
# File: .../Py_SUITE_TRADING/4. REPORT strategia/scripts/plot_strategy_2.py
# parents[0]=scripts, parents[1]=4. REPORT strategia, parents[2]=Py_SUITE_TRADING
SUITE_ROOT = Path(__file__).resolve().parents[2]  # .../Py_SUITE_TRADING
DEFAULT_DATA_DIR = (SUITE_ROOT / "_data" / "Test Data").resolve()


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _to_float_comma(s: pd.Series) -> pd.Series:
    """
    Parser numerico EU robusto:
    - gestisce '1.234,56' -> 1234.56
    - gestisce spazi e simboli comuni
    - lascia intatti i numeri già in formato '1234.56'
    """
    x = s.astype(str).str.strip()

    # rimuovi spazi e simboli comuni (se presenti)
    x = (
        x.str.replace(" ", "", regex=False)
         .str.replace("€", "", regex=False)
         .str.replace("\u00a0", "", regex=False)  # non-breaking space
    )

    # se contiene virgola, assumo formato EU: '.' migliaia, ',' decimali
    m_comma = x.str.contains(",", na=False)
    x_eu = x.where(~m_comma, x.str.replace(".", "", regex=False))
    x_eu = x_eu.str.replace(",", ".", regex=False)

    return pd.to_numeric(x_eu, errors="coerce")

def _dbg_series(df: pd.DataFrame, col: str, parsed: pd.Series, max_show: int = 5) -> None:
    raw = df[col].astype(str)
    ok = int(parsed.notna().sum())
    tot = len(parsed)
    print(f"[DBG] col={col!r} ok={ok}/{tot} dtype_raw={df[col].dtype} dtype_parsed={parsed.dtype}")
    if ok == 0:
        print(f"[DBG] sample raw {col!r}:", raw.head(max_show).tolist())


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
# Plot
# ------------------------------------------------------------
def plot_signal_csv_hold_switch_plotly(csv_path: str, out_base_path: str, fmt: str = "png") -> None:
    df = pd.read_csv(csv_path, sep=";")

    # ============================================================
    # METADATA: Symbol / Strategia (se presenti come colonne)
    # ============================================================
    symbol_val = None
    strategy_val = None
    if "Symbol" in df.columns and len(df) > 0:
        symbol_val = str(df["Symbol"].iloc[0]).strip()
    if "Strategia" in df.columns and len(df) > 0:
        strategy_val = str(df["Strategia"].iloc[0]).strip()

    if "datetime" not in df.columns:
        raise ValueError("Colonna 'datetime' mancante nel CSV.")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")

    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df[c] = _to_float_comma(df[c])

    # --- trade_id robusto ---
    trade_id_col = None
    for cand in ["Trade_ID", "trade_id", "TRADE_ID"]:
        if cand in df.columns:
            trade_id_col = cand
            break

    trade_ids = pd.Series(dtype="float64")
    if trade_id_col is not None:
        trade_ids = pd.to_numeric(df[trade_id_col], errors="coerce")

    # CumPnL
    cum_col = None
    for cand in ["Sum Profit/Trade", "CumPnL", "CUMPNL", "SUM_PROFIT_TRADE"]:
        if cand in df.columns:
            cum_col = cand
            break

    # Profit per trade
    prof_col = None
    for cand in ["Profit/Trade", "profit", "PROFIT_TRADE", "Pnl", "pnl_eur"]:
        if cand in df.columns:
            prof_col = cand
            break

    # ------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.07,
        row_heights=[0.45, 0.25, 0.30],
        subplot_titles=("Price (OHLC)", "CumPnL", "Profit/Trade")
    )

    # ============================================================
    # TITLE + SUBTITLE (Symbol / Strategia)
    # ============================================================
    title_main = "Strategy Performance"
    subtitle = []

    if symbol_val:
        subtitle.append(f"Symbol: {symbol_val}")
    if strategy_val:
        subtitle.append(f"Strategia: {strategy_val}")

    if subtitle:
        fig.update_layout(
            title=dict(
                text=title_main + "<br><span style='font-size:12px; color:gray'>" +
                     " | ".join(subtitle) +
                     "</span>",
                x=0.5,
            )
        )
    else:
        fig.update_layout(title=dict(text=title_main, x=0.5))



    # Row 1: Price
    fig.add_trace(
        go.Candlestick(
            x=df["datetime"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            legendgroup="price",
            legendgrouptitle_text="Price",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # ------------------------------------------------------------
    # Auto-zoom verticale Price (row 1)
    # ------------------------------------------------------------
    ymin = df["low"].min()
    ymax = df["high"].max()
    pad = (ymax - ymin) * 0.05  # 5% padding

    fig.update_yaxes(
        range=[ymin - pad, ymax + pad],
        row=1,
        col=1,
    )

    # Row 2: CumPnL (2 linee: gated e no_regime)
    cum_cols = []
    if "Sum Profit/Trade" in df.columns:
        cum_cols.append(("Sum Profit/Trade", "CumPnL (gated)"))
    if "Sum Profit/Trade_no_regime" in df.columns:
        cum_cols.append(("Sum Profit/Trade_no_regime", "CumPnL (no_regime)"))

    if len(cum_cols) > 0:
        # CumPnL deve essere sincronizzata coi Trade: asse X = Trade_ID (obbligatorio)
        if trade_id_col is None:
            raise ValueError("[CumPnL] Trade_ID mancante: impossibile sincronizzare CumPnL con i Trade.")
        if not trade_ids.notna().any():
            raise ValueError("[CumPnL] Trade_ID presente ma tutti NaN: impossibile usare asse X a Trade_ID.")

        # Trade_ID spesso è valorizzato solo su EXIT: propaghiamo l'ultimo Trade_ID noto
        x_src = trade_ids.ffill()

        for col, label in cum_cols:
            y = _to_float_comma(df[col]) if df[col].dtype == object else pd.to_numeric(df[col], errors="coerce")
            _dbg_series(df, col, y)

            m = y.notna() & x_src.notna()
            x = x_src[m].astype("Int64")
            yy = y[m]

            # colori coerenti
            if col == "Sum Profit/Trade":
                color = COLOR_GATED
            else:
                color = COLOR_NO_REGIME

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=yy,
                    mode="lines",
                    name=label,
                    line_shape="hv",      # scalini
                    line=dict(width=2, color=color),   # colore FORZATO
                    marker=dict(color=color),  # ridondante ma evita override del colorway

                    legendgroup="cumpnl",
                    legendgrouptitle_text="CumPnL",
                    showlegend=True,
                ),
                row=2,
                col=1,
            )

        # Asse X row2: Trade_ID numerico, niente titolo (evita overlap)
        fig.update_xaxes(title_text="", row=2, col=1, type="linear")


    # Row 3: Profit/Trade (2 barre: gated e no_regime)
    prof_cols = []
    if "Profit/Trade" in df.columns:
        prof_cols.append(("Profit/Trade", "Profit/Trade (gated)"))
    if "Profit/Trade_no_regime" in df.columns:
        prof_cols.append(("Profit/Trade_no_regime", "Profit/Trade (no_regime)"))

    if len(prof_cols) > 0:
        use_trade_id_x = (trade_id_col is not None)

        for col, label in prof_cols:
            y = _to_float_comma(df[col]) if df[col].dtype == object else pd.to_numeric(df[col], errors="coerce")
            _dbg_series(df, col, y)
            m_ok = y.notna()

            if use_trade_id_x:
                x = trade_ids[m_ok].astype("Int64")
                # colori coerenti
                if col == "Profit/Trade":
                    color = COLOR_GATED
                else:
                    color = COLOR_NO_REGIME

                fig.add_trace(
                    go.Bar(
                        x=x,
                        y=y[m_ok],
                        name=label,
                        marker=dict(color=color),
                        legendgroup="profit",
                        legendgrouptitle_text="Profit / Trade",
                        showlegend=True,
                    ),
                    row=3,
                    col=1
                )

                fig.update_xaxes(title_text="Trade_ID", row=3, col=1, type="linear", tickmode="linear")
            else:
                x = df.loc[m_ok, "datetime"]
                fig.add_trace(go.Bar(x=x, y=y[m_ok], name=label), row=3, col=1)
                fig.update_xaxes(title_text="Datetime", row=3, col=1)


    # Se row 2 è su Trade_ID, allineiamo gli assi X di row 2 e row 3
    if trade_id_col is not None and (
            ("Sum Profit/Trade" in df.columns) or ("Sum Profit/Trade_no_regime" in df.columns)
    ) and (
            ("Profit/Trade" in df.columns) or ("Profit/Trade_no_regime" in df.columns)
    ):
        fig.update_xaxes(matches="x3", row=2, col=1)

    # Layout (wide)
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        width=1800,
        height=950,
        margin=dict(l=30, r=30, t=120, b=30),

        legend=dict(
                title_text="Legenda",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                groupclick="toggleitem",  # click su  un item non tutto il gruppo
            ),

    )

    out_base = Path(out_base_path).expanduser().resolve()
    out_base.parent.mkdir(parents=True, exist_ok=True)

    fmt = fmt.lower().strip()
    png_path = out_base.with_suffix(".png")
    html_path = out_base.with_suffix(".html")

    if fmt not in ("png", "html", "both"):
        raise ValueError("fmt deve essere: png | html | both")

    # --- PNG (richiede kaleido) ---
    if fmt in ("png", "both"):
        try:
            fig.write_image(str(png_path))
            print(f"[PLOT][OK] PNG scritto: {png_path}")
        except Exception as e:
            print(f"[PLOT][WARN] PNG non scritto: {png_path} err={repr(e)}")

    # --- HTML ---
    if fmt in ("html", "both"):
        try:
            fig.write_html(str(html_path), include_plotlyjs="cdn")
            print(f"[PLOT][OK] HTML scritto: {html_path}")
        except Exception as e:
            print(f"[PLOT][WARN] HTML non scritto: {html_path} err={repr(e)}")

    # --- AUTO-OPEN (HTML) ---
    # In modalità pipeline/macOS è più affidabile usare 'open -a Google Chrome'
    try:
        if not html_path.exists():
            # garantiamo HTML per apertura anche se fmt=png
            fig.write_html(str(html_path), include_plotlyjs="cdn")

        import sys, subprocess
        if sys.platform == "darwin":
            subprocess.run(["open", "-a", "Google Chrome", str(html_path)], check=False)
        else:
            webbrowser.open(html_path.as_uri())
    except Exception as e:
        print("[PLOT][WARN] apertura HTML fallita:", repr(e))


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--format", default="png", help="png | html | both (default=png)")
    p.add_argument("--input", default="", help="Path al SIGNAL_*.csv (se vuoto => selezione interattiva)")
    p.add_argument("--outdir", default=str(DEFAULT_DATA_DIR), help="Directory output (default = _data/Test Data)")
    p.add_argument("--datadir", default=str(DEFAULT_DATA_DIR), help="Directory dove cercare i SIGNAL_*.csv (default = _data/Test Data)")
    p.add_argument("--no-recursive", action="store_true", help="Disabilita ricerca ricorsiva in sottocartelle")
    args = p.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.input.strip():
        in_path = Path(args.input).expanduser().resolve()
    else:
        datadir = args.datadir.strip() or str(DEFAULT_DATA_DIR)
        in_path = pick_signal_file_interactive(datadir, recursive=(not args.no_recursive))

    out_base = outdir / f"PLOT_{in_path.stem}"
    plot_signal_csv_hold_switch_plotly(str(in_path), str(out_base), fmt=args.format)
    print(f"[OK] Plotly plot salvato: {out_base}.[png/html]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
