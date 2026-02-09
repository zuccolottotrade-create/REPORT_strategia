from pathlib import Path
import argparse
# ... resto

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle


def _to_float_comma(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def plot_signal_csv_hold_switch(csv_path: str, out_png: str):
    df = pd.read_csv(csv_path, sep=";")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")

    for c in ["open", "high", "low", "close"]:
        df[c] = _to_float_comma(df[c])

    # Cum PnL
    if "Sum Profit/Trade" in df.columns:
        df["CumPnL"] = _to_float_comma(df["Sum Profit/Trade"])
    else:
        df["Profit/Trade_num"] = _to_float_comma(df["Profit/Trade"]).fillna(0.0)
        df["CumPnL"] = df["Profit/Trade_num"].cumsum()

    # HOLD switch
    h = df["HOLD"].astype(str).str.upper().str.strip()
    prev = h.shift(1)
    entry = (prev == "OUT") & (h == "IN")
    exit_ = (prev == "IN") & (h == "OUT")

    x = mdates.date2num(np.array(df["datetime"].dt.to_pydatetime()))
    w = 0.7 * np.median(np.diff(x)) if len(x) > 2 else 0.02

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # Candles (outline)
    for xi, o, hi, lo, c in zip(x, df["open"], df["high"], df["low"], df["close"]):
        if np.isnan([o, hi, lo, c]).any():
            continue
        ax1.vlines(xi, lo, hi, linewidth=1)
        y0 = min(o, c)
        height = abs(c - o) or (hi - lo) * 0.001
        ax1.add_patch(Rectangle((xi - w / 2, y0), w, height, fill=False, linewidth=1))

    ax1.scatter(x[entry.to_numpy()], df.loc[entry, "close"], marker="^", s=60)
    ax1.scatter(x[exit_.to_numpy()], df.loc[exit_, "close"], marker="v", s=60)

    ax1.set_title(Path(csv_path).name)
    ax1.set_ylabel("Price")

    ax2.plot(df["datetime"], df["CumPnL"])
    ax2.set_ylabel("Cum PnL")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax2.grid(True)

    fig.autofmt_xdate()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
def pick_signal_file_interactive(data_dir: str) -> Path:
    d = Path(data_dir).expanduser().resolve()
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"[ERR] Directory non trovata: {d}")

    files = sorted([
        p for p in d.iterdir()
        if p.is_file()
           and ("signal_" in p.name.lower())
           and p.suffix.lower() == ".csv"
    ])

    if not files:
        raise FileNotFoundError(f"[ERR] Nessun file trovato in {d} che inizi con 'SIGNAL_' e finisca con '.csv'")

    print("\nSeleziona un file SIGNAL_*.csv:\n")
    for i, p in enumerate(files, start=1):
        print(f"  {i:2d}) {p.name}")

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

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input", default="", help="Path al SIGNAL_*.csv (se vuoto => selezione interattiva)")
    p.add_argument("--outdir", required=True, help="Directory output PNG")
    p.add_argument("--datadir", default="", help="Directory dove cercare i SIGNAL_*.csv (default = dirname di --outdir)")
    args = p.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()

    if args.input.strip():
        in_path = Path(args.input).expanduser().resolve()
    else:
        datadir = args.datadir.strip() or str(outdir)
        in_path = pick_signal_file_interactive(datadir)

    out_png = outdir / f"PLOT_{in_path.stem}.png"

    plot_signal_csv_hold_switch(str(in_path), str(out_png))
    print(f"[OK] Plot salvato: {out_png}")

