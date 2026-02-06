#!/usr/bin/env python3
from __future__ import annotations

# ============================================================
# BOOTSTRAP – sys.path robusto per suite + modulo report
# ============================================================
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()

# 1) Root suite: da env se disponibile (pipeline), altrimenti fallback
SUITE_ROOT = Path(os.environ.get("PY_SUITE_ROOT", str(HERE.parents[2]))).resolve()

# 2) Root modulo report (contiene app_io/engine/metrics se sono locali al modulo)
REPORT_ROOT = HERE.parents[1].resolve()  # .../4. REPORT strategia

# Mettiamo prima REPORT_ROOT, poi SUITE_ROOT
for p in (REPORT_ROOT, SUITE_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# ============================================================
# IMPORT STANDARD / PROGETTO
# ============================================================
import argparse
import math
from typing import List, Tuple

import numpy as np
import pandas as pd

from shared.paths import DATA_DIR
from app_io.loader import load_signal_csv
from app_io.exporter import export_report_csv

from engine.backtest import BacktestConfig, backtest_from_hold
from engine.equity_additive import equity_curve_from_trades_additive
from metrics import apply_metrics
from pathlib import Path



# ============================================================
# UTILS – Date range + timeframe (per intestazione CSV)
# ============================================================

def _print_report_table(df: pd.DataFrame, out_path: Path, max_rows: int = 30) -> None:
    import pandas as pd

    print("\n==============================")
    print(" REPORT (ANTEPRIMA CSV)")
    print("==============================")
    print(f"Output: {out_path}")

    if df is None or df.empty:
        print("(DataFrame vuoto)\n")
        return

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.max_colwidth", 60)

    print(f"Righe: {len(df)} | Colonne: {len(df.columns)}")
    print("Colonne:", list(df.columns))

    if len(df) <= max_rows:
        print(df.to_string(index=False))
    else:
        head_n = max_rows // 2
        tail_n = max_rows - head_n
        print("\n--- HEAD ---")
        print(df.head(head_n).to_string(index=False))
        print("\n--- TAIL ---")
        print(df.tail(tail_n).to_string(index=False))
    print("")



def _print_df_table(title: str, df, max_rows: int = 20) -> None:
    import pandas as pd

    if df is None or df.empty:
        print(f"\n=== {title} ===\n(DataFrame vuoto)\n")
        return

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 80)

    n = len(df)
    print(f"\n=== {title} ===")
    print(f"Righe: {n} | Colonne: {len(df.columns)}")
    print("Colonne:", list(df.columns))

    if n <= max_rows:
        print(df.to_string(index=False))
    else:
        print("\n--- HEAD ---")
        print(df.head(max_rows // 2).to_string(index=False))
        print("\n--- TAIL ---")
        print(df.tail(max_rows // 2).to_string(index=False))
    print("")






def _strip_prefixes(stem: str) -> str:
    """
    Rimuove prefissi noti per evitare duplicazioni nel REPORT.
    """
    for p in ("REPORT_", "SIGNAL_"):
        if stem.startswith(p):
            stem = stem[len(p):]
    return stem


def extract_date_range(equity_df: pd.DataFrame) -> Tuple[str, str]:
    """
    Estrae Data Inizio e Data Fine dal dataframe equity.
    Formato: YYYY-MM-DD HH:MM:SS
    """
    # 1) DatetimeIndex
    if isinstance(equity_df.index, pd.DatetimeIndex):
        dt = equity_df.index
        start_ts = dt.min()
        end_ts = dt.max()

    # 2) fallback: colonna DATETIME
    elif "DATETIME" in equity_df.columns:
        dt = pd.to_datetime(equity_df["DATETIME"], errors="coerce")
        start_ts = dt.min()
        end_ts = dt.max()

    else:
        raise ValueError(
            "Impossibile determinare Data Inizio/Fine: "
            "manca DatetimeIndex o colonna 'DATETIME'."
        )

    if pd.isna(start_ts) or pd.isna(end_ts):
        raise ValueError("Date Inizio/Fine non valide (NaT).")

    fmt = "%Y-%m-%d %H:%M:%S"
    return start_ts.strftime(fmt), end_ts.strftime(fmt)


def detect_timeframe(equity_df: pd.DataFrame) -> str:
    """
    Rileva il timeframe dai timestamp dell'equity_df.
    Ritorna: '1m','5m','15m','30m','1h','4h','1d' ecc.
    """

    import pandas as pd
    import numpy as np

    # -----------------------------
    # Recupero asse temporale
    # -----------------------------
    if isinstance(equity_df.index, pd.DatetimeIndex):
        dt = equity_df.index
    elif "DATETIME" in equity_df.columns:
        dt = pd.to_datetime(equity_df["DATETIME"], errors="coerce")
    else:
        return "unknown"

    dt = dt.dropna().sort_values()
    if len(dt) < 2:
        return "unknown"

    # -----------------------------
    # Calcolo delta temporali
    # -----------------------------
    deltas = dt.diff().dropna()
    if deltas.empty:
        return "unknown"

    # usa la mediana (robusta a buchi / outlier)
    median_delta = deltas.median()

    # conversione in secondi
    seconds = median_delta.total_seconds()
    if not np.isfinite(seconds) or seconds <= 0:
        return "unknown"

    # -----------------------------
    # Mapping con tolleranza
    # -----------------------------
    TF_MAP = {
        "1m": 60,
        "5m": 5 * 60,
        "15m": 15 * 60,
        "30m": 30 * 60,
        "1h": 60 * 60,
        "2h": 2 * 60 * 60,
        "4h": 4 * 60 * 60,
        "1d": 24 * 60 * 60,
        "1w": 7 * 24 * 60 * 60,
    }

    # tolleranza ±20%
    for tf, tf_seconds in TF_MAP.items():
        if abs(seconds - tf_seconds) / tf_seconds <= 0.20:
            return tf

    # fallback leggibile
    if seconds < 60:
        return "<1m"
    if seconds < 3600:
        return f"{int(round(seconds / 60))}m"
    if seconds < 86400:
        return f"{int(round(seconds / 3600))}h"

    return f"{int(round(seconds / 86400))}d"



# PATH default richiesto (puoi puntarlo alla tua cartella SIGNAL reale)

DEFAULT_SIGNALS_DIR = Path(os.environ.get("PY_SUITE_DATA_DIR", str(DATA_DIR)))
DEFAULT_REPORTS_DIR = Path(os.environ.get("PY_SUITE_OUT_DIR", str(DATA_DIR)))



def ask_yes_no(prompt: str, default_yes: bool = True, **kwargs) -> bool:
    # Compatibilità: alcune chiamate usano default=True/False
    if "default" in kwargs and kwargs["default"] is not None:
        default_yes = bool(kwargs["default"])

    suffix = "Y/n" if default_yes else "y/N"
    while True:
        ans = input(f"{prompt} [{suffix}]: ").strip().lower()
        if not ans:
            return default_yes
        if ans in ("y", "yes", "s", "si", "sì"):
            return True
        if ans in ("n", "no"):
            return False
        print("Risposta non valida. Inserisci y/n.")



def ask_path(prompt: str, default: Path) -> Path:
    raw = input(f"{prompt} [{default}]: ").strip()
    return Path(raw) if raw else default


def ask_choice(items: List[str], prompt: str) -> int:
    if not items:
        raise ValueError("Nessun elemento disponibile per la scelta.")
    while True:
        print("\n" + prompt)
        for i, it in enumerate(items, start=1):
            print(f"  {i}. {it}")
        raw = input("Seleziona numero: ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(items):
                return idx - 1
        print("Scelta non valida. Riprova.")


def ask_float(prompt: str, default: float) -> float:
    """Input float robusto (virgola/punto) con default."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return float(default)
        raw = raw.replace(",", ".")
        try:
            return float(raw)
        except ValueError:
            print("Valore non valido. Inserisci un numero (es. 1.5 oppure 1,5).")


def ask_int(prompt: str, default: int) -> int:
    """Input int robusto con default."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return int(default)
        if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
            return int(raw)
        print("Valore non valido. Inserisci un intero (es. 2).")


def list_signal_files(folder: Path) -> List[Path]:
    """
    Restituisce SOLO i file compatibili con il report:
    - CSV
    - nome che inizia con 'SIGNAL_'
    """
    if not folder.exists():
        return []
    return sorted([p for p in folder.glob("SIGNAL_*.csv") if p.is_file()])


def pick_file_and_load_interactive(signals_dir: Path) -> Tuple[Path, pd.DataFrame]:
    """
    Selezione interattiva:
    - mostra SOLO SIGNAL_*.csv
    - se il CSV non è conforme -> stampa FILE NON ADEGUATAMENTE FORMATTATO e ripete
    """

    # ==========================================================
    # PIPELINE DEFAULT: file SIGNAL già deciso a monte
    # ==========================================================
    env_signal = os.environ.get("PY_SUITE_SIGNAL_INPUT_CSV", "").strip()
    if env_signal:
        src = Path(env_signal)
        if not src.exists():
            raise FileNotFoundError(f"PY_SUITE_SIGNAL_INPUT_CSV non trovato: {src}")

        try:
            df = load_signal_csv(src)
            return src, df
        except Exception as e:
            raise RuntimeError(
                f"File SIGNAL fornito dalla pipeline non valido: {src}\n{e}"
            )

    # ==========================================================
    # FALLBACK: comportamento interattivo originale
    # ==========================================================
    while True:
        files = list_signal_files(signals_dir)
        if not files:
            print(f"\nERRORE: nessun file compatibile trovato in: {signals_dir}")
            print("Attesi file CSV con nome che inizia con 'SIGNAL_'\n")
            raise FileNotFoundError(f"Nessun file SIGNAL_*.csv trovato in: {signals_dir}")

        idx = ask_choice(
            [f.name for f in files],
            f"Seleziona il file SIGNAL_ su cui generare il report (solo SIGNAL_*.csv) (cartella: {signals_dir}):",
        )
        src = files[idx]

        try:
            df = load_signal_csv(src)
            return src, df
        except Exception:
            print("\nFILE NON ADEGUATAMENTE FORMATTATO\n")
            # retry loop


def ask_equity_start(default: float = 100.0) -> float:
    """Chiede Equity Start con prompt esatto richiesto: se invio -> default."""
    while True:
        raw = input(f"Equity Start (default {int(default) if float(default).is_integer() else default}) = ").strip()
        if raw == "":
            return float(default)

        raw = raw.replace(",", ".")
        try:
            value = float(raw)
            if value <= 0:
                print("⚠️ Equity Start deve essere > 0")
                continue
            return value
        except ValueError:
            print("⚠️ Inserisci un numero valido (es. 100 oppure 250.5)")



def derive_equity_start_from_first_in_trade(trades_df: pd.DataFrame, fallback: float = 100.0) -> float:
    """Deriva Equity Start come valore di acquisto della prima operazione IN.

    Regola:
      - prende la prima trade per tempo di ingresso (se disponibile), altrimenti per index
      - equity_start = abs(entry_price * qty) con qty=1 se non presente
      - se non determinabile -> fallback
    """
    try:
        if not isinstance(trades_df, pd.DataFrame) or trades_df.empty:
            return float(fallback)

        df = trades_df.copy()

        # ordine cronologico se possibile
        for tcol in ("entry_time", "entry_datetime", "entry_dt", "entry_date", "open_time"):
            if tcol in df.columns:
                dt = pd.to_datetime(df[tcol], errors="coerce")
                df = df.assign(_entry_dt=dt).sort_values(["_entry_dt"], kind="mergesort")
                break

        first = df.iloc[0]

        entry_price = _coerce_price_scalar(first.get("entry_price", np.nan))
        if not math.isfinite(float(entry_price)):
            return float(fallback)

        qty = None
        for qcol in ("qty", "quantity", "size", "contracts", "n_contracts"):
            if qcol in df.columns:
                qty = _coerce_price_scalar(first.get(qcol, 1.0))
                break
        qty = 1.0 if qty is None or not math.isfinite(float(qty)) else float(qty)

        eq = abs(float(entry_price) * qty)
        if not math.isfinite(eq) or eq <= 0:
            return float(fallback)
        return float(eq)
    except Exception:
        return float(fallback)






def derive_equity_start_from_first_in_signal(df: pd.DataFrame, fallback: float = 100.0) -> float:
    """
    Equity Start = VALUE della prima operazione IN nel file SIGNAL_.

    ⚠️ Nota di dominio (Py_SUITE_TRADING):
      - SIGNAL non assume mai 'IN'
      - la colonna che marca l'ingresso è HOLD == 'IN'
      - il valore monetario da usare è VALUE (prima occorrenza HOLD == 'IN')

    Regola:
      - considera IN dove HOLD == 'IN' (case-insensitive)
      - prende VALUE (deve essere presente e != 0)
      - ritorna abs(VALUE)
      - se nessun IN valido -> fallback
    """
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return float(fallback)

        if "HOLD" not in df.columns or "VALUE" not in df.columns:
            return float(fallback)

        hold = df["HOLD"].astype(str).str.upper().str.strip()
        in_df = df.loc[hold == "IN"]

        if in_df.empty:
            return float(fallback)

        # prima riga HOLD == IN con VALUE valido (!=0)
        for _, row in in_df.iterrows():
            v = _coerce_price_scalar(row.get("VALUE"))
            if math.isfinite(v) and abs(v) > 0:
                return float(abs(v))

        return float(fallback)
    except Exception:
        return float(fallback)


def _debug_df_snapshot(df: pd.DataFrame, label: str, cols: list[str] | None = None, n: int = 5) -> None:
    print("\n" + "=" * 90)
    print(f"[DEBUG] {label}")
    print(f"Righe: {len(df):,} | Colonne: {len(df.columns):,}")
    if "datetime" in df.columns:
        print(f"Datetime min/max: {df['datetime'].min()}  ->  {df['datetime'].max()}")
    key_cols = cols or [c for c in ["symbol", "isin", "datetime", "date", "time", "close", "HOLD"] if c in df.columns]
    if key_cols:
        print("\n[DEBUG] Head:")
        print(df[key_cols].head(n).to_string(index=False))
        print("\n[DEBUG] Tail:")
        print(df[key_cols].tail(n).to_string(index=False))
    print("=" * 90 + "\n")


def _read_report_csv_robust(path: Path) -> pd.DataFrame:
    """Lettura robusta del report CSV esportato (sep=';' e gestione BOM)."""
    try:
        return pd.read_csv(path, sep=";", encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, sep=";", encoding="utf-8")


# =========================
# DEBUG CSV: Trade_ID + Trade_profit
# =========================
def _coerce_price_series(x: pd.Series) -> pd.Series:
    """Converte una serie prezzi a float gestendo virgola/punto e separatori migliaia."""
    if pd.api.types.is_numeric_dtype(x):
        return x.astype(float)

    s = x.astype(str).str.strip()

    has_comma = s.str.contains(",", regex=False)
    s = s.where(~has_comma, s.str.replace(".", "", regex=False))
    s = s.str.replace(",", ".", regex=False)

    return pd.to_numeric(s, errors="coerce")



def _coerce_price_scalar(v: object) -> float:
    """Converte uno scalar prezzo a float gestendo virgola/punto e separatori migliaia."""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return float("nan")
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip()
    if not s:
        return float("nan")
    if "," in s:
        s = s.replace(".", "")
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return float("nan")


def _get_trade_event_series(df: pd.DataFrame) -> pd.Series:
    """Restituisce serie eventi IN/OUT: preferisce HOLD se contiene IN/OUT, altrimenti SIGNAL."""
    candidates = []
    if "HOLD" in df.columns:
        candidates.append("HOLD")
    if "SIGNAL" in df.columns:
        candidates.append("SIGNAL")
    if not candidates:
        raise ValueError("Non trovo colonne eventi trade: attese 'HOLD' o 'SIGNAL'.")

    def norm(col: str) -> pd.Series:
        return df[col].astype(str).str.upper().str.strip()

    def has_in_out(s: pd.Series) -> bool:
        vals = set(s.dropna().unique().tolist())
        return ("IN" in vals) or ("OUT" in vals)

    for col in candidates:
        s = norm(col)
        if has_in_out(s):
            return s

    raise ValueError("Né HOLD né SIGNAL contengono IN/OUT: impossibile calcolare Trade_ID/Trade_profit.")


def add_trade_id_and_profit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trade_ID e Trade_profit robusti su segnali "stato":
    - IN apre un trade SOLO se non siamo già in_trade
    - OUT chiude SOLO se siamo in_trade
    - Trade_profit scritto SOLO su OUT come close(OUT) - close(IN)
    """
    df = df.copy()

    if "close" not in df.columns:
        raise ValueError("Colonna mancante: close")

    df["close"] = _coerce_price_series(df["close"])
    for c in ("open", "high", "low"):
        if c in df.columns:
            df[c] = _coerce_price_series(df[c])

    events = _get_trade_event_series(df)

    df["Trade_ID"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")
    df["Trade_profit"] = np.nan

    trade_id = 0
    in_trade = False
    entry_price = np.nan

    idx_list = df.index.to_list()

    for i in range(len(df)):
        idx = idx_list[i]
        e = events.iat[i]

        if e == "IN":
            if not in_trade:
                trade_id += 1
                in_trade = True
                entry_price = df.loc[idx, "close"]
            df.loc[idx, "Trade_ID"] = trade_id

        elif e == "OUT":
            if in_trade:
                df.loc[idx, "Trade_ID"] = trade_id
                out_price = df.loc[idx, "close"]
                if pd.notna(entry_price) and pd.notna(out_price):
                    df.loc[idx, "Trade_profit"] = out_price - entry_price
            in_trade = False
            entry_price = np.nan

        else:
            if in_trade:
                df.loc[idx, "Trade_ID"] = trade_id

    return df


def add_time_features(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    """Add time-derived columns for reporting/analysis.

    This does **not** alter upstream pipeline files; it only enriches the
    DataFrame used for REPORT_DEBUG / time analytics.

    Adds (if datetime is available):
      - HOUR (0-23)
      - DAY_OF_WEEK (0=Mon)
      - DAY_NAME (localized in English, stable)
      - SESSION (EU/US/ASIA - simple heuristic)
    """
    if df is None or df.empty:
        return df

    if datetime_col in df.columns:
        dt = pd.to_datetime(df[datetime_col], errors="coerce")
    elif "DATETIME" in df.columns:
        dt = pd.to_datetime(df["DATETIME"], errors="coerce")
    elif isinstance(df.index, pd.DatetimeIndex):
        dt = df.index
    else:
        return df

    df = df.copy()
    df["HOUR"] = dt.dt.hour
    df["DAY_OF_WEEK"] = dt.dt.dayofweek
    try:
        df["DAY_NAME"] = dt.dt.day_name()
    except Exception:
        # fallback (very old pandas)
        df["DAY_NAME"] = df["DAY_OF_WEEK"].map({0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"})

    def _session_from_hour(h: float) -> str:
        if pd.isna(h):
            return "NA"
        h = int(h)
        # heuristic for European user; adjust if needed
        if 7 <= h < 16:
            return "EU"
        if 14 <= h < 22:
            return "US"
        return "ASIA"

    df["SESSION"] = df["HOUR"].apply(_session_from_hour)
    return df


def compute_avg_operations_per_week(df: pd.DataFrame, datetime_col: str = "datetime", hold_col: str = "HOLD") -> float:
    """Average number of operations (IN+OUT) per week.

    Definition (robust):
      - an operation occurs when HOLD changes value (0->1 entry, 1->0 exit)
      - operations are counted per ISO week and averaged across weeks with at least 1 operation

    Notes:
      - This function is used in reporting, so it tries to be permissive with column names.
      - If per-week bucketing cannot be computed (missing datetime/hold), caller can fallback to range-based average.
    """
    if df is None or df.empty:
        return float("nan")

    d = df.copy()

    # Datetime: accept common variants
    dt = None
    if datetime_col in d.columns:
        dt = pd.to_datetime(d[datetime_col], errors="coerce")
    else:
        for c in ("DATETIME", "DateTime", "date_time", "timestamp", "Timestamp", "DATE", "Date", "date", "TIME", "Time"):
            if c in d.columns:
                dt = pd.to_datetime(d[c], errors="coerce")
                break
        if dt is None:
            if isinstance(d.index, pd.DatetimeIndex):
                dt = d.index
            else:
                return float("nan")

    # HOLD: accept common variants
    hold_series = None
    if hold_col in d.columns:
        hold_series = d[hold_col]
    else:
        for c in ("HOLD", "hold", "IN_POSITION", "in_position", "IN_POS", "POSITION", "position"):
            if c in d.columns:
                hold_series = d[c]
                break
        if hold_series is None:
            return float("nan")

    hold = pd.to_numeric(hold_series, errors="coerce")
    chg = hold.ne(hold.shift(1))
    # first row may be change from NaN; ignore
    chg.iloc[0] = False

    ops_dt = dt[chg]
    ops_dt = ops_dt.dropna()
    if ops_dt.empty:
        return 0.0

    # ISO week buckets (Mon-Sun)
    ops_week = ops_dt.dt.to_period("W-MON").astype(str)
    counts = ops_week.value_counts()
    if counts.empty:
        return 0.0

    return float(counts.mean())


def compute_time_stats(debug_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compute PnL stats by time buckets from REPORT_DEBUG dataframe.

    Uses Trade_profit (present only on OUT rows) when available.
    Returns:
      - stats_df: aggregated table (HOUR/DAY_NAME/SESSION)
      - summary: best/worst buckets (strings) for quick inclusion in report_df
    """
    if debug_df is None or debug_df.empty:
        return pd.DataFrame(), {}

    df = debug_df.copy()

    # Ensure time features exist
    df = add_time_features(df, datetime_col="datetime")

    # Focus on closed trades (Trade_profit available on OUT rows)
    if "Trade_profit" not in df.columns:
        return pd.DataFrame(), {}

    trade_pnl = pd.to_numeric(df["Trade_profit"], errors="coerce")
    out_mask = trade_pnl.notna()
    tdf = df.loc[out_mask].copy()
    tdf["Trade_profit"] = trade_pnl.loc[out_mask]

    if tdf.empty:
        return pd.DataFrame(), {}

    def _agg(group_cols: list[str]) -> pd.DataFrame:
        g = tdf.groupby(group_cols, dropna=False)["Trade_profit"]
        out = g.agg(Trades="count", Net_Profit="sum", Avg_Profit="mean", Std_Profit="std").reset_index()
        out["Win_Rate"] = g.apply(lambda s: float((s > 0).mean()) if len(s) else float("nan")).values
        return out

    stats_hour = _agg(["HOUR"]).sort_values("HOUR")
    stats_day = _agg(["DAY_OF_WEEK", "DAY_NAME"]).sort_values(["DAY_OF_WEEK"])
    stats_sess = _agg(["SESSION"]).sort_values("SESSION")

    # build a single export-friendly table with a section column
    stats_hour.insert(0, "SECTION", "BY_HOUR")
    stats_day.insert(0, "SECTION", "BY_DAY")
    stats_sess.insert(0, "SECTION", "BY_SESSION")

    stats_df = pd.concat([stats_hour, stats_day, stats_sess], ignore_index=True)

    summary: dict = {}

    try:
        best_hour = stats_hour.loc[stats_hour["Net_Profit"].idxmax()]
        worst_hour = stats_hour.loc[stats_hour["Net_Profit"].idxmin()]
        summary["Best Hour (Net Profit)"] = f"{int(best_hour['HOUR']):02d}:00 (Net={best_hour['Net_Profit']:.2f}, Trades={int(best_hour['Trades'])})"
        summary["Worst Hour (Net Profit)"] = f"{int(worst_hour['HOUR']):02d}:00 (Net={worst_hour['Net_Profit']:.2f}, Trades={int(worst_hour['Trades'])})"
    except Exception:
        pass

    try:
        best_day = stats_day.loc[stats_day["Net_Profit"].idxmax()]
        worst_day = stats_day.loc[stats_day["Net_Profit"].idxmin()]
        summary["Best Day (Net Profit)"] = f"{best_day['DAY_NAME']} (Net={best_day['Net_Profit']:.2f}, Trades={int(best_day['Trades'])})"
        summary["Worst Day (Net Profit)"] = f"{worst_day['DAY_NAME']} (Net={worst_day['Net_Profit']:.2f}, Trades={int(worst_day['Trades'])})"
    except Exception:
        pass

    try:
        best_sess = stats_sess.loc[stats_sess["Net_Profit"].idxmax()]
        worst_sess = stats_sess.loc[stats_sess["Net_Profit"].idxmin()]
        summary["Best Session (Net Profit)"] = f"{best_sess['SESSION']} (Net={best_sess['Net_Profit']:.2f}, Trades={int(best_sess['Trades'])})"
        summary["Worst Session (Net Profit)"] = f"{worst_sess['SESSION']} (Net={worst_sess['Net_Profit']:.2f}, Trades={int(worst_sess['Trades'])})"
    except Exception:
        pass

    return stats_df, summary


def export_debug_signal_csv(path: Path, df: pd.DataFrame) -> None:
    """
    Esporta debug CSV con separatore ';' e virgola decimale (Excel IT).
    Stampa a video un'anteprima tabellare di ciò che viene scritto.
    """
    import pandas as pd

    # ===== STAMPA TABELLARE DI DEBUG =====
    if df is None or df.empty:
        print("\n[REPORT] DataFrame vuoto: nessuna riga da esportare\n")
    else:
        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", 80)
        pd.set_option("display.max_colwidth", 40)

        n = len(df)
        print("\n=== ANTEPRIMA CSV REPORT (in scrittura) ===")
        print(f"Path: {path}")
        print(f"Righe: {n} | Colonne: {len(df.columns)}")
        print("Colonne:", list(df.columns))

        if n <= 20:
            print(df.to_string(index=False))
        else:
            print("\n--- HEAD ---")
            print(df.head(10).to_string(index=False))
            print("\n--- TAIL ---")
            print(df.tail(10).to_string(index=False))
        print("")

    # ===== SCRITTURA CSV =====
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path_or_buf=path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
        decimal=",",
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="REPORT Strategia: genera report CSV da file SIGNAL (HOLD IN/OUT).")
    p.add_argument("--input", type=str, default="", help="Path CSV input (se vuoto: scelta da menu)")
    p.add_argument("--signals-dir", type=str, default=str(DEFAULT_SIGNALS_DIR), help="Cartella segnali")
    p.add_argument("--out-dir", type=str, default="", help="Cartella output report (default: ./reports)")
    p.add_argument("--fee-bps", type=float, default=0.0, help="Commissioni per evento (bps)")
    p.add_argument("--slippage-bps", type=float, default=0.0, help="Slippage per evento (bps)")
    p.add_argument("--initial-capital", type=float, default=10_000.0, help="Capitale iniziale (fallback)")

    # costi transazione additive (fissi)
    p.add_argument("--cost-per-transaction", type=float, default=0.0, help="Costo unitario per transazione (€).")
    p.add_argument(
        "--transactions-per-trade",
        type=int,
        default=2,
        help="Numero transazioni per trade (default 2: entry+exit).",
    )

    p.add_argument(
        "--verify-export",
        action="store_true",
        help="Rilegge il CSV esportato e verifica che contenga la colonna 'Indicatore' e tutte le righe.",
    )
    return p


def _get_metric_raw(report_df: pd.DataFrame, name: str) -> float:
    s = report_df.loc[report_df["Indicatore"] == name, "Valore_raw"]
    if len(s) == 0:
        return float("nan")
    try:
        return float(s.iloc[0])
    except Exception:
        return float("nan")




# ============================================================
# MTM / Unrealized PnL (posizioni aperte)
# ============================================================

def _to_float_eu_safe(x) -> float:
    """Converte numeri EU (es. '1.234,56') in float 1234.56. Ritorna nan se non convertibile."""
    if x is None:
        return float("nan")
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    # EU -> float: "1.234,56" -> "1234.56"
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return float("nan")


def compute_unrealized_pnl_open(signal_df: pd.DataFrame) -> dict:
    """Calcola PnL non realizzato (mark-to-market) per eventuale posizione ancora aperta.

    Assunzioni sul tracciato SIGNAL_*.csv (Run_strategia):
      - HOLD: 'IN' se posizione aperta, 'OUT' se flat (o riga di uscita)
      - SIGNAL: 'LONG' / 'SHORT' sull'entry
      - VALUE: prezzo di entry (sulla riga di entry)
      - close: ultimo prezzo (sull'ultima riga del file)
      - datetime: timestamp (se presente)

    Ritorna:
      {
        has_open: bool,
        side: 'LONG'|'SHORT'|None,
        entry_price: float,
        last_close: float,
        unrealized_pnl: float,
        entry_datetime: str
      }
    """
    out = {
        "has_open": False,
        "side": None,
        "entry_price": float("nan"),
        "last_close": float("nan"),
        "unrealized_pnl": 0.0,
        "entry_datetime": "",
    }

    if signal_df is None or signal_df.empty:
        return out

    if "HOLD" not in signal_df.columns:
        return out

    hold = signal_df["HOLD"].astype(str).fillna("").str.strip().str.upper()
    if hold.iloc[-1] != "IN":
        return out  # nessuna posizione aperta a fine file

    # entry = ultima transizione (prev != IN) -> IN
    prev = hold.shift(1).fillna("")
    entries = signal_df[(hold == "IN") & (prev != "IN")]
    if entries.empty:
        # fallback: prima riga IN disponibile
        entries = signal_df[hold == "IN"]
        if entries.empty:
            return out

    entry_row = entries.iloc[-1]

    side = str(entry_row.get("SIGNAL", "")).strip().upper()
    if side not in ("LONG", "SHORT"):
        # fallback: se non è valorizzato correttamente, assumiamo LONG ma segnaliamo via side
        side = "LONG"

    entry_price = _to_float_eu_safe(entry_row.get("VALUE", entry_row.get("close")))
    last_close = _to_float_eu_safe(signal_df.iloc[-1].get("close"))

    if not (math.isfinite(entry_price) and math.isfinite(last_close)):
        return out

    unreal = (last_close - entry_price) if side == "LONG" else (entry_price - last_close)

    out.update(
        {
            "has_open": True,
            "side": side,
            "entry_price": float(entry_price),
            "last_close": float(last_close),
            "unrealized_pnl": float(unreal),
            "entry_datetime": str(entry_row.get("datetime", "")),
        }
    )
    return out



def _upsert_metric(report_df: pd.DataFrame, name: str, value_raw: float, unit: str) -> pd.DataFrame:
    value_str = f"{value_raw:.3f}".replace(".", ",") if math.isfinite(value_raw) else "nan"

    mask = report_df["Indicatore"] == name
    if mask.any():
        report_df.loc[mask, "Valore_raw"] = value_raw
        report_df.loc[mask, "Valore"] = value_str
        report_df.loc[mask, "Unità"] = unit
        return report_df

    # altrimenti aggiungi nuova riga
    return pd.concat(
        [
            report_df,
            pd.DataFrame(
                [
                    {
                        "Indicatore": name,
                        "Valore": value_str,
                        "Unità": unit,
                        "Valore_raw": value_raw,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )



def main() -> int:
    try:
        args = build_argparser().parse_args()

        # -------------------------
        # Equity Start (richiesta all'avvio) - default 100 se input vuoto
        # -------------------------
        # Equity Start: non interattivo. Verrà derivato dalla prima operazione IN (entry_price*qty) se disponibile.
        equity_start = 100.0  # fallback; verrà ricalcolato dopo il backtest se ci sono trade

        # -------------------------
        # Signals dir
        # -------------------------
        # ==========================================================
        # PATHS: se la pipeline ha già deciso, NON richiedere conferma
        # ==========================================================
        import os as _os  # evita conflitti con variabili locali chiamate "os"

        env_data_dir = _os.environ.get("PY_SUITE_DATA_DIR", "").strip()
        env_out_dir = _os.environ.get("PY_SUITE_OUT_DIR", "").strip()

        # signals_dir
        if env_data_dir:
            signals_dir = Path(env_data_dir)
        else:
            signals_dir = Path(args.signals_dir)

        # Conferma path di default SOLO se non arriva dalla pipeline
        if (not env_data_dir) and str(signals_dir) == str(DEFAULT_SIGNALS_DIR):
            if not ask_yes_no(
                    f"Confermi il path di default dei file SIGNAL?\n{DEFAULT_SIGNALS_DIR}",
                    default=True,
            ):
                signals_dir = ask_path("Inserisci il path alternativo dei file SIGNAL", DEFAULT_SIGNALS_DIR)

        # out_dir
        if env_out_dir:
            out_dir = Path(env_out_dir)
        else:
            out_dir = DEFAULT_REPORTS_DIR

        # ------------------------------------------------------------
        # Output dir (reports)
        # - Se arriva dalla pipeline (env_out_dir), nessun prompt
        # - Altrimenti: conferma default, se NO chiedi path alternativo
        # ------------------------------------------------------------
        if env_out_dir:
            out_dir = Path(env_out_dir)
        else:
            out_dir = DEFAULT_REPORTS_DIR

            if not ask_yes_no(
                    f"Confermi la directory di default dove stampare i report?\n{DEFAULT_REPORTS_DIR}",
                    default_yes=True,
            ):
                out_dir = ask_path(
                    "Inserisci la directory alternativa dove stampare i report",
                    DEFAULT_REPORTS_DIR,
                )

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Report directory: {out_dir.resolve()}")

        # -------------------------
        # Input file
        # -------------------------
        if args.input.strip():
            src = Path(args.input)

            # Hard-check: Report accetta SOLO SIGNAL_*.csv
            if not src.name.startswith("SIGNAL_"):
                print("\nFILE NON ADEGUATAMENTE FORMATTATO")
                print("Motivo: il report accetta solo file con nome che inizia con 'SIGNAL_'\n")
                src, df = pick_file_and_load_interactive(signals_dir)
            else:
                try:
                    df = load_signal_csv(src)
                except Exception:
                    print("\nFILE NON ADEGUATAMENTE FORMATTATO\n")
                    src, df = pick_file_and_load_interactive(signals_dir)
        else:
            src, df = pick_file_and_load_interactive(signals_dir)

        # --- datetime robusto (mantiene interattività invariata) ---
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", dayfirst=True)
        elif "date" in df.columns and "time" in df.columns:
            df["datetime"] = pd.to_datetime(
                df["date"].astype(str).str.strip()
                + " "
                + df["time"].astype(str).str.strip(),
                errors="coerce",
                dayfirst=True,
            )

        # --- warning se datetime non parsabile ---
        if "datetime" in df.columns and df["datetime"].isna().all():
            print(
                "[WARN] Colonna 'datetime' presente ma non parsabile (tutti NaT). "
                "Verifica formato date/time nel CSV."
            )

        _debug_df_snapshot(
            df,
            "DOPO LOAD CSV (prima backtest)",
            cols=[c for c in ["symbol", "isin", "datetime", "close", "HOLD", "SIGNAL"] if c in df.columns],
        )

        # --- datetime robusto (mantiene interattività invariata) ---
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", dayfirst=True)
        elif "date" in df.columns and "time" in df.columns:
            df["datetime"] = pd.to_datetime(
                (df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()),
                errors="coerce",
                dayfirst=True
            )

        # -------------------------
        # Equity Start (AUTO): primo IN da HOLD == 'IN' (VALUE)
        # -------------------------
        equity_start = derive_equity_start_from_first_in_signal(df, fallback=float(equity_start))
        print(f"[INFO] Equity Start (AUTO, first IN) = {equity_start}")

        # -------------------------
        # Config backtest (usa equity_start)
        # -------------------------
        cfg = BacktestConfig(
            initial_capital=float(equity_start),
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slippage_bps),
            position_size=1.0,
        )


        # -------------------------
        # DEBUG / TIME FEATURES (derived from SIGNAL datetime)
        # - calcoliamo Trade_ID + Trade_profit (solo su OUT)
        # - aggiungiamo feature temporali (HOUR/DAY/SESSION)
        # - pre-calcoliamo anche le statistiche time-based per export dedicato
        # -------------------------
        try:
            debug_df = add_trade_id_and_profit(df)
        except Exception as e:
            print(f"[WARN] Impossibile calcolare Trade_ID/Trade_profit (pre): {e}")
            debug_df = df.copy()
            debug_df["Trade_ID"] = pd.Series([pd.NA] * len(debug_df), dtype="Int64")
            debug_df["Trade_profit"] = np.nan

        debug_df = add_time_features(debug_df, datetime_col="datetime")
        time_stats_df, time_summary = compute_time_stats(debug_df)
        # -------------------------
        # Input costi transazione
        # -------------------------
        cost_per_transaction = ask_float(
            "Costo unitario per transazione (€, 0 = nessun costo)",
            float(args.cost_per_transaction),
        )
        # Numero transazioni per trade: FISSO (entry + exit)
        transactions_per_trade = 2

        # -------------------------
        # Backtest + equity additive
        # -------------------------
        print(f"[INFO] Carico: {src}")
        print("[INFO] Backtest basato su HOLD (IN/OUT)...")
        equity_df, trades_df = backtest_from_hold(df, cfg)
        # Equity Start: per specifica deve provenire dal primo IN (VALUE) nel SIGNAL_.
        # Se per qualsiasi motivo non fosse determinabile (fallback), proviamo come ultima chance dal trades_df.
        if float(equity_start) == 100.0:
            equity_start = derive_equity_start_from_first_in_trade(trades_df, fallback=float(equity_start))

        # ------------------------------------------------------------
        # CHECK: strategia senza trade
        # ------------------------------------------------------------
        no_trades = False

        if (
                trades_df is None
                or trades_df.empty
                or ("qty" in trades_df.columns and trades_df["qty"].abs().sum() == 0)
        ):
            no_trades = True
            print("[WARNING] Attenzione: la strategia NON ha generato alcun TRADE.")

        # =========================
        # Canonical PnL in € per trade (qty = 1)
        # LONG:  exit - entry
        # SHORT: entry - exit
        # =========================
        if not no_trades:
            for col in ["entry_price", "exit_price"]:
                if col not in trades_df.columns:
                    print(
                        f"[WARNING] trades_df non contiene '{col}': impossibile calcolare pnl_eur; verificare mapping/colonne del backtest.")
                    no_trades = True
                    break

        if not no_trades:
            entry = _coerce_price_series(trades_df["entry_price"])
            exit_ = _coerce_price_series(trades_df["exit_price"])

            side = (
                trades_df["side"].astype(str).str.upper().str.strip()
                if "side" in trades_df.columns
                else pd.Series("LONG", index=trades_df.index)
            )
            sign = pd.Series(1.0, index=trades_df.index)
            sign[side.isin({"SHORT", "SELL", "-1", "S"})] = -1.0

            trades_df["pnl_eur"] = ((exit_ - entry) * sign).fillna(0.0)

        # --- SAFETY: gestione caso "nessun trade" ---
        if trades_df is None:
            trades_df = pd.DataFrame()

        if "pnl_eur" not in trades_df.columns:
            trades_df["pnl_eur"] = pd.Series(dtype="float64")

        n_trades = int(len(trades_df))

        win_sum = (
            float(trades_df.loc[trades_df["pnl_eur"] > 0, "pnl_eur"].sum())
            if n_trades > 0 else 0.0
        )

        loss_sum = (
            float(trades_df.loc[trades_df["pnl_eur"] < 0, "pnl_eur"].sum())
            if n_trades > 0 else 0.0
        )

        print(
            "[TRADES]",
            "n=", n_trades,
            "win=", win_sum,
            "loss=", loss_sum,
        )

        # ✅ curva equity_additive
        equity_df = equity_curve_from_trades_additive(
            equity_df=equity_df,
            trades_df=trades_df,
            initial_capital=float(equity_start),
            notional_per_trade=None,  # qty=1 -> non serve
            cost_per_transaction=float(cost_per_transaction),
            transactions_per_trade=int(transactions_per_trade),
        )
        # ✅ Data Inizio / Data Fine + Timeframe (dal file SIGNAL_, colonna 'datetime')
        dt = pd.to_datetime(df["datetime"], errors="coerce").dropna().sort_values()
        if len(dt) < 2:
            raise ValueError("Impossibile calcolare Data Inizio/Fine/Timeframe: datetime insufficienti nel SIGNAL_.")

        data_inizio = dt.iloc[0].strftime("%Y-%m-%d %H:%M:%S")
        data_fine = dt.iloc[-1].strftime("%Y-%m-%d %H:%M:%S")

        delta = dt.diff().dropna().mode().iloc[0]
        seconds = int(delta.total_seconds())

        if seconds < 60:
            timeframe = f"{seconds}s"
        elif seconds % 60 == 0 and seconds < 3600:
            timeframe = f"{seconds // 60}m"
        elif seconds % 3600 == 0 and seconds < 86400:
            timeframe = f"{seconds // 3600}h"
        elif seconds % 86400 == 0:
            timeframe = f"{seconds // 86400}d"
        else:
            timeframe = str(delta)

        # utile per metriche che leggono initial_capital
        equity_df["initial_capital"] = float(equity_start)

        # debug: conferma colonne additive presenti
        print("[DEBUG] equity_df columns AFTER equity_additive:", list(equity_df.columns))
        if "datetime" in equity_df.columns and "equity_additive" in equity_df.columns:
            print(
                "[DEBUG] equity_additive tail:\n",
                equity_df[["datetime", "equity_additive"]].tail(5).to_string(index=False),
            )

        # -------------------------
        # Report metriche
        # -------------------------
        import os

        print("\n[DEBUG] RUNNING FILE:", os.path.abspath(__file__))
        print("[DEBUG] BEFORE apply_metrics() -> trades_df columns:", list(trades_df.columns))
        print("[DEBUG] trades_df head:\n", trades_df.head(3).to_string(index=False))

        gp_dbg = (
            trades_df.loc[trades_df.get("pnl_trade_eur", pd.Series(dtype=float)) > 0, "pnl_trade_eur"].sum()
            if "pnl_trade_eur" in trades_df.columns
            else None
        )
        gl_dbg = (
            trades_df.loc[trades_df.get("pnl_trade_eur", pd.Series(dtype=float)) < 0, "pnl_trade_eur"].sum()
            if "pnl_trade_eur" in trades_df.columns
            else None
        )
        if gp_dbg is not None and gl_dbg is not None:
            print("[DEBUG] trades_df pnl_trade_eur sums: GP=", gp_dbg, " GL=", gl_dbg, " GP+GL=", gp_dbg + gl_dbg)

        print("[INFO] Costruisco tabella report metriche...")
        print("[INFO] Costruisco tabella report metriche...")
        print("[INFO] Costruisco tabella report metriche...")

        from metrics.base import get_registry
        from metrics.base import get_registry

        reg = get_registry()
        if isinstance(reg, dict):
            names = sorted(reg.keys())
        else:
            names = []
            for m in reg:
                if isinstance(m, dict) and "name" in m:
                    names.append(m["name"])
                else:
                    names.append(getattr(m, "name", str(m)))
            names = sorted(names)

        print("[DEBUG] Registered metrics (first 50):", names[:50])

        # =========================
        # Calcolo metriche
        # =========================
        print("[DEBUG] pnl_eur present?", "pnl_eur" in trades_df.columns)
        if "pnl_eur" in trades_df.columns:
            s = pd.to_numeric(trades_df["pnl_eur"], errors="coerce")
            print("[DEBUG] pnl_eur describe:\n", s.describe())
            print("[DEBUG] pnl_eur NaN count:", int(s.isna().sum()), " / ", len(s))

        from metrics.gross_pnl_outlier_filtered import (
            gross_profit_outlier_filtered,
            gross_loss_outlier_filtered,
            net_profit_outlier_filtered,
            outliers_removed_count,
            outliers_removed_pct,
        )

        # =========================
        # MTM (Mark-to-Market): include PnL non realizzato su posizione aperta (HOLD=IN a fine file)
        # - aggiunge righe KPI: Unrealized PnL (Open), Net Profit/Gross Profit/Gross Loss (Realized e MTM)
        # - sovrascrive Net Profit / Gross Profit / Gross Loss con versione MTM (coerente con la data di fine file)
        # =========================
        try:
            if "HOLD" in df.columns:
                hold = df["HOLD"].astype(str).fillna("")
                if len(hold) > 0 and hold.iloc[-1] == "IN":
                    prev = hold.shift(1).fillna("")
                    entries = df[(hold == "IN") & (prev != "IN")]
                    if len(entries) == 0:
                        entries = df[hold == "IN"]
                    if len(entries) > 0:
                        entry_row = entries.iloc[-1]
                        side = str(entry_row.get("SIGNAL", "")).strip().upper()
                        if side in ("LONG", "SHORT"):
                            entry_price = _to_float_eu(entry_row.get("VALUE", entry_row.get("close")))
                            last_close = _to_float_eu(df.iloc[-1].get("close"))
                            if math.isfinite(entry_price) and math.isfinite(last_close):
                                unreal = (last_close - entry_price) if side == "LONG" else (entry_price - last_close)
                                # snapshot realized KPI
                                net_real = _get_metric_raw(report_df, "Net Profit")
                                gp_real = _get_metric_raw(report_df, "Gross Profit")
                                gl_real = _get_metric_raw(report_df, "Gross Loss")
                                if math.isfinite(net_real):
                                    report_df = _upsert_metric(report_df, "Net Profit (Realized)", float(net_real), "€")
                                if math.isfinite(gp_real):
                                    report_df = _upsert_metric(report_df, "Gross Profit (Realized)", float(gp_real), "€")
                                if math.isfinite(gl_real):
                                    report_df = _upsert_metric(report_df, "Gross Loss (Realized)", float(gl_real), "€")
                                # open position details
                                report_df = _upsert_metric(report_df, "Open Pos Entry Price", float(entry_price), "€")
                                report_df = _upsert_metric(report_df, "Open Pos Last Close", float(last_close), "€")
                                report_df = _upsert_metric(report_df, "Unrealized PnL (Open)", float(unreal), "€")
                                # MTM totals
                                base_net = float(net_real) if math.isfinite(net_real) else 0.0
                                net_mtm = base_net + float(unreal)
                                report_df = _upsert_metric(report_df, "Net Profit (MTM)", float(net_mtm), "€")
                                report_df = _upsert_metric(report_df, "Net Profit", float(net_mtm), "€")
                                # Gross Loss nel report è tipicamente NEGATIVA (somma delle loss).
                                if math.isfinite(gp_real):
                                    gp_mtm = float(gp_real) + max(float(unreal), 0.0)
                                    report_df = _upsert_metric(report_df, "Gross Profit (MTM)", float(gp_mtm), "€")
                                    report_df = _upsert_metric(report_df, "Gross Profit", float(gp_mtm), "€")
                                if math.isfinite(gl_real):
                                    gl_mtm = float(gl_real) + min(float(unreal), 0.0)
                                    report_df = _upsert_metric(report_df, "Gross Loss (MTM)", float(gl_mtm), "€")
                                    report_df = _upsert_metric(report_df, "Gross Loss", float(gl_mtm), "€")
        except Exception as _e:
            # MTM è best-effort: non deve bloccare il report
            pass

        print("[DEBUG] DIRECT metric call GP(-Outlier):", gross_profit_outlier_filtered(equity_df, trades_df))
        print("[DEBUG] DIRECT metric call GL(-Outlier):", gross_loss_outlier_filtered(equity_df, trades_df))
        print("[DEBUG] DIRECT metric call NP(-Outlier):", net_profit_outlier_filtered(equity_df, trades_df))
        print("[DEBUG] DIRECT metric call Outliers count:", outliers_removed_count(equity_df, trades_df))
        print("[DEBUG] DIRECT metric call Outliers %:", outliers_removed_pct(equity_df, trades_df))

        report_df = apply_metrics(equity_df, trades_df)
        # ------------------------------------------------------------
        # Buy & Hold:
        # - mantieni "Buy & Hold Profit (FILO)" con la definizione originale (da apply_metrics)
        # - aggiungi "Equity End Buy&Hold" = Buy&Hold Profit (FILO) + Equity Start
        # ------------------------------------------------------------
        try:
            bh_filo_raw = _get_metric_raw(report_df, "Buy & Hold Profit (FILO)")
            if math.isfinite(bh_filo_raw) and math.isfinite(float(equity_start)):
                report_df = _upsert_metric(
                    report_df,
                    "Equity End Buy&Hold",
                    float(bh_filo_raw) + float(equity_start),
                    "€",
                )
        except Exception as e:
            print(f"[WARN] Impossibile calcolare Equity End Buy&Hold: {e}")
        # ------------------------------------------------------------
        # B) Buy & Hold Profit (FILO) - MTM: se posizione ancora aperta, aggiungi unrealized fino all'ultima close
        # ------------------------------------------------------------
        try:
            bh_filo_raw = _get_metric_raw(report_df, "Buy & Hold Profit (FILO)")
            if math.isfinite(bh_filo_raw) and "HOLD" in df.columns:
                hold = df["HOLD"].astype(str).fillna("")
                if len(hold) > 0 and hold.iloc[-1] == "IN":
                    prev = hold.shift(1).fillna("")
                    entries = df[(hold == "IN") & (prev != "IN")]
                    if len(entries) == 0:
                        entries = df[hold == "IN"]

                    if len(entries) > 0:
                        entry_row = entries.iloc[-1]
                        side = str(entry_row.get("SIGNAL", "")).strip().upper()
                        if side in ("LONG", "SHORT"):
                            entry_price = _to_float_eu(entry_row.get("VALUE", entry_row.get("close")))
                            last_close = _to_float_eu(df.iloc[-1].get("close"))

                            if math.isfinite(entry_price) and math.isfinite(last_close):
                                unreal = (last_close - entry_price) if side == "LONG" else (entry_price - last_close)

                                # snapshot versione "realized" (quella originale da apply_metrics)
                                report_df = _upsert_metric(
                                    report_df, "Buy & Hold Profit (FILO) (Realized)", float(bh_filo_raw), "€"
                                )

                                # MTM: realized + unrealized
                                bh_filo_mtm = float(bh_filo_raw) + float(unreal)
                                report_df = _upsert_metric(
                                    report_df, "Buy & Hold Profit (FILO) (MTM)", float(bh_filo_mtm), "€"
                                )
                                # sovrascrive la metrica principale che guardi nel report
                                report_df = _upsert_metric(
                                    report_df, "Buy & Hold Profit (FILO)", float(bh_filo_mtm), "€"
                                )

                                # ricalcola Equity End Buy&Hold coerente col MTM
                                if math.isfinite(float(equity_start)):
                                    report_df = _upsert_metric(
                                        report_df,
                                        "Equity End Buy&Hold",
                                        float(bh_filo_mtm) + float(equity_start),
                                        "€",
                                    )
        except Exception:
            pass


        # ============================================================
        # MTM (Mark-to-Market): includi PnL NON realizzato di eventuale posizione aperta
        # - Overwrite: Net Profit / Gross Profit / Gross Loss diventano MTM
        # - Preserve: aggiungiamo le versioni "Realized" + la voce "Unrealized PnL (Open)"
        # ============================================================
        try:
            unr = compute_unrealized_pnl_open(df)

            if unr.get("has_open", False):
                unreal = float(unr.get("unrealized_pnl", 0.0))

                # leggi versioni REALIZZATE (come calcolate da apply_metrics)
                net_real = _get_metric_raw(report_df, "Net Profit")
                gp_real = _get_metric_raw(report_df, "Gross Profit")
                gl_real = _get_metric_raw(report_df, "Gross Loss")

                # preserva i realizzati
                if math.isfinite(net_real):
                    report_df = _upsert_metric(report_df, "Net Profit (Realized)", float(net_real), "€")
                if math.isfinite(gp_real):
                    report_df = _upsert_metric(report_df, "Gross Profit (Realized)", float(gp_real), "€")
                if math.isfinite(gl_real):
                    report_df = _upsert_metric(report_df, "Gross Loss (Realized)", float(gl_real), "€")

                # aggiungi unrealized (open)
                report_df = _upsert_metric(report_df, "Unrealized PnL (Open)", unreal, "€")

                # calcola MTM
                net_mtm = (net_real if math.isfinite(net_real) else 0.0) + unreal

                # gross profit/loss: assumiamo che Gross Loss sia già negativa (somma delle perdite)
                gp_mtm = (gp_real if math.isfinite(gp_real) else 0.0) + (unreal if unreal > 0 else 0.0)
                gl_mtm = (gl_real if math.isfinite(gl_real) else 0.0) + (unreal if unreal < 0 else 0.0)

                # overwrite delle metriche principali (quelle che l'utente guarda)
                report_df = _upsert_metric(report_df, "Net Profit", float(net_mtm), "€")
                report_df = _upsert_metric(report_df, "Gross Profit", float(gp_mtm), "€")
                report_df = _upsert_metric(report_df, "Gross Loss", float(gl_mtm), "€")

                # extra (facoltativi ma utili per debug)
                report_df = _upsert_metric(report_df, "Open Pos Entry Price", float(unr.get("entry_price", float("nan"))), "€")
                report_df = _upsert_metric(report_df, "Open Pos Last Close", float(unr.get("last_close", float("nan"))), "€")

                print(
                    f"[INFO] MTM: posizione aperta {unr.get('side')} entry={unr.get('entry_price')} "
                    f"last_close={unr.get('last_close')} unrealized={unreal:+.2f}"
                )
        except Exception as e:
            print(f"[WARN] MTM/Unrealized non calcolabile: {e}")
        # ------------------------------------------------------------
        # Buy & Hold TRUE (Full Holding): primo IN → ultimo close (posizione sempre aperta)
        # ------------------------------------------------------------
        try:
            if "HOLD" in df.columns:
                hold = df["HOLD"].astype(str).fillna("")
                entries = df[hold == "IN"]

                if len(entries) > 0:
                    first_entry = entries.iloc[0]
                    entry_price = _to_float_eu(first_entry.get("VALUE", first_entry.get("close")))
                    last_close = _to_float_eu(df.iloc[-1].get("close"))

                    if math.isfinite(entry_price) and math.isfinite(last_close):
                        bh_full = last_close - entry_price

                        report_df = _upsert_metric(
                            report_df,
                            "Buy & Hold Profit (Full Holding)",
                            float(bh_full),
                            "€",
                        )

                        report_df = _upsert_metric(
                            report_df,
                            "Equity End Buy&Hold (Full Holding)",
                            float(last_close),
                            "€",
                        )
        except Exception:
            pass

        # ------------------------------------------------------------
        # Alpha Strategy vs Buy & Hold (Full Holding)
        # Alpha = Net Profit (Strategy) - Buy & Hold Profit (Full Holding)
        # ------------------------------------------------------------
        try:
            strat_net = _get_metric_raw(report_df, "Net Profit")
            hold_full = _get_metric_raw(report_df, "Buy & Hold Profit (Full Holding)")

            if math.isfinite(strat_net) and math.isfinite(hold_full):
                alpha = float(strat_net) - float(hold_full)

                report_df = _upsert_metric(
                    report_df,
                    "Alpha vs Buy&Hold (Full Holding)",
                    float(alpha),
                    "€",
                )
        except Exception:
            pass

        # ------------------------------------------------------------
        # Ordering: inserisci "Equity End Buy&Hold" subito dopo "Buy & Hold Profit (FILO)"
        # ------------------------------------------------------------
        try:
            if "Indicatore" in report_df.columns:
                inds = report_df["Indicatore"].astype(str).tolist()
                if "Equity End Buy&Hold" in inds and "Buy & Hold Profit (FILO)" in inds:
                    row_bh_end = report_df[report_df["Indicatore"].eq("Equity End Buy&Hold")]
                    report_df = report_df[~report_df["Indicatore"].eq("Equity End Buy&Hold")].reset_index(drop=True)
                    pos = int(report_df.index[report_df["Indicatore"].eq("Buy & Hold Profit (FILO)")][0]) + 1
                    report_df = pd.concat([report_df.iloc[:pos], row_bh_end, report_df.iloc[pos:]], ignore_index=True)
        except Exception as e:
            print(f"[WARN] Impossibile riordinare Equity End Buy&Hold: {e}")



        # -------------------------
        # Fix 1: evita duplicazioni di indicatori (es. Max Drawdown Additive)
        # -------------------------
        try:
            if "Indicatore" in report_df.columns:
                # specifico: Max Drawdown (Additive) duplicato
                mask_dd = report_df["Indicatore"].eq("Max Drawdown (Additive)")
                if mask_dd.sum() > 1:
                    report_df = pd.concat([report_df.loc[~mask_dd], report_df.loc[mask_dd].head(1)], ignore_index=True)
        except Exception as e:
            print(f"[WARN] Impossibile de-duplicare indicatori: {e}")

        # -------------------------
        # Fix 2: Avg Operations per Week (IN+OUT)
        # -------------------------
        try:
            # calcolo su debug_df (se presente) altrimenti sul SIGNAL df
            base_df = debug_df if "debug_df" in locals() else df
            avg_ops_week = compute_avg_operations_per_week(base_df, datetime_col="datetime", hold_col="HOLD")

            # fallback: if missing datetime/hold, compute average ops/week from date range and total ops
            if not math.isfinite(avg_ops_week):
                try:
                    # total operations
                    total_ops = None
                    if "Indicatore" in report_df.columns and "Valore_raw" in report_df.columns:
                        m_ops = report_df["Indicatore"].eq("Number of Operations (IN+OUT)")
                        if m_ops.any():
                            total_ops = float(report_df.loc[m_ops, "Valore_raw"].iloc[0])
                    if total_ops is None:
                        total_ops = float(num_ops) if "num_ops" in locals() and num_ops is not None else None

                    # date range from report (if present) or equity_df
                    dt_start = dt_end = None
                    if "Indicatore" in report_df.columns and "Valore" in report_df.columns:
                        m_s = report_df["Indicatore"].eq("Data Inizio")
                        m_e = report_df["Indicatore"].eq("Data Fine")
                        if m_s.any():
                            dt_start = pd.to_datetime(report_df.loc[m_s, "Valore"].iloc[0], dayfirst=True, errors="coerce")
                        if m_e.any():
                            dt_end = pd.to_datetime(report_df.loc[m_e, "Valore"].iloc[0], dayfirst=True, errors="coerce")

                    if (dt_start is None or pd.isna(dt_start) or dt_end is None or pd.isna(dt_end)) and "equity_df" in locals():
                        try:
                            if isinstance(equity_df.index, pd.DatetimeIndex) and len(equity_df.index) > 1:
                                dt_start = equity_df.index.min()
                                dt_end = equity_df.index.max()
                        except Exception:
                            pass

                    if total_ops is not None and dt_start is not None and dt_end is not None and pd.notna(dt_start) and pd.notna(dt_end):
                        weeks = max((dt_end - dt_start).total_seconds() / (7 * 86400.0), 1e-9)
                        avg_ops_week = float(total_ops) / weeks
                except Exception:
                    pass


            if "Indicatore" in report_df.columns and "Valore" in report_df.columns:
                row_name = "Avg Operations per Week (IN+OUT)"
                val_str = f"{avg_ops_week:.2f}".replace(".", ",") if math.isfinite(avg_ops_week) else "nan"

                mask = report_df["Indicatore"].eq(row_name)
                if mask.any():
                    report_df.loc[mask, "Valore"] = val_str
                    if "Valore_raw" in report_df.columns:
                        report_df.loc[mask, "Valore_raw"] = float(avg_ops_week)
                    if "Unità" in report_df.columns:
                        report_df.loc[mask, "Unità"] = "ops/week"
                else:
                    new_row = {"Indicatore": row_name, "Valore": val_str}
                    if "Valore_raw" in report_df.columns:
                        new_row["Valore_raw"] = float(avg_ops_week)
                    if "Unità" in report_df.columns:
                        new_row["Unità"] = "ops/week"
                    report_df = pd.concat([report_df, pd.DataFrame([new_row])], ignore_index=True)
        except Exception as e:
            print(f"[WARN] Impossibile calcolare Avg Operations per Week: {e}")


        def _dbg(label: str):
            cols = ["Indicatore", "Valore"]
            if "Valore_raw" in report_df.columns:
                cols.append("Valore_raw")

            sub = report_df.loc[
                report_df["Indicatore"].isin(["Equity Start", "Net Profit", "Equity End", "Equity End (Additive)"]),
                cols,
            ].copy()

            print(f"\n[DEBUG] {label}\n{sub.to_string(index=False)}\n")

        _dbg("DOPO apply_metrics (prima Valore_raw)")

        # -------------------------
        # Garantisce sempre Valore_raw numerico (single source of truth)
        # -------------------------
        def _to_float_eu(x) -> float:
            if x is None:
                return float("nan")
            if isinstance(x, (int, float, np.floating)):
                return float(x)
            s = str(x).strip()
            if s == "" or s.lower() == "nan":
                return float("nan")
            # EU -> float: "1.234,56" -> "1234.56"
            s = s.replace(".", "").replace(",", ".")
            try:
                return float(s)
            except Exception:
                return float("nan")

        if "Valore_raw" not in report_df.columns:
            report_df["Valore_raw"] = report_df["Valore"].apply(_to_float_eu)
        else:
            # prima tenta conversione diretta
            vr = pd.to_numeric(report_df["Valore_raw"], errors="coerce")
            # dove fallisce, ripiega sul parsing EU della colonna Valore
            if "Valore" in report_df.columns:
                fallback = report_df["Valore"].apply(_to_float_eu)
                vr = vr.where(vr.notna(), fallback)
            report_df["Valore_raw"] = vr

        print(
            report_df.loc[
                report_df["Indicatore"].isin(
                    [
                        "Gross Profit (-Outlier)",
                        "Gross Loss (-Outlier)",
                        "Net Profit (-Outlier)",
                        "Outliers Removed (count)",
                        "Outliers Removed (%)",
                    ]
                ),
                ["Indicatore", "Valore", "Valore_raw", "Unità"],
            ].to_string(index=False)
        )
        # =========================
        # MTM (Mark-to-Market): include PnL non realizzato su posizione aperta (HOLD=IN a fine file)
        # - aggiunge righe KPI: Unrealized PnL (Open), Net Profit/Gross Profit/Gross Loss (Realized e MTM)
        # - sovrascrive Net Profit / Gross Profit / Gross Loss con versione finale (coerente con la data di fine file)
        # =========================
        try:
            if "HOLD" in df.columns:
                hold = df["HOLD"].astype(str).fillna("")
                if len(hold) > 0 and hold.iloc[-1] == "IN":
                    prev = hold.shift(1).fillna("")
                    entries = df[(hold == "IN") & (prev != "IN")]
                    if len(entries) == 0:
                        entries = df[hold == "IN"]

                    if len(entries) > 0:
                        entry_row = entries.iloc[-1]
                        side = str(entry_row.get("SIGNAL", "")).strip().upper()

                        if side in ("LONG", "SHORT"):
                            entry_price = _to_float_eu(entry_row.get("VALUE", entry_row.get("close")))
                            last_close = _to_float_eu(df.iloc[-1].get("close"))

                            if math.isfinite(entry_price) and math.isfinite(last_close):
                                unreal = (last_close - entry_price) if side == "LONG" else (entry_price - last_close)

                                # snapshot KPI realizzati (pre-MTM)
                                net_real = _get_metric_raw(report_df, "Net Profit")
                                gp_real = _get_metric_raw(report_df, "Gross Profit")
                                gl_real = _get_metric_raw(report_df, "Gross Loss")

                                if math.isfinite(net_real):
                                    report_df = _upsert_metric(report_df, "Net Profit (Realized)", float(net_real), "€")
                                if math.isfinite(gp_real):
                                    report_df = _upsert_metric(report_df, "Gross Profit (Realized)", float(gp_real), "€")
                                if math.isfinite(gl_real):
                                    report_df = _upsert_metric(report_df, "Gross Loss (Realized)", float(gl_real), "€")

                                # dettagli posizione aperta
                                report_df = _upsert_metric(report_df, "Open Pos Entry Price", float(entry_price), "€")
                                report_df = _upsert_metric(report_df, "Open Pos Last Close", float(last_close), "€")
                                report_df = _upsert_metric(report_df, "Unrealized PnL (Open)", float(unreal), "€")

                                # valori finali (Realized + Unrealized)
                                base_net = float(net_real) if math.isfinite(net_real) else 0.0
                                net_final = base_net + float(unreal)
                                report_df = _upsert_metric(report_df, "Net Profit (Final)", float(net_final), "€")
                                report_df = _upsert_metric(report_df, "Net Profit", float(net_final), "€")

                                # Gross Loss nel report è tipicamente NEGATIVA (somma delle loss)
                                if math.isfinite(gp_real):
                                    gp_final = float(gp_real) + max(float(unreal), 0.0)
                                    report_df = _upsert_metric(report_df, "Gross Profit (Final)", float(gp_final), "€")
                                    report_df = _upsert_metric(report_df, "Gross Profit", float(gp_final), "€")
                                if math.isfinite(gl_real):
                                    gl_final = float(gl_real) + min(float(unreal), 0.0)
                                    report_df = _upsert_metric(report_df, "Gross Loss (Final)", float(gl_final), "€")
                                    report_df = _upsert_metric(report_df, "Gross Loss", float(gl_final), "€")
        except Exception:
            # best-effort: non bloccare il report
            pass

        # =========================
        # FIX: Equity End ASSOLUTO = Equity Start + Net Profit
        # =========================
        equity_start_raw = _get_metric_raw(report_df, "Equity Start")
        net_profit_raw = _get_metric_raw(report_df, "Net Profit")

        equity_end_raw = float("nan")
        if math.isfinite(equity_start_raw) and math.isfinite(net_profit_raw):
            equity_end_raw = equity_start_raw + net_profit_raw  # 100 + (-166.257) = -66.257

        mask_end = report_df["Indicatore"].eq("Equity End")
        if mask_end.any():
            report_df.loc[mask_end, "Valore_raw"] = equity_end_raw
        else:
            new_row = {"Indicatore": "Equity End", "Valore_raw": equity_end_raw, "Valore": ""}
            if "Unita" in report_df.columns:
                new_row["Unita"] = "€"
            report_df = pd.concat([report_df, pd.DataFrame([new_row])], ignore_index=True)

        print("[DEBUG] Equity End OVERRIDDEN =", equity_end_raw)

        print("\n[DEBUG RAW INPUT FOR EQUITY END]")
        print("Equity Start raw =", _get_metric_raw(report_df, "Equity Start"))
        print("Net Profit raw   =", _get_metric_raw(report_df, "Net Profit"))
        print("Equity End raw   =", _get_metric_raw(report_df, "Equity End"))
        print("Equity End (Add) =", _get_metric_raw(report_df, "Equity End (Additive)"))

        # =========================
        # Derivata VALIDATA: Expectancy (per Trade) = Net Profit / Number of Round-Trip Trades
        # Usa SOLO valori RAW (non stringhe formattate)
        # =========================
        net_profit_raw = _get_metric_raw(report_df, "Net Profit")
        n_trades_raw = _get_metric_raw(report_df, "Number of Round-Trip Trades")

        if math.isfinite(net_profit_raw) and math.isfinite(n_trades_raw) and n_trades_raw != 0:
            expectancy_raw = net_profit_raw / n_trades_raw
        else:
            expectancy_raw = float("nan")

        # IMPORTANTISSIMO: upsert aggiorna Valore_raw + Valore + Unità in modo coerente
        report_df = _upsert_metric(report_df, "Expectancy (per Trade)", float(expectancy_raw), "€")

        # =========================
        # Derivata TIME: Avg Operations per Week (IN+OUT)
        # - calcolata da HOLD (cambi di posizione) sul dataframe bars
        # =========================
        avg_ops_week_raw = float("nan")
        try:
            # datetime source
            if "DATETIME" in df.columns:
                _dt = pd.to_datetime(df["DATETIME"], errors="coerce")
            elif isinstance(df.index, pd.DatetimeIndex):
                _dt = df.index
            else:
                _dt = None

            if _dt is not None and "HOLD" in df.columns:
                hold = pd.to_numeric(df["HOLD"], errors="coerce").fillna(0.0)
                ops = hold.diff().fillna(0.0).ne(0.0).astype(int)  # 1 per ogni cambio stato (IN o OUT)
                wk = pd.Series(ops.values, index=_dt).groupby(_dt.to_period("W")).sum()
                if len(wk) > 0:
                    avg_ops_week_raw = float(wk.mean())
        except Exception as _e:
            print("[WARN] Avg Operations per Week non calcolabile:", _e)





# -------------------------
        # Debug metriche (DOPO derivata)
        # -------------------------
        def _pick(name: str):
            r = report_df.loc[report_df["Indicatore"] == name, ["Indicatore", "Valore_raw", "Valore"]]
            print("\n[DEBUG METRIC]", name)
            print(r.to_string(index=False) if not r.empty else "MISSING")

        _pick("Net Profit")
        _pick("Number of Round-Trip Trades")
        _pick("Expectancy (per Trade)")

        # =========================
        # Formattazione europea (virgola) SOLO per display (UNA SOLA VOLTA)
        # =========================
        def fmt_eu(x, nd: int = 6) -> str:
            if x is None:
                return ""
            try:
                v = float(x)
            except Exception:
                return str(x)
            if math.isnan(v) or math.isinf(v):
                return ""
            s = f"{v:,.{nd}f}"  # es: 1,234.568 (US)
            return s.replace(",", "X").replace(".", ",").replace("X", ".")  # -> 1.234,568 (EU)

        # -------------------------
        # LAST FIX (in-main): Avg Operations per Week (IN+OUT)
        # - usa Valore se Valore_raw mancante
        # - usa date/time/datetime se presenti
        # - dayfirst=True
        # -------------------------
        try:
            def _norm(s: object) -> str:
                return " ".join(str(s).strip().split()).lower()
        
            def _parse_float_any(x) -> float:
                try:
                    if x is None:
                        return float("nan")
                    if isinstance(x, (int, float)):
                        return float(x)
                    s = str(x).strip()
                    if s == "" or s.lower() == "nan":
                        return float("nan")
                    # EU -> float
                    s = s.replace(".", "").replace(",", ".")
                    return float(s)
                except Exception:
                    return float("nan")
        
            if "Indicatore" in report_df.columns:
                ind_norm = report_df["Indicatore"].astype(str).map(_norm)
        
                # total ops: Valore_raw -> Valore
                name_ops = "Number of Operations (IN+OUT)"
                m_ops = ind_norm.eq(_norm(name_ops))
                total_ops = float("nan")
                if m_ops.any():
                    if "Valore_raw" in report_df.columns:
                        total_ops = _parse_float_any(report_df.loc[m_ops, "Valore_raw"].iloc[0])
                    if not math.isfinite(total_ops) and "Valore" in report_df.columns:
                        total_ops = _parse_float_any(report_df.loc[m_ops, "Valore"].iloc[0])
        
                    # se Valore_raw mancante, popolalo (evita Valore che si svuota in formattazione)
                    if "Valore_raw" in report_df.columns and math.isfinite(total_ops):
                        report_df.loc[m_ops, "Valore_raw"] = float(total_ops)
        
                # date range: preferisci Data Inizio/Fine dal report
                dt_start = dt_end = None
                if "Valore" in report_df.columns:
                    m_s = ind_norm.eq(_norm("Data Inizio"))
                    m_e = ind_norm.eq(_norm("Data Fine"))
                    if m_s.any():
                        dt_start = pd.to_datetime(report_df.loc[m_s, "Valore"].iloc[0], errors="coerce", dayfirst=True)
                    if m_e.any():
                        dt_end = pd.to_datetime(report_df.loc[m_e, "Valore"].iloc[0], errors="coerce", dayfirst=True)
        
                # fallback: cerca un dataframe con datetime/date+time in locals()
                if (dt_start is None or pd.isna(dt_start) or dt_end is None or pd.isna(dt_end)):
                    candidates = []
                    for _nm in ("debug_df", "df", "trades_df", "equity_df"):
                        if _nm in locals():
                            _obj = locals().get(_nm)
                            if isinstance(_obj, pd.DataFrame) and len(_obj) > 1:
                                candidates.append(_obj)
        
                    def _extract_dt(_d: pd.DataFrame):
                        # 1) datetime col
                        for c in ("datetime", "Datetime", "DATETIME", "dateTime", "DateTime"):
                            if c in _d.columns:
                                dt = pd.to_datetime(_d[c], errors="coerce", dayfirst=True)
                                dt = dt.dropna()
                                if len(dt) > 1:
                                    return dt.min(), dt.max()
                        # 2) date + time
                        if "date" in _d.columns and "time" in _d.columns:
                            dt = pd.to_datetime(_d["date"].astype(str).str.strip() + " " + _d["time"].astype(str).str.strip(),
                                               errors="coerce", dayfirst=True)
                            dt = dt.dropna()
                            if len(dt) > 1:
                                return dt.min(), dt.max()
                        # 3) index
                        if isinstance(_d.index, pd.DatetimeIndex) and len(_d.index) > 1:
                            return _d.index.min(), _d.index.max()
                        return None, None
        
                    for _d in candidates:
                        a,b = _extract_dt(_d)
                        if a is not None and b is not None and pd.notna(a) and pd.notna(b):
                            dt_start, dt_end = a,b
                            break
        
                avg_ops_week = float("nan")
                if math.isfinite(total_ops) and dt_start is not None and dt_end is not None and pd.notna(dt_start) and pd.notna(dt_end):
                    weeks = max((dt_end - dt_start).total_seconds() / (7 * 86400.0), 1e-9)
                    avg_ops_week = float(total_ops) / weeks
        
                # upsert SOLO se finito
                if math.isfinite(avg_ops_week):
                    report_df = _upsert_metric(report_df, "Avg Operations per Week (IN+OUT)", avg_ops_week, "ops/week")
                    # forziamo sempre N sulla verifica
                    if "Verificata" in report_df.columns:
                        m_avg = report_df["Indicatore"].astype(str).str.strip().eq("Avg Operations per Week (IN+OUT)")
                        report_df.loc[m_avg, "Verificata"] = "N"
        
        except Exception as e:
            print(f"[WARN] LAST FIX Avg Ops/Week non applicato: {e}")
        # Formattazione Valore: NON sovrascrivere con vuoto se Valore_raw è NaN
        if "Valore_raw" in report_df.columns and "Valore" in report_df.columns:
            _raw_num = pd.to_numeric(report_df["Valore_raw"], errors="coerce")
            _fmt = _raw_num.apply(lambda v: fmt_eu(v, nd=6))
            report_df["Valore"] = report_df["Valore"].where(_raw_num.isna(), _fmt)

        # =========================
        # Colonna "Verificata" (Y/N) – SOLO fase di stampa
        # =========================
        verified_metrics = {
            # ---- Performance / Equity ----
            "Equity Start",
            "Equity End (Additive)",
            "Equity End",
            "Total Return (Additive)",
            "Total Return",
            "Max Drawdown (Additive)",
            "AVG Win",
            "AVG Loss",
            "Expectancy (per Trade)",
            "Time IN Position",
            "Volatility per Trade",
            "Max Drawdown (Additive)",
            # ---- Buy & Hold ----
            "Buy & Hold Return (First IN -> Last Close)",
            "Buy & Hold Profit (First IN -> Close after Last OUT)",
            "Buy & Hold Profit (FILO)",
            "Strategy Outperformance",
            # ---- Conteggi operativi ----
            "Number of Round-Trip Trades",
            "Number of Operations (IN+OUT)",
            "Avg Operations per Week (IN+OUT)",
            "Entries (OUT->IN)",
            "Exits (IN->OUT)",
            "Win Rate (Round-Trip)",
            # ---- PnL ----
            "Gross Profit (Winning Trades Only)",
            "Gross Loss (Losing Trades Only)",
            "Net Profit",
            "Profit Factor",
        }

        # Verificata = Y solo se (indicatore in set) e Valore_raw è numerico finito
        if "Valore_raw" in report_df.columns:
            _raw_num_v = pd.to_numeric(report_df["Valore_raw"], errors="coerce")
        else:
            _raw_num_v = pd.Series([float("nan")] * len(report_df))
        
        report_df["Verificata"] = [
            "Y" if (str(ind) in verified_metrics and (isinstance(v, (int, float)) and math.isfinite(v))) else "N"
            for ind, v in zip(report_df["Indicatore"].astype(str), _raw_num_v)
        ]
        # Avg Ops/Week non è mai verificata
        m_avg_v = report_df["Indicatore"].astype(str).str.strip().eq("Avg Operations per Week (IN+OUT)")
        report_df.loc[m_avg_v, "Verificata"] = "N"

        # =========================
        # Check di coerenza KPI (DEBUG)
        # =========================
        gp = float(
            report_df.loc[
                report_df["Indicatore"] == "Gross Profit (Winning Trades Only)",
                "Valore_raw",
            ].iloc[0]
        )
        gl = float(
            report_df.loc[
                report_df["Indicatore"] == "Gross Loss (Losing Trades Only)",
                "Valore_raw",
            ].iloc[0]
        )
        npv = float(
            report_df.loc[
                report_df["Indicatore"] == "Net Profit",
                "Valore_raw",
            ].iloc[0]
        )

        print("[DEBUG] CHECK KPI: GP+GL=", gp + gl, " NetProfit=", npv, " Delta=", npv - (gp + gl))
        print("[DEBUG] CHECK KPI: GP+GL=", gp + gl, " NetProfit=", npv, " Delta=", npv - (gp + gl))

        # -------------------------
        # Upsert Transaction Costs (da curva additive)
        # -------------------------
        tx_costs = 0.0
        if "cum_costs" in equity_df.columns:
            s_cost = pd.to_numeric(equity_df["cum_costs"], errors="coerce").dropna()
            tx_costs = float(s_cost.iloc[-1]) if not s_cost.empty else 0.0
        else:
            n_trades = int(len(trades_df)) if isinstance(trades_df, pd.DataFrame) else 0
            tx_costs = float(cost_per_transaction) * float(transactions_per_trade) * float(n_trades)

        report_df = _upsert_metric(report_df, "Transaction Costs", float(tx_costs), "€")

        # -------------------------
        # Profit Factor = GP / |GL|
        # -------------------------
        gp = _get_metric_raw(report_df, "Gross Profit (Winning Trades Only)")
        gl = _get_metric_raw(report_df, "Gross Loss (Losing Trades Only)")
        if math.isfinite(gp) and math.isfinite(gl) and gl != 0:
            pf = gp / abs(gl)
            report_df = _upsert_metric(report_df, "Profit Factor", float(pf), "ratio")

        # -------------------------
        # Nota: POSIZIONE ANCORA APERTA
        # -------------------------
        def _has_no_out_after_first_in(edf: pd.DataFrame) -> bool:
            col = "HOLD" if "HOLD" in edf.columns else ("SIGNAL" if "SIGNAL" in edf.columns else None)
            if col is None:
                return False
            s = edf[col].astype(str).str.upper().str.strip()
            if not (s == "IN").any():
                return False
            first_in_pos = s.index[(s == "IN")][0]
            s_after = s.loc[first_in_pos:]
            return not (s_after == "OUT").any()

        if _has_no_out_after_first_in(equity_df):
            mask = report_df["Indicatore"] == "Buy & Hold Profit (First IN -> Last OUT)"
            if mask.any():
                report_df.loc[mask, "Indicatore"] = "Buy & Hold Profit (First IN -> Last OUT) — POSIZIONE ANCORA APERTA"

            mask2 = report_df["Indicatore"] == "Buy & Hold Return (First IN -> Last Close)"
            if mask2.any():
                report_df.loc[mask2, "Indicatore"] = "Buy & Hold Return (First IN -> Last Close) — POSIZIONE ANCORA APERTA"

        # -------------------------
        # Check contabile: Equity End = Equity Start + Net Profit - Costs
        # -------------------------
        eq_start = float(equity_start)
        eq_end = float("nan")
        if "equity_additive" in equity_df.columns:
            s_eq = pd.to_numeric(equity_df["equity_additive"], errors="coerce").dropna()
            if not s_eq.empty:
                eq_end = float(s_eq.iloc[-1])

        net_profit = _get_metric_raw(report_df, "Net Profit")
        if math.isfinite(eq_end) and math.isfinite(net_profit):
            rhs = eq_start + net_profit - float(tx_costs)
            print(
                "[DEBUG] CHECK EQUITY:",
                "EquityStart=", eq_start,
                "EquityEnd=", eq_end,
                "NetProfit=", net_profit,
                "TxCosts=", float(tx_costs),
                "RHS=", rhs,
                "Delta=", (eq_end - rhs),
            )

        # =========================
        # Rimozione colonna Valore_raw (solo output finale)
        # =========================
        if "Valore_raw" in report_df.columns:
            report_df = report_df.drop(columns=["Valore_raw"])

        # =========================
        # Ordinamento report (ordine richiesto)
        # =========================
        REPORT_ORDER = [
            "Equity Start",
            "Equity End",
            "Gross Profit (Winning Trades Only)",
            "Gross Loss (Losing Trades Only)",
            "Net Profit",
            "Buy & Hold Profit (FILO)",
            "Strategy Outperformance",
            "Total Return",
            "Number of Operations (IN+OUT)",
            "Avg Operations per Week (IN+OUT)",
            "Transaction Costs",
            "Win Rate (Round-Trip)",
            "AVG Win",
            "AVG Loss",
            "Expectancy (per Trade)",            "Gross Profit (-Outlier)",
            "Gross Loss (-Outlier)",
            "Net Profit (-Outlier)",
            "Outliers Removed (count)",
            "Outliers Removed (%)",
            "Number of Round-Trip Trades",
            "Entries (OUT->IN)",
            "Exits (IN->OUT)",
            "Profit Factor",
            "Volatility per Trade",
            "Time IN Position",
            "Equity End (Additive)",
            "Total Return (Additive)",
            "Max Drawdown (Additive)",
        ]

        order_map = {name: i for i, name in enumerate(REPORT_ORDER)}

        # indice di sort: se indicatore non presente in REPORT_ORDER -> va in fondo (stabile)
        # report_df["_sort_key"] = report_df["Indicatore"].map(order_map).fillna(len(REPORT_ORDER)).astype(int)

        # tie-breaker per stabilita': mantiene l'ordine originale tra gli "extra"
        # report_df["_sort_seq"] = range(len(report_df))

        # report_df = report_df.sort_values(by=["_sort_key", "_sort_seq"], ascending=True).drop columns=["_sort_key", "_sort_seq"]


        # -------------------------

        # -------------------------
        # Time summary (best/worst buckets) -> aggiunti nel report principale
        # -------------------------
        try:
            if "time_summary" in locals() and isinstance(time_summary, dict) and time_summary:
                for k, v in time_summary.items():
                    # evita duplicati
                    if not report_df["Indicatore"].eq(k).any():
                        report_df = pd.concat(
                            [
                                report_df,
                                pd.DataFrame(
                                    [{"Indicatore": k, "Valore": str(v), "Valore_raw": float("nan"), "Unità": ""}]
                                ),
                            ],
                            ignore_index=True,
                        )
        except Exception as e:
            print(f"[WARN] Impossibile aggiungere time summary al report: {e}")

        # -------------------------
        # Prepend Data Inizio / Data Fine / Timeframe al CSV
        # -------------------------
        try:
            header_rows = pd.DataFrame(
                [
                    {"Indicatore": "Data Inizio", "Valore": data_inizio, "Valore_raw": float("nan"), "Unità": ""},
                    {"Indicatore": "Data Fine", "Valore": data_fine, "Valore_raw": float("nan"), "Unità": ""},
                    {"Indicatore": "Timeframe", "Valore": timeframe, "Valore_raw": float("nan"), "Unità": ""},
                ]
            )

            # evita duplicazioni accidentali
            if not report_df["Indicatore"].isin(["Data Inizio", "Data Fine", "Timeframe"]).any():
                report_df = pd.concat([header_rows, report_df], ignore_index=True)
        except Exception as e:
            print(f"[WARN] Impossibile inserire Data Inizio/Fine/Timeframe nel report: {e}")

        SEP = "========================================================================="

        # helper: valore da report_df (tabella metriche già calcolata)
        def m(name: str) -> str:
            try:
                s = report_df.loc[report_df["Indicatore"] == name, "Valore"]
                return "" if s.empty else str(s.iloc[0])
            except Exception:
                return ""

        # helper: unità da report_df legacy
        def u(name: str) -> str:
            try:
                if "Unità" not in report_df.columns:
                    return ""
                s = report_df.loc[report_df["Indicatore"] == name, "Unità"]
                return "" if s.empty else str(s.iloc[0])
            except Exception:
                return ""

        # float EU robusto (35,855 -> 35.855)
        def _to_float_eu_local(x) -> float:
            """
            Parsing robusto input:
            - "35,855"   -> 35.855
            - "69.99"    -> 69.99
            - "1.234,56" -> 1234.56
            - "1,234.56" -> 1234.56
            Output del report resta SEMPRE EU (virgola).
            """
            try:
                s = str(x).strip()
                if s == "" or s.lower() == "nan":
                    return float("nan")

                s = s.replace("€", "").replace("EUR", "").replace(" ", "").strip()

                has_comma = "," in s
                has_dot = "." in s

                if has_comma and has_dot:
                    # separatore decimale = ultimo che compare
                    if s.rfind(",") > s.rfind("."):
                        # EU: 1.234,56
                        s = s.replace(".", "").replace(",", ".")
                    else:
                        # US: 1,234.56
                        s = s.replace(",", "")
                    return float(s)

                if has_comma and not has_dot:
                    # EU: 123,45
                    return float(s.replace(",", "."))

                # US: 123.45 oppure intero
                return float(s.replace(",", ""))
            except Exception:
                return float("nan")

        # formattazione EU (float -> "12,345678")
        def _fmt_eu(x: float, nd: int = 6) -> str:
            try:
                if x != x:  # NaN
                    return ""
                return f"{x:.{nd}f}".replace(".", ",")
            except Exception:
                return ""

        def _compute_cash_injections_and_capital(
                trade_rows: list[dict],
                equity_start: float,
        ) -> tuple[float, float]:
            """
            Calcola:
              - Cash Injections: capitale aggiuntivo necessario per aprire trade quando l'equity disponibile non basta
              - Max Capital Employed: massimo entry_value richiesto da un trade

            Assunzioni (coerenti con la suite attuale):
              - operatività "single position" (no posizioni sovrapposte)
              - qty costante o presente per trade; default=1
              - entry_value = entry_price * qty
              - equity cresce con P/L realizzato; quando un entry_value supera l'equity disponibile -> injection
            """
            eq = float(equity_start)
            cash_in = 0.0
            max_cap = 0.0

            for tr in trade_rows:
                # trade dict expected keys: entry_price, exit_price (optional), side ("LONG"/"SHORT"), qty (optional)
                entry = float(tr.get("entry_price", 0.0))
                qty = float(tr.get("qty", 1.0) or 1.0)
                side = (tr.get("side") or "").upper().strip()

                entry_value = entry * qty
                if entry_value > max_cap:
                    max_cap = entry_value

                # Injection se non basta l'equity per "comprare" la posizione
                if entry_value > eq:
                    cash_in += (entry_value - eq)
                    eq = entry_value

                # Se trade chiuso, aggiorna equity con P/L realizzato
                if tr.get("exit_price") is not None:
                    exitp = float(tr.get("exit_price", 0.0))
                    if side in ("LONG", "BUY"):
                        pl = (exitp - entry) * qty
                    elif side in ("SHORT", "SELL"):
                        pl = (entry - exitp) * qty
                    else:
                        # fallback: considera LONG se non specificato
                        pl = (exitp - entry) * qty

                    eq += pl

            return cash_in, max_cap

        # ------------------------------------------------------------
        # intestazione: Symbol / Strategia
        # ------------------------------------------------------------
        symbol_val = ""
        if "symbol" in df.columns and len(df) > 0:
            symbol_val = str(df["symbol"].iloc[0])
        elif "Symbol" in df.columns and len(df) > 0:
            symbol_val = str(df["Symbol"].iloc[0])

        strategy_val = str(src.stem)

        # ------------------------------------------------------------
        # 1) Unrealized Profit/Loss (solo se ultima riga è IN)
        #    regola: last_close - entry_close (LONG) / entry_close - last_close (SHORT)
        # ------------------------------------------------------------
        unreal_pl = float("nan")
        try:
            if "HOLD" in df.columns and len(df) > 0:
                hold = df["HOLD"].astype(str).fillna("")
                if str(hold.iloc[-1]).upper() == "IN":
                    prev = hold.shift(1).fillna("")
                    # entry dell'open position = ultimo punto in cui si passa a IN
                    entries = df[(hold == "IN") & (prev != "IN")]
                    if len(entries) == 0:
                        entries = df[hold == "IN"]
                    if len(entries) > 0:
                        open_entry = entries.iloc[-1]
                        side = str(open_entry.get("SIGNAL", "")).strip().upper()
                        if side not in ("LONG", "SHORT"):
                            side = "LONG"

                        entry_close = _to_float_eu_local(open_entry.get("close"))
                        last_close = _to_float_eu_local(df.iloc[-1].get("close"))

                        if entry_close == entry_close and last_close == last_close:
                            unreal_pl = (last_close - entry_close) if side == "LONG" else (entry_close - last_close)
        except Exception:
            pass

        unreal_pl_str = _fmt_eu(unreal_pl, 6) if unreal_pl == unreal_pl else ""
        unreal_unit = "€" if unreal_pl_str != "" else ""

        # ------------------------------------------------------------
        # 2) Equity End Alpha = Equity Start + (Gross Profit + Gross Loss + Unrealized)
        #    Net Profit Alpha (coerente) = Gross Profit + Gross Loss + Unrealized
        # ------------------------------------------------------------
        equity_start = _to_float_eu_local(m("Equity Start"))
        gp = _to_float_eu_local(m("Gross Profit (Winning Trades Only)"))
        gl = _to_float_eu_local(m("Gross Loss (Losing Trades Only)"))

        # =========================
        # Cash Injections + Max Capital Employed + Equity End coerente
        # =========================

        # 2) Costruzione trade_rows: SEMPRE da trades_df (robusto, no dipendenza da variabili locali)
        trade_rows = []
        cash_injections = 0.0
        max_capital_employed = 0.0

        try:
            if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
                for _, tr in trades_df.iterrows():
                    side = str(tr.get("side", "LONG")).upper().strip()

                    entry_price = _to_float_eu_local(tr.get("entry_price"))
                    exit_price_raw = tr.get("exit_price", None)
                    exit_price = None if exit_price_raw is None else _to_float_eu_local(exit_price_raw)

                    qty = _to_float_eu_local(tr.get("qty") or tr.get("quantity") or tr.get("size") or 1.0)
                    # NaN-safe + default 1
                    qty = float(qty) if (qty is not None and qty == qty) else 1.0

                    # entry_price NaN-safe
                    if entry_price is None or entry_price != entry_price:
                        continue

                    trade_rows.append(
                        {
                            "side": side,
                            "entry_price": float(entry_price),
                            "exit_price": None if (exit_price is None or exit_price != exit_price) else float(
                                exit_price),
                            "qty": float(qty),
                        }
                    )

            if trade_rows:
                cash_injections, max_capital_employed = _compute_cash_injections_and_capital(
                    trade_rows=trade_rows,
                    equity_start=equity_start,
                )
        except Exception as e:
            print(f"[WARN] Impossibile calcolare Cash Injections / Max Capital Employed: {e}")

            # 2) Costruzione trade_rows: SEMPRE da trades_df (robusto, no dipendenza da variabili locali)
            trade_rows = []
            cash_injections = 0.0
            max_capital_employed = 0.0

            try:
                if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
                    for _, tr in trades_df.iterrows():
                        side = str(tr.get("side", "LONG")).upper().strip()

                        entry_price = _to_float_eu_local(tr.get("entry_price"))
                        exit_price_raw = tr.get("exit_price", None)
                        exit_price = None if exit_price_raw is None else _to_float_eu_local(exit_price_raw)

                        qty = _to_float_eu_local(tr.get("qty") or tr.get("quantity") or tr.get("size") or 1.0)
                        qty = float(qty) if (qty == qty and qty is not None) else 1.0  # NaN-safe

                        if entry_price == entry_price:  # not NaN
                            trade_rows.append(
                                {
                                    "side": side,
                                    "entry_price": float(entry_price),
                                    "exit_price": None if (exit_price is None or exit_price != exit_price) else float(
                                        exit_price),
                                    "qty": float(qty),
                                }
                            )

                if trade_rows:
                    cash_injections, max_capital_employed = _compute_cash_injections_and_capital(
                        trade_rows=trade_rows,
                        equity_start=equity_start,
                    )
            except Exception as e:
                print(f"[WARN] Impossibile calcolare Cash Injections / Max Capital Employed: {e}")

        # -------------------------
        # Cash injections / Max capital employed (già calcolati sopra se trade_rows esiste)
        # Se trade_rows vuoto => restano 0.0
        # -------------------------
        cash_injections = cash_injections if cash_injections == cash_injections else 0.0
        max_capital_employed = max_capital_employed if max_capital_employed == max_capital_employed else 0.0

        # -------------------------
        # Unrealized + Market Value Open Position (se c'è una posizione aperta)
        # -------------------------
        unreal_pl = 0.0
        market_value_open = 0.0

        try:
            if "open_entry" in locals() and isinstance(open_entry, pd.Series):
                qty_open = _to_float_eu_local(open_entry.get("qty") or open_entry.get("quantity") or 1.0)
                qty_open = float(qty_open) if (qty_open is not None and qty_open == qty_open) else 1.0

                entry_price = _to_float_eu_local(
                    open_entry.get("close") or open_entry.get("entry_price") or open_entry.get("entry")
                )
                last_close = _to_float_eu_local(df.iloc[-1].get("close"))

                if entry_price == entry_price and last_close == last_close:
                    side_open = str(
                        open_entry.get("side") or open_entry.get("dir") or open_entry.get("signal") or "").upper()
                    if side_open in ("SHORT", "SELL"):
                        unreal_pl = (entry_price - last_close) * qty_open
                    else:
                        unreal_pl = (last_close - entry_price) * qty_open

                    market_value_open = last_close * qty_open
        except Exception as e:
            print(f"[WARN] Unrealized P/L non calcolabile, uso 0.0: {e}")
            unreal_pl = 0.0
            market_value_open = 0.0

        # -------------------------
        # Realized P/L: default robusto + fallback su Gross Profit/Loss
        # -------------------------
        net_profit_realized = 0.0

        # se hai già un realized calcolato altrove (es. da trades_df), mantienilo; altrimenti fallback
        try:
            # gp/gl possono essere già float o NaN
            if gp != gp:
                gp = _to_float_eu_local(m("Gross Profit"))
            if gl != gl:
                gl = _to_float_eu_local(m("Gross Loss"))

            if gp == gp and gl == gl:
                net_profit_realized = float(gp + gl)
        except Exception as e:
            print(f"[WARN] Realized P/L non calcolabile da Gross Profit/Loss, uso 0.0: {e}")
            net_profit_realized = 0.0

        # -------------------------
        # Total P/L (Real + Unreal)
        # -------------------------
        total_pl = net_profit_realized + unreal_pl

        # Equity End: fotografia patrimoniale finale (include cash injections se necessari)
        equity_end = equity_start + total_pl + cash_injections

        # Cash End: utile solo informativamente
        cash_end = equity_end - market_value_open if market_value_open else equity_end

        # -------------------------
        # ROCE (Return on Capital Employed) e Total Return Alpha
        # -------------------------
        roce = (total_pl / max_capital_employed) if (max_capital_employed and max_capital_employed > 0) else float(
            "nan")

        # Net Profit Alpha (economico) = Total P/L (Real + Unreal) [NON include cash flow]
        net_profit_alpha_val = total_pl if total_pl == total_pl else float("nan")
        net_profit_alpha_str = _fmt_eu(net_profit_alpha_val, 6) if net_profit_alpha_val == net_profit_alpha_val else ""
        net_profit_alpha_unit = "€" if net_profit_alpha_str != "" else ""

        # Equity End Alpha (patrimoniale) = equity_end (include cash injections)
        equity_end_alpha_val = equity_end if equity_end == equity_end else float("nan")
        equity_end_alpha_str = _fmt_eu(equity_end_alpha_val, 6) if equity_end_alpha_val == equity_end_alpha_val else ""
        unit_equity_end_alpha = "€" if equity_end_alpha_str != "" else ""

        # Total Return Alpha (coerente): ROCE in %
        total_return_alpha_val = (roce * 100.0) if roce == roce else float("nan")
        total_return_alpha_str = _fmt_eu(total_return_alpha_val,
                                         6) if total_return_alpha_val == total_return_alpha_val else ""
        total_return_alpha = total_return_alpha_str
        unit_total_return_alpha = "%" if total_return_alpha_str != "" else ""

        # ------------------------------------------------------------
        # 3) Buy&Hold (Full Holding) = primo IN -> ultimo close (sempre LONG)
        # ------------------------------------------------------------
        bh_full = float("nan")
        try:
            if "HOLD" in df.columns and len(df) > 0:
                hold = df["HOLD"].astype(str).fillna("")
                prev = hold.shift(1).fillna("")
                entries = df[(hold == "IN") & (prev != "IN")]
                if len(entries) == 0:
                    entries = df[hold == "IN"]
                if len(entries) > 0:
                    first_entry = entries.iloc[0]
                    entry_close = _to_float_eu_local(first_entry.get("close"))
                    last_close = _to_float_eu_local(df.iloc[-1].get("close"))
                    if entry_close == entry_close and last_close == last_close:
                        bh_full = last_close - entry_close  # Full Holding LONG
        except Exception:
            pass

        bh_full_str = _fmt_eu(bh_full, 6) if bh_full == bh_full else ""
        bh_full_unit = "€" if bh_full_str != "" else ""

        # ------------------------------------------------------------
        # 4) Strategy Alpha Outperformance (FH) vs Full Holding
        #    (Net Profit Alpha - BH Full) / |BH Full| * 100
        # ------------------------------------------------------------
        outperf_fh = float("nan")
        if net_profit_alpha_val == net_profit_alpha_val and bh_full == bh_full and bh_full != 0.0:
            outperf_fh = ((net_profit_alpha_val - bh_full) / abs(bh_full)) * 100.0

        outperf_fh_str = _fmt_eu(outperf_fh, 6) if outperf_fh == outperf_fh else ""
        outperf_fh_unit = "%" if outperf_fh_str != "" else ""

        # buy&hold FILO (rimane)
        bh_filo = m("Buy & Hold Profit (FILO)")
        unit_bh_filo = u("Buy & Hold Profit (FILO)") or "€"

        # Equity End Buy&Hold (come da legacy)
        equity_end_bh = m("Equity End Buy&Hold")
        unit_equity_end_bh = u("Equity End Buy&Hold") or "€"

        # Transaction Costs
        tx_costs = m("Transaction Costs")
        unit_tx_costs = u("Transaction Costs") or "€"

        FINAL_ORDER = [
            ("Symbol", symbol_val, ""),
            ("Strategia", strategy_val, ""),
            (SEP, None, None),

            ("Data Inizio", m("Data Inizio"), ""),
            ("Data Fine", m("Data Fine"), ""),
            ("Timeframe", m("Timeframe"), ""),
            (SEP, None, None),

            ("Equity Start", m("Equity Start"), u("Equity Start") or "EUR"),
            ("Cash Injections", _fmt_eu(cash_injections, 6) if cash_injections == cash_injections else "", "€" if (cash_injections == cash_injections and cash_injections != 0.0) else ""),
            ("Max Capital Employed", _fmt_eu(max_capital_employed, 6) if max_capital_employed == max_capital_employed and max_capital_employed > 0 else "", "€" if (max_capital_employed == max_capital_employed and max_capital_employed > 0) else ""),
            ("Total P/L (Real + Unreal)", _fmt_eu(total_pl, 6) if total_pl == total_pl else "", "€" if (total_pl == total_pl) else ""),
            ("ROCE", total_return_alpha, unit_total_return_alpha),
            ("Equity End Alpha", equity_end_alpha_str, unit_equity_end_alpha),
            ("Equity End Buy&Hold", equity_end_bh, unit_equity_end_bh),
            ("Transaction Costs", tx_costs, unit_tx_costs),
            (SEP, None, None),

            ("Total Return Alpha", total_return_alpha, unit_total_return_alpha),
            ("Strategy Alpha Outperformance (FH)", outperf_fh_str, outperf_fh_unit),
            (SEP, None, None),

            ("Gross Profit (Winning Trades Only)", m("Gross Profit (Winning Trades Only)"), u("Gross Profit (Winning Trades Only)") or "€"),
            ("Gross Loss (Losing Trades Only)", m("Gross Loss (Losing Trades Only)"), u("Gross Loss (Losing Trades Only)") or "€"),
            ("Net Profit Alpha", net_profit_alpha_str, net_profit_alpha_unit),
            ("Unrealized Profit/Loss", unreal_pl_str, unreal_unit),
            ("Buy & Hold Profit (FILO)", bh_filo, unit_bh_filo),
            ("Buy&Hold (Full Holding)", bh_full_str, bh_full_unit),
            ("Total Return", m("Total Return"), u("Total Return")),
            ("Max Drawdown (Additive)", m("Max Drawdown (Additive)"), u("Max Drawdown (Additive)") or "€"),
            (SEP, None, None),

            ("Gross Profit (-Outlier)", m("Gross Profit (-Outlier)"), u("Gross Profit (-Outlier)") or "€"),
            ("Gross Loss (-Outlier)", m("Gross Loss (-Outlier)"), u("Gross Loss (-Outlier)") or "€"),
            ("Net Profit (-Outlier)", m("Net Profit (-Outlier)"), u("Net Profit (-Outlier)") or "€"),
            ("Outliers Removed (count)", m("Outliers Removed (count)"), u("Outliers Removed (count)") or "count"),
            ("Outliers Removed (%)", m("Outliers Removed (%)"), u("Outliers Removed (%)") or "%"),
            ("Number of Operations (IN+OUT)", m("Number of Operations (IN+OUT)"), u("Number of Operations (IN+OUT)") or "count"),
            ("Avg Operations per Week (IN+OUT)", m("Avg Operations per Week (IN+OUT)"), u("Avg Operations per Week (IN+OUT)") or "ops/week"),
            ("Win Rate (Round-Trip)", m("Win Rate (Round-Trip)"), u("Win Rate (Round-Trip)") or "%"),
            ("AVG Win", m("AVG Win"), u("AVG Win") or "€"),
            ("AVG Loss", m("AVG Loss"), u("AVG Loss") or "€"),
            ("Expectancy (per Trade)", m("Expectancy (per Trade)"), u("Expectancy (per Trade)") or "€"),
            ("Number of Round-Trip Trades", m("Number of Round-Trip Trades"), u("Number of Round-Trip Trades") or "count"),
            ("Entries (OUT->IN)", m("Entries (OUT->IN)"), u("Entries (OUT->IN)") or "count"),
            ("Exits (IN->OUT)", m("Exits (IN->OUT)"), u("Exits (IN->OUT)") or "count"),
            ("Profit Factor", m("Profit Factor"), u("Profit Factor") or "ratio"),
            ("Volatility per Trade", m("Volatility per Trade"), u("Volatility per Trade") or "%"),
            (SEP, None, None),

            ("Time IN Position", m("Time IN Position"), u("Time IN Position") or "%"),
            ("Equity End (Additive)", m("Equity End (Additive)"), u("Equity End (Additive)") or "€"),
            ("Total Return (Additive)", m("Total Return (Additive)"), u("Total Return (Additive)") or "%"),
            ("Best Hour (Net Profit)", m("Best Hour (Net Profit)"), ""),
            ("Worst Hour (Net Profit)", m("Worst Hour (Net Profit)"), ""),
            ("Best Day (Net Profit)", m("Best Day (Net Profit)"), ""),
            ("Worst Day (Net Profit)", m("Worst Day (Net Profit)"), ""),
            ("Best Session (Net Profit)", m("Best Session (Net Profit)"), ""),
            ("Worst Session (Net Profit)", m("Worst Session (Net Profit)"), ""),
            (SEP, None, None),
        ]

        rows = []
        for label, value, unit in FINAL_ORDER:
            if label == SEP:
                rows.append({"Indicatore": SEP, "Valore": "", "Unità": ""})
            else:
                rows.append({
                    "Indicatore": label,
                    "Valore": "" if value is None else value,
                    "Unità": "" if unit is None else unit
                })

        report_df_final = pd.DataFrame(rows)

        # -------------------------
        # Export (AUTHORITATIVE)
        # -------------------------
        out_report = out_dir / f"REPORT_{src.stem}.csv"

        # stampa a video: usa SOLO il final
        _print_report_table(report_df_final, out_report, max_rows=120)

        # CSV: scrive SOLO il final (bypass totale di export legacy)
        report_df_final.to_csv(out_report, sep=";", index=False, encoding="utf-8-sig")

        # Sanity check: verifica che il CSV esista davvero (utile in pipeline)
        try:
            if not out_report.exists() or out_report.stat().st_size == 0:
                raise RuntimeError(f"Report CSV non creato o vuoto: {out_report}")
        except Exception as _e:
            print(f"[ERROR] Export CSV fallito: {_e}")
            raise
        print(f"[OK] Report export (final): {out_report}")


        # Debug dump completo SIGNAL + Trade_ID + Trade_profit (+ TIME features)
        # Nota: debug_df/time_stats_df sono pre-calcolati sopra (subito dopo il load del SIGNAL_)
        if "debug_df" not in locals():
            try:
                debug_df = add_trade_id_and_profit(df)
            except Exception as e:
                print(f"[WARN] Impossibile calcolare Trade_ID/Trade_profit (fallback debug): {e}")
                debug_df = df.copy()
                debug_df["Trade_ID"] = pd.Series([pd.NA] * len(debug_df), dtype="Int64")
                debug_df["Trade_profit"] = np.nan
            debug_df = add_time_features(debug_df, datetime_col="datetime")

        out_debug = out_dir / f"REPORT_DEBUG_{src.stem}.csv"
        export_debug_signal_csv(out_debug, debug_df)

        # Export statistiche time-based (BY_HOUR / BY_DAY / BY_SESSION)
        if "time_stats_df" in locals() and isinstance(time_stats_df, pd.DataFrame) and not time_stats_df.empty:
            out_time = out_dir / f"REPORT_TIME_{src.stem}.csv"
            time_stats_df.to_csv(out_time, sep=";", index=False, encoding="utf-8-sig")
            print(f"[OK] Time stats export: {out_time}")

        # Verifica export (opzionale)
        if args.verify_export:
            check = _read_report_csv_robust(out_report)
            cols = [c.strip().lstrip("\ufeff") for c in check.columns]
            if "Indicatore" not in cols:
                print(f"[WARN] Verifica export: colonna 'Indicatore' non trovata. Colonne lette: {list(check.columns)}")
            else:
                col_ind = next((c for c in check.columns if c.strip().lstrip("\ufeff") == "Indicatore"), None)
                print(
                    f"[INFO] Verifica export OK: righe={len(check)}; "
                    f"ultimi indicatori={check[col_ind].tail(3).tolist() if col_ind else 'N/A'}"
                )

        print("\n=== REPORT COMPLETATO ===")
        print(f"CSV REPORT: {out_report.resolve()}")
        print(f"CSV DEBUG:  {out_debug.resolve()}")
        return 0

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Processo interrotto dall’utente (Ctrl+C). Uscita pulita.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
