#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from app_io.loader import load_signal_csv
from app_io.exporter import export_report_csv

from engine.backtest import BacktestConfig, backtest_from_hold
from metrics import apply_metrics
from engine.equity_additive import equity_curve_from_trades_additive


# PATH default richiesto (puoi puntarlo alla tua cartella SIGNAL reale)
DEFAULT_SIGNALS_DIR = Path("/Users/claudio 1/n8n-shared/Test Data")
DEFAULT_REPORTS_DIR = Path("/Users/claudio 1/n8n-shared/Test Data")


def ask_yes_no(prompt: str, default_yes: bool = True) -> bool:
    default = "Y/n" if default_yes else "y/N"
    while True:
        ans = input(f"{prompt} [{default}]: ").strip().lower()
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
    if not folder.exists():
        return []
    return sorted([p for p in folder.glob("SIGNAL_*.csv") if p.is_file()])


def pick_file_and_load_interactive(signals_dir: Path) -> Tuple[Path, pd.DataFrame]:
    while True:
        files = list_signal_files(signals_dir)
        if not files:
            raise FileNotFoundError(f"Nessun file SIGNAL_*.csv trovato in: {signals_dir}")

        idx = ask_choice(
            [f.name for f in files],
            f"Seleziona il file SIGNAL su cui generare il report (cartella: {signals_dir}):",
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


def export_debug_signal_csv(path: Path, df: pd.DataFrame) -> None:
    """Esporta debug CSV con separatore ';' e virgola decimale (Excel IT)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(sep=";", path_or_buf=path, index=False, encoding="utf-8-sig", decimal=",")


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


def _upsert_metric(report_df: pd.DataFrame, name: str, value_raw: float, unit: str) -> pd.DataFrame:
    value_str = f"{value_raw:.3f}".replace(".", ",") if math.isfinite(value_raw) else "nan"
    mask = report_df["Indicatore"] == name
    if mask.any():
        report_df.loc[mask, "Valore_raw"] = value_raw
        report_df.loc[mask, "Valore"] = value_str
        report_df.loc[mask, "Unità"] = unit
        return report_df

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
        equity_start = ask_equity_start(default=100.0)

        # -------------------------
        # Signals dir
        # -------------------------
        signals_dir = Path(args.signals_dir)
        if str(signals_dir) == str(DEFAULT_SIGNALS_DIR):
            use_default = ask_yes_no(
                f"Confermi il path di default dei file SIGNAL?\n{DEFAULT_SIGNALS_DIR}",
                default_yes=True,
            )
            if not use_default:
                signals_dir = ask_path("Inserisci il path alternativo dei file SIGNAL", DEFAULT_SIGNALS_DIR)

        if not signals_dir.exists():
            print(f"[ERROR] Cartella SIGNAL non trovata: {signals_dir}")
            return 2

        # -------------------------
        # Output dir (qui usi DEFAULT_REPORTS_DIR)
        # -------------------------
        out_dir = DEFAULT_REPORTS_DIR
        use_default_out = ask_yes_no(
            f"Confermi la directory di default dove stampare i report?\n{DEFAULT_REPORTS_DIR}",
            default_yes=True,
        )
        if not use_default_out:
            out_dir = ask_path("Inserisci la directory alternativa dove stampare i report", DEFAULT_REPORTS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Report directory: {out_dir.resolve()}")

        # -------------------------
        # Input file
        # -------------------------
        if args.input.strip():
            src = Path(args.input)
            try:
                df = load_signal_csv(src)
            except Exception:
                print("\nFILE NON ADEGUATAMENTE FORMATTATO\n")
                src, df = pick_file_and_load_interactive(signals_dir)
        else:
            src, df = pick_file_and_load_interactive(signals_dir)

        _debug_df_snapshot(
            df,
            "DOPO LOAD CSV (prima backtest)",
            cols=[c for c in ["symbol", "isin", "datetime", "close", "HOLD", "SIGNAL"] if c in df.columns],
        )

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

        # =========================
        # Canonical PnL in € per trade (qty = 1)
        # LONG:  exit - entry
        # SHORT: entry - exit
        # =========================
        for col in ["entry_price", "exit_price"]:
            if col not in trades_df.columns:
                raise ValueError(f"trades_df non contiene '{col}': impossibile calcolare pnl_eur (qty=1).")

        entry = pd.to_numeric(trades_df["entry_price"], errors="coerce")
        exit_ = pd.to_numeric(trades_df["exit_price"], errors="coerce")

        side = (
            trades_df["side"].astype(str).str.upper().str.strip()
            if "side" in trades_df.columns
            else pd.Series("LONG", index=trades_df.index)
        )
        sign = pd.Series(1.0, index=trades_df.index)
        sign[side.isin({"SHORT", "SELL", "-1", "S"})] = -1.0

        trades_df["pnl_eur"] = ((exit_ - entry) * sign).fillna(0.0)

        print(
            "[DEBUG] pnl_eur sums:",
            "total=", float(trades_df["pnl_eur"].sum()),
            "win=", float(trades_df.loc[trades_df["pnl_eur"] > 0, "pnl_eur"].sum()),
            "loss=", float(trades_df.loc[trades_df["pnl_eur"] < 0, "pnl_eur"].sum()),
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

        # utile per metriche che leggono initial_capital
        equity_df["initial_capital"] = float(equity_start)

        # debug: conferma colonne additive presenti
        print("[DEBUG] equity_df columns AFTER equity_additive:", list(equity_df.columns))
        if "datetime" in equity_df.columns and "equity_additive" in equity_df.columns:
            print("[DEBUG] equity_additive tail:\n",
                  equity_df[["datetime", "equity_additive"]].tail(5).to_string(index=False))

        # -------------------------
        # Report metriche
        # -------------------------
        import os

        print("\n[DEBUG] RUNNING FILE:", os.path.abspath(__file__))
        print("[DEBUG] BEFORE apply_metrics() -> trades_df columns:", list(trades_df.columns))
        print("[DEBUG] trades_df head:\n", trades_df.head(3).to_string(index=False))

        gp_dbg = trades_df.loc[trades_df.get("pnl_trade_eur", pd.Series(dtype=float)) > 0, "pnl_trade_eur"].sum() if "pnl_trade_eur" in trades_df.columns else None
        gl_dbg = trades_df.loc[trades_df.get("pnl_trade_eur", pd.Series(dtype=float)) < 0, "pnl_trade_eur"].sum() if "pnl_trade_eur" in trades_df.columns else None
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

        print("[DEBUG] DIRECT metric call GP(-Outlier):", gross_profit_outlier_filtered(equity_df, trades_df))
        print("[DEBUG] DIRECT metric call GL(-Outlier):", gross_loss_outlier_filtered(equity_df, trades_df))
        print("[DEBUG] DIRECT metric call NP(-Outlier):", net_profit_outlier_filtered(equity_df, trades_df))
        print("[DEBUG] DIRECT metric call Outliers count:", outliers_removed_count(equity_df, trades_df))
        print("[DEBUG] DIRECT metric call Outliers %:", outliers_removed_pct(equity_df, trades_df))

        report_df = apply_metrics(equity_df, trades_df)

        def _dbg(label: str):
            cols = ["Indicatore", "Valore"]
            if "Valore_raw" in report_df.columns:
                cols.append("Valore_raw")

            sub = report_df.loc[
                report_df["Indicatore"].isin(["Equity Start", "Net Profit", "Equity End", "Equity End (Additive)"]),
                cols
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

        print(report_df.loc[
                  report_df["Indicatore"].isin([
                      "Gross Profit (-Outlier)",
                      "Gross Loss (-Outlier)",
                      "Net Profit (-Outlier)",
                      "Outliers Removed (count)",
                      "Outliers Removed (%)",
                  ]),
                  ["Indicatore", "Valore", "Valore_raw", "Unità"]
              ].to_string(index=False))

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

        report_df["Valore"] = report_df["Valore_raw"].apply(lambda v: fmt_eu(v, nd=6))

        # =========================
        # Colonna "Verificata" (Y/N) – SOLO fase di stampa
        # =========================
        verified_metrics = {
            # ---- Performance / Equity ----
            "Equity Start",
            "Equity End (Additive)",
            "Equity End",
            "Total Return (Additive)",
            "Total Return"
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
            "Entries (OUT->IN)",
            "Exits (IN->OUT)",
            "Win Rate (Round-Trip)",

            # ---- PnL ----
            "Gross Profit (Winning Trades Only)",
            "Gross Loss (Losing Trades Only)",
            "Net Profit",
            "Profit Factor",
        }

        report_df["Verificata"] = report_df["Indicatore"].apply(
            lambda x: "Y" if x in verified_metrics else "N"
        )

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

        print(
            "[DEBUG] CHECK KPI: GP+GL=",
            gp + gl,
            " NetProfit=",
            npv,
            " Delta=",
            npv - (gp + gl),
        )

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
            "Transaction Costs",
            "Win Rate (Round-Trip)",
            "AVG Win",
            "AVG Loss",
            "Expectancy (per Trade)",
            "Max Drawdown (Additive)",
            "Gross Profit (-Outlier)",
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
        report_df["_sort_key"] = report_df["Indicatore"].map(order_map).fillna(len(REPORT_ORDER)).astype(int)

        # tie-breaker per stabilita': mantiene l'ordine originale tra gli "extra"
        report_df["_sort_seq"] = range(len(report_df))

        report_df = report_df.sort_values(by=["_sort_key", "_sort_seq"], ascending=True).drop(
            columns=["_sort_key", "_sort_seq"])

        # -------------------------
        # Export
        # -------------------------
        out_report = out_dir / f"REPORT_SIGNAL_{src.stem}.csv"
        export_report_csv(out_report, report_df)

        # Debug dump completo SIGNAL + Trade_ID + Trade_profit
        try:
            debug_df = add_trade_id_and_profit(df)
        except Exception as e:
            print(f"[WARN] Impossibile calcolare Trade_ID/Trade_profit nel debug: {e}")
            debug_df = df.copy()
            debug_df["Trade_ID"] = pd.Series([pd.NA] * len(debug_df), dtype="Int64")
            debug_df["Trade_profit"] = np.nan

        out_debug = out_dir / f"REPORT_DEBUG_{src.stem}.csv"
        export_debug_signal_csv(out_debug, debug_df)

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
