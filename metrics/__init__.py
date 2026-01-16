from __future__ import annotations

import importlib
import math
from typing import Any

import pandas as pd

from .base import get_registry

# Elenco moduli metriche da importare (1 file = 1 metrica o gruppo metriche)
# Assicurati che esistano come files in metrics/<nome>.py
_METRIC_MODULES = [
    "equity_start",
    "equity_end",
    "total_return",
    "n_roundtrip_trades",
    "n_operations",
    "entries",
    "exits",
    "win_rate",
    "profit_factor",
    "volatility_per_trade",
    "time_in_position",
    "net_pnl_winning",
    "gross_pnl_trades",
    "avg_win",
    "avg_loss",
    "expectancy",
    "strategy_outperformance",
    "buy_and_hold_profit_FILO",
    "net_profit",
    "gross_pnl_outlier_filtered",

    # se usi le additive, lascia:
    "equity_end_additive",
    "total_return_additive",
    "max_drawdown_additive",
]

_LOADED = False


def _ensure_loaded() -> None:
    """
    Importa tutti i moduli metriche una sola volta.
    Ogni modulo, importandosi, registra le metriche via @register_metric.

    Se un modulo elencato non esiste, lo saltiamo (non blocchiamo il report).
    Se invece l'errore avviene *dentro* un modulo importato (syntax/import error interno),
    lo rilanciamo per non nascondere bug reali.
    """
    global _LOADED
    if _LOADED:
        return

    missing: list[str] = []

    for m in _METRIC_MODULES:
        try:
            importlib.import_module(f"{__name__}.{m}")
        except ModuleNotFoundError as e:
            # Salta solo se manca ESATTAMENTE quel modulo (file assente)
            if str(e).endswith(f"'{__name__}.{m}'"):
                missing.append(m)
                continue
            raise

    if missing:
        print("[WARN] Moduli metriche non trovati (saltati):", ", ".join(missing))

    _LOADED = True


def apply_metrics(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame | None,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Calcola tutte le metriche registrate e ritorna un DataFrame report.
    """
    _ensure_loaded()
    params = params or {}

    rows: list[dict[str, Any]] = []

    registry = get_registry()

    # Compatibile con registry sia dict che list/tuple
    if isinstance(registry, dict):
        specs = list(registry.values())
    else:
        specs = list(registry)

        for spec in specs:
            try:
                # Supporta metriche legacy a 2 argomenti (equity_df, trades_df)
                # e metriche nuove a 3 argomenti (equity_df, trades_df, params)
                try:
                    value = spec.fn(equity_df, trades_df, params)
                except TypeError:
                    value = spec.fn(equity_df, trades_df)

                # Normalizza NaN/inf
                if value is None:
                    out = None
                elif isinstance(value, (int, float)) and (math.isnan(value) or math.isinf(value)):
                    out = None
                else:
                    out = value

            except Exception as e:
                # DEBUG consigliato (poi puoi toglierlo)
                print(f"[METRIC ERROR] {spec.name}: {type(e).__name__}: {e}")
                out = None

            raw = out
            if isinstance(out, (int, float)):
                raw = float(out)

            rows.append(
                {
                    "Indicatore": spec.name,
                    "Valore": out,
                    "Valore_raw": raw,
                    "Unità": spec.unit,
                }
            )

        rows.append(
            {
                "Indicatore": spec.name,
                "Valore": out,
                "Valore_raw": raw,
                "Unità": spec.unit,
            }
        )

    return pd.DataFrame(rows)

