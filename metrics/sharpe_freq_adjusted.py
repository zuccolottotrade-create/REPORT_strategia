# metrics/sharpe_freq_adjusted.py
from __future__ import annotations

from typing import Any
import pandas as pd

from .base import register_metric


@register_metric(name="Sharpe (freq-adjusted)", unit="ratio")
def compute(equity_df: pd.DataFrame, trades_df: Any = None, params: Any = None) -> float:
    """
    METRICA TEMPORANEAMENTE DISABILITATA.

    Motivo:
    - in fase di riallineamento concettuale delle metriche (additive vs equity)
    - Sharpe richiede una definizione univoca della curva di ritorni

    Comportamento:
    - restituisce sempre 0.0
    - non genera warning, errori o NaN
    """
    return 0.0
