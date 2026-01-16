from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List
import pandas as pd

MetricFn = Callable[[pd.DataFrame, pd.DataFrame, Dict[str, Any]], float]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    unit: str
    fn: MetricFn


_REGISTRY: List[MetricSpec] = []


def register_metric(name: str, unit: str):
    def decorator(fn: MetricFn):
        _REGISTRY.append(MetricSpec(name=name, unit=unit, fn=fn))
        return fn
    return decorator


def get_registry() -> List[MetricSpec]:
    return list(_REGISTRY)
