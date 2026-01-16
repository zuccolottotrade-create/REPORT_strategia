from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_number_smart(x: object) -> float:
    if x is None:
        return float("nan")
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    s = s.replace(" ", "")

    has_dot = "." in s
    has_comma = "," in s

    if has_dot and has_comma:
        s = s.replace(".", "").replace(",", ".")
    elif has_comma and not has_dot:
        s = s.replace(",", ".")
    return float(s)


def _norm_colname(c: str) -> str:
    return str(c).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _build_norm_map(columns: List[str]) -> Dict[str, str]:
    return {_norm_colname(c): c for c in columns}


def coerce_numeric_like_columns(df: pd.DataFrame, exclude: set[str] | None = None) -> pd.DataFrame:
    exclude = exclude or set()
    dfx = df.copy()
    for c in dfx.columns:
        if c in exclude:
            continue
        if dfx[c].dtype == "object":
            sample = dfx[c].astype(str).head(50)
            numericish = sample.str.contains(r"\d", regex=True).mean()
            if numericish >= 0.7:
                try:
                    dfx[c] = dfx[c].apply(parse_number_smart)
                except Exception:
                    pass
    return dfx


def load_signal_csv(path: Path) -> pd.DataFrame:
    """
    Carica un CSV SIGNAL_*.csv e normalizza:
    - datetime da 'datetime'/'timestamp' oppure date+time (tollerante ai nomi)
    - close float (EU/US)
    - HOLD in {'IN','OUT'}
    """
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, sep=None, engine="python")
    norm_map = _build_norm_map(list(df.columns))

    # datetime
    datetime_candidates = ["datetime", "timestamp", "ts", "date_time", "datetimestamp", "dateandtime"]
    datetime_col = None
    for cand in datetime_candidates:
        key = _norm_colname(cand)
        if key in norm_map:
            datetime_col = norm_map[key]
            break

    if datetime_col:
        df["datetime"] = pd.to_datetime(df[datetime_col], errors="coerce")
    else:
        date_candidates = ["date", "data", "giorno"]
        time_candidates = ["time", "ora", "hour", "hours"]
        date_col = None
        time_col = None

        for cand in date_candidates:
            key = _norm_colname(cand)
            if key in norm_map:
                date_col = norm_map[key]
                break

        for cand in time_candidates:
            key = _norm_colname(cand)
            if key in norm_map:
                time_col = norm_map[key]
                break

        if date_col and time_col:
            df["datetime"] = pd.to_datetime(
                df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip(),
                errors="coerce",
            )
        else:
            raise ValueError("Missing datetime columns")

    if df["datetime"].isna().any():
        raise ValueError("Invalid datetime values")

    # close
    close_candidates = ["close", "last", "chiusura", "prezzochiusura"]
    close_col = None
    for cand in close_candidates:
        key = _norm_colname(cand)
        if key in norm_map:
            close_col = norm_map[key]
            break
    if not close_col:
        raise ValueError("Missing close column")

    df["close"] = df[close_col].apply(parse_number_smart)
    if df["close"].isna().any():
        raise ValueError("Invalid close values")
    if (df["close"] <= 0).any():
        raise ValueError("Non-positive close values")

    # HOLD
    hold_candidates = ["hold", "position", "posizione", "inout"]
    hold_col = None
    for cand in hold_candidates:
        key = _norm_colname(cand)
        if key in norm_map:
            hold_col = norm_map[key]
            break
    if not hold_col:
        raise ValueError("Missing HOLD column")

    df["HOLD"] = df[hold_col].astype(str).str.strip().str.upper()
    invalid = set(df["HOLD"].unique()) - {"IN", "OUT"}
    if invalid:
        raise ValueError(f"Invalid HOLD values: {sorted(list(invalid))}")

    # ordinamento deterministico
    group_cols = [c for c in ["symbol", "isin"] if c in df.columns]
    sort_cols = group_cols + ["datetime"] if group_cols else ["datetime"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # coercizione numerica su altre colonne (se presenti)
    df = coerce_numeric_like_columns(df, exclude=set(group_cols) | {"date", "time", "HOLD", "datetime"})

    return df
