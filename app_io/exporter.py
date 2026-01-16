from __future__ import annotations

from pathlib import Path
import pandas as pd


def export_report_csv(out_path: Path, report_df: pd.DataFrame) -> None:
    """
    Esporta TUTTO il report_df senza filtri (nessuna riga viene eliminata).
    Usa separatore ';' e encoding 'utf-8-sig' per compatibilità Excel (BOM).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = report_df.copy()

    # Normalizza nomi colonne (toglie BOM e spazi)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    # Se per qualche motivo l'ordine colonne cambia, forza ordine standard
    preferred = ["Indicatore", "Valore", "Unità"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    df.to_csv(out_path, sep=";", index=False, encoding="utf-8-sig")
