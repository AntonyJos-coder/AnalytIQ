"""Excel loading: every non-empty sheet becomes its own analyzable table."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.analytics.csv_analyzer import AnalyticsError, CSVAnalyzer


class ExcelAnalyzer(CSVAnalyzer):
    """Same analytics as CSVAnalyzer; the label includes the sheet name."""


def load_excel(path: str | Path, source_name: str | None = None) -> list[ExcelAnalyzer]:
    path = Path(path)
    name = source_name or path.name
    try:
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    except Exception as exc:
        raise AnalyticsError(f"Could not read the Excel workbook: {exc}") from exc
    analyzers: list[ExcelAnalyzer] = []
    for sheet, df in sheets.items():
        if df is None or df.dropna(how="all").empty:
            continue
        try:
            analyzers.append(ExcelAnalyzer(df, name, label=f"{name} [{sheet}]"))
        except AnalyticsError:
            continue
    if not analyzers:
        raise AnalyticsError("The workbook has no sheets with data.")
    return analyzers
