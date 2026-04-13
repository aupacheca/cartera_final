"""Utilidades mínimas sin Streamlit (p. ej. acceso seguro a filas pandas)."""
from __future__ import annotations

import pandas as pd


def safe_get(row, key: str, default=None):
    return row[key] if key in row and not pd.isna(row[key]) else default


def to_float(x, default=0.0):
    """Convierte a float aceptando coma como separador decimal (ej. '45,0' -> 45.0)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return default
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).strip().replace(",", "."))
    except (ValueError, TypeError):
        return default
