"""Importes de venta en EUR por movimiento (análisis / tablas de ventas)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from filios_core.util import safe_get as _safe_get, to_float as _to_float


def fifo_sale_amount_cols(row, frac: float = 1.0) -> dict[str, float]:
    """
    Desglose coherente con el cálculo fiscal de transmisión (totalBase − com − imp en EUR).
    frac permite repartir proporcionalmente (p. ej. cierres parciales de opción).
    """
    try:
        f = float(frac)
    except (TypeError, ValueError):
        f = 1.0
    if not np.isfinite(f):
        f = 1.0

    total_base = _to_float(_safe_get(row, "totalBaseCurrency"), 0.0) * f
    fx = _to_float(_safe_get(row, "exchangeRate"), 1.0) or 1.0
    comm = _to_float(_safe_get(row, "comission"), 0.0) * f
    tax = _to_float(_safe_get(row, "taxes"), 0.0) * f
    comm_ccy = str(_safe_get(row, "comissionCurrency") or "").strip().upper()
    tax_ccy = str(_safe_get(row, "taxesCurrency") or "").strip().upper()
    comm_eur = comm if comm_ccy == "EUR" else (comm * fx if fx and abs(fx) > 1e-9 else comm)
    tax_eur = tax if tax_ccy == "EUR" else (tax * fx if fx and abs(fx) > 1e-9 else tax)
    twc = pd.to_numeric(_safe_get(row, "totalWithComissionBaseCurrency"), errors="coerce")
    liq = float(twc) * f if not pd.isna(twc) else np.nan
    return {
        "Total bruto (€)": float(total_base),
        "Comisión+imp. venta (€)": float(comm_eur + tax_eur),
        "Liquidación neta mov. (€)": float(liq),
    }
