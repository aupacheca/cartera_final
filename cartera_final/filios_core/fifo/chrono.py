"""Orden cronológico estable para FIFO (desempate a igual fecha/hora). No escribe en BD."""
from __future__ import annotations

import pandas as pd

# Entradas antes que salidas a igual timestamp (evita G/P inestable compra+venta mismo segundo).
_STOCKS_INBOUND = frozenset(
    {
        "buy",
        "switchbuy",
        "bonus",
        "stakereward",
        "optionbuy",
        "spinoff",
        "split",
    }
)
_STOCKS_OUTBOUND = frozenset({"sell", "switch", "optionsell"})


def _stocks_chrono_type_order(s: pd.Series) -> pd.Series:
    """A igual timestamp: compras/entradas (0) antes que ventas/salidas (1); resto (2)."""
    t = s.astype(str).str.strip().str.lower()

    def _rank(x: str) -> int:
        if x in _STOCKS_INBOUND:
            return 0
        if x in _STOCKS_OUTBOUND:
            return 1
        return 2

    return t.map(_rank)


def _fondos_chrono_type_order(s: pd.Series) -> pd.Series:
    """Traspaso: switch antes que switchBuy; buy antes que sell; resto al medio."""
    t = s.astype(str).str.strip().str.lower()
    return t.map(
        {
            "switch": 0,
            "switchbuy": 1,
            "buy": 2,
            "sell": 3,
        }
    ).fillna(2)


def sort_movimientos_fifo_chrono(
    df: pd.DataFrame,
    *,
    kind: str = "stocks",
) -> pd.DataFrame:
    """
    Ordena movimientos para FIFO de forma determinista.

    1) datetime_full (o date)
    2) tipo (entradas antes que salidas; en fondos switch antes que switchBuy)
    3) _rowid_ si existe (orden de alta en SQLite)

    Solo afecta al cálculo en memoria; no modifica la base de datos.
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    d = df.copy()
    if "datetime_full" in d.columns:
        cols: list[str] = ["datetime_full"]
        asc: list[bool] = [True]
    elif "date" in d.columns:
        cols = ["date"]
        asc = [True]
    else:
        return d

    tie_col = "_tie_fifo_chrono"
    if "type" in d.columns:
        if kind == "fondos":
            d[tie_col] = _fondos_chrono_type_order(d["type"])
        else:
            d[tie_col] = _stocks_chrono_type_order(d["type"])
        cols.append(tie_col)
        asc.append(True)

    if "_rowid_" in d.columns:
        cols.append("_rowid_")
        asc.append(True)
    else:
        d = d.reset_index(drop=False)
        if "index" in d.columns:
            cols.append("index")
            asc.append(True)

    out = d.sort_values(cols, ascending=asc, kind="mergesort")
    out = out.drop(columns=[tie_col, "index"], errors="ignore")
    return out.reset_index(drop=True)
