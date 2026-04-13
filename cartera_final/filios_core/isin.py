"""ISIN, catálogo global instrument_catalog y resolución para movimientos/FIFO."""
from __future__ import annotations

import sqlite3

from filios_core.db import get_db
from filios_core.util import safe_get as _safe_get


def _init_instrument_catalog() -> None:
    """Tabla ISIN y metadatos globales por ticker_Yahoo (clave única de cotización)."""
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS instrument_catalog (
                ticker_Yahoo TEXT PRIMARY KEY,
                isin TEXT
            )
            """
        )
        conn.commit()


def _looks_like_isin(code: str) -> bool:
    """ISIN estándar: 2 letras de país + 9 identificadores + dígito de control (12 alfanum)."""
    t = (code or "").strip().upper().replace(" ", "")
    if len(t) != 12 or not t.isalnum():
        return False
    return t[:2].isalpha()


def lookup_ticker_yahoo_by_isin(isin: str) -> str | None:
    """
    ticker_Yahoo a partir del ISIN usando instrument_catalog (mapa inverso).
    Rellena en «Instrumentos» el ticker Yahoo correcto y el ISIN del fondo/ETF.
    """
    if not _looks_like_isin(isin):
        return None
    key = isin.strip().upper().replace(" ", "")
    _init_instrument_catalog()
    with get_db() as conn:
        cur = conn.execute(
            """
            SELECT ticker_Yahoo FROM instrument_catalog
            WHERE upper(replace(trim(isin), ' ', '')) = ?
            LIMIT 1
            """,
            (key,),
        )
        row = cur.fetchone()
    if not row or not row[0]:
        return None
    y = str(row[0]).strip()
    return y or None


def _lookup_isin_for_ticker_yahoo(ticker_yahoo: str) -> str:
    """ISIN normalizado desde instrument_catalog para una clave ticker_Yahoo."""
    ty = str(ticker_yahoo or "").strip()
    if not ty:
        return ""
    _init_instrument_catalog()
    with get_db() as conn:
        try:
            cur = conn.execute(
                "SELECT isin FROM instrument_catalog WHERE ticker_Yahoo = ? LIMIT 1",
                (ty,),
            )
            row_db = cur.fetchone()
        except sqlite3.OperationalError:
            return ""
    if not row_db or not row_db[0]:
        return ""
    s = str(row_db[0]).strip().upper().replace(" ", "")
    return s if _looks_like_isin(s) else ""


def _norm_isin_field(val) -> str:
    s = str(val or "").strip().upper().replace(" ", "")
    return s if s and _looks_like_isin(s) else ""


def _isin_required_acciones_etf(tipo_registro: str, position_type: str) -> bool:
    return tipo_registro == "Acciones/ETFs" and str(position_type or "").strip().lower() in ("stock", "etf")


def _isin_required_fondos(tipo_registro: str, position_type: str) -> bool:
    return tipo_registro == "Fondos" or str(position_type or "").strip().lower() == "fund"


def _resolve_movimiento_isin(
    tipo_registro: str,
    es_posicion_nueva: bool,
    position_isin: str,
    ticker_yahoo: str,
    position_type: str,
) -> str:
    """ISIN para columna isin del movimiento: formulario si posición nueva; instrument_catalog si ya existe."""
    ty = str(ticker_yahoo or "").strip()
    if _isin_required_acciones_etf(tipo_registro, position_type):
        if es_posicion_nueva:
            return _norm_isin_field(position_isin)
        return _lookup_isin_for_ticker_yahoo(ty) or ""
    if _isin_required_fondos(tipo_registro, position_type):
        if es_posicion_nueva:
            return _norm_isin_field(position_isin)
        return _lookup_isin_for_ticker_yahoo(ty) or ""
    if es_posicion_nueva:
        return _norm_isin_field(position_isin)
    return ""


def _catalog_origen_requires_isin(origen_cell: str) -> bool:
    o = str(origen_cell or "")
    return "Acciones" in o or "Fondos" in o


def _fifo_resolve_isin_row(row, ticker_y: str, ticker_orig: str, cat_cache: dict[str, str]) -> str:
    """Prioridad: columnas del movimiento → ISIN en ticker fields → catálogo (cache)."""
    for cand in (_safe_get(row, "isin"), _safe_get(row, "ISIN")):
        ni = _norm_isin_field(cand)
        if ni:
            return ni
    ty = str(ticker_y or "").strip()
    to = str(ticker_orig or "").strip()
    for cand in (ty, to):
        ni = _norm_isin_field(cand)
        if ni:
            return ni
    for cand in (ty, to):
        if not cand:
            continue
        if cand not in cat_cache:
            cat_cache[cand] = _lookup_isin_for_ticker_yahoo(cand)
        if cat_cache[cand]:
            return cat_cache[cand]
    return ""
