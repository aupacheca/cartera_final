"""Claves de cola FIFO (homogeneidad por ISIN vs par broker+ticker)."""
from __future__ import annotations

from filios_core.isin import (
    _fifo_resolve_isin_row,
    _lookup_isin_for_ticker_yahoo,
    _norm_isin_field,
)
from filios_core.util import safe_get as _safe_get


def _fifo_queue_key_stocks(row, broker: str, key_ticker: str, cat_cache: dict[str, str]) -> tuple:
    """FIFO fiscal homogéneo: un único saco por ISIN si existe; si no, por broker+ticker Yahoo."""
    isin = _fifo_resolve_isin_row(row, key_ticker, _safe_get(row, "ticker"), cat_cache)
    if isin:
        return ("ISIN", isin)
    return ("PAIR", str(broker or "").strip(), str(key_ticker or "").strip())


def _fifo_split_affected_keys_stocks(
    lots_by_key: dict, row, broker, key_ticker: str, cat_cache: dict[str, str]
) -> list[tuple]:
    qk = _fifo_queue_key_stocks(row, broker or "", key_ticker, cat_cache)
    if qk[0] == "ISIN":
        return [qk]
    if broker:
        return [qk]
    return [k for k in lots_by_key if k[0] == "PAIR" and len(k) == 3 and k[2] == key_ticker]


def _fifo_queue_key_stocks_cartera(row, broker: str, key_ticker: str, cat_cache: dict[str, str]) -> tuple:
    """
    FIFO para posiciones vivas (Cartera): mismo ISIN en otra divisa o cuenta = saco distinto;
    mismo ISIN, misma cuenta y divisa y distinto ticker Yahoo = mismo saco (alias del mismo valor).
    """
    isin = _fifo_resolve_isin_row(row, key_ticker, _safe_get(row, "ticker"), cat_cache)
    br = str(broker or "").strip()
    ccy = str(_safe_get(row, "positionCurrency", "") or "").strip().upper() or "EUR"
    if isin:
        return ("ISIN_BR_CCY", isin, br, ccy)
    return ("PAIR", br, str(key_ticker or "").strip())


def _fifo_split_affected_keys_stocks_cartera(
    lots_by_key: dict, row, broker, key_ticker: str, cat_cache: dict[str, str]
) -> list[tuple]:
    qk = _fifo_queue_key_stocks_cartera(row, broker or "", key_ticker, cat_cache)
    br = str(broker or "").strip()
    if qk[0] == "ISIN_BR_CCY":
        _, isin_k, _, ccy_k = qk
        if br:
            return [qk]
        return [
            k
            for k in lots_by_key
            if len(k) == 4 and k[0] == "ISIN_BR_CCY" and k[1] == isin_k and k[3] == ccy_k
        ]
    if br:
        return [qk]
    return [k for k in lots_by_key if k[0] == "PAIR" and len(k) == 3 and k[2] == key_ticker]


def _fifo_queue_key_fondos(row, broker: str, ticker: str, ticker_y: str, cat_cache: dict[str, str]) -> tuple:
    ty = str(ticker_y or ticker or "").strip()
    to = str(ticker or "").strip()
    isin = _fifo_resolve_isin_row(row, ty, to, cat_cache)
    if isin:
        return ("ISIN", isin)
    return ("PAIR", str(broker or "").strip(), str(to or ty).strip())


def _fifo_fondo_pending_dest_fiscal(dest_ticker: str, cat_cache: dict[str, str]) -> tuple | None:
    d = str(dest_ticker or "").strip()
    if not d:
        return None
    ni = _norm_isin_field(d)
    if ni:
        return ("ISIN", ni)
    if d not in cat_cache:
        cat_cache[d] = _lookup_isin_for_ticker_yahoo(d)
    isin = cat_cache[d]
    return ("ISIN", isin) if isin else None
