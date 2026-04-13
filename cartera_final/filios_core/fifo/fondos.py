"""FIFO y posiciones de fondos (sin Streamlit)."""
from __future__ import annotations

import pandas as pd

from filios_core.fifo.keys import _fifo_fondo_pending_dest_fiscal, _fifo_queue_key_fondos
from filios_core.fifo.sale_extras import fifo_sale_amount_cols
from filios_core.isin import _fifo_resolve_isin_row
from filios_core.util import safe_get as _safe_get, to_float as _to_float

MIN_QTY_FONDOS = 1e-8


def compute_positions_fondos(df: pd.DataFrame) -> list[dict]:
    """
    Posiciones de fondos con traspasos fiscales españoles (coste arrastrado).
    df debe venir ordenado por load_data_fondos. Devuelve lista de dicts con broker, ticker, nombre, cantidad, coste_total_eur.
    """
    data = df.copy()
    lots_by_key: dict[tuple[str, str], list[dict]] = {}
    pending_traspasos: list[dict] = []

    for _, row in data.iterrows():
        broker = _safe_get(row, "broker")
        ticker = _safe_get(row, "ticker") or _safe_get(row, "ticker_Yahoo")
        tipo = (str(_safe_get(row, "type") or "")).strip().lower()
        fecha = _safe_get(row, "date")
        nombre = _safe_get(row, "nombre") or _safe_get(row, "name") or ticker or ""
        qty = _to_float(_safe_get(row, "positionNumber"), None)
        total_eur = _to_float(_safe_get(row, "totalWithComissionBaseCurrency"), None)
        if broker is None or ticker is None or pd.isna(ticker) or ticker == "" or qty is None or qty <= 0:
            continue
        key = (broker, ticker)
        if key not in lots_by_key:
            lots_by_key[key] = []

        if tipo == "buy":
            if total_eur is None:
                continue
            price_eur = total_eur / qty if qty > 0 else 0.0
            ty_row = str(_safe_get(row, "ticker_Yahoo") or "").strip()
            lots_by_key[key].append(
                {
                    "broker": broker,
                    "ticker": ticker,
                    "nombre": nombre,
                    "cantidad": float(qty),
                    "precio_medio_eur": float(price_eur),
                    "coste_total_eur": float(total_eur),
                    "fecha": fecha,
                    "ticker_yahoo": ty_row,
                }
            )
            continue
        if tipo == "switch":
            dest_ticker = str(_safe_get(row, "switchBuyPosition") or "").strip()
            if not dest_ticker:
                continue
            remaining, cost_trasladado = float(qty), 0.0
            fechas_consumidas: list[str] = []
            lots = lots_by_key.get(key, [])
            while remaining > MIN_QTY_FONDOS and lots:
                lote = lots[0]
                lote_qty = lote["cantidad"]
                lote_fecha = lote.get("fecha") or ""
                if lote_qty <= remaining + MIN_QTY_FONDOS:
                    cost_trasladado += lote["coste_total_eur"]
                    if lote_fecha:
                        fechas_consumidas.append(lote_fecha)
                    remaining -= lote_qty
                    lots.pop(0)
                else:
                    frac = remaining / lote_qty
                    cost_trasladado += lote["coste_total_eur"] * frac
                    if lote_fecha:
                        fechas_consumidas.append(lote_fecha)
                    lote["cantidad"] -= remaining
                    lote["coste_total_eur"] -= lote["coste_total_eur"] * frac
                    lote["precio_medio_eur"] = lote["coste_total_eur"] / lote["cantidad"] if lote["cantidad"] > 0 else 0
                    remaining = 0.0
            fecha_origen = min(fechas_consumidas) if fechas_consumidas else fecha
            pending_traspasos.append({"broker": broker, "dest_ticker": dest_ticker, "cost_eur": cost_trasladado, "fecha_origen": fecha_origen})
            continue
        if tipo == "switchbuy":
            ticker_s = str(ticker or "").strip()
            ticker_yahoo = str(_safe_get(row, "ticker_Yahoo") or "").strip()
            match_idx = None
            for i, p in enumerate(pending_traspasos):
                if p["broker"] != broker:
                    continue
                d = str(p["dest_ticker"] or "").strip()
                if d == ticker_s or d == ticker_yahoo:
                    match_idx = i
                    break
            if match_idx is not None:
                p = pending_traspasos.pop(match_idx)
                cost_eur = p["cost_eur"]
                fecha_origen = p.get("fecha_origen") or fecha
                price_eur = cost_eur / qty if qty > 0 else 0.0
                lots_by_key[key].append(
                    {
                        "broker": broker,
                        "ticker": ticker,
                        "nombre": nombre,
                        "cantidad": float(qty),
                        "precio_medio_eur": float(price_eur),
                        "coste_total_eur": float(cost_eur),
                        "fecha": fecha_origen,
                        "ticker_yahoo": ticker_yahoo,
                    }
                )
            elif total_eur is not None:
                price_eur = total_eur / qty if qty > 0 else 0.0
                lots_by_key[key].append(
                    {
                        "broker": broker,
                        "ticker": ticker,
                        "nombre": nombre,
                        "cantidad": float(qty),
                        "precio_medio_eur": float(price_eur),
                        "coste_total_eur": float(total_eur),
                        "fecha": fecha,
                        "ticker_yahoo": ticker_yahoo,
                    }
                )
            continue
        if tipo == "sell":
            remaining = float(qty)
            lots = lots_by_key.get(key, [])
            while remaining > MIN_QTY_FONDOS and lots:
                lote = lots[0]
                lote_qty = lote["cantidad"]
                if lote_qty <= remaining + MIN_QTY_FONDOS:
                    remaining -= lote_qty
                    lots.pop(0)
                else:
                    lote["cantidad"] -= remaining
                    lote["coste_total_eur"] -= remaining * lote["precio_medio_eur"]
                    remaining = 0.0
            continue

    resumen = []
    for (broker, ticker), lots in lots_by_key.items():
        total_cant = sum(l["cantidad"] for l in lots)
        if total_cant <= MIN_QTY_FONDOS:
            continue
        total_coste = sum(l["coste_total_eur"] for l in lots)
        nombre = lots[0]["nombre"] if lots else ""
        fechas = [l.get("fecha") for l in lots if l.get("fecha")]
        fecha_origen = min(fechas) if fechas else ""
        yahoo_candidates = [str(l.get("ticker_yahoo") or "").strip() for l in lots]
        raw_ty = ""
        for y in reversed(yahoo_candidates):
            if y:
                raw_ty = y
                break
        resumen.append(
            {
                "broker": broker,
                "ticker": ticker,
                "nombre": nombre,
                "cantidad": total_cant,
                "coste_total_eur": total_coste,
                "precio_medio_eur": total_coste / total_cant if total_cant > 0 else 0,
                "fecha_origen": fecha_origen,
                "ticker_yahoo": raw_ty,
            }
        )
    return resumen


def compute_fifo_fondos(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    FIFO para fondos: lotes vivas y ventas con plusvalía/minusvalía.
    Traspasos (switch→switchBuy) no generan venta fiscal; solo sell genera plusvalía/minusvalía.

    Con ISIN conocido (movimiento o catálogo), un único saco FIFO por ISIN.
    """
    data = df.copy()
    cat_cache: dict[str, str] = {}
    lots_by_key: dict[tuple, list[dict]] = {}
    pending_traspasos: list[dict] = []
    sales_rows: list[dict] = []
    sales_detail_rows: list[dict] = []

    for _, row in data.iterrows():
        broker = _safe_get(row, "broker")
        ticker = _safe_get(row, "ticker") or _safe_get(row, "ticker_Yahoo")
        ticker_y = _safe_get(row, "ticker_Yahoo") or ticker
        tipo = (str(_safe_get(row, "type") or "")).strip().lower()
        fecha = _safe_get(row, "date")
        nombre = _safe_get(row, "nombre") or _safe_get(row, "name") or ticker or ""
        qty = _to_float(_safe_get(row, "positionNumber"), None)
        total_eur = _to_float(_safe_get(row, "totalWithComissionBaseCurrency"), None)
        if broker is None or ticker is None or pd.isna(ticker) or ticker == "" or qty is None or qty <= 0:
            continue
        key = _fifo_queue_key_fondos(row, broker, ticker, ticker_y, cat_cache)
        if key not in lots_by_key:
            lots_by_key[key] = []

        if tipo == "buy":
            if total_eur is None:
                continue
            price_eur = total_eur / qty if qty > 0 else 0.0
            isin_l = _fifo_resolve_isin_row(row, str(ticker_y or ""), str(ticker or ""), cat_cache)
            lots_by_key[key].append({
                "Broker": broker,
                "Ticker": ticker,
                "Ticker_Yahoo": ticker_y,
                "ISIN": isin_l,
                "Nombre": nombre,
                "Fecha origen": fecha,
                "Cantidad": float(qty),
                "Precio medio €": float(price_eur),
                "Tipo activo": "fund",
            })
            continue
        if tipo == "switch":
            dest_ticker = str(_safe_get(row, "switchBuyPosition") or "").strip()
            if not dest_ticker:
                continue
            remaining, cost_trasladado = float(qty), 0.0
            fechas_consumidas: list[str] = []
            lots = lots_by_key.get(key, [])
            while remaining > MIN_QTY_FONDOS and lots:
                lote = lots[0]
                lote_qty = lote["Cantidad"]
                lote_fecha = lote.get("Fecha origen") or ""
                if lote_qty <= remaining + MIN_QTY_FONDOS:
                    cost_trasladado += lote["Cantidad"] * lote["Precio medio €"]
                    if lote_fecha:
                        fechas_consumidas.append(lote_fecha)
                    remaining -= lote_qty
                    lots.pop(0)
                else:
                    consumed = remaining
                    cost_trasladado += consumed * lote["Precio medio €"]
                    if lote_fecha:
                        fechas_consumidas.append(lote_fecha)
                    lote["Cantidad"] -= consumed
                    remaining = 0.0
            fecha_origen = min(fechas_consumidas) if fechas_consumidas else fecha
            pending_traspasos.append({
                "broker": broker,
                "dest_ticker": dest_ticker,
                "cost_eur": cost_trasladado,
                "fecha_origen": fecha_origen,
                "dest_fiscal": _fifo_fondo_pending_dest_fiscal(dest_ticker, cat_cache),
            })
            continue
        if tipo == "switchbuy":
            ticker_s = str(ticker or "").strip()
            ticker_yahoo_s = str(_safe_get(row, "ticker_Yahoo") or "").strip()
            cur_fk = _fifo_queue_key_fondos(row, broker, ticker_s, ticker_yahoo_s or ticker_s, cat_cache)
            match_idx = None
            for i, p in enumerate(pending_traspasos):
                dfk = p.get("dest_fiscal")
                if dfk and dfk == cur_fk:
                    match_idx = i
                    break
            if match_idx is None:
                for i, p in enumerate(pending_traspasos):
                    if p["broker"] != broker:
                        continue
                    d = str(p["dest_ticker"] or "").strip()
                    if d == ticker_s or d == ticker_yahoo_s:
                        match_idx = i
                        break
            if match_idx is not None:
                p = pending_traspasos.pop(match_idx)
                cost_eur = p["cost_eur"]
                fecha_origen = p.get("fecha_origen") or fecha
                price_eur = cost_eur / qty if qty > 0 else 0.0
                isin_l = _fifo_resolve_isin_row(row, str(ticker_yahoo_s or ticker_s), str(ticker_s or ""), cat_cache)
                lots_by_key[key].append({
                    "Broker": broker,
                    "Ticker": ticker,
                    "Ticker_Yahoo": ticker_y,
                    "ISIN": isin_l,
                    "Nombre": nombre,
                    "Fecha origen": fecha_origen,
                    "Cantidad": float(qty),
                    "Precio medio €": float(price_eur),
                    "Tipo activo": "fund",
                })
            elif total_eur is not None:
                price_eur = total_eur / qty if qty > 0 else 0.0
                isin_l = _fifo_resolve_isin_row(row, str(ticker_yahoo_s or ticker_s), str(ticker_s or ""), cat_cache)
                lots_by_key[key].append({
                    "Broker": broker,
                    "Ticker": ticker,
                    "Ticker_Yahoo": ticker_y,
                    "ISIN": isin_l,
                    "Nombre": nombre,
                    "Fecha origen": fecha,
                    "Cantidad": float(qty),
                    "Precio medio €": float(price_eur),
                    "Tipo activo": "fund",
                })
            continue
        if tipo == "sell":
            if total_eur is None:
                continue
            qty_sell = float(qty)
            remaining = qty_sell
            cost_hist = 0.0
            tranches: list[dict] = []
            lots = lots_by_key.get(key, [])
            while remaining > MIN_QTY_FONDOS and lots:
                lote = lots[0]
                lote_qty = lote["Cantidad"]
                if lote_qty <= remaining + MIN_QTY_FONDOS:
                    consumed = lote_qty
                    cost_hist += consumed * lote["Precio medio €"]
                    tranches.append(
                        {
                            "consumed": consumed,
                            "Fecha origen lote": lote.get("Fecha origen"),
                            "Precio medio €": float(lote["Precio medio €"]),
                        }
                    )
                    remaining -= consumed
                    lots.pop(0)
                else:
                    consumed = remaining
                    cost_hist += consumed * lote["Precio medio €"]
                    tranches.append(
                        {
                            "consumed": consumed,
                            "Fecha origen lote": lote.get("Fecha origen"),
                            "Precio medio €": float(lote["Precio medio €"]),
                        }
                    )
                    lote["Cantidad"] -= consumed
                    remaining = 0.0
            # Valor de transmisión fiscal: totalBase - comisión - impuestos (normativa española)
            total_base = _to_float(_safe_get(row, "totalBaseCurrency"), 0.0)
            fx = _to_float(_safe_get(row, "exchangeRate"), 1.0) or 1.0
            comm = _to_float(_safe_get(row, "comission"), 0.0)
            tax = _to_float(_safe_get(row, "taxes"), 0.0)
            comm_ccy = str(_safe_get(row, "comissionCurrency") or "").strip().upper()
            tax_ccy = str(_safe_get(row, "taxesCurrency") or "").strip().upper()
            comm_eur = comm if comm_ccy == "EUR" else (comm * fx if fx and abs(fx) > 1e-9 else comm)
            tax_eur = tax if tax_ccy == "EUR" else (tax * fx if fx and abs(fx) > 1e-9 else tax)
            valor_transmision = total_base - comm_eur - tax_eur
            vt = float(valor_transmision)

            dest_ret = _to_float(_safe_get(row, "destinationRetentionBaseCurrency"), 0.0)
            sale_id = len(sales_rows)
            isin_sale = _fifo_resolve_isin_row(row, str(ticker_y or ""), str(ticker or ""), cat_cache)
            v_allocated = 0.0
            for i, t in enumerate(tranches):
                c_tramo = t["consumed"] * t["Precio medio €"]
                if i == len(tranches) - 1:
                    v_tramo = vt - v_allocated
                else:
                    v_tramo = vt * (t["consumed"] / qty_sell) if qty_sell > 1e-12 else 0.0
                    v_allocated += v_tramo
                sales_detail_rows.append(
                    {
                        "Venta #": sale_id,
                        "Origen FIFO": "Fondos",
                        "Broker": broker,
                        "Ticker": ticker,
                        "ISIN": isin_sale,
                        "Ticker_Yahoo": ticker_y,
                        "Nombre": nombre,
                        "Tipo activo": "fund",
                        "Tipo movimiento": "sell",
                        "Fecha venta": fecha,
                        "Cantidad venta (total)": qty_sell,
                        "Cantidad (tramo)": t["consumed"],
                        "Fecha origen (lote)": t["Fecha origen lote"],
                        "Valor compra histórico (€)": c_tramo,
                        "Valor venta (€)": v_tramo,
                        "Plusvalía / Minusvalía (€)": v_tramo - c_tramo,
                    }
                )

            plusvalia = vt - cost_hist
            sales_rows.append({
                "Broker": broker,
                "Ticker": ticker,
                "ISIN": isin_sale,
                "Ticker_Yahoo": ticker_y,
                "Nombre": nombre,
                "Fecha venta": fecha,
                "Cantidad vendida": qty_sell,
                "Valor compra histórico (€)": cost_hist,
                "Valor venta (€)": vt,
                "Plusvalía / Minusvalía (€)": plusvalia,
                "Retención dest. (€)": dest_ret,
                "Tipo activo": "fund",
                **fifo_sale_amount_cols(row, 1.0),
            })
            continue

    lots_rows = []
    for key, lots in lots_by_key.items():
        for lote in lots:
            if lote["Cantidad"] > MIN_QTY_FONDOS:
                lots_rows.append(lote)
    lots_df = pd.DataFrame(lots_rows)
    sales_df = pd.DataFrame(sales_rows)
    sales_detail_df = pd.DataFrame(sales_detail_rows)
    return lots_df, sales_df, sales_detail_df
