"""FIFO acciones/ETFs (compute_fifo_all), sin Streamlit."""
from __future__ import annotations

import pandas as pd

from filios_core.constants import MIN_POSITION
from filios_core.fifo.keys import _fifo_queue_key_stocks, _fifo_split_affected_keys_stocks
from filios_core.fifo.sale_extras import fifo_sale_amount_cols
from filios_core.isin import _fifo_resolve_isin_row
from filios_core.util import safe_get as _safe_get, to_float as _to_float


def compute_fifo_all(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calcula posiciones vivas y ventas para todos los tickers y brokers
    usando FIFO por lotes, aplicando splits sobre los lotes previos.

    Si el instrumento tiene ISIN (movimiento o instrument_catalog), el FIFO es **global por ISIN**
    (mismo saco aunque el listing o el broker difieran). Sin ISIN, se mantiene FIFO por (broker, ticker Yahoo).

    Devuelve:
      - lots_df: lotes vivos (cada lote conserva Broker/Nombre originales)
      - sales_df: ventas con coste histórico y plusvalía/minusvalía
      - sales_detail_df: una fila por cada tramo FIFO consumido en cada venta/permuta
    """
    if "datetime_full" in df.columns:
        data = df.sort_values("datetime_full").copy()
    elif "date" in df.columns:
        data = df.sort_values("date").copy()
    else:
        data = df.copy()

    cat_cache: dict[str, str] = {}
    lots_by_key: dict[tuple, list[dict]] = {}
    sales_rows: list[dict] = []
    sales_detail_rows: list[dict] = []

    for _, row in data.iterrows():
        broker = _safe_get(row, "broker")
        ticker_y = _safe_get(row, "ticker_Yahoo") or _safe_get(row, "ticker")
        ticker_orig = _safe_get(row, "ticker")
        nombre = _safe_get(row, "name") or ticker_orig or ticker_y or ""
        tipo = _safe_get(row, "type")
        tipo_lower = str(tipo or "").strip().lower()
        fecha = _safe_get(row, "date")
        tipo_activo = str(_safe_get(row, "positionType", "") or "").strip().lower()

        qty = pd.to_numeric(_safe_get(row, "positionNumber"), errors="coerce")
        total_eur = pd.to_numeric(
            _safe_get(row, "totalWithComissionBaseCurrency"), errors="coerce"
        )
        if pd.isna(total_eur):
            total_eur = pd.to_numeric(_safe_get(row, "totalBaseCurrency"), errors="coerce")

        # Ignoramos filas sin ticker (usa ticker_Yahoo o ticker como fallback para Otros/warrants)
        if ticker_y is None or (isinstance(ticker_y, str) and not str(ticker_y).strip()):
            continue

        key_ticker = str(ticker_y).strip()

        # -------- SPLIT (puede venir sin broker, como BY6) --------
        if tipo_lower == "split":
            factor = pd.to_numeric(_safe_get(row, "positionNumber"), errors="coerce")
            if pd.isna(factor) or float(factor) <= 0:
                continue
            factor = float(factor)

            affected_keys = _fifo_split_affected_keys_stocks(lots_by_key, row, broker, key_ticker, cat_cache)

            for key_s in affected_keys:
                for lote in lots_by_key.get(key_s, []):
                    lote["Cantidad"] *= factor
                    if factor != 0:
                        lote["Precio medio €"] /= factor
            continue

        if broker is None or pd.isna(qty):
            continue
        qty_f = float(qty)
        if tipo_lower in ("sell", "switch", "optionsell"):
            if abs(qty_f) < MIN_POSITION:
                continue
        elif tipo_lower in ("buy", "switchbuy", "bonus", "stakereward", "optionbuy"):
            if qty_f <= 0:
                continue
        elif qty_f <= 0:
            continue

        key = _fifo_queue_key_stocks(row, broker, key_ticker, cat_cache)
        if key not in lots_by_key:
            lots_by_key[key] = []

        # -------- COMPRAS: crean lotes (acciones / permutas; no opciones Put/Call) --------
        # bonus / stakereward: títulos a coste 0 (camp promo, regalo, etc.); mismo tratamiento fiscal FIFO que compra a 0 €
        if tipo_lower in ("buy", "switchbuy", "bonus", "stakereward"):
            if pd.isna(total_eur):
                total_eur = 0.0  # regalo / sin contrapartida en €: lote a coste 0
            price_eur = float(total_eur) / float(qty_f) if qty_f > 0 else 0.0
            isin_buy = _fifo_resolve_isin_row(row, key_ticker, ticker_orig, cat_cache)
            lots_by_key[key].append(
                {
                    "Broker": broker,
                    "Ticker": ticker_orig or key_ticker,
                    "Ticker_Yahoo": ticker_y,
                    "ISIN": isin_buy,
                    "Nombre": nombre,
                    "Fecha origen": fecha,
                    "Cantidad": float(qty_f),
                    "Precio medio €": float(price_eur),
                    "Tipo activo": tipo_activo,
                }
            )

        # -------- Compra prima: cerrar corto Put/Call o abrir largo --------
        elif tipo_lower == "optionbuy" and tipo_activo in ("putoption", "calloption"):
            if pd.isna(total_eur) or qty_f <= 0:
                continue
            remaining = float(qty_f)
            lots = lots_by_key[key]
            while remaining > 1e-8 and lots and float(lots[0]["Cantidad"]) < -1e-8:
                lote = lots[0]
                abs_s = abs(float(lote["Cantidad"]))
                cover = min(remaining, abs_s)
                premio_cobrado = cover * float(lote["Precio medio €"])
                cover_frac = cover / qty_f if qty_f > 1e-12 else 1.0
                twc_full = pd.to_numeric(_safe_get(row, "totalWithComissionBaseCurrency"), errors="coerce")
                if pd.isna(twc_full):
                    total_base = _to_float(_safe_get(row, "totalBaseCurrency"), 0.0)
                    fx = _to_float(_safe_get(row, "exchangeRate"), 1.0) or 1.0
                    comm = _to_float(_safe_get(row, "comission"), 0.0)
                    tax = _to_float(_safe_get(row, "taxes"), 0.0)
                    comm_ccy = str(_safe_get(row, "comissionCurrency") or "").strip().upper()
                    tax_ccy = str(_safe_get(row, "taxesCurrency") or "").strip().upper()
                    comm_eur = comm if comm_ccy == "EUR" else (comm * fx if fx and abs(fx) > 1e-9 else comm)
                    tax_eur = tax if tax_ccy == "EUR" else (tax * fx if fx and abs(fx) > 1e-9 else tax)
                    twc_full = total_base + comm_eur + tax_eur
                pagado_cierre = float(twc_full) * cover_frac
                sale_id = len(sales_rows)
                isin_sale = _fifo_resolve_isin_row(row, key_ticker, ticker_orig, cat_cache)
                dest_ret = _to_float(_safe_get(row, "destinationRetentionBaseCurrency"), 0.0)
                _cov_frac = cover / qty_f if qty_f > 1e-12 else 1.0
                _sale_ex = fifo_sale_amount_cols(row, _cov_frac)
                sales_detail_rows.append(
                    {
                        "Venta #": sale_id,
                        "Origen FIFO": "Acciones/ETFs",
                        "Broker": broker,
                        "Ticker": ticker_orig or key_ticker,
                        "ISIN": isin_sale,
                        "Ticker_Yahoo": ticker_y,
                        "Nombre": nombre,
                        "Tipo activo": tipo_activo,
                        "Tipo movimiento": tipo_lower,
                        "Fecha venta": fecha,
                        "Cantidad venta (total)": cover,
                        "Cantidad (tramo)": cover,
                        "Fecha origen (lote)": lote.get("Fecha origen"),
                        "Valor compra histórico (€)": premio_cobrado,
                        "Valor venta (€)": pagado_cierre,
                        "Plusvalía / Minusvalía (€)": premio_cobrado - pagado_cierre,
                    }
                )
                sales_rows.append(
                    {
                        "Broker": broker,
                        "Ticker": ticker_orig or key_ticker,
                        "ISIN": isin_sale,
                        "Ticker_Yahoo": ticker_y,
                        "Nombre": nombre,
                        "Fecha venta": fecha,
                        "Cantidad vendida": float(cover),
                        "Valor compra histórico (€)": premio_cobrado,
                        "Valor venta (€)": pagado_cierre,
                        "Plusvalía / Minusvalía (€)": premio_cobrado - pagado_cierre,
                        "Retención dest. (€)": dest_ret,
                        "Tipo activo": tipo_activo,
                        **_sale_ex,
                    }
                )
                lote["Cantidad"] += cover
                remaining -= cover
                if abs(float(lote["Cantidad"])) < 1e-8:
                    lots.pop(0)
            if remaining > 1e-8:
                te = float(total_eur) * (remaining / qty_f)
                price_eur = te / remaining if remaining > 1e-12 else 0.0
                isin_buy = _fifo_resolve_isin_row(row, key_ticker, ticker_orig, cat_cache)
                lots_by_key[key].append(
                    {
                        "Broker": broker,
                        "Ticker": ticker_orig or key_ticker,
                        "Ticker_Yahoo": ticker_y,
                        "ISIN": isin_buy,
                        "Nombre": nombre,
                        "Fecha origen": fecha,
                        "Cantidad": remaining,
                        "Precio medio €": float(price_eur),
                        "Tipo activo": tipo_activo,
                    }
                )

        elif tipo_lower == "optionbuy":
            if pd.isna(total_eur) or qty_f <= 0:
                continue
            price_eur = float(total_eur) / qty_f if qty_f > 0 else 0.0
            isin_buy = _fifo_resolve_isin_row(row, key_ticker, ticker_orig, cat_cache)
            lots_by_key[key].append(
                {
                    "Broker": broker,
                    "Ticker": ticker_orig or key_ticker,
                    "Ticker_Yahoo": ticker_y,
                    "ISIN": isin_buy,
                    "Nombre": nombre,
                    "Fecha origen": fecha,
                    "Cantidad": float(qty_f),
                    "Precio medio €": float(price_eur),
                    "Tipo activo": tipo_activo,
                }
            )

        # -------- VENTA prima opción: cerrar largo o abrir corto (sin fila G/P hasta el cierre) --------
        elif tipo_lower == "optionsell" and tipo_activo in ("putoption", "calloption"):
            if pd.isna(total_eur):
                continue
            qty_sell = abs(qty_f)
            lots = lots_by_key[key]
            total_long = sum(max(0.0, float(l["Cantidad"])) for l in lots)
            if total_long > 1e-8:
                remaining = qty_sell
                cost_hist = 0.0
                tranches: list[dict] = []
                while remaining > 0 and lots:
                    lote = lots[0]
                    lote_qty = float(lote["Cantidad"])
                    if lote_qty <= 1e-8:
                        lots.pop(0)
                        continue
                    if lote_qty <= remaining + 1e-8:
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
                isin_sale = _fifo_resolve_isin_row(row, key_ticker, ticker_orig, cat_cache)
                v_allocated = 0.0
                for i, ttr in enumerate(tranches):
                    c_tramo = ttr["consumed"] * ttr["Precio medio €"]
                    if i == len(tranches) - 1:
                        v_tramo = vt - v_allocated
                    else:
                        v_tramo = vt * (ttr["consumed"] / qty_sell) if qty_sell > 1e-12 else 0.0
                        v_allocated += v_tramo
                    sales_detail_rows.append(
                        {
                            "Venta #": sale_id,
                            "Origen FIFO": "Acciones/ETFs",
                            "Broker": broker,
                            "Ticker": ticker_orig or key_ticker,
                            "ISIN": isin_sale,
                            "Ticker_Yahoo": ticker_y,
                            "Nombre": nombre,
                            "Tipo activo": tipo_activo,
                            "Tipo movimiento": tipo_lower,
                            "Fecha venta": fecha,
                            "Cantidad venta (total)": qty_sell,
                            "Cantidad (tramo)": ttr["consumed"],
                            "Fecha origen (lote)": ttr["Fecha origen lote"],
                            "Valor compra histórico (€)": c_tramo,
                            "Valor venta (€)": v_tramo,
                            "Plusvalía / Minusvalía (€)": v_tramo - c_tramo,
                        }
                    )
                plusvalia = vt - cost_hist
                sales_rows.append(
                    {
                        "Broker": broker,
                        "Ticker": ticker_orig or key_ticker,
                        "ISIN": isin_sale,
                        "Ticker_Yahoo": ticker_y,
                        "Nombre": nombre,
                        "Fecha venta": fecha,
                        "Cantidad vendida": float(qty_sell),
                        "Valor compra histórico (€)": cost_hist,
                        "Valor venta (€)": vt,
                        "Plusvalía / Minusvalía (€)": plusvalia,
                        "Retención dest. (€)": dest_ret,
                        "Tipo activo": tipo_activo,
                        **fifo_sale_amount_cols(row, 1.0),
                    }
                )
            else:
                if qty_sell <= 0:
                    continue
                price_eur = float(total_eur) / qty_sell if qty_sell > 0 else 0.0
                isin_s = _fifo_resolve_isin_row(row, key_ticker, ticker_orig, cat_cache)
                lots_by_key[key].append(
                    {
                        "Broker": broker,
                        "Ticker": ticker_orig or key_ticker,
                        "Ticker_Yahoo": ticker_y,
                        "ISIN": isin_s,
                        "Nombre": nombre,
                        "Fecha origen": fecha,
                        "Cantidad": -qty_sell,
                        "Precio medio €": float(price_eur),
                        "Tipo activo": tipo_activo,
                    }
                )

        # -------- VENTAS / SWITCH SALIDA: consumen lotes FIFO (valores) --------
        elif tipo_lower in ["sell", "switch"] or (
            tipo_lower == "optionsell" and tipo_activo not in ("putoption", "calloption")
        ):
            qty_sell = abs(qty_f)
            remaining = qty_sell
            cost_hist = 0.0
            tranches: list[dict] = []

            lots = lots_by_key.get(key, [])
            while remaining > 0 and lots:
                lote = lots[0]
                lote_qty = lote["Cantidad"]
                if lote_qty <= remaining + 1e-8:
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
            isin_sale = _fifo_resolve_isin_row(row, key_ticker, ticker_orig, cat_cache)
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
                        "Origen FIFO": "Acciones/ETFs",
                        "Broker": broker,
                        "Ticker": ticker_orig or key_ticker,
                        "ISIN": isin_sale,
                        "Ticker_Yahoo": ticker_y,
                        "Nombre": nombre,
                        "Tipo activo": tipo_activo,
                        "Tipo movimiento": tipo_lower,
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
            sales_rows.append(
                {
                    "Broker": broker,
                    "Ticker": ticker_orig or key_ticker,
                    "ISIN": isin_sale,
                    "Ticker_Yahoo": ticker_y,
                    "Nombre": nombre,
                    "Fecha venta": fecha,
                    "Cantidad vendida": float(qty_sell),
                    "Valor compra histórico (€)": cost_hist,
                    "Valor venta (€)": vt,
                    "Plusvalía / Minusvalía (€)": plusvalia,
                    "Retención dest. (€)": dest_ret,
                    "Tipo activo": tipo_activo,
                    **fifo_sale_amount_cols(row, 1.0),
                }
            )

        # Otros tipos no afectan a los lotes en este contexto

    # Construimos DataFrames de salida
    lots_rows: list[dict] = []
    for key, lots in lots_by_key.items():
        for lote in lots:
            lots_rows.append(lote)

    lots_df = pd.DataFrame(lots_rows)
    sales_df = pd.DataFrame(sales_rows)
    sales_detail_df = pd.DataFrame(sales_detail_rows)
    return lots_df, sales_df, sales_detail_df
