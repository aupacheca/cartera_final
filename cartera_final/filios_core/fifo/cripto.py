"""FIFO fiscal global cripto y posiciones por broker (sin Streamlit)."""
from __future__ import annotations

import pandas as pd

from filios_core.config import DATA_DIR
from filios_core.constants import CRYPTO_BROKER_IDS, MIN_POSITION
from filios_core.fifo.sale_extras import fifo_sale_amount_cols
from filios_core.isin import _lookup_isin_for_ticker_yahoo, _norm_isin_field
from filios_core.util import safe_get as _safe_get, to_float as _to_float


def _cripto_chrono_type_order(s: pd.Series) -> pd.Series:
    """A igual timestamp: traspaso antes que comisión (coherente con compute_positions_criptos)."""
    t = s.astype(str).str.strip().str.lower()
    return t.map({"brokertransfer": 0, "commission": 1}).fillna(2)


def _consume_fifo_lotes_cripto_global(lots: list[dict], qty_to_consume: float) -> tuple[list[dict], float]:
    """
    Consume cantidad FIFO sobre la lista de lotes de compute_fifo_criptos (misma forma que sell).
    Devuelve (nueva_lista_de_lotes, coste_histórico_consumido_en_eur).
    """
    remaining = qty_to_consume
    cost_hist = 0.0
    new_lots: list[dict] = []

    for lote in lots:
        if remaining <= 0:
            new_lots.append(lote)
            continue
        lote_qty = lote["Cantidad"]
        if lote_qty <= 0:
            continue
        if lote_qty <= remaining + 1e-8:
            consumed = lote_qty
            remaining -= consumed
            cost_hist += consumed * lote["Precio medio €"]
        else:
            consumed = remaining
            remaining = 0.0
            cost_hist += consumed * lote["Precio medio €"]
            lote_rest = lote_qty - consumed
            new_lots.append(
                {
                    **lote,
                    "Cantidad": lote_rest,
                    "Coste histórico €": lote_rest * lote["Precio medio €"],
                }
            )

    return new_lots, cost_hist


def _consume_fifo_lotes_cripto_global_detail(lots: list[dict], qty_to_consume: float) -> tuple[list[dict], list[dict]]:
    """Como _consume_fifo_lotes_cripto_global pero devuelve tramos consumidos para informe FIFO."""
    remaining = qty_to_consume
    new_lots: list[dict] = []
    tranches: list[dict] = []

    for lote in lots:
        if remaining <= 0:
            new_lots.append(lote)
            continue
        lote_qty = lote["Cantidad"]
        if lote_qty <= 0:
            continue
        if lote_qty <= remaining + 1e-8:
            consumed = lote_qty
            remaining -= consumed
            tranches.append(
                {
                    "consumed": consumed,
                    "Fecha origen lote": lote.get("Fecha origen"),
                    "Precio medio €": float(lote["Precio medio €"]),
                }
            )
        else:
            consumed = remaining
            remaining = 0.0
            tranches.append(
                {
                    "consumed": consumed,
                    "Fecha origen lote": lote.get("Fecha origen"),
                    "Precio medio €": float(lote["Precio medio €"]),
                }
            )
            lote_rest = lote_qty - consumed
            new_lots.append(
                {
                    **lote,
                    "Cantidad": lote_rest,
                    "Coste histórico €": lote_rest * lote["Precio medio €"],
                }
            )

    return new_lots, tranches


def compute_fifo_criptos(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calcula lotes vivos y ventas (permuta incluida) para CRIPTOS usando FIFO GLOBAL por ticker.
    Reglas:
    - buy / switchBuy: crean lotes (cantidad y coste histórico en EUR).
    - sell / switch: consumen lotes FIFO global del ticker y generan plusvalía/minusvalía.
    - stakeReward: crea lotes con coste 0 (ganancia futura al vender).
    - brokerTransfer: neutro fiscalmente (se ignora para FIFO global).
    - commission: si la cantidad es en cripto (positionNumber > 0), consume FIFO global como la cartera
      (comisión en moneda del activo); no genera fila de venta. Comisión solo en EUR sin cantidad en
      cripto no altera lotes.
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if "datetime_full" in df.columns:
        d = df.copy()
        sc, sa = ["datetime_full"], [True]
        if "type" in d.columns:
            d["_tie_cf"] = _cripto_chrono_type_order(d["type"])
            sc.append("_tie_cf")
            sa.append(True)
        if "_rowid_" in d.columns:
            sc.append("_rowid_")
            sa.append(True)
        data = d.sort_values(sc, ascending=sa, kind="mergesort").drop(columns=["_tie_cf"], errors="ignore").copy()
    elif "date" in df.columns:
        data = df.sort_values("date").copy()
    else:
        data = df.copy()

    lots_by_ticker: dict[str, list[dict]] = {}
    sales_rows: list[dict] = []
    sales_detail_rows: list[dict] = []
    cripto_isin_cache: dict[str, str] = {}

    for _, row in data.iterrows():
        ticker_raw = str(row.get("ticker") or row.get("ticker_Yahoo") or "").strip()
        if not ticker_raw:
            continue
        ticker = ticker_raw.upper()
        if ticker.endswith("-EUR"):
            ticker = ticker[:-4]

        tipo = str(row.get("type") or "").strip().lower()
        if not tipo:
            continue

        qty_raw = pd.to_numeric(row.get("positionNumber"), errors="coerce")
        if pd.isna(qty_raw) or float(qty_raw) <= 0:
            continue
        qty = float(qty_raw)

        total_eur_col = (
            "totalWithComissionBaseCurrency"
            if "totalWithComissionBaseCurrency" in row.index
            else "totalBaseCurrency"
        )
        total_eur = pd.to_numeric(row.get(total_eur_col), errors="coerce")
        total_eur = float(total_eur) if not pd.isna(total_eur) else 0.0

        date_str = str(row.get("date") or "")
        broker = str(row.get("broker") or "")
        nombre = str(row.get("name") or ticker).strip()
        isin_c = _norm_isin_field(_safe_get(row, "isin")) or _norm_isin_field(_safe_get(row, "ISIN"))
        if not isin_c:
            if ticker not in cripto_isin_cache:
                cripto_isin_cache[ticker] = _lookup_isin_for_ticker_yahoo(ticker) or ""
            isin_c = cripto_isin_cache[ticker]

        if ticker not in lots_by_ticker:
            lots_by_ticker[ticker] = []

        # Compras (incluye permutas de entrada)
        if tipo in ("buy", "switchbuy"):
            cost_eur = total_eur
            price_eur = cost_eur / qty if qty > 0 else 0.0
            lots_by_ticker[ticker].append(
                {
                    "Broker": broker,
                    "Ticker": ticker,
                    "ISIN": isin_c,
                    "Nombre": nombre,
                    "Fecha origen": date_str,
                    "Cantidad": qty,
                    "Precio medio €": price_eur,
                    "Coste histórico €": cost_eur,
                    "Tipo activo": "crypto",
                }
            )
            continue

        # Recompensas de staking → lote con coste 0
        if tipo == "stakereward":
            lots_by_ticker[ticker].append(
                {
                    "Broker": broker,
                    "Ticker": ticker,
                    "ISIN": isin_c,
                    "Nombre": nombre,
                    "Fecha origen": date_str,
                    "Cantidad": qty,
                    "Precio medio €": 0.0,
                    "Coste histórico €": 0.0,
                    "Tipo activo": "crypto",
                }
            )
            continue

        # Traspasos entre wallets/cuentas: neutros fiscalmente (FIFO global)
        if tipo == "brokertransfer":
            continue

        # Comisión pagada en cripto: reduce lotes FIFO global (alineado con compute_positions_criptos)
        if tipo == "commission":
            lots = lots_by_ticker.get(ticker, [])
            new_lots, _ = _consume_fifo_lotes_cripto_global(lots, qty)
            lots_by_ticker[ticker] = new_lots
            continue

        # Ventas / permutas de salida: sell / switch
        if tipo in ("sell", "switch"):
            if total_eur == 0.0:
                # Sin total en EUR no podemos valorar la venta
                continue

            lots = lots_by_ticker.get(ticker, [])
            new_lots, tranches = _consume_fifo_lotes_cripto_global_detail(lots, qty)
            lots_by_ticker[ticker] = new_lots
            cost_hist = sum(t["consumed"] * t["Precio medio €"] for t in tranches)

            # Valor de transmisión fiscal: totalBase - comisión - impuestos (normativa española)
            total_base = _to_float(row.get("totalBaseCurrency"), 0.0)
            fx = _to_float(row.get("exchangeRate"), 1.0) or 1.0
            comm = _to_float(row.get("comission"), 0.0)
            tax = _to_float(row.get("taxes"), 0.0)
            comm_ccy = str(row.get("comissionCurrency") or "").strip().upper()
            tax_ccy = str(row.get("taxesCurrency") or "").strip().upper()
            comm_eur = comm if comm_ccy == "EUR" else (comm * fx if fx and abs(fx) > 1e-9 else comm)
            tax_eur = tax if tax_ccy == "EUR" else (tax * fx if fx and abs(fx) > 1e-9 else tax)
            valor_transmision = total_base - comm_eur - tax_eur
            vt = float(valor_transmision)

            dest_ret = _to_float(row.get("destinationRetentionBaseCurrency"), 0.0)
            sale_id = len(sales_rows)
            v_allocated = 0.0
            for i, t in enumerate(tranches):
                c_tramo = t["consumed"] * t["Precio medio €"]
                if i == len(tranches) - 1:
                    v_tramo = vt - v_allocated
                else:
                    v_tramo = vt * (t["consumed"] / qty) if qty > 1e-12 else 0.0
                    v_allocated += v_tramo
                sales_detail_rows.append(
                    {
                        "Venta #": sale_id,
                        "Origen FIFO": "Cripto",
                        "Broker": broker,
                        "Ticker": ticker,
                        "ISIN": isin_c,
                        "Ticker_Yahoo": ticker,
                        "Nombre": nombre,
                        "Tipo activo": "crypto",
                        "Tipo movimiento": tipo,
                        "Fecha venta": date_str,
                        "Cantidad venta (total)": qty,
                        "Cantidad (tramo)": t["consumed"],
                        "Fecha origen (lote)": t["Fecha origen lote"],
                        "Valor compra histórico (€)": c_tramo,
                        "Valor venta (€)": v_tramo,
                        "Plusvalía / Minusvalía (€)": v_tramo - c_tramo,
                    }
                )

            sales_rows.append(
                {
                    "Broker": broker,
                    "Ticker": ticker,
                    "ISIN": isin_c,
                    "Nombre": nombre,
                    "Fecha venta": date_str,
                    "Cantidad vendida": qty,
                    "Valor venta (€)": vt,
                    "Valor compra histórico (€)": cost_hist,
                    "Plusvalía / Minusvalía (€)": vt - cost_hist,
                    "Retención dest. (€)": dest_ret,
                    "Tipo activo": "crypto",
                    **fifo_sale_amount_cols(row, 1.0),
                }
            )

    # Construimos DataFrames de salida
    lots_rows: list[dict] = []
    for ticker, lots in lots_by_ticker.items():
        for lote in lots:
            if lote["Cantidad"] <= 0:
                continue
            lots_rows.append(lote)

    lots_df = pd.DataFrame(lots_rows)
    sales_df = pd.DataFrame(sales_rows)
    sales_detail_df = pd.DataFrame(sales_detail_rows)
    return lots_df, sales_df, sales_detail_df


def _consume_lots_fifo(lots: list[dict], qty_to_consume: float) -> float:
    """
    Consume qty_to_consume de los lotes en orden FIFO. Devuelve el coste consumido en EUR.
    Modifica lots in-place.
    """
    remaining = qty_to_consume
    cost_consumed = 0.0
    while remaining > MIN_POSITION and lots:
        lot = lots[0]
        lot_qty = lot["qty"]
        if lot_qty <= remaining + MIN_POSITION:
            cost_consumed += lot["cost_eur"]
            remaining -= lot_qty
            lots.pop(0)
        else:
            frac = remaining / lot_qty
            cost_consumed += lot["cost_eur"] * frac
            lot["qty"] -= remaining
            lot["cost_eur"] -= lot["cost_eur"] * frac
            remaining = 0
    return cost_consumed


def compute_positions_criptos(
    df: pd.DataFrame, *, use_kraken_ledger_override: bool = True
) -> pd.DataFrame:
    """
    Calcula posiciones de cripto por broker y ticker a partir de movimientos_criptos.
    Usa FIFO para ventas, switch y traspasos (como Filios).
    - buy / switchBuy: añade lote FIFO.
    - sell / switch: consume lotes FIFO.
    - brokerTransfer: mueve lotes FIFO de origen a destino.
    - commission: consume cantidad FIFO (sin coste).
    - stakeReward: añade lote con coste 0.

    use_kraken_ledger_override: si False (p. ej. snapshots históricos mensuales), solo FIFO sobre movimientos.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Broker", "Ticker", "Ticker_Yahoo", "Nombre", "Cantidad"])

    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "time" in df.columns:
        df["time"] = df["time"].fillna("00:00")
    df["datetime"] = pd.to_datetime(
        df.get("date", pd.NaT).astype(str) + " " + df.get("time", "00:00").astype(str),
        errors="coerce",
    )
    df["_order"] = (
        df.get("type", "")
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"brokertransfer": 0, "commission": 1})
        .fillna(2)
    )
    df = df.sort_values(["datetime", "_order"]).reset_index(drop=True)
    df = df.drop(columns=["_order"], errors="ignore")

    # positions[(broker, ticker)] = {"lots": [{"qty", "cost_eur"}, ...]} orden FIFO
    positions: dict[tuple[str, str], dict] = {}
    meta: dict[tuple[str, str], dict[str, str]] = {}

    def ensure(broker: str, ticker: str, row: pd.Series):
        key = (broker, ticker)
        if key not in positions:
            positions[key] = {"lots": []}
            meta[key] = {
                "Broker": broker,
                "Ticker": ticker,
                "Ticker_Yahoo": str(row.get("ticker_Yahoo") or f"{ticker}-EUR"),
                "Nombre": str(row.get("name") or ticker),
            }
        return key

    for _, row in df.iterrows():
        broker = str(row.get("broker", "") or "").strip()
        ticker = str(row.get("ticker", "") or "").strip().upper()
        if ticker.endswith("-EUR"):
            ticker = ticker[:-4]
        if not broker or not ticker:
            continue

        tipo = str(row.get("type", "") or "").strip().lower()
        qty = _to_float(row.get("positionNumber"), 0.0)
        total_eur = _to_float(
            row.get("totalWithComissionBaseCurrency")
            if "totalWithComissionBaseCurrency" in row
            else row.get("totalBaseCurrency", 0.0),
            0.0,
        )
        com_ccy = str(row.get("comissionCurrency", "") or "").strip().upper()
        com_val = _to_float(row.get("comission"), 0.0)
        dest_raw = str(row.get("brokerTransferNewBroker", "") or "").strip()
        dest = CRYPTO_BROKER_IDS.get(dest_raw, dest_raw if dest_raw else "")

        if tipo == "buy":
            # Comisión en misma moneda: reducir qty y coste proporcionalmente (la comisión no aumenta la base)
            qty_orig = qty
            if com_ccy == ticker and com_val > 0:
                qty = max(0.0, qty - com_val)
                # Coste solo sobre la cantidad neta recibida
                total_eur = total_eur * (qty / qty_orig) if qty_orig > 0 else 0.0
            if qty <= 0:
                continue
            key = ensure(broker, ticker, row)
            positions[key]["lots"].append({"qty": qty, "cost_eur": total_eur})
        elif tipo == "sell":
            key = ensure(broker, ticker, row)
            lots = positions[key]["lots"]
            if qty <= 0:
                continue
            _consume_lots_fifo(lots, min(qty, sum(l["qty"] for l in lots)))
        elif tipo == "switch":
            if com_ccy == ticker and com_val > 0:
                qty = max(0.0, qty - com_val)
            key = ensure(broker, ticker, row)
            lots = positions[key]["lots"]
            total_q = sum(l["qty"] for l in lots)
            if qty <= 0 or total_q <= 0:
                continue
            _consume_lots_fifo(lots, min(qty, total_q))
        elif tipo == "switchbuy":
            qty_orig = qty
            if com_ccy == ticker and com_val > 0:
                qty = max(0.0, qty - com_val)
                total_eur = total_eur * (qty / qty_orig) if qty_orig > 0 else 0.0
            if qty <= 0:
                continue
            key = ensure(broker, ticker, row)
            positions[key]["lots"].append({"qty": qty, "cost_eur": total_eur})
        elif tipo == "brokertransfer":
            if not dest:
                continue
            sk = ensure(broker, ticker, row)
            dk = ensure(dest, ticker, row)
            src_lots = positions[sk]["lots"]
            dst_lots = positions[dk]["lots"]
            total_src = sum(l["qty"] for l in src_lots)
            if qty <= 0 or total_src <= 0:
                continue
            transfer_qty = min(qty, total_src)
            cost_moved = _consume_lots_fifo(src_lots, transfer_qty)
            dst_lots.append({"qty": transfer_qty, "cost_eur": cost_moved})
        elif tipo == "commission":
            key = ensure(broker, ticker, row)
            lots = positions[key]["lots"]
            # Comisión: consume cantidad FIFO (solo reduce qty, el coste se pierde)
            total_q = sum(l["qty"] for l in lots)
            if qty > 0 and total_q > 0:
                _consume_lots_fifo(lots, min(qty, total_q))
        elif tipo == "stakereward":
            key = ensure(broker, ticker, row)
            if qty > 0:
                positions[key]["lots"].append({"qty": qty, "cost_eur": 0.0})

    if not use_kraken_ledger_override:
        rows_pos: list[dict] = []
        for key, pos in positions.items():
            if not isinstance(pos, dict):
                qty = float(pos)
                cost_eur = 0.0
            else:
                lots = pos.get("lots", [])
                qty = sum(l["qty"] for l in lots)
                cost_eur = sum(l["cost_eur"] for l in lots)
            if abs(qty) < MIN_POSITION:
                continue
            info = meta.get(key, {})
            rows_pos.append(
                {
                    "Broker": info.get("Broker", key[0]),
                    "Ticker": info.get("Ticker", key[1]),
                    "Ticker_Yahoo": info.get("Ticker_Yahoo", f"{key[1]}-EUR"),
                    "Nombre": info.get("Nombre", key[1]),
                    "Cantidad": float(qty),
                    "Inversion €": cost_eur,
                }
            )
        if not rows_pos:
            return pd.DataFrame(columns=["Broker", "Ticker", "Ticker_Yahoo", "Nombre", "Cantidad"])
        pos_df = pd.DataFrame(rows_pos)
        return pos_df.sort_values(["Broker", "Ticker"]).reset_index(drop=True)

    # Ajuste Kraken BTC con ledger oficial: solo mostrar si hay saldo > 0
    # Si no hay ledger o saldo 0, eliminar Kraken para no mostrar posiciones obsoletas
    ledger_path = DATA_DIR / "kraken_stocks_etfs_ledgers_2025-01-13-2025-12-31.csv"
    kraken_btc: float | None = None
    if ledger_path.exists():
        try:
            led = pd.read_csv(
                ledger_path,
                dtype={"asset": str, "aclass": str, "subclass": str, "wallet": str},
            )
            led["asset"] = led["asset"].astype(str)
            led["aclass"] = led["aclass"].astype(str)
            led["subclass"] = led["subclass"].astype(str)
            led["wallet"] = led["wallet"].astype(str)
            mask_btc = (
                led["asset"].str.upper().eq("BTC")
                & led["aclass"].str.lower().eq("currency")
                & led["subclass"].str.lower().eq("crypto")
            )
            btc_ledger = led.loc[mask_btc].copy()
            if "balance" in btc_ledger.columns:
                btc_ledger["balance_f"] = pd.to_numeric(
                    btc_ledger["balance"], errors="coerce"
                )
                last_balances = (
                    btc_ledger.sort_values("time")
                    .groupby("wallet")["balance_f"]
                    .last()
                    .fillna(0.0)
                )
                kraken_btc = float(last_balances.sum())
                if abs(kraken_btc) < 10 ** -8:
                    kraken_btc = 0.0
        except Exception:
            kraken_btc = None
    # Sin ledger o saldo 0: eliminar Kraken
    if kraken_btc is None or kraken_btc == 0.0:
        for k in [("Kraken", "BTC")]:
            positions.pop(k, None)
            meta.pop(k, None)
    else:
        # Saldo > 0: añadir o actualizar
        positions[("Kraken", "BTC")] = kraken_btc
        meta[("Kraken", "BTC")] = {
            "Broker": "Kraken",
            "Ticker": "BTC",
            "Ticker_Yahoo": "BTC-EUR",
            "Nombre": "Bitcoin",
        }

    # Construir DataFrame de posiciones abiertas (solo brokers con saldo > 0)
    rows_pos: list[dict] = []
    for key, pos in positions.items():
        # Kraken BTC puede ser float (ledger); resto usa lots FIFO
        if not isinstance(pos, dict):
            qty = float(pos)
            cost_eur = 0.0
        else:
            lots = pos.get("lots", [])
            qty = sum(l["qty"] for l in lots)
            cost_eur = sum(l["cost_eur"] for l in lots)
        if abs(qty) < MIN_POSITION:
            continue
        info = meta.get(key, {})
        rows_pos.append(
            {
                "Broker": info.get("Broker", key[0]),
                "Ticker": info.get("Ticker", key[1]),
                "Ticker_Yahoo": info.get("Ticker_Yahoo", f"{key[1]}-EUR"),
                "Nombre": info.get("Nombre", key[1]),
                "Cantidad": float(qty),
                "Inversion €": cost_eur,
            }
        )

    if not rows_pos:
        return pd.DataFrame(columns=["Broker", "Ticker", "Ticker_Yahoo", "Nombre", "Cantidad"])

    pos_df = pd.DataFrame(rows_pos)
    # Ordenar brokers con saldo (ocultará cuentas totalmente a 0, como Kraken)
    pos_df = pos_df.sort_values(["Broker", "Ticker"]).reset_index(drop=True)
    return pos_df
