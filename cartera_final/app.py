import hashlib
import json
import math
import os
import re
import sqlite3
import urllib.request
from datetime import datetime, time as dt_time
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


DECIMALS_POSITION = 8
MIN_POSITION = 10 ** -DECIMALS_POSITION

# Directorio de datos (Add-on HA: DATA_DIR, p. ej. /share/cartera_final; en local: directorio actual)
_DATA_DIR = Path(os.environ.get("DATA_DIR", ".")).resolve()

# Nombre del archivo de BD (run.sh del add-on exporta DB_FILENAME=acciones_final.db)
_DB_FILENAME = os.environ.get("DB_FILENAME", "acciones.db")

# Base de datos SQLite (fuente de verdad); CSV solo para exportar/backup
DB_PATH = str(_DATA_DIR / _DB_FILENAME)
CSV_PATH = str(_DATA_DIR / "acciones.csv")
FONDOS_CSV_PATH = str(_DATA_DIR / "fondos.csv")
MOVIMIENTOS_CRIPTOS_CSV_PATH = str(_DATA_DIR / "movimientos_criptos.csv")
DIVIDENDOS_CSV_PATH = str(_DATA_DIR / "dividendos.csv")
COTIZACIONES_CACHE_PATH = str(_DATA_DIR / "cartera_cotizaciones_cache.pkl")
COTIZACIONES_META_PATH = str(_DATA_DIR / "cartera_cotizaciones_meta.json")
PRECIOS_MANUALES_PATH = str(_DATA_DIR / "precios_manuales.json")
CSV_ENCODING = "latin-1"
CSV_DECIMAL = ","
CSV_SEP = ","

# Columnas del CSV de dividendos (Filios); se guardan en tabla dividendos
DIVIDENDOS_COLUMNS = [
    "type", "date", "time", "ticker", "ticker_Yahoo", "nombre", "positionType", "positionCountry",
    "positionCurrency", "positionExchange", "broker", "positionNumber", "currency", "quantity",
    "quantityCurrency", "comission", "comissionCurrency", "exchangeRate", "comissionBaseCurrency",
    "autoFx", "total", "totalBaseCurrency", "originRetention", "neto", "netoBaseCurrency",
    "destinationRetentionBaseCurrency", "totalNeto", "totalNetoBaseCurrency", "retentionReturned",
    "retentionReturnedBaseCurrency", "unrealizedDestinationRetentionBaseCurrency",
    "netoWithReturnBaseCurrency", "originRetentionLossBaseCurrency", "description",
]

# Columnas de la tabla movimientos (sin datetime_full, que se calcula al cargar)
MOVIMIENTOS_COLUMNS = [
    "date", "time", "ticker", "ticker_Yahoo", "name", "positionType", "positionCountry",
    "positionCurrency", "positionExchange", "broker", "type", "positionNumber", "price",
    "comission", "comissionCurrency", "destinationRetentionBaseCurrency", "taxes", "taxesCurrency",
    "exchangeRate", "positionQuantity", "autoFx", "switchBuyPosition", "switchBuyPositionType",
    "switchBuyPositionNumber", "switchBuyExchangeRate", "switchBuyBroker", "spinOffBuyPosition",
    "spinOffBuyPositionNumber", "spinOffBuyPositionAllocation", "brokerTransferNewBroker",
    "total", "totalBaseCurrency", "totalWithComission", "totalWithComissionBaseCurrency",
]

# Columnas de la tabla movimientos_criptos (en acciones.db, migradas desde criptos.csv)
MOVIMIENTOS_CRIPTOS_COLUMNS = [
    "date", "time", "ticker", "ticker_Yahoo", "name", "positionType", "positionCountry",
    "positionCurrency", "positionExchange", "broker", "type", "positionNumber", "price",
    "comission", "comissionCurrency", "destinationRetentionBaseCurrency", "taxes", "taxesCurrency",
    "exchangeRate", "positionQuantity", "autoFx", "switchBuyPosition", "switchBuyPositionType",
    "switchBuyPositionNumber", "switchBuyExchangeRate", "switchBuyBroker", "spinOffBuyPosition",
    "spinOffBuyPositionNumber", "spinOffBuyPositionAllocation", "brokerTransferNewBroker",
    "total", "totalBaseCurrency", "totalWithComission", "totalWithComissionBaseCurrency",
    "positionCustomType", "description",
]

# Mapa de IDs a brokers para traspasos de cripto (wallet IDs → nombre)
CRYPTO_BROKER_IDS = {
    "67b242abada74321db44e91b": "Binance",
    "67c8ac4deb09ee2b1a4121d3": "Tangem",
}

# Nombres legibles para criptos (normaliza cuando name == ticker_Yahoo, ej. ETH-EUR → Ethereum)
CRYPTO_TICKER_NAMES = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "XRP": "Ripple", "SOL": "Solana",
    "BNB": "Binance", "TRX": "Tron", "AVAX": "Avalanche", "HBAR": "Hedera",
    "ADA": "Cardano", "DOT": "Polkadot", "LINK": "Chainlink", "MATIC": "Polygon",
    "DOGE": "Dogecoin", "UNI": "Uniswap", "ATOM": "Cosmos", "LTC": "Litecoin",
}


def _get_data_mount_source() -> str | None:
    """Intenta obtener la ruta del host donde está montado DATA_DIR (add-on Home Assistant)."""
    try:
        mountinfo = Path("/proc/self/mountinfo")
        if not mountinfo.exists():
            return None
        for mount_point in ["/config", "/data"]:
            for line in mountinfo.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[4] == mount_point:
                    try:
                        idx = parts.index("-", 5)
                        if idx + 2 < len(parts):
                            return parts[idx + 2]
                    except ValueError:
                        pass
    except Exception:
        pass
    return None


def _get_db():
    return sqlite3.connect(DB_PATH)


def _init_db():
    """Crea la tabla movimientos si no existe."""
    cols_sql = ", ".join(f'"{c}" TEXT' for c in MOVIMIENTOS_COLUMNS)
    with _get_db() as conn:
        conn.execute(f"CREATE TABLE IF NOT EXISTS movimientos ({cols_sql})")


def _migrate_csv_to_db():
    """Una sola vez: lee acciones.csv y vuelca los datos en SQLite."""
    if not Path(CSV_PATH).exists():
        return
    with _get_db() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM movimientos")
        if cur.fetchone()[0] > 0:
            return  # ya migrado
    df = pd.read_csv(CSV_PATH, decimal=CSV_DECIMAL, sep=CSV_SEP, encoding=CSV_ENCODING, parse_dates=False, dtype={"date": str, "time": str})
    cols = [c for c in MOVIMIENTOS_COLUMNS if c in df.columns]
    if not cols:
        return
    df = df[cols].copy()
    for col in ("date", "time"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "date" in df.columns:
        df["date"] = df["date"].astype(str).str.split("T").str[0].str.strip()
    if "time" in df.columns:
        df["time"] = df["time"].astype(str).str.strip().apply(_normalize_time_to_24h)
    placeholders = ", ".join("?" for _ in MOVIMIENTOS_COLUMNS)
    with _get_db() as conn:
        for _, row in df.iterrows():
            vals = [_row_to_db_val(row.get(c, "")) for c in MOVIMIENTOS_COLUMNS]
            conn.execute(
                f'INSERT INTO movimientos ({", ".join(MOVIMIENTOS_COLUMNS)}) VALUES ({placeholders})',
                vals,
            )
        conn.commit()


def _init_db_brokers():
    """Crea la tabla brokers si no existe (id, name, country, multidivisa, retiene_en_destino)."""
    with _get_db() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS brokers (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL)"
        )
        # Añadir columnas de ficha de cuenta si no existen
        cur = conn.execute("PRAGMA table_info(brokers)")
        cols = [row[1] for row in cur.fetchall()]
        if "country" not in cols:
            conn.execute("ALTER TABLE brokers ADD COLUMN country TEXT")
        if "multidivisa" not in cols:
            conn.execute("ALTER TABLE brokers ADD COLUMN multidivisa INTEGER DEFAULT 0")
        if "retiene_en_destino" not in cols:
            conn.execute("ALTER TABLE brokers ADD COLUMN retiene_en_destino INTEGER DEFAULT 0")
        conn.commit()


def _migrate_brokers_from_data():
    """Rellena brokers con los nombres distintos de movimientos y movimientos_fondos (INSERT OR IGNORE)."""
    with _get_db() as conn:
        names = set()
        for table in ("movimientos", "movimientos_fondos"):
            try:
                cur = conn.execute(f'SELECT DISTINCT broker FROM {table} WHERE broker IS NOT NULL AND trim(broker) != ""')
                for (b,) in cur.fetchall():
                    if b and str(b).strip():
                        names.add(str(b).strip())
            except sqlite3.OperationalError:
                pass
        for n in sorted(names):
            try:
                conn.execute("INSERT OR IGNORE INTO brokers (name) VALUES (?)", (n,))
            except Exception:
                pass
        conn.commit()


def get_brokers_list() -> list[str]:
    """Devuelve la lista de brokers (tabla brokers), inicializando y migrando si hace falta."""
    _init_db_brokers()
    _migrate_brokers_from_data()
    with _get_db() as conn:
        cur = conn.execute("SELECT name FROM brokers ORDER BY name")
        return [r[0] for r in cur.fetchall()]


def get_brokers_with_details() -> list[dict]:
    """Devuelve lista de cuentas con id, name, country, multidivisa, retiene_en_destino para la ficha."""
    _init_db_brokers()
    _migrate_brokers_from_data()
    with _get_db() as conn:
        cur = conn.execute(
            "SELECT id, name, COALESCE(country, ''), COALESCE(multidivisa, 0), COALESCE(retiene_en_destino, 0) FROM brokers ORDER BY name"
        )
        return [
            {"id": r[0], "name": r[1], "country": r[2] or "", "multidivisa": bool(r[3]), "retiene_en_destino": bool(r[4])}
            for r in cur.fetchall()
        ]


def add_broker(name: str) -> tuple[bool, str]:
    """Añade un broker. Devuelve (éxito, mensaje)."""
    n = (name or "").strip()
    if not n:
        return False, "El nombre no puede estar vacío."
    _init_db_brokers()
    try:
        with _get_db() as conn:
            conn.execute("INSERT INTO brokers (name) VALUES (?)", (n,))
            conn.commit()
        return True, f"Broker «{n}» añadido."
    except sqlite3.IntegrityError:
        return False, f"Ya existe un broker con el nombre «{n}»."


def rename_broker(old_name: str, new_name: str) -> tuple[bool, str]:
    """Renombra un broker en la tabla brokers y en movimientos y movimientos_fondos."""
    old_n = (old_name or "").strip()
    new_n = (new_name or "").strip()
    if not old_n or not new_n:
        return False, "Nombres no válidos."
    if old_n == new_n:
        return True, "Sin cambios."
    _init_db_brokers()
    try:
        with _get_db() as conn:
            conn.execute("UPDATE movimientos SET broker = ? WHERE broker = ?", (new_n, old_n))
            conn.execute("UPDATE movimientos_fondos SET broker = ? WHERE broker = ?", (new_n, old_n))
            conn.execute("UPDATE brokers SET name = ? WHERE name = ?", (new_n, old_n))
            conn.commit()
        load_data.clear()
        load_data_fondos.clear()
        return True, f"Broker renombrado a «{new_n}». Actualizados movimientos y fondos."
    except sqlite3.IntegrityError:
        return False, f"Ya existe un broker con el nombre «{new_n}»."


def get_broker_by_id(broker_id: int) -> dict | None:
    """Devuelve {id, name, country, multidivisa, retiene_en_destino} o None."""
    _init_db_brokers()
    with _get_db() as conn:
        cur = conn.execute(
            "SELECT id, name, COALESCE(country, ''), COALESCE(multidivisa, 0), COALESCE(retiene_en_destino, 0) FROM brokers WHERE id = ?",
            (broker_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {"id": row[0], "name": row[1], "country": row[2] or "", "multidivisa": bool(row[3]), "retiene_en_destino": bool(row[4])}


def get_broker_retiene_en_destino(broker_name: str) -> bool:
    """Indica si la cuenta tiene activado 'Retiene en destino' (para prefijar formulario dividendos)."""
    if not (broker_name or "").strip():
        return False
    _init_db_brokers()
    with _get_db() as conn:
        cur = conn.execute("SELECT retiene_en_destino FROM brokers WHERE name = ?", (broker_name.strip(),))
        row = cur.fetchone()
    return bool(row and row[0])


def update_broker_account(broker_id: int, name: str, country: str = "", multidivisa: bool = False, retiene_en_destino: bool = False) -> tuple[bool, str]:
    """Actualiza la ficha de la cuenta (nombre, país, toggles). Si cambia el nombre, actualiza movimientos/fondos/dividendos."""
    n = (name or "").strip()
    if not n:
        return False, "El nombre de la cuenta no puede estar vacío."
    _init_db_brokers()
    with _get_db() as conn:
        cur = conn.execute("SELECT name FROM brokers WHERE id = ?", (broker_id,))
        row = cur.fetchone()
        if not row:
            return False, "Cuenta no encontrada."
        old_name = row[0]
        if old_name != n:
            try:
                conn.execute("UPDATE movimientos SET broker = ? WHERE broker = ?", (n, old_name))
                conn.execute("UPDATE movimientos_fondos SET broker = ? WHERE broker = ?", (n, old_name))
                try:
                    conn.execute("UPDATE dividendos SET broker = ? WHERE broker = ?", (n, old_name))
                except sqlite3.OperationalError:
                    pass
                conn.execute(
                    "UPDATE brokers SET name = ?, country = ?, multidivisa = ?, retiene_en_destino = ? WHERE id = ?",
                    (n, (country or "").strip(), 1 if multidivisa else 0, 1 if retiene_en_destino else 0, broker_id),
                )
            except sqlite3.IntegrityError:
                return False, f"Ya existe una cuenta con el nombre «{n}»."
        else:
            conn.execute(
                "UPDATE brokers SET country = ?, multidivisa = ?, retiene_en_destino = ? WHERE id = ?",
                ((country or "").strip(), 1 if multidivisa else 0, 1 if retiene_en_destino else 0, broker_id),
            )
        conn.commit()
    load_data.clear()
    load_data_fondos.clear()
    return True, "Cuenta guardada."


def delete_broker(broker_id: int) -> tuple[bool, str]:
    """Elimina la cuenta de la tabla brokers. Los movimientos/fondos/dividendos conservan el nombre como texto."""
    _init_db_brokers()
    with _get_db() as conn:
        cur = conn.execute("SELECT name FROM brokers WHERE id = ?", (broker_id,))
        row = cur.fetchone()
        if not row:
            return False, "Cuenta no encontrada."
        conn.execute("DELETE FROM brokers WHERE id = ?", (broker_id,))
        conn.commit()
    return True, "Cuenta eliminada. Los movimientos existentes siguen mostrando ese nombre."


def _init_db_dividendos():
    """Crea la tabla dividendos si no existe (columnas como export Filios)."""
    cols_sql = ", ".join(f'"{c}" TEXT' for c in DIVIDENDOS_COLUMNS)
    with _get_db() as conn:
        conn.execute(f"CREATE TABLE IF NOT EXISTS dividendos ({cols_sql})")


@st.cache_data
def load_dividendos() -> pd.DataFrame:
    """Carga todos los dividendos desde la tabla dividendos (migra desde dividendos.csv si existe y tabla vacía)."""
    _init_db_dividendos()
    _migrate_dividendos_csv_to_db()
    with _get_db() as conn:
        df = pd.read_sql("SELECT rowid AS _rowid_, * FROM dividendos", conn)
    if df.empty:
        return pd.DataFrame(columns=["_rowid_"] + DIVIDENDOS_COLUMNS)
    if "date" in df.columns:
        df["date"] = df["date"].astype(str).str.split("T").str[0].str.strip()
    if "time" in df.columns:
        df["time"] = df["time"].astype(str).str.strip().apply(_normalize_time_to_24h)
    dt_str = df["date"].astype(str).str.strip() + " " + df["time"]
    df["datetime_full"] = pd.to_datetime(dt_str, format="mixed", errors="coerce")
    df = df.sort_values("datetime_full", ascending=False).reset_index(drop=True)
    return df


def update_dividendo(rowid: int, row_dict: dict) -> None:
    """Actualiza un dividendo por rowid."""
    _init_db_dividendos()
    update_cols = [c for c in DIVIDENDOS_COLUMNS if c in row_dict]
    if not update_cols:
        return
    sets = ", ".join(f'"{c}" = ?' for c in update_cols)
    vals = [_row_to_db_val(row_dict.get(c, "")) for c in update_cols]
    with _get_db() as conn:
        conn.execute(f"UPDATE dividendos SET {sets} WHERE rowid = ?", vals + [rowid])
        conn.commit()
    if hasattr(load_dividendos, "clear"):
        load_dividendos.clear()


def delete_dividendos_by_rowids(rowids: list[int]) -> int:
    """Elimina dividendos por sus rowids. Devuelve el número de filas eliminadas."""
    if not rowids:
        return 0
    _init_db_dividendos()
    with _get_db() as conn:
        n = 0
        for rid in rowids:
            cur = conn.execute("DELETE FROM dividendos WHERE rowid = ?", (rid,))
            n += cur.rowcount
        conn.commit()
    if hasattr(load_dividendos, "clear"):
        load_dividendos.clear()
    return n


def append_dividendo(row: dict) -> None:
    """Inserta un registro en la tabla dividendos."""
    _init_db_dividendos()
    cols = [c for c in DIVIDENDOS_COLUMNS if c in row]
    if not cols:
        return
    placeholders = ", ".join("?" for _ in DIVIDENDOS_COLUMNS)
    vals = [_row_to_db_val(row.get(c, "")) for c in DIVIDENDOS_COLUMNS]
    with _get_db() as conn:
        conn.execute(
            f'INSERT INTO dividendos ({", ".join(DIVIDENDOS_COLUMNS)}) VALUES ({placeholders})',
            vals,
        )
        conn.commit()


def sync_dividendos_from_filios_csv(filios_path: str | Path | None = None) -> tuple[bool, str]:
    """
    Reemplaza la tabla dividendos con los datos de un CSV exportado desde Filios.
    Si filios_path es None, usa dividendos_filios.csv en el directorio del proyecto.
    Devuelve (éxito, mensaje).
    """
    path = Path(filios_path) if filios_path else Path(_DATA_DIR) / "dividendos_filios.csv"
    if not path.exists():
        return False, f"No existe el archivo: {path}"
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as e:
        return False, f"Error leyendo CSV: {e}"
    cols = [c for c in DIVIDENDOS_COLUMNS if c in df.columns]
    if not cols:
        return False, "El CSV no tiene las columnas esperadas de dividendos."
    df = df[[c for c in DIVIDENDOS_COLUMNS if c in df.columns]].copy()
    if "ticker_Yahoo" not in df.columns:
        df["ticker_Yahoo"] = df.get("ticker", pd.Series([""] * len(df)))
    if "nombre" not in df.columns:
        df["nombre"] = df.get("ticker", pd.Series([""] * len(df)))
    for c in DIVIDENDOS_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    for c in DIVIDENDOS_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[DIVIDENDOS_COLUMNS]
    if "date" in df.columns:
        df["date"] = df["date"].astype(str).str.strip().str.split("T").str[0].str.strip()
    if "time" in df.columns:
        df["time"] = df["time"].astype(str).str.strip().apply(_normalize_time_to_24h)
    _init_db_dividendos()
    placeholders = ", ".join("?" for _ in DIVIDENDOS_COLUMNS)
    with _get_db() as conn:
        conn.execute("DELETE FROM dividendos")
        for _, row in df.iterrows():
            vals = [_row_to_db_val(row.get(c, "")) for c in DIVIDENDOS_COLUMNS]
            conn.execute(
                f'INSERT INTO dividendos ({", ".join(DIVIDENDOS_COLUMNS)}) VALUES ({placeholders})',
                vals,
            )
        conn.commit()
    if hasattr(load_dividendos, "clear"):
        load_dividendos.clear()
    return True, f"Sincronizados {len(df)} dividendos desde {path.name}."


def _migrate_dividendos_csv_to_db():
    """Una sola vez: lee dividendos.csv (export Filios) y vuelca en la tabla dividendos."""
    if not Path(DIVIDENDOS_CSV_PATH).exists():
        return
    _init_db_dividendos()
    with _get_db() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM dividendos")
        if cur.fetchone()[0] > 0:
            return  # ya migrado
    df = pd.read_csv(
        DIVIDENDOS_CSV_PATH,
        decimal=CSV_DECIMAL,
        sep=CSV_SEP,
        encoding=CSV_ENCODING,
        dtype=str,
        keep_default_na=False,
    )
    cols = [c for c in DIVIDENDOS_COLUMNS if c in df.columns]
    if not cols:
        return
    df = df[[c for c in DIVIDENDOS_COLUMNS if c in df.columns]].copy()
    for c in DIVIDENDOS_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[DIVIDENDOS_COLUMNS]
    if "date" in df.columns:
        df["date"] = df["date"].astype(str).str.strip().str.split("T").str[0].str.strip()
    if "time" in df.columns:
        df["time"] = df["time"].astype(str).str.strip().apply(_normalize_time_to_24h)
    placeholders = ", ".join("?" for _ in DIVIDENDOS_COLUMNS)
    with _get_db() as conn:
        for _, row in df.iterrows():
            vals = [_row_to_db_val(row.get(c, "")) for c in DIVIDENDOS_COLUMNS]
            conn.execute(
                f'INSERT INTO dividendos ({", ".join(DIVIDENDOS_COLUMNS)}) VALUES ({placeholders})',
                vals,
            )
        conn.commit()


def _init_db_fondos():
    """Crea la tabla movimientos_fondos si no existe (mismo esquema que movimientos)."""
    cols_sql = ", ".join(f'"{c}" TEXT' for c in MOVIMIENTOS_COLUMNS)
    with _get_db() as conn:
        conn.execute(f"CREATE TABLE IF NOT EXISTS movimientos_fondos ({cols_sql})")


def _migrate_fondos_csv_to_db():
    """Una sola vez: lee fondos.csv y vuelca en movimientos_fondos (nombre -> name)."""
    if not Path(FONDOS_CSV_PATH).exists():
        return
    with _get_db() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM movimientos_fondos")
        if cur.fetchone()[0] > 0:
            return
    df = pd.read_csv(FONDOS_CSV_PATH, sep=CSV_SEP, encoding=CSV_ENCODING, dtype=str, keep_default_na=False)
    for col in ["positionNumber", "price", "total", "totalBaseCurrency", "totalWithComission", "totalWithComissionBaseCurrency", "comission", "taxes", "exchangeRate"]:
        if col in df.columns:
            s = df[col].astype(str).str.strip().str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(s, errors="coerce")
    if "nombre" in df.columns:
        df["name"] = df["nombre"].astype(str).str.strip()
    cols = [c for c in MOVIMIENTOS_COLUMNS if c in df.columns]
    if not cols:
        return
    if "date" in df.columns:
        df["date"] = df["date"].astype(str).str.split("T").str[0].str.strip()
    if "time" in df.columns:
        df["time"] = df["time"].astype(str).str.strip().apply(_normalize_time_to_24h)
    placeholders = ", ".join("?" for _ in MOVIMIENTOS_COLUMNS)
    with _get_db() as conn:
        for _, row in df.iterrows():
            vals = []
            for c in MOVIMIENTOS_COLUMNS:
                v = row.get(c, row.get("nombre", "") if c == "name" else "")
                vals.append(_row_to_db_val(v))
            conn.execute(
                f'INSERT INTO movimientos_fondos ({", ".join(MOVIMIENTOS_COLUMNS)}) VALUES ({placeholders})',
                vals,
            )
        conn.commit()


@st.cache_data
def load_data_fondos() -> pd.DataFrame:
    """
    Carga movimientos de fondos desde movimientos_fondos (migra desde fondos.csv si existe y tabla vacía).
    Devuelve DataFrame con misma estructura que load_data() (datetime_full, name, etc.).
    """
    _init_db_fondos()
    _migrate_fondos_csv_to_db()
    with _get_db() as conn:
        df = pd.read_sql("SELECT rowid AS _rowid_, * FROM movimientos_fondos", conn)
    if df.empty:
        return pd.DataFrame(columns=["_rowid_"] + MOVIMIENTOS_COLUMNS)
    if "date" in df.columns:
        df["date"] = df["date"].astype(str).str.split("T").str[0].str.strip()
    if {"date", "time"}.issubset(df.columns):
        df["time"] = df["time"].astype(str).str.strip().apply(_normalize_time_to_24h)
        dt_str = df["date"].astype(str).str.strip() + " " + df["time"]
        df["datetime_full"] = pd.to_datetime(dt_str, format="mixed", errors="coerce")
    else:
        df["datetime_full"] = pd.Series(pd.RangeIndex(len(df)), index=df.index)
    _order = df["type"].astype(str).str.strip().str.lower().map({"switch": 0, "switchbuy": 1})
    df["_type_order"] = _order.fillna(2)
    df = df.reset_index().sort_values(["datetime_full", "_type_order", "index"]).drop(columns=["index", "_type_order"], errors="ignore").reset_index(drop=True)
    for col in ["positionNumber", "price", "totalWithComissionBaseCurrency", "totalBaseCurrency", "total", "exchangeRate", "comission", "taxes"]:
        if col in df.columns:
            s = df[col].astype(str).str.strip().str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(s, errors="coerce")
    return df


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
            lots_by_key[key].append({"broker": broker, "ticker": ticker, "nombre": nombre, "cantidad": float(qty), "precio_medio_eur": float(price_eur), "coste_total_eur": float(total_eur), "fecha": fecha})
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
                lots_by_key[key].append({"broker": broker, "ticker": ticker, "nombre": nombre, "cantidad": float(qty), "precio_medio_eur": float(price_eur), "coste_total_eur": float(cost_eur), "fecha": fecha_origen})
            elif total_eur is not None:
                price_eur = total_eur / qty if qty > 0 else 0.0
                lots_by_key[key].append({"broker": broker, "ticker": ticker, "nombre": nombre, "cantidad": float(qty), "precio_medio_eur": float(price_eur), "coste_total_eur": float(total_eur), "fecha": fecha})
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
        resumen.append({"broker": broker, "ticker": ticker, "nombre": nombre, "cantidad": total_cant, "coste_total_eur": total_coste, "precio_medio_eur": total_coste / total_cant if total_cant > 0 else 0, "fecha_origen": fecha_origen})
    return resumen


def compute_fifo_fondos(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    FIFO para fondos: lotes vivas y ventas con plusvalía/minusvalía.
    Traspasos (switch→switchBuy) no generan venta fiscal; solo sell genera plusvalía/minusvalía.
    """
    data = df.copy()
    lots_by_key: dict[tuple[str, str], list[dict]] = {}
    pending_traspasos: list[dict] = []
    sales_rows: list[dict] = []

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
        key = (broker, ticker)
        if key not in lots_by_key:
            lots_by_key[key] = []

        if tipo == "buy":
            if total_eur is None:
                continue
            price_eur = total_eur / qty if qty > 0 else 0.0
            lots_by_key[key].append({
                "Broker": broker,
                "Ticker": ticker,
                "Ticker_Yahoo": ticker_y,
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
            pending_traspasos.append({"broker": broker, "dest_ticker": dest_ticker, "cost_eur": cost_trasladado, "fecha_origen": fecha_origen})
            continue
        if tipo == "switchbuy":
            ticker_s = str(ticker or "").strip()
            ticker_yahoo_s = str(_safe_get(row, "ticker_Yahoo") or "").strip()
            match_idx = None
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
                lots_by_key[key].append({
                    "Broker": broker,
                    "Ticker": ticker,
                    "Ticker_Yahoo": ticker_y,
                    "Nombre": nombre,
                    "Fecha origen": fecha_origen,
                    "Cantidad": float(qty),
                    "Precio medio €": float(price_eur),
                    "Tipo activo": "fund",
                })
            elif total_eur is not None:
                price_eur = total_eur / qty if qty > 0 else 0.0
                lots_by_key[key].append({
                    "Broker": broker,
                    "Ticker": ticker,
                    "Ticker_Yahoo": ticker_y,
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
            remaining = float(qty)
            cost_hist = 0.0
            lots = lots_by_key.get(key, [])
            while remaining > MIN_QTY_FONDOS and lots:
                lote = lots[0]
                lote_qty = lote["Cantidad"]
                if lote_qty <= remaining + MIN_QTY_FONDOS:
                    consumed = lote_qty
                    cost_hist += consumed * lote["Precio medio €"]
                    remaining -= consumed
                    lots.pop(0)
                else:
                    consumed = remaining
                    cost_hist += consumed * lote["Precio medio €"]
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

            dest_ret = _to_float(_safe_get(row, "destinationRetentionBaseCurrency"), 0.0)
            plusvalia = float(valor_transmision) - cost_hist
            sales_rows.append({
                "Broker": broker,
                "Ticker": ticker,
                "Ticker_Yahoo": ticker_y,
                "Nombre": nombre,
                "Fecha venta": fecha,
                "Cantidad vendida": float(qty),
                "Valor compra histórico (€)": cost_hist,
                "Valor venta (€)": float(valor_transmision),
                "Plusvalía / Minusvalía (€)": plusvalia,
                "Retención dest. (€)": dest_ret,
                "Tipo activo": "fund",
            })
            continue

    lots_rows = []
    for key, lots in lots_by_key.items():
        for lote in lots:
            if lote["Cantidad"] > MIN_QTY_FONDOS:
                lots_rows.append(lote)
    lots_df = pd.DataFrame(lots_rows)
    sales_df = pd.DataFrame(sales_rows)
    return lots_df, sales_df


def positions_fondos_to_dataframe(resumen: list[dict]) -> pd.DataFrame:
    """Convierte el resumen de compute_positions_fondos a DataFrame con columnas como positions (Cartera)."""
    if not resumen:
        return pd.DataFrame(columns=["Broker", "Ticker", "Ticker_Yahoo", "Nombre", "Titulos", "Precio Medio €", "Inversion €", "Moneda Activo", "Tipo activo", "Fecha origen"])
    rows = []
    for r in resumen:
        rows.append({
            "Broker": r["broker"],
            "Ticker": r["ticker"],
            "Ticker_Yahoo": r.get("ticker_yahoo") or r["ticker"],
            "Nombre": r["nombre"],
            "Titulos": r["cantidad"],
            "Precio Medio Moneda": r["precio_medio_eur"],
            "Precio Medio €": r["precio_medio_eur"],
            "Inversion €": r["coste_total_eur"],
            "Moneda Activo": "EUR",
            "Tipo activo": "fund",
            "Fecha origen": r.get("fecha_origen", ""),
        })
    return pd.DataFrame(rows)


def get_ticker_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Catálogo único de tickers: ticker, ticker_Yahoo, name, positionCurrency, positionExchange, positionCountry, positionType."""
    req = ["ticker_Yahoo", "ticker", "name", "positionCurrency", "positionExchange", "positionCountry", "positionType"]
    if not all(c in df.columns for c in req):
        return pd.DataFrame(columns=req)
    return (
        df.drop_duplicates(subset=["ticker_Yahoo"], keep="first")[req]
        .sort_values("ticker_Yahoo")
        .reset_index(drop=True)
    )


def get_ticker_catalog_criptos(df: pd.DataFrame) -> pd.DataFrame:
    """Catálogo único de tickers para criptos (usa ticker como fallback de ticker_Yahoo)."""
    req = ["ticker_Yahoo", "ticker", "name", "positionCurrency", "positionExchange", "positionCountry", "positionType"]
    if df is None or df.empty:
        return pd.DataFrame(columns=req)
    df_cat = df.copy()
    for c in req:
        if c not in df_cat.columns:
            df_cat[c] = "" if c != "positionCurrency" else "EUR"
    df_cat["ticker_Yahoo"] = df_cat["ticker_Yahoo"].fillna("").astype(str).str.strip()
    mask_empty = df_cat["ticker_Yahoo"] == ""
    df_cat.loc[mask_empty, "ticker_Yahoo"] = df_cat.loc[mask_empty, "ticker"].astype(str).str.strip()
    df_cat = df_cat[df_cat["ticker_Yahoo"] != ""]
    if df_cat.empty:
        return pd.DataFrame(columns=req)
    return df_cat.drop_duplicates(subset=["ticker_Yahoo"], keep="first")[req].sort_values("ticker_Yahoo").reset_index(drop=True)


def _init_instrument_catalog() -> None:
    """Tabla ISIN y metadatos globales por ticker_Yahoo (clave única de cotización)."""
    with _get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS instrument_catalog (
                ticker_Yahoo TEXT PRIMARY KEY,
                isin TEXT
            )
            """
        )
        conn.commit()


def get_universe_instruments_table() -> pd.DataFrame:
    """
    Instrumentos distintos por ticker_Yahoo en acciones, fondos y criptos, con ISIN desde instrument_catalog.
    """
    _init_instrument_catalog()
    by_yahoo: dict[str, dict] = {}

    def _feed(df: pd.DataFrame | None, label: str) -> None:
        if df is None or df.empty:
            return
        cat = get_ticker_catalog_criptos(df) if label == "Criptos" else get_ticker_catalog(df)
        if cat.empty:
            return
        for _, r in cat.iterrows():
            y = str(r.get("ticker_Yahoo") or "").strip()
            if not y:
                continue
            if y not in by_yahoo:
                by_yahoo[y] = {"ticker": "", "name": "", "origenes": set()}
            o = by_yahoo[y]
            o["origenes"].add(label)
            t = str(r.get("ticker") or "").strip()
            n = str(r.get("name") or "").strip()
            if t and not o["ticker"]:
                o["ticker"] = t
            if n and not o["name"]:
                o["name"] = n

    _feed(load_data(), "Acciones")
    _feed(load_data_fondos(), "Fondos")
    _feed(load_data_criptos(), "Criptos")

    with _get_db() as conn:
        cur = conn.execute("SELECT ticker_Yahoo, isin FROM instrument_catalog")
        isin_map = {str(a or "").strip(): str(b or "").strip() for a, b in cur.fetchall()}

    rows = []
    for y in sorted(by_yahoo.keys()):
        o = by_yahoo[y]
        rows.append({
            "ticker_Yahoo": y,
            "ticker": o["ticker"],
            "name": o["name"],
            "ISIN": isin_map.get(y, ""),
            "Origen": ", ".join(sorted(o["origenes"])),
        })
    return pd.DataFrame(rows)


def apply_global_instrument_update(
    old_yahoo: str,
    new_yahoo: str,
    ticker: str,
    name: str,
    isin: str,
) -> tuple[bool, str]:
    """
    Actualiza ticker_Yahoo, ticker y name en movimientos, fondos, criptos y dividendos.
    ISIN se guarda en instrument_catalog (por instrumento global). Si isin vacío, se elimina la fila del catálogo.
    """
    old_yahoo = (old_yahoo or "").strip()
    new_yahoo = (new_yahoo or "").strip()
    ticker = (ticker or "").strip()
    name = (name or "").strip()
    isin = (isin or "").strip()
    if not old_yahoo:
        return False, "Selecciona un instrumento válido."
    if not new_yahoo:
        return False, "Ticker Yahoo no puede estar vacío."

    if new_yahoo != old_yahoo:
        with _get_db() as conn:
            for tbl in ("movimientos", "movimientos_fondos", "movimientos_criptos"):
                try:
                    n = conn.execute(
                        f'SELECT COUNT(*) FROM "{tbl}" WHERE ticker_Yahoo = ?', (new_yahoo,)
                    ).fetchone()[0]
                    if n and n > 0:
                        return (
                            False,
                            f"Ya hay movimientos con ticker_Yahoo «{new_yahoo}». Elige otro símbolo o unifica antes los datos.",
                        )
                except sqlite3.OperationalError:
                    pass

    with _get_db() as conn:
        for tbl in ("movimientos", "movimientos_fondos", "movimientos_criptos"):
            try:
                conn.execute(
                    f'UPDATE "{tbl}" SET ticker_Yahoo = ?, ticker = ?, name = ? WHERE ticker_Yahoo = ?',
                    (new_yahoo, ticker, name, old_yahoo),
                )
            except sqlite3.OperationalError:
                pass
        try:
            conn.execute(
                "UPDATE dividendos SET ticker_Yahoo = ?, ticker = ?, nombre = ? WHERE ticker_Yahoo = ?",
                (new_yahoo, ticker, name, old_yahoo),
            )
        except sqlite3.OperationalError:
            pass

        _init_instrument_catalog()
        conn.execute(
            "DELETE FROM instrument_catalog WHERE ticker_Yahoo IN (?, ?)",
            (old_yahoo, new_yahoo),
        )
        if isin:
            conn.execute(
                "INSERT INTO instrument_catalog (ticker_Yahoo, isin) VALUES (?, ?)",
                (new_yahoo, isin),
            )
        conn.commit()

    load_data.clear()
    load_data_fondos.clear()
    if hasattr(load_data_criptos, "clear"):
        load_data_criptos.clear()
    if hasattr(load_dividendos, "clear"):
        load_dividendos.clear()

    return True, f"Actualizado «{old_yahoo}» → datos guardados."


def _num_to_csv(val):
    """Formatea número para CSV con coma decimal."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if isinstance(val, (int, float)):
        s = str(val).replace(".", CSV_DECIMAL)
        return s
    return str(val)


def _row_to_db_val(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, (int, float)):
        return str(v).replace(".", CSV_DECIMAL)
    return str(v).strip()


def append_operation(new_row: dict) -> None:
    """Añade una fila a la tabla movimientos (acciones/ETFs)."""
    vals = [_row_to_db_val(new_row.get(c, "")) for c in MOVIMIENTOS_COLUMNS]
    placeholders = ", ".join("?" for _ in MOVIMIENTOS_COLUMNS)
    with _get_db() as conn:
        conn.execute(
            f'INSERT INTO movimientos ({", ".join(MOVIMIENTOS_COLUMNS)}) VALUES ({placeholders})',
            vals,
        )
        conn.commit()


def append_operation_fondos(new_row: dict) -> None:
    """Añade una fila a la tabla movimientos_fondos."""
    vals = [_row_to_db_val(new_row.get(c, "")) for c in MOVIMIENTOS_COLUMNS]
    placeholders = ", ".join("?" for _ in MOVIMIENTOS_COLUMNS)
    with _get_db() as conn:
        conn.execute(
            f'INSERT INTO movimientos_fondos ({", ".join(MOVIMIENTOS_COLUMNS)}) VALUES ({placeholders})',
            vals,
        )
        conn.commit()


def _init_db_criptos():
    """Crea la tabla movimientos_criptos si no existe."""
    cols_sql = ", ".join(f'"{c}" TEXT' for c in MOVIMIENTOS_CRIPTOS_COLUMNS)
    with _get_db() as conn:
        conn.execute(f"CREATE TABLE IF NOT EXISTS movimientos_criptos ({cols_sql})")


def append_operation_criptos(new_row: dict) -> None:
    """Añade una fila a la tabla movimientos_criptos."""
    _init_db_criptos()
    base_cols = [c for c in MOVIMIENTOS_COLUMNS if c in MOVIMIENTOS_CRIPTOS_COLUMNS]
    extra_cols = [c for c in MOVIMIENTOS_CRIPTOS_COLUMNS if c not in MOVIMIENTOS_COLUMNS]
    all_cols = [c for c in MOVIMIENTOS_CRIPTOS_COLUMNS]
    vals = [_row_to_db_val(new_row.get(c, "")) for c in all_cols]
    placeholders = ", ".join("?" for _ in all_cols)
    with _get_db() as conn:
        conn.execute(
            f'INSERT INTO movimientos_criptos ({", ".join(all_cols)}) VALUES ({placeholders})',
            vals,
        )
        conn.commit()


def _clear_form_nueva_operacion() -> None:
    """Limpia solo totales/cantidades del formulario tras guardar (posición, fecha, broker se mantienen)."""
    keys_to_clear = [
        "op_qty_nuevo", "op_precio_nuevo", "op_total_nuevo",
        "op_com_nuevo", "op_tax_nuevo", "op_dest_nuevo",
        "tf_qty", "tf_valor_eur",
        "bt_qty",
        "perm_qty_origen", "perm_qty_destino", "perm_valor_eur",
        "ctf_qty", "ctf_comision",
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]


def recalc_all_totals(use_ecb_rates: bool = False) -> tuple[int, str]:
    """
    Recalcula total, totalBaseCurrency, totalWithComission, totalWithComissionBaseCurrency
    para TODOS los movimientos (acciones, fondos, criptos) usando _recalc_totals.
    Si use_ecb_rates=True, actualiza exchangeRate con tipos del BCE (como Filios) antes de recalcular.
    Devuelve (filas actualizadas, mensaje).
    """
    updated = 0

    # Umbral: si el nuevo total difiere del anterior >100x o <0.01x, no actualizar
    RATIO_MAX = 100.0
    RATIO_MIN = 0.01

    def _recalc_table(conn: sqlite3.Connection, tabla: str) -> int:
        cols = "rowid, type, date, positionNumber, price, comission, taxes, exchangeRate, positionCurrency, comissionCurrency, taxesCurrency, totalWithComissionBaseCurrency"
        cur = conn.execute(f'SELECT {cols} FROM "{tabla}"')
        rows = cur.fetchall()
        n = 0
        tipos_recalc = ("buy", "sell", "switch", "switchbuy")
        for r in rows:
            tipo = str(r[1] or "").strip().lower()
            if tipo not in tipos_recalc:
                continue
            rowid = r[0]
            op_date = str(r[2] or "").strip() if len(r) > 2 else ""
            qty = _to_float(r[3])
            price = _to_float(r[4])
            comm = _to_float(r[5])
            tax = _to_float(r[6])
            fx = _to_float(r[7], 1.0)
            pos_ccy = str(r[8] or "").strip() or "EUR"
            comm_ccy = str(r[9] or "").strip()
            tax_ccy = str(r[10] or "").strip()
            old_twc = _to_float(r[11])

            # Si use_ecb_rates y la operación está en divisa distinta de EUR, usar tipo BCE
            if use_ecb_rates and pos_ccy and pos_ccy.upper() != "EUR" and op_date:
                ecb_rate = get_fx_rate_ecb(pos_ccy, op_date)
                if not math.isnan(ecb_rate) and ecb_rate > 0:
                    fx = ecb_rate
                    conn.execute(f'UPDATE "{tabla}" SET exchangeRate=? WHERE rowid=?', (fx, rowid))

            recalc = _recalc_totals(qty, price, comm, tax, fx, pos_ccy, comm_ccy, tax_ccy, tipo=tipo)
            new_twc = recalc["totalWithComissionBaseCurrency"]
            if old_twc and abs(old_twc) > 1e-6:
                ratio = new_twc / old_twc
                if ratio > RATIO_MAX or ratio < RATIO_MIN:
                    continue
            conn.execute(
                f'UPDATE "{tabla}" SET total=?, totalBaseCurrency=?, totalWithComission=?, totalWithComissionBaseCurrency=? WHERE rowid=?',
                (recalc["total"], recalc["totalBaseCurrency"], recalc["totalWithComission"], recalc["totalWithComissionBaseCurrency"], rowid),
            )
            n += 1
        return n

    with _get_db() as conn:
        updated += _recalc_table(conn, "movimientos")
        try:
            cur = conn.execute("SELECT COUNT(*) FROM movimientos_fondos")
            if cur.fetchone()[0] > 0:
                updated += _recalc_table(conn, "movimientos_fondos")
        except sqlite3.OperationalError:
            pass
        try:
            _init_db_criptos()
            cur = conn.execute("SELECT COUNT(*) FROM movimientos_criptos")
            if cur.fetchone()[0] > 0:
                updated += _recalc_table(conn, "movimientos_criptos")
        except sqlite3.OperationalError:
            pass
        conn.commit()

    load_data.clear()
    load_data_fondos.clear()
    if hasattr(load_data_criptos, "clear"):
        load_data_criptos.clear()
    return updated, f"Recalculados totales en {updated} movimientos (acciones, fondos y criptos)."


def write_full_db(df: pd.DataFrame) -> None:
    """Reescribe todos los movimientos en la base de datos. df sin columnas Tipo, Comisión (€), datetime_full."""
    cols = [c for c in MOVIMIENTOS_COLUMNS if c in df.columns]
    if not cols:
        return
    out = df[cols].copy()
    for col in ("date", "time"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    placeholders = ", ".join("?" for _ in MOVIMIENTOS_COLUMNS)
    with _get_db() as conn:
        conn.execute("DELETE FROM movimientos")
        for _, row in out.iterrows():
            vals = [_row_to_db_val(row.get(c, "")) for c in MOVIMIENTOS_COLUMNS]
            conn.execute(
                f'INSERT INTO movimientos ({", ".join(MOVIMIENTOS_COLUMNS)}) VALUES ({placeholders})',
                vals,
            )
        conn.commit()


def write_full_db_fondos(df: pd.DataFrame) -> None:
    """Reescribe todos los movimientos de fondos en la tabla movimientos_fondos."""
    cols = [c for c in MOVIMIENTOS_COLUMNS if c in df.columns]
    if not cols:
        return
    out = df[cols].copy()
    for col in ("date", "time"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    placeholders = ", ".join("?" for _ in MOVIMIENTOS_COLUMNS)
    with _get_db() as conn:
        conn.execute("DELETE FROM movimientos_fondos")
        for _, row in out.iterrows():
            vals = [_row_to_db_val(row.get(c, "")) for c in MOVIMIENTOS_COLUMNS]
            conn.execute(
                f'INSERT INTO movimientos_fondos ({", ".join(MOVIMIENTOS_COLUMNS)}) VALUES ({placeholders})',
                vals,
            )
        conn.commit()


def _normalize_time_to_24h(time_str: str) -> str:
    """
    Convierte hora en formato 12h (con a.m./p.m. o variantes corruptas) a 24h 'HH:MM:SS'.
    Si ya está en 24h o no se reconoce, devuelve el valor normalizado o el original.
    """
    if not time_str or not isinstance(time_str, str):
        return "00:00:00"
    s = time_str.strip()
    # Reconocer "HH:MM:SS" seguido opcionalmente de espacio y a.m./p.m. (con posibles caracteres raros)
    m = re.match(r"^(\d{1,2}):(\d{2}):(\d{2})\s*(.*)$", s, re.IGNORECASE)
    if not m:
        return s if re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", s) else "00:00:00"
    h, mm, ss, suffix = int(m.group(1)), m.group(2), m.group(3), (m.group(4) or "").strip().lower()
    is_pm = "p" in suffix and "m" in suffix
    is_am = "a" in suffix and "m" in suffix
    if is_pm and h != 12:
        h = h + 12
    elif is_am and h == 12:
        h = 0
    return f"{h:02d}:{mm}:{ss}"


def export_to_csv() -> bool:
    """
    Exporta los datos actuales a acciones.csv (respaldo, formato coma decimal).
    La fuente de verdad es la base SQLite; el CSV es solo copia de seguridad.
    """
    try:
        df = load_data()
        cols = [c for c in MOVIMIENTOS_COLUMNS if c in df.columns]
        if not cols:
            return False
        out = df[cols].copy()
        for col in ("date", "time"):
            if col in out.columns:
                out[col] = out[col].astype(str)
        out.to_csv(CSV_PATH, index=False, decimal=CSV_DECIMAL, sep=CSV_SEP, encoding=CSV_ENCODING)
        load_data.clear()
        return True
    except Exception:
        return False


def export_fondos_to_csv() -> bool:
    """
    Exporta los movimientos de fondos a fondos.csv (respaldo, formato coma decimal).
    """
    try:
        _init_db_fondos()
        df = load_data_fondos()
        cols = [c for c in MOVIMIENTOS_COLUMNS if c in df.columns]
        if not cols:
            return False
        out = df[cols].copy()
        for col in ("date", "time"):
            if col in out.columns:
                out[col] = out[col].astype(str)
        out.to_csv(FONDOS_CSV_PATH, index=False, decimal=CSV_DECIMAL, sep=CSV_SEP, encoding=CSV_ENCODING)
        load_data_fondos.clear()
        return True
    except Exception:
        return False


def export_criptos_to_csv() -> bool:
    """
    Exporta los movimientos de cripto desde la base SQLite a movimientos_criptos.csv
    (respaldo, formato coma decimal).
    """
    try:
        _init_db_criptos()
        df = load_data_criptos()
        cols = [c for c in MOVIMIENTOS_CRIPTOS_COLUMNS if c in df.columns]
        if not cols:
            return False
        out = df[cols].copy()
        for col in ("date", "time"):
            if col in out.columns:
                out[col] = out[col].astype(str)
        out.to_csv(MOVIMIENTOS_CRIPTOS_CSV_PATH, index=False, decimal=CSV_DECIMAL, sep=CSV_SEP, encoding=CSV_ENCODING)
        load_data_criptos.clear()
        return True
    except Exception:
        return False


def restore_movimientos_from_csv() -> tuple[bool, str]:
    """
    Restaura la tabla movimientos desde acciones.csv. No toca movimientos_fondos.
    Devuelve (éxito, mensaje).
    """
    if not Path(CSV_PATH).exists():
        return False, f"No existe el archivo {CSV_PATH}."
    try:
        df = pd.read_csv(CSV_PATH, sep=CSV_SEP, encoding=CSV_ENCODING, dtype=str, keep_default_na=False)
    except Exception as e:
        return False, f"No se pudo leer el CSV: {e}"
    cols = [c for c in MOVIMIENTOS_COLUMNS if c in df.columns]
    if not cols:
        return False, "El CSV no tiene las columnas esperadas."
    for col in ["positionNumber", "price", "total", "totalBaseCurrency", "totalWithComission", "totalWithComissionBaseCurrency", "comission", "taxes", "exchangeRate"]:
        if col in df.columns:
            s = df[col].astype(str).str.strip().str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(s, errors="coerce")
    if "date" in df.columns:
        df["date"] = df["date"].astype(str).str.split("T").str[0].str.strip()
    if "time" in df.columns:
        df["time"] = df["time"].astype(str).str.strip().apply(_normalize_time_to_24h)
    write_full_db(df[[c for c in MOVIMIENTOS_COLUMNS if c in df.columns]])
    load_data.clear()
    return True, f"Restaurados {len(df)} movimientos de acciones desde {CSV_PATH}. Los fondos no se han modificado."


def restore_fondos_from_csv() -> tuple[bool, str]:
    """
    Restaura la tabla movimientos_fondos desde fondos.csv. No toca movimientos (acciones).
    """
    if not Path(FONDOS_CSV_PATH).exists():
        return False, f"No existe el archivo {FONDOS_CSV_PATH}."
    try:
        df = pd.read_csv(FONDOS_CSV_PATH, sep=CSV_SEP, encoding=CSV_ENCODING, dtype=str, keep_default_na=False)
    except Exception as e:
        return False, f"No se pudo leer el CSV: {e}"
    if "nombre" in df.columns and "name" not in df.columns:
        df["name"] = df["nombre"].astype(str).str.strip()
    cols = [c for c in MOVIMIENTOS_COLUMNS if c in df.columns]
    if not cols:
        return False, "El CSV no tiene las columnas esperadas."
    for col in ["positionNumber", "price", "total", "totalBaseCurrency", "totalWithComission", "totalWithComissionBaseCurrency", "comission", "taxes", "exchangeRate"]:
        if col in df.columns:
            s = df[col].astype(str).str.strip().str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(s, errors="coerce")
    if "date" in df.columns:
        df["date"] = df["date"].astype(str).str.split("T").str[0].str.strip()
    if "time" in df.columns:
        df["time"] = df["time"].astype(str).str.strip().apply(_normalize_time_to_24h)
    write_full_db_fondos(df[[c for c in MOVIMIENTOS_COLUMNS if c in df.columns]])
    load_data_fondos.clear()
    return True, f"Restaurados {len(df)} movimientos de fondos desde {FONDOS_CSV_PATH}. Las acciones no se han modificado."


st.set_page_config(
    page_title="Cartera de Inversión",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Carga movimientos desde la base SQLite, ordena por datetime_full
    y normaliza tipos numéricos. La primera vez migra desde CSV si existe.
    """
    _init_db()
    _migrate_csv_to_db()

    with _get_db() as conn:
        df = pd.read_sql("SELECT rowid AS _rowid_, * FROM movimientos", conn)

    if df.empty:
        df = pd.DataFrame(columns=["_rowid_"] + MOVIMIENTOS_COLUMNS)

    # Limpiamos date y construimos datetime_full
    if "date" in df.columns:
        df["date"] = (
            df["date"].astype(str).str.split("T").str[0].str.strip()
        )
    if {"date", "time"}.issubset(df.columns):
        df["time"] = df["time"].astype(str).str.strip().apply(_normalize_time_to_24h)
        dt_str = df["date"].astype(str).str.strip() + " " + df["time"]
        df["datetime_full"] = pd.to_datetime(dt_str, format="mixed", errors="coerce")
    elif "date" in df.columns:
        df["datetime_full"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="mixed", errors="coerce")
    else:
        df["datetime_full"] = pd.Series(pd.RangeIndex(len(df)), index=df.index)

    df = df.sort_values("datetime_full").reset_index(drop=True)

    numeric_cols = [
        "positionNumber", "price", "totalWithComissionBaseCurrency",
        "totalBaseCurrency", "total", "exchangeRate", "comission", "taxes",
    ]
    for col in numeric_cols:
        if col in df.columns:
            s = df[col].astype(str).str.strip().str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(s, errors="coerce")

    return df


@st.cache_data
def load_data_criptos() -> pd.DataFrame:
    """
    Carga movimientos de cripto desde la base SQLite (tabla movimientos_criptos),
    ordena por datetime_full y normaliza tipos numéricos.
    """
    _init_db_criptos()
    with _get_db() as conn:
        try:
            # Usamos rowid como identificador estable para poder editar filas
            df = pd.read_sql("SELECT rowid AS _rowid_, * FROM movimientos_criptos", conn)
        except Exception:
            return pd.DataFrame(columns=["_rowid_"] + MOVIMIENTOS_CRIPTOS_COLUMNS)

    if df.empty:
        return pd.DataFrame(columns=["_rowid_"] + MOVIMIENTOS_CRIPTOS_COLUMNS)

    # Normalizar fecha y hora
    if "date" in df.columns:
        df["date"] = df["date"].astype(str).str.split("T").str[0].str.strip()
    if {"date", "time"}.issubset(df.columns):
        df["time"] = df["time"].astype(str).str.strip().apply(_normalize_time_to_24h)
        dt_str = df["date"].astype(str).str.strip() + " " + df["time"]
        df["datetime_full"] = pd.to_datetime(dt_str, format="mixed", errors="coerce")
    elif "date" in df.columns:
        df["datetime_full"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="mixed", errors="coerce")
    else:
        df["datetime_full"] = pd.Series(pd.RangeIndex(len(df)), index=df.index)

    df = df.sort_values("datetime_full").reset_index(drop=True)

    # Normalizar numéricos
    numeric_cols = [
        "positionNumber", "price", "totalWithComissionBaseCurrency",
        "totalBaseCurrency", "total", "exchangeRate", "comission", "taxes",
    ]
    for col in numeric_cols:
        if col in df.columns:
            s = df[col].astype(str).str.strip().str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(s, errors="coerce")

    # Normalizar name: si name == ticker_Yahoo (ej. ETH-EUR), usar nombre humano
    if {"name", "ticker", "ticker_Yahoo"}.issubset(df.columns):
        name_str = df["name"].astype(str).str.strip()
        yahoo_str = df["ticker_Yahoo"].astype(str).str.strip()
        mask = (name_str == yahoo_str) & (yahoo_str != "")
        if mask.any():
            def _norm_name(row):
                t = str(row["ticker"]).strip().upper()
                if "-" in t:
                    t = t.split("-")[0]
                if not t and str(row.get("ticker_Yahoo", "")).strip():
                    t = str(row["ticker_Yahoo"]).strip().split("-")[0].upper()
                return CRYPTO_TICKER_NAMES.get(t, t or str(row.get("ticker_Yahoo", "")).strip())

            df.loc[mask, "name"] = df.loc[mask].apply(_norm_name, axis=1)

    return df


def _safe_get(row: pd.Series, key: str, default=None):
    return row[key] if key in row and not pd.isna(row[key]) else default


def _crypto_ticker_yahoo(ticker: str, yahoo: str) -> str:
    """Para criptos: si yahoo está vacío, devuelve {ticker}-EUR (evita duplicar si ya termina en -EUR)."""
    if yahoo and str(yahoo).strip():
        return str(yahoo).strip()
    t = str(ticker or "").strip()
    if not t:
        return ""
    if t.upper().endswith("-EUR"):
        return t
    return f"{t}-EUR"


def _to_float(x, default=0.0):
    """Convierte a float aceptando coma como separador decimal (ej. '45,0' -> 45.0)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return default
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).strip().replace(",", "."))
    except (ValueError, TypeError):
        return default


def _recalc_totals(
    qty: float,
    price: float,
    comm: float,
    tax: float,
    fx: float,
    pos_ccy: str,
    comm_ccy: str,
    tax_ccy: str,
    tipo: str = "",
) -> dict[str, float]:
    """
    Recalcula total, totalBaseCurrency, totalWithComission, totalWithComissionBaseCurrency.
    - Compras (buy, switchbuy): totalWithComission = totalBase + comisión + impuestos.
    - Ventas (sell, switch): totalWithComission = totalBase - comisión - impuestos (lo que realmente recibes).
    """
    total_local = qty * price
    total_base = total_local * fx if fx and abs(fx) > 1e-9 else total_local
    comm_eur = comm if (comm_ccy or "").strip().upper() == "EUR" else comm * fx
    tax_eur = tax if (tax_ccy or "").strip().upper() == "EUR" else tax * fx
    tipo_lower = str(tipo or "").strip().lower()
    if tipo_lower in ("sell", "switch"):
        total_with_comm_base = total_base - comm_eur - tax_eur
        comm_local = comm if (comm_ccy or "").strip().upper() == (pos_ccy or "").strip().upper() else (comm_eur / fx if fx and abs(fx) > 1e-9 else 0.0)
        tax_local = tax if (tax_ccy or "").strip().upper() == (pos_ccy or "").strip().upper() else (tax_eur / fx if fx and abs(fx) > 1e-9 else 0.0)
        total_with_comm_local = total_local - comm_local - tax_local
    else:
        total_with_comm_base = total_base + comm_eur + tax_eur
        comm_local = comm if (comm_ccy or "").strip().upper() == (pos_ccy or "").strip().upper() else (comm_eur / fx if fx and abs(fx) > 1e-9 else 0.0)
        tax_local = tax if (tax_ccy or "").strip().upper() == (pos_ccy or "").strip().upper() else (tax_eur / fx if fx and abs(fx) > 1e-9 else 0.0)
        total_with_comm_local = total_local + comm_local + tax_local
    return {
        "total": total_local,
        "totalBaseCurrency": total_base,
        "totalWithComission": total_with_comm_local,
        "totalWithComissionBaseCurrency": total_with_comm_base,
    }


def compute_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Procesa todos los movimientos para obtener las posiciones abiertas
    por broker y ticker (Yahoo).

    - Usa totalWithComissionBaseCurrency como coste histórico en EUR.
    - Aplica traspasos entre brokers manteniendo el coste acumulado.
    - Descarta posiciones prácticamente cerradas (< 1e-8 títulos).
    """
    # Clave: (broker, ticker_Yahoo); guardamos también el ticker original del CSV,
    # el tipo de activo (acción/ETF, etc.) y el coste tanto en EUR como en la divisa original.
    positions: dict[tuple[str, str], dict[str, float | str]] = {}

    for _, row in df.iterrows():
        broker = _safe_get(row, "broker")
        ticker_y = _safe_get(row, "ticker_Yahoo")
        ticker_orig = _safe_get(row, "ticker")
        name = _safe_get(row, "name") or _safe_get(row, "ticker") or ticker_y
        pos_currency = _safe_get(row, "positionCurrency", "EUR")
        pos_type = _safe_get(row, "positionType")
        movement_type_raw = _safe_get(row, "type")
        movement_type = str(movement_type_raw or "").strip() if movement_type_raw is not None else ""

        # Siempre exigimos ticker_Yahoo, pero permitimos broker vacío
        # solo para movimientos de tipo 'split' (como BY6).
        if ticker_y is None:
            continue
        if movement_type and str(movement_type).strip().lower() != "split" and broker is None:
            continue

        key_ticker = ticker_y
        qty = _safe_get(row, "positionNumber", 0.0) or 0.0
        total_eur = _safe_get(row, "totalWithComissionBaseCurrency", 0.0) or 0.0
        price_local = _safe_get(row, "price", 0.0) or 0.0

        key = (broker, key_ticker)

        def ensure_position(k: tuple[str, str]):
            if k not in positions:
                positions[k] = {
                    "broker": k[0],
                    "ticker_yahoo": k[1],
                    "ticker_original": ticker_orig,
                    "name": name,
                    "positionCurrency": pos_currency,
                    "positionType": pos_type,
                    "quantity": 0.0,
                    "cost_eur": 0.0,
                    "cost_local": 0.0,
                }
            else:
                if name:
                    positions[k]["name"] = name
                if pos_currency:
                    positions[k]["positionCurrency"] = pos_currency
                if ticker_orig:
                    positions[k]["ticker_original"] = ticker_orig
                if pos_type:
                    positions[k]["positionType"] = pos_type

        # Split de acciones: ajustamos títulos y precios medios
        if movement_type and str(movement_type).strip().lower() == "split":
            factor = _to_float(_safe_get(row, "positionNumber", 1.0), 1.0)
            if factor <= 0:
                continue

            # Si viene broker informado, solo ajustamos ese broker;
            # si viene vacío (caso BY6), aplicamos el split a todos los brokers
            # que tengan ese ticker en posiciones ya acumuladas.
            if broker:
                split_keys = [(broker, key_ticker)]
            else:
                split_keys = [k for k in positions.keys() if k[1] == key_ticker]

            for k in split_keys:
                if k not in positions:
                    continue
                pos_split = positions[k]
                if abs(pos_split["quantity"]) < MIN_POSITION:
                    continue
                pos_split["quantity"] *= factor
                # cost_eur y cost_local se mantienen -> precio medio se divide por factor

            continue

        # Traspaso entre brokers: movemos cantidad y coste proporcional
        if movement_type and str(movement_type).strip().lower() == "brokertransfer":
            new_broker_raw = _safe_get(row, "brokerTransferNewBroker")
            if not new_broker_raw:
                continue
            new_broker = str(new_broker_raw).strip()

            source_key = (broker, key_ticker)
            target_key = (new_broker, key_ticker)

            ensure_position(source_key)
            ensure_position(target_key)

            src = positions[source_key]
            tgt = positions[target_key]

            transfer_qty = _to_float(qty)
            if abs(src["quantity"]) < MIN_POSITION or abs(transfer_qty) < MIN_POSITION:
                continue

            ratio = transfer_qty / src["quantity"]
            transfer_cost_eur = src["cost_eur"] * ratio
            transfer_cost_local = src["cost_local"] * ratio

            src["quantity"] -= transfer_qty
            src["cost_eur"] -= transfer_cost_eur
            src["cost_local"] -= transfer_cost_local

            tgt["quantity"] += transfer_qty
            tgt["cost_eur"] += transfer_cost_eur
            tgt["cost_local"] += transfer_cost_local

            continue

        # Operativas normales: buys / sells, switch / switchBuy
        ensure_position(key)
        pos = positions[key]

        qty_change = _to_float(qty)
        _mt = str(movement_type or "").strip().lower()

        if _mt in ("buy", "switchbuy"):
            pos["quantity"] += qty_change
            pos["cost_eur"] += _to_float(total_eur)
            pos["cost_local"] += _to_float(price_local) * qty_change
        elif _mt in ("sell", "switch"):
            if abs(pos["quantity"]) < MIN_POSITION:
                continue

            sell_qty = qty_change
            if sell_qty <= 0:
                sell_qty = abs(sell_qty)

            qty_before = pos["quantity"]
            if sell_qty > qty_before:
                sell_qty = qty_before

            if qty_before > 0:
                avg_cost_per_unit = pos["cost_eur"] / qty_before
                avg_cost_local = pos["cost_local"] / qty_before
            else:
                avg_cost_per_unit = 0.0
                avg_cost_local = 0.0

            qty_after = qty_before - sell_qty
            cost_after = avg_cost_per_unit * qty_after
            cost_local_after = avg_cost_local * qty_after

            pos["quantity"] = qty_after
            pos["cost_eur"] = cost_after
            pos["cost_local"] = cost_local_after
        else:
            continue

    # Construimos DataFrame de posiciones abiertas
    rows = []
    for (broker, ticker_y), p in positions.items():
        qty = float(p["quantity"])
        cost_eur = float(p["cost_eur"])
        cost_local = float(p["cost_local"])
        if abs(qty) < MIN_POSITION:
            continue

        avg_price_eur = cost_eur / qty if qty != 0 else math.nan
        avg_price_local = cost_local / qty if qty != 0 else math.nan

        rows.append(
            {
                "Broker": broker,
                # Mostramos el ticker original del CSV y guardamos también el de Yahoo
                "Ticker": p.get("ticker_original") or ticker_y,
                "Ticker_Yahoo": p.get("ticker_yahoo") or ticker_y,
                "Nombre": p["name"],
                "Titulos": qty,
                "Precio Medio Moneda": avg_price_local,
                "Precio Medio €": avg_price_eur,
                "Inversion €": cost_eur,
                "Moneda Activo": p["positionCurrency"] or "EUR",
                "Tipo activo": p.get("positionType") or "",
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Broker",
                "Ticker",
                "Ticker_Yahoo",
                "Nombre",
                "Titulos",
                "Precio Medio Moneda",
                "Precio Medio €",
                "Inversion €",
                "Moneda Activo",
            ]
        )

    return pd.DataFrame(rows)


def compute_positions_fifo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Posiciones con FIFO en ventas y traspasos (como Filios).
    Usa lotes internamente: buy añade lote, sell consume FIFO, transfer mueve lotes FIFO.
    Misma firma y formato de salida que compute_positions.
    """
    positions: dict[tuple[str, str], dict] = {}

    def ensure(key: tuple[str, str], row: pd.Series):
        if key not in positions:
            positions[key] = {
                "lots": [],
                "ticker_orig": _safe_get(row, "ticker"),
                "name": _safe_get(row, "name") or _safe_get(row, "ticker") or "",
                "pos_currency": _safe_get(row, "positionCurrency", "EUR"),
                "pos_type": _safe_get(row, "positionType", ""),
            }

    for _, row in df.iterrows():
        broker = _safe_get(row, "broker")
        ticker_y = _safe_get(row, "ticker_Yahoo") or _safe_get(row, "ticker")
        if ticker_y is None or (isinstance(ticker_y, str) and not str(ticker_y).strip()):
            continue
        key_ticker = str(ticker_y).strip()
        if broker is None or (isinstance(broker, str) and not str(broker).strip()):
            if str(_safe_get(row, "type", "") or "").strip().lower() != "split":
                continue
        broker = str(broker or "").strip()

        qty = _to_float(_safe_get(row, "positionNumber"), 0)
        total_eur = _to_float(_safe_get(row, "totalWithComissionBaseCurrency"), 0)
        price_local = _to_float(_safe_get(row, "price"), 0)
        t = str(_safe_get(row, "type", "") or "").strip().lower()

        if t == "split" and qty > 0:
            factor = qty
            keys = [(broker, key_ticker)] if broker else [k for k in positions if k[1] == key_ticker]
            for k in keys:
                if k in positions:
                    for lot in positions[k]["lots"]:
                        lot["qty"] *= factor
            continue

        if not broker or qty is None or qty <= 0:
            if t not in ("buy", "switchbuy", "sell", "switch", "brokertransfer"):
                continue

        key = (broker, key_ticker)

        if t == "brokertransfer":
            dest_raw = _safe_get(row, "brokerTransferNewBroker")
            if not dest_raw:
                continue
            dest_broker = str(dest_raw).strip()
            src_key, dst_key = key, (dest_broker, key_ticker)
            ensure(src_key, row)
            ensure(dst_key, row)
            src_lots = positions[src_key]["lots"]
            remaining = qty
            while remaining > MIN_POSITION and src_lots:
                lot = src_lots[0]
                lot_qty = lot["qty"]
                if lot_qty <= remaining + MIN_POSITION:
                    positions[dst_key]["lots"].append({"qty": lot_qty, "cost_eur": lot["cost_eur"], "cost_local": lot["cost_local"]})
                    remaining -= lot_qty
                    src_lots.pop(0)
                else:
                    frac = remaining / lot_qty
                    positions[dst_key]["lots"].append({
                        "qty": remaining,
                        "cost_eur": lot["cost_eur"] * frac,
                        "cost_local": lot["cost_local"] * frac,
                    })
                    lot["qty"] -= remaining
                    lot["cost_eur"] -= lot["cost_eur"] * frac
                    lot["cost_local"] -= lot["cost_local"] * frac
                    remaining = 0
            continue

        if t in ("buy", "switchbuy"):
            ensure(key, row)
            cost_eur = total_eur if total_eur is not None else 0
            cost_local = price_local * qty if price_local else 0
            positions[key]["lots"].append({"qty": qty, "cost_eur": cost_eur, "cost_local": cost_local})
            continue

        if t in ("sell", "switch"):
            ensure(key, row)
            lots = positions[key]["lots"]
            remaining = abs(qty)
            while remaining > MIN_POSITION and lots:
                lot = lots[0]
                lot_qty = lot["qty"]
                if lot_qty <= remaining + MIN_POSITION:
                    remaining -= lot_qty
                    lots.pop(0)
                else:
                    frac = remaining / lot_qty
                    lot["qty"] -= remaining
                    lot["cost_eur"] -= lot["cost_eur"] * frac
                    lot["cost_local"] -= lot["cost_local"] * frac
                    remaining = 0
            continue

    rows = []
    for (broker, ticker_y), p in positions.items():
        total_qty = sum(l["qty"] for l in p["lots"])
        total_cost_eur = sum(l["cost_eur"] for l in p["lots"])
        total_cost_local = sum(l["cost_local"] for l in p["lots"])
        if abs(total_qty) < MIN_POSITION:
            continue
        avg_eur = total_cost_eur / total_qty if total_qty else math.nan
        avg_local = total_cost_local / total_qty if total_qty else math.nan
        rows.append({
            "Broker": broker,
            "Ticker": p.get("ticker_orig") or ticker_y,
            "Ticker_Yahoo": ticker_y,
            "Nombre": p["name"],
            "Titulos": total_qty,
            "Precio Medio Moneda": avg_local,
            "Precio Medio €": avg_eur,
            "Inversion €": total_cost_eur,
            "Moneda Activo": p["pos_currency"] or "EUR",
            "Tipo activo": p.get("pos_type") or "",
        })

    if not rows:
        return pd.DataFrame(columns=["Broker", "Ticker", "Ticker_Yahoo", "Nombre", "Titulos", "Precio Medio Moneda", "Precio Medio €", "Inversion €", "Moneda Activo"])
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def get_quotes(tickers: list[str]) -> pd.DataFrame:
    """
    Descarga precios actuales y moneda desde Yahoo Finance.
    """
    tickers = sorted(set(t for t in tickers if t))
    data: dict[str, dict[str, float | str]] = {}

    for t in tickers:
        try:
            tk = yf.Ticker(t)
            info = getattr(tk, "fast_info", {}) or {}
            base_info = getattr(tk, "info", {}) or {}

            # Último precio
            last = info.get("last_price") or info.get("lastPrice")
            if last is None:
                last = base_info.get("regularMarketPrice")
                if last is None:
                    hist = tk.history(period="1d")
                    last = hist["Close"].iloc[-1] if not hist.empty else math.nan
            last = float(last)

            # Cierre previo para calcular GyP de hoy
            prev_close = info.get("previous_close") or info.get("previousClose")
            if prev_close is None:
                prev_close = base_info.get("previousClose")
                if prev_close is None:
                    hist = tk.history(period="2d")
                    if len(hist) >= 2:
                        prev_close = hist["Close"].iloc[-2]
                    else:
                        prev_close = math.nan
            prev_close = float(prev_close) if prev_close is not None else math.nan

            currency = info.get("currency") or base_info.get("currency")
        except Exception:
            last = math.nan
            prev_close = math.nan
            currency = None

        data[t] = {
            "Precio Actual": last,
            "Cierre Previo": prev_close,
            "Moneda Yahoo": currency,
        }

    return pd.DataFrame.from_dict(data, orient="index")


@st.cache_data(ttl=300)
def get_fx_rate(pair: str) -> float:
    """
    Devuelve el tipo de cambio para un par tipo 'EURUSD=X' o 'EURCAD=X'.
    """
    try:
        tk = yf.Ticker(pair)
        info = getattr(tk, "fast_info", {}) or {}
        base_info = getattr(tk, "info", {}) or {}

        fx = info.get("last_price") or info.get("lastPrice")
        if fx is None:
            fx = base_info.get("regularMarketPrice")
            if fx is None:
                hist = tk.history(period="1d")
                fx = hist["Close"].iloc[-1] if not hist.empty else math.nan
        return float(fx)
    except Exception:
        return math.nan


@st.cache_data(ttl=3600)
def get_fx_rate_ecb(currency: str, as_of_date) -> float:
    """
    Tipo de cambio BCE (Frankfurter API, datos oficiales) para una fecha.
    Devuelve cuántos EUR por 1 unidad de moneda. Ej: USD -> 0.92 significa 1 USD = 0.92 EUR.
    Usa la misma fuente que Filios para coincidir en declaraciones fiscales.
    """
    if not currency or str(currency).upper() == "EUR":
        return 1.0
    ccy = str(currency).upper()
    try:
        date_str = pd.Timestamp(as_of_date).strftime("%Y-%m-%d")
        url = f"https://api.frankfurter.app/{date_str}?from={ccy}&to=EUR"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        rates = data.get("rates", {})
        if "EUR" in rates:
            return float(rates["EUR"])
        return math.nan
    except Exception:
        return math.nan


@st.cache_data(ttl=3600)
def get_fx_rate_for_date(currency: str, as_of_date) -> float:
    """
    Tipo de cambio de cierre para una fecha: cuántos EUR por 1 unidad de moneda.
    Usa BCE (Frankfurter) por defecto - misma fuente que Filios.
    Fallback a Yahoo Finance si la API BCE no responde.
    """
    if not currency or str(currency).upper() == "EUR":
        return 1.0
    rate = get_fx_rate_ecb(currency, as_of_date)
    if not math.isnan(rate) and rate > 0:
        return rate
    # Fallback: Yahoo Finance
    ccy = str(currency).upper()
    try:
        pair = f"{ccy}EUR=X"
        tk = yf.Ticker(pair)
        start = pd.Timestamp(as_of_date)
        end = start + pd.Timedelta(days=1)
        hist = tk.history(start=start, end=end)
        if not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[-1])
        pair_eur = f"EUR{ccy}=X"
        tk2 = yf.Ticker(pair_eur)
        hist2 = tk2.history(start=start, end=end)
        if not hist2.empty and "Close" in hist2.columns:
            r = float(hist2["Close"].iloc[-1])
            return 1.0 / r if r and r != 0 else math.nan
    except Exception:
        pass
    return math.nan


@st.cache_data(ttl=1800)
def get_fx_rate_at_datetime(currency: str, as_of_datetime: str) -> float:
    """
    Tipo de cambio aproximado en el momento de la operación.
    Usa datos intradía (intervalo 5m) para el par {CCY}EUR=X y toma
    el dato más cercano a la fecha y hora indicadas.
    as_of_datetime debe ser una cadena 'YYYY-MM-DD HH:MM:SS'.
    """
    if not currency or str(currency).upper() == "EUR":
        return 1.0
    ccy = str(currency).upper()
    try:
        ts = pd.to_datetime(as_of_datetime)
        if pd.isna(ts):
            return math.nan
        pair = f"{ccy}EUR=X"
        tk = yf.Ticker(pair)
        start = ts - pd.Timedelta(minutes=30)
        end = ts + pd.Timedelta(minutes=30)
        hist = tk.history(start=start, end=end, interval="5m")
        if hist.empty or "Close" not in hist.columns:
            return math.nan
        # Buscar el punto más cercano en el tiempo
        diffs = (hist.index - ts).to_series().abs()
        idx = diffs.idxmin()
        return float(hist.loc[idx, "Close"])
    except Exception:
        return math.nan


def _cotizaciones_signature(positions: pd.DataFrame) -> str:
    """Firma de la cartera (broker + ticker + inversión) para validar si la caché aplica."""
    if positions.empty or "Broker" not in positions.columns:
        return ""
    ticker_col = "Ticker_Yahoo" if "Ticker_Yahoo" in positions.columns else "Ticker"
    if ticker_col not in positions.columns:
        return ""
    inv_col = "Inversion €" if "Inversion €" in positions.columns else None
    inv_vals = (
        positions[inv_col].fillna(0).round(2).astype(str)
        if inv_col
        else ["0"] * len(positions)
    )
    keys = sorted(
        zip(
            positions["Broker"].astype(str),
            positions[ticker_col].astype(str),
            inv_vals,
        )
    )
    return hashlib.sha256(repr(keys).encode()).hexdigest()


def load_cotizaciones_cache(signature: str) -> tuple[pd.DataFrame | None, str | None]:
    """Carga cotizaciones desde disco si existen y la firma coincide. Devuelve (df, updated_at) o (None, None)."""
    if not signature or not Path(COTIZACIONES_CACHE_PATH).exists() or not Path(COTIZACIONES_META_PATH).exists():
        return None, None
    try:
        with open(COTIZACIONES_META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("signature") != signature:
            return None, None
        df = pd.read_pickle(COTIZACIONES_CACHE_PATH)
        return df, meta.get("updated_at")
    except Exception:
        return None, None


def save_cotizaciones_cache(df: pd.DataFrame, signature: str) -> None:
    """Guarda cotizaciones y metadatos en disco."""
    try:
        df.to_pickle(COTIZACIONES_CACHE_PATH)
        meta = {"signature": signature, "updated_at": datetime.now().isoformat()}
        with open(COTIZACIONES_META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=0)
    except Exception:
        pass


def clear_cotizaciones_cache() -> None:
    """Borra la caché de cotizaciones en disco (para forzar recarga con datos frescos)."""
    try:
        if Path(COTIZACIONES_CACHE_PATH).exists():
            Path(COTIZACIONES_CACHE_PATH).unlink()
        if Path(COTIZACIONES_META_PATH).exists():
            Path(COTIZACIONES_META_PATH).unlink()
    except Exception:
        pass


def load_precios_manuales() -> dict[str, float]:
    """Carga precios manuales desde JSON (ticker_yahoo -> precio en EUR)."""
    try:
        if Path(PRECIOS_MANUALES_PATH).exists():
            with open(PRECIOS_MANUALES_PATH, encoding="utf-8") as f:
                data = json.load(f)
                return {k: float(v) for k, v in (data or {}).items()}
    except Exception:
        pass
    return {}


def save_precios_manuales(data: dict[str, float]) -> None:
    """Guarda precios manuales en JSON."""
    try:
        with open(PRECIOS_MANUALES_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def enrich_with_market_data(positions: pd.DataFrame, manual_prices: dict[str, float] | None = None) -> pd.DataFrame:
    """
    Añade precios actuales, valor de mercado y plusvalías en EUR.
    manual_prices: dict ticker_yahoo -> precio en EUR para posiciones sin cotización en Yahoo.
    """
    if positions.empty:
        return positions

    # Usamos siempre el ticker de Yahoo para las cotizaciones
    ticker_col = "Ticker_Yahoo" if "Ticker_Yahoo" in positions.columns else "Ticker"
    quotes = get_quotes(positions[ticker_col].tolist())
    positions = positions.copy()

    # Unimos precios y moneda de Yahoo
    positions = positions.merge(
        quotes,
        left_on=ticker_col,
        right_index=True,
        how="left",
    )

    # Aplicar precios manuales para posiciones sin cotización en Yahoo
    if manual_prices:
        for idx, row in positions.iterrows():
            t = str(row.get(ticker_col) or "").strip()
            if t and t in manual_prices and (pd.isna(row.get("Precio Actual")) or row.get("Precio Actual") is None):
                positions.at[idx, "Precio Actual"] = manual_prices[t]
                positions.at[idx, "Moneda Yahoo"] = "EUR"
                positions.at[idx, "Moneda Activo"] = "EUR"

    # Determinamos moneda del activo (prioridad Yahoo, luego CSV)
    def decide_ccy(row: pd.Series) -> str:
        c1 = row.get("Moneda Yahoo")
        c2 = row.get("Moneda Activo")
        if isinstance(c1, str) and c1:
            return c1
        if isinstance(c2, str) and c2:
            return c2
        return "EUR"

    positions["Moneda Activo"] = positions.apply(decide_ccy, axis=1)

    # Conversión a EUR
    fx_cache: dict[str, float] = {}

    def get_fx_pair(ccy: str) -> tuple[float, str]:
        """
        Intenta obtener el tipo de cambio para pasar de ccy -> EUR.
        Primero prueba EUR{ccy}=X (ccy por 1 EUR),
        luego {ccy}EUR=X (EUR por 1 ccy).
        Devuelve (factor, modo):
          - modo = 'div': precio_ccy / factor
          - modo = 'mul': precio_ccy * factor
        """
        ccy = ccy.upper()
        # Ejemplo habitual: EURUSD=X (USD por 1 EUR)
        pair1 = f"EUR{ccy}=X"
        if pair1 not in fx_cache:
            fx_cache[pair1] = get_fx_rate(pair1)
        fx1 = fx_cache[pair1]
        if not pd.isna(fx1) and fx1 != 0:
            return float(fx1), "div"

        # Alternativa: USDEUR=X (EUR por 1 USD)
        pair2 = f"{ccy}EUR=X"
        if pair2 not in fx_cache:
            fx_cache[pair2] = get_fx_rate(pair2)
        fx2 = fx_cache[pair2]
        if not pd.isna(fx2) and fx2 != 0:
            return float(fx2), "mul"

        return math.nan, ""

    def price_in_eur(row: pd.Series) -> float:
        price = row.get("Precio Actual")
        ccy = row.get("Moneda Activo", "EUR")
        if not isinstance(ccy, str) or ccy.upper() == "EUR" or pd.isna(price):
            return float(price) if not pd.isna(price) else math.nan

        fx, mode = get_fx_pair(ccy)
        if pd.isna(fx) or fx == 0 or mode == "":
            return math.nan

        price_val = float(price)
        if mode == "div":
            # precio_ccy / (ccy por EUR) = precio en EUR
            return price_val / fx
        else:
            # precio_ccy * (EUR por ccy) = precio en EUR
            return price_val * fx

    positions["Precio Actual €"] = positions.apply(price_in_eur, axis=1)
    positions["Valor Mercado €"] = positions["Titulos"] * positions["Precio Actual €"]
    positions["Plusvalia €"] = positions["Valor Mercado €"] - positions["Inversion €"]
    positions["Plusvalia %"] = np.where(
        positions["Inversion €"].abs() > 0,
        positions["Plusvalia €"] / positions["Inversion €"] * 100.0,
        np.nan,
    )

    # GyP de hoy (% y €), usando la variación entre último precio y cierre previo
    def day_pnl_pct(row: pd.Series) -> float:
        last = row.get("Precio Actual")
        prev = row.get("Cierre Previo")
        if pd.isna(last) or pd.isna(prev) or prev == 0:
            return math.nan
        return (float(last) - float(prev)) / float(prev) * 100.0

    positions["GyP hoy %"] = positions.apply(day_pnl_pct, axis=1)
    positions["GyP hoy €"] = positions["Valor Mercado €"] * positions["GyP hoy %"] / 100.0

    return positions


def fmt_eur(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:,.2f} €"


def _fmt_div_currency(val, currency: str = "EUR") -> str:
    """Formatea número para listado dividendos: coma decimal y símbolo de moneda a la derecha."""
    if val is None or (isinstance(val, float) and pd.isna(val)) or str(val).strip() == "":
        return ""
    try:
        x = float(str(val).replace(",", ".").strip())
    except (ValueError, TypeError):
        return str(val).strip()
    sym = "€" if currency.upper() == "EUR" else "$" if currency.upper() == "USD" else "£" if currency.upper() == "GBP" else f" {currency.upper()}"
    return f"{x:.2f}".replace(".", ",") + f" {sym}".rstrip()


def fmt_qty(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.{DECIMALS_POSITION}f}".rstrip("0").rstrip(".")


def color_pnl(val):
    try:
        v = float(val)
    except Exception:
        return ""
    if v > 0:
        return "color: green;"
    if v < 0:
        return "color: red;"
    return ""


def compute_fifo_all(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula posiciones vivas y ventas para todos los tickers y brokers
    usando FIFO por lotes, aplicando splits sobre los lotes previos.

    Devuelve:
      - lots_df: lotes vivos (por broker/ticker)
      - sales_df: ventas con coste histórico y plusvalía/minusvalía
    """
    if "datetime_full" in df.columns:
        data = df.sort_values("datetime_full").copy()
    elif "date" in df.columns:
        data = df.sort_values("date").copy()
    else:
        data = df.copy()

    # Mantenemos un diccionario de lotes por (broker, ticker_yahoo)
    lots_by_key: dict[tuple[str, str], list[dict]] = {}
    sales_rows: list[dict] = []

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

            if broker:
                affected_keys = [(broker, key_ticker)]
            else:
                affected_keys = [k for k in lots_by_key.keys() if k[1] == key_ticker]

            for key in affected_keys:
                for lote in lots_by_key.get(key, []):
                    lote["Cantidad"] *= factor
                    if factor != 0:
                        lote["Precio medio €"] /= factor
            continue

        if broker is None or pd.isna(qty) or qty <= 0:
            continue

        key = (broker, key_ticker)
        if key not in lots_by_key:
            lots_by_key[key] = []

        # -------- COMPRAS: crean lotes --------
        if tipo_lower in ["buy", "switchbuy"]:
            if pd.isna(total_eur):
                continue
            price_eur = float(total_eur) / float(qty) if qty > 0 else 0.0
            lots_by_key[key].append(
                {
                    "Broker": broker,
                    "Ticker": ticker_orig or key_ticker,
                    "Ticker_Yahoo": ticker_y,
                    "Nombre": nombre,
                    "Fecha origen": fecha,
                    "Cantidad": float(qty),
                    "Precio medio €": float(price_eur),
                    "Tipo activo": tipo_activo,
                }
            )

        # -------- VENTAS / SWITCH SALIDA: consumen lotes FIFO --------
        elif tipo_lower in ["sell", "switch"]:
            if pd.isna(total_eur):
                continue
            qty_sell = float(qty)
            remaining = qty_sell
            cost_hist = 0.0

            lots = lots_by_key.get(key, [])
            while remaining > 0 and lots:
                lote = lots[0]
                lote_qty = lote["Cantidad"]
                if lote_qty <= remaining + 1e-8:
                    consumed = lote_qty
                    cost_hist += consumed * lote["Precio medio €"]
                    remaining -= consumed
                    lots.pop(0)
                else:
                    consumed = remaining
                    cost_hist += consumed * lote["Precio medio €"]
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

            dest_ret = _to_float(_safe_get(row, "destinationRetentionBaseCurrency"), 0.0)
            plusvalia = float(valor_transmision) - cost_hist
            sales_rows.append(
                {
                    "Broker": broker,
                    "Ticker": ticker_orig or key_ticker,
                    "Ticker_Yahoo": ticker_y,
                    "Nombre": nombre,
                    "Fecha venta": fecha,
                    "Cantidad vendida": float(qty_sell),
                    "Valor compra histórico (€)": cost_hist,
                    "Valor venta (€)": float(valor_transmision),
                    "Plusvalía / Minusvalía (€)": plusvalia,
                    "Retención dest. (€)": dest_ret,
                    "Tipo activo": tipo_activo,
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
    return lots_df, sales_df


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


def compute_fifo_criptos(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        return pd.DataFrame(), pd.DataFrame()

    if "datetime_full" in df.columns:
        data = df.sort_values("datetime_full").copy()
    elif "date" in df.columns:
        data = df.sort_values("date").copy()
    else:
        data = df.copy()

    lots_by_ticker: dict[str, list[dict]] = {}
    sales_rows: list[dict] = []

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
            new_lots, cost_hist = _consume_fifo_lotes_cripto_global(lots, qty)
            lots_by_ticker[ticker] = new_lots

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

            dest_ret = _to_float(row.get("destinationRetentionBaseCurrency"), 0.0)
            sales_rows.append(
                {
                    "Broker": broker,
                    "Ticker": ticker,
                    "Nombre": nombre,
                    "Fecha venta": date_str,
                    "Cantidad vendida": qty,
                    "Valor venta (€)": valor_transmision,
                    "Valor compra histórico (€)": cost_hist,
                    "Plusvalía / Minusvalía (€)": valor_transmision - cost_hist,
                    "Retención dest. (€)": dest_ret,
                    "Tipo activo": "crypto",
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
    return lots_df, sales_df


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


def compute_positions_criptos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula posiciones de cripto por broker y ticker a partir de movimientos_criptos.
    Usa FIFO para ventas, switch y traspasos (como Filios).
    - buy / switchBuy: añade lote FIFO.
    - sell / switch: consume lotes FIFO.
    - brokerTransfer: mueve lotes FIFO de origen a destino.
    - commission: consume cantidad FIFO (sin coste).
    - stakeReward: añade lote con coste 0.
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

    # Ajuste Kraken BTC con ledger oficial: solo mostrar si hay saldo > 0
    # Si no hay ledger o saldo 0, eliminar Kraken para no mostrar posiciones obsoletas
    ledger_path = Path(__file__).parent / "kraken_stocks_etfs_ledgers_2025-01-13-2025-12-31.csv"
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


def main() -> None:
    st.title("Cartera de Inversión")

    df = load_data()

    # Menú izquierda: solo páginas
    vista = st.sidebar.radio(
        "Página",
        ["Cartera", "Movimientos", "Fiscalidad", "Brokers", "Catálogo"],
        index=0,
        label_visibility="collapsed",
    )

    # Solo relevante en el add-on (Linux): en Windows /proc no existe y no se muestra ruido extra
    st.sidebar.caption("**📁 Ubicación de datos:**")
    mount_src = _get_data_mount_source()
    if mount_src:
        st.sidebar.code(f"Host: {mount_src}\nBD: {DB_PATH}\nCSV: {CSV_PATH}", language=None)
        st.sidebar.caption("Samba: share\\\\cartera_final (o addon_configs si usas /config)")
    else:
        st.sidebar.code(f"{DB_PATH}", language=None)

    CONFIG_PATH_FILE = Path("/data/data_path_override.txt")
    with st.sidebar.expander("⚙️ Ruta de datos (configurar)"):
        st.caption("Por defecto: /share/cartera_final. Reinicia el add-on tras cambiar.")
        current_override = CONFIG_PATH_FILE.read_text().strip() if CONFIG_PATH_FILE.exists() else ""
        new_path = st.text_input("Ruta (ej. /share/cartera_final, /config)", value=current_override or str(_DATA_DIR), key="data_path_input")
        if st.button("Guardar ruta", key="save_data_path"):
            try:
                if new_path and new_path.strip():
                    p = new_path.strip()
                    if not p.startswith("/"):
                        p = "/" + p
                    CONFIG_PATH_FILE.parent.mkdir(parents=True, exist_ok=True)
                    CONFIG_PATH_FILE.write_text(p, encoding="utf-8")
                    st.success(f"Guardado. Reinicia el add-on para usar: {p}")
                else:
                    if CONFIG_PATH_FILE.exists():
                        CONFIG_PATH_FILE.unlink()
                    st.info("Eliminado. Se usará la ruta por defecto. Reinicia el add-on.")
            except Exception as e:
                st.error(f"No se pudo guardar: {e}")

    with st.sidebar.expander("Mantenimiento"):
        st.caption(
            f"Los datos se guardan en la base SQLite ({Path(DB_PATH).name}). Exporta a CSV para respaldo; "
            "no abras los CSV con Excel si no quieres corromper el formato."
        )
        st.caption("**Acciones / ETFs:**")
        if st.button("Exportar acciones a CSV (respaldo)"):
            if export_to_csv():
                st.success("Exportado a acciones.csv. Recargando…")
                st.rerun()
            else:
                st.error("No se pudo exportar.")
        if st.button("Restaurar acciones desde acciones.csv"):
            ok, msg = restore_movimientos_from_csv()
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
        st.caption("**Fondos:**")
        if st.button("Exportar fondos a CSV (respaldo)"):
            if export_fondos_to_csv():
                st.success("Exportado a fondos.csv. Recargando…")
                st.rerun()
            else:
                st.error("No se pudo exportar fondos.")
        if st.button("Restaurar fondos desde fondos.csv"):
            ok, msg = restore_fondos_from_csv()
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
        st.caption("**Criptos:**")
        if st.button("Exportar criptos a movimientos_criptos.csv (respaldo)"):
            if export_criptos_to_csv():
                st.success("Exportado a movimientos_criptos.csv. Recargando…")
                st.rerun()
            else:
                st.error("No se pudo exportar criptos.")
        st.caption("**Totales:**")
        if st.button("📐 Recalcular totales", key="btn_recalc_totals", help="Recalcula total, totalBaseCurrency, totalWithComission y totalWithComissionBaseCurrency para todos los movimientos (acciones, fondos, criptos). No actualiza si detecta anomalías."):
            n, msg = recalc_all_totals()
            st.success(msg)
            st.rerun()
        if st.button("🔄 Recalcular con tipos BCE (como Filios)", key="btn_recalc_ecb", help="Actualiza exchangeRate con tipos del Banco Central Europeo y recalcula totales. Usa la misma fuente que Filios para coincidir en costes en EUR."):
            n, msg = recalc_all_totals(use_ecb_rates=True)
            st.success(msg)
            st.rerun()

    if vista == "Catálogo":
        st.header("Catálogo de instrumentos")
        st.caption(
            "ISIN es **global** por ticker Yahoo (clave de cotización). "
            "Ticker, nombre y Yahoo se escriben en **todos** los movimientos y dividendos que compartan ese Yahoo. "
            "No se actualizan referencias en campos de permuta (switch) que apunten a otro texto."
        )
        uni = get_universe_instruments_table()
        if uni.empty:
            st.info("No hay instrumentos en movimientos.")
        else:
            _CAT_PLACEHOLDER = "—— Elige instrumento ——"
            uni_pick = uni.copy()
            uni_pick["_sn"] = uni_pick["name"].fillna("").astype(str).str.strip().str.lower()
            uni_pick = uni_pick.sort_values(["_sn", "ticker_Yahoo"]).drop(columns=["_sn"])
            yahoo_ordered = uni_pick["ticker_Yahoo"].tolist()
            cat_options = [_CAT_PLACEHOLDER] + yahoo_ordered

            def _fmt_cat_row(opt: str) -> str:
                if opt == _CAT_PLACEHOLDER:
                    return _CAT_PLACEHOLDER
                r = uni.loc[uni["ticker_Yahoo"] == opt].iloc[0]
                nm = str(r.get("name") or "").strip() or "—"
                t = str(r.get("ticker") or "").strip()
                o = str(r.get("Origen") or "").strip()
                y = str(opt).strip()
                bits = [nm, y]
                if t and t != y:
                    bits.append(t)
                tail = f" ({o})" if o else ""
                return " · ".join(bits) + tail

            sel = st.selectbox(
                "Instrumento (busca por nombre, ticker o Yahoo)",
                cat_options,
                index=0,
                format_func=_fmt_cat_row,
                key="cat_instr_sel",
            )
            if sel != _CAT_PLACEHOLDER:
                row = uni.loc[uni["ticker_Yahoo"] == sel].iloc[0]
                c1, c2 = st.columns(2)
                with c1:
                    ny = st.text_input("Ticker Yahoo", value=sel, key=f"cat_y_{sel}")
                    nt = st.text_input("Ticker", value=str(row.get("ticker") or ""), key=f"cat_t_{sel}")
                with c2:
                    nn = st.text_input("Nombre", value=str(row.get("name") or ""), key=f"cat_n_{sel}")
                    ni = st.text_input("ISIN (opcional)", value=str(row.get("ISIN") or ""), key=f"cat_i_{sel}")
                if st.button("Guardar cambios", type="primary", key="cat_save"):
                    ok, msg = apply_global_instrument_update(sel, ny.strip(), nt.strip(), nn.strip(), ni.strip())
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            else:
                st.info("Selecciona un instrumento en el desplegable para cargar y editar sus datos.")
            st.subheader("Vista previa")
            st.dataframe(
                uni.sort_values("ticker_Yahoo").reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )
        return

    if vista == "Movimientos":
        st.header("Movimientos")

        # --- Catálogos para formulario nueva operación ---
        catalog = get_ticker_catalog(df)
        df_fondos_mov = load_data_fondos()
        catalog_fondos = get_ticker_catalog(df_fondos_mov) if df_fondos_mov is not None and not df_fondos_mov.empty else pd.DataFrame()
        df_crip_mov = load_data_criptos()
        catalog_criptos = get_ticker_catalog_criptos(df_crip_mov) if df_crip_mov is not None and not df_crip_mov.empty else pd.DataFrame()
        brokers_list = get_brokers_list()
        if not brokers_list and "broker" in df.columns:
            brokers_list = sorted(df["broker"].dropna().astype(str).unique().tolist())
        catalogs_currencies = []
        if not catalog.empty and "positionCurrency" in catalog.columns:
            catalogs_currencies.extend(catalog["positionCurrency"].dropna().astype(str).str.strip().unique().tolist())
        if not catalog_fondos.empty and "positionCurrency" in catalog_fondos.columns:
            catalogs_currencies.extend(catalog_fondos["positionCurrency"].dropna().astype(str).str.strip().unique().tolist())
        currencies_in_data = sorted(set(catalogs_currencies) | {"EUR"}) if catalogs_currencies else ["EUR", "USD", "GBP", "CHF"]

        tab_mov, tab_div = st.tabs(["Movimientos", "Dividendos"])

        with tab_mov:
            with st.expander("➕ Nueva operación", expanded=False):
                # Paso 1: ¿Qué quieres registrar?
                tipo_registro = st.radio(
                    "¿Qué quieres registrar?",
                    ["Acciones/ETFs", "Fondos", "Criptos", "Otros"],
                    index=0,
                    horizontal=True,
                    key="tipo_registro_nuevo",
                )
                if tipo_registro == "Acciones/ETFs":
                    position_type_base = "stock"
                    catalog_activo = catalog
                    tipo_nuevo = st.selectbox("Tipo de activo", ["Acción", "ETF"], key="new_tipo")
                    position_type = "stock" if tipo_nuevo == "Acción" else "etf"
                elif tipo_registro == "Fondos":
                    position_type_base = "fund"
                    catalog_activo = catalog_fondos
                elif tipo_registro == "Otros":
                    position_type_base = "warrant"
                    catalog_activo = catalog
                    if not catalog.empty and "positionType" in catalog.columns:
                        catalog_activo = catalog[catalog["positionType"].astype(str).str.strip().str.lower() == "warrant"].copy()
                    position_type = "warrant"
                else:
                    position_type_base = "crypto"
                    catalog_activo = catalog_criptos
                    position_type = "crypto"

                # Paso 2: Tipo de operación (primero, porque define qué campos se muestran)
                if tipo_registro in ("Acciones/ETFs", "Otros"):
                    op_options = [("buy", "Compra"), ("sell", "Venta"), ("dividend", "Dividendo"), ("split", "Split"), ("brokerTransfer", "Transferencia entre brokers")]
                elif tipo_registro == "Fondos":
                    op_options = [("buy", "Compra"), ("sell", "Venta"), ("traspaso_fondos", "Traspaso")]
                else:
                    op_options = [("buy", "Compra"), ("sell", "Venta"), ("switch", "Permuta"), ("stakeReward", "Stake Reward"), ("brokerTransfer", "Transferencia a wallet")]

                op_type = st.selectbox(
                    "Tipo de operación",
                    options=[o[0] for o in op_options],
                    format_func=lambda x: dict(op_options).get(x, x),
                    key="op_type_nuevo",
                )

                # Formularios con selector propio (sin Posición genérica): traspaso, transferencias, permuta
                _form_propio = (
                    (tipo_registro == "Fondos" and op_type == "traspaso_fondos")
                    or (tipo_registro in ("Acciones/ETFs", "Otros") and op_type == "brokerTransfer")
                    or (tipo_registro == "Criptos" and op_type == "brokerTransfer")
                    or (tipo_registro == "Criptos" and op_type == "switch")
                )

                if not _form_propio:
                    # Paso 3: Posición (solo para compra, venta, dividendo, split, stake reward)
                    st.caption("Elige una posición existente o crea una nueva.")
                    pos_origen = st.radio(
                        "¿La posición ya existe?",
                        ["Sí, elegir de la lista", "No, es una posición nueva"],
                        index=0,
                        horizontal=True,
                        key="pos_existente_o_nueva",
                    )

                    position_currency = "EUR"
                    position_ticker = position_yahoo = position_name = position_exchange = position_country = ""
                    if tipo_registro not in ("Acciones/ETFs", "Otros"):
                        position_type = position_type_base

                    if pos_origen == "No, es una posición nueva":
                        nc1, nc2, nc3 = st.columns(3)
                        with nc1:
                            ticker_placeholder = "Ej: BTC, ETH…" if tipo_registro == "Criptos" else "AAPL, ES01234567890…"
                            position_ticker = st.text_input("Ticker", key="new_ticker", placeholder=ticker_placeholder)
                            if tipo_registro != "Criptos":
                                position_yahoo = st.text_input("Ticker Yahoo", key="new_yahoo", placeholder="Para cotizaciones; puede ser = ticker (vacío si no hay)")
                        with nc2:
                            position_name = st.text_input("Nombre del activo", key="new_name", placeholder="Ej. Apple Inc.")
                            position_currency = st.selectbox("Moneda", currencies_in_data, key="new_ccy")
                        with nc3:
                            if tipo_registro == "Fondos":
                                position_type = "fund"
                                st.caption("Fondo")
                            elif tipo_registro == "Criptos":
                                position_type = "crypto"
                                st.caption("Cripto")
                            elif tipo_registro == "Otros":
                                st.caption("Otros (warrants, etc.)")
                            else:
                                st.caption(f"Tipo: **{tipo_nuevo}**")
                            if tipo_registro != "Fondos":
                                position_exchange = st.text_input("Bolsa (opcional)", key="new_exchange", placeholder="XETRA, NASDAQ…")
                                position_country = st.text_input("País (opcional)", key="new_country", placeholder="DE, US…")
                    else:
                        ticker_options = ["—— Elige posición ——"]
                        option_to_catalog = []
                        if catalog_activo.empty and tipo_registro == "Criptos":
                            st.info("No hay criptos en cartera. Elige «posición nueva» para registrar tu primera operación.")
                        if catalog_activo.empty and tipo_registro == "Otros":
                            st.info("No hay posiciones de Otros en cartera. Elige «posición nueva» para registrar tu primera operación.")
                        if not catalog_activo.empty:
                            for idx, (_, r) in enumerate(catalog_activo.iterrows()):
                                lab = f"{r['ticker']} | {r['name']} ({r.get('positionCurrency', 'EUR')})"
                                if tipo_registro == "Fondos":
                                    lab += " [Fondo]"
                                elif tipo_registro == "Criptos":
                                    lab += " [Cripto]"
                                elif tipo_registro in ("Acciones/ETFs", "Otros"):
                                    pt = str(r.get("positionType", "")).strip().lower()
                                    if pt == "etf":
                                        lab += " [ETF]"
                                    elif pt == "warrant":
                                        lab += " [Otros]"
                                    else:
                                        lab += " [Acción]"
                                ticker_options.append(lab)
                                option_to_catalog.append(idx)
                        sel_pos = st.selectbox("Posición", ticker_options, key="sel_pos_nuevo")
                        if sel_pos and sel_pos != "—— Elige posición ——" and option_to_catalog and not catalog_activo.empty:
                            idx_opt = ticker_options.index(sel_pos) - 1
                            if 0 <= idx_opt < len(catalog_activo):
                                r = catalog_activo.iloc[idx_opt]
                                position_currency = str(r.get("positionCurrency", "EUR")) if pd.notna(r.get("positionCurrency")) else "EUR"
                                position_ticker = str(r["ticker"]) if pd.notna(r["ticker"]) else ""
                                position_yahoo = str(r.get("ticker_Yahoo", position_ticker)) if pd.notna(r.get("ticker_Yahoo")) else position_ticker
                                position_name = str(r.get("name", position_ticker)) if pd.notna(r.get("name")) else position_ticker
                                position_exchange = str(r.get("positionExchange", "")) if pd.notna(r.get("positionExchange")) else ""
                                position_country = str(r.get("positionCountry", "")) if pd.notna(r.get("positionCountry")) else ""
                                if "positionType" in r and pd.notna(r["positionType"]):
                                    position_type = str(r["positionType"]).strip().lower()
                                else:
                                    position_type = position_type_base
                            st.caption(f"Moneda: **{position_currency}** · Tipo: **{position_type}**")

                    sel_pos = st.session_state.get("sel_pos_nuevo", "—— Elige posición ——") if pos_origen == "Sí, elegir de la lista" else "➕ Nueva posición"
                    es_posicion_nueva = pos_origen == "No, es una posición nueva"

                    default_ccy_for_fees = position_currency or "EUR"
                    if "last_pos_for_ccy" not in st.session_state or st.session_state["last_pos_for_ccy"] != (sel_pos + tipo_registro):
                        st.session_state["ccy_com"] = default_ccy_for_fees
                        st.session_state["ccy_tax"] = default_ccy_for_fees
                        st.session_state["last_pos_for_ccy"] = sel_pos + tipo_registro
                else:
                    position_currency = "EUR"
                    position_ticker = position_yahoo = position_name = position_exchange = position_country = ""
                    if tipo_registro not in ("Acciones/ETFs", "Otros"):
                        position_type = position_type_base
                    sel_pos = "——"
                    es_posicion_nueva = False

                # --- Formulario específico: Traspaso entre fondos (solo Fondos) ---
                if tipo_registro == "Fondos" and op_type == "traspaso_fondos":
                    st.caption("Genera dos movimientos en Fondos: salida del fondo origen y entrada en el fondo destino (coste arrastrado, no tributa).")
                    fondos_options = ["—— Elige fondo origen ——"]
                    fondos_dest_options = ["—— Elige fondo destino ——", "➕ Nuevo fondo destino"]
                    if not catalog_fondos.empty:
                        for _, r in catalog_fondos.iterrows():
                            lab = f"{r['ticker']} | {r['name']}"
                            fondos_options.append(lab)
                            fondos_dest_options.append(lab)
                    tf_c1, tf_c2 = st.columns(2)
                    with tf_c1:
                        tf_origen = st.selectbox("Fondo origen", fondos_options, key="tf_origen")
                        tf_qty = st.text_input("Participaciones", placeholder="0 o 0,00", key="tf_qty")
                        tf_fecha = st.date_input("Fecha", key="tf_fecha")
                        tf_broker = st.selectbox("Cuenta (broker)", options=brokers_list, key="tf_broker") if brokers_list else st.text_input("Cuenta (broker)", key="tf_broker")
                    with tf_c2:
                        tf_destino = st.selectbox("Fondo destino", fondos_dest_options, key="tf_destino")
                        tf_destino_nuevo_ticker = ""
                        tf_destino_nuevo_name = ""
                        if tf_destino == "➕ Nuevo fondo destino":
                            tf_destino_nuevo_ticker = st.text_input("ISIN / Ticker del fondo destino", key="tf_dest_ticker", placeholder="ES01234567890")
                            tf_destino_nuevo_name = st.text_input("Nombre del fondo destino", key="tf_dest_name", placeholder="Nombre del fondo")
                        tf_valor_eur = st.text_input("Valor reembolso (EUR)", placeholder="0 o 0,00", key="tf_valor_eur", help="Valor en euros del reembolso en el fondo origen")
                        tf_hora = st.time_input("Hora", value=dt_time(12, 0), key="tf_hora", step=60)
                    if st.button("Guardar traspaso entre fondos", type="primary", key="guardar_traspaso_fondos"):
                        if tf_origen == "—— Elige fondo origen ——":
                            st.error("Elige el fondo origen.")
                        elif tf_destino == "—— Elige fondo destino ——":
                            st.error("Elige el fondo destino o «Nuevo fondo destino».")
                        elif tf_destino == "➕ Nuevo fondo destino" and (not tf_destino_nuevo_ticker or not tf_destino_nuevo_ticker.strip()):
                            st.error("Indica el ISIN o ticker del fondo destino.")
                        elif not tf_qty or _to_float(tf_qty, 0.0) <= 0:
                            st.error("Indica la cantidad de participaciones.")
                        else:
                            idx_o = fondos_options.index(tf_origen) - 1
                            if idx_o < 0 or catalog_fondos.empty or idx_o >= len(catalog_fondos):
                                st.error("Fondo origen no encontrado en el catálogo.")
                            else:
                                ro = catalog_fondos.iloc[idx_o]
                                ticker_d = name_d = None
                                if tf_destino == "➕ Nuevo fondo destino":
                                    ticker_d = (tf_destino_nuevo_ticker or "").strip()
                                    name_d = (tf_destino_nuevo_name or "").strip() or ticker_d
                                else:
                                    idx_d = fondos_dest_options.index(tf_destino) - 2
                                    if idx_d < 0 or idx_d >= len(catalog_fondos):
                                        st.error("Fondo destino no encontrado.")
                                    else:
                                        rd = catalog_fondos.iloc[idx_d]
                                        ticker_d = str(rd.get("ticker") or rd.get("ticker_Yahoo") or "")
                                        name_d = str(rd.get("name") or ticker_d)
                                if ticker_d:
                                    ticker_o = str(ro.get("ticker") or ro.get("ticker_Yahoo") or "")
                                    name_o = str(ro.get("name") or ticker_o)
                                    qty_val = _to_float(tf_qty, 0.0)
                                    valor_eur = _to_float(tf_valor_eur, 0.0)
                                    date_str = tf_fecha.strftime("%Y-%m-%d") if hasattr(tf_fecha, "strftime") else str(tf_fecha)
                                    time_str = tf_hora.strftime("%H:%M:%S") if hasattr(tf_hora, "strftime") else "12:00:00"
                                    row_switch = {
                                        "date": date_str, "time": time_str,
                                        "ticker": ticker_o, "ticker_Yahoo": ro.get("ticker_Yahoo") or ticker_o, "name": name_o,
                                        "positionType": "fund", "positionCountry": "", "positionCurrency": "EUR", "positionExchange": "",
                                        "broker": tf_broker, "type": "switch",
                                        "positionNumber": qty_val, "price": valor_eur / qty_val if qty_val else 0,
                                        "comission": 0, "comissionCurrency": "EUR", "destinationRetentionBaseCurrency": "", "taxes": 0, "taxesCurrency": "EUR",
                                        "exchangeRate": 1.0, "positionQuantity": "", "autoFx": "No",
                                        "switchBuyPosition": ticker_d,
                                        "switchBuyPositionType": "", "switchBuyPositionNumber": "", "switchBuyExchangeRate": "", "switchBuyBroker": "",
                                        "spinOffBuyPosition": "", "spinOffBuyPositionNumber": "", "spinOffBuyPositionAllocation": "",
                                        "brokerTransferNewBroker": "",
                                        "total": valor_eur, "totalBaseCurrency": valor_eur, "totalWithComission": valor_eur, "totalWithComissionBaseCurrency": valor_eur,
                                    }
                                    row_switchbuy = {
                                        "date": date_str, "time": time_str,
                                        "ticker": ticker_d, "ticker_Yahoo": ticker_d, "name": name_d,
                                        "positionType": "fund", "positionCountry": "", "positionCurrency": "EUR", "positionExchange": "",
                                        "broker": tf_broker, "type": "switchBuy",
                                        "positionNumber": qty_val, "price": valor_eur / qty_val if qty_val else 0,
                                        "comission": 0, "comissionCurrency": "EUR", "destinationRetentionBaseCurrency": "", "taxes": 0, "taxesCurrency": "EUR",
                                        "exchangeRate": 1.0, "positionQuantity": "", "autoFx": "No",
                                        "switchBuyPosition": "", "switchBuyPositionType": "", "switchBuyPositionNumber": "", "switchBuyExchangeRate": "", "switchBuyBroker": "",
                                        "spinOffBuyPosition": "", "spinOffBuyPositionNumber": "", "spinOffBuyPositionAllocation": "",
                                        "brokerTransferNewBroker": "",
                                        "total": valor_eur, "totalBaseCurrency": valor_eur, "totalWithComission": valor_eur, "totalWithComissionBaseCurrency": valor_eur,
                                    }
                                    try:
                                        append_operation_fondos(row_switch)
                                        append_operation_fondos(row_switchbuy)
                                        load_data_fondos.clear()
                                        _clear_form_nueva_operacion()
                                        st.success("Traspaso guardado (switch + switchBuy en Fondos).")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error al guardar: {e}")
                # --- Formulario específico: Transferencia entre brokers (solo Acciones/ETFs) ---
                elif tipo_registro in ("Acciones/ETFs", "Otros") and op_type == "brokerTransfer":
                    st.caption("Transfiere títulos de una cuenta (broker) a otra. El coste se arrastra, no hay tributo.")
                    bt_pos_options = ["—— Elige posición ——"]
                    if not catalog_activo.empty:
                        for idx, (_, r) in enumerate(catalog_activo.iterrows()):
                            lab = f"{r.get('ticker', '')} | {r.get('name', '')}"
                            pt = str(r.get("positionType", "stock")).strip().lower()
                            lab += " [ETF]" if pt == "etf" else (" [Otros]" if pt == "warrant" else " [Acción]")
                            bt_pos_options.append(lab)
                    bt_c1, bt_c2 = st.columns(2)
                    with bt_c1:
                        bt_broker_origen = st.selectbox("Broker origen", options=brokers_list, key="bt_broker_origen") if brokers_list else st.text_input("Broker origen", key="bt_broker_origen")
                        bt_posicion = st.selectbox("Posición a transferir", bt_pos_options, key="bt_posicion")
                        bt_fecha = st.date_input("Fecha", key="bt_fecha")
                    with bt_c2:
                        bt_broker_destino = st.selectbox("Broker destino", options=brokers_list, key="bt_broker_destino") if brokers_list else st.text_input("Broker destino", key="bt_broker_destino")
                        bt_qty = st.text_input("Cantidad a transferir", placeholder="0 o 0,0000", key="bt_qty")
                        bt_hora = st.time_input("Hora", value=dt_time(12, 0), key="bt_hora", step=60)
                    if st.button("Guardar transferencia entre brokers", type="primary", key="guardar_bt"):
                        if bt_posicion == "—— Elige posición ——":
                            st.error("Elige la posición a transferir.")
                        elif not bt_qty or _to_float(bt_qty, 0.0) <= 0:
                            st.error("Indica la cantidad a transferir.")
                        elif not bt_broker_origen or not str(bt_broker_origen).strip():
                            st.error("Indica el broker origen.")
                        elif not bt_broker_destino or not str(bt_broker_destino).strip():
                            st.error("Indica el broker destino.")
                        elif str(bt_broker_origen).strip() == str(bt_broker_destino).strip():
                            st.error("El broker origen y destino deben ser diferentes.")
                        else:
                            idx_bt = bt_pos_options.index(bt_posicion) - 1
                            if idx_bt < 0 or catalog_activo.empty or idx_bt >= len(catalog_activo):
                                st.error("Posición no encontrada en el catálogo.")
                            else:
                                ro = catalog_activo.iloc[idx_bt]
                                ticker_o = str(ro.get("ticker") or ro.get("ticker_Yahoo") or "")
                                name_o = str(ro.get("name") or ticker_o)
                                ticker_yahoo = str(ro.get("ticker_Yahoo") or ticker_o)
                                pt = str(ro.get("positionType", "stock")).strip().lower()
                                position_type = pt if pt in ("stock", "etf", "warrant") else "stock"
                                qty_val = _to_float(bt_qty, 0.0)
                                date_str = bt_fecha.strftime("%Y-%m-%d") if hasattr(bt_fecha, "strftime") else str(bt_fecha)
                                time_str = bt_hora.strftime("%H:%M:%S") if hasattr(bt_hora, "strftime") else "12:00:00"
                                row_bt = {
                                    "date": date_str, "time": time_str,
                                    "ticker": ticker_o, "ticker_Yahoo": ticker_yahoo, "name": name_o,
                                    "positionType": position_type, "positionCountry": ro.get("positionCountry", ""), "positionCurrency": ro.get("positionCurrency", "EUR"), "positionExchange": ro.get("positionExchange", ""),
                                    "broker": str(bt_broker_origen).strip(),
                                    "type": "brokerTransfer",
                                    "positionNumber": qty_val, "price": 0,
                                    "comission": 0, "comissionCurrency": "EUR", "destinationRetentionBaseCurrency": "", "taxes": 0, "taxesCurrency": "EUR",
                                    "exchangeRate": 1.0, "positionQuantity": "", "autoFx": "No",
                                    "switchBuyPosition": "", "switchBuyPositionType": "", "switchBuyPositionNumber": "", "switchBuyExchangeRate": "", "switchBuyBroker": "",
                                    "spinOffBuyPosition": "", "spinOffBuyPositionNumber": "", "spinOffBuyPositionAllocation": "",
                                    "brokerTransferNewBroker": str(bt_broker_destino).strip(),
                                    "total": 0, "totalBaseCurrency": 0, "totalWithComission": 0, "totalWithComissionBaseCurrency": 0,
                                }
                                try:
                                    append_operation(row_bt)
                                    load_data.clear()
                                    _clear_form_nueva_operacion()
                                    st.success("Transferencia entre brokers guardada.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error al guardar: {e}")
                # --- Formulario específico: Transferencia cripto a wallet (brokerTransfer) ---
                elif tipo_registro == "Criptos" and op_type == "brokerTransfer":
                    st.caption("Transfiere cripto de una cuenta (broker/wallet) a otra. La comisión, si la hay, se descuenta en origen.")
                    ctf_options = ["—— Elige cripto ——"]
                    if not catalog_criptos.empty:
                        for _, r in catalog_criptos.iterrows():
                            ctf_options.append(f"{r['ticker']} | {r['name']}")
                    ctf_crypto = st.selectbox("Cripto", ctf_options, key="ctf_crypto")
                    ctf_c1, ctf_c2 = st.columns(2)
                    with ctf_c1:
                        ctf_broker_origen = st.selectbox("Broker/wallet origen", options=brokers_list, key="ctf_broker_origen") if brokers_list else st.text_input("Broker/wallet origen", key="ctf_broker_origen")
                        ctf_qty = st.text_input("Cantidad recibida en destino", placeholder="0,00", key="ctf_qty", help="Lo que efectivamente llega a la wallet destino")
                        ctf_fecha = st.date_input("Fecha", key="ctf_fecha")
                    with ctf_c2:
                        ctf_broker_destino = st.selectbox("Broker/wallet destino", options=brokers_list, key="ctf_broker_destino") if brokers_list else st.text_input("Broker/wallet destino", key="ctf_broker_destino")
                        ctf_comision = st.text_input("Comisión (opcional)", placeholder="0 o 0,00", key="ctf_comision", help="Fee de red o del exchange; se descuenta en origen")
                        ctf_hora = st.time_input("Hora", value=dt_time(12, 0), key="ctf_hora", step=60)
                    if st.button("Guardar transferencia", type="primary", key="guardar_ctf"):
                        if ctf_crypto == "—— Elige cripto ——":
                            st.error("Elige la cripto a transferir.")
                        elif not ctf_qty or _to_float(ctf_qty, 0.0) <= 0:
                            st.error("Indica la cantidad recibida en destino.")
                        elif not ctf_broker_origen or not str(ctf_broker_origen).strip():
                            st.error("Indica el broker/wallet origen.")
                        elif not ctf_broker_destino or not str(ctf_broker_destino).strip():
                            st.error("Indica el broker/wallet destino.")
                        elif str(ctf_broker_origen).strip() == str(ctf_broker_destino).strip():
                            st.error("Origen y destino deben ser diferentes.")
                        else:
                            idx_ctf = ctf_options.index(ctf_crypto) - 1
                            if idx_ctf < 0 or catalog_criptos.empty or idx_ctf >= len(catalog_criptos):
                                st.error("Cripto no encontrada en el catálogo.")
                            else:
                                ro = catalog_criptos.iloc[idx_ctf]
                                ticker = str(ro.get("ticker") or ro.get("ticker_Yahoo") or "")
                                name = str(ro.get("name") or ticker)
                                ticker_yahoo = _crypto_ticker_yahoo(ticker, ro.get("ticker_Yahoo") or "")
                                qty_val = _to_float(ctf_qty, 0.0)
                                com_val = _to_float(ctf_comision, 0.0)
                                date_str = ctf_fecha.strftime("%Y-%m-%d") if hasattr(ctf_fecha, "strftime") else str(ctf_fecha)
                                time_str = ctf_hora.strftime("%H:%M:%S") if hasattr(ctf_hora, "strftime") else "12:00:00"
                                broker_orig = str(ctf_broker_origen).strip()
                                broker_dest = str(ctf_broker_destino).strip()
                                def _ctf_row(qty):
                                    return {
                                        "date": date_str, "time": time_str,
                                        "ticker": ticker, "ticker_Yahoo": ticker_yahoo, "name": name,
                                        "positionType": "crypto", "positionCountry": "", "positionCurrency": "EUR", "positionExchange": "",
                                        "broker": broker_orig, "type": "brokerTransfer", "positionNumber": qty, "price": 0,
                                        "comission": 0, "comissionCurrency": "", "destinationRetentionBaseCurrency": "", "taxes": 0, "taxesCurrency": "EUR",
                                        "exchangeRate": 1.0, "positionQuantity": "", "autoFx": "No",
                                        "switchBuyPosition": "", "switchBuyPositionType": "", "switchBuyPositionNumber": "", "switchBuyExchangeRate": "", "switchBuyBroker": "",
                                        "spinOffBuyPosition": "", "spinOffBuyPositionNumber": "", "spinOffBuyPositionAllocation": "",
                                        "brokerTransferNewBroker": broker_dest,
                                        "total": 0, "totalBaseCurrency": 0, "totalWithComission": 0, "totalWithComissionBaseCurrency": 0,
                                        "positionCustomType": "", "description": "",
                                    }
                                try:
                                    if com_val > 0:
                                        row_comm = _ctf_row(com_val)
                                        row_comm["type"] = "commission"
                                        row_comm["brokerTransferNewBroker"] = ""
                                        append_operation_criptos(row_comm)
                                    row_bt = _ctf_row(qty_val)
                                    append_operation_criptos(row_bt)
                                    load_data_criptos.clear()
                                    _clear_form_nueva_operacion()
                                    st.success("Transferencia guardada." + (" (comisión + transferencia)" if com_val > 0 else ""))
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error al guardar: {e}")
                # --- Formulario específico: Permuta criptos (switch + switchBuy) ---
                elif tipo_registro == "Criptos" and op_type == "switch":
                    st.caption("Genera dos movimientos: salida de la cripto origen y entrada en la cripto destino.")
                    cripto_options = ["—— Elige cripto origen ——"]
                    cripto_dest_options = ["—— Elige cripto destino ——", "➕ Nueva cripto destino"]
                    if not catalog_criptos.empty:
                        for _, r in catalog_criptos.iterrows():
                            lab = f"{r['ticker']} | {r['name']}"
                            cripto_options.append(lab)
                            cripto_dest_options.append(lab)
                    pc1, pc2 = st.columns(2)
                    with pc1:
                        perm_origen = st.selectbox("Cripto origen", cripto_options, key="perm_origen")
                        perm_qty_origen = st.text_input("Cantidad origen", placeholder="0,00", key="perm_qty_origen")
                        perm_qty_destino = st.text_input("Cantidad destino", placeholder="0,00", key="perm_qty_destino", help="Cantidad que recibes de la cripto destino")
                        perm_valor_eur = st.text_input("Valor (€)", placeholder="0,00", key="perm_valor_eur")
                        perm_fecha = st.date_input("Fecha", key="perm_fecha")
                        perm_broker = st.selectbox("Cuenta", options=brokers_list, key="perm_broker") if brokers_list else st.text_input("Cuenta", key="perm_broker")
                    with pc2:
                        perm_destino = st.selectbox("Cripto destino", cripto_dest_options, key="perm_destino")
                        perm_dest_nuevo_ticker = ""
                        perm_dest_nuevo_name = ""
                        if perm_destino == "➕ Nueva cripto destino":
                            perm_dest_nuevo_ticker = st.text_input("Ticker", key="perm_dest_ticker", placeholder="Ej: BTC, ETH…")
                            perm_dest_nuevo_name = st.text_input("Nombre destino", key="perm_dest_name", placeholder="Ethereum")
                        perm_hora = st.time_input("Hora", value=dt_time(12, 0), key="perm_hora", step=60)
                    if st.button("Guardar permuta", type="primary", key="guardar_permuta"):
                        if perm_origen == "—— Elige cripto origen ——":
                            st.error("Elige la cripto origen.")
                        elif perm_destino == "—— Elige cripto destino ——":
                            st.error("Elige la cripto destino o «Nueva cripto destino».")
                        elif perm_destino == "➕ Nueva cripto destino" and (not perm_dest_nuevo_ticker or not perm_dest_nuevo_ticker.strip()):
                            st.error("Indica el ticker de la cripto destino.")
                        elif not perm_qty_origen or _to_float(perm_qty_origen, 0.0) <= 0:
                            st.error("Indica la cantidad origen.")
                        elif not perm_qty_destino or _to_float(perm_qty_destino, 0.0) <= 0:
                            st.error("Indica la cantidad destino.")
                        elif not perm_valor_eur or _to_float(perm_valor_eur, 0.0) <= 0:
                            st.error("Indica el valor en euros.")
                        else:
                            idx_o = cripto_options.index(perm_origen) - 1
                            if idx_o < 0 or catalog_criptos.empty or idx_o >= len(catalog_criptos):
                                st.error("Cripto origen no encontrada.")
                            else:
                                ro = catalog_criptos.iloc[idx_o]
                                ticker_d = name_d = ""
                                if perm_destino == "➕ Nueva cripto destino":
                                    ticker_d = (perm_dest_nuevo_ticker or "").strip()
                                    name_d = (perm_dest_nuevo_name or "").strip() or ticker_d
                                else:
                                    idx_d = cripto_dest_options.index(perm_destino) - 2
                                    if idx_d >= 0 and idx_d < len(catalog_criptos):
                                        rd = catalog_criptos.iloc[idx_d]
                                        ticker_d = str(rd.get("ticker") or rd.get("ticker_Yahoo") or "")
                                        name_d = str(rd.get("name") or ticker_d)
                                if ticker_d:
                                    ticker_o = str(ro.get("ticker") or ro.get("ticker_Yahoo") or "")
                                    name_o = str(ro.get("name") or ticker_o)
                                    ticker_yahoo_o = _crypto_ticker_yahoo(ticker_o, ro.get("ticker_Yahoo") or "")
                                    if perm_destino == "➕ Nueva cripto destino":
                                        ticker_yahoo_d = _crypto_ticker_yahoo(ticker_d, "")
                                    else:
                                        rd = catalog_criptos.iloc[cripto_dest_options.index(perm_destino) - 2]
                                        ticker_yahoo_d = _crypto_ticker_yahoo(ticker_d, rd.get("ticker_Yahoo") or "")
                                    qty_o = _to_float(perm_qty_origen, 0.0)
                                    qty_d = _to_float(perm_qty_destino, 0.0)
                                    valor_eur = _to_float(perm_valor_eur, 0.0)
                                    date_str = perm_fecha.strftime("%Y-%m-%d") if hasattr(perm_fecha, "strftime") else str(perm_fecha)
                                    time_str = perm_hora.strftime("%H:%M:%S") if hasattr(perm_hora, "strftime") else "12:00:00"
                                    row_switch = {
                                        "date": date_str, "time": time_str,
                                        "ticker": ticker_o, "ticker_Yahoo": ticker_yahoo_o, "name": name_o,
                                        "positionType": "crypto", "positionCountry": "", "positionCurrency": "EUR", "positionExchange": "",
                                        "broker": perm_broker, "type": "switch",
                                        "positionNumber": qty_o, "price": valor_eur / qty_o if qty_o else 0,
                                        "comission": 0, "comissionCurrency": "EUR", "destinationRetentionBaseCurrency": "", "taxes": 0, "taxesCurrency": "EUR",
                                        "exchangeRate": 1.0, "positionQuantity": "", "autoFx": "No",
                                        "switchBuyPosition": ticker_d,
                                        "switchBuyPositionType": "", "switchBuyPositionNumber": "", "switchBuyExchangeRate": "", "switchBuyBroker": "",
                                        "spinOffBuyPosition": "", "spinOffBuyPositionNumber": "", "spinOffBuyPositionAllocation": "",
                                        "brokerTransferNewBroker": "",
                                        "total": valor_eur, "totalBaseCurrency": valor_eur, "totalWithComission": valor_eur, "totalWithComissionBaseCurrency": valor_eur,
                                        "positionCustomType": "", "description": "",
                                    }
                                    row_switchbuy = {
                                        "date": date_str, "time": time_str,
                                        "ticker": ticker_d, "ticker_Yahoo": ticker_yahoo_d, "name": name_d,
                                        "positionType": "crypto", "positionCountry": "", "positionCurrency": "EUR", "positionExchange": "",
                                        "broker": perm_broker, "type": "switchBuy",
                                        "positionNumber": qty_d, "price": valor_eur / qty_d if qty_d else 0,
                                        "comission": 0, "comissionCurrency": "EUR", "destinationRetentionBaseCurrency": "", "taxes": 0, "taxesCurrency": "EUR",
                                        "exchangeRate": 1.0, "positionQuantity": "", "autoFx": "No",
                                        "switchBuyPosition": "", "switchBuyPositionType": "", "switchBuyPositionNumber": "", "switchBuyExchangeRate": "", "switchBuyBroker": "",
                                        "spinOffBuyPosition": "", "spinOffBuyPositionNumber": "", "spinOffBuyPositionAllocation": "",
                                        "brokerTransferNewBroker": "",
                                        "total": valor_eur, "totalBaseCurrency": valor_eur, "totalWithComission": valor_eur, "totalWithComissionBaseCurrency": valor_eur,
                                        "positionCustomType": "", "description": "",
                                    }
                                    try:
                                        append_operation_criptos(row_switch)
                                        append_operation_criptos(row_switchbuy)
                                        load_data_criptos.clear()
                                        _clear_form_nueva_operacion()
                                        st.success("Permuta guardada (switch + switchBuy en Criptos).")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error al guardar: {e}")
                else:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        op_date = st.date_input("Fecha", key="op_date_nuevo")
                    with c2:
                        op_time = st.time_input("Hora", value=dt_time(12, 0), key="op_time_nuevo", step=60, help="Hora de la operación (selección por minutos)")
                    with c3:
                        op_broker = st.selectbox("Cuenta (broker)", options=brokers_list, key="op_broker_nuevo") if brokers_list else st.text_input("Cuenta (broker)", key="op_broker_nuevo")

                    # Títulos y Precio/Total (text_input para que al hacer clic el cursor no quede en medio del número)
                    precio_o_total = st.radio("Introducir", ["Precio unitario", "Total"], horizontal=True, key="precio_o_total")
                    c_qty, c_val = st.columns(2)
                    with c_qty:
                        _qty_str = st.text_input("Títulos", placeholder="0 o 0,0000", key="op_qty_nuevo")
                        op_quantity = _to_float(_qty_str, 0.0)
                    with c_val:
                        if precio_o_total == "Precio unitario":
                            _price_str = st.text_input(f"Precio ({position_currency})", placeholder="0 o 0,00", key="op_precio_nuevo")
                            op_price = _to_float(_price_str, 0.0)
                            op_total_local = op_quantity * op_price if op_quantity else 0.0
                        else:
                            _total_str = st.text_input(f"Total ({position_currency})", placeholder="0 o 0,00", key="op_total_nuevo")
                            op_total_local = _to_float(_total_str, 0.0)
                            op_price = (op_total_local / op_quantity) if op_quantity else 0.0

                    # Modificar tipo de cambio: si está OFF se usa cierre del día automático; si está ON se muestra el campo y botones
                    mod_fx = st.toggle("Modificar tipo de cambio", value=True, key="mod_fx_nuevo", help="Desactivado: se usa el tipo de cambio de cierre del día de la fecha de la operación. Activado: puedes indicar o buscar el tipo de cambio.")
                    op_exchange_rate = 1.0
                    if position_currency == "EUR":
                        op_exchange_rate = 1.0
                    else:
                        if not mod_fx:
                            op_exchange_rate = get_fx_rate_for_date(position_currency, op_date)
                            if math.isnan(op_exchange_rate) or op_exchange_rate <= 0:
                                op_exchange_rate = 1.0
                            st.caption(f"Tipo de cambio de cierre del día: **{op_exchange_rate:.4f}** {position_currency}/EUR. Activa el switch para indicar otro valor.")
                        else:
                            if "op_fx_nuevo_pending" in st.session_state:
                                st.session_state["op_fx_nuevo"] = str(st.session_state["op_fx_nuevo_pending"]).replace(".", ",")
                                del st.session_state["op_fx_nuevo_pending"]
                            col_fx, col_btn1, col_btn2 = st.columns([2, 1, 1])
                            with col_fx:
                                _fx_str = st.text_input(
                                    f"Tipo de cambio ({position_currency}/EUR)",
                                    placeholder="1 o 0,92",
                                    help="Ej: 0,92 significa 1 " + position_currency + " = 0,92 EUR",
                                    key="op_fx_nuevo",
                                )
                                op_exchange_rate = _to_float(_fx_str, 1.0) if (_fx_str or "").strip() else 1.0
                            with col_btn1:
                                st.caption("")
                                if st.button("Cierre del día", key="btn_fx_cierre", help="Obtener tipo de cambio de cierre para la fecha de la operación (Yahoo Finance)"):
                                    rate = get_fx_rate_for_date(position_currency, op_date)
                                    if not math.isnan(rate) and rate > 0:
                                        st.session_state["op_fx_nuevo_pending"] = rate
                                        st.rerun()
                                    else:
                                        st.warning("No se pudo obtener el tipo de cambio para esa fecha. Introduce el valor a mano.")
                            with col_btn2:
                                st.caption("")
                                if st.button("A la hora de compra", key="btn_fx_intra", help="Obtener tipo de cambio aproximado en el momento de la operación (Yahoo Finance intradía)"):
                                    _hora_txt = op_time.strftime("%H:%M:%S") if hasattr(op_time, "strftime") else (str(op_time).strip() if op_time else "00:00:00")
                                    dt_txt = f"{op_date.strftime('%Y-%m-%d')} {_hora_txt}"
                                    rate = get_fx_rate_at_datetime(position_currency, dt_txt)
                                    if not math.isnan(rate) and rate > 0:
                                        st.session_state["op_fx_nuevo_pending"] = rate
                                        st.rerun()
                                    else:
                                        st.warning("No se pudo obtener el tipo de cambio intradía para ese momento. Prueba con cierre del día o introduce el valor a mano.")
                    auto_fx = st.toggle("AutoFx", value=False, help="Tipo de cambio automático del broker", key="auto_fx_nuevo")

                    ccy_options = currencies_in_data
                    cc1, cc2, cc3 = st.columns(3)
                    with cc1:
                        _com_str = st.text_input("Comisión", placeholder="0 o 0,00", key="op_com_nuevo")
                        op_commission = _to_float(_com_str, 0.0)
                        op_commission_ccy = st.selectbox("Moneda comisión", ccy_options, key="ccy_com")
                    with cc2:
                        _tax_str = st.text_input("Impuestos (Tasa Tobin, Stamp Duty, etc.)", placeholder="0 o 0,00", key="op_tax_nuevo")
                        op_taxes = _to_float(_tax_str, 0.0)
                        op_taxes_ccy = st.selectbox("Moneda impuestos", ccy_options, key="ccy_tax")
                    with cc3:
                        _dest_str = st.text_input("Retención en destino (€)", placeholder="0 o 0,00", key="op_dest_nuevo")
                        op_dest_ret = _to_float(_dest_str, 0.0)

                    # Previsualizar totales (misma lógica que al guardar, vía _recalc_totals)
                    _prev = _recalc_totals(
                        float(op_quantity or 0),
                        float(op_price or 0),
                        float(op_commission or 0),
                        float(op_taxes or 0),
                        float(op_exchange_rate or 1.0),
                        str(position_currency or "EUR"),
                        str(op_commission_ccy or ""),
                        str(op_taxes_ccy or ""),
                        tipo=str(op_type or ""),
                    )
                    _fmt_prev = lambda v: f"{v:,.2f}".replace(",", " ").replace(".", ",")

                    st.subheader("Previsualizar totales")
                    if str(position_currency or "EUR").strip().upper() == "EUR":
                        prev1, prev2 = st.columns(2)
                        with prev1:
                            st.metric("Total (EUR)", _fmt_prev(_prev["totalBaseCurrency"]) + " €")
                        with prev2:
                            st.metric("Total + com. + imp. (EUR)", _fmt_prev(_prev["totalWithComissionBaseCurrency"]) + " €")
                    else:
                        prev1, prev2, prev3, prev4 = st.columns(4)
                        with prev1:
                            st.metric(f"Total ({position_currency})", _fmt_prev(_prev["total"]))
                        with prev2:
                            st.metric(
                                f"Total + com. + imp. ({position_currency})",
                                _fmt_prev(_prev["totalWithComission"]),
                            )
                        with prev3:
                            st.metric("Total (EUR)", _fmt_prev(_prev["totalBaseCurrency"]) + " €")
                        with prev4:
                            st.metric(
                                "Total + com. + imp. (EUR)",
                                _fmt_prev(_prev["totalWithComissionBaseCurrency"]) + " €",
                            )

                    if st.button("Guardar operación", type="primary", key="guardar_nuevo"):
                        if not es_posicion_nueva and (not sel_pos or sel_pos == "—— Elige posición ——"):
                            st.error("Elige una posición de la lista.")
                        elif es_posicion_nueva and (not (position_ticker or position_yahoo)):
                            st.error("Indica al menos el ticker o el ticker Yahoo para la posición nueva.")
                        else:
                            if precio_o_total == "Total":
                                total_local = op_total_local
                                op_price = (op_total_local / op_quantity) if op_quantity else 0.0
                            else:
                                total_local = op_quantity * op_price if op_quantity else 0.0
                            recalc = _recalc_totals(
                                float(op_quantity or 0),
                                float(op_price or 0),
                                float(op_commission or 0),
                                float(op_taxes or 0),
                                float(op_exchange_rate or 1.0),
                                str(position_currency or "EUR"),
                                str(op_commission_ccy or ""),
                                str(op_taxes_ccy or ""),
                                tipo=str(op_type or ""),
                            )
                            total_local = recalc["total"]
                            total_base = recalc["totalBaseCurrency"]
                            total_with_comm_local = recalc["totalWithComission"]
                            total_with_comm_base = recalc["totalWithComissionBaseCurrency"]

                            if hasattr(op_time, "strftime"):
                                time_str = op_time.strftime("%H:%M:%S")
                            else:
                                _t = str(op_time).strip() if op_time else "00:00:00"
                                if ":" in _t:
                                    parts = _t.split(":")
                                    time_str = f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:00" if len(parts) == 2 else f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{str(parts[2]).zfill(2)}"
                                else:
                                    time_str = _t or "00:00:00"
                            date_str = op_date.strftime("%Y-%m-%d") if hasattr(op_date, "strftime") else str(op_date)

                            new_row = {
                            "date": date_str,
                            "time": time_str,
                            "ticker": position_ticker or position_yahoo,
                            "ticker_Yahoo": position_yahoo or position_ticker,
                            "name": position_name or position_ticker,
                            "positionType": position_type,
                            "positionCountry": position_country or "",
                            "positionCurrency": position_currency,
                            "positionExchange": position_exchange or "",
                            "broker": op_broker,
                            "type": op_type,
                            "positionNumber": op_quantity,
                            "price": op_price,
                            "comission": op_commission,
                            "comissionCurrency": op_commission_ccy,
                            "destinationRetentionBaseCurrency": op_dest_ret if op_dest_ret else "",
                            "taxes": op_taxes,
                            "taxesCurrency": op_taxes_ccy,
                            "exchangeRate": op_exchange_rate,
                            "positionQuantity": "",
                            "autoFx": "Yes" if auto_fx else "No",
                            "switchBuyPosition": "",
                            "switchBuyPositionType": "",
                            "switchBuyPositionNumber": "",
                            "switchBuyExchangeRate": "",
                            "switchBuyBroker": "",
                            "spinOffBuyPosition": "",
                            "spinOffBuyPositionNumber": "",
                            "spinOffBuyPositionAllocation": "",
                            "brokerTransferNewBroker": "",
                            "total": total_local,
                            "totalBaseCurrency": total_base,
                            "totalWithComission": total_with_comm_local,
                            "totalWithComissionBaseCurrency": total_with_comm_base,
                            }
                            try:
                                if position_type == "fund":
                                    append_operation_fondos(new_row)
                                    load_data_fondos.clear()
                                elif position_type == "crypto":
                                    row_crip = {c: new_row.get(c, "") for c in MOVIMIENTOS_COLUMNS}
                                    row_crip["positionType"] = "crypto"
                                    row_crip["positionCustomType"] = ""
                                    row_crip["description"] = ""
                                    yahoo_arg = "" if es_posicion_nueva else row_crip.get("ticker_Yahoo", "")
                                    row_crip["ticker_Yahoo"] = _crypto_ticker_yahoo(row_crip.get("ticker", ""), yahoo_arg)
                                    append_operation_criptos(row_crip)
                                    load_data_criptos.clear()
                                else:
                                    append_operation(new_row)
                                    load_data.clear()
                                if "nuevo_form_abierto" in st.session_state:
                                    st.session_state["nuevo_form_abierto"] = False
                                _clear_form_nueva_operacion()
                                st.success("Operación guardada.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error al guardar: {e}")

            df_fondos_mov = load_data_fondos()
            mov_acc = df.copy()
            mov_acc["origen"] = "Acciones"
            mov_acc["_acc_idx"] = df.index
            to_concat: list[pd.DataFrame] = [mov_acc]
            if df_fondos_mov is not None and not df_fondos_mov.empty:
                mov_fondos = df_fondos_mov.copy()
                if "name" not in mov_fondos.columns and "nombre" in mov_fondos.columns:
                    mov_fondos["name"] = mov_fondos["nombre"]
                mov_fondos["origen"] = "Fondos"
                mov_fondos["_acc_idx"] = pd.NA
                to_concat.append(mov_fondos)
            # Añadimos movimientos de cripto
            df_crip_mov = load_data_criptos()
            if df_crip_mov is not None and not df_crip_mov.empty:
                mov_crip = df_crip_mov.copy()
                if "name" not in mov_crip.columns and "nombre" in mov_crip.columns:
                    mov_crip["name"] = mov_crip["nombre"]
                mov_crip["origen"] = "Criptos"
                mov_crip["_acc_idx"] = pd.NA
                to_concat.append(mov_crip)

            mov = pd.concat(to_concat, ignore_index=True)
            if "datetime_full" in mov.columns:
                mov = mov.sort_values("datetime_full", ascending=False)

            col_filtro, col_refresh = st.columns([4, 1])
            with col_filtro:
                filtro_origen = st.radio("Origen", ["Todos", "Acciones", "ETFs", "Otros", "Fondos", "Criptos"], index=0, horizontal=True)
            with col_refresh:
                st.caption("")
                if st.button("🔄 Refrescar datos", key="btn_refresh_mov", help="Recarga movimientos desde la base de datos (útil tras actualizar nombres con scripts externos)"):
                    load_data.clear()
                    load_data_fondos.clear()
                    if hasattr(load_data_criptos, "clear"):
                        load_data_criptos.clear()
                    st.rerun()
            if filtro_origen == "Acciones":
                pt = mov["positionType"].astype(str).str.strip().str.lower() if "positionType" in mov.columns else pd.Series(["stock"] * len(mov))
                mov = mov[(mov["origen"] == "Acciones") & (pt == "stock")].copy()
            elif filtro_origen == "ETFs":
                pt = mov["positionType"].astype(str).str.strip().str.lower() if "positionType" in mov.columns else pd.Series([""] * len(mov))
                mov = mov[(mov["origen"] == "Acciones") & (pt == "etf")].copy()
            elif filtro_origen == "Otros":
                pt = mov["positionType"].astype(str).str.strip().str.lower() if "positionType" in mov.columns else pd.Series([""] * len(mov))
                mov = mov[(mov["origen"] == "Acciones") & (pt == "warrant")].copy()
            elif filtro_origen == "Fondos":
                mov = mov[mov["origen"] == "Fondos"].copy()
            elif filtro_origen == "Criptos":
                mov = mov[mov["origen"] == "Criptos"].copy()
            # Filtros por tipo, posición, cantidad y total
            with st.expander("Filtros", expanded=False):
                tipos_unicos = sorted(mov["type"].dropna().astype(str).str.strip().unique().tolist()) if "type" in mov.columns else []
                posiciones_unicas = sorted(mov["name"].dropna().astype(str).str.strip().unique().tolist()) if "name" in mov.columns else []
                sel_tipos = st.multiselect("Tipo de operación", options=tipos_unicos, default=tipos_unicos, key="filtro_tipo_mov")
                sel_posiciones = st.multiselect("Posición", options=posiciones_unicas, default=posiciones_unicas, key="filtro_pos_mov")
                c_cant, c_tot = st.columns(2)
                with c_cant:
                    min_cant = st.number_input("Cantidad mín.", value=None, placeholder="Sin mínimo", key="filtro_min_cant")
                    max_cant = st.number_input("Cantidad máx.", value=None, placeholder="Sin máximo", key="filtro_max_cant")
                with c_tot:
                    min_tot = st.number_input("Total (€) mín.", value=None, placeholder="Sin mínimo", key="filtro_min_tot")
                    max_tot = st.number_input("Total (€) máx.", value=None, placeholder="Sin máximo", key="filtro_max_tot")

            if sel_tipos and "type" in mov.columns:
                mov = mov[mov["type"].astype(str).str.strip().isin(sel_tipos)].copy()
            if sel_posiciones and "name" in mov.columns:
                mov = mov[mov["name"].astype(str).str.strip().isin(sel_posiciones)].copy()
            if min_cant is not None:
                mov = mov[mov["positionNumber"].fillna(0) >= min_cant].copy()
            if max_cant is not None:
                mov = mov[mov["positionNumber"].fillna(0) <= max_cant].copy()
            col_tot_eur = "totalWithComissionBaseCurrency" if "totalWithComissionBaseCurrency" in mov.columns else "totalBaseCurrency"
            if col_tot_eur not in mov.columns:
                col_tot_eur = "total"
            if min_tot is not None and col_tot_eur in mov.columns:
                mov = mov[mov[col_tot_eur].fillna(0) >= min_tot].copy()
            if max_tot is not None and col_tot_eur in mov.columns:
                mov = mov[mov[col_tot_eur].fillna(0) <= max_tot].copy()

            # Primera columna: tipo de operación con símbolo (y luego color)
            def tipo_simbolo(t):
                if pd.isna(t):
                    return "•"
                t = str(t).strip().lower()
                if t in ("buy", "switchbuy", "deposit", "bonus", "stakereward"):
                    return "▲ Compra"
                if t in ("sell", "switch", "withdrawal", "commission"):
                    return "▼ Venta"
                if t == "split":
                    return "⇄ Split"
                if t == "brokertransfer":
                    return "⇄ Traspaso"
                if t == "dividend":
                    return "💰 Div"
                return f"• {t}"

            mov["Tipo"] = mov["type"].apply(tipo_simbolo) if "type" in mov.columns else "•"

            col_map = {
                "datetime_full": "Fecha",
                "origen": "Origen",
                "broker": "Cuenta",
                "name": "Posición",
                "positionNumber": "Cantidad",
                "price": "Precio",
                "total": "Total",
                "comission": "Comisión",
                "taxes": "Impuestos",
                "exchangeRate": "Tipo de Cambio",
                "totalBaseCurrency": "Total (€)",
                "totalWithComissionBaseCurrency": "Total + com. + imp. (€)",
                "destinationRetentionBaseCurrency": "Retención en dest. realizada (€)",
            }
            if "comission" in mov.columns and "comissionCurrency" in mov.columns:
                mov["Comisión (€)"] = mov.apply(
                    lambda r: r["comission"] if str(r.get("comissionCurrency", "")).upper() == "EUR" else pd.NA,
                    axis=1,
                )
            else:
                mov["Comisión (€)"] = pd.NA
            display_mov = mov.rename(columns={k: v for k, v in col_map.items() if k in mov.columns})
            if "Tipo" not in display_mov.columns:
                display_mov["Tipo"] = mov["Tipo"]

            cols_final = [
                "Tipo",
                "Fecha",
                "Origen",
                "Cuenta",
                "Posición",
                "Cantidad",
                "Precio",
                "Total",
                "Comisión",
                "Comisión (€)",
                "Impuestos",
                "Tipo de Cambio",
                "Total (€)",
                "Total + com. + imp. (€)",
                "Retención en dest. realizada (€)",
            ]
            cols_presentes = [c for c in cols_final if c in display_mov.columns]

            def color_tipo(val):
                if pd.isna(val):
                    return ""
                v = str(val)
                if "Compra" in v or "▲" in v:
                    return "color: #1f77b4; font-weight: 500;"  # azul
                if "Venta" in v or "▼" in v:
                    return "color: #d62728; font-weight: 500;"  # rojo
                if "Split" in v or "⇄" in v:
                    return "color: #2ca02c; font-weight: 500;"  # verde para split
                if "Traspaso" in v:
                    return "color: #9467bd; font-weight: 500;"  # morado
                return ""

            habilitar_edicion = st.checkbox("Habilitar edición de datos", key="habilitar_edicion_mov")
            puede_editar = habilitar_edicion and not mov.empty and "_rowid_" in mov.columns

            if habilitar_edicion and not puede_editar and not mov.empty:
                st.warning("No se puede habilitar la edición. Prueba a pulsar «Refrescar datos» para recargar desde la base de datos.")

            if puede_editar:
                db_cols = MOVIMIENTOS_CRIPTOS_COLUMNS if filtro_origen == "Criptos" else MOVIMIENTOS_COLUMNS
                edit_cols = ["_rowid_"] + [c for c in db_cols if c in mov.columns]
                if filtro_origen == "Todos" and "origen" in mov.columns:
                    edit_cols = ["_rowid_", "origen"] + [c for c in db_cols if c in mov.columns]
                edit_df = mov[edit_cols].copy()
                editor_key = f"editor_mov_{filtro_origen}_{st.session_state.get('editor_mov_ver', 0)}"
                st.session_state[f"mov_original_{filtro_origen}"] = edit_df.copy()
                disabled_cols = ["_rowid_"]
                if filtro_origen == "Todos" and "origen" in edit_df.columns:
                    disabled_cols.append("origen")
                edited = st.data_editor(
                    edit_df,
                    num_rows="fixed",
                    use_container_width=True,
                    key=editor_key,
                    disabled=disabled_cols,
                )
                try:
                    orig = st.session_state[f"mov_original_{filtro_origen}"]
                    has_changes = not edited.astype(str).fillna("").equals(orig.astype(str).fillna(""))
                except Exception:
                    has_changes = False
                if has_changes:
                    def _tabla_por_origen(orig: str) -> str:
                        o = str(orig or "").strip()
                        if o == "Fondos":
                            return "movimientos_fondos"
                        if o == "Criptos":
                            return "movimientos_criptos"
                        return "movimientos"
                    tabla_por_filtro = {"Acciones": "movimientos", "ETFs": "movimientos", "Otros": "movimientos", "Fondos": "movimientos_fondos", "Criptos": "movimientos_criptos"}
                    if st.button("Guardar cambios", type="primary", key="btn_guardar_mov_edit"):
                        try:
                            with _get_db() as conn:
                                update_cols = [c for c in db_cols if c in edited.columns]
                                for col in ["total", "totalBaseCurrency", "totalWithComission", "totalWithComissionBaseCurrency"]:
                                    if col not in update_cols and col in db_cols:
                                        update_cols = update_cols + [col]
                                for _, row in edited.iterrows():
                                    row_id = int(row["_rowid_"])
                                    row_dict = {c: row.get(c, "") for c in update_cols}
                                    if filtro_origen == "Todos":
                                        tabla_nombre = _tabla_por_origen(row.get("origen", ""))
                                    else:
                                        tabla_nombre = tabla_por_filtro[filtro_origen]
                                    qty = float(_to_float(row_dict.get("positionNumber"), 0.0))
                                    price = float(_to_float(row_dict.get("price"), 0.0))
                                    comm = float(_to_float(row_dict.get("comission"), 0.0))
                                    tax = float(_to_float(row_dict.get("taxes"), 0.0))
                                    fx = float(_to_float(row_dict.get("exchangeRate"), 1.0))
                                    pos_ccy = str(row_dict.get("positionCurrency", "") or "EUR").strip()
                                    comm_ccy = str(row_dict.get("comissionCurrency", "") or "").strip()
                                    tax_ccy = str(row_dict.get("taxesCurrency", "") or "").strip()
                                    row_tipo = str(row.get("type", row_dict.get("type", "")) or "").strip().lower()
                                    recalc = _recalc_totals(qty, price, comm, tax, fx, pos_ccy, comm_ccy, tax_ccy, tipo=row_tipo)
                                    row_dict["total"] = recalc["total"]
                                    row_dict["totalBaseCurrency"] = recalc["totalBaseCurrency"]
                                    row_dict["totalWithComission"] = recalc["totalWithComission"]
                                    row_dict["totalWithComissionBaseCurrency"] = recalc["totalWithComissionBaseCurrency"]
                                    vals = [_row_to_db_val(row_dict.get(c, "")) for c in update_cols]
                                    sets = ", ".join(f'"{c}" = ?' for c in update_cols)
                                    conn.execute(f"UPDATE {tabla_nombre} SET {sets} WHERE rowid = ?", vals + [row_id])
                                conn.commit()
                            del st.session_state[f"mov_original_{filtro_origen}"]
                            st.session_state["editor_mov_ver"] = st.session_state.get("editor_mov_ver", 0) + 1
                            load_data.clear()
                            load_data_fondos.clear()
                            if hasattr(load_data_criptos, "clear"):
                                load_data_criptos.clear()
                            st.success("Cambios guardados correctamente.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error al guardar: {e}")
            else:
                st.dataframe(
                    display_mov[cols_presentes].style.applymap(
                        color_tipo, subset=["Tipo"]
                    ),
                    use_container_width=True,
                )

            # Eliminar operaciones: solo movimientos de Acciones (se actualiza solo tabla movimientos)
            st.subheader("Eliminar operaciones")
            if not mov.empty:
                def _etiqueta_fila(i: int) -> str:
                    r = mov.iloc[i]
                    fecha = str(r.get("date", "")) + " " + str(r.get("time", ""))[:8]
                    nombre = str(r.get("name", r.get("ticker", "")))
                    tipo = str(r.get("Tipo", r.get("type", "")))
                    qty = r.get("positionNumber", "")
                    orig = str(r.get("origen", ""))
                    return f"{fecha} | {nombre} | {tipo} | {qty} [{orig}]"

                opciones = list(range(len(mov)))
                eliminar = st.multiselect(
                    "Selecciona las operaciones a eliminar (solo se pueden eliminar movimientos de Acciones)",
                    opciones,
                    format_func=_etiqueta_fila,
                    key="eliminar_operaciones",
                )
                if eliminar and st.button("Eliminar seleccionadas", type="primary", key="btn_eliminar"):
                    # Las posiciones en eliminar son índices en la tabla mostrada (mov filtrado)
                    acc_indices_to_drop = []
                    for pos in eliminar:
                        if pos < 0 or pos >= len(mov):
                            continue
                        row = mov.iloc[pos]
                        if str(row.get("origen", "")) != "Acciones":
                            continue
                        idx = row.get("_acc_idx")
                        if pd.notna(idx) and idx is not None:
                            acc_indices_to_drop.append(int(idx))
                    if not acc_indices_to_drop:
                        st.warning("Ninguna de las filas seleccionadas es de Acciones. Solo se pueden eliminar movimientos de acciones/ETFs.")
                    else:
                        # Quitar solo esas filas del dataset completo de acciones (df), no del filtrado
                        df_sin = df.drop(index=acc_indices_to_drop)
                        cols_csv = [c for c in MOVIMIENTOS_COLUMNS if c in df_sin.columns]
                        write_full_db(df_sin[cols_csv])
                        load_data.clear()
                        st.rerun()

        with tab_div:
            st.subheader("Dividendos")
            st.caption("Solo puedes registrar dividendos de posiciones que ya tengas en la cartera (elige de la lista).")
            div_catalog = get_ticker_catalog(df)
            if div_catalog.empty:
                st.info("No hay posiciones en la cartera. Añade primero operaciones de compra para poder registrar dividendos.")
            else:
                with st.expander("➕ Nuevo dividendo", expanded=False):
                    st.markdown("**Dividendo**")
                    ticker_options_div = ["—— Elige posición ——"]
                    for _, r in div_catalog.iterrows():
                        ticker_options_div.append(f"{r['ticker']} | {r['name']} ({r['positionCurrency']})")
                    sel_pos_div = st.selectbox("Posición", ticker_options_div, key="sel_pos_dividendo", placeholder="Selecciona una posición")
                    div_ticker = div_yahoo = div_nombre = div_ccy = div_type = div_country = div_exchange = ""
                    if sel_pos_div and sel_pos_div != "—— Elige posición ——":
                        idx_div = ticker_options_div.index(sel_pos_div) - 1
                        if idx_div >= 0 and idx_div < len(div_catalog):
                            r = div_catalog.iloc[idx_div]
                            div_ccy = str(r["positionCurrency"]) if pd.notna(r["positionCurrency"]) else "EUR"
                            div_ticker = str(r["ticker"]) if pd.notna(r["ticker"]) else ""
                            div_yahoo = str(r["ticker_Yahoo"]) if pd.notna(r["ticker_Yahoo"]) else div_ticker
                            div_nombre = str(r["name"]) if pd.notna(r["name"]) else div_ticker
                            div_type = str(r.get("positionType", "stock")).strip().lower() if pd.notna(r.get("positionType")) else "stock"
                            div_country = str(r["positionCountry"]) if pd.notna(r["positionCountry"]) else ""
                            div_exchange = str(r["positionExchange"]) if pd.notna(r["positionExchange"]) else ""

                    div_broker = st.selectbox("Cuenta", options=brokers_list if brokers_list else ["——"], key="div_broker", placeholder="Selecciona una cuenta bróker o wallet") if brokers_list else st.text_input("Cuenta", key="div_broker", placeholder="Nombre del bróker")

                    col_fecha, col_hora, col_tit = st.columns(3)
                    with col_fecha:
                        div_date = st.date_input("Fecha de pago", key="div_date")
                    with col_hora:
                        div_time = st.time_input("Hora", value=dt_time(22, 0), key="div_time", step=60, help="Hora de cobro del dividendo (selección por minutos)")
                    with col_tit:
                        div_position_number = st.text_input("Número de títulos", placeholder="0", key="div_position_number")

                    _div_ccy = (div_ccy or "EUR").strip().upper()
                    _div_currencies = ["EUR", "USD", "GBP", "CAD", "DKK", "HKD", "JPY", "CHF"]
                    _div_cur_idx = _div_currencies.index(_div_ccy) if _div_ccy in _div_currencies else 0
                    div_currency = st.selectbox("Divisa", _div_currencies, index=_div_cur_idx, key="div_currency")
                    div_ccy_sel = (div_currency or "EUR").strip()

                    use_total_bruto = st.toggle("Introducir TOTAL bruto en lugar de cantidad por título", value=False, key="div_use_total")
                    col_cant, col_com = st.columns(2)
                    with col_cant:
                        if use_total_bruto:
                            _total_str = st.text_input(f"Total bruto ({div_ccy_sel})", placeholder="0,00", key="div_total_bruto")
                            div_total_val = _to_float(_total_str, 0.0)
                            div_quantity_val = (div_total_val / _to_float(div_position_number, 1.0)) if _to_float(div_position_number, 0.0) else 0.0
                        else:
                            _cant_str = st.text_input(f"Cantidad bruta por título ({div_ccy_sel})", placeholder="0,00", key="div_cantidad_titulo")
                            div_quantity_val = _to_float(_cant_str, 0.0)
                            div_total_val = div_quantity_val * _to_float(div_position_number, 0.0)
                    with col_com:
                        _com_label = f"Comisión ({div_ccy_sel})" if div_ccy_sel != "EUR" else "Comisión (EUR)"
                        _com_str = st.text_input(_com_label, value="0", placeholder="0", key="div_comision")
                        div_comision_ccy = _to_float(_com_str, 0.0)

                    div_exchange_rate = 1.0
                    if div_ccy_sel != "EUR":
                        mod_fx_div = st.toggle("Modificar tipo de cambio", value=False, key="div_mod_fx", help="Desactivado: se usa el tipo de cambio de cierre del día. Activado: puedes indicar o buscar el tipo de cambio.")
                        if not mod_fx_div:
                            div_exchange_rate = get_fx_rate_for_date(div_ccy_sel, div_date)
                            if math.isnan(div_exchange_rate) or div_exchange_rate <= 0:
                                div_exchange_rate = 1.0
                            st.caption(f"Tipo de cambio de cierre del día: **{div_exchange_rate:.4f}** {div_ccy_sel}/EUR. Activa el switch para indicar otro valor.")
                        else:
                            if "div_fx_pending" in st.session_state:
                                st.session_state["div_fx"] = str(st.session_state["div_fx_pending"]).replace(".", ",")
                                del st.session_state["div_fx_pending"]
                            col_fx, col_btn1, col_btn2 = st.columns([2, 1, 1])
                            with col_fx:
                                _fx_str = st.text_input(
                                    f"Tipo de cambio ({div_ccy_sel}/EUR)",
                                    placeholder="0,92",
                                    help=f"Ej: 0,92 significa 1 {div_ccy_sel} = 0,92 EUR",
                                    key="div_fx",
                                )
                                div_exchange_rate = _to_float(_fx_str, 1.0) if (_fx_str or "").strip() else 1.0
                            with col_btn1:
                                st.caption("")
                                if st.button("Cierre del día", key="btn_div_fx_cierre", help="Obtener tipo de cambio de cierre para la fecha de pago (Yahoo Finance)"):
                                    rate = get_fx_rate_for_date(div_ccy_sel, div_date)
                                    if not math.isnan(rate) and rate > 0:
                                        st.session_state["div_fx_pending"] = rate
                                        st.rerun()
                                    else:
                                        st.warning("No se pudo obtener el tipo de cambio. Introduce el valor a mano.")
                            with col_btn2:
                                st.caption("")
                                if st.button("A la hora de pago", key="btn_div_fx_intra", help="Obtener tipo de cambio aproximado en el momento del cobro (Yahoo Finance)"):
                                    _hora_txt = div_time.strftime("%H:%M:%S") if hasattr(div_time, "strftime") else (str(div_time).strip() if div_time else "22:00:00")
                                    dt_txt = f"{div_date.strftime('%Y-%m-%d')} {_hora_txt}"
                                    rate = get_fx_rate_at_datetime(div_ccy_sel, dt_txt)
                                    if not math.isnan(rate) and rate > 0:
                                        st.session_state["div_fx_pending"] = rate
                                        st.rerun()
                                    else:
                                        st.warning("No se pudo obtener el tipo de cambio intradía. Prueba con cierre del día o introduce el valor a mano.")
                    div_comision_eur = div_comision_ccy if div_ccy_sel == "EUR" else div_comision_ccy * div_exchange_rate
                    div_total_base_val = div_total_val if div_ccy_sel == "EUR" else div_total_val * div_exchange_rate

                    st.markdown("**Retenciones**")
                    mod_ret_origen = st.toggle("Modificar la retención en origen", value=False, key="div_mod_ret_origen")
                    div_origin_ret_ccy = 0.0
                    div_origin_ret_eur = 0.0
                    if mod_ret_origen:
                        _ret_origen_label = f"Retención en origen ({div_ccy_sel})" if div_ccy_sel != "EUR" else "Retención en origen (EUR)"
                        div_origin_ret_ccy = _to_float(st.text_input(_ret_origen_label, placeholder="0,00", key="div_origin_ret_eur"), 0.0)
                        div_origin_ret_eur = div_origin_ret_ccy if div_ccy_sel == "EUR" else div_origin_ret_ccy * div_exchange_rate
                    mod_ret_destino = st.toggle("Modificar la retención en destino", value=False, key="div_mod_ret_destino", help="Si la cuenta tiene 'Retiene en destino' activado (ficha de cuenta), suele ser Sí.")
                    div_dest_ret_eur = 0.0
                    if mod_ret_destino:
                        div_dest_ret_eur = _to_float(st.text_input("Retención en destino (EUR)", placeholder="0,00", key="div_dest_ret_eur"), 0.0)
                    mod_pct_recup = st.toggle("Modificar % recuperable por doble imposición (15% por defecto)", value=False, key="div_mod_pct_recup")
                    div_pct_recup = 15
                    if mod_pct_recup:
                        div_pct_recup = _to_float(st.text_input("Porcentaje (%)", value="15", placeholder="15", key="div_pct_recup"), 15.0)

                    div_description = st.text_area("Notas (opcional)", key="div_description", placeholder="", height=80)

                    st.markdown("**Previsualizar totales**")
                    total_bruto_eur = div_total_base_val
                    ret_origen_eur = div_origin_ret_eur
                    ret_dest_eur = div_dest_ret_eur
                    total_neto_eur = total_bruto_eur - ret_origen_eur - ret_dest_eur - div_comision_eur
                    _dest_ret_ccy = div_dest_ret_eur / div_exchange_rate if div_exchange_rate and div_ccy_sel != "EUR" else div_dest_ret_eur
                    total_neto_ccy = div_total_val - div_origin_ret_ccy - _dest_ret_ccy
                    def _fmt_prev(val_eur, val_ccy=None):
                        eur_str = f"{val_eur:,.2f}".replace(",", " ").replace(".", ",") + " €" if val_eur else "–"
                        if div_ccy_sel == "EUR" or val_ccy is None:
                            return eur_str
                        ccy_str = f"{val_ccy:,.2f}".replace(",", " ").replace(".", ",") + " " + div_ccy_sel if val_ccy is not None else "–"
                        eur_str_full = f"{val_eur:,.2f}".replace(",", " ").replace(".", ",") + " €" if val_eur is not None else "–"
                        return f"{ccy_str} ({eur_str_full})"
                    prev1, prev2, prev3, prev4 = st.columns(4)
                    with prev1:
                        st.metric("Total bruto (€)", _fmt_prev(total_bruto_eur, div_total_val if div_ccy_sel != "EUR" else None))
                    with prev2:
                        st.metric("Retención en origen (€)", _fmt_prev(ret_origen_eur, div_origin_ret_ccy if div_ccy_sel != "EUR" else None))
                    with prev3:
                        st.metric("Retención en dest. realizada (€)", f"{ret_dest_eur:,.2f}".replace(",", " ").replace(".", ",") + " €" if ret_dest_eur else "–")
                    with prev4:
                        st.metric("Total neto cobrado (€)", _fmt_prev(total_neto_eur, total_neto_ccy if div_ccy_sel != "EUR" else None))

                    col_btn1, col_btn2, _ = st.columns([1, 1, 2])
                    with col_btn1:
                        cancelar = st.button("CANCELAR", key="div_cancelar")
                    with col_btn2:
                        guardar = st.button("GUARDAR DIVIDENDO", type="primary", key="guardar_dividendo")
                    if cancelar:
                        st.rerun()
                    if guardar:
                        if not sel_pos_div or sel_pos_div == "—— Elige posición ——":
                            st.error("Elige una posición de la lista.")
                        else:
                            if hasattr(div_time, "strftime"):
                                time_str = div_time.strftime("%H:%M:%S")
                            else:
                                _t = str(div_time or "22:00").strip()
                                if ":" in _t:
                                    parts = _t.split(":")
                                    _t = f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:00" if len(parts) >= 2 else _t
                                time_str = _t if _t else "22:00:00"
                            date_str = div_date.strftime("%Y-%m-%d") if hasattr(div_date, "strftime") else str(div_date)
                            neto_base = total_bruto_eur - div_origin_ret_eur
                            origin_ret_ccy = div_origin_ret_ccy
                            dest_ret_ccy = div_dest_ret_eur / div_exchange_rate if div_exchange_rate and div_ccy_sel != "EUR" else div_dest_ret_eur
                            total_neto_ccy = div_total_val - origin_ret_ccy - dest_ret_ccy
                            row_div = {
                                "type": "stockDividend",
                                "date": date_str,
                                "time": time_str,
                                "ticker": div_ticker,
                                "ticker_Yahoo": div_yahoo,
                                "nombre": div_nombre,
                                "positionType": div_type,
                                "positionCountry": div_country or "",
                                "positionCurrency": div_ccy or "EUR",
                                "positionExchange": div_exchange or "",
                                "broker": div_broker or "",
                                "positionNumber": _to_float(div_position_number, 0.0),
                                "currency": div_currency or div_ccy or "EUR",
                                "quantity": div_quantity_val,
                                "quantityCurrency": div_currency or div_ccy or "EUR",
                                "comission": div_comision_eur,
                                "comissionCurrency": "EUR",
                                "exchangeRate": div_exchange_rate,
                                "comissionBaseCurrency": div_comision_eur,
                                "autoFx": "No",
                                "total": div_total_val,
                                "totalBaseCurrency": total_bruto_eur,
                                "originRetention": origin_ret_ccy,
                                "neto": div_total_val - origin_ret_ccy,
                                "netoBaseCurrency": neto_base,
                                "destinationRetentionBaseCurrency": ret_dest_eur,
                                "totalNeto": total_neto_ccy,
                                "totalNetoBaseCurrency": total_neto_eur,
                                "retentionReturned": origin_ret_ccy * (div_pct_recup / 100.0),
                                "retentionReturnedBaseCurrency": div_origin_ret_eur * (div_pct_recup / 100.0),
                                "unrealizedDestinationRetentionBaseCurrency": "",
                                "netoWithReturnBaseCurrency": "",
                                "originRetentionLossBaseCurrency": "",
                                "description": (div_description or "").strip(),
                            }
                            try:
                                append_dividendo(row_div)
                                st.success("Dividendo guardado.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error al guardar: {e}")

                div_df = load_dividendos()
                if div_df.empty:
                    st.info("No hay dividendos registrados.")
                else:
                    st.subheader("Listado de dividendos")
                    _meses = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
                    def _to_float_div(x, default=0.0):
                        if x is None or (isinstance(x, float) and pd.isna(x)) or str(x).strip() == "":
                            return default
                        try:
                            return float(str(x).replace(",", ".").strip())
                        except (ValueError, TypeError):
                            return default
                    show_div = pd.DataFrame()
                    show_div["Fecha"] = div_df["date"].astype(str).apply(
                        lambda s: (lambda d: f"{d.day:02d} {_meses[d.month - 1]} {d.year}" if pd.notna(d) else s)(pd.to_datetime(s.split("T")[0] if "T" in s else s, errors="coerce"))
                    )
                    show_div["Cuenta"] = div_df["broker"].astype(str)
                    show_div["Posición"] = div_df["ticker"].astype(str)
                    show_div["Títulos / Particip."] = div_df["positionNumber"].astype(str).str.replace(".", ",", regex=False)
                    div_ccy = div_df.get("currency", div_df.get("positionCurrency", "EUR")).astype(str)
                    show_div["Cantidad por título/particip."] = [
                        _fmt_div_currency(_to_float_div(div_df["quantity"].iloc[i]), div_ccy.iloc[i] if i < len(div_ccy) else "EUR")
                        for i in range(len(div_df))
                    ]
                    show_div["Total bruto"] = [
                        _fmt_div_currency(_to_float_div(div_df["total"].iloc[i]), div_ccy.iloc[i] if i < len(div_ccy) else "EUR")
                        for i in range(len(div_df))
                    ]
                    show_div["Tipo de cambio"] = div_df["exchangeRate"].astype(str).str.replace(".", ",", regex=False)
                    show_div["Total bruto (€)"] = [_fmt_div_currency(_to_float_div(div_df["totalBaseCurrency"].iloc[i]), "EUR") for i in range(len(div_df))]
                    show_div["Retención en origen"] = [
                        _fmt_div_currency(_to_float_div(div_df["originRetention"].iloc[i]), div_ccy.iloc[i] if i < len(div_ccy) else "EUR")
                        for i in range(len(div_df))
                    ]
                    show_div["Total bruto después de origen (€)"] = [_fmt_div_currency(_to_float_div(div_df["netoBaseCurrency"].iloc[i]), "EUR") for i in range(len(div_df))]
                    show_div["Retención en dest. realizada"] = [_fmt_div_currency(_to_float_div(div_df["destinationRetentionBaseCurrency"].iloc[i]), "EUR") for i in range(len(div_df))]
                    show_div["Comisión"] = div_df.get("comission", pd.Series([""] * len(div_df))).astype(str).str.replace(".", ",", regex=False)
                    show_div["Comisión (€)"] = [_fmt_div_currency(_to_float_div(div_df["comissionBaseCurrency"].iloc[i]) if "comissionBaseCurrency" in div_df.columns else 0, "EUR") for i in range(len(div_df))]
                    show_div["Total neto cobrado (€)"] = [_fmt_div_currency(_to_float_div(div_df["totalNetoBaseCurrency"].iloc[i]), "EUR") for i in range(len(div_df))]
                    show_div["Impuesto satisf. en el extranjero"] = [
                        _fmt_div_currency(_to_float_div(div_df["retentionReturned"].iloc[i]), div_ccy.iloc[i] if i < len(div_ccy) else "EUR")
                        for i in range(len(div_df))
                    ]
                    show_div["Impuesto satisf. en el extranjero (€)"] = [_fmt_div_currency(_to_float_div(div_df["retentionReturnedBaseCurrency"].iloc[i]), "EUR") for i in range(len(div_df))]
                    show_div["Retención a realizar o devolver (€)"] = [_fmt_div_currency(_to_float_div(div_df["unrealizedDestinationRetentionBaseCurrency"].iloc[i]) if "unrealizedDestinationRetentionBaseCurrency" in div_df.columns else 0, "EUR") for i in range(len(div_df))]
                    show_div["Total neto con devolución (€)"] = [_fmt_div_currency(_to_float_div(div_df["netoWithReturnBaseCurrency"].iloc[i]) if "netoWithReturnBaseCurrency" in div_df.columns else 0, "EUR") for i in range(len(div_df))]
                    show_div["Retención no recuperada (€)"] = [_fmt_div_currency(_to_float_div(div_df["originRetentionLossBaseCurrency"].iloc[i]) if "originRetentionLossBaseCurrency" in div_df.columns else 0, "EUR") for i in range(len(div_df))]
                    show_div["AutoFx"] = div_df.get("autoFx", pd.Series(["No"] * len(div_df))).astype(str)

                    habilitar_edicion_div = st.checkbox("Habilitar edición de dividendos", key="habilitar_edicion_div")
                    puede_editar_div = habilitar_edicion_div and "_rowid_" in div_df.columns

                    if puede_editar_div:
                        edit_cols_div = ["_rowid_"] + [c for c in DIVIDENDOS_COLUMNS if c in div_df.columns]
                        edit_df_div = div_df[edit_cols_div].copy()
                        st.session_state["div_original"] = edit_df_div.copy()
                        edited_div = st.data_editor(
                            edit_df_div,
                            num_rows="fixed",
                            use_container_width=True,
                            key="editor_dividendos",
                            disabled=["_rowid_"],
                        )
                        try:
                            orig_div = st.session_state["div_original"]
                            has_changes_div = not edited_div.astype(str).fillna("").equals(orig_div.astype(str).fillna(""))
                        except Exception:
                            has_changes_div = False
                        if has_changes_div and st.button("Guardar cambios en dividendos", type="primary", key="btn_guardar_div_edit"):
                            try:
                                for _, row in edited_div.iterrows():
                                    row_id = int(row["_rowid_"])
                                    row_dict = {c: row.get(c, "") for c in DIVIDENDOS_COLUMNS if c in edited_div.columns}
                                    update_dividendo(row_id, row_dict)
                                del st.session_state["div_original"]
                                st.success("Cambios guardados correctamente.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error al guardar: {e}")
                    else:
                        st.dataframe(show_div, use_container_width=True)

                    st.subheader("Eliminar dividendos")
                    def _etiqueta_div(i: int) -> str:
                        r = div_df.iloc[i]
                        fecha = str(r.get("date", "")) + " " + str(r.get("time", ""))[:8]
                        ticker = str(r.get("ticker", r.get("ticker_Yahoo", "")))
                        broker = str(r.get("broker", ""))
                        total = r.get("totalBaseCurrency", r.get("total", ""))
                        return f"{fecha} | {ticker} | {broker} | {total} €"

                    opciones_div = list(range(len(div_df)))
                    eliminar_div = st.multiselect(
                        "Selecciona los dividendos a eliminar",
                        opciones_div,
                        format_func=_etiqueta_div,
                        key="eliminar_dividendos",
                    )
                    if eliminar_div and st.button("Eliminar dividendos seleccionados", type="primary", key="btn_eliminar_div"):
                        rowids_to_del = [int(div_df.iloc[pos]["_rowid_"]) for pos in eliminar_div if 0 <= pos < len(div_df)]
                        n = delete_dividendos_by_rowids(rowids_to_del)
                        st.success(f"Eliminados {n} dividendo(s).")
                        st.rerun()
        return

    if vista == "Fiscalidad":
        st.header("Fiscalidad")

        def _to_float_div(x, default=0.0):
            if x is None or (isinstance(x, float) and pd.isna(x)) or str(x).strip() == "":
                return default
            try:
                return float(str(x).replace(",", ".").strip())
            except (ValueError, TypeError):
                return default

        # --- Selector de ejercicio ---
        anio_actual = datetime.now().year
        ejercicios = [anio_actual, anio_actual - 1]
        ejercicio = st.radio("Ejercicio", ejercicios, format_func=lambda x: f"Ejercicio {x}", horizontal=True, key="fisc_ejercicio")

        # --- Cargar datos FIFO ---
        df_fondos_fisc = load_data_fondos()
        df_crip_fisc = load_data_criptos()

        lots_df, sales_df = compute_fifo_all(df)
        lots_fondos, sales_fondos = compute_fifo_fondos(df_fondos_fisc)
        lots_crip, sales_crip = compute_fifo_criptos(df_crip_fisc)

        if not sales_crip.empty and "Retención dest. (€)" not in sales_crip.columns:
            sales_crip["Retención dest. (€)"] = 0.0

        if not lots_fondos.empty:
            lots_df = pd.concat([lots_df, lots_fondos], ignore_index=True) if not lots_df.empty else lots_fondos
        if not sales_fondos.empty:
            sales_df = pd.concat([sales_df, sales_fondos], ignore_index=True) if not sales_df.empty else sales_fondos
        if not lots_crip.empty:
            lots_df = pd.concat([lots_df, lots_crip], ignore_index=True) if not lots_df.empty else lots_crip
        if not sales_crip.empty:
            sales_df = pd.concat([sales_df, sales_crip], ignore_index=True) if not sales_df.empty else sales_crip

        if not sales_df.empty and "Retención dest. (€)" not in sales_df.columns:
            sales_df["Retención dest. (€)"] = 0.0

        # Filtrar ventas por ejercicio
        sales_ejercicio = sales_df.copy()
        if not sales_ejercicio.empty and "Fecha venta" in sales_ejercicio.columns:
            fechas = pd.to_datetime(sales_ejercicio["Fecha venta"], errors="coerce")
            sales_ejercicio = sales_ejercicio[fechas.dt.year == ejercicio].copy()

        # --- Resumen fiscal (tarjetas) ---
        g_p_brutas = sales_ejercicio["Plusvalía / Minusvalía (€)"].sum() if not sales_ejercicio.empty and "Plusvalía / Minusvalía (€)" in sales_ejercicio.columns else 0.0

        div_df = load_dividendos()
        div_ejercicio = pd.DataFrame()
        if not div_df.empty:
            div_df = div_df.copy()
            div_df["year"] = pd.to_datetime(div_df["date"], errors="coerce").dt.year
            div_ejercicio = div_df[div_df["year"] == ejercicio].copy()

        total_dividendos_bruto = div_ejercicio["totalBaseCurrency"].apply(lambda x: _to_float_div(x, 0.0)).sum() if not div_ejercicio.empty else 0.0
        retencion_origen_div = 0.0
        retencion_destino_div = 0.0
        if not div_ejercicio.empty:
            for _, r in div_ejercicio.iterrows():
                total_b = _to_float_div(r.get("totalBaseCurrency"), 0.0)
                neto_b = _to_float_div(r.get("netoBaseCurrency"), 0.0)
                retencion_origen_div += total_b - neto_b
                retencion_destino_div += _to_float_div(r.get("destinationRetentionBaseCurrency"), 0.0)
        total_neto_dividendos = div_ejercicio["totalNetoBaseCurrency"].apply(lambda x: _to_float_div(x, 0.0)).sum() if not div_ejercicio.empty and "totalNetoBaseCurrency" in div_ejercicio.columns else (total_dividendos_bruto - retencion_origen_div - retencion_destino_div)

        # Comisiones: de movimientos y dividendos del ejercicio
        df_all = df.copy()
        if not df_fondos_fisc.empty:
            df_all = pd.concat([df_all, df_fondos_fisc], ignore_index=True)
        if not df_crip_fisc.empty:
            df_all = pd.concat([df_all, df_crip_fisc], ignore_index=True)
        df_all["year"] = pd.to_datetime(df_all["date"], errors="coerce").dt.year
        df_ejercicio = df_all[df_all["year"] == ejercicio] if "year" in df_all.columns else df_all
        comisiones_mov = df_ejercicio["comission"].apply(lambda x: _to_float_div(x, 0.0)).sum() if not df_ejercicio.empty and "comission" in df_ejercicio.columns else 0.0
        comisiones_div = div_ejercicio["comission"].apply(lambda x: _to_float_div(x, 0.0)).sum() if not div_ejercicio.empty and "comission" in div_ejercicio.columns else 0.0
        if not div_ejercicio.empty and "comissionBaseCurrency" in div_ejercicio.columns:
            comisiones_div = div_ejercicio["comissionBaseCurrency"].apply(lambda x: _to_float_div(x, 0.0)).sum()
        total_comisiones = comisiones_mov + comisiones_div

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Resumen fiscal")
            st.metric("G/P realizadas brutas (€)", fmt_eur(g_p_brutas))
            st.metric("Dividendos cobrados brutos (€)", fmt_eur(total_dividendos_bruto))
            st.metric("Total neto dividendos (€)", fmt_eur(total_neto_dividendos))
            st.metric("Comisiones (€)", fmt_eur(total_comisiones))
        with col2:
            st.subheader("Dividendos y cupones")
            st.caption("Total bruto (€)")
            st.write(fmt_eur(total_dividendos_bruto))
            st.caption("Retenciones origen (€)")
            st.write(fmt_eur(retencion_origen_div))
            st.caption("Retenciones destino (€)")
            st.write(fmt_eur(retencion_destino_div))
            st.caption("Total neto cobrado (€)")
            st.write(fmt_eur(total_neto_dividendos))

        # --- Tabs ---
        tab_resumen, tab_gp_tipo, tab_gp_activo, tab_div_pos = st.tabs([
            "G/P por tipo de posición",
            "G/P por activo",
            "Dividendos por posición",
            "Posiciones vivas / Ventas (detalle)",
        ])

        # Mapeo tipo activo → etiqueta
        def _tipo_label(t):
            t = str(t or "").strip().lower()
            if t == "stock": return "Valores"
            if t == "etf": return "ETFs"
            if t == "fund": return "Fondos"
            if t == "crypto": return "Cripto"
            if t == "warrant": return "Otros"
            return "Valores" if t else "Otros"

        with tab_resumen:
            st.subheader("G/P realizadas por tipo de posición")
            if sales_ejercicio.empty:
                st.info("No hay ventas en este ejercicio.")
            else:
                sales_tipo = sales_ejercicio.copy()
                sales_tipo["Tipo"] = sales_tipo["Tipo activo"].apply(_tipo_label)
                agg_dict = {
                    "total_recibido": ("Valor venta (€)", "sum"),
                    "coste_total": ("Valor compra histórico (€)", "sum"),
                    "ganancia_perdida": ("Plusvalía / Minusvalía (€)", "sum"),
                }
                if "Retención dest. (€)" in sales_tipo.columns:
                    agg_dict["retencion_dest"] = ("Retención dest. (€)", "sum")
                agg = sales_tipo.groupby("Tipo", as_index=False).agg(**agg_dict)
                agg = agg.rename(columns={
                    "total_recibido": "Total recibido con comisión (€)",
                    "coste_total": "Coste total con com. e imp. (€)",
                    "ganancia_perdida": "Ganancia/Pérd. con comisión (€)",
                    "retencion_dest": "Rtción. en dest. realz. (€)",
                })
                if "Rtción. en dest. realz. (€)" not in agg.columns:
                    agg["Rtción. en dest. realz. (€)"] = 0.0
                st.dataframe(
                    agg.style.format({
                        "Total recibido con comisión (€)": lambda x: fmt_eur(x),
                        "Coste total con com. e imp. (€)": lambda x: fmt_eur(x),
                        "Ganancia/Pérd. con comisión (€)": lambda x: fmt_eur(x),
                        "Rtción. en dest. realz. (€)": lambda x: fmt_eur(x),
                    }, na_rep="–"),
                    use_container_width=True,
                )

        with tab_gp_tipo:
            st.subheader("Ganancia/Pérdida realizada por activo")
            if sales_ejercicio.empty:
                st.info("No hay ventas en este ejercicio.")
            else:
                cols_activo = ["Ticker", "Tipo activo", "Valor compra histórico (€)", "Valor venta (€)", "Plusvalía / Minusvalía (€)"]
                if "Retención dest. (€)" in sales_ejercicio.columns:
                    cols_activo.append("Retención dest. (€)")
                sales_activo = sales_ejercicio.copy()
                sales_activo["Tipo"] = sales_activo["Tipo activo"].apply(_tipo_label)
                agg_act_dict = {
                    "valor_adq": ("Valor compra histórico (€)", "sum"),
                    "valor_trans": ("Valor venta (€)", "sum"),
                    "ganancia": ("Plusvalía / Minusvalía (€)", "sum"),
                }
                if "Retención dest. (€)" in sales_activo.columns:
                    agg_act_dict["ret_dest"] = ("Retención dest. (€)", "sum")
                agg_activo = sales_activo.groupby(["Ticker", "Tipo"], as_index=False).agg(**agg_act_dict).rename(columns={
                    "valor_adq": "Valor adquisición (€)",
                    "valor_trans": "Valor transmisión (€)",
                    "ganancia": "Ganancia/Pérdida (€)",
                    "ret_dest": "Rtción. en dest. realz. (€)",
                })
                if "Rtción. en dest. realz. (€)" not in agg_activo.columns:
                    agg_activo["Rtción. en dest. realz. (€)"] = 0.0
                agg_activo = agg_activo.sort_values("Ganancia/Pérdida (€)", ascending=False)
                st.dataframe(
                    agg_activo.style.format({c: lambda x: fmt_eur(x) for c in agg_activo.columns if "€" in str(c)}, na_rep="–"),
                    use_container_width=True,
                )

        with tab_gp_activo:
            st.subheader("Dividendos y cupones por posición")
            if div_ejercicio.empty:
                st.info("No hay dividendos en este ejercicio.")
            else:
                div_pos = div_ejercicio.copy()
                div_pos["ticker"] = div_pos["ticker"] if "ticker" in div_pos.columns else div_pos["ticker_Yahoo"]
                div_pos["total_bruto"] = div_pos["totalBaseCurrency"].apply(lambda x: _to_float_div(x, 0.0))
                div_pos["impuesto_ext"] = div_pos["totalBaseCurrency"].apply(lambda x: _to_float_div(x, 0.0)) - div_pos["netoBaseCurrency"].apply(lambda x: _to_float_div(x, 0.0))
                div_pos["total_despues_origen"] = div_pos["netoBaseCurrency"].apply(lambda x: _to_float_div(x, 0.0))
                div_pos["ret_dest"] = div_pos["destinationRetentionBaseCurrency"].apply(lambda x: _to_float_div(x, 0.0))
                div_pos["total_neto"] = div_pos["totalNetoBaseCurrency"].apply(lambda x: _to_float_div(x, 0.0)) if "totalNetoBaseCurrency" in div_pos.columns else (div_pos["total_despues_origen"] - div_pos["ret_dest"])
                ret_devol = div_pos["retentionReturnedBaseCurrency"].apply(lambda x: _to_float_div(x, 0.0)) if "retentionReturnedBaseCurrency" in div_pos.columns else 0.0
                div_pos["total_neto_devol"] = div_pos["total_neto"] + ret_devol

                agg_div = div_pos.groupby("ticker", as_index=False).agg(
                    total_bruto=("total_bruto", "sum"),
                    impuesto_ext=("impuesto_ext", "sum"),
                    total_despues_origen=("total_despues_origen", "sum"),
                    ret_dest=("ret_dest", "sum"),
                    total_neto=("total_neto", "sum"),
                    total_neto_devol=("total_neto_devol", "sum"),
                ).rename(columns={
                    "ticker": "Posición",
                    "total_bruto": "Total bruto (€)",
                    "impuesto_ext": "Impuesto satisf. en el extranjero (€)",
                    "total_despues_origen": "Total bruto después de origen (€)",
                    "ret_dest": "Retención en dest. realizada (€)",
                    "total_neto": "Total neto cobrado (€)",
                    "total_neto_devol": "Total neto con devolución (€)",
                })
                # Fila TOTAL
                fila_total = pd.DataFrame([{
                    "Posición": "TOTAL",
                    "Total bruto (€)": agg_div["Total bruto (€)"].sum(),
                    "Impuesto satisf. en el extranjero (€)": agg_div["Impuesto satisf. en el extranjero (€)"].sum(),
                    "Total bruto después de origen (€)": agg_div["Total bruto después de origen (€)"].sum(),
                    "Retención en dest. realizada (€)": agg_div["Retención en dest. realizada (€)"].sum(),
                    "Total neto cobrado (€)": agg_div["Total neto cobrado (€)"].sum(),
                    "Total neto con devolución (€)": agg_div["Total neto con devolución (€)"].sum(),
                }])
                agg_div = pd.concat([fila_total, agg_div], ignore_index=True)
                st.dataframe(
                    agg_div.style.format({c: lambda x: fmt_eur(x) if isinstance(x, (int, float)) else x for c in agg_div.columns if "€" in str(c)}, na_rep="–"),
                    use_container_width=True,
                )

        with tab_div_pos:
            st.subheader("Posiciones vivas (FIFO por lotes)")
            if lots_df.empty:
                st.info("No hay lotes vivos.")
            else:
                lots_show = lots_df.sort_values(["Broker", "Ticker", "Fecha origen"]).copy()
                for old, new in [("ticker", "Ticker"), ("broker", "Broker"), ("nombre", "Nombre")]:
                    if old in lots_show.columns and new not in lots_show.columns:
                        lots_show = lots_show.rename(columns={old: new})
                col_order = [c for c in ["Ticker", "Broker", "Nombre", "Fecha origen", "Cantidad", "Precio medio €", "Tipo activo"] if c in lots_show.columns]
                st.dataframe(lots_show[col_order] if col_order else lots_show, use_container_width=True)

            st.subheader("Ventas (impacto fiscal FIFO)")
            if sales_df.empty:
                st.info("No hay ventas registradas.")
            else:
                sales_show = sales_df.sort_values(["Fecha venta", "Broker", "Ticker"]).copy()
                col_order_s = [c for c in ["Ticker", "Broker", "Nombre", "Fecha venta", "Cantidad vendida", "Valor compra histórico (€)", "Valor venta (€)", "Plusvalía / Minusvalía (€)", "Retención dest. (€)", "Tipo activo"] if c in sales_show.columns]
                st.dataframe(sales_show[col_order_s] if col_order_s else sales_show, use_container_width=True)
                total_pnl = sales_show["Plusvalía / Minusvalía (€)"].sum() if "Plusvalía / Minusvalía (€)" in sales_show.columns else 0
                st.write(f"**Plusvalía/Minusvalía total**: {fmt_eur(total_pnl)}")

        return

    if vista == "Brokers":
        st.header("Cuentas bróker o wallet")
        _init_db_brokers()
        _migrate_brokers_from_data()

        st.subheader("Añadir cuenta")
        with st.form("form_nuevo_broker"):
            nuevo_broker = st.text_input("Nombre de la cuenta", placeholder="Ej. Trade Republic, MyInvestor", key="nuevo_broker_nombre")
            if st.form_submit_button("Añadir"):
                ok, msg = add_broker(nuevo_broker)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        st.subheader("Cuentas existentes")
        brokers_detail = get_brokers_with_details()
        if not brokers_detail:
            st.info("No hay cuentas. Añade una arriba o aparecerán aquí al tener movimientos.")
        else:
            paises = ["", "España", "Alemania", "Reino Unido", "Francia", "Italia", "Países Bajos", "Portugal", "Otro"]
            for i, acc in enumerate(brokers_detail):
                with st.expander(f"**{acc['name']}**", expanded=False):
                    st.markdown("**Cuenta bróker o wallet**")
                    st.caption("Nombre y país identifican la cuenta. Puedes crear tantas cuentas como quieras del mismo bróker.")
                    nombre_cuenta = st.text_input("Nombre de la cuenta", value=acc["name"], key=f"acc_name_{acc['id']}")
                    idx_pais = paises.index(acc["country"]) if acc["country"] in paises else 0
                    pais = st.selectbox("País", options=paises, index=idx_pais, key=f"acc_country_{acc['id']}")
                    multidivisa = st.toggle("Multidivisa", value=acc["multidivisa"], key=f"acc_multidivisa_{acc['id']}", help="Cuenta en varias monedas")
                    retiene_destino = st.toggle("Retiene en destino", value=acc["retiene_en_destino"], key=f"acc_retiene_{acc['id']}", help="El bróker retiene impuestos en España (retención en destino)")
                    col_b1, col_b2, col_b3 = st.columns(3)
                    with col_b1:
                        if st.button("BORRAR", key=f"btn_borrar_{acc['id']}", type="secondary"):
                            ok, msg = delete_broker(acc["id"])
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                    with col_b2:
                        if st.button("CANCELAR", key=f"btn_cancel_{acc['id']}"):
                            st.rerun()
                    with col_b3:
                        if st.button("GUARDAR CUENTA", type="primary", key=f"btn_guardar_{acc['id']}"):
                            ok, msg = update_broker_account(
                                acc["id"],
                                nombre_cuenta,
                                country=pais or "",
                                multidivisa=multidivisa,
                                retiene_en_destino=retiene_destino,
                            )
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
        return

    # Vista normal de cartera agregada (acciones + fondos; filtro por tipo abajo)
    # FIFO en ventas y traspasos para coincidir con Filios
    positions_acc = compute_positions_fifo(df)
    positions_acc["Origen"] = "Acciones"
    df_fondos = load_data_fondos()
    positions_fondos_df = positions_fondos_to_dataframe(compute_positions_fondos(df_fondos))
    positions_fondos_df["Origen"] = "Fondos"
    # Criptos: posiciones desde movimientos_criptos
    df_crip_cartera = load_data_criptos()
    positions_crip_df = compute_positions_criptos(df_crip_cartera)
    if not positions_crip_df.empty:
        # Normalizar nombres de columnas (problemas de encoding del símbolo € en consola)
        positions_crip_df = positions_crip_df.rename(
            columns={
                "Broker": "Broker",
                "Ticker": "Ticker",
                "Ticker_Yahoo": "Ticker_Yahoo",
                "Nombre": "Nombre",
                "Cantidad": "Titulos",
                "Inversion \uFFFD": "Inversion €",
                "Inversion ?": "Inversion €",
            }
        )
        if "Inversion €" not in positions_crip_df.columns and "Inversion \uFFFD" in positions_crip_df.columns:
            positions_crip_df["Inversion €"] = positions_crip_df["Inversion \uFFFD"]
        positions_crip_df["Tipo activo"] = "crypto"
        positions_crip_df["Origen"] = "Criptos"
        # Rellenar campos mínimos para compatibilidad con enrich_with_market_data
        positions_crip_df["Moneda Activo"] = "EUR"
        positions_crip_df["Moneda Yahoo"] = "EUR"
        positions_base = pd.concat([positions_acc, positions_fondos_df, positions_crip_df], ignore_index=True)
    else:
        positions_base = pd.concat([positions_acc, positions_fondos_df], ignore_index=True)

    if positions_base.empty:
        st.info("No hay posiciones abiertas en la cartera.")
        return

    # Cotizaciones: caché en disco + sesión; actualizar solo al pulsar el botón
    cotiz_signature = _cotizaciones_signature(positions_base)
    if "cartera_enriched" not in st.session_state:
        st.session_state["cartera_enriched"] = None
    if "cartera_enriched_updated_at" not in st.session_state:
        st.session_state["cartera_enriched_updated_at"] = None
    # Si hemos introducido posiciones de criptos, forzamos a recalcular (para evitar arrastrar caché antigua sin coste).
    if "Tipo activo" in positions_base.columns:
        tipos_all = positions_base["Tipo activo"].astype(str).str.strip().str.lower()
        if (tipos_all == "crypto").any():
            st.session_state["cartera_enriched"] = None
            st.session_state["cartera_enriched_updated_at"] = None

    # Si no hay datos en sesión, intentar cargar última cotización guardada
    if st.session_state["cartera_enriched"] is None and cotiz_signature:
        cached_df, cached_at = load_cotizaciones_cache(cotiz_signature)
        if cached_df is not None and len(cached_df) == len(positions_base):
            st.session_state["cartera_enriched"] = cached_df
            st.session_state["cartera_enriched_updated_at"] = cached_at

    precios_manuales = load_precios_manuales()

    col_act, col_ref = st.columns([1, 1])
    with col_act:
        if st.button("Actualizar cotizaciones", type="primary"):
            with st.spinner("Obteniendo precios actuales..."):
                st.session_state["cartera_enriched"] = enrich_with_market_data(
                    positions_base.copy(), manual_prices=precios_manuales
                )
                st.session_state["cartera_enriched_updated_at"] = datetime.now().isoformat()
                if cotiz_signature:
                    save_cotizaciones_cache(st.session_state["cartera_enriched"], cotiz_signature)
            st.rerun()
    with col_ref:
        if st.button("🔄 Refrescar datos", key="btn_refresh_cartera", help="Recarga movimientos desde la base de datos (útil tras correcciones externas o Recalcular totales)"):
            load_data.clear()
            load_data_fondos.clear()
            if hasattr(load_data_criptos, "clear"):
                load_data_criptos.clear()
            clear_cotizaciones_cache()
            st.session_state["cartera_enriched"] = None
            st.session_state["cartera_enriched_updated_at"] = None
            st.rerun()

    # Si las posiciones base han cambiado (nº filas, brokers/tickers o inversión), invalidamos el enriquecido
    inv_changed = False
    if (
        st.session_state["cartera_enriched"] is not None
        and "Inversion €" in positions_base.columns
        and "Inversion €" in st.session_state["cartera_enriched"].columns
        and len(st.session_state["cartera_enriched"]) == len(positions_base)
    ):
        ce = st.session_state["cartera_enriched"]["Inversion €"].fillna(0).round(2)
        pb = positions_base["Inversion €"].fillna(0).round(2)
        inv_changed = not ce.equals(pb)
    if (
        st.session_state["cartera_enriched"] is not None
        and (
            len(st.session_state["cartera_enriched"]) != len(positions_base)
            or set(st.session_state["cartera_enriched"]["Broker"].unique()) != set(positions_base["Broker"].unique())
            or set(st.session_state["cartera_enriched"]["Ticker_Yahoo"].unique())
            != set(positions_base["Ticker_Yahoo"].unique())
            or inv_changed
        )
    ):
        st.session_state["cartera_enriched"] = None
        st.session_state["cartera_enriched_updated_at"] = None

    if st.session_state["cartera_enriched"] is None:
        # Sin datos de mercado: mostramos posiciones con columnas vacías para precios
        positions = positions_base.copy()
        positions["Precio Actual"] = math.nan
        positions["Valor Mercado €"] = math.nan
        positions["Plusvalia €"] = math.nan
        positions["Plusvalia %"] = math.nan
        positions["GyP hoy %"] = math.nan
        positions["GyP hoy €"] = math.nan
        positions["Moneda Yahoo"] = positions.get("Moneda Activo", "EUR")
        positions["Cierre Previo"] = math.nan
        # Aplicar precios manuales si existen
        ticker_col = "Ticker_Yahoo" if "Ticker_Yahoo" in positions.columns else "Ticker"
        for idx, row in positions.iterrows():
            t = str(row.get(ticker_col) or "").strip()
            if t and t in precios_manuales:
                positions.at[idx, "Precio Actual"] = precios_manuales[t]
                positions.at[idx, "Valor Mercado €"] = positions.at[idx, "Titulos"] * precios_manuales[t]
                positions.at[idx, "Plusvalia €"] = positions.at[idx, "Valor Mercado €"] - positions.at[idx, "Inversion €"]
                inv = positions.at[idx, "Inversion €"]
                positions.at[idx, "Plusvalia %"] = (positions.at[idx, "Plusvalia €"] / inv * 100) if inv and abs(inv) > 0 else math.nan
                positions.at[idx, "GyP hoy %"] = math.nan
                positions.at[idx, "GyP hoy €"] = 0.0
        st.info("Pulsa **Actualizar cotizaciones** para cargar precios actuales y valor de mercado.")
    else:
        positions = st.session_state["cartera_enriched"].copy()
        # Reaplicar precios manuales por si se añadieron nuevos (p. ej. warrants sin Yahoo)
        ticker_col = "Ticker_Yahoo" if "Ticker_Yahoo" in positions.columns else "Ticker"
        for idx, row in positions.iterrows():
            t = str(row.get(ticker_col) or "").strip()
            if t and t in precios_manuales and (pd.isna(row.get("Precio Actual")) or row.get("Precio Actual") is None):
                positions.at[idx, "Precio Actual"] = precios_manuales[t]
                positions.at[idx, "Precio Actual €"] = precios_manuales[t]
                positions.at[idx, "Valor Mercado €"] = positions.at[idx, "Titulos"] * precios_manuales[t]
                positions.at[idx, "Plusvalia €"] = positions.at[idx, "Valor Mercado €"] - positions.at[idx, "Inversion €"]
                inv = positions.at[idx, "Inversion €"]
                positions.at[idx, "Plusvalia %"] = (positions.at[idx, "Plusvalia €"] / inv * 100) if inv and abs(inv) > 0 else math.nan
        updated_at = st.session_state.get("cartera_enriched_updated_at")
        if updated_at:
            try:
                dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                st.caption(f"Cotizaciones del {dt.strftime('%d/%m/%Y %H:%M')}. Pulsa **Actualizar cotizaciones** para refrescar.")
            except Exception:
                st.caption("Pulsa **Actualizar cotizaciones** para refrescar los precios.")
        else:
            st.caption("Pulsa **Actualizar cotizaciones** para refrescar los precios.")

    # Expander para precios manuales (Otros/warrants sin cotización en Yahoo)
    ticker_col_pm = "Ticker_Yahoo" if "Ticker_Yahoo" in positions.columns else "Ticker"
    precio_col = "Precio Actual €" if "Precio Actual €" in positions.columns else "Precio Actual"
    sin_cotiz = []
    for idx, row in positions.iterrows():
        t = str(row.get(ticker_col_pm) or "").strip()
        if t and (pd.isna(row.get(precio_col)) or row.get(precio_col) is None):
            tipo = str(row.get("Tipo activo", "") or "").strip().lower()
            sin_cotiz.append((row.get("Broker"), t, row.get("Nombre", t), tipo))
    if sin_cotiz:
        with st.expander("📝 Precios manuales (posiciones sin cotización en Yahoo)", expanded=False):
            st.caption("Introduce el precio actual en EUR para calcular valor de mercado y plusvalía.")
            # Selector de tipo de activo
            tipos_presentes = set()
            for _, _, _, tipo in sin_cotiz:
                t = (tipo or "").strip().lower()
                if t in ("stock", "etf", "fund", "crypto", "warrant"):
                    tipos_presentes.add(t)
                elif t and t not in ("nan", ""):
                    tipos_presentes.add("stock")
            pm_tipo_options = ["Todos", "Acciones", "ETFs", "Fondos", "Criptos", "Otros"]
            pm_tipo_sel = st.radio("Tipo de activo", options=pm_tipo_options, index=0, horizontal=True, key="pm_tipo_activo")
            # Filtrar sin_cotiz según tipo
            def _pm_match_tipo(tipo: str, sel: str) -> bool:
                if sel == "Todos":
                    return True
                if sel == "Acciones":
                    return tipo in ("stock", "") or tipo not in ("etf", "fund", "crypto", "warrant")
                if sel == "ETFs":
                    return tipo == "etf"
                if sel == "Fondos":
                    return tipo == "fund"
                if sel == "Criptos":
                    return tipo == "crypto"
                if sel == "Otros":
                    return tipo == "warrant"
                return True
            sin_cotiz_filtrado = [(b, t, n) for b, t, n, tipo in sin_cotiz if _pm_match_tipo(tipo, pm_tipo_sel)]
            pm_updated = dict(precios_manuales)
            for broker, ticker, nombre in sin_cotiz_filtrado:
                key_pm = f"pm_{broker}_{ticker}".replace(" ", "_")
                val_actual = precios_manuales.get(ticker, "")
                nuevo = st.number_input(
                    f"{ticker} ({nombre})",
                    value=float(val_actual) if val_actual else 0.0,
                    min_value=0.0,
                    step=0.01,
                    format="%.4f",
                    key=key_pm,
                )
                if nuevo and nuevo > 0:
                    pm_updated[ticker] = nuevo
            if st.button("Guardar precios manuales", key="btn_guardar_pm"):
                save_precios_manuales(pm_updated)
                st.success("Precios guardados. Recargando…")
                st.rerun()

    # Selector de broker en la página: GLOBAL a la izquierda, luego brokers ordenados
    brokers = sorted(positions["Broker"].dropna().unique().tolist())
    options = ["GLOBAL"] + brokers
    selected = st.radio(
        "Broker",
        options=options,
        index=0,
        horizontal=True,
    )

    if selected == "GLOBAL":
        view = positions.copy()
    else:
        view = positions[positions["Broker"] == selected].copy()

    # Selector de tipo de activo: Todos / Acciones / ETFs / Fondos / Criptos / Otros
    tipo_options = ["Todos", "Acciones", "ETFs"]
    if "Tipo activo" in view.columns:
        tipos_all = view["Tipo activo"].astype(str).str.strip().str.lower()
        if (tipos_all == "fund").any():
            tipo_options.append("Fondos")
        if (tipos_all == "crypto").any():
            tipo_options.append("Criptos")
        if (tipos_all == "warrant").any():
            tipo_options.append("Otros")
    tipo_sel = st.radio(
        "Tipo de activo",
        options=tipo_options,
        index=0,
        horizontal=True,
    )
    if "Tipo activo" in view.columns and tipo_sel != "Todos":
        tipos = view["Tipo activo"].astype(str).str.strip().str.lower()
        if tipo_sel == "Acciones":
            mask = (tipos == "stock") | (
                (tipos != "etf")
                & (tipos != "fund")
                & (tipos != "crypto")
                & (tipos != "warrant")
                & (tipos != "")
                & (tipos != "nan")
            )
            view = view[mask]
        elif tipo_sel == "ETFs":
            view = view[tipos == "etf"]
        elif tipo_sel == "Fondos":
            view = view[tipos == "fund"]
        elif tipo_sel == "Criptos":
            view = view[tipos == "crypto"]
        elif tipo_sel == "Otros":
            view = view[tipos == "warrant"]

    # Aseguramos columnas necesarias antes de agrupar
    if "Moneda Yahoo" not in view.columns:
        # Si no existe, usamos Moneda Activo o EUR como respaldo
        view["Moneda Yahoo"] = view.get("Moneda Activo", pd.Series(["EUR"] * len(view), index=view.index))
    if "Precio Medio Moneda" not in view.columns:
        view["Precio Medio Moneda"] = 0.0

    # Agrupar por mismo activo (Ticker_Yahoo): una fila por ticker con totales
    view["_cost_local"] = view["Precio Medio Moneda"].fillna(0) * view["Titulos"]
    view["GyP hoy €"] = view.get("GyP hoy €", pd.Series(0.0, index=view.index)).fillna(0)

    grouped = view.groupby("Ticker_Yahoo", as_index=False).agg(
        {
            "Ticker": "first",
            "Nombre": "first",
            "Titulos": "sum",
            "Inversion €": "sum",
            "Valor Mercado €": "sum",
            "Plusvalia €": "sum",
            "GyP hoy €": "sum",
            "Precio Actual": "first",
            "Moneda Activo": "first",
            "Moneda Yahoo": "first",
            "Cierre Previo": "first",
            "_cost_local": "sum",
        }
    )
    grouped["Precio Medio €"] = grouped["Inversion €"] / grouped["Titulos"].replace(0, np.nan)
    grouped["Precio Medio Moneda"] = grouped["_cost_local"] / grouped["Titulos"].replace(0, np.nan)
    grouped["Plusvalia %"] = (
        grouped["Plusvalia €"] / grouped["Inversion €"].replace(0, np.nan) * 100.0
    )
    grouped["GyP hoy %"] = (
        grouped["GyP hoy €"] / grouped["Valor Mercado €"].replace(0, np.nan) * 100.0
    )
    grouped = grouped.drop(columns=["_cost_local"])
    grouped["Broker"] = "Todos" if selected == "GLOBAL" else selected
    view = grouped

    # Fila de métricas (4 columnas) bajo los selectores de broker
    total_inversion = view["Inversion €"].sum()
    total_valor = view["Valor Mercado €"].sum()
    total_plusvalia = total_valor - total_inversion
    total_plusvalia_pct = (
        (total_plusvalia / total_inversion * 100.0) if abs(total_inversion) > 0 else math.nan
    )
    total_gyp_hoy = view.get("GyP hoy €", pd.Series(0.0, index=view.index)).sum()
    total_valor_mercado = view["Valor Mercado €"].sum()
    gyp_hoy_pct = (
        (total_gyp_hoy / total_valor_mercado * 100.0)
        if total_valor_mercado and abs(total_valor_mercado) > 0
        else math.nan
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Valor cartera", fmt_eur(total_valor))
        st.caption(f"Total invertido: {fmt_eur(total_inversion)}")
    with c2:
        st.metric(
            "G&P totales",
            fmt_eur(total_plusvalia),
            f"{total_plusvalia_pct:.2f} %" if not pd.isna(total_plusvalia_pct) else None,
        )
    with c3:
        st.metric(
            "G&P hoy",
            fmt_eur(total_gyp_hoy),
            f"{gyp_hoy_pct:.2f} %" if not pd.isna(gyp_hoy_pct) else None,
        )
    with c4:
        div_df = load_dividendos()
        total_dividendos = 0.0
        if not div_df.empty:
            if selected != "GLOBAL" and "broker" in div_df.columns:
                div_df = div_df[div_df["broker"].astype(str).str.strip() == str(selected).strip()]
            def _div_val(row):
                for col in ("netoWithReturnBaseCurrency", "totalNetoBaseCurrency", "netoBaseCurrency"):
                    v = row.get(col)
                    if v is not None and str(v).strip() != "" and not (isinstance(v, float) and pd.isna(v)):
                        return _to_float(v, 0.0)
                return 0.0

            total_dividendos = sum(_div_val(row) for _, row in div_df.iterrows())
        rent_dvdos_pct = (
            (total_dividendos / total_inversion * 100.0) if total_inversion and abs(total_inversion) > 1e-9 else 0.0
        )
        st.metric("Total Dividendos", fmt_eur(total_dividendos))
        st.caption(f"Rent. dvdos sobre invertido: {rent_dvdos_pct:.2f}%")

    # Tabla de detalle
    table = view.copy()

    # Componer precio actual con símbolo/código de moneda
    def format_price_with_ccy(row: pd.Series) -> str:
        price = row.get("Precio Actual")
        ccy = row.get("Moneda Activo") or row.get("Moneda Yahoo") or ""
        if pd.isna(price):
            return "-"
        try:
            val = float(price)
        except Exception:
            return str(price)
        # Mapeo simple de código de moneda a símbolo
        symbol_map = {
            "EUR": "€",
            "USD": "$",
            "GBP": "£",
            "CHF": "CHF",
            "JPY": "¥",
        }
        symbol = symbol_map.get(str(ccy).upper(), str(ccy))
        # Mostramos siempre el símbolo/código a la derecha de la cifra
        return f"{val:,.2f} {symbol}"

    # Precio medio en la moneda original + símbolo
    def format_avg_price_with_ccy(row: pd.Series) -> str:
        price = row.get("Precio Medio Moneda")
        ccy = row.get("Moneda Activo") or row.get("Moneda Yahoo") or ""
        if pd.isna(price):
            return "-"
        try:
            val = float(price)
        except Exception:
            return str(price)
        symbol_map = {
            "EUR": "€",
            "USD": "$",
            "GBP": "£",
            "CHF": "CHF",
            "JPY": "¥",
        }
        symbol = symbol_map.get(str(ccy).upper(), str(ccy))
        return f"{val:,.2f} {symbol}"

    table["Ultima cotizacion (moneda)"] = table.apply(format_price_with_ccy, axis=1)
    table["Precio medio (moneda)"] = table.apply(format_avg_price_with_ccy, axis=1)
    table["Precio md + com/imp (€)"] = table["Precio Medio €"]
    table["Total inv + com/imp (€)"] = table["Inversion €"]
    table["Valor mercado (€)"] = table["Valor Mercado €"]
    table["GyP no realizadas %"] = table["Plusvalia %"]
    table["GyP no realizadas (€)"] = table["Plusvalia €"]
    table["GyP hoy %"] = table.get("GyP hoy %")
    table["GyP hoy (€)"] = table.get("GyP hoy €")

    display_cols = [
        "Ticker",
        "Nombre",
        "Titulos",
        "Ultima cotizacion (moneda)",
        "Precio medio (moneda)",
        "Precio md + com/imp (€)",
        "Total inv + com/imp (€)",
        "Valor mercado (€)",
        "GyP no realizadas %",
        "GyP no realizadas (€)",
        "GyP hoy %",
        "GyP hoy (€)",
        "Broker",
    ]

    # Formateo numérico para la visualización
    styler = (
        table[display_cols]
        .style.format(
            {
                "Titulos": fmt_qty,
                "Precio md + com/imp (€)": fmt_eur,
                "Total inv + com/imp (€)": fmt_eur,
                "Valor mercado (€)": fmt_eur,
                "GyP no realizadas (€)": fmt_eur,
                "GyP no realizadas %": lambda v: "-" if pd.isna(v) else f"{v:.2f} %",
                "GyP hoy (€)": fmt_eur,
                "GyP hoy %": lambda v: "-" if pd.isna(v) else f"{v:.2f} %",
            }
        )
        .applymap(color_pnl, subset=["GyP no realizadas (€)", "GyP no realizadas %", "GyP hoy (€)", "GyP hoy %"])
    )

    st.dataframe(styler, use_container_width=True)


if __name__ == "__main__":
    main()

