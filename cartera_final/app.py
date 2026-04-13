import hashlib
import json
import math
import re
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime, time as dt_time
from collections import OrderedDict
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from filios_core.config import (
    COTIZACIONES_CACHE_PATH,
    COTIZACIONES_META_PATH,
    CSV_PATH,
    DATA_DIR as _DATA_DIR,
    DB_PATH,
    DIVIDENDOS_CSV_PATH,
    FONDOS_CSV_PATH,
    MOVIMIENTOS_CRIPTOS_CSV_PATH,
    PRECIOS_MANUALES_PATH,
)
from filios_core.constants import (
    CRYPTO_BROKER_IDS,
    CRYPTO_TICKER_NAMES,
    CSV_DECIMAL,
    CSV_ENCODING,
    CSV_SEP,
    DECIMALS_POSITION,
    DIVIDENDOS_COLUMNS,
    MIN_POSITION,
    MOVIMIENTOS_COLUMNS,
    MOVIMIENTOS_CRIPTOS_COLUMNS,
)
from filios_core.db import get_db as _get_db
from filios_core.fifo import (
    _cripto_chrono_type_order,
    _fifo_queue_key_stocks,
    _fifo_queue_key_stocks_cartera,
    _fifo_split_affected_keys_stocks,
    _fifo_split_affected_keys_stocks_cartera,
    compute_fifo_all,
    compute_fifo_criptos,
    compute_fifo_fondos,
    compute_positions_criptos,
    compute_positions_fondos,
)
from filios_core.util import safe_get as _safe_get, to_float as _to_float
from filios_core.isin import (
    _catalog_origen_requires_isin,
    _fifo_resolve_isin_row,
    _init_instrument_catalog,
    _isin_required_acciones_etf,
    _isin_required_fondos,
    _lookup_isin_for_ticker_yahoo,
    _looks_like_isin,
    _norm_isin_field,
    _resolve_movimiento_isin,
    lookup_ticker_yahoo_by_isin,
)


def _style_map(styler, func, **kwargs):
    """pandas ≥2.1: Styler.map; versiones anteriores: Styler.applymap."""
    mapper = getattr(styler, "map", None)
    if mapper is not None:
        return mapper(func, **kwargs)
    return styler.applymap(func, **kwargs)


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


def _ensure_movimientos_isin_column():
    """Añade columna isin si la tabla existe pero aún no la tiene (bases antiguas)."""
    with _get_db() as conn:
        for tbl in ("movimientos", "movimientos_fondos", "movimientos_criptos"):
            try:
                cur = conn.execute(f'PRAGMA table_info("{tbl}")')
                names = {row[1] for row in cur.fetchall()}
                if names and "isin" not in names:
                    conn.execute(f'ALTER TABLE "{tbl}" ADD COLUMN isin TEXT')
            except sqlite3.OperationalError:
                pass
        conn.commit()


def _ensure_movimientos_criptos_schema():
    """Añade columnas faltantes a movimientos_criptos (BDs anteriores a MOVIMIENTOS_CRIPTOS_COLUMNS)."""
    with _get_db() as conn:
        try:
            cur = conn.execute('PRAGMA table_info("movimientos_criptos")')
            names = {row[1] for row in cur.fetchall()}
        except sqlite3.OperationalError:
            return
        if not names:
            return
        for col in MOVIMIENTOS_CRIPTOS_COLUMNS:
            if col not in names:
                try:
                    conn.execute(f'ALTER TABLE "movimientos_criptos" ADD COLUMN "{col}" TEXT')
                except sqlite3.OperationalError:
                    pass
        conn.commit()


def _init_db():
    """Crea la tabla movimientos si no existe."""
    cols_sql = ", ".join(f'"{c}" TEXT' for c in MOVIMIENTOS_COLUMNS)
    with _get_db() as conn:
        conn.execute(f"CREATE TABLE IF NOT EXISTS movimientos ({cols_sql})")
    _ensure_movimientos_isin_column()


def _init_db_cartera_snapshot_mes():
    """Serie mensual de valor de mercado (cierre ~último día hábil del mes). Recalculable bajo demanda."""
    with _get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cartera_snapshot_mes (
                anio INTEGER NOT NULL,
                mes INTEGER NOT NULL,
                fecha_valoracion TEXT NOT NULL,
                valor_mercado_eur REAL NOT NULL,
                valor_invertido_eur REAL NOT NULL DEFAULT 0,
                num_lineas INTEGER,
                computed_at TEXT NOT NULL,
                PRIMARY KEY (anio, mes)
            )
            """
        )
        cur = conn.execute("PRAGMA table_info(cartera_snapshot_mes)")
        col_names = {row[1] for row in cur.fetchall()}
        if "valor_invertido_eur" not in col_names:
            conn.execute(
                "ALTER TABLE cartera_snapshot_mes ADD COLUMN valor_invertido_eur REAL NOT NULL DEFAULT 0"
            )
        conn.commit()


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
        try:
            cur = conn.execute("PRAGMA table_info(dividendos)")
            names = {row[1] for row in cur.fetchall()}
            if names and "isin" not in names:
                conn.execute('ALTER TABLE dividendos ADD COLUMN "isin" TEXT')
        except sqlite3.OperationalError:
            pass
        conn.commit()


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
    placeholders = ", ".join("?" for _ in DIVIDENDOS_COLUMNS)
    vals = [_row_to_db_val(row.get(c, "")) for c in DIVIDENDOS_COLUMNS]
    with _get_db() as conn:
        conn.execute(
            f'INSERT INTO dividendos ({", ".join(DIVIDENDOS_COLUMNS)}) VALUES ({placeholders})',
            vals,
        )
        conn.commit()
    if hasattr(load_dividendos, "clear"):
        load_dividendos.clear()


def _init_db_intereses_extranjero() -> None:
    """Intereses P2P / crowdlending: bruto, retención en origen/destino, bonus (registro manual por ejercicio)."""
    with _get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS intereses_extranjero (
                ejercicio INTEGER NOT NULL,
                plataforma TEXT NOT NULL,
                bruto_eur REAL NOT NULL,
                retencion_extranjero_eur REAL NOT NULL DEFAULT 0,
                bonus_eur REAL NOT NULL DEFAULT 0,
                retencion_destino_eur REAL NOT NULL DEFAULT 0,
                notas TEXT
            )
            """
        )
        cur = conn.execute("PRAGMA table_info(intereses_extranjero)")
        col_names = {row[1] for row in cur.fetchall()}
        if "bonus_eur" not in col_names:
            conn.execute("ALTER TABLE intereses_extranjero ADD COLUMN bonus_eur REAL NOT NULL DEFAULT 0")
        if "retencion_destino_eur" not in col_names:
            conn.execute("ALTER TABLE intereses_extranjero ADD COLUMN retencion_destino_eur REAL NOT NULL DEFAULT 0")
        conn.commit()


@st.cache_data
def load_intereses_extranjero() -> pd.DataFrame:
    _init_db_intereses_extranjero()
    with _get_db() as conn:
        df = pd.read_sql(
            "SELECT rowid AS _rowid_, ejercicio, plataforma, bruto_eur, retencion_extranjero_eur, bonus_eur, retencion_destino_eur, notas FROM intereses_extranjero ORDER BY ejercicio DESC, plataforma ASC",
            conn,
        )
    if df.empty:
        return pd.DataFrame(
            columns=[
                "_rowid_",
                "ejercicio",
                "plataforma",
                "bruto_eur",
                "retencion_extranjero_eur",
                "bonus_eur",
                "retencion_destino_eur",
                "notas",
            ]
        )
    if "bonus_eur" not in df.columns:
        df["bonus_eur"] = 0.0
    if "retencion_destino_eur" not in df.columns:
        df["retencion_destino_eur"] = 0.0
    return df


def append_interes_extranjero(
    ejercicio: int,
    plataforma: str,
    bruto_eur: float,
    retencion_extranjero_eur: float,
    bonus_eur: float = 0.0,
    retencion_destino_eur: float = 0.0,
    notas: str = "",
) -> None:
    _init_db_intereses_extranjero()
    with _get_db() as conn:
        conn.execute(
            """
            INSERT INTO intereses_extranjero (ejercicio, plataforma, bruto_eur, retencion_extranjero_eur, bonus_eur, retencion_destino_eur, notas)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(ejercicio),
                (plataforma or "").strip(),
                float(bruto_eur),
                float(retencion_extranjero_eur),
                float(bonus_eur),
                float(retencion_destino_eur),
                (notas or "").strip(),
            ),
        )
        conn.commit()
    if hasattr(load_intereses_extranjero, "clear"):
        load_intereses_extranjero.clear()


def delete_intereses_extranjero_by_rowids(rowids: list[int]) -> int:
    if not rowids:
        return 0
    _init_db_intereses_extranjero()
    n = 0
    with _get_db() as conn:
        for rid in rowids:
            cur = conn.execute("DELETE FROM intereses_extranjero WHERE rowid = ?", (int(rid),))
            n += cur.rowcount
        conn.commit()
    if hasattr(load_intereses_extranjero, "clear"):
        load_intereses_extranjero.clear()
    return n


def update_interes_extranjero(
    rowid: int,
    ejercicio: int,
    plataforma: str,
    bruto_eur: float,
    retencion_extranjero_eur: float,
    bonus_eur: float,
    retencion_destino_eur: float = 0.0,
    notas: str = "",
) -> None:
    _init_db_intereses_extranjero()
    with _get_db() as conn:
        conn.execute(
            """
            UPDATE intereses_extranjero
            SET ejercicio = ?, plataforma = ?, bruto_eur = ?, retencion_extranjero_eur = ?, bonus_eur = ?, retencion_destino_eur = ?, notas = ?
            WHERE rowid = ?
            """,
            (
                int(ejercicio),
                (plataforma or "").strip(),
                float(bruto_eur),
                float(retencion_extranjero_eur),
                float(bonus_eur),
                float(retencion_destino_eur),
                (notas or "").strip(),
                int(rowid),
            ),
        )
        conn.commit()
    if hasattr(load_intereses_extranjero, "clear"):
        load_intereses_extranjero.clear()


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
    _ensure_movimientos_isin_column()


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

def _fondo_ticker_yahoo_valuation(r: dict) -> str:
    """Ticker Yahoo para cotización: movimientos, o ISIN → catálogo, o ISIN tal cual si no hay mapa."""
    ty = str(r.get("ticker_yahoo") or "").strip()
    t = str(r.get("ticker") or "").strip()
    for cand in (ty, t):
        if not cand:
            continue
        if not _looks_like_isin(cand):
            return cand
        mapped = lookup_ticker_yahoo_by_isin(cand)
        if mapped:
            return mapped
    return ty or t


def positions_fondos_to_dataframe(resumen: list[dict]) -> pd.DataFrame:
    """Convierte el resumen de compute_positions_fondos a DataFrame con columnas como positions (Cartera)."""
    if not resumen:
        return pd.DataFrame(columns=["Broker", "Ticker", "Ticker_Yahoo", "Nombre", "Titulos", "Precio Medio €", "Inversion €", "Moneda Activo", "Tipo activo", "Fecha origen"])
    rows = []
    for r in resumen:
        rows.append({
            "Broker": r["broker"],
            "Ticker": r["ticker"],
            "Ticker_Yahoo": _fondo_ticker_yahoo_valuation(r),
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


def _format_qty_streamlit_form(q: float) -> str:
    """Cantidad para st.text_input con coma decimal (estilo formulario operaciones)."""
    if q is None or (isinstance(q, float) and (math.isnan(q) or math.isinf(q))):
        return ""
    qf = float(q)
    if abs(qf) < MIN_POSITION:
        return ""
    fmt = f"{qf:.12f}".rstrip("0").rstrip(".")
    return fmt.replace(".", ",") if fmt else "0"


def qty_en_cartera_broker_yahoo(
    tipo_registro: str,
    broker: str,
    ticker_yahoo: str,
    pos_acc: pd.DataFrame,
    pos_fon: pd.DataFrame,
    pos_crip: pd.DataFrame,
) -> float:
    """
    Títulos en posición viva para (cuenta, ticker Yahoo). 0 si no hay fila o cantidad residual.
    """
    if not broker or not str(broker).strip():
        return 0.0
    ty = str(ticker_yahoo or "").strip()
    if not ty:
        return 0.0
    br = str(broker).strip()

    if tipo_registro in ("Acciones/ETFs", "Otros", "Opciones (Put/Call)"):
        pdf = pos_acc
        qty_col = "Titulos"
    elif tipo_registro == "Fondos":
        pdf = pos_fon
        qty_col = "Titulos"
    elif tipo_registro == "Criptos":
        pdf = pos_crip
        qty_col = "Cantidad" if pos_crip is not None and "Cantidad" in pos_crip.columns else "Titulos"
    else:
        return 0.0

    if pdf is None or pdf.empty or qty_col not in pdf.columns:
        return 0.0
    bcol = pdf["Broker"].astype(str).str.strip()
    ycol = pdf["Ticker_Yahoo"].astype(str).str.strip()
    m = (bcol == br) & (ycol == ty)
    sub = pdf.loc[m, qty_col]
    if sub.empty:
        return 0.0
    q = float(sub.iloc[0])
    return q if q > MIN_POSITION else 0.0


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


def _yahoo_symbol_for_history_lookup(key: str) -> str:
    """Si key es un ISIN catalogado, devuelve el Yahoo; si no, la misma key."""
    raw = (key or "").strip()
    if not raw:
        return raw
    if _looks_like_isin(raw):
        return lookup_ticker_yahoo_by_isin(raw) or raw
    return raw


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
    isin_raw = (isin or "").strip()
    isin = _norm_isin_field(isin_raw)
    if isin_raw and not isin:
        return False, "El ISIN indicado no tiene un formato válido (12 caracteres alfanuméricos)."
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
    _ensure_movimientos_isin_column()
    _ensure_movimientos_criptos_schema()


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
        "op_qty_nuevo", "op_qty_nuevo_pending_fill", "op_precio_nuevo", "op_total_nuevo",
        "op_com_nuevo", "op_tax_nuevo", "op_dest_nuevo",
        "new_isin",
        "tf_qty", "tf_qty_dest", "tf_qty_pending_fill", "tf_valor_eur",
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
        tipos_recalc = ("buy", "sell", "switch", "switchbuy", "optionbuy", "optionsell")
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

def _cripto_movimiento_tab_type_order(s: pd.Series) -> pd.Series:
    """Lista Movimientos (fecha descendente): comisión encima de traspaso a igual segundo."""
    t = s.astype(str).str.strip().str.lower()
    return t.map({"commission": 0, "brokertransfer": 1}).fillna(2)


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

    _sort_c = ["datetime_full"]
    _asc = [True]
    if "type" in df.columns:
        df["_tie_ct"] = _cripto_chrono_type_order(df["type"])
        _sort_c.append("_tie_ct")
        _asc.append(True)
    if "_rowid_" in df.columns:
        _sort_c.append("_rowid_")
        _asc.append(True)
    df = df.sort_values(_sort_c, ascending=_asc, kind="mergesort").drop(columns=["_tie_ct"], errors="ignore")
    df = df.reset_index(drop=True)

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
    if tipo_lower in ("sell", "switch", "optionsell"):
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

        if _mt in ("buy", "switchbuy", "optionbuy"):
            pos["quantity"] += qty_change
            pos["cost_eur"] += _to_float(total_eur)
            pos["cost_local"] += _to_float(price_local) * qty_change
        elif _mt == "optionsell":
            # Put/Call: venta de prima puede ser cierre de largo o apertura/aumento de corto (vendido en IBKR primero).
            sell_qty = abs(qty_change) if qty_change else 0.0
            if sell_qty < MIN_POSITION:
                continue
            qty_before = pos["quantity"]
            if qty_before > MIN_POSITION:
                if sell_qty > qty_before:
                    sell_qty = qty_before
                avg_cost_per_unit = pos["cost_eur"] / qty_before
                avg_cost_local = pos["cost_local"] / qty_before
                qty_after = qty_before - sell_qty
                pos["quantity"] = qty_after
                pos["cost_eur"] = avg_cost_per_unit * qty_after
                pos["cost_local"] = avg_cost_local * qty_after
            else:
                pos["quantity"] = qty_before - sell_qty
                pos["cost_eur"] -= _to_float(total_eur)
                pos["cost_local"] -= _to_float(price_local) * sell_qty
        elif _mt in ("sell", "switch"):
            sell_qty = abs(_to_float(qty_change))
            if sell_qty < MIN_POSITION:
                continue
            qty_before = float(pos["quantity"])
            te_row = _to_float(total_eur)
            pl_row = _to_float(price_local)
            if qty_before > MIN_POSITION:
                close = min(sell_qty, qty_before)
                avg_e = pos["cost_eur"] / qty_before
                avg_l = pos["cost_local"] / qty_before
                pos["quantity"] = qty_before - close
                pos["cost_eur"] = avg_e * (qty_before - close)
                pos["cost_local"] = avg_l * (qty_before - close)
                excess = sell_qty - close
                if excess > MIN_POSITION:
                    fr = excess / sell_qty if sell_qty > 1e-12 else 1.0
                    pos["quantity"] -= excess
                    pos["cost_eur"] -= te_row * fr
                    pos["cost_local"] -= pl_row * excess
            elif qty_before < -MIN_POSITION:
                pos["quantity"] = qty_before - sell_qty
                pos["cost_eur"] -= te_row
                pos["cost_local"] -= pl_row * sell_qty
            else:
                pos["quantity"] = -sell_qty
                pos["cost_eur"] = -te_row
                pos["cost_local"] = -pl_row * sell_qty
        else:
            continue

        if _mt in ("optionbuy", "optionsell"):
            pt_cl = str(_safe_get(row, "positionType") or "").strip().lower()
            if pt_cl in ("putoption", "calloption") and abs(pos["quantity"]) < MIN_POSITION:
                pos["quantity"] = 0.0
                pos["cost_eur"] = 0.0
                pos["cost_local"] = 0.0

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
    Cola por (cuenta, ISIN, divisa) cuando hay ISIN: el mismo ISIN en otro mercado/divisa
    no se mezcla; en la misma cuenta y divisa, tickers Yahoo distintos sí unifican (evita cortos ficticios).
    La vista fiscal sigue usando su lógica propia (_fifo_queue_key_stocks).
    """
    positions: dict[tuple, dict] = {}
    cat_cache: dict[str, str] = {}

    def ensure(key: tuple, row: pd.Series) -> None:
        if key not in positions:
            positions[key] = {
                "lots": [],
                "ticker_orig": _safe_get(row, "ticker"),
                "name": _safe_get(row, "name") or _safe_get(row, "ticker") or "",
                "pos_currency": _safe_get(row, "positionCurrency", "EUR"),
                "pos_type": _safe_get(row, "positionType", ""),
                "Broker": str(_safe_get(row, "broker") or "").strip(),
                "Ticker_Yahoo": str(
                    (_safe_get(row, "ticker_Yahoo") or _safe_get(row, "ticker") or "")
                ).strip(),
            }
        else:
            b = str(_safe_get(row, "broker") or "").strip()
            if b:
                positions[key]["Broker"] = b
            ty = str(_safe_get(row, "ticker_Yahoo") or _safe_get(row, "ticker") or "").strip()
            if ty:
                positions[key]["Ticker_Yahoo"] = ty
            to = _safe_get(row, "ticker")
            if to:
                positions[key]["ticker_orig"] = to
            nm = _safe_get(row, "name")
            if nm:
                positions[key]["name"] = nm
            pt = _safe_get(row, "positionType")
            if pt:
                positions[key]["pos_type"] = pt

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
            idx_split = {k: positions[k]["lots"] for k in positions}
            for k in _fifo_split_affected_keys_stocks_cartera(idx_split, row, broker, key_ticker, cat_cache):
                if k in positions:
                    for lot in positions[k]["lots"]:
                        lot["qty"] *= factor
            continue

        if not broker or qty is None or qty <= 0:
            if t not in (
                "buy",
                "switchbuy",
                "bonus",
                "stakereward",
                "sell",
                "switch",
                "brokertransfer",
                "optionbuy",
                "optionsell",
            ):
                continue

        if t == "brokertransfer":
            dest_raw = _safe_get(row, "brokerTransferNewBroker")
            if not dest_raw:
                continue
            dest_broker = str(dest_raw).strip()
            src_key = _fifo_queue_key_stocks_cartera(row, broker, key_ticker, cat_cache)
            ensure(src_key, row)
            if src_key[0] == "ISIN_BR_CCY":
                _, isin_k, _, ccy_k = src_key
                dst_key = ("ISIN_BR_CCY", isin_k, dest_broker, ccy_k)
            else:
                dst_key = ("PAIR", dest_broker, src_key[2])
            ensure(dst_key, row)
            positions[dst_key]["Broker"] = dest_broker
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

        key = _fifo_queue_key_stocks_cartera(row, broker, key_ticker, cat_cache)

        if t in ("buy", "switchbuy", "bonus", "stakereward"):
            ensure(key, row)
            lots = positions[key]["lots"]
            rem = float(qty)
            cost_eur_tot = _to_float(total_eur)
            rem = rem if rem > MIN_POSITION else 0.0
            while rem > MIN_POSITION and lots and float(lots[0]["qty"]) < -MIN_POSITION:
                lot = lots[0]
                abs_s = abs(float(lot["qty"]))
                take = min(rem, abs_s)
                frac = take / abs_s if abs_s > 1e-12 else 1.0
                lot["cost_eur"] *= 1.0 - frac
                lot["cost_local"] *= 1.0 - frac
                lot["qty"] += take
                rem -= take
                if abs(float(lot["qty"])) < MIN_POSITION:
                    lots.pop(0)
            if rem > MIN_POSITION:
                cost_eur = cost_eur_tot * (rem / qty) if qty > 1e-12 else cost_eur_tot
                cost_local = (price_local * rem) if price_local else 0.0
                lots.append({"qty": rem, "cost_eur": cost_eur, "cost_local": cost_local})
            continue

        if t == "optionbuy":
            ensure(key, row)
            lots = positions[key]["lots"]
            rem = qty
            while rem > MIN_POSITION and lots and lots[0]["qty"] < -MIN_POSITION:
                lot = lots[0]
                abs_s = abs(float(lot["qty"]))
                take = min(rem, abs_s)
                frac = take / abs_s if abs_s > 1e-12 else 1.0
                lot["cost_eur"] *= 1.0 - frac
                lot["cost_local"] *= 1.0 - frac
                lot["qty"] += take
                rem -= take
                if abs(float(lot["qty"])) < MIN_POSITION:
                    lots.pop(0)
            if rem > MIN_POSITION:
                cost_eur = _to_float(total_eur) * (rem / qty) if qty > 1e-12 else _to_float(total_eur)
                cost_local = (price_local * rem) if price_local else 0.0
                positions[key]["lots"].append({"qty": rem, "cost_eur": cost_eur, "cost_local": cost_local})
            continue

        if t in ("sell", "switch"):
            ensure(key, row)
            lots = positions[key]["lots"]
            remaining = abs(qty)
            sell_total = remaining if remaining > 1e-12 else 1.0
            total_eur_f = _to_float(total_eur)
            price_local_f = _to_float(price_local)
            while remaining > MIN_POSITION and lots:
                lot = lots[0]
                lot_qty = float(lot["qty"])
                if lot_qty < -MIN_POSITION:
                    te_part = total_eur_f * (remaining / sell_total) if sell_total > 1e-12 else total_eur_f
                    lot["qty"] -= remaining
                    lot["cost_eur"] -= te_part
                    lot["cost_local"] -= price_local_f * remaining
                    remaining = 0
                    break
                if lot_qty < MIN_POSITION:
                    lots.pop(0)
                    continue
                if lot_qty <= remaining + MIN_POSITION:
                    remaining -= lot_qty
                    lots.pop(0)
                else:
                    frac = remaining / lot_qty
                    lot["qty"] -= remaining
                    lot["cost_eur"] -= lot["cost_eur"] * frac
                    lot["cost_local"] -= lot["cost_local"] * frac
                    remaining = 0
            if remaining > MIN_POSITION:
                fr = remaining / sell_total if sell_total > 1e-12 else 1.0
                lots.append({
                    "qty": -remaining,
                    "cost_eur": -total_eur_f * fr,
                    "cost_local": -price_local_f * remaining,
                })
            continue

        if t == "optionsell":
            ensure(key, row)
            lots = positions[key]["lots"]
            rem = abs(qty)
            total_long = sum(max(0.0, float(l["qty"])) for l in lots)
            if total_long > MIN_POSITION:
                remaining = rem
                while remaining > MIN_POSITION and lots:
                    lot = lots[0]
                    lot_qty = float(lot["qty"])
                    if lot_qty < MIN_POSITION:
                        lots.pop(0)
                        continue
                    if lot_qty <= remaining + MIN_POSITION:
                        remaining -= lot_qty
                        lots.pop(0)
                    else:
                        frac = remaining / lot_qty
                        lot["qty"] -= remaining
                        lot["cost_eur"] -= lot["cost_eur"] * frac
                        lot["cost_local"] -= lot["cost_local"] * frac
                        remaining = 0
            else:
                te = _to_float(total_eur)
                pl = _to_float(price_local)
                lots.append({"qty": -rem, "cost_eur": -te, "cost_local": -pl * rem})
            continue

    rows = []
    for _fifo_key, p in positions.items():
        total_qty = sum(l["qty"] for l in p["lots"])
        total_cost_eur = sum(l["cost_eur"] for l in p["lots"])
        total_cost_local = sum(l["cost_local"] for l in p["lots"])
        if abs(total_qty) < MIN_POSITION:
            continue
        avg_eur = total_cost_eur / total_qty if total_qty else math.nan
        avg_local = total_cost_local / total_qty if total_qty else math.nan
        ty_disp = p.get("Ticker_Yahoo") or ""
        rows.append({
            "Broker": p.get("Broker") or "",
            "Ticker": p.get("ticker_orig") or ty_disp,
            "Ticker_Yahoo": ty_disp,
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


_ECB_RATE_MEMO: OrderedDict[tuple[str, str], float] = OrderedDict()
_ECB_MEMO_LIMIT = 600


def _ecb_memo_put(ccy: str, ymd: str, rate: float) -> None:
    k = (ccy, ymd)
    _ECB_RATE_MEMO[k] = rate
    _ECB_RATE_MEMO.move_to_end(k)
    while len(_ECB_RATE_MEMO) > _ECB_MEMO_LIMIT:
        _ECB_RATE_MEMO.popitem(last=False)


def _frankfurter_v1_eur_per_unit(ccy: str, ymd: str) -> float:
    ccy = str(ccy).upper().strip()
    k = (ccy, ymd)
    if k in _ECB_RATE_MEMO:
        _ECB_RATE_MEMO.move_to_end(k)
        return float(_ECB_RATE_MEMO[k])
    url = f"https://api.frankfurter.dev/v1/{ymd}?base={ccy}&symbols=EUR"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "FiliosPortfolio/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        rates = data.get("rates") or {}
        if "EUR" not in rates:
            return math.nan
        r = float(rates["EUR"])
        if r > 0:
            _ecb_memo_put(ccy, ymd, r)
            return r
    except urllib.error.HTTPError:
        return math.nan
    except Exception:
        return math.nan
    return math.nan


def get_fx_rate_ecb(currency: str, as_of_date) -> float:
    """
    Tipo de cambio BCE (Frankfurter v1, datos de referencia diarios) hacia EUR.
    Devuelve cuántos EUR por 1 unidad de moneda. Ej: USD -> 0.92 = 1 USD = 0.92 EUR.
    Si la fecha es festivo/fin de semana sin publicación, retrocede hasta 15 días.
    """
    if not currency or str(currency).upper() == "EUR":
        return 1.0
    ccy = str(currency).upper().strip()
    ts = pd.Timestamp(as_of_date)
    if pd.isna(ts):
        return math.nan
    for back in range(0, 15):
        d = (ts - pd.Timedelta(days=back)).strftime("%Y-%m-%d")
        r = _frankfurter_v1_eur_per_unit(ccy, d)
        if not math.isnan(r) and r > 0:
            return r
    return math.nan


def get_fx_rate_for_date(currency: str, as_of_date) -> float:
    """
    Tipo de cambio de cierre para una fecha: cuántos EUR por 1 unidad de moneda.
    Usa BCE (Frankfurter v1) primero; solo si no hay dato válido, Yahoo Finance.
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


def _valuation_business_day_month_cap_today(year: int, month: int) -> pd.Timestamp | None:
    """
    Último día hábil del mes, sin ser posterior a hoy.
    None si el mes aún no ha comenzado (respecto a hoy).
    """
    month_start = pd.Timestamp(year=year, month=month, day=1)
    today = pd.Timestamp.now().normalize()
    if month_start > today:
        return None
    month_end_cal = month_start + pd.offsets.MonthEnd(0)
    end_cap = min(month_end_cal.normalize(), today)
    bdays = pd.bdate_range(start=month_start, end=end_cap, freq="C")
    if len(bdays) == 0:
        return None
    return pd.Timestamp(bdays[-1].date())


def _yahoo_close_and_currency_on_or_before(ticker: str, as_of: pd.Timestamp) -> tuple[float, str | None]:
    """Cierre Yahoo en divisa de cotización en o antes de as_of (día)."""
    if not ticker or not str(ticker).strip():
        return math.nan, None
    t = str(ticker).strip()
    try:
        tk = yf.Ticker(t)
        as_of_naive = pd.Timestamp(as_of)
        if as_of_naive.tzinfo is not None:
            as_of_naive = as_of_naive.tz_localize(None)

        start = as_of_naive - pd.Timedelta(days=14)
        end = as_of_naive + pd.Timedelta(days=3)
        hist = tk.history(start=start, end=end, auto_adjust=True)
        if hist.empty:
            hist = tk.history(period="2y", auto_adjust=True)
        if hist.empty:
            return math.nan, None

        idx = pd.to_datetime(hist.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        hist = hist.copy()
        hist.index = idx
        mask = hist.index.normalize() <= as_of_naive.normalize()
        sub = hist.loc[mask]
        if sub.empty:
            return math.nan, None
        close = float(sub["Close"].iloc[-1])
        info = getattr(tk, "fast_info", {}) or {}
        base = getattr(tk, "info", {}) or {}
        ccy = info.get("currency") or base.get("currency") or "USD"
        return close, str(ccy).upper() if ccy else "USD"
    except Exception:
        return math.nan, None


def _precio_eur_historico(ticker_yahoo: str, as_of: pd.Timestamp, manual_eur: dict[str, float]) -> float:
    """Precio en EUR al cierre de as_of (manual JSON tiene prioridad: ya en EUR)."""
    ty = str(ticker_yahoo or "").strip()
    if ty and ty in manual_eur:
        return float(manual_eur[ty])
    ysym = _yahoo_symbol_for_history_lookup(ty)
    close, ccy = _yahoo_close_and_currency_on_or_before(ysym, as_of)
    if math.isnan(close):
        return math.nan
    if not ccy or ccy == "EUR":
        return close
    fx = get_fx_rate_for_date(ccy, as_of)
    if math.isnan(fx) or fx <= 0:
        return math.nan
    return close * fx


def _positions_union_snapshot(
    pos_acc: pd.DataFrame, pos_fon: pd.DataFrame, pos_crip: pd.DataFrame
) -> pd.DataFrame:
    """Une acciones/ETF, fondos y cripto con columnas mínimas para valorar."""
    parts: list[pd.DataFrame] = []
    if pos_acc is not None and not pos_acc.empty:
        p = pos_acc.copy()
        p["Origen"] = "Acciones"
        parts.append(p)
    if pos_fon is not None and not pos_fon.empty:
        p = pos_fon.copy()
        p["Origen"] = "Fondos"
        parts.append(p)
    if pos_crip is not None and not pos_crip.empty:
        p = pos_crip.copy()
        if "Titulos" not in p.columns and "Cantidad" in p.columns:
            p = p.rename(columns={"Cantidad": "Titulos"})
        p["Origen"] = "Criptos"
        if "Moneda Activo" not in p.columns:
            p["Moneda Activo"] = "EUR"
        parts.append(p)
    if not parts:
        return pd.DataFrame(
            columns=["Broker", "Ticker_Yahoo", "Titulos", "Origen", "Moneda Activo", "Inversion €"]
        )
    return pd.concat(parts, ignore_index=True)


def _snapshot_excluir_otros_warrant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quitar del cálculo de Análisis mensual los activos «Otros» (positionType warrant):
    derivados sin precio Yahoo fiable; evita avisos y coste/mercado incoherentes si no hay manual.
    """
    if df.empty or "Tipo activo" not in df.columns:
        return df
    tipos = df["Tipo activo"].astype(str).str.strip().str.lower()
    return df.loc[tipos != "warrant"].reset_index(drop=True)


def _inversion_abierta_snapshot_eur(positions: pd.DataFrame) -> float:
    """Suma coste en € (FIFO / fondos / cripto) solo de líneas con cantidad abierta > 0."""
    if positions.empty or "Inversion €" not in positions.columns:
        return 0.0
    total = 0.0
    for _, row in positions.iterrows():
        qty = float(row.get("Titulos") or 0.0)
        if abs(qty) < MIN_POSITION:
            continue
        inv = row.get("Inversion €")
        if inv is None or (isinstance(inv, float) and pd.isna(inv)):
            continue
        total += float(inv)
    return float(total)


def _valor_mercado_historico_total_eur(
    positions: pd.DataFrame, as_of: pd.Timestamp, manual_eur: dict[str, float]
) -> tuple[float, int, int, list[str]]:
    """
    Suma valor de mercado en EUR.
    Devuelve (total_mercado, líneas_con_cantidad>0, líneas_sin_precio, tickers_Yahoo_sin_precio únicos).
    Las líneas sin precio (NaN) no suman al total pero siguen contando en coste de cartera.
    """
    if positions.empty:
        return 0.0, 0, 0, []
    tcol = "Ticker_Yahoo" if "Ticker_Yahoo" in positions.columns else "Ticker"
    if tcol not in positions.columns or "Titulos" not in positions.columns:
        return 0.0, 0, 0, []
    as_of_d = pd.Timestamp(as_of).normalize()
    total = 0.0
    n_lines = 0
    n_sin_precio = 0
    sin_tickers: list[str] = []
    seen_sin: set[str] = set()
    for _, row in positions.iterrows():
        qty = float(row.get("Titulos") or 0.0)
        if abs(qty) < MIN_POSITION:
            continue
        n_lines += 1
        ty = str(row.get(tcol) or "").strip()
        pe = _precio_eur_historico(ty, as_of_d, manual_eur)
        if math.isnan(pe):
            n_sin_precio += 1
            if ty and ty not in seen_sin:
                seen_sin.add(ty)
                sin_tickers.append(ty)
        else:
            total += qty * pe
    return float(total), n_lines, n_sin_precio, sin_tickers


def compute_valor_mercado_snapshot_mes(
    anio: int,
    mes: int,
    df_acc: pd.DataFrame,
    df_fon: pd.DataFrame,
    df_crip: pd.DataFrame,
    manual_eur: dict[str, float],
) -> dict | None:
    """
    Valor de mercado consolidado al último día hábil del mes (cap a hoy).
    None si el mes es futuro.
    """
    val_day = _valuation_business_day_month_cap_today(anio, mes)
    if val_day is None:
        return None
    cutoff = pd.Timestamp(val_day.year, val_day.month, val_day.day, 23, 59, 59)

    def _cut(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if "datetime_full" not in df.columns:
            return pd.DataFrame()
        return df[df["datetime_full"].notna() & (df["datetime_full"] <= cutoff)].copy()

    fa, ff, fc = _cut(df_acc), _cut(df_fon), _cut(df_crip)
    pos_acc = compute_positions_fifo(fa)
    pos_fon = (
        positions_fondos_to_dataframe(compute_positions_fondos(ff))
        if not ff.empty
        else pd.DataFrame()
    )
    pos_crip = (
        compute_positions_criptos(fc, use_kraken_ledger_override=False)
        if not fc.empty
        else pd.DataFrame()
    )
    comb = _positions_union_snapshot(pos_acc, pos_fon, pos_crip)
    comb = _snapshot_excluir_otros_warrant(comb)
    if comb.empty:
        return {
            "fecha_valoracion": val_day.strftime("%Y-%m-%d"),
            "valor_mercado_eur": 0.0,
            "valor_invertido_eur": 0.0,
            "num_lineas": 0,
            "num_lineas_sin_precio": 0,
            "tickers_sin_precio": [],
        }
    vm, nln, n_sin_px, tickers_sin_px = _valor_mercado_historico_total_eur(
        comb, val_day.normalize(), manual_eur
    )
    inv_eur = _inversion_abierta_snapshot_eur(comb)
    return {
        "fecha_valoracion": val_day.strftime("%Y-%m-%d"),
        "valor_mercado_eur": vm,
        "valor_invertido_eur": inv_eur,
        "num_lineas": int(nln),
        "num_lineas_sin_precio": int(n_sin_px),
        "tickers_sin_precio": tickers_sin_px,
    }


def load_cartera_snapshots_mes() -> pd.DataFrame:
    _init_db_cartera_snapshot_mes()
    with _get_db() as conn:
        try:
            df = pd.read_sql(
                "SELECT anio, mes, fecha_valoracion, valor_mercado_eur, valor_invertido_eur, "
                "num_lineas, computed_at FROM cartera_snapshot_mes ORDER BY anio, mes",
                conn,
            )
        except Exception:
            return pd.DataFrame(
                columns=[
                    "anio",
                    "mes",
                    "fecha_valoracion",
                    "valor_mercado_eur",
                    "valor_invertido_eur",
                    "num_lineas",
                    "computed_at",
                ]
            )
    if "valor_invertido_eur" not in df.columns:
        df["valor_invertido_eur"] = 0.0
    df["valor_invertido_eur"] = pd.to_numeric(df["valor_invertido_eur"], errors="coerce").fillna(0.0)
    return df


def save_cartera_snapshot_mes(
    anio: int,
    mes: int,
    fecha_valoracion: str,
    valor_mercado_eur: float,
    valor_invertido_eur: float,
    num_lineas: int,
) -> None:
    _init_db_cartera_snapshot_mes()
    now_iso = datetime.now().isoformat(timespec="seconds")
    with _get_db() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO cartera_snapshot_mes
            (anio, mes, fecha_valoracion, valor_mercado_eur, valor_invertido_eur, num_lineas, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (anio, mes, fecha_valoracion, valor_mercado_eur, valor_invertido_eur, num_lineas, now_iso),
        )
        conn.commit()


def refresh_cartera_snapshots_mensual_desde(anio_ini: int, mes_ini: int) -> tuple[int, str]:
    """
    Recalcula y guarda todos los meses desde (anio_ini, mes_ini) hasta el mes en curso.
    Devuelve (n_meses_escritos, mensaje).
    """
    manual = load_precios_manuales()
    df_acc = load_data()
    df_fon = load_data_fondos()
    df_crip = load_data_criptos()
    now = datetime.now()
    y, m = anio_ini, mes_ini
    n = 0
    errs: list[str] = []
    while y < now.year or (y == now.year and m <= now.month):
        try:
            snap = compute_valor_mercado_snapshot_mes(y, m, df_acc, df_fon, df_crip, manual)
            if snap is None:
                break
            save_cartera_snapshot_mes(
                y,
                m,
                snap["fecha_valoracion"],
                snap["valor_mercado_eur"],
                snap["valor_invertido_eur"],
                snap["num_lineas"],
            )
            n += 1
        except Exception as ex:
            errs.append(f"{y}-{m:02d}: {ex}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return n, "; ".join(errs) if errs else "OK"


def refresh_cartera_snapshot_un_solo_mes(anio: int, mes: int) -> tuple[bool, str]:
    """
    Calcula y guarda un único mes. False si el mes es futuro o hay error.
    """
    now = datetime.now()
    if mes < 1 or mes > 12:
        return False, "Mes inválido."
    if anio > now.year or (anio == now.year and mes > now.month):
        return False, "Ese mes es futuro; aún no tiene cierre disponible."
    try:
        manual = load_precios_manuales()
        df_acc = load_data()
        df_fon = load_data_fondos()
        df_crip = load_data_criptos()
        snap = compute_valor_mercado_snapshot_mes(anio, mes, df_acc, df_fon, df_crip, manual)
        if snap is None:
            return False, "No se pudo calcular (sin días hábiles en el mes o fecha fuera de rango)."
        save_cartera_snapshot_mes(
            anio,
            mes,
            snap["fecha_valoracion"],
            snap["valor_mercado_eur"],
            snap["valor_invertido_eur"],
            snap["num_lineas"],
        )
        msg = (
            f"Guardado {anio}-{mes:02d} (valoración {snap['fecha_valoracion']}). "
            f"Mercado {snap['valor_mercado_eur']:,.2f} € · Coste posiciones {snap['valor_invertido_eur']:,.2f} €."
        )
        nsin = int(snap.get("num_lineas_sin_precio") or 0)
        if nsin > 0:
            muestra = snap.get("tickers_sin_precio") or []
            tlist = ", ".join(str(t) for t in muestra[:12])
            if len(muestra) > 12:
                tlist += "…"
            msg += (
                f" Advertencia: {nsin} línea(s) sin cotización Yahoo/manual a esa fecha "
                f"(no entran en mercado; el total puede quedar bajo). Tickers: {tlist}"
            )
        return (True, msg)
    except Exception as ex:
        return False, str(ex)


def _sugerir_siguiente_mes_snapshot(snaps: pd.DataFrame) -> tuple[int, int]:
    """Primer mes desde 2025-01 hasta el mes actual que no esté en la tabla; si todos, el mes actual."""
    now = datetime.now()
    existing: set[tuple[int, int]] = set()
    if not snaps.empty and "anio" in snaps.columns and "mes" in snaps.columns:
        for _, r in snaps.iterrows():
            try:
                existing.add((int(r["anio"]), int(r["mes"])))
            except (ValueError, TypeError):
                continue
    y, m = 2025, 1
    while (y < now.year) or (y == now.year and m <= now.month):
        if (y, m) not in existing:
            return y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return now.year, now.month


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


def positions_base_cartera_unificada(df_acciones: pd.DataFrame) -> pd.DataFrame:
    """Acciones FIFO + fondos + criptos (misma base que la vista Cartera)."""
    positions_acc = compute_positions_fifo(df_acciones)
    positions_acc["Origen"] = "Acciones"
    df_fondos = load_data_fondos()
    positions_fondos_df = positions_fondos_to_dataframe(compute_positions_fondos(df_fondos))
    positions_fondos_df["Origen"] = "Fondos"
    df_crip = load_data_criptos()
    positions_crip_df = compute_positions_criptos(df_crip)
    if not positions_crip_df.empty:
        positions_crip_df = positions_crip_df.rename(
            columns={
                "Cantidad": "Titulos",
                "Inversion \uFFFD": "Inversion €",
                "Inversion ?": "Inversion €",
            }
        )
        if "Inversion €" not in positions_crip_df.columns and "Inversion \uFFFD" in positions_crip_df.columns:
            positions_crip_df["Inversion €"] = positions_crip_df["Inversion \uFFFD"]
        positions_crip_df["Tipo activo"] = "crypto"
        positions_crip_df["Origen"] = "Criptos"
        positions_crip_df["Moneda Activo"] = "EUR"
        positions_crip_df["Moneda Yahoo"] = "EUR"
        return pd.concat([positions_acc, positions_fondos_df, positions_crip_df], ignore_index=True)
    return pd.concat([positions_acc, positions_fondos_df], ignore_index=True)


def _distribucion_filtrar_lineas(pos: pd.DataFrame) -> pd.DataFrame:
    """Excluye warrants y líneas con cantidad ~0 (coherente con Cartera / snapshots)."""
    if pos.empty:
        return pos
    out = _snapshot_excluir_otros_warrant(pos.copy())
    if "Titulos" not in out.columns:
        return out
    q = pd.to_numeric(out["Titulos"], errors="coerce").fillna(0.0)
    return out.loc[q.abs() >= MIN_POSITION].reset_index(drop=True)


DISTRIBUCION_GRUPOS_CLASE = ("Acciones", "ETFs", "Cripto", "Fondos")


def _distribucion_grupo_linea(row: pd.Series) -> str | None:
    """Acciones / ETFs / Cripto / Fondos para la vista Distribución; excluye puts, calls y warrants."""
    o = str(row.get("Origen", "") or "").strip()
    t = str(row.get("Tipo activo", "") or "").strip().lower()
    if o == "Criptos":
        return "Cripto"
    if o == "Fondos":
        return "Fondos"
    if o == "Acciones":
        if t in ("putoption", "calloption", "warrant"):
            return None
        if t == "etf":
            return "ETFs"
        return "Acciones"
    return None


def _distribucion_cripto_ticker_agg_key(row: pd.Series) -> str:
    """Clave por moneda (sin broker) para agrupar cripto en Distribución."""
    ty = str(row.get("Ticker_Yahoo") or "").strip().upper()
    if ty.endswith("-EUR"):
        ty = ty[:-4]
    if not ty:
        ty = str(row.get("Ticker") or "").strip().upper()
        if ty.endswith("-EUR"):
            ty = ty[:-4]
    return ty or "__"


def _distribucion_agregar_cripto_por_activo(enr: pd.DataFrame) -> pd.DataFrame:
    """Una fila por cripto en «Cartera por posición»: suma cuentas con el mismo activo."""
    if enr.empty or "_Grupo" not in enr.columns:
        return enr
    mask = enr["_Grupo"].astype(str).eq("Cripto")
    if not mask.any():
        return enr
    left = enr.loc[~mask].copy()
    c = enr.loc[mask].copy()
    c["_ck"] = c.apply(_distribucion_cripto_ticker_agg_key, axis=1)
    pieces: list[pd.Series] = []
    for _, sub in c.groupby("_ck", sort=False):
        sub = sub.reset_index(drop=True)
        vm = pd.to_numeric(sub["Valor Mercado €"], errors="coerce").fillna(0.0)
        idx_max = int(vm.idxmax()) if len(vm) else 0
        row = sub.loc[idx_max].copy()
        row["Valor Mercado €"] = float(vm.sum())
        row["Inversion €"] = float(pd.to_numeric(sub["Inversion €"], errors="coerce").fillna(0.0).sum())
        if "Div_EUR" in sub.columns:
            row["Div_EUR"] = float(pd.to_numeric(sub["Div_EUR"], errors="coerce").fillna(0.0).sum())
        if "Titulos" in sub.columns:
            row["Titulos"] = float(pd.to_numeric(sub["Titulos"], errors="coerce").fillna(0.0).sum())
        if "Plusvalia €" in row.index:
            row["Plusvalia €"] = float(row["Valor Mercado €"]) - float(row["Inversion €"] or 0)
        inv = float(row.get("Inversion €") or 0)
        if "Plusvalia %" in row.index and abs(inv) > 1e-12:
            row["Plusvalia %"] = (float(row["Plusvalia €"]) / inv) * 100.0
        pieces.append(row)
    merged_c = pd.DataFrame(pieces).reset_index(drop=True)
    merged_c = merged_c.drop(columns=["_ck"], errors="ignore")
    return pd.concat([left, merged_c], ignore_index=True)


def _distribucion_agregar_acciones_etf_fondos_mismo_broker_ticker(enr: pd.DataFrame) -> pd.DataFrame:
    """
    `compute_positions_fifo` / fondos pueden generar varias filas con el mismo (Broker, Ticker_Yahoo)
    (p. ej. sacos distintos ISIN×divisa que cotizan con el mismo Yahoo). En Distribución se suman
    inversión, valor mercado y dividendos para coincidir con el total real de la posición.
    """
    if enr.empty or "_Grupo" not in enr.columns:
        return enr
    mask = enr["_Grupo"].isin(["Acciones", "ETFs", "Fondos"])
    if not mask.any():
        return enr
    left = enr.loc[~mask].copy()
    sub = enr.loc[mask].copy()
    sub["_vm"] = pd.to_numeric(sub["Valor Mercado €"], errors="coerce").fillna(0.0)
    sub["_br"] = sub["Broker"].astype(str).str.strip()
    sub["_ty"] = sub["Ticker_Yahoo"].astype(str).str.strip()
    pieces: list[pd.Series] = []
    for _, part in sub.groupby(["_br", "_ty", "_Grupo"], sort=False):
        part = part.reset_index(drop=True)
        vm = part["_vm"]
        idx_max = int(vm.idxmax()) if len(vm) else 0
        row = part.loc[idx_max].copy()
        row["Valor Mercado €"] = float(pd.to_numeric(part["Valor Mercado €"], errors="coerce").fillna(0.0).sum())
        row["Inversion €"] = float(pd.to_numeric(part["Inversion €"], errors="coerce").fillna(0.0).sum())
        if "Div_EUR" in part.columns:
            row["Div_EUR"] = float(pd.to_numeric(part["Div_EUR"], errors="coerce").fillna(0.0).sum())
        if "Titulos" in part.columns:
            row["Titulos"] = float(pd.to_numeric(part["Titulos"], errors="coerce").fillna(0.0).sum())
        if "Plusvalia €" in row.index:
            row["Plusvalia €"] = float(row["Valor Mercado €"]) - float(row["Inversion €"] or 0)
        inv = float(row.get("Inversion €") or 0)
        if "Plusvalia %" in row.index and abs(inv) > 1e-12:
            row["Plusvalia %"] = (float(row["Plusvalia €"]) / inv) * 100.0
        row = row.drop(labels=[c for c in ("_vm", "_br", "_ty") if c in row.index])
        pieces.append(row)
    merged_a = pd.DataFrame(pieces).reset_index(drop=True)
    return pd.concat([left, merged_a], ignore_index=True)


_NOMBRE_CLAVE_SUFFIX_RE = re.compile(
    r"\s*,?\s*(inc\.?|incorporated|corp\.?|corporation|ltd\.?|limited|plc|s\.?a\.?|n\.?v\.?)\s*$",
    re.IGNORECASE,
)


def _distribucion_nombre_clave_posicion(row: pd.Series) -> str:
    """Clave estable para agrupar la misma empresa en Distribución (nombre o ticker)."""
    n = str(row.get("Nombre") or "").strip()
    if n:
        s = " ".join(n.split()).casefold()
        s = _NOMBRE_CLAVE_SUFFIX_RE.sub("", s).strip()
        return s if s else "__"
    ty = str(row.get("Ticker_Yahoo") or row.get("Ticker") or "").strip()
    return ty.casefold() if ty else "__"


def _distribucion_fila_clave_agrupacion(row: pd.Series, cat_cache: dict[str, str]) -> str:
    """
    Prioridad: ISIN en la fila → ISIN en instrument_catalog (ticker Yahoo / ticker) → nombre normalizado.
    Prefijos evitan colisión ISIN vs texto accidental.
    """
    for c in ("isin", "ISIN"):
        ni = _norm_isin_field(row.get(c))
        if ni:
            return f"ISIN:{ni}"
    ty = str(row.get("Ticker_Yahoo") or "").strip()
    to = str(row.get("Ticker") or "").strip()
    for cand in (ty, to):
        if not cand:
            continue
        if cand not in cat_cache:
            cat_cache[cand] = _lookup_isin_for_ticker_yahoo(cand) or ""
        ni = _norm_isin_field(cat_cache[cand])
        if ni:
            return f"ISIN:{ni}"
    return f"NOM:{_distribucion_nombre_clave_posicion(row)}"


def _distribucion_agregar_acciones_etf_fondos_mismo_isin_o_nombre(enr: pd.DataFrame) -> pd.DataFrame:
    """
    Colapsa filas por ISIN (movimiento o catálogo) y clase; si no hay ISIN, por nombre normalizado
    (misma lógica que antes). Así se unen brokers/listings del mismo valor negociable.
    """
    if enr.empty or "_Grupo" not in enr.columns:
        return enr
    mask = enr["_Grupo"].isin(["Acciones", "ETFs", "Fondos"])
    if not mask.any():
        return enr
    left = enr.loc[~mask].copy()
    sub = enr.loc[mask].copy()
    cat_cache: dict[str, str] = {}
    sub["_gk"] = sub.apply(lambda r: _distribucion_fila_clave_agrupacion(r, cat_cache), axis=1)
    sub["_vm"] = pd.to_numeric(sub["Valor Mercado €"], errors="coerce").fillna(0.0)
    pieces: list[pd.Series] = []
    for (_, _), part in sub.groupby(["_gk", "_Grupo"], sort=False):
        part = part.reset_index(drop=True)
        vm = part["_vm"]
        idx_max = int(vm.idxmax()) if len(vm) else 0
        row = part.loc[idx_max].copy()
        row["Valor Mercado €"] = float(vm.sum())
        row["Inversion €"] = float(pd.to_numeric(part["Inversion €"], errors="coerce").fillna(0.0).sum())
        if "Div_EUR" in part.columns:
            row["Div_EUR"] = float(pd.to_numeric(part["Div_EUR"], errors="coerce").fillna(0.0).sum())
        if "Titulos" in part.columns:
            row["Titulos"] = float(pd.to_numeric(part["Titulos"], errors="coerce").fillna(0.0).sum())
        if "Plusvalia €" in row.index:
            row["Plusvalia €"] = float(row["Valor Mercado €"]) - float(row["Inversion €"] or 0)
        inv = float(row.get("Inversion €") or 0)
        if "Plusvalia %" in row.index and abs(inv) > 1e-12:
            row["Plusvalia %"] = (float(row["Plusvalia €"]) / inv) * 100.0
        row = row.drop(labels=[c for c in ("_vm", "_gk") if c in row.index])
        pieces.append(row)
    merged_n = pd.DataFrame(pieces).reset_index(drop=True)
    return pd.concat([left, merged_n], ignore_index=True)


def _distribucion_shell_sin_mercado(base_df: pd.DataFrame, precios_m: dict[str, float]) -> pd.DataFrame:
    """Filas listas para la vista Distribución sin descargar Yahoo; solo precios manuales si los hay."""
    positions = base_df.copy()
    ticker_col = "Ticker_Yahoo" if "Ticker_Yahoo" in positions.columns else "Ticker"
    positions["Precio Actual"] = math.nan
    positions["Precio Actual €"] = math.nan
    positions["Valor Mercado €"] = math.nan
    positions["Plusvalia €"] = math.nan
    positions["Plusvalia %"] = math.nan
    positions["GyP hoy %"] = math.nan
    positions["GyP hoy €"] = math.nan
    positions["Cierre Previo"] = math.nan
    if "Moneda Activo" in positions.columns:
        positions["Moneda Yahoo"] = positions["Moneda Activo"]
    else:
        positions["Moneda Yahoo"] = "EUR"
    if not precios_m:
        return positions
    for idx, row in positions.iterrows():
        t = str(row.get(ticker_col) or "").strip()
        if not t or t not in precios_m:
            continue
        px = precios_m[t]
        positions.at[idx, "Precio Actual"] = px
        positions.at[idx, "Precio Actual €"] = px
        tit = float(positions.at[idx, "Titulos"] or 0)
        inv = float(positions.at[idx, "Inversion €"] or 0)
        positions.at[idx, "Valor Mercado €"] = tit * px
        positions.at[idx, "Plusvalia €"] = positions.at[idx, "Valor Mercado €"] - inv
        positions.at[idx, "Plusvalia %"] = (
            (positions.at[idx, "Plusvalia €"] / inv * 100.0) if abs(inv) > 1e-12 else math.nan
        )
    return positions


def _distribucion_reaplicar_precios_manuales(enr: pd.DataFrame, precios_m: dict[str, float]) -> None:
    """Actualiza filas del DataFrame enriquecido con precios manuales (in-place)."""
    if not precios_m:
        return
    ticker_col = "Ticker_Yahoo" if "Ticker_Yahoo" in enr.columns else "Ticker"
    for idx, row in enr.iterrows():
        t = str(row.get(ticker_col) or "").strip()
        if not t or t not in precios_m:
            continue
        if not (pd.isna(row.get("Precio Actual")) or row.get("Precio Actual") is None):
            continue
        px = precios_m[t]
        enr.at[idx, "Precio Actual"] = px
        if "Precio Actual €" in enr.columns:
            enr.at[idx, "Precio Actual €"] = px
        tit = float(enr.at[idx, "Titulos"] or 0)
        inv = float(enr.at[idx, "Inversion €"] or 0)
        enr.at[idx, "Valor Mercado €"] = tit * px
        enr.at[idx, "Plusvalia €"] = enr.at[idx, "Valor Mercado €"] - inv
        enr.at[idx, "Plusvalia %"] = (
            (enr.at[idx, "Plusvalia €"] / inv * 100.0) if abs(inv) > 1e-12 else math.nan
        )


_MERGE_MKT_FROM_CARTERA = (
    "Precio Actual",
    "Cierre Previo",
    "Moneda Yahoo",
    "Precio Actual €",
    "Valor Mercado €",
    "Plusvalia €",
    "Plusvalia %",
    "GyP hoy %",
    "GyP hoy €",
)


def _distribucion_merge_mercado_desde_cartera(base: pd.DataFrame, car: pd.DataFrame) -> pd.DataFrame:
    """Añade columnas de mercado desde el DataFrame enriquecido de Cartera (mismas filas cartera completa)."""
    keys = ["Broker", "Ticker_Yahoo"]
    m_cols = [c for c in _MERGE_MKT_FROM_CARTERA if c in car.columns]
    sub = car[keys + m_cols].drop_duplicates(subset=keys, keep="first")
    return base.merge(sub, on=keys, how="left")


def _ensure_cartera_enriched_session(df_acciones: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Igual que la vista Cartera: base unificada, firma y carga en sesión desde disco si aplica."""
    positions_full = positions_base_cartera_unificada(df_acciones)
    if "cartera_enriched" not in st.session_state:
        st.session_state["cartera_enriched"] = None
    if "cartera_enriched_updated_at" not in st.session_state:
        st.session_state["cartera_enriched_updated_at"] = None
    if "Tipo activo" in positions_full.columns:
        tipos_all = positions_full["Tipo activo"].astype(str).str.strip().str.lower()
        if (tipos_all == "crypto").any():
            st.session_state["cartera_enriched"] = None
            st.session_state["cartera_enriched_updated_at"] = None
    sig_full = _cotizaciones_signature(positions_full)
    if st.session_state["cartera_enriched"] is None and sig_full:
        cached_df, cached_at = load_cotizaciones_cache(sig_full)
        if cached_df is not None and len(cached_df) == len(positions_full):
            st.session_state["cartera_enriched"] = cached_df
            st.session_state["cartera_enriched_updated_at"] = cached_at
    return positions_full, sig_full


def _cartera_enriched_alineado_con_base(
    car: pd.DataFrame | None, positions_full: pd.DataFrame, sig_full: str
) -> bool:
    if car is None or positions_full.empty or not sig_full:
        return False
    if len(car) != len(positions_full):
        return False
    return _cotizaciones_signature(car) == sig_full


def _distribucion_dividendos_eur_por_linea(div_df: pd.DataFrame) -> pd.DataFrame:
    """Suma totalBaseCurrency (€) por (broker, ticker_Yahoo)."""
    if div_df is None or div_df.empty:
        return pd.DataFrame(columns=["Broker", "Ticker_Yahoo", "Div_EUR"])
    d = div_df.copy()
    if "broker" not in d.columns:
        return pd.DataFrame(columns=["Broker", "Ticker_Yahoo", "Div_EUR"])
    d["Broker"] = d["broker"].astype(str).str.strip()
    ty = d["ticker_Yahoo"] if "ticker_Yahoo" in d.columns else d.get("ticker", "")
    d["Ticker_Yahoo"] = ty.astype(str).str.strip()
    d["Ticker_Yahoo"] = d["Ticker_Yahoo"].replace("", pd.NA).fillna(d.get("ticker", pd.Series("", index=d.index)).astype(str).str.strip())
    col_eur = "totalBaseCurrency" if "totalBaseCurrency" in d.columns else None
    if not col_eur:
        return pd.DataFrame(columns=["Broker", "Ticker_Yahoo", "Div_EUR"])
    d["_eur"] = pd.to_numeric(d[col_eur], errors="coerce").fillna(0.0)
    g = d.groupby(["Broker", "Ticker_Yahoo"], as_index=False)["_eur"].sum()
    return g.rename(columns={"_eur": "Div_EUR"})


def _distribucion_donut_plot(labels: list[str], values: list[float], title: str) -> None:
    import plotly.graph_objects as go

    pairs = [(str(l), float(v)) for l, v in zip(labels, values) if v and float(v) > 1e-9]
    if not pairs:
        st.info("No hay importes positivos para mostrar en el gráfico.")
        return
    lab, val = zip(*pairs)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(lab),
                values=list(val),
                hole=0.45,
                textinfo="percent",
                sort=True,
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        margin=dict(t=50, b=30, l=30, r=30),
        showlegend=True,
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=-0.35),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_analisis_evolucion_mensual() -> None:
    st.subheader("Evolución mensual")
    st.caption(
        "**Valor mercado**: suma de (títulos × precio) en € al cierre; cada precio viene de Yahoo (o precio manual) "
        "más tipo de cambio BCE si la divisa no es EUR — **si no hay precio, esa línea no suma al mercado pero sí al coste**. "
        "**Coste posiciones**: coste remanente en € (FIFO / fondos / cripto) de lo que sigue abierto a esa fecha, "
        "no el total histórico ingresado en la cuenta. **No incluye activos «Otros» (warrants/derivados)** en este gráfico. "
        "Sin efectivo ni dividendos. Último día hábil del mes (tope hoy en el mes actual). "
        "Datos en `cartera_snapshot_mes`; **solo se actualizan al pulsar recalcular**."
    )
    fb = st.session_state.get("analisis_feedback")
    if fb:
        kind, body = fb
        if kind == "warning":
            st.warning(body)
        elif kind == "error":
            st.error(body)
        else:
            st.success(body)
        if st.button("Ocultar este mensaje", key="analisis_feedback_dismiss"):
            del st.session_state["analisis_feedback"]
            st.rerun()
    snaps = load_cartera_snapshots_mes()
    now = datetime.now()
    anos_opts = list(range(2025, now.year + 1))
    meses_es = [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
    ]
    sy, sm = _sugerir_siguiente_mes_snapshot(snaps)
    if "_analisis_snap_next_anio" in st.session_state:
        st.session_state["analisis_snap_anio"] = st.session_state.pop(
            "_analisis_snap_next_anio"
        )
        st.session_state["analisis_snap_mes"] = st.session_state.pop(
            "_analisis_snap_next_mes"
        )
    if "analisis_snap_anio" not in st.session_state:
        st.session_state["analisis_snap_anio"] = sy
    if "analisis_snap_mes" not in st.session_state:
        st.session_state["analisis_snap_mes"] = sm
    if st.session_state["analisis_snap_anio"] not in anos_opts:
        st.session_state["analisis_snap_anio"] = anos_opts[-1]
    st.subheader("Actualizar")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        sel_anio = st.selectbox("Año", anos_opts, key="analisis_snap_anio")
    with c2:
        sel_mes = st.selectbox(
            "Mes",
            list(range(1, 13)),
            format_func=lambda x: f"{x:02d} · {meses_es[x - 1]}",
            key="analisis_snap_mes",
        )
    with c3:
        st.write("")
        st.write("")
        if st.button("Recalcular este mes", type="primary", key="btn_snapshot_un_mes"):
            with st.spinner(f"Cotizaciones para {sel_anio}-{sel_mes:02d}…"):
                ok_m, msg_m = refresh_cartera_snapshot_un_solo_mes(sel_anio, sel_mes)
            if ok_m:
                ny, nm = sel_anio, sel_mes + 1
                if nm > 12:
                    ny, nm = ny + 1, 1
                if ny > now.year or (ny == now.year and nm > now.month):
                    ny, nm = now.year, now.month
                if ny in anos_opts:
                    st.session_state["_analisis_snap_next_anio"] = ny
                    st.session_state["_analisis_snap_next_mes"] = nm
                kind = (
                    "warning"
                    if "Advertencia:" in msg_m or "sin cotización" in msg_m
                    else "success"
                )
                st.session_state["analisis_feedback"] = (kind, msg_m)
            else:
                st.session_state["analisis_feedback"] = ("error", msg_m)
            st.rerun()
    with c4:
        st.write("")
        st.write("")
        if st.button("Recalcular todo (ene 2025 → mes actual)", key="btn_refresh_snapshots_mes"):
            with st.spinner("Descargando cotizaciones históricas… puede tardar mucho."):
                n_meses, detalle = refresh_cartera_snapshots_mensual_desde(2025, 1)
            if detalle == "OK":
                st.session_state["analisis_feedback"] = (
                    "success",
                    f"Actualizados {n_meses} mes(es).",
                )
            else:
                st.session_state["analisis_feedback"] = (
                    "warning",
                    f"Actualizados {n_meses} mes(es). Avisos: {detalle}",
                )
            st.rerun()
    st.caption(
        "Recomendado: **un mes cada vez** (menos tiempo de espera y menos riesgo de límites de Yahoo). "
        "El desplegable sugiere el **primer mes vacío** desde ene. 2025; puedes cambiarlo a mano."
    )
    if snaps.empty:
        st.info("Aún no hay filas guardadas. Calcula al menos un mes arriba para ver tabla y gráfico.")
    else:
        ch = snaps.sort_values(["anio", "mes"]).copy()
        ch["Período"] = ch["anio"].astype(str) + "-" + ch["mes"].astype(str).str.zfill(2)
        st.subheader("Evolución")
        st.line_chart(
            ch.set_index("Período")[
                ["valor_mercado_eur", "valor_invertido_eur"]
            ].rename(
                columns={
                    "valor_mercado_eur": "Valor mercado (€)",
                    "valor_invertido_eur": "Coste posiciones (€)",
                }
            )
        )
        disp = snaps.copy()
        disp["Período"] = disp["anio"].astype(str) + "-" + disp["mes"].astype(str).str.zfill(2)
        disp = disp.sort_values(["anio", "mes"])
        disp = disp.rename(
            columns={
                "fecha_valoracion": "Fecha valoración",
                "valor_mercado_eur": "Valor mercado (€)",
                "valor_invertido_eur": "Coste posiciones (€)",
                "num_lineas": "Líneas posición",
                "computed_at": "Calculado el",
            }
        )
        cols_show = [
            "Período",
            "Fecha valoración",
            "Valor mercado (€)",
            "Coste posiciones (€)",
            "Líneas posición",
            "Calculado el",
        ]
        disp = disp[[c for c in cols_show if c in disp.columns]]
        fmt_cols = {c: fmt_eur for c in disp.columns if "€" in c}
        st.dataframe(
            disp.style.format(fmt_cols, na_rep="–"),
            use_container_width=True,
            hide_index=True,
        )


def render_analisis_gp_ventas() -> None:
    """
    Ventas/permutas con P&L FIFO en EUR: tabla detalle, gráfico mensual y resumen por mes.
    """
    import plotly.graph_objects as go

    st.subheader("G/P realizadas (ventas y permutas)")
    st.caption(
        "Una fila por **venta o salida** (sell / switch / cierre de opción). **Coste FIFO** y **valor transmisión fiscal** "
        "coinciden con **Fiscalidad** (base en EUR menos comisión e impuestos del movimiento). "
        "**Total bruto**, **Comisión+imp.** y **Liquidación neta mov.** vienen del movimiento; si faltan en el CSV, pueden salir vacíos o 0."
    )

    df_acc = load_data()
    df_f = load_data_fondos()
    df_c = load_data_criptos()

    sales_acc = pd.DataFrame()
    if df_acc is not None and not df_acc.empty:
        _, sales_acc, _ = compute_fifo_all(df_acc)
    sales_f = pd.DataFrame()
    if df_f is not None and not df_f.empty:
        _, sales_f, _ = compute_fifo_fondos(df_f)
    sales_c = pd.DataFrame()
    if df_c is not None and not df_c.empty:
        _, sales_c, _ = compute_fifo_criptos(df_c)

    parts: list[pd.DataFrame] = []
    if not sales_acc.empty:
        a = sales_acc.copy()
        a["Origen"] = "Acciones/ETFs"
        parts.append(a)
    if not sales_f.empty:
        b = sales_f.copy()
        b["Origen"] = "Fondos"
        parts.append(b)
    if not sales_c.empty:
        c = sales_c.copy()
        c["Origen"] = "Cripto"
        if "Ticker_Yahoo" not in c.columns:
            c["Ticker_Yahoo"] = ""
        parts.append(c)

    if not parts:
        st.info("No hay ventas registradas (FIFO vacío).")
        return

    sales = pd.concat(parts, ignore_index=True)
    if "Retención dest. (€)" not in sales.columns:
        sales["Retención dest. (€)"] = 0.0
    else:
        sales["Retención dest. (€)"] = pd.to_numeric(sales["Retención dest. (€)"], errors="coerce").fillna(0.0)
    for _col_ex in ("Total bruto (€)", "Comisión+imp. venta (€)", "Liquidación neta mov. (€)"):
        if _col_ex not in sales.columns:
            sales[_col_ex] = np.nan
        else:
            sales[_col_ex] = pd.to_numeric(sales[_col_ex], errors="coerce")

    sales["_dt"] = pd.to_datetime(sales["Fecha venta"].astype(str).str.strip().str[:10], errors="coerce")
    sales = sales[sales["_dt"].notna()].copy()
    if sales.empty:
        st.info("No hay fechas de venta válidas.")
        return

    d_min = sales["_dt"].min()
    d_max = sales["_dt"].max()
    now = datetime.now()

    p_opts = [
        "Últimos 12 meses",
        "Año actual",
        "Año pasado",
        "Este mes",
        "Mes pasado",
        "Siempre",
        "Personalizado",
    ]
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        periodo = st.radio("Período", p_opts, horizontal=True, key="analisis_gp_periodo")
    with c2:
        cols_mode = st.radio(
            "Columnas",
            ["Completa (bruto, liquidación, fiscal)", "Resumida (fiscal)"],
            horizontal=True,
            key="analisis_gp_cols",
        )
    with c3:
        sort_desc = st.checkbox("Ventas más recientes arriba", value=True, key="analisis_gp_sort")

    custom_from = d_min.date() if hasattr(d_min, "date") else pd.Timestamp(d_min).date()
    custom_to = d_max.date() if hasattr(d_max, "date") else pd.Timestamp(d_max).date()
    if periodo == "Personalizado":
        d1, d2 = st.columns(2)
        with d1:
            fd = st.date_input("Desde", value=custom_from, key="analisis_gp_desde")
        with d2:
            fh = st.date_input("Hasta", value=custom_to, key="analisis_gp_hasta")
    else:
        fd = fh = None

    if periodo == "Últimos 12 meses":
        t0 = pd.Timestamp(now) - pd.DateOffset(months=12)
        filt_m = sales["_dt"] >= t0
    elif periodo == "Año actual":
        filt_m = sales["_dt"].dt.year == now.year
    elif periodo == "Año pasado":
        filt_m = sales["_dt"].dt.year == now.year - 1
    elif periodo == "Este mes":
        filt_m = (sales["_dt"].dt.year == now.year) & (sales["_dt"].dt.month == now.month)
    elif periodo == "Mes pasado":
        pm = now - pd.DateOffset(months=1)
        filt_m = (sales["_dt"].dt.year == pm.year) & (sales["_dt"].dt.month == pm.month)
    elif periodo == "Siempre":
        filt_m = pd.Series(True, index=sales.index)
    else:
        filt_m = (sales["_dt"] >= pd.Timestamp(fd)) & (sales["_dt"] <= pd.Timestamp(fh) + pd.Timedelta(days=1))

    d = sales.loc[filt_m].copy()
    if d.empty:
        st.warning("No hay ventas en el período seleccionado.")
        return

    pnl_col = "Plusvalía / Minusvalía (€)"
    d[pnl_col] = pd.to_numeric(d[pnl_col], errors="coerce").fillna(0.0)
    d["Valor compra histórico (€)"] = pd.to_numeric(d["Valor compra histórico (€)"], errors="coerce").fillna(0.0)
    d["Valor venta (€)"] = pd.to_numeric(d["Valor venta (€)"], errors="coerce").fillna(0.0)
    d["Cantidad vendida"] = pd.to_numeric(d["Cantidad vendida"], errors="coerce").fillna(0.0)

    disp = d.copy()
    disp["Fecha"] = disp["_dt"].dt.strftime("%Y-%m-%d")
    base_cols = [
        "Fecha",
        "Origen",
        "Ticker_Yahoo",
        "Ticker",
        "ISIN",
        "Broker",
        "Nombre",
        "Tipo activo",
        "Cantidad vendida",
    ]
    extra_cols = [
        "Total bruto (€)",
        "Liquidación neta mov. (€)",
        "Comisión+imp. venta (€)",
        "Valor venta (€)",
        "Valor compra histórico (€)",
        pnl_col,
        "Retención dest. (€)",
    ]
    if "Resumida" in cols_mode:
        show_cols = base_cols + [
            "Valor venta (€)",
            "Valor compra histórico (€)",
            pnl_col,
            "Retención dest. (€)",
        ]
    else:
        show_cols = base_cols + extra_cols
    show_cols = [c for c in show_cols if c in disp.columns]
    disp_tab = disp[show_cols].copy()
    if sort_desc:
        disp_tab = disp_tab.sort_values("Fecha", ascending=False).reset_index(drop=True)
    else:
        disp_tab = disp_tab.sort_values("Fecha", ascending=True).reset_index(drop=True)

    tot = {
        "Fecha": "TOTAL",
        "Origen": "",
        "Ticker_Yahoo": "",
        "Ticker": "",
        "ISIN": "",
        "Broker": "",
        "Nombre": "",
        "Tipo activo": "",
        "Cantidad vendida": np.nan,
    }
    for c in disp_tab.columns:
        if c in tot:
            continue
        if c in ("Total bruto (€)", "Liquidación neta mov. (€)", "Comisión+imp. venta (€)", "Valor venta (€)", "Valor compra histórico (€)", pnl_col, "Retención dest. (€)"):
            tot[c] = float(pd.to_numeric(disp_tab[c], errors="coerce").fillna(0.0).sum())
        else:
            tot[c] = ""
    disp_show = pd.concat([pd.DataFrame([tot]), disp_tab], ignore_index=True)

    fmt_m = {c: fmt_eur for c in disp_show.columns if "€" in str(c)}
    sty = disp_show.style.format(fmt_m, na_rep="–")
    if pnl_col in disp_show.columns:
        sty = _style_map(sty, color_pnl, subset=[pnl_col])
    st.dataframe(sty, use_container_width=True, hide_index=True)

    d["_ym"] = d["_dt"].dt.to_period("M")
    _agg_d: dict = {
        "Valor venta (€)": "sum",
        "Valor compra histórico (€)": "sum",
        pnl_col: "sum",
        "Retención dest. (€)": "sum",
    }
    if "Liquidación neta mov. (€)" in d.columns:
        _agg_d["Liquidación neta mov. (€)"] = lambda s: float(
            pd.to_numeric(s, errors="coerce").fillna(0.0).sum()
        )
    agg_m = d.groupby("_ym", as_index=False).agg(_agg_d)
    agg_m["Mes"] = agg_m["_ym"].astype(str)
    agg_m = agg_m.drop(columns=["_ym"]).sort_values("Mes").reset_index(drop=True)

    mes_l = [
        "Ene", "Feb", "Mar", "Abr", "May", "Jun",
        "Jul", "Ago", "Sep", "Oct", "Nov", "Dic",
    ]
    agg_m["_lab"] = agg_m["Mes"].map(
        lambda p: f"{mes_l[int(p.split('-')[1]) - 1]} {p.split('-')[0][2:]}" if "-" in str(p) else str(p)
    )

    colors = ["#2e7d32" if v >= 0 else "#c62828" for v in agg_m[pnl_col]]
    fig = go.Figure(
        data=[
            go.Bar(
                x=agg_m["_lab"],
                y=agg_m[pnl_col],
                marker_color=colors,
                name="Plusvalía neta",
            )
        ]
    )
    fig.update_layout(
        title="G/P realizadas por mes",
        xaxis_title="",
        yaxis_title="€",
        showlegend=False,
        margin=dict(l=40, r=20, t=50, b=40),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Resumen por mes")
    agg_show = agg_m.drop(columns=["_lab"], errors="ignore")
    col_order = [
        "Mes",
        "Valor venta (€)",
        "Valor compra histórico (€)",
        pnl_col,
        "Retención dest. (€)",
    ]
    if "Resumida" not in cols_mode and "Liquidación neta mov. (€)" in agg_show.columns:
        col_order.insert(1, "Liquidación neta mov. (€)")
    col_order = [c for c in col_order if c in agg_show.columns]
    sty_m = (
        agg_show[col_order]
        .style.format({c: fmt_eur for c in col_order if "€" in str(c)}, na_rep="–")
    )
    if pnl_col in col_order:
        sty_m = _style_map(sty_m, color_pnl, subset=[pnl_col])
    st.dataframe(sty_m, use_container_width=True, hide_index=True)


def render_analisis_dividendos() -> None:
    """
    Dividendos y cupones por mes/año y por posición (ISIN preferente), alineado con columnas EUR del listado de dividendos.
    """
    import plotly.graph_objects as go

    st.subheader("Dividendos y cupones")
    st.caption(
        "Importes en **EUR** (mismos conceptos que en **Movimientos → Dividendos**). "
        "**Por activo** suma por posición (ISIN o ticker Yahoo / ticker si no hay ISIN); el nombre en tabla/gráfico sale del **dividendo**, y si falta o es solo el ticker se completa con el **Catálogo** (mismos instrumentos que en movimientos, p. ej. `ABEA` → `ABEA.DE`). "
        "Los gráficos por período comparan **total neto cobrado** y **total neto con devolución** (€)."
    )
    div_df = load_dividendos()
    if div_df is None or div_df.empty:
        st.info("No hay dividendos registrados.")
        return
    d = div_df.copy()
    if "date" not in d.columns:
        st.warning("No hay columna de fecha en dividendos.")
        return
    d["_dt"] = pd.to_datetime(d["date"].astype(str).str.strip().str[:10], errors="coerce")
    d = d[d["_dt"].notna()]
    if d.empty:
        st.info("No hay fechas válidas en dividendos.")
        return

    num_cols = [
        "totalBaseCurrency",
        "retentionReturnedBaseCurrency",
        "netoBaseCurrency",
        "destinationRetentionBaseCurrency",
        "totalNetoBaseCurrency",
        "netoWithReturnBaseCurrency",
    ]
    disp_names = {
        "totalBaseCurrency": "Total bruto (€)",
        "retentionReturnedBaseCurrency": "Impuesto satisf. en el extranjero (€)",
        "netoBaseCurrency": "Total bruto después de origen (€)",
        "destinationRetentionBaseCurrency": "Retención en dest. realizada (€)",
        "totalNetoBaseCurrency": "Total neto cobrado (€)",
        "netoWithReturnBaseCurrency": "Total neto con devolución (€)",
    }
    for c in num_cols:
        if c not in d.columns:
            d[c] = 0.0
        else:
            s = d[c].astype(str).str.strip().str.replace(",", ".", regex=False)
            d[c] = pd.to_numeric(s, errors="coerce").fillna(0.0)

    meses_l = [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
    ]
    now = datetime.now()
    d_min = d["_dt"].min()
    d_max = d["_dt"].max()

    p_opts = [
        "Últimos 12 meses",
        "Año actual",
        "Año pasado",
        "Este mes",
        "Mes pasado",
        "Siempre",
        "Personalizado",
    ]
    periodo = st.radio("Período", p_opts, horizontal=True, key="analisis_div_periodo")
    custom_from = d_min.date() if hasattr(d_min, "date") else pd.Timestamp(d_min).date()
    custom_to = d_max.date() if hasattr(d_max, "date") else pd.Timestamp(d_max).date()
    if periodo == "Personalizado":
        c_d1, c_d2 = st.columns(2)
        with c_d1:
            fd = st.date_input("Desde", value=custom_from, key="analisis_div_desde")
        with c_d2:
            fh = st.date_input("Hasta", value=custom_to, key="analisis_div_hasta")
    else:
        fd = fh = None

    if periodo == "Últimos 12 meses":
        t0 = pd.Timestamp(now) - pd.DateOffset(months=12)
        filt = d["_dt"] >= t0
    elif periodo == "Año actual":
        filt = d["_dt"].dt.year == now.year
    elif periodo == "Año pasado":
        filt = d["_dt"].dt.year == now.year - 1
    elif periodo == "Este mes":
        filt = (d["_dt"].dt.year == now.year) & (d["_dt"].dt.month == now.month)
    elif periodo == "Mes pasado":
        pm = now - pd.DateOffset(months=1)
        filt = (d["_dt"].dt.year == pm.year) & (d["_dt"].dt.month == pm.month)
    elif periodo == "Siempre":
        filt = pd.Series(True, index=d.index)
    else:
        t0 = pd.Timestamp(fd)
        t1 = pd.Timestamp(fh) + pd.Timedelta(days=1)
        filt = (d["_dt"] >= t0) & (d["_dt"] < t1)

    d = d.loc[filt].copy()
    if d.empty:
        st.info("No hay dividendos en el período seleccionado.")
        return

    agg_mode = st.radio("Agregación temporal (gráfico y tabla superiores)", ["Mes", "Año"], horizontal=True, key="analisis_div_agg")

    if agg_mode == "Mes":
        d["_per"] = d["_dt"].dt.to_period("M")
        g = d.groupby("_per", as_index=False)[num_cols].sum()
        g = g.sort_values("_per")
        g["Etiqueta"] = g["_per"].apply(lambda p: f"{meses_l[int(p.month) - 1]} {p.year}")
    else:
        d["_per"] = d["_dt"].dt.year
        g = d.groupby("_per", as_index=False)[num_cols].sum()
        g = g.sort_values("_per")
        g["Etiqueta"] = g["_per"].astype(str)

    total_row = {c: float(g[c].sum()) for c in num_cols}
    tbl_m = g[["Etiqueta"] + num_cols].rename(columns=disp_names)
    tot_m = pd.DataFrame([{"Etiqueta": "TOTAL", **{disp_names[c]: total_row[c] for c in num_cols}}])
    tbl_show_m = pd.concat([tot_m, tbl_m], ignore_index=True)

    st.markdown("##### Por período")
    fig_m = go.Figure(
        data=[
            go.Bar(name=disp_names["totalNetoBaseCurrency"], x=g["Etiqueta"], y=g["totalNetoBaseCurrency"], marker_color="#5DADE2"),
            go.Bar(name=disp_names["netoWithReturnBaseCurrency"], x=g["Etiqueta"], y=g["netoWithReturnBaseCurrency"], marker_color="#AAB7B8"),
        ]
    )
    fig_m.update_layout(
        barmode="group",
        title="Dividendos y cupones por período (€)",
        margin=dict(t=50, b=40, l=40, r=20),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="€",
    )
    st.plotly_chart(fig_m, use_container_width=True)
    fmt_m = {c: (lambda x, col=c: fmt_eur(x) if pd.notna(x) else "–") for c in tbl_show_m.columns if "€" in c}
    st.dataframe(tbl_show_m.style.format(fmt_m, na_rep="–"), use_container_width=True, hide_index=True)

    st.markdown("##### Por activo")

    def _div_resolve_isin_norm(r: pd.Series) -> str:
        ni = _norm_isin_field(r.get("isin")) if "isin" in r.index else ""
        if ni:
            return ni
        for key in (r.get("ticker_Yahoo"), r.get("ticker")):
            k = str(key or "").strip()
            if not k:
                continue
            got = _lookup_isin_for_ticker_yahoo(k)
            if got:
                return got
        return ""

    d["_isin_norm"] = d.apply(_div_resolve_isin_norm, axis=1)
    d["_pk"] = d.apply(
        lambda r: (
            str(r["_isin_norm"]).strip()
            if str(r["_isin_norm"] or "").strip()
            else str(r.get("ticker_Yahoo") or r.get("ticker") or "").strip().upper() or "—"
        ),
        axis=1,
    )

    uni = get_universe_instruments_table()
    yahoo_to_name: dict[str, str] = {}
    isin_to_name: dict[str, str] = {}
    ticker_u_to_name: dict[str, str] = {}
    if not uni.empty:
        for _, ur in uni.iterrows():
            y = str(ur.get("ticker_Yahoo") or "").strip()
            nm = str(ur.get("name") or "").strip()
            isnu = _norm_isin_field(ur.get("ISIN"))
            tick = str(ur.get("ticker") or "").strip().upper()
            if y and nm:
                yahoo_to_name[y] = nm
            if isnu and nm:
                isin_to_name[isnu] = nm
            if tick and nm:
                ticker_u_to_name[tick] = nm
    all_yahoos = list(yahoo_to_name.keys())

    def _fuzzy_yahoo_one(base_sym: str) -> str | None:
        b = (base_sym or "").strip()
        if not b or "." in b:
            return None
        cands = [y for y in all_yahoos if y == b or y.startswith(b + ".")]
        if len(cands) == 1:
            return cands[0]
        return None

    def _trunc_disp(s: str) -> str:
        s = (s or "").strip()
        if len(s) <= 48:
            return s
        return s[:48] + "…"

    def _div_display_name_row(r: pd.Series) -> str:
        nm0 = str(r.get("nombre") or "").strip()
        tk = str(r.get("ticker") or "").strip()
        ty = str(r.get("ticker_Yahoo") or "").strip()
        isn = str(r.get("_isin_norm") or "").strip()

        def _is_placeholder_name(n: str) -> bool:
            if not n:
                return True
            nu, tku, tyu = n.upper(), tk.upper(), ty.upper()
            return nu == tku or nu == tyu

        if not _is_placeholder_name(nm0):
            return _trunc_disp(nm0)
        for k in (ty, tk):
            if k and k in yahoo_to_name:
                return _trunc_disp(yahoo_to_name[k])
        if isn:
            if isn in isin_to_name:
                return _trunc_disp(isin_to_name[isn])
            yy = lookup_ticker_yahoo_by_isin(isn)
            if yy and yy in yahoo_to_name:
                return _trunc_disp(yahoo_to_name[yy])
        for k in (ty, tk):
            fy = _fuzzy_yahoo_one(k)
            if fy and fy in yahoo_to_name:
                return _trunc_disp(yahoo_to_name[fy])
        for k in (tk, ty):
            ku = (k or "").strip().upper()
            if ku and ku in ticker_u_to_name:
                return _trunc_disp(ticker_u_to_name[ku])
        return _trunc_disp(nm0) if nm0 else (tk or ty or "—")

    d["_disp"] = d.apply(_div_display_name_row, axis=1)

    def _agg_disp_labels(s: pd.Series) -> str:
        vals = [str(x).strip() for x in s if str(x).strip() and str(x).strip() != "—"]
        return max(vals, key=len) if vals else "—"

    lab_map = d.groupby(d["_pk"].astype(str), sort=False)["_disp"].agg(_agg_disp_labels).to_dict()
    gp = d.groupby("_pk", as_index=False)[num_cols].sum()
    gp["Activo"] = gp["_pk"].astype(str).map(lab_map).fillna(gp["_pk"].astype(str))
    gp = gp.sort_values("netoWithReturnBaseCurrency", ascending=False).reset_index(drop=True)
    total_p = {c: float(gp[c].sum()) for c in num_cols}
    tbl_p = gp[["Activo"] + num_cols].rename(columns=disp_names)
    tot_p = pd.DataFrame([{"Activo": "TOTAL", **{disp_names[c]: total_p[c] for c in num_cols}}])
    tbl_show_p = pd.concat([tot_p, tbl_p], ignore_index=True)

    top_n = min(35, len(gp))
    if top_n > 0:
        fig_p = go.Figure(
            data=[
                go.Bar(
                    x=gp["Activo"].head(top_n),
                    y=gp["netoWithReturnBaseCurrency"].head(top_n),
                    marker_color="#5DADE2",
                    name=disp_names["netoWithReturnBaseCurrency"],
                )
            ]
        )
        fig_p.update_layout(
            title=f"Total neto con devolución (€) — top {top_n} activos",
            margin=dict(t=50, b=120, l=50, r=20),
            height=max(400, 80 + 14 * top_n),
            xaxis_tickangle=-45,
            yaxis_title="€",
            showlegend=False,
        )
        st.plotly_chart(fig_p, use_container_width=True)
    if len(gp) > top_n:
        st.caption(f"Gráfico: **{top_n}** activos con mayor total neto con devolución; la tabla incluye todos.")

    fmt_p = {c: (lambda x, col=c: fmt_eur(x) if pd.notna(x) else "–") for c in tbl_show_p.columns if "€" in c}
    st.dataframe(tbl_show_p.style.format(fmt_p, na_rep="–"), use_container_width=True, hide_index=True)


def render_analisis_distribucion(df_acciones: pd.DataFrame) -> None:
    """
    Distribución tipo Filios: tabla por posición y por tipo, con donut valor mercado / coste.
    G/P % sobre valor de mercado; dividendos en EUR por línea (totalBaseCurrency).
    """
    st.subheader("Distribución")
    st.caption(
        "Misma cartera que **Cartera** (FIFO acciones, fondos, cripto). Sin warrants ni opciones; sin líneas con cantidad ~0. "
        "Las **cotizaciones** son las mismas que en Cartera (un solo «Actualizar» sirve para ambas vistas). "
        "Elige **Todos** o una clase (acciones, ETF, cripto, fondos). **G/P %** = plusvalía / valor de mercado. "
        "Dividendos: suma de **totalBaseCurrency** (€) por cuenta y ticker Yahoo. "
        "**Cripto:** una fila por moneda (suma de cuentas). **Acciones / ETFs / fondos:** primero por ticker+cuenta; "
        "luego se agrupan por **ISIN** (columna del movimiento o catálogo de instrumentos) y clase; sin ISIN, por **nombre** "
        "normalizado. Conviene tener el ISIN bien cargado en movimientos o en Catálogo."
    )
    positions_full, sig_full = _ensure_cartera_enriched_session(df_acciones)
    base = _distribucion_filtrar_lineas(positions_full.copy())
    if base.empty:
        st.info("No hay posiciones para distribuir.")
        return
    base = base.copy()
    base["_Grupo"] = base.apply(_distribucion_grupo_linea, axis=1)
    base = base[base["_Grupo"].notna()].reset_index(drop=True)
    if base.empty:
        st.info("No hay posiciones en las clases mostradas (acciones, ETF, fondos, cripto).")
        return
    grupos_disp = sorted(
        base["_Grupo"].dropna().unique().tolist(), key=lambda g: DISTRIBUCION_GRUPOS_CLASE.index(g) if g in DISTRIBUCION_GRUPOS_CLASE else 99
    )
    sel_clase = st.radio(
        "Clases a incluir",
        options=["Todos", *DISTRIBUCION_GRUPOS_CLASE],
        horizontal=True,
        key="distrib_grupo_radio",
        help="«Todos» muestra todas las clases con posiciones. Puts, calls y warrants no entran en esta vista.",
    )
    st.caption("Puts, calls y otros derivados no están en esta vista.")
    grupos_sel = list(grupos_disp) if sel_clase == "Todos" else [sel_clase]
    base = base[base["_Grupo"].isin(grupos_sel)].reset_index(drop=True)
    if base.empty:
        st.info("No hay posiciones con la selección actual.")
        return

    precios_m = load_precios_manuales()
    car = st.session_state.get("cartera_enriched")
    can_use_cartera = _cartera_enriched_alineado_con_base(car, positions_full, sig_full)

    col_cot, _pad = st.columns([1, 2])
    with col_cot:
        if st.button("Actualizar cotizaciones", type="primary", key="btn_distrib_actualizar_cotiz"):
            with st.spinner("Obteniendo precios actuales…"):
                st.session_state["cartera_enriched"] = enrich_with_market_data(
                    positions_full.copy(), manual_prices=precios_m
                )
                st.session_state["cartera_enriched_updated_at"] = datetime.now().isoformat()
                if sig_full:
                    save_cotizaciones_cache(st.session_state["cartera_enriched"], sig_full)
            st.rerun()

    if can_use_cartera:
        enr = _distribucion_merge_mercado_desde_cartera(base, car)
        _distribucion_reaplicar_precios_manuales(enr, precios_m)
        upd = st.session_state.get("cartera_enriched_updated_at")
        if upd:
            try:
                dtu = datetime.fromisoformat(upd.replace("Z", "+00:00"))
                st.caption(
                    f"Cotizaciones del {dtu.strftime('%d/%m/%Y %H:%M')} (mismas que en Cartera). "
                    "Pulsa **Actualizar cotizaciones** aquí o en Cartera para refrescar."
                )
            except Exception:
                st.caption("Pulsa **Actualizar cotizaciones** aquí o en Cartera para refrescar.")
        else:
            st.caption("Cotizaciones cargadas (Cartera). Pulsa **Actualizar cotizaciones** para refrescar.")
    else:
        enr = _distribucion_shell_sin_mercado(base, precios_m)
        st.info(
            "Aún no hay cotizaciones en sesión para esta cartera. "
            "Pulsa **Actualizar cotizaciones** en **Cartera** o el botón de arriba: valdrá para ambas vistas."
        )

    div_agg = _distribucion_dividendos_eur_por_linea(load_dividendos())
    if not div_agg.empty:
        enr = enr.merge(div_agg, on=["Broker", "Ticker_Yahoo"], how="left")
    else:
        enr["Div_EUR"] = 0.0
    enr["Div_EUR"] = pd.to_numeric(enr["Div_EUR"], errors="coerce").fillna(0.0)
    enr = _distribucion_agregar_cripto_por_activo(enr)
    enr = _distribucion_agregar_acciones_etf_fondos_mismo_broker_ticker(enr)
    enr = _distribucion_agregar_acciones_etf_fondos_mismo_isin_o_nombre(enr)

    vm = pd.to_numeric(enr.get("Valor Mercado €"), errors="coerce")
    inv = pd.to_numeric(enr.get("Inversion €"), errors="coerce").fillna(0.0)
    gp = vm - inv
    tot_vm = float(np.nansum(vm))
    tot_inv = float(np.nansum(inv))
    pct_vm = np.where(tot_vm > 1e-9, (vm / tot_vm) * 100.0, np.nan)
    pct_inv = np.where(tot_inv > 1e-9, (inv / tot_inv) * 100.0, np.nan)
    pct_gp_m = np.where((vm.abs() > 1e-12) & (~pd.isna(vm)), (gp / vm) * 100.0, np.nan)

    tbl = pd.DataFrame(
        {
            "Posición": enr.get("Nombre", enr.get("Ticker_Yahoo", "")).astype(str),
            "Vlr mercado (€)": vm,
            "Vlr mercado (%)": pct_vm,
            "Tot inv + com/imp (€)": inv,
            "Tot inv + com/imp (%)": pct_inv,
            "G/P n/realiz (€)": gp,
            "G/P n/realiz (%)": pct_gp_m,
            "Tot dividendo (€)": enr["Div_EUR"],
        }
    )
    tot_row = pd.DataFrame(
        [
            {
                "Posición": "TOTAL",
                "Vlr mercado (€)": tot_vm,
                "Vlr mercado (%)": 100.0 if tot_vm > 1e-9 else np.nan,
                "Tot inv + com/imp (€)": tot_inv,
                "Tot inv + com/imp (%)": 100.0 if tot_inv > 1e-9 else np.nan,
                "G/P n/realiz (€)": tot_vm - tot_inv,
                "G/P n/realiz (%)": ((tot_vm - tot_inv) / tot_vm * 100.0) if abs(tot_vm) > 1e-9 else np.nan,
                "Tot dividendo (€)": float(enr["Div_EUR"].sum()),
            }
        ]
    )
    tbl_disp = pd.concat([tbl, tot_row], ignore_index=True)

    modo_graf = st.radio(
        "Gráficos (por posición y por tipo)",
        ["Valor de mercado", "Total invertido"],
        horizontal=True,
        key="distribucion_modo_grafana",
    )
    use_mercado = modo_graf == "Valor de mercado"

    c1, c2 = st.columns([1.25, 1.0])
    with c1:
        st.markdown("**Cartera por posición**")
        sty = tbl_disp.style.format(
            {
                "Vlr mercado (€)": lambda x: fmt_eur(x) if not (isinstance(x, float) and pd.isna(x)) else "–",
                "Vlr mercado (%)": "{:.2f}%".format,
                "Tot inv + com/imp (€)": lambda x: fmt_eur(x) if not (isinstance(x, float) and pd.isna(x)) else "–",
                "Tot inv + com/imp (%)": "{:.2f}%".format,
                "G/P n/realiz (€)": lambda x: fmt_eur(x) if not (isinstance(x, float) and pd.isna(x)) else "–",
                "G/P n/realiz (%)": "{:.2f}%".format,
                "Tot dividendo (€)": lambda x: fmt_eur(x) if not (isinstance(x, float) and pd.isna(x)) else "–",
            },
            na_rep="–",
        )

        def _gpcolor(s: pd.Series) -> list[str]:
            return [
                f"color: {_plusvalia_color_css(v)}; font-weight: 600" if i < len(s) - 1 else ""
                for i, v in enumerate(s)
            ]

        sty = sty.apply(_gpcolor, subset=["G/P n/realiz (€)"], axis=0)
        sty = sty.apply(_gpcolor, subset=["G/P n/realiz (%)"], axis=0)
        st.dataframe(sty, use_container_width=True, hide_index=True)
    with c2:
        vals = vm.fillna(0.0).tolist()
        if use_mercado:
            _distribucion_donut_plot(tbl["Posición"].tolist(), vals, "Distribución — valor mercado")
        else:
            _distribucion_donut_plot(tbl["Posición"].tolist(), inv.tolist(), "Distribución — total invertido")

    agg = (
        enr.groupby("_Grupo", as_index=False)
        .agg(
            {
                "Valor Mercado €": lambda s: float(pd.to_numeric(s, errors="coerce").sum()),
                "Inversion €": lambda s: float(pd.to_numeric(s, errors="coerce").sum()),
                "Div_EUR": "sum",
            }
        )
        .rename(columns={"_Grupo": "Tipo", "Valor Mercado €": "vm", "Inversion €": "inv", "Div_EUR": "div"})
    )
    _ord_map = {g: i for i, g in enumerate(DISTRIBUCION_GRUPOS_CLASE)}
    agg["_ord"] = agg["Tipo"].map(lambda x: _ord_map.get(str(x), 99))
    agg = agg.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)
    agg["gp"] = agg["vm"] - agg["inv"]
    t_vm2 = float(agg["vm"].sum())
    t_inv2 = float(agg["inv"].sum())
    agg["pct_vm"] = np.where(t_vm2 > 1e-9, agg["vm"] / t_vm2 * 100.0, np.nan)
    agg["pct_inv"] = np.where(t_inv2 > 1e-9, agg["inv"] / t_inv2 * 100.0, np.nan)
    agg["pct_gp_m"] = np.where(agg["vm"].abs() > 1e-9, agg["gp"] / agg["vm"] * 100.0, np.nan)

    tbl2 = pd.DataFrame(
        {
            "Tipo": agg["Tipo"],
            "Vlr mercado (€)": agg["vm"],
            "Vlr mercado (%)": agg["pct_vm"],
            "Tot inv + com/imp (€)": agg["inv"],
            "Tot inv + com/imp (%)": agg["pct_inv"],
            "G/P n/realiz (€)": agg["gp"],
            "G/P n/realiz (%)": agg["pct_gp_m"],
            "Tot dividendo (€)": agg["div"],
        }
    )
    tot_row2 = pd.DataFrame(
        [
            {
                "Tipo": "TOTAL",
                "Vlr mercado (€)": t_vm2,
                "Vlr mercado (%)": 100.0 if t_vm2 > 1e-9 else np.nan,
                "Tot inv + com/imp (€)": t_inv2,
                "Tot inv + com/imp (%)": 100.0 if t_inv2 > 1e-9 else np.nan,
                "G/P n/realiz (€)": t_vm2 - t_inv2,
                "G/P n/realiz (%)": ((t_vm2 - t_inv2) / t_vm2 * 100.0) if abs(t_vm2) > 1e-9 else np.nan,
                "Tot dividendo (€)": float(agg["div"].sum()),
            }
        ]
    )
    tbl2_disp = pd.concat([tbl2, tot_row2], ignore_index=True)

    c3, c4 = st.columns([1.25, 1.0])
    with c3:
        st.markdown("**Cartera por tipo de posición**")
        sty2 = tbl2_disp.style.format(
            {
                "Vlr mercado (€)": lambda x: fmt_eur(x) if not (isinstance(x, float) and pd.isna(x)) else "–",
                "Vlr mercado (%)": "{:.2f}%".format,
                "Tot inv + com/imp (€)": lambda x: fmt_eur(x) if not (isinstance(x, float) and pd.isna(x)) else "–",
                "Tot inv + com/imp (%)": "{:.2f}%".format,
                "G/P n/realiz (€)": lambda x: fmt_eur(x) if not (isinstance(x, float) and pd.isna(x)) else "–",
                "G/P n/realiz (%)": "{:.2f}%".format,
                "Tot dividendo (€)": lambda x: fmt_eur(x) if not (isinstance(x, float) and pd.isna(x)) else "–",
            },
            na_rep="–",
        )
        sty2 = sty2.apply(
            lambda s: [f"color: {_plusvalia_color_css(v)}; font-weight: 600" for v in s],
            subset=["G/P n/realiz (€)"],
            axis=0,
        )
        sty2 = sty2.apply(
            lambda s: [f"color: {_plusvalia_color_css(v)}; font-weight: 600" for v in s],
            subset=["G/P n/realiz (%)"],
            axis=0,
        )
        st.dataframe(sty2, use_container_width=True, hide_index=True)
    with c4:
        if use_mercado:
            _distribucion_donut_plot(agg["Tipo"].tolist(), agg["vm"].tolist(), "Por tipo — valor mercado")
        else:
            _distribucion_donut_plot(agg["Tipo"].tolist(), agg["inv"].tolist(), "Por tipo — total invertido")


def fmt_eur(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:,.2f} €"


# Color solo en la cifra (CSS sobre st.metric nativo; ver marcadores #fiscal-accent-* en Fiscalidad)
_FISC_ACCENT_RENTA = "#6DB3FF"
_FISC_ACCENT_P2P = "#C4E860"
_FISC_ACCENT_TOTAL_EXT = "#D4B896"
_FISC_BROWN_BRUTO = "#C48A4E"
_FISC_BROWN_RET = "#E6C9A8"


def _plusvalia_color_css(value: float | None) -> str:
    """Verde si ganancia, rojo si pérdida, gris si cero o sin dato."""
    if value is None or pd.isna(value):
        return "#9e9e9e"
    v = float(value)
    if v > 0:
        return "#2e7d32"
    if v < 0:
        return "#c62828"
    return "#9e9e9e"


def _st_metric_colored(label: str, value_text: str, value_color_css: str) -> None:
    """Imita st.metric con el valor en color; tamaño de cifra alineado con Streamlit (~stMetricValue)."""
    st.markdown(
        f'<div style="margin:0;padding:0;">'
        f'<div style="color:#a0a0a0;font-size:0.8rem;line-height:1.2;margin-bottom:0.25rem;">{label}</div>'
        f'<div style="color:{value_color_css};font-size:2.5rem;font-weight:700;line-height:1.05;letter-spacing:-0.02em;">'
        f"{value_text}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


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


def _cartera_positions_col_width_px(df: pd.DataFrame, col: str, *, min_px: int = 76, max_px: int = 520, char_px: int = 7, pad: int = 26) -> int:
    """Ancho de columna en px según cabecera y contenido (coherente con el formateo del styler de Cartera)."""
    header_len = len(str(col))
    if col not in df.columns or df.empty:
        return min_px

    if col == "Titulos":
        data_max = max((len(fmt_qty(v)) for v in df[col]), default=0)
    elif col in ("Ultima cotizacion (moneda)", "Precio medio (moneda)"):
        data_max = int(df[col].astype(str).str.len().max())
    elif col in (
        "Precio md + com/imp (€)",
        "Total inv + com/imp (€)",
        "Valor mercado (€)",
        "GyP no realizadas (€)",
        "GyP hoy (€)",
    ):
        lens = []
        for v in pd.to_numeric(df[col], errors="coerce"):
            if pd.isna(v):
                continue
            lens.append(len(fmt_eur(float(v))))
        data_max = max(lens) if lens else 12
    elif col in ("GyP no realizadas %", "GyP hoy %"):
        lens = []
        for v in pd.to_numeric(df[col], errors="coerce"):
            if pd.isna(v):
                continue
            lens.append(len(f"{float(v):.2f} %"))
        data_max = max(lens) if lens else 6
    else:
        s = df[col].fillna("").astype(str).str.replace("\n", " ", regex=False)
        data_max = int(s.str.len().max()) if len(s) else 0

    ml = max(header_len, data_max)
    return int(min(max_px, max(min_px, pad + ml * char_px)))


def _cartera_positions_column_config(df: pd.DataFrame, cols: list[str]) -> dict:
    tc = st.column_config
    cfg = {}
    for c in cols:
        cap = 560 if c == "Nombre" else 360 if c == "Ticker" else 520
        w = _cartera_positions_col_width_px(df, c, max_px=cap)
        if c == "Ticker":
            cfg[c] = tc.TextColumn(c, pinned=True, width=w)
        elif c == "Nombre":
            cfg[c] = tc.TextColumn(c, width=w)
        else:
            cfg[c] = tc.Column(c, width=w)
    return cfg


def _fifo_norm_fecha_hist(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    if not s:
        return ""
    return s.split("T")[0].strip()[:10]


def _fifo_tipo_to_origen_fifo(tipo: str) -> str:
    t = str(tipo or "").strip().lower()
    if t == "fund":
        return "Fondos"
    if t == "crypto":
        return "Cripto"
    return "Acciones/ETFs"


def _fifo_first_nonempty(series: pd.Series) -> str:
    """Primer valor no vacío en la columna (p. ej. nombre/ticker repetidos en tramos de una misma venta)."""
    for x in series:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            continue
        s = str(x).strip()
        if s:
            return s
    return ""


_fifo_cons_cols = [
    "Origen FIFO",
    "Broker",
    "_yahoo",
    "_fecha",
    "_pm",
    "consumida",
    "Nombre_ref",
    "Ticker_ref",
]


def _fifo_consume_groups_from_detail(sd: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrupa tramos FIFO por lote: cantidad consumida y refs nombre/ticker."""
    empty_non = pd.DataFrame(columns=_fifo_cons_cols)
    empty_c = pd.DataFrame(columns=_fifo_cons_cols)
    if sd is None or sd.empty:
        return empty_non, empty_c
    sdc = sd.copy()
    sdc["_yahoo"] = sdc.apply(
        lambda r: str(r.get("Ticker_Yahoo") or r.get("Ticker") or "").strip(),
        axis=1,
    )
    sdc["_fecha"] = sdc["Fecha origen (lote)"].map(_fifo_norm_fecha_hist)
    q = pd.to_numeric(sdc["Cantidad (tramo)"], errors="coerce").fillna(0.0)
    v = pd.to_numeric(sdc["Valor compra histórico (€)"], errors="coerce").fillna(0.0)
    sdc["_pm"] = (v / q.replace(0, np.nan)).fillna(0.0).round(8)
    sdc["Origen FIFO"] = sdc["Origen FIFO"].astype(str)
    mask_c = sdc["Origen FIFO"] == "Cripto"
    cons_non = empty_non
    cons_c = empty_c
    if not sdc[~mask_c].empty:
        cons_non = (
            sdc[~mask_c]
            .groupby(["Origen FIFO", "Broker", "_yahoo", "_fecha", "_pm"], as_index=False, dropna=False)
            .agg(
                consumida=("Cantidad (tramo)", "sum"),
                Nombre_ref=("Nombre", _fifo_first_nonempty),
                Ticker_ref=("Ticker", _fifo_first_nonempty),
            )
        )
    if not sdc[mask_c].empty:
        cons_c = (
            sdc[mask_c]
            .groupby(["Origen FIFO", "_yahoo", "_fecha", "_pm"], as_index=False, dropna=False)
            .agg(
                consumida=("Cantidad (tramo)", "sum"),
                Nombre_ref=("Nombre", _fifo_first_nonempty),
                Ticker_ref=("Ticker", _fifo_first_nonempty),
            )
        )
        cons_c["Broker"] = "—"
    return cons_non, cons_c


def _fifo_merge_consumida_y(cons: pd.DataFrame, cy: pd.DataFrame | None, keys: list[str]) -> pd.DataFrame:
    """Añade consumida_y desde tramos del ejercicio; outer para claves solo en uno de los lados."""
    if cy is None or cy.empty:
        out = cons.copy()
        out["consumida_y"] = 0.0
        return out
    cy_m = cy[keys + ["consumida"]].rename(columns={"consumida": "consumida_y"})
    if cons.empty:
        out = cy.copy()
        out["consumida_y"] = pd.to_numeric(out["consumida"], errors="coerce").fillna(0.0)
        return out
    out = cons.merge(cy_m, on=keys, how="outer")
    out["consumida"] = pd.to_numeric(out["consumida"], errors="coerce")
    out["consumida_y"] = pd.to_numeric(out["consumida_y"], errors="coerce")
    out["consumida"] = out["consumida"].fillna(out["consumida_y"])
    out["consumida_y"] = out["consumida_y"].fillna(0.0)
    return out


def _fifo_mismo_anio_label(fecha_lote, fechas_venta_ej: object) -> str:
    """
    «✓» si el año natural de la compra (fecha lote) coincide con el de todas las fechas
    de venta del ejercicio en esa celda; vacío si no hay ventas en ejercicio o los años no coinciden.
    """
    fl = pd.to_datetime(fecha_lote, errors="coerce")
    if pd.isna(fl):
        return ""
    y_lote = int(fl.year)
    s = str(fechas_venta_ej or "").strip()
    if not s:
        return ""
    years_v: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        dv = pd.to_datetime(part, errors="coerce")
        if not pd.isna(dv):
            years_v.add(int(dv.year))
    if not years_v:
        return ""
    if years_v == {y_lote}:
        return "✓"
    return ""


def build_fifo_lote_estado_ledger(
    lots_df: pd.DataFrame | None,
    sales_detail_df: pd.DataFrame | None,
    sales_detail_ejercicio: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Reconstruye cada lote FIFO: cantidad inicial ≈ restante + consumida, y Estado Vivo / Parcial / Agotado.
    Cripto usa FIFO global por ticker: Broker «—» en esta vista.
    Si sales_detail_ejercicio no es None, se añade «Consumida (ejercicio)» (solo tramos con Fecha venta en ese año);
    «Consumida», Estado y % consumido siguen usando el histórico en sales_detail_df.
    Con fechas de venta en ejercicio, se añade «Mismo año» (✓ si compra y venta comparten año natural).
    """
    has_full = sales_detail_df is not None and not sales_detail_df.empty
    has_y = sales_detail_ejercicio is not None and not sales_detail_ejercicio.empty
    has_lots = lots_df is not None and not lots_df.empty
    ejercicio_cols = sales_detail_ejercicio is not None
    keys_merge = ["Origen FIFO", "Broker", "_yahoo", "_fecha", "_pm"]
    if not has_lots and not has_full and not has_y:
        return pd.DataFrame()

    live_non = pd.DataFrame(
        columns=["Origen FIFO", "Broker", "_yahoo", "_fecha", "_pm", "restante", "Ticker", "Nombre"]
    )
    live_c = pd.DataFrame(
        columns=["Origen FIFO", "Broker", "_yahoo", "_fecha", "_pm", "restante", "Ticker", "Nombre"]
    )

    if lots_df is not None and not lots_df.empty:
        ld = lots_df.copy()
        if "Ticker_Yahoo" in ld.columns:
            ld["_yahoo"] = ld.apply(
                lambda r: str(r.get("Ticker_Yahoo") or r.get("Ticker") or "").strip(),
                axis=1,
            )
        else:
            ld["_yahoo"] = ld.get("Ticker", pd.Series([""] * len(ld))).astype(str).str.strip()
        ld["_fecha"] = ld["Fecha origen"].map(_fifo_norm_fecha_hist)
        ld["_pm"] = pd.to_numeric(ld["Precio medio €"], errors="coerce").fillna(0.0).round(8)
        tipo_ser = ld.get("Tipo activo", pd.Series([""] * len(ld))).astype(str).str.lower()
        ld["_is_c"] = tipo_ser == "crypto"
        if not ld[~ld["_is_c"]].empty:
            live_non = (
                ld[~ld["_is_c"]]
                .groupby(["Broker", "_yahoo", "_fecha", "_pm"], as_index=False, dropna=False)
                .agg(
                    restante=("Cantidad", "sum"),
                    Ticker=("Ticker", "first"),
                    Nombre=("Nombre", "first"),
                    Tipo_act=("Tipo activo", "first"),
                )
            )
            live_non["Origen FIFO"] = live_non["Tipo_act"].map(_fifo_tipo_to_origen_fifo)
            live_non = live_non.drop(columns=["Tipo_act"], errors="ignore")
        if not ld[ld["_is_c"]].empty:
            live_c = (
                ld[ld["_is_c"]]
                .groupby(["_yahoo", "_fecha", "_pm"], as_index=False, dropna=False)
                .agg(
                    restante=("Cantidad", "sum"),
                    Ticker=("Ticker", "first"),
                    Nombre=("Nombre", "first"),
                )
            )
            live_c["Origen FIFO"] = "Cripto"
            live_c["Broker"] = "—"

    if has_full:
        cons_non, cons_c = _fifo_consume_groups_from_detail(sales_detail_df)
        if ejercicio_cols:
            if has_y:
                cny, ccy = _fifo_consume_groups_from_detail(sales_detail_ejercicio)
                cons_non = _fifo_merge_consumida_y(cons_non, cny, keys_merge)
                cons_c = _fifo_merge_consumida_y(cons_c, ccy, keys_merge)
            else:
                cons_non = cons_non.copy()
                cons_c = cons_c.copy()
                cons_non["consumida_y"] = 0.0
                cons_c["consumida_y"] = 0.0
    elif has_y:
        cons_non, cons_c = _fifo_consume_groups_from_detail(sales_detail_ejercicio)
        if ejercicio_cols:
            cons_non = cons_non.copy()
            cons_c = cons_c.copy()
            cons_non["consumida_y"] = pd.to_numeric(cons_non["consumida"], errors="coerce").fillna(0.0)
            cons_c["consumida_y"] = pd.to_numeric(cons_c["consumida"], errors="coerce").fillna(0.0)
    else:
        cols_cons = _fifo_cons_cols + (["consumida_y"] if ejercicio_cols else [])
        cons_non = pd.DataFrame(columns=cols_cons)
        cons_c = pd.DataFrame(columns=cols_cons)

    m_non = pd.merge(
        live_non,
        cons_non,
        on=["Origen FIFO", "Broker", "_yahoo", "_fecha", "_pm"],
        how="outer",
    )
    m_c = pd.merge(
        live_c,
        cons_c,
        on=["Origen FIFO", "Broker", "_yahoo", "_fecha", "_pm"],
        how="outer",
    )
    ledger = pd.concat([m_non, m_c], ignore_index=True)
    ledger["restante"] = pd.to_numeric(ledger["restante"], errors="coerce").fillna(0.0)
    ledger["consumida"] = pd.to_numeric(ledger["consumida"], errors="coerce").fillna(0.0)
    if ejercicio_cols:
        ledger["consumida_y"] = pd.to_numeric(ledger.get("consumida_y"), errors="coerce").fillna(0.0)
    # Lotes solo en ventas (agotados) no tienen fila en lots_df: nombre/ticker vienen del detalle FIFO.
    ledger["Ticker"] = ledger.get("Ticker", pd.Series([np.nan] * len(ledger)))
    ledger["Ticker_ref"] = ledger.get("Ticker_ref", pd.Series([np.nan] * len(ledger)))
    ledger["Ticker"] = ledger["Ticker"].fillna(ledger["Ticker_ref"]).fillna(ledger["_yahoo"]).fillna("")
    ledger["Nombre"] = ledger.get("Nombre", pd.Series([np.nan] * len(ledger)))
    ledger["Nombre_ref"] = ledger.get("Nombre_ref", pd.Series([np.nan] * len(ledger)))
    ledger["Nombre"] = (
        ledger["Nombre"].fillna("")
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .fillna(ledger["Nombre_ref"])
        .fillna("")
        .astype(str)
    )
    ledger = ledger.drop(columns=["Nombre_ref", "Ticker_ref"], errors="ignore")
    ledger["inicial"] = ledger["restante"] + ledger["consumida"]
    eps = max(MIN_POSITION * 100, 1e-6)
    ledger["% consumido"] = np.where(
        ledger["inicial"] > eps,
        (ledger["consumida"] / ledger["inicial"] * 100.0).round(1),
        0.0,
    )

    def _est(s: pd.Series) -> str:
        rest = float(s["restante"])
        cons = float(s["consumida"])
        if rest <= eps:
            return "Agotado"
        if cons <= eps:
            return "Vivo"
        return "Parcial"

    ledger["Estado"] = ledger.apply(_est, axis=1)
    ren = {
        "_yahoo": "Yahoo/Ticker",
        "_pm": "Precio medio €",
        "_fecha": "Fecha lote",
        "inicial": "Cant. inicial",
        "consumida": "Consumida",
        "restante": "Restante",
    }
    if ejercicio_cols:
        ren["consumida_y"] = "Consumida (ejercicio)"
    out = ledger.rename(columns=ren)
    isins: list[str] = []
    cat_cache_isin: dict[str, str] = {}
    for _, r in out.iterrows():
        if str(r.get("Origen FIFO", "")).strip() == "Cripto":
            isins.append("")
            continue
        got = ""
        for cand in (r.get("Yahoo/Ticker"), r.get("Ticker")):
            ni = _norm_isin_field(cand)
            if ni:
                got = ni
                break
        if not got:
            ty = str(r.get("Yahoo/Ticker") or "").strip()
            if ty:
                if ty not in cat_cache_isin:
                    cat_cache_isin[ty] = _lookup_isin_for_ticker_yahoo(ty)
                got = cat_cache_isin[ty] or ""
        isins.append(got)
    out = out.copy()
    out["ISIN"] = isins
    if (
        has_y
        and sales_detail_ejercicio is not None
        and not sales_detail_ejercicio.empty
        and "Fecha venta" in sales_detail_ejercicio.columns
    ):
        dn_fv = _fifo_sales_detail_norm_keys(sales_detail_ejercicio)
        fv_by_lote = (
            dn_fv.groupby(
                ["Origen FIFO", "Broker", "_yahoo", "_fecha", "_pm"],
                dropna=False,
            )["Fecha venta"]
            .apply(
                lambda s: ", ".join(
                    sorted(
                        {
                            pd.Timestamp(x).strftime("%Y-%m-%d")
                            for x in pd.to_datetime(s, errors="coerce").dropna()
                        }
                    )
                )
            )
            .reset_index(name="_fechas_v_ej")
        )
        fv_by_lote = fv_by_lote.rename(
            columns={
                "_yahoo": "Yahoo/Ticker",
                "_fecha": "Fecha lote",
                "_pm": "Precio medio €",
            }
        )
        out["Precio medio €"] = pd.to_numeric(out["Precio medio €"], errors="coerce").round(8)
        fv_by_lote["Precio medio €"] = pd.to_numeric(
            fv_by_lote["Precio medio €"], errors="coerce"
        ).round(8)
        out = out.merge(
            fv_by_lote,
            on=["Origen FIFO", "Broker", "Yahoo/Ticker", "Fecha lote", "Precio medio €"],
            how="left",
        )
        out["Fechas venta (ej.)"] = out["_fechas_v_ej"].fillna("").astype(str)
        out = out.drop(columns=["_fechas_v_ej"])
        out["Mismo año"] = out.apply(
            lambda r: _fifo_mismo_anio_label(r.get("Fecha lote"), r.get("Fechas venta (ej.)")),
            axis=1,
        )
    cols_show = [
        "Origen FIFO",
        "Broker",
        "Ticker",
        "Yahoo/Ticker",
        "ISIN",
        "Nombre",
        "Fecha lote",
        "Fechas venta (ej.)",
        "Mismo año",
        "Precio medio €",
        "Cant. inicial",
        "Consumida",
        "Consumida (ejercicio)",
        "Restante",
        "% consumido",
        "Estado",
    ]
    cols_show = [c for c in cols_show if c in out.columns]
    out = out[cols_show]
    return out.sort_values(
        ["Origen FIFO", "Broker", "Fecha lote", "Ticker"],
        ascending=True,
        na_position="last",
    ).reset_index(drop=True)


def style_fifo_lote_estado_row(row: pd.Series):
    """Fondo gris = agotado; ámbar = parcial; sin estilo = vivo."""
    n = len(row)
    e = str(row.get("Estado", ""))
    if e == "Agotado":
        return ["background-color: #e8e8e8; color: #424242;"] * n
    if e == "Parcial":
        return ["background-color: #fff8e1; color: #4e342e;"] * n
    return [""] * n


def _regla2m_build_isin_events(
    df_mov_acciones: pd.DataFrame,
    df_mov_fondos: pd.DataFrame,
    cat_cache: dict[str, str],
) -> list[dict]:
    """Cronología unificada por ISIN (saldo global del ISIN, alineado con FIFO fiscal por ISIN)."""
    events: list[dict] = []
    def _push(df: pd.DataFrame, src_tag: str, base_seq: int) -> None:
        if df is None or df.empty:
            return
        for k, (_, row) in enumerate(df.iterrows()):
            tipo_m = str(row.get("type") or "").strip().lower()
            if tipo_m in ("buy", "switchbuy", "sell", "switch", "split"):
                pass
            else:
                continue
            ts = row.get("datetime_full")
            if ts is None or pd.isna(ts):
                ts = pd.to_datetime(row.get("date"), errors="coerce")
            if pd.isna(ts):
                continue
            ty = str(row.get("ticker_Yahoo") or row.get("ticker") or "").strip()
            to = str(row.get("ticker") or "").strip()
            isin_e = _fifo_resolve_isin_row(row, ty, to, cat_cache)
            if not isin_e:
                continue
            br = str(row.get("broker") or "").strip()
            qty = pd.to_numeric(row.get("positionNumber"), errors="coerce")
            if tipo_m == "split":
                if pd.isna(qty) or float(qty) <= 0:
                    continue
                tord = 0
                events.append({
                    "ts": pd.Timestamp(ts),
                    "isin": isin_e,
                    "kind": "split",
                    "factor": float(qty),
                    "delta": 0.0,
                    "qty": float(qty),
                    "broker": br,
                    "yahoo": ty,
                    "ticker": to,
                    "orig": src_tag,
                    "tord": tord,
                    "idx": k + base_seq,
                })
                continue
            if pd.isna(qty) or float(qty) <= 0:
                continue
            q = float(qty)
            if tipo_m in ("buy", "switchbuy"):
                delta = q
                kind = "buy"
                tord = 2 if tipo_m == "switchbuy" else 1
            elif tipo_m in ("sell", "switch"):
                delta = -q
                kind = "sell"
                tord = 0 if tipo_m == "switch" else 1
            else:
                continue
            events.append({
                "ts": pd.Timestamp(ts),
                "isin": isin_e,
                "kind": kind,
                "factor": 1.0,
                "delta": delta,
                "qty": q,
                "broker": br,
                "yahoo": ty,
                "ticker": to,
                "orig": src_tag,
                "tord": tord,
                "idx": k + base_seq,
            })

    _push(df_mov_acciones, "acc", 0)
    _push(df_mov_fondos, "fdo", 10_000_000)
    events.sort(key=lambda e: (e["ts"], e.get("tord", 9), e["idx"]))
    return events


def deteccion_regla_dos_meses_isin_alerts(
    df_mov_acciones: pd.DataFrame,
    df_mov_fondos: pd.DataFrame,
    sales_acciones: pd.DataFrame,
    sales_fondos: pd.DataFrame,
    ejercicio: int | None = None,
) -> pd.DataFrame:
    """
    Ventas en pérdida y compras del mismo ISIN en ±2 meses; la alerta fuerte solo si
    tras la venta queda posición del ISIN o hay recompra en la ventana (aplazamiento típico).
    Liquidación total sin recompras posteriores → No revisar (orientativo).
    """
    cols_out = [
        "Ejercicio",
        "Origen venta",
        "Fecha venta",
        "Broker",
        "Ticker",
        "Ticker Yahoo",
        "ISIN",
        "Pérdida (€)",
        "N compras ±2m",
        "Compras candidatas (fechas)",
        "Pos. ISIN tras venta",
        "Alerta",
    ]
    eps = 1e-6
    q_eps = max(MIN_POSITION * 1000, 1e-8)
    parts_s: list[pd.DataFrame] = []
    if sales_acciones is not None and not sales_acciones.empty:
        parts_s.append(sales_acciones.assign(**{"_origen_venta": "Acciones/ETFs"}))
    if sales_fondos is not None and not sales_fondos.empty:
        parts_s.append(sales_fondos.assign(**{"_origen_venta": "Fondos"}))
    if not parts_s:
        return pd.DataFrame(columns=cols_out)
    sales = pd.concat(parts_s, ignore_index=True)
    tipo_a = sales.get("Tipo activo", pd.Series([""] * len(sales))).astype(str).str.strip().str.lower()
    sales = sales[(tipo_a != "crypto") & (~tipo_a.isin(["putoption", "calloption"]))].copy()
    if sales.empty:
        return pd.DataFrame(columns=cols_out)
    pnl = pd.to_numeric(sales.get("Plusvalía / Minusvalía (€)"), errors="coerce").fillna(0.0)
    sales = sales[pnl < -eps].copy()
    if ejercicio is not None and "Fecha venta" in sales.columns:
        fv_y = pd.to_datetime(sales["Fecha venta"], errors="coerce").dt.year
        sales = sales[fv_y == ejercicio]
    if sales.empty:
        return pd.DataFrame(columns=cols_out)

    cat_cache: dict[str, str] = {}
    pur_rows: list[dict] = []

    def _ingest_compras(df: pd.DataFrame, origen_mov: str) -> None:
        if df is None or df.empty:
            return
        for _, row in df.iterrows():
            tipo_m = str(row.get("type") or "").strip().lower()
            if tipo_m not in ("buy", "switchbuy"):
                continue
            qty = pd.to_numeric(row.get("positionNumber"), errors="coerce")
            if pd.isna(qty) or float(qty) <= 0:
                continue
            br = row.get("broker")
            if br is None or (isinstance(br, float) and pd.isna(br)) or str(br).strip() == "":
                continue
            dt = pd.to_datetime(row.get("date"), errors="coerce")
            if pd.isna(dt):
                continue
            ty = str(row.get("ticker_Yahoo") or row.get("ticker") or "").strip()
            to = str(row.get("ticker") or "").strip()
            isin_c = _fifo_resolve_isin_row(row, ty, to, cat_cache)
            if not isin_c:
                continue
            pur_rows.append(
                {
                    "fecha": pd.Timestamp(dt).normalize(),
                    "isin": isin_c,
                    "origen_mov": origen_mov,
                }
            )

    _ingest_compras(df_mov_acciones, "Acciones/ETFs")
    _ingest_compras(df_mov_fondos, "Fondos")
    purchases = pd.DataFrame(pur_rows)
    events = _regla2m_build_isin_events(df_mov_acciones, df_mov_fondos, cat_cache)
    alert_rows: list[dict] = []

    for _, sr in sales.iterrows():
        fv = pd.to_datetime(sr.get("Fecha venta"), errors="coerce")
        if pd.isna(fv):
            continue
        fv_n = pd.Timestamp(fv).normalize()
        low = fv_n - pd.DateOffset(months=2)
        high = fv_n + pd.DateOffset(months=2)
        ty_sale = str(sr.get("Ticker_Yahoo") or sr.get("Ticker") or "").strip()
        to_sale = str(sr.get("Ticker") or "").strip()
        fake = pd.Series({"ticker": to_sale, "ticker_Yahoo": ty_sale, "isin": "", "ISIN": ""})
        isin_sale = _fifo_resolve_isin_row(fake, ty_sale, to_sale, cat_cache)
        _pn = pd.to_numeric(sr.get("Plusvalía / Minusvalía (€)"), errors="coerce")
        pnl_v = float(_pn) if not pd.isna(_pn) else 0.0
        origen_v = str(sr.get("_origen_venta") or "")
        qty_sale = float(pd.to_numeric(sr.get("Cantidad vendida"), errors="coerce") or 0.0)
        br_sale = str(sr.get("Broker") or "").strip()

        if not isin_sale:
            alert_rows.append({
                "Ejercicio": int(fv_n.year),
                "Origen venta": origen_v,
                "Fecha venta": fv_n.strftime("%Y-%m-%d"),
                "Broker": sr.get("Broker", ""),
                "Ticker": sr.get("Ticker", ""),
                "Ticker Yahoo": ty_sale,
                "ISIN": "",
                "Pérdida (€)": pnl_v,
                "N compras ±2m": 0,
                "Compras candidatas (fechas)": "",
                "Pos. ISIN tras venta": "—",
                "Alerta": "Sin ISIN en catálogo/movimiento — no se puede cruzar por ISIN",
            })
            continue

        if purchases.empty:
            cand = pd.DataFrame()
        else:
            m = (
                (purchases["isin"] == isin_sale)
                & (purchases["fecha"] >= low)
                & (purchases["fecha"] <= high)
            )
            cand = purchases.loc[m]
        n_c = int(len(cand))
        fechas_u = sorted({d.strftime("%Y-%m-%d") for d in cand["fecha"]}) if n_c else []
        detalle = "; ".join(fechas_u[:25])
        if len(fechas_u) > 25:
            detalle += " …"

        pos_after: str | float = "n/d"
        recompra = False
        if events:
            bal: dict[str, float] = {}
            matched_ts: pd.Timestamp | None = None
            remain = 0.0
            qtol = max(q_eps, abs(qty_sale) * 1e-9 + 1e-8)
            for e in events:
                isin_ev = e["isin"]
                if e["kind"] == "split":
                    bal[isin_ev] = bal.get(isin_ev, 0.0) * e["factor"]
                else:
                    bal[isin_ev] = bal.get(isin_ev, 0.0) + e["delta"]
                if (
                    isin_ev == isin_sale
                    and e["kind"] == "sell"
                    and str(e["broker"]).strip() == br_sale
                    and abs(e["qty"] - qty_sale) <= qtol
                    and pd.Timestamp(e["ts"]).normalize() == fv_n
                    and (
                        not ty_sale
                        or str(e.get("yahoo") or "").strip() == ty_sale
                        or str(e.get("ticker") or "").strip() == to_sale
                    )
                ):
                    matched_ts = e["ts"]
                    remain = bal.get(isin_sale, 0.0)

            if matched_ts is not None:
                pos_after = round(remain, 6)
                sale_ts_eff = matched_ts
            else:
                bal_eod: dict[str, float] = {}
                for e in events:
                    if pd.Timestamp(e["ts"]).normalize() > fv_n:
                        break
                    if e["kind"] == "split":
                        bal_eod[e["isin"]] = bal_eod.get(e["isin"], 0.0) * e["factor"]
                    else:
                        bal_eod[e["isin"]] = bal_eod.get(e["isin"], 0.0) + e["delta"]
                remain = bal_eod.get(isin_sale, 0.0)
                pos_after = round(remain, 6)
                sale_ts_eff = pd.Timestamp(fv_n) + pd.Timedelta(hours=23, minutes=59, seconds=59)

            if isinstance(pos_after, float) and abs(pos_after) <= q_eps:
                pos_after = 0.0

            for e in events:
                if e["isin"] != isin_sale or e["kind"] != "buy":
                    continue
                if e["ts"] <= sale_ts_eff:
                    continue
                if pd.Timestamp(e["ts"]).normalize() <= high:
                    recompra = True
                    break

        if n_c == 0:
            msg = "No — sin compras con mismo ISIN en ±2 meses naturales desde fecha venta"
        elif pos_after == "n/d" and n_c > 0:
            msg = "Revisar — compras en ventana; no se pudo acoplar la venta a un movimiento (comprueba fechas/cantidades)"
        elif n_c > 0 and isinstance(pos_after, float) and pos_after <= q_eps and not recompra:
            msg = (
                "No — liquidación total del ISIN y sin recompras tras la venta en la ventana; "
                "la minusvalía suele integrarse en esa venta (orientativo)"
            )
        elif n_c > 0 and isinstance(pos_after, float) and pos_after <= q_eps and recompra:
            msg = "Sí — recompra del ISIN tras la venta dentro de ±2 meses (revisar integración aplazada)"
        elif n_c > 0 and isinstance(pos_after, float) and pos_after > q_eps:
            msg = "Sí — siguen existiendo títulos del ISIN en cartera tras la venta; posible aplazamiento (revisar)"
        else:
            msg = "No — sin compras con mismo ISIN en ±2 meses naturales desde fecha venta"

        alert_rows.append({
            "Ejercicio": int(fv_n.year),
            "Origen venta": origen_v,
            "Fecha venta": fv_n.strftime("%Y-%m-%d"),
            "Broker": sr.get("Broker", ""),
            "Ticker": sr.get("Ticker", ""),
            "Ticker Yahoo": ty_sale,
            "ISIN": isin_sale,
            "Pérdida (€)": pnl_v,
            "N compras ±2m": n_c,
            "Compras candidatas (fechas)": detalle,
            "Pos. ISIN tras venta": pos_after,
            "Alerta": msg,
        })

    return pd.DataFrame(alert_rows)[cols_out] if alert_rows else pd.DataFrame(columns=cols_out)


def _fifo_sales_detail_norm_keys(sd: pd.DataFrame) -> pd.DataFrame:
    """Alinea columnas de detalle con las claves del ledger FIFO."""
    x = sd.copy()
    if "Origen FIFO" in x.columns and "Broker" in x.columns:
        orig = x["Origen FIFO"].astype(str).str.strip()
        mask_c = orig.str.casefold() == "cripto"
        if mask_c.any():
            # Misma convención que build_fifo_lote_estado_ledger (FIFO cripto global → Broker «—»).
            x.loc[mask_c, "Broker"] = "—"
        x["Origen FIFO"] = orig.mask(mask_c, "Cripto")
    x["_yahoo"] = x.apply(
        lambda r: str(r.get("Ticker_Yahoo") or r.get("Ticker") or "").strip(),
        axis=1,
    )
    x["_fecha"] = x["Fecha origen (lote)"].map(_fifo_norm_fecha_hist)
    q = pd.to_numeric(x["Cantidad (tramo)"], errors="coerce").fillna(0.0)
    v = pd.to_numeric(x["Valor compra histórico (€)"], errors="coerce").fillna(0.0)
    x["_pm"] = (v / q.replace(0, np.nan)).fillna(0.0).round(8)
    return x


def fifo_tramos_ejercicio_totales_para_lotes_visibles(
    ledger_vis: pd.DataFrame,
    sales_detail_ejercicio: pd.DataFrame,
) -> dict[str, float | int | list[str]]:
    """
    Suma valor adquisición, transmisión y P&L de los tramos del ejercicio cuyo lote
    coincide con las filas visibles del ledger (mismas claves que build_fifo_lote_estado_ledger).
    Incluye fechas_venta (Y-m-d únicas ordenadas) para declarar día de transmisión.
    """
    empty = {
        "n_tramos": 0,
        "adquisicion": 0.0,
        "transmision": 0.0,
        "ganancia": 0.0,
        "perdida": 0.0,
        "neto": 0.0,
        "fechas_venta": [],
    }
    if ledger_vis is None or ledger_vis.empty:
        return empty
    if sales_detail_ejercicio is None or sales_detail_ejercicio.empty:
        return empty
    need = {"Origen FIFO", "Broker", "Yahoo/Ticker", "Fecha lote", "Precio medio €"}
    if not need.issubset(set(ledger_vis.columns)):
        return empty
    col_adq = "Valor compra histórico (€)"
    col_vta = "Valor venta (€)"
    col_pnl = "Plusvalía / Minusvalía (€)"
    if col_adq not in sales_detail_ejercicio.columns or col_vta not in sales_detail_ejercicio.columns:
        return empty

    kdf = pd.DataFrame(
        {
            "Origen FIFO": ledger_vis["Origen FIFO"].astype(str).str.strip(),
            "Broker": ledger_vis["Broker"].astype(str).str.strip(),
            "_yahoo": ledger_vis["Yahoo/Ticker"].astype(str).str.strip(),
            "_fecha": ledger_vis["Fecha lote"].map(_fifo_norm_fecha_hist),
            "_pm": pd.to_numeric(ledger_vis["Precio medio €"], errors="coerce").fillna(0.0).round(8),
        }
    ).drop_duplicates()

    dn = _fifo_sales_detail_norm_keys(sales_detail_ejercicio)
    merged = dn.merge(kdf, on=["Origen FIFO", "Broker", "_yahoo", "_fecha", "_pm"], how="inner")
    if merged.empty:
        return empty

    adq = pd.to_numeric(merged[col_adq], errors="coerce").fillna(0.0).sum()
    vta = pd.to_numeric(merged[col_vta], errors="coerce").fillna(0.0).sum()
    if col_pnl in merged.columns:
        pnl = pd.to_numeric(merged[col_pnl], errors="coerce").fillna(0.0)
        gan = float(pnl[pnl > 0].sum())
        per = float(pnl[pnl < 0].sum())
        net = float(pnl.sum())
    else:
        gan = max(0.0, float(vta - adq))
        per = min(0.0, float(vta - adq))
        net = float(vta - adq)
    fechas_venta: list[str] = []
    if "Fecha venta" in merged.columns:
        dtp = pd.to_datetime(merged["Fecha venta"], errors="coerce").dropna()
        if len(dtp):
            fechas_venta = sorted({pd.Timestamp(x).strftime("%Y-%m-%d") for x in dtp.unique()})
    return {
        "n_tramos": int(len(merged)),
        "adquisicion": float(adq),
        "transmision": float(vta),
        "ganancia": gan,
        "perdida": per,
        "neto": net,
        "fechas_venta": fechas_venta,
    }


def fifo_tramos_ejercicio_desglose_por_fecha_venta(
    ledger_vis: pd.DataFrame,
    sales_detail_ejercicio: pd.DataFrame,
) -> pd.DataFrame:
    """
    Por cada fecha de venta del ejercicio, totales de los tramos FIFO que enlazan con los lotes visibles.
    Útil cuando hubo varias ventas del mismo activo en el año.
    """
    empty_cols = [
        "Fecha venta",
        "Tramos",
        "Valor adquisición (€)",
        "Valor transmisión (€)",
        "Ganancia (€)",
        "Pérdida (€)",
        "Plusvalía neta (€)",
    ]
    if ledger_vis is None or ledger_vis.empty:
        return pd.DataFrame(columns=empty_cols)
    if sales_detail_ejercicio is None or sales_detail_ejercicio.empty:
        return pd.DataFrame(columns=empty_cols)
    need = {"Origen FIFO", "Broker", "Yahoo/Ticker", "Fecha lote", "Precio medio €"}
    if not need.issubset(set(ledger_vis.columns)):
        return pd.DataFrame(columns=empty_cols)
    col_adq = "Valor compra histórico (€)"
    col_vta = "Valor venta (€)"
    col_pnl = "Plusvalía / Minusvalía (€)"
    if col_adq not in sales_detail_ejercicio.columns or col_vta not in sales_detail_ejercicio.columns:
        return pd.DataFrame(columns=empty_cols)
    if "Fecha venta" not in sales_detail_ejercicio.columns:
        return pd.DataFrame(columns=empty_cols)

    kdf = pd.DataFrame(
        {
            "Origen FIFO": ledger_vis["Origen FIFO"].astype(str).str.strip(),
            "Broker": ledger_vis["Broker"].astype(str).str.strip(),
            "_yahoo": ledger_vis["Yahoo/Ticker"].astype(str).str.strip(),
            "_fecha": ledger_vis["Fecha lote"].map(_fifo_norm_fecha_hist),
            "_pm": pd.to_numeric(ledger_vis["Precio medio €"], errors="coerce").fillna(0.0).round(8),
        }
    ).drop_duplicates()

    dn = _fifo_sales_detail_norm_keys(sales_detail_ejercicio)
    merged = dn.merge(kdf, on=["Origen FIFO", "Broker", "_yahoo", "_fecha", "_pm"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=empty_cols)

    merged = merged.copy()
    merged["_fv_key"] = pd.to_datetime(merged["Fecha venta"], errors="coerce")
    rows: list[dict] = []
    for fv_ts, g in merged.groupby("_fv_key", dropna=False):
        if fv_ts is None or pd.isna(fv_ts):
            continue
        fv_s = pd.Timestamp(fv_ts).strftime("%Y-%m-%d")
        adq = float(pd.to_numeric(g[col_adq], errors="coerce").fillna(0.0).sum())
        vta = float(pd.to_numeric(g[col_vta], errors="coerce").fillna(0.0).sum())
        if col_pnl in g.columns:
            pnl = pd.to_numeric(g[col_pnl], errors="coerce").fillna(0.0)
            gan = float(pnl[pnl > 0].sum())
            per_neg = float(pnl[pnl < 0].sum())
            net = float(pnl.sum())
        else:
            net = vta - adq
            gan = max(0.0, net)
            per_neg = min(0.0, net)
        per_e = -per_neg if per_neg < 0 else 0.0
        rows.append(
            {
                "Fecha venta": fv_s,
                "Tramos": int(len(g)),
                "Valor adquisición (€)": adq,
                "Valor transmisión (€)": vta,
                "Ganancia (€)": gan,
                "Pérdida (€)": per_e,
                "Plusvalía neta (€)": net,
            }
        )
    if not rows:
        return pd.DataFrame(columns=empty_cols)
    out = pd.DataFrame(rows).sort_values("Fecha venta").reset_index(drop=True)
    return out


def main() -> None:
    st.title("Cartera de Inversión")

    df = load_data()

    # Menú izquierda: solo páginas
    vista = st.sidebar.radio(
        "Página",
        ["Cartera", "Movimientos", "Análisis", "Intereses extranjero", "Fiscalidad", "Brokers", "Catálogo"],
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
        new_path = st.text_input(
            "Ruta (ej. /share/cartera_final, /config)",
            value=current_override or str(_DATA_DIR),
            key="data_path_input",
        )
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
            "Los datos se guardan en la base SQLite (acciones.db). Exporta a CSV para respaldo; "
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
                    _orig_cat = str(row.get("Origen") or "")
                    _isin_lbl = "ISIN (obligatorio)" if _catalog_origen_requires_isin(_orig_cat) else "ISIN (opcional)"
                    ni = st.text_input(_isin_lbl, value=str(row.get("ISIN") or ""), key=f"cat_i_{sel}")
                if st.button("Guardar cambios", type="primary", key="cat_save"):
                    ni_st = (ni or "").strip()
                    ni_ok = _norm_isin_field(ni_st)
                    if _catalog_origen_requires_isin(_orig_cat) and not ni_ok:
                        if ni_st:
                            st.error("El ISIN no tiene un formato válido (12 caracteres alfanuméricos).")
                        else:
                            st.error("El ISIN es obligatorio para instrumentos de Acciones o Fondos en el catálogo.")
                    else:
                        ok, msg = apply_global_instrument_update(sel, ny.strip(), nt.strip(), nn.strip(), ni_st)
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

    if vista == "Análisis":
        st.header("Análisis")
        tab_evol, tab_dist, tab_gp, tab_div = st.tabs(
            ["Evolución mensual", "Distribución", "G/P realizadas", "Dividendos"]
        )
        with tab_evol:
            _render_analisis_evolucion_mensual()
        with tab_dist:
            render_analisis_distribucion(df)
        with tab_gp:
            render_analisis_gp_ventas()
        with tab_div:
            render_analisis_dividendos()
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

        # Posiciones vivas por cuenta (para «Vender todo» en nueva operación)
        _pos_nueva_op_acc = compute_positions_fifo(df)
        _pos_nueva_op_fon = positions_fondos_to_dataframe(compute_positions_fondos(df_fondos_mov))
        _pos_nueva_op_crip = compute_positions_criptos(df_crip_mov)

        tab_mov, tab_div = st.tabs(["Movimientos", "Dividendos"])

        with tab_mov:
            if "_flash_mov_eliminado" in st.session_state:
                st.success(st.session_state.pop("_flash_mov_eliminado"))
            with st.expander("➕ Nueva operación", expanded=False):
                # Paso 1: ¿Qué quieres registrar?
                tipo_registro = st.radio(
                    "¿Qué quieres registrar?",
                    ["Acciones/ETFs", "Opciones (Put/Call)", "Fondos", "Criptos", "Otros"],
                    index=0,
                    horizontal=True,
                    key="tipo_registro_nuevo",
                )
                if tipo_registro == "Acciones/ETFs":
                    position_type_base = "stock"
                    catalog_activo = catalog
                    tipo_nuevo = st.selectbox("Tipo de activo", ["Acción", "ETF"], key="new_tipo")
                    position_type = "stock" if tipo_nuevo == "Acción" else "etf"
                elif tipo_registro == "Opciones (Put/Call)":
                    position_type_base = "putOption"
                    catalog_activo = catalog
                    if catalog is not None and not catalog.empty and "positionType" in catalog.columns:
                        pt_cat = catalog["positionType"].astype(str).str.strip().str.lower()
                        catalog_activo = catalog[pt_cat.isin(["putoption", "calloption"])].copy()
                    else:
                        catalog_activo = pd.DataFrame()
                    position_type = "putOption"
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
                    op_options = [("buy", "Compra"), ("sell", "Venta"), ("split", "Split"), ("brokerTransfer", "Transferencia entre brokers")]
                elif tipo_registro == "Opciones (Put/Call)":
                    op_options = [
                        ("optionBuy", "Compra de prima"),
                        ("optionSell", "Venta de prima"),
                    ]
                elif tipo_registro == "Fondos":
                    op_options = [("buy", "Compra"), ("sell", "Venta"), ("traspaso_fondos", "Traspaso fondo")]
                else:
                    op_options = [("buy", "Compra"), ("sell", "Venta"), ("switch", "Permuta"), ("stakeReward", "Stake"), ("brokerTransfer", "Transferencia a wallet")]

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
                    # Paso 3: Posición (solo para compra, venta, split, stake reward; dividendos → pestaña Dividendos)
                    st.caption("Elige una posición existente o crea una nueva.")
                    pos_origen = st.radio(
                        "¿La posición ya existe?",
                        ["Sí, elegir de la lista", "No, es una posición nueva"],
                        index=0,
                        horizontal=True,
                        key="pos_existente_o_nueva",
                    )

                    position_currency = "EUR"
                    position_ticker = position_yahoo = position_name = position_exchange = position_country = position_isin = ""
                    if tipo_registro not in ("Acciones/ETFs", "Otros", "Opciones (Put/Call)"):
                        position_type = position_type_base

                    if pos_origen == "No, es una posición nueva":
                        nc1, nc2, nc3 = st.columns(3)
                        with nc1:
                            ticker_placeholder = (
                                "Ej: BTC, ETH…"
                                if tipo_registro == "Criptos"
                                else ("Ej. símbolo OCC o ticker de la opción" if tipo_registro == "Opciones (Put/Call)" else "AAPL, ES01234567890…")
                            )
                            position_ticker = st.text_input("Ticker", key="new_ticker", placeholder=ticker_placeholder)
                            if tipo_registro != "Criptos":
                                ph_yh = "Para cotizaciones; puede ser = ticker (vacío si no hay)"
                                if tipo_registro == "Opciones (Put/Call)":
                                    ph_yh = "Mismo que ticker si no hay cotización Yahoo"
                                position_yahoo = st.text_input("Ticker Yahoo", key="new_yahoo", placeholder=ph_yh)
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
                            elif tipo_registro == "Opciones (Put/Call)":
                                pc = st.radio("Put o Call", ["Put", "Call"], horizontal=True, key="opcion_pc_nuevo")
                                position_type = "putOption" if pc == "Put" else "callOption"
                                st.caption("Opción (primas; ejercicio ≠ compra/venta de acciones)")
                            else:
                                st.caption(f"Tipo: **{tipo_nuevo}**")
                            if tipo_registro != "Fondos":
                                position_exchange = st.text_input("Bolsa (opcional)", key="new_exchange", placeholder="XETRA, NASDAQ…")
                                position_country = st.text_input("País (opcional)", key="new_country", placeholder="DE, US…")
                        _isin_new_lbl = (
                            "ISIN (obligatorio, 12 caracteres)"
                            if tipo_registro in ("Acciones/ETFs", "Fondos")
                            else "ISIN (opcional)"
                        )
                        _isin_new_help = (
                            "Obligatorio para acciones, ETFs y fondos. Se guarda en **Instrumentos** con el ticker Yahoo."
                            if tipo_registro in ("Acciones/ETFs", "Fondos")
                            else "Para Renta / FIFO por ISIN. Si lo indicas, también se guarda en **Instrumentos**."
                        )
                        position_isin = st.text_input(
                            _isin_new_lbl,
                            key="new_isin",
                            placeholder="Ej. US0378331005",
                            help=_isin_new_help,
                        )
                    else:
                        ticker_options = ["—— Elige posición ——"]
                        option_to_catalog = []
                        if catalog_activo.empty and tipo_registro == "Criptos":
                            st.info("No hay criptos en cartera. Elige «posición nueva» para registrar tu primera operación.")
                        if catalog_activo.empty and tipo_registro == "Otros":
                            st.info("No hay posiciones de Otros en cartera. Elige «posición nueva» para registrar tu primera operación.")
                        if catalog_activo.empty and tipo_registro == "Opciones (Put/Call)":
                            st.info("No hay opciones en cartera. Elige «posición nueva» para registrar la primera.")
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
                                elif tipo_registro == "Opciones (Put/Call)":
                                    pt = str(r.get("positionType", "")).strip().lower()
                                    if pt == "putoption":
                                        lab += " [Put]"
                                    elif pt == "calloption":
                                        lab += " [Call]"
                                    else:
                                        lab += " [Opción]"
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

                    default_ccy_for_fees = (position_currency or "EUR").strip()
                    if default_ccy_for_fees not in currencies_in_data:
                        default_ccy_for_fees = "EUR" if "EUR" in currencies_in_data else currencies_in_data[0]
                    _ccy_sync_sig = (str(sel_pos), str(tipo_registro), default_ccy_for_fees)
                    if "last_pos_for_ccy" not in st.session_state or st.session_state["last_pos_for_ccy"] != _ccy_sync_sig:
                        st.session_state["ccy_com"] = default_ccy_for_fees
                        st.session_state["ccy_tax"] = default_ccy_for_fees
                        st.session_state["last_pos_for_ccy"] = _ccy_sync_sig
                else:
                    position_currency = "EUR"
                    position_ticker = position_yahoo = position_name = position_exchange = position_country = ""
                    if tipo_registro not in ("Acciones/ETFs", "Otros", "Opciones (Put/Call)"):
                        position_type = position_type_base
                    sel_pos = "——"
                    es_posicion_nueva = False

                # --- Formulario específico: Traspaso entre fondos (solo Fondos) ---
                if tipo_registro == "Fondos" and op_type == "traspaso_fondos":
                    st.caption(
                        "Genera dos movimientos en Fondos: salida del fondo origen y entrada en el fondo destino (coste arrastrado, no tributa). "
                        "Puedes indicar participaciones distintas en destino si el valor liquidativo no coincide entre fondos."
                    )
                    fondos_options = ["—— Elige fondo origen ——"]
                    fondos_dest_options = ["—— Elige fondo destino ——", "➕ Nuevo fondo destino"]
                    if not catalog_fondos.empty:
                        for _, r in catalog_fondos.iterrows():
                            lab = f"{r['ticker']} | {r['name']}"
                            fondos_options.append(lab)
                            fondos_dest_options.append(lab)
                    _tf_broker_ix = 0
                    if brokers_list:
                        for _i_tf, _nm_tf in enumerate(brokers_list):
                            if str(_nm_tf).strip().lower() == "myinvestor":
                                _tf_broker_ix = _i_tf
                                break
                    tf_c1, tf_c2 = st.columns(2)
                    with tf_c1:
                        tf_origen = st.selectbox("Fondo origen", fondos_options, key="tf_origen")
                        tf_broker = (
                            st.selectbox(
                                "Cuenta (broker)",
                                options=brokers_list,
                                index=_tf_broker_ix,
                                key="tf_broker",
                            )
                            if brokers_list
                            else st.text_input("Cuenta (broker)", key="tf_broker")
                        )
                        if "tf_qty_pending_fill" in st.session_state:
                            st.session_state["tf_qty"] = st.session_state.pop("tf_qty_pending_fill")
                        tf_qty = st.text_input(
                            "Participaciones (origen)",
                            placeholder="0 o 0,00",
                            key="tf_qty",
                            help="Participaciones que sales del fondo origen (reembolso).",
                        )
                        _br_tf = str(tf_broker or "").strip()
                        if tf_origen != "—— Elige fondo origen ——" and not catalog_fondos.empty:
                            _io_tf = fondos_options.index(tf_origen) - 1
                            if 0 <= _io_tf < len(catalog_fondos):
                                _ro_tf = catalog_fondos.iloc[_io_tf]
                                _yh_tf = str(_ro_tf.get("ticker_Yahoo") or _ro_tf.get("ticker") or "").strip()
                                _qtf = qty_en_cartera_broker_yahoo(
                                    "Fondos",
                                    _br_tf,
                                    _yh_tf,
                                    _pos_nueva_op_acc,
                                    _pos_nueva_op_fon,
                                    _pos_nueva_op_crip,
                                )
                                if _qtf > MIN_POSITION and _br_tf and _yh_tf:
                                    if st.button(
                                        "Traspasar todo",
                                        key="btn_traspaso_todo_tf",
                                        help=f"Rellenar participaciones con la posición en esta cuenta ({_qtf:g})",
                                    ):
                                        st.session_state["tf_qty_pending_fill"] = _format_qty_streamlit_form(_qtf)
                                        st.rerun()
                                elif _br_tf and _yh_tf:
                                    st.caption("Sin posición en cartera en esta cuenta para este fondo.")
                        tf_fecha = st.date_input("Fecha", key="tf_fecha")
                    with tf_c2:
                        tf_destino = st.selectbox("Fondo destino", fondos_dest_options, key="tf_destino")
                        tf_destino_nuevo_ticker = ""
                        tf_destino_nuevo_name = ""
                        if tf_destino == "➕ Nuevo fondo destino":
                            tf_destino_nuevo_ticker = st.text_input("ISIN / Ticker del fondo destino", key="tf_dest_ticker", placeholder="ES01234567890")
                            tf_destino_nuevo_name = st.text_input("Nombre del fondo destino", key="tf_dest_name", placeholder="Nombre del fondo")
                        tf_valor_eur = st.text_input("Valor reembolso (EUR)", placeholder="0 o 0,00", key="tf_valor_eur", help="Valor en euros del reembolso en el fondo origen")
                        tf_qty_dest = st.text_input(
                            "Participaciones en destino",
                            placeholder="Vacío = mismas que en origen",
                            key="tf_qty_dest",
                            help="Participaciones que recibes en el fondo destino (distinto VL u otro reparto). Si lo dejas vacío, se usa la misma cantidad que en origen.",
                        )
                        tf_hora = st.time_input("Hora", value=dt_time(12, 0), key="tf_hora", step=60)
                    if st.button("Guardar traspaso entre fondos", type="primary", key="guardar_traspaso_fondos"):
                        if tf_origen == "—— Elige fondo origen ——":
                            st.error("Elige el fondo origen.")
                        elif tf_destino == "—— Elige fondo destino ——":
                            st.error("Elige el fondo destino o «Nuevo fondo destino».")
                        elif tf_destino == "➕ Nuevo fondo destino" and (not tf_destino_nuevo_ticker or not tf_destino_nuevo_ticker.strip()):
                            st.error("Indica el ISIN o ticker del fondo destino.")
                        elif not tf_qty or _to_float(tf_qty, 0.0) <= 0:
                            st.error("Indica la cantidad de participaciones en origen.")
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
                                    yahoo_o = str(ro.get("ticker_Yahoo") or ticker_o).strip()
                                    isin_o = _norm_isin_field(ro.get("isin")) or _lookup_isin_for_ticker_yahoo(yahoo_o) or ""
                                    if tf_destino == "➕ Nuevo fondo destino":
                                        raw_d = (tf_destino_nuevo_ticker or "").strip()
                                        isin_d = _norm_isin_field(raw_d) or _lookup_isin_for_ticker_yahoo(raw_d) or ""
                                    else:
                                        yahoo_d = str(rd.get("ticker_Yahoo") or ticker_d).strip()
                                        isin_d = _norm_isin_field(rd.get("isin")) or _lookup_isin_for_ticker_yahoo(yahoo_d) or ""
                                    if not isin_o or not isin_d:
                                        st.error(
                                            "El **ISIN** es obligatorio en traspasos entre fondos. "
                                            "Complétalo en **Catálogo** o indica un ISIN/ticker válido en destino."
                                        )
                                    else:
                                        qty_orig = _to_float(tf_qty, 0.0)
                                        _raw_qty_dest = (tf_qty_dest or "").strip()
                                        qty_dest = _to_float(tf_qty_dest, 0.0) if _raw_qty_dest else qty_orig
                                        if _raw_qty_dest and qty_dest <= 0:
                                            st.error("Las participaciones en destino deben ser mayores que cero.")
                                        else:
                                            valor_eur = _to_float(tf_valor_eur, 0.0)
                                            date_str = tf_fecha.strftime("%Y-%m-%d") if hasattr(tf_fecha, "strftime") else str(tf_fecha)
                                            time_str = tf_hora.strftime("%H:%M:%S") if hasattr(tf_hora, "strftime") else "12:00:00"
                                            row_switch = {
                                                "date": date_str, "time": time_str,
                                                "ticker": ticker_o, "ticker_Yahoo": ro.get("ticker_Yahoo") or ticker_o, "isin": isin_o, "name": name_o,
                                                "positionType": "fund", "positionCountry": "", "positionCurrency": "EUR", "positionExchange": "",
                                                "broker": tf_broker, "type": "switch",
                                                "positionNumber": qty_orig, "price": valor_eur / qty_orig if qty_orig else 0,
                                                "comission": 0, "comissionCurrency": "EUR", "destinationRetentionBaseCurrency": "", "taxes": 0, "taxesCurrency": "EUR",
                                                "exchangeRate": 1.0, "positionQuantity": "", "autoFx": "No",
                                                "switchBuyPosition": ticker_d,
                                                "switchBuyPositionType": "", "switchBuyPositionNumber": "", "switchBuyExchangeRate": "", "switchBuyBroker": "",
                                                "spinOffBuyPosition": "", "spinOffBuyPositionNumber": "", "spinOffBuyPositionAllocation": "",
                                                "brokerTransferNewBroker": "",
                                                "total": valor_eur, "totalBaseCurrency": valor_eur, "totalWithComission": valor_eur, "totalWithComissionBaseCurrency": valor_eur,
                                            }
                                            yahoo_d_row = str(rd.get("ticker_Yahoo") or ticker_d).strip() if tf_destino != "➕ Nuevo fondo destino" else ticker_d
                                            row_switchbuy = {
                                                "date": date_str, "time": time_str,
                                                "ticker": ticker_d, "ticker_Yahoo": yahoo_d_row, "isin": isin_d, "name": name_d,
                                                "positionType": "fund", "positionCountry": "", "positionCurrency": "EUR", "positionExchange": "",
                                                "broker": tf_broker, "type": "switchBuy",
                                                "positionNumber": qty_dest, "price": valor_eur / qty_dest if qty_dest else 0,
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
                                                st.success("Traspaso entre fondos guardado.")
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
                                _iso_bt = _norm_isin_field(ro.get("isin")) or _lookup_isin_for_ticker_yahoo(ticker_yahoo) or ""
                                _need_bt_isin = tipo_registro == "Acciones/ETFs" and position_type in ("stock", "etf")
                                if _need_bt_isin and not _iso_bt:
                                    st.error(
                                        f"El **ISIN** es obligatorio para transferir acciones/ETF. "
                                        f"Añádelo en **Catálogo** para `{ticker_yahoo}`."
                                    )
                                else:
                                    row_bt = {
                                        "date": date_str, "time": time_str,
                                        "ticker": ticker_o, "ticker_Yahoo": ticker_yahoo, "isin": _iso_bt, "name": name_o,
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
                    _is_acc_split = op_type == "split" and tipo_registro in ("Acciones/ETFs", "Otros")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        op_date = st.date_input("Fecha", key="op_date_nuevo")
                    with c2:
                        op_time = st.time_input("Hora", value=dt_time(12, 0), key="op_time_nuevo", step=60, help="Hora de la operación (selección por minutos)")
                    with c3:
                        if _is_acc_split and brokers_list:
                            _split_all_lbl = "(Todas las cuentas — mismo ISIN)"
                            _sb = st.selectbox(
                                "Cuenta (broker)",
                                options=[_split_all_lbl] + list(brokers_list),
                                key="op_broker_nuevo",
                                help="**Todas las cuentas:** aplica el factor en cada bróker donde tengas este valor (misma divisa; en cartera, mismo ISIN y divisa). **Una cuenta:** solo ajusta esa posición. Fiscalidad: un solo saco por ISIN, el split afecta a todos los tramos.",
                            )
                            op_broker = "" if _sb == _split_all_lbl else str(_sb).strip()
                        elif _is_acc_split:
                            op_broker = st.text_input(
                                "Cuenta (broker, opcional)",
                                key="op_broker_nuevo",
                                placeholder="Vacío = todas las cuentas con este ISIN",
                                help="Sin lista de cuentas configurada: deja vacío para repartir el split en todas las posiciones con el mismo ISIN.",
                            )
                            op_broker = str(op_broker or "").strip()
                        else:
                            op_broker = (
                                st.selectbox("Cuenta (broker)", options=brokers_list, key="op_broker_nuevo")
                                if brokers_list
                                else st.text_input("Cuenta (broker)", key="op_broker_nuevo")
                            )

                    if _is_acc_split:
                        st.info(
                            "**Split bursátil:** indica solo el **factor** (p. ej. **2** en un 2:1). "
                            "Se multiplican los títulos de cada lote y se divide el precio medio; **no hay contraparte en efectivo**."
                        )
                        if "op_qty_nuevo_pending_fill" in st.session_state:
                            st.session_state["op_qty_nuevo"] = st.session_state.pop("op_qty_nuevo_pending_fill")
                        _qty_str = st.text_input(
                            "Factor del split",
                            placeholder="p. ej. 2",
                            key="op_qty_nuevo",
                            help="Por cada título **antes** del split, cuántos tienes **después**. No es «cuántas acciones tengo en total».",
                        )
                        op_quantity = _to_float(_qty_str, 0.0)
                        op_price = 0.0
                        op_total_local = 0.0
                        precio_o_total = "Precio unitario"
                        mod_fx = False
                        op_exchange_rate = 1.0
                        auto_fx = False
                        op_commission = 0.0
                        op_taxes = 0.0
                        op_commission_ccy = (position_currency or "EUR").strip() or "EUR"
                        op_taxes_ccy = op_commission_ccy
                        op_dest_ret = 0.0
                    else:
                        # Títulos y Precio/Total (text_input para que al hacer clic el cursor no quede en medio del número)
                        precio_o_total = st.radio("Introducir", ["Precio unitario", "Total"], horizontal=True, key="precio_o_total")
                        c_qty, c_val = st.columns(2)
                        with c_qty:
                            if "op_qty_nuevo_pending_fill" in st.session_state:
                                st.session_state["op_qty_nuevo"] = st.session_state.pop("op_qty_nuevo_pending_fill")
                            _qty_str = st.text_input("Títulos", placeholder="0 o 0,0000", key="op_qty_nuevo")
                            op_quantity = _to_float(_qty_str, 0.0)
                            if op_type in ("sell", "optionSell") and not es_posicion_nueva:
                                if sel_pos not in ("—— Elige posición ——", "➕ Nueva posición", "——"):
                                    _br_v = str(op_broker or "").strip() if op_broker is not None else ""
                                    _yh_v = str(position_yahoo or "").strip()
                                    _qc = qty_en_cartera_broker_yahoo(
                                        tipo_registro,
                                        _br_v,
                                        _yh_v,
                                        _pos_nueva_op_acc,
                                        _pos_nueva_op_fon,
                                        _pos_nueva_op_crip,
                                    )
                                    if _qc > MIN_POSITION and _br_v and _yh_v:
                                        if st.button(
                                            "Vender todo",
                                            key="btn_vender_todo_nuevo",
                                            help=f"Rellenar títulos con la posición en esta cuenta ({_qc:g})",
                                        ):
                                            st.session_state["op_qty_nuevo_pending_fill"] = _format_qty_streamlit_form(_qc)
                                            st.rerun()
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
                        mod_fx = st.toggle(
                            "Modificar tipo de cambio",
                            value=False,
                            key="mod_fx_nuevo",
                            help="Desactivado: tipo BCE (Frankfurter v1) para la fecha de la operación, retrocediendo a último día publicado si es festivo. Activado: indicar tipo a mano o usar los botones.",
                        )
                        op_exchange_rate = 1.0
                        if position_currency == "EUR":
                            op_exchange_rate = 1.0
                        else:
                            if not mod_fx:
                                op_exchange_rate = get_fx_rate_for_date(position_currency, op_date)
                                if math.isnan(op_exchange_rate) or op_exchange_rate <= 0:
                                    op_exchange_rate = 1.0
                                st.caption(
                                    f"Tipo BCE (Frankfurter), EUR por 1 {position_currency}: **{op_exchange_rate:.4f}**. "
                                    "Si no hubiera dato para ese día, se usa el último publicado. Activa el switch para otro valor o Yahoo (intradía)."
                                )
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
                                    if st.button(
                                        "Cierre del día",
                                        key="btn_fx_cierre",
                                        help="Tipo BCE (Frankfurter v1) para la fecha; si falla, Yahoo como respaldo.",
                                    ):
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
                        if _is_acc_split:
                            if es_posicion_nueva:
                                st.error(
                                    "El split aplica sobre una posición **ya existente**: elige «Sí, elegir de la lista» y la posición."
                                )
                            elif sel_pos in ("—— Elige posición ——", "——") or not sel_pos:
                                st.error("Elige una posición de la lista.")
                            elif op_quantity <= 0:
                                st.error("Indica un factor de split mayor que 0 (p. ej. 2 en un 2:1).")
                            else:
                                _ty_sp = (position_yahoo or position_ticker or "").strip()
                                _iso_sp = _resolve_movimiento_isin(
                                    "Acciones/ETFs",
                                    es_posicion_nueva,
                                    position_isin,
                                    _ty_sp,
                                    position_type,
                                )
                                if not _iso_sp:
                                    st.error(
                                        f"El **ISIN** es obligatorio para splits de acciones/ETF. "
                                        f"Añádelo en **Catálogo** para `{_ty_sp}`."
                                    )
                                else:
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
                                    new_row_split = {
                                        "date": date_str,
                                        "time": time_str,
                                        "ticker": position_ticker or position_yahoo,
                                        "ticker_Yahoo": position_yahoo or position_ticker,
                                        "isin": _iso_sp,
                                        "name": position_name or position_ticker,
                                        "positionType": position_type,
                                        "positionCountry": position_country or "",
                                        "positionCurrency": position_currency,
                                        "positionExchange": position_exchange or "",
                                        "broker": op_broker,
                                        "type": "split",
                                        "positionNumber": op_quantity,
                                        "price": 0.0,
                                        "comission": 0.0,
                                        "comissionCurrency": op_commission_ccy,
                                        "destinationRetentionBaseCurrency": "",
                                        "taxes": 0.0,
                                        "taxesCurrency": op_taxes_ccy,
                                        "exchangeRate": 1.0,
                                        "positionQuantity": "",
                                        "autoFx": "No",
                                        "switchBuyPosition": "",
                                        "switchBuyPositionType": "",
                                        "switchBuyPositionNumber": "",
                                        "switchBuyExchangeRate": "",
                                        "switchBuyBroker": "",
                                        "spinOffBuyPosition": "",
                                        "spinOffBuyPositionNumber": "",
                                        "spinOffBuyPositionAllocation": "",
                                        "brokerTransferNewBroker": "",
                                        "total": 0.0,
                                        "totalBaseCurrency": 0.0,
                                        "totalWithComission": 0.0,
                                        "totalWithComissionBaseCurrency": 0.0,
                                    }
                                    try:
                                        append_operation(new_row_split)
                                        load_data.clear()
                                        if "nuevo_form_abierto" in st.session_state:
                                            st.session_state["nuevo_form_abierto"] = False
                                        _clear_form_nueva_operacion()
                                        st.success("Split registrado.")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error al guardar: {e}")
                        elif not es_posicion_nueva and (not sel_pos or sel_pos == "—— Elige posición ——"):
                            st.error("Elige una posición de la lista.")
                        elif es_posicion_nueva and (not (position_ticker or position_yahoo)):
                            st.error("Indica al menos el ticker o el ticker Yahoo para la posición nueva.")
                        else:
                            iso_final = _resolve_movimiento_isin(
                                tipo_registro,
                                es_posicion_nueva,
                                position_isin,
                                (position_yahoo or position_ticker or "").strip(),
                                position_type,
                            )
                            need_isin = _isin_required_acciones_etf(tipo_registro, position_type) or _isin_required_fondos(
                                tipo_registro, position_type
                            )
                            if need_isin and not iso_final:
                                st.error(
                                    "El **ISIN** es obligatorio para acciones, ETFs y fondos. "
                                    "Indícalo en el formulario (posición nueva) o complétalo en **Catálogo** (posición existente)."
                                )
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
                                    "isin": iso_final,
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
                                    yk = (position_yahoo or position_ticker or "").strip()
                                    if yk and iso_final:
                                        _init_instrument_catalog()
                                        with _get_db() as conn:
                                            conn.execute(
                                                "DELETE FROM instrument_catalog WHERE ticker_Yahoo = ?",
                                                (yk,),
                                            )
                                            conn.execute(
                                                "INSERT INTO instrument_catalog (ticker_Yahoo, isin) VALUES (?, ?)",
                                                (yk, iso_final),
                                            )
                                            conn.commit()
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
                _sm = ["datetime_full"]
                _am = [False]
                if "type" in mov.columns and "origen" in mov.columns:
                    mov["_tie_mov"] = 0
                    _m = mov["origen"].astype(str).eq("Criptos")
                    mov.loc[_m, "_tie_mov"] = _cripto_movimiento_tab_type_order(mov.loc[_m, "type"])
                    _sm.append("_tie_mov")
                    _am.append(True)
                if "_rowid_" in mov.columns:
                    _sm.append("_rowid_")
                    _am.append(True)
                mov = mov.sort_values(_sm, ascending=_am, kind="mergesort").drop(columns=["_tie_mov"], errors="ignore")

            col_filtro, col_refresh = st.columns([4, 1])
            with col_filtro:
                filtro_origen = st.radio(
                    "Origen",
                    ["Todos", "Acciones", "ETFs", "Otros", "Puts", "Calls", "Fondos", "Criptos"],
                    index=0,
                    horizontal=True,
                )
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
            elif filtro_origen == "Puts":
                pt = mov["positionType"].astype(str).str.strip().str.lower() if "positionType" in mov.columns else pd.Series([""] * len(mov))
                mov = mov[(mov["origen"] == "Acciones") & (pt == "putoption")].copy()
            elif filtro_origen == "Calls":
                pt = mov["positionType"].astype(str).str.strip().str.lower() if "positionType" in mov.columns else pd.Series([""] * len(mov))
                mov = mov[(mov["origen"] == "Acciones") & (pt == "calloption")].copy()
            elif filtro_origen == "Fondos":
                mov = mov[mov["origen"] == "Fondos"].copy()
            elif filtro_origen == "Criptos":
                mov = mov[mov["origen"] == "Criptos"].copy()

            cat_isin_mov: dict[str, str] = {}

            def _mov_row_isin(row) -> str:
                ty = str(row.get("ticker_Yahoo") or row.get("ticker") or "").strip()
                to = str(row.get("ticker") or "").strip()
                return _fifo_resolve_isin_row(row, ty, to, cat_isin_mov)

            if mov.empty:
                mov["_isin_disp"] = pd.Series(dtype=str)
            else:
                mov["_isin_disp"] = mov.apply(_mov_row_isin, axis=1)

            tipos_unicos = (
                sorted(mov["type"].dropna().astype(str).str.strip().unique().tolist())
                if "type" in mov.columns and not mov.empty
                else []
            )

            def _lbl_tipo_op(t: str) -> str:
                x = str(t).strip().lower()
                if filtro_origen == "Fondos" and x == "switch":
                    return "Traspaso salida"
                if filtro_origen == "Fondos" and x == "switchbuy":
                    return "Traspaso entrada"
                m = {
                    "buy": "Compra",
                    "sell": "Venta",
                    "switch": "Permuta (salida)",
                    "switchbuy": "Permuta (entrada)",
                    "dividend": "Dividendo",
                    "split": "Split",
                    "brokertransfer": "Transferencia bróker",
                    "traspaso_fondos": "Traspaso fondos",
                    "deposit": "Depósito",
                    "withdrawal": "Retirada",
                    "bonus": "Bonus",
                    "commission": "Comisión",
                    "stakereward": "Stake",
                    "optionbuy": "Compra prima (opción)",
                    "optionsell": "Venta prima (opción)",
                }
                return m.get(x, str(t))

            with st.expander("Filtros de búsqueda", expanded=False):
                fx1, fx2, fx3 = st.columns(3)
                with fx1:
                    q_tick = st.text_input(
                        "Ticker / Yahoo",
                        key="mov_f_ticker",
                        placeholder="Contiene… (vacío = todos)",
                        help="Subcadena en ticker o ticker Yahoo (sin distinguir mayúsculas).",
                    )
                with fx2:
                    q_nom = st.text_input(
                        "Nombre posición",
                        key="mov_f_nombre",
                        placeholder="Contiene… (vacío = todos)",
                        help="Subcadena en el nombre del activo.",
                    )
                with fx3:
                    q_isin = st.text_input(
                        "ISIN",
                        key="mov_f_isin",
                        placeholder="Contiene… (vacío = todos)",
                        help="Subcadena en el ISIN (movimiento o catálogo). Cripto suele quedar vacío.",
                    )

                ft1, ft2 = st.columns(2)
                with ft1:
                    sel_tipos = st.multiselect(
                        "Tipo de operación",
                        options=tipos_unicos,
                        default=[],
                        key="mov_f_tipos",
                        format_func=_lbl_tipo_op,
                        help="Sin selección = todas las operaciones.",
                    )
                with ft2:
                    sel_pt: list[str] = []
                    if filtro_origen == "Todos" and "positionType" in mov.columns and not mov.empty:
                        pt_vals = mov["positionType"].dropna().astype(str).str.strip().str.lower().unique().tolist()
                        PT_ORDER = ["stock", "etf", "fund", "crypto", "warrant", "putoption", "calloption"]
                        pt_opts = [p for p in PT_ORDER if p in pt_vals]
                        for p in sorted(pt_vals):
                            if p not in pt_opts:
                                pt_opts.append(p)
                        sel_pt = st.multiselect(
                            "Tipo de activo",
                            options=pt_opts,
                            default=[],
                            key="mov_f_pt",
                            format_func=lambda p: {
                                "stock": "Acción",
                                "etf": "ETF",
                                "fund": "Fondo",
                                "crypto": "Cripto",
                                "warrant": "Otros",
                                "putoption": "Put",
                                "calloption": "Call",
                            }.get(str(p).lower(), p),
                            help="Solo con Origen «Todos». Vacío = todos.",
                        )
                    else:
                        st.caption("Tipo de activo: usa el selector **Origen** arriba (Acciones, ETFs, …).")

                fd_a, fd_b = st.columns(2)
                with fd_a:
                    d_desde_pick = st.date_input(
                        "Fecha desde",
                        value=None,
                        format="YYYY-MM-DD",
                        key="mov_cal_desde",
                        help="Calendario como en nueva operación. Sin fecha seleccionada = no acota por el inicio.",
                    )
                with fd_b:
                    d_hasta_pick = st.date_input(
                        "Fecha hasta",
                        value=None,
                        format="YYYY-MM-DD",
                        key="mov_cal_hasta",
                        help="Sin fecha seleccionada = no acota por el final.",
                    )

                st.caption("Rangos numéricos (opcional)")
                c_cant, c_tot = st.columns(2)
                with c_cant:
                    min_cant = st.number_input("Cantidad mín.", value=None, placeholder="Sin mínimo", key="filtro_min_cant")
                    max_cant = st.number_input("Cantidad máx.", value=None, placeholder="Sin máximo", key="filtro_max_cant")
                with c_tot:
                    min_tot = st.number_input("Total (€) mín.", value=None, placeholder="Sin mínimo", key="filtro_min_tot")
                    max_tot = st.number_input("Total (€) máx.", value=None, placeholder="Sin máximo", key="filtro_max_tot")

            if sel_tipos and "type" in mov.columns:
                mov = mov[mov["type"].astype(str).str.strip().isin(sel_tipos)].copy()
            if sel_pt and "positionType" in mov.columns:
                mov = mov[mov["positionType"].astype(str).str.strip().str.lower().isin(sel_pt)].copy()
            if (q_tick or "").strip() and not mov.empty:
                qt = (q_tick or "").strip().lower()
                tk = mov["ticker"].astype(str).str.lower() if "ticker" in mov.columns else pd.Series([""] * len(mov))
                tyk = mov["ticker_Yahoo"].astype(str).str.lower() if "ticker_Yahoo" in mov.columns else pd.Series([""] * len(mov))
                mov = mov[tk.str.contains(qt, regex=False, na=False) | tyk.str.contains(qt, regex=False, na=False)].copy()
            if (q_nom or "").strip() and "name" in mov.columns and not mov.empty:
                qn = (q_nom or "").strip().lower()
                mov = mov[mov["name"].astype(str).str.lower().str.contains(qn, regex=False, na=False)].copy()
            if (q_isin or "").strip() and not mov.empty:
                qi = (q_isin or "").strip().lower()
                iscol = mov["_isin_disp"].fillna("").astype(str).str.lower()
                mov = mov[iscol.str.contains(qi, regex=False, na=False)].copy()
            if "date" in mov.columns and not mov.empty:
                d_mov = mov["date"].astype(str).str.strip().str[:10]
                if d_desde_pick is not None:
                    ds = pd.Timestamp(d_desde_pick).strftime("%Y-%m-%d")
                    mov = mov[d_mov >= ds].copy()
                if d_hasta_pick is not None:
                    hs = pd.Timestamp(d_hasta_pick).strftime("%Y-%m-%d")
                    mov = mov[d_mov <= hs].copy()

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

            st.caption(f"**{len(mov)}** movimiento(s) con los filtros actuales.")

            # Primera columna: tipo de operación con símbolo (y luego color)
            def tipo_simbolo(t, origen: str | None = None):
                if pd.isna(t):
                    return "•"
                t = str(t).strip().lower()
                o = str(origen or "").strip()
                if o == "Fondos" and t == "switch":
                    return "Traspaso salida"
                if o == "Fondos" and t == "switchbuy":
                    return "Traspaso entrada"
                if t == "stakereward":
                    return "▲ Stake"
                if t in ("buy", "switchbuy", "deposit", "bonus", "optionbuy"):
                    return "▲ Compra" if t != "optionbuy" else "▲ Prima (opción)"
                if t == "commission":
                    # Traspaso cripto: fee en origen (no es venta al mercado); se mostraba como «Venta» solo por el estilo de salida.
                    return "▼ Comisión"
                if t in ("sell", "switch", "withdrawal", "optionsell"):
                    return "▼ Venta" if t != "optionsell" else "▼ Prima (opción)"
                if t == "split":
                    return "⇄ Split"
                if t == "brokertransfer":
                    return "⇄ Traspaso"
                if t == "dividend":
                    return "💰 Div"
                return f"• {t}"

            if "type" in mov.columns:
                if "origen" in mov.columns:
                    mov["Tipo"] = mov.apply(lambda r: tipo_simbolo(r.get("type"), r.get("origen")), axis=1)
                else:
                    mov["Tipo"] = mov["type"].apply(tipo_simbolo)
            else:
                mov["Tipo"] = "•"

            col_map = {
                "datetime_full": "Fecha",
                "origen": "Origen",
                "broker": "Cuenta",
                "name": "Posición",
                "ticker": "Ticker",
                "ticker_Yahoo": "Ticker Yahoo",
                "_isin_disp": "ISIN",
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
                "Ticker",
                "Ticker Yahoo",
                "ISIN",
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
                if "Stake" in v:
                    return "color: #26a69a; font-weight: 500;"  # teal, distinto de compra
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
                    tabla_por_filtro = {
                        "Acciones": "movimientos",
                        "ETFs": "movimientos",
                        "Otros": "movimientos",
                        "Puts": "movimientos",
                        "Calls": "movimientos",
                        "Fondos": "movimientos_fondos",
                        "Criptos": "movimientos_criptos",
                    }
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
                    _style_map(display_mov[cols_presentes].style, color_tipo, subset=["Tipo"]),
                    use_container_width=True,
                )

            # Eliminar operaciones: tabla movimientos (acciones/ETFs/etc.) y movimientos_criptos
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
                    "Selecciona operaciones de Acciones (incl. ETFs, opciones…), Fondos y/o Criptos.",
                    opciones,
                    format_func=_etiqueta_fila,
                    key="eliminar_operaciones",
                )
                if eliminar and st.button("Eliminar seleccionadas", type="primary", key="btn_eliminar"):
                    acc_indices_to_drop: list[int] = []
                    cripto_rowids: list[int] = []
                    fondos_rowids: list[int] = []
                    for pos in eliminar:
                        if pos < 0 or pos >= len(mov):
                            continue
                        row = mov.iloc[pos]
                        orig = str(row.get("origen", "")).strip()
                        if orig == "Acciones":
                            idx = row.get("_acc_idx")
                            if pd.notna(idx) and idx is not None:
                                acc_indices_to_drop.append(int(idx))
                        elif orig == "Criptos":
                            rid = row.get("_rowid_")
                            if pd.notna(rid) and rid is not None and str(rid).strip() != "":
                                try:
                                    cripto_rowids.append(int(rid))
                                except (TypeError, ValueError):
                                    pass
                        elif orig == "Fondos":
                            rid = row.get("_rowid_")
                            if pd.notna(rid) and rid is not None and str(rid).strip() != "":
                                try:
                                    fondos_rowids.append(int(rid))
                                except (TypeError, ValueError):
                                    pass

                    cripto_rowids = list(dict.fromkeys(cripto_rowids))
                    fondos_rowids = list(dict.fromkeys(fondos_rowids))
                    if not acc_indices_to_drop and not cripto_rowids and not fondos_rowids:
                        st.warning(
                            "Ninguna fila aplicable. Comprueba que los movimientos tengan identificador interno (recarga con «Refrescar datos» si hace falta)."
                        )
                    else:
                        try:
                            if acc_indices_to_drop:
                                df_sin = df.drop(index=acc_indices_to_drop)
                                cols_csv = [c for c in MOVIMIENTOS_COLUMNS if c in df_sin.columns]
                                write_full_db(df_sin[cols_csv])
                                load_data.clear()
                            if cripto_rowids:
                                with _get_db() as conn:
                                    ph = ",".join("?" * len(cripto_rowids))
                                    conn.execute(
                                        f"DELETE FROM movimientos_criptos WHERE rowid IN ({ph})",
                                        cripto_rowids,
                                    )
                                    conn.commit()
                                load_data_criptos.clear()
                            if fondos_rowids:
                                with _get_db() as conn:
                                    ph = ",".join("?" * len(fondos_rowids))
                                    conn.execute(
                                        f"DELETE FROM movimientos_fondos WHERE rowid IN ({ph})",
                                        fondos_rowids,
                                    )
                                    conn.commit()
                                load_data_fondos.clear()
                            n_a, n_c, n_f = len(acc_indices_to_drop), len(cripto_rowids), len(fondos_rowids)
                            total = n_a + n_c + n_f
                            if total == 1:
                                _msg = "Se eliminó 1 operación correctamente."
                            else:
                                _msg = f"Se eliminaron {total} operaciones correctamente."
                            _parts = []
                            if n_a:
                                _parts.append(f"Acciones: {n_a}")
                            if n_c:
                                _parts.append(f"Criptos: {n_c}")
                            if n_f:
                                _parts.append(f"Fondos: {n_f}")
                            if _parts:
                                _msg += " (" + ", ".join(_parts) + ")"
                            st.session_state["_flash_mov_eliminado"] = _msg
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error al eliminar: {e}")

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
                    _div_currencies = ["EUR", "USD", "GBP", "AUD", "CAD", "CHF", "DKK", "HKD", "JPY", "NZD"]
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
                        mod_fx_div = st.toggle(
                            "Modificar tipo de cambio",
                            value=False,
                            key="div_mod_fx",
                            help="Desactivado: tipo BCE (Frankfurter v1) para la fecha de pago. Activado: manual o botones.",
                        )
                        if not mod_fx_div:
                            div_exchange_rate = get_fx_rate_for_date(div_ccy_sel, div_date)
                            if math.isnan(div_exchange_rate) or div_exchange_rate <= 0:
                                div_exchange_rate = 1.0
                            st.caption(
                                f"Tipo BCE (Frankfurter), EUR por 1 {div_ccy_sel}: **{div_exchange_rate:.4f}**. "
                                "Si no hubiera dato para ese día, se usa el último publicado."
                            )
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
                                if st.button(
                                    "Cierre del día",
                                    key="btn_div_fx_cierre",
                                    help="Tipo BCE (Frankfurter v1) para la fecha de pago; si falla, Yahoo.",
                                ):
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
                        ro_mode = st.radio(
                            "Cómo indicar la retención en origen",
                            [f"Importe ({div_ccy_sel})", "% del bruto"],
                            horizontal=True,
                            key="div_ret_origen_mode",
                        )
                        if ro_mode.startswith("Importe"):
                            _ret_origen_label = f"Retención en origen ({div_ccy_sel})" if div_ccy_sel != "EUR" else "Retención en origen (EUR)"
                            div_origin_ret_ccy = _to_float(
                                st.text_input(_ret_origen_label, placeholder="0,00", key="div_origin_amt"),
                                0.0,
                            )
                        else:
                            _pct_ro = _to_float(
                                st.text_input(
                                    f"% sobre el bruto del dividendo ({div_ccy_sel})",
                                    placeholder="15",
                                    key="div_origin_pct",
                                    help="Se aplica sobre el total bruto en la divisa del cupón.",
                                ),
                                0.0,
                            )
                            div_origin_ret_ccy = div_total_val * (_pct_ro / 100.0)
                        div_origin_ret_eur = div_origin_ret_ccy if div_ccy_sel == "EUR" else div_origin_ret_ccy * div_exchange_rate
                    mod_ret_destino = st.toggle("Modificar la retención en destino", value=False, key="div_mod_ret_destino", help="Si la cuenta tiene 'Retiene en destino' activado (ficha de cuenta), suele ser Sí.")
                    div_dest_ret_eur = 0.0
                    if mod_ret_destino:
                        div_dest_ret_eur = _to_float(st.text_input("Retención en destino (EUR)", placeholder="0,00", key="div_dest_ret_eur"), 0.0)
                    mod_pct_recup = st.toggle(
                        "Modificar % recuperable por doble imposición (tope imputable)",
                        value=False,
                        key="div_mod_pct_recup",
                        help="Desactivado: tope **15%** del bruto (típico convenio). Activado: indica otro % máximo imputable sobre el bruto.",
                    )
                    div_pct_recup = 15.0
                    if mod_pct_recup:
                        div_pct_recup = _to_float(
                            st.text_input("Porcentaje (%)", value="15", placeholder="15", key="div_pct_recup"),
                            15.0,
                        )
                    _pct_tope_conv = (div_pct_recup / 100.0) if mod_pct_recup else 0.15
                    _pct_tope_label = div_pct_recup if mod_pct_recup else 15.0
                    _credito_imputable_eur = (
                        min(div_origin_ret_eur, div_total_base_val * _pct_tope_conv)
                        if div_origin_ret_eur > 0
                        else 0.0
                    )
                    _credito_imputable_ccy = (
                        min(div_origin_ret_ccy, div_total_val * _pct_tope_conv)
                        if div_origin_ret_ccy > 0
                        else 0.0
                    )
                    st.caption(
                        f"**Doble imposición:** con retención en origen, «Impuesto satisf. en el extranjero» en CSV = importe **imputable** "
                        f"(mínimo entre retención real y **{_pct_tope_label:g}%** del bruto). "
                        f"{'El % lo defines abajo.' if mod_pct_recup else 'Interruptor desactivado → se usa **15%** del bruto.'} "
                        "La retención **real** del bróker sigue en «Retención en origen»."
                    )
                    st.caption(
                        "«Retención a realizar o devolver» = 19% bruto € − crédito imputable − retención destino €; "
                        "«Total neto con devolución» = total neto cobrado € − esa cantidad."
                    )

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
                        elif div_type in ("stock", "etf") and not (
                            _lookup_isin_for_ticker_yahoo((div_yahoo or "").strip()) or ""
                        ):
                            st.error(
                                f"El **ISIN** es obligatorio para dividendos de acciones/ETF. "
                                f"Complétalo en **Catálogo** para `{div_yahoo}`."
                            )
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
                            _filio_dest_teor_eur = 0.19 * total_bruto_eur - _credito_imputable_eur - div_dest_ret_eur
                            _filio_neto_con_dev_eur = total_neto_eur - _filio_dest_teor_eur
                            _ret_loss_eur = max(0.0, div_origin_ret_eur - _credito_imputable_eur)
                            row_div = {
                                "type": "stockDividend",
                                "date": date_str,
                                "time": time_str,
                                "ticker": div_ticker,
                                "ticker_Yahoo": div_yahoo,
                                "isin": (
                                    _lookup_isin_for_ticker_yahoo((div_yahoo or "").strip()) or ""
                                    if div_type in ("stock", "etf")
                                    else ""
                                ),
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
                                "retentionReturned": _credito_imputable_ccy,
                                "retentionReturnedBaseCurrency": _credito_imputable_eur,
                                "unrealizedDestinationRetentionBaseCurrency": _filio_dest_teor_eur,
                                "netoWithReturnBaseCurrency": _filio_neto_con_dev_eur,
                                "originRetentionLossBaseCurrency": _ret_loss_eur,
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

                    def _lbl_div_type_filt(t: str) -> str:
                        x = str(t).strip().lower()
                        m = {
                            "stockdividend": "Dividendo acción/ETF",
                            "cashdividend": "Dividendo (efectivo)",
                        }
                        return m.get(x, str(t))

                    with st.expander("Filtros de búsqueda", expanded=False):
                        fx1, fx2, fx3 = st.columns(3)
                        with fx1:
                            div_q_ticker = st.text_input(
                                "Ticker / Yahoo",
                                key="div_f_ticker",
                                placeholder="Contiene…",
                                help="Subcadena en ticker o ticker Yahoo (sin distinguir mayúsculas).",
                            )
                        with fx2:
                            div_q_nom = st.text_input(
                                "Nombre posición",
                                key="div_f_nombre",
                                placeholder="Contiene…",
                                help="Subcadena en el nombre del instrumento.",
                            )
                        with fx3:
                            div_q_isin = st.text_input(
                                "ISIN",
                                key="div_f_isin",
                                placeholder="Contiene…",
                            )
                        ft1, ft2 = st.columns(2)
                        with ft1:
                            _brokers_div = (
                                sorted(div_df["broker"].dropna().astype(str).str.strip().unique().tolist())
                                if "broker" in div_df.columns
                                else []
                            )
                            div_sel_brokers = st.multiselect(
                                "Cuenta (broker)",
                                options=_brokers_div,
                                default=[],
                                key="div_f_brokers",
                                help="Vacío = todas las cuentas.",
                            )
                        with ft2:
                            _tipos_div = (
                                sorted(div_df["type"].dropna().astype(str).str.strip().unique().tolist())
                                if "type" in div_df.columns
                                else []
                            )
                            div_sel_tipos = st.multiselect(
                                "Tipo de dividendo",
                                options=_tipos_div,
                                default=[],
                                key="div_f_tipos",
                                format_func=_lbl_div_type_filt,
                            )
                        fd1, fd2, fd3, fd4 = st.columns(4)
                        with fd1:
                            div_d_desde = st.date_input(
                                "Fecha desde",
                                value=None,
                                format="YYYY-MM-DD",
                                key="div_cal_desde",
                                help="Sin fecha = sin límite inferior.",
                            )
                        with fd2:
                            div_d_hasta = st.date_input(
                                "Fecha hasta",
                                value=None,
                                format="YYYY-MM-DD",
                                key="div_cal_hasta",
                                help="Sin fecha = sin límite superior.",
                            )
                        with fd3:
                            div_min_bruto = st.text_input(
                                "Total bruto (€) mín.",
                                key="div_f_min_bruto",
                                placeholder="vacío",
                            )
                        with fd4:
                            div_max_bruto = st.text_input(
                                "Total bruto (€) máx.",
                                key="div_f_max_bruto",
                                placeholder="vacío",
                            )
                        div_q_desc = st.text_input(
                            "Notas (descripción)",
                            key="div_f_desc",
                            placeholder="Contiene…",
                            help="Busca en el campo descripción / notas.",
                        )

                    div_f = div_df.copy()
                    qtk = (div_q_ticker or "").strip().lower()
                    if qtk:
                        ty = (
                            div_f["ticker_Yahoo"].astype(str).str.lower().str.contains(qtk, regex=False)
                            if "ticker_Yahoo" in div_f.columns
                            else pd.Series(False, index=div_f.index)
                        )
                        tk = div_f["ticker"].astype(str).str.lower().str.contains(qtk, regex=False) if "ticker" in div_f.columns else pd.Series(False, index=div_f.index)
                        div_f = div_f[ty | tk].copy()
                    qn = (div_q_nom or "").strip().lower()
                    if qn and "nombre" in div_f.columns:
                        div_f = div_f[div_f["nombre"].astype(str).str.lower().str.contains(qn, regex=False)].copy()
                    qi = (div_q_isin or "").strip().lower()
                    if qi and "isin" in div_f.columns:
                        div_f = div_f[div_f["isin"].astype(str).str.lower().str.contains(qi, regex=False)].copy()
                    if div_sel_brokers and "broker" in div_f.columns:
                        div_f = div_f[div_f["broker"].astype(str).str.strip().isin(div_sel_brokers)].copy()
                    if div_sel_tipos and "type" in div_f.columns:
                        div_f = div_f[div_f["type"].astype(str).str.strip().isin(div_sel_tipos)].copy()
                    if "date" in div_f.columns:
                        d_ser = div_f["date"].astype(str).str.strip().str[:10]
                        if div_d_desde is not None:
                            ds = pd.Timestamp(div_d_desde).strftime("%Y-%m-%d")
                            div_f = div_f[d_ser >= ds].copy()
                            d_ser = div_f["date"].astype(str).str.strip().str[:10]
                        if div_d_hasta is not None:
                            hs = pd.Timestamp(div_d_hasta).strftime("%Y-%m-%d")
                            div_f = div_f[d_ser <= hs].copy()
                    min_b = _to_float((div_min_bruto or "").strip().replace(",", "."), None) if (div_min_bruto or "").strip() else None
                    max_b = _to_float((div_max_bruto or "").strip().replace(",", "."), None) if (div_max_bruto or "").strip() else None
                    if min_b is not None and "totalBaseCurrency" in div_f.columns:
                        div_f = div_f[pd.to_numeric(div_f["totalBaseCurrency"].astype(str).str.replace(",", ".", regex=False), errors="coerce").fillna(0.0) >= min_b].copy()
                    if max_b is not None and "totalBaseCurrency" in div_f.columns:
                        div_f = div_f[pd.to_numeric(div_f["totalBaseCurrency"].astype(str).str.replace(",", ".", regex=False), errors="coerce").fillna(0.0) <= max_b].copy()
                    qd = (div_q_desc or "").strip().lower()
                    if qd and "description" in div_f.columns:
                        div_f = div_f[div_f["description"].astype(str).str.lower().str.contains(qd, regex=False)].copy()

                    st.caption(f"**{len(div_f)}** dividendo(s) con los filtros actuales (de **{len(div_df)}** en total).")

                    _meses = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
                    def _to_float_div(x, default=0.0):
                        if x is None or (isinstance(x, float) and pd.isna(x)) or str(x).strip() == "":
                            return default
                        try:
                            return float(str(x).replace(",", ".").strip())
                        except (ValueError, TypeError):
                            return default
                    if div_f.empty:
                        st.info("Ningún dividendo coincide con los filtros. Ajusta o vacía los criterios.")
                    else:
                        show_div = pd.DataFrame()
                        show_div["Fecha"] = div_f["date"].astype(str).apply(
                            lambda s: (lambda d: f"{d.day:02d} {_meses[d.month - 1]} {d.year}" if pd.notna(d) else s)(pd.to_datetime(s.split("T")[0] if "T" in s else s, errors="coerce"))
                        )
                        show_div["Cuenta"] = div_f["broker"].astype(str)
                        show_div["Posición"] = div_f["ticker"].astype(str)
                        show_div["Títulos / Particip."] = div_f["positionNumber"].astype(str).str.replace(".", ",", regex=False)
                        div_ccy = div_f.get("currency", div_f.get("positionCurrency", "EUR")).astype(str)
                        show_div["Cantidad por título/particip."] = [
                            _fmt_div_currency(_to_float_div(div_f["quantity"].iloc[i]), div_ccy.iloc[i] if i < len(div_ccy) else "EUR")
                            for i in range(len(div_f))
                        ]
                        show_div["Total bruto"] = [
                            _fmt_div_currency(_to_float_div(div_f["total"].iloc[i]), div_ccy.iloc[i] if i < len(div_ccy) else "EUR")
                            for i in range(len(div_f))
                        ]
                        show_div["Tipo de cambio"] = div_f["exchangeRate"].astype(str).str.replace(".", ",", regex=False)
                        show_div["Total bruto (€)"] = [_fmt_div_currency(_to_float_div(div_f["totalBaseCurrency"].iloc[i]), "EUR") for i in range(len(div_f))]
                        show_div["Retención en origen"] = [
                            _fmt_div_currency(_to_float_div(div_f["originRetention"].iloc[i]), div_ccy.iloc[i] if i < len(div_ccy) else "EUR")
                            for i in range(len(div_f))
                        ]
                        show_div["Total bruto después de origen (€)"] = [_fmt_div_currency(_to_float_div(div_f["netoBaseCurrency"].iloc[i]), "EUR") for i in range(len(div_f))]
                        show_div["Retención en dest. realizada"] = [_fmt_div_currency(_to_float_div(div_f["destinationRetentionBaseCurrency"].iloc[i]), "EUR") for i in range(len(div_f))]
                        show_div["Comisión"] = div_f.get("comission", pd.Series([""] * len(div_f))).astype(str).str.replace(".", ",", regex=False)
                        show_div["Comisión (€)"] = [_fmt_div_currency(_to_float_div(div_f["comissionBaseCurrency"].iloc[i]) if "comissionBaseCurrency" in div_f.columns else 0, "EUR") for i in range(len(div_f))]
                        show_div["Total neto cobrado (€)"] = [_fmt_div_currency(_to_float_div(div_f["totalNetoBaseCurrency"].iloc[i]), "EUR") for i in range(len(div_f))]
                        show_div["Impuesto satisf. en el extranjero"] = [
                            _fmt_div_currency(_to_float_div(div_f["retentionReturned"].iloc[i]), div_ccy.iloc[i] if i < len(div_ccy) else "EUR")
                            for i in range(len(div_f))
                        ]
                        show_div["Impuesto satisf. en el extranjero (€)"] = [_fmt_div_currency(_to_float_div(div_f["retentionReturnedBaseCurrency"].iloc[i]), "EUR") for i in range(len(div_f))]
                        show_div["Retención a realizar o devolver (€)"] = [_fmt_div_currency(_to_float_div(div_f["unrealizedDestinationRetentionBaseCurrency"].iloc[i]) if "unrealizedDestinationRetentionBaseCurrency" in div_f.columns else 0, "EUR") for i in range(len(div_f))]
                        show_div["Total neto con devolución (€)"] = [_fmt_div_currency(_to_float_div(div_f["netoWithReturnBaseCurrency"].iloc[i]) if "netoWithReturnBaseCurrency" in div_f.columns else 0, "EUR") for i in range(len(div_f))]
                        show_div["Retención no recuperada (€)"] = [_fmt_div_currency(_to_float_div(div_f["originRetentionLossBaseCurrency"].iloc[i]) if "originRetentionLossBaseCurrency" in div_f.columns else 0, "EUR") for i in range(len(div_f))]
                        show_div["AutoFx"] = div_f.get("autoFx", pd.Series(["No"] * len(div_f))).astype(str)

                        habilitar_edicion_div = st.checkbox("Habilitar edición de dividendos", key="habilitar_edicion_div")
                        puede_editar_div = habilitar_edicion_div and "_rowid_" in div_f.columns

                        if puede_editar_div:
                            edit_cols_div = ["_rowid_"] + [c for c in DIVIDENDOS_COLUMNS if c in div_f.columns]
                            edit_df_div = div_f[edit_cols_div].copy()
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
                    if div_f.empty:
                        st.caption("No hay dividendos en el listado filtrado para eliminar.")
                    else:
                        def _etiqueta_div(i: int) -> str:
                            r = div_f.iloc[i]
                            fecha = str(r.get("date", "")) + " " + str(r.get("time", ""))[:8]
                            ticker = str(r.get("ticker", r.get("ticker_Yahoo", "")))
                            broker = str(r.get("broker", ""))
                            total = r.get("totalBaseCurrency", r.get("total", ""))
                            return f"{fecha} | {ticker} | {broker} | {total} €"

                        opciones_div = list(range(len(div_f)))
                        eliminar_div = st.multiselect(
                            "Selecciona los dividendos a eliminar (según filtros actuales)",
                            opciones_div,
                            format_func=_etiqueta_div,
                            key="eliminar_dividendos",
                        )
                        if eliminar_div and st.button("Eliminar dividendos seleccionados", type="primary", key="btn_eliminar_div"):
                            rowids_to_del = [int(div_f.iloc[pos]["_rowid_"]) for pos in eliminar_div if 0 <= pos < len(div_f) and "_rowid_" in div_f.columns]
                            n = delete_dividendos_by_rowids(rowids_to_del)
                            st.success(f"Eliminados {n} dividendo(s).")
                            st.rerun()
        return

    if vista == "Intereses extranjero":
        st.header("Intereses en el extranjero (P2P / crowdlending)")
        st.caption(
            "Registro manual por **ejercicio** y **plataforma** (Mintos, Debitum, Maclear, …): intereses cobrados "
            "en el extranjero, retención allí y, si aplica, retención en destino (p. ej. España). Los totales se incorporan a **Fiscalidad** del mismo año."
        )
        df_ie = load_intereses_extranjero()
        anio_actual_ie = datetime.now().year
        ej_opts_ie = list(range(anio_actual_ie + 1, anio_actual_ie - 16, -1))

        with st.form("form_intereses_ext"):
            r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns([1, 2, 1, 1, 1])
            with r1c1:
                fejer_in = st.selectbox("Ejercicio", ej_opts_ie, key="ie_form_ejer")
            with r1c2:
                plat_in = st.text_input("Plataforma", placeholder="Ej. Mintos, Debitum, Maclear", key="ie_form_plat")
            with r1c3:
                bruto_in = st.number_input("Bruto intereses (€)", min_value=0.0, value=0.0, step=0.01, format="%.2f", key="ie_form_bruto")
            with r1c4:
                ret_in = st.number_input("Retención en el extranjero (€)", min_value=0.0, value=0.0, step=0.01, format="%.2f", key="ie_form_ret")
            with r1c5:
                ret_dest_in = st.number_input(
                    "Retención en destino (€)",
                    min_value=0.0,
                    value=0.0,
                    step=0.01,
                    format="%.2f",
                    key="ie_form_ret_dest",
                    help="Retención aplicada en tu país de residencia fiscal (p. ej. España), si la hubiera sobre estos intereses.",
                )
            r2c1, r2c2 = st.columns([1, 3])
            with r2c1:
                bonus_in = st.number_input("Bonus / promociones cobrados (€)", min_value=0.0, value=0.0, step=0.01, format="%.2f", key="ie_form_bonus")
            with r2c2:
                st.caption("Los bonus se suman al **total cobrado en el extranjero** en Fiscalidad (junto al bruto de intereses).")
            notas_in = st.text_input("Notas (opcional)", key="ie_form_notas")
            add_ie = st.form_submit_button("Añadir registro", type="primary")

        if add_ie:
            if not (plat_in or "").strip():
                st.error("Indica el nombre de la plataforma.")
            else:
                append_interes_extranjero(
                    int(fejer_in),
                    plat_in.strip(),
                    float(bruto_in),
                    float(ret_in),
                    float(bonus_in),
                    float(ret_dest_in),
                    (notas_in or "").strip(),
                )
                st.success("Registro guardado.")
                st.rerun()

        if df_ie.empty:
            st.info("Aún no hay registros. Usa el formulario superior para añadir el resumen anual de cada plataforma.")
        else:
            st.subheader("Registros guardados")

            def _etiqueta_ie(i: int) -> str:
                r = df_ie.iloc[i]
                bon = float(r["bonus_eur"] or 0) if "bonus_eur" in r.index else 0.0
                rd = float(r["retencion_destino_eur"] or 0) if "retencion_destino_eur" in r.index else 0.0
                return (
                    f"{int(r['ejercicio'])} | {r['plataforma']} | "
                    f"bruto {fmt_eur(float(r['bruto_eur'] or 0))} | bonus {fmt_eur(bon)} | ret.ext {fmt_eur(float(r['retencion_extranjero_eur'] or 0))} | ret.dest {fmt_eur(rd)}"
                )

            habilitar_edicion_ie = st.checkbox("Habilitar edición de datos", key="habilitar_edicion_ie")
            puede_editar_ie = habilitar_edicion_ie and not df_ie.empty and "_rowid_" in df_ie.columns

            if habilitar_edicion_ie and not puede_editar_ie and not df_ie.empty:
                st.warning("No se puede habilitar la edición. Falta el identificador interno de fila (_rowid_).")

            if puede_editar_ie:
                edit_cols_ie = [
                    c
                    for c in (
                        "_rowid_",
                        "ejercicio",
                        "plataforma",
                        "bruto_eur",
                        "retencion_extranjero_eur",
                        "bonus_eur",
                        "retencion_destino_eur",
                        "notas",
                    )
                    if c in df_ie.columns
                ]
                edit_df_ie = df_ie[edit_cols_ie].copy()
                for c in ("bruto_eur", "retencion_extranjero_eur", "bonus_eur", "retencion_destino_eur"):
                    if c in edit_df_ie.columns:
                        edit_df_ie[c] = pd.to_numeric(edit_df_ie[c], errors="coerce").fillna(0.0)
                if "ejercicio" in edit_df_ie.columns:
                    edit_df_ie["ejercicio"] = pd.to_numeric(edit_df_ie["ejercicio"], errors="coerce").fillna(0).astype(int)
                st.session_state["ie_original"] = edit_df_ie.copy()
                editor_key_ie = f"editor_intereses_ext_{st.session_state.get('editor_ie_ver', 0)}"
                tc = st.column_config
                col_cfg_ie = {
                    "_rowid_": tc.NumberColumn("ID", disabled=True, format="%d"),
                    "ejercicio": tc.NumberColumn("Ejercicio", step=1, format="%d"),
                    "plataforma": tc.TextColumn("Plataforma"),
                    "bruto_eur": tc.NumberColumn("Bruto intereses (€)", format="%.2f", min_value=0.0),
                    "retencion_extranjero_eur": tc.NumberColumn("Retención extranjero (€)", format="%.2f", min_value=0.0),
                    "bonus_eur": tc.NumberColumn("Bonus / promoc. (€)", format="%.2f", min_value=0.0),
                    "retencion_destino_eur": tc.NumberColumn("Retención destino (€)", format="%.2f", min_value=0.0),
                    "notas": tc.TextColumn("Notas"),
                }
                col_cfg_ie = {k: v for k, v in col_cfg_ie.items() if k in edit_cols_ie}
                edited_ie = st.data_editor(
                    edit_df_ie,
                    num_rows="fixed",
                    use_container_width=True,
                    key=editor_key_ie,
                    disabled=["_rowid_"],
                    column_config=col_cfg_ie,
                )
                try:
                    orig_ie = st.session_state["ie_original"]
                    has_changes_ie = not edited_ie.astype(str).fillna("").equals(orig_ie.astype(str).fillna(""))
                except Exception:
                    has_changes_ie = False

                def _ie_save_float(v, default=0.0):
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return default

                def _ie_save_int(v, default=0):
                    try:
                        return int(float(v))
                    except (TypeError, ValueError):
                        return default

                if has_changes_ie and st.button("Guardar cambios", type="primary", key="btn_guardar_ie_dataeditor"):
                    try:
                        for _, row in edited_ie.iterrows():
                            if not str(row.get("plataforma") or "").strip():
                                st.error("La plataforma no puede estar vacía.")
                                break
                        else:
                            for _, row in edited_ie.iterrows():
                                update_interes_extranjero(
                                    int(row["_rowid_"]),
                                    _ie_save_int(row.get("ejercicio")),
                                    str(row.get("plataforma") or "").strip(),
                                    _ie_save_float(row.get("bruto_eur")),
                                    _ie_save_float(row.get("retencion_extranjero_eur")),
                                    _ie_save_float(row.get("bonus_eur")),
                                    _ie_save_float(row.get("retencion_destino_eur")),
                                    str(row.get("notas") or "").strip(),
                                )
                            if "ie_original" in st.session_state:
                                del st.session_state["ie_original"]
                            st.session_state["editor_ie_ver"] = st.session_state.get("editor_ie_ver", 0) + 1
                            st.success("Cambios guardados correctamente.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error al guardar: {e}")
            else:
                show_ie = df_ie.drop(columns=["_rowid_"], errors="ignore").rename(columns={
                    "ejercicio": "Ejercicio",
                    "plataforma": "Plataforma",
                    "bruto_eur": "Bruto intereses (€)",
                    "retencion_extranjero_eur": "Retención extranjero (€)",
                    "bonus_eur": "Bonus / promociones (€)",
                    "retencion_destino_eur": "Retención destino (€)",
                    "notas": "Notas",
                })
                _fmt_cols = {c: (lambda v, _f=fmt_eur: _f(v) if pd.notna(v) else "–") for c in show_ie.columns if "€" in c}
                st.dataframe(show_ie.style.format(_fmt_cols, na_rep="–"), use_container_width=True, hide_index=True)

            opciones_ie = list(range(len(df_ie)))
            elim_ie = st.multiselect(
                "Eliminar registros seleccionados",
                opciones_ie,
                format_func=_etiqueta_ie,
                key="eliminar_intereses_ext",
            )
            if elim_ie and st.button("Eliminar seleccionados", type="primary", key="btn_eliminar_ie"):
                rowids_del = [int(df_ie.iloc[pos]["_rowid_"]) for pos in elim_ie if "_rowid_" in df_ie.columns]
                n_del = delete_intereses_extranjero_by_rowids(rowids_del)
                st.success(f"Eliminados {n_del} registro(s).")
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

        # --- Selector de ejercicio (año natural; ventas/tramos por «Fecha venta») ---
        anio_actual = datetime.now().year
        _ej_opts = list(range(anio_actual + 1, anio_actual - 16, -1))
        _ej_def = _ej_opts.index(anio_actual) if anio_actual in _ej_opts else min(1, len(_ej_opts) - 1)
        ejercicio = st.selectbox(
            "Ejercicio fiscal",
            options=_ej_opts,
            index=_ej_def,
            help="Resumen fiscal, G/P, dividendos, comisiones, ventas y tramos FIFO se filtran por este año (fecha de venta o de cobro). "
            "Posiciones vivas = cartera actual (sin cortar por año).",
            key="fisc_ejercicio_year",
        )
        st.caption(
            f"El selector **arranca en {anio_actual}**. Si ves la venta en Movimientos pero no aquí, comprueba que el año sea el de la **fecha de venta** (p. ej. 2025)."
        )

        # --- Cargar datos FIFO ---
        df_fondos_fisc = load_data_fondos()
        df_crip_fisc = load_data_criptos()

        lots_df, sales_df, sales_detail_acc = compute_fifo_all(df)
        sales_acc_para_regla_2m = sales_df.copy()
        lots_fondos, sales_fondos, sales_detail_fondos = compute_fifo_fondos(df_fondos_fisc)
        sales_fondos_para_regla_2m = sales_fondos.copy()
        lots_crip, sales_crip, sales_detail_crip = compute_fifo_criptos(df_crip_fisc)

        sales_detail_parts = [sales_detail_acc, sales_detail_fondos, sales_detail_crip]
        _sd_nonempty = [p for p in sales_detail_parts if p is not None and not p.empty]
        sales_detail_df = pd.concat(_sd_nonempty, ignore_index=True) if _sd_nonempty else pd.DataFrame()

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

        sales_detail_ejercicio = sales_detail_df.copy()
        if not sales_detail_ejercicio.empty and "Fecha venta" in sales_detail_ejercicio.columns:
            fechas_d = pd.to_datetime(sales_detail_ejercicio["Fecha venta"], errors="coerce")
            sales_detail_ejercicio = sales_detail_ejercicio[fechas_d.dt.year == ejercicio].copy()

        # --- Resumen fiscal (tarjetas) ---
        g_p_brutas = sales_ejercicio["Plusvalía / Minusvalía (€)"].sum() if not sales_ejercicio.empty and "Plusvalía / Minusvalía (€)" in sales_ejercicio.columns else 0.0

        div_df = load_dividendos()
        div_ejercicio = pd.DataFrame()
        if not div_df.empty:
            div_df = div_df.copy()
            div_df["year"] = pd.to_datetime(div_df["date"], errors="coerce").dt.year
            div_ejercicio = div_df[div_df["year"] == ejercicio].copy()

        total_dividendos_bruto = div_ejercicio["totalBaseCurrency"].apply(lambda x: _to_float_div(x, 0.0)).sum() if not div_ejercicio.empty else 0.0
        retencion_origen_efectiva_div = 0.0
        impuesto_ext_imputable_div = 0.0
        retencion_destino_div = 0.0
        if not div_ejercicio.empty:
            for _, r in div_ejercicio.iterrows():
                total_b = _to_float_div(r.get("totalBaseCurrency"), 0.0)
                neto_b = _to_float_div(r.get("netoBaseCurrency"), 0.0)
                retencion_origen_efectiva_div += total_b - neto_b
                retencion_destino_div += _to_float_div(r.get("destinationRetentionBaseCurrency"), 0.0)
            if "retentionReturnedBaseCurrency" in div_ejercicio.columns:
                impuesto_ext_imputable_div = div_ejercicio["retentionReturnedBaseCurrency"].apply(
                    lambda x: _to_float_div(x, 0.0)
                ).sum()
            else:
                impuesto_ext_imputable_div = retencion_origen_efectiva_div
        total_neto_dividendos = div_ejercicio["totalNetoBaseCurrency"].apply(lambda x: _to_float_div(x, 0.0)).sum() if not div_ejercicio.empty and "totalNetoBaseCurrency" in div_ejercicio.columns else (
            total_dividendos_bruto - retencion_origen_efectiva_div - retencion_destino_div
        )

        bruto_div_extranjero_con_ret_origen = 0.0
        _has_rr_div = not div_ejercicio.empty and "retentionReturnedBaseCurrency" in div_ejercicio.columns
        if not div_ejercicio.empty:
            for _, r in div_ejercicio.iterrows():
                isn = _norm_isin_field(r.get("isin")) if "isin" in div_ejercicio.columns else ""
                if not isn:
                    continue
                if isn[:2] == "ES":
                    continue
                total_b = _to_float_div(r.get("totalBaseCurrency"), 0.0)
                _incluye_sigma = False
                if _has_rr_div:
                    rr = r.get("retentionReturnedBaseCurrency")
                    if rr is not None and not (isinstance(rr, float) and pd.isna(rr)) and str(rr).strip() != "":
                        _incluye_sigma = _to_float_div(rr, 0.0) > 1e-9
                    else:
                        neto_b = _to_float_div(r.get("netoBaseCurrency"), 0.0)
                        _incluye_sigma = (total_b - neto_b) > 1e-9
                else:
                    neto_b = _to_float_div(r.get("netoBaseCurrency"), 0.0)
                    _incluye_sigma = (total_b - neto_b) > 1e-9
                if _incluye_sigma:
                    bruto_div_extranjero_con_ret_origen += total_b

        df_ie_fisc = load_intereses_extranjero()
        p2p_bruto_sum = 0.0
        p2p_bonus_sum = 0.0
        p2p_cobrado_con_ret_extranjero = 0.0
        p2p_retencion_extranjero_sum = 0.0
        p2p_retencion_destino_sum = 0.0
        if not df_ie_fisc.empty and "ejercicio" in df_ie_fisc.columns:
            _mie = df_ie_fisc[df_ie_fisc["ejercicio"] == ejercicio]
            p2p_bruto_sum = float(_mie["bruto_eur"].fillna(0).astype(float).sum()) if "bruto_eur" in _mie.columns else 0.0
            p2p_bonus_sum = float(_mie["bonus_eur"].fillna(0).astype(float).sum()) if "bonus_eur" in _mie.columns else 0.0
            p2p_retencion_extranjero_sum = (
                float(_mie["retencion_extranjero_eur"].fillna(0).astype(float).sum()) if "retencion_extranjero_eur" in _mie.columns else 0.0
            )
            p2p_retencion_destino_sum = (
                float(_mie["retencion_destino_eur"].fillna(0).astype(float).sum()) if "retencion_destino_eur" in _mie.columns else 0.0
            )
            if "retencion_extranjero_eur" in _mie.columns:
                _mie_sigma = _mie[_mie["retencion_extranjero_eur"].fillna(0).astype(float) > 0]
                _br_sig = float(_mie_sigma["bruto_eur"].fillna(0).astype(float).sum()) if "bruto_eur" in _mie_sigma.columns else 0.0
                _bon_sig = float(_mie_sigma["bonus_eur"].fillna(0).astype(float).sum()) if "bonus_eur" in _mie_sigma.columns else 0.0
                p2p_cobrado_con_ret_extranjero = _br_sig + _bon_sig

        p2p_cobrado_total = p2p_bruto_sum + p2p_bonus_sum
        total_bruto_extranjero = bruto_div_extranjero_con_ret_origen + p2p_cobrado_con_ret_extranjero
        total_retencion_extranjero = impuesto_ext_imputable_div + p2p_retencion_extranjero_sum

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

        impuesto_ext_no_rec_div = max(0.0, retencion_origen_efectiva_div - impuesto_ext_imputable_div)

        col1, col2 = st.columns([0.9, 1.35])
        with col1:
            st.subheader("Resumen fiscal")
            st.metric(
                "G/P realizadas brutas (€)",
                fmt_eur(g_p_brutas),
                help="Suma de plusvalías y minusvalías realizadas en el ejercicio (según fecha de venta), en EUR.",
            )
            st.metric(
                "Dividendos cobrados brutos (€)",
                fmt_eur(total_dividendos_bruto),
                help="Suma del bruto en EUR de todos los dividendos del ejercicio (campo totalBaseCurrency u homólogo).",
            )
            st.metric(
                "Total neto dividendos (€)",
                fmt_eur(total_neto_dividendos),
                help="Suma del neto cobrado de dividendos del ejercicio (totalNetoBaseCurrency u homólogo).",
            )
            st.metric(
                "Comisiones (€)",
                fmt_eur(total_comisiones),
                help="Comisiones de movimientos del ejercicio (acciones, fondos, cripto) más comisiones ligadas a dividendos del ejercicio.",
            )
        with col2:
            _ar, _ap = _FISC_ACCENT_RENTA, _FISC_ACCENT_P2P
            _abb, _abr = _FISC_BROWN_BRUTO, _FISC_BROWN_RET
            st.markdown(
                "<style>"
                "[data-testid='stMetricLabel'] { white-space: normal !important; overflow: visible !important; "
                "text-overflow: clip !important; max-width: none !important; line-height: 1.25 !important; }"
                "span[id^='fiscal-accent-'] { display: none !important; }"
                "div.element-container:has(#fiscal-accent-bruto-div) + div.element-container [data-testid='stMetricValue'],"
                "div[data-testid='stVerticalBlock'] > div:has(#fiscal-accent-bruto-div) + div [data-testid='stMetricValue'] "
                f"{{ color: {_ar} !important; }}"
                "div.element-container:has(#fiscal-accent-retdest-div) + div.element-container [data-testid='stMetricValue'],"
                "div[data-testid='stVerticalBlock'] > div:has(#fiscal-accent-retdest-div) + div [data-testid='stMetricValue'] "
                f"{{ color: {_ar} !important; }}"
                "div.element-container:has(#fiscal-accent-p2p-bruto) + div.element-container [data-testid='stMetricValue'],"
                "div[data-testid='stVerticalBlock'] > div:has(#fiscal-accent-p2p-bruto) + div [data-testid='stMetricValue'] "
                f"{{ color: {_ap} !important; }}"
                "div.element-container:has(#fiscal-accent-p2p-retdest) + div.element-container [data-testid='stMetricValue'],"
                "div[data-testid='stVerticalBlock'] > div:has(#fiscal-accent-p2p-retdest) + div [data-testid='stMetricValue'] "
                f"{{ color: {_ap} !important; }}"
                "div.element-container:has(#fiscal-accent-div-bruto-sigma) + div.element-container [data-testid='stMetricValue'],"
                "div[data-testid='stVerticalBlock'] > div:has(#fiscal-accent-div-bruto-sigma) + div [data-testid='stMetricValue'] "
                f"{{ color: {_abb} !important; }}"
                "div.element-container:has(#fiscal-accent-div-imp-imputable) + div.element-container [data-testid='stMetricValue'],"
                "div[data-testid='stVerticalBlock'] > div:has(#fiscal-accent-div-imp-imputable) + div [data-testid='stMetricValue'] "
                f"{{ color: {_abr} !important; }}"
                "div.element-container:has(#fiscal-accent-div-no-rec) + div.element-container [data-testid='stMetricValue'],"
                "div[data-testid='stVerticalBlock'] > div:has(#fiscal-accent-div-no-rec) + div [data-testid='stMetricValue'] "
                "{ color: #ffffff !important; }"
                "div.element-container:has(#fiscal-accent-div-neto-cobrado) + div.element-container [data-testid='stMetricValue'],"
                "div[data-testid='stVerticalBlock'] > div:has(#fiscal-accent-div-neto-cobrado) + div [data-testid='stMetricValue'] "
                "{ color: #ffffff !important; }"
                "div.element-container:has(#fiscal-accent-p2p-bruto-sigma) + div.element-container [data-testid='stMetricValue'],"
                "div[data-testid='stVerticalBlock'] > div:has(#fiscal-accent-p2p-bruto-sigma) + div [data-testid='stMetricValue'] "
                f"{{ color: {_abb} !important; }}"
                "div.element-container:has(#fiscal-accent-p2p-retorigen) + div.element-container [data-testid='stMetricValue'],"
                "div[data-testid='stVerticalBlock'] > div:has(#fiscal-accent-p2p-retorigen) + div [data-testid='stMetricValue'] "
                f"{{ color: {_abr} !important; }}"
                "div.element-container:has(#fiscal-accent-sigma-bruto) + div.element-container [data-testid='stMetricValue'],"
                "div[data-testid='stVerticalBlock'] > div:has(#fiscal-accent-sigma-bruto) + div [data-testid='stMetricValue'] "
                f"{{ color: {_abb} !important; }}"
                "div.element-container:has(#fiscal-accent-sigma-reten) + div.element-container [data-testid='stMetricValue'],"
                "div[data-testid='stVerticalBlock'] > div:has(#fiscal-accent-sigma-reten) + div [data-testid='stMetricValue'] "
                f"{{ color: {_abr} !important; }}"
                "</style>",
                unsafe_allow_html=True,
            )
            st.subheader("Dividendos y cupones")
            st.caption(
                "Dos bloques paralelos: **izquierda** = cupones e impuestos desde Filios; **derecha** = P2P (hoja Intereses extranjero). "
                "Abajo: Σ bruto extranjero y Σ retenido en el extranjero (origen). Retenciones en destino: arriba en dividendos y en P2P."
            )
            _filios_col, _p2p_col = st.columns([1.15, 1])
            with _filios_col:
                with st.container(border=True):
                    st.markdown("###### Renta variable — dividendos (Filios)")
                    _dv1, _dv2 = st.columns(2)
                    with _dv1:
                        st.markdown(
                            '<span id="fiscal-accent-bruto-div"></span>',
                            unsafe_allow_html=True,
                        )
                        st.metric(
                            "Total bruto dividendos (€)",
                            fmt_eur(total_dividendos_bruto),
                            help="Igual que «Dividendos cobrados brutos» del resumen: suma de cupones brutos del ejercicio.",
                        )
                    with _dv2:
                        st.markdown(
                            '<span id="fiscal-accent-retdest-div"></span>',
                            unsafe_allow_html=True,
                        )
                        st.metric(
                            "Retenciones en destino (€)",
                            fmt_eur(retencion_destino_div),
                            help="Suma de destinationRetentionBaseCurrency: retención aplicada en destino (p. ej. España) sobre dividendos del ejercicio.",
                        )
                    st.markdown("###### Impuestos y resultado (solo dividendos Filios)")
                    _ir1a, _ir1b = st.columns(2)
                    with _ir1a:
                        st.markdown(
                            '<span id="fiscal-accent-div-bruto-sigma"></span>',
                            unsafe_allow_html=True,
                        )
                        st.metric(
                            "Bruto dividendos en Σ extranjero (€)",
                            fmt_eur(bruto_div_extranjero_con_ret_origen),
                            help="Parte de dividendos que entra en «Σ Bruto cobrado (extranjero)»: ISIN no ES y retención en origen extranjero imputable (retentionReturned > 0, o bruto − neto si falta).",
                        )
                    with _ir1b:
                        st.markdown(
                            '<span id="fiscal-accent-div-imp-imputable"></span>',
                            unsafe_allow_html=True,
                        )
                        st.metric(
                            "Imp. extranjero imputable (€)",
                            fmt_eur(impuesto_ext_imputable_div),
                            help="Suma de retentionReturnedBaseCurrency (o, si no existe, retención en origen efectiva) de los dividendos del ejercicio.",
                        )
                    _ir2a, _ir2b = st.columns(2)
                    with _ir2a:
                        st.markdown(
                            '<span id="fiscal-accent-div-no-rec"></span>',
                            unsafe_allow_html=True,
                        )
                        st.metric(
                            "Imp. extranjero no recuperable (€)",
                            fmt_eur(impuesto_ext_no_rec_div),
                            help="Parte de la retención en origen que no se considera imputable según el cálculo de Filios (máx. entre 0 y diferencia retención efectiva − imputable). "
                            "No entra en «Σ Retenido en el extranjero».",
                        )
                    with _ir2b:
                        st.markdown(
                            '<span id="fiscal-accent-div-neto-cobrado"></span>',
                            unsafe_allow_html=True,
                        )
                        st.metric(
                            "Total neto cobrado (€)",
                            fmt_eur(total_neto_dividendos),
                            help="Neto total cobrado de dividendos tras retenciones en origen y destino (totalNetoBaseCurrency u equivalente). "
                            "No forma parte del bruto agregado «Σ Bruto cobrado (extranjero)».",
                        )
            with _p2p_col:
                with st.container(border=True):
                    st.markdown("###### P2P / crowdlending (Intereses extranjero)")
                    _p2p_r1a, _p2p_r1b = st.columns(2)
                    with _p2p_r1a:
                        st.markdown(
                            '<span id="fiscal-accent-p2p-bruto"></span>',
                            unsafe_allow_html=True,
                        )
                        st.metric(
                            "P2P — intereses brutos + bonus (€)",
                            fmt_eur(p2p_cobrado_total),
                            help="Suma de intereses brutos y bonus/promociones registrados en «Intereses extranjero» para este ejercicio.",
                        )
                    with _p2p_r1b:
                        st.markdown(
                            '<span id="fiscal-accent-p2p-retdest"></span>',
                            unsafe_allow_html=True,
                        )
                        st.metric(
                            "P2P — retención en destino (€)",
                            fmt_eur(p2p_retencion_destino_sum),
                            help="Retención en destino (p. ej. en España) que registras por plataforma en «Intereses extranjero» (suma del ejercicio).",
                        )
                    _p2p_r2a, _p2p_r2b = st.columns(2)
                    with _p2p_r2a:
                        st.markdown(
                            '<span id="fiscal-accent-p2p-bruto-sigma"></span>',
                            unsafe_allow_html=True,
                        )
                        st.metric(
                            "P2P — bruto en Σ extranjero (€)",
                            fmt_eur(p2p_cobrado_con_ret_extranjero),
                            help="Parte P2P que entra en «Σ Bruto cobrado (extranjero)»: bruto + bonus solo en filas con retención en origen extranjero > 0.",
                        )
                    with _p2p_r2b:
                        st.markdown(
                            '<span id="fiscal-accent-p2p-retorigen"></span>',
                            unsafe_allow_html=True,
                        )
                        st.metric(
                            "P2P — retención en origen (€)",
                            fmt_eur(p2p_retencion_extranjero_sum),
                            help="Retención en el extranjero que indicas en cada fila de «Intereses extranjero» (suma del ejercicio).",
                        )
            st.markdown("###### Totales en el extranjero (agregado)")
            _tc1, _tc2 = st.columns(2)
            _help_sigma_bruto = (
                "Suma de bruto solo cuando hay retención en origen en el extranjero: dividendos con ISIN no español "
                "y retención imputable (retentionReturned) o, si falta, bruto − neto > 0; "
                "más por cada fila P2P del ejercicio con retención en origen extranjero > 0, el bruto + bonus cobrado "
                "(mismo concepto que «P2P — cobrado» pero solo en esas filas). "
                "No incluye dividendos extranjeros sin retención en origen ni plataformas P2P con retención extranjera 0."
            )
            _help_sigma_retenido = (
                "Impuesto satisfecho en el extranjero que puede ser imputable: retención en origen de dividendos "
                "según Filios (p. ej. retentionReturned) más la retención en origen indicada por plataforma P2P. "
                "No incluye retenciones en destino (España); esas figuran en «Retenciones en destino» (dividendos) y en P2P."
            )
            with _tc1:
                st.markdown(
                    '<span id="fiscal-accent-sigma-bruto"></span>',
                    unsafe_allow_html=True,
                )
                st.metric(
                    "Σ Bruto cobrado (extranjero) (€)",
                    fmt_eur(total_bruto_extranjero),
                    help=_help_sigma_bruto,
                )
            with _tc2:
                st.markdown(
                    '<span id="fiscal-accent-sigma-reten"></span>',
                    unsafe_allow_html=True,
                )
                st.metric(
                    "Σ Retenido en el extranjero (€)",
                    fmt_eur(total_retencion_extranjero),
                    help=_help_sigma_retenido,
                )

        # --- Tabs ---
        tab_resumen, tab_gp_tipo, tab_gp_activo, tab_div_pos, tab_fifo_tramos, tab_regla_2m = st.tabs([
            "G/P por tipo de posición",
            "G/P por activo",
            "Dividendos por posición",
            "Posiciones vivas / Ventas (detalle)",
            "Detalle FIFO (tramos)",
            "Regla 2 meses (ISIN)",
        ])

        # Mapeo tipo activo → etiqueta
        def _tipo_label(t):
            t = str(t or "").strip().lower()
            if t == "stock": return "Valores"
            if t == "etf": return "ETFs"
            if t == "fund": return "Fondos"
            if t == "crypto": return "Cripto"
            if t == "warrant": return "Otros"
            if t == "putoption": return "Puts"
            if t == "calloption": return "Calls"
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
                if "retentionReturnedBaseCurrency" in div_pos.columns:
                    div_pos["impuesto_ext"] = div_pos["retentionReturnedBaseCurrency"].apply(lambda x: _to_float_div(x, 0.0))
                else:
                    div_pos["impuesto_ext"] = div_pos["totalBaseCurrency"].apply(lambda x: _to_float_div(x, 0.0)) - div_pos["netoBaseCurrency"].apply(
                        lambda x: _to_float_div(x, 0.0)
                    )
                div_pos["total_despues_origen"] = div_pos["netoBaseCurrency"].apply(lambda x: _to_float_div(x, 0.0))
                div_pos["ret_dest"] = div_pos["destinationRetentionBaseCurrency"].apply(lambda x: _to_float_div(x, 0.0))
                div_pos["total_neto"] = div_pos["totalNetoBaseCurrency"].apply(lambda x: _to_float_div(x, 0.0)) if "totalNetoBaseCurrency" in div_pos.columns else (div_pos["total_despues_origen"] - div_pos["ret_dest"])
                if "netoWithReturnBaseCurrency" in div_pos.columns:
                    div_pos["total_neto_devol"] = div_pos["netoWithReturnBaseCurrency"].apply(lambda x: _to_float_div(x, 0.0))
                elif "unrealizedDestinationRetentionBaseCurrency" in div_pos.columns:
                    u = div_pos["unrealizedDestinationRetentionBaseCurrency"].apply(lambda x: _to_float_div(x, 0.0))
                    div_pos["total_neto_devol"] = div_pos["total_neto"] - u
                else:
                    div_pos["total_neto_devol"] = div_pos["total_neto"]

                if "isin" not in div_pos.columns:
                    div_pos["isin"] = ""
                div_pos["isin"] = div_pos["isin"].fillna("").map(_norm_isin_field)

                def _isin_agrupado(s: pd.Series) -> str:
                    seen: list[str] = []
                    for x in s:
                        t = str(x).strip() if pd.notna(x) else ""
                        if t and t not in seen:
                            seen.append(t)
                    if not seen:
                        return ""
                    if len(seen) == 1:
                        return seen[0]
                    return " / ".join(seen)

                agg_div = div_pos.groupby("ticker", as_index=False).agg(
                    isin=("isin", _isin_agrupado),
                    total_bruto=("total_bruto", "sum"),
                    impuesto_ext=("impuesto_ext", "sum"),
                    total_despues_origen=("total_despues_origen", "sum"),
                    ret_dest=("ret_dest", "sum"),
                    total_neto=("total_neto", "sum"),
                    total_neto_devol=("total_neto_devol", "sum"),
                ).rename(columns={
                    "ticker": "Posición",
                    "isin": "ISIN",
                    "total_bruto": "Total bruto (€)",
                    "impuesto_ext": "Impuesto satisf. en el extranjero (€)",
                    "total_despues_origen": "Total bruto después de origen (€)",
                    "ret_dest": "Retención en dest. realizada (€)",
                    "total_neto": "Total neto cobrado (€)",
                    "total_neto_devol": "Total neto con devolución (€)",
                })
                _cols_div = ["Posición", "ISIN"] + [c for c in agg_div.columns if c not in ("Posición", "ISIN")]
                agg_div = agg_div[_cols_div]
                # Fila TOTAL
                fila_total = pd.DataFrame([{
                    "Posición": "TOTAL",
                    "ISIN": "",
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
            st.subheader("Estado de lotes FIFO (consumo acumulado)")
            st.caption(
                "**Vivo** (fila en blanco): el lote no ha sido vendido/permutado. "
                "**Parcial** (fondo ámbar): parte de la cantidad ya se consumió en ventas o permutas. "
                "**Agotado** (fondo gris): todo consumido; solo aparece en histórico. "
                "Cripto usa FIFO global por ticker (**Broker** «—»). "
                "**Consumida** y **% consumido** son históricos (todos los tramos). "
                f"**Consumida (ejercicio)** = tramos con fecha de venta en **{ejercicio}**. "
                "**Mismo año**: columna estrecha con **✓** si la fecha de lote y la(s) venta(s) del ejercicio son del **mismo año natural** (compra y venta en ese año); vacío si la compra fue en un año anterior u otra mezcla de años."
            )
            ledger_fifo = build_fifo_lote_estado_ledger(lots_df, sales_detail_df, sales_detail_ejercicio)
            if ledger_fifo.empty:
                st.info("No hay datos de lotes ni tramos de venta para mostrar.")
            else:
                n_tot = len(ledger_fifo)
                fc1, fc2, fc3, fc4, fc5 = st.columns(5)
                with fc1:
                    q_ticker = st.text_input(
                        "Ticker / Yahoo",
                        key="fifo_led_q_ticker",
                        placeholder="Contiene… (vacío = todos)",
                        help="Filtra filas donde «Ticker» o «Yahoo/Ticker» contengan el texto (ignora mayúsculas).",
                    )
                with fc2:
                    q_nombre = st.text_input(
                        "Nombre",
                        key="fifo_led_q_nombre",
                        placeholder="Contiene… (vacío = todos)",
                        help="Filtra por la columna «Nombre» del lote (ignora mayúsculas).",
                    )
                with fc3:
                    br_opts = ["(Todos)"] + sorted(
                        {str(x).strip() for x in ledger_fifo["Broker"].dropna().unique() if str(x).strip()}
                    )
                    sel_br = st.selectbox("Broker", br_opts, key="fifo_led_broker")
                with fc4:
                    orig_opts = ["(Todos)"] + sorted(
                        {str(x).strip() for x in ledger_fifo["Origen FIFO"].dropna().unique() if str(x).strip()}
                    )
                    sel_orig = st.selectbox("Origen FIFO", orig_opts, key="fifo_led_origen")
                with fc5:
                    est_opts = ["(Todos)", "Vivo", "Parcial", "Agotado"]
                    sel_est = st.selectbox("Estado", est_opts, key="fifo_led_estado")
                sort_opts = [
                    "Fecha lote (cronológico)",
                    "Fecha lote (reciente primero)",
                    "Ticker A-Z",
                    "Estado: Vivo → Parcial → Agotado",
                    "% consumido (mayor a menor)",
                    "% consumido (menor a mayor)",
                ]
                sel_sort = st.selectbox("Ordenar por", sort_opts, key="fifo_led_sort")
                solo_consumo_ej = st.checkbox(
                    f"Recortar: solo tramos vendidos en {ejercicio}",
                    value=True,
                    key="fifo_led_solo_ej",
                    help="Mantiene filas donde «Consumida (ejercicio)» > 0 (hubo al menos una venta con Fecha venta en ese año). "
                    "Un lote en estado **Parcial** con consumo solo en **otros años** tiene aquí 0 en esa columna y **no se muestra**; "
                    "desmarca para ver también esos parciales.",
                )

                lv = ledger_fifo.copy()
                if solo_consumo_ej and "Consumida (ejercicio)" in lv.columns:
                    _eps_c = max(MIN_POSITION * 100, 1e-6)
                    lv = lv[pd.to_numeric(lv["Consumida (ejercicio)"], errors="coerce").fillna(0.0) > _eps_c]
                if (q_ticker or "").strip():
                    qt = (q_ticker or "").strip().lower()
                    t_col = lv["Ticker"].astype(str).str.lower()
                    y_col = lv["Yahoo/Ticker"].astype(str).str.lower()
                    lv = lv[t_col.str.contains(qt, regex=False, na=False) | y_col.str.contains(qt, regex=False, na=False)]
                if (q_nombre or "").strip() and "Nombre" in lv.columns:
                    qn = (q_nombre or "").strip().lower()
                    n_col = lv["Nombre"].astype(str).str.lower()
                    lv = lv[n_col.str.contains(qn, regex=False, na=False)]
                if sel_br != "(Todos)":
                    lv = lv[lv["Broker"].astype(str) == sel_br]
                if sel_orig != "(Todos)":
                    lv = lv[lv["Origen FIFO"].astype(str) == sel_orig]
                if sel_est != "(Todos)":
                    lv = lv[lv["Estado"].astype(str) == sel_est]

                if lv.empty:
                    st.warning("Ningún lote cumple los filtros.")
                else:
                    lv = lv.copy()
                    if sel_sort == "Fecha lote (cronológico)":
                        lv["_fd"] = pd.to_datetime(lv["Fecha lote"], errors="coerce")
                        lv = lv.sort_values(["_fd", "Ticker"], ascending=[True, True]).drop(columns=["_fd"])
                    elif sel_sort == "Fecha lote (reciente primero)":
                        lv["_fd"] = pd.to_datetime(lv["Fecha lote"], errors="coerce")
                        lv = lv.sort_values(["_fd", "Ticker"], ascending=[False, True]).drop(columns=["_fd"])
                    elif sel_sort == "Ticker A-Z":
                        lv = lv.sort_values(["Ticker", "Fecha lote"], ascending=[True, True])
                    elif sel_sort == "Estado: Vivo → Parcial → Agotado":
                        _ord_map = {"Vivo": 0, "Parcial": 1, "Agotado": 2}
                        lv["_so"] = lv["Estado"].map(lambda x: _ord_map.get(str(x).strip(), 9))
                        lv = lv.sort_values(["_so", "Fecha lote", "Ticker"], ascending=[True, True, True]).drop(
                            columns=["_so"]
                        )
                    elif sel_sort == "% consumido (mayor a menor)":
                        lv = lv.sort_values(["% consumido", "Fecha lote"], ascending=[False, True])
                    else:
                        lv = lv.sort_values(["% consumido", "Fecha lote"], ascending=[True, True])
                    lv = lv.reset_index(drop=True)
                    st.caption(f"**{len(lv)}** fila(s) mostradas · **{n_tot}** lotes en total (sin filtrar).")
                    st.dataframe(
                        lv.style.apply(style_fifo_lote_estado_row, axis=1),
                        use_container_width=True,
                    )
                    st.subheader("Totales (ejercicio · lotes visibles)")
                    st.caption(
                        f"Suma de **tramos FIFO** con fecha de venta en **{ejercicio}** que corresponden a los lotes de la tabla "
                        "(mismo origen, broker, ticker Yahoo, fecha de lote y precio medio). Útil para cotejar con declaraciones tipo Zergabidea. "
                        "**Ganancia** y **Pérdida** acumulan por separado el resultado de cada tramo (±); en el mismo filtrado puede haber "
                        "ventas a ganancia y a pérdida. **Plusvalía neta** es la suma algebraica de todos los tramos."
                    )
                    _t_vis = fifo_tramos_ejercicio_totales_para_lotes_visibles(lv, sales_detail_ejercicio)
                    if _t_vis["n_tramos"] == 0:
                        st.info(
                            "No hay tramos del ejercicio enlazados a estos lotes (sin coincidencia de claves o sin detalle FIFO en el año)."
                        )
                    else:
                        _fa_s = ""
                        if "Fecha lote" in lv.columns:
                            _fd_lo = pd.to_datetime(lv["Fecha lote"], errors="coerce").dropna()
                            if len(_fd_lo):
                                _fa_s = pd.Timestamp(_fd_lo.min()).strftime("%Y-%m-%d")
                        _fv = _t_vis.get("fechas_venta") or []
                        _ft_s = ", ".join(_fv) if _fv else "—"
                        m1, m2, m3, m4, m5 = st.columns(5)
                        with m1:
                            st.metric("Tramos (filas)", f"{_t_vis['n_tramos']}")
                        with m2:
                            st.metric("Valor adquisición (€)", fmt_eur(_t_vis["adquisicion"]))
                            st.caption(f"Fecha adquisición: {_fa_s or '—'}")
                        with m3:
                            st.metric("Valor transmisión (€)", fmt_eur(_t_vis["transmision"]))
                            st.caption(
                                f"Fecha(s) transmisión: {_ft_s}"
                                + (" (varias ventas en el ejercicio)" if len(_fv) > 1 else "")
                            )
                        _per_e = abs(_t_vis["perdida"]) if _t_vis["perdida"] < 0 else 0.0
                        with m4:
                            _st_metric_colored(
                                "Ganancia (€)",
                                fmt_eur(_t_vis["ganancia"]),
                                _plusvalia_color_css(_t_vis["ganancia"]),
                            )
                        with m5:
                            _st_metric_colored(
                                "Pérdida (€)",
                                fmt_eur(_per_e),
                                "#c62828" if _per_e > 0 else "#9e9e9e",
                            )
                        _c_n = _plusvalia_color_css(_t_vis["neto"])
                        st.markdown(
                            f"<p style='color:{_c_n}; font-size:0.875rem; margin:0.35rem 0 0 0;'>"
                            f"<strong>Plusvalía neta</strong> (ganancia + pérdidas): {fmt_eur(_t_vis['neto'])}</p>",
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            "Adquisición = fecha de lote más antigua entre filas visibles. "
                            f"Transmisión = todas las fechas de venta del ejercicio **{ejercicio}** en esos tramos. "
                            "En Renta declaras cada transmisión con su día concreto."
                        )
                        _desg_fv = fifo_tramos_ejercicio_desglose_por_fecha_venta(lv, sales_detail_ejercicio)
                        if not _desg_fv.empty:
                            st.markdown("**Desglose por fecha de venta** (mismos lotes filtrados)")
                            _fmt_d = {
                                c: fmt_eur
                                for c in _desg_fv.columns
                                if "€" in str(c) and c != "Fecha venta"
                            }
                            st.dataframe(
                                _desg_fv.style.format(_fmt_d, na_rep="–"),
                                use_container_width=True,
                            )

            st.subheader("Posiciones vivas (FIFO por lotes)")
            if lots_df.empty:
                st.info("No hay lotes vivos.")
            else:
                lots_show = lots_df.sort_values(["Broker", "Ticker", "Fecha origen"]).copy()
                for old, new in [("ticker", "Ticker"), ("broker", "Broker"), ("nombre", "Nombre")]:
                    if old in lots_show.columns and new not in lots_show.columns:
                        lots_show = lots_show.rename(columns={old: new})
                col_order = [c for c in ["Ticker", "ISIN", "Broker", "Nombre", "Fecha origen", "Cantidad", "Precio medio €", "Tipo activo"] if c in lots_show.columns]
                st.dataframe(lots_show[col_order] if col_order else lots_show, use_container_width=True)

            st.subheader("Ventas (impacto fiscal FIFO)")
            st.caption(f"Solo ventas con **Fecha venta** en **{ejercicio}** (mismo criterio que el resumen superior).")
            if sales_ejercicio.empty:
                st.info("No hay ventas en este ejercicio.")
            else:
                sales_show = sales_ejercicio.sort_values(["Fecha venta", "Broker", "Ticker"]).copy()
                col_order_s = [c for c in ["Ticker", "ISIN", "Broker", "Nombre", "Fecha venta", "Cantidad vendida", "Valor compra histórico (€)", "Valor venta (€)", "Plusvalía / Minusvalía (€)", "Retención dest. (€)", "Tipo activo"] if c in sales_show.columns]
                st.dataframe(sales_show[col_order_s] if col_order_s else sales_show, use_container_width=True)
                total_pnl = sales_show["Plusvalía / Minusvalía (€)"].sum() if "Plusvalía / Minusvalía (€)" in sales_show.columns else 0
                _c_p = _plusvalia_color_css(total_pnl)
                st.markdown(
                    f"<p style='color:{_c_p};'><strong>Plusvalía/Minusvalía total (ejercicio {ejercicio})</strong>: "
                    f"{fmt_eur(total_pnl)}</p>",
                    unsafe_allow_html=True,
                )

        with tab_fifo_tramos:
            st.subheader("Desglose FIFO por lote consumido")
            st.caption(
                "Cada venta o permuta (switch) puede repartirse en varios tramos según los lotes "
                "comprados. El valor de transmisión se reparte proporcionalmente por cantidad; "
                "la última fila del tramo absorbe el redondeo para que coincida con el total de la venta."
            )
            if sales_detail_ejercicio.empty:
                st.info("No hay tramos FIFO en este ejercicio.")
            else:
                det_show = sales_detail_ejercicio.sort_values(
                    ["Fecha venta", "Broker", "Ticker", "Fecha origen (lote)"],
                    ascending=[True, True, True, True],
                ).copy()
                col_det = [
                    c for c in [
                        "Origen FIFO", "Venta #", "Broker", "Ticker", "ISIN", "Ticker_Yahoo", "Nombre", "Tipo activo",
                        "Tipo movimiento", "Fecha venta", "Cantidad venta (total)", "Cantidad (tramo)",
                        "Fecha origen (lote)", "Valor compra histórico (€)", "Valor venta (€)",
                        "Plusvalía / Minusvalía (€)",
                    ]
                    if c in det_show.columns
                ]
                sty = det_show[col_det].style.format(
                    {c: lambda x: fmt_eur(x) for c in col_det if "€" in str(c)},
                    na_rep="–",
                )
                pnl_cols = [c for c in col_det if "Plusvalía" in str(c)]
                if pnl_cols:
                    sty = _style_map(sty, color_pnl, subset=pnl_cols)
                st.dataframe(sty, use_container_width=True)
                with st.expander(
                    "Cómo usar este CSV en el programa de Renta (p. ej. Zergabidea)",
                    expanded=False,
                ):
                    st.markdown(
                        """
**Ejercicio en el filtro superior:** debe ser el **año de la venta** (transmisión que declaras), **no** el año en que compraste los lotes. Las compras aparecen en **Fecha origen (lote)**.

**Columnas útiles**

| Columna | Significado |
|--------|-------------|
| **Venta #** | Mismo número = una misma venta partida en varios tramos FIFO. |
| **ISIN** | Identificador del valor (movimiento o catálogo **Instrumentos**); puede ir vacío si falta mapeo. |
| **Ticker_Yahoo** | Clave de cotización en la app. |
| **Fecha venta** | Transmisión; suele repetirse en todas las filas de ese `Venta #`. |
| **Fecha origen (lote)** | Compra de ese tramo (Hacienda aplicará el coeficiente que corresponda). |
| **Valor compra histórico (€)** | Valor de adquisición de ese tramo para trasladar a la declaración. |
| **Valor venta (€)** | Parte del valor de transmisión asignada a ese tramo. |

**Trabajo habitual:** abre el CSV en una hoja de cálculo, ordena por **Venta #** y por **Fecha origen (lote)**. Para cada `Venta #`, o bien introduces **una línea en Renta por cada fila** (recomendable si hay varias fechas de compra), o bien agrupas por **año** de `Fecha origen (lote)` sumando compra y venta **solo si** tu programa lo admite con una sola fecha de adquisición por grupo.

Los **coeficientes de actualización** y el cuadre final los calcula el **software de Hacienda**; este CSV no los incluye.

**Comprueba:** la suma de **Valor venta (€)** de todas las filas con el mismo **Venta #** debe igualar el total de esa venta (la app reparte proporcionalmente y el último tramo absorbe el redondeo).
"""
                    )
                csv_bytes = det_show[col_det].to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
                st.download_button(
                    "Descargar CSV (detalle FIFO)",
                    data=csv_bytes,
                    file_name=f"fiscalidad_fifo_tramos_{ejercicio}.csv",
                    mime="text/csv",
                    key="dl_fifo_tramos",
                )

        with tab_regla_2m:
            st.subheader("Prueba: regla 2 meses (ISIN)")
            st.caption(
                "Para cada **venta en pérdida** del ejercicio (acciones/ETF y fondos, sin cripto), "
                "se buscan **compras** (`buy` / `switchbuy`) con el **mismo ISIN** entre "
                "**fecha de venta − 2 meses** y **fecha de venta + 2 meses** (meses naturales). "
                "Solo se marca **revisión fuerte** si tras la venta **quedan títulos** del ISIN o hay **recompra** "
                "en la ventana; una **liquidación total** sin recompras posteriores se etiqueta como **No — integración orientativa**. "
                "Homogeneidad legal puede ser más amplia que el ISIN."
            )
            regla_df = deteccion_regla_dos_meses_isin_alerts(
                df,
                df_fondos_fisc,
                sales_acc_para_regla_2m,
                sales_fondos_para_regla_2m,
                ejercicio=ejercicio,
            )
            if regla_df.empty:
                st.info("No hay ventas en pérdida en este ejercicio (con datos para analizar), o no quedan filas tras filtrar.")
            else:
                def _alerta_fuerte(txt) -> bool:
                    t = str(txt or "")
                    return t.startswith("Sí") or t.startswith("Revisar")

                n_rev = int(regla_df["Alerta"].map(_alerta_fuerte).sum())
                st.caption(
                    f"**{len(regla_df)}** venta(s) en pérdida en **{ejercicio}** · "
                    f"**{n_rev}** con alerta de revisión (posición residual o recompra en ventana)."
                )
                disp = regla_df.copy()
                sty_r = disp.style.format({"Pérdida (€)": lambda x: fmt_eur(x)}, na_rep="–")
                pnl_col = [c for c in disp.columns if "Pérdida" in str(c)]
                if pnl_col:
                    sty_r = _style_map(sty_r, color_pnl, subset=pnl_col)
                st.dataframe(sty_r, use_container_width=True)
                csv_r = disp.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
                st.download_button(
                    "Descargar CSV (regla 2 meses)",
                    data=csv_r,
                    file_name=f"fiscalidad_regla_2m_isin_{ejercicio}.csv",
                    mime="text/csv",
                    key="dl_regla_2m",
                )

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
                if t in ("stock", "etf", "fund", "crypto", "warrant", "putoption", "calloption"):
                    tipos_presentes.add(t)
                elif t and t not in ("nan", ""):
                    tipos_presentes.add("stock")
            pm_tipo_options = ["Todos", "Acciones", "ETFs", "Fondos", "Criptos", "Otros"]
            if "putoption" in tipos_presentes:
                pm_tipo_options.append("Puts")
            if "calloption" in tipos_presentes:
                pm_tipo_options.append("Calls")
            pm_tipo_sel = st.radio("Tipo de activo", options=pm_tipo_options, index=0, horizontal=True, key="pm_tipo_activo")
            # Filtrar sin_cotiz según tipo
            def _pm_match_tipo(tipo: str, sel: str) -> bool:
                if sel == "Todos":
                    return True
                if sel == "Acciones":
                    return (
                        (tipo in ("stock", "") or tipo not in ("etf", "fund", "crypto", "warrant", "putoption", "calloption"))
                        and tipo not in ("putoption", "calloption")
                    )
                if sel == "ETFs":
                    return tipo == "etf"
                if sel == "Fondos":
                    return tipo == "fund"
                if sel == "Criptos":
                    return tipo == "crypto"
                if sel == "Otros":
                    return tipo == "warrant"
                if sel == "Puts":
                    return tipo == "putoption"
                if sel == "Calls":
                    return tipo == "calloption"
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
        if (tipos_all == "putoption").any():
            tipo_options.append("Puts")
        if (tipos_all == "calloption").any():
            tipo_options.append("Calls")
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
                & (tipos != "putoption")
                & (tipos != "calloption")
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
        elif tipo_sel == "Puts":
            view = view[tipos == "putoption"]
        elif tipo_sel == "Calls":
            view = view[tipos == "calloption"]

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
    styler = table[display_cols].style.format(
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
    styler = _style_map(
        styler,
        color_pnl,
        subset=["GyP no realizadas (€)", "GyP no realizadas %", "GyP hoy (€)", "GyP hoy %"],
    )

    _disp_tbl = table[display_cols]
    st.dataframe(
        styler,
        use_container_width=True,
        hide_index=True,
        column_config=_cartera_positions_column_config(_disp_tbl, display_cols),
    )


if __name__ == "__main__":
    main()

