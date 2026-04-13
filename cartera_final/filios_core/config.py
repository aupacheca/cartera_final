"""Rutas de datos y SQLite (DATA_DIR configurable por entorno)."""
from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", ".")).resolve()

DB_FILENAME = os.environ.get("DB_FILENAME", "acciones.db")
DB_PATH = str(DATA_DIR / DB_FILENAME)
CSV_PATH = str(DATA_DIR / "acciones.csv")
FONDOS_CSV_PATH = str(DATA_DIR / "fondos.csv")
MOVIMIENTOS_CRIPTOS_CSV_PATH = str(DATA_DIR / "movimientos_criptos.csv")
DIVIDENDOS_CSV_PATH = str(DATA_DIR / "dividendos.csv")
COTIZACIONES_CACHE_PATH = str(DATA_DIR / "cartera_cotizaciones_cache.pkl")
COTIZACIONES_META_PATH = str(DATA_DIR / "cartera_cotizaciones_meta.json")
PRECIOS_MANUALES_PATH = str(DATA_DIR / "precios_manuales.json")
