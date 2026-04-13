"""Conexión SQLite centralizada."""
from __future__ import annotations

import sqlite3

from filios_core.config import DB_PATH


def get_db():
    return sqlite3.connect(DB_PATH)
