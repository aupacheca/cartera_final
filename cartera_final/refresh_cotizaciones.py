#!/usr/bin/env python3
"""Actualiza cartera_cotizaciones_cache.pkl desde CLI (cron del add-on o pruebas locales)."""
from __future__ import annotations

import sys
from pathlib import Path


def _app_dir() -> str:
    if Path("/app/app.py").exists():
        return "/app"
    here = Path(__file__).resolve().parent
    if (here / "app.py").exists():
        return str(here)
    print("No se encuentra app.py.", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    import logging

    for _name in (
        "streamlit",
        "streamlit.runtime",
        "streamlit.runtime.caching",
        "streamlit.runtime.scriptrunner_utils",
    ):
        logging.getLogger(_name).setLevel(logging.CRITICAL)

    sys.path.insert(0, _app_dir())
    import app  # noqa: PLC0415

    ok, msg = app.refresh_cotizaciones_to_disk()
    print(msg, flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
