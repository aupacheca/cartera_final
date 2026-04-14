#!/usr/bin/env python3
"""
Planificador de cotizaciones (lun–vie 9:30, 16:00, 22:00 hora Europe/Madrid).
Sustituye a crond en el add-on (más fiable que busybox crond en algunos entornos HA).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Madrid")
# Mismo horario que antes en crontabs_root
SLOTS_HM = ((9, 30), (16, 0), (22, 0))


def _next_delay_sec(now: datetime) -> float:
    best: datetime | None = None
    for day_off in range(0, 10):
        d = now.date() + timedelta(days=day_off)
        if d.weekday() >= 5:
            continue
        for h, m in SLOTS_HM:
            slot = datetime(d.year, d.month, d.day, h, m, tzinfo=TZ)
            if slot <= now:
                continue
            if best is None or slot < best:
                best = slot
    if best is None:
        return 3600.0
    return max(5.0, (best - now).total_seconds())


def _log_path() -> str:
    base = os.environ.get("DATA_DIR", "/share/cartera_final").strip() or "/share/cartera_final"
    return os.path.join(base, "cron_cotizaciones.log")


def main() -> None:
    logf = _log_path()
    while True:
        now = datetime.now(TZ)
        delay = _next_delay_sec(now)
        time.sleep(delay)
        try:
            with open(logf, "a", encoding="utf-8") as f:
                f.write(f"=== {datetime.now(TZ).isoformat(timespec='seconds')} ejecutar refresh ===\n")
        except OSError:
            pass
        env = os.environ.copy()
        r = subprocess.run(
            [sys.executable, "/app/refresh_cotizaciones.py"],
            capture_output=True,
            text=True,
            timeout=900,
            env=env,
        )
        try:
            with open(logf, "a", encoding="utf-8") as f:
                if r.stdout:
                    f.write(r.stdout)
                if r.stderr:
                    f.write(r.stderr)
                f.write(f"exit={r.returncode}\n")
        except OSError:
            pass


if __name__ == "__main__":
    main()
