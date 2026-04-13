#!/bin/sh
# Misma resolución de DATA_DIR que run.sh (cron no hereda el entorno del proceso principal).
if [ -f /data/data_path_override.txt ]; then
    DATA_DIR=$(cat /data/data_path_override.txt | head -1 | tr -d '\r\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
elif [ -f /config/data_path.txt ]; then
    DATA_DIR=$(cat /config/data_path.txt | head -1 | tr -d '\r\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
fi
if [ -z "$DATA_DIR" ] && [ -f /data/options.json ]; then
    DATA_DIR=$(python3 -c "import json; d=json.load(open('/data/options.json')); p=(d.get('data_path') or '/share/cartera_final').strip(); print(p or '/share/cartera_final')" 2>/dev/null)
fi
DATA_DIR=${DATA_DIR:-/share/cartera_final}
export DATA_DIR
export DB_FILENAME="${DB_FILENAME:-acciones_final.db}"

LOG="$DATA_DIR/cron_cotizaciones.log"
{
    echo "=== $(date -Iseconds) ==="
    python3 /app/refresh_cotizaciones.py
} >>"$LOG" 2>&1
