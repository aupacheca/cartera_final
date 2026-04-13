#!/bin/sh
# Cartera FINAL - Add-on Home Assistant
# Datos en /share/cartera_final (Samba: share\cartera_final)

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
export DB_FILENAME=acciones_final.db

mkdir -p "$DATA_DIR"

# Migración desde otras rutas
if [ -f /data/acciones_final.db ] && [ ! -f "$DATA_DIR/acciones_final.db" ]; then
    cp -a /data/acciones_final.db "$DATA_DIR/" 2>/dev/null || true
    [ -f "$DATA_DIR/acciones_final.db" ] && echo "Migrada BD de /data a $DATA_DIR"
fi
if [ -f /data/acciones_test.db ] && [ ! -f "$DATA_DIR/acciones_final.db" ]; then
    cp -a /data/acciones_test.db "$DATA_DIR/acciones_final.db" 2>/dev/null || true
    [ -f "$DATA_DIR/acciones_final.db" ] && echo "Migrada BD acciones_test.db a $DATA_DIR/acciones_final.db"
fi
if [ "$DATA_DIR" != "/config" ] && [ -f /config/acciones_final.db ] && [ ! -f "$DATA_DIR/acciones_final.db" ]; then
    cp -a /config/acciones_final.db "$DATA_DIR/" 2>/dev/null || true
    [ -f "$DATA_DIR/acciones_final.db" ] && echo "Migrada BD de /config a $DATA_DIR"
fi

echo "Cartera FINAL - Datos en: $DATA_DIR (Samba: share/cartera_final)"

# Cotizaciones automáticas (lun–vie 9:30, 16:00, 22:00 hora $TZ → crontabs_root)
crond

exec python3 -m streamlit run /app/app.py \
    --server.port=8503 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
