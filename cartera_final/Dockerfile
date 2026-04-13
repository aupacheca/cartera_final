# Cartera FINAL - Add-on Home Assistant
# Repo plano: filios_core/ y pyproject.toml junto a este archivo (p. ej. tras sync_filios_core_to_addon.py).
#
ARG BUILD_FROM
FROM ${BUILD_FROM}

RUN apk add --no-cache build-base libffi-dev libstdc++

WORKDIR /app

COPY pyproject.toml .
COPY filios_core ./filios_core/
RUN pip3 install --no-cache-dir /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir --prefer-binary -r requirements.txt

COPY app.py .
COPY refresh_cotizaciones.py /app/refresh_cotizaciones.py

ENV DATA_DIR=/share/cartera_final
# Hora del cron de cotizaciones (lun–vie 9:30, 16:00, 22:00)
ENV TZ=Europe/Madrid
WORKDIR /share/cartera_final

EXPOSE 8503

COPY run.sh /run.sh
COPY entry_refresh.sh /app/entry_refresh.sh
COPY crontabs_root /etc/crontabs/root
RUN sed -i 's/\r$//' /run.sh /app/entry_refresh.sh \
    && chmod a+x /run.sh /app/entry_refresh.sh \
    && chmod 600 /etc/crontabs/root

CMD ["/run.sh"]
