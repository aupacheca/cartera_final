# Cartera FINAL - Add-on Home Assistant
# Repo plano: filios_core/ y pyproject.toml junto a este archivo (p. ej. tras sync_filios_core_to_addon.py).
#
ARG BUILD_FROM
FROM ${BUILD_FROM}

RUN apk add --no-cache build-base libffi-dev libstdc++ tzdata

WORKDIR /app

COPY pyproject.toml .
COPY filios_core ./filios_core/
RUN pip3 install --no-cache-dir /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir --prefer-binary -r requirements.txt

COPY app.py .
COPY refresh_cotizaciones.py /app/refresh_cotizaciones.py
COPY cotizaciones_scheduler.py /app/cotizaciones_scheduler.py

ENV DATA_DIR=/share/cartera_final
ENV TZ=Europe/Madrid
WORKDIR /share/cartera_final

EXPOSE 8503

COPY run.sh /run.sh
COPY entry_refresh.sh /app/entry_refresh.sh
RUN sed -i 's/\r$//' /run.sh /app/entry_refresh.sh \
    && chmod a+x /run.sh /app/entry_refresh.sh /app/cotizaciones_scheduler.py

CMD ["/run.sh"]
