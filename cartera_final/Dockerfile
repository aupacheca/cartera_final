# Cartera FINAL - Add-on Home Assistant
ARG BUILD_FROM
FROM ${BUILD_FROM}

RUN apk add --no-cache build-base libffi-dev libstdc++

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir --prefer-binary -r requirements.txt

COPY app.py .

ENV DATA_DIR=/share/cartera_final
WORKDIR /share/cartera_final

EXPOSE 8503

COPY run.sh /run.sh
RUN sed -i 's/\r$//' /run.sh && chmod a+x /run.sh

CMD ["/run.sh"]
