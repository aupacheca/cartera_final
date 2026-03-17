# Cartera FINAL - Add-on Home Assistant

Cartera de inversiones (acciones, ETFs, fondos, criptos) con Streamlit.

## Instalación

1. Añade el repositorio en Home Assistant: **Configuración** → **Complementos** → **Repositorios**
2. URL: `https://github.com/aupacheca/cartera_final`
3. Instala **Cartera FINAL**
4. Puerto: **8503**
5. Datos en: **share/cartera_final** (accesible por Samba: `\\homeassistant.local\share\cartera_final`)

## Estructura del repo

Para que Home Assistant reconozca el add-on, el repo debe tener:

```
repository.yaml    (en la raíz)
cartera_final/
  config.yaml
  build.yaml
  Dockerfile
  run.sh
  app.py
  requirements.txt
```
