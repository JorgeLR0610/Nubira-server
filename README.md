---
title: Nubira
emoji: 🐢
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# Nubira-server

Backend de Nubira Weather, una aplicación para evaluar riesgos climatológicos antes de un evento al aire libre. La consulta es pública; guardar eventos y suscribirse a alertas requiere una cuenta con correo verificado. La evaluación es climatológica: no sustituye un pronóstico meteorológico para el día del evento.

- **Web:** [nubira-weather.vercel.app](https://nubira-weather.vercel.app) · [Repositorio del frontend](https://github.com/JesusPichardo15/NubiraWeather)
- **API:** [api.nubiraweather.date](https://api.nubiraweather.date) · [Estado de la API](https://api.nubiraweather.date/health)

## Producción

El frontend en Vercel llama a la API desde su servidor. En una VM Linux de **AWS Lightsail de 2 GB**, [Docker Compose](docker-compose.yml) ejecuta Caddy y un único contenedor FastAPI con PyTorch CPU y el monitor de alertas. [Caddy](Caddyfile) obtiene y renueva el certificado de Let's Encrypt y redirige HTTP a HTTPS. El puerto interno `8000` no se publica; Uvicorn solo confía en las cabeceras reenviadas por Caddy.

La persistencia usa **Neon (PostgreSQL)** mediante `DATABASE_URL`, con los parámetros TLS de Neon. La API crea las tablas de forma idempotente al arrancar y abre conexiones por operación. La imagen requiere PostgreSQL por defecto: un error de conexión no provoca un cambio a SQLite. Los correos de verificación, restablecimiento y alertas se envían con **Amazon SES por SMTP STARTTLS** (puerto `587`). Si SES está en el sandbox, los destinatarios de prueba también deben estar verificados.

### Operación en Lightsail

Para recrear el entorno, usa una IP pública estática y un registro DNS `A` de `api.nubiraweather.date` que apunte a la VM. Abre **TCP `80` y `443`** en el firewall; **UDP `443`** es opcional para HTTP/3. No abras `8000` ni dejes un registro `AAAA` que apunte a otro servidor.

Guarda las variables en un `.env` privado (permisos `600`), con [`.env.example`](.env.example) como referencia. Además de `DATABASE_URL`, configura `MAIL_PROVIDER=smtp`, `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS` y `MAIL_FROM`. Usa el endpoint y las credenciales de la misma región de SES. `NUBIRA_API_DOMAIN` identifica la API; `FRONTEND_URL` y `CORS_ORIGINS` deben usar el origen de la **web** (`https://nubira-weather.vercel.app`), no el de la API.

Para actualizar, desde el clon del backend en la VM:

```sh
git pull --ff-only
docker compose config --quiet
docker compose up -d --build
docker compose ps
curl -fsS https://api.nubiraweather.date/health
```

Usa `docker compose logs --tail=100 -f api caddy` para seguir los logs y `docker compose down` para detener los servicios. **No añadas `-v`**: borraría los volúmenes que conservan los certificados y la configuración de Caddy. Si la subred de Compose se solapa con una red de la VM, ajusta la subred y su rango dinámico y mantén la IP fija de Caddy fuera de ese rango y sincronizada con `FORWARDED_ALLOW_IPS`.

`GET /health` devuelve `200` si los artefactos de inferencia están disponibles y `503` si faltan o no son válidos. No verifica Neon ni SES. Después de un cambio de despliegue, comprueba también una consulta sin cuenta, la entrega del enlace de verificación, el inicio de sesión y el guardado de un evento desde Vercel.

## Desarrollo local

Requiere Python 3.12. El entrenamiento y la API tienen dependencias separadas. Para iniciar la API sin entrenar ni conectarte a servicios de producción:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-api.txt
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
python fetch_artifacts.py --check
DATABASE_URL= NUBIRA_REQUIRE_POSTGRES=0 FRONTEND_URL=http://localhost:3000 \
  MAIL_PROVIDER=file NUBIRA_MAIL_DIR="$PWD/workspace/mail" \
  NUBIRA_BIND_HOST=127.0.0.1 PORT=8000 OMP_NUM_THREADS=2 python start.py
```

La API escucha en el puerto `8000` por defecto. En `NubiraWeather/.env.local`, establece `NUBIRA_API_URL=http://127.0.0.1:8000`. Si ejecutas Python directamente, exporta las variables en el entorno: `start.py` no carga `.env` automáticamente.

Sin `DATABASE_URL`, el servidor local usa SQLite (`workspace/nubira.sqlite3` o `NUBIRA_DB_PATH`). Con `MAIL_PROVIDER=file`, los enlaces de cuenta llegan como archivos `.eml` a `NUBIRA_MAIL_DIR`; `console` simula el envío sin imprimir destinatarios ni enlaces. No publiques la bandeja ni guardes credenciales en Git.

### Inferencia y lugares

El modelo y su contexto están versionados directamente en [`artifacts/`](artifacts/) y se empaquetan en la imagen. `python fetch_artifacts.py --check` verifica tamaño y SHA-256 sin red. El script sin `--check` sirve para la recuperación manual en desarrollo; el arranque no descarga archivos. Para usar otros artefactos de confianza, configura `NUBIRA_MODEL_PATH` y `NUBIRA_CLIMATE_CONTEXT_PATH`. `POST /clima` acepta fechas futuras.

`NUBIRA_GEOAPIFY_API_KEY` habilita el autocompletado y la dirección más cercana al seleccionar un punto. Ambas funciones comparten el límite de 1.000 solicitudes por ventana de 24 horas, además del límite de ráfaga. Las consultas se envían a Geoapify y la API no guarda las direcciones parciales. Sin clave siguen disponibles el mapa y la entrada manual de coordenadas. La búsqueda manual opcional compatible con Nominatim no interviene en el autocompletado.

### Monitor de alertas

En producción, el monitor integrado se ejecuta cada 900 segundos por defecto. En local puedes activarlo con `NUBIRA_MONITOR_INTERVAL=900` al iniciar la API, o ejecutarlo por separado **con el mismo entorno, base de datos y artefactos**:

```sh
TZ=UTC DATABASE_URL= NUBIRA_REQUIRE_POSTGRES=0 FRONTEND_URL=http://localhost:3000 \
  MAIL_PROVIDER=file NUBIRA_MAIL_DIR="$PWD/workspace/mail" \
  OMP_NUM_THREADS=2 python -m monitor --interval 900
```

Sin `--interval` se ejecuta un solo ciclo. Activa solo un monitor por entorno. El umbral de aviso es de 10 **puntos porcentuales** por defecto (`NUBIRA_ALERT_THRESHOLD_PP`), respecto a la evaluación inicial o a la que generó el último aviso. No hay avisos por el mero paso de los días ni por eventos vencidos. La cola persiste y reintenta los envíos fallidos.

## Pruebas

En el entorno local con PyTorch CPU:

```sh
python -m pip install -r requirements-test.txt
TZ=UTC OMP_NUM_THREADS=2 python -m pytest
```

La suite usa SQLite temporal y proveedores simulados. [GitHub Actions](.github/workflows/checks.yml) la repite con PostgreSQL local mediante `NUBIRA_TEST_POSTGRES_URL`. Si reproduces esa variante, usa solo una base de pruebas descartable: la suite crea y elimina un esquema por prueba. No se usa Neon ni SES reales.
