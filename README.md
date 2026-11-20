---
title: Nubira
emoji: 🐢
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# API, inferencia y alertas

Requiere Python 3.12. El entrenamiento y el servicio tienen dependencias separadas. Para iniciar la API sin entrenar:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-api.txt
pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
python fetch_artifacts.py --check
MAIL_PROVIDER=file NUBIRA_MAIL_DIR="$PWD/workspace/mail" \
  NUBIRA_BIND_HOST=127.0.0.1 OMP_NUM_THREADS=2 python start.py
```

La API escucha en el puerto `8000` por defecto. Si inicias el frontend local por separado, configura `NUBIRA_API_URL=http://127.0.0.1:8000` en `NubiraWeather/.env.local`.

Los dos binarios de producción ya están en `artifacts/`, sin depender de Git LFS. `fetch_artifacts.py --check` valida tamaño y SHA-256 sin red. El script sin `--check` queda como recuperación explícita en desarrollo; no se ejecuta al arrancar. Puedes montar otros artefactos de confianza con `NUBIRA_MODEL_PATH` y `NUBIRA_CLIMATE_CONTEXT_PATH`. `/health` devuelve `200` si están disponibles y `503` si faltan o son punteros Git LFS; no comprueba Neon ni SES. `/clima` requiere una fecha futura.

Con `DATABASE_URL` se usa PostgreSQL, se abre una conexión por operación y se crean las tablas al arrancar. Sin la variable, solo para desarrollo y pruebas, se usa SQLite (`workspace/nubira.sqlite3` o `NUBIRA_DB_PATH`). La imagen exige PostgreSQL por defecto y no cambia a SQLite ante un fallo de conexión. No se importan los datos del prototipo.

### Autocompletado de direcciones

Configura `NUBIRA_GEOAPIFY_API_KEY` solo en el entorno de la API y reiníciala. Las sugerencias se consultan en español, priorizan México sin excluir otros países y tienen un límite compartido de 1.000 solicitudes al proveedor por ventana de 24 horas (además del límite de ráfaga). Las direcciones parciales se envían al proveedor y la API no las guarda. Sin clave siguen disponibles el mapa y la entrada manual. La búsqueda opcional compatible con Nominatim conserva su modo de envío explícito: nunca se usa para el autocompletado.

Con la misma clave, un clic del mapa busca la dirección más cercana. La API comparte el límite de solicitudes con el autocompletado y solo devuelve una etiqueta (o `null` si no hay coincidencia); el punto original se utiliza para la evaluación.

### Correo y reevaluación

Las variables están documentadas en [`.env.example`](.env.example); expórtalas desde el entorno o configúralas en Compose. `file` guarda mensajes en una bandeja local para probar los enlaces. `console` simula el envío sin imprimir el destinatario ni los enlaces. `smtp` requiere `FRONTEND_URL` con HTTPS, un remitente autorizado y credenciales del proveedor. Por defecto usa STARTTLS en el puerto 587. No subas credenciales ni la bandeja de correo al repositorio.

La imagen de Lightsail ejecuta el monitor dentro de la API con un único proceso servidor. En local puedes activarlo con `NUBIRA_MONITOR_INTERVAL=900` o lanzarlo manualmente con el mismo entorno, base de datos y artefactos que la API:

```sh
TZ=UTC MAIL_PROVIDER=file NUBIRA_MAIL_DIR="$PWD/workspace/mail" \
  OMP_NUM_THREADS=2 python -m monitor --interval 900
```

Sin `--interval` se ejecuta un solo ciclo (útil en cron). El umbral por defecto es 10 **puntos porcentuales** (`NUBIRA_ALERT_THRESHOLD_PP`); se compara con la evaluación inicial o la que generó el último aviso. Los días restantes no activan avisos; el modelo no incorpora un pronóstico que vaya cambiando a medida que se acerca el evento. La cola persiste y aplica reintentos; los avisos llevan identificador estable. Como con cualquier SMTP, un fallo tras la aceptación remota y antes de guardar la confirmación puede causar reenvío. El monitor no notifica eventos vencidos.

### Despliegue manual: Lightsail, Neon, SES y Vercel

Desde `Nubira-server`, con Docker, AWS CLI y el complemento `lightsailctl` instalados, compila y sube la imagen a un servicio que ya hayas creado:

```sh
docker build --platform linux/amd64 --provenance=false -t nubira-api:demo .
aws lightsail push-container-image --region REGION --service-name SERVICIO \
  --label nubira-api --image nubira-api:demo
```

1. Crea el despliegue con la imagen que devuelve el comando, escala **1**, puerto **HTTP 8000** y ruta de salud **`/health`** (código `200`). El endpoint público de Lightsail ofrece HTTPS. El plan **Medium** da 2 GB; consulta [el coste actual](https://aws.amazon.com/lightsail/pricing/) antes de crearlo.
2. En las variables del contenedor, añade la URL de Neon (`DATABASE_URL`) con los parámetros TLS que proporciona Neon. Para el correo usa `MAIL_PROVIDER=smtp`, `SMTP_HOST`, `SMTP_PORT=587`, `SMTP_USER`, `SMTP_PASS` y `MAIL_FROM`. El endpoint SMTP y las credenciales deben ser de la misma región de SES. Si SES está en el sandbox, verifica también los correos destinatarios de la demo.
3. Configura `FRONTEND_URL` con el origen público que usarán los enlaces de correo y `CORS_ORIGINS` con los orígenes exactos permitidos. En Vercel configura `NUBIRA_PUBLIC_URL` con el mismo origen público y `NUBIRA_API_URL` con el endpoint HTTPS de Lightsail; después vuelve a desplegar el frontend.
4. Comprueba `/health`, una consulta sin cuenta, la llegada del correo de verificación, el acceso, un evento guardado tras reiniciar el contenedor y el restablecimiento de contraseña. Activa alertas en un evento futuro: el correo de alerta solo se envía si cambia lo suficiente la evaluación climatológica.

### Pruebas

Instala `requirements-test.txt` en el entorno que ya tiene PyTorch CPU y ejecuta `TZ=UTC OMP_NUM_THREADS=2 python -m pytest`. La suite por defecto usa SQLite temporal y proveedores simulados. GitHub Actions repite la misma suite con PostgreSQL local a través de `NUBIRA_TEST_POSTGRES_URL`. Para repetir esa prueba localmente usa solo una base descartable: la suite crea y borra un esquema propio por prueba. No se usa Neon ni SES reales.
