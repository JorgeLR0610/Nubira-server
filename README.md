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
python fetch_artifacts.py
NUBIRA_MAIL_MODE=file NUBIRA_MAIL_DIR="$PWD/workspace/mail" OMP_NUM_THREADS=2 python start.py
```

También puedes ejecutar `git lfs pull` para materializar los archivos del repositorio en lugar de usar el script. Este verifica tamaño y SHA-256 antes de usar la copia del espejo. Se pueden montar otros artefactos confiables con `NUBIRA_MODEL_PATH` y `NUBIRA_CLIMATE_CONTEXT_PATH`. `/health` responde `503` si faltan los artefactos o son punteros de Git LFS. `/clima` acepta el ejemplo de `../input.json`; las fechas deben ser futuras. No hay simulación cuando el modelo no está disponible.

`NUBIRA_DB_PATH` apunta a una base de datos SQLite persistente (por defecto `workspace/nubira.sqlite3`), distinta de la base del prototipo. Usa almacenamiento local persistente para SQLite, no un sistema de archivos de red. No se importan automáticamente datos temporales del prototipo.

### Autocompletado de direcciones

Configura `NUBIRA_GEOAPIFY_API_KEY` solo en el entorno de la API y reiníciala. Las sugerencias se consultan en español, priorizan México sin excluir otros países y tienen un límite compartido de 1.000 solicitudes al proveedor por ventana de 24 horas (además del límite de ráfaga). Las direcciones parciales se envían al proveedor y la API no las guarda. Sin clave siguen disponibles el mapa y la entrada manual. La búsqueda opcional compatible con Nominatim conserva su modo de envío explícito: nunca se usa para el autocompletado.

### Correo y reevaluación

Las variables están documentadas en [`.env.example`](.env.example); expórtalas desde el entorno o configúralas en Compose. Usa `file` solo con un origen local, o `smtp` con un origen HTTPS, remitente autorizado y credenciales del proveedor. Por defecto se usa STARTTLS (puerto 587); para TLS implícito, `NUBIRA_SMTP_IMPLICIT_TLS=1` (puerto 465). No subas credenciales ni la bandeja de correo al repositorio.

Usa el mismo entorno, ruta de base de datos y versión de artefactos para el monitor:

```sh
TZ=UTC NUBIRA_MAIL_MODE=file NUBIRA_MAIL_DIR="$PWD/workspace/mail" \
  OMP_NUM_THREADS=2 python -m monitor --interval 900
```

Sin `--interval` se ejecuta un solo ciclo (útil en cron). El umbral por defecto es 10 **puntos porcentuales** (`NUBIRA_ALERT_THRESHOLD_PP`); se compara con la evaluación inicial o la que generó el último aviso. Los días restantes no activan avisos; el modelo no incorpora un pronóstico que vaya cambiando a medida que se acerca el evento. La cola persiste y aplica reintentos; los avisos llevan identificador estable. Como con cualquier SMTP, un fallo tras la aceptación remota y antes de guardar la confirmación puede causar reenvío. El monitor no notifica eventos vencidos.

### Pruebas

Instala `requirements-test.txt` en el entorno que ya tiene PyTorch, recupera los artefactos y ejecuta `TZ=UTC OMP_NUM_THREADS=2 python -m pytest`. La prueba de inferencia compara dos llamadas de la API con los ejemplos del directorio padre. El resto usa bases y bandejas temporales y no envía correos reales.
