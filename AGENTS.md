# AGENTS.md — Mantenimiento de Nubira-server

## Estado y arquitectura

La migración a AWS Lightsail está completada. Nubira Weather evalúa riesgos climatológicos antes de un evento al aire libre. La consulta es pública; para guardar eventos y activar alertas se requiere una cuenta con correo verificado.

- **Frontend:** [`NubiraWeather`](https://github.com/JesusPichardo15/NubiraWeather), Next.js en [Vercel](https://nubira-weather.vercel.app). Sus rutas `/api/*` hacen de proxy del lado servidor; el navegador no llama directamente a FastAPI.
- **Backend:** [api.nubiraweather.date](https://api.nubiraweather.date), VM Linux de Lightsail de 2 GB. `docker-compose.yml` levanta Caddy y un único contenedor FastAPI con PyTorch CPU y el monitor integrado. Caddy gestiona el TLS y es el único servicio que publica puertos. Uvicorn escucha en `8000` dentro de la red de Docker y confía solo en la IP de Caddy.
- **Persistencia y correo:** Neon (PostgreSQL) vía `DATABASE_URL` con TLS y Amazon SES vía SMTP STARTTLS en `587`. SQLite y el correo `file` o `console` son para desarrollo y pruebas, no un respaldo silencioso en producción.
- **Configuración:** consulta [`.env.example`](.env.example) y el [README](README.md#operación-en-lightsail). El remitente SMTP se configura con `MAIL_FROM`. `FRONTEND_URL` corresponde al origen de la web y debe coincidir con `NUBIRA_PUBLIC_URL` en Vercel; `NUBIRA_API_DOMAIN` corresponde al host de la API.

## Reglas de mantenimiento

1. **Preserva los contratos.** Mantén rutas, códigos de estado, formas JSON y comportamiento de sesión de `/clima`, `/places`, `/auth` y `/events`. No relajes la verificación de correo, el control de acceso a eventos, los límites de solicitudes ni las garantías de la cola de alertas. Las predicciones son climatológicas, no pronósticos en tiempo real.
2. **Mantén el despliegue simple.** Conserva API, inferencia y monitor en un único proceso servidor del contenedor de la API. No agregues microservicios, S3 (`boto3` ni lógica de buckets) ni infraestructura adicional. No publiques `8000`, no amplíes la confianza en cabeceras de proxy y conserva los volúmenes de Caddy al detener o actualizar el despliegue.
3. **No descargues artefactos al arrancar.** La imagen empaqueta `artifacts/best_weather_risk_model.pt` y `artifacts/climate_context.npz`. Mantén ambos binarios en Git y en el contexto de Docker. `python fetch_artifacts.py --check` comprueba los archivos empaquetados sin red.
4. **Protege credenciales y datos locales.** No versiones ni imprimas `.env`, credenciales de Neon o SMTP, claves de API, bases de datos locales ni mensajes `.eml`. Lee los valores sensibles del entorno mediante `config.py` y documenta nombres y ejemplos ficticios en `.env.example`. No despliegues ni cambies DNS o configuración de producción sin autorización explícita.
5. **Mantén la persistencia compatible.** Usa `storage.py` para encapsular PostgreSQL y SQLite. Conserva la inicialización idempotente, las transacciones y las conexiones por operación. No cambies a SQLite si PostgreSQL falla. Mantén `/health` ligero y sin consultas a Neon o SES.

## Verificación

- Al terminar cada tarea, ejecuta `TZ=UTC OMP_NUM_THREADS=2 python -m pytest` con Python 3.12, PyTorch CPU y `requirements-test.txt`. La suite predeterminada debe correr sin acceso a internet ni a Neon o SES reales.
- Para cambios de persistencia, ejecuta también la suite con `NUBIRA_TEST_POSTGRES_URL` apuntando a una base local descartable. [GitHub Actions](.github/workflows/checks.yml) verifica los artefactos y ejecuta las dos variantes.
- Si cambias Compose o Caddy, valida el archivo con `docker compose config --quiet` y el `Caddyfile` con `caddy validate`. Comprueba que `/health` responde a través de Caddy y que el puerto `8000` sigue sin publicarse. El checklist de operación está en el [README](README.md#operación-en-lightsail).
