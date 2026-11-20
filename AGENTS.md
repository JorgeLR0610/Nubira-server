# AGENTS.md — Nubira Weather (Despliegue Simplificado en AWS Lightsail + Neon)

Este documento define el contexto del proyecto, el estado actual, las restricciones de arquitectura y el plan de ejecución exacto para migrar el backend `Nubira-server` hacia **AWS Lightsail** con base de datos en **Neon (PostgreSQL)**, priorizando la máxima simplicidad y estabilidad para una presentación académica sin tráfico real.

---

## 1. Contexto y Estado Actual

El ecosistema de **Nubira Weather** consta de dos repositorios:
1. **Frontend (`NubiraWeather`):** Aplicación en Next.js desplegada en Vercel. Consume los endpoints de clima, lugares, cuentas (`/cuenta/verificar`, `/cuenta/restablecer`) y eventos (`/eventos`). Ya desplegada en `https://nubira-weather.vercel.app`
2. **Backend (`Nubira-server`):** Servidor en FastAPI (`appPowerFastAPI.py`, `start.py`) que gestiona autenticación (`accounts.py`), eventos (`events.py`), monitoreo (`monitor.py`), lugares (`places.py`), correos (`mailer.py`), persistencia (`storage.py`) e inferencia climática con PyTorch (`model.py`, `predict.py`, `weather_service.py`).

### Problemas actuales a resolver
* **Persistencia efímera:** Actualmente se usa SQLite local (`tmp/weather.db` y `workspace/weather.db`), lo que impide persistir datos de forma fiable entre despliegues.
* **Despliegue inestable:** La versión en Hugging Face Spaces está desactualizada (commit `6f89071`) y entra en suspensión por inactividad.
* **Correos transaccionales:** Se integrará un dominio propio con **Amazon SES** en `mailer.py`.
* **Higiene del repositorio:** Existen bases de datos rastreadas en Git, un archivo duplicado (`requeriments.txt`) y riesgo de exposición de `.env`.

### Decisión de Arquitectura (NO MODIFICAR)
* **Simplicidad:** Es para una demostración académica. **NO** separar microservicios, **NO** usar Amazon S3 y **NO** agregar infraestructura compleja (sin EC2, sin VPC, sin roles IAM).
* **Backend + Modelo unificados en AWS Lightsail:** Todo correrá en un único contenedor Docker en **AWS Lightsail** (instancia/servicio de 2 GB de RAM) usando **PyTorch en CPU**.
* **Artefactos locales empaquetados:** Los archivos reales del modelo (`artifacts/best_weather_risk_model.pt` ~433 KB y `artifacts/climate_context.npz` ~19.4 MB) pesan menos de 20 MB en total y se incluirán **directamente dentro del contenedor Docker**, omitiendo descargas remotas en producción.
* **Base de datos:** **Neon** (PostgreSQL serverless gratuito), conectado mediante `DATABASE_URL`.
* **Correos:** **Amazon SES vía SMTP** (o el mecanismo más simple que ya soporte `mailer.py`), configurado solo con variables de entorno.

---

## 2. Reglas Estrictas para el Agente

1. **Seguridad:** NUNCA commitear archivos `.env`, contraseñas, llaves SMTP/API ni archivos `*.db`. Toda configuración sensible debe leerse desde variables de entorno (`config.py`).
2. **Sin dependencias de AWS S3:** NO agregar `boto3` para S3 ni lógica de buckets. El servidor debe arrancar usando los archivos locales en `artifacts/`.
3. **Contratos de API:** NO romper los endpoints, rutas ni formatos JSON actuales que consume el frontend `NubiraWeather` en Vercel.
4. **Pruebas aisladas:** `pytest` (`tests/test_api.py`) debe seguir ejecutándose localmente en segundos sin depender de internet, Neon real ni SES real.
5. **Verificación obligatoria:** Al terminar cada tarea, ejecutar `pytest` y confirmar que todas las pruebas pasen sin regresiones.

---

## 3. Plan de Tareas (Ejecución Secuencial)

### Tarea 1: Limpieza de repositorio y reglas de exclusión
* [ ] Eliminar el archivo residual `requeriments.txt` (mantener los válidos: `requirements.txt`, `requirements-api.txt`, `requirements-test.txt`).
* [ ] Asegurar que `.gitignore` y `.dockerignore` excluyan:
  * `.env` y `.env.*` (excepto `.env.example`)
  * `*.db`, `tmp/*.db`, `workspace/*.db`
  * `__pycache__/`, `.pytest_cache/`
* [ ] **Importante:** Verificar que `.dockerignore` **NO** excluya la carpeta `artifacts/`, ya que los pesos reales del modelo se empaquetarán dentro de la imagen Docker.
* [ ] Dejar listo el retiro de seguimiento en Git de `tmp/weather.db` y `workspace/weather.db`.

### Tarea 2: Carga directa de artefactos locales (Sin S3)
* [ ] Revisar `config.py`, `predict.py` y `fetch_artifacts.py`.
* [ ] Asegurar que la aplicación cargue el modelo directamente desde `artifacts/best_weather_risk_model.pt` y `artifacts/climate_context.npz`.
* [ ] Validar en `fetch_artifacts.py` (o al iniciar) que, si ambos archivos existen localmente y superan un tamaño mínimo válido (ej. `> 1024` bytes, descartando punteros Git LFS de ~130 bytes), **no intente realizar ninguna descarga remota** y continúe el arranque inmediatamente.

### Tarea 3: Migración de persistencia (SQLite a PostgreSQL / Neon)
* [ ] Auditar `storage.py`, `accounts.py`, `events.py`, `monitor.py` y cualquier módulo con consultas SQL.
* [ ] Agregar el driver de PostgreSQL (`psycopg[binary]` o `psycopg2-binary`) a `requirements-api.txt` y `requirements.txt`.
* [ ] Refactorizar `storage.py` para usar PostgreSQL cuando `DATABASE_URL` inicie con `postgresql://` o `postgres://` (compatible con `sslmode=require` de Neon).
* [ ] Adaptar la sintaxis SQL incompatible entre SQLite y PostgreSQL (placeholders `?` vs `%s`, `AUTOINCREMENT` vs `SERIAL`/`IDENTITY`, tipos booleanos/fechas y cláusulas `UPSERT`).
* [ ] Manejar de forma resiliente las reconexiones (ya que Neon pausa conexiones inactivas tras unos minutos; abrir conexiones por operación o validar la conexión antes de consultar).
* [ ] Asegurar la creación idempotente de tablas al iniciar (`CREATE TABLE IF NOT EXISTS ...`).
* [ ] Conservar el respaldo de SQLite local/en memoria únicamente para que `tests/test_api.py` corra aislado sin necesitar Neon.

### Tarea 4: Correos transaccionales (Amazon SES vía SMTP)
* [ ] Revisar `mailer.py` y `config.py`.
* [ ] Asegurar que `mailer.py` soporte envío de correos (verificación de cuenta y restablecimiento de contraseña) mediante **SMTP con TLS** (puerto `587`), compatible directamente con las credenciales SMTP de **Amazon SES** (`SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`, `MAIL_FROM`).
* [ ] Mantener un modo `console` / mock cuando no se configuren credenciales SMTP en desarrollo o pruebas automatizadas.

### Tarea 5: Optimización del `Dockerfile` para AWS Lightsail
* [ ] Revisar `Dockerfile`, `start.py` y `requirements-api.txt`.
* [ ] Usar `python:3.12-slim` e instalar **PyTorch exclusivo para CPU** (`--extra-index-url https://download.pytorch.org/whl/cpu`) para reducir drásticamente el peso de la imagen y el consumo de RAM dentro de los 2 GB de Lightsail.
* [ ] Copiar explícitamente la carpeta `artifacts/` dentro de la imagen Docker.
* [ ] Configurar el servidor para escuchar en `0.0.0.0` y en el puerto definido por `PORT` (por defecto `8000`).

### Tarea 6: CORS, Healthcheck, `.env.example` y Pruebas
* [ ] Revisar `appPowerFastAPI.py` para leer `CORS_ORIGINS` desde variables de entorno (permitiendo la URL de Vercel en producción y `http://localhost:3000` en local).
* [ ] Verificar o agregar un endpoint ligero `GET /health` que devuelva HTTP `200` (útil para los healthchecks de AWS Lightsail).
* [ ] Actualizar `.env.example` documentando todas las variables requeridas (sin valores reales).
* [ ] Actualizar y ejecutar `tests/test_api.py`, y validar que `.github/workflows/checks.yml` pase en verde.

---

## 4. Variables de Entorno Esperadas (`.env.example`)

| Variable | Descripción | Ejemplo |
| :--- | :--- | :--- |
| `DATABASE_URL` | Cadena de conexión de Neon (PostgreSQL) | `postgresql://user:pass@ep-xxxx.us-east-2.aws.neon.tech/neondb?sslmode=require` |
| `MAIL_PROVIDER` | Modo de correo (`smtp` o `console`) | `smtp` |
| `SMTP_HOST` | Servidor SMTP de Amazon SES | `email-smtp.us-east-1.amazonaws.com` |
| `SMTP_PORT` | Puerto SMTP con STARTTLS | `587` |
| `SMTP_USER` | Usuario SMTP generado en Amazon SES | `AKIA...` |
| `SMTP_PASS` | Contraseña SMTP generada en Amazon SES | `B...` |
| `SMTOP_FROM` | Remitente verificado en Amazon SES | `no-reply@nubiraweather.date` |
| `CORS_ORIGINS` | Orígenes permitidos separados por coma | `https://nubiraweather.date,http://localhost:3000` |
| `FRONTEND_URL` | URL del frontend para enlaces de correo | `https://nubiraweather.date` |
| `PORT` | Puerto HTTP del contenedor | `8000` |
| `NUBIRA_GEOAPIFY_API_KEY` | API Key de Geoapify | `B...` |

---

## 5. Checklist del Usuario Humano (Infraestructura Externa)

*Pasos manuales fuera del alcance del agente:*

- [x] **Control de costos:** Configurar alerta en **AWS Budgets** (ej. 15–20 USD/mes) para cuidar los créditos.
- [x] **Base de datos en Neon:** Crear proyecto gratuito en [neon.tech](https://neon.tech) y copiar `DATABASE_URL`.
- [x] **Dominio y Amazon SES:** Comprar dominio, verificar registros DNS (DKIM, SPF, DMARC) en Amazon SES y generar credenciales SMTP desde la consola de SES. *(Nota: si solo probarán con sus propios correos en la presentación, basta con verificar esos correos destino en SES sin siquiera esperar a salir del Sandbox).*
- [ ] **Despliegue en AWS Lightsail:**
   * **Opción recomendada (HTTPS automático):** Crear un **Lightsail Container Service** (plan *Small* o *Medium* con 2 GB de RAM), subir la imagen Docker con AWS CLI/Lightsail CLI e inyectar las variables de entorno. Lightsail entrega automáticamente una URL HTTPS lista para Vercel.
   * **Alternativa (VM):** Crear una instancia Linux de Lightsail (2 GB RAM), clonar el repo con los `artifacts/` y correr el contenedor Docker.
- [ ] **Vercel (`NubiraWeather`):** Actualizar la variable de entorno del backend con la nueva URL HTTPS de Lightsail y probar el flujo completo antes de la presentación.