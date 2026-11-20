import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

# Directorio raíz del proyecto (el mismo donde viven los scripts y se descarga el .nc)
PROJECT_DIR = Path(__file__).resolve().parent

# Rutas automáticas dentro del mismo directorio del proyecto
DATASET_FILENAME = "clima_diario_global_1980_2022.nc"
DATASET_PATH = PROJECT_DIR / DATASET_FILENAME
ARTIFACT_DIR = PROJECT_DIR / "artifacts"


def _artifact_path(env_var: str, filename: str) -> Path:
    return Path(os.environ.get(env_var, ARTIFACT_DIR / filename))


CONTEXT_PATH = _artifact_path("NUBIRA_CLIMATE_CONTEXT_PATH", "climate_context.npz")
BEST_MODEL_PATH = _artifact_path("NUBIRA_MODEL_PATH", "best_weather_risk_model.pt")


def database_url() -> str | None:
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        if not url.startswith(("postgresql://", "postgres://")):
            raise RuntimeError("DATABASE_URL debe ser una URL de PostgreSQL.")
        return url
    if os.environ.get("NUBIRA_REQUIRE_POSTGRES") == "1":
        raise RuntimeError("DATABASE_URL es obligatoria para este despliegue.")
    return None


def monitor_interval() -> int:
    value = os.environ.get("NUBIRA_MONITOR_INTERVAL", "0")
    if not value.isascii() or not value.isdecimal():
        raise RuntimeError("NUBIRA_MONITOR_INTERVAL debe ser 0 o un intervalo de al menos 60 segundos.")
    interval = int(value)
    if 0 < interval < 60:
        raise RuntimeError("NUBIRA_MONITOR_INTERVAL debe ser 0 o un intervalo de al menos 60 segundos.")
    return interval


def env_value(name: str, legacy: str, default: str = "") -> str:
    return os.environ.get(name, os.environ.get(legacy, default))


def is_local_origin(origin: str) -> bool:
    return urlsplit(origin).hostname in {"localhost", "127.0.0.1", "::1"}


def web_origin(value: str) -> str:
    origin = value.strip().rstrip("/")
    parts = urlsplit(origin)
    host, port = parts.hostname, parts.port
    if (not host or "*" in host or parts.username is not None or parts.password is not None or parts.path
            or re.search(r"[\s\\?#]", origin)
            or (parts.scheme != "https" and not (parts.scheme == "http" and is_local_origin(origin)))):
        raise ValueError("Se requiere un origen HTTPS (o HTTP en localhost).")
    authority = f"[{host}]" if ":" in host else host
    if port is not None and port != (443 if parts.scheme == "https" else 80):
        authority += f":{port}"
    return f"{parts.scheme}://{authority}"


def public_origin() -> str:
    try:
        return web_origin(env_value("FRONTEND_URL", "NUBIRA_PUBLIC_URL", "http://localhost:3000"))
    except ValueError:
        raise RuntimeError("FRONTEND_URL debe ser un origen HTTPS (o HTTP en localhost).") from None


def cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS")
    if raw is None:
        return [public_origin()]
    try:
        return list(dict.fromkeys(web_origin(value) for value in raw.split(",") if value.strip()))
    except ValueError:
        raise RuntimeError("CORS_ORIGINS debe contener orígenes HTTP(S) explícitos, separados por coma.") from None


@dataclass(frozen=True)
class MailSettings:
    mode: str
    sender: str
    origin: str
    directory: Path | None = None
    host: str = ""
    port: int = 587
    user: str = field(default="", repr=False)
    password: str = field(default="", repr=False)
    implicit_tls: bool = False


def mail_settings() -> MailSettings:
    origin = public_origin()
    local = is_local_origin(origin)
    mode = env_value("MAIL_PROVIDER", "NUBIRA_MAIL_MODE") or ("console" if local else "disabled")
    sender = env_value("MAIL_FROM", "NUBIRA_MAIL_FROM")
    if mode == "console" and local:
        return MailSettings(mode, sender or "nubira@localhost", origin)
    if mode == "file" and local and (directory := os.environ.get("NUBIRA_MAIL_DIR")):
        return MailSettings(mode, sender or "nubira@localhost", origin, directory=Path(directory))
    if mode == "smtp" and origin.startswith("https://"):
        host = env_value("SMTP_HOST", "NUBIRA_SMTP_HOST").strip()
        user = env_value("SMTP_USER", "NUBIRA_SMTP_USER")
        password = env_value("SMTP_PASS", "NUBIRA_SMTP_PASSWORD")
        implicit_tls = os.environ.get("NUBIRA_SMTP_IMPLICIT_TLS") == "1"
        port = env_value("SMTP_PORT", "NUBIRA_SMTP_PORT") or ("465" if implicit_tls else "587")
        if (host and user and password and re.fullmatch(r"[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+", sender)
                and re.fullmatch(r"[0-9]{1,5}", port) and 1 <= int(port) <= 65535):
            return MailSettings(mode, sender, origin, host=host, port=int(port), user=user,
                                password=password, implicit_tls=implicit_tls)
    raise RuntimeError("El servicio de correo no está configurado.")


# Fuente pública en Google Cloud Storage (WeatherBench 2 / ERA5 64x32 cada 6h, 1959 a enero 2023)
WB2_ZARR_URL = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"

# Rango de años completos (43 años enteros: 1980 a 2022)
START_YEAR = 1980
END_YEAR = 2022

# División cronológica estricta por años completos (cero fuga de datos)
TRAIN_END_YEAR = 2016  # Entrenamiento: 1980 - 2016 (37 años)
VAL_END_YEAR = 2019    # Validación: 2017 - 2019 (3 años)
                       # Verificación: 2020 - 2022 (3 años inéditos)

# Umbrales meteorológicos diarios para etiquetas probabilísticas
RAIN_THRESHOLD_MM = 1.0       # Día con lluvia >= 1.0 mm acumulado en 24h
WIND_THRESHOLD_MS = 7.5       # Día con viento fuerte >= 7.5 m/s (~27 km/h) promedio diario
TEMP_STD_MULTIPLIER = 1.25    # Desviaciones sobre promedio local de Tmax / Tmin
HOT_MIN_FLOOR_C = 30.0        # Piso mínimo para considerar calor adverso
HOT_ABS_CEILING_C = 35.0      # Calor absoluto adverso garantizado (>= 35 °C)
COLD_MAX_CEILING_C = 5.0      # Techo máximo para considerar frío adverso
COLD_ABS_FLOOR_C = 0.0        # Frío absoluto adverso garantizado (<= 0 °C)

# Configuración espacial y de entrenamiento
PATCH_SIZE = 5                # Vecindario espacial 5 x 5 alrededor de la coordenada
CLIM_WINDOW_DAYS = 7          # Suavizado circular +-7 días para climatología histórica base
BATCH_SIZE = 512
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SAMPLES_PER_EPOCH = 300_000
VAL_SAMPLES = 60_000
TEST_SAMPLES = 80_000
NUM_WORKERS = 4
SEED = 42
