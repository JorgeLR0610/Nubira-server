from pathlib import Path

# Directorio raíz del proyecto (el mismo donde viven los scripts y se descarga el .nc)
PROJECT_DIR = Path(__file__).resolve().parent

# Rutas automáticas dentro del mismo directorio del proyecto
DATASET_FILENAME = "clima_diario_global_1980_2022.nc"
DATASET_PATH = PROJECT_DIR / DATASET_FILENAME
CONTEXT_PATH = PROJECT_DIR / "climate_context.npz"
BEST_MODEL_PATH = PROJECT_DIR / "best_weather_risk_model.pt"

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

