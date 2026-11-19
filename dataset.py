import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset

from config import (
    DATASET_PATH,
    CONTEXT_PATH,
    TRAIN_END_YEAR,
    RAIN_THRESHOLD_MM,
    WIND_THRESHOLD_MS,
    TEMP_STD_MULTIPLIER,
    HOT_MIN_FLOOR_C,
    HOT_ABS_CEILING_C,
    COLD_MAX_CEILING_C,
    COLD_ABS_FLOOR_C,
    PATCH_SIZE,
    CLIM_WINDOW_DAYS,
)


def extract_doy_and_year(times_np: np.ndarray):
    """Extrae el día del año en [0, 365] y el año calendario entero."""
    dates_d = times_np.astype("datetime64[D]")
    years_dt = dates_d.astype("datetime64[Y]")
    years = years_dt.astype(int) + 1970
    doy_idx = np.clip((dates_d - years_dt.astype("datetime64[D]")).astype(int), 0, 365)
    return doy_idx, years


def build_coord_time_vector(lat: float, lon: float, doy_idx: int, year: int) -> np.ndarray:
    """Construye el vector continuo de 10 dimensiones (esférico + 3 armónicos + tendencia anual)."""
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    x_sphere = np.cos(lat_rad) * np.cos(lon_rad)
    y_sphere = np.cos(lat_rad) * np.sin(lon_rad)
    z_sphere = np.sin(lat_rad)

    base_angle = 2.0 * np.pi * (float(doy_idx) / 365.25)
    sin1, cos1 = np.sin(base_angle), np.cos(base_angle)
    sin2, cos2 = np.sin(2.0 * base_angle), np.cos(2.0 * base_angle)
    sin3, cos3 = np.sin(3.0 * base_angle), np.cos(3.0 * base_angle)

    year_norm = (float(year) - 2000.0) / 30.0

    return np.array(
        [x_sphere, y_sphere, z_sphere, sin1, cos1, sin2, cos2, sin3, cos3, year_norm],
        dtype=np.float32,
    )


def extract_clim_patch(
    clim_table_norm: np.ndarray,
    doy_idx: int,
    lat_idx: int,
    lon_idx: int,
    patch_size: int = PATCH_SIZE,
) -> np.ndarray:
    """Extrae el parche (7, 5, 5) para un día del año y celda dados con envoltura circular en longitud."""
    radius = patch_size // 2
    n_lat, n_lon = clim_table_norm.shape[2], clim_table_norm.shape[3]
    lat_indices = np.clip(np.arange(lat_idx - radius, lat_idx + radius + 1), 0, n_lat - 1)
    lon_indices = (np.arange(lon_idx - radius, lon_idx + radius + 1)) % n_lon
    day_map = clim_table_norm[doy_idx]  # (7, Lat, Lon)
    return day_map[:, lat_indices, :][:, :, lon_indices]


def load_daily_dataset_and_context():
    """
    Carga clima_diario_global_1980_2022.nc desde el directorio del proyecto,
    construye las etiquetas binarias diarias y calcula/guarda climate_context.npz
    estrictamente con el bloque de entrenamiento (1980-2016).
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos en {DATASET_PATH}.\n"
            "Ejecuta primero: python download_data.py"
        )

    ds = xr.open_dataset(DATASET_PATH)
    ds = ds.transpose("time", "latitude", "longitude").sortby(["latitude", "longitude"])

    temp_mean = ds["daily_temp_mean"].values.astype(np.float32)
    temp_max = ds["daily_temp_max"].values.astype(np.float32)
    temp_min = ds["daily_temp_min"].values.astype(np.float32)
    precip_mm = np.clip(ds["daily_precip_mm"].values.astype(np.float32), 0.0, None)
    wind_mean = np.clip(ds["daily_wind_mean"].values.astype(np.float32), 0.0, None)
    mslp_mean = ds["daily_mslp_mean"].values.astype(np.float32)

    times = ds["time"].values
    lats = ds["latitude"].values.astype(np.float32)
    lons = ds["longitude"].values.astype(np.float32)
    ds.close()

    for name, arr in [
        ("daily_temp_mean", temp_mean),
        ("daily_temp_max", temp_max),
        ("daily_temp_min", temp_min),
        ("daily_precip_mm", precip_mm),
        ("daily_wind_mean", wind_mean),
        ("daily_mslp_mean", mslp_mean),
    ]:
        if not np.isfinite(arr).all():
            raise ValueError(f"Se detectaron valores NaN o infinitos en {name}.")

    doy_idx, years = extract_doy_and_year(times)
    train_mask = years <= TRAIN_END_YEAR

    # 1. Umbrales locales de temperaturas adversas usando SOLO años de entrenamiento (1980-2016)
    mean_tmax = temp_max[train_mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    std_tmax = temp_max[train_mask].std(axis=0, dtype=np.float64).astype(np.float32)
    mean_tmin = temp_min[train_mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    std_tmin = temp_min[train_mask].std(axis=0, dtype=np.float64).astype(np.float32)

    hot_threshold = np.minimum(
        np.maximum(mean_tmax + TEMP_STD_MULTIPLIER * std_tmax, HOT_MIN_FLOOR_C),
        HOT_ABS_CEILING_C,
    ).astype(np.float32)

    cold_threshold = np.maximum(
        np.minimum(mean_tmin - TEMP_STD_MULTIPLIER * std_tmin, COLD_MAX_CEILING_C),
        COLD_ABS_FLOOR_C,
    ).astype(np.float32)

    # 2. Etiquetas binarias diarias (T, 3, Lat, Lon): [lluvia, viento_fuerte, temp_adversa]
    y_rain = (precip_mm >= RAIN_THRESHOLD_MM).astype(np.float32)
    y_wind = (wind_mean >= WIND_THRESHOLD_MS).astype(np.float32)
    y_adverse_temp = (
        (temp_max >= hot_threshold[None, :, :]) |
        (temp_min <= cold_threshold[None, :, :])
    ).astype(np.float32)

    labels = np.stack([y_rain, y_wind, y_adverse_temp], axis=1)  # (T, 3, Lat, Lon)

    # 3. Tabla de contexto climatológico diario (366, 7, Lat, Lon) SOLO con entrenamiento
    n_lat, n_lon = len(lats), len(lons)
    sum_raw = np.zeros((366, 6, n_lat, n_lon), dtype=np.float64)
    sum_tmean_sq = np.zeros((366, n_lat, n_lon), dtype=np.float64)
    day_counts = np.zeros(366, dtype=np.float64)
    precip_log = np.log1p(precip_mm)
    train_indices = np.where(train_mask)[0]

    for t in train_indices:
        d = int(doy_idx[t])
        day_counts[d] += 1.0
        sum_raw[d, 0] += temp_mean[t]
        sum_tmean_sq[d] += temp_mean[t].astype(np.float64) ** 2
        sum_raw[d, 1] += temp_max[t]
        sum_raw[d, 2] += temp_min[t]
        sum_raw[d, 3] += wind_mean[t]
        sum_raw[d, 4] += precip_log[t]
        sum_raw[d, 5] += mslp_mean[t]

    clim_table = np.zeros((366, 7, n_lat, n_lon), dtype=np.float32)
    offsets = list(range(-CLIM_WINDOW_DAYS, CLIM_WINDOW_DAYS + 1))
    for d in range(366):
        win_days = [(d + off) % 366 for off in offsets]
        total_c = day_counts[win_days].sum()
        mean_6 = sum_raw[win_days].sum(axis=0) / total_c
        mean_t_sq = sum_tmean_sq[win_days].sum(axis=0) / total_c
        std_t = np.sqrt(np.maximum(mean_t_sq - (mean_6[0] ** 2), 1e-6))

        clim_table[d, 0] = mean_6[0].astype(np.float32)  # media daily_temp_mean
        clim_table[d, 1] = std_t.astype(np.float32)      # std daily_temp_mean
        clim_table[d, 2] = mean_6[1].astype(np.float32)  # media daily_temp_max
        clim_table[d, 3] = mean_6[2].astype(np.float32)  # media daily_temp_min
        clim_table[d, 4] = mean_6[3].astype(np.float32)  # media daily_wind_mean
        clim_table[d, 5] = mean_6[4].astype(np.float32)  # media log1p(precip_mm)
        clim_table[d, 6] = mean_6[5].astype(np.float32)  # media daily_mslp_mean

    clim_means = clim_table.mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    clim_stds = clim_table.std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    clim_stds = np.where(clim_stds < 1e-6, 1.0, clim_stds)
    clim_table_norm = (clim_table - clim_means[None, :, None, None]) / clim_stds[None, :, None, None]

    # Guardar archivo ligero de contexto (~20 MB) para inferencia rápida sin abrir el .nc
    np.savez_compressed(
        CONTEXT_PATH,
        clim_table_norm=clim_table_norm,
        lats=lats,
        lons=lons,
        hot_threshold=hot_threshold,
        cold_threshold=cold_threshold,
        clim_means=clim_means,
        clim_stds=clim_stds,
    )
    return labels, doy_idx, years, lats, lons, clim_table_norm


class DailyRiskDataset(Dataset):
    def __init__(
        self,
        labels: np.ndarray,
        doy_idx: np.ndarray,
        years: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        clim_table_norm: np.ndarray,
        num_samples: int,
        deterministic: bool = False,
        seed: int = 42,
    ):
        self.labels = labels
        self.doy_idx = doy_idx
        self.years = years
        self.lats = lats
        self.lons = lons
        self.clim_table_norm = clim_table_norm
        self.num_samples = num_samples
        self.deterministic = deterministic
        self.n_days = len(doy_idx)
        self.n_lat = len(lats)
        self.n_lon = len(lons)

        if self.deterministic:
            rng = np.random.default_rng(seed)
            self.fixed_t = rng.integers(0, self.n_days, size=num_samples)
            self.fixed_lat = rng.integers(0, self.n_lat, size=num_samples)
            self.fixed_lon = rng.integers(0, self.n_lon, size=num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        if self.deterministic:
            t = int(self.fixed_t[idx])
            lat_i = int(self.fixed_lat[idx])
            lon_i = int(self.fixed_lon[idx])
        else:
            t = int(np.random.randint(0, self.n_days))
            lat_i = int(np.random.randint(0, self.n_lat))
            lon_i = int(np.random.randint(0, self.n_lon))

        d = int(self.doy_idx[t])
        yr = int(self.years[t])

        x_patch = extract_clim_patch(self.clim_table_norm, d, lat_i, lon_i, PATCH_SIZE)
        x_meta = build_coord_time_vector(
            lat=float(self.lats[lat_i]),
            lon=float(self.lons[lon_i]),
            doy_idx=d,
            year=yr,
        )
        y = self.labels[t, :, lat_i, lon_i]  # [y_rain, y_wind, y_adverse_temp]

        return (
            torch.from_numpy(np.ascontiguousarray(x_patch)),
            torch.from_numpy(x_meta),
            torch.from_numpy(np.ascontiguousarray(y)),
        )

