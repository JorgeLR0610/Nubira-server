import numpy as np
import xarray as xr
from tqdm import tqdm

from config import WB2_ZARR_URL, DATASET_PATH, START_YEAR, END_YEAR

SURFACE_VARS = [
    "2m_temperature",
    "total_precipitation_6hr",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]


def download_and_build_daily_dataset() -> None:
    if DATASET_PATH.exists():
        print(f"El archivo ya existe en: {DATASET_PATH}")
        print("Si deseas descargarlo nuevamente, elimina ese archivo y vuelve a ejecutar.")
        return

    print("Conectando a WeatherBench 2 en Google Cloud Storage (modo anónimo)...")
    ds = xr.open_zarr(WB2_ZARR_URL, storage_options={"token": "anon"})

    missing = [v for v in SURFACE_VARS if v not in ds.data_vars]
    if missing:
        raise KeyError(f"Variables no encontradas en WeatherBench 2: {missing}")

    yearly_datasets = []
    print(f"Descargando y agregando a promedio diario ({START_YEAR}-{END_YEAR})...")

    for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="Descargando años"):
        sub = ds[SURFACE_VARS].sel(time=slice(f"{year}-01-01", f"{year}-12-31")).load()
        sub = sub.transpose("time", "latitude", "longitude").sortby(["latitude", "longitude"])

        # Conversión de unidades físicas
        temp_c = sub["2m_temperature"] - 273.15
        precip_mm = (sub["total_precipitation_6hr"] * 1000.0).clip(min=0.0)
        wind_ms = np.sqrt(
            sub["10m_u_component_of_wind"] ** 2 +
            sub["10m_v_component_of_wind"] ** 2
        )
        mslp_pa = sub["mean_sea_level_pressure"]

        # Agregación de todo el día (1D): promedio/resumen diario sin división por hora
        daily_year = xr.Dataset(
            {
                "daily_temp_mean": temp_c.resample(time="1D").mean(),
                "daily_temp_max": temp_c.resample(time="1D").max(),
                "daily_temp_min": temp_c.resample(time="1D").min(),
                "daily_precip_mm": precip_mm.resample(time="1D").sum(),
                "daily_wind_mean": wind_ms.resample(time="1D").mean(),
                "daily_mslp_mean": mslp_pa.resample(time="1D").mean(),
            }
        ).astype("float32")

        yearly_datasets.append(daily_year)

    print("Combinando todos los años en un solo archivo NetCDF comprimido...")
    full_daily = xr.concat(yearly_datasets, dim="time")
    full_daily = full_daily.transpose("time", "latitude", "longitude").sortby(["latitude", "longitude"])

    encoding = {
        var: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for var in full_daily.data_vars
    }
    full_daily.to_netcdf(DATASET_PATH, encoding=encoding)

    size_mb = DATASET_PATH.stat().st_size / (1024 * 1024)
    print(f"Descarga completada con éxito en: {DATASET_PATH} ({size_mb:.1f} MB)")
    print(full_daily)


if __name__ == "__main__":
    download_and_build_daily_dataset()

