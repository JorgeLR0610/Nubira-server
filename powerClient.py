import os
import requests
from datetime import datetime
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
TEMP_DIR = PROJECT_DIR / "temp"


# 🌍 Descargar datos del clima (NASA POWER)
def fetch_power_daily(lat, lon, start, end, parameters=None):
    if parameters is None:
        parameters = ['T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'PRECSNO', 'WS2M_MAX']

    base = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start": start.strftime("%Y%m%d") if isinstance(start, (datetime,)) else str(start),
        "end": end.strftime("%Y%m%d") if isinstance(end, (datetime,)) else str(end),
        "parameters": ",".join(parameters),
        "community": "AG",
        "format": "JSON",
    }
    resp = requests.get(base, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


# 📁 Guardar y convertir a DataFrame / CSV en temp o ruta especificada
def convert_to_csv(api_response, filename=None, filter_month=None, filter_day=None):
    if not api_response or 'properties' not in api_response or 'parameter' not in api_response['properties']:
        return pd.DataFrame()

    parameters_data = api_response['properties']['parameter']
    if not parameters_data:
        return pd.DataFrame()

    first_param = list(parameters_data.keys())[0]
    dates = list(parameters_data[first_param].keys())

    result = []
    for d_str in dates:
        year = int(d_str[:4])
        month = int(d_str[4:6])
        day = int(d_str[6:8])

        # Filtros opcionales
        if filter_month is not None and month != filter_month:
            continue
        if filter_day is not None and day != filter_day:
            continue

        daily_dict = {'year': year, 'month': month, 'day': day}
        for param_name, param_values in parameters_data.items():
            daily_dict[param_name] = param_values.get(d_str)
        result.append(daily_dict)

    df = pd.DataFrame(result)

    if filename:
        save_path = Path(filename)
        if not save_path.is_absolute():
            save_path = TEMP_DIR / save_path.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

    return df
