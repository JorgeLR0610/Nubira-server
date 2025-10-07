import requests
from datetime import datetime
import pandas as pd
import os

# 🌍 Descargar datos del clima (NASA POWER)
def fetch_power_daily(lat, lon, start, end, parameters=['T2M_MAX','T2M_MIN','PRECTOTCORR', 'PRECSNO', 'WS2M_MAX']):
    base = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "parameters": ",".join(parameters),
        "community": "AG",
        "format": "JSON"
    }
    resp = requests.get(base, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

# 📁 Guardar como CSV en la raíz del proyecto
def convert_to_csv(api_response, filename='datos.csv', filter_month=None, filter_day=None):
    parameters_data = api_response['properties']['parameter']
    first_param = list(parameters_data.keys())[0]
    dates = list(parameters_data[first_param].keys())
    
    result = []
    for date in dates:
        year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:8])
        
        # Filtros opcionales
        if filter_month is not None and month != filter_month:
            continue
        if filter_day is not None and day != filter_day:
            continue
        
        daily_dict = {'year': year, 'month': month, 'day': day}
        for param_name, param_values in parameters_data.items():
            daily_dict[param_name] = param_values.get(date)
        result.append(daily_dict)
    
    df = pd.DataFrame(result)

    # ✅ Guardar en la raíz del proyecto (un nivel arriba de neuralNetwork)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    save_path = os.path.join(project_root, filename)

    df.to_csv(save_path, index=False)
    print(f"✅ Archivo CSV creado correctamente en: {save_path}")

    return df