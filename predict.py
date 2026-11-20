from datetime import date, datetime
import numpy as np
import torch

from config import BEST_MODEL_PATH, CONTEXT_PATH, PATCH_SIZE
from dataset import build_coord_time_vector, extract_clim_patch
from model import WeatherRiskProbNet


def _bilinear_grid_corners(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray):
    """Calcula los 4 nodos vecinos (lat_idx, lon_idx) y sus pesos bilineales exactos."""
    if not (-90.0 <= lat <= 90.0):
        raise ValueError("La latitud debe estar entre -90.0 y 90.0.")

    # Eje latitud (orden ascendente)
    if lat <= lats[0]:
        i0 = i1 = 0
        wy = 0.0
    elif lat >= lats[-1]:
        i0 = i1 = len(lats) - 1
        wy = 0.0
    else:
        i1 = int(np.searchsorted(lats, lat))
        i0 = i1 - 1
        wy = float((lat - lats[i0]) / (lats[i1] - lats[i0]))

    # Eje longitud en convención circular [0, 360)
    lon_360 = lon % 360.0
    wrap_span = float((360.0 - lons[-1]) + lons[0])
    if lon_360 >= lons[-1]:
        j0, j1 = len(lons) - 1, 0
        wx = float((lon_360 - lons[-1]) / wrap_span)
    elif lon_360 < lons[0]:
        j0, j1 = len(lons) - 1, 0
        wx = float((lon_360 + 360.0 - lons[-1]) / wrap_span)
    else:
        j1 = int(np.searchsorted(lons, lon_360))
        j0 = j1 - 1
        wx = float((lon_360 - lons[j0]) / (lons[j1] - lons[j0]))

    corners = [
        (i0, j0, (1.0 - wy) * (1.0 - wx)),
        (i0, j1, (1.0 - wy) * wx),
        (i1, j0, wy * (1.0 - wx)),
        (i1, j1, wy * wx),
    ]
    return corners


def predict_daily_risk_probabilities(
    lat: float,
    lon: float,
    target_date_str: str,
) -> dict:
    """
    Recibe coordenadas exactas (lat, lon) y una fecha futura ('YYYY-MM-DD') a semanas
    o meses desde hoy. Retorna únicamente los porcentajes de probabilidad diaria (0-100%).
    """
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    today = date.today()
    days_ahead = (target_date - today).days

    if days_ahead <= 0:
        raise ValueError(
            f"La fecha consultada ({target_date_str}) debe ser futura (posterior a hoy: {today.isoformat()})."
        )

    if not BEST_MODEL_PATH.exists() or not CONTEXT_PATH.exists():
        raise FileNotFoundError(
            "No se encontraron los artefactos de inferencia locales en las rutas configuradas."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = np.load(CONTEXT_PATH)
    clim_table_norm = ctx["clim_table_norm"]
    lats = ctx["lats"]
    lons = ctx["lons"]
    hot_threshold = ctx["hot_threshold"]
    cold_threshold = ctx["cold_threshold"]

    model = WeatherRiskProbNet(in_channels=7, meta_dim=10, out_probs=3).to(device)
    ckpt = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    doy_idx = min(target_date.timetuple().tm_yday - 1, 365)
    corners = _bilinear_grid_corners(lat, lon, lats, lons)

    patches = []
    metas = []
    weights = []
    hot_lim = 0.0
    cold_lim = 0.0

    for i_lat, j_lon, w in corners:
        patches.append(extract_clim_patch(clim_table_norm, doy_idx, i_lat, j_lon, PATCH_SIZE))
        metas.append(build_coord_time_vector(float(lats[i_lat]), float(lons[j_lon]), doy_idx, target_date.year))
        weights.append(w)
        hot_lim += w * float(hot_threshold[i_lat, j_lon])
        cold_lim += w * float(cold_threshold[i_lat, j_lon])

    x_patch_t = torch.from_numpy(np.stack(patches, axis=0)).to(device)
    x_meta_t = torch.from_numpy(np.stack(metas, axis=0)).to(device)
    w_np = np.array(weights, dtype=np.float32)[:, None]

    probs_4 = model.predict_percentages(x_patch_t, x_meta_t).cpu().numpy()  # (4, 3)
    blended = (probs_4 * w_np).sum(axis=0)

    return {
        "latitud": lat,
        "longitud": lon,
        "fecha_objetivo": target_date.isoformat(),
        "dias_hacia_el_futuro": days_ahead,
        "probabilidad_lluvia_pct": round(float(blended[0]), 2),
        "probabilidad_viento_fuerte_pct": round(float(blended[1]), 2),
        "probabilidad_temperaturas_adversas_pct": round(float(blended[2]), 2),
        "referencia_umbral_adversa_local_c": {
            "calor_extremo_desde": round(hot_lim, 1),
            "frio_extremo_hasta": round(cold_lim, 1),
        },
    }


if __name__ == "__main__":
    # Ejemplo de consulta a varios meses hacia el futuro
    resultado = predict_daily_risk_probabilities(
        lat=28.1901,
        lon=-105.4701,
        target_date_str="2027-07-15",
    )
    print("Pronóstico probabilístico diario:")
    for clave, valor in resultado.items():
        print(f"  {clave}: {valor}")
