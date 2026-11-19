import json
import os
import time
from typing import Annotated
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException
from pydantic import StringConstraints

from accounts import Payload, digest, rate_limit
from storage import database

router = APIRouter(prefix="/places", tags=["places"])
GEOAPIFY_AUTOCOMPLETE = "https://api.geoapify.com/v1/geocode/autocomplete"
PLACE_ZOOM = {"country": 5, "state": 7, "county": 9, "city": 11, "district": 12, "suburb": 13, "postcode": 13}


class Search(Payload):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=140)]


class AutocompleteSearch(Payload):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=4, max_length=140)]


def config():
    contact = os.environ.get("NUBIRA_GEOCODER_CONTACT")
    url = os.environ.get("NUBIRA_GEOCODER_URL", "https://nominatim.openstreetmap.org/search")
    parts = urlsplit(url)
    return (url, contact) if contact and parts.scheme == "https" and parts.hostname and not (
        parts.query or parts.fragment or parts.username or parts.password
    ) else None


def autocomplete_key():
    return os.environ.get("NUBIRA_GEOAPIFY_API_KEY", "").strip()


@router.get("")
def feature():
    return {"enabled": config() is not None, "autocomplete": bool(autocomplete_key())}


@router.post("/autocomplete")
def autocomplete(body: AutocompleteSearch):
    key = autocomplete_key()
    if not key:
        raise HTTPException(503, "El autocompletado no está disponible. Puedes usar el mapa o las coordenadas.")
    query = " ".join(body.query.split())
    if len(query) < 4:
        raise HTTPException(422, "Escribe al menos cuatro caracteres.")
    # Keep autocomplete off the public Nominatim service and limit quota consumption across API workers.
    rate_limit("autocomplete-burst", "geoapify", 2, 1)
    rate_limit("autocomplete-budget", "geoapify", 1000, 86400)
    params = urlencode({"text": query, "format": "geojson", "lang": "es", "limit": 5,
                        "bias": "countrycode:mx", "apiKey": key})
    request = Request(f"{GEOAPIFY_AUTOCOMPLETE}?{params}", headers={"Accept": "application/json"})
    unavailable = "No pudimos sugerir direcciones. Puedes usar el mapa o ingresar las coordenadas."
    try:
        with urlopen(request, timeout=4) as response:
            upstream = json.loads(response.read(65536))
        features = upstream.get("features") if isinstance(upstream, dict) else None
        if not isinstance(features, list):
            raise ValueError("Respuesta de autocompletado inválida.")
    except HTTPError as exc:
        if exc.code == 429:
            raise HTTPException(429, "La búsqueda está ocupada. Intenta de nuevo más tarde.") from exc
        raise HTTPException(502, unavailable) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(502, unavailable) from exc

    results, seen = [], set()
    for feature in features[:5]:
        place = feature.get("properties") if isinstance(feature, dict) else None
        if not isinstance(place, dict):
            continue
        lat, lon, name = place.get("lat"), place.get("lon"), place.get("formatted")
        if (type(lat) not in (float, int) or type(lon) not in (float, int) or
                not (-90 <= lat <= 90 and -180 <= lon <= 180) or not isinstance(name, str) or not name.strip()):
            continue
        label = name.strip()[:300]
        identity = (lat, lon, label)
        if identity in seen:
            continue
        seen.add(identity)
        kind = place.get("result_type")
        zoom = PLACE_ZOOM.get(kind, 15) if isinstance(kind, str) else 15
        results.append({"lat": lat, "lon": lon, "label": label, "zoom": zoom})
    return results


@router.post("")
def search(body: Search):
    provider = config()
    if not provider:
        raise HTTPException(503, "La búsqueda de lugares no está disponible. Puedes usar las coordenadas.")
    url, contact = provider
    query = " ".join(body.query.split())
    if len(query) < 3:
        raise HTTPException(422, "Escribe al menos tres caracteres.")
    now = int(time.time())
    key = digest(f"{url}:{query.casefold()}")
    with database() as conn:
        hit = conn.execute("SELECT response FROM place_cache WHERE key = ? AND expires > ?", (key, now)).fetchone()
    if hit:
        return json.loads(hit["response"])
    # Limit across all web processes sharing this database; cached queries are exempt.
    rate_limit("geocoder", "shared", 1, 2)
    params = urlencode({"q": query, "format": "jsonv2", "limit": 5, "accept-language": "es"})
    request = Request(f"{url}?{params}", headers={"User-Agent": f"NubiraWeather/3.0 ({contact})",
                                                "Accept": "application/json"})
    try:
        with urlopen(request, timeout=5) as response:
            upstream = json.loads(response.read(65536))
        results = []
        for item in upstream[:5]:
            lat, lon = float(item["lat"]), float(item["lon"])
            if -90 <= lat <= 90 and -180 <= lon <= 180 and isinstance(item["display_name"], str):
                results.append({"lat": lat, "lon": lon, "label": item["display_name"][:300]})
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(502, "No pudimos buscar ese lugar. Puedes usar el mapa o ingresar las coordenadas.") from exc
    with database() as conn:
        conn.execute("DELETE FROM place_cache WHERE expires <= ?", (now,))
        conn.execute("INSERT OR REPLACE INTO place_cache VALUES (?, ?, ?)", (key, json.dumps(results), now + 86400))
    return results
