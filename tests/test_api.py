import json
import os
import re
import select
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from email import policy
from email.parser import BytesParser
from pathlib import Path
from threading import Barrier, Event
from urllib.error import HTTPError
from urllib.parse import parse_qs, parse_qsl, quote, urlencode, urlsplit
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import appPowerFastAPI
import events
import monitor
import places
import weather_service
from storage import database
from weather_service import Prediction

ROOT = Path(__file__).resolve().parents[2]
PASSWORD = "Un evento para recordar 2027"
NEW_PASSWORD = "Otro evento para recordar 2027"


def sample_prediction(lat, lon, target, *, rain=69.13, days=42):
    return Prediction.model_validate({
        "latitud": lat, "longitud": lon, "fecha_objetivo": target,
        "dias_hacia_el_futuro": days, "probabilidad_lluvia_pct": rain,
        "probabilidad_viento_fuerte_pct": 0.19, "probabilidad_temperaturas_adversas_pct": 20.4,
        "referencia_umbral_adversa_local_c": {"calor_extremo_desde": 31.3, "frio_extremo_hasta": 3.6},
    })


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("NUBIRA_REQUIRE_POSTGRES", raising=False)
    monkeypatch.delenv("NUBIRA_MONITOR_INTERVAL", raising=False)
    monkeypatch.delenv("FRONTEND_URL", raising=False)
    monkeypatch.delenv("MAIL_PROVIDER", raising=False)
    monkeypatch.setenv("NUBIRA_DB_PATH", str(tmp_path / "test.sqlite3"))
    monkeypatch.setenv("NUBIRA_MAIL_MODE", "file")
    monkeypatch.setenv("NUBIRA_MAIL_DIR", str(tmp_path / "mail"))
    monkeypatch.setenv("NUBIRA_PUBLIC_URL", "http://localhost:3000")
    monkeypatch.delenv("NUBIRA_GEOCODER_CONTACT", raising=False)
    monkeypatch.delenv("NUBIRA_GEOAPIFY_API_KEY", raising=False)
    monkeypatch.setattr(events, "evaluate", sample_prediction)
    if not (test_url := os.environ.get("NUBIRA_TEST_POSTGRES_URL")):
        yield tmp_path
        return

    # Explicit opt-in only, using a disposable schema per test. The normal run never connects.
    import psycopg
    from psycopg import sql

    schema = f"test_nubira_{uuid4().hex}"
    parts = urlsplit(test_url)
    query = dict(parse_qsl(parts.query), options=f"-c search_path={schema}")
    with psycopg.connect(test_url, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        monkeypatch.setenv("DATABASE_URL", parts._replace(query=urlencode(query, quote_via=quote)).geturl())
        try:
            yield tmp_path
        finally:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def messages(folder, subject, recipient):
    return [
        message
        for file in folder.glob("*.eml")
        if (message := BytesParser(policy=policy.default).parsebytes(file.read_bytes()))
        and message["Subject"] == subject and str(message["To"]) == recipient
    ]


def mail_token(folder, subject, recipient):
    message, = messages(folder, subject, recipient)
    return re.search(r"#token=([A-Za-z0-9_-]{43})", message.get_content())[1]


def create_account(client, mail, email):
    assert client.post("/auth/register", json={"email": email, "name": "Organizadora"}).status_code == 202
    token = mail_token(mail, "Verifica tu correo en Nubira", email)
    assert client.post("/auth/verify", json={"token": token, "password": PASSWORD}).status_code == 200
    return client.post("/auth/login", json={"email": email, "password": PASSWORD})


def event_request(alerts_enabled=False):
    target = date.today() + timedelta(days=42)
    return {"latitude": 28.1901, "longitude": -105.4701, "day": target.day, "month": target.month,
            "year": target.year, "name": "Festival", "alerts_enabled": alerts_enabled}


def test_inference_matches_archived_example(setup, monkeypatch):
    if not weather_service.artifacts_ready():
        pytest.fail("Faltan los artefactos del modelo. Ejecuta `python fetch_artifacts.py`.")
    import predict

    sample_in, sample_out = ROOT / "input.json", ROOT / "output.json"
    if sample_in.is_file() and sample_out.is_file():
        payload, expected = json.loads(sample_in.read_text()), json.loads(sample_out.read_text())
    else:
        # Also run on CI for this standalone repository.
        payload = {"latitude": 28.1901, "longitude": -105.4701, "day": 15, "month": 7, "year": 2027}
        expected = sample_prediction(28.1901, -105.4701, "2027-07-15", days=239).model_dump(mode="json")

    class ExampleDate(date):
        @classmethod
        def today(cls):
            return date.fromisoformat(expected["fecha_objetivo"]) - timedelta(days=expected["dias_hacia_el_futuro"])

    monkeypatch.setattr(predict, "date", ExampleDate)
    monkeypatch.setattr(weather_service, "date", ExampleDate)
    with TestClient(appPowerFastAPI.app) as client:
        assert client.get("/health").status_code == 200
        first = client.post("/clima", json=payload)
        second = client.post("/clima", json=payload)
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json() == expected


def test_weather_rejects_invalid_or_missing_data_without_fake_results(setup, monkeypatch):
    monkeypatch.setattr(weather_service, "artifacts_ready", lambda: False)
    with TestClient(appPowerFastAPI.app) as client:
        def predict_request(changes):
            body = event_request()
            del body["name"], body["alerts_enabled"]
            return client.post("/clima", json={**body, **changes})

        assert predict_request({"latitude": 100}).status_code == 422
        assert predict_request({"longitude": "-105"}).status_code == 422
        assert predict_request({"day": 31, "month": 2}).status_code == 422
        assert predict_request({"day": 31, "month": 12, "year": 2025}).status_code == 422
        unavailable = predict_request({})
        assert unavailable.status_code == 503
        assert unavailable.json() == {"detail": "El servicio de evaluación no está disponible."}


def test_health_only_checks_local_artifacts(setup, monkeypatch):
    with TestClient(appPowerFastAPI.app) as client:
        monkeypatch.setenv("DATABASE_URL", "invalid://unavailable")
        monkeypatch.setenv("MAIL_PROVIDER", "unavailable")
        assert client.get("/health").status_code == 200
        monkeypatch.setattr(weather_service, "BEST_MODEL_PATH", setup / "missing.pt")
        failed = client.get("/health")
        assert failed.status_code == 503
        assert failed.json()["detail"] == "Artefactos de inferencia no disponibles."


def test_cors_allows_only_configured_frontends(setup, monkeypatch):
    vercel = "https://nubira-weather.vercel.app"
    local = "http://localhost:3000"
    monkeypatch.setenv("CORS_ORIGINS", f" {vercel}, {local}/ ")
    with TestClient(appPowerFastAPI.create_app()) as client:
        for origin in (vercel, local):
            preflight = client.options("/events", headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            })
            assert preflight.status_code == 200
            assert preflight.headers["access-control-allow-origin"] == origin
            assert preflight.headers["access-control-allow-credentials"] == "true"
        rejected = client.options("/clima", headers={
            "Origin": "https://unknown.example.com", "Access-Control-Request-Method": "POST",
        })
        assert rejected.status_code == 400
        assert "access-control-allow-origin" not in rejected.headers
        assert client.get("/health", headers={"Origin": vercel}).headers["access-control-allow-origin"] == vercel
    monkeypatch.setenv("CORS_ORIGINS", "*")
    with pytest.raises(RuntimeError, match="CORS_ORIGINS"):
        appPowerFastAPI.create_app()


def test_accounts_use_verified_links_and_revoke_sessions_on_password_reset(setup):
    email = "planificacion@example.com"
    with TestClient(appPowerFastAPI.app) as client:
        mail = setup / "mail"
        registration = {"email": email, "name": "Equipo Nubira"}
        assert client.post("/auth/register", json=registration).status_code == 202
        assert client.post("/auth/register", json=registration).status_code == 202
        token = mail_token(mail, "Verifica tu correo en Nubira", email)
        assert client.post("/auth/login", json={"email": email, "password": PASSWORD}).status_code == 401
        assert client.post("/auth/verify", json={"token": token, "password": PASSWORD}).status_code == 200
        assert client.post("/auth/verify", json={"token": token, "password": PASSWORD}).status_code == 400

        login = client.post("/auth/login", json={"email": email.upper(), "password": PASSWORD})
        assert login.status_code == 200
        assert "HttpOnly" in login.headers["set-cookie"] and "SameSite=lax" in login.headers["set-cookie"]
        original_token = client.cookies.get("nubira_session")
        assert client.patch("/auth/me", json={"name": "María"}).json() == {"email": email, "name": "María", "verified": True}
        with database() as conn:
            assert original_token != conn.execute("SELECT digest FROM sessions").fetchone()["digest"]
            assert conn.execute("SELECT password_hash FROM users").fetchone()["password_hash"].startswith("$argon2id$")

        recover = client.post("/auth/forgot-password", json={"email": email})
        assert recover.status_code == 202
        assert recover.json() == client.post("/auth/forgot-password", json={"email": "otra@example.com"}).json()
        token = mail_token(mail, "Restablece tu contraseña de Nubira", email)
        assert client.post("/auth/reset-password", json={"token": token, "password": NEW_PASSWORD}).status_code == 200
        assert client.get("/auth/me", headers={"Cookie": f"nubira_session={original_token}"}).status_code == 401
        assert client.post("/auth/reset-password", json={"token": token, "password": NEW_PASSWORD}).status_code == 400
        assert client.post("/auth/login", json={"email": email, "password": PASSWORD}).status_code == 401
        assert client.post("/auth/login", json={"email": email, "password": NEW_PASSWORD}).status_code == 200
        assert client.post("/auth/logout").status_code == 204
        assert client.get("/auth/me").status_code == 401


def test_user_only_sees_their_events_even_after_restart(setup):
    with TestClient(appPowerFastAPI.app) as original:
        assert original.get("/events").status_code == 401
        assert create_account(original, setup / "mail", "owner@example.com").status_code == 200
        assert original.post("/events", json={**event_request(), "prediction": {"fake": True}}).status_code == 422
        response = original.post("/events", json=event_request())
        assert response.status_code == 201
        saved = response.json()
        event_path = f"/events/{saved['id']}"
        cookie = original.cookies.get("nubira_session")
        assert saved["prediction"]["probabilidad_lluvia_pct"] == 69.13

    with TestClient(appPowerFastAPI.app) as visitor:
        assert create_account(visitor, setup / "mail", "visitor@example.com").status_code == 200
        assert visitor.get("/events").json() == []
        assert visitor.get(event_path).status_code == 404
        assert visitor.patch(event_path, json={"name": "Intruso"}).status_code == 404
        assert visitor.delete(event_path).status_code == 404

    with TestClient(appPowerFastAPI.app, cookies={"nubira_session": cookie}) as restarted:
        assert restarted.get("/events").json() == [saved]
        assert restarted.patch(event_path, json={"name": "Expo", "alerts_enabled": True}).status_code == 200
        response = restarted.get(event_path).json()
        assert (response["name"], response["alerts_enabled"]) == ("Expo", True)
        assert restarted.delete(event_path).status_code == 204
        assert restarted.get(event_path).status_code == 404


def test_production_session_requires_https_and_change_password_revokes_it(setup, monkeypatch):
    email = "profile@example.com"
    with TestClient(appPowerFastAPI.app) as client:
        assert create_account(client, setup / "mail", email).status_code == 200
    monkeypatch.setenv("NUBIRA_PUBLIC_URL", "https://nubira.example.com")
    with TestClient(appPowerFastAPI.app, base_url="https://nubira.example.com") as client:
        login = client.post("/auth/login", json={"email": email, "password": PASSWORD})
        assert login.status_code == 200 and "Secure" in login.headers["set-cookie"]
        old = client.cookies.get("nubira_session")
        action = {"current_password": PASSWORD, "password": NEW_PASSWORD}
        assert client.post("/auth/change-password", json={**action, "current_password": "incorrect"}).status_code == 400
        assert client.post("/auth/change-password", json=action).status_code == 204
        assert client.get("/auth/me", headers={"Cookie": f"nubira_session={old}"}).status_code == 401
        assert client.post("/auth/login", json={"email": email, "password": NEW_PASSWORD}).status_code == 200


def test_alerts_require_a_real_change_and_retry_once_available(setup, monkeypatch):
    email = "alerts@example.com"
    with TestClient(appPowerFastAPI.app) as client:
        assert create_account(client, setup / "mail", email).status_code == 200
        saved = client.post("/events", json=event_request(True)).json()
        monkeypatch.setattr(monitor, "evaluate", lambda *a: sample_prediction(*a, days=41))
        assert monitor.reevaluate(10) == 0
        monkeypatch.setattr(monitor, "evaluate", lambda *a: sample_prediction(*a, rain=75, days=41))
        assert monitor.reevaluate(10) == 0
        monkeypatch.setattr(monitor, "evaluate", lambda *a: sample_prediction(*a, rain=84, days=40))
        assert monitor.reevaluate(10) == 1
        assert monitor.reevaluate(10) == 0
        assert client.get(f"/events/{saved['id']}").json()["prediction"]["probabilidad_lluvia_pct"] == 84

        with monkeypatch.context() as patch:
            patch.setattr(monitor, "send_mail", lambda *a, **kw: (_ for _ in ()).throw(OSError("offline")))
            assert monitor.deliver_pending() == 0
        with database() as conn:
            job = conn.execute("SELECT * FROM alert_outbox").fetchone()
            assert job["attempts"] == 1 and job["due"] > time.time() and job["sent_at"] is None
            conn.execute("UPDATE alert_outbox SET due = 0")
        assert monitor.deliver_pending() == 1
        assert monitor.deliver_pending() == 0
        alert, = messages(setup / "mail", "Cambio en el riesgo climático de tu evento", email)
        assert "69.13% → 84%" in alert.get_content()
        assert f"/eventos/{saved['id']}" in alert.get_content()
        assert client.patch(f"/events/{saved['id']}", json={"alerts_enabled": False}).status_code == 200
        monkeypatch.setattr(monitor, "evaluate", lambda *a: sample_prediction(*a, rain=10, days=39))
        assert monitor.reevaluate(10) == 0


def test_concurrent_requests_cannot_exceed_event_limit(setup, monkeypatch):
    both_ready = Barrier(2, timeout=5)

    def evaluate_together(*args):
        both_ready.wait()
        return sample_prediction(*args)

    with TestClient(appPowerFastAPI.app) as client:
        assert create_account(client, setup / "mail", "limit@example.com").status_code == 200
        assert client.post("/events", json=event_request()).status_code == 201
        with database() as conn:
            for _ in range(98):
                conn.execute("""
                    INSERT INTO events (id, user_id, name, latitude, longitude, target_date, prediction,
                                        evaluated_at, created_at)
                    SELECT ?, user_id, name, latitude, longitude, target_date, prediction, evaluated_at, created_at
                    FROM events LIMIT 1
                """, (str(uuid4()),))
        monkeypatch.setattr(events, "evaluate", evaluate_together)
        with ThreadPoolExecutor(max_workers=2) as pool:
            statuses = list(pool.map(lambda _: client.post("/events", json=event_request()).status_code, range(2)))
        assert sorted(statuses) == [201, 409]
        assert len(client.get("/events").json()) == 100


def test_concurrent_monitors_only_claim_an_alert_once(setup, monkeypatch):
    both_ready = Barrier(2, timeout=5)
    sent = []

    def deliver():
        both_ready.wait()
        return monitor.deliver_pending()

    with TestClient(appPowerFastAPI.app) as client:
        assert create_account(client, setup / "mail", "claim@example.com").status_code == 200
        assert client.post("/events", json=event_request(True)).status_code == 201
        monkeypatch.setattr(monitor, "evaluate", lambda *args: sample_prediction(*args, rain=90))
        monkeypatch.setattr(monitor, "send_mail", lambda *args, message_id: sent.append(message_id))
        assert monitor.reevaluate(10) == 1
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = [pool.submit(deliver) for _ in range(2)]
            assert sum(result.result(timeout=10) for result in results) == 1
        assert len(sent) == 1


def test_optional_geocoder_is_cached_and_not_needed_for_consultation(setup, monkeypatch):
    with TestClient(appPowerFastAPI.app) as client:
        assert client.get("/places").json() == {"enabled": False, "autocomplete": False}
        assert client.post("/places", json={"query": "Chihuahua"}).status_code == 503
        assert client.post("/places/autocomplete", json={"query": "Chihuahua"}).status_code == 503
        assert client.post("/places/reverse", json={"lat": 28.63, "lon": -106.06}).status_code == 503
        monkeypatch.setenv("NUBIRA_GEOCODER_CONTACT", "ops@example.com")
        entered = []

        class Upstream:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self, _):
                return b'[{"lat":"28.6","lon":"-106","display_name":"Chihuahua"}]'

        def lookup(request, timeout):
            entered.append(request.full_url)
            return Upstream()

        monkeypatch.setattr(places, "urlopen", lookup)
        expected = [{"lat": 28.6, "lon": -106, "label": "Chihuahua"}]
        assert client.get("/places").json() == {"enabled": True, "autocomplete": False}
        assert client.post("/places", json={"query": "Chihuahua"}).json() == expected
        assert client.post("/places", json={"query": "chihuahua"}).json() == expected
        assert client.post("/places", json={"query": "Otro"}).status_code == 429
        assert len(entered) == 1


def test_address_autocomplete_returns_only_valid_geoapify_suggestions(setup, monkeypatch):
    monkeypatch.setenv("NUBIRA_GEOAPIFY_API_KEY", "test-only-key")
    entered = []
    venue = {"lat": 28.63, "lon": -106.06, "formatted": "Avenida Independencia 100, Chihuahua",
             "result_type": "building"}
    city = {"lat": 28.64, "lon": -106.08, "formatted": "Chihuahua, México", "result_type": "city"}

    class Upstream:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, _):
            return json.dumps({"features": [{"properties": entry} for entry in
                [venue, venue, {**venue, "lat": 101}, city, None]]}).encode()

    def lookup(request, timeout):
        entered.append(urlsplit(request.full_url))
        return Upstream()

    monkeypatch.setattr(places, "urlopen", lookup)
    with TestClient(appPowerFastAPI.app) as client:
        assert client.get("/places").json() == {"enabled": False, "autocomplete": True}
        assert client.post("/places/autocomplete", json={"query": "Av"}).status_code == 422
        response = client.post("/places/autocomplete", json={"query": "Avenida   Independencia 100"})
        (called,) = entered
        assert (called.netloc, called.path) == ("api.geoapify.com", "/v1/geocode/autocomplete")
        params = parse_qs(called.query)
        assert params["text"] == ["Avenida Independencia 100"] and params["apiKey"] == ["test-only-key"]
        assert response.json() == [
            {"lat": 28.63, "lon": -106.06, "label": venue["formatted"], "zoom": 15},
            {"lat": 28.64, "lon": -106.08, "label": city["formatted"], "zoom": 11},
        ]


def test_address_provider_errors_are_safe_to_show(setup, monkeypatch):
    monkeypatch.setenv("NUBIRA_GEOAPIFY_API_KEY", "test-only-key")

    def lookup(request, timeout):
        raise HTTPError(request.full_url, 403, "provider error", {}, None)

    monkeypatch.setattr(places, "urlopen", lookup)
    with TestClient(appPowerFastAPI.app) as client:
        response = client.post("/places/autocomplete", json={"query": "Una dirección de prueba"})
        assert response.status_code == 502
        assert response.json() == {
            "detail": "No pudimos sugerir direcciones. Puedes usar el mapa o ingresar las coordenadas."
        }


def test_reverse_geocoding_returns_only_a_label_and_accepts_no_matches(setup, monkeypatch):
    monkeypatch.setenv("NUBIRA_GEOAPIFY_API_KEY", "test-only-key")
    calls = []
    lookups = iter([
        [{"properties": {"lat": 28.6302, "lon": -106.061, "formatted": " Avenida Independencia, Chihuahua "}}],
        [],
    ])

    class Upstream:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, _):
            return json.dumps({"features": next(lookups)}).encode()

    def lookup(request, timeout):
        calls.append(urlsplit(request.full_url))
        return Upstream()

    monkeypatch.setattr(places, "urlopen", lookup)
    with TestClient(appPowerFastAPI.app) as client:
        for payload in ({"lat": 90.01, "lon": 0}, {"lat": 0, "lon": -181}, {"lat": "0", "lon": 0}):
            assert client.post("/places/reverse", json=payload).status_code == 422
        assert calls == []
        found = client.post("/places/reverse", json={"lat": 28.63, "lon": -106.06})
        assert found.status_code == 200
        assert found.json() == {"label": "Avenida Independencia, Chihuahua"}
        assert (calls[0].netloc, calls[0].path) == ("api.geoapify.com", "/v1/geocode/reverse")
        assert parse_qs(calls[0].query) == {
            "lat": ["28.63"], "lon": ["-106.06"], "lang": ["es"], "limit": ["1"],
            "format": ["geojson"], "apiKey": ["test-only-key"],
        }
        assert client.post("/places/reverse", json={"lat": 0, "lon": 0}).json() == {"label": None}


@pytest.mark.skipif(sys.platform == "win32", reason="Verifica señales POSIX de los contenedores.")
def test_monitor_stops_gracefully_between_cycles(setup):
    with subprocess.Popen([sys.executable, "-m", "monitor", "--interval", "900"],
                          cwd=Path(monitor.__file__).parent, stderr=subprocess.PIPE, text=True) as process:
        try:
            assert select.select([process.stderr], [], [], 10)[0]
            assert "Alertas en cola:" in process.stderr.readline()
            process.terminate()
            assert process.wait(timeout=3) == 0
            assert "Monitor detenido." in process.stderr.read()
        finally:
            if process.poll() is None:
                process.kill()


def test_api_runs_the_monitor_when_enabled(setup, monkeypatch):
    started = Event()

    def cycle(threshold, running, *, retry):
        assert threshold == 10 and running() and retry
        started.set()

    monkeypatch.setenv("NUBIRA_MONITOR_INTERVAL", "60")
    monkeypatch.setenv("NUBIRA_ALERT_THRESHOLD_PP", "10")
    monkeypatch.setattr(monitor, "run_cycle", cycle)
    with TestClient(appPowerFastAPI.app) as client:
        assert started.wait(3)
        assert client.get("/health").status_code == 200
