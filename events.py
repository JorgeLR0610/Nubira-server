import json
from datetime import date, datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException
from pydantic import Field

from accounts import Name, Payload, User, rate_limit, require_email_service
from storage import database
from weather_service import ClimaRequest, evaluate

router = APIRouter(prefix="/events", tags=["events"])


class EventCreate(ClimaRequest):
    name: Name
    alerts_enabled: bool = Field(default=False, strict=True)


class EventUpdate(Payload):
    name: Name | None = None
    alerts_enabled: bool | None = Field(default=None, strict=True)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def event_view(row):
    return {
        "id": row["id"], "name": row["name"],
        "latitude": row["latitude"], "longitude": row["longitude"],
        "target_date": row["target_date"], "prediction": json.loads(row["prediction"]),
        "evaluated_at": row["evaluated_at"], "created_at": row["created_at"],
        "alerts_enabled": bool(row["alerts"]),
    }


def find_event(conn, event_id, owner, *, lock=False):
    select = conn.lock if lock else conn.execute
    row = select("SELECT * FROM events WHERE id = ? AND user_id = ?", (str(event_id), owner)).fetchone()
    if not row:
        raise HTTPException(404, "No encontramos ese evento.")
    return row


@router.get("")
def list_events(user: User):
    with database() as conn:
        return [event_view(row) for row in conn.execute("""
            SELECT * FROM events WHERE user_id = ? ORDER BY target_date, created_at DESC
        """, (user["id"],))]


@router.post("", status_code=201)
def create_event(body: EventCreate, user: User):
    target = body.future_date()
    if body.alerts_enabled:
        require_email_service()
    rate_limit("save-event", str(user["id"]), 25, 3600)
    prediction = evaluate(body.latitude, body.longitude, target).model_dump_json()
    now = timestamp()
    with database() as conn:
        owner = conn.lock("SELECT id FROM users WHERE id = ?", (user["id"],), clause="FOR NO KEY UPDATE").fetchone()
        if not owner:
            raise HTTPException(401, "Inicia sesión para continuar.")
        count = conn.execute("SELECT count(*) AS total FROM events WHERE user_id = ?", (user["id"],)).fetchone()["total"]
        if count >= 100:
            raise HTTPException(409, "Llegaste al límite de 100 eventos. Elimina uno antes de guardar otro.")
        row = conn.execute("""
            INSERT INTO events (id, user_id, name, latitude, longitude, target_date, prediction, evaluated_at,
                                created_at, alerts, alert_baseline)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING *
        """, (str(uuid4()), user["id"], body.name, body.latitude, body.longitude, target.isoformat(),
              prediction, now, now, int(body.alerts_enabled), prediction if body.alerts_enabled else None)).fetchone()
    return event_view(row)


@router.get("/{event_id}")
def get_event(event_id: UUID, user: User):
    with database() as conn:
        return event_view(find_event(conn, event_id, user["id"]))


@router.patch("/{event_id}")
def update_event(event_id: UUID, body: EventUpdate, user: User):
    with database() as conn:
        row = find_event(conn, event_id, user["id"], lock=True)
        alerts = bool(row["alerts"])
        baseline = row["alert_baseline"]
        if body.alerts_enabled is not None and body.alerts_enabled != alerts:
            if body.alerts_enabled:
                require_email_service()
                if row["target_date"] <= date.today().isoformat():
                    raise HTTPException(422, "No puedes activar alertas para un evento que ya pasó.")
            alerts = body.alerts_enabled
            baseline = row["prediction"] if alerts else None
            if not alerts:
                conn.execute("DELETE FROM alert_outbox WHERE event_id = ? AND sent_at IS NULL", (row["id"],))
        row = conn.execute("""
            UPDATE events SET name = ?, alerts = ?, alert_baseline = ? WHERE id = ? AND user_id = ? RETURNING *
        """, (body.name if body.name is not None else row["name"], int(alerts), baseline, str(event_id),
              user["id"])).fetchone()
    return event_view(row)


@router.delete("/{event_id}", status_code=204)
def delete_event(event_id: UUID, user: User):
    with database() as conn:
        deleted = conn.execute("DELETE FROM events WHERE id = ? AND user_id = ?", (str(event_id), user["id"]))
        if deleted.rowcount != 1:
            raise HTTPException(404, "No encontramos ese evento.")
