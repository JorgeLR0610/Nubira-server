"""Evaluate and send alerts from the API scheduler or `python -m monitor`."""

import argparse
import json
import logging
import math
import os
import signal
import time
from collections.abc import Callable
from datetime import date
from uuid import uuid4

from fastapi import HTTPException

from config import public_origin
from events import timestamp
from mailer import check_mail_config, send_mail
from storage import DatabaseUnavailable, database, init_database, prune_database
from weather_service import PROBABILITIES, evaluate

logger = logging.getLogger(__name__)


def _keep_running() -> bool:
    return True


def reevaluate(threshold: float, running: Callable[[], bool] = _keep_running) -> int:
    today = date.today().isoformat()
    with database() as conn:
        rows = conn.execute("""
            SELECT id, name, latitude, longitude, target_date, prediction, evaluated_at, alert_baseline
            FROM events WHERE alerts = 1 AND target_date > ?
        """, (today,)).fetchall()
    predictions = {}
    notified = 0
    for row in rows:
        if not running():
            break
        key = (row["latitude"], row["longitude"], date.fromisoformat(row["target_date"]))
        try:
            if key not in predictions:
                predictions[key] = evaluate(*key)
        except HTTPException:
            logger.warning("Reevaluación interrumpida: el servicio de inferencia no está disponible.")
            break

        prediction = predictions[key]
        updated = prediction.model_dump_json()
        previous = json.loads(row["alert_baseline"] or row["prediction"])
        current = prediction.model_dump()
        changes = [
            f"{label}: {previous[field]:g}% → {current[field]:g}%"
            for field, label in PROBABILITIES.items()
            if abs(round(current[field] - previous[field], 2)) >= threshold
        ]
        with database() as conn:
            null_equals = "IS NOT DISTINCT FROM" if conn.postgres else "IS"
            changed = conn.execute(f"""
                UPDATE events SET prediction = ?, evaluated_at = ?, alert_baseline = ?
                WHERE id = ? AND alerts = 1 AND evaluated_at = ?
                    AND alert_baseline {null_equals} ?
            """, (updated, timestamp(), updated if changes else row["alert_baseline"],
                  row["id"], row["evaluated_at"], row["alert_baseline"]))
            if changed.rowcount == 1 and changes:
                link = f"{public_origin()}/eventos/{row['id']}"
                body = (f"Cambió la evaluación climatológica de «{row['name']}» ({row['target_date']}):\n\n"
                        + "\n".join(changes)
                        + f"\n\n{link}\n\nEsta evaluación no es un pronóstico meteorológico puntual.")
                conn.execute("""
                    INSERT INTO alert_outbox (id, event_id, subject, body, due) VALUES (?, ?, ?, ?, ?)
                """, (uuid4().hex, row["id"], "Cambio en el riesgo climático de tu evento", body, int(time.time())))
                notified += 1
    return notified


def deliver_pending(running: Callable[[], bool] = _keep_running) -> int:
    delivered = 0
    # Bound each run; postpone failures with backoff instead of retrying in a tight loop.
    for _ in range(100):
        if not running():
            break
        now = int(time.time())
        with database() as conn:
            job = conn.lock("""
                SELECT a.*, u.email FROM alert_outbox a JOIN events e ON e.id = a.event_id
                JOIN users u ON u.id = e.user_id
                WHERE a.sent_at IS NULL AND a.due <= ? AND a.lease_until <= ?
                    AND e.alerts = 1 AND e.target_date > ?
                ORDER BY a.due LIMIT 1
            """, (now, now, date.today().isoformat()), clause="FOR UPDATE OF a SKIP LOCKED").fetchone()
            if not job:
                break
            lease = now + 600
            conn.execute("UPDATE alert_outbox SET lease_until = ? WHERE id = ?", (lease, job["id"]))
        try:
            send_mail(job["email"], job["subject"], job["body"], message_id=job["id"])
        except Exception:
            retry = now + min(86400, 60 * 2 ** min(job["attempts"], 11))
            with database() as conn:
                conn.execute("""
                    UPDATE alert_outbox SET attempts = attempts + 1, lease_until = 0, due = ?
                    WHERE id = ? AND lease_until = ?
                """, (retry, job["id"], lease))
            logger.warning("No se pudo enviar una alerta; queda pendiente para reintento.")
        else:
            with database() as conn:
                conn.execute("UPDATE alert_outbox SET sent_at = ? WHERE id = ? AND lease_until = ?",
                             (int(time.time()), job["id"], lease))
            delivered += 1
    return delivered


def alert_threshold() -> float:
    try:
        threshold = float(os.environ.get("NUBIRA_ALERT_THRESHOLD_PP", "10"))
        if math.isfinite(threshold) and 0.01 <= threshold <= 100:
            return threshold
    except ValueError:
        pass
    raise ValueError("NUBIRA_ALERT_THRESHOLD_PP debe estar entre 0.01 y 100.")


def run_cycle(threshold: float, running: Callable[[], bool] = _keep_running, *, retry=False):
    try:
        prune_database()
        queued, sent = reevaluate(threshold, running), deliver_pending(running)
        logger.info("Alertas en cola: %d; enviadas: %d.", queued, sent)
    except DatabaseUnavailable:
        if not retry:
            raise
        logger.warning("No se pudo conectar con la base de datos; se reintentará en el siguiente ciclo.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, help="Ejecutar continuamente, a este intervalo en segundos.")
    args = parser.parse_args()
    if args.interval is not None and args.interval < 60:
        parser.error("El intervalo debe ser de al menos 60 segundos.")
    try:
        threshold = alert_threshold()
    except ValueError as exc:
        parser.error(str(exc))
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    def running():
        return not stopping

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    logging.basicConfig(level=logging.INFO)
    init_database()
    check_mail_config()
    while running():
        run_cycle(threshold, running, retry=args.interval is not None)
        if args.interval is None:
            return
        deadline = time.monotonic() + args.interval
        while running() and (remaining := deadline - time.monotonic()) > 0:
            time.sleep(min(1, remaining))
    logger.info("Monitor detenido.")


if __name__ == "__main__":
    main()
