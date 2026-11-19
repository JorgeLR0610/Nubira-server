import os
import sqlite3
import time
from contextlib import closing, contextmanager
from pathlib import Path


def database_path() -> Path:
    return Path(os.environ.get("NUBIRA_DB_PATH", Path(__file__).parent / "workspace" / "nubira.sqlite3"))


@contextmanager
def database():
    with closing(sqlite3.connect(database_path(), timeout=10)) as conn, conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn


def init_database():
    database_path().parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    database_path().touch(mode=0o600, exist_ok=True)
    with database() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                verified_at INTEGER
            );
            CREATE TABLE IF NOT EXISTS sessions (
                digest TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS account_tokens (
                digest TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                purpose TEXT NOT NULL,
                expires INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rate_limits (
                key TEXT PRIMARY KEY,
                count INTEGER NOT NULL,
                expires INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS place_cache (
                key TEXT PRIMARY KEY,
                response TEXT NOT NULL,
                expires INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                target_date TEXT NOT NULL,
                prediction TEXT NOT NULL,
                evaluated_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                alerts INTEGER NOT NULL DEFAULT 0,
                alert_baseline TEXT
            );
            CREATE INDEX IF NOT EXISTS events_owner ON events(user_id, target_date);
            CREATE TABLE IF NOT EXISTS alert_outbox (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                due INTEGER NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                lease_until INTEGER NOT NULL DEFAULT 0,
                sent_at INTEGER
            );
        """)


def prune_database():
    now = int(time.time())
    with database() as conn:
        conn.execute("DELETE FROM sessions WHERE expires <= ?", (now,))
        conn.execute("DELETE FROM account_tokens WHERE expires <= ?", (now,))
        conn.execute("DELETE FROM rate_limits WHERE expires <= ?", (now,))
        conn.execute("DELETE FROM place_cache WHERE expires <= ?", (now,))
        conn.execute("DELETE FROM alert_outbox WHERE sent_at <= ?", (now - 30 * 86400,))
        conn.execute("""
            DELETE FROM alert_outbox WHERE sent_at IS NULL AND event_id IN
            (SELECT id FROM events WHERE target_date <= date('now'))
        """)
