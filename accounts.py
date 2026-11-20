import hashlib
import logging
import re
import secrets
import time
from typing import Annotated

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import APIRouter, BackgroundTasks, Cookie, Depends, HTTPException, Response
from pydantic import BaseModel, ConfigDict, EmailStr, Field, StringConstraints

from config import public_origin
from mailer import check_mail_config, send_mail
from storage import database

router = APIRouter(prefix="/auth", tags=["accounts"])
logger = logging.getLogger(__name__)
hasher = PasswordHasher()
dummy_hash = hasher.hash(secrets.token_urlsafe(32))
COOKIE = "nubira_session"
SESSION_AGE = 7 * 86400
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{43}\Z")
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
Password = Annotated[str, Field(min_length=12, max_length=128)]
AccountToken = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{43}$")]
MAIL_ACCEPTED = {"message": "Si corresponde, recibirás un correo con el siguiente paso."}


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Address(Payload):
    email: EmailStr


class Registration(Address):
    name: Name


class Credentials(Address):
    password: str = Field(min_length=1, max_length=128)


class TokenPayload(Payload):
    token: AccountToken


class ResetPayload(TokenPayload):
    password: Password


class ProfilePayload(Payload):
    name: Name


class ChangePasswordPayload(Payload):
    current_password: str = Field(min_length=1, max_length=128)
    password: Password


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def rate_limit(action: str, identity: str, limit: int, interval: int):
    now = int(time.time())
    with database() as conn:
        row = conn.execute("""
            INSERT INTO rate_limits VALUES (?, 1, ?)
            ON CONFLICT(key) DO UPDATE SET
                count = CASE WHEN rate_limits.expires <= ? THEN 1
                    WHEN rate_limits.count < ? THEN rate_limits.count + 1 ELSE ? END,
                expires = CASE WHEN rate_limits.expires <= ? THEN excluded.expires ELSE rate_limits.expires END
            RETURNING count, expires
        """, (digest(f"{action}:{identity}"), now + interval, now, limit + 1, limit + 1, now)).fetchone()
    if row["count"] > limit:
        raise HTTPException(429, "Demasiados intentos. Intenta más tarde.", headers={"Retry-After": str(row["expires"] - now)})


def require_email_service():
    try:
        check_mail_config()
    except RuntimeError as exc:
        raise HTTPException(503, "El servicio de correo no está disponible.") from exc


def user_view(row) -> dict:
    return {"name": row["name"], "email": row["email"], "verified": bool(row["verified_at"])}


def current_user(nubira_session: Annotated[str | None, Cookie()] = None):
    if nubira_session and TOKEN_PATTERN.fullmatch(nubira_session):
        with database() as conn:
            row = conn.execute("""
                SELECT u.* FROM users u JOIN sessions s ON s.user_id = u.id
                WHERE s.digest = ? AND s.expires > ? AND u.verified_at IS NOT NULL
            """, (digest(nubira_session), int(time.time()))).fetchone()
        if row:
            return row
    raise HTTPException(401, "Inicia sesión para continuar.")


User = Annotated[object, Depends(current_user)]


def _send_account_link(user_id: int, email: str, purpose: str):
    token = secrets.token_urlsafe(32)
    lifetime = 86400 if purpose == "verify" else 1800
    with database() as conn:
        conn.execute("INSERT INTO account_tokens VALUES (?, ?, ?, ?)",
                     (digest(token), user_id, purpose, int(time.time()) + lifetime))
    if purpose == "verify":
        action, subject = "verificar", "Verifica tu correo en Nubira"
    else:
        action, subject = "restablecer", "Restablece tu contraseña de Nubira"
    link = f"{public_origin()}/cuenta/{action}#token={token}"
    try:
        send_mail(email, subject, f"{subject}:\n\n{link}\n\nSi no solicitaste este correo, puedes ignorarlo.")
    except Exception:
        logger.error("No se pudo enviar el correo de cuenta (%s).", purpose)


def _consume_token(conn, token: str, purpose: str) -> int:
    token_digest = digest(token)
    # Serialize credential changes by user before deleting their tokens or sessions.
    owner = conn.lock("""
        SELECT id FROM users WHERE id = (
            SELECT user_id FROM account_tokens WHERE digest = ? AND purpose = ? AND expires > ?
        )
    """, (token_digest, purpose, int(time.time())), clause="FOR NO KEY UPDATE").fetchone()
    if not owner:
        raise HTTPException(400, "El enlace ya no es válido. Solicita uno nuevo.")
    row = conn.execute("""
        DELETE FROM account_tokens WHERE digest = ? AND purpose = ? AND expires > ? RETURNING user_id
    """, (token_digest, purpose, int(time.time()))).fetchone()
    if not row:
        raise HTTPException(400, "El enlace ya no es válido. Solicita uno nuevo.")
    return row["user_id"]


def _check_token(token: str, purpose: str):
    with database() as conn:
        pending = conn.execute("""
            SELECT 1 FROM account_tokens WHERE digest = ? AND purpose = ? AND expires > ?
        """, (digest(token), purpose, int(time.time()))).fetchone()
    if not pending:
        raise HTTPException(400, "El enlace ya no es válido. Solicita uno nuevo.")


def _clear_session(response: Response):
    response.delete_cookie(COOKIE, httponly=True, secure=public_origin().startswith("https://"), samesite="lax")


@router.post("/register", status_code=202)
def register(body: Registration, tasks: BackgroundTasks):
    require_email_service()
    email = str(body.email).casefold()
    rate_limit("register", email, 3, 3600)
    with database() as conn:
        row = conn.execute("""
            INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)
            ON CONFLICT(email) DO NOTHING RETURNING id
        """, (body.name, email, dummy_hash)).fetchone()
    if row:
        tasks.add_task(_send_account_link, row["id"], email, "verify")
    return MAIL_ACCEPTED


@router.post("/resend-verification", status_code=202)
def resend_verification(body: Address, tasks: BackgroundTasks):
    require_email_service()
    email = str(body.email).casefold()
    rate_limit("resend", email, 3, 3600)
    with database() as conn:
        row = conn.execute("SELECT id FROM users WHERE email = ? AND verified_at IS NULL", (email,)).fetchone()
    if row:
        tasks.add_task(_send_account_link, row["id"], email, "verify")
    return MAIL_ACCEPTED


@router.post("/verify")
def verify(body: ResetPayload):
    _check_token(body.token, "verify")
    new_hash = hasher.hash(body.password)
    with database() as conn:
        user_id = _consume_token(conn, body.token, "verify")
        _replace_password(conn, user_id, new_hash)
        conn.execute("UPDATE users SET verified_at = ? WHERE id = ?", (int(time.time()), user_id))
        conn.execute("DELETE FROM account_tokens WHERE user_id = ? AND purpose = 'verify'", (user_id,))
    return {"message": "Correo verificado. Ya puedes iniciar sesión."}


def _valid_password(password_hash: str, password: str) -> bool:
    try:
        return hasher.verify(password_hash, password)
    except VerifyMismatchError:
        return False


@router.post("/login")
def login(body: Credentials, response: Response):
    email = str(body.email).casefold()
    rate_limit("login", email, 12, 900)
    with database() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if not _valid_password(row["password_hash"] if row else dummy_hash, body.password) or not row:
        raise HTTPException(401, "El correo o la contraseña no son correctos.")
    if not row["verified_at"]:
        raise HTTPException(403, "Verifica tu correo antes de iniciar sesión.")

    token = secrets.token_urlsafe(32)
    now = int(time.time())
    with database() as conn:
        original = conn.lock("SELECT password_hash FROM users WHERE id = ?", (row["id"],),
                             clause="FOR NO KEY UPDATE").fetchone()
        if not original or original["password_hash"] != row["password_hash"]:
            raise HTTPException(401, "La contraseña cambió. Intenta iniciar sesión otra vez.")
        conn.execute("INSERT INTO sessions (digest, user_id, expires) VALUES (?, ?, ?)",
                     (digest(token), row["id"], now + SESSION_AGE))
        conn.execute("DELETE FROM rate_limits WHERE key = ?", (digest(f"login:{email}"),))
    response.set_cookie(COOKIE, token, max_age=SESSION_AGE, httponly=True,
                        secure=public_origin().startswith("https://"), samesite="lax")
    return user_view(row)


@router.post("/logout", status_code=204)
def logout(response: Response, nubira_session: Annotated[str | None, Cookie()] = None):
    if nubira_session and TOKEN_PATTERN.fullmatch(nubira_session):
        with database() as conn:
            conn.execute("DELETE FROM sessions WHERE digest = ?", (digest(nubira_session),))
    _clear_session(response)


@router.get("/me")
def me(user: User):
    return user_view(user)


@router.patch("/me")
def update_profile(body: ProfilePayload, user: User):
    with database() as conn:
        row = conn.execute("UPDATE users SET name = ? WHERE id = ? RETURNING *",
                           (body.name, user["id"])).fetchone()
    return user_view(row)


@router.post("/forgot-password", status_code=202)
def forgot_password(body: Address, tasks: BackgroundTasks):
    require_email_service()
    email = str(body.email).casefold()
    rate_limit("forgot", email, 3, 3600)
    with database() as conn:
        row = conn.execute("SELECT id FROM users WHERE email = ? AND verified_at IS NOT NULL", (email,)).fetchone()
    if row:
        tasks.add_task(_send_account_link, row["id"], email, "reset")
    return MAIL_ACCEPTED


@router.post("/reset-password")
def reset_password(body: ResetPayload, response: Response):
    _check_token(body.token, "reset")
    new_hash = hasher.hash(body.password)
    with database() as conn:
        user_id = _consume_token(conn, body.token, "reset")
        _replace_password(conn, user_id, new_hash)
    _clear_session(response)
    return {"message": "Contraseña actualizada. Ya puedes iniciar sesión."}


def _replace_password(conn, user_id: int, new_hash: str):
    conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user_id))
    conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM account_tokens WHERE user_id = ? AND purpose = 'reset'", (user_id,))


@router.post("/change-password", status_code=204)
def change_password(body: ChangePasswordPayload, user: User, response: Response):
    rate_limit("change", str(user["id"]), 8, 900)
    if not _valid_password(user["password_hash"], body.current_password):
        raise HTTPException(400, "La contraseña actual no es correcta.")
    new_hash = hasher.hash(body.password)
    with database() as conn:
        original = conn.lock("SELECT password_hash FROM users WHERE id = ?", (user["id"],),
                             clause="FOR NO KEY UPDATE").fetchone()
        if not original or original["password_hash"] != user["password_hash"]:
            raise HTTPException(401, "La contraseña cambió. Vuelve a iniciar sesión.")
        _replace_password(conn, user["id"], new_hash)
    _clear_session(response)
