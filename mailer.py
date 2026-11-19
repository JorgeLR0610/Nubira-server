import os
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formatdate
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4


def public_origin() -> str:
    origin = os.environ.get("NUBIRA_PUBLIC_URL", "http://localhost:3000").rstrip("/")
    parts = urlsplit(origin)
    local = parts.hostname in {"localhost", "127.0.0.1", "::1"}
    if (not parts.hostname or parts.username or parts.password or parts.path or parts.query or parts.fragment
            or (parts.scheme != "https" and not (local and parts.scheme == "http"))):
        raise RuntimeError("NUBIRA_PUBLIC_URL debe ser un origen HTTPS (o HTTP en localhost).")
    return origin


def check_mail_config():
    mode = os.environ.get("NUBIRA_MAIL_MODE", "disabled")
    origin = public_origin()
    local = urlsplit(origin).hostname in {"localhost", "127.0.0.1", "::1"}
    if mode == "file" and local and os.environ.get("NUBIRA_MAIL_DIR"):
        return
    if mode == "smtp" and urlsplit(origin).scheme == "https":
        sender = os.environ.get("NUBIRA_MAIL_FROM", "")
        user, password = os.environ.get("NUBIRA_SMTP_USER"), os.environ.get("NUBIRA_SMTP_PASSWORD")
        port = os.environ.get("NUBIRA_SMTP_PORT")
        if (
            os.environ.get("NUBIRA_SMTP_HOST")
            and "." in sender.rpartition("@")[2]
            and bool(user) == bool(password)
            and (not port or (port.isascii() and port.isdecimal() and 1 <= int(port) <= 65535))
        ):
            return
    raise RuntimeError("El servicio de correo no está configurado.")


def send_mail(to: str, subject: str, body: str, message_id: str | None = None):
    check_mail_config()
    sender = os.environ.get("NUBIRA_MAIL_FROM", "nubira@localhost")
    message = EmailMessage()
    message["From"] = sender
    message["To"] = to
    message["Subject"] = subject
    message["Date"] = formatdate(usegmt=True)
    message_id = message_id or uuid4().hex
    message["Message-ID"] = f"<{message_id}.nubira@{urlsplit(public_origin()).hostname}>"
    message.set_content(body)

    if os.environ.get("NUBIRA_MAIL_MODE") == "file":
        outbox = Path(os.environ["NUBIRA_MAIL_DIR"])
        outbox.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = outbox / f"{message_id}.eml"
        if path.exists():
            return
        temp = outbox / f".{uuid4().hex}.tmp"
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(message.as_bytes())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        return

    host = os.environ["NUBIRA_SMTP_HOST"]
    context = ssl.create_default_context()
    implicit_tls = os.environ.get("NUBIRA_SMTP_IMPLICIT_TLS") == "1"
    port = int(os.environ.get("NUBIRA_SMTP_PORT") or ("465" if implicit_tls else "587"))
    client = (smtplib.SMTP_SSL(host, port, context=context, timeout=15) if implicit_tls
              else smtplib.SMTP(host, port, timeout=15))
    with client:
        if not implicit_tls:
            client.starttls(context=context)
        if username := os.environ.get("NUBIRA_SMTP_USER"):
            client.login(username, os.environ["NUBIRA_SMTP_PASSWORD"])
        client.send_message(message)
