import logging
import os
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formatdate
from urllib.parse import urlsplit
from uuid import uuid4

from config import mail_settings

logger = logging.getLogger(__name__)


def check_mail_config():
    return mail_settings()


def send_mail(to: str, subject: str, body: str, message_id: str | None = None):
    settings = check_mail_config()
    if settings.mode == "console":
        logger.info("Correo simulado en modo local; no se realizó ningún envío.")
        return
    message = EmailMessage()
    message["From"] = settings.sender
    message["To"] = to
    message["Subject"] = subject
    message["Date"] = formatdate(usegmt=True)
    message_id = message_id or uuid4().hex
    message["Message-ID"] = f"<{message_id}.nubira@{urlsplit(settings.origin).hostname}>"
    message.set_content(body)

    if (outbox := settings.directory) is not None:
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

    context = ssl.create_default_context()
    client = (smtplib.SMTP_SSL(settings.host, settings.port, context=context, timeout=15) if settings.implicit_tls
              else smtplib.SMTP(settings.host, settings.port, timeout=15))
    with client:
        if not settings.implicit_tls:
            client.starttls(context=context)
        client.login(settings.user, settings.password)
        client.send_message(message)
