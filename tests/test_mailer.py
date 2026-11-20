import logging
import ssl

import pytest

import mailer


def test_ses_starttls_authenticates_before_sending(monkeypatch):
    calls = []

    class SMTP:
        def __init__(self, host, port, *, timeout):
            calls.append((host, port, timeout))

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def starttls(self, *, context):
            assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
            calls.append("STARTTLS")

        def login(self, user, password):
            calls.append((user, password))

        def send_message(self, message):
            calls.append("send")
            assert message["From"] == "notificaciones@example.com"
            assert message["To"] == "recibe@example.com"
            assert message["Message-ID"] == "<test.nubira@nubira.example.com>"
            assert "Enlace de prueba" in message.get_content()

    for key, value in {
        "FRONTEND_URL": "https://nubira.example.com/",
        "MAIL_PROVIDER": "smtp",
        "MAIL_FROM": "notificaciones@example.com",
        "SMTP_HOST": "email-smtp.us-east-1.amazonaws.com",
        "SMTP_PORT": "587",
        "SMTP_USER": "mock-user",
        "SMTP_PASS": "mock-password",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("NUBIRA_SMTP_IMPLICIT_TLS", raising=False)
    monkeypatch.setattr(mailer.smtplib, "SMTP", SMTP)
    mailer.send_mail("recibe@example.com", "Verifica tu correo", "Enlace de prueba", message_id="test")
    assert calls == [
        ("email-smtp.us-east-1.amazonaws.com", 587, 15),
        "STARTTLS",
        ("mock-user", "mock-password"),
        "send",
    ]
    monkeypatch.setenv("SMTP_PASS", "")
    with pytest.raises(RuntimeError, match="no está configurado"):
        mailer.check_mail_config()


def test_console_defaults_to_local_and_does_not_log_personal_data(monkeypatch, caplog):
    monkeypatch.setenv("FRONTEND_URL", "http://localhost:3000")
    monkeypatch.delenv("MAIL_PROVIDER", raising=False)
    monkeypatch.delenv("NUBIRA_MAIL_MODE", raising=False)
    with caplog.at_level(logging.INFO, logger="mailer"):
        mailer.send_mail("privado@example.com", "Verifica tu cuenta", "enlace-privado")
    assert "Correo simulado" in caplog.text
    assert "privado" not in caplog.text
    monkeypatch.setenv("FRONTEND_URL", "https://nubira.example.com")
    with pytest.raises(RuntimeError, match="no está configurado"):
        mailer.check_mail_config()
