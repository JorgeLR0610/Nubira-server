import logging
import os
import time
import uvicorn

if __name__ == "__main__":
    os.environ["TZ"] = "UTC"
    if hasattr(time, "tzset"):
        time.tzset()
    port_str = os.environ.get("PORT", "8000")
    try:
        port = int(port_str)
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        raise SystemExit("PORT debe ser un puerto TCP entre 1 y 65535.") from None

    host = os.environ.get("NUBIRA_BIND_HOST", "0.0.0.0")
    logging.basicConfig(level=logging.INFO)
    print(f"Iniciando Nubira Weather Risk API en {host}:{port}...")
    uvicorn.run("appPowerFastAPI:app", host=host, port=port, workers=1)
