import os
import time
import uvicorn

if __name__ == "__main__":
    os.environ["TZ"] = "UTC"
    if hasattr(time, "tzset"):
        time.tzset()
    port_str = os.environ.get("PORT", "7860")
    try:
        port = int(port_str)
    except ValueError:
        port = 7860

    host = os.environ.get("NUBIRA_BIND_HOST", "127.0.0.1")
    print(f"Iniciando Nubira Weather Risk API en {host}:{port}...")
    uvicorn.run("appPowerFastAPI:app", host=host, port=port)
