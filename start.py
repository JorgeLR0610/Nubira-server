import os
import uvicorn

if __name__ == "__main__":
    port_str = os.environ.get("PORT", "7860")
    try:
        port = int(port_str)
    except ValueError:
        port = 7860
    uvicorn.run("appPowerFastAPI:app", host="0.0.0.0", port=port)
