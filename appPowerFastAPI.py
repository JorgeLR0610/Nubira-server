from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from neuralNetwork.database_operations import insert_data, delete_data
from neuralNetwork.weather_predictions import predict_by_date
import powerClient

app = FastAPI()

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # o ["http://localhost:3000"] si quieres ser más estricto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Esquema del request ------------------
class ClimaRequest(BaseModel):
    latitude: float = Field(..., description="Latitude", ge=-90, le=90)
    longitude: float = Field(..., description="Longitude", ge=-180, le=180)
    day: int = Field(..., description="Day", ge=1, le=31)
    month: int = Field(..., description="Month", ge=1, le=12)
    year: int = Field(default=2025, description="Year for prediction")

# ------------------ Endpoint principal ------------------
@app.post("/clima")
def get_clima(request: ClimaRequest):
    try:
        # 1️⃣ Obtener datos históricos
        start = datetime(1981, 9, 6)
        end = datetime(2024, 10, 7)
        raw_data = powerClient.fetch_power_daily(
            request.latitude, request.longitude, start, end
        )

        # 2️⃣ Convertir y filtrar por mes y día
        df = powerClient.convert_to_csv(raw_data, filter_month=request.month, filter_day=request.day)
        delete_data()
        # 3️⃣ Guardar datos en SQLite (si quieres)
        if not df.empty:
            insert_data(df)

        # 4️⃣ Predicción usando el modelo entrenado
        resultado = predict_by_date(request.year, request.month, request.day)
        pred = resultado["prediccion"]

        # 5️⃣ Retornar resultado
        return {
            "temp_max": pred[0],
            "temp_min": pred[1],
            "precipitacion": pred[2],
            "precipitacion_nieve": pred[3],
            "vel_viento": pred[4],
            "mejor_loss": resultado["mejor_loss"],
            "mejor_epoca": resultado["mejor_epoca"],
            "mejor_batch": resultado["mejor_batch"]
        }

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=f"Error en el procesamiento: {str(e)}")
