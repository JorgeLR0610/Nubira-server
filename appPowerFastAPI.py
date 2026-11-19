from datetime import date, datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from neuralNetwork.database_operations import delete_data, insert_data
from neuralNetwork.weather_predictions import predict_by_date
import powerClient
from predict import predict_daily_risk_probabilities

app = FastAPI(
    title="Nubira Weather Risk API",
    description="API de pronóstico probabilístico de riesgos climáticos con Deep Learning",
    version="2.0.0",
)

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    year: int = Field(default=2027, description="Year for prediction")


# ------------------ Endpoints auxiliares ------------------
@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Nubira Deep Learning Weather Risk API",
        "endpoint": "/clima",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# ------------------ Endpoint principal ------------------
@app.post("/clima")
def get_clima(request: ClimaRequest):
    try:
        target_year = request.year
        today = date.today()

        # Si el año enviado ya caducó (como el antiguo default 2025), proyectar a futuro
        try:
            check_date = date(target_year, request.month, request.day)
            if check_date <= today and target_year <= 2025:
                target_year = today.year + 1
        except Exception:
            pass

        target_date_str = f"{target_year:04d}-{request.month:02d}-{request.day:02d}"

        # Predicción usando el modelo de deep learning pre-entrenado
        resultado = predict_daily_risk_probabilities(
            lat=request.latitude,
            lon=request.longitude,
            target_date_str=target_date_str,
        )

        # Retornar el pronóstico probabilístico diario al frontend
        return resultado

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print("Error en /clima:", e)
        raise HTTPException(status_code=500, detail=f"Error en el procesamiento: {str(e)}")
