from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import accounts
import events
import places
from mailer import public_origin
from storage import init_database
from weather_service import ClimaRequest, Prediction, artifacts_ready, evaluate


@asynccontextmanager
async def lifespan(app: FastAPI):
    public_origin()
    init_database()
    yield


app = FastAPI(
    title="Nubira Weather Risk API",
    description="Evaluación probabilística de riesgos climáticos para eventos.",
    version="3.0.0",
    lifespan=lifespan,
)
app.include_router(accounts.router)
app.include_router(events.router)
app.include_router(places.router)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # No incluir los valores recibidos (pueden incluir contraseñas y tokens).
    return JSONResponse({"detail": "Revisa los campos del formulario e intenta de nuevo."}, status_code=422)


@app.get("/")
def read_root():
    return {"status": "online", "service": app.title, "endpoint": "/clima"}


@app.get("/health")
def health_check():
    if not artifacts_ready():
        raise HTTPException(503, "Artefactos de inferencia no disponibles.")
    return {"status": "healthy"}


@app.post("/clima", response_model=Prediction)
def get_clima(request: ClimaRequest):
    return evaluate(request.latitude, request.longitude, request.future_date())
