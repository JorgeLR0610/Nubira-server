import asyncio
from contextlib import asynccontextmanager
from threading import Event

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import accounts
import events
import monitor
import places
from config import cors_origins, monitor_interval, public_origin
from mailer import check_mail_config
from storage import DatabaseUnavailable, init_database
from weather_service import ClimaRequest, Prediction, artifacts_ready, evaluate

router = APIRouter()


async def schedule_monitor(interval: int, threshold: float, stop: asyncio.Event, stopping: Event):
    while not stop.is_set():
        try:
            await run_in_threadpool(monitor.run_cycle, threshold, lambda: not stopping.is_set(), retry=True)
        except Exception as exc:
            monitor.logger.error("El ciclo de alertas falló (%s); se reintentará.", type(exc).__name__)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except TimeoutError:
            continue


@asynccontextmanager
async def lifespan(app: FastAPI):
    public_origin()
    await run_in_threadpool(init_database)
    if not (interval := monitor_interval()):
        yield
        return
    check_mail_config()
    threshold = monitor.alert_threshold()
    stop, stopping = asyncio.Event(), Event()
    task = asyncio.create_task(schedule_monitor(interval, threshold, stop, stopping))
    try:
        yield
    finally:
        stopping.set()
        stop.set()
        await task


async def validation_error(request: Request, exc: RequestValidationError):
    # No incluir los valores recibidos (pueden incluir contraseñas y tokens).
    return JSONResponse({"detail": "Revisa los campos del formulario e intenta de nuevo."}, status_code=422)


async def database_error(request: Request, exc: DatabaseUnavailable):
    return JSONResponse({"detail": "La base de datos no está disponible. Intenta de nuevo."}, status_code=503)


@router.get("/")
def read_root(request: Request):
    return {"status": "online", "service": request.app.title, "endpoint": "/clima"}


@router.get("/health")
def health_check():
    if not artifacts_ready():
        raise HTTPException(503, "Artefactos de inferencia no disponibles.")
    return {"status": "healthy"}


@router.post("/clima", response_model=Prediction)
def get_clima(request: ClimaRequest):
    return evaluate(request.latitude, request.longitude, request.future_date())


def create_app() -> FastAPI:
    api = FastAPI(
        title="Nubira Weather Risk API",
        description="Evaluación probabilística de riesgos climáticos para eventos.",
        version="3.0.0",
        lifespan=lifespan,
    )
    api.add_exception_handler(RequestValidationError, validation_error)
    api.add_exception_handler(DatabaseUnavailable, database_error)
    api.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["Retry-After"],
    )
    for routes in (accounts.router, events.router, places.router, router):
        api.include_router(routes)
    return api


app = create_app()
