"""
FastAPI application entry point.
"""

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
from loguru import logger

from config import settings
from utils.logger import setup_logger
from api.routes import translate, animation, system, recognize, sigml_translate
from api.websocket import realtime_translate_ws


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logger()
    logger.info(f"🚀 {settings.app_name} v{settings.app_version} starting...")

    # Bootstrap NLTK data in background
    try:
        from services.nlp_engine import _ensure_nltk
        _ensure_nltk()
        logger.info("✅ NLTK data ready")
    except Exception as e:
        logger.warning(f"⚠️  NLTK bootstrap warning: {e}")

    logger.info("✅ Server ready")
    yield
    logger.info("Server shutting down")


app = FastAPI(
    title="ISL Translation System API",
    description="AI-powered Indian Sign Language translation — text/voice to animated ISL",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(translate.router)
app.include_router(animation.router)
app.include_router(system.router)
app.include_router(recognize.router)
app.include_router(sigml_translate.router)

# ── Static files (SiGML sign assets) ─────────────────────────────────────────

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
SIGNFILES_DIR = PROJECT_DIR / "frontend" / "public" / "SignFiles"

app.mount("/SignFiles", StaticFiles(directory=str(SIGNFILES_DIR)), name="signfiles")


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/realtime/translate")
async def ws_translate(websocket: WebSocket):
    await realtime_translate_ws(websocket)


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "status": "/system/status",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
