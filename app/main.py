from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

from core.config import settings
from core.database import engine, Base
from endpoints.endpoints import router
from services.proctor_service import ProctorService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

proctor_service = ProctorService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    Base.metadata.create_all(bind=engine)
    logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info("Database initialized")
    
    # Initialize proctor service models
    await proctor_service.initialize_models()
    
    yield
    
    # Shutdown
    proctor_service.cleanup()
    logger.info("Application shutdown complete")

app = FastAPI(
    title="Proctoring System API",
    version=settings.VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)