"""API routes module."""

from fastapi import APIRouter
from routes.echo import router as echo_router
from routes.weather import router as weather_router

# Create main router
router = APIRouter()

# Include sub-routers
router.include_router(echo_router, tags=["echo"])
router.include_router(weather_router, tags=["weather"])

__all__ = ["router"]

