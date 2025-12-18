"""FastAPI application entry point."""

from fastapi import FastAPI
from routes import router

# Initialize FastAPI app
app = FastAPI(
    title="Day 5 AI Training API",
    description="FastAPI app demonstrating routes, validation, and weather agent",
    version="1.0.0",
)

# Include routers
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "Day 5 AI Training API",
        "endpoints": {
            "echo": "/echo",
            "weather_chat": "/weather/chat",
            "docs": "/docs",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "day_5_api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
