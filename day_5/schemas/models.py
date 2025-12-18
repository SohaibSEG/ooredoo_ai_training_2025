"""Pydantic models for API request/response validation."""

import uuid
from pydantic import BaseModel, Field


class EchoRequest(BaseModel):
    """Request model for echo endpoint."""
    message: str = Field(..., description="The message to echo back", min_length=1, max_length=1000)


class EchoResponse(BaseModel):
    """Response model for echo endpoint."""
    echo: str = Field(..., description="The echoed message")
    received_at: str = Field(..., description="Timestamp when the message was received")


class WeatherChatRequest(BaseModel):
    """Request model for weather agent chat."""
    message: str = Field(..., description="User message to the weather agent", min_length=1)
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Session ID for conversation memory"
    )


class WeatherChatResponse(BaseModel):
    """Response model for weather agent chat."""
    response: str = Field(..., description="Agent's response")
    session_id: str = Field(..., description="Session ID used for this conversation")

