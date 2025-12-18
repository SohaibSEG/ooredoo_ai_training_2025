"""Weather agent chat route endpoints."""

from fastapi import APIRouter, HTTPException
from schemas.models import WeatherChatRequest, WeatherChatResponse
from generation.agent import chat_with_agent

router = APIRouter(prefix="/weather", tags=["weather"])


@router.post("/chat", response_model=WeatherChatResponse)
async def weather_chat(request: WeatherChatRequest):
    """
    Weather agent chat endpoint based on Day 3 weather agent.
    
    Uses LangChain agent with memory to provide weather information and clothing suggestions.
    Maintains conversation context per session_id.
    """
    try:
        response_text = chat_with_agent(request.message, request.session_id)
        
        return WeatherChatResponse(
            response=response_text,
            session_id=request.session_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing weather chat request: {str(e)}"
        ) from e

