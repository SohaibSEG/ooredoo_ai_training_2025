"""Echo route endpoints."""

from datetime import datetime
from fastapi import APIRouter
from schemas.models import EchoRequest, EchoResponse

router = APIRouter(prefix="/echo", tags=["echo"])


@router.post("", response_model=EchoResponse)
async def echo(request: EchoRequest):
    """
    Simple echo route demonstrating input/output validation with Pydantic.
    
    Takes a message and returns it along with a timestamp.
    """
    return EchoResponse(
        echo=request.message,
        received_at=datetime.now().isoformat()
    )

