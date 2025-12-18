# Day 5 — FastAPI Application

FastAPI application demonstrating routes, input/output validation with Pydantic, and a weather agent API based on Day 3.

## Prerequisites

- Python 3.8+
- `.env` file with `GEMINI_API_KEY=...`

## Setup

1. **Create and activate a virtual environment** (recommended to use `.venv` in this directory):

```bash
cd day_5
python -m venv .venv

# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:

Create a `.env` file in the `day_5` directory:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

## Running the Application

Start the FastAPI server:

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload
```

The API will be available at:
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

## API Endpoints

### Root
- `GET /` - API information and available endpoints

### Health Check
- `GET /health` - Health check endpoint

### Echo Route
- `POST /echo` - Simple echo endpoint with Pydantic validation
  - **Request body:**
    ```json
    {
      "message": "Hello, world!"
    }
    ```
  - **Response:**
    ```json
    {
      "echo": "Hello, world!",
      "received_at": "2024-01-01T12:00:00"
    }
    ```

### Weather Agent Chat
- `POST /weather/chat` - Weather agent API with conversation memory
  - **Request body:**
    ```json
    {
      "message": "What's the weather in Paris?",
      "session_id": "optional-session-id"
    }
    ```
  - **Response:**
    ```json
    {
      "response": "Weather in Paris: Cloudy, 12°C, light rain expected...",
      "session_id": "session-id-used"
    }
    ```
  - **Note:** If `session_id` is not provided, a new UUID will be generated. Use the same `session_id` to maintain conversation context.

## Features Demonstrated

1. **FastAPI Routes**: Simple REST API endpoints
2. **Pydantic Validation**: Input/output validation using Pydantic models
   - Field validation (min_length, max_length)
   - Type checking
   - Automatic API documentation
3. **Weather Agent API**: LangChain agent with:
   - Tool integration (weather checking)
   - Session-based memory
   - Conversation context management

## Testing the API

### Using curl

**Echo endpoint:**
```bash
curl -X POST "http://localhost:8000/echo" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello from curl!"}'
```

**Weather chat:**
```bash
curl -X POST "http://localhost:8000/weather/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in Dubai?"}'
```

**Weather chat with session (for conversation context):**
```bash
# First message
curl -X POST "http://localhost:8000/weather/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in Paris?", "session_id": "my-session-123"}'

# Follow-up message (uses same session_id)
curl -X POST "http://localhost:8000/weather/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What should I wear?", "session_id": "my-session-123"}'
```

### Using the Interactive Docs

Visit http://localhost:8000/docs to use the Swagger UI for interactive API testing.

## Project Structure

```
day_5/
├── main.py              # FastAPI application entry point
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── .env                # Environment variables (create this)
├── schemas/            # Pydantic models for request/response validation
│   ├── __init__.py
│   └── models.py       # EchoRequest, EchoResponse, WeatherChatRequest, WeatherChatResponse
├── routes/             # API route handlers
│   ├── __init__.py     # Router aggregation
│   ├── echo.py         # Echo endpoint
│   └── weather.py       # Weather agent chat endpoint
└── generation/         # AI agent and generation logic
    ├── __init__.py
    └── agent.py        # Weather agent implementation with LangChain
```

## Notes

- This project is fully decoupled from other days
- Uses LangChain 1.0 syntax consistent with the rest of the project
- Session memory is stored in-memory (will be lost on server restart)
- The weather data is hardcoded for demonstration purposes

