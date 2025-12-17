# Day 3 — Agent with Weather Tool

A LangChain agent that calls a simple weather tool and keeps conversational memory per session.

## Prereqs
- `.env` with `GEMINI_API_KEY=...`
- `pip install -r requirements.txt`

## Script
- `weather_agent.py` — agent with a `check_weather` tool returning canned conditions plus clothing suggestions; uses `RunnableWithMessageHistory` for session memory.

## Run
From repo root:
```
python day_3/weather_agent.py
```
Then ask about weather by city; type `exit` to quit.
