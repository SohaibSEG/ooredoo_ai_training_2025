# Day 2 — LangChain Memory and Structuring

Examples using LangChain with Gemini for memory, structured output, and simple multi-step chains.

## Prereqs
- `.env` with `GEMINI_API_KEY=...`
- `pip install -r requirements.txt`

## Scripts
- `conversation_memory.py` — chat with managed message history stored per session using `RunnableWithMessageHistory` plus `FileChatMessageHistory`.
- `file_message_chat_history.py` — utility class persisting chat history to JSON on disk.
- `structured_output.py` — classify support tickets into a Pydantic model (`category`, `urgency`, `summary`).
- `two_chain_flow.py` — two-step chain: summarize a ticket, then assign priority; prints intermediate steps.

## Run
From repo root:
```
python day_2/conversation_memory.py
python day_2/structured_output.py
python day_2/two_chain_flow.py
```
( `file_message_chat_history.py` is imported by `conversation_memory.py` and not run directly.)
