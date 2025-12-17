# Day 4 — RAG Ingestion and Chat (pgvector or Chroma)

RAG utilities and CLIs for ingesting PDFs and chatting over vector stores, plus an agentic retrieval example.

## Prereqs
- `.env` with `GEMINI_API_KEY=...`
- `pip install -r requirements.txt`
- For `--store=pgvector`: set `PGVECTOR_CONNECTION_STRING` or `PG_*` env vars.

## Ingestion
Use `rag_pipeline.py` to ingest PDFs into either backend.
```
# Chroma (default persist dir: day_4/chroma_store)
python day_4/rag_pipeline.py --store=chroma --pdf-dir ./day_4/documents --collection day-4

# pgvector
python day_4/rag_pipeline.py --store=pgvector --pdf-dir ./day_4/documents --collection day-4
```

## Chat
Use `rag_chatbot.py` to chat over an existing collection.
```
# Chroma chat
python day_4/rag_chatbot.py --store=chroma --collection day-4

# pgvector chat
python day_4/rag_chatbot.py --store=pgvector --collection day-4
```

## Agentic retrieval
`rag_agentic_chatbot.py` exposes the retriever as a tool within an agent.
```
# Chroma
python day_4/rag_agentic_chatbot.py --store=chroma --collection day-4

# pgvector
python day_4/rag_agentic_chatbot.py --store=pgvector --collection day-4
```

Notes:
- Ensure the collection was ingested with the same embedding model used at query time.
- The Chroma persist dir can be overridden with `--persist-dir` where applicable.
