# AI Training Exercises

This repository contains small, focused Python scripts demonstrating prompting patterns, LangChain basics, and RAG flows across four days. Each day now has its own README with details and commands.

## Setup

1) Create and activate a virtual environment.

2) Install dependencies:

```
pip install -r requirements.txt
```

3) Add your Gemini key to `.env`:

```
GEMINI_API_KEY=your_key_here
```

4) For pgvector examples (optional), set database access env vars or `PGVECTOR_CONNECTION_STRING`.

## Daily guides

- Day 1: prompting patterns — see [day_1/README.md](day_1/README.md)
- Day 2: LangChain memory, structured outputs, and simple chains — see [day_2/README.md](day_2/README.md)
- Day 3: agent with a weather tool — see [day_3/README.md](day_3/README.md)
- Day 4: RAG ingestion and chat (pgvector or Chroma), plus agentic retrieval — see [day_4/README.md](day_4/README.md)
