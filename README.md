# Exercise Set — Prompting Patterns

This workspace contains small Python examples demonstrating different prompting techniques using the `google-genai` client.

Files:

- `exercise_1.py` — simple generation (already present).
- `exercise_2.py` — system prompt vs no system prompt.
- `exercise_3.py` — few-shot prompting.
- `exercise_4.py` — chain-of-thought style prompting.
- `exercise_5.py` — multi-turn chat where history is stored as dicts and resent each interaction.

Setup:

1. Create a virtual environment and activate it.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Gemini key:

```
GEMINI_API_KEY=your_key_here
```

Run:

```bash
python exercise_2.py
python exercise_3.py
python exercise_4.py
python exercise_5.py
```

Notes:
- These examples use `generate_content` with a simple `contents` string. They are intentionally small and meant for learning.
- Exercise 5 demonstrates a naive way to include message history by formatting it into a single prompt string.
