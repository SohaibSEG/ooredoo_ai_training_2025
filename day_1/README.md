# Day 1 — Prompting Patterns

Small, self-contained scripts using the Google Gemini API to illustrate prompting patterns.

## Prereqs
- `.env` with `GEMINI_API_KEY=...`
- `pip install -r requirements.txt`

## Scripts
- `exercise_1.py` — single-turn generation: short poem.
- `exercise_2.py` — compare responses with vs. without a system prompt.
- `exercise_3.py` — few-shot sentiment classification.
- `exercise_4.py` — chain-of-thought style reasoning prompt.
- `exercise_5.py` — multi-turn chat; history manually flattened into the prompt.
- `exercise_6.py` — two-step log handling: classify log level, then branch to tailored analysis.

## Run
From repo root:
```
python day_1/exercise_1.py
python day_1/exercise_2.py
python day_1/exercise_3.py
python day_1/exercise_4.py
python day_1/exercise_5.py
python day_1/exercise_6.py
```
