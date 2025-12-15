from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("Please set GEMINI_API_KEY in your environment or .env file")
    raise SystemExit(1)

client = genai.Client(api_key=API_KEY)


def build_contents_from_history(history):
    """Convert a list of message dicts into a single contents string.

    Each message in history is a dict: {'role': 'user'|'assistant', 'content': str}
    """
    lines = []
    for msg in history:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        lines.append(f"{role.capitalize()}: {content}")
    return "\n\n".join(lines)


def chat_loop():
    print("Starting multi-turn chat. Type 'exit' to quit.")
    history = []

    while True:
        user_input = input("You: ")
        if not user_input:
            continue
        if user_input.strip().lower() in ("exit", "quit"):
            print("Goodbye")
            break

        # Save user message
        history.append({'role': 'user', 'content': user_input})

        # Build a contents string from history and send it each time
        contents = build_contents_from_history(history) + "\n\nAssistant:"

        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=150,
            ),
        )

        assistant_text = resp.text.strip()
        print("Assistant:", assistant_text)

        # Save assistant response to history
        history.append({'role': 'assistant', 'content': assistant_text})


if __name__ == "__main__":
    chat_loop()
