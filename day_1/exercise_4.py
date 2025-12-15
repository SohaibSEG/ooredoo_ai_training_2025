from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("Please set GEMINI_API_KEY in your environment or .env file")
    raise SystemExit(1)

client = genai.Client(api_key=API_KEY)


def chain_of_thought_example():
    prompt = (
        "Solve the puzzle and show your reasoning step-by-step.\n"
        "If 3 machines take 3 minutes to make 3 widgets, how long will 100 machines take to make 100 widgets?\n"
        "Provide the answer and show the chain of thought (step-by-step reasoning)."
    )

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    print("--- Chain of Thought Example ---")
    print(resp.text)


if __name__ == "__main__":
    chain_of_thought_example()
