from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("Please set GEMINI_API_KEY in your environment or .env file")
    raise SystemExit(1)

client = genai.Client(api_key=API_KEY)


def few_shot_classification(example_text: str):
    prompt = '''Classify the following text into NEUTRAL, POSITIVE, or NEGATIVE.

Examples:
Text: "I absolutely love this!" => POSITIVE
Text: "It's fine, nothing special." => NEUTRAL
Text: "This is the worst experience ever." => NEGATIVE

Now classify the new text:
Text: 
'''
    # Insert the example text to classify at the end
    full_prompt = prompt + f"{example_text}\n"

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=full_prompt,
    )

    print("--- Few-shot prompt ---")
    print("Input:", example_text)
    print("Model output:", resp.text)


if __name__ == "__main__":
    sample = "The movie had stunning visuals but left me bored overall."
    few_shot_classification(sample)
