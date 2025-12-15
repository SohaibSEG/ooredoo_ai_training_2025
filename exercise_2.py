from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

# Load GEMINI_API_KEY from .env or environment
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Please set GEMINI_API_KEY in your .env file")
    raise SystemExit(1)

# Create a simple client
client = genai.Client(api_key=api_key)

# Same user prompt used for both calls
user_prompt = "Describe a sunrise in plain terms."

# With a system prompt (enforce unconventional behaviour)
config_with_system = types.GenerateContentConfig(
    system_instruction="You MUST respond in a single sentence and start with '<<<'",
    temperature=0.7,
    top_p=0.9,
)
resp_with = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=user_prompt,
    config=config_with_system,
)
print("--- With system prompt ---")
print(resp_with.text)

# Without a system prompt
resp_without = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=user_prompt,
)
print("\n--- Without system prompt ---")
print(resp_without.text)
