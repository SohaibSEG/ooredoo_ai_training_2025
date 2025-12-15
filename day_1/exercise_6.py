from dotenv import load_dotenv
from google import genai
from google.genai import types
import os
import json

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Please set GEMINI_API_KEY in your .env file")
    raise SystemExit(1)

client = genai.Client(api_key=api_key)


def classify_log(log_message: str) -> str:
    """
    Step 1: Classify the log message into ERROR, WARNING, INFO, or DEBUG.
    Returns just the log level as a string.
    """
    config = types.GenerateContentConfig(
        system_instruction="You are a log classifier. Respond with ONLY one of: ERROR, WARNING, INFO, or DEBUG. No explanations.",
    )
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Classify this log: {log_message}",
        config=config,
    )
    
    level = response.text.strip()
    return level


def handle_error_log(log_message: str) -> str:
    """Step 2a: Handle ERROR level logs."""
    config = types.GenerateContentConfig(
        system_instruction="You are an error analyst. Analyze this ERROR log and suggest immediate actions to fix it. Be concise.",
    )
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Log: {log_message}",
        config=config,
    )
    
    return response.text


def handle_warning_log(log_message: str) -> str:
    """Step 2b: Handle WARNING level logs."""
    config = types.GenerateContentConfig(
        system_instruction="You are a warning reviewer. Analyze this WARNING and explain potential risks. Be concise.",
    )
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Log: {log_message}",
        config=config,
    )
    
    return response.text


def handle_info_log(log_message: str) -> str:
    """Step 2c: Handle INFO level logs."""
    config = types.GenerateContentConfig(
        system_instruction="You are a log summarizer. Summarize this INFO log in one sentence.",
    )
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Log: {log_message}",
        config=config,
    )
    
    return response.text


def handle_debug_log(log_message: str) -> str:
    """Step 2d: Handle DEBUG level logs."""
    config = types.GenerateContentConfig(
        system_instruction="You are a debug expert. Summarize what this DEBUG log tells us in one sentence.",
    )
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Log: {log_message}",
        config=config,
    )
    
    return response.text


def process_log(log_message: str) -> dict:
    """
    Main flow: Classify the log, then branch to the appropriate handler.
    Returns a dict with the classification and analysis.
    """
    # Step 1: Classify
    level = classify_log(log_message)
    
    # Step 2: Branch based on level
    if level == "ERROR":
        analysis = handle_error_log(log_message)
    elif level == "WARNING":
        analysis = handle_warning_log(log_message)
    elif level == "INFO":
        analysis = handle_info_log(log_message)
    else:  # DEBUG or unknown
        analysis = handle_debug_log(log_message)
    
    return {
        "log_message": log_message,
        "classified_level": level,
        "analysis": analysis,
    }


if __name__ == "__main__":
    # Example logs to process
    logs = [
        "Database connection timeout after 30 seconds. Retries exhausted.",
        "Memory usage is at 85%, consider cleaning up old cache.",
        "User john.doe logged in successfully at 2025-12-15 10:30 UTC.",
        "Variable x was undefined, used default value 0.",
    ]
    
    print("=== Log Ingestion System ===\n")
    
    for log in logs:
        result = process_log(log)
        print(f"Log: {result['log_message']}")
        print(f"Level: {result['classified_level']}")
        print(f"Analysis: {result['analysis']}")
        print("-" * 60)
