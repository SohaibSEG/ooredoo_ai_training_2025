from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Union

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set GEMINI_API_KEY in your .env file.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=API_KEY,
    temperature=0.3,
)

@tool
def check_weather(location: str) -> str:
    """
    Return a simplified, hardcoded weather description for a specific city.
    The location must be a single city name (e.g., 'Paris', 'New York').
    """
    weather_data = {
        "paris": "Cloudy, 12°C, light rain expected",
        "new york": "Sunny, 18°C, clear skies",
        "london": "Rainy, 8°C, heavy rain",
        "dubai": "Hot and sunny, 35°C, no rain",
        "tokyo": "Cool, 15°C, partly cloudy",
    }
    
    city = location.strip().lower()

    return f"Weather in {location}: " + weather_data.get(
        city,
        "Mild, 15-20°C, partly cloudy"
    )

agent_executor = create_agent(
    model=llm,
    tools=[check_weather],
    system_prompt=(
        "you are a helpful assistant that provides weather information and clothing suggestions."
        "suggest appropriate clothing based on the weather conditions you provide."
        "ask if the user has any specific preferences or needs."
        "if the user has any specific preferences, take them into account when suggesting clothing."
        "always provide both the weather information and clothing suggestions in your responses."
    ),
)

SESSION_STORE: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Retrieves or creates an InMemoryChatMessageHistory object for a given session ID."""
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return SESSION_STORE[session_id]

agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="messages", 
)

def extract_clean_text(result: dict) -> str:
    """
    Robustly extracts the clean text string from the agent's complex output structure.
    This handles the inconsistency between simple string and structured list outputs.
    """
    final_message: BaseMessage = result.get("messages", [])[-1]
    final_message_content: Union[str, List[Dict[str, Any]]] = final_message.content

    if isinstance(final_message_content, list) and final_message_content:
        # Handles the structured list format (common after tool use)
        return final_message_content[0].get("text", "Error: Could not parse structured text response.")
    
    # Handles the simple string format (common for simple chat turns)
    return str(final_message_content)

if __name__ == "__main__":
    SESSION_ID = "bootcamp-user-1" 
    
    print("=" * 60)
    print("Weather & Clothing Agent")
    print("=" * 60)
    print(f"Current Session ID: {SESSION_ID}")
    print("Type 'exit' to quit. Test by asking for weather and then a follow-up.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        input_message = {"messages": [HumanMessage(content=user_input)]}
        config = {"configurable": {"session_id": SESSION_ID}}

        result = agent_with_memory.invoke(input_message, config=config)

        print("-" * 20)
        print("DEBUG: Raw Agent Output (Internal LangChain format):")
        print(result)
        print("-" * 20)
        
        final_answer = extract_clean_text(result)
        print(f"Assistant: {final_answer}\n")