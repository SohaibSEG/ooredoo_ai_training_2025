"""Weather agent implementation using LangChain."""

import os
from typing import List, Dict, Any, Union
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set GEMINI_API_KEY in your .env file.")

# Initialize LLM
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


# Session store for agent memory
SESSION_STORE: Dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Retrieves or creates an InMemoryChatMessageHistory object for a given session ID."""
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return SESSION_STORE[session_id]


def _create_agent():
    """Create and configure the weather agent."""
    agent_executor = create_agent(
        model=llm,
        tools=[check_weather],
        system_prompt=(
            "you are a helpful assistant that provides weather information and clothing suggestions. "
            "suggest appropriate clothing based on the weather conditions you provide. "
            "ask if the user has any specific preferences or needs. "
            "if the user has any specific preferences, take them into account when suggesting clothing. "
            "always provide both the weather information and clothing suggestions in your responses."
        ),
    )
    
    agent_with_memory = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="messages",
    )
    
    return agent_with_memory


# Singleton agent instance
_agent_instance = None


def get_weather_agent():
    """Get or create the weather agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = _create_agent()
    return _agent_instance


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


def chat_with_agent(message: str, session_id: str) -> str:
    """
    Chat with the weather agent.
    
    Args:
        message: User message to send to the agent
        session_id: Session ID for conversation memory
        
    Returns:
        Agent's response as a string
    """
    agent = get_weather_agent()
    input_message = {"messages": [HumanMessage(content=message)]}
    config = {"configurable": {"session_id": session_id}}
    
    result = agent.invoke(input_message, config=config)
    return extract_clean_text(result)

