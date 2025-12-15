from dotenv import load_dotenv
import os

# LangChain imports: model wrapper, prompt utilities, message types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate

# Pydantic helps define and validate structured data models
from pydantic import BaseModel, Field
from typing import Literal

# 1) Load environment variables from a `.env` file in the project root.
#    Make sure your `.env` contains: GEMINI_API_KEY=your_key_here
load_dotenv()

# 2) Read the Gemini API key from the environment.
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    # Fail fast with a helpful message if the key isn't set.
    raise RuntimeError("Please set GEMINI_API_KEY")

# 3) Create a LangChain chat model instance for Gemini.
#    - `model`: which Gemini variant to use.
#    - `google_api_key`: credentials for the API.
#    - `temperature`: controls randomness (lower = more deterministic).
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=API_KEY,
    temperature=0.2,
)


# 4) Define the shape of the structured output we expect.
#    - `category`: one of three fixed values (Literal type enforces this).
#    - `urgency`: one of low/medium/high.
#    - `summary`: a short free-text sentence.
class TicketClassification(BaseModel):
    category: Literal["billing", "technical", "general"] = Field(
        description="Primary category of the support request"
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="Estimated urgency level"
    )
    summary: str = Field(
        description="One-sentence summary of the issue"
    )


# 5) Wrap the LLM to enforce it returns data matching `TicketClassification`.
#    Under the hood, LangChain guides the model to reply in a structured format,
#    then validates (and parses) it into a Pydantic object.
structured_llm = llm.with_structured_output(TicketClassification)

# 6) Build a simple chat-style prompt.
#    - SystemMessage: sets overall role/instructions for the assistant.
#    - HumanMessagePromptTemplate: inserts the user's message into the prompt.
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=(
        "You are a support dispatch assistant. Classify the support ticket into "
        "category and urgency, and provide a summary."
    )),
    HumanMessagePromptTemplate.from_template(template="{message}"),
])

# 7) Compose the prompt with the structured LLM.
#    Using the `|` operator creates a simple chain: prompt → structured_llm.
chain = prompt | structured_llm

# 8) Invoke the chain with an example message.
#    The result will be an instance of `TicketClassification`.
result = chain.invoke({
    "message": "My internet connection drops every evening after 7pm."
})

# 9) Print the structured result.
#    Pydantic models have a nice `model_dump()` method to convert to dicts.
print(result)
