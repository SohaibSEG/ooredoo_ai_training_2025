from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set GEMINI_API_KEY in your environment")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=API_KEY,
    temperature=0.0,
    max_output_tokens=150,
)

# Step 1: summarize the ticket
summary_prompt = PromptTemplate.from_template(
    """Summarize the following support ticket in one sentence.

Ticket:
{ticket}"""
)

summary_chain = summary_prompt | llm | StrOutputParser()

# Step 2: determine priority from the summary
priority_prompt = PromptTemplate.from_template(
    """Based on the summary below, determine the priority.
Respond with one word only: Low, Medium, or High.

Summary:
{summary}"""
)

priority_chain = priority_prompt | llm | StrOutputParser()

# Compose the workflow explicitly
workflow = (
    {"summary": summary_chain, "ticket": RunnablePassthrough()}
    | priority_chain
)

if __name__ == "__main__":
    ticket_text = "Network latency spikes every evening during peak hours."
    priority = workflow.invoke(ticket_text)
    print("Priority:", priority)