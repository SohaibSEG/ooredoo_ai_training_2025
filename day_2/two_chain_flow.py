from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

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
{ticket_user}"""
)

def print_with_message_and_return(message) :
    def print_and_return(x):
        print(message, x)
        return x
    return print_and_return

summary_chain = summary_prompt |RunnableLambda(print_with_message_and_return("Priority Prompt:"))| llm | StrOutputParser() | RunnableLambda(print_with_message_and_return("Summary:"))

# Step 2: determine priority from the summary
priority_prompt = PromptTemplate.from_template(
    """Based on the summary below, determine the priority.
Respond with one word only: Low, Medium, or High.

Summary:
{summary}"""
)

priority_chain = priority_prompt | RunnableLambda(print_with_message_and_return("Priority Prompt:")) | llm | StrOutputParser() | RunnableLambda(print_with_message_and_return("Priority:"))

# Compose the workflow explicitly
workflow = (
    {"summary": summary_chain, "ticket": RunnablePassthrough()}
    | priority_chain
)

if __name__ == "__main__":
    ticket_text = (
        "Over the past two weeks, our team has experienced intermittent outages and significant slowdowns in the company VPN connection, particularly between 9am and 11am. "
        "Multiple employees have reported being unable to access internal resources, resulting in delays to critical project deliverables. "
        "Attempts to restart routers and switches have not resolved the issue. "
        "We suspect the problem may be related to recent network configuration changes or increased load during peak hours. "
        "Immediate assistance is required to diagnose and resolve the connectivity problems, as they are impacting productivity across several departments."
    )
    priority = workflow.invoke(ticket_text)
    print("Priority:", priority)