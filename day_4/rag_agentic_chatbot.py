"""Agentic retrieval example exposing the retriever as a tool."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_postgres import PGVector
from langchain_core.vectorstores import VectorStoreRetriever
from rag_pipeline import (
    build_embeddings,
    build_llm,
    resolve_pg_connection_string,
)

DEFAULT_CHROMA_DIR = Path(__file__).resolve().parent / "chroma_store"


@dataclass
class AgentArgs:
    store: Literal["pgvector", "chroma"]
    collection: str
    persist_dir: Optional[Path] = None


def parse_args() -> AgentArgs:
    parser = argparse.ArgumentParser(
        description="Chat with the PDFs using an agent that can call a retrieval tool."
    )
    parser.add_argument("--store", choices=["pgvector", "chroma"], required=True)
    parser.add_argument("--collection", required=True, help="Vector store collection name.")
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=DEFAULT_CHROMA_DIR,
        help="Chroma persistence directory (only used when --store=chroma).",
    )
    ns = parser.parse_args()
    return AgentArgs(
        store=ns.store,
        collection=ns.collection,
        persist_dir=ns.persist_dir,
    )


def build_vector_store(args: AgentArgs):
    embeddings = build_embeddings()
    if args.store == "pgvector":
        connection_string = resolve_pg_connection_string()
        return PGVector(
            collection_name=args.collection,
            connection_string=connection_string,
            embedding_function=embeddings,
        )

    persist_dir = args.persist_dir or DEFAULT_CHROMA_DIR
    if not persist_dir.exists():
        raise RuntimeError(
            f"Persist directory '{persist_dir}' does not exist. Ingest documents first."
        )
    return Chroma(
        collection_name=args.collection,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def create_retrieval_tool(retriever : VectorStoreRetriever):
    @tool("pdf_search")
    def pdf_search(query: str) -> str:
        """Searches the embedded PDF knowledge base for passages relevant to the query."""
        documents = retriever.invoke(query)
        if not documents:
            return "No relevant passages found in the PDFs."
        formatted = []
        for idx, doc in enumerate(documents, start=1):
            formatted.append(f"[{idx}] {doc.page_content.strip()}")
        return "\n\n".join(formatted)

    return pdf_search


def extract_final_message(result: dict) -> str:
    messages: List[BaseMessage] = result.get("messages", [])
    if not messages:
        return "No response."
    final_message = messages[-1]
    if isinstance(final_message.content, list):
        for part in final_message.content:
            text = part.get("text")
            if text:
                return text
        return "Could not parse tool response."
    return str(final_message.content)


def interactive_agent_chat(agent, label: str) -> None:
    history: List[BaseMessage] = []
    print(f"Agentic Retrieval Chat ({label})")
    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        user_message = HumanMessage(content=user_input)
        result = agent.invoke({"messages": history + [user_message]})
        answer = extract_final_message(result)
        print(f"Assistant: {answer}\n")

        history.append(user_message)
        history.append(AIMessage(content=answer))


def main() -> None:
    args = parse_args()
    vector_store = build_vector_store(args)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    retrieval_tool = create_retrieval_tool(retriever)

    llm = build_llm()
    agent_executor = create_agent(
        model=llm,
        tools=[retrieval_tool],
        system_prompt=(
            "You are a helpful research assistant. Use the available tools to answer questions "
            "about the ingested PDFs. If the information is not in the PDFs, say you don't know."
        ),
    )

    label = (
        f"{args.store} collection '{args.collection}'"
        if args.store == "pgvector"
        else f"chroma collection '{args.collection}' ({args.persist_dir})"
    )
    interactive_agent_chat(agent_executor, label=label)


if __name__ == "__main__":
    main()
