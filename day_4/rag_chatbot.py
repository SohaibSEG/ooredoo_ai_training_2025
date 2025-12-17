"""CLI entry point for chatting with pgvector or Chroma PDF RAG collections."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from langchain_chroma import Chroma
from langchain_postgres import PGVector
from langchain_core.vectorstores import VectorStore
from langchain_core.prompts import ChatPromptTemplate

from operator import itemgetter
from pathlib import Path
from typing import  List, Literal, Optional


from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import VectorStore

from langchain_postgres import PGVector


from rag_pipeline import (
    build_llm,
    build_embeddings,
    resolve_pg_connection_string,
    format_documents,
)


@dataclass
class ChatArgs:
    store: Literal["pgvector", "chroma"]
    collection: str
    persist_dir: Optional[Path]


DEFAULT_CHROMA_DIR = Path(__file__).resolve().parent / "chroma_store"


def parse_args() -> ChatArgs:
    parser = argparse.ArgumentParser(description="Chat with pgvector or Chroma PDF RAG collections")
    parser.add_argument(
        "--store",
        choices=["pgvector", "chroma"],
        required=True,
        help="Vector store backend to use.",
    )
    parser.add_argument("--collection", required=True, help="Target collection to chat with.")
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=DEFAULT_CHROMA_DIR,
        help=f"Directory where Chroma persisted data (default: {DEFAULT_CHROMA_DIR})",
    )

    ns = parser.parse_args()
    return ChatArgs(
        store=ns.store,
        collection=ns.collection,
        persist_dir=ns.persist_dir,
    )



def build_chat_chain(vector_store: VectorStore):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that answers with grounded information. "
                "Use this context to answer the user's question:\n{context}\n"
                "If the answer cannot be found in the context, say you don't know.",
            ),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )

    llm = build_llm()
    rag_chain = (
        {
            "context": itemgetter("question")
            | retriever
            | RunnableLambda(format_documents),
            "question": itemgetter("question"),
            "history": itemgetter("history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def interactive_chat(vector_store: VectorStore, target_label: Optional[str] = None) -> None:
    rag_chain = build_chat_chain(vector_store)
    history: List[BaseMessage] = []
    label = target_label or "the loaded collection"
    print(f"Chatting with {label}")
    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        response = rag_chain.invoke(
            {
                "question": user_input,
                "history": history,
            }
        )
        print(f"Assistant: {response}\n")
        history.extend([HumanMessage(content=user_input), AIMessage(content=response)])



def main() -> None:
    args = parse_args()
    embeddings = build_embeddings()

    if args.store == "pgvector":
        connection_string = resolve_pg_connection_string()

    if args.store == "pgvector":
        vector_store = PGVector(
            collection_name=args.collection,
            connection_string=connection_string,
            embedding_function=embeddings,
        )
        label = f"pgvector collection '{args.collection}'"
    else:
        persist_dir = args.persist_dir or DEFAULT_CHROMA_DIR
        if not persist_dir.exists():
            raise RuntimeError(
                f"Persist directory '{persist_dir}' does not exist. Ingest documents first."
            )
        vector_store = Chroma(
            collection_name=args.collection,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )
        label = f"Chroma collection '{args.collection}' (persist dir: {persist_dir})"

    interactive_chat(vector_store=vector_store, target_label=label)


if __name__ == "__main__":
    main()
