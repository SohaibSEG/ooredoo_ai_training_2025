"""CLI entry point for chatting with pgvector or Chroma PDF RAG collections."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from langchain_chroma import Chroma
from langchain_postgres import PGVector

from rag_pipeline import (
    interactive_chat,
    build_embeddings,
    resolve_pg_connection_string,
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
