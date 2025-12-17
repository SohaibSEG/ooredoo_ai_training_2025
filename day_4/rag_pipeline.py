"""Shared RAG pipeline utilities and ingestion CLI for LangChain vector stores."""

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Sequence
from urllib.parse import quote_plus

from dotenv import load_dotenv

from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment.")

DEFAULT_CHROMA_DIR = Path(__file__).resolve().parent / "chroma_store"


@dataclass
class IngestArgs:
    store: Literal["pgvector", "chroma"]
    pdf_dir: Path
    collection: str
    chunk_size: int
    chunk_overlap: int
    persist_dir: Optional[Path]


def build_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GEMINI_API_KEY,
    )


def build_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
    )


def resolve_pg_connection_string() -> str:
    direct = os.getenv("PGVECTOR_CONNECTION_STRING")
    if direct:
        return direct

    host = os.getenv("PG_HOST")
    database = os.getenv("PG_DATABASE")
    user = os.getenv("PG_USER")
    password = os.getenv("PG_PASSWORD")
    port = os.getenv("PG_PORT", "5432")
    missing = [name for name, value in {
        "PG_HOST": host,
        "PG_DATABASE": database,
        "PG_USER": user,
        "PG_PASSWORD": password,
    }.items() if not value]
    if missing:
        raise RuntimeError(
            "Set PGVECTOR_CONNECTION_STRING or PG_* variables for database access "
            f"(missing: {', '.join(missing)})"
        )
    user_enc = quote_plus(user)
    password_enc = quote_plus(password)
    database_enc = quote_plus(database)
    return (
        f"postgresql+psycopg://{user_enc}:{password_enc}@{host}:{port}/{database_enc}"
    )


def load_pdf_documents(pdf_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())
    if not docs:
        raise RuntimeError(f"No PDF files found in {pdf_dir}.")
    return docs


def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(list(documents))


def format_documents(docs: Sequence[Document]) -> str:
    formatted = []
    for idx, doc in enumerate(docs, start=1):
        formatted.append(f"[{idx}] {doc.page_content.strip()}")
    return "\n\n".join(formatted)


def ingest_pdfs(
    vector_store: VectorStore,
    pdf_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory '{pdf_dir}' does not exist.")
    if not pdf_dir.is_dir():
        raise RuntimeError(f"'{pdf_dir}' is not a directory.")

    documents = load_pdf_documents(pdf_dir)
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    vector_store.add_documents(chunks)
    return len(chunks)

def parse_args() -> IngestArgs:
    parser = argparse.ArgumentParser(description="Ingest PDFs into pgvector or Chroma stores")
    parser.add_argument(
        "--store",
        choices=["pgvector", "chroma"],
        required=True,
        help="Vector store backend to use.",
    )
    parser.add_argument("--pdf-dir", type=Path, required=True, help="Directory containing PDFs to ingest.")
    parser.add_argument("--collection", required=True, help="Target collection name.")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=DEFAULT_CHROMA_DIR,
        help=f"Directory to persist the Chroma DB (default: {DEFAULT_CHROMA_DIR})",
    )
    ns = parser.parse_args()
    return IngestArgs(
        store=ns.store,
        pdf_dir=ns.pdf_dir,
        collection=ns.collection,
        chunk_size=ns.chunk_size,
        chunk_overlap=ns.chunk_overlap,
        persist_dir=ns.persist_dir,
    )


def main() -> None:
    args = parse_args()
    embeddings = build_embeddings()

    if args.store == "pgvector":
        connection_string = resolve_pg_connection_string()
        vector_store = PGVector(
            collection_name=args.collection,
            connection_string=connection_string,
            embedding_function=embeddings,
        )
        label = f"pgvector collection '{args.collection}'"
    else:
        persist_dir = args.persist_dir or DEFAULT_CHROMA_DIR
        persist_dir.mkdir(parents=True, exist_ok=True)
        vector_store = Chroma(
            collection_name=args.collection,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )
        label = f"Chroma collection '{args.collection}' (persist dir: {persist_dir})"

    chunk_count = ingest_pdfs(
        vector_store=vector_store,
        pdf_dir=args.pdf_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Ingestion complete. Stored {chunk_count} chunks in {label}.")


if __name__ == "__main__":
    main()
