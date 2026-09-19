from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)


def load_single_document(file_path: str | Path) -> List[Document]:
    """Load a single document based on its extension."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found at: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix in [".docx", ".doc"]:
        loader = Docx2txtLoader(str(path))
    elif suffix in [".txt", ".md", ".markdown"]:
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        # Fallback to TextLoader
        loader = TextLoader(str(path), encoding="utf-8")

    docs = loader.load()
    # Enrich metadata with relative path and file name
    for doc in docs:
        doc.metadata["source_name"] = path.name
    return docs


def load_documents_from_directory(directory_path: str | Path) -> List[Document]:
    """Load all supported documents from a directory."""
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory}")

    supported_extensions = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown"}
    all_docs: List[Document] = []

    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                docs = load_single_document(file_path)
                all_docs.extend(docs)
                print(f"[Ingestion] Loaded {len(docs)} page(s)/section(s) from {file_path.name}")
            except Exception as e:
                print(f"[Ingestion Warning] Could not load {file_path.name}: {e}")

    return all_docs
