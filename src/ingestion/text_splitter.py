from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    Split loaded documents into smaller, overlapping chunks suitable for embedding and retrieval.
    """
    size = chunk_size or CHUNK_SIZE
    overlap = chunk_overlap or CHUNK_OVERLAP

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"[Ingestion] Split {len(documents)} document(s) into {len(chunks)} chunks (size={size}, overlap={overlap}).")
    return chunks
