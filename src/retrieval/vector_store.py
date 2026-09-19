from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS
from src.config import FAISS_INDEX_DIR, RETRIEVAL_K


def create_or_update_vector_store(documents: List[Document], embedding_model: Embeddings, save_path: Optional[Path] = None) -> FAISS:
    """
    Build a FAISS vector store from document chunks and optionally persist it to disk.
    """
    if not documents:
        raise ValueError("Cannot create vector store from empty documents list.")

    target_dir = save_path or FAISS_INDEX_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Retrieval] Generating FAISS index for {len(documents)} document chunks...")
    vector_store = FAISS.from_documents(documents, embedding_model)

    vector_store.save_local(str(target_dir))
    print(f"[Retrieval] FAISS index saved successfully at: {target_dir}")
    return vector_store


def load_vector_store(embedding_model: Embeddings, index_path: Optional[Path] = None) -> Optional[FAISS]:
    """
    Load an existing persisted FAISS vector store from disk.
    Returns None if index directory doesn't exist or is empty.
    """
    target_dir = index_path or FAISS_INDEX_DIR
    index_file = target_dir / "index.faiss"

    if not index_file.exists():
        return None

    print(f"[Retrieval] Loading persisted FAISS index from: {target_dir}")
    return FAISS.load_local(
        str(target_dir),
        embedding_model,
        allow_dangerous_deserialization=True,
    )


def get_retriever(vector_store: FAISS, k: Optional[int] = None, search_type: str = "similarity") -> VectorStoreRetriever:
    """
    Convert vector store into a LangChain retriever.
    """
    top_k = k or RETRIEVAL_K
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": top_k},
    )
