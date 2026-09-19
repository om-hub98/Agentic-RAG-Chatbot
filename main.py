"""
Main entry point for Simple LangChain RAG System.
Demonstrates the complete pipeline:
1. Ingestion: Load and split documents
2. Embedding: Generate vector embeddings
3. Retrieval: Store and retrieve using FAISS
4. Generation: Synthesize answers using LangChain LCEL + LLM
"""

from src.config import DOCUMENTS_DIR, FAISS_INDEX_DIR
from src.ingestion import load_documents_from_directory, split_documents
from src.embedding import get_embedding_model
from src.retrieval import create_or_update_vector_store, load_vector_store, get_retriever
from src.generation import get_chat_model, create_rag_chain_with_sources


def setup_rag_pipeline(force_reindex: bool = False):
    """
    Orchestrate the ingestion, embedding, and vector store setup.
    """
    print("\n" + "=" * 60)
    print("[*] Initializing Simple LangChain RAG System")
    print("=" * 60)

    # 1. Initialize Embedding Model
    embedder = get_embedding_model()

    # 2. Check if a FAISS index already exists
    vector_store = None
    if not force_reindex:
        vector_store = load_vector_store(embedding_model=embedder)

    # 3. If no index exists or reindex requested, run ingestion pipeline
    if vector_store is None:
        print("\n[Step 1: Data Ingestion]")
        raw_documents = load_documents_from_directory(DOCUMENTS_DIR)
        if not raw_documents:
            print(f"[!] No documents found in '{DOCUMENTS_DIR}'. Please add documents first.")
            return None

        chunks = split_documents(raw_documents)

        print("\n[Step 2 & 3: Embedding & Vector Storage]")
        vector_store = create_or_update_vector_store(
            documents=chunks,
            embedding_model=embedder,
            save_path=FAISS_INDEX_DIR,
        )
    else:
        print("\n[Step 2 & 3: Vector Store]")
        print("[+] Loaded existing FAISS index from disk.")

    # 4. Create Retriever
    retriever = get_retriever(vector_store, k=3)

    # 5. Initialize LLM & Construct RAG Chain
    print("\n[Step 4: Generation Pipeline]")
    llm = get_chat_model(temperature=0.2)
    rag_chain = create_rag_chain_with_sources(retriever=retriever, llm=llm)

    print("[+] RAG Pipeline ready!\n")
    return rag_chain


def main():
    rag_chain = setup_rag_pipeline()
    if not rag_chain:
        return

    # Sample query to test the pipeline
    while True:
        user_query = input("Ask a Question: ")

        result = rag_chain.invoke({"question": user_query})

        print(f"\n[*] Answer:\n{result['answer']}")
        print("\n" + "=" * 60)
        print("[*] Retrieved Source Documents:")
        for idx, doc in enumerate(result["source_documents"], 1):
            source = doc.metadata.get("source_name", "Unknown")
            snippet = doc.page_content[:150].strip().replace("\n", " ")
            print(f"  [{idx}] Source: {source} | Snippet: {snippet}...")
        print("=" * 60)

        if user_query == "exit":    
            break


if __name__ == "__main__":
    main()
