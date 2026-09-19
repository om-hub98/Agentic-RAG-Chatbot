from typing import List
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.vectorstores import VectorStoreRetriever


DEFAULT_RAG_PROMPT = """You are a helpful and precise assistant. Use the following pieces of retrieved context to answer the user's question.
If you do not know the answer based on the context, state that you don't know rather than fabricating an answer.
Keep your response concise, professional, and well-structured. Whenever possible, mention which document source provided the information.

Context:
{context}

Question:
{question}

Answer:"""


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a clean readable string with source metadata."""
    formatted_chunks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_name", "Unknown Source")
        page = doc.metadata.get("page")
        page_info = f" (Page {page + 1})" if page is not None else ""
        formatted_chunks.append(f"--- Document {i} [{source}{page_info}] ---\n{doc.page_content.strip()}")
    return "\n\n".join(formatted_chunks)


def create_rag_chain(
    retriever: VectorStoreRetriever,
    llm: BaseChatModel,
    prompt_template: str = DEFAULT_RAG_PROMPT,
):
    """
    Constructs a modern LangChain Expression Language (LCEL) RAG chain.
    Returns a Runnable that accepts {"question": "..."} and outputs the answer string,
    while also exposing a helper to retrieve with sources.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Standard answer generation chain
    rag_chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def create_rag_chain_with_sources(
    retriever: VectorStoreRetriever,
    llm: BaseChatModel,
    prompt_template: str = DEFAULT_RAG_PROMPT,
):
    """
    Constructs a RAG chain that returns both the answer and the source documents.
    Output dictionary format: {"answer": str, "source_documents": List[Document]}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Parallel step to keep retrieved documents for returning to the user
    rag_chain_with_docs = (
        RunnableParallel(
            {
                "context": (lambda x: x["question"]) | retriever | format_docs,
                "source_documents": (lambda x: x["question"]) | retriever,
                "question": lambda x: x["question"],
            }
        )
        .assign(answer=(prompt | llm | StrOutputParser()))
    )

    return rag_chain_with_docs
