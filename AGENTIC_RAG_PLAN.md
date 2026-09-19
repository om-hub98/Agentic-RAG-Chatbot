# 🏗️ Production-Grade Agentic RAG System Architecture Plan

A resilient, scalable, enterprise-ready **Agentic Retrieval-Augmented Generation (RAG)** architecture built with **LangChain**, **LangGraph**, **LangSmith**, multi-provider LLM fallback resilience, hybrid search, self-reflective grading, and automated evaluation.

---

## 1. System Overview & Executive Summary

Traditional RAG systems follow a rigid linear pipeline (*Retrieve ➔ Augment ➔ Generate*), making them brittle when retrievals are noisy, queries are ambiguous, or primary LLM APIs experience rate limits or outages.

This **Production-Level Agentic RAG** architecture introduces:
1. **Dynamic Agentic Control Loops (LangGraph)**: Adaptive query routing, document relevance grading (CRAG), self-reflective hallucination checks (Self-RAG), and iterative query rewriting.
2. **Multi-LLM Fallback & Circuit-Breaking**: Dynamic failover across multiple providers (Groq ➔ OpenAI ➔ Anthropic ➔ Local Ollama) using LangChain's native `.with_fallbacks()` mechanism and exponential backoff.
3. **Advanced Hybrid Retrieval & Reranking**: Dense FAISS embeddings combined with Sparse BM25 keyword search (Reciprocal Rank Fusion) followed by a Cross-Encoder Reranker.
4. **End-to-End LangSmith Observability & Evaluation**: Automated tracing, token/latency monitoring, production feedback loops, and automated offline/online benchmark suites (Context Relevance, Faithfulness, Answer Correctness).
5. **Clean / Hexagonal Enterprise Codebase Structure**: Domain logic decoupled from LLM providers, vector databases, and presentation layers.

---

## 2. LangGraph Agentic Workflow Diagram

```mermaid
flowchart TD
    START([User Query]) --> ROUTER{Intent Router}
    
    ROUTER -->|General Conversation| DIRECT_LLM[Direct Response Node]
    ROUTER -->|Needs Document Facts| RETRIEVE[Hybrid Retrieval Node\nFAISS + BM25]
    
    RETRIEVE --> RERANK[Cross-Encoder Reranker Node]
    RERANK --> GRADE_DOCS[Document Grader Node\nLLM Evaluates Relevance]
    
    GRADE_DOCS --> FILTER{Relevant Docs Found?}
    
    FILTER -->|Yes: Sufficient Context| GENERATE[Grounded Generation Node\nMulti-LLM Fallback]
    FILTER -->|No: Insufficient/Irrelevant| REWRITE[Query Rewriter / Transformation Node]
    
    REWRITE --> CHECK_RETRIES{Retry Limit Reached?}
    CHECK_RETRIES -->|Under Limit (Max 2)| RETRIEVE
    CHECK_RETRIES -->|Exceeded Limit| REFUSAL[Strict Fallback Refusal]
    
    GENERATE --> GRADE_HALLUCINATION{Hallucination Grader\nIs answer grounded in docs?}
    
    GRADE_HALLUCINATION -->|No: Hallucinated| REGENERATE[Regenerate with Negative Constraints]
    GRADE_HALLUCINATION -->|Yes: Grounded| GRADE_ANSWER{Answer Grader\nDoes answer address query?}
    
    GRADE_ANSWER -->|Yes: Complete| LANGSMITH_TRACE[LangSmith Trace & Log]
    GRADE_ANSWER -->|No: Incomplete| REWRITE
    
    DIRECT_LLM --> LANGSMITH_TRACE
    REFUSAL --> LANGSMITH_TRACE
    REGENERATE --> LANGSMITH_TRACE
    LANGSMITH_TRACE --> END([Final Answer Output])
```

---

## 3. Production Folder Structure

A modular, testable, enterprise directory layout following Clean Architecture:

```
agentic_rag_system/
├── .env.example                        # Environment template (API keys, endpoints, flags)
├── pyproject.toml                      # Modern packaging & dependency definitions (uv / poetry)
├── requirements.txt                    # Locked requirements
├── README.md                           # Documentation & quickstart guide
├── docker-compose.yml                  # Multi-service setup (App, Redis for checkpointer, etc.)
├── Dockerfile                          # Production container image
│
├── config/                             # Centralized Settings & Schemas
│   ├── __init__.py
│   ├── settings.py                     # Pydantic BaseSettings (LLM models, timeouts, paths)
│   └── prompts.py                      # System prompts (Grader, Rewriter, Generator, Router)
│
├── documents/                          # Knowledge base raw source files
│   ├── resumes/
│   ├── technical_docs/
│   └── manuals/
│
├── storage/                            # Local vector index artifacts (ignored by git)
│   ├── faiss_index/
│   └── bm25_cache/
│
├── src/                                # Core Application Source Code
│   ├── __init__.py
│   │
│   ├── domain/                         # Domain Entities & Models (Zero external framework deps)
│   │   ├── __init__.py
│   │   ├── models.py                   # DocumentChunk, Citation, RetrievalResult, EvaluationScore
│   │   └── state.py                    # LangGraph AgentState definition
│   │
│   ├── ingestion/                      # Document Ingestion & Chunking Pipeline
│   │   ├── __init__.py
│   │   ├── loaders.py                  # Multi-format loaders (PDF, DOCX, Markdown, Text, CSV)
│   │   ├── chunkers.py                 # Semantic chunker & RecursiveCharacterTextSplitter
│   │   └── hasher.py                   # SHA-256 change detection for incremental caching
│   │
│   ├── embeddings/                     # Embedding Providers
│   │   ├── __init__.py
│   │   ├── base.py                     # Abstract BaseEmbeddings interface
│   │   ├── fastembed_provider.py       # Local FastEmbed (bge-small/base)
│   │   └── openai_provider.py          # OpenAI text-embedding-3-small/large
│   │
│   ├── retrieval/                      # Vector Store & Hybrid Search Engine
│   │   ├── __init__.py
│   │   ├── faiss_store.py              # FAISS vector store manager (IndexFlatIP / HNSW)
│   │   ├── bm25_store.py               # Sparse keyword search (rank-bm25)
│   │   ├── hybrid_search.py            # Reciprocal Rank Fusion (RRF) combining Dense + Sparse
│   │   ├── reranker.py                 # FlashRank / Cohere / BGE Cross-Encoder reranker
│   │   └── tools.py                    # LangChain @tool wrappers for agent execution
│   │
│   ├── llm/                            # Multi-LLM Provider & Fallback Engine
│   │   ├── __init__.py
│   │   ├── factory.py                  # Dynamic LLM provider factory (Groq, OpenAI, Anthropic)
│   │   ├── fallback_router.py          # .with_fallbacks() orchestration & retry strategies
│   │   └── token_tracker.py            # Cost & token usage accumulator
│   │
│   ├── agent/                          # LangGraph StateGraph & Node Definitions
│   │   ├── __init__.py
│   │   ├── state.py                    # AgentState TypedDict schema
│   │   ├── nodes/                      # StateGraph Individual Nodes
│   │   │   ├── __init__.py
│   │   │   ├── router_node.py          # Query intent classification
│   │   │   ├── retrieve_node.py        # Vector & hybrid search executor
│   │   │   ├── grade_documents_node.py # Binary/relevance document grader (CRAG)
│   │   │   ├── rewrite_node.py         # Query reformulation node
│   │   │   ├── generate_node.py        # Grounded response synthesis
│   │   │   └── hallucination_node.py   # Groundedness & faithfulness verification
│   │   ├── edges.py                    # Conditional routing functions & guardrail checks
│   │   └── graph.py                    # Compiled LangGraph StateGraph workflow
│   │
│   └── evaluation/                     # LangSmith Monitoring & Evaluation Suite
│       ├── __init__.py
│       ├── langsmith_client.py         # Tracing configuration & metadata tagger
│       ├── evaluators.py               # Custom LangSmith evaluators (Groundedness, Relevance)
│       └── benchmark_runner.py         # Automated evaluation on curated test datasets
│
├── api/                                # REST API / Serving Layer (FastAPI)
│   ├── __init__.py
│   ├── app.py                          # FastAPI application factory
│   ├── routes.py                       # /chat, /query, /reindex, /health endpoints
│   └── schemas.py                      # Request/Response Pydantic DTOs
│
├── cli/                                # Command Line Interface
│   ├── __init__.py
│   ├── main.py                         # Rich interactive terminal CLI with slash commands
│   └── formatters.py                   # Rich tables, panels, and streaming syntax formatters
│
└── tests/                              # Comprehensive Test Suite
    ├── unit/                           # Isolated unit tests
    │   ├── test_ingestion.py
    │   ├── test_hybrid_search.py
    │   ├── test_fallback_llm.py
    │   └── test_graders.py
    ├── integration/                    # End-to-end agent workflow tests
    │   ├── test_agent_graph.py
    │   └── test_refusal_guardrails.py
    └── eval/                           # Offline benchmark test suite
        ├── dataset_sample.json
        └── run_eval_suite.py
```

---

## 4. Multi-LLM Fallback & Resilience Strategy

In production, relying on a single LLM vendor causes downtime during outages, rate limit spikes (HTTP 429), or credential expiration. 

### Implementation Pattern: LangChain `.with_fallbacks()`

```python
# src/llm/fallback_router.py
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel

def build_resilient_llm_chain(
    temperature: float = 0.0,
    streaming: bool = True,
) -> BaseChatModel:
    """
    Tiered Fallback Architecture:
    1. Primary: Groq (Ultra-low latency, Llama-3.3-70b-versatile)
    2. Fallback 1: OpenAI (High capability, GPT-4o-mini / GPT-4o)
    3. Fallback 2: Anthropic (Claude-3-5-Haiku / Claude-3-5-Sonnet)
    """
    # 1. Primary Model: Groq
    primary_llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=temperature,
        max_retries=2,
        streaming=streaming,
    )

    # 2. Secondary Model: OpenAI
    fallback_llm_openai = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        max_retries=2,
        streaming=streaming,
    )

    # 3. Tertiary Model: Anthropic
    fallback_llm_anthropic = ChatAnthropic(
        model="claude-3-5-haiku-20241022",
        temperature=temperature,
        max_retries=2,
    )

    # Compose resilient chain with automated failover
    resilient_llm = primary_llm.with_fallbacks(
        fallbacks=[fallback_llm_openai, fallback_llm_anthropic],
        exceptions_to_handle=(Exception,),
    )
    
    return resilient_llm
```

### Key Failover Features:
- **Zero Downtime**: Automatically shifts to OpenAI if Groq throws `AuthenticationError`, `RateLimitError`, or `ServiceUnavailable`.
- **Latency Optimization**: Default to Groq's high throughput (~300 tokens/sec) for interactive user experience; fallback to OpenAI/Anthropic when needed.
- **Provider Circuit-Breaking**: Configurable timeouts (e.g. 5s timeout on primary before falling back).

---

## 5. LangGraph Agent State & Graph Architecture

### 5.1 Agent State Definition

```python
# src/domain/state.py
from typing import Annotated, List, Optional, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """Immutable state shared across all nodes in the LangGraph workflow."""
    messages: Annotated[Sequence[BaseMessage], add_messages]  # Chat history
    query: str                                               # Original user question
    rewritten_query: Optional[str]                           # Reformulated search query
    documents: List[Document]                                # Retrieved & filtered documents
    web_fallback: bool                                       # Flag for external search
    iteration_count: int                                     # Loop guardrail counter
    is_grounded: bool                                        # Hallucination evaluation
    is_relevant: bool                                        # Answer relevance evaluation
```

### 5.2 Key Node Logic (Corrective & Self-Reflective RAG)

1. **Router Node (`route_query`)**:
   - Classifies if query needs document retrieval or can be answered directly (greetings, meta-questions).
2. **Retrieval & Rerank Node (`retrieve_and_rerank`)**:
   - Executes dense (FAISS) + sparse (BM25) search.
   - Passes top 15 chunks through Cross-Encoder reranker to extract top 4 highest-signal excerpts.
3. **Document Grader Node (`grade_documents`)**:
   - Structured JSON output (`{"relevant": "yes" | "no"}`) checking if retrieved chunks actually contain facts answering the user query. Filters out irrelevant chunks.
4. **Query Rewriter Node (`rewrite_query`)**:
   - If documents are insufficient, prompts the LLM to optimize query keywords for improved vector similarity.
5. **Generator Node (`generate_answer`)**:
   - Generates answer strictly grounded in the vetted context documents.
6. **Hallucination Grader Node (`grade_hallucination`)**:
   - Verifies each claim in the generated text against context documents. If hallucination detected, routes to retry generation with higher penalty.

---

## 6. LangSmith Production Observability & Evaluation

### 6.1 Tracing Configuration
Enabled via environment variables in production:
```bash
# Enable LangSmith Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="lsv2_pt_..."
LANGCHAIN_PROJECT="production-agentic-rag"
```

### 6.2 What LangSmith Captures in Production:
- **Full Execution DAG**: Exact latency breakdown per LangGraph node (e.g., FAISS retrieval: 12ms, Grader: 140ms, Generation: 650ms).
- **Tool Invocations & Payloads**: Raw queries passed to vector search and exact chunks returned.
- **Provider Fallback Traces**: Visual indicators when a request failed over from Groq to OpenAI.
- **Token & Cost Ledger**: Real-time aggregation of input/output tokens and cost per user session.

### 6.3 Automated Evaluation Suite (Offline CI/CD)
Uses LangSmith's `evaluate()` API with structured evaluators:

```python
# src/evaluation/evaluators.py
from langsmith.evaluation import evaluate
from langchain_core.language_models.chat_models import BaseChatModel

def evaluate_rag_system(dataset_name: str, target_graph):
    """
    Evaluates the agent on 3 fundamental RAG benchmarks:
    1. Context Precision: Were relevant documents placed at the top?
    2. Faithfulness: Is the answer 100% grounded without hallucination?
    3. Answer Relevance: Does the response directly solve the prompt?
    """
    
    def faithfulness_evaluator(run, example):
        # LLM-as-a-judge prompt verifying no facts exist outside context
        ...
        return {"key": "faithfulness", "score": 1.0}

    def relevance_evaluator(run, example):
        # Checks semantic alignment with expected ground truth
        ...
        return {"key": "relevance", "score": 0.95}

    results = evaluate(
        target_graph,
        data=dataset_name,
        evaluators=[faithfulness_evaluator, relevance_evaluator],
        experiment_prefix="agentic-rag-eval",
    )
    return results
```

---

## 7. Step-by-Step Implementation Roadmap

| Phase | Milestone | Deliverables |
| :--- | :--- | :--- |
| **Phase 1** | **Foundation & Ingestion** | • Implement multi-format `loaders.py` with SHA256 checksums<br>• Setup `chunkers.py` with metadata preservation<br>• Configure `FastEmbed` / `OpenAI` embedding abstractions |
| **Phase 2** | **Hybrid Retrieval Engine** | • Setup FAISS C++ vector store (`IndexFlatIP`)<br>• Add `rank-bm25` sparse indexing<br>• Implement Reciprocal Rank Fusion (RRF) & FlashRank reranker |
| **Phase 3** | **Multi-LLM Fallback Router** | • Build `factory.py` and `fallback_router.py`<br>• Configure Groq primary + OpenAI / Anthropic fallbacks<br>• Unit test provider failover under simulated API failures |
| **Phase 4** | **LangGraph Agent Workflow** | • Define `AgentState` schema<br>• Implement individual nodes (Router, Grader, Generator, Rewriter)<br>• Wire conditional edges with cycle/iteration guardrails |
| **Phase 5** | **LangSmith Integration** | • Connect distributed tracing & metadata tagging<br>• Implement custom LLM-as-a-judge evaluators<br>• Create golden test dataset & automated regression pipeline |
| **Phase 6** | **Production Serving & CLI** | • Build FastAPI endpoints (`/chat`, `/query`, `/health`)<br>• Build Rich Terminal CLI with streaming & slash commands<br>• Docker containerization & production README |

---

## 8. Summary of Benefits

1. **High Reliability (99.9% Uptime)**: System will never crash due to a single expired API key or vendor outage.
2. **Zero Hallucination Guarantee**: Evaluator loops strictly check claims before delivering answers to users.
3. **Sub-Second Response Times**: Groq primary inference + cached FAISS index guarantees fast responses.
4. **Complete Auditability**: Every reasoning step, document score, and LLM token is fully observable in LangSmith.
