from fastapi import FastAPI, HTTPException
from loguru import logger
from contextlib import asynccontextmanager

from retrieval.retriever import HybridRetriever
from retrieval.reranker import Reranker
from generation.generator import Generator
from api.schemas import QueryRequest, QueryResponse
from config import settings


# Initialise heavy components once at startup, not per request
retriever: HybridRetriever | None = None
reranker: Reranker | None = None
generator: Generator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, reranker, generator
    logger.info("Loading RAG components...")
    retriever = HybridRetriever()
    reranker = Reranker()
    generator = Generator()
    logger.success("All components ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="arXiv RAG API",
    description="Hybrid retrieval-augmented generation over arXiv AI papers",
    version="0.3.0",
    lifespan=lifespan,
)
@app.get("/health")
def health():
    return {"status": "ok", "collection": settings.QDRANT_COLLECTION}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not retriever or not reranker or not generator:
        raise HTTPException(status_code=503, detail="RAG components not ready")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info(f"Query: {request.query!r}")

    # Step 1: Hybrid retrieval — fetch top-K candidates
    candidates = retriever.retrieve(request.query, top_k=settings.RETRIEVAL_TOP_K)

    # Step 2: Rerank — narrow to best N chunks
    reranked = reranker.rerank(request.query, candidates)

    # Step 3: Generate answer
    result = generator.generate(request.query, reranked)

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        contexts=[c["content"] for c in reranked],
        retrieved_chunks=len(reranked),
        usage=result["usage"],
    )
@app.get("/health")
def health():
    provider_ok = retriever is not None
    return {
        "status": "ok",
        "collection": settings.QDRANT_COLLECTION,
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.LLM_MODEL,
        "provider_healthy": retriever is not None,
    }