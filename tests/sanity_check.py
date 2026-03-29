"""
Full sanity check for every RAG component.
Run with: python tests/sanity_check.py
Or target specific sections: python tests/sanity_check.py --section retrieval
"""
import sys
import argparse
import json
import time
import requests
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="<level>{level: <8}</level> | {message}", colorize=True)

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_URL = "http://localhost:8000"
TEST_QUERY = "What is retrieval-augmented generation?"

PASS = "  ✓"
FAIL = "  ✗"
INFO = "  ·"

def section(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
def ok(msg: str):   print(f"\033[32m{PASS}\033[0m {msg}")
def err(msg: str):  print(f"\033[31m{FAIL}\033[0m {msg}")
def info(msg: str): print(f"\033[90m{INFO}\033[0m {msg}")

# ─────────────────────────────────────────────
# 1. INFRASTRUCTURE
# ─────────────────────────────────────────────
def check_infrastructure():
    section("1. Infrastructure")

    # Docker / Qdrant
    try:
        r = requests.get("http://localhost:6333/collections", timeout=5)
        r.raise_for_status()
        collections = [c["name"] for c in r.json()["result"]["collections"]]
        ok(f"Qdrant reachable — collections: {collections}")
    except Exception as e:
        err(f"Qdrant not reachable: {e}")

    # Postgres
    try:
        import psycopg2
        from config import settings
        conn = psycopg2.connect(settings.POSTGRES_URL)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM chunks")
        count = cur.fetchone()[0]
        conn.close()
        ok(f"PostgreSQL reachable — {count} chunk records")
    except Exception as e:
        err(f"PostgreSQL not reachable: {e}")

    # FastAPI
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        r.raise_for_status()
        data = r.json()
        ok(f"FastAPI running — provider: {data.get('llm_provider')} / model: {data.get('llm_model')}")
        info(f"Collection: {data.get('collection')}")
    except Exception as e:
        err(f"FastAPI not reachable at {BASE_URL}: {e}")
        print("\n  → Start it with: uvicorn api.main:app --reload --port 8000\n")

# ─────────────────────────────────────────────
# 2. EMBEDDING
# ─────────────────────────────────────────────
def check_embedding():
    section("2. Embedding model")
    try:
        from embedding.embedder import Embedder
        from ingestion.chunker import Chunk

        t0 = time.time()
        embedder = Embedder()
        load_time = time.time() - t0
        ok(f"Model loaded in {load_time:.2f}s — dimension: {embedder.dimension}")

        # Embed a test string
        class FakeChunk:
            content = TEST_QUERY

        t0 = time.time()
        results = embedder.embed_chunks([FakeChunk()])
        embed_time = time.time() - t0

        _, vector = results[0]
        ok(f"Embedding produced in {embed_time*1000:.1f}ms — vector length: {len(vector)}")
        info(f"First 5 values: {[round(v, 4) for v in vector[:5]]}")

        # Check normalisation
        magnitude = sum(v**2 for v in vector) ** 0.5
        if 0.99 < magnitude < 1.01:
            ok(f"Vector is normalised (magnitude ≈ 1.0)")
        else:
            err(f"Vector is NOT normalised (magnitude = {magnitude:.4f})")

    except Exception as e:
        err(f"Embedding check failed: {e}")

# ─────────────────────────────────────────────
# 3. VECTOR STORE (QDRANT)
# ─────────────────────────────────────────────
def check_qdrant():
    section("3. Qdrant vector store")
    try:
        from storage.qdrant_store import QdrantStore
        from config import settings

        store = QdrantStore()
        count = store.count()
        ok(f"Collection '{settings.QDRANT_COLLECTION}' has {count} vectors")

        if count == 0:
            err("Collection is empty — run python ingest.py first")
            return

        # Quick search
        from embedding.embedder import Embedder
        class FakeChunk:
            content = TEST_QUERY

        embedder = Embedder()
        _, vector = embedder.embed_chunks([FakeChunk()])[0]

        t0 = time.time()
        results = store.search(vector, top_k=3)
        search_time = time.time() - t0

        ok(f"Vector search returned {len(results)} results in {search_time*1000:.1f}ms")
        for i, r in enumerate(results[:3], 1):
            info(f"[{i}] score={r.get('score', 'n/a'):.4f} | {r.get('title', 'n/a')[:55]}")

    except Exception as e:
        err(f"Qdrant check failed: {e}")

# ─────────────────────────────────────────────
# 4. BM25 + HYBRID RETRIEVAL
# ─────────────────────────────────────────────
def check_retrieval():
    section("4. Hybrid retrieval (BM25 + vector + RRF)")
    try:
        from retrieval.retriever import HybridRetriever
        from config import settings

        t0 = time.time()
        retriever = HybridRetriever()
        init_time = time.time() - t0
        ok(f"HybridRetriever initialised in {init_time:.2f}s")
        info(f"BM25 index covers {len(retriever._bm25_chunks)} chunks")

        t0 = time.time()
        results = retriever.retrieve(TEST_QUERY, top_k=settings.RETRIEVAL_TOP_K)
        retrieve_time = time.time() - t0

        ok(f"Retrieved {len(results)} candidates in {retrieve_time*1000:.1f}ms")

        # Check RRF scores are present and descending
        scores = [r.get("rrf_score", 0) for r in results]
        if scores == sorted(scores, reverse=True):
            ok("RRF scores are correctly sorted (descending)")
        else:
            err("RRF scores are NOT sorted — check fusion logic")

        # Check diversity
        sources = [r.get("source") for r in results[:10]]
        unique_sources = len(set(sources))
        info(f"Top 10 results span {unique_sources} unique papers")

        for i, r in enumerate(results[:5], 1):
            info(f"[{i}] rrf={r.get('rrf_score', 0):.5f} | {r.get('title', 'n/a')[:50]}")

    except Exception as e:
        err(f"Retrieval check failed: {e}")
        import traceback; traceback.print_exc()

# ─────────────────────────────────────────────
# 5. RERANKER
# ─────────────────────────────────────────────
def check_reranker():
    section("5. Reranker (cross-encoder)")
    try:
        from retrieval.retriever import HybridRetriever
        from retrieval.reranker import Reranker
        from config import settings

        retriever = HybridRetriever()
        candidates = retriever.retrieve(TEST_QUERY, top_k=settings.RETRIEVAL_TOP_K)

        t0 = time.time()
        reranker = Reranker()
        reranked = reranker.rerank(TEST_QUERY, candidates)
        rerank_time = time.time() - t0

        ok(f"Reranked {len(candidates)} → {len(reranked)} chunks in {rerank_time*1000:.1f}ms")

        # Check scores are descending
        scores = [r.get("rerank_score", 0) for r in reranked]
        if scores == sorted(scores, reverse=True):
            ok("Rerank scores correctly sorted (descending)")
        else:
            err("Rerank scores NOT sorted")

        for i, r in enumerate(reranked, 1):
            info(f"[{i}] score={r.get('rerank_score', 0):.4f} | {r.get('title', 'n/a')[:50]}")

    except Exception as e:
        err(f"Reranker check failed: {e}")
        import traceback; traceback.print_exc()
        
# ─────────────────────────────────────────────
# 6. LLM PROVIDER
# ─────────────────────────────────────────────
def check_provider():
    section("6. LLM provider")
    from config import settings
    info(f"Provider: {settings.LLM_PROVIDER} | Model: {settings.LLM_MODEL}")

    try:
        from generation.providers import get_provider
        provider = get_provider()

        alive = provider.health_check()
        if alive:
            ok(f"Provider health check passed")
        else:
            err(f"Provider health check failed — is the service running?")
            if settings.LLM_PROVIDER == "ollama":
                print("    → Start Ollama: ollama serve")
                print(f"    → Pull model:   ollama pull {settings.LLM_MODEL}")
            return

        # Minimal completion test
        t0 = time.time()
        answer, usage = provider.complete(
            system="You are a helpful assistant. Be very brief.",
            user="Reply with exactly: 'RAG pipeline operational.'"
        )
        latency = time.time() - t0

        ok(f"Completion returned in {latency:.2f}s")
        info(f"Response: {answer.strip()[:80]}")
        info(f"Tokens — in: {usage.get('input_tokens', '?')} / out: {usage.get('output_tokens', '?')}")

    except Exception as e:
        err(f"Provider check failed: {e}")
        import traceback; traceback.print_exc()

# ─────────────────────────────────────────────
# 7. FULL END-TO-END via API
# ─────────────────────────────────────────────
def check_end_to_end():
    section("7. End-to-end query via API")
    try:
        t0 = time.time()
        r = requests.post(
            f"{BASE_URL}/query",
            json={"query": TEST_QUERY, "top_k": 3},
            timeout=240,
        )
        total_time = time.time() - t0

        if r.status_code != 200:
            err(f"API returned {r.status_code}: {r.text[:200]}")
            return

        data = r.json()
        ok(f"Full pipeline completed in {total_time:.2f}s")
        info(f"Chunks used: {data.get('retrieved_chunks')}")
        info(f"Sources: {len(data.get('sources', []))}")
        info(f"Tokens — in: {data.get('usage', {}).get('input_tokens', '?')} / "
             f"out: {data.get('usage', {}).get('output_tokens', '?')}")

        print(f"\n  Answer preview:")
        answer = data.get("answer", "")
        # Print first 400 chars, indented
        for line in answer[:400].split("\n"):
            print(f"    {line}")
        if len(answer) > 400:
            print(f"    ... ({len(answer)} chars total)")

        print(f"\n  Sources cited:")
        for s in data.get("sources", []):
            print(f"    · [{s.get('arxiv_id')}] {s.get('title', '')[:55]}")

    except requests.exceptions.Timeout:
        err("Request timed out after 120s — model may be too slow or not running")
    except Exception as e:
        err(f"End-to-end check failed: {e}")
        import traceback; traceback.print_exc()

# ─────────────────────────────────────────────
# 8. SECOND QUERY — different topic
# ─────────────────────────────────────────────
def check_second_query():
    section("8. Second query — different topic")
    query = "How does reinforcement learning from human feedback work?"
    try:
        t0 = time.time()
        r = requests.post(
            f"{BASE_URL}/query",
            json={"query": query, "top_k": 3},
            timeout=240,
        )
        total_time = time.time() - t0

        if r.status_code != 200:
            err(f"API returned {r.status_code}: {r.text[:200]}")
            return

        data = r.json()
        ok(f"Query completed in {total_time:.2f}s")

        # Check sources differ from first query
        sources = [s.get("arxiv_id") for s in data.get("sources", [])]
        info(f"Papers cited: {sources}")

        answer = data.get("answer", "")
        for line in answer[:300].split("\n"):
            print(f"    {line}")
        if len(answer) > 300:
            print(f"    ... ({len(answer)} chars total)")

    except Exception as e:
        err(f"Second query check failed: {e}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
SECTIONS = {
    "infra":      check_infrastructure,
    "embedding":  check_embedding,
    "qdrant":     check_qdrant,
    "retrieval":  check_retrieval,
    "reranker":   check_reranker,
    "provider":   check_provider,
    "e2e":        check_end_to_end,
    "e2e2":       check_second_query,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG system sanity checker")
    parser.add_argument(
        "--section",
        choices=list(SECTIONS.keys()),
        default=None,
        help="Run only a specific section"
    )
    args = parser.parse_args()

    print("\n\033[1m  RAG arXiv — Full System Sanity Check\033[0m")

    if args.section:
        SECTIONS[args.section]()
    else:
        for fn in SECTIONS.values():
            fn()

    print(f"\n{'='*55}\n")