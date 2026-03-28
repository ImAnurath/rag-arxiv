import json
import uuid
import requests
from pathlib import Path
from loguru import logger
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings

from evaluation.eval_store import save_run
from config import settings

DATASET_PATH = Path("evaluation/golden_dataset.json")
API_URL = "http://localhost:8000"


class Evaluator:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.golden = self._load_golden()

        # RAGAS uses an LLM for faithfulness + answer relevancy
        # and an embedding model for context metrics
        self.llm = ChatAnthropic( # TODO: need to change this whole thing at some point to support multiple LLMs
            model="claude-haiku-4-5-20251001",   # cheapest model — eval only
            api_key=settings.LLM_API_KEY,
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )

        self.metrics = [
            Faithfulness(llm=self.llm),
            AnswerRelevancy(llm=self.llm, embeddings=self.embeddings),
            ContextRecall(llm=self.llm),
            ContextPrecision(llm=self.llm),
        ]

    def _load_golden(self) -> list[dict]:
        return json.loads(DATASET_PATH.read_text())

    def _query_rag(self, question: str) -> dict:
        response = requests.post(
            f"{API_URL}/query",
            json={"query": question, "top_k": self.top_k},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def run(self, notes: str = "", limit: int | None = None) -> dict:
        golden = self.golden[:limit] if limit else self.golden
        run_id = str(uuid.uuid4())[:8]

        logger.info(f"Run {run_id} — evaluating {len(golden)} questions")

        samples = []
        ragas_samples = []

        for i, item in enumerate(golden):
            question = item["question"]
            ground_truth = item["ground_truth"]

            logger.debug(f"[{i+1}/{len(golden)}] {question[:60]}...")

            try:
                result = self._query_rag(question)
            except Exception as e:
                logger.warning(f"RAG query failed for question {i+1}: {e}")
                continue

            answer = result["answer"]
            contexts = result.get("contexts", [])

            samples.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "contexts": contexts,
            })

            ragas_samples.append(SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth,
            ))

        if not ragas_samples:
            logger.error("No samples collected — check the API is running")
            return {}

        logger.info(f"Running RAGAS evaluation on {len(ragas_samples)} samples...")
        dataset = EvaluationDataset(samples=ragas_samples)
        result = evaluate(dataset=dataset, metrics=self.metrics)

        scores = {
            "num_questions": len(ragas_samples),
            "faithfulness":        round(float(result["faithfulness"]), 4),
            "answer_relevancy":    round(float(result["answer_relevancy"]), 4),
            "context_recall":      round(float(result["context_recall"]), 4),
            "context_precision":   round(float(result["context_precision"]), 4),
        }

        # Attach per-sample scores back
        df = result.to_pandas()
        for i, row in df.iterrows():
            if i < len(samples):
                samples[i]["faithfulness"]      = row.get("faithfulness")
                samples[i]["answer_relevancy"]  = row.get("answer_relevancy")
                samples[i]["context_recall"]    = row.get("context_recall")
                samples[i]["context_precision"] = row.get("context_precision")

        save_run(run_id, scores, samples, notes=notes)
        return scores