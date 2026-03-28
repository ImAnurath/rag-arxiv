import argparse
from loguru import logger
from evaluation.evaluator import Evaluator
from evaluation.eval_store import get_runs

def print_scores(scores: dict) -> None:
    print("\n" + "="*48)
    print(f"  RAGAS Evaluation Results")
    print("="*48)
    print(f"  Questions evaluated : {scores['num_questions']}")
    print(f"  Faithfulness        : {scores['faithfulness']:.4f}")
    print(f"  Answer relevancy    : {scores['answer_relevancy']:.4f}")
    print(f"  Context recall      : {scores['context_recall']:.4f}")
    print(f"  Context precision   : {scores['context_precision']:.4f}")
    print("="*48 + "\n")

def print_history() -> None:
    runs = get_runs()
    if not runs:
        print("No evaluation runs found.")
        return
    print(f"\n{'Run ID':<10} {'Date':<22} {'Q':<4} {'Faith':>7} {'Relev':>7} {'Recall':>8} {'Prec':>7}")
    print("-" * 70)
    for r in runs:
        print(
            f"{r['run_id']:<10} {r['ran_at'][:19]:<22} {r['num_questions']:<4} "
            f"{r['faithfulness']:>7.4f} {r['answer_relevancy']:>7.4f} "
            f"{r['context_recall']:>8.4f} {r['context_precision']:>7.4f}"
        )
    print()

'''
How to use it

python eval.py --limit 5 --notes "first test run"  first 5 questions only

python eval.py --notes "120 papers ingested" # all questions

python eval.py --history # print past run history
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--limit",  type=int, default=None, help="Evaluate only first N questions")
    parser.add_argument("--notes",  type=str, default="",   help="Tag this run with a note")
    parser.add_argument("--history", action="store_true",   help="Print past run history")
    args = parser.parse_args()

    if args.history:
        print_history()
    else:
        evaluator = Evaluator()
        scores = evaluator.run(notes=args.notes, limit=args.limit)
        if scores:
            print_scores(scores)