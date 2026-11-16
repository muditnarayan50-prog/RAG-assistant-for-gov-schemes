"""
Evaluation script: Recall@5, MRR, abstention rate, latency on eval set.
"""
import json
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

import yaml
from retrieve import retrieve
from answer import AnswerGenerator


@dataclass
class EvalResult:
    """Single evaluation result."""
    question: str
    scheme: str
    slug: str
    retrieved: bool
    rank: int  # 1-based rank of correct scheme, 0 if not retrieved
    score: float
    latency_ms: float
    abstained: bool
    answer: str


def load_eval_candidates(path: str) -> List[Dict[str, Any]]:
    """Load evaluation candidates from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(
    eval_candidates: List[Dict[str, Any]],
    k: int = 5,
) -> List[EvalResult]:
    """
    Evaluate retrieval performance.
    
    For each FAQ question, check if the correct scheme appears in top-k results.
    """
    results = []
    
    for candidate in eval_candidates:
        question = candidate["question"]
        target_slug = candidate["slug"]
        target_scheme = candidate["scheme"]
        
        start = time.perf_counter()
        chunks, best_score = retrieve(question, k=k)
        latency_ms = (time.perf_counter() - start) * 1000
        
        # Find rank of target scheme
        rank = 0
        for i, chunk in enumerate(chunks):
            if chunk["slug"] == target_slug:
                rank = i + 1
                break
        
        retrieved = rank > 0 and rank <= k
        
        results.append(EvalResult(
            question=question,
            scheme=target_scheme,
            slug=target_slug,
            retrieved=retrieved,
            rank=rank,
            score=best_score,
            latency_ms=latency_ms,
            abstained=False,  # Will be set by answer evaluation
            answer="",
        ))
    
    return results


def evaluate_answer_generation(
    generator: AnswerGenerator,
    eval_candidates: List[Dict[str, Any]],
    language: str = "English",
) -> List[EvalResult]:
    """
    Evaluate full answer generation pipeline.
    """
    results = []
    
    for candidate in eval_candidates:
        question = candidate["question"]
        
        start = time.perf_counter()
        result = generator.generate(question, language)
        latency_ms = (time.perf_counter() - start) * 1000
        
        results.append(EvalResult(
            question=question,
            scheme=candidate["scheme"],
            slug=candidate["slug"],
            retrieved=not result.abstained,
            rank=0,  # Not applicable for answer generation
            score=result.best_score,
            latency_ms=latency_ms,
            abstained=result.abstained,
            answer=result.answer,
        ))
    
    return results


def compute_metrics(results: List[EvalResult], k: int = 5) -> Dict[str, Any]:
    """Compute evaluation metrics."""
    total = len(results)
    if total == 0:
        return {}
    
    # Recall@k
    retrieved_at_k = sum(1 for r in results if r.retrieved and r.rank <= k)
    recall_at_k = retrieved_at_k / total
    
    # MRR (Mean Reciprocal Rank)
    reciprocal_ranks = [1.0 / r.rank if r.rank > 0 else 0.0 for r in results]
    mrr = sum(reciprocal_ranks) / total
    
    # Abstention rate
    abstained = sum(1 for r in results if r.abstained)
    abstention_rate = abstained / total
    
    # Latency stats
    latencies = [r.latency_ms for r in results]
    
    # Score stats for retrieved vs not retrieved
    retrieved_scores = [r.score for r in results if r.retrieved]
    not_retrieved_scores = [r.score for r in results if not r.retrieved]
    
    return {
        "total_questions": total,
        f"recall@{k}": recall_at_k,
        "mrr": mrr,
        "abstention_rate": abstention_rate,
        "latency_ms": {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p95": sorted(latencies)[int(0.95 * total)] if total > 0 else 0,
            "min": min(latencies),
            "max": max(latencies),
        },
        "retrieved_score": {
            "mean": statistics.mean(retrieved_scores) if retrieved_scores else 0,
            "count": len(retrieved_scores),
        },
        "not_retrieved_score": {
            "mean": statistics.mean(not_retrieved_scores) if not_retrieved_scores else 0,
            "count": len(not_retrieved_scores),
        },
    }


def print_metrics(metrics: Dict[str, Any], prefix: str = "") -> None:
    """Print metrics in a readable format."""
    print(f"\n{prefix}Evaluation Metrics:")
    print(f"  Total Questions: {metrics['total_questions']}")
    recall_key = next(key for key in metrics if key.startswith("recall@"))
    print(f"  {recall_key}: {metrics[recall_key]:.3f} ({metrics[recall_key]*100:.1f}%)")
    print(f"  MRR: {metrics['mrr']:.3f}")
    print(f"  Abstention Rate: {metrics['abstention_rate']:.3f} ({metrics['abstention_rate']*100:.1f}%)")
    print(f"  Latency (ms): mean={metrics['latency_ms']['mean']:.1f}, "
          f"median={metrics['latency_ms']['median']:.1f}, "
          f"p95={metrics['latency_ms']['p95']:.1f}")
    print(f"  Retrieved Score: mean={metrics['retrieved_score']['mean']:.3f} "
          f"(n={metrics['retrieved_score']['count']})")
    print(f"  Not Retrieved Score: mean={metrics['not_retrieved_score']['mean']:.3f} "
          f"(n={metrics['not_retrieved_score']['count']})")


def save_results(results: List[EvalResult], output_path: str) -> None:
    """Save detailed results to JSON."""
    data = [
        {
            "question": r.question,
            "scheme": r.scheme,
            "slug": r.slug,
            "retrieved": r.retrieved,
            "rank": r.rank,
            "score": r.score,
            "latency_ms": r.latency_ms,
            "abstained": r.abstained,
            "answer": r.answer,
        }
        for r in results
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def main(config_path: str = "config.yaml"):
    """Run evaluation."""
    config = yaml.safe_load(open(config_path))
    
    eval_path = config["evaluation"]["eval_candidates_path"]
    print(f"Loading evaluation candidates from {eval_path}...")
    eval_candidates = load_eval_candidates(eval_path)
    print(f"Loaded {len(eval_candidates)} evaluation candidates")
    
    # Sample for faster iteration; pass --full to evaluate every candidate.
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate scheme retrieval and answers")
    parser.add_argument("--full", action="store_true", help="Evaluate all candidates")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip LLM answer generation")
    args = parser.parse_args()
    sample_size = len(eval_candidates) if args.full else 200
    if len(eval_candidates) > sample_size:
        import random
        random.seed(42)
        eval_candidates = random.sample(eval_candidates, sample_size)
        print(f"Sampled {sample_size} candidates for evaluation")
    
    # Evaluate retrieval
    print("\nEvaluating retrieval...")
    retrieval_results = evaluate_retrieval(eval_candidates, k=5)
    retrieval_metrics = compute_metrics(retrieval_results, k=5)
    print_metrics(retrieval_metrics, "Retrieval ")
    
    if args.retrieval_only:
        print("Retrieval-only mode: skipping answer generation (Ollama is not required).")
        return

    # Initialize answer generator
    print("\nInitializing answer generator...")
    generator = AnswerGenerator(config_path)
    
    # Evaluate answer generation
    print("Evaluating answer generation...")
    answer_results = evaluate_answer_generation(generator, eval_candidates)
    answer_metrics = compute_metrics(answer_results, k=5)
    print_metrics(answer_metrics, "Answer Generation ")
    
    # Save detailed results
    output_dir = Path("eval_results")
    output_dir.mkdir(exist_ok=True)
    
    save_results(retrieval_results, output_dir / "retrieval_results.json")
    save_results(answer_results, output_dir / "answer_results.json")
    
    # Save metrics summary
    summary = {
        "retrieval": retrieval_metrics,
        "answer_generation": answer_metrics,
        "config": {
            "top_k": config["retrieval"]["top_k"],
            "score_threshold": config["retrieval"]["score_threshold"],
            "provider": config.get("default_provider", "local"),
            "sample_size": len(eval_candidates),
        },
    }
    with open(output_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
