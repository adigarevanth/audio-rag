"""Evaluation tests: measure recall@k, MRR, and precision@k against golden queries."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime

import click
import pytest

from src.config import EVAL_DIR
from src.db import get_connection
from src.search import (
    SearchResult,
    hybrid_search,
    keyword_only_search,
    semantic_only_search,
)

logger = logging.getLogger(__name__)

GOLDEN_QUERIES_PATH = EVAL_DIR / "golden_queries.json"


def load_golden_queries() -> list[dict]:
    """Load the labeled query-result pairs for evaluation."""
    if not GOLDEN_QUERIES_PATH.exists():
        pytest.skip(f"Golden queries file not found: {GOLDEN_QUERIES_PATH}")
    data = json.loads(GOLDEN_QUERIES_PATH.read_text(encoding="utf-8"))
    return data["queries"]


def _is_match(result: SearchResult, expected: dict, time_tolerance: float = 5.0) -> bool:
    """
    Check if a search result matches an expected result.

    A match requires:
      - Same filename
      - Overlapping or close timestamps (within tolerance)
    """
    if result.filename != expected["file"]:
        return False

    # Check time overlap or proximity
    exp_start = expected["timestamp_range"][0]
    exp_end = expected["timestamp_range"][1]

    # Is there overlap?
    overlap = max(0, min(result.end_time, exp_end) - max(result.start_time, exp_start))
    if overlap > 0:
        return True

    # Are they close enough?
    dist = min(abs(result.start_time - exp_start), abs(result.end_time - exp_end))
    return dist <= time_tolerance


def recall_at_k(
    results: list[SearchResult],
    expected_results: list[dict],
    k: int,
) -> float:
    """
    Compute recall@k: fraction of expected results found in the top-k results.

    recall@k = |{relevant found in top-k}| / |{all relevant}|
    """
    if not expected_results:
        return 1.0

    top_k = results[:k]
    found = 0
    for expected in expected_results:
        if any(_is_match(r, expected) for r in top_k):
            found += 1

    return found / len(expected_results)


def precision_at_k(
    results: list[SearchResult],
    expected_results: list[dict],
    k: int,
) -> float:
    """
    Compute precision@k: fraction of top-k results that are relevant.

    precision@k = |{relevant in top-k}| / k
    """
    top_k = results[:k]
    if not top_k:
        return 0.0

    relevant_count = 0
    for result in top_k:
        if any(_is_match(result, exp) for exp in expected_results):
            relevant_count += 1

    return relevant_count / len(top_k)


def mean_reciprocal_rank(
    results: list[SearchResult],
    expected_results: list[dict],
) -> float:
    """
    Compute MRR: 1 / rank of the first relevant result.

    Returns 0 if no relevant result is found.
    """
    for rank, result in enumerate(results, 1):
        if any(_is_match(result, exp) for exp in expected_results):
            return 1.0 / rank
    return 0.0


def _evaluate_search_fn(search_fn, queries, conn, k_values=(5, 10)):
    """
    Run evaluation for a given search function across all golden queries.

    Returns:
        Tuple of (aggregated_metrics, per_query_details).
        per_query_details is a list of dicts with query text and its individual scores.
    """
    metrics = {f"recall@{k}": [] for k in k_values}
    metrics["mrr"] = []
    metrics.update({f"precision@{k}": [] for k in k_values})

    per_query: list[dict] = []

    for q in queries:
        results = search_fn(conn, q["query"], k=max(k_values))

        row = {"query": q["query"]}
        for k in k_values:
            r_at_k = recall_at_k(results, q["expected_results"], k)
            p_at_k = precision_at_k(results, q["expected_results"], k)
            metrics[f"recall@{k}"].append(r_at_k)
            metrics[f"precision@{k}"].append(p_at_k)
            row[f"recall@{k}"] = r_at_k
            row[f"precision@{k}"] = p_at_k

        mrr_val = mean_reciprocal_rank(results, q["expected_results"])
        metrics["mrr"].append(mrr_val)
        row["mrr"] = mrr_val

        # Capture top result info
        if results:
            row["top_result_file"] = results[0].filename
            row["top_result_speaker"] = results[0].speaker
            row["top_result_timestamp"] = results[0].timestamp
            row["top_result_match_type"] = results[0].match_type
            row["top_result_text"] = results[0].text[:150]
        else:
            row["top_result_file"] = ""
            row["top_result_speaker"] = ""
            row["top_result_timestamp"] = ""
            row["top_result_match_type"] = ""
            row["top_result_text"] = "NO RESULTS"

        per_query.append(row)

    # Average across queries
    aggregated = {name: sum(values) / len(values) for name, values in metrics.items()}
    return aggregated, per_query


# --- pytest tests ---

class TestHybridSearch:
    """Tests that verify hybrid search quality against golden queries."""

    def test_recall_at_5(self):
        queries = load_golden_queries()
        with get_connection() as conn:
            metrics, _ = _evaluate_search_fn(hybrid_search, queries, conn, k_values=(5,))
        assert metrics["recall@5"] >= 0.70, f"Recall@5 = {metrics['recall@5']:.2f} (target >= 0.70)"

    def test_recall_at_10(self):
        queries = load_golden_queries()
        with get_connection() as conn:
            metrics, _ = _evaluate_search_fn(hybrid_search, queries, conn, k_values=(10,))
        assert metrics["recall@10"] >= 0.85, f"Recall@10 = {metrics['recall@10']:.2f} (target >= 0.85)"

    def test_mrr(self):
        queries = load_golden_queries()
        with get_connection() as conn:
            metrics, _ = _evaluate_search_fn(hybrid_search, queries, conn, k_values=(10,))
        assert metrics["mrr"] >= 0.60, f"MRR = {metrics['mrr']:.2f} (target >= 0.60)"

    def test_hybrid_beats_individual(self):
        """Hybrid search recall@10 should be >= max of keyword-only and semantic-only."""
        queries = load_golden_queries()
        with get_connection() as conn:
            hybrid_metrics, _ = _evaluate_search_fn(hybrid_search, queries, conn, k_values=(10,))
            keyword_metrics, _ = _evaluate_search_fn(keyword_only_search, queries, conn, k_values=(10,))
            semantic_metrics, _ = _evaluate_search_fn(semantic_only_search, queries, conn, k_values=(10,))

        hybrid_recall = hybrid_metrics["recall@10"]
        best_individual = max(keyword_metrics["recall@10"], semantic_metrics["recall@10"])

        assert hybrid_recall >= best_individual, (
            f"Hybrid recall@10 ({hybrid_recall:.2f}) should be >= "
            f"best individual ({best_individual:.2f})"
        )


# --- CLI evaluation runner ---

def _write_csv(
    hybrid_per_query: list[dict],
    kw_per_query: list[dict],
    sem_per_query: list[dict],
    hybrid_metrics: dict,
    kw_metrics: dict,
    sem_metrics: dict,
) -> str:
    """Write evaluation results to a CSV file and return the path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = EVAL_DIR / f"eval_results_{timestamp}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # --- Per-query results ---
        writer.writerow(["PER-QUERY RESULTS"])
        writer.writerow([])

        # Hybrid per-query
        writer.writerow(["HYBRID SEARCH"])
        headers = [
            "Query", "Recall@5", "Recall@10", "Precision@5", "Precision@10",
            "MRR", "Top Result File", "Top Result Speaker",
            "Top Result Timestamp", "Match Type", "Top Result Text",
        ]
        writer.writerow(headers)
        for row in hybrid_per_query:
            writer.writerow([
                row["query"],
                f"{row.get('recall@5', ''):.3f}" if isinstance(row.get("recall@5"), float) else "",
                f"{row.get('recall@10', ''):.3f}" if isinstance(row.get("recall@10"), float) else "",
                f"{row.get('precision@5', ''):.3f}" if isinstance(row.get("precision@5"), float) else "",
                f"{row.get('precision@10', ''):.3f}" if isinstance(row.get("precision@10"), float) else "",
                f"{row['mrr']:.3f}",
                row["top_result_file"],
                row["top_result_speaker"],
                row["top_result_timestamp"],
                row["top_result_match_type"],
                row["top_result_text"],
            ])
        writer.writerow([])

        # Keyword per-query
        writer.writerow(["KEYWORD ONLY SEARCH"])
        writer.writerow(headers)
        for row in kw_per_query:
            writer.writerow([
                row["query"],
                f"{row.get('recall@5', ''):.3f}" if isinstance(row.get("recall@5"), float) else "",
                f"{row.get('recall@10', ''):.3f}" if isinstance(row.get("recall@10"), float) else "",
                f"{row.get('precision@5', ''):.3f}" if isinstance(row.get("precision@5"), float) else "",
                f"{row.get('precision@10', ''):.3f}" if isinstance(row.get("precision@10"), float) else "",
                f"{row['mrr']:.3f}",
                row["top_result_file"],
                row["top_result_speaker"],
                row["top_result_timestamp"],
                row["top_result_match_type"],
                row["top_result_text"],
            ])
        writer.writerow([])

        # Semantic per-query
        writer.writerow(["SEMANTIC ONLY SEARCH"])
        writer.writerow(headers)
        for row in sem_per_query:
            writer.writerow([
                row["query"],
                f"{row.get('recall@5', ''):.3f}" if isinstance(row.get("recall@5"), float) else "",
                f"{row.get('recall@10', ''):.3f}" if isinstance(row.get("recall@10"), float) else "",
                f"{row.get('precision@5', ''):.3f}" if isinstance(row.get("precision@5"), float) else "",
                f"{row.get('precision@10', ''):.3f}" if isinstance(row.get("precision@10"), float) else "",
                f"{row['mrr']:.3f}",
                row["top_result_file"],
                row["top_result_speaker"],
                row["top_result_timestamp"],
                row["top_result_match_type"],
                row["top_result_text"],
            ])
        writer.writerow([])

        # --- Summary ---
        writer.writerow(["AGGREGATE METRICS"])
        writer.writerow(["Strategy", "Recall@5", "Recall@10", "Precision@5", "Precision@10", "MRR"])
        for name, metrics in [("Hybrid", hybrid_metrics), ("Keyword Only", kw_metrics), ("Semantic Only", sem_metrics)]:
            writer.writerow([
                name,
                f"{metrics.get('recall@5', ''):.3f}" if isinstance(metrics.get("recall@5"), float) else "",
                f"{metrics.get('recall@10', ''):.3f}" if isinstance(metrics.get("recall@10"), float) else "",
                f"{metrics.get('precision@5', ''):.3f}" if isinstance(metrics.get("precision@5"), float) else "",
                f"{metrics.get('precision@10', ''):.3f}" if isinstance(metrics.get("precision@10"), float) else "",
                f"{metrics['mrr']:.3f}",
            ])

    return str(csv_path)


def run_evaluation() -> None:
    """Run full evaluation, print results, and save to CSV."""
    queries_data = json.loads(GOLDEN_QUERIES_PATH.read_text(encoding="utf-8"))
    queries = queries_data["queries"]

    click.echo(f"\nEvaluating against {len(queries)} golden queries...\n")

    with get_connection() as conn:
        click.echo("--- Hybrid Search ---")
        hybrid_metrics, hybrid_pq = _evaluate_search_fn(hybrid_search, queries, conn)
        for name, value in hybrid_metrics.items():
            click.echo(f"  {name}: {value:.3f}")

        click.echo("\n--- Keyword Only ---")
        kw_metrics, kw_pq = _evaluate_search_fn(keyword_only_search, queries, conn)
        for name, value in kw_metrics.items():
            click.echo(f"  {name}: {value:.3f}")

        click.echo("\n--- Semantic Only ---")
        sem_metrics, sem_pq = _evaluate_search_fn(semantic_only_search, queries, conn)
        for name, value in sem_metrics.items():
            click.echo(f"  {name}: {value:.3f}")

    click.echo("\n--- Comparison ---")
    click.echo(f"  Hybrid recall@10:   {hybrid_metrics['recall@10']:.3f}")
    click.echo(f"  Keyword recall@10:  {kw_metrics['recall@10']:.3f}")
    click.echo(f"  Semantic recall@10: {sem_metrics['recall@10']:.3f}")
    best = max(kw_metrics['recall@10'], sem_metrics['recall@10'])
    if hybrid_metrics['recall@10'] >= best:
        click.echo("  ✓ Hybrid beats or ties individual strategies")
    else:
        click.echo("  ✗ Hybrid underperforms individual strategies")

    # Write CSV
    csv_path = _write_csv(hybrid_pq, kw_pq, sem_pq, hybrid_metrics, kw_metrics, sem_metrics)
    click.echo(f"\n📄 Results saved to: {csv_path}")
