"""Evaluation tests: measure recall@k, MRR, and precision@k against golden queries."""

from __future__ import annotations

import json
import logging

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
    """Run evaluation for a given search function across all golden queries."""
    metrics = {f"recall@{k}": [] for k in k_values}
    metrics["mrr"] = []
    metrics.update({f"precision@{k}": [] for k in k_values})

    for q in queries:
        results = search_fn(conn, q["query"], k=max(k_values))

        for k in k_values:
            metrics[f"recall@{k}"].append(recall_at_k(results, q["expected_results"], k))
            metrics[f"precision@{k}"].append(precision_at_k(results, q["expected_results"], k))
        metrics["mrr"].append(mean_reciprocal_rank(results, q["expected_results"]))

    # Average across queries
    return {name: sum(values) / len(values) for name, values in metrics.items()}


# --- pytest tests ---

class TestHybridSearch:
    """Tests that verify hybrid search quality against golden queries."""

    def test_recall_at_5(self):
        queries = load_golden_queries()
        with get_connection() as conn:
            metrics = _evaluate_search_fn(hybrid_search, queries, conn, k_values=(5,))
        assert metrics["recall@5"] >= 0.70, f"Recall@5 = {metrics['recall@5']:.2f} (target >= 0.70)"

    def test_recall_at_10(self):
        queries = load_golden_queries()
        with get_connection() as conn:
            metrics = _evaluate_search_fn(hybrid_search, queries, conn, k_values=(10,))
        assert metrics["recall@10"] >= 0.85, f"Recall@10 = {metrics['recall@10']:.2f} (target >= 0.85)"

    def test_mrr(self):
        queries = load_golden_queries()
        with get_connection() as conn:
            metrics = _evaluate_search_fn(hybrid_search, queries, conn, k_values=(10,))
        assert metrics["mrr"] >= 0.60, f"MRR = {metrics['mrr']:.2f} (target >= 0.60)"

    def test_hybrid_beats_individual(self):
        """Hybrid search recall@10 should be >= max of keyword-only and semantic-only."""
        queries = load_golden_queries()
        with get_connection() as conn:
            hybrid_metrics = _evaluate_search_fn(hybrid_search, queries, conn, k_values=(10,))
            keyword_metrics = _evaluate_search_fn(keyword_only_search, queries, conn, k_values=(10,))
            semantic_metrics = _evaluate_search_fn(semantic_only_search, queries, conn, k_values=(10,))

        hybrid_recall = hybrid_metrics["recall@10"]
        best_individual = max(keyword_metrics["recall@10"], semantic_metrics["recall@10"])

        assert hybrid_recall >= best_individual, (
            f"Hybrid recall@10 ({hybrid_recall:.2f}) should be >= "
            f"best individual ({best_individual:.2f})"
        )


# --- CLI evaluation runner ---

def run_evaluation() -> None:
    """Run full evaluation and print results (called from CLI)."""
    queries_data = json.loads(GOLDEN_QUERIES_PATH.read_text(encoding="utf-8"))
    queries = queries_data["queries"]

    click.echo(f"\nEvaluating against {len(queries)} golden queries...\n")

    with get_connection() as conn:
        click.echo("--- Hybrid Search ---")
        hybrid_metrics = _evaluate_search_fn(hybrid_search, queries, conn)
        for name, value in hybrid_metrics.items():
            click.echo(f"  {name}: {value:.3f}")

        click.echo("\n--- Keyword Only ---")
        kw_metrics = _evaluate_search_fn(keyword_only_search, queries, conn)
        for name, value in kw_metrics.items():
            click.echo(f"  {name}: {value:.3f}")

        click.echo("\n--- Semantic Only ---")
        sem_metrics = _evaluate_search_fn(semantic_only_search, queries, conn)
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
