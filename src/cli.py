"""CLI entry point for audio-rag: indexing and searching."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from src.config import AUDIO_DIR


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """Audio RAG: Hybrid search over audio transcripts."""
    setup_logging(verbose)


@cli.command()
@click.option("--force", is_flag=True, help="Re-index even if already processed.")
@click.option("--file", "filename", default=None, help="Index a single file instead of all.")
def index(force: bool, filename: str | None) -> None:
    """Transcribe, diarize, embed, and index audio files."""
    from src.index import index_all, index_audio_file, load_metadata

    if filename:
        audio_path = AUDIO_DIR / filename
        if not audio_path.exists():
            click.echo(f"Error: File not found: {audio_path}", err=True)
            sys.exit(1)
        metadata = load_metadata()
        file_meta = next(
            (f for f in metadata["files"] if f["filename"] == filename),
            {"filename": filename},
        )
        index_audio_file(audio_path, file_meta, force=force)
    else:
        index_all(force=force)

    click.echo("Indexing complete.")


@cli.command()
@click.argument("query")
@click.option("-k", "--top-k", default=10, help="Number of results to return.")
@click.option("--mode", type=click.Choice(["hybrid", "keyword", "semantic"]), default="hybrid",
              help="Search mode.")
def search(query: str, top_k: int, mode: str) -> None:
    """Search across indexed audio transcripts."""
    from src.db import get_connection
    from src.search import hybrid_search, keyword_only_search, semantic_only_search

    with get_connection() as conn:
        if mode == "hybrid":
            results = hybrid_search(conn, query, k=top_k)
        elif mode == "keyword":
            results = keyword_only_search(conn, query, k=top_k)
        else:
            results = semantic_only_search(conn, query, k=top_k)

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"\n{'='*80}")
    click.echo(f"  Search results for: \"{query}\"  (mode: {mode}, top {top_k})")
    click.echo(f"{'='*80}\n")

    for i, r in enumerate(results, 1):
        click.echo(f"  [{i}] {r.filename}  |  {r.speaker}  |  {r.timestamp}  |  {r.match_type}")
        click.echo(f"      Score: {r.score:.4f}")
        # Show first 200 chars of text
        text_preview = r.text[:200] + ("..." if len(r.text) > 200 else "")
        click.echo(f"      \"{text_preview}\"")
        click.echo()


@cli.command()
def evaluate() -> None:
    """Run recall@k evaluation against the golden query set."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tests.test_search import run_evaluation
    run_evaluation()


if __name__ == "__main__":
    cli()
