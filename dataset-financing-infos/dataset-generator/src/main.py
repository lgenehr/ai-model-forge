"""
Main CLI entry point using Typer.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .collectors import CollectorRegistry
from .config.settings import Settings, get_settings, load_topics_config
from .guardrails import BiasChecker, ContentFilter, LanguageDetector, PIIDetector
from .processors import DatasetFormatter, Deduplicator, QualityFilter, TextCleaner, TokenizerChecker
from .schemas.dataset import DatasetEntry, RawCollectedData
from .storage import CheckpointManager, OutputWriter, StateManager
from .utils.logger import get_logger, setup_logging
from .utils.metrics import MetricsCollector
from .utils.rate_limiter import get_rate_limiter

app = typer.Typer(
    name="dataset-generator",
    help="Multi-source dataset generator for LLM fine-tuning",
    add_completion=False,
)
console = Console()
logger = get_logger(__name__)


def get_available_sources() -> list[str]:
    """Get list of available data sources."""
    return ["news", "encyclopedia", "books", "academic", "legal", "social_media", "videos"]


def get_available_topics() -> list[str]:
    """Get list of available topics."""
    return [
        "financeiro",
        "tecnologia",
        "ciencias",
        "saude",
        "juridico",
        "humanidades",
        "cultura",
        "negocios",
        "educacao",
        "meio_ambiente",
    ]


@app.command()
def collect(
    sources: list[str] = typer.Option(
        ["all"],
        "--sources", "-s",
        help="Data sources to collect from (news, books, academic, etc.)",
    ),
    topics: list[str] = typer.Option(
        ["all"],
        "--topics", "-t",
        help="Topics to collect (financeiro, tecnologia, etc.)",
    ),
    output_dir: Path = typer.Option(
        Path("./outputs"),
        "--output-dir", "-o",
        help="Output directory",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Resume from checkpoint",
    ),
    max_items: int = typer.Option(
        100,
        "--max-items", "-m",
        help="Maximum items per source/topic combination",
    ),
    workers: int = typer.Option(
        4,
        "--workers", "-w",
        help="Number of parallel workers",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level", "-l",
        help="Logging level",
    ),
) -> None:
    """
    Collect data from specified sources.
    """
    setup_logging(log_level=log_level, json_format=False)

    settings = get_settings()
    settings.output_dir = output_dir
    settings.ensure_directories()

    # Resolve "all"
    if "all" in sources:
        sources = get_available_sources()
    if "all" in topics:
        topics = get_available_topics()

    console.print(
        Panel(
            f"[bold green]Dataset Collection[/bold green]\n\n"
            f"Sources: {', '.join(sources)}\n"
            f"Topics: {', '.join(topics)}\n"
            f"Max items per combination: {max_items}\n"
            f"Resume: {resume}",
            title="Configuration",
        )
    )

    # Run async collection
    asyncio.run(
        _run_collection(
            sources=sources,
            topics=topics,
            output_dir=output_dir,
            resume=resume,
            max_items=max_items,
            settings=settings,
        )
    )


async def _run_collection(
    sources: list[str],
    topics: list[str],
    output_dir: Path,
    resume: bool,
    max_items: int,
    settings: Settings,
) -> None:
    """Run async data collection."""
    checkpoint_manager = CheckpointManager(settings.checkpoint_dir)
    state_manager = StateManager(settings.checkpoint_dir)
    rate_limiter = get_rate_limiter()
    metrics = MetricsCollector()

    # Create or resume state
    if resume:
        state = state_manager.load_state()
        if state and state.status == "running":
            console.print("[yellow]Resuming previous collection...[/yellow]")
            pending = state_manager.get_pending_pairs()
        else:
            state = state_manager.create_run(sources, topics)
            pending = [(s, t) for s in sources for t in topics]
    else:
        state_manager.clear_state()
        state = state_manager.create_run(sources, topics)
        pending = [(s, t) for s in sources for t in topics]

    console.print(f"[blue]Pending collections: {len(pending)}[/blue]")

    # Create output writer
    raw_output_dir = output_dir / "raw"
    writer = OutputWriter(raw_output_dir, buffer_size=50)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        main_task = progress.add_task(
            "[cyan]Collecting...", total=len(pending)
        )

        for source_name, topic in pending:
            progress.update(
                main_task,
                description=f"[cyan]{source_name}/{topic}",
            )

            # Get collector
            collector = CollectorRegistry.create(
                source_name,
                settings=settings,
                checkpoint_manager=checkpoint_manager,
                rate_limiter=rate_limiter,
            )

            if collector is None:
                console.print(
                    f"[yellow]Collector not found: {source_name}[/yellow]"
                )
                progress.advance(main_task)
                continue

            run_metrics = metrics.create_run(source_name, topic)

            try:
                async with collector:
                    collected = 0
                    async for item in collector.collect(
                        topic=topic,
                        max_items=max_items,
                    ):
                        writer.write(item)
                        collected += 1
                        run_metrics.record_item(
                            quality_score=0.7,
                            token_count=len(item.text) // 4,
                        )

                    state_manager.update_progress(
                        source=source_name,
                        topic=topic,
                        items_added=collected,
                    )

                state_manager.mark_completed(source_name, topic)

            except Exception as e:
                logger.error(
                    "Collection failed",
                    source=source_name,
                    topic=topic,
                    error=str(e),
                )
                run_metrics.record_error()
                state_manager.mark_failed(source_name, topic, str(e))

            finally:
                metrics.add_run(run_metrics)

            progress.advance(main_task)

    # Finalize
    writer.flush_all()
    state_manager.finish_run("completed")
    metrics.log_final_summary()

    console.print(
        Panel(
            f"[bold green]Collection Complete![/bold green]\n\n"
            f"Total items: {writer.get_total_count()}\n"
            f"Output: {raw_output_dir}",
            title="Summary",
        )
    )


@app.command()
def process(
    input_dir: Path = typer.Option(
        Path("./outputs/raw"),
        "--input-dir", "-i",
        help="Input directory with raw data",
    ),
    output_dir: Path = typer.Option(
        Path("./outputs/processed"),
        "--output-dir", "-o",
        help="Output directory for processed data",
    ),
    quality_threshold: float = typer.Option(
        0.6,
        "--quality-threshold", "-q",
        help="Minimum quality score (0-1)",
    ),
    deduplicate: bool = typer.Option(
        True,
        "--deduplicate/--no-deduplicate",
        help="Remove duplicates",
    ),
    remove_pii: bool = typer.Option(
        True,
        "--remove-pii/--no-remove-pii",
        help="Remove personally identifiable information",
    ),
    filter_content: bool = typer.Option(
        True,
        "--filter-content/--no-filter-content",
        help="Filter inappropriate content",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level", "-l",
        help="Logging level",
    ),
) -> None:
    """
    Process and clean collected data.
    """
    setup_logging(log_level=log_level, json_format=False)

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            f"[bold blue]Data Processing[/bold blue]\n\n"
            f"Input: {input_dir}\n"
            f"Output: {output_dir}\n"
            f"Quality threshold: {quality_threshold}\n"
            f"Deduplicate: {deduplicate}\n"
            f"Remove PII: {remove_pii}",
            title="Configuration",
        )
    )

    # Initialize processors
    cleaner = TextCleaner()
    quality_filter = QualityFilter(min_quality_score=quality_threshold)
    tokenizer = TokenizerChecker()
    dedup = Deduplicator() if deduplicate else None
    pii_detector = PIIDetector() if remove_pii else None
    content_filter = ContentFilter() if filter_content else None
    language_detector = LanguageDetector(target_languages=["pt"])

    # Statistics
    stats = {
        "total_input": 0,
        "total_output": 0,
        "filtered_quality": 0,
        "filtered_duplicate": 0,
        "filtered_content": 0,
        "filtered_language": 0,
        "pii_removed": 0,
    }

    # Process all JSONL files
    input_files = list(input_dir.glob("*.jsonl"))
    console.print(f"[blue]Found {len(input_files)} input files[/blue]")

    writer = OutputWriter(output_dir, buffer_size=100)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        file_task = progress.add_task("[cyan]Processing files...", total=len(input_files))

        for input_file in input_files:
            progress.update(file_task, description=f"[cyan]{input_file.name}")

            with open(input_file) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        stats["total_input"] += 1

                        text = data.get("text", "")

                        # Clean text
                        cleaned_text = cleaner.clean(text)

                        # Check language
                        if not language_detector.is_target_language(cleaned_text):
                            stats["filtered_language"] += 1
                            continue

                        # Check quality
                        passed, score = quality_filter.filter(cleaned_text)
                        if not passed:
                            stats["filtered_quality"] += 1
                            continue

                        # Check duplicates
                        if dedup:
                            item_id = data.get("id", str(stats["total_input"]))
                            if dedup.check_and_add(item_id, cleaned_text):
                                stats["filtered_duplicate"] += 1
                                continue

                        # Check content
                        if content_filter:
                            result = content_filter.filter(cleaned_text)
                            if not result.passed:
                                stats["filtered_content"] += 1
                                continue

                        # Remove PII
                        if pii_detector:
                            pii_result = pii_detector.detect(cleaned_text)
                            if pii_result.has_pii:
                                cleaned_text = pii_result.cleaned_text
                                stats["pii_removed"] += 1

                        # Create processed entry
                        token_count = tokenizer.count_tokens(cleaned_text)
                        data["text"] = cleaned_text
                        data["quality_score"] = score.total_score
                        data["token_count"] = token_count
                        data["word_count"] = len(cleaned_text.split())

                        writer.write(data)
                        stats["total_output"] += 1

                    except json.JSONDecodeError:
                        continue

            progress.advance(file_task)

    writer.flush_all()

    # Print summary
    table = Table(title="Processing Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Input", str(stats["total_input"]))
    table.add_row("Total Output", str(stats["total_output"]))
    table.add_row("Filtered (Quality)", str(stats["filtered_quality"]))
    table.add_row("Filtered (Duplicate)", str(stats["filtered_duplicate"]))
    table.add_row("Filtered (Content)", str(stats["filtered_content"]))
    table.add_row("Filtered (Language)", str(stats["filtered_language"]))
    table.add_row("PII Removed", str(stats["pii_removed"]))
    table.add_row(
        "Pass Rate",
        f"{stats['total_output'] / max(1, stats['total_input']) * 100:.1f}%",
    )

    console.print(table)


@app.command()
def format(
    input_dir: Path = typer.Option(
        Path("./outputs/processed"),
        "--input-dir", "-i",
        help="Input directory with processed data",
    ),
    output_dir: Path = typer.Option(
        Path("./outputs/final"),
        "--output-dir", "-o",
        help="Output directory for formatted data",
    ),
    formats: list[str] = typer.Option(
        ["alpaca"],
        "--formats", "-f",
        help="Output formats (alpaca, sharegpt, chatml)",
    ),
    split_ratio: str = typer.Option(
        "0.9,0.05,0.05",
        "--split-ratio",
        help="Train,val,test split ratio",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level", "-l",
        help="Logging level",
    ),
) -> None:
    """
    Format dataset for different model formats.
    """
    setup_logging(log_level=log_level, json_format=False)

    # Parse split ratio
    ratios = [float(x) for x in split_ratio.split(",")]
    if len(ratios) != 3:
        raise typer.BadParameter("Split ratio must have 3 values")

    console.print(
        Panel(
            f"[bold magenta]Dataset Formatting[/bold magenta]\n\n"
            f"Input: {input_dir}\n"
            f"Output: {output_dir}\n"
            f"Formats: {', '.join(formats)}\n"
            f"Split: {split_ratio}",
            title="Configuration",
        )
    )

    formatter = DatasetFormatter(formats=formats)

    # Load all processed data
    entries = []
    input_files = list(input_dir.glob("*.jsonl"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        load_task = progress.add_task(
            "[cyan]Loading data...", total=len(input_files)
        )

        for input_file in input_files:
            with open(input_file) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        entry = DatasetEntry(
                            id=data.get("id", ""),
                            source=data.get("source", "unknown"),
                            topic=data.get("topic", "unknown"),
                            subtopic=data.get("subtopic"),
                            language=data.get("language", "pt_br"),
                            text=data.get("text", ""),
                            title=data.get("title"),
                            summary=data.get("summary"),
                            url=data.get("source_url"),
                            author=data.get("author"),
                            quality_score=data.get("quality_score", 0.7),
                            token_count=data.get("token_count", 0),
                            word_count=data.get("word_count", 0),
                        )
                        entries.append(entry)
                    except (json.JSONDecodeError, Exception):
                        continue

            progress.advance(load_task)

    console.print(f"[blue]Loaded {len(entries)} entries[/blue]")

    # Format and write for each format
    for fmt in formats:
        fmt_output_dir = output_dir / fmt
        counts = formatter.write_formatted(
            entries,
            fmt_output_dir,
            output_format=fmt,
            split_ratio=tuple(ratios),
        )

        console.print(
            f"[green]{fmt}:[/green] train={counts['train']}, "
            f"val={counts['val']}, test={counts['test']}"
        )

    console.print(
        Panel(
            f"[bold green]Formatting Complete![/bold green]\n\n"
            f"Output: {output_dir}",
            title="Summary",
        )
    )


@app.command()
def stats(
    input_dir: Path = typer.Option(
        Path("./outputs/final"),
        "--input-dir", "-i",
        help="Directory to analyze",
    ),
) -> None:
    """
    Display dataset statistics.
    """
    console.print(
        Panel(
            f"[bold yellow]Dataset Statistics[/bold yellow]\n\n"
            f"Directory: {input_dir}",
            title="Analysis",
        )
    )

    # Collect stats
    total_files = 0
    total_entries = 0
    total_tokens = 0
    by_format: dict[str, int] = {}
    by_split: dict[str, int] = {}
    by_topic: dict[str, int] = {}
    by_source: dict[str, int] = {}

    for fmt_dir in input_dir.iterdir():
        if not fmt_dir.is_dir():
            continue

        fmt_name = fmt_dir.name
        by_format[fmt_name] = 0

        for jsonl_file in fmt_dir.glob("*.jsonl"):
            total_files += 1
            split_name = jsonl_file.stem

            with open(jsonl_file) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        total_entries += 1
                        by_format[fmt_name] += 1
                        by_split[split_name] = by_split.get(split_name, 0) + 1

                        # Get metadata if available
                        metadata = data.get("metadata", {})
                        topic = metadata.get("topic", "unknown")
                        source = metadata.get("source", "unknown")

                        by_topic[topic] = by_topic.get(topic, 0) + 1
                        by_source[source] = by_source.get(source, 0) + 1

                        # Estimate tokens
                        if "output" in data:
                            total_tokens += len(data["output"]) // 4
                        elif "messages" in data:
                            for msg in data["messages"]:
                                total_tokens += len(msg.get("content", "")) // 4

                    except json.JSONDecodeError:
                        continue

    # Display tables
    summary_table = Table(title="Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    summary_table.add_row("Total Files", str(total_files))
    summary_table.add_row("Total Entries", str(total_entries))
    summary_table.add_row("Estimated Tokens", f"{total_tokens:,}")
    console.print(summary_table)

    if by_format:
        format_table = Table(title="By Format")
        format_table.add_column("Format", style="cyan")
        format_table.add_column("Count", style="green")
        for fmt, count in sorted(by_format.items()):
            format_table.add_row(fmt, str(count))
        console.print(format_table)

    if by_split:
        split_table = Table(title="By Split")
        split_table.add_column("Split", style="cyan")
        split_table.add_column("Count", style="green")
        for split, count in sorted(by_split.items()):
            split_table.add_row(split, str(count))
        console.print(split_table)

    if by_topic:
        topic_table = Table(title="By Topic")
        topic_table.add_column("Topic", style="cyan")
        topic_table.add_column("Count", style="green")
        for topic, count in sorted(by_topic.items(), key=lambda x: -x[1])[:10]:
            topic_table.add_row(topic, str(count))
        console.print(topic_table)


@app.command()
def validate(
    input_dir: Path = typer.Option(
        Path("./outputs/final"),
        "--input-dir", "-i",
        help="Directory to validate",
    ),
    sample_size: int = typer.Option(
        100,
        "--sample-size", "-n",
        help="Number of samples to validate",
    ),
) -> None:
    """
    Validate dataset quality.
    """
    import random

    console.print(
        Panel(
            f"[bold cyan]Dataset Validation[/bold cyan]\n\n"
            f"Directory: {input_dir}\n"
            f"Sample size: {sample_size}",
            title="Validation",
        )
    )

    # Load sample entries
    all_entries = []
    for jsonl_file in input_dir.rglob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    all_entries.append(data)
                except json.JSONDecodeError:
                    continue

    if not all_entries:
        console.print("[red]No entries found![/red]")
        return

    # Sample
    sample = random.sample(all_entries, min(sample_size, len(all_entries)))

    # Validate
    quality_filter = QualityFilter()
    content_filter = ContentFilter()
    tokenizer = TokenizerChecker()

    issues = {
        "empty_output": 0,
        "low_quality": 0,
        "content_issues": 0,
        "token_issues": 0,
    }

    for entry in sample:
        # Get the output text
        if "output" in entry:
            text = entry["output"]
        elif "messages" in entry:
            text = entry["messages"][-1].get("content", "")
        elif "conversations" in entry:
            text = entry["conversations"][-1].get("value", "")
        else:
            text = ""

        if not text.strip():
            issues["empty_output"] += 1
            continue

        # Check quality
        passed, score = quality_filter.filter(text)
        if not passed:
            issues["low_quality"] += 1

        # Check content
        result = content_filter.filter(text)
        if not result.passed:
            issues["content_issues"] += 1

        # Check tokens
        valid, info = tokenizer.validate(text)
        if not valid:
            issues["token_issues"] += 1

    # Display results
    table = Table(title="Validation Results")
    table.add_column("Issue", style="cyan")
    table.add_column("Count", style="yellow")
    table.add_column("Rate", style="red")

    for issue, count in issues.items():
        rate = count / len(sample) * 100
        table.add_row(issue, str(count), f"{rate:.1f}%")

    console.print(table)

    # Overall assessment
    total_issues = sum(issues.values())
    issue_rate = total_issues / len(sample) * 100

    if issue_rate < 5:
        console.print("[bold green]Dataset quality: EXCELLENT[/bold green]")
    elif issue_rate < 15:
        console.print("[bold yellow]Dataset quality: GOOD[/bold yellow]")
    elif issue_rate < 30:
        console.print("[bold orange]Dataset quality: FAIR[/bold orange]")
    else:
        console.print("[bold red]Dataset quality: NEEDS IMPROVEMENT[/bold red]")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
