"""
Metrics collection and statistics utilities.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class CollectionMetrics:
    """Metrics for a single collection run."""

    source: str
    topic: str
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    # Counts
    total_fetched: int = 0
    total_processed: int = 0
    total_filtered: int = 0
    total_duplicates: int = 0
    total_errors: int = 0
    total_pii_removed: int = 0

    # Quality metrics
    quality_scores: list[float] = field(default_factory=list)
    token_counts: list[int] = field(default_factory=list)

    # Filter breakdown
    filter_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record_item(
        self,
        quality_score: float,
        token_count: int,
        filtered: bool = False,
        filter_reason: str | None = None,
        is_duplicate: bool = False,
        has_pii: bool = False,
    ) -> None:
        """Record metrics for a processed item."""
        self.total_fetched += 1

        if is_duplicate:
            self.total_duplicates += 1
            return

        if has_pii:
            self.total_pii_removed += 1

        if filtered:
            self.total_filtered += 1
            if filter_reason:
                self.filter_reasons[filter_reason] += 1
            return

        self.total_processed += 1
        self.quality_scores.append(quality_score)
        self.token_counts.append(token_count)

    def record_error(self) -> None:
        """Record an error."""
        self.total_errors += 1

    def finalize(self) -> None:
        """Finalize metrics collection."""
        self.end_time = time.time()

    @property
    def duration_seconds(self) -> float:
        """Get collection duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def avg_quality_score(self) -> float:
        """Get average quality score."""
        if not self.quality_scores:
            return 0.0
        return sum(self.quality_scores) / len(self.quality_scores)

    @property
    def avg_token_count(self) -> float:
        """Get average token count."""
        if not self.token_counts:
            return 0.0
        return sum(self.token_counts) / len(self.token_counts)

    @property
    def items_per_second(self) -> float:
        """Get processing rate."""
        if self.duration_seconds == 0:
            return 0.0
        return self.total_processed / self.duration_seconds

    @property
    def success_rate(self) -> float:
        """Get success rate (processed / fetched)."""
        if self.total_fetched == 0:
            return 0.0
        return self.total_processed / self.total_fetched

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "source": self.source,
            "topic": self.topic,
            "duration_seconds": round(self.duration_seconds, 2),
            "total_fetched": self.total_fetched,
            "total_processed": self.total_processed,
            "total_filtered": self.total_filtered,
            "total_duplicates": self.total_duplicates,
            "total_errors": self.total_errors,
            "total_pii_removed": self.total_pii_removed,
            "avg_quality_score": round(self.avg_quality_score, 4),
            "avg_token_count": round(self.avg_token_count, 1),
            "items_per_second": round(self.items_per_second, 2),
            "success_rate": round(self.success_rate, 4),
            "filter_reasons": dict(self.filter_reasons),
        }

    def log_summary(self) -> None:
        """Log a summary of the metrics."""
        logger.info(
            "Collection metrics summary",
            **self.to_dict(),
        )


class MetricsCollector:
    """
    Aggregates metrics from multiple collection runs.
    """

    def __init__(self) -> None:
        self.runs: list[CollectionMetrics] = []
        self.start_time = datetime.now()

    def add_run(self, metrics: CollectionMetrics) -> None:
        """Add a collection run's metrics."""
        metrics.finalize()
        self.runs.append(metrics)
        metrics.log_summary()

    def create_run(self, source: str, topic: str) -> CollectionMetrics:
        """Create a new metrics collection run."""
        return CollectionMetrics(source=source, topic=topic)

    @property
    def total_items(self) -> int:
        """Get total processed items across all runs."""
        return sum(run.total_processed for run in self.runs)

    @property
    def total_errors(self) -> int:
        """Get total errors across all runs."""
        return sum(run.total_errors for run in self.runs)

    @property
    def total_filtered(self) -> int:
        """Get total filtered items across all runs."""
        return sum(run.total_filtered for run in self.runs)

    @property
    def total_duplicates(self) -> int:
        """Get total duplicates across all runs."""
        return sum(run.total_duplicates for run in self.runs)

    @property
    def overall_quality(self) -> float:
        """Get overall average quality score."""
        all_scores = []
        for run in self.runs:
            all_scores.extend(run.quality_scores)
        if not all_scores:
            return 0.0
        return sum(all_scores) / len(all_scores)

    def get_stats_by_source(self) -> dict[str, dict[str, Any]]:
        """Get statistics grouped by source."""
        by_source: dict[str, list[CollectionMetrics]] = defaultdict(list)
        for run in self.runs:
            by_source[run.source].append(run)

        stats = {}
        for source, runs in by_source.items():
            stats[source] = {
                "total_processed": sum(r.total_processed for r in runs),
                "total_filtered": sum(r.total_filtered for r in runs),
                "total_errors": sum(r.total_errors for r in runs),
                "avg_quality": (
                    sum(r.avg_quality_score * r.total_processed for r in runs)
                    / max(1, sum(r.total_processed for r in runs))
                ),
                "topics_covered": list({r.topic for r in runs}),
            }
        return stats

    def get_stats_by_topic(self) -> dict[str, dict[str, Any]]:
        """Get statistics grouped by topic."""
        by_topic: dict[str, list[CollectionMetrics]] = defaultdict(list)
        for run in self.runs:
            by_topic[run.topic].append(run)

        stats = {}
        for topic, runs in by_topic.items():
            stats[topic] = {
                "total_processed": sum(r.total_processed for r in runs),
                "total_filtered": sum(r.total_filtered for r in runs),
                "total_errors": sum(r.total_errors for r in runs),
                "avg_quality": (
                    sum(r.avg_quality_score * r.total_processed for r in runs)
                    / max(1, sum(r.total_processed for r in runs))
                ),
                "sources_used": list({r.source for r in runs}),
            }
        return stats

    def get_filter_breakdown(self) -> dict[str, int]:
        """Get aggregated filter reasons breakdown."""
        breakdown: dict[str, int] = defaultdict(int)
        for run in self.runs:
            for reason, count in run.filter_reasons.items():
                breakdown[reason] += count
        return dict(breakdown)

    def to_dict(self) -> dict[str, Any]:
        """Get complete statistics as dictionary."""
        return {
            "summary": {
                "total_items": self.total_items,
                "total_filtered": self.total_filtered,
                "total_duplicates": self.total_duplicates,
                "total_errors": self.total_errors,
                "overall_quality": round(self.overall_quality, 4),
                "total_runs": len(self.runs),
                "start_time": self.start_time.isoformat(),
            },
            "by_source": self.get_stats_by_source(),
            "by_topic": self.get_stats_by_topic(),
            "filter_breakdown": self.get_filter_breakdown(),
        }

    def log_final_summary(self) -> None:
        """Log final summary of all collections."""
        stats = self.to_dict()
        logger.info(
            "Final collection summary",
            **stats["summary"],
        )

        logger.info(
            "Statistics by source",
            stats=stats["by_source"],
        )

        logger.info(
            "Statistics by topic",
            stats=stats["by_topic"],
        )

        if stats["filter_breakdown"]:
            logger.info(
                "Filter breakdown",
                reasons=stats["filter_breakdown"],
            )
