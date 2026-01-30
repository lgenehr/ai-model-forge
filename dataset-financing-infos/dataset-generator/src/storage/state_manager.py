"""
State management for collection progress.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CollectionState:
    """Complete state of a collection run."""

    run_id: str
    started_at: str
    sources: list[str]
    topics: list[str]
    status: str = "running"  # running, paused, completed, failed
    current_source: str | None = None
    current_topic: str | None = None
    completed_pairs: list[tuple[str, str]] = field(default_factory=list)
    failed_pairs: list[tuple[str, str]] = field(default_factory=list)
    total_items: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to serializable dict."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "sources": self.sources,
            "topics": self.topics,
            "status": self.status,
            "current_source": self.current_source,
            "current_topic": self.current_topic,
            "completed_pairs": self.completed_pairs,
            "failed_pairs": self.failed_pairs,
            "total_items": self.total_items,
            "errors": self.errors,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CollectionState":
        """Create state from dict."""
        data = data.copy()
        data["completed_pairs"] = [tuple(p) for p in data.get("completed_pairs", [])]
        data["failed_pairs"] = [tuple(p) for p in data.get("failed_pairs", [])]
        return cls(**data)


class StateManager:
    """
    Manages persistent state for collection runs.
    Allows resuming interrupted collections.
    """

    def __init__(self, checkpoint_dir: Path) -> None:
        """
        Initialize state manager.

        Args:
            checkpoint_dir: Directory for state files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self.checkpoint_dir / "collection_state.json"
        self._current_state: CollectionState | None = None

    def create_run(
        self,
        sources: list[str],
        topics: list[str],
        config: dict[str, Any] | None = None,
    ) -> CollectionState:
        """
        Create a new collection run state.

        Args:
            sources: List of sources to collect from
            topics: List of topics to collect
            config: Collection configuration

        Returns:
            New collection state
        """
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_state = CollectionState(
            run_id=run_id,
            started_at=datetime.now().isoformat(),
            sources=sources,
            topics=topics,
            config=config or {},
        )
        self._save_state()
        logger.info(
            "Collection run created",
            run_id=run_id,
            sources=sources,
            topics=topics,
        )
        return self._current_state

    def load_state(self) -> CollectionState | None:
        """
        Load existing state if available.

        Returns:
            Loaded state or None
        """
        if self._current_state is not None:
            return self._current_state

        if self._state_file.exists():
            try:
                with open(self._state_file) as f:
                    data = json.load(f)
                self._current_state = CollectionState.from_dict(data)
                logger.info(
                    "State loaded",
                    run_id=self._current_state.run_id,
                    status=self._current_state.status,
                    total_items=self._current_state.total_items,
                )
                return self._current_state
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to load state", error=str(e))

        return None

    def _save_state(self) -> None:
        """Save current state to file."""
        if self._current_state is None:
            return

        with open(self._state_file, "w") as f:
            json.dump(self._current_state.to_dict(), f, indent=2)

    def update_progress(
        self,
        source: str | None = None,
        topic: str | None = None,
        items_added: int = 0,
    ) -> None:
        """
        Update collection progress.

        Args:
            source: Current source being collected
            topic: Current topic being collected
            items_added: Number of items added
        """
        if self._current_state is None:
            return

        if source is not None:
            self._current_state.current_source = source
        if topic is not None:
            self._current_state.current_topic = topic
        self._current_state.total_items += items_added
        self._save_state()

    def mark_completed(self, source: str, topic: str) -> None:
        """
        Mark a source/topic pair as completed.

        Args:
            source: Completed source
            topic: Completed topic
        """
        if self._current_state is None:
            return

        pair = (source, topic)
        if pair not in self._current_state.completed_pairs:
            self._current_state.completed_pairs.append(pair)
            logger.info(
                "Source/topic completed",
                source=source,
                topic=topic,
            )
        self._save_state()

    def mark_failed(self, source: str, topic: str, error: str) -> None:
        """
        Mark a source/topic pair as failed.

        Args:
            source: Failed source
            topic: Failed topic
            error: Error message
        """
        if self._current_state is None:
            return

        pair = (source, topic)
        if pair not in self._current_state.failed_pairs:
            self._current_state.failed_pairs.append(pair)
        self._current_state.errors.append(
            {
                "source": source,
                "topic": topic,
                "error": error,
                "timestamp": datetime.now().isoformat(),
            }
        )
        logger.error(
            "Source/topic failed",
            source=source,
            topic=topic,
            error=error,
        )
        self._save_state()

    def get_pending_pairs(self) -> list[tuple[str, str]]:
        """
        Get source/topic pairs that haven't been completed.

        Returns:
            List of pending (source, topic) pairs
        """
        if self._current_state is None:
            return []

        all_pairs = [
            (source, topic)
            for source in self._current_state.sources
            for topic in self._current_state.topics
        ]

        completed = set(self._current_state.completed_pairs)
        failed = set(self._current_state.failed_pairs)

        return [p for p in all_pairs if p not in completed and p not in failed]

    def finish_run(self, status: str = "completed") -> None:
        """
        Mark the collection run as finished.

        Args:
            status: Final status (completed, failed)
        """
        if self._current_state is None:
            return

        self._current_state.status = status
        self._current_state.current_source = None
        self._current_state.current_topic = None
        self._save_state()
        logger.info(
            "Collection run finished",
            run_id=self._current_state.run_id,
            status=status,
            total_items=self._current_state.total_items,
        )

    def clear_state(self) -> None:
        """Clear all state (for fresh start)."""
        self._current_state = None
        if self._state_file.exists():
            self._state_file.unlink()
        logger.info("State cleared")

    def get_summary(self) -> dict[str, Any]:
        """Get summary of current state."""
        if self._current_state is None:
            return {"status": "no_state"}

        pending = self.get_pending_pairs()
        return {
            "run_id": self._current_state.run_id,
            "status": self._current_state.status,
            "started_at": self._current_state.started_at,
            "total_items": self._current_state.total_items,
            "completed_pairs": len(self._current_state.completed_pairs),
            "failed_pairs": len(self._current_state.failed_pairs),
            "pending_pairs": len(pending),
            "total_errors": len(self._current_state.errors),
            "current_source": self._current_state.current_source,
            "current_topic": self._current_state.current_topic,
        }
