"""
Checkpoint system for resumable collection.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Checkpoint:
    """Represents a collection checkpoint."""

    source: str
    topic: str
    last_page: int = 0
    last_offset: int = 0
    total_collected: int = 0
    collected_ids: set[str] = field(default_factory=set)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert checkpoint to serializable dict."""
        return {
            "source": self.source,
            "topic": self.topic,
            "last_page": self.last_page,
            "last_offset": self.last_offset,
            "total_collected": self.total_collected,
            "collected_ids": list(self.collected_ids),
            "last_updated": self.last_updated,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        """Create checkpoint from dict."""
        data = data.copy()
        data["collected_ids"] = set(data.get("collected_ids", []))
        return cls(**data)

    def update(
        self,
        page: int | None = None,
        offset: int | None = None,
        item_id: str | None = None,
    ) -> None:
        """Update checkpoint state."""
        if page is not None:
            self.last_page = page
        if offset is not None:
            self.last_offset = offset
        if item_id is not None:
            self.collected_ids.add(item_id)
            self.total_collected = len(self.collected_ids)
        self.last_updated = datetime.now().isoformat()


class CheckpointManager:
    """
    Manages checkpoints for all collection sources.
    """

    def __init__(self, checkpoint_dir: Path) -> None:
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints: dict[str, Checkpoint] = {}

    def _get_checkpoint_path(self, source: str, topic: str) -> Path:
        """Get path for a checkpoint file."""
        return self.checkpoint_dir / f"{source}_{topic}_checkpoint.json"

    def _get_key(self, source: str, topic: str) -> str:
        """Get cache key for source/topic combo."""
        return f"{source}:{topic}"

    def load(self, source: str, topic: str) -> Checkpoint | None:
        """
        Load checkpoint for a source/topic combination.

        Args:
            source: Data source name
            topic: Topic name

        Returns:
            Checkpoint if exists, None otherwise
        """
        key = self._get_key(source, topic)

        # Check cache first
        if key in self._checkpoints:
            return self._checkpoints[key]

        # Load from file
        checkpoint_path = self._get_checkpoint_path(source, topic)
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path) as f:
                    data = json.load(f)
                checkpoint = Checkpoint.from_dict(data)
                self._checkpoints[key] = checkpoint
                logger.info(
                    "Checkpoint loaded",
                    source=source,
                    topic=topic,
                    total_collected=checkpoint.total_collected,
                    last_page=checkpoint.last_page,
                )
                return checkpoint
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    "Failed to load checkpoint",
                    source=source,
                    topic=topic,
                    error=str(e),
                )

        return None

    def save(self, checkpoint: Checkpoint) -> None:
        """
        Save a checkpoint.

        Args:
            checkpoint: Checkpoint to save
        """
        key = self._get_key(checkpoint.source, checkpoint.topic)
        self._checkpoints[key] = checkpoint

        checkpoint_path = self._get_checkpoint_path(
            checkpoint.source, checkpoint.topic
        )
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        logger.debug(
            "Checkpoint saved",
            source=checkpoint.source,
            topic=checkpoint.topic,
            total_collected=checkpoint.total_collected,
        )

    def get_or_create(self, source: str, topic: str) -> Checkpoint:
        """
        Get existing checkpoint or create new one.

        Args:
            source: Data source name
            topic: Topic name

        Returns:
            Existing or new checkpoint
        """
        checkpoint = self.load(source, topic)
        if checkpoint is None:
            checkpoint = Checkpoint(source=source, topic=topic)
            self._checkpoints[self._get_key(source, topic)] = checkpoint
            logger.info(
                "New checkpoint created",
                source=source,
                topic=topic,
            )
        return checkpoint

    def is_collected(self, source: str, topic: str, item_id: str) -> bool:
        """
        Check if an item has already been collected.

        Args:
            source: Data source name
            topic: Topic name
            item_id: Item identifier

        Returns:
            True if already collected
        """
        checkpoint = self.load(source, topic)
        if checkpoint is None:
            return False
        return item_id in checkpoint.collected_ids

    def mark_collected(
        self,
        source: str,
        topic: str,
        item_id: str,
        page: int | None = None,
        save: bool = True,
    ) -> None:
        """
        Mark an item as collected.

        Args:
            source: Data source name
            topic: Topic name
            item_id: Item identifier
            page: Current page number
            save: Whether to save immediately
        """
        checkpoint = self.get_or_create(source, topic)
        checkpoint.update(page=page, item_id=item_id)

        if save:
            self.save(checkpoint)

    def clear(self, source: str, topic: str) -> None:
        """
        Clear checkpoint for a source/topic.

        Args:
            source: Data source name
            topic: Topic name
        """
        key = self._get_key(source, topic)
        if key in self._checkpoints:
            del self._checkpoints[key]

        checkpoint_path = self._get_checkpoint_path(source, topic)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(
                "Checkpoint cleared",
                source=source,
                topic=topic,
            )

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all available checkpoints."""
        checkpoints = []
        for path in self.checkpoint_dir.glob("*_checkpoint.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                checkpoints.append(
                    {
                        "source": data.get("source"),
                        "topic": data.get("topic"),
                        "total_collected": data.get("total_collected", 0),
                        "last_updated": data.get("last_updated"),
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue
        return checkpoints

    def get_progress_summary(self) -> dict[str, Any]:
        """Get summary of all checkpoint progress."""
        checkpoints = self.list_checkpoints()
        return {
            "total_checkpoints": len(checkpoints),
            "total_collected": sum(c["total_collected"] for c in checkpoints),
            "by_source": {
                source: sum(
                    c["total_collected"] for c in checkpoints if c["source"] == source
                )
                for source in {c["source"] for c in checkpoints}
            },
            "by_topic": {
                topic: sum(
                    c["total_collected"] for c in checkpoints if c["topic"] == topic
                )
                for topic in {c["topic"] for c in checkpoints}
            },
        }
