"""
Output writer for JSONL files with buffering.
"""

import gzip
import json
from pathlib import Path
from typing import Any

from ..schemas.dataset import DatasetEntry, RawCollectedData
from ..utils.logger import get_logger

logger = get_logger(__name__)


class OutputWriter:
    """
    Buffered JSONL writer with automatic flushing.
    """

    def __init__(
        self,
        output_dir: Path,
        buffer_size: int = 100,
        compress: bool = False,
    ) -> None:
        """
        Initialize output writer.

        Args:
            output_dir: Directory for output files
            buffer_size: Number of items to buffer before flushing
            compress: Whether to compress output with gzip
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        self.compress = compress
        self._buffers: dict[str, list[dict[str, Any]]] = {}
        self._counts: dict[str, int] = {}

    def _get_output_path(self, source: str, topic: str, language: str) -> Path:
        """Get output file path for source/topic/language."""
        filename = f"{source}-{topic}-{language}.jsonl"
        if self.compress:
            filename += ".gz"
        return self.output_dir / filename

    def _get_buffer_key(self, source: str, topic: str, language: str) -> str:
        """Get buffer key for source/topic/language."""
        return f"{source}:{topic}:{language}"

    def write(
        self,
        data: RawCollectedData | DatasetEntry | dict[str, Any],
        source: str | None = None,
        topic: str | None = None,
        language: str | None = None,
    ) -> None:
        """
        Write a data item to the appropriate file.

        Args:
            data: Data item to write
            source: Override source from data
            topic: Override topic from data
            language: Override language from data
        """
        # Convert to dict if needed
        if isinstance(data, (RawCollectedData, DatasetEntry)):
            item_dict = data.model_dump(mode="json")
            source = source or data.source
            topic = topic or data.topic
            language = language or data.language
        else:
            item_dict = data
            source = source or data.get("source", "unknown")
            topic = topic or data.get("topic", "unknown")
            language = language or data.get("language", "pt_br")

        # Add to buffer
        key = self._get_buffer_key(source, topic, language)
        if key not in self._buffers:
            self._buffers[key] = []
            self._counts[key] = 0

        self._buffers[key].append(item_dict)
        self._counts[key] += 1

        # Flush if buffer is full
        if len(self._buffers[key]) >= self.buffer_size:
            self._flush_buffer(source, topic, language)

    def _flush_buffer(self, source: str, topic: str, language: str) -> None:
        """Flush buffer to file."""
        key = self._get_buffer_key(source, topic, language)
        if key not in self._buffers or not self._buffers[key]:
            return

        output_path = self._get_output_path(source, topic, language)

        # Write to file
        if self.compress:
            with gzip.open(output_path, "at", encoding="utf-8") as f:
                for item in self._buffers[key]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            with open(output_path, "a", encoding="utf-8") as f:
                for item in self._buffers[key]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        items_flushed = len(self._buffers[key])
        self._buffers[key] = []

        logger.debug(
            "Buffer flushed",
            source=source,
            topic=topic,
            language=language,
            items=items_flushed,
            total=self._counts[key],
        )

    def flush_all(self) -> None:
        """Flush all buffers to files."""
        for key in list(self._buffers.keys()):
            source, topic, language = key.split(":")
            self._flush_buffer(source, topic, language)

        logger.info(
            "All buffers flushed",
            total_items=sum(self._counts.values()),
        )

    def get_counts(self) -> dict[str, int]:
        """Get counts by source/topic/language."""
        return self._counts.copy()

    def get_total_count(self) -> int:
        """Get total items written."""
        return sum(self._counts.values())

    def __enter__(self) -> "OutputWriter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.flush_all()


class MultiFormatWriter:
    """
    Writer that outputs to multiple formats simultaneously.
    """

    def __init__(
        self,
        base_dir: Path,
        formats: list[str] | None = None,
        buffer_size: int = 100,
    ) -> None:
        """
        Initialize multi-format writer.

        Args:
            base_dir: Base output directory
            formats: List of formats to write (alpaca, sharegpt, chatml)
            buffer_size: Buffer size for each format writer
        """
        self.base_dir = Path(base_dir)
        self.formats = formats or ["alpaca", "sharegpt", "chatml"]
        self._writers: dict[str, OutputWriter] = {}

        for fmt in self.formats:
            fmt_dir = self.base_dir / fmt
            self._writers[fmt] = OutputWriter(fmt_dir, buffer_size=buffer_size)

    def write(
        self,
        entry: DatasetEntry,
        instruction: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """
        Write entry in all configured formats.

        Args:
            entry: Dataset entry to write
            instruction: Instruction for the entry
            system_prompt: System prompt for ChatML format
        """
        instruction = instruction or f"Sobre o tema {entry.topic}: {entry.title or 'informação relevante'}"

        # Alpaca format
        if "alpaca" in self._writers:
            alpaca_data = {
                "instruction": instruction,
                "input": "",
                "output": entry.text,
                "metadata": {
                    "id": entry.id,
                    "source": entry.source,
                    "topic": entry.topic,
                    "quality_score": entry.quality_score,
                },
            }
            self._writers["alpaca"].write(
                alpaca_data,
                source=entry.source,
                topic=entry.topic,
                language=entry.language,
            )

        # ShareGPT format
        if "sharegpt" in self._writers:
            sharegpt_data = {
                "conversations": [
                    {"from": "human", "value": instruction},
                    {"from": "gpt", "value": entry.text},
                ],
                "metadata": {
                    "id": entry.id,
                    "source": entry.source,
                    "topic": entry.topic,
                    "quality_score": entry.quality_score,
                },
            }
            self._writers["sharegpt"].write(
                sharegpt_data,
                source=entry.source,
                topic=entry.topic,
                language=entry.language,
            )

        # ChatML format
        if "chatml" in self._writers:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(
                [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": entry.text},
                ]
            )
            chatml_data = {
                "messages": messages,
                "metadata": {
                    "id": entry.id,
                    "source": entry.source,
                    "topic": entry.topic,
                    "quality_score": entry.quality_score,
                },
            }
            self._writers["chatml"].write(
                chatml_data,
                source=entry.source,
                topic=entry.topic,
                language=entry.language,
            )

    def flush_all(self) -> None:
        """Flush all format writers."""
        for writer in self._writers.values():
            writer.flush_all()

    def get_counts(self) -> dict[str, dict[str, int]]:
        """Get counts by format."""
        return {fmt: writer.get_counts() for fmt, writer in self._writers.items()}

    def __enter__(self) -> "MultiFormatWriter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.flush_all()
