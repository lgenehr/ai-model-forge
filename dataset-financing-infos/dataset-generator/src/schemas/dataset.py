"""
Pydantic schemas for dataset entries and formats.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class DatasetFormat(str, Enum):
    """Supported dataset output formats."""

    ALPACA = "alpaca"
    SHAREGPT = "sharegpt"
    CHATML = "chatml"
    RAW = "raw"


class SourceType(str, Enum):
    """Types of data sources."""

    NEWS = "news"
    BOOKS = "books"
    ACADEMIC = "academic"
    LEGAL = "legal"
    SOCIAL_MEDIA = "social_media"
    VIDEOS = "videos"
    ENCYCLOPEDIA = "encyclopedia"
    FORUMS = "forums"


class AlpacaFormat(BaseModel):
    """Alpaca format for instruction fine-tuning."""

    instruction: str = Field(..., description="The instruction/question")
    input: str = Field(default="", description="Optional input context")
    output: str = Field(..., description="The expected response")

    def to_dict(self) -> dict[str, str]:
        return {"instruction": self.instruction, "input": self.input, "output": self.output}


class ShareGPTMessage(BaseModel):
    """Single message in ShareGPT format."""

    model_config = {"populate_by_name": True}

    from_: str = Field(..., alias="from", description="Who sent the message (human/gpt)")
    value: str = Field(..., description="Message content")


class ShareGPTFormat(BaseModel):
    """ShareGPT format for conversation fine-tuning."""

    conversations: list[dict[str, str]] = Field(
        ..., description="List of conversation turns"
    )

    @classmethod
    def from_qa(cls, question: str, answer: str) -> "ShareGPTFormat":
        """Create from a simple Q&A pair."""
        return cls(
            conversations=[
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer},
            ]
        )

    def to_dict(self) -> dict[str, list[dict[str, str]]]:
        return {"conversations": self.conversations}


class ChatMLMessage(BaseModel):
    """Single message in ChatML format."""

    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatMLFormat(BaseModel):
    """ChatML format for chat fine-tuning."""

    messages: list[dict[str, str]] = Field(..., description="List of chat messages")

    @classmethod
    def from_qa(
        cls, question: str, answer: str, system: str | None = None
    ) -> "ChatMLFormat":
        """Create from a simple Q&A pair with optional system prompt."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        return cls(messages=messages)

    def to_dict(self) -> dict[str, list[dict[str, str]]]:
        return {"messages": self.messages}


class RawCollectedData(BaseModel):
    """Raw data collected from a source before processing."""

    id: str = Field(..., description="Unique identifier")
    source: str = Field(..., description="Source name (e.g., 'news', 'wikipedia')")
    source_url: str | None = Field(None, description="Original URL")

    title: str | None = Field(None, description="Content title")
    text: str = Field(..., description="Main text content")
    summary: str | None = Field(None, description="Summary if available")

    author: str | None = Field(None, description="Author name")
    published_date: datetime | None = Field(None, description="Publication date")
    collected_date: datetime = Field(default_factory=datetime.now)

    topic: str | None = Field(None, description="Assigned topic")
    subtopic: str | None = Field(None, description="Assigned subtopic")
    language: str = Field(default="pt_br", description="Content language")

    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DatasetEntry(BaseModel):
    """Processed and validated dataset entry ready for formatting."""

    id: str = Field(..., description="Unique identifier")
    source: str = Field(..., description="Data source")
    topic: str = Field(..., description="Main topic")
    subtopic: str | None = Field(None, description="Subtopic")
    language: str = Field(default="pt_br")

    # Content
    text: str = Field(..., description="Main text content")
    title: str | None = Field(None)
    summary: str | None = Field(None)

    # Metadata
    url: str | None = Field(None)
    author: str | None = Field(None)
    published_date: datetime | None = Field(None)
    collected_date: datetime = Field(default_factory=datetime.now)

    # Quality metrics
    quality_score: float = Field(ge=0, le=1)
    token_count: int = Field(ge=0)
    word_count: int = Field(ge=0)

    # Formatting
    alpaca_format: dict[str, str] | None = Field(None)
    sharegpt_format: dict[str, list[dict[str, str]]] | None = Field(None)
    chatml_format: dict[str, list[dict[str, str]]] | None = Field(None)

    @field_validator("quality_score")
    @classmethod
    def validate_quality_score(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("quality_score must be between 0 and 1")
        return round(v, 4)

    def to_alpaca(self, instruction: str) -> AlpacaFormat:
        """Convert to Alpaca format with given instruction."""
        return AlpacaFormat(instruction=instruction, input="", output=self.text)

    def to_sharegpt(self, question: str) -> ShareGPTFormat:
        """Convert to ShareGPT format."""
        return ShareGPTFormat.from_qa(question, self.text)

    def to_chatml(self, question: str, system: str | None = None) -> ChatMLFormat:
        """Convert to ChatML format."""
        return ChatMLFormat.from_qa(question, self.text, system)


class CollectionStats(BaseModel):
    """Statistics for a collection run."""

    source: str
    topic: str
    total_collected: int = 0
    total_filtered: int = 0
    total_duplicates: int = 0
    total_errors: int = 0
    avg_quality_score: float = 0.0
    avg_token_count: float = 0.0
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: datetime | None = None
    duration_seconds: float | None = None

    def finalize(self) -> None:
        """Mark collection as complete."""
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
