"""
Dataset formatting for different model formats.
"""

import json
import random
from pathlib import Path
from typing import Any

from ..schemas.dataset import (
    AlpacaFormat,
    ChatMLFormat,
    DatasetEntry,
    DatasetFormat,
    ShareGPTFormat,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DatasetFormatter:
    """
    Formats dataset entries for different fine-tuning formats.
    """

    # Instruction templates by topic
    INSTRUCTION_TEMPLATES: dict[str, list[str]] = {
        "financeiro": [
            "Explique sobre {title}.",
            "O que você sabe sobre {topic} relacionado a finanças?",
            "Forneça informações detalhadas sobre {title}.",
            "Descreva {title} no contexto financeiro brasileiro.",
            "Como funciona {title}?",
        ],
        "tecnologia": [
            "Explique o conceito de {title}.",
            "O que é {title} em tecnologia?",
            "Descreva como funciona {title}.",
            "Forneça uma explicação técnica sobre {title}.",
            "Quais são os principais aspectos de {title}?",
        ],
        "ciencias": [
            "Explique o fenômeno/conceito de {title}.",
            "O que a ciência diz sobre {title}?",
            "Descreva {title} cientificamente.",
            "Quais são as bases científicas de {title}?",
        ],
        "saude": [
            "O que é {title} na área da saúde?",
            "Explique sobre {title} do ponto de vista médico.",
            "Forneça informações sobre {title} relacionado à saúde.",
            "Descreva {title} e sua importância para a saúde.",
        ],
        "juridico": [
            "Explique o aspecto jurídico de {title}.",
            "O que diz a lei brasileira sobre {title}?",
            "Descreva {title} do ponto de vista legal.",
            "Quais são os aspectos legais de {title}?",
        ],
        "humanidades": [
            "Explique {title} do ponto de vista histórico/filosófico.",
            "O que sabemos sobre {title} nas ciências humanas?",
            "Descreva a importância de {title} para a sociedade.",
            "Analise {title} sob a perspectiva humanística.",
        ],
        "cultura": [
            "Fale sobre {title} na cultura brasileira.",
            "O que representa {title} culturalmente?",
            "Descreva a importância cultural de {title}.",
            "Como {title} influencia a cultura?",
        ],
        "negocios": [
            "Explique {title} no contexto empresarial.",
            "Como funciona {title} nos negócios?",
            "Descreva {title} para empreendedores.",
            "Quais são as melhores práticas de {title}?",
        ],
        "educacao": [
            "Explique {title} no contexto educacional.",
            "Como {title} se aplica à educação?",
            "Descreva a importância de {title} para o aprendizado.",
        ],
        "meio_ambiente": [
            "Explique {title} no contexto ambiental.",
            "Qual a importância de {title} para o meio ambiente?",
            "Descreva {title} e seu impacto ambiental.",
        ],
    }

    DEFAULT_TEMPLATES = [
        "Forneça informações sobre {title}.",
        "Explique {title}.",
        "O que você sabe sobre {title}?",
        "Descreva {title} em detalhes.",
    ]

    SYSTEM_PROMPTS: dict[str, str] = {
        "default": "Você é um assistente útil e preciso que fornece informações detalhadas em português brasileiro.",
        "financeiro": "Você é um especialista em finanças e economia que fornece informações precisas sobre o mercado financeiro brasileiro.",
        "tecnologia": "Você é um especialista em tecnologia que explica conceitos técnicos de forma clara e acessível.",
        "juridico": "Você é um especialista em direito brasileiro que explica conceitos jurídicos de forma clara.",
        "saude": "Você é um profissional de saúde que fornece informações médicas precisas e responsáveis.",
        "ciencias": "Você é um cientista que explica conceitos científicos de forma acessível e precisa.",
    }

    def __init__(
        self,
        formats: list[str] | None = None,
        include_metadata: bool = True,
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize formatter.

        Args:
            formats: List of output formats
            include_metadata: Whether to include metadata in output
            random_seed: Random seed for template selection
        """
        self.formats = formats or ["alpaca", "sharegpt", "chatml"]
        self.include_metadata = include_metadata

        if random_seed is not None:
            random.seed(random_seed)

    def _get_instruction(self, entry: DatasetEntry) -> str:
        """Generate instruction for entry."""
        templates = self.INSTRUCTION_TEMPLATES.get(
            entry.topic, self.DEFAULT_TEMPLATES
        )
        template = random.choice(templates)

        title = entry.title or entry.topic
        return template.format(title=title, topic=entry.topic)

    def _get_system_prompt(self, topic: str) -> str:
        """Get system prompt for topic."""
        return self.SYSTEM_PROMPTS.get(topic, self.SYSTEM_PROMPTS["default"])

    def format_alpaca(self, entry: DatasetEntry) -> dict[str, Any]:
        """
        Format entry as Alpaca format.

        Args:
            entry: Dataset entry

        Returns:
            Alpaca format dict
        """
        instruction = self._get_instruction(entry)

        result = {
            "instruction": instruction,
            "input": "",
            "output": entry.text,
        }

        if self.include_metadata:
            result["metadata"] = {
                "id": entry.id,
                "source": entry.source,
                "topic": entry.topic,
                "quality_score": entry.quality_score,
            }

        return result

    def format_sharegpt(self, entry: DatasetEntry) -> dict[str, Any]:
        """
        Format entry as ShareGPT format.

        Args:
            entry: Dataset entry

        Returns:
            ShareGPT format dict
        """
        instruction = self._get_instruction(entry)

        result = {
            "conversations": [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": entry.text},
            ]
        }

        if self.include_metadata:
            result["metadata"] = {
                "id": entry.id,
                "source": entry.source,
                "topic": entry.topic,
                "quality_score": entry.quality_score,
            }

        return result

    def format_chatml(
        self,
        entry: DatasetEntry,
        include_system: bool = True,
    ) -> dict[str, Any]:
        """
        Format entry as ChatML format.

        Args:
            entry: Dataset entry
            include_system: Whether to include system message

        Returns:
            ChatML format dict
        """
        instruction = self._get_instruction(entry)
        messages = []

        if include_system:
            system_prompt = self._get_system_prompt(entry.topic)
            messages.append({"role": "system", "content": system_prompt})

        messages.extend(
            [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": entry.text},
            ]
        )

        result = {"messages": messages}

        if self.include_metadata:
            result["metadata"] = {
                "id": entry.id,
                "source": entry.source,
                "topic": entry.topic,
                "quality_score": entry.quality_score,
            }

        return result

    def format_entry(
        self,
        entry: DatasetEntry,
        output_format: str = "alpaca",
    ) -> dict[str, Any]:
        """
        Format entry in specified format.

        Args:
            entry: Dataset entry
            output_format: Output format

        Returns:
            Formatted dict
        """
        if output_format == "alpaca":
            return self.format_alpaca(entry)
        elif output_format == "sharegpt":
            return self.format_sharegpt(entry)
        elif output_format == "chatml":
            return self.format_chatml(entry)
        else:
            raise ValueError(f"Unknown format: {output_format}")

    def format_all(self, entry: DatasetEntry) -> dict[str, dict[str, Any]]:
        """
        Format entry in all configured formats.

        Args:
            entry: Dataset entry

        Returns:
            Dict of format -> formatted data
        """
        result = {}
        for fmt in self.formats:
            result[fmt] = self.format_entry(entry, fmt)
        return result

    def write_formatted(
        self,
        entries: list[DatasetEntry],
        output_dir: Path,
        output_format: str = "alpaca",
        split_ratio: tuple[float, float, float] = (0.9, 0.05, 0.05),
    ) -> dict[str, int]:
        """
        Write formatted entries to files with train/val/test split.

        Args:
            entries: List of entries
            output_dir: Output directory
            output_format: Output format
            split_ratio: Train/val/test split ratio

        Returns:
            Dict of split -> count
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Shuffle entries
        shuffled = list(entries)
        random.shuffle(shuffled)

        # Calculate split indices
        total = len(shuffled)
        train_end = int(total * split_ratio[0])
        val_end = train_end + int(total * split_ratio[1])

        splits = {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }

        counts = {}

        for split_name, split_entries in splits.items():
            output_path = output_dir / f"{split_name}.jsonl"

            with open(output_path, "w", encoding="utf-8") as f:
                for entry in split_entries:
                    formatted = self.format_entry(entry, output_format)
                    f.write(json.dumps(formatted, ensure_ascii=False) + "\n")

            counts[split_name] = len(split_entries)
            logger.info(
                "Wrote formatted data",
                split=split_name,
                count=len(split_entries),
                path=str(output_path),
            )

        return counts


def convert_raw_to_entry(
    raw_data: dict[str, Any],
    quality_score: float = 0.7,
    token_count: int = 0,
) -> DatasetEntry:
    """
    Convert raw collected data to DatasetEntry.

    Args:
        raw_data: Raw data dict
        quality_score: Quality score
        token_count: Token count

    Returns:
        DatasetEntry
    """
    text = raw_data.get("text", "")
    word_count = len(text.split())

    return DatasetEntry(
        id=raw_data.get("id", ""),
        source=raw_data.get("source", "unknown"),
        topic=raw_data.get("topic", "unknown"),
        subtopic=raw_data.get("subtopic"),
        language=raw_data.get("language", "pt_br"),
        text=text,
        title=raw_data.get("title"),
        summary=raw_data.get("summary"),
        url=raw_data.get("source_url"),
        author=raw_data.get("author"),
        published_date=raw_data.get("published_date"),
        quality_score=quality_score,
        token_count=token_count or len(text) // 4,
        word_count=word_count,
    )
