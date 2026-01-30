import json
import glob
import hashlib
import os
import logging
from typing import List, Dict, Any
from datasets import Dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_dataset_format(record: Dict[str, Any]) -> str:
    """
    Detects the format of a dataset record.

    Supports:
    - Alpaca: {"instruction": "...", "input": "...", "output": "..."}
    - ShareGPT: {"conversations": [{"from": "...", "value": "..."}]}
    - ChatML: {"messages": [{"role": "...", "content": "..."}]}

    Returns:
        Format name: 'alpaca', 'sharegpt', 'chatml', or 'unknown'
    """
    if "messages" in record:
        return "chatml"
    elif "conversations" in record:
        return "sharegpt"
    elif "instruction" in record and "output" in record:
        return "alpaca"
    else:
        return "unknown"

def normalize_to_chatml(record: Dict[str, Any], format_type: str) -> Dict[str, Any]:
    """
    Converts a record from any supported format to ChatML format.

    Args:
        record: The dataset record
        format_type: Format type ('alpaca', 'sharegpt', 'chatml')

    Returns:
        Record with 'messages' field in ChatML format
    """
    if format_type == "chatml":
        # Already in ChatML format, return as is
        return record

    elif format_type == "alpaca":
        # Convert Alpaca to ChatML
        instruction = record.get("instruction", "")
        input_text = record.get("input", "")
        output_text = record.get("output", "")

        # Combine instruction and input if input exists
        user_content = instruction
        if input_text:
            user_content += f"\n\nContexto: {input_text}"

        messages = [
            {
                "role": "system",
                "content": "Você é um assistente útil e preciso que fornece informações detalhadas em português brasileiro."
            },
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": output_text
            }
        ]

        # Preserve metadata if present
        result = {"messages": messages}
        if "metadata" in record:
            result["metadata"] = record["metadata"]

        return result

    elif format_type == "sharegpt":
        # Convert ShareGPT to ChatML
        conversations = record.get("conversations", [])
        messages = []

        # Add a default system message
        messages.append({
            "role": "system",
            "content": "Você é um assistente útil e preciso que fornece informações detalhadas em português brasileiro."
        })

        # Convert conversation turns
        for turn in conversations:
            from_role = turn.get("from", "")
            value = turn.get("value", "")

            if from_role == "human":
                messages.append({"role": "user", "content": value})
            elif from_role == "gpt":
                messages.append({"role": "assistant", "content": value})

        # Preserve metadata if present
        result = {"messages": messages}
        if "metadata" in record:
            result["metadata"] = record["metadata"]

        return result

    else:
        logger.warning(f"Unknown format type: {format_type}")
        return record

def extract_content_for_hash(record: Dict[str, Any], format_type: str) -> str:
    """
    Extracts content from a record for deduplication hashing.

    Args:
        record: The dataset record
        format_type: Format type

    Returns:
        String representation of the content for hashing
    """
    if format_type == "alpaca":
        return f"{record.get('instruction', '')}|{record.get('input', '')}|{record.get('output', '')}"

    elif format_type == "sharegpt":
        conversations = record.get("conversations", [])
        content_parts = [f"{turn.get('from', '')}:{turn.get('value', '')}" for turn in conversations]
        return "|".join(content_parts)

    elif format_type == "chatml":
        messages = record.get("messages", [])
        content_parts = [f"{msg.get('role', '')}:{msg.get('content', '')}" for msg in messages]
        return "|".join(content_parts)

    else:
        # Fallback: stringify the entire record
        return json.dumps(record, sort_keys=True)

def load_and_merge_datasets(file_patterns: List[str]) -> List[Dict[str, Any]]:
    """
    Loads JSONL datasets from file patterns, deduplicates them, and returns a list of records.

    Supports multiple formats:
    - Alpaca: {"instruction": "...", "input": "...", "output": "..."}
    - ShareGPT: {"conversations": [{"from": "...", "value": "..."}]}
    - ChatML: {"messages": [{"role": "...", "content": "..."}]}

    All formats are normalized to ChatML format with 'messages' field.
    """
    all_records = []
    seen_hashes = set()
    files = []
    format_stats = {"alpaca": 0, "sharegpt": 0, "chatml": 0, "unknown": 0}

    for pattern in file_patterns:
        files.extend(glob.glob(pattern, recursive=True))

    if not files:
        logger.warning(f"No files found for patterns: {file_patterns}")
        return []

    logger.info(f"Loading datasets from {len(files)} files")
    logger.info(f"File patterns: {file_patterns}")

    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)

                        # Detect format
                        format_type = detect_dataset_format(record)
                        format_stats[format_type] += 1

                        # Extract content for deduplication
                        content_str = extract_content_for_hash(record, format_type)
                        content_hash = hashlib.md5(content_str.encode('utf-8')).hexdigest()

                        if content_hash not in seen_hashes:
                            seen_hashes.add(content_hash)

                            # Normalize to ChatML format
                            normalized_record = normalize_to_chatml(record, format_type)
                            all_records.append(normalized_record)

                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line {line_num} in {file_path}: {e}")
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num} in {file_path}: {e}")

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")

    logger.info(f"Total unique records loaded: {len(all_records)}")
    logger.info(f"Format distribution: Alpaca={format_stats['alpaca']}, ShareGPT={format_stats['sharegpt']}, ChatML={format_stats['chatml']}, Unknown={format_stats['unknown']}")

    return all_records

def customize_system_prompt(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Customizes the system prompt based on the topic metadata.

    If the record contains metadata with a topic, uses a specialized system prompt.
    Otherwise, keeps the default system prompt from the normalization step.

    Args:
        record: Record with 'messages' field

    Returns:
        Record with customized system prompt
    """
    # Topic-specific system prompts
    TOPIC_SYSTEM_PROMPTS = {
        "financeiro": "Você é um analista financeiro sênior especializado em mercados globais, criptoativos e macroeconomia. Forneça análises factuais, neutras e fundamentadas. Não dê conselhos de investimento diretos (buy/sell). Baseie-se nos dados fornecidos.",
        "tecnologia": "Você é um especialista em tecnologia que explica conceitos técnicos de forma clara e acessível em português brasileiro.",
        "juridico": "Você é um especialista em direito brasileiro que explica conceitos jurídicos de forma clara e precisa.",
        "saude": "Você é um profissional de saúde que fornece informações médicas precisas e responsáveis em português brasileiro.",
        "ciencias": "Você é um cientista que explica conceitos científicos de forma acessível e precisa em português brasileiro.",
        "humanidades": "Você é um especialista em ciências humanas que analisa questões históricas, filosóficas e sociais com profundidade.",
        "cultura": "Você é um especialista em cultura brasileira e mundial que fornece análises culturais aprofundadas.",
        "negocios": "Você é um consultor de negócios experiente que fornece insights estratégicos e práticos.",
        "educacao": "Você é um educador experiente que explica conceitos de forma didática e acessível.",
        "meio_ambiente": "Você é um especialista em meio ambiente e sustentabilidade que fornece análises baseadas em evidências científicas.",
    }

    messages = record.get("messages", [])
    metadata = record.get("metadata", {})
    topic = metadata.get("topic")

    # If there's a topic and we have a specialized prompt for it
    if topic and topic in TOPIC_SYSTEM_PROMPTS:
        # Find and replace the system message
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                messages[i]["content"] = TOPIC_SYSTEM_PROMPTS[topic]
                break

    return {"messages": messages}

def prepare_hf_dataset(file_patterns: List[str]) -> Dataset:
    """
    Loads, merges, and converts data to HuggingFace Dataset object with 'messages' column.

    Supports multiple input formats (Alpaca, ShareGPT, ChatML) and normalizes them all
    to ChatML format for training with Qwen 2.5.

    Args:
        file_patterns: List of glob patterns for dataset files

    Returns:
        HuggingFace Dataset with 'messages' column
    """
    # Load and normalize all datasets to ChatML format
    raw_data = load_and_merge_datasets(file_patterns)

    # Customize system prompts based on metadata topics
    formatted_data = [customize_system_prompt(record) for record in raw_data]

    logger.info(f"Prepared {len(formatted_data)} examples for training")

    return Dataset.from_list(formatted_data)
