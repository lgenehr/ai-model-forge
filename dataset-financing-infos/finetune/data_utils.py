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

def load_and_merge_datasets(file_patterns: List[str]) -> List[Dict[str, Any]]:
    """
    Loads JSONL datasets from file patterns, deduplicates them, and returns a list of records.
    """
    all_records = []
    seen_hashes = set()
    files = []
    
    for pattern in file_patterns:
        files.extend(glob.glob(pattern))
    
    if not files:
        logger.warning(f"No files found for patterns: {file_patterns}")
        return []

    logger.info(f"Loading datasets from: {files}")

    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        # Create a hash based on instruction + output to identify duplicates
                        # We ignore 'input' if it's empty, and 'meta' for deduplication
                        content_str = f"{record.get('instruction', '')}|{record.get('input', '')}|{record.get('output', '')}"
                        content_hash = hashlib.md5(content_str.encode('utf-8')).hexdigest()
                        
                        if content_hash not in seen_hashes:
                            seen_hashes.add(content_hash)
                            all_records.append(record)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line in {file_path}")
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")

    logger.info(f"Total unique records loaded: {len(all_records)}")
    return all_records

def format_for_qwen25(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Formats a dataset record into the Qwen 2.5 chat format.
    Qwen 2.5 typically uses the ChatML format or similar structured messages:
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")
    
    # Combine instruction and input if input exists
    user_content = instruction
    if input_text:
        user_content += f"\n\nContexto: {input_text}"

    messages = [
        {
            "role": "system",
            "content": "Você é um analista financeiro sênior especializado em mercados globais, criptoativos e macroeconomia. Forneça análises factuais, neutras e fundamentadas. Não dê conselhos de investimento diretos (buy/sell). Baseie-se nos dados fornecidos."
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
    
    return {"messages": messages}

def prepare_hf_dataset(file_patterns: List[str]) -> Dataset:
    """
    Loads, merges, and converts data to HuggingFace Dataset object with 'messages' column.
    """
    raw_data = load_and_merge_datasets(file_patterns)
    formatted_data = [format_for_qwen25(record) for record in raw_data]
    
    return Dataset.from_list(formatted_data)
