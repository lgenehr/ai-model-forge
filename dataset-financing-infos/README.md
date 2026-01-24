# Dataset Financing Infos (2025-2026)

This project generates a fine-tuning dataset for LLMs focused on factual, neutral information from 2025 and 2026. It covers business, markets, crypto, and macroeconomics without providing financial advice.

## Features
- **Neutrality:** Strict guardrails against financial recommendations ("buy", "sell", "pump").
- **Future-Proofing:** Targeted at collecting data for 2025-2026.
- **Safety:** Filters for harmful content.
- **Format:** Outputs JSONL in `instruction`, `input`, `output` format.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the build pipeline:
   ```bash
   python -m src.cli build --year 2025 --year 2026 --out dataset/dataset_2025_2026.jsonl
   ```

## Structure
- `src/connectors`: Modules to fetch data from RSS, APIs, etc.
- `src/pipeline`: Cleaning, deduplication, and guardrails.
- `configs`: Configuration for sources and topics.

## Compliance
This tool is designed to produce educational data only. It actively filters out subjective investment advice.
