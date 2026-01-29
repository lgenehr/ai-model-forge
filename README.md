# ai-model-forge

A practical workspace for training, fine-tuning, evaluating, and experimenting with AI models. Focused on LLMs, datasets, pipelines, quantization, inference optimization, and reproducible ML workflows for research and production-oriented experimentation.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Contributing](#contributing)

## Overview

ai-model-forge is designed for researchers and engineers who need a flexible, production-ready environment for experimenting with large language models. It provides tools and scripts for data preprocessing, model fine-tuning, evaluation, and inference optimization.

## Features

- 🚀 Fine-tuning pipelines for LLMs
- 📊 Dataset management and preprocessing
- ⚡ Quantization and inference optimization
- 📈 Experiment tracking with Weights & Biases
- 🐳 Docker support for reproducible environments
- 🔄 LoRA and parameter-efficient fine-tuning
- 📝 Comprehensive logging and monitoring

## Prerequisites

- Docker with NVIDIA GPU support
- CUDA-capable GPU (recommended for training)
- Python 3.10+
- Git

## Installation

### Using Docker (Recommended)

Build the Docker image:

```bash
docker build -t ai-model-forge-app:latest .
```

### Local Setup

```bash
git clone https://github.com/yourusername/ai-model-forge.git
cd ai-model-forge
pip install -r requirements.txt
```

## Quick Start

### Fine-tuning with Docker

```bash
docker run --gpus all \
  -it --rm \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $(pwd)/..:/workspace \
  -w /workspace/ai-model-forge/dataset-financing-infos/finetune \
  ai-model-forge-app:latest \
  python train.py \
    --model_name "unsloth/Qwen2.5-14B-Instruct" \
    --output_dir "financial_finetune_v3_agressivo" \
    --llama_cpp_path "/opt/llama.cpp" \
    --batch_size 1 \
    --grad_accum_steps 16 \
    --epochs 3 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --dataset_num_proc 16 \
    --wandb_project "finetune-financeiro-qwen" \
    --wandb_run_name "run-v4-docker" --wandb_api_key "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"
```

### Fine-tuning Local

```bash
  python train.py \
    --model_name "unsloth/Qwen2.5-14B-Instruct" \
    --output_dir "financial_finetune_v4" \
    --llama_cpp_path "/opt/llama.cpp" \
    --batch_size 1 \
    --grad_accum_steps 16 \
    --epochs 3 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --dataset_num_proc 16 \
    --wandb_project "finetune-financeiro-qwen-v4" \
    --wandb_run_name "run-v4-docker" --wandb_api_key "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"
  ```

## Project Structure

```
ai-model-forge/
├── dataset-financing-infos/
│   ├── dataset/              # Raw and processed datasets
│   └── finetune/             # Fine-tuning scripts and configs
├── models/                   # Model checkpoints and outputs
├── notebooks/                # Jupyter notebooks for experimentation
├── src/                      # Source code and utilities
├── Dockerfile                # Docker configuration
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Development Workflow

This repository follows a development workflow where features are merged into the `development` branch before making it to `main`.

### Branch Strategy

- `main`: Production-ready code
- `development`: Integration branch for features
- `feature/*`: Feature branches from development

### Making Changes

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes and commit
3. Push to development: `git push origin feature/your-feature`
4. Create a Pull Request for review

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Submit a pull request to the `development` branch
5. Ensure all tests pass and code follows the project's style guide

## License

[Add your license here]

## Support

For issues, questions, or discussions, please open an issue on GitHub.
