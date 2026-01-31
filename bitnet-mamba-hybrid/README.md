# BitNet-Mamba Hybrid Architecture

Uma implementação modular e profissional que combina a eficiência do **Mamba** (State Space Model) com a quantização de **1.58-bit do BitNet**, otimizada para hardware consumer-grade.

## Índice

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Treinamento](#treinamento)
- [Weights & Biases](#weights--biases)
- [Inferência](#inferência)
- [Configuração](#configuração)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Referências](#referências)

---

## Visão Geral

Este projeto implementa um modelo de linguagem híbrido que une duas inovações recentes em arquiteturas de IA:

| Componente | Descrição | Benefício |
|------------|-----------|-----------|
| **Mamba SSM** | State Space Model com seleção dependente de entrada | Complexidade linear O(n) vs O(n²) dos Transformers |
| **BitNet b1.58** | Quantização ternária {-1, 0, 1} | ~10x redução de memória, operações mais rápidas |

### Hardware Alvo

- **GPU:** NVIDIA RTX 4070 Ti Super (16GB VRAM)
- **CPU:** AMD Ryzen 9 7950X
- **Precisão:** bfloat16 (melhor estabilidade numérica)

---

## Arquitetura

### Diagrama de Alto Nível

```
┌─────────────────────────────────────────────────────────────┐
│                    BitNetMambaLM                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐                                            │
│  │  Embedding  │  (vocab_size → d_model)                    │
│  └──────┬──────┘                                            │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────────┐               │
│  │         MambaBlock × n_layers           │               │
│  │  ┌───────────────────────────────────┐  │               │
│  │  │  RMSNorm → BitLinear (in_proj)    │  │               │
│  │  │         ↓                         │  │               │
│  │  │  ┌─────────┐    ┌─────────┐       │  │               │
│  │  │  │ x_path  │    │    z    │       │  │               │
│  │  │  └────┬────┘    └────┬────┘       │  │               │
│  │  │       │              │            │  │               │
│  │  │  Conv1D + SiLU       │            │  │               │
│  │  │       │              │            │  │               │
│  │  │  SSM (Δ, B, C, A)    │            │  │               │
│  │  │       │              │            │  │               │
│  │  │       └──── × ───────┘            │  │               │
│  │  │              │                    │  │               │
│  │  │  BitLinear (out_proj) + Residual  │  │               │
│  │  └───────────────────────────────────┘  │               │
│  └─────────────────────────────────────────┘               │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────┐                                            │
│  │   RMSNorm   │                                            │
│  └──────┬──────┘                                            │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────┐                                            │
│  │   LM Head   │  (d_model → vocab_size, tied weights)      │
│  └─────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
```

### Componentes Principais

#### 1. BitLinear (Camada Linear Quantizada)

```python
# Quantização ternária seguindo o paper BitNet b1.58
def quantize_weights_ternary(weight):
    scale = weight.abs().mean()           # Fator de escala
    weight_norm = weight / scale          # Normalização
    weight_quant = weight_norm.round()    # Arredonda para {-1, 0, 1}
    return weight_quant, scale
```

**Características:**
- Pesos quantizados para {-1, 0, 1} durante forward pass
- RMSNorm aplicado na entrada
- Straight-Through Estimator (STE) para backpropagation
- ~1.58 bits por peso (log₂(3) ≈ 1.58)

#### 2. MambaBlock (State Space Model)

O bloco Mamba implementa a equação de estado:

```
h(t) = Āh(t-1) + B̄x(t)    # Atualização de estado
y(t) = Ch(t)               # Saída
```

Onde:
- `Ā = exp(Δ·A)` - Matriz de transição discretizada
- `B̄ = Δ·B` - Matriz de entrada discretizada
- `Δ` - Passo de tempo (aprendido, dependente da entrada)

#### 3. Pipeline de Dados Bilíngue

```
┌────────────────┐     ┌────────────────┐
│   FineWeb-Edu  │     │   Wikipedia    │
│   (English)    │     │  (Portuguese)  │
│   sample-10BT  │     │      pt        │
└───────┬────────┘     └───────┬────────┘
        │                      │
        │    50%        50%    │
        └───────┬───────┬──────┘
                │       │
                ▼       ▼
        ┌───────────────────┐
        │  Interleaved      │
        │  Generator        │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │  Tokenization     │
        │  (tiktoken GPT-2) │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │  Batching         │
        │  (seq_len=2048)   │
        └───────────────────┘
```

---

## Requisitos

### Hardware Mínimo

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| GPU VRAM | 8GB | 16GB+ |
| RAM | 16GB | 32GB+ |
| Armazenamento | 50GB | 100GB+ (para datasets) |

### Software

```bash
# Python 3.9+
python --version

# CUDA 11.8+ (para GPU)
nvcc --version
```

---

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/lgenehr/ai-model-forge.git
cd ai-model-forge/bitnet-mamba-hybrid
```

### 2. Crie um ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate   # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. (Opcional) Instale mamba-ssm para kernels CUDA otimizados

```bash
# Requer CUDA toolkit instalado
pip install mamba-ssm
```

---

## Treinamento

### Uso Básico

```bash
python train_hybrid-mamba-bitnet.py
```

O script irá:
1. Carregar datasets automaticamente (streaming)
2. Detectar e resumir do último checkpoint (se existir)
3. Treinar até ~4B tokens
4. Salvar checkpoints a cada 1000 steps

### Configuração via Argumentos

```bash
python train_hybrid-mamba-bitnet.py \
    --d_model 768 \
    --n_layers 12 \
    --batch_size 8 \
    --grad_accum 4 \
    --max_seq_len 2048 \
    --lr 3e-4 \
    --max_tokens 4000000000
```

### Todos os Argumentos Disponíveis

| Argumento | Default | Descrição |
|-----------|---------|-----------|
| `--d_model` | 768 | Dimensão do modelo |
| `--n_layers` | 12 | Número de camadas Mamba |
| `--d_state` | 16 | Dimensão do estado SSM |
| `--d_conv` | 4 | Tamanho do kernel de convolução |
| `--expand` | 2 | Fator de expansão do bloco |
| `--dropout` | 0.1 | Taxa de dropout |
| `--batch_size` | 8 | Tamanho do batch |
| `--grad_accum` | 4 | Steps de acumulação de gradiente |
| `--max_seq_len` | 2048 | Comprimento máximo da sequência |
| `--max_tokens` | 4B | Total de tokens para treinar |
| `--lr` | 3e-4 | Learning rate |
| `--warmup_steps` | 2000 | Steps de warmup |
| `--weight_decay` | 0.1 | Weight decay |
| `--en_ratio` | 0.5 | Proporção de dados em inglês |
| `--pt_ratio` | 0.5 | Proporção de dados em português |
| `--output_dir` | ./ai-model-forge/bitnet-mamba-hybrid | Diretório de saída |
| `--seed` | 42 | Seed aleatória |
| `--no_amp` | False | Desabilitar mixed precision |

### Exemplos de Configuração

#### Modelo Pequeno (para testes)

```bash
python train_hybrid-mamba-bitnet.py \
    --d_model 256 \
    --n_layers 4 \
    --batch_size 16 \
    --max_tokens 100000000 \
    --max_seq_len 512
```

#### Modelo Médio (balanceado)

```bash
python train_hybrid-mamba-bitnet.py \
    --d_model 768 \
    --n_layers 12 \
    --batch_size 8 \
    --grad_accum 4
```

#### Modelo Grande (máximo para 16GB VRAM)

```bash
python train_hybrid-mamba-bitnet.py \
    --d_model 1024 \
    --n_layers 24 \
    --batch_size 4 \
    --grad_accum 8 \
    --max_seq_len 2048
```

### Monitoramento do Treinamento

#### Logs em tempo real

```bash
tail -f ai-model-forge/bitnet-mamba-hybrid/training.log
```

#### Histórico de loss (CSV)

```bash
# Visualizar últimas linhas
tail -20 ai-model-forge/bitnet-mamba-hybrid/loss_history.csv

# Plotar com Python
python -c "
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ai-model-forge/bitnet-mamba-hybrid/loss_history.csv')
plt.figure(figsize=(12, 4))
plt.plot(df['step'], df['loss'])
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('loss_plot.png')
print('Gráfico salvo em loss_plot.png')
"
```

### Resumindo Treinamento

O treinamento é automaticamente resumível. Basta executar o mesmo comando:

```bash
# Primeira execução - inicia do zero
python train_hybrid-mamba-bitnet.py

# Interrompido com Ctrl+C...

# Segunda execução - resume automaticamente
python train_hybrid-mamba-bitnet.py
# Output: "Resuming from checkpoint: checkpoints/checkpoint_00005000.pt"
```

---

## Weights & Biases

Integração completa com [Weights & Biases](https://wandb.ai) para rastreamento de experimentos, visualização de métricas e gerenciamento de artefatos.

### Configuração Inicial

1. Crie uma conta em [wandb.ai](https://wandb.ai)
2. Obtenha sua API key em [wandb.ai/authorize](https://wandb.ai/authorize)

### Habilitando W&B no Treinamento

#### Opção 1: Via argumento de linha de comando

```bash
python train_hybrid-mamba-bitnet.py \
    --wandb \
    --wandb_api_key "sua-api-key-aqui" \
    --wandb_project "meu-projeto" \
    --wandb_run_name "experimento-1"
```

#### Opção 2: Via variável de ambiente

```bash
export WANDB_API_KEY="sua-api-key-aqui"

python train_hybrid-mamba-bitnet.py \
    --wandb \
    --wandb_project "bitnet-mamba-hybrid"
```

#### Opção 3: Login persistente

```bash
# Login uma vez (salva credenciais localmente)
wandb login

# Depois, basta usar --wandb
python train_hybrid-mamba-bitnet.py --wandb
```

### Argumentos W&B

| Argumento | Default | Descrição |
|-----------|---------|-----------|
| `--wandb` | False | Habilita Weights & Biases |
| `--wandb_api_key` | None | API key (ou use WANDB_API_KEY env var) |
| `--wandb_project` | bitnet-mamba-hybrid | Nome do projeto no W&B |
| `--wandb_run_name` | None | Nome da run (auto-gerado se não fornecido) |
| `--wandb_entity` | None | Entidade (username ou nome do time) |

### Métricas Rastreadas

O treinamento automaticamente registra:

| Métrica | Descrição |
|---------|-----------|
| `train/loss` | Loss de treinamento |
| `train/learning_rate` | Learning rate atual |
| `train/tokens` | Total de tokens processados |
| `train/tokens_per_sec` | Throughput em tokens/segundo |
| `train/epoch` | Progresso em relação ao total de tokens |
| `train/best_loss` | Melhor loss alcançado |
| `final/*` | Métricas finais do treinamento |

### Visualização no Dashboard

Após iniciar o treinamento com W&B habilitado, você verá uma URL no terminal:

```
Weights & Biases initialized: https://wandb.ai/seu-usuario/bitnet-mamba-hybrid/runs/abc123
```

No dashboard você pode:
- Visualizar gráficos de loss em tempo real
- Comparar múltiplas runs
- Analisar gradientes do modelo
- Baixar checkpoints salvos como artefatos

### Exemplo Completo com W&B

```bash
python train_hybrid-mamba-bitnet.py \
    --d_model 768 \
    --n_layers 12 \
    --batch_size 8 \
    --grad_accum 4 \
    --max_tokens 1000000000 \
    --wandb \
    --wandb_api_key "sua-chave" \
    --wandb_project "bitnet-mamba-experiments" \
    --wandb_run_name "768d-12l-1B-tokens" \
    --wandb_entity "meu-time"
```

### Artefatos

Ao final do treinamento, o melhor modelo é automaticamente salvo como artefato no W&B, permitindo:
- Versionamento de modelos
- Download posterior
- Compartilhamento com o time

---

## Inferência

### Modo Interativo

```bash
python inference_hybrid.py
```

Isso abre um prompt interativo:

```
============================================================
BitNet-Mamba Hybrid Interactive Generation
============================================================
Commands:
  /quit or /exit  - Exit the program
  /temp <value>   - Set temperature (current: 0.8)
  /top_p <value>  - Set top-p (current: 0.95)
  /top_k <value>  - Set top-k (current: 0)
  /max <value>    - Set max tokens (current: 100)
  /greedy         - Use greedy decoding
  /sample         - Use sampling (default)
============================================================

Prompt> Once upon a time
```

### Geração com Prompt Único

```bash
# Geração básica
python inference_hybrid.py --prompt "The future of artificial intelligence"

# Com streaming (tokens aparecem um a um)
python inference_hybrid.py --prompt "Era uma vez" --stream

# Greedy decoding (determinístico)
python inference_hybrid.py --prompt "def fibonacci(n):" --greedy --max_tokens 200

# Ajustando temperatura
python inference_hybrid.py --prompt "Write a poem about" --temperature 1.2 --max_tokens 150
```

### Todos os Argumentos de Inferência

| Argumento | Default | Descrição |
|-----------|---------|-----------|
| `--checkpoint` | best_model.pt | Caminho do checkpoint |
| `--prompt` | None | Prompt (None = modo interativo) |
| `--max_tokens` | 100 | Máximo de tokens a gerar |
| `--temperature` | 0.8 | Temperatura de sampling |
| `--top_p` | 0.95 | Threshold de nucleus sampling |
| `--top_k` | 0 | Threshold de top-k (0 = desabilitado) |
| `--greedy` | False | Usar decodificação greedy |
| `--stream` | False | Mostrar tokens em tempo real |
| `--device` | cuda | Dispositivo (cuda/cpu) |
| `--dtype` | bfloat16 | Tipo de dados |

### Estratégias de Decodificação

#### 1. Greedy Decoding
Sempre escolhe o token mais provável. Determinístico, mas pode ser repetitivo.

```bash
python inference_hybrid.py --prompt "Hello" --greedy
```

#### 2. Nucleus (Top-p) Sampling
Amostra dos tokens que somam probabilidade `p`. Bom balanço qualidade/diversidade.

```bash
python inference_hybrid.py --prompt "Hello" --top_p 0.9 --temperature 0.8
```

#### 3. Top-k Sampling
Amostra apenas dos `k` tokens mais prováveis.

```bash
python inference_hybrid.py --prompt "Hello" --top_k 50 --temperature 1.0
```

#### 4. Combinado (Top-k + Top-p)
```bash
python inference_hybrid.py --prompt "Hello" --top_k 50 --top_p 0.9 --temperature 0.8
```

### Uso Programático

```python
from inference_hybrid import load_model, TextGenerator
import torch

# Carregar modelo
device = torch.device("cuda")
model, tokenizer, config = load_model(
    "ai-model-forge/bitnet-mamba-hybrid/best_model.pt",
    device=device
)

# Criar gerador
generator = TextGenerator(model, tokenizer, device)

# Gerar texto
output = generator.generate(
    prompt="The meaning of life is",
    max_tokens=100,
    temperature=0.8,
    top_p=0.95
)
print(output)

# Geração greedy
output_greedy = generator.generate_greedy(
    prompt="def quicksort(arr):",
    max_tokens=200
)
print(output_greedy)
```

---

## Configuração

### Configuração do Modelo (ModelConfig)

```python
@dataclass
class ModelConfig:
    vocab_size: int = 50304      # Vocabulário (múltiplo de 64)
    d_model: int = 768           # Dimensão do embedding
    n_layers: int = 12           # Número de camadas
    d_state: int = 16            # Dimensão do estado SSM
    d_conv: int = 4              # Kernel de convolução
    expand: int = 2              # Fator de expansão
    dropout: float = 0.1         # Dropout
    bias: bool = False           # Usar bias
    max_seq_len: int = 2048      # Comprimento máximo
```

### Configuração de Treinamento (TrainingConfig)

```python
@dataclass
class TrainingConfig:
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_seq_len: int = 2048
    max_tokens: int = 4_000_000_000
    warmup_steps: int = 2000
    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    dtype: str = "bfloat16"
    log_interval: int = 10
    checkpoint_interval: int = 1000
    en_ratio: float = 0.5
    pt_ratio: float = 0.5
```

### Estimativa de Parâmetros

| d_model | n_layers | Parâmetros | VRAM Est. |
|---------|----------|------------|-----------|
| 256 | 4 | ~10M | ~2GB |
| 512 | 8 | ~50M | ~4GB |
| 768 | 12 | ~125M | ~8GB |
| 1024 | 24 | ~350M | ~14GB |

---

## Estrutura do Projeto

```
bitnet-mamba-hybrid/
├── train_hybrid-mamba-bitnet.py   # Script de treinamento
├── inference_hybrid.py            # Script de inferência
├── requirements.txt               # Dependências
├── README.md                      # Esta documentação
│
├── checkpoints/                   # Checkpoints salvos
│   ├── checkpoint_00001000.pt
│   ├── checkpoint_00002000.pt
│   └── ...
│
├── best_model.pt                  # Melhor modelo
├── training.log                   # Log de treinamento
└── loss_history.csv               # Histórico de loss
```

### Formato do Checkpoint

```python
checkpoint = {
    'model_state_dict': ...,       # Pesos do modelo
    'optimizer_state_dict': ...,   # Estado do otimizador
    'scheduler_state_dict': ...,   # Estado do scheduler
    'global_step': int,            # Step atual
    'total_tokens': int,           # Tokens processados
    'best_loss': float,            # Melhor loss
    'model_config': dict,          # Configuração do modelo
    'train_config': dict           # Configuração de treino
}
```

---

## Referências

### Papers

1. **BitNet b1.58** - [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764)
   - Wang et al., 2024
   - Quantização ternária {-1, 0, 1}

2. **Mamba** - [Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
   - Gu & Dao, 2023
   - State Space Models com seleção

3. **RMSNorm** - [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
   - Zhang & Sennrich, 2019

### Datasets

- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) - Dados educacionais filtrados
- [Wikipedia PT](https://huggingface.co/datasets/wikipedia) - Wikipedia em português

### Bibliotecas

- [PyTorch](https://pytorch.org/) - Framework de deep learning
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/) - Streaming de dados
- [tiktoken](https://github.com/openai/tiktoken) - Tokenização BPE
- [einops](https://github.com/arogozhnikov/einops) - Operações de tensor

---

## Licença

MIT License - Veja o arquivo LICENSE para detalhes.

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduza o batch size
python train_hybrid-mamba-bitnet.py --batch_size 4 --grad_accum 8

# Ou reduza o modelo
python train_hybrid-mamba-bitnet.py --d_model 512 --n_layers 8
```

### Dataset não carrega

```bash
# Verifique conexão com internet
# O streaming requer conexão ativa

# Alternativa: use cache local
export HF_DATASETS_CACHE="/path/to/cache"
```

### tiktoken não disponível

```bash
pip install tiktoken

# Se falhar no Windows:
pip install tiktoken --no-build-isolation
```

### mamba-ssm não instala

```bash
# Requer CUDA toolkit
# Verifique: nvcc --version

# Se não tiver CUDA toolkit, o código usa implementação PyTorch pura
# (mais lento, mas funcional)
```
