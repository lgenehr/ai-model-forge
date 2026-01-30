# Suporte a Múltiplos Formatos de Dataset

Este guia explica como o sistema de treinamento foi adaptado para usar os dados gerados pelo `dataset-generator`.

## 📋 Formatos Suportados

O `data_utils.py` agora detecta e converte automaticamente **3 formatos diferentes** de datasets:

### 1. **Formato Alpaca**
```json
{
  "instruction": "O que é inflação?",
  "input": "",
  "output": "Inflação é o aumento generalizado e contínuo dos preços...",
  "metadata": {
    "id": "fin_001",
    "source": "news",
    "topic": "financeiro",
    "quality_score": 0.85
  }
}
```

### 2. **Formato ShareGPT**
```json
{
  "conversations": [
    {"from": "human", "value": "Como funciona o Tesouro Direto?"},
    {"from": "gpt", "value": "O Tesouro Direto é um programa..."}
  ],
  "metadata": {
    "id": "fin_002",
    "source": "encyclopedia",
    "topic": "financeiro",
    "quality_score": 0.92
  }
}
```

### 3. **Formato ChatML**
```json
{
  "messages": [
    {"role": "system", "content": "Você é um especialista em finanças..."},
    {"role": "user", "content": "O que são fundos imobiliários?"},
    {"role": "assistant", "content": "Fundos imobiliários (FIIs) são..."}
  ],
  "metadata": {
    "id": "fin_003",
    "source": "academic",
    "topic": "financeiro",
    "quality_score": 0.88
  }
}
```

## 🔄 Normalização Automática

Todos os formatos são **automaticamente convertidos** para o formato ChatML durante o carregamento:

```python
# Formato original (qualquer um dos 3)
original_data = load_jsonl("dataset.jsonl")

# Após processamento por prepare_hf_dataset()
normalized_data = {
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

## 🎯 Prompts de Sistema Especializados

O sistema usa prompts de sistema diferentes baseados no tópico dos dados:

| Tópico | Prompt de Sistema |
|--------|-------------------|
| **financeiro** | "Você é um analista financeiro sênior especializado em mercados globais..." |
| **tecnologia** | "Você é um especialista em tecnologia que explica conceitos técnicos..." |
| **juridico** | "Você é um especialista em direito brasileiro..." |
| **saude** | "Você é um profissional de saúde que fornece informações médicas precisas..." |
| **ciencias** | "Você é um cientista que explica conceitos científicos..." |
| **outros** | "Você é um assistente útil e preciso..." (padrão) |

## 📁 Usando Dados do dataset-generator

### Passo 1: Gerar os Dados

```bash
cd ../dataset-generator

# Coletar dados
python -m src.main collect --sources news encyclopedia --topics financeiro --max-items 1000

# Processar dados (limpeza, deduplicação, etc.)
python -m src.main process

# Formatar para treinamento (gera os 3 formatos)
python -m src.main format --formats alpaca sharegpt chatml
```

Isso criará a seguinte estrutura:
```
dataset-generator/outputs/
├── raw/              # Dados brutos coletados
├── processed/        # Dados processados
└── final/            # Dados formatados para treinamento
    ├── alpaca/
    │   ├── train.jsonl
    │   ├── val.jsonl
    │   └── test.jsonl
    ├── sharegpt/
    │   ├── train.jsonl
    │   ├── val.jsonl
    │   └── test.jsonl
    └── chatml/
        ├── train.jsonl
        ├── val.jsonl
        └── test.jsonl
```

### Passo 2: Treinar o Modelo

```bash
cd ../finetune

# Usando formato ChatML (recomendado)
python train.py --dataset_pattern "../dataset-generator/outputs/final/chatml/train.jsonl"

# Ou usando formato Alpaca
python train.py --dataset_pattern "../dataset-generator/outputs/final/alpaca/train.jsonl"

# Ou usando formato ShareGPT
python train.py --dataset_pattern "../dataset-generator/outputs/final/sharegpt/train.jsonl"

# Ou usando TODOS os formatos (serão deduplicados automaticamente)
python train.py --dataset_pattern "../dataset-generator/outputs/final/**/train.jsonl"
```

## 🧪 Testando a Conversão de Formatos

Execute o script de teste para validar a conversão:

```bash
cd finetune
python test_data_formats.py
```

Esse script testa:
- ✅ Detecção automática de formato
- ✅ Conversão de Alpaca → ChatML
- ✅ Conversão de ShareGPT → ChatML
- ✅ Preservação de formato ChatML
- ✅ Preservação de metadata
- ✅ Deduplicação entre formatos

## 🔍 Deduplicação

O sistema deduplica automaticamente registros duplicados, mesmo que estejam em formatos diferentes:

```python
# Estes 3 registros têm o mesmo conteúdo em formatos diferentes
# O sistema detectará e manterá apenas 1

alpaca = {"instruction": "...", "output": "..."}
sharegpt = {"conversations": [{"from": "human", "value": "..."}, ...]}
chatml = {"messages": [{"role": "user", "content": "..."}, ...]}

# Resultado: apenas 1 registro será mantido
```

## 📊 Estatísticas de Carregamento

Durante o carregamento, o sistema exibe estatísticas úteis:

```
INFO - Loading datasets from 3 files
INFO - File patterns: ['../dataset-generator/outputs/final/chatml/train.jsonl']
INFO - Total unique records loaded: 1547
INFO - Format distribution: Alpaca=0, ShareGPT=0, ChatML=1547, Unknown=0
INFO - Prepared 1547 examples for training
INFO - Split do dataset: 1392 treino / 155 validação
```

## ⚙️ Configurações Importantes

### Padrão de Dataset

O padrão de dataset foi atualizado para apontar para os dados gerados pelo dataset-generator:

```python
# train.py - linha ~31
parser.add_argument(
    "--dataset_pattern",
    type=str,
    default="../dataset-generator/outputs/final/chatml/train.jsonl"
)
```

### Compatibilidade com Dados Antigos

O sistema mantém **compatibilidade total** com datasets antigos no formato Alpaca simples (sem metadata):

```json
{
  "instruction": "Explique Bitcoin",
  "input": "",
  "output": "Bitcoin é uma criptomoeda..."
}
```

## 🚀 Melhores Práticas

1. **Use o formato ChatML** quando possível - é o formato nativo do Qwen 2.5
2. **Preserve os metadados** - eles são úteis para análise e debugging
3. **Execute os testes** antes de treinar com novos dados
4. **Monitore as estatísticas** de carregamento para detectar problemas
5. **Use splits do dataset-generator** ao invés de re-dividir os dados

## 🐛 Troubleshooting

### Nenhum dado encontrado

```bash
ERROR - Nenhum dado encontrado! Verifique o caminho do dataset.
```

**Solução**: Gere os dados primeiro usando o dataset-generator conforme o Passo 1 acima.

### Formato desconhecido

```bash
WARNING - Unknown format type: unknown
```

**Solução**: Verifique se seus arquivos JSONL seguem um dos 3 formatos suportados.

### Muitos duplicados removidos

Se muitos registros estão sendo removidos como duplicados:
- Verifique se você não está carregando os mesmos dados múltiplas vezes
- Considere carregar apenas um formato por vez
- Use `--dataset_pattern` mais específico

## 📝 Exemplo Completo

```bash
# 1. Gerar dados
cd dataset-generator
python -m src.main collect --sources news --topics financeiro --max-items 500
python -m src.main process --quality-threshold 0.7
python -m src.main format --formats chatml

# 2. Testar conversão
cd ../finetune
python test_data_formats.py

# 3. Treinar modelo
python train.py \
  --dataset_pattern "../dataset-generator/outputs/final/chatml/train.jsonl" \
  --model_name "unsloth/Qwen2.5-14B-Instruct" \
  --epochs 3 \
  --batch_size 1 \
  --output_dir "financial_model_v1"
```

## 📚 Referências

- [Dataset Generator Documentation](../dataset-generator/README.md)
- [Qwen 2.5 Documentation](https://huggingface.co/Qwen)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
