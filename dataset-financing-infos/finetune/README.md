# Fine-Tuning Qwen 2.5 14B for Financial Prediction

Este diretório contém scripts e instruções para realizar o fine-tuning do modelo
**Qwen/Qwen2.5-14B-Instruct** utilizando **Unsloth** com **QLoRA**, aplicado a
datasets financeiros previamente gerados.

O objetivo é permitir treinamento e exportação do modelo de forma estável mesmo
em GPUs com **16 GB de VRAM**, como a RTX 4070 Ti Super.

---

## Requisitos

### Hardware
- **GPU:** NVIDIA com no mínimo 16 GB de VRAM  
  Exemplo: RTX 4070 Ti Super
- **CPU:** Processador multi-core recomendado  
  Exemplo: Ryzen 9

### Software
- **Sistema Operacional:** Linux (recomendado) ou WSL2
- **Python:** 3.10 ou superior
- **CUDA:** Compatível com a versão do PyTorch instalada
- **Ferramentas externas:**  
  - `llama.cpp` clonado e compilado (necessário para conversão e quantização GGUF)

---

## Setup

### 1. Instalação de dependências

```bash
pip install -r requirements.txt
Observação: certifique-se de que o PyTorch foi instalado com suporte a CUDA.

2. Preparação dos datasets
Os datasets devem estar no formato .jsonl

Cada linha representa um exemplo no formato instrução → resposta

Os arquivos devem estar no diretório dataset/ ou o caminho deve ser ajustado
no comando de treino

Exemplo de padrão esperado:

bash
Copiar código
dataset/dataset_*.jsonl
Uso
Treinamento (configuração segura para 16 GB de VRAM)
Para evitar erros de Out of Memory (OOM) em modelos de 14B parâmetros, utiliza-se:

batch_size = 1

grad_accum_steps elevado para simular batch efetivo maior

Comando padrão de treinamento
bash
Copiar código
python train.py \
    --model_name "unsloth/Qwen2.5-14B-Instruct" \
    --dataset_pattern "../dataset/dataset_*.jsonl" \
    --batch_size 1 \
    --grad_accum_steps 16 \
    --epochs 1 \
    --output_dir "financial_finetune_v2" \
    --lora_dropout 0
Retomar treinamento interrompido
Use este comando caso o treinamento tenha sido interrompido.
O diretório de saída deve ser o mesmo do treino original.

bash
Copiar código
python train.py \
    --model_name "unsloth/Qwen2.5-14B-Instruct" \
    --dataset_pattern "../dataset/dataset_*.jsonl" \
    --batch_size 1 \
    --grad_accum_steps 16 \
    --epochs 1 \
    --output_dir "financial_finetune_v2" \
    --resume_from_checkpoint
Conversão para GGUF (llama.cpp)
Após o fine-tuning, o modelo salvo em formato Hugging Face pode ser convertido
para GGUF, formato otimizado para inferência com llama.cpp, LM Studio,
Ollama e ferramentas similares.

Treino para melhorar o modelo para fatos e não somente estilo
python train.py \
    --model_name "unsloth/Qwen2.5-14B-Instruct" \  <-- Mantenha o original aqui
    --output_dir "financial_finetune_v3_agressivo" \ <-- Mude o nome da saída para não misturar
    --dataset_pattern "../dataset/dataset_*.jsonl" \
    --batch_size 1 \
    --grad_accum_steps 16 \
    --epochs 3 \              <-- Aumentado
    --lora_r 64 \             <-- Rank alto para detalhes
    --lora_alpha 128 \        <-- Alpha dobrado para força
    --lora_dropout 0.05       <-- Um pouco de dropout para inteligência

1. Conversão para GGUF em FP16
bash
Copiar código
python3 ~/llama.cpp/convert_hf_to_gguf.py ~/meu_modelo_temp \
  --outfile ~/meu_modelo_temp/modelo_fp16.gguf \
  --outtype f16
Explicação:

~/meu_modelo_temp: diretório contendo o modelo final em formato Hugging Face

--outfile: caminho e nome do arquivo GGUF de saída

--outtype f16: gera o modelo em FP16, preservando máxima qualidade antes
da quantização

Esse arquivo FP16 é a base recomendada para gerar versões quantizadas.

Quantização do modelo GGUF
A quantização reduz drasticamente o uso de memória e melhora a performance de
inferência, com impacto mínimo na qualidade.

Quantização para Q4_K_M
bash
Copiar código
~/llama.cpp/build/bin/llama-quantize \
  ~/meu_modelo_temp/modelo_fp16.gguf \
  ~/meu_modelo_temp/modelo_final_q4_k_m.gguf \
  q4_k_m \
  32
Explicação:

Primeiro argumento: modelo GGUF em FP16

Segundo argumento: arquivo GGUF quantizado de saída

q4_k_m: esquema de quantização balanceado entre qualidade e performance

32: número de threads utilizadas no processo de quantização

O formato Q4_K_M é amplamente recomendado para:

Inferência local

Uso em LM Studio e Ollama

Máximo custo-benefício entre qualidade e consumo de VRAM/RAM

Parâmetros principais do treinamento
--model_name
ID do modelo compatível com Unsloth
Padrão: unsloth/Qwen2.5-14B-Instruct

--dataset_pattern
Padrão glob para localizar os arquivos .jsonl

--batch_size
Batch por dispositivo
Recomendado manter em 1 para GPUs de 16 GB

--grad_accum_steps
Acumulação de gradientes para simular batch maior
Exemplo: 16

--lora_dropout
Dropout do LoRA
Valor recomendado: 0 para máximo desempenho com Unsloth

--output_dir
Diretório onde checkpoints e o modelo final são salvos

Observações finais
A conversão para GGUF deve ser feita após o fine-tuning final

Sempre gere primeiro a versão FP16, depois aplique quantização

Para produção e inferência local, o formato Q4_K_M é o mais recomendado

Para máxima qualidade, utilize o GGUF FP16 ou Q8