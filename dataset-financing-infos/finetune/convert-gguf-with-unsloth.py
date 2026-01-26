from unsloth import FastLanguageModel
import torch
import gc
import os

# --- CONFIGURAÇÃO ATUALIZADA ---
# O expanduser converte o "~" para "/home/seu_usuario" automaticamente
model_path = os.path.expanduser("~/meu_modelo_temp/model_gguf")

# Onde o arquivo final vai ser salvo (pode manter na pasta do projeto ou mudar se quiser)
output_path = "~/meu_modelo_temp/model_gguf_converted" 
# -------------------------------

print(f"📂 Carregando modelo do WSL (Alta Performance): {model_path}")

# Verificação de segurança para ver se a pasta existe mesmo
if not os.path.exists(model_path):
    print(f"❌ Erro: A pasta {model_path} não foi encontrada!")
    exit()

# 1. Carrega o modelo local em 4-bit (Leve na RAM)
# O Unsloth vai ler os safetensors direto da sua pasta rápida
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path, 
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True,
)

print("🧹 Limpando memória para conversão...")
torch.cuda.empty_cache()
gc.collect()

print("💾 Iniciando conversão para GGUF (q4_k_m)...")

# 2. Converte e Salva
model.save_pretrained_gguf(
    output_path, 
    tokenizer, 
    quantization_method = "q4_k_m"
)

print(f"✅ Sucesso! GGUF salvo em: {output_path}")