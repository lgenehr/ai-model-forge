import argparse
import os
import torch
import gc
import subprocess
import sys
import logging
import multiprocessing
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from data_utils import prepare_hf_dataset  # Certifique-se de que este arquivo existe
from huggingface_hub import snapshot_download # Necessário: pip install huggingface_hub

# Tenta importar wandb
try:
    import wandb
except ImportError:
    wandb = None

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 2.5 14B with Smart Caching & Auto-Llama.cpp")
    
    # Model & Data
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-14B-Instruct")
    # 4096 é o limite seguro para 16GB de VRAM. Tentar 8192 pode causar OOM.
    parser.add_argument("--max_seq_length", type=int, default=4096) 
    parser.add_argument("--dataset_pattern", type=str, default="../dataset/*.jsonl")
    parser.add_argument("--dataset_num_proc", type=int, default=16, help="Cores para processar dataset")
    # Pasta onde o modelo será salvo fisicamente para evitar download repetido
    parser.add_argument("--model_cache_dir", type=str, default="/models_cache", help="Diretório persistente para cache de modelos HF")

    # LoRA Params (Configuração Agressiva)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # Training Params
    parser.add_argument("--batch_size", type=int, default=1) # Mantenha 1 para economizar VRAM
    parser.add_argument("--grad_accum_steps", type=int, default=16) # Compensa o batch size baixo
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--output_dir", type=str, default="financial_finetune_v3_agressivo")
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    
    # WandB
    parser.add_argument("--wandb_project", type=str, default="finetune-financeiro-qwen")
    parser.add_argument("--wandb_run_name", type=str, default="run-v3-agressivo")
    parser.add_argument("--wandb_api_key", type=str, default=None)

    # Llama.cpp Automation
    parser.add_argument("--llama_cpp_path", type=str, default="/opt/llama.cpp")

    return parser.parse_args()

def run_command(command, description):
    """Executa comandos shell e loga o output"""
    logger.info(f"🚀 [Executando]: {description}")
    try:
        subprocess.run(command, shell=True, check=True, executable='/bin/bash')
        logger.info(f"✅ [Sucesso]: {description}")
    except subprocess.CalledProcessError:
        logger.error(f"❌ [Erro] Falha em: {description}")
        sys.exit(1)

def setup_llama_cpp(base_path):
    """Verifica se as ferramentas do llama.cpp estão disponíveis"""
    path = os.path.expanduser(base_path)
    convert_script = os.path.join(path, "convert_hf_to_gguf.py")
    quantize_bin = os.path.join(path, "build", "bin", "llama-quantize")

    if not os.path.isfile(convert_script):
        logger.error(f"❌ Script não encontrado: {convert_script}")
        sys.exit(1)

    if not os.path.isfile(quantize_bin):
        logger.error(f"❌ Binário não encontrado: {quantize_bin}")
        sys.exit(1)

    return convert_script, quantize_bin

def train(args):
    # --- 0. Setup Inicial ---
    if wandb and args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.login(key=args.wandb_api_key)
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    elif wandb:
        os.environ["WANDB_MODE"] = "disabled"
        logger.warning("⚠️ WandB instalado, mas sem API Key. Desativando.")

    convert_script, quantize_bin = setup_llama_cpp(args.llama_cpp_path)

    # Cria diretórios necessários
    os.makedirs(args.model_cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configura variáveis de ambiente do HF para usar nossa pasta de cache
    os.environ["HF_HOME"] = args.model_cache_dir
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # --- 1. Carregar Modelo (Lógica de Cache Inteligente) ---
    logger.info(f"🔍 Verificando modelo base: {args.model_name}")
    
    # Define nome de pasta seguro (ex: unsloth/Qwen -> unsloth__Qwen)
    local_model_name = args.model_name.replace("/", "__")
    local_model_path = os.path.join(args.model_cache_dir, local_model_name)

    # Verifica se o modelo existe fisicamente na pasta mapeada
    if os.path.exists(local_model_path) and os.listdir(local_model_path):
        logger.info(f"✅ Modelo encontrado no cache local: {local_model_path}")
        model_source = local_model_path
    else:
        logger.info(f"⬇️ Modelo não encontrado localmente. Iniciando download para: {local_model_path}...")
        try:
            snapshot_download(
                repo_id=args.model_name,
                local_dir=local_model_path,
                local_dir_use_symlinks=False, # Importante: Baixa os arquivos reais, não links
                token=os.getenv("HF_TOKEN")
            )
            logger.info("✅ Download concluído com sucesso.")
            model_source = local_model_path
        except Exception as e:
            logger.error(f"❌ Erro crítico ao baixar modelo: {e}")
            sys.exit(1)

    logger.info(f"📦 Carregando modelo na memória a partir de: {model_source}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_source, # Usa sempre o caminho local agora
        max_seq_length = args.max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )

    # --- 2. Configurar LoRA ---
    logger.info("🔧 Adicionando adaptadores LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = args.lora_alpha,
        lora_dropout = args.lora_dropout,
        bias = "none",
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
    )

    # --- 3. Dataset ---
    logger.info(f"📚 Processando dataset: {args.dataset_pattern}")
    # Assume que prepare_hf_dataset retorna um Dataset do HF
    dataset = prepare_hf_dataset([args.dataset_pattern])
    
    # Split simples 90/10
    dataset_split = dataset.train_test_split(test_size=0.1, seed=3407)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    
    # Formatação de Chat
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen-2.5",
        mapping = {"role": "role", "content": "content", "user": "user", "assistant": "assistant"}
    )
    
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}
    
    logger.info(f"🔠 Tokenizando com {args.dataset_num_proc} threads...")
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True, num_proc=args.dataset_num_proc)
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True, num_proc=args.dataset_num_proc)

    # --- 4. Trainer Config ---
    training_args = TrainingArguments(
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum_steps,
        warmup_ratio = 0.05,
        num_train_epochs = args.epochs,
        learning_rate = args.learning_rate,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = args.logging_steps,
        optim = "paged_adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = args.output_dir,
        gradient_checkpointing = True, # Obrigatório para 16GB VRAM
        report_to = "wandb" if wandb and args.wandb_api_key else "none",
        run_name = args.wandb_run_name,
        eval_strategy = "steps",
        eval_steps = args.save_steps, 
        save_strategy = "steps",
        save_steps = args.save_steps,
        load_best_model_at_end = True,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        dataset_text_field = "text",
        max_seq_length = args.max_seq_length,
        dataset_num_proc = args.dataset_num_proc,
        packing = False, 
        args = training_args,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # --- 5. Treino ---
    logger.info("🔥 Iniciando Treinamento...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    logger.info("🏁 Treinamento concluído.")

    # --- 6. Salvar e Merge ---
    logger.info("💾 Salvando modelo mergeado (Safetensors 16-bit)...")
    merged_dir = os.path.join(args.output_dir, "merged_model_hf")
    
    # Limpeza de VRAM
    torch.cuda.empty_cache()
    gc.collect()

    # Salva o modelo mergeado (LoRA + Base)
    model.save_pretrained_merged(merged_dir, tokenizer, save_method = "merged_16bit")
    
    if wandb:
        wandb.finish()

    # --- 7. Conversão e Quantização (Llama.cpp) ---
    logger.info("⚙️ Iniciando conversão via llama.cpp...")
    
    fp16_gguf = os.path.join(args.output_dir, "model_fp16.gguf")
    final_q4_gguf = os.path.join(args.output_dir, "modelo_final_q4_k_m.gguf")
    
    # A) HF -> GGUF FP16
    run_command(f"python3 {convert_script} {merged_dir} --outfile {fp16_gguf} --outtype f16", "Conversão FP16")
    
    # B) FP16 -> Q4_K_M (Usando TODOS os núcleos do Ryzen)
    cpu_threads = multiprocessing.cpu_count()
    run_command(f"{quantize_bin} {fp16_gguf} {final_q4_gguf} q4_k_m {cpu_threads}", f"Quantização Q4 com {cpu_threads} threads")

    # C) Limpeza
    if os.path.exists(fp16_gguf):
        os.remove(fp16_gguf)
        logger.info("🗑️ Arquivo temporário FP16 removido.")

    logger.info(f"✨ SUCESSO! Modelo final pronto em: {final_q4_gguf}")

if __name__ == "__main__":
    args = parse_args()
    train(args)