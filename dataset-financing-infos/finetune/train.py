import argparse
import os
import torch
import gc
import subprocess
import shutil
import multiprocessing
import sys
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth import is_bfloat16_supported
from data_utils import prepare_hf_dataset
import logging
import os

# Tenta importar wandb, se não tiver, avisa
try:
    import wandb
except ImportError:
    wandb = None

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 2.5 14B with Auto-Llama.cpp Setup")
    
    # Model & Data
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-14B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--dataset_pattern", type=str, default="../dataset/*.jsonl")
    parser.add_argument("--dataset_num_proc", type=int, default=16, help="Cores para processar dataset")

    # LoRA Params (Configuração Agressiva V3)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # Training Params
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--output_dir", type=str, default="financial_finetune_v3_agressivo")
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--logging_steps",type=int,default=10,help="Número de steps entre logs")
    
    # WandB
    parser.add_argument("--wandb_project", type=str, default="finetune-financeiro-qwen", help="Nome do projeto no WandB")
    parser.add_argument("--wandb_run_name", type=str, default="run-v3-agressivo", help="Nome da run")
    parser.add_argument("--wandb_api_key", type=str, default=None, help="API key do Weights & Biases")

    # Llama.cpp Automation
    parser.add_argument("--llama_cpp_path", type=str, default="/opt/llama.cpp", help="Caminho para instalar/usar o llama.cpp")

    return parser.parse_args()


def run_command(command, description):
    """Executa comandos shell e loga o output em tempo real"""
    logger.info(f"🚀 [Executando]: {description}")
    try:
        # Check=True lança erro se o comando falhar
        subprocess.run(command, shell=True, check=True, executable='/bin/bash')
        logger.info(f"✅ [Sucesso]: {description}")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ [Erro] Falha em: {description}")
        sys.exit(1)

def setup_llama_cpp(base_path):
    path = os.path.expanduser(base_path)

    convert_script = os.path.join(path, "convert_hf_to_gguf.py")
    quantize_bin = os.path.join(path, "build", "bin", "llama-quantize")

    if not os.path.isfile(convert_script):
        raise FileNotFoundError("convert_hf_to_gguf.py não encontrado no llama.cpp")

    if not os.path.isfile(quantize_bin):
        raise FileNotFoundError("llama-quantize não encontrado. Imagem está incorreta.")

    logger.info("✅ llama.cpp já disponível e compilado na imagem.")
    return convert_script, quantize_bin

def train(args):
    # 0. Setup Inicial (WandB e Llama.cpp)
    if wandb and args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.login(key=args.wandb_api_key)
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    elif wandb:
        logger.warning("⚠️ wandb instalado, mas sem --wandb_api_key. Não vou inicializar o wandb para evitar prompt interativo.")
        os.environ["WANDB_MODE"] = "disabled"
    else:
        logger.warning("⚠️ WandB não está instalado. Instale com `pip install wandb` para logs gráficos.")

    # Prepara o ambiente do llama.cpp ANTES de começar o treino para garantir que tudo funciona
    convert_script, quantize_bin = setup_llama_cpp(args.llama_cpp_path)

    # 1. Carregar Modelo
    logger.info(f"Carregando modelo {args.model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_seq_length,
        dtype = None, 
        load_in_4bit = True, 
    )

    # 2. Configurar LoRA (Agressivo V3)
    logger.info("Adicionando adaptadores LoRA...")
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

    # 3. Dataset
    logger.info(f"Processando dataset: {args.dataset_pattern}")
    dataset = prepare_hf_dataset([args.dataset_pattern])
    dataset_split = dataset.train_test_split(test_size=0.1, seed=3407)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    
    # Template Chat
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
    
    # Paralelismo na Tokenização (Usa o Ryzen)
    logger.info(f"Tokenizando com {args.dataset_num_proc} threads...")
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True, num_proc=args.dataset_num_proc)
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True, num_proc=args.dataset_num_proc)

    # 4. Trainer Config
    training_args = TrainingArguments(
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum_steps,
        warmup_ratio = 0.05,
        num_train_epochs = args.epochs,
        learning_rate = args.learning_rate,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = args.logging_steps,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = args.output_dir,
        gradient_checkpointing = True,
        # WandB Integration
        report_to = "wandb" if wandb else "none",
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

    # 5. Treino
    logger.info("🔥 Iniciando Treinamento...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    logger.info("🏁 Treinamento concluído.")

    # 6. Salvar e Merge
    logger.info("💾 Salvando modelo mergeado (Safetensors 16-bit)...")
    merged_dir = os.path.join(args.output_dir, "merged_model_hf")
    
    # Limpeza de VRAM antes do merge pesado
    torch.cuda.empty_cache()
    gc.collect()

    model.save_pretrained_merged(merged_dir, tokenizer, save_method = "merged_16bit")
    
    if wandb:
        wandb.finish()

    # 7. Conversão e Quantização (Llama.cpp)
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

    logger.info(f"✨ SUCESSO! Modelo pronto em: {final_q4_gguf}")

if __name__ == "__main__":
    args = parse_args()
    train(args)