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
from data_utils import prepare_hf_dataset
from huggingface_hub import snapshot_download

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
    # Em Docker: use volumes montados como /models_cache
    # Localmente: fallback para ~/models_cache
    parser.add_argument("--model_cache_dir", type=str, default="/models_cache", help="Diretório persistente para cache de modelos HF. Em Docker, configure volume mount para este caminho.")

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
    parser.add_argument("--llama_cpp_path", type=str, default="/opt/llama.cpp", help="Caminho para llama.cpp. Em Docker, pode usar volume mount ou deixar compilar localmente.")

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

def check_command_exists(command):
    """Verifica se um comando está disponível no sistema"""
    result = subprocess.run(f"which {command}", shell=True, capture_output=True)
    return result.returncode == 0

def get_writable_path(requested_path, default_fallback=""):
    """
    Verifica se um caminho é gravável. Tenta criar se necessário.
    Respeita volumes montados em containers Docker.
    Só faz fallback se absolutamente necessário.
    """
    expanded_path = os.path.expanduser(requested_path)
    parent_dir = os.path.dirname(expanded_path) or expanded_path
    
    # Se o caminho já existe e é gravável, usa
    if os.path.exists(expanded_path) and os.access(expanded_path, os.W_OK):
        logger.info(f"✅ Usando caminho existente: {expanded_path}")
        return expanded_path
    
    # Tenta criar o diretório (importante para volumes Docker montados)
    try:
        os.makedirs(expanded_path, exist_ok=True)
        logger.info(f"✅ Diretório criado/verificado: {expanded_path}")
        return expanded_path
    except PermissionError as e:
        logger.error(f"❌ Sem permissão para criar/acessar: {expanded_path}")
        logger.error(f"   Erro: {e}")
        
        # Só usa fallback se especificado
        if default_fallback:
            fallback = os.path.expanduser(f"~/{default_fallback}")
            logger.warning(f"⚠️ Usando fallback no home directory: {fallback}")
            try:
                os.makedirs(fallback, exist_ok=True)
                return fallback
            except Exception as fallback_error:
                logger.error(f"❌ Falha no fallback também: {fallback_error}")
                sys.exit(1)
        else:
            sys.exit(1)

def setup_llama_cpp_auto(base_path):
    """
    Verifica se llama.cpp existe. Se não, faz download e compila.
    Retorna os caminhos para convert_script e quantize_bin.
    """
    path = os.path.expanduser(base_path)
    
    # Verifica se o diretório pai é gravável
    parent_dir = os.path.dirname(path)
    if not os.access(parent_dir, os.W_OK):
        # Fall back to home directory
        fallback_path = os.path.expanduser("~/llama.cpp")
        logger.warning(f"⚠️ Sem permissão para escrever em {parent_dir}")
        logger.info(f"📁 Usando diretório alternativo: {fallback_path}")
        path = fallback_path
    
    convert_script = os.path.join(path, "convert_hf_to_gguf.py")
    quantize_bin = os.path.join(path, "build", "bin", "llama-quantize")

    # Se ambos os arquivos existem, retorna
    if os.path.isfile(convert_script) and os.path.isfile(quantize_bin):
        logger.info(f"✅ llama.cpp já configurado em: {path}")
        return convert_script, quantize_bin

    # Caso contrário, inicia setup automático
    logger.info(f"⬇️ llama.cpp não encontrado em {path}. Iniciando download e compilação...")

    # Verifica dependências de build
    required_tools = ["git", "cmake", "make", "gcc", "g++"]
    for tool in required_tools:
        if not check_command_exists(tool):
            logger.error(f"❌ {tool} não está instalado. Instale com: sudo apt-get install {tool}")
            sys.exit(1)

    logger.info("✅ Todas as dependências de build estão disponíveis.")

    # Cria diretório pai se não existir
    parent_dir = os.path.dirname(path)
    os.makedirs(parent_dir, exist_ok=True)

    # Clone do repositório
    if not os.path.exists(path):
        logger.info(f"🔗 Clonando llama.cpp de GitHub para {path}...")
        try:
            run_command(f"git clone https://github.com/ggerganov/llama.cpp {path}", "Clone do llama.cpp")
        except SystemExit:
            logger.error(f"❌ Falha ao clonar para {path}. Verifique permissões.")
            sys.exit(1)
    else:
        logger.info(f"📁 Diretório {path} existe, atualizando repositório...")
        try:
            run_command(f"cd {path} && git pull origin master", "Atualização do repositório llama.cpp")
        except SystemExit:
            logger.warning(f"⚠️ Não foi possível atualizar repositório em {path}")

    # Compilação
    logger.info("🔨 Compilando llama.cpp (isso pode levar alguns minutos)...")
    try:
        run_command(f"cd {path} && mkdir -p build && cd build && cmake .. && make -j{multiprocessing.cpu_count()}", 
                    "Compilação do llama.cpp")
    except SystemExit:
        logger.error(f"❌ Erro durante compilação de llama.cpp em {path}")
        sys.exit(1)

    # Verifica se a compilação foi bem-sucedida
    if not os.path.isfile(convert_script):
        logger.error(f"❌ Script não encontrado após compilação: {convert_script}")
        sys.exit(1)

    if not os.path.isfile(quantize_bin):
        logger.error(f"❌ Binário não encontrado após compilação: {quantize_bin}")
        sys.exit(1)

    logger.info(f"✨ llama.cpp compilado com sucesso em: {path}")
    return convert_script, quantize_bin

def get_latest_checkpoint(output_dir):
    """
    Encontra o checkpoint mais recente em output_dir.
    Funciona tanto em Docker quanto em WSL/Linux local.
    Retorna o caminho completo do checkpoint ou None.
    """
    if not os.path.isdir(output_dir):
        logger.warning(f"⚠️ Diretório de output não existe: {output_dir}")
        return None
    
    checkpoint_dirs = []
    try:
        for item in os.listdir(output_dir):
            if item.startswith("checkpoint-"):
                checkpoint_path = os.path.join(output_dir, item)
                if os.path.isdir(checkpoint_path):
                    # Extrai número do checkpoint
                    try:
                        step_num = int(item.split("-")[1])
                        checkpoint_dirs.append((step_num, checkpoint_path))
                    except (ValueError, IndexError):
                        continue
    except PermissionError as e:
        logger.warning(f"⚠️ Sem permissão para ler {output_dir}: {e}")
        return None
    
    if not checkpoint_dirs:
        logger.info(f"ℹ️ Nenhum checkpoint encontrado em {output_dir}")
        return None
    
    # Retorna o checkpoint com maior step_num
    latest_step, latest_checkpoint = max(checkpoint_dirs, key=lambda x: x[0])
    logger.info(f"✅ Checkpoint mais recente encontrado: {latest_checkpoint} (step {latest_step})")
    return latest_checkpoint

def validate_checkpoint(checkpoint_path):
    """
    Valida se um checkpoint é acessível e contém arquivos essenciais.
    """
    required_files = ["adapter_config.json", "adapter_model.bin"]
    
    for file in required_files:
        file_path = os.path.join(checkpoint_path, file)
        if not os.path.isfile(file_path):
            logger.warning(f"⚠️ Arquivo faltando no checkpoint: {file}")
            return False
    
    logger.info(f"✅ Checkpoint validado: {checkpoint_path}")
    return True

def train(args):
    # --- 0. Setup Inicial ---
    if wandb and args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.login(key=args.wandb_api_key)
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    elif wandb:
        os.environ["WANDB_MODE"] = "disabled"
        logger.warning("⚠️ WandB instalado, mas sem API Key. Desativando.")

    convert_script, quantize_bin = setup_llama_cpp_auto(args.llama_cpp_path)

    # Obtém caminhos com fallback para permissões
    model_cache_dir = get_writable_path(args.model_cache_dir, "models_cache")
    output_dir = get_writable_path(args.output_dir, "finetuned_models")
    
    # Cria diretórios necessários
    os.makedirs(model_cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"")
    logger.info(f"📁 ========== DIRETÓRIOS CONFIGURADOS ==========")
    logger.info(f"📁 model_cache_dir: {model_cache_dir}")
    logger.info(f"📁 output_dir: {output_dir}")
    logger.info(f"📁 llama.cpp path: {args.llama_cpp_path}")
    logger.info(f"📁 =============================================")
    logger.info(f"")
    
    # Configura variáveis de ambiente do HF para usar nossa pasta de cache
    os.environ["HF_HOME"] = model_cache_dir
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # --- 1. Carregar Modelo (Lógica de Cache Inteligente) ---
    logger.info(f"🔍 Verificando modelo base: {args.model_name}")
    
    # Define nome de pasta seguro (ex: unsloth/Qwen -> unsloth__Qwen)
    local_model_name = args.model_name.replace("/", "__")
    local_model_path = os.path.join(model_cache_dir, local_model_name)

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
        model_name = model_source,
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
        output_dir = output_dir,
        gradient_checkpointing = True,
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

    # --- 5. Treino com Checkpoint Automático ---
    logger.info("🔥 Iniciando Treinamento...")
    
    # Detecta e valida checkpoint se --resume_from_checkpoint foi passado
    resume_checkpoint = None
    if args.resume_from_checkpoint:
        latest_checkpoint = get_latest_checkpoint(output_dir)
        if latest_checkpoint and validate_checkpoint(latest_checkpoint):
            resume_checkpoint = latest_checkpoint
            logger.info(f"📂 Retomando do checkpoint: {resume_checkpoint}")
        else:
            logger.warning("⚠️ Flag --resume_from_checkpoint ativada, mas checkpoint não encontrado ou inválido. Iniciando do zero.")
    
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    logger.info("🏁 Treinamento concluído.")

    # --- 6. Salvar e Merge ---
    logger.info("💾 Salvando modelo mergeado (Safetensors 16-bit)...")
    merged_dir = os.path.join(output_dir, "merged_model_hf")
    
    # Limpeza de VRAM ANTES do merge (crítico!)
    torch.cuda.empty_cache()
    gc.collect()

    try:
        model.save_pretrained_merged(merged_dir, tokenizer, save_method = "merged_16bit")
        logger.info(f"✅ Modelo mergeado salvo em: {merged_dir}")
    except Exception as e:
        logger.error(f"❌ Erro ao salvar modelo mergeado: {e}")
        sys.exit(1)

    if wandb:
        wandb.finish()

    # --- 7. Conversão e Quantização (Llama.cpp) ---
    logger.info("⚙️ Iniciando conversão via llama.cpp...")
    
    # Validação crítica
    if not os.path.isdir(merged_dir) or not os.listdir(merged_dir):
        logger.error(f"❌ Diretório do modelo mergeado vazio ou inexistente: {merged_dir}")
        sys.exit(1)
    
    logger.info(f"✅ Validado modelo mergeado em: {merged_dir}")
    
    fp16_gguf = os.path.join(output_dir, "model_fp16.gguf")
    final_q4_gguf = os.path.join(output_dir, "modelo_final_q4_k_m.gguf")
    
    # Limpeza antes da conversão
    torch.cuda.empty_cache()
    gc.collect()
    
    # A) HF -> GGUF FP16
    try:
        run_command(f"python3 {convert_script} {merged_dir} --outfile {fp16_gguf} --outtype f16", "Conversão FP16")
        if not os.path.isfile(fp16_gguf):
            logger.error(f"❌ Conversão falhou: arquivo não gerado {fp16_gguf}")
            sys.exit(1)
        logger.info(f"✅ Conversão FP16 concluída: {fp16_gguf}")
    except SystemExit:
        logger.error("❌ Falha na conversão FP16")
        sys.exit(1)
    
    # B) FP16 -> Q4_K_M
    try:
        cpu_threads = multiprocessing.cpu_count()
        run_command(f"{quantize_bin} {fp16_gguf} {final_q4_gguf} q4_k_m {cpu_threads}", f"Quantização Q4 com {cpu_threads} threads")
        if not os.path.isfile(final_q4_gguf):
            logger.error(f"❌ Quantização falhou: arquivo não gerado {final_q4_gguf}")
            sys.exit(1)
        logger.info(f"✅ Quantização Q4 concluída: {final_q4_gguf}")
    except SystemExit:
        logger.error("❌ Falha na quantização Q4")
        sys.exit(1)

    # C) Limpeza
    if os.path.exists(fp16_gguf):
        try:
            os.remove(fp16_gguf)
            logger.info("🗑️ Arquivo temporário FP16 removido.")
        except Exception as e:
            logger.warning(f"⚠️ Não foi possível remover {fp16_gguf}: {e}")

    logger.info(f"✨ SUCESSO! Modelo final pronto em: {final_q4_gguf}")

if __name__ == "__main__":
    args = parse_args()
    train(args)
