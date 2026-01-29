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

    # Merge & GGUF Conversion
    parser.add_argument("--merge_only", action="store_true", help="Apenas mergea checkpoint existente com modelo base e converte para GGUF (sem treinar)")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Caminho específico para o checkpoint a ser mergeado. Se não informado, usa o checkpoint mais recente do output_dir")
    parser.add_argument("--convert_to_gguf", action="store_true", help="Converte modelo mergeado para GGUF Q4_K_M após treinamento ou merge")
    parser.add_argument("--skip_gguf_conversion", action="store_true", help="Pula conversão GGUF mesmo se --convert_to_gguf estiver ativo (útil para debug)")
    parser.add_argument("--gguf_quantization", type=str, default="q4_k_m", choices=["q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q8_0", "f16"], help="Método de quantização GGUF")

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
    Suporta tanto formato .bin quanto .safetensors.
    """
    # adapter_config.json é obrigatório
    config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if not os.path.isfile(config_path):
        logger.warning(f"⚠️ Arquivo faltando no checkpoint: adapter_config.json")
        return False

    # Verifica se existe adapter_model.bin OU adapter_model.safetensors
    bin_path = os.path.join(checkpoint_path, "adapter_model.bin")
    safetensors_path = os.path.join(checkpoint_path, "adapter_model.safetensors")

    if not os.path.isfile(bin_path) and not os.path.isfile(safetensors_path):
        logger.warning(f"⚠️ Arquivo de modelo faltando no checkpoint (nem .bin nem .safetensors)")
        return False

    logger.info(f"✅ Checkpoint validado: {checkpoint_path}")
    return True

def merge_and_convert_gguf(args, model=None, tokenizer=None, checkpoint_path=None):
    """
    Mergea um modelo LoRA com o modelo base e converte para GGUF.

    Pode ser usado:
    1. Standalone: carrega checkpoint e modelo base do zero (--merge_only)
    2. Pós-treino: usa modelo já carregado em memória

    Args:
        args: argumentos do CLI
        model: modelo já carregado (opcional, para pós-treino)
        tokenizer: tokenizer já carregado (opcional, para pós-treino)
        checkpoint_path: caminho do checkpoint (opcional, detecta automaticamente se não informado)

    Returns:
        str: caminho do arquivo GGUF final
    """
    # Setup llama.cpp
    convert_script, quantize_bin = setup_llama_cpp_auto(args.llama_cpp_path)

    # Obtém caminhos com fallback para permissões
    model_cache_dir = get_writable_path(args.model_cache_dir, "models_cache")
    output_dir = get_writable_path(args.output_dir, "finetuned_models")

    os.makedirs(model_cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Configura variáveis de ambiente do HF
    os.environ["HF_HOME"] = model_cache_dir
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # Determina o checkpoint a usar
    if checkpoint_path is None:
        checkpoint_path = args.checkpoint_path

    if checkpoint_path is None:
        # Detecta checkpoint mais recente
        checkpoint_path = get_latest_checkpoint(output_dir)
        if checkpoint_path is None:
            logger.error("❌ Nenhum checkpoint encontrado. Use --checkpoint_path para especificar.")
            sys.exit(1)

    if not validate_checkpoint(checkpoint_path):
        logger.error(f"❌ Checkpoint inválido: {checkpoint_path}")
        sys.exit(1)

    logger.info(f"📂 Usando checkpoint: {checkpoint_path}")

    # Se modelo não foi passado, carrega do zero
    if model is None or tokenizer is None:
        logger.info(f"🔍 Carregando modelo base: {args.model_name}")

        # Define nome de pasta seguro
        local_model_name = args.model_name.replace("/", "__")
        local_model_path = os.path.join(model_cache_dir, local_model_name)

        # Verifica cache local
        if os.path.exists(local_model_path) and os.listdir(local_model_path):
            logger.info(f"✅ Modelo encontrado no cache local: {local_model_path}")
            model_source = local_model_path
        else:
            logger.info(f"⬇️ Baixando modelo para: {local_model_path}...")
            try:
                snapshot_download(
                    repo_id=args.model_name,
                    local_dir=local_model_path,
                    local_dir_use_symlinks=False,
                    token=os.getenv("HF_TOKEN")
                )
                model_source = local_model_path
            except Exception as e:
                logger.error(f"❌ Erro ao baixar modelo: {e}")
                sys.exit(1)

        # Carrega modelo base
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_source,
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

        # Carrega adaptadores LoRA do checkpoint
        logger.info(f"🔧 Carregando adaptadores LoRA do checkpoint...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_path)
        logger.info("✅ Adaptadores LoRA carregados com sucesso.")

    # Merge do modelo
    logger.info("💾 Mergeando modelo (Safetensors 16-bit)...")
    merged_dir = os.path.join(output_dir, "merged_model_hf")

    torch.cuda.empty_cache()
    gc.collect()

    try:
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        logger.info(f"✅ Modelo mergeado salvo em: {merged_dir}")
    except Exception as e:
        logger.error(f"❌ Erro ao salvar modelo mergeado: {e}")
        sys.exit(1)

    # Skip GGUF se solicitado
    if args.skip_gguf_conversion:
        logger.info("⏭️ Conversão GGUF ignorada (--skip_gguf_conversion)")
        return merged_dir

    # Conversão GGUF
    logger.info("⚙️ Iniciando conversão via llama.cpp...")

    if not os.path.isdir(merged_dir) or not os.listdir(merged_dir):
        logger.error(f"❌ Diretório do modelo mergeado vazio: {merged_dir}")
        sys.exit(1)

    fp16_gguf = os.path.join(output_dir, "model_fp16.gguf")
    quantization = args.gguf_quantization
    final_gguf = os.path.join(output_dir, f"modelo_final_{quantization}.gguf")

    torch.cuda.empty_cache()
    gc.collect()

    # A) HF -> GGUF FP16
    try:
        run_command(f"python3 {convert_script} {merged_dir} --outfile {fp16_gguf} --outtype f16", "Conversão FP16")
        if not os.path.isfile(fp16_gguf):
            logger.error(f"❌ Conversão falhou: {fp16_gguf} não foi gerado")
            sys.exit(1)
        logger.info(f"✅ Conversão FP16 concluída: {fp16_gguf}")
    except SystemExit:
        logger.error("❌ Falha na conversão FP16")
        sys.exit(1)

    # B) FP16 -> Quantização escolhida
    if quantization != "f16":
        try:
            cpu_threads = multiprocessing.cpu_count()
            run_command(f"{quantize_bin} {fp16_gguf} {final_gguf} {quantization} {cpu_threads}",
                       f"Quantização {quantization.upper()} com {cpu_threads} threads")
            if not os.path.isfile(final_gguf):
                logger.error(f"❌ Quantização falhou: {final_gguf} não foi gerado")
                sys.exit(1)
            logger.info(f"✅ Quantização {quantization.upper()} concluída: {final_gguf}")
        except SystemExit:
            logger.error(f"❌ Falha na quantização {quantization.upper()}")
            sys.exit(1)

        # Remove FP16 temporário
        if os.path.exists(fp16_gguf):
            try:
                os.remove(fp16_gguf)
                logger.info("🗑️ Arquivo temporário FP16 removido.")
            except Exception as e:
                logger.warning(f"⚠️ Não foi possível remover {fp16_gguf}: {e}")
    else:
        final_gguf = fp16_gguf

    logger.info(f"✨ SUCESSO! Modelo GGUF pronto em: {final_gguf}")
    return final_gguf


def train(args):
    # --- 0. Setup Inicial ---
    if wandb and args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.login(key=args.wandb_api_key)
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    elif wandb:
        os.environ["WANDB_MODE"] = "disabled"
        logger.warning("⚠️ WandB instalado, mas sem API Key. Desativando.")

    # Verifica llama.cpp antecipadamente se conversão GGUF foi solicitada
    if args.convert_to_gguf and not args.skip_gguf_conversion:
        logger.info("🔍 Verificando llama.cpp (conversão GGUF ativada)...")
        setup_llama_cpp_auto(args.llama_cpp_path)

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
    if args.convert_to_gguf:
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

    if wandb:
        wandb.finish()

    # --- 6. Merge e Conversão GGUF (se solicitado) ---
    if args.convert_to_gguf:
        logger.info("🔄 Iniciando merge e conversão para GGUF...")
        final_gguf = merge_and_convert_gguf(args, model=model, tokenizer=tokenizer)
        logger.info(f"✨ SUCESSO! Modelo GGUF pronto em: {final_gguf}")
    else:
        # Apenas salva o modelo mergeado sem GGUF
        logger.info("💾 Salvando modelo mergeado (Safetensors 16-bit)...")
        merged_dir = os.path.join(output_dir, "merged_model_hf")

        torch.cuda.empty_cache()
        gc.collect()

        try:
            model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
            logger.info(f"✅ Modelo mergeado salvo em: {merged_dir}")
        except Exception as e:
            logger.error(f"❌ Erro ao salvar modelo mergeado: {e}")
            sys.exit(1)

        logger.info(f"✨ SUCESSO! Modelo final pronto em: {merged_dir}")
        logger.info("💡 Dica: Use --convert_to_gguf para converter para GGUF automaticamente")

if __name__ == "__main__":
    args = parse_args()

    if args.merge_only:
        # Modo merge-only: apenas mergea checkpoint existente e converte para GGUF
        logger.info("=" * 60)
        logger.info("🔀 MODO MERGE-ONLY ATIVADO")
        logger.info("=" * 60)
        logger.info("Este modo mergeia um checkpoint LoRA existente com o modelo base")
        logger.info("e opcionalmente converte para GGUF.")
        logger.info("")

        # Força conversão GGUF no merge_only (a menos que skip_gguf esteja ativo)
        if not args.skip_gguf_conversion:
            args.convert_to_gguf = True

        merge_and_convert_gguf(args)
    else:
        # Modo treinamento normal
        train(args)
