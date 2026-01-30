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
from tui_utils import TrainingUI, console, format_size, format_time

try:
    import wandb
except ImportError:
    wandb = None

# Configuração de Logs (mantido para compatibilidade)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# UI Global
ui = TrainingUI()

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 2.5 14B with Smart Caching & Auto-Llama.cpp")
    
    # Model & Data
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-14B-Instruct")
    # 4096 é o limite seguro para 16GB de VRAM. Tentar 8192 pode causar OOM.
    parser.add_argument("--max_seq_length", type=int, default=4096)
    # Caminho padrão para dados gerados pelo dataset-generator
    # Formatos suportados: Alpaca, ShareGPT, ChatML (todos são automaticamente normalizados para ChatML)
    # Você pode usar um dos seguintes caminhos:
    #   - "../dataset-generator/outputs/final/chatml/train.jsonl" (ChatML format)
    #   - "../dataset-generator/outputs/final/alpaca/train.jsonl" (Alpaca format)
    #   - "../dataset-generator/outputs/final/sharegpt/train.jsonl" (ShareGPT format)
    #   - "../dataset-generator/outputs/final/**/train.jsonl" (todos os formatos)
    #   - "../dataset/*.jsonl" (legado - qualquer dataset antigo)
    parser.add_argument("--dataset_pattern", type=str, default="../dataset-generator/outputs/final/chatml/train.jsonl")
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
    """Executa comandos shell e loga o output com UI rica"""
    ui.print_step(f"Executando: {description}", "running")
    try:
        subprocess.run(command, shell=True, check=True, executable='/bin/bash')
        ui.print_step(f"Concluído: {description}", "success")
    except subprocess.CalledProcessError:
        ui.print_error(f"Falha em: {description}")
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
        ui.print_step(f"Usando caminho existente: {expanded_path}", "success")
        return expanded_path

    # Tenta criar o diretório (importante para volumes Docker montados)
    try:
        os.makedirs(expanded_path, exist_ok=True)
        ui.print_step(f"Diretório criado/verificado: {expanded_path}", "success")
        return expanded_path
    except PermissionError as e:
        ui.print_error(f"Sem permissão para criar/acessar: {expanded_path}\nErro: {e}")

        # Só usa fallback se especificado
        if default_fallback:
            fallback = os.path.expanduser(f"~/{default_fallback}")
            ui.print_warning(f"Usando fallback no home directory: {fallback}")
            try:
                os.makedirs(fallback, exist_ok=True)
                return fallback
            except Exception as fallback_error:
                ui.print_error(f"Falha no fallback também: {fallback_error}")
                sys.exit(1)
        else:
            sys.exit(1)

def setup_llama_cpp_auto(base_path):
    """
    Verifica se llama.cpp existe. Se não, faz download e compila.
    Retorna os caminhos para convert_script e quantize_bin.
    """
    ui.print_section("Setup llama.cpp", style="cyan")

    path = os.path.expanduser(base_path)

    # Verifica se o diretório pai é gravável
    parent_dir = os.path.dirname(path)
    if not os.access(parent_dir, os.W_OK):
        # Fall back to home directory
        fallback_path = os.path.expanduser("~/llama.cpp")
        ui.print_warning(f"Sem permissão para escrever em {parent_dir}\nUsando diretório alternativo: {fallback_path}")
        path = fallback_path

    convert_script = os.path.join(path, "convert_hf_to_gguf.py")
    quantize_bin = os.path.join(path, "build", "bin", "llama-quantize")

    # Se ambos os arquivos existem, retorna
    if os.path.isfile(convert_script) and os.path.isfile(quantize_bin):
        ui.print_step(f"llama.cpp já configurado em: {path}", "success")
        return convert_script, quantize_bin

    # Caso contrário, inicia setup automático
    ui.print_step(f"llama.cpp não encontrado em {path}", "download")
    ui.print_step("Iniciando download e compilação...", "running")

    # Verifica dependências de build
    required_tools = ["git", "cmake", "make", "gcc", "g++"]
    missing_tools = []
    for tool in required_tools:
        if not check_command_exists(tool):
            missing_tools.append(tool)

    if missing_tools:
        ui.print_error(f"Ferramentas faltando: {', '.join(missing_tools)}\nInstale com: sudo apt-get install {' '.join(missing_tools)}")
        sys.exit(1)

    ui.print_step("Todas as dependências de build estão disponíveis", "success")

    # Cria diretório pai se não existir
    parent_dir = os.path.dirname(path)
    os.makedirs(parent_dir, exist_ok=True)

    # Clone do repositório
    if not os.path.exists(path):
        ui.print_step(f"Clonando llama.cpp de GitHub para {path}...", "download")
        try:
            run_command(f"git clone https://github.com/ggerganov/llama.cpp {path}", "Clone do llama.cpp")
        except SystemExit:
            ui.print_error(f"Falha ao clonar para {path}. Verifique permissões.")
            sys.exit(1)
    else:
        ui.print_step(f"Diretório {path} existe, atualizando repositório...", "info")
        try:
            run_command(f"cd {path} && git pull origin master", "Atualização do repositório llama.cpp")
        except SystemExit:
            ui.print_warning(f"Não foi possível atualizar repositório em {path}")

    # Compilação
    ui.print_step("Compilando llama.cpp (isso pode levar alguns minutos)...", "running")

    with ui.create_simple_progress() as progress:
        task = progress.add_task("Compilando llama.cpp...", total=100)
        try:
            progress.update(task, advance=20)
            run_command(f"cd {path} && mkdir -p build && cd build && cmake ..", "CMake configuration")
            progress.update(task, advance=30)
            run_command(f"cd {path}/build && make -j{multiprocessing.cpu_count()}", "Compilação")
            progress.update(task, advance=50)
        except SystemExit:
            ui.print_error(f"Erro durante compilação de llama.cpp em {path}")
            sys.exit(1)

    # Verifica se a compilação foi bem-sucedida
    if not os.path.isfile(convert_script):
        ui.print_error(f"Script não encontrado após compilação: {convert_script}")
        sys.exit(1)

    if not os.path.isfile(quantize_bin):
        ui.print_error(f"Binário não encontrado após compilação: {quantize_bin}")
        sys.exit(1)

    ui.print_success(f"llama.cpp compilado com sucesso em: {path}")
    return convert_script, quantize_bin

def get_latest_checkpoint(output_dir):
    """
    Encontra o checkpoint mais recente em output_dir.
    Funciona tanto em Docker quanto em WSL/Linux local.
    Retorna o caminho completo do checkpoint ou None.
    """
    if not os.path.isdir(output_dir):
        ui.print_warning(f"Diretório de output não existe: {output_dir}")
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
        ui.print_warning(f"Sem permissão para ler {output_dir}: {e}")
        return None

    if not checkpoint_dirs:
        ui.print_step(f"Nenhum checkpoint encontrado em {output_dir}", "info")
        return None

    # Retorna o checkpoint com maior step_num
    latest_step, latest_checkpoint = max(checkpoint_dirs, key=lambda x: x[0])
    ui.print_step(f"Checkpoint mais recente: {latest_checkpoint} (step {latest_step})", "checkpoint")
    return latest_checkpoint

def validate_checkpoint(checkpoint_path):
    """
    Valida se um checkpoint é acessível e contém arquivos essenciais.
    Suporta tanto formato .bin quanto .safetensors.
    """
    # adapter_config.json é obrigatório
    config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if not os.path.isfile(config_path):
        ui.print_warning(f"Arquivo faltando no checkpoint: adapter_config.json")
        return False

    # Verifica se existe adapter_model.bin OU adapter_model.safetensors
    bin_path = os.path.join(checkpoint_path, "adapter_model.bin")
    safetensors_path = os.path.join(checkpoint_path, "adapter_model.safetensors")

    if not os.path.isfile(bin_path) and not os.path.isfile(safetensors_path):
        ui.print_warning(f"Arquivo de modelo faltando no checkpoint (nem .bin nem .safetensors)")
        return False

    ui.print_step(f"Checkpoint validado: {checkpoint_path}", "success")
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
    ui.print_header("MERGE E CONVERSÃO GGUF", "Mergeando modelo LoRA com modelo base")

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
            ui.print_error("Nenhum checkpoint encontrado. Use --checkpoint_path para especificar.")
            sys.exit(1)

    if not validate_checkpoint(checkpoint_path):
        ui.print_error(f"Checkpoint inválido: {checkpoint_path}")
        sys.exit(1)

    ui.print_step(f"Usando checkpoint: {checkpoint_path}", "checkpoint")

    # Se modelo não foi passado, carrega do zero
    if model is None or tokenizer is None:
        ui.print_step(f"Carregando modelo e adaptadores de: {checkpoint_path}", "download")

        # Carrega modelo e adaptadores via Unsloth (mais robusto para merge)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        ui.print_step("Modelo e adaptadores carregados com sucesso", "success")

    # Merge do modelo
    ui.print_divider("MERGE DO MODELO")
    ui.print_step("Mergeando modelo (Safetensors 16-bit)...", "merge")
    merged_dir = os.path.join(output_dir, "merged_model_hf")

    torch.cuda.empty_cache()
    gc.collect()

    try:
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        ui.print_step(f"Modelo mergeado salvo em: {merged_dir}", "save")
    except Exception as e:
        ui.print_error(f"Erro ao salvar modelo mergeado: {e}")
        sys.exit(1)

    # Skip GGUF se solicitado
    if args.skip_gguf_conversion:
        ui.print_step("Conversão GGUF ignorada (--skip_gguf_conversion)", "info")
        return merged_dir

    # Conversão GGUF
    ui.print_divider("CONVERSÃO GGUF")
    ui.print_step("Iniciando conversão via llama.cpp...", "convert")

    if not os.path.isdir(merged_dir) or not os.listdir(merged_dir):
        ui.print_error(f"Diretório do modelo mergeado vazio: {merged_dir}")
        sys.exit(1)

    fp16_gguf = os.path.join(output_dir, "model_fp16.gguf")
    quantization = args.gguf_quantization
    final_gguf = os.path.join(output_dir, f"modelo_final_{quantization}.gguf")

    torch.cuda.empty_cache()
    gc.collect()

    # A) HF -> GGUF FP16
    with ui.create_progress("Conversão GGUF") as progress:
        task = progress.add_task("Convertendo para FP16...", total=2)

        try:
            run_command(f"{sys.executable} {convert_script} {merged_dir} --outfile {fp16_gguf} --outtype f16", "Conversão FP16")
            if not os.path.isfile(fp16_gguf):
                ui.print_error(f"Conversão falhou: {fp16_gguf} não foi gerado")
                sys.exit(1)
            progress.update(task, advance=1)
            ui.print_step(f"Conversão FP16 concluída: {fp16_gguf}", "success")
        except SystemExit:
            ui.print_error("Falha na conversão FP16")
            sys.exit(1)

        # B) FP16 -> Quantização escolhida
        if quantization != "f16":
            try:
                cpu_threads = multiprocessing.cpu_count()
                run_command(f"{quantize_bin} {fp16_gguf} {final_gguf} {quantization} {cpu_threads}",
                           f"Quantização {quantization.upper()} com {cpu_threads} threads")
                if not os.path.isfile(final_gguf):
                    ui.print_error(f"Quantização falhou: {final_gguf} não foi gerado")
                    sys.exit(1)
                progress.update(task, advance=1)
                ui.print_step(f"Quantização {quantization.upper()} concluída: {final_gguf}", "success")
            except SystemExit:
                ui.print_error(f"Falha na quantização {quantization.upper()}")
                sys.exit(1)

            # Remove FP16 temporário
            if os.path.exists(fp16_gguf):
                try:
                    os.remove(fp16_gguf)
                    ui.print_step("Arquivo temporário FP16 removido", "info")
                except Exception as e:
                    ui.print_warning(f"Não foi possível remover {fp16_gguf}: {e}")
        else:
            final_gguf = fp16_gguf
            progress.update(task, advance=1)

    ui.print_success(f"Modelo GGUF pronto em:\n{final_gguf}")
    return final_gguf


def train(args):
    # Inicia o timer
    ui.start_timer()

    # Banner inicial
    ui.print_header(
        "🚀 FINE-TUNING QWEN 2.5 14B",
        "Sistema de treinamento com LoRA, Merge e Conversão GGUF"
    )

    # --- 0. Setup Inicial ---
    ui.print_section("Configuração Inicial", style="cyan")

    # Configuração do WandB
    if wandb and args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.login(key=args.wandb_api_key)
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
        ui.print_step(f"WandB ativado: {args.wandb_project}/{args.wandb_run_name}", "success")
    elif wandb:
        os.environ["WANDB_MODE"] = "disabled"
        ui.print_warning("WandB instalado, mas sem API Key. Desativando.")

    # Verifica llama.cpp antecipadamente se conversão GGUF foi solicitada
    if args.convert_to_gguf and not args.skip_gguf_conversion:
        ui.print_step("Verificando llama.cpp (conversão GGUF ativada)...", "info")
        setup_llama_cpp_auto(args.llama_cpp_path)

    # Obtém caminhos com fallback para permissões
    model_cache_dir = get_writable_path(args.model_cache_dir, "models_cache")
    output_dir = get_writable_path(args.output_dir, "finetuned_models")

    # Cria diretórios necessários
    os.makedirs(model_cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Tabela de diretórios configurados
    dirs_config = {
        "Cache de Modelos": model_cache_dir,
        "Diretório de Output": output_dir,
    }
    if args.convert_to_gguf:
        dirs_config["llama.cpp"] = args.llama_cpp_path

    ui.print_config_table(dirs_config, "Diretórios Configurados")

    # Configura variáveis de ambiente do HF para usar nossa pasta de cache
    os.environ["HF_HOME"] = model_cache_dir
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # --- 1. Carregar Modelo (Lógica de Cache Inteligente) ---
    ui.print_divider("CARREGAMENTO DO MODELO")
    ui.print_step(f"Verificando modelo base: {args.model_name}", "info")

    # Define nome de pasta seguro (ex: unsloth/Qwen -> unsloth__Qwen)
    local_model_name = args.model_name.replace("/", "__")
    local_model_path = os.path.join(model_cache_dir, local_model_name)

    # Verifica se o modelo existe fisicamente na pasta mapeada
    if os.path.exists(local_model_path) and os.listdir(local_model_path):
        ui.print_step(f"Modelo encontrado no cache local: {local_model_path}", "success")
        model_source = local_model_path
    else:
        ui.print_step(f"Modelo não encontrado localmente", "download")
        ui.print_step(f"Iniciando download para: {local_model_path}", "download")

        with ui.create_simple_progress() as progress:
            task = progress.add_task("Baixando modelo...", total=100)
            try:
                snapshot_download(
                    repo_id=args.model_name,
                    local_dir=local_model_path,
                    local_dir_use_symlinks=False,  # Importante: Baixa os arquivos reais, não links
                    token=os.getenv("HF_TOKEN")
                )
                progress.update(task, completed=100)
                ui.print_step("Download concluído com sucesso", "success")
                model_source = local_model_path
            except Exception as e:
                ui.print_error(f"Erro crítico ao baixar modelo: {e}")
                sys.exit(1)

    ui.print_step(f"Carregando modelo na memória a partir de: {model_source}", "running")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_source,
        max_seq_length = args.max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )

    ui.print_step("Modelo carregado com sucesso", "success")

    # --- 2. Configurar LoRA ---
    ui.print_divider("CONFIGURAÇÃO LORA")

    lora_config = {
        "LoRA Rank (r)": args.lora_r,
        "LoRA Alpha": args.lora_alpha,
        "LoRA Dropout": args.lora_dropout,
        "Target Modules": "7 módulos (q, k, v, o, gate, up, down)",
        "Gradient Checkpointing": "unsloth",
    }
    ui.print_config_table(lora_config, "Parâmetros LoRA")

    ui.print_step("Adicionando adaptadores LoRA...", "running")
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
    ui.print_step("Adaptadores LoRA configurados", "success")

    # --- 3. Dataset ---
    ui.print_divider("PROCESSAMENTO DO DATASET")
    ui.print_step(f"Carregando dataset: {args.dataset_pattern}", "running")
    ui.print_step("Formatos suportados: Alpaca, ShareGPT, ChatML (auto-detectados)", "info")

    # prepare_hf_dataset detecta automaticamente o formato e normaliza para ChatML
    dataset = prepare_hf_dataset([args.dataset_pattern])

    if len(dataset) == 0:
        error_msg = f"""Nenhum dado encontrado!

Caminho configurado: {args.dataset_pattern}

Dica: Use o dataset-generator para gerar dados primeiro:
  cd ../dataset-generator
  python -m src.main collect --sources all --topics financeiro
  python -m src.main process
  python -m src.main format --formats chatml"""
        ui.print_error(error_msg)
        sys.exit(1)

    ui.print_step(f"{len(dataset)} exemplos carregados com sucesso", "success")

    # Split simples 90/10
    dataset_split = dataset.train_test_split(test_size=0.1, seed=3407)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    dataset_info = {
        "Total de exemplos": len(dataset),
        "Dataset de treino": len(train_dataset),
        "Dataset de validação": len(eval_dataset),
        "Split ratio": "90/10",
    }
    ui.print_config_table(dataset_info, "Informações do Dataset")
    
    # Formatação de Chat
    ui.print_step("Configurando chat template Qwen 2.5...", "running")
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

    ui.print_step(f"Tokenizando com {args.dataset_num_proc} threads...", "running")
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True, num_proc=args.dataset_num_proc)
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True, num_proc=args.dataset_num_proc)
    ui.print_step("Tokenização concluída", "success")

    # --- 4. Trainer Config ---
    ui.print_divider("CONFIGURAÇÃO DO TREINAMENTO")

    training_config = {
        "Batch Size": args.batch_size,
        "Gradient Accumulation": args.grad_accum_steps,
        "Effective Batch Size": args.batch_size * args.grad_accum_steps,
        "Epochs": args.epochs,
        "Learning Rate": args.learning_rate,
        "Optimizer": "paged_adamw_8bit",
        "LR Scheduler": "cosine",
        "Warmup Ratio": "5%",
        "Max Seq Length": args.max_seq_length,
        "Precision": "BF16" if is_bfloat16_supported() else "FP16",
        "Logging Steps": args.logging_steps,
        "Save Steps": args.save_steps,
        "Early Stopping": "3 steps",
    }
    ui.print_config_table(training_config, "Parâmetros de Treinamento")

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

    ui.print_step("Criando trainer...", "running")
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
    ui.print_step("Trainer configurado", "success")

    # --- 5. Treino com Checkpoint Automático ---
    ui.print_divider("TREINAMENTO")

    # Detecta e valida checkpoint se --resume_from_checkpoint foi passado
    resume_checkpoint = None
    if args.resume_from_checkpoint:
        latest_checkpoint = get_latest_checkpoint(output_dir)
        if latest_checkpoint and validate_checkpoint(latest_checkpoint):
            resume_checkpoint = latest_checkpoint
            ui.print_step(f"Retomando do checkpoint: {resume_checkpoint}", "checkpoint")
        else:
            ui.print_warning("Flag --resume_from_checkpoint ativada, mas checkpoint não encontrado ou inválido. Iniciando do zero.")

    ui.print_header("🔥 INICIANDO TREINAMENTO", f"Tempo decorrido: {ui.get_elapsed_time()}")
    console.print()

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    ui.print_success(f"Treinamento concluído!\nTempo total: {ui.get_elapsed_time()}")

    if wandb:
        wandb.finish()

    # --- 6. Merge e Conversão GGUF (se solicitado) ---
    if args.convert_to_gguf:
        final_gguf = merge_and_convert_gguf(args, model=model, tokenizer=tokenizer)

        # Sumário final
        summary = {
            "Modelo": args.model_name,
            "Tempo total": ui.get_elapsed_time(),
            "Dataset": f"{len(dataset)} exemplos",
            "Epochs": args.epochs,
            "Modelo GGUF": final_gguf,
            "Quantização": args.gguf_quantization.upper(),
        }
        ui.print_summary(summary, "TREINAMENTO CONCLUÍDO COM SUCESSO")
    else:
        # Apenas salva o modelo mergeado sem GGUF
        ui.print_divider("SALVAMENTO DO MODELO")
        ui.print_step("Salvando modelo mergeado (Safetensors 16-bit)...", "save")
        merged_dir = os.path.join(output_dir, "merged_model_hf")

        torch.cuda.empty_cache()
        gc.collect()

        try:
            model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
            ui.print_step(f"Modelo mergeado salvo em: {merged_dir}", "success")
        except Exception as e:
            ui.print_error(f"Erro ao salvar modelo mergeado: {e}")
            sys.exit(1)

        # Sumário final
        summary = {
            "Modelo": args.model_name,
            "Tempo total": ui.get_elapsed_time(),
            "Dataset": f"{len(dataset)} exemplos",
            "Epochs": args.epochs,
            "Modelo final": merged_dir,
        }
        ui.print_summary(summary, "TREINAMENTO CONCLUÍDO COM SUCESSO")
        console.print("[dim]💡 Dica: Use --convert_to_gguf para converter para GGUF automaticamente[/dim]")
        console.print()

if __name__ == "__main__":
    args = parse_args()

    if args.merge_only:
        # Modo merge-only: apenas mergea checkpoint existente e converte para GGUF
        ui.print_header(
            "🔀 MODO MERGE-ONLY",
            "Mergeando checkpoint LoRA com modelo base e convertendo para GGUF"
        )

        merge_info = """Este modo mergeia um checkpoint LoRA existente com o modelo base
e opcionalmente converte para GGUF.

Nenhum treinamento será realizado."""
        ui.print_section("Informações do Modo", merge_info, style="yellow")

        # Força conversão GGUF no merge_only (a menos que skip_gguf esteja ativo)
        if not args.skip_gguf_conversion:
            args.convert_to_gguf = True

        merge_and_convert_gguf(args)
    else:
        # Modo treinamento normal
        train(args)
