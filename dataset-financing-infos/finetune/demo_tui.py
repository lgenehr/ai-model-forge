#!/usr/bin/env python3
"""
Demo da TUI - Demonstra as capacidades da interface rica do train.py
"""

import time
from tui_utils import TrainingUI, console

def demo_tui():
    """Demonstra as funcionalidades da TUI"""
    ui = TrainingUI()
    ui.start_timer()

    # Banner inicial
    ui.print_header(
        "🚀 DEMONSTRAÇÃO DA TUI",
        "Visualização das capacidades da interface rica"
    )

    # Seção 1: Configuração
    ui.print_section("Inicialização do Sistema", style="cyan")

    config = {
        "Modelo": "unsloth/Qwen2.5-14B-Instruct",
        "Batch Size": 1,
        "Gradient Accumulation": 16,
        "Epochs": 3,
        "Learning Rate": 0.0002,
        "LoRA Rank": 64,
        "LoRA Alpha": 128,
        "Max Seq Length": 4096,
        "Quantização GGUF": "Q4_K_M",
        "Conversão GGUF": True,
    }

    ui.print_config_table(config, "Configuração do Treinamento")
    time.sleep(1)

    # Seção 2: Steps com diferentes status
    ui.print_divider("DEMONSTRAÇÃO DE STEPS")

    ui.print_step("Verificando sistema...", "info")
    time.sleep(0.5)
    ui.print_step("Sistema verificado com sucesso", "success")
    time.sleep(0.5)
    ui.print_step("Atenção: Memória GPU em 85%", "warning")
    time.sleep(0.5)
    ui.print_step("Iniciando download do modelo...", "download")
    time.sleep(0.5)
    ui.print_step("Executando compilação...", "running")
    time.sleep(0.5)
    ui.print_step("Checkpoint salvo", "checkpoint")
    time.sleep(0.5)
    ui.print_step("Mergeando modelos...", "merge")
    time.sleep(0.5)
    ui.print_step("Convertendo para GGUF...", "convert")
    time.sleep(0.5)
    ui.print_step("Modelo salvo com sucesso", "save")
    time.sleep(1)

    # Seção 3: Barra de progresso
    ui.print_divider("DEMONSTRAÇÃO DE PROGRESSO")

    with ui.create_progress() as progress:
        task = progress.add_task("Processando dataset...", total=100)
        for i in range(100):
            time.sleep(0.02)
            progress.update(task, advance=1)

    ui.print_step("Dataset processado", "success")
    time.sleep(1)

    # Seção 4: Barra de progresso simples
    with ui.create_simple_progress() as progress:
        task = progress.add_task("Baixando arquivos...", total=100)
        for i in range(100):
            time.sleep(0.02)
            progress.update(task, advance=1)

    ui.print_step("Arquivos baixados", "success")
    time.sleep(1)

    # Seção 5: Métricas de treinamento
    ui.print_divider("MÉTRICAS DE TREINAMENTO")

    metrics = {
        "Loss": 0.234567,
        "Learning Rate": 0.0002,
        "Gradient Norm": 1.234,
        "Step": 1500,
        "Epoch": "2/3",
        "GPU Memory": "14.2 GB / 16 GB",
        "Training Speed": "1.23 steps/s",
    }

    ui.print_metrics_table(metrics)
    time.sleep(1)

    # Seção 6: Informações do dataset
    ui.print_divider("INFORMAÇÕES DO DATASET")

    dataset_info = {
        "Total de exemplos": 15234,
        "Dataset de treino": 13710,
        "Dataset de validação": 1524,
        "Split ratio": "90/10",
        "Formato": "ChatML",
        "Threads de processamento": 16,
    }

    ui.print_config_table(dataset_info, "Dataset")
    time.sleep(1)

    # Seção 7: Árvore de arquivos
    ui.print_divider("ARQUIVOS GERADOS")

    files = [
        "checkpoint-500/adapter_model.safetensors",
        "checkpoint-500/adapter_config.json",
        "checkpoint-1000/adapter_model.safetensors",
        "checkpoint-1000/adapter_config.json",
        "merged_model_hf/model.safetensors",
        "merged_model_hf/config.json",
        "modelo_final_q4_k_m.gguf",
    ]

    ui.print_file_tree("output_dir", files, "Estrutura de Arquivos")
    time.sleep(1)

    # Seção 8: Mensagens especiais
    ui.print_divider("MENSAGENS ESPECIAIS")

    ui.print_warning("Este é um aviso importante sobre uso de memória")
    time.sleep(1)

    console.print()
    ui.print_success("Operação concluída com sucesso!")
    time.sleep(1)

    # Seção 9: Sumário final
    console.print()
    summary = {
        "Modelo": "Qwen2.5-14B-Instruct",
        "Tempo total": ui.get_elapsed_time(),
        "Dataset": "15234 exemplos",
        "Epochs completadas": "3/3",
        "Loss final": 0.123456,
        "Modelo GGUF": "/output/modelo_final_q4_k_m.gguf",
        "Tamanho do arquivo": "8.2 GB",
        "Treinamento bem-sucedido": True,
    }

    ui.print_summary(summary, "DEMONSTRAÇÃO CONCLUÍDA")

    console.print("[bold cyan]✨ Todas as funcionalidades da TUI foram demonstradas![/bold cyan]")
    console.print()


if __name__ == "__main__":
    try:
        demo_tui()
    except KeyboardInterrupt:
        console.print("\n[yellow]Demonstração interrompida pelo usuário[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Erro na demonstração: {e}[/red]")
