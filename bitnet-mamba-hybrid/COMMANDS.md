# BitNet-Mamba Hybrid (204M) - Comandos

## Treino

### Treino completo (Phase 4 - recomendado)

```bash
nohup python train_hybrid-mamba-bitnet.py \
    --d_model 1280 --n_layers 14 \
    --batch_size 4 --grad_accum 8 \
    --lr 1e-4 --min_lr 1e-6 \
    --warmup_steps 500 \
    --weight_decay 0.03 \
    --max_grad_norm 0.5 \
    --max_tokens 8000000000 \
    --dropout 0.1 \
    --en_ratio 0.3 --pt_ratio 0.7 \
    --gradient_checkpointing \
    --output_dir model_204m \
    --data_dir data/tokenized \
    --weights_only \
    --wandb \
    --wandb_project "bitnet-mamba-hybrid" \
    --wandb_run_name "recovery-phase4-bitlinear-$(date +%Y%m%d-%H%M%S)" \
    > model_204m/training_resume_phase4.log 2>&1 &
```

### Treino do zero (sem checkpoint)

```bash
nohup python train_hybrid-mamba-bitnet.py \
    --d_model 1280 --n_layers 14 \
    --batch_size 4 --grad_accum 8 \
    --lr 1e-4 --min_lr 1e-6 \
    --warmup_steps 500 \
    --weight_decay 0.03 \
    --max_grad_norm 0.5 \
    --max_tokens 8000000000 \
    --dropout 0.1 \
    --en_ratio 0.3 --pt_ratio 0.7 \
    --gradient_checkpointing \
    --output_dir model_204m \
    --data_dir data/tokenized \
    > model_204m/training.log 2>&1 &
```

### Monitorar treino

```bash
# Acompanhar log em tempo real
tail -f model_204m/training_resume_phase4.log

# Ver ultima loss
grep "loss" model_204m/training_resume_phase4.log | tail -5

# Ver uso de GPU
nvidia-smi
```

---

## Inferencia

### Modo interativo

```bash
python inference_hybrid.py \
    --checkpoint model_204m/checkpoints/best_model.pt
```

### Prompt unico

```bash
python inference_hybrid.py \
    --checkpoint model_204m/checkpoints/best_model.pt \
    --prompt "O Brasil e um pais" \
    --max_tokens 200 \
    --temperature 0.8 \
    --top_p 0.95
```

### Greedy decoding (deterministico)

```bash
python inference_hybrid.py \
    --checkpoint model_204m/checkpoints/best_model.pt \
    --prompt "The meaning of life is" \
    --max_tokens 100 \
    --greedy
```

### Com streaming de tokens

```bash
python inference_hybrid.py \
    --checkpoint model_204m/checkpoints/best_model.pt \
    --prompt "Era uma vez" \
    --max_tokens 300 \
    --temperature 0.7 \
    --stream
```

---

## Parametros principais

### Treino

| Parametro | Default | Descricao |
|---|---|---|
| `--d_model` | 768 | Dimensao do modelo |
| `--n_layers` | 12 | Numero de camadas |
| `--batch_size` | 8 | Batch size |
| `--grad_accum` | 4 | Steps de acumulo de gradiente |
| `--lr` | 3e-4 | Learning rate |
| `--min_lr` | 1e-6 | Learning rate minimo |
| `--warmup_steps` | 2000 | Steps de warmup |
| `--weight_decay` | 0.1 | Weight decay |
| `--max_grad_norm` | 0.5 | Norma maxima do gradiente |
| `--max_tokens` | 4B | Total de tokens para treinar |
| `--dropout` | 0.1 | Taxa de dropout |
| `--en_ratio` | 0.5 | Proporcao de dados em ingles |
| `--pt_ratio` | 0.5 | Proporcao de dados em portugues |
| `--gradient_checkpointing` | off | Ativa checkpointing (economiza VRAM) |
| `--weights_only` | off | Resume apenas pesos (ignora estado do optimizer) |
| `--output_dir` | . | Diretorio de saida |
| `--data_dir` | ./data/tokenized | Diretorio dos dados tokenizados |

### Inferencia

| Parametro | Default | Descricao |
|---|---|---|
| `--checkpoint` | best_model.pt | Caminho do checkpoint |
| `--prompt` | None | Texto de entrada (None = modo interativo) |
| `--max_tokens` | 100 | Maximo de tokens a gerar |
| `--temperature` | 0.8 | Temperatura de amostragem |
| `--top_p` | 0.95 | Limiar de nucleus sampling |
| `--top_k` | 0 | Top-k sampling (0 = desativado) |
| `--greedy` | off | Decodificacao gulosa |
| `--stream` | off | Streaming de tokens |
| `--device` | cuda | Dispositivo (cuda/cpu) |
| `--dtype` | bfloat16 | Tipo de dado (bfloat16/float16/float32) |

---

## Scripts auxiliares

Os scripts `.sh` ficam em `scripts/` e podem ser executados de qualquer diretorio (usam `cd` automatico para a raiz do projeto):

```bash
# Executar de qualquer lugar
bash scripts/train_scaled_204m.sh
```

| Script | Descricao |
|---|---|
| `scripts/train_stable.sh` | Treino com config conservadora |
| `scripts/train_high_throughput.sh` | Treino otimizado para throughput |
| `scripts/train_memory_optimized.sh` | Treino otimizado para memoria |
| `scripts/train_ultra_memory_safe.sh` | Treino com uso minimo de VRAM |
| `scripts/train_scaled_204m.sh` | Config do modelo 204M |
| `scripts/train_scaled_204m_safe.sh` | Config 204M com protecoes extras |
| `scripts/train_recovery_204m.sh` | Recovery phase 1 |
| `scripts/train_recovery_phase2.sh` | Recovery phase 2 |
| `scripts/train_recovery_phase3.sh` | Recovery phase 3 |
