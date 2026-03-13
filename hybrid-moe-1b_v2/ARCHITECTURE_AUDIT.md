# Architecture Audit — Hybrid Transformer-Mamba-MoE (~820M)

**Date:** 2026-03-12
**Audited codebase:** `hybrid-moe-1b/` (model.py, train.py, inference.py, config.yaml)

---

## 1. Resumo Executivo

O projeto implementa um **decoder-only Language Model híbrido** de ~820M parâmetros que combina três famílias arquiteturais:

| Família | Papel no modelo | Camadas |
|---------|----------------|---------|
| **Mamba SSM** (Gu & Dao 2023) | Backbone principal — eficiência O(L) | 14 de 18 camadas |
| **Grouped-Query Attention** (LLaMA/Qwen-style) | Atenção global esparsa | 4 de 18 camadas |
| **Mixture-of-Experts** (Switch Transformer-style) | FFN esparso acoplado à atenção | 4 de 18 camadas (mesmas que atenção) |

**Parâmetros totais:** ~820M · **Ativos por forward:** ~560M (top-2 de 8 experts)
**Hardware alvo:** RTX 4070 Ti Super (16-17 GB VRAM)

A arquitetura é **tecnicamente sólida** e segue boas práticas para o tamanho. As principais oportunidades de melhoria estão no **despacho MoE ineficiente** (computa todos experts para todos tokens), **ausência de shared expert**, **geração sem KV cache real**, e **ausência de z-loss no router**.

---

## 2. Visão Geral da Arquitetura

```
Tipo: Decoder-only hybrid LM
Inspiração primária: Mamba + LLaMA + Switch Transformer
Tamanho: ~820M total, ~560M ativos/step

tokens [B, L]
  │
  ▼
┌───────────────────┐
│  Embedding         │  vocab=50304, d_model=2048, weight-tied com lm_head
│  (nn.Embedding)    │
└───────┬───────────┘
        │  [B, L, 2048]
        ▼
┌───────────────────┐
│  Layer 0: Mamba   │  RMSNorm → MambaMixer → residual
│  Layer 1: Mamba   │
│  Layer 2: Mamba   │
│  Layer 3: Attn+MoE│  RMSNorm → GQA(16Q/4KV) → residual → RMSNorm → MoE(8exp,top2) → residual
│  Layer 4: Mamba   │
│  Layer 5: Mamba   │
│  Layer 6: Mamba   │
│  Layer 7: Attn+MoE│
│  Layer 8: Mamba   │
│  Layer 9: Mamba   │
│  Layer 10: Mamba  │
│  Layer 11: Attn+MoE│
│  Layer 12: Mamba  │
│  ...              │
│  Layer 16: Mamba  │
│  Layer 17: Attn+MoE│
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  RMSNorm final    │
│  lm_head (Linear) │  → [B, L, 50304] logits
└───────────────────┘
```

---

## 3. Mapa dos Arquivos e Classes

| Arquivo | Classe / Função | Linhas | Papel |
|---------|----------------|--------|-------|
| `model.py` | `ModelConfig` | 48-117 | Dataclass com todos os hiperparâmetros |
| `model.py` | `RMSNorm` | 121-132 | Root-Mean-Square Layer Normalization |
| `model.py` | `precompute_freqs_cis` | 135-141 | Pre-computa frequências RoPE (complex64) |
| `model.py` | `apply_rotary_emb` | 144-157 | Aplica RoPE a Q e K |
| `model.py` | `_selective_scan_fallback` | 162-205 | Selective scan TorchScript (fallback) |
| `model.py` | `MambaMixer` | 210-316 | Core Mamba SSM (in_proj → conv1d → SSM → out_proj) |
| `model.py` | `MambaBlock` | 319-329 | Pre-norm wrapper: RMSNorm → MambaMixer → residual |
| `model.py` | `GroupedQueryAttention` | 334-401 | GQA com RoPE e SDPA |
| `model.py` | `SwiGLUExpert` | 406-416 | Expert individual SwiGLU FFN |
| `model.py` | `MoELayer` | 419-486 | Router + N experts + aux loss |
| `model.py` | `AttentionMoEBlock` | 489-520 | Bloco composto: GQA + MoE com pre-norm |
| `model.py` | `HybridMoEModel` | 525-731 | Modelo completo com forward, generate, init |
| `train.py` | `main()` | 508-914 | Training loop principal |
| `train.py` | `build_optimizer` | 185-221 | AdamW com parameter groups |
| `train.py` | `get_lr` | 171-180 | Cosine schedule com linear warmup |
| `train.py` | `evaluate` | 361-420 | Validação determinística |
| `train.py` | `save_checkpoint` | 226-268 | Checkpoint com metadata JSON sidecar |
| `inference.py` | `generate` | 98-187 | Geração com streaming |
| `config.yaml` | — | 1-114 | Configuração YAML completa |
| `data_loader.py` | `PreTokenizedDataset` | 94-321 | Dataset memory-mapped com split train/val |

---

## 4. Explicação Detalhada dos Componentes

### 4.1 Embedding (`model.py:555`)

- **Tipo:** `nn.Embedding(50304, 2048)`
- **Classificação:** GPT-2 / LLaMA style
- **Vocab:** 50304 (GPT-2 BPE padded to multiple of 64 for tensor core alignment)
- **Weight tying:** `lm_head.weight = embed_tokens.weight` (model.py:575) — economia de ~103M params
- **Análise:** Boa prática. LLaMA, GPT-2, Qwen todos usam weight tying.

### 4.2 RMSNorm (`model.py:121-132`)

- **Tipo:** Root-Mean-Square Layer Normalization sem bias
- **Classificação:** LLaMA / Qwen style (moderno)
- **Implementação:** Cast para float32 internamente para estabilidade numérica, retorna ao dtype original
- **Análise:** Excelente escolha. Mais eficiente que LayerNorm e padrão em todos LLMs modernos.

### 4.3 RoPE — Rotary Position Embeddings (`model.py:135-157`)

- **Tipo:** RoPE via números complexos (complex64)
- **Classificação:** LLaMA / GPT-NeoX style
- **Base:** 10000.0 (padrão RoPE original)
- **Pre-computação:** `precompute_freqs_cis` gera `torch.polar` de frequências
- **Aplicação:** `apply_rotary_emb` converte Q/K para complex, multiplica por freqs, reconverte
- **Buffer:** Registrado como buffer persistente em `freqs_cis` (model.py:558-559)
- **Análise:** Implementação correta e eficiente. A base 10000 limita a extrapolação para contextos > 2048. Para sequências mais longas, considerar NTK-aware scaling ou YaRN.

### 4.4 Mamba SSM (`model.py:210-316`)

- **Tipo:** Mamba-1 (Gu & Dao 2023) — Selective State Space Model
- **Classificação:** Mamba original
- **Componentes internos:**
  - `in_proj`: Linear(d_model → 2×d_inner) — projeta x e gating z simultaneamente
  - `conv1d`: Causal depthwise conv (kernel=4, groups=d_inner)
  - `x_proj`: Linear(d_inner → dt_rank + 2×d_state) — projeta dt, B, C do SSM
  - `dt_proj`: Linear(dt_rank → d_inner) com bias (dt_bias para softplus)
  - `A_log`: Parâmetro log-space [d_inner, d_state] — inicializado como log(arange(1, N+1))
  - `D`: Parâmetro skip connection [d_inner]
  - `out_proj`: Linear(d_inner → d_model)
- **Selective Scan:**
  - CUDA path: `mamba_ssm.ops.selective_scan_fn` (channels-first format)
  - Fallback: `_selective_scan_fallback` via TorchScript (sequential loop)
- **Dimensões:** d_inner = 2 × 2048 = 4096, d_state = 16, d_conv = 4, dt_rank = 128
- **Análise:** Implementação fiel ao paper original do Mamba. O d_state=16 é o padrão do Mamba-1. O Mamba-2 usa d_state muito maior (64-128) com algoritmo SSD mais eficiente.

### 4.5 Grouped-Query Attention (`model.py:334-401`)

- **Tipo:** GQA com 16 Q heads e 4 KV heads (ratio 4:1)
- **Classificação:** LLaMA-2 / Qwen-2 style
- **RoPE:** Aplicado a Q e K antes do SDPA
- **KV expansion:** `repeat_interleave(n_rep, dim=2)` para igualar Q heads
- **Backend:** `F.scaled_dot_product_attention` — usa FlashAttention/cuDNN no Ampere+
- **Causal masking:** `is_causal=True` durante treino (sem mask explícita)
- **KV cache:** Suportado na interface mas **não utilizado na geração** (ver seção 7)
- **Análise:** Implementação correta e moderna. GQA 4:1 é o padrão LLaMA-2 70B. Para ~820M, MHA ou GQA 2:1 também seriam válidos. Falta QK-Norm para estabilidade (usado em Qwen-2.5).

### 4.6 MoE Layer (`model.py:419-486`)

- **Tipo:** Token-choice top-k routing com SwiGLU experts
- **Classificação:** Switch Transformer / GShard style (com SwiGLU de LLaMA)
- **Router:** Linear(d_model → n_experts) → softmax → topk
- **Experts:** 8 × SwiGLUExpert(d_model=2048, intermediate=1024)
- **Top-k:** 2 (cada token usa 2 dos 8 experts)
- **Renormalização:** `top_scores / top_scores.sum(-1)` — renormalized routing
- **Aux loss:** Switch Transformer eq. 4: `coeff × E × sum(f_i × p_i)` com coeff=0.01
- **Despacho:** **DENSE** — computa **todos** 8 experts para **todos** tokens, depois seleciona top-2

#### Problemas críticos do MoE:
1. **Despacho denso** (model.py:475-478): `torch.stack([expert(x) for e in range(n_experts)])` — computa 8× o trabalho necessário. Apenas 2 experts são usados por token.
2. **intermediate=1024** é 0.5× d_model — muito pequeno para experts. Padrão moderno é 2-4× d_model.
3. **Sem shared expert**: DeepSeek-V2/V3 usam um expert compartilhado que processa todos tokens.
4. **Sem z-loss**: Apenas aux_loss (balanceamento). Z-loss estabiliza logits do router.
5. **Sem expert capacity**: Não limita tokens por expert, risco de desequilíbrio.

### 4.7 AttentionMoEBlock (`model.py:489-520`)

- **Tipo:** Bloco composto GQA + MoE com double pre-norm
- **Classificação:** LLaMA / Mixtral style
- **Estrutura:**
  ```
  x → RMSNorm → GQA → + x (residual)
  x → RMSNorm → MoE → + x (residual)
  ```
- **Análise:** Estrutura correta. Segue o padrão Mixtral (attention + MoE-FFN).

### 4.8 Weight Initialization (`model.py:578-582`)

- **Base:** Normal(0, 0.02) para Linear e Embedding
- **Residual scaling:** Projeções de saída (`out_proj`, `o_proj`, `down_proj`) escalonadas por `0.02 / sqrt(2 × n_layers)`
- **Classificação:** GPT-2 style com residual scaling
- **Análise:** O fator `sqrt(2 × n_layers)` é conservador mas razoável. GPT-2 usa `1/sqrt(2*n_layers)`, LLaMA usa variantes. A escala é aplicada corretamente às projeções de saída dos sub-blocos.

### 4.9 Optimizer (`train.py:185-221`)

- **Tipo:** AdamW com fused backend
- **Parameter groups:** 2 grupos — com weight_decay (matrizes) e sem weight_decay (bias, norms, A_log, D, embeddings)
- **Hiperparâmetros:** lr=3e-4, β1=0.9, β2=0.95, ε=1e-8, wd=0.1
- **Classificação:** Padrão LLaMA / Chinchilla
- **Análise:** Configuração sólida. β2=0.95 é mais conservador que 0.999 (bom para estabilidade). Fused AdamW é ótimo para performance.

### 4.10 LR Schedule (`train.py:171-180`)

- **Tipo:** Linear warmup → Cosine decay → min_lr
- **Warmup:** 2000 steps
- **Peak LR:** 3e-4
- **Min LR:** 3e-5 (10% do peak)
- **Classificação:** Chinchilla / LLaMA style
- **Análise:** Padrão e correto. Alternativa moderna: WSD (Warmup-Stable-Decay) usado em MiniCPM.

### 4.11 Mixed Precision (`train.py:703`)

- **GradScaler:** Habilitado quando dtype=float16; **desabilitado para bfloat16** implicitamente (`enabled=(dtype == torch.float16)`)
- **Autocast:** `torch.autocast(device_type, dtype)` para forward/backward
- **Análise:** Correto — GradScaler não é necessário para bf16.

### 4.12 Gradient Checkpointing (`train.py:637-660`)

- **Tipo:** Block-level checkpointing via `torch.utils.checkpoint.checkpoint`
- **use_reentrant:** True (necessário para compatibilidade com mamba-ssm CUDA extension)
- **Análise:** Correto. Economiza ~10× memória de ativação ao custo de ~33% mais compute.

---

## 5. Fluxo Forward Real do Modelo

```python
# model.py:607-679 — HybridMoEModel.forward()

1. input_ids [B, L] (int64)
2. x = embed_tokens(input_ids)                          # [B, L, 2048]
3. freqs = freqs_cis[:L]                                # [L, 64] complex64

4. for i, layer in enumerate(self.layers):               # 18 layers
     if i in {3, 7, 11, 17}:  # AttentionMoEBlock
       a) x_norm = RMSNorm(x)                            # [B, L, 2048]
       b) q = q_proj(x_norm)  → [B, L, 16, 128]
          k = k_proj(x_norm)  → [B, L, 4, 128]
          v = v_proj(x_norm)  → [B, L, 4, 128]
       c) q, k = apply_rotary_emb(q, k, freqs)           # RoPE
       d) k = k.repeat_interleave(4, dim=2)               # [B, L, 16, 128]
          v = v.repeat_interleave(4, dim=2)
       e) attn_out = SDPA(q, k, v, is_causal=True)       # [B, L, 16, 128]
       f) attn_out = o_proj(merge_heads(attn_out))        # [B, L, 2048]
       g) x = x + attn_out                                # residual

       h) x_norm = RMSNorm(x)                             # [B, L, 2048]
       i) gate_logits = gate(x_norm.reshape(N, D))        # [N, 8]
       j) scores = softmax(gate_logits)                    # [N, 8]
       k) top_scores, top_idx = topk(scores, k=2)         # [N, 2]
       l) expert_outs = stack([expert_i(x) for i in 0..7]) # [N, 8, 2048] ← DENSE
       m) selected = gather(expert_outs, top_idx)          # [N, 2, 2048]
       n) out = (selected × top_scores).sum(dim=1)         # [N, 2048]
       o) x = x + out.reshape(B, L, D)                    # residual
       p) total_aux_loss += aux_loss

     else:  # MambaBlock
       a) x_norm = RMSNorm(x)                              # [B, L, 2048]
       b) xz = in_proj(x_norm)                             # [B, L, 8192]
          x_in, z = chunk(xz, 2)                           # each [B, L, 4096]
       c) x_conv = conv1d(x_in.transpose) → SiLU          # [B, L, 4096]
       d) dt, B_ssm, C_ssm = x_proj(x_conv) → dt_proj    # SSM params
       e) A = -exp(A_log)                                   # [4096, 16]
       f) y = selective_scan(x_conv, dt, A, B, C, D, z)    # [B, L, 4096]
       g) y = out_proj(y)                                   # [B, L, 2048]
       h) x = x + y                                        # residual

5. x = RMSNorm(x)                                          # [B, L, 2048]
6. logits = lm_head(x)                                     # [B, L, 50304]
7. if targets: ce_loss = cross_entropy(logits, targets)
8. total_loss = ce_loss + total_aux_loss
```

---

## 6. Classificação Arquitetural

### Tipo exato da arquitetura atual

> **Decoder-only hybrid LM com backbone Mamba-1 SSM, atenção GQA esparsa estilo LLaMA-2, e MoE SwiGLU estilo Switch Transformer, usando pre-norm RMSNorm em todos os blocos.**

### Comparação com arquiteturas conhecidas

| Aspecto | Este modelo | GPT-2 | LLaMA-2 | Qwen-2.5 | Mixtral | Mamba | DeepSeek-V2 |
|---------|-------------|-------|---------|-----------|---------|-------|-------------|
| Tipo | Decoder-only hybrid | Decoder-only Transformer | Decoder-only Transformer | Decoder-only Transformer | Decoder-only MoE Transformer | Pure SSM | Decoder-only MoE Transformer |
| Normalização | RMSNorm pre-norm | LayerNorm post-norm | RMSNorm pre-norm | RMSNorm pre-norm | RMSNorm pre-norm | RMSNorm pre-norm | RMSNorm pre-norm |
| Atenção | GQA 16Q/4KV | MHA | GQA | GQA | GQA | Nenhuma | MLA (Multi-head Latent Attention) |
| Posicional | RoPE complex | Absolute learned | RoPE | RoPE | RoPE | Nenhum (causal conv) | RoPE |
| FFN | SwiGLU (MoE) | GELU FFN | SwiGLU | SwiGLU | SwiGLU (MoE) | Nenhum (gating interno) | SwiGLU (MoE) |
| SSM | Mamba-1 | Não | Não | Não | Não | Mamba-1/2 | Não |
| MoE | top-2/8 experts | Não | Não | Não (base) | top-2/8 experts | Não | top-2/routed + shared |
| Aux loss | Switch eq.4 | N/A | N/A | N/A | Switch + z-loss | N/A | Balanced + z-loss |
| Weight tying | Sim | Sim | Sim | Sim | Sim | Sim | Sim |

### Em que se parece com cada arquitetura

- **GPT-2:** Weight tying, vocab arredondado, init std=0.02
- **LLaMA-2:** RMSNorm pre-norm, GQA, RoPE complex, SwiGLU, AdamW β2=0.95, cosine decay
- **Qwen-2:** Quase idêntico ao LLaMA nos componentes Transformer
- **Mamba:** Blocos SSM fiéis ao paper original (d_state=16, causal conv, selective scan)
- **Mixtral:** Padrão Attn+MoE block com SwiGLU experts
- **Switch Transformer:** Aux loss de balanceamento (eq. 4)

### O que é realmente customizado

1. **Layout híbrido Mamba + Attn+MoE:** A combinação de 14 Mamba + 4 Attn+MoE não corresponde a nenhuma arquitetura publicada específica. É inspirado em trabalhos como Jamba (AI21) mas com proporção diferente.
2. **MoE apenas em camadas de atenção:** A decisão de acoplar MoE exclusivamente às camadas de atenção é uma escolha de design específica deste projeto.
3. **moe_intermediate=1024 (0.5× d_model):** Significativamente menor que o padrão (geralmente ≥ 2× d_model).

---

## 7. Pontos Fracos e Limitações

### 7.1 CRÍTICO — MoE Dense Dispatch

**Arquivo:** `model.py:475-478`
```python
expert_outs = torch.stack(
    [self.experts[e](x_flat) for e in range(self.n_experts)],
    dim=1
)  # [N, E, D]
```

**Problema:** Computa TODOS os 8 experts para TODOS os tokens, depois seleciona apenas top-2. Isso é 4× mais compute e memória do que necessário.

**Impacto:** Throughput reduzido em ~3-4× no MoE. Para 4 camadas MoE, representa ~15-20% do compute total desperdiçado.

**Solução:** Usar despacho esparso com scatter/gather indexado, ou `torch.where` com máscara.

### 7.2 ALTO — Geração sem KV Cache Real

**Arquivo:** `inference.py:175-176` e `model.py:700-701, 728`
```python
# inference.py
logits_new, _, _ = model(next_input)  # Sem kv_caches!

# model.py generate()
logits, _, _ = self.forward(ids, kv_caches=None)  # Prefill sem cache
logits_new, _, _ = self.forward(next_token, kv_caches=None)  # Decode sem cache!
```

**Problema:** A interface de KV cache existe em `GroupedQueryAttention.forward()` mas **nunca é usada** na geração. Cada novo token recomputa toda a sequência do zero.

**Impacto:** Geração O(L²) em vez de O(L). Para 200 tokens, ~100× mais lento que deveria.

**Nota:** Para Mamba, não há KV cache — o estado do SSM deveria ser mantido entre steps, mas isso não está implementado. A geração Mamba eficiente requer `step()` mode (processar 1 token por vez mantendo o estado h).

### 7.3 ALTO — Ausência de Shared Expert no MoE

**Referência:** DeepSeek-V2 (2024), DeepSeek-V3 (2024)

**Problema:** Todos os experts são esparsos. Se o router distribuir mal, nenhuma capacidade fixa processa o token.

**Solução:** Adicionar um expert "shared" (denso) que processa todos os tokens, com experts esparsos em cima.

### 7.4 MÉDIO — MoE Intermediate Muito Pequeno

**Arquivo:** `config.yaml:54`
```yaml
moe_intermediate: 1024  # 0.5× d_model
```

**Problema:** Experts com intermediate=1024 para d_model=2048 são muito estreitos. O FFN padrão em LLMs é 4× d_model (8192). Mesmo com MoE, experts típicos usam 2-4× d_model.

**Impacto:** Capacidade por expert limitada. Cada expert SwiGLU tem apenas ~6.3M params.

**Contexto:** A escolha foi por limitação de VRAM (WSL2). É uma troca válida, mas limita qualidade.

### 7.5 MÉDIO — Sem Z-Loss no Router

**Referência:** ST-MoE (Zoph et al. 2022), DeepSeek

**Problema:** Apenas aux_loss (balanceamento de carga). Z-loss penaliza logits do router muito grandes, evitando softmax saturation.

```python
# Falta isso:
z_loss = coeff * (gate_logits.float().logsumexp(-1) ** 2).mean()
```

### 7.6 MÉDIO — Mamba-1 vs Mamba-2

**Problema:** Usando Mamba-1 (d_state=16). Mamba-2 (Dao & Gu 2024) usa SSD (State Space Duality) com d_state=64-128 e é 2-8× mais rápido em hardware moderno.

**Impacto:** Performance subótima em GPUs com tensor cores. Mamba-2 explora melhor a largura de banda de memória.

**Nota:** Migrar para Mamba-2 requer treino do zero e mudança no kernel CUDA.

### 7.7 BAIXO — GradScaler Instanciado Desnecessariamente

**Arquivo:** `train.py:703`
```python
scaler = torch.amp.GradScaler(device="cuda", enabled=(dtype == torch.float16))
```

**Problema:** Quando dtype=bfloat16 (padrão), o scaler é instanciado com `enabled=False`. Funciona corretamente mas adiciona calls desnecessários (`.scale()`, `.unscale_()`, `.step()`, `.update()` são todos no-ops).

**Impacto:** Negligível em performance, mas adiciona complexidade ao código.

### 7.8 BAIXO — Sem QK-Norm

**Referência:** Qwen-2.5, Gemma-2

**Problema:** Dot product entre Q e K pode crescer sem bounds, causando instabilidade.

**Solução:** Aplicar RMSNorm a Q e K antes do dot product.

### 7.9 BAIXO — Layout Fixo de Camadas

**Arquivo:** `config.yaml:35-36`
```yaml
attention_layers: [3, 7, 11, 17]
```

**Problema:** O espaçamento não é uniforme (3 Mamba → 1 Attn, 3 Mamba → 1 Attn, 3 Mamba → 1 Attn, 5 Mamba → 1 Attn). A última sequência de Mamba (12-16) é muito longa sem atenção global.

**Sugestão:** Espaçamento mais uniforme, ex: [2, 6, 10, 14] ou [3, 7, 12, 17].

### 7.10 INFO — Contagem de Parâmetros usa `id(p)`

**Arquivo:** `model.py:599-604`

**Análise:** Correto para Python — `id()` retorna endereço único do objeto. Funciona para deduplicar weight-tied params. Porém, `id()` pode reutilizar endereços se objetos forem coletados pelo GC. No contexto de parameters() de um model vivo, isso não é um problema.

---

## 8. Comparação Detalhada com Arquiteturas Modernas

### Atenção

| Aspecto | Este modelo | Estado da arte |
|---------|-------------|---------------|
| GQA 4:1 | ✓ Moderno | ✓ LLaMA-2, Qwen-2 |
| RoPE | ✓ Standard | ✓ Considerar NTK-aware para ctx > 2048 |
| QK-Norm | ✗ Ausente | ✓ Qwen-2.5, Gemma-2 |
| FlashAttention | ✓ Via SDPA | ✓ FA-2/FA-3 |
| Sliding window | ✗ | ✓ Mistral, Gemma-2 (alternated) |

### FFN / MoE

| Aspecto | Este modelo | Estado da arte |
|---------|-------------|---------------|
| SwiGLU | ✓ Moderno | ✓ LLaMA, Qwen, Mistral |
| Top-k routing | ✓ top-2 | ✓ Mixtral, DeepSeek |
| Shared expert | ✗ Ausente | ✓ DeepSeek-V2/V3 |
| Z-loss | ✗ Ausente | ✓ ST-MoE, DeepSeek |
| Expert dispatch | ✗ Dense (ineficiente) | ✓ Sparse indexed |
| Expert capacity | ✗ Sem limite | ✓/✗ Depende (DeepSeek não usa) |

### Normalização e Estabilidade

| Aspecto | Este modelo | Estado da arte |
|---------|-------------|---------------|
| RMSNorm | ✓ Moderno | ✓ Universal em LLMs modernos |
| Pre-norm | ✓ Moderno | ✓ Universal |
| Residual scaling | ✓ 1/sqrt(2N) | ✓ Variantes existem |
| QK-Norm | ✗ | ✓ Qwen-2.5 |

### SSM

| Aspecto | Este modelo | Estado da arte |
|---------|-------------|---------------|
| Mamba-1 | ✓ Funcional | Mamba-2 é mais eficiente |
| d_state=16 | Original Mamba-1 | Mamba-2 usa 64-128 |
| Causal conv | ✓ kernel=4 | ✓ |
| Hybrid com Attn | ✓ | ✓ Jamba, Zamba, Samba |

---

## 9. Recomendações de Melhoria

### A. Baixo Risco (compatível com checkpoint existente)

| # | Melhoria | Impacto | Requer novo treino? |
|---|----------|---------|---------------------|
| A1 | Implementar despacho MoE esparso | +15-20% throughput | **Não** (mantém pesos) |
| A2 | Adicionar z-loss ao router | Melhor estabilidade | **Não** (novo loss term) |
| A3 | Remover GradScaler no modo bf16 | Código mais limpo | **Não** |
| A4 | Implementar KV cache na geração | 10-100× mais rápido | **Não** |
| A5 | Implementar Mamba step mode para geração | Geração muito mais rápida | **Não** |

### B. Médio Impacto (altera blocos, possível compatibilidade parcial)

| # | Melhoria | Impacto | Requer novo treino? |
|---|----------|---------|---------------------|
| B1 | Adicionar shared expert ao MoE | Melhor qualidade | **Sim** (novo parâmetro) |
| B2 | Aumentar moe_intermediate | Maior capacidade | **Sim** (mudança de shape) |
| B3 | Adicionar QK-Norm à atenção | Estabilidade | **Sim** (novo parâmetro) |
| B4 | Reajustar layout de camadas | Possível melhora | **Sim** |

### C. Alto Impacto / V2

| # | Melhoria | Impacto | Requer novo treino? |
|---|----------|---------|---------------------|
| C1 | Migrar para Mamba-2 (SSD) | 2-8× mais rápido em SSM | **Sim** |
| C2 | MoE com shared expert + z-loss + sparse dispatch | Qualidade + eficiência | **Sim** |
| C3 | Sliding window attention alternada | Melhor long-context | **Sim** |
| C4 | Aumento de experts (16-64 granulares) | Maior capacidade esparsa | **Sim** |

---

## 10. Compatibilidade com Checkpoints

### Mudanças SEM quebrar checkpoint (hot-fixáveis)

1. **Sparse MoE dispatch** — mesmo cálculo, diferente implementação → pesos idênticos
2. **Z-loss** — novo termo de loss, não altera forward → pesos carregam normalmente
3. **KV cache na geração** — lógica de inferência apenas
4. **Mamba step mode** — lógica de inferência apenas
5. **Remover GradScaler para bf16** — apenas otimizador

### Mudanças QUE quebram checkpoint

1. **Shared expert** — novo módulo com novos pesos
2. **Aumentar moe_intermediate** — shapes diferentes em Linear layers
3. **QK-Norm** — novos parâmetros de normalização
4. **Mamba-2** — arquitetura SSM totalmente diferente
5. **Alterar n_heads/n_kv_heads** — shapes das projeções Q/K/V
6. **Mudar layout de camadas** — tipo de bloco muda por posição

### Possibilidade de conversão parcial de checkpoint

É possível criar um script que:
- Carregue o checkpoint V1
- Copie todos os pesos compatíveis (embedding, lm_head, normas, projeções Q/K/V/O se shapes iguais)
- Inicialize pesos novos (shared expert, QK-Norm) aleatoriamente
- Salve como checkpoint V2

Isso permitiria "warm start" da V2, mas não seria equivalente a um treino completo.

---

## 11. Proposta V2

A V2 implementa:

1. **Despacho MoE esparso** — elimina compute desperdiçado
2. **Shared expert** — 1 expert denso + K experts esparsos (DeepSeek-style)
3. **Z-loss no router** — estabilidade
4. **QK-Norm opcional** — estabilidade de atenção
5. **Mamba-2-like improvements** — d_state maior, melhor init
6. **Geração com KV cache real** e Mamba step mode
7. **Config centralizado** em dataclass com validação
8. **Sem GradScaler desnecessário**
9. **Melhor modularização** — blocos claramente separados
10. **Layout de camadas mais flexível** — configurável por tipo

---

## 12. Tabela Final

| Componente | Arquivo:Linha | Função | Arquitetura | Estado | Recomendação |
|------------|--------------|--------|-------------|--------|-------------|
| Embedding | model.py:555 | Input tokens → vetores | GPT-2/LLaMA | ✓ Bom | Manter |
| Weight tying | model.py:574-575 | Compartilha embed↔lm_head | GPT-2/LLaMA | ✓ Bom | Manter |
| RMSNorm | model.py:121-132 | Normalização | LLaMA | ✓ Moderno | Manter |
| RoPE | model.py:135-157 | Pos encoding | LLaMA | ✓ Moderno | Considerar NTK scaling |
| GQA | model.py:334-401 | Atenção | LLaMA-2 | ✓ Moderno | Adicionar QK-Norm |
| SDPA | model.py:392-397 | Kernel de atenção | PyTorch | ✓ Bom | Manter |
| MambaMixer | model.py:210-316 | SSM core | Mamba-1 | ○ Funcional | Considerar Mamba-2 |
| MambaBlock | model.py:319-329 | Pre-norm wrapper | Mamba | ✓ Bom | Manter |
| SwiGLUExpert | model.py:406-416 | Expert FFN | LLaMA | ✓ Moderno | Aumentar intermediate |
| MoELayer dispatch | model.py:475-478 | Expert computation | Custom | ✗ Ineficiente | Sparse dispatch |
| MoE router | model.py:456-460 | Token routing | Switch Transformer | ○ Básico | + z-loss + shared expert |
| MoE aux loss | model.py:462-470 | Load balancing | Switch Transformer | ○ Funcional | + z-loss |
| AttentionMoEBlock | model.py:489-520 | Bloco composto | Mixtral | ✓ Bom | Manter estrutura |
| Init | model.py:578-593 | Weight init | GPT-2 | ✓ Razoável | Manter |
| Optimizer | train.py:185-221 | AdamW | LLaMA | ✓ Bom | Manter |
| LR Schedule | train.py:171-180 | Cosine | Chinchilla | ✓ Bom | Considerar WSD |
| Grad checkpointing | train.py:637-660 | Memória | Standard | ✓ Bom | Manter |
| KV cache | model.py:373-379 | Inferência | LLaMA | ✗ Não usado | Implementar |
| Generate | model.py:681-731 | Geração | Custom | ✗ Ineficiente | Reescrever com cache |

---

## 13. Plano de Evolução Arquitetural

### Versão Atual (V1)
- Mamba-1 SSM + GQA + MoE dense dispatch
- ~820M params, ~560M ativos
- Funcional mas com gargalos de eficiência

### V2 Recomendada
- Sparse MoE dispatch + shared expert
- QK-Norm + z-loss
- Mamba com d_state melhorado
- Geração eficiente com KV cache + SSM state
- Código modular e extensível

### V3 Futura (sugestão)
- Mamba-2 (SSD algorithm)
- MLA (Multi-head Latent Attention) estilo DeepSeek-V2
- Fine-grained experts (64+ granulares)
- Sliding window + global attention alternados
- Sequence parallelism para treino multi-GPU

---

## 14. Próximos Experimentos Recomendados

### Ablações Arquiteturais
1. **V1 vs V2 sparse MoE:** Mesmo checkpoint, comparar throughput (tokens/s)
2. **Shared expert vs sem shared expert:** Treinar 1B tokens cada, comparar val_loss
3. **QK-Norm vs sem QK-Norm:** Treinar com e sem, comparar estabilidade (grad_norm)
4. **d_state=16 vs d_state=32:** Impacto em perplexidade vs throughput

### Comparações V1 vs V2
5. **Throughput:** Medir tokens/s com V1 e V2 no mesmo hardware
6. **VRAM:** Medir peak VRAM com V1 e V2
7. **Convergência:** Treinar ambos por 2B tokens, comparar val_loss curves

### Eficiência
8. **MoE dispatch benchmark:** Dense vs sparse, medir tempo por batch
9. **Gradient checkpointing overhead:** Medir com e sem, tempo vs VRAM
10. **torch.compile:** Testar na V2 (sem mamba-ssm CUDA, apenas fallback)

### Qualidade
11. **Expert utilization:** Medir distribuição de routing ao longo do treino
12. **Mamba vs Attention perplexity:** Ablation removendo camadas Mamba ou Attention
13. **Chinchilla-optimal:** Verificar se 16-20× tokens/params é suficiente

### Impacto do Layout
14. **Layer arrangement:** Comparar [3,7,11,17] vs [2,6,10,14] vs [4,8,12,16]
15. **Ratio Mamba/Attn:** Comparar 14/4 vs 12/6 vs 10/8

### Impacto de Datasets
16. **EN/PT ratio:** Comparar 0.3/0.7 vs 0.5/0.5 vs 0.2/0.8
17. **Dataset size scaling:** Treinar com 4B, 8B, 16B tokens, medir scaling law

---

## 15. Conclusão

A arquitetura atual é **fundamentalmente sólida** e bem implementada. Segue boas práticas modernas (RMSNorm, GQA, SwiGLU, RoPE, pre-norm) e a combinação Mamba+Attention+MoE é uma abordagem válida para modelos sub-1B.

As **maiores oportunidades de melhoria** são:
1. **MoE sparse dispatch** (impacto imediato em throughput, sem retreino)
2. **KV cache + Mamba state na geração** (impacto massivo em inferência)
3. **Shared expert + z-loss** (requer retreino, melhora qualidade)

A V2 proposta endereça todos esses pontos mantendo a filosofia original do projeto.
