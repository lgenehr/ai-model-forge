#!/usr/bin/env python3
"""
Inference script V2 for Hybrid Transformer-Mamba-MoE.

Improvements over V1 (inference.py):
  1. Real KV cache for attention layers (V1 recomputed full seq every token)
  2. Separate prefill + decode phases with independent timing
  3. Combined top-k + top-p sampling (V1 only had top-p)
  4. Min-p sampling support (modern alternative, used in llama.cpp/vLLM)
  5. Frequency + presence penalty (V1 only had basic repetition penalty)
  6. Perplexity evaluation mode (compute PPL on text or file)
  7. Benchmark mode (measure prefill and decode throughput separately)
  8. Auto-detect V1/V2 checkpoints
  9. Multi-prompt batch file mode
  10. Configurable stop tokens/strings

Usage:
    # Interactive mode
    python inference_v2.py

    # Single prompt
    python inference_v2.py --prompt "Era uma vez"

    # Greedy decoding
    python inference_v2.py --prompt "Hello" --temperature 0

    # Combined top-k + top-p
    python inference_v2.py --prompt "Oi" --top_k 50 --top_p 0.9

    # Min-p sampling (alternative to top-p)
    python inference_v2.py --prompt "Oi" --min_p 0.05

    # Perplexity evaluation
    python inference_v2.py --eval_ppl --eval_text "O Brasil e um pais tropical"
    python inference_v2.py --eval_ppl --eval_file corpus.txt

    # Benchmark throughput
    python inference_v2.py --benchmark --prompt "Era uma vez" --max_new_tokens 512

    # Batch prompts from file (one per line)
    python inference_v2.py --prompt_file prompts.txt --no_stream

    # Use V1 checkpoint (auto-detected)
    python inference_v2.py --checkpoint ../hybrid-moe-1b/model_1b/checkpoints/best.pt

    # Use V2 checkpoint
    python inference_v2.py --checkpoint model_1b/v2/checkpoints/best.pt
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import tiktoken

from model_v2 import HybridMoEModelV2, ModelConfigV2


# ═══════════════════════════════════════════════════════════════════════════════
# Tokenizer
# ═══════════════════════════════════════════════════════════════════════════════

def get_tokenizer() -> tiktoken.Encoding:
    """GPT-2 BPE tokenizer (same used during training)."""
    return tiktoken.get_encoding("gpt2")


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint loading (auto-detect V1 vs V2)
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, device: torch.device) -> HybridMoEModelV2:
    """
    Load model from checkpoint. Auto-detects V1 or V2 format.

    V1 checkpoints are loaded into the V2 model architecture with partial
    weight migration (new V2 modules get random init).
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_version = state.get("model_version", "v1")
    cfg_dict = state["model_config"]

    if "attention_layers" in cfg_dict:
        cfg_dict["attention_layers"] = list(cfg_dict["attention_layers"])

    # Build V2 config from saved dict (ignore unknown keys from V1)
    v2_fields = {f.name for f in ModelConfigV2.__dataclass_fields__.values()}
    config_kwargs = {k: v for k, v in cfg_dict.items() if k in v2_fields}
    config = ModelConfigV2(**config_kwargs)

    model = HybridMoEModelV2(config)

    if model_version == "v2":
        model.load_state_dict(state["model"])
        print(f"  Loaded V2 checkpoint")
    else:
        # V1 checkpoint: partial migration (shared_expert, qk_norm will be random)
        v2_sd = model.state_dict()
        loaded, skipped, new_keys = 0, 0, 0
        for key in v2_sd:
            if key in state["model"]:
                if state["model"][key].shape == v2_sd[key].shape:
                    v2_sd[key] = state["model"][key]
                    loaded += 1
                else:
                    skipped += 1
            else:
                new_keys += 1
        model.load_state_dict(v2_sd)
        print(f"  Loaded V1 checkpoint (migrated: {loaded} keys, "
              f"{new_keys} new random, {skipped} skipped)")

    model.to(device)
    model.eval()

    step = state.get("step", 0)
    tokens_trained = state.get("tokens_trained", 0)
    val_loss = state.get("best_val_loss", float("nan"))

    print(f"  Step:          {step:,}")
    print(f"  Tokens seen:   {tokens_trained / 1e9:.3f}B")
    print(f"  Best val loss: {val_loss:.4f}")
    print(f"  Parameters:    {model.num_parameters():,}")
    print(f"  Device:        {device}")
    print(f"  Architecture:  V2 (shared_expert={config.moe_shared_expert}, "
          f"qk_norm={config.qk_norm})")
    return model


def find_default_checkpoint(output_dir: str = "model_1b") -> str:
    """Search for best.pt or latest.pt in output_dir and output_dir/v2."""
    candidates = [
        Path(output_dir) / "v2" / "checkpoints",
        Path(output_dir) / "checkpoints",
    ]
    for base in candidates:
        for name in ("best.pt", "latest.pt"):
            p = base / name
            if p.exists():
                return str(p)
    raise FileNotFoundError(
        f"No checkpoint found in {candidates}. "
        "Train the model first or pass --checkpoint <path>."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Sampling strategies
# ═══════════════════════════════════════════════════════════════════════════════

def apply_repetition_penalties(
    logits: torch.Tensor,          # [B, vocab]
    generated_ids: List[int],
    all_ids: List[int],
    repetition_penalty: float = 1.1,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> torch.Tensor:
    """
    Apply repetition, frequency, and presence penalties.

    - repetition_penalty: divides logits of previously seen tokens (>1 = penalize)
    - frequency_penalty: subtracts penalty proportional to token count in context
    - presence_penalty: subtracts flat penalty for any token seen at least once

    V1 only had repetition_penalty with a naive per-token loop.
    V2 uses vectorized operations and adds frequency/presence penalties.
    """
    if repetition_penalty == 1.0 and frequency_penalty == 0.0 and presence_penalty == 0.0:
        return logits

    logits = logits.clone()

    if not all_ids:
        return logits

    # Repetition penalty (multiplicative, as in V1 but vectorized)
    if repetition_penalty != 1.0:
        unique_ids = list(set(all_ids))
        token_ids = torch.tensor(unique_ids, device=logits.device, dtype=torch.long)
        token_logits = logits[0, token_ids]
        # Penalize: divide positive logits, multiply negative logits
        logits[0, token_ids] = torch.where(
            token_logits > 0,
            token_logits / repetition_penalty,
            token_logits * repetition_penalty,
        )

    # Frequency and presence penalties (additive, OpenAI-style)
    if frequency_penalty != 0.0 or presence_penalty != 0.0:
        from collections import Counter
        counts = Counter(all_ids)
        ids_tensor = torch.tensor(list(counts.keys()), device=logits.device, dtype=torch.long)
        freq_tensor = torch.tensor(list(counts.values()), device=logits.device, dtype=logits.dtype)

        logits[0, ids_tensor] -= frequency_penalty * freq_tensor
        if presence_penalty != 0.0:
            logits[0, ids_tensor] -= presence_penalty

    return logits


def sample_token(
    logits: torch.Tensor,    # [1, vocab]
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.9,
    min_p: float = 0.0,
) -> torch.Tensor:
    """
    Sample next token with combined top-k + top-p + min-p filtering.

    V1 only supported top-p. V2 adds:
      - top_k: keep only the k most probable tokens before sampling
      - min_p: keep tokens with probability >= min_p * max_prob (llama.cpp style)
      - Combined: top_k -> top_p -> sample (standard modern pipeline)

    Args:
        logits: raw logits [1, vocab_size]
        temperature: 0 = greedy, higher = more random
        top_k: 0 = disabled, >0 = keep top k tokens
        top_p: 1.0 = disabled, <1.0 = nucleus sampling
        min_p: 0.0 = disabled, >0 = keep tokens with prob >= min_p * max_prob
    """
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)  # greedy

    logits = logits / temperature

    # ── Top-k filtering ───────────────────────────────────────────────────
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, float('-inf'))

    # ── Min-p filtering (applied before top-p) ────────────────────────────
    if min_p > 0.0:
        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True).values
        min_threshold = min_p * max_prob
        logits = logits.masked_fill(probs < min_threshold, float('-inf'))

    # ── Top-p (nucleus) filtering ─────────────────────────────────────────
    if top_p < 1.0:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cumprobs = sorted_probs.cumsum(-1)
        # Remove tokens above the cumulative threshold (keep first token always)
        mask = (cumprobs - sorted_probs) > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum(-1, keepdim=True)
        return sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))

    # No top-p: sample from (possibly top-k / min-p filtered) distribution
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Generation with KV cache
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(
    model: HybridMoEModelV2,
    enc: tiktoken.Encoding,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.9,
    min_p: float = 0.0,
    repetition_penalty: float = 1.1,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stop_strings: Optional[List[str]] = None,
    device: torch.device = torch.device("cpu"),
    stream: bool = True,
) -> Dict[str, object]:
    """
    Generate text from a prompt using V2 model with KV cache.

    Returns dict with:
      - text: generated text (excluding prompt)
      - token_ids: list of generated token IDs
      - prefill_time: seconds for prompt processing
      - decode_time: seconds for token generation
      - prefill_tok_s: prefill throughput
      - decode_tok_s: decode throughput
      - total_tokens: number of generated tokens
    """
    ids = enc.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    model.eval()

    config = model.config
    n_attn = len(config.attention_layers)
    attn_set = config.attention_set

    # ── Prefill phase ─────────────────────────────────────────────────────
    t_prefill_start = time.perf_counter()

    kv_caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_attn

    x = model.embed_tokens(input_ids)
    freqs = model.freqs_cis[:input_ids.shape[1]]
    attn_idx = 0

    for i, layer in enumerate(model.layers):
        if i in attn_set:
            x, _, new_cache = layer(x, freqs, kv_cache=None)
            kv_caches[attn_idx] = new_cache
            attn_idx += 1
        else:
            x = layer(x)

    x = model.norm(x)
    next_logits = model.lm_head(x[:, -1:, :]).squeeze(1)  # [1, vocab]

    t_prefill_end = time.perf_counter()
    prefill_time = t_prefill_end - t_prefill_start

    # ── Decode phase (autoregressive with KV cache) ───────────────────────
    generated_ids: List[int] = []
    all_ids = list(ids)
    decoded_text = ""

    if stream:
        print(prompt, end="", flush=True)

    t_decode_start = time.perf_counter()

    for step_idx in range(max_new_tokens):
        # Apply penalties
        work_logits = apply_repetition_penalties(
            next_logits, generated_ids, all_ids,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        # Sample
        next_token = sample_token(
            work_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
        )

        tok_id = next_token.item()
        generated_ids.append(tok_id)
        all_ids.append(tok_id)

        # Stream output
        if stream:
            try:
                char = enc.decode([tok_id])
                print(char, end="", flush=True)
            except Exception:
                pass

        # Check EOS
        if tok_id == enc.eot_token:
            break

        # Check stop strings
        if stop_strings:
            decoded_text = enc.decode(generated_ids)
            if any(s in decoded_text for s in stop_strings):
                # Trim to stop string boundary
                for s in stop_strings:
                    idx = decoded_text.find(s)
                    if idx >= 0:
                        decoded_text = decoded_text[:idx]
                        break
                break

        # ── Single-token forward with KV cache ─────────────────────────
        seq_pos = len(ids) + len(generated_ids) - 1
        freqs_step = model.freqs_cis[seq_pos:seq_pos + 1]

        x = model.embed_tokens(next_token.view(1, 1))
        attn_idx = 0
        for i, layer in enumerate(model.layers):
            if i in attn_set:
                x, _, new_cache = layer(x, freqs_step, kv_cache=kv_caches[attn_idx])
                kv_caches[attn_idx] = new_cache
                attn_idx += 1
            else:
                x = layer(x)

        x = model.norm(x)
        next_logits = model.lm_head(x).squeeze(1)

    t_decode_end = time.perf_counter()
    decode_time = t_decode_end - t_decode_start
    n_gen = len(generated_ids)

    if stream:
        print()

    # Final decoded text
    if not decoded_text:
        decoded_text = enc.decode(generated_ids) if generated_ids else ""

    prefill_tok_s = len(ids) / prefill_time if prefill_time > 0 else 0
    decode_tok_s = n_gen / decode_time if decode_time > 0 else 0

    print(f"\n[{n_gen} tokens | prefill: {len(ids)} tok in {prefill_time:.2f}s "
          f"({prefill_tok_s:.0f} tok/s) | decode: {n_gen} tok in {decode_time:.2f}s "
          f"({decode_tok_s:.0f} tok/s)]")

    return {
        "text": decoded_text,
        "token_ids": generated_ids,
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
        "total_tokens": n_gen,
        "prompt_tokens": len(ids),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Perplexity evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_perplexity(
    model: HybridMoEModelV2,
    enc: tiktoken.Encoding,
    text: str,
    device: torch.device,
    stride: int = 512,
) -> Dict[str, float]:
    """
    Compute perplexity on a text using sliding window.

    Uses a strided approach to handle texts longer than max_seq_len:
    processes overlapping windows, only counts loss on the non-overlapping
    portion of each window (except the first).

    Args:
        text: input text to evaluate
        stride: step size between windows (smaller = more accurate but slower)

    Returns dict with: ppl, avg_loss, total_tokens, time_s
    """
    model.eval()
    max_len = model.config.max_seq_len

    tokens = enc.encode(text)
    if not tokens:
        return {"ppl": float("nan"), "avg_loss": float("nan"),
                "total_tokens": 0, "time_s": 0.0}

    token_ids = torch.tensor(tokens, dtype=torch.long, device=device)
    seq_len = token_ids.size(0)

    total_nll = 0.0
    total_tokens = 0
    t0 = time.perf_counter()

    # Sliding window with stride
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_len, seq_len)
        chunk = token_ids[begin:end].unsqueeze(0)  # [1, chunk_len]

        input_ids = chunk[:, :-1]
        targets = chunk[:, 1:]

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            logits, _, _ = model(input_ids)

        # Only count loss on the new (non-overlapping) tokens
        # For the first window, count all; for subsequent, skip the overlap
        target_start = max(0, prev_end - begin - 1)
        target_logits = logits[:, target_start:, :]
        target_labels = targets[:, target_start:]

        loss = F.cross_entropy(
            target_logits.reshape(-1, target_logits.size(-1)),
            target_labels.reshape(-1),
            reduction="sum",
        )
        n_tokens = target_labels.numel()
        total_nll += loss.item()
        total_tokens += n_tokens
        prev_end = end

        if end >= seq_len:
            break

    elapsed = time.perf_counter() - t0
    avg_loss = total_nll / total_tokens if total_tokens > 0 else float("nan")
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    return {
        "ppl": ppl,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "time_s": elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark mode
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def benchmark(
    model: HybridMoEModelV2,
    enc: tiktoken.Encoding,
    prompt: str,
    max_new_tokens: int = 256,
    device: torch.device = torch.device("cpu"),
    warmup_rounds: int = 2,
    benchmark_rounds: int = 5,
) -> Dict[str, float]:
    """
    Benchmark prefill and decode throughput.

    Runs warmup rounds to stabilize CUDA kernels, then averages over
    benchmark_rounds. Reports separate prefill and decode tok/s.
    """
    model.eval()
    ids = enc.encode(prompt)
    prompt_len = len(ids)

    print(f"Benchmarking: prompt={prompt_len} tokens, "
          f"generate={max_new_tokens} tokens, "
          f"{warmup_rounds} warmup + {benchmark_rounds} runs")

    all_prefill_times = []
    all_decode_times = []
    all_decode_tokens = []

    for run_idx in range(warmup_rounds + benchmark_rounds):
        is_warmup = run_idx < warmup_rounds

        # Run generation silently
        result = generate(
            model=model,
            enc=enc,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,  # greedy for deterministic benchmark
            device=device,
            stream=False,
        )

        if not is_warmup:
            all_prefill_times.append(result["prefill_time"])
            all_decode_times.append(result["decode_time"])
            all_decode_tokens.append(result["total_tokens"])

        label = "warmup" if is_warmup else f"run {run_idx - warmup_rounds + 1}"
        print(f"  [{label}] prefill: {result['prefill_tok_s']:.0f} tok/s | "
              f"decode: {result['decode_tok_s']:.0f} tok/s | "
              f"tokens: {result['total_tokens']}")

        # Clear cache between runs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Aggregate
    avg_prefill_time = sum(all_prefill_times) / len(all_prefill_times)
    avg_decode_time = sum(all_decode_times) / len(all_decode_times)
    avg_decode_tokens = sum(all_decode_tokens) / len(all_decode_tokens)

    avg_prefill_tps = prompt_len / avg_prefill_time if avg_prefill_time > 0 else 0
    avg_decode_tps = avg_decode_tokens / avg_decode_time if avg_decode_time > 0 else 0

    print(f"\n{'='*60}")
    print(f"Benchmark results ({benchmark_rounds} runs):")
    print(f"  Prefill: {avg_prefill_tps:.0f} tok/s "
          f"({prompt_len} tokens, {avg_prefill_time*1000:.1f} ms avg)")
    print(f"  Decode:  {avg_decode_tps:.1f} tok/s "
          f"({avg_decode_tokens:.0f} tokens, {avg_decode_time*1000:.1f} ms avg)")
    print(f"  Total:   {(prompt_len + avg_decode_tokens) / (avg_prefill_time + avg_decode_time):.0f} tok/s")

    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak VRAM: {peak_mem:.2f} GB")

    return {
        "prefill_tok_s": avg_prefill_tps,
        "decode_tok_s": avg_decode_tps,
        "avg_prefill_ms": avg_prefill_time * 1000,
        "avg_decode_ms": avg_decode_time * 1000,
        "prompt_tokens": prompt_len,
        "avg_decode_tokens": avg_decode_tokens,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference V2 for Hybrid Transformer-Mamba-MoE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to checkpoint (.pt). Auto-detects V1/V2.")
    p.add_argument("--output_dir", type=str, default="model_1b",
                   help="Base dir to search for checkpoints")
    p.add_argument("--device", type=str, default=None,
                   help="Device (cuda / cpu). Auto-detects.")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])

    # Generation
    gen = p.add_argument_group("Generation")
    gen.add_argument("--prompt", type=str, default=None,
                     help="Input prompt. Omit for interactive mode.")
    gen.add_argument("--prompt_file", type=str, default=None,
                     help="File with one prompt per line (batch mode)")
    gen.add_argument("--max_new_tokens", type=int, default=200)
    gen.add_argument("--temperature", type=float, default=0.8,
                     help="Sampling temperature. 0 = greedy.")
    gen.add_argument("--top_k", type=int, default=0,
                     help="Top-k sampling. 0 = disabled.")
    gen.add_argument("--top_p", type=float, default=0.9,
                     help="Nucleus sampling threshold. 1.0 = disabled.")
    gen.add_argument("--min_p", type=float, default=0.0,
                     help="Min-p sampling (llama.cpp style). 0 = disabled.")
    gen.add_argument("--repetition_penalty", type=float, default=1.1,
                     help="Repetition penalty (multiplicative). 1.0 = disabled.")
    gen.add_argument("--frequency_penalty", type=float, default=0.0,
                     help="Frequency penalty (additive, OpenAI-style). 0 = disabled.")
    gen.add_argument("--presence_penalty", type=float, default=0.0,
                     help="Presence penalty (additive, OpenAI-style). 0 = disabled.")
    gen.add_argument("--stop", type=str, nargs="*", default=None,
                     help="Stop strings (generation stops when any is produced)")
    gen.add_argument("--no_stream", action="store_true",
                     help="Disable streaming output")

    # Evaluation
    ev = p.add_argument_group("Evaluation")
    ev.add_argument("--eval_ppl", action="store_true",
                    help="Compute perplexity instead of generating")
    ev.add_argument("--eval_text", type=str, default=None,
                    help="Text to evaluate perplexity on")
    ev.add_argument("--eval_file", type=str, default=None,
                    help="File to evaluate perplexity on")
    ev.add_argument("--eval_stride", type=int, default=512,
                    help="Stride for sliding-window perplexity")

    # Benchmark
    bm = p.add_argument_group("Benchmark")
    bm.add_argument("--benchmark", action="store_true",
                    help="Run throughput benchmark")
    bm.add_argument("--benchmark_rounds", type=int, default=5)
    bm.add_argument("--warmup_rounds", type=int, default=2)

    # Output
    p.add_argument("--json_out", type=str, default=None,
                   help="Write generation/eval results as JSON to this file")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── dtype ─────────────────────────────────────────────────────────────
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    infer_dtype = dtype_map[args.dtype]

    # ── Checkpoint ────────────────────────────────────────────────────────
    ckpt_path = args.checkpoint or find_default_checkpoint(args.output_dir)

    # ── Load model ────────────────────────────────────────────────────────
    model = load_model(ckpt_path, device)
    if infer_dtype != torch.float32:
        model = model.to(infer_dtype)

    enc = get_tokenizer()

    # ── Perplexity evaluation mode ────────────────────────────────────────
    if args.eval_ppl:
        if args.eval_file:
            text = Path(args.eval_file).read_text(encoding="utf-8")
            source = args.eval_file
        elif args.eval_text:
            text = args.eval_text
            source = "cli"
        else:
            print("Error: --eval_ppl requires --eval_text or --eval_file")
            sys.exit(1)

        print(f"\nEvaluating perplexity on {len(text)} chars from {source}...")
        result = evaluate_perplexity(
            model, enc, text, device, stride=args.eval_stride,
        )
        print(f"\n{'='*50}")
        print(f"Perplexity:   {result['ppl']:.2f}")
        print(f"Avg CE loss:  {result['avg_loss']:.4f}")
        print(f"Tokens:       {result['total_tokens']:,}")
        print(f"Time:         {result['time_s']:.2f}s")

        if args.json_out:
            result["source"] = source
            result["checkpoint"] = ckpt_path
            Path(args.json_out).write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            print(f"Results saved to {args.json_out}")
        return

    # ── Benchmark mode ────────────────────────────────────────────────────
    if args.benchmark:
        prompt = args.prompt or "Era uma vez, em uma terra distante, havia um"
        result = benchmark(
            model, enc, prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
            warmup_rounds=args.warmup_rounds,
            benchmark_rounds=args.benchmark_rounds,
        )
        if args.json_out:
            result["checkpoint"] = ckpt_path
            Path(args.json_out).write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
        return

    # ── Common generation kwargs ──────────────────────────────────────────
    gen_kwargs = dict(
        model=model,
        enc=enc,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        stop_strings=args.stop,
        device=device,
        stream=not args.no_stream,
    )

    # ── Batch prompt file mode ────────────────────────────────────────────
    if args.prompt_file:
        prompts = [
            line.strip() for line in
            Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        print(f"\nProcessing {len(prompts)} prompts from {args.prompt_file}\n")
        all_results = []
        for i, prompt in enumerate(prompts):
            print(f"--- Prompt {i+1}/{len(prompts)} ---")
            result = generate(prompt=prompt, **gen_kwargs)
            all_results.append({"prompt": prompt, **result})
            print()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if args.json_out:
            # Convert non-serializable values
            for r in all_results:
                r.pop("token_ids", None)
            Path(args.json_out).write_text(
                json.dumps(all_results, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"Results saved to {args.json_out}")
        return

    # ── Single prompt mode ────────────────────────────────────────────────
    if args.prompt is not None:
        result = generate(prompt=args.prompt, **gen_kwargs)
        if args.json_out:
            result.pop("token_ids", None)
            result["prompt"] = args.prompt
            result["checkpoint"] = ckpt_path
            Path(args.json_out).write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return

    # ── Interactive mode ──────────────────────────────────────────────────
    print(f"\nHybrid-MoE V2 — Interactive inference")
    print(f"  checkpoint:   {ckpt_path}")
    print(f"  device:       {device}  |  dtype: {args.dtype}")
    print(f"  temperature:  {args.temperature}  |  top_k: {args.top_k}  "
          f"|  top_p: {args.top_p}  |  min_p: {args.min_p}")
    print(f"  rep_penalty:  {args.repetition_penalty}  "
          f"|  freq_penalty: {args.frequency_penalty}  "
          f"|  pres_penalty: {args.presence_penalty}")
    print(f"  Commands: 'quit'=exit, '/ppl <text>'=perplexity, "
          f"'/set key=val'=change param")
    print()

    while True:
        try:
            prompt = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        # Interactive perplexity command
        if prompt.startswith("/ppl "):
            eval_text = prompt[5:].strip()
            if eval_text:
                result = evaluate_perplexity(model, enc, eval_text, device)
                print(f"PPL: {result['ppl']:.2f}  |  loss: {result['avg_loss']:.4f}  "
                      f"|  tokens: {result['total_tokens']}")
            continue

        # Interactive parameter setting
        if prompt.startswith("/set "):
            try:
                key, val = prompt[5:].split("=", 1)
                key = key.strip()
                val = val.strip()
                if key in gen_kwargs:
                    if key in ("stream",):
                        gen_kwargs[key] = val.lower() in ("true", "1", "yes")
                    elif key in ("top_k", "max_new_tokens"):
                        gen_kwargs[key] = int(val)
                    elif key in ("stop_strings",):
                        gen_kwargs[key] = [val]
                    else:
                        gen_kwargs[key] = float(val)
                    print(f"  {key} = {gen_kwargs[key]}")
                else:
                    print(f"  Unknown parameter: {key}")
            except Exception as e:
                print(f"  Error: {e}")
            continue

        generate(prompt=prompt, **gen_kwargs)
        print()

        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
