#!/usr/bin/env python3
"""
Optimized inference script for Hybrid Transformer-Mamba-MoE (V1 model).

Drop-in replacement for inference.py using the SAME model.py architecture,
with no model code changes required.

Improvements over inference.py:
  1. Real KV cache for attention layers (V1 had the interface but never used it)
  2. Separate prefill + decode phases with independent timing
  3. Combined top-k + top-p + min-p sampling (V1 only had top-p)
  4. Frequency + presence penalties (V1 only had basic repetition penalty)
  5. Vectorized repetition penalty (V1 used a slow Python loop)
  6. Perplexity evaluation mode (compute PPL on text or file)
  7. Benchmark mode (measure prefill and decode throughput separately)
  8. Multi-prompt batch file mode
  9. Configurable stop tokens/strings
  10. JSON output for all modes
  11. Interactive commands: /ppl, /set, /bench

Usage:
    # Interactive mode
    python inference_optimized.py

    # Single prompt
    python inference_optimized.py --prompt "Era uma vez"

    # Greedy decoding
    python inference_optimized.py --prompt "Hello" --temperature 0

    # Combined top-k + top-p
    python inference_optimized.py --prompt "Oi" --top_k 50 --top_p 0.9

    # Min-p sampling (modern alternative to top-p)
    python inference_optimized.py --prompt "Oi" --min_p 0.05

    # Perplexity evaluation
    python inference_optimized.py --eval_ppl --eval_text "O Brasil e um pais"
    python inference_optimized.py --eval_ppl --eval_file corpus.txt

    # Benchmark throughput
    python inference_optimized.py --benchmark --prompt "Era uma vez" --max_new_tokens 512

    # Batch prompts from file (one per line)
    python inference_optimized.py --prompt_file prompts.txt --no_stream

    # Save results as JSON
    python inference_optimized.py --prompt "Oi" --json_out result.json
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

from model import HybridMoEModel, ModelConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Tokenizer
# ═══════════════════════════════════════════════════════════════════════════════

def get_tokenizer() -> tiktoken.Encoding:
    """GPT-2 BPE tokenizer (same used during training)."""
    return tiktoken.get_encoding("gpt2")


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, device: torch.device) -> HybridMoEModel:
    """Load model from checkpoint. Reconstructs ModelConfig from saved state."""
    print(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg_dict = state["model_config"]
    if "attention_layers" in cfg_dict:
        cfg_dict["attention_layers"] = list(cfg_dict["attention_layers"])
    config = ModelConfig(**{k: v for k, v in cfg_dict.items()
                            if k in ModelConfig.__dataclass_fields__})

    model = HybridMoEModel(config)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    step           = state.get("step", 0)
    tokens_trained = state.get("tokens_trained", 0)
    val_loss       = state.get("best_val_loss", float("nan"))

    print(f"  Step:          {step:,}")
    print(f"  Tokens seen:   {tokens_trained / 1e9:.3f}B")
    print(f"  Best val loss: {val_loss:.4f}")
    print(f"  Parameters:    {model.num_parameters():,}")
    print(f"  Device:        {device}")
    return model


def set_inference_dtype(model: HybridMoEModel, infer_dtype: torch.dtype) -> HybridMoEModel:
    """Cast floating tensors for inference while preserving complex buffers."""
    if infer_dtype == torch.float32:
        return model

    complex_buffers = {
        name: buf.detach().clone()
        for name, buf in model.named_buffers()
        if torch.is_complex(buf)
    }
    model = model.to(infer_dtype)
    for name, buf in complex_buffers.items():
        module = model
        parts = name.split(".")
        for part in parts[:-1]:
            module = getattr(module, part)
        setattr(module, parts[-1], buf.to(device=next(model.parameters()).device))
    return model


def find_best_checkpoint_from_metadata(base: Path) -> Optional[Path]:
    """Recover the best step checkpoint from sidecar metadata when best.pt is missing."""
    best_path: Optional[Path] = None
    best_val = float("inf")
    for meta_path in sorted(base.glob("step_*.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        val_ce = meta.get("val_ce_loss")
        if val_ce is None:
            continue
        ckpt_path = meta_path.with_suffix(".pt")
        if not ckpt_path.exists():
            continue
        if float(val_ce) < best_val:
            best_val = float(val_ce)
            best_path = ckpt_path
    return best_path


def find_default_checkpoint(output_dir: str = "model_1b") -> str:
    """Return best.pt, inferred-best step checkpoint, or latest.pt."""
    base = Path(output_dir) / "checkpoints"
    best_link = base / "best.pt"
    if best_link.exists():
        return str(best_link)

    inferred_best = find_best_checkpoint_from_metadata(base)
    if inferred_best is not None:
        return str(inferred_best)

    latest_link = base / "latest.pt"
    if latest_link.exists():
        return str(latest_link)
    raise FileNotFoundError(
        f"No checkpoint found in {base}. "
        "Train the model first or pass --checkpoint <path>."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Sampling strategies
# ═══════════════════════════════════════════════════════════════════════════════

def apply_repetition_penalties(
    logits: torch.Tensor,          # [1, vocab]
    all_ids: List[int],
    repetition_penalty: float = 1.1,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> torch.Tensor:
    """
    Apply repetition, frequency, and presence penalties.

    - repetition_penalty: divides logits of previously seen tokens (>1.0 = penalize)
    - frequency_penalty: subtracts penalty proportional to token count in context
    - presence_penalty: subtracts flat penalty for any token seen at least once

    inference.py only had repetition_penalty with a slow per-token Python loop.
    This version vectorizes it and adds frequency/presence penalties.
    """
    if repetition_penalty == 1.0 and frequency_penalty == 0.0 and presence_penalty == 0.0:
        return logits

    if not all_ids:
        return logits

    logits = logits.clone()

    # Repetition penalty (multiplicative, vectorized)
    if repetition_penalty != 1.0:
        unique_ids = list(set(all_ids))
        token_ids = torch.tensor(unique_ids, device=logits.device, dtype=torch.long)
        token_logits = logits[0, token_ids]
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

    inference.py only supported top-p. This adds:
      - top_k: keep only the k most probable tokens before sampling
      - min_p: keep tokens with prob >= min_p * max_prob (llama.cpp style)
      - Pipeline: top_k -> min_p -> top_p -> sample
    """
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)

    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, float('-inf'))

    # Min-p filtering
    if min_p > 0.0:
        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True).values
        logits = logits.masked_fill(probs < min_p * max_prob, float('-inf'))

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cumprobs = sorted_probs.cumsum(-1)
        mask = (cumprobs - sorted_probs) > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum(-1, keepdim=True)
        return sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Generation with KV cache
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(
    model: HybridMoEModel,
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
    Generate text using the V1 model with real KV cache.

    The key fix: inference.py always passed kv_caches=None, recomputing the
    full sequence at every token (O(L^2) total). This version uses the KV
    cache infrastructure that model.py already had but inference.py never used.

    Returns dict with timing, throughput, and generated text.
    """
    ids = enc.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    model.eval()

    config = model.config
    n_attn = len(config.attention_layers)
    attn_set = config.attention_set

    # ── Prefill: full prompt in one pass, populating KV caches ────────────
    t_prefill_start = time.perf_counter()

    # Initialize KV caches list (one per attention layer)
    kv_caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_attn

    # Run prefill through the model layer-by-layer to capture KV caches.
    # We replicate the forward logic because model.forward() uses
    # freqs_cis[:L] which is correct for prefill but we need position-aware
    # slicing during decode.
    x = model.embed_tokens(input_ids)  # [1, prompt_len, d_model]
    freqs = model.freqs_cis[:input_ids.shape[1]]
    attn_idx = 0

    for i, layer in enumerate(model.layers):
        if i in attn_set:
            # AttentionMoEBlock: pass kv_cache=None to get initial cache
            x, _aux, new_cache = layer(x, freqs, kv_cache=None)
            kv_caches[attn_idx] = new_cache
            attn_idx += 1
        else:
            # MambaBlock
            x = layer(x)

    x = model.norm(x)
    next_logits = model.lm_head(x[:, -1:, :]).squeeze(1)  # [1, vocab]

    t_prefill_end = time.perf_counter()
    prefill_time = t_prefill_end - t_prefill_start

    # ── Decode: autoregressive with KV cache ──────────────────────────────
    generated_ids: List[int] = []
    all_ids = list(ids)
    decoded_text = ""

    if stream:
        print(prompt, end="", flush=True)

    t_decode_start = time.perf_counter()

    for _step in range(max_new_tokens):
        # Apply penalties
        work_logits = apply_repetition_penalties(
            next_logits, all_ids,
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
                for s in stop_strings:
                    idx = decoded_text.find(s)
                    if idx >= 0:
                        decoded_text = decoded_text[:idx]
                        break
                break

        # ── Single-token forward with KV cache ────────────────────────
        # Position in the full sequence (prompt + generated so far)
        seq_pos = len(ids) + len(generated_ids) - 1

        x = model.embed_tokens(next_token.view(1, 1))  # [1, 1, d_model]
        freqs_step = model.freqs_cis[seq_pos:seq_pos + 1]  # [1, head_dim//2]
        attn_idx = 0

        for i, layer in enumerate(model.layers):
            if i in attn_set:
                # Use KV cache: attention only computes for the new token
                x, _aux, new_cache = layer(x, freqs_step, kv_cache=kv_caches[attn_idx])
                kv_caches[attn_idx] = new_cache
                attn_idx += 1
            else:
                # Mamba: processes single token (no KV cache needed)
                x = layer(x)

        x = model.norm(x)
        next_logits = model.lm_head(x).squeeze(1)  # [1, vocab]

    t_decode_end = time.perf_counter()
    decode_time = t_decode_end - t_decode_start
    n_gen = len(generated_ids)

    if stream:
        print()

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
    model: HybridMoEModel,
    enc: tiktoken.Encoding,
    text: str,
    device: torch.device,
    stride: int = 512,
) -> Dict[str, float]:
    """
    Compute perplexity on a text using sliding window.

    Processes overlapping windows of max_seq_len tokens, only counting loss
    on the non-overlapping portion of each window (except the first).

    Args:
        text: input text to evaluate
        stride: step size between windows (smaller = more accurate but slower)
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

    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_len, seq_len)
        chunk = token_ids[begin:end].unsqueeze(0)  # [1, chunk_len]

        input_ids = chunk[:, :-1]
        targets = chunk[:, 1:]

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            logits, _, _ = model(input_ids)

        # Only count loss on the new (non-overlapping) portion
        target_start = max(0, prev_end - begin - 1)
        target_logits = logits[:, target_start:, :]
        target_labels = targets[:, target_start:]

        loss = F.cross_entropy(
            target_logits.reshape(-1, target_logits.size(-1)),
            target_labels.reshape(-1),
            reduction="sum",
        )
        total_nll += loss.item()
        total_tokens += target_labels.numel()
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
    model: HybridMoEModel,
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
    benchmark_rounds.
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

        result = generate(
            model=model,
            enc=enc,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,  # greedy for deterministic benchmarks
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

        if device.type == "cuda":
            torch.cuda.empty_cache()

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
        description="Optimized inference for Hybrid Transformer-Mamba-MoE (V1 model)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to checkpoint (.pt)")
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
                     help="Frequency penalty (additive). 0 = disabled.")
    gen.add_argument("--presence_penalty", type=float, default=0.0,
                     help="Presence penalty (additive). 0 = disabled.")
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
                   help="Write results as JSON to this file")

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
    model = set_inference_dtype(model, infer_dtype)

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
    print(f"\nHybrid-MoE-1B — Optimized interactive inference")
    print(f"  checkpoint:   {ckpt_path}")
    print(f"  device:       {device}  |  dtype: {args.dtype}")
    print(f"  temperature:  {args.temperature}  |  top_k: {args.top_k}  "
          f"|  top_p: {args.top_p}  |  min_p: {args.min_p}")
    print(f"  rep_penalty:  {args.repetition_penalty}  "
          f"|  freq_penalty: {args.frequency_penalty}  "
          f"|  pres_penalty: {args.presence_penalty}")
    print(f"  Commands: 'quit'=exit, '/ppl <text>'=perplexity, "
          f"'/set key=val'=change param, '/bench'=quick benchmark")
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

        # Interactive benchmark
        if prompt.strip() == "/bench":
            benchmark(
                model, enc,
                "Era uma vez, em uma terra distante, havia um",
                max_new_tokens=128,
                device=device,
                warmup_rounds=1,
                benchmark_rounds=3,
            )
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
