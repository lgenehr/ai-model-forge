#!/usr/bin/env python3
"""
Inference script for Hybrid Transformer-Mamba-MoE (~820M parameters).

Usage:
    # Interactive mode
    python inference.py

    # Single prompt
    python inference.py --prompt "Era uma vez"

    # Use specific checkpoint
    python inference.py --checkpoint model_1b/checkpoints/best.pt

    # Greedy decoding (no randomness)
    python inference.py --prompt "Hello" --temperature 0

    # More tokens, different sampling
    python inference.py --prompt "Oi" --max_new_tokens 500 --temperature 0.7 --top_p 0.95
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import tiktoken

from model import HybridMoEModel, ModelConfig


# ─── Tokenizer ────────────────────────────────────────────────────────────────

def get_tokenizer() -> tiktoken.Encoding:
    """GPT-2 BPE tokenizer (same used during training)."""
    return tiktoken.get_encoding("gpt2")


def encode(enc: tiktoken.Encoding, text: str) -> List[int]:
    return enc.encode(text)


def decode(enc: tiktoken.Encoding, ids: List[int]) -> str:
    return enc.decode(ids)


# ─── Checkpoint loading ────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> HybridMoEModel:
    """Load model from checkpoint. Reconstructs ModelConfig from saved state."""
    print(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config from saved dict
    cfg_dict = state["model_config"]
    # attention_layers may be stored as a list or set — ensure it's a list
    if "attention_layers" in cfg_dict:
        cfg_dict["attention_layers"] = list(cfg_dict["attention_layers"])
    # mamba_dt_rank is saved as int after __post_init__, pass it directly
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


def find_default_checkpoint(output_dir: str = "model_1b") -> str:
    """Return best.pt if it exists, otherwise latest.pt."""
    base = Path(output_dir) / "checkpoints"
    for name in ("best.pt", "latest.pt"):
        p = base / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"No checkpoint found in {base}. "
        "Train the model first or pass --checkpoint <path>."
    )


# ─── Generation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model: HybridMoEModel,
    enc: tiktoken.Encoding,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    device: torch.device = torch.device("cpu"),
    stream: bool = True,
) -> str:
    """
    Generate text from a prompt.

    Args:
        stream: If True, print tokens as they are generated.
    Returns:
        Generated text (excluding the prompt).
    """
    ids = encode(enc, prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    model.eval()

    # ── Prefill ──────────────────────────────────────────────────────────────
    logits, _, _ = model(input_ids)
    next_logits = logits[:, -1, :]  # [1, vocab]

    generated_ids: List[int] = []
    all_ids = list(ids)

    if stream:
        print(prompt, end="", flush=True)

    t0 = time.perf_counter()

    for _ in range(max_new_tokens):
        # Repetition penalty (applied in-place on a copy)
        logits_work = next_logits.clone()
        if repetition_penalty != 1.0:
            for tok_id in all_ids:
                logits_work[0, tok_id] /= repetition_penalty

        # Sampling
        if temperature <= 0:
            next_token = logits_work.argmax(-1, keepdim=True)  # greedy
        else:
            probs = torch.softmax(logits_work / temperature, dim=-1)
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                cumprobs = sorted_probs.cumsum(-1)
                mask = (cumprobs - sorted_probs) > top_p
                sorted_probs[mask] = 0.0
                sorted_probs /= sorted_probs.sum(-1, keepdim=True)
                next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
            else:
                next_token = torch.multinomial(probs, 1)

        tok_id = next_token.item()
        generated_ids.append(tok_id)
        all_ids.append(tok_id)

        if stream:
            # Decode incrementally — tiktoken may need multiple tokens for one char
            try:
                char = decode(enc, [tok_id])
                print(char, end="", flush=True)
            except Exception:
                pass

        # Check for EOS (GPT-2 EOS = 50256)
        if tok_id == enc.eot_token:
            break

        # Next-token forward pass (single token)
        next_input = next_token.view(1, 1)
        logits_new, _, _ = model(next_input)
        next_logits = logits_new[:, -1, :]

    elapsed = time.perf_counter() - t0
    n_gen = len(generated_ids)

    if stream:
        print()  # newline after streamed output

    tok_per_sec = n_gen / elapsed if elapsed > 0 else 0
    print(f"\n[{n_gen} tokens in {elapsed:.1f}s — {tok_per_sec:.0f} tok/s]")

    return decode(enc, generated_ids)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference for Hybrid Transformer-Mamba-MoE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to checkpoint (.pt). Defaults to model_1b/checkpoints/best.pt")
    p.add_argument("--output_dir", type=str, default="model_1b",
                   help="Output dir to find checkpoints when --checkpoint is not set")
    p.add_argument("--prompt", type=str, default=None,
                   help="Input prompt. If not set, enters interactive mode")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature. 0 = greedy")
    p.add_argument("--top_p", type=float, default=0.9,
                   help="Nucleus sampling probability threshold")
    p.add_argument("--repetition_penalty", type=float, default=1.1,
                   help="Penalize repeated tokens (1.0 = disabled)")
    p.add_argument("--device", type=str, default=None,
                   help="Device override (cuda / cpu). Auto-detects by default")
    p.add_argument("--no_stream", action="store_true",
                   help="Disable streaming output (print everything at once)")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="Inference dtype")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── dtype ─────────────────────────────────────────────────────────────────
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    infer_dtype = dtype_map[args.dtype]

    # ── Checkpoint ────────────────────────────────────────────────────────────
    ckpt_path = args.checkpoint or find_default_checkpoint(args.output_dir)

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(ckpt_path, device)

    # Cast to inference dtype
    if infer_dtype != torch.float32:
        model = model.to(infer_dtype)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    enc = get_tokenizer()

    gen_kwargs = dict(
        model=model,
        enc=enc,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=device,
        stream=not args.no_stream,
    )

    # ── Single prompt mode ────────────────────────────────────────────────────
    if args.prompt is not None:
        generate(prompt=args.prompt, **gen_kwargs)
        return

    # ── Interactive mode ──────────────────────────────────────────────────────
    print("\nHybrid-MoE-1B — Interactive inference")
    print(f"  checkpoint:   {ckpt_path}")
    print(f"  device:       {device}  |  dtype: {args.dtype}")
    print(f"  temperature:  {args.temperature}  |  top_p: {args.top_p}  "
          f"|  rep_penalty: {args.repetition_penalty}")
    print("  Type your prompt and press Enter. Ctrl+C or 'quit' to exit.\n")

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

        generate(prompt=prompt, **gen_kwargs)
        print()


if __name__ == "__main__":
    main()
