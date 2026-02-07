#!/usr/bin/env python3
"""
Early Validation Script for BitNet-Mamba Hybrid Training

This script performs comprehensive validation checks on a trained/training model
to detect potential issues EARLY, before wasting compute on a broken run.

Run this after:
- 1000 steps (~65M tokens)
- 5000 steps (~327M tokens)
- 10000 steps (~655M tokens)

If any check fails, STOP TRAINING and investigate.

Usage:
    python validate_training.py --checkpoint model/checkpoints/checkpoint_00001000.pt
    python validate_training.py --checkpoint model/best_model.pt --verbose
"""

import os
import sys
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def load_checkpoint_for_validation(checkpoint_path: str, device: torch.device):
    """Load checkpoint and extract model + config"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract training state
    global_step = checkpoint.get('global_step', 0)
    total_tokens = checkpoint.get('total_tokens', 0)
    best_loss = checkpoint.get('best_loss', float('inf'))

    # Extract configs
    model_config = checkpoint.get('model_config', {})
    train_config = checkpoint.get('train_config', {})

    return {
        'state_dict': checkpoint.get('model_state_dict', checkpoint),
        'global_step': global_step,
        'total_tokens': total_tokens,
        'best_loss': best_loss,
        'model_config': model_config,
        'train_config': train_config,
    }


def check_weight_statistics(state_dict: Dict[str, torch.Tensor], verbose: bool = False) -> List[str]:
    """
    Check weight statistics for signs of training problems.

    Returns list of warnings/errors.
    """
    issues = []

    for name, param in state_dict.items():
        param = param.float()  # Convert to float for analysis

        # Check for NaN/Inf
        if torch.isnan(param).any():
            issues.append(f"CRITICAL: {name} contains NaN values!")
        if torch.isinf(param).any():
            issues.append(f"CRITICAL: {name} contains Inf values!")

        # Compute statistics
        mean = param.mean().item()
        std = param.std().item()
        min_val = param.min().item()
        max_val = param.max().item()

        if verbose:
            print(f"  {name}: mean={mean:.4f}, std={std:.4f}, min={min_val:.4f}, max={max_val:.4f}")

        # Check for dead weights (all zeros or near-zero variance)
        if std < 1e-7:
            issues.append(f"WARNING: {name} has near-zero variance (std={std:.2e}). May be dead.")

        # Check for exploding weights
        if abs(mean) > 10 or std > 10:
            issues.append(f"WARNING: {name} has large values (mean={mean:.2f}, std={std:.2f})")

        # Special checks for SSM parameters
        if 'A_log' in name:
            # A_log should be in reasonable range [-20, 2] after clamping
            if min_val < -25 or max_val > 5:
                issues.append(f"WARNING: A_log out of expected range [{min_val:.2f}, {max_val:.2f}]")

        if '.D' in name and 'in_proj' not in name and 'out_proj' not in name:
            # D (skip connection) should be near 1.0 initially
            if abs(mean - 1.0) > 2.0:
                issues.append(f"WARNING: D parameter mean ({mean:.2f}) far from initial value 1.0")

    return issues


def check_embedding_statistics(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    """Check embedding layer for learning signals."""
    issues = []

    # Find embedding weights
    emb_key = None
    for key in state_dict.keys():
        if 'embedding.weight' in key:
            emb_key = key
            break

    if emb_key is None:
        issues.append("WARNING: Could not find embedding weights")
        return issues

    emb = state_dict[emb_key].float()

    # Check if embeddings have differentiated (not all same row)
    row_stds = emb.std(dim=1)
    if row_stds.mean() < 0.01:
        issues.append("CRITICAL: Embedding rows have low variance - may not be learning")

    # Check cosine similarity between random pairs
    n_samples = min(100, emb.shape[0])
    indices = torch.randperm(emb.shape[0])[:n_samples]
    sample_emb = F.normalize(emb[indices], dim=1)
    similarities = (sample_emb @ sample_emb.T).abs()
    # Exclude diagonal
    mask = ~torch.eye(n_samples, dtype=torch.bool)
    avg_sim = similarities[mask].mean().item()

    if avg_sim > 0.9:
        issues.append(f"WARNING: Embeddings are too similar (avg cosine sim={avg_sim:.3f}). May have collapsed.")

    return issues


def check_loss_trajectory(best_loss: float, global_step: int, total_tokens: int) -> List[str]:
    """Check if loss is progressing as expected."""
    issues = []

    # Expected loss at various stages (approximate)
    # Random model: ~ln(50304) ≈ 10.8
    # After 1M tokens: < 10
    # After 10M tokens: < 8
    # After 100M tokens: < 6
    # After 1B tokens: < 4

    tokens_M = total_tokens / 1_000_000

    if tokens_M < 10:
        expected_max = 11.0
    elif tokens_M < 50:
        expected_max = 9.0
    elif tokens_M < 200:
        expected_max = 7.5
    elif tokens_M < 500:
        expected_max = 6.0
    elif tokens_M < 1000:
        expected_max = 5.0
    else:
        expected_max = 4.0

    if best_loss > expected_max:
        issues.append(
            f"WARNING: Loss ({best_loss:.2f}) is higher than expected ({expected_max:.1f}) "
            f"at {tokens_M:.1f}M tokens. Model may not be learning effectively."
        )

    # Check for catastrophic values
    if best_loss > 15 and tokens_M > 10:
        issues.append(f"CRITICAL: Loss ({best_loss:.2f}) is very high after {tokens_M:.1f}M tokens!")

    if math.isnan(best_loss) or math.isinf(best_loss):
        issues.append(f"CRITICAL: Loss is {best_loss}! Training has diverged.")

    return issues


def check_bitlinear_quantization(state_dict: Dict[str, torch.Tensor], verbose: bool = False) -> List[str]:
    """Check BitLinear weight distribution for quantization health."""
    issues = []

    for name, param in state_dict.items():
        # Check BitLinear projection weights
        if 'in_proj.weight' in name or 'out_proj.weight' in name:
            param = param.float()

            # Compute what the quantized weights would look like
            scale = param.abs().mean().clamp(min=1e-5)
            normalized = param / scale

            # Check distribution of normalized weights
            # Should be roughly uniform across [-1, 0, 1] bins
            near_neg1 = (normalized < -0.5).float().mean().item()
            near_zero = ((normalized >= -0.5) & (normalized <= 0.5)).float().mean().item()
            near_pos1 = (normalized > 0.5).float().mean().item()

            if verbose:
                print(f"  {name}: -1:{near_neg1:.2%}, 0:{near_zero:.2%}, +1:{near_pos1:.2%}")

            # Check for collapsed distributions
            if near_zero > 0.9:
                issues.append(f"WARNING: {name} has 90%+ weights near zero. May be undertrained.")

            if near_neg1 + near_pos1 < 0.1:
                issues.append(f"WARNING: {name} has <10% non-zero weights. Potential collapse.")

    return issues


def run_generation_test(
    checkpoint_path: str,
    device: torch.device,
    prompts: List[str] = None
) -> List[str]:
    """Run generation test to check for degenerate outputs."""
    issues = []

    if prompts is None:
        prompts = [
            "The capital of France is",
            "In the year 2024",
            "Once upon a time",
            "The quick brown fox",
        ]

    try:
        # Import inference module
        from inference_hybrid import load_model, TextGenerator

        model, tokenizer, config = load_model(checkpoint_path, device=device)
        generator = TextGenerator(model, tokenizer, device)

        print("\nGeneration Test:")
        print("-" * 40)

        for prompt in prompts:
            output = generator.generate(
                prompt,
                max_tokens=30,
                temperature=0.8,
                top_p=0.95
            )

            print(f"Prompt: {prompt}")
            print(f"Output: {output}")
            print()

            # Check for degenerate outputs
            if len(set(output)) < 3:
                issues.append(f"CRITICAL: Repetitive output for '{prompt}': {output[:50]}")

            # Check for empty output
            if len(output.strip()) == 0:
                issues.append(f"WARNING: Empty output for '{prompt}'")

            # Check for pure punctuation
            if output.replace('.', '').replace(',', '').replace(' ', '').strip() == '':
                issues.append(f"WARNING: Output is mostly punctuation for '{prompt}'")

        print("-" * 40)

    except Exception as e:
        issues.append(f"WARNING: Could not run generation test: {e}")

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Validate BitNet-Mamba training checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
VALIDATION CRITERIA:

CRITICAL (stop training immediately):
- NaN or Inf in weights
- Loss > 15 after 10M tokens
- Repetitive/degenerate generation output
- Embedding collapse (all rows similar)

WARNING (investigate):
- Loss higher than expected for token count
- Near-zero variance in any layer
- Quantized weights heavily biased to zero
- A_log parameters out of expected range

EXPECTED LOSS TRAJECTORY:
- Random init: ~10.8
- 10M tokens: < 9.0
- 100M tokens: < 7.0
- 500M tokens: < 5.0
- 2B tokens: < 4.0
- 4B tokens: < 3.5
        """
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed statistics"
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip generation test (faster)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 60)
    print("BitNet-Mamba Training Validation")
    print("=" * 60)

    # Load checkpoint
    checkpoint_data = load_checkpoint_for_validation(args.checkpoint, device)

    print(f"\nCheckpoint Info:")
    print(f"  Global Step: {checkpoint_data['global_step']:,}")
    print(f"  Total Tokens: {checkpoint_data['total_tokens']:,}")
    print(f"  Best Loss: {checkpoint_data['best_loss']:.4f}")
    print()

    all_issues = []

    # Run checks
    print("Checking weight statistics...")
    issues = check_weight_statistics(checkpoint_data['state_dict'], args.verbose)
    all_issues.extend(issues)

    print("Checking embedding statistics...")
    issues = check_embedding_statistics(checkpoint_data['state_dict'])
    all_issues.extend(issues)

    print("Checking loss trajectory...")
    issues = check_loss_trajectory(
        checkpoint_data['best_loss'],
        checkpoint_data['global_step'],
        checkpoint_data['total_tokens']
    )
    all_issues.extend(issues)

    print("Checking BitLinear quantization...")
    issues = check_bitlinear_quantization(checkpoint_data['state_dict'], args.verbose)
    all_issues.extend(issues)

    if not args.skip_generation:
        print("Running generation test...")
        issues = run_generation_test(args.checkpoint, device)
        all_issues.extend(issues)

    # Report results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    critical_issues = [i for i in all_issues if 'CRITICAL' in i]
    warning_issues = [i for i in all_issues if 'WARNING' in i]

    if critical_issues:
        print("\nCRITICAL ISSUES (STOP TRAINING):")
        for issue in critical_issues:
            print(f"  {issue}")

    if warning_issues:
        print("\nWARNINGS (INVESTIGATE):")
        for issue in warning_issues:
            print(f"  {issue}")

    if not all_issues:
        print("\nAll checks passed! Training appears healthy.")
        print("Continue training and re-validate periodically.")

    print("\n" + "=" * 60)

    # Exit with error code if critical issues found
    if critical_issues:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
