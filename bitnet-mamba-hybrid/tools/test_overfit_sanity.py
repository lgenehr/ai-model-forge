#!/usr/bin/env python3
"""
FASE 3: TESTE DE OVERFITTING EM MINI-DATASET (TESTE MAIS CRÍTICO)

Este é o teste definitivo para validar se a arquitetura funciona.
Se o modelo NÃO conseguir decorar 3 frases, a arquitetura está quebrada.
Se conseguir, o problema está nos hyperparameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import math
from datetime import datetime

# Import model architecture from train script
import sys
sys.path.insert(0, '.')

# Import necessário pois o arquivo tem hífen no nome
import importlib.util
spec = importlib.util.spec_from_file_location("train_module", "train_hybrid-mamba-bitnet.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

ModelConfig = train_module.ModelConfig
BitNetMambaLM = train_module.BitNetMambaLM

def test_overfit_capability():
    """
    Teste definitivo: tentar overfit em 3 frases.

    Objetivo: Loss < 0.5 em 500 epochs
    Se falhar: Arquitetura está quebrada
    Se passar: Problema está nos hyperparameters do treino principal
    """
    print("=" * 80)
    print("OVERFIT SANITY TEST - THE MOST CRITICAL TEST")
    print("=" * 80)
    print()
    print("Goal: Model must memorize 3 simple sentences")
    print("Success: Loss < 0.5 after 500 epochs")
    print("Failure: Architecture is broken")
    print()

    # Mini dataset - 3 frases simples em português
    mini_texts = [
        "O Brasil é um país grande.",
        "São Paulo tem muitas pessoas.",
        "O futebol é popular no Brasil."
    ] * 5  # Repeat 5x for a total of 15 samples

    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize
    print("Tokenizing mini dataset...")
    tokenized = []
    for text in mini_texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) < 20:  # Pad to at least 20 tokens
            tokens = tokens + [tokenizer.eos_token_id] * (20 - len(tokens))
        tokenized.append(tokens[:20])  # Limit to 20 tokens

    # Convert to tensor
    input_ids = torch.tensor(tokenized, dtype=torch.long)
    print(f"Dataset shape: {input_ids.shape}")
    print(f"Sample tokens: {input_ids[0].tolist()}")
    print()

    # SMALL model config (para teste rápido)
    print("Creating SMALL model for overfitting test...")
    config = ModelConfig(
        vocab_size=50257,
        d_model=256,  # MUITO menor que 1280
        n_layers=4,   # MUITO menor que 14
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.0,  # SEM dropout para overfit
        max_seq_len=20,
        use_gradient_checkpointing=False
    )

    model = BitNetMambaLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    input_ids = input_ids.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Device: {device}")
    print()

    # Optimizer - configuração agressiva para overfit
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,  # LR alto para overfit rápido
        betas=(0.9, 0.95),
        weight_decay=0.0  # SEM weight decay
    )

    # Training loop
    print("=" * 80)
    print("TRAINING - Target: Loss < 0.5")
    print("=" * 80)
    print()

    max_epochs = 500
    log_interval = 50
    success_threshold = 0.5

    start_time = datetime.now()
    best_loss = float('inf')
    success = False

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, labels=input_ids)
        loss = outputs['loss']

        # Backward pass
        loss.backward()

        # Compute gradient norm BEFORE clipping
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # Gradient clipping (relaxed)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        # Track best
        if loss.item() < best_loss:
            best_loss = loss.item()

        # Check success
        if loss.item() < success_threshold:
            success = True
            print(f"\n🎉 SUCCESS at epoch {epoch}!")
            print(f"   Loss: {loss.item():.6f} < {success_threshold}")
            print(f"   Grad norm: {total_norm:.4f}")
            break

        # Logging
        if (epoch + 1) % log_interval == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"Epoch {epoch+1:4d}/{max_epochs} | "
                  f"Loss: {loss.item():.6f} | "
                  f"Best: {best_loss:.6f} | "
                  f"GradNorm: {total_norm:.4f} | "
                  f"Time: {elapsed:.1f}s")

            # Early warning if not improving
            if epoch >= 200 and best_loss > 2.0:
                print("   ⚠️  WARNING: Loss not decreasing fast enough")
                print("   → Model may have architecture issues")

    print()
    print("=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print()
    print(f"Best loss achieved: {best_loss:.6f}")
    print(f"Success threshold:  {success_threshold}")
    print()

    if success:
        print("✅ TEST PASSED - Architecture is FUNCTIONAL")
        print()
        print("Conclusion:")
        print("  - Model CAN learn (architecture works)")
        print("  - Main training issues are likely:")
        print("    1. Learning rate too low")
        print("    2. Gradient clipping too aggressive")
        print("    3. Weight decay too high")
        print("    4. Numerical instability in SSM at scale")
        print()
        print("Recommended action: RECOVER training with adjusted hyperparameters")
    else:
        print("❌ TEST FAILED - Architecture may be BROKEN")
        print()
        print(f"Model could not overfit simple dataset after {max_epochs} epochs")
        print("Possible causes:")
        print("  1. BitNet quantization preventing learning")
        print("  2. Mamba SSM not converging")
        print("  3. Gradient flow issues")
        print("  4. Numerical instability")
        print()
        print("Recommended action: RESTART with architecture fixes")

    print()
    print("=" * 80)

    # Test generation from memorized sequence
    if success:
        print("\nTesting generation from memorized sequence...")
        print("-" * 80)
        model.eval()

        # Take first 5 tokens as prompt
        prompt = input_ids[0, :5].unsqueeze(0)
        print(f"Prompt tokens: {prompt[0].tolist()}")
        print(f"Prompt text: {tokenizer.decode(prompt[0])}")
        print()

        # Generate
        with torch.no_grad():
            generated = prompt.clone()
            for _ in range(10):  # Generate 10 tokens
                outputs = model(generated)
                logits = outputs['logits']
                next_token = logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
                generated = torch.cat([generated, next_token], dim=1)

        print(f"Generated text: {tokenizer.decode(generated[0])}")
        print()

    return success, best_loss


if __name__ == "__main__":
    try:
        success, best_loss = test_overfit_capability()

        if success:
            print("\n✅ ARCHITECTURE VALIDATED - PROCEED TO HYPERPARAMETER TUNING")
            exit(0)
        else:
            print("\n❌ ARCHITECTURE BROKEN - REQUIRES FIXES")
            exit(1)

    except Exception as e:
        print(f"\n❌ TEST CRASHED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
