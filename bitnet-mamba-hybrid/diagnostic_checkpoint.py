#!/usr/bin/env python3
"""
FASE 1: DIAGNÓSTICO DE CHECKPOINT
Inspeciona checkpoint_interrupt_00028438.pt para validar integridade,
configurações e estado do modelo.
"""

import torch
import json
from pathlib import Path

def inspect_checkpoint(checkpoint_path):
    """Carrega e inspeciona checkpoint detalhadamente"""
    print("=" * 80)
    print("CHECKPOINT INSPECTION")
    print("=" * 80)
    print(f"Loading: {checkpoint_path}\n")

    try:
        # Carregar checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 1. ESTRUTURA DO CHECKPOINT
        print("1. CHECKPOINT STRUCTURE:")
        print("-" * 80)
        for key in checkpoint.keys():
            if key == 'model_state_dict':
                print(f"  ✓ {key}: {len(checkpoint[key])} parameters")
            elif key == 'optimizer_state_dict':
                print(f"  ✓ {key}: present")
            elif key == 'scheduler_state_dict':
                print(f"  ✓ {key}: present")
            else:
                print(f"  ✓ {key}: {checkpoint[key]}")
        print()

        # 2. MODEL CONFIGURATION
        print("2. MODEL CONFIGURATION:")
        print("-" * 80)
        model_config = checkpoint['model_config']
        for key, value in model_config.items():
            print(f"  {key:30s}: {value}")
        print()

        # 3. TRAINING STATE
        print("3. TRAINING STATE:")
        print("-" * 80)
        print(f"  global_step:         {checkpoint['global_step']:,}")
        print(f"  total_tokens:        {checkpoint['total_tokens']:,}")
        print(f"  best_loss:           {checkpoint['best_loss']:.4f}")
        print(f"  interrupted:         {checkpoint.get('interrupted', False)}")
        print()

        # 4. TRAINING CONFIGURATION
        print("4. TRAINING CONFIGURATION:")
        print("-" * 80)
        train_config = checkpoint['train_config']
        critical_params = [
            'learning_rate', 'min_lr', 'warmup_steps', 'weight_decay',
            'max_grad_norm', 'batch_size', 'gradient_accumulation_steps',
            'max_seq_len'
        ]
        for key in critical_params:
            if key in train_config:
                print(f"  {key:30s}: {train_config[key]}")
        print()

        # 5. OPTIMIZER STATE
        print("5. OPTIMIZER STATE:")
        print("-" * 80)
        opt_state = checkpoint['optimizer_state_dict']
        print(f"  param_groups: {len(opt_state['param_groups'])}")
        for i, group in enumerate(opt_state['param_groups']):
            print(f"\n  Group {i}:")
            print(f"    lr:           {group['lr']:.2e}")
            print(f"    betas:        {group['betas']}")
            print(f"    eps:          {group['eps']}")
            print(f"    weight_decay: {group['weight_decay']}")
            print(f"    params:       {len(group['params'])} tensors")
        print()

        # 6. WEIGHT STATISTICS
        print("6. WEIGHT STATISTICS:")
        print("-" * 80)
        model_state = checkpoint['model_state_dict']

        total_params = 0
        nan_params = 0
        inf_params = 0

        # Estatísticas por tipo de camada
        embedding_stats = {'count': 0, 'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        bitlinear_stats = {'count': 0, 'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        ssm_stats = {'count': 0, 'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        norm_stats = {'count': 0, 'mean': 0, 'std': 0, 'min': 0, 'max': 0}

        for name, param in model_state.items():
            total_params += param.numel()

            # Check for NaN/Inf
            if torch.isnan(param).any():
                nan_params += 1
                print(f"  ⚠️  NaN detected in: {name}")
            if torch.isinf(param).any():
                inf_params += 1
                print(f"  ⚠️  Inf detected in: {name}")

            # Categorizar por tipo
            if 'embedding' in name or 'lm_head' in name:
                embedding_stats['count'] += 1
                embedding_stats['mean'] += param.abs().mean().item()
                embedding_stats['std'] += param.std().item()
                embedding_stats['min'] += param.min().item()
                embedding_stats['max'] += param.max().item()
            elif 'in_proj' in name or 'out_proj' in name:
                bitlinear_stats['count'] += 1
                bitlinear_stats['mean'] += param.abs().mean().item()
                bitlinear_stats['std'] += param.std().item()
                bitlinear_stats['min'] += param.min().item()
                bitlinear_stats['max'] += param.max().item()
            elif 'A_log' in name or '.D' in name or 'dt_proj' in name or 'x_proj' in name:
                ssm_stats['count'] += 1
                ssm_stats['mean'] += param.abs().mean().item()
                ssm_stats['std'] += param.std().item()
                ssm_stats['min'] += param.min().item()
                ssm_stats['max'] += param.max().item()
            elif 'norm' in name or 'weight' in name.split('.')[-1]:
                norm_stats['count'] += 1
                norm_stats['mean'] += param.abs().mean().item()
                norm_stats['std'] += param.std().item()
                norm_stats['min'] += param.min().item()
                norm_stats['max'] += param.max().item()

        print(f"  Total parameters:    {total_params:,}")
        print(f"  Parameters with NaN: {nan_params}")
        print(f"  Parameters with Inf: {inf_params}")
        print()

        # Print stats por categoria
        def print_layer_stats(name, stats):
            if stats['count'] > 0:
                print(f"  {name}:")
                print(f"    Layers:   {stats['count']}")
                print(f"    Mean:     {stats['mean'] / stats['count']:.6f}")
                print(f"    Std:      {stats['std'] / stats['count']:.6f}")
                print(f"    Min:      {stats['min'] / stats['count']:.6f}")
                print(f"    Max:      {stats['max'] / stats['count']:.6f}")

        print_layer_stats("Embeddings", embedding_stats)
        print_layer_stats("BitLinear", bitlinear_stats)
        print_layer_stats("SSM (Mamba)", ssm_stats)
        print_layer_stats("Normalization", norm_stats)
        print()

        # 7. SPECIFIC LAYER INSPECTION (AMOSTRAS)
        print("7. SAMPLE LAYER INSPECTION:")
        print("-" * 80)

        # Primeiro BitLinear weight
        for name, param in model_state.items():
            if 'in_proj.weight' in name and 'layers.0' in name:
                print(f"\n  {name}:")
                print(f"    Shape:        {param.shape}")
                print(f"    Mean:         {param.mean().item():.6f}")
                print(f"    Std:          {param.std().item():.6f}")
                print(f"    Min:          {param.min().item():.6f}")
                print(f"    Max:          {param.max().item():.6f}")

                # Verificar distribuição ternária
                unique_vals = torch.unique(param.round())
                print(f"    Unique (rounded): {unique_vals.tolist()[:10]}...")
                break

        # A_log do primeiro Mamba
        for name, param in model_state.items():
            if 'A_log' in name and 'layers.0' in name:
                print(f"\n  {name}:")
                print(f"    Shape:        {param.shape}")
                print(f"    Mean:         {param.mean().item():.6f}")
                print(f"    Std:          {param.std().item():.6f}")
                print(f"    Min:          {param.min().item():.6f}")
                print(f"    Max:          {param.max().item():.6f}")

                # A = -exp(A_log) deve estar em range razoável
                A = -torch.exp(param.clamp(min=-20, max=2))
                print(f"    A (=-exp(A_log)):")
                print(f"      Mean:       {A.mean().item():.6f}")
                print(f"      Min:        {A.min().item():.6f}")
                print(f"      Max:        {A.max().item():.6f}")
                break

        print()

        # 8. DECISÃO
        print("=" * 80)
        print("CHECKPOINT HEALTH:")
        print("=" * 80)

        issues = []

        if nan_params > 0:
            issues.append(f"❌ CRITICAL: {nan_params} parameters contain NaN")
        if inf_params > 0:
            issues.append(f"❌ CRITICAL: {inf_params} parameters contain Inf")
        if checkpoint['best_loss'] > 10.0:
            issues.append(f"⚠️  WARNING: Best loss ({checkpoint['best_loss']:.2f}) is very high")
        if train_config['max_grad_norm'] < 0.5:
            issues.append(f"⚠️  WARNING: Gradient clipping ({train_config['max_grad_norm']}) may be too aggressive")
        if train_config['learning_rate'] < 5e-5:
            issues.append(f"⚠️  WARNING: Learning rate ({train_config['learning_rate']:.2e}) may be too low")

        if issues:
            print("Issues found:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("✅ No critical issues detected in checkpoint structure")

        print()
        print("=" * 80)

        # Salvar configurações em JSON para referência
        output = {
            'model_config': model_config,
            'train_config': train_config,
            'training_state': {
                'global_step': checkpoint['global_step'],
                'total_tokens': checkpoint['total_tokens'],
                'best_loss': checkpoint['best_loss']
            },
            'health_check': {
                'total_params': total_params,
                'nan_params': nan_params,
                'inf_params': inf_params,
                'issues': issues
            }
        }

        with open('checkpoint_config.json', 'w') as f:
            json.dump(output, f, indent=2)
        print("✅ Configuration saved to checkpoint_config.json")

        return checkpoint, issues

    except Exception as e:
        print(f"❌ ERROR loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None, [f"Failed to load checkpoint: {e}"]


if __name__ == "__main__":
    checkpoint_path = Path("model_204m/checkpoints/checkpoint_interrupt_00028438.pt")

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please check the path and try again.")
        exit(1)

    checkpoint, issues = inspect_checkpoint(checkpoint_path)

    if checkpoint is None:
        print("\n❌ CHECKPOINT INSPECTION FAILED")
        exit(1)

    if any("CRITICAL" in issue for issue in issues):
        print("\n❌ CRITICAL ISSUES DETECTED - CHECKPOINT MAY BE CORRUPTED")
        exit(1)
    else:
        print("\n✅ CHECKPOINT INSPECTION COMPLETE")
        exit(0)
