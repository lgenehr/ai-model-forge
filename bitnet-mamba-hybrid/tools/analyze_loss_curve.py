#!/usr/bin/env python3
"""
FASE 2: ANÁLISE DE CURVA DE LOSS
Analisa loss_history.csv para detectar estagnação, colapso ou aprendizado lento.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_loss_curve(csv_path):
    """Analisa curva de loss do treino"""
    print("=" * 80)
    print("LOSS CURVE ANALYSIS")
    print("=" * 80)
    print(f"Loading: {csv_path}\n")

    # Carregar dados
    df = pd.read_csv(csv_path)
    print(f"Total training steps: {len(df):,}")
    print(f"Steps range: {df['step'].min():,} -> {df['step'].max():,}")
    print()

    # 1. ESTATÍSTICAS GERAIS
    print("1. OVERALL STATISTICS:")
    print("-" * 80)
    print(f"  Initial loss (first 10 steps):  {df['loss'].head(10).mean():.4f}")
    print(f"  Current loss (last 100 steps):  {df['loss'].tail(100).mean():.4f}")
    print(f"  Minimum loss (global):           {df['loss'].min():.4f} @ step {df.loc[df['loss'].idxmin(), 'step']:.0f}")
    print(f"  Maximum loss:                    {df['loss'].max():.4f}")
    print(f"  Learning rate (current):         {df['lr'].iloc[-1]:.2e}")
    print()

    # 2. TENDÊNCIA RECENTE (últimos 1000 steps)
    print("2. RECENT TREND (last 1000 steps):")
    print("-" * 80)
    recent = df.tail(1000)
    if len(recent) > 100:
        first_100 = recent.head(100)['loss'].mean()
        last_100 = recent.tail(100)['loss'].mean()
        change = last_100 - first_100
        pct_change = (change / first_100) * 100

        print(f"  Loss 1000 steps ago:   {first_100:.4f}")
        print(f"  Loss now:              {last_100:.4f}")
        print(f"  Change:                {change:+.4f} ({pct_change:+.2f}%)")

        if abs(pct_change) < 1.0:
            print("  Status:                ⚠️  STAGNANT (< 1% change)")
        elif change < 0:
            print("  Status:                ✅ IMPROVING")
        else:
            print("  Status:                ❌ DEGRADING")
    print()

    # 3. JANELAS DE ANÁLISE
    print("3. LOSS BY TRAINING PHASE:")
    print("-" * 80)
    phases = [
        (0, 500, "Warmup phase"),
        (500, 2000, "Early training"),
        (2000, 10000, "Mid training"),
        (10000, 20000, "Late training"),
        (20000, None, "Current phase")
    ]

    for start, end, name in phases:
        if end is None:
            phase_data = df[df['step'] >= start]
        else:
            phase_data = df[(df['step'] >= start) & (df['step'] < end)]

        if len(phase_data) > 0:
            avg_loss = phase_data['loss'].mean()
            min_loss = phase_data['loss'].min()
            max_loss = phase_data['loss'].max()
            std_loss = phase_data['loss'].std()
            print(f"  {name:20s}: avg={avg_loss:.4f}, min={min_loss:.4f}, max={max_loss:.4f}, std={std_loss:.4f}")
    print()

    # 4. DETECTAR COLAPSO
    print("4. COLLAPSE DETECTION:")
    print("-" * 80)
    # Procurar por spikes grandes (loss aumenta > 50%)
    df['loss_change'] = df['loss'].pct_change()
    spikes = df[df['loss_change'] > 0.5]

    if len(spikes) > 0:
        print(f"  ⚠️  Found {len(spikes)} loss spikes (>50% increase):")
        for idx, row in spikes.head(10).iterrows():
            print(f"    Step {row['step']:.0f}: {df.loc[idx-1, 'loss']:.4f} -> {row['loss']:.4f}")
    else:
        print("  ✅ No major loss spikes detected")
    print()

    # 5. GRADIENT NORM
    print("5. GRADIENT HEALTH:")
    print("-" * 80)
    if 'GradNorm' in df.columns or any('grad' in col.lower() for col in df.columns):
        # Tentar encontrar coluna de gradient norm
        grad_col = None
        for col in df.columns:
            if 'grad' in col.lower() and 'norm' in col.lower():
                grad_col = col
                break

        if grad_col:
            recent_grad = df[grad_col].tail(1000).mean()
            print(f"  Recent gradient norm (avg): {recent_grad:.4f}")

            if recent_grad < 0.1:
                print("  Status: ⚠️  VERY LOW (< 0.1) - gradients may be clipped too much")
            elif recent_grad < 0.5:
                print("  Status: ⚠️  LOW (< 0.5) - consider relaxing gradient clipping")
            elif recent_grad < 2.0:
                print("  Status: ✅ HEALTHY (0.5-2.0)")
            else:
                print("  Status: ⚠️  HIGH (> 2.0) - may need stronger clipping")
        else:
            print("  ⚠️  Gradient norm not found in CSV")
    else:
        print("  ⚠️  Gradient norm column not found")
    print()

    # 6. THROUGHPUT
    print("6. THROUGHPUT ANALYSIS:")
    print("-" * 80)
    recent_throughput = df['tokens_per_sec'].tail(1000).mean()
    print(f"  Recent throughput: {recent_throughput:,.0f} tokens/sec")

    total_time = pd.to_datetime(df['timestamp'].iloc[-1]) - pd.to_datetime(df['timestamp'].iloc[0])
    print(f"  Total training time: {total_time}")
    print()

    # 7. RECOMENDAÇÕES
    print("=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)

    recommendations = []

    # Verificar estagnação
    if len(recent) > 100:
        if abs(pct_change) < 1.0:
            recommendations.append("❌ Loss is STAGNANT (< 1% change in 1000 steps)")
            recommendations.append("   → Possible causes:")
            recommendations.append("     - Learning rate too low (current: {:.2e})".format(df['lr'].iloc[-1]))
            recommendations.append("     - Gradient clipping too aggressive")
            recommendations.append("     - Model capacity mismatch")
        elif pct_change > 0:
            recommendations.append("❌ Loss is INCREASING (model degrading)")
            recommendations.append("   → Likely causes:")
            recommendations.append("     - Learning rate too high (causing divergence)")
            recommendations.append("     - Optimizer state corrupted")

    # Verificar gradient clipping
    if 'max_grad_norm' in df.columns or recent_grad < 0.5:
        recommendations.append("⚠️  Gradient clipping appears very aggressive")
        recommendations.append("   → Try increasing max_grad_norm from 0.3 to 1.0 or higher")

    # Verificar learning rate
    current_lr = df['lr'].iloc[-1]
    if current_lr < 1e-5:
        recommendations.append("⚠️  Learning rate is very low ({:.2e})".format(current_lr))
        recommendations.append("   → Consider:")
        recommendations.append("     - Restarting with higher LR (1e-4 or 3e-4)")
        recommendations.append("     - Using --weights_only to reset optimizer with new LR")

    # Verificar se loss está razoável
    current_loss = df['loss'].tail(100).mean()
    if current_loss > 7.0:
        recommendations.append("⚠️  Loss is still very high (> 7.0) after {:,} steps".format(len(df)))
        recommendations.append("   → Model may not be learning effectively")
    elif current_loss > 5.0:
        recommendations.append("⚠️  Loss is high (> 5.0) - learning is slow")
    elif current_loss < 3.0:
        recommendations.append("✅ Loss is good (< 3.0) - model is learning well")

    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        print("✅ No major issues detected")

    print()
    print("=" * 80)

    return df

if __name__ == "__main__":
    csv_path = Path("model_204m/loss_history.csv")

    if not csv_path.exists():
        print(f"❌ Loss history not found: {csv_path}")
        exit(1)

    df = analyze_loss_curve(csv_path)
    print("✅ LOSS CURVE ANALYSIS COMPLETE")
