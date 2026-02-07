#!/usr/bin/env python3
"""
RECOVERY VALIDATION SCRIPT

Monitora o progresso da recuperação e valida se está funcionando.
Execute periodicamente durante o treino de recuperação.
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

def validate_recovery():
    """Valida se a recuperação está funcionando"""
    print("=" * 80)
    print("RECOVERY VALIDATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load loss history
    csv_path = Path("model_204m/loss_history.csv")
    if not csv_path.exists():
        print("❌ Loss history not found!")
        return False

    df = pd.read_csv(csv_path)
    recovery_start_step = 28438

    # Check if recovery has started
    if df['step'].max() <= recovery_start_step:
        print("⚠️  Recovery has not started yet")
        print(f"   Current step: {df['step'].max()}")
        print(f"   Expected: > {recovery_start_step}")
        return False

    # Get recovery data
    recovery_df = df[df['step'] > recovery_start_step]
    pre_recovery_df = df[df['step'] <= recovery_start_step]

    recovery_steps = len(recovery_df)
    print(f"Recovery steps completed: {recovery_steps}")
    print()

    # 1. LOSS TREND
    print("1. LOSS ANALYSIS:")
    print("-" * 80)

    pre_recovery_loss = pre_recovery_df['loss'].tail(100).mean()
    current_loss = recovery_df['loss'].tail(100).mean() if len(recovery_df) >= 100 else recovery_df['loss'].mean()

    print(f"  Before recovery:  {pre_recovery_loss:.4f}")
    print(f"  Current:          {current_loss:.4f}")
    print(f"  Change:           {current_loss - pre_recovery_loss:+.4f}")
    print()

    # Check milestones
    if recovery_steps >= 1000:
        loss_1k = recovery_df.head(1000)['loss'].tail(100).mean()
        print(f"  After 1000 steps: {loss_1k:.4f} (target: < 4.0)")
        if loss_1k < 4.0:
            print("  ✅ 1000-step milestone ACHIEVED")
        else:
            print("  ⚠️  1000-step milestone NOT reached yet")
        print()

    if recovery_steps >= 5000:
        loss_5k = recovery_df.head(5000)['loss'].tail(100).mean()
        print(f"  After 5000 steps: {loss_5k:.4f} (target: < 3.5)")
        if loss_5k < 3.5:
            print("  ✅ 5000-step milestone ACHIEVED")
        else:
            print("  ⚠️  5000-step milestone NOT reached yet")
        print()

    # 2. GRADIENT HEALTH
    print("2. GRADIENT HEALTH:")
    print("-" * 80)

    # Try to find gradient norm in logs
    log_path = Path("model_204m/training.log")
    if log_path.exists():
        import re
        recent_grads = []

        with open(log_path, 'r') as f:
            for line in f:
                match = re.search(r'GradNorm:\s*([\d.]+)', line)
                if match:
                    recent_grads.append(float(match.group(1)))

        if recent_grads:
            recent_grad_avg = sum(recent_grads[-100:]) / min(len(recent_grads), 100)
            print(f"  Recent GradNorm avg: {recent_grad_avg:.4f}")

            if recent_grad_avg < 0.4:
                print("  ⚠️  GradNorm VERY LOW (< 0.4) - still too aggressive clipping?")
            elif recent_grad_avg < 0.5:
                print("  ⚠️  GradNorm LOW (< 0.5)")
            elif recent_grad_avg <= 2.0:
                print("  ✅ GradNorm HEALTHY (0.5-2.0)")
            else:
                print("  ⚠️  GradNorm HIGH (> 2.0) - may need investigation")
        else:
            print("  ⚠️  No GradNorm data found in logs")
    else:
        print("  ⚠️  Training log not found")
    print()

    # 3. STABILITY CHECK
    print("3. STABILITY CHECK:")
    print("-" * 80)

    # Check for loss spikes in recovery period
    if len(recovery_df) > 10:
        recovery_df_copy = recovery_df.copy()
        recovery_df_copy['loss_change'] = recovery_df_copy['loss'].pct_change()
        spikes = recovery_df_copy[recovery_df_copy['loss_change'] > 0.5]

        if len(spikes) > 0:
            print(f"  ⚠️  Found {len(spikes)} loss spikes (>50% increase) during recovery:")
            for idx, row in spikes.head(5).iterrows():
                prev_idx = recovery_df_copy.index[recovery_df_copy.index.get_loc(idx) - 1]
                prev_loss = recovery_df_copy.loc[prev_idx, 'loss']
                print(f"    Step {row['step']:.0f}: {prev_loss:.4f} -> {row['loss']:.4f}")
        else:
            print("  ✅ No major loss spikes during recovery")
    print()

    # 4. LEARNING RATE
    print("4. LEARNING RATE:")
    print("-" * 80)
    current_lr = recovery_df['lr'].iloc[-1]
    print(f"  Current LR: {current_lr:.2e}")
    print(f"  Expected range: 1e-4 to 1e-6")

    if current_lr > 5e-5:
        print("  ✅ LR is in healthy range")
    else:
        print("  ⚠️  LR is getting low")
    print()

    # 5. OVERALL ASSESSMENT
    print("=" * 80)
    print("OVERALL ASSESSMENT:")
    print("=" * 80)

    success_criteria = []
    warnings = []

    # Check improvements
    if current_loss < pre_recovery_loss:
        improvement_pct = ((pre_recovery_loss - current_loss) / pre_recovery_loss) * 100
        success_criteria.append(f"✅ Loss improved by {improvement_pct:.2f}%")
    else:
        warnings.append(f"⚠️  Loss has not improved yet")

    # Check milestones
    if recovery_steps >= 1000:
        if loss_1k < 4.0:
            success_criteria.append("✅ 1000-step milestone achieved")
        else:
            warnings.append("⚠️  1000-step milestone not achieved")

    if recovery_steps >= 5000:
        if loss_5k < 3.5:
            success_criteria.append("✅ 5000-step milestone achieved")
        else:
            warnings.append("⚠️  5000-step milestone not achieved")

    # Print results
    if success_criteria:
        print("\nSuccesses:")
        for item in success_criteria:
            print(f"  {item}")

    if warnings:
        print("\nWarnings:")
        for item in warnings:
            print(f"  {item}")

    print()

    # Final recommendation
    if len(warnings) == 0 and len(success_criteria) > 0:
        print("🎉 RECOVERY IS WORKING WELL - CONTINUE TRAINING")
        return True
    elif recovery_steps < 1000:
        print("⏳ TOO EARLY TO ASSESS - CONTINUE FOR AT LEAST 1000 STEPS")
        return True
    elif len(warnings) > 0 and recovery_steps >= 5000:
        print("⚠️  RECOVERY MAY NOT BE WORKING - CONSIDER ALTERNATIVES")
        print("   See DIAGNOSTIC_REPORT.md for next steps")
        return False
    else:
        print("⏳ CONTINUE MONITORING - CHECK AGAIN AFTER MORE STEPS")
        return True


if __name__ == "__main__":
    try:
        success = validate_recovery()
        print()
        print("=" * 80)

        if success:
            exit(0)
        else:
            exit(1)

    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
