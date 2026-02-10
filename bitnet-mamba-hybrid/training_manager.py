#!/usr/bin/env python3
"""
HybridTrainingManager for BitNet-Mamba Hybrid Model

A comprehensive, paper-based training management system that monitors
training dynamics and applies conservative, validated policy actions
to optimize learning rate schedules, gradient clipping, and other
hyperparameters during training.

Designed for BitNet b1.58 + Mamba SSM hybrid architecture with
awareness of ternary quantization noise and SSM-specific dynamics.

Author: AI Model Forge
License: MIT
"""

import json
import math
import logging
import os
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingManagerConfig:
    """All tunable parameters for the HybridTrainingManager.

    Defaults are conservative and tuned for a 204M BitNet-Mamba hybrid model
    with eval_interval=500 steps.
    """

    # --- Metric collection ---
    train_window_size: int = 50      # Rolling window for train metrics (steps)
    val_window_size: int = 20        # Rolling window for val metrics (evals)
    clipping_window: int = 100       # Steps to track clipping frequency

    # --- Plateau detection ---
    plateau_patience: int = 10       # Evals without improvement before plateau
    plateau_min_delta: float = 0.02  # Min improvement to count as progress
    plateau_window: int = 10         # Window of evals for variation check

    # --- ReduceLROnPlateau ---
    lr_reduction_factor: float = 0.8   # Max 20% reduction per trigger
    lr_reduction_cooldown: int = 5     # Evals of cooldown after LR reduction

    # --- SGDR (Cosine Annealing with Warm Restarts) ---
    sgdr_trigger_patience: int = 20    # Evals of stagnation before SGDR (2x plateau)
    sgdr_max_lr_fraction: float = 0.65 # Restart to 65% of historical peak LR
    sgdr_cooldown: int = 20            # Evals of cooldown after SGDR

    # --- Clipping-aware policy ---
    clipping_threshold: float = 0.5    # 50% frequency triggers action
    clipping_lr_reduction: float = 0.9 # 10% LR reduction when clipping
    clipping_grad_increase: float = 1.2 # 20% grad_clip increase
    max_grad_clip: float = 2.0         # Absolute maximum grad_clip

    # --- Overfitting detection ---
    overfit_consecutive_evals: int = 5  # Consecutive evals of diverging losses
    overfit_lr_reduction: float = 0.9   # 10% LR reduction on overfit

    # --- Adaptive warmup ---
    warmup_grad_norm_multiplier: float = 5.0  # Extend warmup if norm > 5x median
    warmup_oscillation_multiplier: float = 3.0 # Extend warmup if oscillation > 3x normal
    warmup_extension_fraction: float = 0.1     # Extend by 10% of original warmup
    warmup_max_token_limit: int = 1_000_000_000 # Never apply after 1B tokens

    # --- Safety ---
    max_lr_change_pct: float = 0.20    # Max 20% LR change in a single decision
    min_evals_before_action: int = 3   # Require at least 3 evals before any policy

    # --- BitNet-specific ---
    bitnet_patience_scale: float = 1.5  # Patience multiplied by this for BitNet
    bitnet_min_delta_scale: float = 1.3 # Min delta multiplied for BitNet tolerance

    # --- Logging ---
    log_file: str = "training_manager.log.jsonl"
    log_observations: bool = True       # Log even when no action taken


# =============================================================================
# Enums and Data Classes
# =============================================================================

class TrainingRegime(Enum):
    """Classification of the current training dynamics."""
    UNSTABLE_EXPLORATION = auto()
    HEALTHY_LEARNING = auto()
    NOISY_PLATEAU = auto()
    REAL_PLATEAU = auto()
    CONVERGENCE = auto()
    LATE_OVERFITTING = auto()


class ActionType(Enum):
    """Types of actions the manager can take."""
    LR_REDUCTION = "lr_reduction"
    LR_INCREASE = "lr_increase"       # SGDR warm restart
    GRAD_CLIP_INCREASE = "grad_clip_increase"
    WARMUP_EXTENSION = "warmup_extension"
    OBSERVATION = "observation"        # No action, just logging


@dataclass
class PolicyDecision:
    """A decision proposed by a policy engine."""
    policy_name: str
    action_type: ActionType
    param_name: str               # e.g. "base_lr", "max_grad_norm", "warmup_steps"
    before_value: float
    after_value: float
    justification: str
    priority: int = 0             # Lower = higher priority
    requires_checkpoint: bool = True


@dataclass
class ValidationResult:
    """Result of decision validation."""
    is_valid: bool
    reason: str


# =============================================================================
# MetricCollector
# =============================================================================

class MetricCollector:
    """Collects and stores training metrics in rolling windows.

    Maintains separate windows for step-level (train) and eval-level
    (validation) metrics. Provides trend analysis via simple linear
    regression over configurable window sizes.
    """

    def __init__(self, config: TrainingManagerConfig):
        self.config = config

        # Step-level metrics (recorded every optimizer step)
        self.loss: Deque[float] = deque(maxlen=config.train_window_size)
        self.lr: Deque[float] = deque(maxlen=config.train_window_size)
        self.grad_norm: Deque[float] = deque(maxlen=config.train_window_size)
        self.grad_norm_bitlinear: Deque[float] = deque(maxlen=config.train_window_size)
        self.grad_norm_ssm: Deque[float] = deque(maxlen=config.train_window_size)
        self.grad_norm_embedding: Deque[float] = deque(maxlen=config.train_window_size)
        self.tokens_per_sec: Deque[float] = deque(maxlen=config.train_window_size)

        # Clipping frequency tracking (binary: was grad clipped this step?)
        self.clipping_events: Deque[int] = deque(maxlen=config.clipping_window)

        # Eval-level metrics
        self.val_loss: Deque[float] = deque(maxlen=config.val_window_size)
        self.train_loss_at_eval: Deque[float] = deque(maxlen=config.val_window_size)

        # Monotonic counters
        self.total_steps_recorded: int = 0
        self.total_evals_recorded: int = 0

    def record_step(
        self,
        loss: float,
        grad_norm: float,
        grad_stats: Dict[str, float],
        lr: float,
        tokens_per_sec: float = 0.0,
        grad_clip_value: float = 1.0,
    ):
        """Record metrics from a single optimizer step.

        Args:
            loss: Training loss for this step.
            grad_norm: Total gradient norm (before clipping).
            grad_stats: Dict from _compute_gradient_stats().
            lr: Current base learning rate.
            tokens_per_sec: Throughput for this step.
            grad_clip_value: Current max_grad_norm for clipping detection.
        """
        self.loss.append(loss)
        self.lr.append(lr)
        self.grad_norm.append(grad_norm)
        self.grad_norm_bitlinear.append(grad_stats.get('grad_norm_bitlinear', 0.0))
        self.grad_norm_ssm.append(grad_stats.get('grad_norm_ssm', 0.0))
        self.grad_norm_embedding.append(grad_stats.get('grad_norm_embedding', 0.0))
        self.tokens_per_sec.append(tokens_per_sec)

        # Detect clipping: grad_norm within 5% of clip value means it was clipped
        was_clipped = 1 if grad_norm >= grad_clip_value * 0.95 else 0
        self.clipping_events.append(was_clipped)

        self.total_steps_recorded += 1

    def record_eval(self, val_loss: float, train_loss: float):
        """Record metrics from an evaluation cycle.

        Args:
            val_loss: Validation loss.
            train_loss: Most recent training loss at time of eval.
        """
        self.val_loss.append(val_loss)
        self.train_loss_at_eval.append(train_loss)
        self.total_evals_recorded += 1

    def get_window(self, metric_name: str, size: Optional[int] = None) -> List[float]:
        """Get the last `size` values of a metric.

        Args:
            metric_name: Name of the metric attribute.
            size: Number of values to return. None = all available.

        Returns:
            List of float values (most recent last).
        """
        data = getattr(self, metric_name, None)
        if data is None:
            return []
        values = list(data)
        if size is not None:
            values = values[-size:]
        return values

    def get_trend(self, metric_name: str, size: Optional[int] = None) -> float:
        """Compute linear trend (slope) of a metric over a window.

        A negative trend for loss means loss is decreasing (good).
        Uses least-squares linear regression.

        Args:
            metric_name: Name of the metric attribute.
            size: Window size. None = use all available data.

        Returns:
            Slope of the linear fit. Returns 0.0 if insufficient data.
        """
        values = self.get_window(metric_name, size)
        if len(values) < 3:
            return 0.0

        n = len(values)
        x = np.arange(n, dtype=np.float64)
        y = np.array(values, dtype=np.float64)

        # Simple linear regression: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
        sum_x = x.sum()
        sum_y = y.sum()
        sum_xy = (x * y).sum()
        sum_x2 = (x * x).sum()
        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) < 1e-12:
            return 0.0
        return float((n * sum_xy - sum_x * sum_y) / denom)

    def get_clipping_frequency(self) -> float:
        """Get fraction of recent steps where gradient clipping was triggered.

        Returns:
            Float between 0.0 and 1.0.
        """
        if len(self.clipping_events) == 0:
            return 0.0
        return sum(self.clipping_events) / len(self.clipping_events)

    def get_std(self, metric_name: str, size: Optional[int] = None) -> float:
        """Get standard deviation of a metric over a window.

        Args:
            metric_name: Name of the metric attribute.
            size: Window size.

        Returns:
            Standard deviation, or 0.0 if insufficient data.
        """
        values = self.get_window(metric_name, size)
        if len(values) < 2:
            return 0.0
        return float(np.std(values, ddof=1))

    def get_mean(self, metric_name: str, size: Optional[int] = None) -> float:
        """Get mean of a metric over a window.

        Args:
            metric_name: Name of the metric attribute.
            size: Window size.

        Returns:
            Mean value, or 0.0 if no data.
        """
        values = self.get_window(metric_name, size)
        if len(values) == 0:
            return 0.0
        return float(np.mean(values))

    def get_variation(self, metric_name: str, size: Optional[int] = None) -> float:
        """Get max - min of a metric over a window.

        Args:
            metric_name: Name of the metric attribute.
            size: Window size.

        Returns:
            Variation (max - min), or 0.0 if insufficient data.
        """
        values = self.get_window(metric_name, size)
        if len(values) < 2:
            return 0.0
        return float(max(values) - min(values))


# =============================================================================
# StateAnalyzer
# =============================================================================

class StateAnalyzer:
    """Classifies the current training regime based on collected metrics.

    The classification determines which policies are eligible to act.
    """

    def __init__(self, config: TrainingManagerConfig):
        self.config = config

        # Thresholds for classification
        self.loss_variance_high = 0.5      # High variance in loss window
        self.grad_variance_high = 2.0      # High variance in grad_norm window
        self.noisy_std_threshold = 0.01    # Std above this = noisy (not flat)

    def classify(self, metrics: MetricCollector) -> TrainingRegime:
        """Classify the current training regime.

        Checks are ordered from most severe to least:
        1. UNSTABLE_EXPLORATION (dangerous)
        2. LATE_OVERFITTING (action needed)
        3. REAL_PLATEAU (action needed)
        4. NOISY_PLATEAU (patience)
        5. CONVERGENCE (near end)
        6. HEALTHY_LEARNING (default good state)

        Args:
            metrics: MetricCollector with recent history.

        Returns:
            TrainingRegime enum value.
        """
        # Need minimum data to classify
        if metrics.total_steps_recorded < 10:
            return TrainingRegime.HEALTHY_LEARNING

        # 1. UNSTABLE_EXPLORATION: high loss or grad variance
        loss_std = metrics.get_std('loss')
        grad_std = metrics.get_std('grad_norm')
        loss_mean = metrics.get_mean('loss')

        # Coefficient of variation for loss (std/mean) handles scale differences
        loss_cv = loss_std / max(loss_mean, 1e-6)
        if loss_cv > 0.3 or grad_std > self.grad_variance_high:
            return TrainingRegime.UNSTABLE_EXPLORATION

        # 2. LATE_OVERFITTING: train_loss decreasing but val_loss increasing
        if metrics.total_evals_recorded >= self.config.overfit_consecutive_evals:
            train_losses = metrics.get_window('train_loss_at_eval',
                                               self.config.overfit_consecutive_evals)
            val_losses = metrics.get_window('val_loss',
                                             self.config.overfit_consecutive_evals)
            if len(train_losses) >= self.config.overfit_consecutive_evals and \
               len(val_losses) >= self.config.overfit_consecutive_evals:
                consecutive_overfit = 0
                for i in range(1, len(train_losses)):
                    if train_losses[i] < train_losses[i - 1] and \
                       val_losses[i] > val_losses[i - 1]:
                        consecutive_overfit += 1
                    else:
                        consecutive_overfit = 0
                if consecutive_overfit >= self.config.overfit_consecutive_evals - 1:
                    return TrainingRegime.LATE_OVERFITTING

        # 3. REAL_PLATEAU: val_loss stagnant with low std for extended period
        patience = self.config.plateau_patience
        if metrics.total_evals_recorded >= patience:
            val_variation = metrics.get_variation('val_loss', patience)
            val_std = metrics.get_std('val_loss', patience)
            if val_variation < self.config.plateau_min_delta and \
               val_std < self.noisy_std_threshold:
                return TrainingRegime.REAL_PLATEAU

        # 4. NOISY_PLATEAU: val_loss stagnant but with noise
        if metrics.total_evals_recorded >= patience:
            val_variation = metrics.get_variation('val_loss', patience)
            val_std = metrics.get_std('val_loss', patience)
            if val_variation < self.config.plateau_min_delta and \
               val_std >= self.noisy_std_threshold:
                return TrainingRegime.NOISY_PLATEAU

        # 5. CONVERGENCE: very low LR and stable loss
        if len(metrics.lr) > 0:
            lr_values = metrics.get_window('lr', 10)
            if len(lr_values) >= 5:
                lr_variation = max(lr_values) - min(lr_values)
                loss_variation = metrics.get_variation('loss', 20)
                if lr_variation < 1e-6 and loss_variation < 0.05:
                    return TrainingRegime.CONVERGENCE

        # 6. HEALTHY_LEARNING: loss trending downward, moderate grads
        loss_trend = metrics.get_trend('loss', 30)
        if loss_trend < 0:
            return TrainingRegime.HEALTHY_LEARNING

        # Default to healthy if nothing else matches
        return TrainingRegime.HEALTHY_LEARNING


class BitNetStateAnalyzer(StateAnalyzer):
    """State analyzer with BitNet-specific adjustments.

    BitNet b1.58 models have inherently noisier gradients due to the
    Straight-Through Estimator (STE) for ternary quantization. This
    analyzer adjusts thresholds to be more tolerant of that noise.
    """

    def __init__(self, config: TrainingManagerConfig):
        super().__init__(config)

        # BitNet-specific: scale up thresholds for quantization noise
        self.loss_variance_high *= 1.3  # More tolerant of loss variance
        self.grad_variance_high *= 1.5  # STE adds gradient noise
        self.noisy_std_threshold *= config.bitnet_min_delta_scale

        # Override patience with BitNet scaling
        self._scaled_patience = int(config.plateau_patience * config.bitnet_patience_scale)

    def classify(self, metrics: MetricCollector) -> TrainingRegime:
        """Classify with BitNet-aware thresholds.

        Uses larger patience windows and more tolerant min_delta
        to account for ternary quantization noise.
        """
        # Need minimum data
        if metrics.total_steps_recorded < 10:
            return TrainingRegime.HEALTHY_LEARNING

        # 1. UNSTABLE_EXPLORATION
        loss_std = metrics.get_std('loss')
        grad_std = metrics.get_std('grad_norm')
        loss_mean = metrics.get_mean('loss')

        loss_cv = loss_std / max(loss_mean, 1e-6)
        if loss_cv > 0.3 or grad_std > self.grad_variance_high:
            return TrainingRegime.UNSTABLE_EXPLORATION

        # 2. LATE_OVERFITTING - use scaled patience
        scaled_overfit = max(
            self.config.overfit_consecutive_evals,
            int(self.config.overfit_consecutive_evals * self.config.bitnet_patience_scale)
        )
        if metrics.total_evals_recorded >= scaled_overfit:
            train_losses = metrics.get_window('train_loss_at_eval', scaled_overfit)
            val_losses = metrics.get_window('val_loss', scaled_overfit)
            if len(train_losses) >= scaled_overfit and len(val_losses) >= scaled_overfit:
                consecutive_overfit = 0
                for i in range(1, len(train_losses)):
                    if train_losses[i] < train_losses[i - 1] and \
                       val_losses[i] > val_losses[i - 1]:
                        consecutive_overfit += 1
                    else:
                        consecutive_overfit = 0
                if consecutive_overfit >= scaled_overfit - 1:
                    return TrainingRegime.LATE_OVERFITTING

        # 3. REAL_PLATEAU with scaled patience and tolerant delta
        patience = self._scaled_patience
        scaled_min_delta = self.config.plateau_min_delta * self.config.bitnet_min_delta_scale
        if metrics.total_evals_recorded >= patience:
            val_variation = metrics.get_variation('val_loss', patience)
            val_std = metrics.get_std('val_loss', patience)
            if val_variation < scaled_min_delta and val_std < self.noisy_std_threshold:
                return TrainingRegime.REAL_PLATEAU

        # 4. NOISY_PLATEAU
        if metrics.total_evals_recorded >= patience:
            val_variation = metrics.get_variation('val_loss', patience)
            val_std = metrics.get_std('val_loss', patience)
            if val_variation < scaled_min_delta and val_std >= self.noisy_std_threshold:
                return TrainingRegime.NOISY_PLATEAU

        # 5. CONVERGENCE
        if len(metrics.lr) > 0:
            lr_values = metrics.get_window('lr', 10)
            if len(lr_values) >= 5:
                lr_variation = max(lr_values) - min(lr_values)
                loss_variation = metrics.get_variation('loss', 20)
                if lr_variation < 1e-6 and loss_variation < 0.05:
                    return TrainingRegime.CONVERGENCE

        # 6. HEALTHY_LEARNING
        loss_trend = metrics.get_trend('loss', 30)
        if loss_trend < 0:
            return TrainingRegime.HEALTHY_LEARNING

        return TrainingRegime.HEALTHY_LEARNING


# =============================================================================
# HybridPolicyEngine
# =============================================================================

class HybridPolicyEngine:
    """Contains all paper-based policies for training management.

    Each policy is a method that returns Optional[PolicyDecision].
    Policies are evaluated in priority order; only the highest-priority
    decision is executed per eval cycle.
    """

    def __init__(
        self,
        config: TrainingManagerConfig,
        min_lr: float,
        initial_lr: float,
        warmup_steps: int,
        max_grad_norm: float,
        bitlinear_lr_scale: float,
    ):
        self.config = config
        self.min_lr = min_lr
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.current_warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.bitlinear_lr_scale = bitlinear_lr_scale

        # Tracking state
        self.historical_peak_lr: float = initial_lr
        self.lr_reduction_cooldown_remaining: int = 0
        self.sgdr_cooldown_remaining: int = 0
        self.clipping_cooldown_remaining: int = 0
        self.overfit_cooldown_remaining: int = 0

        # SGDR stagnation counter (increments each eval with no val improvement)
        self.stagnation_counter: int = 0
        self.best_val_loss_for_stagnation: float = float('inf')

        # Overfitting consecutive counter
        self.overfit_consecutive: int = 0

    def update_cooldowns(self):
        """Decrement all cooldown counters by 1 (called per eval)."""
        self.lr_reduction_cooldown_remaining = max(0, self.lr_reduction_cooldown_remaining - 1)
        self.sgdr_cooldown_remaining = max(0, self.sgdr_cooldown_remaining - 1)
        self.clipping_cooldown_remaining = max(0, self.clipping_cooldown_remaining - 1)
        self.overfit_cooldown_remaining = max(0, self.overfit_cooldown_remaining - 1)

    def update_stagnation(self, val_loss: float):
        """Track stagnation for SGDR trigger.

        Args:
            val_loss: Current validation loss.
        """
        if val_loss < self.best_val_loss_for_stagnation - self.config.plateau_min_delta:
            self.best_val_loss_for_stagnation = val_loss
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

    def update_overfitting_counter(self, metrics: MetricCollector):
        """Update consecutive overfitting counter.

        Args:
            metrics: MetricCollector with recent eval history.
        """
        if metrics.total_evals_recorded < 2:
            self.overfit_consecutive = 0
            return

        train_losses = metrics.get_window('train_loss_at_eval', 2)
        val_losses = metrics.get_window('val_loss', 2)

        if len(train_losses) >= 2 and len(val_losses) >= 2:
            train_decreasing = train_losses[-1] < train_losses[-2]
            val_increasing = val_losses[-1] > val_losses[-2]
            if train_decreasing and val_increasing:
                self.overfit_consecutive += 1
            else:
                self.overfit_consecutive = 0
        else:
            self.overfit_consecutive = 0

    def evaluate_all(
        self,
        metrics: MetricCollector,
        regime: TrainingRegime,
        current_lr: float,
        current_step: int,
        total_tokens: int,
    ) -> Optional[PolicyDecision]:
        """Evaluate all policies in priority order and return the winning decision.

        Args:
            metrics: MetricCollector with recent history.
            regime: Current TrainingRegime classification.
            current_lr: Current base learning rate.
            current_step: Global training step.
            total_tokens: Total tokens processed so far.

        Returns:
            PolicyDecision if a policy triggers, None otherwise.
        """
        # Require minimum data before acting
        if metrics.total_evals_recorded < self.config.min_evals_before_action:
            return None

        # Policies in priority order (lower number = higher priority)
        policies = [
            (1, self._policy_adaptive_warmup, current_step, total_tokens, current_lr, metrics),
            (2, self._policy_clipping_aware, current_lr, metrics),
            (3, self._policy_conservative_overfitting, current_lr, metrics, regime),
            (4, self._policy_reduce_lr_on_plateau, current_lr, metrics, regime),
            (5, self._policy_sgdr_partial, current_lr, metrics, regime),
            (6, self._policy_cosine_observation, current_lr, current_step, metrics),
        ]

        for priority, policy_fn, *args in policies:
            try:
                decision = policy_fn(*args)
                if decision is not None and decision.action_type != ActionType.OBSERVATION:
                    decision.priority = priority
                    return decision
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"Policy {policy_fn.__name__} raised exception: {e}"
                )

        return None

    # --- Policy 1: ReduceLROnPlateau ---

    def _policy_reduce_lr_on_plateau(
        self,
        current_lr: float,
        metrics: MetricCollector,
        regime: TrainingRegime,
    ) -> Optional[PolicyDecision]:
        """Reduce learning rate when validation loss plateaus.

        Triggered when regime is REAL_PLATEAU or NOISY_PLATEAU (with
        additional confirmation for noisy case).

        Returns:
            PolicyDecision to reduce LR, or None.
        """
        if self.lr_reduction_cooldown_remaining > 0:
            return None

        if regime not in (TrainingRegime.REAL_PLATEAU, TrainingRegime.NOISY_PLATEAU):
            return None

        # For NOISY_PLATEAU, require longer patience before acting
        if regime == TrainingRegime.NOISY_PLATEAU:
            patience = int(self.config.plateau_patience * 1.5)
            if metrics.total_evals_recorded < patience:
                return None
            val_variation = metrics.get_variation('val_loss', patience)
            if val_variation >= self.config.plateau_min_delta:
                return None

        # Already at min_lr?
        if current_lr <= self.min_lr * 1.01:
            return None

        new_lr = max(current_lr * self.config.lr_reduction_factor, self.min_lr)

        variation = metrics.get_variation(
            'val_loss', self.config.plateau_patience
        )
        return PolicyDecision(
            policy_name="ReduceLROnPlateau",
            action_type=ActionType.LR_REDUCTION,
            param_name="base_lr",
            before_value=current_lr,
            after_value=new_lr,
            justification=(
                f"val_loss stagnated (variation {variation:.4f} over "
                f"{self.config.plateau_patience} evals, "
                f"min_delta={self.config.plateau_min_delta}). "
                f"ReduceLROnPlateau factor={self.config.lr_reduction_factor} applied. "
                f"Cooldown {self.config.lr_reduction_cooldown} evals."
            ),
            priority=4,
            requires_checkpoint=True,
        )

    # --- Policy 2: CosineAnnealing (observation-only) ---

    def _policy_cosine_observation(
        self,
        current_lr: float,
        current_step: int,
        metrics: MetricCollector,
    ) -> Optional[PolicyDecision]:
        """Observation-only: reports current position in cosine schedule.

        Never produces an actionable decision. The existing cosine schedule
        in the Trainer is not overridden.

        Returns:
            None always (observation is logged separately).
        """
        # This policy is observation-only; the audit logger will record
        # the current LR as part of the metrics snapshot. No action needed.
        return None

    # --- Policy 3: SGDR Partial (Cosine Annealing with Warm Restarts) ---

    def _policy_sgdr_partial(
        self,
        current_lr: float,
        metrics: MetricCollector,
        regime: TrainingRegime,
    ) -> Optional[PolicyDecision]:
        """Partial warm restart triggered by prolonged stagnation.

        Only triggers after stagnation_counter >= sgdr_trigger_patience (2x
        normal plateau patience). Restarts LR to 60-70% of historical peak.

        Returns:
            PolicyDecision to increase LR, or None.
        """
        if self.sgdr_cooldown_remaining > 0:
            return None

        if self.stagnation_counter < self.config.sgdr_trigger_patience:
            return None

        if regime not in (TrainingRegime.REAL_PLATEAU, TrainingRegime.NOISY_PLATEAU):
            return None

        # SGDR target: fraction of historical peak
        target_lr = self.historical_peak_lr * self.config.sgdr_max_lr_fraction

        # Don't increase if already above target
        if current_lr >= target_lr:
            return None

        # Don't increase if we're very close to min_lr and have been decaying
        # (convergence phase should not be restarted)
        if current_lr <= self.min_lr * 1.5:
            return None

        return PolicyDecision(
            policy_name="SGDR_Partial",
            action_type=ActionType.LR_INCREASE,
            param_name="base_lr",
            before_value=current_lr,
            after_value=target_lr,
            justification=(
                f"Prolonged stagnation ({self.stagnation_counter} evals without "
                f"improvement > {self.config.plateau_min_delta}). "
                f"SGDR partial restart to {self.config.sgdr_max_lr_fraction*100:.0f}% "
                f"of historical peak LR ({self.historical_peak_lr:.2e}). "
                f"Cooldown {self.config.sgdr_cooldown} evals."
            ),
            priority=5,
            requires_checkpoint=True,
        )

    # --- Policy 4: Adaptive Warmup ---

    def _policy_adaptive_warmup(
        self,
        current_step: int,
        total_tokens: int,
        current_lr: float,
        metrics: MetricCollector,
    ) -> Optional[PolicyDecision]:
        """Extend warmup if early instability is detected.

        Only active during warmup phase and before 1B tokens.
        Extends warmup by 10% if gradient norm spikes or loss oscillates.

        Returns:
            PolicyDecision to extend warmup, or None.
        """
        # Never apply after warmup is complete or after token limit
        if current_step >= self.current_warmup_steps:
            return None
        if total_tokens >= self.config.warmup_max_token_limit:
            return None

        # Need enough data to detect instability
        if metrics.total_steps_recorded < 20:
            return None

        grad_norms = metrics.get_window('grad_norm', 20)
        if len(grad_norms) < 10:
            return None

        median_grad = float(np.median(grad_norms))
        current_grad = grad_norms[-1] if grad_norms else 0.0

        # Check 1: grad_norm spike > 5x median
        grad_spike = current_grad > self.config.warmup_grad_norm_multiplier * median_grad

        # Check 2: loss oscillation > 3x normal
        losses = metrics.get_window('loss', 20)
        if len(losses) >= 10:
            first_half_std = float(np.std(losses[:len(losses)//2], ddof=1)) if len(losses) >= 4 else 0.0
            second_half_std = float(np.std(losses[len(losses)//2:], ddof=1)) if len(losses) >= 4 else 0.0
            baseline_std = max(first_half_std, 0.01)
            loss_oscillation = second_half_std > self.config.warmup_oscillation_multiplier * baseline_std
        else:
            loss_oscillation = False

        if not (grad_spike or loss_oscillation):
            return None

        # Extend warmup by 10%
        extension = int(self.warmup_steps * self.config.warmup_extension_fraction)
        new_warmup = self.current_warmup_steps + extension

        reason_parts = []
        if grad_spike:
            reason_parts.append(
                f"grad_norm spike ({current_grad:.2f} > "
                f"{self.config.warmup_grad_norm_multiplier}x median {median_grad:.2f})"
            )
        if loss_oscillation:
            reason_parts.append("loss oscillation detected")

        return PolicyDecision(
            policy_name="AdaptiveWarmup",
            action_type=ActionType.WARMUP_EXTENSION,
            param_name="warmup_steps",
            before_value=float(self.current_warmup_steps),
            after_value=float(new_warmup),
            justification=(
                f"Early instability: {'; '.join(reason_parts)}. "
                f"Extending warmup by {extension} steps "
                f"({self.current_warmup_steps} -> {new_warmup})."
            ),
            priority=1,
            requires_checkpoint=False,
        )

    # --- Policy 5: BitNet Differential LR (passive) ---
    # This is handled automatically by ActionExecutor.adjust_lr() which
    # preserves lr_scale ratios across all param groups. No explicit
    # policy decision is needed - it's built into the execution layer.

    # --- Policy 6: Clipping-Aware BitNet ---

    def _policy_clipping_aware(
        self,
        current_lr: float,
        metrics: MetricCollector,
    ) -> Optional[PolicyDecision]:
        """React to frequent gradient clipping.

        If clipping frequency > 50% over recent window, take ONE action:
        - Option A: Reduce LR by 10% (preferred if LR > 2x min_lr)
        - Option B: Increase grad_clip by 20% (only if LR already low)

        Returns:
            PolicyDecision to reduce LR or increase grad_clip, or None.
        """
        if self.clipping_cooldown_remaining > 0:
            return None

        clip_freq = metrics.get_clipping_frequency()
        if clip_freq < self.config.clipping_threshold:
            return None

        # Need enough data in the clipping window
        if len(metrics.clipping_events) < self.config.clipping_window * 0.5:
            return None

        # Option A: Reduce LR (preferred if LR is not already very low)
        if current_lr > self.min_lr * 2.0:
            new_lr = max(current_lr * self.config.clipping_lr_reduction, self.min_lr)
            return PolicyDecision(
                policy_name="ClippingAwareBitNet",
                action_type=ActionType.LR_REDUCTION,
                param_name="base_lr",
                before_value=current_lr,
                after_value=new_lr,
                justification=(
                    f"Gradient clipping frequency {clip_freq:.1%} > "
                    f"{self.config.clipping_threshold:.0%} threshold over last "
                    f"{len(metrics.clipping_events)} steps. "
                    f"LR reduction (Option A) preferred since LR ({current_lr:.2e}) > "
                    f"2x min_lr ({self.min_lr:.2e})."
                ),
                priority=2,
                requires_checkpoint=True,
            )

        # Option B: Increase grad_clip (only if LR is already low)
        if self.max_grad_norm < self.config.max_grad_clip:
            new_clip = min(
                self.max_grad_norm * self.config.clipping_grad_increase,
                self.config.max_grad_clip,
            )
            return PolicyDecision(
                policy_name="ClippingAwareBitNet",
                action_type=ActionType.GRAD_CLIP_INCREASE,
                param_name="max_grad_norm",
                before_value=self.max_grad_norm,
                after_value=new_clip,
                justification=(
                    f"Gradient clipping frequency {clip_freq:.1%} > "
                    f"{self.config.clipping_threshold:.0%} threshold. "
                    f"LR already low ({current_lr:.2e} <= 2x min_lr). "
                    f"Increasing grad_clip (Option B): "
                    f"{self.max_grad_norm:.2f} -> {new_clip:.2f}."
                ),
                priority=2,
                requires_checkpoint=True,
            )

        return None

    # --- Policy 7: Conservative Overfitting Detection ---

    def _policy_conservative_overfitting(
        self,
        current_lr: float,
        metrics: MetricCollector,
        regime: TrainingRegime,
    ) -> Optional[PolicyDecision]:
        """Detect overfitting conservatively and reduce LR mildly.

        Only triggers after 5+ consecutive evals where train_loss decreases
        but val_loss increases. Single eval oscillations are ignored.

        Returns:
            PolicyDecision for mild LR reduction, or None.
        """
        if self.overfit_cooldown_remaining > 0:
            return None

        if regime != TrainingRegime.LATE_OVERFITTING:
            return None

        if self.overfit_consecutive < self.config.overfit_consecutive_evals:
            return None

        if current_lr <= self.min_lr * 1.01:
            return None

        new_lr = max(current_lr * self.config.overfit_lr_reduction, self.min_lr)

        return PolicyDecision(
            policy_name="ConservativeOverfitting",
            action_type=ActionType.LR_REDUCTION,
            param_name="base_lr",
            before_value=current_lr,
            after_value=new_lr,
            justification=(
                f"Overfitting detected: train_loss decreasing but val_loss increasing "
                f"for {self.overfit_consecutive} consecutive evals "
                f"(threshold={self.config.overfit_consecutive_evals}). "
                f"Mild LR reduction: {current_lr:.2e} -> {new_lr:.2e}."
            ),
            priority=3,
            requires_checkpoint=True,
        )


# =============================================================================
# DecisionValidator
# =============================================================================

class DecisionValidator:
    """Validates every policy decision before execution.

    Ensures safety constraints are respected:
    - No LR change > 20% in a single step
    - No LR increase above historical max
    - min_lr is absolute floor
    - grad_clip max is 2.0
    - Cooldown periods are respected
    - Only ONE policy action per eval cycle (enforced by caller)
    """

    def __init__(
        self,
        config: TrainingManagerConfig,
        min_lr: float,
        initial_lr: float,
    ):
        self.config = config
        self.min_lr = min_lr
        self.historical_max_lr: float = initial_lr

    def update_historical_max(self, lr: float):
        """Track the highest LR ever used during training.

        Args:
            lr: Current base learning rate.
        """
        self.historical_max_lr = max(self.historical_max_lr, lr)

    def validate(self, decision: PolicyDecision) -> ValidationResult:
        """Validate a policy decision.

        Args:
            decision: The proposed PolicyDecision.

        Returns:
            ValidationResult with is_valid and reason.
        """
        if decision.action_type == ActionType.OBSERVATION:
            return ValidationResult(True, "Observation, no action needed.")

        if decision.action_type == ActionType.WARMUP_EXTENSION:
            # Warmup extensions are always safe
            return ValidationResult(True, "Warmup extension is safe.")

        if decision.action_type in (ActionType.LR_REDUCTION, ActionType.LR_INCREASE):
            return self._validate_lr_change(decision)

        if decision.action_type == ActionType.GRAD_CLIP_INCREASE:
            return self._validate_grad_clip_change(decision)

        return ValidationResult(False, f"Unknown action type: {decision.action_type}")

    def _validate_lr_change(self, decision: PolicyDecision) -> ValidationResult:
        """Validate a learning rate change.

        Rules:
        - Change must be <= max_lr_change_pct (20%)
        - New LR must be >= min_lr
        - New LR must be <= historical_max_lr (for increases)
        """
        old_lr = decision.before_value
        new_lr = decision.after_value

        # min_lr floor
        if new_lr < self.min_lr:
            return ValidationResult(
                False,
                f"New LR ({new_lr:.2e}) would be below min_lr ({self.min_lr:.2e})."
            )

        # Max change percentage
        if old_lr > 0:
            change_pct = abs(new_lr - old_lr) / old_lr
            if change_pct > self.config.max_lr_change_pct + 1e-6:
                return ValidationResult(
                    False,
                    f"LR change ({change_pct:.1%}) exceeds max allowed "
                    f"({self.config.max_lr_change_pct:.1%}). "
                    f"Before: {old_lr:.2e}, After: {new_lr:.2e}."
                )

        # No increase above historical max
        if decision.action_type == ActionType.LR_INCREASE:
            if new_lr > self.historical_max_lr:
                return ValidationResult(
                    False,
                    f"New LR ({new_lr:.2e}) exceeds historical max "
                    f"({self.historical_max_lr:.2e}). Not allowed."
                )

        return ValidationResult(True, "LR change within safe bounds.")

    def _validate_grad_clip_change(self, decision: PolicyDecision) -> ValidationResult:
        """Validate a gradient clip change.

        Rule: grad_clip must not exceed max_grad_clip (2.0).
        """
        new_clip = decision.after_value
        if new_clip > self.config.max_grad_clip:
            return ValidationResult(
                False,
                f"New grad_clip ({new_clip:.2f}) exceeds maximum "
                f"({self.config.max_grad_clip:.2f})."
            )
        return ValidationResult(True, "Grad clip change within safe bounds.")


# =============================================================================
# ActionExecutor
# =============================================================================

class ActionExecutor:
    """Applies validated decisions to the actual optimizer and trainer.

    All changes are applied immediately (no gradual interpolation for
    simplicity and auditability). Checkpoints are saved BEFORE changes.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def execute(self, decision: PolicyDecision, trainer) -> bool:
        """Execute a validated decision.

        Args:
            decision: The validated PolicyDecision.
            trainer: Reference to the Trainer instance.

        Returns:
            True if execution succeeded, False otherwise.
        """
        try:
            if decision.requires_checkpoint:
                self.force_checkpoint(trainer, reason=decision.policy_name)

            if decision.action_type == ActionType.LR_REDUCTION:
                self.adjust_lr(trainer, decision.after_value)
                return True

            elif decision.action_type == ActionType.LR_INCREASE:
                self.adjust_lr(trainer, decision.after_value)
                return True

            elif decision.action_type == ActionType.GRAD_CLIP_INCREASE:
                self.adjust_grad_clip(trainer, decision.after_value)
                return True

            elif decision.action_type == ActionType.WARMUP_EXTENSION:
                self.extend_warmup(trainer, int(decision.after_value))
                return True

            elif decision.action_type == ActionType.OBSERVATION:
                return True

            else:
                self.logger.warning(f"Unknown action type: {decision.action_type}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to execute decision: {e}")
            return False

    def adjust_lr(self, trainer, new_base_lr: float):
        """Adjust all param groups preserving lr_scale ratios.

        This implements Policy 5 (BitNet Differential LR) transparently:
        the bitlinear_lr_scale multiplier is preserved for all groups.

        Args:
            trainer: Trainer instance with optimizer.
            new_base_lr: New base learning rate (before scaling).
        """
        for param_group in trainer.optimizer.param_groups:
            scale = param_group.get('lr_scale', 1.0)
            param_group['lr'] = new_base_lr * scale

        self.logger.info(
            f"[ActionExecutor] LR adjusted to base={new_base_lr:.2e} "
            f"(bitlinear={new_base_lr * trainer.train_config.bitlinear_lr_scale:.2e})"
        )

    def adjust_grad_clip(self, trainer, new_clip_value: float):
        """Update max_grad_norm on the training config.

        Args:
            trainer: Trainer instance.
            new_clip_value: New maximum gradient norm.
        """
        old_clip = trainer.train_config.max_grad_norm
        trainer.train_config.max_grad_norm = new_clip_value
        self.logger.info(
            f"[ActionExecutor] grad_clip adjusted: {old_clip:.2f} -> {new_clip_value:.2f}"
        )

    def extend_warmup(self, trainer, new_warmup_steps: int):
        """Extend warmup steps in the training config.

        Note: The Trainer's _get_lr() uses train_config.warmup_steps,
        so updating it here affects the cosine schedule seamlessly.

        Args:
            trainer: Trainer instance.
            new_warmup_steps: New warmup step count.
        """
        old_warmup = trainer.train_config.warmup_steps
        trainer.train_config.warmup_steps = new_warmup_steps
        self.logger.info(
            f"[ActionExecutor] Warmup extended: {old_warmup} -> {new_warmup_steps} steps"
        )

    def force_checkpoint(self, trainer, reason: str = "policy_action"):
        """Trigger checkpoint save before making changes.

        Args:
            trainer: Trainer instance.
            reason: Reason for the checkpoint (logged).
        """
        try:
            self.logger.info(
                f"[ActionExecutor] Saving pre-action checkpoint (reason: {reason})"
            )
            trainer._save_checkpoint(is_best=False)
        except Exception as e:
            self.logger.warning(f"[ActionExecutor] Checkpoint save failed: {e}")


# =============================================================================
# AuditLogger
# =============================================================================

class AuditLogger:
    """Writes structured JSON logs for full auditability.

    Each entry is a single JSON line in a .jsonl file. Both action
    entries (when a policy triggers) and observation entries (every eval
    with no action) are logged for complete traceability.
    """

    def __init__(self, log_path: str, log_observations: bool = True):
        self.log_path = log_path
        self.log_observations = log_observations
        self.logger = logging.getLogger(__name__)

        # Ensure parent directory exists
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    def log_entry(self, entry: Dict[str, Any]):
        """Write a single log entry as a JSON line.

        Args:
            entry: Dictionary to serialize as JSON.
        """
        try:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(entry, default=str) + '\n')
        except Exception as e:
            self.logger.warning(f"Failed to write audit log: {e}")

    def log_action(
        self,
        step: int,
        tokens: int,
        regime: TrainingRegime,
        decision: PolicyDecision,
        metrics_snapshot: Dict[str, Any],
        validation_result: ValidationResult,
        checkpoint_saved: bool,
        executed: bool,
    ):
        """Log a policy action (or rejected action).

        Args:
            step: Global training step.
            tokens: Total tokens processed.
            regime: Current training regime.
            decision: The policy decision.
            metrics_snapshot: Current metrics values.
            validation_result: Whether the decision was valid.
            checkpoint_saved: Whether a pre-action checkpoint was saved.
            executed: Whether the action was successfully executed.
        """
        change_pct = 0.0
        if decision.before_value != 0:
            change_pct = (decision.after_value - decision.before_value) / decision.before_value * 100

        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "tokens": tokens,
            "regime": regime.name,
            "policy": decision.policy_name,
            "metrics": metrics_snapshot,
            "action": {
                "type": decision.action_type.value,
                "param": decision.param_name,
                "before": decision.before_value,
                "after": decision.after_value,
                "change_pct": round(change_pct, 2),
            },
            "justification": decision.justification,
            "checkpoint_saved": checkpoint_saved,
            "decision_valid": validation_result.is_valid,
            "validation_reason": validation_result.reason,
            "executed": executed,
        }
        self.log_entry(entry)

    def log_observation(
        self,
        step: int,
        tokens: int,
        regime: TrainingRegime,
        metrics_snapshot: Dict[str, Any],
    ):
        """Log an observation entry (no action taken).

        Args:
            step: Global training step.
            tokens: Total tokens processed.
            regime: Current training regime.
            metrics_snapshot: Current metrics values.
        """
        if not self.log_observations:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "tokens": tokens,
            "regime": regime.name,
            "policy": None,
            "metrics": metrics_snapshot,
            "action": None,
            "justification": "No policy triggered. Training proceeding normally.",
            "checkpoint_saved": False,
            "decision_valid": True,
            "executed": False,
        }
        self.log_entry(entry)


# =============================================================================
# HybridTrainingManager (Main Orchestrator)
# =============================================================================

class HybridTrainingManager:
    """Integrates all training management components.

    Provides hook-based integration with the Trainer class:
    - on_step(): called after each optimizer.step()
    - on_eval(): called after each evaluation
    - on_checkpoint(): called when checkpoints are saved

    The manager is conservative by design: it prioritizes stability over
    aggressiveness and enforces strict safety constraints on all decisions.
    """

    def __init__(
        self,
        trainer,
        config: Optional[TrainingManagerConfig] = None,
    ):
        """Initialize the HybridTrainingManager.

        Args:
            trainer: Reference to the Trainer instance.
            config: Optional configuration overrides.
        """
        self.trainer = trainer
        self.config = config or TrainingManagerConfig()
        self.logger = logging.getLogger(__name__)

        # Extract training parameters from trainer
        tc = trainer.train_config
        self._base_lr = tc.learning_rate
        self._min_lr = tc.min_lr
        self._warmup_steps = tc.warmup_steps
        self._max_grad_norm = tc.max_grad_norm
        self._bitlinear_lr_scale = tc.bitlinear_lr_scale

        # Build components
        self.metrics = MetricCollector(self.config)
        self.analyzer = BitNetStateAnalyzer(self.config)
        self.policy_engine = HybridPolicyEngine(
            config=self.config,
            min_lr=self._min_lr,
            initial_lr=self._base_lr,
            warmup_steps=self._warmup_steps,
            max_grad_norm=self._max_grad_norm,
            bitlinear_lr_scale=self._bitlinear_lr_scale,
        )
        self.validator = DecisionValidator(
            config=self.config,
            min_lr=self._min_lr,
            initial_lr=self._base_lr,
        )
        self.executor = ActionExecutor(self.logger)

        # Audit log path
        log_path = os.path.join(tc.output_dir, self.config.log_file)
        self.audit_logger = AuditLogger(
            log_path=log_path,
            log_observations=self.config.log_observations,
        )

        # Decision history for dashboard/inspection
        self._decision_history: List[Dict[str, Any]] = []

        # Current state
        self._current_regime = TrainingRegime.HEALTHY_LEARNING
        self._current_base_lr = self._base_lr
        self._lr_overridden = False  # True once the manager has set LR

        # Track the last train loss for on_eval
        self._last_train_loss: float = 0.0

        self.logger.info(
            f"[TrainingManager] Initialized with config: "
            f"plateau_patience={self.config.plateau_patience}, "
            f"lr_reduction_factor={self.config.lr_reduction_factor}, "
            f"bitnet_patience_scale={self.config.bitnet_patience_scale}"
        )

    # ----- Hook: on_step -----

    def on_step(
        self,
        step: int,
        loss: float,
        grad_norm: float,
        grad_stats: Dict[str, float],
        lr: float,
        tokens_per_sec: float = 0.0,
    ):
        """Called after each optimizer.step() in the training loop.

        Records step-level metrics. Does NOT trigger policy evaluation
        (that happens only at eval time for stability).

        Args:
            step: Global training step.
            loss: Training loss for this step.
            grad_norm: Total gradient norm (before clipping).
            grad_stats: Dict from _compute_gradient_stats().
            lr: Current base learning rate.
            tokens_per_sec: Throughput for this step.
        """
        self.metrics.record_step(
            loss=loss,
            grad_norm=grad_norm,
            grad_stats=grad_stats,
            lr=lr,
            tokens_per_sec=tokens_per_sec,
            grad_clip_value=self.trainer.train_config.max_grad_norm,
        )

        self._last_train_loss = loss
        self._current_base_lr = lr

        # Update historical peak LR
        self.validator.update_historical_max(lr)
        self.policy_engine.historical_peak_lr = max(
            self.policy_engine.historical_peak_lr, lr
        )

    # ----- Hook: on_eval -----

    def on_eval(
        self,
        step: int,
        val_loss: float,
        train_loss: float,
    ):
        """Called after each evaluation cycle.

        This is the primary decision point. The manager:
        1. Records eval metrics
        2. Classifies the training regime
        3. Evaluates all policies
        4. Validates the winning decision
        5. Executes if valid
        6. Logs everything

        Args:
            step: Global training step.
            val_loss: Validation loss.
            train_loss: Most recent training loss.
        """
        # 1. Record eval metrics
        self.metrics.record_eval(val_loss, train_loss)

        # Update policy tracking
        self.policy_engine.update_cooldowns()
        self.policy_engine.update_stagnation(val_loss)
        self.policy_engine.update_overfitting_counter(self.metrics)

        # 2. Classify regime
        self._current_regime = self.analyzer.classify(self.metrics)

        # 3. Build metrics snapshot for logging
        metrics_snapshot = self._build_metrics_snapshot(val_loss, train_loss)

        # 4. Evaluate policies
        decision = self.policy_engine.evaluate_all(
            metrics=self.metrics,
            regime=self._current_regime,
            current_lr=self._current_base_lr,
            current_step=step,
            total_tokens=self.trainer.total_tokens,
        )

        if decision is not None and decision.action_type != ActionType.OBSERVATION:
            # 5. Validate
            validation = self.validator.validate(decision)

            executed = False
            checkpoint_saved = False

            if validation.is_valid:
                # 6. Execute
                executed = self.executor.execute(decision, self.trainer)
                checkpoint_saved = decision.requires_checkpoint

                if executed:
                    self._apply_post_execution_state(decision)
                    self.logger.info(
                        f"[TrainingManager] Step {step} | Regime: {self._current_regime.name} | "
                        f"Policy: {decision.policy_name} | "
                        f"{decision.param_name}: {decision.before_value:.2e} -> {decision.after_value:.2e}"
                    )
            else:
                self.logger.warning(
                    f"[TrainingManager] Decision REJECTED: {validation.reason}"
                )

            # 7. Log action
            self.audit_logger.log_action(
                step=step,
                tokens=self.trainer.total_tokens,
                regime=self._current_regime,
                decision=decision,
                metrics_snapshot=metrics_snapshot,
                validation_result=validation,
                checkpoint_saved=checkpoint_saved,
                executed=executed,
            )

            # Record in history
            self._decision_history.append({
                "step": step,
                "regime": self._current_regime.name,
                "policy": decision.policy_name,
                "action_type": decision.action_type.value,
                "before": decision.before_value,
                "after": decision.after_value,
                "valid": validation.is_valid,
                "executed": executed,
            })

        else:
            # No action - log observation
            self.audit_logger.log_observation(
                step=step,
                tokens=self.trainer.total_tokens,
                regime=self._current_regime,
                metrics_snapshot=metrics_snapshot,
            )

            self.logger.info(
                f"[TrainingManager] Step {step} | Regime: {self._current_regime.name} | "
                f"No policy triggered."
            )

    # ----- Hook: on_checkpoint -----

    def on_checkpoint(self, step: int, reason: str = "scheduled"):
        """Called when a checkpoint is saved.

        Args:
            step: Global training step.
            reason: Why the checkpoint was saved.
        """
        self.logger.debug(
            f"[TrainingManager] Checkpoint at step {step} (reason: {reason})"
        )

    # ----- Status / Inspection -----

    def get_status(self) -> Dict[str, Any]:
        """Get current manager status for monitoring.

        Returns:
            Dict with regime, metrics summary, and recent decisions.
        """
        return {
            "regime": self._current_regime.name,
            "total_steps_recorded": self.metrics.total_steps_recorded,
            "total_evals_recorded": self.metrics.total_evals_recorded,
            "current_base_lr": self._current_base_lr,
            "clipping_frequency": self.metrics.get_clipping_frequency(),
            "stagnation_counter": self.policy_engine.stagnation_counter,
            "overfit_consecutive": self.policy_engine.overfit_consecutive,
            "cooldowns": {
                "lr_reduction": self.policy_engine.lr_reduction_cooldown_remaining,
                "sgdr": self.policy_engine.sgdr_cooldown_remaining,
                "clipping": self.policy_engine.clipping_cooldown_remaining,
                "overfitting": self.policy_engine.overfit_cooldown_remaining,
            },
            "recent_val_losses": self.metrics.get_window('val_loss', 5),
            "loss_trend": self.metrics.get_trend('loss', 30),
            "val_loss_trend": self.metrics.get_trend('val_loss', 10),
            "last_decisions": self._decision_history[-5:] if self._decision_history else [],
        }

    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Get all decisions made during training.

        Returns:
            List of decision dictionaries.
        """
        return list(self._decision_history)

    # ----- Serialization -----

    def state_dict(self) -> Dict[str, Any]:
        """Serialize manager state for checkpoint saving.

        Returns:
            Dict that can be saved alongside the trainer checkpoint.
        """
        return {
            "config": asdict(self.config),
            "metrics": {
                "loss": list(self.metrics.loss),
                "val_loss": list(self.metrics.val_loss),
                "train_loss_at_eval": list(self.metrics.train_loss_at_eval),
                "lr": list(self.metrics.lr),
                "grad_norm": list(self.metrics.grad_norm),
                "grad_norm_bitlinear": list(self.metrics.grad_norm_bitlinear),
                "grad_norm_ssm": list(self.metrics.grad_norm_ssm),
                "grad_norm_embedding": list(self.metrics.grad_norm_embedding),
                "tokens_per_sec": list(self.metrics.tokens_per_sec),
                "clipping_events": list(self.metrics.clipping_events),
                "total_steps_recorded": self.metrics.total_steps_recorded,
                "total_evals_recorded": self.metrics.total_evals_recorded,
            },
            "policy_state": {
                "historical_peak_lr": self.policy_engine.historical_peak_lr,
                "lr_reduction_cooldown_remaining": self.policy_engine.lr_reduction_cooldown_remaining,
                "sgdr_cooldown_remaining": self.policy_engine.sgdr_cooldown_remaining,
                "clipping_cooldown_remaining": self.policy_engine.clipping_cooldown_remaining,
                "overfit_cooldown_remaining": self.policy_engine.overfit_cooldown_remaining,
                "stagnation_counter": self.policy_engine.stagnation_counter,
                "best_val_loss_for_stagnation": self.policy_engine.best_val_loss_for_stagnation,
                "overfit_consecutive": self.policy_engine.overfit_consecutive,
                "current_warmup_steps": self.policy_engine.current_warmup_steps,
                "max_grad_norm": self.policy_engine.max_grad_norm,
            },
            "validator_state": {
                "historical_max_lr": self.validator.historical_max_lr,
            },
            "decision_history": self._decision_history,
            "current_regime": self._current_regime.name,
            "current_base_lr": self._current_base_lr,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Restore manager state from a checkpoint.

        Args:
            state: Dict from a previous state_dict() call.
        """
        try:
            # Restore metrics
            ms = state.get("metrics", {})
            self.metrics.loss.extend(ms.get("loss", []))
            self.metrics.val_loss.extend(ms.get("val_loss", []))
            self.metrics.train_loss_at_eval.extend(ms.get("train_loss_at_eval", []))
            self.metrics.lr.extend(ms.get("lr", []))
            self.metrics.grad_norm.extend(ms.get("grad_norm", []))
            self.metrics.grad_norm_bitlinear.extend(ms.get("grad_norm_bitlinear", []))
            self.metrics.grad_norm_ssm.extend(ms.get("grad_norm_ssm", []))
            self.metrics.grad_norm_embedding.extend(ms.get("grad_norm_embedding", []))
            self.metrics.tokens_per_sec.extend(ms.get("tokens_per_sec", []))
            self.metrics.clipping_events.extend(ms.get("clipping_events", []))
            self.metrics.total_steps_recorded = ms.get("total_steps_recorded", 0)
            self.metrics.total_evals_recorded = ms.get("total_evals_recorded", 0)

            # Restore policy state
            ps = state.get("policy_state", {})
            self.policy_engine.historical_peak_lr = ps.get(
                "historical_peak_lr", self._base_lr
            )
            self.policy_engine.lr_reduction_cooldown_remaining = ps.get(
                "lr_reduction_cooldown_remaining", 0
            )
            self.policy_engine.sgdr_cooldown_remaining = ps.get(
                "sgdr_cooldown_remaining", 0
            )
            self.policy_engine.clipping_cooldown_remaining = ps.get(
                "clipping_cooldown_remaining", 0
            )
            self.policy_engine.overfit_cooldown_remaining = ps.get(
                "overfit_cooldown_remaining", 0
            )
            self.policy_engine.stagnation_counter = ps.get("stagnation_counter", 0)
            self.policy_engine.best_val_loss_for_stagnation = ps.get(
                "best_val_loss_for_stagnation", float('inf')
            )
            self.policy_engine.overfit_consecutive = ps.get("overfit_consecutive", 0)
            self.policy_engine.current_warmup_steps = ps.get(
                "current_warmup_steps", self._warmup_steps
            )
            self.policy_engine.max_grad_norm = ps.get(
                "max_grad_norm", self._max_grad_norm
            )

            # Restore validator state
            vs = state.get("validator_state", {})
            self.validator.historical_max_lr = vs.get(
                "historical_max_lr", self._base_lr
            )

            # Restore history
            self._decision_history = state.get("decision_history", [])
            regime_name = state.get("current_regime", "HEALTHY_LEARNING")
            self._current_regime = TrainingRegime[regime_name]
            self._current_base_lr = state.get("current_base_lr", self._base_lr)

            self.logger.info(
                f"[TrainingManager] State restored: "
                f"{self.metrics.total_steps_recorded} steps, "
                f"{self.metrics.total_evals_recorded} evals, "
                f"regime={self._current_regime.name}"
            )

        except Exception as e:
            self.logger.warning(
                f"[TrainingManager] Failed to restore state: {e}. "
                f"Starting with fresh manager state."
            )

    # ----- Internal Helpers -----

    def _build_metrics_snapshot(
        self,
        val_loss: float,
        train_loss: float,
    ) -> Dict[str, Any]:
        """Build a metrics snapshot dict for logging.

        Args:
            val_loss: Current validation loss.
            train_loss: Current training loss.

        Returns:
            Dict with key metrics.
        """
        return {
            "val_loss": round(val_loss, 6),
            "val_loss_trend": round(self.metrics.get_trend('val_loss', 10), 6),
            "val_loss_std": round(self.metrics.get_std('val_loss', 10), 6),
            "train_loss": round(train_loss, 6),
            "loss_trend": round(self.metrics.get_trend('loss', 30), 6),
            "grad_norm": round(self.metrics.get_mean('grad_norm', 20), 4),
            "grad_norm_bitlinear": round(self.metrics.get_mean('grad_norm_bitlinear', 20), 4),
            "grad_norm_ssm": round(self.metrics.get_mean('grad_norm_ssm', 20), 4),
            "clipping_freq": round(self.metrics.get_clipping_frequency(), 4),
            "lr_current": self._current_base_lr,
            "stagnation_counter": self.policy_engine.stagnation_counter,
            "overfit_consecutive": self.policy_engine.overfit_consecutive,
        }

    def _apply_post_execution_state(self, decision: PolicyDecision):
        """Update internal state after a decision is executed.

        Sets cooldowns, updates tracking variables, etc.

        Args:
            decision: The executed PolicyDecision.
        """
        if decision.policy_name == "ReduceLROnPlateau":
            self.policy_engine.lr_reduction_cooldown_remaining = self.config.lr_reduction_cooldown
            self._current_base_lr = decision.after_value
            self._lr_overridden = True

        elif decision.policy_name == "SGDR_Partial":
            self.policy_engine.sgdr_cooldown_remaining = self.config.sgdr_cooldown
            self.policy_engine.stagnation_counter = 0  # Reset stagnation
            self._current_base_lr = decision.after_value
            self._lr_overridden = True

        elif decision.policy_name == "ClippingAwareBitNet":
            self.policy_engine.clipping_cooldown_remaining = self.config.lr_reduction_cooldown
            if decision.action_type == ActionType.LR_REDUCTION:
                self._current_base_lr = decision.after_value
                self._lr_overridden = True
            elif decision.action_type == ActionType.GRAD_CLIP_INCREASE:
                self.policy_engine.max_grad_norm = decision.after_value

        elif decision.policy_name == "ConservativeOverfitting":
            self.policy_engine.overfit_cooldown_remaining = self.config.lr_reduction_cooldown
            self.policy_engine.overfit_consecutive = 0  # Reset counter
            self._current_base_lr = decision.after_value
            self._lr_overridden = True

        elif decision.policy_name == "AdaptiveWarmup":
            self.policy_engine.current_warmup_steps = int(decision.after_value)
