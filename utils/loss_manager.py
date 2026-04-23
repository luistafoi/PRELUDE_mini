# utils/loss_manager.py

import torch
from collections import defaultdict


class DynamicLossManager:
    """Adaptive loss weighting that normalizes raw loss magnitudes to target ratios.

    Tracks exponential moving average (EMA) of each loss component's magnitude,
    then computes weights so each loss contributes at its target ratio regardless
    of raw scale. LP weight is anchored at 1.0.

    Features:
        - EMA magnitude tracking per batch
        - Warmup period with flat weights
        - Staleness detection (halves weight if loss barely changes)
    """

    def __init__(self, target_ratios, ema_decay=0.99, warmup_batches=50, max_weight=None):
        """
        Args:
            target_ratios: dict like {"lp": 0.6, "triplet": 0.3, "rw": 0.1}
            ema_decay: decay factor for EMA (higher = smoother)
            warmup_batches: number of batches before adaptive weighting activates
            max_weight: cap on any single weight (relative to LP=1.0). None = uncapped.
        """
        self.target_ratios = target_ratios
        self.ema_decay = ema_decay
        self.warmup_batches = warmup_batches
        self.max_weight = max_weight

        # EMA of |loss| per component
        self.ema = {k: None for k in target_ratios}
        # Adaptive weights (start flat)
        self.weights = {k: 1.0 for k in target_ratios}

        self.batch_count = 0

        # Staleness detection: track epoch-level average loss
        self.epoch_losses = defaultdict(list)  # per-batch values for current epoch
        self.prev_epoch_avg = {}  # previous epoch average
        self.stale_count = {k: 0 for k in target_ratios}
        self.stale_threshold = 1e-6
        self.stale_patience = 5
        self.staleness_penalty = {k: 1.0 for k in target_ratios}

    def combine(self, raw_losses):
        """Combine raw loss tensors with adaptive weights.

        Args:
            raw_losses: dict of {"lp": tensor, "triplet": tensor, ...}
                        Only keys present will be weighted. Missing keys are skipped.

        Returns:
            weighted total loss (tensor)
        """
        self.batch_count += 1

        # Update EMAs
        for k, loss_val in raw_losses.items():
            if k not in self.target_ratios:
                continue
            mag = loss_val.detach().abs().item()
            self.epoch_losses[k].append(mag)

            if self.ema[k] is None:
                self.ema[k] = mag
            else:
                self.ema[k] = self.ema_decay * self.ema[k] + (1 - self.ema_decay) * mag

        # Compute adaptive weights after warmup
        if self.batch_count > self.warmup_batches:
            self._update_weights()

        # Weighted sum
        total = 0.0
        for k, loss_val in raw_losses.items():
            if k in self.weights:
                total = total + self.weights[k] * self.staleness_penalty[k] * loss_val

        return total

    def _update_weights(self):
        """Recompute weights: w_i = target_ratio_i / EMA(|L_i|), anchored so LP weight = 1.0."""
        raw_weights = {}
        for k in self.target_ratios:
            ema_val = self.ema.get(k)
            if ema_val is not None and ema_val > 1e-12:
                raw_weights[k] = self.target_ratios[k] / ema_val
            else:
                raw_weights[k] = 1.0

        # Anchor: normalize so LP weight = 1.0
        lp_w = raw_weights.get("lp", 1.0)
        if lp_w > 1e-12:
            for k in raw_weights:
                self.weights[k] = raw_weights[k] / lp_w
        else:
            self.weights = raw_weights

        # Clamp to max_weight (LP is always 1.0, aux weights get capped)
        if self.max_weight is not None:
            for k in self.weights:
                if k != "lp":
                    self.weights[k] = min(self.weights[k], self.max_weight)

    def step_epoch(self):
        """Call at end of each epoch. Checks staleness and resets epoch accumulators."""
        for k in self.target_ratios:
            batch_vals = self.epoch_losses.get(k, [])
            if not batch_vals:
                continue

            current_avg = sum(batch_vals) / len(batch_vals)
            prev_avg = self.prev_epoch_avg.get(k)

            if prev_avg is not None:
                delta = abs(current_avg - prev_avg)
                if delta < self.stale_threshold:
                    self.stale_count[k] += 1
                else:
                    self.stale_count[k] = 0

                # Halve weight if stale for too long
                if self.stale_count[k] >= self.stale_patience:
                    self.staleness_penalty[k] *= 0.5
                    self.stale_count[k] = 0  # reset counter after penalty

            self.prev_epoch_avg[k] = current_avg

        # Reset epoch accumulators
        self.epoch_losses = defaultdict(list)

    def epoch_summary(self):
        """Return dict of logging info for the current epoch."""
        summary = {}
        for k in self.target_ratios:
            summary[f"ema_{k}"] = self.ema.get(k, 0.0)
            summary[f"weight_{k}"] = self.weights.get(k, 1.0) * self.staleness_penalty.get(k, 1.0)
            summary[f"stale_{k}"] = self.stale_count.get(k, 0)
        summary["warmup_done"] = self.batch_count > self.warmup_batches
        return summary
