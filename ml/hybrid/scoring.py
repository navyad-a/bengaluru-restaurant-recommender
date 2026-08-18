# -*- coding: utf-8 -*-
r"""
Hybrid Score Normalization & Dynamic Weight Fusion Module
=========================================================
Normalizes heterogeneous recommendation signals to [0, 1] and computes
the convex combination hybrid score with dynamic cold-start weight redistribution.

Mathematical Formulation:
    S_hybrid = w_content * S_content + w_collab * S_collab + w_location * S_location + w_quality * S_quality
    
    Subject to: \sum_{s \in Available} w_s = 1.0
"""

from typing import Dict, Any, Set, Optional


DEFAULT_HYBRID_WEIGHTS: Dict[str, float] = {
    "content": 0.40,
    "collaborative": 0.20,
    "location": 0.15,
    "quality": 0.25
}


def normalize_content_score(raw_score: float) -> float:
    """Clips cosine similarity to [0.0, 1.0]."""
    return max(0.0, min(1.0, float(raw_score)))


def normalize_collaborative_score(predicted_rating: float) -> float:
    """Maps SVD predicted rating [1.0, 5.0] to [0.0, 1.0]."""
    norm = (float(predicted_rating) - 1.0) / 4.0
    return max(0.0, min(1.0, norm))


def compute_effective_weights(
    base_weights: Optional[Dict[str, float]] = None,
    available_signals: Optional[Set[str]] = None
) -> Dict[str, float]:
    """
    Computes normalized effective weights dynamically based on active recommendation signals.
    
    If a signal (e.g. 'collaborative' or 'location') is absent or cold-start, its weight
    is set to 0.0 and the remaining active weights are scaled proportionally so their sum is 1.0.
    """
    weights = dict(base_weights or DEFAULT_HYBRID_WEIGHTS)
    
    if available_signals is None:
        available_signals = set(weights.keys())

    # Filter active signals
    active_weights: Dict[str, float] = {}
    for signal, w in weights.items():
        if signal in available_signals and w > 0.0:
            active_weights[signal] = float(w)
        else:
            active_weights[signal] = 0.0

    total_active = sum(active_weights[s] for s in available_signals if s in active_weights)
    
    if total_active <= 0.0:
        # Fallback to equal weighting among available signals
        n = len(available_signals) if available_signals else 1
        return {s: round(1.0 / n, 4) for s in available_signals}

    # Normalize active signals to sum to 1.0
    effective: Dict[str, float] = {}
    for signal, w in active_weights.items():
        effective[signal] = round(w / total_active, 4) if w > 0.0 else 0.0

    # Clean small floating-point residual on the primary active signal
    active_sum = sum(effective.values())
    if active_sum != 1.0 and available_signals:
        primary_signal = max(effective, key=effective.get)
        effective[primary_signal] = round(effective[primary_signal] + (1.0 - active_sum), 4)

    return effective


def compute_hybrid_score(
    scores: Dict[str, float],
    effective_weights: Dict[str, float]
) -> float:
    r"""
    Computes the weighted hybrid score S_hybrid = \sum w_s * S_s.
    """
    hybrid = sum(
        effective_weights.get(signal, 0.0) * scores.get(signal, 0.0)
        for signal in effective_weights
    )
    return max(0.0, min(1.0, round(float(hybrid), 4)))
