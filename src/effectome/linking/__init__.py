"""Linking effectome dynamics to behavior (Stage 6): stats, decoding, lead-lag."""

from .alignment import LeadLagResult, change_signal, lead_lag, manifold_speed, manifold_velocity
from .decoding import (
    DecodeResult,
    IncrementalDecodeResult,
    community_features,
    connectivity_features,
    decode_behavior,
    incremental_decode_behavior,
    purged_blocked_splits,
    state_features,
)
from .stats import (
    AssociationResult,
    ConfidenceInterval,
    association_with_null,
    benjamini_hochberg,
    block_shuffle_null,
    bootstrap_mean_ci,
    circular_shift_null,
    partition_stability,
    state_behavior_mi,
)

__all__ = [
    "AssociationResult",
    "ConfidenceInterval",
    "DecodeResult",
    "IncrementalDecodeResult",
    "LeadLagResult",
    "association_with_null",
    "benjamini_hochberg",
    "block_shuffle_null",
    "bootstrap_mean_ci",
    "change_signal",
    "circular_shift_null",
    "community_features",
    "connectivity_features",
    "decode_behavior",
    "incremental_decode_behavior",
    "lead_lag",
    "manifold_speed",
    "manifold_velocity",
    "partition_stability",
    "purged_blocked_splits",
    "state_behavior_mi",
    "state_features",
]
