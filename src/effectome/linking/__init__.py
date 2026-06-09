"""Linking effectome dynamics to behavior (Stage 6): stats, decoding, lead-lag."""

from .alignment import LeadLagResult, change_signal, lead_lag
from .decoding import (
    DecodeResult,
    community_features,
    connectivity_features,
    decode_behavior,
    state_features,
)
from .stats import (
    AssociationResult,
    association_with_null,
    partition_stability,
    state_behavior_mi,
)

__all__ = [
    "association_with_null",
    "AssociationResult",
    "state_behavior_mi",
    "partition_stability",
    "decode_behavior",
    "DecodeResult",
    "connectivity_features",
    "state_features",
    "community_features",
    "lead_lag",
    "LeadLagResult",
    "change_signal",
]
