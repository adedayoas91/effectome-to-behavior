"""Prospective prediction and attribution in a recurrent-coupling control."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from effectome.connectivity import ConnectivityConfig
from effectome.manifold import ManifoldConfig, ManifoldFactory
from effectome.manifold.behavior_dynamics import BehaviorDynamicsManifold

from .behavior_readout import FrozenBehaviorMap
from .behavior_supervised_config import BehaviorLearningConfig
from .behavior_transition_attribution import (
    nominate_transitions,
    paired_transition_outcomes,
    summarize_interventions,
)
from .behavior_transition_controls import TransitionControlConfig, make_transition_control
from .behavior_transition_forecast import fit_transition_forecast, transition_features
from .behavior_transition_graphs import (
    discover_transition_supports,
    fit_transition_mechanism,
    trailing_histories,
)
from .mechanism_validation import _write_json
from .structural_mechanism import MechanismRolloutConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransitionExperimentConfig:
    generator: TransitionControlConfig = field(default_factory=TransitionControlConfig)
    learning: BehaviorLearningConfig = field(default_factory=BehaviorLearningConfig)
    # Readout fit, forecast fit, candidate nomination, and prospective test trials.
    split_sizes: tuple[int, int, int, int] = (10, 10, 4, 8)
    ridge_alpha: float = 1.0
    consensus_frequency: float = 0.25
    n_rollouts: int = 128
    active_effect_floor: float = 0.05
    practical_forecast_gain: float = 0.005
    primary_method: str = "fast_cgc"

    def __post_init__(self) -> None:
        if len(self.split_sizes) != 4 or any(type(size) is not int or size < 2 for size in self.split_sizes):
            raise ValueError("four whole-trial partitions must each contain at least two trials")
        if sum(self.split_sizes) != self.generator.n_trials:
            raise ValueError("the trial partitions must cover the generator's trials")
        if type(self.n_rollouts) is not int or self.n_rollouts < 2:
            raise ValueError("paired intervention simulations require at least two rollouts")
        for value in (self.ridge_alpha, self.active_effect_floor, self.practical_forecast_gain):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("ridge, intervention and forecast budgets must be positive and finite")
        if not 0 < self.consensus_frequency <= 1:
            raise ValueError("the static support consensus must be in (0, 1]")


def _fit_encoder(
    config: BehaviorLearningConfig,
    neural: np.ndarray,
    behavior: np.ndarray,
    path: Path,
    signature: str,
) -> FrozenBehaviorMap:
    if path.is_file():
        saved = json.loads(path.read_text())
        if saved["signature"] != signature:
            raise ValueError("the saved behavior readout belongs to another study")
        return FrozenBehaviorMap.from_dict(saved["readout"])
    extras = {
        **asdict(config),
        "dynamics_weights": (0.0,),
        "trial_lengths": [neural.shape[1]] * len(neural),
    }
    encoder = ManifoldFactory(
        ManifoldConfig(
            name="bunddle_behavior",
            behavior_key="continuous",
            target_length=1,
            n_dims=2,
            seed=config.seed,
            extra=extras,
        )
    )
    if not isinstance(encoder, BehaviorDynamicsManifold):
        raise TypeError("unexpected behavior encoder registry entry")
    encoder.fit(neural.reshape(-1, neural.shape[-1]), {"continuous": behavior.reshape(-1, 2)})
    readout = FrozenBehaviorMap.from_export(encoder.export_artifact(), name="behavior_only_2d")
    _write_json(path, {"signature": signature, "readout": readout.to_dict()})
    return readout


def _matched_state_diagnostic(
    states: np.ndarray,
    recent_behavior: np.ndarray,
    target: np.ndarray,
    regimes: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> dict[str, Any]:
    # Full origin state and history of the measured behavior, scaled on forecast-fit trials.
    values = np.column_stack((states, recent_behavior))
    features = StandardScaler().fit(values[train]).transform(values[test])
    contexts = regimes[test]
    first, second = np.flatnonzero(contexts == 0), np.flatnonzero(contexts == 1)
    if not first.size or not second.size:
        return {"status": "unavailable", "reason": "one generating regime has no test origins"}
    distances = np.mean((features[first, None] - features[None, second]) ** 2, axis=2)
    match = second[distances.argmin(axis=1)]
    return {
        "status": "available",
        "pairs_with_replacement": len(first),
        "origin_event_pairs": np.column_stack((test[first], test[match])),
        "mean_scaled_state_distance": float(np.sqrt(distances.min(axis=1)).mean()),
        "mean_future_behavior_difference": float(np.abs(target[test[first]] - target[test[match]]).mean()),
    }


def run_transition_case(
    config: TransitionExperimentConfig,
    scenario: str,
    seed: int,
    methods: dict[str, ConnectivityConfig],
    folder: Path,
    signature: str,
) -> dict[str, Any]:
    """Fit and score one process seed and scenario with fixed sample ownership."""
    path = folder / "case.json"
    saved = json.loads(path.read_text()) if path.is_file() else {}
    if saved and saved["signature"] != signature:
        raise ValueError("the saved transition case belongs to another study")
    if saved.get("complete"):
        return saved
    generated = make_transition_control(config.generator, scenario=scenario, seed=seed)
    neural, behavior = generated.neural, generated.behavior
    origins, regimes = generated.origins, generated.regimes.ravel()
    roles = np.split(np.arange(len(neural)), np.cumsum(config.split_sizes)[:-1])
    readout_trials = roles[0]
    groups = np.repeat(np.arange(len(neural)), origins.shape[1])
    _, train, nominate, test = (np.flatnonzero(np.isin(groups, trials)) for trials in roles)
    rollout = MechanismRolloutConfig(
        config.generator.horizon,
        config.generator.target_length,
        config.n_rollouts,
        seed,
    )
    logger.info("%s fitting behavioral encoder", folder.name)
    readout = _fit_encoder(
        config.learning,
        neural[readout_trials],
        behavior[readout_trials],
        folder / "encoder.json",
        signature,
    )
    codes = readout.encode(neural)
    oracle_models = [generated.mechanisms[regime] for regime in regimes]
    oracle_weights = np.stack([model.cross_weights for model in oracle_models])
    event_states = neural[groups, origins.ravel()]
    event_behavior = behavior[groups, origins.ravel()]
    recent = origins.ravel()[:, None] - np.arange(min(min(config.generator.history_lengths), 5))
    recent_behavior = behavior[groups[:, None], recent].reshape(len(groups), -1)
    oracle_factual, truth = paired_transition_outcomes(
        [oracle_models[i] for i in test], event_states[test], test, rollout
    )
    oracle_rollout_predictions = oracle_factual - event_behavior[test]
    if not saved:
        saved = {
            "signature": signature,
            "seed": seed,
            "scenario": scenario,
            "complete": False,
            "trial_partitions": dict(zip(("readout", "forecast", "nomination", "test"), roles, strict=True)),
            "encoder_origin_mae": float(
                np.mean(np.abs(readout.predict(event_states[test]) - event_behavior[test]))
            ),
            "history_lengths": {},
        }
    for length in config.generator.history_lengths:
        key = str(length)
        if saved["history_lengths"].get(key, {}).get("complete"):
            continue
        histories, windows = trailing_histories(neural, origins, length)
        features, states, target = transition_features(
            neural,
            behavior,
            codes,
            origins,
            history_length=length,
            horizon=rollout.horizon,
            target_length=rollout.target_length,
        )
        if not np.array_equal(states, event_states):
            raise ValueError("forecasts and intervention origins disagree")

        fit = partial(fit_transition_forecast, features[train], states[train], target[train], groups[train])
        m0, oracle = fit(), fit(weights=oracle_weights[train])
        m0_predictions = m0.predict(features[test], states[test])
        oracle_predictions = oracle.predict(features[test], states[test], oracle_weights[test])
        m0_error = float(np.abs(m0_predictions - target[test]).mean())
        oracle_error = float(np.abs(oracle_predictions - target[test]).mean())
        current = saved["history_lengths"].setdefault(
            key,
            {
                "complete": False,
                "methods": {},
                "history_only_mae": m0_error,
                "oracle_dynamic_mae": oracle_error,
                "oracle_gain": m0_error - oracle_error,
                "oracle_rollout_mae": float(np.abs(oracle_rollout_predictions - target[test]).mean()),
                "history_only_tuning": m0.tuning,
                "oracle_tuning": oracle.tuning,
                "test_origins": {
                    "events": test,
                    "trial": groups[test],
                    "sample": origins.ravel()[test],
                    "regime": regimes[test],
                    "seconds_since_regime_start": (origins - generated.regime_starts).ravel()[test]
                    / config.generator.fps,
                    "target_delta": target[test],
                    "history_only": m0_predictions,
                    "oracle_dynamic": oracle_predictions,
                    "oracle_rollout": oracle_rollout_predictions,
                },
                "matched_origin_audit": _matched_state_diagnostic(
                    states,
                    recent_behavior,
                    target,
                    regimes,
                    train,
                    test,
                ),
                "oracle_intervention_vectors": truth,
            },
        )
        for method, method_config in methods.items():
            if method in current["methods"]:
                continue
            logger.info(
                "%s estimating %s on %d histories of %d samples", folder.name, method, len(windows), length
            )
            try:
                supports = discover_transition_supports(
                    histories,
                    windows,
                    method_config,
                    config.generator.fps,
                    folder / f"{method}-{length}-native.json",
                    signature,
                )
                pooled_mask = supports[train].mean(axis=0) >= config.consensus_frequency
                static = fit_transition_mechanism(histories[train], pooled_mask, config.ridge_alpha)
                local = [
                    fit_transition_mechanism(history[None], mask, config.ridge_alpha)
                    for history, mask in zip(histories, supports, strict=True)
                ]
                weights = np.stack([model.cross_weights for model in local])
                static_weights = np.broadcast_to(static.cross_weights, weights.shape)
                ms, md = fit(weights=static_weights[train]), fit(weights=weights[train])
                ps = ms.predict(features[test], states[test], static_weights[test])
                pd = md.predict(features[test], states[test], weights[test])
                es = float(np.abs(ps - target[test]).mean())
                ed = float(np.abs(pd - target[test]).mean())
                nominations = nominate_transitions(
                    md,
                    features[nominate],
                    states[nominate],
                    weights[nominate],
                    target[nominate],
                )
                dynamic_factual, dynamic_effects = paired_transition_outcomes(
                    [local[i] for i in test], states[test], test, rollout, readout
                )
                static_factual, static_effects = paired_transition_outcomes(
                    [static] * len(test), states[test], test, rollout, readout
                )
                true_support = oracle_weights[test] != 0
                intersection = (supports[test] & true_support).sum(axis=(1, 2))
                union = (supports[test] | true_support).sum(axis=(1, 2))
                dynamic_rollout = dynamic_factual - event_behavior[test]
                static_rollout = static_factual - event_behavior[test]
                current["methods"][method] = {
                    "status": "available",
                    "window_fits": len(histories),
                    "static_mae": es,
                    "dynamic_mae": ed,
                    "gain_over_history": m0_error - ed,
                    "gain_over_static": es - ed,
                    "test_support_jaccard": float(np.mean(intersection / union)),
                    "static_tuning": ms.tuning,
                    "dynamic_tuning": md.tuning,
                    "static_predictions": ps,
                    "dynamic_predictions": pd,
                    "dynamic_rollout_predictions": dynamic_rollout,
                    "static_rollout_predictions": static_rollout,
                    "dynamic_rollout_mae": float(np.abs(dynamic_rollout - target[test]).mean()),
                    "static_rollout_mae": float(np.abs(static_rollout - target[test]).mean()),
                    "nominations": nominations,
                    "dynamic_intervention_vectors": dynamic_effects,
                    "static_intervention_vectors": static_effects,
                    "interventions": summarize_interventions(
                        pd,
                        regimes[test],
                        truth,
                        dynamic_effects,
                        static_effects,
                        nominations,
                        config.active_effect_floor,
                    ),
                }
            except (ImportError, RuntimeError, ValueError, OSError, np.linalg.LinAlgError) as error:
                logger.error("%s %s %s unavailable: %s", folder.name, method, key, error)
                current["methods"][method] = {"status": "unavailable", "reason": str(error)}
            _write_json(path, saved)
        current["complete"] = True
        _write_json(path, saved)
    saved["complete"] = True
    _write_json(path, saved)
    return json.loads(path.read_text())
