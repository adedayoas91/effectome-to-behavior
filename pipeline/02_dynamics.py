"""Stage 3: cluster connectivity matrices into states and model their Markov transitions."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from effectome.dynamics import (
    GraphStateConfig,
    ProbabilisticStateConfig,
    StateEvaluationConfig,
    TransitionConfig,
    evaluate_state_models,
    fit_frozen_state_transitions,
    fit_graph_states,
    fit_probabilistic_states,
    fit_transitions,
)
from effectome.utils import hydra, set_seed
from effectome.utils.io import load_artifact, save_artifact
from effectome.viz import plot_transition_matrix

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    art = Path(cfg.paths.artifacts)
    fig = Path(cfg.paths.figures)

    series = load_artifact(art / "connectivity.pkl")
    gs_cfg_data = cast(dict[str, Any], OmegaConf.to_container(cfg.states.graph_states, resolve=True))
    gs_cfg = GraphStateConfig(**gs_cfg_data)
    scope = str(cfg.states.get("analysis_scope", "prospective"))
    evaluation = None
    if scope == "prospective":
        selection_options = cast(dict[str, Any], OmegaConf.to_container(cfg.states.selection, resolve=True))
        selection = StateEvaluationConfig(graph=gs_cfg, **selection_options)
        evaluation = evaluate_state_models(series, selection)
        save_artifact(evaluation, art / "state_evaluation.pkl")
        (art / "state_evaluation.json").write_text(
            json.dumps(evaluation.summary(), indent=2, allow_nan=False)
        )
        if evaluation.status != "available" or evaluation.selected_model is None:
            raise ValueError(f"state evaluation unavailable: {evaluation.diagnostics.get('reason')}")
        states = evaluation.selected_model
        logger.info(
            "Selected graph summary: %s, codes=%d, held-out metrics=%s",
            evaluation.selected_algorithm,
            evaluation.selected_n_states,
            evaluation.metrics,
        )
    elif scope == "descriptive":
        states = fit_graph_states(series, gs_cfg)
    else:
        raise ValueError("state analysis_scope must be prospective or descriptive")

    ps_cfg_data = cast(dict[str, Any], OmegaConf.to_container(cfg.states.probabilistic, resolve=True))
    probabilistic = None
    if ps_cfg_data.pop("enabled", False):
        ps_cfg = ProbabilisticStateConfig(**ps_cfg_data)
        probabilistic = fit_probabilistic_states(series, ps_cfg)

    tr_cfg_data = cast(dict[str, Any], OmegaConf.to_container(cfg.states.transitions, resolve=True))
    tr_cfg = TransitionConfig(**tr_cfg_data)
    transitions = (
        fit_frozen_state_transitions(series, evaluation, tr_cfg)
        if evaluation is not None
        else fit_transitions(
            states.labels,
            states.n_states,
            tr_cfg,
            window_starts=states.window_starts,
            boundary_indices=states.boundary_indices,
        )
    )

    save_artifact(states, art / "graph_states.pkl")
    if probabilistic is not None:
        save_artifact(probabilistic, art / "probabilistic_states.pkl")
    save_artifact(transitions, art / "transitions.pkl")
    plot_transition_matrix(
        transitions.transition_matrix,
        "frozen graph-code transitions; latent regimes unvalidated"
        if evaluation is not None
        else f"conditional label-sequence diagnostic (p={transitions.p_value:.3f})",
        fig / "transition_matrix.png",
    )
    if evaluation is not None:
        logger.info(
            "Stage 3 done: %d candidate graph codes, predictive validation=%s; latent regimes unvalidated",
            states.n_states,
            evaluation.diagnostics["predictive_validation_passed"],
        )
    else:
        logger.info(
            "Descriptive Stage 3 done: %d graph codes, conditional label-sequence p=%.4f",
            states.n_states,
            transitions.p_value,
        )


if __name__ == "__main__":
    main()
