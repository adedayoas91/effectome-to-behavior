"""Stage 3: cluster connectivity matrices into states and model their Markov transitions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from effectome.dynamics import (
    GraphStateConfig,
    ProbabilisticStateConfig,
    TransitionConfig,
    fit_graph_states,
    fit_probabilistic_states,
    fit_transitions,
)
from effectome.utils import set_seed
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
    states = fit_graph_states(series, gs_cfg)

    ps_cfg_data = cast(dict[str, Any], OmegaConf.to_container(cfg.states.probabilistic, resolve=True))
    ps_cfg = ProbabilisticStateConfig(**ps_cfg_data)
    probabilistic = fit_probabilistic_states(series, ps_cfg)

    tr_cfg_data = cast(dict[str, Any], OmegaConf.to_container(cfg.states.transitions, resolve=True))
    tr_cfg = TransitionConfig(**tr_cfg_data)
    transitions = fit_transitions(
        states.labels,
        states.n_states,
        tr_cfg,
        window_starts=states.window_starts,
        boundary_indices=states.boundary_indices,
    )

    save_artifact(states, art / "graph_states.pkl")
    save_artifact(probabilistic, art / "probabilistic_states.pkl")
    save_artifact(transitions, art / "transitions.pkl")
    plot_transition_matrix(
        transitions.transition_matrix,
        f"state transitions (p={transitions.p_value:.3f})",
        fig / "transition_matrix.png",
    )
    logger.info(
        "Stage 3 done: %d states, silhouette=%.3f, transition p=%.4f, hmm ll=%.2f",
        states.n_states,
        states.silhouette,
        transitions.p_value,
        probabilistic.log_likelihood,
    )


if __name__ == "__main__":
    main()
