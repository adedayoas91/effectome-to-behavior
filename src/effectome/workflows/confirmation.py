"""Registered paired-performance inference for a frozen fit on independent new animals.

This tests the symmetry null for animal-level predictive-loss differences. It does not
test graph/endpoint conditional independence or the existence of dynamic structure;
those require their separately registered raw-process simulation experiments.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from effectome.linking.stats import (
    RegisteredFamily,
    plus_one_monte_carlo_pvalue,
    require_monte_carlo_resolution,
)

if TYPE_CHECKING:
    from .evaluation import EvaluationConfig, EvaluationResult


_PAIRED_TESTS = frozenset({"paired_M0_M1_log_loss", "paired_log_loss"})


def validate_source_files(data: Mapping[str, Any], *, required: bool) -> dict[str, str]:
    """Verify every consumed external file against frozen byte hashes before loading.

    Directory-loader hash keys are paths relative to data.path. The legacy
    downloaded_sha256 mapping is accepted under the same exact-coverage contract.
    """
    name = str(data["name"])
    if name == "synthetic":
        return {}
    base = Path(str(data.get("path", "")))
    consumed: set[Path]
    if name in {"c_elegans", "zebrafish"}:
        consumed = {base.resolve()}
        hash_base = base.parent
    elif name in {"bundle_net_c_elegans", "v2a_rsns"}:
        from effectome.data_module.loaders import _resolve_file

        files = data.get("files", {})
        if not isinstance(files, dict) or not files:
            raise ValueError("external directory sources require a non-empty files mapping")
        keys = (
            ("recordings", "class_ids", "neuron_order") if name == "bundle_net_c_elegans"
            else ("analysis_info", "traces", "tail_angle", "emitter_cells", "receiver_cells")
        )
        consumed = {_resolve_file(base, files, key) for key in keys}
        if name == "v2a_rsns":
            raw_fallback = str(files.get("coords", "")).replace("registered_cells_pos", "cells_positions")
            if str(data.get("coords_source", "registered")).lower() == "registered":
                consumed.add(_resolve_file(base, files, "coords"))
                if "raw_coords" in files or "registered_cells_pos" in str(files.get("coords", "")):
                    consumed.add(_resolve_file(base, files, "raw_coords", fallback=raw_fallback))
            else:
                consumed.add(_resolve_file(base, files, "raw_coords", fallback=raw_fallback))
        hash_base = base
    elif required:
        raise ValueError("confirmatory external loader has no registered consumed-file hashing contract")
    else:
        return {}
    declaration = data.get("source_file_sha256", data.get("downloaded_sha256"))
    if declaration is None:
        if required:
            raise ValueError(
                "confirmatory external recordings require source_file_sha256 for every consumed file"
            )
        return {}
    if isinstance(declaration, str) and len(consumed) == 1:
        hashes = {next(iter(consumed)): declaration}
    elif isinstance(declaration, Mapping):
        hashes = {(hash_base / str(path)).resolve(): str(digest) for path, digest in declaration.items()}
        if len(hashes) != len(declaration):
            raise ValueError("source_file_sha256 contains duplicate aliases for one source file")
    else:
        raise ValueError("source_file_sha256 must be a digest or a relative-path-to-digest mapping")
    if required and set(hashes) != consumed:
        raise ValueError("source_file_sha256 must cover exactly every consumed source file")
    verified: dict[str, str] = {}
    for path, declared in hashes.items():
        expected = declared.removeprefix("sha256:").lower()
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            raise ValueError(f"invalid SHA-256 declaration for {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(f"source file SHA-256 mismatch: {path}")
        verified[str(path)] = actual
    return verified


def _members(cfg: EvaluationConfig) -> tuple[str, ...]:
    return cfg.multiplicity_members or tuple(dict.fromkeys(test for _, test in cfg.claim_to_test_map))


def preflight_confirmation(cfg: EvaluationConfig) -> dict[str, Any]:
    """Reject unsupported designs and insufficient null resolution before fitting."""
    decoder_grid = cfg.decoder_c_grid if cfg.endpoint_type == "categorical" else cfg.decoder_alpha_grid
    tuning_trials = len(set(cfg.neural_lambda1_grid)) + 2 * len(set(decoder_grid))
    if cfg.tuning_trial_ceiling is not None and tuning_trials > cfg.tuning_trial_ceiling:
        raise ValueError("registered per-fold tuning trial ceiling is smaller than planned candidates")
    if cfg.protocol != "fixed_development" or cfg.update_policy != "frozen":
        raise ValueError("confirmation requires a frozen fixed_development cohort, not overlapping CV fits")
    if not cfg.development_animal_ids or not cfg.evaluation_animal_ids:
        raise ValueError("confirmation requires explicit disjoint development and evaluation animals")
    if not cfg.sign_symmetry_justified:
        raise ValueError("animal sign-flip inference requires a registered sign-symmetry justification")
    if cfg.null_families != ("animal_sign_flip",):
        raise ValueError(
            "paired-performance confirmation supports only animal_sign_flip; raw-process and "
            "conditional-information nulls belong to their separately registered benchmark suites"
        )
    members = _members(cfg)
    RegisteredFamily(members)
    if not _PAIRED_TESTS.intersection(members):
        raise ValueError("the complete multiplicity family must include the paired log-loss contrast")
    if not {test for _, test in cfg.claim_to_test_map}.issubset(members):
        raise ValueError("multiplicity_members must contain every registered claim-to-test member")
    harmonic = float(np.sum(1.0 / np.arange(1, len(members) + 1)))
    alpha_min = cfg.confirmation_alpha / (len(members) * harmonic)
    n_animals = len(cfg.evaluation_animal_ids)
    finite_minimum = 2.0 ** (1 - n_animals)
    if finite_minimum > alpha_min:
        raise ValueError(
            f"{n_animals} independent animals cannot attain the two-sided sign-flip "
            f"threshold {alpha_min:g}; repeated draws do not increase the finite control pool"
        )
    method = cfg.sign_flip_method
    if method == "auto":
        method = "exact" if n_animals <= 16 else "monte_carlo"
    if method == "exact" and n_animals > 20:
        raise ValueError("exact enumeration is limited to 20 animals; register monte_carlo instead")
    if cfg.null_replicates is None:
        raise ValueError("confirmation requires a registered null_replicates count")
    if method == "monte_carlo":
        require_monte_carlo_resolution(cfg.null_replicates, alpha_min)
    return {
        "method": method,
        "members": members,
        "alpha_min": alpha_min,
        "finite_minimum_two_sided_p": finite_minimum,
        "n_evaluation_animals": n_animals,
        "tuning_candidate_count_per_fold": tuning_trials,
    }


def _sign_null(values: np.ndarray, *, method: str, draws: int, seed: int) -> np.ndarray:
    """Sign whole animal effects; never shuffle dependent anchor/fold observations."""
    n_animals = len(values)
    rng = np.random.default_rng(seed)
    total = (1 << n_animals) if method == "exact" else draws
    null = np.empty(total, dtype=float)
    for start in range(0, total, 1024):
        stop = min(start + 1024, total)
        if method == "exact":
            numbers = np.arange(start, stop, dtype=np.uint64)
            signs = 2 * ((numbers[:, None] >> np.arange(n_animals, dtype=np.uint64)) & 1).astype(int) - 1
        else:
            signs = 2 * rng.integers(0, 2, size=(stop - start, n_animals)) - 1
        null[start:stop] = np.abs((signs * values).mean(axis=1))
    return null


def attach_confirmation(result: EvaluationResult, cfg: EvaluationConfig) -> EvaluationResult:
    """Attach a complete, failure-preserving registered inference report to frozen predictions."""
    from .evaluation import paired_log_loss_delta, score_classification_records, score_regression_records
    from .resume import content_fingerprint

    plan = preflight_confirmation(cfg)
    if (
        result.manifest.protocol != "fixed_development"
        or len(result.manifest.outer_folds) != 1
        or result.provenance.analysis_mode != "confirmatory"
        or result.provenance.config_hash != content_fingerprint(cfg)
    ):
        raise ValueError(
            "confirmation requires the matching frozen whole-cohort result and config provenance"
        )
    paired = paired_log_loss_delta(
        result.predictions,
        baseline_model="M0_activity",
        candidate_model="M1_effectome",
        endpoint_name=cfg.endpoint_name,
    )
    reason: str | None = None
    expected_pairs = sum(len(fold.test_indices) for fold in result.manifest.outer_folds)
    complete_records = (
        len(result.predictions) == 2 * expected_pairs
        and all(
            record.model_name in {"M0_activity", "M1_effectome"}
            and record.endpoint_name == cfg.endpoint_name
            and record.animal_id in cfg.evaluation_animal_ids
            for record in result.predictions
        )
    )
    if result.diagnostics.get("resource_accounting", {}).get("failures"):
        reason = "registered_resource_budget_exceeded"
    elif not complete_records:
        reason = "registered_prediction_pairs_incomplete"
    elif any(fold.status != "available" for fold in result.manifest.outer_folds):
        reason = "registered_outer_fold_unavailable"
    elif set(paired.per_animal_delta) != set(cfg.evaluation_animal_ids):
        reason = "registered_evaluation_animals_missing"
    elif (
        paired.n_baseline_only or paired.n_candidate_only
        or paired.n_failed_baseline or paired.n_failed_candidate
    ):
        reason = "registered_prediction_pairs_failed_or_unavailable"
    animal_order = tuple(cfg.evaluation_animal_ids)
    values = np.asarray([paired.per_animal_delta.get(animal, np.nan) for animal in animal_order])
    if np.any(~np.isfinite(values)):
        reason = reason or "nonfinite_animal_effect"
    null = np.empty(0, dtype=float)
    p_value: float | None = None
    interval: tuple[float, float] | None = None
    if reason is None:
        assert cfg.null_replicates is not None
        null = _sign_null(
            values, method=plan["method"], draws=cfg.null_replicates, seed=cfg.selection_seed
        )
        statistic = abs(float(values.mean()))
        # A tolerance prevents arithmetic order from making an equivalent sign pattern less extreme.
        extreme = int(np.sum(null >= statistic - 1e-12))
        p_value = (
            float(extreme / len(null)) if plan["method"] == "exact"
            else plus_one_monte_carlo_pvalue(extreme, len(null))
        )
        # Conditional on the single frozen development fit, each new animal is an independent unit.
        rng = np.random.default_rng(np.random.SeedSequence([cfg.selection_seed, 1]))
        boot = np.empty(cfg.null_replicates, dtype=float)
        for start in range(0, len(boot), 1024):
            stop = min(start + 1024, len(boot))
            boot[start:stop] = rng.choice(values, size=(stop - start, len(values)), replace=True).mean(axis=1)
        bounds = np.quantile(boot, [cfg.confirmation_alpha / 2, 1 - cfg.confirmation_alpha / 2])
        interval = (float(bounds[0]), float(bounds[1]))
    tests: dict[str, tuple[float | None, str]] = {}
    for member in plan["members"]:
        if member not in _PAIRED_TESTS:
            tests[member] = (None, "unavailable:registered_test_not_implemented")
        elif reason is not None:
            tests[member] = (None, f"unavailable:{reason}")
        else:
            tests[member] = (p_value, "computed")
    adjusted = RegisteredFamily(plan["members"]).adjust(tests)
    for row in adjusted.values():
        row["rejected"] = bool(row["p_value"] is not None and row["q_value"] <= cfg.confirmation_alpha)
        row["correction_input_when_unavailable"] = None if row["p_value"] is not None else 1.0
    scores: dict[str, Any] = {}
    for model_name in ("M0_activity", "M1_effectome"):
        if cfg.endpoint_type == "categorical":
            score = score_classification_records(
                result.predictions, model_name=model_name, endpoint_name=cfg.endpoint_name
            )
            scores[model_name] = {"log_loss_nats": score.log_loss_nats, "n_predictions": score.n_predictions}
        else:
            score_continuous = score_regression_records(
                result.predictions, model_name=model_name, endpoint_name=cfg.endpoint_name
            )
            scores[model_name] = {
                "log_loss_nats": score_continuous.gaussian_log_loss_nats,
                "rmse": score_continuous.rmse,
                "n_predictions": score_continuous.n_predictions,
            }
    report = {
        "status": "computed" if reason is None else "unavailable",
        "failure_reason": reason,
        "estimand": "equal-animal predictive log-loss gain conditional on a frozen development fit",
        "null": "independent animal paired-effect sign symmetry about zero",
        "conditional_information_null_tested": False,
        "raw_process_structure_null_tested": False,
        "sign_symmetry_justified": cfg.sign_symmetry_justified,
        "development_animal_ids": cfg.development_animal_ids,
        "evaluation_animal_ids": animal_order,
        "per_animal_delta": paired.per_animal_delta,
        "mean_delta_log_loss_nats": paired.equal_animal_delta_log_loss,
        "n_matched_predictions": paired.n_matched_predictions,
        "n_expected_prediction_pairs": expected_pairs,
        "absolute_prediction_scores": scores,
        "interval": interval,
        "interval_method": "fixed-fit independent-animal percentile bootstrap",
        "interval_is_small_sample_calibrated": False,
        "practical_gain_nats": cfg.practical_gain_nats,
        "interval_exceeds_practical_gain": bool(
            interval is not None and cfg.practical_gain_nats is not None
            and interval[0] > cfg.practical_gain_nats
        ),
        "p_value": p_value,
        "method": plan["method"],
        "null_statistics": null.tolist(),
        "null_draw_count": int(len(null)),
        "requested_null_replicates": cfg.null_replicates,
        "finite_minimum_two_sided_p": plan["finite_minimum_two_sided_p"],
        "alpha_min": plan["alpha_min"],
        "multiplicity_family": cfg.multiplicity_family,
        "multiplicity_method": "Benjamini-Yekutieli",
        "family": adjusted,
        "family_status": (
            "complete" if all(row["status"] == "computed" for row in adjusted.values())
            else "contains_unavailable_tests"
        ),
        "unavailable_family_members_retained": True,
    }
    return replace(result, diagnostics={**result.diagnostics, "confirmation": report})
