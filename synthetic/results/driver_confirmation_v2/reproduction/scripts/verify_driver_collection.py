"""Verify paired scoring from frozen native graphs before collecting method shards.

The original merger hashed fitted floating-point baseline outputs bit for bit.
Equivalent repeated logistic fits can differ by ~2e-16. This audit reconstructs
one common reference, replays scoring WITHOUT refitting discovery, checks all
reported values against originals, and hashes the exact reference/event inputs.
Original numerical scores and native evidence are retained without alteration.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from effectome.connectivity.native_gc import _NativeGCConnectivity
from effectome.experiments.driver_ground_truth import make_driver_benchmark
from effectome.experiments.driver_study import collect_driver_study
from effectome.experiments.driver_validation import (
    _cell_evidence,
    driver_validation_cells,
    driver_validation_windows,
    load_driver_validation_profile,
)
from effectome.utils.seed import set_seed
from effectome.workflows.cross_method import (
    _json_safe,
    load_cross_method_profile,
    run_cross_method_comparison,
)
from effectome.workflows.resume import content_fingerprint


def _verify(arguments: tuple[str, dict, str]) -> dict:
    root_name, cell, profile = arguments
    root = Path(root_name)
    stage = root / 'verified-collection'
    config = load_driver_validation_profile(profile)
    base, all_methods = load_cross_method_profile(config.estimator_profile)
    payloads = {}
    for method in all_methods:
        path = root / 'methods' / method / 'cells' / f"{cell['cell_id']}.json"
        if path.is_file():
            payloads[method] = json.loads(path.read_text())
    audit_path = stage / 'audits' / f"{cell['cell_id']}.json"
    inputs = {name: content_fingerprint(payload) for name, payload in payloads.items()}
    for name, payload in payloads.items():
        artifact = payload.get('graph_artifacts', {}).get(name)
        if artifact:
            if Path(artifact['path']).name != artifact['path']:
                raise ValueError('native evidence path must be a local archive filename')
            path = root / 'methods' / name / 'cells' / artifact['path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != artifact['sha256']:
                raise ValueError('native evidence archive checksum mismatch')
    if audit_path.is_file():
        saved = json.loads(audit_path.read_text())
        if saved['source_payload_fingerprints'] == inputs:
            return saved
    methods = {name: all_methods[name] for name in payloads}
    if not methods:
        return {}
    archives = {}
    for name, payload in payloads.items():
        artifact = payload.get('graph_artifacts', {}).get(name)
        if artifact:
            path = root / 'methods' / name / 'cells' / artifact['path']
            with np.load(path, allow_pickle=False) as data:
                archives[name] = np.asarray(data['matrices']).copy()
    original = _NativeGCConnectivity._estimate_windows

    def restored(self, values):
        name = self.cfg.name
        if name not in archives:
            raise RuntimeError('original method was unavailable; no graph or fallback is invented')
        matrices = archives[name]
        if matrices.shape != (len(values), values.shape[-1], values.shape[-1]):
            raise ValueError('saved native graphs do not match scheduled origin axes')
        self._window_lagged = matrices[:, None].copy()
        self._window_diagnostics = []
        self._upstream_provenance = payloads[name]['graph_artifacts'][name]['upstream_provenance']
        return matrices.copy()

    _NativeGCConnectivity._estimate_windows = restored
    try:
        set_seed(cell['seed'])
        recording = make_driver_benchmark(
            replace(config.generator, scenario=cell['scenario'], seed=cell['seed'])
        )
        windows = driver_validation_windows(recording, config)
        comparison = run_cross_method_comparison(
            windows, replace(base, n_dims=config.manifold_dims, n_manifold_states=2,
                             evaluation_fraction=config.evaluation_fraction, seed=cell['seed'],
                             positive_lag=config.positive_lag),
            method_configs=methods, neuron_ids=[str(i) for i in recording.neuron_ids],
            groups={'driver_pair':['0','1'], 'control_pair':['2',str(recording.n_neurons-1)]},
            graph_origin_only=True,
        )
        replayed = _cell_evidence(recording, windows, comparison, cell, config)
    finally:
        _NativeGCConnectivity._estimate_windows = original
    lookup = {(row['method'], row['source_state']): row for row in replayed}
    maximum = 0.0
    for name, payload in payloads.items():
        for row in payload['seed_rows']:
            checked = lookup[name, row['source_state']]
            assert row['status'] == checked['status']
            if row['status'] != 'available':
                continue
            for key in ('n_origins','n_events','n_independent_trials'):
                if row[key] != checked[key]:
                    raise ValueError(f'paired event identity differs: {cell} {name} {key}')
            for key in ('alignment_accuracy','global_alignment_accuracy','log_loss_gain'):
                difference = abs(row[key] - checked[key])
                maximum = max(maximum, difference)
                np.testing.assert_allclose(row[key], checked[key], rtol=0, atol=1e-10)
            for key in ('driver_scores', 'intervention_effects'):
                assert row[key].keys() == checked[key].keys()
                for unit, value in row[key].items():
                    difference = abs(value - checked[key][unit])
                    maximum = max(maximum, difference)
                    np.testing.assert_allclose(value, checked[key][unit], rtol=0, atol=1e-10)
            np.testing.assert_allclose(
                comparison['methods'][name]['prediction']['baseline_probabilities'],
                comparison['common_baseline']['baseline_probabilities'], rtol=0, atol=1e-12,
            )
    canonical = content_fingerprint((comparison['reference'], comparison['events']))
    for name, payload in payloads.items():
        folder = stage / 'methods' / name / 'cells'
        folder.mkdir(parents=True, exist_ok=True)
        artifact = payload.get('graph_artifacts', {}).get(name)
        if artifact:
            target = folder / artifact['path']
            if not target.exists():
                os.link(root / 'methods' / name / 'cells' / artifact['path'], target)
        changed = {**payload, 'shared_reference_fingerprint': canonical,
                   'original_fitted_baseline_fingerprint': payload['shared_reference_fingerprint'],
                   'collection_audit': 'same reference/events/baseline inputs; '
                   'scoring replay within1e-10; scores unchanged'}
        path = folder / f"{cell['cell_id']}.json"
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(_json_safe(changed), indent=2, allow_nan=False)+'\n')
        temporary.replace(path)
    audit = {**cell, 'source_payload_fingerprints': inputs, 'reference_event_fingerprint': canonical,
             'maximum_replayed_score_difference': maximum, 'absolute_tolerance': 1e-10,
             'native_estimator_refit': False, 'original_scores_changed': False}
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2)+'\n')
    return audit


def verify_and_collect(root: Path, profile: str = 'confirmatory', workers: int = 2) -> dict:
    config = load_driver_validation_profile(profile)
    _, methods = load_cross_method_profile(config.estimator_profile)
    arguments = [(str(root), cell, profile) for cell in driver_validation_cells(config)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        audits = list(pool.map(_verify, arguments))
    report = collect_driver_study(config, root / 'verified-collection', methods)
    report['collection_audit'] = {
        'reason': 'bitwise fitted-baseline identity was too strict for equivalent logistic fits',
        'identity': 'exact reference including baseline features, states, events and ownership',
        'score_replay_absolute_tolerance': 1e-10,
        'maximum_replayed_score_difference': max(
            (a.get('maximum_replayed_score_difference',0) for a in audits), default=0),
        'native_discovery_refit': False, 'original_scores_changed': False,
        'audits': 'verified-collection/audits/',
    }
    path = root / 'driver-validation.json'
    temporary = path.with_suffix('.verified.tmp')
    temporary.write_text(json.dumps(_json_safe(report),indent=2,allow_nan=False)+'\n')
    temporary.replace(path)
    return report


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('outputs/driver-validation/csl_sanity/scheduled-v2/confirmatory'))
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    result=verify_and_collect(args.source,workers=args.workers)
    print(result['attempted_method_runs'],result['available_method_runs'],result['acceptance']['overall_status'])
