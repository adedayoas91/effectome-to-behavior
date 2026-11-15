"""A fitted solver's last bits must not redefine the shared scientific reference."""

import json

import effectome.experiments.driver_validation as validation
from effectome.connectivity import ConnectivityConfig
from effectome.experiments.driver_ground_truth import DriverBenchmarkConfig
from effectome.experiments.driver_validation import DriverValidationConfig, run_driver_validation


def test_fitted_baseline_roundoff_does_not_change_reference_identity(tmp_path, monkeypatch):
    config = DriverValidationConfig(
        scenarios=('single_driver',), positive_seeds=(0,), null_seeds=(0,),
        generator=DriverBenchmarkConfig(n_trials=8), history_length=64,
        estimator_profile='smoke', intervention_rollouts=8,
    )
    methods = {'partial': ConnectivityConfig(name='correlation', partial=True)}
    first = run_driver_validation(config, tmp_path/'first', methods=methods)
    original = validation.run_cross_method_comparison

    def equivalent_fit(*args, **kwargs):
        result = original(*args, **kwargs)
        probabilities = result['common_baseline']['baseline_probabilities']
        probabilities[0][0] += 1e-15
        probabilities[0][1] -= 1e-15
        return result

    monkeypatch.setattr(validation, 'run_cross_method_comparison', equivalent_fit)
    second = run_driver_validation(config, tmp_path/'second', methods=methods)
    fields = ('status', 'n_origins', 'n_events')
    assert [[row[key] for key in fields] for row in first['seed_rows']] == [
        [row[key] for key in fields] for row in second['seed_rows']
    ]
    cell = first['planned_cells'][0]['cell_id'] + '.json'
    a = json.loads((tmp_path/'first/cells'/cell).read_text())
    b = json.loads((tmp_path/'second/cells'/cell).read_text())
    assert a['shared_reference_fingerprint'] == b['shared_reference_fingerprint']

    def changed_inputs(*args, **kwargs):
        result = equivalent_fit(*args, **kwargs)
        result['reference']['baseline_features'] = result['reference']['baseline_features'].copy()
        result['reference']['baseline_features'][0, 0] += 1e-4
        return result

    monkeypatch.setattr(validation, 'run_cross_method_comparison', changed_inputs)
    run_driver_validation(config, tmp_path/'third', methods=methods)
    changed = json.loads((tmp_path/'third/cells'/cell).read_text())
    assert changed['shared_reference_fingerprint'] != a['shared_reference_fingerprint']
