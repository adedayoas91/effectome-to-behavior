"""Test frozen categorical motif probabilities and class alignment."""

import json

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

from effectome.experiments import celegans_motif_readout
from effectome.experiments.celegans_motif_readout import MotifReadout, fit_motif_readout

_FIT = dict(seed=30, hidden_width=7, alpha=0.5, max_iter=500)


@pytest.mark.parametrize("classes", [(1, 3), (1, 2, 4)])
def test_probabilities_match_sklearn_in_declared_class_order(classes, monkeypatch):
    states = np.random.default_rng(7).normal(size=(120, 4))
    thresholds = np.linspace(-0.5, 0.5, len(classes) - 1)
    labels = np.asarray(classes)[np.digitize(states[:, 0] - states[:, 1], thresholds)]
    fitted = []

    def capture_classifier(**kwargs):
        classifier = MLPClassifier(**kwargs)
        fitted.append(classifier)
        return classifier

    monkeypatch.setattr(celegans_motif_readout, "MLPClassifier", capture_classifier)
    readout = fit_motif_readout(states, labels, 5, **_FIT)

    batch = states[:12].reshape(3, 4, 4)
    probabilities = readout.predict(batch)
    assert probabilities.shape == (3, 4, 5)
    assert readout.encode(batch).shape == (3, 4, 2)
    assert readout.classes == classes
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0)
    np.testing.assert_array_equal(probabilities[..., sorted(set(range(5)) - set(classes))], 0.0)
    (model,) = fitted
    standardized = (states[:12] - readout.input_mean) / readout.input_scale
    expected = model.predict_proba(standardized).reshape(3, 4, -1)
    np.testing.assert_allclose(probabilities[..., list(classes)], expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(readout.predict(states[0]), probabilities[0, 0])


def test_readout_artifact_round_trip_and_integrity():
    states = np.random.default_rng(28).normal(size=(60, 4))
    labels = np.where(states[:, 0] > 0, 0, 2)
    readout = fit_motif_readout(states, labels, 3, **_FIT)
    payload = json.loads(json.dumps(readout.to_dict()))
    restored = MotifReadout.from_dict(payload)
    np.testing.assert_array_equal(restored.predict(states), readout.predict(states))
    assert restored.to_dict() == readout.to_dict()
    assert restored.metadata["n_samples"] == len(states)
    assert not restored.input_scale.flags.writeable
    payload["classes"] = [1, 2]
    with pytest.raises(ValueError, match="fingerprint"):
        MotifReadout.from_dict(payload)


def test_readout_rejects_single_class_and_invalid_training_labels():
    states = np.ones((4, 2))
    with pytest.raises(ValueError, match="at least two distinct"):
        fit_motif_readout(states, np.array([0, 0, 0, 0]), 3, **_FIT)
    with pytest.raises(ValueError, match="integer IDs"):
        fit_motif_readout(states, np.array([0.0, 1.0, 0.5, 2.0]), 3, **_FIT)
    with pytest.raises(ValueError, match="integer IDs"):
        fit_motif_readout(states, np.array([0, 1, 3, 2]), 3, **_FIT)
