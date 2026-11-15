"""Interrupted work is distinguishable from durable completed/failed attempts."""

import hashlib
import json
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from effectome.experiments.driver_validation import load_driver_validation_profile, run_driver_validation
from effectome.workflows.cross_method import _json_safe, load_cross_method_profile
from effectome.workflows.resume import implementation_fingerprint
from synthetic.resume_driver_study import inventory


def _study(root):
    config = replace(load_driver_validation_profile('smoke'), scenarios=('single_driver',),
                     positive_seeds=(0,), null_seeds=(0,))
    _, methods = load_cross_method_profile('csl_sanity')
    frozen = {'config':asdict(config), 'methods':{name:asdict(cfg) for name,cfg in methods.items()},
              'implementation_fingerprint':implementation_fingerprint()}
    (root/'frozen-protocol.json').write_text(json.dumps(_json_safe(frozen)))
    project = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(root/'implementation-snapshot.zip','w') as archive:
        for name in ('conf/cross_method/csl_sanity.yaml','conf/driver_validation/acceptance.yaml'):
            archive.write(project/name,name)
    name, method = next(iter(methods.items()))
    folder = root/'methods'/name/'cells'
    folder.mkdir(parents=True)
    template = run_driver_validation(config,root/'methods'/name,methods={name:method},
                                     cell_ids=set(),execute=False,write_report=False)
    cell = template['planned_cells'][0]
    payload = {'signature':template['manifest_fingerprint'], 'attempt':{**cell,'status':'failed'},
               'seed_rows':[], 'graph_artifacts':{}}
    path = folder/f"{cell['cell_id']}.json"
    path.write_text(json.dumps(payload))
    return path,payload


def test_inventory_reuses_recorded_failures_and_ignores_interrupted_temp_files(tmp_path):
    path, _ = _study(tmp_path)
    path.with_suffix('.tmp').write_text('unfinished')
    before = path.read_bytes()
    _, _, result = inventory(tmp_path)
    assert result['completed'] == 1 and result['remaining'] == 3
    assert path.read_bytes() == before


def test_inventory_rejects_checkpoint_from_another_protocol(tmp_path):
    path, payload = _study(tmp_path)
    payload['signature'] = 'another-protocol'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='another protocol'):
        inventory(tmp_path)


def test_inventory_checks_native_evidence_before_skipping_completed_work(tmp_path):
    path, payload = _study(tmp_path)
    archive = path.with_suffix('.npz')
    archive.write_bytes(b'original evidence')
    payload['graph_artifacts'] = {'native':{'path':archive.name,
                                          'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}}
    path.write_text(json.dumps(payload))
    archive.write_bytes(b'interrupted or changed evidence')
    with pytest.raises(ValueError,match='integrity failed'):
        inventory(tmp_path)
