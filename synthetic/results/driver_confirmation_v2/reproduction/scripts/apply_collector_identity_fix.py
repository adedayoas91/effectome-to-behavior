"""Apply and test the reviewed collection-only fix AFTER frozen execution ends."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from effectome.workflows.resume import implementation_fingerprint


def _restored_identity(root: Path, target: Path, original: str) -> str:
    paths = list((root/'src/effectome').rglob('*.py')) + list((root/'pipeline').rglob('*.py'))
    paths += [root/name for name in ('notebooks/_shared.py','notebooks/_build_notebooks.py',
                                     'pyproject.toml','uv.lock')]
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(path.relative_to(root).as_posix().encode())
        data = original.encode() if path == target else path.read_bytes()
        digest.update(hashlib.sha256(data).hexdigest().encode())
    return digest.hexdigest()


def apply_after_completion(source: Path) -> dict:
    report = json.loads((source/'driver-validation.json').read_text())
    if not report['execution_complete'] or not report.get('collection_audit'):
        raise ValueError('collection fix requires complete, replay-verified frozen execution')
    frozen = json.loads((source/'frozen-protocol.json').read_text())
    root = Path(__file__).resolve().parents[1]
    path = root/'src/effectome/experiments/driver_validation.py'
    backup = source/'driver_validation.before-collector-fix.py'
    marker = source/'collector-fix.json'
    old = '(comparison["reference"], comparison["events"], comparison["common_baseline"])'
    new = '(comparison["reference"], comparison["events"])'
    current = path.read_text()
    if marker.is_file():
        saved = json.loads(marker.read_text())
        if saved['updated_implementation_fingerprint'] == implementation_fingerprint():
            return saved
    if implementation_fingerprint() == frozen['implementation_fingerprint']:
        original = current
        if original.count(old) != 1:
            raise ValueError('reviewed collection patch no longer matches exactly')
        backup.write_text(original)
        path.write_text(original.replace(old,new))
    elif backup.is_file():
        original = backup.read_text()
        if (current != original.replace(old,new)
                or _restored_identity(root,path,original) != frozen['implementation_fingerprint']):
            raise RuntimeError('unexpected source changes; no automatic overwrite or completion')
        # Resume verification interrupted after the exact reviewed one-line edit.
    else:
        raise RuntimeError('source changed unexpectedly; no automatic overwrite or completion')
    environment = {**os.environ, 'MPLBACKEND':'Agg', 'MPLCONFIGDIR':'/tmp/effectome-validation-mpl',
                   'XDG_CACHE_HOME':'/tmp/effectome-validation-cache', 'LOKY_MAX_CPU_COUNT':'1'}
    try:
        with (source/'post-study-verification.log').open('w') as log:
            for command in ([sys.executable,'-m','pytest','tests','docs/collector_identity_regression.py'],
                            [sys.executable,'-m','ruff','check','src','pipeline'],
                            [sys.executable,'-m','mypy','src']):
                subprocess.run(command,cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    except BaseException:
        if path.read_text() == original.replace(old,new):
            path.write_text(original)
        raise
    result={'applied':True,'scope':'collection identity only; estimator and scoring algorithms unchanged',
            'experiment_implementation_fingerprint':frozen['implementation_fingerprint'],
            'updated_implementation_fingerprint':implementation_fingerprint(),
            'updated_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'verification_log':'post-study-verification.log'}
    temporary = marker.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2)+'\n')
    temporary.replace(marker)
    return result
