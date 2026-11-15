"""Finalize a completed frozen study; optionally wait for its recorded process."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from effectome.experiments.csl_sanity import CSL_METHODS
from synthetic.export_driver_validation import export_driver_validation


def _reproduction_bundle(source: Path, snapshot: Path) -> dict:
    destination = snapshot / 'reproduction'
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / 'implementation-snapshot.zip', destination / 'implementation-snapshot.zip')
    shutil.copy2(source / 'implementation-snapshot.json', destination / 'implementation-snapshot.json')
    shutil.copytree(source / 'verified-collection/audits', destination / 'audits', dirs_exist_ok=True)
    scripts = destination / 'scripts'
    scripts.mkdir(exist_ok=True)
    for name in ('verify_driver_collection.py', 'finalize_driver_study.py',
                 'export_driver_validation.py', 'apply_collector_identity_fix.py'):
        shutil.copy2(Path(__file__).with_name(name), scripts / name)
    for method in CSL_METHODS:
        files = sorted((source / 'methods' / method / 'cells').glob('*'))
        part, size, archive = 0, 0, None
        try:
            for path in files:
                if path.suffix not in {'.json', '.npz'}:
                    continue
                if archive is None or size + path.stat().st_size > 40 * 1024**2:
                    if archive is not None:
                        archive.close()
                    archive = zipfile.ZipFile(destination / f'{method}-part-{part:03}.zip', 'w')
                    part, size = part + 1, 0
                archive.write(path, path.relative_to(source))
                size += path.stat().st_size
        finally:
            if archive is not None:
                archive.close()
    (destination / 'README.md').write_text(
        '# Frozen driver-study reproduction\n\n'
        'The source ZIP preserves the implementation used for discovery/scoring. '
        'Method archives retain original NPZ native graphs/evidence and per-recording JSON. '
        'Extract all method parts into one study directory; archive paths retain methods/.../cells/. '
        'Restore the source snapshot and its pinned main/conventional locks to replay scores '
        'with the included verification script. Discovery need not be rerun: upstream CPU '
        'permutation randomness is not fully controlled. Original numerical scores are retained; '
        'audits document machine-precision baseline differences and exact sample ownership. '
        'Parts are below41MiB to avoid oversized repository blobs.\n'
    )
    return {str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in destination.rglob('*') if path.is_file()}


def finalize(source: Path, manuscript: Path, snapshot: Path) -> dict:
    report = json.loads((source / 'driver-validation.json').read_text())
    audit = report.get('collection_audit', {})
    expected = len(report['planned_cells']) * len(CSL_METHODS)
    if (not report['execution_complete'] or set(report['planned_methods']) != set(CSL_METHODS)
            or report['attempted_method_runs'] != expected
            or audit.get('original_scores_changed') is not False
            or audit.get('native_discovery_refit') is not False
            or audit.get('maximum_replayed_score_difference', 1) > 1e-10):
        raise ValueError('finalization requires complete, replay-verified four-method execution')
    for path in (source / 'methods').glob('*/cells/*.json'):
        payload = json.loads(path.read_text())
        for artifact in payload.get('graph_artifacts', {}).values():
            actual = hashlib.sha256((path.parent / artifact['path']).read_bytes()).hexdigest()
            if actual != artifact['sha256']:
                raise ValueError('native evidence integrity failed before manuscript export')
    export_driver_validation(source, manuscript)
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in ('driver-validation.json', 'frozen-protocol.json'):
        shutil.copy2(source / name, snapshot / name)
    shutil.copytree(source / 'analysis', snapshot / 'analysis', dirs_exist_ok=True)
    evidence = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (source / 'methods').rglob('*.npz')}
    provenance = {'completed_at': datetime.now(timezone.utc).isoformat(),
                  'source': str(source), 'execution_complete': True,
                  'attempted_method_runs': report['attempted_method_runs'],
                  'available_method_runs': report['available_method_runs'],
                  'acceptance': report['acceptance']['overall_status'],
                  'numerical_sha256': hashlib.sha256(
                      (source / 'driver-validation.json').read_bytes()).hexdigest(),
                  'native_evidence_sha256': evidence,
                  'native_evidence_location': 'Retained under source/methods; numerical source is simulated.'}
    (snapshot / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    provenance['reproduction_files_sha256'] = _reproduction_bundle(source, snapshot)
    tectonic = shutil.which('tectonic')
    if tectonic:
        with (source / 'manuscript-build.log').open('w') as log:
            subprocess.run([tectonic, '--keep-logs', 'main.tex'], cwd=manuscript,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    from synthetic.apply_collector_identity_fix import apply_after_completion

    provenance['collection_identity_fix'] = apply_after_completion(source)
    (snapshot / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    for name in ('collector-fix.json', 'post-study-verification.log'):
        if (source / name).is_file():
            shutil.copy2(source / name, snapshot / name)
    (source / 'finalization.json').write_text(json.dumps(provenance, indent=2) + '\n')
    state_path = Path(__file__).resolve().parents[2] / '.omx/state/driver-confirmation.json'
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({'active': False, 'phase': 'execution_complete',
                                     'completed_at': provenance['completed_at'],
                                     'acceptance': provenance['acceptance'],
                                     'output_directory': str(source.resolve())}, indent=2) + '\n')
    decisions = report['acceptance']['methods']
    text = ['# Four-CSL driver-study conclusion', '',
            f"Execution complete: {report['attempted_method_runs']} method/recording runs; "
            f"{report['available_method_runs']} numerically available.", '',
            f"Overall frozen-criteria decision: **{report['acceptance']['overall_status']}**.", '',
            '| Method | Recovery decision |', '|---|---|']
    text.extend(f"| {name} | {decision['status']} |" for name, decision in decisions.items())
    text.extend(['', 'A pass applies only to this synthetic process and nomination rule. A failed '
                 'or inconclusive decision does not establish reliable transition-driver identification. '
                 'Related methods are one family; agreement is not independent causal confirmation. '
                 'Biological necessity and hidden-confounder identifiability remain unvalidated.'])
    (snapshot / 'conclusion.md').write_text('\n'.join(text) + '\n')
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=Path('outputs/driver-validation/csl_sanity/scheduled-v2/confirmatory'))
    parser.add_argument('--manuscript', type=Path, default=Path('../manuscript-260926'))
    parser.add_argument('--snapshot', type=Path, default=Path('synthetic/results/driver_confirmation_v2'))
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    if args.watch:
        last_verified_count = 0
        while True:
            path = args.source / 'driver-validation.json'
            if path.is_file() and json.loads(path.read_text())['execution_complete']:
                break
            cached_count = len(list((args.source / 'methods').glob('*/cells/*.json')))
            if cached_count >= last_verified_count + 16 or cached_count == 560:
                from synthetic.verify_driver_collection import verify_and_collect

                report = verify_and_collect(args.source)
                last_verified_count = report['attempted_method_runs']
                print(f"Verified collection: {last_verified_count}/560; "
                      f"decision={report['acceptance']['overall_status']}", flush=True)
                if report['execution_complete']:
                    break
            # A restricted monitor may not inspect another process. Observe the
            # authorized atomic result files instead of requiring signal access.
            files = list((args.source / 'methods').glob('*/cells/*.json'))
            latest = max((item.stat().st_mtime for item in files),
                         default=(args.source / 'frozen-protocol.json').stat().st_mtime)
            if time.time() - latest > 6 * 3600:
                raise RuntimeError('no new method artifacts for six hours; inspect execution.log')
            time.sleep(30)
    print(json.dumps(finalize(args.source, args.manuscript.resolve(), args.snapshot), indent=2))


if __name__ == '__main__':
    main()
