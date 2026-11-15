"""Inventory or resume unfinished native fits without rerunning the old merger."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from effectome.connectivity import ConnectivityConfig
from effectome.experiments.driver_ground_truth import DriverBenchmarkConfig
from effectome.experiments.driver_study import _shard
from effectome.experiments.driver_validation import (
    DriverValidationConfig,
    driver_validation_cells,
    run_driver_validation,
)
from effectome.workflows.resume import implementation_fingerprint


def inventory(source: Path) -> tuple[DriverValidationConfig, dict, dict]:
    frozen = json.loads((source/'frozen-protocol.json').read_text())
    values = dict(frozen['config'])
    for name in ('scenarios','positive_seeds','null_seeds'):
        values[name] = tuple(values[name])
    values['generator'] = DriverBenchmarkConfig(**values['generator'])
    config = DriverValidationConfig(**values)
    methods = {name: ConnectivityConfig(**data) for name,data in frozen['methods'].items()}
    expected = len(driver_validation_cells(config)) * len(methods)
    summary_path = source/'driver-validation.json'
    if summary_path.is_file() and json.loads(summary_path.read_text())['execution_complete']:
        return config, methods, {'completed':expected,'remaining':0,'reporting_only':True}
    if implementation_fingerprint() != frozen['implementation_fingerprint']:
        raise ValueError('scientific source differs from the frozen study; do not mix versions')
    project = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(source/'implementation-snapshot.zip') as archive:
        for name in (f'conf/cross_method/{config.estimator_profile}.yaml',
                     'conf/driver_validation/acceptance.yaml'):
            if (project/name).read_bytes() != archive.read(name):
                raise ValueError(f'configuration differs from the frozen study: {name}')
    counts = {}
    for name, method in methods.items():
        template = run_driver_validation(config, source/'methods'/name, methods={name:method},
                                         cell_ids=set(), execute=False, write_report=False)
        done = 0
        for cell in driver_validation_cells(config):
            path = source/'methods'/name/'cells'/f"{cell['cell_id']}.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text())
            if payload['signature'] != template['manifest_fingerprint']:
                raise ValueError(f'checkpoint belongs to another protocol: {path}')
            for artifact in payload.get('graph_artifacts',{}).values():
                archive = path.parent/artifact['path']
                if hashlib.sha256(archive.read_bytes()).hexdigest() != artifact['sha256']:
                    raise ValueError(f'native evidence integrity failed: {archive}')
            done += 1
        counts[name] = done
    return config, methods, {'completed':sum(counts.values()),
                             'remaining':expected-sum(counts.values()),'per_method':counts,
                             'reporting_only':False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,
                        default=Path('outputs/driver-validation/csl_sanity/scheduled-v2/confirmatory'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--run',action='store_true',help='Resume only after previous compute workers stopped')
    args = parser.parse_args()
    config, methods, progress = inventory(args.source)
    print(json.dumps(progress,indent=2),flush=True)
    if not args.run or not progress['remaining']:
        return
    if args.workers < 1:
        parser.error('workers must be positive')
    cells = driver_validation_cells(config)
    shards = [{cell['cell_id'] for cell in cells[index::args.workers]} for index in range(args.workers)]
    process_record = {'pid':os.getpid(),'command':'synthetic.resume_driver_study --run',
                      'output_directory':str(args.source.resolve())}
    (args.source/'process.json').write_text(json.dumps(process_record,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(_shard,config,args.source,name,method,ids)
                   for name,method in methods.items() for ids in shards if ids]
        for future in as_completed(futures):
            future.result()
    print('Native computation finished; '
          'the artifact-based finalizer performs verified collection and export.')


if __name__ == '__main__':
    main()
