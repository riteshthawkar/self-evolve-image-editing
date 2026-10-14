from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from . import common as c


def cached_summary(run_dir):
    from .statistics import schedule, summarize
    run_dir, plan = c.load_run(run_dir)
    cases = c.read(run_dir/'cases.json')['cases']; tasks = schedule(cases)
    receipts = [c.read(run_dir/'evaluation/receipts'/(r['id']+'.json')) for r in tasks]
    if any(r['state'] != 'complete' for r in receipts): raise RuntimeError('Judgments are not complete')
    case_rows = [c.read(run_dir/'case_receipts'/(r['id']+'.json')) for r in cases]
    result = summarize(cases,tasks,receipts,case_rows,plan['config'])
    result['accounted_usd'] = sum(r['accounted_usd'] for r in receipts)
    c.seal(run_dir/'summary.json',result)
    print(json.dumps({k:result[k] for k in ('status','passed','decision','coverage','accounted_usd')},indent=2))


def main():
    parser = argparse.ArgumentParser(description='Native-only source-path diagnostic; no training or automatic job submission')
    parser.add_argument('--runtime', choices=('dedicated','slurm'), default='dedicated',
                        help='Direct execution is authorized on the new dedicated machine; old cluster still requires Slurm')
    sub = parser.add_subparsers(dest='phase',required=True)
    data = sub.add_parser('prepare',help='CPU-only checkpoint hashing and source-disjoint image preparation')
    data.add_argument('--images',required=True); data.add_argument('--model-dir',required=True)
    data.add_argument('--out',required=True); data.add_argument('--config',default=str(c.PACKAGE/'config.yaml'))
    for name in ('canary','generate'):
        p = sub.add_parser(name); p.add_argument('--model-dir',required=True); p.add_argument('--out',required=True)
        p.add_argument('--resume',action='store_true',help='Reuse only committed, hash-verified work in this exact run')
    for name in ('score','summarize'):
        p = sub.add_parser(name); p.add_argument('--out',required=True)
    sub.add_parser('check',help='Metadata-only code/configuration/dependency inspection; safe on login')
    args = parser.parse_args(); os.environ['SP_EXECUTION_MODE'] = args.runtime
    if args.phase in ('prepare','score'): os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if args.phase == 'check':
        cfg = c.config(c.PACKAGE/'config.yaml')
        from importlib.metadata import version, PackageNotFoundError
        from .runtime import LIBRARIES
        versions = {}
        for name in LIBRARIES:
            try: versions[name] = version(name)
            except PackageNotFoundError: versions[name] = 'not installed'
        print(json.dumps({'configuration':'valid','cases':len(c.FAMILIES)*cfg['cases_per_family'],
              'candidate_images':2560,'external_judgments':4104,'training_updates':0,'versions':versions},indent=2)); return
    if args.phase == 'prepare':
        from .data import prepare
        prepare(args.images,args.model_dir,args.out,args.config)
    elif args.phase in ('canary','generate'):
        root = Path(args.out)
        prior = root/'generation_receipts'
        started = root/'generation_started.json'
        existing = (prior.exists() and any(prior.iterdir())) if args.phase=='canary' else started.exists()
        if existing and not args.resume:
            raise RuntimeError('Existing partial/completed work: explicitly pass --resume to retain it')
        from .worker import canary, generate
        with c.execution_lock(args.out,args.phase):
            (canary if args.phase=='canary' else generate)(args.model_dir,args.out)
    elif args.phase == 'score':
        from .scoring import score
        with c.execution_lock(args.out,args.phase): score(args.out)
    elif args.phase == 'summarize':
        with c.execution_lock(args.out,args.phase): cached_summary(args.out)
