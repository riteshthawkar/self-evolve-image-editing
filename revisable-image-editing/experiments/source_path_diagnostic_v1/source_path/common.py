from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager

PACKAGE = Path(__file__).resolve().parents[1]
MODEL = 'Qwen/Qwen-Image-Edit-2509'
REVISION = 'd3968ef930e841f4c73640fb8afa3b306a78167e'
FAMILIES = ('add', 'remove', 'replace', 'attribute', 'background', 'style', 'text', 'pose')
FIELDS = ('semantic', 'instance_fidelity', 'preservation', 'quality', 'overall')
CHECKS = ('requested_change', 'old_state_resolved', 'instance_retained', 'protected_content_retained', 'no_new_defects')
PAIR_ARMS = {
    'semantic_vs_base': ('semantic', 'base'),
    'reference_vs_base': ('reference', 'base'),
    'both_vs_base': ('both', 'base'),
    'both_vs_semantic': ('both', 'semantic'),
    'both_vs_reference': ('both', 'reference'),
    'selected_vs_base': ('selected', 'base'),
    'selected_vs_resample': ('selected', 'resample'),
    'selected_vs_clarify': ('selected', 'clarify'),
}
PRIMARY_PAIRS = ('selected_vs_base', 'selected_vs_resample', 'selected_vs_clarify')


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def seal(path, value):
    if 'payload_sha256' in value:
        raise ValueError('Already sealed')
    write(path, {**value, 'payload_sha256': canonical(value)})


def read(path):
    value = json.loads(Path(path).read_text())
    digest = value.pop('payload_sha256', None)
    if digest != canonical(value):
        raise ValueError('Artifact seal mismatch: ' + str(path))
    return value


def code_identity():
    paths = [*PACKAGE.glob('*.py'), *PACKAGE.glob('*.yaml'), *PACKAGE.glob('*.json'),
             *PACKAGE.glob('*.sh'), *PACKAGE.glob('*.slurm'), *PACKAGE.glob('*.md'),
             *PACKAGE.glob('requirements*.txt'), *PACKAGE.glob('environment*.txt'),
             *PACKAGE.glob('source_path/*.py'), *PACKAGE.glob('assets/*.json')]
    return {str(p.relative_to(PACKAGE)): sha(p) for p in sorted(paths)}


def config(path):
    import yaml
    value = yaml.safe_load(Path(path).read_text())
    if value['model'] != MODEL or value['revision'] != REVISION:
        raise ValueError('Only the pinned native Qwen editor is supported')
    for field in ('cases_per_family', 'maximum_sources_per_family', 'controls_per_kind',
                  'inference_steps', 'native_max_new_tokens', 'bootstrap_samples', 'judge_workers'):
        if type(value[field]) is not int or value[field] <= 0:
            raise ValueError('Invalid integer: ' + field)
    if value['seed'] != 42 or value['native_verification_orders'] != 2 or value['inference_steps'] != 40 or value['true_cfg_scale'] != 4.0:
        raise ValueError('The v1 diagnostic fixes two verification orders, 40 steps and true CFG 4')
    if value['cases_per_family'] != 32 or value['controls_per_kind'] != 32:
        raise ValueError('The v1 protocol fixes 256 natural cases and 128 controls')
    for field in ('semantic_key_scale', 'reference_key_scale'):
        if value[field] != 0.5:
            raise ValueError('The v1 factorial uses predeclared attention-mass scale 0.5')
    if value['judge_model'] != 'gpt-4.1-2025-04-14' or not 0 < value['judge_max_usd'] <= 40:
        raise ValueError('Unexpected evaluation judge/budget')
    return value


def load_run(run_dir):
    run_dir = Path(run_dir).resolve()
    plan = read(run_dir / 'plan.json')
    if plan['code'] != code_identity():
        raise ValueError('Experiment code changed; use a new version and output directory')
    if canonical(plan['config']) != plan['config_sha256']:
        raise ValueError('Configuration identity changed')
    return run_dir, plan


def asset(run_dir, relative, digest=None):
    root = Path(run_dir).resolve(); path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Artifact path escapes run directory')
    if digest is not None and sha(path) != digest:
        raise ValueError('Artifact image changed: ' + relative)
    return path


def save_image(run_dir, relative, image):
    path = asset(run_dir, relative); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    image.save(tmp, format='PNG'); os.replace(tmp, path)
    return {'path': relative, 'sha256': sha(path), 'size': list(image.size)}


def require_native_checks(value):
    if not isinstance(value, dict) or any(value.get(k) not in (True, False, 'unknown') for k in CHECKS):
        return None
    # Python equality lets 1 == True through; explicitly exclude numbers.
    if any(type(value[k]) not in (bool, str) for k in CHECKS):
        return None
    return {k: value[k] for k in CHECKS}


def choose(pool, receipts):
    valid = [name for name in pool if receipts[name]['accepted']]
    # Pool order is preregistered. No reward magnitude or external judgment enters selection.
    return (valid[0], True) if valid else ('base42', False)


def progress(run_dir, stage, **values):
    seal(Path(run_dir) / 'progress.json', {'stage': stage, 'utc': utc(), **values})
    print(stage, json.dumps(values), flush=True)


@contextmanager
def execution_lock(run_dir, stage):
    """One process per run; OS releases the advisory lock after interruption."""
    import fcntl
    path = Path(run_dir) / '.execution.lock'
    with path.open('a+') as handle:
        try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another process owns this run; do not launch duplicate workers') from None
        handle.seek(0); handle.truncate(); handle.write(json.dumps({'pid':os.getpid(),'stage':stage})); handle.flush()
        try: yield
        finally: fcntl.flock(handle, fcntl.LOCK_UN)
