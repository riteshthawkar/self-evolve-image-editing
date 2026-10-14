"""CPU-only source preparation; no captions, annotations or model calls."""
from __future__ import annotations
import hashlib
import io
import json
import struct
import shutil
from pathlib import Path
from . import common as c
from .runtime import require_allocation


def fingerprint(path):
    from PIL import Image, ImageOps
    raw = Path(path).read_bytes()
    with Image.open(io.BytesIO(raw)) as handle:
        if handle.width * handle.height > 50_000_000:
            raise ValueError('Source exceeds decode limit')
        image = ImageOps.exif_transpose(handle).convert('RGB')
    try:
        w, h = image.size
        rgb_sha = hashlib.sha256(b'RGB-v1\0' + struct.pack('>II', w, h) + image.tobytes()).hexdigest()
        small = image.convert('L').resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(small.getdata()); bits = 0
        for row in range(8):
            for col in range(8):
                bits = (bits << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
        small.close()
        return {'encoded_sha256': hashlib.sha256(raw).hexdigest(), 'decoded_rgb_sha256': rgb_sha,
                'dhash64': f'{bits:016x}', 'width': w, 'height': h}
    finally:
        image.close()


class Exclusions:
    def __init__(self, values):
        self.encoded = set(); self.decoded = set(); self.hashes = []
        for value in values: self.add(value)

    def add(self, value):
        self.encoded.add(value['encoded_sha256']); self.decoded.add(value['decoded_rgb_sha256'])
        self.hashes.append(int(value['dhash64'], 16))

    def contains(self, value):
        h = int(value['dhash64'], 16)
        return (value['encoded_sha256'] in self.encoded or value['decoded_rgb_sha256'] in self.decoded
                or any((h ^ other).bit_count() <= 4 for other in self.hashes))


def verify_model(model_dir):
    model_dir = Path(model_dir).resolve()
    contract = json.loads((c.PACKAGE / 'assets/model_contract.json').read_text())
    for relative, expected in contract['files'].items():
        path = model_dir / relative
        if not path.is_file() or path.stat().st_size != expected['bytes'] or c.sha(path) != expected['sha256']:
            raise ValueError('Pinned model file absent or changed: ' + relative)
    return {'model': c.MODEL, 'revision': c.REVISION, 'contract_sha256': c.sha(c.PACKAGE / 'assets/model_contract.json'),
            'verified_files': len(contract['files']), 'weight_hashes_verified': True}


def prepare(images, model_dir, run_dir, config_path):
    require_allocation('cpu')
    cfg = c.config(config_path); run_dir = Path(run_dir).resolve(); images = Path(images).resolve()
    if run_dir.exists(): raise FileExistsError('Choose a new output directory; preparation never overwrites')
    if not images.is_dir(): raise ValueError('Unlabeled source-image folder is missing')
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(run_dir.parent).free < cfg['minimum_disk_free_gib'] * 1024**3:
        raise RuntimeError('Insufficient output storage')
    model_receipt = verify_model(model_dir)
    protected = json.loads((c.PACKAGE / 'assets/protected_sources.json').read_text())
    exclusion = Exclusions(protected['fingerprints'])
    options = [p for p in images.rglob('*') if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')]
    # Relative filenames determine ordering before model outcomes. No annotations are loaded.
    options.sort(key=lambda p: c.canonical(['source-order-v1', cfg['seed'], str(p.relative_to(images))]))
    needed = len(c.FAMILIES) * cfg['maximum_sources_per_family'] + cfg['controls_per_kind']
    selected = []; rejected = []
    for path in options:
        try: fp = fingerprint(path)
        except (OSError, ValueError) as error:
            rejected.append({'relative_path': str(path.relative_to(images)), 'reason': type(error).__name__}); continue
        if exclusion.contains(fp):
            rejected.append({'relative_path': str(path.relative_to(images)), 'reason': 'protected_or_duplicate_source'}); continue
        selected.append({'path': path, 'fingerprint': fp}); exclusion.add(fp)
        if len(selected) == needed: break
    if len(selected) != needed:
        raise ValueError(f'Need {needed} fresh source-disjoint photographs; found {len(selected)}. No quota reduction.')
    run_dir.mkdir(parents=True, exist_ok=False); sources = []; controls = []
    for index, row in enumerate(selected):
        relative = f'sources/{index:05d}{row["path"].suffix.lower()}'
        target = c.asset(run_dir, relative); target.parent.mkdir(exist_ok=True)
        shutil.copyfile(row['path'], target)
        value = {'id': f's{index:05d}', 'source': relative, 'fingerprint': row['fingerprint']}
        if index < cfg['controls_per_kind']: controls.append(value)
        else:
            family_index = (index - cfg['controls_per_kind']) // cfg['maximum_sources_per_family']
            sources.append({**value, 'family': c.FAMILIES[family_index]})
    plan = {'schema': 'source-path-diagnostic/v1', 'config': cfg, 'config_sha256': c.canonical(cfg),
            'code': c.code_identity(), 'sources': sources, 'control_sources': controls,
            'model': model_receipt, 'protected_fingerprint_count': len(protected['fingerprints']),
            'source_exclusions': rejected, 'family_denominator': cfg['cases_per_family'],
            'external_acquisition_models': 0, 'external_acquisition_api_calls': 0,
            'training_updates': 0, 'automatic_training': False, 'utc': c.utc()}
    c.seal(run_dir / 'plan.json', plan)
    c.seal(run_dir / 'prepare_complete.json', {'status': 'complete', 'plan_sha256': c.sha(run_dir / 'plan.json')})
    print('Prepared portable run:', run_dir, 'candidate sources:', len(sources), flush=True)
