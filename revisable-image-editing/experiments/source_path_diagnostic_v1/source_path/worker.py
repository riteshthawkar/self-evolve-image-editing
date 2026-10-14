"""Finite, resumable diagnostic. Never trains or loads an adapted checkpoint."""
from __future__ import annotations
import json
import shutil
import time
from pathlib import Path
from . import common as c, native
from .attention import Intervention
from .runtime import gpu_identity, offline_network, require_allocation


def open_rgb(run_dir, entry):
    from PIL import Image, ImageOps
    path = c.asset(run_dir, entry['source'], entry['fingerprint']['encoded_sha256'])
    with Image.open(path) as handle: return ImageOps.exif_transpose(handle).convert('RGB')


def load_pipe(model_dir, cfg):
    require_allocation('gpu'); offline_network()
    import torch
    from diffusers import QwenImageEditPlusPipeline
    contract = json.loads((c.PACKAGE / 'assets/model_contract.json').read_text())
    for relative, row in contract['files'].items():
        path = Path(model_dir) / relative
        if not path.is_file() or path.stat().st_size != row['bytes']:
            raise ValueError('Incomplete pinned checkpoint: ' + relative)
        if not relative.endswith('.safetensors') and c.sha(path) != row['sha256']:
            raise ValueError('Pinned checkpoint metadata changed: ' + relative)
    torch.backends.cuda.enable_cudnn_sdp(False)
    pipe = QwenImageEditPlusPipeline.from_pretrained(str(Path(model_dir).resolve()),
        local_files_only=True, torch_dtype=torch.bfloat16)
    config = pipe.transformer.config
    if (config.num_layers != 60 or config.in_channels != 64 or config.joint_attention_dim != 3584
            or config.guidance_embeds or pipe.vae.config.z_dim != 16):
        raise ValueError('Unexpected editor architecture')
    for module in (pipe.text_encoder, pipe.vae, pipe.transformer): module.requires_grad_(False).eval()
    if cfg['cpu_offload']: pipe.enable_model_cpu_offload(device='cuda')
    else: pipe.to('cuda')
    pipe.set_progress_bar_config(disable=True)
    intervention = Intervention(pipe); native.install_conditioning(pipe, intervention)
    return pipe, intervention


def runtime_receipt(run_dir, cfg):
    identity = gpu_identity(cfg)
    path = run_dir / 'gpu_environment.json'
    if path.exists() and c.read(path)['identity'] != identity:
        raise RuntimeError('GPU/runtime changed during resume; preserve this run and use a new namespace')
    if not path.exists(): c.seal(path, {'identity': identity, 'utc': c.utc()})
    return identity


def generated(pipe, intervention, source, prompt, seed, run_dir, name, cfg, scales=(1.0, 1.0)):
    import torch
    receipt_path = run_dir / 'generation_receipts' / (name + '.json')
    contract = {'prompt_sha256': c.canonical(prompt), 'source': native.image_identity(source), 'seed': seed, 'scales': list(scales),
                'steps': cfg['inference_steps'], 'true_cfg_scale': cfg['true_cfg_scale']}
    if receipt_path.exists():
        row = c.read(receipt_path)
        if row['contract'] != contract: raise ValueError('Saved candidate has a different generation contract')
        c.asset(run_dir, row['image']['path'], row['image']['sha256'])
        return row
    pipe.maybe_free_model_hooks(); pipe._source_path_phase = 'generation'
    intervention.configure(*scales); start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    def finite_step(pipeline,index,timestep,values):
        if not bool(values['latents'].isfinite().all()):
            raise RuntimeError('Nonfinite latent state; stop without selecting this candidate')
        return values
    with torch.inference_mode():
        image = pipe(image=[source], prompt=prompt, negative_prompt=' ', num_inference_steps=cfg['inference_steps'],
            true_cfg_scale=cfg['true_cfg_scale'], num_images_per_prompt=1,
            generator=torch.Generator(device='cuda').manual_seed(seed), callback_on_step_end=finite_step,
            callback_on_step_end_tensor_inputs=['latents']).images[0]
    expected_calls = cfg['inference_steps'] * 2 * 60 if intervention.handles else 0
    if intervention.calls != expected_calls:
        raise RuntimeError('Unexpected attention call count; do not assume the CFG branch order')
    if scales != (1.0, 1.0) and intervention.changed_calls != expected_calls:
        raise RuntimeError('Not every declared attention call received the intervention')
    saved = c.save_image(run_dir, f'images/{name}.png', image); image.close()
    pipe._source_path_phase = 'idle'
    row = {'contract': contract, 'image': saved, 'hook': intervention.receipt(),
           'seconds': time.monotonic() - start,
           'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
           'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3}
    c.seal(receipt_path, row)
    return row


def calibration(pipe, run_dir, plan):
    from PIL import Image, ImageDraw, ImageOps
    cfg = plan['config']; rows = []
    for index, entry in enumerate(plan['control_sources']):
        source_path = f'controls/{index:03d}/source.png'
        source = open_rgb(run_dir, entry)
        square = ImageOps.fit(source, (512, 512)); source.close()
        left = (48 + index % 4 * 4, 48, 192 + index % 4 * 4, 192)
        right = (320, 48 + index % 3 * 4, 464, 192 + index % 3 * 4)
        draw = ImageDraw.Draw(square); draw.rectangle(left, fill='red'); draw.ellipse(right, fill='yellow')
        source = square; c.save_image(run_dir, source_path, source)
        case = {'prompt': 'Change the solid red square in the upper-left to blue. Keep the yellow circle in the upper-right and the rest of the image unchanged.',
                'source_state': 'upper-left square is red', 'target_state': 'upper-left square is blue',
                'preserve': ['the upper-right circle stays yellow', 'the rest of the photograph stays unchanged']}
        for kind in ('correct', 'unchanged', 'wrong', 'damage'):
            receipt = run_dir / 'control_receipts' / f'{index:03d}_{kind}.json'
            if receipt.exists(): rows.append(c.read(receipt)); continue
            candidate = source.copy(); draw = ImageDraw.Draw(candidate)
            if kind in ('correct', 'damage'): draw.rectangle(left, fill='blue')
            if kind == 'wrong': draw.rectangle(left, fill='green')
            if kind == 'damage': draw.ellipse(right, fill='purple')
            saved = c.save_image(run_dir, f'controls/{index:03d}/{kind}.png', candidate)
            result = native.verify(pipe, source, candidate, case, run_dir,
                                   f'native/controls/{index:03d}_{kind}', cfg)
            row = {'id': f'{index:03d}_{kind}', 'kind': kind, 'expected_accept': kind == 'correct',
                   'image': saved, 'verification': result}
            c.seal(receipt, row); rows.append(row); candidate.close()
        source.close(); c.progress(run_dir, 'calibration', sources=index+1, total=len(plan['control_sources']))
    correct = [r for r in rows if r['expected_accept']]; bad = [r for r in rows if not r['expected_accept']]
    correct_rate = sum(r['verification']['accepted'] for r in correct) / len(correct)
    bad_rate = sum(r['verification']['accepted'] for r in bad) / len(bad)
    accuracy = sum(r['verification']['accepted'] == r['expected_accept'] for r in rows) / len(rows)
    passed = (correct_rate >= cfg['minimum_control_accuracy'] and accuracy >= cfg['minimum_control_accuracy']
              and bad_rate <= cfg['maximum_bad_control_acceptance'])
    result = {'status': 'complete', 'passed': passed, 'correct_acceptance': correct_rate,
              'bad_acceptance': bad_rate, 'accuracy': accuracy, 'controls': len(rows),
              'scope': 'known source/draft interface controls only; not proof of natural semantic validity'}
    c.seal(run_dir / 'calibration.json', result)
    if not passed: raise RuntimeError('Native control gate failed; do not run the full diagnostic')
    return result


def proposal(pipe, run_dir, row, cfg):
    path = run_dir / 'proposal_receipts' / (row['id'] + '.json')
    if path.exists():
        saved = c.read(path)
        if saved['source_sha256'] != row['fingerprint']['encoded_sha256'] or saved['family'] != row['family']:
            raise ValueError('Cached proposal belongs to a different source or task')
        return saved['proposal']
    source = open_rgb(run_dir, row)
    try: value = native.propose(pipe, source, row['family'], run_dir, f'native/proposals/{row["id"]}.json', cfg)
    finally: source.close()
    c.seal(path, {'source_id': row['id'], 'source_sha256': row['fingerprint']['encoded_sha256'], 'family': row['family'], 'proposal': value})
    return value


def canary(model_dir, run_dir):
    run_dir, plan = c.load_run(run_dir); cfg = plan['config']; runtime_receipt(run_dir, cfg)
    if (run_dir / 'canary.json').exists():
        result = c.read(run_dir / 'canary.json')
        if result['passed']: print('Canary already complete; recorded images retained.'); return
        raise RuntimeError('Canary failed; no silent retry')
    pipe, intervention = load_pipe(model_dir, cfg)
    try:
        calibration(pipe, run_dir, plan)
        selected = None
        for row in (r for r in plan['sources'] if r['family'] == c.FAMILIES[0]):
            value = proposal(pipe, run_dir, row, cfg)
            if value is not None: selected = {**row, **value}; break
        if selected is None: raise RuntimeError('No eligible canary source in the frozen addition pool')
        source = open_rgb(run_dir, selected)
        intervention.remove()
        plain = generated(pipe, intervention, source, selected['prompt'], cfg['seed'], run_dir, 'canary/unhooked', cfg)
        intervention.attach()
        unity = generated(pipe, intervention, source, selected['prompt'], cfg['seed'], run_dir, 'canary/unity', cfg)
        from PIL import Image
        with Image.open(c.asset(run_dir, plain['image']['path'])) as a, Image.open(c.asset(run_dir, unity['image']['path'])) as b:
            parity = a.size == b.size and a.convert('RGB').tobytes() == b.convert('RGB').tobytes()
        if not parity: raise RuntimeError('Unity attention hook changed baseline pixels')
        trials = []
        for name, scales in [('semantic', (.5, 1.)), ('reference', (1., .5)), ('both', (.5, .5))]:
            trials.append(generated(pipe, intervention, source, selected['prompt'], cfg['seed'], run_dir, 'canary/'+name, cfg, scales))
        source.close()
        if not all(row['hook']['changed_attention_calls'] == 4800 for row in trials):
            raise RuntimeError('Canary intervention did not run on both CFG branches')
        seconds = [row['seconds'] for row in [plain, unity, *trials]]
        result = {'status': 'complete', 'passed': True, 'unity_rgb_identical': True,
                  'canary_source': selected['id'], 'generation_seconds': seconds,
                  'maximum_peak_reserved_gib': max(row['peak_reserved_gib'] for row in [plain, unity, *trials]),
                  'full_candidate_images': 256 * 10, 'estimated_generation_hours': sum(seconds)/len(seconds)*256*10/3600,
                  'estimate_excludes_acquisition_and_verification': True, 'cpu_offload': cfg['cpu_offload']}
        c.seal(run_dir / 'canary.json', result); print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        if not (run_dir / 'canary.json').exists():
            c.seal(run_dir / 'canary.json', {'status': 'failed', 'passed': False, 'error_type': type(error).__name__})
        raise
    finally:
        intervention.remove(); pipe.maybe_free_model_hooks()


def acquire(pipe, run_dir, plan):
    path = run_dir / 'cases.json'
    if path.exists(): return c.read(path)['cases']
    cases = []; cfg = plan['config']
    for family in c.FAMILIES:
        admitted = []
        for row in (r for r in plan['sources'] if r['family'] == family):
            value = proposal(pipe, run_dir, row, cfg)
            if value is not None: admitted.append({**row, **value})
            c.progress(run_dir, 'source_acquisition', family=family, accepted=len(admitted), required=cfg['cases_per_family'])
            if len(admitted) == cfg['cases_per_family']: break
        if len(admitted) != cfg['cases_per_family']:
            c.seal(run_dir / 'acquisition_failure.json', {'family': family, 'accepted': len(admitted), 'required': cfg['cases_per_family'],
                   'reason': 'fixed source-pool quota not met; no replacements or category deletion'})
            raise RuntimeError('Source-family quota failed: ' + family)
        cases.extend(admitted)
    c.seal(path, {'status': 'complete', 'cases': cases, 'external_models': 0, 'external_api_calls': 0})
    return cases


def candidate_definitions(case, cfg):
    prompt = case['prompt']; seed = cfg['seed']
    definitions = [('base42', prompt, seed, (1., 1.)), ('semantic42', prompt, seed, (.5, 1.)),
                   ('reference42', prompt, seed, (1., .5)), ('both42', prompt, seed, (.5, .5))]
    definitions += [(f'base{s}', prompt, s, (1., 1.)) for s in (101, 202, 303)]
    clarify = 'Carry out exactly this request. Preserve source content and relationships except what this request necessarily changes: ' + prompt
    definitions += [(f'clarify{s}', clarify, s, (1., 1.)) for s in (101, 202, 303)]
    return definitions


def generate(model_dir, run_dir):
    run_dir, plan = c.load_run(run_dir); cfg = plan['config']; runtime_receipt(run_dir, cfg)
    if not c.read(run_dir / 'canary.json')['passed']: raise RuntimeError('A passing target-machine canary is required')
    if (run_dir / 'generation_complete.json').exists():
        print('Generation already complete; use score.'); return
    if shutil.disk_usage(run_dir).free < cfg['minimum_disk_free_gib'] * 1024**3:
        raise RuntimeError('Insufficient output storage; no files are deleted')
    pipe, intervention = load_pipe(model_dir, cfg)
    try:
        if not (run_dir/'generation_started.json').exists():
            c.seal(run_dir/'generation_started.json', {'started': c.utc(), 'plan_sha256': c.sha(run_dir/'plan.json')})
        cases = acquire(pipe, run_dir, plan)
        for index, case in enumerate(cases):
            path = run_dir / 'case_receipts' / (case['id']+'.json')
            if path.exists():
                saved = c.read(path)
                if saved['case'] != case: raise ValueError('Committed case identity changed')
                for row in saved['candidates'].values(): c.asset(run_dir, row['image']['path'], row['image']['sha256'])
                continue
            source = open_rgb(run_dir, case); images = {}; verification = {}
            for name, prompt, seed, scales in candidate_definitions(case, cfg):
                image = generated(pipe, intervention, source, prompt, seed, run_dir, f'{case["id"]}/{name}', cfg, scales)
                images[name] = image
                verification_path = run_dir / 'verification_receipts' / f'{case["id"]}_{name}.json'
                if verification_path.exists(): verification[name] = c.read(verification_path)
                else:
                    from PIL import Image
                    with Image.open(c.asset(run_dir, image['image']['path'], image['image']['sha256'])) as handle:
                        candidate = handle.convert('RGB')
                    verification[name] = native.verify(pipe, source, candidate, case, run_dir,
                        f'native/candidates/{case["id"]}_{name}', cfg)
                    candidate.close(); c.seal(verification_path, verification[name])
            source.close()
            pools = {'selected': ['base42','semantic42','reference42','both42'],
                     'resample': ['base42','base101','base202','base303'],
                     'clarify': ['base42','clarify101','clarify202','clarify303']}
            arms = {'base': 'base42', 'semantic': 'semantic42', 'reference': 'reference42', 'both': 'both42'}
            selection = {}
            for arm, pool in pools.items():
                choice, accepted = c.choose(pool, verification)
                arms[arm] = choice; selection[arm] = {'candidate': choice, 'accepted': accepted, 'pool_size': 4}
            c.seal(path, {'case': case, 'candidates': images, 'verification': verification, 'arms': arms, 'selection': selection})
            c.progress(run_dir, 'generation', completed=index+1, total=len(cases), images_per_case=10)
        c.seal(run_dir / 'generation_complete.json', {'status': 'complete', 'cases': len(cases), 'candidate_images': len(cases)*10,
                'training_updates': 0, 'external_acquisition_models': 0, 'external_acquisition_api_calls': 0})
    finally:
        intervention.remove(); pipe.maybe_free_model_hooks()
