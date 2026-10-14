"""Only the original editor's bundled Qwen2.5-VL is used for acquisition."""
from __future__ import annotations
import json
import hashlib
from types import MethodType
from . import common as c
from .runtime import require_allocation

FAMILY_RULES = {
    'add': 'Add one clearly specified object that is not already present.',
    'remove': 'Remove one clearly visible object. Preserve other subjects and the scene.',
    'replace': 'Replace one clearly visible object with a different object at the same position.',
    'attribute': 'Change one visible subject color or material to a different explicit value.',
    'background': 'Change the background setting while preserving the main subject identity.',
    'style': 'Change the artistic rendering style while preserving scene subjects and composition.',
    'text': 'Replace a clearly legible existing word or short phrase with an explicit different string.',
    'pose': 'Change one clearly visible human or animal pose/action while preserving its identity.',
}


def image_identity(image):
    value = image.convert('RGB')
    return {'size': list(value.size), 'rgb_sha256': hashlib.sha256(value.tobytes()).hexdigest()}


def parse_native_json(raw):
    def unique(items):
        value = {}
        for key,item in items:
            if key in value: raise ValueError('Duplicate native JSON field')
            value[key] = item
        return value
    def constant(value): raise ValueError('Nonfinite native JSON constant')
    try: value = json.loads(raw, object_pairs_hook=unique, parse_constant=constant)
    except (ValueError,TypeError): return {'unknown': True}
    return value if isinstance(value,dict) else {'unknown':True}


def install_conditioning(pipe, intervention):
    """Preserve the native template/extraction, but explicitly pass visual modality IDs."""
    if pipe.prompt_template_encode_start_idx != 64:
        raise ValueError('Unexpected native prompt-prefix extraction')
    def get_embeds(self, prompt=None, image=None, device=None, dtype=None):
        require_allocation('gpu')
        import torch
        device = device or self._execution_device; dtype = dtype or self.text_encoder.dtype
        prompts = [prompt] if isinstance(prompt, str) else prompt
        images = image if isinstance(image, list) else ([] if image is None else [image])
        prefix = ''.join(f'Picture {i+1}: <|vision_start|><|image_pad|><|vision_end|>' for i in range(len(images)))
        text = [self.prompt_template_encode.format(prefix + p) for p in prompts]
        batch = self.processor(text=text, images=image, padding=True, return_tensors='pt',
                               return_mm_token_type_ids=True).to(device)
        modality = batch.get('mm_token_type_ids')
        if modality is None or modality.shape != batch['input_ids'].shape:
            raise RuntimeError('Native processor must expose explicit modality IDs')
        outputs = self.text_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
            pixel_values=batch.get('pixel_values'), image_grid_thw=batch.get('image_grid_thw'),
            mm_token_type_ids=modality, output_hidden_states=True, use_cache=False, past_key_values=None)
        last = outputs.hidden_states[-1]; values = []; masks = []
        for i, attention in enumerate(batch['attention_mask']):
            valid = attention.bool()
            values.append(last[i][valid][64:])
            masks.append((batch['input_ids'][i][valid] == self.text_encoder.config.image_token_id)[64:])
        length = max(v.shape[0] for v in values)
        embeddings = last.new_zeros((len(values), length, last.shape[-1]))
        attention = torch.zeros((len(values), length), device=device, dtype=torch.long)
        visual = torch.zeros((len(values), length), device=device, dtype=torch.bool)
        for i, value in enumerate(values):
            embeddings[i, :value.shape[0]] = value; attention[i, :value.shape[0]] = 1
            visual[i, :value.shape[0]] = masks[i]
        if not bool(visual.any()) or not bool(embeddings.isfinite().all()):
            raise RuntimeError('No finite native visual conditioning')
        intervention.visual_masks.append(visual)
        return embeddings.to(dtype=dtype), attention
    pipe._get_qwen_prompt_embeds = MethodType(get_embeds, pipe)
    original = pipe.prepare_latents
    def prepare(self, *args, **kwargs):
        result = original(*args, **kwargs)
        if result[1] is None or result[1].ndim != 3:
            raise RuntimeError('Exactly one packed source reference is required')
        intervention.reference_length = result[1].shape[1]
        return result
    pipe.prepare_latents = MethodType(prepare, pipe)


def observe(pipe, images, question, run_dir, receipt_path, cfg):
    require_allocation('gpu')
    import torch
    # On a 48GB GPU the denoiser and VLM must not be resident together.
    if getattr(pipe,'_source_path_phase',None) != 'native':
        pipe.maybe_free_model_hooks()
    pipe._source_path_phase = 'native'
    content = [{'type': 'image', 'image': 'image'} for _ in images]
    content.append({'type': 'text', 'text': question})
    messages = [{'role': 'system', 'content': 'Report visible evidence only. Return one JSON object, no markdown. Use unknown when uncertain.'},
                {'role': 'user', 'content': content}]
    text = pipe.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    views = []
    for image in images:
        view = image.convert('RGB').copy(); view.thumbnail((768, 768)); views.append(view)
    try:
        batch = pipe.processor(text=[text], images=views, padding=True, return_tensors='pt',
                               return_mm_token_type_ids=True).to('cuda')
        if batch.get('mm_token_type_ids') is None:
            raise RuntimeError('Native observation omitted modality IDs')
        with torch.inference_mode():
            tokens = pipe.text_encoder.generate(**batch, do_sample=False, max_new_tokens=cfg['native_max_new_tokens'], use_cache=False)
        raw = pipe.processor.batch_decode(tokens[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
        if raw.startswith('```'):
            raw = raw.removeprefix('```json').removeprefix('```').removesuffix('```').strip()
        value = parse_native_json(raw)
        receipt = {'question': question, 'raw_response': raw, 'parsed': value,
                   'input_images': [image_identity(image) for image in images],
                   'native_view_sizes': [list(view.size) for view in views],
                   'model': c.MODEL, 'revision': c.REVISION, 'component': 'text_encoder',
                   'external_models': 0, 'external_api_calls': 0}
        c.seal(c.asset(run_dir, receipt_path), receipt)
        return value
    finally:
        for view in views: view.close()
        # Keep the VLM resident for consecutive native checks. The next generation
        # explicitly frees it before loading the denoiser on the 48GB device.


def parse_proposal(value, family):
    if value.get('eligible') is not True: return None
    keys = ('instruction', 'subject', 'source_state', 'target_state')
    if any(not isinstance(value.get(k), str) or not value[k].strip() or len(value[k]) > 500 for k in keys):
        return None
    preserve = value.get('preserve')
    if not isinstance(preserve, list) or not 1 <= len(preserve) <= 5 or any(not isinstance(v, str) or not v.strip() or len(v)>200 for v in preserve):
        return None
    if value['source_state'].strip().lower() == value['target_state'].strip().lower(): return None
    if any('unknown' in value[k].lower() for k in keys): return None
    return {'family': family, 'prompt': value['instruction'].strip(), 'subject': value['subject'].strip(),
            'source_state': value['source_state'].strip(), 'target_state': value['target_state'].strip(),
            'preserve': [v.strip() for v in preserve]}


def propose(pipe, source, family, run_dir, receipt_path, cfg):
    question = ('Propose one feasible editing request grounded in this photograph. ' + FAMILY_RULES[family] +
        ' Do not change an already satisfied state. Preserve only facts outside the requested change; do not list the old state as immutable. '
        'Return {"eligible":true,"instruction":"self-contained English edit instruction",'
        '"subject":"visible subject or scene", "source_state":"observed current state",'
        '"target_state":"explicit intended new state", "preserve":["applicable protected fact"]}. '
        'Return {"eligible":false} if the source is unsuitable or uncertain. No bounding boxes or masks.')
    return parse_proposal(observe(pipe, [source], question, run_dir, receipt_path, cfg), family)


def verification_question(case, reverse=False):
    first, second = ('second', 'first') if reverse else ('first', 'second')
    return (f'The {first} image is SOURCE and the {second} is CANDIDATE. Assess the candidate as an edit of SOURCE. '
        f'Original request: {json.dumps(case["prompt"])}. '
        f'Intended new state: {json.dumps(case["target_state"])}. '
        f'Original state to resolve when applicable: {json.dumps(case["source_state"])}. '
        f'Protected facts: {json.dumps(case["preserve"])}. '
        'Allow changes necessarily requested: style can change texture and colors; pose can change geometry; '
        'replacement can change the specified object identity; background edits can change the whole background. '
        'Do not accept an unchanged image just because it is attractive. Do not require the old mutable state to remain. '
        'For old_state_resolved, use true when the request does not require eliminating an old state, such as addition. '
        'Return {"requested_change":true/false/"unknown","old_state_resolved":true/false/"unknown",'
        '"instance_retained":true/false/"unknown","protected_content_retained":true/false/"unknown",'
        '"no_new_defects":true/false/"unknown"}. For instance_retained assess only identities the request requires retaining.')


def verify(pipe, source, candidate, case, run_dir, prefix, cfg):
    answers = []
    for order in range(2):
        path = f'{prefix}_order{order}.json'
        if c.asset(run_dir, path).exists():
            old = c.read(c.asset(run_dir, path))
            expected = [image_identity(candidate),image_identity(source)] if order else [image_identity(source),image_identity(candidate)]
            if old['question'] != verification_question(case,bool(order)) or old['input_images'] != expected:
                raise ValueError('Cached native observation belongs to different images or instruction')
            value = old['parsed']
        else:
            views = [candidate, source] if order else [source, candidate]
            value = observe(pipe, views, verification_question(case, bool(order)), run_dir, path, cfg)
        answers.append(c.require_native_checks(value))
    valid = all(value is not None for value in answers)
    accepted = valid and all(value[k] is True for value in answers for k in c.CHECKS)
    return {'accepted': bool(accepted), 'schema_valid': valid, 'orders': answers,
            'external_models': 0, 'external_judgments_used': False}
