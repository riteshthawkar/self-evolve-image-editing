"""Bias existing joint-attention logits; leave learned weights/positions unchanged.

The conditioning sequence has contextualized visual and textual tokens. The
semantic gate targets visual-token positions, not all image information in it.
Biases are broadcast over ALL queries and heads, in ALL 60 blocks and both CFG
branches. Unity returns without modifying kwargs or the original attention mask.
"""
from __future__ import annotations
import math


def key_bias(visual_mask, image_length, reference_length, semantic_scale, reference_scale, *, torch):
    if visual_mask.ndim != 2 or visual_mask.shape[0] != 1 or visual_mask.dtype != torch.bool:
        raise ValueError('One boolean conditioning-token mask is required')
    if not bool(visual_mask.any()) or not 0 < reference_length < image_length:
        raise ValueError('Missing visual tokens or invalid output/reference sequence lengths')
    if not 0 < semantic_scale <= 1 or not 0 < reference_scale <= 1:
        raise ValueError('Attention-mass scales must be in (0,1]')
    length = visual_mask.shape[1] + image_length
    result = torch.zeros((1, 1, 1, length), device=visual_mask.device, dtype=torch.float32)
    result[..., :visual_mask.shape[1]] = visual_mask[:, None, None, :] * math.log(semantic_scale)
    result[..., -reference_length:] += math.log(reference_scale)
    return result


def combine_mask(original, bias, *, torch):
    if original is None:
        return bias
    if original.shape[-1] != bias.shape[-1]:
        raise ValueError('Joint attention mask length mismatch')
    if original.dtype == torch.bool:
        additive = torch.zeros_like(original, dtype=torch.float32).masked_fill(~original, -torch.inf)
    else:
        additive = original.to(dtype=torch.float32)
    return additive + bias


class Intervention:
    def __init__(self, pipe):
        self.pipe = pipe
        self.semantic_scale = self.reference_scale = 1.0
        self.reference_length = None
        self.visual_masks = []
        self.calls = 0; self.changed_calls = 0
        self.handles = []
        self.attach()

    def attach(self):
        if self.handles: raise RuntimeError('Attention hooks already attached')
        blocks = self.pipe.transformer.transformer_blocks
        if len(blocks) != 60:
            raise ValueError('Expected the pinned 60-block Qwen transformer')
        for block in blocks:
            if type(block.attn.processor).__name__ != 'QwenDoubleStreamAttnProcessor2_0':
                raise ValueError('Unknown attention processor; no silent replacement')
            self.handles.append(block.attn.register_forward_pre_hook(self.hook, with_kwargs=True))

    def configure(self, semantic_scale=1.0, reference_scale=1.0):
        self.semantic_scale, self.reference_scale = semantic_scale, reference_scale
        self.calls = self.changed_calls = 0; self.visual_masks.clear(); self.reference_length = None

    def hook(self, module, args, kwargs):
        self.calls += 1
        if self.semantic_scale == 1.0 and self.reference_scale == 1.0:
            return None
        import torch
        image = kwargs.get('hidden_states')
        text = kwargs.get('encoder_hidden_states')
        if image is None or text is None or self.reference_length is None or not self.visual_masks:
            raise RuntimeError('Intervention lacks native token-layout metadata')
        # Pipeline calls positive, then negative CFG; each branch has 60 blocks.
        branch = ((self.calls - 1) // 60) % len(self.visual_masks)
        mask = self.visual_masks[branch]
        if mask.shape[1] != text.shape[1] or image.shape[0] != 1:
            raise RuntimeError('Conditioning branch/layout mismatch')
        bias = key_bias(mask, image.shape[1], self.reference_length,
                        self.semantic_scale, self.reference_scale, torch=torch)
        updated = dict(kwargs)
        updated['attention_mask'] = combine_mask(kwargs.get('attention_mask'), bias, torch=torch)
        self.changed_calls += 1
        return args, updated

    def receipt(self):
        return {'semantic_key_scale': self.semantic_scale, 'reference_key_scale': self.reference_scale,
                'attention_calls': self.calls, 'changed_attention_calls': self.changed_calls,
                'semantic_visual_tokens_per_branch': [int(m.sum()) for m in self.visual_masks],
                'reference_tokens': self.reference_length, 'queries': 'all', 'blocks': 60}

    def remove(self):
        for handle in self.handles: handle.remove()
        self.handles.clear()
