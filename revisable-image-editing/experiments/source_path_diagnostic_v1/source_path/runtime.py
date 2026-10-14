"""Allocation checks, offline generation, and portable environment receipts."""
from __future__ import annotations
import getpass
import os
import socket
import subprocess
import sys
import json
from importlib.metadata import version
from pathlib import Path
from .common import sha, PACKAGE

LIBRARIES = ('torch', 'torchvision', 'diffusers', 'transformers', 'accelerate',
             'safetensors', 'huggingface-hub', 'Pillow', 'numpy', 'openai', 'PyYAML', 'peft')


def require_allocation(kind):
    mode = os.environ.get('SP_EXECUTION_MODE', 'slurm')
    if mode == 'dedicated':
        origin = json.loads((PACKAGE / 'assets/source_machine.json').read_text())
        if socket.gethostname() == origin['hostname']:
            raise RuntimeError('Direct execution is authorized on the new dedicated machine, not this shared cluster')
        if os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Use Slurm mode inside an allocation')
        if kind == 'gpu' and os.environ.get('CUDA_VISIBLE_DEVICES') is None:
            raise RuntimeError('Set CUDA_VISIBLE_DEVICES to exactly one dedicated GPU')
        if kind == 'cpu' and os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
            raise RuntimeError('CPU preparation/scoring must explicitly hide GPUs')
        return {'mode': mode, 'user': getpass.getuser(), 'hostname': socket.gethostname()}
    if mode != 'slurm':
        raise ValueError('Execution mode must be slurm or dedicated')
    job = os.environ.get('SLURM_JOB_ID', '')
    step = os.environ.get('SLURM_STEP_ID', '')
    if not job.isdigit() or not step.isdigit() or not os.environ.get('TMUX'):
        raise RuntimeError('Use sbatch from inspected tmux and a numeric srun step; never the login node')
    result = subprocess.check_output(['scontrol', 'show', 'job', '-o', job], text=True)
    fields = dict(v.split('=', 1) for v in result.split() if '=' in v)
    if fields.get('JobState') != 'RUNNING' or not fields.get('UserId', '').startswith(getpass.getuser() + '('):
        raise RuntimeError('Allocation is not RUNNING and owned by this user')
    prefix = os.environ.get('CONDA_PREFIX', '')
    if not prefix or Path(sys.executable).resolve() != (Path(prefix) / 'bin/python').resolve():
        raise RuntimeError('Activate the chosen conda environment inside the allocation')
    gpu = any(v.startswith('gres/gpu') for v in fields.get('AllocTRES', '').split(','))
    if (kind == 'gpu') != gpu:
        raise RuntimeError('GPU generation requires one GPU; preparation/API scoring requires zero GPUs')
    if kind == 'cpu' and os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
        raise RuntimeError('CPU worker must explicitly hide CUDA devices')
    old_root = Path('/share_6/users/ritesh_thawkar/self-evolve-image-editing')
    if Path(__file__).resolve().is_relative_to(old_root):
        if getpass.getuser() != 'ritesh_thawkar' or Path(prefix).resolve() != Path('/share_6/users/ritesh_thawkar/condaenvs/qedit'):
            raise RuntimeError('This cluster requires ritesh_thawkar and the existing qedit environment')
    return {'job': job, 'step': step, 'user': getpass.getuser(), 'hostname': socket.gethostname()}


def environment():
    import diffusers, transformers
    files = [Path(diffusers.__file__).parent / 'pipelines/qwenimage/pipeline_qwenimage_edit_plus.py',
             Path(diffusers.__file__).parent / 'models/transformers/transformer_qwenimage.py',
             Path(transformers.__file__).parent / 'models/qwen2_5_vl/modeling_qwen2_5_vl.py']
    return {'versions': {name: version(name) for name in LIBRARIES},
            'implementation_hashes': {p.name: sha(p) for p in files}}


def gpu_identity(cfg):
    require_allocation('gpu')
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one Slurm-assigned GPU is required')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('The diagnostic requires native BF16 support')
    p = torch.cuda.get_device_properties(0)
    if p.total_memory < cfg['minimum_gpu_memory_gib'] * 1024**3:
        raise RuntimeError('Assigned GPU is smaller than the configured memory requirement')
    return {**environment(), 'gpu_name': p.name, 'gpu_memory_bytes': p.total_memory,
            'cuda_runtime': torch.version.cuda}


def offline_network():
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    original, original_ex = socket.socket.connect, socket.socket.connect_ex
    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError('External networking is forbidden during acquisition and generation')
        return original(sock, address)
    def connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError('External networking is forbidden during acquisition and generation')
        return original_ex(sock, address)
    socket.socket.connect, socket.socket.connect_ex = connect, connect_ex
