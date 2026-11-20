"""
PyTorch backend for src/embed.py, used where MLX is unavailable (Linux x86-64).

It runs the same network as the MLX path -- a plain Qwen3 backbone, no lm_head --
on the same token ids, padded the same way, and pools the same position (the
appended <|endoftext|>), so it produces the same vectors up to floating-point
rounding. A corpus embedded on a Mac can be searched on Linux and vice versa:
`chunk_embeddings.model` stays the Hugging Face repo id, and `text_hash` covers
only the embedded text, so switching backend never marks chunks as pending.

Device and precision:
  - a CUDA GPU of compute capability 8.0 or newer (Ampere, Ada, Hopper, ...):
    bfloat16, as on MLX;
  - an older CUDA GPU (Turing, Volta, Pascal): float32. PyTorch's fused attention
    kernels handle bf16 only from 8.0 on; below that, bf16 attention falls back to
    the "math" kernel, which materialises the full length x length score matrix
    (about 4 GiB for two 8K-token chunks) and runs out of memory. In fp32 the
    memory-efficient kernel is available, so attention memory stays linear.
    fp16 would also work there but risks overflow in Qwen3's activations;
  - no GPU: the CPU, in float32 (most x86 CPUs have no fast bf16 matmul).
Override with SCIX_EMBED_DEVICE (cpu, cuda, cuda:1, ...) and SCIX_EMBED_DTYPE
(bfloat16, float16, float32).

Memory: batches arrive sized by config.EMBED_BATCH_TOKENS / EMBED_MAX_BATCH. If a
forward pass still runs out of GPU memory, the batch is split in half (with its
padding trimmed) and retried, and the smaller size is remembered for the rest of
the run, so a too-large setting costs a few retries, not the run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Must be set before the first CUDA allocation; it reduces fragmentation, which the
# variable-length batches here cause. Respect a value the user already set.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch
from transformers import AutoModel

BACKEND = 'torch'

_DTYPES = {
    'bfloat16': torch.bfloat16,
    'bf16': torch.bfloat16,
    'float16': torch.float16,
    'fp16': torch.float16,
    'float32': torch.float32,
    'fp32': torch.float32,
}

# Largest rows x length^2 that has fit in one forward pass on this run, learned from
# out-of-memory errors (attention cost grows with length^2). None = no limit seen yet.
_quadratic_cap: int | None = None
_warned = False


def _device() -> torch.device:
    if name := os.environ.get('SCIX_EMBED_DEVICE'):
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _dtype(device: torch.device) -> torch.dtype:
    if name := os.environ.get('SCIX_EMBED_DTYPE'):
        try:
            return _DTYPES[name.lower()]
        except KeyError:
            raise SystemExit(
                f'SCIX_EMBED_DTYPE={name!r}; use one of {", ".join(sorted(_DTYPES))}'
            ) from None
    if device.type == 'cuda':
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float32
    return torch.float32


def load(path: Path):
    """Load the Qwen3 backbone from a local HF snapshot. The embedding checkpoints
    store the backbone's keys without the "model." prefix, which is exactly what
    AutoModel (Qwen3Model, no lm_head) expects."""
    device = _device()
    net = AutoModel.from_pretrained(
        str(path), dtype=_dtype(device), attn_implementation='sdpa'
    )
    net.to(device).eval()
    return net


def describe(net) -> str:
    p = next(net.parameters())
    dtype = str(p.dtype).removeprefix('torch.')
    if p.device.type == 'cuda':
        major, minor = torch.cuda.get_device_capability(p.device)
        free, total = torch.cuda.mem_get_info(p.device)
        gpu = (
            f'{torch.cuda.get_device_name(p.device)}, compute {major}.{minor}, '
            f'{free / 2**30:.1f}/{total / 2**30:.1f} GiB free'
        )
        return f'PyTorch {torch.__version__} on {p.device} ({gpu}; {dtype})'
    return f'PyTorch {torch.__version__} on {p.device} ({dtype})'


def _forward(net, rows: list[list[int]], last: list[int]) -> np.ndarray:
    """One forward pass. Rows are trimmed to the longest pooled position, so a
    sub-batch split off a longer batch carries no needless padding."""
    device = next(net.parameters()).device
    width = max(last) + 1
    ids = torch.tensor([r[:width] for r in rows], dtype=torch.long, device=device)
    hidden = net(input_ids=ids, use_cache=False).last_hidden_state
    idx = torch.tensor(last, dtype=torch.long, device=device)
    v = hidden[torch.arange(len(rows), device=device), idx].float()
    v = torch.nn.functional.normalize(v, dim=-1)
    return v.cpu().numpy()


def _cost(last: list[int]) -> int:
    return len(last) * (max(last) + 1) ** 2


def _split(rows, last):
    """Split a batch so each part stays within the learned quadratic cap."""
    if _quadratic_cap is None or _cost(last) <= _quadratic_cap:
        return [(rows, last)]
    width = max(last) + 1
    step = max(1, _quadratic_cap // (width * width))
    return [(rows[s : s + step], last[s : s + step]) for s in range(0, len(rows), step)]


@torch.inference_mode()
def last_token_vectors(net, rows: list[list[int]], last: list[int]) -> np.ndarray:
    """Right-padded id rows -> L2-normalized float32 hidden state at `last[i]`.
    No attention mask, as on MLX: attention is causal and padding sits after
    the pooled token, so it cannot change it. Splits the batch on GPU OOM."""
    global _quadratic_cap, _warned
    out: list[np.ndarray] = []
    todo = _split(rows, last)
    while todo:
        r, l = todo.pop(0)
        try:
            out.append(_forward(net, r, l))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(r) == 1:
                raise SystemExit(
                    f'CUDA ran out of memory embedding a single {l[0] + 1}-token chunk '
                    f'(dtype {next(net.parameters()).dtype}). Free GPU memory, lower '
                    'MAX_EMBED_TOKENS in config.py, or set SCIX_EMBED_DEVICE=cpu.'
                ) from None
            half = len(r) // 2
            parts = [(r[:half], l[:half]), (r[half:], l[half:])]
            _quadratic_cap = max(_cost(p[1]) for p in parts)
            if not _warned:
                print(
                    f'\nGPU out of memory on a {len(r)}x{max(l) + 1}-token batch; '
                    'splitting batches from here on (lower EMBED_BATCH_TOKENS in '
                    'config.py to avoid the retries)',
                    file=sys.stderr,
                )
                _warned = True
            todo = parts + todo
    return np.concatenate(out)


def clear_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
