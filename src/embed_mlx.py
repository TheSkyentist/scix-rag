"""
MLX backend for src/embed.py (Apple Silicon). The original embedding path, moved
here unchanged so src/embed.py can pick a backend per platform; see
src/embed_torch.py for the PyTorch counterpart used on Linux.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.qwen3 import Model, ModelArgs

BACKEND = 'mlx'


def load(path: Path) -> Model:
    net = Model(ModelArgs.from_dict(json.loads((path / 'config.json').read_text())))
    weights: dict[str, mx.array] = {}
    for f in sorted(path.glob('*.safetensors')):
        weights.update(mx.load(str(f)))
    # The HF embedding checkpoints omit the "model." key prefix mlx_lm's
    # causal-LM wrapper expects (it is a plain Qwen3 backbone, no lm_head).
    net.load_weights(
        [
            (k if k.startswith(('model.', 'lm_head')) else f'model.{k}', v)
            for k, v in weights.items()
        ]
    )
    mx.eval(net.parameters())
    return net


def describe(net: Model) -> str:
    return f'MLX {mx.__version__} on {mx.default_device()}'


def last_token_vectors(
    net: Model, rows: list[list[int]], last: list[int]
) -> np.ndarray:
    arr = mx.array(rows)
    hidden = net.model(arr)  # causal, so right-padding can't affect earlier positions
    v = hidden[mx.arange(len(rows)), mx.array(last)].astype(mx.float32)
    v = v / mx.linalg.norm(v, axis=-1, keepdims=True)
    mx.eval(v)
    return np.array(v)


def clear_cache() -> None:
    mx.clear_cache()
