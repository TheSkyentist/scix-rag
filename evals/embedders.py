"""
Embedding models compared in Phase 2, behind one interface. All candidates are Qwen3
backbones with last-token pooling + L2 normalisation, so they share src.embed's loading
trick (mlx_lm's qwen3 Model, "model." key prefix added where the checkpoint lacks it) and
batching; they differ in repo, query/document formatting and whether an end-of-text token
is appended before pooling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from mlx_lm.models.qwen3 import Model, ModelArgs
from transformers import AutoTokenizer

import config
from src.embed import _batches


@dataclass(frozen=True)
class Spec:
    repo: str
    query: Callable[[str], str]
    doc: Callable[[str], str]
    eot: str | None  # token appended before last-token pooling (None: pool the last text token)


_INSTR = config.QUERY_INSTRUCTION
SPECS: dict[str, Spec] = {
    "qwen0.6": Spec("Qwen/Qwen3-Embedding-0.6B", lambda q: f"Instruct: {_INSTR}\nQuery:{q}", lambda t: t, "<|endoftext|>"),
    "qwen4": Spec("Qwen/Qwen3-Embedding-4B", lambda q: f"Instruct: {_INSTR}\nQuery:{q}", lambda t: t, "<|endoftext|>"),
    "harrier0.6": Spec("microsoft/harrier-oss-v1-0.6b", lambda q: f"Instruct: {_INSTR}\nQuery: {q}", lambda t: t, "<|endoftext|>"),
    "jina5s": Spec("jinaai/jina-embeddings-v5-text-small-retrieval-mlx", lambda q: f"Query: {q}", lambda t: f"Document: {t}", None),
}


class Embedder:
    def __init__(self, name: str, max_tokens: int = config.MAX_EMBED_TOKENS):
        self.name, self.spec, self.max_tokens = name, SPECS[name], max_tokens
        path = Path(snapshot_download(self.spec.repo, allow_patterns=["*.json", "*.safetensors", "*.txt"]))
        self.model = Model(ModelArgs.from_dict(json.loads((path / "config.json").read_text())))
        weights: dict[str, mx.array] = {}
        for f in sorted(path.glob("*.safetensors")):
            weights.update(mx.load(str(f)))
        self.model.load_weights([
            (k if k.startswith(("model.", "lm_head")) else f"model.{k}", v)
            for k, v in weights.items() if not k.startswith("lm_head")
        ])
        mx.eval(self.model.parameters())
        self.tok = AutoTokenizer.from_pretrained(str(path))
        self._eot = self.tok.convert_tokens_to_ids(self.spec.eot) if self.spec.eot else None

    def _encode(self, texts: list[str]) -> tuple[np.ndarray, int]:
        cap = self.max_tokens - (1 if self._eot is not None else 0)
        ids = [self.tok.encode(t, add_special_tokens=False)[:cap] for t in texts]
        if self._eot is not None:
            ids = [r + [self._eot] for r in ids]
        pad = self._eot if self._eot is not None else (self.tok.pad_token_id or 0)
        out: list[np.ndarray | None] = [None] * len(ids)
        for batch in _batches(sorted(range(len(ids)), key=lambda i: len(ids[i])), ids):
            longest = max(len(ids[i]) for i in batch)
            hidden = self.model.model(mx.array([ids[i] + [pad] * (longest - len(ids[i])) for i in batch]))
            v = hidden[mx.arange(len(batch)), mx.array([len(ids[i]) - 1 for i in batch])].astype(mx.float32)
            v = v / mx.linalg.norm(v, axis=-1, keepdims=True)
            mx.eval(v)
            v = np.array(v)
            for j, i in enumerate(batch):
                out[i] = v[j]
        return np.stack(out), sum(map(len, ids))

    def queries(self, qs: list[str]) -> np.ndarray:
        return self._encode([self.spec.query(q) for q in qs])[0]

    def docs(self, texts: list[str]) -> tuple[np.ndarray, int]:
        return self._encode([self.spec.doc(t) for t in texts])
