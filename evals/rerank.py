"""
Qwen3-Reranker on MLX, for the retrieval eval (not used by production search yet).

The reranker is a Qwen3 causal LM asked whether a document meets the query; its score
is P("yes") vs P("no") at the last position. Both checkpoints tie the LM head to the
token embeddings, so only the two rows for "yes"/"no" are needed: the score is
sigmoid(h . (E[yes] - E[no])) on the backbone's final (normed) hidden state, which
avoids a full-vocabulary projection.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm.models.qwen3 import Model, ModelArgs
from transformers import AutoTokenizer

from src.embed import _batches

DEFAULT_INSTRUCTION = (
    "Given a question from astrophysics research, judge whether this passage from a scientific paper "
    "answers or substantially discusses it"
)
_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Reranker:
    def __init__(self, size: str = "0.6B", max_doc_tokens: int = 400):
        path = Path(snapshot_download(f"Qwen/Qwen3-Reranker-{size}", allow_patterns=["*.json", "*.safetensors", "*.txt"]))
        cfg = json.loads((path / "config.json").read_text())
        if not cfg.get("tie_word_embeddings", False):
            raise ValueError("expected tied embeddings (untied lm_head not handled)")
        self.model = Model(ModelArgs.from_dict(cfg))
        weights: dict[str, mx.array] = {}
        for f in sorted(path.glob("*.safetensors")):
            weights.update(mx.load(str(f)))
        self.model.load_weights([
            (k if k.startswith(("model.", "lm_head")) else f"model.{k}", v)
            for k, v in weights.items() if not k.startswith("lm_head")
        ])
        mx.eval(self.model.parameters())
        self.tok = AutoTokenizer.from_pretrained(str(path))
        yes, no = self.tok.convert_tokens_to_ids("yes"), self.tok.convert_tokens_to_ids("no")
        emb = self.model.model.embed_tokens.weight
        self.direction = (emb[yes] - emb[no]).astype(mx.float32)
        self.max_doc_tokens = max_doc_tokens
        self._pre = self.tok.encode(_PREFIX, add_special_tokens=False)
        self._suf = self.tok.encode(_SUFFIX, add_special_tokens=False)
        self.n_tokens = 0

    def score(self, query: str, docs: list[str], instruction: str = DEFAULT_INSTRUCTION) -> list[float]:
        head = self.tok.encode(f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: ", add_special_tokens=False)
        ids = [
            self._pre + head + self.tok.encode(d, add_special_tokens=False)[: self.max_doc_tokens] + self._suf
            for d in docs
        ]
        pad = self.tok.pad_token_id or 0
        out = [0.0] * len(ids)
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        for batch in _batches(order, ids):
            longest = max(len(ids[i]) for i in batch)
            arr = mx.array([ids[i] + [pad] * (longest - len(ids[i])) for i in batch])
            hidden = self.model.model(arr)  # causal: right padding can't affect earlier positions
            last = hidden[mx.arange(len(batch)), mx.array([len(ids[i]) - 1 for i in batch])].astype(mx.float32)
            s = mx.sigmoid(last @ self.direction)
            mx.eval(s)
            for j, i in enumerate(batch):
                out[i] = float(s[j])
        self.n_tokens += sum(map(len, ids))
        return out
