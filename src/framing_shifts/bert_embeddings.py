"""Extract final-layer BERT term vectors and optional Procrustes alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator < 1e-12:
        return float("nan")
    return float(np.dot(vector_a, vector_b) / denominator)


@dataclass
class BertEmbeddingModel:
    year: int
    model_path: Path
    tokenizer: object
    model: object
    device: object
    key_to_index: dict[str, int]
    vectors: np.ndarray
    rotation: np.ndarray | None = None
    _cache: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_directory(cls, year: int, model_path: Path, device: str | None = None):
        import torch
        from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

        torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        # The saved checkpoints contain explicit copies of both nominally tied
        # tensors. Loading them untied avoids a Transformers 5 warning and leaves
        # the encoder used by this analysis unchanged.
        model_config.tie_word_embeddings = False
        masked_language_model = AutoModelForMaskedLM.from_pretrained(
            model_path,
            config=model_config,
            local_files_only=True,
        ).to(torch_device)
        # The analysis uses encoder states, while loading the matching masked-LM
        # architecture also verifies that every bundled checkpoint tensor loads.
        model = masked_language_model.base_model
        model.eval()
        vocabulary = tokenizer.get_vocab()
        ordered_tokens = sorted(vocabulary, key=vocabulary.get)
        input_vectors = (
            model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float64, copy=True)
        )
        return cls(
            year=year,
            model_path=model_path,
            tokenizer=tokenizer,
            model=model,
            device=torch_device,
            key_to_index={token: vocabulary[token] for token in ordered_tokens},
            vectors=input_vectors,
        )

    def content_token_ids(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=True)
        special_ids = set(self.tokenizer.all_special_ids)
        return [token_id for token_id in encoded["input_ids"] if token_id not in special_ids]

    def __contains__(self, text: str) -> bool:
        token_ids = self.content_token_ids(text)
        return bool(token_ids) and self.tokenizer.unk_token_id not in token_ids

    def vector(self, text: str) -> np.ndarray:
        normalized = text.casefold()
        if normalized not in self._cache:
            self._cache[normalized] = self._encode(normalized)
        vector = self._cache[normalized]
        return vector @ self.rotation if self.rotation is not None else vector

    def similarity(self, text_a: str, text_b: str) -> float:
        return cosine_similarity(self.vector(text_a), self.vector(text_b))

    def vocab_vectors(self, tokens: Iterable[str]) -> np.ndarray:
        vectors = self.vectors[[self.key_to_index[token] for token in tokens]].astype(np.float64)
        return vectors @ self.rotation if self.rotation is not None else vectors

    def _encode(self, text: str) -> np.ndarray:
        import torch

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state[0]
        attention = inputs["attention_mask"][0].bool()
        special_ids = set(self.tokenizer.all_special_ids)
        ids = inputs["input_ids"][0].detach().cpu().tolist()
        mask = torch.tensor(
            [bool(used) and token_id not in special_ids for used, token_id in zip(attention, ids)],
            device=self.device,
            dtype=torch.bool,
        )
        if not mask.any():
            mask = attention
        return hidden[mask].mean(dim=0).detach().cpu().numpy().astype(np.float64)


def align_to_anchor(
    model: BertEmbeddingModel,
    anchor: BertEmbeddingModel,
    max_shared_vocabulary: int = 50_000,
) -> None:
    special = set(model.tokenizer.all_special_tokens) | set(anchor.tokenizer.all_special_tokens)
    shared = [
        token
        for token in model.key_to_index
        if token in anchor.key_to_index and token not in special and not token.startswith("[")
    ][:max_shared_vocabulary]
    if len(shared) < 10:
        raise ValueError(f"Only {len(shared)} shared vocabulary items are available")
    cross_covariance = model.vocab_vectors(shared).T @ anchor.vocab_vectors(shared)
    left, _, right_transpose = np.linalg.svd(cross_covariance)
    model.rotation = left @ right_transpose
