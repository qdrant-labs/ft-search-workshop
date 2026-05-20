"""SPLADE encoder used by the workshop notebook and helper scripts.

The fine-tuned SPLADE checkpoint (``thierrydamiba/splade-ecommerce-esci``) is
loaded via ``transformers`` rather than FastEmbed because we want a single
``encode`` entry point shared between:

* ``scripts/setup_collections.py`` (one-shot, populates the ``products``
  collection's ``splade_finetuned`` named sparse vector)
* ``scripts/benchmark_latency.py`` (per-query encoding inside the latency
  loop)
* the lab notebook (CP2 sparse-vector inspection, CP3 hybrid fusion)

Public API::

    from eval import SpladeEncoder
    enc = SpladeEncoder("thierrydamiba/splade-ecommerce-esci", device="cpu")
    pairs = enc.encode(["iphone 256gb"])  # -> List[Tuple[List[int], List[float]]]
    indices, values = pairs[0]
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


class SpladeEncoder:
    """Minimal SPLADE-max encoder: log(1 + relu(logits)) over masked-LM head."""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> List[Tuple[List[int], List[float]]]:
        enc = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**enc).logits  # (B, T, V)
        relu = torch.relu(out)
        weighted = torch.log1p(relu)
        mask = enc["attention_mask"].unsqueeze(-1)
        weighted = weighted * mask
        vec, _ = weighted.max(dim=1)  # SPLADE-max over tokens -> (B, V)
        results: List[Tuple[List[int], List[float]]] = []
        for row in vec:
            nz = torch.nonzero(row, as_tuple=False).squeeze(-1)
            results.append((nz.cpu().tolist(), row[nz].cpu().tolist()))
        return results
