"""Extract frozen instruction embeddings from a Hugging Face language model.

Run this once per chosen backbone, record the exact revision, and point the
experiment config's `language_embeddings` field to the resulting .npy file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from neuro_vla_ib.environment import INSTRUCTIONS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model name")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModel.from_pretrained(args.model, revision=args.revision).eval()
    tokens = tokenizer(list(INSTRUCTIONS), padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1)
        embeddings = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings.cpu().numpy().astype(np.float32))
    metadata = {"model": args.model, "revision": args.revision, "instructions": list(INSTRUCTIONS)}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

