from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from neuro_vla_ib.environment import generate_dataset
from neuro_vla_ib.models import BottleneckWorldModel
from neuro_vla_ib.train import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/pilot"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    test_data = generate_dataset(
        cfg["test_episodes"], cfg["sequence_length"], cfg["seed"] + 1, cfg["observation_noise"],
        tuple(cfg.get("test_paraphrases", (0, 1, 2))),
    )
    embedding_path = cfg.get("language_embeddings")
    pretrained = torch.from_numpy(np.load(embedding_path)) if embedding_path else None
    results = {}
    for name, structured in (("gaussian_ib", False), ("modular_ring_ib", True)):
        model = BottleneckWorldModel(
            hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"], goal_embed_dim=cfg["goal_embed_dim"],
            structured=structured, pretrained_instruction_embeddings=pretrained,
        ).to(cfg["device"])
        model.load_state_dict(torch.load(args.checkpoint_dir / f"{name}.pt", map_location=cfg["device"]))
        results[name] = {
            "clean": evaluate(model, test_data, cfg, corruption=0.0),
            "corrupt": evaluate(model, test_data, cfg, corruption=0.12),
        }
    path = args.checkpoint_dir / "evaluation_metrics.json"
    path.write_text(json.dumps(results, indent=2))
    print(path)


if __name__ == "__main__":
    main()

