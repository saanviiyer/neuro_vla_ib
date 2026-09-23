from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from .analysis import representation_metrics
from .environment import TrajectoryBatch, generate_dataset
from .models import BottleneckWorldModel, objective


def _loader(data: TrajectoryBatch, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(data.observations, data.actions, data.rewards, data.states, data.instruction_ids, data.goal_ids)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _as_batch(items: tuple[torch.Tensor, ...], device: str) -> TrajectoryBatch:
    return TrajectoryBatch(*items).to(device)


@torch.no_grad()
def evaluate(model: BottleneckWorldModel, data: TrajectoryBatch, cfg: dict, corruption: float = 0.0) -> dict[str, float]:
    model.eval()
    latents, states, goals, aggregates = [], [], [], []
    for items in _loader(data, cfg["batch_size"], False):
        batch = _as_batch(items, cfg["device"])
        observations = batch.observations + corruption * torch.randn_like(batch.observations)
        output = model(observations, batch.actions, batch.instruction_ids, sample=False, latent_noise=corruption)
        _, metrics = objective(output, batch.observations[:, 1:], batch.rewards, cfg["beta_rate"], cfg["topology_weight"])
        aggregates.append(metrics)
        latents.append(output.latent.cpu().numpy())
        states.append(batch.states[:, :-1].cpu().numpy())
        goals.append(batch.goal_ids.cpu().numpy())
    result = {key: float(np.mean([row[key] for row in aggregates])) for key in aggregates[0]}
    result.update(representation_metrics(np.concatenate(latents), np.concatenate(states), np.concatenate(goals)))
    return result


def run(config_path: Path, output_dir: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    train_data = generate_dataset(
        cfg["train_episodes"], cfg["sequence_length"], cfg["seed"], cfg["observation_noise"],
        tuple(cfg.get("train_paraphrases", (0, 1, 2))),
    )
    test_data = generate_dataset(
        cfg["test_episodes"], cfg["sequence_length"], cfg["seed"] + 1, cfg["observation_noise"],
        tuple(cfg.get("test_paraphrases", (0, 1, 2))),
    )
    embedding_path = cfg.get("language_embeddings")
    pretrained_embeddings = torch.from_numpy(np.load(embedding_path)) if embedding_path else None
    results: dict[str, dict] = {}

    for name, structured in (("gaussian_ib", False), ("modular_ring_ib", True)):
        torch.manual_seed(cfg["seed"])
        model = BottleneckWorldModel(
            hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"],
            goal_embed_dim=cfg["goal_embed_dim"], structured=structured,
            pretrained_instruction_embeddings=pretrained_embeddings,
        ).to(cfg["device"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
        history = []
        for epoch in range(cfg["epochs"]):
            model.train(); epoch_metrics = []
            for items in _loader(train_data, cfg["batch_size"], True):
                batch = _as_batch(items, cfg["device"])
                output = model(batch.observations, batch.actions, batch.instruction_ids)
                loss, metrics = objective(output, batch.observations[:, 1:], batch.rewards, cfg["beta_rate"], cfg["topology_weight"])
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                epoch_metrics.append(metrics)
            history.append({key: float(np.mean([m[key] for m in epoch_metrics])) for key in epoch_metrics[0]})
            print(f"{name:16s} epoch {epoch + 1:02d}/{cfg['epochs']} loss={history[-1]['loss']:.4f}")

        clean = evaluate(model, test_data, cfg, corruption=0.0)
        corrupt = evaluate(model, test_data, cfg, corruption=0.12)
        results[name] = {"clean": clean, "corrupt": corrupt, "history": history}
        torch.save(model.state_dict(), output_dir / f"{name}.pt")

    (output_dir / "pilot_metrics.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/pilot"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run(args.config, args.output)


if __name__ == "__main__":
    main()
