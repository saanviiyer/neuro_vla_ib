"""Run multi-seed beta sweeps and collect rate–distortion records."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import yaml

from neuro_vla_ib.train import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/rate_sweep"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 11, 19, 23])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0001, 0.0005, 0.002, 0.01])
    args = parser.parse_args()
    base = yaml.safe_load(args.config.read_text())
    records = []

    for seed in args.seeds:
        for beta in args.betas:
            cfg = dict(base, seed=seed, beta_rate=beta)
            with tempfile.NamedTemporaryFile("w", suffix=".yaml") as handle:
                yaml.safe_dump(cfg, handle); handle.flush()
                run_dir = args.output / f"seed-{seed}" / f"beta-{beta:g}"
                result = run(Path(handle.name), run_dir)
            for model, values in result.items():
                for split in ("clean", "corrupt"):
                    records.append({"seed": seed, "beta": beta, "model": model, "split": split, **values[split]})
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "rate_distortion_records.json").write_text(json.dumps(records, indent=2))

    print(args.output / "rate_distortion_records.json")


if __name__ == "__main__":
    main()

