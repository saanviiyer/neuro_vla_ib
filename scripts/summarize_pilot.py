from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path, nargs="?", default=Path("results/pilot/pilot_metrics.json"))
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    keys = ("observation_mse", "reward_mse", "rate_nats", "state_r2", "heading_ring_r2", "goal_accuracy")
    print("model,split," + ",".join(keys))
    for model, record in data.items():
        for split in ("clean", "corrupt"):
            values = [str(record[split][key]) for key in keys]
            print(f"{model},{split}," + ",".join(values))


if __name__ == "__main__":
    main()

