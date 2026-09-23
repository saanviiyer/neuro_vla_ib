from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


INSTRUCTIONS = (
    "move to the red beacon", "navigate toward the crimson marker", "approach the red target",
    "move to the blue beacon", "navigate toward the azure marker", "approach the blue target",
    "move to the green beacon", "navigate toward the emerald marker", "approach the green target",
    "move to the yellow beacon", "navigate toward the golden marker", "approach the yellow target",
)

GOALS = np.asarray([[-0.72, -0.72], [0.72, -0.72], [-0.72, 0.72], [0.72, 0.72]], dtype=np.float32)
LANDMARKS = np.asarray([[-0.9, -0.9], [0.9, -0.9], [-0.9, 0.9], [0.9, 0.9]], dtype=np.float32)


@dataclass
class TrajectoryBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    states: torch.Tensor
    instruction_ids: torch.Tensor
    goal_ids: torch.Tensor

    def to(self, device: str) -> "TrajectoryBatch":
        return TrajectoryBatch(*(x.to(device) for x in self.__dict__.values()))


def _observe(state: np.ndarray, rng: np.random.Generator, noise: float) -> np.ndarray:
    pos, theta = state[:2], state[2]
    delta = LANDMARKS - pos[None, :]
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.asarray([[c, s], [-s, c]], dtype=np.float32)
    egocentric = (delta @ rotation.T).reshape(-1) / 2.0
    wall_clearance = np.asarray([1 - abs(pos[0]), 1 - abs(pos[1])], dtype=np.float32)
    obs = np.concatenate([egocentric, wall_clearance])
    return obs + rng.normal(0, noise, size=obs.shape).astype(np.float32)


def _angle_wrap(x: np.ndarray | float) -> np.ndarray | float:
    return (x + np.pi) % (2 * np.pi) - np.pi


def generate_dataset(
    episodes: int,
    sequence_length: int,
    seed: int,
    observation_noise: float = 0.015,
    paraphrases: tuple[int, ...] = (0, 1, 2),
) -> TrajectoryBatch:
    """Generate language-conditioned differential-drive trajectories.

    Each instruction has three paraphrases. The simulator exposes only noisy,
    egocentric landmark measurements and actions; global pose is retained only
    for representational analysis.
    """
    rng = np.random.default_rng(seed)
    obs = np.zeros((episodes, sequence_length + 1, 10), np.float32)
    actions = np.zeros((episodes, sequence_length, 2), np.float32)
    rewards = np.zeros((episodes, sequence_length, 1), np.float32)
    states = np.zeros((episodes, sequence_length + 1, 4), np.float32)
    goal_ids = rng.integers(0, 4, episodes)
    if not paraphrases or any(p not in (0, 1, 2) for p in paraphrases):
        raise ValueError("paraphrases must be a non-empty subset of (0, 1, 2)")
    paraphrase = rng.choice(np.asarray(paraphrases), episodes)
    instruction_ids = goal_ids * 3 + paraphrase

    for episode in range(episodes):
        state = np.asarray([
            rng.uniform(-0.75, 0.75), rng.uniform(-0.75, 0.75),
            rng.uniform(-np.pi, np.pi), 0.0,
        ], dtype=np.float32)
        goal = GOALS[goal_ids[episode]]
        states[episode, 0] = state
        obs[episode, 0] = _observe(state, rng, observation_noise)

        for t in range(sequence_length):
            vector = goal - state[:2]
            desired = np.arctan2(vector[1], vector[0])
            turn = float(np.clip(1.8 * _angle_wrap(desired - state[2]) + rng.normal(0, 0.18), -1, 1))
            speed = float(np.clip(0.65 * np.linalg.norm(vector) + rng.normal(0, 0.06), 0, 0.75))
            if rng.random() < 0.18:
                speed, turn = rng.uniform(0, 0.7), rng.uniform(-1, 1)
            actions[episode, t] = (speed, turn)

            theta = float(_angle_wrap(state[2] + 0.22 * turn))
            position = state[:2] + 0.11 * speed * np.asarray([np.cos(theta), np.sin(theta)])
            clipped = np.clip(position, -0.98, 0.98)
            collision = float(np.any(np.abs(position) >= 0.98))
            unsafe = float(np.linalg.norm(clipped) < 0.22)
            state = np.asarray([clipped[0], clipped[1], theta, speed], dtype=np.float32)
            reward = -np.linalg.norm(goal - clipped) - 1.5 * collision - 0.8 * unsafe

            states[episode, t + 1] = state
            obs[episode, t + 1] = _observe(state, rng, observation_noise)
            rewards[episode, t, 0] = reward

    return TrajectoryBatch(
        observations=torch.from_numpy(obs), actions=torch.from_numpy(actions),
        rewards=torch.from_numpy(rewards), states=torch.from_numpy(states),
        instruction_ids=torch.from_numpy(instruction_ids.astype(np.int64)),
        goal_ids=torch.from_numpy(goal_ids.astype(np.int64)),
    )
