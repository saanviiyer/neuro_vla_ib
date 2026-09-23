from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class WorldModelOutput:
    next_observation: torch.Tensor
    reward: torch.Tensor
    latent: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    rate_nats: torch.Tensor
    topology_loss: torch.Tensor


class BottleneckWorldModel(nn.Module):
    """Language-conditioned recurrent world model with a variational channel."""

    def __init__(
        self, observation_dim: int = 10, action_dim: int = 2,
        instruction_count: int = 12, goal_embed_dim: int = 16,
        hidden_dim: int = 96, latent_dim: int = 16, structured: bool = False,
        pretrained_instruction_embeddings: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if structured and latent_dim < 8:
            raise ValueError("structured models require latent_dim >= 8")
        self.structured = structured
        self.latent_dim = latent_dim
        self.phase_dim = 8 if structured else 0
        if pretrained_instruction_embeddings is None:
            self.goal_embedding = nn.Embedding(instruction_count, goal_embed_dim)
            language_dim = goal_embed_dim
        else:
            if pretrained_instruction_embeddings.ndim != 2 or pretrained_instruction_embeddings.shape[0] != instruction_count:
                raise ValueError(f"expected instruction embeddings with shape ({instruction_count}, d)")
            self.goal_embedding = nn.Embedding.from_pretrained(pretrained_instruction_embeddings.float(), freeze=True)
            language_dim = pretrained_instruction_embeddings.shape[1]
        self.recurrent = nn.GRU(observation_dim + action_dim + language_dim, hidden_dim, batch_first=True)
        self.posterior = nn.Linear(hidden_dim, 2 * latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim + action_dim + language_dim, hidden_dim), nn.SiLU())
        self.observation_head = nn.Linear(hidden_dim, observation_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor,
        instruction_ids: torch.Tensor, sample: bool = True,
        latent_noise: float = 0.0,
    ) -> WorldModelOutput:
        length = actions.shape[1]
        goal = self.goal_embedding(instruction_ids)[:, None, :].expand(-1, length, -1)
        recurrent_input = torch.cat([observations[:, :length], actions, goal], dim=-1)
        hidden, _ = self.recurrent(recurrent_input)
        mu, logvar = self.posterior(hidden).chunk(2, dim=-1)
        logvar = logvar.clamp(-8, 5)
        latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu

        topology_loss = latent.new_zeros(())
        if self.structured:
            phase_raw = latent[..., :self.phase_dim].reshape(*latent.shape[:-1], -1, 2)
            norms = phase_raw.norm(dim=-1, keepdim=True).clamp_min(1e-5)
            topology_loss = ((norms - 1.0) ** 2).mean()
            phase = (phase_raw / norms).flatten(-2)
            latent = torch.cat([phase, latent[..., self.phase_dim:]], dim=-1)
        if latent_noise:
            latent = latent + latent_noise * torch.randn_like(latent)

        decoded = self.decoder(torch.cat([latent, actions, goal], dim=-1))
        rate = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(dim=-1).mean()
        return WorldModelOutput(
            next_observation=self.observation_head(decoded), reward=self.reward_head(decoded),
            latent=latent, mu=mu, logvar=logvar, rate_nats=rate,
            topology_loss=topology_loss,
        )


def objective(
    output: WorldModelOutput, next_observation: torch.Tensor, reward: torch.Tensor,
    beta_rate: float, topology_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    observation_loss = F.mse_loss(output.next_observation, next_observation)
    reward_loss = F.mse_loss(output.reward, reward)
    total = observation_loss + reward_loss + beta_rate * output.rate_nats + topology_weight * output.topology_loss
    metrics = {
        "loss": float(total.detach()), "observation_mse": float(observation_loss.detach()),
        "reward_mse": float(reward_loss.detach()), "rate_nats": float(output.rate_nats.detach()),
        "topology_loss": float(output.topology_loss.detach()),
    }
    return total, metrics
