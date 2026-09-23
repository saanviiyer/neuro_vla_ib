import torch

from neuro_vla_ib.environment import generate_dataset
from neuro_vla_ib.models import BottleneckWorldModel, objective


def test_dataset_and_models_are_shape_compatible():
    data = generate_dataset(8, 6, seed=3)
    assert data.observations.shape == (8, 7, 10)
    for structured in (False, True):
        model = BottleneckWorldModel(hidden_dim=24, latent_dim=12, structured=structured)
        output = model(data.observations, data.actions, data.instruction_ids)
        loss, metrics = objective(output, data.observations[:, 1:], data.rewards, 1e-3, 1e-2)
        assert output.next_observation.shape == (8, 6, 10)
        assert output.reward.shape == (8, 6, 1)
        assert torch.isfinite(loss)
        assert metrics["rate_nats"] >= 0


def test_frozen_language_embeddings_are_supported():
    data = generate_dataset(4, 3, seed=5, paraphrases=(2,))
    embeddings = torch.randn(12, 7)
    model = BottleneckWorldModel(hidden_dim=20, latent_dim=10, pretrained_instruction_embeddings=embeddings)
    output = model(data.observations, data.actions, data.instruction_ids)
    assert output.latent.shape == (4, 3, 10)
    assert not model.goal_embedding.weight.requires_grad
