from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def _split(x: np.ndarray, y: np.ndarray):
    return train_test_split(x, y, test_size=0.3, random_state=19)


def representation_metrics(latent: np.ndarray, states: np.ndarray, goal_ids: np.ndarray) -> dict[str, float]:
    """Probe distinct physical and semantic information in the latent channel."""
    x = latent.reshape(-1, latent.shape[-1])
    state = states.reshape(-1, states.shape[-1])
    goals = np.repeat(goal_ids, latent.shape[1])
    x_train, x_test, state_train, state_test = _split(x, state)
    state_prediction = LinearRegression().fit(x_train, state_train).predict(x_test)

    heading_train = np.stack([np.cos(state_train[:, 2]), np.sin(state_train[:, 2])], axis=1)
    heading_test = np.stack([np.cos(state_test[:, 2]), np.sin(state_test[:, 2])], axis=1)
    heading_prediction = LinearRegression().fit(x_train, heading_train).predict(x_test)

    gx_train, gx_test, gy_train, gy_test = _split(x, goals)
    classifier = LogisticRegression(max_iter=500).fit(gx_train, gy_train)
    probabilities = classifier.predict_proba(gx_test)
    goal_entropy_bits = _discrete_entropy_bits(gy_test)
    goal_information_bits = max(0.0, goal_entropy_bits - log_loss(gy_test, probabilities) / np.log(2))
    state_information_bits = _gaussian_information_proxy_bits(state_test, state_prediction)
    return {
        "state_r2": float(r2_score(state_test, state_prediction, multioutput="variance_weighted")),
        "heading_ring_r2": float(r2_score(heading_test, heading_prediction, multioutput="variance_weighted")),
        "goal_accuracy": float(accuracy_score(gy_test, classifier.predict(gx_test))),
        "goal_information_lower_bound_bits": float(goal_information_bits),
        "state_information_gaussian_proxy_bits": float(state_information_bits),
        "latent_participation_ratio": float(_participation_ratio(x)),
    }


def _participation_ratio(x: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(np.cov(x, rowvar=False)).clip(min=0)
    return float(eigenvalues.sum() ** 2 / (np.square(eigenvalues).sum() + 1e-12))


def _discrete_entropy_bits(labels: np.ndarray) -> float:
    probabilities = np.bincount(labels) / len(labels)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def _gaussian_information_proxy_bits(target: np.ndarray, prediction: np.ndarray) -> float:
    """Gaussian-decoder information proxy, reported explicitly as model-dependent."""
    variance = np.var(target, axis=0).clip(min=1e-8)
    mse = np.mean(np.square(target - prediction), axis=0).clip(min=1e-8)
    return float(np.maximum(0.0, 0.5 * np.log2(variance / mse)).sum())
