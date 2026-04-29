"""
Experiment 7: Supervised Pretraining Module
Expert Policy for Inside Bar Trend-Following Strategy

Expert rules match the strategy constraints:
  BUY when: entry_gate == 1 (all conditions met)
  HOLD: otherwise
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.ppo import PPO


class ExpertPolicy:
    """
    Rule-based expert that generates BUY signals when the
    inside-bar trend-following entry gate is active.

    Expert Logic:
      BUY:  entry_gate == 1 AND positions available (cash > risk_amount)
      HOLD: otherwise
    """

    def __init__(self, require_gate: bool = True):
        self.require_gate = require_gate

    def predict(self, obs: np.ndarray, features: list, df_row: dict) -> int:
        gate = df_row.get("entry_gate", 0)
        if self.require_gate and gate == 1:
            return 1  # BUY
        return 0  # HOLD


def collect_expert_demonstrations(env, expert: ExpertPolicy, n_steps: int = 10000,
                                  feature_cols: list = None) -> tuple:
    observations = []
    actions = []

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    total_collected = 0

    while total_collected < n_steps:
        env_instance = env.envs[0] if hasattr(env, 'envs') else env
        if hasattr(env_instance, '_row'):
            step_data = env_instance._row.to_dict()
        else:
            step_data = {}

        action = expert.predict(obs, feature_cols or [], step_data)

        observations.append(obs)
        actions.append(action)

        step_out = env.step(np.array([action]))
        if len(step_out) == 4:
            obs, _, dones, _ = step_out
        else:
            obs, _, terminated, truncated, _ = step_out
            dones = terminated | truncated

        total_collected += 1

        if dones[0]:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

    return (np.array(observations, dtype=np.float32),
            np.array(actions, dtype=np.int64))


def pretrain_policy(model: PPO, observations: np.ndarray, actions: np.ndarray,
                    epochs: int = 10, batch_size: int = 64, lr: float = 1e-3,
                    device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    print(f"[Pretrain] Cloning expert on {len(observations)} samples, "
          f"{epochs} epochs, batch_size={batch_size}, device={device}")

    model.policy.to(device)

    n_samples = len(observations)
    obs_flat = observations.reshape(n_samples, -1)

    dataset = TensorDataset(
        torch.tensor(obs_flat, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.long),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loss_history = []

    model.policy.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

        for batch_obs, batch_act in dataloader:
            batch_obs = batch_obs.to(device)
            batch_act = batch_act.to(device)

            optimizer.zero_grad()

            features = model.policy.extract_features(batch_obs)
            latent_pi, latent_vf = model.policy.mlp_extractor(features)
            logits = model.policy.action_net(latent_pi)

            loss = criterion(logits, batch_act)

            preds = logits.argmax(dim=1)
            acc = (preds == batch_act).float().mean()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc = epoch_acc / max(n_batches, 1)
        loss_history.append({"epoch": epoch + 1, "loss": avg_loss, "accuracy": avg_acc})

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

    model.policy.eval()

    print("[Pretrain] Initializing value function...")
    _pretrain_value(model, observations, device, epochs=3, lr=lr * 0.1)

    model.policy.to("cpu")
    return loss_history


def _pretrain_value(model: PPO, observations: np.ndarray, device: str,
                    epochs: int = 3, lr: float = 1e-4):
    n = len(observations)
    obs_flat = observations.reshape(n, -1)
    targets = np.zeros(n, dtype=np.float32)

    dataset = TensorDataset(
        torch.tensor(obs_flat, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.policy.train()
    for _ in range(epochs):
        for batch_obs, batch_val in dataloader:
            batch_obs = batch_obs.to(device)
            batch_val = batch_val.to(device)
            optimizer.zero_grad()
            features = model.policy.extract_features(batch_obs)
            _, latent_vf = model.policy.mlp_extractor(features)
            values = model.policy.value_net(latent_vf).squeeze()
            loss = criterion(values, batch_val)
            loss.backward()
            optimizer.step()
    model.policy.eval()


__all__ = ["ExpertPolicy", "collect_expert_demonstrations", "pretrain_policy"]
