"""
Experiment 6: Enterprise-Grade RL Trading Agent
Supervised Pretraining Module — Rule-Based Expert Policy

Pretrains the PPO policy network using behavioral cloning from
expert rules: EMA Crossover + BB Squeeze + Breakout signals.

The expert generates (state, action) pairs for ~10k steps,
then the policy is cloned via supervised learning before PPO fine-tuning.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.ppo import PPO


class ExpertPolicy:
    """
    Rule-based expert that generates trading signals from:
      - EMA 20/50 crossover (trend)
      - BB Squeeze (volatility contraction)
      - Breakout confirmation (price action)
    
    Expert Logic:
      BUY:  EMA_20 > EMA_50 AND BB_Squeeze ≥ 1 AND Breakout_Up AND RSI < 70
      SELL: EMA_20 < EMA_50 AND BB_Squeeze ≥ 1 AND Breakout_Down AND RSI > 30
      HOLD: otherwise
    """

    def __init__(self, ema_spread_threshold: float = 0.0, squeeze_min: int = 1):
        self.ema_spread_threshold = ema_spread_threshold
        self.squeeze_min = squeeze_min

    def predict(self, obs: np.ndarray, features: list, df_row: dict) -> int:
        """
        Generate expert action from observation and row data.
        
        Args:
            obs: Full observation array (window_size × num_features)
            features: List of feature column names (for index lookup)
            df_row: Current row data dict with indicator values
        
        Returns:
            action: 0=HOLD, 1=BUY, 2=SELL
        """
        ema_spread = df_row.get("ema_20_50_spread", 0)
        squeeze = df_row.get("squeeze_signal", 0)
        bo_up = df_row.get("breakout_up", 0)
        bo_dn = df_row.get("breakout_down", 0)
        rsi = df_row.get("rsi_14", 50)
        adx = df_row.get("adx_14", 0)

        has_squeeze = squeeze >= self.squeeze_min
        trending = adx > 0.15

        # BUY conditions
        buy_signal = (
            ema_spread > self.ema_spread_threshold
            and has_squeeze
            and bo_up == 1
            and rsi < 70
            and trending
        )

        # SELL conditions
        sell_signal = (
            ema_spread < -self.ema_spread_threshold
            and has_squeeze
            and bo_dn == 1
            and rsi > 30
            and trending
        )

        if buy_signal:
            return 1
        elif sell_signal:
            return 2
        return 0


def collect_expert_demonstrations(env, expert: ExpertPolicy, n_steps: int = 10000,
                                  feature_cols: list = None) -> tuple:
    """
    Run the expert policy through the environment to collect (state, action) pairs.
    
    Returns:
        observations: (N, window_size, num_features) array
        actions: (N,) array of expert actions
    """
    observations = []
    actions = []

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # Gymnasium returns (obs, info)
    episode_steps = 0
    total_collected = 0

    while total_collected < n_steps:
        # Get current step data from env
        env_instance = env.envs[0] if hasattr(env, 'envs') else env
        if hasattr(env_instance, '_get_step_data'):
            step_data = env_instance._get_step_data
        else:
            step_data = {}

        # Expert decides action
        action = expert.predict(obs, feature_cols or [], step_data)

        observations.append(obs)
        actions.append(action)

        # Step environment
        step_out = env.step(np.array([action]))
        if len(step_out) == 4:
            obs, _, dones, _ = step_out
        else:
            obs, _, terminated, truncated, _ = step_out
            dones = terminated | truncated

        total_collected += 1
        episode_steps += 1

        if dones[0]:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            episode_steps = 0

    return (np.array(observations, dtype=np.float32),
            np.array(actions, dtype=np.int64))


def pretrain_policy(model: PPO, observations: np.ndarray, actions: np.ndarray,
                    epochs: int = 10, batch_size: int = 64, lr: float = 1e-3,
                    device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Pretrain the PPO policy network via behavioral cloning (supervised learning).
    
    Clones the expert actions using cross-entropy loss on the policy head.
    After pretraining, the value function is roughly initialized.
    
    Args:
        model: PPO model to pretrain
        observations: Expert observations (N, window, features)
        actions: Expert actions (N,)
        epochs: Number of pretraining epochs
        batch_size: Batch size for training
        lr: Learning rate for pretraining
        device: 'cuda' or 'cpu'
    
    Returns:
        history: Dict with loss history
    """
    print(f"[Pretrain] Cloning expert on {len(observations)} samples, "
          f"{epochs} epochs, batch_size={batch_size}, device={device}")

    model.policy.to(device)

    # Flatten observations for SB3 policy (it expects (N, window*features) or handles internally)
    n_samples = len(observations)
    obs_flat = observations.reshape(n_samples, -1)

    # Create dataset
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

            # Forward pass through SB3 policy
            # SB3 MLP policy: extract_features → mlp_extractor → action_net
            features = model.policy.extract_features(batch_obs)
            latent_pi, latent_vf = model.policy.mlp_extractor(features)
            logits = model.policy.action_net(latent_pi)

            loss = criterion(logits, batch_act)

            # Accuracy
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

    # Also pretrain the value function to roughly match
    print("[Pretrain] Initializing value function...")
    _pretrain_value(model, observations, device, epochs=3, lr=lr * 0.1)

    model.policy.to("cpu")
    return loss_history


def _pretrain_value(model: PPO, observations: np.ndarray, device: str,
                    epochs: int = 3, lr: float = 1e-4):
    """Rough value function initialization using recent returns as targets."""
    n = len(observations)
    obs_flat = observations.reshape(n, -1)

    # Simple target: 0 (neutral) since we're initializing
    targets = np.zeros(n, dtype=np.float32)

    dataset = TensorDataset(
        torch.tensor(obs_flat, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.policy.train()
    for epoch in range(epochs):
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
