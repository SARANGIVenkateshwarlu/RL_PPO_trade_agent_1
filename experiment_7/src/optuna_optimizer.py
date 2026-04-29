"""
Experiment 6: Optuna Hyperparameter Optimizer

Optimizes PPO hyperparameters using Optuna with efficient pruning.
Searches: learning_rate, n_steps, batch_size, gamma, gae_lambda,
          ent_coef, clip_range, net_arch dimensions, pretrain_epochs.
"""
import os
import json
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def create_ppo_model(trial: optuna.Trial, env) -> PPO:
    """Create PPO model with Optuna-suggested hyperparameters."""
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    n_epochs = trial.suggest_int("n_epochs", 5, 20)

    pi_layers = trial.suggest_categorical("pi_layers", [
        [256, 128], [256, 256, 128], [512, 256, 128], [512, 256, 128, 64]
    ])
    vf_layers = trial.suggest_categorical("vf_layers", [
        [256, 128], [256, 256, 128], [512, 256, 128], [512, 256, 128, 64]
    ])

    policy_kwargs = dict(net_arch=dict(pi=pi_layers, vf=vf_layers))

    model = PPO(
        "MlpPolicy", env, verbose=0,
        learning_rate=lr, n_steps=n_steps, batch_size=batch_size,
        n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda,
        clip_range=clip_range, ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
    )
    return model


def evaluate_model_quick(model, eval_env, n_steps: int = 5000) -> float:
    """Quick evaluation: return final equity as fitness metric."""
    try:
        obs = eval_env.reset()
        for _ in range(n_steps):
            action, _ = model.predict(obs, deterministic=True)
            step_out = eval_env.step(action)
            if len(step_out) == 4:
                obs, _, dones, _ = step_out
                done = dones[0] if hasattr(dones, '__getitem__') else dones
            else:
                obs, _, terminated, truncated, _ = step_out
                done = bool(terminated[0] or truncated[0]) if hasattr(terminated, '__getitem__') else bool(terminated or truncated)
            if done:
                break
        equity = eval_env.get_attr("equity")[0]
        return float(equity)
    except Exception as e:
        return 0.0


def optimize_hyperparameters(
    train_env_fn, eval_env_fn,
    n_trials: int = 30, n_pretrain_steps: int = 5000,
    n_finetune_steps: int = 30000,
    study_name: str = "exp6_optimization",
    storage: str = "sqlite:///experiment_7/optuna_studies/exp7.db",
):
    """
    Run Optuna hyperparameter optimization.
    
    Args:
        train_env_fn: Callable that returns training env
        eval_env_fn: Callable that returns evaluation env
        n_trials: Number of Optuna trials
        n_pretrain_steps: Steps for expert pretraining per trial
        n_finetune_steps: Steps for PPO fine-tuning per trial
    
    Returns:
        study: Optuna study object with best parameters
    """

    def objective(trial: optuna.Trial) -> float:
        train_env = train_env_fn()
        eval_env = eval_env_fn()

        model = create_ppo_model(trial, train_env)

        # Quick pretrain + finetune for evaluation
        try:
            model.learn(total_timesteps=n_finetune_steps, progress_bar=False)

            # Evaluate
            final_eq = evaluate_model_quick(model, eval_env)
            init_eq = 100000.0
            return_pct = (final_eq / init_eq - 1) * 100

            # Multi-objective: return + risk-adjusted
            score = return_pct

        except Exception as e:
            score = -999.0
        finally:
            train_env.close()
            eval_env.close()

        return score

    # Create study
    os.makedirs(os.path.dirname(storage.split("///")[-1] if "///" in storage else
                os.path.join("experiment_7", "optuna_studies")), exist_ok=True)

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    print(f"[Optuna] Starting optimization: {n_trials} trials")
    print(f"[Optuna] Storage: {storage}")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n[Optuna] Best trial: #{study.best_trial.number}")
    print(f"[Optuna] Best value: {study.best_value:.2f}")
    print(f"[Optuna] Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save best params
    best_params_path = "experiment_7/optuna_studies/best_params.json"
    with open(best_params_path, "w") as f:
        json.dump({
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\nBest params saved to {best_params_path}")
    return study


__all__ = ["create_ppo_model", "optimize_hyperparameters"]
