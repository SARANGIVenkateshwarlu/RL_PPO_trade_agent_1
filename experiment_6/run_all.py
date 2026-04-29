"""
Experiment 6 - Single Entry Point for Cloud Training
Handles data download, training, and evaluation in one command.

Usage:
  # Quick test run
  python3 run_all.py --mode test

  # Full training with Optuna optimization on 40 GPUs
  python3 run_all.py --mode full --optuna --trials 50 --finetune 500000

  # Evaluate existing model
  python3 run_all.py --mode eval --model experiment_6/models/exp6_final.zip
"""
import os
import sys
import argparse
import torch

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_6.train import train_pipeline


def download_data_if_needed():
    """Ensure all required data files exist."""
    print("[Data] Checking data files...")
    needed = {
        "experiment_3/data/SP500_daily.csv": "S&P 500 index",
        "experiment_3/data/stocks_daily.parquet": "Stock data",
        "data/Tickers_list_USA.csv": "Ticker list",
    }
    missing = []
    for path, desc in needed.items():
        if not os.path.exists(path):
            missing.append(f"  {path} ({desc})")

    if missing:
        print("[Data] Missing files:")
        for m in missing:
            print(m)
        print("[Data] Downloading...")

        # Download S&P 500
        if not os.path.exists("experiment_3/data/SP500_daily.csv"):
            from experiment_3.download_data import download_sp500
            os.makedirs("experiment_3/data", exist_ok=True)
            download_sp500("experiment_3/data/SP500_daily.csv")

        # Download stocks
        if not os.path.exists("experiment_3/data/stocks_daily.parquet"):
            from experiment_3.download_data import download_stocks
            download_stocks("experiment_3/data/stocks_daily.parquet")

        print("[Data] All data downloaded.")
    else:
        print("[Data] All files present.")


def main():
    parser = argparse.ArgumentParser(description="Experiment 6 - Enterprise RL Trading Agent")
    parser.add_argument("--mode", type=str, default="test",
                        choices=["test", "full", "eval"],
                        help="test=quick run, full=optuna+finetune, eval=evaluate only")
    parser.add_argument("--optuna", action="store_true", help="Run Optuna optimization")
    parser.add_argument("--trials", type=int, default=20, help="Optuna trials")
    parser.add_argument("--pretrain", type=int, default=5000, help="Pretraining steps")
    parser.add_argument("--finetune", type=int, default=100000, help="Fine-tuning steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", type=str, default=None, help="Model path for eval mode")
    args = parser.parse_args()

    # Auto-configure based on mode
    if args.mode == "test":
        args.pretrain = args.pretrain or 2000
        args.finetune = args.finetune or 30000
        args.optuna = False
        args.trials = 5
    elif args.mode == "full":
        args.pretrain = args.pretrain or 10000
        args.finetune = args.finetune or 500000
        args.optuna = args.optuna or True
        args.trials = args.trials or 50

    print("=" * 70)
    print("EXPERIMENT 6 - Enterprise RL Trading Agent")
    print("=" * 70)
    print(f"  Mode:     {args.mode}")
    print(f"  Device:   {args.device}")
    print(f"  Pretrain: {args.pretrain:,} steps")
    print(f"  Finetune: {args.finetune:,} steps")
    print(f"  Optuna:   {'ON' if args.optuna else 'OFF'} ({args.trials} trials)")
    if torch.cuda.is_available():
        print(f"  GPU:      {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:     {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        print(f"  GPU Count:{torch.cuda.device_count()}")
    else:
        print(f"  GPU:      N/A (CPU only)")
    print("=" * 70)

    # Download data
    download_data_if_needed()

    # Train or evaluate
    if args.mode == "eval" and args.model:
        print(f"\n[Eval] Loading model from {args.model}...")
        from stable_baselines3 import PPO
        model = PPO.load(args.model)
        print("Model loaded. Use experiment_6.train.quick_eval() for metrics.")
    else:
        train_pipeline(
            run_optuna=args.optuna,
            n_optuna_trials=args.trials,
            pretrain_steps=args.pretrain,
            finetune_steps=args.finetune,
            device=args.device,
        )

    print("\n" + "=" * 70)
    print("DONE - Check experiment_6/results/ for outputs")
    print("=" * 70)


if __name__ == "__main__":
    main()
