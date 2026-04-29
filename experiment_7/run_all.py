"""
Experiment 7 - Single Entry Point for Training
Inside Bar Trend-Following Strategy

Usage:
  python3 experiment_7/run_all.py --mode test
  python3 experiment_7/run_all.py --mode full --optuna --trials 50 --finetune 500000
  python3 experiment_7/run_all.py --mode eval --model experiment_7/models/exp7_final.zip
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_7.train import train_pipeline


def download_data_if_needed():
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

        if not os.path.exists("experiment_3/data/SP500_daily.csv"):
            from experiment_3.download_data import download_sp500
            os.makedirs("experiment_3/data", exist_ok=True)
            download_sp500("experiment_3/data/SP500_daily.csv")

        if not os.path.exists("experiment_3/data/stocks_daily.parquet"):
            from experiment_3.download_data import download_stocks
            download_stocks("experiment_3/data/stocks_daily.parquet")

        print("[Data] All data downloaded.")
    else:
        print("[Data] All files present.")


def main():
    parser = argparse.ArgumentParser(description="Experiment 7 - Inside Bar Trend-Following RL Agent")
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
    print("EXPERIMENT 7 - Inside Bar Trend-Following RL Agent")
    print("=" * 70)
    print(f"  Mode:     {args.mode}")
    print(f"  Device:   {args.device}")
    print(f"  Pretrain: {args.pretrain:,} steps")
    print(f"  Finetune: {args.finetune:,} steps")
    print(f"  Optuna:   {'ON' if args.optuna else 'OFF'} ({args.trials} trials)")
    if torch.cuda.is_available():
        print(f"  GPU:      {torch.cuda.get_device_name(0)}")
    else:
        print("  GPU:      N/A (CPU only)")
    print("=" * 70)

    download_data_if_needed()

    if args.mode == "eval" and args.model:
        print(f"\n[Eval] Loading model from {args.model}...")
        from stable_baselines3 import PPO
        model = PPO.load(args.model)
        print("Model loaded. Use experiment_7.train.quick_eval() for metrics.")
    else:
        train_pipeline(
            run_optuna=args.optuna,
            n_optuna_trials=args.trials,
            pretrain_steps=args.pretrain,
            finetune_steps=args.finetune,
            device=args.device,
        )

    print("\n" + "=" * 70)
    print("DONE - Check experiment_7/results/ for outputs")
    print("=" * 70)


if __name__ == "__main__":
    main()
