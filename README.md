# Step 1: Download data (now saves as CSV, no pyarrow needed)
python3 download_2000.py --max_stocks 2000 --workers 12

# Step 2: Train (loads CSV automatically)
python3 train_2000.py --symbols 2000 --pretrain 20000 --finetune 500000 --optuna --trials 50


---

You can pass `device='cpu'` or `export CUDA_VISIBLE_DEVICES=` to force using the CPU.Note: The model will train, but the GPU utilization will be poor and the training might take longer than on CPU.

---
Option B — Force CPU (sometimes faster for MLP PPO):

python3 train_2000.py --symbols 2000 --pretrain 20000 --finetune 500000 --device cpu

---
The pretraining code has a CPU/GPU tensor mismatch. Quick fix — force CPU for everything since MLP PPO doesn't benefit much from GPU:

python3 train_2000.py --symbols 2000 --pretrain 20000 --finetune 500000 --device cpu