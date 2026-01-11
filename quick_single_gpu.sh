#!/bin/bash

# This script is modified for single GPU, minimal data, and very short training.
# Aiming for ~30 minutes total runtime on a single GPU.

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat_quick_test" # Use a different cache dir
mkdir -p $NANOCHAT_BASE_DIR

# --- Python venv setup with uv ---
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu # Keep GPU dependencies for single GPU usage
source .venv/bin/activate

# --- Wandb setup (optional, can be skipped for quick tests) ---
# For a quick test, you might want to skip wandb logging entirely or use a dummy run.
# If WANDB_RUN is not set, it defaults to 'dummy' which skips actual logging.
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# --- Report system initialization ---
python -m nanochat.report reset

# --- Tokenizer ---
# Reduce tokenizer training data significantly
# Original: python -m nanochat.dataset -n 8
#           python -m scripts.tok_train --max_chars=2000000000
python -m nanochat.dataset -n 8 # Download only 1 shard for tokenizer training
# Note: For even faster results, you could potentially skip tok_train and use a pre-existing tokenizer if available.
# Immediately also kick off downloading more shards in the background while tokenizer trains
# See comment below for why 240 is the right number here
python -m nanochat.dataset -n 10 &
DATASET_DOWNLOAD_PID=$!

python -m scripts.tok_train --max_chars=50000000 # Train on 50M chars (original was 2B)
python -m scripts.tok_eval

# --- Base model (pretraining) ---
# Skip background data download completely or make it minimal for quick test
# Original: python -m nanochat.dataset -n 240 & DATASET_DOWNLOAD_PID=$!
#           wait $DATASET_DOWNLOAD_PID
# For a quick test, rely on the single shard already downloaded or a very small number.
# If you didn't download any more data after tok_train, base_train might reuse the existing small dataset,
# which is fine for a quick, non-accurate run.

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Number of processes/GPUs to use
NPROC_PER_NODE=1

# pretrain the d20 model
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
     -m scripts.base_train -- \
    --depth=4 \
    --max_seq_len=512 \
    --device_batch_size=1 \
    --total_batch_size=512 \
    --eval_every=-1 \
    --core_metric_every=-1 \
    --sample_every=-1 \
    --num_iterations=20 \
    --run=$WANDB_RUN

# evaluate the model on a larger chunk of train/val data and draw some samples
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss -- \
    --split_tokens=512 # Evaluate on very few tokens
# evaluate the model on CORE tasks
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval -- \
    --max-per-task=16 # Evaluate on very few tasks/examples


# --- Midtraining ---
# Download synthetic data is small, can keep.
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run midtraining and eval the model
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- \
    --max_seq_len=512 \
    --device_batch_size=1 \
    --total_batch_size=512 \
    --num_iterations=10 \
    --eval_every=-1 \
    --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- \
    --source=mid --max-new-tokens=32 --max-problems=5 # Very minimal chat eval


# --- Supervised Finetuning (SFT) ---
# Reduce SFT iterations and evaluation scope
# Original: torchrun ... scripts.chat_sft ...
#           torchrun ... scripts.chat_eval ...
 
# train sft and re-eval right away (should see a small bump)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- \
    --device_batch_size=1 \
    --target_examples_per_step=16 \
    --num_iterations=10 \
    --eval_every=-1 \
    --eval_metrics_every=-1 \
    --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- \
    --source=sft --max-new-tokens=32 --max-problems=5 # Very minimal chat eval


# --- Reinforcement Learning (Optional, skipping for quick test) ---
# Keep commented out.

# --- Generate the full report ---
python -m nanochat.report generate

# --- Talk to it (optional, for interaction) ---
# python -m scripts.chat_web
# For a quick test, you might only run chat_cli if you don't need a web UI.
# python -m scripts.chat_cli -p "hello"