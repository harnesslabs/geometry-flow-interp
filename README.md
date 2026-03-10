# Interpreting the Geometric Structure of Flow Matching

Currently training a [JiT](https://arxiv.org/abs/2511.13720)-style ViT with x-prediction and v-loss on rotated class-conditional MNIST. 

## Setup

```bash
brew install uv
uv sync
```

## Usage

```bash
# train (use --offline to skip wandb)
uv run python -m scripts.train --offline

# generate samples from a checkpoint
uv run python -m scripts.generate --experiment <name>
```

## File Structure

```bash
geoflow/       # library (model, denoiser, checkpoint, optim, etc.)
scripts/       # entry points
  train.py
  generate.py
```
