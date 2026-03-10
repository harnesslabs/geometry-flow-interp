# Geometric Structure of Flow Matching

Investigating the internal geometry of flow matching models.

## Setup

```bash
brew install uv
uv sync
```

## Usage

```bash
# train (use --offline to skip wandb)
uv run python -m geoflow.train --offline

# generate samples from a checkpoint
uv run python -m geoflow.generate --experiment <name>
```
