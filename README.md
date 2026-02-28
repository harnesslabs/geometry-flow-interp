# Geometric Structure of Flow Matching

## Getting Started

```bash
# install dependencies
brew install uv
uv sync
```

### Train

```bash
# train default model (use --offline to skip wandb)
uv run python -m gf.train --offline

# generate sample grid
uv run python -m gf.generate --experiment default
```
