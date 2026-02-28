# Geometric Structure of Flow Matching

Investigating the geometric and topological structure of class-conditioned flow matching models through Hodge decomposition, persistent homology, and mechanistic interpretability of AdaLN conditioning.

## Getting Started

```bash
# install dependencies
brew install uv
uv sync
```

### Train

```bash
# train default model (use --offline to skip wandb)
uv run python -m gf.main --offline

# generate sample grid
uv run python -m gf.generate --experiment default
```
