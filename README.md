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

### Analysis Pipeline

Run each phase in order — each depends on the previous:

```bash
# Phase 0: Collect trajectory atlas (positions, velocities, modulations)
#   Use --sampling-method heun to match the denoiser's default integrator
uv run python -m gf.analysis.atlas --experiment default --sampling-method heun

# Phase 1: Helmholtz-Hodge decomposition (neural 2-way + graph 3-way)
uv run python -m gf.analysis.hodge --experiment default

# Phase 2: AdaLN mechanistic analysis (modulation fingerprints + patching)
uv run python -m gf.analysis.adaln_analysis --experiment default

# Phase 3: Trajectory topology (branching, persistence, cross-class MMD)
uv run python -m gf.analysis.topology --experiment default

# Phase 4: Assemble paper-ready figures from all results
uv run python -m gf.analysis.figures --experiment default
```

Results are saved to `checkpoints/<experiment>/analysis/`.
Figures are saved to `checkpoints/<experiment>/analysis/figures/`.
