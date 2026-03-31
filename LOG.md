# Experiment Log

A running summary documenting some experiments and findings. Started Mar 26, 2026.

---

## 2026-03-31: Timestep sampling, inference, and training tweaks

### Denoiser (`geoflow/denoiser.py`)

- Timestep sampling reverted from `Uniform[0,1]` back to logit-normal: `sigmoid(randn * P_std + P_mean)` with `P_mean=1.2`, `P_std=1.2` (positive P_mean biases toward higher timesteps)
- Re-added `P_mean` / `P_std` config fields to `DenoiserConfig`
- `generate()` now accepts optional `T` argument to override `num_sampling_steps` (e.g., 1-step inference)
- Renamed `drop_cond` -> `_drop_cond`, `sample_t` -> `_sample_t` (private methods)
- EMA decay: reduced from two checkpoints `(0.9980, 0.9995)` to single `(0.9980,)`

### Training (`scripts/train.py`)

- Gradient clipping: `--grad-norm` default `1.0` -> `inf` (effectively disabled)
- Muon weight decay schedule: default `--muon-wd-type` changed from `constant` to `cosine`
- `_generate_samples` helper now accepts `T` kwarg for configurable inference steps
- Warmup `frac` computation moved before the `if not args.adamw` block so it's available for all optimizer paths

### Misc

- `torch>=2.10.0` -> `torch>=2.11.0` in `pyproject.toml`
- `type: ignore` comments updated to `ty: ignore` across `datasets.py`, `optim.py`

---

## 2026-03-24: Model and Denoiser changes for CIFAR10

### Model (`geoflow/model.py`)

- `nn.Linear` -> `CastedLinear` (fp32 weights, bf16 compute) across all layers
- `RMSNorm`: removed learnable weight param, use `F.rms_norm` directly
- Attention: fused QKV -> separate Q/K/V projections with grouped query attention (GQA, `num_kv_heads = num_heads // 2`)
- Attention: removed `qk_norm` RMSNorm modules, use inline `F.rms_norm` on Q/K instead
- Attention: added learnable `q_gain` scalar per head (init 1.5)
- Attention: `qkv_bias=True` -> `qkv_bias=False`, output proj bias removed
- SwiGLU FFN: removed bias (`bias=True` -> `bias=False`)
- JiTBlock: added `resid_mix` parameter -- learnable blend of current `x` with initial patch embedding `emb` before each block
- JiTBlock: forward now takes `emb` argument for residual mixing
- `torch.compile`: removed `dynamic=False, fullgraph=False` kwargs (use defaults)
- Init: `orthogonal_` -> `xavier_uniform_`
- Added `JiT-S/8` and `JiT-B/4` model configs

### Denoiser (`geoflow/denoiser.py`)

- Timestep sampling: `sigmoid(Normal(P_mean, P_std))` (EDM-style lognormal) -> `Uniform[0,1]`
- Removed `P_mean`/`P_std` config fields
- Velocity clamp eps: single `t_eps` -> use `t_eps` during training, `1e-5` during inference
- `swap_ema`: `.get(decay, default)` -> explicit `None` check with `[]` access

### Training (`scripts/train.py`)

- Dataset: MNIST-only -> configurable `--dataset` (`mnist`, `cifar10`), default `cifar10`
- Default model: `JiT-S/7` -> `JiT-B/4`
- Default epochs: 100 -> 500
- Warmup: fractional ratio (`--warmup 0.05`) -> absolute steps (`--warmup 1000`)
- Muon defaults: `lr 0.01 -> 0.005`, `wd 0.1 -> 0.05`
- Muon weight decay schedule: linear decay -> cosine decay (with `--muon-wd-type` flag)
- Evaluation: removed train-time classifier for `val/acc`, replaced with FID/IS (`compute_fid_is`)
- Sample generation: added `_generate_samples` helper for batched generation
- Validation samples: added 4x nearest-neighbor upscale for wandb logging
- Data loading: added `non_blocking=True` for GPU transfers
- `swap_ema` moved outside `torch.inference_mode()` block
- Checkpoint interval: 2 -> 10
