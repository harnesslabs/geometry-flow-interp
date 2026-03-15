import shlex

import modal

app = modal.App("geoflow")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "torch",
        "torchvision",
        "einops",
        "torchinfo",
        "wandb",
        "matplotlib",
        "numpy",
    )
    .add_local_python_source("geoflow")
    .add_local_file("scripts/train.py", "/root/train.py")
    .add_local_dir("fid_stats", "/root/fid_stats")
)

volume = modal.Volume.from_name("geoflow-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=6 * 60 * 60,
)
def train(train_args: str = ""):
    import os
    import sys

    # Make train.py importable
    sys.path.insert(0, "/root")

    # Symlink dataset cache to volume so downloads persist
    os.makedirs("/data/cache", exist_ok=True)
    cache_dir = os.path.expanduser("~/.cache/data")
    os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
    if not os.path.exists(cache_dir):
        os.symlink("/data/cache", cache_dir)

    # Set working directory to /root so fid_stats/ is found
    os.chdir("/root")

    # Default to offline wandb unless key is set
    if not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_MODE"] = "offline"

    import train as train_module

    argv = shlex.split(train_args)
    # Override checkpoint dir to volume
    if "--checkpoint-dir" not in train_args:
        argv += ["--checkpoint-dir", "/data/checkpoints"]

    args = train_module.parser.parse_args(argv)
    train_module.main(args)

    volume.commit()


@app.local_entrypoint()
def main(train_args: str = ""):
    train.remote(train_args)
