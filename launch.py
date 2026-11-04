import json
import os
import random
import string
from datetime import datetime

import gcsfs
import hydra
import jax
import wandb
from hydra.core.hydra_config import HydraConfig
from jax.experimental.multihost_utils import sync_global_devices
from omegaconf import OmegaConf

from src.train import train


def new_run_id():
    # wandb.util.generate_id was removed in newer wandb; same 8-char [a-z0-9] format
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


def wandb_identity(experiment, seed):
    # experiment is the hydra choice, e.g. "124M/EpiSelect"; one W&B group per experiment, one run per seed
    if experiment is None:
        return None, None
    group = experiment.replace("/", "-")
    return group, f"{group}-seed{seed}"


def _filesystem(rundir):
    if rundir.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        return fs, fs.open, fs.exists
    return os, open, os.path.exists


@hydra.main(config_name="main.yaml", config_path="./configs", version_base=None)
def main(cfg=None):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.multihost:
        jax.distributed.initialize()

    if cfg.rundir is None and not cfg.debug:
        assert not cfg.multihost, "Multihost runs must specify rundir explicitly."
        cfg.rundir = os.path.join("outputs", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    if not cfg.debug and not cfg.rundir.startswith("gs://"):
        cfg.rundir = os.path.abspath(cfg.rundir)

    if jax.process_index() == 0:
        wandb_key = os.environ.get("WANDB_API_KEY")
        if wandb_key is None and os.path.exists(".wandbkey"):
            with open(".wandbkey") as f:
                wandb_key = f.read().strip()
        if wandb_key:
            wandb.login(key=wandb_key)

        wandb_id = None
        cfg_dict = OmegaConf.to_object(cfg)
        if not cfg.debug:
            print(f"Writing to {cfg.rundir}")
            fs, fopen, exists = _filesystem(cfg.rundir)
            fs.makedirs(cfg.rundir, exist_ok=True)
            with fopen(os.path.join(cfg.rundir, "cfg.json"), "w") as f:
                f.write(json.dumps(cfg_dict))

            wandb_id_path = os.path.join(cfg.rundir, "wandb_id.txt")
            if exists(wandb_id_path):
                with fopen(wandb_id_path, "r") as f:
                    wandb_id = f.read()
            else:
                wandb_id = new_run_id()
                with fopen(wandb_id_path, "w") as f:
                    f.write(wandb_id)

        group, name = wandb_identity(HydraConfig.get().runtime.choices.get("experiment"), cfg.seed)
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "episelect-iclr26"),
            id=wandb_id,
            resume="allow",
            group=group,
            name=name,
            config=cfg_dict,
            tags=[cfg.tag] if cfg.get("tag") else None,
        )
    elif not cfg.debug:
        fs, _, _ = _filesystem(cfg.rundir)
        fs.makedirs(cfg.rundir, exist_ok=True)

    if cfg.multihost:
        sync_global_devices("end_wandb_init")

    train(cfg)


if __name__ == "__main__":
    main()
