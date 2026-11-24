from dataclasses import dataclass
import random
import numpy as np
import torch
import wandb

from recsys.config import Config


@dataclass
class Runtime:
    device: torch.device
    data_device: torch.device
    rgn: torch.Generator | None
    wandb_run: wandb.Run | None = None


def build_runtime(cfg: Config, wandb_mode: str = "disabled") -> Runtime:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_device = torch.device("cpu")

    rgn = None
    if cfg.train.seed is not None:
        rgn = torch.Generator(device=device).manual_seed(cfg.train.seed)
        torch.manual_seed(cfg.train.seed)
        torch.cuda.manual_seed_all(cfg.train.seed)
        random.seed(cfg.train.seed)
        np.random.seed(cfg.train.seed)

    tags = [
        tag
        for tag in [
            cfg.data.dataset,
            f"leave_k={cfg.data.leave_k}",
            cfg.sampler.negative_sampling,
            (  # Caso o amostrador seja mixed, mostrar o ratio de permutações
                f"perm_r={cfg.sampler.perm_ratio}"
                if cfg.sampler.negative_sampling == "mixed"
                else None
            ),
            f"ratio={cfg.sampler.ratio}",
            f"seed={cfg.train.seed}",
        ]
        if tag is not None
    ]

    if cfg.sampler.negative_sampling == "mixed":
        run_name = f"{cfg.data.dataset}-{cfg.model.name}-{cfg.sampler.negative_sampling}-ratio={cfg.sampler.ratio}-perm_ratio={cfg.sampler.perm_ratio}"
    else:
        run_name = f"{cfg.data.dataset}-{cfg.model.name}-{cfg.sampler.negative_sampling}-ratio={cfg.sampler.ratio}"

    wandb_run = wandb.init(
        name=run_name,
        entity="rs-lp-gnn-xai",
        project="gnn_recsys",
        config=cfg.to_dict(),
        tags=tags,
        mode=wandb_mode,
    )

    return Runtime(
        device=device,
        data_device=data_device,
        rgn=rgn,
        wandb_run=wandb_run,
    )
