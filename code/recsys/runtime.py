from dataclasses import dataclass
from pathlib import Path
import random
import numpy as np
import torch
import wandb
from datetime import datetime

from recsys.config import Config
from recsys.config_io import save_config


@dataclass
class Runtime:
    device: torch.device
    data_device: torch.device
    rgn: torch.Generator | None
    wandb_run: wandb.Run | None = None


def build_runtime(
    cfg: Config,
    output_dir="runs",
    wandb_mode: str = "disabled",
    run_dir: Path | None = None,
    wandb_run_id: str | None = None,
) -> Runtime:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_device = torch.device("cpu")
    run_dir: Path
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
            f"norm_o={cfg.model.normalize_output}",
            f"filter_fn={cfg.sampler.filter_false_negatives}",
        ]
        if tag is not None
    ]

    if cfg.sampler.negative_sampling == "mixed":
        run_name = f"{cfg.data.dataset}-{cfg.model.name}-{cfg.sampler.negative_sampling}-ratio={cfg.sampler.ratio}-perm_ratio={cfg.sampler.perm_ratio}"
    else:
        run_name = f"{cfg.data.dataset}-{cfg.model.name}-{cfg.sampler.negative_sampling}-ratio={cfg.sampler.ratio}"

    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(output_dir) / f"{run_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, run_dir / "config.json")
    else:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = wandb.init(
        id=wandb_run_id,
        resume="allow" if wandb_run_id else None,
        name=run_name,
        dir=str(run_dir),
        entity="rs-lp-gnn-xai",
        project="gnn_recsys",
        config=cfg.to_dict(),
        tags=tags,
        mode=wandb_mode,
    )

    return Runtime(
        device=device,
        data_device=data_device,
        run_dir=run_dir,
        rgn=rgn,
        wandb_run=wandb_run,
    )
