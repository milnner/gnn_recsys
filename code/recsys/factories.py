import torch
from recsys.config import ModelConfig, OptimizerConfig
from recsys.models.hetero_sage import HeteroGraphSAGE


def build_model(
    cfg: ModelConfig,
    metadata: tuple,
    in_channels_dict: dict[str, int],
) -> torch.nn.Module:
    if cfg.name == "hetero_sage":
        return HeteroGraphSAGE(
            metadata,
            in_channels_dict=in_channels_dict,
            hidden_channels=cfg.hidden_channels,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            aggr=cfg.conv_aggr,
            hetero_aggr=cfg.hetero_aggr,
            normalize=cfg.normalize_conv,
            normalize_intern=cfg.normalize_conv,
            input_proj=cfg.input_proj,
        )
    raise ValueError(f"Modelo desconhecido: {cfg.name!r}")


def build_optimizer(
    cfg: OptimizerConfig,
    parameters,
) -> torch.optim.Optimizer:
    if cfg.name == "adam":
        return torch.optim.Adam(
            parameters, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas
        )
    if cfg.name == "adamw":
        return torch.optim.AdamW(
            parameters, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas
        )
    if cfg.name == "sgd":
        return torch.optim.SGD(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Otimizador desconhecido: {cfg.name!r}")
