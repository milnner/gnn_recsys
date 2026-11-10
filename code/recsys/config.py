from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class Config:
    # amostragem
  ratio: int = 1
  rgn: torch.Generator | None = None
  negative_sampling: Literal["perm", "mixed", "uniform", None] = "uniform"
  perm_ratio: int | None = None
  filter_false_negatives: bool = True

  # treino
  num_epochs: int = 10_000
  l2_reg: float = 0.0
  grad_clip: float | None = 5.0
  min_delta: float = 1e-4
  log_every: int = 1
  bpr_reduction: Literal["mean_triples", "mean_positives"] = "mean_positives"
  lr: float = 1e-3

  # early stopping
  patience: int = 10
  patience_metric: str = "ndcg@{k}"
  patience_mode: Literal["max", "min"] = "max"

  # avaliação / loader
  k: int = 10
  eval_mode: Literal["loader", "full_graph"] = "full_graph"
  num_neighbors: list[int] | None = None       # ← usado no LinkNeighborLoader
  eval_num_neighbors: list[int] | None = None  # ← usado em get_seed_embeddings
  eval_num_negatives: int = 100
  eval_seed: int = 42
  eval_batch_size: int = 4096

  # grafo
  n_users: int | None = None
  n_items: int | None = None