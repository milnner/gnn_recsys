from __future__ import annotations


import torch
from torch import Tensor
from torch_geometric.data import HeteroData


def _num_dst_nodes_in_batch(batch, edge_type: tuple[str, str, str]) -> int:
  """Número de nós de destino no batch amostrado para este edge type.
  """
  _, _, dst_type = edge_type
  return int(batch[dst_type].num_nodes)

def _finalize_in_batch(pos_edge_index: Tensor, neg_dst: Tensor, ratio: int):
    """Positivos primeiro, depois negativos agrupados por usuário (repeat_interleave)."""
    device = pos_edge_index.device
    num_pos = pos_edge_index.size(1)

    neg_users = pos_edge_index[0].repeat_interleave(ratio)
    neg_edge_index = torch.stack([neg_users, neg_dst.to(device)], dim=0)
    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

    labels = torch.cat([
      torch.ones(num_pos, device=device, dtype=torch.float32),
      torch.zeros(num_pos * ratio, device=device, dtype=torch.float32),
    ])
    return edge_label_index, labels


def _perm_negatives(pos: Tensor, rgn: torch.Generator) -> Tensor:
  return pos[torch.randperm(pos.size(0), device=pos.device, generator=rgn)]


def _uniform_negatives(num_items_in_batch: int, num_pos: int,
  rgn: torch.Generator) -> Tensor:
  return torch.randint(0, num_items_in_batch, (num_pos,),
    device=rgn.device, generator=rgn)


def sample_negatives_permutation(batch, edge_type, ratio, rgn):
  if not isinstance(ratio, int) or ratio < 0:
    raise ValueError(f"ratio must be a non-negative int, got {ratio!r}")
  
  pos = batch[edge_type].edge_label_index
  if ratio == 0:
    return _finalize_in_batch(pos, torch.empty(0, dtype=pos.dtype, device=pos.device), 0)
  pieces = [_perm_negatives(pos[1], rgn) for _ in range(ratio)]
  negs = torch.stack(pieces, dim=1).flatten()          # [num_pos*ratio] agrupado por user
  return _finalize_in_batch(pos, negs, ratio)


def sample_negatives_uniform(batch, edge_type, ratio, rgn):
  if not isinstance(ratio, int) or ratio < 0:
    raise ValueError(f"ratio must be a non-negative int, got {ratio!r}")
  pos = batch[edge_type].edge_label_index
  num_pos = pos.size(1)
  if ratio == 0:
    return _finalize_in_batch(pos, torch.empty(0, dtype=pos.dtype, device=pos.device), 0)
  M = _num_dst_nodes_in_batch(batch, edge_type)
  negs = _uniform_negatives(M, num_pos * ratio, rgn)
  return _finalize_in_batch(pos, negs, ratio)


def sample_negatives_mixed(batch, edge_type, ratio, perm_ratio, rgn):
  if not isinstance(ratio, int) or ratio < 0:
    raise ValueError(f"ratio must be a non-negative int, got {ratio!r}")
  if perm_ratio is None:
    perm_ratio = ratio // 2
  if not 0 <= perm_ratio <= ratio:
    raise ValueError(f"perm_ratio={perm_ratio} fora de [0, {ratio}]")
  pos = batch[edge_type].edge_label_index
  num_pos = pos.size(1)
  if ratio == 0:
    return _finalize_in_batch(pos, torch.empty(0, dtype=pos.dtype, device=pos.device), 0)
  M = _num_dst_nodes_in_batch(batch, edge_type)
  pieces = []
  for _ in range(perm_ratio):
    pieces.append(_perm_negatives(pos[1], rgn))
  for _ in range(ratio - perm_ratio):
    pieces.append(_uniform_negatives(M, num_pos, rgn))
  negs = torch.stack(pieces, dim=1).flatten()
  return _finalize_in_batch(pos, negs, ratio)

NEGATIVE_SAMPLING = {
  "perm": sample_negatives_permutation,
  "uniform": sample_negatives_uniform,
  "mixed": sample_negatives_mixed,
}