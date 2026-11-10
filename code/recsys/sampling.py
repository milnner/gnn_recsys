from __future__ import annotations


import torch
from torch import Tensor
from torch_geometric.data import HeteroData


def _num_dst_nodes_in_batch(batch, edge_type: tuple[str, str, str]) -> int:
  """Número de nós de destino no batch amostrado para este edge type.
  """
  _, _, dst_type = edge_type
  return int(batch[dst_type].num_nodes)


def _finalize_in_batch(
    pos_edge_index: Tensor,
    neg_dst: Tensor,
    pos_target : Tensor,
    ratio: int,
) -> tuple[Tensor, Tensor]:
    """Monta ``(edge_label_index, target_labels)`` com um layout interleaved.

    Positivos antes e depois negativos. Negativos para o mesmo usuário são
    contiguos (``repeat_interleave``), necessário para rank por usuário.
    """
    device = pos_edge_index.device
    num_pos = pos_edge_index.size(1)
    neg_users = pos_edge_index[0].repeat_interleave(ratio)
    neg_edge_index = torch.stack([neg_users, neg_dst], dim=0)

    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    target_labels = torch.cat(
      [
        pos_target,
        torch.zeros(num_pos * ratio, device=device, dtype=pos_target.dtype),
      ],
      dim=0,
    )
    return edge_label_index, target_labels


def _perm_negatives(pos: Tensor, rgn : torch.Generator) -> Tensor:
    """Uma permutação dos positivos."""
    return pos[torch.randperm(pos.size(1), device=pos.device, generator=rgn)]


def _uniform_negatives(
    num_items_in_batch: int, num_pos: int,  rgn : torch.Generator
) -> Tensor:
  """A single draw of uniform negatives from the sampled subgraph."""
  return torch.randint(
    0, num_items_in_batch, (num_pos,), device=rgn.device, generator=rgn
  )


def _validate_ratio(ratio: int) -> None:
  if not isinstance(ratio, int) or ratio < 0:
    raise ValueError(f"ratio must be a non-negative int, got {ratio!r}")


def sample_negatives_permutation(
  batch : HeteroData,
  edge_type: tuple[str, str, str],
  ratio: int,
  rgn : torch.Generator,
) -> tuple[Tensor, Tensor]:
  """Todos os negativos são permutações dos positivos no batch"""
  if not isinstance(ratio, int) or ratio < 0:
    raise ValueError(f"ratio must be a non-negative int, got {ratio!r}")
  
  pos = batch[edge_type].edge_label_index
  device = pos.device

  if ratio == 0:
    return _finalize_in_batch(
      pos, torch.empty(0, dtype=pos.dtype, device=device), 0, device
    )
  
  negs = torch.stack(
    [_perm_negatives(pos[1], rgn) for _ in range(ratio)], dim=1
  ).flatten()

  return _finalize_in_batch(pos, negs, ratio, device)

def sample_negatives_uniform_only(
    batch : HeteroData,
    edge_type: tuple[str, str, str],
    ratio: int,
) -> tuple[Tensor, Tensor]:
    """Todos negativos são de uma distribuição uniforme"""
    if not isinstance(ratio, int) or ratio < 0:
      raise ValueError(f"ratio must be a non-negative int, got {ratio!r}")
    
    pos = batch[edge_type].edge_label_index
    device = pos.device
    num_pos = pos.size(1)

    if ratio == 0:
      return _finalize_in_batch(
        pos, torch.empty(0, dtype=pos.dtype, device=device), 0
      )

    M = _num_dst_nodes_in_batch(batch,edge_type)
    negs = _uniform_negatives(M, num_pos * ratio, device)
    return _finalize_in_batch(pos, negs, ratio, device)


def sample_negatives_mixed(
    batch,
    edge_type: tuple[str, str, str],
    ratio: int,
    perm_ratio: int,
    rgn : torch.Generator
) -> tuple[Tensor, Tensor]:
    """
    Mistura de negativos a partir de permutação e uniformes.

    ``perm_ratio`` (número de permutações) ;
    """
    if not isinstance(ratio, int) or ratio < 0:
      raise ValueError(f"ratio must be a non-negative int, got {ratio!r}")
        
    pos = batch[edge_type].edge_label_index
    device = pos.device
    num_pos = pos.size(1)

    if perm_ratio is None:
        perm_ratio = ratio // 2

    if not 0 <= perm_ratio <= ratio:
        raise ValueError(f"perm_ratio={perm_ratio} outside [0, ratio={ratio}]")
    
    uniform_ratio = ratio - perm_ratio

    if ratio == 0:
        return _finalize_in_batch(
            pos, torch.empty(0, dtype=pos.dtype, device=device), 0, device
        )

    M = _num_dst_nodes_in_batch(batch, edge_type)
    pieces: list[Tensor] = []

    for _ in range(perm_ratio):
        pieces.append(_perm_negatives(pos[1], num_pos, device))

    for _ in range(uniform_ratio):
        pieces.append(_uniform_negatives(M, num_pos, device))

    negs = torch.stack(pieces, dim=1).flatten()
    return _finalize_in_batch(pos, negs, ratio, device)

