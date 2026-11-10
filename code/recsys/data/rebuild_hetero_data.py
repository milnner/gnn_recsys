import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

# =============================================================
def rebuild_hetero(template: HeteroData, keep_edge_idx: torch.Tensor,
  edge, rev_edge) -> HeteroData:
  d = HeteroData()
  # nós
  for ntype in template.node_types:
    d[ntype].num_nodes = template[ntype].num_nodes
    if 'x' in template[ntype]:
      d[ntype].x = template[ntype].x
  # arestas (somente as de treino, para não vazar)
  ei = template[edge].edge_index[:, keep_edge_idx]
  d[edge].edge_index     = ei
  d[rev_edge].edge_index = ei.flip(0)
  if 'edge_attr' in template[edge]:
    d[edge].edge_attr     = template[edge].edge_attr[keep_edge_idx]
    d[rev_edge].edge_attr = template[edge].edge_attr[keep_edge_idx]
  return d
