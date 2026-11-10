
from typing import List

import torch
from torchmetrics import RetrievalNormalizedDCG, RetrievalMRR, RetrievalMAP, RetrievalRecall

import torch
from collections import defaultdict
from torch_geometric.utils import degree
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from torchmetrics import (
  RetrievalNormalizedDCG,
  RetrievalMRR,
  RetrievalMAP,
  RetrievalRecall,
)


@torch.no_grad()
def get_seed_embeddings(model : torch.nn.Module, data : HeteroData, node_type : str, num_neighbors: List ,device : torch.device, batch_size : int):
  """
  model : modelo
  data : HeteroData
  node_type : str
  num_neighbors : List -> vizinhos a serem amostrados
  Retorna tensor [N_node_type, D] com embedding de todos os nós de `node_type`,
  calculados como sementes em NeighborLoader.
  """
  model.eval()
  N = data[node_type].num_nodes

  loader = NeighborLoader(
    data,
    num_neighbors=num_neighbors,
    batch_size=batch_size,
    input_nodes=(node_type, torch.arange(N)),
    shuffle=False,
  )

  embs = None
  for batch in loader:
    batch = batch.to(device)
    out = model(batch.x_dict, batch.edge_index_dict)

    num_seeds = batch[node_type].batch_size
    seed_n_id = batch[node_type].n_id[:num_seeds].cpu()
    seed_out  = out[node_type][:num_seeds].cpu()

    if embs is None:
      embs = torch.empty(N, seed_out.size(1),dtype=seed_out.dtype)

    embs[seed_n_id] = seed_out
  return embs

@torch.no_grad()
def evaluate_full_graph(
  model: torch.nn.Module,
  data: HeteroData,               # grafo completo (ou fold_data)
  node_type_user: str,
  node_type_item: str,
  edge_type: tuple,
  device: torch.device = torch.device("cpu"),
  k: int = 10,
  num_negatives: int = 100,       # negativos amostrados por usuário
  eval_edge_index: torch.Tensor = None,  # arestas positivas para avaliação
  seed: int = 42,
) -> dict:
    """
    Avaliação no grafo completo (sem numpy).

    Retorna métricas de ranking (NDCG, MRR, MAP, Recall) + Coverage + Novelty.
    """
    model.eval()

    # 1. Embeddings finais de todos os nós
    user_emb = get_seed_embeddings(
        model, data, node_type_user,
        num_neighbors=[15, 10], device=device, batch_size=4096,
    )
    item_emb = get_seed_embeddings(
        model, data, node_type_item,
        num_neighbors=[15, 10], device=device, batch_size=4096,
    )

    # 2. Define arestas de avaliação
    if eval_edge_index is None:
        eval_edge_index = data[edge_type].edge_index
    u_eval, v_eval = eval_edge_index

    # 3. Popularidade dos itens (baseada no grafo de treino)
    ei = data[edge_type].edge_index
    item_pop = degree(ei[1], data[node_type_item].num_nodes, dtype=torch.float)
    total_interactions = item_pop.sum()
    popularity = item_pop / total_interactions        # em CPU

    n_items = data[node_type_item].num_nodes

    # 4. Agrupa itens positivos por usuário
    pos_per_user = defaultdict(list)
    for u, v in zip(u_eval.tolist(), v_eval.tolist()):
        pos_per_user[u].append(v)

    # 5. Amostragem de negativos e cálculo das métricas
    generator = torch.Generator().manual_seed(seed)
    all_preds, all_target, all_indexes = [], [], []
    topk_items = []

    for u, pos_items in pos_per_user.items():
        pos_set = set(pos_items)
        n_neg = min(num_negatives, n_items - len(pos_set))

        perm = torch.randperm(n_items, generator=generator)
        neg_items = []
        for idx in perm.tolist():
            if idx in pos_set:
                continue
            neg_items.append(idx)
            if len(neg_items) >= n_neg:
                break

        candidates = pos_items + neg_items

        # Pontuação para o usuário u em todos os candidatos
        u_tensor = torch.tensor([u], device=device)
        cand_tensor = torch.tensor(candidates, device=device)
        scores = (user_emb[u_tensor] * item_emb[cand_tensor]).sum(dim=-1).cpu()

        # Rótulos: 1 para positivos, 0 para negativos
        labels = torch.tensor(
            [1.0] * len(pos_items) + [0.0] * len(neg_items), dtype=torch.float
        )

        all_preds.append(scores)
        all_target.append(labels)
        all_indexes.append(torch.full((len(candidates),), u, dtype=torch.long))

        # Top-K para coverage e novelty
        topk_idx = torch.topk(scores, k=min(k, len(candidates))).indices
        topk_items.extend([candidates[i] for i in topk_idx.tolist()])

    # Concatena tudo
    preds   = torch.cat(all_preds)
    target  = torch.cat(all_target)
    indexes = torch.cat(all_indexes)

    # 6. Métricas de ranking (torchmetrics em CPU, sem mover métricas)
    ndcg_metric   = RetrievalNormalizedDCG(top_k=k)
    mrr_metric    = RetrievalMRR(top_k=k)
    map_metric    = RetrievalMAP(top_k=k)
    recall_metric = RetrievalRecall(top_k=k)

    ndcg   = ndcg_metric(preds, target, indexes=indexes).item()
    mrr    = mrr_metric(preds, target, indexes=indexes).item()
    mapk   = map_metric(preds, target, indexes=indexes).item()
    recall = recall_metric(preds, target, indexes=indexes).item()

    # 7. Coverage
    coverage = len(set(topk_items)) / n_items

    # 8. Novelty (auto-informação em log2)
    if topk_items:
        item_ids = torch.tensor(topk_items, dtype=torch.long)
        p = popularity[item_ids]
        valid = p > 0
        if valid.any():
            novelty = (-torch.log2(p[valid])).mean().item()
        else:
            novelty = 0.0
    else:
        novelty = 0.0

    return {
        f"ndcg@{k}": ndcg,
        f"mrr@{k}": mrr,
        f"map@{k}": mapk,
        f"recall@{k}": recall,
        "coverage": coverage,
        "novelty": novelty,
    }

@torch.no_grad()
def evaluate_loader(
    model, loader, edge_type, device, k=10,
    train_edge_index: torch.Tensor = None,
    n_items: int = None,
):
    user_type, _, item_type = edge_type          # <- fix
    model.eval()
    ndcg_m = RetrievalNormalizedDCG(top_k=k).to(device)
    mrr_m  = RetrievalMRR(top_k=k).to(device)
    map_m  = RetrievalMAP(top_k=k).to(device)
    rec_m  = RetrievalRecall(top_k=k).to(device)

    all_preds, all_target, all_idx, topk_items = [], [], [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict)

        u, v = batch[edge_type].edge_label_index
        scores = (out[user_type][u] * out[item_type][v]).sum(-1)   # <- fix
        labels = batch[edge_type].edge_label.float()

        order = torch.argsort(u, stable=True)
        u_sorted = u[order]
        _, counts = torch.unique_consecutive(u_sorted, return_counts=True)
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                            counts.cumsum(0)[:-1]])
        for s, c in zip(starts.tolist(), counts.tolist()):
            grp = order[s:s + c]
            top_local = grp[torch.topk(scores[grp], k=min(k, c)).indices]
            topk_items.append(v[top_local].cpu())

        all_preds.append(scores.cpu())
        all_target.append(labels.cpu())
        all_idx.append(u.cpu())

    preds, target, idx = (torch.cat(all_preds),
                          torch.cat(all_target),
                          torch.cat(all_idx))

    ndcg = ndcg_m(preds, target, indexes=idx).item()
    mrr  = mrr_m(preds, target, indexes=idx).item()
    mapk = map_m(preds, target, indexes=idx).item()
    rec  = rec_m(preds, target, indexes=idx).item()

    coverage, novelty = None, None
    if n_items is not None and topk_items:
        coverage = torch.cat(topk_items).unique().numel() / n_items
    if train_edge_index is not None and n_items is not None and topk_items:
        pop = torch.zeros(n_items, dtype=torch.float)
        pop.scatter_add_(0, train_edge_index[1].cpu(),
                         torch.ones(train_edge_index.size(1)))
        pop = pop / pop.sum().clamp(min=1)
        p = pop[torch.cat(topk_items)]
        valid = p > 0
        novelty = (-torch.log2(p[valid])).mean().item() if valid.any() else 0.0

    return {f"ndcg@{k}": ndcg, f"mrr@{k}": mrr,
            f"map@{k}": mapk, f"recall@{k}": rec,
            "coverage": coverage, "novelty": novelty}
