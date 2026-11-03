import time
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import degree

from recsys.config import EvalConfig
from recsys.runtime import Runtime


@torch.no_grad()
def _compute_dot_scores_chunked(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    u_flat: torch.Tensor,
    cand_flat: torch.Tensor,
    chunk_size: int = 2_000_000,
) -> torch.Tensor:
    """
    Calcula o produto escalar (user_emb[u] * item_emb[v]).sum(-1) em blocos
    para evitar alocações gigantes de VRAM (evita OutOfMemoryError).
    """
    total = u_flat.numel()
    scores = torch.empty(total, dtype=user_emb.dtype, device=user_emb.device)

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        u_chunk = u_flat[start:end]
        v_chunk = cand_flat[start:end]
        # torch.linalg.vecdot (ou mul + sum) apenas na fatia atual
        scores[start:end] = (user_emb[u_chunk] * item_emb[v_chunk]).sum(dim=-1)

    return scores


@torch.no_grad()
def _sample_negatives_vectorized(
    unique_users: torch.Tensor,  # [U]
    counts: torch.Tensor,  # [U] positivos únicos por usuário
    v_sorted: torch.Tensor,  # [P] itens ordenados por usuário
    n_items: int,
    num_negatives: int,
    device: torch.device,
    rgn: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Amostra itens negativos de forma 100% vetorizada na GPU (`device`),
    mesmo que `rgn` tenha sido instanciado na CPU.
    """
    unique_users = unique_users.to(device, non_blocking=True)
    counts = counts.to(device, non_blocking=True)
    v_sorted = v_sorted.to(device, non_blocking=True)

    U, P = unique_users.numel(), v_sorted.numel()
    if U == 0:
        empty = torch.empty((0, num_negatives), dtype=torch.long, device=device)
        return empty, torch.empty(0, dtype=torch.long, device=device)

    # 1. Hashes lineares (user_local_idx * n_items + item_id) na GPU
    pos_user_local = torch.repeat_interleave(torch.arange(U, device=device), counts)
    pos_hashes = (pos_user_local * n_items + v_sorted).sort().values.contiguous()

    margin = min(n_items, max(num_negatives + 32, int(num_negatives * 1.25)))

    # Se o gerador estiver na CPU, sorteia ou usa seed na GPU para não travar a CPU no argsort/searchsorted
    if rgn is not None and rgn.device != device:
        # Extrai uma seed determinística do gerador CPU para rodar o randint rápido na GPU
        seed = int(
            torch.randint(0, 2**62, (1,), generator=rgn, device=rgn.device).item()
        )
        gpu_rgn = torch.Generator(device=device).manual_seed(seed)
        cand = torch.randint(0, n_items, (U, margin), generator=gpu_rgn, device=device)
    else:
        cand = torch.randint(0, n_items, (U, margin), generator=rgn, device=device)

    # 2. Rejeita candidatos que são positivos via torch.searchsorted na GPU
    if P > 0:
        cand_hash = (
            torch.arange(U, device=device).unsqueeze(1) * n_items + cand
        ).reshape(-1)
        idx = torch.searchsorted(pos_hashes, cand_hash).clamp_(max=P - 1)
        is_pos = (pos_hashes[idx] == cand_hash).view(U, margin)
        del cand_hash, idx
    else:
        is_pos = torch.zeros((U, margin), dtype=torch.bool, device=device)
    valid = ~is_pos
    del is_pos

    # 3. Deduplicação intra-linha na GPU
    sort_idx = cand.argsort(dim=1)
    cand_sorted = cand.gather(1, sort_idx)
    first_occ = torch.ones((U, margin), dtype=torch.bool, device=device)
    first_occ[:, 1:] = cand_sorted[:, 1:] != cand_sorted[:, :-1]
    del cand_sorted
    valid.scatter_(1, sort_idx, first_occ & valid.gather(1, sort_idx))
    del sort_idx, first_occ

    # 4. Seleciona até `max_neg` candidatos válidos por usuário
    max_neg = (n_items - counts).clamp_(min=0, max=num_negatives)
    cumsum = valid.cumsum(dim=1)
    keep = valid & (cumsum <= max_neg.unsqueeze(1))

    negs = torch.full((U, num_negatives), -1, dtype=torch.long, device=device)
    rows, _ = keep.nonzero(as_tuple=True)
    negs[rows, cumsum[keep] - 1] = cand[keep]
    del cand, valid, cumsum, rows

    # 5. Fallback para casos extremos
    n_filled = keep.sum(dim=1)
    incomplete_users = (n_filled < max_neg).nonzero(as_tuple=True)[0]
    if incomplete_users.numel() > 0:
        starts = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)[:-1]]
        )
        for u in incomplete_users.tolist():
            n_have = int(n_filled[u].item())
            n_need = int(max_neg[u].item()) - n_have
            u_pos = v_sorted[starts[u] : starts[u] + counts[u]]

            avail_mask = torch.ones(n_items, dtype=torch.bool, device=device)
            avail_mask[u_pos] = False
            if n_have > 0:
                avail_mask[negs[u, :n_have]] = False

            avail_items = avail_mask.nonzero(as_tuple=True)[0]
            perm = torch.randperm(avail_items.numel(), device=device)[:n_need]
            negs[u, n_have : n_have + n_need] = avail_items[perm]

    return negs, max_neg


@torch.no_grad()
def _compute_metrics_and_topk(
    scores: torch.Tensor,
    target: torch.Tensor,
    user_ids: torch.Tensor,
    item_ids: torch.Tensor,
    k: int,
) -> Tuple[float, float, float, float, torch.Tensor]:
    """
    Calcula NDCG@k, MRR@k, Recall@k, MAP@k e extrai os itens Top-K na GPU.
    """
    dev = scores.device
    N = scores.numel()
    if N == 0:
        return 0.0, 0.0, 0.0, 0.0, torch.empty(0, dtype=torch.long, device=dev)

    # Ordenação composta (user_ids ASC, scores DESC)
    order_score = scores.argsort(descending=True, stable=True)
    order_user = user_ids[order_score].argsort(stable=True)
    fo = order_score[order_user]
    del order_score, order_user

    t_s = target[fo]
    u_s = user_ids[fo]
    v_s = item_ids[fo]
    del fo

    uniq_u, cts = torch.unique_consecutive(u_s, return_counts=True)
    U = uniq_u.numel()
    del u_s, uniq_u

    starts = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=dev), cts.cumsum(0)[:-1]]
    )
    gid = torch.repeat_interleave(torch.arange(U, device=dev), cts)
    rank = torch.arange(N, device=dev) - torch.repeat_interleave(starts, cts)

    in_topk_mask = rank < k
    topk_items = v_s[in_topk_mask]
    del v_s

    is_pos = (t_s > 0).float()
    del t_s
    in_topk_pos = is_pos * in_topk_mask.float()

    # Total de positivos por usuário
    n_pos_user = torch.zeros(U, device=dev).scatter_add_(0, gid, is_pos)
    n_pos_clamped = n_pos_user.clamp(min=1.0)

    # 1. NDCG@k
    disc = torch.where(
        in_topk_mask, 1.0 / torch.log2(rank.float() + 2.0), torch.zeros_like(is_pos)
    )
    dcg_user = torch.zeros(U, device=dev).scatter_add_(0, gid, disc * is_pos)
    del disc, in_topk_mask

    ideal_ranks = torch.arange(k, device=dev, dtype=torch.float)
    ideal_disc = 1.0 / torch.log2(ideal_ranks + 2.0)
    cum_disc = torch.cat([torch.zeros(1, device=dev), ideal_disc.cumsum(0)])
    idcg_user = cum_disc[n_pos_user.clamp(max=k).long()]

    ndcg = (dcg_user / idcg_user.clamp(min=1e-9)).mean().item()

    # 2. MRR@k
    inf_tensor = torch.tensor(float("inf"), device=dev)
    pos_ranks = torch.where(in_topk_pos > 0, rank.float(), inf_tensor)
    first_pos = torch.full((U,), float("inf"), device=dev).scatter_reduce_(
        0, gid, pos_ranks, reduce="amin", include_self=True
    )
    del pos_ranks
    mrr = (
        torch.where(torch.isfinite(first_pos), 1.0 / (first_pos + 1.0), 0.0)
        .mean()
        .item()
    )

    # 3. Recall@k
    hits_user = torch.zeros(U, device=dev).scatter_add_(0, gid, in_topk_pos)
    recall = (hits_user / n_pos_clamped).mean().item()

    # 4. MAP@k
    cs = is_pos.cumsum(0)
    cs_offset = cs - is_pos
    user_cs_start = cs_offset[starts]
    pos_cum_per_user = cs - torch.repeat_interleave(user_cs_start, cts)
    del cs, cs_offset, user_cs_start

    prec_at_k = (pos_cum_per_user / (rank.float() + 1.0)) * in_topk_pos
    ap_user = torch.zeros(U, device=dev).scatter_add_(0, gid, prec_at_k)
    denom_map = torch.minimum(n_pos_clamped, torch.tensor(float(k), device=dev))
    mapk = (ap_user / denom_map).mean().item()

    return ndcg, mrr, recall, mapk, topk_items


@torch.no_grad()
def get_seed_embeddings(
    model: torch.nn.Module,
    data: HeteroData,
    node_type: str,
    num_neighbors: List[int],
    device: torch.device,
    batch_size: int = 4096,
) -> torch.Tensor:
    model.eval()
    N = data[node_type].num_nodes

    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=(node_type, torch.arange(N)),
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    embs: Optional[torch.Tensor] = None
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        out = model(batch.x_dict, batch.edge_index_dict)

        num_seeds = batch[node_type].batch_size
        seed_n_id = batch[node_type].n_id[:num_seeds]
        seed_out = out[node_type][:num_seeds]

        if embs is None:
            embs = torch.empty(
                (N, seed_out.size(1)), dtype=seed_out.dtype, device=device
            )

        embs[seed_n_id] = seed_out

    return embs


@torch.no_grad()
def evaluate_full_graph(
    model: torch.nn.Module,
    data: HeteroData,
    edge_type: tuple[str, str, str],
    eval_cfg: EvalConfig,
    runtime: Runtime,
    eval_edge_index: Optional[torch.Tensor] = None,
    score_chunk_size: int = 2_000_000,
    verbose: bool = True,
) -> Dict[str, float]:
    node_type_user, _, node_type_item = edge_type
    device = runtime.device
    k = eval_cfg.k

    model.eval()

    t0 = time.perf_counter()
    user_emb = get_seed_embeddings(
        model,
        data,
        node_type_user,
        num_neighbors=eval_cfg.num_neighbors,
        device=device,
        batch_size=eval_cfg.eval_batch_size,
    )
    if verbose:
        print(f"[eval] user_emb: {time.perf_counter() - t0:.2f}s", flush=True)

    t0 = time.perf_counter()
    item_emb = get_seed_embeddings(
        model,
        data,
        node_type_item,
        num_neighbors=eval_cfg.num_neighbors,
        device=device,
        batch_size=eval_cfg.eval_batch_size,
    )
    if verbose:
        print(f"[eval] item_emb: {time.perf_counter() - t0:.2f}s", flush=True)

    t0 = time.perf_counter()
    if eval_edge_index is None:
        eval_edge_index = data[edge_type].edge_index

    n_items = data[node_type_item].num_nodes

    # 1. Popularidade dos itens na GPU
    train_items = data[edge_type].edge_index[1].to(device, non_blocking=True)
    item_pop = degree(train_items, num_nodes=n_items, dtype=torch.float)
    popularity = item_pop / item_pop.sum().clamp(min=1.0)
    del train_items, item_pop

    # 2. Ordenação e deduplicação de arestas positivas (u, v) na GPU
    eval_ei = eval_edge_index.to(device, non_blocking=True)
    edge_hashes = eval_ei[0] * n_items + eval_ei[1]
    del eval_ei
    unique_hashes = torch.unique(edge_hashes, sorted=True)
    del edge_hashes
    u_sorted = torch.div(unique_hashes, n_items, rounding_mode="floor")
    v_sorted = unique_hashes % n_items
    del unique_hashes

    unique_users, counts = torch.unique_consecutive(u_sorted, return_counts=True)

    # 3. Amostragem 100% na GPU
    negs, _ = _sample_negatives_vectorized(
        unique_users=unique_users,
        counts=counts,
        v_sorted=v_sorted,
        n_items=n_items,
        num_negatives=eval_cfg.num_negatives,
        device=device,
        rgn=runtime.rgn,
    )
    valid_neg_mask = negs >= 0

    # 4. Construção direta dos tensores flat (positivos + negativos) na GPU
    u_neg = unique_users.unsqueeze(1).expand_as(negs)[valid_neg_mask]
    v_neg = negs[valid_neg_mask]
    del negs, valid_neg_mask, unique_users, counts

    n_pos = v_sorted.numel()
    n_neg = v_neg.numel()

    u_flat = torch.cat([u_sorted, u_neg])
    cand_flat = torch.cat([v_sorted, v_neg])
    del u_sorted, v_sorted, u_neg, v_neg

    # 5. Cálculo de scores em chunks para não estourar VRAM, liberando user_emb e item_emb logo em seguida
    scores = _compute_dot_scores_chunked(
        user_emb=user_emb,
        item_emb=item_emb,
        u_flat=u_flat,
        cand_flat=cand_flat,
        chunk_size=score_chunk_size,
    )
    del user_emb, item_emb

    target = torch.empty(n_pos + n_neg, dtype=torch.float, device=device)
    target[:n_pos] = 1.0
    target[n_pos:] = 0.0

    # 6. Métricas de ranking e Top-K unificados na GPU
    ndcg, mrr, recall, mapk, topk_cat = _compute_metrics_and_topk(
        scores=scores,
        target=target,
        user_ids=u_flat,
        item_ids=cand_flat,
        k=k,
    )
    del scores, target, u_flat, cand_flat

    # 7. Coverage & Novelty na GPU
    if topk_cat.numel() > 0:
        coverage = topk_cat.unique().numel() / n_items
        p = popularity[topk_cat]
        valid_p = p > 0
        novelty = (-torch.log2(p[valid_p])).mean().item() if valid_p.any() else 0.0
    else:
        coverage, novelty = 0.0, 0.0

    if verbose:
        print(f"[eval] metrics & sampling: {time.perf_counter() - t0:.2f}s", flush=True)

    return {
        f"ndcg@{k}": ndcg,
        f"mrr@{k}": mrr,
        f"map@{k}": mapk,
        f"recall@{k}": recall,
        "coverage": coverage,
        "novelty": novelty,
    }
