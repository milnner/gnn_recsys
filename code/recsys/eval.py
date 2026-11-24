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
def _compute_metrics_and_topk(
    topk_items: torch.Tensor,  # [B, max_k]
    hits: torch.Tensor,  # [B, max_k] float (0.0 ou 1.0)
    n_pos: torch.Tensor,  # [B] total de positivos do usuário na avaliação
    popularity: torch.Tensor,  # [n_items] distribuição de popularidade no treino
    ks: List[int],
    n_items: int,
    ranks: torch.Tensor,  # [max_k] 1..max_k
    disc: torch.Tensor,  # [max_k] 1 / log2(rank + 1)
    cum_disc: torch.Tensor,  # [max_k + 1] IDCG lookup
    seen_in_topk: Dict[int, torch.Tensor],  # {k: bool[n_items]} atualizado in-place
) -> Dict[int, Dict[str, float]]:
    """
    Calcula as somas das métricas de ranking (NDCG, MRR, Recall, MAP) e Novelty
    para múltiplos valores de k em um batch de usuários, atualizando Coverage in-place.
    """
    n_pos_clamped = n_pos.clamp(min=1.0)

    # Acumulados ao longo do eixo de rank (calculados 1x para todos os ks)
    hits_cum = hits.cumsum(dim=1)
    dcg_cum = (hits * disc).cumsum(dim=1)
    ap_cum = ((hits_cum / ranks) * hits).cumsum(dim=1)

    has_hit_max = hits.any(dim=1)
    first_rank = hits.argmax(dim=1)  # 0-based

    p = popularity[topk_items]
    valid_p = p > 0
    nov_vals = torch.where(valid_p, -torch.log2(p.clamp(min=1e-12)), 0.0)
    nov_cum = nov_vals.cumsum(dim=1)
    valid_p_cum = valid_p.cumsum(dim=1)

    batch_stats: Dict[int, Dict[str, float]] = {}

    for k in ks:
        k_eff = min(k, n_items)
        idx = k_eff - 1

        idcg = cum_disc[n_pos.clamp(max=k_eff).long()]
        ndcg = (dcg_cum[:, idx] / idcg.clamp(min=1e-9)).sum().item()

        has_hit_k = has_hit_max & (first_rank < k_eff)
        mrr = torch.where(has_hit_k, 1.0 / (first_rank.float() + 1.0), 0.0).sum().item()

        recall = (hits_cum[:, idx] / n_pos_clamped).sum().item()

        denom_map = n_pos_clamped.clamp(max=float(k_eff))
        mapk = (ap_cum[:, idx] / denom_map).sum().item()

        seen_in_topk[k][topk_items[:, :k_eff].reshape(-1)] = True
        nov_sum = nov_cum[:, idx].sum().item()
        nov_cnt = float(valid_p_cum[:, idx].sum().item())

        batch_stats[k] = {
            "ndcg": ndcg,
            "mrr": mrr,
            "recall": recall,
            "map": mapk,
            "novelty_sum": nov_sum,
            "novelty_count": nov_cnt,
        }

    return batch_stats


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
def _sample_negatives_vectorized(
    unique_users: torch.Tensor,  # [U] IDs globais dos usuários avaliados
    exclude_hashes: torch.Tensor,  # [E] hashes (u * n_items + v) ordenados e únicos (treino + eval)
    n_items: int,
    num_negatives: int,
    device: torch.device,
    rgn: Optional[torch.Generator] = None,
    item_probs: Optional[
        torch.Tensor
    ] = None,  # [n_items] distribuição para amostragem por popularidade
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Amostra itens negativos 100% na GPU, rejeitando todas as interações em `exclude_hashes`
    (treino + avaliação). Suporta amostragem uniforme ou por popularidade (`item_probs`).
    """
    unique_users = unique_users.to(device, non_blocking=True)
    exclude_hashes = exclude_hashes.to(device, non_blocking=True)

    U = unique_users.numel()
    E = exclude_hashes.numel()
    if U == 0:
        empty = torch.empty((0, num_negatives), dtype=torch.long, device=device)
        return empty, torch.empty(0, dtype=torch.long, device=device)

    # 1. Intervalo [low, high) de itens excluídos para cada usuário em `exclude_hashes`
    if E > 0:
        low = torch.searchsorted(exclude_hashes, unique_users * n_items, right=False)
        high = torch.searchsorted(
            exclude_hashes, (unique_users + 1) * n_items, right=False
        )
        excl_counts = high - low
    else:
        low = torch.zeros(U, dtype=torch.long, device=device)
        high = torch.zeros(U, dtype=torch.long, device=device)
        excl_counts = torch.zeros(U, dtype=torch.long, device=device)

    # Margem maior para popularidade, pois há mais colisões (itens repetidos e positivos de treino)
    if item_probs is not None:
        margin = min(n_items, max(num_negatives + 64, int(num_negatives * 2.0)))
    else:
        margin = min(n_items, max(num_negatives + 32, int(num_negatives * 1.25)))

    # Garante gerador na GPU para não travar a CPU
    if rgn is not None and rgn.device != device:
        seed = int(
            torch.randint(0, 2**62, (1,), generator=rgn, device=rgn.device).item()
        )
        gen = torch.Generator(device=device).manual_seed(seed)
    else:
        gen = rgn

    # 2. Sorteio dos candidatos (uniforme ou por popularidade via CDF + searchsorted)
    if item_probs is not None:
        probs = item_probs.to(device=device, dtype=torch.float32, non_blocking=True)
        cdf = probs.cumsum(dim=0)
        cdf[-1] = 1.0  # evita erro de arredondamento no limite superior
        rand_vals = torch.rand((U, margin), generator=gen, device=device)
        cand = torch.searchsorted(cdf, rand_vals).clamp_(max=n_items - 1)
        del cdf, rand_vals
    else:
        cand = torch.randint(0, n_items, (U, margin), generator=gen, device=device)

    # 3. Rejeita candidatos que pertencem a treino ou avaliação
    if E > 0:
        cand_hash = (unique_users.unsqueeze(1) * n_items + cand).reshape(-1)
        idx = torch.searchsorted(exclude_hashes, cand_hash).clamp_(max=E - 1)
        is_excl = (exclude_hashes[idx] == cand_hash).view(U, margin)
        del cand_hash, idx
    else:
        is_excl = torch.zeros((U, margin), dtype=torch.bool, device=device)
    valid = ~is_excl
    del is_excl

    # 4. Deduplicação intra-linha na GPU
    sort_idx = cand.argsort(dim=1)
    cand_sorted = cand.gather(1, sort_idx)
    first_occ = torch.ones((U, margin), dtype=torch.bool, device=device)
    first_occ[:, 1:] = cand_sorted[:, 1:] != cand_sorted[:, :-1]
    del cand_sorted
    valid.scatter_(1, sort_idx, first_occ & valid.gather(1, sort_idx))
    del sort_idx, first_occ

    # 5. Seleciona até `max_neg` candidatos válidos por usuário
    max_neg = (n_items - excl_counts).clamp_(min=0, max=num_negatives)
    cumsum = valid.cumsum(dim=1)
    keep = valid & (cumsum <= max_neg.unsqueeze(1))

    negs = torch.full((U, num_negatives), -1, dtype=torch.long, device=device)
    rows, _ = keep.nonzero(as_tuple=True)
    negs[rows, cumsum[keep] - 1] = cand[keep]
    del cand, valid, cumsum, rows

    # 6. Fallback para usuários que não atingiram `max_neg`
    n_filled = keep.sum(dim=1)
    incomplete_users = (n_filled < max_neg).nonzero(as_tuple=True)[0]
    if incomplete_users.numel() > 0:
        for u in incomplete_users.tolist():
            n_have = int(n_filled[u].item())
            n_need = int(max_neg[u].item()) - n_have

            avail_mask = torch.ones(n_items, dtype=torch.bool, device=device)
            if high[u] > low[u]:
                u_excl_items = exclude_hashes[low[u] : high[u]] % n_items
                avail_mask[u_excl_items] = False
            if n_have > 0:
                avail_mask[negs[u, :n_have]] = False

            avail_items = avail_mask.nonzero(as_tuple=True)[0]
            if item_probs is not None:
                avail_probs = probs[avail_items]
                if avail_probs.sum() > 0:
                    chosen_idx = torch.multinomial(
                        avail_probs,
                        num_samples=n_need,
                        replacement=False,
                        generator=gen,
                    )
                else:
                    chosen_idx = torch.randperm(
                        avail_items.numel(), generator=gen, device=device
                    )[:n_need]
            else:
                chosen_idx = torch.randperm(
                    avail_items.numel(), generator=gen, device=device
                )[:n_need]

            negs[u, n_have : n_have + n_need] = avail_items[chosen_idx]

    return negs, max_neg


@torch.no_grad()
def evaluate_full_ranking(
    model: torch.nn.Module,
    data: HeteroData,
    edge_type: tuple[str, str, str],
    eval_cfg: EvalConfig,
    runtime: Runtime,
    eval_edge_index: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    node_type_user, _, node_type_item = edge_type
    device = runtime.device
    ks = sorted(set(eval_cfg.ks))
    if not ks or ks[0] <= 0:
        raise ValueError(
            f"eval_cfg.ks deve conter inteiros positivos, recebido: {eval_cfg.ks}"
        )
    # print(model.__class__.__name__)
    # print(data.node_types)
    # print(node_type_user)
    # print(eval_cfg.num_neighbors)
    # print(device)
    # print(eval_cfg.eval_batch_size)
    if eval_cfg.eval_batch_size == None:
        raise ValueError(
            "eval_cfg.eval_batch_size=None, para fazer aavaliação é necessário entrar com um inteiro para o batch"
        )

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
    train_ei = data[edge_type].edge_index.to(device, non_blocking=True)
    mask_train = eval_edge_index is not None
    eval_ei = (
        train_ei
        if eval_edge_index is None
        else eval_edge_index.to(device, non_blocking=True)
    )
    n_items = data[node_type_item].num_nodes
    max_k = min(ks[-1], n_items)

    # 1. Popularidade dos itens no treino (para Novelty)
    item_pop = degree(train_ei[1], num_nodes=n_items, dtype=torch.float)
    popularity = item_pop / item_pop.sum().clamp(min=1.0)
    del item_pop

    # 2. Arestas de avaliação ordenadas e deduplicadas
    eval_hashes = torch.unique(eval_ei[0] * n_items + eval_ei[1], sorted=True)
    eval_u = torch.div(eval_hashes, n_items, rounding_mode="floor")
    del eval_ei

    unique_users, eval_counts = torch.unique_consecutive(eval_u, return_counts=True)
    del eval_u
    U = unique_users.numel()

    if U == 0:
        return {
            f"{m}@{k}": 0.0
            for k in ks
            for m in ("ndcg", "mrr", "map", "recall", "coverage", "novelty")
        }

    eval_starts = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), eval_counts.cumsum(0)[:-1]]
    )

    # 3. Ordena arestas de treino por usuário para máscara rápida por batch
    if mask_train and train_ei.numel() > 0:
        train_order = train_ei[0].argsort()
        train_u = train_ei[0, train_order].contiguous()
        train_v = train_ei[1, train_order].contiguous()
        del train_order
    else:
        train_u = torch.empty(0, dtype=torch.long, device=device)
        train_v = torch.empty(0, dtype=torch.long, device=device)
    del train_ei

    # Constantes pré-calculadas para _compute_metrics_and_topk
    ranks = torch.arange(1, max_k + 1, device=device, dtype=torch.float)
    disc = 1.0 / torch.log2(ranks + 1.0)
    cum_disc = torch.cat([torch.zeros(1, device=device), disc.cumsum(0)])

    item_emb_t = item_emb.T.contiguous()
    del item_emb

    # Acumuladores globais
    totals: Dict[int, Dict[str, float]] = {
        k: {
            "ndcg": 0.0,
            "mrr": 0.0,
            "recall": 0.0,
            "map": 0.0,
            "novelty_sum": 0.0,
            "novelty_count": 0.0,
        }
        for k in ks
    }
    seen_in_topk = {
        k: torch.zeros(n_items, dtype=torch.bool, device=device) for k in ks
    }

    # 4. Loop em batches de usuários na GPU
    for b_start in range(0, U, eval_cfg.user_batch_size):
        b_end = min(b_start + eval_cfg.user_batch_size, U)
        batch_u = unique_users[b_start:b_end]
        B = batch_u.numel()

        scores = user_emb[batch_u] @ item_emb_t

        # Mascara itens de treino com -inf
        if mask_train and train_u.numel() > 0:
            t_low = int(torch.searchsorted(train_u, batch_u[0], right=False).item())
            t_high = int(torch.searchsorted(train_u, batch_u[-1], right=True).item())
            if t_high > t_low:
                tu = train_u[t_low:t_high]
                tv = train_v[t_low:t_high]
                local_u = torch.searchsorted(batch_u, tu).clamp_(max=B - 1)
                in_batch = batch_u[local_u] == tu
                scores[local_u[in_batch], tv[in_batch]] = float("-inf")

        _, topk_items = torch.topk(scores, k=max_k, dim=1, largest=True, sorted=True)
        del scores

        # Verifica hits contra os positivos do batch
        e_start = int(eval_starts[b_start].item())
        e_end = int((eval_starts[b_end - 1] + eval_counts[b_end - 1]).item())
        batch_eval_hashes = eval_hashes[e_start:e_end]

        topk_hashes = (batch_u.unsqueeze(1) * n_items + topk_items).reshape(-1)
        pos_idx = torch.searchsorted(batch_eval_hashes, topk_hashes).clamp_(
            max=batch_eval_hashes.numel() - 1
        )
        hits = (batch_eval_hashes[pos_idx] == topk_hashes).view(B, max_k).float()
        del topk_hashes, pos_idx

        batch_stats = _compute_metrics_and_topk(
            topk_items=topk_items,
            hits=hits,
            n_pos=eval_counts[b_start:b_end].float(),
            popularity=popularity,
            ks=ks,
            n_items=n_items,
            ranks=ranks,
            disc=disc,
            cum_disc=cum_disc,
            seen_in_topk=seen_in_topk,
        )

        for k in ks:
            for key, val in batch_stats[k].items():
                totals[k][key] += val

    del user_emb, item_emb_t, eval_hashes, train_u, train_v

    if verbose:
        print(
            f"[eval] full-ranking ({U} users, ks={ks}): {time.perf_counter() - t0:.2f}s",
            flush=True,
        )

    metrics: Dict[str, float] = {}
    for k in ks:
        t = totals[k]
        metrics[f"ndcg@{k}"] = t["ndcg"] / U
        metrics[f"mrr@{k}"] = t["mrr"] / U
        metrics[f"map@{k}"] = t["map"] / U
        metrics[f"recall@{k}"] = t["recall"] / U
        metrics[f"coverage@{k}"] = seen_in_topk[k].sum().item() / n_items
        metrics[f"novelty@{k}"] = (
            t["novelty_sum"] / t["novelty_count"] if t["novelty_count"] > 0 else 0.0
        )

    return metrics
