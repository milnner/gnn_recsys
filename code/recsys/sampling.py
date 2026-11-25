from __future__ import annotations

import torch
from torch import Tensor


def _num_dst_nodes_in_batch(batch, edge_type: tuple[str, str, str]) -> int:
    _, _, dst_type = edge_type
    return int(batch[dst_type].num_nodes)


def build_global_pos_keys(
    edge_index: Tensor, num_items: int, device: torch.device
) -> Tensor:
    """Pré-computa uma única vez antes do treino o vetor ordenado de chaves (u, v)."""
    u = edge_index[0].to(device=device, dtype=torch.int64)
    v = edge_index[1].to(device=device, dtype=torch.int64)
    keys = u * num_items + v
    return torch.unique(keys, sorted=True)


def _isin_sorted(query: Tensor, sorted_keys: Tensor) -> Tensor:
    """Retorna máscara booleana indicando quais elementos de `query` estão em `sorted_keys`."""
    if sorted_keys.numel() == 0 or query.numel() == 0:
        return torch.zeros_like(query, dtype=torch.bool)
    idx = torch.searchsorted(sorted_keys, query)
    idx_clamped = idx.clamp(max=sorted_keys.numel() - 1)
    return sorted_keys[idx_clamped] == query


def _compute_fn_mask(
    batch,
    edge_type: tuple[str, str, str],
    neg_users: Tensor,
    neg_dst: Tensor,
    num_pos: int,
    ratio: int,
    global_pos_keys: Tensor | None = None,
    num_total_items: int | None = None,
) -> Tensor:
    """Retorna máscara booleana [num_pos, ratio] onde True = falso negativo."""
    src_type, _, dst_type = edge_type

    if global_pos_keys is not None:
        if num_total_items is None:
            raise ValueError(
                "num_total_items é obrigatório quando global_pos_keys é usado."
            )
        # Mapeia IDs locais do batch -> IDs globais
        g_u = batch[src_type].n_id[neg_users].to(torch.int64)
        g_v = batch[dst_type].n_id[neg_dst].to(torch.int64)
        query_keys = g_u * num_total_items + g_v
        fn_flat = _isin_sorted(query_keys, global_pos_keys)
    else:
        # Fallback: checa contra todas as arestas positivas presentes no batch
        M = _num_dst_nodes_in_batch(batch, edge_type)
        pos_label = batch[edge_type].edge_label_index
        if (
            hasattr(batch[edge_type], "edge_index")
            and batch[edge_type].edge_index.numel() > 0
        ):
            known = torch.cat([batch[edge_type].edge_index, pos_label], dim=1)
        else:
            known = pos_label

        known_keys = known[0].to(torch.int64) * M + known[1].to(torch.int64)
        known_keys, _ = torch.sort(known_keys)
        query_keys = neg_users.to(torch.int64) * M + neg_dst.to(torch.int64)
        fn_flat = _isin_sorted(query_keys, known_keys)

    return fn_flat.view(num_pos, ratio)


def _finalize_in_batch(
    batch,
    edge_type: tuple[str, str, str],
    pos_edge_index: Tensor,
    neg_dst: Tensor,
    ratio: int,
    filter_false_negatives: bool = False,
    global_pos_keys: Tensor | None = None,
    num_total_items: int | None = None,
):
    num_pos = pos_edge_index.size(1)
    neg_users = pos_edge_index[0].repeat_interleave(ratio)
    neg_edge_index = torch.stack([neg_users, neg_dst], dim=0)
    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

    fn_mask = None
    if filter_false_negatives and ratio > 0 and num_pos > 0:
        fn_mask = _compute_fn_mask(
            batch=batch,
            edge_type=edge_type,
            neg_users=neg_users,
            neg_dst=neg_dst,
            num_pos=num_pos,
            ratio=ratio,
            global_pos_keys=global_pos_keys,
            num_total_items=num_total_items,
        )

    return edge_label_index, fn_mask


# # [TODO] Adicionar o retorno de uma máscara de falsos negativos
# def _finalize_in_batch(pos_edge_index: Tensor, neg_dst: Tensor, ratio: int):
#     # device = pos_edge_index.device
#     # num_pos = pos_edge_index.size(1)

#     neg_users = pos_edge_index[0].repeat_interleave(ratio)
#     neg_edge_index = torch.stack([neg_users, neg_dst], dim=0)
#     edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

#     return edge_label_index, None


def _perm_negatives(
    pos_dst: Tensor, num_cols: int, rgn: torch.Generator | None
) -> Tensor:
    """Retorna matriz [num_pos, num_cols] de negativos permutados na GPU."""
    num_pos = pos_dst.size(0)
    device = pos_dst.device
    gen = rgn if (rgn is not None and rgn.device == device) else None

    # Se quiser permutação exata por coluna:
    cols = [
        pos_dst[torch.randperm(num_pos, device=device, generator=gen)]
        for _ in range(num_cols)
    ]
    return torch.stack(cols, dim=1)


def _uniform_negatives(
    num_items_in_batch: int,
    shape: tuple[int, ...],
    device: torch.device,
    rgn: torch.Generator | None,
) -> Tensor:
    gen = rgn if (rgn is not None and rgn.device == device) else None
    return torch.randint(0, num_items_in_batch, shape, device=device, generator=gen)


def sample_negatives_permutation(
    batch,
    edge_type,
    ratio: int,
    rgn=None,
    filter_false_negatives: bool = False,
    global_pos_keys: Tensor | None = None,
    num_total_items: int | None = None,
):
    pos = batch[edge_type].edge_label_index
    if ratio == 0:
        return _finalize_in_batch(
            pos, torch.empty(0, dtype=pos.dtype, device=pos.device), 0
        )

    negs = _perm_negatives(pos[1], ratio, rgn).reshape(-1)
    return _finalize_in_batch(
        batch,
        edge_type,
        pos,
        negs,
        ratio,
        filter_false_negatives=filter_false_negatives,
        global_pos_keys=global_pos_keys,
        num_total_items=num_total_items,
    )


def sample_negatives_uniform(
    batch,
    edge_type,
    ratio: int,
    rgn=None,
    filter_false_negatives: bool = False,
    global_pos_keys: Tensor | None = None,
    num_total_items: int | None = None,
):
    pos = batch[edge_type].edge_label_index
    num_pos = pos.size(1)
    if ratio == 0:
        return _finalize_in_batch(
            pos, torch.empty(0, dtype=pos.dtype, device=pos.device), 0
        )

    M = _num_dst_nodes_in_batch(batch, edge_type)
    negs = _uniform_negatives(M, (num_pos * ratio,), pos.device, rgn)
    return _finalize_in_batch(
        batch,
        edge_type,
        pos,
        negs,
        ratio,
        filter_false_negatives=filter_false_negatives,
        global_pos_keys=global_pos_keys,
        num_total_items=num_total_items,
    )


def _resolve_perm_frac(perm_ratio, ratio: int) -> float:
    """Normaliza perm_ratio para uma fração em [0,1]."""
    if perm_ratio is None:
        raise ValueError("perm_ratio não pode ser None")
    # int (não-bool): compatibilidade retroativa -> contagem por positivo
    if isinstance(perm_ratio, int) and not isinstance(perm_ratio, bool):
        if not (0 <= perm_ratio <= ratio):
            raise ValueError(
                f"perm_ratio (int) deve estar em [0, {ratio}], recebido {perm_ratio}"
            )
        return perm_ratio / ratio if ratio > 0 else 0.0
    # float: fração em [0,1]
    f = float(perm_ratio)
    if 0.0 <= f <= 1.0:
        return f
    raise ValueError(
        f"perm_ratio deve ser None, int em [0,{ratio}] (contagem) ou float em [0,1] (fração); "
        f"recebido {perm_ratio!r}"
    )


def _split_perm_counts(
    num_pos: int,
    ratio: int,
    perm_frac: float,
    device: torch.device,
    rgn: torch.Generator | None = None,
) -> Tensor:
    """Retorna p_i [num_pos], cada em [0, ratio], com sum(p_i) = round(num_pos*ratio*perm_frac).

    A escolha de quais positivos recebem permutações é aleatória (usa `rgn`),
    em vez de determinística por índice.
    """
    if num_pos < 2 or ratio == 0 or perm_frac <= 0.0:
        # num_pos < 2: não há "outro positivo" para permutar.
        return torch.zeros(num_pos, dtype=torch.long, device=device)

    total_negs = num_pos * ratio
    total_perm = int(round(total_negs * perm_frac))
    total_perm = max(0, min(total_perm, total_negs))
    if total_perm == 0:
        return torch.zeros(num_pos, dtype=torch.long, device=device)

    # [TODO] Warning
    gen = rgn if (rgn is not None and rgn.device == device) else None

    if total_perm <= num_pos:
        # Cada positivo recebe 0 ou 1; `total_perm` sorteados ganham 1.
        p_i = torch.zeros(num_pos, dtype=torch.long, device=device)
        chosen = torch.randperm(num_pos, device=device, generator=gen)[:total_perm]
        p_i[chosen] = 1
        return p_i

    # total_perm > num_pos: base por positivo + 1 extra para `extra` sorteados.
    base = total_perm // num_pos
    extra = total_perm - base * num_pos
    p_i = torch.full((num_pos,), base, dtype=torch.long, device=device)
    if extra > 0:
        chosen = torch.randperm(num_pos, device=device, generator=gen)[:extra]
        p_i[chosen] += 1
    return p_i


def sample_negatives_mixed(
    batch,
    edge_type,
    ratio: int,
    perm_ratio=None,
    rgn: torch.Generator | None = None,
    filter_false_negatives: bool = False,
    global_pos_keys: Tensor | None = None,
    num_total_items: int | None = None,
):
    pos = batch[edge_type].edge_label_index
    num_pos = pos.size(1)
    if ratio == 0 or num_pos == 0:
        return _finalize_in_batch(
            pos, torch.empty(0, dtype=pos.dtype, device=pos.device), ratio
        )

    perm_frac = _resolve_perm_frac(perm_ratio, ratio)
    M = _num_dst_nodes_in_batch(batch, edge_type)
    device = pos.device

    # Negativos uniformes para todos os slots: [num_pos, ratio]
    unif_negs = _uniform_negatives(M, (num_pos, ratio), device, rgn)

    # Guards host-side (sem sync): sem permutação possível
    if num_pos < 2 or perm_frac <= 0.0:
        return _finalize_in_batch(pos, unif_negs.reshape(-1), ratio)

    p_i = _split_perm_counts(num_pos, ratio, perm_frac, device, rgn)  # [num_pos]
    # Sempre `ratio` colunas: o excedente é mascarado abaixo. Elimina o
    # p_i.max().item() e o branch max_p >= ratio / zero-pad.
    perm_filled = _perm_negatives(pos[1], ratio, rgn)  # [num_pos, ratio]

    ar = torch.arange(ratio, device=device).unsqueeze(0)  # [1, ratio]
    perm_mask = ar < p_i.unsqueeze(1)  # [num_pos, ratio]

    negs = torch.where(perm_mask, perm_filled, unif_negs).reshape(-1)
    return _finalize_in_batch(
        batch,
        edge_type,
        pos,
        negs,
        ratio,
        filter_false_negatives=filter_false_negatives,
        global_pos_keys=global_pos_keys,
        num_total_items=num_total_items,
    )


NEGATIVE_SAMPLING = {
    "perm": sample_negatives_permutation,
    "uniform": sample_negatives_uniform,
    "mixed": sample_negatives_mixed,
}
