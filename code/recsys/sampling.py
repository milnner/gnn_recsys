from __future__ import annotations

import torch
from torch import Tensor


def _num_dst_nodes_in_batch(batch, edge_type: tuple[str, str, str]) -> int:
    _, _, dst_type = edge_type
    return int(batch[dst_type].num_nodes)


# [TODO] Remover o retorno de labels e passar a retornar uma máscara de falsos negativos
def _finalize_in_batch(pos_edge_index: Tensor, neg_dst: Tensor, ratio: int):
    # device = pos_edge_index.device
    # num_pos = pos_edge_index.size(1)

    neg_users = pos_edge_index[0].repeat_interleave(ratio)
    neg_edge_index = torch.stack([neg_users, neg_dst], dim=0)
    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

    # labels = torch.cat([
    #     torch.ones(num_pos, device=device, dtype=torch.float32),
    #     torch.zeros(num_pos * ratio, device=device, dtype=torch.float32),
    # ])
    return edge_label_index, None


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


def sample_negatives_permutation(batch, edge_type, ratio: int, rgn=None):
    pos = batch[edge_type].edge_label_index
    if ratio == 0:
        return _finalize_in_batch(
            pos, torch.empty(0, dtype=pos.dtype, device=pos.device), 0
        )

    negs = _perm_negatives(pos[1], ratio, rgn).reshape(-1)
    return _finalize_in_batch(pos, negs, ratio)


def sample_negatives_uniform(batch, edge_type, ratio: int, rgn=None):
    pos = batch[edge_type].edge_label_index
    num_pos = pos.size(1)
    if ratio == 0:
        return _finalize_in_batch(
            pos, torch.empty(0, dtype=pos.dtype, device=pos.device), 0
        )

    M = _num_dst_nodes_in_batch(batch, edge_type)
    negs = _uniform_negatives(M, (num_pos * ratio,), pos.device, rgn)
    return _finalize_in_batch(pos, negs, ratio)


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
    return _finalize_in_batch(pos, negs, ratio)


NEGATIVE_SAMPLING = {
    "perm": sample_negatives_permutation,
    "uniform": sample_negatives_uniform,
    "mixed": sample_negatives_mixed,
}
