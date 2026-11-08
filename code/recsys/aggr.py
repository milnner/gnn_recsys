from typing import Callable, Union
from torch import Tensor
import torch

AggFn = Union[str, Callable[[Tensor, Tensor, int], Tensor]]


def aggregate_neighbors(
    edge_index: Tensor,
    features: Tensor,
    num_nodes: int | None = None,
    *,
    agg: AggFn = "mean",
    include_self: bool = False,
    chunk_size: int | None = None,
) -> Tensor:
    """h'_i = AGG_{j ∈ N(i)} h_j.

    Agrega as features dos vizinhos ``j`` definidos por ``edge_index`` nas
    features dos nós ``i``.

    Parameters
    ----------
    edge_index : Tensor
        ``(2, E)`` — arestas usadas na agregação (ex.: apenas as de treino).
        Coluna ``(i, j)`` significa "``j`` é vizinho de ``i``" e ``h_j`` é
        agregado em ``h_i``.
    features : Tensor
        ``(N, D)`` — representação dos nós. Precisa ter pelo menos
        ``edge_index[1].max() + 1`` linhas (pode ser ``> num_nodes`` quando
        se agrega de um tipo para outro — ex.: filmes → usuários).
    num_nodes : int, default = features.size(0)
        Tamanho do conjunto de saída (ex.: ``g.num_users`` ao agregar
        ``movie_features`` em usuários).
    agg : {"mean", "sum", "max", "min"} ou Callable
        - str: agregação padrão.
        - Callable ``(msg, index, dim_size) -> Tensor``: permite definir
          agregação customizada (attention, weighted, etc.). Nesse caso
          o controle de memória é responsabilidade do chamador.
    include_self : bool, default False
        Se True, adiciona self-loops antes de agregar. Exige
        ``features.size(0) >= num_nodes``.
    chunk_size : int, default adaptativo
        Número de arestas por chunk. ``None`` escolhe ~512 MiB de mensagens
        por chunk, dado ``D`` e o dtype de ``features``.

    Returns
    -------
    Tensor
        ``(num_nodes, D)`` — features agregadas. Nós sem vizinhos ficam
        zerados (após máscara explícita, não por `clamp`).
    """
    if num_nodes is None:
        num_nodes = features.size(0)

    src, dst = edge_index[0], edge_index[1]

    if include_self:
        if features.size(0) < num_nodes:
            raise ValueError(
                f"include_self=True requer features.size(0) >= num_nodes, "
                f"obtido {features.size(0)} < {num_nodes}"
            )
        self_idx = torch.arange(num_nodes, device=src.device, dtype=src.dtype)
        edge_index = torch.cat(
            [edge_index, self_idx.unsqueeze(0).expand(2, -1)], dim=1
        )
        src, dst = edge_index[0], edge_index[1]

    E = src.size(0)
    D = features.size(-1)
    device = features.device
    dtype = features.dtype

    if callable(agg):
        return agg(features[dst], src, num_nodes)

    if chunk_size is None:
        target_bytes = 512 << 20  # ~512 MiB por chunk
        bytes_per_edge = max(1, D * features.element_size())
        chunk_size = max(1, target_bytes // bytes_per_edge)

    if agg == "sum":
        out = torch.zeros(num_nodes, D, dtype=dtype, device=device)
        for s in range(0, E, chunk_size):
            e = min(s + chunk_size, E)
            out.index_add_(0, src[s:e], features[dst[s:e]])
        return out

    if agg == "mean":
        out = torch.zeros(num_nodes, D, dtype=dtype, device=device)
        cnt = torch.zeros(num_nodes, 1, dtype=dtype, device=device)
        for s in range(0, E, chunk_size):
            e = min(s + chunk_size, E)
            out.index_add_(0, src[s:e], features[dst[s:e]])
            cnt.index_add_(
                0, src[s:e],
                torch.ones(e - s, 1, dtype=dtype, device=device),
            )
        return out / cnt.clamp_min(1)

    if agg in ("max", "min"):
        reduce = "amax" if agg == "max" else "amin"
        init = float("-inf") if agg == "max" else float("inf")
        out = torch.full((num_nodes, D), init, dtype=dtype, device=device)
        touched = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        for s in range(0, E, chunk_size):
            e = min(s + chunk_size, E)
            chunk_src = src[s:e]
            idx = chunk_src.unsqueeze(-1).expand(-1, D)
            out.scatter_reduce_(
                0, idx, features[dst[s:e]], reduce=reduce, include_self=True,
            )
            touched[chunk_src] = True
        return torch.where(touched.unsqueeze(-1), out, torch.zeros_like(out))

    raise ValueError(
        f"agg desconhecida: {agg!r}; use 'sum'|'mean'|'max'|'min'|Callable"
    )