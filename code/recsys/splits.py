import torch
from torch import Tensor


def random_split(num_edges : int, *, seed: int,
        test_frac: float):
    """
    
    """
    E = num_edges
    perm = torch.randperm(E, generator=torch.Generator().manual_seed(seed),
        dtype=torch.long)
    n_test  = int(E * test_frac)
    n_train = E - n_test     
    return perm[:n_train], perm[n_train:]


def per_src_leave_k_edges(
    edge_index: torch.Tensor,
    num_src: int,
    *,
    k: int = 1,
    seed: int,
):
    """Leave k arestas aleatórias por source (``edge_index[0]``) para teste.

    As ``k`` arestas retidas por source vão **todas** para ``test_idx``.
    Não há ``val_idx``: quem chamar decide se/quando extrair validação de
    ``train_idx`` (ver notas na docstring da resposta).

    Complexidade ``O(E log E)`` dominada pelos dois ``argsort``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Tensor ``(2, E)`` com as arestas ``(src, dst)`` do grafo bipartido.
    num_src : int
        Quantidade de nós do lado ``src``.
    k : int, default=1
        Número de arestas de teste retidas por source. Sources com
        ``count <= k`` são ignorados (garante ao menos 1 aresta de treino).
    seed : int
        Semente do ``torch.Generator``.

    Returns
    -------
    TrainTestSplit
        ``train_idx`` (arestas não retidas) e ``test_idx`` (agrupado por
        source, ordem ``source0, source1, ...``).
    """
    src = edge_index[0]
    num_edges = edge_index.size(1)
    device = src.device
    gen = torch.Generator(device).manual_seed(seed)

    empty = torch.empty(0, dtype=torch.long, device=device)

    if num_edges == 0:
        return empty, empty

    # --- 1. Prioridade aleatória por aresta --------------------------------
    rand = torch.rand(num_edges, generator=gen, device=device)

    # --- 2. Ordena por (src, rand) via dois sorts estáveis -----------------
    by_rand = torch.argsort(rand)
    order = by_rand[torch.argsort(src[by_rand], stable=True)]

    # --- 3. Rank 0-indexed de cada aresta dentro do seu source -------------
    counts = torch.bincount(src, minlength=num_src)
    offsets = torch.cat([
        torch.zeros(1, dtype=counts.dtype, device=device),
        counts.cumsum(0),
    ])
    order_src = src[order]
    rank = torch.arange(num_edges, device=device) - offsets[order_src]

    # --- 4. Mantém as k arestas de menor rand por source elegível ----------
    # Elegível <-> count > k, garantindo ao menos 1 aresta de treino.
    eligible = counts > k
    keep = (rank < k) & eligible[order_src]
    test_idx = order[keep]

    # --- 5. Treino = complemento -------------------------------------------
    mask = torch.ones(num_edges, dtype=torch.bool, device=device)
    mask[test_idx] = False
    train_idx = mask.nonzero(as_tuple=True)[0]

    return train_idx, test_idx

def per_src_k_fold(
    edge_index: torch.Tensor,
    num_src: int,
    *,
    n_folds: int = 5,
    seed: int,
):
    """Atribui fold_id ∈ {0..n_folds-1} a cada aresta, agrupado por source.

    Garante:
      * fold_id = rank % n_folds  → distribuição round-robin balanceada por source;
      * sources com count <= 1 nunca vão para val (ficariam sem aresta de treino)
        → recebem fold_id = -1 (sempre treino).

    Complexidade O(E log E), análoga a ``per_src_leave_k_edges``.

    Returns
    -------
    fold_ids : (E,) long
    folds    : list[(train_idx, val_idx)] de tamanho ``n_folds``,
               ambos em índices **locais** ao ``edge_index`` de entrada.
    """
    src = edge_index[0]
    E   = edge_index.size(1)
    dev = src.device
    gen = torch.Generator(dev).manual_seed(seed)

    empty = torch.empty(0, dtype=torch.long, device=dev)
    if E == 0:
        return empty, [(empty, empty) for _ in range(n_folds)]

    # 1. Prioridade aleatória
    rand = torch.rand(E, generator=gen, device=dev)

    # 2. Ordena por (src, rand) — mesmo padrão do per_src_leave_k_edges
    by_rand = torch.argsort(rand)
    order   = by_rand[torch.argsort(src[by_rand], stable=True)]

    # 3. Rank 0-indexed dentro de cada source
    counts  = torch.bincount(src, minlength=num_src)
    offsets = torch.cat([torch.zeros(1, dtype=counts.dtype, device=dev),
                         counts.cumsum(0)])
    order_src = src[order]
    rank      = torch.arange(E, device=dev) - offsets[order_src]
    
    # 4. Round-robin: fold = rank % n_folds
    fold_local = (rank % n_folds).to(torch.long)

    # 5. Exclui sources com count <= 1 (não podem ser val)
    fold_local[counts[order_src] <= 1] = -1

    # 6. Volta à ordem original
    fold_ids = torch.empty(E, dtype=torch.long, device=dev)
    fold_ids[order] = fold_local

    # 7. Deriva (train, val) como complemento
    all_idx = torch.arange(E, device=dev)
    folds = []
    for f in range(n_folds):
        val_mask  = fold_ids == f
        tr_mask   = ~val_mask
        folds.append((all_idx[tr_mask], all_idx[val_mask]))

    return fold_ids, folds

def temporal_global(num_edges : int, edge_ts : Tensor, *,
                     test_frac=(0.8, 0.1, 0.1)):
    ts = edge_ts
    order = torch.argsort(ts, stable=True)
    E = num_edges
    n_train = int(E * (1. - test_frac))

    return order[:n_train], order[n_train:]
        
