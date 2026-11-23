from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from torch_geometric.data import HeteroData

from recsys.datasets import MovielensGraphs
from recsys.splits import per_src_leave_k_edges

EdgeType = Tuple[str, str, str]


@dataclass
class PrepDataset:
    train_idx: torch.Tensor
    test_idx: torch.Tensor
    hetero_data: HeteroData
    edge: EdgeType
    rev_edge: EdgeType
    user_key: str
    item_key: str
    num_users: int
    num_items: int
    in_channels_dict: Dict[str, int]
    has_user_features: bool = False

    @property
    def train_edge_index(self) -> torch.Tensor:
        return self.hetero_data[self.edge].edge_index[:, self.train_idx]

    @property
    def eval_test_edge_index(self) -> torch.Tensor:
        return self.hetero_data[self.edge].edge_index[:, self.test_idx]


def _load_movielens(
    g: Union[MovielensGraphs],
    leave_k: int,
    device: torch.device,
    data_device: torch.device,
    rgn: Optional[torch.Generator] = None,
) -> PrepDataset:

    train_idx, test_idx = per_src_leave_k_edges(
        g.edge_index.to(device),
        g.num_users,
        k=leave_k,
        rgn=rgn,
    )
    train_idx = train_idx.to(data_device)
    test_idx = test_idx.to(data_device)

    hetero_data = g.to_hetero()
    user_key = "user"
    item_key = "movie"
    edge = (user_key, "rates", item_key)
    rev_edge = (item_key, "rated_by", user_key)

    # Recorta features do item para 1024 dimensões
    hetero_data[item_key].x = hetero_data[item_key].x[:, :1024].contiguous()

    item_dim = hetero_data[item_key].x.size(-1)
    has_user_features = (
        user_key in hetero_data.node_types
        and hasattr(hetero_data[user_key], "x")
        and hetero_data[user_key].x is not None
    )
    user_dim = hetero_data[user_key].x.size(-1) if has_user_features else item_dim

    return PrepDataset(
        train_idx=train_idx,
        test_idx=test_idx,
        hetero_data=hetero_data,
        edge=edge,
        rev_edge=rev_edge,
        user_key=user_key,
        item_key=item_key,
        num_users=g.num_users,
        num_items=hetero_data[item_key].num_nodes,
        in_channels_dict={user_key: user_dim, item_key: item_dim},
        has_user_features=has_user_features,
    )


def get_dataset(
    dataset: str,
    root: Union[str, Path],
    leave_k: int,
    device: torch.device,
    data_device: torch.device = torch.device("cpu"),
    rgn: Optional[torch.Generator] = None,
    load_edge_ts: bool = False,
) -> PrepDataset:
    name = dataset.lower().strip()

    if name in MovielensGraphs.get_valid_datasets():
        graph = MovielensGraphs(
            root=root,
            dataset=dataset,
            device=str(data_device),
            load_edge_ts=load_edge_ts,
        )
        return _load_movielens(
            graph,
            leave_k=leave_k,
            device=device,
            data_device=data_device,
            rgn=rgn,
        )

    raise ValueError(
        f"Dataset desconhecido: '{dataset}', datasets válidos: {', '.join(MovielensGraphs.get_valid_datasets())}."
    )
