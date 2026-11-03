from abc import ABC, abstractmethod
from torch import Tensor, cat
import torch.nn as nn

class Head(nn.Module, ABC):
    """Recebe embeddings por tipo de nó e um edge_label_index, produz scores."""

    @abstractmethod
    def forward(
        self,
        h_dict: dict[str, Tensor],
        edge_label_index: Tensor,   # [2, B]
        edge_type: tuple[str, str, str],
    ) -> Tensor:
        ...


class DotProduct(Head):
    """Produto interno entre embeddings de origem e destino."""

    def forward(self, h_dict, edge_label_index, edge_type):
        s, _, d = edge_type
        src, dst = edge_label_index
        return (h_dict[s][src] * h_dict[d][dst]).sum(dim=-1)



class MLP(Head):
    """MLP sobre [h_src ‖ h_dst ‖ h_src * h_dst]."""

    def __init__(self, dim: int, *, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_dict, edge_label_index, edge_type):
        s, _, d = edge_type
        src, dst = edge_label_index
        h_s, h_d = h_dict[s][src], h_dict[d][dst]
        x = cat([h_s, h_d, h_s * h_d], dim=-1)
        return self.net(x).squeeze(-1)