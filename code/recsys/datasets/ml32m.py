from pathlib import Path
import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData


class ML32MGraph:
  """Grafo bipartido heterogêno do ML-32M, com adaptadores PyG."""

  def __init__(self, root, device="cpu", edge_dtype=torch.int64,
  feature_dtype=torch.float32, load_edge_ts=False):
    self.root = Path(root)
    self.device = torch.device(device)
    self._edge_dtype = edge_dtype
    self._feature_dtype = feature_dtype
    self._cache: dict[str, Tensor] = {}
    self.load_edge_ts = load_edge_ts

  @property
  def edge_index(self) -> Tensor:
    self._load()
    return self._cache["edge_index"]

  @property
  def edge_ratings(self) -> Tensor:
    self._load()
    return self._cache["edge_ratings"]

  @property
  def edge_ts(self) -> Tensor:
    if not self.load_edge_ts:
      raise RuntimeError("edge_ts não carregado (load_edge_ts=False)")
    self._load()
    return self._cache["edge_ts"]

  @property
  def user_ids(self) -> Tensor:
    self._load()
    return self._cache["user_ids"]

  @property
  def movie_ids(self) -> Tensor:
    self._load()
    return self._cache["movie_ids"]

  @property
  def num_users(self) -> int:
    self._load()
    return int(self._cache["user_ids"].numel())

  @property
  def num_items(self) -> int:
    self._load()
    return int(self._cache["movie_ids"].numel())

  @property
  def num_nodes(self) -> int:
    return self.num_users + self.num_items

  @property
  def num_edges(self) -> int:
    self._load()
    return int(self._cache["edge_index"].shape[1])

  @property
  def movie_features(self) -> Tensor | None:
    self._load()
    if "movie_features" not in self._cache:
      ckpt = torch.load(
          self.root / "processed" / "pt" / "movies_qwen.pt",
          map_location="cpu",
      )
      M_ckpt = ckpt["v_title"].shape[0]
      if M_ckpt != self.num_items:
        raise ValueError(
          f"movies_qwen.pt tem {M_ckpt} filmes, grafo tem {self.num_items}."
)
      
      ckpt_ids = ckpt["movie_ids"].to(dtype=self.movie_ids.dtype, 
                                      device=self.movie_ids.device)
      
      if not torch.equal(ckpt_ids, self.movie_ids):
        raise ValueError("movies_qwen.pt desalinhado com graph.pt")

      V = torch.cat(
        [ckpt["v_title"], ckpt["v_tagline"], ckpt["v_storyline"]], dim=1
      )
      V = torch.nn.functional.normalize(V, p=2, dim=1)
      self._cache["movie_features"] = V.to(self._feature_dtype).to(self.device)
    return self._cache["movie_features"]
  
  def _load(self) -> None:
    if "edge_index" in self._cache:
        return
    g = torch.load(self.root / "processed/pt/graph.pt", map_location="cpu")
    g["edge_index"] = g["edge_index"].to(self._edge_dtype).to(self.device)
    g["user_ids"]   = g["user_ids"].to(self.device)
    g["movie_ids"]  = g["movie_ids"].to(self.device)
    g["edge_ratings"] = g["edge_ratings"].to(self.device)
    if self.load_edge_ts:
      g["edge_ts"] = g["edge_ts"].to(self.device)
    else:
      g.pop("edge_ts", None)
    self._cache.update(g)

  def to(self, device):
    device = torch.device(device)
    self.device = device
    for key in ("edge_index", "user_ids", 
                "movie_ids", "edge_ratings",
                "movie_features", "edge_ts"):
      if key in self._cache:
        self._cache[key] = self._cache[key].to(device)

    return self

  def release_features(self) -> None:
    self._cache.pop("movie_features", None)
    if self.device.type == "cuda":
        torch.cuda.empty_cache()

  def to_data(self, *, include_features: bool = True) -> Data:
    """Homogeneous PyG view: users in [0, U), movies in [U, U+M)."""
    U, M = self.num_users, self.num_items
    ei = self.edge_index
    forward = torch.stack([ei[0], ei[1] + U], dim=0)
    reverse = forward.flip(0)
    edge_index = torch.cat([forward, reverse], dim=1)

    x = None
    if include_features and self.movie_features is not None:
      d = self.movie_features.size(1)
      x = torch.zeros(U + M, d, dtype=self.movie_features.dtype,
                      device=self.movie_features.device)
      x[U:] = self.movie_features
    return Data(edge_index=edge_index, x=x, num_nodes=U + M)
  
  def to_hetero(self, *, include_features: bool = True) -> HeteroData:
    """Heterogeneous PyG view: user/movie node types, both edge directions."""
    data = HeteroData()
    data["user"].num_nodes  = self.num_users
    data["movie"].num_nodes = self.num_items
    if include_features and self.movie_features is not None:
      data["movie"].x = self.movie_features
    data["user", "rates", "movie"].edge_index = self.edge_index
    data["movie", "rated_by", "user"].edge_index = self.edge_index.flip(0)
    return data
    


if __name__ == "__main__":
    g = ML32MGraph(
        root="/home/milnner00/projects/recsys_experiment/data/ml-32m",
        device="cpu",
        load_edge_ts=True,
    )
    assert g.edge_index.shape[0] == 2
    assert g.user_ids.numel() == g.num_users
    assert g.movie_ids.numel() == g.num_items
    assert g.movie_features.shape == (g.num_items, 3072)
    assert g.edge_ts.numel() == g.num_edges
    g.to("cpu")
    g.release_features()
    assert "movie_features" not in g._cache