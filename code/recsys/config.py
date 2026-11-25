from dataclasses import dataclass
from typing import Literal, Tuple


@dataclass
class DataConfig:
    dataset: str
    root: str
    leave_k: int
    load_edge_ts: bool
    feature_dim: int | None


@dataclass
class SamplerConfig:
    ratio: int
    negative_sampling: Literal["perm", "mixed", "uniform"]
    perm_ratio: float | None  # sempre float; int vira fração na serialização
    filter_false_negatives: bool | None  # [TODO] implementar a máscara


@dataclass
class ModelConfig:
    name: str
    hidden_channels: int
    num_layers: int
    dropout: float
    hetero_aggr: str  # agregação das convoluções
    conv_aggr: str  # agregação da convolução
    activation: str
    normalize_conv: bool  # normalização L2 a cada convolução, para aquelas como sage.
    normalize_output: bool  # normalização L2 o output do modelo.
    input_proj: bool  # fazer projeção da camada-0 para a dimenção de `hidden_channels`


@dataclass
class OptimizerConfig:
    name: Literal["adam", "adamw", "sgd"]
    lr: float
    weight_decay: float
    betas: Tuple[float, float]
    grad_clip: float | None


@dataclass
class TrainConfig:
    batch_size: int
    num_epochs: int
    l2_reg: float
    bpr_reduction: Literal["mean_triples", "mean_positives"]
    num_workers: int
    seed: int
    score_scale: int


@dataclass
class EvalConfig:
    ks: int  # Para avaliação de rank
    patience_k: int  # k referência para early-stop deve estar em ks
    eval_mode: Literal["full_graph", "loader"] | None
    num_neighbors: list[int]
    eval_every: int
    user_batch_size: int | None = None
    num_negatives: int | None = None
    eval_batch_size: int | None = None


@dataclass
class EarlyStopConfig:
    patience: int
    patience_metric: str  # "ndcg@{k}"
    patience_mode: Literal["max", "min"]
    min_delta: float


@dataclass
class Config:
    data: DataConfig
    sampler: SamplerConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    train: TrainConfig
    eval: EvalConfig
    early_stop: EarlyStopConfig
    log_every: int

    def __post_init__(self):
        # Validações que hoje explodem no meio do treino
        if self.sampler.negative_sampling == "perm" and self.sampler.perm_ratio is None:
            raise ValueError("negative_sampling='perm' exige perm_ratio")
        if self.sampler.perm_ratio is not None and not (
            0.0 <= float(self.sampler.perm_ratio) <= 1.0
        ):
            raise ValueError("perm_ratio deve estar em [0, 1] como fração")
        if self.model.num_layers < 1:
            raise ValueError("num_layers deve ser >= 1")
        if self.sampler.ratio == 0:
            raise ValueError("ratio=0 com sampler não-uniforme não faz sentido")
        # patience_metric consistente com k
        if self.eval.patience_k not in self.eval.ks:
            raise ValueError(
                f"patience_k={self.eval.patience_k} not in ks={self.eval.ks}"
            )
        expected = self.early_stop.patience_metric.format(k=self.eval.patience_k)
        if "{" in expected:  # sobrou placeholder
            raise ValueError(
                f"patience_metric inválido: {self.early_stop.patience_metric!r}"
            )

    def to_dict(self) -> dict:
        """JSON-serializable."""
        from dataclasses import asdict

        return asdict(self)
