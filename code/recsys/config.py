# from dataclasses import dataclass, field
# from typing import Literal
# import argparse
# import torch


# @dataclass
# class Config:
#   # dataset
#   dataset : str
#   leave_k: int

#   # amostragem
#   ratio: int = 1                 # Variar
#   rgn: torch.Generator | None = None
#   negative_sampling: Literal["perm", "mixed", "uniform", None] = "uniform" # Variar
#   perm_ratio: int | float | None = None   # Variar
#   filter_false_negatives: bool = True # Variar

#   # treino
#   num_epochs: int = 10_000
#   l2_reg: float = 0.0
#   grad_clip: float | None = 5.0
#   min_delta: float = 1e-4
#   log_every: int = 1
#   bpr_reduction: Literal["mean_triples", "mean_positives"] = "mean_positives" # Variar
#   lr: float = 1e-3
#   batch_size : int = 2000

#   # early stopping
#   patience: int = 10
#   patience_metric: str = "ndcg@{k}"
#   patience_mode: Literal["max", "min"] = "max"

#   # avaliação / loader
#   k: int = 10
#   eval_mode: Literal["loader", "full_graph"] = "full_graph"
#   num_neighbors: list[int] | None = None
#   eval_num_negatives: int = 100
#   seed: int = 42
#   eval_batch_size: int = 4096

#   # grafo
#   n_users: int | None = None
#   n_items: int | None = None


# def str_to_bool(v: str) -> bool:
#     if v.lower() in ("true", "t", "1", "yes", "y"):
#         return True
#     elif v.lower() in ("false", "f", "0", "no", "n"):
#         return False
#     raise argparse.ArgumentTypeError(f"Valor booleano esperado, recebido: '{v}'")


# def optional_int(v: str) -> int | None:
#     if v.lower() == "none":
#         return None
#     return int(v)

# def optional_int_or_float(v: str) -> int | None:
#     if v.lower() == "none":
#         return None
#     if v.lower().find(".") == -1:
#         return int(v)
#     return float(v)


# def optional_float(v: str) -> float | None:
#     if v.lower() == "none":
#         return None
#     return float(v)


# def optional_negative_sampling(v: str) -> str | None:
#     if v.lower() == "none":
#         return None
#     if v in ("perm", "mixed", "uniform"):
#         return v
#     raise argparse.ArgumentTypeError(
#         "Escolha entre 'perm', 'mixed', 'uniform' ou 'none'"
#     )


# def optional_int_list(v: list[str]) -> list[int] | None:
#     if len(v) == 1 and v[0].lower() == "none":
#         return None
#     return [int(x) for x in v]


# def get_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(description=" Argumentos de Configuração")

#     # Amostragem
#     parser.add_argument("--ratio", type=int, required=True)
#     parser.add_argument(
#         "--negative_sampling",
#         type=optional_negative_sampling,
#         required=True,
#         help="Opções: 'perm', 'mixed', 'uniform' ou 'none'",
#     )
#     parser.add_argument(
#         "--perm_ratio",
#         type=optional_int_or_float,
#         required=True,
#         help="Inteiro ou 'none'",
#     )
#     parser.add_argument(
#         "--filter_false_negatives",
#         type=str_to_bool,
#         required=True,
#         help="Booleano: 'true' ou 'false'",
#     )

#     # Treino
#     parser.add_argument("--batch_size", type=int, required=True)

#     parser.add_argument("--num_epochs", type=int, required=True)
#     parser.add_argument("--l2_reg", type=float, required=True)
#     parser.add_argument(
#         "--grad_clip",
#         type=optional_float,
#         required=True,
#         help="Float ou 'none'",
#     )
#     parser.add_argument("--min_delta", type=float, required=True)
#     parser.add_argument("--log_every", type=int, required=True)
#     parser.add_argument(
#         "--bpr_reduction",
#         type=str,
#         choices=["mean_triples", "mean_positives"],
#         required=True,
#     )
#     parser.add_argument("--lr", type=float, required=True)

#     # Early stopping
#     parser.add_argument("--patience", type=int, required=True)
#     parser.add_argument("--patience_metric", type=str, required=True)
#     parser.add_argument(
#         "--patience_mode",
#         type=str,
#         choices=["max", "min"],
#         required=True,
#     )

#     # Avaliação / Loader
#     parser.add_argument("--k", type=int, required=True)
#     parser.add_argument(
#         "--eval_mode",
#         type=str,
#         choices=["loader", "full_graph"],
#         required=True,
#     )
#     parser.add_argument(
#         "--num_neighbors",
#         nargs="+",
#         required=True,
#         help="Lista de inteiros separados por espaço (ex: 10 5) ou 'none'",
#     )

#     parser.add_argument("--eval_num_negatives", type=int, required=True)
#     parser.add_argument("--seed", type=int, required=True)
#     parser.add_argument("--eval_batch_size", type=int, required=True)

#     # Dataset
#     parser.add_argument("--dataset", type=str, required=True)
#     parser.add_argument("--leave_k", type=int, required=True)
#     return parser

# def parse_args() -> argparse.Namespace:
#     parser = get_parser()
#     args = parser.parse_args()

#     # Pós-processamento para listas opcionais e o torch.Generator
#     args.num_neighbors = optional_int_list(args.num_neighbors)


#     return args

# def args_to_config(args: argparse.Namespace) -> Config:
#     return Config(**vars(args))

# def build_config(args: argparse.Namespace) -> Config:
#     if args.seed is not None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         args.rgn = torch.Generator(device).manual_seed(args.seed)
#     else:
#         args.rgn = None

#     return Config(
#         dataset=args.dataset,
#         leave_k=args.leave_k,
#         # Amostragem
#         batch_size=args.batch_size,
#         ratio=args.ratio,
#         rgn=args.rgn,
#         negative_sampling=args.negative_sampling,
#         perm_ratio=args.perm_ratio,
#         filter_false_negatives=args.filter_false_negatives,

#         # Treino
#         num_epochs=args.num_epochs,
#         l2_reg=args.l2_reg,
#         grad_clip=args.grad_clip,
#         min_delta=args.min_delta,
#         log_every=args.log_every,
#         bpr_reduction=args.bpr_reduction,
#         lr=args.lr,

#         # Early stopping
#         patience=args.patience,
#         patience_metric=args.patience_metric,
#         patience_mode=args.patience_mode,

#         # Avaliação / Loader
#         k=args.k,
#         eval_mode=args.eval_mode,
#         num_neighbors=args.num_neighbors,
#         eval_num_negatives=args.eval_num_negatives,
#         seed=args.seed,
#         eval_batch_size=args.eval_batch_size,
#     )

from dataclasses import dataclass, field
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
    # [TODO] reimplementar a lógia de bpr_reduction para quando experimentar a remoção dos falso negativos.
    # bpr_reduction: Literal["mean_triples", "mean_positives"] = "mean_positives"
    num_workers: int
    seed: int


@dataclass
class EvalConfig:
    k: int  # Para avaliação de rank
    eval_mode: Literal["full_graph", "loader"]
    num_neighbors: list[int]
    num_negatives: int
    eval_batch_size: int
    eval_every: int  # separado de log_every para ml32M 1 é um valor que não é caro


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
        expected = self.early_stop.patience_metric.format(k=self.eval.k)
        if "{" in expected:  # sobrou placeholder
            raise ValueError(
                f"patience_metric inválido: {self.early_stop.patience_metric!r}"
            )

    def to_dict(self) -> dict:
        """JSON-serializable."""
        from dataclasses import asdict

        return asdict(self)
