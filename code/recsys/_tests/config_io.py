import json
import tempfile
from pathlib import Path

from recsys.config import (
    Config,
    DataConfig,
    EarlyStopConfig,
    EvalConfig,
    ModelConfig,
    OptimizerConfig,
    SamplerConfig,
    TrainConfig,
)
from recsys.config_io import load_config, save_config


def _full_config() -> Config:
    """Config completa — todos os campos, sem defaults (dataclasses exigem)."""
    return Config(
        data=DataConfig(
            dataset="ml-32m",
            root="data/ml-32m",
            leave_k=1,
            feature_dim=1024,
            load_edge_ts=True,
        ),
        sampler=SamplerConfig(
            ratio=1,
            negative_sampling="perm",
            perm_ratio=0.3,
            filter_false_negatives=True,
        ),
        model=ModelConfig(
            name="hetero_sage",
            num_layers=3,
            hidden_channels=128,
            dropout=0.1,
            hetero_aggr="mean",
            conv_aggr="mean",
            activation="relu",
            normalize_conv=True,
            normalize_output=True,
            input_proj=True,
        ),
        optimizer=OptimizerConfig(
            name="adam",
            lr=0.0005,
            betas=(0.85, 0.995),
            weight_decay=1e-4,
            grad_clip=0.00001,
        ),
        train=TrainConfig(
            batch_size=2000,
            num_epochs=10_000,
            l2_reg=0.0,
            # bpr_reduction="mean_positives",
            num_workers=0,
            seed=42,
        ),
        eval=EvalConfig(
            k=10,
            eval_mode="full_graph",
            num_neighbors=[10, 5],
            num_negatives=100,
            eval_batch_size=4096,
            eval_every=1,
        ),
        early_stop=EarlyStopConfig(
            patience=10,
            patience_metric="ndcg@{k}",
            patience_mode="max",
            min_delta=1e-4,
        ),
        log_every=1,
    )


def _full_config_dict() -> dict:
    """Serializa `_full_config()` e retorna o dict — base para mutações."""
    cfg = _full_config()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cfg.json"
        save_config(cfg, p)
        return json.loads(p.read_text(encoding="utf-8"))


def test_roundtrip_preserves_types():
    cfg = _full_config()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cfg.json"
        save_config(cfg, p)
        cfg2 = load_config(p)

    assert cfg == cfg2, "round-trip divergiu"
    assert isinstance(cfg2.optimizer.betas, tuple), "tuple virou list"
    assert cfg2.optimizer.betas == (0.85, 0.995)
    assert isinstance(cfg2.sampler.negative_sampling, str)


def test_literal_validation():
    data = _full_config_dict()
    data["sampler"]["negative_sampling"] = "INVALIDO"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cfg.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        try:
            load_config(p)
        except ValueError as e:
            msg = str(e)
            assert "INVALIDO" in msg and "perm" in msg, msg
        else:
            raise AssertionError("deveria ter falhado em Literal inválido")


def test_rejects_unknown_field():
    data = _full_config_dict()
    data["model"]["camada_extra"] = 99
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cfg.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        try:
            load_config(p)
        except ValueError as e:
            msg = str(e)
            assert "campos desconhecidos" in msg and "camada_extra" in msg, msg
        else:
            raise AssertionError("deveria ter rejeitado campo extra")


if __name__ == "__main__":
    test_roundtrip_preserves_types()
    test_literal_validation()
    test_rejects_unknown_field()
    print("OK: 3 testes passaram")
