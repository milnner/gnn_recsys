import argparse
import logging
from pathlib import Path

import torch
from torch_geometric.loader import LinkNeighborLoader

from recsys.aggr import aggregate_neighbors
from recsys.config_io import load_config
from recsys.data import rebuild_hetero
from recsys.eval import evaluate_full_ranking
from recsys.factories import build_model, build_optimizer
from recsys.prep_dataset import get_dataset
from recsys.runtime import build_runtime
from recsys.train import train

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treinamento GNN RecSys")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Caminho para o arquivo JSON de configuração.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Diretório base onde a pasta da execução será criada.",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="disabled",
        choices=["online", "offline", "disabled"],
        help="Modo de execução do Weights & Biases.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = parse_args()
    cfg = load_config(args.config)

    resume_ckpt = None
    resume_run_dir = None
    resume_wandb_id = None

    if args.resume is not None:
        ckpt_path = args.resume / "last.pt" if args.resume.is_dir() else args.resume
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")
        resume_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        resume_run_dir = ckpt_path.parent
        resume_wandb_id = resume_ckpt.get("wandb_run_id")

    runtime = build_runtime(
        cfg,
        args.output_dir,
        wandb_mode=args.wandb_mode,
        run_dir=resume_run_dir,
        wandb_run_id=resume_wandb_id,
    )

    logger.info("Artefatos de execução serão salvos em: %s", args.output_dir)
    try:
        dataset = get_dataset(
            cfg.data.dataset,
            cfg.data.root,
            cfg.data.leave_k,
            runtime.device,
            runtime.data_device,
            runtime.rgn,
            cfg.data.load_edge_ts,
        )

        user_type, _, item_type = dataset.edge
        logger.info("=== Treino final em train_idx completo ===")

        # 1) Reconstrói grafo na CPU (data_device) usando train_idx
        final_data = rebuild_hetero(
            dataset.hetero_data, dataset.train_idx, dataset.edge, dataset.rev_edge
        )

        # 2) Agrega features de user na GPU e move de volta para data_device
        if not dataset.has_user_features:
            final_data[user_type].x = aggregate_neighbors(
                final_data[dataset.edge].edge_index.to(runtime.device),
                final_data[item_type].x.to(runtime.device),
                num_nodes=dataset.num_users,
            ).to(runtime.data_device)

        if runtime.device.type == "cuda":
            torch.cuda.empty_cache()

        # 3) Modelo e otimizador via factories
        model = build_model(
            cfg.model,
            metadata=final_data.metadata(),
            in_channels_dict=dataset.in_channels_dict,
        ).to(runtime.device)
        opt = build_optimizer(cfg.optimizer, model.parameters())

        # 4) Loader na CPU com pin_memory se CUDA estiver ativo
        use_cuda = runtime.device.type == "cuda"
        num_workers = cfg.train.num_workers if use_cuda else 0

        train_loader = LinkNeighborLoader(
            final_data,
            num_neighbors=cfg.eval.num_neighbors,
            batch_size=cfg.train.batch_size,
            pin_memory=use_cuda,
            edge_label_index=(dataset.edge, final_data[dataset.edge].edge_index),
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=use_cuda and num_workers > 0,
        )

        # 5) Treino
        train(
            train_loader=train_loader,
            model=model,
            opt=opt,
            edge=dataset.edge,
            cfg=cfg,
            runtime=runtime,
            full_data=final_data,
            eval_edge_index=dataset.eval_test_edge_index,
            resume_ckpt=resume_ckpt,
        )

    finally:
        if runtime.wandb_run is not None:
            runtime.wandb_run.finish()


if __name__ == "__main__":
    main()
