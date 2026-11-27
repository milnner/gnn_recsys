import logging
import math
from typing import Any
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from tqdm.auto import tqdm

from recsys.config import Config, EarlyStopConfig
from recsys.eval import evaluate_full_ranking
from recsys.runtime import Runtime
from .sampling import NEGATIVE_SAMPLING, build_global_pos_keys

logger = logging.getLogger(__name__)


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    early_stop_counter: int,
    wandb_run_id: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "best_metric": best_metric,
        "early_stop_counter": early_stop_counter,
        "wandb_run_id": wandb_run_id,
        "metrics": metrics or {},
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _bpr_per_triple(
    pos_scores: Tensor,
    neg_scores: Tensor,
    score_scale: float,
    fn_mask: Tensor | None = None,
    reduction: str = "mean_triples",
) -> Tensor:
    diff = (pos_scores.unsqueeze(1) - neg_scores) * score_scale  # [P, R]
    loss_mat = -F.logsigmoid(diff)
    if fn_mask is None:
        return loss_mat.mean()

    valid = ~fn_mask
    if reduction == "mean_positives":
        valid_per_pos = valid.sum(dim=1)  # [P]
        has_valid = valid_per_pos > 0
        per_pos_loss = (loss_mat * valid).sum(dim=1) / valid_per_pos.clamp_min(1)
        return (per_pos_loss * has_valid).sum() / has_valid.sum().clamp_min(1)

    return (loss_mat * valid).sum() / valid.sum().clamp_min(1)


class EarlyStopping:
    """Controla early stopping e mantém o melhor state_dict em CPU."""

    def __init__(self, cfg: EarlyStopConfig, k: int) -> None:
        self.cfg = cfg
        self.metric_key = cfg.patience_metric.format(k=k)
        self.best_metric = -float("inf") if cfg.patience_mode == "max" else float("inf")
        self.best_state: dict[str, Tensor] | None = None
        self.counter = 0

    def step(
        self, metrics: dict[str, float], model: torch.nn.Module
    ) -> tuple[bool, bool]:
        """
        return improved, should_stop
        improved: melhorou nesta época?
        should_stop: paciência esgotada.
        """
        if self.metric_key not in metrics:
            raise KeyError(
                f"patience_metric='{self.metric_key}' ausente em {list(metrics)}"
            )

        cur = metrics[self.metric_key]
        improved = (
            cur > self.best_metric + self.cfg.min_delta
            if self.cfg.patience_mode == "max"
            else cur < self.best_metric - self.cfg.min_delta
        )

        if improved:
            self.best_metric = cur
            self.best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            self.counter = 0
            return True, False

        self.counter += 1
        return improved, self.counter >= self.cfg.patience

    def restore(self, model: torch.nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _train_one_epoch(
    train_loader: LinkNeighborLoader,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    edge: tuple[str, str, str],
    cfg: Config,
    runtime: Runtime,
    epoch: int,
    global_pos_keys: Tensor | None = None,
    num_total_items: int | None = None,
) -> tuple[float, float | None]:
    user_type, _, item_type = edge
    sampler_fn = NEGATIVE_SAMPLING[cfg.sampler.negative_sampling]
    ratio = cfg.sampler.ratio

    model.train()
    total_loss_t = torch.zeros(1, device=runtime.device)
    total_fn_t = torch.zeros(1, device=runtime.device)

    n_batches = 0

    batch_pbar = tqdm(
        train_loader,
        desc=f"  Ep {epoch:04d}",
        unit="batch",
        leave=False,
        total=len(train_loader),
        dynamic_ncols=True,
    )

    for batch in batch_pbar:
        batch = batch.to(runtime.device, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        filter_fn = bool(cfg.sampler.filter_false_negatives)

        if cfg.sampler.negative_sampling == "mixed":
            edge_label_index, fn_mask = sampler_fn(
                batch,
                edge,
                ratio,
                cfg.sampler.perm_ratio,
                runtime.rgn,
                filter_false_negatives=filter_fn,
                global_pos_keys=global_pos_keys,
                num_total_items=num_total_items,
            )
        else:
            edge_label_index, fn_mask = sampler_fn(
                batch,
                edge,
                ratio,
                runtime.rgn,
                filter_false_negatives=filter_fn,
                global_pos_keys=global_pos_keys,
                num_total_items=num_total_items,
            )
        u, v = edge_label_index
        num_pos = u.numel() // (1 + ratio)
        if num_pos == 0:
            continue

        u_pos, v_pos = u[:num_pos], v[:num_pos]
        v_neg = v[num_pos:]

        out = model(batch.x_dict, batch.edge_index_dict)

        u_emb = out[user_type][u_pos]  # [P, D]
        v_pos_emb = out[item_type][v_pos]  # [P, D]
        v_neg_emb = out[item_type][v_neg].view(num_pos, ratio, -1)  # [P, R, D]

        pos_scores = (u_emb * v_pos_emb).sum(dim=-1)  # [P]
        neg_scores = torch.einsum("pd,prd->pr", u_emb, v_neg_emb)  # [P, R]

        loss = _bpr_per_triple(
            pos_scores,
            neg_scores,
            cfg.train.score_scale,
            fn_mask=fn_mask,
            reduction=cfg.train.bpr_reduction,
        )
        if cfg.train.l2_reg > 0:
            reg = (
                u_emb.pow(2).sum()
                + v_pos_emb.pow(2).sum()
                + v_neg_emb.pow(2).sum() / cfg.sampler.ratio
            ) / num_pos
            loss = loss + cfg.train.l2_reg * reg

        loss.backward()
        if cfg.optimizer.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
        opt.step()

        total_loss_t += loss.detach()
        if fn_mask is not None:
            total_fn_t += fn_mask.float().mean()
        n_batches += 1

    avg_loss = total_loss_t.item() / max(n_batches, 1)
    avg_fn_rate = (
        total_fn_t.item() / max(n_batches, 1)
        if cfg.sampler.filter_false_negatives
        else None
    )
    return avg_loss, avg_fn_rate


def train(
    train_loader: LinkNeighborLoader,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    edge: tuple[str, str, str],
    cfg: Config,
    runtime: Runtime,
    full_data: HeteroData | None = None,
    eval_edge_index: Tensor | None = None,
    resume_ckpt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model = model.to(runtime.device)

    early_stopper = EarlyStopping(cfg.early_stop, k=cfg.eval.patience_k)
    history: list[dict[str, Any]] = []
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"])
        opt.load_state_dict(resume_ckpt["optimizer_state_dict"])
        early_stopper.best_metric = resume_ckpt["best_metric"]
        early_stopper.counter = resume_ckpt.get("early_stop_counter", 0)
        start_epoch = resume_ckpt["epoch"] + 1
        logger.info(
            "Retomando treino da época %d (best %s=%.4f, patience=%d/%d)",
            start_epoch,
            early_stopper.metric_key,
            early_stopper.best_metric,
            early_stopper.counter,
            cfg.early_stop.patience,
        )
    best_ckpt_path = runtime.run_dir / "best.pt"
    last_ckpt_path = runtime.run_dir / "last.pt"
    wandb_run_id = runtime.wandb_run.id if runtime.wandb_run is not None else None

    epoch_pbar = tqdm(
        range(1, cfg.train.num_epochs + 1),
        desc="Épocas",
        unit="ep",
        dynamic_ncols=True,
    )

    global_pos_keys = None
    num_total_items = None
    if cfg.sampler.filter_false_negatives and full_data is not None:
        _, _, item_type = edge
        num_total_items = full_data[item_type].num_nodes
        global_pos_keys = build_global_pos_keys(
            full_data[edge].edge_index, num_total_items, runtime.device
        )

    for epoch in epoch_pbar:
        avg_loss, avg_fn_rate = _train_one_epoch(
            train_loader=train_loader,
            model=model,
            opt=opt,
            edge=edge,
            cfg=cfg,
            runtime=runtime,
            epoch=epoch,
            global_pos_keys=global_pos_keys,
            num_total_items=num_total_items,
        )

        if not math.isfinite(avg_loss):
            logger.warning(
                "Loss não finita na época %d: %s. Interrompendo.", epoch, avg_loss
            )
            epoch_pbar.set_postfix(loss=f"{avg_loss:.4f}")
            break

        log_dict: dict[str, Any] = {"epoch": epoch, "loss": avg_loss}
        if avg_fn_rate is not None:
            log_dict["fn_rate"] = avg_fn_rate

        should_eval = (epoch % cfg.eval.eval_every == 0) or (
            epoch == cfg.train.num_epochs
        )
        should_stop = False

        if should_eval:
            if full_data is None or eval_edge_index is None:
                raise ValueError(
                    "eval_mode='full_graph' exige full_data e eval_edge_index"
                )

            val_metrics = evaluate_full_ranking(
                model=model,
                data=full_data,
                edge_type=edge,
                eval_cfg=cfg.eval,
                runtime=runtime,
                eval_edge_index=eval_edge_index,
            )
            log_dict.update(val_metrics)

            metrics_str = " | ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in val_metrics.items()
            )
            logger.info("[Ep %04d] loss=%.4f | %s", epoch, avg_loss, metrics_str)

            ep_postfix = {"loss": f"{avg_loss:.4f}"}
            if early_stopper.metric_key in val_metrics:
                ep_postfix[early_stopper.metric_key] = (
                    f"{val_metrics[early_stopper.metric_key]:.4f}"
                )
            epoch_pbar.set_postfix(ep_postfix)

            improved, should_stop = early_stopper.step(val_metrics, model)
            if improved:
                _save_checkpoint(
                    best_ckpt_path,
                    model=model,
                    opt=opt,
                    epoch=epoch,
                    best_metric=early_stopper.best_metric,
                    metrics=log_dict,
                    early_stop_counter=early_stopper.counter,
                    wandb_run_id=wandb_run_id,
                )
            # Salvemento dos pesos mais recentes.
            _save_checkpoint(
                last_ckpt_path,
                model=model,
                opt=opt,
                epoch=epoch,
                best_metric=early_stopper.best_metric,
                metrics=log_dict,
                early_stop_counter=early_stopper.counter,
                wandb_run_id=wandb_run_id,
            )

        if runtime.wandb_run is not None and (
            epoch % cfg.log_every == 0 or should_eval or should_stop
        ):
            runtime.wandb_run.log(log_dict, commit=True)

        history.append(log_dict)

        if should_stop:
            logger.info(
                "Early stop ep=%d | best %s=%.4f",
                epoch,
                early_stopper.metric_key,
                early_stopper.best_metric,
            )
            break

    epoch_pbar.close()

    if early_stopper.best_state is not None:
        early_stopper.restore(model)
    elif best_ckpt_path.exists():
        best_ckpt = torch.load(
            best_ckpt_path, map_location=runtime.device, weights_only=False
        )
        model.load_state_dict(best_ckpt["model_state_dict"])

    return {
        f"best_{early_stopper.metric_key}": early_stopper.best_metric,
        "history": history,
    }
