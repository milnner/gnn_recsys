import logging
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from tqdm.auto import tqdm

from recsys.config import Config, EarlyStopConfig
from recsys.eval import evaluate_full_ranking
from recsys.runtime import Runtime
from .sampling import NEGATIVE_SAMPLING

logger = logging.getLogger(__name__)


def _bpr_per_triple(pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
    diff = pos_scores.unsqueeze(1) - neg_scores  # [P, R]
    return -F.logsigmoid(diff).mean()


class EarlyStopping:
    """Controla early stopping e mantém o melhor state_dict em CPU."""

    def __init__(self, cfg: EarlyStopConfig, k: int) -> None:
        self.cfg = cfg
        self.metric_key = cfg.patience_metric.format(k=k)
        self.best_metric = -float("inf") if cfg.patience_mode == "max" else float("inf")
        self.best_state: dict[str, Tensor] | None = None
        self.counter = 0

    def step(self, metrics: dict[str, float], model: torch.nn.Module) -> bool:
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
            return False

        self.counter += 1
        return self.counter >= self.cfg.patience

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
) -> float:
    user_type, _, item_type = edge
    sampler_fn = NEGATIVE_SAMPLING[cfg.sampler.negative_sampling]
    ratio = cfg.sampler.ratio

    model.train()
    total_loss_t = torch.zeros(1, device=runtime.device)
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

        if cfg.sampler.negative_sampling == "mixed":
            edge_label_index, _ = sampler_fn(
                batch, edge, ratio, cfg.sampler.perm_ratio, runtime.rgn
            )
        else:
            edge_label_index, _ = sampler_fn(batch, edge, ratio, runtime.rgn)

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

        loss = _bpr_per_triple(pos_scores, neg_scores)
        if cfg.train.l2_reg > 0:
            reg = (
                u_emb.pow(2).sum() + v_pos_emb.pow(2).sum() + v_neg_emb.pow(2).sum()
            ) / num_pos
            loss = loss + cfg.train.l2_reg * reg

        loss.backward()
        if cfg.optimizer.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
        opt.step()

        total_loss_t += loss.detach()
        n_batches += 1

    return total_loss_t.item() / max(n_batches, 1)


def train(
    train_loader: LinkNeighborLoader,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    edge: tuple[str, str, str],
    cfg: Config,
    runtime: Runtime,
    full_data: HeteroData | None = None,
    eval_edge_index: Tensor | None = None,
) -> dict[str, Any]:
    model = model.to(runtime.device)

    early_stopper = EarlyStopping(cfg.early_stop, k=cfg.eval.patience_k)
    history: list[dict[str, Any]] = []

    epoch_pbar = tqdm(
        range(1, cfg.train.num_epochs + 1),
        desc="Épocas",
        unit="ep",
        dynamic_ncols=True,
    )

    for epoch in epoch_pbar:
        avg_loss = _train_one_epoch(
            train_loader=train_loader,
            model=model,
            opt=opt,
            edge=edge,
            cfg=cfg,
            runtime=runtime,
            epoch=epoch,
        )

        if not math.isfinite(avg_loss):
            logger.warning(
                "Loss não finita na época %d: %s. Interrompendo.", epoch, avg_loss
            )
            epoch_pbar.set_postfix(loss=f"{avg_loss:.4f}")
            break

        log_dict: dict[str, Any] = {"epoch": epoch, "loss": avg_loss}
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

            should_stop = early_stopper.step(val_metrics, model)

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
    early_stopper.restore(model)

    return {
        f"best_{early_stopper.metric_key}": early_stopper.best_metric,
        "history": history,
    }
