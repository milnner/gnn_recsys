import torch

from torch import Tensor
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.data import HeteroData

from recsys.eval import evaluate_full_graph, evaluate_loader
from .config import Config

import numpy as np
import scipy.sparse as sp

def build_train_positive_adj(train_edge_index: Tensor, n_users: int, n_items: int):
  u = train_edge_index[0].detach().cpu().numpy()
  v = train_edge_index[1].detach().cpu().numpy()
  data = np.ones(u.size, dtype=bool)
  return sp.csr_matrix((data, (u, v)), shape=(n_users, n_items))

def is_false_negative(u_sub, v_sub, batch, user_type, item_type, adj):
  """Mapeia índices locais (subgrafo) → IDs globais e consulta a CSR."""
  u_glob = batch[user_type].n_id[u_sub].cpu().numpy()
  v_glob = batch[item_type].n_id[v_sub].cpu().numpy()
  out = adj[u_glob, v_glob]
  if hasattr(out, "toarray"):
    out = out.toarray()
  return torch.from_numpy(np.asarray(out).flatten()).bool().to(u_sub.device)


def _bpr_per_triple(
  pos_scores, neg_scores, valid_mask=None,
  reduction: str = "mean_triples",
):
  """pos_scores: [P] | neg_scores: [P, R] | valid_mask: [P, R]"""
  diff = pos_scores.unsqueeze(1) - neg_scores                 # [P, R]
  loss_per = -F.logsigmoid(diff)

  if valid_mask is None:
    return loss_per.mean()

  loss_per = loss_per * valid_mask
  if reduction == "mean_triples":
    return loss_per.sum() / valid_mask.sum().clamp(min=1.0)

  # mean_positives: cada positivo pesa igual, independente de
  # quantos negativos válidos sobraram após o filtro
  n_val = valid_mask.sum(-1)
  per_pos = loss_per.sum(-1) / n_val.clamp(min=1.0)
  valid_pos = n_val > 0
  return per_pos[valid_pos].mean() if valid_pos.any() else loss_per.sum() * 0.0


def train(
  train_loader: LinkNeighborLoader,
  val_loader: LinkNeighborLoader,
  model: torch.nn.Module,
  opt: torch.optim.Optimizer,
  device: torch.device,
  edge: tuple[str, str, str],
  train_edge_index: torch.Tensor,
  cfg: Config,
  # necessários apenas quando cfg.eval_mode == "full_graph"
  full_data: HeteroData | None = None,
  eval_edge_index: torch.Tensor | None = None,
):
    user_type, _, item_type = edge

    # ---------------- Inferência de n_users/n_items ----------------
    if cfg.filter_false_negatives:
      n_users = (cfg.n_users if cfg.n_users is not None
                  else int(train_edge_index[0].max().item()) + 1)
      n_items = (cfg.n_items if cfg.n_items is not None
                  else int(train_edge_index[1].max().item()) + 1)
      pos_adj = build_train_positive_adj(train_edge_index, n_users, n_items)
    else:
      n_items = cfg.n_items or int(train_edge_index[1].max().item()) + 1
      pos_adj = None

    # ---------------- Setup ----------------
    patience_key   = cfg.patience_metric.format(k=cfg.k)
    from .sampling import NEGATIVE_SAMPLING
    sampling       = NEGATIVE_SAMPLING[cfg.negative_sampling]
    model          = model.to(device)

    best_metric = -float("inf") if cfg.patience_mode == "max" else float("inf")
    best_state, patience_ctr, history = None, 0, []

    for epoch in range(1, cfg.num_epochs + 1):
      model.train()
      total_loss, n_batches = 0.0, 0
      n_neg_total, n_neg_valid = 0, 0

      # ======================= TREINO =======================
      for batch in train_loader:
        batch = batch.to(device)
        opt.zero_grad()

        out = model(batch.x_dict, batch.edge_index_dict)

        if cfg.negative_sampling == "mixed":
          edge_label_index, labels = sampling(
            batch, edge, cfg.ratio, cfg.perm_ratio, cfg.rgn)
        else:
          edge_label_index, labels = sampling(
            batch, edge, cfg.ratio, cfg.rgn)

        u, v = edge_label_index
        scores = (out[user_type][u] * out[item_type][v]).sum(-1)

        num_pos = int((labels > 0.5).sum())
        if num_pos == 0 or num_pos == scores.numel():
          continue

        pos_scores = scores[:num_pos]
        neg_scores = scores[num_pos:].reshape(num_pos, cfg.ratio)

        # máscara de falso-negativo
        if pos_adj is not None and cfg.negative_sampling in ("perm", "mixed"):
          fn_mask = is_false_negative(
            u[num_pos:], v[num_pos:],
            batch, user_type, item_type, pos_adj,
          )
          valid_mask = (~fn_mask).reshape(num_pos, cfg.ratio)
        else:
          valid_mask = torch.ones(
            num_pos, cfg.ratio, dtype=torch.bool, device=device)

        n_neg_total += valid_mask.numel()
        n_neg_valid += int(valid_mask.sum())
        if valid_mask.sum() == 0:
          continue

        loss = _bpr_per_triple(
          pos_scores, neg_scores, valid_mask,
          reduction=cfg.bpr_reduction,
        )
        if cfg.l2_reg > 0:
          loss = loss + cfg.l2_reg * sum(
            t.pow(2).sum() for t in out.values())

        if not torch.isfinite(loss):
          continue

        loss.backward()
        if cfg.grad_clip is not None:
          torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        total_loss += loss.item()
        n_batches  += 1

      avg_loss = total_loss / max(n_batches, 1)
      fn_rate  = 1.0 - (n_neg_valid / max(n_neg_total, 1))

      # ======================= VALIDAÇÃO =======================
      if cfg.eval_mode == "full_graph":
        if full_data is None or eval_edge_index is None:
          raise ValueError(
            "eval_mode='full_graph' exige full_data e eval_edge_index")
        val_metrics = evaluate_full_graph(
          model, full_data, user_type, item_type, edge,
          device=device, k=cfg.k,
          num_negatives=cfg.eval_num_negatives,
          eval_edge_index=eval_edge_index,
          seed=cfg.eval_seed,
        )
      else:
          val_metrics = evaluate_loader(
            model, val_loader, edge, device, k=cfg.k,
            train_edge_index=train_edge_index, n_items=n_items,
          )

      history.append({"epoch": epoch, "loss": avg_loss,
                      "fn_rate": fn_rate, **val_metrics})

      if epoch % cfg.log_every == 0 or epoch == 1:
          fmt = lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
          print(
            f"[Ep {epoch:04d}] loss={avg_loss:.4f} fn={fn_rate:.3f} | "
            + " | ".join(f"{kk}={fmt(vv)}" for kk, vv in val_metrics.items())
          )

      # ======================= EARLY STOP =======================
      if patience_key not in val_metrics:
        raise KeyError(f"patience_metric='{patience_key}' ausente em "
                  f"{list(val_metrics)}")
      cur = val_metrics[patience_key]
      improved = (cur > best_metric + cfg.min_delta if cfg.patience_mode == "max"
                  else cur < best_metric - cfg.min_delta)

      if improved:
        best_metric = cur
        best_state  = {n: t.detach().cpu().clone()
                        for n, t in model.state_dict().items()}
        patience_ctr = 0
      else:
        patience_ctr += 1
        if patience_ctr >= cfg.patience:
          print(f"Early stop ep={epoch} | best {patience_key}={best_metric:.4f}")
          break

    if best_state is not None:
      model.load_state_dict(best_state)

    return {f"best_{patience_key}": best_metric, "history": history}

