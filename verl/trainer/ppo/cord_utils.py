"""Fixed distillation recipes and mini-batch CORD gradient projection."""

import torch

from verl.trainer.ppo.core_algos import agg_loss


# The coefficient multiplies the auxiliary loss before gradient accumulation.
CORD_RECIPES = {
    "skillsd": {"coefficient": 0.001, "estimator": "k3", "gate_beta": 0.0},
    "sdar": {"coefficient": 0.01, "estimator": "k1", "gate_beta": 5.0},
}


def compute_cord_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    recipe: str,
    loss_agg_mode: str = "token-mean",
) -> torch.Tensor:
    """Compute the selected base method's distillation loss.

    Skill-SD uses an importance-weighted k3 term with an importance-ratio cap
    of 2. SDAR uses the teacher/student log-probability gap with a detached
    confidence gate. Teacher and rollout probabilities are always detached.
    """
    if recipe not in CORD_RECIPES:
        raise ValueError(f"Unknown CORD recipe: {recipe}")
    teacher_log_probs = teacher_log_probs.detach()
    old_log_probs = old_log_probs.detach()
    ell = student_log_probs - teacher_log_probs
    if recipe == "skillsd":
        base = torch.exp((-ell).clamp(max=20.0)) - 1.0 + ell
        ratio = torch.exp((student_log_probs - old_log_probs).clamp(max=10.0)).clamp(max=2.0)
        per_token_loss = ratio * base
    else:
        gate = torch.sigmoid(5.0 * (teacher_log_probs - student_log_probs)).detach()
        per_token_loss = -ell * gate
    return agg_loss(loss_mat=per_token_loss, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)


def combine_cord_gradients(parameters, main_grads, auxiliary_grads):
    """Combine accumulated gradient shards, protecting the first objective.

    The auxiliary gradients already include the fixed loss coefficient.
    For a negative global inner product, remove their component along the
    first objective: g = g_main + g_aux - <g_main,g_aux>/||g_main||² g_main.
    Otherwise use their sum. Statistics are reduced across FSDP shards once;
    clipping and the optimizer step happen after this function returns.
    """
    if not (len(parameters) == len(main_grads) == len(auxiliary_grads)):
        raise ValueError("Parameter and gradient lists must have matching lengths")
    metrics = {"actor/cord_projection_applied": 0.0, "actor/cord_cosine": 0.0}
    grad = next((g for grads in (main_grads, auxiliary_grads) for g in grads if g is not None), None)
    if grad is None:
        for parameter in parameters:
            parameter.grad = None
        return metrics

    main_sq = torch.zeros((), device=grad.device, dtype=torch.float32)
    aux_sq = torch.zeros_like(main_sq)
    dot = torch.zeros_like(main_sq)
    for main, auxiliary in zip(main_grads, auxiliary_grads):
        main = main.detach().float() if main is not None else None
        auxiliary = auxiliary.detach().float() if auxiliary is not None else None
        if main is not None:
            main_sq = main_sq + torch.sum(main * main)
        if auxiliary is not None:
            aux_sq = aux_sq + torch.sum(auxiliary * auxiliary)
        if main is not None and auxiliary is not None:
            dot = dot + torch.sum(main * auxiliary)
    totals = torch.stack([main_sq, aux_sq, dot])
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
    main_sq, aux_sq, dot = totals.unbind()
    eps = 1e-12
    applied = main_sq.item() > eps and dot.item() < 0.0
    coefficient = dot / (main_sq + eps) if applied else torch.zeros_like(dot)
    for parameter, main, auxiliary in zip(parameters, main_grads, auxiliary_grads):
        combined = main.clone() if main is not None else None
        if auxiliary is not None:
            projected = auxiliary
            if applied and main is not None:
                projected = auxiliary - coefficient.to(device=auxiliary.device, dtype=auxiliary.dtype) * main
            combined = projected.clone() if combined is None else combined + projected
        parameter.grad = combined
    cosine = dot / (main_sq.clamp_min(0).sqrt() * aux_sq.clamp_min(0).sqrt() + eps)
    metrics["actor/cord_projection_applied"] = float(applied)
    metrics["actor/cord_cosine"] = cosine.item()
    return metrics
