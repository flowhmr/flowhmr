import math
from typing import Optional

import torch
from torch import Tensor


def flow_sde_step(
    v_pred: Tensor,
    x_t: Tensor,
    t: float,
    t_next: float,
    eta: float,
    *,
    sigma_max: Optional[float] = None,
    prev_sample: Optional[Tensor] = None,
    deterministic: bool = False,
    generator: Optional[torch.Generator] = None,
    log_prob_slice: Optional[slice] = None,
):
    device = v_pred.device
    dtype = v_pred.dtype

    sigma = 1.0 - t
    sigma_prev = 1.0 - t_next
    model_output = -v_pred
    dt = sigma_prev - sigma # = -(t_next - t) < 0

    if sigma_max is None:
        sigma_max = 1.0 - (t_next - t)

    one_minus_sigma = (1.0 - sigma_max) if abs(sigma - 1.0) < 1e-8 else (1.0 - sigma)
    one_minus_sigma = max(one_minus_sigma, 1e-8)

    std_dev_t = math.sqrt(max(sigma, 0.0) / one_minus_sigma) * eta

    pred_x1 = x_t - sigma * model_output

    prev_sample_mean = (
        x_t * (1.0 + std_dev_t**2 / (2.0 * max(sigma, 1e-8)) * dt)
        + model_output * (1.0 + std_dev_t**2 * (1.0 - sigma) / (2.0 * max(sigma, 1e-8))) * dt
    )

    std_full = std_dev_t * math.sqrt(max(-dt, 0.0))

    if prev_sample is None:
        if deterministic:
            prev_sample = x_t + dt * model_output
        else:
            noise = torch.randn(
                v_pred.shape, generator=generator, device=device, dtype=dtype
            )
            prev_sample = prev_sample_mean + std_full * noise

    safe_std = max(std_full, 1e-8)
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2.0 * safe_std**2)
        - math.log(safe_std)
        - 0.5 * math.log(2.0 * math.pi)
    )
    if log_prob_slice is not None:
        log_prob = log_prob[..., log_prob_slice]
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample, pred_x1, log_prob, prev_sample_mean, std_full


def _self_check():
    torch.manual_seed(0)
    B, T, D = 2, 5, 7
    x_t = torch.randn(B, T, D)
    v = torch.randn(B, T, D)
    steps = 20
    for i in range(steps):
        t = i / steps
        t_next = (i + 1) / steps
        x_next, pred_x1, lp, mean, std = flow_sde_step(
            v, x_t, t, t_next, eta=0.7, deterministic=True
        )
        euler = x_t + (t_next - t) * v
        assert torch.allclose(x_next, euler, atol=1e-6), (
            f"step {i}: deterministic != euler, max diff "
            f"{(x_next - euler).abs().max().item()}"
        )
        assert lp.shape == (B,), f"log_prob shape {lp.shape}"
    print("[motion_sde] self-check passed: deterministic branch == euler step")


if __name__ == "__main__":
    _self_check()
