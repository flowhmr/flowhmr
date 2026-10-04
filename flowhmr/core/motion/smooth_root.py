# Standalone smooth-root utility (adapted from an earlier codebase, Apache-2.0).

# The original batched-input helper dependency was removed.
#          Rewritten to use pure PyTorch (no scipy/numpy dependency).
"""Smooth root trajectory: ADMM-based smoother with margin constraints and get_smooth_root_pos helper."""

import math

import torch


class TrajectorySmoother:
    """Modify trajectories to hit target values while respecting soft constraints.

    This smoother keeps the trajectory close to the original positions while minimizing
    accelerations. Targets are enforced at specified frames via soft constraints.

    All internal operations use PyTorch tensors. The system matrix is pre-factorized
    via Cholesky decomposition for efficient repeated solves.
    """

    def __init__(
        self,
        margins,
        pos_weight=0.0,
        loop=False,
        admm_iters=100,
        alpha_overrelax=1.0,
        circle_project=False,
        device=None,
        dtype=torch.float64,
    ):
        self.pos_weight = pos_weight
        self.admm_iters = admm_iters
        self.alpha_overrelax = alpha_overrelax
        self.circle_project = circle_project
        self.dtype = dtype

        if not isinstance(margins, torch.Tensor):
            margins = torch.tensor(margins, dtype=dtype, device=device)
        else:
            margins = margins.to(dtype=dtype, device=device)
        self.margin_vals = margins
        N = len(margins)
        self.device = margins.device

        A = torch.zeros(N, N, dtype=dtype, device=self.device)

        for i in range(1, N - 1):
            A[i, i - 1] += -1.0
            A[i, i] += 2.0
            A[i, i + 1] += -1.0

        if loop:
            A[0, N - 1] += -1.0
            A[0, 0] += 2.0
            A[0, 1] += -1.0

            A[N - 1, N - 2] += -1.0
            A[N - 1, N - 1] += 2.0
            A[N - 1, 0] += -1.0

        identity = torch.eye(N, dtype=dtype, device=self.device)
        M = pos_weight * identity + A.T @ A

        diag_max = M.diagonal().abs().max().item()
        self.admm_stepsize = 0.25 * math.sqrt(diag_max)

        M = M + self.admm_stepsize * identity

        # Cholesky factorization for efficient repeated solves (M is SPD)
        self.L = torch.linalg.cholesky(M)

    def smooth(self, targets, x0):
        x_target = targets.clone()
        x = x0.clone()
        z = torch.zeros_like(x)
        u = torch.zeros_like(x)

        for _ in range(self.admm_iters):
            self._z_update(z, x, x_target, u)
            self._u_update(u, x, z)
            self._x_update(x, z, u, x_target)

        return x

    def _x_update(self, x, z, u, x_t):
        """Update x in the ADMM iteration."""
        r = self.pos_weight * x_t + self.admm_stepsize * (z - u)
        # Solve M @ x = r via pre-computed Cholesky L: L @ L^T @ x = r
        # Use solve_triangular (forward + backward substitution) instead of
        # cholesky_solve for ~20x speedup on medium-sized matrices.
        y = torch.linalg.solve_triangular(self.L, r, upper=False)
        x[:] = torch.linalg.solve_triangular(self.L.mT, y, upper=True)

    def _z_update(self, z, x, z_t, u):
        """Update z in the ADMM iteration."""
        z[:] = x + u - z_t

        z_diff_norms = torch.linalg.norm(z, dim=1)
        mask = z_diff_norms > self.margin_vals
        if mask.any():
            scale_factors = self.margin_vals[mask] / z_diff_norms[mask]
            z[mask] *= scale_factors.unsqueeze(1)

        z[:] += z_t

        if self.circle_project:
            z[:] = z / (torch.linalg.norm(z, dim=1, keepdim=True) + 1.0e-6)

    def _u_update(self, u, x, z):
        """Update u in the ADMM iteration."""
        u[:] += self.alpha_overrelax * (x - z)


def smooth_signal(x, margins, pos_weight=0, alpha_overrelax=1.8, admm_iters=100, circle_project=False):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float64)
    if not isinstance(margins, torch.Tensor):
        margins = torch.tensor(margins, dtype=x.dtype, device=x.device)

    if len(x) <= 4:
        return x.clone()

    x_smoothed = x.clone()
    x_smoothed[:] = x.mean(dim=0, keepdim=True)

    levels = int(math.floor(math.log2(len(x))))
    levels = max(levels - 4, 1)

    stepsize = 2**levels
    while True:
        smoother = TrajectorySmoother(
            margins=margins[::stepsize],
            pos_weight=pos_weight,
            alpha_overrelax=alpha_overrelax,
            admm_iters=admm_iters,
            circle_project=circle_project,
            device=x.device,
            dtype=x.dtype,
        )
        x_smoothed[::stepsize] = smoother.smooth(x[::stepsize], x_smoothed[::stepsize])

        next_stepsize = stepsize // 2
        num_interleaved = len(x_smoothed[next_stepsize::stepsize])
        num_steps = len(x_smoothed[::stepsize])
        if num_interleaved == num_steps:
            x_smoothed[next_stepsize::stepsize][-1] = (
                x_smoothed[::stepsize][-1] + (x_smoothed[::stepsize][-1] - x_smoothed[::stepsize][-2]) / 2
            )
            num_interleaved = num_interleaved - 1

        x_smoothed[next_stepsize::stepsize][:num_interleaved] = (
            x_smoothed[::stepsize][:-1] + x_smoothed[::stepsize][1:]
        ) / 2

        if stepsize == 1:
            break

        stepsize //= 2

    return x_smoothed


def get_smooth_root_pos(hip_translations):
    unbatched = hip_translations.ndim == 2
    if unbatched:
        hip_translations = hip_translations.unsqueeze(0)

    root_translations_xz = hip_translations[..., [0, 2]]
    root_translations_y = hip_translations[..., [1]]

    batch_size = root_translations_xz.shape[0]
    T = root_translations_xz.shape[1]
    margins = torch.full((T,), 0.06, dtype=hip_translations.dtype, device=hip_translations.device)

    smoothed_xz_list = []
    for b in range(batch_size):
        smoothed = smooth_signal(root_translations_xz[b], margins)
        smoothed_xz_list.append(smoothed.unsqueeze(0))

    smoothed_xz = torch.cat(smoothed_xz_list, dim=0)

    # cat → (B, T, 3) with layout [smoothed_x, smoothed_z, y]
    root_translations = torch.cat([smoothed_xz, root_translations_y], dim=-1)[..., [0, 2, 1]]

    if unbatched:
        root_translations = root_translations.squeeze(0)

    return root_translations
