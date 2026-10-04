import torch
from torch import Tensor
from torchdiffeq import odeint


def euler_step(fn, t, y, dt):
    return y + dt * fn(t, y)


def midpoint_step(fn, t, y, dt):
    k1 = fn(t, y)
    k2 = fn(t + dt / 2, y + dt / 2 * k1)
    return y + dt * k2


def heun_step(fn, t, y, dt):
    k1 = fn(t, y)
    k2 = fn(t + dt, y + dt * k1)
    return y + dt / 2 * (k1 + k2)


def rk4_step(fn, t, y, dt):
    k1 = fn(t, y)
    k2 = fn(t + dt / 2, y + dt / 2 * k1)
    k3 = fn(t + dt / 2, y + dt / 2 * k2)
    k4 = fn(t + dt, y + dt * k3)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk38_step(fn, t, y, dt):
    k1 = fn(t, y)
    k2 = fn(t + dt / 3, y + dt / 3 * k1)
    k3 = fn(t + 2 * dt / 3, y - dt / 3 * k1 + dt * k2)
    k4 = fn(t + dt, y + dt * k1 - dt * k2 + dt * k3)
    return y + dt / 8 * (k1 + 3 * k2 + 3 * k3 + k4)


def ralston_step(fn, t, y, dt):
    k1 = fn(t, y)
    k2 = fn(t + 2 * dt / 3, y + 2 * dt / 3 * k1)
    return y + dt * (k1 / 4 + 3 * k2 / 4)


def ssprk3_step(fn, t, y, dt):
    y1 = y + dt * fn(t, y)
    y2 = 0.75 * y + 0.25 * (y1 + dt * fn(t + dt, y1))
    return y / 3 + 2 / 3 * (y2 + dt * fn(t + dt / 2, y2))


CUSTOM_ODE_SOLVERS = {
    "heun_custom": heun_step,
    "rk4_custom": rk4_step,
    "rk38": rk38_step,
    "ralston": ralston_step,
    "ssprk3": ssprk3_step,
}

TORCHDIFFEQ_METHODS = [
    "euler",
    "midpoint",
    "rk4",
    "explicit_adams",
    "implicit_adams",
    "dopri5",
    "dopri8",
    "bosh3",
    "adaptive_heun",
    "scipy_solver",
]


def odeint_custom(fn, y0, t, method="euler", **kwargs):
    if method in TORCHDIFFEQ_METHODS:
        return odeint(fn, y0, t, method=method, **kwargs)

    if method not in CUSTOM_ODE_SOLVERS:
        raise ValueError(
            f"Unknown ODE solver method: {method}. "
            f"Available torchdiffeq methods: {TORCHDIFFEQ_METHODS}\n"
            f"Available custom methods: {list(CUSTOM_ODE_SOLVERS.keys())}"
        )

    step_fn = CUSTOM_ODE_SOLVERS[method]
    trajectory = [y0]
    y = y0

    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        y = step_fn(fn, t[i], y, dt)
        trajectory.append(y)

    return torch.stack(trajectory, dim=0)
