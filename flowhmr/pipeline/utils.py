import torch
from torch import Tensor


def length_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    assert lengths.max() <= max_len, f"lengths.max()={lengths.max()} > max_len={max_len}"
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(1)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths
    return mask


def rollout_local_transl_vel(local_transl_vel, global_orient_R, camera_vel=0, fps=30):
    transl_vel = torch.einsum("...lij,...lj->...li", global_orient_R, local_transl_vel) / fps
    transl_vel = transl_vel + camera_vel
    transl = torch.cumsum(transl_vel, dim=-2)
    return transl


def randn_tensor(shape, generator=None, device=None, dtype=None, layout=None):
    rand_device = device
    batch_size = shape[0]
    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if generator.device != torch.device("cpu"):
                msg = f"The `generator` device is `{generator.device}` and does not match the pipeline device `{device}`."
                raise ValueError(msg)
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(
                f"Cannot generate a {device} tensor from a generator of type {gen_device_type}."
            )

    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents
