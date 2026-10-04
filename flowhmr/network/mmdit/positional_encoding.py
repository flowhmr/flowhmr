from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        num_feats: int,
        max_seq_len: Union[Tensor, int],
        temperature: float = 10000.0,
        use_real: bool = False,
        theta_rescale_factor: float = 1.0,
        interpolation_factor: float = 1.0,
    ) -> None:
        super(RotaryEmbedding, self).__init__()
        assert num_feats % 2 == 0, "num_feats (head_dim) must be even for RoPE."
        self.num_feats = num_feats
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        self.use_real = use_real
        self.theta_rescale_factor = theta_rescale_factor
        self.interpolation_factor = interpolation_factor

        if isinstance(max_seq_len, int):
            max_seq_len = torch.arange(max_seq_len, dtype=torch.float32)

        if theta_rescale_factor != 1.0:
            temperature *= theta_rescale_factor ** (self.num_feats / (self.num_feats - 2))
        dim_t = torch.arange(0, self.num_feats, 2, dtype=torch.float32)
        freqs = 1.0 / (temperature ** (2 * torch.div(dim_t, 2, rounding_mode="trunc") / self.num_feats))  # [D/2]
        freqs = torch.outer(max_seq_len.to(dtype=torch.float32) * interpolation_factor, freqs)
        if use_real:
            freqs_cos = freqs.cos().repeat_interleave(2, dim=1)
            freqs_sin = freqs.sin().repeat_interleave(2, dim=1)
            self.register_buffer("_freqs_cos", freqs_cos, persistent=False)
            self.register_buffer("_freqs_sin", freqs_sin, persistent=False)
            self.register_buffer("_freqs_complex", None, persistent=False)
        else:
            freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
            self.register_buffer("_freqs_cos", None, persistent=False)
            self.register_buffer("_freqs_sin", None, persistent=False)
            self.register_buffer("_freqs_complex", freqs_complex, persistent=False)

    @property
    def freqs_cis(self) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if self.use_real:
            return self._freqs_cos, self._freqs_sin
        return self._freqs_complex

    def reshape_for_broadcast(
        self, freqs_cis: Union[Tensor, Tuple[Tensor, Tensor]], x: Tensor
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        ndim = x.ndim
        assert 0 <= 1 < ndim

        if isinstance(freqs_cis, tuple):
            # freqs_cis: (cos, sin) in real space
            assert (
                freqs_cis[0].shape[-1] == x.shape[-1]
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape} on the head_dim dimension"
            assert (
                freqs_cis[0].shape[0] >= x.shape[1]
            ), f"freqs_cis shape {freqs_cis[0].shape} should be larger than or equal to x shape {x.shape} on the time dimension"
            shape = []
            for i, d in enumerate(x.shape):
                if i == 1:
                    shape.append(-1)
                elif i == ndim - 1:
                    shape.append(d)
                else:
                    shape.append(1)
            return (
                freqs_cis[0].view(*shape)[:, : x.shape[1], ...],
                freqs_cis[1].view(*shape)[:, : x.shape[1], ...],
            )
        else:
            # freqs_cis: values in complex space
            assert (
                freqs_cis.shape[-1] == x.shape[-1]
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape} on the head_dim dimension"
            assert (
                freqs_cis.shape[0] >= x.shape[1]
            ), f"freqs_cis shape {freqs_cis.shape} should be larger than or equal to x shape {x.shape} on the time dimension"
            shape = []
            for i, d in enumerate(x.shape):
                if i == 1:
                    shape.append(-1)
                elif i == ndim - 1:
                    shape.append(d)
                else:
                    shape.append(1)
            return freqs_cis.view(*shape)[:, : x.shape[1], ...]

    def rotate_half(self, x: Tensor) -> Tensor:
        work_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        x_cast = x.to(dtype=work_dtype)
        x_real, x_imag = x_cast.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        out = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        return out.to(dtype=x.dtype)

    def apply_rotary_emb(self, xq: Tensor, xk: Tensor, split_len: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        # - split_len is intentionally ignored for 1D RoPE (RotaryEmbedding).
        # - It is used by MultimodalRotaryEmbedding to align the motion/text boundary dynamically.
        xk_out = None
        work_dtype = torch.float32
        if isinstance(self.freqs_cis, tuple):
            cos, sin = self.reshape_for_broadcast(self.freqs_cis, xq)
            cos = cos.to(device=xq.device, dtype=work_dtype)
            sin = sin.to(device=xq.device, dtype=work_dtype)
            # real * cos - imag * sin
            # imag * cos + real * sin
            xq_cast = xq.to(dtype=work_dtype)
            xk_cast = xk.to(dtype=work_dtype)
            xq_out = (xq_cast * cos + self.rotate_half(xq_cast) * sin).type_as(xq)
            xk_out = (xk_cast * cos + self.rotate_half(xk_cast) * sin).type_as(xk)
        else:
            # view_as_complex will pack [..., D/2, 2](real) to [..., D/2](complex)
            xq_ = torch.view_as_complex(xq.to(dtype=work_dtype).reshape(*xq.shape[:-1], -1, 2))
            freqs_cis = self.reshape_for_broadcast(self.freqs_cis, xq_)
            # Handle device transfer based on return type
            if isinstance(freqs_cis, tuple):
                freqs_cis = (
                    freqs_cis[0].to(device=xq.device, dtype=work_dtype),
                    freqs_cis[1].to(device=xq.device, dtype=work_dtype),
                )
            else:
                freqs_cis = freqs_cis.to(device=xq.device)
            # (real, imag) * (cos, sin) = (real * cos - imag * sin, imag * cos + real * sin)
            # view_as_real will expand [..., D/2](complex) to [..., D/2, 2](real)
            xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
            xk_ = torch.view_as_complex(xk.to(dtype=work_dtype).reshape(*xk.shape[:-1], -1, 2))
            xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)
        return xq_out, xk_out

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(num_feats={self.num_feats}, "
        repr_str += f"max_seq_len={self.max_seq_len}, "
        repr_str += f"temperature={self.temperature}, "
        repr_str += f"use_real={self.use_real}, "
        repr_str += f"theta_rescale_factor={self.theta_rescale_factor}, "
        repr_str += f"interpolation_factor={self.interpolation_factor})"
        return repr_str


class MultimodalRotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        num_feats: int,
        max_text_len: int,
        max_motion_len: int,
        text_rope_base: float = 10000.0,
        motion_rope_base: float = 10000.0,
        use_real: bool = False,
        theta_rescale_factor: float = 1.0,
        interpolation_factor: float = 1.0,
    ) -> None:
        # We keep the base class initialization for shared utilities (rotate_half / reshape_for_broadcast),
        # but MultimodalRoPE will build freqs dynamically at runtime based on the *actual* split_len.
        super().__init__(
            num_feats=num_feats,
            max_seq_len=max_text_len + max_motion_len,
            temperature=text_rope_base,
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor,
            interpolation_factor=interpolation_factor,
        )

        self.max_text_len = int(max_text_len)
        self.max_motion_len = int(max_motion_len)
        self.text_rope_base = float(text_rope_base)
        self.motion_rope_base = float(motion_rope_base)

        assert num_feats % 4 == 0, "For 2D split RoPE, num_feats must be divisible by 4."
        sub_dim = num_feats // 2
        dim_t = torch.arange(0, sub_dim, 2, dtype=torch.float32)
        self.inv_freq_text = 1.0 / (text_rope_base ** (dim_t / sub_dim))
        self.inv_freq_motion = 1.0 / (motion_rope_base ** (dim_t / sub_dim))

        # Cache freqs on per-device to avoid rebuilding every layer call
        # Key: (seq_len, split_len, device_str, use_real)
        self._mm_freqs_cache: Dict[Tuple[int, int, str, bool], Union[Tensor, Tuple[Tensor, Tensor]]] = {}

    def _get_mm_freqs_cis(
        self,
        seq_len: int,
        split_len: int,
        device: torch.device,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if split_len < 0 or split_len > seq_len:
            raise ValueError(f"split_len must be in [0, {seq_len}], got {split_len}")
        text_len = seq_len - split_len
        if split_len > self.max_motion_len or text_len > self.max_text_len:
            raise ValueError(
                f"seq_len exceeds configured maxima: motion_len={split_len} (max {self.max_motion_len}), "
                f"text_len={text_len} (max {self.max_text_len})"
            )

        device_key = f"{device.type}:{device.index}" if device.type != "cpu" else "cpu"
        cache_key = (seq_len, split_len, device_key, bool(self.use_real))
        cached = self._mm_freqs_cache.get(cache_key)
        if cached is not None:
            return cached

        # dynamic orthogonal coordinate system for [Motion, Text]
        pos_text_axis = torch.cat(
            [
                torch.zeros(split_len, dtype=torch.float32, device=device),
                torch.arange(text_len, dtype=torch.float32, device=device),
            ],
            dim=0,
        ) * float(self.interpolation_factor)
        pos_motion_axis = torch.cat(
            [
                torch.arange(split_len, dtype=torch.float32, device=device),
                torch.zeros(text_len, dtype=torch.float32, device=device),
            ],
            dim=0,
        ) * float(self.interpolation_factor)

        inv_freq_text = self.inv_freq_text.to(device=device, non_blocking=True)
        inv_freq_motion = self.inv_freq_motion.to(device=device, non_blocking=True)

        freqs_part1 = torch.outer(pos_text_axis, inv_freq_text)
        freqs_part2 = torch.outer(pos_motion_axis, inv_freq_motion)
        freqs = torch.cat([freqs_part1, freqs_part2], dim=-1)

        if self.use_real:
            out: Union[Tensor, Tuple[Tensor, Tensor]] = (
                freqs.cos().repeat_interleave(2, dim=1),
                freqs.sin().repeat_interleave(2, dim=1),
            )
        else:
            out = torch.polar(torch.ones_like(freqs), freqs) # [S, D/2] (complex)

        self._mm_freqs_cache[cache_key] = out
        return out

    def apply_rotary_emb(self, xq: Tensor, xk: Tensor, split_len: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        if split_len is None:
            split_len = xq.shape[1]
        freqs_cis = self._get_mm_freqs_cis(seq_len=xq.shape[1], split_len=int(split_len), device=xq.device)

        work_dtype = torch.float32
        if isinstance(freqs_cis, tuple):
            cos, sin = self.reshape_for_broadcast(freqs_cis, xq)
            cos = cos.to(device=xq.device, dtype=work_dtype)
            sin = sin.to(device=xq.device, dtype=work_dtype)
            # real * cos - imag * sin
            # imag * cos + real * sin
            xq_cast = xq.to(dtype=work_dtype)
            xk_cast = xk.to(dtype=work_dtype)
            xq_out = (xq_cast * cos + self.rotate_half(xq_cast) * sin).type_as(xq)
            xk_out = (xk_cast * cos + self.rotate_half(xk_cast) * sin).type_as(xk)
        else:
            # complex path
            xq_ = torch.view_as_complex(xq.to(dtype=work_dtype).reshape(*xq.shape[:-1], -1, 2))
            freqs_cis_bc = self.reshape_for_broadcast(freqs_cis, xq_)
            if isinstance(freqs_cis_bc, tuple):
                freqs_cis_bc = (
                    freqs_cis_bc[0].to(device=xq.device, dtype=work_dtype),
                    freqs_cis_bc[1].to(device=xq.device, dtype=work_dtype),
                )
            else:
                freqs_cis_bc = freqs_cis_bc.to(device=xq.device)
            xq_out = torch.view_as_real(xq_ * freqs_cis_bc).flatten(3).type_as(xq)
            xk_ = torch.view_as_complex(xk.to(dtype=work_dtype).reshape(*xk.shape[:-1], -1, 2))
            xk_out = torch.view_as_real(xk_ * freqs_cis_bc).flatten(3).type_as(xk)
        return xq_out, xk_out


class PositionalEncoding(nn.Module):
    def __init__(self, num_feats: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, num_feats, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_feats, 2, dtype=torch.float32) * (-np.log(10000.0) / num_feats))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # shape of [1, L, D]
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        pe = self.pe[:, : x.shape[1], :].to(device=x.device, dtype=x.dtype)
        x = x + pe # shape of [B, L, D]
        return self.dropout(x)


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies. (Only for image-like inputs)
    """

    def __init__(self, num_feats: int, scale: Optional[float] = 1.0, dropout: float = 0.1) -> None:
        super().__init__()

        self.num_feats = num_feats
        self.scale = scale if scale is not None and scale > 0.0 else 1.0

        self.dropout = nn.Dropout(p=dropout)

        # NOTE: Keep the random matrix in the state_dict so the encoding is deterministic
        # after initialization / checkpoint loading.
        self.register_buffer(
            "gaussian_matrix",
            self.scale * torch.randn((2, self.num_feats), dtype=torch.float32),
        )

    def _process_coords(self, coords: Tensor, *, gaussian_matrix: Tensor) -> Tensor:
        if coords.shape[-1] != 2:
            raise ValueError(f"coords last dim must be 2 (x, y), got {coords.shape}")
        # coords: [..., 2] in float32
        coords = 2.0 * coords - 1.0
        coords = coords @ gaussian_matrix # [..., num_feats]
        coords = 2.0 * np.pi * coords
        return torch.cat([coords.sin(), coords.cos()], dim=-1) # [..., 2*num_feats]

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Input x must be 4D, got shape {tuple(x.shape)}")

        is_channel_last = (x.shape[-1] == self.num_feats * 2) or (x.shape[-1] == x.shape[1])

        c_expected = int(self.num_feats) * 2

        if x.shape[-1] == c_expected:
            # channel-last: [B, H, W, C]
            h, w = x.shape[1], x.shape[2]
            permute_output = False
        elif x.shape[1] == c_expected:
            # channel-first: [B, C, H, W]
            h, w = x.shape[2], x.shape[3]
            permute_output = True
        else:
            raise ValueError(
                f"Cannot infer layout: expected channel dim == 2*num_feats ({c_expected}), got shape {tuple(x.shape)}"
            )

        device = x.device
        dtype = torch.float32
        gm = self.gaussian_matrix.to(device=device, dtype=dtype)

        # Grid Generation
        y = (torch.arange(h, device=device, dtype=dtype) + 0.5) / float(h)
        x_axis = (torch.arange(w, device=device, dtype=dtype) + 0.5) / float(w)
        y_grid, x_grid = torch.meshgrid(y, x_axis, indexing="ij")  # [H, W]
        coords = torch.stack([x_grid, y_grid], dim=-1)

        pe = self._process_coords(coords, gaussian_matrix=gm)
        pe = pe.to(dtype=x.dtype)

        if permute_output:
            pe = pe.permute(2, 0, 1)

        pe = pe.unsqueeze(0) # [1, H, W, C] or [1, C, H, W]
        return pe

    def forward_at_coords(self, coords: Tensor, image_size: Tuple[int, int]) -> Tensor:
        """
        Specific usage for sparse coordinates (e.g. prompt points).
        Renamed from 'forward_with_coords' to sound more specific.
        """
        # Normalize coords based on image_size
        h, w = image_size
        coords_norm = coords.clone()
        coords_norm[..., 0] = coords_norm[..., 0] / float(w)
        coords_norm[..., 1] = coords_norm[..., 1] / float(h)

        device = coords.device
        dtype = torch.float32
        gm = self.gaussian_matrix.to(device=device, dtype=dtype)
        return self._process_coords(coords_norm, gaussian_matrix=gm)


def visualize_multimodal_rope(
    head_dim: int, motion_len: int, text_len: int, max_len_motion: int = 5000, max_len_text: int = 5000
):
    import os

    import matplotlib.pyplot as plt
    import seaborn as sns

    seq_len = motion_len + text_len

    rope_model = MultimodalRotaryEmbedding(
        num_feats=head_dim, max_text_len=max_len_text, max_motion_len=max_len_motion, use_real=True
    )

    # Shape: [Batch=1, Seq=Total, Head=1, Dim=Head_Dim]
    q = torch.ones(1, seq_len, 1, head_dim)
    k = torch.ones(1, seq_len, 1, head_dim)

    q_rope, k_rope = rope_model.apply_rotary_emb(q, k, split_len=motion_len)

    q_out = q_rope.squeeze(2)
    k_out = k_rope.squeeze(2)

    scores = torch.matmul(q_out, k_out.transpose(-2, -1)).squeeze(0)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(scores.detach().numpy(), cmap="viridis", square=True)

    plt.axvline(x=motion_len, color="white", linestyle="--", linewidth=2)
    plt.axhline(y=motion_len, color="white", linestyle="--", linewidth=2)

    plt.text(
        motion_len / 2,
        motion_len / 2,
        "Motion-Motion\n(Diagonal)",
        ha="center",
        va="center",
        color="black",
        weight="bold",
    )
    plt.text(
        motion_len + text_len / 2,
        motion_len + text_len / 2,
        "Text-Text\n(Diagonal)",
        ha="center",
        va="center",
        color="black",
        weight="bold",
    )
    plt.text(
        motion_len + text_len / 2,
        motion_len / 2,
        "Motion -> Text\n(Grid Pattern)",
        ha="center",
        va="center",
        color="black",
        weight="bold",
    )
    plt.text(
        motion_len / 2,
        motion_len + text_len / 2,
        "Text -> Motion\n(Grid Pattern)",
        ha="center",
        va="center",
        color="black",
        weight="bold",
    )

    plt.title(f"RoPE Visualization")
    plt.xlabel(f"Key Position (0-{motion_len-1}: Motion, {motion_len}-{seq_len-1}: Text)")
    plt.ylabel(f"Query Position")

    plt.tight_layout()
    os.makedirs("debug", exist_ok=True)
    plt.savefig("debug/multimodal_rope.png")
    print(f"Saved RoPE visualization to debug/multimodal_rope.png")


if __name__ == "__main__":
    visualize_multimodal_rope(head_dim=64, motion_len=360, text_len=128)
