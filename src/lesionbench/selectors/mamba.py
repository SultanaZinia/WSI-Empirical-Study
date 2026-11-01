"""Bidirectional Mamba-1 mixer, as in SLICEChat-encoder's models/mamba/mamba_custom.py (biscan=True).

The selective scan uses mamba_ssm's CUDA kernel when it is installed and the input is on a GPU, and a pure
PyTorch reference (the recurrence written out, slow) otherwise, so unit tests run on CPU.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:  # mamba_ssm imports transformers at package level; any import failure means "no kernel"
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except Exception:  # noqa: BLE001
    selective_scan_fn = None


def selective_scan_ref(u, delta, A, B, C, D, z, delta_bias) -> Tensor:
    """u, delta, z (b, d, l); A (d, n); B, C (b, n, l); D, delta_bias (d,).

    Same maths as mamba_ssm's kernel.
    """
    delta = F.softplus(delta + delta_bias[:, None])
    dA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    dBu = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    h, ys = u.new_zeros(u.shape[0], u.shape[1], A.shape[1]), []
    for i in range(u.shape[2]):
        h = dA[:, :, i] * h + dBu[:, :, i]
        ys.append(torch.einsum("bdn,bn->bd", h, C[:, :, i]))
    y = torch.stack(ys, dim=-1) + u * D[:, None]
    return y * F.silu(z)


def selective_scan(u, delta, A, B, C, D, z, delta_bias, use_kernel: bool | None = None) -> Tensor:
    if use_kernel is None:
        use_kernel = selective_scan_fn is not None and u.is_cuda
    if use_kernel:
        return selective_scan_fn(u, delta, A, B, C, D, z, delta_bias, delta_softplus=True)
    return selective_scan_ref(u, delta, A, B, C, D, z, delta_bias)


class _Direction(nn.Module):
    """Conv + SSM parameters for one scan direction (mamba_ssm.Mamba initialisation)."""

    def __init__(self, d_inner: int, d_state: int, d_conv: int, dt_rank: int, dt_min=1e-3, dt_max=0.1):
        super().__init__()
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner)
        nn.init.uniform_(self.dt_proj.weight, -dt_rank**-0.5, dt_rank**-0.5)
        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)  # S4D real init
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.dt_rank, self.d_state = dt_rank, d_state

    def forward(self, x: Tensor, z: Tensor, use_kernel=None) -> Tensor:
        L = x.shape[-1]
        x = F.silu(self.conv1d(x)[..., :L])  # causal depthwise conv
        dt, B, C = self.x_proj(x.transpose(1, 2)).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = (dt @ self.dt_proj.weight.T).transpose(1, 2)
        return selective_scan(x, delta, -torch.exp(self.A_log.float()), B.transpose(1, 2).contiguous(),
                              C.transpose(1, 2).contiguous(), self.D.float(), z, self.dt_proj.bias.float(),
                              use_kernel)


class BiMamba(nn.Module):
    """(b, l, d_model) -> (b, l, d_model). Forward and backward scans with separate SSM parameters, shared
    in/out projections, outputs averaged (SLICEChat's biscan)."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner, dt_rank = expand * d_model, math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.fwd = _Direction(d_inner, d_state, d_conv, dt_rank)
        self.bwd = _Direction(d_inner, d_state, d_conv, dt_rank)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, h: Tensor, use_kernel=None) -> Tensor:
        x, z = self.in_proj(h).transpose(1, 2).chunk(2, dim=1)
        y_f = self.fwd(x, z, use_kernel)
        y_b = self.bwd(x.flip(-1), z.flip(-1), use_kernel).flip(-1)
        return (self.out_proj(y_f.transpose(1, 2)) + self.out_proj(y_b.transpose(1, 2))) / 2
