import torch
import torch.nn as nn

from koopmanlab.physics_encoder import (
    first_derivative_1d,
    laplacian_1d,
    first_derivatives_2d,
    laplacian_2d,
)


# ============================================================
# PHYSICS-AWARE RESIDUAL DECODER 1D
# ============================================================

class PhysicsDecoder1D(nn.Module):
    """
    Physics-Aware Residual Decoder for 1D PDEs.

    Input:
        z: [B, N, op_size]

    Output:
        u: [B, N, t_len]

    Architecture:

        u_base = W_base(z)

        p = [z, dz/dx, d²z/dx²]

        u_phys = W_phys(p)

        u = u_base + beta * u_phys

    beta starts at zero.
    """

    def __init__(
        self,
        t_len,
        op_size,
        hidden_size=None,
        dx=1.0,
    ):
        super().__init__()

        self.t_len = t_len
        self.op_size = op_size
        self.dx = dx

        # Standard KNO-compatible decoding
        self.base_projection = nn.Linear(
            op_size,
            t_len,
        )

        # [z, zx, zxx]
        physics_dim = 3 * op_size

        self.physics_projection = nn.Linear(
            physics_dim,
            t_len,
            bias=False,
        )

        # Learnable residual gate
        self.beta = nn.Parameter(
            torch.tensor(0.0)
        )

    def forward(self, z):

        if z.ndim != 3:
            raise ValueError(
                "PhysicsDecoder1D expects [B, N, C], "
                f"received {tuple(z.shape)}."
            )

        if z.shape[-1] != self.op_size:
            raise ValueError(
                f"Expected latent dimension {self.op_size}, "
                f"received {z.shape[-1]}."
            )

        # ----------------------------------------------------
        # Base decoding
        # ----------------------------------------------------

        u_base = self.base_projection(z)

        # ----------------------------------------------------
        # Latent spatial physics descriptors
        # ----------------------------------------------------

        zx = first_derivative_1d(
            z,
            self.dx,
        )

        zxx = laplacian_1d(
            z,
            self.dx,
        )

        physics_features = torch.cat(
            [
                z,
                zx,
                zxx,
            ],
            dim=-1,
        )

        # ----------------------------------------------------
        # Physics correction
        # ----------------------------------------------------

        u_phys = self.physics_projection(
            physics_features
        )

        # ----------------------------------------------------
        # Residual decoding
        # ----------------------------------------------------

        u = u_base + self.beta * u_phys

        return u


# ============================================================
# PHYSICS-AWARE RESIDUAL DECODER 2D
# ============================================================

class PhysicsDecoder2D(nn.Module):
    """
    Physics-Aware Residual Decoder for 2D PDEs.

    Input:
        z: [B, H, W, op_size]

    Output:
        u: [B, H, W, t_len]

    Architecture:

        u_base = W_base(z)

        p = [z, zx, zy, Laplacian(z)]

        u_phys = W_phys(p)

        u = u_base + beta * u_phys

    beta starts at zero.
    """

    def __init__(
        self,
        t_len,
        op_size,
        hidden_size=None,
        dx=1.0,
        dy=1.0,
    ):
        super().__init__()

        self.t_len = t_len
        self.op_size = op_size

        self.dx = dx
        self.dy = dy

        # Base decoding
        self.base_projection = nn.Linear(
            op_size,
            t_len,
        )

        # [z, zx, zy, Laplacian(z)]
        physics_dim = 4 * op_size

        self.physics_projection = nn.Linear(
            physics_dim,
            t_len,
            bias=False,
        )

        # Residual gate
        self.beta = nn.Parameter(
            torch.tensor(0.0)
        )

    def forward(self, z):

        if z.ndim != 4:
            raise ValueError(
                "PhysicsDecoder2D expects [B, H, W, C], "
                f"received {tuple(z.shape)}."
            )

        if z.shape[-1] != self.op_size:
            raise ValueError(
                f"Expected latent dimension {self.op_size}, "
                f"received {z.shape[-1]}."
            )

        # ----------------------------------------------------
        # Base decoding
        # ----------------------------------------------------

        u_base = self.base_projection(z)

        # ----------------------------------------------------
        # Latent physics descriptors
        # ----------------------------------------------------

        zx, zy = first_derivatives_2d(
            z,
            self.dx,
            self.dy,
        )

        z_lap = laplacian_2d(
            z,
            self.dx,
            self.dy,
        )

        physics_features = torch.cat(
            [
                z,
                zx,
                zy,
                z_lap,
            ],
            dim=-1,
        )

        # ----------------------------------------------------
        # Physics correction
        # ----------------------------------------------------

        u_phys = self.physics_projection(
            physics_features
        )

        # ----------------------------------------------------
        # Residual fusion
        # ----------------------------------------------------

        u = u_base + self.beta * u_phys

        return u
