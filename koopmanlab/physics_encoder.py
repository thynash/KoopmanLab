import torch
import torch.nn as nn


# ============================================================
# FINITE-DIFFERENCE PHYSICS FEATURES
# ============================================================

def first_derivative_1d(u, dx):
    """
    u: [B, N, C]

    Returns du/dx with the same shape.
    """
    dudx = torch.zeros_like(u)

    # Central difference: interior
    dudx[:, 1:-1, :] = (
        u[:, 2:, :] - u[:, :-2, :]
    ) / (2.0 * dx)

    # One-sided differences: boundaries
    dudx[:, 0, :] = (
        u[:, 1, :] - u[:, 0, :]
    ) / dx

    dudx[:, -1, :] = (
        u[:, -1, :] - u[:, -2, :]
    ) / dx

    return dudx


def laplacian_1d(u, dx):
    """
    Second spatial derivative.

    u: [B, N, C]
    """
    uxx = torch.zeros_like(u)

    uxx[:, 1:-1, :] = (
        u[:, 2:, :]
        - 2.0 * u[:, 1:-1, :]
        + u[:, :-2, :]
    ) / (dx * dx)

    # Replicate nearest interior second derivative
    uxx[:, 0, :] = uxx[:, 1, :]
    uxx[:, -1, :] = uxx[:, -2, :]

    return uxx


def first_derivatives_2d(u, dx, dy):
    """
    u: [B, H, W, C]

    Returns:
        ux, uy
    """
    ux = torch.zeros_like(u)
    uy = torch.zeros_like(u)

    # --------------------------------------------------------
    # x-direction: H dimension
    # --------------------------------------------------------

    ux[:, 1:-1, :, :] = (
        u[:, 2:, :, :] - u[:, :-2, :, :]
    ) / (2.0 * dx)

    ux[:, 0, :, :] = (
        u[:, 1, :, :] - u[:, 0, :, :]
    ) / dx

    ux[:, -1, :, :] = (
        u[:, -1, :, :] - u[:, -2, :, :]
    ) / dx

    # --------------------------------------------------------
    # y-direction: W dimension
    # --------------------------------------------------------

    uy[:, :, 1:-1, :] = (
        u[:, :, 2:, :] - u[:, :, :-2, :]
    ) / (2.0 * dy)

    uy[:, :, 0, :] = (
        u[:, :, 1, :] - u[:, :, 0, :]
    ) / dy

    uy[:, :, -1, :] = (
        u[:, :, -1, :] - u[:, :, -2, :]
    ) / dy

    return ux, uy


def laplacian_2d(u, dx, dy):
    """
    u: [B, H, W, C]

    Returns:
        Laplacian(u)
    """
    uxx = torch.zeros_like(u)
    uyy = torch.zeros_like(u)

    # --------------------------------------------------------
    # d²u/dx²
    # --------------------------------------------------------

    uxx[:, 1:-1, :, :] = (
        u[:, 2:, :, :]
        - 2.0 * u[:, 1:-1, :, :]
        + u[:, :-2, :, :]
    ) / (dx * dx)

    uxx[:, 0, :, :] = uxx[:, 1, :, :]
    uxx[:, -1, :, :] = uxx[:, -2, :, :]

    # --------------------------------------------------------
    # d²u/dy²
    # --------------------------------------------------------

    uyy[:, :, 1:-1, :] = (
        u[:, :, 2:, :]
        - 2.0 * u[:, :, 1:-1, :]
        + u[:, :, :-2, :]
    ) / (dy * dy)

    uyy[:, :, 0, :] = uyy[:, :, 1, :]
    uyy[:, :, -1, :] = uyy[:, :, -2, :]

    return uxx + uyy


# ============================================================
# PHYSICS-AWARE RESIDUAL ENCODER 1D
# ============================================================

class PhysicsEncoder1D(nn.Module):
    """
    Physics-Aware Residual Encoder for 1D PDEs.

    Input:
        x: [B, N, t_len]

    Output:
        z: [B, N, op_size]

    Architecture:
        z_base = W_base(x)

        p = [x, dx(x), dxx(x)]

        z_phys = W_phys(p)

        z = z_base + alpha * z_phys

    alpha is initialized to zero so the model initially behaves
    like a simple KNO-compatible lifting layer.
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

        # Original-style base lifting
        self.base_projection = nn.Linear(
            t_len,
            op_size,
        )

        # Physics descriptor:
        # [u, ux, uxx]
        physics_dim = 3 * t_len

        # Lightweight physics correction
        self.physics_projection = nn.Linear(
            physics_dim,
            op_size,
            bias=False,
        )

        # Learnable residual gate
        self.alpha = nn.Parameter(
            torch.tensor(0.0)
        )

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError(
                "PhysicsEncoder1D expects [B, N, C], "
                f"received {tuple(x.shape)}."
            )

        if x.shape[-1] != self.t_len:
            raise ValueError(
                f"Expected {self.t_len} channels, "
                f"received {x.shape[-1]}."
            )

        # ----------------------------------------------------
        # Base representation
        # ----------------------------------------------------

        z_base = self.base_projection(x)

        # ----------------------------------------------------
        # Parameter-free physics descriptors
        # ----------------------------------------------------

        ux = first_derivative_1d(
            x,
            self.dx,
        )

        uxx = laplacian_1d(
            x,
            self.dx,
        )

        physics_features = torch.cat(
            [
                x,
                ux,
                uxx,
            ],
            dim=-1,
        )

        # ----------------------------------------------------
        # Physics residual
        # ----------------------------------------------------

        z_phys = self.physics_projection(
            physics_features
        )

        # ----------------------------------------------------
        # Residual fusion
        # ----------------------------------------------------

        z = z_base + self.alpha * z_phys

        return z


# ============================================================
# PHYSICS-AWARE RESIDUAL ENCODER 2D
# ============================================================

class PhysicsEncoder2D(nn.Module):
    """
    Physics-Aware Residual Encoder for 2D PDEs.

    Input:
        x: [B, H, W, t_len]

    Output:
        z: [B, H, W, op_size]

    Architecture:
        z_base = W_base(x)

        p = [x, ux, uy, Laplacian(x)]

        z_phys = W_phys(p)

        z = z_base + alpha * z_phys
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

        # Base KNO-compatible channel lifting
        self.base_projection = nn.Linear(
            t_len,
            op_size,
        )

        # [u, ux, uy, laplacian]
        physics_dim = 4 * t_len

        self.physics_projection = nn.Linear(
            physics_dim,
            op_size,
            bias=False,
        )

        # Residual gate
        self.alpha = nn.Parameter(
            torch.tensor(0.0)
        )

    def forward(self, x):

        if x.ndim != 4:
            raise ValueError(
                "PhysicsEncoder2D expects [B, H, W, C], "
                f"received {tuple(x.shape)}."
            )

        if x.shape[-1] != self.t_len:
            raise ValueError(
                f"Expected {self.t_len} channels, "
                f"received {x.shape[-1]}."
            )

        # ----------------------------------------------------
        # Base representation
        # ----------------------------------------------------

        z_base = self.base_projection(x)

        # ----------------------------------------------------
        # Parameter-free PDE descriptors
        # ----------------------------------------------------

        ux, uy = first_derivatives_2d(
            x,
            self.dx,
            self.dy,
        )

        lap = laplacian_2d(
            x,
            self.dx,
            self.dy,
        )

        physics_features = torch.cat(
            [
                x,
                ux,
                uy,
                lap,
            ],
            dim=-1,
        )

        # ----------------------------------------------------
        # Physics correction
        # ----------------------------------------------------

        z_phys = self.physics_projection(
            physics_features
        )

        # ----------------------------------------------------
        # Safe residual fusion
        # ----------------------------------------------------

        z = z_base + self.alpha * z_phys

        return z
