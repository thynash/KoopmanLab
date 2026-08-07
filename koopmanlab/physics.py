"""
physics.py
===========

Physics loss module for Physics-Informed Koopman Neural Operator (PIKNO)

Current PDE:
    2D Incompressible Navier-Stokes (without pressure term)

Author:
"""

import torch
import torch.nn as nn


class NavierStokesLoss(nn.Module):
    """
    Physics loss for Navier-Stokes equation.

    Computes

        R = du/dt + u·∇u - ν∇²u

    Physics Loss:

        L_phy = mean(R²)
    """

    def __init__(
        self,
        nu=1e-3,
        dx=1/64,
        dy=1/64,
        dt=1.0
    ):
        super().__init__()

        self.nu = nu
        self.dx = dx
        self.dy = dy
        self.dt = dt

    # --------------------------------------------------------
    # First-order derivatives
    # --------------------------------------------------------

    def gradient_x(self, u):
        return (
            torch.roll(u, -1, dims=2)
            - torch.roll(u, 1, dims=2)
        ) / (2 * self.dx)

    def gradient_y(self, u):
        return (
            torch.roll(u, -1, dims=1)
            - torch.roll(u, 1, dims=1)
        ) / (2 * self.dy)

    # --------------------------------------------------------
    # Laplacian
    # --------------------------------------------------------

    def laplacian(self, u):

        u_xx = (
            torch.roll(u, -1, dims=2)
            - 2*u
            + torch.roll(u, 1, dims=2)
        ) / (self.dx**2)

        u_yy = (
            torch.roll(u, -1, dims=1)
            - 2*u
            + torch.roll(u, 1, dims=1)
        ) / (self.dy**2)

        return u_xx + u_yy

    # --------------------------------------------------------
    # Time derivative
    # --------------------------------------------------------

    def time_derivative(self, prev, pred):

        return (pred - prev) / self.dt

    # --------------------------------------------------------
    # Advection
    # --------------------------------------------------------

    def advection(self, u):

        ux = self.gradient_x(u)
        uy = self.gradient_y(u)

        return u * ux + u * uy

    # --------------------------------------------------------
    # PDE Residual
    # --------------------------------------------------------

    def residual(self, prev, pred):

        ut = self.time_derivative(prev, pred)

        adv = self.advection(pred)

        diff = self.nu * self.laplacian(pred)

        residual = ut + adv - diff

        return residual

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(self, previous_state, prediction):

        residual = self.residual(
            previous_state,
            prediction
        )

        return torch.mean(residual ** 2)
