import torch
import torch.nn as nn


# ============================================================
# NUMERICAL INTEGRATION
# ============================================================

def integrate_1d(integrand, dx):
    """
    Integrate a 1D pointwise quantity.

    Input:
        [B, N, C]

    Output:
        [B]
    """
    if integrand.ndim != 3:
        raise ValueError(
            "integrate_1d expects [B, N, C], "
            f"got {tuple(integrand.shape)}."
        )

    return integrand.sum(dim=(1, 2)) * dx


def integrate_2d(integrand, dx, dy):
    """
    Integrate a 2D pointwise quantity.

    Input:
        [B, H, W, C]

    Output:
        [B]
    """
    if integrand.ndim != 4:
        raise ValueError(
            "integrate_2d expects [B, H, W, C], "
            f"got {tuple(integrand.shape)}."
        )

    return integrand.sum(dim=(1, 2, 3)) * dx * dy


# ============================================================
# 1D DIFFERENTIAL OPERATORS
# ============================================================

def first_derivative_1d(u, dx):

    if u.ndim != 3:
        raise ValueError(
            f"Expected [B, N, C], got {tuple(u.shape)}"
        )

    ux = torch.zeros_like(u)

    ux[:, 1:-1, :] = (
        u[:, 2:, :] - u[:, :-2, :]
    ) / (2.0 * dx)

    ux[:, 0, :] = (
        u[:, 1, :] - u[:, 0, :]
    ) / dx

    ux[:, -1, :] = (
        u[:, -1, :] - u[:, -2, :]
    ) / dx

    return ux


def second_derivative_1d(u, dx):

    if u.ndim != 3:
        raise ValueError(
            f"Expected [B, N, C], got {tuple(u.shape)}"
        )

    uxx = torch.zeros_like(u)

    dx2 = dx * dx

    uxx[:, 1:-1, :] = (
        u[:, 2:, :]
        - 2.0 * u[:, 1:-1, :]
        + u[:, :-2, :]
    ) / dx2

    if u.shape[1] >= 4:

        uxx[:, 0, :] = (
            2.0 * u[:, 0, :]
            - 5.0 * u[:, 1, :]
            + 4.0 * u[:, 2, :]
            - u[:, 3, :]
        ) / dx2

        uxx[:, -1, :] = (
            2.0 * u[:, -1, :]
            - 5.0 * u[:, -2, :]
            + 4.0 * u[:, -3, :]
            - u[:, -4, :]
        ) / dx2

    else:

        uxx[:, 0, :] = uxx[:, 1, :]
        uxx[:, -1, :] = uxx[:, -2, :]

    return uxx


# ============================================================
# 2D DIFFERENTIAL OPERATORS
# ============================================================

def first_derivatives_2d(u, dx, dy):

    if u.ndim != 4:
        raise ValueError(
            f"Expected [B, H, W, C], got {tuple(u.shape)}"
        )

    ux = torch.zeros_like(u)
    uy = torch.zeros_like(u)

    # x direction = H dimension
    ux[:, 1:-1, :, :] = (
        u[:, 2:, :, :]
        - u[:, :-2, :, :]
    ) / (2.0 * dx)

    ux[:, 0, :, :] = (
        u[:, 1, :, :]
        - u[:, 0, :, :]
    ) / dx

    ux[:, -1, :, :] = (
        u[:, -1, :, :]
        - u[:, -2, :, :]
    ) / dx

    # y direction = W dimension
    uy[:, :, 1:-1, :] = (
        u[:, :, 2:, :]
        - u[:, :, :-2, :]
    ) / (2.0 * dy)

    uy[:, :, 0, :] = (
        u[:, :, 1, :]
        - u[:, :, 0, :]
    ) / dy

    uy[:, :, -1, :] = (
        u[:, :, -1, :]
        - u[:, :, -2, :]
    ) / dy

    return ux, uy


def second_derivatives_2d(u, dx, dy):

    if u.ndim != 4:
        raise ValueError(
            f"Expected [B, H, W, C], got {tuple(u.shape)}"
        )

    _, H, W, _ = u.shape

    uxx = torch.zeros_like(u)
    uyy = torch.zeros_like(u)

    dx2 = dx * dx
    dy2 = dy * dy

    uxx[:, 1:-1, :, :] = (
        u[:, 2:, :, :]
        - 2.0 * u[:, 1:-1, :, :]
        + u[:, :-2, :, :]
    ) / dx2

    if H >= 4:

        uxx[:, 0, :, :] = (
            2.0 * u[:, 0, :, :]
            - 5.0 * u[:, 1, :, :]
            + 4.0 * u[:, 2, :, :]
            - u[:, 3, :, :]
        ) / dx2

        uxx[:, -1, :, :] = (
            2.0 * u[:, -1, :, :]
            - 5.0 * u[:, -2, :, :]
            + 4.0 * u[:, -3, :, :]
            - u[:, -4, :, :]
        ) / dx2

    else:

        uxx[:, 0, :, :] = uxx[:, 1, :, :]
        uxx[:, -1, :, :] = uxx[:, -2, :, :]

    uyy[:, :, 1:-1, :] = (
        u[:, :, 2:, :]
        - 2.0 * u[:, :, 1:-1, :]
        + u[:, :, :-2, :]
    ) / dy2

    if W >= 4:

        uyy[:, :, 0, :] = (
            2.0 * u[:, :, 0, :]
            - 5.0 * u[:, :, 1, :]
            + 4.0 * u[:, :, 2, :]
            - u[:, :, 3, :]
        ) / dy2

        uyy[:, :, -1, :] = (
            2.0 * u[:, :, -1, :]
            - 5.0 * u[:, :, -2, :]
            + 4.0 * u[:, :, -3, :]
            - u[:, :, -4, :]
        ) / dy2

    else:

        uyy[:, :, 0, :] = uyy[:, :, 1, :]
        uyy[:, :, -1, :] = uyy[:, :, -2, :]

    laplacian = uxx + uyy

    return uxx, uyy, laplacian


# ============================================================
# PDE FUNCTIONAL BASE CLASS
# ============================================================

class PDEFunctional(nn.Module):
    """
    Base interface for PDE variational functionals.

    Each PDE defines its own pointwise energy density:

        Pi[u] = Integral_Omega L(u, grad(u), ...; params) dOmega

    The general engine handles:
        - numerical derivatives
        - numerical integration
        - batch-wise functional evaluation

    Each subclass only defines the PDE-specific density.
    """

    def __init__(self):
        super().__init__()

    def energy_density(self, fields, params=None):
        raise NotImplementedError(
            "Each PDEFunctional must implement energy_density()."
        )

    def boundary_energy(self, fields, params=None):
        return None


# ============================================================
# GENERAL VARIATIONAL ENGINE
# ============================================================

class GeneralVariationalLoss(nn.Module):
    """
    General PDE variational engine.

    Crucial invariant:

        compute_functional(...)

    ALWAYS returns one variational functional value per sample:

        [Pi_1, Pi_2, ..., Pi_B]

    Batch reduction happens only in forward().
    """

    def __init__(
        self,
        functional,
        spatial_dim,
        dx=1.0,
        dy=1.0,
        dt=1.0,
        reduction="mean",
        mode="functional",
        eps=1e-8,
    ):
        super().__init__()

        if not isinstance(functional, PDEFunctional):
            raise TypeError(
                "functional must inherit from PDEFunctional."
            )

        if spatial_dim not in (1, 2):
            raise ValueError(
                "spatial_dim must be 1 or 2."
            )

        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                "reduction must be 'mean', 'sum', or 'none'."
            )

        if mode not in (
            "functional",
            "energy_gap",
            "relative_energy_gap",
        ):
            raise ValueError(
                "Invalid variational mode."
            )

        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.functional = functional
        self.spatial_dim = spatial_dim

        self.dx = float(dx)
        self.dy = float(dy)
        self.dt = float(dt)

        self.reduction = reduction
        self.mode = mode
        self.eps = eps

    # ========================================================
    # DIFFERENTIAL FIELDS
    # ========================================================

    def compute_fields(
        self,
        prediction,
        previous_state=None,
    ):

        fields = {
            "u": prediction,
            "u_prev": previous_state,
            "dx": self.dx,
            "dy": self.dy,
            "dt": self.dt,
        }

        if previous_state is not None:

            if previous_state.shape != prediction.shape:
                raise ValueError(
                    "previous_state and prediction must "
                    "have identical shapes."
                )

            fields["ut"] = (
                prediction - previous_state
            ) / self.dt

        else:

            fields["ut"] = None

        if self.spatial_dim == 1:

            if prediction.ndim != 3:
                raise ValueError(
                    "Expected 1D prediction [B, N, C]."
                )

            ux = first_derivative_1d(
                prediction,
                self.dx,
            )

            uxx = second_derivative_1d(
                prediction,
                self.dx,
            )

            fields.update({
                "ux": ux,
                "uy": None,
                "uxx": uxx,
                "uyy": None,
                "laplacian": uxx,
            })

        else:

            if prediction.ndim != 4:
                raise ValueError(
                    "Expected 2D prediction [B, H, W, C]."
                )

            ux, uy = first_derivatives_2d(
                prediction,
                self.dx,
                self.dy,
            )

            uxx, uyy, laplacian = second_derivatives_2d(
                prediction,
                self.dx,
                self.dy,
            )

            fields.update({
                "ux": ux,
                "uy": uy,
                "uxx": uxx,
                "uyy": uyy,
                "laplacian": laplacian,
            })

        return fields

    # ========================================================
    # INTEGRATION
    # ========================================================

    def integrate(self, integrand):

        if self.spatial_dim == 1:

            return integrate_1d(
                integrand,
                self.dx,
            )

        return integrate_2d(
            integrand,
            self.dx,
            self.dy,
        )

    # ========================================================
    # FUNCTIONAL PER SAMPLE
    # ========================================================

    def compute_functional(
        self,
        prediction,
        previous_state=None,
        params=None,
        return_components=False,
    ):
        """
        Compute:

            Pi_i[u]

        independently for every batch sample.

        Returns:
            Tensor [B]
        """

        fields = self.compute_fields(
            prediction=prediction,
            previous_state=previous_state,
        )

        energy_density = self.functional.energy_density(
            fields=fields,
            params=params,
        )

        if not torch.is_tensor(energy_density):
            raise TypeError(
                "energy_density must return a tensor."
            )

        if energy_density.shape != prediction.shape:

            try:
                energy_density = energy_density.expand_as(
                    prediction
                )

            except RuntimeError as error:

                raise ValueError(
                    "energy_density must be broadcastable "
                    "to prediction."
                ) from error

        interior_energy = self.integrate(
            energy_density
        )

        if interior_energy.ndim != 1:
            raise RuntimeError(
                "Integration must return [B]."
            )

        boundary_energy = self.functional.boundary_energy(
            fields=fields,
            params=params,
        )

        if boundary_energy is None:

            boundary_energy = torch.zeros_like(
                interior_energy
            )

        elif boundary_energy.ndim == 0:

            boundary_energy = boundary_energy.expand_as(
                interior_energy
            )

        elif boundary_energy.ndim != 1:

            raise ValueError(
                "boundary_energy must be scalar or [B]."
            )

        if boundary_energy.shape != interior_energy.shape:
            raise ValueError(
                "Boundary and interior energies must "
                "have identical shapes."
            )

        total_energy = (
            interior_energy + boundary_energy
        )

        if return_components:

            return total_energy, {
                "interior": interior_energy,
                "boundary": boundary_energy,
                "energy_density": energy_density,
            }

        return total_energy

    # ========================================================
    # REDUCTION
    # ========================================================

    def reduce_loss(self, values):

        if self.reduction == "mean":
            return values.mean()

        if self.reduction == "sum":
            return values.sum()

        return values

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        prediction,
        previous_state=None,
        params=None,
        reference=None,
        reference_previous_state=None,
        reference_params=None,
        return_components=False,
    ):

        prediction_energy = self.compute_functional(
            prediction=prediction,
            previous_state=previous_state,
            params=params,
        )

        if self.mode == "functional":

            loss_per_sample = prediction_energy

            loss = self.reduce_loss(
                loss_per_sample
            )

            if return_components:

                return loss, {
                    "prediction_energy": prediction_energy,
                    "reference_energy": None,
                    "loss_per_sample": loss_per_sample,
                }

            return loss

        if reference is None:
            raise ValueError(
                f"mode='{self.mode}' requires reference."
            )

        if reference.shape != prediction.shape:
            raise ValueError(
                "prediction and reference must have "
                "identical shapes."
            )

        if reference_params is None:
            reference_params = params

        reference_energy = self.compute_functional(
            prediction=reference,
            previous_state=reference_previous_state,
            params=reference_params,
        )

        energy_difference = (
            prediction_energy - reference_energy
        )

        if self.mode == "energy_gap":

            loss_per_sample = energy_difference.pow(2)

        elif self.mode == "relative_energy_gap":

            denominator = (
                reference_energy.abs() + self.eps
            )

            loss_per_sample = (
                energy_difference / denominator
            ).pow(2)

        else:

            raise RuntimeError(
                f"Unexpected mode: {self.mode}"
            )

        loss = self.reduce_loss(
            loss_per_sample
        )

        if return_components:

            return loss, {
                "prediction_energy": prediction_energy,
                "reference_energy": reference_energy,
                "energy_difference": energy_difference,
                "loss_per_sample": loss_per_sample,
            }

        return loss
