import torch

from koopmanlab.variational_loss import PDEFunctional


# ============================================================
# POISSON PDE
# ============================================================

class PoissonFunctional(PDEFunctional):
    """
    Variational functional for the Poisson equation:

        -Delta(u) = f

    The corresponding energy functional is:

        Pi[u] =
            Integral_Omega [
                1/2 |grad(u)|^2 - f*u
            ] dOmega

    For 1D:

        Pi[u] =
            Integral [
                1/2 (u_x)^2 - f*u
            ] dx

    For 2D:

        Pi[u] =
            Integral [
                1/2 (u_x^2 + u_y^2) - f*u
            ] dOmega
    """

    def energy_density(self, fields, params=None):
        """
        Compute the Poisson energy density.

        Required fields:
            u
            ux

        For 2D:
            uy is additionally used.

        Required params:
            {
                "forcing": f
            }

        The forcing tensor should be broadcastable to the
        prediction shape.
        """

        if params is None or "forcing" not in params:
            raise ValueError(
                "PoissonFunctional requires params['forcing']."
            )

        u = fields["u"]
        ux = fields["ux"]
        uy = fields["uy"]

        forcing = params["forcing"]

        # ----------------------------------------------------
        # 1D Poisson:
        #
        # 1/2 (u_x)^2 - f*u
        # ----------------------------------------------------
        if uy is None:
            grad_energy = 0.5 * ux.pow(2)

        # ----------------------------------------------------
        # 2D Poisson:
        #
        # 1/2 (u_x^2 + u_y^2) - f*u
        # ----------------------------------------------------
        else:
            grad_energy = 0.5 * (
                ux.pow(2) + uy.pow(2)
            )

        return grad_energy - forcing * u


# ============================================================
# DARCY FLOW
# ============================================================

class DarcyFunctional(PDEFunctional):
    """
    Variational functional for Darcy flow:

        -div(a grad(u)) = f

    where:
        a = spatial permeability/coefficient

    The energy functional is:

        Pi[u] =
            Integral_Omega [
                1/2 a |grad(u)|^2 - f*u
            ] dOmega

    Required params:
        {
            "coefficient": a,
            "forcing": f
        }
    """

    def energy_density(self, fields, params=None):

        if params is None:
            raise ValueError(
                "DarcyFunctional requires PDE parameters."
            )

        if "coefficient" not in params:
            raise ValueError(
                "DarcyFunctional requires params['coefficient']."
            )

        if "forcing" not in params:
            raise ValueError(
                "DarcyFunctional requires params['forcing']."
            )

        u = fields["u"]
        ux = fields["ux"]
        uy = fields["uy"]

        coefficient = params["coefficient"]
        forcing = params["forcing"]

        # ----------------------------------------------------
        # |grad(u)|^2
        # ----------------------------------------------------
        if uy is None:
            grad_squared = ux.pow(2)
        else:
            grad_squared = (
                ux.pow(2)
                + uy.pow(2)
            )

        # ----------------------------------------------------
        # Pi[u] = 1/2 a |grad(u)|^2 - f*u
        # ----------------------------------------------------
        return (
            0.5 * coefficient * grad_squared
            - forcing * u
        )


# ============================================================
# DIFFUSION / ELLIPTIC ENERGY
# ============================================================

class DiffusionFunctional(PDEFunctional):
    """
    Generic elliptic diffusion functional:

        -div(k grad(u)) = f

    Energy:

        Pi[u] =
            Integral_Omega [
                1/2 k |grad(u)|^2 - f*u
            ] dOmega

    This is mathematically the same general variational
    structure as Darcy, but is provided separately because
    k may represent diffusivity rather than permeability.

    Required params:
        {
            "diffusivity": k,
            "forcing": f
        }
    """

    def energy_density(self, fields, params=None):

        if params is None:
            raise ValueError(
                "DiffusionFunctional requires PDE parameters."
            )

        if "diffusivity" not in params:
            raise ValueError(
                "DiffusionFunctional requires "
                "params['diffusivity']."
            )

        if "forcing" not in params:
            raise ValueError(
                "DiffusionFunctional requires "
                "params['forcing']."
            )

        u = fields["u"]
        ux = fields["ux"]
        uy = fields["uy"]

        diffusivity = params["diffusivity"]
        forcing = params["forcing"]

        if uy is None:
            grad_squared = ux.pow(2)
        else:
            grad_squared = (
                ux.pow(2)
                + uy.pow(2)
            )

        return (
            0.5 * diffusivity * grad_squared
            - forcing * u
        )
