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
# 1D VISCOUS BURGERS EQUATION
# ============================================================

class BurgersFunctional(PDEFunctional):
    """
    Least-squares variational functional for the 1D
    viscous Burgers equation:

        u_t + u u_x - nu u_xx = 0

    Since Burgers is a nonlinear time-dependent PDE, we use
    a residual-based least-squares functional:

        Pi[u] =
            Integral
            1/2 * (
                u_t + u u_x - nu u_xx
            )^2 dx

    Minimizing this functional drives the PDE residual toward
    zero.

    Required fields:
        u
        ut
        ux
        uxx

    Required params:
        {
            "nu": viscosity
        }

    The output has the same pointwise shape as u, allowing
    GeneralVariationalLoss to perform spatial integration.
    """

    def energy_density(self, fields, params=None):

        # ----------------------------------------------------
        # VALIDATE PDE PARAMETERS
        # ----------------------------------------------------

        if params is None:
            raise ValueError(
                "BurgersFunctional requires PDE parameters."
            )

        if "nu" not in params:
            raise ValueError(
                "BurgersFunctional requires params['nu']."
            )

        # ----------------------------------------------------
        # REQUIRED FIELDS
        # ----------------------------------------------------

        u = fields["u"]
        ut = fields["ut"]
        ux = fields["ux"]
        uxx = fields["uxx"]

        # Burgers experiment is 1D.
        if ux is None:
            raise ValueError(
                "BurgersFunctional requires spatial derivative "
                "fields['ux']."
            )

        if uxx is None:
            raise ValueError(
                "BurgersFunctional requires second spatial "
                "derivative fields['uxx']."
            )

        # u_t is obtained from the current prediction and
        # previous_state supplied to GeneralVariationalLoss.
        if ut is None:
            raise ValueError(
                "BurgersFunctional requires fields['ut']. "
                "Provide previous_state so the temporal "
                "derivative can be computed."
            )

        # ----------------------------------------------------
        # VISCOSITY
        # ----------------------------------------------------

        nu = params["nu"]

        if not torch.is_tensor(nu):
            nu = torch.tensor(
                nu,
                device=u.device,
                dtype=u.dtype,
            )
        else:
            nu = nu.to(
                device=u.device,
                dtype=u.dtype,
            )

        # ----------------------------------------------------
        # BURGERS PDE RESIDUAL
        #
        # R[u] = u_t + u u_x - nu u_xx
        # ----------------------------------------------------

        residual = (
            ut
            + u * ux
            - nu * uxx
        )

        # ----------------------------------------------------
        # LEAST-SQUARES VARIATIONAL DENSITY
        #
        # E[u] = 1/2 R[u]^2
        # ----------------------------------------------------

        return (
            0.5
            * residual.pow(2)
        )
# ============================================================
# 1D ALLEN-CAHN EQUATION
# ============================================================

class AllenCahnFunctional(PDEFunctional):
    """
    Variational energy functional for the 1D Allen-Cahn equation.

    Allen-Cahn equation:

        u_t = epsilon^2 u_xx - (u^3 - u)

    or equivalently:

        u_t = -delta E / delta u

    where E[u] is the Ginzburg-Landau free energy:

        E[u] =
            Integral_Omega [
                (epsilon^2 / 2) * |u_x|^2
                +
                (1 / 4) * (u^2 - 1)^2
            ] dx

    This is the natural energy functional associated with the
    Allen-Cahn gradient-flow equation.

    IMPORTANT
    ---------
    PEDVINO learns the operator:

        u_0(x) -> u(x, T)

    Therefore the model predicts a single state at the target
    time rather than an explicit time trajectory. Consequently,
    this functional intentionally DOES NOT require u_t.

    Required fields:
        u
        ux

    Required params:
        {
            "epsilon": epsilon
        }

    Alternative accepted parameter name:
        "eps"

    The returned energy density has the same spatial shape as u.
    GeneralVariationalLoss performs the spatial integration.
    """

    def energy_density(self, fields, params=None):

        # ====================================================
        # VALIDATE INPUT
        # ====================================================

        if params is None:
            raise ValueError(
                "AllenCahnFunctional requires PDE parameters."
            )

        # ====================================================
        # REQUIRED FIELDS
        # ====================================================

        if "u" not in fields:
            raise ValueError(
                "AllenCahnFunctional requires fields['u']."
            )

        if "ux" not in fields:
            raise ValueError(
                "AllenCahnFunctional requires fields['ux']."
            )

        u = fields["u"]
        ux = fields["ux"]

        if ux is None:
            raise ValueError(
                "AllenCahnFunctional requires the spatial "
                "derivative fields['ux']."
            )

        # ====================================================
        # EPSILON PARAMETER
        # ====================================================

        if "epsilon" in params:

            epsilon = params["epsilon"]

        elif "eps" in params:

            epsilon = params["eps"]

        else:

            raise ValueError(
                "AllenCahnFunctional requires either "
                "params['epsilon'] or params['eps']."
            )

        # Convert scalar epsilon to tensor if necessary.
        if not torch.is_tensor(epsilon):

            epsilon = torch.tensor(
                float(epsilon),
                device=u.device,
                dtype=u.dtype,
            )

        else:

            epsilon = epsilon.to(
                device=u.device,
                dtype=u.dtype,
            )

        # ====================================================
        # GRADIENT ENERGY
        #
        # (epsilon^2 / 2) * (u_x)^2
        # ====================================================

        gradient_energy = (
            0.5
            * epsilon.pow(2)
            * ux.pow(2)
        )

        # ====================================================
        # DOUBLE-WELL POTENTIAL
        #
        # 1/4 * (u^2 - 1)^2
        # ====================================================

        potential_energy = (
            0.25
            * (u.pow(2) - 1.0).pow(2)
        )

        # ====================================================
        # TOTAL GINZBURG-LANDAU ENERGY DENSITY
        #
        # e[u] =
        #
        # epsilon^2 / 2 * |u_x|^2
        # +
        # 1 / 4 * (u^2 - 1)^2
        # ====================================================

        energy_density = (
            gradient_energy
            + potential_energy
        )

        return energy_density
