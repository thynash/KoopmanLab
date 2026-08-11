from koopmanlab.models import kno
from koopmanlab import utils

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from timeit import default_timer


# ============================================================
# Variational Navier-Stokes Energy
# ============================================================

class NavierStokesVariational:
    """
    Variational / weak formulation for 2D incompressible
    Navier-Stokes in vorticity form.

    PDE:

        omega_t + u . grad(omega)
            = nu * Delta(omega) + f

    or

        omega_t + u . grad(omega)
        - nu * Delta(omega) - f = 0

    Instead of minimizing the strong-form pointwise residual,
    we construct weak-form residuals using test functions.

    We also include an energy-balance term.

    NOTE:
    This is deliberately different from NavierStokesLoss used
    in model_physics.py.
    """

    def __init__(
        self,
        nu=1e-5,
        dx=1.0 / 64,
        dy=1.0 / 64,
        dt=1.0,
        n_test=8,
        lambda_weak=1.0,
        lambda_energy=0.1,
    ):

        self.nu = nu
        self.dx = dx
        self.dy = dy
        self.dt = dt

        self.n_test = n_test

        self.lambda_weak = lambda_weak
        self.lambda_energy = lambda_energy

        self.test_functions = None

    # --------------------------------------------------------
    # Periodic derivative
    # --------------------------------------------------------

    def dx_periodic(self, u):

        return (
            torch.roll(u, -1, dims=-3)
            -
            torch.roll(u, 1, dims=-3)
        ) / (2.0 * self.dx)

    def dy_periodic(self, u):

        return (
            torch.roll(u, -1, dims=-2)
            -
            torch.roll(u, 1, dims=-2)
        ) / (2.0 * self.dy)

    def laplacian_periodic(self, u):

        u_xx = (
            torch.roll(u, -1, dims=-3)
            - 2.0 * u
            + torch.roll(u, 1, dims=-3)
        ) / (self.dx ** 2)

        u_yy = (
            torch.roll(u, -1, dims=-2)
            - 2.0 * u
            + torch.roll(u, 1, dims=-2)
        ) / (self.dy ** 2)

        return u_xx + u_yy

    # --------------------------------------------------------
    # Build Fourier test functions
    # --------------------------------------------------------

    def build_test_functions(self, shape, device, dtype):

        nx = shape[-3]
        ny = shape[-2]

        x = torch.arange(
            nx,
            device=device,
            dtype=dtype
        ) * self.dx

        y = torch.arange(
            ny,
            device=device,
            dtype=dtype
        ) * self.dy

        X, Y = torch.meshgrid(
            x,
            y,
            indexing="ij"
        )

        tests = []

        # Low-frequency periodic Fourier test functions.
        #
        # These are useful because the Navier-Stokes dataset
        # is periodic and they form a natural weak test basis.

        max_mode = int(np.sqrt(self.n_test)) + 1

        for kx in range(0, max_mode):

            for ky in range(0, max_mode):

                if kx == 0 and ky == 0:
                    continue

                if len(tests) >= self.n_test:
                    break

                phase = (
                    2.0
                    * np.pi
                    * (
                        kx * X / (nx * self.dx)
                        +
                        ky * Y / (ny * self.dy)
                    )
                )

                tests.append(torch.sin(phase))

            if len(tests) >= self.n_test:
                break

        self.test_functions = torch.stack(tests, dim=0)

        return self.test_functions

    # --------------------------------------------------------
    # Recover velocity from vorticity
    #
    # For 2D incompressible flow:
    #
    #     u = d psi / dy
    #     v = -d psi / dx
    #
    # and
    #
    #     omega = -Delta psi
    #
    # --------------------------------------------------------

    def velocity_from_vorticity(self, omega):

        nx = omega.shape[-3]
        ny = omega.shape[-2]

        device = omega.device
        dtype = omega.dtype

        # Fourier frequencies
        kx = torch.fft.fftfreq(
            nx,
            d=self.dx,
            device=device
        ) * 2.0 * np.pi

        ky = torch.fft.fftfreq(
            ny,
            d=self.dy,
            device=device
        ) * 2.0 * np.pi

        KX, KY = torch.meshgrid(
            kx,
            ky,
            indexing="ij"
        )

        KX = KX.to(dtype)
        KY = KY.to(dtype)

        K2 = KX ** 2 + KY ** 2

        # Avoid zero division
        K2 = K2.clone()

        K2[0, 0] = 1.0

        omega_hat = torch.fft.fft2(
            omega.squeeze(-1)
        )

        # omega = -Delta psi
        #
        # Fourier:
        #
        # omega_hat = K^2 psi_hat
        #
        psi_hat = omega_hat / K2

        # Zero mean streamfunction mode
        psi_hat[..., 0, 0] = 0.0

        # u = d psi / dy
        u_hat = 1j * KY * psi_hat

        # v = -d psi / dx
        v_hat = -1j * KX * psi_hat

        u = torch.fft.ifft2(u_hat).real
        v = torch.fft.ifft2(v_hat).real

        return u.unsqueeze(-1), v.unsqueeze(-1)

    # --------------------------------------------------------
    # Weak residual
    # --------------------------------------------------------

    def weak_residual(
        self,
        current,
        predicted
    ):

        # ----------------------------------------------------
        # current / predicted:
        #
        # [B, Nx, Ny, 1]
        # ----------------------------------------------------

        omega_t = (
            predicted - current
        ) / self.dt

        omega = predicted

        # ----------------------------------------------------
        # Velocity from vorticity
        # ----------------------------------------------------

        u, v = self.velocity_from_vorticity(
            omega
        )

        # ----------------------------------------------------
        # Vorticity gradients
        # ----------------------------------------------------

        omega_x = self.dx_periodic(
            omega
        )

        omega_y = self.dy_periodic(
            omega
        )

        # nonlinear convection
        convection = (
            u * omega_x
            +
            v * omega_y
        )

        # ----------------------------------------------------
        # Test functions
        # ----------------------------------------------------

        tests = self.build_test_functions(
            omega.shape,
            omega.device,
            omega.dtype
        )

        residuals = []

        for phi in tests:

            phi = phi.unsqueeze(0).unsqueeze(-1)

            # ------------------------------------------------
            # Weak form:
            #
            # ∫ [
            #       omega_t
            #       + u.grad(omega)
            # ] phi dx
            #
            # + nu ∫ grad(omega).grad(phi) dx
            #
            # = 0
            #
            # Diffusion term is integrated by parts.
            # ------------------------------------------------

            phi_x = self.dx_periodic(
                phi
            )

            phi_y = self.dy_periodic(
                phi
            )

            temporal_convective = (
                omega_t
                +
                convection
            )

            diffusion_weak = (
                self.nu
                *
                (
                    omega_x * phi_x
                    +
                    omega_y * phi_y
                )
            )

            integrand = (
                temporal_convective * phi
                +
                diffusion_weak
            )

            integral = (
                integrand.mean(
                    dim=(-3, -2)
                )
            )

            residuals.append(
                integral
            )

        residuals = torch.stack(
            residuals,
            dim=0
        )

        return residuals

    # --------------------------------------------------------
    # Weak variational loss
    # --------------------------------------------------------

    def weak_loss(
        self,
        current,
        predicted
    ):

        residual = self.weak_residual(
            current,
            predicted
        )

        return torch.mean(
            residual ** 2
        )

    # --------------------------------------------------------
    # Kinetic energy
    #
    # E = 1/2 ∫ |u|² dx
    # --------------------------------------------------------

    def kinetic_energy(
        self,
        omega
    ):

        u, v = self.velocity_from_vorticity(
            omega
        )

        energy_density = (
            0.5
            *
            (
                u ** 2
                +
                v ** 2
            )
        )

        return energy_density.mean(
            dim=(-3, -2)
        )

    # --------------------------------------------------------
    # Viscous dissipation
    #
    # D = nu ∫ |grad u|² dx
    # --------------------------------------------------------

    def viscous_dissipation(
        self,
        omega
    ):

        u, v = self.velocity_from_vorticity(
            omega
        )

        ux = self.dx_periodic(u)
        uy = self.dy_periodic(u)

        vx = self.dx_periodic(v)
        vy = self.dy_periodic(v)

        dissipation = self.nu * (
            ux ** 2
            +
            uy ** 2
            +
            vx ** 2
            +
            vy ** 2
        )

        return dissipation.mean(
            dim=(-3, -2)
        )

    # --------------------------------------------------------
    # Energy balance
    #
    # dE/dt + D = P
    #
    # For the unforced case:
    #
    # dE/dt + D = 0
    # --------------------------------------------------------

    def energy_loss(
        self,
        current,
        predicted
    ):

        E_current = self.kinetic_energy(
            current
        )

        E_predicted = self.kinetic_energy(
            predicted
        )

        dE_dt = (
            E_predicted
            -
            E_current
        ) / self.dt

        D = self.viscous_dissipation(
            predicted
        )

        balance = dE_dt + D

        return torch.mean(
            balance ** 2
        )

    # --------------------------------------------------------
    # Total variational loss
    # --------------------------------------------------------

    def __call__(
        self,
        current,
        predicted
    ):

        l_weak = self.weak_loss(
            current,
            predicted
        )

        l_energy = self.energy_loss(
            current,
            predicted
        )

        total = (
            self.lambda_weak * l_weak
            +
            self.lambda_energy * l_energy
        )

        return total, l_weak, l_energy


# ============================================================
# Variational Koopman Neural Operator
# ============================================================

class koopman:

    def __init__(
        self,
        backbone="KNO1d",
        autoencoder="MLP",
        o=16,
        m=16,
        r=8,
        t_in=1,
        device=False,

        # ----------------------------------------------------
        # Variational parameters
        # ----------------------------------------------------

        lambda_var=1.0,
        lambda_pred=5.0,
        lambda_recon=0.5,

        nu=1e-5,
        dx=1/64,
        dy=1/64,
        dt=1.0,

        n_test=8,
        lambda_weak=1.0,
        lambda_energy=0.1,
    ):

        self.backbone = backbone
        self.autoencoder = autoencoder

        self.operator_size = o
        self.modes = m
        self.decompose = r

        self.device = device
        self.t_in = t_in

        # ----------------------------------------------------
        # Loss weights
        # ----------------------------------------------------

        self.lambda_var = lambda_var
        self.lambda_pred = lambda_pred
        self.lambda_recon = lambda_recon

        # ----------------------------------------------------
        # Navier-Stokes variational physics
        # ----------------------------------------------------

        self.variational = NavierStokesVariational(
            nu=nu,
            dx=dx,
            dy=dy,
            dt=dt,
            n_test=n_test,
            lambda_weak=lambda_weak,
            lambda_energy=lambda_energy,
        )

        # ----------------------------------------------------
        # Core model
        # ----------------------------------------------------

        self.params = 0
        self.kernel = False

        # ----------------------------------------------------
        # Optimization
        # ----------------------------------------------------

        self.optimizer = False
        self.scheduler = None

        self.loss = torch.nn.MSELoss()

    # ========================================================
    # Compile
    # ========================================================

    def compile(self):

        if self.autoencoder == "MLP":

            encoder = kno.encoder_mlp(
                self.t_in,
                self.operator_size
            )

            decoder = kno.decoder_mlp(
                self.t_in,
                self.operator_size
            )

            print(
                "The autoencoder type is MLP."
            )

        elif self.autoencoder == "Conv1d":

            encoder = kno.encoder_conv1d(
                self.t_in,
                self.operator_size
            )

            decoder = kno.decoder_conv1d(
                self.t_in,
                self.operator_size
            )

            print(
                "The autoencoder type is Conv1d."
            )

        elif self.autoencoder == "Conv2d":

            encoder = kno.encoder_conv2d(
                self.t_in,
                self.operator_size
            )

            decoder = kno.decoder_conv2d(
                self.t_in,
                self.operator_size
            )

            print(
                "The autoencoder type is Conv2d."
            )

        else:

            raise ValueError(
                "Wrong autoencoder type."
            )

        # ----------------------------------------------------
        # KNO backbone
        # ----------------------------------------------------

        if self.backbone == "KNO1d":

            self.kernel = kno.KNO1d(
                encoder,
                decoder,
                self.operator_size,
                modes_x=self.modes,
                decompose=self.decompose
            ).to(self.device)

            print(
                "KNO1d model is completed."
            )

        elif self.backbone == "KNO2d":

            self.kernel = kno.KNO2d(
                encoder,
                decoder,
                self.operator_size,
                modes_x=self.modes,
                modes_y=self.modes,
                decompose=self.decompose
            ).to(self.device)

            print(
                "KNO2d model is completed."
            )

        else:

            raise ValueError(
                "Wrong backbone type."
            )

        self.params = utils.count_params(
            self.kernel
        )

        print(
            "Variational Koopman Model has been compiled!"
        )

        print(
            "The Model Parameters Number is ",
            self.params
        )

    # ========================================================
    # Optimizer
    # ========================================================

    def opt_init(
        self,
        opt,
        lr,
        step_size,
        gamma
    ):

        if opt == "Adam":

            self.optimizer = utils.Adam(
                self.kernel.parameters(),
                lr=lr,
                weight_decay=1e-4
            )

        else:

            raise ValueError(
                "Only Adam is currently implemented."
            )

        if step_size is not False:

            self.scheduler = (
                torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=step_size,
                    gamma=gamma
                )
            )

    # ========================================================
    # One-step variational training
    # ========================================================

    def train_single(
        self,
        epochs,
        trainloader,
        evalloader=False
    ):

        for ep in range(epochs):

            self.kernel.train()

            t1 = default_timer()

            train_pred_full = 0.0
            train_recon_full = 0.0
            train_var_full = 0.0
            train_weak_full = 0.0
            train_energy_full = 0.0

            # ------------------------------------------------
            # Training
            # ------------------------------------------------

            for xx, yy in trainloader:

                xx = xx.to(self.device)
                yy = yy.to(self.device)

                bs = xx.shape[0]

                pred, im_re = self.kernel(xx)

                # Reconstruction
                l_recon = self.loss(
                    im_re.reshape(bs, -1),
                    xx.reshape(bs, -1)
                )

                # Prediction
                l_pred = self.loss(
                    pred.reshape(bs, -1),
                    yy.reshape(bs, -1)
                )

                # Current and predicted states
                current = xx[..., -1:]

                predicted = pred[..., -1:]

                # ------------------------------------------------
                # Variational physics
                # ------------------------------------------------

                (
                    l_var,
                    l_weak,
                    l_energy
                ) = self.variational(
                    current,
                    predicted
                )

                # ------------------------------------------------
                # Total loss
                # ------------------------------------------------

                loss = (
                    self.lambda_pred * l_pred
                    +
                    self.lambda_recon * l_recon
                    +
                    self.lambda_var * l_var
                )

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                train_pred_full += l_pred.item()
                train_recon_full += l_recon.item()
                train_var_full += l_var.item()
                train_weak_full += l_weak.item()
                train_energy_full += l_energy.item()

            # ------------------------------------------------
            # Average
            # ------------------------------------------------

            n = len(trainloader)

            train_pred_full /= n
            train_recon_full /= n
            train_var_full /= n
            train_weak_full /= n
            train_energy_full /= n

            t2 = default_timer()

            # ------------------------------------------------
            # Evaluation
            # ------------------------------------------------

            eval_pred = 0.0
            eval_recon = 0.0
            eval_var = 0.0
            eval_weak = 0.0
            eval_energy = 0.0

            if evalloader:

                self.kernel.eval()

                with torch.no_grad():

                    for xx, yy in evalloader:

                        xx = xx.to(self.device)
                        yy = yy.to(self.device)

                        bs = xx.shape[0]

                        pred, im_re = self.kernel(xx)

                        l_recon = self.loss(
                            im_re.reshape(bs, -1),
                            xx.reshape(bs, -1)
                        )

                        l_pred = self.loss(
                            pred.reshape(bs, -1),
                            yy.reshape(bs, -1)
                        )

                        current = xx[..., -1:]

                        predicted = pred[..., -1:]

                        (
                            l_var,
                            l_weak,
                            l_energy
                        ) = self.variational(
                            current,
                            predicted
                        )

                        eval_pred += l_pred.item()
                        eval_recon += l_recon.item()
                        eval_var += l_var.item()
                        eval_weak += l_weak.item()
                        eval_energy += l_energy.item()

                n_eval = len(evalloader)

                eval_pred /= n_eval
                eval_recon /= n_eval
                eval_var /= n_eval
                eval_weak /= n_eval
                eval_energy /= n_eval

            # ------------------------------------------------
            # Scheduler
            # ------------------------------------------------

            if self.scheduler is not None:

                self.scheduler.step()

            # ------------------------------------------------
            # Logging
            # ------------------------------------------------

            if ep == 0:

                if evalloader:

                    print(
                        "Epoch",
                        "Time",
                        "Train Pred",
                        "Train Recon",
                        "Train Var",
                        "Train Weak",
                        "Train Energy",
                        "Eval Pred",
                        "Eval Recon",
                        "Eval Var",
                        "Eval Weak",
                        "Eval Energy"
                    )

                else:

                    print(
                        "Epoch",
                        "Time",
                        "Train Pred",
                        "Train Recon",
                        "Train Var",
                        "Train Weak",
                        "Train Energy"
                    )

            if evalloader:

                print(
                    ep,
                    t2 - t1,
                    train_pred_full,
                    train_recon_full,
                    train_var_full,
                    train_weak_full,
                    train_energy_full,
                    eval_pred,
                    eval_recon,
                    eval_var,
                    eval_weak,
                    eval_energy
                )

            else:

                print(
                    ep,
                    t2 - t1,
                    train_pred_full,
                    train_recon_full,
                    train_var_full,
                    train_weak_full,
                    train_energy_full
                )

    # ========================================================
    # Autoregressive variational training
    # ========================================================

    def train(
        self,
        epochs,
        trainloader,
        step=1,
        T_out=40,
        evalloader=False
    ):

        for ep in range(epochs):

            self.kernel.train()

            t1 = default_timer()

            train_pred_full = 0.0
            train_recon_full = 0.0
            train_var_full = 0.0
            train_weak_full = 0.0
            train_energy_full = 0.0

            for xx, yy in trainloader:

                xx = xx.to(self.device)
                yy = yy.to(self.device)

                bs = xx.shape[0]

                l_recon = 0.0
                l_var = 0.0
                l_weak = 0.0
                l_energy = 0.0

                predictions = []

                # ------------------------------------------------
                # Autoregressive rollout
                # ------------------------------------------------

                for t in range(T_out):

                    im, im_re = self.kernel(xx)

                    # Reconstruction
                    l_recon += self.loss(
                        im_re.reshape(bs, -1),
                        xx.reshape(bs, -1)
                    )

                    current = xx[..., -1:]

                    predicted = im[..., -1:]

                    # ------------------------------------------------
                    # Variational loss for this transition
                    # ------------------------------------------------

                    (
                        l_var_step,
                        l_weak_step,
                        l_energy_step
                    ) = self.variational(
                        current,
                        predicted
                    )

                    l_var += l_var_step
                    l_weak += l_weak_step
                    l_energy += l_energy_step

                    predictions.append(
                        predicted
                    )

                    # ------------------------------------------------
                    # Shift temporal window
                    # ------------------------------------------------

                    xx = torch.cat(
                        (
                            xx[..., step:],
                            predicted
                        ),
                        dim=-1
                    )

                # ------------------------------------------------
                # Combine predictions
                # ------------------------------------------------

                pred = torch.cat(
                    predictions,
                    dim=-1
                )

                # ------------------------------------------------
                # Average physics terms
                # ------------------------------------------------

                l_recon /= T_out
                l_var /= T_out
                l_weak /= T_out
                l_energy /= T_out

                # ------------------------------------------------
                # Supervised prediction
                # ------------------------------------------------

                l_pred = self.loss(
                    pred.reshape(bs, -1),
                    yy.reshape(bs, -1)
                )

                # ------------------------------------------------
                # Total loss
                # ------------------------------------------------

                loss = (
                    self.lambda_pred * l_pred
                    +
                    self.lambda_recon * l_recon
                    +
                    self.lambda_var * l_var
                )

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                train_pred_full += l_pred.item()
                train_recon_full += l_recon.item()
                train_var_full += l_var.item()
                train_weak_full += l_weak.item()
                train_energy_full += l_energy.item()

            n = len(trainloader)

            train_pred_full /= n
            train_recon_full /= n
            train_var_full /= n
            train_weak_full /= n
            train_energy_full /= n

            t2 = default_timer()

            # ------------------------------------------------
            # Evaluation
            # ------------------------------------------------

            eval_pred = 0.0
            eval_recon = 0.0
            eval_var = 0.0
            eval_weak = 0.0
            eval_energy = 0.0

            if evalloader:

                self.kernel.eval()

                with torch.no_grad():

                    for xx, yy in evalloader:

                        xx = xx.to(self.device)
                        yy = yy.to(self.device)

                        bs = xx.shape[0]

                        l_recon = 0.0
                        l_var = 0.0
                        l_weak = 0.0
                        l_energy = 0.0

                        predictions = []

                        for t in range(T_out):

                            im, im_re = self.kernel(xx)

                            l_recon += self.loss(
                                im_re.reshape(bs, -1),
                                xx.reshape(bs, -1)
                            )

                            current = xx[..., -1:]

                            predicted = im[..., -1:]

                            (
                                l_var_step,
                                l_weak_step,
                                l_energy_step
                            ) = self.variational(
                                current,
                                predicted
                            )

                            l_var += l_var_step
                            l_weak += l_weak_step
                            l_energy += l_energy_step

                            predictions.append(
                                predicted
                            )

                            xx = torch.cat(
                                (
                                    xx[..., 1:],
                                    predicted
                                ),
                                dim=-1
                            )

                        pred = torch.cat(
                            predictions,
                            dim=-1
                        )

                        l_recon /= T_out
                        l_var /= T_out
                        l_weak /= T_out
                        l_energy /= T_out

                        l_pred = self.loss(
                            pred.reshape(bs, -1),
                            yy.reshape(bs, -1)
                        )

                        eval_pred += l_pred.item()
                        eval_recon += l_recon.item()
                        eval_var += l_var.item()
                        eval_weak += l_weak.item()
                        eval_energy += l_energy.item()

                n_eval = len(evalloader)

                eval_pred /= n_eval
                eval_recon /= n_eval
                eval_var /= n_eval
                eval_weak /= n_eval
                eval_energy /= n_eval

            if self.scheduler is not None:

                self.scheduler.step()

            # ------------------------------------------------
            # Print
            # ------------------------------------------------

            if ep == 0:

                print(
                    "Epoch",
                    "Time",
                    "Train Pred",
                    "Train Recon",
                    "Train Var",
                    "Train Weak",
                    "Train Energy",
                    "Eval Pred",
                    "Eval Recon",
                    "Eval Var",
                    "Eval Weak",
                    "Eval Energy"
                )

            print(
                ep,
                t2 - t1,
                train_pred_full,
                train_recon_full,
                train_var_full,
                train_weak_full,
                train_energy_full,
                eval_pred,
                eval_recon,
                eval_var,
                eval_weak,
                eval_energy
            )

    # ========================================================
    # Test
    # ========================================================

    def test(
        self,
        testloader,
        step=1,
        T_out=40,
        path=False,
        is_save=False,
        is_plot=False
    ):

        time_error = torch.zeros(
            [T_out, 1]
        )

        test_pred_full = 0.0
        test_recon_full = 0.0
        test_var_full = 0.0
        test_weak_full = 0.0
        test_energy_full = 0.0

        loc = 0

        self.kernel.eval()

        with torch.no_grad():

            for xx, yy in testloader:

                bs = xx.shape[0]

                xx = xx.to(self.device)
                yy = yy.to(self.device)

                l_recon = 0.0
                l_var = 0.0
                l_weak = 0.0
                l_energy = 0.0

                predictions = []

                for t in range(T_out):

                    y = yy[..., t:t + 1]

                    im, im_re = self.kernel(xx)

                    l_recon += self.loss(
                        im_re.reshape(bs, -1),
                        xx.reshape(bs, -1)
                    )

                    predicted = im[..., -1:]

                    current = xx[..., -1:]

                    (
                        l_var_step,
                        l_weak_step,
                        l_energy_step
                    ) = self.variational(
                        current,
                        predicted
                    )

                    l_var += l_var_step
                    l_weak += l_weak_step
                    l_energy += l_energy_step

                    t_error = self.loss(
                        predicted,
                        y
                    )

                    time_error[t] += (
                        t_error.item()
                    )

                    predictions.append(
                        predicted
                    )

                    xx = torch.cat(
                        (
                            xx[..., 1:],
                            predicted
                        ),
                        dim=-1
                    )

                pred = torch.cat(
                    predictions,
                    dim=-1
                )

                l_recon /= T_out
                l_var /= T_out
                l_weak /= T_out
                l_energy /= T_out

                l_pred = self.loss(
                    pred.reshape(bs, -1),
                    yy.reshape(bs, -1)
                )

                test_pred_full += l_pred.item()
                test_recon_full += l_recon.item()
                test_var_full += l_var.item()
                test_weak_full += l_weak.item()
                test_energy_full += l_energy.item()

                # ------------------------------------------------
                # Save
                # ------------------------------------------------

                if loc == 0 and is_save:

                    torch.save(
                        {
                            "pred": pred,
                            "yy": yy
                        },
                        path + "pred_yy.pt"
                    )

                # ------------------------------------------------
                # Plot
                # ------------------------------------------------

                if loc == 0 and is_plot:

                    for i in range(T_out):

                        plt.figure(
                            figsize=(12, 4)
                        )

                        plt.subplot(1, 3, 1)

                        plt.title(
                            "Prediction"
                        )

                        plt.imshow(
                            pred[
                                0, ..., i
                            ]
                            .cpu()
                            .detach()
                            .numpy()
                        )

                        plt.subplot(1, 3, 2)

                        plt.title(
                            "Ground Truth"
                        )

                        plt.imshow(
                            yy[
                                0, ..., i
                            ]
                            .cpu()
                            .detach()
                            .numpy()
                        )

                        plt.subplot(1, 3, 3)

                        plt.title(
                            "Error"
                        )

                        plt.imshow(
                            (
                                pred[
                                    0, ..., i
                                ]
                                -
                                yy[
                                    0, ..., i
                                ]
                            )
                            .cpu()
                            .detach()
                            .numpy()
                        )

                        plt.tight_layout()

                        plt.savefig(
                            path
                            +
                            "time_"
                            +
                            str(i)
                            +
                            ".png"
                        )

                        plt.close()

                loc += 1

        test_pred_full /= loc
        test_recon_full /= loc
        test_var_full /= loc
        test_weak_full /= loc
        test_energy_full /= loc

        time_error /= len(testloader)

        print(
            "Total prediction test mse error is ",
            test_pred_full
        )

        print(
            "Total reconstruction test mse error is ",
            test_recon_full
        )

        print(
            "Total variational test loss is ",
            test_var_full
        )

        print(
            "Total weak-form test loss is ",
            test_weak_full
        )

        print(
            "Total energy-balance test loss is ",
            test_energy_full
        )

        return time_error

    # ========================================================
    # Save
    # ========================================================

    def save(self, path):

        fpath, _ = os.path.split(path)

        if fpath and not os.path.isdir(fpath):

            os.makedirs(fpath)

        torch.save(
            {
                "koopman": self,
                "model": self.kernel,
                "model_params": self.kernel.state_dict()
            },
            path
        )
