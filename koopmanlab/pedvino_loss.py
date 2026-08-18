import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# PEDVINO COMPOSITE LOSS
# ============================================================

class PEDVINOLoss(nn.Module):
    """
    General PEDVINO objective.

    L =
        lambda_pred   * L_prediction
      + lambda_recon  * L_reconstruction
      + lambda_energy * L_energy
      + lambda_grad   * L_gradient
      + lambda_bc     * L_boundary

    The variational engine is PDE-independent at this level.

    PDE-specific mathematics enters only through:

        GeneralVariationalLoss
            ->
        PDEFunctional.energy_density(...)
    """

    def __init__(
        self,
        variational_loss=None,
        lambda_pred=1.0,
        lambda_recon=0.0,
        lambda_energy=0.0,
        lambda_grad=0.0,
        lambda_bc=0.0,
        energy_loss_type="relative",
        energy_eps=1e-8,
        use_relative_prediction_loss=True,
        prediction_eps=1e-8,
    ):
        super().__init__()

        self.variational_loss = variational_loss

        self.lambda_pred = float(lambda_pred)
        self.lambda_recon = float(lambda_recon)
        self.lambda_energy = float(lambda_energy)
        self.lambda_grad = float(lambda_grad)
        self.lambda_bc = float(lambda_bc)

        if energy_loss_type not in (
            "absolute",
            "relative",
        ):
            raise ValueError(
                "energy_loss_type must be "
                "'absolute' or 'relative'."
            )

        self.energy_loss_type = energy_loss_type
        self.energy_eps = float(energy_eps)

        self.use_relative_prediction_loss = bool(
            use_relative_prediction_loss
        )

        self.prediction_eps = float(
            prediction_eps
        )

    # ========================================================
    # PREDICTION LOSS
    # ========================================================

    def prediction_loss(
        self,
        prediction,
        target,
    ):
        """
        Relative sample-wise L2 loss by default.

        This is more appropriate than raw MSE when different
        PDE samples have different solution magnitudes.
        """

        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target must have "
                "identical shapes."
            )

        if not self.use_relative_prediction_loss:
            return F.mse_loss(
                prediction,
                target,
            )

        batch_size = prediction.shape[0]

        error = (
            prediction - target
        ).reshape(batch_size, -1)

        target_flat = target.reshape(
            batch_size,
            -1,
        )

        numerator = error.norm(
            p=2,
            dim=1,
        )

        denominator = target_flat.norm(
            p=2,
            dim=1,
        ).clamp_min(
            self.prediction_eps
        )

        return (
            numerator / denominator
        ).mean()

    # ========================================================
    # RECONSTRUCTION LOSS
    # ========================================================

    def reconstruction_loss(
        self,
        reconstruction,
        input_field,
    ):

        if reconstruction.shape != input_field.shape:
            raise ValueError(
                "reconstruction and input_field must "
                "have identical shapes."
            )

        return F.mse_loss(
            reconstruction,
            input_field,
        )

    # ========================================================
    # GRADIENT LOSS
    # ========================================================

    def gradient_consistency_loss(
        self,
        prediction,
        target,
        spatial_dim,
        dx,
        dy,
    ):
        """
        Match first spatial derivatives using the same
        physical grid spacing used by the variational engine.
        """

        if spatial_dim == 1:

            pred_grad = (
                prediction[:, 1:, :]
                - prediction[:, :-1, :]
            ) / dx

            target_grad = (
                target[:, 1:, :]
                - target[:, :-1, :]
            ) / dx

            return F.mse_loss(
                pred_grad,
                target_grad,
            )

        if spatial_dim == 2:

            pred_dx = (
                prediction[:, 1:, :, :]
                - prediction[:, :-1, :, :]
            ) / dx

            target_dx = (
                target[:, 1:, :, :]
                - target[:, :-1, :, :]
            ) / dx

            pred_dy = (
                prediction[:, :, 1:, :]
                - prediction[:, :, :-1, :]
            ) / dy

            target_dy = (
                target[:, :, 1:, :]
                - target[:, :, :-1, :]
            ) / dy

            loss_x = F.mse_loss(
                pred_dx,
                target_dx,
            )

            loss_y = F.mse_loss(
                pred_dy,
                target_dy,
            )

            return 0.5 * (
                loss_x + loss_y
            )

        raise ValueError(
            "spatial_dim must be 1 or 2."
        )

    # ========================================================
    # BOUNDARY LOSS
    # ========================================================

    def boundary_loss(
        self,
        prediction,
        target,
        spatial_dim,
    ):

        if spatial_dim == 1:

            left_loss = F.mse_loss(
                prediction[:, 0, :],
                target[:, 0, :],
            )

            right_loss = F.mse_loss(
                prediction[:, -1, :],
                target[:, -1, :],
            )

            return 0.5 * (
                left_loss + right_loss
            )

        if spatial_dim == 2:

            top_loss = F.mse_loss(
                prediction[:, 0, :, :],
                target[:, 0, :, :],
            )

            bottom_loss = F.mse_loss(
                prediction[:, -1, :, :],
                target[:, -1, :, :],
            )

            left_loss = F.mse_loss(
                prediction[:, :, 0, :],
                target[:, :, 0, :],
            )

            right_loss = F.mse_loss(
                prediction[:, :, -1, :],
                target[:, :, -1, :],
            )

            return 0.25 * (
                top_loss
                + bottom_loss
                + left_loss
                + right_loss
            )

        raise ValueError(
            "spatial_dim must be 1 or 2."
        )

    # ========================================================
    # PER-SAMPLE VARIATIONAL ENERGY
    # ========================================================

    def compute_energy_per_sample(
        self,
        field,
        previous_state=None,
        params=None,
    ):
        """
        Compute exactly one variational functional value per
        sample.

        Returns:
            [B]

        IMPORTANT:
        We call compute_functional() directly.

        We never use a batch-reduced scalar and never expand
        it back to [B].
        """

        if self.variational_loss is None:
            raise RuntimeError(
                "variational_loss is required."
            )

        energy = (
            self.variational_loss.compute_functional(
                prediction=field,
                previous_state=previous_state,
                params=params,
            )
        )

        if energy.ndim != 1:
            raise RuntimeError(
                "compute_functional must return [B], "
                f"got {tuple(energy.shape)}."
            )

        if energy.shape[0] != field.shape[0]:
            raise RuntimeError(
                "Energy batch size does not match field "
                "batch size."
            )

        return energy

    # ========================================================
    # ENERGY LOSS
    # ========================================================

    def energy_consistency_loss(
        self,
        prediction,
        target,
        previous_state=None,
        params=None,
    ):
        """
        Per-sample variational consistency:

            Pi_hat_i = Pi[u_hat_i]
            Pi_true_i = Pi[u_i]

        Absolute:
            mean((Pi_hat_i - Pi_true_i)^2)

        Relative:
            mean(
                ((Pi_hat_i - Pi_true_i) /
                 (abs(Pi_true_i) + eps))^2
            )

        No batch cancellation is possible.
        """

        predicted_energy = (
            self.compute_energy_per_sample(
                field=prediction,
                previous_state=previous_state,
                params=params,
            )
        )

        with torch.no_grad():

            target_energy = (
                self.compute_energy_per_sample(
                    field=target,
                    previous_state=previous_state,
                    params=params,
                )
            )

        energy_difference = (
            predicted_energy - target_energy
        )

        if self.energy_loss_type == "absolute":

            return energy_difference.pow(2).mean()

        denominator = (
            target_energy.abs()
            + self.energy_eps
        )

        return (
            energy_difference / denominator
        ).pow(2).mean()

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        prediction,
        target,
        reconstruction=None,
        input_field=None,
        spatial_dim=2,
        previous_state=None,
        params=None,
        dx=1.0,
        dy=1.0,
    ):

        if spatial_dim not in (1, 2):
            raise ValueError(
                "spatial_dim must be 1 or 2."
            )

        if params is None:
            params = {}
        else:
            params = dict(params)

        # Automatic forcing propagation for forcing -> solution
        # PDEs such as Poisson.
        if (
            input_field is not None
            and "forcing" not in params
        ):
            params["forcing"] = input_field

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction_loss = self.prediction_loss(
            prediction=prediction,
            target=target,
        )

        # ----------------------------------------------------
        # Reconstruction
        # ----------------------------------------------------

        if (
            self.lambda_recon > 0.0
            and reconstruction is not None
            and input_field is not None
        ):

            reconstruction_loss = (
                self.reconstruction_loss(
                    reconstruction=reconstruction,
                    input_field=input_field,
                )
            )

        else:

            reconstruction_loss = prediction.new_zeros(())

        # ----------------------------------------------------
        # Variational energy
        # ----------------------------------------------------

        if self.lambda_energy > 0.0:

            energy_loss = (
                self.energy_consistency_loss(
                    prediction=prediction,
                    target=target,
                    previous_state=previous_state,
                    params=params,
                )
            )

        else:

            energy_loss = prediction.new_zeros(())

        # ----------------------------------------------------
        # Gradient
        # ----------------------------------------------------

        if self.lambda_grad > 0.0:

            gradient_loss = (
                self.gradient_consistency_loss(
                    prediction=prediction,
                    target=target,
                    spatial_dim=spatial_dim,
                    dx=dx,
                    dy=dy,
                )
            )

        else:

            gradient_loss = prediction.new_zeros(())

        # ----------------------------------------------------
        # Boundary
        # ----------------------------------------------------

        if self.lambda_bc > 0.0:

            boundary_loss = self.boundary_loss(
                prediction=prediction,
                target=target,
                spatial_dim=spatial_dim,
            )

        else:

            boundary_loss = prediction.new_zeros(())

        # ----------------------------------------------------
        # Total
        # ----------------------------------------------------

        total_loss = (
            self.lambda_pred * prediction_loss
            + self.lambda_recon * reconstruction_loss
            + self.lambda_energy * energy_loss
            + self.lambda_grad * gradient_loss
            + self.lambda_bc * boundary_loss
        )

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                "PEDVINOLoss produced non-finite loss. "
                f"prediction={prediction_loss.detach().item()}, "
                f"reconstruction={reconstruction_loss.detach().item()}, "
                f"energy={energy_loss.detach().item()}, "
                f"gradient={gradient_loss.detach().item()}, "
                f"boundary={boundary_loss.detach().item()}"
            )

        return {
            "total_loss": total_loss,
            "prediction_loss": prediction_loss,
            "reconstruction_loss": reconstruction_loss,
            "energy_loss": energy_loss,
            "gradient_loss": gradient_loss,
            "boundary_loss": boundary_loss,
        }
