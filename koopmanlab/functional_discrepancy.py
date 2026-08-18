import torch
import torch.nn as nn


class FunctionalDiscrepancy(nn.Module):
    """
    Convert two PDE functional values into a non-negative
    optimization objective.

    Given:

        J_pred = J[u_hat]
        J_ref  = J[u_ref]

    this module computes:

        L_var = D(J_pred, J_ref)

    where:

        L_var >= 0

    This layer is PDE-independent.

    Supported modes
    ---------------

    mse:
        mean((J_pred - J_ref)^2)

    normalized_mse:
        mean(((J_pred - J_ref) / (|J_ref| + eps))^2)

    relative_mse:
        mean((J_pred - J_ref)^2 /
             (J_ref^2 + eps))

    absolute:
        mean(|J_pred - J_ref|)

    relative_absolute:
        mean(|J_pred - J_ref| /
             (|J_ref| + eps))
    """

    def __init__(
        self,
        mode="normalized_mse",
        eps=1e-8,
        reduction="mean",
    ):
        super().__init__()

        valid_modes = (
            "mse",
            "normalized_mse",
            "relative_mse",
            "absolute",
            "relative_absolute",
        )

        if mode not in valid_modes:
            raise ValueError(
                f"Unsupported discrepancy mode '{mode}'. "
                f"Choose from {valid_modes}."
            )

        if eps <= 0:
            raise ValueError(
                "eps must be positive."
            )

        if reduction not in (
            "mean",
            "sum",
            "none",
        ):
            raise ValueError(
                "reduction must be "
                "'mean', 'sum', or 'none'."
            )

        self.mode = mode
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        predicted_functional,
        reference_functional,
    ):

        if not torch.is_tensor(
            predicted_functional
        ):
            raise TypeError(
                "predicted_functional must be a tensor."
            )

        if not torch.is_tensor(
            reference_functional
        ):
            raise TypeError(
                "reference_functional must be a tensor."
            )

        if predicted_functional.shape != (
            reference_functional.shape
        ):
            raise ValueError(
                "Functional shape mismatch: "
                f"{tuple(predicted_functional.shape)} vs "
                f"{tuple(reference_functional.shape)}."
            )

        difference = (
            predicted_functional
            - reference_functional
        )

        # ====================================================
        # MSE
        # ====================================================

        if self.mode == "mse":

            discrepancy = difference.pow(2)

        # ====================================================
        # NORMALIZED MSE
        # ====================================================

        elif self.mode == "normalized_mse":

            scale = (
                reference_functional.abs()
                + self.eps
            )

            discrepancy = (
                difference / scale
            ).pow(2)

        # ====================================================
        # RELATIVE MSE
        # ====================================================

        elif self.mode == "relative_mse":

            denominator = (
                reference_functional.pow(2)
                + self.eps
            )

            discrepancy = (
                difference.pow(2)
                / denominator
            )

        # ====================================================
        # ABSOLUTE
        # ====================================================

        elif self.mode == "absolute":

            discrepancy = difference.abs()

        # ====================================================
        # RELATIVE ABSOLUTE
        # ====================================================

        else:

            scale = (
                reference_functional.abs()
                + self.eps
            )

            discrepancy = (
                difference.abs()
                / scale
            )

        # ====================================================
        # REDUCTION
        # ====================================================

        if self.reduction == "mean":
            return discrepancy.mean()

        if self.reduction == "sum":
            return discrepancy.sum()

        return discrepancy
