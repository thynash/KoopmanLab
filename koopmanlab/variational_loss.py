import math
import torch
import torch.nn as nn


class VariationalLoss(nn.Module):
    """
    Generic weak/variational loss.

    Given a PDE residual R on a spatial grid, minimize its projections
    onto a set of test functions:

        L = mean_k |∫ R(x) phi_k(x) dx|^2

    Supports 1D and 2D spatial residual tensors.
    """

    def __init__(
        self,
        num_test_functions=8,
        test_function="fourier",
        normalize=True,
    ):
        super().__init__()

        self.num_test_functions = num_test_functions
        self.test_function = test_function
        self.normalize = normalize

    def _fourier_1d(self, n, device, dtype):
        x = torch.linspace(
            0.0, 1.0, n,
            device=device,
            dtype=dtype,
        )

        functions = []

        for k in range(1, self.num_test_functions + 1):
            phi = torch.sin(math.pi * k * x)
            functions.append(phi)

        return torch.stack(functions, dim=0)

    def _fourier_2d(self, h, w, device, dtype):
        x = torch.linspace(
            0.0, 1.0, h,
            device=device,
            dtype=dtype,
        )

        y = torch.linspace(
            0.0, 1.0, w,
            device=device,
            dtype=dtype,
        )

        xx, yy = torch.meshgrid(x, y, indexing="ij")

        functions = []

        max_mode = int(math.ceil(math.sqrt(self.num_test_functions)))

        for i in range(1, max_mode + 1):
            for j in range(1, max_mode + 1):

                phi = (
                    torch.sin(math.pi * i * xx)
                    * torch.sin(math.pi * j * yy)
                )

                functions.append(phi)

                if len(functions) == self.num_test_functions:
                    break

            if len(functions) == self.num_test_functions:
                break

        return torch.stack(functions, dim=0)

    def _build_test_functions(self, residual):

        if residual.dim() == 3:
            # [B, X, C]
            _, n, _ = residual.shape

            return self._fourier_1d(
                n,
                residual.device,
                residual.dtype,
            )

        elif residual.dim() == 4:
            # [B, H, W, C]
            _, h, w, _ = residual.shape

            return self._fourier_2d(
                h,
                w,
                residual.device,
                residual.dtype,
            )

        else:
            raise ValueError(
                "Residual must have shape [B, X, C] "
                "or [B, H, W, C]."
            )

    def forward(self, residual):
        """
        Parameters
        ----------
        residual : torch.Tensor

            1D:
                [B, X, C]

            2D:
                [B, H, W, C]

        Returns
        -------
        torch.Tensor
            Scalar variational loss.
        """

        test_functions = self._build_test_functions(residual)

        if residual.dim() == 3:
            # residual: [B, X, C]
            # phi:      [K, X]

            weak_residual = torch.einsum(
                "bxc,kx->bkc",
                residual,
                test_functions,
            )

            if self.normalize:
                weak_residual = weak_residual / residual.shape[1]

        else:
            # residual: [B, H, W, C]
            # phi:      [K, H, W]

            weak_residual = torch.einsum(
                "bhwc,khw->bkc",
                residual,
                test_functions,
            )

            if self.normalize:
                weak_residual = weak_residual / (
                    residual.shape[1] * residual.shape[2]
                )

        return torch.mean(weak_residual ** 2)
