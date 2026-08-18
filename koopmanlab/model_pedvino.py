import torch
import torch.nn as nn

from koopmanlab.models import kno
from koopmanlab.physics_encoder import (
    PhysicsEncoder1D,
    PhysicsEncoder2D,
)
from koopmanlab.physics_decoder import (
    PhysicsDecoder1D,
    PhysicsDecoder2D,
)


class PEDVINO(nn.Module):
    """
    Physics Encoder-Decoder Koopman Neural Operator.

    Architecture
    ------------
    1D:
        x [B, N, t_len]
            ->
        PhysicsEncoder1D
            ->
        Existing KNO1d core
            ->
        PhysicsDecoder1D
            ->
        prediction [B, N, t_len]

    2D:
        x [B, H, W, t_len]
            ->
        PhysicsEncoder2D
            ->
        Existing KNO2d core
            ->
        PhysicsDecoder2D
            ->
        prediction [B, H, W, t_len]

    Important:
        The Koopman implementation itself is unchanged.
    """

    def __init__(
        self,
        backbone,
        t_len,
        operator_size,
        modes_x=16,
        modes_y=16,
        decompose=4,
        linear_type=True,
        normalization=False,
        hidden_size=None,
        dx=1.0,
        dy=1.0,
    ):
        super().__init__()

        self.backbone = backbone
        self.t_len = t_len
        self.operator_size = operator_size
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.decompose = decompose

        # =====================================================
        # KNO1d
        # =====================================================
        if backbone == "KNO1d":

            self.encoder = PhysicsEncoder1D(
                t_len=t_len,
                op_size=operator_size,
                hidden_size=hidden_size,
                dx=dx,
            )

            self.decoder = PhysicsDecoder1D(
                t_len=t_len,
                op_size=operator_size,
                hidden_size=hidden_size,
                dx=dx,
            )

            # -------------------------------------------------
            # ORIGINAL KNO CORE -- UNCHANGED
            # -------------------------------------------------
            self.kernel = kno.KNO1d(
                encoder=self.encoder,
                decoder=self.decoder,
                op_size=operator_size,
                modes_x=modes_x,
                decompose=decompose,
                linear_type=linear_type,
                normalization=normalization,
            )

        # =====================================================
        # KNO2d
        # =====================================================
        elif backbone == "KNO2d":

            self.encoder = PhysicsEncoder2D(
                t_len=t_len,
                op_size=operator_size,
                hidden_size=hidden_size,
                dx=dx,
                dy=dy,
            )

            self.decoder = PhysicsDecoder2D(
                t_len=t_len,
                op_size=operator_size,
                hidden_size=hidden_size,
                dx=dx,
                dy=dy,
            )

            # -------------------------------------------------
            # ORIGINAL KNO CORE -- UNCHANGED
            # -------------------------------------------------
            self.kernel = kno.KNO2d(
                encoder=self.encoder,
                decoder=self.decoder,
                op_size=operator_size,
                modes_x=modes_x,
                modes_y=modes_y,
                decompose=decompose,
                linear_type=linear_type,
                normalization=normalization,
            )

        else:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Choose 'KNO1d' or 'KNO2d'."
            )

    def forward(self, x):
        """
        Forward pass.

        Returns
        -------
        prediction:
            PDE solution prediction.

        reconstruction:
            Autoencoder reconstruction used by the
            original KNO training framework.
        """

        return self.kernel(x)
