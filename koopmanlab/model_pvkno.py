import torch
import torch.nn as nn

from .models.kno import KNO1d, KNO2d
from .physics_encoder_decoder import (
    physics_encoder_conv1d,
    physics_decoder_conv1d,
    physics_encoder_conv2d,
    physics_decoder_conv2d,
)


class PVKNO1d(nn.Module):
    """
    Physics-aware KNO for 1D problems.

    The original KNO1d remains unchanged. This class only constructs it
    with physics-aware encoder and decoder modules.
    """

    def __init__(
        self,
        t_len,
        op_size,
        modes_x=16,
        decompose=4,
        linear_type=True,
        normalization=False,
        encoder_kwargs=None,
        decoder_kwargs=None,
    ):
        super().__init__()

        encoder_kwargs = encoder_kwargs or {}
        decoder_kwargs = decoder_kwargs or {}

        encoder = physics_encoder_conv1d(
            t_len=t_len,
            op_size=op_size,
            **encoder_kwargs,
        )

        decoder = physics_decoder_conv1d(
            t_len=t_len,
            op_size=op_size,
            **decoder_kwargs,
        )

        self.kno = KNO1d(
            encoder=encoder,
            decoder=decoder,
            op_size=op_size,
            modes_x=modes_x,
            decompose=decompose,
            linear_type=linear_type,
            normalization=normalization,
        )

    def forward(self, x):
        return self.kno(x)


class PVKNO2d(nn.Module):
    """
    Physics-aware Variational Koopman Neural Operator for 2D problems.

    The original KNO2d architecture is not modified.

    Physics awareness is introduced through the encoder and decoder:
        Input
          -> physics_encoder_conv2d
          -> existing KNO2d
          -> physics_decoder_conv2d
          -> prediction

    The variational loss is intentionally kept outside forward(),
    so the same model can be used with different PDEs.
    """

    def __init__(
        self,
        t_len,
        op_size,
        modes_x=10,
        modes_y=10,
        decompose=6,
        linear_type=True,
        normalization=False,
        encoder_kwargs=None,
        decoder_kwargs=None,
    ):
        super().__init__()

        encoder_kwargs = encoder_kwargs or {}
        decoder_kwargs = decoder_kwargs or {}

        self.encoder = physics_encoder_conv2d(
            t_len=t_len,
            op_size=op_size,
            **encoder_kwargs,
        )

        self.decoder = physics_decoder_conv2d(
            t_len=t_len,
            op_size=op_size,
            **decoder_kwargs,
        )

        self.kno = KNO2d(
            encoder=self.encoder,
            decoder=self.decoder,
            op_size=op_size,
            modes_x=modes_x,
            modes_y=modes_y,
            decompose=decompose,
            linear_type=linear_type,
            normalization=normalization,
        )

    def forward(self, x):
        return self.kno(x)
