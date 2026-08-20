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


# ============================================================
# DUAL-NESTED TANH ACTIVATION
# ============================================================

class DualTanh(nn.Module):
    """
    Adaptive dual-nested tanh activation.

    DualTanh_beta(z)
        = tanh(
            z * (tanh(beta * z) + 1)
          )

    beta is trainable.

    beta_init:
        Initial value of the adaptive parameter beta.
    """

    def __init__(self, beta_init=1.0):

        super().__init__()

        self.beta = nn.Parameter(
            torch.tensor(
                float(beta_init),
                dtype=torch.float32,
            )
        )

    def forward(self, z):

        return torch.tanh(
            z * (
                torch.tanh(
                    self.beta * z
                )
                + 1.0
            )
        )


# ============================================================
# ACTIVATION REPLACEMENT
# ============================================================

def replace_activations(
    module,
    beta_init=1.0,
):
    """
    Recursively replace standard activation modules
    with DualTanh.

    This operates on the complete PEDVINO network,
    including:

        - physics encoder
        - Koopman core
        - physics decoder

    Only explicit nn.Module activations can be replaced.
    Functional calls such as torch.tanh(...) inside
    external code cannot be intercepted by this method.
    """

    activation_types = (
        nn.ReLU,
        nn.GELU,
        nn.Tanh,
        nn.Sigmoid,
        nn.SiLU,
        nn.LeakyReLU,
        nn.ELU,
        nn.SELU,
        nn.Softplus,
    )

    for name, child in module.named_children():

        if isinstance(
            child,
            activation_types,
        ):

            setattr(
                module,
                name,
                DualTanh(
                    beta_init=beta_init
                ),
            )

        else:

            replace_activations(
                child,
                beta_init=beta_init,
            )

    return module


# ============================================================
# PEDVINO
# ============================================================

class PEDVINO(nn.Module):
    """
    Physics Encoder-Decoder Koopman Neural Operator.

    Architecture
    ------------

    1D:

        input
          |
          v
    Physics Encoder
          |
          v
    Koopman Operator
          |
          v
    Physics Decoder
          |
          v
      prediction


    2D:

        input
          |
          v
    Physics Encoder
          |
          v
    Koopman Operator
          |
          v
    Physics Decoder
          |
          v
      prediction


    PEDVINO additionally uses the adaptive
    Dual-Tanh activation throughout all explicit
    activation modules in the constructed network.

    The Koopman mathematical operator itself is unchanged.
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
        use_dual_tanh=True,
        beta_init=1.0,
    ):

        super().__init__()

        self.backbone = backbone
        self.t_len = t_len
        self.operator_size = operator_size
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.decompose = decompose

        self.use_dual_tanh = use_dual_tanh
        self.beta_init = beta_init

        # ====================================================
        # KNO1d
        # ====================================================

        if backbone == "KNO1d":

            # ------------------------------------------------
            # Physics Encoder
            # ------------------------------------------------

            self.encoder = PhysicsEncoder1D(
                t_len=t_len,
                op_size=operator_size,
                hidden_size=hidden_size,
                dx=dx,
            )

            # ------------------------------------------------
            # Physics Decoder
            # ------------------------------------------------

            self.decoder = PhysicsDecoder1D(
                t_len=t_len,
                op_size=operator_size,
                hidden_size=hidden_size,
                dx=dx,
            )

            # ------------------------------------------------
            # Koopman Core
            # ------------------------------------------------

            self.kernel = kno.KNO1d(
                encoder=self.encoder,
                decoder=self.decoder,
                op_size=operator_size,
                modes_x=modes_x,
                decompose=decompose,
                linear_type=linear_type,
                normalization=normalization,
            )

        # ====================================================
        # KNO2d
        # ====================================================

        elif backbone == "KNO2d":

            # ------------------------------------------------
            # Physics Encoder
            # ------------------------------------------------

            self.encoder = PhysicsEncoder2D(
                t_len=t_len,
                op_size=operator_size,
                hidden_size=hidden_size,
                dx=dx,
                dy=dy,
            )

            # ------------------------------------------------
            # Physics Decoder
            # ------------------------------------------------

            self.decoder = PhysicsDecoder2D(
                t_len=t_len,
                op_size=operator_size,
                hidden_size=hidden_size,
                dx=dx,
                dy=dy,
            )

            # ------------------------------------------------
            # Koopman Core
            # ------------------------------------------------

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
                "Choose 'KNO1d' or 'KNO2d'."
            )

        # ====================================================
        # DUAL-TANH ACTIVATION
        # ====================================================

        if self.use_dual_tanh:

            replace_activations(
                self,
                beta_init=beta_init,
            )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        return self.kernel(x)

    # ========================================================
    # DUAL-TANH PARAMETERS
    # ========================================================

    def get_dual_tanh_parameters(self):

        betas = {}

        for name, module in self.named_modules():

            if isinstance(
                module,
                DualTanh,
            ):

                betas[name] = (
                    module.beta.detach()
                    .cpu()
                    .item()
                )

        return betas
