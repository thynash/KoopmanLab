import torch
import torch.nn as nn
import torch.nn.functional as F



# ============================================================
# Physics Feature Extraction: 1D
# ============================================================

class PhysicsFeatures1D(nn.Module):
    """
    Computes local differential features from a 1D field.

    Input:
        x : [B, X, C]

    Output:
        features : [B, X, 3*C]

    Features for every channel:
        u
        du/dx
        d²u/dx²
    """

    def __init__(self, channels):
        super().__init__()

        self.channels = channels

        # Central difference:
        # du/dx ≈ (u[i+1] - u[i-1]) / 2
        dx_kernel = torch.tensor(
            [-0.5, 0.0, 0.5],
            dtype=torch.float32,
        ).view(1, 1, 3)

        # Second derivative:
        # d²u/dx² ≈ u[i-1] - 2u[i] + u[i+1]
        dxx_kernel = torch.tensor(
            [1.0, -2.0, 1.0],
            dtype=torch.float32,
        ).view(1, 1, 3)

        self.register_buffer("dx_kernel", dx_kernel)
        self.register_buffer("dxx_kernel", dxx_kernel)

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError(
                "PhysicsFeatures1D expects [B, X, C], "
                f"got {tuple(x.shape)}"
            )

        # [B, X, C] -> [B, C, X]
        x_cf = x.permute(0, 2, 1)

        _, C, _ = x_cf.shape

        dx_kernel = self.dx_kernel.repeat(C, 1, 1)
        dxx_kernel = self.dxx_kernel.repeat(C, 1, 1)

        # Periodic padding
        x_pad = F.pad(
            x_cf,
            (1, 1),
            mode="circular",
        )

        dx = F.conv1d(
            x_pad,
            dx_kernel,
            groups=C,
        )

        dxx = F.conv1d(
            x_pad,
            dxx_kernel,
            groups=C,
        )

        # [B, C, X] -> [B, X, 3C]
        features = torch.cat(
            [x_cf, dx, dxx],
            dim=1,
        ).permute(0, 2, 1)

        return features



# ============================================================
# Physics-Aware Encoder: 1D
# ============================================================

class physics_encoder_conv1d(nn.Module):

    def __init__(
        self,
        t_len,
        op_size,
        hidden_size=None,
        physics_scale=0.0,
    ):
        super().__init__()

        self.t_len = t_len
        self.op_size = op_size

        if hidden_size is None:
            hidden_size = max(op_size, 32)

        # Base branch equivalent to original encoder_conv1d
        self.base_encoder = nn.Conv1d(
            t_len,
            op_size,
            kernel_size=1,
        )

        self.physics_features = PhysicsFeatures1D(
            channels=t_len
        )

        # 3*t_len: [u, ux, uxx]
        self.physics_encoder = nn.Sequential(
            nn.Conv1d(
                3 * t_len,
                hidden_size,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Conv1d(
                hidden_size,
                op_size,
                kernel_size=1,
            ),
        )

        # Starts as original encoder
        self.physics_scale = nn.Parameter(
            torch.tensor(
                float(physics_scale),
                dtype=torch.float32,
            )
        )

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError(
                "physics_encoder_conv1d expects [B, X, C], "
                f"got {tuple(x.shape)}"
            )

        if x.shape[-1] != self.t_len:
            raise ValueError(
                f"Expected {self.t_len} channels, "
                f"got {x.shape[-1]}"
            )

        # [B, X, C] -> [B, C, X]
        x_cf = x.permute(0, 2, 1)

        z_base = self.base_encoder(x_cf)

        features = self.physics_features(x)
        features_cf = features.permute(0, 2, 1)

        z_physics = self.physics_encoder(features_cf)

        z = (
            z_base
            + self.physics_scale * z_physics
        )

        # [B, op_size, X] -> [B, X, op_size]
        return z.permute(0, 2, 1)



# ============================================================
# Physics-Aware Decoder: 1D
# ============================================================

class physics_decoder_conv1d(nn.Module):

    def __init__(
        self,
        t_len,
        op_size,
        hidden_size=None,
        physics_scale=0.0,
    ):
        super().__init__()

        self.t_len = t_len
        self.op_size = op_size

        if hidden_size is None:
            hidden_size = max(op_size, 32)

        # Base decoder equivalent to original decoder_conv1d
        self.base_decoder = nn.Conv1d(
            op_size,
            t_len,
            kernel_size=1,
        )

        # Learn a physics-informed correction from latent space
        self.physics_decoder = nn.Sequential(
            nn.Conv1d(
                op_size,
                hidden_size,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Conv1d(
                hidden_size,
                t_len,
                kernel_size=1,
            ),
        )

        self.physics_scale = nn.Parameter(
            torch.tensor(
                float(physics_scale),
                dtype=torch.float32,
            )
        )

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError(
                "physics_decoder_conv1d expects "
                "[B, X, op_size], "
                f"got {tuple(x.shape)}"
            )

        if x.shape[-1] != self.op_size:
            raise ValueError(
                f"Expected latent dimension {self.op_size}, "
                f"got {x.shape[-1]}"
            )

        # [B, X, op_size] -> [B, op_size, X]
        x_cf = x.permute(0, 2, 1)

        u_base = self.base_decoder(x_cf)
        u_physics = self.physics_decoder(x_cf)

        u = (
            u_base
            + self.physics_scale * u_physics
        )

        # [B, t_len, X] -> [B, X, t_len]
        return u.permute(0, 2, 1)


# ============================================================
# Physics Feature Extraction 2D
# ============================================================

class PhysicsFeatures2D(nn.Module):
    """
    Computes local differential features from a 2D field.

    Input:
        x : [B, H, W, C]

    Output:
        features : [B, H, W, 4*C]

    Features for every input channel:
        u
        du/dx
        du/dy
        Laplacian(u)
    """

    def __init__(self, channels):
        super().__init__()

        self.channels = channels

        # ----------------------------------------------------
        # First derivative kernels
        #
        # Central difference:
        #
        # du/dx ≈ (u[i+1] - u[i-1]) / 2
        # du/dy ≈ (u[j+1] - u[j-1]) / 2
        # ----------------------------------------------------

        dx_kernel = torch.tensor(
            [
                [-0.5, 0.0, 0.5]
            ],
            dtype=torch.float32
        ).view(1, 1, 1, 3)

        dy_kernel = torch.tensor(
            [
                [-0.5],
                [0.0],
                [0.5]
            ],
            dtype=torch.float32
        ).view(1, 1, 3, 1)

        # ----------------------------------------------------
        # Laplacian kernel
        #
        # Δu ≈
        # u[i+1,j] + u[i-1,j]
        # + u[i,j+1] + u[i,j-1]
        # - 4u[i,j]
        # ----------------------------------------------------

        lap_kernel = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, -4.0, 1.0],
                [0.0, 1.0, 0.0]
            ],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        # Register as buffers.
        # They are moved automatically with the model.
        self.register_buffer(
            "dx_kernel",
            dx_kernel
        )

        self.register_buffer(
            "dy_kernel",
            dy_kernel
        )

        self.register_buffer(
            "lap_kernel",
            lap_kernel
        )

    def forward(self, x):

        # ----------------------------------------------------
        # [B,H,W,C] -> [B,C,H,W]
        # ----------------------------------------------------

        x = x.permute(
            0, 3, 1, 2
        )

        B, C, H, W = x.shape

        # ----------------------------------------------------
        # Repeat kernels for depthwise convolution.
        #
        # groups=C means every physical variable gets its
        # own derivative.
        # ----------------------------------------------------

        dx_kernel = self.dx_kernel.repeat(
            C, 1, 1, 1
        )

        dy_kernel = self.dy_kernel.repeat(
            C, 1, 1, 1
        )

        lap_kernel = self.lap_kernel.repeat(
            C, 1, 1, 1
        )

        # ----------------------------------------------------
        # Periodic padding
        #
        # Navier-Stokes datasets are commonly represented on
        # periodic spatial domains.
        # ----------------------------------------------------

        x_dx = F.pad(
            x,
            (1, 1, 0, 0),
            mode="circular"
        )

        x_dy = F.pad(
            x,
            (0, 0, 1, 1),
            mode="circular"
        )

        x_lap = F.pad(
            x,
            (1, 1, 1, 1),
            mode="circular"
        )

        # ----------------------------------------------------
        # Derivatives
        # ----------------------------------------------------

        dx = F.conv2d(
            x_dx,
            dx_kernel,
            groups=C
        )

        dy = F.conv2d(
            x_dy,
            dy_kernel,
            groups=C
        )

        lap = F.conv2d(
            x_lap,
            lap_kernel,
            groups=C
        )

        # ----------------------------------------------------
        # Concatenate physical descriptors
        #
        # [u, ux, uy, laplacian]
        # ----------------------------------------------------

        features = torch.cat(
            [
                x,
                dx,
                dy,
                lap
            ],
            dim=1
        )

        # ----------------------------------------------------
        # [B,4C,H,W] -> [B,H,W,4C]
        # ----------------------------------------------------

        features = features.permute(
            0, 2, 3, 1
        )

        return features


# ============================================================
# Physics-Aware Encoder
# ============================================================

class physics_encoder_conv2d(nn.Module):
    """
    Physics-aware replacement for the original encoder_conv2d.

    Original interface:

        [B,H,W,t_len]
            ->
        [B,H,W,op_size]

    New interface:

        [B,H,W,t_len]
            ->
        [B,H,W,op_size]

    The output shape is IDENTICAL to the original encoder.

    Internally:

        x
         |
         +---- Base Conv Encoder
         |
         +---- Physics Features
                    |
                    +-- u
                    +-- du/dx
                    +-- du/dy
                    +-- Laplacian
                    |
                    +---- Physics Projection
         |
         +---- Residual Fusion
         |
         v
       latent
    """

    def __init__(
        self,
        t_len,
        op_size,
        hidden_size=None,
        physics_scale=0.0
    ):

        super().__init__()

        self.t_len = t_len
        self.op_size = op_size

        if hidden_size is None:
            hidden_size = max(
                op_size,
                32
            )

        self.hidden_size = hidden_size

        # ----------------------------------------------------
        # Original encoder branch
        #
        # EXACT same operation as encoder_conv2d.
        # ----------------------------------------------------

        self.base_encoder = nn.Conv2d(
            t_len,
            op_size,
            kernel_size=1
        )

        # ----------------------------------------------------
        # Physics feature extractor
        #
        # t_len -> 4*t_len
        # ----------------------------------------------------

        self.physics_features = PhysicsFeatures2D(
            channels=t_len
        )

        # ----------------------------------------------------
        # Physics projection
        #
        # 4*t_len -> hidden -> op_size
        # ----------------------------------------------------

        self.physics_encoder = nn.Sequential(

            nn.Conv2d(
                4 * t_len,
                hidden_size,
                kernel_size=1
            ),

            nn.GELU(),

            nn.Conv2d(
                hidden_size,
                op_size,
                kernel_size=1
            )
        )

        # ----------------------------------------------------
        # Learnable physics strength.
        #
        # physics_scale=0 means:
        #
        # new encoder == original encoder
        #
        # This makes ablation very clean.
        # ----------------------------------------------------

        self.physics_scale = nn.Parameter(
            torch.tensor(
                float(physics_scale),
                dtype=torch.float32
            )
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Check input
        # ----------------------------------------------------

        if x.ndim != 4:

            raise ValueError(
                "physics_encoder_conv2d expects "
                "[B,H,W,C] input, got "
                f"{tuple(x.shape)}"
            )

        if x.shape[-1] != self.t_len:

            raise ValueError(
                f"Expected {self.t_len} input channels, "
                f"got {x.shape[-1]}"
            )

        # ----------------------------------------------------
        # Convert:
        #
        # [B,H,W,C]
        #
        # ->
        #
        # [B,C,H,W]
        # ----------------------------------------------------

        x_cf = x.permute(
            0, 3, 1, 2
        )

        # ----------------------------------------------------
        # Base encoder
        # ----------------------------------------------------

        z_base = self.base_encoder(
            x_cf
        )

        # ----------------------------------------------------
        # Physics features
        #
        # Returns [B,H,W,4C]
        # ----------------------------------------------------

        physics_features = self.physics_features(
            x
        )

        physics_features = physics_features.permute(
            0, 3, 1, 2
        )

        # ----------------------------------------------------
        # Physics encoder
        # ----------------------------------------------------

        z_physics = self.physics_encoder(
            physics_features
        )

        # ----------------------------------------------------
        # Residual fusion
        # ----------------------------------------------------

        z = (
            z_base
            +
            self.physics_scale * z_physics
        )

        # ----------------------------------------------------
        # Return SAME format as original encoder:
        #
        # [B,H,W,op_size]
        # ----------------------------------------------------

        z = z.permute(
            0, 2, 3, 1
        )

        return z


# ============================================================
# Physics-Aware Decoder
# ============================================================

class physics_decoder_conv2d(nn.Module):
    """
    Physics-aware replacement for decoder_conv2d.

    Original:

        [B,H,W,op_size]
            ->
        [B,H,W,t_len]

    New:

        [B,H,W,op_size]
            ->
        [B,H,W,t_len]

    Output dimension is EXACTLY the same.
    """

    def __init__(
        self,
        t_len,
        op_size,
        hidden_size=None,
        physics_scale=0.0
    ):

        super().__init__()

        self.t_len = t_len
        self.op_size = op_size

        if hidden_size is None:
            hidden_size = max(
                op_size,
                32
            )

        self.hidden_size = hidden_size

        # ----------------------------------------------------
        # Original decoder branch
        # ----------------------------------------------------

        self.base_decoder = nn.Conv2d(
            op_size,
            t_len,
            kernel_size=1
        )

        # ----------------------------------------------------
        # Physics decoder
        #
        # latent -> hidden -> output
        #
        # IMPORTANT:
        # final output = t_len
        # ----------------------------------------------------

        self.physics_decoder = nn.Sequential(

            nn.Conv2d(
                op_size,
                hidden_size,
                kernel_size=1
            ),

            nn.GELU(),

            nn.Conv2d(
                hidden_size,
                t_len,
                kernel_size=1
            )
        )

        # ----------------------------------------------------
        # Learnable physics strength
        # ----------------------------------------------------

        self.physics_scale = nn.Parameter(
            torch.tensor(
                float(physics_scale),
                dtype=torch.float32
            )
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Check input
        # ----------------------------------------------------

        if x.ndim != 4:

            raise ValueError(
                "physics_decoder_conv2d expects "
                "[B,H,W,C] input, got "
                f"{tuple(x.shape)}"
            )

        if x.shape[-1] != self.op_size:

            raise ValueError(
                f"Expected {self.op_size} latent channels, "
                f"got {x.shape[-1]}"
            )

        # ----------------------------------------------------
        # [B,H,W,C] -> [B,C,H,W]
        # ----------------------------------------------------

        x_cf = x.permute(
            0, 3, 1, 2
        )

        # ----------------------------------------------------
        # Base decoder
        # ----------------------------------------------------

        y_base = self.base_decoder(
            x_cf
        )

        # ----------------------------------------------------
        # Physics decoder
        # ----------------------------------------------------

        y_physics = self.physics_decoder(
            x_cf
        )

        # ----------------------------------------------------
        # Residual physics correction
        # ----------------------------------------------------

        y = (
            y_base
            +
            self.physics_scale * y_physics
        )

        # ----------------------------------------------------
        # [B,C,H,W] -> [B,H,W,C]
        #
        # SAME output interface as original decoder.
        # ----------------------------------------------------

        y = y.permute(
            0, 2, 3, 1
        )

        return y
