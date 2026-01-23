# model/stage2/user_scorer.py
"""
UNISON Stage 2: User Scorer Module

This module implements the preference modeling component of UNISON (Stage 2 from the paper).
It learns to score items for warm (previously seen) users by combining:
1. A shared MLP (phi) that projects item embeddings into a discriminative space
2. Per-user linear parameters (theta_u, b_u) that capture individual preferences

Architecture:
    z_i ∈ R^{D_in}           # Input item embedding (from Stage 1 encoder)
    z_tilde = phi(z_i)       # Shared MLP projection to R^{D_model}
    score = <theta_u, z_tilde> + b_u  # Linear scoring (Eq. 5 in paper)

Key Design Choices:
- Linear scoring suffices when phi produces discriminative embeddings (Theorem 1)
- Optional L2 normalization for stability (normalize_user, normalize_item flags)
- Per-user parameters are learned during training, frozen during cold-start inference

Typical Usage:
    # Initialize for 1000 warm users with 128-dim item embeddings
    scorer = UserScorer(
        num_users=1000,
        d_in=128,
        d_model=64,
        mlp_hidden=(256, 128),
        use_bias=True,
        normalize_user=True
    )

    # Forward pass for warm users
    user_indices = torch.tensor([0, 5, 10])  # [B]
    item_features = torch.randn(3, 20, 128)  # [B, num_items, d_in]
    scores = scorer(user_indices, item_features)  # [B, num_items]

    # Extract user embeddings for Stage 3
    theta_u, b_u = scorer.get_user_embeddings([0, 5, 10])
"""

from __future__ import annotations
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(
        d_in: int,
        d_out: int,
        hidden: Sequence[int] = (),
        dropout: float = 0.0,
        activation: str = "relu",
        use_batchnorm: bool = False,
        use_layernorm: bool = False
) -> nn.Sequential:
    """
    Build a simple feedforward MLP.

    The final layer has no activation (linear output) to serve as a flexible
    projection layer for downstream scoring.

    Args:
        d_in: Input dimension
        d_out: Output dimension
        hidden: Tuple of hidden layer sizes, e.g., (256, 128). Empty tuple = single linear layer
        dropout: Dropout probability applied after each hidden layer activation
        activation: Activation function name ("relu", "gelu", "silu", "tanh")
        use_batchnorm: Apply BatchNorm1d after each linear layer (before activation)
        use_layernorm: Apply LayerNorm after each linear layer (before activation)

    Returns:
        nn.Sequential module implementing the MLP
    """
    acts = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    Act = acts.get(activation.lower(), nn.ReLU)

    layers: List[nn.Module] = []
    prev = d_in

    # Build hidden layers with optional normalization and dropout
    for h in hidden:
        layers.append(nn.Linear(prev, h))

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(h))
        if use_layernorm:
            layers.append(nn.LayerNorm(h))

        layers.append(Act())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h

    # Final projection layer (no activation)
    layers.append(nn.Linear(prev, d_out))
    return nn.Sequential(*layers)


class UserScorer(nn.Module):
    """
    Stage 2 preference modeling for warm (previously seen) users.

    This module implements Equation 5 from the paper:
        z_tilde = phi(z_i)                    # Shared MLP projection
        score = <theta_u, z_tilde> + b_u      # Per-user linear scoring

    The shared MLP (phi) learns a discriminative embedding space where linear
    scoring is sufficient (Theorem 1). Per-user parameters (theta_u, b_u) capture
    individual preferences and serve as "functional embeddings" for Stage 3.

    Parameters
    ----------
    num_users : int
        Number of warm users in the training set (fixed at initialization)
    d_in : int
        Dimensionality of input item features (from Stage 1 encoder)
    d_model : int
        Dimensionality of the shared embedding space (z_tilde and theta_u)
    mlp_hidden : Sequence[int]
        Hidden layer sizes for the shared MLP phi. Use [] or () for identity mapping.
    mlp_dropout : float
        Dropout probability after hidden layer activations (not applied to final layer)
    mlp_activation : str
        Activation function: "relu", "gelu", "silu", or "tanh"
    use_batchnorm : bool
        Apply BatchNorm1d in the shared MLP (useful for large batch sizes)
    use_layernorm : bool
        Apply LayerNorm in the shared MLP (alternative to batchnorm)
    use_bias : bool
        Learn per-user bias terms b_u (typically True for better expressiveness)
    normalize_user : bool
        L2-normalize theta_u before dot product (improves training stability)
    normalize_item : bool
        L2-normalize z_tilde before dot product (optional; typically False)
    init_std : float
        Standard deviation for normal initialization of user embeddings theta_u
    dtype : torch.dtype
        Data type for parameters (typically torch.float32)
    device : torch.device or str
        Device placement for parameters ("cuda" or "cpu")

    Attributes
    ----------
    item_mlp : nn.Module
        Shared MLP phi that maps z_i -> z_tilde
    user_table : nn.Embedding
        Learnable user embeddings theta_u ∈ R^{num_users x d_model}
    bias : nn.Embedding or None
        Learnable user biases b_u ∈ R^{num_users x 1} (if use_bias=True)
    """

    def __init__(
            self,
            num_users: int,
            d_in: int,
            d_model: int,
            *,
            mlp_hidden: Sequence[int] = (128,),
            mlp_dropout: float = 0.0,
            mlp_activation: str = "relu",
            use_batchnorm: bool = False,
            use_layernorm: bool = False,
            use_bias: bool = True,
            normalize_user: bool = True,
            normalize_item: bool = False,
            init_std: float = 0.02,
            dtype: torch.dtype = torch.float32,
            device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__()
        assert num_users > 0, "num_users must be > 0"
        assert d_in > 0 and d_model > 0, "d_in and d_model must be > 0"

        self.num_users = int(num_users)
        self.d_in = int(d_in)
        self.d_model = int(d_model)
        self.use_bias = bool(use_bias)
        self.normalize_user = bool(normalize_user)
        self.normalize_item = bool(normalize_item)
        self.init_std = float(init_std)
        self.use_batchnorm = use_batchnorm
        self.use_layernorm = use_layernorm

        # Shared MLP: phi(z_i) -> z_tilde
        # Special case: if dimensions match and no hidden layers specified, use identity
        if mlp_hidden is None and d_in == d_model:
            self.item_mlp = nn.Identity()
            print("[UserScorer] No MLP projection: using identity mapping (d_in == d_model)")
        else:
            self.item_mlp = _build_mlp(
                d_in=self.d_in,
                d_out=self.d_model,
                hidden=tuple(mlp_hidden),
                dropout=mlp_dropout,
                activation=mlp_activation,
                use_batchnorm=use_batchnorm,
                use_layernorm=use_layernorm,
            )

        # Per-user embedding table: theta_u for each user
        self.user_table = nn.Embedding(
            self.num_users, self.d_model, dtype=dtype, device=device
        )

        # Optional per-user bias: b_u for each user
        if self.use_bias:
            self.bias = nn.Embedding(self.num_users, 1, dtype=dtype, device=device)
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    # ====================== Initialization ====================== #

    def reset_parameters(self) -> None:
        """
        Initialize user embeddings and MLP weights.

        User embeddings (theta_u) are initialized with small random values to avoid
        large initial scores. MLP weights use Xavier initialization for stable gradients.
        """
        # Initialize user embeddings with small normal noise
        nn.init.normal_(self.user_table.weight, mean=0.0, std=self.init_std)

        # Initialize biases to zero
        if self.bias is not None:
            nn.init.zeros_(self.bias.weight)

        # Initialize MLP weights (Xavier initialization for stable training)
        for m in self.item_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def clamp_user_norm_(self, max_norm: Optional[float] = None) -> None:
        """
        Optional post-processing: clip user embedding norms for stability.

        Can be called after optimizer steps to prevent embeddings from growing too large.
        Typically not necessary when using normalize_user=True.

        Args:
            max_norm: Maximum L2 norm for user embeddings. No-op if None.
        """
        if not max_norm:
            return
        w = self.user_table.weight
        norms = w.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        scale = (max_norm / norms).clamp(max=1.0)
        self.user_table.weight.mul_(scale)

    # ====================== Core Computation ====================== #

    def compute_z_i_tilde(self, z_in: torch.Tensor) -> torch.Tensor:
        """
        Project item embeddings through the shared MLP: z_tilde = phi(z_i).

        This is the key transformation that makes linear scoring sufficient (Theorem 1).
        The shared MLP learns a space where user preferences can be captured by
        simple linear functions.

        Args:
            z_in: Item embeddings [B, Q, D_in] from Stage 1 encoder

        Returns:
            z_tilde: Projected embeddings [B, Q, D_model]
        """
        assert z_in.dim() == 3, f"z_in must be [B, Q, D_in], got {tuple(z_in.shape)}"
        B, Q, D = z_in.shape

        # Handle BatchNorm: requires 2D input, so flatten batch and query dims
        if self.use_batchnorm:
            z_flat = z_in.view(B * Q, D)  # [B*Q, D_in]
            z_i_tilde_flat = self.item_mlp(z_flat)  # [B*Q, D_model]
            z_i_tilde = z_i_tilde_flat.view(B, Q, -1)  # [B, Q, D_model]
        else:
            z_i_tilde = self.item_mlp(z_in)

        # Optional L2 normalization for numerical stability
        if self.normalize_item:
            z_i_tilde = F.normalize(z_i_tilde, dim=-1)

        return z_i_tilde

    def forward(
            self,
            user_idx: torch.LongTensor,
            z_in: torch.Tensor
    ) -> torch.Tensor:
        """
        Score items for warm users using learned preferences.

        Implements Equation 5:
            z_tilde = phi(z_i)
            score = <theta_u, z_tilde> + b_u

        Args:
            user_idx: User indices [B] in range [0, num_users)
            z_in: Item features [B, Q, D_in] from Stage 1 encoder

        Returns:
            scores: Item scores [B, Q] for each (user, item) pair
        """
        assert user_idx.dtype == torch.long, "user_idx must be torch.long"
        assert z_in.dim() == 3, f"z_in must be [B, Q, D_in], got {tuple(z_in.shape)}"

        B, Q, D_in = z_in.shape
        assert D_in == self.d_in, f"D mismatch: got {D_in}, expected {self.d_in}"
        assert user_idx.shape[0] == B, f"user_idx length {user_idx.shape[0]} != batch {B}"
        assert (user_idx >= 0).all() and (user_idx < self.num_users).all(), \
            "user_idx out of range [0, num_users)"

        # Step 1: Project items through shared MLP
        z_i_tilde = self.compute_z_i_tilde(z_in)  # [B, Q, D_model]

        # Step 2: Fetch user-specific parameters
        theta = self.user_table(user_idx.to(z_in.device))  # [B, D_model]
        if self.normalize_user:
            theta = F.normalize(theta, dim=-1)

        # Step 3: Compute linear scores via dot product
        # scores[b,q] = sum_d( z_tilde[b,q,d] * theta[b,d] )
        scores = (z_i_tilde * theta.unsqueeze(1)).sum(dim=-1)  # [B, Q]

        # Step 4: Add per-user bias (if enabled)
        if self.bias is not None:
            b_u = self.bias(user_idx.to(z_in.device)).squeeze(-1)  # [B]
            scores = scores + b_u.unsqueeze(1)  # broadcast to [B, Q]

        return scores

    # ====================== Utilities ====================== #

    def regularization(self) -> torch.Tensor:
        """
        Compute L2 regularization term for user parameters.

        Can be added to training loss to prevent overfitting:
            loss_total = loss_task + lambda_reg * scorer.regularization()

        Note: MLP regularization should be handled via optimizer weight_decay.

        Returns:
            Scalar tensor: sum of squared user embeddings and biases
        """
        reg = (self.user_table.weight ** 2).sum()
        if self.bias is not None:
            reg = reg + (self.bias.weight ** 2).sum()
        return reg

    @torch.no_grad()
    def get_user_embeddings(
            self,
            idx
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract user embeddings for Stage 3 (bag classification).

        The learned parameters (theta_u, b_u) serve as "functional embeddings"
        that capture user preferences. These are fed to Stage 3 to predict
        user attributes (e.g., demographics, disease state).

        Args:
            idx: User indices as 1D LongTensor [B] or list/array of ints

        Returns:
            theta_u: User embedding vectors [B, d_model]
            b_u: User bias terms [B, 1] (zeros if use_bias=False)
        """
        # Convert input to tensor on correct device
        if not torch.is_tensor(idx):
            idx = torch.as_tensor(
                idx, dtype=torch.long, device=self.user_table.weight.device
            )
        else:
            idx = idx.to(dtype=torch.long, device=self.user_table.weight.device)

        # Extract theta_u from embedding table
        theta_u = self.user_table.weight[idx].detach().clone()  # [B, d_model]

        # Extract b_u (or create zeros if bias disabled)
        if self.bias is not None:
            # self.bias.weight is [num_users, 1]; select rows and reshape to [B, 1]
            b_u = self.bias.weight[idx, 0].detach().clone().unsqueeze(-1)
        else:
            b_u = torch.zeros(
                theta_u.size(0), 1, dtype=theta_u.dtype, device=theta_u.device
            )

        return theta_u, b_u

    def get_user_embeddings_with_grad(
            self,
            idx
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract user embeddings with gradient tracking (for training Stage 3).

        Similar to get_user_embeddings(), but keeps gradients attached for
        backpropagation. Used during joint training of Stage 2 + Stage 3.

        Args:
            idx: User indices as 1D LongTensor [B] or list/array of ints

        Returns:
            theta_u: User embeddings [B, d_model] with gradients
            b_u: User biases [B, 1] with gradients
        """
        # Convert input to tensor on correct device
        if not torch.is_tensor(idx):
            idx = torch.as_tensor(
                idx, dtype=torch.long, device=self.user_table.weight.device
            )
        else:
            idx = idx.to(dtype=torch.long, device=self.user_table.weight.device)

        # Lookup through forward pass (keeps gradients)
        theta_u = self.user_table(idx)  # [B, d_model]

        # Apply normalization if enabled
        if self.normalize_user:
            theta_u = F.normalize(theta_u, dim=-1)

        # Get bias (or zeros)
        if self.bias is not None:
            b_u = self.bias(idx)  # [B, 1]
        else:
            b_u = torch.zeros(
                theta_u.size(0), 1, dtype=theta_u.dtype, device=theta_u.device
            )

        return theta_u, b_u


# ====================== Example Usage ====================== #

if __name__ == "__main__":
    torch.manual_seed(0)

    # Setup example parameters
    B, Q, D_in, D_model = 4, 7, 8, 12

    # Initialize scorer for 4 warm users
    scorer = UserScorer(
        num_users=4,
        d_in=D_in,
        d_model=D_model,
        mlp_hidden=(16,),
        use_bias=True,
        normalize_user=True,
        normalize_item=True
    )

    # Example forward pass
    user_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)  # [B]
    z_in = torch.randn(B, Q, D_in)  # [B, Q, D_in]
    scores = scorer(user_idx, z_in)  # [B, Q]

    print("scores shape:", scores.shape)
    print("scores sample:\n", scores[0])

    # Extract embeddings for Stage 3
    theta_u, b_u = scorer.get_user_embeddings(3)
    print("\nUser 3 embedding theta_u shape:", theta_u.shape)
    print("User 3 bias b_u:", b_u.item())