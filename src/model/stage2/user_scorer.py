# model/stage2/user_scorer.py
# ------------------------------------------------------------
# Warm-user scorer (Stage 2).
#
# This module scores items for warm users (users seen during training)
# using learned per-user embeddings combined with item features.
#
# Interface:
#   Inputs:
#     - user_idx : LongTensor [B]          User indices in range [0, num_users)
#     - z_in     : Tensor     [B, Q, D_in] Item features from encoder
#   Output:
#     - scores   : Tensor     [B, Q]       Predicted relevance scores
#
# Architecture:
#   1) Transform item features: z_hat = MLP(z_in) -> [B, Q, d_model]
#   2) Retrieve per-user embedding θ_u ∈ R^{d_model} and optional bias b_u
#   3) Compute scores: <θ_u, z_hat> + b_u
#
# ------------------------------------------------------------

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
    use_layernorm: bool = False,
) -> nn.Sequential:
    """
    Build a multi-layer perceptron (MLP) with configurable architecture.
    
    The final layer has no activation function applied.
    
    Parameters
    ----------
    d_in : int
        Input dimension.
    d_out : int
        Output dimension.
    hidden : Sequence[int], optional
        Hidden layer dimensions. Empty sequence creates a single linear layer.
    dropout : float, default=0.0
        Dropout probability applied after each hidden layer activation.
    activation : str, default='relu'
        Activation function name. Options: 'relu', 'gelu', 'silu', 'tanh'.
    use_batchnorm : bool, default=False
        Whether to apply BatchNorm1d after each hidden linear layer.
    use_layernorm : bool, default=False
        Whether to apply LayerNorm after each hidden linear layer.
    
    Returns
    -------
    nn.Sequential
        The constructed MLP module.
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

    # Final projection without activation
    layers.append(nn.Linear(prev, d_out))
    return nn.Sequential(*layers)


class UserScorer(nn.Module):
    """
    Scores items for warm users using learned per-user embeddings.
    
    This module is designed for Stage 2 of the training pipeline, where we have
    a fixed set of known users. Each user is represented by a learnable embedding
    vector θ_u and an optional bias term b_u.
    
    The scoring process:
        1. Transform item features via MLP: z_hat = MLP(z_in)
        2. Retrieve user embedding: θ_u
        3. Compute dot product: score = <θ_u, z_hat> + b_u
    
    Optional normalization can be applied to user embeddings and/or item features
    for improved stability and generalization.
    
    Parameters
    ----------
    num_users : int
        Number of warm users in the training set.
    d_in : int
        Dimensionality of input item features.
    d_model : int
        Dimensionality of the internal embedding space (for both users and items).
    mlp_hidden : Sequence[int], default=(128,)
        Hidden layer sizes for the item feature MLP. Use empty sequence for
        a single linear projection.
    mlp_dropout : float, default=0.0
        Dropout probability in the MLP (applied after activations).
    mlp_activation : str, default='relu'
        Activation function for MLP hidden layers.
    use_batchnorm : bool, default=False
        Apply batch normalization in the MLP.
    use_layernorm : bool, default=False
        Apply layer normalization in the MLP.
    use_bias : bool, default=True
        Learn a scalar bias term for each user.
    normalize_user : bool, default=True
        L2-normalize user embeddings before scoring (recommended for stability).
    normalize_item : bool, default=False
        L2-normalize item features after MLP transformation.
    init_std : float, default=0.02
        Standard deviation for normal initialization of user embeddings.
    dtype : torch.dtype, default=torch.float32
        Data type for parameters.
    device : torch.device or str, optional
        Device for parameter placement.
    
    Examples
    --------
    >>> scorer = UserScorer(
    ...     num_users=1000,
    ...     d_in=64,
    ...     d_model=128,
    ...     mlp_hidden=(256,),
    ...     use_bias=True,
    ...     normalize_user=True
    ... )
    >>> user_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    >>> z_in = torch.randn(3, 10, 64)  # 3 users, 10 items, 64 features
    >>> scores = scorer(user_idx, z_in)
    >>> scores.shape
    torch.Size([3, 10])
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

        # MLP for transforming item features: z_in -> z_hat
        if mlp_hidden is None and d_in == d_model:
            # Identity transformation when dimensions match and no hidden layers
            self.item_mlp = nn.Identity()
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

        # Learnable per-user parameters
        self.user_table = nn.Embedding(
            self.num_users, self.d_model, dtype=dtype, device=device
        )
        if self.use_bias:
            self.bias = nn.Embedding(self.num_users, 1, dtype=dtype, device=device)
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize user embeddings and MLP weights."""
        # Initialize user embeddings with small random values
        nn.init.normal_(self.user_table.weight, mean=0.0, std=self.init_std)
        if self.bias is not None:
            nn.init.zeros_(self.bias.weight)
        
        # Initialize MLP with Xavier uniform initialization
        for m in self.item_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def clamp_user_norm_(self, max_norm: Optional[float] = None) -> None:
        """
        Clip user embeddings to a maximum L2 norm (in-place).
        
        This can be useful as a post-processing step after gradient updates
        to prevent embeddings from growing too large.
        
        Parameters
        ----------
        max_norm : float, optional
            Maximum allowed L2 norm. If None, no clipping is performed.
        """
        if not max_norm:
            return
        w = self.user_table.weight
        norms = w.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        scale = (max_norm / norms).clamp(max=1.0)
        self.user_table.weight.mul_(scale)

    def compute_z_i_tilde(self, z_in: torch.Tensor) -> torch.Tensor:
        """
        Transform item features through the MLP.
        
        Parameters
        ----------
        z_in : Tensor [B, Q, D_in]
            Raw item features.
        
        Returns
        -------
        z_i_tilde : Tensor [B, Q, d_model]
            Transformed item features.
        """
        assert z_in.dim() == 3, f"z_in must be [B, Q, D_in], got {tuple(z_in.shape)}"
        B, Q, D = z_in.shape

        # Handle BatchNorm which expects [N, C] input
        if self.use_batchnorm:
            z_flat = z_in.view(B * Q, D)
            z_i_tilde_flat = self.item_mlp(z_flat)
            z_i_tilde = z_i_tilde_flat.view(B, Q, -1)
        else:
            z_i_tilde = self.item_mlp(z_in)

        # Optional L2 normalization
        if self.normalize_item:
            z_i_tilde = F.normalize(z_i_tilde, dim=-1)
        
        return z_i_tilde

    def forward(
        self, user_idx: torch.LongTensor, z_in: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute relevance scores for items given users.
        
        Parameters
        ----------
        user_idx : LongTensor [B]
            User indices in range [0, num_users).
        z_in : Tensor [B, Q, D_in]
            Item features for each user in the batch.
        
        Returns
        -------
        scores : Tensor [B, Q]
            Predicted relevance scores for each (user, item) pair.
        """
        assert user_idx.dtype == torch.long, "user_idx must be torch.long"
        assert z_in.dim() == 3, f"z_in must be [B, Q, D_in], got {tuple(z_in.shape)}"
        B, Q, D_in = z_in.shape
        assert D_in == self.d_in, f"D mismatch: got {D_in}, expected {self.d_in}"
        assert user_idx.shape[0] == B, f"Batch size mismatch: {user_idx.shape[0]} != {B}"
        assert (user_idx >= 0).all() and (
            user_idx < self.num_users
        ).all(), "user_idx out of range"

        # Transform item features
        z_i_tilde = self.compute_z_i_tilde(z_in)  # [B, Q, d_model]

        # Retrieve user embeddings
        theta = self.user_table(user_idx.to(z_in.device))  # [B, d_model]
        if self.normalize_user:
            theta = F.normalize(theta, dim=-1)

        # Compute dot product scores
        scores = (z_i_tilde * theta.unsqueeze(1)).sum(dim=-1)  # [B, Q]
        
        # Add per-user bias if enabled
        if self.bias is not None:
            scores = scores + self.bias(user_idx.to(z_in.device)).squeeze(-1).unsqueeze(1)
        
        return scores

    def regularization(self) -> torch.Tensor:
        """
        Compute L2 regularization penalty on user parameters.
        
        Returns
        -------
        reg : Tensor (scalar)
            Sum of squared norms of user embeddings and biases.
        
        Note
        ----
        For MLP weight decay, use the optimizer's weight_decay parameter.
        """
        reg = (self.user_table.weight**2).sum()
        if self.bias is not None:
            reg = reg + (self.bias.weight**2).sum()
        return reg

    @torch.no_grad()
    def get_user_embeddings(
        self, idx: torch.Tensor | Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve user embeddings and biases for specified users.
        
        Parameters
        ----------
        idx : LongTensor [B] or Sequence[int]
            User indices.
        
        Returns
        -------
        theta_u : Tensor [B, d_model]
            User embeddings (not normalized, raw from table).
        b_u : Tensor [B, 1]
            User biases (zeros if bias is disabled).
        """
        if not torch.is_tensor(idx):
            idx = torch.as_tensor(
                idx, dtype=torch.long, device=self.user_table.weight.device
            )
        else:
            idx = idx.to(dtype=torch.long, device=self.user_table.weight.device)

        theta_u = self.user_table.weight[idx].detach().clone()

        if self.bias is not None:
            b_u = self.bias.weight[idx, 0].detach().clone().unsqueeze(-1)
        else:
            b_u = torch.zeros(
                theta_u.size(0), 1, dtype=theta_u.dtype, device=theta_u.device
            )

        return theta_u, b_u

    def get_user_embeddings_with_grad(
        self, idx: torch.Tensor | Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve user embeddings with gradient tracking enabled.
        
        This is useful for meta-learning or adaptation scenarios where
        user embeddings need to be updated via gradients.
        
        Parameters
        ----------
        idx : LongTensor [B] or Sequence[int]
            User indices.
        
        Returns
        -------
        theta_u : Tensor [B, d_model]
            User embeddings (normalized if normalize_user=True).
        b_u : Tensor [B, 1]
            User biases (zeros if bias is disabled).
        """
        if not torch.is_tensor(idx):
            idx = torch.as_tensor(
                idx, dtype=torch.long, device=self.user_table.weight.device
            )
        else:
            idx = idx.to(dtype=torch.long, device=self.user_table.weight.device)

        theta_u = self.user_table(idx)

        if self.normalize_user:
            theta_u = F.normalize(theta_u, dim=-1)

        if self.bias is not None:
            b_u = self.bias(idx)
        else:
            b_u = torch.zeros(
                theta_u.size(0), 1, dtype=theta_u.dtype, device=theta_u.device
            )

        return theta_u, b_u


if __name__ == "__main__":
    # Simple sanity check
    torch.manual_seed(0)

    B, Q, D_in, D_model = 4, 7, 8, 12
    scorer = UserScorer(
        num_users=4,
        d_in=D_in,
        d_model=D_model,
        mlp_hidden=(16,),
        use_bias=True,
        normalize_user=True,
        normalize_item=True,
    )

    user_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    z_in = torch.randn(B, Q, D_in)

    scores = scorer(user_idx, z_in)
    print("Scores shape:", scores.shape)  # [B, Q]
    print("Sample scores for first user:\n", scores[0])

    # Inspect learned parameters for a user
    theta_u, b_u = scorer.get_user_embeddings([3])
    print(f"\nUser 3 embedding shape: {theta_u.shape}")
    print(f"User 3 bias: {b_u.item():.4f}")