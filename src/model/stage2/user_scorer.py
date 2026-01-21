"""
User Scorer Module - Stage 2
Contains the implementation of the Warm-user scorer, mapping user embeddings
and item features to preference scores.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Setup basic logging
logger = logging.getLogger(__name__)


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
    Constructs a Multi-Layer Perceptron (MLP).

    Args:
        d_in: Input dimensionality.
        d_out: Output dimensionality.
        hidden: Sizes of hidden layers. Empty sequence results in a linear projection.
        dropout: Dropout probability applied after activations.
        activation: Activation function name ('relu', 'gelu', 'silu', 'tanh').
        use_batchnorm: Whether to apply BatchNorm1d after hidden layers.
        use_layernorm: Whether to apply LayerNorm after hidden layers.

    Returns:
        nn.Sequential: The constructed MLP model.
    """
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    act_fn = activations.get(activation.lower(), nn.ReLU)

    layers: List[nn.Module] = []
    prev_dim = d_in

    for h_dim in hidden:
        layers.append(nn.Linear(prev_dim, h_dim))

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(h_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(h_dim))

        layers.append(act_fn())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = h_dim

    # Final projection layer (no activation)
    layers.append(nn.Linear(prev_dim, d_out))
    return nn.Sequential(*layers)


class UserScorer(nn.Module):
    """
    Scores items for warm users using learned user embeddings and an item feature MLP.

    The scoring logic follows:
        z_hat = MLP(z_in)
        score = <theta_u, z_hat> + b_u

    Attributes:
        num_users (int): Total number of unique users.
        d_in (int): Input feature dimension.
        d_model (int): Shared embedding space dimension.
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
            device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """Initializes the UserScorer with given architectural hyper-parameters."""
        super().__init__()

        self.num_users = int(num_users)
        self.d_in = int(d_in)
        self.d_model = int(d_model)
        self.use_bias = bool(use_bias)
        self.normalize_user = bool(normalize_user)
        self.normalize_item = bool(normalize_item)
        self.init_std = float(init_std)
        self.use_batchnorm = use_batchnorm
        self.use_layernorm = use_layernorm

        # MLP for item feature transformation
        if (mlp_hidden is None or len(mlp_hidden) == 0) and d_in == d_model:
            self.item_mlp = nn.Identity()
        else:
            self.item_mlp = _build_mlp(
                d_in=self.d_in,
                d_out=self.d_model,
                hidden=tuple(mlp_hidden) if mlp_hidden else (),
                dropout=mlp_dropout,
                activation=mlp_activation,
                use_batchnorm=use_batchnorm,
                use_layernorm=use_layernorm,
            )

        self.user_table = nn.Embedding(self.num_users, self.d_model, dtype=dtype, device=device)

        if self.use_bias:
            self.bias = nn.Embedding(self.num_users, 1, dtype=dtype, device=device)
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes weights using standard normalization and Xavier uniform."""
        nn.init.normal_(self.user_table.weight, mean=0.0, std=self.init_std)
        if self.bias is not None:
            nn.init.zeros_(self.bias.weight)

        for m in self.item_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def clamp_user_norm_(self, max_norm: Optional[float] = None) -> None:
        """
        Clamps user embedding weights to a maximum L2 norm.

        Args:
            max_norm: Maximum allowed L2 norm. If None, no clamping is performed.
        """
        if max_norm is None:
            return
        w = self.user_table.weight
        norms = w.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        scale = (max_norm / norms).clamp(max=1.0)
        self.user_table.weight.mul_(scale)

    def compute_transformed_items(self, z_in: torch.Tensor) -> torch.Tensor:
        """
        Applies the MLP transformation to input item features.

        Args:
            z_in: Input tensor of shape [Batch, Query, D_in].

        Returns:
            torch.Tensor: Transformed features of shape [Batch, Query, D_model].
        """
        if z_in.dim() != 3:
            raise ValueError(f"Expected z_in to have 3 dims [B, Q, D_in], got {z_in.dim()}")

        b_size, q_size, d_dim = z_in.shape

        if self.use_batchnorm:
            # Flatten to apply BatchNorm1d correctly across the batch/query dimensions
            z_flat = z_in.view(b_size * q_size, d_dim)
            z_tilde_flat = self.item_mlp(z_flat)
            z_tilde = z_tilde_flat.view(b_size, q_size, -1)
        else:
            z_tilde = self.item_mlp(z_in)

        if self.normalize_item:
            z_tilde = F.normalize(z_tilde, dim=-1)

        return z_tilde

    def forward(self, user_idx: torch.LongTensor, z_in: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute scores for user-item pairs.

        Args:
            user_idx: LongTensor of shape [Batch] containing user indices.
            z_in: FloatTensor of shape [Batch, Query, D_in] containing item features.

        Returns:
            torch.Tensor: Interaction scores of shape [Batch, Query].
        """
        # Feature transformation
        z_tilde = self.compute_transformed_items(z_in)

        # User lookup
        theta = self.user_table(user_idx.to(z_in.device))
        if self.normalize_user:
            theta = F.normalize(theta, dim=-1)

        # Scoring (Dot product)
        # z_tilde: [B, Q, D], theta: [B, D] -> score: [B, Q]
        scores = (z_tilde * theta.unsqueeze(1)).sum(dim=-1)

        if self.bias is not None:
            user_bias = self.bias(user_idx.to(z_in.device)).squeeze(-1)  # [B]
            scores = scores + user_bias.unsqueeze(1)

        return scores

    def get_regularization_loss(self) -> torch.Tensor:
        """Computes L2 regularization loss for user parameters."""
        reg_loss = (self.user_table.weight ** 2).sum()
        if self.bias is not None:
            reg_loss += (self.bias.weight ** 2).sum()
        return reg_loss

    @torch.no_grad()
    def get_user_embeddings(self, idx: Union[torch.Tensor, Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves detached user embeddings and biases.

        Args:
            idx: User indices (Tensor or list).

        Returns:
            Tuple containing (theta_u, b_u).
        """
        if not torch.is_tensor(idx):
            idx = torch.as_tensor(idx, dtype=torch.long, device=self.user_table.weight.device)
        else:
            idx = idx.to(dtype=torch.long, device=self.user_table.weight.device)

        theta_u = self.user_table.weight[idx].detach().clone()

        if self.bias is not None:
            b_u = self.bias.weight[idx].detach().clone()
        else:
            b_u = torch.zeros(theta_u.size(0), 1, dtype=theta_u.dtype, device=theta_u.device)

        return theta_u, b_u

    def get_user_embeddings_with_grad(self, idx: Union[torch.Tensor, Sequence[int]]) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """Similar to get_user_embeddings but preserves the computation graph."""
        if not torch.is_tensor(idx):
            idx = torch.as_tensor(idx, dtype=torch.long, device=self.user_table.weight.device)

        theta_u = self.user_table(idx)
        if self.normalize_user:
            theta_u = F.normalize(theta_u, dim=-1)

        b_u = self.bias(idx) if self.bias is not None else \
            torch.zeros(theta_u.size(0), 1, dtype=theta_u.dtype, device=theta_u.device)

        return theta_u, b_u