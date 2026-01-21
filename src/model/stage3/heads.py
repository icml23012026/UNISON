"""
UNISON Framework - Stage 3 Prediction Heads
This module defines the architectural heads for downstream tasks,
specifically focusing on predicting user attributes from learned embeddings.
"""

from typing import Union, Sequence, List, Optional
import torch
import torch.nn as nn


class UserAttrHead(nn.Module):
    """
    Predicts user-level attributes (e.g., demographic traits, preferences)
    by processing the concatenated user embedding (theta) and bias (b).

    The head consists of a configurable MLP with support for various
    activations, normalization layers, and dropout.

    Args:
        d_model: Dimensionality of the user embedding (theta_u).
        out_dim: Output dimension (e.g., 1 for regression/binary, N for multiclass).
        hidden: Sequence of hidden layer sizes.
        dropout: Dropout probability (float or sequence matching 'hidden').
        activation: Activation function ('relu', 'gelu', 'tanh', 'sigmoid').
        use_batchnorm: Apply BatchNorm1d after linear layers.
        use_layernorm: Apply LayerNorm after linear layers.
    """

    def __init__(
            self,
            d_model: int,
            out_dim: int = 1,
            hidden: Sequence[int] = (128, 64),
            dropout: Union[float, Sequence[float]] = 0.0,
            activation: str = "relu",
            use_batchnorm: bool = False,
            use_layernorm: bool = False,
    ):
        super().__init__()

        if use_batchnorm and use_layernorm:
            raise ValueError("Configuration error: choose either batchnorm or layernorm, not both.")

        # Normalize hidden layers to tuple
        hidden_dims = (hidden,) if isinstance(hidden, int) else tuple(hidden)

        # Build dropout sequence
        if isinstance(dropout, (float, int)):
            dropout_probs = [float(dropout)] * len(hidden_dims)
        else:
            dropout_probs = list(dropout)
            if len(dropout_probs) != len(hidden_dims):
                raise ValueError(
                    f"Dropout sequence length ({len(dropout_probs)}) must match hidden layers ({len(hidden_dims)})."
                )

        # Construction of the MLP
        layers: List[nn.Module] = []
        # Input is concatenated [theta_u, b_u], hence d_model + 1
        current_dim = d_model + 1

        act_layer_type = self._get_activation_type(activation)

        for h_dim, p_drop in zip(hidden_dims, dropout_probs):
            layers.append(nn.Linear(current_dim, h_dim))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            elif use_layernorm:
                layers.append(nn.LayerNorm(h_dim))

            layers.append(act_layer_type())

            if p_drop > 0.0:
                layers.append(nn.Dropout(p_drop))

            current_dim = h_dim

        # Final output projection (no activation/norm)
        layers.append(nn.Linear(current_dim, out_dim))

        self.net = nn.Sequential(*layers)

    @staticmethod
    def _get_activation_type(name: str) -> type[nn.Module]:
        """Maps string names to nn.Module activation classes."""
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "silu": nn.SiLU
        }
        name = name.lower()
        if name not in activations:
            raise ValueError(f"Activation '{name}' is not supported. Options: {list(activations.keys())}")
        return activations[name]

    def forward(self, theta_u: torch.Tensor, b_u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for attribute prediction.

        Args:
            theta_u: User embedding tensor of shape [Batch, D_model].
            b_u: User bias tensor of shape [Batch, 1].

        Returns:
            torch.Tensor: Predicted attributes of shape [Batch, out_dim].
        """
        # Ensure b_u is [B, 1] for concatenation
        if b_u.dim() == 1:
            b_u = b_u.unsqueeze(-1)

        combined_input = torch.cat([theta_u, b_u], dim=-1)
        return self.net(combined_input)