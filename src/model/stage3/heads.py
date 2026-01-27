# model/stage3/heads.py
"""
UNISON Stage 3: Prediction Heads

This module implements Stage 3 of UNISON, which uses the learned user parameters
(theta_u, b_u) from Stage 2 as "functional embeddings" to predict bag-level
characteristics.

Architecture:
    Input: [theta_u, b_u] ∈ R^{d_model + 1}  # Concatenated user parameters
    Output: Bag-level predictions (e.g., user demographics, disease state)

Key Insight:
    The parameters (theta_u, b_u) that capture item-scoring preferences also encode
    higher-level bag characteristics. This eliminates the need for auxiliary features
    while achieving strong classification performance (Section 3.1, Equation 6).

Modules:
    - UserAttrHead: MLP classifier for bag-level attributes
      Supports flexible architectures with per-layer dropout, normalization options

Typical Usage:
    # Initialize classifier for binary prediction (e.g., gender)
    head = UserAttrHead(
        d_model=64,           # dimension of theta_u
        out_dim=1,            # binary classification
        hidden=(128, 64),     # two hidden layers
        dropout=0.3,          # dropout after each hidden layer
        activation="relu"
    )

    # Forward pass
    theta_u = torch.randn(32, 64)   # [batch_size, d_model]
    b_u = torch.randn(32, 1)        # [batch_size, 1]
    logits = head(theta_u, b_u)     # [batch_size, 1]
"""

from typing import Union, Sequence, List

import torch
import torch.nn as nn


def _build_mlp(
    d_in,
    d_out,
    hidden=(128, 64),
    dropout=0.0,
    activation="relu"
):
    """
    Build a simple feedforward MLP for classification.

    Args:
        d_in: Input dimension
        d_out: Output dimension (e.g., 1 for binary, num_classes for multiclass)
        hidden: Tuple of hidden layer sizes
        dropout: Dropout probability (applied uniformly after all hidden layers)
        activation: Activation function name

    Returns:
        nn.Sequential module implementing the MLP
    """
    acts = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "none": nn.Identity
    }
    Act = acts.get(activation.lower(), nn.ReLU)

    layers = []
    prev = d_in

    # Build hidden layers
    for h in hidden:
        layers += [nn.Linear(prev, h), Act()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h

    # Final output layer (no activation - raw logits)
    layers.append(nn.Linear(prev, d_out))
    return nn.Sequential(*layers)


class UserAttrHead(nn.Module):
    """
    Stage 3 classifier for predicting bag-level attributes from functional embeddings.

    This module takes the concatenated user parameters [theta_u; b_u] learned in Stage 2
    and maps them to bag-level predictions. The network architecture is flexible to
    accommodate different prediction tasks and dataset characteristics.

    Design Rationale:
        - Functional embeddings (theta_u, b_u) encode both preference patterns and
          underlying bag characteristics (e.g., user demographics, disease state)
        - Per-layer dropout control enables fine-grained regularization
        - Optional normalization (BatchNorm/LayerNorm) for training stability

    Parameters
    ----------
    d_model : int
        Dimension of theta_u (user embedding from Stage 2)
    out_dim : int
        Output dimension:
        - 1 for binary classification or scalar regression
        - num_classes for multiclass classification
    hidden : tuple of int
        Hidden layer sizes, e.g., (128, 64) for two hidden layers.
        Use () for direct linear mapping from input to output.
    dropout : float or Sequence[float]
        Dropout probability:
        - If float: same dropout rate applied after all hidden layers
        - If sequence: per-layer dropout rates (must match len(hidden))
    activation : str
        Activation function: "relu", "gelu", "tanh", "sigmoid"
    use_batchnorm : bool
        Apply BatchNorm1d after each linear layer (before activation).
        Useful for large batch sizes and deep networks.
    use_layernorm : bool
        Apply LayerNorm after each linear layer (before activation).
        Alternative to BatchNorm; cannot be used simultaneously.

    Raises
    ------
    ValueError
        If both use_batchnorm and use_layernorm are True
        If dropout sequence length doesn't match hidden layer count

    Examples
    --------
    Binary classification with uniform dropout:
        >>> head = UserAttrHead(d_model=64, out_dim=1, hidden=(128, 64), dropout=0.3)

    Multiclass with per-layer dropout:
        >>> head = UserAttrHead(
        ...     d_model=64,
        ...     out_dim=5,
        ...     hidden=(128, 64, 32),
        ...     dropout=[0.3, 0.4, 0.5]  # increasing dropout in deeper layers
        ... )

    Shallow network with BatchNorm:
        >>> head = UserAttrHead(
        ...     d_model=64,
        ...     out_dim=1,
        ...     hidden=(128,),
        ...     use_batchnorm=True
        ... )
    """

    def __init__(
        self,
        d_model: int,
        out_dim: int = 1,
        hidden=(128, 64),
        dropout: Union[float, Sequence[float]] = 0.0,
        activation: str = "relu",
        use_batchnorm: bool = False,
        use_layernorm: bool = False,
    ):
        super().__init__()

        # Validate normalization options
        if use_batchnorm and use_layernorm:
            raise ValueError("Choose either batchnorm or layernorm, not both.")

        # Normalize hidden to tuple
        if isinstance(hidden, int):
            hidden = (hidden,)
        hidden = tuple(hidden)

        # Build dropout schedule
        # If single float: replicate for all layers
        # If sequence: validate length matches hidden layers
        if isinstance(dropout, (float, int)):
            dropouts = [float(dropout)] * len(hidden)
        else:
            dropouts = list(dropout)
            if len(dropouts) != len(hidden):
                raise ValueError(
                    f"dropout length ({len(dropouts)}) must match hidden length ({len(hidden)})."
                )

        # Get activation layer factory
        act_layer = self._get_activation(activation)

        # Build network architecture
        layers: List[nn.Module] = []
        in_dim = d_model + 1  # Input: [theta_u; b_u]

        for h_dim, p in zip(hidden, dropouts):
            # Linear transformation
            layers.append(nn.Linear(in_dim, h_dim))

            # Optional normalization (before activation)
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(h_dim))

            # Activation
            layers.append(act_layer)

            # Dropout regularization
            if p > 0.0:
                layers.append(nn.Dropout(p))

            in_dim = h_dim


        # Final output layer (no normalization, activation, or dropout)
        # Returns raw logits for classification or unbounded values for regression
        layers.append(nn.Linear(in_dim, out_dim))

        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights for Stage 3 head."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """
        Factory method for activation layers.

        Args:
            name: Activation function name (case-insensitive)

        Returns:
            Instantiated activation module

        Raises:
            ValueError: If activation name is not recognized
        """
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "tanh":
            return nn.Tanh()
        if name == "sigmoid":
            return nn.Sigmoid()
        raise ValueError(f"Unsupported activation: {name}")

    def forward(
        self,
        theta_u: torch.Tensor,
        b_u: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict bag-level attributes from functional embeddings.

        Args:
            theta_u: User embeddings from Stage 2 [B, d_model]
                     These are the learned linear weights for item scoring
            b_u: User biases from Stage 2 [B, 1]
                 These are the learned intercepts for item scoring

        Returns:
            attr_pred: Bag-level predictions [B, out_dim]
                       - For binary classification: logits (apply sigmoid/BCEWithLogitsLoss)
                       - For multiclass: logits (apply softmax/CrossEntropyLoss)
                       - For regression: unbounded predictions (apply MSE loss)

        Notes:
            The concatenation [theta_u; b_u] serves as a compact representation of
            the bag's scoring behavior, which empirically correlates with bag-level
            characteristics across diverse domains (entertainment, immunology, ecology).
        """
        # Concatenate functional embeddings
        x = torch.cat([theta_u, b_u], dim=-1)  # [B, d_model + 1]

        # Forward through MLP classifier
        return self.net(x)  # [B, out_dim]


# ====================== Example Usage ====================== #

if __name__ == "__main__":
    torch.manual_seed(42)

    # Example 1: Binary classification with uniform dropout
    print("=== Example 1: Binary Classification ===")
    head_binary = UserAttrHead(
        d_model=64,
        out_dim=1,
        hidden=(128, 64),
        dropout=0.3,
        activation="relu"
    )

    # Simulate functional embeddings from Stage 2
    batch_size = 16
    theta_u = torch.randn(batch_size, 64)
    b_u = torch.randn(batch_size, 1)

    logits = head_binary(theta_u, b_u)
    print(f"Output shape: {logits.shape}")  # [16, 1]
    print(f"Sample logits: {logits[:3].squeeze()}\n")

    # Example 2: Multiclass classification with per-layer dropout
    print("=== Example 2: Multiclass Classification ===")
    head_multiclass = UserAttrHead(
        d_model=64,
        out_dim=5,  # 5 classes
        hidden=(128, 64, 32),
        dropout=[0.2, 0.3, 0.4],  # increasing dropout
        activation="gelu"
    )

    logits = head_multiclass(theta_u, b_u)
    print(f"Output shape: {logits.shape}")  # [16, 5]
    print(f"Sample logits:\n{logits[:2]}\n")

    # Example 3: Shallow network with BatchNorm
    print("=== Example 3: With BatchNorm ===")
    head_bn = UserAttrHead(
        d_model=64,
        out_dim=1,
        hidden=(128,),
        dropout=0.2,
        use_batchnorm=True
    )

    logits = head_bn(theta_u, b_u)
    print(f"Output shape: {logits.shape}")  # [16, 1]
    print(f"Sample logits: {logits[:3].squeeze()}")