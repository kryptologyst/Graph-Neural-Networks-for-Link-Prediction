"""Graph Attention Network encoder for link prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATEncoder(nn.Module):
    """Graph Attention Network encoder for link prediction.
    
    This implementation follows the GAT paper by Veličković et al. (2018)
    with multi-head attention and support for different attention mechanisms.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        activation: str = "elu",
        heads: int = 8,
        concat: bool = True,
        negative_slope: float = 0.2,
    ):
        """Initialize the GAT encoder.
        
        Args:
            in_channels: Number of input features.
            hidden_dim: Hidden dimension size.
            out_channels: Number of output features.
            num_layers: Number of GAT layers.
            dropout: Dropout rate.
            activation: Activation function name.
            heads: Number of attention heads.
            concat: Whether to concatenate multi-head outputs.
            negative_slope: Negative slope for LeakyReLU.
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        self.concat = concat
        
        # Activation function
        self.activation = getattr(F, activation)
        
        # Build layers
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(
            GATConv(
                in_channels,
                hidden_dim,
                heads=heads,
                dropout=dropout,
                concat=concat,
                negative_slope=negative_slope,
            )
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            in_dim = hidden_dim * heads if concat else hidden_dim
            self.convs.append(
                GATConv(
                    in_dim,
                    hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    concat=concat,
                    negative_slope=negative_slope,
                )
            )
        
        # Output layer
        if num_layers > 1:
            in_dim = hidden_dim * heads if concat else hidden_dim
            self.convs.append(
                GATConv(
                    in_dim,
                    out_channels,
                    heads=1,  # Single head for output
                    dropout=dropout,
                    concat=False,
                    negative_slope=negative_slope,
                )
            )
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            torch.Tensor: Node embeddings.
        """
        h = x
        
        for i, conv in enumerate(self.convs):
            # Convolution
            h = conv(h, edge_index)
            
            # Activation (except for last layer)
            if i < len(self.convs) - 1:
                h = self.activation(h)
                h = self.dropout_layer(h)
        
        return h
    
    def get_attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor) -> list:
        """Get attention weights for visualization.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            list: List of attention weights for each layer.
        """
        attention_weights = []
        h = x
        
        for conv in self.convs:
            # Get attention weights
            _, att = conv(h, edge_index, return_attention_weights=True)
            attention_weights.append(att)
            
            # Forward pass
            h = conv(h, edge_index)
            if conv != self.convs[-1]:  # Not last layer
                h = self.activation(h)
                h = self.dropout_layer(h)
        
        return attention_weights
