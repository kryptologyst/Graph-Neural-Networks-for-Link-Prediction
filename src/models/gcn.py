"""GCN encoder for link prediction."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    """Graph Convolutional Network encoder for link prediction.
    
    This implementation follows the original GCN paper by Kipf & Welling (2017)
    with additional features like residual connections and batch normalization.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        activation: str = "relu",
        normalization: str = "batch",
        residual: bool = True,
    ):
        """Initialize the GCN encoder.
        
        Args:
            in_channels: Number of input features.
            hidden_dim: Hidden dimension size.
            out_channels: Number of output features.
            num_layers: Number of GCN layers.
            dropout: Dropout rate.
            activation: Activation function name.
            normalization: Normalization type ('batch', 'layer', 'none').
            residual: Whether to use residual connections.
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        
        # Activation function
        self.activation = getattr(F, activation)
        
        # Build layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_dim))
        if normalization == "batch":
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        elif normalization == "layer":
            self.norms.append(nn.LayerNorm(hidden_dim))
        else:
            self.norms.append(nn.Identity())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            if normalization == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif normalization == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))
            else:
                self.norms.append(nn.Identity())
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, out_channels))
            if normalization == "batch":
                self.norms.append(nn.BatchNorm1d(out_channels))
            elif normalization == "layer":
                self.norms.append(nn.LayerNorm(out_channels))
            else:
                self.norms.append(nn.Identity())
        
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
        
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Store residual connection input
            if self.residual and i > 0 and h.size(-1) == conv.out_channels:
                residual = h
            
            # Convolution
            h = conv(h, edge_index)
            
            # Normalization
            h = norm(h)
            
            # Activation (except for last layer)
            if i < len(self.convs) - 1:
                h = self.activation(h)
            
            # Residual connection
            if self.residual and i > 0 and h.size(-1) == residual.size(-1):
                h = h + residual
            
            # Dropout
            if i < len(self.convs) - 1:
                h = self.dropout_layer(h)
        
        return h
