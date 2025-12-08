"""GraphSAGE encoder for link prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(nn.Module):
    """GraphSAGE encoder for link prediction.
    
    This implementation follows the GraphSAGE paper by Hamilton et al. (2017)
    with support for different aggregation methods.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        activation: str = "relu",
        aggregation: str = "mean",
        normalize: bool = True,
    ):
        """Initialize the GraphSAGE encoder.
        
        Args:
            in_channels: Number of input features.
            hidden_dim: Hidden dimension size.
            out_channels: Number of output features.
            num_layers: Number of GraphSAGE layers.
            dropout: Dropout rate.
            activation: Activation function name.
            aggregation: Aggregation method ('mean', 'max', 'lstm').
            normalize: Whether to normalize embeddings.
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.normalize = normalize
        
        # Activation function
        self.activation = getattr(F, activation)
        
        # Build layers
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_dim, aggr=aggregation))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregation))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, out_channels, aggr=aggregation))
        
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
        
        # Normalize embeddings
        if self.normalize:
            h = F.normalize(h, p=2, dim=1)
        
        return h
