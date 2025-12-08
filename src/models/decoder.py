"""Decoders for link prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductDecoder(nn.Module):
    """Dot product decoder for link prediction.
    
    This decoder computes the similarity between node embeddings using dot product.
    """
    
    def __init__(self):
        """Initialize the dot product decoder."""
        super().__init__()
    
    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            z: Node embeddings.
            edge_index: Edge indices.
            
        Returns:
            torch.Tensor: Edge scores.
        """
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)


class CosineDecoder(nn.Module):
    """Cosine similarity decoder for link prediction.
    
    This decoder computes the cosine similarity between node embeddings.
    """
    
    def __init__(self):
        """Initialize the cosine decoder."""
        super().__init__()
    
    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            z: Node embeddings.
            edge_index: Edge indices.
            
        Returns:
            torch.Tensor: Edge scores.
        """
        z_src = z[edge_index[0]]
        z_dst = z[edge_index[1]]
        
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(z_src, z_dst, dim=1)
        
        return cos_sim


class MLPDecoder(nn.Module):
    """Multi-layer perceptron decoder for link prediction.
    
    This decoder uses an MLP to predict edge existence from concatenated embeddings.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
        activation: str = "relu",
    ):
        """Initialize the MLP decoder.
        
        Args:
            in_channels: Input dimension (2 * embedding_dim).
            hidden_dim: Hidden dimension size.
            num_layers: Number of layers.
            dropout: Dropout rate.
            activation: Activation function name.
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Activation function
        self.activation = getattr(F, activation)
        
        # Build layers
        layers = []
        in_dim = in_channels
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            z: Node embeddings.
            edge_index: Edge indices.
            
        Returns:
            torch.Tensor: Edge scores.
        """
        z_src = z[edge_index[0]]
        z_dst = z[edge_index[1]]
        
        # Concatenate embeddings
        z_concat = torch.cat([z_src, z_dst], dim=1)
        
        # MLP forward pass
        scores = self.mlp(z_concat).squeeze()
        
        return scores


class BilinearDecoder(nn.Module):
    """Bilinear decoder for link prediction.
    
    This decoder uses a bilinear transformation to compute edge scores.
    """
    
    def __init__(self, in_channels: int):
        """Initialize the bilinear decoder.
        
        Args:
            in_channels: Input dimension (embedding_dim).
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.bilinear = nn.Bilinear(in_channels, in_channels, 1)
    
    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            z: Node embeddings.
            edge_index: Edge indices.
            
        Returns:
            torch.Tensor: Edge scores.
        """
        z_src = z[edge_index[0]]
        z_dst = z[edge_index[1]]
        
        scores = self.bilinear(z_src, z_dst).squeeze()
        
        return scores
