"""Unit tests for GNN models."""

import pytest
import torch
import torch.nn as nn

from src.models.gcn import GCNEncoder
from src.models.graphsage import GraphSAGEEncoder
from src.models.gat import GATEncoder
from src.models.decoder import DotProductDecoder, CosineDecoder, MLPDecoder, BilinearDecoder


class TestGCNEncoder:
    """Test GCN encoder."""
    
    def test_gcn_initialization(self):
        """Test GCN encoder initialization."""
        model = GCNEncoder(
            in_channels=64,
            hidden_dim=32,
            out_channels=16,
            num_layers=2,
            dropout=0.5,
        )
        
        assert model.in_channels == 64
        assert model.hidden_dim == 32
        assert model.out_channels == 16
        assert model.num_layers == 2
        assert model.dropout == 0.5
    
    def test_gcn_forward(self):
        """Test GCN encoder forward pass."""
        model = GCNEncoder(
            in_channels=64,
            hidden_dim=32,
            out_channels=16,
            num_layers=2,
        )
        
        x = torch.randn(100, 64)
        edge_index = torch.randint(0, 100, (2, 200))
        
        output = model(x, edge_index)
        
        assert output.shape == (100, 16)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestGraphSAGEEncoder:
    """Test GraphSAGE encoder."""
    
    def test_graphsage_initialization(self):
        """Test GraphSAGE encoder initialization."""
        model = GraphSAGEEncoder(
            in_channels=64,
            hidden_dim=32,
            out_channels=16,
            num_layers=2,
            aggregation="mean",
        )
        
        assert model.in_channels == 64
        assert model.hidden_dim == 32
        assert model.out_channels == 16
        assert model.num_layers == 2
        assert model.aggregation == "mean"
    
    def test_graphsage_forward(self):
        """Test GraphSAGE encoder forward pass."""
        model = GraphSAGEEncoder(
            in_channels=64,
            hidden_dim=32,
            out_channels=16,
            num_layers=2,
        )
        
        x = torch.randn(100, 64)
        edge_index = torch.randint(0, 100, (2, 200))
        
        output = model(x, edge_index)
        
        assert output.shape == (100, 16)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestGATEncoder:
    """Test GAT encoder."""
    
    def test_gat_initialization(self):
        """Test GAT encoder initialization."""
        model = GATEncoder(
            in_channels=64,
            hidden_dim=32,
            out_channels=16,
            num_layers=2,
            heads=8,
        )
        
        assert model.in_channels == 64
        assert model.hidden_dim == 32
        assert model.out_channels == 16
        assert model.num_layers == 2
        assert model.heads == 8
    
    def test_gat_forward(self):
        """Test GAT encoder forward pass."""
        model = GATEncoder(
            in_channels=64,
            hidden_dim=32,
            out_channels=16,
            num_layers=2,
            heads=4,
        )
        
        x = torch.randn(100, 64)
        edge_index = torch.randint(0, 100, (2, 200))
        
        output = model(x, edge_index)
        
        assert output.shape == (100, 16)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gat_attention_weights(self):
        """Test GAT attention weights extraction."""
        model = GATEncoder(
            in_channels=64,
            hidden_dim=32,
            out_channels=16,
            num_layers=2,
            heads=4,
        )
        
        x = torch.randn(100, 64)
        edge_index = torch.randint(0, 100, (2, 200))
        
        attention_weights = model.get_attention_weights(x, edge_index)
        
        assert len(attention_weights) == 2  # Number of layers
        assert attention_weights[0].shape == (2, 200)  # Edge index shape


class TestDecoders:
    """Test decoders."""
    
    def test_dot_product_decoder(self):
        """Test dot product decoder."""
        decoder = DotProductDecoder()
        
        z = torch.randn(100, 64)
        edge_index = torch.randint(0, 100, (2, 50))
        
        scores = decoder(z, edge_index)
        
        assert scores.shape == (50,)
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()
    
    def test_cosine_decoder(self):
        """Test cosine decoder."""
        decoder = CosineDecoder()
        
        z = torch.randn(100, 64)
        edge_index = torch.randint(0, 100, (2, 50))
        
        scores = decoder(z, edge_index)
        
        assert scores.shape == (50,)
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()
        assert torch.all(scores >= -1) and torch.all(scores <= 1)  # Cosine similarity range
    
    def test_mlp_decoder(self):
        """Test MLP decoder."""
        decoder = MLPDecoder(
            in_channels=128,  # 2 * 64
            hidden_dim=32,
            num_layers=2,
        )
        
        z = torch.randn(100, 64)
        edge_index = torch.randint(0, 100, (2, 50))
        
        scores = decoder(z, edge_index)
        
        assert scores.shape == (50,)
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()
    
    def test_bilinear_decoder(self):
        """Test bilinear decoder."""
        decoder = BilinearDecoder(in_channels=64)
        
        z = torch.randn(100, 64)
        edge_index = torch.randint(0, 100, (2, 50))
        
        scores = decoder(z, edge_index)
        
        assert scores.shape == (50,)
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()


class TestModelIntegration:
    """Test model integration."""
    
    def test_gcn_with_dot_product(self):
        """Test GCN with dot product decoder."""
        encoder = GCNEncoder(
            in_channels=64,
            hidden_dim=32,
            out_channels=16,
            num_layers=2,
        )
        decoder = DotProductDecoder()
        
        x = torch.randn(100, 64)
        edge_index = torch.randint(0, 100, (2, 200))
        
        embeddings = encoder(x, edge_index)
        scores = decoder(embeddings, edge_index)
        
        assert embeddings.shape == (100, 16)
        assert scores.shape == (200,)
    
    def test_graphsage_with_cosine(self):
        """Test GraphSAGE with cosine decoder."""
        encoder = GraphSAGEEncoder(
            in_channels=64,
            hidden_dim=32,
            out_channels=16,
            num_layers=2,
        )
        decoder = CosineDecoder()
        
        x = torch.randn(100, 64)
        edge_index = torch.randint(0, 100, (2, 200))
        
        embeddings = encoder(x, edge_index)
        scores = decoder(embeddings, edge_index)
        
        assert embeddings.shape == (100, 16)
        assert scores.shape == (200,)
    
    def test_gat_with_mlp(self):
        """Test GAT with MLP decoder."""
        encoder = GATEncoder(
            in_channels=64,
            hidden_dim=32,
            out_channels=16,
            num_layers=2,
            heads=4,
        )
        decoder = MLPDecoder(
            in_channels=32,  # 2 * 16
            hidden_dim=16,
            num_layers=2,
        )
        
        x = torch.randn(100, 64)
        edge_index = torch.randint(0, 100, (2, 200))
        
        embeddings = encoder(x, edge_index)
        scores = decoder(embeddings, edge_index)
        
        assert embeddings.shape == (100, 16)
        assert scores.shape == (200,)


if __name__ == "__main__":
    pytest.main([__file__])
