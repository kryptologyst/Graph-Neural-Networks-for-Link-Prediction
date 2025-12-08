"""Unit tests for data loading and preprocessing."""

import pytest
import torch
from torch_geometric.data import Data

from src.data.dataset import LinkPredictionDataset, create_synthetic_dataset


class TestLinkPredictionDataset:
    """Test LinkPredictionDataset class."""
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        dataset = LinkPredictionDataset(
            dataset_name="cora",
            root="data/",
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )
        
        assert dataset.dataset_name == "cora"
        assert dataset.train_ratio == 0.8
        assert dataset.val_ratio == 0.1
        assert dataset.test_ratio == 0.1
    
    def test_get_data(self):
        """Test getting processed data."""
        dataset = LinkPredictionDataset(
            dataset_name="cora",
            root="data/",
        )
        
        data = dataset.get_data()
        
        assert isinstance(data, Data)
        assert hasattr(data, "train_pos_edge_index")
        assert hasattr(data, "val_pos_edge_index")
        assert hasattr(data, "test_pos_edge_index")
        assert hasattr(data, "val_neg_edge_index")
        assert hasattr(data, "test_neg_edge_index")
    
    def test_get_num_features(self):
        """Test getting number of features."""
        dataset = LinkPredictionDataset(
            dataset_name="cora",
            root="data/",
        )
        
        num_features = dataset.get_num_features()
        
        assert isinstance(num_features, int)
        assert num_features > 0
    
    def test_get_num_nodes(self):
        """Test getting number of nodes."""
        dataset = LinkPredictionDataset(
            dataset_name="cora",
            root="data/",
        )
        
        num_nodes = dataset.get_num_nodes()
        
        assert isinstance(num_nodes, int)
        assert num_nodes > 0
    
    def test_get_num_classes(self):
        """Test getting number of classes."""
        dataset = LinkPredictionDataset(
            dataset_name="cora",
            root="data/",
        )
        
        num_classes = dataset.get_num_classes()
        
        assert isinstance(num_classes, int)
        assert num_classes > 0
    
    def test_generate_negative_samples(self):
        """Test negative sample generation."""
        dataset = LinkPredictionDataset(
            dataset_name="cora",
            root="data/",
        )
        
        edge_index = torch.randint(0, 100, (2, 50))
        num_neg_samples = 100
        
        neg_edge_index = dataset.generate_negative_samples(edge_index, num_neg_samples)
        
        assert neg_edge_index.shape == (2, num_neg_samples)
        assert not torch.isnan(neg_edge_index).any()


class TestSyntheticDataset:
    """Test synthetic dataset creation."""
    
    def test_create_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        data = create_synthetic_dataset(
            num_nodes=100,
            num_features=64,
            num_classes=5,
            edge_prob=0.01,
            seed=42,
        )
        
        assert isinstance(data, Data)
        assert data.num_nodes == 100
        assert data.num_node_features == 64
        assert data.y.max().item() < 5
        assert data.edge_index.size(1) > 0
    
    def test_synthetic_dataset_deterministic(self):
        """Test that synthetic dataset is deterministic."""
        data1 = create_synthetic_dataset(seed=42)
        data2 = create_synthetic_dataset(seed=42)
        
        assert torch.equal(data1.x, data2.x)
        assert torch.equal(data1.y, data2.y)
        assert torch.equal(data1.edge_index, data2.edge_index)
    
    def test_synthetic_dataset_different_seeds(self):
        """Test that different seeds produce different datasets."""
        data1 = create_synthetic_dataset(seed=42)
        data2 = create_synthetic_dataset(seed=123)
        
        assert not torch.equal(data1.x, data2.x)
        assert not torch.equal(data1.y, data2.y)
        assert not torch.equal(data1.edge_index, data2.edge_index)


class TestDataValidation:
    """Test data validation."""
    
    def test_edge_index_shape(self):
        """Test edge index shape validation."""
        dataset = LinkPredictionDataset(
            dataset_name="cora",
            root="data/",
        )
        
        data = dataset.get_data()
        
        # Check that edge indices have correct shape
        assert data.train_pos_edge_index.shape[0] == 2
        assert data.val_pos_edge_index.shape[0] == 2
        assert data.test_pos_edge_index.shape[0] == 2
        assert data.val_neg_edge_index.shape[0] == 2
        assert data.test_neg_edge_index.shape[0] == 2
    
    def test_edge_index_values(self):
        """Test edge index value validation."""
        dataset = LinkPredictionDataset(
            dataset_name="cora",
            root="data/",
        )
        
        data = dataset.get_data()
        num_nodes = data.num_nodes
        
        # Check that edge indices are within valid range
        assert data.train_pos_edge_index.max() < num_nodes
        assert data.val_pos_edge_index.max() < num_nodes
        assert data.test_pos_edge_index.max() < num_nodes
        assert data.val_neg_edge_index.max() < num_nodes
        assert data.test_neg_edge_index.max() < num_nodes
        
        assert data.train_pos_edge_index.min() >= 0
        assert data.val_pos_edge_index.min() >= 0
        assert data.test_pos_edge_index.min() >= 0
        assert data.val_neg_edge_index.min() >= 0
        assert data.test_neg_edge_index.min() >= 0
    
    def test_feature_dimensions(self):
        """Test feature dimension validation."""
        dataset = LinkPredictionDataset(
            dataset_name="cora",
            root="data/",
        )
        
        data = dataset.get_data()
        
        # Check that features have correct dimensions
        assert data.x.shape[0] == data.num_nodes
        assert data.x.shape[1] == data.num_node_features
        assert not torch.isnan(data.x).any()
        assert not torch.isinf(data.x).any()


if __name__ == "__main__":
    pytest.main([__file__])
