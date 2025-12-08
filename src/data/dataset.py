"""Data loading and preprocessing utilities for link prediction."""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import (
    add_self_loops,
    negative_sampling,
    remove_self_loops,
    train_test_split_edges,
)
from torch_geometric.transforms import NormalizeFeatures


class LinkPredictionDataset:
    """Dataset wrapper for link prediction tasks.
    
    This class handles loading, preprocessing, and splitting of graph datasets
    for link prediction tasks.
    """
    
    def __init__(
        self,
        dataset_name: str,
        root: str = "data/",
        transform: Optional[object] = None,
        train_ratio: float = 0.85,
        val_ratio: float = 0.10,
        test_ratio: float = 0.05,
        negative_sampling_ratio: float = 1.0,
        inductive: bool = False,
    ):
        """Initialize the dataset.
        
        Args:
            dataset_name: Name of the dataset (cora, citeseer, pubmed).
            root: Root directory for data storage.
            transform: Optional transform to apply to the data.
            train_ratio: Ratio of edges for training.
            val_ratio: Ratio of edges for validation.
            test_ratio: Ratio of edges for testing.
            negative_sampling_ratio: Ratio of negative samples to positive samples.
            inductive: Whether to use inductive setting.
        """
        self.dataset_name = dataset_name
        self.root = root
        self.transform = transform or NormalizeFeatures()
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.negative_sampling_ratio = negative_sampling_ratio
        self.inductive = inductive
        
        self._load_dataset()
        self._prepare_link_prediction_data()
    
    def _load_dataset(self) -> None:
        """Load the dataset."""
        if self.dataset_name.lower() in ["cora", "citeseer", "pubmed"]:
            dataset = Planetoid(
                root=self.root,
                name=self.dataset_name.capitalize(),
                transform=self.transform,
            )
            self.data = dataset[0]
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        # Add self-loops if needed
        self.data.edge_index = add_self_loops(self.data.edge_index)[0]
    
    def _prepare_link_prediction_data(self) -> None:
        """Prepare data for link prediction by splitting edges."""
        # Split edges into train/val/test
        self.data = train_test_split_edges(
            self.data,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
        )
        
        # Generate negative samples for validation and test
        num_nodes = self.data.num_nodes
        
        # Validation negative edges
        val_neg_edge_index = negative_sampling(
            edge_index=self.data.train_pos_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=self.data.val_pos_edge_index.size(1),
        )
        self.data.val_neg_edge_index = val_neg_edge_index
        
        # Test negative edges
        test_neg_edge_index = negative_sampling(
            edge_index=self.data.train_pos_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=self.data.test_pos_edge_index.size(1),
        )
        self.data.test_neg_edge_index = test_neg_edge_index
    
    def get_data(self) -> Data:
        """Get the processed data.
        
        Returns:
            Data: Processed PyTorch Geometric data object.
        """
        return self.data
    
    def get_num_features(self) -> int:
        """Get the number of node features.
        
        Returns:
            int: Number of node features.
        """
        return self.data.num_node_features
    
    def get_num_nodes(self) -> int:
        """Get the number of nodes.
        
        Returns:
            int: Number of nodes.
        """
        return self.data.num_nodes
    
    def get_num_classes(self) -> int:
        """Get the number of node classes.
        
        Returns:
            int: Number of node classes.
        """
        return int(self.data.y.max().item()) + 1
    
    def generate_negative_samples(
        self,
        edge_index: torch.Tensor,
        num_neg_samples: int,
    ) -> torch.Tensor:
        """Generate negative samples for training.
        
        Args:
            edge_index: Positive edge indices.
            num_neg_samples: Number of negative samples to generate.
            
        Returns:
            torch.Tensor: Negative edge indices.
        """
        return negative_sampling(
            edge_index=edge_index,
            num_nodes=self.data.num_nodes,
            num_neg_samples=num_neg_samples,
        )


def create_synthetic_dataset(
    num_nodes: int = 1000,
    num_features: int = 64,
    num_classes: int = 5,
    edge_prob: float = 0.01,
    seed: int = 42,
) -> Data:
    """Create a synthetic graph dataset for testing.
    
    Args:
        num_nodes: Number of nodes in the graph.
        num_features: Number of features per node.
        num_classes: Number of node classes.
        edge_prob: Probability of edge existence.
        seed: Random seed.
        
    Returns:
        Data: Synthetic graph data.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate random features
    x = torch.randn(num_nodes, num_features)
    
    # Generate random labels
    y = torch.randint(0, num_classes, (num_nodes,))
    
    # Generate random edges
    edge_index = torch.randint(0, num_nodes, (2, int(num_nodes * (num_nodes - 1) * edge_prob)))
    
    # Remove self-loops
    edge_index = remove_self_loops(edge_index)[0]
    
    # Create data object
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data
