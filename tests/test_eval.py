"""Unit tests for evaluation metrics."""

import pytest
import torch
import numpy as np

from src.eval.metrics import LinkPredictionEvaluator, compute_filtered_metrics


class TestLinkPredictionEvaluator:
    """Test LinkPredictionEvaluator class."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = LinkPredictionEvaluator(hits_at_k_values=[1, 3, 10])
        
        assert evaluator.hits_at_k_values == [1, 3, 10]
        assert hasattr(evaluator, "auroc_metric")
        assert hasattr(evaluator, "ap_metric")
    
    def test_evaluate_perfect_scores(self):
        """Test evaluation with perfect scores."""
        evaluator = LinkPredictionEvaluator()
        
        # Perfect separation
        pos_scores = torch.ones(100)
        neg_scores = torch.zeros(100)
        
        metrics = evaluator.evaluate(pos_scores, neg_scores)
        
        assert metrics["roc_auc"] == 1.0
        assert metrics["average_precision"] == 1.0
    
    def test_evaluate_random_scores(self):
        """Test evaluation with random scores."""
        evaluator = LinkPredictionEvaluator()
        
        # Random scores
        pos_scores = torch.rand(100)
        neg_scores = torch.rand(100)
        
        metrics = evaluator.evaluate(pos_scores, neg_scores)
        
        # Should be around 0.5 for random scores
        assert 0.3 <= metrics["roc_auc"] <= 0.7
        assert 0.3 <= metrics["average_precision"] <= 0.7
    
    def test_evaluate_hits_at_k(self):
        """Test Hits@K evaluation."""
        evaluator = LinkPredictionEvaluator(hits_at_k_values=[1, 3, 10])
        
        # Perfect separation
        pos_scores = torch.ones(100)
        neg_scores = torch.zeros(100)
        
        metrics = evaluator.evaluate(pos_scores, neg_scores)
        
        assert "hits_at_1" in metrics
        assert "hits_at_3" in metrics
        assert "hits_at_10" in metrics
        
        # For perfect separation, Hits@K should be 1.0
        assert metrics["hits_at_1"] == 1.0
        assert metrics["hits_at_3"] == 1.0
        assert metrics["hits_at_10"] == 1.0
    
    def test_evaluate_batch(self):
        """Test batch evaluation."""
        evaluator = LinkPredictionEvaluator()
        
        # Large dataset
        pos_scores = torch.rand(1000)
        neg_scores = torch.rand(1000)
        
        metrics = evaluator.evaluate_batch(pos_scores, neg_scores, batch_size=100)
        
        assert "roc_auc" in metrics
        assert "average_precision" in metrics
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["average_precision"] <= 1.0
    
    def test_evaluate_edge_cases(self):
        """Test evaluation edge cases."""
        evaluator = LinkPredictionEvaluator()
        
        # Single positive and negative sample
        pos_scores = torch.tensor([1.0])
        neg_scores = torch.tensor([0.0])
        
        metrics = evaluator.evaluate(pos_scores, neg_scores)
        
        assert metrics["roc_auc"] == 1.0
        assert metrics["average_precision"] == 1.0
        
        # All positive scores higher than negative scores
        pos_scores = torch.tensor([0.8, 0.9, 1.0])
        neg_scores = torch.tensor([0.1, 0.2, 0.3])
        
        metrics = evaluator.evaluate(pos_scores, neg_scores)
        
        assert metrics["roc_auc"] == 1.0
        assert metrics["average_precision"] == 1.0
        
        # All negative scores higher than positive scores
        pos_scores = torch.tensor([0.1, 0.2, 0.3])
        neg_scores = torch.tensor([0.8, 0.9, 1.0])
        
        metrics = evaluator.evaluate(pos_scores, neg_scores)
        
        assert metrics["roc_auc"] == 0.0
        assert metrics["average_precision"] == 0.0
    
    def test_evaluate_nan_handling(self):
        """Test evaluation with NaN values."""
        evaluator = LinkPredictionEvaluator()
        
        # Scores with NaN values
        pos_scores = torch.tensor([1.0, float('nan'), 0.8])
        neg_scores = torch.tensor([0.5, 0.3, float('nan')])
        
        # Should handle NaN values gracefully
        try:
            metrics = evaluator.evaluate(pos_scores, neg_scores)
            # If no exception is raised, check that metrics are valid
            assert not np.isnan(metrics["roc_auc"])
            assert not np.isnan(metrics["average_precision"])
        except ValueError:
            # It's acceptable to raise ValueError for NaN inputs
            pass
    
    def test_evaluate_inf_handling(self):
        """Test evaluation with infinite values."""
        evaluator = LinkPredictionEvaluator()
        
        # Scores with infinite values
        pos_scores = torch.tensor([1.0, float('inf'), 0.8])
        neg_scores = torch.tensor([0.5, 0.3, float('-inf')])
        
        # Should handle infinite values gracefully
        try:
            metrics = evaluator.evaluate(pos_scores, neg_scores)
            # If no exception is raised, check that metrics are valid
            assert not np.isnan(metrics["roc_auc"])
            assert not np.isnan(metrics["average_precision"])
        except ValueError:
            # It's acceptable to raise ValueError for infinite inputs
            pass


class TestFilteredMetrics:
    """Test filtered metrics computation."""
    
    def test_compute_filtered_metrics(self):
        """Test filtered metrics computation."""
        pos_scores = torch.ones(100)
        neg_scores = torch.zeros(100)
        pos_edges = torch.randint(0, 100, (2, 100))
        neg_edges = torch.randint(0, 100, (2, 100))
        all_edges = torch.randint(0, 100, (2, 200))
        
        metrics = compute_filtered_metrics(
            pos_scores, neg_scores, pos_edges, neg_edges, all_edges
        )
        
        assert "filtered_roc_auc" in metrics
        assert "filtered_average_precision" in metrics
        assert 0.0 <= metrics["filtered_roc_auc"] <= 1.0
        assert 0.0 <= metrics["filtered_average_precision"] <= 1.0
    
    def test_compute_filtered_metrics_perfect(self):
        """Test filtered metrics with perfect scores."""
        pos_scores = torch.ones(100)
        neg_scores = torch.zeros(100)
        pos_edges = torch.randint(0, 100, (2, 100))
        neg_edges = torch.randint(0, 100, (2, 100))
        all_edges = torch.randint(0, 100, (2, 200))
        
        metrics = compute_filtered_metrics(
            pos_scores, neg_scores, pos_edges, neg_edges, all_edges
        )
        
        assert metrics["filtered_roc_auc"] == 1.0
        assert metrics["filtered_average_precision"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__])
