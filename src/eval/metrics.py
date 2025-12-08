"""Evaluation metrics for link prediction."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torchmetrics import AUROC, AveragePrecision


class LinkPredictionEvaluator:
    """Evaluator for link prediction tasks.
    
    This class provides comprehensive evaluation metrics for link prediction
    including ROC-AUC, Average Precision, and Hits@K.
    """
    
    def __init__(self, hits_at_k_values: List[int] = [1, 3, 10, 50]):
        """Initialize the evaluator.
        
        Args:
            hits_at_k_values: List of K values for Hits@K metric.
        """
        self.hits_at_k_values = hits_at_k_values
        self.auroc_metric = AUROC(task="binary")
        self.ap_metric = AveragePrecision(task="binary")
    
    def evaluate(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        return_metrics: bool = True,
    ) -> Dict[str, float]:
        """Evaluate link prediction performance.
        
        Args:
            pos_scores: Scores for positive edges.
            neg_scores: Scores for negative edges.
            return_metrics: Whether to return detailed metrics.
            
        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        # Combine scores and labels
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones(pos_scores.size(0)),
            torch.zeros(neg_scores.size(0))
        ])
        
        # Convert to numpy for sklearn metrics
        scores_np = scores.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # Compute metrics
        metrics = {}
        
        # ROC-AUC
        metrics["roc_auc"] = roc_auc_score(labels_np, scores_np)
        
        # Average Precision
        metrics["average_precision"] = average_precision_score(labels_np, scores_np)
        
        # Hits@K
        if return_metrics:
            hits_at_k = self._compute_hits_at_k(pos_scores, neg_scores)
            metrics.update(hits_at_k)
        
        return metrics
    
    def _compute_hits_at_k(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute Hits@K metrics.
        
        Args:
            pos_scores: Scores for positive edges.
            neg_scores: Scores for negative edges.
            
        Returns:
            Dict[str, float]: Hits@K metrics.
        """
        hits_at_k = {}
        
        for k in self.hits_at_k_values:
            hits = 0
            total = pos_scores.size(0)
            
            for i in range(total):
                pos_score = pos_scores[i]
                neg_scores_sample = neg_scores[torch.randperm(neg_scores.size(0))[:k-1]]
                
                # Count how many negative scores are higher than positive score
                higher_neg = (neg_scores_sample > pos_score).sum().item()
                
                if higher_neg < k:
                    hits += 1
            
            hits_at_k[f"hits_at_{k}"] = hits / total
        
        return hits_at_k
    
    def evaluate_batch(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        batch_size: int = 1000,
    ) -> Dict[str, float]:
        """Evaluate in batches for large datasets.
        
        Args:
            pos_scores: Scores for positive edges.
            neg_scores: Scores for negative edges.
            batch_size: Batch size for evaluation.
            
        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        all_pos_scores = []
        all_neg_scores = []
        
        # Process positive scores in batches
        for i in range(0, pos_scores.size(0), batch_size):
            batch_pos = pos_scores[i:i+batch_size]
            all_pos_scores.append(batch_pos)
        
        # Process negative scores in batches
        for i in range(0, neg_scores.size(0), batch_size):
            batch_neg = neg_scores[i:i+batch_size]
            all_neg_scores.append(batch_neg)
        
        # Concatenate all batches
        pos_scores_batch = torch.cat(all_pos_scores)
        neg_scores_batch = torch.cat(all_neg_scores)
        
        return self.evaluate(pos_scores_batch, neg_scores_batch)


def compute_filtered_metrics(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    pos_edges: torch.Tensor,
    neg_edges: torch.Tensor,
    all_edges: torch.Tensor,
) -> Dict[str, float]:
    """Compute filtered metrics for knowledge graphs.
    
    Args:
        pos_scores: Scores for positive edges.
        neg_scores: Scores for negative edges.
        pos_edges: Positive edge indices.
        neg_edges: Negative edge indices.
        all_edges: All existing edges in the graph.
        
    Returns:
        Dict[str, float]: Filtered metrics.
    """
    # This is a simplified implementation
    # In practice, you would filter out existing edges from rankings
    metrics = {}
    
    # Combine scores and labels
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([
        torch.ones(pos_scores.size(0)),
        torch.zeros(neg_scores.size(0))
    ])
    
    # Compute standard metrics
    scores_np = scores.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    
    metrics["filtered_roc_auc"] = roc_auc_score(labels_np, scores_np)
    metrics["filtered_average_precision"] = average_precision_score(labels_np, scores_np)
    
    return metrics
