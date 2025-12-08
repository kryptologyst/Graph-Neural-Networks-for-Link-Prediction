"""Training framework for link prediction."""

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from tqdm import tqdm

from src.eval.metrics import LinkPredictionEvaluator
from src.models.decoder import DotProductDecoder
from src.utils.device import get_device


class LinkPredictionTrainer:
    """Trainer for link prediction models.
    
    This class handles training, validation, and testing of link prediction models
    with comprehensive logging and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        decoder: nn.Module,
        data: Data,
        device: str = "auto",
        learning_rate: float = 0.01,
        weight_decay: float = 5e-4,
        patience: int = 50,
        min_delta: float = 0.001,
        gradient_clip_norm: float = 1.0,
        negative_sampling_ratio: float = 1.0,
        checkpoint_dir: str = "checkpoints/",
        log_dir: str = "logs/",
    ):
        """Initialize the trainer.
        
        Args:
            model: The GNN encoder model.
            decoder: The decoder for link prediction.
            data: The graph data.
            device: Device to use for training.
            learning_rate: Learning rate for optimizer.
            weight_decay: Weight decay for optimizer.
            patience: Early stopping patience.
            min_delta: Minimum change for early stopping.
            gradient_clip_norm: Gradient clipping norm.
            negative_sampling_ratio: Ratio of negative samples to positive samples.
            checkpoint_dir: Directory to save checkpoints.
            log_dir: Directory to save logs.
        """
        self.model = model
        self.decoder = decoder
        self.data = data
        self.device = get_device(device)
        self.negative_sampling_ratio = negative_sampling_ratio
        
        # Move models to device
        self.model = self.model.to(self.device)
        self.decoder = self.decoder.to(self.device)
        
        # Move data to device
        self.data = self.data.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Setup evaluator
        self.evaluator = LinkPredictionEvaluator()
        
        # Training state
        self.patience = patience
        self.min_delta = min_delta
        self.gradient_clip_norm = gradient_clip_norm
        self.best_val_score = 0.0
        self.epochs_without_improvement = 0
        
        # Directories
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Training history
        self.train_losses = []
        self.val_scores = []
        self.test_scores = []
    
    def train_epoch(self) -> float:
        """Train for one epoch.
        
        Returns:
            float: Training loss.
        """
        self.model.train()
        self.decoder.train()
        
        self.optimizer.zero_grad()
        
        # Forward pass
        z = self.model(self.data.x, self.data.train_pos_edge_index)
        
        # Positive edge scores
        pos_scores = self.decoder(z, self.data.train_pos_edge_index)
        pos_loss = -F.logsigmoid(pos_scores).mean()
        
        # Negative edge scores
        num_neg_samples = int(self.data.train_pos_edge_index.size(1) * self.negative_sampling_ratio)
        neg_edge_index = negative_sampling(
            edge_index=self.data.train_pos_edge_index,
            num_nodes=self.data.num_nodes,
            num_neg_samples=num_neg_samples,
        )
        neg_scores = self.decoder(z, neg_edge_index)
        neg_loss = -F.logsigmoid(-neg_scores).mean()
        
        # Total loss
        loss = pos_loss + neg_loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.decoder.parameters()),
                self.gradient_clip_norm,
            )
        
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, split: str = "val") -> Dict[str, float]:
        """Evaluate the model on validation or test set.
        
        Args:
            split: Split to evaluate on ('val' or 'test').
            
        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        self.model.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            # Forward pass
            z = self.model(self.data.x, self.data.train_pos_edge_index)
            
            if split == "val":
                pos_edge_index = self.data.val_pos_edge_index
                neg_edge_index = self.data.val_neg_edge_index
            elif split == "test":
                pos_edge_index = self.data.test_pos_edge_index
                neg_edge_index = self.data.test_neg_edge_index
            else:
                raise ValueError(f"Unknown split: {split}")
            
            # Compute scores
            pos_scores = self.decoder(z, pos_edge_index)
            neg_scores = self.decoder(z, neg_edge_index)
            
            # Evaluate
            metrics = self.evaluator.evaluate(pos_scores, neg_scores)
            
            return metrics
    
    def train(
        self,
        epochs: int = 200,
        eval_every: int = 10,
        save_best: bool = True,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """Train the model.
        
        Args:
            epochs: Number of training epochs.
            eval_every: Evaluate every N epochs.
            save_best: Whether to save the best model.
            verbose: Whether to print progress.
            
        Returns:
            Dict[str, list]: Training history.
        """
        if verbose:
            print(f"Training on device: {self.device}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        
        for epoch in tqdm(range(epochs), desc="Training", disable=not verbose):
            # Training
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Evaluation
            if (epoch + 1) % eval_every == 0:
                val_metrics = self.evaluate("val")
                test_metrics = self.evaluate("test")
                
                val_score = val_metrics["roc_auc"]
                test_score = test_metrics["roc_auc"]
                
                self.val_scores.append(val_score)
                self.test_scores.append(test_score)
                
                if verbose:
                    print(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f}, "
                          f"Val AUC={val_score:.4f}, Test AUC={test_score:.4f}")
                
                # Early stopping
                if val_score > self.best_val_score + self.min_delta:
                    self.best_val_score = val_score
                    self.epochs_without_improvement = 0
                    
                    if save_best:
                        self.save_checkpoint(epoch, val_score, test_score)
                else:
                    self.epochs_without_improvement += 1
                
                if self.epochs_without_improvement >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return {
            "train_losses": self.train_losses,
            "val_scores": self.val_scores,
            "test_scores": self.test_scores,
        }
    
    def save_checkpoint(
        self,
        epoch: int,
        val_score: float,
        test_score: float,
        filename: Optional[str] = None,
    ) -> None:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch.
            val_score: Validation score.
            test_score: Test score.
            filename: Optional filename for checkpoint.
        """
        if filename is None:
            filename = f"best_model_epoch_{epoch}_val_{val_score:.4f}.pt"
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_score": val_score,
            "test_score": test_score,
            "best_val_score": self.best_val_score,
        }
        
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint.
        
        Args:
            filename: Checkpoint filename.
        """
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, filename), map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        self.best_val_score = checkpoint["best_val_score"]
        
        return checkpoint["epoch"], checkpoint["val_score"], checkpoint["test_score"]
