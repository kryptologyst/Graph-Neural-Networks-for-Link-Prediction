"""Data loading and preprocessing utilities."""

from .dataset import LinkPredictionDataset, create_synthetic_dataset

__all__ = [
    "LinkPredictionDataset",
    "create_synthetic_dataset",
]
