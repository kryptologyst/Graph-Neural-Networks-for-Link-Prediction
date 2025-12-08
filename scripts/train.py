"""Main training script for link prediction experiments."""

import argparse
import os
from typing import Dict, Any

import torch
from omegaconf import OmegaConf

from src.data.dataset import LinkPredictionDataset
from src.models.gcn import GCNEncoder
from src.models.graphsage import GraphSAGEEncoder
from src.models.gat import GATEncoder
from src.models.decoder import DotProductDecoder, CosineDecoder, MLPDecoder, BilinearDecoder
from src.train.trainer import LinkPredictionTrainer
from src.utils.device import set_seed


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    return OmegaConf.load(config_path)


def create_model(config: Dict[str, Any], num_features: int) -> torch.nn.Module:
    """Create model based on configuration.
    
    Args:
        config: Model configuration.
        num_features: Number of input features.
        
    Returns:
        torch.nn.Module: The created model.
    """
    model_type = config.get("type", "gcn")
    
    if model_type == "gcn":
        return GCNEncoder(
            in_channels=num_features,
            hidden_dim=config.get("hidden_dim", 64),
            out_channels=config.get("out_channels", 64),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.5),
            activation=config.get("activation", "relu"),
            normalization=config.get("normalization", "batch"),
            residual=config.get("residual", True),
        )
    elif model_type == "graphsage":
        return GraphSAGEEncoder(
            in_channels=num_features,
            hidden_dim=config.get("hidden_dim", 64),
            out_channels=config.get("out_channels", 64),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.5),
            activation=config.get("activation", "relu"),
            aggregation=config.get("aggregation", "mean"),
            normalize=config.get("normalize", True),
        )
    elif model_type == "gat":
        return GATEncoder(
            in_channels=num_features,
            hidden_dim=config.get("hidden_dim", 64),
            out_channels=config.get("out_channels", 64),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.5),
            activation=config.get("activation", "elu"),
            heads=config.get("heads", 8),
            concat=config.get("concat", True),
            negative_slope=config.get("negative_slope", 0.2),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_decoder(config: Dict[str, Any], embedding_dim: int) -> torch.nn.Module:
    """Create decoder based on configuration.
    
    Args:
        config: Decoder configuration.
        embedding_dim: Embedding dimension.
        
    Returns:
        torch.nn.Module: The created decoder.
    """
    decoder_type = config.get("type", "dot_product")
    
    if decoder_type == "dot_product":
        return DotProductDecoder()
    elif decoder_type == "cosine":
        return CosineDecoder()
    elif decoder_type == "mlp":
        return MLPDecoder(
            in_channels=2 * embedding_dim,
            hidden_dim=config.get("hidden_dim", 64),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.5),
            activation=config.get("activation", "relu"),
        )
    elif decoder_type == "bilinear":
        return BilinearDecoder(embedding_dim)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train link prediction model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--model", type=str, default="gcn", help="Model type")
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config.experiment.seed = args.seed
    config.experiment.device = args.device
    config.train.epochs = args.epochs
    config.train.learning_rate = args.lr
    
    # Set random seed
    set_seed(config.experiment.seed)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = LinkPredictionDataset(
        dataset_name=args.dataset,
        root=config.data.root,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        negative_sampling_ratio=config.data.negative_sampling_ratio,
        inductive=config.data.inductive,
    )
    
    data = dataset.get_data()
    num_features = dataset.get_num_features()
    
    print(f"Dataset: {args.dataset}")
    print(f"Nodes: {data.num_nodes}")
    print(f"Features: {num_features}")
    print(f"Train edges: {data.train_pos_edge_index.size(1)}")
    print(f"Val edges: {data.val_pos_edge_index.size(1)}")
    print(f"Test edges: {data.test_pos_edge_index.size(1)}")
    
    # Create model
    model_config = config.model
    model_config.type = args.model
    model = create_model(model_config, num_features)
    
    # Create decoder
    decoder_config = config.get("decoder", {"type": "dot_product"})
    embedding_dim = model_config.get("out_channels", 64)
    decoder = create_decoder(decoder_config, embedding_dim)
    
    print(f"Model: {args.model}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Decoder: {decoder_config.get('type', 'dot_product')}")
    
    # Create trainer
    trainer = LinkPredictionTrainer(
        model=model,
        decoder=decoder,
        data=data,
        device=config.experiment.device,
        learning_rate=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        patience=config.train.patience,
        min_delta=config.train.min_delta,
        gradient_clip_norm=config.train.gradient_clip_norm,
        negative_sampling_ratio=config.data.negative_sampling_ratio,
        checkpoint_dir=config.logging.checkpoint_dir,
        log_dir=config.logging.log_dir,
    )
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        epochs=config.train.epochs,
        eval_every=10,
        save_best=True,
        verbose=args.verbose,
    )
    
    # Final evaluation
    print("Final evaluation...")
    val_metrics = trainer.evaluate("val")
    test_metrics = trainer.evaluate("test")
    
    print("\nFinal Results:")
    print(f"Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(f"Validation AP: {val_metrics['average_precision']:.4f}")
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"Test AP: {test_metrics['average_precision']:.4f}")
    
    # Save results
    results = {
        "config": config,
        "history": history,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    
    results_path = os.path.join(config.logging.log_dir, "results.yaml")
    OmegaConf.save(results, results_path)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
