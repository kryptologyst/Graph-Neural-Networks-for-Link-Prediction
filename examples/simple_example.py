"""Example script demonstrating link prediction with GNNs."""

import torch
from src.data.dataset import LinkPredictionDataset
from src.models.gcn import GCNEncoder
from src.models.decoder import DotProductDecoder
from src.train.trainer import LinkPredictionTrainer
from src.utils.device import set_seed


def main():
    """Run a simple link prediction example."""
    # Set random seed for reproducibility
    set_seed(42)
    
    print("Loading Cora dataset...")
    dataset = LinkPredictionDataset(dataset_name="cora", root="data/")
    data = dataset.get_data()
    
    print(f"Dataset loaded:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Features: {data.num_node_features}")
    print(f"  Train edges: {data.train_pos_edge_index.size(1)}")
    print(f"  Val edges: {data.val_pos_edge_index.size(1)}")
    print(f"  Test edges: {data.test_pos_edge_index.size(1)}")
    
    # Create model
    print("\nCreating GCN model...")
    encoder = GCNEncoder(
        in_channels=data.num_node_features,
        hidden_dim=64,
        out_channels=64,
        num_layers=2,
        dropout=0.5,
    )
    decoder = DotProductDecoder()
    
    print(f"Model parameters: {sum(p.numel() for p in encoder.parameters() if p.requires_grad)}")
    
    # Create trainer
    print("\nSetting up trainer...")
    trainer = LinkPredictionTrainer(
        model=encoder,
        decoder=decoder,
        data=data,
        device="auto",
        learning_rate=0.01,
        weight_decay=5e-4,
        patience=50,
    )
    
    # Train model
    print("\nTraining model...")
    history = trainer.train(
        epochs=100,
        eval_every=10,
        save_best=True,
        verbose=True,
    )
    
    # Final evaluation
    print("\nFinal evaluation...")
    val_metrics = trainer.evaluate("val")
    test_metrics = trainer.evaluate("test")
    
    print(f"\nResults:")
    print(f"  Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(f"  Validation AP: {val_metrics['average_precision']:.4f}")
    print(f"  Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"  Test AP: {test_metrics['average_precision']:.4f}")
    
    # Example prediction
    print("\nExample link prediction...")
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        embeddings = encoder(data.x, data.train_pos_edge_index)
        
        # Predict score for first test edge
        test_edge = data.test_pos_edge_index[:, 0]
        score = decoder(embeddings, test_edge.unsqueeze(1)).item()
        
        print(f"  Test edge ({test_edge[0].item()}, {test_edge[1].item()}) score: {score:.4f}")
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
