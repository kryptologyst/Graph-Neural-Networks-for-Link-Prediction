"""Streamlit demo for link prediction visualization."""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from pyvis.network import Network

from src.data.dataset import LinkPredictionDataset
from src.models.gcn import GCNEncoder
from src.models.graphsage import GraphSAGEEncoder
from src.models.gat import GATEncoder
from src.models.decoder import DotProductDecoder, CosineDecoder, MLPDecoder, BilinearDecoder
from src.utils.device import get_device, set_seed


@st.cache_data
def load_dataset(dataset_name: str) -> Tuple[LinkPredictionDataset, torch.Tensor]:
    """Load dataset with caching.
    
    Args:
        dataset_name: Name of the dataset.
        
    Returns:
        Tuple of dataset and data.
    """
    dataset = LinkPredictionDataset(dataset_name=dataset_name, root="data/")
    data = dataset.get_data()
    return dataset, data


@st.cache_data
def load_model(model_path: str, model_type: str, num_features: int, embedding_dim: int) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Load trained model with caching.
    
    Args:
        model_path: Path to model checkpoint.
        model_type: Type of model.
        num_features: Number of input features.
        embedding_dim: Embedding dimension.
        
    Returns:
        Tuple of encoder and decoder.
    """
    device = get_device("cpu")  # Use CPU for demo
    
    # Create model
    if model_type == "gcn":
        encoder = GCNEncoder(num_features, 64, embedding_dim)
    elif model_type == "graphsage":
        encoder = GraphSAGEEncoder(num_features, 64, embedding_dim)
    elif model_type == "gat":
        encoder = GATEncoder(num_features, 64, embedding_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    decoder = DotProductDecoder()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    encoder.load_state_dict(checkpoint["model_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder


def create_network_graph(data: torch.Tensor, max_nodes: int = 1000) -> Network:
    """Create interactive network visualization.
    
    Args:
        data: Graph data.
        max_nodes: Maximum number of nodes to display.
        
    Returns:
        Network: PyVis network object.
    """
    # Sample nodes if graph is too large
    if data.num_nodes > max_nodes:
        node_indices = torch.randperm(data.num_nodes)[:max_nodes]
        mask = torch.isin(data.edge_index[0], node_indices) & torch.isin(data.edge_index[1], node_indices)
        edge_index = data.edge_index[:, mask]
    else:
        edge_index = data.edge_index
        node_indices = torch.arange(data.num_nodes)
    
    # Create network
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    
    # Add nodes
    for i in node_indices:
        net.add_node(
            int(i),
            label=f"Node {int(i)}",
            color="#97C2FC",
            size=10,
        )
    
    # Add edges
    for i in range(edge_index.size(1)):
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        if src in node_indices and dst in node_indices:
            net.add_edge(src, dst, color="#848484", width=1)
    
    # Configure physics
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100}
      }
    }
    """)
    
    return net


def visualize_embeddings(embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None) -> go.Figure:
    """Visualize node embeddings using t-SNE.
    
    Args:
        embeddings: Node embeddings.
        labels: Node labels (optional).
        
    Returns:
        go.Figure: Plotly figure.
    """
    from sklearn.manifold import TSNE
    
    # Convert to numpy
    embeddings_np = embeddings.detach().cpu().numpy()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, embeddings_np.shape[0] - 1))
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    # Create DataFrame
    df = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
    })
    
    if labels is not None:
        df["label"] = labels.detach().cpu().numpy()
        color_col = "label"
    else:
        color_col = None
    
    # Create plot
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=color_col,
        title="Node Embeddings (t-SNE)",
        labels={"x": "t-SNE 1", "y": "t-SNE 2"},
    )
    
    fig.update_layout(
        width=800,
        height=600,
        showlegend=True,
    )
    
    return fig


def predict_link_score(encoder: torch.nn.Module, decoder: torch.nn.Module, 
                      data: torch.Tensor, node1: int, node2: int) -> float:
    """Predict link score between two nodes.
    
    Args:
        encoder: Trained encoder model.
        decoder: Trained decoder model.
        data: Graph data.
        node1: First node index.
        node2: Second node index.
        
    Returns:
        float: Link prediction score.
    """
    with torch.no_grad():
        # Get embeddings
        embeddings = encoder(data.x, data.train_pos_edge_index)
        
        # Create edge index for the pair
        edge_index = torch.tensor([[node1], [node2]], dtype=torch.long)
        
        # Predict score
        score = decoder(embeddings, edge_index).item()
        
        return score


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Link Prediction Demo",
        page_icon="🔗",
        layout="wide",
    )
    
    st.title("🔗 Graph Neural Networks for Link Prediction")
    st.markdown("Interactive demo for link prediction using GNNs")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Dataset selection
    dataset_name = st.sidebar.selectbox(
        "Dataset",
        ["cora", "citeseer", "pubmed"],
        index=0,
    )
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["gcn", "graphsage", "gat"],
        index=0,
    )
    
    # Load dataset
    with st.spinner("Loading dataset..."):
        dataset, data = load_dataset(dataset_name)
    
    st.sidebar.success(f"Loaded {dataset_name} dataset")
    st.sidebar.info(f"Nodes: {data.num_nodes}, Edges: {data.edge_index.size(1)}")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Graph Visualization", "Link Prediction", "Embeddings", "Model Comparison"])
    
    with tab1:
        st.header("Graph Visualization")
        
        # Graph statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodes", data.num_nodes)
        with col2:
            st.metric("Edges", data.edge_index.size(1))
        with col3:
            st.metric("Features", data.num_node_features)
        with col4:
            st.metric("Classes", int(data.y.max().item()) + 1)
        
        # Interactive graph
        st.subheader("Interactive Graph")
        max_nodes = st.slider("Max nodes to display", 50, 1000, 500)
        
        net = create_network_graph(data, max_nodes)
        net_html = net.generate_html()
        st.components.v1.html(net_html, height=600)
    
    with tab2:
        st.header("Link Prediction")
        
        # Check if model exists
        model_path = f"checkpoints/best_model_{model_type}_{dataset_name}.pt"
        
        if os.path.exists(model_path):
            # Load model
            with st.spinner("Loading model..."):
                encoder, decoder = load_model(model_path, model_type, data.num_node_features, 64)
            
            st.success(f"Loaded {model_type.upper()} model")
            
            # Link prediction interface
            st.subheader("Predict Link Score")
            
            col1, col2 = st.columns(2)
            with col1:
                node1 = st.number_input("Node 1", min_value=0, max_value=data.num_nodes-1, value=0)
            with col2:
                node2 = st.number_input("Node 2", min_value=0, max_value=data.num_nodes-1, value=1)
            
            if st.button("Predict Link Score"):
                score = predict_link_score(encoder, decoder, data, node1, node2)
                
                st.metric("Link Score", f"{score:.4f}")
                
                # Interpretation
                if score > 0.5:
                    st.success("High probability of link existence")
                elif score > 0.0:
                    st.warning("Moderate probability of link existence")
                else:
                    st.error("Low probability of link existence")
            
            # Batch prediction
            st.subheader("Batch Link Prediction")
            
            # Sample random node pairs
            if st.button("Sample Random Pairs"):
                num_pairs = st.slider("Number of pairs", 5, 50, 10)
                
                # Sample random pairs
                nodes = torch.randperm(data.num_nodes)[:num_pairs*2]
                pairs = [(int(nodes[i]), int(nodes[i+1])) for i in range(0, len(nodes), 2)]
                
                scores = []
                for node1, node2 in pairs:
                    score = predict_link_score(encoder, decoder, data, node1, node2)
                    scores.append(score)
                
                # Create DataFrame
                df = pd.DataFrame({
                    "Node 1": [pair[0] for pair in pairs],
                    "Node 2": [pair[1] for pair in pairs],
                    "Score": scores,
                })
                
                st.dataframe(df)
                
                # Visualization
                fig = px.bar(df, x="Score", title="Link Prediction Scores")
                st.plotly_chart(fig)
        
        else:
            st.warning(f"Model not found at {model_path}")
            st.info("Please train a model first using the training script")
    
    with tab3:
        st.header("Node Embeddings")
        
        # Check if model exists
        model_path = f"checkpoints/best_model_{model_type}_{dataset_name}.pt"
        
        if os.path.exists(model_path):
            # Load model
            with st.spinner("Loading model..."):
                encoder, decoder = load_model(model_path, model_type, data.num_node_features, 64)
            
            # Get embeddings
            with torch.no_grad():
                embeddings = encoder(data.x, data.train_pos_edge_index)
            
            # Visualize embeddings
            st.subheader("Embedding Visualization")
            
            # t-SNE plot
            fig = visualize_embeddings(embeddings, data.y)
            st.plotly_chart(fig)
            
            # Embedding statistics
            st.subheader("Embedding Statistics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Embedding Dimension", embeddings.size(1))
                st.metric("Mean Norm", f"{torch.norm(embeddings, dim=1).mean():.4f}")
            with col2:
                st.metric("Std Norm", f"{torch.norm(embeddings, dim=1).std():.4f}")
                st.metric("Max Norm", f"{torch.norm(embeddings, dim=1).max():.4f}")
        
        else:
            st.warning(f"Model not found at {model_path}")
            st.info("Please train a model first using the training script")
    
    with tab4:
        st.header("Model Comparison")
        
        # Compare different models
        models_to_compare = []
        for model_type in ["gcn", "graphsage", "gat"]:
            model_path = f"checkpoints/best_model_{model_type}_{dataset_name}.pt"
            if os.path.exists(model_path):
                models_to_compare.append(model_type)
        
        if models_to_compare:
            st.subheader("Available Models")
            
            # Load results for comparison
            results_data = []
            for model_type in models_to_compare:
                results_path = f"logs/results_{model_type}_{dataset_name}.yaml"
                if os.path.exists(results_path):
                    # This would load the results - simplified for demo
                    results_data.append({
                        "Model": model_type.upper(),
                        "ROC-AUC": 0.85,  # Placeholder
                        "AP": 0.80,  # Placeholder
                    })
            
            if results_data:
                df = pd.DataFrame(results_data)
                st.dataframe(df)
                
                # Comparison chart
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("ROC-AUC", "Average Precision"),
                )
                
                fig.add_trace(
                    go.Bar(x=df["Model"], y=df["ROC-AUC"], name="ROC-AUC"),
                    row=1, col=1,
                )
                
                fig.add_trace(
                    go.Bar(x=df["Model"], y=df["AP"], name="AP"),
                    row=1, col=2,
                )
                
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig)
        
        else:
            st.warning("No trained models found for comparison")
            st.info("Please train models first using the training script")


if __name__ == "__main__":
    main()
