#!/usr/bin/env python3
"""
Graph Representation Learning Module
GCN implementation for learning graph embeddings
"""

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


def networkx_to_pyg(graph: nx.Graph, node_features: Optional[np.ndarray] = None) -> Data:
    """
    Convert NetworkX graph to PyTorch Geometric Data object
    
    Args:
        graph: NetworkX graph
        node_features: Optional node features (n_nodes, n_features)
    
    Returns:
        PyTorch Geometric Data object
    """
    # Get edge indices
    edge_index = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous()
    if edge_index.size(0) == 0:
        # Empty graph - create self-loops
        n_nodes = graph.number_of_nodes()
        if n_nodes == 0:
            edge_index = torch.zeros((2, 1), dtype=torch.long)
        else:
            edge_index = torch.arange(n_nodes, dtype=torch.long).repeat(2, 1)
    
    # Get node features
    if node_features is not None:
        x = torch.tensor(node_features, dtype=torch.float32)
    else:
        # Use degree as default feature
        n_nodes = graph.number_of_nodes()
        if n_nodes == 0:
            x = torch.zeros((1, 1), dtype=torch.float32)
        else:
            degrees = torch.tensor([graph.degree(n) for n in graph.nodes()], dtype=torch.float32).unsqueeze(1)
            x = degrees
    
    return Data(x=x, edge_index=edge_index)


class GCN(nn.Module):
    """
    Graph Convolutional Network for graph-level classification
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 128,
                 num_layers: int = 2, pooling: str = 'mean', dropout: float = 0.5):
        """
        Initialize GCN
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of GCN layers
            pooling: Pooling method ('mean', 'max', 'sum', or 'concat')
            dropout: Dropout rate
        """
        super(GCN, self).__init__()
        
        self.num_layers = num_layers
        self.pooling = pooling
        self.dropout = dropout
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
        
        # Pooling
        if pooling == 'concat':
            self.fc = nn.Linear(output_dim * 3, output_dim)  # mean, max, sum
        else:
            self.fc = nn.Linear(output_dim, output_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch):
        """
        Forward pass
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch vector
        
        Returns:
            Graph embeddings
        """
        # GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        # Graph-level pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch)
        elif self.pooling == 'concat':
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x_sum = global_add_pool(x, batch)
            x = torch.cat([x_mean, x_max, x_sum], dim=1)
            x = self.fc(x)
        else:
            x = global_mean_pool(x, batch)  # Default to mean
        
        return x


class GCNTrainer:
    """
    Trainer for GCN model
    """
    
    def __init__(self, model: GCN, device: Optional[torch.device] = None):
        """
        Initialize trainer
        
        Args:
            model: GCN model
            device: Device to use (CPU or CUDA)
        """
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Using device: {self.device}")
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              epochs: int = 100, lr: float = 0.001, weight_decay: float = 5e-4,
              early_stopping_patience: int = 10, save_path: Optional[str] = None) -> Dict:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs
            lr: Learning rate
            weight_decay: Weight decay
            early_stopping_patience: Early stopping patience
            save_path: Path to save best model
        
        Returns:
            Dictionary with training history
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for data in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                
                # Forward pass (we're just extracting embeddings, not classifying)
                # So we use a simple reconstruction loss or just extract embeddings
                out = self.model(data.x, data.edge_index, data.batch)
                
                # For now, use a simple loss (can be modified for specific tasks)
                # Here we'll use a dummy loss since we're just extracting embeddings
                loss = torch.mean(out ** 2)  # Simple regularization loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for data in val_loader:
                        data = data.to(self.device)
                        out = self.model(data.x, data.edge_index, data.batch)
                        loss = torch.mean(out ** 2)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        if save_path and Path(save_path).exists():
                            self.model.load_state_dict(torch.load(save_path))
                        break
            else:
                history['val_loss'].append(train_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {history['val_loss'][-1]:.4f}")
        
        return history
    
    def extract_embeddings(self, data_loader: DataLoader) -> np.ndarray:
        """
        Extract graph embeddings
        
        Args:
            data_loader: Data loader
        
        Returns:
            Graph embeddings array
        """
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for data in tqdm(data_loader, desc="Extracting embeddings"):
                data = data.to(self.device)
                emb = self.model(data.x, data.edge_index, data.batch)
                embeddings.append(emb.cpu().numpy())
        
        return np.vstack(embeddings)


def train_gcn_and_extract_embeddings(graphs: List[nx.Graph],
                                    labels: Optional[List] = None,
                                    node_features: Optional[List[np.ndarray]] = None,
                                    train_test_split: Optional[Tuple] = None,
                                    hidden_dim: int = 64,
                                    output_dim: int = 128,
                                    num_layers: int = 2,
                                    pooling: str = 'mean',
                                    epochs: int = 50,
                                    batch_size: int = 32,
                                    lr: float = 0.001,
                                    save_model_path: Optional[str] = None,
                                    save_embeddings_path: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
    """
    Train GCN and extract embeddings for all graphs
    
    Args:
        graphs: List of NetworkX graphs
        labels: Optional labels (for train/test split)
        node_features: Optional node features for each graph
        train_test_split: Optional (train_indices, test_indices) tuple
        hidden_dim: Hidden dimension
        output_dim: Output embedding dimension
        num_layers: Number of GCN layers
        pooling: Pooling method
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        save_model_path: Path to save trained model
        save_embeddings_path: Path to save embeddings
    
    Returns:
        Tuple of (embeddings, training_info)
    """
    print("="*60)
    print("GCN Training and Embedding Extraction")
    print("="*60)
    
    # Convert graphs to PyG format
    print("\nConverting graphs to PyTorch Geometric format...")
    pyg_data_list = []
    
    for i, graph in enumerate(tqdm(graphs, desc="Converting graphs")):
        node_feat = node_features[i] if node_features and i < len(node_features) else None
        data = networkx_to_pyg(graph, node_feat)
        pyg_data_list.append(data)
    
    # Determine input dimension
    input_dim = pyg_data_list[0].x.shape[1] if len(pyg_data_list) > 0 else 1
    
    # Create model
    model = GCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                num_layers=num_layers, pooling=pooling)
    
    # Create data loaders
    if train_test_split is not None:
        train_indices, test_indices = train_test_split
        train_data = [pyg_data_list[i] for i in train_indices]
        test_data = [pyg_data_list[i] for i in test_indices]
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        # Train model
        trainer = GCNTrainer(model)
        history = trainer.train(train_loader, val_loader, epochs=epochs, lr=lr,
                              save_path=save_model_path)
        
        # Extract embeddings for all graphs
        all_loader = DataLoader(pyg_data_list, batch_size=batch_size, shuffle=False)
        embeddings = trainer.extract_embeddings(all_loader)
    else:
        # No train/test split - just extract embeddings (untrained model)
        print("No train/test split provided - extracting embeddings with untrained model")
        all_loader = DataLoader(pyg_data_list, batch_size=batch_size, shuffle=False)
        trainer = GCNTrainer(model)
        embeddings = trainer.extract_embeddings(all_loader)
        history = {}
    
    # Save embeddings
    if save_embeddings_path:
        print(f"\nSaving embeddings to: {save_embeddings_path}")
        Path(save_embeddings_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"  Saved {len(embeddings)} embeddings of shape {embeddings.shape}")
    
    print("\n" + "="*60)
    print("GCN Training Complete!")
    print("="*60)
    
    return embeddings, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train GCN and extract embeddings')
    parser.add_argument('--graphs-file', type=str, required=True,
                       help='Pickle file with list of NetworkX graphs')
    parser.add_argument('--output', type=str, default='data/embeddings/gcn_embeddings.pkl',
                       help='Output embeddings file')
    parser.add_argument('--model-output', type=str, default='data/models/gcn_model.pth',
                       help='Output model file')
    parser.add_argument('--hidden-dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--output-dim', type=int, default=128,
                       help='Output embedding dimension')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    
    args = parser.parse_args()
    
    # Load graphs
    with open(args.graphs_file, 'rb') as f:
        graphs = pickle.load(f)
    
    embeddings, history = train_gcn_and_extract_embeddings(
        graphs,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        epochs=args.epochs,
        save_model_path=args.model_output,
        save_embeddings_path=args.output
    )

