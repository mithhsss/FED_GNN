"""
Main federated learning orchestration for FedGATSage.
Handles client-server coordination, community overlay graph construction (Algorithm 2),
and performance-weighted adaptive federated model aggregation (Eq. 4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import time
import logging
import os

from gnn_models import TemporalGATDetector, ContentGATDetector, BehavioralGATDetector, GlobalGraphSAGE
from feature_engineering import FeatureEngineer, CentralityFeatureExtractor
from community_detection import CommunityAwareProcessor
from dataset_utils import load_and_standardize_dataset

logger = logging.getLogger(__name__)

class OverlayGraphBuilder:
    """
    Constructs the community overlay graph across client flow/community embeddings.
    Implements Algorithm 2 from the FedGATSage paper using dynamic cosine similarity thresholds.
    """
    
    def __init__(self, initial_threshold: float = 0.7, min_threshold: float = 0.3, 
                 step: float = 0.05, min_connections: int = 3, max_connections: int = 10):
        self.initial_threshold = initial_threshold
        self.min_threshold = min_threshold
        self.step = step
        self.min_connections = min_connections
        self.max_connections = max_connections
        
    def build_overlay_graph(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Build edge_index for overlay graph from node/flow embeddings using dynamic cosine similarity.
        
        Args:
            embeddings: Tensor of shape (num_nodes, embedding_dim)
            
        Returns:
            edge_index: LongTensor of shape (2, num_edges)
        """
        num_nodes = embeddings.shape[0]
        device = embeddings.device
        
        if num_nodes <= 1:
            return torch.empty((2, 0), dtype=torch.long, device=device)
            
        target_connections = min(self.min_connections, num_nodes - 1)
        max_conn = min(self.max_connections, num_nodes - 1)
        
        # 1. Normalize embeddings to unit vectors for cosine similarity
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 2. Pairwise Cosine Similarity matrix: S_ij in [-1, 1]
        sim_matrix = torch.mm(norm_embeddings, norm_embeddings.t())
        
        # Zero out self-similarity on the diagonal
        sim_matrix.fill_diagonal_(-1.0)
        
        edges = set()
        
        # 3. Dynamic thresholding per node (Algorithm 2)
        for i in range(num_nodes):
            threshold = self.initial_threshold
            row_sims = sim_matrix[i]
            
            # Count connections at current threshold
            matching = (row_sims >= threshold).nonzero(as_tuple=True)[0].tolist()
            
            # Progressively lower threshold until at least target_connections are made or min_threshold reached
            while len(matching) < target_connections and threshold > self.min_threshold:
                threshold = round(threshold - self.step, 4)
                matching = (row_sims >= threshold).nonzero(as_tuple=True)[0].tolist()
                
            # If still fewer than target_connections, connect to top-k most similar
            if len(matching) < target_connections:
                topk_indices = torch.topk(row_sims, k=target_connections).indices.tolist()
                matching = list(set(matching + topk_indices))
            elif len(matching) > max_conn:
                # Keep top max_conn strongest connections
                topk_indices = torch.topk(row_sims, k=max_conn).indices.tolist()
                matching = topk_indices
                
            for j in matching:
                if i != j:
                    u, v = min(i, j), max(i, j)
                    edges.add((u, v))
                    edges.add((v, u))
                    
        if not edges:
            # Fallback: connect each node to its nearest neighbor
            for i in range(num_nodes):
                nearest = torch.argmax(sim_matrix[i]).item()
                edges.add((i, nearest))
                edges.add((nearest, i))
                
        edge_list = list(edges)
        src_nodes = [e[0] for e in edge_list]
        dst_nodes = [e[1] for e in edge_list]
        
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long, device=device)
        logger.info(f"Overlay graph constructed: {num_nodes} nodes, {edge_index.shape[1]} edges (avg degree: {edge_index.shape[1]/num_nodes:.2f})")
        return edge_index

class FlowEmbeddingGenerator:
    """Generates flow embeddings as community abstractions (Algorithm 1, Step 4)"""
    
    def __init__(self, detector_type: str = 'temporal'):
        self.detector_type = detector_type
        
    def generate_embeddings(self, model, data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate flow embeddings from GAT node embeddings.
        Implements the community abstraction mechanism from Algorithm 1.
        """
        model.eval()
        with torch.no_grad():
            device = next(model.parameters()).device
            x = data['features'].to(device)
            edge_index = data['edge_index'].to(device)
            edge_labels = data['edge_labels'].to(device)
            
            if edge_index.shape[1] == 0:
                return torch.empty(0, device=device), torch.empty(0, device=device)
                
            # Generate node embeddings using GAT
            try:
                node_embeddings, _ = model(x, edge_index)
            except Exception as e:
                logger.error(f"Error in GAT forward pass: {e}")
                return torch.empty(0, device=device), torch.empty(0, device=device)
            
            # Sample flows for efficiency and privacy (up to 250 per class)
            flow_embeddings = []
            flow_labels = []
            
            unique_labels = torch.unique(edge_labels)
            max_per_class = min(250, max(10, len(edge_labels) // max(1, len(unique_labels))))
            
            for label in unique_labels:
                mask = edge_labels == label
                if mask.sum() > 0:
                    label_indices = mask.nonzero(as_tuple=True)[0]
                    if len(label_indices) > max_per_class:
                        perm = torch.randperm(len(label_indices))[:max_per_class]
                        selected_indices = label_indices[perm]
                    else:
                        selected_indices = label_indices
                    
                    for idx in selected_indices:
                        src_idx = edge_index[0, idx]
                        dst_idx = edge_index[1, idx]
                        
                        src_emb = node_embeddings[src_idx]
                        dst_emb = node_embeddings[dst_idx]
                        
                        # Flow embedding: [src || dst || (src * dst) || |src - dst|]
                        flow_emb = self._create_flow_embedding(src_emb, dst_emb, data, idx)
                        flow_embeddings.append(flow_emb.unsqueeze(0))
                        flow_labels.append(label)
            
            if flow_embeddings:
                flow_embeddings = torch.cat(flow_embeddings, dim=0)
                flow_labels = torch.stack(flow_labels)
                return flow_embeddings, flow_labels
            else:
                return torch.empty(0, device=device), torch.empty(0, device=device)
    
    def _create_flow_embedding(self, src_emb: torch.Tensor, dst_emb: torch.Tensor, 
                               data: Dict[str, Any], idx: int) -> torch.Tensor:
        """Create flow embedding representing community relationship"""
        embedding_parts = [
            src_emb,
            dst_emb,
            src_emb * dst_emb,              # Element-wise product
            torch.abs(src_emb - dst_emb)     # Absolute difference
        ]
        
        if 'traffic_features' in data and data['traffic_features'] is not None:
            traffic_feat = data['traffic_features'][idx].to(src_emb.device)
            embedding_parts.append(traffic_feat)
            
        return torch.cat(embedding_parts)

class DataLoader:
    """Load and process data for FedGATSage clients"""
    
    def __init__(self, data_dir: str, detector_type: str = 'temporal'):
        self.data_dir = data_dir
        self.detector_type = detector_type
        self.feature_engineer = FeatureEngineer(detector_type)
        self.centrality_extractor = CentralityFeatureExtractor()
        self.community_processor = CommunityAwareProcessor()
        self.label_mapper = None
    
    def load_client_data(self, client_id: int) -> Optional[Dict[str, Any]]:
        """Load and process client data from CSV or Parquet"""
        csv_path = os.path.join(self.data_dir, f'client_{client_id}.csv')
        parquet_path = os.path.join(self.data_dir, f'client_{client_id}.parquet')
        
        if os.path.exists(parquet_path):
            file_path = parquet_path
        elif os.path.exists(csv_path):
            file_path = csv_path
        else:
            logger.error(f"Client file not found for client {client_id} in {self.data_dir}")
            return None
        
        try:
            df = load_and_standardize_dataset(file_path)
            logger.info(f"Loaded {len(df)} records for client {client_id}")
            
            if self.label_mapper is None:
                self._create_label_mapper(df)
            
            # Apply feature engineering and graph centrality extraction
            df = self.feature_engineer.extract_features(df)
            df = self.centrality_extractor.extract_centrality_features(df)
            df = self.community_processor.create_community_enhanced_features(df, {})
            
            return self._process_to_graph(df)
            
        except Exception as e:
            logger.error(f"Error loading client {client_id} data: {e}", exc_info=True)
            return None
    
    def _create_label_mapper(self, df: pd.DataFrame):
        """Create consistent label mapping across clients"""
        unique_attacks = sorted(df['Attack'].unique())
        self.label_mapper = {attack: idx for idx, attack in enumerate(unique_attacks)}
        logger.info(f"Created label mapper with {len(self.label_mapper)} classes: {self.label_mapper}")
    
    def _process_to_graph(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Convert DataFrame to graph format for GNN processing"""
        src_ips = df['Src IP'].astype(str)
        dst_ips = df['Dst IP'].astype(str)
        unique_ips = pd.concat([src_ips, dst_ips]).unique()
        ip_to_idx = {ip: idx for idx, ip in enumerate(unique_ips)}
        
        # Extract node features
        feature_cols = [col for col in df.columns if any(measure in col.lower() for measure in [
            'betweenness', 'pagerank', 'degree', 'closeness', 'eigenvector',
            'k_core', 'k_truss', 'modularity', 'flow_rate', 'avg_payload',
            'iat_variance', 'burst_ratio', 'syn_rst_ratio', 'unusual_payload',
            'port_spread', 'is_short_session'
        ])]
        
        if not feature_cols:
            feature_cols = ['flow_rate', 'avg_payload_fwd', 'protocol_encoded']
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
                    
        # Node features aggregated per unique IP
        features = []
        for ip in unique_ips:
            ip_rows = df[(src_ips == ip) | (dst_ips == ip)]
            avg_features = ip_rows[feature_cols].mean().fillna(0.0).values
            features.append(avg_features)
            
        features = torch.tensor(np.array(features), dtype=torch.float32)
        
        # Edges from flows
        edges = []
        edge_labels = []
        
        for _, row in df.iterrows():
            s_ip, d_ip = str(row['Src IP']), str(row['Dst IP'])
            if s_ip in ip_to_idx and d_ip in ip_to_idx:
                edges.append([ip_to_idx[s_ip], ip_to_idx[d_ip]])
                attack_label = row['Attack']
                label_idx = self.label_mapper.get(attack_label, 0) if self.label_mapper else 0
                edge_labels.append(label_idx)
                
        edge_index = torch.tensor(edges, dtype=torch.long).t() if edges else torch.empty((2, 0), dtype=torch.long)
        edge_labels = torch.tensor(edge_labels, dtype=torch.long) if edge_labels else torch.empty(0, dtype=torch.long)
        
        return {
            'features': features,
            'edge_index': edge_index,
            'edge_labels': edge_labels,
            'ip_to_idx': ip_to_idx,
            'df': df
        }

class FedGATSageSystem:
    """
    Main FedGATSage federated learning system.
    Orchestrates local GAT training, community overlay GraphSAGE aggregation,
    and adaptive weighted model redistribution.
    """
    
    def __init__(self, data_dir: str, num_clients: int = 5, 
                 detector_types: List[str] = ['temporal', 'content', 'behavioral'],
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.data_dir = data_dir
        self.num_clients = num_clients
        self.detector_types = detector_types
        self.device = device
        
        self.client_models = {}
        self.data_loaders = {}
        self.flow_generators = {}
        
        for detector_type in detector_types:
            detector_dir = os.path.join(data_dir, f'{detector_type}_detector')
            self.data_loaders[detector_type] = DataLoader(detector_dir, detector_type)
            self.flow_generators[detector_type] = FlowEmbeddingGenerator(detector_type)
            self.client_models[detector_type] = {}
            
        self.overlay_builder = OverlayGraphBuilder()
        self.global_model = None
        self.results = {'training_losses': [], 'round_times': [], 'client_metrics': []}
        
        logger.info(f"Initialized FedGATSage on device {device} with detector types: {detector_types}")
    
    def initialize_models(self, input_dims: Optional[Dict[str, int]] = None, default_input_dim: int = 64, hidden_dim: int = 256, num_classes: int = 8):
        """Initialize client GAT detectors with detector-specific input dimensions and server GlobalGraphSAGE model"""
        if input_dims is None:
            input_dims = {}
            for detector_type in self.detector_types:
                sample_data = self.data_loaders[detector_type].load_client_data(1)
                if sample_data and 'features' in sample_data:
                    input_dims[detector_type] = sample_data['features'].shape[1]
                else:
                    input_dims[detector_type] = default_input_dim
                    
        for detector_type in self.detector_types:
            d_input_dim = input_dims.get(detector_type, default_input_dim)
            self.client_models[detector_type] = {}
            for client_id in range(self.num_clients):
                if detector_type == 'temporal':
                    model = TemporalGATDetector(d_input_dim, hidden_dim, num_classes=num_classes)
                elif detector_type == 'content':
                    model = ContentGATDetector(d_input_dim, hidden_dim, num_classes=num_classes)
                elif detector_type == 'behavioral':
                    model = BehavioralGATDetector(d_input_dim, hidden_dim, num_classes=num_classes)
                else:
                    model = TemporalGATDetector(d_input_dim, hidden_dim, num_classes=num_classes)
                    
                self.client_models[detector_type][client_id] = model.to(self.device)
                
        # Flow embedding dimension = hidden_dim * 4 (src, dst, product, abs_diff)
        flow_embedding_dim = hidden_dim * 4
        
        self.global_model = GlobalGraphSAGE(
            input_dim=flow_embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        ).to(self.device)
        
        logger.info(f"Models initialized with detector input dims: {input_dims}, hidden_dim={hidden_dim}, GlobalGraphSAGE input_dim={flow_embedding_dim}")
        
    def train_federated(self, num_rounds: int = 15, local_epochs: int = 5) -> Dict[str, Any]:
        """Main federated training loop across federation rounds"""
        logger.info(f"Starting FedGATSage training for {num_rounds} rounds ({local_epochs} local epochs/round)...")
        
        for round_idx in range(num_rounds):
            round_start = time.time()
            logger.info(f"\n--- Federation Round {round_idx + 1}/{num_rounds} ---")
            
            all_client_updates = []
            round_client_metrics = {}
            
            # Train each specialized GAT detector sequentially (Paper Section 3)
            for detector_type in self.detector_types:
                client_updates = self._collect_client_updates(detector_type, local_epochs=local_epochs)
                all_client_updates.extend(client_updates)
                round_client_metrics[detector_type] = [u['metrics'] for u in client_updates]
                
            # Server-side aggregation with GraphSAGE on dynamic Community Overlay Graph (Algorithm 2)
            global_loss = self._aggregate_updates(all_client_updates)
            
            # Adaptive performance-weighted model redistribution (Paper Eq. 4)
            self._redistribute_models(all_client_updates)
            
            round_time = time.time() - round_start
            self.results['training_losses'].append(global_loss)
            self.results['round_times'].append(round_time)
            self.results['client_metrics'].append(round_client_metrics)
            
            logger.info(f"Round {round_idx + 1} finished in {round_time:.2f}s | Global GraphSAGE loss: {global_loss:.4f}")
            
        logger.info("Federated training completed successfully.")
        return self.results
        
    def _collect_client_updates(self, detector_type: str, local_epochs: int = 5) -> List[Dict[str, Any]]:
        """Collect updates from clients for a specific detector type"""
        client_updates = []
        
        for client_id in range(self.num_clients):
            client_data = self.data_loaders[detector_type].load_client_data(client_id + 1)
            if client_data is None or client_data['edge_index'].shape[1] == 0:
                continue
                
            client_model = self.client_models[detector_type][client_id]
            
            # Local GAT training with train/val split for validation accuracy Ak
            metrics = self._train_client_model(client_model, client_data, epochs=local_epochs)
            
            # Generate flow embeddings as community abstractions
            flow_gen = self.flow_generators[detector_type]
            flow_embeddings, flow_labels = flow_gen.generate_embeddings(client_model, client_data)
            
            if len(flow_embeddings) > 0:
                client_updates.append({
                    'client_id': client_id,
                    'detector_type': detector_type,
                    'flow_embeddings': flow_embeddings,
                    'flow_labels': flow_labels,
                    'model_state': {k: v.cpu() for k, v in client_model.state_dict().items()},
                    'metrics': metrics
                })
                
        return client_updates
        
    def _train_client_model(self, model: nn.Module, data: Dict[str, Any], epochs: int = 5) -> Dict[str, float]:
        """
        Train a single client GAT model locally with train/val split.
        Returns local train loss and validation accuracy (Ak for Eq. 4).
        """
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        x = data['features'].to(self.device)
        edge_index = data['edge_index'].to(self.device)
        edge_labels = data['edge_labels'].to(self.device)
        
        num_edges = edge_index.shape[1]
        if num_edges == 0:
            return {'loss': 0.0, 'val_accuracy': 0.5, 'val_loss': 0.0}
            
        # 80/20 train/val split of edge indices for local validation
        perm = torch.randperm(num_edges)
        val_size = max(1, int(0.2 * num_edges))
        val_indices = perm[:val_size]
        train_indices = perm[val_size:]
        
        train_edges = edge_index[:, train_indices]
        train_labels = edge_labels[train_indices]
        val_edges = edge_index[:, val_indices]
        val_labels = edge_labels[val_indices]
        
        last_train_loss = 0.0
        for _ in range(epochs):
            optimizer.zero_grad()
            _, predictions = model(x, train_edges)
            loss = criterion(predictions, train_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            last_train_loss = loss.item()
            
        # Compute local validation accuracy Ak
        model.eval()
        with torch.no_grad():
            _, val_predictions = model(x, val_edges)
            val_loss = criterion(val_predictions, val_labels).item()
            val_preds = val_predictions.argmax(dim=1)
            correct = (val_preds == val_labels).sum().item()
            val_acc = correct / len(val_labels) if len(val_labels) > 0 else 0.5
            
        model.train()
        return {
            'loss': float(last_train_loss),
            'val_loss': float(val_loss),
            'val_accuracy': float(val_acc)
        }
        
    def _aggregate_updates(self, client_updates: List[Dict[str, Any]]) -> float:
        """
        Server-side aggregation on dynamic Community Overlay Graph (Algorithm 2).
        Processes community flow embeddings across clients using GlobalGraphSAGE.
        """
        if not client_updates:
            return 0.0
            
        all_embeddings = [u['flow_embeddings'].to(self.device) for u in client_updates if len(u['flow_embeddings']) > 0]
        all_labels = [u['flow_labels'].to(self.device) for u in client_updates if len(u['flow_labels']) > 0]
        
        if not all_embeddings:
            return 0.0
            
        global_x = torch.cat(all_embeddings, dim=0)
        global_y = torch.cat(all_labels, dim=0)
        
        # Build community overlay graph using Algorithm 2
        overlay_edge_index = self.overlay_builder.build_overlay_graph(global_x)
        
        # Train Global GraphSAGE model on overlay graph
        self.global_model.train()
        optimizer = torch.optim.Adam(self.global_model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        optimizer.zero_grad()
        _, predictions = self.global_model(global_x, overlay_edge_index)
        loss = criterion(predictions, global_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        return float(loss.item())
        
    def _redistribute_models(self, client_updates: List[Dict[str, Any]]):
        """
        Redistribute global parameters back to clients using Adaptive Weighted FedAvg (Eq. 4).
        Weight: wk = alpha + (1 - alpha) * (Ak - Amin) / (Amax - Amin), alpha = 0.2
        """
        for detector_type in self.detector_types:
            updates = [u for u in client_updates if u['detector_type'] == detector_type]
            if not updates:
                continue
                
            accuracies = [u['metrics']['val_accuracy'] for u in updates]
            a_min = min(accuracies)
            a_max = max(accuracies)
            alpha = 0.2
            
            # Compute adaptive weights
            raw_weights = []
            for acc in accuracies:
                if a_max > a_min:
                    w = alpha + (1.0 - alpha) * ((acc - a_min) / (a_max - a_min))
                else:
                    w = 1.0
                raw_weights.append(w)
                
            total_weight = sum(raw_weights)
            normalized_weights = [w / total_weight for w in raw_weights]
            
            logger.info(f"Adaptive weights for {detector_type} detector: {dict(zip([u['client_id'] for u in updates], [round(nw, 4) for nw in normalized_weights]))}")
            
            # Weighted FedAvg aggregation
            aggregated_state = {}
            first_state = updates[0]['model_state']
            
            for key in first_state.keys():
                param_type = first_state[key].dtype
                if first_state[key].is_floating_point():
                    weighted_sum = torch.zeros_like(first_state[key], dtype=torch.float32)
                    for i, update in enumerate(updates):
                        weighted_sum += normalized_weights[i] * update['model_state'][key].to(torch.float32)
                    aggregated_state[key] = weighted_sum.to(param_type).to(self.device)
                else:
                    aggregated_state[key] = first_state[key].to(self.device)
                    
            # Update all client models with aggregated state
            for client_id in self.client_models[detector_type]:
                self.client_models[detector_type][client_id].load_state_dict(aggregated_state)