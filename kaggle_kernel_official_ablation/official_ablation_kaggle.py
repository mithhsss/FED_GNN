import os
import sys
import json
import time
import glob
import logging
from typing import Dict, List, Tuple, Any

# Ensure graph and deep learning packages are installed on Kaggle
try:
    import igraph as ig
    import leidenalg
    import torch_geometric
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "igraph", "leidenalg", "torch-geometric", "-q"], check=False)
    import igraph as ig
    import leidenalg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, classification_report, confusion_matrix)
from sklearn.model_selection import StratifiedGroupKFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FedGATSage-MultiSeed] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FedGATSage_MultiSeed")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Execution Device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

PAPER_8_CLASSES = ['Benign', 'Backdoor', 'DDoS', 'DoS', 'Injection', 'Password', 'Scanning', 'XSS']
CLASS_MAP = {c.lower(): i for i, c in enumerate(PAPER_8_CLASSES)}
INV_CLASS_MAP = {i: c for i, c in enumerate(PAPER_8_CLASSES)}

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def map_nf_toniot_columns(df: pd.DataFrame) -> pd.DataFrame:
    """1-to-1 exact mapping from NetFlow V9 columns to pipeline feature names."""
    df = df.copy()
    mapping = {
        'IPV4_SRC_ADDR': 'Src IP', 'IPV4_DST_ADDR': 'Dst IP',
        'L4_SRC_PORT': 'Src Port', 'L4_DST_PORT': 'Dst Port',
        'PROTOCOL': 'Protocol', 'FLOW_DURATION_MILLISECONDS': 'Flow Duration',
        'IN_BYTES': 'TotLen Fwd Pkts', 'OUT_BYTES': 'TotLen Bwd Pkts',
        'IN_PKTS': 'Tot Fwd Pkts', 'OUT_PKTS': 'Tot Bwd Pkts',
    }
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

    # Bitwise RFC 793 decoding of TCP_FLAGS
    if 'TCP_FLAGS' in df.columns:
        flags = df['TCP_FLAGS'].fillna(0).astype(int)
        df['FIN Flag Cnt'] = ((flags & 0x01) > 0).astype(float)
        df['SYN Flag Cnt'] = ((flags & 0x02) > 0).astype(float)
        df['RST Flag Cnt'] = ((flags & 0x04) > 0).astype(float)
        df['PSH Flag Cnt'] = ((flags & 0x08) > 0).astype(float)
        df['ACK Flag Cnt'] = ((flags & 0x10) > 0).astype(float)
        df['URG Flag Cnt'] = ((flags & 0x20) > 0).astype(float)
    else:
        for f in ['FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt']:
            df[f] = 0.0

    dur_sec = df['Flow Duration'].fillna(0) / 1000.0 + 1e-6
    df['Flow Pkts/s'] = (df['Tot Fwd Pkts'].fillna(0) + df['Tot Bwd Pkts'].fillna(0)) / dur_sec
    df['Flow Bytes/s'] = (df['TotLen Fwd Pkts'].fillna(0) + df['TotLen Bwd Pkts'].fillna(0)) / dur_sec
    return df

class FeatureEngineer:
    def __init__(self, detector_type: str):
        self.detector_type = detector_type

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.detector_type == 'temporal':
            return self._add_temporal_features(df)
        elif self.detector_type == 'content':
            return self._add_content_features(df)
        elif self.detector_type == 'behavioral':
            return self._add_behavioral_features(df)
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        dur_ms = df['Flow Duration'].fillna(0) + 1.0
        df['Pkt_Rate'] = (df['Tot Fwd Pkts'] + df['Tot Bwd Pkts']) / (dur_ms / 1000.0)
        df['Byte_Rate'] = (df['TotLen Fwd Pkts'] + df['TotLen Bwd Pkts']) / (dur_ms / 1000.0)
        df['Fwd_Pkt_Ratio'] = df['Tot Fwd Pkts'] / (df['Tot Fwd Pkts'] + df['Tot Bwd Pkts'] + 1e-5)
        df['Duration_Log'] = np.log1p(np.maximum(df['Flow Duration'].fillna(0), 0.0))
        return df

    def _add_content_features(self, df: pd.DataFrame) -> pd.DataFrame:
        tot_pkts = df['Tot Fwd Pkts'] + df['Tot Bwd Pkts'] + 1e-5
        tot_bytes = df['TotLen Fwd Pkts'] + df['TotLen Bwd Pkts'] + 1e-5
        df['Avg_Pkt_Size'] = tot_bytes / tot_pkts
        df['Byte_Asymmetry'] = (df['TotLen Fwd Pkts'] - df['TotLen Bwd Pkts']) / tot_bytes
        df['Pkt_Asymmetry'] = (df['Tot Fwd Pkts'] - df['Tot Bwd Pkts']) / tot_pkts
        df['SYN_to_Pkt'] = df['SYN Flag Cnt'] / tot_pkts
        df['ACK_to_Pkt'] = df['ACK Flag Cnt'] / tot_pkts
        return df

    def _add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Is_Privileged_Port'] = ((df['Src Port'] < 1024) | (df['Dst Port'] < 1024)).astype(float)
        df['Is_Ephemeral_Src'] = (df['Src Port'] >= 49152).astype(float)
        df['Port_Spread'] = np.abs(df['Src Port'] - df['Dst Port'])
        df['TCP_Flag_Sum'] = (df['FIN Flag Cnt'] + df['SYN Flag Cnt'] + df['RST Flag Cnt'] +
                              df['PSH Flag Cnt'] + df['ACK Flag Cnt'] + df['URG Flag Cnt'])
        return df

def stratified_split_clients(df: pd.DataFrame, num_clients: int = 5, seed: int = 42) -> List[pd.DataFrame]:
    rng = np.random.RandomState(seed)
    client_indices = [[] for _ in range(num_clients)]
    for _, group in df.groupby('label_idx'):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        splits = np.array_split(indices, num_clients)
        for i, split in enumerate(splits):
            client_indices[i].extend(split)
    return [df.loc[idx].reset_index(drop=True) for idx in client_indices]

def b7_edge_sampling(df: pd.DataFrame, feat_cols: List[str], ratio: float = 0.50,
                      max_interp_factor: float = 10.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    counts = df['label_idx'].value_counts()
    if len(counts) == 0:
        return df
    max_count = counts.max()
    new_dfs = [df]

    for c, count in counts.items():
        if count >= max_count or count < 2:
            continue
        cls_df = df[df['label_idx'] == c]
        needed = int(min(count * max_interp_factor, max(0, int(ratio * (max_count - count)))))
        if needed <= 0:
            continue

        idx1 = rng.choice(len(cls_df), size=needed, replace=True)
        idx2 = rng.choice(len(cls_df), size=needed, replace=True)
        alphas = rng.beta(0.5, 0.5, size=(needed, 1)).astype(np.float32)

        feat_vals = cls_df[feat_cols].values.astype(np.float32)
        interp_feats = alphas * feat_vals[idx1] + (1.0 - alphas) * feat_vals[idx2]

        proto_vals = cls_df['Protocol'].values[idx1]
        syn_vals = cls_df['SYN Flag Cnt'].values[idx1]
        interp_df = pd.DataFrame(interp_feats, columns=feat_cols)
        interp_df['Attack'] = cls_df['Attack'].iloc[0]
        interp_df['label_idx'] = c
        interp_df['Protocol'] = proto_vals
        interp_df['SYN Flag Cnt'] = syn_vals
        interp_df['Src IP'] = [cls_df['Src IP'].iloc[i] for i in idx1]
        interp_df['Dst IP'] = [cls_df['Dst IP'].iloc[i] for i in idx2]
        new_dfs.append(interp_df)

    res = pd.concat(new_dfs, ignore_index=True)
    return res

class SpecializedGAT(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_heads: int = 8,
                 num_classes: int = 8, detector_type: str = 'temporal', dropout: float = 0.2):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward_batch(self, x: torch.Tensor, edge_index: torch.Tensor, batch_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fast batch forward: computes edge embeddings only for sampled batch_idx during training."""
        h1 = F.elu(self.gat1(x, edge_index))
        h2 = self.gat2(h1, edge_index)
        src = edge_index[0, batch_idx]
        dst = edge_index[1, batch_idx]
        edge_embs = self.edge_mlp(torch.cat([h2[src], h2[dst]], dim=1))
        logits = self.classifier(edge_embs)
        return h2, edge_embs, logits

    def forward_full(self, x: torch.Tensor, edge_index: torch.Tensor, chunk_size: int = 50000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chunked full evaluation to prevent CUDA out-of-memory on large test graphs."""
        h1 = F.elu(self.gat1(x, edge_index))
        h2 = self.gat2(h1, edge_index)
        num_edges = edge_index.size(1)
        if num_edges == 0:
            return h2, torch.empty((0, 8), device=x.device)
        
        all_logits = []
        for start in range(0, num_edges, chunk_size):
            end = min(start + chunk_size, num_edges)
            src = edge_index[0, start:end]
            dst = edge_index[1, start:end]
            embs = self.edge_mlp(torch.cat([h2[src], h2[dst]], dim=1))
            all_logits.append(self.classifier(embs))
        logits = torch.cat(all_logits, dim=0)
        return h2, logits

class OverlayGraphBuilder:
    def __init__(self, min_size: int = 5, resolution: float = 1.0):
        self.min_size = min_size
        self.resolution = resolution

    def run_leiden(self, G: nx.Graph, edge_index: torch.Tensor, edge_labels: torch.Tensor, seed: int = 42) -> List[int]:
        if G.number_of_edges() == 0 or G.number_of_nodes() == 0:
            return [0] * G.number_of_nodes()
        ig_graph = ig.Graph(directed=False)
        ig_graph.add_vertices(G.number_of_nodes())
        edges = list(G.edges())
        if len(edges) > 0:
            ig_graph.add_edges(edges)
            weights = [G[u][v].get('weight', 1.0) for u, v in edges]
            ig_graph.es['weight'] = weights
        try:
            part = leidenalg.find_partition(
                ig_graph, leidenalg.CPMVertexPartition,
                weights='weight' if len(edges) > 0 else None,
                resolution_parameter=self.resolution, seed=seed
            )
            return list(part.membership)
        except Exception:
            return [0] * G.number_of_nodes()

    def build_overlay(self, client_graphs: List[Dict[str, Any]], client_node_embs: List[torch.Tensor],
                      community_memberships: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c_embs, c_sizes, c_mal = [], [], []
        for i in range(len(client_graphs)):
            embs = client_node_embs[i]
            mems = community_memberships[i]
            labels = client_graphs[i]['edge_labels']
            e_idx = client_graphs[i]['edge_index']

            num_comm = max(mems) + 1 if len(mems) > 0 else 1
            for cid in range(num_comm):
                nodes = [n for n, c in enumerate(mems) if c == cid]
                if len(nodes) < self.min_size:
                    continue
                node_idx = torch.tensor(nodes, dtype=torch.long, device=embs.device)
                c_emb = embs[node_idx].mean(dim=0)
                node_set = set(nodes)
                if e_idx.size(1) > 0:
                    src_m = torch.tensor([s in node_set for s in e_idx[0].tolist()], dtype=torch.bool)
                    dst_m = torch.tensor([d in node_set for d in e_idx[1].tolist()], dtype=torch.bool)
                    c_edges = src_m & dst_m
                    mal_ratio = (labels[c_edges] > 0).float().mean().item() if c_edges.any() else 0.0
                else:
                    mal_ratio = 0.0
                c_embs.append(c_emb)
                c_sizes.append(float(len(nodes)))
                c_mal.append(float(mal_ratio))

        if len(c_embs) == 0:
            dummy_x = torch.zeros((1, 258), device=client_node_embs[0].device)
            dummy_edge = torch.empty((2, 0), dtype=torch.long, device=client_node_embs[0].device)
            dummy_labels = torch.zeros(1, dtype=torch.long, device=client_node_embs[0].device)
            return dummy_x, dummy_edge, dummy_labels

        c_embs = torch.stack(c_embs)
        sizes = torch.tensor(c_sizes, device=c_embs.device).unsqueeze(1)
        sizes = (sizes - sizes.mean()) / (sizes.std() + 1e-5)
        mals = torch.tensor(c_mal, device=c_embs.device).unsqueeze(1)
        overlay_x = torch.cat([c_embs, sizes, mals], dim=1)

        sims = F.cosine_similarity(c_embs.unsqueeze(1), c_embs.unsqueeze(0), dim=2)
        topk = min(5, len(c_embs))
        vals, indices = torch.topk(sims, k=topk, dim=1)
        srcs, dsts = [], []
        for u in range(len(c_embs)):
            for v_idx in indices[u]:
                v = v_idx.item()
                if u != v:
                    srcs.append(u)
                    dsts.append(v)

        overlay_edges = torch.tensor([srcs, dsts], dtype=torch.long, device=c_embs.device) if len(srcs) > 0 else torch.empty((2, 0), dtype=torch.long, device=c_embs.device)
        overlay_labels = (torch.tensor(c_mal, device=c_embs.device) > 0.5).long()
        return overlay_x, overlay_edges, overlay_labels

class GlobalGraphSAGE(nn.Module):
    def __init__(self, in_dim: int = 258, hidden_dim: int = 256, num_classes: int = 8):
        super().__init__()
        self.sage1 = SAGEConv(in_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.clf = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if edge_index.size(1) == 0:
            h = F.relu(self.sage1(x, torch.empty((2, 0), dtype=torch.long, device=x.device)))
            return h, self.clf(h)
        h = F.relu(self.norm(self.sage1(x, edge_index)))
        h = F.relu(self.norm(self.sage2(h, edge_index)))
        return h, self.clf(h)

def compute_cb_weights(labels: np.ndarray, num_classes: int = 8, beta: float = 0.9999) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes)
    eff_num = 1.0 - np.power(beta, np.maximum(counts, 1))
    weights = (1.0 - beta) / np.array(eff_num, dtype=np.float32)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)

def focal_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor = None, gamma: float = 2.0) -> torch.Tensor:
    if weights is not None:
        ce_loss = F.cross_entropy(logits, targets, weight=weights, reduction='none')
    else:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
    if gamma > 0.0:
        p_t = torch.exp(-ce_loss)
        focal = ((1.0 - p_t) ** gamma) * ce_loss
        return focal.mean()
    return ce_loss.mean()

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        z = F.normalize(features, p=2, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        self_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        label_match = torch.eq(labels.view(-1, 1), labels.view(1, -1))
        pos_mask = label_match & ~self_mask
        card_p = pos_mask.sum(dim=1)

        valid_anchors = card_p > 0
        if not valid_anchors.any():
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        sim_denom = sim.masked_fill(self_mask, -1e9)
        log_denom = torch.logsumexp(sim_denom, dim=1, keepdim=True)

        log_prob = sim - log_denom
        pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1)
        loss_per_anchor = - pos_log_prob[valid_anchors] / card_p[valid_anchors].float()
        return loss_per_anchor.mean()

def sample_class_balanced_batch(edge_labels: torch.Tensor, samples_per_class: int = 64) -> torch.Tensor:
    """Guarantees M=64 edges per class across all 8 classes in every batch."""
    indices = []
    num_classes = 8
    for c in range(num_classes):
        c_mask = torch.where(edge_labels == c)[0]
        if len(c_mask) == 0:
            continue
        if len(c_mask) >= samples_per_class:
            perm = torch.randperm(len(c_mask), device=edge_labels.device)[:samples_per_class]
            indices.append(c_mask[perm])
        else:
            rep = torch.randint(0, len(c_mask), (samples_per_class,), device=edge_labels.device)
            indices.append(c_mask[rep])
    if len(indices) == 0:
        return torch.arange(min(len(edge_labels), samples_per_class * 8), device=edge_labels.device)
    return torch.cat(indices)

class TwoStageEnsemble:
    def __init__(self, n_estimators: int = 100, max_depth: int = 20, random_state: int = 42):
        self.rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                         class_weight='balanced', max_samples=0.25, random_state=random_state, n_jobs=-1)
        self.is_fitted = False

    def extract_meta_features(self, prob_list: List[np.ndarray], raw_feats: np.ndarray) -> np.ndarray:
        feats = []
        for p in prob_list:
            preds = np.argmax(p, axis=1, keepdims=True)
            conf = np.max(p, axis=1, keepdims=True)
            feats.extend([p, preds, conf])
        feats.append(np.mean(prob_list, axis=0))
        feats.append(raw_feats)
        return np.concatenate(feats, axis=1)

    def fit(self, prob_list: List[np.ndarray], raw_feats: np.ndarray, y_true: np.ndarray):
        meta_x = self.extract_meta_features(prob_list, raw_feats)
        self.rf.fit(meta_x, y_true)
        self.is_fitted = True

    def predict(self, prob_list: List[np.ndarray], raw_feats: np.ndarray) -> np.ndarray:
        if self.is_fitted:
            meta_x = self.extract_meta_features(prob_list, raw_feats)
            return self.rf.predict(meta_x)
        return np.argmax(np.mean(prob_list, axis=0), axis=1)

def build_client_graph(df: pd.DataFrame, feat_cols: List[str], label_mapper: Dict[str, int]) -> Dict[str, Any]:
    all_ips = list(pd.concat([df['Src IP'], df['Dst IP']]).unique())
    ip2idx = {ip: i for i, ip in enumerate(all_ips)}
    src_idx = [ip2idx[ip] for ip in df['Src IP']]
    dst_idx = [ip2idx[ip] for ip in df['Dst IP']]

    G = nx.Graph()
    G.add_nodes_from(range(len(all_ips)))
    for s, d in zip(src_idx, dst_idx):
        if G.has_edge(s, d):
            G[s][d]['weight'] += 1
        else:
            G.add_edge(s, d, weight=1)

    raw_feats = df[feat_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    raw_feats = np.log1p(np.maximum(raw_feats, 0.0))

    node_feats = np.zeros((len(all_ips), len(feat_cols)), dtype=np.float32)
    node_counts = np.ones(len(all_ips), dtype=np.float32)
    for i in range(len(df)):
        s, d = src_idx[i], dst_idx[i]
        node_feats[s] += raw_feats[i]
        node_feats[d] += raw_feats[i]
        node_counts[s] += 1
        node_counts[d] += 1
    node_feats /= node_counts[:, None]

    f_mean = np.mean(node_feats, axis=0, keepdims=True)
    f_std = np.std(node_feats, axis=0, keepdims=True) + 1e-6
    node_feats = (node_feats - f_mean) / f_std

    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
    edge_labels = torch.tensor([label_mapper.get(str(a).lower(), 0) for a in df['Attack']], dtype=torch.long)

    return {
        'x': torch.tensor(node_feats, dtype=torch.float32),
        'edge_index': edge_index,
        'edge_labels': edge_labels,
        'nx_graph': G,
        'num_nodes': len(all_ips),
        'num_edges': len(df),
        'flow_raw_feats': raw_feats
    }

def run_single_experiment(train_df: pd.DataFrame, test_df: pd.DataFrame,
                          config_name: str, split_name: str, seed: int,
                          use_cb_focal: bool, use_edge_interp: bool, mu_supcon: float,
                          num_clients: int = 5, num_rounds: int = 15, local_epochs: int = 5,
                          out_dir: str = "/kaggle/working") -> Dict[str, Any]:
    t0 = time.time()
    set_seed(seed)

    logger.info(f"\n{'='*80}")
    logger.info(f"STARTING RUN: Config='{config_name}' | Split='{split_name}' | Seed={seed}")
    logger.info(f"  CB-Focal: {use_cb_focal} | Interp: {use_edge_interp} | mu: {mu_supcon:.2f}")
    logger.info(f"{'='*80}")

    client_dfs = stratified_split_clients(train_df, num_clients, seed=seed)
    detector_types = ['temporal', 'content', 'behavioral']
    feature_engineers = {d: FeatureEngineer(d) for d in detector_types}

    train_dfs = {d: [feature_engineers[d].extract_features(cdf) for cdf in client_dfs] for d in detector_types}
    test_dfs = {d: feature_engineers[d].extract_features(test_df) for d in detector_types}
    full_train_dfs = {d: feature_engineers[d].extract_features(train_df) for d in detector_types}

    exclude = ['Src IP', 'Dst IP', 'Label', 'Attack', 'Timestamp', 'TCP_FLAGS', 'attack_lower', 'flow_signature']
    feat_cols = {d: [c for c in train_dfs[d][0].columns if c not in exclude and np.issubdtype(train_dfs[d][0][c].dtype, np.number)]
                 for d in detector_types}

    if use_edge_interp:
        sampled_train_dfs = {
            d: [b7_edge_sampling(train_dfs[d][i], feat_cols[d], ratio=0.50, seed=seed+i) for i in range(num_clients)]
            for d in detector_types
        }
    else:
        sampled_train_dfs = train_dfs

    client_graphs = {
        d: [build_client_graph(sampled_train_dfs[d][i], feat_cols[d], CLASS_MAP) for i in range(num_clients)]
        for d in detector_types
    }
    test_graphs = {d: build_client_graph(test_dfs[d], feat_cols[d], CLASS_MAP) for d in detector_types}
    full_train_graphs = {d: build_client_graph(full_train_dfs[d], feat_cols[d], CLASS_MAP) for d in detector_types}

    if use_cb_focal:
        cb_weights = {
            d: [compute_cb_weights(client_graphs[d][i]['edge_labels'].numpy(), 8, beta=0.9999) for i in range(num_clients)]
            for d in detector_types
        }
    else:
        cb_weights = {d: [None for _ in range(num_clients)] for d in detector_types}

    client_models = {
        d: [SpecializedGAT(input_dim=len(feat_cols[d]), hidden_dim=256, num_heads=8,
                           num_classes=8, detector_type=d, dropout=0.2).to(device)
            for _ in range(num_clients)]
        for d in detector_types
    }
    optimizers = {
        d: [torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4) for m in client_models[d]]
        for d in detector_types
    }

    overlay_builder = OverlayGraphBuilder()
    supcon_criterion = SupervisedContrastiveLoss(temperature=0.07)

    for r in range(num_rounds):
        for d in detector_types:
            client_states = []
            community_embs_all = []
            client_node_embs = []

            for i in range(num_clients):
                model = client_models[d][i]
                opt = optimizers[d][i]
                g = client_graphs[d][i]

                x = g['x'].to(device)
                edge_index = g['edge_index'].to(device)
                edge_labels = g['edge_labels'].to(device)
                weights = cb_weights[d][i]

                model.train()
                for epoch in range(local_epochs):
                    opt.zero_grad()
                    batch_idx = sample_class_balanced_batch(edge_labels, samples_per_class=64)
                    _, b_embs, b_logits = model.forward_batch(x, edge_index, batch_idx)
                    b_targets = edge_labels[batch_idx]

                    if use_cb_focal:
                        loss_cls = focal_loss(b_logits, b_targets, weights=weights, gamma=2.0)
                    else:
                        loss_cls = F.cross_entropy(b_logits, b_targets)

                    if mu_supcon > 0.0:
                        loss_con = supcon_criterion(b_embs, b_targets)
                        total_loss = loss_cls + mu_supcon * loss_con
                    else:
                        total_loss = loss_cls

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    opt.step()

                model.eval()
                with torch.no_grad():
                    node_embs, _ = model.forward_full(x, edge_index)
                    client_states.append(model.state_dict())
                    client_node_embs.append(node_embs)
                    mems = overlay_builder.run_leiden(g['nx_graph'], edge_index, edge_labels, seed=seed+r)
                    community_embs_all.append(mems)

            # FedAvg aggregation
            total_edges = sum(g['num_edges'] for g in client_graphs[d])
            client_weights = [g['num_edges'] / total_edges for g in client_graphs[d]]

            agg_state = {}
            for k in client_states[0].keys():
                agg_state[k] = sum(client_weights[i] * client_states[i][k].float() for i in range(num_clients))

            for i in range(num_clients):
                client_models[d][i].load_state_dict({k: v.to(device) for k, v in agg_state.items()})

    # Full Evaluation via Two-Stage Meta-Learner
    train_probs_list = []
    test_probs_list = []

    for d in detector_types:
        eval_model = client_models[d][0]
        eval_model.eval()
        with torch.no_grad():
            gx_tr = full_train_graphs[d]['x'].to(device)
            gei_tr = full_train_graphs[d]['edge_index'].to(device)
            _, tr_logits = eval_model.forward_full(gx_tr, gei_tr)
            train_probs_list.append(F.softmax(tr_logits, dim=1).cpu().numpy())

            gx_te = test_graphs[d]['x'].to(device)
            gei_te = test_graphs[d]['edge_index'].to(device)
            _, te_logits = eval_model.forward_full(gx_te, gei_te)
            test_probs_list.append(F.softmax(te_logits, dim=1).cpu().numpy())

    y_train = full_train_graphs['temporal']['edge_labels'].numpy()
    y_test = test_graphs['temporal']['edge_labels'].numpy()
    tr_raw_feats = full_train_graphs['temporal']['flow_raw_feats']
    te_raw_feats = test_graphs['temporal']['flow_raw_feats']

    ensemble = TwoStageEnsemble(n_estimators=100, max_depth=20, random_state=seed)
    ensemble.fit(train_probs_list, tr_raw_feats, y_train)
    y_pred = ensemble.predict(test_probs_list, te_raw_feats)

    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average='macro', zero_division=0))

    labels_8 = list(range(len(PAPER_8_CLASSES)))
    report = classification_report(y_test, y_pred, labels=labels_8, target_names=PAPER_8_CLASSES, output_dict=True, zero_division=0)
    per_class_table = {}
    for cls in PAPER_8_CLASSES:
        rec = float(report[cls]['recall'])
        f1_k = float(report[cls]['f1-score'])
        supp = int(report[cls]['support'])
        per_class_table[cls] = {
            'recall': rec,
            'f1': f1_k,
            'support': supp
        }

    elapsed_min = (time.time() - t0) / 60.0
    logger.info(f"DONE ({elapsed_min:.2f}m) -> Acc: {acc*100:.2f}%, BalAcc: {bal_acc*100:.2f}%, MacroF1: {macro_f1*100:.2f}%")
    logger.info(f"  Backdoor Recall: {per_class_table['Backdoor']['recall']*100:.2f}% | Scanning: {per_class_table['Scanning']['recall']*100:.2f}% | Password: {per_class_table['Password']['recall']*100:.2f}%")

    run_record = {
        'config': config_name,
        'split': split_name,
        'seed': seed,
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'macro_f1': macro_f1,
        'per_class': per_class_table,
        'elapsed_minutes': elapsed_min
    }

    # Generate slug for filename
    cfg_slug = config_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace('+', '_')
    split_slug = split_name.lower().replace('-', '_')
    fname = f"run_{cfg_slug}_{split_slug}_seed{seed}.json"
    fpath = os.path.join(out_dir, fname)

    with open(fpath, 'w') as f:
        json.dump(run_record, f, indent=2)
    logger.info(f"Saved run JSON: {fpath}")

    return run_record

def main():
    logger.info("=" * 90)
    logger.info("FEDGATSAGE OFFICIAL NF-ToN-IoT FULL MULTI-SEED RUNNER (SEEDS 42, 43, 44)")
    logger.info("=" * 90)

    # 1. Dataset Loading & Genuine IP Verification
    search_dirs = [
        '/kaggle/input/official-nf-ton-iot/NF-ToN-IoT.csv',
        'kaggle_dataset_official_nf/NF-ToN-IoT.csv',
        'data/nf_ton_unzipped/7ca78ae35fa4961a_MOHANAD_A4706/data/NF-ToN-IoT.csv'
    ]
    data_file = None
    for p in search_dirs:
        if os.path.exists(p):
            data_file = p
            break
    if not data_file:
        raise FileNotFoundError(f"Official NF-ToN-IoT CSV not found in: {search_dirs}")

    logger.info(f"Loading official dataset from: {data_file}")
    raw_df = pd.read_csv(data_file)
    row_count = len(raw_df)
    logger.info(f"Official Row Count: {row_count:,} (Expected: 1,379,274)")
    assert row_count == 1379274, f"FATAL: Mismatch! Expected 1,379,274, got {row_count}"

    logger.info("\nFirst 5 Sample Rows of Genuine Network IPs:")
    logger.info("\n" + str(raw_df[['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT', 'Attack']].head(5)))

    # 2. Preprocessing & 8-Class Filtering
    df = map_nf_toniot_columns(raw_df)
    df['attack_lower'] = df['Attack'].astype(str).str.lower()
    df = df[df['attack_lower'].isin(CLASS_MAP)].copy().reset_index(drop=True)
    df['label_idx'] = df['attack_lower'].map(CLASS_MAP)
    logger.info(f"Cleaned 8-Class Dataset: {len(df):,} flows")

    # Output directory
    out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "kaggle_results"
    os.makedirs(out_dir, exist_ok=True)

    # 3. Setup Split A: Signature-Grouped Split (Audited, Zero Leakage)
    hash_cols = ['Src Port', 'Dst Port', 'Protocol', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'Flow Duration']
    df['flow_signature'] = df[hash_cols].astype(str).agg('|'.join, axis=1)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    sig_train_idx, sig_test_idx = next(sgkf.split(df, df['label_idx'], groups=df['flow_signature']))
    sig_train_df = df.iloc[sig_train_idx].copy().reset_index(drop=True)
    sig_test_df = df.iloc[sig_test_idx].copy().reset_index(drop=True)

    sig_overlap = len(set(sig_train_df['flow_signature']).intersection(set(sig_test_df['flow_signature'])))
    logger.info(f"Signature-Grouped Split -> Train: {len(sig_train_df):,} | Test: {len(sig_test_df):,} | Overlap Signatures: {sig_overlap} (0.00%)")
    assert sig_overlap == 0, "FATAL: Data leakage detected across signature groups!"

    # 4. Setup Split B: IP-Disjoint Split (0 Shared Source IPs)
    test_candidate_ips = {'192.168.1.31', '192.168.1.35', '192.168.1.37'}
    benign_ips = df[df['attack_lower'] == 'benign']['Src IP'].unique()
    for b_ip in benign_ips[:35]:
        if b_ip not in {'192.168.1.30', '192.168.1.33', '192.168.1.36', '192.168.1.38', '192.168.1.193'}:
            test_candidate_ips.add(b_ip)

    te_ip_mask = df['Src IP'].isin(test_candidate_ips)
    ip_train_df = df[~te_ip_mask].copy().reset_index(drop=True)
    ip_test_df = df[te_ip_mask].copy().reset_index(drop=True)

    ip_overlap = len(set(ip_train_df['Src IP']).intersection(set(ip_test_df['Src IP'])))
    logger.info(f"IP-Disjoint Split -> Train: {len(ip_train_df):,} | Test: {len(ip_test_df):,} | Shared Source IPs: {ip_overlap}")
    assert ip_overlap == 0, "FATAL: Shared source IPs detected in IP-disjoint split!"

    # Print IP-disjoint test supports and reliability audit
    logger.info("\n" + "=" * 60)
    logger.info("IP-DISJOINT PER-CLASS TEST SUPPORTS & RELIABILITY AUDIT")
    logger.info("=" * 60)
    logger.info(f"{'Class':<14} {'Test Support':>14} {'Reliability Status':>24}")
    logger.info("-" * 60)
    for c in PAPER_8_CLASSES:
        cnt = int((ip_test_df['attack_lower'] == c.lower()).sum())
        rel = "[!] NOT RELIABLE (<500 flows)" if cnt < 500 else "RELIABLE (>=500 flows)"
        logger.info(f"{c:<14} {cnt:>14,} {rel:>24}")
    logger.info("=" * 60 + "\n")

    # 5. Define All 21 Runs
    seeds = [42, 43, 44]
    runs_to_execute = []

    # Signature-grouped configs (5 configs x 3 seeds = 15 runs)
    sig_configs = [
        {"name": "GNN Anchor (Plain CE)", "use_cb_focal": False, "use_edge_interp": False, "mu_supcon": 0.0},
        {"name": "mu=0.0 (CB-Focal + Interp)", "use_cb_focal": True, "use_edge_interp": True, "mu_supcon": 0.0},
        {"name": "mu=0.1 (SupCon + CB-Focal)", "use_cb_focal": True, "use_edge_interp": True, "mu_supcon": 0.1},
        {"name": "mu=0.3 (SupCon + CB-Focal)", "use_cb_focal": True, "use_edge_interp": True, "mu_supcon": 0.3},
        {"name": "mu=0.5 (SupCon + CB-Focal)", "use_cb_focal": True, "use_edge_interp": True, "mu_supcon": 0.5},
    ]
    for cfg in sig_configs:
        for s in seeds:
            runs_to_execute.append({
                'split_name': 'signature-grouped',
                'train_df': sig_train_df,
                'test_df': sig_test_df,
                'config': cfg,
                'seed': s
            })

    # IP-disjoint configs (2 configs x 3 seeds = 6 runs)
    ip_configs = [
        {"name": "GNN Anchor (Plain CE)", "use_cb_focal": False, "use_edge_interp": False, "mu_supcon": 0.0},
        {"name": "mu=0.1 (SupCon + CB-Focal)", "use_cb_focal": True, "use_edge_interp": True, "mu_supcon": 0.1},
    ]
    for cfg in ip_configs:
        for s in seeds:
            runs_to_execute.append({
                'split_name': 'ip-disjoint',
                'train_df': ip_train_df,
                'test_df': ip_test_df,
                'config': cfg,
                'seed': s
            })

    logger.info(f"Total Scheduled Runs: {len(runs_to_execute)} (15 signature-grouped + 6 IP-disjoint)")

    # 6. Execute All Runs Sequentially
    executed_records = []
    for idx, item in enumerate(runs_to_execute, 1):
        cfg = item['config']
        s = item['seed']
        split_n = item['split_name']
        logger.info(f"\n--- Progress: Run {idx}/{len(runs_to_execute)} ---")
        rec = run_single_experiment(
            train_df=item['train_df'],
            test_df=item['test_df'],
            config_name=cfg['name'],
            split_name=split_n,
            seed=s,
            use_cb_focal=cfg['use_cb_focal'],
            use_edge_interp=cfg['use_edge_interp'],
            mu_supcon=cfg['mu_supcon'],
            num_clients=5, num_rounds=15, local_epochs=5,
            out_dir=out_dir
        )
        executed_records.append(rec)

    # 7. Multi-Seed Aggregation & Reporting from JSON Files Only
    logger.info("\n" + "=" * 95)
    logger.info("MULTI-SEED AGGREGATION & REPORTING FROM JSON FILES ONLY")
    logger.info("=" * 95)

    json_files = sorted(glob.glob(os.path.join(out_dir, "run_*_seed*.json")))
    logger.info(f"Found {len(json_files)} individual run JSON files:")
    for jf in json_files:
        logger.info(f"  [FILE] {os.path.basename(jf)}")

    loaded_records = []
    for jf in json_files:
        with open(jf, 'r') as f:
            loaded_records.append(json.load(f))

    # Group by (config, split)
    from collections import defaultdict
    grouped = defaultdict(list)
    for rec in loaded_records:
        k = (rec['config'], rec['split'])
        grouped[k].append(rec)

    table_rows = []
    logger.info("\n" + "=" * 135)
    logger.info(f"{'Config':<28} {'Split':<18} {'Balanced Acc (%)':<18} {'Macro F1 (%)':<18} {'Backdoor Rec (%)':<18} {'Scanning Rec (%)':<18} {'Password Rec (%)':<18}")
    logger.info("=" * 135)

    for (cfg_name, split_name), recs in grouped.items():
        if len(recs) != 3:
            logger.warning(f"Warning: {cfg_name} on {split_name} has {len(recs)} runs (expected 3)")

        b_accs = [r['balanced_accuracy'] * 100 for r in recs]
        mf1s = [r['macro_f1'] * 100 for r in recs]
        bd_recs = [r['per_class']['Backdoor']['recall'] * 100 for r in recs]
        sc_recs = [r['per_class']['Scanning']['recall'] * 100 for r in recs]
        pw_recs = [r['per_class']['Password']['recall'] * 100 for r in recs]

        b_acc_str = f"{np.mean(b_accs):.2f} ± {np.std(b_accs):.2f}"
        mf1_str = f"{np.mean(mf1s):.2f} ± {np.std(mf1s):.2f}"
        bd_str = f"{np.mean(bd_recs):.2f} ± {np.std(bd_recs):.2f}"
        sc_str = f"{np.mean(sc_recs):.2f} ± {np.std(sc_recs):.2f}"
        pw_str = f"{np.mean(pw_recs):.2f} ± {np.std(pw_recs):.2f}"

        logger.info(f"{cfg_name:<28} {split_name:<18} {b_acc_str:<18} {mf1_str:<18} {bd_str:<18} {sc_str:<18} {pw_str:<18}")

        table_rows.append({
            'config': cfg_name,
            'split': split_name,
            'num_seeds': len(recs),
            'seeds': [r['seed'] for r in recs],
            'balanced_accuracy_mean': float(np.mean(b_accs)),
            'balanced_accuracy_std': float(np.std(b_accs)),
            'macro_f1_mean': float(np.mean(mf1s)),
            'macro_f1_std': float(np.std(mf1s)),
            'backdoor_recall_mean': float(np.mean(bd_recs)),
            'backdoor_recall_std': float(np.std(bd_recs)),
            'scanning_recall_mean': float(np.mean(sc_recs)),
            'scanning_recall_std': float(np.std(sc_recs)),
            'password_recall_mean': float(np.mean(pw_recs)),
            'password_recall_std': float(np.std(pw_recs)),
        })

    logger.info("=" * 135)

    summary_file = os.path.join(out_dir, "multi_seed_ablation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'files_used': [os.path.basename(jf) for jf in json_files],
            'summary_table': table_rows
        }, f, indent=2)
    logger.info(f"Summary written to: {summary_file}")
    logger.info("ALL 21 RUNS AND STATISTICAL SUMMARY COMPLETED SUCCESSFULLY!")

if __name__ == '__main__':
    main()
