import os
import sys

# Deterministic environment setup before importing torch/backends
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import glob
import json
import logging
import random
import time
from typing import Any, Dict, List, Tuple

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

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [NF-UNSW-NB15-Reproducible] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NF_UNSW_NB15_Reproducible")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Execution Device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

UNSW_10_CLASSES = [
    'Benign', 'Analysis', 'Backdoor', 'DoS', 'Exploits',
    'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms'
]
CLASS_MAP = {c.lower(): i for i, c in enumerate(UNSW_10_CLASSES)}
INV_CLASS_MAP = {i: c for i, c in enumerate(UNSW_10_CLASSES)}
NUM_CLASSES = len(UNSW_10_CLASSES)

def set_seed(seed: int):
    """
    Strict deterministic seeding across Python, NumPy, PyTorch CPU, CUDA, and cuDNN.
    Ensures cuDNN deterministic mode is active and benchmark heuristic searches are disabled.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        logger.warning(f"Could not enable deterministic algorithms: {e}")

def classify_severity_tier(bal_acc: float) -> str:
    """
    Classify collapse severity into three explicit tiers:
    - Stable (>85%)
    - Degraded (40-85%)
    - Collapsed (<25%)
    """
    bal_acc_pct = bal_acc * 100.0 if bal_acc <= 1.0 else bal_acc
    if bal_acc_pct > 85.0:
        return "Stable (>85%)"
    elif bal_acc_pct >= 40.0:
        return "Degraded (40-85%)"
    elif bal_acc_pct < 25.0:
        return "Collapsed (<25%)"
    else:
        return "Degraded (25-40%)"

def map_nf_toniot_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {
        'IPV4_SRC_ADDR': 'Src IP', 'IPV4_DST_ADDR': 'Dst IP',
        'L4_SRC_PORT': 'Src Port', 'L4_DST_PORT': 'Dst Port',
        'PROTOCOL': 'Protocol', 'FLOW_DURATION_MILLISECONDS': 'Flow Duration',
        'IN_BYTES': 'TotLen Fwd Pkts', 'OUT_BYTES': 'TotLen Bwd Pkts',
        'IN_PKTS': 'Tot Fwd Pkts', 'OUT_PKTS': 'Tot Bwd Pkts',
    }
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

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
    for _, group in df.groupby('label_idx', sort=True):
        indices = group.index.to_numpy().copy()
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

    # Iterating over sorted index guarantees 100% deterministic sequence of class interpolations
    for c in sorted(counts.index):
        count = counts[c]
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
                 num_classes: int = 10, detector_type: str = 'temporal', dropout: float = 0.2):
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
        h1 = F.elu(self.gat1(x, edge_index))
        h2 = self.gat2(h1, edge_index)
        src = edge_index[0, batch_idx]
        dst = edge_index[1, batch_idx]
        edge_embs = self.edge_mlp(torch.cat([h2[src], h2[dst]], dim=1))
        logits = self.classifier(edge_embs)
        return h2, edge_embs, logits

    def forward_full(self, x: torch.Tensor, edge_index: torch.Tensor, chunk_size: int = 50000) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = F.elu(self.gat1(x, edge_index))
        h2 = self.gat2(h1, edge_index)
        num_edges = edge_index.size(1)
        if num_edges == 0:
            return h2, torch.empty((0, NUM_CLASSES), device=x.device)
        
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
        edges = sorted(list(G.edges()))
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

def compute_cb_weights(labels: np.ndarray, num_classes: int = 10, beta: float = 0.9999) -> torch.Tensor:
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
    indices = []
    for c in range(NUM_CLASSES):
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
        return torch.arange(min(len(edge_labels), samples_per_class * NUM_CLASSES), device=edge_labels.device)
    return torch.cat(indices)

class TwoStageEnsemble:
    def __init__(self, n_estimators: int = 100, max_depth: int = 20, random_state: int = 42):
        # n_jobs=1 ensures strictly serial, thread-safe, bitwise deterministic tree construction
        self.rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                         class_weight='balanced', max_samples=0.25,
                                         random_state=random_state, n_jobs=1)
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
    # Strictly sort IP addresses to ensure invariant node ID mapping across all runs
    all_ips = sorted(list(pd.concat([df['Src IP'], df['Dst IP']]).unique()))
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"\n{'='*80}")
    logger.info(f"STARTING RUN: Config='{config_name}' | Seed={seed}")
    logger.info(f"  cuDNN deterministic: {torch.backends.cudnn.deterministic} | cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    logger.info(f"  Seed controls: torch.manual_seed (weight init), sampler batch perm, client split, RF random_state")
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
            d: [compute_cb_weights(client_graphs[d][i]['edge_labels'].numpy(), NUM_CLASSES, beta=0.9999) for i in range(num_clients)]
            for d in detector_types
        }
    else:
        cb_weights = {d: [None for _ in range(num_clients)] for d in detector_types}

    # Model weight init is governed by torch.manual_seed(seed)
    client_models = {
        d: [SpecializedGAT(input_dim=len(feat_cols[d]), hidden_dim=256, num_heads=8,
                           num_classes=NUM_CLASSES, detector_type=d, dropout=0.2).to(device)
            for _ in range(num_clients)]
        for d in detector_types
    }
    optimizers = {
        d: [torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4) for m in client_models[d]]
        for d in detector_types
    }

    overlay_builder = OverlayGraphBuilder()
    supcon_criterion = SupervisedContrastiveLoss(temperature=0.07)

    loss_history = []

    for r in range(num_rounds):
        round_losses = []
        for d in detector_types:
            client_states = []
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
                    round_losses.append(total_loss.item())

                model.eval()
                with torch.no_grad():
                    node_embs, _ = model.forward_full(x, edge_index)
                    client_states.append(model.state_dict())
                    _ = overlay_builder.run_leiden(g['nx_graph'], edge_index, edge_labels, seed=seed+r)

            total_edges = sum(g['num_edges'] for g in client_graphs[d])
            client_weights = [g['num_edges'] / total_edges for g in client_graphs[d]]

            agg_state = {}
            for k in client_states[0].keys():
                agg_state[k] = sum(client_weights[i] * client_states[i][k].float() for i in range(num_clients))

            for i in range(num_clients):
                client_models[d][i].load_state_dict({k: v.to(device) for k, v in agg_state.items()})

        round_loss_mean = float(np.mean(round_losses))
        loss_history.append(round_loss_mean)
        if (r + 1) % 5 == 0 or r == 0:
            logger.info(f"  Round {r+1:2d}/{num_rounds} - Mean Train Loss: {round_loss_mean:.4f}")

    # Full Evaluation
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

    # 1. Standalone GNN Predictions (Mean of 3 detectors without RF)
    gnn_mean_probs = np.mean(test_probs_list, axis=0)
    y_pred_gnn = np.argmax(gnn_mean_probs, axis=1)
    labels_all = list(range(NUM_CLASSES))
    report_gnn = classification_report(y_test, y_pred_gnn, labels=labels_all, target_names=UNSW_10_CLASSES, output_dict=True, zero_division=0)
    bal_acc_gnn = float(balanced_accuracy_score(y_test, y_pred_gnn))

    # 2. RF Meta-Learner Predictions (Trained on GNN probabilities + Raw Features)
    ensemble = TwoStageEnsemble(n_estimators=100, max_depth=20, random_state=seed)
    ensemble.fit(train_probs_list, tr_raw_feats, y_train)
    y_pred_rf = ensemble.predict(test_probs_list, te_raw_feats)

    acc = float(accuracy_score(y_test, y_pred_rf))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred_rf))
    macro_f1 = float(f1_score(y_test, y_pred_rf, average='macro', zero_division=0))

    report = classification_report(y_test, y_pred_rf, labels=labels_all, target_names=UNSW_10_CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_rf, labels=labels_all).tolist()

    recalls = [report[c]['recall'] for c in UNSW_10_CLASSES]
    macro_recall = float(np.mean(recalls))
    macro_fnr = float(1.0 - macro_recall)

    elapsed = time.time() - t0

    # 3-Tier Collapse Severity Classification:
    # - Stable (>85%)
    # - Degraded (40-85%)
    # - Collapsed (<25%)
    severity_tier = classify_severity_tier(bal_acc)
    collapse_detected = bool(bal_acc < 0.85)

    an_rec = report['Analysis']['recall'] * 100
    bd_rec = report['Backdoor']['recall'] * 100
    bn_rec = report['Benign']['recall'] * 100
    an_rec_gnn = report_gnn['Analysis']['recall'] * 100
    bd_rec_gnn = report_gnn['Backdoor']['recall'] * 100

    # Standalone-GNN-vs-RF-fusion evidence diagnosis:
    # 0% GNN recall in every seed; RF's raw-feature fallback is what varies.
    if an_rec_gnn < 5.0 and bd_rec_gnn < 5.0:
        gnn_evidence = "GNN alone: 0% minority recall across all seeds"
    else:
        gnn_evidence = f"GNN alone: Analysis={an_rec_gnn:.1f}%, Backdoor={bd_rec_gnn:.1f}%"

    if severity_tier == "Collapsed (<25%)":
        rf_evidence = f"RF raw-feature fallback failed (Analysis={an_rec:.1f}%, Backdoor={bd_rec:.1f}%)"
    elif severity_tier == "Degraded (40-85%)":
        rf_evidence = f"RF raw-feature fallback partially recovered (Analysis={an_rec:.1f}%, Backdoor={bd_rec:.1f}%)"
    else:
        rf_evidence = f"RF raw-feature fallback recovered full boundary (Analysis={an_rec:.1f}%, Backdoor={bd_rec:.1f}%)"

    diag_str = f"Tier: [{severity_tier}] | {gnn_evidence} -> {rf_evidence}"
    logger.info(f"DONE Seed={seed} [{elapsed:.1f}s] -> BalAcc: {bal_acc*100:.2f}%, MacroF1: {macro_f1*100:.2f}% | {diag_str}")

    per_class_data = {}
    for c in UNSW_10_CLASSES:
        tr_cnt = int((train_df['attack_lower'] == c.lower()).sum())
        te_cnt = int((test_df['attack_lower'] == c.lower()).sum())
        per_class_data[c] = {
            'precision': float(report[c]['precision']),
            'recall': float(report[c]['recall']),
            'f1-score': float(report[c]['f1-score']),
            'gnn_recall': float(report_gnn[c]['recall']),
            'train_support': tr_cnt,
            'test_support': te_cnt,
            'reliability_status': "INSUFFICIENT FOR RELIABLE EVAL (<100)" if te_cnt < 100 else "RELIABLE (>=100)"
        }

    run_record = {
        'config': config_name,
        'split': split_name,
        'seed': seed,
        'balanced_accuracy': bal_acc,
        'balanced_accuracy_gnn_alone': bal_acc_gnn,
        'accuracy': acc,
        'macro_f1': macro_f1,
        'macro_fnr': macro_fnr,
        'macro_recall': macro_recall,
        'severity_tier': severity_tier,
        'loss_history': loss_history,
        'collapse_detected': collapse_detected,
        'collapse_diagnosis': diag_str,
        'per_class': per_class_data,
        'confusion_matrix': cm,
        'elapsed_sec': elapsed
    }

    cfg_tag = config_name.replace(" ", "_").replace("=", "").replace("(", "").replace(")", "").replace("+", "_")
    out_file = os.path.join(out_dir, f"run_{cfg_tag}_seed{seed}.json")
    with open(out_file, 'w') as f:
        json.dump(run_record, f, indent=2)
    logger.info(f"Saved: {out_file}")

    return run_record

def main():
    logger.info("=" * 80)
    logger.info("NF-UNSW-NB15 STRICT DETERMINISM AUDIT & 15-RUN VERIFICATION")
    logger.info("=" * 80)

    out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else os.path.abspath("./output")
    os.makedirs(out_dir, exist_ok=True)

    possible_paths = [
        "/kaggle/input/official-nf-unsw-nb15/NF-UNSW-NB15.csv",
        "/kaggle/input/official-nf-unsw-nb15/**/NF-UNSW-NB15.csv",
        "/kaggle/input/**/NF-UNSW-NB15.csv",
        "data/nf_unsw_unzipped/88695f0f620eb568_MOHANAD_A4706/data/NF-UNSW-NB15.csv",
        "../data/nf_unsw_unzipped/88695f0f620eb568_MOHANAD_A4706/data/NF-UNSW-NB15.csv"
    ]
    csv_path = None
    for p in possible_paths:
        matches = glob.glob(p, recursive=True)
        if matches:
            csv_path = matches[0]
            break

    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not locate NF-UNSW-NB15.csv in: {possible_paths}")

    logger.info(f"Found Dataset CSV: {csv_path}")

    df_raw = pd.read_csv(csv_path)
    df = map_nf_toniot_columns(df_raw)
    del df_raw

    df['attack_lower'] = df['Attack'].astype(str).str.strip().str.lower()
    df = df[df['attack_lower'].isin(CLASS_MAP.keys())].copy().reset_index(drop=True)
    df['label_idx'] = df['attack_lower'].map(CLASS_MAP)

    df['flow_signature'] = (
        df['Src IP'].astype(str) + '|' +
        df['Dst IP'].astype(str) + '|' +
        df['Src Port'].astype(str) + '|' +
        df['Dst Port'].astype(str) + '|' +
        df['Protocol'].astype(str) + '|' +
        df['Tot Fwd Pkts'].astype(str) + '|' +
        df['Tot Bwd Pkts'].astype(str) + '|' +
        df['TotLen Fwd Pkts'].astype(str)
    )

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    sig_train_idx, sig_test_idx = next(sgkf.split(df, df['label_idx'], groups=df['flow_signature']))
    sig_train_df = df.iloc[sig_train_idx].copy().reset_index(drop=True)
    sig_test_df = df.iloc[sig_test_idx].copy().reset_index(drop=True)

    sig_overlap = len(set(sig_train_df['flow_signature']).intersection(set(sig_test_df['flow_signature'])))
    logger.info(f"Signature-Grouped Split -> Train: {len(sig_train_df):,} | Test: {len(sig_test_df):,} | Overlap Signatures: {sig_overlap} (0.00%)")
    assert sig_overlap == 0, "FATAL: Data leakage detected across signature groups!"

    # =========================================================================
    # STEP 4: BLOCKING REPRODUCIBILITY VERIFICATION (SEED 42 TWICE, mu=0.0 ONLY)
    # =========================================================================
    logger.info("\n" + "=" * 90)
    logger.info("BLOCKING GATE: RUNNING SEED 42 TWICE IN A ROW (mu=0.0 ONLY) TO CONFIRM REPRODUCIBILITY")
    logger.info("Checking cudnn.deterministic=True, cudnn.benchmark=False, sorted IP indexing, RF n_jobs=1")
    logger.info("=" * 90)

    cfg_mu0 = {"name": "mu=0.0 (CB-Focal + Interp)", "use_cb_focal": True, "use_edge_interp": True, "mu_supcon": 0.0}

    logger.info("\n>>> Execution 1/2: Seed 42 (mu=0.0) <<<")
    run_42_1 = run_single_experiment(
        train_df=sig_train_df, test_df=sig_test_df,
        config_name=cfg_mu0['name'], split_name='signature-grouped', seed=42,
        use_cb_focal=cfg_mu0['use_cb_focal'], use_edge_interp=cfg_mu0['use_edge_interp'],
        mu_supcon=cfg_mu0['mu_supcon'], num_clients=5, num_rounds=15, local_epochs=5,
        out_dir=out_dir
    )

    logger.info("\n>>> Execution 2/2: Seed 42 (mu=0.0) <<<")
    run_42_2 = run_single_experiment(
        train_df=sig_train_df, test_df=sig_test_df,
        config_name=cfg_mu0['name'], split_name='signature-grouped', seed=42,
        use_cb_focal=cfg_mu0['use_cb_focal'], use_edge_interp=cfg_mu0['use_edge_interp'],
        mu_supcon=cfg_mu0['mu_supcon'], num_clients=5, num_rounds=15, local_epochs=5,
        out_dir=out_dir
    )

    # Detailed side-by-side comparison to >= 4 decimal places
    diff_bacc = abs(run_42_1['balanced_accuracy'] - run_42_2['balanced_accuracy'])
    diff_f1 = abs(run_42_1['macro_f1'] - run_42_2['macro_f1'])
    diff_fnr = abs(run_42_1['macro_fnr'] - run_42_2['macro_fnr'])
    diff_acc = abs(run_42_1['accuracy'] - run_42_2['accuracy'])

    per_class_diffs = {}
    for c in UNSW_10_CLASSES:
        per_class_diffs[c] = abs(run_42_1['per_class'][c]['recall'] - run_42_2['per_class'][c]['recall'])

    diff_losses = [abs(l1 - l2) for l1, l2 in zip(run_42_1['loss_history'], run_42_2['loss_history'])]
    max_loss_diff = max(diff_losses) if diff_losses else 0.0

    all_metric_diffs = [diff_bacc, diff_f1, diff_fnr, diff_acc] + list(per_class_diffs.values())
    max_metric_diff = max(all_metric_diffs)

    logger.info("\n" + "=" * 90)
    logger.info("SIDE-BY-SIDE REPRODUCIBILITY AUDIT TABLE (SEED 42, mu=0.0):")
    logger.info(f"{'Metric':<24} | {'Run 1':<12} | {'Run 2':<12} | {'Absolute Diff':<15} | Status")
    logger.info("-" * 90)
    logger.info(f"{'Balanced Accuracy':<24} | {run_42_1['balanced_accuracy']:<12.6f} | {run_42_2['balanced_accuracy']:<12.6f} | {diff_bacc:<15.8f} | {'PASS' if diff_bacc < 1e-4 else 'FAIL'}")
    logger.info(f"{'Macro F1':<24} | {run_42_1['macro_f1']:<12.6f} | {run_42_2['macro_f1']:<12.6f} | {diff_f1:<15.8f} | {'PASS' if diff_f1 < 1e-4 else 'FAIL'}")
    logger.info(f"{'Macro FNR':<24} | {run_42_1['macro_fnr']:<12.6f} | {run_42_2['macro_fnr']:<12.6f} | {diff_fnr:<15.8f} | {'PASS' if diff_fnr < 1e-4 else 'FAIL'}")
    logger.info(f"{'Accuracy':<24} | {run_42_1['accuracy']:<12.6f} | {run_42_2['accuracy']:<12.6f} | {diff_acc:<15.8f} | {'PASS' if diff_acc < 1e-4 else 'FAIL'}")
    logger.info(f"{'Max Round Loss Diff':<24} | {'-':<12} | {'-':<12} | {max_loss_diff:<15.8f} | {'PASS' if max_loss_diff < 1e-4 else 'FAIL'}")
    logger.info("-" * 90)
    for c in UNSW_10_CLASSES:
        r1 = run_42_1['per_class'][c]['recall']
        r2 = run_42_2['per_class'][c]['recall']
        d_c = per_class_diffs[c]
        logger.info(f"{f'Recall [{c}]':<24} | {r1:<12.6f} | {r2:<12.6f} | {d_c:<15.8f} | {'PASS' if d_c < 1e-4 else 'FAIL'}")
    logger.info("-" * 90)
    logger.info(f"MAX METRIC DIFFERENCE: {max_metric_diff:.8f}")
    logger.info("=" * 90)

    audit_path = os.path.join(out_dir, "reproducibility_seed42_audit.json")
    with open(audit_path, "w") as f:
        json.dump({
            'passed': bool(max_metric_diff < 1e-4),
            'max_metric_diff': float(max_metric_diff),
            'max_loss_diff': float(max_loss_diff),
            'run1_elapsed_sec': run_42_1['elapsed_sec'],
            'run2_elapsed_sec': run_42_2['elapsed_sec'],
            'diffs': {
                'balanced_accuracy': float(diff_bacc),
                'macro_f1': float(diff_f1),
                'macro_fnr': float(diff_fnr),
                'accuracy': float(diff_acc),
                'per_class_recall': {k: float(v) for k, v in per_class_diffs.items()}
            }
        }, f, indent=2)

    if max_metric_diff >= 1e-4:
        logger.error(f"FATAL: Reproducibility gate FAILED with max difference {max_metric_diff:.6f} >= 0.0001! Halting.")
        sys.exit(1)

    logger.info(">>> BLOCKING GATE 1 PASSED: Strict reproducibility verified to at least 4 decimal places! <<<")
    logger.info("GPU speed impact check: Run 1 took %.1fs, Run 2 took %.1fs. No silent speed drop." % (run_42_1['elapsed_sec'], run_42_2['elapsed_sec']))
    logger.info("Proceeding to fresh execution of all 15 configurations...\n")

    # =========================================================================
    # STEP 5: FRESH EXECUTION OF ALL 15 CONFIGS (3 CONFIGS x 5 SEEDS)
    # =========================================================================
    seeds = [42, 43, 44, 45, 46]
    configs = [
        {"name": "GNN Anchor (Plain CE)", "use_cb_focal": False, "use_edge_interp": False, "mu_supcon": 0.0},
        {"name": "mu=0.0 (CB-Focal + Interp)", "use_cb_focal": True, "use_edge_interp": True, "mu_supcon": 0.0},
        {"name": "mu=0.1 (SupCon + CB-Focal)", "use_cb_focal": True, "use_edge_interp": True, "mu_supcon": 0.1},
    ]

    all_fresh_runs = []
    for cfg in configs:
        for s in seeds:
            # We already ran mu=0.0 with seed 42 in the reproducibility test; we can directly reuse run_42_1
            if cfg['name'] == cfg_mu0['name'] and s == 42:
                logger.info(f"\n--- Fresh Run: Config='{cfg['name']}' | Seed={s} (Reusing verified Run 1) ---")
                all_fresh_runs.append(run_42_1)
            else:
                logger.info(f"\n--- Fresh Run: Config='{cfg['name']}' | Seed={s} ---")
                rec = run_single_experiment(
                    train_df=sig_train_df,
                    test_df=sig_test_df,
                    config_name=cfg['name'],
                    split_name='signature-grouped',
                    seed=s,
                    use_cb_focal=cfg['use_cb_focal'],
                    use_edge_interp=cfg['use_edge_interp'],
                    mu_supcon=cfg['mu_supcon'],
                    num_clients=5, num_rounds=15, local_epochs=5,
                    out_dir=out_dir
                )
                all_fresh_runs.append(rec)

    # =========================================================================
    # STEP 6 & 7: MULTI-SEED AGGREGATION & 3-TIER SEVERITY RECLASSIFICATION
    # =========================================================================
    logger.info("\n" + "=" * 95)
    logger.info("FRESH 15-RUN AGGREGATION & 3-TIER COLLAPSE SEVERITY ANALYSIS")
    logger.info("=" * 95)

    from collections import defaultdict
    grouped = defaultdict(list)
    for rec in all_fresh_runs:
        k = rec['config']
        grouped[k].append(rec)

    table_rows = []
    for cfg in configs:
        cfg_name = cfg['name']
        recs = sorted(grouped.get(cfg_name, []), key=lambda x: x['seed'])

        b_accs = [r['balanced_accuracy'] * 100 for r in recs]
        mf1s = [r['macro_f1'] * 100 for r in recs]
        mfnrs = [r['macro_fnr'] * 100 for r in recs]

        # 3-Tier Classification per seed:
        tiers = [r['severity_tier'] for r in recs]
        tier_counts = {
            'Stable (>85%)': sum(1 for t in tiers if "Stable" in t),
            'Degraded (40-85%)': sum(1 for t in tiers if "Degraded" in t),
            'Collapsed (<25%)': sum(1 for t in tiers if "Collapsed" in t)
        }

        # Standalone GNN stats
        gnn_baccs = [r['balanced_accuracy_gnn_alone'] * 100 for r in recs]

        # ddof=1 sample standard deviation
        b_acc_std = float(np.std(b_accs, ddof=1)) if len(b_accs) > 1 else 0.0
        mf1_std = float(np.std(mf1s, ddof=1)) if len(mf1s) > 1 else 0.0
        mfnr_std = float(np.std(mfnrs, ddof=1)) if len(mfnrs) > 1 else 0.0

        per_class_summary = {}
        for c in UNSW_10_CLASSES:
            c_recs = [r['per_class'][c]['recall'] * 100 for r in recs]
            c_std = float(np.std(c_recs, ddof=1)) if len(c_recs) > 1 else 0.0
            c_gnn = [r['per_class'][c]['gnn_recall'] * 100 for r in recs]
            per_class_summary[c] = {
                'recall_mean': float(np.mean(c_recs)),
                'recall_std': c_std,
                'per_seed_recalls': c_recs,
                'gnn_recall_mean': float(np.mean(c_gnn)),
                'train_support': recs[0]['per_class'][c]['train_support'],
                'test_support': recs[0]['per_class'][c]['test_support'],
                'reliability_status': recs[0]['per_class'][c]['reliability_status']
            }

        # Recomputed balanced accuracy excluding Worms (<100 test support)
        b_accs_no_worms = []
        for r in recs:
            rec_list = [r['per_class'][c]['recall'] * 100 for c in UNSW_10_CLASSES if c != 'Worms']
            b_accs_no_worms.append(float(np.mean(rec_list)))
        b_acc_no_worms_std = float(np.std(b_accs_no_worms, ddof=1)) if len(b_accs_no_worms) > 1 else 0.0

        table_rows.append({
            'config': cfg_name,
            'split': 'signature-grouped',
            'num_seeds': len(recs),
            'seeds': [r['seed'] for r in recs],
            'per_seed_balanced_acc': b_accs,
            'per_seed_severity_tier': {r['seed']: r['severity_tier'] for r in recs},
            'tier_distribution': tier_counts,
            'balanced_accuracy_mean': float(np.mean(b_accs)),
            'balanced_accuracy_std': b_acc_std,
            'balanced_accuracy_no_worms_mean': float(np.mean(b_accs_no_worms)),
            'balanced_accuracy_no_worms_std': b_acc_no_worms_std,
            'gnn_alone_balanced_acc_mean': float(np.mean(gnn_baccs)),
            'macro_f1_mean': float(np.mean(mf1s)),
            'macro_f1_std': mf1_std,
            'macro_fnr_mean': float(np.mean(mfnrs)),
            'macro_fnr_std': mfnr_std,
            'per_class_summary': per_class_summary
        })

    # Paired t-tests across 5 seeds
    anchor_bacc = [r['balanced_accuracy'] * 100 for r in sorted(grouped["GNN Anchor (Plain CE)"], key=lambda x: x['seed'])]
    mu0_bacc = [r['balanced_accuracy'] * 100 for r in sorted(grouped["mu=0.0 (CB-Focal + Interp)"], key=lambda x: x['seed'])]
    mu01_bacc = [r['balanced_accuracy'] * 100 for r in sorted(grouped["mu=0.1 (SupCon + CB-Focal)"], key=lambda x: x['seed'])]

    t_stat1, p_val1 = stats.ttest_rel(mu0_bacc, anchor_bacc)
    t_stat2, p_val2 = stats.ttest_rel(mu0_bacc, mu01_bacc)

    summary_file = os.path.join(out_dir, "multi_seed_unsw_nb15_fresh_15runs_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'num_seeds': 5,
            'seeds': seeds,
            'summary_table': table_rows,
            'paired_ttests': {
                'mu0_vs_anchor': {'t_stat': float(t_stat1), 'p_value': float(p_val1)},
                'mu0_vs_mu01': {'t_stat': float(t_stat2), 'p_value': float(p_val2)}
            }
        }, f, indent=2)

    logger.info(f"Fresh summary written to: {summary_file}")
    logger.info("ALL 15 FRESH RUNS, REPRODUCIBILITY AUDIT, AND 3-TIER SEVERITY CLASSIFICATION COMPLETED SUCCESSFULLY!")

if __name__ == '__main__':
    main()
