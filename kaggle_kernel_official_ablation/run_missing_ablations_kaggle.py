import os
import sys
import json
import time
import glob
import logging
from typing import Dict, List, Tuple, Any

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
                             f1_score, classification_report)
from sklearn.model_selection import StratifiedGroupKFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FedGATSage-Ablations] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FedGATSage_Ablations")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Execution Device: {device}")

PAPER_8_CLASSES = ['Benign', 'Backdoor', 'DDoS', 'DoS', 'Injection', 'Password', 'Scanning', 'XSS']
CLASS_MAP = {c.lower(): i for i, c in enumerate(PAPER_8_CLASSES)}

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
            dur_ms = df['Flow Duration'].fillna(0) + 1.0
            df['Pkt_Rate'] = (df['Tot Fwd Pkts'] + df['Tot Bwd Pkts']) / (dur_ms / 1000.0)
            df['Byte_Rate'] = (df['TotLen Fwd Pkts'] + df['TotLen Bwd Pkts']) / (dur_ms / 1000.0)
            df['Fwd_Pkt_Ratio'] = df['Tot Fwd Pkts'] / (df['Tot Fwd Pkts'] + df['Tot Bwd Pkts'] + 1e-5)
            df['Duration_Log'] = np.log1p(np.maximum(df['Flow Duration'].fillna(0), 0.0))
        elif self.detector_type == 'content':
            tot_pkts = df['Tot Fwd Pkts'] + df['Tot Bwd Pkts'] + 1e-5
            tot_bytes = df['TotLen Fwd Pkts'] + df['TotLen Bwd Pkts'] + 1e-5
            df['Avg_Pkt_Size'] = tot_bytes / tot_pkts
            df['Byte_Asymmetry'] = (df['TotLen Fwd Pkts'] - df['TotLen Bwd Pkts']) / tot_bytes
            df['Pkt_Asymmetry'] = (df['Tot Fwd Pkts'] - df['Tot Bwd Pkts']) / tot_pkts
            df['SYN_to_Pkt'] = df['SYN Flag Cnt'] / tot_pkts
            df['ACK_to_Pkt'] = df['ACK Flag Cnt'] / tot_pkts
        elif self.detector_type == 'behavioral':
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
                 num_classes: int = 8, disable_node_agg: bool = False, dropout: float = 0.2):
        super().__init__()
        self.disable_node_agg = disable_node_agg
        if not self.disable_node_agg:
            self.gat1 = GATConv(input_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
            self.gat2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            # Pure Edge MLP without node aggregation
            self.edge_mlp = nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward_batch(self, x: torch.Tensor, edge_index: torch.Tensor, batch_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.disable_node_agg:
            h1 = F.elu(self.gat1(x, edge_index))
            h2 = self.gat2(h1, edge_index)
            src = edge_index[0, batch_idx]
            dst = edge_index[1, batch_idx]
            edge_embs = self.edge_mlp(torch.cat([h2[src], h2[dst]], dim=1))
        else:
            h2 = x
            src = edge_index[0, batch_idx]
            dst = edge_index[1, batch_idx]
            edge_embs = self.edge_mlp(torch.cat([x[src], x[dst]], dim=1))
        logits = self.classifier(edge_embs)
        return h2, edge_embs, logits

    def forward_full(self, x: torch.Tensor, edge_index: torch.Tensor, chunk_size: int = 50000) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.disable_node_agg:
            h1 = F.elu(self.gat1(x, edge_index))
            h2 = self.gat2(h1, edge_index)
        else:
            h2 = x
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
    def __init__(self, include_raw_features: bool = True, n_estimators: int = 100, max_depth: int = 20, random_state: int = 42):
        self.include_raw_features = include_raw_features
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
        if self.include_raw_features:
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

def run_experiment_run(train_df: pd.DataFrame, test_df: pd.DataFrame,
                       config_name: str, seed: int,
                       use_cb_focal: bool, use_edge_interp: bool, mu_supcon: float,
                       include_raw_in_rf: bool = True, disable_node_agg: bool = False,
                       num_clients: int = 5, num_rounds: int = 15, local_epochs: int = 5,
                       out_dir: str = "/kaggle/working") -> Dict[str, Any]:
    t0 = time.time()
    set_seed(seed)

    logger.info(f"RUNNING: Config='{config_name}' | Seed={seed} | NodeAgg={not disable_node_agg} | RF_RawFeats={include_raw_in_rf}")

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
                           num_classes=8, disable_node_agg=disable_node_agg, dropout=0.2).to(device)
            for _ in range(num_clients)]
        for d in detector_types
    }
    optimizers = {
        d: [torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4) for m in client_models[d]]
        for d in detector_types
    }

    supcon_criterion = SupervisedContrastiveLoss(temperature=0.07)

    for r in range(num_rounds):
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

                model.eval()
                with torch.no_grad():
                    client_states.append(model.state_dict())

            # FedAvg
            total_edges = sum(g['num_edges'] for g in client_graphs[d])
            client_weights = [g['num_edges'] / total_edges for g in client_graphs[d]]

            agg_state = {}
            for k in client_states[0].keys():
                agg_state[k] = sum(client_weights[i] * client_states[i][k].float() for i in range(num_clients))

            for i in range(num_clients):
                client_models[d][i].load_state_dict({k: v.to(device) for k, v in agg_state.items()})

    # Evaluation via TwoStageEnsemble
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

    ensemble = TwoStageEnsemble(include_raw_features=include_raw_in_rf, n_estimators=100, max_depth=20, random_state=seed)
    ensemble.fit(train_probs_list, tr_raw_feats, y_train)
    y_pred = ensemble.predict(test_probs_list, te_raw_feats)

    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average='macro', zero_division=0))

    report = classification_report(y_test, y_pred, labels=list(range(8)), target_names=PAPER_8_CLASSES, output_dict=True, zero_division=0)
    per_class_table = {}
    for cls in PAPER_8_CLASSES:
        per_class_table[cls] = {
            'recall': float(report[cls]['recall']),
            'f1': float(report[cls]['f1-score']),
            'support': int(report[cls]['support'])
        }

    elapsed_min = (time.time() - t0) / 60.0
    logger.info(f"Done ({elapsed_min:.2f}m) -> Acc: {acc*100:.2f}%, BalAcc: {bal_acc*100:.2f}%, MacroF1: {macro_f1*100:.2f}%")

    run_record = {
        'config': config_name,
        'split': 'signature-grouped',
        'seed': seed,
        'dataset_file': 'NF-ToN-IoT.csv',
        'row_count': len(train_df) + len(test_df),
        'split_hash': 'sgkf_n5_r42_hash_8tuple',
        'hyperparameters': {
            'use_cb_focal': use_cb_focal,
            'use_edge_interp': use_edge_interp,
            'mu_supcon': mu_supcon,
            'include_raw_in_rf': include_raw_in_rf,
            'disable_node_agg': disable_node_agg
        },
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'macro_f1': macro_f1,
        'per_class': per_class_table,
        'elapsed_minutes': elapsed_min
    }

    cfg_slug = config_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace('+', '_')
    fname = f"run_ablation_{cfg_slug}_seed{seed}.json"
    fpath = os.path.join(out_dir, fname)

    with open(fpath, 'w') as f:
        json.dump(run_record, f, indent=2)
    logger.info(f"Saved JSON: {fpath}")

    return run_record

def main():
    logger.info("=" * 80)
    logger.info("RUNNING MISSING ABLATION EXPERIMENTS (TASKS 4 & 5)")
    logger.info("=" * 80)

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
        raise FileNotFoundError("Official NF-ToN-IoT CSV not found")

    raw_df = pd.read_csv(data_file)
    df = map_nf_toniot_columns(raw_df)
    df['attack_lower'] = df['Attack'].astype(str).str.lower()
    df = df[df['attack_lower'].isin(CLASS_MAP)].copy().reset_index(drop=True)
    df['label_idx'] = df['attack_lower'].map(CLASS_MAP)

    hash_cols = ['Src Port', 'Dst Port', 'Protocol', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'Flow Duration']
    df['flow_signature'] = df[hash_cols].astype(str).agg('|'.join, axis=1)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    sig_train_idx, sig_test_idx = next(sgkf.split(df, df['label_idx'], groups=df['flow_signature']))
    sig_train_df = df.iloc[sig_train_idx].copy().reset_index(drop=True)
    sig_test_df = df.iloc[sig_test_idx].copy().reset_index(drop=True)

    out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "kaggle_results_ablations"
    os.makedirs(out_dir, exist_ok=True)

    seeds = [42, 43, 44]
    
    # Task 4 Missing Configs
    # (b) Plain CE + Sampler + CB-Focal (no interp, no supcon)
    # (e) Full Model (mu=0.1) without Raw Features in RF
    # Task 5 Missing Configs
    # (f) Edge-Only Anchor (no node aggregation)
    # (g) Edge-Only mu=0.1 (no node aggregation)
    
    ablation_tasks = [
        {"name": "b_plain_ce_plus_cb_focal", "use_cb_focal": True, "use_edge_interp": False, "mu_supcon": 0.0, "include_raw_in_rf": True, "disable_node_agg": False},
        {"name": "e_full_model_no_raw_rf", "use_cb_focal": True, "use_edge_interp": True, "mu_supcon": 0.1, "include_raw_in_rf": False, "disable_node_agg": False},
        {"name": "f_edge_only_anchor", "use_cb_focal": False, "use_edge_interp": False, "mu_supcon": 0.0, "include_raw_in_rf": True, "disable_node_agg": True},
        {"name": "g_edge_only_mu0.1", "use_cb_focal": True, "use_edge_interp": True, "mu_supcon": 0.1, "include_raw_in_rf": True, "disable_node_agg": True},
    ]

    for task in ablation_tasks:
        for s in seeds:
            run_experiment_run(
                train_df=sig_train_df,
                test_df=sig_test_df,
                config_name=task['name'],
                seed=s,
                use_cb_focal=task['use_cb_focal'],
                use_edge_interp=task['use_edge_interp'],
                mu_supcon=task['mu_supcon'],
                include_raw_in_rf=task['include_raw_in_rf'],
                disable_node_agg=task['disable_node_agg'],
                out_dir=out_dir
            )

if __name__ == '__main__':
    main()
