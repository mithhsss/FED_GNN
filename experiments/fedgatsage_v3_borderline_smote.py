"""
FedGATSage - Version 3: Local Borderline-SMOTE Strategy
Technique: Local Borderline-SMOTE oversampling on borderline/danger minority samples.
Paper Citation:
    Hui Han, Wen-Yuan Wang, and Bing-Huan Mao.
    "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning."
    In International Conference on Intelligent Computing (ICIC 2005),
    Lecture Notes in Computer Science (LNCS 3644), pp. 257-264. Springer, 2005.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support, f1_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [V3-Han2005-BorderlineSMOTE] %(message)s')
logger = logging.getLogger("FedGATSage_V3")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# 1. Dataset Loading (500K-Scale Stratified CIC-ToN-IoT)
# ---------------------------------------------------------------------------
NF_MAPPINGS = {
    'L4_SRC_PORT': 'Src Port', 'L4_DST_PORT': 'Dst Port',
    'PROTOCOL': 'Protocol', 'L7_PROTO': 'L7 Proto',
    'IN_BYTES': 'TotLen Fwd Pkts', 'OUT_BYTES': 'TotLen Bwd Pkts',
    'IN_PKTS': 'Tot Fwd Pkts', 'OUT_PKTS': 'Tot Bwd Pkts',
    'TCP_FLAGS': 'TCP Flags', 'FLOW_DURATION_MILLISECONDS': 'Flow Duration',
    'Total Fwd Packets': 'Tot Fwd Pkts', 'Total Backward Packets': 'Tot Bwd Pkts',
    'Fwd Packets Length Total': 'TotLen Fwd Pkts', 'Bwd Packets Length Total': 'TotLen Bwd Pkts',
    'Label': 'Label', 'Attack': 'Attack'
}

STANDARD_ATTACKS = {
    'benign': 'Benign', 'ddos': 'DDoS', 'dos': 'DoS',
    'scanning': 'Scanning', 'injection': 'Injection', 'xss': 'XSS',
    'password': 'Password', 'backdoor': 'Backdoor',
    'mitm': 'MITM', 'ransomware': 'Ransomware'
}

PAPER_8_CLASSES = ['Benign', 'Backdoor', 'DDoS', 'DoS', 'Injection', 'Password', 'Scanning', 'XSS']

def load_dataset(data_dir: str = 'data', max_samples: int = 500000, dataset_type: str = 'cic_ton_iot') -> Tuple[pd.DataFrame, str]:
    input_dirs = [
        os.path.join(data_dir, 'cic_ton_iot'),
        os.path.join(data_dir, 'nftoniot'),
        '/kaggle/input/cictoniot', '/kaggle/input/cic-ton-iot',
        '/kaggle/input/nftoniot',
        data_dir
    ]
    file_path = None
    dataset_name = "CIC-ToN-IoT"
    
    for d in input_dirs:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in sorted(files):
                    if dataset_type == 'cic_ton_iot' and ('cic' in f.lower() or 'ton-iot' in f.lower()) and (f.endswith('.parquet') or f.endswith('.csv')):
                        file_path = os.path.join(root, f)
                        dataset_name = "CIC-ToN-IoT"
                        break
                    elif f.endswith('.parquet') or f.endswith('.csv'):
                        file_path = os.path.join(root, f)
                        dataset_name = "CIC-ToN-IoT" if "cic" in f.lower() else "NF-ToN-IoT"
                        break
                if file_path: break
        if file_path: break

    if not file_path:
        raise FileNotFoundError(f"No dataset found in {input_dirs}")

    logger.info(f"Loading data from: {file_path}")
    df = pd.read_parquet(file_path) if file_path.endswith('.parquet') else pd.read_csv(file_path, low_memory=False)
    df = df.rename(columns={k: v for k, v in NF_MAPPINGS.items() if k in df.columns})

    if 'Src IP' not in df.columns:
        proto = df.get('Protocol', pd.Series(np.zeros(len(df))))
        fwd_p = df.get('Tot Fwd Pkts', pd.Series(np.zeros(len(df))))
        df['Src IP'] = [f"10.0.{int(abs(p)) % 254 + 1}.{int(abs(s)) % 254 + 1}" for p, s in zip(proto, fwd_p)]
    if 'Dst IP' not in df.columns:
        proto = df.get('Protocol', pd.Series(np.zeros(len(df))))
        bwd_p = df.get('Tot Bwd Pkts', pd.Series(np.zeros(len(df))))
        df['Dst IP'] = [f"192.168.{int(abs(p)) % 254 + 1}.{int(abs(d)) % 254 + 1}" for p, d in zip(proto, bwd_p)]

    if 'Attack' in df.columns:
        df['Attack'] = (df['Attack'].astype(str).str.strip().str.lower()
                        .map(lambda a: STANDARD_ATTACKS.get(a, a.capitalize())))
    df = df.dropna(subset=['Attack'])
    
    # Filter to 8 paper-evaluated classes
    df = df[df['Attack'].isin(PAPER_8_CLASSES)].copy()

    # Stratified sampling preserving realistic natural distribution
    if len(df) > max_samples:
        logger.info(f"Drawing stratified {max_samples:,}-flow subset with realistic class ratios preserved...")
        df = df.groupby('Attack', group_keys=False).apply(
            lambda x: x.sample(n=max(1, int(np.round(len(x) / len(df) * max_samples))), random_state=42)
        ).sample(frac=1.0, random_state=42).reset_index(drop=True)

    logger.info(f"Loaded {len(df):,} flows ({dataset_name}). Realistic Class Distribution:\n{df['Attack'].value_counts().to_string()}")
    return df, dataset_name

# ---------------------------------------------------------------------------
# 2. Strategy 3: Local Borderline-SMOTE Implementation (Han et al., ICIC 2005)
# ---------------------------------------------------------------------------
def local_borderline_smote_han2005(
    client_df: pd.DataFrame,
    feat_cols: List[str],
    target_ratio: float = 0.25,
    m_neighbors: int = 5,
    k_neighbors: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    r"""
    Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning.
    
    Paper Citation:
        Hui Han, Wen-Yuan Wang, and Bing-Huan Mao.
        "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning."
        In International Conference on Intelligent Computing (ICIC 2005),
        Lecture Notes in Computer Science (LNCS 3644), pp. 257-264. Springer, 2005.
    
    Exact Algorithm (Han et al. 2005):
        Let T be the entire training set of client k, P be the minority class flows,
        and N be all remaining (majority/other-class) flows.
        
        Step 1: Standardize continuous flow features to zero mean, unit variance:
                \mathbf{z} = (\mathbf{x} - \boldsymbol{\mu}) / (\boldsymbol{\sigma} + \epsilon)
                so Euclidean distances are not distorted by disparate feature units.
                
        Step 2: For every minority sample \mathbf{p}_i \in P, calculate its m nearest
                neighbors in the entire local dataset T. Let m' be the count of
                majority/other-class samples among these m neighbors:
                - If m' = m: \mathbf{p}_i is NOISE (completely surrounded by other classes).
                  Discard, do not synthesize.
                - If 0 <= m' <= m/2: \mathbf{p}_i is SAFE (mostly surrounded by its own class).
                  Discard from synthesis.
                - If m/2 < m' < m: \mathbf{p}_i is in the DANGER set (decision boundary).
                  \mathbf{p}_i \in \text{DANGER}.
                  
        Step 3: For each \mathbf{p}_i \in \text{DANGER}, calculate its k nearest neighbors
                strictly from within the minority class P (k-NN(p_i) \cap P).
                
        Step 4: Synthesize:
                \mathbf{s} = \mathbf{p}_i + r \cdot (\mathbf{p}_{nn} - \mathbf{p}_i),
                where r \sim \mathcal{U}(0, 1) and \mathbf{p}_{nn} \in k\text{-NN}(\mathbf{p}_i) \cap P.
                
        Step 5: Map back to unstandardized feature space: \mathbf{x} = \mathbf{z} \cdot \boldsymbol{\sigma} + \boldsymbol{\mu}.
                Attribute synthetic flows to authentic IP endpoints (Src IP, Dst IP) of parent flow
                \mathbf{p}_i to preserve authentic client communication topology.
    """
    rng = np.random.RandomState(random_state)
    counts = client_df['Attack'].value_counts()
    majority_count = counts.max()
    target_count = max(int(majority_count * target_ratio), 500)

    # Step 1: Feature Standardization for stable Euclidean k-NN
    X_raw = client_df[feat_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0).values.astype(np.float32)
    means = np.mean(X_raw, axis=0)
    stds = np.std(X_raw, axis=0)
    stds[stds == 0] = 1.0
    X_norm = (X_raw - means) / stds

    y = client_df['Attack'].values
    all_indices = np.arange(len(client_df))

    # Step 2: Global m-NN over entire training set T = P \cup N
    m = m_neighbors
    global_nn = NearestNeighbors(n_neighbors=m + 1, metric='euclidean', n_jobs=-1).fit(X_norm)
    _, global_neigh_idx = global_nn.kneighbors(X_norm)
    # Remove self from neighbors (column 0)
    global_neigh_idx = global_neigh_idx[:, 1:]

    synth_records = []

    for cls_name, grp in client_df.groupby('Attack'):
        cnt = len(grp)
        if cnt < target_count and cnt >= 2:
            needed = target_count - cnt
            cls_mask = (y == cls_name)
            cls_indices = all_indices[cls_mask]

            # Step 2: Identify DANGER set: m/2 < m' < m
            danger_indices = []
            for idx in cls_indices:
                neighs = global_neigh_idx[idx]
                m_prime = np.sum(y[neighs] != cls_name)
                # DANGER condition
                if (m / 2.0) < m_prime < m:
                    danger_indices.append(idx)

            # Fallback if no strict DANGER points exist (e.g. extremely isolated or compact clusters)
            if len(danger_indices) == 0:
                m_primes = [np.sum(y[global_neigh_idx[idx]] != cls_name) for idx in cls_indices]
                non_noise = [idx for idx, mp in zip(cls_indices, m_primes) if mp < m]
                danger_indices = non_noise if non_noise else list(cls_indices)

            danger_indices = np.array(danger_indices)

            # Step 3: k-NN within minority class P
            k_val = min(k_neighbors, cnt - 1)
            minority_feats = X_norm[cls_indices]
            cls_nn = NearestNeighbors(n_neighbors=k_val + 1, metric='euclidean', n_jobs=-1).fit(minority_feats)
            _, min_neigh_idx = cls_nn.kneighbors(X_norm[danger_indices])
            min_neigh_idx = min_neigh_idx[:, 1:]

            # Step 4: Synthesize flows
            for n_i in range(needed):
                d_idx = n_i % len(danger_indices)
                parent_global_idx = danger_indices[d_idx]
                parent_row = client_df.iloc[parent_global_idx]

                chosen_neighbor_local_idx = min_neigh_idx[d_idx, n_i % k_val]
                chosen_neighbor_global_idx = cls_indices[chosen_neighbor_local_idx]

                p_norm = X_norm[parent_global_idx]
                p_nn_norm = X_norm[chosen_neighbor_global_idx]

                # Exact formula: s = p_i + r * (p_{nn} - p_i), r ~ Uniform(0, 1)
                r = rng.uniform(0.0, 1.0)
                s_norm = p_norm + r * (p_nn_norm - p_norm)
                # Unnormalize back to natural flow feature domain
                s_raw = np.maximum(s_norm * stds + means, 0.0)

                row_dict = {feat_cols[f_i]: s_raw[f_i] for f_i in range(len(feat_cols))}
                # Preserve authentic communication endpoints
                row_dict['Src IP'] = parent_row['Src IP']
                row_dict['Dst IP'] = parent_row['Dst IP']
                row_dict['Attack'] = cls_name
                if 'Label' in client_df.columns:
                    row_dict['Label'] = parent_row['Label']
                synth_records.append(row_dict)

    if synth_records:
        synth_df = pd.DataFrame(synth_records)
        balanced_df = pd.concat([client_df, synth_df], ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        logger.info(f"Client Borderline-SMOTE: {len(client_df):,} -> {len(balanced_df):,} samples (+{len(synth_records):,} synthetic flows).")
        return balanced_df
    return client_df

# ---------------------------------------------------------------------------
# 3. Model Architecture
# ---------------------------------------------------------------------------
from torch_geometric.nn import GATConv, SAGEConv

class TemporalGATDetector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, num_heads: int = 8, num_classes: int = 8, dropout: float = 0.2):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim // num_heads, heads=num_heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.gat3 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.skip1 = nn.Linear(in_dim, hidden_dim)
        self.skip2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        res1 = self.skip1(x)
        x = F.elu(self.norm1(self.gat1(x, edge_index))) + res1
        x = self.dropout(x)
        res2 = self.skip2(x)
        x = F.elu(self.norm2(self.gat2(x, edge_index))) + res2
        x = self.dropout(x)
        x = F.elu(self.norm3(self.gat3(x, edge_index))) + x
        x = self.dropout(x)
        u, v = edge_index
        flow_feat = torch.cat([x[u], x[v], x[u] * x[v], torch.abs(x[u] - x[v])], dim=1)
        return x, self.classifier(flow_feat)

# ---------------------------------------------------------------------------
# 4. Leiden Graph Builder
# ---------------------------------------------------------------------------
def build_graph(df: pd.DataFrame, feat_cols: List[str], label_mapper: Dict[str, int]) -> Dict[str, Any]:
    src_ips = df['Src IP'].astype(str).values
    dst_ips = df['Dst IP'].astype(str).values
    all_ips = list(pd.concat([pd.Series(src_ips), pd.Series(dst_ips)]).unique())
    ip_map = {ip: i for i, ip in enumerate(all_ips)}

    feat_vals = np.log1p(np.maximum(df[feat_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0).values.astype(np.float32), 0.0))
    feat_mat = np.zeros((len(all_ips), len(feat_cols)), dtype=np.float32)
    ip_cnt = np.ones(len(all_ips), dtype=np.float32)
    for i in range(len(df)):
        si, di = ip_map[src_ips[i]], ip_map[dst_ips[i]]
        feat_mat[si] += feat_vals[i]; ip_cnt[si] += 1
        feat_mat[di] += feat_vals[i]; ip_cnt[di] += 1
    feat_mat /= ip_cnt[:, None]

    edge_idx = torch.tensor([[ip_map[s] for s in src_ips], [ip_map[d] for d in dst_ips]], dtype=torch.long)
    edge_lbl = torch.tensor([label_mapper.get(a, 0) for a in df['Attack'].values], dtype=torch.long)

    # Leiden Community Detection (guarantees well-connected communities, eliminates Louvain badly connected nodes)
    try:
        import igraph as ig
        import leidenalg as la
        edge_tuples = [(ip_map[s], ip_map[d]) for s, d in zip(src_ips, dst_ips)]
        g_ig = ig.Graph(n=len(all_ips), edges=edge_tuples, directed=False)
        partition = la.find_partition(g_ig, la.ModularityVertexPartition, seed=42)
        ip_comm = list(partition.membership)
    except Exception as e:
        logger.warning(f"Leiden community fallback: {e}")
        ip_comm = [i % 10 for i in range(len(all_ips))]

    return {
        'features': torch.tensor(feat_mat, dtype=torch.float32),
        'edge_index': edge_idx,
        'edge_labels': edge_lbl,
        'ip_community': ip_comm,
        'flow_raw_feats': feat_vals
    }

# ---------------------------------------------------------------------------
# 5. Execution Pipeline
# ---------------------------------------------------------------------------
def run_v3_experiment(
    num_clients: int = 5,
    num_rounds: int = 15,
    local_epochs: int = 10,
    max_samples: int = 500000,
    target_ratio: float = 0.25
) -> Dict[str, Any]:
    raw_df, dataset_name = load_dataset(max_samples=max_samples)
    unique_attacks = sorted(raw_df['Attack'].unique())
    label_mapper = {a: i for i, a in enumerate(unique_attacks)}
    inv_mapper = {i: a for a, i in label_mapper.items()}
    num_classes = len(unique_attacks)

    train_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42, stratify=raw_df['Attack'])
    client_dfs = np.array_split(train_df.sample(frac=1.0, random_state=42).reset_index(drop=True), num_clients)

    feat_cols = [c for c in raw_df.columns if c not in ['Label', 'Attack', 'Src IP', 'Dst IP', 'Timestamp']]
    test_graph = build_graph(test_df, feat_cols, label_mapper)

    # Strategy 3: Apply Local Borderline-SMOTE strictly within each client's training fold
    logger.info("Applying Strategy 3 (Han et al. 2005 Borderline-SMOTE) locally to clients...")
    smote_client_dfs = [local_borderline_smote_han2005(cdf, feat_cols, target_ratio=target_ratio) for cdf in client_dfs]
    client_graphs = [build_graph(scdf, feat_cols, label_mapper) for scdf in smote_client_dfs]

    in_dim = client_graphs[0]['features'].shape[1]
    detector_types = ['temporal', 'content', 'behavioral']
    models = {d: [TemporalGATDetector(in_dim, 256, 8, num_classes).to(device) for _ in range(num_clients)] for d in detector_types}
    opts = {d: [torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4) for m in models[d]] for d in detector_types}
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Starting {num_rounds} rounds with Strategy 3 (Han et al. 2005 Borderline-SMOTE)...")
    for r in range(num_rounds):
        for d in detector_types:
            c_states, c_accs = [], []
            for c_idx in range(num_clients):
                m, opt = models[d][c_idx], opts[d][c_idx]
                g = client_graphs[c_idx]
                x, ei, el = g['features'].to(device), g['edge_index'].to(device), g['edge_labels'].to(device)

                m.train()
                for _ in range(local_epochs):
                    opt.zero_grad()
                    _, preds = m(x, ei)
                    loss = criterion(preds, el)
                    loss.backward()
                    opt.step()

                m.eval()
                with torch.no_grad():
                    _, vp = m(x, ei)
                    acc = (vp.argmax(1) == el).float().mean().item()
                    c_accs.append(acc)
                    c_states.append({k: v.cpu() for k, v in m.state_dict().items()})

            agg_state = {k: sum(c_states[i][k].float() for i in range(num_clients)) / num_clients for k in c_states[0]}
            for c_idx in range(num_clients):
                models[d][c_idx].load_state_dict({k: v.to(device) for k, v in agg_state.items()})

        logger.info(f"Round {r+1:02d}/{num_rounds} complete.")

    # Two-Stage Meta Evaluation
    tx, te = test_graph['features'].to(device), test_graph['edge_index'].to(device)
    test_probs = []
    with torch.no_grad():
        for d in detector_types:
            _, logits = models[d][0](tx, te)
            test_probs.append(F.softmax(logits, dim=1).cpu().numpy())
    
    X_meta = np.concatenate(test_probs + [test_graph['flow_raw_feats']], axis=1)
    y_test = test_graph['edge_labels'].numpy()
    
    s_idx = len(y_test) // 2
    rf = RandomForestClassifier(n_estimators=100, max_depth=16, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_meta[:s_idx], y_test[:s_idx])
    final_preds = rf.predict(X_meta[s_idx:])
    y_eval = y_test[s_idx:]

    b_acc = balanced_accuracy_score(y_eval, final_preds)
    m_f1 = f1_score(y_eval, final_preds, average='macro', zero_division=0)
    w_f1 = f1_score(y_eval, final_preds, average='weighted', zero_division=0)
    acc = accuracy_score(y_eval, final_preds)
    
    target_names = [inv_mapper[i] for i in range(num_classes)]
    cls_report = classification_report(y_eval, final_preds, target_names=target_names, digits=4, zero_division=0)
    per_class_f1 = f1_score(y_eval, final_preds, average=None, zero_division=0)
    per_class_dict = {inv_mapper[i]: round(float(per_class_f1[i]), 4) for i in range(num_classes)}

    logger.info(f"\n--- V3 Evaluation Summary (Han et al. 2005 Borderline-SMOTE) ---\n"
                f"Accuracy: {acc*100:.2f}%\n"
                f"Balanced Accuracy: {b_acc*100:.2f}%\n"
                f"Macro F1: {m_f1*100:.2f}%\n"
                f"Weighted F1: {w_f1*100:.2f}%\n"
                f"Per-Class F1:\n{json.dumps(per_class_dict, indent=2)}\n"
                f"\nDetailed Classification Report:\n{cls_report}")

    return {
        'method': 'V3_BorderlineSMOTE_Han2005',
        'accuracy': acc,
        'balanced_accuracy': b_acc,
        'macro_f1': m_f1,
        'weighted_f1': w_f1,
        'per_class_f1': per_class_dict,
        'report': cls_report
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run FedGATSage V3 (Han et al. 2005 Borderline-SMOTE + Leiden)")
    parser.add_argument('--clients', type=int, default=5, help="Number of clients")
    parser.add_argument('--rounds', type=int, default=15, help="Number of FL rounds")
    parser.add_argument('--epochs', type=int, default=10, help="Local epochs per round")
    parser.add_argument('--samples', type=int, default=500000, help="Max total dataset samples")
    parser.add_argument('--target_ratio', type=float, default=0.25, help="Oversampling ratio relative to majority class")
    args = parser.parse_args()

    run_v3_experiment(
        num_clients=args.clients,
        num_rounds=args.rounds,
        local_epochs=args.epochs,
        max_samples=args.samples,
        target_ratio=args.target_ratio
    )
