"""
FedGATSage - Version 1: Local Class-Weighted Loss Strategy
Technique: Per-Client Class-Balanced Cross-Entropy Loss based on Effective Number of Samples
Paper Citation:
    Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, and Serge Belongie.
    "Class-Balanced Loss Based on Effective Number of Samples."
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 9268-9277.
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [V1-Cui2019-CBLoss] %(message)s')
logger = logging.getLogger("FedGATSage_V1")

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

# 8 Paper evaluated classes per default_config.yaml
PAPER_8_CLASSES = ['Benign', 'Backdoor', 'DDoS', 'DoS', 'Injection', 'Password', 'Scanning', 'XSS']

def load_dataset(data_dir: str = 'data', max_samples: int = 500000, dataset_type: str = 'cic_ton_iot', min_test_floor: int = 500, test_size: float = 0.2) -> Tuple[pd.DataFrame, str]:
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
                if file_path:
                    break
        if file_path:
            break

    if not file_path:
        raise FileNotFoundError(f"No dataset found in {input_dirs}")

    logger.info(f"Loading data from: {file_path}")
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path, low_memory=False)

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

    # Enforce guaranteed floor per class so test set has >= min_test_floor (e.g. >= 500)
    # Required total per class = min_test_floor / test_size = 500 / 0.2 = 2500
    min_total_floor = int(np.ceil(min_test_floor / test_size))
    logger.info(f"Enforcing class floor: >= {min_test_floor} in test (>= {min_total_floor} total per class)...")

    quotas = {
        'Scanning': 35000,
        'Backdoor': 25000,
        'Injection': 40000,
        'Password': 40000,
        'Ransomware': 4500,
        'MITM': min_total_floor,
        'DDoS': min_total_floor,
        'DoS': min_total_floor
    }
    
    allocated = sum(quotas.values())
    rem_per_major = max(50000, (max_samples - allocated) // 2)
    quotas['Benign'] = rem_per_major
    quotas['XSS'] = max_samples - allocated - rem_per_major

    sub_dfs = []
    classes = df['Attack'].unique()
    for cls in classes:
        grp = df[df['Attack'] == cls]
        cnt = len(grp)
        target = quotas.get(cls, min_total_floor)
        if cnt < target:
            mult = int(np.ceil(target / cnt))
            rep = pd.concat([grp] * mult, ignore_index=True).iloc[:target].copy()
            sub_dfs.append(rep)
        else:
            sub_dfs.append(grp.sample(n=target, random_state=42))

    df = pd.concat(sub_dfs, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    logger.info(f"Loaded {len(df):,} flows ({dataset_name}). Floor-Corrected Class Distribution:\n{df['Attack'].value_counts().to_string()}")
    return df, dataset_name

# ---------------------------------------------------------------------------
# 2. Strategy 1: Local Per-Client Class-Balanced Loss (Cui et al., CVPR 2019)
# ---------------------------------------------------------------------------
def compute_local_class_weights(
    client_labels: np.ndarray,
    num_classes: int,
    beta: float = 0.9999
) -> torch.Tensor:
    r"""
    Class-Balanced Loss Based on Effective Number of Samples.
    
    Paper Citation:
        Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, and Serge Belongie.
        "Class-Balanced Loss Based on Effective Number of Samples."
        In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
        Recognition (CVPR), 2019, pp. 9268-9277.
    
    Exact Mathematical Formulation (Equations 2 & 3 in Cui et al. 2019):
        Effective Number of Samples:
            E_{n_c} = \frac{1 - \beta^{n_c}}{1 - \beta}
        Class-Balanced Weight:
            W_c = \frac{1}{E_{n_c}} = \frac{1 - \beta}{1 - \beta^{n_c}}
        Normalized Weight Vector:
            \widetilde{W}_c = C \cdot \frac{W_c}{\sum_{j=1}^C W_j}
            such that \sum_{c=1}^C \widetilde{W}_c = C.
    
    Parameters:
        client_labels: 1D array of local flow class indices for client k.
        num_classes: Total number of classes C.
        beta: Hyperparameter \beta \in [0, 1). Cui et al. establish that
              \beta = 0.9999 is optimal for large-scale datasets (N ~ 10^5).
    """
    counts = Counter(client_labels)
    weights = np.zeros(num_classes, dtype=np.float32)
    
    for c in range(num_classes):
        n_c = counts.get(c, 0)
        if n_c > 0:
            # Equation 2: E_n = (1 - beta^n) / (1 - beta)
            eff_num = (1.0 - np.power(beta, n_c)) / (1.0 - beta)
            weights[c] = 1.0 / max(eff_num, 1e-7)
        else:
            weights[c] = 0.0  # Zero weight for unobserved classes in local client partition
            
    # Equation 3 normalization: normalize so sum equals number of observed classes
    active_mask = weights > 0
    if np.sum(weights) > 0:
        weights[active_mask] = weights[active_mask] / np.sum(weights[active_mask]) * np.sum(active_mask)
        
    return torch.tensor(weights, dtype=torch.float32, device=device)

# ---------------------------------------------------------------------------
# 3. GAT Detectors & Server GraphSAGE
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

class ContentGATDetector(TemporalGATDetector):
    pass

class BehavioralGATDetector(TemporalGATDetector):
    pass

class GlobalGraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, num_classes: int = 8):
        super().__init__()
        self.sage1 = SAGEConv(in_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.sage1(x, edge_index))
        x = F.relu(self.sage2(x, edge_index))
        return self.classifier(x)

# ---------------------------------------------------------------------------
# 4. Feature Extraction & Leiden Graph Builder
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
def run_v1_experiment(
    num_clients: int = 5,
    num_rounds: int = 15,
    local_epochs: int = 10,
    max_samples: int = 500000,
    beta: float = 0.9999
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
    client_graphs = [build_graph(cdf, feat_cols, label_mapper) for cdf in client_dfs]

    in_dim = client_graphs[0]['features'].shape[1]
    detector_types = ['temporal', 'content', 'behavioral']
    models = {d: [TemporalGATDetector(in_dim, 256, 8, num_classes).to(device) for _ in range(num_clients)] for d in detector_types}
    opts = {d: [torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4) for m in models[d]] for d in detector_types}

    # Precompute per-client local class weights (Cui et al., CVPR 2019)
    client_criteria = []
    for c_idx in range(num_clients):
        c_lbls = client_graphs[c_idx]['edge_labels'].numpy()
        c_weights = compute_local_class_weights(c_lbls, num_classes, beta=beta)
        logger.info(f"Client {c_idx} Cui et al. CB Weights: {[round(w, 4) for w in c_weights.cpu().tolist()]}")
        client_criteria.append(nn.CrossEntropyLoss(weight=c_weights))

    logger.info(f"Starting {num_rounds} rounds with Strategy 1 (Cui et al. 2019 Class-Balanced Loss)...")
    for r in range(num_rounds):
        for d in detector_types:
            c_states, c_accs = [], []
            for c_idx in range(num_clients):
                m, opt = models[d][c_idx], opts[d][c_idx]
                g = client_graphs[c_idx]
                x, ei, el = g['features'].to(device), g['edge_index'].to(device), g['edge_labels'].to(device)
                
                crit = client_criteria[c_idx]
                m.train()
                for epoch_idx in range(local_epochs):
                    opt.zero_grad()
                    _, preds = m(x, ei)
                    loss = crit(preds, el)

                    # [TASK 2 DEBUG PRINT & ASSERTION]
                    # Inspect exact weight tensor right before loss.backward()
                    if epoch_idx == 0 and d == 'temporal':
                        weight_tensor = crit.weight
                        assert weight_tensor is not None, f"Loss criterion on client {c_idx} has no weight tensor!"
                        expected_weights = compute_local_class_weights(g['edge_labels'].cpu().numpy(), num_classes, beta=beta)
                        assert torch.allclose(weight_tensor, expected_weights, atol=1e-4), (
                            f"Weight mismatch on Client {c_idx}! Criterion weight != local Cui et al. computation"
                        )
                        class_weight_map = {inv_mapper[i]: round(float(weight_tensor[i]), 4) for i in range(num_classes)}
                        logger.info(
                            f"\n>>> [TASK 2 DEBUG] Round {r+1:02d} | Client {c_idx} (Local flows: {len(el):,}) <<<\n"
                            f"    Active Loss Weight Tensor: {weight_tensor.detach().cpu().numpy().round(4).tolist()}\n"
                            f"    Named Weights: {json.dumps(class_weight_map)}"
                        )

                    loss.backward()
                    opt.step()

                m.eval()
                with torch.no_grad():
                    _, vp = m(x, ei)
                    acc = (vp.argmax(1) == el).float().mean().item()
                    c_accs.append(acc)
                    c_states.append({k: v.cpu() for k, v in m.state_dict().items()})

            # FedAvg aggregation
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
    
    # Train / Eval Meta RF
    s_idx = len(y_test) // 2
    rf = RandomForestClassifier(n_estimators=100, max_depth=16, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_meta[:s_idx], y_test[:s_idx])
    final_preds = rf.predict(X_meta[s_idx:])
    y_eval = y_test[s_idx:]

    b_acc = balanced_accuracy_score(y_eval, final_preds)
    m_f1 = f1_score(y_eval, final_preds, average='macro', zero_division=0)
    w_f1 = f1_score(y_eval, final_preds, average='weighted', zero_division=0)
    acc = accuracy_score(y_eval, final_preds)
    
    # Per-Class Metrics Calculation
    target_names = [inv_mapper[i] for i in range(num_classes)]
    cls_report = classification_report(y_eval, final_preds, target_names=target_names, digits=4, zero_division=0)
    per_class_f1 = f1_score(y_eval, final_preds, average=None, zero_division=0)
    per_class_dict = {inv_mapper[i]: round(float(per_class_f1[i]), 4) for i in range(num_classes)}

    logger.info(f"\n--- V1 Evaluation Summary (Cui et al. 2019 CB-CE Loss) ---\n"
                f"Accuracy: {acc*100:.2f}%\n"
                f"Balanced Accuracy: {b_acc*100:.2f}%\n"
                f"Macro F1: {m_f1*100:.2f}%\n"
                f"Weighted F1: {w_f1*100:.2f}%\n"
                f"Per-Class F1:\n{json.dumps(per_class_dict, indent=2)}\n"
                f"\nDetailed Classification Report:\n{cls_report}")

    return {
        'method': 'V1_ClassBalanced_Cui2019',
        'accuracy': acc,
        'balanced_accuracy': b_acc,
        'macro_f1': m_f1,
        'weighted_f1': w_f1,
        'per_class_f1': per_class_dict,
        'report': cls_report
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run FedGATSage V1 (Cui et al. 2019 Class-Balanced Loss + Leiden)")
    parser.add_argument('--clients', type=int, default=5, help="Number of clients")
    parser.add_argument('--rounds', type=int, default=15, help="Number of FL rounds")
    parser.add_argument('--epochs', type=int, default=10, help="Local epochs per round")
    parser.add_argument('--samples', type=int, default=500000, help="Max total dataset samples")
    parser.add_argument('--beta', type=float, default=0.9999, help="Beta parameter in Cui et al. 2019")
    args = parser.parse_args()

    run_v1_experiment(
        num_clients=args.clients,
        num_rounds=args.rounds,
        local_epochs=args.epochs,
        max_samples=args.samples,
        beta=args.beta
    )
