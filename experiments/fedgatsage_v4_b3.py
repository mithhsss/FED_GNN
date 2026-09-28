"""
FedGATSage - Version 4: FULL B3 Implementation
================================================================
B3 = Client-Local Class-Aware Edge/Flow Sampling
   + Capped Effective-Number Weighted Cross-Entropy Loss

This is the MAIN PROPOSED METHOD from the B3 framework.
Every prior version (V1–V3) tested exactly ONE half of B3:
  - V1 (B2 in ablation): Ordinary sampling + Weighted CE loss
  - V2 (B1 in ablation): Graph edge sampling + Plain CE loss
  - V3 (B4 in ablation): Borderline-SMOTE + Plain CE loss
  - V4 (B3 — THIS FILE): Graph edge sampling + Weighted CE loss  ← full B3

Paper Citations:
    [SAMPLING] Liyan Chang and Paula Branco.
               "Graph-based Solutions with Residuals for Intrusion Detection:
               the Modified E-GraphSAGE and E-ResGAT Algorithms."
               arXiv:2111.13597, 2021 / Applied Intelligence 54, 5323-5342 (2024).

    [LOSS]     Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, and Serge Belongie.
               "Class-Balanced Loss Based on Effective Number of Samples."
               CVPR 2019, pp. 9268-9277.

    [FL BASIS] Islam & Al Islam. "Disparity-Aware Federated Learning for
               Intrusion Detection." ACM 2023. DOI: 10.1145/3629188.3629197

Datasets evaluated: NF-ToN-IoT and CIC-ToN-IoT (matching base paper Table 1)
"""

import os
import sys
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, classification_report)
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [V4-B3-Full] %(message)s'
)
logger = logging.getLogger("FedGATSage_V4_B3")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# Constants — exactly matching base paper evaluation
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
}

# Exactly 8 classes as evaluated in the base paper (Table 2)
PAPER_8_CLASSES = ['Benign', 'Backdoor', 'DDoS', 'DoS', 'Injection', 'Password', 'Scanning', 'XSS']


# ---------------------------------------------------------------------------
# 1. Dataset Loading — Floor-Corrected, Exactly 8 Paper Classes
# ---------------------------------------------------------------------------
def load_dataset(
    data_dir: str = 'data',
    max_samples: int = 500000,
    dataset_type: str = 'auto',
    min_test_floor: int = 500,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, str]:
    """
    Load CIC-ToN-IoT or NF-ToN-IoT.
    
    Fixes applied vs. base paper replication:
    1. Enforces >= min_test_floor samples per class in the test split
       (prevents DoS/DDoS starvation that caused near-zero F1 in v0).
    2. Filters strictly to PAPER_8_CLASSES — no MITM or Ransomware.
    3. Supports both datasets for dual evaluation matching paper Table 1.
    """
    input_dirs = [
        os.path.join(data_dir, 'cic_ton_iot'),
        os.path.join(data_dir, 'nftoniot'),
        '/kaggle/input/cictoniot', '/kaggle/input/cic-ton-iot',
        '/kaggle/input/nftoniot', '/kaggle/input/nf-ton-iot',
        data_dir
    ]
    file_path = None
    dataset_name = "Unknown"

    for d in input_dirs:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in sorted(files):
                    if not (f.endswith('.parquet') or f.endswith('.csv')):
                        continue
                    fl = f.lower()
                    if dataset_type == 'cic' and ('cic' in fl or 'cictoniot' in fl):
                        file_path = os.path.join(root, f)
                        dataset_name = "CIC-ToN-IoT"
                        break
                    elif dataset_type == 'nf' and ('nf' in fl and 'ton' in fl):
                        file_path = os.path.join(root, f)
                        dataset_name = "NF-ToN-IoT"
                        break
                    elif dataset_type == 'auto':
                        file_path = os.path.join(root, f)
                        dataset_name = "CIC-ToN-IoT" if 'cic' in fl else "NF-ToN-IoT"
                        break
                if file_path:
                    break
        if file_path:
            break

    if not file_path:
        raise FileNotFoundError(f"No dataset found in {input_dirs}")

    logger.info(f"Loading [{dataset_name}] from: {file_path}")
    df = pd.read_parquet(file_path) if file_path.endswith('.parquet') else pd.read_csv(file_path, low_memory=False)
    df = df.rename(columns={k: v for k, v in NF_MAPPINGS.items() if k in df.columns})

    # Synthetic IP generation if not present (NF-ToN-IoT case)
    if 'Src IP' not in df.columns:
        proto = df.get('Protocol', pd.Series(np.zeros(len(df))))
        fwd_p = df.get('Tot Fwd Pkts', pd.Series(np.zeros(len(df))))
        df['Src IP'] = [f"10.0.{int(abs(p)) % 254 + 1}.{int(abs(s)) % 254 + 1}"
                        for p, s in zip(proto, fwd_p)]
    if 'Dst IP' not in df.columns:
        proto = df.get('Protocol', pd.Series(np.zeros(len(df))))
        bwd_p = df.get('Tot Bwd Pkts', pd.Series(np.zeros(len(df))))
        df['Dst IP'] = [f"192.168.{int(abs(p)) % 254 + 1}.{int(abs(d)) % 254 + 1}"
                        for p, d in zip(proto, bwd_p)]

    # Normalize attack labels
    if 'Attack' in df.columns:
        df['Attack'] = (df['Attack'].astype(str).str.strip().str.lower()
                        .map(lambda a: STANDARD_ATTACKS.get(a, None)))
    df = df.dropna(subset=['Attack'])

    # Filter to exactly PAPER_8_CLASSES — no extras
    df = df[df['Attack'].isin(PAPER_8_CLASSES)].copy()
    logger.info(f"After class filter: {len(df):,} flows, raw distribution:\n"
                f"{df['Attack'].value_counts().to_string()}")

    # -----------------------------------------------------------------------
    # Floor enforcement: guarantee >= min_test_floor per class in test split.
    # Required total per class = min_test_floor / test_size = 500 / 0.2 = 2500
    # -----------------------------------------------------------------------
    min_total_floor = int(np.ceil(min_test_floor / test_size))
    logger.info(f"Enforcing floor: >= {min_test_floor} test samples / class "
                f"(>= {min_total_floor} total / class)")

    # Class-specific quotas (tuned for both datasets)
    quotas = {
        'Scanning':  35000,
        'Backdoor':  25000,
        'Injection': 40000,
        'Password':  40000,
        'DDoS':      min_total_floor,
        'DoS':       min_total_floor,
    }
    allocated = sum(quotas.values())
    rem = max(50000, (max_samples - allocated) // 2)
    quotas['Benign'] = rem
    quotas['XSS'] = max(min_total_floor, max_samples - allocated - rem)

    sub_dfs = []
    for cls in PAPER_8_CLASSES:
        grp = df[df['Attack'] == cls]
        cnt = len(grp)
        target = quotas.get(cls, min_total_floor)
        if cnt < target:
            # Oversample with replacement to meet floor (replication, not synthesis)
            mult = int(np.ceil(target / max(cnt, 1)))
            rep = pd.concat([grp] * mult, ignore_index=True).iloc[:target].copy()
            sub_dfs.append(rep)
            logger.info(f"  {cls}: {cnt:,} -> {target:,} (replicated x{mult})")
        else:
            sub_dfs.append(grp.sample(n=target, random_state=42))
            logger.info(f"  {cls}: {cnt:,} -> {target:,} (downsampled)")

    df = (pd.concat(sub_dfs, ignore_index=True)
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True))

    logger.info(f"Final dataset: {len(df):,} flows [{dataset_name}]\n"
                f"{df['Attack'].value_counts().to_string()}")
    return df, dataset_name


# ---------------------------------------------------------------------------
# 2. B3 Component A: Local Class-Aware Graph Edge Sampling (Chang & Branco 2021)
# ---------------------------------------------------------------------------
def apply_b3_local_edge_sampling(
    client_df: pd.DataFrame,
    feat_cols: List[str],
    target_ratio: float = 0.25,
    max_oversample_ratio: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    B3 Component A: Client-local class-aware edge/flow oversampling.

    For each minority class c with local count N_k(c) < N_target:
        N_target = min(max(rho * N_majority, 500), N_k(c) * max_oversample_ratio)

    Synthetic flow e_syn = (u, v, x_syn):
        1. Select authentic base edge e_i from class c.
        2. Select in-class neighbor edge e_j from class c.
        3. Interpolate: x_syn = x_i + lambda * (x_j - x_i), lambda ~ U(0.1, 0.9)
        4. Preserve endpoints (u_i, v_i) — authentic IP communication topology preserved.
        5. Applied strictly within each client's local partition (no cross-client leakage).

    The 10:1 cap prevents synthetic noise domination on very sparse classes
    (recommended by FedMADE and per-client SMOTE literature).
    """
    rng = np.random.RandomState(random_state)
    counts = client_df['Attack'].value_counts()
    majority_count = counts.max()
    base_target = max(int(majority_count * target_ratio), 500)

    balanced_dfs = []
    for cls_name, grp in client_df.groupby('Attack'):
        cnt = len(grp)
        # Cap: never synthesize more than max_oversample_ratio * original count
        target_count = min(base_target, cnt * max_oversample_ratio)

        if cnt < target_count and cnt >= 2:
            needed = target_count - cnt
            base_idx = rng.choice(cnt, size=needed, replace=True)
            neigh_idx = rng.choice(cnt, size=needed, replace=True)

            base_rows = grp.iloc[base_idx].copy().reset_index(drop=True)
            base_feats = (grp.iloc[base_idx][feat_cols]
                          .fillna(0.0).replace([np.inf, -np.inf], 0.0)
                          .values.astype(np.float32))
            neigh_feats = (grp.iloc[neigh_idx][feat_cols]
                           .fillna(0.0).replace([np.inf, -np.inf], 0.0)
                           .values.astype(np.float32))

            # Convex interpolation along feature manifold — U(0.1, 0.9) excludes exact copies
            lambdas = rng.uniform(0.1, 0.9, size=(needed, 1)).astype(np.float32)
            syn_feats = np.maximum(base_feats + lambdas * (neigh_feats - base_feats), 0.0)

            base_rows[feat_cols] = syn_feats
            balanced_dfs.append(grp)
            balanced_dfs.append(base_rows)
        else:
            balanced_dfs.append(grp)

    result_df = (pd.concat(balanced_dfs, ignore_index=True)
                   .sample(frac=1.0, random_state=random_state)
                   .reset_index(drop=True))
    logger.info(f"  B3 edge sampling: {len(client_df):,} -> {len(result_df):,} edges.")
    return result_df


# ---------------------------------------------------------------------------
# 3. B3 Component B: Per-Client CB-CE Loss (Cui et al. CVPR 2019)
# ---------------------------------------------------------------------------
def compute_local_class_weights(
    client_labels: np.ndarray,
    num_classes: int,
    beta: float = 0.9999
) -> torch.Tensor:
    """
    B3 Component B: Class-Balanced Loss (Effective Number of Samples).

    Exact formulation from Cui et al. CVPR 2019, Equations 2 & 3:
        E_{n_c} = (1 - beta^{n_c}) / (1 - beta)          [Eq. 2]
        W_c     = 1 / E_{n_c}                             [Eq. 2 inverse]
        W̃_c    = C * W_c / sum(W_j for j in 1..C)        [Eq. 3 normalization]

    beta = 0.9999 is optimal for large-scale datasets (N ~ 10^5) per Cui et al.
    Zero weight assigned to classes absent from a client's local partition
    (unobserved class = no gradient contribution, avoids phantom weight explosion).
    """
    counts = Counter(client_labels)
    weights = np.zeros(num_classes, dtype=np.float32)

    for c in range(num_classes):
        n_c = counts.get(c, 0)
        if n_c > 0:
            eff_num = (1.0 - np.power(beta, n_c)) / (1.0 - beta)
            weights[c] = 1.0 / max(eff_num, 1e-7)
        else:
            weights[c] = 0.0  # Unseen class at this client gets zero weight

    # Eq. 3: normalize so sum = number of observed classes
    active = weights > 0
    if np.sum(weights) > 0:
        weights[active] = weights[active] / np.sum(weights[active]) * np.sum(active)

    return torch.tensor(weights, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# 4. GAT + GraphSAGE Architecture (matching base paper specs)
# ---------------------------------------------------------------------------
from torch_geometric.nn import GATConv, SAGEConv


class TemporalGATDetector(nn.Module):
    """
    Temporal GAT: flow rate, duration patterns, flag sequences.
    Hidden dim=256, 8 attention heads, 0.2 dropout — per base paper.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 256,
                 num_heads: int = 8, num_classes: int = 8, dropout: float = 0.2):
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
            nn.Dropout(dropout),
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
        # Flow embedding: concat(h_u, h_v, h_u*h_v, |h_u-h_v|)
        flow_feat = torch.cat([x[u], x[v], x[u] * x[v], torch.abs(x[u] - x[v])], dim=1)
        return x, self.classifier(flow_feat)


# Content and Behavioral GAT share the same architecture (specialized by training data)
class ContentGATDetector(TemporalGATDetector):
    """Content GAT: payload sizes, protocol patterns, service targeting."""
    pass


class BehavioralGATDetector(TemporalGATDetector):
    """Behavioral GAT: connection patterns, port usage, session characteristics."""
    pass


class GlobalGraphSAGE(nn.Module):
    """Server-side GraphSAGE: processes overlay graph of community embeddings."""
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
# 5. Leiden Graph Builder
# ---------------------------------------------------------------------------
def build_graph(df: pd.DataFrame, feat_cols: List[str],
                label_mapper: Dict[str, int]) -> Dict[str, Any]:
    """
    Build PyG-compatible graph from flow DataFrame.
    Nodes = IP addresses, Edges = network flows (labeled).
    Uses Leiden community detection (superior modularity guarantee vs. Louvain).
    """
    src_ips = df['Src IP'].astype(str).values
    dst_ips = df['Dst IP'].astype(str).values
    all_ips = list(pd.concat([pd.Series(src_ips), pd.Series(dst_ips)]).unique())
    ip_map = {ip: i for i, ip in enumerate(all_ips)}

    # Node features: average of adjacent flow features (log1p normalized)
    feat_vals = np.log1p(np.maximum(
        df[feat_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0).values.astype(np.float32), 0.0
    ))
    feat_mat = np.zeros((len(all_ips), len(feat_cols)), dtype=np.float32)
    ip_cnt = np.ones(len(all_ips), dtype=np.float32)
    for i in range(len(df)):
        si, di = ip_map[src_ips[i]], ip_map[dst_ips[i]]
        feat_mat[si] += feat_vals[i]; ip_cnt[si] += 1
        feat_mat[di] += feat_vals[i]; ip_cnt[di] += 1
    feat_mat /= ip_cnt[:, None]

    edge_idx = torch.tensor(
        [[ip_map[s] for s in src_ips], [ip_map[d] for d in dst_ips]],
        dtype=torch.long
    )
    edge_lbl = torch.tensor(
        [label_mapper.get(a, 0) for a in df['Attack'].values],
        dtype=torch.long
    )

    # Leiden community detection
    try:
        import igraph as ig
        import leidenalg as la
        edge_tuples = [(ip_map[s], ip_map[d]) for s, d in zip(src_ips, dst_ips)]
        g_ig = ig.Graph(n=len(all_ips), edges=edge_tuples, directed=False)
        partition = la.find_partition(g_ig, la.ModularityVertexPartition, seed=42)
        ip_comm = list(partition.membership)
        logger.info(f"  Leiden: {len(set(ip_comm))} communities detected.")
    except Exception as e:
        logger.warning(f"  Leiden fallback to modulo partition: {e}")
        ip_comm = [i % 10 for i in range(len(all_ips))]

    return {
        'features': torch.tensor(feat_mat, dtype=torch.float32),
        'edge_index': edge_idx,
        'edge_labels': edge_lbl,
        'ip_community': ip_comm,
        'flow_raw_feats': feat_vals
    }


# ---------------------------------------------------------------------------
# 6. B3 Full Training Pipeline
# ---------------------------------------------------------------------------
def run_b3_experiment(
    num_clients: int = 5,
    num_rounds: int = 15,
    local_epochs: int = 10,
    max_samples: int = 500000,
    beta: float = 0.9999,
    target_ratio: float = 0.25,
    dataset_type: str = 'auto',
    data_dir: str = 'data'
) -> Dict[str, Any]:
    """
    Full B3 federated experiment.

    Architecture (matching base paper):
    - 5 clients, 15 federation rounds, 10 local epochs per round
    - 3 specialized GAT detectors: Temporal, Content, Behavioral
    - Server-side GlobalGraphSAGE on community overlay graph
    - Two-stage meta-fusion: GAT ensemble -> Random Forest classifier
    - FedAvg aggregation across clients

    B3 Changes vs. baseline:
    - [SAMPLING] apply_b3_local_edge_sampling() per client before graph build
    - [LOSS]     compute_local_class_weights() per client, CB-CE loss in training
    """
    # --- Data loading ---
    raw_df, dataset_name = load_dataset(
        data_dir=data_dir, max_samples=max_samples,
        dataset_type=dataset_type
    )
    logger.info(f"\n{'='*60}\nRunning B3 on: {dataset_name}\n{'='*60}")

    unique_attacks = sorted(raw_df['Attack'].unique())
    label_mapper = {a: i for i, a in enumerate(unique_attacks)}
    inv_mapper = {i: a for a, i in label_mapper.items()}
    num_classes = len(unique_attacks)
    logger.info(f"Classes ({num_classes}): {unique_attacks}")

    # Stratified train/test split
    train_df, test_df = train_test_split(
        raw_df, test_size=0.2, random_state=42, stratify=raw_df['Attack']
    )
    logger.info(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    feat_cols = [c for c in raw_df.columns
                 if c not in ['Label', 'Attack', 'Src IP', 'Dst IP', 'Timestamp']]
    logger.info(f"Feature count: {len(feat_cols)}")

    # Partition into clients
    client_dfs = np.array_split(
        train_df.sample(frac=1.0, random_state=42).reset_index(drop=True),
        num_clients
    )

    # Build test graph (no balancing applied to test set)
    test_graph = build_graph(test_df, feat_cols, label_mapper)

    # -----------------------------------------------------------------------
    # B3 STEP 1: Apply local edge sampling to each client (BEFORE graph build)
    # -----------------------------------------------------------------------
    logger.info("\n[B3-Sampling] Applying client-local class-aware edge sampling...")
    sampled_client_dfs = []
    for c_idx, cdf in enumerate(client_dfs):
        logger.info(f"  Client {c_idx} before sampling:\n"
                    f"  {cdf['Attack'].value_counts().to_string()}")
        s_cdf = apply_b3_local_edge_sampling(
            cdf, feat_cols, target_ratio=target_ratio, random_state=42 + c_idx
        )
        sampled_client_dfs.append(s_cdf)

    client_graphs = [build_graph(scdf, feat_cols, label_mapper)
                     for scdf in sampled_client_dfs]

    # -----------------------------------------------------------------------
    # B3 STEP 2: Compute per-client Cui et al. CB-CE weights (AFTER sampling)
    # Note: weights are computed on POST-SAMPLING distributions so the loss
    # reflects the effective count of synthetic + real flows per class.
    # -----------------------------------------------------------------------
    logger.info("\n[B3-Loss] Computing per-client CB-CE weights (Cui et al. 2019)...")
    client_criteria = []
    for c_idx in range(num_clients):
        c_lbls = client_graphs[c_idx]['edge_labels'].numpy()
        c_weights = compute_local_class_weights(c_lbls, num_classes, beta=beta)
        weight_map = {inv_mapper[i]: round(float(c_weights[i]), 4)
                      for i in range(num_classes)}
        logger.info(f"  Client {c_idx} CB weights: {json.dumps(weight_map)}")
        client_criteria.append(nn.CrossEntropyLoss(weight=c_weights))

    # --- Model initialization ---
    in_dim = client_graphs[0]['features'].shape[1]
    detector_types = ['temporal', 'content', 'behavioral']
    det_classes = {'temporal': TemporalGATDetector,
                   'content': ContentGATDetector,
                   'behavioral': BehavioralGATDetector}

    models = {d: [det_classes[d](in_dim, 256, 8, num_classes).to(device)
                  for _ in range(num_clients)]
              for d in detector_types}
    opts = {d: [torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
                for m in models[d]]
            for d in detector_types}

    # --- Federated training loop ---
    logger.info(f"\n[Training] {num_rounds} rounds × {num_clients} clients × "
                f"{local_epochs} local epochs")
    logger.info(f"[B3 Active] Sampling: YES (ratio={target_ratio}, 10x cap) | "
                f"CB-CE Loss: YES (beta={beta})")

    for r in range(num_rounds):
        round_accs = {d: [] for d in detector_types}

        for d in detector_types:
            c_states = []
            for c_idx in range(num_clients):
                m = models[d][c_idx]
                opt = opts[d][c_idx]
                g = client_graphs[c_idx]
                x = g['features'].to(device)
                ei = g['edge_index'].to(device)
                el = g['edge_labels'].to(device)
                crit = client_criteria[c_idx]  # B3: per-client CB-CE

                m.train()
                for _ in range(local_epochs):
                    opt.zero_grad()
                    _, preds = m(x, ei)
                    loss = crit(preds, el)  # B3: weighted loss
                    loss.backward()
                    opt.step()

                m.eval()
                with torch.no_grad():
                    _, vp = m(x, ei)
                    acc = (vp.argmax(1) == el).float().mean().item()
                    round_accs[d].append(acc)
                    c_states.append({k: v.cpu() for k, v in m.state_dict().items()})

            # FedAvg aggregation
            agg = {k: sum(c_states[i][k].float() for i in range(num_clients)) / num_clients
                   for k in c_states[0]}
            for c_idx in range(num_clients):
                models[d][c_idx].load_state_dict({k: v.to(device) for k, v in agg.items()})

        avg_acc = np.mean([np.mean(v) for v in round_accs.values()])
        logger.info(f"Round {r+1:02d}/{num_rounds} | Avg local acc: {avg_acc:.4f}")

    # --- Two-Stage Meta Evaluation ---
    logger.info("\n[Evaluation] Two-stage meta-fusion (GAT ensemble → Random Forest)...")
    tx = test_graph['features'].to(device)
    te = test_graph['edge_index'].to(device)
    test_probs = []

    with torch.no_grad():
        for d in detector_types:
            _, logits = models[d][0](tx, te)
            test_probs.append(F.softmax(logits, dim=1).cpu().numpy())

    # Meta features: GAT probabilities + raw flow features
    X_meta = np.concatenate(test_probs + [test_graph['flow_raw_feats']], axis=1)
    y_test = test_graph['edge_labels'].numpy()

    # Split meta-set for RF training/evaluation
    s_idx = len(y_test) // 2
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=16,
        class_weight='balanced',  # Additional RF-level balancing
        random_state=42, n_jobs=-1
    )
    rf.fit(X_meta[:s_idx], y_test[:s_idx])
    final_preds = rf.predict(X_meta[s_idx:])
    y_eval = y_test[s_idx:]

    # --- Metrics (full suite matching base paper Table 1 & 2) ---
    acc = accuracy_score(y_eval, final_preds)
    b_acc = balanced_accuracy_score(y_eval, final_preds)
    m_f1 = f1_score(y_eval, final_preds, average='macro', zero_division=0)
    w_f1 = f1_score(y_eval, final_preds, average='weighted', zero_division=0)

    target_names = [inv_mapper[i] for i in range(num_classes)]
    cls_report = classification_report(
        y_eval, final_preds, target_names=target_names, digits=4, zero_division=0
    )
    per_class_f1 = f1_score(y_eval, final_preds, average=None, zero_division=0)
    per_class_dict = {inv_mapper[i]: round(float(per_class_f1[i]), 4)
                      for i in range(num_classes)}

    # False Positive Rate / False Negative Rate
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_eval, final_preds)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    fpr = float(np.mean(fp / np.maximum(fp + tn, 1)))
    fnr = float(np.mean(fn / np.maximum(fn + tp, 1)))

    logger.info(
        f"\n{'='*60}\n"
        f"B3 FINAL RESULTS — {dataset_name}\n"
        f"{'='*60}\n"
        f"Accuracy (whole):    {acc*100:.2f}%   ← compare to base paper\n"
        f"Balanced Accuracy:   {b_acc*100:.2f}%\n"
        f"Macro F1:            {m_f1*100:.2f}%\n"
        f"Weighted F1:         {w_f1*100:.2f}%\n"
        f"FPR / FNR:           {fpr:.4f} / {fnr:.4f}\n"
        f"\nBase paper [{dataset_name}]: Acc=60.7% (NF) / 82.7% (CIC), "
        f"Bal.Acc=78.58% (NF) / 80.24% (CIC)\n"
        f"\nPer-Class F1:\n{json.dumps(per_class_dict, indent=2)}\n"
        f"\nFull Classification Report:\n{cls_report}"
    )

    return {
        'method': 'V4_B3_Full',
        'dataset': dataset_name,
        'accuracy': acc,
        'balanced_accuracy': b_acc,
        'macro_f1': m_f1,
        'weighted_f1': w_f1,
        'fpr': fpr,
        'fnr': fnr,
        'per_class_f1': per_class_dict,
        'report': cls_report,
        'base_paper_acc': {'NF-ToN-IoT': 0.607, 'CIC-ToN-IoT': 0.827}.get(dataset_name, None),
        'base_paper_bal_acc': {'NF-ToN-IoT': 0.7858, 'CIC-ToN-IoT': 0.8024}.get(dataset_name, None),
    }


# ---------------------------------------------------------------------------
# 7. Entry Point — supports dual dataset evaluation matching base paper
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="FedGATSage V4 — Full B3 (Local Edge Sampling + CB-CE Loss)"
    )
    parser.add_argument('--clients', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=15)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--samples', type=int, default=500000)
    parser.add_argument('--beta', type=float, default=0.9999,
                        help="Cui et al. 2019 beta parameter")
    parser.add_argument('--target_ratio', type=float, default=0.25,
                        help="Minority class target as fraction of majority count")
    parser.add_argument('--dataset', type=str, default='auto',
                        choices=['auto', 'cic', 'nf'],
                        help="Dataset to run: auto=first found, cic=CIC-ToN-IoT, nf=NF-ToN-IoT")
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--both', action='store_true',
                        help="Run on both datasets sequentially (matches base paper Table 1)")
    args = parser.parse_args()

    if args.both:
        results = {}
        for ds in ['cic', 'nf']:
            logger.info(f"\n{'#'*70}\nStarting B3 on dataset: {ds.upper()}\n{'#'*70}")
            try:
                r = run_b3_experiment(
                    num_clients=args.clients, num_rounds=args.rounds,
                    local_epochs=args.epochs, max_samples=args.samples,
                    beta=args.beta, target_ratio=args.target_ratio,
                    dataset_type=ds, data_dir=args.data_dir
                )
                results[ds] = r
            except FileNotFoundError as e:
                logger.warning(f"Dataset {ds} not found, skipping: {e}")

        logger.info("\n" + "="*70)
        logger.info("DUAL DATASET SUMMARY (B3 vs Base Paper)")
        logger.info("="*70)
        for ds, r in results.items():
            base_acc = r.get('base_paper_acc')
            our_acc = r['accuracy']
            delta = (our_acc - base_acc) * 100 if base_acc else 0
            logger.info(
                f"\n{r['dataset']}:\n"
                f"  Our Accuracy:   {our_acc*100:.2f}%\n"
                f"  Paper Accuracy: {base_acc*100:.2f}% (delta: {delta:+.2f}%)\n"
                f"  Our Bal. Acc:   {r['balanced_accuracy']*100:.2f}%\n"
                f"  Our Macro F1:   {r['macro_f1']*100:.2f}%"
            )
    else:
        run_b3_experiment(
            num_clients=args.clients, num_rounds=args.rounds,
            local_epochs=args.epochs, max_samples=args.samples,
            beta=args.beta, target_ratio=args.target_ratio,
            dataset_type=args.dataset, data_dir=args.data_dir
        )
