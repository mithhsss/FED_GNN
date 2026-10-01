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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FedGATSage-MoE] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FedGATSage_MoE")

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
            df['Is_Ephemeral_Dst'] = (df['Dst Port'] >= 49152).astype(float)
            df['Total_Bytes_Log'] = np.log1p(np.maximum(df['TotLen Fwd Pkts'] + df['TotLen Bwd Pkts'], 0.0))
            df['Total_Pkts_Log'] = np.log1p(np.maximum(df['Tot Fwd Pkts'] + df['Tot Bwd Pkts'], 0.0))
        return df

def stratified_split_clients(df: pd.DataFrame, num_clients: int = 5, seed: int = 42) -> List[pd.DataFrame]:
    rng = np.random.RandomState(seed)
    client_dfs = [[] for _ in range(num_clients)]
    for c in range(8):
        c_df = df[df['label_idx'] == c]
        if len(c_df) == 0:
            continue
        indices = c_df.index.values.copy()
        rng.shuffle(indices)
        splits = np.array_split(indices, num_clients)
        for i in range(num_clients):
            client_dfs[i].append(df.loc[splits[i]])
    res = [pd.concat(client_dfs[i], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
           for i in range(num_clients)]
    return res

def b7_edge_sampling(df: pd.DataFrame, feat_cols: List[str], ratio: float = 0.50,
                     max_interp_factor: int = 4, seed: int = 42) -> pd.DataFrame:
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
                 num_classes: int = 8, dropout: float = 0.2):
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

    def forward_full(self, x: torch.Tensor, edge_index: torch.Tensor, chunk_size: int = 50000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h1 = F.elu(self.gat1(x, edge_index))
        h2 = self.gat2(h1, edge_index)
        num_edges = edge_index.size(1)
        if num_edges == 0:
            return h2, torch.empty((0, 256), device=x.device), torch.empty((0, 8), device=x.device)

        all_embs, all_logits = [], []
        for start in range(0, num_edges, chunk_size):
            end = min(start + chunk_size, num_edges)
            src = edge_index[0, start:end]
            dst = edge_index[1, start:end]
            embs = self.edge_mlp(torch.cat([h2[src], h2[dst]], dim=1))
            all_embs.append(embs)
            all_logits.append(self.classifier(embs))
        return h2, torch.cat(all_embs, dim=0), torch.cat(all_logits, dim=0)

class SparselyGatedMoE(nn.Module):
    """
    Sparsely-Gated Mixture-of-Experts Layer (Shazeer et al., ICLR 2017).
    Routes flow representations to the top-k experts with noisy gating and auxiliary load-balancing loss.
    """
    def __init__(self, input_dim: int, num_experts: int = 3, num_classes: int = 8, k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.k = k

        self.w_gate = nn.Linear(input_dim, num_experts, bias=False)
        self.w_noise = nn.Linear(input_dim, num_experts, bias=False)
        self.softplus = nn.Softplus()

    def noisy_top_k_gating(self, x: torch.Tensor, train: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clean_logits = self.w_gate(x)
        if train:
            raw_noise_std = self.w_noise(x)
            noise_std = self.softplus(raw_noise_std) + 1e-2
            noise = torch.randn_like(clean_logits) * noise_std
            noisy_logits = clean_logits + noise
        else:
            noisy_logits = clean_logits

        top_k_logits, top_k_indices = torch.topk(noisy_logits, self.k, dim=1)
        top_k_gates = F.softmax(top_k_logits, dim=1)

        zeros = torch.zeros_like(noisy_logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        # Load balancing loss:
        mask = torch.zeros_like(noisy_logits)
        mask.scatter_(1, top_k_indices, 1.0)
        f = mask.mean(dim=0)
        p = F.softmax(clean_logits, dim=1).mean(dim=0)
        loss_balance = self.num_experts * torch.sum(f * p)

        return gates, loss_balance, f

    def forward(self, expert_probs: List[torch.Tensor], router_input: torch.Tensor, train: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gates, loss_balance, expert_fractions = self.noisy_top_k_gating(router_input, train=train)
        stacked_probs = torch.stack(expert_probs, dim=1)
        g = gates.unsqueeze(2)
        fused_probs = torch.sum(g * stacked_probs, dim=1)
        return fused_probs, loss_balance, expert_fractions

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

def build_client_graph(df: pd.DataFrame, feat_cols: List[str], label_mapper: Dict[str, int]) -> Dict[str, Any]:
    all_ips = list(pd.concat([df['Src IP'], df['Dst IP']]).unique())
    ip2idx = {ip: i for i, ip in enumerate(all_ips)}
    src_idx = [ip2idx[ip] for ip in df['Src IP']]
    dst_idx = [ip2idx[ip] for ip in df['Dst IP']]

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
        'num_nodes': len(all_ips),
        'num_edges': len(df),
        'flow_raw_feats': raw_feats
    }

def run_moe_experiment_run(train_df: pd.DataFrame, test_df: pd.DataFrame,
                           config_name: str, seed: int,
                           use_cb_focal: bool, use_edge_interp: bool, mu_supcon: float,
                           include_raw_in_router: bool,
                           num_clients: int = 5, num_rounds: int = 15, local_epochs: int = 5,
                           out_dir: str = "/kaggle/working") -> Dict[str, Any]:
    t0 = time.time()
    set_seed(seed)

    logger.info(f"\n{'='*80}")
    logger.info(f"STARTING MOE RUN: Config='{config_name}' | Seed={seed}")
    logger.info(f"  CB-Focal: {use_cb_focal} | Interp: {use_edge_interp} | mu: {mu_supcon:.2f} | RawInRouter: {include_raw_in_router}")
    logger.info(f"{'='*80}")

    client_dfs = stratified_split_clients(train_df, num_clients, seed=seed)
    detector_types = ['temporal', 'content', 'behavioral']
    feature_engineers = {d: FeatureEngineer(d) for d in detector_types}

    train_dfs = {d: [feature_engineers[d].extract_features(cdf) for cdf in client_dfs] for d in detector_types}
    test_dfs = {d: feature_engineers[d].extract_features(test_df) for d in detector_types}

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

    client_graphs = {d: [build_client_graph(sampled_train_dfs[d][i], feat_cols[d], CLASS_MAP) for i in range(num_clients)]
                     for d in detector_types}
    test_graphs = {d: build_client_graph(test_dfs[d], feat_cols[d], CLASS_MAP) for d in detector_types}

    # Setup models
    client_models = {}
    for d in detector_types:
        dim = len(feat_cols[d])
        client_models[d] = [SpecializedGAT(input_dim=dim, hidden_dim=256, num_heads=8, num_classes=8, dropout=0.2).to(device)
                            for _ in range(num_clients)]

    # Gating router input dimension:
    # 3 experts * 256 embedding dimension = 768
    # plus raw flow features if include_raw_in_router
    raw_dim = len(feat_cols['temporal']) if include_raw_in_router else 0
    router_dim = 256 * 3 + raw_dim
    client_moe = [SparselyGatedMoE(input_dim=router_dim, num_experts=3, num_classes=8, k=2).to(device)
                  for _ in range(num_clients)]

    optimizers = []
    for i in range(num_clients):
        params = []
        for d in detector_types:
            params.extend(list(client_models[d][i].parameters()))
        params.extend(list(client_moe[i].parameters()))
        optimizers.append(torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4))

    cb_weights = {}
    if use_cb_focal:
        for d in detector_types:
            cb_weights[d] = [compute_cb_weights(client_graphs[d][i]['edge_labels'].numpy(), num_classes=8)
                             for i in range(num_clients)]

    supcon_criterion = SupervisedContrastiveLoss(temperature=0.07).to(device)

    # Federated Rounds
    for r in range(num_rounds):
        client_states = {d: [] for d in detector_types}
        moe_states = []

        for i in range(num_clients):
            # Local client training
            opt = optimizers[i]
            for d in detector_types:
                client_models[d][i].train()
            client_moe[i].train()

            edge_labels = client_graphs['temporal'][i]['edge_labels'].to(device)

            for epoch in range(local_epochs):
                opt.zero_grad()
                batch_idx = sample_class_balanced_batch(edge_labels, samples_per_class=64)
                b_targets = edge_labels[batch_idx]

                b_embs, b_logits, b_probs = {}, {}, []
                total_detector_loss = 0.0

                for d in detector_types:
                    g = client_graphs[d][i]
                    x = g['x'].to(device)
                    edge_index = g['edge_index'].to(device)
                    _, emb, logit = client_models[d][i].forward_batch(x, edge_index, batch_idx)
                    b_embs[d] = emb
                    b_logits[d] = logit
                    b_probs.append(F.softmax(logit, dim=1))

                    if use_cb_focal:
                        loss_cls = focal_loss(logit, b_targets, weights=cb_weights[d][i], gamma=2.0)
                    else:
                        loss_cls = F.cross_entropy(logit, b_targets)

                    if mu_supcon > 0.0:
                        loss_con = supcon_criterion(emb, b_targets)
                        total_detector_loss = total_detector_loss + loss_cls + mu_supcon * loss_con
                    else:
                        total_detector_loss = total_detector_loss + loss_cls

                # Build router input
                cat_embs = torch.cat([b_embs['temporal'], b_embs['content'], b_embs['behavioral']], dim=1)
                if include_raw_in_router:
                    raw_b = torch.tensor(client_graphs['temporal'][i]['flow_raw_feats'][batch_idx.cpu().numpy()],
                                         dtype=torch.float32, device=device)
                    router_in = torch.cat([cat_embs, raw_b], dim=1)
                else:
                    router_in = cat_embs

                fused_probs, l_balance, _ = client_moe[i](b_probs, router_in, train=True)
                l_moe_task = F.nll_loss(torch.log(fused_probs + 1e-8), b_targets)

                total_loss = total_detector_loss + l_moe_task + 0.01 * l_balance
                total_loss.backward()
                for d in detector_types:
                    torch.nn.utils.clip_grad_norm_(client_models[d][i].parameters(), max_norm=2.0)
                torch.nn.utils.clip_grad_norm_(client_moe[i].parameters(), max_norm=2.0)
                opt.step()

            # Record states for FedAvg
            for d in detector_types:
                client_states[d].append(client_models[d][i].state_dict())
            moe_states.append(client_moe[i].state_dict())

        # Server FedAvg
        total_edges = sum(client_graphs['temporal'][i]['num_edges'] for i in range(num_clients))
        client_weights = [client_graphs['temporal'][i]['num_edges'] / total_edges for i in range(num_clients)]

        for d in detector_types:
            agg_state = {}
            for k in client_states[d][0].keys():
                agg_state[k] = sum(client_weights[i] * client_states[d][i][k].float() for i in range(num_clients))
            for i in range(num_clients):
                client_models[d][i].load_state_dict({k: v.to(device) for k, v in agg_state.items()})

        agg_moe_state = {}
        for k in moe_states[0].keys():
            agg_moe_state[k] = sum(client_weights[i] * moe_states[i][k].float() for i in range(num_clients))
        for i in range(num_clients):
            client_moe[i].load_state_dict({k: v.to(device) for k, v in agg_moe_state.items()})

    # Full Evaluation on Test Set (using Client 0 aggregated model)
    eval_models = {d: client_models[d][0] for d in detector_types}
    eval_moe = client_moe[0]

    for d in detector_types:
        eval_models[d].eval()
    eval_moe.eval()

    with torch.no_grad():
        test_embs, test_probs = {}, []
        for d in detector_types:
            gx_te = test_graphs[d]['x'].to(device)
            gei_te = test_graphs[d]['edge_index'].to(device)
            _, embs_te, logits_te = eval_models[d].forward_full(gx_te, gei_te)
            test_embs[d] = embs_te
            test_probs.append(F.softmax(logits_te, dim=1))

        cat_test_embs = torch.cat([test_embs['temporal'], test_embs['content'], test_embs['behavioral']], dim=1)
        if include_raw_in_router:
            raw_te = torch.tensor(test_graphs['temporal']['flow_raw_feats'], dtype=torch.float32, device=device)
            test_router_in = torch.cat([cat_test_embs, raw_te], dim=1)
        else:
            test_router_in = cat_test_embs

        # Chunked MoE evaluation to prevent OOM
        num_test_edges = test_router_in.size(0)
        chunk_size = 50000
        all_fused_preds = []
        for start in range(0, num_test_edges, chunk_size):
            end = min(start + chunk_size, num_test_edges)
            chunk_probs = [p[start:end] for p in test_probs]
            chunk_r_in = test_router_in[start:end]
            f_probs, _, _ = eval_moe(chunk_probs, chunk_r_in, train=False)
            all_fused_preds.append(torch.argmax(f_probs, dim=1).cpu())

        y_pred = torch.cat(all_fused_preds, dim=0).numpy()
        y_test = test_graphs['temporal']['edge_labels'].numpy()

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
            'fusion_method': 'Mixture-of-Experts (MoE)',
            'use_cb_focal': use_cb_focal,
            'use_edge_interp': use_edge_interp,
            'mu_supcon': mu_supcon,
            'include_raw_in_router': include_raw_in_router,
            'k_experts': 2
        },
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'macro_f1': macro_f1,
        'per_class': per_class_table,
        'elapsed_minutes': elapsed_min
    }

    cfg_slug = config_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace('+', '_')
    fname = f"run_moe_{cfg_slug}_seed{seed}.json"
    fpath = os.path.join(out_dir, fname)

    with open(fpath, 'w') as f:
        json.dump(run_record, f, indent=2)
    logger.info(f"Saved JSON: {fpath}")

    return run_record

def main():
    logger.info("=" * 80)
    logger.info("RUNNING FEDERATED MIXTURE-OF-EXPERTS (MoE) ABLATIONS")
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

    out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "kaggle_results_multiseed"
    os.makedirs(out_dir, exist_ok=True)

    seeds = [42, 43, 44]

    # Required MoE Configs:
    # (b) Plain-CE + MoE (embeddings only)
    # (b2) Plain-CE + MoE (embeddings + raw features in router)
    # (d) SupCon (mu=0.1) + MoE (embeddings only)
    moe_tasks = [
        {"name": "b_plain_ce_moe", "use_cb_focal": False, "use_edge_interp": False, "mu_supcon": 0.0, "include_raw_in_router": False},
        {"name": "b2_plain_ce_moe_raw", "use_cb_focal": False, "use_edge_interp": False, "mu_supcon": 0.0, "include_raw_in_router": True},
        {"name": "d_supcon_mu0.1_moe", "use_cb_focal": True, "use_edge_interp": True, "mu_supcon": 0.1, "include_raw_in_router": False},
    ]

    for task in moe_tasks:
        for s in seeds:
            run_moe_experiment_run(
                train_df=sig_train_df,
                test_df=sig_test_df,
                config_name=task['name'],
                seed=s,
                use_cb_focal=task['use_cb_focal'],
                use_edge_interp=task['use_edge_interp'],
                mu_supcon=task['mu_supcon'],
                include_raw_in_router=task['include_raw_in_router'],
                out_dir=out_dir
            )

if __name__ == '__main__':
    main()
