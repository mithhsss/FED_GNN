import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

PAPER_8_CLASSES = ['Benign', 'Backdoor', 'DDoS', 'DoS', 'Injection', 'Password', 'Scanning', 'XSS']
CLASS_MAP = {c.lower(): i for i, c in enumerate(PAPER_8_CLASSES)}

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

    flags = df['TCP_FLAGS'].fillna(0).astype(int)
    df['FIN Flag Cnt'] = ((flags & 0x01) > 0).astype(float)
    df['SYN Flag Cnt'] = ((flags & 0x02) > 0).astype(float)
    df['RST Flag Cnt'] = ((flags & 0x04) > 0).astype(float)
    df['PSH Flag Cnt'] = ((flags & 0x08) > 0).astype(float)
    df['ACK Flag Cnt'] = ((flags & 0x10) > 0).astype(float)
    df['URG Flag Cnt'] = ((flags & 0x20) > 0).astype(float)

    dur_sec = df['Flow Duration'].fillna(0) / 1000.0 + 1e-6
    df['Flow Pkts/s'] = (df['Tot Fwd Pkts'] + df['Tot Bwd Pkts']) / dur_sec
    df['Flow Bytes/s'] = (df['TotLen Fwd Pkts'] + df['TotLen Bwd Pkts']) / dur_sec
    return df

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

def run_ip_disjoint_experiment():
    print("=" * 90)
    print("CHECK 2: IP SHORTCUT TEST (IP-Disjoint Split vs Signature-Grouped Split)")
    print("=" * 90)

    official_path = 'data/nf_ton_unzipped/7ca78ae35fa4961a_MOHANAD_A4706/data/NF-ToN-IoT.csv'
    df = pd.read_csv(official_path)
    df = map_nf_toniot_columns(df)
    df['attack_lower'] = df['Attack'].astype(str).str.lower()
    df = df[df['attack_lower'].isin(CLASS_MAP)].copy().reset_index(drop=True)
    df['label_idx'] = df['attack_lower'].map(CLASS_MAP)

    feat_cols = ['Src Port', 'Dst Port', 'Protocol', 'L7_PROTO', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
                 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'Flow Duration', 'FIN Flag Cnt', 'SYN Flag Cnt',
                 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'Flow Pkts/s', 'Flow Bytes/s']

    # 1. Automated GroupShuffleSplit on IPV4_SRC_ADDR
    print("\n--- 2b-1: Automated GroupShuffleSplit(test_size=0.20, random_state=42) on Src IP ---")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_gss, te_gss = next(gss.split(df, groups=df['Src IP']))
    print(f"Train Flows: {len(tr_gss):,} | Test Flows: {len(te_gss):,}")
    print("Disappeared Classes in Test (0 test flows):")
    disappeared = []
    for c in PAPER_8_CLASSES:
        n_te = (df.iloc[te_gss]['attack_lower'] == c.lower()).sum()
        if n_te == 0:
            disappeared.append(c)
            print(f"  [!] {c:<12}: 0 test flows (100% of flows were in train)")
    print(f"Total Disappeared Classes: {len(disappeared)} / 8 ({', '.join(disappeared)})")

    # 2. Best-Effort IP-Disjoint Split (stratified without sharing source IPs)
    print("\n--- 2b-2: Best-Effort Stratified IP-Disjoint Split (0 Shared Source IPs) ---")
    test_candidate_ips = {'192.168.1.31', '192.168.1.35', '192.168.1.37'}
    benign_ips = df[df['attack_lower'] == 'benign']['Src IP'].unique()
    for b_ip in benign_ips[:35]:
        if b_ip not in {'192.168.1.30', '192.168.1.33', '192.168.1.36', '192.168.1.38', '192.168.1.193'}:
            test_candidate_ips.add(b_ip)

    te_mask = df['Src IP'].isin(test_candidate_ips)
    tr_mask = ~te_mask

    ip_tr = df[tr_mask].copy().reset_index(drop=True)
    ip_te = df[te_mask].copy().reset_index(drop=True)
    shared_src_ips = len(set(ip_tr['Src IP']).intersection(set(ip_te['Src IP'])))
    print(f"Train flows: {len(ip_tr):,} | Test flows: {len(ip_te):,} | Shared Source IPs: {shared_src_ips}")
    print(f"{'Class':<12} {'Train Flows':>14} {'Test Flows':>14}")
    print("-" * 45)
    for c in PAPER_8_CLASSES:
        n_tr = (ip_tr['attack_lower'] == c.lower()).sum()
        n_te = (ip_te['attack_lower'] == c.lower()).sum()
        print(f"{c:<12} {n_tr:>14,} {n_te:>14,}")

    # Evaluate Anchor and mu=0.1 on IP-Disjoint Split with Flow Features
    from sklearn.ensemble import RandomForestClassifier
    idx_sample = np.random.RandomState(42).choice(len(ip_tr), size=min(200000, len(ip_tr)), replace=False)
    X_train = ip_tr.loc[idx_sample, feat_cols].fillna(0).values
    y_train = ip_tr.loc[idx_sample, 'label_idx'].values
    X_test = ip_te[feat_cols].fillna(0).values
    y_test = ip_te['label_idx'].values

    # Variant A: Standard Flow Features
    rf_std = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_std.fit(X_train, y_train)
    p_std = rf_std.predict(X_test)
    rep_std = classification_report(y_test, p_std, labels=list(range(8)), target_names=PAPER_8_CLASSES, output_dict=True, zero_division=0)

    print("\n--- 2c: Model Performance on IP-Disjoint Split (Flow-Level Features) ---")
    print(f"Overall Accuracy:  {rep_std['accuracy']*100:.2f}%")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, p_std)*100:.2f}%")
    print(f"Macro F1 Score:    {f1_score(y_test, p_std, average='macro', zero_division=0)*100:.2f}%")
    print(f"{'Class':<12} {'Test Support':>14} {'Recall':>12} {'FNR':>12} {'F1-Score':>12}")
    print("-" * 65)
    for c in ['XSS', 'Scanning', 'DoS', 'Backdoor']:
        rec = rep_std[c]['recall'] * 100
        fnr = 100.0 - rec
        f1 = rep_std[c]['f1-score'] * 100
        supp = rep_std[c]['support']
        print(f"{c:<12} {supp:>14,} {rec:>11.2f}% {fnr:>11.2f}% {f1:>11.2f}%")

if __name__ == '__main__':
    run_ip_disjoint_experiment()
