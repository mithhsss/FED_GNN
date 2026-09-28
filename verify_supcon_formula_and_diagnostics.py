import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold

PAPER_8_CLASSES = ['Benign', 'Backdoor', 'DDoS', 'DoS', 'Injection', 'Password', 'Scanning', 'XSS']
CLASS_MAP = {c.lower(): i for i, c in enumerate(PAPER_8_CLASSES)}
INV_CLASS_MAP = {i: c for i, c in enumerate(PAPER_8_CLASSES)}

# ==============================================================================
# KHOSLA ET AL. (NEURIPS 2020) SUPERVISED CONTRASTIVE LOSS (L_out^sup)
# ==============================================================================
class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (L_out^sup) from:
    Khosla et al., 'Supervised Contrastive Learning', NeurIPS 2020.
    
    Formula:
      L_out_i = (-1 / |P(i)|) * sum_{p in P(i)} log( exp(z_i . z_p / tau) / sum_{a in A(i)} exp(z_i . z_a / tau) )
      
    Where:
      - z_i = L2-normalized embedding: z_i = h_i / ||h_i||_2
      - A(i) = all other samples in current batch (excluding i)
      - P(i) = {p in A(i) : y_p == y_i} (positives sharing label y_i)
      - |P(i)| = cardinality of P(i)
      - tau = temperature hyperparameter
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: [B, D] raw or projected embeddings
        labels: [B] class labels
        """
        device = features.device
        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 1. L2 Normalize Embeddings: z_i = h_i / ||h_i||_2
        z = F.normalize(features, p=2, dim=1)

        # 2. Cosine Similarity Matrix: sim_{i, j} = z_i . z_j / tau
        sim = torch.matmul(z, z.T) / self.temperature

        # For numerical stability: subtract row-wise max
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # 3. Mask Construction
        # A(i): all indices except i (diagonal is 0)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        denom_mask = ~self_mask # 1 for all a in A(i)

        # P(i): all j != i where labels[j] == labels[i]
        label_match = torch.eq(labels.view(-1, 1), labels.view(1, -1))
        pos_mask = label_match & denom_mask # [B, B] boolean mask

        # Card of P(i)
        card_p = pos_mask.sum(dim=1) # [B]

        # Valid anchors: anchors with at least 1 positive in the batch (|P(i)| >= 1)
        valid_anchors = card_p > 0
        if not valid_anchors.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 4. Denominator: sum_{a in A(i)} exp(z_i . z_a / tau)
        # We mask self-contrast by setting diagonal entries in denominator to a very large negative value
        sim_denom = sim.masked_fill(self_mask, -1e9)
        log_denom = torch.logsumexp(sim_denom, dim=1, keepdim=True) # [B, 1]

        # 5. Log-probability for all pairs (i, p):
        # log( exp(sim_{i,p}) / sum_a exp(sim_{i,a}) ) = sim_{i,p} - log_denom_i
        log_prob = sim - log_denom # [B, B]

        # 6. Outer summation for L_out_i:
        # (-1 / |P(i)|) * sum_{p in P(i)} log_prob_{i, p}
        pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1) # [B]
        loss_per_anchor = - pos_log_prob[valid_anchors] / card_p[valid_anchors].float()

        # Mean over all valid anchors
        return loss_per_anchor.mean()

# ==============================================================================
# UNIT TEST FOR SUPCON FORMULA
# ==============================================================================
def unit_test_supcon():
    print("=" * 80)
    print("UNIT TEST: VERIFYING KHOSLA ET AL. L_out^sup FORMULATION")
    print("=" * 80)
    criterion = SupervisedContrastiveLoss(temperature=0.1)

    # Synthetic batch: 4 samples, 2 classes (labels: [0, 0, 1, 1])
    # Let samples 0 and 1 be identical (dot product = 1.0)
    # Let samples 2 and 3 be identical (dot product = 1.0)
    # Let class 0 and class 1 be orthogonal (dot product = 0.0)
    feats = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0]
    ], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    loss = criterion(feats, labels)
    
    # Manual calculation:
    # For sample 0:
    #   z_0 . z_1 = 1.0 / 0.1 = 10.0
    #   z_0 . z_2 = 0.0 / 0.1 = 0.0
    #   z_0 . z_3 = 0.0 / 0.1 = 0.0
    # Denominator sum_{a in A(0)} = exp(10) + exp(0) + exp(0) = e^10 + 2
    # Numerator for positive p=1: exp(10)
    # log( exp(10) / (exp(10) + 2) ) = 10 - log(exp(10) + 2)
    # loss_0 = - (10 - log(exp(10) + 2)) = log(1 + 2*e^-10) ≈ 2 * e^-10 ≈ 9.08e-5
    expected_manual = - (10.0 - np.log(np.exp(10.0) + 2.0))
    print(f"  PyTorch SupCon Loss: {loss.item():.8f}")
    print(f"  Manual Exact Eq. 2:  {expected_manual:.8f}")
    diff = abs(loss.item() - expected_manual)
    print(f"  Absolute Difference: {diff:.2e}")
    assert diff < 1e-6, "Unit test failed! Math does not match Eq. 2!"
    print("  --> UNIT TEST PASSED: Formula matches Khosla et al. L_out^sup exactly.\n")

# ==============================================================================
# BATCH-LEVEL POSITIVE-COUNT DIAGNOSTICS
# ==============================================================================
def run_batch_diagnostics():
    print("=" * 80)
    print("CHECK 4: BATCH-LEVEL POSITIVE-COUNT DIAGNOSTICS ON GROUPED DATASET")
    print("=" * 80)
    
    data_path = 'data/nftoniot/NF-ToN-IoT.parquet'
    if not os.path.exists(data_path):
        data_path = 'data/nf_ton_unzipped/7ca78ae35fa4961a_MOHANAD_A4706/data/NF-ToN-IoT.csv'
        
    df = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)
    df['attack_lower'] = df['Attack'].astype(str).str.lower()
    df = df[df['attack_lower'].isin(CLASS_MAP)].copy().reset_index(drop=True)
    df['label_idx'] = df['attack_lower'].map(CLASS_MAP)

    hash_cols = ['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'FLOW_DURATION_MILLISECONDS', 
                 'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS']
    # Check if raw columns or mapped columns
    cols = [c for c in hash_cols if c in df.columns]
    df['flow_signature'] = df[cols].astype(str).agg('|'.join, axis=1)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, _ = next(sgkf.split(df, df['label_idx'], groups=df['flow_signature']))
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)

    print(f"Grouped Training Set: {len(train_df):,} flows")
    print("Class Distribution in Grouped Training Set:")
    for cls in PAPER_8_CLASSES:
        cnt = (train_df['label_idx'] == CLASS_MAP[cls.lower()]).sum()
        pct = cnt / len(train_df) * 100
        print(f"  {cls:<12}: {cnt:>10,} ({pct:>5.2f}%)")

    # Client-level breakdown (e.g. 1 client out of K=5 has ~185,000 flows)
    client_train = train_df.sample(len(train_df)//5, random_state=42).reset_index(drop=True)
    y_client = client_train['label_idx'].values
    N_client = len(y_client)

    # 1. UNIFORM RANDOM BATCH SAMPLING
    print("\n--- Diagnostic 1: Uniform Random Sampling (Batch sizes: 512, 1024, 2048) ---")
    for B in [512, 1024, 2048]:
        np.random.seed(42)
        n_trials = 100
        pos_counts = {cls: [] for cls in PAPER_8_CLASSES}
        zero_pos_rate = {cls: 0 for cls in PAPER_8_CLASSES}

        for _ in range(n_trials):
            batch_idx = np.random.choice(N_client, size=B, replace=False)
            batch_y = y_client[batch_idx]
            for c_name in PAPER_8_CLASSES:
                c_idx = CLASS_MAP[c_name.lower()]
                n_c = np.sum(batch_y == c_idx)
                # |P(i)| for a sample of class c is n_c - 1
                card_p = max(0, n_c - 1)
                pos_counts[c_name].append(card_p)
                if card_p == 0:
                    zero_pos_rate[c_name] += 1

        print(f"\nBatch Size B={B} (Average over {n_trials} batches):")
        print(f"{'Class':<12} {'Mean |P(i)|':>15} {'Batches with |P(i)|=0':>25}")
        print("-" * 55)
        for cls in PAPER_8_CLASSES:
            mean_p = np.mean(pos_counts[cls])
            zero_pct = (zero_pos_rate[cls] / n_trials) * 100
            flag = " [!] ZERO POSITIVES FREQUENT" if zero_pct > 20 else ""
            print(f"{cls:<12} {mean_p:>15.1f} {zero_pct:>23.1f}%{flag}")

    # 2. CLASS-BALANCED BATCH SAMPLING
    print("\n--- Diagnostic 2: Class-Balanced Batch Sampling ---")
    print("Sample exactly M edges per class per batch (Total Batch Size = 8 * M)")
    for M in [32, 64, 128]:
        total_B = 8 * M
        print(f"\nConfig: M={M} per class -> Total Batch B={total_B}")
        print(f"{'Class':<12} {'Guaranteed |P(i)|':>20} {'Batches with |P(i)|=0':>25}")
        print("-" * 60)
        for cls in PAPER_8_CLASSES:
            # Under class-balanced sampling, every class has M samples in EVERY batch
            # So for any anchor i of class c, |P(i)| = M - 1 ALWAYS.
            card_p = M - 1
            print(f"{cls:<12} {card_p:>20} {'0.0% (Guaranteed)':>25}")

if __name__ == '__main__':
    unit_test_supcon()
    run_batch_diagnostics()
