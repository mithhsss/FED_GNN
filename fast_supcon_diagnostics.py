import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAPER_8_CLASSES = ['Benign', 'Backdoor', 'DDoS', 'DoS', 'Injection', 'Password', 'Scanning', 'XSS']
CLASS_MAP = {c.lower(): i for i, c in enumerate(PAPER_8_CLASSES)}
INV_CLASS_MAP = {i: c for i, c in enumerate(PAPER_8_CLASSES)}

# Exact class fractions in NF-ToN-IoT (verified from 1,156,564 flows)
CLASS_FRACTIONS = {
    'Injection': 0.3984,
    'Benign': 0.1716,
    'DDoS': 0.1709,
    'Password': 0.1252,
    'XSS': 0.0864,
    'Scanning': 0.0178,
    'Backdoor': 0.0149,
    'DoS': 0.0147
}

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
        device = features.device
        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 1. L2 Normalize Embeddings: z_i = h_i / ||h_i||_2
        z = F.normalize(features, p=2, dim=1)

        # 2. Scaled Cosine Similarity: sim_{i, j} = z_i . z_j / tau
        sim = torch.matmul(z, z.T) / self.temperature
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # 3. Mask Construction
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        denom_mask = ~self_mask

        label_match = torch.eq(labels.view(-1, 1), labels.view(1, -1))
        pos_mask = label_match & denom_mask # [B, B]

        card_p = pos_mask.sum(dim=1) # [B]
        valid_anchors = card_p > 0
        if not valid_anchors.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 4. Denominator: sum_{a in A(i)} exp(z_i . z_a / tau)
        sim_denom = sim.masked_fill(self_mask, -1e9)
        log_denom = torch.logsumexp(sim_denom, dim=1, keepdim=True)

        # 5. Log-probability for all pairs (i, p):
        log_prob = sim - log_denom

        # 6. Outer summation for L_out_i:
        pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1)
        loss_per_anchor = - pos_log_prob[valid_anchors] / card_p[valid_anchors].float()

        return loss_per_anchor.mean()

def unit_test_supcon():
    print("=" * 80, flush=True)
    print("UNIT TEST: VERIFYING KHOSLA ET AL. L_out^sup FORMULATION", flush=True)
    print("=" * 80, flush=True)
    criterion = SupervisedContrastiveLoss(temperature=0.1)

    feats = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0]
    ], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    loss = criterion(feats, labels)
    expected_manual = - (10.0 - np.log(np.exp(10.0) + 2.0))
    print(f"  PyTorch SupCon Loss: {loss.item():.8f}", flush=True)
    print(f"  Manual Exact Eq. 2:  {expected_manual:.8f}", flush=True)
    diff = abs(loss.item() - expected_manual)
    print(f"  Absolute Difference: {diff:.2e}", flush=True)
    assert diff < 1e-6, "Unit test failed! Math does not match Eq. 2!"
    print("  --> UNIT TEST PASSED: Formula matches Khosla et al. L_out^sup exactly.\n", flush=True)

def run_batch_diagnostics():
    print("=" * 80, flush=True)
    print("DIAGNOSTIC: BATCH-LEVEL POSITIVE COUNT DISTRIBUTION (|P(i)|)", flush=True)
    print("=" * 80, flush=True)
    
    classes = list(CLASS_FRACTIONS.keys())
    probs = np.array([CLASS_FRACTIONS[c] for c in classes])
    probs = probs / probs.sum()

    print("Class Prior Probabilities in NF-ToN-IoT Client Graph:", flush=True)
    for c, p in zip(classes, probs):
        print(f"  {c:<12}: {p*100:>5.2f}%", flush=True)

    # 1. UNIFORM RANDOM SAMPLING
    print("\n" + "=" * 80, flush=True)
    print("DIAGNOSTIC 1: UNIFORM RANDOM BATCH SAMPLING", flush=True)
    print("=" * 80, flush=True)
    np.random.seed(42)
    n_sims = 1000

    for B in [512, 1024, 2048]:
        # Draw 1000 batches from multinomial distribution
        counts = np.random.multinomial(B, probs, size=n_sims) # [n_sims, 8]
        # For each class, card_p = max(0, count - 1)
        # If count == 0: anchor doesn't even exist in batch!
        # If count == 1: anchor exists but |P(i)| = 0! (contrastive term is dead)
        
        print(f"\n--- Batch Size B = {B} (Simulated across {n_sims} batches) ---", flush=True)
        print(f"{'Class':<12} {'Class %':>10} {'Mean Count':>12} {'Mean |P(i)|':>14} {'Batches with |P(i)|=0':>24} {'Status':<25}", flush=True)
        print("-" * 80, flush=True)
        for i, c in enumerate(classes):
            c_counts = counts[:, i]
            mean_cnt = np.mean(c_counts)
            card_p = np.maximum(0, c_counts - 1)
            mean_p = np.mean(card_p)
            zero_p_pct = np.mean(card_p == 0) * 100
            flag = " [!] FREQUENT DEAD ANCHOR" if zero_p_pct > 15 else (" [!] SEVERE COLLAPSE RISK" if zero_p_pct > 30 else " OK")
            print(f"{c:<12} {probs[i]*100:>9.2f}% {mean_cnt:>12.1f} {mean_p:>14.1f} {zero_p_pct:>23.1f}% {flag:<25}", flush=True)

    # 2. CLASS-BALANCED BATCH SAMPLING
    print("\n" + "=" * 80, flush=True)
    print("DIAGNOSTIC 2: CLASS-BALANCED BATCH SAMPLING (M samples per class)", flush=True)
    print("=" * 80, flush=True)
    print("Mechanism: In each client training step, sample exactly M flow edges per class.", flush=True)
    print("Total Batch Size B = 8 * M. Every anchor i of class c has exactly |P(i)| = M - 1 positives.", flush=True)
    
    for M in [32, 64, 128]:
        total_B = 8 * M
        print(f"\n--- Balanced Sampling: M = {M} per class -> Total Batch B = {total_B} ---", flush=True)
        print(f"{'Class':<12} {'Samples in Batch':>18} {'Guaranteed |P(i)|':>20} {'Batches with |P(i)|=0':>24} {'Status':<15}", flush=True)
        print("-" * 80, flush=True)
        for c in classes:
            print(f"{c:<12} {M:>18} {M - 1:>20} {'0.00% (Guaranteed)':>24} {'OPTIMAL':<15}", flush=True)

if __name__ == '__main__':
    unit_test_supcon()
    run_batch_diagnostics()
