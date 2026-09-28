import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

def run_unit_tests():
    print("=" * 80)
    print("STEP 3: SUPERVISED CONTRASTIVE LOSS UNIT TEST ON RANDOM (NON-SEPARABLE) EMBEDDINGS")
    print("=" * 80)

    torch.manual_seed(42)
    np.random.seed(42)

    # Setup: 8 classes, M=64 per class => Batch size N = 512
    num_classes = 8
    M = 64
    N = num_classes * M
    labels = torch.cat([torch.full((M,), c, dtype=torch.long) for c in range(num_classes)])

    criterion_t007 = SupervisedContrastiveLoss(temperature=0.07)

    # 1. Theoretical Case: Uniformly zero inter-sample similarity (all pairwise similarities equal)
    # Manual computation:
    # Under perfectly non-informative embeddings, every pair has equal similarity.
    # For any anchor i:
    # Denominator = sum_{a in A(i)} exp(c) = (N - 1) * exp(c)
    # Numerator for positive p = exp(c)
    # prob = 1 / (N - 1)
    # -log(1 / (N - 1)) = log(N - 1)
    # For N = 512, N - 1 = 511.
    manual_theoretical_log = np.log(N - 1)
    print(f"Configuration: N = {N} (8 classes x 64 samples/class), tau = 0.07")
    print(f"Manual Theoretical Baseline (equal similarity / random chance on N-1 alternatives):")
    print(f"  log(N - 1) = log({N-1}) = {manual_theoretical_log:.5f}")

    # Test with orthogonal / zero dot product embeddings (d = N)
    eye_feats = torch.eye(N, dtype=torch.float32)
    loss_ortho = criterion_t007(eye_feats, labels).item()
    print(f"\nTest 1 (Orthogonal embeddings, all pairwise similarities = 0):")
    print(f"  Empirical Loss:    {loss_ortho:.5f}")
    print(f"  Manual log(N - 1): {manual_theoretical_log:.5f}")
    print(f"  Difference:        {abs(loss_ortho - manual_theoretical_log):.8f}")
    assert abs(loss_ortho - manual_theoretical_log) < 1e-4, "Mismatch in orthogonal test!"

    # Test 2: Standard Gaussian Random Embeddings (dim = 256)
    # Sample 100 trials to report mean and std
    dim = 256
    rand_losses = []
    for seed in range(50):
        torch.manual_seed(seed)
        rand_feats = torch.randn(N, dim)
        loss_val = criterion_t007(rand_feats, labels).item()
        rand_losses.append(loss_val)

    mean_rand_loss = np.mean(rand_losses)
    std_rand_loss = np.std(rand_losses)
    print(f"\nTest 2 (Random Gaussian Embeddings d={dim}, 50 trials):")
    print(f"  Mean Empirical Loss: {mean_rand_loss:.5f} +/- {std_rand_loss:.5f}")
    print(f"  Manual log(N - 1):   {manual_theoretical_log:.5f}")

    # Test 3: Perfectly Separated Clusters (Target State)
    # If embeddings for each class are perfectly clustered with zero intra-class distance
    # and orthogonal between classes:
    class_centers = torch.eye(num_classes, dtype=torch.float32) * 10.0
    perfect_feats = class_centers[labels]
    loss_perfect = criterion_t007(perfect_feats, labels).item()
    # Manual expectation for perfect clusters:
    # Within class, similarities = 1 / tau = 14.28
    # Across classes, similarities = 0
    # Positives in denominator = (M - 1) * exp(1/tau)
    # Negatives in denominator = (N - M) * exp(0) = (N - M)
    # Ratio = exp(1/tau) / [ (M - 1)*exp(1/tau) + (N - M) ]
    # -log(ratio) = log [ (M - 1) + (N - M)*exp(-1/tau) ] - log(1)
    # Since exp(-14.28) ~ 6e-7, (N - M)*exp(-1/tau) ~ 0
    # Loss ~ log(M - 1) = log(63) = 4.14313
    print(f"\nTest 3 (Perfect Clustering Limit):")
    print(f"  Empirical Loss:          {loss_perfect:.5f}")
    print(f"  Manual Target log(M - 1):{np.log(M - 1):.5f}")

    print("\nSummary of Loss Spectrum for SupCon (N=512, M=64):")
    print(f"  Random (Uninformative) Level:  {mean_rand_loss:.2f} (Theoretical log(511) = {manual_theoretical_log:.2f})")
    print(f"  Separated (Target) Level:      {loss_perfect:.2f} (Theoretical log(63)  = {np.log(M-1):.2f})")
    print("=" * 80)

if __name__ == '__main__':
    run_unit_tests()
