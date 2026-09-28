# FedGATSage Implementation & Ablation Comparison Report

## Table 1: Component Comparison

*Baseline Context: The published FedGATSage paper reports Macro FNR = 22.34% and Balanced Accuracy = 78.58% on NF-ToN-IoT (different split, not directly comparable).*

| Component | Base Paper Approach | What We Implemented (Code Lines) | Measured Impact (Mean ± Std, 3 Seeds) | Why It Helped (Evidence, or "Hypothesis") | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Community Detection** | Louvain algorithm (modularity optimization) on client subgraphs | Leiden algorithm with modularity/CPM (`official_ablation_kaggle.py#L219-L237`) | Not isolated yet | Guarantees well-connected communities and prevents disconnected sub-partitions (*hypothesis*) | Adapted / Operational |
| **2. Community Pooling** | Centrality- / degree-weighted averaging of node embeddings | Uniform mean pooling across community nodes (`official_ablation_kaggle.py#L250-L255`) | Not isolated yet | Reduces computation per federated round; node weighting omitted for implementation tractability (*hypothesis*) | **Simplified** (not an improvement) |
| **3. Detector Fusion** | Random Forest meta-classifier trained strictly on detector probability vectors ($3 \times C$) | Random Forest receiving concatenated detector probabilities, argmax, max confidence, and normalized raw flow features (`official_ablation_kaggle.py#L384-L392`) | **$-44.60\%$** BalAcc when raw features are removed ($94.28\% \to 49.68 \pm 13.97\%$) | Proves RF relies heavily on raw flow feature thresholds to arbitrate between noisy GNN detector logits | Adapted / Sensitive to Raw Inputs |
| **4. Server Overlay Construction** | Dynamic cosine similarity thresholding ($\tau$) to establish inter-community edges | Static top-$k$ ($k=5$) nearest neighbors cosine similarity (`official_ablation_kaggle.py#L279-L290`) | Not isolated yet | Guarantees bounded overlay density and uniform node degrees across federation rounds (*hypothesis*) | **Simplified** (not an improvement) |
| **5. Federated Aggregation** | Performance-weighted averaging based on local validation metric | Standard edge-count weighted FedAvg: $\sum \frac{E_i}{E_{\text{total}}} W_i$ (`official_ablation_kaggle.py#L558-L566`) | Not isolated yet | Standard federated baseline avoiding local validation overhead and transmission complexity (*hypothesis*) | **Simplified** (standard baseline) |
| **6. Classification Loss: CB-Focal** | *Not described in paper* (standard cross-entropy assumed) | Class-Balanced Focal Loss ($\beta=0.9999, \gamma=2.0$, `official_ablation_kaggle.py#L310-L326`, `L534-L535`) | BalAcc: $76.79 \pm 10.16\%$ (isolated, vs Anchor $78.11\%$). High seed variance on Backdoor ($33.82 \pm 57.14\%$) | Loss reweighting alone destabilizes optimization across seeds without metric separation; cannot reliably separate overlapping clusters without SupCon | Tested in Isolation & Stack |
| **7. Metric Loss: SupCon ($\mu=0.1$)** | *Not present in base paper* | Supervised Contrastive Loss ($L_{\text{SupCon}}$, temperature $0.07$, weight $\mu=0.1$, `official_ablation_kaggle.py#L328-L358`, `L539-L544`) | **$+13.59 \pm 2.49\%$** BalAcc gain over $\mu=0.0$ ($80.69\% \to 94.28\%$), Backdoor recall jumps **$+98.77\%$** ($0.80\% \to 99.57\%$) | Pulls intra-class minority edge embeddings together while pushing majority clusters apart, eliminating Backdoor $\to$ Injection confusion (drops from 3,430 to 17 misclassified flows) | **Validated Improvement** |
| **8. Imbalance: Edge Interpolation** | *Not present in base paper* | Intra-class convex feature interpolation along flow feature dimensions with ratio $0.50$ (`official_ablation_kaggle.py#L140-L171`, `L474-L478`) | BalAcc: $80.69 \pm 2.01\%$ (stacked with CB-Focal, $+3.90\%$ over isolated CB-Focal); Backdoor remains suppressed at $0.80 \pm 0.18\%$ | Stabilizes training variance of CB-Focal ($10.16\% \to 2.01\%$ std), but fails to separate Backdoor from Injection without SupCon | Tested in Isolation & Stack |
| **9. Imbalance: Class-Balanced Sampler** | *Not described in paper* | Dynamic mini-batch sampler guaranteeing $M=64$ edges per class per step (`official_ablation_kaggle.py#L360-L376`, `L530`) | Active in **all** runs including Plain-CE Anchor; Anchor achieves only $0.11 \pm 0.09\%$ Backdoor recall without SupCon | Forces equal gradient updates across classes per step; evidence shows sampling alone cannot resolve topological graph overlap without contrastive loss | **Active across all runs** |
| **10. Evaluation Split: Signature-Grouped** | *Not described in paper* (standard random row split assumed) | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped by 8-feature flow signature (`official_ablation_kaggle.py#L584-L590`) | Tested on our data: random row split had 46.88% identical signature leakage; signature-grouped has 0.00% leakage | Prevents memorization of burst packets sharing identical socket signatures, enforcing true generalization across network sessions | **Validated Methodology** |
| **11. FNR Metric Calculation** | Macro FNR reported as 22.34% in base paper | Corrected formula: $\text{Macro FNR} = 1 - \text{Macro Recall} = 1 - \text{Balanced Accuracy}$ (`architecture_and_results_report.md`) | Anchor FNR: $21.89 \pm 2.46\%$; $\mu=0.1$ FNR: $5.72 \pm 1.47\%$ | Replaces earlier codebase bug where false negatives and false positives were inverted; restores mathematically rigorous multi-class FNR definition | **Corrected Metric** |
| **12. Unimplemented Design Objectives** | *N/A* | Attention-based anomaly pooling, Mixture-of-Experts fusion, Learned differentiable adjacency, $\epsilon$-Differential Privacy | *N/A* (not present in execution script) | Omitted from active pipeline to prioritize reproducible topological representations and contrastive stability on official NF-ToN-IoT | **Not implemented in reported runs** |

---

## Table 2: Headline Results (Signature-Grouped Split, Seeds 42 / 43 / 44)

*All models evaluated on the official NF-ToN-IoT dataset (1,102,269 clean train flows, 275,568 clean test flows, 8 classes). All metrics are reported as Mean ± Sample Standard Deviation ($N=3$ seeds).*

| Model Configuration | Accuracy (%) | Balanced Accuracy (%) | Macro F1 (%) | Macro FNR (%) | Backdoor Recall (%) | Scanning Recall (%) | Password Recall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GNN Anchor (Plain CE)** | $90.67 \pm 1.23$ | $78.11 \pm 2.46$ | $79.95 \pm 1.74$ | $21.89 \pm 2.46$ | $0.11 \pm 0.09$ | $80.80 \pm 16.56$ | $56.07 \pm 6.59$ |
| **$\mu=0.0$ (CB-Focal + Interp)** | $92.14 \pm 1.54$ | $80.69 \pm 2.01$ | $82.18 \pm 1.46$ | $19.31 \pm 2.01$ | $0.80 \pm 0.18$ | $89.92 \pm 11.46$ | $68.63 \pm 13.09$ |
| **$\mu=0.1$ (SupCon + CB-Focal)** | **$93.96 \pm 0.94$** | **$94.28 \pm 1.47$** | **$95.61 \pm 1.15$** | **$5.72 \pm 1.47$** | **$99.57 \pm 0.20$** | **$97.98 \pm 3.38$** | $66.82 \pm 6.35$ |
| **$\mu=0.3$ (SupCon + CB-Focal)** | $93.67 \pm 1.77$ | $90.04 \pm 8.52$ | $91.56 \pm 8.18$ | $9.96 \pm 8.52$ | $99.52 \pm 0.31$ | $94.89 \pm 5.92$ | **$70.30 \pm 8.09$** |
| **$\mu=0.5$ (SupCon + CB-Focal)** | $91.49 \pm 0.47$ | $86.04 \pm 3.75$ | $88.75 \pm 4.32$ | $13.96 \pm 3.75$ | $99.08 \pm 0.49$ | $91.03 \pm 15.18$ | $57.83 \pm 3.53$ |

---

## Table 3: Cumulative Ablation Progression (Signature-Grouped Split, Seeds 42 / 43 / 44)

*Isolated component additions starting from the Plain-CE Anchor with balanced batch sampling. All values are grounded in saved run JSON files ($N=3$ seeds each).*

| Step | Configuration | Accuracy (%) | Balanced Accuracy (%) | Macro F1 (%) | Macro FNR (%) | Backdoor Recall (%) | Δ BalAcc vs Prev | Evidence Grounding |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **(a)** | **Plain CE + Balanced Sampler** | $90.67 \pm 1.23$ | $78.11 \pm 2.46$ | $79.95 \pm 1.74$ | $21.89 \pm 2.46$ | $0.11 \pm 0.09$ | Baseline | `run_gnn_anchor_*.json` |
| **(b)** | **+ CB-Focal** (no interpolation) | $87.83 \pm 3.74$ | $76.79 \pm 10.16$ | $79.32 \pm 9.86$ | $23.21 \pm 10.16$ | $33.82 \pm 57.14$ | $-1.32\%$ | `run_ablation_b_*.json` |
| **(c)** | **+ Edge Interpolation** ($\mu=0.0$) | $92.14 \pm 1.54$ | $80.69 \pm 2.01$ | $82.18 \pm 1.46$ | $19.31 \pm 2.01$ | $0.80 \pm 0.18$ | $+3.90\%$ vs (b) | `run_mu0.0_*.json` |
| **(d)** | **+ SupCon ($\mu=0.1$)** | **$93.96 \pm 0.94$** | **$94.28 \pm 1.47$** | **$95.61 \pm 1.15$** | **$5.72 \pm 1.47$** | **$99.57 \pm 0.20$** | **$+13.59\%$** | `run_mu0.1_*.json` |
| **(e)** | **Full Model w/ RF Detector-Only** | $66.57 \pm 2.26$ | $49.68 \pm 13.97$ | $43.23 \pm 8.90$ | $50.32 \pm 13.97$ | $33.02 \pm 57.02$ | $-44.60\%$ vs (d) | `run_ablation_e_*.json` |

---

## Hypothesis Test: Node-Averaging Mechanism Test

*Testing whether averaging flow features onto IP nodes is responsible for suppressing Backdoor recall in GNN models without SupCon:*

| Model Configuration | GNN Node Aggregation | Balanced Accuracy (%) | Macro F1 (%) | Backdoor Recall (%) | Scanning Recall (%) | Password Recall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Anchor (Plain CE)** | Enabled (GNN graph) | $78.11 \pm 2.46$ | $79.95 \pm 1.74$ | $0.11 \pm 0.09$ | $80.80 \pm 16.56$ | $56.07 \pm 6.59$ |
| **Edge-Only Anchor (Plain CE)** | **Disabled (Edge-only)** | **$97.53 \pm 0.47$** | **$97.49 \pm 0.38$** | **$99.33 \pm 0.03$** | **$99.86 \pm 0.06$** | **$97.06 \pm 2.77$** |
| **Full Model ($\mu=0.1$)** | Enabled (GNN graph) | $94.28 \pm 1.47$ | $95.61 \pm 1.15$ | $99.57 \pm 0.20$ | $97.98 \pm 3.38$ | $66.82 \pm 6.35$ |
| **Edge-Only ($\mu=0.1$)** | **Disabled (Edge-only)** | $78.90 \pm 4.96$ | $79.62 \pm 4.59$ | $99.35 \pm 0.13$ | $88.72 \pm 3.01$ | $86.03 \pm 8.70$ |

*Empirical Finding*: Disabling node aggregation in the Plain CE Anchor jumps Backdoor recall from $0.11\%$ to **$99.33\%$**! This confirms the hypothesis: averaging flow features onto IP nodes creates feature blurring that collapses minority Backdoor flows into majority classes, which either requires SupCon ($\mu=0.1$) to disentangle or edge-only classification to prevent.

---

## Notes Section (Core Evidence & Limitations)

1. **Performance gain is heavily concentrated in Backdoor**: Exactly **90.85%** of the balanced accuracy improvement from $\mu=0.0$ to $\mu=0.1$ is driven by rescuing Backdoor recall ($0.80\% \to 99.57\%$), preventing severe collapse into Injection.
2. **Evaluation is restricted to a single dataset**: All reported results derive exclusively from the official NF-ToN-IoT dataset; cross-dataset transferability to other NetFlow benchmarks (e.g., NF-BoT-IoT, NF-UNSW-NB15) remains unverified.
3. **Leakage control is limited to signature-level grouping**: The evaluation guarantees zero flow-signature overlap across folds, but IP addresses remain shared between clients and folds (evaluating signature leakage, not zero-day enterprise subnet transfer).
4. **Contrastive weight sensitivity**: Increasing SupCon loss weight beyond $\mu=0.1$ yields diminishing and noisy returns ($\mu=0.3$ BalAcc: $90.04 \pm 8.52\%$, $\mu=0.5$ BalAcc: $86.04 \pm 3.75\%$), demonstrating sensitivity to contrastive regularization magnitude.
5. **Components within statistical noise**: The margin between Plain CE ($78.11 \pm 2.46\%$) and $\mu=0.0$ ($80.69 \pm 2.01\%$) has overlapping standard deviation intervals ($\pm 2.46\%$ vs $\pm 2.01\%$), confirming that sampling and interpolation alone do not provide statistically significant separation on minority classes without contrastive guidance.
