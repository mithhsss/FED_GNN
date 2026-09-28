# B3-related balancing methods on similar intrusion-detection datasets

## Executive conclusion

For the proposed CIC-ToN-IoT method—**client-local class-aware edge/flow sampling + class-weighted loss under federated GNN training**—the literature provides useful partial precedents but no verified exact match.

The strongest evidence comes from two directions:

1. **Closest graph/flow task:** Shin & Kim use endpoint nodes and flow edges, with benign undersampling, minority SMOTE, and cost-sensitive learning on NF-ToN-IoT-v2 and related NetFlow datasets.
2. **Closest federated B3 design:** DAFL combines local random oversampling with an imbalance-aware asymmetric focal loss in federated learning on NSL-KDD and UNSW-NB15.

Therefore, B3 is not an invented idea. It combines techniques already supported separately or partially by prior work, but its exact application to **CIC-ToN-IoT + federated edge-level GNN IDS** remains insufficiently studied.

## 1. What is B3?

B3 consists of two coordinated components:

- **class-aware local sampling:** minority-class flows/edges are sampled more often during each client’s local training;
- **imbalance-aware loss:** minority classes receive higher loss weight, using effective-number weighting, capped inverse-frequency weighting, cost-sensitive weighting, or a focal variant.

The recommended B3 implementation for CIC-ToN-IoT is:

> FedAvg + client-local class-aware edge/flow sampling + capped effective-number weighted cross-entropy.

This is different from globally applying SMOTE before federated partitioning. The latter can leak information across clients and create synthetic records without respecting graph ownership.

## 2. CIC-ToN-IoT versus comparable datasets

| Property | CIC-ToN-IoT | NF-ToN-IoT / NF-UNSW-NB15 / NF-BoT-IoT | UNSW-NB15 | CIC-IDS2017 | CICIoT2023 |
|---|---|---|---|---|---|
| Data type | CICFlowMeter network flows derived from ToN-IoT packet captures | NetFlow-style network-flow records | Network-flow and host/network traffic records | Network-flow records from controlled attack scenarios | IoT network traffic flows |
| Typical graph construction | IP/endpoints as nodes; flows as edges | IP/port endpoints as nodes; flows as edges | Often endpoints or abstract traffic records | Often converted to graphs or client silos | IoT devices/flows; graph design varies |
| Natural label unit for your project | **Edge/flow** | **Edge/flow** | Flow or traffic record; sometimes edge | Flow/record | Flow/record or device/edge |
| Feature convention | Often described as 83 CICFlowMeter features, but releases/papers disagree | NetFlow feature schemas, often fewer than raw CICFlowMeter fields | Dataset-specific | Dataset-specific | Dataset-specific |
| Federated difficulty | Severe client-level imbalance and possible missing classes | Similar non-IID and rare-flow issues | Commonly used for non-IID FL studies | Commonly partitioned into silos/clients | Large-scale IoT client distributions |
| Main comparability issue | Exact release and class counts must be fixed | NF datasets are related to ToN-IoT but are not identical to CIC-ToN-IoT | Different feature semantics and class distribution | Different attack taxonomy and graph structure | Different IoT environment and release |

CIC-ToN-IoT should not be treated as identical to NF-ToN-IoT, original ToN-IoT, CIC-IDS2017, or CICIoT2023. They are useful methodological comparators, not interchangeable datasets.

## 3. Strongest papers using B3-like methods

### A. Shin & Kim — closest graph/edge-task match

**Paper:** “Graph-Based Intrusion Detection with Explainable Edge Classification Learning.” 2025/2026. DOI: [10.32604/cmc.2025.068767](https://doi.org/10.32604/cmc.2025.068767). [Full paper](https://www.techscience.com/cmc/v86n1/64433/html)

**Datasets:** CIC-IDS2017, NF-CIC-IDS2018-v2, NF-ToN-IoT-v2, NF-UNSW-NB15-v2, NF-BoT-IoT-v2, and NF-UQ-NIDS-v2.

**Graph task:** This is an edge-classification design. IP/port endpoints represent nodes, and network flows represent labeled edges. This is structurally close to a CIC-ToN-IoT flow-level GNN IDS.

**Balancing implementation:**

1. majority/benign under-sampling;
2. minority-class SMOTE over-sampling;
3. cost-sensitive learning, with class costs related to inverse class prevalence.

This is the closest methodological match to B3 because it combines graph edge classification with both sampling and imbalance-aware loss.

**Reported improvement:**

| Evaluation | Default/baseline | Hybrid sampling + cost-sensitive learning |
|---|---:|---:|
| F2-score | 0.668 | 0.884 |
| Recall | 0.658 | 0.875 |
| NF-ToN-IoT-v2 F1 | Not clearly reported for the default comparison | 0.913 |
| NF-ToN-IoT-v2 accuracy | Not clearly reported for the default comparison | 0.921 |
| Average across six datasets: F1 | Not applicable | 0.960 |
| Average across six datasets: accuracy | Not applicable | 0.964 |

**Interpretation:** The improvement is substantial, especially for recall and F2-score, which emphasize minority detection more than ordinary accuracy. However, the paper’s main ablation is not a clean macro-F1/balanced-accuracy comparison for every dataset, so the numbers should not be directly copied as expected CIC-ToN-IoT gains.

**How it transfers to CIC-ToN-IoT:**

- use its endpoint-node/flow-edge design as the graph structure;
- replace global sampling with client-local sampling;
- fit SMOTE or any synthetic generator only inside each client’s training partition;
- use class-weighted loss;
- evaluate with macro-F1, balanced accuracy, and minority recall.

**Limitation:** It is not federated, and NF-ToN-IoT-v2 is not the same dataset as CIC-ToN-IoT.

### B. DAFL — closest federated sampling + imbalance-loss match

**Paper:** “Disparity-Aware Federated Learning for Intrusion Detection.” Islam & Al Islam, 2023. DOI: [10.1145/3629188.3629197](https://doi.org/10.1145/3629188.3629197). [ACM full text](https://dl.acm.org/doi/full/10.1145/3629188.3629197)

**Datasets:** NSL-KDD and UNSW-NB15.

**Federated setup:** Ten clients with heterogeneous class distributions. Clients are clustered using cosine similarity of imbalance vectors.

**Balancing implementation:**

- local random oversampling at clients;
- asymmetric Unified Focal Loss combining modified asymmetric focal loss and focal-Tversky loss;
- inverse class-distribution weights;
- client clustering based on imbalance profiles.

This is a strong B3-like federated precedent, although it does not use a GNN.

**Reported F1 improvement:**

| Dataset/task | Baseline DNN F1 | DAFL F1 | Improvement |
|---|---:|---:|---:|
| NSL-KDD binary | 76.38% | 87.30% | +10.92 percentage points |
| NSL-KDD multiclass | 72.72% | 83.48% | +10.76 percentage points |
| UNSW-NB15 binary | 95.39% | 97.25% | +1.86 percentage points |
| UNSW-NB15 multiclass | 81.31% | 87.71% | +6.40 percentage points |

**Interpretation:** DAFL demonstrates that local oversampling and imbalance-aware loss can improve minority-sensitive federated IDS performance, especially in multiclass settings. It also shows that the client’s imbalance profile can be used in federated coordination.

**How it transfers to CIC-ToN-IoT:**

- calculate each client’s class-distribution vector;
- sample rare flow/edge classes more frequently locally;
- use effective-number weighted CE or a focal variant;
- optionally cluster clients by imbalance profile;
- preserve FedAvg as the first aggregation baseline.

**Limitation:** It is not GNN-based, and NSL-KDD/UNSW-NB15 have different feature and attack distributions.

### C. ETASR CPS study — direct hybrid sampling + focal-loss ablation

**Paper:** “Class-Imbalance-Aware Federated Intrusion Detection for Cyber-Physical Systems.” Swetha & Shelke, 2026. DOI: [10.48084/etasr.15034](https://doi.org/10.48084/etasr.15034). [PDF](https://pdfs.semanticscholar.org/4c05/b3cdd9c8aa4806740a5e2daa9d2afc5177cb.pdf)

**Dataset:** CIC-IDS2017.

**Federated setup:** Three non-IID clients representing different attack distributions: infiltration, DDoS, and PortScan plus benign traffic.

**Balancing implementation:**

- focal loss only;
- oversampling only;
- oversampling plus focal loss.

The PortScan results are especially relevant:

| Method | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|
| FedProx baseline | 0.00 | 0.00 | 0.00 | 96.8% |
| Focal loss only | 0.57 | 0.41 | 0.48 | 97.1% |
| Oversampling only | 0.66 | 0.59 | 0.62 | 97.4% |
| Oversampling + focal loss | 0.78 | 0.72 | 0.75 | 97.7% |

**Interpretation:** This is a clear example where the hybrid method improves minority-class recall and F1 much more meaningfully than ordinary accuracy. Accuracy increases only from 96.8% to 97.7%, but PortScan F1 increases from 0.00 to 0.75.

**How it transfers to CIC-ToN-IoT:**

This supports testing B3 as a combination rather than assuming that sampling or loss weighting alone is sufficient. It also demonstrates why your evaluation must include minority-class F1 and recall.

**Limitation:** This is not a GNN study, and CIC-IDS2017 is not CIC-ToN-IoT.

### D. FedMADE — client-local SMOTE precedent

**Paper:** “FedMADE,” Sun et al., 2024. DOI: [10.1007/978-3-031-75764-8_15](https://doi.org/10.1007/978-3-031-75764-8_15). [Open version](https://arxiv.org/html/2408.07152v1)

**Dataset:** CICIoT2023.

**Balancing implementation:** Client-local SMOTE. Web-based and brute-force classes are quadrupled, while other eligible minority classes are doubled. Some classes are excluded from SMOTE.

**Reported findings:** Minority attack accuracy improves by up to 71.07%. Reported web-attack accuracy reaches 58.5% for CNN and 69.5% for FCNN, compared with at most 13.8% for competing federated methods.

**Relevance:** This supports performing synthetic balancing after client partitioning rather than globally before federation.

**Limitation:** It uses local SMOTE but does not provide the same weighted-loss component as B3. It is also based on CICIoT2023, not CIC-ToN-IoT, and the graph formulation is not equivalent to your proposed flow-edge GNN.

### E. Per-client SMOTE study — local balancing after partitioning

**Paper:** “Per-Client SMOTE for Federated Intrusion Detection,” Demirbaş Paray & Aydos, 2026. DOI: [10.3390/app16020801](https://doi.org/10.3390/app16020801)

**Datasets:** CICIDS-2017, InSDN-OVS, and 5G-NIDD.

**Balancing implementation:** SMOTE independently after client partitioning, with a maximum 10:1 ratio cap and adaptive neighbor selection. Focal loss is also evaluated as an ablation.

**Reported CICIDS-2017 macro-F1:**

| Method | Macro-F1 |
|---|---:|
| Centralized reference | 0.8600 |
| Federated + per-client SMOTE | 0.9835 |
| No-SMOTE ablation | 0.8068 |
| Full balancing ablation | 0.9774 |

**Interpretation:** This is useful evidence for local—not global—balancing. However, the large improvement should be independently reproduced because differences in split, client partition, and evaluation protocol can strongly affect the result.

**Limitation:** No GNN and no exact class-aware edge sampling plus weighted loss.

### F. FIDS-CL — federated cost-sensitive weighting without sampling

**Paper:** “FIDS-CL,” IEEE Internet of Things Journal, 2025. DOI: [10.1109/JIOT.2025.3586938](https://doi.org/10.1109/JIOT.2025.3586938). [Author PDF](https://delta.cs.cinvestav.mx/~ccoello/journals/lin-internet-2025-final.pdf.gz)

**Datasets:** CICIDS2017, UNSW-NB15, and Bot-IoT.

**Method:** Dynamic cost-sensitive class weights are calculated from client class F1 values and sent to clients for local weighted cross-entropy. A dynamic aggregation mechanism gives more influence to clients that perform well on weak classes.

**Example F1 results:**

| Dataset, three clients | FedAvg F1 | FIDS-CL F1 | Improvement |
|---|---:|---:|---:|
| CICIDS2017 | 0.8553 | 0.8651 | +0.0098 |
| UNSW-NB15 | 0.6292 | 0.6532 | +0.0240 |
| Bot-IoT | 0.9844 | 0.9906 | +0.0062 |

**Relevance:** It supports the weighted-loss half of B3 and shows that weights can be adapted using client-level minority performance.

**Limitation:** It does not include class-aware sampling or GNNs.

### G. DS-FedIDS — dynamic local class sampling without weighted loss

**Paper:** “DS-FedIDS,” Youm & Kim, Applied Sciences 2025. DOI: [10.3390/app15095067](https://doi.org/10.3390/app15095067)

**Dataset:** UNSW-NB15-derived data.

**Balancing implementation:** Each client computes a class threshold based on its local sample count. Majority classes above the threshold are downsampled, while minority classes below it are upsampled with replacement.

**Reported comparison:** DS-FedIDS accuracy is 0.9266 versus 0.8787 for FedPer; training time is 20.08 seconds versus 661.72 seconds.

**Relevance:** It supports local class-aware sampling in federated learning.

**Limitation:** The specialized loss is not reported as a weighted-loss method, and the study is not GNN-based.

### H. GEAFL-IDS — edge GNN with imbalance-aware loss only

**Paper:** “Intrusion Detection Method Based on Graph Edge Attention and Focal Loss,” 2025. DOI: [10.1145/3723890.3723895](https://dl.acm.org/doi/full/10.1145/3723890.3723895)

**Datasets:** NF-BoT-IoT and NF-UNSW-NB15.

**Graph task:** Edge classification using network flows as edge features.

**Balancing:** Class-weighted focal loss. The paper explicitly identifies oversampling/sampling as future work, so it does not use both B3 ingredients.

| Dataset | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| NF-BoT-IoT | 0.8308 | 0.8633 | 0.8308 | 0.8412 |
| NF-UNSW-NB15 | 0.9787 | 0.9775 | 0.9787 | 0.9780 |

**Relevance:** It is a strong graph/edge-task precedent for the loss component, but not a B3 implementation.

### I. LGSMOTE-IDS — synthetic balancing on a related graph dataset

**Paper:** “LGSMOTE-IDS,” Expert Systems with Applications, 2025. DOI: [10.1016/j.eswa.2025.127645](https://doi.org/10.1016/j.eswa.2025.127645)

**Datasets:** NF-UNSW-NB15, NF-BoT-IoT, and NF-ToN-IoT.

**Balancing:** Weighted-distance SMOTE. It transforms the original edge-classification problem into node classification on a protocol-service line graph.

**Reported weighted-F1 gains:** approximately 18.11%, 45.91%, and 36.41% across reported minority-class settings.

**Relevance:** It shows that graph-aware synthetic balancing can improve minority performance on related NetFlow data.

**Limitation:** It does not verify the required weighted-loss component and changes the task from original flow-edge classification to line-graph node classification.

## 4. Papers that are related but not direct B3 evidence

Several graph-imbalance methods are important conceptually but should not be presented as direct CIC-ToN-IoT evidence:

| Method | Main task | Why not directly transferable |
|---|---|---|
| GraphSMOTE | Imbalanced node classification | Synthesizes minority nodes and graph connections, not traffic-flow edges |
| GraphENS | Minority ego-network synthesis | Node labels and graph assumptions differ from flow-edge labels |
| ReNode | Topology-aware node reweighting | Designed for node-classification topology imbalance |
| PC-GNN | Fraud-node classification with class-aware neighbor sampling | Useful sampling idea, but prediction target is a node |
| CARE-GNN | Fraud-node classification | Neighbor selection and node imbalance, not network-flow edge prediction |
| GraphSHA | Synthetic minority node/neighbor generation | Requires redesign for client-local flow-edge classification |

The most useful idea to borrow from these methods is **class-aware neighborhood/edge selection**, not their complete synthetic-node algorithms.

## 5. Comparison with your CIC-ToN-IoT project

| Dimension | CIC-ToN-IoT project | Closest paper evidence | What you can implement |
|---|---|---|---|
| Dataset | CIC-ToN-IoT, exact release to be fixed | NF-ToN-IoT-v2, UNSW-NB15, CIC-IDS2017, CICIoT2023 | Publish archive hash, counts, and label table |
| Graph task | Recommended edge/flow classification | Shin & Kim, GEAFL-IDS, FedGATSage | IP/endpoint nodes; flows as labeled edges |
| Federation | Multiple private clients | DAFL, FedMADE, FIDS-CL, DS-FedIDS | Partition clients before balancing |
| Sampling | Client-local class-aware edge sampling | DAFL, FedMADE, DS-FedIDS, Shin & Kim | Oversample minority edge IDs in local training batches |
| Loss | Effective-number/capped class-weighted CE | Shin & Kim, FIDS-CL, GEAFL-IDS, DAFL | Compute weights from each client’s local training counts |
| Graph preservation | Must preserve client graph | Mostly absent from B3 papers | Never create cross-client edges or synthetic global graphs |
| Main evaluation | Balanced accuracy, macro-F1, per-class recall, PR-AUC | Many papers use accuracy/F1 only | Add worst-client macro-F1 and minimum class recall |
| Main novelty | Exact CIC-ToN-IoT + federated GNN + B3 | No verified exact paper found | Evaluate B0–B6 under one fixed architecture |

## 6. What improvement should you realistically expect?

Do not promise a specific accuracy improvement before experimentation. Results depend heavily on:

- exact CIC-ToN-IoT release;
- number of classes retained;
- client label skew;
- graph construction;
- split leakage;
- whether the evaluation is macro-F1 or weighted-F1;
- whether synthetic data are generated locally or globally.

The comparable papers suggest the largest improvements usually appear in **minority recall and macro-F1**, not necessarily ordinary accuracy.

Examples:

- The ETASR CPS study improved PortScan F1 from 0.00 to 0.75, while ordinary accuracy moved only from 96.8% to 97.7%.
- DAFL improved UNSW-NB15 multiclass F1 by about 6.40 percentage points.
- Shin & Kim improved F2-score from 0.668 to 0.884 and recall from 0.658 to 0.875 using hybrid sampling and cost-sensitive learning.
- FIDS-CL improved UNSW-NB15 F1 from 0.6292 to 0.6532 through cost-sensitive federated weighting.

These are not expected CIC-ToN-IoT results. They justify testing B3, but they do not prove its eventual performance on your dataset.

## 7. Recommended experiment for CIC-ToN-IoT

Keep the GNN architecture, client partition, optimizer, and FedAvg aggregation fixed. Change only the balancing method:

| Experiment | Local sampling | Loss | Purpose |
|---|---|---|---|
| B0 | Ordinary sampling | Ordinary CE | Reference baseline |
| B1 | Class-aware edge sampling | Ordinary CE | Sampling effect |
| B2 | Ordinary sampling | Weighted CE | Loss effect |
| **B3** | **Class-aware edge sampling** | **Effective-number/capped weighted CE** | **Main proposed method** |
| B4 | Local Borderline-SMOTE/ADASYN where valid | Weighted/ordinary CE | Synthetic tabular ablation |
| B5 | Ordinary sampling | Focal loss | Hard-example baseline |
| B6 | Optional graph-aware synthesis | Weighted CE | Advanced ablation only |

Fit all weights, scalers, feature selectors, and synthetic generators on each client’s local training data only. Never apply global SMOTE before client partitioning.

Report:

- accuracy;
- balanced accuracy;
- macro-F1;
- weighted F1;
- per-class precision, recall, and F1;
- PR-AUC for rare classes;
- confusion matrix;
- FPR/FNR;
- worst-client macro-F1;
- minimum per-class client recall;
- communication cost and training time;
- mean and standard deviation across multiple seeds.

## Final answer

Yes, there are research papers supporting the components of B3 on similar IDS datasets.

- **Closest edge/graph paper:** Shin & Kim, using NF-ToN-IoT-v2 and related NetFlow datasets with undersampling + SMOTE + cost-sensitive learning.
- **Closest federated paper:** DAFL, using local oversampling + asymmetric focal loss on UNSW-NB15 and NSL-KDD.
- **Closest direct hybrid ablation:** the 2026 CIC-IDS2017 federated study showing oversampling + focal loss improved minority PortScan F1 from 0.00 to 0.75.
- **Closest exact CIC-ToN-IoT graph-FL baseline:** FedGATSage, but it does not implement explicit B3 balancing.

So your project is best described as a **new combination and evaluation of established imbalance principles in an under-studied CIC-ToN-IoT federated edge-GNN setting**, rather than claiming that the individual techniques themselves are new.

Access and evidence limitations remain for some IEEE/ACM/Elsevier papers. Values marked in the tables come from accessible paper text or official metadata; unavailable values were not estimated.
