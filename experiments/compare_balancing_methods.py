"""
Benchmark Script: Compare Balancing Methods for FedGATSage on CIC-ToN-IoT (500K-Flow Split)
Evaluates:
  1. Unbalanced Baseline (Standard FedGATSage, no balancing)
  2. V1: Local Class-Balanced Loss (Cui et al., CVPR 2019)
  3. V2: Local Class-Aware Graph Edge Sampling (Chang & Branco 2021 / GraphSMOTE)
  4. V3: Local Borderline-SMOTE (Han et al., ICIC 2005)
"""

import os
import sys
import time
import json
import logging
import copy
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, classification_report)
from sklearn.model_selection import train_test_split

# Import modules from V1, V2, V3
from fedgatsage_v1_class_weighted import (
    load_dataset, build_graph, compute_local_class_weights,
    TemporalGATDetector, PAPER_8_CLASSES
)
from fedgatsage_v2_graph_sampling import apply_local_graph_edge_sampling
from fedgatsage_v3_borderline_smote import local_borderline_smote_han2005

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [Benchmark] %(message)s')
logger = logging.getLogger("CompareBalancing")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_and_eval_system(
    client_graphs: List[Dict],
    test_graph: Dict,
    num_classes: int,
    inv_mapper: Dict[int, str],
    client_criteria: Optional[List[nn.Module]] = None,
    num_rounds: int = 3,
    local_epochs: int = 2
) -> Dict[str, Any]:
    num_clients = len(client_graphs)
    in_dim = client_graphs[0]['features'].shape[1]
    detector_types = ['temporal', 'content', 'behavioral']
    
    models = {d: [TemporalGATDetector(in_dim, 256, 8, num_classes).to(device) for _ in range(num_clients)] for d in detector_types}
    opts = {d: [torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4) for m in models[d]] for d in detector_types}
    default_criterion = nn.CrossEntropyLoss()

    for r in range(num_rounds):
        for d in detector_types:
            c_states = []
            for c_idx in range(num_clients):
                m, opt = models[d][c_idx], opts[d][c_idx]
                g = client_graphs[c_idx]
                x, ei, el = g['features'].to(device), g['edge_index'].to(device), g['edge_labels'].to(device)
                
                crit = client_criteria[c_idx] if client_criteria is not None else default_criterion
                m.train()
                for _ in range(local_epochs):
                    opt.zero_grad()
                    _, preds = m(x, ei)
                    loss = crit(preds, el)
                    loss.backward()
                    opt.step()

                m.eval()
                c_states.append({k: v.cpu() for k, v in m.state_dict().items()})

            # FedAvg
            agg_state = {k: sum(c_states[i][k].float() for i in range(num_clients)) / num_clients for k in c_states[0]}
            for c_idx in range(num_clients):
                models[d][c_idx].load_state_dict({k: v.to(device) for k, v in agg_state.items()})

    # Meta Evaluation
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

    acc = accuracy_score(y_eval, final_preds)
    b_acc = balanced_accuracy_score(y_eval, final_preds)
    m_f1 = f1_score(y_eval, final_preds, average='macro', zero_division=0)
    w_f1 = f1_score(y_eval, final_preds, average='weighted', zero_division=0)
    per_class_f1 = f1_score(y_eval, final_preds, average=None, zero_division=0)
    per_class_dict = {inv_mapper[i]: round(float(per_class_f1[i]), 4) for i in range(num_classes)}

    return {
        'accuracy': acc,
        'balanced_accuracy': b_acc,
        'macro_f1': m_f1,
        'weighted_f1': w_f1,
        'per_class_f1': per_class_dict
    }

def run_comparison(num_samples: int = 500000, num_rounds: int = 3, local_epochs: int = 2):
    logger.info(f"Loading {num_samples:,}-flow CIC-ToN-IoT dataset...")
    raw_df, dataset_name = load_dataset(max_samples=num_samples)
    unique_attacks = sorted(raw_df['Attack'].unique())
    label_mapper = {a: i for i, a in enumerate(unique_attacks)}
    inv_mapper = {i: a for a, i in label_mapper.items()}
    num_classes = len(unique_attacks)

    train_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42, stratify=raw_df['Attack'])
    client_dfs = np.array_split(train_df.sample(frac=1.0, random_state=42).reset_index(drop=True), 5)
    feat_cols = [c for c in raw_df.columns if c not in ['Label', 'Attack', 'Src IP', 'Dst IP', 'Timestamp']]

    logger.info("Constructing test graph with Leiden community detection...")
    test_graph = build_graph(test_df, feat_cols, label_mapper)

    results = {}

    # 1. Unbalanced Baseline
    logger.info("\n>>> Running UNBALANCED BASELINE (No Balancing)...")
    base_client_graphs = [build_graph(cdf, feat_cols, label_mapper) for cdf in client_dfs]
    results['Unbalanced Baseline'] = train_and_eval_system(
        base_client_graphs, test_graph, num_classes, inv_mapper,
        client_criteria=None, num_rounds=num_rounds, local_epochs=local_epochs
    )

    # 2. Version 1: Local Class-Balanced Loss (Cui et al., CVPR 2019)
    logger.info("\n>>> Running VERSION 1: Local Class-Balanced Loss (Cui et al. 2019)...")
    v1_criteria = []
    for c_idx in range(5):
        c_lbls = base_client_graphs[c_idx]['edge_labels'].numpy()
        c_weights = compute_local_class_weights(c_lbls, num_classes, beta=0.9999)
        v1_criteria.append(nn.CrossEntropyLoss(weight=c_weights))
    results['V1: Class-Balanced Loss (Cui et al. 2019)'] = train_and_eval_system(
        base_client_graphs, test_graph, num_classes, inv_mapper,
        client_criteria=v1_criteria, num_rounds=num_rounds, local_epochs=local_epochs
    )

    # 3. Version 2: Local Graph Edge Sampling (Chang & Branco 2021)
    logger.info("\n>>> Running VERSION 2: Local Graph Edge Sampling (Chang & Branco 2021)...")
    v2_client_dfs = [apply_local_graph_edge_sampling(cdf, feat_cols, target_ratio=0.25) for cdf in client_dfs]
    v2_client_graphs = [build_graph(scdf, feat_cols, label_mapper) for scdf in v2_client_dfs]
    results['V2: Graph Edge Sampling (Chang & Branco 2021)'] = train_and_eval_system(
        v2_client_graphs, test_graph, num_classes, inv_mapper,
        client_criteria=None, num_rounds=num_rounds, local_epochs=local_epochs
    )

    # 4. Version 3: Local Borderline-SMOTE (Han et al., ICIC 2005)
    logger.info("\n>>> Running VERSION 3: Local Borderline-SMOTE (Han et al. 2005)...")
    v3_client_dfs = [local_borderline_smote_han2005(cdf, feat_cols, target_ratio=0.25) for cdf in client_dfs]
    v3_client_graphs = [build_graph(scdf, feat_cols, label_mapper) for scdf in v3_client_dfs]
    results['V3: Borderline-SMOTE (Han et al. 2005)'] = train_and_eval_system(
        v3_client_graphs, test_graph, num_classes, inv_mapper,
        client_criteria=None, num_rounds=num_rounds, local_epochs=local_epochs
    )

    # Save and display comparison table
    logger.info("\n" + "="*80)
    logger.info("FINAL COMPARISON OF BALANCING TECHNIQUES ON 500K CIC-ToN-IoT")
    logger.info("="*80)
    
    summary_rows = []
    for method, metrics in results.items():
        row = {
            'Method': method,
            'Accuracy': f"{metrics['accuracy']*100:.2f}%",
            'Balanced Acc': f"{metrics['balanced_accuracy']*100:.2f}%",
            'Macro F1': f"{metrics['macro_f1']*100:.2f}%",
            'Weighted F1': f"{metrics['weighted_f1']*100:.2f}%"
        }
        for cls_name, f1_val in metrics['per_class_f1'].items():
            row[f"{cls_name} F1"] = f"{f1_val:.4f}"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    with open('results_comparison_500k.json', 'w') as f:
        json.dump(results, f, indent=2)
    summary_df.to_csv('results_comparison_500k.csv', index=False)
    logger.info("Saved comparison results to results_comparison_500k.json and .csv")

if __name__ == '__main__':
    run_comparison(num_samples=500000, num_rounds=3, local_epochs=2)
