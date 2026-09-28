import json

with open('kaggle_kernel_official_ablation/output/official_ablation_results.json') as f:
    res = json.load(f)

classes = ['Benign', 'Backdoor', 'DDoS', 'DoS', 'Injection', 'Password', 'Scanning', 'XSS']

print("=" * 115)
print("TABLE 1: OVERALL METRIC COMPARISON (OFFICIAL NF-ToN-IoT, ZERO-LEAKAGE GROUPED SPLIT)")
print("=" * 115)
headers = ['Model / Configuration', 'Accuracy', 'Balanced Acc', 'Macro F1', 'Macro FNR', 'Binary FNR', 'Binary FPR']
print(f"{headers[0]:<35} {headers[1]:>10} {headers[2]:>14} {headers[3]:>10} {headers[4]:>11} {headers[5]:>11} {headers[6]:>11}")
print("-" * 115)

# Step 2 RF Anchor
print(f"{'RF Anchor (Baseline)':<35} {'72.27%':>10} {'48.60%':>14} {'47.30%':>10} {'51.40%':>11} {'0.05%':>11} {'0.16%':>11}")

for name, cfg in res.items():
    acc = f"{cfg['accuracy']*100:.2f}%"
    bal_acc = f"{cfg['balanced_accuracy']*100:.2f}%"
    mf1 = f"{cfg['macro_f1']*100:.2f}%"
    mfnr = f"{cfg['macro_fnr']*100:.2f}%"
    bfnr = f"{cfg['binary_fnr']*100:.2f}%"
    bfpr = f"{cfg['binary_fpr']*100:.2f}%"
    print(f"{name:<35} {acc:>10} {bal_acc:>14} {mf1:>10} {mfnr:>11} {bfnr:>11} {bfpr:>11}")

print("\n" + "=" * 115)
print("TABLE 2: PER-CLASS RECALL & FNR (1 - RECALL) ACROSS ALL CONFIGURATIONS")
print("=" * 115)
print(f"{'Class':<12} {'RF Anchor':>14} {'GNN Anchor':>14} {'mu=0.0':>14} {'mu=0.1':>14} {'mu=0.3':>14} {'mu=0.5':>14}")
print("-" * 115)

# RF Recalls from Step 2:
rf_rec = {
    'Benign': 0.9984, 'Backdoor': 0.9907, 'DDoS': 0.8045, 'DoS': 0.0979,
    'Injection': 0.9329, 'Password': 0.0419, 'Scanning': 0.0165, 'XSS': 0.0047
}

for cls in classes:
    rf_str = f"{rf_rec[cls]*100:.2f}%"
    gnn_str = f"{res['GNN Anchor (Plain CE)']['per_class'][cls]['recall']*100:.2f}%"
    m0_str = f"{res['mu=0.0 (CB-Focal + Interp)']['per_class'][cls]['recall']*100:.2f}%"
    m1_str = f"{res['mu=0.1 (SupCon + CB-Focal)']['per_class'][cls]['recall']*100:.2f}%"
    m3_str = f"{res['mu=0.3 (SupCon + CB-Focal)']['per_class'][cls]['recall']*100:.2f}%"
    m5_str = f"{res['mu=0.5 (SupCon + CB-Focal)']['per_class'][cls]['recall']*100:.2f}%"
    print(f"{cls:<12} {rf_str:>14} {gnn_str:>14} {m0_str:>14} {m1_str:>14} {m3_str:>14} {m5_str:>14}")

print("\n" + "=" * 115)
print("TABLE 3: PER-CLASS F1-SCORE ACROSS ALL CONFIGURATIONS")
print("=" * 115)
print(f"{'Class':<12} {'RF Anchor':>14} {'GNN Anchor':>14} {'mu=0.0':>14} {'mu=0.1':>14} {'mu=0.3':>14} {'mu=0.5':>14}")
print("-" * 115)

rf_f1 = {
    'Benign': 0.9982, 'Backdoor': 0.9923, 'DDoS': 0.7869, 'DoS': 0.1574,
    'Injection': 0.7341, 'Password': 0.0745, 'Scanning': 0.0311, 'XSS': 0.0093
}

for cls in classes:
    rf_str = f"{rf_f1[cls]*100:.2f}%"
    gnn_str = f"{res['GNN Anchor (Plain CE)']['per_class'][cls]['f1']*100:.2f}%"
    m0_str = f"{res['mu=0.0 (CB-Focal + Interp)']['per_class'][cls]['f1']*100:.2f}%"
    m1_str = f"{res['mu=0.1 (SupCon + CB-Focal)']['per_class'][cls]['f1']*100:.2f}%"
    m3_str = f"{res['mu=0.3 (SupCon + CB-Focal)']['per_class'][cls]['f1']*100:.2f}%"
    m5_str = f"{res['mu=0.5 (SupCon + CB-Focal)']['per_class'][cls]['f1']*100:.2f}%"
    print(f"{cls:<12} {rf_str:>14} {gnn_str:>14} {m0_str:>14} {m1_str:>14} {m3_str:>14} {m5_str:>14}")
