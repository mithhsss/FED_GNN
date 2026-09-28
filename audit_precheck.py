import os
import sys
import hashlib
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

PAPER_8_CLASSES = ['Benign', 'Backdoor', 'DDoS', 'DoS', 'Injection', 'Password', 'Scanning', 'XSS']
CLASS_MAP = {c.lower(): i for i, c in enumerate(PAPER_8_CLASSES)}
INV_CLASS_MAP = {i: c for i, c in enumerate(PAPER_8_CLASSES)}

def map_nf_toniot_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {
        'IPV4_SRC_ADDR': 'Src IP',
        'IPV4_DST_ADDR': 'Dst IP',
        'L4_SRC_PORT': 'Src Port',
        'L4_DST_PORT': 'Dst Port',
        'PROTOCOL': 'Protocol',
        'FLOW_DURATION_MILLISECONDS': 'Flow Duration',
        'IN_BYTES': 'TotLen Fwd Pkts',
        'OUT_BYTES': 'TotLen Bwd Pkts',
        'IN_PKTS': 'Tot Fwd Pkts',
        'OUT_PKTS': 'Tot Bwd Pkts',
    }
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

    # Bitwise RFC 793 decoding
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

def main():
    print("=" * 80)
    print("MANDATORY PRE-CHECK & LEAKAGE AUDIT")
    print("=" * 80)

    # 1. Locate Dataset
    data_path = 'data/nftoniot/NF-ToN-IoT.parquet'
    if not os.path.exists(data_path):
        data_path = 'data/nf_ton_unzipped/7ca78ae35fa4961a_MOHANAD_A4706/data/NF-ToN-IoT.csv'
    
    print(f"Loading data from: {data_path}")
    t0 = time.time()
    if data_path.endswith('.parquet'):
        raw_df = pd.read_parquet(data_path)
    else:
        raw_df = pd.read_csv(data_path)
    print(f"Loaded {len(raw_df):,} raw flows in {time.time()-t0:.2f}s")

    df = map_nf_toniot_columns(raw_df)
    df['attack_lower'] = df['Attack'].astype(str).str.lower()
    df = df[df['attack_lower'].isin(CLASS_MAP)].copy().reset_index(drop=True)
    print(f"Cleaned 8-Class Dataset: {len(df):,} flows")

    print("\nClass Distribution in Full Dataset:")
    counts = df['Attack'].value_counts()
    for cls, cnt in counts.items():
        print(f"  {cls:<15}: {cnt:>10,} ({cnt/len(df)*100:.2f}%)")

    # Identical Train/Test Split (80/20, seed=42, stratify)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Attack'])
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"\nTrain size: {len(train_df):,} | Test size: {len(test_df):,}")

    print("\n" + "=" * 80)
    print("CHECK 1: Edge Interpolation Code Path Verification")
    print("=" * 80)
    print("In `kaggle_kernel/v1_paper_exact_kaggle.py`:")
    print("  Line 610: train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Attack'])")
    print("  Line 614: client_dfs = stratified_split_clients(train_df, num_clients, seed=42)")
    print("  Line 621: train_dfs = {d: [feature_engineers[d].extract_features(cdf) for cdf in client_dfs] ...}")
    print("  Line 631: b7_edge_sampling(train_dfs[d][i], ...)")
    print("  Line 622: test_dfs = {d: feature_engineers[d].extract_features(test_df) ...}")
    print("  Line 642: test_graphs = {d: build_client_graph(test_dfs[d], ...)}")
    print("VERDICT: Edge interpolation is strictly applied ONLY to local train_dfs post-split. test_df is NEVER interpolated.")

    print("\n" + "=" * 80)
    print("CHECK 2: Train/Test Flow Hash & Duplicate Leakage Audit")
    print("=" * 80)
    # Define flow hash tuple: src IP, dst IP, src port, dst port, protocol, duration, in_bytes, out_bytes, in_pkts, out_pkts
    hash_cols = ['Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol', 'Flow Duration', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Tot Fwd Pkts', 'Tot Bwd Pkts']
    # Check which exist
    hash_cols = [c for c in hash_cols if c in train_df.columns]
    print(f"Hashing based on columns: {hash_cols}")

    t_hash = time.time()
    def compute_flow_hashes(sub_df):
        # Create string representation of the tuple
        tuples = sub_df[hash_cols].astype(str).agg('|'.join, axis=1)
        # Hash each
        return set(tuples.values)

    print("Hashing training flows...")
    train_flow_set = compute_flow_hashes(train_df)
    print(f"Unique flow signatures in Train: {len(train_flow_set):,} / {len(train_df):,}")

    print("Checking test flows against training flow signatures...")
    test_tuples = test_df[hash_cols].astype(str).agg('|'.join, axis=1)
    overlap_mask = test_tuples.isin(train_flow_set)
    overlap_count = overlap_mask.sum()
    overlap_pct = (overlap_count / len(test_df)) * 100
    print(f"Exact Flow Tuple Overlap between Train and Test: {overlap_count:,} / {len(test_df):,} ({overlap_pct:.2f}%)")
    print(f"Hash audit completed in {time.time()-t_hash:.2f}s")

    if overlap_count > 0:
        print("\nWARNING: Repeated identical flow tuples exist across dataset rows in NetFlow!")
        print("Examining overlap by class:")
        overlap_classes = test_df.loc[overlap_mask, 'Attack'].value_counts()
        for cls, cnt in overlap_classes.items():
            print(f"  {cls:<15}: {cnt:>8,} ({cnt/len(test_df[test_df['Attack']==cls])*100:.2f}% of test class)")

    print("\n" + "=" * 80)
    print("CHECK 3: Per-Class FNR Analysis & Metric Audit")
    print("=" * 80)
    # Load previously reported V1 main results
    v1_main_json = 'kaggle_results_v1_main/v1_ablation_results.json'
    if os.path.exists(v1_main_json):
        import json
        with open(v1_main_json) as f:
            v1_res = json.load(f)
        print("Previously Reported Metrics (K=5):")
        print(f"  Aggregate Accuracy: {v1_res['K_5']['accuracy']*100:.2f}%")
        print(f"  Aggregate FNR:      {v1_res['K_5']['fnr']*100:.2f}%")
        print(f"  Aggregate FPR:      {v1_res['K_5']['fpr']*100:.2f}%")
        print("  Reported Per-Class F1:")
        for cls, val in v1_res['K_5']['per_class_f1'].items():
            print(f"    {cls:<12}: {val*100:.2f}%")

    print("\n" + "=" * 80)
    print("CHECK 4: Plain Unbalanced Baseline (No CB-Focal, No Edge Interpolation)")
    print("=" * 80)
    print("Running Fast Plain Baseline on identical split using raw features...")
    exclude = ['Src IP', 'Dst IP', 'Label', 'Attack', 'Timestamp', 'TCP_FLAGS', 'attack_lower']
    feat_cols = [c for c in train_df.columns if c not in exclude and np.issubdtype(train_df[c].dtype, np.number)]
    print(f"Evaluating {len(feat_cols)} raw features: {feat_cols}")

    # Subsample 50k train / 20k test for speed
    sub_tr = train_df.sample(min(100000, len(train_df)), random_state=42)
    sub_te = test_df.sample(min(50000, len(test_df)), random_state=42)

    y_tr = [CLASS_MAP[a.lower()] for a in sub_tr['Attack']]
    y_te = [CLASS_MAP[a.lower()] for a in sub_te['Attack']]
    X_tr = sub_tr[feat_cols].fillna(0).values
    X_te = sub_te[feat_cols].fillna(0).values

    # 1. Plain Unbalanced RF (default weights, no balancing, no interpolation)
    rf_plain = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
    rf_plain.fit(X_tr, y_tr)
    preds_plain = rf_plain.predict(X_te)

    acc_plain = accuracy_score(y_te, preds_plain)
    bal_acc_plain = balanced_accuracy_score(y_te, preds_plain)
    m_f1_plain = f1_score(y_te, preds_plain, average='macro')
    
    cm_plain = confusion_matrix(y_te, preds_plain, labels=list(range(8)))
    print(f"\nPlain Baseline Results:")
    print(f"  Accuracy:          {acc_plain*100:.2f}%")
    print(f"  Balanced Accuracy: {bal_acc_plain*100:.2f}%")
    print(f"  Macro F1:          {m_f1_plain*100:.2f}%")

    print("\nPer-Class FNR (Defined as 1 - Recall = FN / (FN + TP)):")
    labels_8 = PAPER_8_CLASSES
    rep_plain = classification_report(y_te, preds_plain, labels=list(range(8)), target_names=labels_8, output_dict=True, zero_division=0)
    for i, cls in enumerate(labels_8):
        rec = rep_plain[cls]['recall']
        fnr_cls = 1.0 - rec
        f1_cls = rep_plain[cls]['f1-score']
        support = rep_plain[cls]['support']
        flag = " [!] FNR < 5% on small class" if support < 1000 and fnr_cls < 0.05 else ""
        print(f"  {cls:<12} (Support: {support:>6}): Recall = {rec*100:>6.2f}%, FNR = {fnr_cls*100:>6.2f}%, F1 = {f1_cls*100:>6.2f}%{flag}")

    # Binary Intrusion FNR: Attack classified as Benign
    benign_idx = CLASS_MAP['benign']
    # False negatives in binary sense: True is Attack, Pred is Benign
    # In CM, row is true, col is pred.
    # Rows 1..7 (attacks), Col 0 (benign)
    binary_fn = cm_plain[1:, benign_idx].sum()
    total_attacks = cm_plain[1:, :].sum()
    binary_fnr = binary_fn / total_attacks if total_attacks > 0 else 0.0

    # What the V1 script computed as 'fnr':
    # fn = float(cm[benign_idx, :].sum() - cm[benign_idx, benign_idx])
    # tp = float(cm.sum() - fp - tn - fn)
    # fnr_script = fn / (fn + tp)
    script_fn = float(cm_plain[benign_idx, :].sum() - cm_plain[benign_idx, benign_idx])
    script_fp = float(cm_plain[:, benign_idx].sum() - cm_plain[benign_idx, benign_idx])
    script_tn = float(cm_plain.sum() - cm_plain[benign_idx, :].sum() - cm_plain[:, benign_idx].sum() + cm_plain[benign_idx, benign_idx])
    script_tp = float(cm_plain.sum() - script_fp - script_tn - script_fn)
    script_fnr = script_fn / (script_fn + script_tp) if (script_fn + script_tp) > 0 else 0.0

    print(f"\nMetric Formula Comparison:")
    print(f"  1. True Binary IDS FNR (True Attacks misclassified as Benign / Total Attacks): {binary_fnr*100:.2f}% ({binary_fn:,} / {total_attacks:,})")
    print(f"  2. Macro Multi-Class FNR (Mean of 1 - Recall across all 8 classes):           {np.mean([1.0 - rep_plain[cls]['recall'] for cls in labels_8])*100:.2f}%")
    print(f"  3. Base Paper Table 1 Reported FNR:                                           22.34%")
    print(f"  4. Formula used in V1 script (script_fn / (script_fn + script_tp)):           {script_fnr*100:.2f}%")

if __name__ == '__main__':
    main()
