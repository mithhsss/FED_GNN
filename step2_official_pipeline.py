import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score, f1_score

PAPER_8_CLASSES = ['Benign', 'Backdoor', 'DDoS', 'DoS', 'Injection', 'Password', 'Scanning', 'XSS']
CLASS_MAP = {c.lower(): i for i, c in enumerate(PAPER_8_CLASSES)}
INV_CLASS_MAP = {i: c for i, c in enumerate(PAPER_8_CLASSES)}

def compute_corrected_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names=PAPER_8_CLASSES):
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    benign_idx = CLASS_MAP['benign']

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    total_samples = len(y_true)
    total_benign = float(cm[benign_idx, :].sum())
    total_attacks = float(total_samples - total_benign)

    fn_binary = float(cm[:, benign_idx].sum() - cm[benign_idx, benign_idx])
    fp_binary = float(cm[benign_idx, :].sum() - cm[benign_idx, benign_idx])
    tp_binary = float(total_attacks - fn_binary)
    tn_binary = float(cm[benign_idx, benign_idx])

    binary_fnr = fn_binary / total_attacks if total_attacks > 0 else 0.0
    binary_fpr = fp_binary / total_benign if total_benign > 0 else 0.0

    rep = classification_report(y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
    per_class = {}
    per_class_fnr_list = []
    for cls in class_names:
        rec = rep[cls]['recall']
        fnr_k = 1.0 - rec
        f1_k = rep[cls]['f1-score']
        supp = rep[cls]['support']
        per_class[cls] = {
            'recall': rec,
            'fnr': fnr_k,
            'f1': f1_k,
            'support': supp
        }
        per_class_fnr_list.append(fnr_k)

    macro_fnr = float(np.mean(per_class_fnr_list))

    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'macro_fnr': macro_fnr,
        'binary_fnr': binary_fnr,
        'binary_fpr': binary_fpr,
        'fn_binary': fn_binary,
        'fp_binary': fp_binary,
        'tp_binary': tp_binary,
        'tn_binary': tn_binary,
        'total_attacks': total_attacks,
        'total_benign': total_benign,
        'per_class': per_class,
        'confusion_matrix': cm
    }

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
    print("=" * 85, flush=True)
    print("STEP 2: OFFICIAL MOHANAD SARHAN NF-ToN-IoT PIPELINE", flush=True)
    print("=" * 85, flush=True)

    official_path = 'data/nf_ton_unzipped/7ca78ae35fa4961a_MOHANAD_A4706/data/NF-ToN-IoT.csv'
    print(f"Loading official dataset from: {official_path}", flush=True)
    t0 = time.time()
    raw_df = pd.read_csv(official_path)
    print(f"Loaded {len(raw_df):,} raw flows in {time.time()-t0:.2f}s", flush=True)
    assert len(raw_df) == 1379274, f"Row count mismatch! Expected 1,379,274, got {len(raw_df)}"

    print("\nFirst 5 Sample Rows of Real Network IPs:", flush=True)
    print(raw_df[['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT', 'Attack']].head(5), flush=True)

    df = map_nf_toniot_columns(raw_df)
    df['attack_lower'] = df['Attack'].astype(str).str.lower()
    df = df[df['attack_lower'].isin(CLASS_MAP)].copy().reset_index(drop=True)
    df['label_idx'] = df['attack_lower'].map(CLASS_MAP)
    print(f"\nCleaned 8-Class Dataset: {len(df):,} flows (excluded MITM and Ransomware)", flush=True)

    print("\nClass Value Counts on Official Dataset:")
    counts = df['Attack'].value_counts()
    for cls, cnt in counts.items():
        print(f"  {cls:<15}: {cnt:>10,} ({cnt/len(df)*100:.2f}%)", flush=True)

    # Rebuild flow-signature hash from NF columns
    hash_cols = ['Src Port', 'Dst Port', 'Protocol', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'Flow Duration']
    print(f"\nDefining flow signature hash on 8 NF columns: {hash_cols}", flush=True)
    t_sig = time.time()
    df['flow_signature'] = df[hash_cols].astype(str).agg('|'.join, axis=1)
    unique_sigs = df['flow_signature'].nunique()
    print(f"Unique Flow Signatures: {unique_sigs:,} out of {len(df):,} total flows ({time.time()-t_sig:.2f}s)", flush=True)

    # StratifiedGroupKFold on flow signatures
    print("\nExecuting StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)...", flush=True)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(df, df['label_idx'], groups=df['flow_signature']))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    print(f"Grouped Split Created:", flush=True)
    print(f"  Training Flows:   {len(train_df):,} ({len(train_df)/len(df)*100:.2f}%)", flush=True)
    print(f"  Evaluation Flows: {len(test_df):,} ({len(test_df)/len(df)*100:.2f}%)", flush=True)

    # 1. Flow Signature Leakage Audit
    print("\n" + "=" * 85, flush=True)
    print("LEAKAGE AUDIT 1: FLOW SIGNATURE OVERLAP (Zero Tolerance)", flush=True)
    print("=" * 85, flush=True)
    train_sig_set = set(train_df['flow_signature'].unique())
    test_sig_set = set(test_df['flow_signature'].unique())
    sig_overlap = train_sig_set.intersection(test_sig_set)
    sig_overlap_count = test_df['flow_signature'].isin(train_sig_set).sum()

    print(f"  Train Unique Signatures: {len(train_sig_set):,}", flush=True)
    print(f"  Test Unique Signatures:  {len(test_sig_set):,}", flush=True)
    print(f"  Overlapping Signatures:  {len(sig_overlap):,}", flush=True)
    print(f"  Exact Matching Flows in Test: {sig_overlap_count:,} / {len(test_df):,} ({sig_overlap_count/len(test_df)*100:.4f}%)", flush=True)
    assert sig_overlap_count == 0, "FATAL: Flow signature leakage detected!"
    print("  --> RESULT: 0.0000% MATCH. ZERO SIGNATURE LEAKAGE.", flush=True)

    # 2. Source IP Overlap Analysis
    print("\n" + "=" * 85, flush=True)
    print("LEAKAGE AUDIT 2: SOURCE IP (Src IP) DISTRIBUTION ACROSS SPLITS", flush=True)
    print("=" * 85, flush=True)
    train_src_ips = set(train_df['Src IP'].unique())
    test_src_ips = set(test_df['Src IP'].unique())
    ip_overlap = train_src_ips.intersection(test_src_ips)
    all_ips = train_src_ips.union(test_src_ips)

    print(f"  Total Unique Source IPs:     {len(all_ips):,}", flush=True)
    print(f"  Train Unique Source IPs:     {len(train_src_ips):,}", flush=True)
    print(f"  Test Unique Source IPs:      {len(test_src_ips):,}", flush=True)
    print(f"  Source IPs in BOTH Splits:   {len(ip_overlap):,} ({len(ip_overlap)/len(all_ips)*100:.2f}% of unique Src IPs)", flush=True)
    print(f"  Source IPs ONLY in Train:    {len(train_src_ips - test_src_ips):,}", flush=True)
    print(f"  Source IPs ONLY in Test:     {len(test_src_ips - train_src_ips):,}", flush=True)
    print("  Note: Shared source IP nodes represent network infrastructure endpoints (e.g. gateways, testbed servers)", flush=True)
    print("        where flow signatures are distinct and strictly disjoint between splits.", flush=True)

    # Per-Class Counts
    print("\n" + "=" * 85, flush=True)
    print("PER-CLASS FLOW COUNTS & SIGNATURES (Official Dataset Grouped Split)", flush=True)
    print("=" * 85, flush=True)
    print(f"{'Class':<12} {'Full Data':>12} {'Train Flows':>14} {'Test Flows':>14} {'Test Sigs':>12}", flush=True)
    print("-" * 70, flush=True)
    for cls in PAPER_8_CLASSES:
        c_lower = cls.lower()
        full_cnt = (df['attack_lower'] == c_lower).sum()
        tr_cnt = (train_df['attack_lower'] == c_lower).sum()
        te_cnt = (test_df['attack_lower'] == c_lower).sum()
        te_sigs = test_df[test_df['attack_lower'] == c_lower]['flow_signature'].nunique()
        print(f"{cls:<12} {full_cnt:>12,} {tr_cnt:>14,} {te_cnt:>14,} {te_sigs:>12,}", flush=True)

    # Plain Unbalanced Baseline Evaluation
    print("\n" + "=" * 85, flush=True)
    print("NEW TRUSTWORTHY ANCHOR BASELINE (Official Release, Full Grouped Test Set)", flush=True)
    print("=" * 85, flush=True)

    exclude = ['Src IP', 'Dst IP', 'Label', 'Attack', 'Timestamp', 'TCP_FLAGS', 'attack_lower', 'label_idx', 'flow_signature']
    feat_cols = [c for c in train_df.columns if c not in exclude and np.issubdtype(train_df[c].dtype, np.number)]
    print(f"Evaluating {len(feat_cols)} features: {feat_cols}", flush=True)

    # Fit plain unweighted RF on representative stratified sample (300,000 flows) for speed/memory
    train_sample = train_df.groupby('label_idx', group_keys=False).apply(
        lambda x: x.sample(min(len(x), max(1000, int(len(x) * 0.3))), random_state=42)
    ).reset_index(drop=True)
    print(f"Fitting on {len(train_sample):,} representative training flows...", flush=True)

    X_train = train_sample[feat_cols].fillna(0).values
    y_train = train_sample['label_idx'].values

    # Test on 100% of the full grouped test set
    print(f"Predicting on 100% of the Full Grouped Test Set ({len(test_df):,} flows)...", flush=True)
    X_test = test_df[feat_cols].fillna(0).values
    y_test = test_df['label_idx'].values

    t_rf = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    print(f"Fitted in {time.time()-t_rf:.2f}s", flush=True)

    preds = rf.predict(X_test)
    res = compute_corrected_metrics(y_test, preds, class_names=PAPER_8_CLASSES)

    print("\n" + "=" * 85, flush=True)
    print("ANCHOR BASELINE METRICS (Official Sarhan Release, Zero Leakage)", flush=True)
    print("=" * 85, flush=True)
    print(f"Overall Accuracy:                  {res['accuracy']*100:.2f}%", flush=True)
    print(f"Balanced Accuracy:                 {res['balanced_accuracy']*100:.2f}%", flush=True)
    print(f"Macro F1 Score:                    {res['macro_f1']*100:.2f}%", flush=True)
    print(f"Weighted F1 Score:                 {res['weighted_f1']*100:.2f}%", flush=True)
    print(f"Macro Per-Class FNR (Paper Metric):{res['macro_fnr']*100:.2f}%  (Base Paper NF-ToN-IoT Target: 22.34%)", flush=True)
    print(f"Binary IDS FNR (Attacks -> Benign):{res['binary_fnr']*100:.2f}%  ({int(res['fn_binary']):,} / {int(res['total_attacks']):,} missed)", flush=True)
    print(f"Binary IDS FPR (Benign -> Attack): {res['binary_fpr']*100:.2f}%  ({int(res['fp_binary']):,} / {int(res['total_benign']):,} false alarms)", flush=True)

    print("\nPER-CLASS PERFORMANCE BREAKDOWN (New Trustworthy Anchor Table):", flush=True)
    print(f"{'Class':<12} {'Support':>10} {'Recall':>10} {'FNR (1-Rec)':>14} {'F1-Score':>12} {'Status / Flag':<25}", flush=True)
    print("-" * 85, flush=True)
    for cls in PAPER_8_CLASSES:
        p = res['per_class'][cls]
        supp = p['support']
        rec = p['recall'] * 100
        fnr = p['fnr'] * 100
        f1 = p['f1'] * 100
        flag = " [!] FNR < 5% on small class" if supp < 1000 and fnr < 5.0 else ""
        if fnr > 90.0:
            flag = " [!] Class Representation Collapsed"
        print(f"{cls:<12} {supp:>10,} {rec:>9.2f}% {fnr:>13.2f}% {f1:>11.2f}% {flag:<25}", flush=True)
    print("=" * 85, flush=True)

if __name__ == '__main__':
    main()
