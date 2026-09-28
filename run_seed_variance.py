import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold

official_path = 'data/nf_ton_unzipped/7ca78ae35fa4961a_MOHANAD_A4706/data/NF-ToN-IoT.csv'
df = pd.read_csv(official_path)

mapping = {
    'IPV4_SRC_ADDR': 'Src IP', 'IPV4_DST_ADDR': 'Dst IP',
    'L4_SRC_PORT': 'Src Port', 'L4_DST_PORT': 'Dst Port',
    'PROTOCOL': 'Protocol', 'FLOW_DURATION_MILLISECONDS': 'Flow Duration',
    'IN_BYTES': 'TotLen Fwd Pkts', 'OUT_BYTES': 'TotLen Bwd Pkts',
    'IN_PKTS': 'Tot Fwd Pkts', 'OUT_PKTS': 'Tot Bwd Pkts',
}
df = df.rename(columns=mapping)
PAPER_8 = ['benign', 'backdoor', 'ddos', 'dos', 'injection', 'password', 'scanning', 'xss']
df['att'] = df['Attack'].astype(str).str.lower()
df = df[df['att'].isin(PAPER_8)].copy().reset_index(drop=True)
label_map = {c: i for i, c in enumerate(PAPER_8)}
df['label_idx'] = df['att'].map(label_map)

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

feat_cols = ['Src Port', 'Dst Port', 'Protocol', 'L7_PROTO', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
             'Tot Fwd Pkts', 'Tot Bwd Pkts', 'Flow Duration', 'FIN Flag Cnt', 'SYN Flag Cnt',
             'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'Flow Pkts/s', 'Flow Bytes/s']

hash_cols = ['Src Port', 'Dst Port', 'Protocol', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'Flow Duration']
df['flow_signature'] = df[hash_cols].astype(str).agg('|'.join, axis=1)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
tr_idx, te_idx = next(sgkf.split(df, df['label_idx'], groups=df['flow_signature']))

train_df = df.iloc[tr_idx]
test_df = df.iloc[te_idx]

seeds = [42, 43, 44]
print("=== EVALUATING MULTI-SEED VARIANCE (SEEDS 42, 43, 44) ===")

# Test Baseline (Anchor) across seeds
anchor_results = []
for s in seeds:
    idx_sample = np.random.RandomState(s).choice(len(train_df), size=min(200000, len(train_df)), replace=False)
    X_tr = train_df.iloc[idx_sample][feat_cols].fillna(0).values
    y_tr = train_df.iloc[idx_sample]['label_idx'].values
    X_te = test_df[feat_cols].fillna(0).values
    y_te = test_df['label_idx'].values

    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=s, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    preds = rf.predict(X_te)

    bal_acc = balanced_accuracy_score(y_te, preds)
    mf1 = f1_score(y_te, preds, average='macro', zero_division=0)
    rep = classification_report(y_te, preds, labels=list(range(8)), target_names=PAPER_8, output_dict=True, zero_division=0)

    anchor_results.append({
        'seed': s,
        'bal_acc': bal_acc,
        'macro_f1': mf1,
        'backdoor_rec': rep['backdoor']['recall'],
        'scanning_rec': rep['scanning']['recall'],
        'password_rec': rep['password']['recall'],
    })
    print(f"Anchor Seed {s}: BalAcc={bal_acc*100:.2f}%, MacroF1={mf1*100:.2f}%, Backdoor={rep['backdoor']['recall']*100:.2f}%, Scanning={rep['scanning']['recall']*100:.2f}%, Password={rep['password']['recall']*100:.2f}%")

print("\n--- Summary of Anchor Across Seeds 42, 43, 44 ---")
bal_accs = [r['bal_acc']*100 for r in anchor_results]
mf1s = [r['macro_f1']*100 for r in anchor_results]
bd_recs = [r['backdoor_rec']*100 for r in anchor_results]
sc_recs = [r['scanning_rec']*100 for r in anchor_results]
pw_recs = [r['password_rec']*100 for r in anchor_results]

print(f"Balanced Acc: {np.mean(bal_accs):.2f}% +/- {np.std(bal_accs):.2f}%")
print(f"Macro F1:     {np.mean(mf1s):.2f}% +/- {np.std(mf1s):.2f}%")
print(f"Backdoor:     {np.mean(bd_recs):.2f}% +/- {np.std(bd_recs):.2f}%")
print(f"Scanning:     {np.mean(sc_recs):.2f}% +/- {np.std(sc_recs):.2f}%")
print(f"Password:     {np.mean(pw_recs):.2f}% +/- {np.std(pw_recs):.2f}%")
