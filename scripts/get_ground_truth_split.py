import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

path = 'data/nf_ton_unzipped/7ca78ae35fa4961a_MOHANAD_A4706/data/NF-ToN-IoT.csv'
print(f"Reading {path}...")
cols = ['IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 'PROTOCOL', 'FLOW_DURATION_MILLISECONDS', 'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'Attack']
df = pd.read_csv(path, usecols=cols)

CLASS_MAP = {
    'benign': 0, 'backdoor': 1, 'ddos': 2, 'dos': 3,
    'injection': 4, 'password': 5, 'scanning': 6, 'xss': 7
}

df['attack_lower'] = df['Attack'].astype(str).str.lower()
df = df[df['attack_lower'].isin(CLASS_MAP)].copy().reset_index(drop=True)
df['label_idx'] = df['attack_lower'].map(CLASS_MAP)

mapping = {
    'IPV4_SRC_ADDR': 'Src IP', 'IPV4_DST_ADDR': 'Dst IP',
    'L4_SRC_PORT': 'Src Port', 'L4_DST_PORT': 'Dst Port',
    'PROTOCOL': 'Protocol', 'FLOW_DURATION_MILLISECONDS': 'Flow Duration',
    'IN_BYTES': 'TotLen Fwd Pkts', 'OUT_BYTES': 'TotLen Bwd Pkts',
    'IN_PKTS': 'Tot Fwd Pkts', 'OUT_PKTS': 'Tot Bwd Pkts',
}
df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

hash_cols = ['Src Port', 'Dst Port', 'Protocol', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'Flow Duration']
df['flow_signature'] = df[hash_cols].astype(str).agg('|'.join, axis=1)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(df, df['label_idx'], groups=df['flow_signature']))

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

print("\n" + "="*85)
print("GROUND TRUTH CLASS COUNTS FROM NF-ToN-IoT.csv & SGKF(seed=42):")
print("="*85)
print(f"{'Class':<12} {'Total Flows':>12} {'Train Support':>15} {'Test Support':>15} {'Test %':>10} {'Imbalance (Maj:Min)':>22}")
print("-" * 85)

tot_benign = (df['attack_lower'] == 'benign').sum()
for c in ['benign', 'backdoor', 'ddos', 'dos', 'injection', 'password', 'scanning', 'xss']:
    tot = int((df['attack_lower'] == c).sum())
    tr = int((train_df['attack_lower'] == c).sum())
    te = int((test_df['attack_lower'] == c).sum())
    pct = te / tot * 100.0
    ratio = f"{tot_benign / tot:.2f} : 1" if tot > 0 else "1.00 : 1"
    print(f"{c.capitalize():<12} {tot:>12,} {tr:>15,} {te:>15,} {pct:>9.2f}% {ratio:>22}")

print("-" * 85)
print(f"{'Total':<12} {len(df):>12,} {len(train_df):>15,} {len(test_df):>15,} {len(test_df)/len(df)*100:>9.2f}%")
print("="*85)
