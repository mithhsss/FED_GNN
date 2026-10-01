import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

path = 'data/nf_ton_unzipped/7ca78ae35fa4961a_MOHANAD_A4706/data/NF-ToN-IoT.csv'
print(f"Reading {path}...")
df = pd.read_csv(path)
print(f"Total rows: {len(df):,}")

# Exact filtering used in official pipeline
CLASS_MAP = {
    'benign': 0, 'backdoor': 1, 'ddos': 2, 'dos': 3,
    'injection': 4, 'password': 5, 'scanning': 6, 'xss': 7
}

df['attack_lower'] = df['Attack'].astype(str).str.lower()
print("Raw class counts in NF-ToN-IoT.csv:")
print(df['attack_lower'].value_counts())

df = df[df['attack_lower'].isin(CLASS_MAP)].copy().reset_index(drop=True)
df['label_idx'] = df['attack_lower'].map(CLASS_MAP)
print(f"\nFiltered 8-class dataset total rows: {len(df):,}")

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

print("\n" + "="*80)
print("EXACT VERIFIED CLASS COUNTS AND SPLIT SUPPORT:")
print("="*80)
print(f"{'Class Name':<15} {'Total Flows':>12} {'Train Support':>15} {'Test Support':>15} {'Imbalance Ratio':>18}")
print("-" * 80)

inv_map = {v: k for k, v in CLASS_MAP.items()}
tot_benign = (df['attack_lower'] == 'benign').sum()

for c_idx in range(8):
    c_name = inv_map[c_idx].capitalize()
    tot = int((df['attack_lower'] == inv_map[c_idx]).sum())
    tr = int((train_df['attack_lower'] == inv_map[c_idx]).sum())
    te = int((test_df['attack_lower'] == inv_map[c_idx]).sum())
    ratio = f"{tot_benign / tot:.2f} : 1" if tot > 0 else "N/A"
    print(f"{c_name:<15} {tot:>12,} {tr:>15,} {te:>15,} {ratio:>18}")

print("-" * 80)
print(f"{'Total':<15} {len(df):>12,} {len(train_df):>15,} {len(test_df):>15,}")
print("="*80)
