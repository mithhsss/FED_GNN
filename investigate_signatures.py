import pandas as pd
import numpy as np

p_csv = 'data/nf_ton_unzipped/7ca78ae35fa4961a_MOHANAD_A4706/data/NF-ToN-IoT.csv'
p_pq = 'data/nftoniot/NF-ToN-IoT.parquet'

print("Loading CSV and Parquet...", flush=True)
df_csv = pd.read_csv(p_csv)
df_pq = pd.read_parquet(p_pq)

PAPER_8 = ['benign', 'backdoor', 'ddos', 'dos', 'injection', 'password', 'scanning', 'xss']
df_csv = df_csv[df_csv['Attack'].astype(str).str.lower().isin(PAPER_8)].copy()
df_pq = df_pq[df_pq['Attack'].astype(str).str.lower().isin(PAPER_8)].copy()

cols = ['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS']

df_csv['sig'] = df_csv[cols].astype(str).agg('|'.join, axis=1)
df_pq['sig'] = df_pq[cols].astype(str).agg('|'.join, axis=1)

csv_sigs = set(df_csv['sig'].unique())
pq_sigs = set(df_pq['sig'].unique())

print(f"CSV Total 8-Class Rows: {len(df_csv):,}", flush=True)
print(f"PQ Total 8-Class Rows:  {len(df_pq):,}", flush=True)
print(f"Extra Rows in CSV:      {len(df_csv) - len(df_pq):,}", flush=True)
print(f"CSV Unique Signatures:  {len(csv_sigs):,}", flush=True)
print(f"PQ Unique Signatures:   {len(pq_sigs):,}", flush=True)
print(f"Sigs in PQ but not in CSV: {len(pq_sigs - csv_sigs):,}", flush=True)
print(f"Sigs in CSV but not in PQ: {len(csv_sigs - pq_sigs):,}", flush=True)
print(f"Are signature sets 100% IDENTICAL? {csv_sigs == pq_sigs}", flush=True)

# Class-by-class analysis
print("\n" + "=" * 90, flush=True)
print(f"{'Class':<12} {'Official (CSV)':>15} {'Mirror (PQ)':>15} {'Extra Rows':>14} {'CSV Sigs':>12} {'PQ Sigs':>12} {'Extra Sigs':>12}", flush=True)
print("-" * 90, flush=True)

for cls in ['Benign', 'Backdoor', 'DDoS', 'DoS', 'Injection', 'Password', 'Scanning', 'XSS']:
    c = cls.lower()
    sub_csv = df_csv[df_csv['Attack'].astype(str).str.lower() == c]
    sub_pq = df_pq[df_pq['Attack'].astype(str).str.lower() == c]
    extra_rows = len(sub_csv) - len(sub_pq)
    csv_s = set(sub_csv['sig'].unique())
    pq_s = set(sub_pq['sig'].unique())
    extra_sigs = len(csv_s - pq_s)
    print(f"{cls:<12} {len(sub_csv):>15,} {len(sub_pq):>15,} {extra_rows:>14,} {len(csv_s):>12,} {len(pq_s):>12,} {extra_sigs:>12,}", flush=True)

# Check if dhoogla dropped exact duplicate rows based on feature columns
print("\n" + "=" * 90, flush=True)
print("INVESTIGATING WHY DHOOGLA DROPPED 221,273 ROWS:", flush=True)
print("=" * 90, flush=True)
pq_feature_cols = [c for c in df_pq.columns if c not in ['Label', 'Attack', 'sig']]
print(f"Feature columns in PQ: {pq_feature_cols}", flush=True)

csv_dups = df_csv.duplicated(subset=pq_feature_cols + ['Attack']).sum()
print(f"Official CSV: Duplicates across all 10 feature cols + Attack: {csv_dups:,}", flush=True)

csv_deduped = df_csv.drop_duplicates(subset=pq_feature_cols + ['Attack'])
print(f"Official CSV after deduplication on (10 features + Attack): {len(csv_deduped):,} rows", flush=True)
print(f"Mirror PQ length: {len(df_pq):,} rows", flush=True)
print(f"Difference: {len(csv_deduped) - len(df_pq):,} rows", flush=True)
