import os
import sys
import time
import re
import pandas as pd
import numpy as np
from collections import Counter

csv_path = r'c:\Users\Mithul\Desktop\final_year_base\data\cic_ton_unzipped\a40a412453292fe6_MOHANAD_A4706\data\CIC-ToN-IoT.csv'

print("=" * 80)
print("CIC-ToN-IoT COMPREHENSIVE DATASET PROFILING & PREPROCESSING REPORT")
print("=" * 80)

t0 = time.time()

# 1. First Pass: Read header & inspect schema
head = pd.read_csv(csv_path, nrows=100)
cols = list(head.columns)
print(f"Total Column Count: {len(cols)}")
print(f"File Size on Disk: {os.path.getsize(csv_path)/(1024*1024):.2f} MB (~{os.path.getsize(csv_path)/(1024*1024*1024):.2f} GB)")

# 2. Chunked processing for full dataset statistics
print("\nScanning dataset in chunks of 500,000 rows...")
total_rows = 0
attack_counts = Counter()
label_counts = Counter()
null_counts = {c: 0 for c in cols}
inf_counts = {'Flow Byts/s': 0, 'Flow Pkts/s': 0, 'Fwd Pkts/s': 0, 'Bwd Pkts/s': 0}
sample_ips_src = set()
sample_ips_dst = set()
protocols = Counter()

ipv4_regex = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')

for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=500000, low_memory=False)):
    nrows = len(chunk)
    total_rows += nrows
    
    # Class counts
    attack_counts.update(chunk['Attack'].astype(str).tolist())
    if 'Label' in chunk.columns:
        label_counts.update(chunk['Label'].tolist())
    if 'Protocol' in chunk.columns:
        protocols.update(chunk['Protocol'].dropna().tolist())

    # Null counts
    for c in cols:
        null_counts[c] += chunk[c].isna().sum()

    # Inf counts
    for ic in inf_counts:
        if ic in chunk.columns:
            s = pd.to_numeric(chunk[ic], errors='coerce')
            inf_counts[ic] += np.isinf(s.values).sum()

    # IP sample check
    if len(sample_ips_src) < 1000:
        sample_ips_src.update(chunk['Src IP'].dropna().astype(str).unique()[:200])
    if len(sample_ips_dst) < 1000:
        sample_ips_dst.update(chunk['Dst IP'].dropna().astype(str).unique()[:200])

    print(f"  Processed {total_rows:,} rows... ({time.time()-t0:.1f}s)")

elapsed = time.time() - t0
print(f"\nScan completed in {elapsed:.2f} seconds.")
print(f"TOTAL ROWS: {total_rows:,}")
print(f"TOTAL COLUMNS: {len(cols)}")

# IP validation
real_src_ips = sum(1 for ip in sample_ips_src if ipv4_regex.match(ip))
real_dst_ips = sum(1 for ip in sample_ips_dst if ipv4_regex.match(ip))
pct_real_src = (real_src_ips / len(sample_ips_src) * 100) if sample_ips_src else 0
pct_real_dst = (real_dst_ips / len(sample_ips_dst) * 100) if sample_ips_dst else 0

print("\n" + "=" * 80)
print("1. IP ENDPOINTS & GRAPH TOPOLOGY VERIFICATION")
print("=" * 80)
print(f"Source IP Regex Valid IPv4:      {pct_real_src:.2f}% ({real_src_ips}/{len(sample_ips_src)} sampled)")
print(f"Destination IP Regex Valid IPv4: {pct_real_dst:.2f}% ({real_dst_ips}/{len(sample_ips_dst)} sampled)")
print("Sample Src IPs:", list(sample_ips_src)[:5])
print("Sample Dst IPs:", list(sample_ips_dst)[:5])

print("\n" + "=" * 80)
print("2. CLASS DISTRIBUTION (Attack Column)")
print("=" * 80)
for k, v in sorted(attack_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {k:<20}: {v:>10,} ({v/total_rows*100:>6.2f}%)")

print("\n" + "=" * 80)
print("3. BINARY LABEL DISTRIBUTION (Label Column)")
print("=" * 80)
for k, v in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
    lbl_str = "Benign (0)" if k == 0 else ("Malicious (1)" if k == 1 else str(k))
    print(f"  {lbl_str:<20}: {v:>10,} ({v/total_rows*100:>6.2f}%)")

print("\n" + "=" * 80)
print("4. PROTOCOL DISTRIBUTION")
print("=" * 80)
proto_names = {6: 'TCP (6)', 17: 'UDP (17)', 0: 'HOPOPT (0)', 1: 'ICMP (1)'}
for k, v in sorted(protocols.items(), key=lambda x: x[1], reverse=True):
    p_name = proto_names.get(int(k), f'Proto {k}') if str(k).replace('.','').isdigit() else str(k)
    print(f"  {p_name:<20}: {v:>10,} ({v/total_rows*100:>6.2f}%)")

print("\n" + "=" * 80)
print("5. DATA QUALITY: NULL & INFINITY AUDIT")
print("=" * 80)
null_cols = {c: v for c, v in null_counts.items() if v > 0}
if null_cols:
    print(f"Columns with Nulls ({len(null_cols)}):")
    for c, v in null_cols.items():
        print(f"  {c:<30}: {v:>10,} ({v/total_rows*100:.2f}%)")
else:
    print("Zero null values across all 85 columns!")

print("\nInfinity Values Found in Flow Rate Columns:")
for c, v in inf_counts.items():
    print(f"  {c:<30}: {v:>10,} ({v/total_rows*100:.4f}%)")

print("\n" + "=" * 80)
print("6. COLUMN NAMES & SYSTEM SCHEMA CLASSIFICATION")
print("=" * 80)
for i, col in enumerate(cols, 1):
    dtype = str(head[col].dtype)
    ex_val = str(head[col].iloc[0])
    if len(ex_val) > 25:
        ex_val = ex_val[:22] + "..."
    print(f"{i:>2}. {col:<32} | Dtype: {dtype:<10} | Sample: {ex_val}")

print("\n" + "=" * 80)
print("REPORT COMPLETE")
print("=" * 80)
