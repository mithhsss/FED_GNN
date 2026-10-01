import polars as pl
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import time

t0 = time.time()
path = 'data/nf_ton_unzipped/7ca78ae35fa4961a_MOHANAD_A4706/data/NF-ToN-IoT.csv'
print(f"Reading with Polars: {path}...", flush=True)
df_pl = pl.read_csv(path)
print(f"Read in {time.time()-t0:.2f}s. Rows: {df_pl.height:,}", flush=True)

CLASS_MAP = {
    'benign': 0, 'backdoor': 1, 'ddos': 2, 'dos': 3,
    'injection': 4, 'password': 5, 'scanning': 6, 'xss': 7
}

df_pl = df_pl.with_columns(pl.col('Attack').str.to_lowercase().alias('attack_lower'))
df_pl = df_pl.filter(pl.col('attack_lower').is_in(list(CLASS_MAP.keys())))

hash_cols = ['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS']

df_pl = df_pl.with_columns(
    pl.concat_str([pl.col(c).cast(pl.Utf8) for c in hash_cols], separator="|").alias("flow_signature"),
    pl.col("attack_lower").replace(CLASS_MAP).cast(pl.Int32).alias("label_idx")
)
print(f"Signatures built in {time.time()-t0:.2f}s. Unique signatures: {df_pl['flow_signature'].n_unique():,}", flush=True)

labels = df_pl['label_idx'].to_numpy()
attacks = df_pl['attack_lower'].to_numpy()

# In scikit-learn StratifiedGroupKFold:
# unique_groups, groups = np.unique(groups, return_inverse=True)
# We can do this with polars dense_rank / factorize which is 100x faster:
t1 = time.time()
print("Factoring groups via Polars...", flush=True)
# To match np.unique order exactly:
groups_cat = df_pl['flow_signature'].cast(pl.Categorical)
group_ids = groups_cat.to_physical().to_numpy()
print(f"Groups factored in {time.time()-t1:.2f}s", flush=True)

t2 = time.time()
print("Executing StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)...", flush=True)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(np.zeros(len(labels)), labels, groups=group_ids))
print(f"SGKF split completed in {time.time()-t2:.2f}s", flush=True)

train_attacks = attacks[train_idx]
test_attacks = attacks[test_idx]

print("\n" + "="*85, flush=True)
print("EXACT VERIFIED TABLE I NUMBERS (Directly from NF-ToN-IoT.csv on disk):", flush=True)
print("="*85, flush=True)
print(f"{'Class Name':<15} {'Total Flows':>12} {'Train Support':>15} {'Test Support':>15} {'Imbalance Ratio':>18}", flush=True)
print("-" * 85, flush=True)

tot_benign = np.sum(attacks == 'benign')

for c_name, c_idx in sorted(CLASS_MAP.items(), key=lambda x: x[1]):
    tot = int(np.sum(attacks == c_name))
    tr = int(np.sum(train_attacks == c_name))
    te = int(np.sum(test_attacks == c_name))
    ratio = f"{tot_benign / tot:.2f} : 1" if c_name != 'benign' else "1.00 : 1"
    print(f"{c_name.capitalize():<15} {tot:>12,} {tr:>15,} {te:>15,} {ratio:>18}", flush=True)

print("-" * 85, flush=True)
print(f"{'Total':<15} {len(attacks):>12,} {len(train_attacks):>15,} {len(test_attacks):>15,}", flush=True)
print("="*85, flush=True)
