import os
import sys

# Run a quick smoke test on 5000 samples for 1 round to verify graph building, SupCon, Leiden, and RF
os.environ['ABLATION_MODE'] = 'all'

# Import from kaggle_kernel_v_a.version_a_supcon_kaggle
sys.path.insert(0, os.path.abspath('kaggle_kernel_v_a'))
from version_a_supcon_kaggle import *

def smoke_test():
    print("=" * 80)
    print("LOCAL SMOKE TEST: 1 ROUND ON SUBSET")
    print("=" * 80)
    data_path = 'data/nftoniot/NF-ToN-IoT.parquet'
    df = pd.read_parquet(data_path)
    df = map_nf_toniot_columns(df)
    df['attack_lower'] = df['Attack'].astype(str).str.lower()
    df = df[df['attack_lower'].isin(CLASS_MAP)].copy().reset_index(drop=True)
    df['label_idx'] = df['attack_lower'].map(CLASS_MAP)

    sub_df = df.sample(10000, random_state=42).reset_index(drop=True)
    hash_cols = ['Src Port', 'Dst Port', 'Protocol', 'Flow Duration',
                 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Tot Fwd Pkts', 'Tot Bwd Pkts']
    sub_df['flow_signature'] = sub_df[hash_cols].astype(str).agg('|'.join, axis=1)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(sub_df, sub_df['label_idx'], groups=sub_df['flow_signature']))
    tr_df = sub_df.iloc[train_idx].reset_index(drop=True)
    te_df = sub_df.iloc[test_idx].reset_index(drop=True)

    print(f"Smoke test train: {len(tr_df):,} | test: {len(te_df):,}")
    res = run_version_a_experiment(tr_df, te_df, mu_supcon=0.3, num_clients=2, num_rounds=1, local_epochs=1)
    print("\nSMOKE TEST COMPLETED SUCCESSFULLY!")
    print(f"Accuracy: {res['accuracy']*100:.2f}%, Macro FNR: {res['macro_fnr']*100:.2f}%")

if __name__ == '__main__':
    smoke_test()
