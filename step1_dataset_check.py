import os
import sys
import pandas as pd

def inspect_file(filepath):
    print("=" * 80)
    print(f"INSPECTING: {filepath}")
    print("=" * 80)
    if not os.path.exists(filepath):
        print("  [-] File does not exist!")
        return None

    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)

    print(f"Total Rows: {len(df):,}")
    print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")

    print("\nClass Value Counts ('Attack' column):")
    if 'Attack' in df.columns:
        print(df['Attack'].value_counts(dropna=False))
    elif 'attack' in df.columns:
        print(df['attack'].value_counts(dropna=False))

    print("\nSample 5 Rows of IP addresses:")
    src_col = None
    dst_col = None
    for c in ['IPV4_SRC_ADDR', 'Src IP', 'src_ip', 'IPV4_SRC']:
        if c in df.columns:
            src_col = c
            break
    for c in ['IPV4_DST_ADDR', 'Dst IP', 'dst_ip', 'IPV4_DST']:
        if c in df.columns:
            dst_col = c
            break

    print(f"  Source IP Column:      {src_col}")
    print(f"  Destination IP Column: {dst_col}")
    if src_col and dst_col:
        print(df[[src_col, dst_col]].head(5))
    else:
        print("  [!] IP COLUMNS ARE MISSING FROM THIS FILE!")
        print("  Showing first 5 rows of available columns:")
        print(df.iloc[:5, :6])

    return df

if __name__ == '__main__':
    # 1. Inspect local parquet (old dhoogla mirror)
    inspect_file('data/nftoniot/NF-ToN-IoT.parquet')

    # 2. Inspect official Sarhan unzipped release
    inspect_file('data/nf_ton_unzipped/7ca78ae35fa4961a_MOHANAD_A4706/data/NF-ToN-IoT.csv')
