import pandas as pd
import numpy as np

path = 'data/nf_ton_unzipped/7ca78ae35fa4961a_MOHANAD_A4706/data/NF-ToN-IoT.csv'
raw_df = pd.read_csv(path, nrows=50)

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

df = map_nf_toniot_columns(raw_df)

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dur_ms = df['Flow Duration'].fillna(0) + 1.0
    df['Pkt_Rate'] = (df['Tot Fwd Pkts'] + df['Tot Bwd Pkts']) / (dur_ms / 1000.0)
    df['Byte_Rate'] = (df['TotLen Fwd Pkts'] + df['TotLen Bwd Pkts']) / (dur_ms / 1000.0)
    df['Fwd_Pkt_Ratio'] = df['Tot Fwd Pkts'] / (df['Tot Fwd Pkts'] + df['Tot Bwd Pkts'] + 1e-5)
    df['Duration_Log'] = np.log1p(np.maximum(df['Flow Duration'].fillna(0), 0.0))
    return df

def add_content_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tot_pkts = df['Tot Fwd Pkts'] + df['Tot Bwd Pkts'] + 1e-5
    tot_bytes = df['TotLen Fwd Pkts'] + df['TotLen Bwd Pkts'] + 1e-5
    df['Avg_Pkt_Size'] = tot_bytes / tot_pkts
    df['Byte_Asymmetry'] = (df['TotLen Fwd Pkts'] - df['TotLen Bwd Pkts']) / tot_bytes
    df['Pkt_Asymmetry'] = (df['Tot Fwd Pkts'] - df['Tot Bwd Pkts']) / tot_pkts
    df['SYN_to_Pkt'] = df['SYN Flag Cnt'] / tot_pkts
    df['ACK_to_Pkt'] = df['ACK Flag Cnt'] / tot_pkts
    return df

def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Is_Privileged_Port'] = ((df['Src Port'] < 1024) | (df['Dst Port'] < 1024)).astype(float)
    df['Is_Ephemeral_Src'] = (df['Src Port'] >= 49152).astype(float)
    df['Port_Spread'] = np.abs(df['Src Port'] - df['Dst Port'])
    df['TCP_Flag_Sum'] = (df['FIN Flag Cnt'] + df['SYN Flag Cnt'] + df['RST Flag Cnt'] +
                          df['PSH Flag Cnt'] + df['ACK Flag Cnt'] + df['URG Flag Cnt'])
    return df

exclude = ['Src IP', 'Dst IP', 'Label', 'Attack', 'Timestamp', 'TCP_FLAGS', 'attack_lower', 'flow_signature']

t_df = add_temporal_features(df)
t_cols = [c for c in t_df.columns if c not in exclude and np.issubdtype(t_df[c].dtype, np.number)]
print(f"Temporal features ({len(t_cols)}): {t_cols}")

c_df = add_content_features(df)
c_cols = [c for c in c_df.columns if c not in exclude and np.issubdtype(c_df[c].dtype, np.number)]
print(f"Content features ({len(c_cols)}): {c_cols}")

b_df = add_behavioral_features(df)
b_cols = [c for c in b_df.columns if c not in exclude and np.issubdtype(b_df[c].dtype, np.number)]
print(f"Behavioral features ({len(b_cols)}): {b_cols}")

base_numeric = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
print(f"Base mapped numeric features ({len(base_numeric)}): {base_numeric}")
