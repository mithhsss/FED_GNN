"""
Dataset-specific column mappers for FedGATSage.
Provides isolated, explicitly named mapping functions for NF-ToN-IoT and CIC-ToN-IoT.
Includes bitwise TCP flag decoding and unit-test validation.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def map_nf_toniot_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explicit column mapper for official NF-ToN-IoT (Mohanad Sarhan NetFlow v1 release).
    - Maps genuine IPv4 endpoints: IPV4_SRC_ADDR -> Src IP, IPV4_DST_ADDR -> Dst IP
    - Maps ports: L4_SRC_PORT -> Src Port, L4_DST_PORT -> Dst Port
    - Maps bytes/packets to forward/backward equivalents
    - Performs bitwise RFC 793 decoding of TCP_FLAGS into SYN, RST, ACK, FIN, PSH, URG counts
    - Computes Flow Pkts/s and Flow Bytes/s from duration and counts
    """
    df = df.copy()
    
    # 1. Direct column renames
    mapping = {
        'IPV4_SRC_ADDR': 'Src IP',
        'IPV4_DST_ADDR': 'Dst IP',
        'L4_SRC_PORT': 'Src Port',
        'L4_DST_PORT': 'Dst Port',
        'PROTOCOL': 'Protocol',
        'FLOW_DURATION_MILLISECONDS': 'Flow Duration',
        'IN_BYTES': 'TotLen Fwd Pkts',
        'OUT_BYTES': 'TotLen Bwd Pkts',
        'IN_PKTS': 'Tot Fwd Pkts',
        'OUT_PKTS': 'Tot Bwd Pkts',
    }
    df = df.rename(columns=mapping)
    
    # 2. Bitwise TCP_FLAGS extraction (RFC 793 standard bitmask)
    if 'TCP_FLAGS' in df.columns:
        tcp_flags = df['TCP_FLAGS'].fillna(0).astype(int)
        df['FIN Flag Cnt'] = ((tcp_flags & 0x01) > 0).astype(float)
        df['SYN Flag Cnt'] = ((tcp_flags & 0x02) > 0).astype(float)
        df['RST Flag Cnt'] = ((tcp_flags & 0x04) > 0).astype(float)
        df['PSH Flag Cnt'] = ((tcp_flags & 0x08) > 0).astype(float)
        df['ACK Flag Cnt'] = ((tcp_flags & 0x10) > 0).astype(float)
        df['URG Flag Cnt'] = ((tcp_flags & 0x20) > 0).astype(float)
    else:
        for flg in ['FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt']:
            df[flg] = 0.0
            
    # 3. Flow rates (Duration in milliseconds -> seconds)
    dur_sec = df['Flow Duration'].fillna(0) / 1000.0 + 1e-6
    df['Flow Pkts/s'] = (df['Tot Fwd Pkts'].fillna(0) + df['Tot Bwd Pkts'].fillna(0)) / dur_sec
    df['Flow Bytes/s'] = (df['TotLen Fwd Pkts'].fillna(0) + df['TotLen Bwd Pkts'].fillna(0)) / dur_sec
    
    # Assertions
    required_cols = ['Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol',
                     'Flow Duration', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
                     'Tot Fwd Pkts', 'Tot Bwd Pkts', 'SYN Flag Cnt', 'RST Flag Cnt',
                     'ACK Flag Cnt', 'Flow Pkts/s', 'Flow Bytes/s']
    for col in required_cols:
        assert col in df.columns, f"NF mapping failed: missing '{col}'"
        
    return df


def map_cic_toniot_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dedicated column mapper for official CIC-ToN-IoT (Mohanad Sarhan release).
    - Maps/normalizes CICFlowMeter columns:
      * 'Flow Byts/s' -> 'Flow Bytes/s'
      * Handles alternative header naming conventions if present
    - Robust 99.9th-percentile infinity clipping on rate features to preserve
      discriminative high-intensity attack signatures (DDoS/DoS) without numerical explosion
    - Fills NaNs with 0.0
    - Validates presence of essential graph & detector features
    """
    df = df.copy()

    # 1. Alternative naming synonyms
    synonym_map = {
        'Flow Byts/s': 'Flow Bytes/s',
        'Total Fwd Packets': 'Tot Fwd Pkts',
        'Total Backward Packets': 'Tot Bwd Pkts',
        'Fwd Packets Length Total': 'TotLen Fwd Pkts',
        'Bwd Packets Length Total': 'TotLen Bwd Pkts',
        'SYN Flag Count': 'SYN Flag Cnt',
        'RST Flag Count': 'RST Flag Cnt',
        'ACK Flag Count': 'ACK Flag Cnt',
        'FIN Flag Count': 'FIN Flag Cnt',
        'PSH Flag Count': 'PSH Flag Cnt',
        'URG Flag Count': 'URG Flag Cnt',
        'Flow Packets/s': 'Flow Pkts/s',
    }
    df = df.rename(columns={k: v for k, v in synonym_map.items() if k in df.columns})

    # If 'Flow Bytes/s' not yet present but 'Flow Byts/s' exists
    if 'Flow Bytes/s' not in df.columns and 'Flow Byts/s' in df.columns:
        df['Flow Bytes/s'] = df['Flow Byts/s']

    # 2. Flow rate infinity and NaN clipping (B6/B7 research finding)
    rate_cols = ['Flow Bytes/s', 'Flow Pkts/s', 'Fwd Pkts/s', 'Bwd Pkts/s']
    for col in rate_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce')
            s = s.replace([np.inf, -np.inf], np.nan)
            finite = s.dropna()
            if len(finite) > 0:
                hi = np.percentile(finite, 99.9)
                lo = np.percentile(finite, 0.1)
                s = s.clip(lower=lo, upper=hi)
            df[col] = s.fillna(0.0)

    # 3. Required columns validation
    required_cols = [
        'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol',
        'Flow Duration', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
        'Tot Fwd Pkts', 'Tot Bwd Pkts', 'SYN Flag Cnt', 'RST Flag Cnt',
        'ACK Flag Cnt', 'Flow Pkts/s', 'Flow Bytes/s'
    ]
    for col in required_cols:
        assert col in df.columns, f"CIC mapping error: missing required column '{col}'"

    return df

