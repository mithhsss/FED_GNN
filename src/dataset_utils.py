"""
Dataset Utilities for FedGATSage.
Handles loading, schema detection, column normalization, and format conversions
for NF-ToN-IoT, CIC-ToN-IoT, and synthetic datasets.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger(__name__)

# Column mappings for NF-ToN-IoT
NF_TON_IOT_MAPPINGS = {
    'L4_SRC_PORT': 'Src Port',
    'L4_DST_PORT': 'Dst Port',
    'PROTOCOL': 'Protocol',
    'L7_PROTO': 'L7 Proto',
    'IN_BYTES': 'TotLen Fwd Pkts',
    'OUT_BYTES': 'TotLen Bwd Pkts',
    'IN_PKTS': 'Tot Fwd Pkts',
    'OUT_PKTS': 'Tot Bwd Pkts',
    'TCP_FLAGS': 'TCP Flags',
    'FLOW_DURATION_MILLISECONDS': 'Flow Duration',
    'Label': 'Label',
    'Attack': 'Attack'
}

# Column mappings for CIC-ToN-IoT
CIC_TON_IOT_MAPPINGS = {
    'Protocol': 'Protocol',
    'Flow Duration': 'Flow Duration',
    'Total Fwd Packets': 'Tot Fwd Pkts',
    'Total Backward Packets': 'Tot Bwd Pkts',
    'Fwd Packets Length Total': 'TotLen Fwd Pkts',
    'Bwd Packets Length Total': 'TotLen Bwd Pkts',
    'Flow IAT Mean': 'Flow IAT Mean',
    'Flow IAT Std': 'Flow IAT Std',
    'Flow Packets/s': 'Flow Pkts/s',
    'Flow Bytes/s': 'Flow Bytes/s',
    'SYN Flag Count': 'SYN Flag Cnt',
    'RST Flag Count': 'RST Flag Cnt',
    'ACK Flag Count': 'ACK Flag Cnt',
    'Label': 'Label',
    'Attack': 'Attack'
}

def detect_dataset_format(df: pd.DataFrame) -> str:
    """Detect dataset schema format from DataFrame columns"""
    cols = set(df.columns)
    if 'L4_SRC_PORT' in cols and 'FLOW_DURATION_MILLISECONDS' in cols:
        return 'nf_ton_iot'
    elif 'Total Fwd Packets' in cols and 'Flow Duration' in cols:
        return 'cic_ton_iot'
    elif 'Src IP' in cols and 'Dst IP' in cols:
        return 'standard'
    else:
        return 'generic'

def load_and_standardize_dataset(file_path: str, max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Load dataset from CSV or Parquet and standardize column names.
    Synthesizes Src IP and Dst IP if not present using endpoints/ports.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    logger.info(f"Loading dataset from {file_path}")
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)
        
    if max_samples and len(df) > max_samples:
        logger.info(f"Subsampling dataset from {len(df)} to {max_samples} rows")
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        
    dataset_format = detect_dataset_format(df)
    logger.info(f"Detected dataset format: {dataset_format}")
    
    if dataset_format == 'nf_ton_iot':
        # Apply renaming
        rename_dict = {k: v for k, v in NF_TON_IOT_MAPPINGS.items() if k in df.columns}
        df = df.rename(columns=rename_dict)
        # Synthesize IP endpoints from port and protocol for graph nodes
        if 'Src IP' not in df.columns:
            df['Src IP'] = df.apply(lambda r: f"10.0.{int(r['Src Port']) % 254 + 1}.{(int(r['Protocol']) * 17) % 254 + 1}", axis=1)
        if 'Dst IP' not in df.columns:
            df['Dst IP'] = df.apply(lambda r: f"192.168.{int(r['Dst Port']) % 254 + 1}.{(int(r['Protocol']) * 23) % 254 + 1}", axis=1)
            
    elif dataset_format == 'cic_ton_iot':
        rename_dict = {k: v for k, v in CIC_TON_IOT_MAPPINGS.items() if k in df.columns}
        df = df.rename(columns=rename_dict)
        if 'Src IP' not in df.columns:
            # Create synthetic IP topologies based on protocol and flow characteristics
            df['Src IP'] = df.apply(lambda r: f"10.1.{(int(r.get('Protocol', 6)) * 31) % 254 + 1}.{(int(r.get('Tot Fwd Pkts', 1)) * 7) % 254 + 1}", axis=1)
        if 'Dst IP' not in df.columns:
            df['Dst IP'] = df.apply(lambda r: f"192.168.1.{(int(r.get('TotLen Fwd Pkts', 100)) * 13) % 254 + 1}", axis=1)
            
    # Ensure Attack column exists
    if 'Attack' not in df.columns and 'Label' in df.columns:
        df['Attack'] = df['Label'].apply(lambda x: 'Malicious' if x == 1 else 'Benign')
    elif 'Attack' in df.columns:
        # Standardize attack strings
        df['Attack'] = df['Attack'].astype(str).str.strip().str.lower()
        # Capitalize standard attacks
        mapping = {
            'benign': 'Benign',
            'ddos': 'DDoS',
            'dos': 'DoS',
            'scanning': 'Scanning',
            'injection': 'Injection',
            'xss': 'XSS',
            'password': 'Password',
            'backdoor': 'Backdoor',
            'mitm': 'MITM',
            'ransomware': 'Ransomware'
        }
        df['Attack'] = df['Attack'].map(lambda a: mapping.get(a, a.capitalize()))
        
    return df
