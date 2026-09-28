"""
Data Preprocessing Script for FedGATSage
========================================
Prepares raw IoT network traffic datasets (NF-ToN-IoT, CIC-ToN-IoT, or CSV/Parquet)
for federated GNN experiments across 5 simulated clients.
Implements IID and Non-IID heterogeneous partitioning (Paper Section 4).
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))
from dataset_utils import load_and_standardize_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='FedGATSage Data Preprocessing')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to the raw CSV or Parquet dataset')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Directory to save processed client data')
    parser.add_argument('--num_clients', type=int, default=5,
                       help='Number of federated clients (default: 5)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='Ratio of data to use for testing (default: 0.2)')
    parser.add_argument('--non_iid', action='store_true',
                       help='Use non-IID partitioning across clients')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Optional maximum number of samples to load/subsample')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    return parser.parse_args()

def partition_data_non_iid(df: pd.DataFrame, num_clients: int, seed: int = 42) -> list:
    """
    Partition dataset into non-IID client splits with size variations (up to 40%)
    and class distribution skews (up to 25%) as described in the paper.
    """
    np.random.seed(seed)
    client_dfs = [[] for _ in range(num_clients)]
    
    # Group by attack class and distribute via Dirichlet
    unique_classes = df['Attack'].unique()
    alpha = np.ones(num_clients) * 1.5  # Moderate skew
    
    for cls in unique_classes:
        cls_df = df[df['Attack'] == cls]
        proportions = np.random.dirichlet(alpha)
        counts = (proportions * len(cls_df)).astype(int)
        # Ensure all samples are assigned
        counts[-1] = len(cls_df) - counts[:-1].sum()
        
        start = 0
        for i, count in enumerate(counts):
            if count > 0:
                client_dfs[i].append(cls_df.iloc[start:start + count])
                start += count
                
    result = []
    for i, parts in enumerate(client_dfs):
        if parts:
            c_df = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed + i).reset_index(drop=True)
            result.append(c_df)
        else:
            result.append(pd.DataFrame(columns=df.columns))
            
    return result

def save_split_data(df: pd.DataFrame, output_dir: str, num_clients: int, test_ratio: float = 0.2, non_iid: bool = False, seed: int = 42):
    """Split data into test set and client-specific partitions"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Stratified or standard train/test split
    stratify_col = df['Attack'] if 'Attack' in df.columns and df['Attack'].nunique() > 1 else None
    train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=seed, stratify=stratify_col)
    
    # Save test set
    test_path = os.path.join(output_dir, 'test.csv')
    test_df.to_csv(test_path, index=False)
    logger.info(f"Saved test set to {test_path} ({len(test_df)} records)")
    
    # Partition training data among clients
    if non_iid:
        client_dfs = partition_data_non_iid(train_df, num_clients, seed=seed)
    else:
        client_dfs = np.array_split(train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True), num_clients)
        
    for i, client_df in enumerate(client_dfs):
        client_id = i + 1
        client_path = os.path.join(output_dir, f'client_{client_id}.csv')
        client_df.to_csv(client_path, index=False)
        logger.info(f"Saved client {client_id} data to {client_path} ({len(client_df)} records)")

def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    logger.info(f"Loading and standardizing dataset from: {args.input_file}")
    df = load_and_standardize_dataset(args.input_file, max_samples=args.max_samples)
    logger.info(f"Dataset successfully loaded. Shape: {df.shape}")
    logger.info(f"Attack classes: {dict(df['Attack'].value_counts())}")
    
    detector_types = ['temporal', 'content', 'behavioral']
    for detector in detector_types:
        detector_dir = os.path.join(args.output_dir, f'{detector}_detector')
        logger.info(f"Partitioning data for {detector} detector at {detector_dir}...")
        save_split_data(df, detector_dir, args.num_clients, test_ratio=args.test_ratio, non_iid=args.non_iid, seed=args.seed)
        
    logger.info("Data preprocessing completed successfully.")

if __name__ == '__main__':
    main()
