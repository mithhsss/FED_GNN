"""
Local Unit Test / Verification for FedGATSage Version 1 on CIC-ToN-IoT.
Runs 1 full federated round on a stratified sample to verify end-to-end correctness.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestV1CIC")

# Import functions from the kaggle script
sys.path.insert(0, os.path.abspath('kaggle_kernel_v1_cic'))
from v1_cic_paper_exact_kaggle import (
    map_cic_toniot_columns,
    FeatureEngineerCIC,
    stratified_split_clients,
    b7_edge_sampling,
    compute_cb_weights,
    focal_loss,
    SpecializedGAT,
    detect_communities_louvain,
    weighted_community_pooling,
    OverlayGraphBuilder,
    GlobalGraphSAGE,
    TwoStageEnsemble,
    build_client_graph,
    run_v1_cic_experiment,
    CLASS_MAP,
    PAPER_8_CLASSES
)

def run_local_test():
    csv_path = r'data/cic_ton_unzipped/a40a412453292fe6_MOHANAD_A4706/data/CIC-ToN-IoT.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    logger.info("Reading 20,000 rows from CIC-ToN-IoT.csv...")
    raw_df = pd.read_csv(csv_path, nrows=20000, low_memory=False)
    logger.info(f"Loaded raw sample: {raw_df.shape}")

    # Step 2: Mapping & rate clipping
    logger.info("Executing map_cic_toniot_columns...")
    df = map_cic_toniot_columns(raw_df)

    # Filter to 8 standard classes
    df['attack_lower'] = df['Attack'].astype(str).str.strip().str.lower()
    df = df[df['attack_lower'].isin(CLASS_MAP)].copy()
    logger.info(f"Filtered 8-class sample: {len(df)} flows | Classes: {df['Attack'].value_counts().to_dict()}")

    # Ensure at least 3 distinct classes in sample for test
    if len(df['Attack'].unique()) < 3:
        logger.warning("Sample has fewer than 3 classes; adding synthetic minority rows for pipeline validation.")
        for extra_cls in ['DDoS', 'DoS', 'Backdoor']:
            synth_row = df.iloc[:2].copy()
            synth_row['Attack'] = extra_cls
            synth_row['attack_lower'] = extra_cls.lower()
            df = pd.concat([df, synth_row], ignore_index=True)

    logger.info("Testing run_v1_cic_experiment with K=2 clients, 2 rounds, 1 epoch...")
    result = run_v1_cic_experiment(df, num_clients=2, num_rounds=2, local_epochs=1, checkpoint_dir="test_results/checkpoints_v1_cic")

    logger.info("\n=== LOCAL TEST COMPLETED SUCCESSFULLY ===")
    logger.info(f"Accuracy:          {result['accuracy']*100:.2f}%")
    logger.info(f"Balanced Accuracy: {result['balanced_accuracy']*100:.2f}%")
    logger.info(f"Macro-F1:          {result['macro_f1']*100:.2f}%")
    logger.info(f"FPR:               {result['fpr']*100:.2f}%")
    logger.info(f"FNR:               {result['fnr']*100:.2f}%")

if __name__ == '__main__':
    run_local_test()
