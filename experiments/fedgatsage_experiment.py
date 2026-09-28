"""
Main experiment runner for FedGATSage.
Demonstrates the complete pipeline from data preprocessing to multi-detector federated training
and 2-stage ensemble evaluation with Random Forest.
"""

import os
import sys
import argparse
import time
from pathlib import Path
import subprocess
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from federated_learning import FedGATSageSystem
from ensemble import EnsembleFusion
from utils import (setup_logging, set_random_seeds, calculate_metrics, 
                   plot_confusion_matrix, plot_training_progress, save_results,
                   load_dataset_info, ExperimentTracker)
from community_detection import CommunityAwareProcessor
from dataset_utils import load_and_standardize_dataset

import logging
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='FedGATSage Experiment Runner')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to dataset directory')
    parser.add_argument('--input_file', type=str, default=None,
                       help='Path to raw input Parquet/CSV file')
    parser.add_argument('--dataset', type=str, choices=['nf_ton_iot', 'cic_ton_iot', 'custom'],
                       default='nf_ton_iot', help='Dataset to use')
    parser.add_argument('--num_clients', type=int, default=5,
                       help='Number of federated clients (default: 5)')
    parser.add_argument('--num_rounds', type=int, default=15,
                       help='Number of federation rounds (default: 15)')
    parser.add_argument('--local_epochs', type=int, default=5,
                       help='Number of local epochs per round (default: 5)')
    parser.add_argument('--detector_types', nargs='+', 
                       default=['temporal', 'content', 'behavioral'],
                       help='Specialized GAT detectors to train')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples for fast preprocessing/testing')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for metrics and plots')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--preprocess', action='store_true',
                       help='Force data preprocessing before training')
    return parser.parse_args()

def check_and_preprocess_data(args):
    """Ensure client partitions exist for all detector types"""
    data_ready = True
    for detector in args.detector_types:
        det_dir = os.path.join(args.data_dir, f'{detector}_detector')
        if not os.path.exists(det_dir):
            data_ready = False
            break
        client_files = [f for f in os.listdir(det_dir) if f.startswith('client_')]
        if len(client_files) < args.num_clients or not os.path.exists(os.path.join(det_dir, 'test.csv')):
            data_ready = False
            break
            
    if args.preprocess or not data_ready:
        logger.info("Client data not ready. Running preprocessing...")
        input_file = args.input_file
        if not input_file:
            # Check standard paths
            candidate_paths = [
                os.path.join(args.data_dir, 'nftoniot', 'NF-ToN-IoT.parquet'),
                os.path.join(args.data_dir, 'cic_ton_iot', 'CIC-ToN-IoT-V2.parquet'),
                'dummy_data.csv'
            ]
            for p in candidate_paths:
                if os.path.exists(p):
                    input_file = p
                    break
                    
        if not input_file or not os.path.exists(input_file):
            raise FileNotFoundError(f"Input dataset file could not be found: {input_file}")
            
        logger.info(f"Using dataset file: {input_file}")
        cmd = [
            sys.executable,
            'preprocess_data.py',
            '--input_file', input_file,
            '--output_dir', args.data_dir,
            '--num_clients', str(args.num_clients),
            '--seed', str(args.seed)
        ]
        if args.max_samples:
            cmd.extend(['--max_samples', str(args.max_samples)])
            
        subprocess.check_call(cmd)
        logger.info("Preprocessing completed.")

def evaluate_ensemble(fed_system: FedGATSageSystem, args) -> dict:
    """
    Perform 2-Stage Ensemble Evaluation matching Paper Section 3 & 4.
    Stage 1: Collect probability vectors from Temporal, Content, and Behavioral GAT models.
    Stage 2: Train & evaluate Random Forest meta-classifier on validation/test sets.
    """
    logger.info("\n=== Evaluating Multi-Detector Ensemble with Random Forest ===")
    
    # 1. Process test datasets for each detector type
    detector_test_probs = {}
    test_labels = None
    label_mapper = None
    
    for detector_type in fed_system.detector_types:
        test_loader = fed_system.data_loaders[detector_type]
        test_path = os.path.join(args.data_dir, f'{detector_type}_detector', 'test.csv')
        
        if not os.path.exists(test_path):
            logger.warning(f"Test file not found for {detector_type}")
            continue
            
        df_test = load_and_standardize_dataset(test_path)
        if args.max_samples and len(df_test) > (args.max_samples // 4):
            df_test = df_test.head(args.max_samples // 4)
            
        # Extract features and convert to graph
        df_test = test_loader.feature_engineer.extract_features(df_test)
        df_test = test_loader.centrality_extractor.extract_centrality_features(df_test)
        df_test = test_loader.community_processor.create_community_enhanced_features(df_test, {})
        
        if label_mapper is None and test_loader.label_mapper:
            label_mapper = test_loader.label_mapper
            
        graph_data = test_loader._process_to_graph(df_test)
        if graph_data is None or graph_data['edge_index'].shape[1] == 0:
            continue
            
        if test_labels is None:
            test_labels = graph_data['edge_labels'].cpu().numpy()
            
        # Use average of all clients' models for this detector
        client_models = list(fed_system.client_models[detector_type].values())
        
        with torch.no_grad():
            x = graph_data['features'].to(fed_system.device)
            edge_index = graph_data['edge_index'].to(fed_system.device)
            
            probs_list = []
            for m in client_models:
                m.eval()
                _, logits = m(x, edge_index)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs)
                
            avg_detector_probs = np.mean(probs_list, axis=0)
            detector_test_probs[detector_type] = avg_detector_probs
            
    if not detector_test_probs or test_labels is None:
        logger.error("Could not run ensemble evaluation due to missing test data.")
        return {}
        
    prob_arrays = [detector_test_probs[d] for d in fed_system.detector_types if d in detector_test_probs]
    
    # 2. Fit and evaluate Random Forest meta-classifier on train/test split of test set
    num_samples = len(test_labels)
    split_point = int(0.5 * num_samples) # 50% for fitting RF meta-classifier, 50% for unbiased test reporting
    
    train_probs = [p[:split_point] for p in prob_arrays]
    train_y = test_labels[:split_point]
    
    eval_probs = [p[split_point:] for p in prob_arrays]
    eval_y = test_labels[split_point:]
    
    if len(train_y) > 10 and len(np.unique(train_y)) > 1:
        fusion = EnsembleFusion(n_estimators=100, random_state=args.seed)
        fusion.fit(train_probs, train_y)
        final_preds = fusion.predict(eval_probs)
        y_true_final = eval_y
    else:
        # Fallback to simple ensemble mean
        mean_p = np.mean(prob_arrays, axis=0)
        final_preds = np.argmax(mean_p, axis=1)
        y_true_final = test_labels
        
    # Class names mapping
    class_names = None
    if label_mapper:
        sorted_items = sorted(label_mapper.items(), key=lambda x: x[1])
        class_names = [k for k, v in sorted_items]
        
    metrics = calculate_metrics(y_true_final, final_preds, class_names)
    
    # Add False Negative Rate (FNR) per class (matching Paper Table 2)
    fnr_dict = {}
    if 'per_class_detailed' in metrics:
        for cname, m in metrics['per_class_detailed'].items():
            recall = m.get('recall', 0.0)
            fnr = 1.0 - recall
            metrics['per_class_detailed'][cname]['fnr'] = float(fnr)
            fnr_dict[cname] = float(fnr)
        metrics['per_class_fnr'] = fnr_dict
        
    logger.info("=== ENSEMBLE EVALUATION RESULTS ===")
    logger.info(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy'] * 100:.2f}%")
    logger.info(f"Macro F1 Score: {metrics['macro_f1'] * 100:.2f}%")
    logger.info(f"Weighted F1 Score: {metrics['weighted_f1'] * 100:.2f}%")
    
    # Save Confusion Matrix
    cm_path = os.path.join(args.output_dir, 'ensemble_confusion_matrix.png')
    plot_confusion_matrix(y_true_final, final_preds, class_names, cm_path)
    
    return metrics

def run_experiment(args):
    """Run end-to-end experiment"""
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging('INFO', os.path.join(args.output_dir, 'experiment.log'))
    set_random_seeds(args.seed)
    
    device = 'cuda' if (args.device == 'auto' and torch.cuda.is_available()) or args.device == 'cuda' else 'cpu'
    logger.info(f"Running FedGATSage on device: {device}")
    
    check_and_preprocess_data(args)
    
    # Initialize Federated System
    fed_system = FedGATSageSystem(
        data_dir=args.data_dir,
        num_clients=args.num_clients,
        detector_types=args.detector_types,
        device=device
    )
    
    # Get number of classes from sample loader
    sample_loader = fed_system.data_loaders[args.detector_types[0]]
    sample_data = sample_loader.load_client_data(1)
    num_classes = len(sample_loader.label_mapper) if (sample_loader and sample_loader.label_mapper) else 8
    
    logger.info(f"Number of Attack Classes: {num_classes}")
    
    fed_system.initialize_models(
        hidden_dim=256,
        num_classes=num_classes
    )
    
    # Train
    train_results = fed_system.train_federated(num_rounds=args.num_rounds, local_epochs=args.local_epochs)
    
    # Evaluate
    eval_results = evaluate_ensemble(fed_system, args)
    
    # Save full results
    full_results = {
        'training': train_results,
        'evaluation': eval_results,
        'config': vars(args)
    }
    
    save_results(full_results, os.path.join(args.output_dir, 'experiment_results.json'))
    
    # Plot training loss & times
    if 'training_losses' in train_results and len(train_results['training_losses']) > 0:
        plot_training_progress(
            train_results['training_losses'],
            train_results['round_times'],
            os.path.join(args.output_dir, 'training_progress.png')
        )
        
    logger.info("Experiment successfully completed!")
    return full_results

if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)