"""
Ensemble Fusion Module for FedGATSage.
Implements the 2-stage fusion process (Paper Section: Inference & Server Processing):
Stage 1: Softmax probability averaging across specialized GAT detectors.
Stage 2: Random Forest Meta-Classifier trained on concatenated detector probabilities,
predicted labels, confidence scores, and attack-type priority weights.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import logging

logger = logging.getLogger(__name__)

class EnsembleFusion:
    """
    Two-stage ensemble fusion combining Temporal, Content, and Behavioral GAT detectors
    with a Random Forest meta-learner.
    """
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.rf_classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        
    def extract_meta_features(self, detector_prob_list: List[np.ndarray]) -> np.ndarray:
        """
        Extract meta-features from individual detector softmax probabilities:
        - Raw probabilities from each detector
        - Argmax predicted class per detector
        - Maximum confidence per detector
        - Simple ensemble mean probability
        
        Args:
            detector_prob_list: List of numpy arrays, each of shape (N, num_classes)
            
        Returns:
            meta_features: Array of shape (N, meta_dim)
        """
        features = []
        for probs in detector_prob_list:
            preds = np.argmax(probs, axis=1, keepdims=True)
            confidences = np.max(probs, axis=1, keepdims=True)
            features.extend([probs, preds, confidences])
            
        # Add ensemble mean
        avg_probs = np.mean(detector_prob_list, axis=0)
        features.append(avg_probs)
        
        return np.concatenate(features, axis=1)
        
    def fit(self, detector_prob_list: List[np.ndarray], y_true: np.ndarray):
        """Fit Random Forest on meta-features"""
        meta_features = self.extract_meta_features(detector_prob_list)
        logger.info(f"Training Random Forest ensemble on {meta_features.shape[0]} samples with {meta_features.shape[1]} meta-features...")
        self.rf_classifier.fit(meta_features, y_true)
        self.is_fitted = True
        logger.info("Random Forest ensemble fitted successfully.")
        
    def predict(self, detector_prob_list: List[np.ndarray]) -> np.ndarray:
        """
        Predict final labels using the Random Forest classifier.
        Falls back to argmax of averaged probabilities if RF is not fitted.
        """
        if self.is_fitted:
            meta_features = self.extract_meta_features(detector_prob_list)
            return self.rf_classifier.predict(meta_features)
        else:
            # Fallback: simple ensemble average
            avg_probs = np.mean(detector_prob_list, axis=0)
            return np.argmax(avg_probs, axis=1)
            
    def predict_proba(self, detector_prob_list: List[np.ndarray]) -> np.ndarray:
        """Predict probabilities from ensemble"""
        if self.is_fitted:
            meta_features = self.extract_meta_features(detector_prob_list)
            return self.rf_classifier.predict_proba(meta_features)
        else:
            return np.mean(detector_prob_list, axis=0)
