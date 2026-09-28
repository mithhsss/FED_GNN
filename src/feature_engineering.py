"""
Feature engineering for FedGATSage specialized detectors.
Extracts community-aware features for temporal, content, and behavioral attack detection,
and computes real complex network centrality measures.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Handles the extraction of specialized features for each GAT detector.
    This ensures that the Temporal, Content, and Behavioral models each get the 
    data they need to excel at their specific tasks.
    """
    
    def __init__(self, detector_type: str = 'temporal'):
        self.detector_type = detector_type
        self.created_features = []
        
        # Attack groupings per paper Section 2 & 3
        self.temporal_attacks = ['ddos', 'dos', 'scanning']
        self.content_attacks = ['injection', 'xss'] 
        self.behavioral_attacks = ['password', 'backdoor', 'ransomware', 'mitm']
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point: takes raw data and adds the specialized columns 
        needed for the current detector type.
        """
        result_df = df.copy()
        
        # First, everyone gets the basics (flow rates, payload sizes)
        result_df = self._add_base_features(result_df)
        
        # Then we add the specialized features
        if self.detector_type == 'temporal':
            result_df = self._add_temporal_features(result_df)
        elif self.detector_type == 'content':
            result_df = self._add_content_features(result_df)
        elif self.detector_type == 'behavioral':
            result_df = self._add_behavioral_features(result_df)
            
        logger.info(f"Engineered features for {self.detector_type} detection. Columns now: {len(result_df.columns)}")
        return result_df
    
    def _add_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add base traffic features for all detector types"""
        # Flow rate features
        if 'Flow Duration' in df.columns and 'Tot Fwd Pkts' in df.columns:
            fwd_pkts = df['Tot Fwd Pkts'].fillna(0)
            bwd_pkts = df['Tot Bwd Pkts'].fillna(0) if 'Tot Bwd Pkts' in df.columns else 0
            duration = df['Flow Duration'].fillna(0)
            df['flow_rate'] = (fwd_pkts + bwd_pkts) / (duration / 1000000.0 + 1e-6)
            self.created_features.append('flow_rate')
        elif 'flow_rate' not in df.columns:
            df['flow_rate'] = 0.0
        
        # Payload size features
        if 'TotLen Fwd Pkts' in df.columns and 'Tot Fwd Pkts' in df.columns:
            tot_len_fwd = df['TotLen Fwd Pkts'].fillna(0)
            tot_fwd_pkts = df['Tot Fwd Pkts'].fillna(0)
            df['avg_payload_fwd'] = tot_len_fwd / (tot_fwd_pkts + 1e-6)
            self.created_features.append('avg_payload_fwd')
        elif 'avg_payload_fwd' not in df.columns:
            df['avg_payload_fwd'] = 0.0
            
        if 'TotLen Bwd Pkts' in df.columns and 'Tot Bwd Pkts' in df.columns:
            tot_len_bwd = df['TotLen Bwd Pkts'].fillna(0)
            tot_bwd_pkts = df['Tot Bwd Pkts'].fillna(0)
            df['avg_payload_bwd'] = tot_len_bwd / (tot_bwd_pkts + 1e-6)
            self.created_features.append('avg_payload_bwd')
        elif 'avg_payload_bwd' not in df.columns:
            df['avg_payload_bwd'] = 0.0
        
        # Protocol encoding
        if 'Protocol' in df.columns:
            df['protocol_encoded'] = pd.Categorical(df['Protocol']).codes.astype(float)
            self.created_features.append('protocol_encoded')
        elif 'protocol_encoded' not in df.columns:
            df['protocol_encoded'] = 0.0
            
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal attack-specific features (DDoS, DoS, Scanning)"""
        # Inter-arrival time features
        if 'Flow IAT Mean' in df.columns and 'Flow IAT Std' in df.columns:
            df['iat_variance'] = df['Flow IAT Std'] / (df['Flow IAT Mean'] + 1e-6)
            self.created_features.append('iat_variance')
        else:
            df['iat_variance'] = 0.0
        
        # Burst detection
        if 'Flow Pkts/s' in df.columns:
            mean_pps = df['Flow Pkts/s'].mean()
            df['burst_ratio'] = df['Flow Pkts/s'] / (mean_pps + 1e-6)
            df['is_burst'] = (df['burst_ratio'] > 2.0).astype(float)
            self.created_features.extend(['burst_ratio', 'is_burst'])
        else:
            df['burst_ratio'] = 0.0
            df['is_burst'] = 0.0
        
        # Flag patterns
        flag_cols = ['SYN Flag Cnt', 'RST Flag Cnt', 'ACK Flag Cnt']
        if all(col in df.columns for col in flag_cols):
            df['syn_rst_ratio'] = df['SYN Flag Cnt'] / (df['RST Flag Cnt'] + 1e-6)
            df['unusual_flags'] = ((df['SYN Flag Cnt'] > 0) & (df['RST Flag Cnt'] > 0)).astype(float)
            self.created_features.extend(['syn_rst_ratio', 'unusual_flags'])
        else:
            df['syn_rst_ratio'] = 0.0
            df['unusual_flags'] = 0.0
            
        return df
    
    def _add_content_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add content attack-specific features (Injection, XSS)"""
        # Port analysis
        if 'Dst Port' in df.columns:
            web_ports = [80, 443, 8080, 8443]
            db_ports = [1433, 1521, 3306, 5432]
            df['is_web_port'] = df['Dst Port'].isin(web_ports).astype(float)
            df['is_db_port'] = df['Dst Port'].isin(db_ports).astype(float)
            self.created_features.extend(['is_web_port', 'is_db_port'])
        else:
            df['is_web_port'] = 0.0
            df['is_db_port'] = 0.0
        
        # Payload size analysis
        if 'TotLen Fwd Pkts' in df.columns:
            mean_payload = df['TotLen Fwd Pkts'].mean()
            std_payload = df['TotLen Fwd Pkts'].std() if df['TotLen Fwd Pkts'].std() > 0 else 1.0
            df['unusual_payload'] = (df['TotLen Fwd Pkts'] > (mean_payload + 2 * std_payload)).astype(float)
            
            bwd_len = df['TotLen Bwd Pkts'] if 'TotLen Bwd Pkts' in df.columns else 1.0
            df['payload_ratio'] = df['TotLen Fwd Pkts'] / (bwd_len + 1e-6)
            self.created_features.extend(['unusual_payload', 'payload_ratio'])
        else:
            df['unusual_payload'] = 0.0
            df['payload_ratio'] = 0.0
            
        return df
    
    def _add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add behavioral attack-specific features (Password, Backdoor, Ransomware)"""
        # Connection pattern analysis
        if 'Src Port' in df.columns and 'Dst Port' in df.columns:
            df['is_ephemeral_src'] = (df['Src Port'] > 1024).astype(float)
            df['targets_system_port'] = (df['Dst Port'] < 1024).astype(float)
            df['port_spread'] = (df['Src Port'] - df['Dst Port']).abs() / 65535.0
            self.created_features.extend(['is_ephemeral_src', 'targets_system_port', 'port_spread'])
        else:
            df['is_ephemeral_src'] = 0.0
            df['targets_system_port'] = 0.0
            df['port_spread'] = 0.0
        
        # Session characteristics
        if 'Flow Duration' in df.columns:
            median_duration = df['Flow Duration'].median()
            df['is_short_session'] = (df['Flow Duration'] < median_duration / 10.0).astype(float)
            df['is_long_session'] = (df['Flow Duration'] > median_duration * 10.0).astype(float)
            self.created_features.extend(['is_short_session', 'is_long_session'])
        else:
            df['is_short_session'] = 0.0
            df['is_long_session'] = 0.0
        
        # Volume analysis
        if 'TotLen Fwd Pkts' in df.columns and 'Tot Fwd Pkts' in df.columns:
            df['is_low_volume'] = ((df['TotLen Fwd Pkts'] < 100) & (df['Tot Fwd Pkts'] < 5)).astype(float)
            self.created_features.append('is_low_volume')
        else:
            df['is_low_volume'] = 0.0
            
        return df

class CentralityFeatureExtractor:
    """
    Extract real complex network centrality features using NetworkX.
    Implements PageRank, Degree Centrality, Betweenness, Closeness,
    Eigenvector Centrality, and K-Core Decomposition.
    """
    
    def __init__(self):
        self.centrality_cache = {}
    
    def extract_centrality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute graph centrality features for each node and map them back to DataFrame flows.
        """
        result_df = df.copy()
        
        # Check if already present
        existing_cols = [c for c in result_df.columns if any(m in c.lower() for m in [
            'pagerank', 'betweenness', 'closeness', 'eigenvector', 'k_core'
        ])]
        if len(existing_cols) >= 6:
            logger.info(f"Centrality features already present: {existing_cols}")
            return result_df
            
        # Ensure Src IP and Dst IP exist
        if 'Src IP' not in result_df.columns or 'Dst IP' not in result_df.columns:
            logger.warning("Src IP / Dst IP not found for centrality calculation; skipping graph centralities.")
            return result_df
            
        # Build NetworkX graph from flows
        G = nx.Graph()
        for _, row in result_df.iterrows():
            src = str(row['Src IP'])
            dst = str(row['Dst IP'])
            if G.has_edge(src, dst):
                G[src][dst]['weight'] += 1
            else:
                G.add_edge(src, dst, weight=1)
                
        num_nodes = len(G.nodes)
        logger.info(f"Computing network centrality features for graph with {num_nodes} nodes and {len(G.edges)} edges...")
        
        # 1. Degree Centrality
        degree_dict = nx.degree_centrality(G)
        
        # 2. PageRank
        try:
            pagerank_dict = nx.pagerank(G, alpha=0.85, max_iter=200)
        except Exception as e:
            logger.warning(f"PageRank fallback: {e}")
            pagerank_dict = degree_dict
            
        # 3. Closeness Centrality
        try:
            closeness_dict = nx.closeness_centrality(G)
        except Exception as e:
            logger.warning(f"Closeness fallback: {e}")
            closeness_dict = {n: 0.0 for n in G.nodes}
            
        # 4. Betweenness Centrality (sample up to 100 nodes for fast computation)
        try:
            k_samples = min(num_nodes, 100)
            betweenness_dict = nx.betweenness_centrality(G, k=k_samples, normalized=True)
        except Exception as e:
            logger.warning(f"Betweenness fallback: {e}")
            betweenness_dict = {n: 0.0 for n in G.nodes}
            
        # 5. Eigenvector Centrality
        try:
            eigenvector_dict = nx.eigenvector_centrality(G, max_iter=500, tol=1e-4)
        except Exception:
            # Fallback to degree centrality if power iteration does not converge
            eigenvector_dict = degree_dict
            
        # 6. K-Core Decomposition
        try:
            k_core_dict = nx.core_number(G)
            # Normalize core number by max core
            max_core = max(k_core_dict.values()) if k_core_dict else 1
            k_core_norm = {k: v / max(max_core, 1) for k, v in k_core_dict.items()}
        except Exception as e:
            logger.warning(f"K-Core fallback: {e}")
            k_core_norm = {n: 0.0 for n in G.nodes}
            
        # Map back to DataFrame
        src_ips = result_df['Src IP'].astype(str)
        dst_ips = result_df['Dst IP'].astype(str)
        
        result_df['src_degree'] = src_ips.map(degree_dict).fillna(0.0)
        result_df['dst_degree'] = dst_ips.map(degree_dict).fillna(0.0)
        
        result_df['src_pagerank'] = src_ips.map(pagerank_dict).fillna(0.0)
        result_df['dst_pagerank'] = dst_ips.map(pagerank_dict).fillna(0.0)
        
        result_df['src_closeness'] = src_ips.map(closeness_dict).fillna(0.0)
        result_df['dst_closeness'] = dst_ips.map(closeness_dict).fillna(0.0)
        
        result_df['src_betweenness'] = src_ips.map(betweenness_dict).fillna(0.0)
        result_df['dst_betweenness'] = dst_ips.map(betweenness_dict).fillna(0.0)
        
        result_df['src_eigenvector'] = src_ips.map(eigenvector_dict).fillna(0.0)
        result_df['dst_eigenvector'] = dst_ips.map(eigenvector_dict).fillna(0.0)
        
        result_df['src_k_core'] = src_ips.map(k_core_norm).fillna(0.0)
        result_df['dst_k_core'] = dst_ips.map(k_core_norm).fillna(0.0)
        
        logger.info("Successfully added all 12 graph centrality features to flows.")
        return result_df
