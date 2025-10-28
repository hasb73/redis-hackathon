#!/usr/bin/env python3
"""
HDFS Anomaly-Only Loader for Stress Testing
Loads only anomalous HDFS log entries using SAME labeling logic as hdfs_line_level_loader.py
and streams them to Kafka for testing the accuracy of the anomaly detection engine
"""
import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional
import re
import time
import random

class HDFSAnomalyOnlyLoader:
    def __init__(self, base_path="HDFS_v1"):
        self.base_path = base_path
        self.parsed_path = os.path.join(base_path, "parsed")
        self.preprocessed_path = os.path.join(base_path, "labels")
        
        print("Loading HDFS Anomaly-Only Dataset for Stress Testing (using original labeling logic)...")
        
        # Load structured logs
        self.structured_df = pd.read_csv(os.path.join(self.parsed_path, "HDFS.log_structured.csv"))
        self.templates_df = pd.read_csv(os.path.join(self.parsed_path, "HDFS.log_templates.csv"))
        self.labels_df = pd.read_csv(os.path.join(self.preprocessed_path, "anomaly_label.csv"))
        
        # Create block label mapping
        self.block_labels = dict(zip(self.labels_df['BlockId'], self.labels_df['Label']))
        
        print(f"Loaded HDFS dataset:")
        print(f"  - {len(self.structured_df):,} total log lines")
        print(f"  - {len(self.templates_df)} event templates")
        print(f"  - {len(self.labels_df):,} labeled blocks")
        
        # Filter for anomalies only
        self._prepare_anomaly_only_data()
        
    def _extract_block_id(self, param_list_str: str) -> Optional[str]:
        """Extract block ID from parameter list"""
        if pd.isna(param_list_str):
            return None
        try:
            params = eval(param_list_str)
            for param in params:
                if isinstance(param, str) and param.startswith('blk_'):
                    block_id = param.split()[0]
                    if block_id.startswith('blk_'):
                        return block_id
        except:
            pass
        return None
    
    def _prepare_anomaly_only_data(self):
        """Prepare anomaly-only dataset for stress testing"""
        print("Filtering for anomalous log entries only...")
        
        # Extract block IDs from structured logs
        self.structured_df['BlockId'] = self.structured_df['ParameterList'].apply(self._extract_block_id)
        
        # Map block labels to individual lines
        self.structured_df['BlockLabel'] = self.structured_df['BlockId'].map(self.block_labels)
        
        # Filter only lines from ANOMALOUS blocks
        anomaly_mask = (
            (self.structured_df['BlockId'].notna()) & 
            (self.structured_df['BlockLabel'] == 'Anomaly')
        )
        self.anomaly_lines = self.structured_df[anomaly_mask].copy()
        
        print(f"  - Found {len(self.anomaly_lines):,} lines from anomalous blocks")
        
        # Create sophisticated line-level anomaly labels for stress testing
        self._create_line_level_anomaly_labels()
        
    def _create_line_level_anomaly_labels(self):
        """Create sophisticated line-level anomaly labels using SAME logic as hdfs_line_level_loader"""
        print("Creating line-level anomaly labels using EXACT logic from hdfs_line_level_loader...")
        
        # Initialize all lines as normal
        self.anomaly_lines['LineLabel'] = 0  # 0 = Normal
        
        # Load event templates for analysis
        template_map = dict(zip(self.templates_df['EventId'], self.templates_df['EventTemplate']))
        
        # Define anomaly-indicative event patterns (EXACT SAME as hdfs_line_level_loader)
        anomaly_patterns = [
            # Core exception/error patterns
            'exception', 'error', 'fatal', 'fail', 'failure',
            
            # Network and I/O issues (very common in HDFS)
            'timeout', 'sockettimeoutexception', 'ioexception', 'eofexception',
            'connection reset', 'broken pipe', 'no route to host', 'noroutetohostexception',
            'interruptedioexception', 'closedbyinterruptexception', 'stream is closed',
            
            # Block and data integrity issues
            'corrupt', 'corrupted', 'checksum', 'not found', 
            'does not belong', 'blockinfo not found', 'cannot be written',
            
            # Access and permission issues
            'denied', 'refused', 'unauthorized', 'permission',
            
            # System resource and capacity issues
            'unable', 'cannot', 'insufficient', 'quota exceeded', 'disk full',
            'out of space', 'resource', 'capacity',
            
            # Process and thread issues
            'interrupted', 'abort', 'terminate', 'killed', 'dead', 'hang',
            'deadlock', 'thread', 'process',
            
            # Network connectivity
            'disconnect', 'lost', 'unreachable', 'refused connection',
            'network', 'socket', 'channel closed',
            
            # HDFS-specific operations that indicate problems
            'transfer failed', 'replication failed', 'delete failed',
            'write failed', 'read failed', 'receive failed',
            'mirror failed', 'serving failed',
            
            # Critical system states
            'critical', 'severe', 'emergency', 'panic', 'crash'
        ]
        
        # Event types that are likely anomalous
        anomaly_event_keywords = {}
        for event_id, template in template_map.items():
            template_lower = template.lower()
            is_anomaly_event = any(pattern in template_lower for pattern in anomaly_patterns)
            anomaly_event_keywords[event_id] = is_anomaly_event
        
        print(f"Identified {sum(anomaly_event_keywords.values())} potentially anomalous event types")
        
        # Strategy 1: Lines from anomalous blocks with anomaly-indicative events
        anomaly_blocks_mask = self.anomaly_lines['BlockLabel'] == 'Anomaly'
        anomaly_event_mask = self.anomaly_lines['EventId'].map(anomaly_event_keywords).fillna(False)
        
        strategy1_mask = anomaly_blocks_mask & anomaly_event_mask
        self.anomaly_lines.loc[strategy1_mask, 'LineLabel'] = 1
        
        # Strategy 2: Rare events in anomalous blocks (statistical approach)
        event_counts = self.templates_df.set_index('EventId')['Occurrences'].to_dict()
        rare_threshold = np.percentile(list(event_counts.values()), 10)  # Bottom 10% of events
        
        rare_event_mask = self.anomaly_lines['EventId'].map(
            lambda x: event_counts.get(x, 0) < rare_threshold
        )
        
        strategy2_mask = anomaly_blocks_mask & rare_event_mask
        self.anomaly_lines.loc[strategy2_mask, 'LineLabel'] = 1
        
        # Report results
        line_anomalies = (self.anomaly_lines['LineLabel'] == 1).sum()
        line_normals = (self.anomaly_lines['LineLabel'] == 0).sum()
        total_lines = len(self.anomaly_lines)
        
        print(f"Line-level anomaly labeling results (using original logic):")
        print(f"  - Total lines from anomalous blocks: {total_lines:,}")
        print(f"  - Labeled as anomalous: {line_anomalies:,} ({100*line_anomalies/total_lines:.1f}%)")
        print(f"  - Labeled as normal: {line_normals:,} ({100*line_normals/total_lines:.1f}%)")
        
    def get_stress_test_data(self, num_samples: int = 1000, anomaly_ratio: float = 0.8) -> Tuple[List[str], List[int]]:
        """
        Get balanced dataset for stress testing
        Args:
            num_samples: Total number of samples to return
            anomaly_ratio: Proportion of actual anomalies (0.0 to 1.0)
        """
        print(f"Preparing {num_samples} samples for stress testing (anomaly ratio: {anomaly_ratio:.1f})")
        
        # Separate normal and anomaly lines
        anomaly_entries = self.anomaly_lines[self.anomaly_lines['LineLabel'] == 1]
        normal_entries = self.anomaly_lines[self.anomaly_lines['LineLabel'] == 0]
        
        # Calculate sample sizes
        n_anomalies = min(int(num_samples * anomaly_ratio), len(anomaly_entries))
        n_normals = min(num_samples - n_anomalies, len(normal_entries))
        
        # Sample data
        sampled_anomalies = anomaly_entries.sample(n=n_anomalies, random_state=42) if n_anomalies > 0 else pd.DataFrame()
        sampled_normals = normal_entries.sample(n=n_normals, random_state=42) if n_normals > 0 else pd.DataFrame()
        
        # Combine and shuffle
        combined_df = pd.concat([sampled_anomalies, sampled_normals])
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Extract messages and labels
        messages = []
        labels = []
        
        for _, row in combined_df.iterrows():
            messages.append(row['Content'])
            labels.append(int(row['LineLabel']))
        
        print(f"Stress test dataset prepared:")
        print(f"   - Total samples: {len(messages)}")
        print(f"   - Anomalies: {labels.count(1)} ({100*labels.count(1)/len(labels):.1f}%)")
        print(f"   - Normal: {labels.count(0)} ({100*labels.count(0)/len(labels):.1f}%)")
        
        return messages, labels
    
    
    def save_stress_test_data(self, output_file='hdfs_stress_test_data.jsonl', num_samples=1000, anomaly_ratio=0.8):
        """Save stress test data in JSONL format"""
        messages, labels = self.get_stress_test_data(num_samples, anomaly_ratio)
        
        # Create output directory path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'anomaly-injection-tests')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create full output path
        full_output_path = os.path.join(output_dir, output_file)
        
        data = []
        for i, (msg, label) in enumerate(zip(messages, labels)):
            data.append({
                'text': msg,
                'message': msg,  # For compatibility
                'label': int(label),
                'index': i,
                'id': i,
                'timestamp': pd.Timestamp.now().isoformat(),
                'source': 'stress_test'
            })
        
        with open(full_output_path, 'w') as f:
            for record in data:
                f.write(json.dumps(record) + '\n')
        
        print(f"Saved {len(data)} stress test records to {full_output_path}")
        return full_output_path
    

if __name__ == "__main__":
    # Demo usage
    loader = HDFSAnomalyOnlyLoader()
    
    # Generate different types of stress test data
    print("\n" + "="*60)
    print("STRESS TEST DATA GENERATION")
    print("="*60)
    
    loader.save_stress_test_data('hdfs_eval_test_5.jsonl', num_samples=10000, anomaly_ratio=0.05)
    loader.save_stress_test_data('hdfs_eval_test_10.jsonl', num_samples=10000 , anomaly_ratio=0.10)
    loader.save_stress_test_data('hdfs_eval_test_15.jsonl', num_samples=10000 , anomaly_ratio=0.15)
    loader.save_stress_test_data('hdfs_eval_test_20.jsonl', num_samples=10000, anomaly_ratio=0.20)
    loader.save_stress_test_data('hdfs_eval_test_40.jsonl', num_samples=10000, anomaly_ratio=0.40)

    print("\nStress test datasets ready for anomaly detection engine testing!")
    print("Use these files with kafka_producer.py to test the enhanced scoring service")
