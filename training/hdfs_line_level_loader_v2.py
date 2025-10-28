#!/usr/bin/env python3
"""
HDFS Line-Level Dataset Loader
Loads parsed HDFS logs and creates proper line-level training data
addressing the block-level vs line-level labeling issue
"""
import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional
import re

class HDFSLineLevelLoader:
    def __init__(self, base_path="HDFS_dataset"):
        self.base_path = base_path
        self.parsed_path = os.path.join(base_path, "parsed")
        self.preprocessed_path = os.path.join(base_path, "labels")
        
        print("Loading parsed HDFS data...")
        
        # Load structured logs (line-level)
        self.structured_df = pd.read_csv(os.path.join(self.parsed_path, "HDFS.log_structured.csv"))
        
        # Load templates
        self.templates_df = pd.read_csv(os.path.join(self.parsed_path, "HDFS.log_templates.csv"))
        
        # Load block-level labels
        self.labels_df = pd.read_csv(os.path.join(self.preprocessed_path, "anomaly_label.csv"))
        
        # Create block label mapping
        self.block_labels = dict(zip(self.labels_df['BlockId'], self.labels_df['Label']))
        
        print(f"Loaded HDFS line-level dataset:")
        print(f"  - {len(self.structured_df):,} structured log lines")
        print(f"  - {len(self.templates_df)} event templates")
        print(f"  - {len(self.labels_df):,} labeled blocks")
        
        # Cache parsed data
        self._prepare_line_level_data()
        
    def _extract_block_id(self, param_list_str: str) -> Optional[str]:
        """Extract block ID from parameter list"""
        if pd.isna(param_list_str):
            return None
        try:
            params = eval(param_list_str)
            for param in params:
                if isinstance(param, str) and param.startswith('blk_'):
                    # Handle cases like "blk_123 terminating"
                    block_id = param.split()[0]
                    if block_id.startswith('blk_'):
                        return block_id
        except:
            pass
        return None
    
    def _prepare_line_level_data(self):
        """Prepare line-level data with proper labeling"""
        print("Preparing line-level training data...")
        
        # Extract block IDs from structured logs
        self.structured_df['BlockId'] = self.structured_df['ParameterList'].apply(self._extract_block_id)
        
        # Map block labels to individual lines
        self.structured_df['BlockLabel'] = self.structured_df['BlockId'].map(self.block_labels)
        
        # Filter only lines with valid block IDs and labels
        self.valid_lines = self.structured_df[
            (self.structured_df['BlockId'].notna()) & 
            (self.structured_df['BlockLabel'].notna())
        ].copy()
        
        print(f"  - {len(self.valid_lines):,} lines with block IDs and labels")
        print(f"  - Block label distribution in lines:")
        print(self.valid_lines['BlockLabel'].value_counts())
        
        # Create line-level anomaly labels using sophisticated heuristics
        self._create_line_level_labels()
        
    def _create_line_level_labels(self):
        """Create sophisticated line-level anomaly labels"""
        print("Creating line-level anomaly labels...")
        
        # Initialize all lines as normal
        self.valid_lines['LineLabel'] = 0  # 0 = Normal
        
        # Load event templates for analysis
        template_map = dict(zip(self.templates_df['EventId'], self.templates_df['EventTemplate']))
        
        # Define anomaly-indicative event patterns (enhanced for HDFS)
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
        anomaly_blocks_mask = self.valid_lines['BlockLabel'] == 'Anomaly'
        anomaly_event_mask = self.valid_lines['EventId'].map(anomaly_event_keywords).fillna(False)
        
        strategy1_mask = anomaly_blocks_mask & anomaly_event_mask
        self.valid_lines.loc[strategy1_mask, 'LineLabel'] = 1
        
        # Strategy 2: Rare events in anomalous blocks (statistical approach)
        event_counts = self.templates_df.set_index('EventId')['Occurrences'].to_dict()
        rare_threshold = np.percentile(list(event_counts.values()), 10)  # Bottom 10% of events
        
        rare_event_mask = self.valid_lines['EventId'].map(
            lambda x: event_counts.get(x, 0) < rare_threshold
        )
        
        strategy2_mask = anomaly_blocks_mask & rare_event_mask
        self.valid_lines.loc[strategy2_mask, 'LineLabel'] = 1
        
        # Strategy 3: High-frequency events that suddenly appear in anomalous blocks
        # (This would require temporal analysis, simplified here)
        
        # Report results
        line_anomalies = self.valid_lines['LineLabel'].sum()
        total_lines = len(self.valid_lines)
        
        print(f"Line-level labeling results:")
        print(f"  - Total valid lines: {total_lines:,}")
        print(f"  - Normal lines: {total_lines - line_anomalies:,} ({100*(total_lines-line_anomalies)/total_lines:.2f}%)")
        print(f"  - Anomalous lines: {line_anomalies:,} ({100*line_anomalies/total_lines:.2f}%)")
        
    def get_line_level_data(self, sample_size: Optional[int] = None, include_template_ids: bool = False) -> Tuple[List[str], List[int], Optional[List[str]]]:
        """
        Get individual log lines with line-level anomaly labels
        Returns: (log_messages, labels, template_ids) where labels are 0=Normal, 1=Anomaly
        If include_template_ids=False, template_ids will be None for backward compatibility
        """
        # Sample data if requested
        if sample_size and len(self.valid_lines) > sample_size:
            # Stratified sampling to preserve label distribution
            normal_lines = self.valid_lines[self.valid_lines['LineLabel'] == 0]
            anomaly_lines = self.valid_lines[self.valid_lines['LineLabel'] == 1]
            
            anomaly_ratio = len(anomaly_lines) / len(self.valid_lines)
            n_anomaly = min(int(sample_size * anomaly_ratio), len(anomaly_lines))
            n_normal = sample_size - n_anomaly
            
            sampled_normal = normal_lines.sample(n=min(n_normal, len(normal_lines)), random_state=42)
            sampled_anomaly = anomaly_lines.sample(n=n_anomaly, random_state=42)
            
            sample_df = pd.concat([sampled_normal, sampled_anomaly]).sample(frac=1, random_state=42)
        else:
            sample_df = self.valid_lines
        
        # Create realistic log messages
        log_messages = []
        labels = []
        template_ids = [] if include_template_ids else None
        
        for _, row in sample_df.iterrows():
            # Use the actual log content
            log_message = row['Content']
            label = int(row['LineLabel'])
            
            log_messages.append(log_message)
            labels.append(label)
            
            if include_template_ids:
                template_ids.append(row['EventId'])
        
        return log_messages, labels, template_ids
    
    def get_event_statistics(self) -> Dict:
        """Get statistics about event types and their anomaly rates"""
        stats = {}
        
        for event_id in self.valid_lines['EventId'].unique():
            event_lines = self.valid_lines[self.valid_lines['EventId'] == event_id]
            total_count = len(event_lines)
            anomaly_count = (event_lines['LineLabel'] == 1).sum()
            anomaly_rate = anomaly_count / total_count if total_count > 0 else 0
            
            # Get template
            template = self.templates_df[self.templates_df['EventId'] == event_id]['EventTemplate'].iloc[0]
            
            stats[event_id] = {
                'template': template,
                'total_lines': total_count,
                'anomaly_lines': anomaly_count,
                'anomaly_rate': anomaly_rate
            }
        
        return stats
    
    def save_line_level_data(self, output_file='hdfs_line_level_data.jsonl', max_records=10600568):
        """Save line-level data for streaming/training"""
        messages, labels, template_ids = self.get_line_level_data(sample_size=max_records, include_template_ids=True)
        
        data = []
        for msg, label, template_id in zip(messages, labels, template_ids):
            data.append({
                'text': msg,
                'label': int(label),
                'template_id': template_id,
                'timestamp': pd.Timestamp.now().isoformat()
            })
        
        with open(output_file, 'w') as f:
            for record in data:
                f.write(json.dumps(record) + '\n')
        
        print(f"Saved {len(data)} line-level records to {output_file}")
        print(f"  - Each record includes: text, label, template_id, timestamp")
        return output_file

if __name__ == "__main__":
    # Demo usage
    loader = HDFSLineLevelLoader()
    
    # Get line-level training data
    messages, labels, _ = loader.get_line_level_data(sample_size=10600568)
    print(f"\nLine-level training data:")
    print(f"  - Total samples: {len(messages)}")
    print(f"  - Normal: {labels.count(0)}")
    print(f"  - Anomalies: {labels.count(1)}")
    
    print(f"\nSample anomalous log lines:")
    for i, (msg, label) in enumerate(zip(messages, labels)):
        if label == 1:
            print(f"  🚨 {msg}")
            if i > 5:  # Show only first few
                break
    
    # Get event statistics
    stats = loader.get_event_statistics()
    print(f"\nTop anomalous event types:")
    sorted_events = sorted(stats.items(), key=lambda x: x[1]['anomaly_rate'], reverse=True)
    for event_id, stat in sorted_events:
        if stat['anomaly_rate'] > 0:
            print(f"  {event_id}: {stat['anomaly_rate']:.3f} ({stat['anomaly_lines']}/{stat['total_lines']}) - {stat['template'][:60]}...")
    
    # Save data
    loader.save_line_level_data()
