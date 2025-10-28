#!/usr/bin/env python3
"""
Anomaly Detection Engine Stress Test
Comprehensive testing of the enhanced scoring service with anomaly-only data
"""
# Suppress urllib3 SSL warnings
import warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL 1.1.1+')

import time
import requests
import json
import sys
from kafka import KafkaProducer
import pandas as pd
from typing import Dict, List
import threading
import logging
from collections import defaultdict
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDetectionStressTester:
    def __init__(self, scoring_service_url="http://localhost:8003", kafka_servers="localhost:9092"):
        self.scoring_service_url = scoring_service_url
        self.kafka_servers = kafka_servers
        self.producer = None
        
    def initialize_kafka(self):
        """Initialize Kafka producer"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                batch_size=16384,
                linger_ms=10
            )
            logger.info(f" Kafka producer initialized: {self.kafka_servers}")
            return True
        except Exception as e:
            logger.error(f" Kafka initialization failed: {e}")
            return False
    
    def check_scoring_service(self):
        """Check if scoring service is available"""
        try:
            response = requests.get(f"{self.scoring_service_url}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                logger.info(f" Scoring service healthy: {health['status']}")
                return True
            else:
                logger.error(f" Scoring service unhealthy: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f" Scoring service check failed: {e}")
            return False
    
    def reset_metrics(self):
        """Reset all performance metrics and DB before testing"""
        try:
            response = requests.post(f"{self.scoring_service_url}/reset_metrics")
            if response.status_code == 200:
                logger.info(" Performance metrics reset")
                return True
        except Exception as e:
            logger.error(f" Failed to reset metrics: {e}")
        return False
    
    def load_test_data(self, data_file: str) -> List[Dict]:
        """Load test data from JSONL file"""
        data = []
        try:
            with open(data_file, 'r') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            logger.info(f"Loaded {len(data)} test samples from {data_file}")
            return data
        except Exception as e:
            logger.error(f" Failed to load test data: {e}")
            return []
    
    def analyze_dataset(self, data: List[Dict], filename: str) -> Dict:
        """Analyze dataset to count anomalies and normal samples"""
        normal_count = 0
        anomaly_count = 0
        
        for record in data:
            label = record.get('label', 0)
            if label == 1:
                anomaly_count += 1
            else:
                normal_count += 1
        
        return {
            'filename': filename,
            'total_samples': len(data),
            'normal_count': normal_count,
            'anomaly_count': anomaly_count,
            'anomaly_ratio': anomaly_count / len(data) if len(data) > 0 else 0
        }
    
    def stream_to_kafka(self, data: List[Dict], topic="logs", delay_ms=100):
        """Stream test data to Kafka"""
        if not self.producer:
            logger.error(" Kafka producer not initialized")
            return False
        
        logger.info(f" Streaming {len(data)} messages to Kafka topic '{topic}' (delay: {delay_ms}ms)")
        
        success_count = 0
        start_time = time.time()
        
        for i, record in enumerate(data):
            try:
                # Send to Kafka
                future = self.producer.send(topic, value=record)
                future.get(timeout=10)  # Wait for confirmation
                success_count += 1
                
                # Progress reporting
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    logger.info(f"   Progress: {i+1}/{len(data)} ({rate:.1f} msg/sec)")
                
                # Delay between messages
                time.sleep(delay_ms / 1000)
                
            except Exception as e:
                logger.error(f" Failed to send message {i}: {e}")
        
        # Flush remaining messages
        self.producer.flush()
        
        elapsed = time.time() - start_time
        logger.info(f" Streamed {success_count}/{len(data)} messages in {elapsed:.1f}s ({success_count/elapsed:.1f} msg/sec)")
        return success_count == len(data)
    
    def direct_api_test(self, data: List[Dict], max_samples=100):
        """Test scoring service directly via API calls with labels"""
        logger.info(f"Direct API testing with {min(len(data), max_samples)} samples...")
        
        results = []
        start_time = time.time()
        
        for i, record in enumerate(data[:max_samples]):
            try:
                # Use the labeled endpoint
                text = record['text']
                actual_label = record['label']
                
                response = requests.post(
                    f"{self.scoring_service_url}/score_with_label",
                    params={'text': text, 'actual_label': actual_label},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    results.append(result)
                    
                    if (i + 1) % 20 == 0:
                        current_acc = result.get('current_accuracy', 0)
                        logger.info(f"   Progress: {i+1}/{max_samples} (Current accuracy: {current_acc:.3f})")
                else:
                    logger.error(f" API call failed: {response.status_code}")
                    
            except Exception as e:
                logger.error(f" API call error for sample {i}: {e}")
        
        elapsed = time.time() - start_time
        
        if results:
            final_result = results[-1]
            logger.info(f".  Direct API test completed in {elapsed:.1f}s")
            logger.info(f"   Final Accuracy: {final_result.get('current_accuracy', 0):.3f}")
            logger.info(f"   Final Precision: {final_result.get('current_precision', 0):.3f}")
            logger.info(f"   Final Recall: {final_result.get('current_recall', 0):.3f}")
            logger.info(f"   Final F1-Score: {final_result.get('current_f1', 0):.3f}")
            logger.info(f"   Avg Processing Time: {final_result.get('processing_time_ms', 0):.1f}ms")
        
        return results
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        try:
            response = requests.get(f"{self.scoring_service_url}/performance_metrics")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f" Failed to get performance metrics: {e}")
        return {}
    
    def get_accuracy_report(self) -> Dict:
        """Get comprehensive accuracy report"""
        try:
            response = requests.get(f"{self.scoring_service_url}/accuracy_report")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f" Failed to get accuracy report: {e}")
        return {}
    
    def wait_for_kafka_processing(self, expected_messages: int, max_wait_seconds=300):
        """Wait for Kafka messages to be processed"""
        logger.info(f"Waiting for {expected_messages} Kafka messages to be processed...")
        
        start_time = time.time()
        last_count = 0
        
        while time.time() - start_time < max_wait_seconds:
            try:
                metrics = self.get_performance_metrics()
                kafka_processed = metrics.get('system_stats', {}).get('kafka_messages_processed', 0)
                
                if kafka_processed >= expected_messages:
                    logger.info(f" All {kafka_processed} Kafka messages processed!")
                    return True
                
                if kafka_processed != last_count:
                    logger.info(f"   Progress: {kafka_processed}/{expected_messages} messages processed")
                    last_count = kafka_processed
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f" Error checking Kafka processing: {e}")
                time.sleep(5)
        
        logger.warning(f"Timeout waiting for Kafka processing (processed: {last_count}/{expected_messages})")
        return False
    
    def run_comprehensive_stress_test(self, test_files: List[str]):
        """Run comprehensive stress test with multiple datasets"""
        logger.info("Starting Comprehensive Anomaly Detection Stress Test")
        logger.info("="*70)
        
        # Pre-checks
        if not self.check_scoring_service():
            logger.error(" Scoring service not available. Start it first!")
            return False
        
        if not self.initialize_kafka():
            logger.error(" Kafka not available. Check Kafka setup!")
            return False
        
        # Reset metrics
        self.reset_metrics()
        
        overall_results = {}
        dataset_analysis = {}
        
        for test_file in test_files:
            logger.info(f"\n Testing with dataset: {test_file}")
            logger.info("-" * 50)
            
            # Load test data
            data = self.load_test_data(test_file)
            if not data:
                continue
            
            test_name = test_file.replace('.jsonl', '').replace('hdfs_', '')
            
            # Analyze dataset composition
            dataset_info = self.analyze_dataset(data, test_file)
            dataset_analysis[test_name] = dataset_info
            
            logger.info(f"Dataset Analysis:")
            logger.info(f"   Total samples: {dataset_info['total_samples']}")
            logger.info(f"   Normal samples: {dataset_info['normal_count']}")
            logger.info(f"   Anomaly samples: {dataset_info['anomaly_count']}")
            logger.info(f"   Anomaly ratio: {dataset_info['anomaly_ratio']:.4f} ({dataset_info['anomaly_ratio']*100:.2f}%)")
            
            # Single Test: Kafka streaming with labeled data
            logger.info(" Kafka Streaming Test with Labeled Data...")
            stream_success = self.stream_to_kafka(data, delay_ms=50)
            
            if stream_success:
                # Wait for processing
                self.wait_for_kafka_processing(len(data))
                
                # Get final metrics (now based entirely on Kafka processing)
                final_metrics = self.get_performance_metrics()
                accuracy_report = self.get_accuracy_report()
                
                overall_results[test_name] = {
                    'dataset_size': len(data),
                    'kafka_streaming_success': stream_success,
                    'final_metrics': final_metrics,
                    'accuracy_report': accuracy_report
                }
                
                # Display results
                self.display_test_results(test_name, overall_results[test_name])
            
            logger.info(f" Completed testing with {test_file}")
        
        # Display overall summary
        self.display_overall_summary(overall_results)
        
        # Save results to CSV with dataset analysis
        if overall_results:
            csv_file = self.save_results_to_csv(overall_results, dataset_analysis)
            logger.info(f"Detailed results exported to: {csv_file}")
        
        return True
    
    def display_test_results(self, test_name: str, results: Dict):
        """Display results for a single test"""
        logger.info(f"\n RESULTS FOR {test_name.upper()}")
        logger.info("=" * 40)
        
        final_metrics = results.get('final_metrics', {})
        accuracy_metrics = final_metrics.get('accuracy_metrics', {})
        performance_stats = final_metrics.get('performance_stats', {})
        
        logger.info(f"Dataset Size: {results['dataset_size']}")
        logger.info(f"Kafka Streaming: {' Success' if results['kafka_streaming_success'] else ' Failed'}")
        logger.info(f"\nACCURACY METRICS (from Kafka processing):")
        logger.info(f"  Accuracy:  {accuracy_metrics.get('accuracy', 0):.3f}")
        logger.info(f"  Precision: {accuracy_metrics.get('precision', 0):.3f}")
        logger.info(f"  Recall:    {accuracy_metrics.get('recall', 0):.3f}")
        logger.info(f"  F1-Score:  {accuracy_metrics.get('f1_score', 0):.3f}")
        logger.info(f"\nPERFORMANCE METRICS:")
        logger.info(f"  Total Predictions: {performance_stats.get('total_predictions', 0)}")
        logger.info(f"  Avg Processing Time: {performance_stats.get('avg_processing_time_ms', 0):.1f}ms")
        logger.info(f"  Cache Hit Rate: {performance_stats.get('cache_hit_rate', 0):.3f}")
        logger.info(f"  Kafka Messages Processed: {final_metrics.get('system_stats', {}).get('kafka_messages_processed', 0)}")
    
    def display_overall_summary(self, overall_results: Dict):
        """Display overall summary of all tests"""
        logger.info("\n" + "="*70)
        logger.info(" OVERALL STRESS TEST SUMMARY")
        logger.info("="*70)
        
        for test_name, results in overall_results.items():
            final_metrics = results.get('final_metrics', {})
            accuracy_metrics = final_metrics.get('accuracy_metrics', {})
            
            logger.info(f"{test_name:25} | "
                       f"Acc: {accuracy_metrics.get('accuracy', 0):.3f} | "
                       f"P: {accuracy_metrics.get('precision', 0):.3f} | "
                       f"R: {accuracy_metrics.get('recall', 0):.3f} | "
                       f"F1: {accuracy_metrics.get('f1_score', 0):.3f}")
        
        logger.info("="*70)
        logger.info(" Stress test complete! Check the scoring service logs and database for detailed results.")
    
    def save_results_to_csv(self, overall_results: Dict, dataset_analysis: Dict):
        """Save test results to CSV file with datetime"""
        # Create evaluation_results directory if it doesn't exist
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, "evaluation_results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            logger.info(f"Created directory: {results_dir}")
        
        # Generate filename with current datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(results_dir, f"anomaly_evaluation_results_{timestamp}.csv")
        
        # Prepare data for CSV
        csv_data = []
        
        for test_name, results in overall_results.items():
            # Get dataset analysis for this test
            dataset_info = dataset_analysis.get(test_name, {})
            
            final_metrics = results.get('final_metrics', {})
            accuracy_metrics = final_metrics.get('accuracy_metrics', {})
            performance_stats = final_metrics.get('performance_stats', {})
            system_stats = final_metrics.get('system_stats', {})
            
            row_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'test_name': test_name,
                'jsonl_filename': dataset_info.get('filename', 'unknown'),
                'total_samples': dataset_info.get('total_samples', 0),
                'normal_count': dataset_info.get('normal_count', 0),
                'anomaly_count': dataset_info.get('anomaly_count', 0),
                'anomaly_ratio': dataset_info.get('anomaly_ratio', 0),
                'kafka_streaming_success': results.get('kafka_streaming_success', False),
                'accuracy': accuracy_metrics.get('accuracy', 0),
                'precision': accuracy_metrics.get('precision', 0),
                'recall': accuracy_metrics.get('recall', 0),
                'f1_score': accuracy_metrics.get('f1_score', 0),
                'total_predictions': performance_stats.get('total_predictions', 0),
                'avg_processing_time_ms': performance_stats.get('avg_processing_time_ms', 0),
                'cache_hit_rate': performance_stats.get('cache_hit_rate', 0),
                'kafka_messages_processed': system_stats.get('kafka_messages_processed', 0)
            }
            
            csv_data.append(row_data)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(csv_data)
        
        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(csv_filename)
        
        # Save to CSV (append mode)
        df.to_csv(csv_filename, mode='a', header=not file_exists, index=False)
        
        logger.info(f" Results saved to: {csv_filename}")
        logger.info(f"Saved {len(csv_data)} test results with dataset analysis")
        
        return csv_filename

def main():
    """Main stress test execution"""
    if len(sys.argv) < 2:
        print("Usage: python3 anomaly_stress_test.py <test_data_files...>")
        print("Example: python3 anomaly_stress_test.py hdfs_stress_test_balanced.jsonl hdfs_pure_anomalies.jsonl")
        sys.exit(1)
    
    test_files = sys.argv[1:]
    
    # Validate files exist
    import os
    for file in test_files:
        if not os.path.exists(file):
            logger.error(f" Test file not found: {file}")
            sys.exit(1)
    
    # Run stress test
    tester = AnomalyDetectionStressTester()
    success = tester.run_comprehensive_stress_test(test_files)
    
    if success:
        logger.info(" Stress testing completed successfully!")
        sys.exit(0)
    else:
        logger.error("Stress testing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
