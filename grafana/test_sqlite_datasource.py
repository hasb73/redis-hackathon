#!/usr/bin/env python3
"""
SQLite Data Source Validation and Testing Script

This script validates the SQLite database connection and tests various queries
that will be used in Grafana dashboards for anomaly detection monitoring.
"""

import sqlite3
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional


class SQLiteDataSourceTester:
    """Test SQLite data source connectivity and query performance."""
    
    def __init__(self, db_path: str = "anomaly_detection.db"):
        self.db_path = db_path
        self.connection = None
        self.test_results = []
    
    def connect(self) -> bool:
        """Establish connection to SQLite database with error handling."""
        try:
            self.connection = sqlite3.connect(self.db_path, timeout=30.0)
            self.connection.row_factory = sqlite3.Row  # Enable column access by name
            
            # Test basic connectivity
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
            if result and result[0] == 1:
                print("✓ Database connection established successfully")
                return True
            else:
                print("✗ Database connection test failed")
                return False
                
        except sqlite3.Error as e:
            print(f"✗ Database connection error: {e}")
            return False
    
    def validate_schema(self) -> bool:
        """Validate that required tables and columns exist."""
        try:
            cursor = self.connection.cursor()
            
            # Check for required tables
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('anomaly_detections', 'performance_metrics')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['anomaly_detections', 'performance_metrics']
            missing_tables = set(required_tables) - set(tables)
            
            if missing_tables:
                print(f"✗ Missing required tables: {missing_tables}")
                return False
            
            # Validate anomaly_detections table structure
            cursor.execute("PRAGMA table_info(anomaly_detections)")
            columns = [row[1] for row in cursor.fetchall()]
            
            required_columns = [
                'id', 'timestamp', 'text', 'text_hash', 'predicted_label',
                'actual_label', 'anomaly_score', 'confidence', 'model_votes',
                'source', 'processing_time_ms', 'is_correct', 'created_at'
            ]
            
            missing_columns = set(required_columns) - set(columns)
            if missing_columns:
                print(f"✗ Missing columns in anomaly_detections: {missing_columns}")
                return False
            
            print("✓ Database schema validation passed")
            return True
            
        except sqlite3.Error as e:
            print(f"✗ Schema validation error: {e}")
            return False
    
    def test_query_performance(self, query: str, description: str, 
                             expected_max_time: float = 5.0) -> Dict[str, Any]:
        """Test query performance and validate results."""
        try:
            start_time = time.time()
            cursor = self.connection.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            end_time = time.time()
            
            execution_time = end_time - start_time
            row_count = len(results)
            
            test_result = {
                'description': description,
                'query': query,
                'execution_time': execution_time,
                'row_count': row_count,
                'success': True,
                'within_threshold': execution_time <= expected_max_time
            }
            
            status = "✓" if test_result['within_threshold'] else "⚠"
            print(f"{status} {description}: {execution_time:.3f}s ({row_count} rows)")
            
            if not test_result['within_threshold']:
                print(f"  Warning: Query exceeded expected time threshold of {expected_max_time}s")
            
            return test_result
            
        except sqlite3.Error as e:
            test_result = {
                'description': description,
                'query': query,
                'execution_time': None,
                'row_count': None,
                'success': False,
                'error': str(e)
            }
            print(f"✗ {description}: Query failed - {e}")
            return test_result
    
    def run_dashboard_queries(self) -> List[Dict[str, Any]]:
        """Test all queries that will be used in Grafana dashboards."""
        
        test_queries = [
            # System Overview Dashboard Queries
            {
                'query': """
                    SELECT 
                        COUNT(*) as total_predictions,
                        AVG(CASE WHEN is_correct = 1 THEN 1.0 ELSE 0.0 END) as accuracy,
                        AVG(anomaly_score) as avg_anomaly_score,
                        AVG(confidence) as avg_confidence,
                        AVG(processing_time_ms) as avg_processing_time
                    FROM anomaly_detections 
                    WHERE datetime(created_at) >= datetime('now', '-1 hour')
                """,
                'description': 'Real-time system metrics (last hour)',
                'max_time': 2.0
            },
            
            # Anomaly rate calculation
            {
                'query': """
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) as anomalies,
                        (SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as anomaly_rate
                    FROM anomaly_detections 
                    WHERE datetime(created_at) >= datetime('now', '-1 hour')
                """,
                'description': 'Current anomaly rate calculation',
                'max_time': 1.0
            },
            
            # Performance trends over time
            {
                'query': """
                    SELECT 
                        DATE(created_at) as date,
                        AVG(CASE WHEN is_correct = 1 THEN 1.0 ELSE 0.0 END) as daily_accuracy,
                        COUNT(*) as daily_predictions,
                        AVG(processing_time_ms) as avg_processing_time
                    FROM anomaly_detections 
                    WHERE created_at >= datetime('now', '-30 days')
                    GROUP BY DATE(created_at)
                    ORDER BY date
                """,
                'description': 'Performance trends (30 days)',
                'max_time': 3.0
            },
            
            # Confusion matrix data
            {
                'query': """
                    SELECT 
                        predicted_label,
                        actual_label,
                        COUNT(*) as count
                    FROM anomaly_detections 
                    WHERE actual_label IS NOT NULL
                    GROUP BY predicted_label, actual_label
                """,
                'description': 'Confusion matrix data',
                'max_time': 1.0
            },
            
            # Model voting analysis
            {
                'query': """
                    SELECT 
                        model_votes,
                        predicted_label,
                        COUNT(*) as vote_count,
                        AVG(confidence) as avg_confidence
                    FROM anomaly_detections 
                    GROUP BY model_votes, predicted_label
                    ORDER BY vote_count DESC
                    LIMIT 20
                """,
                'description': 'Model voting patterns analysis',
                'max_time': 2.0
            },
            
            # Source-based analysis
            {
                'query': """
                    SELECT 
                        source,
                        COUNT(*) as total_predictions,
                        SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) as anomaly_count,
                        AVG(CASE WHEN is_correct = 1 THEN 1.0 ELSE 0.0 END) as accuracy,
                        AVG(processing_time_ms) as avg_processing_time
                    FROM anomaly_detections 
                    GROUP BY source
                    ORDER BY total_predictions DESC
                """,
                'description': 'Source-based performance analysis',
                'max_time': 2.0
            },
            
            # Performance metrics table query
            {
                'query': """
                    SELECT 
                        timestamp,
                        accuracy,
                        precision,
                        recall,
                        f1_score,
                        total_predictions,
                        avg_processing_time_ms
                    FROM performance_metrics 
                    ORDER BY created_at DESC 
                    LIMIT 10
                """,
                'description': 'Latest performance metrics',
                'max_time': 1.0
            },
            
            # Time-series anomaly detection events
            {
                'query': """
                    SELECT 
                        datetime(timestamp) as event_time,
                        predicted_label,
                        anomaly_score,
                        confidence,
                        source
                    FROM anomaly_detections 
                    WHERE predicted_label = 1
                    ORDER BY timestamp DESC 
                    LIMIT 100
                """,
                'description': 'Recent anomaly events timeline',
                'max_time': 2.0
            }
        ]
        
        results = []
        print("\n=== Testing Dashboard Queries ===")
        
        for test in test_queries:
            result = self.test_query_performance(
                test['query'], 
                test['description'], 
                test.get('max_time', 5.0)
            )
            results.append(result)
            self.test_results.append(result)
        
        return results
    
    def test_data_quality(self) -> Dict[str, Any]:
        """Test data quality and completeness."""
        print("\n=== Testing Data Quality ===")
        
        quality_tests = []
        
        try:
            cursor = self.connection.cursor()
            
            # Test for null values in critical fields
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_rows,
                    SUM(CASE WHEN predicted_label IS NULL THEN 1 ELSE 0 END) as null_predictions,
                    SUM(CASE WHEN anomaly_score IS NULL THEN 1 ELSE 0 END) as null_scores,
                    SUM(CASE WHEN confidence IS NULL THEN 1 ELSE 0 END) as null_confidence
                FROM anomaly_detections
            """)
            
            result = cursor.fetchone()
            total_rows = result[0]
            null_predictions = result[1]
            null_scores = result[2] 
            null_confidence = result[3]
            
            print(f"✓ Total records: {total_rows}")
            
            if null_predictions > 0:
                print(f"⚠ Found {null_predictions} records with null predicted_label")
            else:
                print("✓ No null predicted_label values found")
            
            if null_scores > 0:
                print(f"⚠ Found {null_scores} records with null anomaly_score")
            else:
                print("✓ No null anomaly_score values found")
            
            if null_confidence > 0:
                print(f"⚠ Found {null_confidence} records with null confidence")
            else:
                print("✓ No null confidence values found")
            
            # Test data ranges
            cursor.execute("""
                SELECT 
                    MIN(anomaly_score) as min_score,
                    MAX(anomaly_score) as max_score,
                    MIN(confidence) as min_confidence,
                    MAX(confidence) as max_confidence,
                    MIN(processing_time_ms) as min_processing_time,
                    MAX(processing_time_ms) as max_processing_time
                FROM anomaly_detections
            """)
            
            ranges = cursor.fetchone()
            print(f"✓ Anomaly score range: {ranges[0]:.3f} to {ranges[1]:.3f}")
            print(f"✓ Confidence range: {ranges[2]:.3f} to {ranges[3]:.3f}")
            print(f"✓ Processing time range: {ranges[4]:.1f}ms to {ranges[5]:.1f}ms")
            
            return {
                'total_rows': total_rows,
                'data_quality_passed': null_predictions == 0 and null_scores == 0 and null_confidence == 0,
                'anomaly_score_range': [ranges[0], ranges[1]],
                'confidence_range': [ranges[2], ranges[3]],
                'processing_time_range': [ranges[4], ranges[5]]
            }
            
        except sqlite3.Error as e:
            print(f"✗ Data quality test failed: {e}")
            return {'data_quality_passed': False, 'error': str(e)}
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        
        successful_queries = sum(1 for result in self.test_results if result['success'])
        total_queries = len(self.test_results)
        
        fast_queries = sum(1 for result in self.test_results 
                          if result['success'] and result['within_threshold'])
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'database_path': self.db_path,
            'connection_successful': self.connection is not None,
            'total_queries_tested': total_queries,
            'successful_queries': successful_queries,
            'queries_within_threshold': fast_queries,
            'success_rate': successful_queries / total_queries if total_queries > 0 else 0,
            'performance_rate': fast_queries / total_queries if total_queries > 0 else 0,
            'test_results': self.test_results
        }
        
        return report
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            print("✓ Database connection closed")


def main():
    """Main test execution function."""
    print("=== SQLite Data Source Validation Test ===")
    print(f"Testing database: anomaly_detection.db")
    print(f"Test started at: {datetime.now()}")
    
    tester = SQLiteDataSourceTester()
    
    try:
        # Test connection
        if not tester.connect():
            print("✗ Cannot proceed - database connection failed")
            return False
        
        # Validate schema
        if not tester.validate_schema():
            print("✗ Cannot proceed - schema validation failed")
            return False
        
        # Test data quality
        data_quality = tester.test_data_quality()
        
        # Test dashboard queries
        query_results = tester.run_dashboard_queries()
        
        # Generate report
        report = tester.generate_test_report()
        
        # Save report to file
        with open('sqlite_datasource_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n=== Test Summary ===")
        print(f"Connection: {'✓ Success' if report['connection_successful'] else '✗ Failed'}")
        print(f"Query Success Rate: {report['success_rate']:.1%} ({report['successful_queries']}/{report['total_queries_tested']})")
        print(f"Performance Rate: {report['performance_rate']:.1%} ({report['queries_within_threshold']}/{report['total_queries_tested']})")
        print(f"Data Quality: {'✓ Passed' if data_quality.get('data_quality_passed', False) else '⚠ Issues Found'}")
        print(f"Report saved to: sqlite_datasource_test_report.json")
        
        overall_success = (
            report['connection_successful'] and 
            report['success_rate'] >= 0.9 and 
            data_quality.get('data_quality_passed', False)
        )
        
        if overall_success:
            print("\n✓ All tests passed - SQLite data source is ready for Grafana integration")
        else:
            print("\n⚠ Some tests failed - review issues before proceeding")
        
        return overall_success
        
    finally:
        tester.close()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)