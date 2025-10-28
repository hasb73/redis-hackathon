-- Grafana Dashboard Test Queries for Anomaly Detection System
-- This file contains all SQL queries that will be used in Grafana dashboards
-- Each query is tested for performance and accuracy

-- =============================================================================
-- SYSTEM OVERVIEW DASHBOARD QUERIES
-- =============================================================================

-- Query 1: Real-time System Metrics (Current Performance)
-- Used in: System Overview Dashboard - Main Stats Panel
SELECT 
    COUNT(*) as total_predictions,
    ROUND(AVG(CASE WHEN is_correct = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) as accuracy_percent,
    ROUND(AVG(anomaly_score), 3) as avg_anomaly_score,
    ROUND(AVG(confidence), 3) as avg_confidence,
    ROUND(AVG(processing_time_ms), 1) as avg_processing_time_ms,
    datetime('now') as last_updated
FROM anomaly_detections 
WHERE datetime(created_at) >= datetime('now', '-1 hour');

-- Query 2: Current Anomaly Rate (Gauge Panel)
-- Used in: System Overview Dashboard - Anomaly Rate Gauge
SELECT 
    COUNT(*) as total_predictions,
    SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) as anomaly_count,
    ROUND((SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) as anomaly_rate_percent,
    datetime('now', '-1 hour') as time_window_start,
    datetime('now') as time_window_end
FROM anomaly_detections 
WHERE datetime(created_at) >= datetime('now', '-1 hour');

-- Query 3: System Health Status
-- Used in: System Overview Dashboard - Health Indicator
SELECT 
    CASE 
        WHEN accuracy >= 0.9 AND avg_processing_time <= 100 THEN 'Healthy'
        WHEN accuracy >= 0.8 AND avg_processing_time <= 200 THEN 'Warning'
        ELSE 'Critical'
    END as health_status,
    accuracy,
    avg_processing_time,
    total_predictions,
    last_prediction_time
FROM (
    SELECT 
        ROUND(AVG(CASE WHEN is_correct = 1 THEN 1.0 ELSE 0.0 END), 3) as accuracy,
        ROUND(AVG(processing_time_ms), 1) as avg_processing_time,
        COUNT(*) as total_predictions,
        MAX(datetime(created_at)) as last_prediction_time
    FROM anomaly_detections 
    WHERE datetime(created_at) >= datetime('now', '-1 hour')
);

-- =============================================================================
-- MODEL PERFORMANCE DASHBOARD QUERIES
-- =============================================================================

-- Query 4: Confusion Matrix Data
-- Used in: Model Performance Dashboard - Confusion Matrix Heatmap
SELECT 
    predicted_label,
    COALESCE(actual_label, -1) as actual_label,  -- Handle NULL actual labels
    COUNT(*) as prediction_count,
    ROUND(AVG(confidence), 3) as avg_confidence,
    ROUND(AVG(anomaly_score), 3) as avg_anomaly_score
FROM anomaly_detections 
WHERE datetime(created_at) >= datetime('now', '-24 hours')
GROUP BY predicted_label, actual_label
ORDER BY predicted_label, actual_label;

-- Query 5: Model Voting Analysis
-- Used in: Model Performance Dashboard - Voting Patterns
SELECT 
    model_votes,
    predicted_label,
    COUNT(*) as vote_count,
    ROUND(AVG(confidence), 3) as avg_confidence,
    ROUND(AVG(anomaly_score), 3) as avg_anomaly_score,
    ROUND(AVG(CASE WHEN is_correct = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) as accuracy_percent
FROM anomaly_detections 
WHERE datetime(created_at) >= datetime('now', '-24 hours')
GROUP BY model_votes, predicted_label
ORDER BY vote_count DESC
LIMIT 20;

-- Query 6: Confidence Score Distribution
-- Used in: Model Performance Dashboard - Confidence Histogram
SELECT 
    ROUND(confidence, 1) as confidence_bucket,
    COUNT(*) as frequency,
    predicted_label,
    ROUND(AVG(CASE WHEN is_correct = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) as accuracy_percent
FROM anomaly_detections 
WHERE datetime(created_at) >= datetime('now', '-24 hours')
GROUP BY ROUND(confidence, 1), predicted_label
ORDER BY confidence_bucket, predicted_label;

-- Query 7: Performance Trends Over Time
-- Used in: Model Performance Dashboard - Time Series Charts
SELECT 
    datetime(created_at, 'start of hour') as hour_bucket,
    COUNT(*) as predictions_per_hour,
    ROUND(AVG(CASE WHEN is_correct = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) as hourly_accuracy,
    ROUND(AVG(processing_time_ms), 1) as avg_processing_time,
    ROUND(AVG(confidence), 3) as avg_confidence,
    SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) as anomalies_detected
FROM anomaly_detections 
WHERE datetime(created_at) >= datetime('now', '-7 days')
GROUP BY datetime(created_at, 'start of hour')
ORDER BY hour_bucket;

-- =============================================================================
-- ANOMALY ANALYSIS DASHBOARD QUERIES
-- =============================================================================

-- Query 8: Anomaly Timeline (Recent Events)
-- Used in: Anomaly Analysis Dashboard - Timeline Visualization
SELECT 
    datetime(timestamp) as event_timestamp,
    predicted_label,
    ROUND(anomaly_score, 3) as anomaly_score,
    ROUND(confidence, 3) as confidence,
    source,
    SUBSTR(text, 1, 100) || '...' as text_preview,
    model_votes,
    CASE WHEN is_correct = 1 THEN 'Correct' ELSE 'Incorrect' END as prediction_accuracy
FROM anomaly_detections 
WHERE predicted_label = 1  -- Only anomalies
    AND datetime(created_at) >= datetime('now', '-24 hours')
ORDER BY timestamp DESC 
LIMIT 100;

-- Query 9: Source-based Analysis
-- Used in: Anomaly Analysis Dashboard - Source Breakdown
SELECT 
    source,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) as anomaly_count,
    ROUND((SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) as anomaly_rate_percent,
    ROUND(AVG(CASE WHEN is_correct = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) as accuracy_percent,
    ROUND(AVG(processing_time_ms), 1) as avg_processing_time_ms,
    ROUND(AVG(anomaly_score), 3) as avg_anomaly_score,
    ROUND(AVG(confidence), 3) as avg_confidence
FROM anomaly_detections 
WHERE datetime(created_at) >= datetime('now', '-24 hours')
GROUP BY source
ORDER BY total_predictions DESC;

-- Query 10: Anomaly Score Distribution Comparison
-- Used in: Anomaly Analysis Dashboard - Score Distribution Charts
SELECT 
    ROUND(anomaly_score, 1) as score_bucket,
    predicted_label,
    COUNT(*) as frequency,
    ROUND(AVG(confidence), 3) as avg_confidence
FROM anomaly_detections 
WHERE datetime(created_at) >= datetime('now', '-24 hours')
GROUP BY ROUND(anomaly_score, 1), predicted_label
ORDER BY score_bucket, predicted_label;

-- =============================================================================
-- OPERATIONS MONITOR DASHBOARD QUERIES
-- =============================================================================

-- Query 11: Processing Performance Metrics
-- Used in: Operations Monitor Dashboard - Performance Panels
SELECT 
    datetime(created_at, 'start of hour') as time_bucket,
    COUNT(*) as predictions_count,
    ROUND(AVG(processing_time_ms), 1) as avg_processing_time,
    ROUND(MIN(processing_time_ms), 1) as min_processing_time,
    ROUND(MAX(processing_time_ms), 1) as max_processing_time
FROM anomaly_detections 
WHERE datetime(created_at) >= datetime('now', '-24 hours')
GROUP BY datetime(created_at, 'start of hour')
ORDER BY time_bucket;

-- Query 12: Error Rate Monitoring
-- Used in: Operations Monitor Dashboard - Error Rate Panels
SELECT 
    datetime(created_at, 'start of hour') as time_bucket,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) as incorrect_predictions,
    SUM(CASE WHEN actual_label IS NULL THEN 1 ELSE 0 END) as missing_labels,
    ROUND((SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) as error_rate_percent,
    ROUND((SUM(CASE WHEN actual_label IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) as missing_label_rate_percent
FROM anomaly_detections 
WHERE datetime(created_at) >= datetime('now', '-24 hours')
GROUP BY datetime(created_at, 'start of hour')
ORDER BY time_bucket;

-- Query 13: Data Volume Metrics
-- Used in: Operations Monitor Dashboard - Volume Tracking
SELECT 
    DATE(created_at) as date,
    COUNT(*) as daily_predictions,
    SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) as daily_anomalies,
    COUNT(DISTINCT source) as active_sources,
    ROUND(AVG(processing_time_ms), 1) as avg_daily_processing_time,
    MIN(datetime(created_at)) as first_prediction,
    MAX(datetime(created_at)) as last_prediction
FROM anomaly_detections 
WHERE datetime(created_at) >= datetime('now', '-30 days')
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- =============================================================================
-- PERFORMANCE METRICS TABLE QUERIES
-- =============================================================================

-- Query 14: Latest Performance Metrics from Dedicated Table
-- Used in: All Dashboards - Performance Metrics Panels
SELECT 
    datetime(timestamp) as metric_timestamp,
    total_predictions,
    ROUND(accuracy * 100, 2) as accuracy_percent,
    ROUND(precision * 100, 2) as precision_percent,
    ROUND(recall * 100, 2) as recall_percent,
    ROUND(f1_score * 100, 2) as f1_score_percent,
    true_positives,
    false_positives,
    true_negatives,
    false_negatives,
    ROUND(avg_processing_time_ms, 1) as avg_processing_time_ms,
    datetime(created_at) as recorded_at
FROM performance_metrics 
ORDER BY created_at DESC 
LIMIT 10;

-- Query 15: Performance Metrics Trends
-- Used in: Model Performance Dashboard - Metrics Trends
SELECT 
    DATE(created_at) as date,
    ROUND(AVG(accuracy) * 100, 2) as daily_accuracy,
    ROUND(AVG(precision) * 100, 2) as daily_precision,
    ROUND(AVG(recall) * 100, 2) as daily_recall,
    ROUND(AVG(f1_score) * 100, 2) as daily_f1_score,
    SUM(total_predictions) as daily_total_predictions,
    ROUND(AVG(avg_processing_time_ms), 1) as daily_avg_processing_time
FROM performance_metrics 
WHERE datetime(created_at) >= datetime('now', '-30 days')
GROUP BY DATE(created_at)
ORDER BY date;

-- =============================================================================
-- ALERT AND THRESHOLD QUERIES
-- =============================================================================

-- Query 16: Alert Conditions Check
-- Used in: Operations Monitor Dashboard - Alert Panels
SELECT 
    'Accuracy Alert' as alert_type,
    CASE 
        WHEN current_accuracy < 0.8 THEN 'CRITICAL'
        WHEN current_accuracy < 0.9 THEN 'WARNING'
        ELSE 'OK'
    END as alert_level,
    ROUND(current_accuracy * 100, 2) as current_value,
    90.0 as warning_threshold,
    80.0 as critical_threshold,
    datetime('now') as check_time
FROM (
    SELECT AVG(CASE WHEN is_correct = 1 THEN 1.0 ELSE 0.0 END) as current_accuracy
    FROM anomaly_detections 
    WHERE datetime(created_at) >= datetime('now', '-1 hour')
)

UNION ALL

SELECT 
    'Processing Time Alert' as alert_type,
    CASE 
        WHEN current_processing_time > 200 THEN 'CRITICAL'
        WHEN current_processing_time > 100 THEN 'WARNING'
        ELSE 'OK'
    END as alert_level,
    ROUND(current_processing_time, 1) as current_value,
    100.0 as warning_threshold,
    200.0 as critical_threshold,
    datetime('now') as check_time
FROM (
    SELECT AVG(processing_time_ms) as current_processing_time
    FROM anomaly_detections 
    WHERE datetime(created_at) >= datetime('now', '-1 hour')
)

UNION ALL

SELECT 
    'Anomaly Rate Alert' as alert_type,
    CASE 
        WHEN current_anomaly_rate > 50 THEN 'CRITICAL'
        WHEN current_anomaly_rate > 30 THEN 'WARNING'
        ELSE 'OK'
    END as alert_level,
    ROUND(current_anomaly_rate, 2) as current_value,
    30.0 as warning_threshold,
    50.0 as critical_threshold,
    datetime('now') as check_time
FROM (
    SELECT (SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as current_anomaly_rate
    FROM anomaly_detections 
    WHERE datetime(created_at) >= datetime('now', '-1 hour')
);

-- =============================================================================
-- DATA FRESHNESS AND HEALTH QUERIES
-- =============================================================================

-- Query 17: Data Freshness Check
-- Used in: Operations Monitor Dashboard - Data Health Panel
SELECT 
    MAX(datetime(created_at)) as last_data_timestamp,
    COUNT(*) as records_last_hour,
    ROUND((julianday('now') - julianday(MAX(created_at))) * 24 * 60, 1) as minutes_since_last_data,
    CASE 
        WHEN (julianday('now') - julianday(MAX(created_at))) * 24 * 60 > 60 THEN 'STALE'
        WHEN (julianday('now') - julianday(MAX(created_at))) * 24 * 60 > 30 THEN 'WARNING'
        ELSE 'FRESH'
    END as data_freshness_status
FROM anomaly_detections 
WHERE datetime(created_at) >= datetime('now', '-2 hours');

-- Query 18: Database Statistics
-- Used in: Operations Monitor Dashboard - System Info Panel
SELECT 
    'anomaly_detections' as table_name,
    COUNT(*) as total_records,
    MIN(datetime(created_at)) as oldest_record,
    MAX(datetime(created_at)) as newest_record,
    COUNT(DISTINCT source) as unique_sources,
    ROUND(AVG(LENGTH(text)), 0) as avg_text_length
FROM anomaly_detections

UNION ALL

SELECT 
    'performance_metrics' as table_name,
    COUNT(*) as total_records,
    MIN(datetime(created_at)) as oldest_record,
    MAX(datetime(created_at)) as newest_record,
    NULL as unique_sources,
    NULL as avg_text_length
FROM performance_metrics;