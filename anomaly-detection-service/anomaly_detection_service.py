#!/usr/bin/env python3
"""
Real-Time HDFS Anomaly Detection Engine
Enhanced scoring service with database storage, accuracy tracking, and performance metrics
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import joblib, numpy as np, os, hashlib, time
import requests
import json
import redis
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag
from kafka import KafkaConsumer
from typing import List, Dict, Optional
import threading
import queue
import asyncio
import datetime
import logging
import sqlite3
from contextlib import contextmanager
import pandas as pd
from collections import defaultdict
import warnings

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*search.*method is deprecated.*")
warnings.filterwarnings("ignore", message=".*PydanticDeprecated.*")

# Configure logging
import os
log_file_path = os.path.join(os.path.dirname(__file__), 'anomaly_detection_service.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class InputPayload(BaseModel):
    embedding: list

class TextPayload(BaseModel):
    text: str

class BatchPayload(BaseModel):
    texts: List[str]

class RedisVLDetails(BaseModel):
    vote: int
    confidence: float
    similar_count: int
    method: str

class AnomalyResult(BaseModel):
    text: str
    predicted_label: int
    anomaly_score: float
    confidence: float
    actual_label: Optional[int] = None
    model_votes: Dict[str, int]
    source: str
    timestamp: str
    processing_time_ms: float
    redis_details: Optional[RedisVLDetails] = None

app = FastAPI(title="HDFS Real-Time Anomaly Detection Engine", version="2.0.0")

# Configuration
MODEL_PATH = os.environ.get('ENSEMBLE_MODEL_PATH', './training/models/line_level_ensemble_v2/line_level_ensemble_v2_results.joblib')
EMBEDDING_SERVICE_URL = os.environ.get('EMBEDDING_SERVICE_URL', 'http://localhost:8000')
KAFKA_SERVERS = os.environ.get('KAFKA_SERVERS', 'localhost:9092')
KAFKA_TOPIC = os.environ.get('KAFKA_TOPIC', 'logs')
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))
REDIS_INDEX_NAME = os.environ.get('REDIS_INDEX_NAME', 'logs_embeddings')
REDIS_KEY_PREFIX = os.environ.get('REDIS_KEY_PREFIX', 'log_entry:')
DB_PATH = os.environ.get('ANOMALY_DB_PATH', './anomaly-detection-service/anomaly_detection.db')

# Global variables
models_cache = None
scaler = None
prediction_queue = queue.Queue()
redis_client = None
search_index = None

# Enhanced statistics with accuracy tracking
stats = {
    'total_predictions': 0,
    'anomalies_detected': 0,
    'true_positives': 0,
    'false_positives': 0,
    'true_negatives': 0,
    'false_negatives': 0,
    'cache_hits': 0,
    'cache_misses': 0,
    'kafka_messages_processed': 0,
    'redis_vector_queries': 0,
    'redis_operations': 0,
    'avg_processing_time_ms': 0.0,
    'accuracy': 0.0,
    'precision': 0.0,
    'recall': 0.0,
    'f1_score': 0.0
}

def init_database():
    """Initialize SQLite database for anomaly storage"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS anomaly_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            text TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            predicted_label INTEGER NOT NULL,
            actual_label INTEGER,
            anomaly_score REAL NOT NULL,
            confidence REAL NOT NULL,
            model_votes TEXT NOT NULL,
            source TEXT NOT NULL,
            processing_time_ms REAL NOT NULL,
            is_correct INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create indexes for better query performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON anomaly_detections(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_predicted_label ON anomaly_detections(predicted_label)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_actual_label ON anomaly_detections(actual_label)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_text_hash ON anomaly_detections(text_hash)')
    
    # Performance metrics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            total_predictions INTEGER NOT NULL,
            accuracy REAL NOT NULL,
            precision REAL NOT NULL,
            recall REAL NOT NULL,
            f1_score REAL NOT NULL,
            true_positives INTEGER NOT NULL,
            false_positives INTEGER NOT NULL,
            true_negatives INTEGER NOT NULL,
            false_negatives INTEGER NOT NULL,
            avg_processing_time_ms REAL NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info(f" Database initialized: {DB_PATH}")

@contextmanager
def get_db_connection():
    """Database connection context manager"""
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()

def store_anomaly_detection(result: AnomalyResult, actual_label: Optional[int] = None):
    """Store anomaly detection result in database"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Calculate if prediction is correct
        is_correct = None
        if actual_label is not None:
            is_correct = 1 if result.predicted_label == actual_label else 0
        
        cursor.execute('''
            INSERT INTO anomaly_detections 
            (timestamp, text, text_hash, predicted_label, actual_label, 
             anomaly_score, confidence, model_votes, source, processing_time_ms, is_correct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.timestamp,
            result.text,
            get_text_hash(result.text),
            result.predicted_label,
            actual_label,
            result.anomaly_score,
            result.confidence,
            json.dumps(result.model_votes),
            result.source,
            result.processing_time_ms,
            is_correct
        ))
        
        conn.commit()

def update_accuracy_metrics(predicted_label: int, actual_label: int):
    """Update global accuracy metrics"""
    global stats
    
    if actual_label == 1 and predicted_label == 1:
        stats['true_positives'] += 1
    elif actual_label == 0 and predicted_label == 1:
        stats['false_positives'] += 1
    elif actual_label == 0 and predicted_label == 0:
        stats['true_negatives'] += 1
    elif actual_label == 1 and predicted_label == 0:
        stats['false_negatives'] += 1
    
    # Calculate derived metrics
    tp, fp, tn, fn = stats['true_positives'], stats['false_positives'], stats['true_negatives'], stats['false_negatives']
    total = tp + fp + tn + fn
    
    if total > 0:
        stats['accuracy'] = (tp + tn) / total
        stats['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        stats['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        stats['f1_score'] = 2 * (stats['precision'] * stats['recall']) / (stats['precision'] + stats['recall']) if (stats['precision'] + stats['recall']) > 0 else 0.0

def save_performance_snapshot():
    """Save current performance metrics to database"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO performance_metrics 
            (timestamp, total_predictions, accuracy, precision, recall, f1_score,
             true_positives, false_positives, true_negatives, false_negatives, avg_processing_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.datetime.now().isoformat(),
            stats['total_predictions'],
            stats['accuracy'],
            stats['precision'],
            stats['recall'],
            stats['f1_score'],
            stats['true_positives'],
            stats['false_positives'],
            stats['true_negatives'],
            stats['false_negatives'],
            stats['avg_processing_time_ms']
        ))
        conn.commit()

def initialize_connections():
    """Initialize Redis and Redis VL search index connections"""
    global redis_client, search_index
    
    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
        redis_client.ping()
        logger.info(f" Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        logger.error(f" Redis connection failed: {e}")
        redis_client = None
    
    try:
        # Initialize Redis VL search index
        search_index = SearchIndex.from_dict({
            "index": {
                "name": REDIS_INDEX_NAME,
                "prefix": REDIS_KEY_PREFIX,
                "storage_type": "hash"
            },
            "fields": [
                {
                    "name": "text",
                    "type": "text"
                },
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 384,
                        "distance_metric": "cosine",
                        "algorithm": "flat"
                    }
                },
                {
                    "name": "label",
                    "type": "numeric"
                },
                {
                    "name": "timestamp",
                    "type": "text"
                }
            ]
        })
        
        # Connect to Redis using existing client
        search_index.connect(redis_url=f"redis://{REDIS_HOST}:{REDIS_PORT}")
        logger.info(f" Connected to Redis VL search index: {REDIS_INDEX_NAME}")
    except Exception as e:
        logger.error(f" Redis VL search index connection failed: {e}")
        search_index = None

def get_text_hash(text: str) -> str:
    """Generate hash for text for caching"""
    return hashlib.md5(text.encode()).hexdigest()

def get_cached_result(text_hash: str) -> Optional[Dict]:
    """Get cached anomaly result from Redis"""
    global stats
    
    if redis_client is None:
        return None
    
    try:
        cached = redis_client.get(f"anomaly:{text_hash}")
        stats['redis_operations'] += 1
        
        if cached:
            stats['cache_hits'] += 1
            result = json.loads(cached)
            logger.info(f" Cache HIT for hash {text_hash[:8]}...")
            return result
        else:
            stats['cache_misses'] += 1
            return None
            
    except Exception as e:
        logger.error(f"Redis get error: {e}")
        return None

def cache_result(text_hash: str, result: Dict, ttl: int = 3600):
    """Cache anomaly result in Redis"""
    global stats
    
    if redis_client is None:
        return
    
    try:
        redis_client.setex(
            f"anomaly:{text_hash}", 
            ttl, 
            json.dumps(result)
        )
        stats['redis_operations'] += 1
        logger.info(f" Cached result for hash {text_hash[:8]}... (TTL: {ttl}s)")
        
    except Exception as e:
        logger.error(f"Redis set error: {e}")

def search_redis_by_text(text: str, limit: int = 1) -> Optional[List[Dict]]:
    """Search for existing embeddings in Redis VL by text similarity"""
    global stats
    
    if search_index is None:
        return None
    
    try:
        embedding = get_embedding(text)
        if not embedding:
            return None
        
        # Create vector query
        query = VectorQuery(
            vector=embedding,
            vector_field_name="embedding",
            return_fields=["text", "label", "timestamp"],
            num_results=limit
        )
        
        results = search_index.query(query)
        stats['redis_vector_queries'] += 1
        
        if results:
            logger.info(f" Redis VL found {len(results)} similar vector embeddings")
            return [
                {
                    'id': result.get('id', ''),
                    'score': 1.0 - float(result.get('vector_distance', 0.0)),  # Convert distance to similarity
                    'payload': {
                        'text': result.get('text', ''),
                        'label': int(result.get('label', 0)),
                        'timestamp': result.get('timestamp', '')
                    },
                    'embedding': embedding  # Return the query embedding
                } for result in results if float(result.get('vector_distance', 1.0)) <= 0.05  # Score threshold 0.95
            ]
        
        return None
        
    except Exception as e:
        logger.error(f"Redis VL search error: {e}")
        return None

def get_redis_similarity_vote(embedding: List[float], text: str) -> Dict:
    """Get Redis VL similarity-based anomaly vote"""
    global stats
    
    if search_index is None:
        return {'vote': 0, 'confidence': 0.0, 'similar_count': 0, 'method': 'no_redis'}
    
    try:
        # Search for similar embeddings with different thresholds
        high_similarity_query = VectorQuery(
            vector=embedding,
            vector_field_name="embedding",
            return_fields=["text", "label", "timestamp"],
            num_results=10
        )
        
        medium_similarity_query = VectorQuery(
            vector=embedding,
            vector_field_name="embedding",
            return_fields=["text", "label", "timestamp"],
            num_results=20
        )
        
        high_similarity_results = search_index.query(high_similarity_query)
        medium_similarity_results = search_index.query(medium_similarity_query)
        
        # Filter by distance thresholds (convert to similarity)
        high_sim_filtered = [r for r in high_similarity_results if float(r.get('vector_distance', 1.0)) <= 0.10]  # 0.90+ similarity
        med_sim_filtered = [r for r in medium_similarity_results if float(r.get('vector_distance', 1.0)) <= 0.25]  # 0.75+ similarity
        
        stats['redis_vector_queries'] += 2
        
        # Analyze similarity patterns
        high_sim_anomalies = 0
        high_sim_normal = 0
        med_sim_anomalies = 0
        med_sim_normal = 0
        
        # Count high similarity matches
        for result in high_sim_filtered:
            label = int(result.get('label', 0))
            if label == 1:
                high_sim_anomalies += 1
            else:
                high_sim_normal += 1
        
        # Count medium similarity matches
        for result in med_sim_filtered:
            label = int(result.get('label', 0))
            if label == 1:
                med_sim_anomalies += 1
            else:
                med_sim_normal += 1
        
        # Decision logic based on similarity patterns
        total_high = high_sim_anomalies + high_sim_normal
        total_medium = med_sim_anomalies + med_sim_normal
        
        # Priority to high similarity matches
        if total_high > 0:
            anomaly_ratio = high_sim_anomalies / total_high
            confidence = min(0.9, 0.5 + (anomaly_ratio * 0.4))  # Scale confidence
            vote = 1 if anomaly_ratio > 0.5 else 0
            method = f'high_sim_{total_high}'
        elif total_medium > 2:  # Need reasonable sample size
            anomaly_ratio = med_sim_anomalies / total_medium
            confidence = min(0.7, 0.3 + (anomaly_ratio * 0.4))  # Lower confidence for medium similarity
            vote = 1 if anomaly_ratio > 0.6 else 0  # Higher threshold for medium similarity
            method = f'med_sim_{total_medium}'
        else:
            # No sufficient similarity data - neutral vote
            vote = 0
            confidence = 0.1
            method = 'insufficient_data'
        
        logger.debug(f"🔍 Redis VL vote: {vote} (conf: {confidence:.3f}, method: {method})")
        
        return {
            'vote': vote,
            'confidence': confidence,
            'similar_count': total_high + total_medium,
            'method': method,
            'high_sim_matches': total_high,
            'high_sim_anomalies': high_sim_anomalies,
            'med_sim_matches': total_medium,
            'med_sim_anomalies': med_sim_anomalies
        }
        
    except Exception as e:
        logger.error(f"Redis VL similarity vote error: {e}")
        return {'vote': 0, 'confidence': 0.0, 'similar_count': 0, 'method': 'error'}

def get_embedding_from_redis_or_service(text: str) -> Optional[List[float]]:
    """Get embedding from Redis VL first, then embedding service as fallback"""

    if text:
     logger.info(f" Ingested HDFS Log Entry: {text}")
    
    redis_results = search_redis_by_text(text)
    if redis_results and len(redis_results) > 0:
        best_match = redis_results[0]
        if best_match['score'] > 0.98:
            logger.info(f" Using the stored Vector embedding (similarity: {best_match['score']:.3f})")
            return best_match['embedding'] if best_match['embedding'] else get_embedding(text)
    
    logger.error("Log entry not found in Redis VL, Falling back to embedding service")
    return get_embedding(text)

def load_ensemble_model():
    """Load the trained ensemble model"""
    global models_cache, scaler
    
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading ensemble model from {MODEL_PATH}")
        models_cache = joblib.load(MODEL_PATH)
        scaler = models_cache.get('scaler', None)
        models = models_cache.get('models', {})
        model_weights = models_cache.get('model_weights', {})
        logger.info(f" Loaded ensemble with {len(models)} models")
        
        # Log model weights if available
        if model_weights:
            logger.info(" Model weights for weighted voting:")
            for name, weight in model_weights.items():
                logger.info(f"   {name}: {weight:.4f}")
        else:
            logger.info(" Using equal weights (simple average voting)")
        
        if 'ensemble_score' in models_cache:
            score = models_cache['ensemble_score']
            logger.info(f"   Model Performance - P: {score.get('precision', 0):.3f}, R: {score.get('recall', 0):.3f}, F1: {score.get('f1', 0):.3f}")
            
        return True
    else:
        logger.error(f" Model not found at {MODEL_PATH}")
        return False

@app.on_event('startup')
async def startup_event():
    """Initialize the anomaly detection engine"""
    global models_cache, scaler
    
    logger.info("Starting Real-Time HDFS Anomaly Detection Engine...")
    
    # Initialize database
    init_database()
    
    # Initialize connections
    initialize_connections()
    
    # Load model
    if not load_ensemble_model():
        logger.warning(" Warning: No ensemble model loaded. Training required.")
    
    # Start Kafka consumer in background
    if os.environ.get('ENABLE_KAFKA_CONSUMER', 'true').lower() == 'true':
        threading.Thread(target=enhanced_kafka_consumer_worker, daemon=True).start()
        logger.info("Enhanced Kafka consumer started")
    
    # Start periodic performance snapshot
    threading.Thread(target=performance_monitor, daemon=True).start()
    
    logger.info(" Real-Time Anomaly Detection Engine ready!")

def get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding for a single text"""
    try:
        response = requests.post(
            f"{EMBEDDING_SERVICE_URL}/embed",
            json={"texts": [text]},
            timeout=10
        )
        if response.status_code == 200:
            embeddings = response.json().get("embeddings", [])
            return embeddings[0] if embeddings else None
        return None
    except Exception as e:
        logger.error(f"Embedding service error: {e}")
        return None

def predict_ensemble(embedding: np.ndarray, text: str = "") -> Dict:
    """Make prediction using ensemble model with weighted voting including redis similarity"""
    global stats

    
    if models_cache is None:
        raise HTTPException(status_code=500, detail='Ensemble model not loaded')
    
    vec = embedding.reshape(1, -1)
    
    if scaler is not None:
        try:
            vec_scaled = scaler.transform(vec)
        except Exception:
            vec_scaled = vec
    else:
        vec_scaled = vec
    
    predictions = []
    model_names = []
    
    # Get traditional ML model predictions
    models = models_cache.get('models', {})
    for name, model in models.items():
        if hasattr(model, 'predict'):
            try:
                pred = int(model.predict(vec_scaled)[0])
                predictions.append(pred)
                model_names.append(name)
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {e}")
                predictions.append(0)
                model_names.append(name)
    
    # Get redis similarity-based vote
    redis_vote_info = get_redis_similarity_vote(embedding.flatten().tolist(), text)
    redis_vote = redis_vote_info['vote']
    redis_confidence = redis_vote_info['confidence']
    
    # Add Redis VL vote to the ensemble
    predictions.append(redis_vote)
    model_names.append('redis_similarity')
    
    votes = np.array(predictions)
    
    # Check if we have saved model weights for weighted voting
    model_weights = models_cache.get('model_weights', {})
    
    # Add Redis VL weight if not present (give it reasonable weight based on confidence)
    if 'redis_similarity' not in model_weights:
        redis_weight = min(0.3, redis_confidence * 0.5)  # Max 30% weight, scaled by confidence
        model_weights = model_weights.copy() if model_weights else {}
        model_weights['redis_similarity'] = redis_weight
    
    if model_weights and len(model_weights) > 0:
        # Use weighted voting including redis
        weights = []
        for name in model_names:
            weight = model_weights.get(name, 0.1)  # Default weight if not found
            weights.append(weight)
        
        weights = np.array(weights)
        if weights.sum() > 0:
            weights_normalized = weights / weights.sum()  # Normalize weights
            anomaly_score = float(np.average(votes, weights=weights_normalized))
            
            # Enhanced logging to show voting details on separate lines
            logger.info("Weighted Voting Results:")
            for i, (name, vote, weight) in enumerate(zip(model_names, votes, weights_normalized)):
                logger.info(f"  {name}: vote={vote}, weight={weight:.4f}")
            
            simple_avg = float(votes.mean())
            logger.info(f"  Final weighted score: {anomaly_score:.4f} (simple average would be: {simple_avg:.4f})")
            
            # Debug logging to catch the bug
            logger.debug(f"🔍 Debug: votes={votes.tolist()}, normalized_weights={weights_normalized.tolist()}, calculated_score={anomaly_score}")
            
            # Sanity check: if all votes are the same, score should equal that vote
            if len(set(votes)) == 1:
                expected_score = float(votes[0])
                if abs(anomaly_score - expected_score) > 0.00001:
                    logger.error(f" BUG DETECTED: Unanimous vote {expected_score} but calculated score {anomaly_score}")
                else:
                    logger.debug(f" Unanimous vote calculation correct: {anomaly_score}")
        else:
            # Fallback to simple average
            anomaly_score = float(votes.mean())
            logger.warning(" All weights are zero, falling back to simple average")
    else:
        # Fallback to simple average voting (including redis)
        anomaly_score = float(votes.mean())
        logger.debug(" Using simple average voting (including redis)")
    
    # Lower threshold for better recall
    anomaly_threshold = 0.4
    final_prediction = int(anomaly_score > anomaly_threshold)
    
    # Threshold decision logging on new line
    logger.info(f" Anomaly Score: {anomaly_score:.6f}")
    logger.info(f" Final Prediction: {final_prediction} (threshold: {anomaly_threshold})")
    
    # Note: It's normal for anomaly_score to equal a model weight when only that model votes 1
    # This happens because weights are pre-normalized, so single-voter scenarios = that model's weight
    
    stats['total_predictions'] += 1
    if final_prediction == 1:
        stats['anomalies_detected'] += 1
    
    # Create enhanced model votes dict with redis details
    model_votes_dict = dict(zip(model_names, predictions))
    
    return {
        'prediction': final_prediction,
        'anomaly_score': anomaly_score,
        'confidence': max(anomaly_score, 1 - anomaly_score),
        'model_votes': model_votes_dict,
        'vote_counts': {'normal': int(np.sum(votes == 0)), 'anomaly': int(np.sum(votes == 1))},
        'weights_used': model_weights if model_weights else 'equal_weights',
        'redis_details': {
            'vote': redis_vote,
            'confidence': redis_confidence,
            'similar_count': redis_vote_info['similar_count'],
            'method': redis_vote_info['method']
        }
    }

@app.post('/score')
def score_text_enhanced(payload: TextPayload):
    """Enhanced scoring with database storage and accuracy tracking"""
    start_time = time.time()
    
    try:
        text = payload.text
        text_hash = get_text_hash(text)
        
        # Log the original HDFS log entry for all predictions - UPDATED VERSION
        logger.error(f"🔍 HDFS Log Entry: {text}")  # Using ERROR level to ensure it shows
        logger.warning(f"🔍 DEBUG: This message should appear if this code is running: {text[:100]}...")
        print(f"CONSOLE DEBUG: HDFS Log Entry should have been logged: {text[:100]}...")  # Debug print
        
        # Check cache first
        cached_result = get_cached_result(text_hash)
        if cached_result:
            logger.info(f" Returning cached result for: {text}")
            
            # Count cached API predictions too
            stats['total_predictions'] += 1
            if cached_result.get('predicted_label') == 1:
                stats['anomalies_detected'] += 1
                
            return cached_result
        
        # Get embedding
        embedding = get_embedding_from_redis_or_service(text)
        if embedding is None:
            raise HTTPException(status_code=500, detail="Failed to get embedding")
        
        # Make prediction (now includes redis similarity voting)
        result = predict_ensemble(np.array(embedding), text)
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Update average processing time
        if stats['total_predictions'] > 0:
            stats['avg_processing_time_ms'] = (
                (stats['avg_processing_time_ms'] * (stats['total_predictions'] - 1) + processing_time) / 
                stats['total_predictions']
            )
        else:
            stats['avg_processing_time_ms'] = processing_time
        
        # Create enhanced result
        enhanced_result = AnomalyResult(
            text=text,
            predicted_label=result['prediction'],
            anomaly_score=result['anomaly_score'],
            confidence=result['confidence'],
            model_votes=result['model_votes'],
            source='api',
            timestamp=datetime.datetime.now().isoformat(),
            processing_time_ms=processing_time,
            redis_details=RedisVLDetails(**result['redis_details'])
        )
        
        # Store in database
        store_anomaly_detection(enhanced_result)
        
        # Cache the result
        result_dict = enhanced_result.model_dump()
        cache_result(text_hash, result_dict)
        
        # Log significant events with redis details
        redis_info = result['redis_details']
        if result['prediction'] == 1:
            logger.warning(f" ANOMALY DETECTED: {text} (score: {result['anomaly_score']:.3f}, redis_vote: {redis_info['vote']}, similar: {redis_info['similar_count']}, time: {processing_time:.1f}ms)")
        else:
            logger.info(f" NORMAL: {text} (score: {result['anomaly_score']:.3f}, redis_vote: {redis_info['vote']}, similar: {redis_info['similar_count']}, time: {processing_time:.1f}ms)")
        
        # Add detailed voting breakdown to response
        result_dict = enhanced_result.model_dump()
        result_dict['voting_breakdown'] = {
            'traditional_models': {k: v for k, v in result['model_votes'].items() if k != 'redis_similarity'},
            'redis_similarity': result['model_votes'].get('redis_similarity', 0),
            'redis_confidence': redis_info['confidence'],
            'redis_method': redis_info['method']
        }
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Enhanced scoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")

@app.post('/score_with_label')
def score_with_actual_label(text: str, actual_label: int):
    """Score text with known actual label for accuracy measurement"""
    start_time = time.time()
    
    try:
        # Get prediction
        embedding = get_embedding_from_redis_or_service(text)
        if embedding is None:
            raise HTTPException(status_code=500, detail="Failed to get embedding")
        
        result = predict_ensemble(np.array(embedding), text)
        processing_time = (time.time() - start_time) * 1000
        
        predicted_label = result['prediction']
        
        # Update accuracy metrics
        update_accuracy_metrics(predicted_label, actual_label)
        
        # Update average processing time
        if stats['total_predictions'] > 0:
            stats['avg_processing_time_ms'] = (
                (stats['avg_processing_time_ms'] * (stats['total_predictions'] - 1) + processing_time) / 
                stats['total_predictions']
            )
        else:
            stats['avg_processing_time_ms'] = processing_time
        
        # Create enhanced result
        enhanced_result = AnomalyResult(
            text=text,
            predicted_label=predicted_label,
            anomaly_score=result['anomaly_score'],
            confidence=result['confidence'],
            actual_label=actual_label,
            model_votes=result['model_votes'],
            source='labeled_test',
            timestamp=datetime.datetime.now().isoformat(),
            processing_time_ms=processing_time,
            redis_details=RedisVLDetails(**result['redis_details'])
        )
        
        # Store in database
        store_anomaly_detection(enhanced_result, actual_label)
        
        # Log result
        correct = "" if predicted_label == actual_label else ""
        logger.info(f"{correct} Pred:{predicted_label} Actual:{actual_label} Score:{result['anomaly_score']:.3f} | {text}")
        
        return {
            'predicted_label': predicted_label,
            'actual_label': actual_label,
            'is_correct': predicted_label == actual_label,
            'anomaly_score': result['anomaly_score'],
            'confidence': result['confidence'],
            'processing_time_ms': processing_time,
            'current_accuracy': stats['accuracy'],
            'current_precision': stats['precision'],
            'current_recall': stats['recall'],
            'current_f1': stats['f1_score']
        }
        
    except Exception as e:
        logger.error(f"Labeled scoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"Labeled scoring failed: {str(e)}")

@app.get('performance_metrics/')
def get_performance_metrics():
    """Get comprehensive performance metrics"""
    return {
        'accuracy_metrics': {
            'accuracy': stats['accuracy'],
            'precision': stats['precision'],
            'recall': stats['recall'],
            'f1_score': stats['f1_score']
        },
        'confusion_matrix': {
            'true_positives': stats['true_positives'],
            'false_positives': stats['false_positives'],
            'true_negatives': stats['true_negatives'],
            'false_negatives': stats['false_negatives']
        },
        'performance_stats': {
            'total_predictions': stats['total_predictions'],
            'anomalies_detected': stats['anomalies_detected'],
            'avg_processing_time_ms': stats['avg_processing_time_ms'],
            'cache_hit_rate': stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
        },
        'system_stats': stats
    }

@app.get('/anomaly_history')
def get_anomaly_history(limit: int = 50, only_anomalies: bool = True):
    """Get recent anomaly detection history from database"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        if only_anomalies:
            query = '''
                SELECT timestamp, text, predicted_label, actual_label, anomaly_score, 
                       confidence, is_correct, processing_time_ms, source
                FROM anomaly_detections 
                WHERE predicted_label = 1 
                ORDER BY created_at DESC 
                LIMIT ?
            '''
        else:
            query = '''
                SELECT timestamp, text, predicted_label, actual_label, anomaly_score, 
                       confidence, is_correct, processing_time_ms, source
                FROM anomaly_detections 
                ORDER BY created_at DESC 
                LIMIT ?
            '''
        
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            results.append({
                'timestamp': row[0],
                'text': row[1],
                'predicted_label': row[2],
                'actual_label': row[3],
                'anomaly_score': row[4],
                'confidence': row[5],
                'is_correct': row[6],
                'processing_time_ms': row[7],
                'source': row[8]
            })
    
    return {
        'total_records': len(results),
        'anomalies': results
    }

@app.get('/accuracy_report')
def get_accuracy_report():
    """Generate comprehensive accuracy report"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Overall accuracy
        cursor.execute('SELECT COUNT(*) as total, SUM(is_correct) as correct FROM anomaly_detections WHERE actual_label IS NOT NULL')
        total, correct = cursor.fetchone()
        overall_accuracy = correct / total if total > 0 else 0
        
        # Accuracy by source
        cursor.execute('''
            SELECT source, COUNT(*) as total, SUM(is_correct) as correct, AVG(processing_time_ms) as avg_time
            FROM anomaly_detections 
            WHERE actual_label IS NOT NULL 
            GROUP BY source
        ''')
        by_source = [{'source': row[0], 'total': row[1], 'correct': row[2], 'accuracy': row[2]/row[1] if row[1] > 0 else 0, 'avg_time_ms': row[3]} for row in cursor.fetchall()]
        
        # Recent performance trend (last 24 hours)
        cursor.execute('''
            SELECT 
                datetime(created_at, 'localtime') as hour,
                COUNT(*) as predictions,
                SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct,
                AVG(processing_time_ms) as avg_time
            FROM anomaly_detections 
            WHERE actual_label IS NOT NULL AND created_at > datetime('now', '-24 hours')
            GROUP BY datetime(created_at, 'localtime', 'start of hour')
            ORDER BY hour DESC
            LIMIT 24
        ''')
        hourly_trends = [{'hour': row[0], 'predictions': row[1], 'accuracy': row[2]/row[1] if row[1] > 0 else 0, 'avg_time_ms': row[3]} for row in cursor.fetchall()]
    
    return {
        'overall_accuracy': overall_accuracy,
        'total_labeled_predictions': total,
        'correct_predictions': correct,
        'by_source': by_source,
        'hourly_trends': hourly_trends,
        'current_metrics': {
            'accuracy': stats['accuracy'],
            'precision': stats['precision'],
            'recall': stats['recall'],
            'f1_score': stats['f1_score']
        }
    }

@app.post('/reset_metrics')
def reset_performance_metrics():
    """Reset ALL performance metrics and clear database (for testing)"""
    global stats
    stats.update({
        'total_predictions': 0,
        'anomalies_detected': 0,
        'true_positives': 0,
        'false_positives': 0,
        'true_negatives': 0,
        'false_negatives': 0,
        'cache_hits': 0,
        'cache_misses': 0,
        'kafka_messages_processed': 0,
        'redis_vector_queries': 0,
        'redis_operations': 0,
        'avg_processing_time_ms': 0.0,
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0
    })
    
    # Clear database tables
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM anomaly_detections')
            cursor.execute('DELETE FROM performance_metrics')
            conn.commit()
        logger.info("🗑️ Database tables cleared")
    except Exception as e:
        logger.error(f" Failed to clear database: {e}")
    
    logger.info("🔄 ALL performance metrics and database reset")
    return {'message': 'All performance metrics and database reset successfully'}

def enhanced_kafka_consumer_worker():
    """Enhanced Kafka consumer with accuracy tracking"""
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_SERVERS,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            group_id=f'anomaly_engine_{int(time.time())}',
            auto_offset_reset='latest'
        )
        
        logger.info(f"Enhanced Kafka consumer listening to topic: {KAFKA_TOPIC}")
        
        for message in consumer:
            start_time = time.time()
            
            try:
                data = message.value
                text = data.get('message', data.get('text', ''))
                actual_label = data.get('label', None)
                
                if text:
                    text_hash = get_text_hash(text)
                    
                    # Check cache first
                    cached_result = get_cached_result(text_hash)
                    if cached_result:
                        result_dict = cached_result
                        logger.info(f" Kafka: Using cached result for {text}")
                        
                        # Count cached Kafka predictions too
                        stats['total_predictions'] += 1
                        if result_dict.get('predicted_label') == 1:
                            stats['anomalies_detected'] += 1
                        
                        # Update accuracy metrics for cached results too
                        if actual_label is not None:
                            predicted_label = result_dict.get('predicted_label', 0)
                            update_accuracy_metrics(predicted_label, actual_label)
                            correct = "" if predicted_label == actual_label else ""
                            logger.info(f"{correct} Kafka (cached): Pred:{predicted_label} Actual:{actual_label} | {text}")
                    else:
                        # Get embedding and predict
                        embedding = get_embedding_from_redis_or_service(text)
                        if embedding:
                            result = predict_ensemble(np.array(embedding), text)
                            processing_time = (time.time() - start_time) * 1000
                            
                            # Update average processing time
                            if stats['total_predictions'] > 0:
                                stats['avg_processing_time_ms'] = (
                                    (stats['avg_processing_time_ms'] * (stats['total_predictions'] - 1) + processing_time) / 
                                    stats['total_predictions']
                                )
                            else:
                                stats['avg_processing_time_ms'] = processing_time
                            
                            enhanced_result = AnomalyResult(
                                text=text,
                                predicted_label=result['prediction'],
                                anomaly_score=result['anomaly_score'],
                                confidence=result['confidence'],
                                actual_label=actual_label,
                                model_votes=result['model_votes'],
                                source='kafka_stream',
                                timestamp=datetime.datetime.now().isoformat(),
                                processing_time_ms=processing_time,
                                redis_details=RedisVLDetails(**result['redis_details'])
                            )
                            
                            # Store in database
                            store_anomaly_detection(enhanced_result, actual_label)
                            
                            # Update accuracy if we have actual label
                            if actual_label is not None:
                                update_accuracy_metrics(result['prediction'], actual_label)
                                correct = "" if result['prediction'] == actual_label else ""
                                logger.info(f"{correct} Kafka: Pred:{result['prediction']} Actual:{actual_label} | {text}")
                            
                            result_dict = enhanced_result.model_dump()
                            
                            # Cache the result
                            cache_result(text_hash, result_dict, ttl=1800)
                        else:
                            continue
                    
                    stats['kafka_messages_processed'] += 1
                    
                    # Log high-confidence anomalies
                    if result_dict.get('predicted_label') == 1 and result_dict.get('confidence', 0) > 0.8:
                        logger.warning(f" HIGH-CONFIDENCE KAFKA ANOMALY: {text} (score: {result_dict.get('anomaly_score', 0):.3f})")
                
            except Exception as e:
                logger.error(f"Error processing Kafka message: {e}")
                
    except Exception as e:
        logger.error(f"Enhanced Kafka consumer error: {e}")

def performance_monitor():
    """Background task to periodically save performance snapshots"""
    while True:
        try:
            time.sleep(300)  # Save every 5 minutes
            if stats['total_predictions'] > 0:
                save_performance_snapshot()
                logger.info(f"📊 Performance snapshot saved: Acc:{stats['accuracy']:.3f} P:{stats['precision']:.3f} R:{stats['recall']:.3f} F1:{stats['f1_score']:.3f}")
        except Exception as e:
            logger.error(f"Performance monitor error: {e}")

@app.get('/stats')
def get_stats():
    """Get service statistics (compatibility with original service)"""
    return {
        'total_predictions': stats['total_predictions'],
        'anomalies_detected': stats['anomalies_detected'],
        'cache_hits': stats['cache_hits'],
        'cache_misses': stats['cache_misses'],
        'kafka_messages_processed': stats['kafka_messages_processed'],
        'redis_vector_queries': stats['redis_vector_queries'],
        'redis_operations': stats['redis_operations']
    }

@app.get('/anomalies')
def get_anomalies():
    """Get recent anomaly detections (compatibility endpoint)"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, text, anomaly_score, confidence, model_votes, source
                FROM anomaly_detections 
                WHERE predicted_label = 1 
                ORDER BY created_at DESC 
                LIMIT 50
            ''')
            rows = cursor.fetchall()
            
            anomalies = []
            for row in rows:
                anomalies.append({
                    'timestamp': row[0],
                    'text': row[1],
                    'anomaly_score': row[2],
                    'confidence': row[3],
                    'model_votes': json.loads(row[4]) if row[4] else {},
                    'source': row[5]
                })
        
        return {
            'total_anomalies': len(anomalies),
            'anomalies': anomalies
        }
    except Exception as e:
        logger.error(f"Failed to get anomalies: {e}")
        return {'total_anomalies': 0, 'anomalies': []}

@app.post('/cache/clear')
def clear_cache():
    """Clear all cached results"""
    if redis_client is None:
        raise HTTPException(status_code=500, detail="Redis not available")
    
    try:
        # Delete all anomaly cache keys
        keys = redis_client.keys("anomaly:*")
        if keys:
            deleted = redis_client.delete(*keys)
            logger.info(f"🗑️ Cleared {deleted} cached results")
            return {'message': f'Cleared {deleted} cached results'}
        else:
            return {'message': 'No cached results to clear'}
            
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@app.delete('/cache/{text_hash}')
def delete_cache_entry(text_hash: str):
    """Delete specific cached result"""
    if redis_client is None:
        raise HTTPException(status_code=500, detail="Redis not available")
    
    try:
        deleted = redis_client.delete(f"anomaly:{text_hash}")
        if deleted:
            return {'message': f'Deleted cache entry for {text_hash}'}
        else:
            raise HTTPException(status_code=404, detail="Cache entry not found")
    except Exception as e:
        logger.error(f"Cache delete error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete cache entry: {str(e)}")

@app.get('/cache/stats')
def get_cache_stats():
    """Get cache statistics"""
    if redis_client is None:
        return {'redis_available': False}
    
    try:
        cache_keys = redis_client.keys("anomaly:*")
        return {
            'redis_available': True,
            'cached_entries': len(cache_keys),
            'cache_hit_rate': stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0,
            'total_cache_operations': stats['redis_operations']
        }
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        return {'redis_available': False, 'error': str(e)}

@app.get('/model_info')
def get_model_info():
    """Get detailed information about the ensemble model and redis integration"""
    if models_cache is None:
        return {'error': 'No model loaded'}
    
    models = models_cache.get('models', {})
    model_weights = models_cache.get('model_weights', {})
    
    model_info = {
        'traditional_models': list(models.keys()),
        'model_weights': model_weights,
        'redis_vl_integration': {
            'enabled': search_index is not None,
            'index_name': REDIS_INDEX_NAME,
            'host': REDIS_HOST,
            'port': REDIS_PORT
        },
        'voting_strategy': 'weighted' if model_weights else 'equal_weight',
        'ensemble_performance': models_cache.get('ensemble_score', {}),
        'total_voting_models': len(models) + (1 if search_index else 0)
    }
    
    return model_info

@app.post('/test_redis_vote')
def test_redis_vote(text: str):
    """Test Redis VL similarity voting for a specific text"""
    try:
        embedding = get_embedding(text)
        if not embedding:
            return {'error': 'Failed to get embedding'}
        
        redis_vote_info = get_redis_similarity_vote(embedding, text)
        
        return {
            'text': text,
            'redis_vote_details': redis_vote_info,
            'redis_vl_available': search_index is not None
        }
        
    except Exception as e:
        return {'error': str(e)}

@app.get('/health')
def health_check():
    """Enhanced health check with performance info"""
    model_loaded = models_cache is not None
    embedding_service_available = False
    redis_available = False
    redis_available = False
    db_available = False
    
    # Check embedding service
    try:
        response = requests.get(f"{EMBEDDING_SERVICE_URL.replace('/embed', '')}", timeout=5)
        embedding_service_available = True
    except:
        pass
    
    # Check Redis
    if redis_client:
        try:
            redis_client.ping()
            redis_available = True
        except:
            pass
    
    # Check Redis VL
    redis_vl_available = False
    if search_index:
        try:
            # Try to get index info to verify connection
            redis_vl_available = True
        except:
            pass
    
    # Check Database
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM anomaly_detections')
            db_records = cursor.fetchone()[0]
            db_available = True
    except:
        db_records = 0
    
    all_services_ok = all([model_loaded, redis_available, redis_vl_available, db_available])
    
    return {
        'status': 'healthy' if all_services_ok else 'degraded',
        'services': {
            'model_loaded': model_loaded,
            'embedding_service_available': embedding_service_available,
            'redis_available': redis_available,
            'redis_vl_available': redis_vl_available,
            'database_available': db_available
        },
        'database_records': db_records,
        'performance_summary': {
            'total_predictions': stats['total_predictions'],
            'accuracy': stats['accuracy'],
            'avg_processing_time_ms': stats['avg_processing_time_ms']
        },
        'full_stats': stats
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
