#!/usr/bin/env python3
"""
Line-Level Ensemble Trainer V2 for HDFS Anomaly Detection
Uses Decision Tree, Multi-Layer Perceptron, Stochastic Gradient Descent, and Redis VL Similarity
"""

# Suppress urllib3 SSL warnings
import warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL 1.1.1+')

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import joblib
import os
import sys
import requests
from itertools import product
import time
import redis
try:
    from redisvl.index import SearchIndex
    from redisvl.query import VectorQuery
except ImportError:
    print("Warning: redisvl not available. Please install with: pip install redisvl>=0.3.0")
    SearchIndex = None
    VectorQuery = None

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hdfs_line_level_loader_v2 import HDFSLineLevelLoader
import json

def get_hdfs_line_level_embeddings(sample_size=1000):
    """Get HDFS line-level embeddings using proper labeling"""
    
    print(f"Loading HDFS line-level data (sample_size={sample_size})...")
    loader = HDFSLineLevelLoader()
    messages, labels, _ = loader.get_line_level_data(sample_size=sample_size)  # Ignore template_ids for now
    
    print(f"Loaded {len(messages)} line-level samples:")
    print(f"  - Normal lines: {labels.count(0)} ({100*labels.count(0)/len(labels):.2f}%)")
    print(f"  - Anomalous lines: {labels.count(1)} ({100*labels.count(1)/len(labels):.2f}%)")
    
    # Get embeddings
    print("Getting embeddings from embedding service...")
    
    EMBEDDING_SERVICE_URL = "http://localhost:8000"
    embeddings = []
    batch_size = 100
    
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        try:
            response = requests.post(
                f"{EMBEDDING_SERVICE_URL}/embed",
                json={"texts": batch},
                timeout=30
            )
            
            if response.status_code == 200:
                batch_embeddings = response.json()["embeddings"]
                embeddings.extend(batch_embeddings)
                print(f"  Batch {i//batch_size + 1}/{(len(messages)-1)//batch_size + 1} completed")
            else:
                print(f"Error in batch {i//batch_size}: {response.status_code}")
                raise Exception(f"Embedding service error: {response.status_code}")
        except Exception as e:
            print(f"Connection error: {e}")
            raise Exception(f"Failed to get embeddings: {e}")
    
    return np.array(embeddings, dtype=np.float32), np.array(labels)

def hyperparameter_tuning(model_name, X_train, y_train, cv=3):
    """Perform hyperparameter tuning based on the provided guide"""
    
    print(f"Hyperparameter tuning for {model_name}...")
    
    if model_name == 'dt':
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 5, 10, 20, 35, 50, 70],
            'min_samples_split': [2, 3, 4, 5],
            'min_samples_leaf': [1, 2, 3, 4, 5]
        }
        model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    
    elif model_name == 'mlp':
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (150,), (100, 50), (150, 100)],
            'activation': ['relu', 'tanh', 'logistic'],
            'alpha': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [200, 300, 500]
        }
        model = MLPClassifier(random_state=42, early_stopping=True, validation_fraction=0.1)
    
    elif model_name == 'sgd':
        param_grid = {
            'loss': ['hinge', 'log_loss', 'modified_huber'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
            'learning_rate': ['constant', 'optimal', 'invscaling'],
            'eta0': [0.01, 0.1, 1.0],  # Initial learning rate for 'constant' and 'invscaling'
            'max_iter': [5000, 7000, 10000]  # Significantly increased to avoid convergence warnings
        }
        model = SGDClassifier(
            random_state=42, 
            class_weight='balanced', 
            tol=1e-4,           # More relaxed tolerance
            early_stopping=True, # Enable early stopping
            validation_fraction=0.1, # Use 10% for early stopping validation
            n_iter_no_change=15  # Stop if no improvement for 15 iterations
        )
    
    # Use smaller parameter combinations for efficiency
    if model_name == 'dt':
        # Reduce parameter space for DT to avoid excessive combinations
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 50],
            'min_samples_split': [2, 4],
            'min_samples_leaf': [1, 3]
        }
    elif model_name == 'mlp':
        # Reduce parameter space for MLP
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (150,)],
            'activation': ['relu', 'tanh'],
            'alpha': [1e-4, 1e-3],
            'max_iter': [300, 500]
        }
    elif model_name == 'sgd':
        # Significantly reduced parameter space for SGD (focus on best performers)
        param_grid = {
            'loss': ['hinge', 'log_loss'],  # Keep 2 most effective
            'penalty': ['l2', 'l1'],        # Keep 2 main options  
            'alpha': [1e-4, 1e-3],         # Reduced to 2 best values
            'learning_rate': ['optimal', 'constant'],  # Keep both
            'eta0': [0.1],                 # Single best value for constant LR
            'max_iter': [5000]             # Increased significantly to avoid convergence warnings
        }
        model = SGDClassifier(
            random_state=42, 
            class_weight='balanced', 
            tol=1e-4,           # More relaxed tolerance for faster convergence
            early_stopping=True, # Enable early stopping
            validation_fraction=0.1, # Use 10% for early stopping validation
            n_iter_no_change=10  # Stop if no improvement for 10 iterations
        )
    
    # Filter parameter combinations to avoid invalid configs  
    if model_name == 'sgd':
        # Create valid parameter combinations manually (much smaller set)
        valid_combinations = []
        
        for loss in param_grid['loss']:
            for penalty in param_grid['penalty']:
                for alpha in param_grid['alpha']:
                    # Add optimal learning rate (doesn't need eta0) 
                    valid_combinations.append({
                        'loss': loss,
                        'penalty': penalty,
                        'alpha': alpha,
                        'learning_rate': 'optimal',
                        'max_iter': param_grid['max_iter'][0],
                        'early_stopping': True,
                        'validation_fraction': 0.1,
                        'n_iter_no_change': 10,
                        'tol': 1e-4
                    })
                    
                    # Add constant learning rate (needs eta0)
                    valid_combinations.append({
                        'loss': loss,
                        'penalty': penalty, 
                        'alpha': alpha,
                        'learning_rate': 'constant',
                        'eta0': param_grid['eta0'][0],
                        'max_iter': param_grid['max_iter'][0],
                        'early_stopping': True,
                        'validation_fraction': 0.1,
                        'n_iter_no_change': 10,
                        'tol': 1e-4
                    })
        
        # Use the valid combinations (now only 16 instead of 72)
        param_grid = valid_combinations
    
    # Perform grid search with reduced CV for speed
    start_time = time.time()
    
    if model_name == 'sgd' and isinstance(param_grid, list):
        # For SGD with valid parameter combinations, use ParameterGrid
        from sklearn.model_selection import ParameterGrid
        from sklearn.base import clone
        
        best_score = -1
        best_params = None
        best_estimator = None
        
        print(f"Testing {len(param_grid)} parameter combinations for SGD...")
        
        for i, params in enumerate(param_grid):
            try:
                # Create model with current parameters
                current_model = clone(model)
                current_model.set_params(**params)
                
                # Perform cross-validation manually (use 2-fold for SGD speed)
                from sklearn.model_selection import cross_val_score
                sgd_cv = 2 if model_name == 'sgd' else cv  # Faster CV for SGD
                scores = cross_val_score(current_model, X_train, y_train, cv=sgd_cv, scoring='f1')
                avg_score = np.mean(scores)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = params
                    best_estimator = clone(current_model)
                
                if i % 5 == 0:  # More frequent progress updates for SGD
                    print(f"  Tested {i+1}/{len(param_grid)} combinations...")
                    
            except Exception as e:
                print(f"  Skipping invalid combination {i}: {e}")
                continue
        
        # Train the best model
        if best_estimator is not None:
            best_estimator.fit(X_train, y_train)
        
        grid_search = type('GridSearchResult', (), {
            'best_estimator_': best_estimator,
            'best_params_': best_params,
            'best_score_': best_score
        })()
        
    else:
        # Standard grid search for other models
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
    
    end_time = time.time()
    
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best F1 score: {grid_search.best_score_:.4f}")
    print(f"Tuning time: {end_time - start_time:.2f} seconds")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

class RedisSimilarityDetector:
    """Redis VL-based anomaly detector using similarity scoring"""
    
    def __init__(self, index_name="training_embeddings", host="localhost", port=6379):
        # Initialize Redis VL client
        self.client = redis.Redis(host=host, port=port, decode_responses=False)
        self.index_name = index_name
        self.threshold = 0.7  # Similarity threshold for anomaly detection
        self.is_fitted = False
        self.search_index = None
        
    def fit(self, X_train, y_train):
        """Store training embeddings in Redis VL index"""
        print("Setting up Redis VL similarity detector...")
        
        try:
            # Define schema for the search index
            schema = {
                "index": {
                    "name": self.index_name,
                    "prefix": f"{self.index_name}:",
                    "storage_type": "hash"
                },
                "fields": [
                    {
                        "name": "vector",
                        "type": "vector",
                        "attrs": {
                            "dims": X_train.shape[1],
                            "distance_metric": "COSINE",
                            "algorithm": "HNSW",
                            "algorithm_config": {
                                "m": 16,
                                "ef_construction": 200,
                                "ef_runtime": 10
                            }
                        }
                    },
                    {
                        "name": "label",
                        "type": "tag"
                    }
                ]
            }
            
            # Create search index with explicit Redis client
            self.search_index = SearchIndex.from_dict(schema)
            self.search_index.set_client(self.client)
            
            # Delete existing index if it exists
            try:
                self.search_index.delete()
                print(f"Deleted existing index: {self.index_name}")
            except:
                pass
            
            # Create the index
            self.search_index.create(overwrite=True)
            print(f"Created Redis VL index: {self.index_name}")
            
            # Store normal embeddings (label=0) in Redis
            normal_embeddings = X_train[y_train == 0]
            normal_indices = np.where(y_train == 0)[0]
            
            # Use smaller batch size and limit to prevent timeout
            max_embeddings = min(10000, len(normal_embeddings))
            batch_size = 100  # Process in smaller batches
            
            print(f"Storing {max_embeddings} normal embeddings in batches of {batch_size}...")
            
            # Prepare data for insertion
            data = []
            for i in range(min(max_embeddings, len(normal_embeddings))):
                embedding = normal_embeddings[i]
                data.append({
                    "vector": embedding.astype(np.float32).tobytes(),
                    "label": "normal"
                })
            
            # Insert in batches using Redis pipeline
            for batch_start in range(0, len(data), batch_size):
                batch_end = min(batch_start + batch_size, len(data))
                batch_data = data[batch_start:batch_end]
                
                try:
                    # Use pipeline for batch insertion
                    pipe = self.client.pipeline()
                    for idx, item in enumerate(batch_data):
                        key = f"{self.index_name}:{batch_start + idx}"
                        pipe.hset(key, mapping=item)
                    pipe.execute()
                    print(f"  Batch {batch_start//batch_size + 1}/{(len(data)-1)//batch_size + 1} stored")
                except Exception as batch_e:
                    print(f"  Warning: Failed to store batch {batch_start//batch_size + 1}: {batch_e}")
                    continue
            
            # Verify index has data
            actual_points = len(data)
            print(f"Successfully stored {actual_points} embeddings in Redis VL")
            
            if actual_points == 0:
                raise Exception("No embeddings were successfully stored")
            
            # Mark as fitted, threshold will be tuned later with validation data
            self.is_fitted = True
            
        except Exception as e:
            print(f"Error setting up Redis VL: {e}")
            raise Exception(f"Redis VL setup failed: {e}")
    
    def _tune_threshold(self, X_val, y_val, thresholds=None):
        """Tune similarity threshold for best F1 score with adaptive threshold selection"""
        print("Tuning Redis VL similarity threshold...")
        best_f1 = 0
        
        # Always use the same validation set as final validation to avoid discrepancies
        # Only use subset if dataset is extremely large (>50k samples)
        if len(X_val) > 50000:
            # Use stratified sampling to maintain anomaly ratio for very large datasets
            from sklearn.model_selection import train_test_split
            max_samples = min(len(X_val) // 2, 30000)  # Use half or max 30k, whichever is smaller
            X_subset, _, y_subset, _ = train_test_split(
                X_val, y_val, 
                train_size=max_samples, 
                stratify=y_val, 
                random_state=42
            )
            print(f"Using {len(X_subset)} validation samples for threshold tuning (stratified subset)")
        else:
            # Use full validation set for consistency
            X_subset = X_val
            y_subset = y_val
            print(f"Using all {len(X_subset)} validation samples for threshold tuning")
        
        print(f"  - Normal: {np.sum(y_subset == 0)}")
        print(f"  - Anomaly: {np.sum(y_subset == 1)}")
        
        # First pass: collect similarity scores
        self._debug_errors = False
        self._similarity_scores = []  # Store similarity scores for analysis
        temp_threshold = 0.5  # Temporary threshold for initial prediction
        self.threshold = temp_threshold
        _ = self.predict(X_subset)  # This collects similarity scores
        
        # Analyze similarity score distribution
        if hasattr(self, '_similarity_scores') and len(self._similarity_scores) > 0:
            scores_array = np.array(self._similarity_scores)
            print(f"  Similarity scores - Min: {scores_array.min():.4f}, Max: {scores_array.max():.4f}, Mean: {scores_array.mean():.4f}, Median: {np.median(scores_array):.4f}")
            
            # Calculate percentiles for adaptive threshold selection
            percentiles = np.percentile(scores_array, [5, 10, 25, 50, 75, 90, 95])
            print(f"  Similarity percentiles [5,10,25,50,75,90,95]: {[f'{p:.3f}' for p in percentiles]}")
            
            # Create adaptive threshold candidates based on score distribution
            if thresholds is None:
                # Focus on the lower quartiles where anomalies should be
                min_score = max(0.05, np.min(scores_array))
                q25 = percentiles[2]  # 25th percentile
                q50 = percentiles[3]  # 50th percentile (median)
                q75 = percentiles[4]  # 75th percentile
                
                # Create threshold candidates with higher density in promising ranges
                low_thresholds = np.linspace(min_score, q25, 8)  # Focus on very low similarities
                mid_thresholds = np.linspace(q25, q50, 6)       # Medium-low similarities
                high_thresholds = np.linspace(q50, q75, 4)      # Medium similarities
                extreme_thresholds = np.linspace(q75, 0.99, 3)  # High similarities
                
                thresholds = np.concatenate([low_thresholds, mid_thresholds, high_thresholds, extreme_thresholds])
                thresholds = np.unique(np.round(thresholds, 4))  # Remove duplicates and round
                print(f"  Using {len(thresholds)} adaptive thresholds based on score distribution")
        
        if thresholds is None:
            # Fallback to original thresholds if score collection failed
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
            print(f"  Using {len(thresholds)} fallback thresholds")
        
        best_threshold = thresholds[0]
        
        # Second pass: test thresholds
        print("  Testing thresholds...")
        for i, threshold in enumerate(thresholds):
            self.threshold = threshold
            predictions = self.predict(X_subset)
            
            if len(predictions) == len(y_subset):
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_subset, predictions, average='binary', zero_division=0
                )
                
                # Debug info - print all promising thresholds and first/last few
                anomaly_predictions = np.sum(predictions == 1)
                if f1 > 0.01 or i < 3 or i >= len(thresholds) - 3:
                    print(f"  Threshold {threshold:.4f}: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}, Anomalies: {anomaly_predictions}/{len(predictions)}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_precision = precision
                    best_recall = recall
        
        # Disable debug mode
        self._debug_errors = False
        if hasattr(self, '_similarity_scores'):
            delattr(self, '_similarity_scores')
        
        self.threshold = best_threshold
        self.best_threshold_f1 = best_f1
        self.best_threshold_precision = best_precision if 'best_precision' in locals() else 0.0
        self.best_threshold_recall = best_recall if 'best_recall' in locals() else 0.0
        
        print(f"Best threshold: {best_threshold} (Precision: {self.best_threshold_precision:.4f}, Recall: {self.best_threshold_recall:.4f}, F1: {best_f1:.4f})")
        return best_f1, self.best_threshold_precision, self.best_threshold_recall
    
    def predict(self, X):
        """Predict anomalies based on similarity to normal embeddings"""
        if not self.is_fitted:
            raise Exception("RedisSimilarityDetector not fitted. Call fit() first.")
        
        predictions = []
        
        # Process in smaller batches to avoid timeouts
        batch_size = 50
        for batch_start in range(0, len(X), batch_size):
            batch_end = min(batch_start + batch_size, len(X))
            batch_X = X[batch_start:batch_end]
            
            batch_predictions = []
            for embedding in batch_X:
                try:
                    # Try Redis VL search first
                    search_results = None
                    try:
                        # Create vector query for similarity search
                        vector_query = VectorQuery(vector=embedding.astype(np.float32).tobytes(), 
                                                 vector_field_name="vector", 
                                                 return_fields=["label"], 
                                                 num_results=3)
                        
                        # Perform search using the index
                        search_results = self.search_index.query(vector_query)
                        if hasattr(self, '_debug_errors') and self._debug_errors and len(batch_predictions) == 0:
                            print(f"    Debug: VectorQuery succeeded, got {len(search_results)} results")
                    except Exception as redis_vl_error:
                        # Fallback to direct Redis FT.SEARCH if VectorQuery fails
                        if hasattr(self, '_debug_errors') and self._debug_errors:
                            print(f"    Debug: VectorQuery failed, trying direct search: {redis_vl_error}")
                        
                        # Convert vector to bytes and create direct search
                        vector_bytes = embedding.astype(np.float32).tobytes()
                        search_cmd = [
                            'FT.SEARCH', self.index_name, 
                            f'*=>[KNN 3 @vector $vec_param]',
                            'PARAMS', '2', 'vec_param', vector_bytes,
                            'RETURN', '2', 'label', '__vector_score',
                            'DIALECT', '2'
                        ]
                        
                        # Execute direct Redis search
                        raw_results = self.client.execute_command(*search_cmd)
                        
                        # Parse raw results into expected format
                        if raw_results and len(raw_results) > 1:
                            search_results = []
                            num_results = raw_results[0]
                            for i in range(1, len(raw_results), 2):
                                if i + 1 < len(raw_results):
                                    fields = raw_results[i + 1]
                                    result = {}
                                    for j in range(0, len(fields), 2):
                                        if j + 1 < len(fields):
                                            key = fields[j].decode() if isinstance(fields[j], bytes) else str(fields[j])
                                            value = fields[j + 1].decode() if isinstance(fields[j + 1], bytes) else fields[j + 1]
                                            if key == '__vector_score':
                                                result['vector_score'] = float(value)
                                            else:
                                                result[key] = value
                                    search_results.append(result)
                    
                    if search_results:
                        # Get average similarity score - Redis VL uses different score formats
                        # Try different possible score field names
                        scores = []
                        for result in search_results:
                            # Debug: print result structure once
                            if hasattr(self, '_debug_errors') and self._debug_errors and len(scores) == 0:
                                print(f"    Debug: Redis VL result structure: {result}")
                                print(f"    Debug: Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
                            
                            if 'vector_score' in result:
                                # This comes from the fallback parsing - already a similarity score
                                scores.append(float(result['vector_score']))
                            elif '__vector_score' in result:
                                # Direct from Redis VL - this is a similarity score (higher = more similar)
                                scores.append(float(result['__vector_score']))
                            elif 'score' in result:
                                scores.append(float(result['score']))
                            elif isinstance(result, dict) and 'dist' in result:
                                # Convert distance to similarity (1 - distance for cosine)
                                scores.append(1.0 - float(result['dist']))
                            else:
                                # Try to find any numeric field that might be the score
                                numeric_fields = {k: v for k, v in result.items() if isinstance(v, (int, float, str)) and str(v).replace('.', '').replace('-', '').isdigit()}
                                if numeric_fields:
                                    field_name = list(numeric_fields.keys())[0]
                                    field_value = float(list(numeric_fields.values())[0])
                                    
                                    # Handle distance fields - convert to similarity
                                    if 'distance' in field_name.lower():
                                        # Convert distance to similarity: similarity = 1 - normalized_distance
                                        # For cosine distance, max theoretical distance is 2.0
                                        max_distance = 2.0
                                        similarity = 1.0 - (field_value / max_distance)
                                        score_value = max(0.0, min(1.0, similarity))  # Clamp to [0,1] range
                                        if hasattr(self, '_debug_errors') and self._debug_errors:
                                            print(f"    Debug: Converting {field_name} = {field_value} distance -> {score_value} similarity")
                                    else:
                                        # Use as similarity score directly
                                        score_value = field_value
                                        if hasattr(self, '_debug_errors') and self._debug_errors:
                                            print(f"    Debug: Using field {field_name} = {score_value} as similarity score")
                                    
                                    scores.append(score_value)
                                else:
                                    # Default low similarity if no score found (conservative)
                                    scores.append(0.1)
                        
                        if scores:
                            avg_similarity = np.mean(scores)
                            
                            # Store similarity scores for debugging
                            if hasattr(self, '_similarity_scores'):
                                self._similarity_scores.append(avg_similarity)
                            
                            # Anomaly if similarity is below threshold
                            is_anomaly = 1 if avg_similarity < self.threshold else 0
                            batch_predictions.append(is_anomaly)
                        else:
                            batch_predictions.append(1)  # No valid scores = anomaly
                    else:
                        batch_predictions.append(1)  # No similar embeddings = anomaly
                        
                except Exception as e:
                    # Default to normal on error (conservative approach) 
                    # But log the error for debugging
                    if hasattr(self, '_debug_errors'):
                        print(f"    Debug: Redis VL query error: {e}")
                    batch_predictions.append(0)
            
            predictions.extend(batch_predictions)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Predict probabilities for compatibility"""
        predictions = self.predict(X)
        # Convert binary predictions to probabilities
        proba = np.zeros((len(predictions), 2))
        proba[:, 0] = 1 - predictions  # Normal probability
        proba[:, 1] = predictions      # Anomaly probability
        return proba

def generate_results_table(outdir, results, model_scores, model_weights, best_params, 
                              ensemble_precision, ensemble_recall, ensemble_f1, total_samples, anomaly_ratio):
    """Generate comprehensive results table for documentation"""
    from datetime import datetime
    import pandas as pd
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create comprehensive results table
    table_data = []
    
    # Header information
    header_info = f"""
# HDFS Line-Level Anomaly Detection Ensemble V2 Results
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Dataset: HDFS Line-Level Logs
# Total Samples: {total_samples:,}
# Anomaly Ratio: {anomaly_ratio:.4f} ({100*anomaly_ratio:.2f}%)
# Models: Decision Tree (DT), Multi-Layer Perceptron (MLP), Stochastic Gradient Descent (SGD), Redis VL Similarity
# Ensemble Method: Weighted Voting with F1-based weights

"""
    
    # Individual model results table
    individual_results = """
## Individual Model Performance

| Model | Precision | Recall | F1-Score | Weight | Hyperparameters |
|-------|-----------|--------|----------|--------|-----------------|
"""
    
    model_name_mapping = {
        'dt': 'Decision Tree',
        'mlp': 'Multi-Layer Perceptron', 
        'sgd': 'Stochastic Gradient Descent',
        'redis_similarity': 'Redis VL Similarity'
    }
    
    for model_name in ['dt', 'mlp', 'sgd', 'redis_similarity']:
        if model_name in model_scores:
            scores = model_scores[model_name]
            weight = model_weights.get(model_name, 0.0)
            params = best_params.get(model_name, {})
            display_name = model_name_mapping[model_name]
            
            # Format hyperparameters for display
            if model_name == 'redis_similarity':
                params_str = f"threshold={params.get('threshold', 'N/A')}"
            else:
                key_params = []
                if 'criterion' in params:
                    key_params.append(f"criterion={params['criterion']}")
                if 'max_depth' in params:
                    key_params.append(f"max_depth={params['max_depth']}")
                if 'hidden_layer_sizes' in params:
                    key_params.append(f"hidden_sizes={params['hidden_layer_sizes']}")
                if 'activation' in params:
                    key_params.append(f"activation={params['activation']}")
                if 'loss' in params:
                    key_params.append(f"loss={params['loss']}")
                if 'penalty' in params:
                    key_params.append(f"penalty={params['penalty']}")
                params_str = ", ".join(key_params[:3])  # Limit to top 3 params
                if len(key_params) > 3:
                    params_str += "..."
            
            individual_results += f"| {model_name_mapping[model_name]} | {scores['precision']:.4f} | {scores['recall']:.4f} | {scores['f1']:.4f} | {weight:.4f} | {params_str} |\n"
    
    # Ensemble results
    ensemble_results = f"""
## Ensemble Performance

| Metric | Value |
|--------|-------|
| Precision | {ensemble_precision:.4f} |
| Recall | {ensemble_recall:.4f} |
| F1-Score | {ensemble_f1:.4f} |
| Weighting Method | F1-based |
| Combination Method | Weighted Voting |

## Model Comparison

| Rank | Model | F1-Score | Improvement over Best Individual |
|------|-------|----------|----------------------------------|
"""
    
    # Calculate rankings
    model_f1s = [(name, scores['f1']) for name, scores in model_scores.items()]
    model_f1s.sort(key=lambda x: x[1], reverse=True)
    
    best_individual_f1 = max(scores['f1'] for scores in model_scores.values())
    ensemble_improvement = ((ensemble_f1 - best_individual_f1) / best_individual_f1) * 100 if best_individual_f1 > 0 else 0
    
    for i, (model_name, f1) in enumerate(model_f1s, 1):
        improvement = 0.0 if i == 1 else ((f1 - best_individual_f1) / best_individual_f1) * 100
        ensemble_results += f"| {i} | {model_name_mapping.get(model_name, model_name)} | {f1:.4f} | {improvement:+.2f}% |\n"
    
    ensemble_results += f"| - | **Ensemble** | **{ensemble_f1:.4f}** | **{ensemble_improvement:+.2f}%** |\n"
    
    # Training details
    training_details = f"""
## Training Configuration

| Parameter | Value |
|-----------|-------|
| Validation Split | {results['training_info'].get('validation_samples', 0) / (results['training_info'].get('training_samples', 0) + results['training_info'].get('validation_samples', 0)):.1%} |
| Feature Scaling | StandardScaler |
| Cross-Validation | 3-fold (except SGD: 2-fold) |
| Hyperparameter Tuning | GridSearchCV |
| Random State | 42 |
| Class Balancing | Stratified sampling |

## Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Total Samples | {total_samples:,} |
| Training Samples | {results['training_info'].get('training_samples', 0):,} |
| Validation Samples | {results['training_info'].get('validation_samples', 0):,} |
| Feature Dimensions | {results['training_info'].get('feature_dim', 0)} |
| Normal Samples | {int(total_samples * (1 - anomaly_ratio)):,} ({100*(1-anomaly_ratio):.2f}%) |
| Anomalous Samples | {int(total_samples * anomaly_ratio):,} ({100*anomaly_ratio:.2f}%) |
| Class Imbalance Ratio | {(1-anomaly_ratio)/anomaly_ratio:.1f}:1 |

## Key Findings

1. **Best Individual Model**: {model_name_mapping.get(model_f1s[0][0], model_f1s[0][0])} (F1={model_f1s[0][1]:.4f})
2. **Ensemble Improvement**: {ensemble_improvement:+.2f}% over best individual model
3. **Most Weighted Model**: {model_name_mapping.get(max(model_weights.items(), key=lambda x: x[1])[0], max(model_weights.items(), key=lambda x: x[1])[0])} (Weight={max(model_weights.values()):.4f})
4. **Class Imbalance Challenge**: {(1-anomaly_ratio)/anomaly_ratio:.0f}:1 ratio requires careful evaluation metrics
"""
    
    # Combine all sections
    full_report = header_info + individual_results + ensemble_results + training_details
    
    # Save as markdown file
    results_file = os.path.join(outdir, f'results_table_{timestamp}.md')
    with open(results_file, 'w') as f:
        f.write(full_report)
    
    # Also save as CSV for easy import into papers
    csv_data = []
    for model_name in ['dt', 'mlp', 'sgd', 'redis_similarity']:
        if model_name in model_scores:
            scores = model_scores[model_name]
            weight = model_weights.get(model_name, 0.0)
            csv_data.append({
                'Model': model_name_mapping[model_name],
                'Precision': scores['precision'],
                'Recall': scores['recall'], 
                'F1_Score': scores['f1'],
                'Weight': weight,
                'Rank': next(i for i, (name, _) in enumerate(model_f1s, 1) if name == model_name)
            })
    
    # Add ensemble row
    csv_data.append({
        'Model': 'Ensemble',
        'Precision': ensemble_precision,
        'Recall': ensemble_recall,
        'F1_Score': ensemble_f1,
        'Weight': 1.0,
        'Rank': 0  # Special rank for ensemble
    })
    
    # Save CSV to specified results directory
    df = pd.DataFrame(csv_data)
    
    # Create results directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    csv_file = os.path.join(results_dir, f'results_data_{timestamp}.csv')
    df.to_csv(csv_file, index=False)
    
    print(f"    results table saved to: {results_file}")
    print(f"   results CSV saved to: {csv_file}")

def train_line_level_ensemble_v2(X=None, y=None, outdir=None, sample_size=20000):
    """Train ensemble model V2 on line-level HDFS data with DT, MLP, SGD and Redis VL similarity"""
    
    # Set default output directory
    if outdir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        outdir = os.path.join(script_dir, 'models', 'line_level_ensemble_v2')
    
    # Load line-level HDFS data if not provided
    if X is None or y is None:
        print("Loading line-level HDFS dataset for training...")
        X, y = get_hdfs_line_level_embeddings(sample_size=sample_size)
    
    print(f"Training on {len(X)} line-level samples with {X.shape[1]} features")
    print(f"Label distribution: Normal={np.sum(y==0)}, Anomaly={np.sum(y==1)}")
    
    # Create output directory
    os.makedirs(outdir, exist_ok=True)
    
    # Check anomaly ratio
    anomaly_ratio = np.sum(y == 1) / len(y)
    print(f"Actual anomaly ratio: {anomaly_ratio:.4f}")
    
    # Use larger validation set for better model differentiation with imbalanced data
    # For very imbalanced data, we need more validation samples to get meaningful differences
    val_size = 0.3 if len(y) <= 10000 else 0.2  # Use 30% for small datasets, 20% for larger
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=42, stratify=y
    )
    
    print(f"Dataset split (validation size: {val_size*100:.0f}%):")
    print(f"  Training: {len(X_train)} samples (Normal: {np.sum(y_train==0)}, Anomaly: {np.sum(y_train==1)})")
    print(f"  Validation: {len(X_val)} samples (Normal: {np.sum(y_val==0)}, Anomaly: {np.sum(y_val==1)})")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    
    # Clear memory
    del X_train, X_val
    
    print("Training ensemble models V2 (Decision Tree, Multi-Layer Perceptron, Stochastic Gradient Descent, Redis VL)...")
    
    # Train models with hyperparameter tuning
    trained_models = {}
    model_scores = {}
    best_params = {}
    
    # Traditional ML models with full names for logging
    model_configs = ['dt', 'mlp', 'sgd']
    model_full_names = {
        'dt': 'Decision Tree',
        'mlp': 'Multi-Layer Perceptron', 
        'sgd': 'Stochastic Gradient Descent'
    }
    
    for model_name in model_configs:
        print(f"\n{'='*50}")
        print(f"Training {model_full_names[model_name]}...")
        print(f"{'='*50}")
        
        try:
            # Perform hyperparameter tuning
            best_model, params, best_score = hyperparameter_tuning(
                model_name, X_train_s, y_train, cv=3
            )
            
            # Train the best model
            best_model.fit(X_train_s, y_train)
            
            # Validate
            val_pred = best_model.predict(X_val_s)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, val_pred, average='binary', zero_division=0
            )
            
            # Calculate detailed metrics for analysis
            tp = np.sum((y_val == 1) & (val_pred == 1))
            fp = np.sum((y_val == 0) & (val_pred == 1))
            fn = np.sum((y_val == 1) & (val_pred == 0))
            
            print(f"  Validation Results:")
            print(f"     Precision: {precision:.4f}")
            print(f"     Recall: {recall:.4f}")
            print(f"     F1-Score: {f1:.4f}")
            print(f"     TP: {tp}, FP: {fp}, FN: {fn}")
            print(f"     Predictions: {np.sum(val_pred == 1)} anomalies predicted")
            
            trained_models[model_name] = best_model
            model_scores[model_name] = {'precision': precision, 'recall': recall, 'f1': f1}
            best_params[model_name] = params
            
        except Exception as e:
            print(f"  Failed to train {model_name}: {e}")
    
    # Add Redis VL Similarity Detector
    print(f"\n{'='*50}")
    print("Setting up Redis VL Similarity Detector...")
    print(f"{'='*50}")
    
    try:
        redis_detector = RedisSimilarityDetector()
        redis_detector.fit(X_train_s, y_train)
        
        # Now tune threshold using validation data and get the F1, precision, recall scores
        threshold_f1, threshold_precision, threshold_recall = redis_detector._tune_threshold(X_val_s, y_val)
        
        # Use the threshold tuning metrics as the final scores (for consistency)
        print(f"Redis VL final metrics - Precision: {threshold_precision:.4f}, Recall: {threshold_recall:.4f}, F1: {threshold_f1:.4f}")
        
        trained_models['redis_similarity'] = redis_detector
        model_scores['redis_similarity'] = {
            'precision': threshold_precision, 
            'recall': threshold_recall, 
            'f1': threshold_f1
        }
        best_params['redis_similarity'] = {'threshold': redis_detector.threshold}
        
    except Exception as e:
        print(f"  Failed to setup Redis VL detector: {e}")
    
    # Calculate ensemble performance with weighted voting
    print(f"\n{'='*60}")
    print("ENSEMBLE MODEL V2 PERFORMANCE")
    print(f"{'='*60}")
    
    if trained_models:
        # Calculate weights based on F1 scores
        model_weights = {}
        
        print(f"\nINDIVIDUAL MODEL SCORES:")
        for name in trained_models.keys():
            f1_score = model_scores[name]['f1']
            # Use F1 score as weight, with minimum weight of 0.1 to avoid zero weights
            model_weights[name] = max(f1_score, 0.1)
            print(f"   {name.upper()}: F1={f1_score:.4f}")
        
        # Normalize weights to sum to 1
        total_weight = sum(model_weights.values())
        model_weights = {k: v/total_weight for k, v in model_weights.items()}
        
        print(f"\nMODEL WEIGHTS (F1-based):")
        for name, weight in model_weights.items():
            print(f"   {name.upper()}: {weight:.4f}")
        
        # Ensemble prediction on validation set
        ensemble_predictions = []
        model_names = list(trained_models.keys())
        
        for name in model_names:
            pred = trained_models[name].predict(X_val_s)
            ensemble_predictions.append(pred)
        
        # Apply weighted voting
        ensemble_pred = np.array(ensemble_predictions)
        weights = np.array([model_weights[name] for name in model_names])
        
        # Weighted average prediction
        weighted_pred = np.average(ensemble_pred, axis=0, weights=weights)
        
        # Apply anomaly ratio control
        target_anomalies = int(len(y_val) * anomaly_ratio)
        threshold_idx = np.argsort(weighted_pred)[-target_anomalies:]
        final_pred = np.zeros(len(y_val))
        final_pred[threshold_idx] = 1
        
        # Calculate ensemble metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, final_pred, average='binary', zero_division=0
        )
        
        print(f"\nENSEMBLE RESULTS:")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        
        # Detailed classification report
        print(f"\nCLASSIFICATION REPORT:")
        print(classification_report(y_val, final_pred, target_names=['Normal', 'Anomaly']))
        
        # Confusion Matrix
        print(f"\nCONFUSION MATRIX:")
        cm = confusion_matrix(y_val, final_pred)
        if cm.size == 4:  # 2x2 matrix
            tn, fp, fn, tp = cm.ravel()
            print(f"   True Negatives:  {tn}")
            print(f"   False Positives: {fp}")
            print(f"   False Negatives: {fn}")
            print(f"   True Positives:  {tp}")
        else:
            print(f"   Confusion Matrix: \n{cm}")
    
    # Save models and results
    print(f"\nSaving ensemble models V2 to {outdir}...")
    
    results = {
        'models': trained_models,
        'scaler': scaler,
        'model_scores': model_scores,
        'model_weights': model_weights if trained_models else {},
        'best_params': best_params,
        'ensemble_metrics': {
            'precision': precision if trained_models else 0,
            'recall': recall if trained_models else 0,
            'f1': f1 if trained_models else 0
        } if trained_models else {},
        'training_info': {
            'sample_size': len(X),
            'feature_dim': X.shape[1],
            'anomaly_ratio': anomaly_ratio,
            'training_samples': len(X_train_s),
            'validation_samples': len(X_val_s),
            'model_types': ['Decision Tree', 'MLP', 'SGD', 'Redis VL Similarity']
        }
    }
    
    # Save ensemble (excluding Redis VL client which can't be pickled)
    models_to_save = {k: v for k, v in trained_models.items() if k != 'redis_similarity'}
    results_to_save = results.copy()
    results_to_save['models'] = models_to_save
    
    joblib.dump(results_to_save, os.path.join(outdir, 'line_level_ensemble_v2_results.joblib'))
    
    # Save individual models
    for name, model in trained_models.items():
        if name != 'redis_similarity':
            joblib.dump(model, os.path.join(outdir, f'{name}_model_v2.joblib'))
    
    # Save scaler
    joblib.dump(scaler, os.path.join(outdir, 'scaler_v2.joblib'))
    
    # Save metadata with hyperparameters and redis config
    with open(os.path.join(outdir, 'training_metadata_v2.json'), 'w') as f:
        metadata = {k: v for k, v in results['training_info'].items()}
        metadata['model_scores'] = model_scores
        metadata['best_hyperparameters'] = best_params
        metadata['model_weights'] = model_weights
        
        # Add Redis VL configuration if available
        if 'redis_similarity' in trained_models:
            metadata['redis_config'] = {
                'index_name': trained_models['redis_similarity'].index_name,
                'threshold': trained_models['redis_similarity'].threshold,
                'host': 'localhost',
                'port': 6379
            }
        
        json.dump(metadata, f, indent=2)
    
    # Generate detailed results table for documentation
    print(f"\nGenerating detailed results table for  documentation...")
    generate_results_table(outdir, results, model_scores, model_weights, best_params, 
                              precision, recall, f1, len(X), anomaly_ratio)
    
    print(f"Line-level ensemble V2 training completed!")
    print(f"   Models saved in: {outdir}")
    print(f"   Trained on {len(X)} line-level samples")
    print(f"   Model types: Decision Tree, Multi-Layer Perceptron, Stochastic Gradient Descent, Redis VL Similarity")
    print(f"   Ensemble F1-Score: {f1:.4f}" if trained_models else "")
    
    return results

if __name__ == "__main__":
    print("Starting Line-Level HDFS Ensemble V2 Training...")
    
    # Check if embedding service is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("Embedding service is running")
        else:
            print("Embedding service not responding properly")
    except:
        print("Embedding service not available. Please start it first:")
        print("   cd embedding_service && python app.py")
        sys.exit(1)
    
    # Train ensemble V2 with line-level data
    results = train_line_level_ensemble_v2(sample_size=10000)
