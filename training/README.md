# Training Module

## Overview
Machine learning model training and hyperparameter optimization for HDFS anomaly detection.

## Files
- `hdfs_line_level_loader_v2.py` - Labelling the HDFS logs at line level using heuristical strategies
- `train_line_level_ensemble_v1.py` - Original ensemble training script
- `train_line_level_ensemble_v2.py` - Advanced ensemble using vector similarity and Decision Tree, Multi layer perceptron and SGD

- - VectorSimilarityDetector:
- - - Stores a sample of normal embeddings in Qdrant vector DB
- - - Uses cosine similarity for anomaly detection of embeddings in the validation set
- - - Automatically tunes the cosine similarity threshold ( 0.3 to 0.9 ), which gives the best F1
- - - Example: For a new embedding, if avg similarity to top-k normals < threshold, label as anomaly.


- `models/` - Trained model artifacts and metadata
- `results/` - Training metrics and reports

## Features
- **Ensemble Learning**: Multiple algorithm training (Decision Tree, MLP, SGD)
- **Hyperparameter Tuning**: GridSearch optimization for each model
- **Model Persistence**: Storing the model in Joblib serialization for deployment
- **Performance Tracking**: Comprehensive metrics logging in: training/results/ directory 


## Dependencies
- Embedding service should be available and running on port 8000


## Usage
```bash
# Train ensemble models (run from project root)
# set the sample size ( defauilt is 200,000)
python3 training/train_line_level_ensemble_v2.py

# Models saved to: training/models/line_level_ensemble_v2/
# Results saved to: training/results/
```

## Model Output
- Trained classifiers with optimized hyperparameters
- Training metadata and performance metrics
- Ensemble weights based on F1 score for voting classifier


## RedisSimilarityDetector
RedisSimilarityDetector is a wrapper that:

Creates a Redis vector search index (via redisvl.SearchIndex schema) to store normal training embeddings.

Stores embeddings in Redis (each embedding as a hash with a vector field and a label tag).

Uses Redis vector search (preferred VectorQuery via redisvl) to find nearest neighbors for a query embedding.

Converts Redis search results into a similarity score (0..1), compares to a threshold and emits a binary anomaly prediction (1 = anomaly).

Provides a _tune_threshold routine that scans many candidate thresholds and picks the one with best F1 on validation data.

It’s intended to detect anomalies by measuring how similar each line-level embedding is to the set of normal embeddings

Key steps:

Build a schema dict for a vector field named "vector" using:

dims = X_train.shape[1]

distance_metric: "COSINE"

algorithm: "HNSW" with config (m, ef_construction, ef_runtime)

Create SearchIndex.from_dict(schema) and call .create(overwrite=True).

Filter training set to normal embeddings (y_train == 0) and prepare up to max_embeddings (min(10000, len(normal_embeddings))) — so the code protects from inserting huge numbers by default.

Convert each float32 embedding to bytes via embedding.astype(np.float32).tobytes() and store with label "normal" in Redis hashes using hset in a pipeline (batches of batch_size=100).

Vector storage: the code writes raw float32 bytes into the hash field vector. The search/index expects byte representation for the vector field.

Batched pipeline: reduces round trips and helps with throughput.

Only normal embeddings stored: anomalies are not stored (so search compares new embeddings to the normal manifold).

self.is_fitted = True is set after successful storage.
