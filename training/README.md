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
