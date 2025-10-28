# Anomaly Detection Service

## Overview
REST API service for real-time HDFS log anomaly detection using ensemble model voting.

## Files
- `anomaly_detection_service.py` - Main FastAPI service
- `anomaly_detection.db` - SQLite database for logging predictions
- `anomaly_detection_service.log` - Service operation logs
- `restart_service.sh` - Service restart utility

## Features
- **Ensemble Voting**: Combines 4 ML algorithms (Decision Tree, MLP, SGD, Qdrant)
- **REST API**: FastAPI endpoints for scoring and health checks
- **Caching**: Redis performance optimization
- **Database Logging**: SQLite persistence for predictions and metrics
- **Real-time Processing**: Sub-second prediction latency

## Configuration

### Environment Variables

```bash
export ENSEMBLE_MODEL_PATH="training/models/line_level_ensemble_v2/line_level_ensemble_v2_results.joblib"
export EMBEDDING_SERVICE_URL="http://localhost:8000"
export KAFKA_SERVERS="localhost:9092"
export KAFKA_TOPIC="logs"
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"
export QDRANT_COLLECTION="logs_embeddings"
export ANOMALY_DB_PATH="./anomaly-detection-serviceanomaly_detection.db"
```

## Usage

### Start the Service

**Run from project root:**

```bash
sh sh restart_anomaly_detection_service.sh  && tail -f anomaly_detection_engine.log
python3 anomaly-detection-service/anomaly_detection_service.py
```

Service will be available at `http://localhost:8003`

### API Endpoints

#### Core Endpoints

- `POST /score` - Score individual log entry
  ```json
  {
    "text": "HDFS log entry here"
  }
  ```

- `POST /score_with_label` - Score with known label for accuracy tracking
  ```bash
  curl -X POST "http://localhost:8003/score_with_label?text=LOG_TEXT&actual_label=1"
  ```

#### Monitoring Endpoints

- `GET /stats` - Service statistics
- `GET /anomalies` - Recent anomaly detections
- `GET /performance_metrics` - Comprehensive performance metrics
- `GET /health` - Health check and service status
- `GET /model_info` - Ensemble model information

#### Cache Management

- `POST /cache/clear` - Clear all cached results
- `DELETE /cache/{text_hash}` - Delete specific cache entry
- `GET /cache/stats` - Cache statistics

### Health Check

```bash
curl http://localhost:8003/health
```

## Dependencies

- FastAPI
- Redis (optional, for caching)
- Qdrant (optional, for similarity voting, if qdrant is not available, it will fallback to embedding service)
- SQLite (for persistence)
- Embedding service running on port 8000

## Model Loading

The service loads ensemble models from `training/models/`. Ensure the training module has generated the required model files:

- `line_level_ensemble_v2_results.joblib` - Main ensemble model
- Individual model files (dt, mlp, sgd)
- Scaler files

## Database Schema

### Anomaly Detections Table
- Individual prediction records
- Model votes and confidence scores
- Processing time metrics
- Source tracking

### Performance Metrics Table
- Aggregated performance statistics
- Confusion matrix data
- System performance metrics



Should return service status and dependency health.

## Performance Tuning

1. **Enable Redis** for caching frequently queried logs
2. **Configure Qdrant** for similarity-based voting
3. **Adjust ensemble weights** in model training
4. **Monitor processing times** via metrics endpoint

## Logging

Logs are written to:
- `./anomaly-detection-service/anomaly_detection_service.log'`
- Console output (configurable level)

Log levels can be adjusted in the service configuration.
