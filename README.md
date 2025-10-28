# HDFS Anomaly Detection System using Redis VL as vector DB

### Abstract

Anomaly detection in system log streams is a challenging task due to the volume, throughput, variety, and lack of high-quality labelled data. This research proposes a novel ensemble-based anomaly detection system that addresses the challenges faced by traditional methods while also being usable in real-time scenarios, along with its ability to train on a small subset of the actual log data.  We use Redis Vector Library (RedisVL) for as the embedding store and use Redis for caching the predictions of the ensemble mode;l


## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                       HDFS Anomaly Detection Pipeline with RedisVL                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   HDFS Logs     │───▶│ Log Processor   │───▶│   Kafka Queue   │───▶│ Spark Streaming │
│  (575K records) │    │   (EMR/Local)   │    │   (:9092)       │    │   (Processing)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                                                              │
                                                                              ▼
                                                                    ┌─────────────────┐
                                                                    │ Embedding Svc   │
                                                                    │   (:8000)       │
                                                                    │ SBERT 384-dim   │
                                                                    └─────────────────┘
                                                                             │
                                                                             ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Redis Cache   │◀───│ Anomaly Service │◀───│  Redis VDB      │◀───│  Vector Store   │
│    (:6379)      │    │   (:8003)       │    │                 │    │  (RedisVL)      │
└─────────────────┘    │   FastAPI       │    │ Cosine Search   │    └─────────────────┘
                       └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    
                       │ Ensemble Models │    
                       │ • Decision Tree │    
                       │ • MLP Neural    │    
                       │ • SGD Classifier│    
                       │ • redis Vector  │    
                       └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  SQLite DB      │◀───│  Prediction     │───▶│   Grafana       │
│ (Predictions)   │    │   Results       │    │   (:3000)       │
│ (Metrics)       │    │   & Metrics     │    │  Dashboards     │
└─────────────────┘    └─────────────────┘    └─────────────────┘

```

---

## Technology Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Stream Processing** | Apache Spark | 4.0.0 | Real-time data processing |
| **Message Queue** | Apache Kafka | 7.5.0 | Event streaming platform |
| **Vector Database** | redis VL | Latest | High-performance vector storage |
| **ML Framework** | Scikit-learn | 1.3+ | Ensemble learning models |
| **Embeddings** | Sentence-BERT | 384-dim | Text vectorization |
| **API Framework** | FastAPI | Latest | RESTful scoring service |
| **Caching** | Redis | 7.0 | Performance optimization |
| **Monitoring** | Grafana | Latest | System dashboards |
| **Orchestration** | Docker Compose | Latest | Service management |

---

## Project Structure

The codebase is organized into modular components for academic submission and production deployment:

```
redis-hackathon/
├── anomaly-detection-service/         # Core anomaly detection API service
│   ├── anomaly_detection_service.py   # Main REST API service
│   ├── anomaly_detection.db           # SQLite database for logging
│   ├── anomaly_detection_service.log  # Service logs
│   ├── restart_service.sh             # Service restart script
│   └── README.md                      # Service documentation
│
├── training/                          # Model training and data preparation
│   ├── train_ensemble_model.py # Advanced ensemble training
│   ├── hdfs_line_level_loader_v2.py   # Data loading utilities
│   ├── models/                        # Trained model artifacts
│   │   ├── line_level_ensemble_v1/    # Version 1 models
│   │   │   ├── agg_clustering_model.joblib
│   │   │   ├── dbscan_model.joblib
│   │   │   ├── dt_model.joblib
│   │   │   ├── mlp_model.joblib
│   │   │   ├── rf_model.joblib
│   │   │   ├── scaler.joblib
│   │   │   ├── sgd_model.joblib
│   │   │   ├── line_level_ensemble_v1_results.joblib
│   │   │   └── training_metadata.json
│   │   └── line_level_ensemble_v2/    # Version 2 models
│   │       ├── dt_model_v2.joblib
│   │       ├── mlp_model_v2.joblib
│   │       ├── scaler_v2.joblib
│   │       ├── sgd_model_v2.joblib
│   │       ├── line_level_ensemble_v2_results.joblib
│   │       └── training_metadata_v2.json
│   ├── results/                       # Training results and metrics
│   │   └── results_data_2025-09-14_16-19-02.csv
│   └── README.md                      # Training documentation
│
├── evaluation/                        # Performance evaluation and testing
│   ├── anomaly_evaluation.py          # Comprehensive evaluation suite
│   ├── hdfs_anomaly_injection_loader.py  # Stress testing utilities
│   ├── hdfs_anomaly_injection_loader.py # Anomaly injection for testing
│   ├── anomaly-injection-tests/       # Test datasets (JSONL format)
│   │   ├── hdfs_eval_test_5.jsonl     # 5% anomaly rate
│   │   ├── hdfs_eval_test_10.jsonl    # 10% anomaly rate
│   │   ├── hdfs_eval_test_15.jsonl    # 15% anomaly rate
│   │   ├── hdfs_eval_test_40.jsonl    # 40% anomaly rate
│   │   ├── hdfs_eval_test_60.jsonl    # 60% anomaly rate
│   │   ├── hdfs_eval_test_80.jsonl    # 80% anomaly rate
│   │   ├── hdfs_stress_test_balanced.jsonl
│   │   └── hdfs_stress_test_challenging.jsonl
│   ├── evaluation_results/            # Evaluation metrics and reports
│   │   ├── anomaly_evaluation_results_5pc.csv
│   │   ├── anomaly_evaluation_results_10pc.csv
│   │   ├── anomaly_evaluation_results_15pc.csv
│   │   ├── anomaly_evaluation_results_20pc.csv
│   │   ├── anomaly_evaluation_results_40pc.csv
│   │   ├── anomaly_evaluation_results_100pc.csv
│   │   └── results.md
│   └── README.md                      # Evaluation documentation
│
├── local-deployment/                  # Local development components
│   ├── spark_job.py                   # Local Spark job
│   └── README.md                      # Local deployment guide
│
├── cloud-deployment/                  # Production cloud components
│   ├── hdfs_production_log_processor.py # Production log processor
│   ├── spark_job.py                   # Cloud Spark job
│   ├── spark_to_redis.py             # Spark to redis pipeline
│   ├── commands.txt                   # Deployment commands
│   ├── EMR_PRODUCTION_DEPLOYMENT.md   # EMR deployment guide
│   ├── SPARK_README.md                # Spark configuration guide
│   └── README.md                      # Cloud deployment guide
│
├── helper-scripts/                    # Utility scripts
│   ├── redis_entries.py              # redis data viewer
│   ├── redis_embeddings.py           # Embedding analysis
│   ├── manage_redis.py               # redis management utilities
│   ├── view_redis.py                 # redis visualization
│   ├── analyze_embeddings.py          # Embedding analytics
│   ├── clear_redis.py                # Database cleanup
│   ├── deploy_emr_log_processor.sh    # EMR deployment script
│   ├── stop_scoring_service.sh        # Service stop script
│   ├── commands.sh                    # Helper commands
│   ├── scoring.log                    # Scoring service logs
│   └── README.md                      # Utilities documentation
│
├── grafana/                           # Monitoring and dashboards
│   ├── dashboards/                    # Grafana dashboard configs
│   │   ├── model-performance-dashboard.json
│   │   ├── system-overview-dashboard.json
│   │   ├── simple-anomaly-dashboard.json
│   │   ├── MODEL_PERFORMANCE_README.md
│   │   └── SYSTEM_OVERVIEW_README.md
│   ├── provisioning/                  # Grafana provisioning
│   │   ├── dashboards/
│   │   │   └── dashboards.yml
│   │   └── datasources/
│   │       └── sqlite.yml
│   ├── deploy_dashboards.py           # Dashboard deployment
│   ├── grafana.ini                    # Grafana configuration
│   ├── grafana_test_queries.sql       # Test queries
│   ├── start-grafana.sh               # Start script
│   ├── stop-grafana.sh                # Stop script
│   ├── test-grafana-setup.sh          # Setup testing
│   ├── verify-grafana-setup.sh        # Setup verification
│   ├── validate_grafana_datasource.sh # Datasource validation
│   ├── test_sqlite_datasource.py      # SQLite datasource test
│   ├── sqlite_datasource_test_report.json # Test report
│   ├── DASHBOARD_CLEANUP_SUMMARY.md   # Cleanup documentation
│   ├── DASHBOARD_UPDATE_SUMMARY.md    # Update documentation
│   ├── GRAFANA_ORGANIZATION_SUMMARY.md # Organization guide
│   ├── GRAFANA_SETUP.md               # Setup documentation
│   ├── SQLITE_DATASOURCE_CONFIG.md    # Datasource configuration
│   └── README.md                      # Monitoring documentation
│
├── embedding_service/                 # Sentence embedding microservice
│   ├── app.py                         # FastAPI embedding service
│   ├── requirements.txt               # Python dependencies
│   └── Dockerfile                     # Container configuration
│
├── HDFS_dataset/                      # Primary HDFS dataset
│   ├── HDFS.log                       # Raw HDFS log data (575K records)
│   ├── parser.py                      # Log parsing utilities
│   ├── parsed/                        # Structured log data
│   │   ├── HDFS.log_structured.csv    # Structured log entries
│   │   └── HDFS.log_templates.csv     # Log templates
│   └── labels/                        # Ground truth labels
│       └── anomaly_label.csv          # Anomaly classifications
│
├── HDFS_v1/                          # Legacy HDFS parsing utilities
│   └── parser.py                      # Original parser implementation
│
├── docker-compose.yml                 # Integrated service orchestration
├── requirements.txt                   # Python dependencies
└── README.md                          # This documentation
```

---

## Quick Start

### Prerequisites

1. **System Requirements**:
   - Python 3.9+
   - Docker & Docker Compose
   - 16GB+ RAM recommended

2. **Docker Services** (Integrated Setup):
```bash
# Start all services including monitoring
docker-compose up --build -d

# Service endpoints
# - Kafka: localhost:9092
# - redis: localhost:6333
# - Redis: localhost:6379
# - Grafana: http://localhost:3000 (admin/password)
```

3. **Install Dependencies**:
```bash
pip3 install -r requirements.txt
```

### Basic Usage

1. **Train Models**:
```bash
# Run from project root
python3 training/train_ensemble_model.py
```

2. **Generate Evaluation Data with varying anomaly ratios**:
```bash
# Run from project root
python3 evaluation/hdfs_anomaly_injection_loader.py
```

3. **Start Anomaly Detection Service**:
```bash
# Run from project root
python3 anomaly-detection-service/anomaly_detection_service.py > anomaly-detection-service/anomaly-detection-service.log
```

4. 

```bash
# Initialise spark job to store embeddings in RedisVL
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.0 spark_to_qdrant.py
```

5. **Run Evaluation Test with Anomaly injection**:
```bash
# Run from project root (Example running the evaluation on 40% anomaly rate)
python3 evaluation/anomaly_evaluation.py evaluation/anomaly-injection-tests/hdfs_eval_test_40.jsonl
```



---

## Core Components

### 1. Anomaly Detection Service
- **Purpose**: REST API service for real-time anomaly detection
- **Technology**: FastAPI with ensemble model integration
- **Features**: Caching, performance metrics, health monitoring
- **Endpoint**: http://localhost:8003

### 2. Training Module
- **Purpose**: Machine learning model training and optimization
- **Models**: Decision Tree, MLP, SGD, redis Similarity
- **Features**: Hyperparameter tuning, cross-validation, ensemble weighting
- **Output**: Trained models, performance metrics, documentation

### 3. Evaluation Module
- **Purpose**: Comprehensive system testing and validation
- **Features**: Stress testing, performance benchmarking, report generation
- **Test Data**: Multiple anomaly ratios (5%, 10%, 15%, 20%, 40%)
- **Metrics**: Precision, Recall, F1-Score, Processing Time

### 4. Monitoring System
- **Purpose**: Real-time system monitoring and alerting
- **Technology**: Grafana with SQLite datasource
- **Dashboards**: System overview, model performance, anomaly analysis
- **Access**: http://localhost:3000

---

## Key Features

### Machine Learning Pipeline
- **Ensemble Learning**: Weighted voting with 4 diverse algorithms
- **Vector Similarity**: redis-based nearest neighbor detection using HNSW index and cosine similarity
- **Feature Engineering**: Sentence-BERT embeddings (384 dimensions)
- **Real-time Processing**: Sub-second prediction latency

### System Architecture
- **Microservices**: Modular, containerized components
- **Stream Processing**: Kafka + Spark for real-time data flow
- **Caching**: Redis for performance optimization
- **Monitoring**: Comprehensive dashboards and alerting
---

## Contributions

### Novel Techniques
1. **Hybrid Ensemble**: Combination of classical ML and vector similarity using Redis VL
2. **Real-time Architecture**: Production-ready streaming pipeline
3. **Evaluation Framework**: Comprehensive testing methodology


### Value
- **Extensibility**: Modular architecture to deploy on any machine / cloud 
