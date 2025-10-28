#test embedding service

curl -X POST http://localhost:8000/embed -H "Content-Type: application/json" -d '{"texts": ["Test HDFS log message"]}'

#kill 
lsof -ti:8003 | xargs kill -9

#scoring service

#start scoring service
nohup ENABLE_KAFKA_CONSUMER=true scoring_service/python3 app.py

nohup python3 app.py > scoring_service.log 2>&1 &

curl -s -X GET "http://localhost:8003/stats" | python3 -m json.tool

curl -s -X GET "http://localhost:8003/anomalies" | python3 -c "import json, sys; data=json.load(sys.stdin); [print(f\"🚨 {a['timestamp']}: {a['text'][:60]}... (score: {a['anomaly_score']}, votes: {a['model_votes']})\") for a in data['anomalies']]"

          :
#caching 
curl -X GET http://localhost:8003/cache/stats


#spark submit 

spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.0 spark_job.py



#Test embedding creation

curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Unexpected error trying to delete block blk_8156514969688064600. BlockInfo not found in volumeMap"]}'

#Test embedding score

curl -X POST http://localhost:8003/score \
  -H "Content-Type: application/json" \
  -d '{"text": "2025-09-26 11:04:12,208 DEBUG org.apache.hadoop.hdfs.server.datanode.DataNode (DataXceiver for client DFSClient_NONMAPREDUCE_1480472633_1 at /172.31.36.192:59076 [Sending block BP-904282469-172.31.36.192-1758638658492:blk_1073742025_1201]): Error reading client status response. Will close connection."}'



#start in background
nohup python3 kafka_producer_hdfs.py > kafka_producer.log 2>&1 &


#list kafka messages
docker exec -it $(docker ps -q --filter "name=kafka") kafk
a-console-consumer.sh --bootstrap-server localhost:9092 --topic logs --from-beginning --max-messages 5


#Test enhanced scoring service

curl -X POST "http://localhost:8003/score_with_label?text=HDFS:%20Exception%20in%20createBlockOutputStream%20java.io.IOException:%20Bad%20connect%20ack%20with%20firstBadLink&actual_label=1" | python3 -m json.tool


curl -X GET "http://localhost:8003/accuracy_report" | python3 -m json.tool

curl -X GET "http://localhost:8003/performance_metrics" | python3 -m json.tool


grep '"label": 1' hdfs_line_level_data.jsonl | head -5 | jq -r '.text'



#DB operations 

sqlite3 anomaly_detection.db "SELECT COUNT(*) as total_records FROM anomaly_detections;"

sqlite3 anomaly_detection.db "SELECT predicted_label, COUNT(*) as count FROM anomaly_detections GROUP BY predicted_label;"

sqlite3 anomaly_detection.db "SELECT id, timestamp, text, predicted_label, anomaly_score, confidence, source FROM anomaly_detections LIMIT 5;"





Helper python

/usr/bin/python3 -c "
import joblib
import numpy as np

# Load the line level ensemble
ensemble_path = 'ensemble/models/line_level_ensemble/line_level_ensemble_results.joblib'
ensemble_data = joblib.load(ensemble_path)

print('=== CHECKING FOR WEIGHTED VOTING ===')
print('Available keys:', list(ensemble_data.keys()))

# Check if there are any weight-related keys
for key, value in ensemble_data.items():
    print(f'{key}: {type(value)}')
    if isinstance(value, dict):
        print(f'  Subkeys: {list(value.keys())}')

# Check if models have any weight attributes
models = ensemble_data.get('models', {})
print(f'\\n=== MODEL ATTRIBUTES ===')
for name, model in models.items():
    print(f'{name}: {type(model).__name__}')
    # Check common weight attributes
    weight_attrs = ['feature_importances_', 'coef_', 'weights_', 'model_weights']
    for attr in weight_attrs:
        if hasattr(model, attr):
            val = getattr(model, attr)
            print(f'  {attr}: {type(val)} shape={getattr(val, \"shape\", \"N/A\")}')
"


##### Ensemble Metrics ##########

/usr/bin/python3 -c "
import joblib
import json

# Load the line level ensemble
ensemble_path = 'ensemble/models/line_level_ensemble/line_level_ensemble_results.joblib'
ensemble_data = joblib.load(ensemble_path)

print('=== MODEL SCORES (potential weights) ===')
model_scores = ensemble_data.get('model_scores', {})
for model_name, scores in model_scores.items():
    print(f'{model_name}:')
    for metric, value in scores.items():
        print(f'  {metric}: {value:.4f}')
    print()

print('=== ENSEMBLE METRICS ===')
ensemble_metrics = ensemble_data.get('ensemble_metrics', {})
for metric, value in ensemble_metrics.items():
    print(f'{metric}: {value:.4f}')
"

/usr/bin/python3 -c "
import joblib
import os

print('=== LINE LEVEL ENSEMBLE MODELS ===')
line_level_path = 'ensemble/models/line_level_ensemble/line_level_ensemble_results.joblib'
line_level_data = joblib.load(line_level_path)
print(f'Models: {list(line_level_data[\"models\"].keys())}')
print(f'Training Info: {line_level_data[\"training_info\"]}')
print(f'Ensemble Metrics: {line_level_data[\"ensemble_metrics\"]}')
print()

print('=== REGULAR ENSEMBLE MODELS ===')
ensemble_path = 'ensemble/models/ensemble/ensemble_results.joblib'
ensemble_data = joblib.load(ensemble_path)
print(f'Models: {list(ensemble_data[\"models\"].keys())}')
print(f'Feature Dim: {ensemble_data[\"feature_dim\"]}')
print(f'Sample Size: {ensemble_data[\"sample_size\"]}')
print(f'Individual Scores: {ensemble_data[\"individual_scores\"]}')
print(f'Ensemble Score: {ensemble_data[\"ensemble_score\"]}')
"


###get weights and model scores 

/usr/bin/python3 -c "
import joblib
import numpy as np

# Load current model
ensemble_path = 'models/line_level_ensemble/line_level_ensemble_results.joblib'
ensemble_data = joblib.load(ensemble_path)

# Check if weights already exist
if 'model_weights' in ensemble_data:
    print('Model weights already exist:')
    for name, weight in ensemble_data['model_weights'].items():
        print(f'  {name}: {weight:.4f}')
    exit()

# Simulate adding weights to the existing model
model_scores = ensemble_data.get('model_scores', {})
print('Current model scores:')
for name, scores in model_scores.items():
    print(f'  {name}: F1={scores[\"f1\"]:.4f}')

# Calculate weights based on F1 scores
model_weights = {}
for name, scores in model_scores.items():
    f1_score = scores['f1']
    model_weights[name] = max(f1_score, 0.1)  # minimum weight of 0.1

# Normalize weights
total_weight = sum(model_weights.values())
model_weights = {k: v/total_weight for k, v in model_weights.items()}

print('\\nCalculated weights:')
for name, weight in model_weights.items():
    print(f'  {name}: {weight:.4f}')

# Add weights to ensemble and save
ensemble_data['model_weights'] = model_weights
joblib.dump(ensemble_data, ensemble_path)

print('\\n✅ Updated ensemble model with weighted voting!')
"