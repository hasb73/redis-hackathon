#!/usr/bin/env python3
"""
Production Spark Streaming Job for HDFS Log Processing
Processes HDFS log lines from Kafka, generates embeddings, and stores in Redis VL.
The scoring service handles anomaly detection and predictions.
"""
# Suppress urllib3 SSL warnings
import warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL 1.1.1+')

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, from_json
from pyspark.sql.types import StringType, StructType, StructField, IntegerType
import json, requests, hashlib, time
import redis
import numpy as np

# Configuration
KAFKA_SERVERS = "localhost:9092"
KAFKA_TOPICS = "logs"
EMBEDDING_SERVICE_URL = "http://localhost:8000/embed"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_INDEX_NAME = "logs_embeddings"
REDIS_KEY_PREFIX = "log_entry:"
DIM = 384

# Initialize Redis client
print("🔧 Initializing Redis connection...")
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)

# Test Redis connection
try:
    redis_client.ping()
    print(f" Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    print(f" Redis connection failed: {e}")
    exit(1)

# Check if search index exists, create if not
def setup_redis_index():
    """Setup Redis search index for vector storage"""
    try:
        # Check if index exists
        redis_client.execute_command("FT.INFO", REDIS_INDEX_NAME)
        print(f" Redis search index '{REDIS_INDEX_NAME}' already exists")
    except redis.ResponseError:
        # Create the index
        try:
            redis_client.execute_command(
                "FT.CREATE", REDIS_INDEX_NAME,
                "ON", "HASH",
                "PREFIX", "1", REDIS_KEY_PREFIX,
                "SCHEMA",
                "text", "TEXT",
                "embedding", "VECTOR", "FLAT", "6",
                "TYPE", "FLOAT32",
                "DIM", str(DIM),
                "DISTANCE_METRIC", "COSINE",
                "label", "NUMERIC",
                "timestamp", "TEXT"
            )
            print(f" Created Redis search index: {REDIS_INDEX_NAME}")
        except Exception as e:
            print(f" Failed to create Redis index: {e}")

setup_redis_index()

# Initialize Spark Session
print("🚀 Initializing Spark Session...")
spark = SparkSession.builder \
    .appName("HDFSLineLevelEmbeddingPipeline") \
    .config("spark.streaming.stopGracefullyOnShutdown", "true") \
    .config("spark.sql.streaming.checkpointLocation", "/tmp/spark-checkpoint-line-level") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Define schema for HDFS production log messages from hdfs_production_log_processor
hdfs_schema = StructType([
    StructField("text", StringType(), True),           # Processed log text
    StructField("original_text", StringType(), True),  # Original raw log line
    StructField("log_level", StringType(), True),      # INFO, WARN, ERROR, etc.
    StructField("source", StringType(), True),         # hdfs_datanode
    StructField("timestamp", StringType(), True),      # ISO timestamp
    StructField("node_type", StringType(), True),      # datanode
    StructField("label", IntegerType(), True)          # Anomaly label (nullable)
])

# Read from Kafka
print(f"📡 Connecting to Kafka: {KAFKA_SERVERS}")
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_SERVERS) \
    .option("subscribe", KAFKA_TOPICS) \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .load()

# Parse JSON messages
json_df = df.selectExpr("CAST(value AS STRING) as raw_json")

# Add debug function to inspect raw JSON
def debug_json_parsing():
    print("🔍 Sample JSON messages from Kafka:")
    sample_df = json_df.limit(5)
    for row in sample_df.collect():
        print(f"   Raw JSON: {row.raw_json}")

# Extract structured data from JSON - Updated to match hdfs_production_log_processor format
parsed_df = json_df.withColumn(
    "parsed_data", 
    from_json(col("raw_json"), hdfs_schema)
).select(
    col("parsed_data.text").alias("message"),                    # Use 'text' field as message
    col("parsed_data.original_text").alias("original_text"),    # Keep original for reference
    col("parsed_data.log_level").alias("log_level"),            # Log severity level
    col("parsed_data.source").alias("source"),                  # Source identifier
    col("parsed_data.timestamp").alias("timestamp"),            # Processing timestamp
    col("parsed_data.node_type").alias("node_type"),            # Node type
    col("parsed_data.label").alias("label")                     # Anomaly label (nullable)
).filter(col("message").isNotNull())

def foreach_batch_hdfs(df, epoch_id):
    """Process each batch of HDFS production log messages"""
    print(f"\n Processing production log batch {epoch_id}...")
    
    # Debug: Show schema and sample data
    print(f"   📊 DataFrame schema: {df.schema}")
    
    # Collect batch data
    rows = df.collect()
    if not rows:
        print("   No data in batch")
        return
    
    # Debug: Show first few rows    
    print(f"   Batch size: {len(rows)} messages")
    
    # Extract messages and metadata - Production mode: no labels needed
    messages = []
    metadata = []
    
    for row in rows:
        messages.append(row['message'])
        metadata.append({
            'timestamp': row['timestamp'],
            'original_text': row['original_text'],
            'log_level': row['log_level'],
            'source': row['source'],
            'node_type': row['node_type']
        })
    
    # Generate embeddings
    try:
        print("    Generating embeddings...")
        resp = requests.post(
            EMBEDDING_SERVICE_URL, 
            json={"texts": messages}, 
            timeout=30
        )
        
        if resp.status_code != 200:
            print(f"    Embedding service error: {resp.status_code}")
            return
            
        embs = resp.json().get("embeddings", [])
        print(f"   Generated {len(embs)} embeddings")
        
    except Exception as e:
        print(f"    Embedding call failed: {e}")
        return
    
    # Prepare data for Redis VL - Production mode: just store embeddings
    redis_entries = []
    
    for i, (embedding, meta) in enumerate(zip(embs, metadata)):
        # Create unique ID using hash of message + timestamp
        entry_id = hashlib.md5(f"{messages[i]}_{meta.get('timestamp', str(time.time()))}".encode()).hexdigest()
        redis_key = f"{REDIS_KEY_PREFIX}{entry_id}"
        
        # Convert embedding to bytes for Redis storage
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
        
        # Handle None values by converting to empty strings or defaults
        def safe_str(value, default=""):
            return str(value) if value is not None else default
        
        redis_entry = {
            "key": redis_key,
            "data": {
                "text": safe_str(messages[i]),
                "embedding": embedding_bytes,
                "original_text": safe_str(meta.get('original_text')),
                "timestamp": safe_str(meta.get('timestamp'), str(int(time.time()))),
                "log_level": safe_str(meta.get('log_level'), "INFO"),
                "source": safe_str(meta.get('source'), "hdfs"),
                "node_type": safe_str(meta.get('node_type'), "datanode"),
                "label": 0  # Default to 0, scoring service will update if anomaly detected
            }
        }
        redis_entries.append(redis_entry)
    
    # Insert into Redis using pipeline for better performance
    try:
        print(f"    Inserting {len(redis_entries)} entries to Redis VL...")
        
        pipe = redis_client.pipeline()
        for entry in redis_entries:
            pipe.hset(entry["key"], mapping=entry["data"])
        pipe.execute()
        
        print(f"    Batch {epoch_id} processed: {len(redis_entries)} log entries stored in Redis VL")
        
    except Exception as e:
        print(f"    Redis VL insertion failed: {e}")

# Start streaming
print(" Starting HDFS line-level log processing stream...")
query = parsed_df.writeStream \
    .foreachBatch(foreach_batch_hdfs) \
    .outputMode("append") \
    .trigger(processingTime='10 seconds') \
    .start()

print(" Streaming started! Processing HDFS line-level logs...")
print("   - Reading from Kafka topic: logs")
print("   - Processing individual log lines with line-level anomaly labels")
print("   - Generating embeddings via embedding service")
print(f"   - Storing vectors in Redis VL index: {REDIS_INDEX_NAME}")
print("   - Processing every 10 seconds")
print("\nPress Ctrl+C to stop...")

try:
    query.awaitTermination()
except KeyboardInterrupt:
    print("\n  Stopping stream...")
    query.stop()
    spark.stop()
    print(" Stream stopped gracefully")
