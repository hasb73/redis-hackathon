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
import json, requests, hashlib, time, re
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

def normalize_log_message(message):
    """
    Strip dynamic elements from log messages to create a normalized template.
    This allows us to detect duplicate log patterns even with different IPs, ports, etc.
    
    Replaces:
    - IP addresses (IPv4) with <IP>
    - Ports with <PORT>
    - Block IDs with <BLOCK_ID>
    - File sizes with <SIZE>
    - Timestamps with <TIMESTAMP>
    - File paths with <PATH>
    """
    if not message:
        return message
    
    normalized = message
    
    # Replace IP addresses (IPv4 format: xxx.xxx.xxx.xxx)
    normalized = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '<IP>', normalized)
    
    # Replace ports (numbers after colons in network addresses)
    normalized = re.sub(r':<PORT>:\d+', ':<PORT>', normalized)  # Handle already replaced IPs
    normalized = re.sub(r':(\d{4,5})\b', ':<PORT>', normalized)
    
    # Replace block IDs (blk_-1234567890123456789 format)
    normalized = re.sub(r'blk_-?\d+', '<BLOCK_ID>', normalized)
    
    # Replace file sizes (numbers followed by size indicators or standalone large numbers)
    normalized = re.sub(r'\bsize\s+\d+', 'size <SIZE>', normalized)
    normalized = re.sub(r'\bof\s+size\s+\d+', 'of size <SIZE>', normalized)
    
    # Replace timestamps at the beginning (YYMMDD HHMMSS format)
    normalized = re.sub(r'^\d{6}\s+\d{6}', '<TIMESTAMP>', normalized)
    
    # Replace file paths (starting with / and containing slashes)
    normalized = re.sub(r'/[\w/\.\-]+\.(jar|split|xml|txt)', '<PATH>', normalized)
    normalized = re.sub(r'/mnt/[\w/\.\-]+', '<PATH>', normalized)
    
    # Replace job IDs
    normalized = re.sub(r'job_\d+_\d+', '<JOB_ID>', normalized)
    
    return normalized

def check_log_exists_in_redis(normalized_message):
    """
    Check if a normalized log message already exists in Redis.
    Returns True if exists, False otherwise.
    """
    try:
        # Create a hash of the normalized message to use as a lookup key
        message_hash = hashlib.md5(normalized_message.encode()).hexdigest()
        lookup_key = f"normalized:{message_hash}"
        
        # Check if this normalized message exists
        exists = redis_client.exists(lookup_key)
        return bool(exists)
    except Exception as e:
        print(f"    Error checking Redis for existing log: {e}")
        return False

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
    # Also normalize messages and filter out duplicates
    messages = []
    normalized_messages = []
    metadata = []
    original_messages = []
    
    skipped_count = 0
    
    for row in rows:
        original_msg = row['message']
        normalized_msg = normalize_log_message(original_msg)
        
        # Check if this normalized message already exists in Redis
        if check_log_exists_in_redis(normalized_msg):
            skipped_count += 1
            continue
        
        messages.append(original_msg)
        normalized_messages.append(normalized_msg)
        original_messages.append(original_msg)
        metadata.append({
            'timestamp': row['timestamp'],
            'original_text': row['original_text'],
            'log_level': row['log_level'],
            'source': row['source'],
            'node_type': row['node_type']
        })
    
    if skipped_count > 0:
        print(f"   ⏭️  Skipped {skipped_count} duplicate log patterns already in Redis")
    
    if not messages:
        print("   ℹ️  All logs in batch already exist in Redis, skipping embedding generation")
        return
    
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
    normalized_keys = []
    
    for i, (embedding, meta, normalized_msg) in enumerate(zip(embs, metadata, normalized_messages)):
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
                "label": 0,  # Default to 0, scoring service will update if anomaly detected
                "normalized_text": safe_str(normalized_msg)  # Store normalized version for reference
            }
        }
        redis_entries.append(redis_entry)
        
        # Track normalized message for deduplication
        message_hash = hashlib.md5(normalized_msg.encode()).hexdigest()
        normalized_keys.append(f"normalized:{message_hash}")
    
    # Insert into Redis using pipeline for better performance
    try:
        print(f"    Inserting {len(redis_entries)} entries to Redis VL...")
        
        pipe = redis_client.pipeline()
        for entry, norm_key in zip(redis_entries, normalized_keys):
            # Store the actual log entry with embedding
            pipe.hset(entry["key"], mapping=entry["data"])
            # Store normalized message marker for deduplication (expires in 30 days)
            pipe.setex(norm_key, 2592000, entry["key"])
        pipe.execute()
        
        print(f"    Batch {epoch_id} processed: {len(redis_entries)} new log entries stored in Redis VL")
        
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
