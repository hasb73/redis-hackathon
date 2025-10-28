#!/usr/bin/env python3
"""
Redis VL Embeddings Viewer for HDFS Anomaly Detection

This script replaces qdrant_embeddings.py to work with Redis VL vector database.
"""

import redis
import numpy as np
import json
import sys
import os

# Redis connection configuration
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))
REDIS_DB = int(os.environ.get('REDIS_DB', '0'))

# Index name for HDFS log embeddings
INDEX_NAME = "logs_embeddings"

print('HDFS REDIS VL EMBEDDINGS VIEWER')
print('=' * 50)

try:
    # Connect to Redis
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
    
    # Test connection
    redis_client.ping()
    print(f"✅ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    
    # Check if index exists and get info
    try:
        # Get index info
        index_info = redis_client.execute_command('FT.INFO', INDEX_NAME)
        info_dict = {}
        for i in range(0, len(index_info), 2):
            key = index_info[i].decode() if isinstance(index_info[i], bytes) else index_info[i]
            value = index_info[i + 1]
            if isinstance(value, bytes):
                value = value.decode()
            info_dict[key] = value
        
        total_docs = info_dict.get('num_docs', 0)
        print(f'📊 Index: {INDEX_NAME}')
        print(f'📈 Total Documents: {total_docs}')
        
        # Get vector dimension info
        vector_dim = None
        distance_metric = None
        if 'attributes' in info_dict:
            attributes = info_dict['attributes']
            for attr in attributes:
                if len(attr) > 1 and attr[1] == b'VECTOR':
                    # Parse vector attributes
                    for j in range(2, len(attr), 2):
                        if attr[j] == b'DIM':
                            vector_dim = int(attr[j + 1])
                        elif attr[j] == b'DISTANCE_METRIC':
                            distance_metric = attr[j + 1].decode()
        
        print(f'📐 Vector Dimensions: {vector_dim if vector_dim else "Unknown"}')
        print(f'🔍 Distance Metric: {distance_metric if distance_metric else "Unknown"}')
        print()
        
        if total_docs == 0:
            print('📭 Index is empty. Run the streaming pipeline to populate it.')
            sys.exit(0)
            
    except redis.ResponseError as e:
        if "Unknown index name" in str(e):
            print(f'Index "{INDEX_NAME}" not found!')
            print('Run the Spark streaming job to create and populate the index')
            sys.exit(1)
        else:
            raise e

    # Get sample embeddings
    print('📄 Sample HDFS Log Embeddings (with vectors):')
    print('-' * 40)
    
    # Search for documents with their vectors
    try:
        result = redis_client.execute_command(
            'FT.SEARCH', INDEX_NAME, '*', 
            'LIMIT', '0', '5',  # Get 5 samples
            'RETURN', '8', 'text', 'label', 'timestamp', 'block_id', 'score', 'embedding_id', 'vector', 'embedding'
        )
        
        if len(result) > 1:
            documents = result[1:]
            
            for i in range(0, len(documents), 2):
                doc_id = documents[i].decode() if isinstance(documents[i], bytes) else documents[i]
                doc_fields = documents[i + 1]
                
                # Parse document fields
                fields_dict = {}
                for j in range(0, len(doc_fields), 2):
                    key = doc_fields[j].decode() if isinstance(doc_fields[j], bytes) else doc_fields[j]
                    value = doc_fields[j + 1]
                    if isinstance(value, bytes):
                        try:
                            value = value.decode()
                        except UnicodeDecodeError:
                            # This might be binary vector data
                            pass
                    fields_dict[key] = value
                
                message = fields_dict.get('text', 'N/A')
                label = fields_dict.get('label', 'N/A')
                timestamp = fields_dict.get('timestamp', 'N/A')
                block_id = fields_dict.get('block_id', 'N/A')
                
                # Try to get vector data
                vector_data = fields_dict.get('vector') or fields_dict.get('embedding')
                vector_info = "No vector data found"
                
                if vector_data:
                    try:
                        # Try to parse as JSON first
                        if isinstance(vector_data, str):
                            vector_array = json.loads(vector_data)
                        elif isinstance(vector_data, bytes):
                            # Try to decode as float32 array
                            vector_array = np.frombuffer(vector_data, dtype=np.float32)
                        else:
                            vector_array = vector_data
                        
                        if isinstance(vector_array, (list, np.ndarray)):
                            vector_array = np.array(vector_array)
                            vector_info = f"Vector shape: {vector_array.shape}, Mean: {vector_array.mean():.4f}, Std: {vector_array.std():.4f}"
                            vector_preview = f"First 5 values: {vector_array[:5].tolist()}"
                        else:
                            vector_info = f"Vector data type: {type(vector_array)}"
                            vector_preview = ""
                    except Exception as ve:
                        vector_info = f"Error parsing vector: {ve}"
                        vector_preview = ""
                else:
                    vector_preview = ""
                
                label_display = "✅ Normal" if label in ['0', 'normal'] else "🚨 Anomaly" if label in ['1', 'anomaly'] else f"❓ {label}"
                
                print(f"{(i//2)+1}. Document ID: {doc_id}")
                print(f"   Label: {label_display}")
                print(f"   Block: {block_id}")
                print(f"   Time:  {timestamp}")
                print(f"   Text:  {message[:60]}{'...' if len(message) > 60 else ''}")
                print(f"   📊 {vector_info}")
                if vector_preview:
                    print(f"   🔢 {vector_preview}")
                print()
        else:
            print("No documents found in the index.")
            
    except redis.ResponseError as e:
        print(f"Error searching index: {e}")
        sys.exit(1)

    # Additional vector statistics
    print("=" * 50)
    print("📈 VECTOR ANALYSIS SUMMARY:")
    print(f"   Index contains {total_docs} log embeddings")
    print(f"   Each vector has {vector_dim if vector_dim else 'unknown'} dimensions")
    print(f"   Using {distance_metric if distance_metric else 'unknown'} distance metric")
    print()
    print("🔍 For advanced vector similarity searches, use:")
    print("   - Redis CLI with FT.SEARCH commands")
    print("   - RedisVL Python client for vector queries")
    print("   - RedisInsight UI at http://localhost:8001")

except redis.ConnectionError:
    print(f"❌ Failed to connect to Redis at {REDIS_HOST}:{REDIS_PORT}")
    print("Make sure Redis Stack is running with vector search capabilities")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)