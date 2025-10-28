#!/usr/bin/env python3
"""
Redis VL Entries Viewer for HDFS Anomaly Detection

This script replaces qdrant_entries.py to work with Redis VL vector database.
"""

import redis
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
import json
import sys
import os

# Redis connection configuration
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))
REDIS_DB = int(os.environ.get('REDIS_DB', '0'))

# Index name for HDFS log embeddings
INDEX_NAME = "logs_embeddings"

print(f"HDFS REDIS VL ENTRIES VIEWER")
print("=" * 50)

try:
    # Connect to Redis
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    
    # Test connection
    redis_client.ping()
    print(f"✅ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    
    # Check if index exists
    try:
        # Try to get index info
        index_info = redis_client.execute_command('FT.INFO', INDEX_NAME)
        print(f"Index: {INDEX_NAME}")
        
        # Parse index info to get document count
        info_dict = {}
        for i in range(0, len(index_info), 2):
            info_dict[index_info[i]] = index_info[i + 1]
        
        total_docs = info_dict.get('num_docs', 0)
        print(f"Total Documents: {total_docs}")
        
        if 'attributes' in info_dict:
            attributes = info_dict['attributes']
            for attr in attributes:
                if attr[1] == 'VECTOR':
                    vector_info = dict(zip(attr[::2], attr[1::2]))
                    print(f"📐 Vector Dimensions: {vector_info.get('DIM', 'Unknown')}")
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

    # Get recent entries
    print("Recent 20 HDFS log entries:")
    print("-" * 40)
    
    # Search for all documents (using wildcard)
    try:
        # Use FT.SEARCH to get recent documents
        result = redis_client.execute_command(
            'FT.SEARCH', INDEX_NAME, '*', 
            'LIMIT', '0', '20',
            'RETURN', '6', 'text', 'label', 'timestamp', 'block_id', 'score', 'embedding_id'
        )
        
        # Parse results
        if len(result) > 1:
            total_results = result[0]
            documents = result[1:]
            
            normal_count = 0
            anomaly_count = 0
            
            for i in range(0, len(documents), 2):
                doc_id = documents[i]
                doc_fields = documents[i + 1]
                
                # Parse document fields
                fields_dict = {}
                for j in range(0, len(doc_fields), 2):
                    fields_dict[doc_fields[j]] = doc_fields[j + 1]
                
                message = fields_dict.get('text', 'N/A')
                label = fields_dict.get('label', 'N/A')
                timestamp = fields_dict.get('timestamp', 'N/A')
                block_id = fields_dict.get('block_id', 'N/A')
                
                # Count labels
                if label == '0' or label == 'normal':
                    normal_count += 1
                    label_display = "✅ Normal"
                elif label == '1' or label == 'anomaly':
                    anomaly_count += 1
                    label_display = "🚨 Anomaly"
                else:
                    label_display = f"❓ {label}"
                
                print(f"{(i//2)+1:2d}. ID: {doc_id}")
                print(f"    Label: {label_display}")
                print(f"    Block: {block_id}")
                print(f"    Time:  {timestamp}")
                print(f"    Text:  {message[:80]}{'...' if len(message) > 80 else ''}")
                print()
            
            print("=" * 50)
            print(f"📊 SUMMARY:")
            print(f"   Total entries shown: {len(documents)//2}")
            print(f"   ✅ Normal entries: {normal_count}")
            print(f"   🚨 Anomaly entries: {anomaly_count}")
            print(f"   📈 Anomaly rate: {(anomaly_count/(normal_count+anomaly_count)*100):.1f}%" if (normal_count+anomaly_count) > 0 else "   📈 Anomaly rate: 0.0%")
        else:
            print("No documents found in the index.")
            
    except redis.ResponseError as e:
        print(f"Error searching index: {e}")
        sys.exit(1)

except redis.ConnectionError:
    print(f"❌ Failed to connect to Redis at {REDIS_HOST}:{REDIS_PORT}")
    print("Make sure Redis Stack is running with vector search capabilities")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)

print()
print("🔍 To explore more entries or perform vector searches,")
print("   use the Redis CLI or RedisInsight UI at http://localhost:8001")