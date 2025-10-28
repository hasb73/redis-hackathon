#!/usr/bin/env python3
"""
Direct test of vector search in Redis
"""

import redis
import requests
import numpy as np
import json

# Configuration
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
EMBEDDING_URL = 'http://localhost:8000/embed'
INDEX_NAME = 'logs_embeddings'

def get_embedding(text):
    """Get embedding from service"""
    response = requests.post(
        EMBEDDING_URL,
        json={"texts": [text]},
        timeout=10
    )
    if response.status_code == 200:
        embeddings = response.json().get("embeddings", [])
        return embeddings[0] if embeddings else None
    return None

def test_vector_search():
    """Test vector search"""
    print("🧪 Testing Vector Search in Redis VL")
    print("=" * 50)
    
    # Connect to Redis
    print("\n1. Connecting to Redis...")
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
    redis_client.ping()
    print("   ✅ Connected")
    
    # Check index
    print("\n2. Checking index...")
    info = redis_client.execute_command('FT.INFO', INDEX_NAME)
    num_docs = None
    for i in range(len(info)):
        if info[i] == b'num_docs':
            num_docs = info[i+1]
            break
    print(f"   ✅ Index exists with {num_docs} documents")
    
    # Get sample log text from Redis
    print("\n3. Getting sample log...")
    keys = redis_client.keys('log_entry:*')
    if not keys:
        print("   ❌ No log entries found!")
        return False
    
    sample_key = keys[0]
    sample_data = redis_client.hgetall(sample_key)
    sample_text = sample_data[b'text'].decode('utf-8')
    print(f"   ✅ Sample: {sample_text[:80]}...")
    
    # Test query
    query_text = "receiving block"
    print(f"\n4. Testing query: '{query_text}'")
    
    # Get embedding
    print("   Getting embedding...")
    query_embedding = get_embedding(query_text)
    if not query_embedding:
        print("   ❌ Failed to get embedding")
        return False
    print(f"   ✅ Got embedding: {len(query_embedding)} dimensions")
    
    # Convert to bytes
    embedding_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
    
    # Perform search
    print("   Performing vector search...")
    try:
        result = redis_client.execute_command(
            'FT.SEARCH', INDEX_NAME,
            '*=>[KNN 5 @embedding $vec AS score]',
            'PARAMS', '2', 'vec', embedding_bytes,
            'SORTBY', 'score',
            'RETURN', '4', 'text', 'timestamp', 'label', 'score',
            'DIALECT', '2'
        )
        
        print(f"   ✅ Search completed")
        print(f"\n5. Results:")
        print("   " + "-" * 46)
        
        if result and len(result) > 1:
            num_results = result[0]
            print(f"   Found {num_results} results\n")
            
            for i in range(1, min(len(result), 11), 2):
                if i + 1 < len(result):
                    doc_id = result[i].decode('utf-8')
                    fields = result[i + 1]
                    
                    # Parse fields
                    field_dict = {}
                    for j in range(0, len(fields), 2):
                        if j + 1 < len(fields):
                            key = fields[j].decode('utf-8')
                            value = fields[j + 1]
                            if isinstance(value, bytes):
                                try:
                                    value = value.decode('utf-8')
                                except:
                                    value = str(value)
                            field_dict[key] = value
                    
                    # Calculate similarity
                    distance = float(field_dict.get('score', 1.0))
                    similarity = 1.0 - distance
                    
                    result_num = (i + 1) // 2
                    print(f"   Result {result_num}:")
                    print(f"   Text: {field_dict.get('text', '')[:70]}...")
                    print(f"   Similarity: {similarity:.4f}")
                    print(f"   Label: {field_dict.get('label', 'N/A')}")
                    print()
            
            return True
        else:
            print("   ❌ No results returned")
            return False
            
    except Exception as e:
        print(f"   ❌ Search failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vector_search()
    print("\n" + "=" * 50)
    if success:
        print("✅ Vector search is working!")
    else:
        print("❌ Vector search failed")
    print("=" * 50)
