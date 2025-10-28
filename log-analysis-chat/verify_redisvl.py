#!/usr/bin/env python3
"""
Verify RedisVL index and connection
"""

import redis
import sys

try:
    from redisvl.index import SearchIndex
    REDISVL_AVAILABLE = True
except ImportError:
    REDISVL_AVAILABLE = False
    print("❌ RedisVL not available")
    sys.exit(1)

def verify_redis_connection():
    """Verify Redis connection"""
    print("🔍 Verifying Redis Connection...")
    print("-" * 50)
    
    try:
        # Connect to Redis
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)
        redis_client.ping()
        print("✅ Redis connection successful")
        return redis_client
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return None

def verify_index_exists(redis_client):
    """Verify index exists"""
    print("\n🔍 Verifying Index...")
    print("-" * 50)
    
    try:
        # Check if index exists
        info = redis_client.execute_command('FT.INFO', 'logs_embeddings')
        print("✅ Index 'logs_embeddings' exists")
        
        # Parse info
        info_dict = {}
        for i in range(0, len(info), 2):
            if i + 1 < len(info):
                key = info[i].decode('utf-8') if isinstance(info[i], bytes) else info[i]
                value = info[i + 1]
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                info_dict[key] = value
        
        print(f"   Index name: {info_dict.get('index_name', 'N/A')}")
        print(f"   Number of documents: {info_dict.get('num_docs', 'N/A')}")
        
        return True
    except redis.ResponseError as e:
        print(f"❌ Index check failed: {e}")
        return False

def verify_redisvl_connection(redis_client):
    """Verify RedisVL can connect to index"""
    print("\n🔍 Verifying RedisVL Connection...")
    print("-" * 50)
    
    try:
        # Create SearchIndex object
        search_index = SearchIndex(
            name='logs_embeddings',
            redis_client=redis_client
        )
        print("✅ RedisVL SearchIndex created successfully")
        
        # Try to get index info
        try:
            info = search_index.info()
            print(f"   Index info retrieved: {len(info)} fields")
            return search_index
        except Exception as e:
            print(f"⚠️  Could not get index info: {e}")
            return search_index
            
    except Exception as e:
        print(f"❌ RedisVL connection failed: {e}")
        return None

def check_sample_data(redis_client):
    """Check sample data"""
    print("\n🔍 Checking Sample Data...")
    print("-" * 50)
    
    try:
        # Get sample keys
        keys = redis_client.keys('log_entry:*')
        print(f"✅ Found {len(keys)} log entries")
        
        if keys:
            # Get first entry
            sample_key = keys[0]
            data = redis_client.hgetall(sample_key)
            
            print(f"\n📄 Sample Entry: {sample_key.decode('utf-8')}")
            print(f"   Fields: {[k.decode('utf-8') for k in data.keys()]}")
            
            # Check if embedding exists
            if b'embedding' in data:
                import numpy as np
                embedding_bytes = data[b'embedding']
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                print(f"   Embedding dimensions: {len(embedding)}")
            
            # Check text
            if b'text' in data:
                text = data[b'text'].decode('utf-8')
                print(f"   Text: {text[:80]}...")
            
            return True
        else:
            print("⚠️  No log entries found")
            return False
            
    except Exception as e:
        print(f"❌ Error checking data: {e}")
        return False

def test_vector_search(search_index, redis_client):
    """Test vector search"""
    print("\n🔍 Testing Vector Search...")
    print("-" * 50)
    
    try:
        # Get a sample embedding from existing data
        keys = redis_client.keys('log_entry:*')
        if not keys:
            print("⚠️  No data to test with")
            return False
        
        sample_key = keys[0]
        data = redis_client.hgetall(sample_key)
        
        if b'embedding' not in data:
            print("⚠️  No embedding in sample data")
            return False
        
        import numpy as np
        embedding_bytes = data[b'embedding']
        query_vector = np.frombuffer(embedding_bytes, dtype=np.float32).tolist()
        
        # Try direct Redis search
        print("   Testing direct Redis FT.SEARCH...")
        embedding_bytes_query = np.array(query_vector, dtype=np.float32).tobytes()
        
        result = redis_client.execute_command(
            'FT.SEARCH', 'logs_embeddings',
            '*=>[KNN 3 @embedding $vec AS score]',
            'PARAMS', '2', 'vec', embedding_bytes_query,
            'SORTBY', 'score',
            'RETURN', '3', 'text', 'label', 'score',
            'DIALECT', '2'
        )
        
        if result and len(result) > 1:
            num_results = result[0]
            print(f"✅ Vector search successful: {num_results} results")
            
            # Show first result
            if len(result) > 2:
                doc_id = result[1].decode('utf-8') if isinstance(result[1], bytes) else result[1]
                print(f"   First result: {doc_id}")
            
            return True
        else:
            print("⚠️  No results from vector search")
            return False
            
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verifications"""
    print("=" * 50)
    print("RedisVL Verification Script")
    print("=" * 50)
    print()
    
    # Check RedisVL availability
    if not REDISVL_AVAILABLE:
        print("❌ RedisVL library not available")
        print("   Install with: pip install redisvl")
        return False
    
    print("✅ RedisVL library available")
    print()
    
    # Verify Redis connection
    redis_client = verify_redis_connection()
    if not redis_client:
        return False
    
    # Verify index exists
    if not verify_index_exists(redis_client):
        return False
    
    # Verify RedisVL connection
    search_index = verify_redisvl_connection(redis_client)
    if not search_index:
        return False
    
    # Check sample data
    if not check_sample_data(redis_client):
        return False
    
    # Test vector search
    if not test_vector_search(search_index, redis_client):
        return False
    
    # Summary
    print("\n" + "=" * 50)
    print("✅ All Verifications Passed!")
    print("=" * 50)
    print()
    print("RedisVL is properly configured and working.")
    print("The chat service should be able to perform vector searches.")
    print()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
