#!/usr/bin/env python3
"""
Vector Search Helper for Redis VL
Performs semantic search on logs using embeddings
"""

import sys
import json
import redis
import numpy as np
import requests
from typing import List, Dict, Any

# Configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_INDEX = "logs_embeddings"
EMBEDDING_URL = "http://localhost:8000/embed"

def get_embedding(text: str) -> List[float]:
    """Get embedding from embedding service"""
    response = requests.post(
        EMBEDDING_URL,
        json={"texts": [text]},
        timeout=10
    )
    if response.status_code == 200:
        embeddings = response.json().get("embeddings", [])
        return embeddings[0] if embeddings else None
    return None

def vector_search(query: str, limit: int = 5) -> Dict[str, Any]:
    """Perform vector similarity search"""
    print(f"🔍 Searching for: '{query}'")
    print("-" * 60)
    
    # Get embedding
    print("📊 Generating embedding...")
    embedding = get_embedding(query)
    if not embedding:
        return {"error": "Failed to generate embedding"}
    
    print(f"✅ Got embedding: {len(embedding)} dimensions")
    
    # Connect to Redis
    print("🔗 Connecting to Redis...")
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
    redis_client.ping()
    print("✅ Connected")
    
    # Convert embedding to bytes
    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
    
    # Perform search
    print(f"🔎 Searching index '{REDIS_INDEX}'...")
    result = redis_client.execute_command(
        'FT.SEARCH', REDIS_INDEX,
        f'*=>[KNN {limit} @embedding $vec AS score]',
        'PARAMS', '2', 'vec', embedding_bytes,
        'SORTBY', 'score',
        'RETURN', '6', 'text', 'timestamp', 'label', 'normalized_text', 'log_level', 'score',
        'DIALECT', '2'
    )
    
    # Parse results
    results = []
    if result and len(result) > 1:
        num_results = result[0]
        print(f"✅ Found {num_results} results\n")
        
        for i in range(1, len(result), 2):
            if i + 1 < len(result):
                doc_id = result[i].decode('utf-8')
                fields = result[i + 1]
                
                # Parse fields
                log_data = {"id": doc_id}
                for j in range(0, len(fields), 2):
                    if j + 1 < len(fields):
                        key = fields[j].decode('utf-8')
                        value = fields[j + 1]
                        if isinstance(value, bytes):
                            try:
                                value = value.decode('utf-8')
                            except:
                                value = str(value)
                        log_data[key] = value
                
                # Calculate similarity
                distance = float(log_data.get('score', 1.0))
                log_data['similarity'] = round(1.0 - distance, 4)
                log_data['label'] = int(log_data.get('label', 0))
                
                results.append(log_data)
    
    return {
        "query": query,
        "num_results": len(results),
        "results": results
    }

def print_results(data: Dict[str, Any]):
    """Pretty print search results"""
    if "error" in data:
        print(f"❌ Error: {data['error']}")
        return
    
    print("=" * 60)
    print(f"Query: {data['query']}")
    print(f"Results: {data['num_results']}")
    print("=" * 60)
    
    for i, log in enumerate(data['results'], 1):
        label_icon = "🚨" if log['label'] == 1 else "✅"
        print(f"\n{i}. {label_icon} Similarity: {log['similarity']:.4f}")
        print(f"   ID: {log['id']}")
        print(f"   Text: {log.get('text', '')[:80]}...")
        print(f"   Timestamp: {log.get('timestamp', 'N/A')}")
        print(f"   Level: {log.get('log_level', 'N/A')}")
        print(f"   Label: {'Anomaly' if log['label'] == 1 else 'Normal'}")

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python vector_search_helper.py <query> [limit]")
        print("\nExamples:")
        print("  python vector_search_helper.py 'receiving block'")
        print("  python vector_search_helper.py 'DataNode error' 10")
        print("  python vector_search_helper.py 'PacketResponder terminating' 3")
        sys.exit(1)
    
    query = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    try:
        results = vector_search(query, limit)
        print_results(results)
        
        # Also output JSON for programmatic use
        print("\n" + "=" * 60)
        print("JSON Output:")
        print("=" * 60)
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
