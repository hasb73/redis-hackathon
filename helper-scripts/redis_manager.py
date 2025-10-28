#!/usr/bin/env python3
"""
Redis VL Management Script for HDFS Anomaly Detection

This script replaces manage_qdrant.py to work with Redis VL vector database.
Provides utilities for managing Redis VL indexes and vector data.
"""

import redis
import json
import sys
import os
import numpy as np
from typing import Dict, List, Optional

# Redis connection configuration
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))
REDIS_DB = int(os.environ.get('REDIS_DB', '0'))

# Index name for HDFS log embeddings
INDEX_NAME = "logs_embeddings"

class RedisVLManager:
    def __init__(self):
        """Initialize Redis VL manager"""
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST, 
                port=REDIS_PORT, 
                db=REDIS_DB, 
                decode_responses=False
            )
            self.redis_client.ping()
            print(f"✅ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except redis.ConnectionError:
            print(f"❌ Failed to connect to Redis at {REDIS_HOST}:{REDIS_PORT}")
            sys.exit(1)
    
    def list_indexes(self):
        """List all Redis search indexes"""
        try:
            indexes = self.redis_client.execute_command('FT._LIST')
            if indexes:
                print("📋 Available Redis Search Indexes:")
                for idx in indexes:
                    idx_name = idx.decode() if isinstance(idx, bytes) else idx
                    print(f"   - {idx_name}")
                    try:
                        info = self.redis_client.execute_command('FT.INFO', idx_name)
                        info_dict = {}
                        for i in range(0, len(info), 2):
                            key = info[i].decode() if isinstance(info[i], bytes) else info[i]
                            value = info[i + 1]
                            if isinstance(value, bytes):
                                value = value.decode()
                            info_dict[key] = value
                        print(f"     Documents: {info_dict.get('num_docs', 'Unknown')}")
                    except:
                        pass
                print()
            else:
                print("📭 No Redis search indexes found")
        except redis.ResponseError as e:
            print(f"Error listing indexes: {e}")
    
    def create_index(self, vector_dim: int = 384):
        """Create the HDFS logs embeddings index"""
        try:
            # Check if index already exists
            try:
                self.redis_client.execute_command('FT.INFO', INDEX_NAME)
                print(f"⚠️  Index '{INDEX_NAME}' already exists")
                return False
            except redis.ResponseError:
                pass  # Index doesn't exist, we can create it
            
            # Create index with vector and text fields
            index_definition = [
                'FT.CREATE', INDEX_NAME,
                'ON', 'HASH',
                'PREFIX', '1', f'{INDEX_NAME}:',
                'SCHEMA',
                'text', 'TEXT', 'SORTABLE',
                'label', 'TAG',
                'timestamp', 'TEXT',
                'block_id', 'TEXT',
                'score', 'NUMERIC',
                'embedding_id', 'TEXT',
                'vector', 'VECTOR', 'FLAT', '6', 'TYPE', 'FLOAT32', 'DIM', str(vector_dim), 'DISTANCE_METRIC', 'COSINE'
            ]
            
            self.redis_client.execute_command(*index_definition)
            print(f"✅ Created index '{INDEX_NAME}' with {vector_dim}-dimensional vectors")
            return True
            
        except redis.ResponseError as e:
            print(f"❌ Error creating index: {e}")
            return False
    
    def delete_index(self, confirm: bool = False):
        """Delete the HDFS logs embeddings index"""
        if not confirm:
            response = input(f"⚠️  Are you sure you want to delete index '{INDEX_NAME}'? (y/N): ")
            if response.lower() != 'y':
                print("❌ Operation cancelled")
                return False
        
        try:
            self.redis_client.execute_command('FT.DROPINDEX', INDEX_NAME, 'DD')
            print(f"✅ Deleted index '{INDEX_NAME}' and all associated documents")
            return True
        except redis.ResponseError as e:
            if "Unknown index name" in str(e):
                print(f"⚠️  Index '{INDEX_NAME}' doesn't exist")
            else:
                print(f"❌ Error deleting index: {e}")
            return False
    
    def get_index_info(self):
        """Get detailed information about the index"""
        try:
            info = self.redis_client.execute_command('FT.INFO', INDEX_NAME)
            info_dict = {}
            for i in range(0, len(info), 2):
                key = info[i].decode() if isinstance(info[i], bytes) else info[i]
                value = info[i + 1]
                if isinstance(value, bytes):
                    try:
                        value = value.decode()
                    except UnicodeDecodeError:
                        pass  # Keep as bytes for binary data
                info_dict[key] = value
            
            print(f"📊 Index Information: {INDEX_NAME}")
            print("-" * 40)
            print(f"Documents: {info_dict.get('num_docs', 'Unknown')}")
            print(f"Terms: {info_dict.get('num_terms', 'Unknown')}")
            print(f"Records: {info_dict.get('num_records', 'Unknown')}")
            print(f"Indexing: {info_dict.get('indexing', 'Unknown')}")
            
            # Parse attributes
            if 'attributes' in info_dict:
                print("\n📋 Schema Attributes:")
                attributes = info_dict['attributes']
                for attr in attributes:
                    if len(attr) > 1:
                        field_name = attr[0].decode() if isinstance(attr[0], bytes) else attr[0]
                        field_type = attr[1].decode() if isinstance(attr[1], bytes) else attr[1]
                        print(f"   - {field_name}: {field_type}")
                        
                        # Additional vector info
                        if field_type == 'VECTOR':
                            for j in range(2, len(attr), 2):
                                param_name = attr[j].decode() if isinstance(attr[j], bytes) else attr[j]
                                param_value = attr[j + 1]
                                if isinstance(param_value, bytes):
                                    param_value = param_value.decode()
                                print(f"     {param_name}: {param_value}")
            
            return info_dict
            
        except redis.ResponseError as e:
            if "Unknown index name" in str(e):
                print(f"❌ Index '{INDEX_NAME}' not found")
            else:
                print(f"❌ Error getting index info: {e}")
            return None
    
    def search_similar(self, query_vector: List[float], top_k: int = 5):
        """Search for similar vectors"""
        try:
            # Convert query vector to bytes
            vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()
            
            # Perform vector similarity search
            result = self.redis_client.execute_command(
                'FT.SEARCH', INDEX_NAME,
                f'*=>[KNN {top_k} @vector $vec AS score]',
                'PARAMS', '2', 'vec', vector_bytes,
                'SORTBY', 'score',
                'RETURN', '6', 'text', 'label', 'timestamp', 'block_id', 'score', 'embedding_id'
            )
            
            if len(result) > 1:
                total_results = result[0]
                documents = result[1:]
                
                print(f"🔍 Top {top_k} Similar Documents:")
                print("-" * 40)
                
                for i in range(0, len(documents), 2):
                    doc_id = documents[i].decode() if isinstance(documents[i], bytes) else documents[i]
                    doc_fields = documents[i + 1]
                    
                    # Parse document fields
                    fields_dict = {}
                    for j in range(0, len(doc_fields), 2):
                        key = doc_fields[j].decode() if isinstance(doc_fields[j], bytes) else doc_fields[j]
                        value = doc_fields[j + 1]
                        if isinstance(value, bytes):
                            value = value.decode()
                        fields_dict[key] = value
                    
                    similarity_score = float(fields_dict.get('score', 0))
                    message = fields_dict.get('text', 'N/A')
                    label = fields_dict.get('label', 'N/A')
                    
                    label_display = "✅ Normal" if label in ['0', 'normal'] else "🚨 Anomaly" if label in ['1', 'anomaly'] else f"❓ {label}"
                    
                    print(f"{(i//2)+1}. Similarity: {similarity_score:.4f}")
                    print(f"   ID: {doc_id}")
                    print(f"   Label: {label_display}")
                    print(f"   Text: {message[:80]}{'...' if len(message) > 80 else ''}")
                    print()
            else:
                print("No similar documents found")
                
        except redis.ResponseError as e:
            print(f"❌ Error performing similarity search: {e}")
    
    def clear_all_data(self, confirm: bool = False):
        """Clear all data from the index"""
        if not confirm:
            response = input(f"⚠️  Are you sure you want to clear all data from '{INDEX_NAME}'? (y/N): ")
            if response.lower() != 'y':
                print("❌ Operation cancelled")
                return False
        
        try:
            # Get all document keys
            keys = self.redis_client.keys(f'{INDEX_NAME}:*')
            if keys:
                deleted_count = self.redis_client.delete(*keys)
                print(f"✅ Deleted {deleted_count} documents from index")
            else:
                print("📭 No documents found to delete")
            return True
        except Exception as e:
            print(f"❌ Error clearing data: {e}")
            return False

def main():
    """Main function with command-line interface"""
    manager = RedisVLManager()
    
    if len(sys.argv) < 2:
        print("Redis VL Management Tool for HDFS Anomaly Detection")
        print("=" * 50)
        print("Usage: python redis_manager.py <command>")
        print()
        print("Commands:")
        print("  list         - List all indexes")
        print("  info         - Show index information")
        print("  create       - Create the HDFS embeddings index")
        print("  delete       - Delete the index")
        print("  clear        - Clear all data from index")
        print("  search       - Test similarity search (requires test vector)")
        print()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == 'list':
        manager.list_indexes()
    elif command == 'info':
        manager.get_index_info()
    elif command == 'create':
        manager.create_index()
    elif command == 'delete':
        manager.delete_index()
    elif command == 'clear':
        manager.clear_all_data()
    elif command == 'search':
        print("🔍 Testing similarity search with random vector...")
        # Generate a random test vector (384 dimensions for sentence transformers)
        test_vector = np.random.rand(384).tolist()
        manager.search_similar(test_vector, top_k=3)
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()