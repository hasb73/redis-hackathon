#!/usr/bin/env python3
"""
Qdrant Embedding Analyzer
Analyze embeddings stored in Qdrant collection with similarity search and clustering insights
"""
import sys
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from typing import List, Dict, Tuple
import json

# Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION = "logs_embeddings"

def analyze_embedding_dimensions(qdrant_client: QdrantClient, sample_size: int = 100):
    """Analyze the dimensions and statistics of embeddings"""
    try:
        print(f"Analyzing embedding dimensions (sample size: {sample_size})...")
        
        result = qdrant_client.scroll(
            collection_name=COLLECTION,
            limit=sample_size,
            with_payload=True,
            with_vectors=True
        )
        
        points, _ = result
        
        if not points:
            print("No points found in collection")
            return
        
        vectors = []
        for point in points:
            if point.vector:
                vectors.append(point.vector)
        
        if not vectors:
            print("No vectors found in sample")
            return
        
        vectors_array = np.array(vectors)
        
        print(f"Sample size: {len(vectors)} vectors")
        print(f"Vector dimension: {vectors_array.shape[1]}")
        print(f"Vector statistics:")
        print(f"  Mean magnitude: {np.mean(np.linalg.norm(vectors_array, axis=1)):.4f}")
        print(f"  Std magnitude: {np.std(np.linalg.norm(vectors_array, axis=1)):.4f}")
        print(f"  Min value: {np.min(vectors_array):.4f}")
        print(f"  Max value: {np.max(vectors_array):.4f}")
        print(f"  Mean value: {np.mean(vectors_array):.4f}")
        print("")
        
        return vectors_array
        
    except Exception as e:
        print(f"Error analyzing embeddings: {e}")
        return None

def find_similar_entries(qdrant_client: QdrantClient, query_text: str, top_k: int = 5):
    """Find entries similar to given text using embedding similarity"""
    try:
        # First, find the query text in the collection
        result = qdrant_client.scroll(
            collection_name=COLLECTION,
            limit=1000,
            with_payload=True,
            with_vectors=True
        )
        
        points, _ = result
        query_vector = None
        
        # Find the embedding for the query text
        for point in points:
            if point.payload and "text" in point.payload:
                if query_text.lower() in point.payload["text"].lower():
                    query_vector = point.vector
                    print(f"Found query text in entry ID: {point.id}")
                    print(f"Query text: {point.payload['text'][:100]}...")
                    break
        
        if query_vector is None:
            print(f"Query text '{query_text}' not found in collection")
            return
        
        # Search for similar entries
        search_result = qdrant_client.search(
            collection_name=COLLECTION,
            query_vector=query_vector,
            limit=top_k + 1,  # +1 because the query itself will be included
            with_payload=True
        )
        
        print(f"Top {top_k} similar entries:")
        print("")
        
        for i, hit in enumerate(search_result[1:], 1):  # Skip the first one (query itself)
            print(f"Rank {i} (Score: {hit.score:.4f}):")
            print(f"  ID: {hit.id}")
            if hit.payload:
                for key, value in hit.payload.items():
                    if key == "text" and len(str(value)) > 100:
                        print(f"    {key}: {str(value)[:100]}...")
                    else:
                        print(f"    {key}: {value}")
            print("")
            
    except Exception as e:
        print(f"Error finding similar entries: {e}")

def compare_embeddings(qdrant_client: QdrantClient, id1: int, id2: int):
    """Compare two embeddings by their IDs"""
    try:
        points = qdrant_client.retrieve(
            collection_name=COLLECTION,
            ids=[id1, id2],
            with_payload=True,
            with_vectors=True
        )
        
        if len(points) != 2:
            print(f"Could not find both points. Found {len(points)} points.")
            return
        
        point1, point2 = points[0], points[1]
        
        if not point1.vector or not point2.vector:
            print("One or both points missing vector data")
            return
        
        # Calculate similarity
        vec1 = np.array(point1.vector)
        vec2 = np.array(point2.vector)
        
        # Cosine similarity
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        # Euclidean distance
        euclidean_dist = np.linalg.norm(vec1 - vec2)
        
        print("Embedding Comparison:")
        print(f"Point 1 ID: {point1.id}")
        if point1.payload and "text" in point1.payload:
            print(f"  Text: {point1.payload['text'][:100]}...")
        
        print(f"Point 2 ID: {point2.id}")
        if point2.payload and "text" in point2.payload:
            print(f"  Text: {point2.payload['text'][:100]}...")
        
        print(f"Cosine Similarity: {cosine_sim:.4f}")
        print(f"Euclidean Distance: {euclidean_dist:.4f}")
        print("")
        
    except Exception as e:
        print(f"Error comparing embeddings: {e}")

def find_outliers(qdrant_client: QdrantClient, sample_size: int = 500):
    """Find potential outlier embeddings based on distance from centroid"""
    try:
        print(f"Finding outliers in sample of {sample_size} embeddings...")
        
        result = qdrant_client.scroll(
            collection_name=COLLECTION,
            limit=sample_size,
            with_payload=True,
            with_vectors=True
        )
        
        points, _ = result
        
        if len(points) < 10:
            print("Need at least 10 points to find outliers")
            return
        
        vectors = []
        point_data = []
        
        for point in points:
            if point.vector:
                vectors.append(point.vector)
                point_data.append({
                    'id': point.id,
                    'payload': point.payload,
                    'vector': point.vector
                })
        
        vectors_array = np.array(vectors)
        
        # Calculate centroid
        centroid = np.mean(vectors_array, axis=0)
        
        # Calculate distances from centroid
        distances = []
        for i, vector in enumerate(vectors):
            dist = np.linalg.norm(np.array(vector) - centroid)
            distances.append((dist, i))
        
        # Sort by distance (descending)
        distances.sort(reverse=True)
        
        # Show top outliers
        print("Top 5 potential outliers (furthest from centroid):")
        print("")
        
        for i, (dist, idx) in enumerate(distances[:5]):
            point_info = point_data[idx]
            print(f"Outlier {i+1} (Distance: {dist:.4f}):")
            print(f"  ID: {point_info['id']}")
            if point_info['payload'] and "text" in point_info['payload']:
                print(f"  Text: {point_info['payload']['text'][:100]}...")
            if point_info['payload']:
                print(f"  Log Level: {point_info['payload'].get('log_level', 'unknown')}")
            print("")
        
    except Exception as e:
        print(f"Error finding outliers: {e}")

def main():
    """Main function with command line interface"""
    print("Qdrant Embedding Analyzer")
    print("========================")
    print("")
    
    # Initialize Qdrant client
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        print(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        print("")
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")
        return
    
    # Check collection exists
    try:
        info = qdrant_client.get_collection(COLLECTION)
        print(f"Collection: {COLLECTION}")
        print(f"Points count: {info.points_count}")
        print("")
    except Exception as e:
        print(f"Collection error: {e}")
        return
    
    # Interactive menu
    while True:
        print("Options:")
        print("1. Analyze embedding dimensions and statistics")
        print("2. Find similar entries to a text query")
        print("3. Compare two embeddings by ID")
        print("4. Find potential outlier embeddings")
        print("5. Exit")
        print("")
        
        choice = input("Enter your choice (1-5): ").strip()
        print("")
        
        if choice == "1":
            try:
                sample = input("Enter sample size (default 100): ").strip()
                sample_size = int(sample) if sample else 100
                analyze_embedding_dimensions(qdrant_client, sample_size)
            except ValueError:
                analyze_embedding_dimensions(qdrant_client)
                
        elif choice == "2":
            query = input("Enter text to search for: ").strip()
            if query:
                try:
                    top_k = input("Enter number of similar entries to find (default 5): ").strip()
                    top_k = int(top_k) if top_k else 5
                    find_similar_entries(qdrant_client, query, top_k)
                except ValueError:
                    find_similar_entries(qdrant_client, query)
                    
        elif choice == "3":
            try:
                id1 = int(input("Enter first point ID: "))
                id2 = int(input("Enter second point ID: "))
                compare_embeddings(qdrant_client, id1, id2)
            except ValueError:
                print("Invalid ID entered.")
                
        elif choice == "4":
            try:
                sample = input("Enter sample size for outlier detection (default 500): ").strip()
                sample_size = int(sample) if sample else 500
                find_outliers(qdrant_client, sample_size)
            except ValueError:
                find_outliers(qdrant_client)
                
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
        
        print("")

if __name__ == "__main__":
    main()
