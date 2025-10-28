#!/usr/bin/env python3
"""
Qdrant Viewer Utility
View all entries and embeddings stored in the Qdrant collection
"""
import sys
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from typing import Optional
import json

# Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION = "logs_embeddings"

def get_collection_info(qdrant_client: QdrantClient):
    """Get basic information about the collection"""
    try:
        info = qdrant_client.get_collection(COLLECTION)
        print(f"Collection: {COLLECTION}")
        print(f"  Points count: {info.points_count}")
        print(f"  Vector size: {info.config.params.vectors.size}")
        print(f"  Distance metric: {info.config.params.vectors.distance}")
        print(f"  Status: {info.status}")
        print("")
        return info.points_count
    except Exception as e:
        print(f"Error getting collection info: {e}")
        return 0

def view_all_entries(qdrant_client: QdrantClient, show_vectors: bool = False, limit: Optional[int] = None):
    """View all entries in the collection"""
    try:
        print("Fetching all entries...")
        
        all_points = []
        next_page_offset = None
        page_size = 100
        total_fetched = 0
        
        while True:
            # Use scroll to get all points
            result = qdrant_client.scroll(
                collection_name=COLLECTION,
                limit=page_size,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=show_vectors
            )
            
            points, next_page_offset = result
            
            if not points:
                break
                
            all_points.extend(points)
            total_fetched += len(points)
            
            print(f"  Fetched {total_fetched} points...")
            
            if limit and total_fetched >= limit:
                all_points = all_points[:limit]
                break
                
            if next_page_offset is None:
                break
        
        print(f"Total entries retrieved: {len(all_points)}")
        print("")
        
        # Display entries
        for i, point in enumerate(all_points, 1):
            print(f"Entry {i}:")
            print(f"  ID: {point.id}")
            
            if point.payload:
                print("  Payload:")
                for key, value in point.payload.items():
                    if key == "text" and len(str(value)) > 100:
                        print(f"    {key}: {str(value)[:100]}...")
                    else:
                        print(f"    {key}: {value}")
            
            if show_vectors and point.vector:
                vector_preview = point.vector[:5] if len(point.vector) > 5 else point.vector
                print(f"  Vector (first 5 dims): {vector_preview}")
                print(f"  Vector dimension: {len(point.vector)}")
            
            print("")
            
    except Exception as e:
        print(f"Error viewing entries: {e}")

def search_entries(qdrant_client: QdrantClient, query_text: str, limit: int = 10):
    """Search for entries containing specific text"""
    try:
        print(f"Searching for entries containing: '{query_text}'")
        
        # Use scroll with payload filter
        result = qdrant_client.scroll(
            collection_name=COLLECTION,
            limit=1000,  # Get more points to search through
            with_payload=True,
            with_vectors=False
        )
        
        points, _ = result
        matching_points = []
        
        for point in points:
            if point.payload and "text" in point.payload:
                if query_text.lower() in point.payload["text"].lower():
                    matching_points.append(point)
                    if len(matching_points) >= limit:
                        break
        
        print(f"Found {len(matching_points)} matching entries:")
        print("")
        
        for i, point in enumerate(matching_points, 1):
            print(f"Match {i}:")
            print(f"  ID: {point.id}")
            if point.payload:
                for key, value in point.payload.items():
                    if key == "text" and len(str(value)) > 150:
                        print(f"    {key}: {str(value)[:150]}...")
                    else:
                        print(f"    {key}: {value}")
            print("")
            
    except Exception as e:
        print(f"Error searching entries: {e}")

def get_collection_stats(qdrant_client: QdrantClient):
    """Get detailed statistics about the collection"""
    try:
        print("Collection Statistics:")
        
        # Get all points to analyze
        result = qdrant_client.scroll(
            collection_name=COLLECTION,
            limit=10000,  # Adjust based on your collection size
            with_payload=True,
            with_vectors=False
        )
        
        points, _ = result
        
        if not points:
            print("  No points in collection")
            return
        
        # Analyze payload statistics
        log_levels = {}
        sources = {}
        node_types = {}
        
        for point in points:
            if point.payload:
                # Count log levels
                log_level = point.payload.get("log_level", "unknown")
                log_levels[log_level] = log_levels.get(log_level, 0) + 1
                
                # Count sources
                source = point.payload.get("source", "unknown")
                sources[source] = sources.get(source, 0) + 1
                
                # Count node types
                node_type = point.payload.get("node_type", "unknown")
                node_types[node_type] = node_types.get(node_type, 0) + 1
        
        print(f"  Total points analyzed: {len(points)}")
        print("")
        
        print("  Log Level Distribution:")
        for level, count in sorted(log_levels.items()):
            percentage = (count / len(points)) * 100
            print(f"    {level}: {count} ({percentage:.1f}%)")
        print("")
        
        print("  Source Distribution:")
        for source, count in sorted(sources.items()):
            percentage = (count / len(points)) * 100
            print(f"    {source}: {count} ({percentage:.1f}%)")
        print("")
        
        print("  Node Type Distribution:")
        for node_type, count in sorted(node_types.items()):
            percentage = (count / len(points)) * 100
            print(f"    {node_type}: {count} ({percentage:.1f}%)")
        print("")
        
    except Exception as e:
        print(f"Error getting collection stats: {e}")

def main():
    """Main function with command line interface"""
    print("Qdrant Viewer Utility")
    print("===================")
    print("")
    
    # Initialize Qdrant client
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        print(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        print("")
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")
        return
    
    # Get collection info
    points_count = get_collection_info(qdrant_client)
    
    if points_count == 0:
        print("Collection is empty or does not exist.")
        return
    
    # Interactive menu
    while True:
        print("Options:")
        print("1. View all entries (without vectors)")
        print("2. View all entries (with vectors)")
        print("3. View limited entries (specify count)")
        print("4. Search entries by text")
        print("5. Show collection statistics")
        print("6. Exit")
        print("")
        
        choice = input("Enter your choice (1-6): ").strip()
        print("")
        
        if choice == "1":
            view_all_entries(qdrant_client, show_vectors=False)
        elif choice == "2":
            view_all_entries(qdrant_client, show_vectors=True)
        elif choice == "3":
            try:
                limit = int(input("Enter number of entries to view: "))
                view_all_entries(qdrant_client, show_vectors=False, limit=limit)
            except ValueError:
                print("Invalid number entered.")
        elif choice == "4":
            query = input("Enter search text: ").strip()
            if query:
                search_entries(qdrant_client, query)
        elif choice == "5":
            get_collection_stats(qdrant_client)
        elif choice == "6":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
        
        print("")

if __name__ == "__main__":
    main()
