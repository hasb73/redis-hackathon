#!/usr/bin/env python3
"""
Test script for Log Analysis Chat integration with Redis VL
"""

import requests
import json
import time

CHAT_URL = "http://localhost:8004"
EMBEDDING_URL = "http://localhost:8000"

def test_service_health():
    """Test if services are running"""
    print("🔍 Testing Service Health...")
    print("-" * 50)
    
    # Test chat service
    try:
        response = requests.get(f"{CHAT_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Chat Service: {data['status']}")
            print(f"   Redis AI Available: {data.get('redis_ai_available', False)}")
        else:
            print(f"❌ Chat Service: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Chat Service: {e}")
    
    # Test embedding service
    try:
        response = requests.get(f"{EMBEDDING_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ Embedding Service: Healthy")
        else:
            print(f"❌ Embedding Service: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Embedding Service: {e}")
    
    print()

def test_chat_query(query, description):
    """Test a chat query"""
    print(f"📝 Testing: {description}")
    print(f"   Query: \"{query}\"")
    
    try:
        response = requests.post(
            f"{CHAT_URL}/chat",
            json={"message": query},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response received:")
            
            # Print first 200 chars of response
            response_text = data.get('response', '')
            if len(response_text) > 200:
                print(f"   {response_text[:200]}...")
            else:
                print(f"   {response_text}")
            
            # Print analysis data if available
            if data.get('analysis_data'):
                print(f"   📊 Analysis Data: {list(data['analysis_data'].keys())}")
            
            # Print suggestions
            if data.get('suggestions'):
                print(f"   💡 Suggestions: {len(data['suggestions'])} available")
            
            return True
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    finally:
        print()

def test_embedding_service():
    """Test embedding service directly"""
    print("🧪 Testing Embedding Service...")
    print("-" * 50)
    
    test_text = "Receiving block blk_123 src: /10.1.1.1:5000"
    
    try:
        response = requests.post(
            f"{EMBEDDING_URL}/embed",
            json={"texts": [test_text]},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            embeddings = data.get('embeddings', [])
            if embeddings:
                print(f"✅ Embedding generated: {len(embeddings[0])} dimensions")
                print(f"   Sample values: {embeddings[0][:5]}...")
            else:
                print(f"❌ No embeddings returned")
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()

def main():
    """Run all tests"""
    print("=" * 50)
    print("Log Analysis Chat Integration Tests")
    print("=" * 50)
    print()
    
    # Test service health
    test_service_health()
    
    # Test embedding service
    test_embedding_service()
    
    # Test basic queries
    print("🧪 Testing Chat Queries...")
    print("-" * 50)
    print()
    
    test_queries = [
        ("How many anomalies were detected?", "Anomaly Count Query"),
        ("Show me the latest anomalies", "Latest Anomalies Query"),
        ("What's the system performance?", "Performance Query"),
        ("Check system health", "System Health Query"),
        ("Give me a log summary", "Log Summary Query"),
        ("Find logs similar to receiving block", "Vector Search Query"),
        ("Show me logs with DataNode errors", "Natural Language Search"),
        ("What can you help me with?", "Help Query"),
    ]
    
    results = []
    for query, description in test_queries:
        success = test_chat_query(query, description)
        results.append((description, success))
        time.sleep(1)  # Rate limiting
    
    # Summary
    print("=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {description}")
    
    print()
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print()
    
    if passed == total:
        print("🎉 All tests passed! Chat integration is working correctly.")
    elif passed > 0:
        print("⚠️ Some tests failed. Check the logs above for details.")
    else:
        print("❌ All tests failed. Check if services are running:")
        print("   docker-compose ps")
        print("   docker-compose logs log-analysis-chat")

if __name__ == "__main__":
    main()
