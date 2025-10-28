# Log Analysis Chat - Testing Guide

## Overview
The chat interface integrates with Redis VL to provide natural language querying of log data using vector similarity search.

## Fixed Issues

### 1. Embedding Service Integration
- **Fixed:** Corrected endpoint from `/embeddings` to `/embed`
- **Fixed:** Updated request format to match embedding service API (`{"texts": [text]}`)

### 2. Redis VL Vector Search
- **Enhanced:** Improved vector similarity search with fallback mechanisms
- **Added:** Direct Redis FT.SEARCH support when RedisVL is unavailable
- **Fixed:** Field names to match Spark job schema (`text`, `label`, `normalized_text`)

### 3. Natural Language Query Support
- **Added:** Better pattern extraction from user queries
- **Enhanced:** Support for queries like:
  - "Find logs similar to receiving block"
  - "Show me logs with errors"
  - "Search for logs containing DataNode"
  - "Find patterns like PacketResponder"

## Testing the Chat Interface

### 1. Start the Services

```bash
# Start all services
docker-compose up -d

# Check if chat service is running
curl http://localhost:8004/health
```

### 2. Access the Chat Interface

Open your browser to: **http://localhost:8004**

### 3. Test Queries

#### Basic Queries
```
1. "How many anomalies were detected?"
2. "Show me the latest anomalies"
3. "What's the system performance?"
4. "Check system health"
5. "Give me a log summary"
```

#### Vector Search Queries (Natural Language)
```
1. "Find logs similar to receiving block"
2. "Show me logs like PacketResponder terminating"
3. "Search for logs containing DataNode errors"
4. "Find patterns similar to block allocation"
5. "Show logs with NameSystem operations"
```

#### Advanced Queries
```
1. "What caused these anomalies?"
2. "Show anomaly trends"
3. "Find similar error patterns"
4. "Search for logs with high anomaly scores"
```

## How Vector Search Works

### 1. Query Processing
```
User Query → Extract Keywords → Generate Embedding → Vector Search → Return Similar Logs
```

### 2. Example Flow
```
Query: "Find logs similar to receiving block"
↓
Embedding Service: Converts text to 384-dim vector
↓
Redis VL: KNN search in vector space
↓
Results: Top 5 most similar logs with similarity scores
```

### 3. Redis VL Integration
The chat uses Redis Vector Library (RedisVL) to:
- Store log embeddings in Redis
- Perform fast KNN (K-Nearest Neighbors) search
- Find semantically similar logs even with different wording

## Troubleshooting

### Issue: "No similar logs found"

**Possible Causes:**
1. Embedding service not running
2. Redis VL index not populated
3. No logs processed yet

**Solutions:**
```bash
# Check embedding service
curl http://localhost:8000/health

# Check if logs are in Redis
docker exec -it redis-hackathon-redis-stack-1 redis-cli
> FT.INFO logs_embeddings
> KEYS log_entry:*

# Check Spark job is running and processing logs
docker-compose logs spark
```

### Issue: "Send button not working"

**Possible Causes:**
1. WebSocket connection failed
2. JavaScript errors in browser console

**Solutions:**
1. Check browser console for errors (F12)
2. Verify chat service is running: `docker-compose ps log-analysis-chat`
3. Check logs: `docker-compose logs log-analysis-chat`

### Issue: "Database not found"

**Possible Causes:**
1. Anomaly detection service hasn't created the database yet
2. Volume mount issue

**Solutions:**
```bash
# Check if database exists
ls -la anomaly-detection-service/anomaly_detection.db

# Check volume mount in docker-compose
docker-compose config | grep anomaly-detection-service -A 10
```

## API Endpoints

### HTTP Endpoint
```bash
# Send a chat message via HTTP
curl -X POST http://localhost:8004/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How many anomalies were detected?"}'
```

### WebSocket Endpoint
```javascript
// Connect via WebSocket
const ws = new WebSocket('ws://localhost:8004/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({message: "Show me the latest anomalies"}));
};

ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  console.log(response);
};
```

### Health Check
```bash
curl http://localhost:8004/health
```

## Example Responses

### Anomaly Count Query
```json
{
  "response": "📊 **Anomaly Statistics:**\n\n• **Total Anomalies:** 1,234\n• **Total Logs Processed:** 10,000\n• **Anomaly Rate:** 12.34%\n• **Recent Anomalies (1h):** 5",
  "analysis_data": {
    "total_anomalies": 1234,
    "total_logs": 10000,
    "anomaly_rate": 12.34,
    "recent_anomalies": 5
  },
  "suggestions": [
    "Show me the latest anomalies",
    "What's the system performance?",
    "Give me a log summary"
  ],
  "timestamp": "2025-10-28T12:00:00"
}
```

### Vector Search Query
```json
{
  "response": "🔍 **Found 5 Similar Logs:**\n\n**1.** ✅ `2025-10-28 12:00:00`\n   📝 Receiving block blk_123 src: /10.1.1.1:5000...\n   🎯 Similarity: 0.945\n\n...",
  "analysis_data": {
    "similar_logs": [...],
    "query": "receiving block"
  },
  "suggestions": [
    "Find more patterns like these",
    "Show anomaly trends"
  ],
  "timestamp": "2025-10-28T12:00:00"
}
```

## Performance Tips

1. **Batch Processing**: The Spark job processes logs in batches every 10 seconds
2. **Caching**: Normalized log patterns are cached in Redis for 30 days
3. **Vector Search**: KNN search is optimized with HNSW indexing in Redis
4. **Deduplication**: Duplicate log patterns are automatically skipped

## Next Steps

1. Test with real HDFS logs
2. Monitor embedding generation performance
3. Tune vector search parameters (num_results, similarity threshold)
4. Add more natural language patterns
5. Implement query history and favorites
