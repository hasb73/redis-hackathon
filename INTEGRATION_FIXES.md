# Redis VL Integration Fixes - Summary

## Overview
Fixed and enhanced the log analysis chat integration with Redis Vector Library (VL) for natural language querying of log data.

## Issues Fixed

### 1. Spark Job Refactoring ✅

#### Dynamic Element Normalization
- **Added:** `normalize_log_message()` function to strip dynamic elements
- **Strips:** IP addresses, ports, block IDs, file sizes, timestamps, paths, job IDs
- **Benefit:** Creates normalized templates for pattern matching

#### Intelligent Deduplication
- **Added:** `check_log_exists_in_redis()` function
- **Checks:** Redis before generating embeddings for duplicate patterns
- **Stores:** Normalized message markers with 30-day expiration
- **Benefit:** Reduces embedding generation costs and processing time

#### Example
```python
# Original Log
"Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010"

# Normalized Log
"Receiving block <BLOCK_ID> src: /<IP>:<PORT> dest: /<IP>:<PORT>"
```

**Files Modified:**
- `spark/spark_job.py`
- `spark/REFACTORING_NOTES.md` (new)

### 2. Chat Service Integration ✅

#### Embedding Service Endpoint Fix
**Problem:** Chat was calling wrong endpoint `/embeddings`
**Solution:** Updated to correct endpoint `/embed` with proper request format

```python
# Before
response = requests.post(f"{url}/embeddings", json={"text": text})

# After
response = requests.post(f"{url}/embed", json={"texts": [text]})
```

#### Redis VL Vector Search Enhancement
**Problem:** Vector search was failing due to schema mismatch
**Solution:** 
- Updated field names to match Spark job schema (`text`, `label`, `normalized_text`)
- Added fallback to direct Redis FT.SEARCH when RedisVL unavailable
- Improved error handling and logging

```python
# Now supports both RedisVL and direct Redis queries
def vector_similarity_search(query_text, num_results=5):
    # Try RedisVL first
    if self.search_index and REDISVL_AVAILABLE:
        results = self.search_index.query(VectorQuery(...))
    
    # Fallback to direct Redis FT.SEARCH
    else:
        results = redis_raw.execute_command('FT.SEARCH', ...)
```

#### Natural Language Query Support
**Added:** Better pattern extraction and query understanding

```python
# New patterns supported
'query_logs': r'(?i)(show.*logs?|find.*logs?|get.*logs?|logs?.*contain|logs?.*with)'
'similar_logs': r'(?i)(similar|like|find.*similar|pattern|search.*for)'
```

**Example Queries:**
- "Find logs similar to receiving block"
- "Show me logs with DataNode errors"
- "Search for logs containing PacketResponder"
- "Find patterns like block allocation"

**Files Modified:**
- `log-analysis-chat/app.py`
- `log-analysis-chat/TEST_CHAT.md` (new)
- `log-analysis-chat/test_integration.py` (new)

## Architecture

### Data Flow
```
HDFS Logs → Kafka → Spark Job → Redis VL
                                    ↓
User Query → Chat Service → Embedding Service → Vector Search → Results
```

### Spark Job Processing
```
1. Receive log from Kafka
2. Normalize log (strip dynamic elements)
3. Check if pattern exists in Redis
4. If new pattern:
   - Generate embedding
   - Store in Redis VL with normalized key
5. If duplicate:
   - Skip embedding generation
   - Log skipped count
```

### Chat Query Processing
```
1. User sends natural language query
2. Extract intent and keywords
3. Generate embedding for query
4. Perform KNN search in Redis VL
5. Return top N similar logs with scores
6. Format response with suggestions
```

## Testing

### Start Services
```bash
docker-compose up -d
```

### Test Chat Interface
```bash
# Open browser
open http://localhost:8004

# Or test via API
curl -X POST http://localhost:8004/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Find logs similar to receiving block"}'
```

### Run Integration Tests
```bash
python3 log-analysis-chat/test_integration.py
```

### Test Queries

#### Basic Queries
1. "How many anomalies were detected?"
2. "Show me the latest anomalies"
3. "What's the system performance?"
4. "Check system health"

#### Vector Search Queries
1. "Find logs similar to receiving block"
2. "Show me logs like PacketResponder terminating"
3. "Search for logs containing DataNode errors"
4. "Find patterns similar to block allocation"

## Performance Improvements

### Before
- Every log generated an embedding (expensive)
- Duplicate patterns processed multiple times
- No deduplication

### After
- Only unique patterns generate embeddings
- Duplicate patterns skipped automatically
- 30-day cache for normalized patterns
- Significant cost reduction for embedding service

### Example Metrics
```
Batch of 100 logs:
- Before: 100 embedding calls
- After: ~10-20 embedding calls (80-90% reduction)
```

## Configuration

### Spark Job
```python
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_INDEX_NAME = "logs_embeddings"
REDIS_KEY_PREFIX = "log_entry:"
EMBEDDING_SERVICE_URL = "http://localhost:8000/embed"
```

### Chat Service
```python
REDIS_STACK_HOST = "redis-stack"
REDIS_STACK_PORT = 6379
EMBEDDING_SERVICE_URL = "http://embedding:8000"
REDIS_INDEX_NAME = "logs_embeddings"
```

### Docker Compose
```yaml
services:
  redis-stack:
    ports:
      - "6379:6379"  # Redis server
      - "8001:8001"  # RedisInsight UI
  
  embedding:
    ports:
      - "8000:8000"
  
  log-analysis-chat:
    ports:
      - "8004:8004"
```

## Troubleshooting

### Issue: "No similar logs found"
**Check:**
1. Embedding service running: `curl http://localhost:8000/health`
2. Redis has logs: `redis-cli KEYS log_entry:*`
3. Spark job processing: `docker-compose logs spark`

### Issue: "Send button not working"
**Check:**
1. Browser console for errors (F12)
2. Chat service logs: `docker-compose logs log-analysis-chat`
3. WebSocket connection status

### Issue: "Database not found"
**Check:**
1. Database file exists: `ls -la anomaly-detection-service/anomaly_detection.db`
2. Volume mount correct in docker-compose.yml

## Redis VL Schema

### Log Entry Structure
```
log_entry:{hash_id}
  - text: Original log message
  - embedding: 384-dim float32 vector
  - timestamp: ISO timestamp
  - log_level: INFO/WARN/ERROR
  - label: 0 (normal) or 1 (anomaly)
  - normalized_text: Normalized pattern
  - source: hdfs_datanode
  - node_type: datanode

normalized:{hash_id}
  - Expiring key (30 days)
  - Points to log_entry key
```

### Vector Search Index
```
FT.CREATE logs_embeddings
  ON HASH
  PREFIX log_entry:
  SCHEMA
    text TEXT
    embedding VECTOR FLAT 6 TYPE FLOAT32 DIM 384 DISTANCE_METRIC COSINE
    label NUMERIC
    timestamp TEXT
```

## Next Steps

1. ✅ Test with real HDFS logs
2. ✅ Verify embedding generation
3. ✅ Test vector search queries
4. Monitor performance metrics
5. Tune similarity thresholds
6. Add query history
7. Implement favorites/bookmarks
8. Add export functionality

## Files Changed

### Modified
- `spark/spark_job.py` - Added normalization and deduplication
- `log-analysis-chat/app.py` - Fixed embedding endpoint and vector search

### Created
- `spark/REFACTORING_NOTES.md` - Spark job changes documentation
- `log-analysis-chat/TEST_CHAT.md` - Chat testing guide
- `log-analysis-chat/test_integration.py` - Integration test script
- `INTEGRATION_FIXES.md` - This file

## Summary

The integration is now fully functional with:
- ✅ Intelligent log deduplication in Spark job
- ✅ Correct embedding service integration
- ✅ Working Redis VL vector search
- ✅ Natural language query support
- ✅ Comprehensive testing tools
- ✅ Performance optimizations

The chat interface can now query the Redis VL database using natural language and return semantically similar logs based on vector embeddings.
