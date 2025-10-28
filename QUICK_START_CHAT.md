# Quick Start Guide - Log Analysis Chat with Redis VL

## 🚀 Getting Started

### 1. Start All Services
```bash
docker-compose up -d
```

### 2. Verify Services are Running
```bash
# Check all services
docker-compose ps

# Should see:
# - zookeeper (port 2181)
# - kafka (port 9092)
# - redis-stack (ports 6379, 8001)
# - redis-ai (port 6380)
# - embedding (port 8000)
# - log-analysis-chat (port 8004)
```

### 3. Access the Chat Interface
Open your browser to: **http://localhost:8004**

## 💬 Example Queries

### Basic Queries
```
1. "How many anomalies were detected?"
   → Returns total anomaly count and statistics

2. "Show me the latest anomalies"
   → Returns the 5 most recent anomalies

3. "What's the system performance?"
   → Returns processing time and throughput metrics

4. "Check system health"
   → Returns status of all services

5. "Give me a log summary"
   → Returns comprehensive overview
```

### Natural Language Vector Search
```
1. "Find logs similar to receiving block"
   → Searches for logs about block reception

2. "Show me logs with DataNode errors"
   → Finds DataNode-related error logs

3. "Search for logs containing PacketResponder"
   → Finds PacketResponder-related logs

4. "Find patterns like block allocation"
   → Searches for block allocation patterns

5. "Show logs about NameSystem operations"
   → Finds NameSystem-related logs
```

## 🧪 Testing the Integration

### Quick Test
```bash
# Test chat service
curl http://localhost:8004/health

# Test embedding service
curl http://localhost:8000/health

# Send a test query
curl -X POST http://localhost:8004/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How many anomalies were detected?"}'
```

### Run Full Integration Tests
```bash
python3 log-analysis-chat/test_integration.py
```

## 🔍 How It Works

### 1. Log Processing Flow
```
HDFS Logs → Kafka → Spark Job → Normalize → Check Redis → Generate Embedding → Store in Redis VL
```

### 2. Query Processing Flow
```
User Query → Extract Intent → Generate Embedding → Vector Search → Return Similar Logs
```

### 3. Deduplication
```
Log: "Receiving block blk_123 src: /10.1.1.1:5000"
Normalized: "Receiving block <BLOCK_ID> src: /<IP>:<PORT>"
Hash: MD5(normalized) → Check Redis → Skip if exists
```

## 📊 Monitoring

### Check Redis Data
```bash
# Connect to Redis
docker exec -it redis-hackathon-redis-stack-1 redis-cli

# Check index info
FT.INFO logs_embeddings

# Count log entries
KEYS log_entry:* | wc -l

# Check normalized patterns
KEYS normalized:* | wc -l
```

### View Logs
```bash
# Chat service logs
docker-compose logs -f log-analysis-chat

# Spark job logs
docker-compose logs -f spark

# Embedding service logs
docker-compose logs -f embedding
```

## 🐛 Troubleshooting

### Problem: "No similar logs found"

**Solution 1:** Check if logs are being processed
```bash
# Check Spark job
docker-compose logs spark | grep "processed"

# Check Redis
docker exec -it redis-hackathon-redis-stack-1 redis-cli
> KEYS log_entry:*
```

**Solution 2:** Verify embedding service
```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["test log message"]}'
```

### Problem: "Send button not working"

**Solution 1:** Check browser console (F12)
- Look for JavaScript errors
- Check WebSocket connection status

**Solution 2:** Check chat service
```bash
docker-compose logs log-analysis-chat
docker-compose restart log-analysis-chat
```

### Problem: "Database not found"

**Solution:** Check if anomaly detection service created the database
```bash
ls -la anomaly-detection-service/anomaly_detection.db

# If missing, the service needs to process some logs first
```

## 📈 Performance Tips

### 1. Batch Processing
- Spark processes logs every 10 seconds
- Adjust with: `.trigger(processingTime='10 seconds')`

### 2. Deduplication
- Normalized patterns cached for 30 days
- Reduces embedding calls by 80-90%

### 3. Vector Search
- KNN search optimized with FLAT index
- Adjust num_results for speed vs accuracy

### 4. Embedding Service
- Uses sentence-transformers model
- 384-dimensional embeddings
- ~50ms per embedding

## 🎯 Key Features

### ✅ Implemented
- [x] Dynamic element normalization
- [x] Intelligent deduplication
- [x] Vector similarity search
- [x] Natural language queries
- [x] WebSocket support
- [x] HTTP API fallback
- [x] System health monitoring
- [x] Performance metrics

### 🚧 Future Enhancements
- [ ] Query history
- [ ] Favorite queries
- [ ] Export results
- [ ] Advanced filters
- [ ] Time range queries
- [ ] Anomaly clustering
- [ ] Trend analysis
- [ ] Alert notifications

## 📚 Documentation

- **Integration Fixes:** `INTEGRATION_FIXES.md`
- **Spark Refactoring:** `spark/REFACTORING_NOTES.md`
- **Chat Testing:** `log-analysis-chat/TEST_CHAT.md`
- **This Guide:** `QUICK_START_CHAT.md`

## 🆘 Need Help?

### Check Service Status
```bash
docker-compose ps
docker-compose logs --tail=50 log-analysis-chat
```

### Restart Services
```bash
docker-compose restart log-analysis-chat
docker-compose restart embedding
```

### Full Reset
```bash
docker-compose down
docker-compose up -d
```

## 🎉 Success Indicators

You'll know everything is working when:
1. ✅ Chat interface loads at http://localhost:8004
2. ✅ Health check returns "healthy"
3. ✅ Queries return responses with suggestions
4. ✅ Vector search finds similar logs
5. ✅ System shows anomaly statistics

Happy querying! 🚀
