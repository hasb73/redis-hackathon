# Redis VL Query with MCP - Complete Guide

## Overview

Query your Redis Vector Library database using the official Redis MCP server in Kiro IDE. This provides natural language access to your log embeddings and vector search capabilities.

## ✅ What's Working

- ✅ Redis Stack with 127 log entries
- ✅ Vector embeddings (384 dimensions)
- ✅ logs_embeddings index
- ✅ Embedding service (port 8000)
- ✅ Vector similarity search
- ✅ Official Redis MCP server support

## 🚀 Quick Start

### 1. MCP Configuration

The MCP configuration is already set up in `.kiro/settings/mcp.json`:

```json
{
  "mcpServers": {
    "redis": {
      "command": "uvx",
      "args": ["mcp-server-redis"],
      "env": {
        "REDIS_URL": "redis://localhost:6379"
      },
      "disabled": false,
      "autoApprove": [
        "redis_get",
        "redis_keys",
        "redis_hgetall",
        "redis_ft_search"
      ]
    }
  }
}
```

### 2. Restart Kiro or Reconnect MCP

- Command Palette → "MCP: Reconnect Servers"
- Or restart Kiro IDE

### 3. Test the Connection

In Kiro chat, ask:
```
Show me all keys matching log_entry:*
```

## 📊 Available Queries

### Basic Redis Queries

#### List All Log Entries
```
Show me all keys with pattern log_entry:*
```

#### Get Specific Log
```
Get the hash for key log_entry:abc123
```

#### Index Statistics
```
Run FT.INFO on logs_embeddings index
```

### Vector Search (Using Helper Script)

```bash
# Search for similar logs
python3 redis-mcp-server/vector_search_helper.py "receiving block" 5

# Search for errors
python3 redis-mcp-server/vector_search_helper.py "DataNode error" 10

# Search for specific patterns
python3 redis-mcp-server/vector_search_helper.py "PacketResponder terminating" 3
```

## 🔧 Helper Script Usage

The `vector_search_helper.py` script provides semantic search:

```bash
# Basic usage
python3 redis-mcp-server/vector_search_helper.py "<query>" [limit]

# Examples
python3 redis-mcp-server/vector_search_helper.py "receiving block"
python3 redis-mcp-server/vector_search_helper.py "block allocation" 10
python3 redis-mcp-server/vector_search_helper.py "NameSystem operations" 5
```

### Output Format

```
🔍 Searching for: 'receiving block'
📊 Generating embedding...
✅ Got embedding: 384 dimensions
🔗 Connecting to Redis...
✅ Connected
🔎 Searching index 'logs_embeddings'...
✅ Found 3 results

1. ✅ Similarity: 0.6703
   ID: log_entry:abc123
   Text: Received block blk_123 of size 67108864...
   Timestamp: 2025-10-28T17:53:52
   Level: INFO
   Label: Normal
```

## 🎯 Example Workflows

### Workflow 1: Explore Log Data

```
1. "How many log entries are in Redis?"
   → Use: redis_keys with pattern "log_entry:*"

2. "Show me a sample log entry"
   → Use: redis_hgetall with a key

3. "What's in the logs_embeddings index?"
   → Use: FT.INFO logs_embeddings
```

### Workflow 2: Find Similar Logs

```bash
# 1. Search for logs about receiving blocks
python3 redis-mcp-server/vector_search_helper.py "receiving block" 5

# 2. Get details of a specific log
# In Kiro: "Get log_entry:<id_from_search>"

# 3. Find more similar logs
python3 redis-mcp-server/vector_search_helper.py "block reception" 10
```

### Workflow 3: Analyze Anomalies

```bash
# 1. Search for error patterns
python3 redis-mcp-server/vector_search_helper.py "error" 10

# 2. Check if any are anomalies (label=1)
# Look at the "Label" field in results

# 3. Find similar anomalous patterns
python3 redis-mcp-server/vector_search_helper.py "failed operation" 5
```

## 📋 Redis Commands via MCP

### Available Tools

| Tool | Description | Example |
|------|-------------|---------|
| `redis_keys` | List keys matching pattern | `redis_keys("log_entry:*")` |
| `redis_get` | Get string value | `redis_get("key")` |
| `redis_hgetall` | Get all hash fields | `redis_hgetall("log_entry:abc")` |
| `redis_hget` | Get hash field | `redis_hget("log_entry:abc", "text")` |
| `redis_ft_search` | Full-text search | `redis_ft_search("logs_embeddings", "*")` |

### Example MCP Queries in Kiro

```
1. "List all log entry keys"
2. "Get the content of log_entry:abc123"
3. "Search the logs_embeddings index"
4. "Show me index statistics for logs_embeddings"
5. "Get the text field from log_entry:xyz789"
```

## 🧪 Testing

### Test 1: Redis Connection
```bash
redis-cli PING
# Expected: PONG
```

### Test 2: Index Exists
```bash
redis-cli FT.INFO logs_embeddings
# Expected: Index information
```

### Test 3: Log Count
```bash
redis-cli KEYS "log_entry:*" | wc -l
# Expected: 127
```

### Test 4: Vector Search
```bash
python3 redis-mcp-server/vector_search_helper.py "test query" 3
# Expected: 3 similar logs with similarity scores
```

### Test 5: Embedding Service
```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["test"]}'
# Expected: {"embeddings": [[...]]}
```

## 🏗️ Architecture

```
┌─────────────┐
│  Kiro IDE   │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Redis MCP Server    │
│ (Official)          │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Redis Stack         │
│ localhost:6379      │
│                     │
│ ┌─────────────────┐ │
│ │ logs_embeddings │ │
│ │ 127 documents   │ │
│ │ 384-dim vectors │ │
│ └─────────────────┘ │
└─────────────────────┘
       ▲
       │
┌──────┴──────────────┐
│ Embedding Service   │
│ localhost:8000      │
└─────────────────────┘
```

## 📚 Data Schema

### Log Entry Structure
```
log_entry:<hash_id>
  - text: Original log message
  - embedding: 384-dim float32 vector
  - timestamp: ISO timestamp
  - log_level: INFO/WARN/ERROR
  - label: 0 (normal) or 1 (anomaly)
  - normalized_text: Pattern template
  - source: hdfs_datanode
  - node_type: datanode
```

### Index Schema
```
Index: logs_embeddings
Fields:
  - text: TEXT
  - embedding: VECTOR (FLAT, FLOAT32, 384, COSINE)
  - label: NUMERIC
  - timestamp: TEXT
```

## 🔍 Query Examples

### Natural Language (in Kiro)

```
1. "How many logs are stored in Redis?"
2. "Show me the first 5 log entry keys"
3. "Get the details of log_entry:abc123"
4. "What's the size of the logs_embeddings index?"
5. "List all keys starting with log_entry"
```

### Programmatic (Python)

```python
import redis

# Connect
r = redis.Redis(host='localhost', port=6379, decode_responses=False)

# List logs
keys = r.keys('log_entry:*')
print(f"Total logs: {len(keys)}")

# Get log
log = r.hgetall('log_entry:abc123')
print(f"Text: {log[b'text'].decode('utf-8')}")

# Index stats
info = r.execute_command('FT.INFO', 'logs_embeddings')
```

### Vector Search (Helper Script)

```bash
# Find logs about blocks
python3 redis-mcp-server/vector_search_helper.py "receiving block"

# Find errors
python3 redis-mcp-server/vector_search_helper.py "error failed"

# Find specific operations
python3 redis-mcp-server/vector_search_helper.py "NameSystem allocate"
```

## 🐛 Troubleshooting

### Issue: "MCP server not responding"

**Solution:**
```bash
# Reconnect MCP servers in Kiro
# Command Palette → "MCP: Reconnect Servers"

# Or restart Kiro IDE
```

### Issue: "Redis connection failed"

**Solution:**
```bash
# Check Redis is running
docker-compose ps redis-stack

# Test connection
redis-cli PING

# Restart if needed
docker-compose restart redis-stack
```

### Issue: "Index not found"

**Solution:**
```bash
# Check index exists
redis-cli FT.INFO logs_embeddings

# If missing, run Spark job
docker-compose up spark
```

### Issue: "Embedding service unavailable"

**Solution:**
```bash
# Check service
curl http://localhost:8000/health

# Restart if needed
docker-compose restart embedding
```

## 📈 Performance

- **Redis Query**: <1ms
- **Vector Search**: 10-20ms
- **Embedding Generation**: ~50ms
- **Index Size**: ~1.5MB for 127 documents

## 🎉 Success Indicators

You'll know everything is working when:

1. ✅ Kiro shows Redis MCP server as connected
2. ✅ `redis_keys` returns log entry keys
3. ✅ `redis_hgetall` returns log details
4. ✅ Vector search helper returns similar logs
5. ✅ Similarity scores are between 0 and 1

## 🚀 Next Steps

1. ✅ Configure MCP in Kiro (already done)
2. ✅ Test basic Redis queries
3. ✅ Use vector search helper for semantic search
4. Build custom workflows combining MCP queries
5. Create dashboards using the data
6. Integrate with other MCP servers

## 📖 Resources

- **Official Redis MCP**: https://github.com/redis/mcp-redis
- **MCP Protocol**: https://modelcontextprotocol.io/
- **Redis Stack**: https://redis.io/docs/stack/
- **RedisSearch**: https://redis.io/docs/stack/search/
- **Vector Search Guide**: https://redis.io/docs/stack/search/reference/vectors/

---

**Status**: ✅ Ready to use!

Query your Redis VL database with natural language in Kiro IDE! 🎉
