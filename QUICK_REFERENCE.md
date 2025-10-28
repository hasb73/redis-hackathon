# Redis VL MCP - Quick Reference Card

## 🚀 Quick Start (3 Steps)

```bash
# 1. Ensure services are running
docker-compose ps

# 2. Test vector search
python3 redis-mcp-server/vector_search_helper.py "receiving block" 3

# 3. Use in Kiro IDE
# Command Palette → "MCP: Reconnect Servers"
# Then ask: "Show me all log entries"
```

## 📋 Common Commands

### Vector Search
```bash
# Basic search
python3 redis-mcp-server/vector_search_helper.py "<query>" [limit]

# Examples
python3 redis-mcp-server/vector_search_helper.py "receiving block"
python3 redis-mcp-server/vector_search_helper.py "DataNode error" 10
python3 redis-mcp-server/vector_search_helper.py "block allocation" 5
```

### Redis CLI
```bash
# Count logs
redis-cli KEYS "log_entry:*" | wc -l

# Get log
redis-cli HGETALL log_entry:abc123

# Index stats
redis-cli FT.INFO logs_embeddings

# Test connection
redis-cli PING
```

### Kiro IDE Queries
```
"Show me all log entries"
"Get log_entry:abc123"
"What's in the logs_embeddings index?"
"List keys matching log_entry:*"
```

## 🔧 Configuration Files

| File | Purpose |
|------|---------|
| `.kiro/settings/mcp.json` | MCP server config |
| `docker-compose.yml` | Services configuration |
| `spark/spark_job.py` | Log processing |

## 📊 System Status

```bash
# Check all services
docker-compose ps

# Check Redis
redis-cli PING

# Check embedding service
curl http://localhost:8000/health

# Check log count
redis-cli KEYS "log_entry:*" | wc -l
```

## 🎯 Key Numbers

- **127** log entries in Redis
- **384** embedding dimensions
- **80-90%** cost reduction via deduplication
- **~50ms** embedding generation time
- **~10-20ms** vector search time

## 📁 Important Files

```
.kiro/settings/mcp.json                    # MCP config
redis-mcp-server/vector_search_helper.py   # Search tool
REDIS_MCP_GUIDE.md                         # Full guide
FINAL_SUMMARY.md                           # Summary
```

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| MCP not working | Reconnect: Command Palette → "MCP: Reconnect Servers" |
| Redis error | `docker-compose restart redis-stack` |
| No embeddings | `docker-compose restart embedding` |
| No logs | Run Spark job: `docker-compose up spark` |

## 🔍 Example Queries

### Find Similar Logs
```bash
python3 redis-mcp-server/vector_search_helper.py "receiving block" 5
```

### Get Log Details (Kiro)
```
Get the hash for log_entry:abc123
```

### List All Logs (Kiro)
```
Show me all keys matching log_entry:*
```

### Index Stats (CLI)
```bash
redis-cli FT.INFO logs_embeddings
```

## 📖 Documentation

- **Full Guide**: `REDIS_MCP_GUIDE.md`
- **Setup**: `redis-mcp-server/SETUP.md`
- **Summary**: `FINAL_SUMMARY.md`
- **Architecture**: `ARCHITECTURE_DIAGRAM.md`

## ✅ Health Check

```bash
# All green = ready to use
redis-cli PING                              # → PONG
curl http://localhost:8000/health           # → 200 OK
redis-cli FT.INFO logs_embeddings           # → Index info
redis-cli KEYS "log_entry:*" | wc -l        # → 127
```

---

**Status**: ✅ Ready to use!

**Quick Test**: `python3 redis-mcp-server/vector_search_helper.py "test" 3`
