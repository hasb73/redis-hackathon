# Redis VL Query Setup with Official Redis MCP

## Using Official Redis MCP Server

Instead of building a custom MCP server, we'll use the official Redis MCP server from Redis Labs:
https://github.com/redis/mcp-redis

## Installation

### Option 1: Using uvx (Recommended)

```bash
# The Redis MCP server will be auto-installed when configured in Kiro
# No manual installation needed!
```

### Option 2: Manual Installation

```bash
# Install via pip
pip install mcp-server-redis

# Or using uv
uv pip install mcp-server-redis
```

## Kiro IDE Configuration

Add to `.kiro/settings/mcp.json`:

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
        "redis_set",
        "redis_keys",
        "redis_hgetall",
        "redis_ft_search"
      ]
    }
  }
}
```

## Available Redis Commands for Vector Search

### 1. FT.SEARCH - Vector Similarity Search

```
Use redis_ft_search tool with:
- index: "logs_embeddings"
- query: "*=>[KNN 5 @embedding $vec AS score]"
- params: {"vec": <embedding_bytes>}
```

### 2. HGETALL - Get Log Entry

```
Use redis_hgetall tool with:
- key: "log_entry:abc123"
```

### 3. KEYS - List Log Entries

```
Use redis_keys tool with:
- pattern: "log_entry:*"
```

### 4. FT.INFO - Index Statistics

```
Use redis_ft_search tool with:
- command: "FT.INFO"
- args: ["logs_embeddings"]
```

## Example Queries in Kiro

### Query 1: Get Index Stats

```
Can you run FT.INFO on the logs_embeddings index?
```

### Query 2: List All Log Keys

```
Show me all keys matching log_entry:*
```

### Query 3: Get a Specific Log

```
Get the log entry with key log_entry:abc123
```

### Query 4: Vector Search (Advanced)

For vector search, you'll need to:
1. Get embedding from embedding service
2. Convert to bytes
3. Use FT.SEARCH with KNN query

```
Search the logs_embeddings index for similar logs using vector search
```

## Helper Script for Vector Search

Since vector search requires embedding generation, use this helper script:

```bash
python3 redis-mcp-server/vector_search_helper.py "receiving block"
```

## Testing

```bash
# Test Redis connection
redis-cli PING

# Test index exists
redis-cli FT.INFO logs_embeddings

# Test log entries exist
redis-cli KEYS "log_entry:*" | head -5

# Get a sample log
redis-cli HGETALL log_entry:$(redis-cli KEYS "log_entry:*" | head -1 | cut -d: -f2)
```

## Architecture

```
Kiro IDE
   ↓
Redis MCP Server (official)
   ↓
Redis Stack (localhost:6379)
   ↓
logs_embeddings index
   ↓
127 log entries with embeddings
```

## Advantages of Official Redis MCP

1. ✅ **Maintained by Redis Labs** - Always up-to-date
2. ✅ **Full Redis Command Support** - All Redis commands available
3. ✅ **No Custom Code** - Less maintenance
4. ✅ **Battle-tested** - Used in production
5. ✅ **Auto-installation** - uvx handles everything

## Natural Language Queries

Once configured in Kiro, you can ask:

- "Show me all log entries in Redis"
- "Get the details of log entry abc123"
- "What's the status of the logs_embeddings index?"
- "List all keys matching log_entry pattern"
- "Get statistics about the Redis index"

## Advanced: Vector Search Workflow

For semantic search, create a workflow:

1. **Get Embedding**
   ```bash
   curl -X POST http://localhost:8000/embed \
     -H "Content-Type: application/json" \
     -d '{"texts": ["receiving block"]}'
   ```

2. **Convert to Bytes**
   ```python
   import numpy as np
   embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
   ```

3. **Search Redis**
   ```
   FT.SEARCH logs_embeddings "*=>[KNN 5 @embedding $vec AS score]" 
   PARAMS 2 vec <embedding_bytes> 
   SORTBY score 
   RETURN 4 text timestamp label score
   ```

## Troubleshooting

### "Connection refused"
```bash
# Check Redis is running
docker-compose ps redis-stack

# Test connection
redis-cli PING
```

### "Unknown index name"
```bash
# Check index exists
redis-cli FT.INFO logs_embeddings

# If not, run Spark job to create it
docker-compose up spark
```

### "MCP server not found"
```bash
# Reinstall
uvx --force mcp-server-redis

# Or check Kiro MCP settings
# Command Palette → "MCP: Reconnect Servers"
```

## Next Steps

1. Configure `.kiro/settings/mcp.json` with Redis MCP
2. Restart Kiro or reconnect MCP servers
3. Test with: "Show me all log entries"
4. Use vector_search_helper.py for semantic search

## Resources

- Official Redis MCP: https://github.com/redis/mcp-redis
- Redis Stack Docs: https://redis.io/docs/stack/
- RedisSearch Docs: https://redis.io/docs/stack/search/
- MCP Protocol: https://modelcontextprotocol.io/
