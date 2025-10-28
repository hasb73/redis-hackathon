# Redis MCP Troubleshooting Guide

## Issue: "Connection closed" Error

### Problem
```
[error] [redis] Error connecting to MCP server: MCP error -32000: Connection closed
```

### Root Cause
The MCP server couldn't establish a stable connection to Redis.

### Solution Applied ✅

Updated the MCP configuration to use:
- `127.0.0.1` instead of `localhost` (more reliable)
- Added explicit `REDIS_HOST`, `REDIS_PORT`, and `REDIS_DB` environment variables

### Updated Configuration

**File**: `.kiro/settings/mcp.json`
```json
{
  "mcpServers": {
    "redis": {
      "command": "/opt/homebrew/bin/uvx",
      "args": ["mcp-server-redis"],
      "env": {
        "REDIS_URL": "redis://127.0.0.1:6379",
        "REDIS_HOST": "127.0.0.1",
        "REDIS_PORT": "6379",
        "REDIS_DB": "0"
      },
      "disabled": false,
      "autoApprove": [
        "redis_get",
        "redis_set",
        "redis_keys",
        "redis_hgetall",
        "redis_hget",
        "redis_ft_search",
        "redis_ft_info"
      ]
    }
  }
}
```

## Next Steps

### 1. Reconnect MCP Servers

**In Kiro IDE:**
```
Cmd+Shift+P → "MCP: Reconnect Servers"
```

**Or restart Kiro completely**

### 2. Check MCP Logs

**View → Output → Select "MCP" from dropdown**

Look for:
- ✅ `[redis] MCP connection established`
- ✅ `[redis] Server ready`

### 3. Test Connection

Try this query in Kiro chat:
```
"Show me all keys matching log_entry:*"
```

Expected result: List of 127 keys

## Verification Tests

### Test 1: Redis Connection
```bash
python3 -c "
import redis
r = redis.Redis(host='127.0.0.1', port=6379)
print('✅ PING:', r.ping())
print('✅ Keys:', len(r.keys('log_entry:*')))
"
```

Expected output:
```
✅ PING: True
✅ Keys: 127
```

### Test 2: Docker Container
```bash
docker ps --filter "name=redis-stack"
```

Expected: Container running and healthy

### Test 3: Port Accessibility
```bash
nc -zv 127.0.0.1 6379
```

Expected: `Connection to 127.0.0.1 port 6379 [tcp/*] succeeded!`

### Test 4: MCP Server Manual Start
```bash
REDIS_URL="redis://127.0.0.1:6379" uvx mcp-server-redis
```

Expected: Server starts without errors (Ctrl+C to stop)

## Common Issues & Solutions

### Issue 1: "Connection refused"

**Cause**: Redis container not running

**Solution**:
```bash
docker-compose up -d redis-stack
docker ps --filter "name=redis-stack"
```

### Issue 2: "Unknown command"

**Cause**: Redis doesn't support the command (e.g., FT.SEARCH not available)

**Solution**: Ensure you're using Redis Stack, not regular Redis
```bash
docker exec redis-hackathon-redis-stack-1 redis-cli MODULE LIST
```

Should show: `search` module loaded

### Issue 3: "Authentication required"

**Cause**: Redis requires password

**Solution**: Add password to REDIS_URL
```json
"REDIS_URL": "redis://:password@127.0.0.1:6379"
```

### Issue 4: MCP Server Crashes

**Cause**: Missing dependencies or configuration

**Solution**: Reinstall MCP server
```bash
uvx --force mcp-server-redis
```

### Issue 5: "spawn uvx ENOENT"

**Cause**: uvx not in PATH

**Solution**: Use full path
```json
"command": "/opt/homebrew/bin/uvx"
```

## Debug Mode

### Enable Verbose Logging

Add to MCP config:
```json
"env": {
  "REDIS_URL": "redis://127.0.0.1:6379",
  "LOG_LEVEL": "DEBUG"
}
```

### Check MCP Logs

**In Kiro:**
1. View → Output
2. Select "MCP" from dropdown
3. Look for connection attempts and errors

### Manual Testing

Test the MCP server directly:
```bash
# Start server
REDIS_URL="redis://127.0.0.1:6379" uvx mcp-server-redis

# In another terminal, test Redis
docker exec redis-hackathon-redis-stack-1 redis-cli PING
```

## Alternative: Use Vector Search Helper

If MCP continues to have issues, you can still query Redis using the helper script:

```bash
# Semantic search
python3 redis-mcp-server/vector_search_helper.py "receiving block" 5

# Direct Redis queries
docker exec redis-hackathon-redis-stack-1 redis-cli KEYS "log_entry:*"
docker exec redis-hackathon-redis-stack-1 redis-cli HGETALL log_entry:abc123
```

## Success Indicators

You'll know it's working when:

1. ✅ MCP logs show "Connection established"
2. ✅ No "Connection closed" errors
3. ✅ Queries return data from Redis
4. ✅ MCP panel shows Redis server as "Connected"
5. ✅ Can list and retrieve log entries

## Still Having Issues?

### Option 1: Use Python Script Instead

Create a simple Python script to query Redis:
```python
import redis
r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)

# List logs
keys = r.keys('log_entry:*')
print(f"Found {len(keys)} logs")

# Get log
log = r.hgetall('log_entry:abc123')
print(log)
```

### Option 2: Use Redis CLI

```bash
# Via Docker
docker exec redis-hackathon-redis-stack-1 redis-cli KEYS "log_entry:*"
docker exec redis-hackathon-redis-stack-1 redis-cli HGETALL log_entry:abc123
```

### Option 3: Use Vector Search Helper

```bash
python3 redis-mcp-server/vector_search_helper.py "your query" 5
```

## Contact & Resources

- **Redis MCP Server**: https://github.com/redis/mcp-redis
- **MCP Protocol**: https://modelcontextprotocol.io/
- **Redis Stack Docs**: https://redis.io/docs/stack/

---

## Current Status

- ✅ Configuration updated to use `127.0.0.1`
- ✅ Redis connection verified
- ✅ 127 log entries available
- ⏳ **Next**: Reconnect MCP servers in Kiro

**Action Required**: Reconnect MCP servers in Kiro IDE
```
Cmd+Shift+P → "MCP: Reconnect Servers"
```
