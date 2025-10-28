# Redis MCP Setup - Complete! ✅

## What Was Fixed

The error `spawn uvx ENOENT` was caused by `uvx` not being in the system PATH. This has been resolved by:

1. ✅ Installed `uv` (which includes `uvx`)
2. ✅ Updated MCP config to use full path: `/opt/homebrew/bin/uvx`
3. ✅ Configured both workspace and user-level MCP settings

## Configuration Files Updated

### 1. Workspace Config
**File**: `.kiro/settings/mcp.json`
```json
{
  "mcpServers": {
    "redis": {
      "command": "/opt/homebrew/bin/uvx",
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
        "redis_hget"
      ]
    }
  }
}
```

### 2. User Config
**File**: `~/.kiro/settings/mcp.json`
- Added Redis MCP server configuration
- Uses full path to `uvx`

## Next Steps

### 1. Reconnect MCP Servers in Kiro

**Option A**: Command Palette
```
Cmd+Shift+P → "MCP: Reconnect Servers"
```

**Option B**: Restart Kiro IDE

### 2. Test the Connection

Once reconnected, try these queries in Kiro chat:

```
1. "Show me all keys matching log_entry:*"
2. "Get the hash for key log_entry:abc123"
3. "How many log entries are in Redis?"
```

### 3. Verify It's Working

You should see:
- ✅ Redis MCP server shows as "Connected" in MCP panel
- ✅ Queries return results from Redis
- ✅ No more "spawn uvx ENOENT" errors

## Testing Redis Connection

```bash
# Test Redis is accessible
redis-cli PING
# Expected: PONG

# Count log entries
redis-cli KEYS "log_entry:*" | wc -l
# Expected: 127

# Test MCP server manually (optional)
REDIS_URL="redis://localhost:6379" uvx mcp-server-redis
# Should start the server (Ctrl+C to stop)
```

## Available Redis Commands via MCP

Once connected, you can use these tools:

| Tool | Description | Example |
|------|-------------|---------|
| `redis_keys` | List keys matching pattern | "Show keys log_entry:*" |
| `redis_get` | Get string value | "Get value of key X" |
| `redis_hgetall` | Get all hash fields | "Get hash log_entry:abc" |
| `redis_hget` | Get specific hash field | "Get text from log_entry:abc" |
| `redis_set` | Set string value | "Set key X to value Y" |

## Vector Search

For semantic search, continue using the helper script:

```bash
python3 redis-mcp-server/vector_search_helper.py "receiving block" 5
```

## Troubleshooting

### Issue: Still getting "spawn uvx ENOENT"

**Solution 1**: Restart Kiro completely
```bash
# Quit Kiro
# Reopen Kiro
```

**Solution 2**: Check uvx path
```bash
which uvx
# Should show: /opt/homebrew/bin/uvx
```

**Solution 3**: Update config with your actual path
```bash
# Find uvx location
which uvx

# Update .kiro/settings/mcp.json with that path
```

### Issue: "Connection refused"

**Solution**: Make sure Redis is running
```bash
docker-compose ps redis-stack
docker-compose up -d redis-stack
```

### Issue: MCP server not showing in Kiro

**Solution**: Check MCP panel
```
View → MCP Servers
# Should show "redis" server
```

## Success Indicators

You'll know it's working when:

1. ✅ MCP panel shows Redis server as "Connected"
2. ✅ No errors in MCP logs
3. ✅ Queries return data from Redis
4. ✅ Can list log entries
5. ✅ Can get log details

## Example Session

```
You: "Show me all keys matching log_entry:*"
Kiro: [Uses redis_keys tool]
Result: Returns list of 127 log entry keys

You: "Get the first log entry"
Kiro: [Uses redis_hgetall tool]
Result: Returns log details (text, timestamp, label, etc.)

You: "How many logs are there?"
Kiro: [Uses redis_keys and counts]
Result: "There are 127 log entries"
```

## Documentation

- **Full Guide**: `REDIS_MCP_GUIDE.md`
- **Quick Reference**: `QUICK_REFERENCE.md`
- **Vector Search**: `redis-mcp-server/vector_search_helper.py`

---

## Status: ✅ READY

**Next Action**: Reconnect MCP servers in Kiro and start querying!

```
Command Palette → "MCP: Reconnect Servers"
```

Then ask: **"Show me all log entries in Redis"**

🎉 You're all set!
