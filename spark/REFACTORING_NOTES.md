# Spark Job Refactoring Summary

## Changes Made

### 1. Dynamic Element Normalization
Added `normalize_log_message()` function that strips dynamic elements from log messages:

- **IP Addresses**: `10.250.19.102` → `<IP>`
- **Ports**: `:54106` → `:<PORT>`
- **Block IDs**: `blk_-1608999687919862906` → `<BLOCK_ID>`
- **File Sizes**: `size 91178` → `size <SIZE>`
- **Timestamps**: `081109 203518` → `<TIMESTAMP>`
- **File Paths**: `/mnt/hadoop/mapred/system/job.jar` → `<PATH>`
- **Job IDs**: `job_200811092030_0001` → `<JOB_ID>`

This creates a normalized template that allows detection of duplicate log patterns regardless of specific values.

### 2. Intelligent Deduplication
Added `check_log_exists_in_redis()` function that:

- Checks if a normalized log pattern already exists in Redis before generating embeddings
- Uses MD5 hash of normalized message as lookup key
- Stores normalized message markers with 30-day expiration
- Skips embedding generation for duplicate patterns

### 3. Batch Processing Optimization
Modified `foreach_batch_hdfs()` to:

- Normalize all messages in the batch
- Filter out duplicates before calling the embedding service
- Track and report number of skipped duplicates
- Only generate embeddings for new log patterns
- Store both original and normalized text in Redis

### 4. Redis Storage Enhancement
Updated Redis storage to:

- Store normalized text alongside original text for reference
- Create lookup keys for normalized messages (expires in 30 days)
- Use pipeline operations for efficient batch inserts

## Benefits

1. **Cost Reduction**: Avoids generating embeddings for duplicate log patterns
2. **Performance**: Reduces embedding service load and processing time
3. **Storage Efficiency**: Prevents storing redundant embeddings in Redis
4. **Better Pattern Recognition**: Normalized logs help identify similar issues across different nodes/IPs

## Example

**Original Log:**
```
Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
```

**Normalized Log:**
```
Receiving block <BLOCK_ID> src: /<IP>:<PORT> dest: /<IP>:<PORT>
```

If this pattern was already processed, subsequent logs with different IPs/blocks will be skipped.
