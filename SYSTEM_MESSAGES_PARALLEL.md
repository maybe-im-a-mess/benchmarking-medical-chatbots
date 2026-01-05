# System Messages & Parallelization Guide

## Summary of Changes

### 1. System Message Support ✅
**Minimal change**: Added optional `system_message` parameter to `make_api_call()`

**Why this matters:**
- **Better prompt engineering**: Separates instructions (system) from content (user)
- **Cleaner architecture**: System message defines role, user message contains document
- **Model optimization**: Some models (like GPT-4) perform better with proper system messages
- **Reusability**: Same system message can be used across all documents

**How it works:**
```python
# Before (everything in one message)
prompt = "Instructions... Document: " + text
make_api_call(prompt, model_name)

# After (separated)
system_message = "Instructions..."
user_message = f"Document: {text}"
make_api_call(user_message, model_name, system_message=system_message)
```

### 2. Parallelization Support ✅
**Minimal change**: Added `ThreadPoolExecutor` to process multiple extractions concurrently

**Why this matters:**
- **Speed**: API calls are I/O-bound, so threading provides ~4x speedup
- **Efficiency**: While waiting for one API response, other requests are processing
- **Cost-effective**: No additional API costs, just better utilization

**Configuration:**
```python
# In run_extraction.py
PARALLEL = True       # Enable/disable parallel processing
MAX_WORKERS = 4       # Number of concurrent API calls
```

## Best Practices - Answers to Your Questions

### Q1: One Dialogue vs Separate Dialogues?

**Answer: Separate dialogues (one per document)**

**Why:**
- ✅ **Stateless & Clean**: Each extraction is independent
- ✅ **Parallel-friendly**: Can process multiple documents simultaneously
- ✅ **No context pollution**: Previous documents don't affect current one
- ✅ **Error isolation**: One failure doesn't break the entire pipeline
- ✅ **Easier debugging**: Each extraction has its own complete context

**Implementation:**
Each API call creates a fresh conversation:
```
System: "You are a medical extraction specialist..."
User: "Document: [full text]"
```

### Q2: Should System Message Be Reused?

**Answer: YES - One system message per extraction method**

**Why:**
- ✅ **Efficient**: System message is identical across all documents
- ✅ **Consistent**: Ensures same instructions for all extractions
- ✅ **Clear separation**: Instructions don't clutter the document text

**Example** (naive extraction):
```python
# Defined once
SYSTEM_MESSAGE = "You are a medical extraction specialist..."

# Reused for all documents
for document in documents:
    make_api_call(document, system_message=SYSTEM_MESSAGE)
```

### Q3: Parallelization Approach?

**Answer: ThreadPoolExecutor with controlled concurrency**

**Why Threading (not multiprocessing):**
- ✅ **I/O-bound**: API calls wait for network responses (perfect for threads)
- ✅ **Lighter**: Threads share memory, less overhead than processes
- ✅ **Simpler**: No need to serialize/deserialize data between processes

**Why controlled concurrency:**
- ⚠️ **Rate limits**: APIs have rate limits (e.g., OpenAI: 500 requests/minute)
- ⚠️ **Resource limits**: Local model might struggle with too many concurrent requests
- ⚠️ **Memory**: Each request holds document + response in memory

**Recommended MAX_WORKERS:**
- Local models: `2-4` (depends on hardware)
- OpenAI API: `10-20` (depends on tier)
- Google Gemini: `5-10`

## Performance Comparison

### Sequential Processing
```
Document 1, Method 1 → Document 1, Method 2 → Document 1, Method 3 → ...
Total time: 24 extractions × ~300s = ~2 hours
```

### Parallel Processing (4 workers)
```
[Doc1,Method1] [Doc1,Method2] [Doc2,Method1] [Doc2,Method2]  ← Running simultaneously
[Doc1,Method3] [Doc1,Method4] [Doc2,Method3] [Doc2,Method4]  ← Next batch
...
Total time: 24 extractions ÷ 4 workers × ~300s = ~30 minutes
```

**Speedup: ~4x** (with 4 workers)

## Usage Examples

### Basic Usage (Parallel Enabled)
```bash
# Default: parallel processing with 4 workers
python run_extraction.py
```

Output:
```
PARALLEL PROCESSING: ENABLED
Max workers: 4
Submitting 24 extraction tasks...

✓ DRK Geburtshilfe Infos      | naive      |  21 items |  290.5s
✓ Kaiserschnitt                | naive      |  18 items |  310.2s
✓ DRK Geburtshilfe Infos      | atomic     |  15 items |  520.1s
✓ Kaiserschnitt                | cot        |  19 items |  340.8s
...
```

### Sequential Processing (if needed)
```python
# Edit run_extraction.py
PARALLEL = False
```

### Adjust Concurrency
```python
# For local models (conservative)
MAX_WORKERS = 2

# For OpenAI API (aggressive)
MAX_WORKERS = 10
```

## Rate Limit Handling

### OpenAI Rate Limits (Tier 1)
- 500 requests per minute
- 200,000 tokens per minute

**Safe configuration:**
```python
MAX_WORKERS = 8  # 8 workers × ~2 req/min = 16 req/min (well under 500)
```

### Google Gemini Rate Limits
- 60 requests per minute (free tier)

**Safe configuration:**
```python
MAX_WORKERS = 4  # 4 workers × ~2 req/min = 8 req/min (well under 60)
```

### Local Models (No Rate Limits)
Limited by hardware:
```python
MAX_WORKERS = 2-4  # Depends on GPU/CPU capacity
```

## System Message Examples

### Naive Extraction
```python
system_message = """
You are a medical information extraction specialist. 
Extract key discussion points for patient conversations.
Return JSON array: [{"point": "...", "rationale": "..."}]
"""
user_message = f"Document:\n{text}"
```

### Atomic Extraction (Stage 1)
```python
system_message = """
Extract atomic medical facts as bullet points (max 30).
Focus on: risks, contraindications, procedures.
"""
user_message = f"TEXT:\n{text}"
```

### Atomic Extraction (Stage 2)
```python
system_message = """
Group medical facts into discussion points.
Return JSON array (max 25 points).
"""
user_message = f"FACTS:\n{atomic_facts}"
```

## Benefits Summary

| Feature | Before | After | Benefit |
|---------|--------|-------|---------|
| Prompt structure | Mixed | Separated | Better model performance |
| System message | N/A | Reusable | Consistent instructions |
| Processing | Sequential | Parallel | ~4x faster |
| Resource usage | Inefficient | Optimized | Better API utilization |
| Rate limit handling | Manual | Built-in | Configurable concurrency |
| Error handling | Per-document | Per-extraction | Better isolation |

## Troubleshooting

### "Too many concurrent requests"
**Solution:** Reduce `MAX_WORKERS`
```python
MAX_WORKERS = 2  # More conservative
```

### Local model crashes/slows down
**Solution:** Disable parallelization or reduce workers
```python
PARALLEL = False  # or MAX_WORKERS = 1
```

### Mixed results order in logs
**Note:** This is normal with parallel processing. Results complete in any order.
Files are still saved correctly with proper names.

### Memory issues
**Solution:** Reduce concurrency
```python
MAX_WORKERS = 2  # Fewer documents in memory simultaneously
```

## Testing

### Test parallel processing with one model
```python
# In llm_config.py - keep only one model
MODELS = {
    "qwen3-4b": {...}  # Just one for testing
}
```

### Test system messages manually
```python
from utils.llm_config import make_api_call

system = "You are a helpful assistant."
user = "What is 2+2?"

response = make_api_call(user, model_name="gpt-4o-mini", system_message=system)
print(response)
```

### Compare sequential vs parallel timing
```bash
# Run with parallel
python run_extraction.py  # Note the total time

# Edit: PARALLEL = False
python run_extraction.py  # Compare total time
```

## Migration Checklist

- [x] System message parameter added to `make_api_call()`
- [x] All extraction methods updated to use system messages
- [x] Parallelization implemented with ThreadPoolExecutor
- [x] Configurable via `PARALLEL` and `MAX_WORKERS`
- [x] Backward compatible (can disable parallel mode)
- [x] Works with local, OpenAI, and Google APIs

## Recommendations

1. **Start with parallel disabled** to verify system messages work
2. **Enable parallel** once basic functionality confirmed
3. **Start with MAX_WORKERS=2** and increase gradually
4. **Monitor API costs** when using paid models with parallelization
5. **Use sequential mode** for debugging specific extraction issues

