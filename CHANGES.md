# Changes Summary - Information Extraction Pipeline Updates

## Overview
Updated the extraction pipeline to fix timeout issues, support multi-model comparison, and prepare outputs for systematic evaluation.

## Key Changes

### 1. Fixed Timeout Issue in Atomic Extraction
**File**: `information_extraction/atomic_fact_extraction.py`

**Changes**:
- Kept timeout at 600s per LLM call (necessary for local model generation time)
- Optimized Stage 1 prompt: More concise, limited to 30 facts
- Optimized Stage 2 prompt: Clearer instructions, limited to 25 points
- Made prompts more focused on medical context

**Impact**: More efficient extraction while respecting model generation time (~600-1200s total)

### 2. Multi-Model Support
**File**: `utils/llm_config.py`

**Changes**:
- Added `MODELS` dictionary with model configurations:
  ```python
  {
    "model_key": {
      "display_name": "Human-readable name",
      "model_id": "API model identifier",
      "description": "Purpose/notes"
    }
  }
  ```
- Maintained backward compatibility with `MODEL_NAME`

**Impact**: Easy to add new models for comparison (just edit config)

### 3. Standardized Extraction Functions
**Files**: All extraction methods

**Changes**:
- Added `model_name` parameter to all extraction functions:
  - `extract_statements_naive(text, model_name=None)`
  - `extract_statements_atomic(text, model_name=None)`
  - `extract_statements_cot(text, model_name=None)`
  - `extract_statements_schema(text, model_name=None)`
- All methods now accept and pass model_name to API calls
- Default to `MODEL_NAME` if not specified

**Impact**: Consistent API, easy to test different models

### 4. Updated Folder Structure & Output Format
**File**: `run_extraction.py`

**Changes**:
- Output directory structure: `data/processed/{model_name}/`
- Added model metadata to JSON output:
  ```json
  {
    "document": "...",
    "method": "...",
    "model": {
      "key": "qwen3-4b",
      "name": "Qwen 3-4B (2507)"
    },
    "execution_time_seconds": 290.05,
    "item_count": 21,
    "extracted_data": [...]
  }
  ```
- Process **all 6 documents** (not just 2)
- Iterate over all configured models
- Better progress reporting with visual separators

**Impact**: Organized results ready for systematic comparison

### 5. New Comparison Utility
**File**: `utils/compare_results.py` *(NEW)*

**Features**:
- `load_all_results()`: Load all extraction results from processed directory
- `get_summary_statistics()`: Calculate stats by model, method, document
- `print_summary_report()`: Comprehensive console report
- `compare_methods_for_document()`: Focused comparison for specific document
- `export_comparison_csv()`: Export to CSV for external analysis

**Usage**:
```bash
python utils/compare_results.py
```

**Impact**: Easy evaluation of extraction quality and performance

### 6. Test/Debug Script
**File**: `test_extraction.py` *(NEW)*

**Features**:
- Test single document + method + model
- Quick validation without full pipeline
- Shows first 3 results and timing
- Detailed error reporting

**Usage**:
```bash
# Default test
python test_extraction.py

# Custom test
python test_extraction.py "data/raw_md_files/Kaiserschnitt.md" atomic qwen3-4b
```

**Impact**: Fast iteration during development

### 7. Documentation
**File**: `README.md` *(NEW)*

**Content**:
- Project structure overview
- Extraction method descriptions
- Setup and configuration guide
- Usage examples
- Troubleshooting tips
- Thesis phase roadmap

## New Folder Structure

```
data/
├── processed/
│   ├── qwen3-4b/              # Results for Qwen 3-4B
│   │   ├── Document1_naive.json
│   │   ├── Document1_atomic.json
│   │   ├── Document1_cot.json
│   │   └── Document1_uie.json
│   └── llama3-8b/             # Results for other models (if added)
│       └── ...
└── evaluation_results/
    └── comparison.csv         # Exported comparison data
```

## Migration from Old Structure

### Old (Legacy) Outputs
Location: `data/processed/*.json` (flat structure)

These files were from your initial testing. You can:
1. **Keep them**: As baseline for comparison
2. **Archive them**: Move to `data/processed/legacy/`
3. **Delete them**: New runs will regenerate with better structure

### Recommended Action
```bash
# From project root
mkdir -p data/processed/legacy
mv data/processed/*.json data/processed/legacy/ 2>/dev/null || true
```

## Next Steps - How to Use

### 1. Test Single Extraction (Recommended First)
```bash
python test_extraction.py
```
This validates your LM Studio connection without committing to full pipeline.

### 2. Run Full Extraction Pipeline
```bash
python run_extraction.py
```
This processes:
- All 6 documents
- All 4 methods (naive, atomic, cot, uie)
- All configured models (currently just qwen3-4b)

**Expected time**: ~1-2 hours for 1 model

### 3. Analyze Results
```bash
python utils/compare_results.py
```

### 4. Add More Models (Optional)
Edit `utils/llm_config.py`:
```python
MODELS = {
    "qwen3-4b": {...},
    "llama3-8b": {  # Add new model
        "display_name": "Llama 3 8B",
        "model_id": "meta-llama/llama-3-8b",
        "description": "Larger model for comparison"
    }
}
```

Then re-run `python run_extraction.py`

## Performance Improvements

| Metric | Before | After |
|--------|--------|-------|
| Atomic timeout | 600s×2 | 600s×2 (kept - needed for model) |
| Prompt efficiency | Verbose | Optimized with limits |
| Documents processed | 2 | 6 |
| Model comparison | Manual | Automated |
| Output organization | Flat | Hierarchical by model |
| Analysis tools | None | Comparison utilities |

## Backward Compatibility

All existing code remains functional:
- Old imports still work
- Model parameter is optional (defaults to `MODEL_NAME`)
- Legacy flat structure still supported (just not produced)

## Files Modified

1. `information_extraction/atomic_fact_extraction.py` - Timeout fix
2. `information_extraction/naive_llm.py` - Model parameter
3. `information_extraction/cot_extraction.py` - Model parameter
4. `information_extraction/uie.py` - Model parameter
5. `utils/llm_config.py` - Multi-model config
6. `run_extraction.py` - New structure & all docs

## Files Created

1. `utils/compare_results.py` - Analysis utilities
2. `test_extraction.py` - Debug script
3. `README.md` - Documentation
4. `CHANGES.md` - This file

## Testing Checklist

- [ ] LM Studio running on port 1234
- [ ] Correct model loaded in LM Studio
- [ ] Test single extraction: `python test_extraction.py`
- [ ] Run full pipeline: `python run_extraction.py`
- [ ] Check outputs in `data/processed/qwen3-4b/`
- [ ] Run comparison: `python utils/compare_results.py`
- [ ] Verify CSV export in `data/evaluation_results/`

## Known Issues / Limitations

1. **Atomic extraction still slowest**: Two-stage approach inherently takes longer
2. **No parallel processing**: Documents processed sequentially (could parallelize)
3. **Memory usage**: All results held in memory for comparison (fine for current scale)
4. **No caching**: Re-runs regenerate all outputs (intentional for now)

## Future Enhancements (Optional)

1. Add progress bars (tqdm)
2. Implement parallel processing for documents
3. Add retry logic for failed extractions
4. Create visualization dashboard for results
5. Implement evaluation metrics (coverage, accuracy)
6. Add gold standard annotation support
