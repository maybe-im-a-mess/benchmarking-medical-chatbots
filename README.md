# Medical Consent Information Extraction - Master Thesis

This project implements and compares different information extraction approaches for medical consent documents in German (Geburtshilfe/Obstetrics domain).

## Project Structure

```
.
├── data/
│   ├── raw_md_files/          # Input: 6 German medical consent documents
│   ├── processed/             # Output: Extraction results organized by model
│   │   └── {model_name}/      # e.g., qwen3-4b/
│   │       └── {doc}_{method}.json
│   └── evaluation_results/    # Comparison metrics and analysis
├── information_extraction/    # Extraction method implementations
│   ├── naive_llm.py          # Baseline: Direct LLM prompting
│   ├── atomic_fact_extraction.py  # Two-stage atomic fact approach
│   ├── cot_extraction.py     # Chain-of-Thought prompting
│   └── uie.py                # Schema-guided extraction (UIE)
├── utils/
│   ├── llm_config.py         # Model configurations
│   ├── compare_results.py    # Analysis and comparison utilities
│   └── preprocess.py         # (Reserved for future preprocessing)
├── run_extraction.py         # Main pipeline script
└── README.md
```

## Extraction Methods

### 1. Naive LLM (`naive`)
- **Approach**: Direct prompting for discussion points
- **Temperature**: 0.3
- **Output**: `{point, rationale}` pairs
- **Use case**: Baseline for comparison

### 2. Atomic Fact Extraction (`atomic`)
- **Approach**: Two-stage extraction
  - Stage 1: Break text into atomic facts (max 30)
  - Stage 2: Synthesize into discussion points (max 25)
- **Temperature**: 0.2
- **Timeout**: 300s per stage (600s total)
- **Use case**: High granularity with synthesis

### 3. Chain-of-Thought (`cot`)
- **Approach**: Forces reasoning before output
- **Temperature**: 0.2
- **Output**: Thinking process + JSON
- **Use case**: Explainable extraction with reasoning trace

### 4. Schema-Guided (`uie`)
- **Approach**: Type-constrained with predefined schema
- **Categories**: RISK, INSTRUCTION, PREREQUISITE, GENERAL_INFO
- **Priorities**: HIGH, MEDIUM, LOW
- **Temperature**: 0.1 (strictest)
- **Use case**: Structured, categorized extraction

## Setup

### Requirements
- Python 3.8+
- Local LLM server (e.g., LM Studio) running on `http://127.0.0.1:1234`
- Packages: `requests` (install via `pip install requests`)

### Configuration

Edit `utils/llm_config.py` to add/modify models:

```python
MODELS = {
    "qwen3-4b": {
        "display_name": "Qwen 3-4B (2507)",
        "model_id": "qwen/qwen3-4b-2507",
        "description": "Baseline small model"
    },
    # Add more models:
    # "llama3-8b": {
    #     "display_name": "Llama 3 8B",
    #     "model_id": "meta-llama/llama-3-8b",
    #     "description": "Mid-size model"
    # },
}
```

## Usage

### 1. Run Extraction Pipeline

Process all documents with all methods and models:

```bash
python run_extraction.py
```

This will:
- Process all 6 documents in `data/raw_md_files/`
- Apply all 4 extraction methods
- Test all configured models
- Save results to `data/processed/{model_name}/`

Output structure:
```json
{
  "document": "DRK Geburtshilfe Infos",
  "method": "naive",
  "model": {
    "key": "qwen3-4b",
    "name": "Qwen 3-4B (2507)"
  },
  "execution_time_seconds": 290.05,
  "item_count": 21,
  "extracted_data": [
    {
      "point": "Discussion point...",
      "rationale": "Why this matters..."
    }
  ]
}
```

### 2. Analyze Results

Compare extraction methods across models:

```bash
python utils/compare_results.py
```

This will:
- Load all extraction results
- Print summary statistics (by model, method, document)
- Export comparison CSV to `data/evaluation_results/comparison.csv`

Example output:
```
RESULTS BY METHOD (all models):
  NAIVE:
    Total runs: 6
    Successful: 6 | Failed: 0
    Avg items per successful run: 23.5

  ATOMIC:
    Total runs: 6
    Successful: 5 | Failed: 1
    Avg items per successful run: 18.2
```

### 3. Custom Analysis

Programmatic access to results:

```python
from utils.compare_results import load_all_results, compare_methods_for_document

# Load all results
results = load_all_results()

# Compare methods for specific document
compare_methods_for_document(results, "Kaiserschnitt", model_key="qwen3-4b")

# Access raw data
doc_data = results["qwen3-4b"]["Kaiserschnitt"]["naive"]
print(f"Extracted {doc_data['item_count']} items")
```

## Input Documents

Six German obstetrics consent documents:

1. **Äußere Wendung** - External cephalic version
2. **Geburtseinleitung** - Labor induction
3. **Geburtshilfliche Maßnahmen** - Obstetric procedures
4. **Kaiserschnitt** - Cesarean section
5. **Narkose** - Anesthesia
6. **DRK Geburtshilfe Infos** - Birthing facility information

## Output Folder Structure

```
data/processed/
├── qwen3-4b/
│   ├── Äußere Wendung_naive.json
│   ├── Äußere Wendung_atomic.json
│   ├── Äußere Wendung_cot.json
│   ├── Äußere Wendung_uie.json
│   ├── Geburtseinleitung_naive.json
│   └── ... (24 files total: 6 docs × 4 methods)
└── llama3-8b/  # If configured
    └── ... (same structure)
```

## Performance Notes

### Timeout Fix
- Atomic extraction uses 600s per stage (1200s total for two-stage process)
- Prompts optimized for conciseness
- Limits: 30 atomic facts, 25 final points

### Execution Times (approximate, depends on local model speed)
- **Naive**: ~200-600s per document
- **UIE**: ~200-600s per document
- **CoT**: ~250-600s per document
- **Atomic**: ~600-1200s per document (2 stages)

**Total pipeline**: ~2-4 hours for 6 documents × 4 methods × 1 model

## Next Steps (Thesis Phases)

### Phase 1: ✅ Information Extraction (Current)
- [x] Implement 4 extraction methods
- [x] Process all documents
- [x] Support multi-model comparison
- [ ] Build evaluation metrics (coverage, completeness)
- [ ] Create gold standard annotations

### Phase 2: Sender-Side Evaluation
- [ ] Coverage measurement (% topics communicated)
- [ ] Citation accuracy (chunk attribution)
- [ ] Response conciseness metrics

### Phase 3: Recipient-Side Evaluation
- [ ] Patient comprehension testing
- [ ] Synthetic conversation generation
- [ ] LLM patient agent implementation

### Phase 4: Dialogue Management
- [ ] Mandatory question list
- [ ] Context retention system
- [ ] Supervisor node for timing

### Phase 5: Chatbot Implementation
- [ ] RAG system with document retrieval
- [ ] Response generation with citations
- [ ] Full conversation flow

## Troubleshooting

### LM Studio not responding
- Ensure server is running: http://127.0.0.1:1234
- Check model is loaded in LM Studio
- Verify `model_id` in `llm_config.py` matches loaded model

### Timeout errors
- Increase timeout in extraction files (currently 300-600s)
- Use smaller documents for testing
- Check LM Studio performance settings

### Empty results
- Check `data/raw_md_files/` contains `.md` files
- Verify file encoding is UTF-8
- Review error logs in failed extraction JSON files

## License

[Your License Here]

## Contact

[Your Contact Information]
