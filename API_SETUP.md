# API Setup Guide - Local & Paid Models

The extraction pipeline now supports local models (LM Studio) and paid API models (OpenAI, Google Gemini) with minimal configuration changes.

## How It Works

All extraction methods now use a universal `make_api_call()` function from `utils/llm_config.py` that automatically handles different API types based on model configuration.

## Adding Models

### Local Models (LM Studio) - Already Configured

```python
"qwen3-4b": {
    "display_name": "Qwen 3-4B",
    "model_id": "qwen/qwen3-4b-2507",
    "description": "Local small model",
    "api_type": "local",
    "api_url": "http://127.0.0.1:1234/v1/chat/completions",
    "api_key": None
}
```

### OpenAI Models

1. **Set your API key as environment variable:**
   ```bash
   export OPENAI_API_KEY="sk-your-key-here"
   ```

2. **Edit `utils/llm_config.py` and uncomment OpenAI models:**
   ```python
   "gpt-4o-mini": {
       "display_name": "GPT-4o Mini",
       "model_id": "gpt-4o-mini",
       "description": "OpenAI efficient model",
       "api_type": "openai",
       "api_url": "https://api.openai.com/v1/chat/completions",
       "api_key": os.environ.get("OPENAI_API_KEY")
   },
   ```

3. **Run extraction:**
   ```bash
   python run_extraction.py
   ```

### Google Gemini Models

1. **Set your API key as environment variable:**
   ```bash
   export GOOGLE_API_KEY="your-google-api-key"
   ```

2. **Edit `utils/llm_config.py` and uncomment Gemini config:**
   ```python
   "gemini-pro": {
       "display_name": "Gemini Pro",
       "model_id": "gemini-pro",
       "description": "Google Gemini Pro",
       "api_type": "google",
       "api_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
       "api_key": os.environ.get("GOOGLE_API_KEY")
   },
   ```

3. **Run extraction:**
   ```bash
   python run_extraction.py
   ```

## Testing Individual Models

Test a model before running the full pipeline:

```bash
# Test local model (default)
python test_extraction.py

# Test OpenAI model
python test_extraction.py "data/raw_md_files/Kaiserschnitt.md" naive gpt-4o-mini

# Test Gemini model
python test_extraction.py "data/raw_md_files/Kaiserschnitt.md" naive gemini-pro
```

## Model Configuration Reference

Each model in `MODELS` dict requires:

| Field | Description | Example |
|-------|-------------|---------|
| `display_name` | Human-readable name | `"GPT-4o Mini"` |
| `model_id` | API identifier | `"gpt-4o-mini"` |
| `description` | Notes/purpose | `"Fast & cheap"` |
| `api_type` | API type | `"local"`, `"openai"`, or `"google"` |
| `api_url` | API endpoint | `"https://api.openai.com/..."` |
| `api_key` | Authentication | `os.environ.get("OPENAI_API_KEY")` |

## Cost Estimation

### OpenAI Pricing (as of Dec 2024)

**GPT-4o Mini:**
- Input: $0.150 per 1M tokens
- Output: $0.600 per 1M tokens

**GPT-4o:**
- Input: $2.50 per 1M tokens
- Output: $10.00 per 1M tokens

**Estimated costs for full pipeline (6 docs × 4 methods = 24 runs):**

Assuming ~5K tokens input, ~1K tokens output per run:
- **GPT-4o Mini**: ~$0.10 total
- **GPT-4o**: ~$1.50 total

### Google Gemini Pricing

**Gemini Pro:**
- Input: $0.000125 per 1K characters
- Output: $0.000375 per 1K characters

**Estimated for full pipeline:** ~$0.05 total

## Environment Variables Best Practices

### Option 1: Shell Export (temporary)
```bash
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
python run_extraction.py
```

### Option 2: .env File (recommended)

1. Create `.env` file in project root:
   ```
   OPENAI_API_KEY=sk-your-key-here
   GOOGLE_API_KEY=your-google-key
   ```

2. Add to `.gitignore`:
   ```
   .env
   ```

3. Install python-dotenv:
   ```bash
   pip install python-dotenv
   ```

4. Load in your code (add to top of `utils/llm_config.py`):
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```

### Option 3: System Environment Variables

Add to `~/.zshrc` or `~/.bashrc`:
```bash
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
```

Then reload: `source ~/.zshrc`

## Mixing Local and Paid Models

You can run both in the same pipeline:

```python
MODELS = {
    "qwen3-4b": {...},        # Local - free, slower
    "gpt-4o-mini": {...},     # OpenAI - cheap, fast
    "gpt-4o": {...},          # OpenAI - expensive, best quality
}
```

Results will be organized by model:
```
data/processed/
├── qwen3-4b/
├── gpt-4o-mini/
└── gpt-4o/
```

## Troubleshooting

### "API key required but not set"
- Check environment variable: `echo $OPENAI_API_KEY`
- Verify it's exported in current shell session
- Try `.env` file approach instead

### "Rate limit exceeded"
- OpenAI: Wait or upgrade to paid tier
- Add delays between requests in `run_extraction.py`

### "Invalid API key"
- Verify key is correct and active
- Check for extra spaces/quotes
- Ensure key has proper permissions

### Google Gemini 400 Error
- Verify API key format
- Check that Gemini API is enabled in Google Cloud Console
- Ensure model_id matches available models

## Adding Other Providers

To add other API providers (Anthropic Claude, Cohere, etc.):

1. Add model config with new `api_type`:
   ```python
   "claude-3": {
       "api_type": "anthropic",
       ...
   }
   ```

2. Update `make_api_call()` in `llm_config.py`:
   ```python
   elif api_type == "anthropic":
       # Add Anthropic API call logic
       ...
   ```

## Backward Compatibility

All changes are backward compatible:
- Existing local model code still works
- `model_name` parameter is optional
- Old extraction calls work without modification
