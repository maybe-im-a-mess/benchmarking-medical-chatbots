import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, use system env vars

# Local model server settings (LM Studio)
LOCAL_MODEL_URL = "http://127.0.0.1:1234/v1/chat/completions"

# All available models
MODELS = {
    "qwen3-4b": {
        "display_name": "Qwen 3-4B",
        "model_id": "qwen/qwen3-4b-2507",
        "api_type": "local",
        "api_url": LOCAL_MODEL_URL,
        "api_key": None
    },
    "gpt-5-mini": {
        "display_name": "GPT-5 mini",
        "model_id": "gpt-5-mini",
        "api_type": "openai_reasoning",
        "api_url": "https://api.openai.com/v1/responses",
        "api_key": os.environ.get("OPENAI_API_KEY") 
    },
    # "gpt-4o": {
    #     "display_name": "GPT-4o",
    #     "model_id": "gpt-4o",
    #     "api_type": "openai",
    #     "api_url": "https://api.openai.com/v1/chat/completions",
    #     "api_key": os.environ.get("OPENAI_API_KEY")
    # },
    # Example: Google Gemini (uncomment and add your API key)
    # "gemini-pro": {
    #     "display_name": "Gemini Pro",
    #     "model_id": "gemini-pro",
    #     "api_type": "google",
    #     "api_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
    #     "api_key": os.environ.get("GOOGLE_API_KEY")
    # },
}

# Default model (backwards compatibility)
MODEL_NAME = "qwen/qwen3-4b-2507"


def get_model_config(model_name: str = None):
    """Get configuration for a specific model by model_id.
    
    Args:
        model_name: The model_id (e.g., "gpt-4o-mini" or "qwen/qwen3-4b-2507")
                   If None, uses MODEL_NAME
    
    Returns:
        dict: Model configuration with api_type, api_url, api_key, etc.
              Falls back to local config if model not found
    """
    if model_name is None:
        model_name = MODEL_NAME
    
    # Try to find model by model_id
    for model_key, config in MODELS.items():
        if config["model_id"] == model_name:
            return config
    
    # Fallback to local model config
    return {
        "model_id": model_name,
        "api_type": "local",
        "api_url": LOCAL_MODEL_URL,
        "api_key": None
    }


def make_api_call(prompt: str, model_name: str = None, temperature: float = 0.3, 
                  timeout: int = 600, system_message: str = None, reasoning_effort: str = "medium"):
    """Universal function for API calls.
    
    Args:
        prompt: The user message (medical document)
        model_name: Model identifier (model_id from config)
        temperature: Sampling temperature
        timeout: Request timeout in seconds (default 600, because the local models are pretty slow)
        system_message: Optional system message with instructions (if None, prompt goes to user message)
    
    Returns:
        str: The model's response text
    
    Raises:
        RuntimeError: If API call fails
    """
    import requests
    
    config = get_model_config(model_name)
    api_type = config["api_type"]
    api_url = config["api_url"]
    api_key = config["api_key"]
    model_id = config["model_id"]
    
    headers = {"Content-Type": "application/json"}
    
    if api_type == "openai_reasoning":
        # GPT-5 Responses API
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Combine system message and user prompt
        full_input = f"{system_message}\n\n{prompt}" if system_message else prompt
        
        payload = {
            "model": model_id,
            "input": full_input,
            "reasoning": {"effort": reasoning_effort},
            "text": {"verbosity": "high"},
            "max_output_tokens": 30000
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        
        # Parse Responses API format: {"output": [items...]}
        if "output" in result and isinstance(result["output"], list):
            # Find the message item in output
            for item in result["output"]:
                if item.get("type") == "message":
                    # Extract text from content array
                    content = item.get("content", [])
                    for content_item in content:
                        if content_item.get("type") == "output_text":
                            return content_item.get("text", "")
        
        raise RuntimeError(f"Could not extract text from GPT-5 response. Response keys: {result.keys() if isinstance(result, dict) else type(result)}")
    
    elif api_type in ["openai", "local", "local_thinking"]:
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # For local_thinking models, add /no_think if reasoning disabled
        user_content = prompt
        if api_type == "local_thinking" and reasoning_effort == "none":
            user_content = f"{prompt}\n/no_think"
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_content})
        
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
        }
        
        try:
            resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"API call failed ({api_type}): {exc}")
    
    elif api_type == "google":
        if not api_key:
            raise RuntimeError("Google API key required but not set")
        
        full_prompt = f"{system_message}\n\n{prompt}" if system_message else prompt
        
        url_with_key = f"{api_url}?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {"temperature": temperature}
        }
        
        try:
            resp = requests.post(url_with_key, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as exc:
            raise RuntimeError(f"Google API call failed: {exc}")
    
    else:
        raise RuntimeError(f"Unknown API type: {api_type}")
