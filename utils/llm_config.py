import os
import requests
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass

# Model configurations
MODELS = {
    "qwen3-14b": {
        "display_name": "Qwen 3-14B",
        "model_id": "qwen/qwen3-14b",
        "api_type": "local",
        "api_url": "http://127.0.0.1:1234/v1/chat/completions"
    },
    "gpt-5-mini": {
        "display_name": "GPT-5 mini",
        "model_id": "gpt-5-mini",
        "api_type": "openai",
        "api_url": "https://api.openai.com/v1/responses",
        "api_key": os.environ.get("OPENAI_API_KEY")
    }
}


def make_api_call(prompt: str, model_name: str = None, temperature: float = 0.3, 
                  timeout: int = 600, system_message: str = None) -> str:
    """Make LLM API call.
    
    Args:
        prompt: User message
        model_name: Model ID (e.g., "gpt-5-mini", "qwen/qwen3-14b")
        temperature: Sampling temperature
        timeout: Request timeout in seconds
        system_message: System instructions
    
    Returns:
        Model response text
    """
    # Find model config
    config = None
    if model_name:
        for model_config in MODELS.values():
            if model_config["model_id"] == model_name:
                config = model_config
                break
    
    # Default to first model if not found
    if config is None:
        config = list(MODELS.values())[0]
    
    api_type = config["api_type"]
    api_url = config["api_url"]
    api_key = config.get("api_key")
    model_id = config["model_id"]
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # OpenAI API
    if api_type == "openai":
        full_input = f"{system_message}\n\n{prompt}" if system_message else prompt
        payload = {
            "model": model_id,
            "input": full_input,
            "reasoning": {"effort": "medium"},
            "text": {"verbosity": "high"},
            "max_output_tokens": 30000
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        
        # Extract text from response
        if "output" in result:
            for item in result["output"]:
                if item.get("type") == "message":
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            return content["text"]
        
        raise RuntimeError(f"Could not parse OpenAI response: {result.keys()}")
    
    # Local/Standard Chat Completions API
    elif api_type == "local":
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    else:
        raise RuntimeError(f"Unknown API type: {api_type}")
