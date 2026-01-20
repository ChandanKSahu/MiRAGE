
import time
import re
import requests
import logging
import base64
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Callable
import sys
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# ============================================================================
# CONFIGURATION - Lazy loading to allow import without config file
# ============================================================================

# Default values - actual values loaded lazily when needed
_config_initialized = False
BACKEND = os.environ.get("LLM_BACKEND", "GEMINI")
API_URL = ""
LLM_MODEL_NAME = ""
VLM_MODEL_NAME = ""
API_KEY = ""
GEMINI_RPM = int(os.environ.get("GEMINI_RPM", "60"))
GEMINI_BURST = int(os.environ.get("GEMINI_BURST", "15"))
# Log files - default to output/, can be overridden via MIRAGE_OUTPUT_DIR
_default_output = os.environ.get("MIRAGE_OUTPUT_DIR", "output")
LOG_FILE = os.environ.get("LOG_FILE", os.path.join(_default_output, "pipeline.log"))
TERMINAL_LOG_FILE = os.environ.get("TERMINAL_LOG_FILE", os.path.join(_default_output, "terminal_pipeline.log"))
HEADERS = {"Content-Type": "application/json"}
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# ============================================================================
# TOKEN TRACKING - Global counters for input/output tokens
# ============================================================================
import threading as _threading
_token_lock = _threading.Lock()
_token_stats = {
    'total_input_tokens': 0,
    'total_output_tokens': 0,
    'total_calls': 0,
}

def get_token_stats() -> Dict[str, int]:
    """Get current token usage statistics."""
    with _token_lock:
        return _token_stats.copy()

def reset_token_stats():
    """Reset token counters to zero."""
    global _token_stats
    with _token_lock:
        _token_stats = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_calls': 0,
        }

def print_token_stats():
    """Print token usage summary."""
    stats = get_token_stats()
    total = stats['total_input_tokens'] + stats['total_output_tokens']
    print(f"\n{'='*60}")
    print("TOKEN USAGE SUMMARY")
    print(f"{'='*60}")
    print(f"   Total API calls:    {stats['total_calls']:,}")
    print(f"   Input tokens:       {stats['total_input_tokens']:,}")
    print(f"   Output tokens:      {stats['total_output_tokens']:,}")
    print(f"   Total tokens:       {total:,}")
    print(f"{'='*60}\n")

def _update_token_stats(input_tokens: int, output_tokens: int):
    """Thread-safe update of token statistics."""
    with _token_lock:
        _token_stats['total_input_tokens'] += input_tokens
        _token_stats['total_output_tokens'] += output_tokens
        _token_stats['total_calls'] += 1

# Default URLs and Models
_DEFAULT_URLS = {
    "OLLAMA": "http://127.0.0.1:11434/api/chat",
    "GEMINI": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    "OPENAI": "https://api.openai.com/v1/chat/completions"
}

_DEFAULT_MODELS = {
    "OLLAMA": ("llama3.1:8b", "llava:13b"),
    "GEMINI": ("gemini-2.0-flash", "gemini-2.0-flash"),
    "OPENAI": ("gpt-4o-mini", "gpt-4o")
}


def _initialize_config():
    """Initialize configuration lazily on first use.
    
    This is called before any LLM/VLM call to ensure config is loaded.
    Allows the module to be imported without a config file.
    """
    global _config_initialized, BACKEND, API_URL, LLM_MODEL_NAME, VLM_MODEL_NAME
    global API_KEY, GEMINI_RPM, GEMINI_BURST, LOG_FILE, TERMINAL_LOG_FILE, HEADERS
    
    if _config_initialized:
        return
    
    try:
        from mirage.core.config import get_backend_config, get_api_key, get_rate_limit_config, get_paths_config
        
        _backend_cfg = get_backend_config()
        _rate_cfg = get_rate_limit_config()
        _paths_cfg = get_paths_config()
        
        BACKEND = _backend_cfg['name']
        API_URL = _backend_cfg.get('url', _DEFAULT_URLS.get(BACKEND, ''))
        LLM_MODEL_NAME = _backend_cfg.get('llm_model', _DEFAULT_MODELS.get(BACKEND, ('', ''))[0])
        VLM_MODEL_NAME = _backend_cfg.get('vlm_model', _DEFAULT_MODELS.get(BACKEND, ('', ''))[1])
        API_KEY = get_api_key()
        
        # Fallback to environment variables if API key not in config
        if not API_KEY:
            if BACKEND == "GEMINI":
                API_KEY = os.environ.get("GEMINI_API_KEY", "")
            elif BACKEND == "OPENAI":
                API_KEY = os.environ.get("OPENAI_API_KEY", "")
        
        # Rate limiting from config
        GEMINI_RPM = _rate_cfg.get('requests_per_minute', 60)
        GEMINI_BURST = _rate_cfg.get('burst_size', 15)
        
        # Auto-generate log file names from dataset name and LLM model
        _output_dir = _paths_cfg.get('output_dir', 'output')
        _input_pdf_dir = _paths_cfg.get('input_pdf_dir', 'data/documents')
        _dataset_name = Path(_input_pdf_dir).name
        _log_basename = f"{_dataset_name}_{LLM_MODEL_NAME}.log"
        
        LOG_FILE = os.path.join(_output_dir, _log_basename)
        TERMINAL_LOG_FILE = os.path.join(_output_dir, f"terminal_{_log_basename}")
        
    except Exception:
        # Use environment variables and defaults
        BACKEND = os.environ.get("LLM_BACKEND", "GEMINI")
        API_URL = _DEFAULT_URLS.get(BACKEND, _DEFAULT_URLS["GEMINI"])
        LLM_MODEL_NAME, VLM_MODEL_NAME = _DEFAULT_MODELS.get(BACKEND, _DEFAULT_MODELS["GEMINI"])
        
        # Try to load API key from environment
        API_KEY = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    
    # Set headers based on backend
    if BACKEND == "GEMINI":
        HEADERS = {"Content-Type": "application/json"}
    elif BACKEND == "OLLAMA":
        HEADERS = {"Content-Type": "application/json"}
    else:
        HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    # Warn if API key is missing for backends that require it
    if not API_KEY and BACKEND in ("GEMINI", "OPENAI"):
        print(f"⚠️ Warning: No API key found for {BACKEND}. Set {BACKEND}_API_KEY environment variable or configure api_key_path in config.yaml")
    
    _config_initialized = True


def test_llm_connection() -> bool:
    """Test LLM API connection."""
    _initialize_config()
    print(f"Testing LLM connection to {BACKEND}...")
    try:
        response = call_llm_simple("Say 'Hello' in one word.")
        print(f"LLM connection successful: {response[:50]}...")
        return True
    except Exception as e:
        print(f"LLM connection failed: {e}")
        return False

# ============================================================================
# CORE UTILITY FUNCTIONS
# ============================================================================

class TeeOutput:
    """Tee output to both console and file"""
    def __init__(self, file_path, stream):
        self.file = open(file_path, 'a', encoding='utf-8')
        self.stream = stream
        self.encoding = getattr(stream, 'encoding', 'utf-8')
    
    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()
    
    def flush(self):
        self.stream.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()


def setup_logging(enable_terminal_log=True):
    """Setup logging to file and console, optionally capture all terminal output"""
    # Create logs directory if it doesn't exist
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(TERMINAL_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Capture all terminal output (stdout and stderr) to terminal log file
    if enable_terminal_log:
        # Clear the terminal log file at start
        with open(TERMINAL_LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"=== Terminal Log Started ===\n")
        
        sys.stdout = TeeOutput(TERMINAL_LOG_FILE, sys.__stdout__)
        sys.stderr = TeeOutput(TERMINAL_LOG_FILE, sys.__stderr__)
        print(f"📝 Terminal output being captured to: {TERMINAL_LOG_FILE}")

def get_image_mime_type(image_path: str) -> str:
    """Get MIME type based on file extension"""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg', 
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp'
    }
    return mime_types.get(ext, 'image/png')  # Default to PNG if unknown

def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# ============================================================================
# VLM INTERACTION FUNCTIONS
# ============================================================================

def test_vlm_connection(test_image: str = None) -> bool:
    """Test VLM API connection with a sample image
    
    Args:
        test_image: Path to a test image file. If None, returns True (skips image test).
    """
    print("🔍 Testing VLM API connection...")
    try:
        if test_image is None:
            print("⚠️ No test image provided, skipping VLM image test")
            return True
        
        if not Path(test_image).exists():
            print(f"❌ Test image not found: {test_image}")
            return False
            
        call_vlm_simple("Describe this image briefly.", test_image)
        print("✅ VLM API connection successful!")
        return True
    except Exception as e:
        print(f"❌ VLM API connection error: {e}")
        return False

def call_llm_simple(prompt: str, max_retries: int = 5, use_cache: bool = True) -> str:
    """Simple LLM call with text-only input. Supports OLLAMA, GEMINI, OPENAI.
    
    Args:
        prompt: Text prompt to send to the LLM
        max_retries: Maximum number of retry attempts (default: 5)
        use_cache: Whether to use LLM response cache (default: True)
    
    Returns:
        LLM response text
        
    Raises:
        Exception: If all retry attempts fail
    """
    _initialize_config()
    
    # Check cache first
    if use_cache:
        try:
            from mirage.utils.llm_cache import get_llm_cache
            cache = get_llm_cache()
            if cache:
                cached = cache.get('llm_simple', prompt, model=LLM_MODEL_NAME)
                if cached is not None:
                    print(f"📦 Cache hit for LLM call ({len(cached)} chars)")
                    return cached
        except ImportError:
            pass
    
    print(f"Calling LLM (text-only) via {BACKEND}...")
    attempt = 0
    wait_time = 2
    
    # Log LLM request
    content_preview = f"Content: {prompt[:50]}...{prompt[-50:]}" if len(prompt) > 100 else f"Content: {prompt}"
    logging.info(f"LLM Request [{BACKEND}] - {content_preview}")
    
    while attempt < max_retries:
        attempt += 1
        try:
            if BACKEND == "OLLAMA":
                # Ollama API format
                data = {
                    "model": LLM_MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.0}
                }
                response = requests.post(API_URL, json=data, timeout=300)
                if response.status_code == 200:
                    resp_json = response.json()
                    result = resp_json["message"]["content"]
                    # Track token usage (Ollama format)
                    _update_token_stats(
                        resp_json.get("prompt_eval_count", 0),
                        resp_json.get("eval_count", 0)
                    )
                else:
                    raise Exception(f"HTTP {response.status_code}")
                    
            elif BACKEND == "GEMINI":
                # Google Gemini direct API format
                url = GEMINI_URL.format(model=LLM_MODEL_NAME) + f"?key={API_KEY}"
                data = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.0}
                }
                response = requests.post(url, headers=HEADERS, json=data, timeout=300)
                if response.status_code == 200:
                    resp_json = response.json()
                    result = resp_json["candidates"][0]["content"]["parts"][0]["text"]
                    # Track token usage
                    usage = resp_json.get("usageMetadata", {})
                    _update_token_stats(
                        usage.get("promptTokenCount", 0),
                        usage.get("candidatesTokenCount", 0)
                    )
                else:
                    error_msg = response.text[:200] if response.text else f"HTTP {response.status_code}"
                    raise Exception(error_msg)
                    
            elif BACKEND == "OPENAI":
                # OpenAI API format
                data = {
                    "model": LLM_MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0
                }
                response = requests.post(API_URL, headers=HEADERS, json=data, timeout=300)
                if response.status_code == 200:
                    resp_json = response.json()
                    result = resp_json["choices"][0]["message"]["content"]
                    # Track token usage
                    usage = resp_json.get("usage", {})
                    _update_token_stats(
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0)
                    )
                else:
                    raise Exception(f"HTTP {response.status_code}")
                    
            else:  # OpenAI-compatible API
                data = {
                    "model": LLM_MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}]
                }
                response = requests.post(API_URL, headers=HEADERS, json=data, timeout=300)
                if response.status_code == 200:
                    response_data = response.json()
                    result = response_data["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"HTTP {response.status_code}")
            
            print(f"✅ LLM response received ({len(result)} chars)")
            logging.info(f"LLM Response (Complete) - {len(result)} chars")
            logging.info(f"LLM Response Content: {result}")
            logging.info("-" * 60)
            
            # Cache the response
            if use_cache:
                try:
                    from mirage.utils.llm_cache import get_llm_cache
                    cache = get_llm_cache()
                    if cache:
                        cache.set('llm_simple', prompt, result, model=LLM_MODEL_NAME)
                except ImportError:
                    pass
            
            return result
                
        except Exception as e:
            print(f"⚠️ LLM call error (attempt {attempt}/{max_retries}): {e}")
            
            if attempt >= max_retries:
                print(f"❌ LLM call failed after {max_retries} attempts. Giving up.")
                raise Exception(f"LLM call failed after {max_retries} attempts: {e}")
        
        # Wait with exponential backoff (capped at 60 seconds)
        print(f"   Waiting {wait_time}s before retry...")
        time.sleep(wait_time)
        wait_time = min(wait_time * 2, 60)
    
    # Should never reach here, but just in case
    raise Exception(f"LLM call failed after {max_retries} attempts")

def call_vlm_simple(prompt: str, image_path: str) -> str:
    """Simple VLM call with single image. Supports OLLAMA, GEMINI, OPENAI."""
    _initialize_config()
    print(f"Calling VLM (simple) via {BACKEND}...")
    attempt = 0
    wait_time = 2
    
    # Log VLM request
    content_preview = f"Content: {prompt[:50]}...{prompt[-50:]}" if len(prompt) > 100 else f"Content: {prompt}"
    logging.info(f"VLM Request [{BACKEND}] - Image: {image_path}, {content_preview}")
    
    while True:
        attempt += 1
        try:
            base64_image = encode_image_to_base64(image_path)
            mime_type = get_image_mime_type(image_path)
            
            if BACKEND == "OLLAMA":
                # Ollama VLM format - images as base64 list
                data = {
                    "model": VLM_MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt, "images": [base64_image]}],
                    "stream": False,
                    "options": {"temperature": 0.0}
                }
                response = requests.post(API_URL, json=data, timeout=300)
                if response.status_code == 200:
                    result = response.json()["message"]["content"]
                else:
                    raise Exception(f"HTTP {response.status_code}")
                    
            elif BACKEND == "GEMINI":
                # Google Gemini direct API format with inline image
                url = GEMINI_URL.format(model=VLM_MODEL_NAME) + f"?key={API_KEY}"
                data = {
                    "contents": [{
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": mime_type, "data": base64_image}}
                        ]
                    }],
                    "generationConfig": {"temperature": 0.0}
                }
                response = requests.post(url, headers=HEADERS, json=data, timeout=300)
                if response.status_code == 200:
                    resp_json = response.json()
                    result = resp_json["candidates"][0]["content"]["parts"][0]["text"]
                    # Track token usage
                    usage = resp_json.get("usageMetadata", {})
                    _update_token_stats(
                        usage.get("promptTokenCount", 0),
                        usage.get("candidatesTokenCount", 0)
                    )
                else:
                    error_msg = response.text[:200] if response.text else f"HTTP {response.status_code}"
                    raise Exception(error_msg)
                    
            elif BACKEND == "OPENAI":
                # OpenAI Vision API format
                image_url = f"data:{mime_type};base64,{base64_image}"
                data = {
                    "model": VLM_MODEL_NAME,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }],
                    "temperature": 0.0
                }
                response = requests.post(API_URL, headers=HEADERS, json=data, timeout=300)
                if response.status_code == 200:
                    result = response.json()["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"HTTP {response.status_code}")
                    
            else:  # OpenAI-compatible API
                image_url = f"data:{mime_type};base64,{base64_image}"
                data = {
                    "model": VLM_MODEL_NAME,
                    "messages": [{
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }]
                }
                response = requests.post(API_URL, headers=HEADERS, json=data, timeout=300)
                if response.status_code == 200:
                    result = response.json()["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"HTTP {response.status_code}")
            
            print(f"✅ VLM response received ({len(result)} chars)")
            logging.info(f"VLM Response (Complete) - Image: {image_path} - {len(result)} chars")
            logging.info(f"VLM Response Content: {result}")
            logging.info("-" * 60)
            return result
                
        except Exception as e:
            print(f"⚠️ VLM call error (attempt {attempt}): {e}")
        
        if attempt >= 3:
            print(f"❌ VLM call failed after {attempt} attempts. Giving up.")
            raise Exception(f"VLM call failed after {attempt} attempts")
            
        print(f"   Waiting {wait_time}s before retry...")
        time.sleep(wait_time)
        wait_time = min(wait_time * 2, 60)

def call_vlm_with_examples(prompt: str, query_image_path: str, example_image_paths: List[str]) -> str:
    """VLM call with multiple example images and query image"""
    _initialize_config()
    print(f"Calling VLM with examples via {BACKEND}...")
    attempt = 0
    wait_time = 2
    
    logging.info(f"VLM Request [{BACKEND}] - Query Image: {query_image_path}")
    
    while True:
        attempt += 1
        try:
            if BACKEND == "OLLAMA":
                # Ollama: use message chaining for multiple images
                messages = []
                for i, example_path in enumerate(example_image_paths):
                    if Path(example_path).exists():
                        messages.append({
                            "role": "user",
                            "content": f"Example {i+1}",
                            "images": [encode_image_to_base64(example_path)]
                        })
                # Add query image
                messages.append({
                    "role": "user",
                    "content": f"Query image:\n{prompt}",
                    "images": [encode_image_to_base64(query_image_path)]
                })
                data = {
                    "model": VLM_MODEL_NAME,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.0}
                }
                response = requests.post(API_URL, json=data, timeout=300)
                if response.status_code == 200:
                    result = response.json()["message"]["content"]
                else:
                    raise Exception(f"HTTP {response.status_code}")
            else:
                # OpenAI-compatible format
                content = [{"type": "text", "text": prompt}]
                for example_path in example_image_paths:
                    if Path(example_path).exists():
                        base64_image = encode_image_to_base64(example_path)
                        if "qwen" in VLM_MODEL_NAME.lower():
                            image_url = f"data:{get_image_mime_type(example_path)};base64,{base64_image}"
                        else:
                            image_url = f"data:image/png;base64,{base64_image}"
                        content.append({"type": "image_url", "image_url": {"url": image_url}})
                
                base64_query = encode_image_to_base64(query_image_path)
                if "qwen" in VLM_MODEL_NAME.lower():
                    query_url = f"data:{get_image_mime_type(query_image_path)};base64,{base64_query}"
                else:
                    query_url = f"data:image/png;base64,{base64_query}"
                content.append({"type": "image_url", "image_url": {"url": query_url}})
                
                data = {"model": VLM_MODEL_NAME, "messages": [{"role": "user", "content": content}]}
                response = requests.post(API_URL, headers=HEADERS, json=data, timeout=300)
                if response.status_code == 200:
                    result = response.json()["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"HTTP {response.status_code}")
            
            print(f"✅ VLM response received ({len(result)} chars)")
            logging.info(f"VLM Response - Query Image: {query_image_path}")
            logging.info(f"VLM Response Content: {result}")
            logging.info("-" * 60)
            return result
                
        except Exception as e:
            print(f"⚠️ VLM call error (attempt {attempt}): {e}")
        
        if attempt >= 3:
            print(f"❌ VLM call failed after {attempt} attempts. Giving up.")
            raise Exception(f"VLM call failed after {attempt} attempts")
            
        print(f"   Waiting {wait_time}s before retry...")
        time.sleep(wait_time)
        wait_time = min(wait_time * 2, 60)

def call_vlm_with_multiple_images(prompt: str, image_paths: List[str]) -> str:
    """VLM call with multiple images for reranking"""
    _initialize_config()
    print(f"Calling VLM with {len(image_paths)} images via {BACKEND}...")
    attempt = 0
    wait_time = 2
    
    logging.info(f"VLM Request [{BACKEND}] - {len(image_paths)} images")
    
    while True:
        attempt += 1
        try:
            if BACKEND == "OLLAMA":
                # Ollama: message chaining for multiple images
                messages = []
                for i, image_path in enumerate(image_paths):
                    if Path(image_path).exists():
                        messages.append({
                            "role": "user",
                            "content": f"Image {i+1}",
                            "images": [encode_image_to_base64(image_path)]
                        })
                messages.append({"role": "user", "content": prompt})
                
                data = {
                    "model": VLM_MODEL_NAME,
                    "messages": messages,
                    "stream": False,
                    "options": {"num_ctx": 16384, "temperature": 0.0}
                }
                response = requests.post(API_URL, json=data, timeout=300)
                
                if response.status_code == 200:
                    result = response.json()["message"]["content"]
                elif 500 <= response.status_code < 600 or response.status_code == 429:
                    print(f"⚠️ Server error ({response.status_code}). Retrying...")
                    # Note: Do NOT decrement attempt - rate limits count against max retries
                    raise Exception(f"Server error {response.status_code}")
                else:
                    raise Exception(f"HTTP {response.status_code}")
                    
            elif BACKEND == "GEMINI":
                # Google Gemini API format with inline images
                url = GEMINI_URL.format(model=VLM_MODEL_NAME) + f"?key={API_KEY}"
                parts = [{"text": prompt}]
                for image_path in image_paths:
                    if Path(image_path).exists():
                        base64_image = encode_image_to_base64(image_path)
                        mime_type = get_image_mime_type(image_path)
                        parts.append({"inline_data": {"mime_type": mime_type, "data": base64_image}})
                
                data = {
                    "contents": [{"parts": parts}],
                    "generationConfig": {"temperature": 0.0}
                }
                response = requests.post(url, headers=HEADERS, json=data, timeout=300)
                
                if response.status_code == 200:
                    resp_json = response.json()
                    result = resp_json["candidates"][0]["content"]["parts"][0]["text"]
                    # Track token usage
                    usage = resp_json.get("usageMetadata", {})
                    _update_token_stats(
                        usage.get("promptTokenCount", 0),
                        usage.get("candidatesTokenCount", 0)
                    )
                elif 500 <= response.status_code < 600 or response.status_code == 429:
                    print(f"⚠️ Server error ({response.status_code}). Retrying...")
                    # Note: Do NOT decrement attempt - rate limits count against max retries
                    raise Exception(f"Server error {response.status_code}")
                else:
                    error_msg = response.text[:200] if response.text else f"HTTP {response.status_code}"
                    raise Exception(error_msg)
                    
            else:
                # OpenAI-compatible format
                content = [{"type": "text", "text": prompt}]
                for image_path in image_paths:
                    if Path(image_path).exists():
                        base64_image = encode_image_to_base64(image_path)
                        if "qwen" in VLM_MODEL_NAME.lower():
                            image_url = f"data:{get_image_mime_type(image_path)};base64,{base64_image}"
                        else:
                            image_url = f"data:image/png;base64,{base64_image}"
                        content.append({"type": "image_url", "image_url": {"url": image_url}})
                
                data = {"model": VLM_MODEL_NAME, "messages": [{"role": "user", "content": content}]}
                local_headers = HEADERS.copy()
                local_headers["Connection"] = "close"
                response = requests.post(API_URL, headers=local_headers, json=data, timeout=300)
                
                if response.status_code == 200:
                    result = response.json()["choices"][0]["message"]["content"]
                elif 500 <= response.status_code < 600 or response.status_code == 429:
                    print(f"⚠️ Server error ({response.status_code}). Retrying...")
                    # Note: Do NOT decrement attempt - rate limits count against max retries
                    raise Exception(f"Server error {response.status_code}")
                else:
                    raise Exception(f"HTTP {response.status_code}")
            
            print(f"✅ VLM response received ({len(result)} chars)")
            logging.info(f"VLM Response - {len(image_paths)} images")
            logging.info(f"VLM Response Content: {result}")
            logging.info("-" * 60)
            return result
                
        except Exception as e:
            print(f"⚠️ VLM call error (attempt {attempt}): {e}")
        
        if attempt >= 3:
            print(f"❌ VLM call failed after {attempt} attempts. Giving up.")
            raise Exception(f"VLM call failed after {attempt} attempts")
            
        # Wait with exponential backoff (capped at 60 seconds)
        print(f"   Waiting {wait_time}s before retry...")
        time.sleep(wait_time)
        wait_time = min(wait_time * 2, 60)

def call_vlm_multi_images_ollama(prompt: str, image_paths: List[str]) -> str:
    """
    Calls local Ollama with multiple images and a prompt.
    Uses message chaining strategy for correct image ordering in context.
    """
    print(f"👁️ Calling VLM with {len(image_paths)} images (Ollama)...")
    
    url = "http://127.0.0.1:11434/api/chat"
    messages = []
    
    # Add images as separate user messages
    for i, image_path in enumerate(image_paths):
        if Path(image_path).exists():
            try:
                base64_img = encode_image_to_base64(image_path)
                messages.append({
                    "role": "user",
                    "content": f"Image {i+1}",
                    "images": [base64_img]
                })
            except Exception as e:
                print(f"⚠️ Failed to encode image {image_path}: {e}")
        else:
             print(f"⚠️ Image not found: {image_path}")
    
    # Add the actual prompt as the final message
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    payload = {
        "model": "qwen3-vl:32b",
        "messages": messages,
        "options": {
            "num_ctx": 16384,
            "temperature": 0.0
        },
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()['message']['content']
        print(f"✅ VLM response received ({len(result)} chars)")
        return result
    except Exception as e:
        print(f"❌ Ollama call failed: {e}")
        if 'response' in locals():
            print(f"Response: {response.text}")
        return ""

def call_vlm_interweaved(prompt: str, chunks: List[Dict], use_cache: bool = True) -> str:
    """VLM call with interleaved text and images from chunks.
    Supports OLLAMA, GEMINI, OPENAI.
    
    Args:
        prompt: System prompt / instruction
        chunks: List of dicts - supports two formats:
               1. Old format: {'content': str, 'image_path': str|None}
               2. JSON format: {'chunk_type': str, 'content': str, 'artifact': str}
        use_cache: Whether to use LLM response cache (default: True)
    """
    _initialize_config()
    
    # Check cache first
    if use_cache:
        try:
            from mirage.utils.llm_cache import get_llm_cache
            cache = get_llm_cache()
            if cache:
                cached = cache.get('vlm_interweaved', prompt, chunks, model=VLM_MODEL_NAME)
                if cached is not None:
                    print(f"📦 Cache hit for VLM call ({len(cached)} chars)")
                    return cached
        except ImportError:
            pass
    
    print(f"Calling VLM with {len(chunks)} chunks (interweaved) via {BACKEND}...")
    attempt = 0
    wait_time = 2
    
    logging.info(f"VLM Request [{BACKEND}] - {len(chunks)} chunks")
    
    def _extract_image_path(chunk):
        """Helper to extract image path from chunk
        
        Supports multiple formats:
        1. New format: {'artifact': [list of paths]} - uses first image
        2. Old format: {'image_path': str} - backward compatibility
        3. JSON format: {'chunk_type': str, 'artifact': str} - legacy format
        """
        # Check for artifact list (new format)
        artifact = chunk.get('artifact', [])
        if isinstance(artifact, list) and len(artifact) > 0:
            return artifact[0]  # Use first image
        
        # Fallback: check image_path field (backward compatibility)
        image_path = chunk.get('image_path')
        if image_path:
            return image_path
        
        # Fallback: legacy format with chunk_type and artifact string
        if 'chunk_type' in chunk:
            chunk_type = chunk.get('chunk_type', '')
            artifact_str = chunk.get('artifact', 'None')
            if chunk_type in ['standalone image', 'image'] and artifact_str != 'None':
                match = re.search(r'!\[Image\]\(([^)]+)\)', artifact_str)
                if match:
                    return match.group(1)
        
        return None
    
    while True:
        attempt += 1
        try:
            if BACKEND == "OLLAMA":
                # Ollama: message chaining for interleaved content
                messages = [{"role": "user", "content": prompt}]
                for i, chunk in enumerate(chunks):
                    chunk_text = chunk.get('content', '')
                    image_path = _extract_image_path(chunk)
                    msg = {"role": "user", "content": f"CHUNK{i+1}: {chunk_text}"}
                    if image_path and Path(image_path).exists():
                        msg["images"] = [encode_image_to_base64(image_path)]
                    messages.append(msg)
                
                data = {
                    "model": VLM_MODEL_NAME,
                    "messages": messages,
                    "stream": False,
                    "options": {"num_ctx": 16384, "temperature": 0.0}
                }
                response = requests.post(API_URL, json=data, timeout=300)
                if response.status_code == 200:
                    result = response.json()["message"]["content"]
                elif 500 <= response.status_code < 600 or response.status_code == 429:
                    # Note: Do NOT decrement attempt - rate limits count against max retries
                    raise Exception(f"Server error {response.status_code}")
                else:
                    raise Exception(f"HTTP {response.status_code}")
                    
            elif BACKEND == "GEMINI":
                # Google Gemini: parts array with text and inline_data
                parts = [{"text": prompt}]
                for i, chunk in enumerate(chunks):
                    chunk_text = chunk.get('content', '')
                    image_path = _extract_image_path(chunk)
                    
                    parts.append({"text": f"\n\nCHUNK {i+1}:\n{chunk_text}"})
                    
                    if image_path and Path(image_path).exists():
                        base64_image = encode_image_to_base64(image_path)
                        mime_type = get_image_mime_type(image_path)
                        parts.append({"inline_data": {"mime_type": mime_type, "data": base64_image}})
                
                url = GEMINI_URL.format(model=VLM_MODEL_NAME) + f"?key={API_KEY}"
                data = {
                    "contents": [{"parts": parts}],
                    "generationConfig": {"temperature": 0.0}
                }
                response = requests.post(url, headers=HEADERS, json=data, timeout=300)
                if response.status_code == 200:
                    resp_json = response.json()
                    result = resp_json["candidates"][0]["content"]["parts"][0]["text"]
                    # Track token usage
                    usage = resp_json.get("usageMetadata", {})
                    _update_token_stats(
                        usage.get("promptTokenCount", 0),
                        usage.get("candidatesTokenCount", 0)
                    )
                elif 500 <= response.status_code < 600 or response.status_code == 429:
                    # Note: Do NOT decrement attempt - rate limits count against max retries
                    raise Exception(f"Server error {response.status_code}")
                else:
                    error_msg = response.text[:200] if response.text else f"HTTP {response.status_code}"
                    raise Exception(error_msg)
                    
            elif BACKEND == "OPENAI":
                # OpenAI: content array with text and image_url types
                content = [{"type": "text", "text": prompt}]
                for i, chunk in enumerate(chunks):
                    chunk_text = chunk.get('content', '')
                    image_path = _extract_image_path(chunk)
                    
                    content.append({"type": "text", "text": f"\n\nCHUNK {i+1}:\n{chunk_text}"})
                    
                    if image_path and Path(image_path).exists():
                        base64_image = encode_image_to_base64(image_path)
                        mime_type = get_image_mime_type(image_path)
                        image_url = f"data:{mime_type};base64,{base64_image}"
                        content.append({"type": "image_url", "image_url": {"url": image_url}})
                
                data = {
                    "model": VLM_MODEL_NAME,
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0.0
                }
                response = requests.post(API_URL, headers=HEADERS, json=data, timeout=300)
                if response.status_code == 200:
                    result = response.json()["choices"][0]["message"]["content"]
                elif 500 <= response.status_code < 600 or response.status_code == 429:
                    # Note: Do NOT decrement attempt - rate limits count against max retries
                    raise Exception(f"Server error {response.status_code}")
                else:
                    raise Exception(f"HTTP {response.status_code}")
                    
            else:  # OpenAI-compatible API
                content = [{"type": "text", "text": prompt}]
                for i, chunk in enumerate(chunks):
                    chunk_text = chunk.get('content', '')
                    image_path = _extract_image_path(chunk)
                    
                    text_block = f"\n\n<|#|>CHUNK{i+1}<|#|>START<|#|>\n{chunk_text}\n"
                    
                    if image_path and Path(image_path).exists():
                        text_block += "<|#|>Image<|#|>"
                        content.append({"type": "text", "text": text_block})
                        base64_image = encode_image_to_base64(image_path)
                        mime_type = get_image_mime_type(image_path)
                        image_url = f"data:{mime_type};base64,{base64_image}"
                        content.append({"type": "image_url", "image_url": {"url": image_url}})
                        content.append({"type": "text", "text": f"<|#|>CHUNK{i+1}<|#|>END<|#|>"})
                    else:
                        text_block += f"<|#|>Image<|#|>None<|#|>CHUNK{i+1}<|#|>END<|#|>"
                        content.append({"type": "text", "text": text_block})
                
                data = {"model": VLM_MODEL_NAME, "messages": [{"role": "user", "content": content}]}
                local_headers = {**HEADERS, "Connection": "close"}
                response = requests.post(API_URL, headers=local_headers, json=data, timeout=300)
                if response.status_code == 200:
                    result = response.json()["choices"][0]["message"]["content"]
                elif 500 <= response.status_code < 600 or response.status_code == 429:
                    # Note: Do NOT decrement attempt - rate limits count against max retries
                    raise Exception(f"Server error {response.status_code}")
                else:
                    raise Exception(f"HTTP {response.status_code}")
            
            print(f"✅ VLM response received ({len(result)} chars)")
            logging.info(f"VLM Response - {len(chunks)} chunks")
            logging.info(f"VLM Response Content: {result}")
            logging.info("-" * 60)
            
            # Cache the response
            if use_cache:
                try:
                    from mirage.utils.llm_cache import get_llm_cache
                    cache = get_llm_cache()
                    if cache:
                        cache.set('vlm_interweaved', prompt, result, chunks, model=VLM_MODEL_NAME)
                except ImportError:
                    pass
            
            return result
                
        except Exception as e:
            print(f"⚠️ VLM call error (attempt {attempt}): {e}")
        
        if attempt >= 3:
            print(f"❌ VLM call failed after {attempt} attempts. Giving up.")
            raise Exception(f"VLM call failed after {attempt} attempts")
            
        print(f"   Waiting {wait_time}s before retry...")
        time.sleep(wait_time)
        wait_time = min(wait_time * 2, 60)

# def extract_role_context(image_path: str) -> Tuple[str, str, str]:
#     """Extract figure description, expert role, and figure category from image"""
#     print("🔍 Extracting role context from image...")
    
#     # Use the role_context prompt with example images
#     example_paths = [fie_loc1, fie_loc2, fie_loc3, fie_loc4]
#     prompt = PROMPTS_IMAGE["role_context"]
    
#     response = call_vlm_with_examples(prompt, image_path, example_paths)
    
#     # Parse structured response
#     try:
#         # Extract figure_description, expert_role, and figure_category from response
#         # Try multiple patterns for flexibility
#         figure_description_match = re.search(r'figure_description:\s*"([^"]*)"', response)
#         if not figure_description_match:
#             figure_description_match = re.search(r'figure_description:\s*(.*?)(?=\nexpert_role:|$)', response, re.DOTALL)
        
#         expert_role_match = re.search(r'expert_role:\s*"([^"]*)"', response)
#         if not expert_role_match:
#             expert_role_match = re.search(r'expert_role:\s*(.*?)(?=\nfigure_category:|$)', response)
        
#         figure_category_match = re.search(r'figure_category:\s*"([^"]*)"', response)
#         if not figure_category_match:
#             figure_category_match = re.search(r'figure_category:\s*(.*?)(?=\n|$)', response)
        
#         if figure_description_match and expert_role_match and figure_category_match:
#             figure_description = figure_description_match.group(1).strip()
#             expert_role = expert_role_match.group(1).strip()
#             figure_category = figure_category_match.group(1).strip()
#             return figure_description, expert_role, figure_category
#         else:
#             print("⚠️ Could not parse structured response, returning raw response")
#             print(f"Response: {response}")
#             return response, "Technical expert", "Other"
#     except Exception as e:
#         print(f"⚠️ Error parsing response: {e}")
#         return response, "Technical expert", "Other"

# def generate_qa_pair(image_path: str, figure_description: str, expert_role: str) -> Tuple[str, str]:
#     """Generate Q&A pair using image, description, and role"""
#     print("❓ Generating Q&A pair...")
    
#     # Use the qa_single_artifact prompt with example images
#     example_paths = [fie_loc1, fie_loc2, fie_loc3, fie_loc4]
    
#     # Replace the placeholders in the prompt before formatting
#     prompt_template = PROMPTS_IMAGE["qa_single_artifact"]
#     prompt_template = prompt_template.replace("{fie_loc1}", fie_loc1)
#     prompt_template = prompt_template.replace("{fie_loc2}", fie_loc2)
#     prompt_template = prompt_template.replace("{fie_loc3}", fie_loc3)
#     prompt_template = prompt_template.replace("{fie_loc4}", fie_loc4)
    
#     prompt = prompt_template.format(
#         expert_role=expert_role,
#         figure_description=figure_description
#     )
    
#     response = call_vlm_with_examples(prompt, image_path, example_paths)
    
#     # Parse Q&A from response
#     try:
#         question_match = re.search(r'Question:\s*(.*?)(?=\nAnswer:|\n\n|$)', response, re.DOTALL)
#         answer_match = re.search(r'Answer:\s*(.*?)(?=\n\n|$)', response, re.DOTALL)
        
#         if question_match and answer_match:
#             question = question_match.group(1).strip()
#             answer = answer_match.group(1).strip()
#             return question, answer
#         else:
#             print("⚠️ Could not parse Q&A from response, returning raw response")
#             return response, ""
#     except Exception as e:
#         print(f"⚠️ Error parsing Q&A: {e}")
#         return response, ""

# def verify_qa_requires_image(image_path: str, question: str, answer: str) -> str:
#     """Verify if the question requires the image to be answered"""
#     print("🔍 Verifying if question requires image context...")
    
#     prompt = f"""You are evaluating whether a question requires visual information from an image to be answered correctly.

# Question: {question}
# Answer: {answer}

# Please analyze if this question can be answered without seeing the image. Consider:
# 1. Does the question reference specific visual elements (shapes, colors, positions, labels, etc.)?
# 2. Does the answer rely on information that can only be obtained from the image?
# 3. Could someone answer this question accurately using only general knowledge?

# Respond with:
# - "REQUIRES_IMAGE" if the question cannot be answered without the image
# - "CAN_ANSWER_WITHOUT_IMAGE" if the question can be answered without the image
# - Brief explanation of your reasoning

# Your evaluation:"""
    
#     response = call_vlm_simple(prompt, image_path)
#     return response

# def process_image_for_qa_dataset(image_path: str) -> dict:
#     """Complete pipeline: extract role context, generate Q&A pair, and verify"""
#     print(f"🔄 Processing image for QA dataset: {image_path}")
    
#     # Stage 1: Extract role context
#     figure_description, expert_role, figure_category = extract_role_context(image_path)
    
#     # Stage 2: Generate Q&A pair
#     question, answer = generate_qa_pair(image_path, figure_description, expert_role)
    
#     # Stage 3: Verify if question requires image
#     verification_result = verify_qa_requires_image(image_path, question, answer)
    
#     return {
#         "image_path": image_path,
#         "figure_description": figure_description,
#         "expert_role": expert_role,
#         "figure_category": figure_category,
#         "question": question,
#         "answer": answer,
#         "verification_result": verification_result
#     }

def call_vlm_interleaved_ollama(chunks: List[Dict], model: str = "qwen3-vl:32b", image_base_dir: str = "", prompt: str = None) -> str:
    """
    Calls local Ollama with interleaved text and images using message chaining strategy.
    
    Args:
        chunks: List of dicts with 'content' and 'artifact'/'chunk_type'
        model: Ollama model name (default: qwen3-vl:32b)
        image_base_dir: Base directory for resolving relative image paths
        prompt: Optional instruction prompt to append after chunks
    """
    print(f"👁️ Calling VLM interleaved (Ollama) with {len(chunks)} chunks...")
    
    # Local Ollama endpoint
    url = "http://127.0.0.1:11434/api/chat"
    messages = []
    
    for chunk in chunks:
        content = chunk.get("content", "")
        artifact = chunk.get("artifact", "None")
        chunk_type = chunk.get("chunk_type", "")
        
        # Base message
        msg = {
            "role": "user", 
            "content": content
        }
        
        # Check for image
        if chunk_type == "standalone image" and artifact and artifact != "None":
            # Extract image path from markdown: ![Image](path)
            match = re.search(r'\!\[.*?\]\((.*?)\)', artifact)
            if match:
                rel_path = match.group(1)
                img_path = os.path.join(image_base_dir, rel_path) if image_base_dir else rel_path
                
                if os.path.exists(img_path):
                    try:
                        msg["images"] = [encode_image_to_base64(img_path)]
                    except Exception as e:
                         print(f"⚠️ Failed to encode image {img_path}: {e}")
                else:
                    print(f"⚠️ Image not found: {img_path}")
        
        messages.append(msg)
            
    # Append prompt if provided
    if prompt:
        messages.append({
            "role": "user",
            "content": prompt
        })

    payload = {
        "model": model,
        "messages": messages,
        "options": {
            "num_ctx": 16384,  # High context window for images
            "temperature": 0.0
        },
        "stream": False
    }
    
    try:
        print(f"Sending {len(messages)} interleaved blocks to Ollama...")
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()['message']['content']
        
        print(f"✅ VLM response received ({len(result)} chars)")
        logging.info(f"VLM Response (Ollama Interleaved) - {len(result)} chars")
        
        return result
        
    except Exception as e:
        print(f"❌ Ollama call failed: {e}")
        if 'response' in locals():
            print(f"Response: {response.text}")
        return ""

if __name__ == "__main__":
    # Setup logging
    setup_logging()

    chunks_interleaved = [
            {
                "content": "Chunk 1",
                "chunk_type": "standalone image",
                "artifact": r"![Image](Samples/VLM/Singles/Circuits/IEC60034-2-1{ed3.0}b_Fig14_Circuit.png)"
            },
            {
                "content": r"""Chunk 2: This flowchart illustrates the procedure for determining the IES classification and losses for a Power Drive System (PDS). The process begins by selecting a determination method: Test or Calculation. 
                                
                            The Test path (left) involves measuring input/output directly to find PDS losses (PL,PDS​), then adding required uncertainties according to Formula (22). 
                            
                            The Calculation path (right) uses datasheet values. It calculates absolute motor losses and CDM (Complete Drive Module) losses separately. Depending on whether the operating point is at full load (100; 100) or part load, different formulas (12 or 13) are used to sum these into the total absolute PDS losses. 
                            
                            Both paths converge to calculate the relative PDS losses by comparing absolute losses to the rated motor power (PR,M​). Finally, this relative value is used to assign the specific IES class via Formula (20). Notes indicate this cycle repeats for various speed and torque adjustments. """,
                "chunk_type": "text",
                "artifact": "None"
            },
            {
                "content": r"""Chunk 3: This flowchart outlines the procedure for determining the IE classification and losses for a Complete Drive Module (CDM). The process begins by establishing the rated output current (Ir,out​) and equivalent apparent power (Sr,equ​), either from specifications or mechanical power ratings. 

                            Users then select a determination method: Test (measuring via calorimetric or input-output methods) or Calculation (using formulas or manufacturer models). Both paths involve verifying load conditions and power factors (cosΦ) to calculate absolute CDM losses (PL,CDM​), incorporating required uncertainties. Finally, losses are adjusted if the CDM has modified characteristics, and the result is used to determine the specific IE class. The process repeats for various part-load operating points.""",
                "chunk_type": "text",
                "artifact": "None"
            },
            {
                "content": "Chunk 4: Continuous operation periodic duty with related load/speed changes - Duty type S8",
                "chunk_type": "image",
                "artifact": r"![Image](Samples/VLM/Singles/CurvePlots/IEC60034-1{ed14.0}b_Fig8_Plot.png)"
            },
            {
                "content": "Chunk 5: Determination of IES classsification f PDS and loss determination for part loading operations",
                "chunk_type": "image",
                "artifact": r"![Image](Samples/VLM/Singles/Flowcharts/iec61800-9-2{ed2.0}b_Fig27_FlowChart.png)"
            }
        ]
    
    ### Test LLM
    print("Testing LLM...")
    try:
        llm_response = call_llm_simple("Who is the president of the United States?")
        print(f"LLM Response: {llm_response[:100]}...")
    except Exception as e:
        print(f"LLM Test Failed: {e}")
    
    ## Test VLM
    print("\nTesting VLM...")
    test_image = [
        "Samples/VLM/Singles/CurvePlots/IEC60034-2-2{ed2.0}b_Fig10_Plot.png", ### water density and specific heat vs temperature
        "Samples/VLM/Singles/Circuits/IEC60034-2-1{ed3.0}b_Fig14_Circuit.png", # Eh star test circuit
        "Samples/VLM/Singles/Flowcharts/iec61800-9-2{ed2.0}b_Fig27_FlowChart.png", # IES classification and loss determination
        "Samples/VLM/Singles/CurvePlots/IEC60034-1{ed14.0}b_Fig8_Plot.png", # Continuous operation periodic duty with related load/speed changes - Duty type S8
        
        "Samples/VLM/Singles/CurvePlots/iec61800-9-2{ed2.0}b_FigE7_Plot.png", # Efficiency map of exemplary motor in Y or D
        "Samples/VLM/Singles/CurvePlots/IEC60034-2-3{ed2.0}b_FigC1_Plot.png", # torque vs speed in Y / D 
    ]
    
    if Path(test_image[0]).exists():
        ### 1. Simple VLM
        print("\n1. Testing call_vlm_simple...")
        try:
            vlm_response = call_vlm_simple("Describe this image briefly.", test_image[0])
            print(f"VLM Response: {vlm_response[:100]}...")
        except Exception as e:
             print(f"call_vlm_simple Failed: {e}")

        # 3. VLM with Multiple Images
        print("\n3. Testing call_vlm_with_multiple_images...")
        try:
            resp = call_vlm_with_multiple_images("Are these images the same?", [test_image[4], test_image[5]])
            print(f"Response: {resp[:100]}...")
        except Exception as e:
             print(f"call_vlm_with_multiple_images Failed: {e}")
             
        # 4. VLM Interweaved (Original)
        print("\n4. Testing call_vlm_interweaved...")
        try:
            instruction = """Analyze the sequence of chunks provided. Each chunk contains either text or an image related to motor efficiency and IES classification. 

Your task is to:
1. Identify the logical flow and dependencies between the chunks
2. Determine if the current order makes sense for understanding motor classification procedures
3. Suggest the optimal sequence that would help someone learn about IES classification step-by-step
4. Return your analysis followed by the chunks in your recommended order, using the format: CHUNK<|#|><include chunk number><|#|>[textual content of the chunk]<|#|>

Consider how text explanations should relate to visual diagrams, flowcharts, and technical specifications."""

            resp = call_vlm_interweaved(instruction, chunks_interleaved)
            print(f"Response: {resp}...")
        except Exception as e:
            print(f"call_vlm_interweaved Failed: {e}")

        # 5. VLM Interleaved Ollama (New)
        # print("\n5. Testing call_vlm_interleaved_ollama...")
        # # Construct chunks in the expected format for Ollama function
        
        
        # try:
        #     # Using empty image_base_dir because test_image is already a valid path
        #     instruction = "Order these chunks based on their relevance to IES classification of motors. Return the chunks with just the text content separated by <|#|>"
        #     resp = call_vlm_interleaved_ollama(chunks_interleaved, image_base_dir="", prompt=instruction)
        #     print(f"Response:\n{resp}...")
        # except Exception as e:
        #     print(f"call_vlm_interleaved_ollama Failed: {e}")

        # # 6. VLM Multi Images Ollama (New)
        # print("\n6. Testing call_vlm_multi_images_ollama...")
        # try:
        #     multi_images = [
        #         "Samples/VLM/Singles/CurvePlots/IEC60034-2-2{ed2.0}b_Fig10_Plot.png",
        #         "Samples/VLM/Singles/CurvePlots/IEC60034-1{ed14.0}b_Fig8_Plot.png",
        #         "Samples/VLM/Singles/Flowcharts/iec61800-9-2{ed2.0}b_Fig27_FlowChart.png"
        #         "Samples/VLM/Singles/Circuits/IEC60034-2-1{ed3.0}b_Fig14_Circuit.png",
        #         "Samples/VLM/Singles/Equations/IEC60034-2-3{ed2.0}b_Eqn_10-11.png"
        #     ]
        #     resp = call_vlm_multi_images_ollama("Describe each image briefly in order.", multi_images)
        #     print(f"Response:\n{resp}...")
        # except Exception as e:
        #      print(f"call_vlm_multi_images_ollama Failed: {e}")

    else:
        print(f"Test image not found: {test_image}")


# ============================================================================
# ASYNC RATE-LIMITED BATCH PROCESSING
# ============================================================================

class RateLimiter:
    """Token bucket rate limiter for API calls with per-minute limits.
    
    Creates asyncio primitives lazily to work with any event loop.
    """
    
    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        """
        Args:
            requests_per_minute: Maximum requests allowed per minute
            burst_size: Maximum burst size (concurrent requests in a short period)
        """
        self.rpm = requests_per_minute
        self.burst_size = burst_size
        self.interval = 60.0 / requests_per_minute  # Seconds between requests
        self.last_request_time = 0.0
        # Don't create asyncio primitives here - create them lazily per event loop
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._lock: Optional[asyncio.Lock] = None
        self._loop_id: Optional[int] = None
    
    def _ensure_primitives(self):
        """Ensure asyncio primitives exist for the current event loop."""
        try:
            current_loop = asyncio.get_running_loop()
            current_loop_id = id(current_loop)
        except RuntimeError:
            current_loop_id = None
        
        # Recreate primitives if loop changed
        if self._loop_id != current_loop_id:
            self._semaphore = asyncio.Semaphore(self.burst_size)
            self._lock = asyncio.Lock()
            self._loop_id = current_loop_id
    
    async def acquire(self):
        """Acquire permission to make a request"""
        self._ensure_primitives()
        
        async with self._lock:
            now = time.monotonic()
            time_since_last = now - self.last_request_time
            if time_since_last < self.interval:
                await asyncio.sleep(self.interval - time_since_last)
            self.last_request_time = time.monotonic()
        
        await self._semaphore.acquire()
    
    def release(self):
        """Release the semaphore after request completes"""
        if self._semaphore is not None:
            self._semaphore.release()

# Rate limits are loaded from config.yaml above (GEMINI_RPM, GEMINI_BURST)
# Override via environment if needed:
GEMINI_RPM = int(os.environ.get("GEMINI_RPM", str(GEMINI_RPM)))
GEMINI_BURST = int(os.environ.get("GEMINI_BURST", str(GEMINI_BURST)))

_rate_limiter: Optional[RateLimiter] = None

def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(
            requests_per_minute=GEMINI_RPM,
            burst_size=GEMINI_BURST
        )
        print(f"⚡ Rate limiter initialized: {GEMINI_RPM} RPM, burst={GEMINI_BURST}")
    return _rate_limiter


async def _async_call_llm_simple(prompt: str, session: aiohttp.ClientSession, 
                                  rate_limiter: RateLimiter, timeout: int = 300) -> str:
    """Async version of call_llm_simple. Supports all backends."""
    await rate_limiter.acquire()
    try:
        if BACKEND == "OLLAMA":
            data = {
                "model": LLM_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.0}
            }
            async with session.post(API_URL, json=data, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    result_json = await resp.json()
                    # Track token usage (Ollama format)
                    _update_token_stats(
                        result_json.get("prompt_eval_count", 0),
                        result_json.get("eval_count", 0)
                    )
                    return result_json["message"]["content"]
                else:
                    raise Exception(f"HTTP {resp.status}")
                    
        elif BACKEND == "GEMINI":
            url = GEMINI_URL.format(model=LLM_MODEL_NAME) + f"?key={API_KEY}"
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0}
            }
            async with session.post(url, headers=HEADERS, json=data, 
                                   timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    resp_json = await resp.json()
                    # Track token usage
                    usage = resp_json.get("usageMetadata", {})
                    _update_token_stats(
                        usage.get("promptTokenCount", 0),
                        usage.get("candidatesTokenCount", 0)
                    )
                    return resp_json["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    text = await resp.text()
                    raise Exception(f"HTTP {resp.status}: {text[:100]}")
                    
        elif BACKEND == "OPENAI":
            data = {
                "model": LLM_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0
            }
            async with session.post(API_URL, headers=HEADERS, json=data,
                                   timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    resp_json = await resp.json()
                    # Track token usage
                    usage = resp_json.get("usage", {})
                    _update_token_stats(
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0)
                    )
                    return resp_json["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"HTTP {resp.status}")
                    
        else:  # OpenAI-compatible
            data = {
                "model": LLM_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}]
            }
            async with session.post(API_URL, headers=HEADERS, json=data, 
                                   timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    response_data = await resp.json()
                    return response_data["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"HTTP {resp.status}")
    finally:
        rate_limiter.release()


async def _async_call_vlm_interweaved(prompt: str, chunks: List[Dict], 
                                       session: aiohttp.ClientSession,
                                       rate_limiter: RateLimiter, timeout: int = 300) -> str:
    """Async version of call_vlm_interweaved. Supports all backends."""
    await rate_limiter.acquire()
    
    def _extract_image_path(chunk):
        """Helper to extract image path from chunk"""
        if 'chunk_type' in chunk:
            chunk_type = chunk.get('chunk_type', '')
            artifact = chunk.get('artifact', 'None')
            if chunk_type in ['standalone image', 'image'] and artifact != 'None':
                match = re.search(r'!\[Image\]\(([^)]+)\)', artifact)
                if match:
                    return match.group(1)
            return None
        return chunk.get('image_path')
    
    try:
        if BACKEND == "OLLAMA":
            messages = [{"role": "user", "content": prompt}]
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.get('content', '')
                image_path = _extract_image_path(chunk)
                msg = {"role": "user", "content": f"CHUNK{i+1}: {chunk_text}"}
                if image_path and Path(image_path).exists():
                    msg["images"] = [encode_image_to_base64(image_path)]
                messages.append(msg)
            
            data = {
                "model": VLM_MODEL_NAME,
                "messages": messages,
                "stream": False,
                "options": {"num_ctx": 16384, "temperature": 0.0}
            }
            async with session.post(API_URL, json=data, 
                                   timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    result_json = await resp.json()
                    # Track token usage (Ollama format)
                    _update_token_stats(
                        result_json.get("prompt_eval_count", 0),
                        result_json.get("eval_count", 0)
                    )
                    return result_json["message"]["content"]
                else:
                    raise Exception(f"HTTP {resp.status}")
                    
        elif BACKEND == "GEMINI":
            # Gemini: parts array with text and inline_data
            parts = [{"text": prompt}]
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.get('content', '')
                image_path = _extract_image_path(chunk)
                parts.append({"text": f"\n\nCHUNK {i+1}:\n{chunk_text}"})
                if image_path and Path(image_path).exists():
                    base64_image = encode_image_to_base64(image_path)
                    mime_type = get_image_mime_type(image_path)
                    parts.append({"inline_data": {"mime_type": mime_type, "data": base64_image}})
            
            url = GEMINI_URL.format(model=VLM_MODEL_NAME) + f"?key={API_KEY}"
            data = {
                "contents": [{"parts": parts}],
                "generationConfig": {"temperature": 0.0}
            }
            async with session.post(url, headers=HEADERS, json=data,
                                   timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    resp_json = await resp.json()
                    # Track token usage
                    usage = resp_json.get("usageMetadata", {})
                    _update_token_stats(
                        usage.get("promptTokenCount", 0),
                        usage.get("candidatesTokenCount", 0)
                    )
                    return resp_json["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    text = await resp.text()
                    raise Exception(f"HTTP {resp.status}: {text[:100]}")
                    
        elif BACKEND == "OPENAI":
            # OpenAI: content array with text and image_url
            content = [{"type": "text", "text": prompt}]
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.get('content', '')
                image_path = _extract_image_path(chunk)
                content.append({"type": "text", "text": f"\n\nCHUNK {i+1}:\n{chunk_text}"})
                if image_path and Path(image_path).exists():
                    base64_image = encode_image_to_base64(image_path)
                    mime_type = get_image_mime_type(image_path)
                    image_url = f"data:{mime_type};base64,{base64_image}"
                    content.append({"type": "image_url", "image_url": {"url": image_url}})
            
            data = {
                "model": VLM_MODEL_NAME,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.0
            }
            async with session.post(API_URL, headers=HEADERS, json=data,
                                   timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    resp_json = await resp.json()
                    # Track token usage
                    usage = resp_json.get("usage", {})
                    _update_token_stats(
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0)
                    )
                    return resp_json["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"HTTP {resp.status}")
                    
        else:  # OpenAI-compatible
            content = [{"type": "text", "text": prompt}]
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.get('content', '')
                image_path = _extract_image_path(chunk)
                text_block = f"\n\n<|#|>CHUNK{i+1}<|#|>START<|#|>\n{chunk_text}\n"
                
                if image_path and Path(image_path).exists():
                    text_block += "<|#|>Image<|#|>"
                    content.append({"type": "text", "text": text_block})
                    base64_image = encode_image_to_base64(image_path)
                    mime_type = get_image_mime_type(image_path)
                    image_url = f"data:{mime_type};base64,{base64_image}"
                    content.append({"type": "image_url", "image_url": {"url": image_url}})
                    content.append({"type": "text", "text": f"<|#|>CHUNK{i+1}<|#|>END<|#|>"})
                else:
                    text_block += f"<|#|>Image<|#|>None<|#|>CHUNK{i+1}<|#|>END<|#|>"
                    content.append({"type": "text", "text": text_block})
            
            data = {"model": VLM_MODEL_NAME, "messages": [{"role": "user", "content": content}]}
            local_headers = {**HEADERS, "Connection": "close"}
            
            async with session.post(API_URL, headers=local_headers, json=data,
                                   timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    response_data = await resp.json()
                    return response_data["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"HTTP {resp.status}")
    finally:
        rate_limiter.release()


async def _batch_llm_calls_async(prompts: List[str], max_retries: int = 3) -> List[str]:
    """Execute multiple LLM calls concurrently with rate limiting
    
    Args:
        prompts: List of prompts to process
        max_retries: Maximum retries per request
        
    Returns:
        List of responses in same order as prompts
    """
    rate_limiter = get_rate_limiter()
    results = [None] * len(prompts)
    
    async def process_single(idx: int, prompt: str, session: aiohttp.ClientSession):
        for attempt in range(max_retries):
            try:
                result = await _async_call_llm_simple(prompt, session, rate_limiter)
                results[idx] = result
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(f"Batch LLM call {idx} failed after {max_retries} attempts: {e}")
                    results[idx] = f"ERROR: {str(e)}"
                else:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    connector = aiohttp.TCPConnector(limit=GEMINI_BURST)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [process_single(i, prompt, session) for i, prompt in enumerate(prompts)]
        await asyncio.gather(*tasks)
    
    return results


async def _batch_vlm_calls_async(requests: List[Tuple[str, List[Dict]]], 
                                  max_retries: int = 3) -> List[str]:
    """Execute multiple VLM calls concurrently with rate limiting
    
    Args:
        requests: List of (prompt, chunks) tuples
        max_retries: Maximum retries per request
        
    Returns:
        List of responses in same order as requests
    """
    rate_limiter = get_rate_limiter()
    results = [None] * len(requests)
    
    async def process_single(idx: int, prompt: str, chunks: List[Dict], 
                            session: aiohttp.ClientSession):
        for attempt in range(max_retries):
            try:
                result = await _async_call_vlm_interweaved(prompt, chunks, session, rate_limiter)
                results[idx] = result
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(f"Batch VLM call {idx} failed after {max_retries} attempts: {e}")
                    results[idx] = f"ERROR: {str(e)}"
                else:
                    await asyncio.sleep(2 ** attempt)
    
    connector = aiohttp.TCPConnector(limit=GEMINI_BURST)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [process_single(i, prompt, chunks, session) 
                for i, (prompt, chunks) in enumerate(requests)]
        await asyncio.gather(*tasks)
    
    return results


def _run_async_batch(coro):
    """Helper to run async coroutine from sync context, handling various event loop states."""
    try:
        # Try to get existing loop
        try:
            loop = asyncio.get_running_loop()
            # Loop is running - use thread executor
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            # No running loop - safe to use asyncio.run
            pass
        
        return asyncio.run(coro)
        
    except Exception as e:
        logging.error(f"Async batch execution failed: {e}")
        raise


def batch_call_llm(prompts: List[str], show_progress: bool = True,
                   use_cache: bool = True) -> List[str]:
    """Synchronous wrapper for batch LLM calls with rate limiting and caching
    
    Args:
        prompts: List of prompts to process concurrently
        show_progress: Whether to print progress
        use_cache: Whether to use LLM response cache (default: True)
        
    Returns:
        List of responses in same order as prompts
    """
    _initialize_config()
    if not prompts:
        return []
    
    # Check cache for existing responses
    cache = None
    cached_responses = [None] * len(prompts)
    miss_indices = list(range(len(prompts)))
    
    if use_cache:
        try:
            from mirage.utils.llm_cache import get_llm_cache
            cache = get_llm_cache()
            if cache:
                # Convert prompts to (prompt, None) format for get_batch
                requests = [(p, None) for p in prompts]
                cached_responses, miss_indices = cache.get_batch(
                    'llm_simple', requests, model=LLM_MODEL_NAME
                )
                if len(miss_indices) < len(prompts):
                    cache_hits = len(prompts) - len(miss_indices)
                    if show_progress:
                        print(f"📦 Cache: {cache_hits}/{len(prompts)} hits, {len(miss_indices)} API calls needed")
        except ImportError:
            pass
    
    # If all cached, return immediately
    if not miss_indices:
        if show_progress:
            print(f"📦 All {len(prompts)} responses found in cache!")
        return cached_responses
    
    # Only process uncached requests
    uncached_prompts = [prompts[i] for i in miss_indices]
    
    if show_progress:
        print(f"⚡ Batch LLM: Processing {len(uncached_prompts)} requests (RPM={GEMINI_RPM}, burst={GEMINI_BURST})...")
    
    start_time = time.time()
    uncached_results = _run_async_batch(_batch_llm_calls_async(uncached_prompts))
    elapsed = time.time() - start_time
    
    if show_progress:
        print(f"✅ Batch LLM: Completed {len(uncached_prompts)} requests in {elapsed:.1f}s "
              f"({len(uncached_prompts)/max(0.1, elapsed):.1f} req/s)")
    
    # Cache new results and merge with cached
    results = cached_responses.copy()
    for i, idx in enumerate(miss_indices):
        results[idx] = uncached_results[i] if i < len(uncached_results) else ""
    
    # Save new results to cache
    if cache and use_cache:
        requests = [(prompts[i], None) for i in miss_indices]
        cache.set_batch('llm_simple', requests, uncached_results, list(range(len(miss_indices))), model=LLM_MODEL_NAME)
    
    return results


def batch_call_vlm_interweaved(requests: List[Tuple[str, List[Dict]]], 
                                show_progress: bool = True,
                                use_cache: bool = True) -> List[str]:
    """Synchronous wrapper for batch VLM calls with rate limiting and caching
    
    Args:
        requests: List of (prompt, chunks) tuples
        show_progress: Whether to print progress
        use_cache: Whether to use LLM response cache (default: True)
        
    Returns:
        List of responses in same order as requests
    """
    _initialize_config()
    if not requests:
        return []
    
    # Check cache for existing responses
    cache = None
    cached_responses = [None] * len(requests)
    miss_indices = list(range(len(requests)))
    
    if use_cache:
        try:
            from mirage.utils.llm_cache import get_llm_cache
            cache = get_llm_cache()
            if cache:
                cached_responses, miss_indices = cache.get_batch(
                    'vlm_interweaved', requests, model=VLM_MODEL_NAME
                )
                if len(miss_indices) < len(requests):
                    cache_hits = len(requests) - len(miss_indices)
                    if show_progress:
                        print(f"📦 Cache: {cache_hits}/{len(requests)} hits, {len(miss_indices)} API calls needed")
        except ImportError:
            pass
    
    # If all cached, return immediately
    if not miss_indices:
        if show_progress:
            print(f"📦 All {len(requests)} responses found in cache!")
        return cached_responses
    
    # Only process uncached requests
    uncached_requests = [requests[i] for i in miss_indices]
    
    if show_progress:
        print(f"⚡ Batch VLM: Processing {len(uncached_requests)} requests (RPM={GEMINI_RPM}, burst={GEMINI_BURST})...")
    
    start_time = time.time()
    uncached_results = _run_async_batch(_batch_vlm_calls_async(uncached_requests))
    elapsed = time.time() - start_time
    
    if show_progress:
        print(f"✅ Batch VLM: Completed {len(uncached_requests)} requests in {elapsed:.1f}s "
              f"({len(uncached_requests)/max(0.1, elapsed):.1f} req/s)")
    
    # Cache new results and merge with cached
    results = cached_responses.copy()
    for i, idx in enumerate(miss_indices):
        results[idx] = uncached_results[i] if i < len(uncached_results) else ""
    
    # Save new results to cache
    if cache and use_cache:
        cache.set_batch('vlm_interweaved', requests, uncached_results, miss_indices, model=VLM_MODEL_NAME)
    
    return results


# ============================================================================
# BATCH VLM WITH BASE64 IMAGES (for pdf_to_md.py and similar use cases)
# ============================================================================

async def _async_call_vlm_base64(prompt: str, base64_image: str, 
                                  session: aiohttp.ClientSession,
                                  rate_limiter: RateLimiter, 
                                  mime_type: str = "image/png",
                                  timeout: int = 300) -> str:
    """Async VLM call with base64-encoded image. Supports all backends."""
    await rate_limiter.acquire()
    
    try:
        if BACKEND == "OLLAMA":
            data = {
                "model": VLM_MODEL_NAME,
                "messages": [{
                    "role": "user", 
                    "content": prompt,
                    "images": [base64_image]
                }],
                "stream": False,
                "options": {"num_ctx": 16384, "temperature": 0.0}
            }
            async with session.post(API_URL, json=data, 
                                   timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    result_json = await resp.json()
                    # Track token usage (Ollama format)
                    _update_token_stats(
                        result_json.get("prompt_eval_count", 0),
                        result_json.get("eval_count", 0)
                    )
                    return result_json["message"]["content"]
                else:
                    raise Exception(f"HTTP {resp.status}")
                    
        elif BACKEND == "GEMINI":
            parts = [
                {"text": prompt},
                {"inline_data": {"mime_type": mime_type, "data": base64_image}}
            ]
            url = GEMINI_URL.format(model=VLM_MODEL_NAME) + f"?key={API_KEY}"
            data = {
                "contents": [{"parts": parts}],
                "generationConfig": {"temperature": 0.0}
            }
            async with session.post(url, headers=HEADERS, json=data,
                                   timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    resp_json = await resp.json()
                    # Track token usage
                    usage = resp_json.get("usageMetadata", {})
                    _update_token_stats(
                        usage.get("promptTokenCount", 0),
                        usage.get("candidatesTokenCount", 0)
                    )
                    return resp_json["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    text = await resp.text()
                    raise Exception(f"HTTP {resp.status}: {text[:100]}")
                    
        elif BACKEND == "OPENAI":
            image_url = f"data:{mime_type};base64,{base64_image}"
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
            data = {
                "model": VLM_MODEL_NAME,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.0
            }
            async with session.post(API_URL, headers=HEADERS, json=data,
                                   timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    resp_json = await resp.json()
                    # Track token usage
                    usage = resp_json.get("usage", {})
                    _update_token_stats(
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0)
                    )
                    return resp_json["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"HTTP {resp.status}")
                    
        else:  # OpenAI-compatible
            image_url = f"data:{mime_type};base64,{base64_image}"
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
            data = {
                "model": VLM_MODEL_NAME,
                "messages": [{"role": "user", "content": content}]
            }
            async with session.post(API_URL, headers=HEADERS, json=data,
                                   timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    response_data = await resp.json()
                    return response_data["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"HTTP {resp.status}")
    finally:
        rate_limiter.release()


async def _batch_vlm_base64_calls_async(requests: List[Tuple[str, str, str]], 
                                         max_retries: int = 3) -> List[str]:
    """Execute multiple VLM calls with base64 images concurrently.
    
    Args:
        requests: List of (prompt, base64_image, mime_type) tuples
        max_retries: Number of retries on failure
        
    Returns:
        List of responses in same order as requests
    """
    rate_limiter = get_rate_limiter()
    results = [""] * len(requests)
    
    async def process_single(idx: int, prompt: str, base64_image: str, 
                            mime_type: str, session: aiohttp.ClientSession):
        for attempt in range(max_retries):
            try:
                result = await _async_call_vlm_base64(
                    prompt, base64_image, session, rate_limiter, mime_type
                )
                results[idx] = result
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    results[idx] = f"ERROR: {str(e)}"
                else:
                    await asyncio.sleep(2 ** attempt)
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            process_single(i, prompt, b64, mime, session) 
            for i, (prompt, b64, mime) in enumerate(requests)
        ]
        await asyncio.gather(*tasks)
    
    return results


def batch_call_vlm_base64(requests: List[Tuple[str, str, str]], 
                          show_progress: bool = True) -> List[str]:
    """Synchronous wrapper for batch VLM calls with base64 images.
    
    Args:
        requests: List of (prompt, base64_image, mime_type) tuples
                  mime_type is typically "image/png" or "image/jpeg"
        show_progress: Whether to print progress
        
    Returns:
        List of responses in same order as requests
        
    Example:
        requests = [
            ("Describe this image:", base64_data_1, "image/png"),
            ("What's in this table:", base64_data_2, "image/png"),
        ]
        responses = batch_call_vlm_base64(requests)
    """
    _initialize_config()
    if not requests:
        return []
    
    if show_progress:
        print(f"⚡ Batch VLM (base64): Processing {len(requests)} requests "
              f"(RPM={GEMINI_RPM}, burst={GEMINI_BURST})...")
    
    start_time = time.time()
    results = _run_async_batch(_batch_vlm_base64_calls_async(requests))
    elapsed = time.time() - start_time
    
    if show_progress:
        success_count = sum(1 for r in results if not r.startswith("ERROR:"))
        print(f"✅ Batch VLM (base64): Completed {success_count}/{len(requests)} "
              f"in {elapsed:.1f}s ({len(requests)/elapsed:.1f} req/s)")
    
    return results


# Alias for convenience
call_llm = call_llm_simple
call_vlm = call_vlm_interweaved
