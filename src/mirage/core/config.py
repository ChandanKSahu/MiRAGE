"""
Configuration loader for the QA Dataset Generation Pipeline.
Loads settings from config.yaml and provides easy access to all modules.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Find config.yaml relative to this file
_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_config_cache: Optional[Dict[str, Any]] = None


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file with caching.
    
    Returns default configuration if config file not found.
    This allows the package to be imported without a config file.
    """
    global _config_cache
    
    if _config_cache is not None and config_path is None:
        return _config_cache
    
    path = Path(config_path) if config_path else _CONFIG_PATH
    
    # If config file doesn't exist, return defaults
    if not path.exists():
        # Try workspace root config.yaml
        workspace_config = Path.cwd() / "config.yaml"
        if workspace_config.exists():
            path = workspace_config
        else:
            # Return default configuration - allows import without config file
            return _get_default_config()
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config_path is None:
        _config_cache = config
    
    return config


def _get_default_config() -> Dict[str, Any]:
    """Return default configuration when no config file is available.
    
    This enables the package to be imported and basic operations to work
    without requiring a config.yaml file upfront.
    """
    return {
        'backend': {
            'active': os.environ.get('LLM_BACKEND', 'GEMINI'),
            'gemini': {
                'llm_model': 'gemini-2.0-flash',
                'vlm_model': 'gemini-2.0-flash',
            },
            'openai': {
                'llm_model': 'gpt-4o-mini',
                'vlm_model': 'gpt-4o',
            },
            'ollama': {
                'base_url': 'http://localhost:11434',
                'llm_model': 'llama3',
                'vlm_model': 'llava',
            }
        },
        'rate_limiting': {
            'requests_per_minute': 60,
            'burst_size': 15
        },
        'paths': {
            'input_pdf_dir': 'data/documents',
            'output_dir': 'output'
        },
        'parallel': {
            'num_workers': 3,
            'qa_max_workers': 6,
            'dedup_max_workers': 4
        },
        'qa_generation': {
            'num_qa_pairs': 100,
            'type': 'multihop'
        }
    }


def get_backend_config() -> Dict[str, Any]:
    """Get the active backend configuration."""
    config = load_config()
    backend_name = config['backend']['active'].lower()
    backend_config = config['backend'].get(backend_name, {})
    
    return {
        'name': config['backend']['active'].upper(),
        **backend_config
    }


def get_api_key(backend_name: str = None) -> str:
    """Load API key for the specified or active backend."""
    config = load_config()
    
    if backend_name is None:
        backend_name = config['backend']['active'].lower()
    else:
        backend_name = backend_name.lower()
    
    backend_config = config['backend'].get(backend_name, {})
    api_key_path = backend_config.get('api_key_path')
    
    if not api_key_path:
        return ""
    
    try:
        with open(api_key_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"⚠️ API key file not found: {api_key_path}")
        return ""


def get_rate_limit_config() -> Dict[str, int]:
    """Get rate limiting configuration."""
    config = load_config()
    return config.get('rate_limiting', {
        'requests_per_minute': 60,
        'burst_size': 15
    })


def get_parallel_config() -> Dict[str, Any]:
    """Get parallel processing configuration."""
    config = load_config()
    return config.get('parallel', {
        'num_workers': 3,
        'available_gpus': [0, 1, 2],
        'qa_max_workers': 6,
        'dedup_max_workers': 4
    })


def get_retrieval_config() -> Dict[str, Any]:
    """Get context retrieval configuration."""
    config = load_config()
    return config.get('retrieval', {})


def get_embedding_config() -> Dict[str, Any]:
    """Get embedding configuration."""
    config = load_config()
    return config.get('embedding', {})


def get_paths_config() -> Dict[str, Any]:
    """Get input/output paths configuration."""
    config = load_config()
    return config.get('paths', {})


def get_processing_config() -> Dict[str, Any]:
    """Get processing limits configuration."""
    config = load_config()
    return config.get('processing', {})


def get_evaluation_config() -> Dict[str, Any]:
    """Get evaluation configuration."""
    config = load_config()
    return config.get('evaluation', {})


def get_domain_expert_config() -> Dict[str, Any]:
    """Get domain/expert persona configuration.
    
    Returns:
        Dict with 'expert_persona', 'domain' (may be None if auto-detect),
        and other settings like 'use_multimodal_embeddings', 'output_dir'
    """
    config = load_config()
    return config.get('domain_expert', {
        'expert_persona': None,
        'domain': None,
        'use_multimodal_embeddings': True,
        'output_dir': 'trials/domain_analysis'
    })


def get_qa_correction_config() -> Dict[str, Any]:
    """Get QA correction configuration.
    
    Returns:
        Dict with 'enabled' (bool), 'max_attempts' (int)
    """
    config = load_config()
    return config.get('qa_correction', {
        'enabled': True,
        'max_attempts': 1
    })


def get_qa_generation_config() -> Dict[str, Any]:
    """Get QA generation control configuration.
    
    Returns:
        Dict with:
        - 'num_qa_pairs': Target number of QA pairs (None = no limit)
        - 'type': Type of QA to generate ('multihop', 'multimodal', 'text', 'mix')
    """
    config = load_config()
    return config.get('qa_generation', {
        'num_qa_pairs': 1000,
        'type': 'multihop'
    })


def get_faiss_config() -> Dict[str, Any]:
    """Get FAISS configuration.
    
    Returns:
        Dict with:
        - 'index_type': FAISS index type (default: IndexFlatIP)
        - 'normalize_l2': Whether to normalize embeddings (default: True)
        - 'use_gpu': Whether to use GPU for FAISS (default: False)
        - 'gpu_id': GPU device ID to use (default: 0)
    """
    config = load_config()
    return config.get('faiss', {
        'index_type': 'IndexFlatIP',
        'normalize_l2': True,
        'use_gpu': False,
        'gpu_id': 0
    })


def get_deduplication_config() -> Dict[str, Any]:
    """Get deduplication configuration.
    
    Returns:
        Dict with:
        - 'enabled': Whether to run deduplication (default: True)
        - 'alpha': Blending factor for similarity (default: 0.6)
        - 'question_similarity_threshold': Threshold for question similarity (default: 0.75)
        - 'min_community_size': Minimum cluster size (default: 2)
        - 'answer_similarity': Dict of similarity thresholds
    """
    config = load_config()
    return config.get('deduplication', {
        'enabled': True,
        'alpha': 0.6,
        'question_similarity_threshold': 0.75,
        'min_community_size': 2,
        'answer_similarity': {'low': 0.7, 'medium': 0.85, 'high': 0.95}
    })


# Convenience function to print current config
def print_config_summary():
    """Print a summary of the current configuration."""
    config = load_config()
    backend = get_backend_config()
    rate_limit = get_rate_limit_config()
    parallel = get_parallel_config()
    qa_gen = get_qa_generation_config()
    
    print("=" * 60)
    print("📋 CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Backend: {backend['name']}")
    print(f"  LLM Model: {backend.get('llm_model', 'N/A')}")
    print(f"  VLM Model: {backend.get('vlm_model', 'N/A')}")
    print(f"Rate Limiting:")
    print(f"  RPM: {rate_limit.get('requests_per_minute', 60)}")
    print(f"  Burst: {rate_limit.get('burst_size', 15)}")
    print(f"Parallel Processing:")
    print(f"  QA Workers: {parallel.get('qa_max_workers', 6)}")
    print(f"  Dedup Workers: {parallel.get('dedup_max_workers', 4)}")
    print(f"QA Generation:")
    print(f"  Target Pairs: {qa_gen.get('num_qa_pairs', 1000)}")
    print(f"  Type: {qa_gen.get('type', 'multihop')}")
    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()
