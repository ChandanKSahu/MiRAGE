"""
Configuration loader for the QA Dataset Generation Pipeline.

Modular pipeline configuration with options for each stage:
1. Document Processing: PDF/HTML via docling
2. Markdown Conversion: LLM options for tables/images
3. Chunking: LLM-based semantic chunking
4. Embedding: Multimodal/text embedding models
5. Reranking: VLM/text reranker models
6. Domain/Expert: BERTopic + LLM extraction
7. Context Building: Multi-hop retrieval
8. QA Generation: Generation, selection, verification
9. Deduplication: Hierarchical clustering + LLM merge
10. Evaluation: RAGAS-style metrics

Each module can be configured independently with model selection.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

# Find config.yaml relative to this file
_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_config_cache: Optional[Dict[str, Any]] = None


# =============================================================================
# MODEL REGISTRIES - Available options for each pipeline stage
# =============================================================================

# LLM/VLM Models (for text generation, QA, verification)
LLM_MODELS = {
    'gemini-2.0-flash': {'provider': 'gemini', 'multimodal': True, 'context_window': 1000000},
    'gemini-2.5-flash': {'provider': 'gemini', 'multimodal': True, 'context_window': 1000000},
    'gemini-2.5-pro': {'provider': 'gemini', 'multimodal': True, 'context_window': 2000000},
    'gpt-4o': {'provider': 'openai', 'multimodal': True, 'context_window': 128000},
    'gpt-4o-mini': {'provider': 'openai', 'multimodal': True, 'context_window': 128000},
    'gpt-4-turbo': {'provider': 'openai', 'multimodal': True, 'context_window': 128000},
    'gpt-oss-120b': {'provider': 'motor_maven', 'multimodal': False, 'context_window': 32000},
    'qwen2.5vl:32b': {'provider': 'ollama', 'multimodal': True, 'context_window': 32000},
    'qwen3-vl:32b': {'provider': 'ollama', 'multimodal': True, 'context_window': 32000},
    'llama3': {'provider': 'ollama', 'multimodal': False, 'context_window': 8000},
    'llava': {'provider': 'ollama', 'multimodal': True, 'context_window': 4096},
}

# Embedding Models (for semantic search)
EMBEDDING_MODELS = {
    'nomic': {
        'class': 'NomicVLEmbed',
        'hf_name': 'nomic-ai/nomic-embed-multimodal-7b',
        'multimodal': True,
        'dimension': 128,
    },
    'qwen3_vl': {
        'class': 'Qwen3VLEmbed',
        'hf_name': 'Qwen/Qwen3-VL-Embedding-8B',
        'multimodal': True,
        'dimension': 4096,
    },
    'qwen2_vl': {
        'class': 'Qwen2VLEmbed',
        'hf_name': 'Qwen/Qwen2-VL-7B-Instruct',
        'multimodal': True,
        'dimension': 1536,
    },
    'bge_vl': {
        'class': 'BGEVLEmbed',
        'hf_name': 'BAAI/BGE-VL-v1.5-zs',
        'multimodal': True,
        'dimension': 4096,
    },
    'bge_m3': {
        'class': 'SentenceTransformer',
        'hf_name': 'BAAI/bge-m3',
        'multimodal': False,
        'dimension': 1024,
    },
    'bge_large': {
        'class': 'SentenceTransformer',
        'hf_name': 'BAAI/bge-large-en-v1.5',
        'multimodal': False,
        'dimension': 1024,
    },
    'minilm': {
        'class': 'SentenceTransformer',
        'hf_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'multimodal': False,
        'dimension': 384,
    },
}

# Reranker Models (for chunk reranking)
RERANKER_MODELS = {
    'gemini_vlm': {
        'class': 'GeminiVLMReranker',
        'provider': 'gemini',
        'multimodal': True,
    },
    'mono_vlm': {
        'class': 'MonoVLMReranker',
        'hf_name': 'lightonai/MonoQwen2-VL-v0.1',
        'processor': 'Qwen/Qwen2-VL-2B-Instruct',
        'multimodal': True,
    },
    'mm_r5': {
        'class': 'MMR5Reranker',
        'hf_name': 'i2vec/MM-R5',
        'multimodal': True,
    },
    'florence2': {
        'class': 'Florence2Reranker',
        'hf_name': 'microsoft/Florence-2-large',
        'multimodal': True,
    },
    'text_embedding': {
        'class': 'TextEmbeddingReranker',
        'hf_name': 'BAAI/bge-large-en-v1.5',
        'multimodal': False,
    },
    'vlm_description': {
        'class': 'VLMDescriptionEmbed',
        'text_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'multimodal': True,
    },
}


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
    
    Configuration organized by pipeline stage per scikit-learn pattern.
    """
    return {
        # =================================================================
        # BACKEND CONFIGURATION - LLM/VLM Provider Settings
        # =================================================================
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
            },
            'motor_maven': {
                'llm_model': 'gpt-oss-120b',
                'api_url': 'https://dev.motor-maven.com/api/chat/completions',
            }
        },
        
        # =================================================================
        # RATE LIMITING
        # =================================================================
        'rate_limiting': {
            'requests_per_minute': 60,
            'burst_size': 15
        },
        
        # =================================================================
        # PATHS
        # =================================================================
        'paths': {
            'input_pdf_dir': 'data/documents',
            'output_dir': 'output'
        },
        
        # =================================================================
        # PARALLEL PROCESSING
        # =================================================================
        'parallel': {
            'num_workers': 3,
            'available_gpus': [0, 1, 2, 3],
            'qa_max_workers': 6,
            'dedup_max_workers': 4
        },
        
        # =================================================================
        # STAGE 1: DOCUMENT PROCESSING (PDF/HTML to Markdown)
        # =================================================================
        'pdf_processing': {
            'input_format': 'auto',  # auto, pdf, html
            'image_resolution_scale': 2.0,
            'num_threads': 14,
            'cuda_device_id': 1,
            # VLM for image/table annotation
            'annotation_model': 'gemini-2.0-flash',  # Options: any LLM_MODELS key
            # Docling options
            'do_picture_classification': False,
            'do_picture_description': True,
            'do_ocr': True,
            'do_code_enrichment': True,
            'do_formula_enrichment': True,
            'do_table_structure': True,
            'generate_table_images': True,
        },
        
        # =================================================================
        # STAGE 2: CHUNKING (Markdown to Semantic Chunks)
        # =================================================================
        'chunking': {
            'method': 'semantic',  # semantic, fixed, sentence
            'window_size': 20000,  # ~5000 tokens
            'overlap_size': 2000,  # ~500 tokens
            # LLM for semantic chunking
            'llm_model': 'gemini-2.0-flash',  # Options: any LLM_MODELS key
        },
        
        # =================================================================
        # STAGE 3: EMBEDDING
        # =================================================================
        'embedding': {
            'model': 'nomic',  # Options: nomic, qwen3_vl, qwen2_vl, bge_vl, bge_m3, bge_large, minilm
            'batch_size': 16,
            'cache_embeddings': True,
            'gpus': None,  # List of GPU IDs or None for auto
            # Model-specific options
            'models': {
                'nomic': {
                    'attn_implementation': 'sdpa',  # sdpa, flash_attention_2
                    'load_in_4bit': True,
                },
                'qwen3_vl': {
                    'use_quantization': False,
                },
            }
        },
        
        # =================================================================
        # STAGE 4: RERANKING
        # =================================================================
        'reranking': {
            'model': 'gemini_vlm',  # Options: gemini_vlm, mono_vlm, mm_r5, florence2, text_embedding
            'top_k': 5,
            # Model-specific options
            'models': {
                'mono_vlm': {
                    'torch_dtype': 'bfloat16',
                },
                'florence2': {
                    'max_new_tokens': 1024,
                    'num_beams': 3,
                },
            }
        },
        
        # =================================================================
        # STAGE 5: DOMAIN & EXPERT EXTRACTION
        # =================================================================
        'domain_expert': {
            'expert_persona': None,  # Auto-detect if None
            'domain': None,  # Auto-detect if None
            'use_multimodal_embeddings': True,
            'llm_model': 'gemini-2.0-flash',  # For topic interpretation
            'output_dir': 'trials/domain_analysis',
            # BERTopic settings
            'umap': {
                'n_neighbors': 15,
                'n_components': 5,
                'min_dist': 0.0,
                'metric': 'cosine',
            },
            'vectorizer': {
                'stop_words': 'english',
                'min_df': 2,
                'ngram_range': [1, 2],
            },
            'mmr_diversity': 0.5,
        },
        
        # =================================================================
        # STAGE 6: CONTEXT BUILDING (Multi-hop Retrieval)
        # =================================================================
        'retrieval': {
            'method': 'top_k',  # top_k, top_p
            'retrieval_k': 20,
            'retrieval_p': 0.9,
            'rerank_top_k': 10,
            'context_size': 2,
            # Multi-hop settings
            'multihop': {
                'max_depth': 2,
                'max_breadth': 5,
                'chunks_per_search': 2,
                'chunk_addition_mode': 'RELATED',  # EXPLANATORY, RELATED
                'max_consecutive_no_progress': 3,
            },
            # VLM for context verification
            'vlm_model': 'gemini-2.0-flash',
        },
        
        # =================================================================
        # STAGE 7: QA GENERATION
        # =================================================================
        'qa_generation': {
            'num_qa_pairs': 1000,
            'type': 'multihop',  # multihop, multimodal, text, mix
            # VLM for QA generation/selection/verification
            'vlm_model': 'gemini-2.0-flash',
            # Correction settings
            'correction_enabled': True,
            'correction_max_attempts': 1,
        },
        
        # =================================================================
        # STAGE 8: DEDUPLICATION
        # =================================================================
        'deduplication': {
            'enabled': True,
            'alpha': 0.6,  # Weight: α * semantic_sim + (1-α) * chunk_overlap
            'question_similarity_threshold': 0.75,
            'min_community_size': 2,
            'answer_similarity': {
                'high': 0.95,
                'medium': 0.85,
                'low': 0.70,
            },
            # LLM for merge/reorganize
            'llm_model': 'gemini-2.0-flash',
            # Embedding for dedup
            'embedding_model': 'bge_m3',
        },
        
        # =================================================================
        # STAGE 9: EVALUATION
        # =================================================================
        'evaluation': {
            'run_evaluation': False,
            'use_optimized_metrics': True,  # 3-5x faster
            'sample_size': None,
            'run_context_necessity': True,
            # Models for evaluation
            'llm_model': 'gemini-2.0-flash',
            'embedding_model': 'models/text-embedding-004',
            'temperature': 0,
        },
        
        # =================================================================
        # FAISS INDEX
        # =================================================================
        'faiss': {
            'index_type': 'IndexFlatIP',
            'normalize_l2': True,
            'use_gpu': False,
            'gpu_id': 0,
        },
        
        # =================================================================
        # SHUFFLING
        # =================================================================
        'shuffling': {
            'enabled': True,
            'seed': 42,
        },
        
        # =================================================================
        # PROCESSING LIMITS
        # =================================================================
        'processing': {
            'max_pdfs': 5,
            'sort_by_size': True,
            'max_chunks': None,
        },
        
        # =================================================================
        # QA CORRECTION (retry failed verifications)
        # =================================================================
        'qa_correction': {
            'enabled': True,
            'max_attempts': 1,
        },
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


def get_pdf_processing_config() -> Dict[str, Any]:
    """Get document processing configuration."""
    config = load_config()
    return config.get('pdf_processing', {
        'input_format': 'auto',
        'image_resolution_scale': 2.0,
        'annotation_model': 'gemini-2.0-flash',
        'do_ocr': True,
        'do_table_structure': True,
    })


def get_chunking_config() -> Dict[str, Any]:
    """Get chunking configuration."""
    config = load_config()
    return config.get('chunking', {
        'method': 'semantic',
        'window_size': 20000,
        'overlap_size': 2000,
        'llm_model': 'gemini-2.0-flash',
    })


def get_reranking_config() -> Dict[str, Any]:
    """Get reranking configuration."""
    config = load_config()
    return config.get('reranking', {
        'model': 'gemini_vlm',
        'top_k': 5,
    })


def get_model_info(model_name: str, model_type: str = 'llm') -> Dict[str, Any]:
    """Get model metadata from registry.
    
    Args:
        model_name: Name of the model
        model_type: 'llm', 'embedding', or 'reranker'
    
    Returns:
        Model metadata dict
    """
    registries = {
        'llm': LLM_MODELS,
        'embedding': EMBEDDING_MODELS,
        'reranker': RERANKER_MODELS,
    }
    
    registry = registries.get(model_type, {})
    return registry.get(model_name, {})


def list_available_models(model_type: str = 'llm') -> List[str]:
    """List available models for a given type.
    
    Args:
        model_type: 'llm', 'embedding', or 'reranker'
    
    Returns:
        List of model names
    """
    registries = {
        'llm': LLM_MODELS,
        'embedding': EMBEDDING_MODELS,
        'reranker': RERANKER_MODELS,
    }
    
    registry = registries.get(model_type, {})
    return list(registry.keys())


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
