"""
Base classes for modular pipeline components.

Design Philosophy:
- Simple `process()` interface instead of sklearn's fit/transform
- Comprehensive hyperparameter configuration from documentation
- Lazy model loading with caching
- GPU management for parallel processing
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging
import copy

logger = logging.getLogger(__name__)


class BaseModule(ABC):
    """
    Abstract base class for all pipeline modules.
    
    Key methods:
    - __init__: Store hyperparameters (no processing)
    - process(): Main execution method - runs the module
    - get_params/set_params: Parameter access
    
    All modules should be:
    - Configurable via constructor parameters
    - Composable (output of one is input to next)
    - Self-documenting via get_params()
    """
    
    def __init__(self, **kwargs):
        """Initialize module with configuration parameters."""
        self._params = kwargs
        self._logger = logging.getLogger(self.__class__.__name__)
        self._output = None  # Store last output for inspection
    
    @property
    def params(self) -> Dict[str, Any]:
        """Return current parameters."""
        return copy.deepcopy(self._params)
    
    def set_params(self, **params) -> 'BaseModule':
        """Set parameters."""
        for key, value in params.items():
            self._params[key] = value
        return self
    
    def get_params(self) -> Dict[str, Any]:
        """Get all parameters."""
        return self.params
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """Get single parameter value."""
        return self._params.get(key, default)
    
    @abstractmethod
    def process(self, X: Any, **kwargs) -> Any:
        """
        Main processing method - execute this module.
        
        Args:
            X: Input data (format depends on module type)
            **kwargs: Additional runtime parameters
            
        Returns:
            Processed output
        """
        pass
    
    def __call__(self, X: Any, **kwargs) -> Any:
        """Allow module to be called directly."""
        return self.process(X, **kwargs)
    
    def get_output(self) -> Any:
        """Get last output (for debugging/inspection)."""
        return self._output
    
    def __repr__(self) -> str:
        params = ', '.join(f"{k}={v!r}" for k, v in self._params.items() 
                          if not k.startswith('_'))
        return f"{self.__class__.__name__}({params})"


class BaseModelModule(BaseModule):
    """
    Base class for modules that use ML models (LLM, VLM, embeddings, rerankers).
    
    Adds:
    - Model lazy loading
    - GPU/device management
    - Model caching across instances
    """
    
    # Class-level model cache (shared across instances)
    _model_cache: Dict[str, Any] = {}
    
    def __init__(self, 
                 model: str = None,
                 device: str = "auto",
                 gpu_id: int = None,
                 cache_model: bool = True,
                 **kwargs):
        """
        Initialize model module.
        
        Args:
            model: Model name/path
            device: Device ("auto", "cuda", "cuda:0", "cpu")
            gpu_id: Specific GPU ID (overrides device)
            cache_model: Whether to cache loaded models
        """
        super().__init__(
            model=model,
            device=device,
            gpu_id=gpu_id,
            cache_model=cache_model,
            **kwargs
        )
        self._model = None
        self._processor = None
    
    @property
    def model_name(self) -> str:
        return self._params.get('model')
    
    @property
    def device(self) -> str:
        """Resolve device string."""
        gpu_id = self._params.get('gpu_id')
        if gpu_id is not None:
            return f'cuda:{gpu_id}'
        
        device = self._params.get('device', 'auto')
        if device == 'auto':
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _get_cache_key(self) -> str:
        """Generate unique cache key for model."""
        return f"{self.__class__.__name__}:{self.model_name}:{self.device}"
    
    def _load_model(self):
        """Load model - override in subclass."""
        raise NotImplementedError("Subclass must implement _load_model")
    
    def get_model(self):
        """Get model, loading if necessary (with caching)."""
        if self._model is not None:
            return self._model
        
        cache_key = self._get_cache_key()
        if self._params.get('cache_model') and cache_key in self._model_cache:
            self._logger.info(f"Using cached model: {cache_key}")
            self._model = self._model_cache[cache_key]
            return self._model
        
        self._logger.info(f"Loading model: {self.model_name} on {self.device}")
        self._load_model()
        
        if self._params.get('cache_model'):
            self._model_cache[cache_key] = self._model
        
        return self._model
    
    def clear_cache(self):
        """Clear model from cache."""
        cache_key = self._get_cache_key()
        if cache_key in self._model_cache:
            del self._model_cache[cache_key]
        self._model = None
    
    @classmethod
    def clear_all_cache(cls):
        """Clear all cached models."""
        cls._model_cache.clear()


class Pipeline:
    """
    Chain multiple modules into a processing pipeline.
    
    Example:
        pipeline = Pipeline([
            ('document', DocumentProcessor(input_format='pdf')),
            ('chunker', SemanticChunker(window_size=20000)),
            ('embedder', EmbeddingModule(model='nomic')),
        ])
        
        result = pipeline.run(input_path)
    """
    
    def __init__(self, steps: List[Tuple[str, BaseModule]], verbose: bool = True):
        """
        Initialize pipeline.
        
        Args:
            steps: List of (name, module) tuples
            verbose: Print progress during processing
        """
        self.steps = steps
        self.verbose = verbose
        self._validate_steps()
        self._outputs = {}  # Store outputs by step name
    
    def _validate_steps(self):
        """Validate pipeline steps."""
        names = [name for name, _ in self.steps]
        if len(names) != len(set(names)):
            raise ValueError("Step names must be unique")
        
        for name, module in self.steps:
            if not isinstance(module, BaseModule):
                raise TypeError(f"Step '{name}' must be a BaseModule instance")
    
    @property
    def named_steps(self) -> Dict[str, BaseModule]:
        """Return dict of step name -> module."""
        return {name: mod for name, mod in self.steps}
    
    def get_step(self, name: str) -> BaseModule:
        """Get module by name."""
        return self.named_steps.get(name)
    
    def run(self, X: Any, start_from: str = None, stop_after: str = None, 
            **kwargs) -> Any:
        """
        Run pipeline on input.
        
        Args:
            X: Input data
            start_from: Step name to start from (skip earlier steps)
            stop_after: Step name to stop after (skip later steps)
            **kwargs: Per-step kwargs as step_name__param=value
        
        Returns:
            Final output
        """
        Xt = X
        started = start_from is None
        
        for name, module in self.steps:
            if not started:
                if name == start_from:
                    started = True
                else:
                    continue
            
            if self.verbose:
                self._logger.info(f"Running: {name}")
                print(f"{'='*60}")
                print(f"Pipeline Step: {name}")
                print(f"{'='*60}")
            
            # Extract step-specific kwargs
            step_kwargs = {}
            prefix = f"{name}__"
            for k, v in kwargs.items():
                if k.startswith(prefix):
                    step_kwargs[k[len(prefix):]] = v
            
            Xt = module.process(Xt, **step_kwargs)
            self._outputs[name] = Xt
            
            if stop_after and name == stop_after:
                break
        
        return Xt
    
    def __call__(self, X: Any, **kwargs) -> Any:
        """Allow pipeline to be called directly."""
        return self.run(X, **kwargs)
    
    def get_output(self, step_name: str) -> Any:
        """Get output from specific step."""
        return self._outputs.get(step_name)
    
    def __getitem__(self, key: Union[int, str]):
        """Get step by index or name."""
        if isinstance(key, int):
            return self.steps[key][1]
        return self.named_steps[key]
    
    def __repr__(self) -> str:
        step_strs = [f"  ('{name}', {mod!r})" for name, mod in self.steps]
        return f"Pipeline([\n" + ",\n".join(step_strs) + "\n])"
    
    @property
    def _logger(self):
        return logging.getLogger('Pipeline')


# =============================================================================
# Default Configuration with ALL Hyperparameters
# =============================================================================

DEFAULT_CONFIG = {
    # Section 2.6: PDF Processing (pdf_to_md.py)
    'document_processor': {
        'input_format': 'auto',  # auto, pdf, html
        'image_resolution_scale': 2.0,
        'annotation_model': 'qwen2.5vl:32b',  # VLM for image/table descriptions
        'num_threads': 14,
        'cuda_device_id': 1,
        'do_picture_classification': False,  # Disabled to avoid CUDA OOM
        'do_picture_description': True,
        'do_ocr': True,
        'do_code_enrichment': True,
        'do_formula_enrichment': True,
        'do_table_structure': True,
        'generate_table_images': True,
    },
    
    # Section 2.4: Semantic Chunking (md_to_semantic_chunks.py)
    'chunker': {
        'method': 'semantic',  # semantic, fixed, sentence
        'window_size': 20000,  # ~5000 tokens
        'overlap_size': 2000,  # ~500 tokens
        'llm_model': 'gpt-oss-120b',  # Text-only LLM for chunking
        'max_workers': 4,
    },
    
    # Section 2.11: Embedding Models (embed_models.py)
    'embedding': {
        'model': 'nomic',  # nomic, qwen2_vl, bge_vl, bge_m3, bge_large, minilm
        'batch_size': 16,
        'normalize': True,  # L2 normalization for cosine similarity
        'cache_embeddings': True,
        # Model-specific (Nomic)
        'load_in_4bit': True,
        'bnb_4bit_compute_dtype': 'bfloat16',
        'bnb_4bit_use_double_quant': True,
        'torch_dtype': 'bfloat16',
        'attn_implementation': 'flash_attention_2',  # if available
    },
    
    # Section 2.9: Reranking (rerankers_multimodal.py)
    'reranker': {
        'model': 'mono_vlm',  # mono_vlm, mm_r5, florence2, vlm, text_embedding, gemini_vlm
        'top_k': 5,
        # MonoVLM specific
        'mono_vlm_model': 'lightonai/MonoQwen2-VL-v0.1',
        'mono_vlm_processor': 'Qwen/Qwen2-VL-2B-Instruct',
        # MM-R5 specific
        'mm_r5_model': 'i2vec/MM-R5',
        # Florence-2 specific
        'florence2_model': 'microsoft/Florence-2-large',
        'florence2_max_new_tokens': 1024,
        'florence2_num_beams': 3,
        # TextEmbedding specific
        'text_embedding_model': 'BAAI/bge-large-en-v1.5',
        # VLMDescription specific
        'vlm_desc_text_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'vlm_desc_max_tokens': 500,
        'vlm_desc_timeout': 180,
    },
    
    # Section 2.7: Domain/Expert Extraction (domain_expert.py)
    'domain_extractor': {
        'use_multimodal_embeddings': True,  # False = BGE-M3 text-only
        'embedding_model': 'nomic',  # for multimodal
        'text_embedding_model': 'BAAI/bge-m3',  # for text-only
        'llm_model': 'gpt-oss-120b',  # for domain/expert extraction
        'output_dir': 'output/domain_analysis',
        # BERTopic params (Section 2.13)
        'umap_n_neighbors': 15,
        'umap_n_components': 5,
        'umap_min_dist': 0.0,
        'umap_metric': 'cosine',
        'umap_random_state': 42,
        'vectorizer_stop_words': 'english',
        'vectorizer_min_df': 2,
        'vectorizer_ngram_range': (1, 2),
        'mmr_diversity': 0.5,
        'calculate_probabilities': False,
        'num_topics_to_show': 15,
    },
    
    # Section 2.3: Multi-hop Context Retrieval (context_retrieved.py)
    'context_builder': {
        # Multi-hop parameters
        'max_depth': 2,  # main.py default (context.py has 10)
        'max_breadth': 3,  # main.py default (context.py has 5)
        'chunks_per_search': 1,  # main.py default (context.py has 2)
        'chunk_addition_mode': 'RELATED',  # EXPLANATORY or RELATED
        'max_consecutive_no_progress': 3,  # Circuit breaker
        # Simple retrieval parameters
        'retrieval_method': 'top_k',  # top_k or top_p
        'retrieval_k': 20,
        'retrieval_p': 0.9,
        'rerank_top_k': 10,
        'context_size': 2,
        # Models
        'embedding_model': 'nomic',
        'reranker_model': 'mono_vlm',
        'vlm_model': 'qwen2.5vl:32b',  # for verification
    },
    
    # Section 2.8: QA Generation (qa_gen_multi_hop.py)
    'qa_generator': {
        'vlm_model': 'qwen2.5vl:32b',  # or gemini-2.5-flash
        'max_chunks': None,  # None = all
        'max_qa_pairs_per_chunk': 3,
        'output_successful': 'qa_multihop_pass.json',
        'output_failed': 'qa_multihop_fail.json',
        'output_irrelevant': 'irrelevant_chunk.json',
        # QA correction
        'enable_correction': True,
        'max_correction_attempts': 1,
    },
    
    # Section 2.5: Deduplication (deduplication.py)
    'deduplication': {
        'enabled': True,
        'question_similarity_threshold': 0.75,
        'answer_similarity_high': 0.95,
        'answer_similarity_medium': 0.85,
        'answer_similarity_low': 0.70,
        'min_community_size': 2,
        'embedding_model': 'BAAI/bge-m3',
        'vlm_model': 'qwen2.5vl:32b',  # for ranking/merging
        'max_workers': 4,
    },
    
    # Section 2.2: Evaluation (metrics.py)
    'evaluation': {
        'enabled': True,
        'sample_size': 50,  # for expensive metrics
        'use_gemini': True,  # prefer Gemini over OpenAI
        'llm_model': 'gemini-2.0-flash',
        'embedding_model': 'models/text-embedding-004',
        'temperature': 0,  # deterministic
        # OpenAI fallback
        'openai_model': 'gpt-4-turbo',
        'openai_embedding_model': 'text-embedding-3-small',
        # Metrics to run
        'run_faithfulness': True,
        'run_answer_relevance': True,
        'run_context_precision': True,
        'run_context_recall': True,
        'run_multihop_reasoning': True,
        'run_visual_dependency': True,
        'run_context_necessity': True,
        'run_domain_coverage': True,
        'run_semantic_diversity': True,
        'run_multimodal_faithfulness': True,
        'run_multimodal_answer_quality': True,
    },
    
    # Section 2.14: FAISS Index
    'faiss': {
        'index_type': 'IndexFlatIP',  # Inner product for cosine sim
        'use_gpu': False,
        'gpu_id': 0,
    },
    
    # Section 2.1: Main Pipeline
    'pipeline': {
        'input_pdf_dir': 'data/input',
        'input_chunks_file': None,  # Skip PDF->chunks if set
        'output_dir': 'output/results',
        'max_pdfs': None,  # None = all
        'sort_by_size': True,
        'max_chunks': None,
        'shuffle': True,
        'shuffle_seed': 42,
        'num_workers': 4,  # PDF/chunking
        'available_gpus': [1, 2, 3, 4],
        'qa_max_workers': 6,
        'dedup_max_workers': 4,
    },
    
    # Section 2.10: API Configuration (call_llm.py)
    'api': {
        'motor_maven_url': 'https://dev.motor-maven.com/api/chat/completions',
        'timeout': 300,
        'stream': False,
        'retry_initial_delay': 2,
        'retry_max_delay': 60,
        'retry_max_attempts': 3,
        # Ollama (Section 2.15)
        'ollama_num_ctx': 16384,
        'ollama_temperature': 0.0,
    },
    
    # LLM/VLM Model Registry (Table 1)
    'llm_models': {
        'gpt-oss-120b': {'provider': 'motor_maven', 'multimodal': False},
        'qwen2.5:32b-instruct': {'provider': 'ollama', 'multimodal': False},
        'gemini-2.5-flash': {'provider': 'gemini', 'multimodal': True},
        'gemini-2.5-pro': {'provider': 'gemini', 'multimodal': True},
        'gemini-2.0-flash': {'provider': 'gemini', 'multimodal': True},
    },
    
    'vlm_models': {
        'qwen2.5vl:32b': {'provider': 'motor_maven', 'multimodal': True},
        'qwen3-vl:32b': {'provider': 'ollama', 'multimodal': True},
        'gemini-2.5-flash': {'provider': 'gemini', 'multimodal': True},
        'granite3.2-vision:latest': {'provider': 'ollama', 'multimodal': True},
    },
    
    'embedding_models': {
        'nomic': {
            'class': 'NomicVLEmbed',
            'hf_name': 'nomic-ai/nomic-embed-multimodal-7b',
            'multimodal': True,
            'dimension': 128,
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
            'dimension': 768,
        },
        'bge_m3': {
            'hf_name': 'BAAI/bge-m3',
            'multimodal': False,
            'dimension': 1024,
        },
        'bge_large': {
            'hf_name': 'BAAI/bge-large-en-v1.5',
            'multimodal': False,
            'dimension': 1024,
        },
        'minilm': {
            'hf_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'multimodal': False,
            'dimension': 384,
        },
    },
    
    'reranker_models': {
        'mono_vlm': {
            'class': 'MonoVLMReranker',
            'model': 'lightonai/MonoQwen2-VL-v0.1',
            'multimodal': True,
        },
        'mm_r5': {
            'class': 'MMR5Reranker',
            'model': 'i2vec/MM-R5',
            'multimodal': True,
        },
        'florence2': {
            'class': 'Florence2Reranker',
            'model': 'microsoft/Florence-2-large',
            'multimodal': True,
        },
        'vlm': {
            'class': 'VLMReranker',
            'multimodal': True,
        },
        'text_embedding': {
            'class': 'TextEmbeddingReranker',
            'model': 'BAAI/bge-large-en-v1.5',
            'multimodal': False,
        },
        'gemini_vlm': {
            'class': 'GeminiVLMReranker',
            'model': 'gemini-2.5-flash',
            'multimodal': True,
        },
    },
}


class PipelineConfig:
    """
    Configuration manager for pipeline modules.
    
    Provides typed access to all hyperparameters from documentation tables.
    """
    
    def __init__(self, config_path: str = None, **overrides):
        """
        Load configuration.
        
        Args:
            config_path: Path to config.yaml
            **overrides: Override specific settings (component.param=value)
        """
        # Start with defaults
        self._config = copy.deepcopy(DEFAULT_CONFIG)
        
        # Load from file if provided
        if config_path:
            self._load_from_file(config_path)
        
        # Apply overrides
        for key, value in overrides.items():
            if '.' in key:
                component, param = key.split('.', 1)
                if component in self._config:
                    self._config[component][param] = value
            elif key in self._config:
                if isinstance(value, dict):
                    self._config[key].update(value)
    
    def _load_from_file(self, config_path: str):
        """Load configuration from YAML file."""
        import yaml
        
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return
        
        with open(path, 'r') as f:
            file_config = yaml.safe_load(f)
        
        if file_config:
            for section, values in file_config.items():
                if section in self._config and isinstance(values, dict):
                    self._config[section].update(values)
    
    def get(self, component: str, param: str = None, default: Any = None) -> Any:
        """Get configuration value."""
        if component not in self._config:
            return default
        if param is None:
            return self._config[component]
        return self._config[component].get(param, default)
    
    def set(self, component: str, param: str, value: Any):
        """Set configuration value."""
        if component not in self._config:
            self._config[component] = {}
        self._config[component][param] = value
    
    def __getitem__(self, key: str) -> Dict[str, Any]:
        """Get component config by name."""
        return self._config.get(key, {})
    
    def get_model_info(self, model_type: str, model_name: str) -> Dict:
        """Get model info from registry."""
        registry_key = f"{model_type}_models"
        if registry_key in self._config:
            return self._config[registry_key].get(model_name, {})
        return {}
    
    def __repr__(self) -> str:
        return f"PipelineConfig({list(self._config.keys())})"


# =============================================================================
# Factory function
# =============================================================================

def create_pipeline(config: Union[str, PipelineConfig, Dict] = None,
                   stages: List[str] = None) -> Pipeline:
    """
    Create a pipeline from configuration.
    
    Args:
        config: Config path, PipelineConfig, or dict
        stages: List of stages to include (default: all)
        
    Returns:
        Configured Pipeline instance
    """
    # Note: Import modules lazily to avoid circular imports
    # Each module class should be imported here when creating pipeline
    
    # Load config
    if config is None:
        cfg = PipelineConfig()
    elif isinstance(config, str):
        cfg = PipelineConfig(config)
    elif isinstance(config, dict):
        cfg = PipelineConfig(**config)
    else:
        cfg = config
    
    # Available stages
    all_stages = [
        'document_processor', 'chunker', 'embedding', 'reranker',
        'domain_extractor', 'context_builder', 'qa_generator',
        'deduplication', 'evaluation'
    ]
    
    if stages is None:
        stages = all_stages
    
    # Lazy imports to avoid circular dependencies
    def _get_module(stage_name: str) -> BaseModule:
        if stage_name == 'document_processor':
            from mirage.pipeline.pdf_processor import DocumentProcessor
            return DocumentProcessor(**cfg['document_processor'])
        elif stage_name == 'chunker':
            from mirage.pipeline.chunker import SemanticChunker
            return SemanticChunker(**cfg['chunker'])
        elif stage_name == 'embedding':
            from mirage.embeddings.models import EmbeddingModule
            return EmbeddingModule(**cfg['embedding'])
        elif stage_name == 'reranker':
            from mirage.embeddings.rerankers_multimodal import RerankerModule
            return RerankerModule(**cfg['reranker'])
        elif stage_name == 'domain_extractor':
            from mirage.pipeline.domain import DomainExtractor
            return DomainExtractor(**cfg['domain_extractor'])
        elif stage_name == 'context_builder':
            from mirage.pipeline.context import ContextBuilder
            return ContextBuilder(**cfg['context_builder'])
        elif stage_name == 'qa_generator':
            from mirage.pipeline.qa_generator import QAGenerator
            return QAGenerator(**cfg['qa_generator'])
        elif stage_name == 'deduplication':
            from mirage.pipeline.deduplication import Deduplicator
            return Deduplicator(**cfg['deduplication'])
        elif stage_name == 'evaluation':
            from mirage.evaluation.metrics_optimized import Evaluator
            return Evaluator(**cfg['evaluation'])
        else:
            raise ValueError(f"Unknown stage: {stage_name}")
    
    steps = []
    for stage in stages:
        if stage in all_stages:
            steps.append((stage, _get_module(stage)))
    
    return Pipeline(steps)
