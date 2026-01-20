#!/usr/bin/env python3
"""
QA Dataset Generation Pipeline (FULLY PARALLELIZED)
From PDFs -> Semantic Chunks -> Embed All -> Multihop QA -> Deduplication

Configuration: Loads from config.yaml if available, otherwise uses defaults below.
"""

import json
import os
import logging
import numpy as np
import faiss
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

# Package imports
from mirage.core.llm import setup_logging, BACKEND, LLM_MODEL_NAME, VLM_MODEL_NAME, GEMINI_RPM, GEMINI_BURST
from mirage.utils.stats import compute_dataset_stats, print_dataset_stats, compute_qa_category_stats, print_qa_category_stats
from mirage.utils.preflight import run_preflight_checks
from mirage.utils.checkpoint import CheckpointManager
from mirage.utils.llm_cache import init_llm_cache, get_llm_cache

# Global checkpoint manager (initialized in run_pipeline)
_CHECKPOINT_MANAGER = None

# ============================================================================
# CONFIGURATION - Load from config.yaml if available
# ============================================================================
try:
    from mirage.core.config import (
        load_config, get_paths_config, get_processing_config, 
        get_parallel_config, get_retrieval_config, get_embedding_config,
        get_evaluation_config, get_qa_correction_config, get_qa_generation_config,
        get_faiss_config, get_deduplication_config
    )
    
    _cfg = load_config()
    _paths = get_paths_config()
    _proc = get_processing_config()
    _parallel = get_parallel_config()
    _retrieval = get_retrieval_config()
    _embed = get_embedding_config()
    _eval = get_evaluation_config()
    _shuffle = _cfg.get('shuffling', {})
    _qa_correction = get_qa_correction_config()
    _qa_gen = get_qa_generation_config()
    _dedup = get_deduplication_config()
    
    # Input/Output paths
    INPUT_PDF_DIR = _paths.get('input_pdf_dir', "data/documents")
    INPUT_CHUNKS_FILE = _paths.get('input_chunks_file', None)
    OUTPUT_DIR = _paths.get('output_dir', "output/results")
    
    # Processing limits
    MAX_PDFS = _proc.get('max_pdfs', 5)
    SORT_BY_SIZE = _proc.get('sort_by_size', True)
    MAX_CHUNKS = _proc.get('max_chunks', None)
    
    # Shuffling
    SHUFFLE = _shuffle.get('enabled', True)
    SHUFFLE_SEED = _shuffle.get('seed', 42)
    
    # Embedding
    EMBEDDING_MODEL = _embed.get('model', "nomic")
    CACHE_EMBEDDINGS = _embed.get('cache_embeddings', True)
    EMBED_BATCH_SIZE = _embed.get('batch_size', 16)
    EMBEDDING_GPUS = _embed.get('gpus', None)  # GPUs for embedding model
    
    # Parallel processing
    NUM_WORKERS = _parallel.get('num_workers', 3)
    AVAILABLE_GPUS = _parallel.get('available_gpus', [2, 3, 4])
    QA_MAX_WORKERS = _parallel.get('qa_max_workers', 6)
    DEDUP_MAX_WORKERS = _parallel.get('dedup_max_workers', 4)
    
    # Context retrieval
    _multihop = _retrieval.get('multihop', {})
    # SAFETY: Limit depth and breadth to prevent runaway API calls
    # Each iteration can generate depth x breadth x chunks_per_search API calls
    MAX_DEPTH = min(_multihop.get('max_depth') or 5, 20)  # Default 5, max 20
    MAX_BREADTH = min(_multihop.get('max_breadth') or 5, 10)  # Default 5, max 10
    CHUNKS_PER_SEARCH = _multihop.get('chunks_per_search', 2)
    CHUNK_ADDITION_MODE = _multihop.get('chunk_addition_mode', 'RELATED')  # EXPLANATORY or RELATED
    
    # Evaluation
    RUN_EVALUATION = _eval.get('run_evaluation', True)
    EVAL_SAMPLE_SIZE = _eval.get('sample_size', None)
    GEMINI_API_KEY_PATH = _eval.get('gemini_api_key_path', os.path.expanduser("~/.config/gemini/api_key.txt"))
    USE_OPTIMIZED_METRICS = _eval.get('use_optimized_metrics', True)  # 3-5x faster evaluation
    
    # QA Correction
    QA_CORRECTION_ENABLED = _qa_correction.get('enabled', True)
    QA_CORRECTION_MAX_ATTEMPTS = _qa_correction.get('max_attempts', 1)
    
    # QA Generation Control
    QA_NUM_PAIRS = _qa_gen.get('num_qa_pairs', 1000)  # Target number of QA pairs
    QA_TYPE = _qa_gen.get('type', 'multihop')  # Type: 'multihop', 'multimodal', 'text', 'mix'
    
    # FAISS config
    _faiss = get_faiss_config()
    FAISS_USE_GPU = _faiss.get('use_gpu', False)
    FAISS_GPU_ID = _faiss.get('gpu_id', 0)
    
    # Deduplication config
    RUN_DEDUPLICATION = _dedup.get('enabled', True)
    
    print("Configuration loaded from config.yaml")
    
except (ImportError, Exception):
    print("config.yaml not available, using default configuration")
    
    # Default configuration (fallback)
    INPUT_PDF_DIR = "data/documents"
    INPUT_CHUNKS_FILE = None
    OUTPUT_DIR = "output/results"
    
    # SAFETY: Reasonable defaults to prevent runaway API calls
    # Each iteration can generate depth x breadth x chunks_per_search API calls
    MAX_DEPTH = 5  # Maximum 5 iterations per chunk
    MAX_BREADTH = 5  # Maximum 5 search strings per verification
    CHUNKS_PER_SEARCH = 2
    CHUNK_ADDITION_MODE = "RELATED"  # EXPLANATORY or RELATED
    
    MAX_PDFS = 5
    SORT_BY_SIZE = True
    MAX_CHUNKS = None
    
    SHUFFLE = True
    SHUFFLE_SEED = 42
    
    EMBEDDING_MODEL = "nomic"
    CACHE_EMBEDDINGS = True
    EMBED_BATCH_SIZE = 16
    EMBEDDING_GPUS = None
    
    NUM_WORKERS = 3
    AVAILABLE_GPUS = [2, 3, 4]
    QA_MAX_WORKERS = 6
    DEDUP_MAX_WORKERS = 4
    
    RUN_EVALUATION = True
    EVAL_SAMPLE_SIZE = None
    GEMINI_API_KEY_PATH = os.path.expanduser("~/.config/gemini/api_key.txt")
    USE_OPTIMIZED_METRICS = True  # 3-5x faster evaluation
    
    # Deduplication default
    RUN_DEDUPLICATION = True
    
    # QA Correction defaults
    QA_CORRECTION_ENABLED = True
    QA_CORRECTION_MAX_ATTEMPTS = 1
    
    # QA Generation Control defaults
    QA_NUM_PAIRS = 1000  # Default: 1000 QA pairs
    QA_TYPE = "multihop"  # Default: multihop QA pairs
    
    # FAISS defaults
    FAISS_USE_GPU = False
    FAISS_GPU_ID = 0

# Derived output paths
OUTPUT_CHUNKS = os.path.join(OUTPUT_DIR, "chunks.json")
OUTPUT_EMBEDDINGS_DIR = os.path.join(OUTPUT_DIR, "embeddings")
OUTPUT_QA_SUCCESSFUL = os.path.join(OUTPUT_DIR, "qa_multihop_pass.json")
OUTPUT_QA_FAILED = os.path.join(OUTPUT_DIR, "qa_multihop_fail.json")
OUTPUT_MALFORMED_CHUNKS = os.path.join(OUTPUT_DIR, "malformed_chunks.json")
OUTPUT_QA_DEDUPLICATED = os.path.join(OUTPUT_DIR, "qa_deduplicated.json")
OUTPUT_CHUNKS_WITH_CONTEXT = os.path.join(OUTPUT_DIR, "chunks_with_complete_context.json")
OUTPUT_EVAL_REPORT = os.path.join(OUTPUT_DIR, "subset_evaluation_report.json")

# ============================================================================
# GLOBAL MODEL & EMBEDDING CACHE (Load once, reuse throughout pipeline)
# ============================================================================
_MODEL_CACHE = {}  # Generic cache keyed by model identifier
_GPU_LOCK = None  # Threading lock for GPU access (initialized in main)

_EMBEDDING_CACHE = {
    'chunk_embeddings': None,      # Full chunk embeddings array [N, dim]
    'chunk_embeddings_index': None, # FAISS index for chunks
    'chunk_ids': None,              # List of chunk IDs matching embeddings
    'question_embeddings': None,    # Question embeddings for deduplication
    'answer_embeddings': None,      # Answer embeddings for deduplication
    'qa_data': None                 # Original QA data for deduplication
}

def init_gpu_lock():
    """Initialize GPU lock for thread-safe GPU access"""
    global _GPU_LOCK
    import threading
    _GPU_LOCK = threading.Lock()

def get_reranker():
    """Get or create VLM reranker using Gemini API (cached if CACHE_EMBEDDINGS=True)"""
    if CACHE_EMBEDDINGS:
        if 'reranker' not in _MODEL_CACHE or _MODEL_CACHE['reranker'] is None:
            print(f"Loading VLM reranker (uses {BACKEND} API)...")
            from mirage.embeddings.rerankers_multimodal import GeminiVLMReranker
            _MODEL_CACHE['reranker'] = GeminiVLMReranker()
            print(f"VLM reranker loaded and cached")
        return _MODEL_CACHE['reranker']
    else:
        from mirage.embeddings.rerankers_multimodal import GeminiVLMReranker
        return GeminiVLMReranker()

def _load_sentence_transformer(model_name: str):
    """Load SentenceTransformer safely, avoiding meta tensor issues"""
    from sentence_transformers import SentenceTransformer
    import torch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load to CPU first to avoid meta tensor issues, then move to GPU
    # Some models use device_map internally which creates meta tensors
    try:
        model = SentenceTransformer(model_name, device='cpu')
        if device == 'cuda':
            model = model.to(device)
        return model
    except Exception as e:
        # Fallback: try direct loading with device
        print(f"CPU-first loading failed ({e}), trying direct load...")
        return SentenceTransformer(model_name, device=device)

def get_embedder(model_name: Optional[str] = None, use_multimodal: bool = True):
    """Get or create embedding model (cached if CACHE_EMBEDDINGS=True)
    
    Args:
        model_name: Specific model to load, or None to use config/auto-detect
        use_multimodal: If True and model_name is None, try multimodal models first
                       (Qwen3-VL → Nomic → bge-m3 fallback)
    """
    global EMBEDDING_MODEL
    
    if model_name is None:
        model_name = EMBEDDING_MODEL
    
    # If requesting multimodal auto-detection
    if use_multimodal and model_name in ["auto", "multimodal", None, ""]:
        cache_key = "_multimodal_auto"
        if CACHE_EMBEDDINGS and cache_key in _MODEL_CACHE and _MODEL_CACHE[cache_key] is not None:
            return _MODEL_CACHE[cache_key]
        
        from mirage.embeddings.models import get_multimodal_embedder
        embedder, actual_model, is_multimodal = get_multimodal_embedder(gpus=EMBEDDING_GPUS)
        EMBEDDING_MODEL = actual_model  # Update global for consistent naming
        
        if CACHE_EMBEDDINGS:
            _MODEL_CACHE[cache_key] = embedder
            _MODEL_CACHE[actual_model] = embedder
            print(f"{actual_model} model loaded and cached (multimodal={is_multimodal})")
        return embedder
    
    if CACHE_EMBEDDINGS:
        if model_name not in _MODEL_CACHE or _MODEL_CACHE[model_name] is None:
            print(f"Loading embedding model: {model_name}...")
            
            if model_name == "qwen3_vl":
                from mirage.embeddings.models import Qwen3VLEmbed
                _MODEL_CACHE[model_name] = Qwen3VLEmbed(gpus=EMBEDDING_GPUS)
            elif model_name == "nomic":
                from mirage.embeddings.models import NomicVLEmbed as NomicEmbedder
                _MODEL_CACHE[model_name] = NomicEmbedder(gpus=EMBEDDING_GPUS)
            elif model_name in ["bge_m3", "BAAI/bge-m3"]:
                from mirage.embeddings.models import get_best_embedding_model
                actual_model_name = get_best_embedding_model() if model_name == "bge_m3" else model_name
                _MODEL_CACHE[model_name] = _load_sentence_transformer(actual_model_name)
            else:
                if '/' in model_name:
                    _MODEL_CACHE[model_name] = _load_sentence_transformer(model_name)
                else:
                    raise ValueError(f"Unknown embedding model: {model_name}")
            print(f"{model_name} model loaded and cached")
        return _MODEL_CACHE[model_name]
    else:
        if model_name == "qwen3_vl":
            from mirage.embeddings.models import Qwen3VLEmbed
            return Qwen3VLEmbed(gpus=EMBEDDING_GPUS)
        elif model_name == "nomic":
            from mirage.embeddings.models import NomicVLEmbed as NomicEmbedder
            return NomicEmbedder(gpus=EMBEDDING_GPUS)
        elif model_name in ["bge_m3", "BAAI/bge-m3"]:
            from mirage.embeddings.models import get_best_embedding_model
            actual_model_name = get_best_embedding_model() if model_name == "bge_m3" else model_name
            return _load_sentence_transformer(actual_model_name)
        else:
            if '/' in model_name:
                return _load_sentence_transformer(model_name)
            else:
                raise ValueError(f"Unknown embedding model: {model_name}")

def get_nomic_embedder():
    """DEPRECATED: use get_embedder()"""
    return get_embedder("nomic")

def get_sentence_transformer():
    """DEPRECATED: use get_embedder('bge_m3')"""
    return get_embedder("bge_m3")


# ============================================================================
# STEP 1: PDF -> MARKDOWN -> SEMANTIC CHUNKS (PARALLEL)
# ============================================================================
def _convert_single_document(args: Tuple[str, str, dict, int]) -> Tuple[str, str, str]:
    """Worker function to convert a single document (PDF or HTML) to markdown (for parallel processing)
    
    Args:
        args: Tuple of (doc_file_str, output_dir_str, pipeline_options_dict, gpu_id)
    
    Returns:
        Tuple of (doc_name, md_file_path, error_message) - error_message is None on success
    """
    doc_file_str, output_dir_str, pipeline_options_dict, gpu_id = args
    
    # Convert strings back to Path objects inside worker (avoids serialization issues)
    from pathlib import Path
    doc_file = Path(doc_file_str)
    output_dir = Path(output_dir_str)
    
    # Determine if this is a PDF or HTML file
    is_pdf = doc_file.suffix.lower() == '.pdf'
    
    try:
        # Set CUDA_VISIBLE_DEVICES BEFORE importing torch/docling to isolate this worker
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Import torch and clear any cached memory
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Now import docling AFTER setting CUDA_VISIBLE_DEVICES
        from mirage.pipeline.pdf_processor import configure_pipeline_options, annotate_items_with_images, create_multi_format_converter
        from docling.document_converter import DocumentConverter, PdfFormatOption, HTMLFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling_core.types.doc import ImageRefMode, TableItem
        
        print(f"DEBUG: Worker using CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"DEBUG: Processing {'PDF' if is_pdf else 'HTML'} file: {doc_file.name}")
        
        if is_pdf:
            # Recreate pipeline options in worker process for PDFs
            # Use cuda:0 since CUDA_VISIBLE_DEVICES restricts to single GPU
            pipeline_options = configure_pipeline_options(cuda_device_id=0)
            # Disable built-in picture description; annotate_items_with_images handles it
            # and properly skips pictures inside tables via get_pictures_inside_tables()
            pipeline_options.do_picture_description = False
            doc_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                    InputFormat.HTML: HTMLFormatOption(),  # Support both formats
                }
            )
        else:
            # For HTML, use simpler converter (no GPU-heavy processing needed)
            doc_converter = DocumentConverter(
                format_options={
                    InputFormat.HTML: HTMLFormatOption(),
                }
            )
        
        # Convert document to markdown
        conv_res = doc_converter.convert(str(doc_file))
        doc_filename = conv_res.input.file.stem
        
        # Save markdown with referenced images
        md_output_dir = output_dir / "markdown" / doc_file.stem
        md_output_dir.mkdir(parents=True, exist_ok=True)
        md_file = md_output_dir / f"{doc_file.stem}_ref.md"
        
        # For PDFs: annotate images and save table images
        if is_pdf:
            # Annotate all images and tables using unified function
            pictures_to_skip = annotate_items_with_images(conv_res)
            
            # Create organized folders for images (matching pdf_processor structure)
            tables_dir = md_output_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            
            # Save images of tables (pictures are saved by save_as_markdown with REFERENCED mode)
            table_counter = 0
            for element, _level in conv_res.document.iterate_items():
                if isinstance(element, TableItem):
                    table_counter += 1
                    element_image_filename = tables_dir / f"{doc_filename}-table-{table_counter}.png"
                    try:
                        with element_image_filename.open("wb") as fp:
                            element.get_image(conv_res.document).save(fp, "PNG")
                    except Exception as e:
                        print(f"Warning: Could not save table image: {e}")
        
        # Create artifacts directory (relative path like in pdf_processor.py)
        artifacts_dir = Path("ref_artifacts")
        (md_output_dir / "ref_artifacts").mkdir(parents=True, exist_ok=True)
        
        # Save markdown with externally referenced pictures (pass relative Path object)
        conv_res.document.save_as_markdown(
            md_file, 
            image_mode=ImageRefMode.REFERENCED,
            artifacts_dir=artifacts_dir
        )
        
        md_file_abs = str(md_file)
        file_type = "PDF" if is_pdf else "HTML"
        print(f"Exported {file_type} markdown file saved to: {md_file_abs}")
        
        return (doc_file.name, str(md_file), None)
        
    except Exception as e:
        return (doc_file.name, None, str(e))

# Backward compatibility alias
def _convert_single_pdf(args: Tuple[str, str, dict, int]) -> Tuple[str, str, str]:
    """Backward compatibility wrapper for _convert_single_document."""
    return _convert_single_document(args)


def _chunk_single_markdown(args: Tuple[str, str]) -> Tuple[str, List[dict], str]:
    """Worker function to chunk a single markdown file (for parallel processing)
    
    Returns:
        Tuple of (file_stem, chunks, error_message)
    """
    md_file_path, file_stem = args
    
    try:
        from mirage.pipeline.chunker import chunk_with_windows, renumber_chunks
        
        md_file = Path(md_file_path)
        markdown_text = md_file.read_text(encoding='utf-8')
        chunks, _ = chunk_with_windows(markdown_text)
        chunks = renumber_chunks(chunks, file_stem)
        
        return (file_stem, chunks, None)
        
    except Exception as e:
        return (file_stem, [], str(e))


def convert_documents_to_chunks_parallel(doc_dir: str, output_chunks_file: str, 
                                         checkpoint_mgr: CheckpointManager = None) -> list:
    """Convert document files (PDF and HTML) to semantic chunks via markdown (PARALLEL)
    
    Supports checkpoint resume:
    - Resumes markdown conversion from last checkpoint
    - Resumes chunking from last checkpoint
    - Saves progress after each file
    """
    from mirage.pipeline.pdf_processor import SUPPORTED_EXTENSIONS
    from mirage.pipeline.chunker import renumber_chunks
    
    doc_dir = Path(doc_dir)
    output_dir = Path(OUTPUT_DIR)
    markdown_dir = output_dir / "markdown"
    
    # Use global checkpoint manager if not provided
    if checkpoint_mgr is None:
        checkpoint_mgr = _CHECKPOINT_MANAGER
    
    # =========================================================================
    # CHECK FOR CHECKPOINTED CHUNKS (Resume from chunks checkpoint)
    # =========================================================================
    if checkpoint_mgr and checkpoint_mgr.is_final_chunks_saved():
        # Load from checkpoint file instead of output file
        saved_chunks = checkpoint_mgr.get_all_file_chunks()
        if saved_chunks:
            all_chunks = []
            for chunks in saved_chunks.values():
                all_chunks.extend(chunks)
            print(f"📂 Resuming from chunks checkpoint: {len(all_chunks)} chunks from {len(saved_chunks)} files")
            return all_chunks
    
    # =========================================================================
    # CHECK FOR EXISTING/CHECKPOINTED MARKDOWN FILES
    # =========================================================================
    completed_md_stems = set()
    if checkpoint_mgr:
        completed_md_stems = checkpoint_mgr.get_completed_markdown_files()
    
    existing_md_files = []
    if markdown_dir.exists():
        for subdir in markdown_dir.iterdir():
            if subdir.is_dir():
                ref_files = list(subdir.glob("*_ref.md"))
                if ref_files:
                    for ref_file in ref_files:
                        file_stem = subdir.name
                        existing_md_files.append((str(ref_file), file_stem))
                        completed_md_stems.add(file_stem)
    
    # Collect all document files
    doc_files = []
    for ext in SUPPORTED_EXTENSIONS:
        doc_files.extend(doc_dir.glob(f"*{ext}"))
    
    if not doc_files and not existing_md_files:
        print(f"No document files found in {doc_dir}")
        print(f"   Supported formats: {SUPPORTED_EXTENSIONS}")
        return []
    
    # Filter out already-converted documents
    pending_docs = [f for f in doc_files if f.stem not in completed_md_stems]
    
    if existing_md_files:
        print(f"\n📂 Found {len(existing_md_files)} existing markdown files")
        for md_path, stem in existing_md_files:
            print(f"   ✅ {stem}")
    
    md_files = existing_md_files.copy()
    
    # =========================================================================
    # PHASE 1: Convert remaining documents to Markdown
    # =========================================================================
    if pending_docs:
        # Sort by file size if enabled
        if SORT_BY_SIZE:
            pending_docs = sorted(pending_docs, key=lambda f: f.stat().st_size)
        
        # Limit for trial run
        if MAX_PDFS is not None:
            # Account for already completed files
            remaining_slots = MAX_PDFS - len(completed_md_stems)
            if remaining_slots > 0:
                pending_docs = pending_docs[:remaining_slots]
            else:
                pending_docs = []
        
        if pending_docs:
            pdf_count = sum(1 for f in pending_docs if f.suffix.lower() == '.pdf')
            html_count = sum(1 for f in pending_docs if f.suffix.lower() in {'.html', '.htm', '.xhtml'})
            print(f"\nPhase 1: Converting {len(pending_docs)} documents to Markdown (PDF: {pdf_count}, HTML: {html_count})")
            print(f"Using {NUM_WORKERS} parallel workers on GPUs {AVAILABLE_GPUS}")
            
            output_dir_str = str(output_dir.resolve())
            doc_args = [(str(doc_file.resolve()), output_dir_str, {}, AVAILABLE_GPUS[i % len(AVAILABLE_GPUS)]) 
                       for i, doc_file in enumerate(pending_docs)]
            
            ctx = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=NUM_WORKERS, mp_context=ctx) as executor:
                futures = {executor.submit(_convert_single_document, args): Path(args[0]).name for args in doc_args}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Converting documents"):
                    doc_name = futures[future]
                    try:
                        name, md_file_path, error = future.result()
                        file_stem = Path(name).stem
                        if error:
                            print(f"  ❌ Error {name}: {error}")
                            if checkpoint_mgr:
                                checkpoint_mgr.mark_markdown_failed(file_stem, error)
                        else:
                            md_files.append((md_file_path, file_stem))
                            print(f"  ✅ {name} -> markdown")
                            # Checkpoint after each successful conversion
                            if checkpoint_mgr:
                                checkpoint_mgr.mark_markdown_complete(file_stem, md_file_path)
                    except Exception as e:
                        print(f"  ❌ Error {doc_name}: {e}")
                        if checkpoint_mgr:
                            checkpoint_mgr.mark_markdown_failed(Path(doc_name).stem, str(e))
    
    if not md_files:
        print("No documents converted successfully")
        return []
    
    # =========================================================================
    # PHASE 2: Chunk Markdown files (with checkpointing)
    # =========================================================================
    completed_chunk_files = set()
    if checkpoint_mgr:
        completed_chunk_files = checkpoint_mgr.get_completed_chunk_files()
        saved_chunks = checkpoint_mgr.get_all_file_chunks()
    else:
        saved_chunks = {}
    
    # Filter out already-chunked files
    pending_md_files = [(path, stem) for path, stem in md_files if stem not in completed_chunk_files]
    
    print(f"\nPhase 2: Chunking markdown files...")
    if completed_chunk_files:
        print(f"   ✅ {len(completed_chunk_files)} files already chunked (from checkpoint)")
    if pending_md_files:
        print(f"   ⏳ {len(pending_md_files)} files to chunk")
    
    # Process pending files
    if pending_md_files:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(_chunk_single_markdown, args): args[1] for args in pending_md_files}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Chunking"):
                file_stem = futures[future]
                try:
                    name, chunks, error = future.result()
                    if error:
                        print(f"  ❌ Error {name}: {error}")
                    else:
                        saved_chunks[file_stem] = chunks
                        print(f"  ✅ {name}: {len(chunks)} chunks")
                        # Checkpoint after each file is chunked
                        if checkpoint_mgr:
                            checkpoint_mgr.save_file_chunks(file_stem, chunks)
                except Exception as e:
                    print(f"  ❌ Error {file_stem}: {e}")
    
    # =========================================================================
    # PHASE 3: Combine and renumber all chunks
    # =========================================================================
    print(f"\nPhase 3: Combining and renumbering chunks from {len(saved_chunks)} files...")
    
    all_chunks = []
    chunk_index = 1
    for file_stem in sorted(saved_chunks.keys()):
        file_chunks = saved_chunks[file_stem]
        # Renumber chunks with global index
        for chunk in file_chunks:
            chunk['global_chunk_id'] = f"chunk_{chunk_index}"
            chunk_index += 1
        all_chunks.extend(file_chunks)
    
    # Save final chunks file
    os.makedirs(os.path.dirname(output_chunks_file), exist_ok=True)
    with open(output_chunks_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    # Mark final checkpoint
    if checkpoint_mgr:
        checkpoint_mgr.mark_final_chunks_saved()
    
    print(f"💾 Saved {len(all_chunks)} total chunks to {output_chunks_file}")
    return all_chunks

# Backward compatibility alias
def convert_pdfs_to_chunks_parallel(pdf_dir: str, output_chunks_file: str) -> list:
    """Backward compatibility wrapper for convert_documents_to_chunks_parallel."""
    return convert_documents_to_chunks_parallel(pdf_dir, output_chunks_file)


def load_chunks(chunks_file: str) -> list:
    """Load pre-existing chunks from JSON"""
    print(f"Loading chunks from {chunks_file}...")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")
    return chunks


# ============================================================================
# STEP 2: EMBED ALL CHUNKS & BUILD UNIFIED FAISS INDEX (BATCHED GPU)
# ============================================================================
def embed_all_chunks(chunks: list, chunks_file: str, embeddings_dir: str) -> tuple:
    """Embed all chunks and create a unified FAISS index (BATCHED for GPU efficiency)"""
    import torch
    import gc
    import glob
    
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Check for ANY existing embeddings (handles "auto" model name case)
    existing_indices = glob.glob(os.path.join(embeddings_dir, "*_index.faiss"))
    existing_metadata = glob.glob(os.path.join(embeddings_dir, "*_metadata.json"))
    
    if existing_indices and existing_metadata:
        # Use the first found index (there should typically be only one)
        index_path = existing_indices[0]
        metadata_path = existing_metadata[0]
        
        print(f"📂 Found existing embeddings: {os.path.basename(index_path)}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        chunk_ids = metadata['chunk_ids']
        
        # Verify chunk count matches
        if len(chunk_ids) == len(chunks):
            if CACHE_EMBEDDINGS and _EMBEDDING_CACHE['chunk_embeddings'] is not None:
                embeddings_array = _EMBEDDING_CACHE['chunk_embeddings']
                print(f"   Using cached embeddings from memory ({len(chunk_ids)} chunks)")
            else:
                print(f"   Index exists but embeddings not in cache - will recompute for topic modeling")
                embeddings_array = None
            return embeddings_dir, embeddings_array, chunk_ids
        else:
            print(f"   ⚠️ Chunk count mismatch ({len(chunk_ids)} in index vs {len(chunks)} current) - re-embedding")
    
    print(f"\nEmbedding {len(chunks)} chunks (batch_size={EMBED_BATCH_SIZE})...")
    
    embedder = get_embedder()
    
    # Ensure model is fully on GPU before processing
    if hasattr(embedder, 'model'):
        embedder.model.eval()
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    embeddings = []
    chunk_ids = []
    current_batch_size = EMBED_BATCH_SIZE  # Adaptive batch size
    
    # Get embedding dimension for fallback zero embeddings
    def get_embedding_dim():
        if hasattr(embedder, 'get_sentence_embedding_dimension'):
            return embedder.get_sentence_embedding_dimension()
        elif hasattr(embedder, 'embedding_dim'):
            return embedder.embedding_dim
        elif hasattr(embedder, 'model') and hasattr(embedder.model, 'config') and hasattr(embedder.model.config, 'hidden_size'):
            return embedder.model.config.hidden_size
        return 768  # Default fallback
    
    def embed_batch_adaptive(batch_texts, batch_chunk_ids, batch_size):
        """Embed a batch with adaptive sizing on OOM."""
        nonlocal current_batch_size
        import time
        
        if batch_size <= 0:
            # Fall back to individual processing with aggressive memory management
            results = []
            for idx, text in enumerate(batch_texts):
                max_retries = 3
                wait_time = 2  # seconds between retries
                
                for retry in range(max_retries):
                    try:
                        # Aggressive memory cleanup before each embedding
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        
                        # Use unified .encode() API
                        embedding = embedder.encode(text, convert_to_numpy=True)
                        if isinstance(embedding, torch.Tensor):
                            embedding = embedding.cpu().float().numpy()
                        results.append(embedding)
                        
                        # Cleanup after success
                        gc.collect()
                        torch.cuda.empty_cache()
                        break
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() and retry < max_retries - 1:
                            logging.warning(f"OOM on individual embed (attempt {retry+1}/{max_retries}), waiting {wait_time}s...")
                            gc.collect()
                            torch.cuda.empty_cache()
                            time.sleep(wait_time)
                            wait_time *= 2  # Exponential backoff
                        else:
                            logging.error(f"Individual embedding failed after retries: {e}")
                            results.append(np.zeros(get_embedding_dim(), dtype=np.float32))
                            break
                    except Exception as e:
                        logging.error(f"Individual embedding failed: {e}")
                        results.append(np.zeros(get_embedding_dim(), dtype=np.float32))
                        break
            return results
        
        try:
            torch.cuda.empty_cache()
            
            # Use unified .encode() API for all embedders
            with torch.no_grad():
                batch_embeddings = embedder.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
            results = [emb for emb in batch_embeddings]
            
            # Clear GPU cache after each batch
            torch.cuda.empty_cache()
            return results
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Reduce batch size and retry
                new_batch_size = max(1, batch_size // 2)
                if new_batch_size < batch_size:
                    logging.warning(f"OOM: Reducing batch size {batch_size} -> {new_batch_size}")
                    current_batch_size = new_batch_size
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    
                    # Process in smaller batches
                    results = []
                    for i in range(0, len(batch_texts), new_batch_size):
                        sub_batch = batch_texts[i:i + new_batch_size]
                        results.extend(embed_batch_adaptive(sub_batch, batch_chunk_ids[i:i + new_batch_size], new_batch_size))
                    return results
                else:
                    # Already at batch_size=1, fall back to individual
                    return embed_batch_adaptive(batch_texts, batch_chunk_ids, 0)
            else:
                raise
    
    # Process in batches with adaptive sizing
    batch_start = 0
    pbar = tqdm(total=len(chunks), desc="Embedding chunks")
    
    while batch_start < len(chunks):
        batch_end = min(batch_start + current_batch_size, len(chunks))
        batch_chunks = chunks[batch_start:batch_end]
        
        batch_texts = []
        batch_chunk_ids = []
        for i, chunk in enumerate(batch_chunks):
            content = chunk.get('content', str(chunk)) if isinstance(chunk, dict) else str(chunk)
            chunk_id = chunk.get('chunk_id', str(batch_start + i + 1)) if isinstance(chunk, dict) else str(batch_start + i + 1)
            batch_texts.append(content)
            batch_chunk_ids.append(chunk_id)
        
        batch_embeddings = embed_batch_adaptive(batch_texts, batch_chunk_ids, current_batch_size)
        embeddings.extend(batch_embeddings)
        chunk_ids.extend(batch_chunk_ids)
        
        pbar.update(len(batch_chunks))
        batch_start = batch_end
    
    pbar.close()
    
    # Clear GPU memory before stacking
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    
    embeddings_array = np.vstack(embeddings).astype('float32')
    faiss.normalize_L2(embeddings_array)
    
    dim = embeddings_array.shape[1]
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(embeddings_array)
    
    # Transfer to GPU if enabled
    if FAISS_USE_GPU:
        try:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, FAISS_GPU_ID, cpu_index)
            index = gpu_index
            print(f"✅ FAISS index transferred to GPU {FAISS_GPU_ID}")
        except Exception as e:
            print(f"⚠️ Failed to transfer FAISS index to GPU: {e}")
            print("   Falling back to CPU index")
            index = cpu_index
    else:
        index = cpu_index
    
    if CACHE_EMBEDDINGS:
        _EMBEDDING_CACHE['chunk_embeddings'] = embeddings_array
        _EMBEDDING_CACHE['chunk_embeddings_index'] = index
        _EMBEDDING_CACHE['chunk_ids'] = chunk_ids
        # Store GPU resources to prevent garbage collection
        if FAISS_USE_GPU and 'res' in dir():
            _EMBEDDING_CACHE['gpu_resources'] = res
        print(f"Cached {len(chunk_ids)} chunk embeddings in memory")
    
    # Always save CPU index to disk (GPU index can't be serialized)
    faiss.write_index(cpu_index, index_path)
    print(f"Saved FAISS index to {index_path}")
    
    metadata = {
        'chunk_ids': chunk_ids,
        'chunks_file': chunks_file,
        'model': EMBEDDING_MODEL,
        'dimension': dim,
        'count': len(chunk_ids)
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return embeddings_dir, embeddings_array, chunk_ids


def configure_retrieval_paths(chunks_file: str, embeddings_dir: str, image_base_dir: str = None):
    """Configure context module to use unified embeddings and cached embedder/embeddings"""
    from mirage.pipeline import context as context_module
    import sys
    
    context_module.CHUNKS_FILE = chunks_file
    context_module.EMBEDDINGS_DIR = embeddings_dir
    if image_base_dir:
        context_module.IMAGE_BASE_DIR = image_base_dir
    
    context_retrieved_module = sys.modules['mirage.pipeline.context']
    context_retrieved_module._cached_embedder = get_embedder() if CACHE_EMBEDDINGS else None
    context_retrieved_module._gpu_lock = _GPU_LOCK  # Pass GPU lock for thread-safe access
    
    # VLMReranker uses API calls (Gemini), no local GPU needed - keep embedder on GPU
    context_retrieved_module._cached_reranker = get_reranker() if CACHE_EMBEDDINGS else None
    if CACHE_EMBEDDINGS:
        context_retrieved_module._cached_chunk_embeddings = _EMBEDDING_CACHE['chunk_embeddings']
        context_retrieved_module._cached_chunk_index = _EMBEDDING_CACHE['chunk_embeddings_index']
        context_retrieved_module._cached_chunk_ids = _EMBEDDING_CACHE['chunk_ids']
        print(f"Configured retrieval with cached embeddings + reranker + GPU lock")
    else:
        context_retrieved_module._cached_chunk_embeddings = None
        context_retrieved_module._cached_chunk_index = None
        context_retrieved_module._cached_chunk_ids = None


# ============================================================================
# STEP 3: EXTRACT DOMAIN AND EXPERT PERSONA
# ============================================================================
def get_domain_and_expert(chunks_file: str, embeddings: Optional[np.ndarray] = None, chunk_ids: Optional[list] = None) -> tuple:
    """Get domain and expert persona - from config, environment, or auto-detect via BERTopic.
    
    Priority:
    1. Config file (config.yaml domain_expert section)
    2. Environment variables (DATASET_DOMAIN, DATASET_EXPERT_PERSONA)
    3. Auto-detect using BERTopic topic modeling
    """
    from mirage.pipeline.domain import fetch_domain_and_role, load_domain_expert_from_env, save_domain_expert_to_env
    
    # Priority 1: Check config.yaml
    try:
        from mirage.core.config import get_domain_expert_config
        domain_config = get_domain_expert_config()
        config_domain = domain_config.get('domain')
        config_persona = domain_config.get('expert_persona')
        
        if config_domain and config_persona:
            print(f"Using domain and expert persona from config.yaml")
            print(f"   Domain: {config_domain}")
            print(f"   Expert Persona: {config_persona}")
            # Save to env for consistency
            save_domain_expert_to_env(config_domain, config_persona)
            return config_domain, config_persona
    except ImportError:
        pass
    
    # Priority 2: Check environment variables
    env_domain, env_persona = load_domain_expert_from_env()
    if env_domain and env_persona:
        return env_domain, env_persona

    # Priority 3: Auto-detect using BERTopic
    print("\nAuto-detecting domain and expert persona from corpus...")
    domain, expert_persona = fetch_domain_and_role(chunks_file, embeddings=embeddings, chunk_ids=chunk_ids)
    print(f"Domain: {domain}")
    print(f"Expert Persona: {expert_persona}")
    return domain, expert_persona


# ============================================================================
# STEP 4: MULTIHOP CONTEXT -> QA GENERATION -> VERIFICATION (PARALLEL)
# ============================================================================
def process_single_chunk_for_qa(chunk_data: tuple, domain: str, expert_persona: str) -> tuple:
    """Process a single chunk for QA generation (for parallel execution)"""
    i, chunk = chunk_data
    from mirage.pipeline.context import build_complete_context
    from mirage.pipeline.qa_generator import (
        generate_qa, select_qa_pairs, verify_qa, is_verification_successful,
        correct_failed_qa
    )
    
    successful_qa = []
    failed_qa = []
    chunk_with_context = None
    
    try:
        chunk_content = chunk.get('content', str(chunk)) if isinstance(chunk, dict) else str(chunk)
        
        initial_chunk_dict = {
            'content': chunk_content,
            'chunk_id': chunk.get('chunk_id', str(i)) if isinstance(chunk, dict) else str(i),
            'file_name': chunk.get('file_name', 'unknown') if isinstance(chunk, dict) else 'unknown',
            'artifact': chunk.get('artifact', 'None') if isinstance(chunk, dict) else 'None'
        }
        
        # 4.1 Build complete context via multihop retrieval
        context_result = build_complete_context(
            initial_chunk=initial_chunk_dict,
            max_depth=MAX_DEPTH,
            max_breadth=MAX_BREADTH,
            chunks_per_search=CHUNKS_PER_SEARCH,
            expert_persona=expert_persona,
            domain=domain,
            log_details=True,  # Enable detailed iteration logging
            chunk_addition_mode=CHUNK_ADDITION_MODE
        )
        context_chunks = context_result.get('chunks', [])
        
        chunk_with_context = {
            'original_chunk_id': initial_chunk_dict['chunk_id'],
            'original_file_name': initial_chunk_dict['file_name'],
            'original_chunk': chunk_content,
            'complete_context': context_result['context'],
            'context_status': context_result['status'],
            'depth_reached': context_result['depth'],
            'hop_count': context_result.get('hop_count', 0),
            'max_breadth_used': context_result.get('max_breadth_used', 0),
            'chunks_added': context_result['chunks_added'],
            'context_chunks': context_chunks,
            'search_history': context_result.get('search_history', []),
            'termination_reason': context_result.get('termination_reason', ''),
            'iteration_logs': context_result.get('iteration_logs', [])  # Detailed retrieval logs
        }
        
        # 4.2 Generate QA pairs
        qa_pairs, qa_metadata = generate_qa(context_chunks, expert_persona, domain)
        
        # 4.3 Select relevant pairs
        selected, rejected = select_qa_pairs(qa_pairs, context_chunks, expert_persona, domain)
        
        # 4.4 Verify selected pairs with correction support
        pairs_to_verify = selected.copy()
        pairs_for_correction = []  # Collect failed pairs for correction
        
        for qa in pairs_to_verify:
            verification = verify_qa(
                context_chunks, qa['question'], qa['answer'], expert_persona, domain
            )
            
            qa_entry = {
                'chunk_id': i,
                'original_chunk': chunk_content,
                'final_context': context_result['context'],
                'context_chunks': context_chunks,  # Include full chunks with image_path for metrics
                'context_status': context_result['status'],
                'depth_reached': context_result['depth'],
                'hop_count': context_result.get('hop_count', 0),
                'max_breadth_used': context_result.get('max_breadth_used', 0),
                'chunks_added': context_result['chunks_added'],
                'search_history': context_result.get('search_history', []),
                'iteration_logs': context_result.get('iteration_logs', []),
                'keywords_per_chunk': qa_metadata.get('keywords_per_chunk', {}),
                'related_keywords': qa_metadata.get('related_keywords', ''),
                'expert_persona': expert_persona,
                'domain': domain,
                'question': qa['question'],
                'answer': qa['answer'],
                'relevance_score': qa.get('relevance_score', '0'),
                'difficulty_score': qa.get('difficulty_score', '0'),
                'selection_status': qa.get('selection_status', 'SELECTED'),
                'selection_reason': qa.get('selection_reason', ''),
                'verification_result': verification
            }
            
            if is_verification_successful(verification):
                successful_qa.append(qa_entry)
            else:
                # Collect for correction attempt if enabled
                if QA_CORRECTION_ENABLED:
                    pairs_for_correction.append({
                        'question': qa['question'],
                        'answer': qa['answer'],
                        'verification_result': verification,
                        'original_qa_entry': qa_entry
                    })
                else:
                    # Final failure (correction disabled) - store with reason
                    qa_entry['failure_reason'] = 'Failed verification (correction disabled)'
                    failed_qa.append(qa_entry)
        
        # 4.5 Attempt correction for failed pairs
        if QA_CORRECTION_ENABLED and pairs_for_correction:
            correction_attempt = 0
            current_failures = pairs_for_correction
            
            while correction_attempt < QA_CORRECTION_MAX_ATTEMPTS and current_failures:
                correction_attempt += 1
                print(f"  Correction attempt {correction_attempt}/{QA_CORRECTION_MAX_ATTEMPTS} for {len(current_failures)} failed pair(s)")
                
                # Build failed_qa_pairs list for correction
                failed_for_correction = [
                    {
                        'question': f['question'],
                        'answer': f['answer'],
                        'verification_result': f['verification_result']
                    }
                    for f in current_failures
                ]
                
                # Call correction function
                corrected_pairs = correct_failed_qa(
                    context_chunks, failed_for_correction, expert_persona, domain
                )
                
                # Verify corrected pairs
                new_failures = []
                for corrected in corrected_pairs:
                    corrected_verification = verify_qa(
                        context_chunks, corrected['question'], corrected['answer'], 
                        expert_persona, domain
                    )
                    
                    corrected_entry = {
                        'chunk_id': i,
                        'original_chunk': chunk_content,
                        'final_context': context_result['context'],
                        'context_chunks': context_chunks,
                        'context_status': context_result['status'],
                        'depth_reached': context_result['depth'],
                        'hop_count': context_result.get('hop_count', 0),
                        'max_breadth_used': context_result.get('max_breadth_used', 0),
                        'chunks_added': context_result['chunks_added'],
                        'search_history': context_result.get('search_history', []),
                        'iteration_logs': context_result.get('iteration_logs', []),
                        'keywords_per_chunk': qa_metadata.get('keywords_per_chunk', {}),
                        'related_keywords': qa_metadata.get('related_keywords', ''),
                        'expert_persona': expert_persona,
                        'domain': domain,
                        'question': corrected['question'],
                        'answer': corrected['answer'],
                        'relevance_score': corrected.get('relevance_score', '0'),
                        'difficulty_score': corrected.get('difficulty_score', '0'),
                        'selection_status': 'CORRECTED',
                        'selection_reason': f'Corrected on attempt {correction_attempt}',
                        'verification_result': corrected_verification,
                        'correction_attempt': correction_attempt
                    }
                    
                    if is_verification_successful(corrected_verification):
                        print(f"    Corrected pair passed verification")
                        successful_qa.append(corrected_entry)
                    else:
                        # Accumulate feedback for next attempt
                        new_failures.append({
                            'question': corrected['question'],
                            'answer': corrected['answer'],
                            'verification_result': corrected_verification,
                            'original_qa_entry': corrected_entry
                        })
                
                current_failures = new_failures
            
            # Add remaining failures to failed_qa after all correction attempts
            for failure in current_failures:
                qa_entry = failure['original_qa_entry']
                qa_entry['failure_reason'] = f'Failed verification after {correction_attempt} correction attempt(s)'
                failed_qa.append(qa_entry)
            
            # Also add original failed pairs that weren't corrected
            if not corrected_pairs and pairs_for_correction:
                for failure in pairs_for_correction:
                    qa_entry = failure['original_qa_entry']
                    qa_entry['failure_reason'] = 'Correction produced no valid pairs'
                    failed_qa.append(qa_entry)
        
        for qa in rejected:
            failed_qa.append({
                'chunk_id': i,
                'original_chunk': chunk_content,
                'final_context': context_result['context'],
                'context_status': context_result['status'],
                'depth_reached': context_result['depth'],
                'hop_count': context_result.get('hop_count', 0),
                'max_breadth_used': context_result.get('max_breadth_used', 0),
                'chunks_added': context_result['chunks_added'],
                'search_history': context_result.get('search_history', []),
                'iteration_logs': context_result.get('iteration_logs', []),
                'keywords_per_chunk': qa_metadata.get('keywords_per_chunk', {}),
                'related_keywords': qa_metadata.get('related_keywords', ''),
                'question': qa['question'],
                'answer': qa['answer'],
                'relevance_score': qa.get('relevance_score', '0'),
                'difficulty_score': qa.get('difficulty_score', '0'),
                'selection_status': qa.get('selection_status', 'REJECTED'),
                'selection_reason': qa.get('selection_reason', ''),
                'failure_reason': 'Rejected by selection agent'
            })
            
    except Exception as e:
        logging.error(f"Error processing chunk {i}: {e}")
        failed_qa.append({
            'chunk_id': i,
            'error': str(e),
            'original_chunk': chunk_content if 'chunk_content' in locals() else str(chunk)
        })
    
    return successful_qa, failed_qa, chunk_with_context


def filter_chunks_by_qa_type(chunks: list, qa_type: str) -> list:
    """Filter chunks based on desired QA type.
    
    Args:
        chunks: List of chunk dictionaries
        qa_type: 'multihop', 'multimodal', 'text', or 'mix'
    
    Returns:
        Filtered list of chunks appropriate for the QA type
    """
    if qa_type == 'mix':
        # Mix mode: use all chunks
        return chunks
    
    filtered = []
    for chunk in chunks:
        artifact = chunk.get('artifact', 'None') if isinstance(chunk, dict) else 'None'
        has_artifact = artifact and artifact != 'None' and artifact.lower() != 'none'
        
        if qa_type == 'multimodal':
            # Only chunks with images/tables
            if has_artifact:
                filtered.append(chunk)
        elif qa_type == 'text':
            # Only text chunks (no artifacts)
            if not has_artifact:
                filtered.append(chunk)
        elif qa_type == 'multihop':
            # All chunks can produce multihop QA (determined by context retrieval)
            filtered.append(chunk)
        else:
            # Unknown type, include all
            filtered.append(chunk)
    
    return filtered


def is_qa_type_match(qa_entry: dict, qa_type: str) -> bool:
    """Check if a QA pair matches the desired type.
    
    Args:
        qa_entry: QA pair dictionary
        qa_type: 'multihop', 'multimodal', 'text', or 'mix'
    
    Returns:
        True if QA matches the type criteria
    """
    if qa_type == 'mix':
        return True
    
    # Check multihop: uses multiple chunks
    chunks_added = qa_entry.get('chunks_added', [])
    is_multihop = len(chunks_added) > 1 if isinstance(chunks_added, list) else False
    
    # Check multimodal: context contains images/tables
    context_chunks = qa_entry.get('context_chunks', [])
    is_multimodal = False
    for ctx_chunk in context_chunks:
        if isinstance(ctx_chunk, dict):
            artifact = ctx_chunk.get('artifact', 'None')
            if artifact and artifact != 'None' and artifact.lower() != 'none':
                is_multimodal = True
                break
    
    if qa_type == 'multihop':
        return is_multihop
    elif qa_type == 'multimodal':
        return is_multimodal
    elif qa_type == 'text':
        return not is_multimodal
    
    return True


def generate_qa_dataset_parallel(chunks: list, domain: str, expert_persona: str,
                                  checkpoint_mgr: CheckpointManager = None) -> tuple:
    """Generate QA pairs with multihop context retrieval (PARALLEL)
    
    Supports:
    - Early stopping when QA_NUM_PAIRS target is reached
    - Checkpoint resume from previous run
    - Filters output based on QA_TYPE setting
    
    Returns:
        Tuple of (successful_qa, failed_qa, chunks_with_context, generation_stats)
        generation_stats contains target, generated, chunks info for reporting
    """
    import threading
    
    # Use global checkpoint manager if not provided
    if checkpoint_mgr is None:
        checkpoint_mgr = _CHECKPOINT_MANAGER
    
    # Load existing progress from checkpoint
    if checkpoint_mgr:
        completed_chunk_ids = checkpoint_mgr.get_completed_qa_chunk_ids()
        existing_successful, existing_failed, existing_contexts = checkpoint_mgr.get_accumulated_qa()
    else:
        completed_chunk_ids = set()
        existing_successful, existing_failed, existing_contexts = [], [], []
    
    successful_qa = existing_successful.copy()
    failed_qa = existing_failed.copy()
    chunks_with_context = existing_contexts.copy()
    
    # Filter chunks based on QA type
    filtered_chunks = filter_chunks_by_qa_type(chunks, QA_TYPE)
    
    if len(filtered_chunks) < len(chunks):
        print(f"Filtered chunks for '{QA_TYPE}' type: {len(chunks)} -> {len(filtered_chunks)}")
    
    process_chunks = filtered_chunks[:MAX_CHUNKS] if MAX_CHUNKS else filtered_chunks
    chunk_data = [(i+1, chunk) for i, chunk in enumerate(process_chunks)]
    
    # Filter out already-completed chunks
    pending_chunk_data = [(idx, chunk) for idx, chunk in chunk_data 
                          if str(idx) not in completed_chunk_ids]
    
    # Target QA count (None = no limit)
    target_qa_count = QA_NUM_PAIRS
    
    print(f"\nUsing {QA_MAX_WORKERS} parallel workers for QA generation")
    print(f"Target: {target_qa_count if target_qa_count else 'unlimited'} '{QA_TYPE}' QA pairs")
    
    if completed_chunk_ids:
        print(f"📂 Resuming from checkpoint: {len(completed_chunk_ids)} chunks already processed")
        print(f"   ✅ {len(existing_successful)} successful QA pairs loaded")
        print(f"   ⏳ {len(pending_chunk_data)} chunks remaining")
    else:
        print(f"Processing {len(chunk_data)} chunks from corpus")
    
    # Check if we already have enough QA pairs from checkpoint
    initial_matching_count = sum(1 for qa in successful_qa if is_qa_type_match(qa, QA_TYPE))
    if target_qa_count and initial_matching_count >= target_qa_count:
        print(f"\n✅ Target already reached from checkpoint: {initial_matching_count}/{target_qa_count} QA pairs")
        generation_stats = {
            'target_qa_pairs': target_qa_count,
            'qa_type': QA_TYPE,
            'total_chunks': len(chunk_data),
            'chunks_processed': len(completed_chunk_ids),
            'generated_qa_pairs': initial_matching_count,
            'target_reached': True,
            'corpus_exhausted': False,
            'resumed_from_checkpoint': True
        }
        return successful_qa, failed_qa, chunks_with_context, generation_stats
    
    # Thread-safe counters
    matching_qa_count = [initial_matching_count]  # Start from checkpoint count
    chunks_processed = [len(completed_chunk_ids)]  # Start from checkpoint count
    stop_flag = [False]
    count_lock = threading.Lock()
    
    # Generation stats for reporting
    generation_stats = {
        'target_qa_pairs': target_qa_count,
        'qa_type': QA_TYPE,
        'total_chunks': len(chunk_data),
        'chunks_processed': len(completed_chunk_ids),
        'generated_qa_pairs': initial_matching_count,
        'target_reached': False,
        'corpus_exhausted': False,
        'resumed_from_checkpoint': bool(completed_chunk_ids)
    }
    
    if not pending_chunk_data:
        print("No pending chunks to process")
        generation_stats['corpus_exhausted'] = True
        return successful_qa, failed_qa, chunks_with_context, generation_stats
    
    # Process chunks in batches to enable proper early stopping
    # Batch size = 2x workers to keep workers busy while checking results
    batch_size = max(QA_MAX_WORKERS * 2, 10)
    chunk_idx_offset = 0
    
    with ThreadPoolExecutor(max_workers=QA_MAX_WORKERS) as executor:
        pbar = tqdm(total=len(pending_chunk_data), desc="Generating QA")
        
        while chunk_idx_offset < len(pending_chunk_data) and not stop_flag[0]:
            # Submit next batch of chunks
            batch_end = min(chunk_idx_offset + batch_size, len(pending_chunk_data))
            current_batch = pending_chunk_data[chunk_idx_offset:batch_end]
            
            futures = {
                executor.submit(process_single_chunk_for_qa, cd, domain, expert_persona): cd[0]
                for cd in current_batch
            }
            
            for future in as_completed(futures):
                # Check if we should stop (target already reached by another future)
                if stop_flag[0]:
                    # Cancel remaining futures in this batch
                    for f in futures:
                        f.cancel()
                    break
                    
                chunk_idx = futures[future]
                pbar.update(1)
                
                try:
                    chunk_successful, chunk_failed, chunk_context = future.result()
                    
                    with count_lock:
                        chunks_processed[0] += 1
                    
                    # Track successful/failed for this chunk
                    chunk_success_list = []
                    chunk_fail_list = []
                    
                    # Filter successful QA by type and count
                    for qa in chunk_successful:
                        if is_qa_type_match(qa, QA_TYPE):
                            with count_lock:
                                if target_qa_count is None or matching_qa_count[0] < target_qa_count:
                                    successful_qa.append(qa)
                                    chunk_success_list.append(qa)
                                    matching_qa_count[0] += 1
                                    
                                    # Check if target reached
                                    if target_qa_count and matching_qa_count[0] >= target_qa_count:
                                        print(f"\nTarget reached: {matching_qa_count[0]}/{target_qa_count} '{QA_TYPE}' QA pairs")
                                        stop_flag[0] = True
                                        generation_stats['target_reached'] = True
                        else:
                            # QA doesn't match type, add to failed
                            qa['failure_reason'] = f"Does not match target type '{QA_TYPE}'"
                            failed_qa.append(qa)
                            chunk_fail_list.append(qa)
                    
                    failed_qa.extend(chunk_failed)
                    chunk_fail_list.extend(chunk_failed)
                    
                    if chunk_context:
                        chunks_with_context.append(chunk_context)
                    
                    # Checkpoint after each chunk is processed
                    if checkpoint_mgr:
                        checkpoint_mgr.save_qa_result(
                            chunk_id=chunk_idx,
                            successful=chunk_success_list,
                            failed=chunk_fail_list,
                            chunk_with_context=chunk_context
                        )
                        
                except Exception as e:
                    logging.error(f"Error in parallel QA generation for chunk {chunk_idx}: {e}")
                    error_entry = {'chunk_id': chunk_idx, 'error': str(e)}
                    failed_qa.append(error_entry)
                    
                    # Checkpoint the error
                    if checkpoint_mgr:
                        checkpoint_mgr.save_qa_result(
                            chunk_id=chunk_idx,
                            successful=[],
                            failed=[error_entry],
                            chunk_with_context=None
                        )
            
            chunk_idx_offset = batch_end
        
        pbar.close()
    
    # Update generation stats
    generation_stats['chunks_processed'] = chunks_processed[0]
    generation_stats['generated_qa_pairs'] = matching_qa_count[0]
    
    # Check if corpus was exhausted without reaching target
    if target_qa_count and matching_qa_count[0] < target_qa_count:
        generation_stats['corpus_exhausted'] = True
    
    # Summary with clear status
    print("\n" + "=" * 70)
    print("QA GENERATION SUMMARY")
    print("=" * 70)
    
    if target_qa_count:
        print(f"   Target QA pairs:     {target_qa_count}")
        print(f"   Generated QA pairs:  {matching_qa_count[0]}")
        print(f"   QA Type:             {QA_TYPE}")
        print(f"   Chunks processed:    {chunks_processed[0]}/{len(chunk_data)}")
        
        if generation_stats['target_reached']:
            print(f"\n   SUCCESS: Target of {target_qa_count} '{QA_TYPE}' QA pairs reached!")
        else:
            shortfall = target_qa_count - matching_qa_count[0]
            pct_achieved = (matching_qa_count[0] / target_qa_count) * 100 if target_qa_count > 0 else 0
            print(f"\n   CORPUS EXHAUSTED: Generated {matching_qa_count[0]}/{target_qa_count} ({pct_achieved:.1f}%)")
            print(f"       Shortfall: {shortfall} QA pairs")
            print(f"       All {len(chunk_data)} chunks have been processed.")
            print(f"       Consider: Adding more documents or using 'mix' type for higher yield.")
    else:
        print(f"   Generated QA pairs:  {matching_qa_count[0]} (no target limit)")
        print(f"   QA Type:             {QA_TYPE}")
        print(f"   Chunks processed:    {chunks_processed[0]}/{len(chunk_data)}")
    
    print("=" * 70)
    print("   Saving generated QA pairs...")
    
    return successful_qa, failed_qa, chunks_with_context, generation_stats


def save_qa_results(successful: list, failed: list, chunks_with_context: list):
    """Save QA results and chunks with context to JSON files"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if chunks_with_context:
        with open(OUTPUT_CHUNKS_WITH_CONTEXT, 'w', encoding='utf-8') as f:
            json.dump(chunks_with_context, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(chunks_with_context)} chunks with context to {OUTPUT_CHUNKS_WITH_CONTEXT}")
    
    if successful:
        with open(OUTPUT_QA_SUCCESSFUL, 'w', encoding='utf-8') as f:
            json.dump(successful, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(successful)} successful QA pairs to {OUTPUT_QA_SUCCESSFUL}")
    
    if failed:
        # Separate malformed chunks (with 'error' key) from regular QA failures
        malformed = [f for f in failed if 'error' in f]
        qa_failed = [f for f in failed if 'error' not in f]
        
        if malformed:
            with open(OUTPUT_MALFORMED_CHUNKS, 'w', encoding='utf-8') as f:
                json.dump(malformed, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(malformed)} malformed chunks to {OUTPUT_MALFORMED_CHUNKS}")
        
        if qa_failed:
            with open(OUTPUT_QA_FAILED, 'w', encoding='utf-8') as f:
                json.dump(qa_failed, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(qa_failed)} failed QA pairs to {OUTPUT_QA_FAILED}")
    
    # Stats: Count single vs multiple chunks in successful QA pairs
    single_chunk_count = 0
    multiple_chunk_count = 0
    for qa_pair in successful:
        chunks_added = qa_pair.get("chunks_added", [])
        if isinstance(chunks_added, list):
            if len(chunks_added) == 1:
                single_chunk_count += 1
            elif len(chunks_added) > 1:
                multiple_chunk_count += 1
    
    # Count malformed vs regular failures
    malformed_count = len([f for f in failed if 'error' in f]) if failed else 0
    qa_failed_count = len([f for f in failed if 'error' not in f]) if failed else 0
    
    print(f"\nQA STATS")
    print(f"{'='*80}")
    print(f"Number of malformed chunks: {malformed_count}")
    print(f"Number of QA pairs in qa_multihop_fail: {qa_failed_count}")
    print(f"Number of QA pairs with single chunk in qa_multihop_pass: {single_chunk_count}")
    print(f"Number of QA pairs with multiple chunks in qa_multihop_pass: {multiple_chunk_count}")


# ============================================================================
# STEP 5: DEDUPLICATION (PARALLEL CLUSTER PROCESSING)
# ============================================================================
def _process_single_cluster(args: Tuple) -> List[Dict]:
    """Worker function to process a single cluster for deduplication (parallel)"""
    cluster_items, cluster_indices, answer_embeddings_list, enable_reorganization, expert_persona, domain = args

    from mirage.embeddings.rerankers_text import LLMReranker
    from mirage.pipeline.deduplication import process_cluster_by_similarity
    import torch

    # Convert list back to tensor
    answer_embeddings = torch.tensor(answer_embeddings_list)

    llm_merger = LLMReranker(expert_persona=expert_persona, domain=domain)

    try:
        merged_items = process_cluster_by_similarity(
            cluster_items,
            cluster_indices,
            answer_embeddings,
            llm_merger,
            expert_persona,
            domain,
            enable_reorganization
        )
        return merged_items
    except Exception as e:
        logging.error(f"Error processing cluster: {e}")
        return cluster_items[:1] if cluster_items else []


def deduplicate_qa_dataset_parallel(input_file: str, output_file: str, 
                                     expert_persona: str,
                                     domain: str):
    """Deduplicate and reorganize QA pairs (PARALLEL cluster processing)
    
    Args:
        input_file: Path to input QA file
        output_file: Path to output deduplicated file
        expert_persona: Expert role for domain-specific deduplication
        domain: Domain context for deduplication
    """
    from mirage.pipeline.deduplication import (
        load_dataset, save_dataset, hierarchical_clustering,
        process_cluster_by_similarity
    )
    from mirage.embeddings.rerankers_text import LLMReranker
    from sentence_transformers import SentenceTransformer
    from mirage.embeddings.models import get_best_embedding_model
    import torch

    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return

    data = load_dataset(input_file)
    if not data:
        print("Dataset is empty.")
        return
    
    print(f"\n{'='*80}")
    print("HIERARCHICAL DEDUPLICATION (PARALLEL)")
    print(f"{'='*80}\n")
    
    # Prepare embeddings
    questions = [item['question'] for item in data]
    answers = [item['answer'] for item in data]
    
    # Use cached embedder if available
    embedder = get_embedder("bge_m3")
    
    print(f"Generating embeddings for {len(questions)} QA pairs...")
    question_embeddings = embedder.encode(questions, convert_to_tensor=True, show_progress_bar=True)
    answer_embeddings = embedder.encode(answers, convert_to_tensor=True, show_progress_bar=True)
    
    # Hierarchical clustering
    clusters = hierarchical_clustering(data, question_embeddings, answer_embeddings)
    
    clustered_indices = set()
    for cluster in clusters:
        clustered_indices.update(cluster)
    
    final_dataset = []
    stats = {
        'original': len(data),
        'singletons': 0,
        'clusters_processed': 0,
        'items_in_clusters': 0,
        'exact_duplicates': 0,
        'llm_merges': 0,
        'reorganized_packs': 0
    }
    
    # Add singletons
    for i in range(len(data)):
        if i not in clustered_indices:
            final_dataset.append(data[i])
            stats['singletons'] += 1
    
    print(f"Added {stats['singletons']} unique (singleton) items.")
    print(f"Processing {len(clusters)} clusters with {DEDUP_MAX_WORKERS} parallel workers...")
    
    # Process clusters in parallel
    llm_merger = LLMReranker(expert_persona=expert_persona, domain=domain)

    with ThreadPoolExecutor(max_workers=DEDUP_MAX_WORKERS) as executor:
        futures = {}
        for cluster in clusters:
            cluster_items = [data[idx] for idx in cluster]
            future = executor.submit(
                process_cluster_by_similarity,
                cluster_items,
                list(cluster),
                answer_embeddings,
                llm_merger,
                expert_persona,
                domain,
                True  # enable_reorganization
            )
            futures[future] = cluster
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Merging clusters"):
            try:
                merged_items = future.result()
                stats['clusters_processed'] += 1
                stats['items_in_clusters'] += len(futures[future])
                
                if merged_items and merged_items[0].get('dedup_method') == 'select_best':
                    stats['exact_duplicates'] += 1
                else:
                    stats['llm_merges'] += 1
                
                reorganized_count = sum(1 for item in merged_items if item.get('reorganized', False))
                if reorganized_count > 0:
                    stats['reorganized_packs'] += reorganized_count
                
                final_dataset.extend(merged_items)
            except Exception as e:
                logging.error(f"Error processing cluster: {e}")
                # Add first item as fallback
                cluster = futures[future]
                if cluster:
                    final_dataset.append(data[list(cluster)[0]])
    
    # Summary
    print("\n" + "="*80)
    print("DEDUPLICATION SUMMARY")
    print("="*80)
    print(f"Original: {stats['original']} -> Final: {len(final_dataset)}")
    print(f"Reduction: {stats['original'] - len(final_dataset)} ({100*(stats['original'] - len(final_dataset))/max(1,stats['original']):.1f}%)")
    print("="*80)
    
    save_dataset(final_dataset, output_file)
    print(f"Deduplicated dataset saved to {output_file}")


# ============================================================================
# STEP 6: EVALUATION (Choose: OPTIMIZED or STANDARD RAGAS)
# ============================================================================

def run_evaluation(qa_file: str, chunks_file: str, output_dir: str) -> dict:
    """Wrapper function to run evaluation on QA dataset"""
    # Load Gemini API key FIRST, before any imports that might need it
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        if os.path.exists(GEMINI_API_KEY_PATH):
            with open(GEMINI_API_KEY_PATH, 'r') as f:
                api_key = f.read().strip()
                if api_key:
                    os.environ["GOOGLE_API_KEY"] = api_key
                    print(f"Loaded Gemini API key from {GEMINI_API_KEY_PATH}")
                else:
                    raise ValueError(f"Gemini API key file is empty: {GEMINI_API_KEY_PATH}")
        else:
            raise FileNotFoundError(f"Gemini API key file not found: {GEMINI_API_KEY_PATH}")
    else:
        print(f"Using existing GOOGLE_API_KEY/GEMINI_API_KEY from environment")
    
    # Verify API key is set
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("Failed to set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
    
    try:
        if USE_OPTIMIZED_METRICS:
            # OPTIMIZED: 4-6 LLM calls per QA (3-5x faster)
            print("Using OPTIMIZED metrics (harmonized with metrics.py)")
            from mirage.evaluation.metrics_optimized import run_optimized_pipeline_evaluation
            
            results = run_optimized_pipeline_evaluation(
                qa_file=qa_file,
                output_dir=output_dir,
                corpus_path=chunks_file,  # Pass corpus for domain coverage
                enable_multimodal=True,
                max_workers=QA_MAX_WORKERS,
                sample_size=EVAL_SAMPLE_SIZE,
                run_context_necessity=True
            )
        else:
            # STANDARD: Uses RAGAS (10-20+ LLM calls per QA)
            print("Using STANDARD RAGAS metrics (slower, more LLM calls)")
            from mirage.evaluation.metrics import run_subset_evaluation
            
            # Load QA data
            print(f"Loading QA data from {qa_file}...")
            with open(qa_file, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            print(f"Loaded {len(qa_data)} QA pairs")
            
            results = run_subset_evaluation(
                qa_data=qa_data,
                corpus_path=chunks_file,
                output_dir=output_dir,
                sample_size=EVAL_SAMPLE_SIZE,
                run_context_necessity=True
            )
        
        return results
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("EVALUATION STOPPED DUE TO ERROR")
        print("=" * 70)
        print(f"   Error: {str(e)}")
        print()
        print("All data has been saved (chunks, QA pairs, etc.)")
        print("   Only evaluation results are missing.")
        print()
        print("   To retry evaluation later, run:")
        print(f"   python -m mirage.evaluation.metrics_optimized {qa_file} {output_dir}")
        print("=" * 70)
        logging.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_pipeline():
    """Run the complete QA dataset generation pipeline.
    
    Note: Preflight checks are MANDATORY and always run.
    The pipeline supports checkpointing for resume from interruption:
    - Markdown conversion (per file)
    - Chunking (per file)
    - Context building (per chunk)
    - QA generation (per chunk)
    """
    global _CHECKPOINT_MANAGER
    
    setup_logging()
    
    # =========================================================================
    # PREFLIGHT CHECKS - Validate all services before expensive execution
    # MANDATORY: Cannot be skipped - ensures system is properly configured
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUNNING PREFLIGHT CHECKS (mandatory)")
    print("=" * 70)
    
    all_passed, results = run_preflight_checks(skip_expensive=False, quiet=False)
    
    if not all_passed:
        print("\nSTOPPING: Preflight checks failed.")
        print("   Fix the issues above before running the pipeline.")
        print("   This prevents wasted LLM API calls on a misconfigured system.\n")
        return
    
    print("\nAll preflight checks passed! Starting pipeline...\n")
    
    # Initialize GPU lock for thread-safe access
    init_gpu_lock()
    
    # =========================================================================
    # INITIALIZE CHECKPOINT MANAGER AND LLM CACHE
    # =========================================================================
    _CHECKPOINT_MANAGER = CheckpointManager(OUTPUT_DIR)
    _CHECKPOINT_MANAGER.print_status()
    
    # Initialize LLM response cache (saves after every call for crash resilience)
    llm_cache = init_llm_cache(os.path.join(OUTPUT_DIR, ".cache"), enabled=True)
    if llm_cache:
        stats = llm_cache.get_stats()
        if stats['cache_size'] > 0:
            print(f"📦 LLM Cache: {stats['cache_size']} cached responses, {stats['hit_rate']} hit rate")
    
    print("=" * 70)
    print("QA DATASET GENERATION PIPELINE (WITH CHECKPOINTING)")
    print("=" * 70)
    print(f"Configuration:")
    print(f"   - Backend: {BACKEND}")
    print(f"   - LLM Model: {LLM_MODEL_NAME}")
    print(f"   - VLM Model: {VLM_MODEL_NAME}")
    print(f"   - API Rate Limit: {GEMINI_RPM} RPM, burst={GEMINI_BURST}")
    print(f"   - CPU Workers: {NUM_WORKERS}")
    print(f"   - QA Workers: {QA_MAX_WORKERS}")
    print(f"   - Dedup Workers: {DEDUP_MAX_WORKERS}")
    print(f"   - Embed Batch Size: {EMBED_BATCH_SIZE}")
    print(f"   - Target QA Pairs: {QA_NUM_PAIRS if QA_NUM_PAIRS else 'unlimited'}")
    print(f"   - QA Type: {QA_TYPE}")
    print(f"   - Max Depth: {MAX_DEPTH}")
    print(f"   - Run Evaluation: {RUN_EVALUATION}")
    print(f"   - Checkpointing: ENABLED")
    print("=" * 70)
    
    # =========================================================================
    # CHECKPOINT DETECTION - Resume from existing state if available
    # =========================================================================
    print("\n📋 Checking for existing checkpoints...")
    
    # Check for existing chunks file in output directory
    import glob
    output_chunks_file = os.path.join(OUTPUT_DIR, "chunks.json")
    markdown_dir = os.path.join(OUTPUT_DIR, "markdown")
    
    # Check for ANY existing embeddings (handles "auto" model name case)
    existing_embeddings = glob.glob(os.path.join(OUTPUT_EMBEDDINGS_DIR, "*_index.faiss"))
    
    checkpoint_status = {
        'has_chunks': os.path.exists(output_chunks_file) or (INPUT_CHUNKS_FILE and os.path.exists(INPUT_CHUNKS_FILE)),
        'has_markdown': os.path.exists(markdown_dir) and len([f for f in os.listdir(markdown_dir) if os.path.isdir(os.path.join(markdown_dir, f))]) > 0 if os.path.exists(markdown_dir) else False,
        'has_embeddings': len(existing_embeddings) > 0,
    }
    
    if checkpoint_status['has_chunks']:
        print("   ✅ Found existing chunks - will skip document conversion")
    elif checkpoint_status['has_markdown']:
        print("   ✅ Found existing markdown files - will skip PDF conversion, start from chunking")
    else:
        print("   ℹ️  No checkpoints found - starting from scratch")
    
    if checkpoint_status['has_embeddings']:
        print("   ✅ Found existing embeddings - will reuse FAISS index")
    
    # Step 1: Get chunks (from documents or pre-existing file)
    chunks_file = None
    chunks = None
    
    # Priority 1: Use explicitly provided chunks file
    if INPUT_CHUNKS_FILE and os.path.exists(INPUT_CHUNKS_FILE):
        print(f"\n📂 Loading chunks from provided file: {INPUT_CHUNKS_FILE}")
        chunks = load_chunks(INPUT_CHUNKS_FILE)
        chunks_file = INPUT_CHUNKS_FILE
    
    # Priority 2: Use existing chunks in output directory
    elif os.path.exists(output_chunks_file):
        print(f"\n📂 Resuming from existing chunks: {output_chunks_file}")
        chunks = load_chunks(output_chunks_file)
        chunks_file = output_chunks_file
    
    # Priority 3: Convert documents (may resume from checkpoint)
    else:
        chunks = convert_documents_to_chunks_parallel(INPUT_PDF_DIR, OUTPUT_CHUNKS, _CHECKPOINT_MANAGER)
        chunks_file = OUTPUT_CHUNKS
        if not chunks:
            print("No chunks generated. Exiting.")
            return
    
    print(f"   📊 Total chunks: {len(chunks)}")
    
    # Step 2: Embed ALL chunks & build unified FAISS index
    embeddings_dir, embeddings_array, chunk_ids = embed_all_chunks(chunks, chunks_file, OUTPUT_EMBEDDINGS_DIR)
    
    # Configure retrieval module
    configure_retrieval_paths(chunks_file, embeddings_dir)
    
    # Step 3: Get domain and expert role
    domain, expert_persona = get_domain_and_expert(chunks_file, embeddings=embeddings_array, chunk_ids=chunk_ids)
    
    # Step 4: Generate QA with multihop context (PARALLEL)
    # Shuffle chunks if enabled
    if SHUFFLE:
        import random
        random.seed(SHUFFLE_SEED)
        chunks = chunks.copy()  # Don't modify original list
        random.shuffle(chunks)
        print(f"Shuffled {len(chunks)} chunks (seed={SHUFFLE_SEED})")
    
    successful_qa, failed_qa, chunks_with_context, generation_stats = generate_qa_dataset_parallel(
        chunks, domain, expert_persona, _CHECKPOINT_MANAGER
    )
    
    # Save pre-deduplication results (always save, even if target not reached)
    save_qa_results(successful_qa, failed_qa, chunks_with_context)
    
    # Print final checkpoint status
    if _CHECKPOINT_MANAGER:
        _CHECKPOINT_MANAGER.print_status()
    
    # Save generation stats to output directory
    generation_stats_file = os.path.join(OUTPUT_DIR, "generation_stats.json")
    with open(generation_stats_file, 'w', encoding='utf-8') as f:
        json.dump(generation_stats, f, indent=2)
    print(f"Saved generation stats to {generation_stats_file}")
    
    # Step 5: Deduplicate (PARALLEL) - Optional
    qa_file_for_eval = OUTPUT_QA_SUCCESSFUL  # Default to non-deduplicated
    if RUN_DEDUPLICATION and successful_qa:
        print("\n" + "=" * 70)
        print("STEP 5: DEDUPLICATION (PARALLEL)")
        print("=" * 70)
        deduplicate_qa_dataset_parallel(
            OUTPUT_QA_SUCCESSFUL, OUTPUT_QA_DEDUPLICATED,
            expert_persona=expert_persona, domain=domain
        )
        qa_file_for_eval = OUTPUT_QA_DEDUPLICATED
    elif not RUN_DEDUPLICATION:
        print("\n" + "=" * 70)
        print("STEP 5: DEDUPLICATION - SKIPPED (disabled in config)")
        print("=" * 70)
        print("💡 To enable deduplication, set 'deduplication.enabled: true' in config.yaml")
    
    # Step 6: Generate Visualization for first multihop QA
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING MULTIHOP QA VISUALIZATION")
    print("=" * 70)
    if os.path.exists(qa_file_for_eval):
        try:
            from mirage.utils.visualize_multihop import generate_html_visualization
            with open(qa_file_for_eval, 'r', encoding='utf-8') as f:
                qa_data_viz = json.load(f)
            if qa_data_viz:
                viz_output = os.path.join(OUTPUT_DIR, "multihop_visualization.html")
                generate_html_visualization(qa_data_viz[0], viz_output)
                print(f"   Visualization for first QA pair: {viz_output}")
            else:
                print("   No QA pairs to visualize")
        except ImportError:
            print("   ⚠️ visualize_multihop.py not found, skipping visualization")
        except Exception as e:
            print(f"   ⚠️ Visualization failed: {e}")
    else:
        print("   No QA file found to visualize")
    
    # Step 7: Compute dataset statistics
    print("\n" + "=" * 70)
    print("STEP 7: COMPUTING DATASET STATISTICS")
    print("=" * 70)
    dataset_stats = compute_dataset_stats(
        output_dir=OUTPUT_DIR,
        pdf_dir=INPUT_PDF_DIR,
        chunks_file=chunks_file
    )
    print_dataset_stats(dataset_stats)
    
    # Step 8: Evaluate (if enabled)
    eval_results = None
    if RUN_EVALUATION and os.path.exists(qa_file_for_eval):
        print("\n" + "=" * 70)
        print("STEP 8: COMPREHENSIVE EVALUATION")
        print("=" * 70)
        eval_results = run_evaluation(qa_file_for_eval, chunks_file, OUTPUT_DIR)
        
        # Add dataset stats and QA category stats to evaluation results
        if eval_results:
            # Load QA data for category stats
            with open(qa_file_for_eval, 'r', encoding='utf-8') as f:
                qa_data_for_stats = json.load(f)
            
            qa_category_stats = compute_qa_category_stats(qa_data_for_stats)
            print_qa_category_stats(qa_category_stats)
            
            eval_results['dataset_statistics'] = {
                'total_images': dataset_stats['total_images'],
                'total_tables': dataset_stats['total_tables'],
                'total_pages': dataset_stats['total_pages'],
                'total_tokens': dataset_stats['total_tokens'],
                'num_pdfs': dataset_stats['num_pdfs'],
                'total_chunks': dataset_stats['total_chunks']
            }
            
            eval_results['qa_category_stats'] = {
                'total_qa_pairs': qa_category_stats['total_qa_pairs'],
                'multihop_count': qa_category_stats['multihop_count'],
                'multimodal_count': qa_category_stats['multimodal_count'],
                'multihop_multimodal_count': qa_category_stats['multihop_multimodal_count'],
                'multihop_only_count': qa_category_stats['multihop_only_count'],
                'multimodal_only_count': qa_category_stats['multimodal_only_count'],
                'text_only_count': qa_category_stats['text_only_count'],
                'multihop_pct': qa_category_stats['multihop_pct'],
                'multimodal_pct': qa_category_stats['multimodal_pct'],
                'multihop_multimodal_pct': qa_category_stats['multihop_multimodal_pct'],
                'avg_difficulty': qa_category_stats.get('avg_difficulty', 0.0),
                'avg_relevance': qa_category_stats.get('avg_relevance', 0.0)
            }
            
            # Save updated evaluation report with all stats
            with open(OUTPUT_EVAL_REPORT, 'w', encoding='utf-8') as f:
                json.dump(eval_results, f, indent=2, ensure_ascii=False)
            print(f"Updated evaluation report with dataset and QA category statistics")
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    
    # QA Generation stats
    print(f"\nQA Generation:")
    print(f"   - Target QA pairs:       {generation_stats['target_qa_pairs'] if generation_stats['target_qa_pairs'] else 'unlimited'}")
    print(f"   - Generated QA pairs:    {generation_stats['generated_qa_pairs']}")
    print(f"   - QA Type:               {generation_stats['qa_type']}")
    print(f"   - Chunks processed:      {generation_stats['chunks_processed']}/{generation_stats['total_chunks']}")
    
    if generation_stats['target_reached']:
        print(f"   - Status:                TARGET REACHED")
    elif generation_stats['corpus_exhausted']:
        print(f"   - Status:                CORPUS EXHAUSTED")
    else:
        print(f"   - Status:                COMPLETED")
    
    print(f"\nResults:")
    print(f"   - Chunks with complete context: {len(chunks_with_context)}")
    print(f"   - Successful QA pairs (pre-dedup): {len(successful_qa)}")
    print(f"   - Failed QA pairs: {len(failed_qa)}")
    
    if RUN_DEDUPLICATION and os.path.exists(OUTPUT_QA_DEDUPLICATED):
        with open(OUTPUT_QA_DEDUPLICATED) as f:
            dedup_data = json.load(f)
            dedup_count = len(dedup_data)
        print(f"   - After deduplication: {dedup_count} QA pairs")
    else:
        print(f"   - Deduplication: SKIPPED")
    
    # Show dataset statistics
    print("\nDataset Statistics:")
    print(f"   - Images: {dataset_stats['total_images']}")
    print(f"   - Tables: {dataset_stats['total_tables']}")
    print(f"   - Pages: {dataset_stats['total_pages']}")
    print(f"   - Tokens: {dataset_stats['total_tokens']:,}")
    
    # Show QA category breakdown if available
    if os.path.exists(qa_file_for_eval):
        with open(qa_file_for_eval, 'r', encoding='utf-8') as f:
            qa_for_cats = json.load(f)
        cat_stats = compute_qa_category_stats(qa_for_cats)
        print("\nQA Category Breakdown:")
        print(f"   - Multihop: {cat_stats['multihop_count']} ({cat_stats['multihop_pct']}%)")
        print(f"   - Multimodal: {cat_stats['multimodal_count']} ({cat_stats['multimodal_pct']}%)")
        print(f"   - Both (Multihop and Multimodal): {cat_stats['multihop_multimodal_count']} ({cat_stats['multihop_multimodal_pct']}%)")
        print(f"   - Avg Difficulty: {cat_stats.get('avg_difficulty', 0.0):.2f}")
        print(f"   - Avg Relevance: {cat_stats.get('avg_relevance', 0.0):.2f}")
    
    # Show evaluation summary if available
    if eval_results:
        print("\nEvaluation Metrics:")
        # RAGAS core metrics
        rm = eval_results.get('ragas_metrics', {})
        print(f"   - Faithfulness (Faith.):      {rm.get('faithfulness', 0):.3f}")
        print(f"   - Answer Relevancy (Rel.):    {rm.get('answer_relevancy', 0):.3f}")
        print(f"   - Context Precision:          {rm.get('context_precision', 0):.3f}")
        print(f"   - Context Recall:             {rm.get('context_recall', 0):.3f}")
        
        # Concept Hops (H) - from aggregate scores
        agg = eval_results.get('aggregate_scores', {})
        avg_hops = agg.get('avg_concept_hops_question', 0)
        print(f"   - Avg Concept Hops (H):       {avg_hops:.2f}")
        
        # Multihop reasoning metrics
        mh = eval_results.get('multihop_metrics', {})
        if mh:
            print(f"   - Reasoning Score (S_reason): {mh.get('avg_reasoning_score', 0):.3f}")
        
        # Visual grounding
        mm = eval_results.get('multimodal_metrics', {})
        if mm:
            print(f"   - Visual Grounding (Vis.Gr.): {mm.get('avg_visual_dependency', 0):.3f}")
        
        # Domain coverage metrics
        dc = eval_results.get('domain_coverage', {})
        print(f"   - Domain Coverage:            {dc.get('chunk_coverage', 0)*100:.1f}%")
        print(f"   - JSD (Topic Divergence):     {dc.get('topic_divergence_js', 0):.4f}")
        
        # Context Necessity
        cn = eval_results.get('context_necessity', {})
        print(f"   - Context Necessity:          {cn.get('avg_context_necessity_score', 0):.2f}")
        
        # Dataset statistics
        stats = eval_results.get('subset_statistics', {})
        print(f"\nDataset Composition:")
        print(f"   - Total QA Pairs:    {stats.get('total_qa_pairs', 0)}")
        print(f"   - Multihop QA:       {stats.get('multihop_count', 0)}")
        print(f"   - Multimodal QA:     {stats.get('multimodal_count', 0)}")
    
    print("\nOutput files:")
    print(f"   - Chunks with context: {OUTPUT_CHUNKS_WITH_CONTEXT}")
    print(f"   - Successful QA: {OUTPUT_QA_SUCCESSFUL}")
    print(f"   - Failed QA: {OUTPUT_QA_FAILED}")
    if os.path.exists(OUTPUT_QA_DEDUPLICATED):
        print(f"   - Deduplicated QA: {OUTPUT_QA_DEDUPLICATED}")
    if eval_results:
        print(f"   - Evaluation Report: {OUTPUT_EVAL_REPORT}")
    print("=" * 70)
    
    # Print LLM cache statistics
    llm_cache = get_llm_cache()
    if llm_cache:
        llm_cache.print_stats()
    
    print("\nPipeline completed!")


# Backward compatibility alias
def main():
    """Backward compatibility wrapper for run_pipeline."""
    return run_pipeline()


if __name__ == "__main__":
    # Use spawn for multiprocessing to avoid CUDA issues
    mp.set_start_method('spawn', force=True)
    
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        logging.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
