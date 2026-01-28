"""
Multi-hop Context Completion Agent

This agent iteratively verifies chunk completeness and retrieves additional context
when needed, using a breadth-first approach with depth limits.

Architecture:
- Max Breadth: 5 search strings per verification (fewer is better)
- Max Depth: 10 iterations
- Chunks per Search: Top 2 most relevant chunks

This module also includes simple retrieval functions for single-query use cases.
"""

import json
import re
import torch
import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from mirage.core.llm import call_vlm_interweaved, setup_logging, batch_call_vlm_interweaved
from mirage.embeddings.models import NomicVLEmbed as NomicEmbedder
from mirage.embeddings.rerankers_multimodal import MonoVLMReranker, VLMReranker, TextEmbeddingReranker
from sentence_transformers import SentenceTransformer
import threading
import os

# Thread-safe lock for embedder initialization
_embedder_init_lock = threading.Lock()


def _get_embedder_for_model(model_name: str):
    """Get the correct embedder based on model name (thread-safe with caching, GPU-enabled)."""
    import sys
    this_module = sys.modules[__name__]
    
    # Thread-safe check and initialization
    with _embedder_init_lock:
        # Check cache first (inside lock to prevent race condition)
        if hasattr(this_module, '_cached_query_embedder') and this_module._cached_query_embedder is not None:
            return this_module._cached_query_embedder
        
        # Try to use cached embedder from main.py (already on GPU)
        if hasattr(this_module, '_cached_embedder') and this_module._cached_embedder is not None:
            print(f"Using cached embedder from main pipeline (GPU-enabled)...")
            this_module._cached_query_embedder = this_module._cached_embedder
            return this_module._cached_embedder
        
        # Get GPU configuration from config
        try:
            from mirage.core.config import get_embedding_config
            embed_config = get_embedding_config()
            gpus = embed_config.get('gpus', None)
            # Determine device: use first GPU from list, or cuda:0, or cpu
            if gpus and len(gpus) > 0:
                device = f"cuda:{gpus[0]}"
            elif torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
        except:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            gpus = None
        
        # Load the embedder (only one thread will do this)
        if model_name.lower() in ('bge_m3', 'bge-m3', 'baai/bge-m3'):
            print(f"Loading BGE-M3 embedder for query on {device} (thread-safe)...")
            embedder = SentenceTransformer('BAAI/bge-m3', device=device)
        elif model_name.lower() in ('nomic', 'nomic_embed', 'nomic-ai/nomic-embed-multimodal-7b'):
            print(f"Loading Nomic embedder for query on {device}...")
            embedder = NomicEmbedder(gpus=gpus)
        else:
            # Default to BGE-M3 for unknown models
            print(f"Unknown model {model_name}, defaulting to BGE-M3 embedder on {device}...")
            embedder = SentenceTransformer('BAAI/bge-m3', device=device)
        
        # Cache it for future calls
        this_module._cached_query_embedder = embedder
        return embedder
from mirage.core.prompts import PROMPTS_CHUNK

# ============================================================================
# CONFIGURATION
# ============================================================================

# Multi-hop context completion parameters
# SAFETY: Sensible limits to prevent runaway API calls
# Each iteration can generate breadth × chunks_per_search API calls
MAX_DEPTH = 2  # Maximum 2 iterations per chunk (cost-optimized default)
MAX_BREADTH = 5  # Maximum 5 search strings per verification (was 20)
CHUNKS_PER_SEARCH = 2  # Number of chunks to retrieve per search string
# Chunk addition mode: "EXPLANATORY" (only direct answers) or "RELATED" (includes related chunks)
CHUNK_ADDITION_MODE = "RELATED"  # Default: include both EXPLANATORY and RELATED chunks

# Circuit breaker: stop if N consecutive iterations add no new chunks
MAX_CONSECUTIVE_NO_PROGRESS = 3

# Simple retrieval parameters
RETRIEVAL_METHOD = "top_k"  # Options: "top_k" or "top_p"
RETRIEVAL_K = 20            # Number of chunks for top-k retrieval (increased for better recall)
RETRIEVAL_P = 0.9           # Cumulative probability threshold for top-p retrieval
RERANK_TOP_K = 10           # Number of chunks to rerank (increased proportionally)
CONTEXT_SIZE = 2            # Number of chunks to use as final context

# Paths (configured via main.py or config.yaml)
# These defaults are overridden by main.py's configure_retrieval_paths()
_default_output = os.environ.get("MIRAGE_OUTPUT_DIR", "output")
EMBEDDINGS_DIR = os.path.join(_default_output, "embeddings")
CHUNKS_FILE = os.path.join(_default_output, "chunks.json")
IMAGE_BASE_DIR = os.path.join(_default_output, "markdown")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_image_paths(artifact_text, file_name=None):
    """Extract all image paths from artifact field
    
    Args:
        artifact_text: The artifact string (e.g., "![Image](path1) ![Image](path2)")
        file_name: The source document name (used for directory structure)
    
    Returns:
        List of absolute image paths (empty list if no artifacts found)
        Note: Returns paths even if files don't exist (for multimodal detection)
    """
    if artifact_text == "None" or not artifact_text:
        return []
    
    # Find all image references: ![Image](path) or ![alt](path)
    matches = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', artifact_text)
    
    image_paths = []
    for rel_path in matches:
        # If file_name is provided, assume images are in a subdirectory named after the file
        if file_name and file_name != 'unknown':
            # Structure: IMAGE_BASE_DIR / file_name / rel_path
            abs_path = f"{IMAGE_BASE_DIR}/{file_name}/{rel_path}"
        else:
            # Fallback to direct concatenation (legacy behavior)
            abs_path = f"{IMAGE_BASE_DIR}/{rel_path}"
        
        image_paths.append(abs_path)
    
    return image_paths


def extract_image_path(artifact_text, file_name=None):
    """Extract first image path from artifact field (for backward compatibility)
    
    Args:
        artifact_text: The artifact string (e.g., "![Image](path)")
        file_name: The source document name (used for directory structure)
    
    Returns:
        Absolute path to first image, or None if no artifact found
        Note: Returns path even if file doesn't exist (for multimodal detection)
    """
    paths = extract_image_paths(artifact_text, file_name)
    return paths[0] if paths else None


# ============================================================================
# SIMPLE RETRIEVAL FUNCTIONS
# ============================================================================

def retrieve_and_rerank(query: str, model_name: str = None, 
                        retrieval_method: str = "top_k", retrieval_k: int = 10, retrieval_p: float = 0.9,
                        rerank_top_k: int = 5, context_size: int = 2):
    """Retrieve top chunks and rerank them (simple single-query retrieval)
    
    Args:
        query: Query string
        model_name: Embedding model name
        retrieval_method: "top_k" or "top_p"
        retrieval_k: Number of chunks for top-k retrieval
        retrieval_p: Cumulative probability threshold for top-p retrieval
        rerank_top_k: Number of chunks to rerank
        context_size: Number of chunks to use as final context
    
    Returns:
        List of tuples: (orig_idx, relevance, chunk_dict)
    """
    
    # Use cached embeddings and index if available (much faster)
    import sys
    this_module = sys.modules[__name__]
    
    # Get model name and index prefix - read from existing files
    import glob
    import os
    index_prefix = None  # Prefix used in filenames (e.g., "auto")
    
    if model_name is None:
        existing_metadata = glob.glob(os.path.join(EMBEDDINGS_DIR, "*_metadata.json"))
        if existing_metadata:
            metadata_file = existing_metadata[0]
            # Extract prefix from filename (e.g., "auto_metadata.json" -> "auto")
            index_prefix = os.path.basename(metadata_file).replace("_metadata.json", "")
            # Read actual model name from metadata
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    model_name = metadata.get('model', index_prefix)
            except:
                model_name = index_prefix
        
        if model_name is None:
            # Fall back to config if no metadata exists
            try:
                from mirage import main as main_module
                model_name = getattr(main_module, 'EMBEDDING_MODEL', 'bge_m3')
                index_prefix = model_name.replace('/', '_').replace('\\', '_')
            except:
                model_name = 'bge_m3'
                index_prefix = 'bge_m3'
    
    # If index_prefix not set, derive from model_name
    if index_prefix is None:
        index_prefix = model_name.replace('/', '_').replace('\\', '_') if model_name else 'bge_m3'
    
    # Check if caching is enabled and cached embeddings/index are available
    cache_enabled = False
    try:
        import main
        cache_enabled = hasattr(main, 'CACHE_EMBEDDINGS') and main.CACHE_EMBEDDINGS
    except:
        pass
    
    # Check for cached embeddings and index in memory (only if caching is enabled)
    if (cache_enabled and
        hasattr(this_module, '_cached_chunk_index') and 
        this_module._cached_chunk_index is not None and
        hasattr(this_module, '_cached_chunk_ids') and
        this_module._cached_chunk_ids is not None):
        
        # Use cached in-memory index and embeddings
        index = this_module._cached_chunk_index
        chunk_ids = this_module._cached_chunk_ids
        
        # Load chunks (still need to load for content)
        print(f"Loading chunks from {CHUNKS_FILE}...")
        with open(CHUNKS_FILE, 'r') as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks")
        print(f"✅ Using cached embeddings index in memory (fast retrieval)")
        
        # Use thread-safe cached embedder (caching is now inside _get_embedder_for_model)
        embedder = _get_embedder_for_model(model_name)
        
        # Embed query only (chunk embeddings already cached)
        # Use GPU lock if available for thread-safe access
        # Use convert_to_numpy=True to avoid device mismatch when model is on CPU
        gpu_lock = getattr(this_module, '_gpu_lock', None)
        if gpu_lock:
            with gpu_lock:
                query_embedding = embedder.encode(query, convert_to_numpy=True)
        else:
            query_embedding = embedder.encode(query, convert_to_numpy=True)
        if isinstance(query_embedding, torch.Tensor):
            query_array = query_embedding.cpu().float().numpy()
        else:
            query_array = np.array(query_embedding, dtype=np.float32)
        if query_array.ndim == 1:
            query_array = query_array.reshape(1, -1)
        faiss.normalize_L2(query_array)
    else:
        # Fallback to disk-based loading
        print(f"Loading chunks from {CHUNKS_FILE}...")
        with open(CHUNKS_FILE, 'r') as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks")
        
        # Load FAISS index from disk (use index_prefix for filename)
        index_path = f"{EMBEDDINGS_DIR}/{index_prefix}_index.faiss"
        print(f"Loading FAISS index from {index_path}...")
        cpu_index = faiss.read_index(index_path)
        
        # Try to use GPU if available
        try:
            from mirage.core.config import get_faiss_config
            _faiss_cfg = get_faiss_config()
            if _faiss_cfg.get('use_gpu', False):
                res = faiss.StandardGpuResources()
                gpu_id = _faiss_cfg.get('gpu_id', 0)
                index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
                print(f"✅ FAISS index transferred to GPU {gpu_id}")
            else:
                index = cpu_index
        except Exception as e:
            print(f"⚠️ Could not use FAISS GPU: {e}")
            index = cpu_index
        
        # Load metadata (use index_prefix for filename)
        metadata_path = f"{EMBEDDINGS_DIR}/{index_prefix}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        chunk_ids = metadata['chunk_ids']
        
        # Use thread-safe cached embedder (caching is now inside _get_embedder_for_model)
        embedder = _get_embedder_for_model(model_name)
        
        # Embed query with GPU lock for thread-safe access
        # Use convert_to_numpy=True to avoid device mismatch when model is on CPU
        print(f"Embedding query...")
        gpu_lock = getattr(this_module, '_gpu_lock', None)
        if gpu_lock:
            with gpu_lock:
                query_embedding = embedder.encode(query, convert_to_numpy=True)
        else:
            query_embedding = embedder.encode(query, convert_to_numpy=True)
        if isinstance(query_embedding, torch.Tensor):
            query_array = query_embedding.cpu().float().numpy()
        else:
            query_array = np.array(query_embedding, dtype=np.float32)
        if query_array.ndim == 1:
            query_array = query_array.reshape(1, -1)
        faiss.normalize_L2(query_array)
    
    # Retrieve based on method
    if retrieval_method == "top_k":
        print(f"\nRetrieving top-{retrieval_k} chunks...")
        scores, indices = index.search(query_array, retrieval_k)
        num_retrieve = retrieval_k
    elif retrieval_method == "top_p":
        # Retrieve more candidates initially for top-p filtering
        max_candidates = min(100, len(chunk_ids))
        scores, indices = index.search(query_array, max_candidates)
        
        # Apply top-p filtering
        scores_flat = scores[0]
        # Convert cosine similarity to probability distribution
        probs = np.exp(scores_flat) / np.sum(np.exp(scores_flat))
        cumsum = np.cumsum(probs)
        cutoff_idx = np.searchsorted(cumsum, retrieval_p) + 1
        
        scores = scores[:, :cutoff_idx]
        indices = indices[:, :cutoff_idx]
        num_retrieve = cutoff_idx
        print(f"\nRetrieving top-p (p={retrieval_p}) chunks: {num_retrieve} chunks selected...")
    else:
        raise ValueError(f"Unknown retrieval method: {retrieval_method}")
    
    # Get retrieved chunks
    retrieved_chunks = []
    for idx, score in zip(indices[0], scores[0]):
        # idx is the FAISS index which corresponds directly to the position in chunks list
        # chunk_id is document-local and may repeat across documents, so use idx directly
        chunk = chunks[idx]
        chunk_id = chunk_ids[idx]  # For reference/logging only
        
        # Extract file_name (source document) if available
        file_name = chunk.get('file_name', 'unknown')
        
        # Extract all image paths from artifact field (chunks.json contains artifact field)
        artifact_text = chunk.get('artifact', 'None')
        artifact_paths = extract_image_paths(artifact_text, file_name)  # List of image paths
        
        # If no images found in artifact field, also check content field for image references
        # (LLM might have put images in content but not extracted to artifact)
        if not artifact_paths:
            content_text = chunk.get('content', '')
            if content_text and ('![' in content_text and '](' in content_text):
                # Extract image paths from content field as fallback
                artifact_paths = extract_image_paths(content_text, file_name)
        
        image_path = artifact_paths[0] if artifact_paths else None  # First image for backward compatibility
        
        retrieved_chunks.append({
            'text': chunk['content'],
            'artifact': artifact_paths,  # List of image paths (zero, one, or more)
            'image_path': image_path,  # First image path for backward compatibility
            'chunk_id': chunk_id,
            'file_name': file_name,  # Track source document
            'score': float(score)
        })
    
    print(f"\nTop {num_retrieve} Retrieved Chunks (before reranking):")
    for i, chunk in enumerate(retrieved_chunks, 1):
        file_name = chunk.get('file_name', 'unknown')
        print(f"{i}. Document: {file_name}, Chunk ID: {chunk['chunk_id']}, Score: {chunk['score']:.4f}")
        print(f"   Text: {chunk['text'][:150]}...")
        print(f"   Has image: {chunk['image_path'] is not None}")
    
    # Rerank using MonoVLM (can swap with VLMReranker or TextEmbeddingReranker)
    print(f"\n{'='*60}")
    print(f"Reranking with MonoVLM to get top {rerank_top_k}...")
    print(f"{'='*60}")
    
    # Use cached reranker if available
    this_module = sys.modules[__name__]
    if hasattr(this_module, '_cached_reranker') and this_module._cached_reranker is not None:
        reranker = this_module._cached_reranker
        print(f"✅ Using cached MonoVLM reranker")
    else:
        reranker = MonoVLMReranker()
    
    # Use GPU lock if available for thread-safe access
    gpu_lock = getattr(this_module, '_gpu_lock', None)
    if gpu_lock:
        with gpu_lock:
            reranked_results = reranker.rerank(query, retrieved_chunks, top_k=rerank_top_k)
    else:
        reranked_results = reranker.rerank(query, retrieved_chunks, top_k=rerank_top_k)
    
    print(f"\nTop {rerank_top_k} Reranked Chunks:")
    for i, (orig_idx, relevance, chunk) in enumerate(reranked_results, 1):
        file_name = chunk.get('file_name', 'unknown')
        print(f"{i}. Document: {file_name}, Chunk ID: {chunk['chunk_id']}, Relevance: {relevance:.4f}")
        print(f"   Text: {chunk['text'][:150]}...")
        print(f"   Has image: {chunk['image_path'] is not None}")
    
    # Get final context
    final_context = reranked_results[:context_size]
    
    return final_context

# ============================================================================
# MULTI-HOP CONTEXT COMPLETION FUNCTIONS
# ============================================================================

def parse_verification_response(response: str) -> Tuple[str, Optional[List[str]], str]:
    """Parse the LLM verification response
    
    New format: <|#|>START<|#|>Status<|#|><status><|#|>Query<|#|><queries><|#|>Explanation<|#|><text><|#|>Concepts<|#|><concepts><|#|>END<|#|>
    
    Returns:
        (status, search_strings, explanation)
        status: "COMPLETE" or "INCOMPLETE"
        search_strings: List of search strings if incomplete, None if complete
        explanation: Explanation text
    """
    # Try new delimiter-based format first
    tuple_delimiter = "<|#|>"
    
    status = None
    search_strings = None
    explanation = ""
    
    if tuple_delimiter in response:
        parts = response.split(tuple_delimiter)
        for i, part in enumerate(parts):
            part_lower = part.strip().lower()
            if part_lower == "status" and i + 1 < len(parts):
                status = parts[i + 1].strip().upper()
            elif part_lower == "query" and i + 1 < len(parts):
                query_text = parts[i + 1].strip()
                if query_text.lower() != "none" and query_text:
                    search_strings = [s.strip().strip('"') for s in query_text.split('|') if s.strip()]
            elif part_lower == "explanation" and i + 1 < len(parts):
                explanation = parts[i + 1].strip()
    
    # Fallback: try old format
    if not status:
        status_match = re.search(r'Status:\s*(COMPLETE|INCOMPLETE)', response, re.IGNORECASE)
        if not status_match:
            raise ValueError(f"Could not parse status from response: {response}")
        status = status_match.group(1).upper()
    
    if not explanation:
        explanation_match = re.search(r'Explanation:\s*(.+?)(?:\n\n|$)', response, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else ""
    
    # Extract search strings (if incomplete) - fallback for old format
    if status == "INCOMPLETE" and not search_strings:
        query_match = re.search(r'Query:\s*([^,]+?)(?:,\s*Explanation:|$)', response, re.DOTALL)
        if query_match:
            query_text = query_match.group(1).strip()
            search_strings = [s.strip() for s in query_text.split('|') if s.strip()]
        
        # Fallback: generate search query from explanation if Query was missing
        if not search_strings and explanation:
            refs = re.findall(r'(Figure|Table|Annex|Formula)\s*[A-Z]?\.?\d+(?:\.\d+)?', explanation, re.IGNORECASE)
            if refs:
                search_strings = [ref for ref in refs[:5]]
                print(f"    ⚠️ Fallback: Generated search strings from explanation: {search_strings}")
    
    return status, search_strings, explanation


def verify_chunk_completeness(chunks: List[Dict], expert_persona: str, 
                               domain: str) -> Tuple[str, Optional[List[str]], str]:
    """Verify if a set of chunks provide complete context using VLM (interweaved)
    
    Args:
        chunks: List of chunks to verify
        expert_persona: Expert role for domain-specific evaluation
        domain: Domain context for evaluation
    
    Returns:
        (status, search_strings, explanation)
    """
    prompt_template = PROMPTS_CHUNK["completion_verification"]
    prompt = prompt_template.format(expert_persona=expert_persona, domain=domain)
    prompt += "\n\nAnalyze the following chunks to determine if the context is complete:"
    
    response = call_vlm_interweaved(prompt, chunks)
    return parse_verification_response(response)


def parse_addition_verification_response(response: str) -> Tuple[str, str]:
    """Parse the chunk addition verification response
    
    New format: <|#|>START<|#|>Status<|#|><status><|#|>Explanation<|#|><text><|#|>END<|#|>
    
    Returns:
        (status, explanation)
        status: "EXPLANATORY", "RELATED", or "UNRELATED"
        explanation: Explanation text
    """
    tuple_delimiter = "<|#|>"
    status = None
    explanation = ""
    
    # Try new delimiter-based format first
    if tuple_delimiter in response:
        parts = response.split(tuple_delimiter)
        for i, part in enumerate(parts):
            part_lower = part.strip().lower()
            if part_lower == "status" and i + 1 < len(parts):
                status_value = parts[i + 1].strip().upper()
                # Map to valid statuses
                if status_value in ["EXPLANATORY", "RELATED", "UNRELATED"]:
                    status = status_value
                elif status_value == "HELPFUL":
                    status = "EXPLANATORY"  # Legacy mapping
                elif status_value == "NOT_HELPFUL":
                    status = "UNRELATED"  # Legacy mapping
            elif part_lower == "explanation" and i + 1 < len(parts):
                explanation = parts[i + 1].strip()
    
    # Fallback: try old format
    if not status:
        status_match = re.search(r'Status:\s*(EXPLANATORY|RELATED|UNRELATED)', response, re.IGNORECASE)
        if status_match:
            status = status_match.group(1).upper()
        else:
            # Fallback: check for old HELPFUL/NOT_HELPFUL format
            old_status_match = re.search(r'Status:\s*(HELPFUL|NOT_HELPFUL)', response, re.IGNORECASE)
            if old_status_match:
                old_status = old_status_match.group(1).upper()
                status = "EXPLANATORY" if old_status == "HELPFUL" else "UNRELATED"
            else:
                return "UNRELATED", f"Could not parse response: {response[:200]}"
    
    if not explanation:
        explanation_match = re.search(r'Explanation:\s*(.+?)(?:\n\n|$)', response, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else ""
    
    return status, explanation


def verify_chunk_addition(original_chunks: List[Dict], search_query: str, 
                          candidate_chunk: Dict, expert_persona: str,
                          domain: str) -> Tuple[str, str]:
    """Verify if a candidate chunk should be added to the context for QA generation
    
    Args:
        original_chunks: List of current chunks (the original context being built)
        search_query: The search query that was used to find the candidate
        candidate_chunk: The retrieved chunk being considered for addition
        expert_persona: Expert role for domain-specific evaluation
        domain: Domain context for evaluation
    
    Returns:
        (status, explanation)
        status: "EXPLANATORY", "RELATED", or "UNRELATED"
        explanation: Classification reasoning
    """
    prompt_template = PROMPTS_CHUNK["chunk_addition_verification"]
    prompt = prompt_template.format(expert_persona=expert_persona, domain=domain)
    
    # Build the original chunk content for the prompt
    original_content = "\n\n---\n\n".join([c.get('content', '') for c in original_chunks])
    candidate_content = candidate_chunk.get('content', '')
    
    # Format the verification request
    verification_request = f"""
ORIGINAL CHUNK:
{original_content}

SEARCH QUERY: {search_query}

CANDIDATE CHUNK:
{candidate_content}
"""
    
    # Prepare chunks for VLM call (include images if present)
    chunks_for_vlm = []
    
    # Add original chunks with their images
    for chunk in original_chunks:
        chunks_for_vlm.append({
            'content': f"[ORIGINAL CONTEXT]\n{chunk.get('content', '')}",
            'image_path': chunk.get('image_path')
        })
    
    # Add candidate chunk with its image
    chunks_for_vlm.append({
        'content': f"[CANDIDATE CHUNK for query: {search_query}]\n{candidate_content}",
        'image_path': candidate_chunk.get('image_path')
    })
    
    full_prompt = prompt + "\n\n" + verification_request
    
    response = call_vlm_interweaved(full_prompt, chunks_for_vlm)
    return parse_addition_verification_response(response)


def batch_verify_chunk_additions(original_chunks: List[Dict], 
                                  candidates: List[Tuple[str, Dict]],
                                  expert_persona: str,
                                  domain: str) -> List[Tuple[str, str]]:
    """Batch verify multiple candidate chunks using concurrent API calls
    
    Args:
        original_chunks: List of current chunks (the original context being built)
        candidates: List of (search_query, candidate_chunk) tuples
        expert_persona: Expert role for domain-specific evaluation
        domain: Domain context for evaluation
        
    Returns:
        List of (status, explanation) tuples in same order as candidates
    """
    if not candidates:
        return []
    
    prompt_template = PROMPTS_CHUNK["chunk_addition_verification"]
    prompt_base = prompt_template.format(expert_persona=expert_persona, domain=domain)
    original_content = "\n\n---\n\n".join([c.get('content', '') for c in original_chunks])
    
    # Prepare batch requests
    requests = []
    for search_query, candidate_chunk in candidates:
        candidate_content = candidate_chunk.get('content', '')
        
        verification_request = f"""
ORIGINAL CHUNK:
{original_content}

SEARCH QUERY: {search_query}

CANDIDATE CHUNK:
{candidate_content}
"""
        
        # Prepare chunks for VLM call
        chunks_for_vlm = []
        for chunk in original_chunks:
            chunks_for_vlm.append({
                'content': f"[ORIGINAL CONTEXT]\n{chunk.get('content', '')}",
                'image_path': chunk.get('image_path')
            })
        chunks_for_vlm.append({
            'content': f"[CANDIDATE CHUNK for query: {search_query}]\n{candidate_content}",
            'image_path': candidate_chunk.get('image_path')
        })
        
        full_prompt = prompt_base + "\n\n" + verification_request
        requests.append((full_prompt, chunks_for_vlm))
    
    # Execute batch call
    print(f"    ⚡ Batch verifying {len(requests)} candidates...")
    responses = batch_call_vlm_interweaved(requests, show_progress=False)
    
    # Parse all responses
    results = []
    for response in responses:
        if response and not response.startswith("ERROR:"):
            results.append(parse_addition_verification_response(response))
        else:
            results.append(("UNRELATED", f"Error: {response}"))
    
    return results


def retrieve_chunks_for_query(query: str, top_k: int = 2) -> List[Dict]:
    """Retrieve top-k chunks for a search query
    
    Returns:
        List of chunks with 'text', 'chunk_id', 'image_path', 'score'
    """
    # Use the existing retrieve_and_rerank function
    results = retrieve_and_rerank(
        query=query,
        # model_name uses config default (EMBEDDING_MODEL)
        retrieval_method="top_k",
        retrieval_k=10,  # Retrieve more, then rerank
        rerank_top_k=5,
        context_size=top_k
    )
    
    # Convert to our format
    # Note: chunks from retrieve_and_rerank already have artifact (list) and image_path set
    chunks = []
    for orig_idx, relevance, chunk in results:
        chunks.append({
            'content': chunk['text'],
            'chunk_id': chunk['chunk_id'],
            'file_name': chunk.get('file_name', 'unknown'),
            'artifact': chunk.get('artifact', []),  # List of image paths
            'image_path': chunk.get('image_path'),  # First image path for backward compatibility
            'score': relevance
        })
    
    return chunks


def build_complete_context(
    initial_chunk: Union[str, Dict],
    max_depth: int = MAX_DEPTH,
    max_breadth: int = MAX_BREADTH,
    chunks_per_search: int = CHUNKS_PER_SEARCH,
    expert_persona: str = None,
    domain: str = None,
    log_details: bool = True,
    chunk_addition_mode: str = CHUNK_ADDITION_MODE
) -> Dict:
    """Build complete context through iterative verification and retrieval
    
    Args:
        initial_chunk: The starting chunk (string or dict)
        max_depth: Maximum iterations
        max_breadth: Maximum search strings per verification
        chunks_per_search: Number of chunks to retrieve per search string
        expert_persona: Expert role for domain-specific evaluation
        domain: Domain context for evaluation
        log_details: Whether to include detailed iteration logs in output
        chunk_addition_mode: "EXPLANATORY" (only direct answers) or "RELATED" (both)
    
    Returns:
        Dict with:
            - 'status': 'COMPLETE' or 'INCOMPLETE'
            - 'context': Combined context text
            - 'chunks': List of chunk dicts used
            - 'depth': Final depth reached
            - 'chunks_added': List of chunk IDs added
            - 'search_history': List of search strings used
            - 'iteration_logs': Detailed logs per iteration (if log_details=True)
    """
    print("="*80)
    print("MULTI-HOP CONTEXT COMPLETION")
    print(f"Chunk Addition Mode: {chunk_addition_mode}")
    print("="*80)
    
    # Determine which statuses to accept based on mode
    if chunk_addition_mode.upper() == "EXPLANATORY":
        accepted_statuses = ("EXPLANATORY",)
    else:  # Default to RELATED (accept both)
        accepted_statuses = ("EXPLANATORY", "RELATED")
    
    # Initialize
    if isinstance(initial_chunk, str):
        # Fallback if string passed
        current_chunks = [{
            'content': initial_chunk,
            'artifact': [],  # Empty list - no images
            'image_path': None,
            'chunk_id': 'initial',
            'file_name': 'unknown'
        }]
    else:
        # Expect dict with content and artifact/image_path
        chunk_content = initial_chunk.get('content', '')
        file_name = initial_chunk.get('file_name', 'unknown')
        
        # Extract artifact as list of image paths
        artifact_text = initial_chunk.get('artifact', 'None')
        artifact_paths = extract_image_paths(artifact_text, file_name)  # List of image paths
        
        # If no images found in artifact field, also check content field for image references
        # (LLM might have put images in content but not extracted to artifact)
        if not artifact_paths:
            if chunk_content and ('![' in chunk_content and '](' in chunk_content):
                # Extract image paths from content field as fallback
                artifact_paths = extract_image_paths(chunk_content, file_name)
        
        # For backward compatibility, set image_path to first image
        image_path = artifact_paths[0] if artifact_paths else initial_chunk.get('image_path')

        current_chunks = [{
            'content': chunk_content,
            'artifact': artifact_paths,  # List of image paths (zero, one, or more)
            'image_path': image_path,  # First image path for backward compatibility
            'chunk_id': initial_chunk.get('chunk_id', 'initial'),
            'file_name': file_name  # Track source document
        }]

    depth = 0
    chunks_added = []  # List of dicts: {'file_name': str, 'chunk_id': str}
    search_history = []
    max_breadth_used = 0  # Track maximum search strings used in any iteration
    iteration_logs = []  # Detailed logs for each iteration
    consecutive_no_progress = 0  # Circuit breaker: track consecutive iterations with no new chunks
    
    # Helper to get combined text for return value
    def get_combined_text(chunks):
        return "\n\n".join([c['content'] for c in chunks])
    
    # Helper to check if chunk already added (by file_name + chunk_id)
    def is_chunk_added(file_name, chunk_id):
        return any(c.get('file_name') == file_name and c.get('chunk_id') == chunk_id 
                  for c in chunks_added)
    
    # Helper to add chunk to tracking
    def add_chunk_to_tracking(file_name, chunk_id):
        chunks_added.append({'file_name': file_name, 'chunk_id': chunk_id})
    
    # Track initial chunk if it has file_name
    initial_file_name = None
    if isinstance(initial_chunk, dict) and 'file_name' in initial_chunk:
        initial_file_name = initial_chunk['file_name']
    elif isinstance(initial_chunk, dict) and 'chunk_id' in initial_chunk:
        # Try to get file_name from the chunk if available
        initial_file_name = initial_chunk.get('file_name', 'unknown')
    
    if initial_file_name:
        initial_chunk_id = current_chunks[0].get('chunk_id', 'initial')
        add_chunk_to_tracking(initial_file_name, initial_chunk_id)
    
    print(f"\n📄 Initial chunk length: {len(current_chunks[0]['content'])} chars")
    
    while depth < max_depth:
        depth += 1
        print(f"\n{'='*80}")
        print(f"DEPTH {depth}/{max_depth}")
        print(f"{'='*80}")
        
        # Initialize iteration log
        iter_log = {
            'depth': depth,
            'status': None,
            'explanation': None,
            'search_strings': None,
            'retrieved_chunks': [],
            'chunks_added_this_iteration': []
        }
        
        # Verify completeness
        print(f"\n🔍 Verifying chunk completeness...")
        status, search_strings, explanation = verify_chunk_completeness(
            current_chunks, expert_persona=expert_persona, domain=domain
        )
        
        print(f"Status: {status}")
        print(f"Explanation: {explanation}")
        
        # Log verification result
        iter_log['status'] = status
        iter_log['explanation'] = explanation
        iter_log['search_strings'] = search_strings
        
        if status == "COMPLETE":
            hop_count = len(chunks_added) - 1  # Each hop adds one link
            print(f"\n✅ Context is COMPLETE at depth {depth}, hop_count={hop_count}")
            iteration_logs.append(iter_log)
            return {
                'status': 'COMPLETE',
                'context': get_combined_text(current_chunks),
                'chunks': current_chunks,
                'depth': depth,
                'hop_count': hop_count,
                'max_breadth_used': max_breadth_used,
                'chunks_added': chunks_added,
                'search_history': search_history,
                'termination_reason': 'Context verified as complete',
                'iteration_logs': iteration_logs if log_details else None
            }
        
        # Context is incomplete
        if not search_strings:
            print(f"\n⚠️ No search strings generated despite INCOMPLETE status. Stopping.")
            hop_count = len(chunks_added) - 1
            iteration_logs.append(iter_log)
            return {
                'status': 'INCOMPLETE_NO_SEARCH_STRINGS',
                'context': get_combined_text(current_chunks),
                'chunks': current_chunks,
                'depth': depth,
                'hop_count': hop_count,
                'max_breadth_used': max_breadth_used,
                'chunks_added': chunks_added,
                'search_history': search_history,
                'termination_reason': 'LLM marked chunk as INCOMPLETE but failed to generate search strings',
                'iteration_logs': iteration_logs if log_details else None
            }
        
        # Limit breadth and track max used
        search_strings = search_strings[:max_breadth]
        max_breadth_used = max(max_breadth_used, len(search_strings))
        print(f"\n🔎 Generated {len(search_strings)} search strings:")
        for i, s in enumerate(search_strings, 1):
            print(f"  {i}. {s}")
        
        # Retrieve and verify chunks for each search string - BATCH PROCESSING
        print(f"\n📥 Retrieving top {chunks_per_search} chunks per search string...")
        new_chunks = []
        any_relevant_found = False  # Track if ANY chunk was verified as EXPLANATORY or RELATED
        
        # Phase 1: Collect all candidates to verify
        candidates_to_verify = []  # List of (search_string, chunk) tuples
        candidate_info = []  # For logging
        
        for i, search_string in enumerate(search_strings, 1):
            print(f"\n  Search {i}/{len(search_strings)}: {search_string}")
            search_history.append(search_string)
            
            try:
                retrieved = retrieve_chunks_for_query(search_string, top_k=chunks_per_search)
                
                for j, chunk in enumerate(retrieved, 1):
                    chunk_id = chunk['chunk_id']
                    file_name = chunk.get('file_name', 'unknown')
                    
                    # Check if chunk already added (by file_name + chunk_id combination)
                    initial_chunk_id = current_chunks[0].get('chunk_id', 'initial')
                    initial_file_name = current_chunks[0].get('file_name', 'unknown')
                    
                    if (not is_chunk_added(file_name, chunk_id) and 
                        not (file_name == initial_file_name and chunk_id == initial_chunk_id)):
                        candidates_to_verify.append((search_string, chunk))
                        candidate_info.append((file_name, chunk_id, chunk['score']))
                        print(f"    📋 Queued chunk {file_name}:{chunk_id} (score: {chunk['score']:.4f}) for verification")
                    else:
                        print(f"    ⊘ Chunk {file_name}:{chunk_id} already added")
            
            except Exception as e:
                import traceback
                print(f"    ⚠️ Error retrieving chunks: {type(e).__name__}: {e}")
                traceback.print_exc()
        
        # Phase 2: Batch verify all candidates
        if candidates_to_verify:
            if len(candidates_to_verify) > 1:
                # Use batch verification for multiple candidates
                verification_results = batch_verify_chunk_additions(
                    current_chunks, candidates_to_verify, 
                    expert_persona=expert_persona, domain=domain
                )
            else:
                # Single candidate - use regular verification
                search_string, chunk = candidates_to_verify[0]
                status, explanation = verify_chunk_addition(
                    current_chunks, search_string, chunk,
                    expert_persona=expert_persona, domain=domain
                )
                verification_results = [(status, explanation)]
            
            # Process results and log
            for (search_string, chunk), (status, explanation), (file_name, chunk_id, score) in zip(
                candidates_to_verify, verification_results, candidate_info
            ):
                # Log retrieved chunk info
                chunk_log = {
                    'search_query': search_string,
                    'chunk_id': chunk_id,
                    'file_name': file_name,
                    'score': float(score),
                    'verdict': status,
                    'reason': explanation[:200] if explanation else ''
                }
                iter_log['retrieved_chunks'].append(chunk_log)
                
                if status in accepted_statuses:
                    add_chunk_to_tracking(file_name, chunk_id)
                    # Add classification tag to chunk for context building
                    chunk['classification'] = status
                    new_chunks.append(chunk)
                    any_relevant_found = True
                    iter_log['chunks_added_this_iteration'].append({
                        'file_name': file_name, 
                        'chunk_id': chunk_id,
                        'classification': status
                    })
                    emoji = "🎯" if status == "EXPLANATORY" else "🔗"
                    print(f"    {emoji} {status}: {file_name}:{chunk_id} - {explanation[:100]}...")
                elif status == "RELATED" and chunk_addition_mode.upper() == "EXPLANATORY":
                    # Log RELATED chunks that were skipped due to mode
                    print(f"    ⏭️ RELATED (skipped - mode={chunk_addition_mode}): {file_name}:{chunk_id} - {explanation[:100]}...")
                else:
                    print(f"    ❌ UNRELATED: {file_name}:{chunk_id} - {explanation[:100]}...")
        
        # Save iteration log
        iteration_logs.append(iter_log)
        
        # Add verified relevant chunks (EXPLANATORY or RELATED) to context
        if new_chunks:
            explanatory_count = sum(1 for c in new_chunks if c.get('classification') == 'EXPLANATORY')
            related_count = sum(1 for c in new_chunks if c.get('classification') == 'RELATED')
            print(f"\n📝 Adding {len(new_chunks)} verified chunks to context ({explanatory_count} EXPLANATORY, {related_count} RELATED)")
            current_chunks.extend(new_chunks)
            consecutive_no_progress = 0  # Reset circuit breaker
        else:
            # Circuit breaker: track consecutive iterations with no progress
            consecutive_no_progress += 1
            
            # No relevant chunks found across ALL search strings
            if not any_relevant_found:
                print(f"\n🛑 No relevant chunks found across all {len(search_strings)} search strings.")
                print(f"   The corpus likely does not contain the information needed to complete this context.")
                print(f"   Stopping multi-hop context building for this chunk.")
                hop_count = len(chunks_added) - 1
                return {
                    'status': 'INCOMPLETE_NO_RELEVANT_CHUNKS',
                    'context': get_combined_text(current_chunks),
                    'chunks': current_chunks,
                    'depth': depth,
                    'hop_count': hop_count,
                    'max_breadth_used': max_breadth_used,
                    'chunks_added': chunks_added,
                    'search_history': search_history,
                    'termination_reason': 'No retrieved chunks were classified as EXPLANATORY or RELATED',
                    'iteration_logs': iteration_logs if log_details else None
                }
            elif consecutive_no_progress >= MAX_CONSECUTIVE_NO_PROGRESS:
                # Circuit breaker triggered
                print(f"\n🛑 Circuit breaker triggered: {consecutive_no_progress} consecutive iterations with no new chunks.")
                print(f"   Stopping to prevent runaway API calls.")
                hop_count = len(chunks_added) - 1
                return {
                    'status': 'INCOMPLETE_CIRCUIT_BREAKER',
                    'context': get_combined_text(current_chunks),
                    'chunks': current_chunks,
                    'depth': depth,
                    'hop_count': hop_count,
                    'max_breadth_used': max_breadth_used,
                    'chunks_added': chunks_added,
                    'search_history': search_history,
                    'termination_reason': f'Circuit breaker: {consecutive_no_progress} consecutive iterations with no new chunks',
                    'iteration_logs': iteration_logs if log_details else None
                }
            else:
                print(f"\n⚠️ No new chunks to add (all duplicates). Consecutive no-progress: {consecutive_no_progress}/{MAX_CONSECUTIVE_NO_PROGRESS}")
                hop_count = len(chunks_added) - 1
                return {
                    'status': 'INCOMPLETE_ALL_DUPLICATES',
                    'context': get_combined_text(current_chunks),
                    'chunks': current_chunks,
                    'depth': depth,
                    'hop_count': hop_count,
                    'max_breadth_used': max_breadth_used,
                    'chunks_added': chunks_added,
                    'search_history': search_history,
                    'termination_reason': 'All retrieved chunks were duplicates of already-added chunks',
                    'iteration_logs': iteration_logs if log_details else None
                }
    
    # Max depth reached (only if loop completes without early termination)
    hop_count = len(chunks_added) - 1
    print(f"\n⚠️ Max depth {max_depth} reached. Context may still be INCOMPLETE. hop_count={hop_count}")
    
    return {
        'status': 'INCOMPLETE_MAX_DEPTH',
        'context': get_combined_text(current_chunks),
        'chunks': current_chunks,
        'depth': depth,
        'hop_count': hop_count,
        'max_breadth_used': max_breadth_used,
        'chunks_added': chunks_added,
        'search_history': search_history,
        'termination_reason': f'Maximum depth of {max_depth} iterations reached',
        'iteration_logs': iteration_logs if log_details else None
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Setup logging
    setup_logging()
    
    # Check command line argument for mode
    mode = sys.argv[1] if len(sys.argv) > 1 else "multihop"
    
    if mode == "simple":
        # Test simple retrieval
        print("="*80)
        print("TESTING SIMPLE RETRIEVAL")
        print("="*80)
        
        query = "How to minimize the normalized losses of a motor in Y connection?"
        
        print(f"\nQUERY: {query}")
        print("="*80)
        print(f"\nHyperparameters:")
        print(f"  Retrieval method: {RETRIEVAL_METHOD}")
        if RETRIEVAL_METHOD == "top_k":
            print(f"  Retrieval k: {RETRIEVAL_K}")
        else:
            print(f"  Retrieval p: {RETRIEVAL_P}")
        print(f"  Rerank top-k: {RERANK_TOP_K}")
        print(f"  Context size: {CONTEXT_SIZE}")
        print("="*80)
        
        # Retrieve and rerank
        context = retrieve_and_rerank(
            query, 
            # model_name uses config default (EMBEDDING_MODEL)
            retrieval_method=RETRIEVAL_METHOD,
            retrieval_k=RETRIEVAL_K,
            retrieval_p=RETRIEVAL_P,
            rerank_top_k=RERANK_TOP_K,
            context_size=CONTEXT_SIZE
        )
        
        # Print final context
        print("\n" + "="*80)
        print(f"FINAL CONTEXT (Top {CONTEXT_SIZE})")
        print("="*80)
        
        for i, (orig_idx, relevance, chunk) in enumerate(context, 1):
            print(f"\n--- Context Chunk {i} ---")
            print(f"Chunk ID: {chunk['chunk_id']}")
            print(f"Relevance Score: {relevance:.4f}")
            print(f"Has Image: {chunk['image_path'] is not None}")
            if chunk['image_path']:
                print(f"Image Path: {chunk['image_path']}")
            print(f"\nContent:\n{chunk['text']}")
            print("-" * 80)
    
    else:
        # Test multi-hop context completion
        print("="*80)
        print("TESTING MULTI-HOP CONTEXT COMPLETION")
        print("="*80)
        
        # Load first 10 chunks from JSON file
        print(f"\nLoading first 10 chunks from {CHUNKS_FILE}...")
        with open(CHUNKS_FILE, 'r') as f:
            all_chunks = json.load(f)
        
        test_chunks = all_chunks
        print(f"Loaded {len(test_chunks)} chunks for testing")
        
        # Test with each chunk
        for i, chunk in enumerate(test_chunks):
            print(f"\n{'='*80}")
            print(f"TESTING CHUNK {i+1}/{len(test_chunks)}")
            print(f"Chunk ID: {chunk['chunk_id']}, Type: {chunk['chunk_type']}")
            print(f"{'='*80}")
            
            # Extract image path from artifact if it's a standalone image
            image_path = None
            file_name = chunk.get('file_name', 'unknown')
            artifact_field = chunk.get('artifact', 'None')
            if artifact_field != "None" and artifact_field:
                image_path = extract_image_path(artifact_field, file_name)
            
            # If no image found in artifact field, also check content field for image references
            if not image_path:
                content_text = chunk.get('content', '')
                if content_text and ('![' in content_text and '](' in content_text):
                    image_path = extract_image_path(content_text, file_name)
            
            test_chunk_dict = {
                'content': chunk['content'],
                'chunk_id': chunk['chunk_id'],
                'image_path': image_path,
                'artifact': chunk.get('artifact'),
                'file_name': file_name
            }
            
            result = build_complete_context(
                initial_chunk=test_chunk_dict,
                max_depth=3,
                max_breadth=3,
                chunks_per_search=2
            )
            
            print(f"\n{'-'*80}")
            print(f"RESULT FOR CHUNK {chunk['chunk_id']}")
            print(f"{'-'*80}")
            print(f"Status: {result['status']}")
            print(f"Depth reached: {result['depth']}")
            print(f"Chunks added: {len(result['chunks_added'])}")
            print(f"Chunk IDs: {result['chunks_added']}")
            print(f"Search history: {len(result['search_history'])} searches")
            print(f"Final context length: {len(result['context'])} chars")
            print(f"\nFirst 300 chars of final context:")
            print(result['context'][:300])
            print("...")