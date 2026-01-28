"""
Preflight Check Module - Validates all services before pipeline execution.

Checks:
1. LLM API connectivity (text generation)
2. VLM API connectivity (vision + text)
3. Embedding model loading & inference
4. Reranker model loading & inference  
5. API key availability
6. Required directories and files

Run standalone: python preflight_check.py
Or import: from mirage.utils.preflight import run_preflight_checks
"""

import os
import sys
import time
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

class CheckStatus(Enum):
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    WARN = "⚠️ WARN"
    SKIP = "⏭️ SKIP"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0


def _timer(func):
    """Decorator to time check functions."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        result.duration_ms = (time.time() - start) * 1000
        return result
    return wrapper


# ============================================================================
# INDIVIDUAL CHECK FUNCTIONS
# ============================================================================

@_timer
def check_config() -> CheckResult:
    """Check if config.yaml can be loaded."""
    try:
        from mirage.core.config import load_config, get_backend_config, get_paths_config, get_embedding_config
        config = load_config()
        backend = get_backend_config()
        paths = get_paths_config()
        embed = get_embedding_config()
        
        # Use env var override if set (from CLI --output), then fall back to config
        output_dir = os.environ.get("MIRAGE_OUTPUT_DIR") or paths.get('output_dir', 'UNKNOWN')
        
        return CheckResult(
            name="Configuration",
            status=CheckStatus.PASS,
            message="config.yaml loaded successfully",
            details={
                "backend": backend.get('name', 'UNKNOWN'),
                "llm_model": backend.get('llm_model', 'UNKNOWN'),
                "vlm_model": backend.get('vlm_model', 'UNKNOWN'),
                "embedding_model": embed.get('model', 'UNKNOWN'),
                "output_dir": output_dir
            }
        )
    except FileNotFoundError as e:
        return CheckResult(
            name="Configuration", 
            status=CheckStatus.FAIL,
            message=f"config.yaml not found: {e}"
        )
    except Exception as e:
        return CheckResult(
            name="Configuration",
            status=CheckStatus.FAIL,
            message=f"Failed to load config: {e}"
        )


@_timer
def check_api_key() -> CheckResult:
    """Check if API key is available and non-empty."""
    try:
        from mirage.core.llm import _initialize_config, BACKEND
        _initialize_config()  # Ensure config is loaded first
        from mirage.core.llm import API_KEY
        
        if API_KEY and len(API_KEY) > 10:
            # Mask the key for display
            masked = f"{API_KEY[:8]}...{API_KEY[-4:]}" if len(API_KEY) > 12 else "***"
            return CheckResult(
                name="API Key",
                status=CheckStatus.PASS,
                message=f"API key loaded for {BACKEND}",
                details={"backend": BACKEND, "key_preview": masked}
            )
        else:
            return CheckResult(
                name="API Key",
                status=CheckStatus.FAIL,
                message=f"API key missing or too short for {BACKEND}",
                details={"backend": BACKEND}
            )
    except ImportError as e:
        return CheckResult(
            name="API Key",
            status=CheckStatus.FAIL,
            message=f"Cannot import call_llm: {e}"
        )


@_timer
def check_llm_call() -> CheckResult:
    """Test LLM API with a minimal call."""
    try:
        from mirage.core.llm import call_llm_simple, BACKEND, LLM_MODEL_NAME
        
        test_prompt = "Say 'OK' and nothing else."
        response = call_llm_simple(test_prompt)
        
        if response and len(response.strip()) > 0:
            return CheckResult(
                name="LLM API",
                status=CheckStatus.PASS,
                message=f"LLM call successful ({BACKEND})",
                details={
                    "backend": BACKEND,
                    "model": LLM_MODEL_NAME,
                    "response_preview": response[:50] + "..." if len(response) > 50 else response
                }
            )
        else:
            return CheckResult(
                name="LLM API",
                status=CheckStatus.FAIL,
                message="LLM returned empty response",
                details={"backend": BACKEND, "model": LLM_MODEL_NAME}
            )
    except Exception as e:
        return CheckResult(
            name="LLM API",
            status=CheckStatus.FAIL,
            message=f"LLM call failed: {e}",
            details={"error": str(e)}
        )


@_timer
def check_vlm_call() -> CheckResult:
    """Test VLM API with a minimal text-only call (no image needed for connectivity check)."""
    try:
        from mirage.core.llm import call_vlm_interweaved, BACKEND, VLM_MODEL_NAME
        
        # Test with text-only context (simulates VLM call without actual image)
        test_prompt = "Say 'VLM OK' and nothing else."
        test_chunks = [{"content": "Test content", "image_path": None}]
        
        response = call_vlm_interweaved(test_prompt, test_chunks)
        
        if response and len(response.strip()) > 0:
            return CheckResult(
                name="VLM API",
                status=CheckStatus.PASS,
                message=f"VLM call successful ({BACKEND})",
                details={
                    "backend": BACKEND,
                    "model": VLM_MODEL_NAME,
                    "response_preview": response[:50] + "..." if len(response) > 50 else response
                }
            )
        else:
            return CheckResult(
                name="VLM API",
                status=CheckStatus.FAIL,
                message="VLM returned empty response",
                details={"backend": BACKEND, "model": VLM_MODEL_NAME}
            )
    except Exception as e:
        return CheckResult(
            name="VLM API",
            status=CheckStatus.FAIL,
            message=f"VLM call failed: {e}",
            details={"error": str(e)}
        )


@_timer  
def check_embedding_model() -> CheckResult:
    """Test embedding model loading and inference."""
    try:
        from mirage.core.config import get_embedding_config
        embed_config = get_embedding_config()
        model_name = embed_config.get('model', 'bge_m3')
        
        test_text = "This is a test sentence for embedding."
        
        # Handle "auto" or "multimodal" - use fallback logic
        if model_name in ["auto", "multimodal", "", None]:
            from mirage.embeddings.models import get_multimodal_embedder
            gpus = embed_config.get('gpus', None)
            embedder, actual_model, is_multimodal = get_multimodal_embedder(gpus=gpus)
            model_display = f"{actual_model} (multimodal={is_multimodal})"
            # Use encode() which all embedders support
            embedding = embedder.encode(test_text, convert_to_numpy=True)
        elif model_name == "qwen3_vl":
            from mirage.embeddings.models import Qwen3VLEmbed
            gpus = embed_config.get('gpus', None)
            embedder = Qwen3VLEmbed(gpus=gpus)
            model_display = "Qwen3-VL Multimodal"
            embedding = embedder.embed_text(test_text)
            if hasattr(embedding, 'cpu'):
                embedding = embedding.cpu().float().numpy()
        elif model_name in ["nomic", "nomic-ai/nomic-embed-multimodal-7b"]:
            from mirage.embeddings.models import NomicVLEmbed
            gpus = embed_config.get('gpus', None)
            embedder = NomicVLEmbed(gpus=gpus)
            model_display = "Nomic Multimodal"
            embedding = embedder.embed_text(test_text)
            if hasattr(embedding, 'cpu'):
                embedding = embedding.cpu().float().numpy()
        elif model_name in ["bge_m3", "BAAI/bge-m3"]:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer("BAAI/bge-m3", trust_remote_code=True)
            model_display = "BAAI/bge-m3"
            embedding = embedder.encode(test_text, convert_to_numpy=True)
        else:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer(model_name, trust_remote_code=True)
            model_display = model_name
            embedding = embedder.encode(test_text, convert_to_numpy=True)
        
        if embedding is not None and len(embedding) > 0:
            dim = len(embedding) if hasattr(embedding, '__len__') else embedding.shape[-1]
            return CheckResult(
                name="Embedding Model",
                status=CheckStatus.PASS,
                message=f"Embedding model loaded and working",
                details={
                    "model": model_display,
                    "config_name": model_name,
                    "embedding_dim": dim
                }
            )
        else:
            return CheckResult(
                name="Embedding Model",
                status=CheckStatus.FAIL,
                message="Embedding returned empty result",
                details={"model": model_display}
            )
    except ImportError as e:
        return CheckResult(
            name="Embedding Model",
            status=CheckStatus.FAIL,
            message=f"Cannot import embedding module: {e}",
            details={"error": str(e)}
        )
    except Exception as e:
        return CheckResult(
            name="Embedding Model",
            status=CheckStatus.FAIL,
            message=f"Embedding model failed: {e}",
            details={"error": str(e)}
        )


@_timer
def check_reranker() -> CheckResult:
    """Test reranker model loading."""
    try:
        from mirage.core.config import load_config
        config = load_config()
        reranker_config = config.get('reranker', {})
        default_reranker = reranker_config.get('default', 'gemini_vlm')
        
        if default_reranker == "gemini_vlm":
            # Gemini VLM reranker uses API, already tested in VLM check
            # Get actual model name from config
            try:
                from mirage.core.llm import VLM_MODEL_NAME, _initialize_config
                _initialize_config()
                model_name = VLM_MODEL_NAME if VLM_MODEL_NAME else "gemini-2.0-flash"
            except:
                model_name = "gemini-2.0-flash"
            return CheckResult(
                name="Reranker",
                status=CheckStatus.PASS,
                message="Using Gemini VLM reranker (API-based)",
                details={"type": "gemini_vlm", "model": model_name}
            )
        elif default_reranker in ["monovlm", "MonoVLM"]:
            from mirage.embeddings.rerankers_multimodal import MonoVLMReranker
            reranker = MonoVLMReranker()
            
            # Test with minimal query
            test_query = "test query"
            test_chunks = [{"content": "test content", "chunk_id": "1"}]
            rankings = reranker.rerank(test_query, test_chunks, top_k=1)
            
            return CheckResult(
                name="Reranker",
                status=CheckStatus.PASS,
                message="MonoVLM reranker loaded and working",
                details={"type": "monovlm", "model": "lightonai/MonoQwen2-VL-v0.1"}
            )
        else:
            return CheckResult(
                name="Reranker",
                status=CheckStatus.WARN,
                message=f"Unknown reranker type: {default_reranker}",
                details={"type": default_reranker}
            )
    except Exception as e:
        return CheckResult(
            name="Reranker",
            status=CheckStatus.FAIL,
            message=f"Reranker check failed: {e}",
            details={"error": str(e)}
        )


@_timer
def check_metrics_embeddings() -> CheckResult:
    """Test metrics evaluation embeddings (for answer_relevancy and semantic_diversity)."""
    try:
        from mirage.evaluation.metrics_optimized import GEMINI_AVAILABLE, SENTENCE_TRANSFORMERS_AVAILABLE
        
        # Check which embedding backend is available
        if GEMINI_AVAILABLE:
            from mirage.core.llm import API_KEY
            if API_KEY:
                # Try to initialize Gemini embeddings
                try:
                    from langchain_google_genai import GoogleGenerativeAIEmbeddings
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/text-embedding-004", 
                        google_api_key=API_KEY
                    )
                    test_emb = embeddings.embed_query("test")
                    return CheckResult(
                        name="Metrics Embeddings",
                        status=CheckStatus.PASS,
                        message="Using Gemini API embeddings for metrics",
                        details={"backend": "gemini", "model": "text-embedding-004", "dim": len(test_emb)}
                    )
                except Exception as e:
                    pass  # Fall through to sentence-transformers
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            from mirage.evaluation.metrics_optimized import LocalEmbeddingWrapper
            embeddings = LocalEmbeddingWrapper("BAAI/bge-m3")
            test_emb = embeddings.embed_query("test")
            return CheckResult(
                name="Metrics Embeddings",
                status=CheckStatus.PASS,
                message="Using local BGE-M3 embeddings for metrics",
                details={"backend": "sentence-transformers", "model": "BAAI/bge-m3", "dim": len(test_emb)}
            )
        
        return CheckResult(
            name="Metrics Embeddings",
            status=CheckStatus.FAIL,
            message="No embedding backend available for metrics (answer_relevancy will be 0)",
            details={"gemini_available": GEMINI_AVAILABLE, "st_available": SENTENCE_TRANSFORMERS_AVAILABLE}
        )
        
    except Exception as e:
        return CheckResult(
            name="Metrics Embeddings",
            status=CheckStatus.FAIL,
            message=f"Metrics embeddings check failed: {e}",
            details={"error": str(e)}
        )


@_timer
def check_gpu_availability() -> CheckResult:
    """Check GPU availability and memory."""
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpus = []
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                free_mem = torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i)
                total_mem = props.total_memory / (1024**3)
                gpus.append({
                    "id": i,
                    "name": props.name,
                    "total_gb": round(total_mem, 1)
                })
            
            return CheckResult(
                name="GPU",
                status=CheckStatus.PASS,
                message=f"Found {gpu_count} GPU(s)",
                details={"gpus": gpus}
            )
        else:
            return CheckResult(
                name="GPU",
                status=CheckStatus.WARN,
                message="No GPU available, will use CPU (slower)",
                details={}
            )
    except ImportError:
        return CheckResult(
            name="GPU",
            status=CheckStatus.WARN,
            message="PyTorch not available for GPU check",
            details={}
        )
    except Exception as e:
        return CheckResult(
            name="GPU",
            status=CheckStatus.WARN,
            message=f"GPU check error: {e}",
            details={"error": str(e)}
        )


@_timer
def check_output_directory() -> CheckResult:
    """Check if output directory is writable."""
    try:
        from mirage.core.config import get_paths_config
        paths = get_paths_config()
        
        # Use env var override if set (from CLI --output), then fall back to config
        output_dir = os.environ.get("MIRAGE_OUTPUT_DIR") or paths.get('output_dir', 'trials/results')
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Test write
        test_file = os.path.join(output_dir, ".preflight_test")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        
        return CheckResult(
            name="Output Directory",
            status=CheckStatus.PASS,
            message=f"Output directory writable",
            details={"path": output_dir}
        )
    except Exception as e:
        return CheckResult(
            name="Output Directory",
            status=CheckStatus.FAIL,
            message=f"Cannot write to output directory: {e}",
            details={"error": str(e)}
        )


# Module-level override for input directory (set before running checks)
_input_dir_override: Optional[str] = None

@_timer
def check_input_data() -> CheckResult:
    """Check if input data exists."""
    try:
        from mirage.core.config import get_paths_config
        paths = get_paths_config()

        # Use override if set, then check environment variable, then use config
        input_pdf_dir = _input_dir_override or os.environ.get("MIRAGE_INPUT_DIR") or paths.get('input_pdf_dir')
        input_chunks_file = paths.get('input_chunks_file')
        
        if input_chunks_file and os.path.exists(input_chunks_file):
            import json
            with open(input_chunks_file, 'r') as f:
                chunks = json.load(f)
            return CheckResult(
                name="Input Data",
                status=CheckStatus.PASS,
                message=f"Found pre-chunked data with {len(chunks)} chunks",
                details={"source": "chunks_file", "path": input_chunks_file, "count": len(chunks)}
            )
        
        if input_pdf_dir and os.path.exists(input_pdf_dir):
            # Check for PDF files
            pdf_files = list(Path(input_pdf_dir).glob("*.pdf"))
            # Also check for HTML files (News domain uses HTML)
            html_files = list(Path(input_pdf_dir).glob("*.html")) + list(Path(input_pdf_dir).glob("*.htm"))
            total_files = len(pdf_files) + len(html_files)
            
            if total_files > 0:
                file_types = []
                if pdf_files:
                    file_types.append(f"{len(pdf_files)} PDF")
                if html_files:
                    file_types.append(f"{len(html_files)} HTML")
                return CheckResult(
                    name="Input Data",
                    status=CheckStatus.PASS,
                    message=f"Found {', '.join(file_types)} files",
                    details={"source": "input_dir", "path": input_pdf_dir, "count": total_files}
                )
            else:
                return CheckResult(
                    name="Input Data",
                    status=CheckStatus.FAIL,
                    message="No PDF or HTML files in input directory",
                    details={"path": input_pdf_dir}
                )
        
        return CheckResult(
            name="Input Data",
            status=CheckStatus.FAIL,
            message="No input data configured",
            details={}
        )
    except Exception as e:
        return CheckResult(
            name="Input Data",
            status=CheckStatus.FAIL,
            message=f"Input data check failed: {e}",
            details={"error": str(e)}
        )


# ============================================================================
# MAIN PREFLIGHT CHECK RUNNER
# ============================================================================

def run_preflight_checks(
    skip_expensive: bool = False,
    quiet: bool = False,
    input_dir: Optional[str] = None
) -> Tuple[bool, List[CheckResult]]:
    """
    Run all preflight checks.
    
    Args:
        skip_expensive: Skip model loading checks (LLM, VLM, embedding, reranker)
        quiet: Suppress output
        input_dir: Override input directory for check (uses config.yaml if None)
    
    Returns:
        Tuple of (all_passed: bool, results: List[CheckResult])
    """
    global _input_dir_override
    _input_dir_override = input_dir
    
    results = []
    
    # Define checks in order of importance
    checks = [
        ("config", check_config, False),
        ("api_key", check_api_key, False),
        ("input_data", check_input_data, False),
        ("output_dir", check_output_directory, False),
        ("gpu", check_gpu_availability, False),
        ("llm", check_llm_call, True),
        ("vlm", check_vlm_call, True),
        ("embedding", check_embedding_model, True),
        ("reranker", check_reranker, True),
        ("metrics_emb", check_metrics_embeddings, True),
    ]
    
    if not quiet:
        print("\n" + "=" * 70)
        print("🔍 PREFLIGHT CHECKS - Validating all services before execution")
        print("=" * 70 + "\n")
    
    for name, check_func, is_expensive in checks:
        if is_expensive and skip_expensive:
            result = CheckResult(
                name=name.upper(),
                status=CheckStatus.SKIP,
                message="Skipped (--skip-expensive)"
            )
        else:
            if not quiet:
                print(f"  Checking {name}...", end=" ", flush=True)
            result = check_func()
            if not quiet:
                print(f"{result.status.value} ({result.duration_ms:.0f}ms)")
        
        results.append(result)
    
    # Count results
    passed = sum(1 for r in results if r.status == CheckStatus.PASS)
    failed = sum(1 for r in results if r.status == CheckStatus.FAIL)
    warned = sum(1 for r in results if r.status == CheckStatus.WARN)
    skipped = sum(1 for r in results if r.status == CheckStatus.SKIP)
    
    all_passed = failed == 0
    
    if not quiet:
        print("\n" + "=" * 70)
        print("📋 PREFLIGHT CHECK SUMMARY")
        print("=" * 70)
        print(f"\n  Results: {passed} passed, {failed} failed, {warned} warnings, {skipped} skipped\n")
        
        # Print detailed report
        print("  " + "-" * 66)
        print(f"  {'Service':<25} {'Status':<12} {'Details':<30}")
        print("  " + "-" * 66)
        
        for result in results:
            status_str = result.status.value.split()[0]  # Just the emoji
            details_str = ""
            
            if result.details:
                if 'model' in result.details:
                    details_str = result.details['model']
                elif 'backend' in result.details:
                    details_str = result.details['backend']
                elif 'path' in result.details:
                    details_str = os.path.basename(str(result.details['path']))
                elif 'count' in result.details:
                    details_str = f"{result.details['count']} items"
            
            print(f"  {result.name:<25} {status_str:<12} {details_str:<30}")
        
        print("  " + "-" * 66)
        
        # Configuration summary
        config_result = next((r for r in results if r.name == "Configuration"), None)
        if config_result and config_result.status == CheckStatus.PASS:
            d = config_result.details
            print(f"\n  📌 Active Configuration:")
            print(f"     • Backend:    {d.get('backend', 'N/A')}")
            print(f"     • LLM Model:  {d.get('llm_model', 'N/A')}")
            print(f"     • VLM Model:  {d.get('vlm_model', 'N/A')}")
            print(f"     • Embeddings: {d.get('embedding_model', 'N/A')}")
            print(f"     • Output Dir: {d.get('output_dir', 'N/A')}")
        
        print("\n" + "=" * 70)
        
        if all_passed:
            print("✅ ALL CHECKS PASSED - Ready to start pipeline execution")
        else:
            print("❌ PREFLIGHT CHECKS FAILED - Fix issues before proceeding")
            print("\n   Failed checks:")
            for result in results:
                if result.status == CheckStatus.FAIL:
                    print(f"   • {result.name}: {result.message}")
        
        print("=" * 70 + "\n")
    
    return all_passed, results


def require_preflight_checks() -> bool:
    """Run preflight checks and exit if any fail. Returns True if all pass."""
    all_passed, results = run_preflight_checks()
    
    if not all_passed:
        print("\n🛑 STOPPING: Preflight checks failed. Fix the issues above before running the pipeline.")
        print("   This prevents wasted LLM API calls on a misconfigured system.\n")
        sys.exit(1)
    
    return True


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Main entry point for preflight checks CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run preflight checks for QA pipeline")
    parser.add_argument("--skip-expensive", action="store_true", 
                       help="Skip expensive checks (model loading, API calls)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress output")
    
    args = parser.parse_args()
    
    all_passed, _ = run_preflight_checks(
        skip_expensive=args.skip_expensive,
        quiet=args.quiet
    )
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

