#!/usr/bin/env python3
"""
MiRAGE Demo - FastAPI Backend Service

This service processes uploaded documents and generates QA pairs using the MiRAGE pipeline.
It is designed to be hosted on a separate server and receive requests from the Gradio UI.

Usage:
    uvicorn fastapi_service:app --host 0.0.0.0 --port 8000

Environment Variables:
    MIRAGE_DEMO_MAX_PAGES: Maximum pages allowed (default: 20)
    MIRAGE_DEMO_MAX_QA_PAIRS: Maximum QA pairs to generate (default: 50)
    MIRAGE_DEMO_DEVICE: Device for embeddings (default: cpu, set to cuda for GPU)
"""

import os
import sys

# ============================================================================
# IMPORTANT: Set device config BEFORE importing torch
# ============================================================================
# Use CUDA/GPU when available for better performance and memory efficiency
# Fall back to CPU only if explicitly requested or GPU unavailable
DEMO_DEVICE = os.environ.get("MIRAGE_DEMO_DEVICE", "auto")  # auto = use GPU if available
if DEMO_DEVICE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
# If "auto" or "cuda", let PyTorch detect available GPUs

# Use API-based Gemini embeddings for demo (reduces local memory usage)
os.environ["MIRAGE_EMBEDDING_MODEL"] = os.environ.get("MIRAGE_EMBEDDING_MODEL", "gemini")

# Set HuggingFace cache to persist downloaded models
HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(HF_CACHE_DIR, "sentence_transformers")

import json
import shutil
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Setup device environment (auto-detect GPU/CPU, suppress pin_memory warnings)
from mirage.utils.device import setup_device_environment
setup_device_environment()

# ============================================================================
# Configuration - Demo Limits (to control resource consumption)
# ============================================================================
DEMO_MAX_PAGES = int(os.environ.get("MIRAGE_DEMO_MAX_PAGES", 20))
DEMO_MAX_QA_PAIRS = int(os.environ.get("MIRAGE_DEMO_MAX_QA_PAIRS", 50))
DEMO_MAX_DEPTH = 2  # Fixed for demo - prevents excessive API calls
DEMO_MAX_BREADTH = 5  # Fixed for demo
DEMO_MAX_WORKERS = 1  # Single worker to reduce memory usage
DEMO_MAX_FILE_SIZE_MB = 50  # Maximum file size in MB
DEMO_TEMP_DIR = Path(tempfile.gettempdir()) / "mirage_demo"

# Ensure temp directory exists
DEMO_TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FastAPI App
# ============================================================================
app = FastAPI(
    title="MiRAGE Demo API",
    description="Backend API for MiRAGE QA Dataset Generation Demo",
    version="1.3.0"
)

# Enable CORS for Gradio frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your Gradio Space URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Models
# ============================================================================
class ProcessingRequest(BaseModel):
    """Request model for processing configuration."""
    api_key: str
    backend: str = "gemini"  # gemini, openai, ollama
    model_name: Optional[str] = None
    num_qa_pairs: int = 50

class ProcessingResponse(BaseModel):
    """Response model for processing results."""
    success: bool
    message: str
    job_id: Optional[str] = None
    qa_pairs: Optional[List[Dict[str, Any]]] = None
    stats: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class JobStatus(BaseModel):
    """Job status model."""
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: Optional[float] = None
    message: Optional[str] = None
    result: Optional[ProcessingResponse] = None

# In-memory job storage (use Redis/DB in production)
_jobs: Dict[str, JobStatus] = {}

# ============================================================================
# Utility Functions
# ============================================================================
def count_pdf_pages(file_path: Path) -> int:
    """Count the number of pages in a PDF file."""
    try:
        import PyPDF2
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return len(reader.pages)
    except ImportError:
        # Fallback: estimate based on file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        return int(file_size_mb * 10)  # Rough estimate: 10 pages per MB
    except Exception:
        return 0

def validate_files(files: List[UploadFile]) -> tuple[bool, str]:
    """Validate uploaded files."""
    total_size = 0
    for file in files:
        # Check file extension
        ext = Path(file.filename).suffix.lower()
        if ext not in ['.pdf', '.html', '.htm', '.md', '.txt']:
            return False, f"Unsupported file type: {ext}. Allowed: PDF, HTML, MD, TXT"
        
        # Check file size
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)
        total_size += size
        
        if size > DEMO_MAX_FILE_SIZE_MB * 1024 * 1024:
            return False, f"File {file.filename} exceeds {DEMO_MAX_FILE_SIZE_MB}MB limit"
    
    return True, "Files validated successfully"

def cleanup_job_directory(job_dir: Path):
    """Clean up temporary job directory."""
    try:
        if job_dir.exists():
            shutil.rmtree(job_dir)
    except Exception as e:
        print(f"Warning: Failed to cleanup {job_dir}: {e}")

# ============================================================================
# Processing Logic
# ============================================================================
def process_documents(
    job_id: str,
    input_dir: Path,
    output_dir: Path,
    api_key: str,
    backend: str,
    model_name: Optional[str],
    num_qa_pairs: int
) -> ProcessingResponse:
    """
    Process documents using the MiRAGE pipeline.
    
    This function runs the complete pipeline:
    1. PDF/HTML to Markdown conversion
    2. Semantic chunking
    3. Embedding and indexing
    4. Multi-hop QA generation
    
    Args:
        job_id: Unique job identifier
        input_dir: Directory containing uploaded documents
        output_dir: Directory for output files
        api_key: API key for LLM backend
        backend: Backend to use (gemini, openai, ollama)
        model_name: Optional model name override
        num_qa_pairs: Number of QA pairs to generate
    
    Returns:
        ProcessingResponse with results or error
    """
    original_cwd = os.getcwd()  # Save original directory for restoration
    
    try:
        # Update job status
        _jobs[job_id].status = "processing"
        _jobs[job_id].message = "Initializing pipeline..."
        
        # Set environment variables for MiRAGE
        os.environ["MIRAGE_INPUT_DIR"] = str(input_dir)
        os.environ["MIRAGE_OUTPUT_DIR"] = str(output_dir)
        os.environ["MIRAGE_NUM_QA_PAIRS"] = str(min(num_qa_pairs, DEMO_MAX_QA_PAIRS))
        os.environ["MIRAGE_MAX_WORKERS"] = str(DEMO_MAX_WORKERS)
        os.environ["MIRAGE_MAX_DEPTH"] = str(DEMO_MAX_DEPTH)
        
        # Set API key based on backend
        if backend.lower() == "gemini":
            # Set both GEMINI_API_KEY and GOOGLE_API_KEY (different Google libs use different vars)
            os.environ["GEMINI_API_KEY"] = api_key
            os.environ["GOOGLE_API_KEY"] = api_key
            os.environ["MIRAGE_BACKEND"] = "gemini"
        elif backend.lower() == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["MIRAGE_BACKEND"] = "openai"
        elif backend.lower() == "ollama":
            os.environ["MIRAGE_BACKEND"] = "ollama"
        
        if model_name:
            os.environ["MIRAGE_MODEL"] = model_name
        
        _jobs[job_id].message = "Loading MiRAGE pipeline..."
        _jobs[job_id].progress = 0.1
        
        # Change to project root directory (where config.yaml is located)
        # This is required because MiRAGE modules load config.yaml at import time
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)
        
        # Import and run the pipeline
        from mirage.main import run_pipeline
        from mirage.core.llm import get_token_stats
        
        _jobs[job_id].message = "Processing documents..."
        _jobs[job_id].progress = 0.2
        
        # Run the pipeline (this may take a while)
        run_pipeline()
        
        _jobs[job_id].progress = 0.9
        _jobs[job_id].message = "Collecting results..."
        
        # Read the generated QA pairs
        qa_file = output_dir / "qa_multihop_pass.json"
        qa_pairs = []
        
        if qa_file.exists():
            with open(qa_file, 'r') as f:
                qa_pairs = json.load(f)
        
        # Get token stats
        token_stats = get_token_stats()
        
        # Build response
        stats = {
            "total_qa_pairs": len(qa_pairs),
            "input_tokens": token_stats.get("total_input_tokens", 0),
            "output_tokens": token_stats.get("total_output_tokens", 0),
            "processing_time": datetime.now().isoformat()
        }
        
        _jobs[job_id].progress = 1.0
        _jobs[job_id].status = "completed"
        _jobs[job_id].message = f"Generated {len(qa_pairs)} QA pairs"
        
        response = ProcessingResponse(
            success=True,
            message=f"Successfully generated {len(qa_pairs)} QA pairs",
            job_id=job_id,
            qa_pairs=qa_pairs,
            stats=stats
        )
        
        _jobs[job_id].result = response
        
        # Restore original working directory
        os.chdir(original_cwd)
        
        return response
        
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        
        _jobs[job_id].status = "failed"
        _jobs[job_id].message = str(e)
        
        response = ProcessingResponse(
            success=False,
            message="Processing failed",
            job_id=job_id,
            error=str(e)
        )
        
        _jobs[job_id].result = response
        
        # Restore original working directory
        try:
            os.chdir(original_cwd)
        except:
            pass
        
        return response

def run_processing_job(
    job_id: str,
    input_dir: Path,
    output_dir: Path,
    api_key: str,
    backend: str,
    model_name: Optional[str],
    num_qa_pairs: int
):
    """Background task to run document processing."""
    try:
        process_documents(
            job_id=job_id,
            input_dir=input_dir,
            output_dir=output_dir,
            api_key=api_key,
            backend=backend,
            model_name=model_name,
            num_qa_pairs=num_qa_pairs
        )
    finally:
        # Schedule cleanup after processing (keep files for a while for debugging)
        pass

# ============================================================================
# API Endpoints
# ============================================================================
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "MiRAGE Demo API",
        "version": "1.3.0",
        "status": "running",
        "limits": {
            "max_pages": DEMO_MAX_PAGES,
            "max_qa_pairs": DEMO_MAX_QA_PAIRS,
            "max_file_size_mb": DEMO_MAX_FILE_SIZE_MB,
            "max_depth": DEMO_MAX_DEPTH,
            "max_breadth": DEMO_MAX_BREADTH
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len([j for j in _jobs.values() if j.status == "processing"])
    }

@app.post("/process", response_model=ProcessingResponse)
async def process_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="PDF or document files to process"),
    api_key: str = Form(..., description="API key for LLM backend"),
    backend: str = Form(default="gemini", description="Backend: gemini, openai, ollama"),
    model_name: Optional[str] = Form(default=None, description="Optional model name"),
    num_qa_pairs: int = Form(default=50, description="Number of QA pairs to generate (max 50)")
):
    """
    Process uploaded documents and generate QA pairs.
    
    This endpoint:
    1. Validates uploaded files (type, size, page count)
    2. Creates a processing job
    3. Runs the MiRAGE pipeline in the background
    4. Returns a job ID for status tracking
    
    Limits:
    - Maximum 20 pages total
    - Maximum 50 QA pairs
    - Maximum 50MB per file
    """
    # Validate files
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    valid, message = validate_files(files)
    if not valid:
        raise HTTPException(status_code=400, detail=message)
    
    # Create job directory
    job_id = str(uuid.uuid4())[:8]
    job_dir = DEMO_TEMP_DIR / job_id
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded files and count pages
    total_pages = 0
    saved_files = []
    
    for file in files:
        file_path = input_dir / file.filename
        
        try:
            contents = await file.read()
            with open(file_path, 'wb') as f:
                f.write(contents)
            saved_files.append(file_path)
            
            # Count pages for PDFs
            if file_path.suffix.lower() == '.pdf':
                pages = count_pdf_pages(file_path)
                total_pages += pages
                
                if total_pages > DEMO_MAX_PAGES:
                    cleanup_job_directory(job_dir)
                    raise HTTPException(
                        status_code=400,
                        detail=f"Total pages ({total_pages}) exceeds limit ({DEMO_MAX_PAGES}). "
                               f"Please upload fewer or smaller documents."
                    )
        except HTTPException:
            raise
        except Exception as e:
            cleanup_job_directory(job_dir)
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Validate backend and API key
    if backend.lower() not in ["gemini", "openai", "ollama"]:
        cleanup_job_directory(job_dir)
        raise HTTPException(status_code=400, detail=f"Invalid backend: {backend}")
    
    if backend.lower() != "ollama" and not api_key:
        cleanup_job_directory(job_dir)
        raise HTTPException(status_code=400, detail=f"API key required for {backend} backend")
    
    # Enforce QA pairs limit
    num_qa_pairs = min(num_qa_pairs, DEMO_MAX_QA_PAIRS)
    
    # Initialize job
    _jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        progress=0.0,
        message="Job queued for processing"
    )
    
    # Start background processing
    background_tasks.add_task(
        run_processing_job,
        job_id=job_id,
        input_dir=input_dir,
        output_dir=output_dir,
        api_key=api_key,
        backend=backend,
        model_name=model_name,
        num_qa_pairs=num_qa_pairs
    )
    
    return ProcessingResponse(
        success=True,
        message=f"Processing job started. Total pages: {total_pages}",
        job_id=job_id,
        stats={
            "files_uploaded": len(saved_files),
            "total_pages": total_pages,
            "target_qa_pairs": num_qa_pairs
        }
    )

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return _jobs[job_id]

@app.get("/result/{job_id}", response_model=ProcessingResponse)
async def get_job_result(job_id: str):
    """Get the result of a completed processing job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = _jobs[job_id]
    
    if job.status == "processing":
        return ProcessingResponse(
            success=False,
            message="Job is still processing",
            job_id=job_id,
            stats={"progress": job.progress}
        )
    
    if job.status == "pending":
        return ProcessingResponse(
            success=False,
            message="Job is pending",
            job_id=job_id
        )
    
    if job.result:
        return job.result
    
    return ProcessingResponse(
        success=False,
        message=f"Job status: {job.status}",
        job_id=job_id,
        error=job.message
    )

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job_dir = DEMO_TEMP_DIR / job_id
    cleanup_job_directory(job_dir)
    
    del _jobs[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}

# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"Starting MiRAGE Demo API on {host}:{port}")
    print(f"Limits: max_pages={DEMO_MAX_PAGES}, max_qa_pairs={DEMO_MAX_QA_PAIRS}")
    
    uvicorn.run(app, host=host, port=port)
