#!/usr/bin/env python3
"""
MiRAGE Demo - Gradio UI for Hugging Face Spaces

This Gradio application provides an interactive interface for the MiRAGE
QA dataset generation pipeline. It connects to a FastAPI backend service
for document processing.

Usage (local):
    python gradio_app.py

For Hugging Face Spaces:
    Set the FASTAPI_URL environment variable to point to your backend server.

Environment Variables:
    FASTAPI_URL: URL of the FastAPI backend (default: http://localhost:8000)
"""

import os
import json
import time
import requests
from pathlib import Path
from typing import Optional, List, Tuple
import gradio as gr

# ============================================================================
# Configuration
# ============================================================================
FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://localhost:8000")
MAX_PAGES = 20
MAX_QA_PAIRS = 50
POLL_INTERVAL = 2  # seconds between status checks

# ============================================================================
# Backend Communication
# ============================================================================
def check_backend_health() -> Tuple[bool, str]:
    """Check if the FastAPI backend is available."""
    try:
        response = requests.get(f"{FASTAPI_URL}/health", timeout=10)
        if response.status_code == 200:
            return True, "Backend is healthy"
        return False, f"Backend returned status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to backend at {FASTAPI_URL}"
    except Exception as e:
        return False, f"Backend check failed: {str(e)}"

def submit_processing_job(
    files: List[str],
    api_key: str,
    backend: str,
    model_name: Optional[str],
    num_qa_pairs: int
) -> Tuple[bool, str, Optional[str]]:
    """
    Submit documents for processing.
    
    Returns:
        Tuple of (success, message, job_id)
    """
    try:
        # Prepare files for upload
        file_objects = []
        for file_path in files:
            if file_path:
                file_objects.append(
                    ('files', (Path(file_path).name, open(file_path, 'rb')))
                )
        
        if not file_objects:
            return False, "No files selected", None
        
        # Prepare form data
        data = {
            'api_key': api_key or "",
            'backend': backend,
            'num_qa_pairs': min(num_qa_pairs, MAX_QA_PAIRS)
        }
        
        if model_name:
            data['model_name'] = model_name
        
        # Submit request
        response = requests.post(
            f"{FASTAPI_URL}/process",
            files=file_objects,
            data=data,
            timeout=60
        )
        
        # Close file handles
        for _, (_, f) in file_objects:
            f.close()
        
        if response.status_code == 200:
            result = response.json()
            return True, result.get('message', 'Job submitted'), result.get('job_id')
        else:
            error = response.json().get('detail', response.text)
            return False, f"Submission failed: {error}", None
            
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to backend at {FASTAPI_URL}", None
    except Exception as e:
        return False, f"Submission error: {str(e)}", None

def poll_job_status(job_id: str) -> Tuple[str, float, Optional[dict]]:
    """
    Poll the job status.
    
    Returns:
        Tuple of (status, progress, result)
    """
    try:
        response = requests.get(f"{FASTAPI_URL}/status/{job_id}", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return (
                data.get('status', 'unknown'),
                data.get('progress', 0) or 0,
                data.get('result')
            )
        else:
            return 'error', 0, None
            
    except Exception as e:
        return 'error', 0, {'error': str(e)}

def get_job_result(job_id: str) -> Tuple[bool, str, Optional[List[dict]], Optional[dict]]:
    """
    Get the final result of a job.
    
    Returns:
        Tuple of (success, message, qa_pairs, stats)
    """
    try:
        response = requests.get(f"{FASTAPI_URL}/result/{job_id}", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return (
                data.get('success', False),
                data.get('message', ''),
                data.get('qa_pairs'),
                data.get('stats')
            )
        else:
            return False, f"Failed to get result: {response.text}", None, None
            
    except Exception as e:
        return False, f"Error getting result: {str(e)}", None, None

# ============================================================================
# Processing Functions
# ============================================================================
def process_documents(
    files,
    api_key: str,
    backend: str,
    model_name: str,
    num_qa_pairs: int,
    progress=gr.Progress()
) -> Tuple[str, str, str]:
    """
    Main processing function called by Gradio.
    
    Args:
        files: Uploaded files
        api_key: API key for the selected backend
        backend: Backend to use (gemini, openai, ollama)
        model_name: Optional model name
        num_qa_pairs: Number of QA pairs to generate
        progress: Gradio progress tracker
    
    Returns:
        Tuple of (status_message, qa_pairs_json, stats_json)
    """
    # Validate inputs
    if not files:
        return "Error: No files uploaded", "", ""
    
    if backend != "ollama" and not api_key:
        return f"Error: API key is required for {backend} backend", "", ""
    
    # Check backend health
    healthy, health_msg = check_backend_health()
    if not healthy:
        return f"Error: {health_msg}", "", ""
    
    progress(0.1, desc="Submitting documents...")
    
    # Submit job
    file_paths = [f.name if hasattr(f, 'name') else f for f in files]
    success, message, job_id = submit_processing_job(
        files=file_paths,
        api_key=api_key,
        backend=backend,
        model_name=model_name if model_name else None,
        num_qa_pairs=num_qa_pairs
    )
    
    if not success:
        return f"Error: {message}", "", ""
    
    progress(0.2, desc=f"Job {job_id} submitted. Processing...")
    
    # Poll for completion
    max_polls = 300  # 10 minutes with 2-second intervals
    poll_count = 0
    
    while poll_count < max_polls:
        status, job_progress, result = poll_job_status(job_id)
        
        # Update progress bar
        overall_progress = 0.2 + (job_progress * 0.7)  # 20% to 90%
        progress(overall_progress, desc=f"Processing... ({int(job_progress * 100)}%)")
        
        if status == "completed":
            break
        elif status == "failed":
            error_msg = result.get('error', 'Unknown error') if result else 'Processing failed'
            return f"Error: {error_msg}", "", ""
        elif status == "error":
            return "Error: Failed to check job status", "", ""
        
        time.sleep(POLL_INTERVAL)
        poll_count += 1
    
    if poll_count >= max_polls:
        return "Error: Processing timed out", "", ""
    
    progress(0.95, desc="Fetching results...")
    
    # Get final result
    success, message, qa_pairs, stats = get_job_result(job_id)
    
    if not success:
        return f"Error: {message}", "", ""
    
    progress(1.0, desc="Done!")
    
    # Format output
    qa_json = json.dumps(qa_pairs, indent=2) if qa_pairs else "[]"
    stats_json = json.dumps(stats, indent=2) if stats else "{}"
    
    return (
        f"Success! Generated {len(qa_pairs) if qa_pairs else 0} QA pairs",
        qa_json,
        stats_json
    )

def format_qa_display(qa_json: str) -> str:
    """Format QA pairs for display in a readable format."""
    if not qa_json or qa_json == "[]":
        return "No QA pairs generated yet."
    
    try:
        qa_pairs = json.loads(qa_json)
        
        if not qa_pairs:
            return "No QA pairs generated."
        
        display_parts = []
        for i, qa in enumerate(qa_pairs[:10], 1):  # Show first 10
            question = qa.get('question', 'N/A')
            answer = qa.get('answer', 'N/A')
            hop_count = qa.get('hop_count', 'N/A')
            domain = qa.get('domain', 'N/A')
            
            display_parts.append(f"""
### QA Pair {i}
**Question:** {question}

**Answer:** {answer}

**Hops:** {hop_count} | **Domain:** {domain}

---
""")
        
        if len(qa_pairs) > 10:
            display_parts.append(f"\n*... and {len(qa_pairs) - 10} more QA pairs (see JSON tab for full data)*")
        
        return "".join(display_parts)
        
    except json.JSONDecodeError:
        return "Error parsing QA data"

# ============================================================================
# Gradio Interface
# ============================================================================
def create_demo():
    """Create the Gradio demo interface."""
    
    with gr.Blocks(
        title="MiRAGE Demo"
    ) as demo:
        
        # Header
        gr.Markdown("""
        # MiRAGE: Multimodal Multihop QA Dataset Generator
        
        Generate high-quality question-answer pairs from your documents using the MiRAGE framework.
        
        **How it works:**
        1. Upload your PDF documents (max 20 pages total)
        2. Enter your API key and select a backend
        3. Click "Generate QA Pairs" to start processing
        4. Download the generated QA dataset
        """, elem_classes="header")
        
        # Limits info
        gr.Markdown(f"""
        **Demo Limits:** Max {MAX_PAGES} pages | Max {MAX_QA_PAIRS} QA pairs | Supported: PDF, HTML, MD, TXT
        """, elem_classes="limits-info")
        
        with gr.Row():
            # Left column - Inputs
            with gr.Column(scale=1):
                # File upload
                file_input = gr.File(
                    label="Upload Documents",
                    file_count="multiple",
                    file_types=[".pdf", ".html", ".htm", ".md", ".txt"],
                    type="filepath"
                )
                
                # Backend selection
                backend_dropdown = gr.Dropdown(
                    choices=["gemini", "openai", "ollama"],
                    value="gemini",
                    label="LLM Backend",
                    info="Select your LLM provider"
                )
                
                # API Key
                api_key_input = gr.Textbox(
                    label="API Key",
                    placeholder="Enter your API key (not needed for Ollama)",
                    type="password",
                    info="Your API key is sent securely to the backend and not stored"
                )
                
                # Model selection
                model_input = gr.Textbox(
                    label="Model Name (Optional)",
                    placeholder="e.g., gemini-2.0-flash, gpt-4o",
                    info="Leave empty for default model"
                )
                
                # Number of QA pairs
                num_qa_slider = gr.Slider(
                    minimum=5,
                    maximum=MAX_QA_PAIRS,
                    value=50,
                    step=5,
                    label="Number of QA Pairs",
                    info=f"Maximum {MAX_QA_PAIRS} for demo"
                )
                
                # Submit button
                submit_btn = gr.Button(
                    "Generate QA Pairs",
                    variant="primary",
                    size="lg"
                )
            
            # Right column - Outputs
            with gr.Column(scale=2):
                # Status
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2
                )
                
                # Results tabs
                with gr.Tabs():
                    with gr.TabItem("QA Pairs (Preview)"):
                        qa_display = gr.Markdown(
                            value="Upload documents and click 'Generate QA Pairs' to see results.",
                            label="Generated QA Pairs"
                        )
                    
                    with gr.TabItem("QA Pairs (JSON)"):
                        qa_json_output = gr.Code(
                            language="json",
                            label="QA Pairs JSON",
                            lines=20
                        )
                    
                    with gr.TabItem("Statistics"):
                        stats_output = gr.Code(
                            language="json",
                            label="Processing Statistics",
                            lines=10
                        )
                
                # Download button
                download_btn = gr.Button("Download QA Dataset", size="sm")
        
        # Backend info
        with gr.Accordion("Backend Setup Instructions", open=False):
            gr.Markdown("""
            ### Google Gemini (Recommended)
            1. Get API key from: https://makersuite.google.com/app/apikey
            2. Select "gemini" backend and paste your key
            
            ### OpenAI
            1. Get API key from: https://platform.openai.com/api-keys
            2. Select "openai" backend and paste your key
            
            ### Ollama (Local - Free)
            1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
            2. Start server: `ollama serve`
            3. Pull models: `ollama pull llama3 && ollama pull llava`
            4. Select "ollama" backend (no API key needed)
            
            **Note:** For Ollama, the backend server must have access to your local Ollama instance.
            """)
        
        # Event handlers
        def on_submit(files, api_key, backend, model_name, num_qa_pairs):
            status, qa_json, stats_json = process_documents(
                files, api_key, backend, model_name, num_qa_pairs
            )
            qa_preview = format_qa_display(qa_json)
            return status, qa_preview, qa_json, stats_json
        
        submit_btn.click(
            fn=on_submit,
            inputs=[file_input, api_key_input, backend_dropdown, model_input, num_qa_slider],
            outputs=[status_output, qa_display, qa_json_output, stats_output]
        )
        
        def download_qa(qa_json):
            if not qa_json or qa_json == "[]":
                return None
            
            # Create temporary file for download
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(qa_json)
                return f.name
        
        download_btn.click(
            fn=download_qa,
            inputs=[qa_json_output],
            outputs=[gr.File(label="Download")]
        )
        
        # Footer
        gr.Markdown("""
        ---
        **MiRAGE** - A Multiagent Framework for Generating Multimodal Multihop Question-Answer Dataset for RAG Evaluation
        
        [GitHub](https://github.com/ChandanKSahu/MiRAGE) | [Paper](https://arxiv.org/abs/2601.15487) | [PyPI](https://pypi.org/project/mirage-benchmark/)
        """)
    
    return demo

# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    import sys
    print("=" * 50)
    print("MiRAGE Demo - Gradio UI")
    print("=" * 50)
    print(f"Backend URL: {FASTAPI_URL}")
    print(f"Max Pages: {MAX_PAGES}")
    print(f"Max QA Pairs: {MAX_QA_PAIRS}")
    print("=" * 50)
    sys.stdout.flush()
    
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
