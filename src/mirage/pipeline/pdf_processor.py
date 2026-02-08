"""
Docling-Based Document Processing and Table Image Extraction

Supports: single PDF/HTML file, folder of documents, or zip file containing documents.
Supported formats: PDF, HTML, XHTML (via Docling library)
Configuration via config.yaml under pdf_processing section.
"""

import logging
import time
from pathlib import Path
import requests
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
import zipfile
import tempfile
import shutil
import yaml

import pandas as pd
import torch

from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling_core.types.doc.document import DescriptionAnnotation
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorDevice,
    AcceleratorOptions,
    PictureDescriptionApiOptions,
    EasyOcrOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption, HTMLFormatOption
from docling.utils.export import generate_multimodal_pages


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file. Returns defaults if file not found."""
    from pathlib import Path as _Path
    path = _Path(config_path)
    if not path.exists():
        # Try the core config loader which has fallback defaults
        try:
            from mirage.core.config import load_config as _core_load_config
            return _core_load_config()
        except Exception:
            return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}

# Load config (gracefully falls back to defaults if config.yaml is not present)
CONFIG = load_config()
PDF_CONFIG = CONFIG.get("pdf_processing", {})
BACKEND_CONFIG = CONFIG.get("backend", {})

# --- Configuration from config.yaml ---
IMAGE_RESOLUTION_SCALE = PDF_CONFIG.get("image_resolution_scale", 2.0)
INPUT_PATH = PDF_CONFIG.get("input_path", "data/documents")
OUTPUT_DIR = Path(PDF_CONFIG.get("output_dir", "trials/pdf2md/output"))
MODEL_NAME = PDF_CONFIG.get("model_name", "qwen2.5vl:32b")
NUM_THREADS = PDF_CONFIG.get("num_threads", 14)
CUDA_DEVICE_ID = PDF_CONFIG.get("cuda_device_id", 1)

# API Configuration from backend settings
# API configuration from backend config or environment
API_KEY_FILE = os.environ.get("GEMINI_API_KEY_PATH", os.path.expanduser("~/.config/gemini/api_key.txt"))
API_URL = os.environ.get("LLM_API_URL", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent")

def get_api_key():
    """Get API key from environment variable or file."""
    # First check environment variable (set by run_mirage.py --api-key)
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        return api_key
    # Fall back to file
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, 'r') as f:
            return f.read().strip()
    raise FileNotFoundError(f"API key not found. Set GEMINI_API_KEY env var or create {API_KEY_FILE}")

from mirage.core.prompts import PROMPTS_DESC

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


# Supported document extensions
SUPPORTED_EXTENSIONS = {'.pdf', '.html', '.htm', '.xhtml'}

def get_input_format(file_path: Path) -> InputFormat:
    """Get the InputFormat enum for a given file path."""
    ext = file_path.suffix.lower()
    if ext == '.pdf':
        return InputFormat.PDF
    elif ext in {'.html', '.htm', '.xhtml'}:
        return InputFormat.HTML
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def collect_input_files(input_path):
    """
    Collect document files (PDF, HTML) from input path.
    
    Args:
        input_path: str - path to a single document, folder of documents, or zip file
        
    Returns:
        tuple: (list of document paths, temp_dir or None)
               temp_dir is returned if zip was extracted (caller should clean up)
    """
    input_path = Path(input_path)
    temp_dir = None
    doc_files = []
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            # Single document file
            doc_files = [input_path]
        elif input_path.suffix.lower() == '.zip':
            # Zip file - extract to temp directory
            temp_dir = tempfile.mkdtemp(prefix="doc_extract_")
            _log.info(f"Extracting zip file to: {temp_dir}")
            with zipfile.ZipFile(input_path, 'r') as zf:
                zf.extractall(temp_dir)
            # Recursively find all supported documents in extracted content
            for ext in SUPPORTED_EXTENSIONS:
                doc_files.extend(Path(temp_dir).rglob(f"*{ext}"))
        else:
            raise ValueError(f"Unsupported file type: {input_path.suffix}. Supported: {SUPPORTED_EXTENSIONS}")
    elif input_path.is_dir():
        # Folder - recursively find all supported documents
        for ext in SUPPORTED_EXTENSIONS:
            doc_files.extend(input_path.rglob(f"*{ext}"))
    else:
        raise ValueError(f"Invalid input path: {input_path}")
    
    # Sort by file size (smallest first) for faster initial feedback
    doc_files = sorted(doc_files, key=lambda p: p.stat().st_size)
    
    # Log counts by format
    pdf_count = sum(1 for f in doc_files if f.suffix.lower() == '.pdf')
    html_count = sum(1 for f in doc_files if f.suffix.lower() in {'.html', '.htm', '.xhtml'})
    _log.info(f"Found {len(doc_files)} document files to process (PDF: {pdf_count}, HTML: {html_count})")
    return doc_files, temp_dir

# Backward compatibility alias
def collect_pdf_files(input_path):
    """Backward compatibility wrapper for collect_input_files."""
    return collect_input_files(input_path)

def motormaven_vlm_options():
    """Configure PictureDescriptionApiOptions for motormaven endpoint"""
    # Load API key from env var or file
    api_key = get_api_key()
    
    options = PictureDescriptionApiOptions(
        url=API_URL,
        params=dict(
            model=MODEL_NAME,
        ),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        prompt=PROMPTS_DESC["image"],
        timeout=120,
        retries=10,
    )
    return options

def check_cuda_memory(device_id=0):
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(device_id).total_memory
        reserved_memory = torch.cuda.memory_reserved(device_id)
        allocated_memory = torch.cuda.memory_allocated(device_id)
        free_memory = reserved_memory - allocated_memory
        unreserved_memory = total_memory - reserved_memory

        print(f"Total memory:     {total_memory / 1024 ** 3:.2f} GiB")
        print(f"Reserved memory:  {reserved_memory / 1024 ** 3:.2f} GiB")
        print(f"Allocated memory: {allocated_memory / 1024 ** 3:.2f} GiB")
        print(f"Free memory:      {free_memory / 1024 ** 3:.2f} GiB (within reserved)")
        print(f"Unreserved memory: {unreserved_memory / 1024 ** 3:.2f} GiB")
    else:
        print("CUDA is not available.")


def is_bbox_inside(inner_bbox, outer_bbox):
    """Check if inner_bbox is inside outer_bbox. Both are BoundingBox objects with l, t, r, b.
    Coordinate origin is bottom-left: b < t (bottom has lower y than top)."""
    return (inner_bbox.l >= outer_bbox.l and 
            inner_bbox.r <= outer_bbox.r and 
            inner_bbox.b >= outer_bbox.b and 
            inner_bbox.t <= outer_bbox.t)


def get_pictures_inside_tables(conv_res):
    """Return set of picture indices that are inside tables."""
    pictures_to_skip = set()
    
    # Build table info: {page_no: [bbox1, bbox2, ...]}
    table_bboxes_by_page = {}
    for table in conv_res.document.tables:
        if table.prov:
            for prov in table.prov:
                page_no = prov.page_no
                bbox = prov.bbox
                if page_no not in table_bboxes_by_page:
                    table_bboxes_by_page[page_no] = []
                table_bboxes_by_page[page_no].append(bbox)
    
    # Check each picture
    for i, picture in enumerate(conv_res.document.pictures):
        if picture.prov:
            for prov in picture.prov:
                pic_page = prov.page_no
                pic_bbox = prov.bbox
                # Check if this picture is inside any table on the same page
                if pic_page in table_bboxes_by_page:
                    for table_bbox in table_bboxes_by_page[pic_page]:
                        if is_bbox_inside(pic_bbox, table_bbox):
                            pictures_to_skip.add(i)
                            print(f"Picture {i} on page {pic_page} is inside a table - will skip annotation")
                            break
    
    return pictures_to_skip


def make_api_call_with_retries(payload, api_url, headers, item_type, item_index, output_dir):
    """Make API call with retry logic for server errors and response recording"""
    retry_count = 0
    while True:
        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=120)
            
            if response.status_code == 200:
                # Response saving disabled - no longer saving individual response files
                # responses_dir = output_dir / "api_responses"
                # responses_dir.mkdir(parents=True, exist_ok=True)
                # response_file = responses_dir / f"{item_type}_{item_index}_response.json"
                # with open(response_file, 'w') as f:
                #     json.dump(response.json(), f, indent=2)
                return response
            elif 400 <= response.status_code < 500:
                # Client error - skip
                print(f"Client error for {item_type} {item_index}: HTTP {response.status_code} - skipping")
                return None
            elif response.status_code >= 500:
                # Server error - retry with exponential backoff
                retry_count += 1
                wait_time = min(60, 2 ** min(retry_count, 6))  # Cap at 60 seconds
                print(f"Server error for {item_type} {item_index}: HTTP {response.status_code} - retrying in {wait_time}s (attempt {retry_count})")
                time.sleep(wait_time)
                continue
            else:
                print(f"Unexpected status for {item_type} {item_index}: HTTP {response.status_code} - skipping")
                return None
                
        except Exception as e:
            # Network/connection errors - retry
            retry_count += 1
            wait_time = min(60, 2 ** min(retry_count, 6))
            print(f"Network error for {item_type} {item_index}: {str(e)} - retrying in {wait_time}s (attempt {retry_count})")
            time.sleep(wait_time)
            continue

def annotate_items_with_images(conv_res, model_name=MODEL_NAME, api_url=API_URL, max_tokens=1000, use_batch=True):
    """Annotate both pictures and tables that don't have annotations using their base64 image data.
    
    Args:
        conv_res: Conversion result from docling
        model_name: VLM model name
        api_url: API endpoint URL
        max_tokens: Max tokens for response
        use_batch: If True, use batch processing for faster annotation
        
    Returns:
        Set of picture indices that are inside tables (skipped).
    """
    
    # Get pictures that are inside tables (skip these)
    pictures_to_skip = get_pictures_inside_tables(conv_res)
    
    if use_batch:
        return _annotate_items_batch(conv_res, model_name, pictures_to_skip)
    else:
        return _annotate_items_sequential(conv_res, model_name, api_url, pictures_to_skip)


def _annotate_items_batch(conv_res, model_name, pictures_to_skip):
    """Batch annotate pictures and tables using async batch processing."""
    from mirage.core.llm import batch_call_vlm_base64
    
    # Collect all items that need annotation
    batch_requests = []  # List of (prompt, base64, mime_type)
    item_refs = []  # Track which item each request corresponds to: ('picture'|'table', index, item)
    
    # Collect pictures
    for i, item in enumerate(conv_res.document.pictures):
        if i in pictures_to_skip:
            print(f"Picture {i}: Skipping - inside a table")
            continue
        if not item.annotations:
            try:
                base64_data = str(item.image.uri).split(',')[1] if item.image and item.image.uri else None
                if not base64_data:
                    print(f"Picture {i}: No base64 data available")
                    continue
                print(f"Picture {i}: Queued for batch (base64 len: {len(base64_data)})")
                batch_requests.append((PROMPTS_DESC["image"], base64_data, "image/png"))
                item_refs.append(('picture', i, item))
            except Exception as e:
                print(f"Error preparing picture {i}: {str(e)}")
    
    # Collect tables
    for i, item in enumerate(conv_res.document.tables):
        if not item.annotations:
            try:
                base64_data = str(item.image.uri.path).split(',')[1] if item.image and item.image.uri else None
                if not base64_data:
                    print(f"Table {i}: No base64 data available")
                    continue
                print(f"Table {i}: Queued for batch (base64 len: {len(base64_data)})")
                batch_requests.append((PROMPTS_DESC["table"], base64_data, "image/png"))
                item_refs.append(('table', i, item))
            except Exception as e:
                print(f"Error preparing table {i}: {str(e)}")
    
    if not batch_requests:
        print("No items to annotate")
        return pictures_to_skip
    
    # Execute batch VLM call
    print(f"\nBatch annotating {len(batch_requests)} items...")
    responses = batch_call_vlm_base64(batch_requests, show_progress=True)
    
    # Apply responses to items
    success_count = 0
    for (item_type, idx, item), response in zip(item_refs, responses):
        if response and not response.startswith("ERROR:"):
            annotation = DescriptionAnnotation(
                kind='description',
                text=response,
                provenance=model_name
            )
            item.annotations.append(annotation)
            print(f"[OK] Added annotation to {item_type} {idx}")
            success_count += 1
        else:
            print(f"[FAIL] Failed to annotate {item_type} {idx}: {response[:100] if response else 'No response'}")
    
    print(f"\nBatch annotation complete: {success_count}/{len(batch_requests)} successful")
    return pictures_to_skip


def _annotate_items_sequential(conv_res, model_name, api_url, pictures_to_skip):
    """Sequential annotation (original implementation, kept as fallback)."""
    # Load API key for authentication
    api_key = get_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # Process pictures
    for i, item in enumerate(conv_res.document.pictures):
        if i in pictures_to_skip:
            print(f"Picture {i}: Skipping - inside a table")
            continue
        if not item.annotations:
            try:
                base64_data = str(item.image.uri).split(',')[1] if item.image and item.image.uri else None
                if not base64_data:
                    print(f"Picture {i}: No base64 data available")
                    continue
                    
                print(f"Picture {i}: Using base64 data, length: {len(base64_data)}")
                
                payload = {
                    "model": model_name,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPTS_DESC["image"]},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_data}"}}
                        ]
                    }]
                }
                
                response = make_api_call_with_retries(payload, api_url, headers, "picture", i, OUTPUT_DIR)
                
                if response and response.status_code == 200:
                    result = response.json()
                    description_text = result.get('choices', [{}])[0].get('message', {}).get('content', 'No description available')
                    annotation = DescriptionAnnotation(kind='description', text=description_text, provenance=model_name)
                    item.annotations.append(annotation)
                    print(f"Added annotation to picture {i}")
                else:
                    print(f"Failed to annotate picture {i}: HTTP {response.status_code if response else 'No response'}")
            except Exception as e:
                print(f"Error annotating picture {i}: {str(e)}")
    
    # Process tables
    for i, item in enumerate(conv_res.document.tables):
        if not item.annotations:
            try:
                base64_data = str(item.image.uri.path).split(',')[1] if item.image and item.image.uri else None
                if not base64_data:
                    print(f"Table {i}: No base64 data available")
                    continue
                    
                print(f"Table {i}: Using base64 data, length: {len(base64_data)}")
                
                payload = {
                    "model": model_name,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPTS_DESC["table"]},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_data}"}}
                        ]
                    }]
                }
                
                response = make_api_call_with_retries(payload, api_url, headers, "table", i, OUTPUT_DIR)
                
                if response and response.status_code == 200:
                    result = response.json()
                    description_text = result.get('choices', [{}])[0].get('message', {}).get('content', 'No description available')
                    annotation = DescriptionAnnotation(kind='description', text=description_text, provenance=model_name)
                    item.annotations.append(annotation)
                    print(f"Added annotation to table {i}")
                else:
                    print(f"Failed to annotate table {i}: HTTP {response.status_code if response else 'No response'}")
            except Exception as e:
                print(f"Error annotating table {i}: {str(e)}")
    
    return pictures_to_skip


def configure_pipeline_options(model_name:str="granite3.2-vision:latest", cuda_device_id:int=None):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.enable_remote_services = True
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = False # True For debugging
    pipeline_options.generate_parsed_pages = False  # True For debugging
    pipeline_options.generate_picture_images = True
    pipeline_options.do_picture_classification = False  # False to Avoid CUDA OOM
    pipeline_options.do_picture_description = True   # False Avoid CUDA OOM
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = EasyOcrOptions()
    pipeline_options.do_code_enrichment = True
    pipeline_options.do_formula_enrichment = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.generate_table_images = True
    
    print(f"DEBUG: Pipeline options - generate_table_images: {pipeline_options.generate_table_images}")
    print(f"DEBUG: Pipeline options - do_table_structure: {pipeline_options.do_table_structure}")
    
    # Set accelerator with specific CUDA device if provided
    if cuda_device_id is not None:
        print(f"DEBUG: Using CUDA device {cuda_device_id}")
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=NUM_THREADS, device=f"cuda:{cuda_device_id}"
        )
    else:
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=NUM_THREADS, device=AcceleratorDevice.AUTO
        )

    ### Set picture description API options for motormaven endpoint
    pipeline_options.picture_description_options = motormaven_vlm_options()
    
    print("DEBUG: Using manual table annotation (no built-in table description options)")
    
    return pipeline_options

def process_single_document(doc_path, doc_converter, output_dir, is_pdf=True):
    """Process a single document file (PDF or HTML) and save outputs.
    
    Args:
        doc_path: Path to the document file
        doc_converter: DocumentConverter instance
        output_dir: Output directory path
        is_pdf: Whether the document is a PDF (affects annotation behavior)
    """
    doc_start = time.time()
    
    try:
        conv_res = doc_converter.convert(str(doc_path))
        doc_filename = conv_res.input.file.stem
        
        # Create per-document output directory
        doc_output_dir = output_dir / doc_filename
        doc_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Annotate images and tables (only for PDFs with complex layouts)
        # HTML files typically have simpler structure
        if is_pdf:
            pictures_to_skip = annotate_items_with_images(conv_res)
            
            # Create folders for images
            tables_dir = doc_output_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            
            # Save table images
            table_counter = 0
            for element, _level in conv_res.document.iterate_items():
                if isinstance(element, TableItem):
                    table_counter += 1
                    element_image_filename = tables_dir / f"{doc_filename}-table-{table_counter}.png"
                    _log.info(f"Table {element.self_ref} - Caption: {element.caption_text(doc=conv_res.document)}")
                    try:
                        with element_image_filename.open("wb") as fp:
                            element.get_image(conv_res.document).save(fp, "PNG")
                    except Exception as e:
                        _log.warning(f"Could not save table image: {e}")
        
        # Create artifacts directory for referenced images
        artifacts_dir = doc_output_dir / "ref_artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save markdown with externally referenced pictures
        md_filename = doc_output_dir / f"{doc_filename}_ref.md"
        conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED, artifacts_dir=artifacts_dir)
        
        elapsed = time.time() - doc_start
        file_type = "PDF" if is_pdf else "HTML"
        _log.info(f"[OK] Processed {file_type} {doc_filename} in {elapsed:.1f}s")
        return True, doc_filename, elapsed
        
    except Exception as e:
        elapsed = time.time() - doc_start
        _log.error(f"[FAIL] Failed to process {doc_path.name}: {str(e)}")
        return False, doc_path.name, elapsed

# Backward compatibility alias
def process_single_pdf(pdf_path, doc_converter, output_dir):
    """Backward compatibility wrapper for process_single_document."""
    return process_single_document(pdf_path, doc_converter, output_dir, is_pdf=True)


def create_multi_format_converter(cuda_device_id=None):
    """Create a DocumentConverter that supports both PDF and HTML formats."""
    # PDF pipeline options (full processing)
    pdf_pipeline_options = configure_pipeline_options(cuda_device_id=cuda_device_id)
    
    # Initialize document converter with both PDF and HTML support
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
            InputFormat.HTML: HTMLFormatOption(),  # HTML uses default options
        }
    )
    return doc_converter


if __name__ == "__main__":
    start_time = time.time()
    
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    check_cuda_memory(CUDA_DEVICE_ID)
    
    # Collect document files (PDF and HTML) from input path
    doc_files, temp_dir = collect_input_files(INPUT_PATH)
    
    if not doc_files:
        _log.error("No document files found to process")
        exit(1)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize document converter with multi-format support
    doc_converter = create_multi_format_converter(cuda_device_id=CUDA_DEVICE_ID)
    
    # Process all documents
    results = []
    for i, doc_path in enumerate(doc_files, 1):
        is_pdf = doc_path.suffix.lower() == '.pdf'
        file_type = "PDF" if is_pdf else "HTML"
        
        _log.info(f"\n{'='*60}")
        _log.info(f"Processing {file_type} {i}/{len(doc_files)}: {doc_path.name}")
        _log.info(f"{'='*60}")
        
        success, name, elapsed = process_single_document(doc_path, doc_converter, OUTPUT_DIR, is_pdf=is_pdf)
        results.append((success, name, elapsed, file_type))
    
    # Cleanup temp directory if created from zip
    if temp_dir:
        _log.info(f"Cleaning up temp directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Summary
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r[0])
    failed = len(results) - successful
    pdf_count = sum(1 for r in results if r[3] == "PDF")
    html_count = sum(1 for r in results if r[3] == "HTML")
    
    _log.info(f"\n{'='*60}")
    _log.info(f"PROCESSING COMPLETE")
    _log.info(f"{'='*60}")
    _log.info(f"Total documents: {len(results)} (PDF: {pdf_count}, HTML: {html_count})")
    _log.info(f"Successful: {successful}")
    _log.info(f"Failed: {failed}")
    _log.info(f"Total time: {total_time:.1f}s")
    _log.info(f"Output directory: {OUTPUT_DIR}")