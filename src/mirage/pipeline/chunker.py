"""
Simple Semantic Chunking System
Uses a single comprehensive prompt to chunk markdown documents semantically.
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from mirage.core import llm as call_llm
from mirage.core.llm import call_vlm_with_multiple_images, setup_logging, call_llm_simple
from mirage.core.prompts import PROMPTS_CHUNK
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Configuration (from call_llm.py)
INPUT_FILE = "output/results/markdown/document/document_ref.md"
INPUT_DIR = None  # Set to a directory path to process all .md files in it
OUTPUT_DIR = "output/results/chunks"
LLM_MODEL_NAME = "gemini-2.0-flash"

# Windowing parameters (chars, not tokens)
WINDOW_SIZE = 20000  # 5000 tokens
OVERLAP_SIZE = 2000   # 500 tokens

# Parallel processing
NUM_FILE_WORKERS = 4  # Number of files to process in parallel

def parse_chunks_from_response(response: str) -> List[Dict]:
    """Parse structured chunks from LLM response
    
    Expected format per prompt:
    <chunk_id>VALUE<|#|><chunk_type>VALUE<|#|><content>VALUE<|#|><artifact>VALUE<|#|><status>VALUE<|#|><chunk_end>
    """
    chunks = []
    
    # Split by <chunk_end> marker
    chunk_blocks = response.split('<chunk_end>')
    
    for block in chunk_blocks:
        block = block.strip()
        if not block:
            continue
        
        # Parse fields separated by <|#|>
        # Format: <field_name>VALUE<|#|>
        parts = block.split('<|#|>')
        
        if len(parts) >= 5:
            # Extract field values by removing the field name prefix
            # parts[0] = "<chunk_id>VALUE"
            # parts[1] = "<chunk_type>VALUE"
            # parts[2] = "<content>VALUE" (may contain newlines)
            # parts[3] = "<artifact>VALUE"
            # parts[4] = "<status>VALUE"
            
            chunk_id = re.sub(r'^<chunk_id>', '', parts[0]).strip()
            chunk_type = re.sub(r'^<chunk_type>', '', parts[1]).strip()
            content = re.sub(r'^<content>', '', parts[2]).strip()
            artifact = re.sub(r'^<artifact>', '', parts[3]).strip()
            status = re.sub(r'^<status>', '', parts[4]).strip()
            
            chunks.append({
                'chunk_id': chunk_id,
                'chunk_type': chunk_type,
                'content': content,
                'artifact': artifact,
                'status': status
            })
        else:
            logging.warning(f"Skipping malformed chunk block with {len(parts)} parts (expected 5+). Block preview: {block[:200]}")
            print(f"⚠️ Skipping malformed chunk block with {len(parts)} parts")
    
    return chunks


def find_overlap(incomplete_content: str, new_window: str, max_search: int = None) -> int:
    """Find where incomplete content overlaps with new window
    Returns the position in new_window where unique content starts
    
    Args:
        incomplete_content: The content from the incomplete chunk (LLM-parsed markdown)
        new_window: The raw markdown text from the new window
        max_search: Maximum search range (defaults to OVERLAP_SIZE * 2 to account for potential formatting differences)
    """
    if max_search is None:
        max_search = OVERLAP_SIZE * 2  # Search up to 2x overlap size to account for formatting differences
    
    # Try to find overlap by checking last N chars of incomplete content
    # against beginning of new window (where overlap should be)
    search_range = min(max_search, len(new_window))
    incomplete_len = len(incomplete_content)
    
    # Try multiple snippet lengths, starting from larger to smaller
    # This helps find the best match even if there's slight formatting difference
    for length in range(min(max_search, incomplete_len), 50, -50):
        # Get last N characters from incomplete content
        search_snippet = incomplete_content[-length:].strip()
        
        if not search_snippet:
            continue
            
        # Search in the first part of new window (where overlap should be)
        search_text = new_window[:search_range]
        
        # Try exact match first
        if search_snippet in search_text:
            overlap_pos = search_text.find(search_snippet)
            # Return position after the overlap
            return overlap_pos + len(search_snippet)
        
        # Try without leading/trailing whitespace differences
        search_snippet_normalized = ' '.join(search_snippet.split())
        search_text_normalized = ' '.join(search_text[:min(len(search_snippet_normalized) * 2, len(search_text))].split())
        
        if search_snippet_normalized in search_text_normalized:
            # Find approximate position in original text
            # Use a shorter snippet to find the position
            short_snippet = search_snippet[-min(200, len(search_snippet)):]
            if short_snippet in search_text:
                overlap_pos = search_text.find(short_snippet)
                return overlap_pos + len(short_snippet)
    
    # No overlap found, return 0 (start from beginning)
    return 0


def chunk_with_windows(markdown_text: str) -> Tuple[List[Dict], Dict[int, Dict[str, str]]]:
    """Process markdown in windows with smart handling of incomplete chunks
    
    Returns:
        tuple: (list of chunks, dict of window queries and responses)
    """
    print(f"📄 Document size: {len(markdown_text):,} characters")
    print(f"🔧 Window: {WINDOW_SIZE:,} chars, Overlap: {OVERLAP_SIZE:,} chars")
    
    all_chunks = []
    position = 0
    window_num = 0
    incomplete_chunk = None  # Carry over incomplete chunks
    
    # Store queries and responses for debugging
    queries_responses = {}
    
    while position < len(markdown_text):
        window_num += 1
        
        # Calculate window boundaries with overlap
        window_end = min(position + WINDOW_SIZE, len(markdown_text))
        window_text = markdown_text[position:window_end]
        
        # If we have an incomplete chunk from previous window, merge it
        if incomplete_chunk:
            print(f"\n🔗 Merging incomplete chunk from previous window...")
            print(f"   Incomplete chunk content length: {len(incomplete_chunk['content']):,} chars")
            print(f"   New window text length: {len(window_text):,} chars")
            
            # Find overlap between incomplete chunk and current window
            # The new window should start with OVERLAP_SIZE chars from previous window
            overlap_end = find_overlap(incomplete_chunk['content'], window_text)
            
            if overlap_end > 0:
                print(f"   ✅ Found overlap at position {overlap_end} (expected around 0-{OVERLAP_SIZE*2})")
                
                # Debug: Show what's being merged
                overlap_text = window_text[:overlap_end]
                continuation = window_text[overlap_end:]
                print(f"   Overlap text (will be skipped): ...{overlap_text[-50:]}...")
                print(f"   Continuation text (will be appended): {continuation[:50]}...")
                
                # Remove overlapping portion from window
                # This handles duplicates by:
                # 1. Keeping the incomplete chunk's content (which includes the overlapping portion)
                # 2. Appending only the unique continuation from new window (starting after overlap_end)
                window_text = incomplete_chunk['content'] + window_text[overlap_end:]
                print(f"   Merged text length: {len(window_text):,} chars (incomplete: {len(incomplete_chunk['content'])}, continuation: {len(continuation)})")
            else:
                print(f"   ⚠️ No overlap found (searched first {OVERLAP_SIZE*2} chars)")
                print(f"   Debug: Last 100 chars of incomplete: ...{incomplete_chunk['content'][-100:]}")
                print(f"   Debug: First 100 chars of new window: {window_text[:100]}")
                # No overlap, just prepend
                window_text = incomplete_chunk['content'] + "\n\n" + window_text
            
            incomplete_chunk = None  # Reset
        
        print(f"\n🔄 Processing window {window_num} (pos {position:,} - {window_end:,})")
        
        # Call LLM with the semantic chunking prompt
        try:
            full_prompt = f"{PROMPTS_CHUNK['semantic_chunking']}\n\nMarkdown QUERY to chunk:\n\n{window_text}"
            
            response = call_llm_simple(full_prompt)
            
            # Store query (just the text to chunk) and response for debugging
            queries_responses[window_num] = {
                'query': window_text,
                'response': response
            }
            logging.info(f"Window {window_num}: Query {len(window_text)} chars, Response {len(response)} chars")
            print(f"📝 Stored query ({len(window_text)} chars) and response ({len(response)} chars)")
            
            # Check for empty response
            if not response or not response.strip():
                logging.warning(f"Empty response from LLM for window {window_num}")
                print(f"⚠️ Empty response from LLM for window {window_num}, skipping...")
                incomplete_chunk = None
                # Move to next window
                if window_end >= len(markdown_text):
                    break
                position = window_end - OVERLAP_SIZE
                continue
            
            # Parse chunks from response
            window_chunks = parse_chunks_from_response(response)
            print(f"✅ Parsed {len(window_chunks)} chunks from window {window_num}")
            
            # Print character and word count for each chunk
            for idx, chunk in enumerate(window_chunks, 1):
                content = chunk.get('content', '')
                char_count = len(content)
                word_count = len(content.split())
                print(f"   Chunk {idx}: {char_count:,} chars, {word_count:,} words")
            
            # Check if last chunk is incomplete
            if window_chunks and window_chunks[-1]['status'].upper() == 'INCOMPLETE':
                incomplete_chunk = window_chunks[-1]
                window_chunks = window_chunks[:-1]  # Don't add incomplete chunk yet
                print(f"   ⚠️ Last chunk marked INCOMPLETE, will merge with next window")
            
            all_chunks.extend(window_chunks)
            
        except Exception as e:
            print(f"❌ Error processing window {window_num}: {e}")
            incomplete_chunk = None  # Reset on error
        
        # Move to next window with overlap
        if window_end >= len(markdown_text):
            # End of document - add incomplete chunk if any
            if incomplete_chunk:
                print(f"   📝 Adding final incomplete chunk as-is")
                all_chunks.append(incomplete_chunk)
            break
        
        position = window_end - OVERLAP_SIZE
    
    print(f"\n✅ Total chunks from all windows: {len(all_chunks)}")
    return all_chunks, queries_responses

def renumber_chunks(chunks: List[Dict], file_name: str) -> List[Dict]:
    """Renumber chunks with continuous numbering and add file name"""
    for i, chunk in enumerate(chunks, 1):
        # Reconstruct dict to ensure order: file_name, chunk_id, ...
        original = chunk.copy()
        chunk.clear()
        chunk['file_name'] = file_name
        chunk['chunk_id'] = str(i)
        chunk.update({k: v for k, v in original.items() if k != 'chunk_id'})
    
    print(f"🔢 Renumbered {len(chunks)} chunks and added file name")
    return chunks


def export_to_json(chunks: List[Dict], output_path: Path):
    """Export chunks to JSON file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved {len(chunks)} chunks to {output_path}")


def print_summary(chunks: List[Dict]):
    """Print summary statistics"""
    type_counts = {}
    status_counts = {}
    
    for chunk in chunks:
        chunk_type = chunk.get('chunk_type', 'unknown')
        status = chunk.get('status', 'unknown')
        
        type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print("\n" + "="*60)
    print("📊 CHUNKING SUMMARY")
    print("="*60)
    print(f"Total chunks: {len(chunks)}")
    print(f"\nBy type:")
    for ctype, count in sorted(type_counts.items()):
        print(f"  • {ctype}: {count}")
    print(f"\nBy status:")
    for status, count in sorted(status_counts.items()):
        print(f"  • {status}: {count}")
    
    # Calculate and print average word count
    total_words = 0
    for chunk in chunks:
        content = chunk.get('content', '')
        total_words += len(content.split())
    avg_words = total_words / len(chunks) if chunks else 0
    print(f"\nAverage word count per chunk: {avg_words:.1f}")
    print("="*60)


# ============================================================================
# SINGLE FILE PROCESSING
# ============================================================================

def process_single_file(input_path: Path, output_dir: Path) -> Dict:
    """Process a single markdown file and return results.
    
    Args:
        input_path: Path to markdown file
        output_dir: Directory for output files
        
    Returns:
        Dict with 'success', 'file', 'chunks_count', 'error' keys
    """
    result = {
        'success': False,
        'file': str(input_path),
        'chunks_count': 0,
        'error': None
    }
    
    try:
        if not input_path.exists():
            result['error'] = f"File not found: {input_path}"
            return result
        
        print(f"\n📖 Processing: {input_path.name}")
        markdown_text = input_path.read_text(encoding='utf-8')
        
        # Chunk with windows
        chunks, queries_responses = chunk_with_windows(markdown_text)
        
        # Renumber continuously and add file name
        chunks = renumber_chunks(chunks, input_path.stem)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export chunks to JSON
        output_path = output_dir / f"{input_path.stem}_chunks.json"
        export_to_json(chunks, output_path)
        
        # Export queries and responses for debugging
        queries_responses_path = output_dir / f"{input_path.stem}_queries_responses.json"
        with open(queries_responses_path, 'w', encoding='utf-8') as f:
            json.dump(queries_responses, f, indent=2, ensure_ascii=False)
        
        result['success'] = True
        result['chunks_count'] = len(chunks)
        print(f"✅ {input_path.name}: {len(chunks)} chunks")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"❌ {input_path.name}: Error - {e}")
    
    return result


def process_files_parallel(input_files: List[Path], output_dir: Path, 
                          max_workers: int = NUM_FILE_WORKERS) -> List[Dict]:
    """Process multiple markdown files in parallel.
    
    Args:
        input_files: List of markdown file paths
        output_dir: Base directory for output (subdirs created per file)
        max_workers: Number of parallel workers
        
    Returns:
        List of result dicts from process_single_file
    """
    if not input_files:
        print("❌ No files to process")
        return []
    
    print(f"\n🚀 Processing {len(input_files)} files with {max_workers} parallel workers")
    print("="*60)
    
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {}
        for input_path in input_files:
            # Create per-file output directory
            file_output_dir = output_dir / input_path.stem
            future = executor.submit(process_single_file, input_path, file_output_dir)
            futures[future] = input_path
        
        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc="Chunking files"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                input_path = futures[future]
                results.append({
                    'success': False,
                    'file': str(input_path),
                    'chunks_count': 0,
                    'error': str(e)
                })
    
    # Print summary
    print("\n" + "="*60)
    print("📊 PARALLEL CHUNKING SUMMARY")
    print("="*60)
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    total_chunks = sum(r['chunks_count'] for r in successful)
    
    print(f"Files processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total chunks generated: {total_chunks}")
    
    if failed:
        print("\n❌ Failed files:")
        for r in failed:
            print(f"   • {Path(r['file']).name}: {r['error']}")
    
    print("="*60)
    
    return results


def get_markdown_files(input_path: str) -> List[Path]:
    """Get list of markdown files from path (file or directory).
    
    Args:
        input_path: Path to file or directory
        
    Returns:
        List of Path objects for markdown files
    """
    path = Path(input_path)
    
    if path.is_file():
        return [path] if path.suffix.lower() == '.md' else []
    elif path.is_dir():
        # Find all .md files recursively
        return list(path.glob("**/*.md"))
    else:
        return []


# ============================================================================
# MAIN
# ============================================================================

def main(input_path: Optional[str] = None, output_dir: Optional[str] = None,
         parallel: bool = True, max_workers: int = NUM_FILE_WORKERS):
    """Main execution - supports single file or parallel multi-file processing.
    
    Args:
        input_path: Path to file or directory (uses INPUT_FILE/INPUT_DIR if None)
        output_dir: Output directory (uses OUTPUT_DIR if None)
        parallel: Whether to use parallel processing for multiple files
        max_workers: Number of parallel workers
    """
    # Setup logging
    setup_logging()
    
    print("🚀 Starting Simple Semantic Chunking")
    print(f"🤖 Using model: {LLM_MODEL_NAME}")
    
    # Determine input path
    if input_path is None:
        input_path = INPUT_DIR if INPUT_DIR else INPUT_FILE
    
    # Determine output directory
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_path = Path(output_dir)
    
    # Get list of markdown files
    input_files = get_markdown_files(input_path)
    
    if not input_files:
        print(f"❌ No markdown files found at: {input_path}")
        return
    
    print(f"📂 Found {len(input_files)} markdown file(s)")
    
    # Process files
    if len(input_files) == 1:
        # Single file - process directly
        result = process_single_file(input_files[0], output_path)
        if result['success']:
            print_summary_from_file(output_path / f"{input_files[0].stem}_chunks.json")
    elif parallel:
        # Multiple files - process in parallel
        results = process_files_parallel(input_files, output_path, max_workers)
    else:
        # Multiple files - process sequentially
        print(f"\n🔄 Processing {len(input_files)} files sequentially...")
        for input_file in tqdm(input_files, desc="Chunking files"):
            file_output_dir = output_path / input_file.stem
            process_single_file(input_file, file_output_dir)
    
    print("\n✅ Processing complete!")
    print(f"📁 Output directory: {output_path}")
    print(f"   • Log file: {call_llm.LOG_FILE if hasattr(call_llm, 'LOG_FILE') else 'N/A'}")


def print_summary_from_file(chunks_file: Path):
    """Print summary from saved chunks file."""
    if chunks_file.exists():
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print_summary(chunks)


# =============================================================================
# MODULAR SEMANTIC CHUNKER (scikit-learn style)
# =============================================================================

from mirage.core.base import BaseModule


class SemanticChunker(BaseModule):
    """
    Semantic chunking module for markdown to chunks conversion.
    
    All hyperparameters from Table 2.4 (Semantic Chunking):
        method: 'semantic' (LLM-based), 'fixed', 'sentence'
        window_size: Character window size for chunking (default: 20000, ~5000 tokens)
        overlap_size: Character overlap between windows (default: 2000, ~500 tokens)
        llm_model: LLM model for semantic chunking (default: gpt-oss-120b)
        max_workers: Parallel workers for file processing (default: 4)
        output_dir: Output directory for chunk files
        
    Example:
        chunker = SemanticChunker(
            window_size=20000,
            overlap_size=2000,
            llm_model='gpt-oss-120b'
        )
        chunks = chunker.process(markdown_files)
    """
    
    def __init__(self,
                 method: str = 'semantic',
                 window_size: int = 20000,
                 overlap_size: int = 2000,
                 llm_model: str = 'gpt-oss-120b',
                 max_workers: int = 4,
                 output_dir: str = None,
                 **kwargs):
        """Initialize semantic chunker with all hyperparameters from Table 2.4."""
        super().__init__(
            method=method,
            window_size=window_size,
            overlap_size=overlap_size,
            llm_model=llm_model,
            max_workers=max_workers,
            output_dir=output_dir,
            **kwargs
        )
    
    def process(self, X, **kwargs):
        """
        Process markdown files to semantic chunks.
        
        Args:
            X: List of markdown file paths, or list of dicts with 'markdown_path' key,
               or path to single markdown file/directory
            **kwargs: Runtime overrides for parameters
               
        Returns:
            List of chunk dicts with 'content', 'chunk_id', 'file_name', etc.
        """
        # Apply runtime overrides
        params = {**self._params, **kwargs}
        
        # Normalize input to list of paths
        if isinstance(X, str):
            md_files = get_markdown_files(X)
        elif isinstance(X, list):
            if len(X) > 0 and isinstance(X[0], dict):
                # Output from DocumentProcessor
                md_files = [Path(d['markdown_path']) for d in X if 'markdown_path' in d]
            else:
                md_files = [Path(p) for p in X]
        else:
            md_files = [Path(X)]
        
        if not md_files:
            self._logger.warning("No markdown files to process")
            return []
        
        # Determine output directory
        output_dir = Path(params.get('output_dir') or 'output/chunks')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update global config for windowing
        global WINDOW_SIZE, OVERLAP_SIZE
        WINDOW_SIZE = params.get('window_size', 20000)
        OVERLAP_SIZE = params.get('overlap_size', 2000)
        
        # Process files
        max_workers = params.get('max_workers', 4)
        all_chunks = []
        
        if len(md_files) == 1:
            # Single file
            result = process_single_file(md_files[0], output_dir)
            if result['success']:
                chunks_file = output_dir / f"{md_files[0].stem}_chunks.json"
                if chunks_file.exists():
                    with open(chunks_file, 'r', encoding='utf-8') as f:
                        all_chunks = json.load(f)
        else:
            # Multiple files - parallel processing
            results = process_files_parallel(md_files, output_dir, max_workers=max_workers)
            
            # Combine all chunks
            for result in results:
                if result['success']:
                    file_stem = Path(result['file']).stem
                    chunks_file = output_dir / file_stem / f"{file_stem}_chunks.json"
                    if chunks_file.exists():
                        with open(chunks_file, 'r', encoding='utf-8') as f:
                            file_chunks = json.load(f)
                            all_chunks.extend(file_chunks)
        
        # Renumber chunks globally
        for i, chunk in enumerate(all_chunks, 1):
            chunk['global_chunk_id'] = f"chunk_{i}"
        
        self._logger.info(f"Generated {len(all_chunks)} chunks from {len(md_files)} files")
        self._output = all_chunks
        return all_chunks
    
    def process_text(self, markdown_text: str, file_name: str = 'document') -> List[Dict]:
        """
        Process a single markdown text string to chunks.
        
        Args:
            markdown_text: Raw markdown content
            file_name: Name for the source file
            
        Returns:
            List of chunk dicts
        """
        # Update global config
        global WINDOW_SIZE, OVERLAP_SIZE
        WINDOW_SIZE = self._params.get('window_size', 20000)
        OVERLAP_SIZE = self._params.get('overlap_size', 2000)
        
        chunks, _ = chunk_with_windows(markdown_text)
        chunks = renumber_chunks(chunks, file_name)
        
        return chunks


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Semantic chunking for markdown files")
    parser.add_argument("--input", "-i", type=str, default=None,
                       help="Input file or directory path")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--workers", "-w", type=int, default=NUM_FILE_WORKERS,
                       help=f"Number of parallel workers (default: {NUM_FILE_WORKERS})")
    parser.add_argument("--sequential", "-s", action="store_true",
                       help="Process files sequentially instead of in parallel")
    
    args = parser.parse_args()
    
    main(
        input_path=args.input,
        output_dir=args.output,
        parallel=not args.sequential,
        max_workers=args.workers
    )

