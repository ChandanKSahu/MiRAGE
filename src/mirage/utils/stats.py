#!/usr/bin/env python3
"""
Dataset Statistics Calculator
Measures pages, images, and tokens for each dataset using chunks.json files.
"""

import os
import re
import zipfile
import io
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional

import pypdfium2 as pdfium
from tqdm import tqdm


def get_tokenizer():
    """Get tokenizer - uses GPT2 if available, otherwise word-based estimate."""
    try:
        from transformers import GPT2TokenizerFast
        return GPT2TokenizerFast.from_pretrained("gpt2")
    except Exception:
        return None


def count_tokens(text: str, tokenizer=None) -> int:
    """Count tokens - uses tokenizer if available, otherwise word-based estimate."""
    if not text:
        return 0
    
    if tokenizer:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    
    # Fallback: approximate tokens as ~1.3 * words (GPT average)
    words = len(text.split())
    return int(words * 1.3)


def count_pages_from_zip(zip_path: Path) -> Dict[str, int]:
    """Count pages per document from zip file (PDFs or HTML files)."""
    page_counts = {}
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                stem = Path(name).stem
                lower_name = name.lower()
                
                if lower_name.endswith('.pdf'):
                    try:
                        pdf_bytes = zf.read(name)
                        pdf = pdfium.PdfDocument(pdf_bytes)
                        page_counts[stem] = len(pdf)
                    except Exception:
                        page_counts[stem] = 0
                elif lower_name.endswith(('.html', '.htm')):
                    # HTML files count as 1 "page"
                    page_counts[stem] = 1
    except Exception as e:
        print(f"Error reading zip {zip_path}: {e}")
    return page_counts


def get_file_size_mb_from_zip(zip_path: Path) -> Dict[str, float]:
    """Get file sizes per document from zip file."""
    sizes = {}
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for info in zf.infolist():
                lower = info.filename.lower()
                if lower.endswith(('.pdf', '.html', '.htm')):
                    stem = Path(info.filename).stem
                    sizes[stem] = info.file_size / (1024 * 1024)
    except Exception:
        pass
    return sizes


def get_file_type_from_zip(zip_path: Path) -> str:
    """Detect primary file type in zip."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
            pdfs = sum(1 for n in names if n.lower().endswith('.pdf'))
            htmls = sum(1 for n in names if n.lower().endswith(('.html', '.htm')))
            if pdfs > htmls:
                return "pdf"
            elif htmls > 0:
                return "html"
    except Exception:
        pass
    return "unknown"


def analyze_chunks_json(chunks_path: Path, tokenizer=None) -> Dict[str, Any]:
    """Analyze a chunks.json file to extract stats per PDF."""
    pdf_stats = defaultdict(lambda: {
        "chunks": 0,
        "text_chunks": 0,
        "image_chunks": 0,
        "table_chunks": 0,
        "tokens": 0,
        "chars": 0
    })
    
    try:
        with open(chunks_path, 'r') as f:
            chunks = json.load(f)
        
        for chunk in chunks:
            file_name = chunk.get("file_name", "unknown")
            chunk_type = chunk.get("chunk_type", "").lower()
            content = chunk.get("content", "")
            artifact = chunk.get("artifact", "None")
            
            stats = pdf_stats[file_name]
            stats["chunks"] += 1
            stats["chars"] += len(content)
            stats["tokens"] += count_tokens(content, tokenizer)
            
            # Classify chunk type
            if "image" in chunk_type or (artifact and artifact != "None"):
                stats["image_chunks"] += 1
            elif "table" in chunk_type:
                stats["table_chunks"] += 1
            else:
                stats["text_chunks"] += 1
                
    except Exception as e:
        print(f"Error reading {chunks_path}: {e}")
    
    return dict(pdf_stats)


def analyze_dataset(dataset_name: str, chunks_path: Optional[Path], zip_path: Optional[Path], tokenizer=None) -> Dict[str, Any]:
    """Analyze a single dataset using chunks.json and/or zip file."""
    stats = {
        "dataset_name": dataset_name,
        "source": "chunks.json" if chunks_path else "zip",
        "num_pdfs": 0,
        "total_pages": 0,
        "total_chunks": 0,
        "total_text_chunks": 0,
        "total_image_chunks": 0,
        "total_table_chunks": 0,
        "total_tokens": 0,
        "total_chars": 0,
        "total_size_mb": 0.0,
        "pdfs": []
    }
    
    # Get page counts and file sizes from zip
    page_counts = {}
    file_sizes = {}
    file_type = "unknown"
    if zip_path and zip_path.exists():
        page_counts = count_pages_from_zip(zip_path)
        file_sizes = get_file_size_mb_from_zip(zip_path)
        file_type = get_file_type_from_zip(zip_path)
        stats["total_size_mb"] = sum(file_sizes.values())
        stats["file_type"] = file_type
    
    # Get chunk stats from chunks.json
    chunk_stats = {}
    if chunks_path and chunks_path.exists():
        chunk_stats = analyze_chunks_json(chunks_path, tokenizer)
    
    # Merge stats - use chunk_stats keys as primary if available
    all_pdfs = set(chunk_stats.keys()) | set(page_counts.keys())
    stats["num_pdfs"] = len(all_pdfs)
    
    for pdf_name in sorted(all_pdfs):
        pdf_info = {
            "filename": pdf_name,
            "pages": page_counts.get(pdf_name, 0),
            "size_mb": round(file_sizes.get(pdf_name, 0), 2),
            "chunks": chunk_stats.get(pdf_name, {}).get("chunks", 0),
            "text_chunks": chunk_stats.get(pdf_name, {}).get("text_chunks", 0),
            "image_chunks": chunk_stats.get(pdf_name, {}).get("image_chunks", 0),
            "table_chunks": chunk_stats.get(pdf_name, {}).get("table_chunks", 0),
            "tokens": chunk_stats.get(pdf_name, {}).get("tokens", 0),
            "chars": chunk_stats.get(pdf_name, {}).get("chars", 0),
        }
        stats["pdfs"].append(pdf_info)
        
        stats["total_pages"] += pdf_info["pages"]
        stats["total_chunks"] += pdf_info["chunks"]
        stats["total_text_chunks"] += pdf_info["text_chunks"]
        stats["total_image_chunks"] += pdf_info["image_chunks"]
        stats["total_table_chunks"] += pdf_info["table_chunks"]
        stats["total_tokens"] += pdf_info["tokens"]
        stats["total_chars"] += pdf_info["chars"]
    
    stats["total_size_mb"] = round(stats["total_size_mb"], 2)
    
    # Compute averages
    if stats["num_pdfs"] > 0:
        stats["avg_pages_per_pdf"] = round(stats["total_pages"] / stats["num_pdfs"], 1)
        stats["avg_chunks_per_pdf"] = round(stats["total_chunks"] / stats["num_pdfs"], 1)
        stats["avg_images_per_pdf"] = round(stats["total_image_chunks"] / stats["num_pdfs"], 1)
        stats["avg_tokens_per_pdf"] = int(stats["total_tokens"] / stats["num_pdfs"])
    if stats["total_pages"] > 0:
        stats["avg_tokens_per_page"] = int(stats["total_tokens"] / stats["total_pages"])
    
    return stats


def print_summary(all_stats: List[Dict]) -> None:
    """Print formatted summary of all datasets."""
    print("\n" + "=" * 110)
    print("DATASET STATISTICS SUMMARY (from chunks.json)")
    print("=" * 110)
    
    print(f"\n{'Dataset':<28} {'Docs':>5} {'Pages':>6} {'Chunks':>7} {'Images':>7} {'Tables':>7} {'Tokens':>12} {'Size(MB)':>10}")
    print("-" * 110)
    
    totals = defaultdict(int)
    totals["total_size_mb"] = 0.0
    
    for stats in all_stats:
        print(f"{stats['dataset_name']:<28} {stats['num_pdfs']:>5} {stats['total_pages']:>6} "
              f"{stats['total_chunks']:>7} {stats['total_image_chunks']:>7} {stats['total_table_chunks']:>7} "
              f"{stats['total_tokens']:>12,} {stats['total_size_mb']:>10.1f}")
        
        totals["num_pdfs"] += stats["num_pdfs"]
        totals["total_pages"] += stats["total_pages"]
        totals["total_chunks"] += stats["total_chunks"]
        totals["total_image_chunks"] += stats["total_image_chunks"]
        totals["total_table_chunks"] += stats["total_table_chunks"]
        totals["total_tokens"] += stats["total_tokens"]
        totals["total_size_mb"] += stats["total_size_mb"]
    
    print("-" * 110)
    print(f"{'TOTAL':<28} {totals['num_pdfs']:>5} {totals['total_pages']:>6} "
          f"{totals['total_chunks']:>7} {totals['total_image_chunks']:>7} {totals['total_table_chunks']:>7} "
          f"{totals['total_tokens']:>12,} {totals['total_size_mb']:>10.1f}")
    print("=" * 110)
    
    # Print averages
    print("\nAVERAGES PER DATASET:")
    print(f"{'Dataset':<28} {'Pg/Doc':>8} {'Chunks/Doc':>10} {'Img/Doc':>8} {'Tok/Doc':>12} {'Tok/Page':>10}")
    print("-" * 80)
    for stats in all_stats:
        print(f"{stats['dataset_name']:<28} {stats.get('avg_pages_per_pdf', 0):>8.1f} "
              f"{stats.get('avg_chunks_per_pdf', 0):>10.1f} {stats.get('avg_images_per_pdf', 0):>8.1f} "
              f"{stats.get('avg_tokens_per_pdf', 0):>12,} {stats.get('avg_tokens_per_page', 0):>10,}")


def find_datasets(data_dir: Path, results_dir: Path) -> List[Dict]:
    """Find all datasets and their corresponding chunks.json files."""
    datasets = []
    
    # Get all zip files in data directory
    zip_files = {z.stem: z for z in data_dir.glob("*.zip")}
    
    # Get all chunks.json files in results directory
    chunks_files = {}
    if results_dir.exists():
        for chunks_path in results_dir.glob("*/chunks.json"):
            dataset_name = chunks_path.parent.name
            chunks_files[dataset_name] = chunks_path
    
    # Merge: prefer chunks.json when available
    all_datasets = set(zip_files.keys()) | set(chunks_files.keys())
    
    for name in sorted(all_datasets):
        datasets.append({
            "name": name,
            "zip_path": zip_files.get(name),
            "chunks_path": chunks_files.get(name)
        })
    
    return datasets


def compute_dataset_stats(
    output_dir: str,
    pdf_dir: str = None,
    chunks_file: str = None,
    tokenizer=None
) -> Dict[str, Any]:
    """
    Compute comprehensive dataset statistics from a trial results directory.
    
    Args:
        output_dir: Path to the output directory (e.g., output/results/my_dataset)
        pdf_dir: Path to source PDFs directory (for page counts)
        chunks_file: Path to chunks.json (defaults to output_dir/chunks.json)
        tokenizer: Optional tokenizer for accurate token counting
    
    Returns:
        Dict with keys: total_images, total_tables, total_pages, total_tokens,
                       num_pdfs, per_pdf_stats, etc.
    """
    output_path = Path(output_dir)
    
    # Default chunks file path
    if chunks_file is None:
        chunks_file = output_path / "chunks.json"
    else:
        chunks_file = Path(chunks_file)
    
    stats = {
        "num_pdfs": 0,
        "total_images": 0,
        "total_tables": 0,
        "total_pages": 0,
        "total_tokens": 0,
        "total_chars": 0,
        "total_chunks": 0,
        "per_pdf_stats": []
    }
    
    # Get PDF names from chunks.json
    pdf_names = set()
    chunks_data = []
    if chunks_file.exists():
        try:
            with open(chunks_file, 'r') as f:
                chunks_data = json.load(f)
            for chunk in chunks_data:
                file_name = chunk.get("file_name", "")
                if file_name:
                    pdf_names.add(file_name)
            stats["total_chunks"] = len(chunks_data)
        except Exception as e:
            print(f"Warning: Could not read chunks.json: {e}")
    
    stats["num_pdfs"] = len(pdf_names)
    
    # Count images and tables from markdown directory structure
    markdown_dir = output_path / "markdown"
    all_image_files = set()
    all_table_files = set()
    per_pdf_images = defaultdict(set)
    per_pdf_tables = defaultdict(set)
    
    if markdown_dir.exists():
        for subdir in markdown_dir.iterdir():
            if subdir.is_dir():
                pdf_name = subdir.name
                
                # Count images from ref_artifacts/
                artifact_dir = subdir / "ref_artifacts"
                if artifact_dir.exists():
                    for img_file in artifact_dir.glob("image_*"):
                        if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                            all_image_files.add(img_file.name)
                            per_pdf_images[pdf_name].add(img_file.name)
                
                # Count tables from tables/
                table_dir = subdir / "tables"
                if table_dir.exists():
                    for table_file in table_dir.glob("*"):
                        if table_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                            all_table_files.add(table_file.name)
                            per_pdf_tables[pdf_name].add(table_file.name)
    
    stats["total_images"] = len(all_image_files)
    stats["total_tables"] = len(all_table_files)
    
    # Get page counts from source PDFs
    page_counts = {}
    if pdf_dir:
        pdf_path = Path(pdf_dir)
        if pdf_path.exists():
            for pdf_file in pdf_path.glob("*.pdf"):
                pdf_stem = pdf_file.stem
                # Match PDF names to chunk file names
                for chunk_name in pdf_names:
                    if chunk_name.lower() in pdf_stem.lower() or pdf_stem.lower() in chunk_name.lower():
                        try:
                            pdf = pdfium.PdfDocument(pdf_file)
                            page_counts[chunk_name] = len(pdf)
                        except Exception:
                            page_counts[chunk_name] = 0
                        break
    
    stats["total_pages"] = sum(page_counts.values())
    
    # Count tokens from chunks
    total_tokens = 0
    total_chars = 0
    per_pdf_tokens = defaultdict(int)
    per_pdf_chars = defaultdict(int)
    
    for chunk in chunks_data:
        content = chunk.get("content", "")
        file_name = chunk.get("file_name", "unknown")
        
        chars = len(content)
        tokens = count_tokens(content, tokenizer)
        
        total_chars += chars
        total_tokens += tokens
        per_pdf_chars[file_name] += chars
        per_pdf_tokens[file_name] += tokens
    
    stats["total_tokens"] = total_tokens
    stats["total_chars"] = total_chars
    
    # Build per-PDF stats
    for pdf_name in sorted(pdf_names):
        pdf_stats = {
            "filename": pdf_name,
            "pages": page_counts.get(pdf_name, 0),
            "images": len(per_pdf_images.get(pdf_name, set())),
            "tables": len(per_pdf_tables.get(pdf_name, set())),
            "tokens": per_pdf_tokens.get(pdf_name, 0),
            "chars": per_pdf_chars.get(pdf_name, 0)
        }
        stats["per_pdf_stats"].append(pdf_stats)
    
    # Compute averages
    if stats["num_pdfs"] > 0:
        stats["avg_pages_per_pdf"] = round(stats["total_pages"] / stats["num_pdfs"], 1)
        stats["avg_images_per_pdf"] = round(stats["total_images"] / stats["num_pdfs"], 1)
        stats["avg_tables_per_pdf"] = round(stats["total_tables"] / stats["num_pdfs"], 1)
        stats["avg_tokens_per_pdf"] = int(stats["total_tokens"] / stats["num_pdfs"])
    
    if stats["total_pages"] > 0:
        stats["avg_tokens_per_page"] = int(stats["total_tokens"] / stats["total_pages"])
    
    return stats


def print_dataset_stats(stats: Dict[str, Any]) -> None:
    """Print formatted dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"\nImages: {stats['total_images']}")
    print(f"Tables: {stats['total_tables']}")
    print(f"Pages: {stats['total_pages']}")
    print(f"Tokens: {stats['total_tokens']:,}")
    print(f"PDFs: {stats['num_pdfs']}")
    print(f"Chunks: {stats['total_chunks']}")
    print("=" * 60)


def compute_qa_category_stats(qa_data: List[Dict]) -> Dict[str, Any]:
    """
    Compute QA category statistics including multihop/multimodal intersection.
    
    Args:
        qa_data: List of QA pairs from deduplicated dataset
    
    Returns:
        Dict with category counts and percentages
    """
    total = len(qa_data)
    if total == 0:
        return {
            'total_qa_pairs': 0,
            'multihop_count': 0,
            'multimodal_count': 0,
            'multihop_multimodal_count': 0,
            'multihop_only_count': 0,
            'multimodal_only_count': 0,
            'text_only_count': 0,
            'avg_difficulty': 0.0,
            'avg_relevance': 0.0
        }
    
    multihop = 0
    multimodal = 0
    both = 0
    
    # Track difficulty and relevance scores
    difficulty_scores = []
    relevance_scores = []
    
    for qa in qa_data:
        # Multihop: hop_count > 1 or multiple chunks added
        hop_count = qa.get('hop_count', 0)
        chunks_added = qa.get('chunks_added', [])
        is_multihop = hop_count > 1 or (isinstance(chunks_added, list) and len(chunks_added) > 1)
        
        # Multimodal: has image_path in context chunks OR markdown images in content
        # Match logic from metrics_optimized.py for consistency
        context_chunks = qa.get('context_chunks', [])
        is_multimodal = False
        for chunk in context_chunks:
            if isinstance(chunk, dict):
                # Check image_path field
                image_path = chunk.get('image_path')
                if image_path and image_path not in ('None', 'null', None) and str(image_path).strip():
                    is_multimodal = True
                    break
                # Check content for markdown image references (e.g., ![alt](path))
                content = chunk.get('content', '')
                if content and re.search(r'!\[[^\]]*\]\([^)]+\)', content):
                    is_multimodal = True
                    break
        
        if is_multihop:
            multihop += 1
        if is_multimodal:
            multimodal += 1
        if is_multihop and is_multimodal:
            both += 1
        
        # Extract difficulty and relevance scores
        try:
            diff = qa.get('difficulty_score', qa.get('difficulty', 0))
            if diff is not None and str(diff).strip():
                difficulty_scores.append(float(diff))
        except (ValueError, TypeError):
            pass
        try:
            rel = qa.get('relevance_score', qa.get('relevance', 0))
            if rel is not None and str(rel).strip():
                relevance_scores.append(float(rel))
        except (ValueError, TypeError):
            pass
    
    # Exclusive counts
    multihop_only = multihop - both
    multimodal_only = multimodal - both
    text_only = total - multihop - multimodal + both
    
    # Compute averages
    avg_difficulty = round(sum(difficulty_scores) / len(difficulty_scores), 2) if difficulty_scores else 0.0
    avg_relevance = round(sum(relevance_scores) / len(relevance_scores), 2) if relevance_scores else 0.0
    
    return {
        'total_qa_pairs': total,
        'multihop_count': multihop,
        'multimodal_count': multimodal,
        'multihop_multimodal_count': both,
        'multihop_only_count': multihop_only,
        'multimodal_only_count': multimodal_only,
        'text_only_count': text_only,
        'multihop_pct': round(100 * multihop / total, 1),
        'multimodal_pct': round(100 * multimodal / total, 1),
        'multihop_multimodal_pct': round(100 * both / total, 1),
        'multihop_only_pct': round(100 * multihop_only / total, 1),
        'multimodal_only_pct': round(100 * multimodal_only / total, 1),
        'text_only_pct': round(100 * text_only / total, 1),
        'avg_difficulty': avg_difficulty,
        'avg_relevance': avg_relevance
    }


def print_qa_category_stats(stats: Dict[str, Any]) -> None:
    """Print formatted QA category statistics."""
    total = stats.get('total_qa_pairs', 0)
    if total == 0:
        print("\n[WARN] No QA pairs to analyze")
        return
    
    print("\n" + "=" * 60)
    print("QA CATEGORY BREAKDOWN")
    print("=" * 60)
    print(f"\nTotal QA Pairs: {total}")
    print()
    print("Category Counts:")
    print(f"  Multihop:                      {stats['multihop_count']:>3} ({stats['multihop_pct']:>5.1f}%)")
    print(f"  Multimodal:                    {stats['multimodal_count']:>3} ({stats['multimodal_pct']:>5.1f}%)")
    print(f"  Both (Multihop AND Multimodal):  {stats['multihop_multimodal_count']:>3} ({stats['multihop_multimodal_pct']:>5.1f}%)")
    print()
    print("Exclusive Breakdown:")
    print(f"  Multihop only (text):          {stats['multihop_only_count']:>3} ({stats['multihop_only_pct']:>5.1f}%)")
    print(f"  Multimodal only (single-hop):  {stats['multimodal_only_count']:>3} ({stats['multimodal_only_pct']:>5.1f}%)")
    print(f"  Both (multihop + multimodal):  {stats['multihop_multimodal_count']:>3} ({stats['multihop_multimodal_pct']:>5.1f}%)")
    print(f"  Neither (single-hop, text):   {stats['text_only_count']:>3} ({stats['text_only_pct']:>5.1f}%)")
    print()
    print("Quality Scores (0-10 scale):")
    print(f"  Avg Difficulty:                {stats.get('avg_difficulty', 0.0):>5.2f}")
    print(f"  Avg Relevance:                 {stats.get('avg_relevance', 0.0):>5.2f}")
    print("=" * 60)


def main():
    """Main entry point."""
    data_dir = Path(__file__).parent
    project_root = data_dir.parent
    results_dir = project_root / "trials" / "results"
    
    datasets = find_datasets(data_dir, results_dir)
    
    if not datasets:
        print("No datasets found.")
        return
    
    print(f"Found {len(datasets)} datasets to analyze...")
    
    # Show which have chunks.json
    with_chunks = sum(1 for d in datasets if d["chunks_path"])
    print(f"  - {with_chunks} with chunks.json (processed)")
    print(f"  - {len(datasets) - with_chunks} zip-only (not yet processed)")
    
    # Load tokenizer once
    tokenizer = get_tokenizer()
    if tokenizer:
        print("Using GPT2 tokenizer for token counting")
    else:
        print("Using word-based token estimation")
    
    all_stats = []
    for ds in tqdm(datasets, desc="Analyzing datasets"):
        stats = analyze_dataset(
            ds["name"],
            ds["chunks_path"],
            ds["zip_path"],
            tokenizer
        )
        all_stats.append(stats)
    
    # Print summary
    print_summary(all_stats)
    
    # Save detailed results
    output_file = data_dir / "dataset_stats.json"
    with open(output_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nDetailed stats saved to: {output_file}")


if __name__ == "__main__":
    main()
