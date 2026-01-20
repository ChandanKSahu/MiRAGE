#!/usr/bin/env python3
"""
Multihop QA Visualization - Shows keyword chains linking chunks to QA pairs.

Flow: Initial Chunk → [Depth 1: Queries → Chunks] → [Depth 2: ...] → Context → Keywords → QA

Usage:
    python visualize_multihop.py [--qa-file PATH] [--index N] [--output PATH]
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Set

def highlight_keywords_html(text: str, keywords: Set[str], color: str = "#00d4ff") -> str:
    """Highlight keywords in text with HTML spans."""
    if not keywords:
        return text.replace('\n', '<br>')
    
    # Find all keyword matches
    matches = []
    text_lower = text.lower()
    for kw in keywords:
        kw_lower = kw.lower()
        start = 0
        while True:
            pos = text_lower.find(kw_lower, start)
            if pos == -1:
                break
            matches.append((pos, pos + len(kw), kw))
            start = pos + 1
    
    if not matches:
        return text.replace('\n', '<br>')
    
    # Sort and merge overlapping
    matches.sort(key=lambda x: x[0])
    merged = []
    for start, end, kw in matches:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end), merged[-1][2])
        else:
            merged.append((start, end, kw))
    
    # Build highlighted text
    result = []
    last_end = 0
    for start, end, kw in merged:
        result.append(text[last_end:start].replace('\n', '<br>'))
        result.append(f'<span class="kw" style="background:{color};color:#000;padding:2px 6px;border-radius:4px;font-weight:600;">{text[start:end]}</span>')
        last_end = end
    result.append(text[last_end:].replace('\n', '<br>'))
    return ''.join(result)

def generate_html_visualization(qa_item: Dict, output_path: str = None) -> str:
    """Generate an HTML visualization of a multihop QA pair."""
    
    # Extract data
    question = qa_item.get('question', '')
    answer = qa_item.get('answer', '')
    context_chunks = qa_item.get('context_chunks', [])
    keywords_per_chunk = qa_item.get('keywords_per_chunk', {})
    related_keywords = qa_item.get('related_keywords', '')
    iteration_logs = qa_item.get('iteration_logs', [])
    hop_count = qa_item.get('hop_count', 0)
    depth_reached = qa_item.get('depth_reached', 0)
    
    # Collect all keywords
    all_keywords = set()
    for chunk_kws in keywords_per_chunk.values():
        all_keywords.update(chunk_kws)
    
    # Get initial chunk
    initial_chunk = context_chunks[0] if context_chunks else {}
    
    # Color palette
    query_colors = ['#42a5f5', '#ff9800', '#66bb6a', '#ec407a', '#ab47bc', '#26c6da', '#ffca28', '#78909c']
    
    html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multihop QA Visualization</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Space Grotesk', sans-serif;
            background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
            min-height: 100vh;
            color: #e6edf3;
            padding: 2rem;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        h1 {
            text-align: center;
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #58a6ff, #a371f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { text-align: center; color: #8b949e; margin-bottom: 2rem; }
        
        .stats {
            display: flex;
            justify-content: center;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        .stat {
            background: #21262d;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #30363d;
        }
        .stat-value { font-size: 1.8rem; font-weight: 700; color: #58a6ff; }
        .stat-label { color: #8b949e; font-size: 0.85rem; }
        
        .section {
            background: #161b22;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid #30363d;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #58a6ff;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .initial-chunk {
            background: #0d1117;
            border: 2px solid #58a6ff;
            border-radius: 10px;
            padding: 1.25rem;
        }
        .initial-label {
            background: #58a6ff;
            color: #0d1117;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            display: inline-block;
        }
        .chunk-content {
            font-size: 0.9rem;
            line-height: 1.6;
            color: #c9d1d9;
            max-height: 150px;
            overflow-y: auto;
        }
        
        .depth-section {
            margin-bottom: 1.5rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px dashed #30363d;
        }
        .depth-section:last-child { border-bottom: none; }
        .depth-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .depth-badge {
            background: #a371f7;
            color: #0d1117;
            padding: 6px 14px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.9rem;
        }
        .depth-status {
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .status-complete { background: rgba(63,185,80,0.2); color: #3fb950; }
        .status-incomplete { background: rgba(210,153,34,0.2); color: #d29922; }
        .depth-explanation {
            color: #8b949e;
            font-size: 0.85rem;
            margin-bottom: 1rem;
            font-style: italic;
        }
        
        .queries-grid {
            display: grid;
            gap: 1rem;
        }
        .query-column {
            background: #0d1117;
            border-radius: 8px;
            overflow: hidden;
        }
        .query-header {
            padding: 0.75rem 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: #0d1117;
            font-weight: 600;
        }
        .query-chunks {
            padding: 0.75rem;
        }
        .retrieved-chunk {
            background: #21262d;
            border-radius: 6px;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            border-left: 3px solid;
        }
        .retrieved-chunk:last-child { margin-bottom: 0; }
        .chunk-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        .chunk-id {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            background: rgba(255,255,255,0.1);
            padding: 2px 8px;
            border-radius: 4px;
        }
        .chunk-verdict {
            font-size: 0.7rem;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
        }
        .verdict-related { background: rgba(63,185,80,0.2); color: #3fb950; }
        .verdict-explanatory { background: rgba(88,166,255,0.2); color: #58a6ff; }
        .no-chunks {
            color: #8b949e;
            font-size: 0.85rem;
            text-align: center;
            padding: 1rem;
        }
        
        .context-box {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 1rem;
        }
        .context-chunks-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        .context-chunk-tag {
            background: #21262d;
            padding: 6px 12px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            border: 1px solid #30363d;
        }
        
        .keyword-chain {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.5rem;
            padding: 1rem;
            background: #0d1117;
            border-radius: 8px;
        }
        .chain-kw {
            padding: 8px 14px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .chain-arrow { color: #58a6ff; font-size: 1.3rem; }
        .chain-relation { color: #8b949e; font-size: 0.8rem; margin: 0 0.5rem; }
        
        .qa-box {
            background: linear-gradient(135deg, rgba(88,166,255,0.1), rgba(163,113,247,0.1));
            border-radius: 10px;
            padding: 1.5rem;
            border: 1px solid rgba(88,166,255,0.3);
        }
        .qa-label { font-weight: 600; margin-bottom: 0.5rem; }
        .question-label { color: #58a6ff; }
        .answer-label { color: #3fb950; }
        .question-text, .answer-text {
            font-size: 1rem;
            line-height: 1.7;
            margin-bottom: 1rem;
        }
        .answer-text {
            background: rgba(0,0,0,0.3);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 0;
        }
        
        .kw { font-weight: 600; }
        .flow-arrow {
            text-align: center;
            color: #58a6ff;
            font-size: 2rem;
            padding: 0.5rem 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔗 Multihop QA Visualization</h1>
        <p class="subtitle">Initial Chunk → Retrieval Iterations → Context → Keyword Chain → QA</p>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value">''' + str(len(context_chunks)) + '''</div>
                <div class="stat-label">Total Chunks</div>
            </div>
            <div class="stat">
                <div class="stat-value">''' + str(hop_count) + '''</div>
                <div class="stat-label">Hops</div>
            </div>
            <div class="stat">
                <div class="stat-value">''' + str(depth_reached) + '''</div>
                <div class="stat-label">Depth</div>
            </div>
            <div class="stat">
                <div class="stat-value">''' + str(len(all_keywords)) + '''</div>
                <div class="stat-label">Keywords</div>
            </div>
        </div>
'''
    
    # Initial Chunk Section
    initial_keywords = set(keywords_per_chunk.get('chunk_1', []))
    initial_content = initial_chunk.get('content', '')[:600]
    initial_highlighted = highlight_keywords_html(initial_content, initial_keywords, '#58a6ff')
    
    html += f'''
        <div class="section">
            <div class="section-title">📄 Initial Chunk</div>
            <div class="initial-chunk">
                <span class="initial-label">SEED CHUNK · {initial_chunk.get('file_name', 'unknown')}:{initial_chunk.get('chunk_id', '?')}</span>
                <div class="chunk-content">{initial_highlighted}</div>
            </div>
        </div>
        
        <div class="flow-arrow">↓</div>
'''
    
    # Retrieval Iterations Section
    html += '''
        <div class="section">
            <div class="section-title">🔄 Retrieval Iterations</div>
'''
    
    for log in iteration_logs:
        depth = log.get('depth', 0)
        status = log.get('status', 'UNKNOWN')
        explanation = log.get('explanation', '')
        search_strings = log.get('search_strings', [])
        retrieved_chunks = log.get('retrieved_chunks', [])
        
        status_class = 'status-complete' if status == 'COMPLETE' else 'status-incomplete'
        
        # Group retrieved chunks by query
        chunks_by_query = {}
        for rc in retrieved_chunks:
            query = rc.get('search_query', 'Unknown Query')
            if query not in chunks_by_query:
                chunks_by_query[query] = []
            chunks_by_query[query].append(rc)
        
        # Determine grid columns based on number of queries
        num_queries = len(search_strings) if search_strings else 1
        grid_cols = f'repeat({min(num_queries, 4)}, 1fr)'
        
        html += f'''
            <div class="depth-section">
                <div class="depth-header">
                    <span class="depth-badge">DEPTH {depth}</span>
                    <span class="depth-status {status_class}">{status}</span>
                </div>
                <p class="depth-explanation">{explanation}</p>
                <div class="queries-grid" style="grid-template-columns: {grid_cols};">
'''
        
        if search_strings:
            for q_idx, query in enumerate(search_strings):
                color = query_colors[q_idx % len(query_colors)]
                chunks_for_query = chunks_by_query.get(query, [])
                
                html += f'''
                    <div class="query-column">
                        <div class="query-header" style="background:{color};">🔍 Query {q_idx + 1}: {query[:50]}{'...' if len(query) > 50 else ''}</div>
                        <div class="query-chunks">
'''
                
                if chunks_for_query:
                    for chunk in chunks_for_query:
                        chunk_id = chunk.get('chunk_id', '?')
                        file_name = chunk.get('file_name', 'unknown')
                        verdict = chunk.get('verdict', 'RELATED')
                        reason = chunk.get('reason', '')[:100]
                        verdict_class = 'verdict-explanatory' if verdict == 'EXPLANATORY' else 'verdict-related'
                        
                        html += f'''
                            <div class="retrieved-chunk" style="border-color:{color};">
                                <div class="chunk-header">
                                    <span class="chunk-id">{file_name}:{chunk_id}</span>
                                    <span class="chunk-verdict {verdict_class}">{verdict}</span>
                                </div>
                                <div style="font-size:0.8rem;color:#8b949e;">{reason}...</div>
                            </div>
'''
                else:
                    html += '<div class="no-chunks">No chunks retrieved</div>'
                
                html += '''
                        </div>
                    </div>
'''
        else:
            html += '<div class="no-chunks">No queries at this depth</div>'
        
        html += '''
                </div>
            </div>
'''
    
    html += '''
        </div>
        
        <div class="flow-arrow">↓</div>
'''
    
    # Final Context Section
    html += '''
        <div class="section">
            <div class="section-title">📚 Final Context (All Chunks)</div>
            <div class="context-box">
                <div class="context-chunks-list">
'''
    
    for i, chunk in enumerate(context_chunks):
        chunk_id = chunk.get('chunk_id', f'{i+1}')
        file_name = chunk.get('file_name', 'unknown')
        classification = chunk.get('classification', 'INITIAL' if i == 0 else 'RELATED')
        html += f'<span class="context-chunk-tag">{file_name}:{chunk_id} ({classification})</span>\n'
    
    html += '''
                </div>
            </div>
        </div>
        
        <div class="flow-arrow">↓</div>
'''
    
    # Keyword Chain Section
    if related_keywords:
        html += '''
        <div class="section">
            <div class="section-title">🔗 Keyword Chain (Cross-Chunk Connections)</div>
            <div class="keyword-chain">
'''
        relationships = related_keywords.split(';')
        for i, rel in enumerate(relationships):
            rel = rel.strip()
            if rel:
                match = re.search(r'\[([^\]]+)\].*\[([^\]]+)\].*via\s+(.+)', rel)
                if match:
                    kw1, kw2, connection = match.groups()
                    color = query_colors[i % len(query_colors)]
                    html += f'''
                <span class="chain-kw" style="background:{color}33;color:{color};">{kw1}</span>
                <span class="chain-arrow">→</span>
                <span class="chain-kw" style="background:{color}33;color:{color};">{kw2}</span>
                <span class="chain-relation">({connection})</span>
'''
        html += '''
            </div>
        </div>
        
        <div class="flow-arrow">↓</div>
'''
    
    # Generated QA Section
    question_highlighted = highlight_keywords_html(question, all_keywords, '#58a6ff')
    answer_highlighted = highlight_keywords_html(answer, all_keywords, '#3fb950')
    
    html += f'''
        <div class="section">
            <div class="section-title">❓ Generated Question & Answer</div>
            <div class="qa-box">
                <div class="qa-label question-label">Question:</div>
                <div class="question-text">{question_highlighted}</div>
                <div class="qa-label answer-label">Answer:</div>
                <div class="answer-text">{answer_highlighted}</div>
            </div>
        </div>
        
        <div style="text-align:center;color:#8b949e;padding:2rem 0;font-size:0.85rem;">
            Generated by MiRAGe QA Pipeline | Keywords highlighted show concept linkage across chunks
        </div>
    </div>
</body>
</html>
'''
    
    if output_path:
        Path(output_path).write_text(html, encoding='utf-8')
        print(f"✅ Visualization saved to: {output_path}")
    
    return html

def main():
    parser = argparse.ArgumentParser(description='Visualize Multihop QA Pairs')
    parser.add_argument('--qa-file', type=str, default='output/qa_deduplicated.json',
                        help='Path to QA JSON file')
    parser.add_argument('--index', type=int, default=0,
                        help='Index of QA pair to visualize (default: 0 = first)')
    parser.add_argument('--output', type=str, default='output/multihop_visualization.html',
                        help='Output HTML file path')
    args = parser.parse_args()
    
    qa_file = Path(args.qa_file)
    if not qa_file.exists():
        print(f"❌ QA file not found: {qa_file}")
        return
    
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    if not qa_data:
        print("❌ No QA pairs found in file")
        return
    
    index = min(args.index, len(qa_data) - 1)
    qa_item = qa_data[index]
    
    print(f"📊 Visualizing QA pair {index + 1}/{len(qa_data)}")
    print(f"   Question: {qa_item.get('question', '')[:80]}...")
    print(f"   Chunks: {len(qa_item.get('context_chunks', []))}")
    print(f"   Hop count: {qa_item.get('hop_count', 0)}")
    
    generate_html_visualization(qa_item, args.output)

if __name__ == '__main__':
    main()
