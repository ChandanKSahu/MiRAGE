#!/usr/bin/env python3
"""
Multihop QA Visualization - Shows keyword chains linking chunks to QA pairs.

Flow: Context → Keywords per chunk → Related keywords → Keyword chain → QA

Usage:
    python visualize_multihop.py [--qa-file PATH] [--index N] [--output PATH]
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Set

def extract_keywords_from_text(text: str, keywords: Set[str]) -> List[tuple]:
    """Find all keyword occurrences in text with positions."""
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
    return sorted(matches, key=lambda x: x[0])

def highlight_keywords_html(text: str, keywords: Set[str], color: str = "#ffeb3b") -> str:
    """Highlight keywords in text with HTML spans."""
    matches = extract_keywords_from_text(text, keywords)
    if not matches:
        return text.replace('\n', '<br>')
    
    # Merge overlapping matches
    merged = []
    for start, end, kw in matches:
        if merged and start < merged[-1][1]:
            # Overlap - extend previous
            merged[-1] = (merged[-1][0], max(merged[-1][1], end), merged[-1][2])
        else:
            merged.append((start, end, kw))
    
    # Build highlighted text
    result = []
    last_end = 0
    for start, end, kw in merged:
        result.append(text[last_end:start].replace('\n', '<br>'))
        result.append(f'<span class="keyword" style="background:{color};padding:2px 4px;border-radius:3px;font-weight:600;">{text[start:end]}</span>')
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
    search_history = qa_item.get('search_history', [])
    iteration_logs = qa_item.get('iteration_logs', [])
    hop_count = qa_item.get('hop_count', 0)
    depth_reached = qa_item.get('depth_reached', 0)
    
    # Collect all keywords
    all_keywords = set()
    for chunk_kws in keywords_per_chunk.values():
        all_keywords.update(chunk_kws)
    
    # Color palette for chunks
    chunk_colors = ['#e3f2fd', '#fff3e0', '#e8f5e9', '#fce4ec', '#f3e5f5', '#e0f7fa']
    keyword_colors = ['#42a5f5', '#ff9800', '#66bb6a', '#ec407a', '#ab47bc', '#26c6da']
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multihop QA Visualization</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Space Grotesk', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e8e8e8;
            padding: 2rem;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #00d4ff, #7b2ff7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 2rem;
            font-size: 1.1rem;
        }}
        .stats {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-bottom: 2rem;
        }}
        .stat {{
            background: rgba(255,255,255,0.05);
            padding: 1rem 2rem;
            border-radius: 12px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .stat-value {{
            font-size: 2rem;
            font-weight: 700;
            color: #00d4ff;
        }}
        .stat-label {{ color: #888; font-size: 0.9rem; }}
        
        .section {{
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255,255,255,0.08);
        }}
        .section-title {{
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #00d4ff;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .section-title::before {{
            content: '';
            width: 4px;
            height: 20px;
            background: linear-gradient(180deg, #00d4ff, #7b2ff7);
            border-radius: 2px;
        }}
        
        .chunks-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
        }}
        .chunk {{
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            padding: 1rem;
            border-left: 4px solid;
        }}
        .chunk-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
        }}
        .chunk-id {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            padding: 4px 8px;
            border-radius: 6px;
            background: rgba(255,255,255,0.1);
        }}
        .chunk-content {{
            font-size: 0.9rem;
            line-height: 1.6;
            color: #ccc;
            max-height: 200px;
            overflow-y: auto;
        }}
        .chunk-keywords {{
            margin-top: 0.75rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}
        .keyword-tag {{
            font-size: 0.75rem;
            padding: 4px 10px;
            border-radius: 20px;
            font-weight: 500;
        }}
        
        .keyword-chain {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.5rem;
            padding: 1rem;
            background: rgba(0,0,0,0.2);
            border-radius: 12px;
        }}
        .chain-keyword {{
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.9rem;
        }}
        .chain-arrow {{
            color: #00d4ff;
            font-size: 1.5rem;
        }}
        
        .qa-box {{
            background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(123,47,247,0.1));
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid rgba(0,212,255,0.3);
        }}
        .question {{
            font-size: 1.1rem;
            line-height: 1.7;
            margin-bottom: 1rem;
        }}
        .question-label {{
            color: #00d4ff;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}
        .answer {{
            background: rgba(0,0,0,0.3);
            padding: 1rem;
            border-radius: 8px;
            line-height: 1.7;
        }}
        .answer-label {{
            color: #66bb6a;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}
        
        .search-history {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
        }}
        .search-query {{
            background: rgba(255,255,255,0.1);
            padding: 8px 16px;
            border-radius: 20px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        
        .iteration {{
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.75rem;
        }}
        .iteration-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }}
        .iteration-depth {{
            font-weight: 600;
            color: #ab47bc;
        }}
        .iteration-status {{
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .status-complete {{ background: rgba(102,187,106,0.3); color: #66bb6a; }}
        .status-incomplete {{ background: rgba(255,152,0,0.3); color: #ff9800; }}
        
        .keyword {{ font-weight: 600; }}
        
        @media (max-width: 768px) {{
            .chunks-container {{ grid-template-columns: 1fr; }}
            .stats {{ flex-wrap: wrap; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔗 Multihop QA Visualization</h1>
        <p class="subtitle">Context → Keywords → Keyword Chain → QA Generation</p>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{len(context_chunks)}</div>
                <div class="stat-label">Chunks Used</div>
            </div>
            <div class="stat">
                <div class="stat-value">{hop_count}</div>
                <div class="stat-label">Hops</div>
            </div>
            <div class="stat">
                <div class="stat-value">{depth_reached}</div>
                <div class="stat-label">Depth Reached</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(all_keywords)}</div>
                <div class="stat-label">Keywords</div>
            </div>
        </div>
'''
    
    # Search History Section
    if search_history:
        html += '''
        <div class="section">
            <div class="section-title">🔍 Search Queries (Retrieval Process)</div>
            <div class="search-history">
'''
        for query in search_history:
            html += f'                <div class="search-query">{query}</div>\n'
        html += '''            </div>
        </div>
'''
    
    # Chunks Section
    html += '''
        <div class="section">
            <div class="section-title">📄 Context Chunks with Keywords</div>
            <div class="chunks-container">
'''
    
    for i, chunk in enumerate(context_chunks):
        color = chunk_colors[i % len(chunk_colors)]
        kw_color = keyword_colors[i % len(keyword_colors)]
        chunk_id = chunk.get('chunk_id', f'chunk_{i+1}')
        file_name = chunk.get('file_name', 'unknown')
        content = chunk.get('content', '')[:500]  # Truncate for display
        classification = chunk.get('classification', 'INITIAL')
        
        # Get keywords for this chunk
        chunk_key = f'chunk_{i+1}'
        chunk_keywords = set(keywords_per_chunk.get(chunk_key, []))
        
        highlighted_content = highlight_keywords_html(content, chunk_keywords, kw_color)
        
        html += f'''
                <div class="chunk" style="border-color: {kw_color};">
                    <div class="chunk-header">
                        <span class="chunk-id">Chunk {chunk_id}</span>
                        <span style="font-size:0.8rem;color:#888;">{classification}</span>
                    </div>
                    <div class="chunk-content">{highlighted_content}</div>
                    <div class="chunk-keywords">
'''
        for kw in list(chunk_keywords)[:6]:  # Show top 6 keywords
            html += f'                        <span class="keyword-tag" style="background:{kw_color}33;color:{kw_color};">{kw}</span>\n'
        html += '''                    </div>
                </div>
'''
    
    html += '''            </div>
        </div>
'''
    
    # Keyword Chain Section
    if related_keywords:
        html += '''
        <div class="section">
            <div class="section-title">🔗 Keyword Relationships (Chain)</div>
            <div class="keyword-chain">
'''
        # Parse the related_keywords string
        relationships = related_keywords.split(';')
        for i, rel in enumerate(relationships):
            rel = rel.strip()
            if rel:
                # Extract keywords from format: [keyword1] relates to [keyword2] via connection
                match = re.search(r'\[([^\]]+)\].*\[([^\]]+)\].*via\s+(.+)', rel)
                if match:
                    kw1, kw2, connection = match.groups()
                    color = keyword_colors[i % len(keyword_colors)]
                    html += f'''                <span class="chain-keyword" style="background:{color}33;color:{color};">{kw1}</span>
                <span class="chain-arrow">→</span>
                <span class="chain-keyword" style="background:{color}33;color:{color};">{kw2}</span>
                <span style="color:#888;font-size:0.85rem;margin:0 1rem;">({connection})</span>
'''
        html += '''            </div>
        </div>
'''
    
    # Iteration Logs Section
    if iteration_logs:
        html += '''
        <div class="section">
            <div class="section-title">📊 Retrieval Iterations</div>
'''
        for log in iteration_logs:
            depth = log.get('depth', 0)
            status = log.get('status', 'UNKNOWN')
            explanation = log.get('explanation', '')
            chunks_added = log.get('chunks_added_this_iteration', [])
            status_class = 'status-complete' if status == 'COMPLETE' else 'status-incomplete'
            
            html += f'''
            <div class="iteration">
                <div class="iteration-header">
                    <span class="iteration-depth">Depth {depth}</span>
                    <span class="iteration-status {status_class}">{status}</span>
                </div>
                <p style="color:#aaa;font-size:0.9rem;">{explanation}</p>
                <p style="color:#888;font-size:0.85rem;margin-top:0.5rem;">Chunks added: {len(chunks_added)}</p>
            </div>
'''
        html += '''        </div>
'''
    
    # QA Section with highlighted keywords
    html += '''
        <div class="section">
            <div class="section-title">❓ Generated Question & Answer</div>
            <div class="qa-box">
                <div class="question-label">Question:</div>
                <div class="question">
'''
    html += highlight_keywords_html(question, all_keywords, '#00d4ff')
    html += '''
                </div>
                <div class="answer-label">Answer:</div>
                <div class="answer">
'''
    html += highlight_keywords_html(answer, all_keywords, '#66bb6a')
    html += '''
                </div>
            </div>
        </div>
        
        <div style="text-align:center;color:#666;padding:2rem 0;font-size:0.9rem;">
            Generated by Multihop QA Pipeline | Keywords highlighted show concept linkage across chunks
        </div>
    </div>
</body>
</html>
'''
    
    # Write to file if output path provided
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
    
    # Load QA data
    qa_file = Path(args.qa_file)
    if not qa_file.exists():
        print(f"❌ QA file not found: {qa_file}")
        return
    
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    if not qa_data:
        print("❌ No QA pairs found in file")
        return
    
    # Get specified QA pair
    index = min(args.index, len(qa_data) - 1)
    qa_item = qa_data[index]
    
    print(f"📊 Visualizing QA pair {index + 1}/{len(qa_data)}")
    print(f"   Question: {qa_item.get('question', '')[:80]}...")
    print(f"   Chunks: {len(qa_item.get('context_chunks', []))}")
    print(f"   Hop count: {qa_item.get('hop_count', 0)}")
    
    # Generate visualization
    generate_html_visualization(qa_item, args.output)

if __name__ == '__main__':
    main()
