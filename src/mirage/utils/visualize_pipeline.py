"""
Multihop QA Generation Pipeline Visualization

Visualizes the complete pipeline:
1. Source Chunk -> generates >=1 Queries
2. Each Query -> retrieves >=1 Chunks  
3. All chunks -> form Context
4. Each chunk in context -> extracts >=1 Keywords
5. Keywords -> form >=1 Bridge/Keyword Chains linking chunks
6. Context + Bridges -> generate >=1 QA pairs
7. (Optional) Similar QAs -> Rank -> Merge -> Deduplicated QA
"""

import plotly.graph_objects as go
from dataclasses import dataclass, field

# =============================================================================
# Helper Functions
# =============================================================================

def truncate(text: str, max_len: int = 40) -> str:
    """Truncate text with ellipsis."""
    text = text.replace('\n', ' ').strip()
    return text[:max_len] + "..." if len(text) > max_len else text


def wrap_text(text: str, width: int = 50) -> str:
    """Wrap text for hover display."""
    words = text.split()
    lines, current = [], []
    length = 0
    for word in words:
        if length + len(word) + 1 > width:
            lines.append(' '.join(current))
            current, length = [word], len(word)
        else:
            current.append(word)
            length += len(word) + 1
    if current:
        lines.append(' '.join(current))
    return '<br>'.join(lines)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Node:
    id: str
    type: str
    label: str  # Short label for display
    full_text: str  # Full text for hover
    x: float = 0
    y: float = 0
    color: str = "#888888"
    size: int = 30
    
@dataclass  
class Edge:
    source: str
    target: str
    label: str = ""
    color: str = "rgba(150,150,150,0.5)"
    width: float = 1.5


# =============================================================================
# Example Data - Food Safety Officer (from prompts.py)
# =============================================================================

def build_example_graph():
    """Build nodes and edges for the multihop example."""
    nodes = []
    edges = []
    
    # Color scheme
    COLORS = {
        "source_chunk": "#2ECC71",      # Green
        "query": "#F1C40F",              # Yellow
        "retrieved_chunk": "#3498DB",   # Blue
        "context": "#1ABC9C",            # Teal
        "keyword": "#9B59B6",            # Purple
        "bridge": "#E74C3C",             # Red
        "qa": "#E67E22",                 # Orange
        "similar_qa": "#95A5A6",         # Gray
        "dedup_qa": "#27AE60",           # Dark green
    }
    
    # === Layer 0: SOURCE CHUNK ===
    source_chunk_text = (
        '[Chunk 1: Bistro Menu Description] The "Autumn Harvest Risotto" is a '
        'creamy dish featuring Arborio rice, butternut squash, aged parmesan cheese, '
        'and is slow-cooked in a house-made chicken broth.'
    )
    nodes.append(Node(
        id="src_chunk",
        type="source_chunk",
        label=truncate(source_chunk_text, 35),
        full_text=source_chunk_text,
        x=0, y=0,
        color=COLORS["source_chunk"],
        size=45
    ))
    
    # === Layer 1: QUERIES (>=1 per chunk) ===
    queries = [
        '"vegetarian certification" "label requirements"',
        '"animal-derived ingredients" restrictions',
        '"Green-Leaf" certification criteria'
    ]
    for i, q in enumerate(queries):
        y_pos = (i - len(queries)/2 + 0.5) * 1.2
        nodes.append(Node(
            id=f"query_{i}",
            type="query",
            label=truncate(q, 30),
            full_text=f"Search Query: {q}",
            x=1.5, y=y_pos,
            color=COLORS["query"],
            size=32
        ))
        edges.append(Edge("src_chunk", f"query_{i}", "generates", COLORS["query"], 2))
    
    # === Layer 2: RETRIEVED CHUNKS (>=1 per query) ===
    retrieved_chunks = [
        {
            "id": "ret_0",
            "text": '[Green-Leaf Certification Guide] To qualify for the "Green-Leaf Vegetarian Label," '
                   'a dish must be entirely free of meat, poultry, and seafood flesh, including any '
                   'stocks, broths, or gravies derived from animal tissue. Dairy and eggs are permitted.',
            "status": "EXPLANATORY",
            "from_query": 0
        },
        {
            "id": "ret_1", 
            "text": '[Animal Products Definition] Animal-derived ingredients include any substance '
                   'obtained from animal tissue, including fats, stocks, broths, and gelatin. '
                   'Eggs and dairy are classified separately.',
            "status": "RELATED",
            "from_query": 1
        },
        {
            "id": "ret_2",
            "text": '[Certification Standards Overview] The Green-Leaf program certifies dishes as '
                   'vegetarian-compliant based on ingredient sourcing. Certified dishes display '
                   'the Green-Leaf logo on menus.',
            "status": "RELATED", 
            "from_query": 2
        }
    ]
    
    ret_y_positions = {}
    for i, chunk in enumerate(retrieved_chunks):
        y_pos = (i - len(retrieved_chunks)/2 + 0.5) * 1.4
        ret_y_positions[chunk["id"]] = y_pos
        status_tag = f"[{chunk['status']}]"
        nodes.append(Node(
            id=chunk["id"],
            type="retrieved_chunk",
            label=truncate(chunk["text"], 32),
            full_text=f"{status_tag}\n{chunk['text']}",
            x=3, y=y_pos,
            color=COLORS["retrieved_chunk"],
            size=38
        ))
        edges.append(Edge(f"query_{chunk['from_query']}", chunk["id"], "retrieves", COLORS["retrieved_chunk"], 1.5))
    
    # === Layer 3: CONTEXT (all chunks combined) ===
    context_chunks = [
        {"id": "ctx_0", "source": "src_chunk", "text": source_chunk_text, "title": "Bistro Menu"},
        {"id": "ctx_1", "source": "ret_0", "text": retrieved_chunks[0]["text"], "title": "Green-Leaf Guide"},
        {"id": "ctx_2", "source": "ret_1", "text": retrieved_chunks[1]["text"], "title": "Animal Products Def"},
    ]
    
    ctx_y_positions = {}
    for i, ctx in enumerate(context_chunks):
        y_pos = (i - len(context_chunks)/2 + 0.5) * 1.5
        ctx_y_positions[ctx["id"]] = y_pos
        nodes.append(Node(
            id=ctx["id"],
            type="context",
            label=f"Context: {ctx['title'][:15]}...",
            full_text=ctx["text"],
            x=4.8, y=y_pos,
            color=COLORS["context"],
            size=40
        ))
        edges.append(Edge(ctx["source"], ctx["id"], "forms context", COLORS["context"], 2))
    
    # Also connect ret_1 to ctx_2
    edges.append(Edge("ret_1", "ctx_2", "forms context", COLORS["context"], 2))
    
    # === Layer 4: KEYWORDS (>=1 per context chunk) ===
    keywords_map = {
        "ctx_0": [
            ("kw_0_0", "Autumn Harvest Risotto"),
            ("kw_0_1", "chicken broth"),
            ("kw_0_2", "parmesan cheese"),
        ],
        "ctx_1": [
            ("kw_1_0", "Green-Leaf Vegetarian Label"),
            ("kw_1_1", "free of meat/poultry"),
            ("kw_1_2", "stocks/broths from animal tissue"),
        ],
        "ctx_2": [
            ("kw_2_0", "animal-derived ingredients"),
            ("kw_2_1", "fats, stocks, broths"),
        ],
    }
    
    kw_idx = 0
    total_kw = sum(len(v) for v in keywords_map.values())
    for ctx_id, kws in keywords_map.items():
        for kw_id, kw_text in kws:
            y_pos = (kw_idx - total_kw/2 + 0.5) * 0.7
            nodes.append(Node(
                id=kw_id,
                type="keyword",
                label=truncate(kw_text, 22),
                full_text=f"Keyword: {kw_text}\nFrom: {ctx_id}",
                x=6.3, y=y_pos,
                color=COLORS["keyword"],
                size=28
            ))
            edges.append(Edge(ctx_id, kw_id, "extracts", COLORS["keyword"], 1))
            kw_idx += 1
    
    # === Layer 5: BRIDGE KEYWORDS (keyword chains linking chunks) ===
    bridges = [
        {
            "id": "bridge_0",
            "kw1": "kw_0_1",  # chicken broth
            "kw2": "kw_1_2",  # stocks/broths from animal tissue
            "relation": "poultry origin -> animal tissue",
            "description": '"chicken broth" <-> "stocks/broths derived from animal tissue" via poultry origin'
        },
        {
            "id": "bridge_1",
            "kw1": "kw_1_2",  # stocks/broths from animal tissue
            "kw2": "kw_2_1",  # fats, stocks, broths  
            "relation": "category match",
            "description": '"stocks/broths from animal tissue" <-> "fats, stocks, broths" via ingredient category'
        }
    ]
    
    for i, bridge in enumerate(bridges):
        y_pos = (i - len(bridges)/2 + 0.5) * 1.8
        nodes.append(Node(
            id=bridge["id"],
            type="bridge",
            label=f"Link: {truncate(bridge['relation'], 20)}",
            full_text=f"Bridge Keyword Chain:\n{bridge['description']}",
            x=7.8, y=y_pos,
            color=COLORS["bridge"],
            size=36
        ))
        edges.append(Edge(bridge["kw1"], bridge["id"], "links", COLORS["bridge"], 2.5))
        edges.append(Edge(bridge["kw2"], bridge["id"], "links", COLORS["bridge"], 2.5))
    
    # === Layer 6: GENERATED QA PAIRS (>=1) ===
    qa_pairs = [
        {
            "id": "qa_0",
            "question": 'According to the Green-Leaf Certification Guide, why does the "Autumn Harvest Risotto" fail to qualify for the "Green-Leaf Vegetarian Label"?',
            "answer": 'The "Autumn Harvest Risotto" fails to qualify because it is cooked in chicken broth, and the Green-Leaf Certification Guide explicitly excludes dishes containing stocks or broths derived from animal tissue (poultry).',
            "relevance": 10,
            "difficulty": 3
        },
        {
            "id": "qa_1",
            "question": 'What specific ingredient in the "Autumn Harvest Risotto" violates the animal-derived ingredient restrictions defined for Green-Leaf certification?',
            "answer": 'The house-made chicken broth used in the risotto violates the restrictions, as it constitutes a stock/broth derived from animal tissue (poultry), which is explicitly prohibited for Green-Leaf Vegetarian Label certification.',
            "relevance": 9,
            "difficulty": 4
        }
    ]
    
    for i, qa in enumerate(qa_pairs):
        y_pos = (i - len(qa_pairs)/2 + 0.5) * 2.0
        nodes.append(Node(
            id=qa["id"],
            type="qa",
            label=f"QA: {truncate(qa['question'], 25)}",
            full_text=f"Q: {qa['question']}\n\nA: {qa['answer']}\n\nRelevance: {qa['relevance']}/10 | Difficulty: {qa['difficulty']}/10",
            x=9.3, y=y_pos,
            color=COLORS["qa"],
            size=42
        ))
        # Connect from bridges and context
        for bridge in bridges:
            edges.append(Edge(bridge["id"], qa["id"], "enables", COLORS["qa"], 2))
    
    # === Layer 7a: SIMILAR QA PAIRS (for deduplication demo) ===
    similar_qas = [
        {
            "id": "sim_qa_0",
            "question": 'Why can\'t the Autumn Harvest Risotto get vegetarian certification?',
            "answer": 'It contains chicken broth which is animal-derived.',
        },
        {
            "id": "sim_qa_1", 
            "question": 'What makes the risotto ineligible for the Green-Leaf label?',
            "answer": 'The chicken broth used as cooking base violates vegetarian requirements.',
        },
        {
            "id": "sim_qa_2",
            "question": 'Does the Autumn Harvest Risotto qualify for Green-Leaf certification? Why or why not?',
            "answer": 'No, because the dish is slow-cooked in chicken broth, an animal-derived ingredient.',
        }
    ]
    
    for i, qa in enumerate(similar_qas):
        y_pos = (i - len(similar_qas)/2 + 0.5) * 1.3
        nodes.append(Node(
            id=qa["id"],
            type="similar_qa",
            label=f"Similar: {truncate(qa['question'], 22)}",
            full_text=f"[Similar QA - Pre-deduplication]\nQ: {qa['question']}\nA: {qa['answer']}",
            x=11.0, y=y_pos,
            color=COLORS["similar_qa"],
            size=32
        ))
        edges.append(Edge("qa_0", qa["id"], "similar to", "rgba(150,150,150,0.4)", 1))
    
    # === Layer 7b: DEDUPLICATED QA ===
    dedup_qa = {
        "id": "dedup_qa",
        "question": 'According to the Green-Leaf Certification Guide, why does the "Autumn Harvest Risotto" fail to qualify for the "Green-Leaf Vegetarian Label", and what specific ingredient violates the certification requirements?',
        "answer": 'The "Autumn Harvest Risotto" fails to qualify because it is slow-cooked in house-made chicken broth. The Green-Leaf Certification Guide explicitly excludes dishes containing stocks, broths, or gravies derived from animal tissue. Since chicken broth is derived from poultry (animal tissue), the dish cannot receive vegetarian certification despite other ingredients like parmesan cheese and eggs being permitted.',
    }
    
    nodes.append(Node(
        id=dedup_qa["id"],
        type="dedup_qa",
        label=f"[OK] Dedup: {truncate(dedup_qa['question'], 25)}",
        full_text=f"[DEDUPLICATED & REFINED QA]\n\nQ: {dedup_qa['question']}\n\nA: {dedup_qa['answer']}",
        x=12.8, y=0,
        color=COLORS["dedup_qa"],
        size=48
    ))
    
    # Connect similar QAs to dedup
    for qa in similar_qas:
        edges.append(Edge(qa["id"], dedup_qa["id"], "merged into", COLORS["dedup_qa"], 2))
    edges.append(Edge("qa_0", dedup_qa["id"], "merged into", COLORS["dedup_qa"], 2))
    
    return nodes, edges, COLORS


# =============================================================================
# Plotly Visualization
# =============================================================================

def create_graph_visualization():
    """Create interactive network graph visualization."""
    
    nodes, edges, COLORS = build_example_graph()
    
    # Build edge traces
    edge_traces = []
    for edge in edges:
        src = next((n for n in nodes if n.id == edge.source), None)
        tgt = next((n for n in nodes if n.id == edge.target), None)
        if src and tgt:
            # Create curved edges for better visibility
            mid_x = (src.x + tgt.x) / 2
            mid_y = (src.y + tgt.y) / 2 + 0.1 * (src.x - tgt.x)  # Slight curve
            
            edge_traces.append(go.Scatter(
                x=[src.x, mid_x, tgt.x],
                y=[src.y, mid_y, tgt.y],
                mode='lines',
                line=dict(width=edge.width, color=edge.color),
                hoverinfo='text',
                hovertext=edge.label,
                showlegend=False
            ))
    
    # Build node trace
    node_x = [n.x for n in nodes]
    node_y = [n.y for n in nodes]
    node_colors = [n.color for n in nodes]
    node_sizes = [n.size for n in nodes]
    node_labels = [n.label for n in nodes]
    node_hovers = [
        f"<b>{n.type.upper().replace('_', ' ')}</b><br><br>{wrap_text(n.full_text, 60)}"
        for n in nodes
    ]
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='#2C3E50'),
            opacity=0.95
        ),
        text=node_labels,
        textposition="bottom center",
        textfont=dict(size=9, color="#2C3E50", family="Inter, Segoe UI, sans-serif"),
        hovertext=node_hovers,
        hoverinfo='text',
        hoverlabel=dict(
            bgcolor="white",
            font_size=11,
            font_family="Inter, Segoe UI, sans-serif"
        ),
        showlegend=False
    )
    
    # Column headers
    columns = [
        (0, "Source\nChunk"),
        (1.5, "Queries\n(>=1)"),
        (3, "Retrieved\nChunks (>=1)"),
        (4.8, "Context"),
        (6.3, "Keywords\n(>=1 per chunk)"),
        (7.8, "Keyword\nChains (>=1)"),
        (9.3, "Generated\nQA (>=1)"),
        (11.0, "Similar\nQAs"),
        (12.8, "Deduplicated\nQA")
    ]
    
    header_trace = go.Scatter(
        x=[c[0] for c in columns],
        y=[3.5] * len(columns),
        mode='text',
        text=[c[1] for c in columns],
        textfont=dict(size=11, color="#34495E", family="Inter, Segoe UI, sans-serif"),
        hoverinfo='none',
        showlegend=False
    )
    
    # Combine traces
    fig = go.Figure(data=edge_traces + [node_trace, header_trace])
    
    # Layout with legend
    fig.update_layout(
        title=dict(
            text="<b>Multihop QA Generation Pipeline</b>",
            x=0.5,
            font=dict(size=22, color="#2C3E50", family="Inter, Segoe UI, sans-serif")
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=60, l=40, r=40, t=80),
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-0.8, 14]
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-4, 4.5]
        ),
        paper_bgcolor='#ECF0F1',
        plot_bgcolor='#ECF0F1',
        font=dict(family="Inter, Segoe UI, sans-serif"),
    )
    
    # Add stage annotations with arrows
    stage_annotations = [
        dict(x=0.75, y=-3.5, text="1. Analyze chunk<br>for completeness", showarrow=False, font=dict(size=9, color="#7F8C8D")),
        dict(x=2.25, y=-3.5, text="2. Generate search<br>queries if incomplete", showarrow=False, font=dict(size=9, color="#7F8C8D")),
        dict(x=3.9, y=-3.5, text="3. Retrieve &<br>classify chunks", showarrow=False, font=dict(size=9, color="#7F8C8D")),
        dict(x=5.55, y=-3.5, text="4. Build unified<br>context", showarrow=False, font=dict(size=9, color="#7F8C8D")),
        dict(x=7.05, y=-3.5, text="5. Extract keywords<br>& build chains", showarrow=False, font=dict(size=9, color="#7F8C8D")),
        dict(x=8.55, y=-3.5, text="6. Generate<br>multi-hop QA", showarrow=False, font=dict(size=9, color="#7F8C8D")),
        dict(x=11.9, y=-3.5, text="7. Deduplicate<br>similar QAs", showarrow=False, font=dict(size=9, color="#7F8C8D")),
    ]
    
    for ann in stage_annotations:
        fig.add_annotation(**ann)
    
    # Add legend as colored boxes
    legend_items = [
        ("Source Chunk", COLORS["source_chunk"]),
        ("Query", COLORS["query"]),
        ("Retrieved Chunk", COLORS["retrieved_chunk"]),
        ("Context", COLORS["context"]),
        ("Keyword", COLORS["keyword"]),
        ("Bridge/Chain", COLORS["bridge"]),
        ("Generated QA", COLORS["qa"]),
        ("Similar QA", COLORS["similar_qa"]),
        ("Dedup QA", COLORS["dedup_qa"]),
    ]
    
    for i, (label, color) in enumerate(legend_items):
        x_pos = 0.02 + (i % 5) * 0.19
        y_pos = 0.02 if i < 5 else 0.06
        fig.add_shape(
            type="circle",
            x0=x_pos, x1=x_pos + 0.015,
            y0=y_pos, y1=y_pos + 0.025,
            xref="paper", yref="paper",
            fillcolor=color,
            line=dict(color="#2C3E50", width=1)
        )
        fig.add_annotation(
            x=x_pos + 0.02, y=y_pos + 0.012,
            xref="paper", yref="paper",
            text=label,
            showarrow=False,
            font=dict(size=9, color="#2C3E50"),
            xanchor="left"
        )
    
    return fig


def create_detailed_html_page():
    """Create a comprehensive HTML dashboard."""
    
    fig = create_graph_visualization()
    fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multihop QA Pipeline Visualization</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #E0E0E0;
            padding: 20px;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        header {{
            text-align: center;
            padding: 30px 0;
        }}
        header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }}
        header p {{
            color: #B0B0B0;
            font-size: 1.1rem;
        }}
        .pipeline-flow {{
            display: flex;
            justify-content: center;
            gap: 8px;
            padding: 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .flow-step {{
            background: rgba(255,255,255,0.1);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.85rem;
            color: #00d4ff;
            font-weight: 500;
        }}
        .flow-arrow {{
            color: #666;
            font-size: 1.2rem;
        }}
        .card {{
            background: rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 24px;
            margin: 20px 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .card h2 {{
            color: #00d4ff;
            font-size: 1.3rem;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .graph-container {{
            background: white;
            border-radius: 12px;
            overflow: hidden;
            min-height: 600px;
        }}
        .example-section {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 16px;
        }}
        .chunk-box {{
            background: rgba(46, 204, 113, 0.15);
            border-left: 4px solid #2ECC71;
            padding: 14px;
            border-radius: 8px;
            font-size: 0.9rem;
            line-height: 1.5;
        }}
        .chunk-box.retrieved {{
            background: rgba(52, 152, 219, 0.15);
            border-left-color: #3498DB;
        }}
        .chunk-box .label {{
            font-weight: 600;
            color: #2ECC71;
            margin-bottom: 6px;
        }}
        .chunk-box.retrieved .label {{
            color: #3498DB;
        }}
        .keyword-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }}
        .keyword {{
            background: rgba(155, 89, 182, 0.2);
            color: #BB8FCE;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 0.8rem;
        }}
        .bridge-box {{
            background: rgba(231, 76, 60, 0.15);
            border: 2px dashed #E74C3C;
            padding: 14px;
            border-radius: 8px;
            text-align: center;
        }}
        .bridge-box .relation {{
            color: #E74C3C;
            font-weight: 600;
        }}
        .qa-box {{
            background: rgba(230, 126, 34, 0.15);
            border-left: 4px solid #E67E22;
            padding: 16px;
            border-radius: 8px;
        }}
        .qa-box .question {{
            color: #F5B041;
            font-weight: 600;
            margin-bottom: 10px;
        }}
        .qa-box .answer {{
            color: #D5D5D5;
            line-height: 1.6;
        }}
        .dedup-box {{
            background: rgba(39, 174, 96, 0.15);
            border: 2px solid #27AE60;
            padding: 16px;
            border-radius: 8px;
        }}
        .dedup-box h4 {{
            color: #27AE60;
            margin-bottom: 10px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
            margin-top: 16px;
        }}
        .stat {{
            background: rgba(255,255,255,0.05);
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat .num {{
            font-size: 1.8rem;
            font-weight: 700;
            color: #00d4ff;
        }}
        .stat .label {{
            font-size: 0.75rem;
            color: #888;
            text-transform: uppercase;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Multihop QA Generation Pipeline</h1>
            <p>Visualizing the reasoning chain from source chunks to synthesized question-answer pairs</p>
        </header>
        
        <div class="pipeline-flow">
            <span class="flow-step">Source Chunk</span>
            <span class="flow-arrow">-></span>
            <span class="flow-step">Queries (>=1)</span>
            <span class="flow-arrow">-></span>
            <span class="flow-step">Retrieved Chunks (>=1)</span>
            <span class="flow-arrow">-></span>
            <span class="flow-step">Context</span>
            <span class="flow-arrow">-></span>
            <span class="flow-step">Keywords (>=1)</span>
            <span class="flow-arrow">-></span>
            <span class="flow-step">Keyword Chains (>=1)</span>
            <span class="flow-arrow">-></span>
            <span class="flow-step">Generated QA (>=1)</span>
            <span class="flow-arrow">-></span>
            <span class="flow-step">Deduplication</span>
        </div>
        
        <div class="card">
            <h2>Interactive Pipeline Graph</h2>
            <p style="color:#888; margin-bottom:16px; font-size:0.9rem;">Hover over nodes to see full text content. Zoom and pan to explore.</p>
            <div class="graph-container">
                {fig_html}
            </div>
        </div>
        
        <div class="card">
            <h2>Example: Food Safety Officer - Vegetarian Certification Audit</h2>
            
            <div class="stats">
                <div class="stat"><div class="num">1</div><div class="label">Source Chunk</div></div>
                <div class="stat"><div class="num">3</div><div class="label">Queries</div></div>
                <div class="stat"><div class="num">3</div><div class="label">Retrieved</div></div>
                <div class="stat"><div class="num">8</div><div class="label">Keywords</div></div>
                <div class="stat"><div class="num">2</div><div class="label">Bridges</div></div>
                <div class="stat"><div class="num">2</div><div class="label">QA Pairs</div></div>
            </div>
            
            <div class="example-section" style="margin-top: 20px;">
                <div class="chunk-box">
                    <div class="label">Source Chunk (Bistro Menu)</div>
                    The "Autumn Harvest Risotto" is a creamy dish featuring Arborio rice, butternut squash, aged parmesan cheese, and is slow-cooked in a <strong>house-made chicken broth</strong>.
                    <div class="keyword-list">
                        <span class="keyword">Autumn Harvest Risotto</span>
                        <span class="keyword">chicken broth</span>
                        <span class="keyword">parmesan cheese</span>
                    </div>
                </div>
                
                <div class="chunk-box retrieved">
                    <div class="label">Retrieved Chunk [EXPLANATORY]</div>
                    To qualify for the "Green-Leaf Vegetarian Label," a dish must be entirely free of meat, poultry, and seafood flesh, including any <strong>stocks, broths, or gravies derived from animal tissue</strong>. Dairy and eggs are permitted.
                    <div class="keyword-list">
                        <span class="keyword">Green-Leaf Vegetarian Label</span>
                        <span class="keyword">free of meat/poultry</span>
                        <span class="keyword">stocks/broths from animal tissue</span>
                    </div>
                </div>
            </div>
            
            <div class="bridge-box" style="margin-top: 16px;">
                <div class="relation">Bridge Keyword Chain</div>
                <p style="margin-top:8px;">"chicken broth" <-> "stocks/broths derived from animal tissue" <br><em>via poultry origin</em></p>
            </div>
            
            <div class="qa-box" style="margin-top: 16px;">
                <div class="question">Q: According to the Green-Leaf Certification Guide, why does the "Autumn Harvest Risotto" fail to qualify for the "Green-Leaf Vegetarian Label"?</div>
                <div class="answer"><strong>A:</strong> The "Autumn Harvest Risotto" fails to qualify because it is cooked in chicken broth, and the Green-Leaf Certification Guide explicitly excludes dishes containing stocks or broths derived from animal tissue (poultry).</div>
            </div>
            
            <div class="dedup-box" style="margin-top: 16px;">
                <h4>[OK] Deduplicated & Refined QA</h4>
                <p><strong>Q:</strong> According to the Green-Leaf Certification Guide, why does the "Autumn Harvest Risotto" fail to qualify for the "Green-Leaf Vegetarian Label", and what specific ingredient violates the certification requirements?</p>
                <p style="margin-top:10px;"><strong>A:</strong> The "Autumn Harvest Risotto" fails to qualify because it is slow-cooked in house-made chicken broth. The Green-Leaf Certification Guide explicitly excludes dishes containing stocks, broths, or gravies derived from animal tissue. Since chicken broth is derived from poultry (animal tissue), the dish cannot receive vegetarian certification despite other ingredients like parmesan cheese and eggs being permitted.</p>
            </div>
        </div>
        
        <div class="card">
            <h2>Pipeline Stages Explained</h2>
            <ol style="line-height: 2; color: #CCC; padding-left: 20px;">
                <li><strong style="color:#2ECC71;">Source Chunk Analysis:</strong> Evaluate chunk for semantic completeness and extract concepts</li>
                <li><strong style="color:#F1C40F;">Query Generation:</strong> If incomplete, generate >=1 search queries to retrieve missing information</li>
                <li><strong style="color:#3498DB;">Chunk Retrieval:</strong> Each query retrieves >=1 chunks, classified as EXPLANATORY/RELATED/UNRELATED</li>
                <li><strong style="color:#1ABC9C;">Context Building:</strong> Combine source chunk with relevant retrieved chunks</li>
                <li><strong style="color:#9B59B6;">Keyword Extraction:</strong> Extract >=1 keywords from each chunk in context</li>
                <li><strong style="color:#E74C3C;">Bridge Formation:</strong> Identify keyword chains that link concepts across chunks</li>
                <li><strong style="color:#E67E22;">QA Generation:</strong> Synthesize >=1 multi-hop QA pairs requiring information from multiple chunks</li>
                <li><strong style="color:#27AE60;">Deduplication:</strong> Rank similar QAs, merge redundant ones, produce refined final QA</li>
            </ol>
        </div>
    </div>
</body>
</html>
"""
    return html_content


def main():
    """Generate visualizations."""
    
    # Create main graph visualization
    fig = create_graph_visualization()
    fig.write_html("multihop_pipeline_graph.html")
    print("[OK] Saved: multihop_pipeline_graph.html")
    
    # Create comprehensive dashboard
    html_content = create_detailed_html_page()
    with open("multihop_pipeline_dashboard.html", "w") as f:
        f.write(html_content)
    print("[OK] Saved: multihop_pipeline_dashboard.html")
    
    print("\n[OK] Visualizations generated successfully!")
    print("   Open multihop_pipeline_dashboard.html for the complete experience.")
    print("   Hover over nodes to see full content in popup.")


if __name__ == "__main__":
    main()
