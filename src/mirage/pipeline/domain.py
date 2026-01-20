"""
Module to extract domain and expert role from semantic chunks using BERTopic and LLM.
"""

import json
import os
import sys
import logging
from typing import List, Dict, Tuple, Optional, Union

import torch
import pandas as pd
import numpy as np
from umap import UMAP
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image
from sentence_transformers import SentenceTransformer
import datamapplot

# Import from mirage modules
from mirage.core.llm import call_llm_simple, setup_logging
from mirage.embeddings.models import NomicVLEmbed
from mirage.core.prompts import PROMPTS

#%% Setup

def save_domain_expert_to_env(domain: str, expert_persona: str):
    """Save domain and expert persona as environment variables"""
    os.environ['DATASET_DOMAIN'] = domain
    os.environ['DATASET_EXPERT_PERSONA'] = expert_persona
    print(f"💾 Saved to environment: DATASET_DOMAIN={domain}, DATASET_EXPERT_PERSONA={expert_persona}")

def load_domain_expert_from_env() -> Tuple[str, str]:
    """Load domain and expert persona from environment variables"""
    domain = os.environ.get('DATASET_DOMAIN')
    expert_persona = os.environ.get('DATASET_EXPERT_PERSONA')
    if domain and expert_persona:
        print(f"📥 Loaded from environment: domain={domain}, expert_persona={expert_persona}")
        return domain, expert_persona
    return None, None

# Configuration (override via config.yaml)
DEFAULT_CHUNKS_FILE = "output/results/chunks.json"
OUTPUT_DIR = "output/domain_analysis"
# Directory containing images referenced in chunks (update as needed)
IMAGE_BASE_DIR = "output/results/markdown" 
# Directory containing pre-computed embeddings
EMBEDDINGS_DIR = "output/results/embeddings"

# Embedding Mode Configuration
# Set to False to use BGE-M3 (Text Only), True to use NomicVLEmbed (Multimodal)
USE_MULTIMODAL_EMBEDDINGS = True 

#%% Load Pre-computed Embeddings

def load_precomputed_embeddings(model_name: str = "nomic") -> Tuple[np.ndarray, List[str]]:
    """
    Load pre-computed embeddings from .npz file and corresponding chunk IDs.
    
    Returns:
        Tuple of (embeddings_array, chunk_ids_list)
    """
    embeddings_path = os.path.join(EMBEDDINGS_DIR, "embeddings_dict.npz")
    chunk_ids_path = os.path.join(EMBEDDINGS_DIR, "chunk_ids.json")
    
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not os.path.exists(chunk_ids_path):
        raise FileNotFoundError(f"Chunk IDs file not found: {chunk_ids_path}")
    
    # Load embeddings
    print(f"📂 Loading pre-computed embeddings from {embeddings_path}...")
    with np.load(embeddings_path) as data:
        if model_name in data:
             embeddings = data[model_name]
             print(f"✅ Loaded embeddings for {model_name}: {embeddings.shape}")
        else:
             raise ValueError(f"Model {model_name} not found in embeddings file. Available: {data.files}")
             
    # Load chunk IDs
    print(f"📂 Loading chunk IDs from {chunk_ids_path}...")
    with open(chunk_ids_path, 'r') as f:
        chunk_ids_data = json.load(f)
        if model_name in chunk_ids_data:
            chunk_ids = chunk_ids_data[model_name]
            print(f"✅ Loaded {len(chunk_ids)} chunk IDs for {model_name}")
        else:
            raise ValueError(f"Model {model_name} not found in chunk_ids file.")
            
    if len(chunk_ids) != embeddings.shape[0]:
        raise ValueError(f"Mismatch between chunk IDs count ({len(chunk_ids)}) and embeddings rows ({embeddings.shape[0]})")
        
    return embeddings, chunk_ids

def align_chunks_with_embeddings(chunks: List[Dict], chunk_ids: List[str]) -> Tuple[List[Dict], List[int]]:
    """
    Filter and order chunks to match the sequence of pre-computed embeddings.
    Returns:
        aligned_chunks: List of chunks found
        valid_indices: Indices of the embeddings that correspond to the found chunks
    """
    print("🔄 Aligning chunks with pre-computed embeddings...")
    
    # Create map of chunk_id -> chunk for O(1) lookup
    chunk_map = {}
    for c in chunks:
        c_id = str(c.get('chunk_id', ''))
        chunk_map[c_id] = c
        
    aligned_chunks = []
    valid_indices = []
    missing_ids = []
    
    for idx, target_id in enumerate(chunk_ids):
        target_id = str(target_id)
        if target_id in chunk_map:
            aligned_chunks.append(chunk_map[target_id])
            valid_indices.append(idx)
        else:
            missing_ids.append(target_id)
            
    if missing_ids:
        print(f"⚠️ Warning: {len(missing_ids)} chunks from embeddings not found in source file.")
        if len(missing_ids) < 10:
            print(f"   Missing IDs: {missing_ids}")
            
    print(f"✅ Aligned {len(aligned_chunks)} chunks.")
    return aligned_chunks, valid_indices

def get_embeddings_multimodal(chunks: List[Dict], embed_model: NomicVLEmbed) -> np.ndarray:
    """
    Generate multimodal embeddings for chunks using NomicVLEmbed.
    Handles text and optional images (from 'artifact' field).
    """
    print(f"🖼️ Generating Multimodal Embeddings for {len(chunks)} chunks...")
    
    embeddings = []
    
    for i, chunk in enumerate(chunks):
        text = chunk.get('content', '')
        artifact = chunk.get('artifact', 'None')
        image_path = None
        
        # Try to parse image path from artifact string if present
        if artifact and artifact != 'None':
            if "![" in artifact and "](" in artifact:
                start = artifact.find("](") + 2
                end = artifact.find(")", start)
                rel_path = artifact[start:end]
                image_path = os.path.join(IMAGE_BASE_DIR, rel_path.lstrip('/'))
            else:
                pass

        # Verify image existence
        if image_path and not os.path.exists(image_path):
            image_path = None
            
        # Generate embedding
        try:
            emb = embed_model.embed_multimodal(text, image_path)
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().float().numpy()
            embeddings.append(emb)
        except Exception as e:
            print(f"❌ Error embedding chunk {i}: {e}")
            try:
                emb = embed_model.encode(text, convert_to_numpy=True)
                if isinstance(emb, torch.Tensor):
                    emb = emb.cpu().float().numpy()
                embeddings.append(emb)
            except:
                embeddings.append(np.zeros(768)) 

        if (i+1) % 10 == 0:
            print(f"   Processed {i+1}/{len(chunks)}")
            
    return np.vstack(embeddings)

def get_embeddings_text_only(chunks: List[Dict], model_name: str = "BAAI/bge-m3") -> np.ndarray:
    """
    Generate text-only embeddings using SentenceTransformer (BGE-M3).
    """
    print(f"📝 Generating Text-Only Embeddings using {model_name}...")
    
    docs = [c.get('content', '') for c in chunks]
    
    print(f"   Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"   Encoding {len(docs)} documents...")
    embeddings = model.encode(docs, show_progress_bar=True)
    
    return embeddings

def get_domain_model(chunks: List[Dict], embeddings: Optional[Union[torch.Tensor, np.ndarray]] = None) -> BERTopic:
    """
    Train BERTopic model using embeddings (pre-calculated or computed on-the-fly).
    """
    print(f"🚀 Starting Topic Modeling on {len(chunks)} chunks...")
    
    # Extract text content for BERTopic (it still needs docs for representation)
    docs = [c.get('content', '') for c in chunks]
    
    # 1. Get Embeddings
    if embeddings is not None:
        # Convert torch.Tensor to numpy if needed (BERTopic requires numpy)
        if isinstance(embeddings, torch.Tensor):
            print("✅ Using pre-computed embeddings (converting GPU tensor to numpy for BERTopic)")
            embeddings = embeddings.cpu().float().numpy()
        else:
            print("✅ Using pre-computed embeddings")
    else:
        if USE_MULTIMODAL_EMBEDDINGS:
            print("   Mode: Multimodal (NomicVLEmbed) - COMPUTING ON THE FLY")
            # Try to use cached embedder from main.py if available
            try:
                import sys
                if 'main' in sys.modules and hasattr(sys.modules['main'], '_MODEL_CACHE'):
                    cache = sys.modules['main']._MODEL_CACHE
                    if cache.get('nomic_embedder') is not None:
                        embedder = cache['nomic_embedder']
                        print("   Using cached NomicVLEmbed model (no reload needed)")
                    else:
                        embedder = NomicVLEmbed()
                else:
                    embedder = NomicVLEmbed()
            except:
                embedder = NomicVLEmbed()
            embeddings = get_embeddings_multimodal(chunks, embedder)
        else:
            print("   Mode: Text-Only (BGE-M3) - COMPUTING ON THE FLY")
            embeddings = get_embeddings_text_only(chunks, model_name="BAAI/bge-m3")
        
    print(f"✅ Embeddings shape: {embeddings.shape}")
    print(f"✅ Docs length: {len(docs)}")
    
    if embeddings.shape[0] != len(docs):
        raise ValueError(f"Mismatch: Embeddings rows ({embeddings.shape[0]}) != Docs length ({len(docs)})")

    # 2. Prevent Stochastic Behavior & Dimensionality Reduction
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    
    # 3. Improve Default Representation
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
    
    # 4. Maximal Marginal Relevance (MMR) for better topic diversity
    representation_model = MaximalMarginalRelevance(diversity=0.5)
    
    # Initialize BERTopic
    topic_model = BERTopic(
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        calculate_probabilities=False, 
        verbose=True
    )
    
    # Fit the model using pre-calculated embeddings
    topics, probs = topic_model.fit_transform(docs, embeddings)
    
    print(f"✅ Topic Modeling Complete. Found {len(topic_model.get_topic_info()) - 1} topics.")
    return topic_model, docs, embeddings

def query_llm_for_domain(topic_model: BERTopic) -> Tuple[str, str]:
    """
    Extract domain and expert role using LLM based on topics.
    """
    print("🤖 Querying LLM for Domain and Role...")
    
    # Get top topics info
    topic_info = topic_model.get_topic_info()
    total_count = topic_info['Count'].sum()
    
    print("\n📊 TOPIC SUMMARY:")
    print(f"{'ID':<6} {'Count':<8} {'Freq':<8} {'Keywords'}")
    print("-" * 100)
    
    # Filter out outlier topic (-1) and take top 10
    top_topics = topic_info[topic_info['Topic'] != -1].head(15)
    
    # Format topics for prompt
    topic_list_str = ""
    for _, row in top_topics.iterrows():
        # Representation is a list of keywords
        keywords = ", ".join(row['Representation'][:5]) 
        topic_list_str += f"- Topic {row['Topic']} (Count: {row['Count']}): {keywords}\n"
        print(f"{row['Topic']:<6} {row['Count']:<8} {row['Count']/total_count:<8.1%} {keywords}")
    
    print("-" * 100 + "\n")
    
    # Use prompt from prompt.py
    prompt = PROMPTS["domain_and_expert_from_topics"].format(topic_list_str=topic_list_str)
    
    # Call LLM
    response = call_llm_simple(prompt)
    
    # Parse response
    domain = "Unknown"
    role = "Expert"
    
    # Parse new delimiter-based format:
    # <|#|>START<|#|>Domain<|#|><value><|#|>Expert Role<|#|><value><|#|>END<|#|>
    
    if '<|#|>' in response:
        parts = response.split('<|#|>')
        for i, part in enumerate(parts):
            part_lower = part.strip().lower()
            if i + 1 < len(parts):
                value = parts[i + 1].strip()
                
                # New format: Domain<|#|><value>
                if part_lower == "domain":
                    domain = value
                elif part_lower == "expert role":
                    role = value
                # Legacy format: Domain: <value>
                elif part_lower.startswith("domain:"):
                    domain = part.strip().split(":", 1)[1].strip()
                elif part_lower.startswith("expert role:"):
                    role = part.strip().split(":", 1)[1].strip()
    else:
        # Fallback for line-based format if delimiters not found
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        for line in lines:
            if line.lower().startswith("domain:"):
                domain = line.split(":", 1)[1].strip()
            elif line.lower().startswith("expert role:"):
                role = line.split(":", 1)[1].strip()
    
    # Clean up domain string if it contains multiple lines
    if "\n" in domain:
        domain = domain.split("\n")[0].strip()
        
    # Clean up role string if it contains multiple lines or "Expert Role:" prefix repeated
    if "\n" in role:
        role = role.split("\n")[0].strip()
    if role.lower().startswith("expert role:"):
        role = role.split(":", 1)[1].strip()
        
    return domain, role

def visualize_results(topic_model: BERTopic, docs: List[str], output_dir: str, embeddings: Optional[np.ndarray] = None, generate_plots: bool = False):
    """
    Generate and save visualizations.
    """
    print(f"📊 Generating visualizations in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    if generate_plots:
        # 1. Visualize Topics (Distance Map)
        try:
            fig_topics = topic_model.visualize_topics()
            fig_topics.write_html(os.path.join(output_dir, "topics_distance_map.html"))
            print("  - Saved topics_distance_map.html")
        except Exception as e:
            print(f"  ⚠️ Could not generate topic visualization: {e}")

        # 2. Visualize Hierarchy
        try:
            fig_hierarchy = topic_model.visualize_hierarchy()
            fig_hierarchy.write_html(os.path.join(output_dir, "topics_hierarchy.html"))
            print("  - Saved topics_hierarchy.html")
        except Exception as e:
            print(f"  ⚠️ Could not generate hierarchy visualization: {e}")

        # 3. Visualize Document Datamap (Requires datamapplot and 2D embeddings)
        if embeddings is not None:
            try:
                print("  - Calculating 2D UMAP for Datamap...")
                # Reduce dimensionality of embeddings to 2D for visualization
                # User specified params: n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine'
                umap_2d = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
                reduced_embeddings = umap_2d.fit_transform(embeddings)
                
                print("  - Generating Datamap...")
                fig = topic_model.visualize_document_datamap(
                    docs, 
                    reduced_embeddings=reduced_embeddings, 
                    interactive=True
                )
                
                # Handle datamapplot figure save
                try:
                    fig.save(os.path.join(output_dir, "document_datamap.html"))
                except AttributeError:
                    # Fallback for Plotly figure if return type changes
                    fig.write_html(os.path.join(output_dir, "document_datamap.html"))
                    
                print("  - Saved document_datamap.html")
            except Exception as e:
                print(f"  ⚠️ Could not generate document datamap: {e}")
    else:
        print("  - Plot generation skipped (enable with visualization=True)")

    # 4. Save Topic Info CSV (Always save this)
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(os.path.join(output_dir, "topic_info.csv"), index=False)
    print("  - Saved topic_info.csv")

def fetch_domain_and_role(chunks_file: str = DEFAULT_CHUNKS_FILE, 
                          embeddings: Optional[Union[torch.Tensor, np.ndarray]] = None, 
                          chunk_ids: Optional[List[str]] = None) -> Tuple[str, str]:
    """
    Wrapper to load chunks, run model, and return domain/role.
    
    Args:
        chunks_file: Path to chunks JSON file
        embeddings: Pre-computed embeddings (torch.Tensor on GPU or np.ndarray, optional)
        chunk_ids: List of chunk IDs matching embeddings (optional)
    """
    if not os.path.exists(chunks_file):
        # Try to look in current dir
        local_file = os.path.basename(chunks_file)
        if os.path.exists(local_file):
            chunks_file = local_file
        else:
            print(f"❌ Chunks file not found: {chunks_file}")
            return "Unknown", "Expert"

    print(f"📂 Loading chunks from {chunks_file}...")
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    
    # Use provided embeddings if available (from main.py pipeline)
    if embeddings is not None and chunk_ids is not None:
        print(f"✅ Using embeddings provided from main pipeline")
        # Convert torch.Tensor to numpy if needed (BERTopic requires numpy)
        if isinstance(embeddings, torch.Tensor):
            print(f"   Converting GPU tensor to numpy for BERTopic")
            embeddings = embeddings.cpu().float().numpy()
        # Align chunks to embeddings
        chunks, valid_indices = align_chunks_with_embeddings(chunks, chunk_ids)
        # Filter embeddings to match found chunks
        if len(valid_indices) != embeddings.shape[0]:
             print(f"⚠️ Filtering embeddings from {embeddings.shape[0]} to {len(valid_indices)} rows")
             embeddings = embeddings[valid_indices]
    else:
        # Fallback: Try to load pre-computed embeddings from .npz (legacy support)
        try:
            embeddings, chunk_ids = load_precomputed_embeddings()
            # Align chunks to embeddings
            chunks, valid_indices = align_chunks_with_embeddings(chunks, chunk_ids)
            # Filter embeddings to match found chunks
            if len(valid_indices) != embeddings.shape[0]:
                 print(f"⚠️ Filtering embeddings from {embeddings.shape[0]} to {len(valid_indices)} rows")
                 embeddings = embeddings[valid_indices]
        except Exception as e:
            print(f"⚠️ Could not load/align pre-computed embeddings: {e}")
            print("   Falling back to on-the-fly computation (slower)")
            embeddings = None
        
    topic_model, _, _ = get_domain_model(chunks, embeddings=embeddings)
    domain, role = query_llm_for_domain(topic_model)
    
    # Save to environment variables
    save_domain_expert_to_env(domain, role)
    
    return domain, role

def main(visualization: bool = False):
    setup_logging()
    
    # Load Chunks
    chunks_file = DEFAULT_CHUNKS_FILE
    if not os.path.exists(chunks_file):
        print(f"❌ Chunks file not found: {chunks_file}")
        # Try to look in current dir
        local_file = "chunks.json"
        if os.path.exists(local_file):
            chunks_file = local_file
        else:
            return

    print(f"📂 Loading chunks from {chunks_file}...")
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
        
    # Try to load pre-computed embeddings
    try:
        embeddings, chunk_ids = load_precomputed_embeddings()
        # Align chunks to embeddings
        chunks, valid_indices = align_chunks_with_embeddings(chunks, chunk_ids)
        # Filter embeddings to match found chunks
        if len(valid_indices) != embeddings.shape[0]:
             print(f"⚠️ Filtering embeddings from {embeddings.shape[0]} to {len(valid_indices)} rows")
             embeddings = embeddings[valid_indices]
    except Exception as e:
        print(f"⚠️ Could not load/align pre-computed embeddings: {e}")
        import traceback
        traceback.print_exc()
        print("   Falling back to on-the-fly computation (slower)")
        embeddings = None

    # Run Topic Modeling
    topic_model, docs, embeddings = get_domain_model(chunks, embeddings=embeddings)
    
    # Get Domain and Role
    domain, role = query_llm_for_domain(topic_model)
    
    # Save to environment variables
    save_domain_expert_to_env(domain, role)
    
    print("\n" + "="*50)
    print(f"🎯 RESULTS")
    print(f"Domain: {domain}")
    print(f"Expert Role: {role}")
    print("="*50 + "\n")
    
    # Visualization
    visualize_results(topic_model, docs, OUTPUT_DIR, embeddings=embeddings, generate_plots=visualization)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract domain and expert role from semantic chunks")
    parser.add_argument("--vis", action="store_true", help="Enable visualization generation")
    args = parser.parse_args()
    
    main(visualization=args.vis)
