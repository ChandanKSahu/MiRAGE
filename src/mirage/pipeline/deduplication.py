
import json
import logging
import sys
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict

# Import helper modules
from mirage.embeddings.models import get_best_embedding_model
from mirage.embeddings.rerankers_text import LLMReranker
from mirage.core.llm import setup_logging, call_vlm_with_multiple_images
from mirage.core.prompts import PROMPTS

# Try to load configuration from config.yaml
try:
    from mirage.core.config import load_config
    _cfg = load_config()
    _dedup = _cfg.get('deduplication', {})
    
    QUESTION_SIMILARITY_THRESHOLD = _dedup.get('question_similarity_threshold', 0.75)
    _ans_sim = _dedup.get('answer_similarity', {})
    ANSWER_SIMILARITY_HIGH = _ans_sim.get('high', 0.95)
    ANSWER_SIMILARITY_MEDIUM = _ans_sim.get('medium', 0.85)
    ANSWER_SIMILARITY_LOW = _ans_sim.get('low', 0.70)
    MIN_COMMUNITY_SIZE = _dedup.get('min_community_size', 2)
    # alpha parameter: weight for semantic similarity vs chunk lineage (Eq. 10 in manuscript)
    ALPHA = _dedup.get('alpha', 0.6)
    print(f"[OK] Deduplication config loaded: alpha={ALPHA}, question_threshold={QUESTION_SIMILARITY_THRESHOLD}")
except ImportError:
    print("[WARN] config_loader not available, using default deduplication configuration")
    QUESTION_SIMILARITY_THRESHOLD = 0.75
    ANSWER_SIMILARITY_HIGH = 0.95
    ANSWER_SIMILARITY_MEDIUM = 0.85
    ANSWER_SIMILARITY_LOW = 0.70
    MIN_COMMUNITY_SIZE = 2
    ALPHA = 0.6  # Default alpha for Eq. 10: alpha * semantic_sim + (1-alpha) * jaccard

# Configuration
OUTPUT_DIR = "output"
INPUT_FILE = os.path.join(OUTPUT_DIR, "qa_multihop_pass.json")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "qa_dataset_deduplicated.json")

def load_dataset(filepath):
    print(f"Loading dataset from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_dataset(data, filepath):
    print(f"Saving {len(data)} items to {filepath}...")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def compute_chunk_overlap(qa1: Dict, qa2: Dict) -> float:
    """
    Compute Jaccard similarity based on chunk lineage.
    Returns overlap ratio [0, 1] of chunks used to generate each QA.
    """
    # Extract chunk identifiers from chunks_added (list of dicts with file_name and chunk_id)
    def extract_chunk_ids(qa):
        ids = {qa.get('chunk_id', -1)}
        for chunk in qa.get('chunks_added', []):
            if isinstance(chunk, dict):
                ids.add((chunk.get('file_name', ''), chunk.get('chunk_id', '')))
            else:
                ids.add(chunk)
        return ids
    
    chunks1 = extract_chunk_ids(qa1)
    chunks2 = extract_chunk_ids(qa2)
    
    # Remove invalid IDs
    chunks1.discard(-1)
    chunks2.discard(-1)
    
    if not chunks1 or not chunks2:
        return 0.0
    
    intersection = len(chunks1 & chunks2)
    union = len(chunks1 | chunks2)
    
    return intersection / union if union > 0 else 0.0

def select_best_qa(cluster_items: List[Dict]) -> Dict:
    """
    Select the best QA from exact duplicates based on quality metrics.
    Prioritizes: relevance score > difficulty score > answer length
    """
    best = cluster_items[0]
    best_score = (
        float(best.get('relevance_score', 0)),
        float(best.get('difficulty_score', 0)),
        len(best.get('answer', ''))
    )
    
    for item in cluster_items[1:]:
        score = (
            float(item.get('relevance_score', 0)),
            float(item.get('difficulty_score', 0)),
            len(item.get('answer', ''))
        )
        if score > best_score:
            best = item
            best_score = score
    
    return best

def reorganize_qa_packs(merged_items: List[Dict], base_metadata: Dict,
                        expert_persona: str,
                        domain: str) -> List[Dict]:
    """
    Reorganize merged QA pairs into balanced question-answer packs.
    Groups related questions together while keeping packs balanced.
    
    Args:
        merged_items: List of merged QA dicts with 'question' and 'answer' keys
        base_metadata: Metadata to propagate to reorganized items
        expert_persona: Expert role for domain-specific organization
        domain: Domain context for organization
    
    Returns:
        List of reorganized QA dicts
    """
    if len(merged_items) <= 1:
        return merged_items
    
    # Prepare merged questions and answers for the prompt
    merged_questions = "\n".join([
        f"{i+1}. {item['question']}" 
        for i, item in enumerate(merged_items)
    ])
    merged_answers = "\n".join([
        f"{i+1}. {item['answer']}" 
        for i, item in enumerate(merged_items)
    ])
    
    prompt_template = PROMPTS.get("deduplication_reorganize", "")
    if not prompt_template:
        logging.warning("deduplication_reorganize prompt not found, skipping reorganization.")
        return merged_items
    
    # Format prompt with domain and expert role
    formatted_prompt = prompt_template.format(expert_persona=expert_persona, domain=domain)
    prompt = f"{formatted_prompt}\n\nInput:\nMerged Questions:\n{merged_questions}\n\nMerged Answers:\n{merged_answers}"
    
    try:
        response = call_vlm_with_multiple_images(prompt, [])
        reorganized = parse_reorganized_packs(response, base_metadata)
        
        if not reorganized:
            logging.warning("LLM returned empty reorganization, keeping merged items.")
            return merged_items
        
        return reorganized
        
    except Exception as e:
        logging.error(f"Error in LLM reorganization: {e}")
        return merged_items

def parse_reorganized_packs(response_text: str, base_metadata: Dict) -> List[Dict]:
    """
    Parse the LLM response containing reorganized QA packs.
    
    New format: <|#|>START<|#|>Question<|#|><text><|#|>Answer<|#|><text><|#|>NEXT<|#|>...<|#|>END<|#|>
    
    Returns:
        List of reorganized QA dicts
    """
    tuple_delimiter = PROMPTS.get("DEFAULT_TUPLE_DELIMITER", "<|#|>")
    completion_delimiter = PROMPTS.get("DEFAULT_COMPLETION_DELIMITER", "<|#|>END<|#|>")
    
    qa_packs = []
    
    try:
        # Remove completion delimiter if present
        if completion_delimiter in response_text:
            response_text = response_text.split(completion_delimiter)[0].strip()
        
        # Remove START delimiter if present
        start_delimiter = tuple_delimiter + "START" + tuple_delimiter
        if start_delimiter in response_text:
            response_text = response_text.split(start_delimiter, 1)[-1].strip()
        
        # Split by NEXT delimiter to get individual packs
        next_delimiter = tuple_delimiter + "NEXT" + tuple_delimiter
        pack_texts = response_text.split(next_delimiter)
        
        for pack_text in pack_texts:
            pack_text = pack_text.strip()
            if not pack_text:
                continue
            
            # Parse Question and Answer from the pack using delimiter format
            parts = pack_text.split(tuple_delimiter)
            
            question = None
            answer = None
            
            for i, part in enumerate(parts):
                part_lower = part.strip().lower()
                if i + 1 < len(parts):
                    value = parts[i + 1].strip()
                    if part_lower == "question":
                        question = value
                    elif part_lower == "answer":
                        answer = value
            
            if question and answer:
                new_item = base_metadata.copy()
                new_item["question"] = question
                new_item["answer"] = answer
                new_item["reorganized"] = True
                qa_packs.append(new_item)
        
        return qa_packs
        
    except Exception as e:
        logging.error(f"Error parsing reorganized packs: {e}")
        return []

def hierarchical_clustering(
    data: List[Dict],
    question_embeddings: torch.Tensor,
    answer_embeddings: torch.Tensor
) -> List[List[int]]:
    """
    Two-stage hierarchical clustering:
    1. Cluster questions by semantic similarity (topic/intent)
    2. Within each question cluster, sub-cluster answers by similarity
    
    Returns: List of clusters (each cluster is a list of indices)
    """
    print(f"\nStage 1: Clustering questions by topic (threshold: {QUESTION_SIMILARITY_THRESHOLD})...")
    
    # Stage 1: Cluster questions to group by topic/intent
    question_clusters = util.community_detection(
        question_embeddings,
        threshold=QUESTION_SIMILARITY_THRESHOLD,
        min_community_size=1
    )
    
    print(f"[OK] Found {len(question_clusters)} question-based topic groups")
    
    # Stage 2: Within each question cluster, sub-cluster by answer similarity
    final_clusters = []
    singleton_count = 0
    
    print(f"\nStage 2: Sub-clustering answers within each topic group...")
    
    for q_cluster in tqdm(question_clusters, desc="Processing question clusters"):
        if len(q_cluster) == 1:
            singleton_count += 1
            continue  # No duplicates possible
        
        # Extract answer embeddings for this question cluster
        q_cluster_list = list(q_cluster)
        q_cluster_answer_embs = answer_embeddings[q_cluster_list]
        
        # Check for chunk overlap to prioritize merging
        # Build chunk overlap matrix
        chunk_overlap_matrix = np.zeros((len(q_cluster_list), len(q_cluster_list)))
        for i, idx_i in enumerate(q_cluster_list):
            for j, idx_j in enumerate(q_cluster_list):
                if i != j:
                    chunk_overlap_matrix[i, j] = compute_chunk_overlap(data[idx_i], data[idx_j])
        
        # Compute answer similarity matrix
        answer_sim_matrix = util.pytorch_cos_sim(q_cluster_answer_embs, q_cluster_answer_embs)
        
        # Combined similarity per Eq. 10: alpha * cos(e_ai, e_aj) + (1-alpha) * J(C^s_i, C^s_j)
        # alpha weights semantic similarity; (1-alpha) weights chunk lineage Jaccard overlap
        combined_sim = ALPHA * answer_sim_matrix.cpu().numpy() + (1 - ALPHA) * chunk_overlap_matrix
        
        # Find high-similarity pairs and group them
        visited = set()
        for i in range(len(q_cluster_list)):
            if i in visited:
                continue
            
            cluster = [q_cluster_list[i]]
            visited.add(i)
            
            for j in range(i + 1, len(q_cluster_list)):
                if j not in visited and combined_sim[i, j] >= ANSWER_SIMILARITY_LOW:
                    cluster.append(q_cluster_list[j])
                    visited.add(j)
            
            if len(cluster) >= MIN_COMMUNITY_SIZE:
                final_clusters.append(cluster)
    
    print(f"[OK] Found {len(final_clusters)} answer clusters requiring merge")
    print(f"[INFO] {singleton_count} singletons (unique QAs)")
    
    return final_clusters

def process_cluster_by_similarity(
    cluster_items: List[Dict],
    cluster_indices: List[int],
    answer_embeddings: torch.Tensor,
    llm_merger: LLMReranker,
    expert_persona: str,
    domain: str,
    enable_reorganization: bool = True
) -> List[Dict]:
    """
    Process a cluster with stratified handling based on answer similarity.
    - High similarity (>0.95): Exact duplicates -> select best
    - Medium similarity (0.85-0.95): Partial overlap -> LLM merge -> reorganize
    - Low similarity (0.70-0.85): Related -> LLM evaluate -> reorganize
    
    After LLM merging, optionally reorganizes into balanced QA packs.
    
    Args:
        cluster_items: List of QA dicts in the cluster
        cluster_indices: Original indices of items in the cluster
        answer_embeddings: Tensor of answer embeddings
        llm_merger: LLMReranker instance for merging
        enable_reorganization: Whether to reorganize after merging
        expert_persona: Expert role for domain-specific handling
        domain: Domain context for handling
    """
    if len(cluster_items) < 2:
        return cluster_items
    
    # Compute pairwise answer similarities within cluster
    cluster_answer_embs = answer_embeddings[cluster_indices]
    sim_matrix = util.pytorch_cos_sim(cluster_answer_embs, cluster_answer_embs).cpu().numpy()
    
    # Get max similarity (excluding diagonal)
    np.fill_diagonal(sim_matrix, 0)
    max_similarity = np.max(sim_matrix)
    avg_similarity = np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])
    
    # Base metadata for propagation
    base_metadata = cluster_items[0].copy()
    base_metadata.pop('question', None)
    base_metadata.pop('answer', None)
    base_metadata['merged_from_count'] = len(cluster_items)
    
    # Stratified handling
    if max_similarity >= ANSWER_SIMILARITY_HIGH:
        # Tier 1: Exact duplicates - just pick the best one (no reorganization needed)
        logging.info(f"Cluster of {len(cluster_items)} with max_sim={max_similarity:.3f}: Selecting best (exact duplicates)")
        best = select_best_qa(cluster_items)
        best['dedup_method'] = 'select_best'
        best['merged_from_count'] = len(cluster_items)
        return [best]
    
    elif avg_similarity >= ANSWER_SIMILARITY_MEDIUM:
        # Tier 2: High overlap - merge with LLM, then reorganize
        logging.info(f"Cluster of {len(cluster_items)} with avg_sim={avg_similarity:.3f}: LLM merge (high overlap)")
        merged = llm_merger.deduplicate_and_merge(cluster_items)
        
        # Reorganize into balanced packs if enabled and multiple items
        if enable_reorganization and len(merged) > 1:
            logging.info(f"Reorganizing {len(merged)} merged items into balanced packs")
            merged = reorganize_qa_packs(merged, base_metadata, expert_persona=expert_persona, domain=domain)
        
        for item in merged:
            item['dedup_method'] = 'llm_merge_high'
        return merged
    
    else:
        # Tier 3: Related but potentially distinct - let LLM decide, then reorganize
        logging.info(f"Cluster of {len(cluster_items)} with avg_sim={avg_similarity:.3f}: LLM evaluate (medium overlap)")
        merged = llm_merger.deduplicate_and_merge(cluster_items)
        
        # Reorganize into balanced packs if enabled and multiple items
        if enable_reorganization and len(merged) > 1:
            logging.info(f"Reorganizing {len(merged)} merged items into balanced packs")
            merged = reorganize_qa_packs(merged, base_metadata, expert_persona=expert_persona, domain=domain)
        
        for item in merged:
            item['dedup_method'] = 'llm_merge_medium'
        return merged

def deduplicate_dataset():
    setup_logging()
    
    # 1. Load Data
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Input file {INPUT_FILE} not found.")
        return

    data = load_dataset(INPUT_FILE)
    if not data:
        print("[WARN] Dataset is empty.")
        return

    print(f"\n{'='*80}")
    print("HIERARCHICAL DEDUPLICATION: Questions -> Answers")
    print(f"{'='*80}\n")

    # 2. Prepare separate embeddings for questions and answers
    print("Preparing text for embedding...")
    questions = [item['question'] for item in data]
    answers = [item['answer'] for item in data]

    # 3. Load Embedding Model & Embed
    model_name = get_best_embedding_model()
    print(f"Loading embedding model: {model_name}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedder = SentenceTransformer(model_name, device=device)
    
    print(f"Generating question embeddings for {len(questions)} QA pairs...")
    question_embeddings = embedder.encode(questions, convert_to_tensor=True, show_progress_bar=True)
    
    print(f"Generating answer embeddings for {len(answers)} QA pairs...")
    answer_embeddings = embedder.encode(answers, convert_to_tensor=True, show_progress_bar=True)

    # 4. Hierarchical Clustering: Questions first, then Answers
    clusters = hierarchical_clustering(data, question_embeddings, answer_embeddings)
    
    # 5. Track processed items
    clustered_indices = set()
    for cluster in clusters:
        clustered_indices.update(cluster)

    # Initialize LLM merger
    llm_merger = LLMReranker()
    final_dataset = []
    
    # Statistics
    stats = {
        'original': len(data),
        'singletons': 0,
        'clusters_processed': 0,
        'items_in_clusters': 0,
        'exact_duplicates': 0,
        'llm_merges': 0,
        'reorganized_packs': 0
    }

    # Add singletons (items not in any cluster)
    for i in range(len(data)):
        if i not in clustered_indices:
            final_dataset.append(data[i])
            stats['singletons'] += 1

    print(f"\n[INFO] Added {stats['singletons']} unique (singleton) items.")

    # 6. Process clusters with stratified handling
    print(f"\nProcessing {len(clusters)} clusters with stratified merge strategy...")
    
    for cluster in tqdm(clusters, desc="Merging clusters"):
        cluster_items = [data[idx] for idx in cluster]
        stats['clusters_processed'] += 1
        stats['items_in_clusters'] += len(cluster_items)
        
        # Process with stratified approach
        merged_items = process_cluster_by_similarity(
            cluster_items, 
            cluster, 
            answer_embeddings,
            llm_merger
        )
        
        # Track merge type
        if merged_items and merged_items[0].get('dedup_method') == 'select_best':
            stats['exact_duplicates'] += 1
        else:
            stats['llm_merges'] += 1
        
        # Track reorganized packs
        reorganized_count = sum(1 for item in merged_items if item.get('reorganized', False))
        if reorganized_count > 0:
            stats['reorganized_packs'] += reorganized_count
        
        final_dataset.extend(merged_items)

    # 7. Save Results
    print("\n" + "="*80)
    print("HIERARCHICAL DEDUPLICATION SUMMARY")
    print("="*80)
    print(f"Original count:           {stats['original']}")
    print(f"Final count:              {len(final_dataset)}")
    print(f"Reduction:                {stats['original'] - len(final_dataset)} ({100*(stats['original'] - len(final_dataset))/stats['original']:.1f}%)")
    print(f"---")
    print(f"Singleton items:          {stats['singletons']}")
    print(f"Clusters processed:       {stats['clusters_processed']}")
    print(f"Items in clusters:        {stats['items_in_clusters']}")
    print(f"Exact duplicates removed: {stats['exact_duplicates']}")
    print(f"LLM merges performed:     {stats['llm_merges']}")
    print(f"Reorganized QA packs:     {stats['reorganized_packs']}")
    print("="*80)
    
    save_dataset(final_dataset, OUTPUT_FILE)

if __name__ == "__main__":
    deduplicate_dataset()
