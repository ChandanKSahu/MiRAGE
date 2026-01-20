"""
Optimized Metrics Evaluation with Minimal LLM Calls

Harmonized with metrics.py - same metric names, optimized implementation.

Strategy:
- 1 PREPARATION call per QA pair: Extract claims + reverse questions
- 1 EVALUATION call per metric: Batch verify all claims/contexts together
- Total: 4-6 LLM calls per QA vs 10-20+ in standard RAGAS

METRICS IMPLEMENTED (same as metrics.py):
1. Faithfulness - Answer claims supported by context?
2. Answer Relevancy - Answer addresses question?
3. Context Precision - Retrieved chunks relevant and well-ranked?
4. Context Recall - Context contains reference info? (skip for dataset creation)
5. Multimodal Faithfulness - Answer grounded in text+images?
6. Multimodal Relevance - Answer uses multimodal context?
7. Context Necessity - Requires context to answer? (anti-parametric bias)
8. Semantic Diversity - Questions diverse?
9. Domain Coverage - Corpus coverage?
10. Multihop Reasoning - Multi-step reasoning quality?
11. Visual Dependency - Needs image to answer?

Usage modes:
- Dataset Creation: faithfulness, answer_relevancy, context_necessity
- RAG Evaluation: All metrics
"""

import os
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import numpy as np

# Imports
try:
    from mirage.core.llm import call_llm_simple as call_llm, batch_call_vlm_interweaved, call_vlm_interweaved, batch_call_llm, API_KEY
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    API_KEY = None
    print("Warning: mirage.core.llm not available")

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Fallback: sentence-transformers for local embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class LocalEmbeddingWrapper:
    """Wrapper to make sentence-transformers compatible with langchain embeddings interface."""
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        print(f"✅ Loaded local embedding model: {model_name}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        return self.model.encode(text, normalize_embeddings=True).tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        return self.model.encode(texts, normalize_embeddings=True).tolist()

# Import prompts from mirage.core.prompts
try:
    from mirage.core.prompts import PROMPTS_METRICS
    PROMPTS_AVAILABLE = True
except ImportError:
    PROMPTS_AVAILABLE = False
    PROMPTS_METRICS = {}
    print("Warning: PROMPTS_METRICS not available from mirage.core.prompts")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_image_path_from_content(content: str, file_name: str = None) -> Optional[str]:
    """Extract image path from markdown content (e.g., ![Image](path))
    
    Args:
        content: The markdown content string
        file_name: The source document name (used for directory structure)
    
    Returns:
        First valid image path found, or None
    """
    if not content:
        return None
    
    # Find all markdown image references: ![alt](path)
    matches = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', content)
    
    if not matches:
        return None
    
    # Use first match
    rel_path = matches[0]
    
    # Try to construct absolute path (override via OUTPUT_DIR/markdown)
    IMAGE_BASE_DIR = "output/results/markdown"
    
    if file_name and file_name != 'unknown':
        abs_path = f"{IMAGE_BASE_DIR}/{file_name}/{rel_path}"
    else:
        if rel_path.startswith('ref_artifacts/'):
            abs_path = f"{IMAGE_BASE_DIR}/{rel_path}"
        else:
            abs_path = f"{IMAGE_BASE_DIR}/{rel_path}"
    
    # Check if file exists
    if os.path.exists(abs_path):
        return abs_path
    
    # Even if file doesn't exist, return the path if it looks valid
    # (for detection purposes - file might be in different location)
    return abs_path if rel_path else None


def has_image_in_chunk(chunk: Dict) -> bool:
    """Check if a chunk has an image (via image_path or content)
    
    Args:
        chunk: Chunk dictionary with 'image_path' and/or 'content'
    
    Returns:
        True if chunk has an image reference
    """
    if not isinstance(chunk, dict):
        return False
    
    # Check image_path field first
    image_path = chunk.get('image_path')
    if image_path and image_path != 'null' and image_path is not None:
        return True
    
    # Check content for markdown image references
    content = chunk.get('content', '')
    if content and re.search(r'!\[[^\]]*\]\([^)]+\)', content):
        return True
    
    return False


# ============================================================================
# PROMPTS FOR OPTIMIZED METRICS (from prompt.py or fallback)
# ============================================================================

# Use imported prompts or define fallbacks
PROMPT_PREPARE_QA = PROMPTS_METRICS.get("prepare_qa", """You are a QA analysis assistant. Analyze the following QA pair and extract information needed for evaluation.

QUESTION: {question}

ANSWER: {answer}

REFERENCE (Ground Truth): {reference}

TASKS:
1. Extract ALL atomic claims/statements from the ANSWER (factual assertions that can be verified)
2. Extract ALL atomic claims/statements from the REFERENCE
3. Generate {num_reverse_questions} diverse questions that the ANSWER could plausibly answer

OUTPUT FORMAT (use exactly this format):
ANSWER_CLAIMS:
- [claim 1]
- [claim 2]
...

REFERENCE_CLAIMS:
- [claim 1]
- [claim 2]
...

REVERSE_QUESTIONS:
- [question 1]
- [question 2]
- [question 3]
""")

PROMPT_FAITHFULNESS = PROMPTS_METRICS.get("faithfulness", """You are a faithfulness evaluator. Determine if each claim from the answer can be inferred from the given context.

CONTEXT:
{context}

CLAIMS TO VERIFY:
{claims}

For each claim, respond with ONLY "SUPPORTED" or "NOT_SUPPORTED".

OUTPUT FORMAT (one per line, in order):
CLAIM_1: SUPPORTED/NOT_SUPPORTED
CLAIM_2: SUPPORTED/NOT_SUPPORTED
...
""")

PROMPT_CONTEXT_RECALL = PROMPTS_METRICS.get("context_recall", """You are a context recall evaluator. Determine if each claim from the reference/ground truth can be attributed to the retrieved context.

CONTEXT:
{context}

REFERENCE CLAIMS TO VERIFY:
{claims}

For each claim, respond with ONLY "ATTRIBUTED" or "NOT_ATTRIBUTED".

OUTPUT FORMAT (one per line, in order):
CLAIM_1: ATTRIBUTED/NOT_ATTRIBUTED
CLAIM_2: ATTRIBUTED/NOT_ATTRIBUTED
...
""")

PROMPT_CONTEXT_PRECISION = PROMPTS_METRICS.get("context_precision", """You are a context precision evaluator. For each context chunk, determine if it is RELEVANT or NOT_RELEVANT to answering the question, given the reference answer.

QUESTION: {question}
REFERENCE ANSWER: {reference}

CONTEXT CHUNKS:
{contexts}

For each context chunk, respond with ONLY "RELEVANT" or "NOT_RELEVANT".

OUTPUT FORMAT (one per line, in order):
CHUNK_1: RELEVANT/NOT_RELEVANT
CHUNK_2: RELEVANT/NOT_RELEVANT
...
""")

PROMPT_MULTIMODAL_FAITHFULNESS = PROMPTS_METRICS.get("multimodal_faithfulness", """You are a multimodal faithfulness evaluator. Verify if EACH claim from the answer can be inferred from the provided context (text AND/OR images).

QUESTION: {question}

ANSWER: {answer}

CLAIMS TO VERIFY:
{claims}

For EACH claim, determine:
1. Is it SUPPORTED or NOT_SUPPORTED by the context?
2. If supported, is it from TEXT, IMAGE, or BOTH?

OUTPUT FORMAT (one per line, in order):
CLAIM_1: SUPPORTED/NOT_SUPPORTED | SOURCE: TEXT/IMAGE/BOTH/NONE
CLAIM_2: SUPPORTED/NOT_SUPPORTED | SOURCE: TEXT/IMAGE/BOTH/NONE
...

SUMMARY:
TEXT_GROUNDED: YES/NO
VISUAL_GROUNDED: YES/NO/NA
SUPPORTED_COUNT: [number]
TOTAL_CLAIMS: [number]
""")

PROMPT_MULTIMODAL_RELEVANCE = PROMPTS_METRICS.get("multimodal_relevance", """You are a multimodal relevance evaluator. Generate {num_questions} questions that the given answer could plausibly be answering, then evaluate relevance.

ANSWER: {answer}

CONTEXT: (text and images provided below)

TASK:
1. Generate {num_questions} diverse questions that this answer could address
2. For each generated question, indicate if it uses TEXT context, IMAGE context, or BOTH

OUTPUT FORMAT:
GENERATED_QUESTIONS:
Q1: [question] | USES: TEXT/IMAGE/BOTH
Q2: [question] | USES: TEXT/IMAGE/BOTH
Q3: [question] | USES: TEXT/IMAGE/BOTH

CONTEXT_UTILIZATION:
USES_TEXT: YES/NO
USES_IMAGES: YES/NO/NA
RELEVANCE_SCORE: [0.0-1.0]
""")

PROMPT_CONTEXT_NECESSITY_WITHOUT = PROMPTS_METRICS.get("context_necessity_without", """You are an expert assistant. Answer the following question using ONLY your general knowledge. Do NOT make up specific facts.

If you cannot answer confidently without additional context, respond with: CANNOT_ANSWER

QUESTION: {question}

YOUR ANSWER:""")

PROMPT_CONTEXT_NECESSITY_VERIFY = PROMPTS_METRICS.get("context_necessity_verify", """Compare the model's answer to the ground truth answer.

GROUND TRUTH: {ground_truth}

MODEL ANSWER: {model_answer}

Respond with exactly one of:
- MATCH: YES (if model answer is correct and complete)
- MATCH: PARTIAL (if model answer is partially correct)
- MATCH: NO (if model answer is incorrect or missing key information)

YOUR VERDICT:""")

PROMPT_MULTIHOP_REASONING = PROMPTS_METRICS.get("multihop_reasoning", """Analyze if answering this question requires multi-hop reasoning (combining information from multiple sources).

CONTEXTS:
{contexts}

QUESTION: {question}

ANSWER: {answer}

Evaluate:
1. HOP_COUNT: How many distinct pieces of information must be combined? (1 = single fact, 2+ = multi-hop)
2. REASONING_SCORE: How complex is the reasoning? (0.0 = trivial, 1.0 = complex multi-step)
3. BRIDGE_ENTITY: What entity/concept connects the information pieces? (or "None")

OUTPUT FORMAT:
HOP_COUNT: [number]
REASONING_SCORE: [0.0-1.0]
BRIDGE_ENTITY: [entity or None]
EXPLANATION: [brief explanation]
""")

PROMPT_VISUAL_DEPENDENCY = PROMPTS_METRICS.get("visual_dependency", """You are given ONLY text context (no images). Determine if you can fully answer the question.

TEXT CONTEXT:
{contexts}

QUESTION: {question}

If you can answer completely using ONLY the text above, provide your answer.
If you CANNOT answer because visual information (figures, diagrams, images) is missing, respond with: MISSING_VISUAL

YOUR RESPONSE:""")


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PreparedQA:
    """Prepared QA pair with extracted claims and reverse questions"""
    question: str
    answer: str
    reference: str
    contexts: List[str]
    context_chunks: List[Dict] = field(default_factory=list)
    
    # Extracted by preparation call
    answer_claims: List[str] = field(default_factory=list)
    reference_claims: List[str] = field(default_factory=list)
    reverse_questions: List[str] = field(default_factory=list)
    concept_hops_question: str = ""
    
    # Metric scores (filled after evaluation)
    faithfulness_score: float = 0.0
    context_recall_score: float = 0.0
    context_precision_score: float = 0.0
    answer_relevancy_score: float = 0.0
    multimodal_faithfulness_score: float = 0.0
    multimodal_relevance_score: float = 0.0
    
    # Detailed results
    faithfulness_details: Dict = field(default_factory=dict)
    context_recall_details: Dict = field(default_factory=dict)
    context_precision_details: Dict = field(default_factory=dict)
    multimodal_details: Dict = field(default_factory=dict)


# ============================================================================
# OPTIMIZED METRICS EVALUATOR
# ============================================================================

class OptimizedMetricsEvaluator:
    """
    Evaluates RAGAS-style metrics with minimal LLM calls.
    
    Call Pattern per QA pair:
    1. prepare_qa() - 1 LLM call to extract claims + generate reverse questions
    2. evaluate_faithfulness() - 1 LLM call to verify all answer claims
    3. evaluate_context_recall() - 1 LLM call to verify all reference claims
    4. evaluate_context_precision() - 1 LLM call to evaluate all contexts
    5. evaluate_answer_relevancy() - 0 LLM calls (uses embeddings only)
    6. evaluate_multimodal_faithfulness() - 1 VLM call (if images present)
    7. evaluate_multimodal_relevance() - 1 VLM call (if images present)
    
    Total: 4-6 LLM/VLM calls per QA pair (vs 10-20+ with default RAGAS)
    """
    
    def __init__(self, 
                 model_name: str = "gemini-2.0-flash",
                 embedding_model: str = "models/text-embedding-004",
                 num_reverse_questions: int = 3,
                 max_workers: int = 8,
                 enable_multimodal: bool = True):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.num_reverse_questions = num_reverse_questions
        self.max_workers = max_workers
        self.enable_multimodal = enable_multimodal
        
        # Initialize embeddings with fallback
        self.embeddings = None
        self.embedding_type = None
        
        # Try Gemini embeddings first
        if GEMINI_AVAILABLE and API_KEY:
            try:
                self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=API_KEY)
                self.embedding_type = "gemini"
                print(f"✅ Using Gemini embeddings: {embedding_model}")
            except Exception as e:
                print(f"Warning: Failed to initialize Gemini embeddings: {e}")
        
        # Fallback to sentence-transformers (local embeddings)
        if self.embeddings is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embeddings = LocalEmbeddingWrapper("BAAI/bge-m3")
                self.embedding_type = "local"
            except Exception as e:
                print(f"Warning: Failed to initialize local embeddings: {e}")
        
        # No embeddings available
        if self.embeddings is None:
            if not GEMINI_AVAILABLE and not SENTENCE_TRANSFORMERS_AVAILABLE:
                print("Warning: No embedding model available (install langchain-google-genai or sentence-transformers)")
                print("         answer_relevancy and semantic_diversity will be skipped")
            elif not GEMINI_AVAILABLE:
                print("Warning: langchain-google-genai not installed, no fallback available")
            elif not API_KEY:
                print("Warning: API key not available for embeddings")
    
    # ========================================================================
    # STEP 1: PREPARATION (1 LLM call per QA)
    # ========================================================================
    
    def prepare_qa(self, question: str, answer: str, reference: str, 
                   contexts: List[str], context_chunks: List[Dict] = None) -> PreparedQA:
        """
        Prepare QA pair by extracting claims and generating reverse questions.
        1 LLM call.
        """
        prepared = PreparedQA(
            question=question,
            answer=answer,
            reference=reference,
            contexts=contexts,
            context_chunks=context_chunks or []
        )
        
        prompt = PROMPT_PREPARE_QA.format(
            question=question,
            answer=answer,
            reference=reference,
            num_reverse_questions=self.num_reverse_questions
        )
        
        try:
            response = call_llm(prompt)
            
            # Parse answer claims
            answer_claims_match = re.search(
                r'ANSWER_CLAIMS:\s*\n(.*?)(?=REFERENCE_CLAIMS:|$)', 
                response, re.DOTALL
            )
            if answer_claims_match:
                claims_text = answer_claims_match.group(1)
                prepared.answer_claims = [
                    c.strip().lstrip('- ').strip() 
                    for c in claims_text.strip().split('\n') 
                    if c.strip() and c.strip() != '-'
                ]
            
            # Parse reference claims
            ref_claims_match = re.search(
                r'REFERENCE_CLAIMS:\s*\n(.*?)(?=REVERSE_QUESTIONS:|$)', 
                response, re.DOTALL
            )
            if ref_claims_match:
                claims_text = ref_claims_match.group(1)
                prepared.reference_claims = [
                    c.strip().lstrip('- ').strip() 
                    for c in claims_text.strip().split('\n') 
                    if c.strip() and c.strip() != '-'
                ]
            
            # Parse concept hops for question
            concept_hops_match = re.search(
                r'CONCEPT_HOPS_QUESTION:\s*\n(.*?)(?=REVERSE_QUESTIONS:|$)', 
                response, re.DOTALL
            )
            if concept_hops_match:
                prepared.concept_hops_question = concept_hops_match.group(1).strip()
            
            # Parse reverse questions
            rev_q_match = re.search(
                r'REVERSE_QUESTIONS:\s*\n(.*?)$', 
                response, re.DOTALL
            )
            if rev_q_match:
                q_text = rev_q_match.group(1)
                prepared.reverse_questions = [
                    q.strip().lstrip('- ').strip() 
                    for q in q_text.strip().split('\n') 
                    if q.strip() and q.strip() != '-'
                ][:self.num_reverse_questions]
            
        except Exception as e:
            print(f"Error in prepare_qa: {e}")
            # Fallback: use answer/reference as single claim
            prepared.answer_claims = [answer] if answer else []
            prepared.reference_claims = [reference] if reference else []
            prepared.reverse_questions = [question]
        
        return prepared
    
    # ========================================================================
    # STEP 2: METRIC EVALUATION (1 LLM call each)
    # ========================================================================
    
    def evaluate_faithfulness(self, prepared: PreparedQA) -> float:
        """
        Evaluate faithfulness: Do answer claims follow from context?
        1 LLM call to verify ALL claims at once.
        """
        if not prepared.answer_claims:
            prepared.faithfulness_score = 1.0
            prepared.faithfulness_details = {"claims": [], "supported": 0, "total": 0}
            return 1.0
        
        # Format claims for prompt
        claims_text = "\n".join([
            f"CLAIM_{i+1}: {claim}" 
            for i, claim in enumerate(prepared.answer_claims)
        ])
        
        context_text = "\n\n".join(prepared.contexts)
        
        prompt = PROMPT_FAITHFULNESS.format(
            context=context_text,
            claims=claims_text
        )
        
        try:
            response = call_llm(prompt)
            
            # Parse results
            supported_count = 0
            claim_results = []
            
            for i, claim in enumerate(prepared.answer_claims):
                pattern = rf'CLAIM_{i+1}:\s*(SUPPORTED|NOT_SUPPORTED)'
                match = re.search(pattern, response, re.IGNORECASE)
                is_supported = match and 'NOT' not in match.group(1).upper() if match else False
                
                # Also check for simple line-by-line format
                if not match:
                    lines = response.strip().split('\n')
                    if i < len(lines):
                        is_supported = 'SUPPORTED' in lines[i].upper() and 'NOT' not in lines[i].upper()
                
                claim_results.append({
                    "claim": claim,
                    "supported": is_supported
                })
                if is_supported:
                    supported_count += 1
            
            score = supported_count / len(prepared.answer_claims) if prepared.answer_claims else 1.0
            prepared.faithfulness_score = score
            prepared.faithfulness_details = {
                "claims": claim_results,
                "supported": supported_count,
                "total": len(prepared.answer_claims)
            }
            return score
            
        except Exception as e:
            print(f"Error in evaluate_faithfulness: {e}")
            prepared.faithfulness_score = 0.0
            return 0.0
    
    def evaluate_context_recall(self, prepared: PreparedQA) -> float:
        """
        Evaluate context recall: Can reference claims be attributed to context?
        1 LLM call to verify ALL reference claims at once.
        """
        if not prepared.reference_claims:
            prepared.context_recall_score = 1.0
            prepared.context_recall_details = {"claims": [], "attributed": 0, "total": 0}
            return 1.0
        
        claims_text = "\n".join([
            f"CLAIM_{i+1}: {claim}" 
            for i, claim in enumerate(prepared.reference_claims)
        ])
        
        context_text = "\n\n".join(prepared.contexts)
        
        prompt = PROMPT_CONTEXT_RECALL.format(
            context=context_text,
            claims=claims_text
        )
        
        try:
            response = call_llm(prompt)
            
            attributed_count = 0
            claim_results = []
            
            for i, claim in enumerate(prepared.reference_claims):
                pattern = rf'CLAIM_{i+1}:\s*(ATTRIBUTED|NOT_ATTRIBUTED)'
                match = re.search(pattern, response, re.IGNORECASE)
                is_attributed = match and 'NOT' not in match.group(1).upper() if match else False
                
                if not match:
                    lines = response.strip().split('\n')
                    if i < len(lines):
                        is_attributed = 'ATTRIBUTED' in lines[i].upper() and 'NOT' not in lines[i].upper()
                
                claim_results.append({
                    "claim": claim,
                    "attributed": is_attributed
                })
                if is_attributed:
                    attributed_count += 1
            
            score = attributed_count / len(prepared.reference_claims) if prepared.reference_claims else 1.0
            prepared.context_recall_score = score
            prepared.context_recall_details = {
                "claims": claim_results,
                "attributed": attributed_count,
                "total": len(prepared.reference_claims)
            }
            return score
            
        except Exception as e:
            print(f"Error in evaluate_context_recall: {e}")
            prepared.context_recall_score = 0.0
            return 0.0
    
    def evaluate_context_precision(self, prepared: PreparedQA) -> float:
        """
        Evaluate context precision: Are contexts relevant and well-ranked?
        1 LLM call to evaluate ALL contexts at once.
        Uses mean precision@k formula.
        """
        if not prepared.contexts:
            prepared.context_precision_score = 0.0
            prepared.context_precision_details = {"contexts": [], "precision_at_k": []}
            return 0.0
        
        contexts_text = "\n\n".join([
            f"CHUNK_{i+1}:\n{ctx}" 
            for i, ctx in enumerate(prepared.contexts)
        ])
        
        prompt = PROMPT_CONTEXT_PRECISION.format(
            question=prepared.question,
            reference=prepared.reference,
            contexts=contexts_text
        )
        
        try:
            response = call_llm(prompt)
            
            relevance = []
            context_results = []
            
            for i, ctx in enumerate(prepared.contexts):
                pattern = rf'CHUNK_{i+1}:\s*(RELEVANT|NOT_RELEVANT)'
                match = re.search(pattern, response, re.IGNORECASE)
                is_relevant = match and 'NOT' not in match.group(1).upper() if match else False
                
                if not match:
                    lines = response.strip().split('\n')
                    if i < len(lines):
                        is_relevant = 'RELEVANT' in lines[i].upper() and 'NOT' not in lines[i].upper()
                
                relevance.append(1 if is_relevant else 0)
                context_results.append({
                    "context_idx": i,
                    "relevant": is_relevant
                })
            
            # Calculate mean precision@k
            precision_at_k = []
            relevant_so_far = 0
            for k, rel in enumerate(relevance, 1):
                relevant_so_far += rel
                if rel:  # Only count precision at positions where item is relevant
                    precision_at_k.append(relevant_so_far / k)
            
            score = np.mean(precision_at_k) if precision_at_k else 0.0
            prepared.context_precision_score = score
            prepared.context_precision_details = {
                "contexts": context_results,
                "precision_at_k": precision_at_k,
                "relevance_binary": relevance
            }
            return score
            
        except Exception as e:
            print(f"Error in evaluate_context_precision: {e}")
            prepared.context_precision_score = 0.0
            return 0.0
    
    def evaluate_answer_relevancy(self, prepared: PreparedQA) -> float:
        """
        Evaluate answer relevancy using reverse questions.
        0 LLM calls - uses embeddings only (reverse questions already generated).
        """
        if not self.embeddings or not prepared.reverse_questions:
            prepared.answer_relevancy_score = 0.0
            return 0.0
        
        try:
            # Get embeddings for original question and reverse questions
            original_embedding = self.embeddings.embed_query(prepared.question)
            reverse_embeddings = [
                self.embeddings.embed_query(q) 
                for q in prepared.reverse_questions
            ]
            
            # Calculate cosine similarities
            similarities = []
            for rev_emb in reverse_embeddings:
                sim = cosine_similarity(
                    [original_embedding], 
                    [rev_emb]
                )[0][0]
                similarities.append(sim)
            
            score = float(np.mean(similarities))
            prepared.answer_relevancy_score = max(0.0, min(1.0, score))
            return prepared.answer_relevancy_score
            
        except Exception as e:
            print(f"Error in evaluate_answer_relevancy: {e}")
            prepared.answer_relevancy_score = 0.0
            return 0.0
    
    # ========================================================================
    # MULTIMODAL METRICS (1 VLM call each)
    # ========================================================================
    
    def evaluate_multimodal_faithfulness(self, prepared: PreparedQA) -> float:
        """
        Evaluate multimodal faithfulness using VLM.
        1 VLM call - verifies ALL claims at once against text+image context.
        
        Same pattern as text faithfulness:
        - Pass all claims in one prompt
        - VLM returns SUPPORTED/NOT_SUPPORTED for each
        - Compute score as supported_count / total_claims
        """
        if not self.enable_multimodal or not prepared.context_chunks:
            prepared.multimodal_faithfulness_score = 0.0
            return 0.0
        
        # Check if there are any images
        has_images = any(
            chunk.get('image_path') and chunk.get('image_path') != 'null'
            for chunk in prepared.context_chunks
        )
        
        if not has_images:
            # Fall back to text-only faithfulness
            prepared.multimodal_faithfulness_score = prepared.faithfulness_score
            return prepared.faithfulness_score
        
        # Use extracted claims, or fall back to answer as single claim
        claims = prepared.answer_claims if prepared.answer_claims else [prepared.answer]
        
        claims_text = "\n".join([
            f"CLAIM_{i+1}: {claim}" 
            for i, claim in enumerate(claims)
        ])
        
        prompt = PROMPT_MULTIMODAL_FAITHFULNESS.format(
            question=prepared.question,
            answer=prepared.answer,
            claims=claims_text
        )
        
        try:
            response = call_vlm_interweaved(prompt, prepared.context_chunks)
            
            # Parse each claim result (same pattern as text faithfulness)
            supported_count = 0
            claim_results = []
            text_sources = 0
            image_sources = 0
            
            for i, claim in enumerate(claims):
                pattern = rf'CLAIM_{i+1}:\s*(SUPPORTED|NOT_SUPPORTED)\s*\|\s*SOURCE:\s*(TEXT|IMAGE|BOTH|NONE)'
                match = re.search(pattern, response, re.IGNORECASE)
                
                if match:
                    is_supported = 'NOT' not in match.group(1).upper()
                    source = match.group(2).upper()
                else:
                    # Fallback: simpler pattern
                    simple_pattern = rf'CLAIM_{i+1}:\s*(SUPPORTED|NOT_SUPPORTED)'
                    simple_match = re.search(simple_pattern, response, re.IGNORECASE)
                    is_supported = simple_match and 'NOT' not in simple_match.group(1).upper() if simple_match else False
                    source = "UNKNOWN"
                
                claim_results.append({
                    "claim": claim,
                    "supported": is_supported,
                    "source": source
                })
                
                if is_supported:
                    supported_count += 1
                    if source in ['TEXT', 'BOTH']:
                        text_sources += 1
                    if source in ['IMAGE', 'BOTH']:
                        image_sources += 1
            
            # Compute score: supported_count / total_claims
            score = supported_count / len(claims) if claims else 1.0
            
            # Parse summary details
            text_grounded = text_sources > 0 or 'TEXT_GROUNDED: YES' in response.upper()
            visual_grounded = image_sources > 0 or 'VISUAL_GROUNDED: YES' in response.upper()
            
            prepared.multimodal_faithfulness_score = score
            prepared.multimodal_details['faithfulness'] = {
                'score': score,
                'supported_count': supported_count,
                'total_claims': len(claims),
                'text_grounded': text_grounded,
                'visual_grounded': visual_grounded,
                'text_sources': text_sources,
                'image_sources': image_sources,
                'claim_results': claim_results
            }
            return score
            
        except Exception as e:
            print(f"Error in evaluate_multimodal_faithfulness: {e}")
            prepared.multimodal_faithfulness_score = 0.0
            return 0.0
    
    def evaluate_multimodal_relevance(self, prepared: PreparedQA) -> float:
        """
        Evaluate multimodal relevance using VLM.
        1 VLM call - generates reverse questions and evaluates context usage.
        
        Same pattern as answer_relevancy but with VLM for multimodal context:
        - Generate reverse questions from answer
        - Check if questions use text/image context
        - Compute relevance score
        """
        if not self.enable_multimodal or not prepared.context_chunks:
            prepared.multimodal_relevance_score = 0.0
            return 0.0
        
        has_images = any(
            chunk.get('image_path') and chunk.get('image_path') != 'null'
            for chunk in prepared.context_chunks
        )
        
        if not has_images:
            prepared.multimodal_relevance_score = prepared.answer_relevancy_score
            return prepared.answer_relevancy_score
        
        prompt = PROMPT_MULTIMODAL_RELEVANCE.format(
            answer=prepared.answer,
            num_questions=self.num_reverse_questions
        )
        
        try:
            response = call_vlm_interweaved(prompt, prepared.context_chunks)
            
            # Parse generated questions
            generated_questions = []
            uses_text_count = 0
            uses_image_count = 0
            
            for i in range(1, self.num_reverse_questions + 1):
                pattern = rf'Q{i}:\s*(.+?)\s*\|\s*USES:\s*(TEXT|IMAGE|BOTH)'
                match = re.search(pattern, response, re.IGNORECASE)
                
                if match:
                    question = match.group(1).strip()
                    source = match.group(2).upper()
                    generated_questions.append({
                        'question': question,
                        'uses': source
                    })
                    if source in ['TEXT', 'BOTH']:
                        uses_text_count += 1
                    if source in ['IMAGE', 'BOTH']:
                        uses_image_count += 1
            
            # Parse relevance score from VLM
            score_match = re.search(r'RELEVANCE_SCORE:\s*([\d.]+)', response)
            vlm_score = float(score_match.group(1)) if score_match else 0.5
            vlm_score = max(0.0, min(1.0, vlm_score))
            
            # If we have embeddings and generated questions, compute embedding-based score
            embedding_score = 0.0
            if self.embeddings and generated_questions:
                try:
                    original_emb = self.embeddings.embed_query(prepared.question)
                    similarities = []
                    for gq in generated_questions:
                        q_emb = self.embeddings.embed_query(gq['question'])
                        sim = cosine_similarity([original_emb], [q_emb])[0][0]
                        similarities.append(sim)
                    embedding_score = float(np.mean(similarities)) if similarities else 0.0
                except Exception:
                    embedding_score = 0.0
            
            # Combine VLM score and embedding score (if available)
            if embedding_score > 0:
                score = 0.5 * vlm_score + 0.5 * embedding_score
            else:
                score = vlm_score
            
            uses_text = uses_text_count > 0 or 'USES_TEXT: YES' in response.upper()
            uses_images = uses_image_count > 0 or 'USES_IMAGES: YES' in response.upper()
            
            prepared.multimodal_relevance_score = score
            prepared.multimodal_details['relevance'] = {
                'score': score,
                'vlm_score': vlm_score,
                'embedding_score': embedding_score,
                'uses_text': uses_text,
                'uses_images': uses_images,
                'generated_questions': generated_questions
            }
            return score
            
        except Exception as e:
            print(f"Error in evaluate_multimodal_relevance: {e}")
            prepared.multimodal_relevance_score = 0.0
            return 0.0
    
    # ========================================================================
    # ADDITIONAL METRICS (harmonized with metrics.py)
    # ========================================================================
    
    def evaluate_context_necessity(self, question: str, answer: str, context: str) -> Dict:
        """
        Measures if the question REQUIRES context to answer (anti-parametric bias).
        
        High score (1.0) = Context is essential (good for RAG testing)
        Low score (0.0) = Answerable from parametric knowledge (bad for RAG testing)
        
        1 LLM call to answer without context + 1 LLM call to verify.
        """
        try:
            # Step 1: Try to answer WITHOUT context
            prompt_without = PROMPT_CONTEXT_NECESSITY_WITHOUT.format(question=question)
            answer_without = call_llm(prompt_without)
            
            # Check if model refused
            if "CANNOT_ANSWER" in answer_without.upper():
                return {
                    "context_necessity_score": 1.0,
                    "without_context_correct": False,
                    "with_context_correct": True,
                    "answer_without_context": answer_without[:200],
                    "explanation": "Model could not answer without context - context is essential"
                }
            
            # Step 2: Verify if answer without context is correct
            prompt_verify = PROMPT_CONTEXT_NECESSITY_VERIFY.format(
                ground_truth=answer,
                model_answer=answer_without
            )
            verify_response = call_llm(prompt_verify)
            verify_upper = verify_response.upper()
            
            if "MATCH: YES" in verify_upper:
                return {
                    "context_necessity_score": 0.0,
                    "without_context_correct": True,
                    "with_context_correct": True,
                    "answer_without_context": answer_without[:200],
                    "explanation": "Model answered correctly without context - question tests parametric knowledge"
                }
            elif "MATCH: PARTIAL" in verify_upper:
                return {
                    "context_necessity_score": 0.5,
                    "without_context_correct": False,
                    "with_context_correct": True,
                    "answer_without_context": answer_without[:200],
                    "explanation": "Model partially answered without context - context adds value"
                }
            else:
                return {
                    "context_necessity_score": 0.9,
                    "without_context_correct": False,
                    "with_context_correct": True,
                    "answer_without_context": answer_without[:200],
                    "explanation": "Model answered incorrectly without context - context is necessary"
                }
                
        except Exception as e:
            logging.error(f"Error in context_necessity: {e}")
            return {
                "context_necessity_score": 0.5,
                "without_context_correct": None,
                "with_context_correct": None,
                "answer_without_context": "",
                "explanation": f"Error: {str(e)}"
            }
    
    def batch_evaluate_context_necessity(self, qa_items: List[Dict]) -> List[Dict]:
        """Batch evaluation of context necessity using parallel calls."""
        # Phase 1: Batch "answer without context"
        prompts_without = [
            PROMPT_CONTEXT_NECESSITY_WITHOUT.format(question=item['question'])
            for item in qa_items
        ]
        
        print(f"  ⚡ Phase 1: Answering {len(prompts_without)} questions without context...")
        answers_without = batch_call_llm(prompts_without, show_progress=False)
        
        # Phase 2: Batch verification
        verify_prompts = []
        verify_indices = []
        results = [None] * len(qa_items)
        
        for i, (item, answer_without) in enumerate(zip(qa_items, answers_without)):
            if answer_without.startswith("ERROR:"):
                results[i] = {
                    "context_necessity_score": 0.5,
                    "without_context_correct": None,
                    "explanation": f"Error: {answer_without}"
                }
            elif "CANNOT_ANSWER" in answer_without.upper():
                results[i] = {
                    "context_necessity_score": 1.0,
                    "without_context_correct": False,
                    "answer_without_context": answer_without[:200],
                    "explanation": "Model could not answer without context"
                }
            else:
                verify_prompts.append(PROMPT_CONTEXT_NECESSITY_VERIFY.format(
                    ground_truth=item['answer'],
                    model_answer=answer_without
                ))
                verify_indices.append(i)
        
        if verify_prompts:
            print(f"  ⚡ Phase 2: Verifying {len(verify_prompts)} answers...")
            verify_responses = batch_call_llm(verify_prompts, show_progress=False)
            
            for idx, verify_content in zip(verify_indices, verify_responses):
                answer_without = answers_without[idx]
                verify_upper = verify_content.upper() if verify_content else ""
                
                if "MATCH: YES" in verify_upper:
                    results[idx] = {
                        "context_necessity_score": 0.0,
                        "without_context_correct": True,
                        "answer_without_context": answer_without[:200],
                        "explanation": "Answered correctly without context"
                    }
                elif "MATCH: PARTIAL" in verify_upper:
                    results[idx] = {
                        "context_necessity_score": 0.5,
                        "without_context_correct": False,
                        "answer_without_context": answer_without[:200],
                        "explanation": "Partially answered without context"
                    }
                else:
                    results[idx] = {
                        "context_necessity_score": 0.9,
                        "without_context_correct": False,
                        "answer_without_context": answer_without[:200],
                        "explanation": "Answered incorrectly without context"
                    }
        
        return results
    
    def evaluate_multihop_reasoning(self, question: str, answer: str, contexts: List[str]) -> Dict:
        """
        Evaluate if QA requires multi-hop reasoning (combining multiple facts).
        
        Returns: hop_count, reasoning_score, bridge_entity
        """
        contexts_str = "\n".join(contexts) if isinstance(contexts, list) else contexts
        
        prompt = PROMPT_MULTIHOP_REASONING.format(
            contexts=contexts_str,
            question=question,
            answer=answer
        )
        
        try:
            response = call_llm(prompt)
            
            hop_match = re.search(r'HOP_COUNT:\s*(\d+)', response)
            score_match = re.search(r'REASONING_SCORE:\s*([\d.]+)', response)
            bridge_match = re.search(r'BRIDGE_ENTITY:\s*(.+?)(?:\n|$)', response)
            
            hop_count = int(hop_match.group(1)) if hop_match else 1
            reasoning_score = float(score_match.group(1)) if score_match else 0.5
            reasoning_score = min(1.0, max(0.0, reasoning_score))
            bridge_entity = bridge_match.group(1).strip() if bridge_match else "None"
            
            return {
                "hop_count": hop_count,
                "reasoning_score": reasoning_score,
                "bridge_entity": bridge_entity
            }
        except Exception as e:
            logging.error(f"Error in multihop_reasoning: {e}")
            return {"hop_count": 1, "reasoning_score": 0.0, "bridge_entity": "Error"}
    
    def batch_evaluate_multihop_reasoning(self, qa_items: List[Dict]) -> List[Dict]:
        """Batch evaluation of multihop reasoning."""
        prompts = []
        for item in qa_items:
            contexts_str = "\n".join(item['contexts']) if isinstance(item['contexts'], list) else item['contexts']
            prompts.append(PROMPT_MULTIHOP_REASONING.format(
                contexts=contexts_str,
                question=item['question'],
                answer=item['answer']
            ))
        
        print(f"  ⚡ Batch evaluating {len(prompts)} multihop questions...")
        responses = batch_call_llm(prompts, show_progress=False)
        
        results = []
        for response in responses:
            if response.startswith("ERROR:"):
                results.append({"hop_count": 1, "reasoning_score": 0.0, "bridge_entity": "Error"})
                continue
            
            hop_match = re.search(r'HOP_COUNT:\s*(\d+)', response)
            score_match = re.search(r'REASONING_SCORE:\s*([\d.]+)', response)
            bridge_match = re.search(r'BRIDGE_ENTITY:\s*(.+?)(?:\n|$)', response)
            
            results.append({
                "hop_count": int(hop_match.group(1)) if hop_match else 1,
                "reasoning_score": min(1.0, max(0.0, float(score_match.group(1)))) if score_match else 0.5,
                "bridge_entity": bridge_match.group(1).strip() if bridge_match else "None"
            })
        
        return results
    
    def evaluate_visual_dependency(self, question: str, text_contexts: List[str], 
                                      has_image_description: bool = True) -> float:
        """
        Evaluate if visual content is essential for answering the question.
        
        Returns:
            1.0 = Strongly requires visual (image references, diagrams, visual elements)
            0.5 = Partially visual (enhanced by images but answerable from text)
            0.0 = Text-only sufficient
        """
        contexts_str = "\n".join(text_contexts) if isinstance(text_contexts, list) else text_contexts
        
        # Enhanced prompt that checks for visual references in question/answer
        prompt = f"""Analyze if this question REQUIRES visual information (images, diagrams, figures) to answer properly.

CONTEXT:
{contexts_str}

QUESTION: {question}

Evaluate:
1. Does the question ask about visual elements (shapes, colors, layouts, diagrams, figures)?
2. Would seeing an image provide essential information not captured in text?
3. Does the context describe visual content that needs to be seen to understand?

RESPOND WITH ONE OF:
- VISUAL_ESSENTIAL: Question cannot be properly answered without seeing visual content
- VISUAL_HELPFUL: Visual content enhances understanding but text is sufficient
- TEXT_SUFFICIENT: Can be fully answered from text alone

YOUR VERDICT:"""
        
        try:
            response = call_llm(prompt)
            response_upper = response.upper()
            
            if "VISUAL_ESSENTIAL" in response_upper:
                return 1.0
            elif "VISUAL_HELPFUL" in response_upper:
                return 0.5
            else:
                return 0.0
        except Exception as e:
            logging.error(f"Error in visual_dependency: {e}")
            return 0.0
    
    def evaluate_semantic_diversity(self, questions: List[str]) -> float:
        """
        Calculate diversity of questions using embedding similarity.
        
        High score = Diverse questions (good)
        Low score = Repetitive questions (bad)
        """
        if not self.embeddings or not questions or len(questions) < 2:
            return 0.0
        
        try:
            embeddings = [self.embeddings.embed_query(q) for q in questions]
            matrix = np.array(embeddings)
            
            sim_matrix = cosine_similarity(matrix)
            np.fill_diagonal(sim_matrix, np.nan)
            avg_similarity = np.nanmean(sim_matrix)
            
            diversity_score = 1 - avg_similarity
            return max(0.0, min(1.0, diversity_score))
        except Exception as e:
            logging.error(f"Error in semantic_diversity: {e}")
            return 0.0
    
    def evaluate_domain_coverage(self, qa_data: List[Dict], corpus_chunks: List[Dict]) -> Dict:
        """
        Measure how well QA dataset covers the source corpus.
        
        Returns: chunk_coverage, file_coverage, topic_divergence
        """
        try:
            from scipy.stats import entropy
        except ImportError:
            return {"error": "scipy not installed"}
        
        # Build corpus index
        corpus_index = {}
        corpus_by_file = Counter()
        
        for chunk in corpus_chunks:
            key = (chunk.get('file_name'), str(chunk.get('chunk_id')))
            corpus_index[key] = chunk
            corpus_by_file[chunk.get('file_name', 'unknown')] += 1
        
        # Track covered chunks
        covered_chunks = set()
        qa_file_counts = Counter()
        
        for qa in qa_data:
            for chunk_ref in qa.get('chunks_added', []):
                key = (chunk_ref.get('file_name'), str(chunk_ref.get('chunk_id')))
                if key in corpus_index:
                    covered_chunks.add(key)
                qa_file_counts[chunk_ref.get('file_name')] += 1
        
        # Calculate metrics
        total_corpus = len(corpus_chunks)
        total_covered = len(covered_chunks)
        chunk_coverage = total_covered / total_corpus if total_corpus > 0 else 0.0
        
        files_in_corpus = set(corpus_by_file.keys())
        files_covered = set(f for f, _ in covered_chunks)
        file_coverage = len(files_covered) / len(files_in_corpus) if files_in_corpus else 0.0
        
        # Jensen-Shannon divergence
        all_files = sorted(files_in_corpus)
        corpus_dist = np.array([corpus_by_file.get(f, 0) for f in all_files], dtype=float)
        corpus_dist = corpus_dist / corpus_dist.sum() if corpus_dist.sum() > 0 else corpus_dist
        
        qa_dist = np.array([qa_file_counts.get(f, 0) for f in all_files], dtype=float)
        qa_dist = qa_dist / qa_dist.sum() if qa_dist.sum() > 0 else qa_dist
        
        eps = 1e-10
        corpus_dist = (corpus_dist + eps) / (corpus_dist + eps).sum()
        qa_dist = (qa_dist + eps) / (qa_dist + eps).sum()
        
        m = 0.5 * (corpus_dist + qa_dist)
        js_divergence = 0.5 * (entropy(corpus_dist, m) + entropy(qa_dist, m))
        
        return {
            "chunk_coverage": chunk_coverage,
            "file_coverage": file_coverage,
            "chunks_covered": total_covered,
            "chunks_total": total_corpus,
            "uncovered_chunks": total_corpus - total_covered,
            "topic_divergence_js": float(js_divergence)
        }
    
    # ========================================================================
    # FULL EVALUATION PIPELINE
    # ========================================================================
    
    def evaluate_single(self, qa_item: Dict, enable_multimodal: bool = None) -> Dict:
        """
        Evaluate a single QA pair with all metrics.
        
        Total LLM calls: 4-6 per QA pair
        - 1 preparation call
        - 1 faithfulness call
        - 1 context_recall call  
        - 1 context_precision call
        - 0 answer_relevancy call (embeddings only)
        - 1 multimodal_faithfulness call (if images)
        - 1 multimodal_relevance call (if images)
        """
        if enable_multimodal is None:
            enable_multimodal = self.enable_multimodal
        
        # Extract fields
        question = qa_item.get('question', '')
        answer = qa_item.get('answer', '')
        reference = qa_item.get('ground_truth', qa_item.get('answer', ''))
        contexts = qa_item.get('contexts', [])
        context_chunks = qa_item.get('context_chunks', [])
        
        # Step 1: Prepare (1 call)
        prepared = self.prepare_qa(question, answer, reference, contexts, context_chunks)
        
        # Step 2: Evaluate each metric (1 call each)
        self.evaluate_faithfulness(prepared)
        self.evaluate_context_recall(prepared)
        self.evaluate_context_precision(prepared)
        self.evaluate_answer_relevancy(prepared)
        
        # Step 3: Multimodal metrics (1-2 VLM calls if images present)
        if enable_multimodal and context_chunks:
            self.evaluate_multimodal_faithfulness(prepared)
            self.evaluate_multimodal_relevance(prepared)
        
        # Extract metadata from qa_item
        metadata = qa_item.get('metadata', {})
        
        # Determine is_multihop and is_multimodal
        hop_count = metadata.get('hop_count', 0)
        is_multihop = hop_count > 0
        
        # Check for multimodal (has images in context)
        is_multimodal = False
        for chunk in context_chunks:
            if has_image_in_chunk(chunk):
                is_multimodal = True
                break
        
        # Return results with complete metadata
        return {
            'question': question,
            'answer': answer,
            'faithfulness': prepared.faithfulness_score,
            'context_recall': prepared.context_recall_score,
            'context_precision': prepared.context_precision_score,
            'answer_relevancy': prepared.answer_relevancy_score,
            'multimodal_faithfulness': prepared.multimodal_faithfulness_score,
            'multimodal_relevance': prepared.multimodal_relevance_score,
            # Additional fields for visualization/analysis
            'hop_count': hop_count,
            'is_multihop': is_multihop,
            'is_multimodal': is_multimodal,
            'chunk_id': metadata.get('chunk_id'),
            'context_status': metadata.get('context_status'),
            'depth_reached': metadata.get('depth_reached'),
            # Dataset metadata
            'expert_persona': qa_item.get('expert_persona') or metadata.get('expert_persona'),
            'domain': qa_item.get('domain') or metadata.get('domain'),
            'concept_hops_question': prepared.concept_hops_question,
            'details': {
                'answer_claims': prepared.answer_claims,
                'reference_claims': prepared.reference_claims,
                'reverse_questions': prepared.reverse_questions,
                'faithfulness': prepared.faithfulness_details,
                'context_recall': prepared.context_recall_details,
                'context_precision': prepared.context_precision_details,
                'multimodal': prepared.multimodal_details
            }
        }
    
    def evaluate_batch(self, qa_data: List[Dict], 
                       enable_multimodal: bool = None,
                       show_progress: bool = True) -> List[Dict]:
        """
        Evaluate a batch of QA pairs with parallel processing.
        """
        if enable_multimodal is None:
            enable_multimodal = self.enable_multimodal
        
        results = []
        total = len(qa_data)
        
        # Use ThreadPoolExecutor for parallel evaluation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.evaluate_single, qa, enable_multimodal): i
                for i, qa in enumerate(qa_data)
            }
            
            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                except Exception as e:
                    print(f"Error evaluating item {idx}: {e}")
                    results.append((idx, {
                        'question': qa_data[idx].get('question', ''),
                        'faithfulness': 0.0,
                        'context_recall': 0.0,
                        'context_precision': 0.0,
                        'answer_relevancy': 0.0,
                        'multimodal_faithfulness': 0.0,
                        'multimodal_relevance': 0.0,
                        'error': str(e)
                    }))
                
                completed += 1
                if show_progress and completed % 10 == 0:
                    print(f"  Evaluated {completed}/{total} QA pairs...")
        
        # Sort by original index
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
    
    def _count_concept_hops(self, concept_hops_str: str) -> int:
        """Count the number of concept hops from concept hops string.
        
        Concept hops format: concept1 --> concept2 --> concept3
        Number of hops = number of arrows (-->) = number of concepts - 1
        """
        if not concept_hops_str or not concept_hops_str.strip():
            return 0
        # Count occurrences of arrow pattern
        return concept_hops_str.count('-->')
    
    def compute_aggregate_scores(self, results: List[Dict]) -> Dict:
        """Compute aggregate scores from batch results."""
        if not results:
            return {}
        
        valid_results = [r for r in results if 'error' not in r]
        
        scores = {
            'faithfulness': np.mean([r['faithfulness'] for r in valid_results]),
            'context_recall': np.mean([r['context_recall'] for r in valid_results]),
            'context_precision': np.mean([r['context_precision'] for r in valid_results]),
            'answer_relevancy': np.mean([r['answer_relevancy'] for r in valid_results]),
            'multimodal_faithfulness': np.mean([r['multimodal_faithfulness'] for r in valid_results]),
            'multimodal_relevance': np.mean([r['multimodal_relevance'] for r in valid_results]),
            'total_evaluated': len(valid_results),
            'total_errors': len(results) - len(valid_results)
        }
        
        # Calculate average concept hops for questions
        concept_hops_list = []
        for r in valid_results:
            hops_str = r.get('concept_hops_question', '')
            if hops_str:
                hops_count = self._count_concept_hops(hops_str)
                concept_hops_list.append(hops_count)
        
        if concept_hops_list:
            scores['avg_concept_hops_question'] = float(np.mean(concept_hops_list))
            scores['concept_hops_question_count'] = len(concept_hops_list)
        else:
            scores['avg_concept_hops_question'] = 0.0
            scores['concept_hops_question_count'] = 0
        
        # Filter out zero multimodal scores (items without images)
        mm_faithfulness = [r['multimodal_faithfulness'] for r in valid_results if r['multimodal_faithfulness'] > 0]
        mm_relevance = [r['multimodal_relevance'] for r in valid_results if r['multimodal_relevance'] > 0]
        
        if mm_faithfulness:
            scores['multimodal_faithfulness_valid'] = np.mean(mm_faithfulness)
            scores['multimodal_items_count'] = len(mm_faithfulness)
        
        if mm_relevance:
            scores['multimodal_relevance_valid'] = np.mean(mm_relevance)
        
        return scores


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def run_dataset_qa_evaluation(qa_data: List[Dict],
                              output_path: str = None,
                              enable_multimodal: bool = True,
                              max_workers: int = 8) -> Dict:
    """
    Quality assurance evaluation for GOLD-STANDARD DATASET CREATION.
    
    Only runs metrics that matter for dataset creation:
    - Faithfulness: Are gold answers grounded in source context?
    - Answer Relevancy: Do answers address the questions?
    - Context Precision: Are all source chunks relevant? (optional)
    - Multimodal metrics: For image-containing contexts
    
    SKIPS Context Recall (redundant when answer = reference)
    
    Args:
        qa_data: List of QA dicts with question, answer, contexts
        output_path: Optional path to save results
        enable_multimodal: Whether to run VLM metrics for image contexts
        max_workers: Number of parallel workers
    
    Returns:
        Dict with quality scores and flagged issues
    """
    print("=" * 60)
    print("DATASET QUALITY ASSURANCE EVALUATION")
    print("=" * 60)
    print(f"  Total QA pairs: {len(qa_data)}")
    print(f"  Mode: Gold-standard dataset creation")
    print(f"  Key metrics: Faithfulness, Answer Relevancy")
    print()
    
    evaluator = OptimizedMetricsEvaluator(
        enable_multimodal=enable_multimodal,
        max_workers=max_workers
    )
    
    results = []
    issues = []
    
    for i, qa in enumerate(qa_data):
        if i % 10 == 0:
            print(f"  Evaluating {i+1}/{len(qa_data)}...")
        
        # Prepare
        question = qa.get('question', '')
        answer = qa.get('answer', '')
        contexts = qa.get('contexts', [])
        context_chunks = qa.get('context_chunks', [])
        
        prepared = evaluator.prepare_qa(question, answer, answer, contexts, context_chunks)
        
        # Only run critical metrics for dataset creation
        evaluator.evaluate_faithfulness(prepared)
        evaluator.evaluate_answer_relevancy(prepared)
        evaluator.evaluate_context_precision(prepared)
        
        # Multimodal if applicable
        if enable_multimodal and context_chunks:
            evaluator.evaluate_multimodal_faithfulness(prepared)
        
        result = {
            'idx': i,
            'question': question[:100],
            'faithfulness': prepared.faithfulness_score,
            'answer_relevancy': prepared.answer_relevancy_score,
            'context_precision': prepared.context_precision_score,
            'multimodal_faithfulness': prepared.multimodal_faithfulness_score,
            'concept_hops_question': prepared.concept_hops_question,
        }
        results.append(result)
        
        # Flag potential issues
        if prepared.faithfulness_score < 0.8:
            issues.append({
                'idx': i,
                'issue': 'LOW_FAITHFULNESS',
                'score': prepared.faithfulness_score,
                'question': question[:100],
                'unsupported_claims': [
                    c['claim'] for c in prepared.faithfulness_details.get('claims', [])
                    if not c.get('supported', True)
                ]
            })
        
        if prepared.answer_relevancy_score < 0.7:
            issues.append({
                'idx': i,
                'issue': 'LOW_RELEVANCY',
                'score': prepared.answer_relevancy_score,
                'question': question[:100],
            })
    
    # Compute aggregates
    avg_faithfulness = np.mean([r['faithfulness'] for r in results])
    avg_relevancy = np.mean([r['answer_relevancy'] for r in results])
    avg_precision = np.mean([r['context_precision'] for r in results])
    
    # Compute average concept hops
    concept_hops_list = []
    for r in results:
        hops_str = r.get('concept_hops_question', '')
        if hops_str:
            hops_count = evaluator._count_concept_hops(hops_str)
            concept_hops_list.append(hops_count)
    avg_concept_hops = np.mean(concept_hops_list) if concept_hops_list else 0.0
    
    print("\n" + "=" * 60)
    print("DATASET QUALITY SUMMARY")
    print("=" * 60)
    print(f"  📊 Average Faithfulness:     {avg_faithfulness:.3f}")
    print(f"  📊 Average Answer Relevancy: {avg_relevancy:.3f}")
    print(f"  📊 Average Context Precision: {avg_precision:.3f}")
    if concept_hops_list:
        print(f"  📊 Average Concept Hops:     {avg_concept_hops:.2f} ({len(concept_hops_list)} questions)")
    print(f"\n  ⚠️  Issues found: {len(issues)}")
    
    if issues:
        print("\n  Issues breakdown:")
        low_faith = [i for i in issues if i['issue'] == 'LOW_FAITHFULNESS']
        low_rel = [i for i in issues if i['issue'] == 'LOW_RELEVANCY']
        print(f"    - Low faithfulness: {len(low_faith)} QA pairs (answer not grounded)")
        print(f"    - Low relevancy:    {len(low_rel)} QA pairs (Q&A mismatch)")
    
    output = {
        'mode': 'dataset_creation_qa',
        'total_qa_pairs': len(qa_data),
        'aggregate_scores': {
            'faithfulness': float(avg_faithfulness),
            'answer_relevancy': float(avg_relevancy),
            'context_precision': float(avg_precision),
            'avg_concept_hops_question': float(avg_concept_hops),
            'concept_hops_question_count': len(concept_hops_list),
        },
        'issues': issues,
        'detailed_results': results
    }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
    
    return output


def run_optimized_evaluation(qa_data: List[Dict], 
                             output_path: str = None,
                             enable_multimodal: bool = True,
                             max_workers: int = 8) -> Dict:
    """
    Run optimized RAGAS-style evaluation on QA data.
    
    Args:
        qa_data: List of QA dicts with question, answer, contexts, etc.
        output_path: Optional path to save detailed results
        enable_multimodal: Whether to run multimodal metrics
        max_workers: Number of parallel workers
    
    Returns:
        Dict with aggregate scores and detailed results
    """
    print("=" * 60)
    print("OPTIMIZED METRICS EVALUATION")
    print("=" * 60)
    print(f"  Total QA pairs: {len(qa_data)}")
    print(f"  Multimodal: {'Enabled' if enable_multimodal else 'Disabled'}")
    print(f"  Max workers: {max_workers}")
    print()
    
    evaluator = OptimizedMetricsEvaluator(
        enable_multimodal=enable_multimodal,
        max_workers=max_workers
    )
    
    print("📊 Evaluating metrics (4-6 LLM calls per QA pair)...")
    results = evaluator.evaluate_batch(qa_data, show_progress=True)
    
    print("\n📈 Computing aggregate scores...")
    aggregate = evaluator.compute_aggregate_scores(results)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Faithfulness:          {aggregate.get('faithfulness', 0):.3f}")
    print(f"  Context Recall:        {aggregate.get('context_recall', 0):.3f}")
    print(f"  Context Precision:     {aggregate.get('context_precision', 0):.3f}")
    print(f"  Answer Relevancy:      {aggregate.get('answer_relevancy', 0):.3f}")
    if aggregate.get('multimodal_items_count', 0) > 0:
        print(f"  Multimodal Faithfulness: {aggregate.get('multimodal_faithfulness_valid', 0):.3f} ({aggregate['multimodal_items_count']} items)")
        print(f"  Multimodal Relevance:    {aggregate.get('multimodal_relevance_valid', 0):.3f}")
    if aggregate.get('concept_hops_question_count', 0) > 0:
        print(f"  Avg Concept Hops:        {aggregate.get('avg_concept_hops_question', 0):.2f} ({aggregate['concept_hops_question_count']} questions)")
    print(f"\n  Total evaluated: {aggregate.get('total_evaluated', 0)}")
    print(f"  Errors: {aggregate.get('total_errors', 0)}")
    
    output = {
        'aggregate_scores': aggregate,
        'detailed_results': results
    }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
    
    return output


# ============================================================================
# INTEGRATION WITH MAIN PIPELINE
# ============================================================================

def transform_qa_for_evaluation(raw_data: List[Dict]) -> List[Dict]:
    """
    Transform qa_multihop_pass.json / qa_deduplicated.json format to evaluation format.
    
    Input format (from pipeline):
        - chunk_id, original_chunk, final_context, context_chunks, context_status, 
        - depth_reached, chunks_added, expert_persona, domain, question, answer, etc.
    
    Output format (for evaluation):
        - question, answer, contexts (list), ground_truth, context_chunks
    """
    transformed = []
    for item in raw_data:
        context_chunks = item.get("context_chunks", [])
        
        # Extract text contexts
        if context_chunks:
            contexts = [chunk.get("content", "") for chunk in context_chunks if chunk.get("content")]
        else:
            contexts = [item.get("final_context", item.get("original_chunk", ""))]
        
        transformed.append({
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "contexts": contexts,
            "ground_truth": item.get("answer", ""),  # Use answer as ground_truth
            "context_chunks": context_chunks,  # Keep for multimodal evaluation
            "expert_persona": item.get("expert_persona"),  # Pass through for reporting
            "domain": item.get("domain"),  # Pass through for reporting
            "metadata": {
                "chunk_id": item.get("chunk_id"),
                "hop_count": item.get("hop_count", 0),
                "context_status": item.get("context_status"),
                "depth_reached": item.get("depth_reached"),
                "expert_persona": item.get("expert_persona"),
                "domain": item.get("domain"),
            }
        })
    return transformed


def run_optimized_pipeline_evaluation(
    qa_file: str,
    output_dir: str = None,
    corpus_path: str = None,
    enable_multimodal: bool = True,
    max_workers: int = 8,
    sample_size: int = None,
    run_context_necessity: bool = True
) -> Dict:
    """
    Comprehensive evaluation with ALL metrics (harmonized with metrics.py).
    
    Drop-in replacement for metrics.run_subset_evaluation() with optimized implementation.
    
    METRICS EVALUATED:
    1. Faithfulness - Answer grounded in context?
    2. Answer Relevancy - Answer addresses question?
    3. Context Precision - Retrieved chunks relevant?
    4. Context Recall - Context contains reference info?
    5. Multimodal Faithfulness - Answer grounded in text+images?
    6. Multimodal Relevance - Answer uses multimodal context?
    7. Context Necessity - Requires context to answer?
    8. Semantic Diversity - Questions diverse?
    9. Domain Coverage - Corpus coverage?
    10. Multihop Reasoning - Multi-step reasoning quality?
    11. Visual Dependency - Needs image to answer?
    
    Args:
        qa_file: Path to qa_deduplicated.json or qa_multihop_pass.json
        output_dir: Directory to save results
        corpus_path: Path to chunks.json for domain coverage
        enable_multimodal: Whether to run VLM-based multimodal metrics
        max_workers: Parallel workers for evaluation
        sample_size: Limit evaluation to N samples (None = all)
        run_context_necessity: Whether to run context necessity (expensive)
    
    Returns:
        Dict with all metrics, aggregate scores, and detailed results
    """
    import os
    import random
    
    # Load QA data
    print(f"📂 Loading QA data from {qa_file}...")
    with open(qa_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    original_count = len(raw_data)
    
    # Extract expert_persona and domain from QA data (should be consistent across all)
    expert_persona = None
    domain = None
    if raw_data:
        expert_persona = raw_data[0].get('expert_persona')
        domain = raw_data[0].get('domain')
    
    # Sample if requested
    if sample_size and sample_size < len(raw_data):
        raw_data = random.sample(raw_data, sample_size)
        print(f"📊 Sampled {sample_size}/{original_count} QA pairs for evaluation")
    
    print(f"✅ Loaded {len(raw_data)} QA pairs")
    if expert_persona:
        print(f"   Expert Persona: {expert_persona}")
    if domain:
        print(f"   Domain: {domain}")
    
    # Initialize results
    results = {
        'ragas_metrics': {},
        'context_necessity': {},
        'domain_coverage': {},
        'multihop_metrics': {},
        'multimodal_metrics': {},
        'dataset_health': {},
        'subset_statistics': {},
        'dataset_info': {
            'expert_persona': expert_persona,
            'domain': domain,
            'total_qa_pairs_generated': original_count,
            'total_qa_pairs_evaluated': len(raw_data),
            'sampled': sample_size is not None and sample_size < original_count
        }
    }
    
    # Transform to evaluation format
    qa_data = transform_qa_for_evaluation(raw_data)
    
    # Initialize evaluator
    evaluator = OptimizedMetricsEvaluator(
        enable_multimodal=enable_multimodal,
        max_workers=max_workers
    )
    
    # ==========================================
    # 1. CORE RAGAS METRICS (optimized batch)
    # ==========================================
    print("\n" + "=" * 60)
    print("RAGAS-STYLE METRICS (Optimized)")
    print("=" * 60)
    
    print(f"📊 Evaluating {len(qa_data)} QA pairs (4-6 LLM calls each)...")
    batch_results = evaluator.evaluate_batch(qa_data, show_progress=True)
    aggregate = evaluator.compute_aggregate_scores(batch_results)
    
    results['ragas_metrics'] = {
        'faithfulness': float(aggregate.get('faithfulness', 0)),
        'answer_relevancy': float(aggregate.get('answer_relevancy', 0)),
        'context_precision': float(aggregate.get('context_precision', 0)),
        'context_recall': float(aggregate.get('context_recall', 0)),
        'items_evaluated': aggregate.get('total_evaluated', 0)
    }
    
    if aggregate.get('multimodal_items_count', 0) > 0:
        results['ragas_metrics']['multimodal_faithfulness'] = float(aggregate.get('multimodal_faithfulness_valid', 0))
        results['ragas_metrics']['multimodal_relevance'] = float(aggregate.get('multimodal_relevance_valid', 0))
        results['ragas_metrics']['multimodal_items'] = aggregate.get('multimodal_items_count', 0)
    
    # Store aggregate scores including concept hops
    results['aggregate_scores'] = {
        'avg_concept_hops_question': float(aggregate.get('avg_concept_hops_question', 0)),
        'concept_hops_question_count': aggregate.get('concept_hops_question_count', 0)
    }
    
    # Store initial batch results (will be enriched with additional metrics later)
    results['detailed_results'] = batch_results
    
    # ==========================================
    # 2. SUBSET STATISTICS (with intersection)
    # ==========================================
    multihop_count = 0
    multimodal_count = 0
    both_count = 0
    
    for item in raw_data:
        hop_count = item.get('hop_count', 0)
        chunks_added = item.get('chunks_added', [])
        is_multihop = hop_count > 0 or (isinstance(chunks_added, list) and len(chunks_added) > 1)
        
        is_multimodal = any(
            has_image_in_chunk(c) for c in item.get('context_chunks', [])
        )
        
        if is_multihop:
            multihop_count += 1
        if is_multimodal:
            multimodal_count += 1
        if is_multihop and is_multimodal:
            both_count += 1
    
    results['subset_statistics'] = {
        'total_qa_pairs': len(raw_data),
        'multihop_count': multihop_count,
        'multimodal_count': multimodal_count,
        'multihop_multimodal_count': both_count,
        'multihop_only_count': multihop_count - both_count,
        'multimodal_only_count': multimodal_count - both_count,
        'text_only_count': len(raw_data) - multihop_count - multimodal_count + both_count,
        'avg_hop_count': float(np.mean([item.get('hop_count', 0) for item in raw_data]))
    }
    
    # ==========================================
    # 3. CONTEXT NECESSITY (anti-parametric bias)
    # ==========================================
    if run_context_necessity:
        print("\n" + "=" * 60)
        print("CONTEXT NECESSITY (Anti-Parametric Bias)")
        print("=" * 60)
        
        # Prepare batch items
        batch_items = []
        for qa in raw_data:
            context = qa.get('final_context', qa.get('original_chunk', ''))
            batch_items.append({
                'question': qa['question'],
                'answer': qa['answer'],
                'context': context
            })
        
        cn_results = evaluator.batch_evaluate_context_necessity(batch_items)
        necessity_scores = [r['context_necessity_score'] for r in cn_results]
        
        # Merge context_necessity into detailed_results
        for i, cn_result in enumerate(cn_results):
            if i < len(results['detailed_results']):
                results['detailed_results'][i]['context_necessity_score'] = cn_result['context_necessity_score']
                results['detailed_results'][i]['without_context_correct'] = cn_result.get('without_context_correct')
        
        results['context_necessity'] = {
            'avg_context_necessity_score': float(np.mean(necessity_scores)),
            'items_evaluated': len(necessity_scores),
            'items_answerable_without_context': sum(1 for r in cn_results if r.get('without_context_correct')),
            'score_distribution': {
                'high (0.8-1.0)': sum(1 for s in necessity_scores if s >= 0.8),
                'moderate (0.5-0.8)': sum(1 for s in necessity_scores if 0.5 <= s < 0.8),
                'low (0.0-0.5)': sum(1 for s in necessity_scores if s < 0.5)
            }
        }
        
        print(f"  Average Context Necessity: {results['context_necessity']['avg_context_necessity_score']:.3f}")
    
    # ==========================================
    # 4. MULTIHOP REASONING (on multihop subset)
    # ==========================================
    multihop_items = [qa for qa in raw_data if qa.get('hop_count', 0) > 0]
    if multihop_items:
        print("\n" + "=" * 60)
        print(f"MULTIHOP REASONING ({len(multihop_items)} items)")
        print("=" * 60)
        
        mh_batch = [{
            'question': qa['question'],
            'answer': qa['answer'],
            'contexts': [qa.get('final_context', qa.get('original_chunk', ''))]
        } for qa in multihop_items]
        
        mh_results = evaluator.batch_evaluate_multihop_reasoning(mh_batch)
        
        # Merge multihop reasoning into detailed_results
        # Create a mapping from question to multihop index
        mh_questions = set(qa['question'] for qa in multihop_items)
        mh_idx = 0
        for i, detail in enumerate(results['detailed_results']):
            if detail.get('question') in mh_questions and mh_idx < len(mh_results):
                detail['reasoning_score'] = mh_results[mh_idx]['reasoning_score']
                detail['bridge_entity'] = mh_results[mh_idx].get('bridge_entity', 'None')
                detail['llm_hop_count'] = mh_results[mh_idx]['hop_count']
                mh_idx += 1
        
        results['multihop_metrics'] = {
            'items_evaluated': len(multihop_items),
            'avg_hop_count': float(np.mean([r['hop_count'] for r in mh_results])),
            'avg_reasoning_score': float(np.mean([r['reasoning_score'] for r in mh_results])),
            'hop_distribution': dict(Counter([r['hop_count'] for r in mh_results]))
        }
        
        print(f"  Avg Hop Count: {results['multihop_metrics']['avg_hop_count']:.2f}")
        print(f"  Avg Reasoning Score: {results['multihop_metrics']['avg_reasoning_score']:.3f}")
    
    # ==========================================
    # 5. MULTIMODAL METRICS (on multimodal subset)
    # ==========================================
    multimodal_items = [qa for qa in raw_data 
                        if any(c.get('image_path') and c.get('image_path') != 'null' 
                              for c in qa.get('context_chunks', []))]
    if multimodal_items:
        print("\n" + "=" * 60)
        print(f"VISUAL DEPENDENCY ({len(multimodal_items)} items)")
        print("=" * 60)
        
        visual_scores = []
        mm_questions = []
        for qa in multimodal_items:
            contexts = [qa.get('final_context', qa.get('original_chunk', ''))]
            score = evaluator.evaluate_visual_dependency(qa['question'], contexts)
            visual_scores.append(score)
            mm_questions.append(qa['question'])
        
        # Merge visual dependency into detailed_results
        mm_question_set = set(mm_questions)
        mm_idx = 0
        for i, detail in enumerate(results['detailed_results']):
            if detail.get('question') in mm_question_set and mm_idx < len(visual_scores):
                detail['visual_dependency_score'] = visual_scores[mm_idx]
                mm_idx += 1
        
        results['multimodal_metrics'] = {
            'items_evaluated': len(multimodal_items),
            'avg_visual_dependency': float(np.mean(visual_scores)),
            'items_visual_essential': sum(1 for s in visual_scores if s >= 1.0),
            'items_visual_helpful': sum(1 for s in visual_scores if 0.0 < s < 1.0),
            'items_text_sufficient': sum(1 for s in visual_scores if s == 0.0)
        }
        
        print(f"  Avg Visual Dependency: {results['multimodal_metrics']['avg_visual_dependency']:.3f}")
    
    # ==========================================
    # 6. DOMAIN COVERAGE (if corpus provided)
    # ==========================================
    if corpus_path and os.path.exists(corpus_path):
        print("\n" + "=" * 60)
        print("DOMAIN COVERAGE")
        print("=" * 60)
        
        with open(corpus_path, 'r') as f:
            corpus_chunks = json.load(f)
        
        coverage = evaluator.evaluate_domain_coverage(raw_data, corpus_chunks)
        results['domain_coverage'] = coverage
        
        print(f"  Chunk Coverage: {coverage['chunk_coverage']*100:.1f}%")
        print(f"  File Coverage: {coverage['file_coverage']*100:.1f}%")
    
    # ==========================================
    # 7. SEMANTIC DIVERSITY
    # ==========================================
    print("\n" + "=" * 60)
    print("SEMANTIC DIVERSITY")
    print("=" * 60)
    
    questions = [qa.get('question', '') for qa in raw_data]
    diversity = evaluator.evaluate_semantic_diversity(questions)
    results['dataset_health'] = {
        'semantic_diversity': float(diversity),
        'total_samples': len(raw_data)
    }
    
    print(f"  Semantic Diversity: {diversity:.3f}")
    
    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    print("\n" + "=" * 70)
    print("📊 EVALUATION SUMMARY (Harmonized Metrics)")
    print("=" * 70)
    
    # Dataset Info
    ds_info = results.get('dataset_info', {})
    if ds_info.get('expert_persona') or ds_info.get('domain'):
        print(f"\n  Dataset Info:")
        if ds_info.get('expert_persona'):
            print(f"    Expert Persona:     {ds_info['expert_persona']}")
        if ds_info.get('domain'):
            print(f"    Domain:             {ds_info['domain']}")
        print(f"    QA Pairs Generated: {ds_info.get('total_qa_pairs_generated', 0)}")
        print(f"    QA Pairs Evaluated: {ds_info.get('total_qa_pairs_evaluated', 0)}")
        if ds_info.get('sampled'):
            print(f"    ⚠️  Sampled for evaluation")
    
    rm = results['ragas_metrics']
    print(f"\n  RAGAS Metrics:")
    print(f"    Faithfulness:       {rm.get('faithfulness', 0):.3f}")
    print(f"    Answer Relevancy:   {rm.get('answer_relevancy', 0):.3f}")
    print(f"    Context Precision:  {rm.get('context_precision', 0):.3f}")
    print(f"    Context Recall:     {rm.get('context_recall', 0):.3f}")
    
    if results.get('context_necessity'):
        print(f"\n  Context Necessity:    {results['context_necessity'].get('avg_context_necessity_score', 0):.3f}")
    
    if results.get('multihop_metrics'):
        print(f"\n  Multihop Reasoning:")
        print(f"    Avg Hops:           {results['multihop_metrics'].get('avg_hop_count', 0):.2f}")
        print(f"    Reasoning Score:    {results['multihop_metrics'].get('avg_reasoning_score', 0):.3f}")
    
    if results.get('multimodal_metrics'):
        print(f"\n  Multimodal:")
        print(f"    Visual Dependency:  {results['multimodal_metrics'].get('avg_visual_dependency', 0):.3f}")
    
    print(f"\n  Dataset Health:")
    print(f"    Semantic Diversity: {results['dataset_health'].get('semantic_diversity', 0):.3f}")
    print(f"    Total Samples:      {results['subset_statistics'].get('total_qa_pairs', 0)}")
    print("=" * 70)
    
    # Save full results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "subset_evaluation_report.json")  # Same name as metrics.py
        
        def convert_numpy(obj):
            """Convert numpy types to Python types for JSON serialization"""
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj
        
        with open(report_path, 'w') as f:
            json.dump(convert_numpy(results), f, indent=2)
        print(f"\n💾 Report saved to: {report_path}")
    
    return results


# ============================================================================
# MAIN (for standalone/CLI usage)
# ============================================================================

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Run optimized metrics evaluation")
    parser.add_argument("qa_file", nargs="?", help="Path to QA JSON file")
    parser.add_argument("-o", "--output-dir", default=".", help="Output directory")
    parser.add_argument("-c", "--corpus", default=None, help="Path to chunks.json for domain coverage")
    parser.add_argument("-s", "--sample-size", type=int, default=None, help="Sample size for evaluation")
    parser.add_argument("--no-multimodal", action="store_true", help="Disable multimodal metrics")
    parser.add_argument("--no-context-necessity", action="store_true", help="Skip context necessity evaluation")
    
    args = parser.parse_args()
    
    if args.qa_file:
        # Run on provided QA file
        results = run_optimized_pipeline_evaluation(
            qa_file=args.qa_file,
            output_dir=args.output_dir,
            corpus_path=args.corpus,
            enable_multimodal=not args.no_multimodal,
            max_workers=8,
            sample_size=args.sample_size,
            run_context_necessity=not args.no_context_necessity
        )
    else:
        # Test with sample data
        print("No QA file provided. Running with sample data...")
        sample_qa = [
            {
                "question": "What is the capital of France?",
                "answer": "Paris is the capital of France. It is also the largest city in France.",
                "contexts": [
                    "Paris is the capital and most populous city of France.",
                    "France is a country in Western Europe."
                ],
                "ground_truth": "Paris is the capital of France."
            }
        ]
        
        results = run_optimized_evaluation(sample_qa, enable_multimodal=False)
        print("\nDetailed result:")
        print(json.dumps(results['detailed_results'][0], indent=2))

