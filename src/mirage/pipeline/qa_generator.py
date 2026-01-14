#!/usr/bin/env python3
"""
QA Generation from Multi-hop Context Completion

Pipeline for each chunk:
1. Load chunk from INPUT_CHUNKS_FILE
2. Build multihop context using retrieval (adds related chunks)
3. Generate Q&A pairs from the complete multihop context
4. Select best Q&A pairs using selection agent
5. Verify Q&A pairs require the context to answer
6. Save successful and failed Q&A pairs
"""

import json
import re
import logging
import os
from typing import Dict, List
from tqdm import tqdm
from mirage.core.llm import call_vlm_interweaved, setup_logging, batch_call_vlm_interweaved
from mirage.pipeline.context import build_complete_context
from mirage.pipeline.domain import fetch_domain_and_role, load_domain_expert_from_env, save_domain_expert_to_env
from mirage.core.prompts import PROMPTS, PROMPTS_CHUNK

# Configuration (override via config.yaml or command line)
INPUT_CHUNKS_FILE = "output/results/chunks.json"
OUTPUT_SUCCESSFUL = "qa_multihop_pass.json"
OUTPUT_FAILED = "qa_multihop_fail.json"
OUTPUT_IRRELEVANT = "irrelevant_chunk.json"
MAX_CHUNKS = None  # Process all chunks (set to integer for testing, e.g., 100)
CHUNK_ADDITION_MODE = "RELATED"  # EXPLANATORY (only direct answers) or RELATED (both)

def call_ai_service(prompt: str, chunks: List[Dict]) -> str:
    """Unified call to VLM using interleaved chunks"""
    return call_vlm_interweaved(prompt, chunks)

def check_chunk_relevance(chunk_content: str, expert_persona: str, domain: str) -> bool:
    """Check if a chunk is relevant to the expert role and domain. Returns True if relevant, False otherwise."""
    print(f"🔍 Checking chunk relevance for {expert_persona} in {domain}...")
    
    # Format prompt - domain should never be None at this point
    domain_context = domain if domain else "unspecified domain"
    
    prompt = PROMPTS_CHUNK["relevance_check"].format(
        expert_persona=expert_persona,
        domain=domain_context,
        content=chunk_content[:2000]  # Limit content length to avoid token limits
    )
    
    # Use a simple text-only LLM call for relevance check (no images needed)
    from mirage.core.llm import call_llm_simple
    response = call_llm_simple(prompt)
    
    # Parse response - should be "RELEVANT" or "NOT_RELEVANT"
    response_upper = response.strip().upper()
    is_relevant = "RELEVANT" in response_upper and "NOT_RELEVANT" not in response_upper
    
    if is_relevant:
        print(f"  ✅ Chunk is RELEVANT")
    else:
        print(f"  ❌ Chunk is NOT_RELEVANT")
    
    return is_relevant

def generate_qa(chunks: List[Dict], expert_persona: str, domain: str) -> list:
    """Generate one or more Q&A pairs from multihop context chunks using consolidated prompt"""
    print(f"❓ Generating Q&A pairs from {len(chunks)} context chunks...")
    
    # Format prompt with or without domain
    if domain:
        domain_context = f" in the field of {domain}"
        domain_relevance = f" ({domain})"
    else:
        domain_context = ""
        domain_relevance = ""
    
    prompt = PROMPTS["question_answer_generation"].format(
        content="[Refer to the chunks provided below]",
        expert_persona=expert_persona,
        domain_context=domain_context,
        domain_relevance=domain_relevance
    )
    
    response = call_ai_service(prompt, chunks)
    
    # Parse multiple Q&A pairs from response using delimiter format
    qa_pairs = []
    tuple_delimiter = PROMPTS.get("DEFAULT_TUPLE_DELIMITER", "<|#|>")
    completion_delimiter = PROMPTS.get("DEFAULT_COMPLETION_DELIMITER", "<|#|>END<|#|>")
    
    try:
        # Remove completion delimiter if present
        if completion_delimiter in response:
            response = response.split(completion_delimiter)[0].strip()
    
        # Remove START delimiter if present at the beginning
        start_delimiter = tuple_delimiter + "START" + tuple_delimiter
        if response.startswith(start_delimiter):
            response = response[len(start_delimiter):].strip()
        
        # Split by lines and process each Q&A pair
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        for line in lines:
            # Skip NEXT delimiter lines
            next_delimiter = tuple_delimiter + "NEXT" + tuple_delimiter
            if line == next_delimiter:
                continue
            
            # Check if line starts with "Question" delimiter pattern (case-insensitive)
            if line.startswith("Question" + tuple_delimiter) or line.startswith("question" + tuple_delimiter):
                # Split by tuple delimiter
                parts = line.split(tuple_delimiter)
                
                # Expected format: Question<|#|><question_text><|#|>Answer<|#|><answer_text><|#|>Relevance<|#|><score><|#|>Difficulty<|#|><score>
                # Handle both capitalized and lowercase versions
                if len(parts) >= 4 and parts[0].lower() == "question" and parts[2].lower() == "answer":
                    question = parts[1].strip()
                    answer = parts[3].strip()
                    relevance = "0"
                    difficulty = "0"
                    
                    # Try to extract scores if present
                    if len(parts) >= 8:
                        if parts[4].lower() == "relevance":
                            relevance = parts[5].strip()
                        if parts[6].lower() == "difficulty":
                            difficulty = parts[7].strip()
                    
                    if question and answer:
                        qa_pairs.append({
                            "question": question,
                            "answer": answer,
                            "relevance_score": relevance,
                            "difficulty_score": difficulty
                        })
        
        if not qa_pairs:
            # Fallback: try old format patterns for backward compatibility
            # Try delimiter format with case-insensitive matching
            for line in lines:
                if re.match(r'(?i)^question', line):
                    parts = re.split(r'(?i)question' + re.escape(tuple_delimiter), line, maxsplit=1)
                    if len(parts) >= 2:
                        qa_content = parts[1]
                        qa_parts = qa_content.split(tuple_delimiter)
                        if len(qa_parts) >= 3 and qa_parts[1].lower() == "answer":
                            question = qa_parts[0].strip()
                            answer = qa_parts[2].strip()
                            if question and answer:
                                qa_pairs.append({
                                    "question": question, 
                                    "answer": answer,
                                    "relevance_score": "0",
                                    "difficulty_score": "0"
                                })
            
            # Final fallback: try old Question:/Answer: format
            if not qa_pairs:
                question_matches = re.finditer(r'(?i)Question:\s*(.*?)(?=\nAnswer:|\n\n|$)', response, re.DOTALL)
                answer_matches = re.finditer(r'(?i)Answer:\s*(.*?)(?=\nQuestion:|\n\n|$)', response, re.DOTALL)
        
                questions = [m.group(1).strip() for m in question_matches]
                answers = [m.group(1).strip() for m in answer_matches]
                
                if questions and answers and len(questions) == len(answers):
                    for q, a in zip(questions, answers):
                        qa_pairs.append({
                            "question": q, 
                            "answer": a,
                            "relevance_score": "0",
                            "difficulty_score": "0"
                        })
                elif questions and answers:
                    # Mismatched counts, try to pair them
                    for i, q in enumerate(questions):
                        if i < len(answers):
                            qa_pairs.append({
                                "question": q, 
                                "answer": answers[i],
                                "relevance_score": "0",
                                "difficulty_score": "0"
                            })
                
                if not qa_pairs:
                    print("⚠️ Could not parse Q&A from response")
                    qa_pairs.append({
                        "question": response,
                        "answer": "",
                        "relevance_score": "0",
                        "difficulty_score": "0"
                    })
        
        print(f"✅ Generated {len(qa_pairs)} Q&A pair(s)")
        return qa_pairs
        
    except Exception as e:
        print(f"⚠️ Error parsing Q&A: {e}")
        import traceback
        print(traceback.format_exc())
        return [{
            "question": response, 
            "answer": "",
            "relevance_score": "0",
            "difficulty_score": "0"
        }]

def select_qa_pairs(qa_pairs: list, chunks: List[Dict], expert_persona: str, domain: str) -> tuple[list, list]:
    """Select/filter Q&A pairs using the selection agent. Returns (selected, rejected).
    Uses batch processing when multiple QA pairs need to be evaluated.
    """
    print(f"🔍 Selecting Q&A pairs ({len(qa_pairs)} candidates)...")
    
    if not qa_pairs:
        return [], []
    
    tuple_delimiter = PROMPTS.get("DEFAULT_TUPLE_DELIMITER", "<|#|>")
    
    # Format domain context
    if domain:
        domain_context = f" in the field of {domain}"
        domain_relevance = f" ({domain})"
    else:
        domain_context = ""
        domain_relevance = ""
    
    # Prepare batch requests
    requests = []
    for qa_pair in qa_pairs:
        prompt = PROMPTS["question_answer_selection"].format(
            content="[Refer to the chunks provided below]",
            question=qa_pair["question"],
            answer=qa_pair["answer"],
            expert_persona=expert_persona,
            domain_context=domain_context,
            domain_relevance=domain_relevance
        )
        requests.append((prompt, chunks))
    
    # Execute batch or sequential based on count
    if len(requests) > 1:
        print(f"  ⚡ Batch evaluating {len(requests)} QA pairs...")
        responses = batch_call_vlm_interweaved(requests, show_progress=False)
    else:
        responses = [call_ai_service(requests[0][0], chunks)]
    
    # Process responses
    selected_pairs = []
    rejected_pairs = []
    
    for idx, (qa_pair, response) in enumerate(zip(qa_pairs, responses), 1):
        try:
            if response and not response.startswith("ERROR:"):
                parts = response.split(tuple_delimiter)
                status = "REJECTED"
                relevance = "0"
                difficulty = "0"
                reason = "No reason provided"
                
                for i in range(0, len(parts), 2):
                    if i+1 < len(parts):
                        key = parts[i].strip().lower()
                        value = parts[i+1].strip()
                        
                        if key == "status":
                            status = value.upper()
                        elif key == "relevance":
                            relevance = value
                        elif key == "difficulty":
                            difficulty = value
                        elif key == "reason":
                            reason = value
                
                qa_pair["relevance_score"] = relevance
                qa_pair["difficulty_score"] = difficulty
                qa_pair["selection_reason"] = reason
                qa_pair["selection_status"] = status
                
                if status == "SELECTED":
                    selected_pairs.append(qa_pair)
                    print(f"    ✅ Q{idx} SELECTED (R:{relevance}/D:{difficulty})")
                else:
                    rejected_pairs.append(qa_pair)
                    print(f"    ❌ Q{idx} REJECTED: {reason[:60]}...")
            else:
                qa_pair["selection_status"] = "ERROR"
                qa_pair["selection_reason"] = f"API Error: {response}"
                rejected_pairs.append(qa_pair)
                
        except Exception as e:
            print(f"    ⚠️ Error parsing selection response for Q{idx}: {e}")
            qa_pair["selection_status"] = "ERROR"
            qa_pair["selection_reason"] = f"Parsing error: {str(e)}"
            rejected_pairs.append(qa_pair)
    
    print(f"✅ Selection complete: {len(selected_pairs)} selected, {len(rejected_pairs)} rejected")
    return selected_pairs, rejected_pairs

def verify_qa(chunks: List[Dict], question: str, answer: str, expert_persona: str, domain: str) -> str:
    """Verify if the question requires the content to be answered"""
    print("🔍 Verifying Q&A pair...")
    
    # Format prompt with or without domain
    if domain:
        domain_context = f" in the field of {domain}"
    else:
        domain_context = ""
    
    prompt = PROMPTS["question_answer_verification"].format(
        content="[Refer to the chunks provided below]",
        question=question,
        answer=answer,
        expert_persona=expert_persona,
        domain_context=domain_context
    )
    response = call_ai_service(prompt, chunks)
    return response


def batch_verify_qa(chunks: List[Dict], qa_pairs: List[Dict], expert_persona: str, domain: str) -> List[str]:
    """Batch verify multiple Q&A pairs using concurrent API calls
    
    Args:
        chunks: Context chunks for verification
        qa_pairs: List of dicts with 'question' and 'answer' keys
        expert_persona: Expert role string
        domain: Domain string
        
    Returns:
        List of verification result strings in same order as qa_pairs
    """
    if not qa_pairs:
        return []
    
    # Format domain context
    if domain:
        domain_context = f" in the field of {domain}"
    else:
        domain_context = ""
    
    # Prepare batch requests
    requests = []
    for qa in qa_pairs:
        prompt = PROMPTS["question_answer_verification"].format(
            content="[Refer to the chunks provided below]",
            question=qa['question'],
            answer=qa['answer'],
            expert_persona=expert_persona,
            domain_context=domain_context
        )
        requests.append((prompt, chunks))
    
    # Execute batch
    if len(requests) > 1:
        print(f"  ⚡ Batch verifying {len(requests)} QA pairs...")
        responses = batch_call_vlm_interweaved(requests, show_progress=False)
    else:
        responses = [call_ai_service(requests[0][0], chunks)]
    
    return responses

def process_chunk_for_qa(chunk_data: Dict, expert_persona: str, domain: str) -> Dict:
    """Complete pipeline: build context, generate Q&A using pre-extracted role, verify"""
    chunk_id = chunk_data.get('chunk_id', 'unknown')
    chunk_content = chunk_data.get('content', '')
    
    print(f"\n{'='*80}")
    print(f"Processing chunk {chunk_id}")
    print(f"{'='*80}")
    
    # Stage 1: Build complete context using multi-hop retrieval
    print("\n🔄 Stage 1: Building complete context...")
    context_result = build_complete_context(
        initial_chunk=chunk_data,
        max_depth=3,
        max_breadth=3,
        chunks_per_search=2,
        expert_persona=expert_persona,
        domain=domain,
        chunk_addition_mode=CHUNK_ADDITION_MODE
    )
    
    final_context = context_result['context']
    context_chunks = context_result.get('chunks', [])
    completion_status = context_result['status']
    
    print(f"Context status: {completion_status}")
    print(f"Final context length: {len(final_context)} chars")
    print(f"Multihop context: {len(context_chunks)} chunks (original + retrieved)")
    
    # Stage 2: Use pre-extracted expert role and domain (from BERTopic analysis)
    print(f"\n✅ Using expert role: {expert_persona}")
    print(f"✅ Using domain: {domain}")
    
    # Stage 3: Generate Q&A pairs from multihop context
    print(f"\n🔄 Stage 3: Generating Q&A pairs from multihop context ({len(context_chunks)} chunks)...")
    qa_pairs = generate_qa(context_chunks, expert_persona, domain)
    
    # Stage 3.5: Select Q&A pairs
    print("\n🔄 Stage 3.5: Selecting Q&A pairs...")
    selected_pairs, rejected_pairs = select_qa_pairs(qa_pairs, context_chunks, expert_persona, domain)
    
    # Stage 4: Verify selected Q&A pairs - BATCH PROCESSING
    verified_qa_pairs = []
    print(f"\n🔄 Stage 4: Verifying {len(selected_pairs)} selected Q&A pair(s)...")
    
    if selected_pairs:
        # Use batch verification
        verification_results = batch_verify_qa(
            context_chunks, selected_pairs, expert_persona, domain
        )
        
        for i, (qa_pair, verification_result) in enumerate(zip(selected_pairs, verification_results), 1):
            question = qa_pair["question"]
            answer = qa_pair["answer"]
            print(f"\n  Q&A {i}: {question[:80]}...")
            print(f"  Verification: {verification_result[:100] if verification_result else 'ERROR'}...")
            
            verified_qa_pairs.append({
                "question": question,
                "answer": answer,
                "relevance_score": qa_pair.get("relevance_score", "0"),
                "difficulty_score": qa_pair.get("difficulty_score", "0"),
                "selection_status": qa_pair.get("selection_status", "SELECTED"),
                "selection_reason": qa_pair.get("selection_reason", ""),
                "verification_result": verification_result or "ERROR"
            })
    
    return {
        "chunk_id": chunk_id,
        "original_chunk": chunk_content,
        "final_context": final_context,
        "context_chunks": context_chunks,  # Full chunks with image_path for multimodal eval
        "context_status": completion_status,
        "depth_reached": context_result['depth'],
        "chunks_added": context_result['chunks_added'],
        "expert_persona": expert_persona,
        "domain": domain,
        "selected_qa_pairs": verified_qa_pairs,
        "rejected_qa_pairs": rejected_pairs
    }

def is_verification_successful(verification_result: str) -> bool:
    """Check if verification indicates success"""
    required_good = ["QUESTION_CORRECT", "ANSWER_CORRECT", "REQUIRES_CONTENT"]
    bad_values = ["QUESTION_INCORRECT", "ANSWER_INCORRECT", "CAN_ANSWER_WITHOUT_CONTENT"]
    
    has_bad = any(bad in verification_result for bad in bad_values)
    has_all_good = all(good in verification_result for good in required_good)
    
    return has_all_good and not has_bad


def correct_failed_qa(chunks: List[Dict], failed_qa_pairs: List[Dict], 
                      expert_persona: str, domain: str) -> List[Dict]:
    """Generate corrected QA pairs from failed QA pairs using verification feedback.
    
    Args:
        chunks: Context chunks for QA generation
        failed_qa_pairs: List of dicts with 'question', 'answer', 'verification_result'
        expert_persona: Expert role string
        domain: Domain string
        
    Returns:
        List of corrected QA pair dicts with 'question', 'answer', 'relevance_score', 'difficulty_score'
    """
    if not failed_qa_pairs:
        return []
    
    print(f"🔧 Correcting {len(failed_qa_pairs)} failed Q&A pair(s)...")
    
    # Format the failed QA feedback section
    failed_qa_feedback_parts = []
    for i, qa in enumerate(failed_qa_pairs, 1):
        question = qa.get('question', '')
        answer = qa.get('answer', '')
        verification = qa.get('verification_result', 'No verification feedback available')
        
        failed_qa_feedback_parts.append(
            f"--- Failed QA #{i} ---\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Verification Feedback: {verification}\n"
        )
    
    failed_qa_feedback = "\n".join(failed_qa_feedback_parts)
    
    # Format domain context
    if domain:
        domain_context = f" in the field of {domain}"
    else:
        domain_context = ""
    
    prompt = PROMPTS["question_answer_generation_corrected"].format(
        content="[Refer to the chunks provided below]",
        expert_persona=expert_persona,
        domain_context=domain_context,
        failed_qa_feedback=failed_qa_feedback
    )
    
    response = call_ai_service(prompt, chunks)
    
    # Parse corrected QA pairs (same format as generate_qa)
    corrected_pairs = []
    tuple_delimiter = PROMPTS.get("DEFAULT_TUPLE_DELIMITER", "<|#|>")
    completion_delimiter = PROMPTS.get("DEFAULT_COMPLETION_DELIMITER", "<|#|>END<|#|>")
    
    try:
        # Remove completion delimiter if present
        if completion_delimiter in response:
            response = response.split(completion_delimiter)[0].strip()
        
        # Remove START delimiter if present
        start_delimiter = tuple_delimiter + "START" + tuple_delimiter
        if response.startswith(start_delimiter):
            response = response[len(start_delimiter):].strip()
        
        # Check for empty response (no valid correction possible)
        if not response.strip() or response.strip() == tuple_delimiter + "START" + tuple_delimiter:
            print("  ⚠️ No correction possible - content doesn't support the original topic")
            return []
        
        # Split by NEXT delimiter for multiple QA pairs
        next_delimiter = tuple_delimiter + "NEXT" + tuple_delimiter
        qa_sections = response.split(next_delimiter)
        
        for section in qa_sections:
            section = section.strip()
            if not section:
                continue
            
            # Parse Question, Answer, Relevance, Difficulty
            parts = section.split(tuple_delimiter)
            qa_dict = {}
            
            for j in range(0, len(parts) - 1, 2):
                key = parts[j].strip()
                value = parts[j + 1].strip() if j + 1 < len(parts) else ""
                
                if key == "Question":
                    qa_dict["question"] = value
                elif key == "Answer":
                    qa_dict["answer"] = value
                elif key == "Relevance":
                    qa_dict["relevance_score"] = value
                elif key == "Difficulty":
                    qa_dict["difficulty_score"] = value
            
            if qa_dict.get("question") and qa_dict.get("answer"):
                qa_dict["correction_status"] = "CORRECTED"
                corrected_pairs.append(qa_dict)
                print(f"  ✅ Corrected: {qa_dict['question'][:60]}...")
    
    except Exception as e:
        logging.error(f"Error parsing corrected QA response: {e}")
        print(f"  ❌ Error parsing correction response: {e}")
    
    print(f"  📊 Generated {len(corrected_pairs)} corrected Q&A pair(s)")
    return corrected_pairs


if __name__ == "__main__":
    setup_logging()
    
    # Load chunks
    print(f"📂 Loading chunks from {INPUT_CHUNKS_FILE}...")
    with open(INPUT_CHUNKS_FILE, 'r') as f:
        chunks = json.load(f)
    
    # Limit to first N chunks if MAX_CHUNKS is set
    if MAX_CHUNKS is not None:
        chunks = chunks[:MAX_CHUNKS]
        print(f"📊 Processing {len(chunks)} chunks (limited to {MAX_CHUNKS} for testing)...")
    else:
        print(f"📊 Processing all {len(chunks)} chunks...")
    
    # Extract domain and expert role once for all chunks using BERTopic
    print(f"\n{'='*80}")
    
    # Priority 1: Check config.yaml
    domain, expert_persona = None, None
    try:
        from mirage.core.config import get_domain_expert_config
        domain_config = get_domain_expert_config()
        config_domain = domain_config.get('domain')
        config_persona = domain_config.get('expert_persona')
        
        if config_domain and config_persona:
            print(f"✅ Using domain and expert persona from config.yaml")
            domain, expert_persona = config_domain, config_persona
            save_domain_expert_to_env(domain, expert_persona)
    except ImportError:
        pass
    
    # Priority 2: Check environment variables
    if not domain or not expert_persona:
        env_domain, env_persona = load_domain_expert_from_env()
        if env_domain and env_persona:
            domain, expert_persona = env_domain, env_persona
    
    # Priority 3: Auto-detect using BERTopic
    if not domain or not expert_persona:
        print("🔍 Auto-detecting domain and expert persona from corpus...")
        domain, expert_persona = fetch_domain_and_role(INPUT_CHUNKS_FILE)
        # Note: fetch_domain_and_role already saves to environment variables
    
    print(f"✅ Domain: {domain}")
    print(f"✅ Expert Role: {expert_persona}")
    print(f"{'='*80}\n")
    
    # Initialize result containers
    successful_qa_pairs = []
    failed_qa_pairs = []
    irrelevant_chunks = []
    
    # Process each chunk: relevance check -> build multihop context -> generate QA -> select -> verify
    for i, chunk in tqdm(enumerate(chunks, 1), total=len(chunks), desc="Processing chunks"):
        tqdm.write(f"\n{'='*80}")
        tqdm.write(f"Processing Chunk {i}/{len(chunks)}")
        tqdm.write(f"Pipeline: Multihop Context → QA Generation → Selection → Verification")
        tqdm.write(f"{'='*80}")
        
        try:
            # Extract chunk content and metadata
            if isinstance(chunk, dict):
                chunk_content = chunk.get("content", str(chunk))
                source_document = chunk.get("file_name", "unknown")
                chunk_id = chunk.get("chunk_id", str(i))
            else:
                chunk_content = str(chunk)
                source_document = "unknown"
                chunk_id = str(i)
            
            # Stage 0: Check chunk relevance
            print(f"\n🔄 Stage 0: Checking chunk relevance...")
            is_relevant = check_chunk_relevance(chunk_content, expert_persona, domain)
            
            if not is_relevant:
                # Store irrelevant chunk and skip processing
                irrelevant_chunks.append({
                    "chunk_id": chunk_id,
                    "source_document": source_document
                })
                tqdm.write(f"⏭️  Chunk {i} is NOT_RELEVANT - skipping processing")
                continue
            
            # Prepare chunk data
            if isinstance(chunk, dict):
                chunk_data = chunk
            else:
                chunk_data = {
                    "content": str(chunk),
                    "chunk_id": str(i),
                    "file_name": "unknown",
                    "artifact": "None"
                }

            # Process chunk with pre-extracted domain and role
            result = process_chunk_for_qa(chunk_data, expert_persona, domain)
            
            # Process selected Q&A pairs
            successful_count = 0
            for qa_pair in result.get("selected_qa_pairs", []):
                if is_verification_successful(qa_pair.get("verification_result", "")):
                    successful_count += 1
                    # Create individual entry for each successful Q&A pair
                    successful_qa_pairs.append({
                        "chunk_id": result["chunk_id"],
                        "original_chunk": result["original_chunk"],
                        "final_context": result["final_context"],
                        "context_chunks": result.get("context_chunks", []),  # Full chunks with image_path
                        "context_status": result["context_status"],
                        "depth_reached": result["depth_reached"],
                        "chunks_added": result["chunks_added"],
                        "expert_persona": result["expert_persona"],
                        "domain": result.get("domain", ""),
                        "question": qa_pair["question"],
                        "answer": qa_pair["answer"],
                        "relevance_score": qa_pair.get("relevance_score", "0"),
                        "difficulty_score": qa_pair.get("difficulty_score", "0"),
                        "selection_status": qa_pair.get("selection_status", "SELECTED"),
                        "selection_reason": qa_pair.get("selection_reason", ""),
                        "verification_result": qa_pair["verification_result"]
                    })
                else:
                    # Create individual entry for each failed verification
                    failed_qa_pairs.append({
                        "chunk_id": result["chunk_id"],
                        "original_chunk": result["original_chunk"],
                        "final_context": result["final_context"],
                        "context_chunks": result.get("context_chunks", []),  # Full chunks with image_path
                        "context_status": result["context_status"],
                        "depth_reached": result["depth_reached"],
                        "chunks_added": result["chunks_added"],
                        "expert_persona": result["expert_persona"],
                        "domain": result.get("domain", ""),
                        "question": qa_pair["question"],
                        "answer": qa_pair["answer"],
                        "relevance_score": qa_pair.get("relevance_score", "0"),
                        "difficulty_score": qa_pair.get("difficulty_score", "0"),
                        "selection_status": qa_pair.get("selection_status", "SELECTED"),
                        "selection_reason": qa_pair.get("selection_reason", ""),
                        "verification_result": qa_pair["verification_result"],
                        "failure_reason": "Failed verification"
                    })
            
            # Add rejected Q&A pairs to failed list
            for qa_pair in result.get("rejected_qa_pairs", []):
                failed_qa_pairs.append({
                    "chunk_id": result["chunk_id"],
                    "original_chunk": result["original_chunk"],
                    "final_context": result["final_context"],
                    "context_chunks": result.get("context_chunks", []),  # Full chunks with image_path
                    "context_status": result["context_status"],
                    "depth_reached": result["depth_reached"],
                    "chunks_added": result["chunks_added"],
                    "expert_persona": result["expert_persona"],
                    "domain": result.get("domain", ""),
                    "question": qa_pair["question"],
                    "answer": qa_pair["answer"],
                    "relevance_score": qa_pair.get("relevance_score", "0"),
                    "difficulty_score": qa_pair.get("difficulty_score", "0"),
                    "selection_status": qa_pair.get("selection_status", "REJECTED"),
                    "selection_reason": qa_pair.get("selection_reason", ""),
                    "verification_result": "N/A - rejected by selection agent",
                    "failure_reason": "Rejected by selection agent"
                })
            
            total_qa = len(result.get("selected_qa_pairs", [])) + len(result.get("rejected_qa_pairs", []))
            if successful_count > 0:
                tqdm.write(f"✅ {successful_count}/{total_qa} Q&A pair(s) passed all stages")
            else:
                tqdm.write(f"⚠️ 0/{total_qa} Q&A pairs passed all stages")
                
        except Exception as e:
            error_msg = f"❌ Error processing chunk {i}: {e}"
            tqdm.write(error_msg)
            logging.error(error_msg)
            
            import traceback
            logging.error(traceback.format_exc())
            
            failed_qa_pairs.append({
                "chunk_id": i,
                "chunk_content": chunk_content if 'chunk_content' in locals() else str(chunk),
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            continue
    
    # Save results
    print(f"\n{'='*80}")
    print("📁 Saving results...")
    print(f"{'='*80}")
    
    if successful_qa_pairs:
        with open(OUTPUT_SUCCESSFUL, 'w', encoding='utf-8') as f:
            json.dump(successful_qa_pairs, f, indent=2, ensure_ascii=False)
        print(f"✅ Successful QA pairs: {len(successful_qa_pairs)} saved to {OUTPUT_SUCCESSFUL}")
    
    if failed_qa_pairs:
        with open(OUTPUT_FAILED, 'w', encoding='utf-8') as f:
            json.dump(failed_qa_pairs, f, indent=2, ensure_ascii=False)
        print(f"⚠️ Failed QA pairs: {len(failed_qa_pairs)} saved to {OUTPUT_FAILED}")
    
    if irrelevant_chunks:
        with open(OUTPUT_IRRELEVANT, 'w', encoding='utf-8') as f:
            json.dump(irrelevant_chunks, f, indent=2, ensure_ascii=False)
        print(f"⏭️  Irrelevant chunks: {len(irrelevant_chunks)} saved to {OUTPUT_IRRELEVANT}")
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 SUMMARY")
    print(f"{'='*80}")
    print(f"Total chunks processed: {len(chunks)}")
    print(f"Relevant chunks processed: {len(chunks) - len(irrelevant_chunks)}")
    print(f"Irrelevant chunks skipped: {len(irrelevant_chunks)}")
    print(f"Successful QA pairs: {len(successful_qa_pairs)}")
    print(f"Failed QA pairs: {len(failed_qa_pairs)}")
    if len(chunks) - len(irrelevant_chunks) > 0:
        print(f"Success rate: {len(successful_qa_pairs)/(len(chunks) - len(irrelevant_chunks))*100:.1f}%")
    
    # Stats: Count single vs multiple chunks in successful QA pairs
    single_chunk_count = 0
    multiple_chunk_count = 0
    for qa_pair in successful_qa_pairs:
        chunks_added = qa_pair.get("chunks_added", [])
        if isinstance(chunks_added, list):
            if len(chunks_added) == 1:
                single_chunk_count += 1
            elif len(chunks_added) > 1:
                multiple_chunk_count += 1
    
    print(f"\n📊 QA STATS")
    print(f"{'='*80}")
    print(f"Number of QA pairs in qa_multihop_fail: {len(failed_qa_pairs)}")
    print(f"Number of QA pairs with single chunk in qa_multihop_pass: {single_chunk_count}")
    print(f"Number of QA pairs with multiple chunks in qa_multihop_pass: {multiple_chunk_count}")
