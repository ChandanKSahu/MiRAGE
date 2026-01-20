
from typing import List, Dict, Any, Tuple
import logging
import re
from mirage.core.llm import call_vlm_with_multiple_images
from mirage.core.prompts import PROMPTS

class LLMReranker:
    """
    Uses an LLM to evaluate, rank, and merge QA pairs.
    """
    def __init__(self, model_name="gpt-oss", expert_persona: str = None, 
                 domain: str = None):
        self.model_name = model_name
        self.expert_persona = expert_persona
        self.domain = domain
        self.tuple_delimiter = PROMPTS.get("DEFAULT_TUPLE_DELIMITER", "<|#|>")
        self.completion_delimiter = PROMPTS.get("DEFAULT_COMPLETION_DELIMITER", "<|#|>END<|#|>")

    def _parse_qa_pairs(self, response_text: str) -> List[Dict[str, str]]:
        """Parses QA pairs from the delimiter format.
        
        New format: <|#|>START<|#|>Question<|#|><text><|#|>Answer<|#|><text><|#|>NEXT<|#|>...<|#|>END<|#|>
        """
        qa_pairs = []
        try:
            # Remove completion delimiter if present
            if self.completion_delimiter in response_text:
                response_text = response_text.split(self.completion_delimiter)[0].strip()
        
            # Remove START delimiter if present
            start_delimiter = self.tuple_delimiter + "START" + self.tuple_delimiter
            if start_delimiter in response_text:
                response_text = response_text.split(start_delimiter, 1)[-1].strip()
            
            # Split by NEXT delimiter to handle multiple QA pairs
            next_delimiter = self.tuple_delimiter + "NEXT" + self.tuple_delimiter
            sections = response_text.split(next_delimiter)
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                
                parts = section.split(self.tuple_delimiter)
                question = None
                answer = None
                
                # Parse key-value pairs from parts
                for i, part in enumerate(parts):
                    part_lower = part.strip().lower()
                    if i + 1 < len(parts):
                        value = parts[i + 1].strip()
                        if part_lower == "question":
                            question = value
                        elif part_lower == "answer":
                            answer = value
                
                if question and answer:
                    qa_pairs.append({"question": question, "answer": answer})
            
            # Fallback: try line-by-line parsing for inline format
            if not qa_pairs:
                lines = [line.strip() for line in response_text.split('\n') if line.strip()]
                for line in lines:
                    if line == next_delimiter:
                        continue
                    if line.startswith("Question" + self.tuple_delimiter) or line.startswith("question" + self.tuple_delimiter):
                        parts = line.split(self.tuple_delimiter)
                        if len(parts) >= 4 and parts[0].lower() == "question" and parts[2].lower() == "answer":
                            question = parts[1].strip()
                            answer = parts[3].strip()
                            if question and answer:
                                qa_pairs.append({"question": question, "answer": answer})
                            
            return qa_pairs
        except Exception as e:
            logging.error(f"Error parsing QA pairs: {e}")
            return []

    def rank_cluster(self, cluster_candidates: List[Dict]) -> List[Dict]:
        """
        Step 3: Order/Rank the QA pairs in the cluster.
        """
        if not cluster_candidates or len(cluster_candidates) < 2:
            return cluster_candidates

        candidates_text = ""
        for idx, item in enumerate(cluster_candidates, 1):
            candidates_text += f"\n--- Candidate {idx} ---\n"
            candidates_text += f"Question: {item.get('question', '')}\n"
            candidates_text += f"Answer: {item.get('answer', '')}\n"

        prompt_template = PROMPTS.get("deduplication_rank", "")
        if not prompt_template:
            logging.warning("deduplication_rank prompt not found.")
            return cluster_candidates

        prompt = prompt_template.format(
            candidates_text=candidates_text,
            expert_persona=self.expert_persona,
            domain=self.domain
        )
        
        try:
            response = call_vlm_with_multiple_images(prompt, [])
            ordered_pairs = self._parse_qa_pairs(response)
            
            if not ordered_pairs:
                logging.warning("LLM returned empty ranking, using original order.")
                return cluster_candidates
                
            # Attach metadata from original if possible (heuristic matching) or just return new order
            # Since we are just reordering, we can try to match back to original items to keep metadata,
            # but for deduplication, text is primary.
            # For simplicity, we return the parsed pairs as the ordered list.
            return ordered_pairs
            
        except Exception as e:
            logging.error(f"Error in LLM ranking: {e}")
            return cluster_candidates

    def deduplicate_and_merge(self, cluster_candidates: List[Dict]) -> List[Dict]:
        """
        Step 5: Deduplicate and merge based on the ordered cluster.
        """
        if not cluster_candidates:
            return []
        
        # First, Rank/Order them
        ordered_candidates = self.rank_cluster(cluster_candidates)
        
        # Prepare text for merge prompt
        candidates_text = ""
        for idx, item in enumerate(ordered_candidates, 1):
            candidates_text += f"\n--- Candidate {idx} ---\n"
            candidates_text += f"Question: {item.get('question', '')}\n"
            candidates_text += f"Answer: {item.get('answer', '')}\n"

        prompt_template = PROMPTS.get("deduplication_merge", "")
        if not prompt_template:
             logging.warning("deduplication_merge prompt not found.")
             return cluster_candidates[:1] # Fallback

        prompt = prompt_template.format(
            candidates_text=candidates_text,
            expert_persona=self.expert_persona,
            domain=self.domain
        )

        try:
            response = call_vlm_with_multiple_images(prompt, [])
            merged_pairs = self._parse_qa_pairs(response)
            
            if not merged_pairs:
                logging.warning("LLM returned empty merge, returning first candidate.")
                return cluster_candidates[:1]
            
            # Propagate metadata from the first original candidate to all new pairs
            # (approximate, since we might have merged multiple)
            base_metadata = cluster_candidates[0].copy()
            final_results = []
            for pair in merged_pairs:
                new_item = base_metadata.copy()
                new_item["question"] = pair["question"]
                new_item["answer"] = pair["answer"]
                new_item["merged_from_count"] = len(cluster_candidates)
                final_results.append(new_item)
                
            return final_results

        except Exception as e:
            logging.error(f"Error in LLM deduplication: {e}")
            return cluster_candidates[:1]
