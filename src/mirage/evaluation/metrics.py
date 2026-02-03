import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from collections import Counter

# Import prompts for LLM-as-a-Judge metrics
try:
    from mirage.core.prompts import PROMPTS_METRICS
except ImportError:
    PROMPTS_METRICS = {}
    print("Warning: Could not import PROMPTS_METRICS from mirage.core.prompts")

# Ragas Imports for Standard RAG Metrics
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,  # Note: RAGAS 0.4.x uses 'answer_relevancy' not 'answer_relevance'
        context_precision,
        context_recall,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
    print("[OK] RAGAS metrics loaded successfully")
except ImportError as e:
    RAGAS_AVAILABLE = False
    print(f"Warning: 'ragas' or 'datasets' not installed. Error: {e}")

# Optional advanced metrics (may not be available in all ragas versions)
HAS_ENTITY_RECALL = False
HAS_NOISE_SENSITIVITY = False
HAS_MULTIMODAL = False

if RAGAS_AVAILABLE:
    try:
        from ragas.metrics import context_entity_recall
        HAS_ENTITY_RECALL = True
    except ImportError:
        print("Info: context_entity_recall not available in this ragas version.")
    
    try:
        from ragas.metrics import noise_sensitivity_relevant
        HAS_NOISE_SENSITIVITY = True
    except ImportError:
        print("Info: noise_sensitivity metrics not available in this ragas version.")
    
    try:
        from ragas.metrics import multimodal_faithfulness, multimodal_relevance
        HAS_MULTIMODAL = True
    except ImportError:
        print("Info: multimodal metrics not available in this ragas version.")

# LangChain Imports
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.prompts import ChatPromptTemplate
        from langchain.output_parsers import StrOutputParser
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        print("Warning: 'langchain' not installed.")

# Output parsers (may be in different locations)
try:
    from langchain.output_parsers import ResponseSchema, StructuredOutputParser
except ImportError:
    try:
        from langchain_community.output_parsers import ResponseSchema, StructuredOutputParser
    except ImportError:
        try:
            from langchain_core.output_parsers import ResponseSchema, StructuredOutputParser
        except ImportError:
            # Define minimal fallbacks
            ResponseSchema = None
            StructuredOutputParser = None

# LangChain Google Gemini Imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Info: 'langchain-google-genai' not installed. Trying OpenAI...")

# LangChain OpenAI Imports (fallback)
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Info: 'langchain_openai' not installed.")

# Data Science Imports
try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Warning: 'scikit-learn' not installed.")

try:
    from scipy.stats import entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: 'scipy' not installed. Domain coverage metrics may fail.")

# VLM Import for multimodal evaluation
try:
    from mirage.core.llm import call_vlm_interweaved, batch_call_vlm_interweaved
    VLM_AVAILABLE = True
    BATCH_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    BATCH_AVAILABLE = False
    print("Warning: mirage.core.llm not available. VLM-based multimodal metrics will be skipped.")

class MultimodalFrameworkEvaluator:
    def __init__(self, model_name=None, embedding_model=None, use_gemini=True):
        """
        Initialize the evaluator.
        Args:
            model_name: The LLM to use as a Judge (auto-detected if None)
            embedding_model: Model for diversity calculations (auto-detected if None)
            use_gemini: If True, prefer Gemini over OpenAI
        """
        # Determine which API to use
        if use_gemini and GEMINI_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
            model_name = model_name or "gemini-2.0-flash"
            embedding_model = embedding_model or "models/text-embedding-004"
            print(f"Using Gemini API with model: {model_name}")
            self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
            self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
            self.api_type = "gemini"
        elif OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            model_name = model_name or "gpt-4-turbo"
            embedding_model = embedding_model or "text-embedding-3-small"
            print(f"Using OpenAI API with model: {model_name}")
            self.llm = ChatOpenAI(model=model_name, temperature=0)
            self.embeddings = OpenAIEmbeddings(model=embedding_model)
            self.api_type = "openai"
        elif use_gemini and GEMINI_AVAILABLE:
            # Try Gemini without env var check (will fail if key not set)
            model_name = model_name or "gemini-2.0-flash"
            embedding_model = embedding_model or "models/text-embedding-004"
            print(f"Using Gemini API with model: {model_name}")
            self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
            self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
            self.api_type = "gemini"
        else:
            raise RuntimeError("No API available. Install langchain-google-genai or langchain-openai and set API key.")
        
        # Ragas metrics configuration (only if ragas available)
        self.ragas_metrics = []
        if RAGAS_AVAILABLE:
            self.ragas_metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ]
            # Add optional metrics if available
            if HAS_ENTITY_RECALL:
                self.ragas_metrics.append(context_entity_recall)
            if HAS_NOISE_SENSITIVITY:
                self.ragas_metrics.append(noise_sensitivity_relevant)
            if HAS_MULTIMODAL:
                self.ragas_metrics.extend([multimodal_faithfulness, multimodal_relevance])

    def load_dataset(self, json_path: str) -> List[Dict]:
        """
        Loads the generated QA dataset.
        Expected JSON format: List of dicts with keys: 
        ['question', 'answer', 'contexts', 'ground_truth' (optional), 'metadata']
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data

    # ==========================================
    # 1. STANDARD RAG METRICS (RAGAS)
    # ==========================================
    def evaluate_ragas_standard(self, data: List[Dict]) -> pd.DataFrame:
        """
        Evaluates Faithfulness, Relevance, Precision, and Recall using Ragas.
        Uses the same LLM backend (Gemini/OpenAI) as configured in the evaluator.
        """
        print("--- Running Standard RAG Metrics (Ragas) ---")
        
        # Import RAGAS LLM wrapper
        try:
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
        except ImportError:
            # Older RAGAS versions
            LangchainLLMWrapper = None
            LangchainEmbeddingsWrapper = None
        
        # RAGAS 0.4.x requires specific column names
        # user_input (question), response (answer), retrieved_contexts, reference (ground_truth)
        ragas_data = {
            "user_input": [d.get('question', "") for d in data],
            "response": [d.get('answer', "") for d in data],
            "retrieved_contexts": [d.get('contexts', []) for d in data],
            # Use answer as reference if not provided
            "reference": [d.get('ground_truth', d.get('answer', "")) for d in data] 
        }
        
        dataset = Dataset.from_dict(ragas_data)
        
        # Wrap LLM and embeddings for RAGAS
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            
            # Configure parallel execution
            try:
                from ragas import RunConfig
                # Increase parallelism for faster evaluation
                # max_workers=64 for high throughput, timeout=300 for long contexts
                run_config = RunConfig(
                    max_workers=64,  # Parallel LLM calls
                    timeout=300,     # 5 min timeout per call
                    max_retries=3,   # Retry on failures
                )
                print(f"  Using parallel execution with {run_config.max_workers} workers...")
            except ImportError:
                run_config = None
            
            if LangchainLLMWrapper and LangchainEmbeddingsWrapper:
                ragas_llm = LangchainLLMWrapper(self.llm)
                ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
                
                eval_kwargs = {
                    "dataset": dataset,
                    "metrics": self.ragas_metrics,
                    "llm": ragas_llm,
                    "embeddings": ragas_embeddings,
                }
                if run_config:
                    eval_kwargs["run_config"] = run_config
                
                results = evaluate(**eval_kwargs)
            else:
                # Fallback for older RAGAS versions
                results = evaluate(
                    dataset=dataset,
                    metrics=self.ragas_metrics,
                    llm=self.llm,
                    embeddings=self.embeddings,
                )
        
        return results.to_pandas()

    # ==========================================
    # 2. CUSTOM: REASONING COMPLEXITY (MULTI-HOP) - LLM-as-a-Judge
    # ==========================================
    def evaluate_multihop_reasoning(self, question: str, answer: str, contexts: List[str]):
        """
        Uses LLM-as-a-Judge to determine if a question is truly multi-hop.
        Returns: Dict with hop_count (int), reasoning_score (float 0-1), bridge_entity (str)
        """
        import re
        
        if "multihop_reasoning" not in PROMPTS_METRICS:
            raise ValueError("PROMPTS_METRICS['multihop_reasoning'] not found in prompt.py")
        prompt_template = PROMPTS_METRICS["multihop_reasoning"]
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({
                "contexts": "\n".join(contexts), 
                "question": question, 
                "answer": answer
            })
            content = response.content.strip()
            
            # Parse the response
            hop_match = re.search(r'HOP_COUNT:\s*(\d+)', content)
            score_match = re.search(r'REASONING_SCORE:\s*([\d.]+)', content)
            bridge_match = re.search(r'BRIDGE_ENTITY:\s*(.+?)(?:\n|$)', content)
            
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
            print(f"Error in multihop eval: {e}")
            return {"hop_count": 1, "reasoning_score": 0.0, "bridge_entity": "Error"}

    # ==========================================
    # 3. CUSTOM: VISUAL DEPENDENCY (BLIND TEST) - LLM-as-a-Judge
    # ==========================================
    def evaluate_visual_dependency(self, question: str, text_contexts: List[str]):
        """
        The 'Blind Test': Can the question be answered using ONLY text contexts?
        High Score (1.0) = Good for Multimodal (Model FAILED to answer without image).
        Low Score (0.0) = Bad for Multimodal (Model could answer using text only).
        """
        if "visual_dependency" not in PROMPTS_METRICS:
            raise ValueError("PROMPTS_METRICS['visual_dependency'] not found in prompt.py")
        prompt_template = PROMPTS_METRICS["visual_dependency"]
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        chain = prompt | self.llm
        try:
            response = chain.invoke({"contexts": "\n".join(text_contexts), "question": question})
            content = response.content.strip()
            
            # If LLM says it's missing visual info, that's a PASS (1.0) for Visual Dependency
            is_dependent = "MISSING_VISUAL" in content.upper()
            return 1.0 if is_dependent else 0.0
        except Exception as e:
            print(f"Error in visual eval: {e}")
            return 0.0

    # ==========================================
    # 4. CUSTOM: MULTIMODAL VLM METRICS
    # ==========================================
    def evaluate_multimodal_faithfulness_vlm(self, question: str, answer: str, context_chunks: List[Dict]) -> Dict:
        """
        VLM-based faithfulness evaluation: Does the answer faithfully represent 
        information from BOTH text and visual contexts?
        
        Returns:
            Dict with 'score' (0-1), 'text_supported', 'visual_supported', 'explanation'
        """
        if not VLM_AVAILABLE:
            return {"score": 0.0, "text_supported": False, "visual_supported": False, 
                    "explanation": "VLM not available"}
        
        if "multimodal_faithfulness_vlm" not in PROMPTS_METRICS:
            raise ValueError("PROMPTS_METRICS['multimodal_faithfulness_vlm'] not found in prompt.py")
        prompt = PROMPTS_METRICS["multimodal_faithfulness_vlm"].format(question=question, answer=answer)
        
        try:
            response = call_vlm_interweaved(prompt, context_chunks)
            
            # Parse response
            text_supported = "TEXT_SUPPORTED: YES" in response.upper()
            visual_supported = "VISUAL_SUPPORTED: YES" in response.upper()
            visual_na = "VISUAL_SUPPORTED: NA" in response.upper()
            
            # Extract score
            import re
            score_match = re.search(r'FAITHFULNESS_SCORE:\s*([\d.]+)', response)
            score = float(score_match.group(1)) if score_match else 0.5
            score = min(1.0, max(0.0, score))
            
            # Extract explanation
            exp_match = re.search(r'EXPLANATION:\s*(.+?)(?:\n|$)', response, re.DOTALL)
            explanation = exp_match.group(1).strip() if exp_match else ""
            
            return {
                "score": score,
                "text_supported": text_supported,
                "visual_supported": visual_supported if not visual_na else None,
                "explanation": explanation[:200]
            }
        except Exception as e:
            print(f"Error in multimodal faithfulness eval: {e}")
            return {"score": 0.0, "text_supported": False, "visual_supported": False, 
                    "explanation": f"Error: {str(e)}"}

    def evaluate_multimodal_answer_quality_vlm(self, question: str, answer: str, context_chunks: List[Dict]) -> Dict:
        """
        VLM-based answer quality evaluation considering multimodal context.
        
        Returns:
            Dict with 'completeness', 'accuracy', 'uses_visual_info', 'overall_score'
        """
        if not VLM_AVAILABLE:
            return {"completeness": 0.0, "accuracy": 0.0, "uses_visual_info": False, "overall_score": 0.0}
        
        if "multimodal_answer_quality_vlm" not in PROMPTS_METRICS:
            raise ValueError("PROMPTS_METRICS['multimodal_answer_quality_vlm'] not found in prompt.py")
        prompt = PROMPTS_METRICS["multimodal_answer_quality_vlm"].format(question=question, answer=answer)
        
        try:
            response = call_vlm_interweaved(prompt, context_chunks)
            
            # Parse scores
            import re
            completeness = 0.5
            accuracy = 0.5
            overall = 0.5
            
            comp_match = re.search(r'COMPLETENESS:\s*([\d.]+)', response)
            if comp_match:
                completeness = min(1.0, max(0.0, float(comp_match.group(1))))
            
            acc_match = re.search(r'ACCURACY:\s*([\d.]+)', response)
            if acc_match:
                accuracy = min(1.0, max(0.0, float(acc_match.group(1))))
            
            overall_match = re.search(r'OVERALL_SCORE:\s*([\d.]+)', response)
            if overall_match:
                overall = min(1.0, max(0.0, float(overall_match.group(1))))
            
            uses_visual = "VISUAL_INFO_USED: YES" in response.upper()
            
            return {
                "completeness": completeness,
                "accuracy": accuracy,
                "uses_visual_info": uses_visual,
                "overall_score": overall
            }
        except Exception as e:
            print(f"Error in multimodal answer quality eval: {e}")
            return {"completeness": 0.0, "accuracy": 0.0, "uses_visual_info": False, "overall_score": 0.0}

    # ==========================================
    # 5. CUSTOM: DATASET DIVERSITY
    # ==========================================
    def evaluate_semantic_diversity(self, questions: List[str]):
        """
        Calculates diversity based on cosine distance of question embeddings.
        Returns: diversity_score (0-1, higher is better)
        """
        print("--- Calculating Semantic Diversity ---")
        if not questions:
            return 0.0
            
        embeddings = self.embeddings.embed_documents(questions)
        matrix = np.array(embeddings)
        
        # Calculate cosine similarity matrix
        sim_matrix = cosine_similarity(matrix)
        
        # We want diversity (distance), so we look at 1 - average_similarity
        # Exclude diagonal (self-similarity is always 1)
        np.fill_diagonal(sim_matrix, np.nan) 
        avg_similarity = np.nanmean(sim_matrix)
        
        diversity_score = 1 - avg_similarity
        return diversity_score

    # ==========================================
    # 6. CUSTOM: CONTEXT NECESSITY (Anti-Parametric Bias)
    # ==========================================
    def evaluate_context_necessity(self, question: str, answer: str, context: str) -> Dict:
        """
        Measures if the question REQUIRES the context to be answered correctly.
        Tests anti-parametric bias by checking if LLM can answer without context.
        
        Objective: Ensure the question tests retrieval, not just parametric knowledge.
        
        Mathematical Intuition: Maximizes information gain from context.
        High score = context is necessary (good for RAG evaluation).
        Low score = answerable from parametric knowledge (bad for RAG evaluation).
        
        Input:
            - question: The question string
            - answer: The ground truth answer
            - context: The provided context
            
        Output:
            Dict with:
            - context_necessity_score: Float 0-1 (1 = context essential, 0 = not needed)
            - without_context_correct: Boolean (did LLM answer correctly without context?)
            - with_context_correct: Boolean (did LLM answer correctly with context?)
            - explanation: String explaining the assessment
            
        Interpretation:
            - 0.8-1.0: Excellent - question strictly requires context
            - 0.5-0.8: Moderate - context helps but partial answers possible
            - 0.0-0.5: Poor - answerable from parametric knowledge
        """
        # Step 1: Ask LLM to answer WITHOUT context
        if "context_necessity_without" not in PROMPTS_METRICS:
            raise ValueError("PROMPTS_METRICS['context_necessity_without'] not found in prompt.py")
        prompt_without_template = PROMPTS_METRICS["context_necessity_without"]
        prompt_without = ChatPromptTemplate.from_template(prompt_without_template)
        
        # Step 2: Ask LLM to verify answer WITH context  
        if "context_necessity_verify" not in PROMPTS_METRICS:
            raise ValueError("PROMPTS_METRICS['context_necessity_verify'] not found in prompt.py")
        prompt_verify_template = PROMPTS_METRICS["context_necessity_verify"]
        prompt_verify = ChatPromptTemplate.from_template(prompt_verify_template)
        
        try:
            # Get answer without context
            chain_without = prompt_without | self.llm
            response_without = chain_without.invoke({"question": question})
            answer_without_context = response_without.content.strip()
            
            # Check if model refused to answer
            refused = "CANNOT_ANSWER" in answer_without_context.upper()
            
            if refused:
                # Model couldn't answer without context - high context necessity
                return {
                    "context_necessity_score": 1.0,
                    "without_context_correct": False,
                    "with_context_correct": True,  # Assumed since we have the answer
                    "answer_without_context": answer_without_context[:200],
                    "explanation": "Model could not answer without context - context is essential"
                }
            
            # Verify if the answer without context matches ground truth
            chain_verify = prompt_verify | self.llm
            verify_response = chain_verify.invoke({
                "ground_truth": answer,
                "model_answer": answer_without_context
            })
            verify_content = verify_response.content.strip().upper()
            
            if "MATCH: YES" in verify_content:
                # Model answered correctly without context - low context necessity
                return {
                    "context_necessity_score": 0.0,
                    "without_context_correct": True,
                    "with_context_correct": True,
                    "answer_without_context": answer_without_context[:200],
                    "explanation": "Model answered correctly without context - question may test parametric knowledge"
                }
            elif "MATCH: PARTIAL" in verify_content:
                # Partial match - moderate context necessity
                return {
                    "context_necessity_score": 0.5,
                    "without_context_correct": False,
                    "with_context_correct": True,
                    "answer_without_context": answer_without_context[:200],
                    "explanation": "Model partially answered without context - context adds value"
                }
            else:
                # No match - high context necessity
                return {
                    "context_necessity_score": 0.9,
                    "without_context_correct": False,
                    "with_context_correct": True,
                    "answer_without_context": answer_without_context[:200],
                    "explanation": "Model answered incorrectly without context - context is necessary"
                }
                
        except Exception as e:
            print(f"Error in context necessity eval: {e}")
            return {
                "context_necessity_score": 0.5,
                "without_context_correct": None,
                "with_context_correct": None,
                "answer_without_context": "",
                "explanation": f"Error: {str(e)}"
            }
    
    def batch_evaluate_context_necessity(self, qa_items: List[Dict]) -> List[Dict]:
        """
        Batch evaluation of context necessity using concurrent API calls.
        
        Args:
            qa_items: List of dicts with 'question', 'answer', 'context' keys
            
        Returns:
            List of result dicts in same order
        """
        if not BATCH_AVAILABLE:
            # Fallback to sequential
            return [self.evaluate_context_necessity(
                item['question'], item['answer'], item['context']
            ) for item in qa_items]
        
        if "context_necessity_without" not in PROMPTS_METRICS:
            raise ValueError("PROMPTS_METRICS['context_necessity_without'] not found")
        if "context_necessity_verify" not in PROMPTS_METRICS:
            raise ValueError("PROMPTS_METRICS['context_necessity_verify'] not found")
        
        prompt_without_template = PROMPTS_METRICS["context_necessity_without"]
        prompt_verify_template = PROMPTS_METRICS["context_necessity_verify"]
        
        # Phase 1: Batch "answer without context" calls
        prompts_without = []
        for item in qa_items:
            prompt = prompt_without_template.replace("{question}", item['question'])
            prompts_without.append(prompt)
        
        print(f"  Phase 1: Batch answering {len(prompts_without)} questions without context...")
        answers_without = batch_call_llm(prompts_without, show_progress=False)
        
        # Phase 2: Batch verification calls for non-refused answers
        verify_prompts = []
        verify_indices = []
        results = [None] * len(qa_items)
        
        for i, (item, answer_without) in enumerate(zip(qa_items, answers_without)):
            if answer_without.startswith("ERROR:"):
                results[i] = {
                    "context_necessity_score": 0.5,
                    "without_context_correct": None,
                    "with_context_correct": None,
                    "answer_without_context": "",
                    "explanation": f"Error: {answer_without}"
                }
            elif "CANNOT_ANSWER" in answer_without.upper():
                results[i] = {
                    "context_necessity_score": 1.0,
                    "without_context_correct": False,
                    "with_context_correct": True,
                    "answer_without_context": answer_without[:200],
                    "explanation": "Model could not answer without context - context is essential"
                }
            else:
                # Need to verify
                prompt = prompt_verify_template.replace(
                    "{ground_truth}", item['answer']
                ).replace("{model_answer}", answer_without)
                verify_prompts.append(prompt)
                verify_indices.append(i)
        
        if verify_prompts:
            print(f"  Phase 2: Batch verifying {len(verify_prompts)} answers...")
            verify_responses = batch_call_llm(verify_prompts, show_progress=False)
            
            for idx, verify_content in zip(verify_indices, verify_responses):
                answer_without = answers_without[idx]
                verify_upper = verify_content.upper() if verify_content else ""
                
                if "MATCH: YES" in verify_upper:
                    results[idx] = {
                        "context_necessity_score": 0.0,
                        "without_context_correct": True,
                        "with_context_correct": True,
                        "answer_without_context": answer_without[:200],
                        "explanation": "Model answered correctly without context - question may test parametric knowledge"
                    }
                elif "MATCH: PARTIAL" in verify_upper:
                    results[idx] = {
                        "context_necessity_score": 0.5,
                        "without_context_correct": False,
                        "with_context_correct": True,
                        "answer_without_context": answer_without[:200],
                        "explanation": "Model partially answered without context - context adds value"
                    }
                else:
                    results[idx] = {
                        "context_necessity_score": 0.9,
                        "without_context_correct": False,
                        "with_context_correct": True,
                        "answer_without_context": answer_without[:200],
                        "explanation": "Model answered incorrectly without context - context is necessary"
                    }
        
        return results
    
    def batch_evaluate_multihop_reasoning(self, qa_items: List[Dict]) -> List[Dict]:
        """
        Batch evaluation of multihop reasoning using concurrent API calls.
        
        Args:
            qa_items: List of dicts with 'question', 'answer', 'contexts' keys
            
        Returns:
            List of result dicts with hop_count, reasoning_score, bridge_entity
        """
        if not BATCH_AVAILABLE:
            # Fallback to sequential
            return [self.evaluate_multihop_reasoning(
                item['question'], item['answer'], item['contexts']
            ) for item in qa_items]
        
        if "multihop_reasoning" not in PROMPTS_METRICS:
            raise ValueError("PROMPTS_METRICS['multihop_reasoning'] not found")
        
        prompt_template = PROMPTS_METRICS["multihop_reasoning"]
        
        prompts = []
        for item in qa_items:
            contexts_str = "\n".join(item['contexts']) if isinstance(item['contexts'], list) else item['contexts']
            prompt = prompt_template.replace(
                "{contexts}", contexts_str
            ).replace("{question}", item['question']).replace("{answer}", item['answer'])
            prompts.append(prompt)
        
        print(f"  Batch evaluating {len(prompts)} multihop reasoning questions...")
        responses = batch_call_llm(prompts, show_progress=False)
        
        results = []
        import re
        for response in responses:
            if response.startswith("ERROR:"):
                results.append({"hop_count": 1, "reasoning_score": 0.0, "bridge_entity": "Error"})
                continue
            
            hop_match = re.search(r'HOP_COUNT:\s*(\d+)', response)
            score_match = re.search(r'REASONING_SCORE:\s*([\d.]+)', response)
            bridge_match = re.search(r'BRIDGE_ENTITY:\s*(.+?)(?:\n|$)', response)
            
            hop_count = int(hop_match.group(1)) if hop_match else 1
            reasoning_score = float(score_match.group(1)) if score_match else 0.5
            reasoning_score = min(1.0, max(0.0, reasoning_score))
            bridge_entity = bridge_match.group(1).strip() if bridge_match else "None"
            
            results.append({
                "hop_count": hop_count,
                "reasoning_score": reasoning_score,
                "bridge_entity": bridge_entity
            })
        
        return results

    # ==========================================
    # 7. CUSTOM: DOMAIN COVERAGE
    # ==========================================
    def evaluate_domain_coverage(self, qa_data: List[Dict], corpus_chunks: List[Dict]) -> Dict:
        """
        Measures how well the QA dataset covers the source corpus.
        Prevents sampling bias and ensures comprehensive evaluation.
        
        Objective: Ensure QA dataset comprehensively tests knowledge across the corpus.
        
        Mathematical Intuition: Minimizes Jensen-Shannon divergence between
        topic distributions: min D_JS(P_topics(D) || P_topics(C))
        
        Input:
            - qa_data: List of QA pairs with chunk references
            - corpus_chunks: List of all corpus chunks with metadata
            
        Output:
            Dict with:
            - chunk_coverage: Float 0-1 (proportion of corpus chunks covered)
            - file_coverage: Float 0-1 (proportion of source files covered)
            - chunk_type_coverage: Dict (coverage by chunk type)
            - topic_divergence: Float 0-1 (JS divergence, lower is better)
            - uncovered_chunks: Int (number of chunks not referenced)
            - coverage_by_file: Dict (coverage breakdown by file)
            
        Interpretation:
            - chunk_coverage 0.8+: Excellent corpus coverage
            - chunk_coverage 0.5-0.8: Moderate coverage, some gaps
            - chunk_coverage <0.5: Poor coverage, significant gaps
            - topic_divergence <0.2: Good topic balance
            - topic_divergence >0.5: Significant topic imbalance
        """
        # entropy imported at module level from scipy.stats
        
        # Build corpus index
        corpus_index = {}
        corpus_by_file = Counter()
        corpus_by_type = Counter()
        
        for chunk in corpus_chunks:
            key = (chunk.get('file_name'), str(chunk.get('chunk_id')))
            corpus_index[key] = chunk
            corpus_by_file[chunk.get('file_name', 'unknown')] += 1
            corpus_by_type[chunk.get('chunk_type', 'unknown')] += 1
        
        # Track covered chunks
        covered_chunks = set()
        covered_by_file = Counter()
        covered_by_type = Counter()
        
        for qa in qa_data:
            for chunk_ref in qa.get('chunks_added', []):
                key = (chunk_ref.get('file_name'), str(chunk_ref.get('chunk_id')))
                if key in corpus_index:
                    covered_chunks.add(key)
                    chunk_info = corpus_index[key]
                    covered_by_file[chunk_info.get('file_name', 'unknown')] += 1
                    covered_by_type[chunk_info.get('chunk_type', 'unknown')] += 1
        
        # Calculate coverage metrics
        total_corpus = len(corpus_chunks)
        total_covered = len(covered_chunks)
        chunk_coverage = total_covered / total_corpus if total_corpus > 0 else 0.0
        
        # File coverage
        files_in_corpus = set(corpus_by_file.keys())
        files_covered = set(covered_by_file.keys())
        file_coverage = len(files_covered) / len(files_in_corpus) if files_in_corpus else 0.0
        
        # Coverage by file
        coverage_by_file = {}
        for file_name in files_in_corpus:
            total_in_file = corpus_by_file[file_name]
            covered_in_file = len([k for k in covered_chunks if k[0] == file_name])
            coverage_by_file[file_name] = {
                "total_chunks": total_in_file,
                "covered_chunks": covered_in_file,
                "coverage_rate": covered_in_file / total_in_file if total_in_file > 0 else 0.0
            }
        
        # Coverage by chunk type
        chunk_type_coverage = {}
        for chunk_type in corpus_by_type.keys():
            total_of_type = corpus_by_type[chunk_type]
            covered_of_type = sum(1 for k in covered_chunks 
                                  if corpus_index.get(k, {}).get('chunk_type') == chunk_type)
            chunk_type_coverage[chunk_type] = {
                "total": total_of_type,
                "covered": covered_of_type,
                "coverage_rate": covered_of_type / total_of_type if total_of_type > 0 else 0.0
            }
        
        # Calculate Jensen-Shannon divergence for topic distribution
        # Using file distribution as proxy for topic distribution
        all_files = sorted(files_in_corpus)
        corpus_dist = np.array([corpus_by_file.get(f, 0) for f in all_files], dtype=float)
        corpus_dist = corpus_dist / corpus_dist.sum() if corpus_dist.sum() > 0 else corpus_dist
        
        qa_file_counts = Counter()
        for qa in qa_data:
            for chunk_ref in qa.get('chunks_added', []):
                qa_file_counts[chunk_ref.get('file_name')] += 1
        
        qa_dist = np.array([qa_file_counts.get(f, 0) for f in all_files], dtype=float)
        qa_dist = qa_dist / qa_dist.sum() if qa_dist.sum() > 0 else qa_dist
        
        # Jensen-Shannon divergence (symmetric KL divergence)
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        corpus_dist = corpus_dist + eps
        qa_dist = qa_dist + eps
        corpus_dist = corpus_dist / corpus_dist.sum()
        qa_dist = qa_dist / qa_dist.sum()
        
        m = 0.5 * (corpus_dist + qa_dist)
        js_divergence = 0.5 * (entropy(corpus_dist, m) + entropy(qa_dist, m))
        
        return {
            "chunk_coverage": chunk_coverage,
            "file_coverage": file_coverage,
            "chunks_covered": total_covered,
            "chunks_total": total_corpus,
            "uncovered_chunks": total_corpus - total_covered,
            "topic_divergence_js": float(js_divergence),
            "chunk_type_coverage": chunk_type_coverage,
            "coverage_by_file": coverage_by_file
        }

    # ==========================================
    # MAIN PIPELINE
    # ==========================================
    def run_full_evaluation(self, json_path: str, output_path: str = "eval_report.json"):
        """
        Runs the full suite of metrics on the provided JSON dataset.
        """
        data = self.load_dataset(json_path)
        metrics_log = []
        
        print(f"Starting evaluation for {len(data)} items...")

        # 1. Run Standard Metrics (Batch)
        # Note: Ragas requires 'ground_truth' for some metrics. 
        # If your dataset is purely synthetic without human labels, 
        # context_precision/recall might be approximations based on generated answers.
        ragas_df = self.evaluate_ragas_standard(data)
        
        # 2. Run Custom Agentic Metrics (Iterative)
        questions = [d['question'] for d in data]
        
        for i, item in enumerate(data):
            if i % 5 == 0:
                print(f"Processing item {i+1}/{len(data)}...")
            
            # A. Multi-hop Evaluation
            mh_res = self.evaluate_multihop_reasoning(
                item['question'], 
                item['answer'], 
                item['contexts']
            )
            
            # B. Visual Dependency (Only for items marked as visual/multimodal)
            # Checks metadata type or if image_contexts exist
            is_visual = (
                item.get('metadata', {}).get('type') in ['visual', 'chart', 'table'] or 
                len(item.get('image_contexts', [])) > 0
            )
            
            vis_score = 0.0
            if is_visual:
                # For the blind test, we pass ONLY text contexts, excluding image descriptions
                vis_score = self.evaluate_visual_dependency(item['question'], item['contexts'])
            
            # C. VLM-based Multimodal Metrics (only if images available)
            context_chunks = item.get('context_chunks', [])
            vlm_faithfulness = None
            vlm_answer_quality = None
            
            if is_visual and VLM_AVAILABLE and context_chunks:
                print(f"  Running VLM multimodal evaluation for item {i+1}...")
                vlm_faithfulness = self.evaluate_multimodal_faithfulness_vlm(
                    item['question'], item['answer'], context_chunks
                )
                vlm_answer_quality = self.evaluate_multimodal_answer_quality_vlm(
                    item['question'], item['answer'], context_chunks
                )
            
            metrics_log.append({
                "hop_count": mh_res['hop_count'],
                "reasoning_score": mh_res['reasoning_score'],
                "bridge_entity": mh_res['bridge_entity'],
                "visual_dependency": vis_score if is_visual else None,
                "is_visual": is_visual,
                # VLM Multimodal metrics
                "vlm_faithfulness_score": vlm_faithfulness['score'] if vlm_faithfulness else None,
                "vlm_text_supported": vlm_faithfulness['text_supported'] if vlm_faithfulness else None,
                "vlm_visual_supported": vlm_faithfulness['visual_supported'] if vlm_faithfulness else None,
                "vlm_completeness": vlm_answer_quality['completeness'] if vlm_answer_quality else None,
                "vlm_accuracy": vlm_answer_quality['accuracy'] if vlm_answer_quality else None,
                "vlm_uses_visual": vlm_answer_quality['uses_visual_info'] if vlm_answer_quality else None,
                "vlm_overall_score": vlm_answer_quality['overall_score'] if vlm_answer_quality else None,
            })

        # 3. Diversity Evaluation
        diversity_score = self.evaluate_semantic_diversity(questions)
        
        # 4. Aggregate Results
        custom_df = pd.DataFrame(metrics_log)
        final_df = pd.concat([ragas_df, custom_df], axis=1)
        
        # Calculate Summary Statistics
        rag_quality = {
            "Faithfulness": final_df['faithfulness'].mean(),
            "Answer_Relevance": final_df['answer_relevancy'].mean() if 'answer_relevancy' in final_df.columns else final_df.get('answer_relevance', pd.Series([0])).mean(),
            "Context_Precision": final_df['context_precision'].mean(),
            "Context_Recall": final_df['context_recall'].mean(),
        }
        
        # Add optional metrics if they were computed
        if HAS_ENTITY_RECALL and 'context_entity_recall' in final_df.columns:
            rag_quality["Context_Entity_Recall"] = final_df['context_entity_recall'].mean()
        if HAS_NOISE_SENSITIVITY and 'noise_sensitivity_relevant' in final_df.columns:
            rag_quality["Noise_Sensitivity"] = final_df['noise_sensitivity_relevant'].mean()
        
        multimodal_quality = {
            "Visual_Necessity_Rate": final_df[final_df['is_visual'] == True]['visual_dependency'].mean() 
            if not final_df[final_df['is_visual'] == True].empty else 0.0
        }
        
        # Add multimodal ragas metrics if available
        if HAS_MULTIMODAL:
            if 'multimodal_faithfulness' in final_df.columns:
                multimodal_quality["Multimodal_Faithfulness"] = final_df['multimodal_faithfulness'].mean()
            if 'multimodal_relevance' in final_df.columns:
                multimodal_quality["Multimodal_Relevance"] = final_df['multimodal_relevance'].mean()
        
        # Add VLM-based multimodal metrics
        visual_items = final_df[final_df['is_visual'] == True]
        if not visual_items.empty:
            if 'vlm_faithfulness_score' in final_df.columns:
                vlm_faith_scores = visual_items['vlm_faithfulness_score'].dropna()
                if len(vlm_faith_scores) > 0:
                    multimodal_quality["VLM_Faithfulness_Score"] = vlm_faith_scores.mean()
            if 'vlm_overall_score' in final_df.columns:
                vlm_overall_scores = visual_items['vlm_overall_score'].dropna()
                if len(vlm_overall_scores) > 0:
                    multimodal_quality["VLM_Overall_Quality"] = vlm_overall_scores.mean()
            if 'vlm_accuracy' in final_df.columns:
                vlm_accuracy_scores = visual_items['vlm_accuracy'].dropna()
                if len(vlm_accuracy_scores) > 0:
                    multimodal_quality["VLM_Accuracy"] = vlm_accuracy_scores.mean()
            if 'vlm_completeness' in final_df.columns:
                vlm_completeness_scores = visual_items['vlm_completeness'].dropna()
                if len(vlm_completeness_scores) > 0:
                    multimodal_quality["VLM_Completeness"] = vlm_completeness_scores.mean()
            # Count items using visual info
            if 'vlm_uses_visual' in final_df.columns:
                uses_visual_count = visual_items['vlm_uses_visual'].sum()
                multimodal_quality["Items_Using_Visual_Info"] = int(uses_visual_count)
                multimodal_quality["Visual_Info_Usage_Rate"] = uses_visual_count / len(visual_items) if len(visual_items) > 0 else 0.0
        
        report = {
            "RAG_Quality": rag_quality,
            "Reasoning_Complexity": {
                "Avg_Reasoning_Score": final_df['reasoning_score'].mean(),
                "Avg_Hop_Count": final_df['hop_count'].mean(),
            },
            "Multimodal_Quality": multimodal_quality,
            "Dataset_Health": {
                "Semantic_Diversity": diversity_score,
                "Total_Samples": len(data)
            }
        }
        
        # Save detailed results
        final_df.to_csv(output_path.replace(".json", "_detailed.csv"), index=False)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=4)
            
        print("\nEvaluation Complete. Summary:")
        print(json.dumps(report, indent=2))
        
        return final_df, report

def transform_qa_data(raw_data: List[Dict]) -> List[Dict]:
    """
    Transform qa_multihop_pass.json format to the format expected by the evaluator.
    
    Input format (from qa_multihop_pass.json):
        - chunk_id, original_chunk, final_context, context_chunks, context_status, 
        - depth_reached, chunks_added, expert_persona, domain, question, answer, 
        - relevance_score, difficulty_score, selection_status, selection_reason, verification_result
    
    Expected format:
        - question, answer, contexts (list), ground_truth, metadata, image_contexts, context_chunks
    """
    transformed = []
    for item in raw_data:
        # Extract context_chunks which contain both text content and image_path
        context_chunks = item.get("context_chunks", [])
        
        # Extract text contexts from context_chunks or fall back to final_context
        if context_chunks:
            contexts = [chunk.get("content", "") for chunk in context_chunks if chunk.get("content")]
        else:
            contexts = [item.get("final_context", item.get("original_chunk", ""))]
        
        # Extract image paths from context_chunks
        image_contexts = []
        for chunk in context_chunks:
            img_path = chunk.get("image_path")
            if img_path and img_path != "null":
                image_contexts.append(img_path)
        
        # Determine if this is a visual/multimodal item
        has_images = len(image_contexts) > 0
        item_type = "visual" if has_images else "text"
        
        transformed.append({
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "contexts": contexts,
            "ground_truth": item.get("answer", ""),  # Using answer as ground_truth (no human labels)
            "metadata": {
                "chunk_id": item.get("chunk_id"),
                "expert_persona": item.get("expert_persona"),
                "domain": item.get("domain"),
                "context_status": item.get("context_status"),
                "relevance_score": item.get("relevance_score"),
                "difficulty_score": item.get("difficulty_score"),
                "selection_status": item.get("selection_status"),
                "depth_reached": item.get("depth_reached"),
                "chunks_added": item.get("chunks_added", []),
                "type": item_type,
            },
            "image_contexts": image_contexts,
            "context_chunks": context_chunks,  # Keep full chunks for VLM calls
        })
    return transformed


def analyze_missing_information(raw_data: List[Dict]) -> Dict[str, Any]:
    """
    Analyze which metrics cannot be fully evaluated and what information is missing.
    """
    missing_info = {
        "metrics_status": {},
        "missing_fields": [],
        "recommendations": [],
        "image_stats": {}
    }
    
    # Check what's available
    sample = raw_data[0] if raw_data else {}
    available_fields = set(sample.keys())
    
    # Count items with images in context_chunks
    items_with_images = 0
    total_images = 0
    for item in raw_data:
        context_chunks = item.get("context_chunks", [])
        item_has_image = False
        for chunk in context_chunks:
            img_path = chunk.get("image_path")
            if img_path and img_path != "null" and img_path is not None:
                total_images += 1
                item_has_image = True
        if item_has_image:
            items_with_images += 1
    
    missing_info["image_stats"] = {
        "total_items": len(raw_data),
        "items_with_images": items_with_images,
        "total_images": total_images,
        "has_context_chunks": "context_chunks" in available_fields
    }
    
    has_images = items_with_images > 0
    
    # 1. RAGAS Standard Metrics
    missing_info["metrics_status"]["faithfulness"] = {
        "can_evaluate": True,
        "quality": "FULL",
        "notes": "Question, answer, and context available."
    }
    missing_info["metrics_status"]["answer_relevance"] = {
        "can_evaluate": True,
        "quality": "FULL",
        "notes": "Question and answer available."
    }
    missing_info["metrics_status"]["context_precision"] = {
        "can_evaluate": True,
        "quality": "APPROXIMATED",
        "notes": "No human-labeled ground_truth. Using generated answer as proxy.",
        "missing": ["Human-annotated ground truth answers"]
    }
    missing_info["metrics_status"]["context_recall"] = {
        "can_evaluate": True,
        "quality": "APPROXIMATED", 
        "notes": "No human-labeled ground_truth. Using generated answer as proxy.",
        "missing": ["Human-annotated ground truth answers"]
    }
    
    # 2. Multi-hop Reasoning Metric
    missing_info["metrics_status"]["multihop_reasoning"] = {
        "can_evaluate": True,
        "quality": "FULL",
        "notes": "Question, answer, and context available for LLM-as-a-Judge evaluation."
    }
    
    # 3. Visual Dependency Metric
    if has_images:
        missing_info["metrics_status"]["visual_dependency"] = {
            "can_evaluate": True,
            "quality": "FULL",
            "notes": f"Found {items_with_images} items with {total_images} images in context_chunks."
        }
    else:
        missing_info["metrics_status"]["visual_dependency"] = {
            "can_evaluate": False,
            "quality": "NOT_APPLICABLE",
            "notes": "No image data found in context_chunks.",
            "missing": [
                "context_chunks[].image_path: Image file paths in context chunks",
                "Ensure source chunks have 'artifact' fields with image references"
            ]
        }
    
    # 4. Multimodal VLM Metrics (custom implementation)
    if has_images:
        missing_info["metrics_status"]["multimodal_faithfulness_vlm"] = {
            "can_evaluate": True,
            "quality": "FULL",
            "notes": f"VLM-based evaluation using {total_images} images from context_chunks."
        }
        missing_info["metrics_status"]["multimodal_answer_quality_vlm"] = {
            "can_evaluate": True,
            "quality": "FULL",
            "notes": "VLM-based answer quality evaluation with visual context."
        }
    else:
        missing_info["metrics_status"]["multimodal_faithfulness_vlm"] = {
            "can_evaluate": False,
            "quality": "NOT_APPLICABLE",
            "notes": "Requires image data in context_chunks.",
            "missing": ["context_chunks[].image_path: Image file paths"]
        }
        missing_info["metrics_status"]["multimodal_answer_quality_vlm"] = {
            "can_evaluate": False,
            "quality": "NOT_APPLICABLE", 
            "notes": "Requires image data in context_chunks.",
            "missing": ["context_chunks[].image_path: Image file paths"]
        }
    
    # 5. Semantic Diversity
    missing_info["metrics_status"]["semantic_diversity"] = {
        "can_evaluate": True,
        "quality": "FULL",
        "notes": "Questions available for embedding-based diversity calculation."
    }
    
    # 6. Context Necessity (Anti-Parametric Bias)
    missing_info["metrics_status"]["context_necessity"] = {
        "can_evaluate": True,
        "quality": "FULL",
        "notes": "Question, answer, and context available for LLM-based necessity evaluation."
    }
    
    # 7. Domain Coverage
    missing_info["metrics_status"]["domain_coverage"] = {
        "can_evaluate": True,
        "quality": "FULL" if "chunks_added" in available_fields else "LIMITED",
        "notes": "Chunk references available for coverage calculation. Requires corpus chunks.json."
    }
    
    # Recommendations
    recommendations = [
        "Add 'ground_truth' field with human-annotated answers for accurate context_precision/recall"
    ]
    if not has_images:
        recommendations.append("Ensure source chunks have 'artifact' fields with image paths for multimodal metrics")
        recommendations.append("Re-run QA generation with updated context_retrieved.py to capture images")
    
    missing_info["recommendations"] = recommendations
    
    return missing_info


def identify_qa_subsets(raw_data: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Identify QA subsets: multihop, multimodal, and their intersection.
    
    Returns:
        Dict with keys: 'all', 'multihop', 'multimodal', 'multihop_and_multimodal'
    """
    all_qa = raw_data
    
    # Multihop: hop_count > 0 (each hop adds one link between chunks)
    # hop_count = len(chunks_added) - 1
    multihop = [
        qa for qa in raw_data 
        if qa.get('hop_count', 0) > 0
    ]
    
    # Multimodal: content mentions figures/tables/images OR has image artifacts
    multimodal = []
    multimodal_keywords = ['figure', 'diagram', 'table', 'image', 'chart', '![image]', 'block diagram']
    
    for qa in raw_data:
        content = (qa.get('original_chunk', '') + ' ' + qa.get('final_context', '')).lower()
        # Check for visual keywords in content
        has_visual_content = any(kw in content for kw in multimodal_keywords)
        # Check for explicit image paths in context_chunks
        context_chunks = qa.get('context_chunks', [])
        has_image_path = any(
            chunk.get('image_path') and chunk.get('image_path') != 'null' 
            for chunk in context_chunks
        )
        if has_visual_content or has_image_path:
            multimodal.append(qa)
    
    # Intersection
    multihop_ids = set(id(qa) for qa in multihop)
    multihop_and_multimodal = [qa for qa in multimodal if id(qa) in multihop_ids]
    
    return {
        'all': all_qa,
        'multihop': multihop,
        'multimodal': multimodal,
        'multihop_and_multimodal': multihop_and_multimodal
    }


def _count_tokens(text: str) -> int:
    """
    Count tokens using tokenizer if available, otherwise use improved approximation.

    Approximation: ~0.75 tokens per word for English (GPT-4 average).
    For better accuracy, uses tiktoken if available, otherwise word-based estimate.
    """
    if not text:
        return 0

    # Try tiktoken (fast, accurate for GPT models, handles long text without warnings)
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        return len(enc.encode(text))
    except ImportError:
        pass

    # Try transformers tokenizer with large model_max_length to avoid warnings
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", model_max_length=100000)
        return len(tokenizer.encode(text))
    except (ImportError, Exception):
        pass

    # Fallback: improved word-based approximation
    # GPT-4 average: ~0.75 tokens per word, but varies by language
    # Using 1.3 tokens per word as conservative estimate (accounts for punctuation, etc.)
    words = len(text.split())
    return int(words * 1.3)


def _count_pages_from_pdf(pdf_path: str) -> int:
    """Count pages from PDF using pypdfium2 or pypdf."""
    try:
        import pypdfium2 as pdfium
        with open(pdf_path, 'rb') as f:
            pdf = pdfium.PdfDocument(f.read())
            return len(pdf)
    except ImportError:
        try:
            import pypdf
            with open(pdf_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                return len(pdf_reader.pages)
        except ImportError:
            pass
    except Exception:
        pass
    return 0


def _count_pages_from_html_or_markdown(file_path: str) -> int:
    """
    Estimate pages from HTML or Markdown files.
    Uses content length with reasonable assumptions:
    - Average page: ~2000-2500 words or ~12,000-15,000 characters
    - For HTML: strips tags first
    """
    try:
        content = Path(file_path).read_text(encoding='utf-8')
        
        # For HTML, strip tags to get actual content
        if file_path.lower().endswith('.html') or file_path.lower().endswith('.htm'):
            import re
            # Remove HTML tags
            content = re.sub(r'<[^>]+>', '', content)
            # Remove extra whitespace
            content = ' '.join(content.split())
        
        # Estimate: ~2500 words per page or ~13,000 chars per page
        word_count = len(content.split())
        char_count = len(content)
        
        # Use word-based estimate (more reliable)
        pages_from_words = max(1, word_count // 2500)
        # Use char-based estimate as backup
        pages_from_chars = max(1, char_count // 13000)
        
        # Return average, rounded up
        return max(pages_from_words, pages_from_chars)
    except Exception:
        return 0


def compute_corpus_and_dataset_stats(
    qa_data: List[Dict],
    corpus_chunks: List[Dict] = None,
    pdf_dir: str = None,
    markdown_dir: str = None
) -> Dict[str, Any]:
    """
    Compute comprehensive corpus and dataset statistics.
    
    Computes:
    - Corpus stats: #chunks, #multimodal chunks, #tables, #images, #tokens
    - Context stats: Distribution by hop count
    - QA stats: By modality categories (Multimodal, Table, Table+Image) and hop counts
    
    Args:
        qa_data: List of QA pairs
        corpus_chunks: List of corpus chunks (from chunks.json)
        pdf_dir: Path to original PDF files directory (for accurate page count using pypdfium2/pypdf)
        markdown_dir: Path to markdown/HTML files directory (fallback for page count if PDFs not available)
        
    Returns:
        Dict with corpus_stats, context_stats, and qa_category_stats
    """
    import re
    from pathlib import Path
    
    stats = {
        "corpus_stats": {},
        "context_stats": {},
        "qa_category_stats": {}
    }
    
    # ==========================================
    # 1. CORPUS STATS (from chunks.json)
    # ==========================================
    if corpus_chunks:
        total_chunks = len(corpus_chunks)
        
        # Count by chunk type
        text_chunks = 0
        table_chunks = 0
        image_chunks = 0
        multimodal_chunks = 0  # Chunks with images (standalone or embedded)
        total_tokens = 0
        
        for chunk in corpus_chunks:
            chunk_type = chunk.get('chunk_type', 'text').lower()
            content = chunk.get('content', '')
            artifact = chunk.get('artifact', 'None')
            
            # Count tokens using proper tokenizer or improved approximation
            total_tokens += _count_tokens(content)
            
            if chunk_type == 'table':
                table_chunks += 1
            elif chunk_type == 'standalone image':
                image_chunks += 1
                multimodal_chunks += 1
            else:
                text_chunks += 1
            
            # Check for embedded images in content or artifact
            has_image = (
                artifact and artifact != 'None' and '![' in str(artifact)
            ) or '![' in content
            
            if has_image and chunk_type != 'standalone image':
                multimodal_chunks += 1
        
        stats["corpus_stats"] = {
            "total_chunks": total_chunks,
            "text_chunks": text_chunks,
            "table_chunks": table_chunks,
            "image_chunks": image_chunks,
            "multimodal_chunks": multimodal_chunks,
            "total_tokens": total_tokens,  # Renamed from total_tokens_approx
            "avg_tokens_per_chunk": round(total_tokens / total_chunks, 1) if total_chunks > 0 else 0
        }
        
        # Count unique files
        unique_files = set(chunk.get('file_name', '') for chunk in corpus_chunks)
        stats["corpus_stats"]["num_source_files"] = len(unique_files)
    
    # ==========================================
    # PAGE COUNTING (from PDFs if available, otherwise markdown/HTML)
    # ==========================================
    total_pages = 0
    pages_counted = 0
    
    # First, try to count from original PDFs (most accurate)
    if pdf_dir and os.path.exists(pdf_dir):
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        for pdf_file in pdf_files:
            page_count = _count_pages_from_pdf(str(pdf_file))
            if page_count > 0:
                total_pages += page_count
                pages_counted += 1
    
    # If no PDFs found or PDF dir not provided, try markdown files
    if total_pages == 0 and markdown_dir and os.path.exists(markdown_dir):
        md_files = list(Path(markdown_dir).rglob("*.md"))
        html_files = list(Path(markdown_dir).rglob("*.html")) + list(Path(markdown_dir).rglob("*.htm"))
        
        for md_file in md_files:
            page_count = _count_pages_from_html_or_markdown(str(md_file))
            total_pages += page_count
        
        for html_file in html_files:
            page_count = _count_pages_from_html_or_markdown(str(html_file))
            total_pages += page_count
    
    if total_pages > 0:
        stats["corpus_stats"]["total_pages"] = total_pages
        if pages_counted > 0:
            stats["corpus_stats"]["pages_counted_from_pdfs"] = pages_counted
    
    # ==========================================
    # 2. CONTEXT STATS (from QA data)
    # ==========================================
    hop_distribution = Counter()
    context_sizes = []  # Number of chunks per context
    
    for qa in qa_data:
        # Determine hop count from chunks_added
        chunks_added = qa.get('chunks_added', [])
        num_chunks = len(chunks_added) if isinstance(chunks_added, list) else 1
        context_sizes.append(num_chunks)
        
        # Hop count = number of chunks - 1 (0-hop means single chunk)
        hop_count = max(0, num_chunks - 1)
        hop_distribution[hop_count] += 1
    
    stats["context_stats"] = {
        "total_contexts": len(qa_data),
        "hop_distribution": dict(sorted(hop_distribution.items())),
        "avg_chunks_per_context": round(np.mean(context_sizes), 2) if context_sizes else 0,
        "max_chunks_in_context": max(context_sizes) if context_sizes else 0
    }
    
    # Add summary counts
    for hop in range(max(hop_distribution.keys()) + 1 if hop_distribution else 0):
        stats["context_stats"][f"num_{hop}_hop_contexts"] = hop_distribution.get(hop, 0)
    
    # ==========================================
    # 3. QA CATEGORY STATS
    # ==========================================
    # Categories based on context content:
    # - Multimodal: at least one image in context
    # - Table: at least one table in context
    # - Table+Image: both table and image in context
    
    def context_has_table(qa: Dict) -> bool:
        """Check if context contains a table."""
        context_chunks = qa.get('context_chunks', [])
        final_context = qa.get('final_context', '')
        
        # Check in context_chunks
        for chunk in context_chunks:
            content = chunk.get('content', '')
            if '|' in content and '-|' in content:  # Markdown table pattern
                return True
        
        # Check in final_context
        if '|' in final_context and '-|' in final_context:
            return True
        
        return False
    
    def context_has_image(qa: Dict) -> bool:
        """Check if context contains an image."""
        context_chunks = qa.get('context_chunks', [])
        
        for chunk in context_chunks:
            image_path = chunk.get('image_path')
            if image_path and image_path != 'null' and image_path is not None:
                return True
            content = chunk.get('content', '')
            if '![' in content:  # Markdown image pattern
                return True
        
        return False
    
    # Initialize counters
    total_qa = len(qa_data)
    multimodal_qa = []
    table_qa = []
    table_image_qa = []
    text_only_qa = []
    
    # Per-hop counters for each category
    hop_multimodal = Counter()
    hop_table = Counter()
    hop_table_image = Counter()
    hop_text_only = Counter()
    hop_all = Counter()
    
    for qa in qa_data:
        chunks_added = qa.get('chunks_added', [])
        num_chunks = len(chunks_added) if isinstance(chunks_added, list) else 1
        hop_count = max(0, num_chunks - 1)
        
        has_table = context_has_table(qa)
        has_image = context_has_image(qa)
        
        hop_all[hop_count] += 1
        
        if has_table and has_image:
            table_image_qa.append(qa)
            hop_table_image[hop_count] += 1
        elif has_image:
            multimodal_qa.append(qa)
            hop_multimodal[hop_count] += 1
        elif has_table:
            table_qa.append(qa)
            hop_table[hop_count] += 1
        else:
            text_only_qa.append(qa)
            hop_text_only[hop_count] += 1
    
    # Also count inclusive categories (for reporting)
    # Multimodal (any image): includes table_image
    multimodal_inclusive = [qa for qa in qa_data if context_has_image(qa)]
    # Table (any table): includes table_image
    table_inclusive = [qa for qa in qa_data if context_has_table(qa)]
    
    stats["qa_category_stats"] = {
        "total_qa_pairs": total_qa,
        
        # Exclusive categories (mutually exclusive)
        "text_only_qa": len(text_only_qa),
        "table_only_qa": len(table_qa),
        "image_only_qa": len(multimodal_qa),
        "table_and_image_qa": len(table_image_qa),
        
        # Inclusive categories (overlapping)
        "multimodal_qa_inclusive": len(multimodal_inclusive),  # Any QA with image
        "table_qa_inclusive": len(table_inclusive),  # Any QA with table
        
        # Hop distribution for all QA
        "qa_hop_distribution": dict(sorted(hop_all.items())),
        
        # Hop distribution by category
        "text_only_by_hop": dict(sorted(hop_text_only.items())),
        "table_only_by_hop": dict(sorted(hop_table.items())),
        "image_only_by_hop": dict(sorted(hop_multimodal.items())),
        "table_and_image_by_hop": dict(sorted(hop_table_image.items()))
    }
    
    # Add summary counts per hop
    max_hop = max(hop_all.keys()) if hop_all else 0
    for hop in range(max_hop + 1):
        stats["qa_category_stats"][f"num_{hop}_hop_qa"] = hop_all.get(hop, 0)
    
    # Compute multimodal inclusive by hop
    hop_multimodal_inclusive = Counter()
    hop_table_inclusive = Counter()
    for qa in qa_data:
        chunks_added = qa.get('chunks_added', [])
        num_chunks = len(chunks_added) if isinstance(chunks_added, list) else 1
        hop_count = max(0, num_chunks - 1)
        
        if context_has_image(qa):
            hop_multimodal_inclusive[hop_count] += 1
        if context_has_table(qa):
            hop_table_inclusive[hop_count] += 1
    
    stats["qa_category_stats"]["multimodal_inclusive_by_hop"] = dict(sorted(hop_multimodal_inclusive.items()))
    stats["qa_category_stats"]["table_inclusive_by_hop"] = dict(sorted(hop_table_inclusive.items()))
    
    return stats


def run_subset_evaluation(
    qa_data: List[Dict],
    corpus_path: str = None,
    output_dir: str = None,
    sample_size: int = None,
    run_context_necessity: bool = True,
    pdf_dir: str = None,
    markdown_dir: str = None
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation on QA dataset including new metrics.
    
    Evaluates:
    - Corpus and Dataset Statistics (chunks, modalities, hop distributions)
    - Context Necessity (anti-parametric bias)
    - Domain Coverage (corpus coverage)
    - Subset statistics (multihop, multimodal counts)
    - Targeted metrics on subsets
    
    Args:
        qa_data: List of QA pairs (raw format)
        corpus_path: Path to chunks.json for domain coverage
        output_dir: Output directory for reports
        sample_size: If set, sample this many items for expensive evaluations
        run_context_necessity: Whether to run context necessity (expensive)
        pdf_dir: Path to original PDF files directory (optional, for accurate page count)
        markdown_dir: Path to markdown/HTML files directory (optional, fallback for page count)
        
    Returns:
        Dict with evaluation results
    """
    import random
    
    results = {
        "corpus_stats": {},
        "context_stats": {},
        "qa_category_stats": {},
        "subset_statistics": {},
        "ragas_metrics": {},  # Faithfulness, Relevance, Precision, Recall
        "context_necessity": {},
        "domain_coverage": {},
        "multihop_metrics": {},
        "multimodal_metrics": {}
    }
    
    # 0. Compute Corpus and Dataset Statistics
    print("\n" + "=" * 60)
    print("CORPUS AND DATASET STATISTICS")
    print("=" * 60)
    
    corpus_chunks = None
    if corpus_path and os.path.exists(corpus_path):
        with open(corpus_path, 'r') as f:
            corpus_chunks = json.load(f)
    
    # Auto-detect directories if not provided
    pdf_dir_auto = pdf_dir  # Keep original if provided
    if corpus_path:
        base_dir = os.path.dirname(corpus_path)
        
        # Try to find PDF directory (check parent directories)
        if not pdf_dir_auto:
            # Check config for input_pdf_dir
            try:
                from mirage.core.config import get_paths_config
                paths_config = get_paths_config()
                potential_pdf_dir = paths_config.get('input_pdf_dir')
                if potential_pdf_dir and os.path.exists(potential_pdf_dir):
                    pdf_dir_auto = potential_pdf_dir
            except:
                pass
            
            # Fallback: check common locations relative to output_dir
            if not pdf_dir_auto:
                potential_pdf_dirs = [
                    os.path.join(os.path.dirname(base_dir), "data"),
                    os.path.join(base_dir, "..", "data"),
                    "data/documents"  # Default from config
                ]
                for pd in potential_pdf_dirs:
                    if os.path.exists(pd) and any(Path(pd).glob("*.pdf")):
                        pdf_dir_auto = pd
                        break
        
        # Auto-detect markdown_dir if not provided
        if not markdown_dir:
            potential_md_dir = os.path.join(base_dir, "markdown")
            if os.path.exists(potential_md_dir):
                markdown_dir = potential_md_dir
    
    comprehensive_stats = compute_corpus_and_dataset_stats(
        qa_data=qa_data,
        corpus_chunks=corpus_chunks,
        pdf_dir=pdf_dir,
        markdown_dir=markdown_dir
    )
    
    results["corpus_stats"] = comprehensive_stats["corpus_stats"]
    results["context_stats"] = comprehensive_stats["context_stats"]
    results["qa_category_stats"] = comprehensive_stats["qa_category_stats"]
    
    # Print corpus stats
    cs = results["corpus_stats"]
    if cs:
        print(f"\n  CORPUS STATS:")
        print(f"     Total Chunks: {cs.get('total_chunks', 'N/A')}")
        print(f"     Text Chunks: {cs.get('text_chunks', 'N/A')}")
        print(f"     Table Chunks: {cs.get('table_chunks', 'N/A')}")
        print(f"     Image Chunks: {cs.get('image_chunks', 'N/A')}")
        print(f"     Multimodal Chunks: {cs.get('multimodal_chunks', 'N/A')}")
        print(f"     Total Tokens: {cs.get('total_tokens', 'N/A'):,}")
        print(f"     Source Files: {cs.get('num_source_files', 'N/A')}")
        if cs.get('total_pages'):
            page_source = "PDFs" if cs.get('pages_counted_from_pdfs') else "Markdown/HTML"
            print(f"     Total Pages ({page_source}): {cs.get('total_pages')}")
    
    # Print context stats
    ctx = results["context_stats"]
    print(f"\n  CONTEXT STATS:")
    print(f"     Total Contexts: {ctx.get('total_contexts', 'N/A')}")
    print(f"     Avg Chunks/Context: {ctx.get('avg_chunks_per_context', 'N/A')}")
    print(f"     Hop Distribution: {ctx.get('hop_distribution', {})}")
    
    # Print QA category stats
    qa_cat = results["qa_category_stats"]
    print(f"\n  QA CATEGORY STATS:")
    print(f"     Total QA Pairs: {qa_cat.get('total_qa_pairs', 'N/A')}")
    print(f"     Text-only QA: {qa_cat.get('text_only_qa', 'N/A')}")
    print(f"     Table-only QA: {qa_cat.get('table_only_qa', 'N/A')}")
    print(f"     Image-only QA: {qa_cat.get('image_only_qa', 'N/A')}")
    print(f"     Table+Image QA: {qa_cat.get('table_and_image_qa', 'N/A')}")
    print(f"     Multimodal QA (inclusive): {qa_cat.get('multimodal_qa_inclusive', 'N/A')}")
    print(f"     Table QA (inclusive): {qa_cat.get('table_qa_inclusive', 'N/A')}")
    print(f"     QA Hop Distribution: {qa_cat.get('qa_hop_distribution', {})}")
    
    # 1. Identify subsets (legacy, kept for backwards compatibility)
    print("\n" + "=" * 60)
    print("SUBSET ANALYSIS")
    print("=" * 60)
    
    subsets = identify_qa_subsets(qa_data)
    
    results["subset_statistics"] = {
        "total_qa_pairs": len(subsets['all']),
        "multihop_count": len(subsets['multihop']),
        "multimodal_count": len(subsets['multimodal']),
        "multihop_and_multimodal_count": len(subsets['multihop_and_multimodal']),
        "single_hop_text_only": len(subsets['all']) - len(subsets['multihop']) - len(subsets['multimodal']) + len(subsets['multihop_and_multimodal'])
    }
    
    print(f"\n  Total QA pairs: {results['subset_statistics']['total_qa_pairs']}")
    print(f"  Multihop QA pairs: {results['subset_statistics']['multihop_count']}")
    print(f"  Multimodal QA pairs: {results['subset_statistics']['multimodal_count']}")
    print(f"  Multihop AND Multimodal: {results['subset_statistics']['multihop_and_multimodal_count']}")
    
    # 1.5. RAGAS Standard Metrics (Faithfulness, Relevance, Precision, Recall)
    print("\n" + "=" * 60)
    print("RAGAS STANDARD METRICS")
    print("=" * 60)
    
    if RAGAS_AVAILABLE:
        try:
            evaluator = MultimodalFrameworkEvaluator()
            
            # Transform data for RAGAS
            transformed_data = transform_qa_data(qa_data)
            
            # Sample if needed for expensive RAGAS evaluation
            # RAGAS with 64 parallel workers. Limit samples to manage API costs.
            RAGAS_MAX_SAMPLES = 50  # Reasonable sample for statistical significance
            eval_data = transformed_data
            ragas_sample_size = min(sample_size or RAGAS_MAX_SAMPLES, RAGAS_MAX_SAMPLES)
            if len(transformed_data) > ragas_sample_size:
                eval_data = random.sample(transformed_data, ragas_sample_size)
                print(f"\n  Sampling {ragas_sample_size}/{len(transformed_data)} items for RAGAS...")
            
            print(f"\n  Running RAGAS evaluation on {len(eval_data)} items (parallel, ~2-5 min)...")
            ragas_df = evaluator.evaluate_ragas_standard(eval_data)
            
            # Extract metrics
            ragas_results = {
                "faithfulness": float(ragas_df['faithfulness'].mean()) if 'faithfulness' in ragas_df.columns else None,
                "answer_relevance": float(ragas_df['answer_relevancy'].mean()) if 'answer_relevancy' in ragas_df.columns else None,
                "context_precision": float(ragas_df['context_precision'].mean()) if 'context_precision' in ragas_df.columns else None,
                "context_recall": float(ragas_df['context_recall'].mean()) if 'context_recall' in ragas_df.columns else None,
                "items_evaluated": len(eval_data)
            }
            
            # Add optional metrics if available
            if HAS_ENTITY_RECALL and 'context_entity_recall' in ragas_df.columns:
                ragas_results["context_entity_recall"] = float(ragas_df['context_entity_recall'].mean())
            if HAS_MULTIMODAL:
                if 'multimodal_faithfulness' in ragas_df.columns:
                    ragas_results["multimodal_faithfulness"] = float(ragas_df['multimodal_faithfulness'].mean())
                if 'multimodal_relevance' in ragas_df.columns:
                    ragas_results["multimodal_relevance"] = float(ragas_df['multimodal_relevance'].mean())
            
            results["ragas_metrics"] = ragas_results
            
            print(f"\n  RAGAS Results:")
            print(f"     Faithfulness: {ragas_results.get('faithfulness', 'N/A'):.3f}" if ragas_results.get('faithfulness') else "     Faithfulness: N/A")
            print(f"     Answer Relevance: {ragas_results.get('answer_relevance', 'N/A'):.3f}" if ragas_results.get('answer_relevance') else "     Answer Relevance: N/A")
            print(f"     Context Precision: {ragas_results.get('context_precision', 'N/A'):.3f}" if ragas_results.get('context_precision') else "     Context Precision: N/A")
            print(f"     Context Recall: {ragas_results.get('context_recall', 'N/A'):.3f}" if ragas_results.get('context_recall') else "     Context Recall: N/A")
            
        except Exception as e:
            print(f"\n  [WARN] RAGAS evaluation failed: {e}")
            results["ragas_metrics"] = {"error": str(e)}
    else:
        print("\n  [WARN] RAGAS not available. Install with: pip install ragas datasets")
        results["ragas_metrics"] = {"error": "RAGAS not installed"}
    
    # 2. Domain Coverage (if corpus provided)
    if corpus_path and os.path.exists(corpus_path):
        print("\n" + "=" * 60)
        print("DOMAIN COVERAGE EVALUATION")
        print("=" * 60)
        
        with open(corpus_path, 'r') as f:
            corpus_chunks = json.load(f)
        
        evaluator = MultimodalFrameworkEvaluator()
        coverage = evaluator.evaluate_domain_coverage(qa_data, corpus_chunks)
        results["domain_coverage"] = coverage
        
        print(f"\n  Chunk Coverage: {coverage['chunk_coverage']*100:.1f}% ({coverage['chunks_covered']}/{coverage['chunks_total']})")
        print(f"  File Coverage: {coverage['file_coverage']*100:.1f}%")
        print(f"  Topic Divergence (JS): {coverage['topic_divergence_js']:.4f}")
        print(f"  Uncovered Chunks: {coverage['uncovered_chunks']}")
        
        print("\n  Coverage by Chunk Type:")
        for ctype, stats in coverage['chunk_type_coverage'].items():
            print(f"    {ctype}: {stats['coverage_rate']*100:.1f}% ({stats['covered']}/{stats['total']})")
        
        print("\n  Coverage by File:")
        for fname, stats in coverage['coverage_by_file'].items():
            print(f"    {fname}: {stats['coverage_rate']*100:.1f}% ({stats['covered_chunks']}/{stats['total_chunks']})")
    
    # 3. Context Necessity (sample if too large) - BATCH PROCESSING
    if run_context_necessity:
        print("\n" + "=" * 60)
        print("CONTEXT NECESSITY EVALUATION (Anti-Parametric Bias)")
        print("=" * 60)
        
        evaluator = MultimodalFrameworkEvaluator()
        
        # Sample if needed
        eval_data = qa_data
        if sample_size and len(qa_data) > sample_size:
            eval_data = random.sample(qa_data, sample_size)
            print(f"\n  Sampling {sample_size} items for evaluation...")
        
        # Prepare batch items
        batch_items = []
        for qa in eval_data:
            context = qa.get('final_context', qa.get('original_chunk', ''))
            batch_items.append({
                'question': qa['question'],
                'answer': qa['answer'],
                'context': context
            })
        
        # Use batch evaluation if available
        if BATCH_AVAILABLE and len(batch_items) > 1:
            print(f"\n  Using batch processing for {len(batch_items)} items...")
            batch_results = evaluator.batch_evaluate_context_necessity(batch_items)
            
            necessity_scores = [r['context_necessity_score'] for r in batch_results]
            without_context_correct = sum(1 for r in batch_results if r.get('without_context_correct'))
        else:
            # Fallback to sequential processing
            necessity_scores = []
            without_context_correct = 0
            
            for i, item in enumerate(batch_items):
                if i % 10 == 0:
                    print(f"  Processing {i+1}/{len(batch_items)}...")
                
                result = evaluator.evaluate_context_necessity(
                    item['question'], item['answer'], item['context']
                )
                necessity_scores.append(result['context_necessity_score'])
                if result.get('without_context_correct'):
                    without_context_correct += 1
        
        avg_necessity = np.mean(necessity_scores) if necessity_scores else 0.0
        results["context_necessity"] = {
            "avg_context_necessity_score": float(avg_necessity),
            "items_evaluated": len(eval_data),
            "items_answerable_without_context": without_context_correct,
            "parametric_leakage_rate": without_context_correct / len(eval_data) if eval_data else 0.0,
            "score_distribution": {
                "high_necessity (0.8-1.0)": sum(1 for s in necessity_scores if s >= 0.8),
                "moderate_necessity (0.5-0.8)": sum(1 for s in necessity_scores if 0.5 <= s < 0.8),
                "low_necessity (0.0-0.5)": sum(1 for s in necessity_scores if s < 0.5)
            }
        }
        
        print(f"\n  Average Context Necessity Score: {avg_necessity:.3f}")
        print(f"  Items answerable without context: {without_context_correct}/{len(eval_data)} ({without_context_correct/len(eval_data)*100:.1f}%)")
        print(f"  Score Distribution:")
        for k, v in results["context_necessity"]["score_distribution"].items():
            print(f"    {k}: {v} ({v/len(eval_data)*100:.1f}%)")
    
    # 4. Multihop-specific metrics - BATCH PROCESSING
    if subsets['multihop']:
        print("\n" + "=" * 60)
        print("MULTIHOP METRICS (on multihop subset)")
        print("=" * 60)
        
        evaluator = MultimodalFrameworkEvaluator()
        
        # Prepare batch items
        batch_items = []
        for qa in subsets['multihop']:
            contexts = [qa.get('final_context', qa.get('original_chunk', ''))]
            batch_items.append({
                'question': qa['question'],
                'answer': qa['answer'],
                'contexts': contexts
            })
        
        # Use batch evaluation if available
        if BATCH_AVAILABLE and len(batch_items) > 1:
            print(f"\n  Using batch processing for {len(batch_items)} multihop items...")
            batch_results = evaluator.batch_evaluate_multihop_reasoning(batch_items)
            
            hop_counts = [int(r.get('hop_count', 1)) for r in batch_results]
            reasoning_scores = [float(r.get('reasoning_score', 0)) for r in batch_results]
            bridge_entities = [r['bridge_entity'] for r in batch_results 
                            if r.get('bridge_entity') and r['bridge_entity'] != 'None']
        else:
            # Fallback to sequential
            hop_counts = []
            reasoning_scores = []
            bridge_entities = []
            
            for i, item in enumerate(batch_items):
                if i % 5 == 0:
                    print(f"  Processing {i+1}/{len(batch_items)}...")
                
                result = evaluator.evaluate_multihop_reasoning(
                    item['question'], item['answer'], item['contexts']
                )
                hop_counts.append(int(result.get('hop_count', 1)))
                reasoning_scores.append(float(result.get('reasoning_score', 0)))
                if result.get('bridge_entity') and result['bridge_entity'] != 'None':
                    bridge_entities.append(result['bridge_entity'])
        
        results["multihop_metrics"] = {
            "items_evaluated": len(subsets['multihop']),
            "avg_hop_count": float(np.mean(hop_counts)) if hop_counts else 0.0,
            "avg_reasoning_score": float(np.mean(reasoning_scores)) if reasoning_scores else 0.0,
            "hop_distribution": dict(Counter(hop_counts)),
            "items_with_bridge_entity": len(bridge_entities),
            "sample_bridge_entities": bridge_entities[:10]
        }
        
        print(f"\n  Items evaluated: {len(subsets['multihop'])}")
        print(f"  Average Hop Count: {results['multihop_metrics']['avg_hop_count']:.2f}")
        print(f"  Average Reasoning Score: {results['multihop_metrics']['avg_reasoning_score']:.3f}")
        print(f"  Hop Distribution: {results['multihop_metrics']['hop_distribution']}")
    
    # 5. Multimodal-specific metrics
    if subsets['multimodal']:
        print("\n" + "=" * 60)
        print("MULTIMODAL METRICS (on multimodal subset)")
        print("=" * 60)
        
        evaluator = MultimodalFrameworkEvaluator()
        visual_dependency_scores = []
        
        for i, qa in enumerate(subsets['multimodal']):
            if i % 10 == 0:
                print(f"  Processing {i+1}/{len(subsets['multimodal'])}...")
            
            # Visual dependency test (text-only blind test)
            contexts = [qa.get('final_context', qa.get('original_chunk', ''))]
            score = evaluator.evaluate_visual_dependency(qa['question'], contexts)
            visual_dependency_scores.append(score)
        
        results["multimodal_metrics"] = {
            "items_evaluated": len(subsets['multimodal']),
            "avg_visual_dependency": float(np.mean(visual_dependency_scores)) if visual_dependency_scores else 0.0,
            "items_requiring_visual": sum(1 for s in visual_dependency_scores if s > 0.5),
            "visual_necessity_rate": sum(1 for s in visual_dependency_scores if s > 0.5) / len(visual_dependency_scores) if visual_dependency_scores else 0.0
        }
        
        print(f"\n  Items evaluated: {len(subsets['multimodal'])}")
        print(f"  Average Visual Dependency: {results['multimodal_metrics']['avg_visual_dependency']:.3f}")
        print(f"  Items requiring visual info: {results['multimodal_metrics']['items_requiring_visual']}/{len(subsets['multimodal'])}")
    
    # =========================================================================
    # FINAL SUMMARY - Key Metrics from MiRAGE Paper (Table 2)
    # =========================================================================
    print("\n" + "=" * 70)
    print("MiRAGE EVALUATION SUMMARY (Paper Table 2 Metrics)")
    print("=" * 70)
    
    # Extract metrics
    faith = results.get("ragas_metrics", {}).get("faithfulness")
    rel = results.get("ragas_metrics", {}).get("answer_relevance")
    ctx_prec = results.get("ragas_metrics", {}).get("context_precision")
    ctx_rec = results.get("ragas_metrics", {}).get("context_recall")
    
    # Hop count from context_stats (avg_chunks_per_context - 1)
    avg_hops = results.get("context_stats", {}).get("avg_chunks_per_context", 1) - 1
    if avg_hops < 0:
        avg_hops = 0
    
    # Reasoning score from multihop_metrics
    s_reason = results.get("multihop_metrics", {}).get("avg_reasoning_score")
    
    # Visual grounding from multimodal_metrics
    vis_gr = results.get("multimodal_metrics", {}).get("avg_visual_dependency")
    
    # JSD from domain_coverage
    jsd = results.get("domain_coverage", {}).get("topic_divergence_js")
    
    # Context necessity (anti-parametric bias)
    ctx_nec = results.get("context_necessity", {}).get("avg_context_necessity_score")
    
    # Helper to format metric values
    def fmt(val, decimals=3):
        if val is None:
            return "N/A".rjust(8)
        return f"{val:.{decimals}f}".rjust(8)
    
    print("\n  ┌─────────────────────────────────────────────────────────────┐")
    print("  │                    CORE METRICS                             │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    print(f"  │  Faithfulness (Faith.)          │  {fmt(faith)}              │")
    print(f"  │  Answer Relevance (Rel.)        │  {fmt(rel)}              │")
    print(f"  │  Context Precision              │  {fmt(ctx_prec)}              │")
    print(f"  │  Context Recall                 │  {fmt(ctx_rec)}              │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    print("  │                 REASONING COMPLEXITY                        │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    print(f"  │  Avg Hops (H)                   │  {fmt(avg_hops, 2)}              │")
    print(f"  │  Reasoning Score (S_reason)     │  {fmt(s_reason)}              │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    print("  │                 MULTIMODAL & DOMAIN                         │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    print(f"  │  Visual Grounding (Vis. Gr.)    │  {fmt(vis_gr)}              │")
    print(f"  │  Jensen-Shannon Div. (JSD) v    │  {fmt(jsd, 4)}              │")
    print(f"  │  Context Necessity              │  {fmt(ctx_nec)}              │")
    print("  └─────────────────────────────────────────────────────────────┘")
    
    # Dataset summary
    total_qa = results.get("qa_category_stats", {}).get("total_qa_pairs", 0)
    mm_qa = results.get("qa_category_stats", {}).get("multimodal_qa_inclusive", 0)
    table_qa = results.get("qa_category_stats", {}).get("table_qa_inclusive", 0)
    
    print(f"\n  Dataset: {total_qa} QA pairs | {mm_qa} multimodal | {table_qa} with tables")
    print("=" * 70)
    
    # Save results
    if output_dir:
        report_path = os.path.join(output_dir, "subset_evaluation_report.json")
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Report saved to: {report_path}")
    
    return results


def main(json_path: str, output_dir: str = None):
    """
    Main function to evaluate QA dataset quality.
    
    Args:
        json_path: Path to the qa_multihop_pass.json file
        output_dir: Directory to save output reports (defaults to same dir as input)
    """
    import os
    
    if output_dir is None:
        output_dir = os.path.dirname(json_path)
    
    print("=" * 60)
    print("QA DATASET QUALITY EVALUATION")
    print("=" * 60)
    
    # 1. Load raw data
    print(f"\n[1/5] Loading dataset from: {json_path}")
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
    print(f"      Loaded {len(raw_data)} QA pairs")
    
    # 2. Analyze missing information BEFORE transformation
    print("\n[2/5] Analyzing data completeness...")
    missing_info = analyze_missing_information(raw_data)
    
    print("\n" + "-" * 60)
    print("METRICS EVALUATION STATUS:")
    print("-" * 60)
    for metric, status in missing_info["metrics_status"].items():
        symbol = "[OK]" if status["can_evaluate"] else "[X]"
        quality = status["quality"]
        print(f"  {symbol} {metric}: {quality}")
        if status.get("missing"):
            for m in status["missing"]:
                print(f"      Missing: {m}")
    
    print("\n" + "-" * 60)
    print("RECOMMENDATIONS:")
    print("-" * 60)
    for rec in missing_info["recommendations"]:
        print(f"  - {rec}")
    
    # 3. Transform data
    print("\n[3/5] Transforming data to evaluation format...")
    transformed_data = transform_qa_data(raw_data)
    
    # 4. Run evaluation (only metrics that can be evaluated)
    print("\n[4/5] Running evaluation...")
    
    try:
        evaluator = MultimodalFrameworkEvaluator()
        
        # Save transformed data for the evaluator
        transformed_path = os.path.join(output_dir, "qa_transformed_for_eval.json")
        with open(transformed_path, 'w') as f:
            json.dump(transformed_data, f, indent=2)
        
        output_path = os.path.join(output_dir, "eval_report.json")
        final_df, report = evaluator.run_full_evaluation(transformed_path, output_path)
        
        # Add missing info analysis to report
        report["data_completeness"] = missing_info
        
        # Save updated report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)
            
        print(f"\n[5/5] Reports saved to:")
        print(f"      - {output_path}")
        print(f"      - {output_path.replace('.json', '_detailed.csv')}")
        
        return final_df, report
        
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        print("\nRunning basic statistics only...")
        
        # Compute basic statistics without LLM calls
        basic_stats = {
            "total_samples": len(raw_data),
            "context_status_distribution": Counter(
                item.get("context_status", "UNKNOWN") for item in raw_data
            ),
            "avg_relevance_score": np.mean([
                float(item.get("relevance_score", 0)) for item in raw_data
                if item.get("relevance_score")
            ]),
            "avg_difficulty_score": np.mean([
                float(item.get("difficulty_score", 0)) for item in raw_data
                if item.get("difficulty_score")
            ]),
            "domain_distribution": Counter(
                item.get("domain", "UNKNOWN") for item in raw_data
            ),
            "data_completeness": missing_info
        }
        
        output_path = os.path.join(output_dir, "eval_report_basic.json")
        with open(output_path, 'w') as f:
            # Convert Counter objects to dict for JSON serialization
            basic_stats["context_status_distribution"] = dict(basic_stats["context_status_distribution"])
            basic_stats["domain_distribution"] = dict(basic_stats["domain_distribution"])
            json.dump(basic_stats, f, indent=4)
        
        print(f"\n[5/5] Basic report saved to: {output_path}")
        print("\nBasic Statistics:")
        print(json.dumps({k: v for k, v in basic_stats.items() if k != "data_completeness"}, indent=2))
        
        return None, basic_stats


if __name__ == "__main__":
    import argparse
    
    # Load Gemini API key
    API_KEY_PATH = os.environ.get("GEMINI_API_KEY_PATH", os.path.expanduser("~/.config/gemini/api_key.txt"))
    with open(API_KEY_PATH, 'r') as f:
        os.environ["GOOGLE_API_KEY"] = f.read().strip()
    
    # Default paths (override via command line arguments)
    DEFAULT_QA_PATH = "output/results/qa_deduplicated.json"
    DEFAULT_CORPUS_PATH = "output/results/chunks.json"
    
    parser = argparse.ArgumentParser(description="Evaluate QA dataset quality")
    parser.add_argument("--qa-file", "-q", default=DEFAULT_QA_PATH, help="Path to QA JSON file")
    parser.add_argument("--corpus-file", "-c", default=DEFAULT_CORPUS_PATH, help="Path to corpus chunks.json")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory for reports")
    parser.add_argument("--sample-size", "-s", type=int, default=50, help="Sample size for expensive metrics")
    parser.add_argument("--skip-context-necessity", action="store_true", help="Skip context necessity evaluation")
    
    args = parser.parse_args()
    
    # Set output dir
    output_dir = args.output_dir or os.path.dirname(args.qa_file)
    
    # Load QA data
    print(f"Loading QA data from: {args.qa_file}")
    with open(args.qa_file, 'r') as f:
        qa_data = json.load(f)
    print(f"Loaded {len(qa_data)} QA pairs")
    
    # Run evaluation
    results = run_subset_evaluation(
        qa_data=qa_data,
        corpus_path=args.corpus_file,
        output_dir=output_dir,
        sample_size=args.sample_size,
        run_context_necessity=not args.skip_context_necessity
    )
    
    print("\n" + "=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2, default=str))