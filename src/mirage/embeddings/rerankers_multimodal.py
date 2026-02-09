import torch
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from PIL import Image

from mirage.utils.device import get_device, is_gpu_available

class BaseReranker(ABC):
    """Abstract base class for image-based rerankers"""
    
    @abstractmethod
    def rerank(self, query: str, image_paths: List[str], top_k: int = 10) -> List[int]:
        pass

class ChunkReranker(ABC):
    """Abstract base class for chunk-based rerankers (text + optional images)"""
    
    @abstractmethod
    def rerank(self, query: str, chunks: List[Dict[str, str]], top_k: int = 1) -> List[Tuple[int, float, Dict[str, str]]]:
        """
        Rerank chunks based on query relevance
        
        Args:
            query: User query string
            chunks: List of dicts with 'text' and optional 'image_path' keys
            top_k: Number of top chunks to return
        
        Returns:
            List of tuples: (original_index, relevance_score, chunk_dict)
        """
        pass

class MMR5Reranker(BaseReranker):
    """MM-R5: MultiModal Reasoning-Enhanced ReRanker"""
    
    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>"
    )
    
    QUESTION_TEMPLATE = (
        "Please rank the following images according to their relevance to the question. "
        "Provide your response in the format: <think>your reasoning process here</think><answer>[image_id_1, image_id_2, ...]</answer> "
        "where the numbers in the list represent the ranking order of images'id from most to least relevant. "
        "Before outputting the answer, you need to analyze each image and provide your analysis process."
        "For example: <think>Image 1 shows the most relevant content because...</think><answer>[id_most_relevant, id_second_relevant, ...]</answer>"
        "\nThe question is: {Question}"
        "\n\nThere are {image_num} images, id from 1 to {image_num_end}, Image ID to image mapping:\n"
    )
    
    def __init__(self, model_name: str = "i2vec/MM-R5"):
        print(f"Loading MM-R5: {model_name}")
        
        # Try official MM-R5 package first
        try:
            from reranker import QueryReranker  # type: ignore
            self.reranker = QueryReranker(model_name)
            self.use_official_package = True
            print(f"[OK] MM-R5 loaded via official package")
        except ImportError:
            # Fallback to direct implementation using official code
            from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info
            
            attn_kwargs = {}
            if self._flash_attn_available():
                attn_kwargs["attn_implementation"] = "flash_attention_2"
            else:
                print("[WARN] flash_attn not available. Falling back to default attention implementation.")
            
            # Load without device_map="auto" to avoid meta tensor issues in parallel processing
            self.device = get_device()
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                **attn_kwargs,
            ).to(self.device).eval()
            
            self.processor = Qwen2_5_VLProcessor.from_pretrained(model_name)
            self.use_official_package = False
            print(f"[OK] MM-R5 loaded via direct implementation")
    
    def rerank(self, query: str, image_paths: List[str], top_k: int = 10) -> List[int]:
        """Rerank images based on query relevance"""
        import re
        
        if self.use_official_package:
            # Use official package API
            predicted_order = self.reranker.rerank(query, image_paths)
        else:
            # Use direct implementation following official code
            from qwen_vl_utils import process_vision_info
            
            device = self.model.device
            
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self.SYSTEM_PROMPT,
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.QUESTION_TEMPLATE.format(
                                Question=query,
                                image_num=len(image_paths),
                                image_num_end=len(image_paths)
                            ),
                        },
                    ],
                },
            ]
            
            # Add images to messages
            for i, image_path in enumerate(image_paths):
                messages[-1]["content"].extend(
                    [
                        {"type": "text", "text": f"\nImage {i+1}: "},
                        {"type": "image", "image": image_path},
                    ]
                )
                
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            inputs = inputs.to(device)
            
            generated_ids = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=0.3,
                max_new_tokens=8192,
                use_cache=True,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Parse output results
            match = re.search(r'<answer>\[(.*?)\]</answer>', output_text)
            
            if match:
                try:
                    tmp_predicted_order = []
                    predicted_order = [int(x) - 1 for x in match.group(1).strip().split(',') if x.strip()]
                    
                    for idx in predicted_order:
                        if 0 <= idx < len(image_paths):
                            tmp_predicted_order.append(idx)
                            
                    predicted_order = tmp_predicted_order
                    
                    # Handle missing indices
                    if len(set(predicted_order)) < len(image_paths):
                        missing_ids = set(range(len(image_paths))) - set(predicted_order)
                        predicted_order.extend(sorted(list(missing_ids)))
                        
                except Exception as e:
                    predicted_order = [i for i in range(len(image_paths))]
                    print(f"[WARN] Parsing error: {str(e)}, output text: {output_text[:200]}...")
            else:
                predicted_order = [i for i in range(len(image_paths))]
                print(f"[WARN] Could not parse ranking from output: {output_text[:200]}...")
        
        return predicted_order[:top_k]
    
    @staticmethod
    def _flash_attn_available() -> bool:
        try:
            import flash_attn  # noqa: F401
            return True
        except Exception:
            return False

class Florence2Reranker(BaseReranker):
    """Florence-2-large for visual document reranking"""
    
    def __init__(self, model_name: str = "microsoft/Florence-2-large"):
        from transformers import AutoProcessor, AutoModelForCausalLM
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            BitsAndBytesConfig = None
        
        print(f"Loading Florence-2: {model_name}")
        
        self.device = get_device()
        self.torch_dtype = torch.float16 if is_gpu_available() else torch.float32
        
        # Note: When using quantization, device_map is needed for BitsAndBytes
        # When not using quantization, load explicitly to avoid meta tensor issues
        if self.device == "cuda":
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                    device_map="auto"  # Required for BitsAndBytes quantization
                ).eval()
            except Exception as e:
                print(f"[WARN] Quantization failed ({e}), loading without quantization...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                ).to(self.device).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).to(self.device).eval()
        
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print(f"[OK] Florence-2 loaded on {self.device}")
    
    def _score_image(self, query: str, image_path: str) -> float:
        """Score a single image based on query relevance"""
        try:
            if not Path(image_path).exists():
                return 0.0
            
            image = Image.open(image_path).convert('RGB')
            
            # Use caption task to understand image
            prompt = "<CAPTION>"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                    use_cache=False,
                )
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(
                generated_text, task="<CAPTION>", image_size=(image.width, image.height)
            )
            
            caption = parsed_answer.get('<CAPTION>', '')
            
            # Simple relevance score based on query keyword overlap
            query_words = set(query.lower().split())
            caption_words = set(caption.lower().split())
            overlap = len(query_words & caption_words)
            score = overlap / max(len(query_words), 1)
            
            return score
            
        except Exception as e:
            print(f"[WARN] Scoring failed for {image_path}: {e}")
            return 0.0
    
    def rerank(self, query: str, image_paths: List[str], top_k: int = 10) -> List[int]:
        """Rerank images based on query relevance using Florence-2"""
        try:
            scores = []
            for img_path in image_paths:
                score = self._score_image(query, img_path)
                scores.append(score)
            
            # Get top-k indices sorted by score
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            return ranked_indices[:top_k]
            
        except Exception as e:
            print(f"[WARN] Reranking failed: {e}")
            return list(range(min(top_k, len(image_paths))))

class VLMReranker(ChunkReranker):
    """VLM-based reranker using Motor Maven endpoint with multiple images"""
    
    def __init__(self):
        from mirage.core.llm import call_vlm_with_multiple_images
        from mirage.core.prompts import PROMPTS
        
        self.call_vlm_multi = call_vlm_with_multiple_images
        self.rerank_prompt = PROMPTS["rerank_vlm"]
        print("[OK] VLM Reranker initialized")
    
    def rerank(self, query: str, chunks: List[Dict[str, str]], top_k: int = 1) -> List[Tuple[int, float, Dict[str, str]]]:
        """
        Rerank chunks based on query relevance with multiple images
        
        Args:
            query: User query string
            chunks: List of dicts with 'text' and optional 'image_path' keys
            top_k: Number of top chunks to return (default: 1)
        
        Returns:
            List of tuples: (original_index, relevance_score, chunk_dict)
        """
        try:
            # Collect chunk data and images
            chunk_data = []
            image_paths = []
            chunk_to_image_map = {}  # Maps chunk index to image index
            
            for i, chunk in enumerate(chunks):
                chunk_info = {
                    'index': i,
                    'text': chunk.get('text', ''),
                    'image_path': chunk.get('image_path', None),
                    'has_image': False
                }
                
                # Track images
                if chunk_info['image_path'] and Path(chunk_info['image_path']).exists():
                    chunk_info['has_image'] = True
                    chunk_to_image_map[i] = len(image_paths)
                    image_paths.append(chunk_info['image_path'])
                
                chunk_data.append(chunk_info)
            
            # Build structured prompt with explicit chunk boundaries
            formatted_chunks = []
            for chunk_info in chunk_data:
                i = chunk_info['index'] + 1  # 1-indexed for display
                chunk_lines = [f"<CHUNK_START id={i}>"]
                chunk_lines.append(chunk_info['text'])
                
                if chunk_info['has_image']:
                    img_idx = chunk_to_image_map[chunk_info['index']] + 1
                    chunk_lines.append(f"<IMAGE_START id={img_idx} relates_to_chunk={i}>")
                    chunk_lines.append(f"[Image {img_idx} displayed here]")
                    chunk_lines.append(f"<IMAGE_END id={img_idx}>")
                
                chunk_lines.append(f"<CHUNK_END id={i}>")
                formatted_chunks.append("\n".join(chunk_lines))
            
            # Build full prompt
            full_prompt = f"""{self.rerank_prompt}

Query: {query}

Chunks to rank:

{chr(10).join(formatted_chunks)}"""
            
            # Call VLM with all images
            if not image_paths:
                # No images - use LLM fallback instead of VLM
                from mirage.core.llm import call_llm_simple
                response = call_llm_simple(full_prompt)
            else:
                response = self.call_vlm_multi(full_prompt, image_paths)
            
            # Parse response to extract rankings
            rankings = self._parse_rankings(response, chunk_data)
            
            # Return top-k with chunk data
            result = []
            for idx, score in rankings[:top_k]:
                result.append((idx, score, chunks[idx]))
            
            return result
            
        except Exception as e:
            print(f"[WARN] Reranking failed: {e}")
            import traceback
            traceback.print_exc()
            # Return original order with default scores
            return [(i, 1.0, chunks[i]) for i in range(min(top_k, len(chunks)))]
    
    def _parse_rankings(self, response: str, chunk_data: List[Dict]) -> List[Tuple[int, float]]:
        """Parse VLM response to extract chunk rankings from structured format"""
        rankings = []
        num_chunks = len(chunk_data)
        
        # Primary pattern: <Rank X>Chunk Y (simplified format)
        rank_pattern = r'<Rank\s+(\d+)>\s*Chunk\s+(\d+)'
        matches = re.findall(rank_pattern, response, re.IGNORECASE)
        
        seen_indices = set()
        for rank_num, chunk_num in matches:
            idx = int(chunk_num) - 1  # Convert to 0-indexed
            rank = int(rank_num)
            
            # Calculate score based on rank (higher rank = lower score)
            # Rank 1 gets highest score (1.0), decreasing linearly
            relevance = 1.0 - ((rank - 1) / max(num_chunks, 1))
            
            # Ensure valid index and no duplicates
            if 0 <= idx < num_chunks and idx not in seen_indices:
                rankings.append((idx, relevance))
                seen_indices.add(idx)
        
        # If parsing failed or incomplete, fill remaining chunks with low scores
        if len(rankings) < num_chunks:
            missing = set(range(num_chunks)) - seen_indices
            for idx in missing:
                rankings.append((idx, 0.0))
            print(f"[WARN] Parsed {len(seen_indices)}/{num_chunks} chunks from response")
            if len(seen_indices) == 0:
                # Debug: print response when parsing completely fails
                print(f"Debug - VLM Response (first 500 chars):\n{response[:500]}")
                print(f"Debug - VLM Response (last 500 chars):\n{response[-500:]}")
        
        # Sort by relevance score (highest first)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings

class MonoVLMReranker(ChunkReranker):
    """MonoVLM reranker using lightonai/MonoQwen2-VL-v0.1"""
    
    def __init__(
        self,
        model_name: str = "lightonai/MonoQwen2-VL-v0.1",
        processor_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    ):
        print(f"Loading MonoVLM: {model_name}")
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        
        self.device = get_device()
        self.torch_dtype = torch.bfloat16 if is_gpu_available() else torch.float32
        
        self.processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=True)
        
        # Load model without device_map="auto" to avoid meta tensor issues
        # First load to CPU, then move to GPU explicitly
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # Reduces CPU memory during loading
        )
        
        # Move to device and set to eval mode
        self.model = self.model.to(self.device).eval()
        
        # Cache token IDs for True/False classification
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Processor does not expose a tokenizer needed for MonoVLM scoring.")
        
        self.true_token_id = tokenizer.convert_tokens_to_ids("True")
        self.false_token_id = tokenizer.convert_tokens_to_ids("False")
        
        if self.true_token_id is None or self.false_token_id is None:
            raise ValueError("Tokenizer missing True/False tokens required for MonoVLM scoring.")
        
        print(f"[OK] MonoVLM loaded")
    
    def _build_prompt(self, query: str, chunk_text: str) -> str:
        chunk_text = chunk_text.strip() if chunk_text else "[No text provided]"
        return (
            "Assert the relevance of the provided document (text and/or image) to the query.\n"
            "Respond with a single word: True if relevant, otherwise False.\n\n"
            f"Query:\n{query}\n\nDocument:\n{chunk_text}"
        )
    
    def _score_chunk(self, query: str, chunk: Dict[str, str]) -> float:
        image = None
        image_path = chunk.get('image_path')
        
        try:
            if image_path and Path(image_path).exists():
                with Image.open(image_path) as img:
                    image = img.convert("RGB")
        except Exception as img_err:
            print(f"[WARN] Failed to load image {image_path}: {img_err}")
            image = None
        
        prompt = self._build_prompt(query, chunk.get('text', ''))
        
        messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": image}] if image else []
                ) + [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ]
        
        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            processor_kwargs = {
                "text": text,
                "return_tensors": "pt",
            }
            
            if image is not None:
                processor_kwargs["images"] = image
            
            inputs = self.processor(**processor_kwargs)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits[:, -1, :]
            relevance = torch.softmax(
                logits[:, [self.true_token_id, self.false_token_id]],
                dim=-1
            )
            
            return relevance[0, 0].item()
        
        except Exception as e:
            print(f"[WARN] MonoVLM scoring failed: {e}")
            return 0.0
    
    def rerank(self, query: str, chunks: List[Dict[str, str]], top_k: int = 1) -> List[Tuple[int, float, Dict[str, str]]]:
        """
        Rerank chunks based on query relevance using MonoVLM
        
        Args:
            query: User query string
            chunks: List of dicts with 'text' and optional 'image_path' keys
            top_k: Number of top chunks to return (default: 1)
        
        Returns:
            List of tuples: (original_index, relevance_score, chunk_dict)
        """
        from tqdm import tqdm
        scores = []
        print(f"Scoring {len(chunks)} chunks for reranking...")
        for idx, chunk in enumerate(tqdm(chunks, desc="Reranking chunks")):
            score = self._score_chunk(query, chunk)
            scores.append((idx, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        result = []
        for idx, score in scores[:top_k]:
            if 0 <= idx < len(chunks):
                result.append((idx, score, chunks[idx]))
        
        return result

class TextEmbeddingReranker(ChunkReranker):
    """Text embedding reranker using BAAI/bge-large-en-v1.5 with image descriptions"""
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        
        print(f"Loading text embedding model: {model_name}")
        self.device = get_device()
        
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"[OK] Text embedding model loaded on {self.device}")
        
        # For generating image descriptions (reuse VLM)
        from mirage.core.llm import call_vlm_simple
        from mirage.core.prompts import PROMPTS
        self.call_vlm = call_vlm_simple
        self.desc_prompt = PROMPTS.get("rerank_image_desc", "Generate a concise 100-word technical description of this image.")
    
    def _generate_image_description(self, image_path: str) -> str:
        """Generate description for image using VLM"""
        try:
            return self.call_vlm(self.desc_prompt, image_path)
        except Exception as e:
            print(f"[WARN] Image description failed for {image_path}: {e}")
            return "[Image description unavailable]"
    
    def rerank(self, query: str, chunks: List[Dict[str, str]], top_k: int = 1) -> List[Tuple[int, float, Dict[str, str]]]:
        """
        Rerank chunks based on query relevance using text embeddings
        
        Args:
            query: User query string
            chunks: List of dicts with 'text' and optional 'image_path' keys
            top_k: Number of top chunks to return (default: 1)
        
        Returns:
            List of tuples: (original_index, relevance_score, chunk_dict)
        """
        try:
            # Generate image descriptions and combine with text
            chunk_texts = []
            for chunk in chunks:
                text = chunk.get('text', '')
                
                # Add image description if image exists
                if chunk.get('image_path') and Path(chunk['image_path']).exists():
                    img_desc = self._generate_image_description(chunk['image_path'])
                    combined_text = f"{text}\n[Image Description: {img_desc}]"
                else:
                    combined_text = text
                
                chunk_texts.append(combined_text)
            
            # Compute embeddings
            query_embedding = self.model.encode([query], convert_to_tensor=True, normalize_embeddings=True)
            chunk_embeddings = self.model.encode(chunk_texts, convert_to_tensor=True, normalize_embeddings=True)
            
            # Compute cosine similarities
            similarities = torch.nn.functional.cosine_similarity(
                query_embedding, chunk_embeddings, dim=1
            )
            
            # Get top-k indices sorted by similarity
            top_indices = torch.argsort(similarities, descending=True)[:top_k].cpu().tolist()
            
            # Build results with scores
            results = []
            for idx in top_indices:
                if 0 <= idx < len(chunks):
                    score = float(similarities[idx].cpu())
                    results.append((idx, score, chunks[idx]))
            
            return results
            
        except Exception as e:
            print(f"[WARN] Reranking failed: {e}")
            return [(i, 1.0, chunks[i]) for i in range(min(top_k, len(chunks)))]


class GeminiVLMReranker(ChunkReranker):
    """
    Gemini API-based VLM reranker.
    Uses Gemini for multimodal understanding to score chunk relevance.
    Fully API-based - no local GPU required.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        timeout: int = 180
    ):
        import os
        import requests
        print(f"Initializing Gemini VLM Reranker with model: {model_name}")
        
        # Load API key from mirage config
        from mirage.core.llm import API_KEY, _initialize_config
        _initialize_config()
        
        self.api_key = API_KEY
        if not self.api_key:
            # Fallback to environment
            self.api_key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
        
        if not self.api_key:
            raise ValueError("Gemini API key not found")
        
        self.model_name = model_name
        self.timeout = timeout
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        
        print(f"[OK] Gemini VLM Reranker initialized (model: {model_name})")
    
    def _encode_image(self, image_path: str) -> Tuple[str, str]:
        """Encode image to base64 with mime type"""
        import base64
        
        ext = Path(image_path).suffix.lower()
        mime_types = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                      '.gif': 'image/gif', '.webp': 'image/webp'}
        mime_type = mime_types.get(ext, 'image/png')
        
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        return image_data, mime_type
    
    def _score_chunk(self, query: str, chunk: Dict[str, str]) -> float:
        """Score chunk using Gemini API"""
        import requests
        import re
        
        url = self.base_url.format(model=self.model_name) + f"?key={self.api_key}"
        
        parts = [{
            "text": f"""Evaluate if the following document is relevant to the query. 
Rate the relevance on a scale from 0.0 to 1.0, where:
- 0.0 means completely irrelevant
- 1.0 means highly relevant

Query: {query}

Document:
{chunk.get('text', '')}

Respond with ONLY a decimal number between 0.0 and 1.0."""
        }]
        
        # Add image if present
        image_path = chunk.get('image_path')
        if image_path and Path(image_path).exists():
            try:
                image_data, mime_type = self._encode_image(image_path)
                parts.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": image_data
                    }
                })
            except Exception as e:
                print(f"[WARN] Could not encode image {image_path}: {e}")
        
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0.0}
        }
        
        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            
            # Parse score
            try:
                score = float(text)
                return max(0.0, min(1.0, score))
            except ValueError:
                # Try to extract number from response
                match = re.search(r'(\d+\.?\d*)', text)
                if match:
                    return max(0.0, min(1.0, float(match.group(1))))
                return 0.5
                
        except Exception as e:
            print(f"[WARN] Gemini reranking failed: {e}")
            return 0.0
    
    def rerank(self, query: str, chunks: List[Dict[str, str]], top_k: int = 1) -> List[Tuple[int, float, Dict[str, str]]]:
        """Rerank chunks using Gemini API"""
        scores = []
        for idx, chunk in enumerate(chunks):
            score = self._score_chunk(query, chunk)
            scores.append((idx, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        result = []
        for idx, score in scores[:top_k]:
            if 0 <= idx < len(chunks):
                result.append((idx, score, chunks[idx]))
        
        return result


if __name__ == "__main__":
    from pathlib import Path
    
    print("=" * 60)
    print("Testing Multimodal Rerankers")
    print("=" * 60)
    
    # Prepare test data (text-only chunks for testing without specific images)
    test_chunks = [
        {
            "text": "Machine learning models require careful hyperparameter tuning to achieve optimal performance. Common parameters include learning rate, batch size, and regularization strength.",
            "image_path": None
        },
        {
            "text": "This figure shows the characteristic relationship between model accuracy and training epochs. The data demonstrates typical learning curve behavior with diminishing returns after initial rapid improvement.",
            "image_path": None  # Set to actual image path for multimodal testing
        },
        {
            "text": "Neural network architectures vary widely in depth and complexity. Convolutional neural networks excel at image tasks, while transformers dominate natural language processing applications.",
            "image_path": None
        },
        {
            "text": "Data preprocessing is essential for model performance. Standard techniques include normalization, handling missing values, and feature engineering for tabular data.",
            "image_path": None
        },
        {
            "text": "This flowchart illustrates the machine learning pipeline from data collection through model deployment, showing how each stage contributes to the final system performance.",
            "image_path": None  # Set to actual image path for multimodal testing
        }
    ]
    
    # Extract valid image paths for image-based rerankers
    image_paths = [chunk['image_path'] for chunk in test_chunks if chunk.get('image_path') and Path(chunk['image_path']).exists()]
    valid_chunks = [chunk for chunk in test_chunks if chunk['image_path'] is None or (chunk.get('image_path') and Path(chunk['image_path']).exists())]
    query = "How do machine learning models improve with training, and what are the key stages in the ML pipeline?"
    
    # Test 1: VLM Reranker
    print("\n1. Testing VLM Reranker...")
    print("-" * 60)
    try:
        if valid_chunks:
            reranker = VLMReranker()
            print(f"Testing with {len(valid_chunks)} chunks")
            print(f"Chunks with images: {sum(1 for c in valid_chunks if c.get('image_path'))}")
            print(f"\nQuery: {query}\n")
            results = reranker.rerank(query, valid_chunks, top_k=3)
            print("\nTop 3 Reranked Chunks:")
            for i, (orig_idx, score, chunk) in enumerate(results, 1):
                print(f"\n{i}. Original Index: {orig_idx}, Relevance Score: {score:.3f}")
                print(f"   Text preview: {chunk['text'][:100]}...")
                if chunk.get('image_path'):
                    print(f"   Has image: {Path(chunk['image_path']).name}")
                else:
                    print(f"   Text-only chunk")
            print("\n[OK] VLM Reranker test completed!")
        else:
            print("[WARN] No valid test chunks found.")
    except Exception as e:
        print(f"[ERROR] VLM Reranker test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: MonoVLM Reranker
    print("\n2. Testing MonoVLM Reranker (lightonai/MonoQwen2-VL-v0.1)...")
    print("-" * 60)
    try:
        if valid_chunks:
            reranker = MonoVLMReranker()
            print(f"Testing with {len(valid_chunks)} chunks")
            print(f"Chunks with images: {sum(1 for c in valid_chunks if c.get('image_path'))}")
            print(f"\nQuery: {query}\n")
            results = reranker.rerank(query, valid_chunks, top_k=3)
            print("\nTop 3 Reranked Chunks:")
            for i, (orig_idx, score, chunk) in enumerate(results, 1):
                print(f"\n{i}. Original Index: {orig_idx}, Relevance Score: {score:.3f}")
                print(f"   Text preview: {chunk['text'][:100]}...")
                if chunk.get('image_path'):
                    print(f"   Has image: {Path(chunk['image_path']).name}")
                else:
                    print(f"   Text-only chunk")
            print("\n[OK] MonoVLM Reranker test completed!")
        else:
            print("[WARN] No valid test chunks found.")
    except Exception as e:
        print(f"[WARN] MonoVLM Reranker test skipped: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Text Embedding Reranker (BAAI/bge-large-en-v1.5)
    print("\n3. Testing Text Embedding Reranker (BAAI/bge-large-en-v1.5)...")
    print("-" * 60)
    try:
        if valid_chunks:
            reranker = TextEmbeddingReranker()
            print(f"Testing with {len(valid_chunks)} chunks")
            print(f"Chunks with images: {sum(1 for c in valid_chunks if c.get('image_path'))}")
            print(f"\nQuery: {query}\n")
            results = reranker.rerank(query, valid_chunks, top_k=3)
            print("\nTop 3 Reranked Chunks:")
            for i, (orig_idx, score, chunk) in enumerate(results, 1):
                print(f"\n{i}. Original Index: {orig_idx}, Relevance Score: {score:.3f}")
                print(f"   Text preview: {chunk['text'][:100]}...")
                if chunk.get('image_path'):
                    print(f"   Has image: {Path(chunk['image_path']).name}")
                else:
                    print(f"   Text-only chunk")
            print("\n[OK] Text Embedding Reranker test completed!")
        else:
            print("[WARN] No valid test chunks found.")
    except Exception as e:
        print(f"[WARN] Text Embedding Reranker test skipped: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)