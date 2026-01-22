
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List, Union
from pathlib import Path
from PIL import Image
import os
import requests
import base64

# Text Embedding Configuration
EMBEDDING_MODELS_TEXT = {
    "bge_m3": "BAAI/bge-m3" 
}

def get_best_embedding_model():
    """Returns the best text embedding model (BGE-M3)"""
    return EMBEDDING_MODELS_TEXT["bge_m3"]

def get_device_map_for_gpus(gpus: Optional[List[int]] = None) -> str:
    """Returns device_map string for specified GPUs"""
    if gpus and len(gpus) > 0:
        # Use first specified GPU as primary
        return f"cuda:{gpus[0]}"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_local_model_path(model_name: str) -> str:
    """Check for locally downloaded models before falling back to HuggingFace.
    
    Checks these locations in order:
    1. ./models/{model_name}
    2. ../models/{model_name}
    3. ~/models/{model_name}
    4. HuggingFace cache (~/.cache/huggingface/hub/models--{model_name})
    
    Returns the local path if found, otherwise the original model_name for HuggingFace download.
    """
    # Extract model short name from HuggingFace path (e.g., "Qwen/Qwen3-VL-Embedding-8B" -> "Qwen3-VL-Embedding-8B")
    if "/" in model_name:
        short_name = model_name.split("/")[-1]
    else:
        short_name = model_name
    
    # Possible local directories to check
    local_dirs = [
        os.path.join(".", "models", short_name),
        os.path.join("..", "models", short_name),
        os.path.expanduser(os.path.join("~", "models", short_name)),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", short_name),
    ]
    
    for local_path in local_dirs:
        abs_path = os.path.abspath(local_path)
        # Check if directory exists and has config.json (indicates valid model)
        if os.path.isdir(abs_path) and os.path.exists(os.path.join(abs_path, "config.json")):
            print(f"📂 Found local model at: {abs_path}")
            return abs_path
    
    # Check HuggingFace cache
    hf_cache_name = f"models--{model_name.replace('/', '--')}"
    hf_cache_path = os.path.expanduser(os.path.join("~", ".cache", "huggingface", "hub", hf_cache_name))
    if os.path.isdir(hf_cache_path):
        # Find the latest snapshot
        snapshots_dir = os.path.join(hf_cache_path, "snapshots")
        if os.path.isdir(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                latest_snapshot = os.path.join(snapshots_dir, sorted(snapshots)[-1])
                if os.path.exists(os.path.join(latest_snapshot, "config.json")):
                    print(f"📂 Found cached model at: {latest_snapshot}")
                    return latest_snapshot
    
    # No local model found, will download from HuggingFace
    print(f"⬇️  Model not found locally, will download from HuggingFace: {model_name}")
    return model_name


# Multimodal Embedding Classes
class BaseMultimodalEmbedder(ABC):
    """Abstract base class for multimodal embedders"""
    
    @abstractmethod
    def embed_text(self, text: str) -> torch.Tensor:
        """Embed a single text string. Internal method."""
        pass
    
    @abstractmethod
    def embed_image(self, image_path: str) -> torch.Tensor:
        pass
    
    @abstractmethod
    def embed_multimodal(self, text: str, image_path: Optional[str] = None) -> torch.Tensor:
        pass
    
    def encode(
        self,
        sentences: Union[str, List[str]],
        convert_to_tensor: bool = False,
        convert_to_numpy: bool = False,
        show_progress_bar: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, np.ndarray, List]:
        """
        Encode sentences to embeddings. Matches SentenceTransformer API.
        
        Args:
            sentences: Single string or list of strings to encode
            convert_to_tensor: If True, return torch.Tensor
            convert_to_numpy: If True, return numpy array
            show_progress_bar: Ignored (for API compatibility)
            **kwargs: Additional arguments (ignored for compatibility)
        
        Returns:
            Embeddings as tensor, numpy array, or list depending on flags
        """
        # Handle single string
        if isinstance(sentences, str):
            embedding = self.embed_text(sentences)
            if convert_to_numpy:
                return embedding.cpu().float().numpy() if isinstance(embedding, torch.Tensor) else np.array(embedding)
            if convert_to_tensor:
                return embedding if isinstance(embedding, torch.Tensor) else torch.tensor(embedding)
            return embedding.cpu().float().numpy() if isinstance(embedding, torch.Tensor) else embedding
        
        # Handle list of strings
        embeddings = []
        for text in sentences:
            emb = self.embed_text(text)
            embeddings.append(emb)
        
        # Stack embeddings
        if embeddings:
            stacked = torch.stack(embeddings) if isinstance(embeddings[0], torch.Tensor) else torch.tensor(embeddings)
            if convert_to_numpy:
                return stacked.cpu().float().numpy()
            if convert_to_tensor:
                return stacked
            return stacked.cpu().float().numpy()
        
        return np.array([]) if convert_to_numpy else torch.tensor([])

class NomicVLEmbed(BaseMultimodalEmbedder):
    """
    Nomic Embed Multimodal 7B
    """
    
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-multimodal-7b", gpus: Optional[List[int]] = None):
        from transformers import BitsAndBytesConfig
        from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
        
        # Check for local model first
        resolved_path = _resolve_local_model_path(model_name)
        
        print(f"Loading Nomic: {resolved_path}")
        self._setup_hf_auth()
        
        # Use specified GPUs or default to cuda
        self.device = get_device_map_for_gpus(gpus)
        self.gpus = gpus
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Read attention implementation from config (default to sdpa for stability)
        attn_impl = "sdpa"  # Default to PyTorch native attention
        try:
            from config_loader import get_embedding_config
            embed_config = get_embedding_config()
            nomic_config = embed_config.get('models', {}).get('nomic', {})
            attn_impl = nomic_config.get('attn_implementation', 'sdpa')
        except Exception:
            pass
        
        is_cuda = self.device.startswith("cuda")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ) if is_cuda else None
        
        self.model = BiQwen2_5.from_pretrained(
            resolved_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation=attn_impl,
            quantization_config=quantization_config,
        ).eval()
        
        self.processor = BiQwen2_5_Processor.from_pretrained(resolved_path)
        print(f"✅ Nomic loaded on {self.device}")
    
    def _setup_hf_auth(self):
        api_key_path = os.environ.get("HF_TOKEN_PATH", os.path.expanduser("~/.config/huggingface/token"))
        if os.path.exists(api_key_path):
            with open(api_key_path, 'r') as f:
                os.environ["HUGGING_FACE_HUB_TOKEN"] = f.read().strip()
    
    def embed_text(self, text: str) -> torch.Tensor:
        inputs = self.processor.process_queries([text]).to(self.device)
        with torch.no_grad():
            embeddings = self.model(**inputs)
        return embeddings.flatten()
    
    def embed_image(self, image_path: str) -> torch.Tensor:
        if not Path(image_path).exists():
            return torch.zeros(128, device=self.device)
        
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor.process_images([image]).to(self.device)
        with torch.no_grad():
            embeddings = self.model(**inputs)
        return embeddings.flatten()
    
    def embed_multimodal(self, text: str, image_path: Optional[str] = None) -> torch.Tensor:
        if image_path and Path(image_path).exists():
            image = Image.open(image_path).convert('RGB')
            batch_images = self.processor.process_images([image]).to(self.device)
            batch_queries = self.processor.process_queries([text]).to(self.device)
            
            with torch.no_grad():
                query_emb = self.model(**batch_queries)
                image_emb = self.model(**batch_images)
                # Normalize and combine text and image embeddings
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
                image_emb = torch.nn.functional.normalize(image_emb, dim=-1)
                combined = (query_emb + image_emb) / 2
                combined = torch.nn.functional.normalize(combined, dim=-1)
            
            return combined.flatten()
        return self.embed_text(text)

class Qwen2VLEmbed(BaseMultimodalEmbedder):
    """
    Qwen2-VL for multimodal embeddings
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        
        print(f"Loading Qwen2-VL: {model_name}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ) if self.device == "cuda" else None
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config,
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        print(f"✅ Qwen2-VL loaded on {self.device}")
    
    def embed_text(self, text: str) -> torch.Tensor:
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[-1].mean(dim=1).squeeze()
        
        return embedding.flatten()
    
    def embed_image(self, image_path: str) -> torch.Tensor:
        if not Path(image_path).exists():
            return torch.zeros(1536, device=self.device)
        
        image = Image.open(image_path).convert('RGB')
        messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[-1].mean(dim=1).squeeze()
        
        return embedding.flatten()
    
    def embed_multimodal(self, text: str, image_path: Optional[str] = None) -> torch.Tensor:
        if image_path and Path(image_path).exists():
            image = Image.open(image_path).convert('RGB')
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text}
                ]
            }]
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                embedding = outputs.hidden_states[-1].mean(dim=1).squeeze()
            
            return embedding.flatten()
        return self.embed_text(text)


class Qwen3VLEmbed(BaseMultimodalEmbedder):
    """
    Qwen3-VL Embedding model for multimodal embeddings
    Preferred model for document embeddings with tables and images
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-Embedding-8B", gpus: Optional[List[int]] = None):
        from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig
        
        # Check for local model first
        resolved_path = _resolve_local_model_path(model_name)
        
        print(f"Loading Qwen3-VL: {resolved_path}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.device = get_device_map_for_gpus(gpus) if gpus else ("cuda" if torch.cuda.is_available() else "cpu")
        self.gpus = gpus
        
        # NOTE: Qwen3-VL-Embedding-8B has issues with bitsandbytes 4-bit quantization
        # ('weight' is not an nn.Module error). Load without quantization for now.
        # If OOM occurs, try the 2B variant or use 8-bit quantization instead.
        use_quantization = False  # Disabled due to bitsandbytes compatibility issue
        
        if use_quantization and self.device.startswith("cuda"):
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,  # Use 8-bit instead of 4-bit for better compatibility
            )
        else:
            quantization_config = None
        
        self.model = AutoModel.from_pretrained(
            resolved_path,
            torch_dtype=torch.bfloat16,
            device_map="auto" if self.device.startswith("cuda") else None,
            quantization_config=quantization_config,
            trust_remote_code=True,
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(resolved_path, trust_remote_code=True)
        self.embedding_dim = 4096  # Qwen3-VL embedding dimension
        print(f"✅ Qwen3-VL loaded on {self.device}")
    
    def embed_text(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                embedding = outputs.hidden_states[-1].mean(dim=1).squeeze()
            elif hasattr(outputs, 'last_hidden_state'):
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            else:
                embedding = outputs[0].mean(dim=1).squeeze()
        return embedding.float().flatten()
    
    def embed_image(self, image_path: str) -> torch.Tensor:
        if not Path(image_path).exists():
            return torch.zeros(self.embedding_dim, device=self.device, dtype=torch.float32)
        
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                embedding = outputs.hidden_states[-1].mean(dim=1).squeeze()
            elif hasattr(outputs, 'last_hidden_state'):
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            else:
                embedding = outputs[0].mean(dim=1).squeeze()
        return embedding.float().flatten()
    
    def embed_multimodal(self, text: str, image_path: Optional[str] = None) -> torch.Tensor:
        if image_path and Path(image_path).exists():
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    embedding = outputs.hidden_states[-1].mean(dim=1).squeeze()
                elif hasattr(outputs, 'last_hidden_state'):
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                else:
                    embedding = outputs[0].mean(dim=1).squeeze()
            return embedding.float().flatten()
        return self.embed_text(text)


def get_multimodal_embedder(gpus: Optional[List[int]] = None):
    """
    Try to load multimodal embedding models in order of preference:
    1. Qwen3-VL (best for documents with tables/images)
    2. Nomic multimodal
    3. Fall back to bge-m3 (text-only) with warning
    
    Returns:
        Tuple of (embedder, model_name, is_multimodal)
    """
    # Try Qwen3-VL first
    try:
        print("🔍 Trying to load Qwen3-VL multimodal embeddings...")
        embedder = Qwen3VLEmbed(gpus=gpus)
        return embedder, "qwen3_vl", True
    except Exception as e:
        print(f"⚠️  Qwen3-VL not available: {e}")
    
    # Try Nomic multimodal
    try:
        print("🔍 Trying to load Nomic multimodal embeddings...")
        embedder = NomicVLEmbed(gpus=gpus)
        return embedder, "nomic", True
    except Exception as e:
        print(f"⚠️  Nomic multimodal not available: {e}")
    
    # Fall back to bge-m3 (text-only)
    print("=" * 70)
    print("⚠️  MULTIMODAL EMBEDDING MODELS NOT AVAILABLE")
    print("   Falling back to bge-m3 (text-only embeddings)")
    print("   Tables and images will be embedded using text descriptions only")
    print("=" * 70)
    
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(get_best_embedding_model())
    return embedder, "bge_m3", False


class VLMDescriptionEmbed(BaseMultimodalEmbedder):
    """
    VLM Description-based Embedder
    """
    
    def __init__(self, 
                 text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 vlm_api_url: str = "https://api.openai.com/v1/chat/completions",
                 vlm_model_name: str = "gpt-4o"):
        from sentence_transformers import SentenceTransformer
        
        print(f"Loading VLM Description Embedder with text model: {text_model_name}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load text embedding model
        self.text_model = SentenceTransformer(text_model_name, device=self.device)
        
        # Setup VLM API
        self.vlm_api_url = vlm_api_url
        self.vlm_model_name = vlm_model_name
        
        # Load API key (use environment or config file)
        api_key_path = os.environ.get("OPENAI_API_KEY_PATH", os.path.expanduser("~/.config/openai/api_key.txt"))
        with open(api_key_path, 'r') as f:
            self.api_key = f.read().strip()
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        print(f"✅ VLM Description Embedder loaded on {self.device}")
    
    def _describe_image(self, image_path: str) -> str:
        """Use VLM API to generate textual description of image"""
        
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Prepare API request
        payload = {
            "model": self.vlm_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Describe this image with concise details, focusing on technical content, diagrams, charts, tables, and text visible in the image."
                        }
                    ]
                }
            ],
            "max_tokens": 500
        }
        
        # Call API with timeout (increased to 180 seconds for large/complex images)
        response = requests.post(self.vlm_api_url, headers=self.headers, json=payload, timeout=180)
        response.raise_for_status()
        
        result = response.json()
        description = result['choices'][0]['message']['content']
        
        return description
    
    def embed_text(self, text: str) -> torch.Tensor:
        embedding = self.text_model.encode(text, convert_to_tensor=True, device=self.device)
        return embedding.flatten()
    
    def embed_image(self, image_path: str) -> torch.Tensor:
        if not Path(image_path).exists():
            return torch.zeros(384, device=self.device)
        
        description = self._describe_image(image_path)
        return self.embed_text(description)
    
    def embed_multimodal(self, text: str, image_path: Optional[str] = None) -> torch.Tensor:
        if image_path and Path(image_path).exists():
            description = self._describe_image(image_path)
            combined_text = f"{text}\n\nImage description: {description}"
            return self.embed_text(combined_text)
        return self.embed_text(text)

class BGEVLEmbed(BaseMultimodalEmbedder):
    """
    BGE-VL-v1.5-mmeb (MLLM variant)
    """
    
    def __init__(self, model_name: str = "BAAI/BGE-VL-v1.5-zs"):
        from transformers import AutoModel
        
        print(f"Loading BGE-VL-v1.5: {model_name}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
        # Add missing image_newline attribute if needed
        if not hasattr(self.model, 'image_newline'):
            try:
                hidden_size = self.model.config.text_config.hidden_size
            except:
                try:
                    hidden_size = self.model.config.hidden_size
                except:
                    hidden_size = 4096
            import torch.nn as nn
            self.model.image_newline = nn.Parameter(
                torch.zeros(hidden_size, dtype=torch.float16)
            )
        
        # Set processor
        with torch.no_grad():
            self.model.set_processor(model_name)
        
        # Fix processor patch_size issue
        if hasattr(self.model, 'processor'):
            proc = self.model.processor
            if hasattr(proc, 'patch_size') and proc.patch_size is None:
                proc.patch_size = 14
            if hasattr(proc, 'image_processor'):
                img_proc = proc.image_processor
                if hasattr(img_proc, 'patch_size') and img_proc.patch_size is None:
                    img_proc.patch_size = 14
        
        self._patch_forward_method()
        
        # Get embedding dimension
        with torch.no_grad():
            try:
                test_inputs = self.model.data_process(text="test", q_or_c="c")
                test_outputs = self.model(**test_inputs, output_hidden_states=True)
                if hasattr(test_outputs, 'hidden_states'):
                    test_emb = test_outputs.hidden_states[-1][:, -1, :]
                else:
                    test_emb = test_outputs[:, -1, :]
                self.embedding_dim = test_emb.shape[-1]
            except Exception:
                self.embedding_dim = 4096
        
        print(f"✅ BGE-VL-v1.5 loaded on {self.device}, dim: {self.embedding_dim}")
    
    def _patch_forward_method(self):
        import types
        if hasattr(self.model, 'pack_image_features'):
            original_pack = self.model.pack_image_features
            def fixed_pack_image_features(self_model, image_features, image_sizes, **kwargs):
                result, feature_lens = original_pack(image_features, image_sizes, **kwargs)
                if isinstance(result, list):
                    if len(result) > 0 and isinstance(result[0], torch.Tensor):
                        try:
                            result = torch.stack(result, dim=0) if len(result) > 1 else result[0]
                        except:
                            try:
                                result = torch.cat(result, dim=0)
                            except:
                                result = result[0]
                return result, feature_lens
            self.model.pack_image_features = types.MethodType(fixed_pack_image_features, self.model)
            
        model_class = self.model.__class__
        if not hasattr(model_class, '_bgevl_original_forward'):
            model_class._bgevl_original_forward = model_class.forward
            def patched_forward(self, *args, **kwargs):
                 if hasattr(self, 'pack_image_features'):
                    original_pack = self.pack_image_features
                    def fixed_pack_image_features(image_features, image_sizes, **kwargs):
                        result, feature_lens = original_pack(image_features, image_sizes, **kwargs)
                        if isinstance(result, list):
                            if len(result) > 0 and isinstance(result[0], torch.Tensor):
                                try:
                                    result = torch.stack(result, dim=0) if len(result) > 1 else result[0]
                                except:
                                    try:
                                        result = torch.cat(result, dim=0)
                                    except:
                                        result = result[0]
                        return result, feature_lens
                    self.pack_image_features = fixed_pack_image_features
                    try:
                        return model_class._bgevl_original_forward(self, *args, **kwargs)
                    finally:
                        self.pack_image_features = original_pack
                 else:
                    return model_class._bgevl_original_forward(self, *args, **kwargs)
            model_class.forward = patched_forward
            
        if hasattr(self.model, 'vision_tower'):
            vt_class = self.model.vision_tower.__class__
            if not hasattr(vt_class, '_original_vt_forward'):
                vt_class._original_vt_forward = vt_class.forward
                def fixed_vt_forward(vt_self, pixel_values, *args, **kwargs):
                    if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 5:
                        b, n, c, h, w = pixel_values.shape
                        pixel_values = pixel_values.reshape(b * n, c, h, w)
                    return vt_class._original_vt_forward(vt_self, pixel_values, *args, **kwargs)
                vt_class.forward = fixed_vt_forward

    def embed_text(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.model.data_process(text=text, q_or_c="c")
            outputs = self.model(**inputs, output_hidden_states=True)
            if hasattr(outputs, 'hidden_states'):
                embedding = outputs.hidden_states[-1][:, -1, :]
            else:
                embedding = outputs[:, -1, :]
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return embedding.to(device=self.device, dtype=torch.float32).flatten()
    
    def embed_image(self, image_path: str) -> torch.Tensor:
        if not Path(image_path).exists():
            return torch.zeros(self.embedding_dim, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            inputs = self.model.data_process(images=str(image_path), q_or_c="c")
            outputs = self.model(**inputs, output_hidden_states=True)
            if hasattr(outputs, 'hidden_states'):
                embedding = outputs.hidden_states[-1][:, -1, :]
            else:
                embedding = outputs[:, -1, :]
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return embedding.to(device=self.device, dtype=torch.float32).flatten()
    
    def embed_multimodal(self, text: str, image_path: Optional[str] = None) -> torch.Tensor:
        if image_path and Path(image_path).exists():
            with torch.no_grad():
                inputs = self.model.data_process(
                    text=text,
                    images=str(image_path),
                    q_or_c="c"
                )
                outputs = self.model(**inputs, output_hidden_states=True)
                if hasattr(outputs, 'hidden_states'):
                    embedding = outputs.hidden_states[-1][:, -1, :]
                else:
                    embedding = outputs[:, -1, :]
                embedding = torch.nn.functional.normalize(embedding, dim=-1)
                return embedding.to(device=self.device, dtype=torch.float32).flatten()
        return self.embed_text(text)


# =============================================================================
# MODULAR EMBEDDING MODULE (scikit-learn style)
# =============================================================================

from mirage.core.base import BaseModelModule


class EmbeddingModule(BaseModelModule):
    """
    Embedding module for generating document/chunk embeddings.
    
    All hyperparameters from Table 2.11 (Embedding Models):
        model: 'nomic', 'qwen3_vl', 'qwen2_vl', 'bge_vl', 'bge_m3', 'bge_large', 'minilm'
        batch_size: Batch size for embedding (default: 16)
        normalize: Whether to L2 normalize embeddings (default: True)
        cache_embeddings: Cache model in memory (default: True)
        gpu_id: GPU device ID to use
        load_in_4bit: Enable 4-bit quantization (default: True, CUDA only)
        bnb_4bit_compute_dtype: Compute dtype for quantization (default: bfloat16)
        bnb_4bit_use_double_quant: Enable double quantization (default: True)
        torch_dtype: Model data type (default: bfloat16)
        attn_implementation: Attention implementation (default: flash_attention_2)
        
    Model details:
        - nomic: nomic-ai/nomic-embed-multimodal-7b (multimodal, dim=128)
        - qwen2_vl: Qwen/Qwen2-VL-7B-Instruct (multimodal, dim=1536)
        - bge_vl: BAAI/BGE-VL-v1.5-zs (multimodal, dim=768)
        - bge_m3: BAAI/bge-m3 (text-only, dim=1024)
        - bge_large: BAAI/bge-large-en-v1.5 (text-only, dim=1024)
        - minilm: sentence-transformers/all-MiniLM-L6-v2 (text-only, dim=384)
        
    Example:
        embedder = EmbeddingModule(model='nomic', batch_size=16)
        embeddings = embedder.process(chunks)
    """
    
    # Model class mapping
    MODEL_CLASSES = {
        'nomic': NomicVLEmbed,
        'qwen3_vl': Qwen3VLEmbed,
        'qwen2_vl': Qwen2VLEmbed,
        'bge_vl': BGEVLEmbed,
        'vlm_description': VLMDescriptionEmbed,
    }
    
    SENTENCE_TRANSFORMER_MODELS = {
        'bge_m3': 'BAAI/bge-m3',
        'bge_large': 'BAAI/bge-large-en-v1.5',
        'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
    }
    
    MODEL_DIMENSIONS = {
        'nomic': 128,
        'qwen2_vl': 1536,
        'qwen3_vl': 1536,
        'bge_vl': 768,
        'bge_m3': 1024,
        'bge_large': 1024,
        'minilm': 384,
    }
    
    def __init__(self,
                 model: str = 'nomic',
                 batch_size: int = 16,
                 normalize: bool = True,
                 cache_embeddings: bool = True,
                 gpu_id: int = None,
                 load_in_4bit: bool = True,
                 bnb_4bit_compute_dtype: str = 'bfloat16',
                 bnb_4bit_use_double_quant: bool = True,
                 torch_dtype: str = 'bfloat16',
                 attn_implementation: str = 'flash_attention_2',
                 **kwargs):
        """Initialize embedding module with all hyperparameters from Table 2.11."""
        super().__init__(
            model=model,
            cache_model=cache_embeddings,
            gpu_id=gpu_id,
            batch_size=batch_size,
            normalize=normalize,
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            **kwargs
        )
        self._embeddings = None
        self._chunk_ids = None
    
    def _load_model(self):
        """Load the embedding model with configured parameters."""
        model_name = self.model_name
        gpu_id = self._params.get('gpu_id')
        gpus = [gpu_id] if gpu_id is not None else None
        
        if model_name in self.MODEL_CLASSES:
            self._model = self.MODEL_CLASSES[model_name](gpus=gpus)
        elif model_name in self.SENTENCE_TRANSFORMER_MODELS:
            from sentence_transformers import SentenceTransformer
            hf_name = self.SENTENCE_TRANSFORMER_MODELS[model_name]
            self._model = SentenceTransformer(hf_name)
        elif '/' in model_name:
            # HuggingFace model path
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
        else:
            # Try auto-detection
            embedder, actual_model, is_multimodal = get_multimodal_embedder(gpus=gpus)
            self._model = embedder
    
    def process(self, X, **kwargs) -> np.ndarray:
        """
        Process chunks to embeddings.
        
        Args:
            X: List of chunk dicts (with 'content' key) or list of strings
            **kwargs: Runtime overrides for parameters
            
        Returns:
            numpy array of shape (n_samples, embedding_dim)
        """
        # Apply runtime overrides
        params = {**self._params, **kwargs}
        
        model = self.get_model()
        batch_size = params.get('batch_size', 16)
        normalize = params.get('normalize', True)
        
        # Extract text content
        if isinstance(X, list) and len(X) > 0:
            if isinstance(X[0], dict):
                texts = [chunk.get('content', str(chunk)) for chunk in X]
                self._chunk_ids = [chunk.get('chunk_id', str(i)) for i, chunk in enumerate(X)]
            else:
                texts = [str(x) for x in X]
                self._chunk_ids = [str(i) for i in range(len(X))]
        else:
            texts = [str(X)]
            self._chunk_ids = ['0']
        
        # Embed in batches
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            if hasattr(model, 'encode'):
                # SentenceTransformer or compatible API
                batch_emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            else:
                # Custom embedder - encode one by one
                batch_emb = []
                for text in batch:
                    emb = model.embed_text(text)
                    if isinstance(emb, torch.Tensor):
                        emb = emb.cpu().float().numpy()
                    batch_emb.append(emb)
                batch_emb = np.array(batch_emb)
            
            embeddings.append(batch_emb)
        
        # Stack all embeddings
        self._embeddings = np.vstack(embeddings).astype('float32')
        
        # Normalize if requested (required for cosine similarity with FAISS IndexFlatIP)
        if normalize:
            import faiss
            faiss.normalize_L2(self._embeddings)
        
        self._logger.info(f"Generated embeddings: shape {self._embeddings.shape}")
        self._output = self._embeddings
        return self._embeddings
    
    def get_embeddings(self) -> np.ndarray:
        """Return cached embeddings."""
        return self._embeddings
    
    def get_chunk_ids(self) -> List[str]:
        """Return chunk IDs corresponding to embeddings."""
        return self._chunk_ids
    
    def get_dimension(self) -> int:
        """Get embedding dimension for current model."""
        model_name = self.model_name
        return self.MODEL_DIMENSIONS.get(model_name, 768)
    
    def build_index(self, embeddings: np.ndarray = None) -> 'faiss.Index':
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Embeddings array (uses cached if None)
            
        Returns:
            FAISS index (IndexFlatIP for cosine similarity)
        """
        import faiss
        
        if embeddings is None:
            embeddings = self._embeddings
        
        if embeddings is None:
            raise ValueError("No embeddings available. Call process() first.")
        
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product for cosine sim (after L2 normalization)
        index.add(embeddings)
        
        self._logger.info(f"Built FAISS index: {index.ntotal} vectors, dim={dim}")
        return index
