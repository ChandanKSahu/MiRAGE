#!/usr/bin/env python3
"""
LLM Response Cache for MiRAGE Pipeline

Caches all LLM/VLM responses to avoid redundant API calls.
Saves after each call for crash resilience.

Features:
- Content-based hashing for cache keys
- Atomic writes to prevent corruption
- Automatic cache hit/miss logging
- Statistics tracking
"""

import json
import os
import hashlib
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import logging


class LLMCache:
    """Thread-safe LLM response cache with persistent storage."""
    
    def __init__(self, cache_dir: str, enabled: bool = True):
        """Initialize LLM cache.
        
        Args:
            cache_dir: Directory to store cache files
            enabled: Whether caching is enabled
        """
        self.enabled = enabled
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "llm_responses.json"
        self.stats_file = self.cache_dir / "llm_cache_stats.json"
        
        # Thread safety
        self._lock = threading.Lock()
        
        # In-memory cache
        self._cache: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'saves': 0,
            'errors': 0,
            'total_tokens_saved': 0,  # Estimated
            'session_start': datetime.now().isoformat()
        }
        
        if enabled:
            self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
                print(f"📦 LLM Cache: Loaded {len(self._cache)} cached responses")
            except (json.JSONDecodeError, IOError) as e:
                logging.warning(f"Failed to load LLM cache: {e}")
                self._cache = {}
        
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    saved_stats = json.load(f)
                    # Merge saved stats (cumulative)
                    self._stats['hits'] = saved_stats.get('total_hits', 0)
                    self._stats['misses'] = saved_stats.get('total_misses', 0)
                    self._stats['saves'] = saved_stats.get('total_saves', 0)
            except (json.JSONDecodeError, IOError):
                pass
    
    def _save_cache(self):
        """Save cache to disk atomically."""
        if not self.enabled:
            return
        
        temp_file = self.cache_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False)
            temp_file.replace(self.cache_file)
            self._stats['saves'] += 1
        except IOError as e:
            logging.error(f"Failed to save LLM cache: {e}")
            self._stats['errors'] += 1
            if temp_file.exists():
                temp_file.unlink()
    
    def _save_stats(self):
        """Save statistics to disk."""
        if not self.enabled:
            return
        
        stats_to_save = {
            'total_hits': self._stats['hits'],
            'total_misses': self._stats['misses'],
            'total_saves': self._stats['saves'],
            'total_errors': self._stats['errors'],
            'cache_size': len(self._cache),
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_to_save, f, indent=2)
        except IOError:
            pass
    
    @staticmethod
    def _hash_content(content: Any) -> str:
        """Generate hash for content (prompt, chunks, etc.)."""
        if isinstance(content, str):
            data = content
        elif isinstance(content, (list, dict)):
            # Stable JSON serialization
            data = json.dumps(content, sort_keys=True, ensure_ascii=False)
        else:
            data = str(content)
        
        return hashlib.sha256(data.encode('utf-8')).hexdigest()[:32]
    
    @staticmethod
    def _extract_image_path_for_cache(chunk: Dict) -> str:
        """Extract image path from chunk for cache key generation.
        
        Mirrors the logic in llm.py:_extract_image_path() to ensure cache
        keys match actual API call content.
        """
        import re
        
        # Check for artifact list (new format)
        artifact = chunk.get('artifact', [])
        if isinstance(artifact, list) and len(artifact) > 0:
            return artifact[0]
        
        # Check image_path field (backward compatibility)
        image_path = chunk.get('image_path')
        if image_path:
            return image_path
        
        # Legacy format with chunk_type and artifact string
        if 'chunk_type' in chunk:
            chunk_type = chunk.get('chunk_type', '')
            artifact_str = chunk.get('artifact', 'None')
            if chunk_type in ['standalone image', 'image'] and artifact_str != 'None':
                match = re.search(r'!\[Image\]\(([^)]+)\)', str(artifact_str))
                if match:
                    return match.group(1)
        
        return ''
    
    def _make_key(self, call_type: str, prompt: str, chunks: Optional[List[Dict]] = None,
                  model: str = None) -> str:
        """Generate cache key from call parameters.
        
        Args:
            call_type: Type of call (llm, vlm, vlm_batch, etc.)
            prompt: The prompt text
            chunks: Optional list of chunk dicts (for VLM calls)
            model: Optional model name
        
        Returns:
            Cache key string
        """
        # Hash prompt
        prompt_hash = self._hash_content(prompt)
        
        # Hash chunks if present (extract content and resolved image path for stability)
        if chunks:
            chunk_data = []
            for c in chunks:
                if isinstance(c, dict):
                    chunk_data.append({
                        'content': c.get('content', ''),
                        'image_path': self._extract_image_path_for_cache(c),
                        'chunk_id': c.get('chunk_id', '')
                    })
                else:
                    chunk_data.append(str(c))
            chunks_hash = self._hash_content(chunk_data)
        else:
            chunks_hash = "none"
        
        # Include model in key if specified
        model_part = f"_{model}" if model else ""
        
        return f"{call_type}{model_part}_{prompt_hash}_{chunks_hash}"
    
    def get(self, call_type: str, prompt: str, chunks: Optional[List[Dict]] = None,
            model: str = None) -> Optional[str]:
        """Get cached response if available.
        
        Args:
            call_type: Type of call
            prompt: The prompt text
            chunks: Optional chunks for VLM calls
            model: Optional model name
        
        Returns:
            Cached response string or None if not found
        """
        if not self.enabled:
            return None
        
        key = self._make_key(call_type, prompt, chunks, model)
        
        with self._lock:
            if key in self._cache:
                self._stats['hits'] += 1
                entry = self._cache[key]
                # Estimate tokens saved (rough: 4 chars per token)
                self._stats['total_tokens_saved'] += len(entry.get('response', '')) // 4
                return entry.get('response')
            else:
                self._stats['misses'] += 1
                return None
    
    def set(self, call_type: str, prompt: str, response: str, 
            chunks: Optional[List[Dict]] = None, model: str = None,
            metadata: Optional[Dict] = None):
        """Cache a response.
        
        Args:
            call_type: Type of call
            prompt: The prompt text
            response: The LLM response
            chunks: Optional chunks for VLM calls
            model: Optional model name
            metadata: Optional metadata to store
        """
        if not self.enabled:
            return
        
        key = self._make_key(call_type, prompt, chunks, model)
        
        entry = {
            'response': response,
            'call_type': call_type,
            'model': model,
            'timestamp': datetime.now().isoformat(),
            'prompt_length': len(prompt),
            'response_length': len(response)
        }
        
        if metadata:
            entry['metadata'] = metadata
        
        with self._lock:
            self._cache[key] = entry
            self._save_cache()
    
    def get_batch(self, call_type: str, requests: List[Tuple[str, Optional[List[Dict]]]],
                  model: str = None) -> Tuple[List[Optional[str]], List[int]]:
        """Get cached responses for batch requests.
        
        Args:
            call_type: Type of call
            requests: List of (prompt, chunks) tuples
            model: Optional model name
        
        Returns:
            Tuple of (responses list with None for misses, indices of misses)
        """
        if not self.enabled:
            return [None] * len(requests), list(range(len(requests)))
        
        responses = []
        miss_indices = []
        
        for i, (prompt, chunks) in enumerate(requests):
            cached = self.get(call_type, prompt, chunks, model)
            responses.append(cached)
            if cached is None:
                miss_indices.append(i)
        
        return responses, miss_indices
    
    def set_batch(self, call_type: str, requests: List[Tuple[str, Optional[List[Dict]]]],
                  responses: List[str], indices: List[int], model: str = None):
        """Cache multiple responses at once.
        
        Args:
            call_type: Type of call
            requests: List of (prompt, chunks) tuples
            responses: List of responses (matching indices)
            indices: Indices in requests that these responses correspond to
            model: Optional model name
        """
        if not self.enabled:
            return
        
        with self._lock:
            for idx, response in zip(indices, responses):
                if idx < len(requests) and response:
                    prompt, chunks = requests[idx]
                    key = self._make_key(call_type, prompt, chunks, model)
                    self._cache[key] = {
                        'response': response,
                        'call_type': call_type,
                        'model': model,
                        'timestamp': datetime.now().isoformat(),
                        'prompt_length': len(prompt),
                        'response_length': len(response)
                    }
            self._save_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = (self._stats['hits'] / max(1, self._stats['hits'] + self._stats['misses'])) * 100
            return {
                'cache_size': len(self._cache),
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': f"{hit_rate:.1f}%",
                'saves': self._stats['saves'],
                'errors': self._stats['errors'],
                'estimated_tokens_saved': self._stats['total_tokens_saved']
            }
    
    def print_stats(self):
        """Print cache statistics."""
        stats = self.get_stats()
        print("\n📦 LLM CACHE STATISTICS")
        print("=" * 50)
        print(f"   Cache size:     {stats['cache_size']} responses")
        print(f"   Cache hits:     {stats['hits']}")
        print(f"   Cache misses:   {stats['misses']}")
        print(f"   Hit rate:       {stats['hit_rate']}")
        print(f"   Est. tokens saved: ~{stats['estimated_tokens_saved']:,}")
        print("=" * 50)
        self._save_stats()
    
    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache = {}
            if self.cache_file.exists():
                self.cache_file.unlink()
            print("🗑️  LLM cache cleared")


# Global cache instance (initialized in main.py)
_LLM_CACHE: Optional[LLMCache] = None


def get_llm_cache() -> Optional[LLMCache]:
    """Get the global LLM cache instance."""
    return _LLM_CACHE


def init_llm_cache(cache_dir: str, enabled: bool = True) -> LLMCache:
    """Initialize the global LLM cache.
    
    Args:
        cache_dir: Directory for cache files
        enabled: Whether caching is enabled
    
    Returns:
        The initialized cache instance
    """
    global _LLM_CACHE
    _LLM_CACHE = LLMCache(cache_dir, enabled)
    return _LLM_CACHE
