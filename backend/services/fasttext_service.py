# services/fasttext_service.py

"""
FastText Service - Manages loading and caching of FastText models
Load once at startup, reuse for all requests
"""

from gensim.models import KeyedVectors
from typing import Dict, Optional
import os
import time

class FastTextManager:
    """Singleton manager for FastText models"""
    
    _instance = None
    _models: Dict[str, KeyedVectors] = {}
    _loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FastTextManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize (but don't load yet)"""
        pass
    
    def load_models(self, limit: int = None, force_reload: bool = False, use_cache: bool = True):
        """
        Load FastText models into memory
        
        Args:
            limit: Maximum number of vectors to load (None = load all)
            force_reload: Force reload even if already loaded
            use_cache: Use cached binary format for faster loading
        """
        if self._loaded and not force_reload:
            print("FastText models already loaded")
            return
        
        print("\n" + "="*60)
        print("🚀 Loading FastText Models...")
        print("="*60)
        
        tagalog_path = "data/fastText-Tagalog.vec"
        cebuano_path = "data/fastText-Cebuano.vec"
        
        # Cached versions (much faster to load)
        tagalog_cache = "data/fastText-Tagalog.bin"
        cebuano_cache = "data/fastText-Cebuano.bin"
        
        # Load Tagalog model
        if os.path.exists(tagalog_path):
            print(f"\n📚 Loading Tagalog model...")
            start_time = time.time()
            
            try:
                # Try to load from cache first
                if use_cache and os.path.exists(tagalog_cache):
                    print(f"   ⚡ Loading from cache: {tagalog_cache}")
                    self._models['tagalog'] = KeyedVectors.load(tagalog_cache)
                    elapsed = time.time() - start_time
                    print(f"   ✅ Loaded from cache in {elapsed:.1f}s (10x faster!)")
                else:
                    # Load from .vec file
                    print(f"   📖 Loading from: {tagalog_path}")
                    if limit:
                        print(f"   Limit: {limit:,} vectors")
                    else:
                        print(f"   Loading ALL vectors (no limit)")
                    
                    self._models['tagalog'] = KeyedVectors.load_word2vec_format(
                        tagalog_path,
                        binary=False,
                        limit=limit,
                        unicode_errors='ignore'
                    )
                    elapsed = time.time() - start_time
                    print(f"   ✅ Loaded {len(self._models['tagalog']):,} words in {elapsed:.1f}s")
                    
                    # Save to cache for next time
                    if use_cache:
                        print(f"   💾 Saving to cache: {tagalog_cache}")
                        self._models['tagalog'].save(tagalog_cache)
                        print(f"   ✅ Cache saved! Next startup will be 10x faster")
                
                print(f"   Memory: ~{len(self._models['tagalog']) * 300 * 4 / 1024 / 1024:.0f}MB")
            except Exception as e:
                print(f"   ❌ Error loading Tagalog model: {str(e)}")
        else:
            print(f"   ⚠️ Tagalog model not found at: {tagalog_path}")
        
        # Load Cebuano model
        if os.path.exists(cebuano_path):
            print(f"\n📚 Loading Cebuano model...")
            start_time = time.time()
            
            try:
                # Try to load from cache first
                if use_cache and os.path.exists(cebuano_cache):
                    print(f"   ⚡ Loading from cache: {cebuano_cache}")
                    self._models['cebuano'] = KeyedVectors.load(cebuano_cache)
                    elapsed = time.time() - start_time
                    print(f"   ✅ Loaded from cache in {elapsed:.1f}s (10x faster!)")
                else:
                    # Load from .vec file
                    print(f"   📖 Loading from: {cebuano_path}")
                    if limit:
                        print(f"   Limit: {limit:,} vectors")
                    else:
                        print(f"   Loading ALL vectors (no limit)")
                    
                    self._models['cebuano'] = KeyedVectors.load_word2vec_format(
                        cebuano_path,
                        binary=False,
                        limit=limit,
                        unicode_errors='ignore'
                    )
                    elapsed = time.time() - start_time
                    print(f"   ✅ Loaded {len(self._models['cebuano']):,} words in {elapsed:.1f}s")
                    
                    # Save to cache for next time
                    if use_cache:
                        print(f"   💾 Saving to cache: {cebuano_cache}")
                        self._models['cebuano'].save(cebuano_cache)
                        print(f"   ✅ Cache saved! Next startup will be 10x faster")
                
                print(f"   Memory: ~{len(self._models['cebuano']) * 300 * 4 / 1024 / 1024:.0f}MB")
            except Exception as e:
                print(f"   ❌ Error loading Cebuano model: {str(e)}")
        else:
            print(f"   ⚠️ Cebuano model not found at: {cebuano_path}")
        
        if not self._models:
            raise RuntimeError("No FastText models loaded! Check your data/ directory.")
        
        self._loaded = True
        print("\n" + "="*60)
        print(f"✨ FastText models ready! Total: {len(self._models)} languages")
        print("="*60 + "\n")
    
    def get_model(self, language: str) -> Optional[KeyedVectors]:
        """Get a loaded model by language"""
        return self._models.get(language.lower())
    
    def is_loaded(self) -> bool:
        """Check if models are loaded"""
        return self._loaded
    
    def get_status(self) -> dict:
        """Get loading status and info"""
        return {
            "loaded": self._loaded,
            "models": {
                lang: {
                    "vocab_size": len(model),
                    "vector_size": model.vector_size if hasattr(model, 'vector_size') else 300
                }
                for lang, model in self._models.items()
            }
        }
    
    def unload_models(self):
        """Unload models to free memory (for testing/debugging)"""
        self._models.clear()
        self._loaded = False
        print("FastText models unloaded")


# Global instance
fasttext_manager = FastTextManager()


def get_fasttext_manager() -> FastTextManager:
    """Get the global FastText manager instance"""
    return fasttext_manager