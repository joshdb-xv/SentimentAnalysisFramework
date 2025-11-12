import joblib
import pandas as pd
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path


class LexicalDictionaryManager:
    """
    Manages cached lexical dictionaries with ML-style persistence
    Supports loading, saving, and on-the-fly word label updates with recalculation
    """
    
    def __init__(self, cache_dir: str = "data/lexical_dictionary"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.joblib_path = self.cache_dir / "lexical_dictionary.joblib"
        self.csv_path = self.cache_dir / "lexical_dictionary.csv"
        self.metadata_path = self.cache_dir / "lexical_metadata.json"
        
        # In-memory storage
        self.lexicon_df: Optional[pd.DataFrame] = None
        self.word_data: Optional[Dict] = None  # Stores full word info (definition, dialect, etc.)
        self.processor = None  # Reference to LexicalProcessor for recalculation
        self.metadata: Dict[str, Any] = {}
    
    def exists(self) -> bool:
        """Check if a cached dictionary exists"""
        return self.joblib_path.exists()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current cache status"""
        if not self.exists():
            return {
                "cached": False,
                "message": "No cached dictionary found",
                "path": str(self.joblib_path)
            }
        
        metadata = self.load_metadata()
        
        return {
            "cached": True,
            "loaded": self.lexicon_df is not None,
            "path": str(self.joblib_path),
            "metadata": metadata,
            "csv_backup_exists": self.csv_path.exists()
        }
    
    def load(self) -> bool:
        """Load cached dictionary from joblib"""
        try:
            if not self.exists():
                print("No cached dictionary found")
                return False
            
            print(f"Loading cached dictionary from {self.joblib_path}...")
            
            # Load the joblib file
            cache_data = joblib.load(self.joblib_path)
            
            self.lexicon_df = cache_data['lexicon']
            self.word_data = cache_data['word_data']
            self.metadata = cache_data.get('metadata', {})
            
            print(f"✅ Loaded {len(self.lexicon_df)} words from cache")
            print(f"   Created: {self.metadata.get('created_at', 'unknown')}")
            print(f"   Version: {self.metadata.get('version', 'unknown')}")
            
            return True
            
        except Exception as e:
            print(f"Error loading cache: {str(e)}")
            return False
    
    def save(self, lexicon_df: pd.DataFrame, word_data: Dict, 
             processor=None, stats: Dict = None) -> bool:
        """
        Save processed dictionary as cached model
        
        Args:
            lexicon_df: DataFrame with columns [word, sentiment_score]
            word_data: Dict mapping word -> full row data (definition, dialect, etc.)
            processor: LexicalProcessor instance (for recalculation)
            stats: Processing statistics
        """
        try:
            print(f"Saving lexical dictionary to {self.joblib_path}...")
            
            # Prepare metadata
            metadata = {
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "total_words": len(lexicon_df),
                "stats": stats or {},
                "last_updated": datetime.now().isoformat()
            }
            
            # Prepare cache data
            cache_data = {
                "lexicon": lexicon_df,
                "word_data": word_data,
                "metadata": metadata
            }
            
            # Save as joblib
            joblib.dump(cache_data, self.joblib_path)
            print(f"✅ Saved to {self.joblib_path}")
            
            # Save CSV backup
            lexicon_df.to_csv(self.csv_path, index=False)
            print(f"✅ CSV backup saved to {self.csv_path}")
            
            # Save metadata separately
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✅ Metadata saved to {self.metadata_path}")
            
            # Update in-memory
            self.lexicon_df = lexicon_df
            self.word_data = word_data
            self.processor = processor
            self.metadata = metadata
            
            return True
            
        except Exception as e:
            print(f"Error saving cache: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load metadata without loading full dictionary"""
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
        
        return {}
    
    def update_word_label(self, word: str, new_label: str, 
                          processor=None) -> Dict[str, Any]:
        """
        Update a word's sentiment label and recalculate its score
        
        Args:
            word: The word to update
            new_label: New sentiment label (positive/negative/neutral)
            processor: LexicalProcessor instance for recalculation (optional)
            
        Returns:
            Dict with old_score, new_score, breakdown
        """
        word = word.lower().strip()
        
        # Validate word exists
        if self.lexicon_df is None:
            raise ValueError("No dictionary loaded")
        
        if word not in self.lexicon_df['word'].values:
            raise ValueError(f"Word '{word}' not found in dictionary")
        
        # Validate label
        new_label = new_label.lower()
        if new_label not in ['positive', 'negative', 'neutral']:
            raise ValueError(f"Invalid label: {new_label}. Must be positive/negative/neutral")
        
        # Get current data
        old_row = self.lexicon_df[self.lexicon_df['word'] == word].iloc[0]
        old_score = float(old_row['sentiment_score'])
        
        word_info = self.word_data.get(word, {})
        old_label = word_info.get('sentiment', 'unknown')
        
        # Update the label in word_data
        word_info['sentiment'] = new_label
        self.word_data[word] = word_info
        
        # Create a temporary row for recalculation
        temp_row = {
            'word_clean': word,
            'sentiment': new_label,
            'is_climate': word_info.get('is_climate', False),
            'dialect': word_info.get('dialect', 'filipino'),
            'definition': word_info.get('definition', '')
        }
        
        # Check if processor is available (use provided or stored)
        if processor is None:
            processor = self.processor
        
        # Recalculate score - create processor if needed
        if processor is None:
            # No processor available - create a minimal one just for FastText
            print("⚠️ No processor available, initializing FastText for recalculation...")
            from services.fasttext_service import get_fasttext_manager
            fasttext_manager = get_fasttext_manager()
            
            # Create a minimal processor-like object
            class MinimalProcessor:
                def __init__(self, ft_manager):
                    self.fasttext_manager = ft_manager
            
            processor = MinimalProcessor(fasttext_manager)
        
        # Recalculate score using processor's logic
        new_score = self._recalculate_score(temp_row, processor)
        
        # Try to get breakdown if processor supports it
        breakdown = None
        if hasattr(processor, 'get_word_breakdown'):
            breakdown = processor.get_word_breakdown(word)
        
        # Update lexicon DataFrame
        self.lexicon_df.loc[self.lexicon_df['word'] == word, 'sentiment_score'] = new_score
        
        # Update metadata
        self.metadata['last_updated'] = datetime.now().isoformat()
        if 'manual_updates' not in self.metadata:
            self.metadata['manual_updates'] = []
        
        self.metadata['manual_updates'].append({
            'word': word,
            'old_label': old_label,
            'new_label': new_label,
            'old_score': round(old_score, 4),
            'new_score': round(new_score, 4),
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"\n✏️ Updated '{word}':")
        print(f"   Label: {old_label} → {new_label}")
        print(f"   Score: {old_score:.4f} → {new_score:.4f}")
        
        return {
            "word": word,
            "old_label": old_label,
            "new_label": new_label,
            "old_score": round(old_score, 4),
            "new_score": round(new_score, 4),
            "breakdown": breakdown,
            "message": f"Successfully updated '{word}' from {old_label} to {new_label}"
        }
    
    def _recalculate_score(self, row: Dict, processor) -> float:
        """
        Recalculate score for a word using the 3-stage pipeline
        This mirrors the logic in LexicalProcessor.calculate_sentiment_scores
        Works even without a full LexicalProcessor instance
        """
        sentiment = row['sentiment'].lower()
        is_climate = row['is_climate']
        word = row['word_clean']
        dialect = row['dialect'].lower()
        
        # STAGE 1: Base Polarity
        if 'positive' in sentiment:
            polarity = +1
            base_magnitude = 2.5
        elif 'negative' in sentiment:
            polarity = -1
            base_magnitude = 2.5
        else:
            # Neutral
            return 0.0
        
        # STAGE 2: Semantic Intensity
        intensity = 1.0
        
        # Get FastText model
        if hasattr(processor, 'fasttext_manager'):
            model = processor.fasttext_manager.get_model(dialect)
            
            if model and word in model:
                # If we have a full processor, use its method
                if hasattr(processor, 'calculate_semantic_intensity'):
                    intensity, _ = processor.calculate_semantic_intensity(
                        word=word,
                        model=model,
                        dialect=dialect,
                        expected_polarity=polarity
                    )
                else:
                    # Minimal intensity calculation without full processor
                    intensity = self._simple_intensity_calculation(word, model, polarity)
        
        # STAGE 3: Domain Weighting
        domain_weight = 1.3 if is_climate else 1.0
        
        # Final calculation
        final_score = polarity * base_magnitude * intensity * domain_weight
        
        return final_score
    
    def _simple_intensity_calculation(self, word: str, model, expected_polarity: int) -> float:
        """
        Simplified intensity calculation when full processor is not available
        Returns a value between 0.6 and 1.4
        """
        try:
            import numpy as np
            
            if word not in model:
                return 1.0
            
            word_vector = model[word]
            
            # Simple heuristic: use vector magnitude as proxy for intensity
            # Words with stronger semantic content have larger magnitudes
            magnitude = np.linalg.norm(word_vector)
            
            # Normalize to 0.6-1.4 range
            # Typical FastText magnitudes are around 5-15
            normalized = (magnitude - 5) / 10  # maps 5-15 to 0-1
            normalized = max(0, min(1, normalized))  # clamp to 0-1
            
            intensity = 0.6 + (normalized * 0.8)  # map to 0.6-1.4
            
            return intensity
            
        except Exception as e:
            print(f"Error in intensity calculation: {e}")
            return 1.0
    
    def save_changes(self) -> bool:
        """
        Persist in-memory changes to disk
        Call this after updating word labels
        """
        if self.lexicon_df is None or self.word_data is None:
            raise ValueError("No dictionary loaded")
        
        return self.save(
            lexicon_df=self.lexicon_df,
            word_data=self.word_data,
            processor=self.processor,
            stats=self.metadata.get('stats', {})
        )
    
    def reset(self) -> bool:
        """Delete cached dictionary (forces reprocessing)"""
        try:
            if self.joblib_path.exists():
                self.joblib_path.unlink()
                print(f"Deleted {self.joblib_path}")
            
            if self.csv_path.exists():
                self.csv_path.unlink()
                print(f"Deleted {self.csv_path}")
            
            if self.metadata_path.exists():
                self.metadata_path.unlink()
                print(f"Deleted {self.metadata_path}")
            
            # Clear memory
            self.lexicon_df = None
            self.word_data = None
            self.processor = None
            self.metadata = {}
            
            print("✅ Cache reset successfully")
            return True
            
        except Exception as e:
            print(f"Error resetting cache: {e}")
            return False
    
    def export_csv(self, output_path: Optional[str] = None) -> str:
        """Export current dictionary as CSV"""
        if self.lexicon_df is None:
            raise ValueError("No dictionary loaded")
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"lexical_dictionary_export_{timestamp}.csv"
        
        self.lexicon_df.to_csv(output_path, index=False)
        print(f"✅ Exported to {output_path}")
        
        return output_path
    
    def get_word_info(self, word: str) -> Optional[Dict[str, Any]]:
        """Get full information about a word"""
        word = word.lower().strip()
        
        if self.lexicon_df is None:
            return None
        
        if word not in self.lexicon_df['word'].values:
            return None
        
        score_row = self.lexicon_df[self.lexicon_df['word'] == word].iloc[0]
        word_info = self.word_data.get(word, {})
        
        return {
            "word": word,
            "sentiment_score": float(score_row['sentiment_score']),
            "sentiment_label": word_info.get('sentiment', 'unknown'),
            "is_climate": word_info.get('is_climate', False),
            "dialect": word_info.get('dialect', 'unknown'),
            "definition": word_info.get('definition', '')
        }


# Global instance
_dictionary_manager = None


def get_dictionary_manager() -> LexicalDictionaryManager:
    """Get global dictionary manager instance"""
    global _dictionary_manager
    if _dictionary_manager is None:
        _dictionary_manager = LexicalDictionaryManager()
    return _dictionary_manager