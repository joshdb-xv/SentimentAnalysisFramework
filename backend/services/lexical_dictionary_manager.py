# services/lexical_dictionary_manager.py

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
        NOW PROPERLY HANDLES NEUTRAL WORDS with semantic analysis
        """
        sentiment = row['sentiment'].lower()
        is_climate = row['is_climate']
        word = row['word_clean']
        dialect = row['dialect'].lower()
        
        # Normalize dialect
        if 'cebuano' in dialect or 'bisaya' in dialect:
            dialect = 'cebuano'
        elif 'tagalog' in dialect:
            dialect = 'tagalog'
        else:
            dialect = 'filipino'
        
        # STAGE 1: Base Polarity
        if 'positive' in sentiment:
            polarity = +1
            base_magnitude = 2.5
            is_neutral = False
        elif 'negative' in sentiment:
            polarity = -1
            base_magnitude = 2.5
            is_neutral = False
        else:
            # NEUTRAL - needs semantic analysis
            polarity = 0
            base_magnitude = 0.3  # Much smaller than pos/neg
            is_neutral = True
        
        # STAGE 2: Semantic Intensity
        intensity = 1.0
        semantic_polarity = polarity
        
        # Get FastText model
        model = None
        if hasattr(processor, 'fasttext_manager'):
            model = processor.fasttext_manager.get_model(dialect)
        
        if model and word in model:
            if is_neutral:
                # For neutral words, determine polarity from embeddings
                try:
                    import numpy as np
                    
                    word_vector = model[word]
                    
                    # Get prototypes
                    pos_prototype = self._get_sentiment_prototype(model, dialect, 'positive', processor)
                    neg_prototype = self._get_sentiment_prototype(model, dialect, 'negative', processor)
                    
                    if pos_prototype is not None and neg_prototype is not None:
                        # Calculate similarities
                        sim_positive = self._cosine_similarity(word_vector, pos_prototype)
                        sim_negative = self._cosine_similarity(word_vector, neg_prototype)
                        
                        print(f"  Neutral word '{word}' similarities: pos={sim_positive:.4f}, neg={sim_negative:.4f}")
                        
                        # Check if truly neutral (difference < 0.05)
                        diff = abs(sim_positive - sim_negative)
                        if diff < 0.05:
                            # Truly neutral - return 0
                            print(f"  → Truly neutral (diff={diff:.4f} < 0.05), returning 0.0")
                            return 0.0
                        else:
                            # Has slight leaning
                            if sim_positive > sim_negative:
                                semantic_polarity = +1
                                print(f"  → Slight positive leaning (diff={diff:.4f})")
                            else:
                                semantic_polarity = -1
                                print(f"  → Slight negative leaning (diff={diff:.4f})")
                            
                            # Calculate intensity from difference
                            raw_intensity = diff
                            intensity = min(raw_intensity * 2, 1.0)
                            print(f"  → Intensity: {intensity:.4f}")
                    else:
                        # Can't determine - return 0
                        print(f"  → Prototypes unavailable, returning 0.0")
                        return 0.0
                except Exception as e:
                    print(f"  → Error in semantic analysis: {e}, returning 0.0")
                    return 0.0
            else:
                # For positive/negative words, use original intensity calculation
                if hasattr(processor, 'calculate_semantic_intensity'):
                    intensity, _ = processor.calculate_semantic_intensity(
                        word=word,
                        model=model,
                        dialect=dialect,
                        expected_polarity=polarity
                    )
                else:
                    # Fallback to simple calculation
                    intensity = self._simple_intensity_calculation(word, model, polarity)
        elif is_neutral:
            # No model available for neutral word - return 0
            print(f"  → Word '{word}' not in model, returning 0.0")
            return 0.0
        
        # STAGE 3: Domain Weighting
        domain_weight = 1.3 if is_climate else 1.0
        
        # Final calculation
        if is_neutral:
            final_score = semantic_polarity * base_magnitude * intensity * domain_weight
            print(f"  → Final: {semantic_polarity} × {base_magnitude} × {intensity:.4f} × {domain_weight} = {final_score:.4f}")
        else:
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
    
    def _get_sentiment_prototype(self, model, dialect: str, polarity: str, processor):
        """Get sentiment prototype vector"""
        # Try to use processor's method if available
        if hasattr(processor, 'get_sentiment_prototype'):
            return processor.get_sentiment_prototype(model, dialect, polarity)
        
        # Fallback: create prototype from anchor words
        sentiment_anchors = {
                'cebuano': {
                  'positive': [
                      # Quality & Beauty
                      'maayo', 'nindot', 'matahum', 'gwapa', 'gwapo', 'chada',
                      # Happiness & Joy
                      'malipayon', 'lipay', 'kalipay', 'happy', 'malipayong',
                      # Comfort & Peace
                      'hayahay', 'maayong', 'malinawon', 'kalinaw', 'tarong',
                      # Strength & Power
                      'kusog', 'lig-on', 'dako', 'daghan', 'baskog',
                      # Success & Achievement
                      'swerte', 'sakto', 'perpekto', 'daog', 'kadaogan',
                      # Safety & Security
                      'luwas', 'seguro', 'protektado', 'saligan',
                      # Health & Wellness
                      'himsog', 'lawas', 'presko', 'buhi',
                      # Climate Positive (good weather, recovery)
                      'uwan', 'bugnaw', 'lunhaw', 'sariwa', 'lamig'
                  ],
                  'negative': [
                      # Bad Quality
                      'dili', 'dautan', 'daotan', 'grabe', 'grabeng',
                      # Difficulty & Hardship
                      'lisud', 'kabudlay', 'malisud', 'bug-at', 'kalisud',
                      # Danger & Risk
                      'makuyaw', 'katalagman', 'peligro', 'delikado',
                      # Pain & Suffering
                      'sakit', 'kasakit', 'masakit', 'kapit-os', 'kalisang',
                      # Sadness & Sorrow
                      'kasubo', 'masulub-on', 'lungkot', 'kasubo',
                      # Fear & Worry
                      'kahadlok', 'mahadlok', 'kuyaw', 'kakuyaw', 'balaka',
                      # Destruction & Damage
                      'guba', 'dagdag', 'laglag', 'gadaot', 'dungag',
                      # Weakness & Frailty
                      'luya', 'maluya', 'kahuyang', 'kakapoy', 'maskin',
                      # CLIMATE-SPECIFIC NEGATIVE (disasters, extreme weather, discomfort)
                      'init', 'kainit', 'alinsangan', 'uga', 'hubag',
                      'baha', 'bagyo', 'unos', 'linog', 'lindol',
                      'tuyo', 'haw-ang', 'uhaw', 'tigang', 'guwang',
                      'taas', 'lunod', 'hangin', 'duling', 'kusog'
                  ]
              },
              'tagalog': {
                  'positive': [
                      # Quality & Beauty
                      'mabuti', 'maganda', 'ganda', 'magandang', 'marilag',
                      # Happiness & Joy
                      'masaya', 'saya', 'tuwa', 'ligaya', 'galak',
                      # Comfort & Peace
                      'komportable', 'tahimik', 'mapayapa', 'payapa', 'kapanatagan',
                      # Strength & Power
                      'malakas', 'lakas', 'malaki', 'dakila', 'matibay',
                      # Success & Achievement
                      'mahusay', 'tagumpay', 'swerte', 'perpekto', 'sakto',
                      # Safety & Security
                      'ligtas', 'secure', 'protektado', 'sigurado',
                      # Health & Wellness
                      'malusog', 'kalusugan', 'buhay', 'sariwang',
                      # Love & Care
                      'mahal', 'pagmamahal', 'pag-ibig', 'mabait', 'napakaganda',
                      # Climate Positive
                      'ulan', 'malamig', 'luntian', 'sariwa', 'presko'
                  ],
                  'negative': [
                      # Bad Quality
                      'masama', 'sama', 'pangit', 'nakakainis', 'basura',
                      # Difficulty & Hardship
                      'mahirap', 'hirap', 'kahirapan', 'pagod', 'napakahirap',
                      # Danger & Risk
                      'mapanganib', 'panganib', 'delikado', 'peligro', 'banta',
                      # Pain & Suffering
                      'sakit', 'masakit', 'kirot', 'hapdi', 'hirap',
                      # Sadness & Sorrow
                      'malungkot', 'lungkot', 'kalungkutan', 'pighati', 'dalamhati',
                      # Fear & Worry
                      'takot', 'natatakot', 'sindak', 'pangamba', 'alarma',
                      # Destruction & Damage
                      'sira', 'wasak', 'giba', 'pinsala', 'kapinsalaan',
                      # Weakness & Frailty
                      'mahina', 'hina', 'kahinaan', 'kapaguran',
                      # Death & Loss
                      'patay', 'kamatayan', 'nawala', 'pagkalugi', 'malala',
                      # CLIMATE-SPECIFIC NEGATIVE
                      'init', 'mainit', 'sobrang-init', 'alinsangan', 'tuyot',
                      'baha', 'bagyo', 'unos', 'lindol', 'pagguho',
                      'tagtuyot', 'uhaw', 'tuyo', 'tigang', 'lubog',
                      'mataas', 'grabe', 'matindi', 'extreme', 'nakakapaso'
                  ]
              },
              'filipino': {
                  'positive': [
                      # Quality & Beauty
                      'mabuti', 'maganda', 'ganda', 'magandang', 'marilag',
                      # Happiness & Joy
                      'masaya', 'saya', 'tuwa', 'ligaya', 'galak',
                      # Comfort & Peace
                      'komportable', 'tahimik', 'mapayapa', 'payapa', 'kapanatagan',
                      # Strength & Power
                      'malakas', 'lakas', 'malaki', 'dakila', 'matibay',
                      # Success & Achievement
                      'mahusay', 'tagumpay', 'swerte', 'perpekto', 'sakto',
                      # Safety & Security
                      'ligtas', 'secure', 'protektado', 'sigurado',
                      # Health & Wellness
                      'malusog', 'kalusugan', 'buhay', 'sariwang',
                      # Love & Care
                      'mahal', 'pagmamahal', 'pag-ibig', 'mabait', 'napakaganda',
                      # Climate Positive
                      'ulan', 'malamig', 'luntian', 'sariwa', 'presko'
                  ],
                  'negative': [
                      # Bad Quality
                      'masama', 'sama', 'pangit', 'nakakainis', 'basura',
                      # Difficulty & Hardship
                      'mahirap', 'hirap', 'kahirapan', 'pagod', 'napakahirap',
                      # Danger & Risk
                      'mapanganib', 'panganib', 'delikado', 'peligro', 'banta',
                      # Pain & Suffering
                      'sakit', 'masakit', 'kirot', 'hapdi', 'hirap',
                      # Sadness & Sorrow
                      'malungkot', 'lungkot', 'kalungkutan', 'pighati', 'dalamhati',
                      # Fear & Worry
                      'takot', 'natatakot', 'sindak', 'pangamba', 'alarma',
                      # Destruction & Damage
                      'sira', 'wasak', 'giba', 'pinsala', 'kapinsalaan',
                      # Weakness & Frailty
                      'mahina', 'hina', 'kahinaan', 'kapaguran',
                      # Death & Loss
                      'patay', 'kamatayan', 'nawala', 'pagkalugi', 'malala',
                      # CLIMATE-SPECIFIC NEGATIVE
                      'init', 'mainit', 'sobrang-init', 'alinsangan', 'tuyot',
                      'baha', 'bagyo', 'unos', 'lindol', 'pagguho',
                      'tagtuyot', 'uhaw', 'tuyo', 'tigang', 'lubog',
                      'mataas', 'grabe', 'matindi', 'extreme', 'nakakapaso'
                  ]
              }
          }
        
        import numpy as np
        
        dialect_key = dialect if dialect in sentiment_anchors else 'filipino'
        anchors = sentiment_anchors[dialect_key].get(polarity, [])
        
        vectors = []
        for anchor in anchors:
            if anchor in model:
                vectors.append(model[anchor])
        
        if not vectors:
            return None
        
        return np.mean(vectors, axis=0)

    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity"""
        if vec1 is None or vec2 is None:
            return 0.0
        
        import numpy as np
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

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