import pandas as pd
import numpy as np
from typing import Optional, Callable, Dict, Any
import re, os


class LexicalProcessor:
    """
    Processes the Filipino/Cebuano dictionary and generates VADER-compatible lexicon
    Using thesis-quality hybrid approach with semantic validation
    """
    
    def __init__(self, dictionary_path: str, climate_keywords_path: str):
        self.dictionary_path = dictionary_path
        self.climate_keywords_path = climate_keywords_path
        self.progress_callback: Optional[Callable] = None
        
        # Load data
        self.df = None
        self.climate_keywords = None
        
        # Get pre-loaded FastText models from manager
        from services.fasttext_service import get_fasttext_manager
        self.fasttext_manager = get_fasttext_manager()
        
        if not self.fasttext_manager.is_loaded():
            raise RuntimeError("FastText models not loaded! Check server startup logs.")
        
        # Define sentiment anchor words (documented in thesis!)
        self.sentiment_anchors = {
            'cebuano': {
                'positive': ['maayo', 'nindot', 'matahum', 'dako', 'kusog', 
                           'maayong', 'hayahay', 'malipayon'],
                'negative': ['dili', 'dautan', 'grabe', 'lisud', 'makuyaw', 
                           'sakit', 'maskin', 'katalagman']
            },
            'tagalog': {
                'positive': ['mabuti', 'maganda', 'masaya', 'mahusay', 'malaki', 
                           'malakas', 'perpekto', 'napakaganda'],
                'negative': ['masama', 'pangit', 'malungkot', 'mahirap', 
                           'mapanganib', 'sakit', 'delikado', 'malala']
            },
            'filipino': {
                'positive': ['mabuti', 'maganda', 'masaya', 'mahusay', 'malaki', 
                           'malakas', 'perpekto', 'napakaganda'],
                'negative': ['masama', 'pangit', 'malungkot', 'mahirap', 
                           'mapanganib', 'sakit', 'delikado', 'malala']
            }
        }
        
        # Cache for prototypes to avoid recalculation
        self.prototype_cache = {}
        
        # NEW: Store detailed breakdowns for each word
        self.word_breakdowns = {}
        
    def set_progress_callback(self, callback: Callable):
        """Set callback function for progress updates"""
        self.progress_callback = callback
        
    def _update_progress(self, progress: int, message: str):
        """Update progress if callback is set"""
        if self.progress_callback:
            self.progress_callback(progress, message)
    
    def load_data(self):
        """Load dictionary and climate keywords"""
        self._update_progress(10, "Loading dictionary...")
        self.df = pd.read_excel(self.dictionary_path)
        
        print(f"Excel columns: {self.df.columns.tolist()}")
        print(f"First few rows:\n{self.df.head()}")
        
        self._update_progress(20, "Loading climate keywords...")
        self.climate_keywords = pd.read_csv(self.climate_keywords_path)
        
        # Convert keywords to lowercase for matching
        self.climate_keywords['keyword'] = self.climate_keywords['keyword'].str.lower()
        
        print(f"Loaded {len(self.df)} words and {len(self.climate_keywords)} climate keywords")
    
    def clean_word(self, word: str) -> str:
        """
        Clean and simplify word forms
        Prioritize simpler constructs (remove hyphens, keep root forms)
        """
        if pd.isna(word):
            return ""
        
        word = str(word).strip().lower()
        
        # Remove hyphens for simpler form
        word = word.replace('-', '')
        
        return word
    
    def is_climate_related(self, definition: str) -> bool:
        """
        Check if a word's definition is climate-related
        """
        if pd.isna(definition):
            return False
        
        definition_lower = str(definition).lower()
        
        # Check if any climate keyword appears in the definition
        for keyword in self.climate_keywords['keyword']:
            if keyword in definition_lower:
                return True
        
        return False
    
    def handle_duplicates(self):
        """
        Handle duplicate words:
        1. Prioritize climate-related definitions
        2. Keep simpler word forms (without hyphens)
        3. For remaining duplicates, take majority sentiment or average
        """
        self._update_progress(30, "Handling duplicates...")
        
        # Use column names directly
        self.df['word_clean'] = self.df['word'].apply(self.clean_word)
        self.df['is_climate'] = self.df['definition'].apply(self.is_climate_related)
        
        # Group by cleaned word
        grouped = self.df.groupby('word_clean')
        
        processed_rows = []
        
        for word, group in grouped:
            if len(group) == 1:
                # No duplicates, keep as is
                processed_rows.append(group.iloc[0])
            else:
                # Duplicates exist
                climate_related = group[group['is_climate'] == True]
                
                if len(climate_related) > 0:
                    # Prioritize climate-related definitions
                    processed_rows.append(climate_related.iloc[0])
                else:
                    # No climate definitions, keep the first occurrence
                    processed_rows.append(group.iloc[0])
        
        self.df = pd.DataFrame(processed_rows).reset_index(drop=True)
        print(f"After deduplication: {len(self.df)} unique words")
    
    def get_sentiment_prototype(self, model, dialect: str, polarity: str):
        """
        Create prototype vectors by averaging high-quality anchor words
        These anchors are documented in thesis methodology!
        
        Returns: numpy array of prototype vector
        """
        # Check cache first
        cache_key = f"{dialect}_{polarity}"
        if cache_key in self.prototype_cache:
            return self.prototype_cache[cache_key]
        
        # Normalize dialect name
        dialect_key = dialect.lower()
        if 'cebuano' in dialect_key or 'bisaya' in dialect_key:
            dialect_key = 'cebuano'
        elif 'tagalog' in dialect_key:
            dialect_key = 'tagalog'
        else:
            dialect_key = 'filipino'  # default
        
        # Get anchor words
        if dialect_key not in self.sentiment_anchors:
            dialect_key = 'filipino'  # fallback
        
        anchors = self.sentiment_anchors[dialect_key].get(polarity, [])
        
        # Get vectors for available anchors
        vectors = []
        for anchor in anchors:
            if anchor in model:
                vectors.append(model[anchor])
        
        if not vectors:
            print(f"Warning: No anchor words found for {dialect_key} {polarity}")
            return None
        
        # Average the vectors to create prototype
        prototype = np.mean(vectors, axis=0)
        
        # Cache it
        self.prototype_cache[cache_key] = prototype
        
        return prototype
    
    def cosine_similarity(self, vec1, vec2):
        """
        Calculate cosine similarity between two vectors
        Returns: float between -1 and 1 (typically 0 to 1 for similar words)
        """
        if vec1 is None or vec2 is None:
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def calculate_semantic_intensity(self, word: str, model, dialect: str, expected_polarity: int):
        """
        Use embedding space to determine if word is semantically "strong" or "weak"
        
        Args:
            word: The word to analyze
            model: FastText model
            dialect: Language/dialect of the word
            expected_polarity: +1 for positive, -1 for negative
            
        Returns:
            tuple: (intensity, breakdown_dict)
        """
        breakdown = {
            "method": "semantic_embedding_analysis",
            "word_in_model": False,
            "similarity_to_positive": 0.0,
            "similarity_to_negative": 0.0,
            "semantic_strength": 0.0,
            "semantic_clarity": 0.0,
            "raw_intensity": 0.0,
            "notes": []
        }
        
        try:
            if word not in model:
                breakdown["notes"].append("Word not found in FastText model")
                return 1.0, breakdown
            
            breakdown["word_in_model"] = True
            word_vector = model[word]
            
            # Get sentiment prototypes
            pos_prototype = self.get_sentiment_prototype(model, dialect, 'positive')
            neg_prototype = self.get_sentiment_prototype(model, dialect, 'negative')
            
            if pos_prototype is None or neg_prototype is None:
                breakdown["notes"].append("Sentiment prototypes unavailable")
                return 1.0, breakdown
            
            # Calculate cosine similarities
            sim_positive = self.cosine_similarity(word_vector, pos_prototype)
            sim_negative = self.cosine_similarity(word_vector, neg_prototype)
            
            breakdown["similarity_to_positive"] = round(sim_positive, 4)
            breakdown["similarity_to_negative"] = round(sim_negative, 4)
            
            # Key insight: words closer to their expected pole are STRONGER
            if expected_polarity > 0:
                # For positive words
                semantic_strength = sim_positive
                semantic_clarity = sim_positive - sim_negative
                breakdown["notes"].append(f"Positive word: using similarity to positive prototype")
            else:
                # For negative words
                semantic_strength = sim_negative
                semantic_clarity = sim_negative - sim_positive
                breakdown["notes"].append(f"Negative word: using similarity to negative prototype")
            
            # Ensure clarity is non-negative
            semantic_clarity = max(0, semantic_clarity)
            
            breakdown["semantic_strength"] = round(semantic_strength, 4)
            breakdown["semantic_clarity"] = round(semantic_clarity, 4)
            
            # Combine: strong + clear = high intensity
            raw_intensity = (semantic_strength + semantic_clarity) / 2
            breakdown["raw_intensity"] = round(raw_intensity, 4)
            
            # Normalize to 0.6-1.4 range
            intensity = 0.6 + (raw_intensity * 0.8)
            intensity = max(0.6, min(1.4, intensity))
            
            breakdown["notes"].append(f"Normalized to range [0.6, 1.4]")
            
            return intensity, breakdown
            
        except Exception as e:
            breakdown["notes"].append(f"Error: {str(e)}")
            return 1.0, breakdown
    
    def calculate_sentiment_scores(self):
        """
        THREE-STAGE HYBRID APPROACH (Thesis Quality):
        
        Stage 1: Base Polarity (from manual labels)
        Stage 2: Semantic Intensity Refinement (from FastText embeddings)
        Stage 3: Domain-Specific Weighting (climate vs general)
        
        Final Formula:
        score = polarity × base_magnitude × intensity × domain_weight
        """
        self._update_progress(50, "Calculating sentiment scores (3-stage hybrid)...")
        
        sentiment_col = 'sentiment'
        dialect_col = 'dialect' if 'dialect' in self.df.columns else None
        
        def score_word(row):
            sentiment = str(row[sentiment_col]).lower()
            is_climate = row['is_climate']
            word = row['word_clean']
            
            # Get dialect/language
            dialect = 'filipino'  # default
            if dialect_col:
                dialect = str(row[dialect_col]).lower()
            
            # Create detailed breakdown
            breakdown = {
                "word": word,
                "original_sentiment_label": sentiment,
                "dialect": dialect,
                "is_climate_related": is_climate,
                "stages": {}
            }
            
            # ========================================
            # STAGE 1: BASE POLARITY (Manual Labels)
            # ========================================
            if 'positive' in sentiment:
                polarity = +1
                base_magnitude = 2.5
                polarity_label = "positive"
            elif 'negative' in sentiment:
                polarity = -1
                base_magnitude = 2.5
                polarity_label = "negative"
            else:
                # Neutral - store breakdown but return 0
                breakdown["stages"]["stage_1_base"] = {
                    "polarity": 0,
                    "base_magnitude": 0,
                    "explanation": "Word labeled as neutral - no sentiment score"
                }
                self.word_breakdowns[word] = breakdown
                return 0.0
            
            breakdown["stages"]["stage_1_base"] = {
                "polarity": polarity,
                "polarity_label": polarity_label,
                "base_magnitude": base_magnitude,
                "explanation": f"Base VADER magnitude for {polarity_label} words"
            }
            
            # ========================================
            # STAGE 2: SEMANTIC INTENSITY (Embeddings)
            # ========================================
            intensity = 1.0  # default
            intensity_breakdown = None
            
            model = self.fasttext_manager.get_model(dialect)
            if model and word in model:
                intensity, intensity_breakdown = self.calculate_semantic_intensity(
                    word=word,
                    model=model,
                    dialect=dialect,
                    expected_polarity=polarity
                )
            
            breakdown["stages"]["stage_2_intensity"] = {
                "intensity_multiplier": round(intensity, 4),
                "explanation": "Semantic intensity from word embeddings (range: 0.6-1.4)",
                "details": intensity_breakdown or {"note": "Word not in embedding model"}
            }
            
            # ========================================
            # STAGE 3: DOMAIN WEIGHTING (Climate-specific)
            # ========================================
            domain_weight = 1.3 if is_climate else 1.0
            
            breakdown["stages"]["stage_3_domain"] = {
                "domain_weight": domain_weight,
                "explanation": "Climate-related words get 1.3x boost, general words stay at 1.0",
                "reason": "Climate words more impactful in climate context" if is_climate else "General word"
            }
            
            # ========================================
            # FINAL SCORE CALCULATION
            # ========================================
            final_score = polarity * base_magnitude * intensity * domain_weight
            
            breakdown["calculation"] = {
                "formula": "polarity × base_magnitude × intensity × domain_weight",
                "substituted": f"{polarity} × {base_magnitude} × {round(intensity, 4)} × {domain_weight}",
                "result": round(final_score, 4)
            }
            
            breakdown["final_score"] = round(final_score, 4)
            
            # Store the breakdown
            self.word_breakdowns[word] = breakdown
            
            return final_score
        
        self.df['sentiment_score'] = self.df.apply(score_word, axis=1)
        
        print("\nSample scored words:")
        sample = self.df[self.df['sentiment_score'] != 0].head(10)
        print(sample[['word_clean', 'sentiment', 'is_climate', 'sentiment_score']])
    
    def generate_vader_format(self):
        """
        Generate final VADER-compatible CSV format
        Format: word, sentiment_score
        """
        self._update_progress(80, "Generating VADER format...")
        
        # Select only necessary columns
        vader_df = self.df[['word_clean', 'sentiment_score']].copy()
        vader_df.columns = ['word', 'sentiment_score']
        
        # Remove any rows with empty words
        vader_df = vader_df[vader_df['word'] != '']
        
        # Sort by word for easier lookup
        vader_df = vader_df.sort_values('word').reset_index(drop=True)
        
        return vader_df
    
    def get_word_breakdown(self, word: str) -> Optional[Dict[str, Any]]:
        """Get detailed breakdown for a specific word"""
        word_clean = self.clean_word(word)
        return self.word_breakdowns.get(word_clean)
    
    def get_statistics(self):
        """
        Get statistics about the processed lexicon
        Enhanced with stage-by-stage breakdown
        """
        # Basic counts
        is_climate_sum = int(self.df['is_climate'].sum())
        total_words = len(self.df)
        non_climate = total_words - is_climate_sum
        
        positive_count = int((self.df['sentiment_score'] > 0).sum())
        negative_count = int((self.df['sentiment_score'] < 0).sum())
        neutral_count = int((self.df['sentiment_score'] == 0).sum())
        
        # Calculate averages
        climate_mask = self.df['is_climate']
        non_climate_mask = ~self.df['is_climate']
        
        avg_climate = float(self.df[climate_mask]['sentiment_score'].mean()) if climate_mask.any() else 0.0
        avg_general = float(self.df[non_climate_mask]['sentiment_score'].mean()) if non_climate_mask.any() else 0.0
        
        # Score distribution
        pos_climate = int((self.df[climate_mask]['sentiment_score'] > 0).sum()) if climate_mask.any() else 0
        neg_climate = int((self.df[climate_mask]['sentiment_score'] < 0).sum()) if climate_mask.any() else 0
        
        pos_general = int((self.df[non_climate_mask]['sentiment_score'] > 0).sum()) if non_climate_mask.any() else 0
        neg_general = int((self.df[non_climate_mask]['sentiment_score'] < 0).sum()) if non_climate_mask.any() else 0
        
        stats = {
            "total_words": total_words,
            "climate_related": is_climate_sum,
            "non_climate": non_climate,
            "positive_words": positive_count,
            "negative_words": negative_count,
            "neutral_words": neutral_count,
            "avg_score_climate": round(avg_climate, 3),
            "avg_score_general": round(avg_general, 3),
            "climate_breakdown": {
                "positive": pos_climate,
                "negative": neg_climate
            },
            "general_breakdown": {
                "positive": pos_general,
                "negative": neg_general
            },
            "score_range": {
                "min": float(self.df['sentiment_score'].min()),
                "max": float(self.df['sentiment_score'].max()),
                "mean": float(self.df['sentiment_score'].mean()),
                "std": float(self.df['sentiment_score'].std())
            }
        }
        
        # Distribution by dialect if available
        if 'dialect' in self.df.columns:
            dialect_dist = self.df['dialect'].value_counts().to_dict()
            stats['by_dialect'] = {k: int(v) for k, v in dialect_dist.items()}
        
        print(f"\nStatistics generated:")
        print(f"  Total words: {stats['total_words']}")
        print(f"  Climate words: {stats['climate_related']} ({pos_climate}+, {neg_climate}-)")
        print(f"  General words: {stats['non_climate']} ({pos_general}+, {neg_general}-)")
        print(f"  Avg score (climate): {stats['avg_score_climate']}")
        print(f"  Avg score (general): {stats['avg_score_general']}")
        print(f"  Score range: [{stats['score_range']['min']:.3f}, {stats['score_range']['max']:.3f}]")
        
        return stats
    
    def process(self) -> dict:
        """
        Main processing pipeline
        Returns: dict with 'dataframe', 'stats', and 'processor' (for breakdown access)
        """
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Handle duplicates
            self.handle_duplicates()
            
            # Step 3: Calculate sentiment scores (3-stage hybrid approach)
            self.calculate_sentiment_scores()
            
            # Step 4: Generate VADER format
            self._update_progress(90, "Finalizing...")
            vader_df = self.generate_vader_format()
            
            # Step 5: Get statistics
            stats = self.get_statistics()
            
            self._update_progress(100, "Processing completed!")
            
            return {
                "dataframe": vader_df,
                "stats": stats,
                "processor": self  # Return processor instance for breakdown access
            }
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            import traceback
            traceback.print_exc()
            raise