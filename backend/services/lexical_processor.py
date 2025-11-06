import pandas as pd
import numpy as np
from typing import Optional, Callable
import re, os


class LexicalProcessor:
    """
    Processes the Filipino/Cebuano dictionary and generates VADER-compatible lexicon
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
                    # If multiple climate definitions, keep the first one
                    processed_rows.append(climate_related.iloc[0])
                else:
                    # No climate definitions, keep the first occurrence
                    # (Could also do majority vote on sentiment here)
                    processed_rows.append(group.iloc[0])
        
        self.df = pd.DataFrame(processed_rows).reset_index(drop=True)
        print(f"After deduplication: {len(self.df)} unique words")
    
    def calculate_sentiment_scores(self):
        """
        Calculate sentiment scores for VADER using FastText embeddings
        Combines existing labels with semantic similarity
        """
        self._update_progress(50, "Calculating sentiment scores with FastText...")
        
        # Use the sentiment column directly
        sentiment_col = 'sentiment'
        
        # Check if dialect column exists
        dialect_col = 'dialect' if 'dialect' in self.df.columns else None
        
        def score_word(row):
            sentiment = str(row[sentiment_col]).lower()
            is_climate = row['is_climate']
            word = row['word_clean']
            
            # Get dialect/language
            dialect = None
            if dialect_col:
                dialect = str(row[dialect_col]).lower()
            
            # Base score from sentiment label
            if 'positive' in sentiment:
                base_score = 3.0 if is_climate else 2.0
            elif 'negative' in sentiment:
                base_score = -3.0 if is_climate else -2.0
            else:
                base_score = 0.0
            
            # Get FastText model for this dialect
            if dialect and base_score != 0.0:
                model = self.fasttext_manager.get_model(dialect)
                if model and word in model:
                    # Calculate intensity based on embedding similarity
                    intensity = self.calculate_fasttext_intensity(word, model, sentiment)
                    # Adjust base score by intensity
                    base_score *= intensity
            
            return base_score
        
        self.df['sentiment_score'] = self.df.apply(score_word, axis=1)
    
    def calculate_fasttext_intensity(self, word: str, model, sentiment: str) -> float:
        """
        Calculate intensity multiplier based on FastText similarity
        Returns value between 0.5 and 1.5
        """
        try:
            if word not in model:
                return 1.0  # No adjustment
            
            # Define anchor words for different sentiments
            positive_anchors = ['good', 'great', 'excellent', 'positive', 'sustainable', 'protect']
            negative_anchors = ['bad', 'terrible', 'negative', 'crisis', 'disaster', 'destruction']
            
            # Get similarities
            pos_similarities = []
            neg_similarities = []
            
            for anchor in positive_anchors:
                if anchor in model:
                    pos_similarities.append(model.similarity(word, anchor))
            
            for anchor in negative_anchors:
                if anchor in model:
                    neg_similarities.append(model.similarity(word, anchor))
            
            avg_pos = np.mean(pos_similarities) if pos_similarities else 0.0
            avg_neg = np.mean(neg_similarities) if neg_similarities else 0.0
            
            # Calculate intensity based on sentiment direction
            if 'positive' in sentiment:
                # Higher similarity to positive anchors = stronger positive
                intensity = 1.0 + (avg_pos * 0.5)  # 1.0 to 1.5
            elif 'negative' in sentiment:
                # Higher similarity to negative anchors = stronger negative
                intensity = 1.0 + (avg_neg * 0.5)  # 1.0 to 1.5
            else:
                intensity = 1.0
            
            return max(0.5, min(1.5, intensity))  # Clamp between 0.5 and 1.5
            
        except Exception as e:
            print(f"Error calculating FastText intensity: {str(e)}")
            return 1.0
    
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
    
    def get_statistics(self):
        """
        Get statistics about the processed lexicon
        """
        # Ensure all boolean operations return Python bools, not numpy bools
        is_climate_sum = int(self.df['is_climate'].sum())
        total_words = len(self.df)
        non_climate = total_words - is_climate_sum
        
        positive_count = int((self.df['sentiment_score'] > 0).sum())
        negative_count = int((self.df['sentiment_score'] < 0).sum())
        neutral_count = int((self.df['sentiment_score'] == 0).sum())
        
        # Calculate averages safely
        climate_mask = self.df['is_climate']
        non_climate_mask = ~self.df['is_climate']
        
        avg_climate = float(self.df[climate_mask]['sentiment_score'].mean()) if climate_mask.any() else 0.0
        avg_general = float(self.df[non_climate_mask]['sentiment_score'].mean()) if non_climate_mask.any() else 0.0
        
        stats = {
            "total_words": total_words,
            "climate_related": is_climate_sum,
            "non_climate": non_climate,
            "positive_words": positive_count,
            "negative_words": negative_count,
            "neutral_words": neutral_count,
            "avg_score_climate": round(avg_climate, 3),
            "avg_score_general": round(avg_general, 3),
        }
        
        # Distribution by dialect if available
        if 'dialect' in self.df.columns:
            dialect_dist = self.df['dialect'].value_counts().to_dict()
            # Convert numpy int64 to Python int
            stats['by_dialect'] = {k: int(v) for k, v in dialect_dist.items()}
        
        print(f"Statistics generated: {stats}")
        return stats
    
    def process(self) -> dict:
        """
        Main processing pipeline
        Returns: dict with 'dataframe' and 'stats'
        """
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Handle duplicates
            self.handle_duplicates()
            
            # Step 3: Calculate sentiment scores (always uses FastText)
            self.calculate_sentiment_scores()
            
            # Step 4: Generate VADER format
            self._update_progress(90, "Finalizing...")
            vader_df = self.generate_vader_format()
            
            # Step 5: Get statistics
            stats = self.get_statistics()
            
            self._update_progress(100, "Processing completed!")
            
            return {
                "dataframe": vader_df,
                "stats": stats
            }
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            import traceback
            traceback.print_exc()
            raise