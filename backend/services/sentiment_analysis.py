# services/sentiment_analysis.py - WITHOUT AMBIGUOUS TERM CHECKER

import os
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import joblib

@dataclass
class SentimentScore:
    """Data class to hold sentiment analysis results with confidence"""
    positive: float
    negative: float
    neutral: float
    compound: float
    classification: str  # 'positive', 'negative', 'neutral', or 'inconclusive'
    confidence: float  # 0.0 to 1.0
    confidence_tier: str  # 'HIGH', 'MODERATE', 'LOW'
    reasoning: List[str]  # List of reasons for classification
    include_in_stats: bool  # Whether to include in quantitative analysis
    qualitative_category: str  # Category for qualitative analysis

class MultilingualVADER(SentimentIntensityAnalyzer):
    """
    Extended VADER sentiment analyzer with confidence-based classification
    Supports English, Tagalog, and Cebuano words
    
    NOTE: Ambiguous weather term checking removed - pipeline already handles
    climate relevance and domain classification upstream.
    """
    
    # Intensifiers that indicate strong opinion
    INTENSIFIERS = {
        'sobra', 'grabe', 'very', 'super', 'talaga', 
        'really', 'napaka', 'lubha', 'kaayo', 'gyud'
    }
    
    # Positive evaluation words
    POSITIVE_WORDS = {
        'sarap', 'ganda', 'astig', 'nice', 'good', 'perfect',
        'enjoy', 'love', 'great', 'awesome', 'wonderful',
        'nindot', 'maayo', 'lami'
    }
    
    # Negative evaluation words
    NEGATIVE_WORDS = {
        'ayaw', 'badtrip', 'hassle', 'hirap', 'sucks', 'hate',
        'worst', 'bad', 'terrible', 'awful', 'annoying',
        'di ko kinaya', 'di ko gusto', 'ayoko', 'dili'
    }
    
    def __init__(self, lexicon_path: str = None):
        super().__init__()
        
        if lexicon_path is None:
            possible_paths = [
                "data/lexical_dictionary/lexical_dictionary.joblib",
                "../data/lexical_dictionary/lexical_dictionary.joblib",
                "../../data/lexical_dictionary/lexical_dictionary.joblib",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    lexicon_path = path
                    break
            
            if lexicon_path is None:
                lexicon_path = "data/lexical_dictionary/lexical_dictionary.joblib"
        
        self.custom_lexicon_path = lexicon_path
        self.custom_lexicon = {}
        self.load_errors = []
        
        self._load_custom_lexicon()
        self._merge_lexicons()
        self._add_filipino_negations()
    
    def _load_custom_lexicon(self) -> None:
        """Load the custom lexical dictionary from joblib file"""
        try:
            if not os.path.exists(self.custom_lexicon_path):
                raise FileNotFoundError(f"Lexical dictionary not found at: {self.custom_lexicon_path}")
            
            print(f"📖 Loading lexicon from: {self.custom_lexicon_path}")
            
            loaded_data = joblib.load(self.custom_lexicon_path)
            
            if isinstance(loaded_data, dict):
                if 'lexicon' in loaded_data:
                    df = loaded_data['lexicon']
                    metadata = loaded_data.get('metadata', {})
                    print(f"✅ Loaded DataFrame with {len(df)} rows from cached dictionary")
                    if metadata:
                        created = metadata.get('created_at', 'unknown')
                        print(f"   Created: {created}")
                elif 'lexicon_df' in loaded_data:
                    df = loaded_data['lexicon_df']
                    metadata = loaded_data.get('metadata', {})
                    print(f"✅ Loaded DataFrame with {len(df)} rows from cached dictionary")
                else:
                    df = pd.DataFrame(loaded_data)
                    print(f"✅ Converted dict to DataFrame with {len(df)} rows")
            elif isinstance(loaded_data, pd.DataFrame):
                df = loaded_data
                print(f"✅ Loaded DataFrame with {len(df)} rows")
            else:
                raise ValueError(f"Unexpected data type in joblib file: {type(loaded_data)}")
            
            print(f"📋 Columns found: {list(df.columns)}")
            
            if 'sentiment_score' in df.columns and 'word' in df.columns:
                df = df.rename(columns={'sentiment_score': 'sentivalue'})
                print(f"📋 Detected lexical dictionary format, renamed 'sentiment_score' to 'sentivalue'")
            
            required_cols = ['word', 'sentivalue']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Dictionary must contain columns: {required_cols}. Found: {list(df.columns)}")
            
            loaded_count = 0
            skipped_count = 0
            
            for idx, row in df.iterrows():
                try:
                    word = str(row['word']).strip().lower()
                    if not word or word == 'nan':
                        skipped_count += 1
                        continue
                    
                    sentiment_value = float(row['sentivalue'])
                    
                    if pd.isna(sentiment_value):
                        skipped_count += 1
                        continue
                    
                    weight = float(row.get('weight', 1.0)) if 'weight' in row and not pd.isna(row.get('weight')) else 1.0
                    
                    adjusted_sentiment = sentiment_value * weight
                    adjusted_sentiment = max(-4.0, min(4.0, adjusted_sentiment))
                    
                    self.custom_lexicon[word] = adjusted_sentiment
                    loaded_count += 1
                    
                except (ValueError, TypeError, KeyError) as e:
                    self.load_errors.append(f"Row {idx}, word '{row.get('word', 'N/A')}': {e}")
                    skipped_count += 1
                    continue
            
            print(f"✅ Successfully loaded {loaded_count} words into custom lexicon")
            if skipped_count > 0:
                print(f"⚠️  Skipped {skipped_count} invalid entries")
            
            if self.load_errors and len(self.load_errors) <= 10:
                print(f"⚠️  Load errors (showing first 5):")
                for error in self.load_errors[:5]:
                    print(f"     {error}")
            
            if loaded_count > 0:
                sample_words = list(self.custom_lexicon.items())[:5]
                print(f"📝 Sample words:")
                for word, sentiment in sample_words:
                    print(f"     '{word}': {sentiment:+.3f}")
            
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print(f"   Make sure the lexicon file exists at: {self.custom_lexicon_path}")
            self.custom_lexicon = {}
        except Exception as e:
            print(f"❌ Error loading lexical dictionary: {e}")
            print(f"   File path: {self.custom_lexicon_path}")
            import traceback
            traceback.print_exc()
            self.custom_lexicon = {}
    
    def _merge_lexicons(self) -> None:
        """Merge custom lexicon with VADER's base lexicon"""
        if not self.custom_lexicon:
            print("⚠️ No custom lexicon to merge - using base VADER only")
            print("   This means Tagalog/Cebuano words will not be recognized!")
            return
        
        original_count = len(self.lexicon)
        
        overridden_count = 0
        for word, sentiment in self.custom_lexicon.items():
            if word in self.lexicon:
                overridden_count += 1
            self.lexicon[word] = sentiment
        
        new_count = len(self.lexicon)
        added = new_count - original_count
        
        print(f"✅ Merged lexicons:")
        print(f"   Base VADER words: {original_count}")
        print(f"   Custom words: {len(self.custom_lexicon)}")
        print(f"   New words added: {added}")
        print(f"   Words overridden: {overridden_count}")
        print(f"   Total words: {new_count}")
    
    def _add_filipino_negations(self) -> None:
        """Add Tagalog and Cebuano negation words to VADER's negation set"""
        filipino_negations = {
            "hindi", "walang", "wala", "di", "huwag", "ayaw",
            "hinde", "ayoko", "ayaw ko",
            "dili", "walay",
            "hndi", "wla", "d", "hwag"
        }
        
        for neg in filipino_negations:
            if neg not in self.lexicon:
                self.lexicon[neg] = 0.0
    
    def preprocess_text(self, text: str, debug: bool = False) -> str:
        """Preprocess text before sentiment analysis"""
        if not text:
            return ""
        
        original = text
        text = text.lower()
        
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if debug:
            print(f"🔍 Preprocessing:")
            print(f"   Original: '{original}'")
            print(f"   Processed: '{text}'")
        
        return text
    
    def _extract_context_features(self, text: str) -> Dict:
        """Extract contextual features from text"""
        words = text.lower().split()
        
        # Check for emojis
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        
        emojis = emoji_pattern.findall(text)
        
        # Categorize emojis
        positive_emojis = {'😊', '😄', '😃', '☕', '❤️', '👍', '✨', '🌟', '💖', '😍'}
        negative_emojis = {'😭', '😢', '😞', '😩', '🥶', '😰', '😱', '💔', '😤', '😡'}
        
        has_positive_emoji = any(e in positive_emojis for e in emojis)
        has_negative_emoji = any(e in negative_emojis for e in emojis)
        
        # Check for intensifiers
        has_intensifiers = any(word in self.INTENSIFIERS for word in words)
        intensifiers_found = [word for word in words if word in self.INTENSIFIERS]
        
        # Check for positive/negative evaluation words
        has_positive_words = any(word in self.POSITIVE_WORDS for word in words)
        has_negative_words = any(word in self.NEGATIVE_WORDS for word in words)
        
        positive_words_found = [word for word in words if word in self.POSITIVE_WORDS]
        negative_words_found = [word for word in words if word in self.NEGATIVE_WORDS]
        
        # Check for exclamation marks
        has_exclamation = '!' in text
        
        return {
            'word_count': len(words),
            'has_intensifiers': has_intensifiers,
            'intensifiers_found': intensifiers_found,
            'has_positive_words': has_positive_words,
            'positive_words_found': positive_words_found,
            'has_negative_words': has_negative_words,
            'negative_words_found': negative_words_found,
            'has_emoji': len(emojis) > 0,
            'emojis_found': emojis,
            'has_positive_emoji': has_positive_emoji,
            'has_negative_emoji': has_negative_emoji,
            'has_exclamation': has_exclamation
        }
    
    def _calculate_confidence(self, compound: float, features: Dict, 
                            context: Optional[Dict] = None) -> Tuple[float, List[str]]:
        """
        Calculate confidence score for sentiment classification
        Returns: (confidence_score, reasoning_list)
        
        NOTE: No longer penalizes based on ambiguous terms - 
        climate relevance is pre-validated by upstream pipeline
        """
        base_confidence = 1.0
        reasoning = []
        
        word_count = features['word_count']
        
        # Text length adjustment
        if word_count <= 2:
            base_confidence *= 0.6
            reasoning.append(f"Very short text ({word_count} words) reduces confidence")
        elif word_count <= 5:
            base_confidence *= 0.85
            reasoning.append(f"Short text ({word_count} words) slightly reduces confidence")
        elif word_count >= 10:
            base_confidence *= 1.1
            reasoning.append(f"Sufficient text length ({word_count} words) increases confidence")
        
        # Neutral zone penalty (compound near 0)
        if abs(compound) < 0.1:
            base_confidence *= 0.6
            reasoning.append("Sentiment score near neutral zone")
        elif abs(compound) >= 0.5:
            base_confidence *= 1.2
            reasoning.append("Strong sentiment score increases confidence")
        
        # Intensifier boost
        if features['has_intensifiers']:
            base_confidence *= 1.2
            reasoning.append(f"Intensifier(s) present: {', '.join(features['intensifiers_found'])}")
        else:
            if word_count <= 5:
                base_confidence *= 0.9
                reasoning.append("No intensifiers in short text")
        
        # Explicit sentiment words boost
        if features['has_positive_words']:
            base_confidence *= 1.15
            reasoning.append(f"Positive evaluation words: {', '.join(features['positive_words_found'])}")
        
        if features['has_negative_words']:
            base_confidence *= 1.15
            reasoning.append(f"Negative evaluation words: {', '.join(features['negative_words_found'])}")
        
        # Emoji boost
        if features['has_positive_emoji']:
            base_confidence *= 1.15
            reasoning.append("Positive emoji(s) reinforce sentiment")
        
        if features['has_negative_emoji']:
            base_confidence *= 1.15
            reasoning.append("Negative emoji(s) reinforce sentiment")
        
        # Exclamation boost
        if features['has_exclamation']:
            base_confidence *= 1.1
            reasoning.append("Exclamation mark indicates emphasis")
        
        # Context-based adjustments from upstream pipeline
        if context:
            # Climate category provides context
            if context.get('climate_category'):
                base_confidence *= 1.1
                reasoning.append(f"Climate category identified: {context['climate_category']}")
            
            # Weather data provides validation context
            if context.get('weather_data'):
                base_confidence *= 1.05
                reasoning.append("Weather data available for context validation")
            
            # High climate classification confidence boosts sentiment confidence
            classification_confidence = context.get('confidence', 0)
            if classification_confidence > 0.7:
                base_confidence *= 1.05
                reasoning.append(f"High climate classification confidence ({classification_confidence:.2f})")
        
        # Cap confidence at 0.95 (never 100% certain)
        final_confidence = min(0.95, base_confidence)
        
        return final_confidence, reasoning
    
    def analyze_sentiment(self, text: str, debug: bool = False, 
                          context: Optional[Dict] = None) -> SentimentScore:
        """
        Analyze sentiment with confidence scoring
        
        Assumes upstream pipeline has already:
        - Verified climate relevance
        - Classified climate domain/category
        - Validated against weather data (if applicable)
        """
        if not text or not text.strip():
            if debug:
                print("⚠️ Empty text provided")
            return SentimentScore(
                positive=0.0, negative=0.0, neutral=1.0, compound=0.0,
                classification='neutral', confidence=0.0, confidence_tier='LOW',
                reasoning=["Empty text"], include_in_stats=False,
                qualitative_category="Invalid Input"
            )
        
        processed_text = self.preprocess_text(text, debug=debug)
        
        if not processed_text:
            if debug:
                print("⚠️ Processed text is empty after preprocessing")
            return SentimentScore(
                positive=0.0, negative=0.0, neutral=1.0, compound=0.0,
                classification='neutral', confidence=0.0, confidence_tier='LOW',
                reasoning=["Text empty after preprocessing"], include_in_stats=False,
                qualitative_category="Invalid Input"
            )
        
        # Extract context features
        features = self._extract_context_features(processed_text)
        
        if debug:
            print(f"\n🔍 Context Features:")
            for key, value in features.items():
                if value and value != [] and value != False:
                    print(f"   {key}: {value}")
        
        # Get VADER scores
        scores = self.polarity_scores(processed_text)
        compound = scores['compound']
        
        if debug:
            print(f"\n📊 Raw VADER scores:")
            print(f"   Positive: {scores['pos']:.3f}")
            print(f"   Negative: {scores['neg']:.3f}")
            print(f"   Neutral:  {scores['neu']:.3f}")
            print(f"   Compound: {compound:+.3f}")
        
        # Apply context adjustments (minimal now - no ambiguous term penalties)
        adjustments = []
        
        # Amplify if strong context indicators present
        if features['has_intensifiers'] and abs(compound) > 0.3:
            original_compound = compound
            compound = compound * 1.1
            adjustments.append(f"Intensifier amplification: {original_compound:.3f} → {compound:.3f}")
        
        if debug and adjustments:
            print(f"\n🔧 Adjustments Applied:")
            for adj in adjustments:
                print(f"   • {adj}")
        
        # Calculate confidence
        confidence, confidence_reasoning = self._calculate_confidence(compound, features, context)
        
        # Determine classification
        if confidence < 0.4:
            # LOW confidence = INCONCLUSIVE
            classification = 'inconclusive'
            confidence_tier = 'LOW'
            include_in_stats = False
            qualitative_category = "Insufficient Context for Classification"
        else:
            # Determine sentiment
            if compound >= 0.05:
                classification = 'positive'
                qualitative_category = "Weather Appreciation - Positive Experience"
            elif compound <= -0.05:
                classification = 'negative'
                qualitative_category = "Weather Discomfort - Complaint"
            else:
                classification = 'neutral'
                qualitative_category = "Neutral Weather Observation"
            
            # Determine tier
            if confidence >= 0.7:
                confidence_tier = 'HIGH'
                include_in_stats = True
            else:
                confidence_tier = 'MODERATE'
                include_in_stats = True  # Include but flag for review
        
        if debug:
            print(f"\n📊 Confidence: {confidence:.2f} ({confidence_tier})")
            print(f"🎯 Classification: {classification.upper()}")
            print(f"📈 Include in stats: {include_in_stats}")
        
        return SentimentScore(
            positive=round(scores['pos'], 3),
            negative=round(scores['neg'], 3),
            neutral=round(scores['neu'], 3),
            compound=round(compound, 3),
            classification=classification,
            confidence=round(confidence, 3),
            confidence_tier=confidence_tier,
            reasoning=confidence_reasoning,
            include_in_stats=include_in_stats,
            qualitative_category=qualitative_category
        )
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentScore]:
        """Analyze sentiment for a batch of texts"""
        results = []
        for text in texts:
            try:
                sentiment = self.analyze_sentiment(text)
                results.append(sentiment)
            except Exception as e:
                print(f"Error analyzing sentiment for text: {str(e)}")
                results.append(SentimentScore(
                    positive=0.0, negative=0.0, neutral=1.0, compound=0.0,
                    classification='inconclusive', confidence=0.0, confidence_tier='LOW',
                    reasoning=["Analysis error"], include_in_stats=False,
                    qualitative_category="Error"
                ))
        
        return results
    
    def get_lexicon_stats(self) -> Dict:
        """Get statistics about the loaded lexicon"""
        if not self.lexicon:
            return {"error": "No lexicon loaded"}
        
        custom_words = set(self.custom_lexicon.keys())
        base_words = len(self.lexicon) - len(custom_words)
        
        positive_words = sum(1 for score in self.lexicon.values() if score > 0)
        negative_words = sum(1 for score in self.lexicon.values() if score < 0)
        neutral_words = sum(1 for score in self.lexicon.values() if score == 0)
        
        custom_positive = sum(1 for word in custom_words if self.lexicon.get(word, 0) > 0)
        custom_negative = sum(1 for word in custom_words if self.lexicon.get(word, 0) < 0)
        custom_neutral = sum(1 for word in custom_words if self.lexicon.get(word, 0) == 0)
        
        avg_sentiment = sum(self.lexicon.values()) / len(self.lexicon) if self.lexicon else 0
        
        custom_scored = [(w, self.lexicon.get(w, 0)) for w in custom_words]
        if custom_scored:
            most_positive = max(custom_scored, key=lambda x: x[1])
            most_negative = min(custom_scored, key=lambda x: x[1])
        else:
            most_positive = ("N/A", 0)
            most_negative = ("N/A", 0)
        
        return {
            "total_words": len(self.lexicon),
            "base_vader_words": base_words,
            "custom_words": len(custom_words),
            "positive_words": positive_words,
            "negative_words": negative_words,
            "neutral_words": neutral_words,
            "custom_positive": custom_positive,
            "custom_negative": custom_negative,
            "custom_neutral": custom_neutral,
            "average_sentiment": round(avg_sentiment, 4),
            "most_positive_custom_word": {
                "word": most_positive[0],
                "sentiment": round(most_positive[1], 4)
            },
            "most_negative_custom_word": {
                "word": most_negative[0],
                "sentiment": round(most_negative[1], 4)
            },
            "load_errors_count": len(self.load_errors) if hasattr(self, 'load_errors') else 0
        }
    
    def test_word(self, word: str) -> Dict:
        """Test sentiment value for a specific word"""
        word_lower = word.lower()
        
        if word_lower in self.lexicon:
            sentiment = self.lexicon[word_lower]
            is_custom = word_lower in self.custom_lexicon
            
            return {
                "word": word,
                "found": True,
                "sentiment": round(sentiment, 4),
                "source": "custom_lexicon" if is_custom else "base_vader",
                "interpretation": _interpret_sentiment_value(sentiment)
            }
        else:
            similar = [w for w in self.lexicon.keys() if word_lower in w or w in word_lower][:5]
            return {
                "word": word,
                "found": False,
                "message": "Word not found in lexicon",
                "similar_words": similar if similar else None
            }


# Global instance
sentiment_analyzer = MultilingualVADER()


def sentiment_model_status() -> Dict:
    """Get status of the sentiment analysis model"""
    try:
        stats = sentiment_analyzer.get_lexicon_stats()
        if "error" in stats:
            return {
                "status": "error",
                "error": stats["error"],
                "lexicon_loaded": False
            }
        
        return {
            "status": "ready",
            "model": "VADER (Enhanced Multilingual with Confidence Tiers)",
            "lexicon_loaded": True,
            "lexicon_path": sentiment_analyzer.custom_lexicon_path,
            "lexicon_stats": stats,
            "note": "Ambiguous term checking removed - relies on upstream climate classification"
        }
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "lexicon_loaded": False
        }


def analyze_tweet_sentiment(tweet_text: str, debug: bool = False, 
                           context: Optional[Dict] = None) -> Dict:
    """Analyze sentiment of a single tweet with confidence scoring"""
    try:
        if not tweet_text or not tweet_text.strip():
            return {"error": "Tweet text is empty"}
        
        sentiment_score = sentiment_analyzer.analyze_sentiment(
            tweet_text, 
            debug=debug,
            context=context
        )
        
        response = {
            "status": "ok",
            "tweet": tweet_text,
            "sentiment": {
                "positive": sentiment_score.positive,
                "negative": sentiment_score.negative,
                "neutral": sentiment_score.neutral,
                "compound": sentiment_score.compound,
                "classification": sentiment_score.classification
            },
            "confidence": {
                "score": sentiment_score.confidence,
                "tier": sentiment_score.confidence_tier,
                "reasoning": sentiment_score.reasoning
            },
            "metadata": {
                "include_in_statistics": sentiment_score.include_in_stats,
                "qualitative_category": sentiment_score.qualitative_category
            },
            "interpretation": _interpret_compound_score(sentiment_score.compound),
            "processed_text": sentiment_analyzer.preprocess_text(tweet_text)
        }
        
        # Add warning for inconclusive
        if sentiment_score.classification == 'inconclusive':
            response["warning"] = "INCONCLUSIVE: Insufficient context for reliable sentiment classification. Excluded from quantitative analysis."
        
        return response
    
    except Exception as e:
        import traceback
        return {
            "error": f"Sentiment analysis failed: {str(e)}",
            "traceback": traceback.format_exc()
        }


def analyze_batch_sentiment(tweets: List[str]) -> List[Dict]:
    """Analyze sentiment for a batch of tweets"""
    try:
        sentiment_scores = sentiment_analyzer.analyze_batch(tweets)
        
        results = []
        for i, (tweet, score) in enumerate(zip(tweets, sentiment_scores)):
            result = {
                "index": i,
                "tweet": tweet,
                "sentiment": {
                    "positive": score.positive,
                    "negative": score.negative,
                    "neutral": score.neutral,
                    "compound": score.compound,
                    "classification": score.classification
                },
                "confidence": {
                    "score": score.confidence,
                    "tier": score.confidence_tier,
                    "reasoning": score.reasoning
                },
                "metadata": {
                    "include_in_statistics": score.include_in_stats,
                    "qualitative_category": score.qualitative_category
                },
                "interpretation": _interpret_compound_score(score.compound)
            }
            results.append(result)
        
        return results
    
    except Exception as e:
        print(f"Batch sentiment analysis failed: {str(e)}")
        return [{"error": f"Sentiment analysis failed: {str(e)}"} for _ in tweets]


def analyze_sentiment_distribution(sentiment_results: List[Dict]) -> Dict:
    """Analyze distribution with confidence tier breakdown"""
    if not sentiment_results:
        return {"error": "No sentiment results provided"}
    
    # Initialize counters
    total_tweets = len(sentiment_results)
    
    classification_counts = {
        "positive": 0, "negative": 0, "neutral": 0, "inconclusive": 0, "errors": 0
    }
    
    confidence_tier_counts = {
        "HIGH": {"positive": 0, "negative": 0, "neutral": 0},
        "MODERATE": {"positive": 0, "negative": 0, "neutral": 0},
        "LOW": {"positive": 0, "negative": 0, "neutral": 0, "inconclusive": 0}
    }
    
    compound_scores = []
    qualitative_categories = {}
    
    for result in sentiment_results:
        if "error" in result:
            classification_counts["errors"] += 1
            continue
        
        sentiment_data = result.get("sentiment", {})
        confidence_data = result.get("confidence", {})
        metadata = result.get("metadata", {})
        
        classification = sentiment_data.get("classification", "neutral")
        compound = sentiment_data.get("compound", 0)
        confidence_tier = confidence_data.get("tier", "LOW")
        qualitative_category = metadata.get("qualitative_category", "Unknown")
        
        # Count classifications
        classification_counts[classification] += 1
        
        # Count by confidence tier
        if classification in confidence_tier_counts[confidence_tier]:
            confidence_tier_counts[confidence_tier][classification] += 1
        
        # Store compound scores (only for conclusive results)
        if classification != 'inconclusive':
            compound_scores.append(compound)
        
        # Count qualitative categories
        if qualitative_category not in qualitative_categories:
            qualitative_categories[qualitative_category] = 0
        qualitative_categories[qualitative_category] += 1
    
    # Calculate statistics for conclusive results only
    total_conclusive = sum(v for k, v in classification_counts.items() 
                          if k not in ["inconclusive", "errors"])
    
    if total_conclusive > 0:
        conclusive_percentages = {
            "positive": round((classification_counts["positive"] / total_conclusive) * 100, 2),
            "negative": round((classification_counts["negative"] / total_conclusive) * 100, 2),
            "neutral": round((classification_counts["neutral"] / total_conclusive) * 100, 2)
        }
    else:
        conclusive_percentages = {"positive": 0, "negative": 0, "neutral": 0}
    
    # Calculate overall percentages (including inconclusive)
    total_valid = total_tweets - classification_counts["errors"]
    if total_valid > 0:
        overall_percentages = {
            k: round((v / total_valid) * 100, 2) 
            for k, v in classification_counts.items() 
            if k != "errors"
        }
    else:
        overall_percentages = {"positive": 0, "negative": 0, "neutral": 0, "inconclusive": 0}
    
    # Compound score statistics
    if compound_scores:
        avg_compound = sum(compound_scores) / len(compound_scores)
        most_positive = max(compound_scores)
        most_negative = min(compound_scores)
    else:
        avg_compound = most_positive = most_negative = 0
    
    return {
        "total_tweets": total_tweets,
        "classification_counts": classification_counts,
        "confidence_tier_breakdown": confidence_tier_counts,
        "conclusive_statistics": {
            "total_conclusive": total_conclusive,
            "percentages": conclusive_percentages,
            "note": "Percentages calculated from conclusive classifications only (excludes inconclusive)"
        },
        "overall_statistics": {
            "total_valid": total_valid,
            "percentages": overall_percentages,
            "note": "Percentages include all valid classifications (including inconclusive)"
        },
        "compound_statistics": {
            "average": round(avg_compound, 4),
            "most_positive": round(most_positive, 4),
            "most_negative": round(most_negative, 4),
            "total_analyzed": len(compound_scores)
        },
        "qualitative_categories": qualitative_categories,
        "data_quality": {
            "high_confidence": sum(confidence_tier_counts["HIGH"].values()),
            "moderate_confidence": sum(confidence_tier_counts["MODERATE"].values()),
            "low_confidence_inconclusive": sum(confidence_tier_counts["LOW"].values()),
            "errors": classification_counts["errors"]
        }
    }


def get_sentiment_insights(sentiment_results: List[Dict]) -> Dict:
    """Generate insights about sentiment patterns with confidence awareness"""
    try:
        distribution = analyze_sentiment_distribution(sentiment_results)
        
        if "error" in distribution:
            return distribution
        
        insights = []
        
        # Use conclusive statistics for sentiment insights
        conclusive_stats = distribution["conclusive_statistics"]
        percentages = conclusive_stats["percentages"]
        
        total_conclusive = conclusive_stats["total_conclusive"]
        total_inconclusive = distribution["classification_counts"]["inconclusive"]
        
        # Data quality insights
        data_quality = distribution["data_quality"]
        high_conf = data_quality["high_confidence"]
        moderate_conf = data_quality["moderate_confidence"]
        
        insights.append(f"Analyzed {distribution['total_tweets']} tweets: {total_conclusive} conclusive, {total_inconclusive} inconclusive")
        insights.append(f"Data quality: {high_conf} high confidence, {moderate_conf} moderate confidence classifications")
        
        # Overall sentiment insights (from conclusive only)
        if total_conclusive > 0:
            if percentages["positive"] > 50:
                insights.append(f"Overall sentiment leans POSITIVE ({percentages['positive']:.1f}% of conclusive tweets)")
            elif percentages["negative"] > 50:
                insights.append(f"Overall sentiment leans NEGATIVE ({percentages['negative']:.1f}% of conclusive tweets)")
            else:
                insights.append(f"Sentiment distribution is balanced (P:{percentages['positive']:.1f}%, N:{percentages['negative']:.1f}%, Neu:{percentages['neutral']:.1f}%)")
        
        # Extreme sentiment insights
        compound_stats = distribution["compound_statistics"]
        if compound_stats["most_positive"] > 0.7:
            insights.append(f"Contains very strong positive sentiment (max: {compound_stats['most_positive']:.3f})")
        if compound_stats["most_negative"] < -0.7:
            insights.append(f"Contains very strong negative sentiment (min: {compound_stats['most_negative']:.3f})")
        
        # Inconclusive rate insight
        if total_inconclusive > 0:
            inconclusive_rate = (total_inconclusive / distribution['total_tweets']) * 100
            if inconclusive_rate > 20:
                insights.append(f"High inconclusive rate ({inconclusive_rate:.1f}%) - many tweets lack sufficient sentiment indicators")
            else:
                insights.append(f"Low inconclusive rate ({inconclusive_rate:.1f}%) - most tweets have clear sentiment indicators")
        
        # Qualitative category insights
        qual_categories = distribution["qualitative_categories"]
        if qual_categories:
            most_common_category = max(qual_categories.items(), key=lambda x: x[1])
            insights.append(f"Most common category: '{most_common_category[0]}' ({most_common_category[1]} tweets)")
        
        return {
            "status": "ok",
            "insights": insights,
            "distribution_summary": distribution,
            "lexicon_stats": sentiment_analyzer.get_lexicon_stats(),
            "methodology_note": "Sentiment statistics calculated from conclusive classifications only. Inconclusive tweets are excluded to maintain analytical rigor."
        }
    
    except Exception as e:
        return {"error": f"Failed to generate sentiment insights: {str(e)}"}


def reload_lexicon() -> Dict:
    """Reload the lexical dictionary"""
    global sentiment_analyzer
    
    try:
        old_stats = sentiment_analyzer.get_lexicon_stats()
        old_count = old_stats.get("total_words", 0)
        
        sentiment_analyzer = MultilingualVADER()
        
        new_stats = sentiment_analyzer.get_lexicon_stats()
        new_count = new_stats.get("total_words", 0)
        
        return {
            "status": "ok",
            "message": f"Lexicon reloaded: {old_count} -> {new_count} words",
            "lexicon_stats": new_stats
        }
    
    except Exception as e:
        return {"error": f"Failed to reload lexicon: {str(e)}"}


def test_word_sentiment(word: str) -> Dict:
    """Test sentiment analysis for a specific word"""
    return sentiment_analyzer.test_word(word)


# Helper functions
def _interpret_compound_score(compound: float) -> str:
    """Interpret compound score with descriptive labels"""
    if compound >= 0.5:
        return "Very Positive"
    elif compound >= 0.05:
        return "Positive"
    elif compound <= -0.5:
        return "Very Negative"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def _interpret_sentiment_value(value: float) -> str:
    """Interpret sentiment value from lexicon"""
    if value >= 2.0:
        return "Strongly Positive"
    elif value >= 1.0:
        return "Positive"
    elif value > 0:
        return "Slightly Positive"
    elif value <= -2.0:
        return "Strongly Negative"
    elif value <= -1.0:
        return "Negative"
    elif value < 0:
        return "Slightly Negative"
    else:
        return "Neutral"