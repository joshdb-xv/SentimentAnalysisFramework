# services/sentiment_analysis.py - FIXED FOR JOBLIB

import os
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import joblib

@dataclass
class SentimentScore:
    """Data class to hold sentiment analysis results"""
    positive: float
    negative: float
    neutral: float
    compound: float
    classification: str  # 'positive', 'negative', or 'neutral'

class MultilingualVADER(SentimentIntensityAnalyzer):
    """
    Extended VADER sentiment analyzer with custom multilingual lexicon support
    Supports English, Tagalog, and Cebuano words
    """
    
    def __init__(self, lexicon_path: str = None):
        # Initialize the base VADER analyzer
        super().__init__()
        
        # Auto-detect lexicon path if not provided
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
                lexicon_path = "data/lexical_dictionary/lexical_dictionary.joblib"  # fallback
        
        self.custom_lexicon_path = lexicon_path
        self.custom_lexicon = {}
        self.load_errors = []
        
        # Load and merge custom lexicon
        self._load_custom_lexicon()
        self._merge_lexicons()
        
        # Add Tagalog/Cebuano specific negations to VADER's negation set
        self._add_filipino_negations()
    
    def _load_custom_lexicon(self) -> None:
        """Load the custom lexical dictionary from joblib file"""
        try:
            if not os.path.exists(self.custom_lexicon_path):
                raise FileNotFoundError(f"Lexical dictionary not found at: {self.custom_lexicon_path}")
            
            print(f"📖 Loading lexicon from: {self.custom_lexicon_path}")
            
            # Load the joblib file - it contains a dictionary with 'lexicon_df' and 'metadata'
            loaded_data = joblib.load(self.custom_lexicon_path)
            
            # Check what type of data we loaded
            if isinstance(loaded_data, dict):
                # The lexical dictionary manager saves as {'lexicon': df, 'word_data': dict, 'metadata': dict}
                if 'lexicon' in loaded_data:
                    df = loaded_data['lexicon']
                    metadata = loaded_data.get('metadata', {})
                    print(f"✅ Loaded DataFrame with {len(df)} rows from cached dictionary")
                    if metadata:
                        created = metadata.get('created_at', 'unknown')
                        print(f"   Created: {created}")
                elif 'lexicon_df' in loaded_data:
                    # Backwards compatibility
                    df = loaded_data['lexicon_df']
                    metadata = loaded_data.get('metadata', {})
                    print(f"✅ Loaded DataFrame with {len(df)} rows from cached dictionary")
                    if metadata:
                        created = metadata.get('created_at', 'unknown')
                        print(f"   Created: {created}")
                else:
                    # Try to convert dict directly to DataFrame
                    df = pd.DataFrame(loaded_data)
                    print(f"✅ Converted dict to DataFrame with {len(df)} rows")
            elif isinstance(loaded_data, pd.DataFrame):
                df = loaded_data
                print(f"✅ Loaded DataFrame with {len(df)} rows")
            else:
                raise ValueError(f"Unexpected data type in joblib file: {type(loaded_data)}")
            
            # Print column names for debugging
            print(f"📋 Columns found: {list(df.columns)}")
            
            # Ensure required columns exist
            # The lexical dictionary uses 'word' and 'sentiment_score', not 'sentivalue'
            if 'sentiment_score' in df.columns and 'word' in df.columns:
                # This is the format from lexical_dictionary_manager - rename for processing
                df = df.rename(columns={'sentiment_score': 'sentivalue'})
                print(f"📋 Detected lexical dictionary format, renamed 'sentiment_score' to 'sentivalue'")
            
            required_cols = ['word', 'sentivalue']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Dictionary must contain columns: {required_cols}. Found: {list(df.columns)}")
            
            # Load words into custom lexicon
            loaded_count = 0
            skipped_count = 0
            
            for idx, row in df.iterrows():
                try:
                    word = str(row['word']).strip().lower()
                    if not word or word == 'nan':  # Skip empty or invalid words
                        skipped_count += 1
                        continue
                    
                    # Use sentivalue as the primary sentiment score
                    sentiment_value = float(row['sentivalue'])
                    
                    # Skip if sentiment is NaN
                    if pd.isna(sentiment_value):
                        skipped_count += 1
                        continue
                    
                    # Get weight if available, default to 1.0
                    weight = float(row.get('weight', 1.0)) if 'weight' in row and not pd.isna(row.get('weight')) else 1.0
                    
                    # VADER expects sentiment values typically between -4 and 4
                    # Adjust the sentiment value based on weight
                    adjusted_sentiment = sentiment_value * weight
                    
                    # Clamp to VADER's range
                    adjusted_sentiment = max(-4.0, min(4.0, adjusted_sentiment))
                    
                    # Store in custom lexicon
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
            
            # Show sample of loaded words
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
        """
        Merge custom lexicon with VADER's base lexicon
        Custom words will override VADER's default values
        """
        if not self.custom_lexicon:
            print("⚠️ No custom lexicon to merge - using base VADER only")
            print("   This means Tagalog/Cebuano words will not be recognized!")
            return
        
        # VADER stores its lexicon in self.lexicon
        original_count = len(self.lexicon)
        
        # Add/override with custom lexicon entries
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
            # Tagalog negations
            "hindi", "walang", "wala", "di", "huwag", "ayaw",
            "hinde", "ayoko", "ayaw ko",
            
            # Cebuano negations
            "dili", "walay",
            
            # Common variations
            "hndi", "wla", "d", "hwag"
        }
        
        # Add to VADER's negation words (stored in self.lexicon with special handling)
        for neg in filipino_negations:
            if neg not in self.lexicon:
                # Add as neutral word but it will be recognized as negation
                self.lexicon[neg] = 0.0
    
    def preprocess_text(self, text: str, debug: bool = False) -> str:
        """
        Preprocess text before sentiment analysis
        Handles Twitter-specific elements and Filipino text patterns
        """
        if not text:
            return ""
        
        original = text
        
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Handle Twitter-specific elements
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        
        # Keep hashtags but remove the # symbol (content is valuable)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Handle common Filipino text patterns
        # Repeated characters for emphasis (e.g., "sobrangggg" -> "sobrang")
        # BUT keep at least 2 repetitions as they might be intentional
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        if debug:
            print(f"🔍 Preprocessing:")
            print(f"   Original: '{original}'")
            print(f"   Processed: '{text}'")
        
        return text
    
    def analyze_sentiment(self, text: str, debug: bool = False) -> SentimentScore:
        """
        Analyze sentiment using enhanced VADER with custom lexicon
        """
        if not text or not text.strip():
            if debug:
                print("⚠️ Empty text provided")
            return SentimentScore(0.0, 0.0, 1.0, 0.0, 'neutral')
        
        # Preprocess the text
        processed_text = self.preprocess_text(text, debug=debug)
        
        if not processed_text:
            if debug:
                print("⚠️ Processed text is empty after preprocessing")
            return SentimentScore(0.0, 0.0, 1.0, 0.0, 'neutral')
        
        # Debug: Check which words are in lexicon
        if debug:
            words = processed_text.split()
            print(f"\n🔍 Word-by-word lexicon lookup:")
            for word in words:
                if word in self.lexicon:
                    sentiment = self.lexicon[word]
                    source = "CUSTOM" if word in self.custom_lexicon else "VADER"
                    print(f"   ✅ '{word}': {sentiment:+.4f} ({source})")
                else:
                    print(f"   ❌ '{word}': NOT IN LEXICON")
            
            # Check if any words were found
            found_words = [w for w in words if w in self.lexicon]
            if not found_words:
                print(f"\n   ⚠️  WARNING: No words found in lexicon!")
                print(f"   This will result in neutral sentiment (0.0)")
        
        # Use VADER's polarity_scores method
        scores = self.polarity_scores(processed_text)
        
        if debug:
            print(f"\n📊 Raw VADER scores:")
            print(f"   Positive: {scores['pos']:.3f}")
            print(f"   Negative: {scores['neg']:.3f}")
            print(f"   Neutral:  {scores['neu']:.3f}")
            print(f"   Compound: {scores['compound']:+.3f}")
        
        # Determine classification based on compound score
        compound = scores['compound']
        if compound >= 0.05:
            classification = 'positive'
        elif compound <= -0.05:
            classification = 'negative'
        else:
            classification = 'neutral'
        
        result = SentimentScore(
            positive=round(scores['pos'], 3),
            negative=round(scores['neg'], 3),
            neutral=round(scores['neu'], 3),
            compound=round(compound, 3),
            classification=classification
        )
        
        if debug:
            print(f"\n🎯 Final classification: {classification.upper()}")
        
        return result
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentScore]:
        """Analyze sentiment for a batch of texts"""
        results = []
        for text in texts:
            try:
                sentiment = self.analyze_sentiment(text)
                results.append(sentiment)
            except Exception as e:
                print(f"Error analyzing sentiment for text: {str(e)}")
                # Return neutral score on error
                results.append(SentimentScore(0.0, 0.0, 1.0, 0.0, 'neutral'))
        
        return results
    
    def get_lexicon_stats(self) -> Dict:
        """Get statistics about the loaded lexicon"""
        if not self.lexicon:
            return {"error": "No lexicon loaded"}
        
        # Separate custom and base VADER words
        custom_words = set(self.custom_lexicon.keys())
        base_words = len(self.lexicon) - len(custom_words)
        
        # Calculate statistics
        positive_words = sum(1 for score in self.lexicon.values() if score > 0)
        negative_words = sum(1 for score in self.lexicon.values() if score < 0)
        neutral_words = sum(1 for score in self.lexicon.values() if score == 0)
        
        # Custom lexicon stats
        custom_positive = sum(1 for word in custom_words 
                             if self.lexicon.get(word, 0) > 0)
        custom_negative = sum(1 for word in custom_words 
                             if self.lexicon.get(word, 0) < 0)
        custom_neutral = sum(1 for word in custom_words 
                            if self.lexicon.get(word, 0) == 0)
        
        avg_sentiment = sum(self.lexicon.values()) / len(self.lexicon) if self.lexicon else 0
        
        # Find most positive and negative words from custom lexicon
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
            # Check if similar words exist
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
            "model": "VADER (Enhanced Multilingual)",
            "lexicon_loaded": True,
            "lexicon_path": sentiment_analyzer.custom_lexicon_path,
            "lexicon_stats": stats
        }
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "lexicon_loaded": False
        }


def analyze_tweet_sentiment(tweet_text: str, debug: bool = False) -> Dict:
    """Analyze sentiment of a single tweet"""
    try:
        if not tweet_text or not tweet_text.strip():
            return {"error": "Tweet text is empty"}
        
        sentiment_score = sentiment_analyzer.analyze_sentiment(tweet_text, debug=debug)
        
        return {
            "status": "ok",
            "tweet": tweet_text,
            "sentiment": {
                "positive": sentiment_score.positive,
                "negative": sentiment_score.negative,
                "neutral": sentiment_score.neutral,
                "compound": sentiment_score.compound,
                "classification": sentiment_score.classification
            },
            "interpretation": _interpret_compound_score(sentiment_score.compound),
            "processed_text": sentiment_analyzer.preprocess_text(tweet_text)
        }
    
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
                "interpretation": _interpret_compound_score(score.compound)
            }
            results.append(result)
        
        return results
    
    except Exception as e:
        print(f"Batch sentiment analysis failed: {str(e)}")
        return [{"error": f"Sentiment analysis failed: {str(e)}"} for _ in tweets]


def analyze_sentiment_distribution(sentiment_results: List[Dict]) -> Dict:
    """Analyze distribution of sentiment classifications in batch results"""
    if not sentiment_results:
        return {"error": "No sentiment results provided"}
    
    # Count classifications
    classification_counts = {"positive": 0, "negative": 0, "neutral": 0, "errors": 0}
    compound_scores = []
    sentiment_by_category = {}
    
    for result in sentiment_results:
        if "error" in result:
            classification_counts["errors"] += 1
            continue
        
        sentiment_data = result.get("sentiment", {})
        classification = sentiment_data.get("classification", "neutral")
        compound = sentiment_data.get("compound", 0)
        
        classification_counts[classification] += 1
        compound_scores.append(compound)
        
        # Cross-reference with climate category if available
        if "category_classification" in result and "prediction" in result["category_classification"]:
            category = result["category_classification"]["prediction"]
            
            if category not in sentiment_by_category:
                sentiment_by_category[category] = {"positive": 0, "negative": 0, "neutral": 0}
            
            sentiment_by_category[category][classification] += 1
    
    # Calculate statistics
    total_valid = sum(v for k, v in classification_counts.items() if k != "errors")
    
    if total_valid > 0:
        percentages = {
            k: round((v / total_valid) * 100, 2) 
            for k, v in classification_counts.items() 
            if k != "errors"
        }
    else:
        percentages = {"positive": 0, "negative": 0, "neutral": 0}
    
    # Calculate compound score statistics
    if compound_scores:
        avg_compound = sum(compound_scores) / len(compound_scores)
        most_positive = max(compound_scores)
        most_negative = min(compound_scores)
    else:
        avg_compound = most_positive = most_negative = 0
    
    return {
        "classification_counts": classification_counts,
        "percentages": percentages,
        "compound_statistics": {
            "average": round(avg_compound, 4),
            "most_positive": round(most_positive, 4),
            "most_negative": round(most_negative, 4),
            "total_analyzed": len(compound_scores)
        },
        "sentiment_by_climate_category": sentiment_by_category,
        "total_tweets": len(sentiment_results),
        "successful_analysis": total_valid
    }


def get_sentiment_insights(sentiment_results: List[Dict]) -> Dict:
    """Generate insights about sentiment patterns in the data"""
    try:
        distribution = analyze_sentiment_distribution(sentiment_results)
        
        if "error" in distribution:
            return distribution
        
        insights = []
        
        # Overall sentiment insights
        percentages = distribution["percentages"]
        if percentages["positive"] > 50:
            insights.append("The overall sentiment leans positive")
        elif percentages["negative"] > 50:
            insights.append("The overall sentiment leans negative")
        else:
            insights.append("The sentiment distribution is relatively balanced")
        
        # Extreme sentiment insights
        compound_stats = distribution["compound_statistics"]
        if compound_stats["most_positive"] > 0.7:
            insights.append("Contains very strong positive sentiment")
        if compound_stats["most_negative"] < -0.7:
            insights.append("Contains very strong negative sentiment")
        
        # Category-specific insights
        sentiment_by_category = distribution["sentiment_by_climate_category"]
        if sentiment_by_category:
            most_positive_category = None
            most_negative_category = None
            max_pos_ratio = 0
            max_neg_ratio = 0
            
            for category, counts in sentiment_by_category.items():
                total_cat = sum(counts.values())
                if total_cat > 0:
                    pos_ratio = counts["positive"] / total_cat
                    neg_ratio = counts["negative"] / total_cat
                    
                    if pos_ratio > max_pos_ratio:
                        max_pos_ratio = pos_ratio
                        most_positive_category = category
                    
                    if neg_ratio > max_neg_ratio:
                        max_neg_ratio = neg_ratio
                        most_negative_category = category
            
            if most_positive_category and max_pos_ratio > 0.6:
                insights.append(f"'{most_positive_category}' tweets tend to be more positive")
            
            if most_negative_category and max_neg_ratio > 0.6:
                insights.append(f"'{most_negative_category}' tweets tend to be more negative")
        
        return {
            "status": "ok",
            "insights": insights,
            "distribution_summary": distribution,
            "lexicon_stats": sentiment_analyzer.get_lexicon_stats()
        }
    
    except Exception as e:
        return {"error": f"Failed to generate sentiment insights: {str(e)}"}


def reload_lexicon() -> Dict:
    """Reload the lexical dictionary (useful if the file has been updated)"""
    global sentiment_analyzer
    
    try:
        old_stats = sentiment_analyzer.get_lexicon_stats()
        old_count = old_stats.get("total_words", 0)
        
        # Reload the analyzer
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