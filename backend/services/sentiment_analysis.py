# services/sentiment_analysis.py

import csv
import os
import re
import string
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

@dataclass
class SentimentScore:
    """Data class to hold sentiment analysis results"""
    positive: float
    negative: float
    neutral: float
    compound: float
    classification: str  # 'positive', 'negative', or 'neutral'

class CustomVADER:
    """
    Custom VADER-style sentiment analyzer using our annotated lexical dictionary
    """
    
    def __init__(self, lexicon_path: str = "data/lexical_dictionary.csv"):
        self.lexicon_path = lexicon_path
        self.lexicon = {}
        self.booster_dict = {}
        self.negation_dict = set()
        
        # VADER constants
        self.B_INCR = 0.293
        self.B_DECR = -0.293
        self.C_INCR = 0.733
        self.N_SCALAR = -0.74
        self.NORMALIZE_ALPHA = 15
        
        # Initialize lexicon and helper dictionaries
        self._load_custom_lexicon()
        self._initialize_boosters()
        self._initialize_negations()
    
    def _load_custom_lexicon(self) -> None:
        """Load the custom lexical dictionary from CSV"""
        try:
            if not os.path.exists(self.lexicon_path):
                raise FileNotFoundError(f"Lexical dictionary not found at: {self.lexicon_path}")
            
            with open(self.lexicon_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    word = row.get('word', '').strip().lower()
                    
                    # Use sentivalue as primary sentiment score
                    try:
                        sentiment_value = float(row.get('sentivalue', 0))
                        weight = float(row.get('weight', 1.0))
                        
                        # Store both sentiment value and weight for more nuanced analysis
                        self.lexicon[word] = {
                            'sentiment': sentiment_value,
                            'weight': weight,
                            'sumvalue': float(row.get('sumvalue', sentiment_value))
                        }
                        
                    except (ValueError, TypeError):
                        # Skip invalid entries
                        continue
                        
            print(f"Loaded {len(self.lexicon)} words from custom lexical dictionary")
            
        except Exception as e:
            print(f"Error loading lexical dictionary: {e}")
            # Initialize empty lexicon as fallback
            self.lexicon = {}
    
    def _initialize_boosters(self) -> None:
        """Initialize intensity boosters (words that amplify sentiment)"""
        self.booster_dict = {
            "absolutely": self.B_INCR, "amazingly": self.B_INCR, "awfully": self.B_INCR,
            "completely": self.B_INCR, "considerably": self.B_INCR, "decidedly": self.B_INCR,
            "deeply": self.B_INCR, "effing": self.B_INCR, "enormously": self.B_INCR,
            "entirely": self.B_INCR, "especially": self.B_INCR, "exceptionally": self.B_INCR,
            "extremely": self.B_INCR, "fabulously": self.B_INCR, "flipping": self.B_INCR,
            "flippin": self.B_INCR, "fricking": self.B_INCR, "frickin": self.B_INCR,
            "frigging": self.B_INCR, "friggin": self.B_INCR, "fully": self.B_INCR,
            "fucking": self.B_INCR, "greatly": self.B_INCR, "hella": self.B_INCR,
            "highly": self.B_INCR, "hugely": self.B_INCR, "incredibly": self.B_INCR,
            "intensely": self.B_INCR, "majorly": self.B_INCR, "more": self.B_INCR,
            "most": self.B_INCR, "particularly": self.B_INCR, "purely": self.B_INCR,
            "quite": self.B_INCR, "really": self.B_INCR, "remarkably": self.B_INCR,
            "so": self.B_INCR, "substantially": self.B_INCR, "thoroughly": self.B_INCR,
            "totally": self.B_INCR, "tremendously": self.B_INCR, "uber": self.B_INCR,
            "unbelievably": self.B_INCR, "unusually": self.B_INCR, "utterly": self.B_INCR,
            "very": self.B_INCR, "really": self.B_INCR,
            
            # Decreasing boosters
            "almost": self.B_DECR, "barely": self.B_DECR, "hardly": self.B_DECR,
            "just enough": self.B_DECR, "kind of": self.B_DECR, "kinda": self.B_DECR,
            "kindof": self.B_DECR, "kind-of": self.B_DECR, "less": self.B_DECR,
            "little": self.B_DECR, "marginally": self.B_DECR, "occasionally": self.B_DECR,
            "partly": self.B_DECR, "scarcely": self.B_DECR, "slightly": self.B_DECR,
            "somewhat": self.B_DECR, "sort of": self.B_DECR, "sorta": self.B_DECR,
            "sortof": self.B_DECR, "sort-of": self.B_DECR
        }
    
    def _initialize_negations(self) -> None:
        """Initialize negation words"""
        self.negation_dict = {
            "aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", 
            "doesnt", "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", 
            "doesn't", "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", 
            "mustnt", "neither", "don't", "hadn't", "hasn't", "haven't", "isn't", 
            "mightn't", "mustn't", "neednt", "needn't", "never", "none", "nope", 
            "nor", "not", "nothing", "nowhere", "oughtnt", "shant", "shouldnt", 
            "uhuh", "wasnt", "werent", "oughtn't", "shan't", "shouldn't", "uh-uh", 
            "wasn't", "weren't", "without", "wont", "wouldnt", "won't", "wouldn't", 
            "rarely", "seldom", "despite", "hindi", "walang", "wala"  # Added some Filipino negations
        }
    
    def _prepare_text(self, text: str) -> List[str]:
        """
        Prepare tweet text for analysis by:
        1. Cleaning and normalizing
        2. Tokenizing
        3. Handling contractions and special cases
        """
        # Convert to lowercase
        text = text.lower()
        
        # Handle Twitter-specific elements
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'#(\w+)', r'\1', text)  # Convert hashtags to words
        
        # Handle contractions and common patterns
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "'s": " is"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove extra whitespace and punctuation (but keep some for context)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        words = text.split()
        
        return words
    
    def _sentiment_valence(self, word: str, words: List[str], idx: int) -> float:
        """
        Calculate sentiment valence for a word considering context
        """
        if word not in self.lexicon:
            return 0.0
        
        valence = self.lexicon[word]['sentiment']
        weight = self.lexicon[word]['weight']
        
        # Apply weight to sentiment
        valence = valence * weight
        
        # Check for booster words in context (preceding 3 words)
        start_i = max(0, idx - 3)
        booster_impact = 0
        
        for i in range(start_i, idx):
            if i < len(words):
                prev_word = words[i].lower()
                if prev_word in self.booster_dict:
                    booster_impact += self.booster_dict[prev_word]
        
        # Apply booster impact
        if valence > 0:
            valence += booster_impact
        else:
            valence -= booster_impact
        
        # Check for negation in context (preceding 3 words)
        for i in range(max(0, idx - 3), idx):
            if i < len(words) and words[i].lower() in self.negation_dict:
                valence *= self.N_SCALAR
                break
        
        return valence
    
    def _normalize_scores(self, pos_sum: float, neg_sum: float, neu_count: int, total_words: int) -> Tuple[float, float, float, float]:
        """
        Normalize sentiment scores using VADER's algorithm
        """
        if total_words == 0:
            return 0.0, 0.0, 1.0, 0.0
        
        # Calculate raw scores
        pos = abs(pos_sum)
        neg = abs(neg_sum)
        neu = neu_count
        
        # Normalize to sum to 1
        total = pos + neg + neu
        if total > 0:
            pos = pos / total
            neg = neg / total 
            neu = neu / total
        else:
            neu = 1.0
        
        # Calculate compound score
        compound_sum = pos_sum + neg_sum
        
        # Normalize compound score between -1 and 1
        if compound_sum != 0:
            compound = compound_sum / math.sqrt((compound_sum * compound_sum) + self.NORMALIZE_ALPHA)
        else:
            compound = 0.0
        
        # Ensure compound is bounded
        compound = max(-1.0, min(1.0, compound))
        
        return pos, neg, neu, compound
    
    def analyze_sentiment(self, text: str) -> SentimentScore:
        """
        Analyze sentiment of text using custom VADER implementation
        """
        if not text or not text.strip():
            return SentimentScore(0.0, 0.0, 1.0, 0.0, 'neutral')
        
        # Prepare text
        words = self._prepare_text(text)
        
        if not words:
            return SentimentScore(0.0, 0.0, 1.0, 0.0, 'neutral')
        
        # Calculate sentiment scores
        pos_sum = 0.0
        neg_sum = 0.0
        neu_count = 0
        
        for idx, word in enumerate(words):
            valence = self._sentiment_valence(word, words, idx)
            
            if valence > 0:
                pos_sum += valence
            elif valence < 0:
                neg_sum += valence
            else:
                neu_count += 1
        
        # Normalize scores
        pos, neg, neu, compound = self._normalize_scores(pos_sum, neg_sum, neu_count, len(words))
        
        # Determine classification based on compound score
        if compound >= 0.05:
            classification = 'positive'
        elif compound <= -0.05:
            classification = 'negative'
        else:
            classification = 'neutral'
        
        return SentimentScore(
            positive=round(pos, 3),
            negative=round(neg, 3),
            neutral=round(neu, 3),
            compound=round(compound, 3),
            classification=classification
        )
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentScore]:
        """
        Analyze sentiment for a batch of texts
        """
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
        
        positive_words = sum(1 for word_data in self.lexicon.values() 
                        if word_data['sentiment'] > 0)
        negative_words = sum(1 for word_data in self.lexicon.values() 
                        if word_data['sentiment'] < 0)
        neutral_words = sum(1 for word_data in self.lexicon.values() 
                        if word_data['sentiment'] == 0)
        
        avg_sentiment = sum(word_data['sentiment'] for word_data in self.lexicon.values()) / len(self.lexicon)
        avg_weight = sum(word_data['weight'] for word_data in self.lexicon.values()) / len(self.lexicon)
        
        # Find most positive and negative words
        most_positive = max(self.lexicon.items(), key=lambda x: x[1]['sentiment'])
        most_negative = min(self.lexicon.items(), key=lambda x: x[1]['sentiment'])
        
        return {
            "total_words": len(self.lexicon),
            "positive_words": positive_words,
            "negative_words": negative_words,
            "neutral_words": neutral_words,
            "average_sentiment": round(avg_sentiment, 4),
            "average_weight": round(avg_weight, 4),
            "most_positive_word": {
                "word": most_positive[0],
                "sentiment": most_positive[1]['sentiment']
            },
            "most_negative_word": {
                "word": most_negative[0],
                "sentiment": most_negative[1]['sentiment']
            }
        }

# Global instance
sentiment_analyzer = CustomVADER()

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
            "lexicon_loaded": True,
            "lexicon_path": sentiment_analyzer.lexicon_path,
            "lexicon_stats": stats
        }
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "lexicon_loaded": False
        }

def analyze_tweet_sentiment(tweet_text: str) -> Dict:
    """
    Analyze sentiment of a single tweet
    """
    try:
        if not tweet_text or not tweet_text.strip():
            return {"error": "Tweet text is empty"}
        
        sentiment_score = sentiment_analyzer.analyze_sentiment(tweet_text)
        
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
            "prepared_text": " ".join(sentiment_analyzer._prepare_text(tweet_text))
        }
    
    except Exception as e:
        return {"error": f"Sentiment analysis failed: {str(e)}"}

def analyze_batch_sentiment(tweets: List[str]) -> List[Dict]:
    """
    Analyze sentiment for a batch of tweets
    """
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
    """
    Analyze distribution of sentiment classifications in batch results
    """
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
        
        # If this result has climate category info, cross-reference
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

def _interpret_compound_score(compound: float) -> str:
    """
    Interpret compound score with descriptive labels
    """
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

def get_sentiment_insights(sentiment_results: List[Dict]) -> Dict:
    """
    Generate insights about sentiment patterns in the data
    """
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
    """
    Reload the lexical dictionary (useful if the file has been updated)
    """
    try:
        old_count = len(sentiment_analyzer.lexicon)
        sentiment_analyzer._load_custom_lexicon()
        new_count = len(sentiment_analyzer.lexicon)
        
        return {
            "status": "ok",
            "message": f"Lexicon reloaded: {old_count} -> {new_count} words",
            "lexicon_stats": sentiment_analyzer.get_lexicon_stats()
        }
    
    except Exception as e:
        return {"error": f"Failed to reload lexicon: {str(e)}"}

# Helper function to test specific words
def test_word_sentiment(word: str) -> Dict:
    """
    Test sentiment analysis for a specific word
    """
    word_lower = word.lower()
    
    if word_lower in sentiment_analyzer.lexicon:
        word_data = sentiment_analyzer.lexicon[word_lower]
        return {
            "word": word,
            "found": True,
            "sentiment": word_data['sentiment'],
            "weight": word_data['weight'],
            "sumvalue": word_data['sumvalue'],
            "interpretation": _interpret_compound_score(word_data['sentiment'])
        }
    else:
        return {
            "word": word,
            "found": False,
            "message": "Word not found in custom lexicon"
        }