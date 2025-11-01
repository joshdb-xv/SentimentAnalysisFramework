"""
Climate Change Tweet Sentiment Analysis using VADER
Analyzes sentiment of climate-related tweets and social media posts
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from typing import Dict, List
import re

class ClimateSentimentAnalyzer:
    """A class to perform sentiment analysis on climate-related tweets using VADER."""
    
    def __init__(self):
        """Initialize the VADER sentiment analyzer."""
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Climate-related keywords for categorization
        self.climate_keywords = {
            'action': ['action', 'act now', 'change', 'solution', 'renewable', 'sustainable', 
                      'green energy', 'carbon neutral', 'net zero'],
            'crisis': ['crisis', 'emergency', 'disaster', 'catastrophe', 'warming', 'melting',
                      'extinction', 'threat'],
            'policy': ['policy', 'legislation', 'government', 'agreement', 'treaty', 'law',
                      'regulation', 'COP', 'Paris Agreement'],
            'science': ['science', 'research', 'data', 'study', 'evidence', 'scientists',
                       'report', 'IPCC']
        }
    
    def preprocess_tweet(self, tweet: str) -> str:
        """
        Preprocess tweet text for better analysis.
        
        Args:
            tweet: Raw tweet text
            
        Returns:
            Cleaned tweet text
        """
        # Remove URLs
        tweet = re.sub(r'http\S+|www.\S+', '', tweet)
        # Remove @mentions (but keep hashtags as they carry sentiment)
        tweet = re.sub(r'@\w+', '', tweet)
        # Clean up extra whitespace
        tweet = ' '.join(tweet.split())
        return tweet
    
    def categorize_tweet(self, tweet: str) -> List[str]:
        """
        Categorize tweet based on climate keywords.
        
        Args:
            tweet: Tweet text
            
        Returns:
            List of categories the tweet falls into
        """
        tweet_lower = tweet.lower()
        categories = []
        
        for category, keywords in self.climate_keywords.items():
            if any(keyword in tweet_lower for keyword in keywords):
                categories.append(category)
        
        return categories if categories else ['general']
    
    def analyze_tweet(self, tweet: str, preprocess: bool = True) -> Dict:
        """
        Analyze sentiment of a single climate tweet.
        
        Args:
            tweet: The tweet text to analyze
            preprocess: Whether to preprocess the tweet
            
        Returns:
            Dictionary containing sentiment scores and classification
        """
        original_tweet = tweet
        if preprocess:
            tweet = self.preprocess_tweet(tweet)
        
        scores = self.analyzer.polarity_scores(tweet)
        
        # Classify based on compound score
        if scores['compound'] >= 0.05:
            sentiment = 'Positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        
        categories = self.categorize_tweet(tweet)
        
        return {
            'tweet': original_tweet,
            'cleaned_tweet': tweet if preprocess else original_tweet,
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'positive': scores['pos'],
            'compound': scores['compound'],
            'sentiment': sentiment,
            'categories': ', '.join(categories)
        }
    
    def analyze_batch(self, tweets: List[str], preprocess: bool = True) -> pd.DataFrame:
        """
        Analyze sentiment of multiple climate tweets.
        
        Args:
            tweets: List of tweets to analyze
            preprocess: Whether to preprocess tweets
            
        Returns:
            DataFrame with sentiment analysis results
        """
        results = [self.analyze_tweet(tweet, preprocess) for tweet in tweets]
        return pd.DataFrame(results)
    
    def get_climate_sentiment_summary(self, tweets: List[str]) -> Dict:
        """
        Get summary statistics for climate tweets.
        
        Args:
            tweets: List of tweets to analyze
            
        Returns:
            Dictionary with summary statistics
        """
        df = self.analyze_batch(tweets)
        
        summary = {
            'total_tweets': len(tweets),
            'positive_count': len(df[df['sentiment'] == 'Positive']),
            'negative_count': len(df[df['sentiment'] == 'Negative']),
            'neutral_count': len(df[df['sentiment'] == 'Neutral']),
            'avg_compound_score': df['compound'].mean(),
            'most_positive_tweet': df.loc[df['compound'].idxmax()]['tweet'],
            'most_negative_tweet': df.loc[df['compound'].idxmin()]['tweet']
        }
        
        # Category breakdown
        category_counts = {}
        for categories in df['categories']:
            for cat in categories.split(', '):
                category_counts[cat] = category_counts.get(cat, 0) + 1
        
        summary['category_breakdown'] = category_counts
        
        return summary


def main():
    """Main function to demonstrate climate tweet sentiment analysis."""
    
    # Initialize analyzer
    climate_analyzer = ClimateSentimentAnalyzer()
    
    # Sample climate-related tweets
    climate_tweets = [
        "We need immediate climate action NOW! Our planet is burning and we can't wait any longer. #ClimateEmergency",
        "Excited to see renewable energy investments reaching record highs! Solar and wind power are the future 🌞💨 #CleanEnergy",
        "Another devastating wildfire season. Climate change is destroying communities and lives. When will we act?",
        "New IPCC report shows we still have time to limit warming to 1.5°C if we act decisively. Science gives us hope! #ClimateScience",
        "Politicians keep making empty promises while the planet burns. Greenwashing at its finest. #ClimateAction",
        "Just installed solar panels on my roof! Small steps towards carbon neutrality. Everyone can make a difference! ☀️",
        "The Paris Agreement is failing. We need stronger policy and enforcement NOW. #ClimatePolicy",
        "Love seeing more companies commit to net zero emissions! Corporate action on climate is finally happening. #Sustainability",
        "Glaciers melting at unprecedented rates. The science is clear and terrifying. We're running out of time.",
        "Amazing to see youth climate activists leading the charge! The future generation won't accept inaction. #FridaysForFuture",
        "Climate change is the greatest threat to humanity. We need global cooperation and immediate action.",
        "Renewable energy is now cheaper than fossil fuels in most markets. Economics AND environment align! #GreenEnergy",
        "Another climate summit, another round of broken promises. Actions speak louder than words. #COP28",
        "Planting 1 million trees in our community! Local climate action makes a real difference 🌳 #TreePlanting",
        "Extreme weather events are the new normal. Climate crisis is here, it's real, and it's devastating."
    ]
    
    print("=" * 90)
    print("CLIMATE CHANGE TWEET SENTIMENT ANALYSIS using VADER")
    print("=" * 90)
    
    # Analyze individual tweets
    print("\n1. SAMPLE TWEET ANALYSIS (First 3 tweets)")
    print("-" * 90)
    
    for i, tweet in enumerate(climate_tweets[:3], 1):
        result = climate_analyzer.analyze_tweet(tweet)
        print(f"\nTweet {i}: {result['tweet']}")
        print(f"Categories: {result['categories']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Scores - Neg: {result['negative']:.3f}, Neu: {result['neutral']:.3f}, "
              f"Pos: {result['positive']:.3f}, Compound: {result['compound']:.3f}")
    
    # Batch analysis
    print("\n\n2. BATCH ANALYSIS OF ALL CLIMATE TWEETS")
    print("-" * 90)
    
    df_results = climate_analyzer.analyze_batch(climate_tweets)
    
    # Display results in a clean format
    display_df = df_results[['tweet', 'sentiment', 'compound', 'categories']].copy()
    display_df['tweet'] = display_df['tweet'].str[:60] + '...'  # Truncate for display
    print("\n", display_df.to_string(index=False))
    
    # Summary statistics
    print("\n\n3. CLIMATE SENTIMENT SUMMARY")
    print("-" * 90)
    
    summary = climate_analyzer.get_climate_sentiment_summary(climate_tweets)
    print(f"\nTotal Tweets Analyzed: {summary['total_tweets']}")
    print(f"\nSentiment Distribution:")
    print(f"  Positive: {summary['positive_count']} ({summary['positive_count']/summary['total_tweets']*100:.1f}%)")
    print(f"  Negative: {summary['negative_count']} ({summary['negative_count']/summary['total_tweets']*100:.1f}%)")
    print(f"  Neutral:  {summary['neutral_count']} ({summary['neutral_count']/summary['total_tweets']*100:.1f}%)")
    print(f"\nAverage Compound Score: {summary['avg_compound_score']:.3f}")
    
    print(f"\nCategory Breakdown:")
    for category, count in sorted(summary['category_breakdown'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {category.capitalize()}: {count} tweets")
    
    print(f"\nMost Positive Tweet: {summary['most_positive_tweet'][:100]}...")
    print(f"\nMost Negative Tweet: {summary['most_negative_tweet'][:100]}...")
    
    # Sentiment distribution visualization
    print("\n\n4. SENTIMENT DISTRIBUTION VISUALIZATION")
    print("-" * 90)
    
    sentiment_counts = df_results['sentiment'].value_counts()
    max_count = sentiment_counts.max()
    
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        if sentiment in sentiment_counts:
            count = sentiment_counts[sentiment]
            bar_length = int((count / max_count) * 40)
            bar = '█' * bar_length
            print(f"{sentiment:8s}: {bar} ({count})")
    
    # Compound score distribution
    print("\n\n5. COMPOUND SCORE DISTRIBUTION")
    print("-" * 90)
    
    print("\nHighly Positive (>0.5):", len(df_results[df_results['compound'] > 0.5]))
    print("Moderately Positive (0.05 to 0.5):", len(df_results[(df_results['compound'] >= 0.05) & (df_results['compound'] <= 0.5)]))
    print("Neutral (-0.05 to 0.05):", len(df_results[(df_results['compound'] > -0.05) & (df_results['compound'] < 0.05)]))
    print("Moderately Negative (-0.5 to -0.05):", len(df_results[(df_results['compound'] >= -0.5) & (df_results['compound'] <= -0.05)]))
    print("Highly Negative (<-0.5):", len(df_results[df_results['compound'] < -0.5]))
    
    print("\n" + "=" * 90)
    print("\nInsights:")
    print("- Climate tweets often contain mixed sentiment (urgency + hope)")
    print("- Action-oriented tweets tend to be more positive")
    print("- Crisis/emergency framing typically shows negative sentiment")
    print("- Policy discussions show varied sentiment depending on perceived effectiveness")
    print("=" * 90)


if __name__ == "__main__":
    # Installation note
    print("\nRequired packages:")
    print("  pip install vaderSentiment pandas")
    print()
    
    main()