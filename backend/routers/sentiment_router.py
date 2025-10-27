#sentiment_router.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from services.sentiment_analysis import (
    analyze_tweet_sentiment,
    analyze_batch_sentiment,
    sentiment_model_status,
    analyze_sentiment_distribution,
    get_sentiment_insights,
    reload_lexicon,
    test_word_sentiment
)

router = APIRouter()

class SentimentRequest(BaseModel):
    tweet: str

class BatchSentimentRequest(BaseModel):
    tweets: List[str]

class WordTestRequest(BaseModel):
    word: str

@router.post("/analyze-sentiment")
async def analyze_sentiment_endpoint(request: SentimentRequest):
    """Analyze sentiment of a single tweet using custom VADER implementation"""
    return analyze_tweet_sentiment(request.tweet)

@router.post("/analyze-batch-sentiment")
async def analyze_batch_sentiment_endpoint(request: BatchSentimentRequest):
    """Analyze sentiment for multiple tweets"""
    results = analyze_batch_sentiment(request.tweets)
    distribution = analyze_sentiment_distribution(results)
    
    return {
        "status": "ok",
        "total_tweets": len(request.tweets),
        "results": results,
        "distribution": distribution
    }

@router.get("/sentiment-model-status")
async def sentiment_model_status_endpoint():
    """Get detailed status of the sentiment analysis model"""
    return sentiment_model_status()

@router.post("/reload-lexicon")
async def reload_lexicon_endpoint():
    """Reload the lexical dictionary (useful for updates)"""
    return reload_lexicon()

@router.post("/test-word")
async def test_word_endpoint(request: WordTestRequest):
    """Test sentiment analysis for a specific word"""
    return test_word_sentiment(request.word)

@router.post("/sentiment-insights")
async def sentiment_insights_endpoint(results: dict):
    """Generate insights from sentiment analysis results"""
    if "results" not in results:
        return {"error": "Missing 'results' field in request body"}
    
    return get_sentiment_insights(results["results"])

@router.get("/lexicon-stats")
async def lexicon_stats_endpoint():
    """Get statistics about the loaded lexical dictionary"""
    from services.sentiment_analysis import sentiment_analyzer
    return sentiment_analyzer.get_lexicon_stats()