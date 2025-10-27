#analysis_router.py

from datetime import datetime, timedelta
from fastapi import APIRouter, UploadFile, File, Form, Depends, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from database.models import get_db
from database.db_service import TweetDatabaseService
from services.main_service import (
    analyze_single_tweet, 
    analyze_csv_tweets_file,
    analyze_complete_distribution,
    get_climate_tweets_only
)
from services.climate_category import categorize_tweet
from services.weather_forecast_service import weather_forecast_service

router = APIRouter()

class TweetRequest(BaseModel):
    tweet: str
    location: Optional[str] = None

class CategoryRequest(BaseModel):
    tweet: str

@router.post("/analyze-single-tweet")
async def analyze_single_tweet_endpoint(
    request: TweetRequest,
    db: Session = Depends(get_db)
):
    """
    Complete analysis of a single tweet with database storage:
    1. Climate relevance detection
    2. Climate category classification (for climate-related tweets)
    3. Weather consistency validation (if location provided)
    4. Sentiment analysis using custom VADER
    5. Store results in database
    """
    # Perform analysis using existing service
    result = analyze_single_tweet(request.tweet, request.location)
    
    # Save to database if analysis was successful
    if result.get("status") == "ok":
        try:
            saved_tweet = TweetDatabaseService.save_single_tweet(db, result)
            result["database_id"] = saved_tweet.id
            result["saved_to_db"] = True
        except Exception as e:
            print(f"Error saving to database: {str(e)}")
            result["saved_to_db"] = False
            result["db_error"] = str(e)
    
    return result

@router.post("/categorize-tweet")
async def categorize_tweet_endpoint(request: CategoryRequest):
    """
    Directly categorize a tweet (assumes it's climate-related)
    Useful if you already know the tweet is climate-related
    """
    result = categorize_tweet(request.tweet)
    return {
        "status": "ok" if "error" not in result else "error",
        "tweet": request.tweet,
        "categorization": result
    }

@router.post("/analyze-csv-tweets")
async def analyze_csv_tweets_endpoint(
    file: UploadFile = File(...), 
    location: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Complete analysis of a CSV file with database storage:
    1. Climate relevance detection
    2. Climate category classification (for climate-related tweets)
    3. Weather consistency validation (if location provided)
    4. Sentiment analysis for all tweets
    5. Store results in database
    """
    contents = await file.read()
    
    # Perform batch analysis using existing service
    result = analyze_csv_tweets_file(contents, file.filename, location)
    
    # Save to database if analysis was successful
    if result.get("status") == "ok":
        try:
            batch_upload = TweetDatabaseService.save_batch_tweets(db, result)
            result["batch_id"] = batch_upload.id
            result["saved_to_db"] = True
        except Exception as e:
            print(f"Error saving batch to database: {str(e)}")
            result["saved_to_db"] = False
            result["db_error"] = str(e)
    
    return result

@router.post("/analyze-results")
async def analyze_results_endpoint(results: dict):
    """Comprehensive analysis of processed results including sentiment"""
    if "results" not in results:
        return {"error": "Missing 'results' field in request body"}
    
    complete_analysis = analyze_complete_distribution(results["results"])
    climate_only = get_climate_tweets_only(results["results"])
    
    return {
        "status": "ok",
        "analysis": complete_analysis,
        "climate_tweets_count": len(climate_only),
        "sample_climate_tweets": climate_only[:5]  # First 5 for preview
    }

@router.post("/validate-weather-context")
async def validate_weather_context_endpoint(request: TweetRequest):
    """Validate weather context for a single tweet"""
    if not request.location:
        return {"error": "Location is required for weather validation"}
    
    result = weather_forecast_service.analyze_tweet_weather_context(
        request.tweet, 
        request.location
    )
    return result

@router.post("/analyze-sentiment-by-category")
async def analyze_sentiment_by_category_endpoint(results: dict):
    """Analyze sentiment distribution across different climate categories"""
    if "results" not in results:
        return {"error": "Missing 'results' field in request body"}
    
    climate_tweets = get_climate_tweets_only(results["results"])
    
    if not climate_tweets:
        return {"error": "No climate-related tweets found in results"}
    
    category_sentiment = {}
    
    for tweet in climate_tweets:
        if ("category_classification" in tweet and 
            "sentiment_analysis" in tweet and
            "prediction" in tweet["category_classification"] and
            "sentiment" in tweet["sentiment_analysis"]):
            
            category = tweet["category_classification"]["prediction"]
            sentiment = tweet["sentiment_analysis"]["sentiment"]
            
            if category not in category_sentiment:
                category_sentiment[category] = {
                    "tweets": [],
                    "positive_count": 0,
                    "negative_count": 0,
                    "neutral_count": 0,
                    "avg_compound": 0,
                    "compound_scores": []
                }
            
            classification = sentiment["classification"]
            compound = sentiment["compound"]
            
            category_sentiment[category]["tweets"].append({
                "tweet": tweet["tweet"],
                "sentiment": sentiment,
                "compound": compound
            })
            category_sentiment[category][f"{classification}_count"] += 1
            category_sentiment[category]["compound_scores"].append(compound)
    
    # Calculate averages and insights
    for category, data in category_sentiment.items():
        if data["compound_scores"]:
            data["avg_compound"] = sum(data["compound_scores"]) / len(data["compound_scores"])
            data["total_tweets"] = len(data["tweets"])
            
            # Calculate percentages
            total = data["total_tweets"]
            data["positive_percentage"] = round((data["positive_count"] / total) * 100, 2)
            data["negative_percentage"] = round((data["negative_count"] / total) * 100, 2)
            data["neutral_percentage"] = round((data["neutral_count"] / total) * 100, 2)
            
            # Clean up response
            del data["compound_scores"]
            data["sample_tweets"] = data["tweets"][:3]
            del data["tweets"]
    
    return {
        "status": "ok",
        "sentiment_by_category": category_sentiment,
        "total_climate_tweets": len(climate_tweets),
        "categories_analyzed": len(category_sentiment)
    }

# Observations endpoints
@router.get("/observations")
async def get_observations_endpoint(
    db: Session = Depends(get_db),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    location: Optional[str] = Query(None),
    days: Optional[int] = Query(30)  # Default to last 30 days
):
    """
    Get observations data for dashboard visualization
    
    Query parameters:
    - start_date: Start date for filtering (ISO format)
    - end_date: End date for filtering (ISO format)  
    - location: Filter by specific location
    - days: Number of days to look back (if no dates provided)
    """
    # Handle date parameters
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=days)
    
    observations_data = TweetDatabaseService.get_observations_data(
        db, start_date, end_date, location
    )
    
    return {
        "status": "ok",
        "data": observations_data
    }

@router.get("/observations/categories")
async def get_category_breakdown(
    db: Session = Depends(get_db),
    location: Optional[str] = Query(None),
    days: int = Query(30)
):
    """Get detailed category breakdown with sentiment analysis"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = TweetDatabaseService.get_observations_data(db, start_date, end_date, location)
    
    # Format for frontend chart
    categories = []
    for category, count in data["climate_categories"]["distribution"].items():
        sentiment_data = data["climate_categories"]["sentiments_by_category"].get(category, {})
        categories.append({
            "name": category,
            "total": count,
            "positive": sentiment_data.get("positive", 0),
            "neutral": sentiment_data.get("neutral", 0),
            "negative": sentiment_data.get("negative", 0)
        })
    
    # Sort by total count
    categories.sort(key=lambda x: x["total"], reverse=True)
    
    return {
        "status": "ok",
        "categories": categories[:10],  # Top 10 categories
        "total_categories": len(data["climate_categories"]["distribution"])
    }

@router.get("/observations/sentiment")
async def get_sentiment_distribution(
    db: Session = Depends(get_db),
    location: Optional[str] = Query(None),
    days: int = Query(30)
):
    """Get sentiment distribution data for pie chart"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = TweetDatabaseService.get_observations_data(db, start_date, end_date, location)
    
    return {
        "status": "ok",
        "sentiment": data["sentiment_distribution"],
        "dominant": data["dominant_sentiment"],
        "total_analyzed": sum(data["sentiment_distribution"].values())
    }

@router.get("/observations/trends")
async def get_climate_trends(
    db: Session = Depends(get_db),
    location: Optional[str] = Query(None),
    days: int = Query(30)
):
    """Get climate trends data for area chart"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = TweetDatabaseService.get_observations_data(db, start_date, end_date, location)
    
    return {
        "status": "ok",
        "trends": data["climate_trends"],
        "period": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "days": days
        }
    }

@router.get("/observations/locations")
async def get_location_stats(
    db: Session = Depends(get_db),
    days: int = Query(30)
):
    """Get location statistics"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = TweetDatabaseService.get_observations_data(db, start_date, end_date)
    
    return {
        "status": "ok",
        "locations": data["location_stats"]["distribution"],
        "most_active": data["location_stats"]["most_active"]
    }

@router.get("/observations/summary")
async def get_observations_summary(
    db: Session = Depends(get_db),
    location: Optional[str] = Query(None),
    days: int = Query(30)
):
    """Get summary statistics for the observations dashboard"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = TweetDatabaseService.get_observations_data(db, start_date, end_date, location)
    
    # Format the response for dashboard cards
    return {
        "status": "ok",
        "framework_accuracy": f"{data['framework_accuracy']}%",
        "dominant_sentiment": data["dominant_sentiment"],
        "most_active_location": data["location_stats"]["most_active"],
        "climate_categories": {
            "most_positive": data["climate_categories"]["most_positive"],
            "most_negative": data["climate_categories"]["most_negative"],
            "total": len(data["climate_categories"]["distribution"])
        },
        "period": {
            "days": days,
            "total_tweets": data["summary"]["total_climate_tweets"]
        }
    }

@router.get("/recent-tweets")
async def get_recent_tweets_endpoint(
    db: Session = Depends(get_db),
    limit: int = Query(10),
    climate_only: bool = Query(True)
):
    """Get recently analyzed tweets"""
    tweets = TweetDatabaseService.get_recent_tweets(db, limit, climate_only)
    
    return {
        "status": "ok",
        "tweets": tweets,
        "count": len(tweets)
    }