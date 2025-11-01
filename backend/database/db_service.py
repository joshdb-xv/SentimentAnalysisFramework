# database/db_service.py

from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_
from datetime import datetime, timedelta, date
import json
from .models import Tweet, BatchUpload, DailyStats, get_db, init_db

class TweetDatabaseService:
    """Service class for database operations"""
    
    @staticmethod
    def save_single_tweet(db: Session, analysis_result: Dict) -> Tweet:
        """Save a single analyzed tweet to database"""
        
        tweet = Tweet(
            tweet_text=analysis_result.get("tweet"),
            location=analysis_result.get("location"),
            length=analysis_result.get("length"),
            
            # Climate classification
            is_climate_related=analysis_result.get("climate_classification", {}).get("is_climate_related", False),
            climate_confidence=analysis_result.get("climate_classification", {}).get("confidence", 0.0),
            climate_prediction=analysis_result.get("climate_classification", {}).get("prediction", 0),
            
            # Weather flag
            weather_flag=analysis_result.get("weather_flag"),
            
            # Sentiment flag
            sentiment_flag=analysis_result.get("sentiment_flag"),
            
            # Store full response for reference
            full_response=analysis_result
        )
        
        # If climate-related, store additional details
        if tweet.is_climate_related:
            # Category classification
            category_data = analysis_result.get("category_classification", {})
            if not category_data.get("skipped"):
                tweet.category = category_data.get("prediction")
                tweet.category_confidence = category_data.get("confidence")
                tweet.category_probabilities = category_data.get("probabilities")
            
            # Weather validation
            weather_data = analysis_result.get("weather_validation", {})
            if not weather_data.get("skipped") and weather_data.get("status") == "success":
                validation = weather_data.get("validation", {})
                tweet.weather_consistency_score = validation.get("consistency_score")
                tweet.weather_data = weather_data.get("weather_data")
            
            # Sentiment analysis
            sentiment_data = analysis_result.get("sentiment_analysis", {})
            if not sentiment_data.get("skipped") and sentiment_data.get("status") == "ok":
                sentiment = sentiment_data.get("sentiment", {})
                tweet.sentiment_positive = sentiment.get("positive")
                tweet.sentiment_negative = sentiment.get("negative")
                tweet.sentiment_neutral = sentiment.get("neutral")
                tweet.sentiment_compound = sentiment.get("compound")
        
        db.add(tweet)
        db.commit()
        db.refresh(tweet)
        
        # Update daily stats
        TweetDatabaseService._update_daily_stats(db, date.today())
        
        return tweet
    
    @staticmethod
    def save_batch_tweets(db: Session, batch_result: Dict) -> BatchUpload:
        """Save batch analysis results to database"""
        
        # Create batch upload record
        batch = BatchUpload(
            filename=batch_result.get("filename"),
            location=batch_result.get("location"),
            total_tweets=batch_result.get("summary", {}).get("total", 0),
            climate_tweets=batch_result.get("summary", {}).get("climate", 0),
            processed_tweets=len(batch_result.get("results", []))
        )
        
        db.add(batch)
        db.commit()
        db.refresh(batch)
        
        # Save individual tweets
        for tweet_result in batch_result.get("results", []):
            tweet = Tweet(
                tweet_text=tweet_result.get("tweet"),
                location=tweet_result.get("location"),
                length=tweet_result.get("length"),
                batch_id=batch.id,
                
                # Climate classification
                is_climate_related=tweet_result.get("climate_classification", {}).get("is_climate_related", False),
                climate_confidence=tweet_result.get("climate_classification", {}).get("confidence", 0.0),
                climate_prediction=tweet_result.get("climate_classification", {}).get("prediction", 0),
                
                # Weather flag
                weather_flag=tweet_result.get("weather_flag"),
                
                # Sentiment flag
                sentiment_flag=tweet_result.get("sentiment_flag"),
                
                # Store full response
                full_response=tweet_result
            )
            
            # If climate-related, store additional details
            if tweet.is_climate_related:
                # Category classification
                category_data = tweet_result.get("category_classification", {})
                if category_data and not category_data.get("skipped"):
                    tweet.category = category_data.get("prediction")
                    tweet.category_confidence = category_data.get("confidence")
                    tweet.category_probabilities = category_data.get("probabilities")
                
                # Weather validation
                weather_data = tweet_result.get("weather_validation", {})
                if weather_data and not weather_data.get("skipped") and weather_data.get("status") == "success":
                    validation = weather_data.get("validation", {})
                    tweet.weather_consistency_score = validation.get("consistency_score")
                    tweet.weather_data = weather_data.get("weather_data")
                
                # Sentiment analysis
                sentiment_data = tweet_result.get("sentiment_analysis", {})
                if sentiment_data and not sentiment_data.get("skipped") and sentiment_data.get("status") == "ok":
                    sentiment = sentiment_data.get("sentiment", {})
                    tweet.sentiment_positive = sentiment.get("positive")
                    tweet.sentiment_negative = sentiment.get("negative")
                    tweet.sentiment_neutral = sentiment.get("neutral")
                    tweet.sentiment_compound = sentiment.get("compound")
            
            db.add(tweet)
        
        db.commit()
        
        # Update daily stats
        TweetDatabaseService._update_daily_stats(db, date.today())
        
        return batch
    
    @staticmethod
    def get_observations_data(db: Session, 
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None,
                             location: Optional[str] = None) -> Dict:
        """Get data for observations page"""
        
        # Default to last 30 days if no date range specified
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # Base query for climate-related tweets only
        query = db.query(Tweet).filter(
            Tweet.is_climate_related == True,
            Tweet.analyzed_at >= start_date,
            Tweet.analyzed_at <= end_date
        )

        if location:
          query = query.filter(func.lower(func.trim(Tweet.location)) == func.lower(location.strip()))
        
        tweets = query.all()
        
        # Calculate statistics
        total_tweets = len(tweets)
        
        # Category distribution
        category_counts = {}
        category_sentiments = {}
        
        for tweet in tweets:
            if tweet.category:
                # Count categories
                if tweet.category not in category_counts:
                    category_counts[tweet.category] = 0
                    category_sentiments[tweet.category] = {
                        "positive": 0, "neutral": 0, "negative": 0, "total": 0
                    }
                
                category_counts[tweet.category] += 1
                
                # Track sentiment by category
                if tweet.sentiment_flag and tweet.sentiment_flag != "Not Climate Related":
                    category_sentiments[tweet.category]["total"] += 1
                    if tweet.sentiment_flag.lower() == "positive":
                        category_sentiments[tweet.category]["positive"] += 1
                    elif tweet.sentiment_flag.lower() == "negative":
                        category_sentiments[tweet.category]["negative"] += 1
                    else:
                        category_sentiments[tweet.category]["neutral"] += 1
        
        # Overall sentiment distribution
        sentiment_dist = {"positive": 0, "negative": 0, "neutral": 0}
        for tweet in tweets:
            if tweet.sentiment_flag and tweet.sentiment_flag != "Not Climate Related":
                sentiment_key = tweet.sentiment_flag.lower()
                if sentiment_key in sentiment_dist:
                    sentiment_dist[sentiment_key] += 1
        
        # Weather consistency stats
        weather_stats = {"consistent": 0, "inconsistent": 0, "unknown": 0}
        for tweet in tweets:
            if tweet.weather_flag:
                flag = tweet.weather_flag.lower()
                if flag == "consistent":
                    weather_stats["consistent"] += 1
                elif flag == "inconsistent":
                    weather_stats["inconsistent"] += 1
                elif flag not in ["not climate related", "no location", "unavailable"]:
                    weather_stats["unknown"] += 1
        
        # Location distribution
        location_counts = {}
        for tweet in tweets:
            if tweet.location:
                if tweet.location not in location_counts:
                    location_counts[tweet.location] = 0
                location_counts[tweet.location] += 1
        
        # Sort locations by count and get top location
        sorted_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)
        most_active_location = sorted_locations[0] if sorted_locations else (None, 0)
        
        # Calculate accuracy (weather consistency)
        total_validated = weather_stats["consistent"] + weather_stats["inconsistent"]
        accuracy = (weather_stats["consistent"] / total_validated * 100) if total_validated > 0 else 0
        
        # Determine dominant sentiment
        dominant_sentiment = max(sentiment_dist, key=sentiment_dist.get) if sentiment_dist else "neutral"
        
        # Get trend data for chart
        trend_data = TweetDatabaseService._get_climate_trends(db, start_date, end_date, location)
        
        # Find most positive and negative categories
        most_positive_cat = None
        most_negative_cat = None
        
        if category_sentiments:
            # Calculate percentage for each category
            for cat, sentiments in category_sentiments.items():
                if sentiments["total"] > 0:
                    sentiments["positive_pct"] = sentiments["positive"] / sentiments["total"]
                    sentiments["negative_pct"] = sentiments["negative"] / sentiments["total"]
            
            # Find extremes
            valid_cats = [(cat, data) for cat, data in category_sentiments.items() if data["total"] > 0]
            if valid_cats:
                most_positive_cat = max(valid_cats, key=lambda x: x[1]["positive_pct"])[0]
                most_negative_cat = max(valid_cats, key=lambda x: x[1]["negative_pct"])[0]
        
        return {
            "summary": {
                "total_climate_tweets": total_tweets,
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                }
            },
            "climate_categories": {
                "distribution": category_counts,
                "sentiments_by_category": category_sentiments,
                "most_positive": most_positive_cat,
                "most_negative": most_negative_cat
            },
            "sentiment_distribution": sentiment_dist,
            "dominant_sentiment": dominant_sentiment.upper(),
            "weather_consistency": weather_stats,
            "framework_accuracy": round(accuracy, 1),
            "location_stats": {
                "distribution": location_counts,
                "most_active": {
                    "location": most_active_location[0],
                    "count": most_active_location[1],
                    "percentage": round(most_active_location[1] / total_tweets * 100, 1) if total_tweets > 0 else 0
                }
            },
            "climate_trends": trend_data
        }
    
    @staticmethod
    def _get_climate_trends(db: Session, start_date: datetime, end_date: datetime, location: Optional[str] = None) -> List[Dict]:
        """Get daily climate tweet trends for visualization"""
        
        try:
            # Query daily counts grouped by category
            query = db.query(
                func.date(Tweet.analyzed_at).label('date'),
                Tweet.category,
                func.count(Tweet.id).label('count')
            ).filter(
                Tweet.is_climate_related == True,
                Tweet.analyzed_at >= start_date,
                Tweet.analyzed_at <= end_date,
                Tweet.category.isnot(None)
            )
            
            # Add location filter if provided
            if location:
                query = query.filter(func.lower(func.trim(Tweet.location)) == func.lower(location.strip()))
            
            daily_data = query.group_by(
                func.date(Tweet.analyzed_at),
                Tweet.category
            ).all()
            
            # Organize data by date
            trend_dict = {}
            for row in daily_data:
                # Fix: Handle both string and date objects
                if isinstance(row.date, str):
                    date_str = row.date  # Already a string
                else:
                    # It's a date/datetime object, format it
                    date_str = row.date.strftime('%Y-%m-%d')
                
                if date_str not in trend_dict:
                    trend_dict[date_str] = {}
                trend_dict[date_str][row.category] = row.count
            
            # Convert to list format for frontend
            trend_list = []
            for date_str, categories in trend_dict.items():
                trend_list.append({
                    "date": date_str,
                    "categories": categories,
                    "total": sum(categories.values())
                })
            
            return sorted(trend_list, key=lambda x: x["date"])
            
        except Exception as e:
            print(f"Error in _get_climate_trends: {str(e)}")
            return []
    
    @staticmethod
    def _update_daily_stats(db: Session, target_date: date):
        """Update or create daily statistics"""
        
        # Get or create daily stats record
        daily_stat = db.query(DailyStats).filter(
            func.date(DailyStats.date) == target_date
        ).first()
        
        if not daily_stat:
            daily_stat = DailyStats(date=datetime.combine(target_date, datetime.min.time()))
            db.add(daily_stat)
        
        # Calculate stats for the day
        day_tweets = db.query(Tweet).filter(
            func.date(Tweet.analyzed_at) == target_date
        ).all()
        
        daily_stat.total_tweets = len(day_tweets)
        daily_stat.climate_tweets = sum(1 for t in day_tweets if t.is_climate_related)
        daily_stat.non_climate_tweets = daily_stat.total_tweets - daily_stat.climate_tweets
        
        # Category distribution
        cat_dist = {}
        for tweet in day_tweets:
            if tweet.category:
                cat_dist[tweet.category] = cat_dist.get(tweet.category, 0) + 1
        daily_stat.category_distribution = cat_dist
        
        # Weather consistency
        daily_stat.weather_consistent = sum(1 for t in day_tweets if t.weather_flag == "Consistent")
        daily_stat.weather_inconsistent = sum(1 for t in day_tweets if t.weather_flag == "Inconsistent")
        daily_stat.weather_unknown = sum(1 for t in day_tweets if t.weather_flag == "Unknown")
        
        # Sentiment stats
        daily_stat.sentiment_positive = sum(1 for t in day_tweets if t.sentiment_flag == "Positive")
        daily_stat.sentiment_negative = sum(1 for t in day_tweets if t.sentiment_flag == "Negative")
        daily_stat.sentiment_neutral = sum(1 for t in day_tweets if t.sentiment_flag == "Neutral")
        
        # Average compound score
        compound_scores = [t.sentiment_compound for t in day_tweets if t.sentiment_compound is not None]
        if compound_scores:
            daily_stat.avg_compound_score = sum(compound_scores) / len(compound_scores)
        
        # Location distribution
        loc_dist = {}
        for tweet in day_tweets:
            if tweet.location:
                loc_dist[tweet.location] = loc_dist.get(tweet.location, 0) + 1
        daily_stat.location_distribution = loc_dist
        
        db.commit()
    
    @staticmethod
    def get_recent_tweets(db: Session, limit: int = 10, climate_only: bool = True) -> List[Dict]:
        """Get recent analyzed tweets"""
        
        query = db.query(Tweet)
        
        if climate_only:
            query = query.filter(Tweet.is_climate_related == True)
        
        tweets = query.order_by(desc(Tweet.analyzed_at)).limit(limit).all()
        
        return [
            {
                "id": t.id,
                "tweet": t.tweet_text,
                "location": t.location,
                "category": t.category,
                "sentiment": t.sentiment_flag,
                "weather_consistency": t.weather_flag,
                "analyzed_at": t.analyzed_at.isoformat() if t.analyzed_at else None
            }
            for t in tweets
        ]