# services/main_service.py

import csv
import io
from typing import Dict, List, Optional

from .climate_classifier import classify_tweet, classify_many, model_status
from .climate_category import (categorize_tweet, categorize_many_tweets, category_model_status,get_category_insights,CLIMATE_CATEGORIES)
from .weatherapi import weather_validator
from .sentiment_analysis import (
    analyze_tweet_sentiment, 
    analyze_batch_sentiment, 
    sentiment_model_status,
    analyze_sentiment_distribution,
    get_sentiment_insights
)

POSSIBLE_TEXT_COLUMNS = ["tweet", "text", "content", "message"]

def get_model_status() -> Dict:
    """Get status of all models"""
    return {
        "climate_relevance_model": model_status(),
        "climate_category_model": category_model_status(),
        "sentiment_analysis_model": sentiment_model_status()
    }

# -----------------------------
# Single tweet path
# -----------------------------
def analyze_single_tweet(tweet_text: str, location: Optional[str] = None) -> Dict:
    """
    Four-step analysis:
    1. Check if tweet is climate-related (binary classification)
    2. If climate-related, categorize into specific climate domain
    3. If location provided, validate against actual weather data
    4. Analyze sentiment using custom VADER with lexical dictionary + confidence scoring
    """
    # Step 1: Climate relevance check
    classification = classify_tweet(tweet_text)

    print("=" * 50)
    print("RECEIVED SINGLE TWEET:")
    print(f"Tweet: {tweet_text}")
    print(f"Location: {location}")
    print(f"Length: {len(tweet_text)} characters")
    print(f"Climate Classification: {classification}")

    base = {
        "status": "ok" if "error" not in classification else "error",
        "message": "Tweet analyzed for climate relevance, category, weather consistency, and sentiment.",
        "tweet": tweet_text,
        "location": location,
        "length": len(tweet_text),
        "climate_classification": classification
    }

    # If there's an error in step 1, return early
    if "error" in classification:
        print("=" * 50)
        return base

    # Step 2: If not climate-related, stop here and return early
    is_climate_related = classification.get("is_climate_related", False)
    
    if not is_climate_related:
        print("Tweet is not climate-related, stopping analysis pipeline.")
        base["category_classification"] = {
            "skipped": True,
            "reason": "Tweet not identified as climate-related"
        }
        base["weather_validation"] = {
            "skipped": True,
            "reason": "Tweet not identified as climate-related"
        }
        base["weather_flag"] = "Not Climate Related"
        base["sentiment_analysis"] = {
            "skipped": True,
            "reason": "Tweet not identified as climate-related"
        }
        base["sentiment_flag"] = "Not Climate Related"
        print("=" * 50)
        return base

    # Continue with Steps 2-4 only if climate-related
    print("Tweet is climate-related, proceeding to categorization...")
    
    # Step 2: Category classification
    categorization = categorize_tweet(tweet_text)
    base["category_classification"] = categorization
    
    if "error" not in categorization:
        print(f"Category: {categorization.get('prediction', 'unknown')}")
        print(f"Confidence: {categorization.get('confidence', 0):.3f}")
    else:
        print(f"Categorization error: {categorization.get('error', 'unknown error')}")

    # Step 3: Weather validation (if location is provided)
    weather_data = None
    if location:
        print(f"Location provided ({location}), proceeding to weather validation...")
        weather_analysis = weather_validator.analyze_tweet_weather_context(tweet_text, location)
        base["weather_validation"] = weather_analysis
        
        if weather_analysis.get("status") == "success":
            validation = weather_analysis.get("validation", {})
            consistency = validation.get("consistency", "unknown")
            weather_data = weather_analysis.get("weather_data")  # Extract weather data for context
            print(f"Weather consistency: {consistency}")
            
            # Add a simple flag for easier frontend usage
            if consistency == "consistent":
                base["weather_flag"] = "Consistent"
            elif consistency == "inconsistent":
                base["weather_flag"] = "Inconsistent"
            else:
                base["weather_flag"] = "Unknown"
        else:
            print("Weather validation failed or unavailable")
            base["weather_flag"] = "Unavailable"
    else:
        print("No location provided, skipping weather validation.")
        base["weather_validation"] = {
            "skipped": True,
            "reason": "No location provided"
        }
        base["weather_flag"] = "No Location"

    # Step 4: Sentiment Analysis WITH CONTEXT
    print("Proceeding to sentiment analysis with context...")
    
    # 🔥 BUILD CONTEXT FROM PREVIOUS STEPS
    context = {
        'location': location,
        'climate_category': categorization.get('prediction'),
        'weather_data': weather_data,
        'confidence': classification.get('confidence')
    }
    
    print(f"📦 Context for sentiment analysis: {context}")
    
    # 🔥 PASS CONTEXT AND ENABLE DEBUG MODE
    sentiment_analysis = analyze_tweet_sentiment(
        tweet_text, 
        debug=True,  # Enable detailed reasoning output
        context=context  # Pass context for confidence scoring
    )
    
    base["sentiment_analysis"] = sentiment_analysis
    
    if sentiment_analysis.get("status") == "ok":
        sentiment = sentiment_analysis.get("sentiment", {})
        confidence_data = sentiment_analysis.get("confidence", {})
        metadata = sentiment_analysis.get("metadata", {})
        
        classification_result = sentiment.get("classification", "neutral")
        compound = sentiment.get("compound", 0)
        confidence_score = confidence_data.get("score", 0)
        confidence_tier = confidence_data.get("tier", "UNKNOWN")
        
        print(f"Sentiment: {classification_result} (compound: {compound})")
        print(f"Confidence: {confidence_score:.2f} ({confidence_tier})")
        print(f"Include in stats: {metadata.get('include_in_statistics', False)}")
        
        # Enhanced sentiment flag with confidence tier
        if classification_result == "inconclusive":
            base["sentiment_flag"] = "INCONCLUSIVE"
        else:
            base["sentiment_flag"] = f"{classification_result.title()} ({confidence_tier})"
    else:
        print(f"Sentiment analysis error: {sentiment_analysis.get('error', 'unknown error')}")
        base["sentiment_flag"] = "Unavailable"
    
    print("=" * 50)
    return base

# -----------------------------
# CSV path (batch)
# -----------------------------
def _extract_tweets_from_csv(csv_bytes: bytes) -> List[Dict]:
    """
    Reads CSV bytes and returns a list of dicts with:
    {
        "row": int,
        "tweet": str
    }
    """
    csv_data = csv_bytes.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(csv_data))
    tweets: List[Dict] = []

    if reader.fieldnames is None:
        raise ValueError("CSV header row not found.")

    # Try to find a usable column
    lower_headers = [h.lower() for h in reader.fieldnames]
    chosen_col: Optional[str] = None
    for c in POSSIBLE_TEXT_COLUMNS:
        if c in lower_headers:
            # Find original-cased header name to use with row dict
            chosen_col = reader.fieldnames[lower_headers.index(c)]
            break

    if chosen_col is None:
        raise ValueError(
            f"Could not find a text column. Expected one of: {POSSIBLE_TEXT_COLUMNS}. "
            f"Found: {reader.fieldnames}"
        )

    for idx, row in enumerate(reader, 1):
        tweet_text = row.get(chosen_col, "")
        if tweet_text is None:
            tweet_text = ""
        tweets.append({"row": idx, "tweet": tweet_text})

    return tweets

def analyze_csv_tweets_file(file_contents: bytes, filename: str, location: Optional[str] = None) -> Dict:
    """
    Four-step batch analysis:
    1. Classify climate relevance for all tweets
    2. Categorize only the climate-related tweets
    3. If location provided, validate weather consistency for relevant tweets
    4. Analyze sentiment for all tweets using custom VADER
    """
    try:
        tweets = _extract_tweets_from_csv(file_contents)
        raw_texts = [t["tweet"] for t in tweets]

        # Step 1: Batch climate relevance classification
        climate_results = classify_many(raw_texts)

        # Merge climate results back
        enriched: List[Dict] = []
        counts = {
            "total": 0, 
            "climate": 0, 
            "not_climate": 0, 
            "errors": 0,
            "categorized": 0,
            "category_errors": 0,
            "weather_validated": 0,
            "weather_consistent": 0,
            "weather_inconsistent": 0,
            "weather_errors": 0,
            "sentiment_analyzed": 0,
            "sentiment_positive": 0,
            "sentiment_negative": 0,
            "sentiment_neutral": 0,
            "sentiment_errors": 0
        }

        # Collect climate-related tweets for categorization
        climate_related_tweets = []
        climate_related_indices = []

        for i, (t, r) in enumerate(zip(tweets, climate_results)):
            counts["total"] += 1
            record = {
                "row": t["row"],
                "tweet": t["tweet"],
                "location": location,
                "length": len(t["tweet"]),
                "climate_classification": r
            }
            
            if "error" in r:
                counts["errors"] += 1
            else:
                is_climate = r.get("is_climate_related", False)
                if is_climate:
                    counts["climate"] += 1
                    climate_related_tweets.append(t["tweet"])
                    climate_related_indices.append(i)
                else:
                    counts["not_climate"] += 1
                    
            enriched.append(record)

        # Step 2: Categorize climate-related tweets
        if climate_related_tweets:
            print(f"Categorizing {len(climate_related_tweets)} climate-related tweets...")
            category_results = categorize_many_tweets(climate_related_tweets)
            
            # Merge category results back to the corresponding records
            for j, idx in enumerate(climate_related_indices):
                category_result = category_results[j]
                enriched[idx]["category_classification"] = category_result
                
                if "error" in category_result:
                    counts["category_errors"] += 1
                else:
                    counts["categorized"] += 1

        # Step 3: Weather validation for climate-related tweets (if location provided)
        if location and climate_related_tweets:
            print(f"Validating weather consistency for {len(climate_related_tweets)} climate-related tweets at {location}...")
            
            # Process weather validation for climate-related tweets
            for idx in climate_related_indices:
                tweet_text = enriched[idx]["tweet"]
                weather_analysis = weather_validator.analyze_tweet_weather_context(tweet_text, location)
                enriched[idx]["weather_validation"] = weather_analysis
                
                if weather_analysis.get("status") == "success":
                    counts["weather_validated"] += 1
                    validation = weather_analysis.get("validation", {})
                    consistency = validation.get("consistency", "unknown")
                    
                    if consistency == "consistent":
                        enriched[idx]["weather_flag"] = "Consistent"
                        counts["weather_consistent"] += 1
                    elif consistency == "inconsistent":
                        enriched[idx]["weather_flag"] = "Inconsistent"
                        counts["weather_inconsistent"] += 1
                    else:
                        enriched[idx]["weather_flag"] = "Unknown"
                else:
                    enriched[idx]["weather_flag"] = "Unavailable"
                    counts["weather_errors"] += 1

        # Step 4: Sentiment Analysis for climate-related tweets only
        if climate_related_tweets:
            print(f"Analyzing sentiment for {len(climate_related_tweets)} climate-related tweets...")
            
            # Only analyze sentiment for climate-related tweets
            climate_sentiment_results = analyze_batch_sentiment(climate_related_tweets)
            
            # Merge sentiment results back to the corresponding climate-related records
            for j, idx in enumerate(climate_related_indices):
                sentiment_result = climate_sentiment_results[j]
                enriched[idx]["sentiment_analysis"] = sentiment_result
                
                if "error" in sentiment_result:
                    counts["sentiment_errors"] += 1
                    enriched[idx]["sentiment_flag"] = "Unavailable"
                else:
                    counts["sentiment_analyzed"] += 1
                    sentiment_data = sentiment_result.get("sentiment", {})
                    classification = sentiment_data.get("classification", "neutral")
                    
                    enriched[idx]["sentiment_flag"] = classification.title()
                    
                    # Count sentiment classifications
                    if classification == "positive":
                        counts["sentiment_positive"] += 1
                    elif classification == "negative":
                        counts["sentiment_negative"] += 1
                    else:
                        counts["sentiment_neutral"] += 1
            
            # Add "Not Climate Related" flag to non-climate tweets
            for i, record in enumerate(enriched):
                if i not in climate_related_indices:
                    record["sentiment_analysis"] = {
                        "skipped": True,
                        "reason": "Tweet not identified as climate-related"
                    }
                    record["sentiment_flag"] = "Not Climate Related"
        else:
            print("No climate-related tweets found, skipping sentiment analysis.")
            # Mark all tweets as not climate-related for sentiment
            for record in enriched:
                record["sentiment_analysis"] = {
                    "skipped": True,
                    "reason": "Tweet not identified as climate-related"
                }
                record["sentiment_flag"] = "Not Climate Related"

        # Console preview
        print("=" * 50)
        print("RECEIVED CSV FILE:")
        print(f"Filename: {filename}")
        print(f"Location: {location}")
        print(f"Total rows: {counts['total']}")
        print(f"Climate: {counts['climate']} | Not climate: {counts['not_climate']} | Errors: {counts['errors']}")
        print(f"Successfully categorized: {counts['categorized']} | Category errors: {counts['category_errors']}")
        print(f"Sentiment analyzed (climate tweets only): {counts['sentiment_analyzed']} | Sentiment errors: {counts['sentiment_errors']}")
        print(f"Sentiment distribution (climate tweets) - Positive: {counts['sentiment_positive']} | Negative: {counts['sentiment_negative']} | Neutral: {counts['sentiment_neutral']}")
        
        if location:
            print(f"Weather validated: {counts['weather_validated']}")
            print(f"Consistent: {counts['weather_consistent']} | Inconsistent: {counts['weather_inconsistent']} | Weather errors: {counts['weather_errors']}")
        
        print("-" * 30)
        
        for i, row in enumerate(enriched[:5], 1):
            teaser = row["tweet"][:60].replace("\n", " ")
            climate_pred = row["climate_classification"].get("is_climate_related", "err")
            climate_conf = row["climate_classification"].get("confidence", None)
            
            category_info = ""
            if climate_pred and "category_classification" in row:
                cat_result = row["category_classification"]
                if "prediction" in cat_result:
                    category = cat_result["prediction"]
                    cat_conf = cat_result.get("confidence", 0)
                    # Shorten category name for display
                    short_cat = category.split("/")[0].strip() if "/" in category else category
                    category_info = f" -> {short_cat} ({cat_conf:.2f})"
            
            weather_info = ""
            if "weather_flag" in row:
                weather_info = f" [Weather: {row['weather_flag']}]"
            
            sentiment_info = ""
            if "sentiment_flag" in row:
                sentiment_flag = row["sentiment_flag"]
                if "sentiment_analysis" in row and "sentiment" in row["sentiment_analysis"]:
                    compound = row["sentiment_analysis"]["sentiment"].get("compound", 0)
                    sentiment_info = f" [Sentiment: {sentiment_flag} ({compound:+.2f})]"
                else:
                    sentiment_info = f" [Sentiment: {sentiment_flag}]"
            
            conf_str = f" ({climate_conf:.3f})" if isinstance(climate_conf, float) else ""
            print(f"{i}. [{climate_pred}]{conf_str} {teaser}{category_info}{weather_info}{sentiment_info}")
            
        if len(enriched) > 5:
            print(f"... and {len(enriched) - 5} more rows")

        # Generate category insights for climate-related tweets
        category_insights = None
        if climate_related_tweets:
            valid_category_results = []
            for idx in climate_related_indices:
                if "category_classification" in enriched[idx] and "error" not in enriched[idx]["category_classification"]:
                    valid_category_results.append(enriched[idx]["category_classification"])
            
            if valid_category_results:
                category_insights = get_category_insights(valid_category_results)

        # Generate sentiment insights (only for climate-related tweets)
        sentiment_insights = None
        climate_sentiment_results = [r for r in enriched if r.get("sentiment_flag", "") not in ["Not Climate Related", "Unavailable"]]
        if climate_sentiment_results:
            sentiment_insights = get_sentiment_insights(climate_sentiment_results)

        print("=" * 50)

        return {
            "status": "ok",
            "message": f"CSV analyzed. Found {counts['climate']} climate-related tweets out of {counts['total']} total. Sentiment analyzed for climate tweets only.",
            "filename": filename,
            "location": location,
            "summary": counts,
            "category_insights": category_insights,
            "sentiment_insights": sentiment_insights,
            "available_categories": CLIMATE_CATEGORIES,
            "sample": enriched[:3],          # small preview for frontend
            "results": enriched              # full results
        }

    except Exception as e:
        err = f"Error processing CSV: {str(e)}"
        print(f"[ERROR] {err}")
        return {"error": err}

# -----------------------------
# Utility functions for specific analysis
# -----------------------------
def analyze_category_distribution(results: List[Dict]) -> Dict:
    """
    Analyze the distribution of climate categories in results
    """
    category_counts = {}
    confidence_by_category = {}
    
    for result in results:
        if "category_classification" in result and "prediction" in result["category_classification"]:
            category = result["category_classification"]["prediction"]
            confidence = result["category_classification"].get("confidence", 0)
            
            if category not in category_counts:
                category_counts[category] = 0
                confidence_by_category[category] = []
            
            category_counts[category] += 1
            confidence_by_category[category].append(confidence)
    
    # Calculate average confidence per category
    avg_confidence_by_category = {}
    for category, confidences in confidence_by_category.items():
        if confidences:
            avg_confidence_by_category[category] = sum(confidences) / len(confidences)
    
    return {
        "category_counts": category_counts,
        "average_confidence_by_category": avg_confidence_by_category,
        "total_categorized": sum(category_counts.values()),
        "unique_categories_found": len(category_counts)
    }

def get_climate_tweets_only(results: List[Dict]) -> List[Dict]:
    """
    Filter results to return only climate-related tweets with their categories
    """
    climate_tweets = []
    
    for result in results:
        climate_class = result.get("climate_classification", {})
        if climate_class.get("is_climate_related", False):
            climate_tweets.append(result)
    
    return climate_tweets

def analyze_weather_consistency_distribution(results: List[Dict]) -> Dict:
    """
    Analyze the distribution of weather consistency flags
    """
    weather_stats = {
        "consistent": 0,
        "inconsistent": 0,
        "unknown": 0,
        "unavailable": 0,
        "no_location": 0
    }
    
    consistency_by_category = {}
    
    for result in results:
        weather_flag = result.get("weather_flag", "no_location").lower()
        
        if weather_flag in weather_stats:
            weather_stats[weather_flag] += 1
        
        # Cross-reference with climate categories
        if "category_classification" in result and "prediction" in result["category_classification"]:
            category = result["category_classification"]["prediction"]
            
            if category not in consistency_by_category:
                consistency_by_category[category] = {
                    "consistent": 0,
                    "inconsistent": 0,
                    "other": 0
                }
            
            if weather_flag == "consistent":
                consistency_by_category[category]["consistent"] += 1
            elif weather_flag == "inconsistent":
                consistency_by_category[category]["inconsistent"] += 1
            else:
                consistency_by_category[category]["other"] += 1
    
    return {
        "weather_consistency_stats": weather_stats,
        "consistency_by_climate_category": consistency_by_category,
        "total_with_weather_data": weather_stats["consistent"] + weather_stats["inconsistent"] + weather_stats["unknown"]
    }

def analyze_complete_distribution(results: List[Dict]) -> Dict:
    """
    Analyze complete distribution including climate, category, weather, and sentiment
    """
    category_dist = analyze_category_distribution(results)
    weather_dist = analyze_weather_consistency_distribution(results)
    
    # Get climate tweets for sentiment analysis
    climate_tweets = get_climate_tweets_only(results)
    sentiment_dist = analyze_sentiment_distribution(climate_tweets)
    
    # Cross-analysis: sentiment vs category
    sentiment_by_category = {}
    category_by_sentiment = {"positive": {}, "negative": {}, "neutral": {}}
    
    for result in climate_tweets:
        if ("category_classification" in result and 
            "sentiment_analysis" in result and 
            "prediction" in result["category_classification"] and
            "sentiment" in result["sentiment_analysis"]):
            
            category = result["category_classification"]["prediction"]
            sentiment_class = result["sentiment_analysis"]["sentiment"]["classification"]
            
            # Track sentiment distribution by category
            if category not in sentiment_by_category:
                sentiment_by_category[category] = {"positive": 0, "negative": 0, "neutral": 0}
            sentiment_by_category[category][sentiment_class] += 1
            
            # Track category distribution by sentiment
            if category not in category_by_sentiment[sentiment_class]:
                category_by_sentiment[sentiment_class][category] = 0
            category_by_sentiment[sentiment_class][category] += 1
    
    return {
        "category_distribution": category_dist,
        "weather_distribution": weather_dist,
        "sentiment_distribution": sentiment_dist,
        "cross_analysis": {
            "sentiment_by_category": sentiment_by_category,
            "category_by_sentiment": category_by_sentiment
        },
        "summary": {
            "total_tweets": len(results),
            "climate_tweets": len(climate_tweets),
            "categories_found": category_dist.get("unique_categories_found", 0),
            "weather_analyzed": weather_dist.get("total_with_weather_data", 0),
            "sentiment_analyzed": sentiment_dist.get("successful_analysis", 0)
        }
    }