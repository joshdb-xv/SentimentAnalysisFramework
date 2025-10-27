from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database.models import get_db
from services.main_service import get_model_status
from services.climate_category import category_model_status, CLIMATE_CATEGORIES
from services.weather_forecast_service import weather_forecast_service
from services.sentiment_analysis import sentiment_model_status

router = APIRouter()

@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint with all model status and database connectivity"""
    model_status = get_model_status()
    
    # Check database connectivity
    try:
        from database.models import Tweet
        tweet_count = db.query(Tweet).count()
        db_status = "connected"
    except Exception as e:
        tweet_count = 0
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "models": model_status,
        "available_categories": CLIMATE_CATEGORIES,
        "weather_api_status": "configured" if weather_forecast_service.api_key else "not_configured",
        "database": {
            "status": db_status,
            "total_tweets": tweet_count
        }
    }

@router.get("/categories")
async def get_climate_categories():
    """Get all available climate categories"""
    return {
        "categories": CLIMATE_CATEGORIES,
        "total": len(CLIMATE_CATEGORIES),
        "model_status": category_model_status()
    }

@router.get("/model-status")
async def model_status_endpoint():
    """Get detailed status of all models"""
    return get_model_status()

@router.get("/model-info")
async def model_info_endpoint():
    """Get detailed information about all models and their capabilities"""
    status = get_model_status()
    
    return {
        "climate_relevance_model": {
            "purpose": "Determines if a tweet is climate-related or not",
            "output": "Binary classification (climate-related: true/false)",
            "status": status.get("climate_relevance_model", {})
        },
        "climate_category_model": {
            "purpose": "Categorizes climate-related tweets into specific domains",
            "categories": CLIMATE_CATEGORIES,
            "total_categories": len(CLIMATE_CATEGORIES),
            "output": "Multi-class classification with confidence scores",
            "status": status.get("climate_category_model", {})
        },
        "weather_validation_service": {
            "purpose": "Validates tweet weather sentiment against actual weather data",
            "supported_locations": list(weather_forecast_service.location_mappings.keys()),
            "output": "Consistency flag (Consistent/Inconsistent/Unknown)",
            "api_configured": bool(weather_forecast_service.api_key)
        },
        "sentiment_analysis_model": {
            "purpose": "Analyzes tweet sentiment using custom VADER with annotated lexicon",
            "output": "Sentiment scores (positive/negative/neutral/compound) and classification",
            "lexicon_source": "Custom annotated lexical dictionary",
            "status": status.get("sentiment_analysis_model", {})
        },
        "workflow": {
            "step_1": "Tweet → Climate Relevance Check",
            "step_2": "If climate-related → Category Classification", 
            "step_3": "If location provided → Weather Consistency Validation",
            "step_4": "All tweets → Sentiment Analysis (Custom VADER)",
            "step_5": "Store results in database for analytics",
            "batch_processing": "Available for CSV files with location and sentiment support"
        }
    }