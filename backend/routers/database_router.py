from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from database.models import get_db

router = APIRouter()

@router.get("/stats")
async def get_database_stats(db: Session = Depends(get_db)):
    """Get overall database statistics"""
    from database.models import Tweet, BatchUpload, DailyStats
    
    total_tweets = db.query(Tweet).count()
    climate_tweets = db.query(Tweet).filter(Tweet.is_climate_related == True).count()
    total_batches = db.query(BatchUpload).count()
    unique_locations = db.query(Tweet.location).distinct().count()
    
    # Get date range
    first_tweet = db.query(Tweet).order_by(Tweet.analyzed_at).first()
    last_tweet = db.query(Tweet).order_by(Tweet.analyzed_at.desc()).first()
    
    return {
        "status": "ok",
        "stats": {
            "total_tweets": total_tweets,
            "climate_tweets": climate_tweets,
            "non_climate_tweets": total_tweets - climate_tweets,
            "climate_percentage": round((climate_tweets / total_tweets * 100), 2) if total_tweets > 0 else 0,
            "total_batch_uploads": total_batches,
            "unique_locations": unique_locations,
            "date_range": {
                "first": first_tweet.analyzed_at.isoformat() if first_tweet else None,
                "last": last_tweet.analyzed_at.isoformat() if last_tweet else None
            }
        }
    }

@router.delete("/clear")
async def clear_database(
    db: Session = Depends(get_db),
    confirm: bool = Query(False)
):
    """
    Clear all data from database (use with caution!)
    Requires confirm=true parameter
    """
    if not confirm:
        return {
            "status": "error",
            "message": "Please confirm database clearing by setting confirm=true"
        }
    
    from database.models import Tweet, BatchUpload, DailyStats
    
    try:
        # Delete all records
        deleted_tweets = db.query(Tweet).count()
        deleted_batches = db.query(BatchUpload).count()
        deleted_stats = db.query(DailyStats).count()
        
        db.query(Tweet).delete()
        db.query(BatchUpload).delete()
        db.query(DailyStats).delete()
        db.commit()
        
        return {
            "status": "ok",
            "message": "Database cleared successfully",
            "deleted": {
                "tweets": deleted_tweets,
                "batches": deleted_batches,
                "daily_stats": deleted_stats
            }
        }
    except Exception as e:
        db.rollback()
        return {
            "status": "error",
            "message": f"Error clearing database: {str(e)}"
        }