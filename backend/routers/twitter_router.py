from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
from services.twitter_service import (
    scrape_tweets,
    get_scraping_status,
    cancel_scraping,
    get_scraping_history
)

router = APIRouter(prefix="/twitter", tags=["Twitter Scraper"])

class ScrapeRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search keyword or phrase")
    limit: int = Field(default=10, ge=1, le=1000, description="Number of tweets to scrape")
    similarity_threshold: float = Field(default=0.85, ge=0.8, le=0.95, description="Duplicate detection threshold")
    use_expansion: bool = Field(default=True, description="Enable word variation expansion")

class ScrapeResponse(BaseModel):
    task_id: str
    message: str
    query: str
    limit: int
    status: str

@router.post("/scrape", response_model=ScrapeResponse)
async def scrape_tweets_endpoint(request: ScrapeRequest, background_tasks: BackgroundTasks):
    """
    Start a Twitter scraping task in the background
    """
    try:
        task_id = await scrape_tweets(
            query=request.query,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold,
            use_expansion=request.use_expansion,
            background_tasks=background_tasks
        )
        
        return ScrapeResponse(
            task_id=task_id,
            message="Scraping task started successfully",
            query=request.query,
            limit=request.limit,
            status="in_progress"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start scraping: {str(e)}")

@router.get("/status/{task_id}")
async def get_status(task_id: str):
    """
    Get the status of a scraping task
    """
    try:
        status = get_scraping_status(task_id)
        if not status:
            raise HTTPException(status_code=404, detail="Task not found")
        return status
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.delete("/cancel/{task_id}")
async def cancel_task(task_id: str):
    """
    Cancel a running scraping task
    """
    try:
        success = cancel_scraping(task_id)
        if not success:
            raise HTTPException(status_code=404, detail="Task not found or already completed")
        return {"message": "Task cancelled successfully", "task_id": task_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")

@router.get("/history")
async def get_history(limit: int = 10):
    """
    Get recent scraping history
    """
    try:
        history = get_scraping_history(limit)
        return {"history": history, "total": len(history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")