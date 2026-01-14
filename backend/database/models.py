from sqlalchemy import create_engine, Column, Integer, String, Boolean, Text, DateTime, Float, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func
from typing import Generator
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Tweet(Base):
    __tablename__ = "tweets"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Basic tweet info
    tweet_text = Column(Text, nullable=False)
    location = Column(String(255), nullable=True)
    length = Column(Integer, nullable=True)
    
    # Climate classification results
    is_climate_related = Column(Boolean, nullable=False)
    climate_confidence = Column(Float, nullable=True)
    climate_prediction = Column(Integer, nullable=True)
    
    # Climate category classification (if climate-related)
    category = Column(String(100), nullable=True)
    category_confidence = Column(Float, nullable=True)
    category_probabilities = Column(JSON, nullable=True)
    
    # Weather validation results
    weather_flag = Column(String(50), nullable=True)
    weather_consistency_score = Column(Float, nullable=True)
    weather_data = Column(JSON, nullable=True)
    
    # Sentiment analysis results
    sentiment_flag = Column(String(20), nullable=True)
    sentiment_compound = Column(Float, nullable=True)
    sentiment_positive = Column(Float, nullable=True)
    sentiment_negative = Column(Float, nullable=True)
    sentiment_neutral = Column(Float, nullable=True)
    
    # Full pipeline response storage
    full_response = Column(JSON, nullable=True)
    
    # Metadata
    analyzed_at = Column(DateTime(timezone=True), server_default=func.now())
    batch_id = Column(Integer, ForeignKey("batch_uploads.id"), nullable=True)
    
    # Relationship
    batch = relationship("BatchUpload", back_populates="tweets")

class BatchUpload(Base):
    __tablename__ = "batch_uploads"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    location = Column(String(255), nullable=True)
    total_tweets = Column(Integer, nullable=False)
    climate_tweets = Column(Integer, nullable=False, default=0)
    processed_tweets = Column(Integer, nullable=False, default=0)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    processing_time_seconds = Column(Float, nullable=True)
    
    # Relationship
    tweets = relationship("Tweet", back_populates="batch")

class DailyStats(Base):
    __tablename__ = "daily_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime(timezone=True), nullable=False, unique=True)
    
    # Tweet counts
    total_tweets = Column(Integer, nullable=False, default=0)
    climate_tweets = Column(Integer, nullable=False, default=0)
    non_climate_tweets = Column(Integer, nullable=False, default=0)
    
    # Category distribution
    category_distribution = Column(JSON, nullable=True)
    
    # Weather consistency stats
    weather_consistent = Column(Integer, nullable=False, default=0)
    weather_inconsistent = Column(Integer, nullable=False, default=0)
    weather_unknown = Column(Integer, nullable=False, default=0)
    
    # Sentiment stats
    sentiment_positive = Column(Integer, nullable=False, default=0)
    sentiment_negative = Column(Integer, nullable=False, default=0)
    sentiment_neutral = Column(Integer, nullable=False, default=0)
    avg_compound_score = Column(Float, nullable=True)
    
    # Location distribution
    location_distribution = Column(JSON, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)