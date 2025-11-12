from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database.models import init_db
from services.main_service import get_model_status
from services.fasttext_service import get_fasttext_manager
from services.lexical_dictionary_manager import get_dictionary_manager

# Import all the routers
from routers import (
    health_router,
    weather_router,
    sentiment_router,
    analysis_router,
    database_router,
    benchmarks_router,
    lexical_router,
    twitter_router,
    classifier_router  # NEW: Climate classifier router
)

# -----------------------------
# Lifespan Context Manager
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("\n🚀 Starting Climate Tweet Analysis API...")
    
    # Initialize database
    init_db()
    print("✅ Database initialized successfully")
    
    # Load FastText models at startup
    try:
        fasttext_manager = get_fasttext_manager()
        fasttext_manager.load_models(limit=None, use_cache=True)
    except Exception as e:
        print(f"⚠️ Warning: Could not load FastText models: {str(e)}")
        print("   Lexical dictionary processing will not be available.")
    
    # Load cached lexical dictionary if exists
    try:
        dict_manager = get_dictionary_manager()
        if dict_manager.exists():
            print("\n📖 Loading cached lexical dictionary...")
            success = dict_manager.load()
            if success:
                total_words = len(dict_manager.lexicon_df)
                created_at = dict_manager.metadata.get('created_at', 'unknown')
                print(f"✅ Loaded {total_words} words from cache")
                print(f"   Created: {created_at}")
                
                # Check for manual updates
                manual_updates = dict_manager.metadata.get('manual_updates', [])
                if manual_updates:
                    print(f"   📝 {len(manual_updates)} manual word updates recorded")
            else:
                print("⚠️ Failed to load cached dictionary")
        else:
            print("\n📖 No cached lexical dictionary found")
            print("   Process a dictionary and save it to create the cache")
    except Exception as e:
        print(f"⚠️ Warning: Could not load lexical dictionary cache: {str(e)}")
    
    # Initialize Climate Classifier Service
    try:
        from services.classifier_service import get_classifier_service
        classifier_service = get_classifier_service()
        print("✅ Climate Classifier Service initialized")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize classifier service: {str(e)}")
    
    print(f"✅ API Version: {app.version}")
    print(f"✅ Models loaded: {get_model_status()}")
    print("🎉 Server ready!\n")
    
    yield
    
    # Shutdown
    print("\n👋 Shutting down Climate Tweet Analysis API...")

# -----------------------------
# FastAPI App Configuration
# -----------------------------
app = FastAPI(
    title="Climate Tweet Analysis API", 
    version="3.2.0",  # Updated version
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
async def root():
    return {
        "message": "Climate Tweet Analysis API is running!",
        "version": "3.2.0",
        "features": [
            "Climate relevance detection",
            "Climate category classification", 
            "Weather consistency validation",
            "Custom VADER sentiment analysis",
            "Batch CSV processing",
            "Complete distribution analysis",
            "Database storage and analytics",
            "Lexical dictionary generation with FastText",
            "Cached lexical dictionary with on-the-fly updates",
            "Twitter Scraper using Tweepy and Twitter API",
            "Climate Classifier Training & Pseudo-Labeling"  # NEW
        ]
    }

# -----------------------------
# Include Routers
# -----------------------------
app.include_router(health_router.router, tags=["Health & Status"])
app.include_router(weather_router.router, prefix="/weather", tags=["Weather"])
app.include_router(sentiment_router.router, tags=["Sentiment Analysis"])
app.include_router(analysis_router.router, tags=["Tweet Analysis"])
app.include_router(database_router.router, prefix="/database", tags=["Database"])
app.include_router(benchmarks_router.router, tags=["Benchmarks"])
app.include_router(lexical_router.router, tags=["Lexical Dictionary"])
app.include_router(twitter_router.router, tags=["Twitter Scraper"])
app.include_router(classifier_router.router, tags=["Climate Classifier"])  # NEW

# -----------------------------
# Main Entry Point
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)