from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database.models import init_db
from services.main_service import get_model_status
from services.fasttext_service import get_fasttext_manager

# Import all the routers
from routers import (
    health_router,
    weather_router,
    sentiment_router,
    analysis_router,
    database_router,
    benchmarks_router,
    lexical_router
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
    version="3.0.0",
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
        "version": "3.0.0",
        "features": [
            "Climate relevance detection",
            "Climate category classification", 
            "Weather consistency validation",
            "Custom VADER sentiment analysis",
            "Batch CSV processing",
            "Complete distribution analysis",
            "Database storage and analytics",
            "Lexical dictionary generation with FastText"
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

# -----------------------------
# Main Entry Point
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)