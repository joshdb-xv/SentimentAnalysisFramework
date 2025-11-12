from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd
import json
import os
from typing import Optional
from datetime import datetime
import shutil
import traceback
import numpy as np

# Import the new dictionary manager
from services.lexical_dictionary_manager import get_dictionary_manager

router = APIRouter(prefix="/lexical", tags=["Lexical Dictionary"])


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

    
# Directory setup
UPLOAD_DIR = "uploads/lexical"
OUTPUT_DIR = "outputs/lexical"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Store processing status
processing_status = {
    "status": "idle",
    "progress": 0,
    "message": "",
    "result_file": None,
    "stats": {}
}

# Get global dictionary manager
dict_manager = get_dictionary_manager()


# ============================================
# NEW: Cache/Dictionary Management Endpoints
# ============================================

@router.get("/dictionary/status")
async def get_dictionary_status():
    """Check if cached dictionary exists and its status"""
    return dict_manager.get_status()


@router.post("/dictionary/load")
async def load_cached_dictionary():
    """Load cached dictionary into memory"""
    try:
        success = dict_manager.load()
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="No cached dictionary found. Please process a dictionary first."
            )
        
        return {
            "message": "Dictionary loaded successfully",
            "metadata": dict_manager.metadata,
            "total_words": len(dict_manager.lexicon_df)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error loading dictionary: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error loading: {str(e)}")


@router.post("/dictionary/save")
async def save_as_main_dictionary():
    """Save the current processed dictionary as the main cached version"""
    try:
        if processing_status["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail="No completed processing to save. Please process a dictionary first."
            )
        
        if not processing_status["result_file"] or not os.path.exists(processing_status["result_file"]):
            raise HTTPException(status_code=404, detail="Result file not found")
        
        # Load the processed data
        lexicon_df = pd.read_csv(processing_status["result_file"])
        
        # We need to reconstruct word_data from the original processing
        # This requires access to the processor's dataframe
        from services.lexical_processor import LexicalProcessor
        
        # Load original data to get full word info
        if not hasattr(dict_manager, '_temp_processor'):
            raise HTTPException(
                status_code=400,
                detail="Processor not available. Please reprocess the dictionary."
            )
        
        processor = dict_manager._temp_processor
        
        # Build word_data dictionary
        word_data = {}
        for _, row in processor.df.iterrows():
            word = row['word_clean']
            word_data[word] = {
                'sentiment': row['sentiment'],
                'is_climate': row['is_climate'],
                'dialect': row.get('dialect', 'filipino'),
                'definition': row.get('definition', '')
            }
        
        # Save to cache
        success = dict_manager.save(
            lexicon_df=lexicon_df,
            word_data=word_data,
            processor=processor,
            stats=processing_status["stats"]
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save dictionary")
        
        return {
            "message": "Dictionary saved as main cached version",
            "path": str(dict_manager.joblib_path),
            "metadata": dict_manager.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error saving dictionary: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error saving: {str(e)}")


@router.delete("/dictionary/reset")
async def reset_cached_dictionary():
    """Delete cached dictionary (forces reprocessing)"""
    try:
        success = dict_manager.reset()
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to reset cache")
        
        return {
            "message": "Cached dictionary deleted successfully",
            "note": "Next search or load will require reprocessing"
        }
        
    except Exception as e:
        print(f"Error resetting cache: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error resetting: {str(e)}")


class UpdateWordLabelRequest(BaseModel):
    word: str
    new_label: str  # positive/negative/neutral


@router.put("/dictionary/update-word")
async def update_word_label(request: UpdateWordLabelRequest):
    """
    Update a word's sentiment label and recalculate its score
    The 3-stage pipeline will rerun for this word
    """
    try:
        # Ensure dictionary is loaded
        if dict_manager.lexicon_df is None:
            # Try to load from cache
            success = dict_manager.load()
            if not success:
                raise HTTPException(
                    status_code=404,
                    detail="No dictionary loaded. Please load or process a dictionary first."
                )
        
        # Update the word
        result = dict_manager.update_word_label(
            word=request.word,
            new_label=request.new_label,
            processor=dict_manager.processor
        )
        
        # Convert numpy types to native Python types for JSON serialization
        result = convert_numpy_types(result)
        
        # Auto-save changes
        dict_manager.save_changes()
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error updating word: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error updating: {str(e)}")


@router.get("/dictionary/export")
async def export_dictionary_csv():
    """Export current dictionary as CSV"""
    try:
        if dict_manager.lexicon_df is None:
            raise HTTPException(
                status_code=404,
                detail="No dictionary loaded. Please load or process a dictionary first."
            )
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(OUTPUT_DIR, f"lexical_export_{timestamp}.csv")
        
        dict_manager.export_csv(output_path)
        
        return FileResponse(
            output_path,
            media_type="text/csv",
            filename=f"lexical_dictionary_{timestamp}.csv"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error exporting: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error exporting: {str(e)}")


# ============================================
# Original Endpoints (Updated to use cache)
# ============================================

@router.post("/upload-dictionary")
async def upload_dictionary(file: UploadFile = File(...)):
    """Upload the main dictionary Excel file"""
    try:
        print(f"Received file: {file.filename}")
        
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are allowed")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(UPLOAD_DIR, f"dictionary_{timestamp}.xlsx")
        
        print(f"Saving to: {file_path}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"File saved successfully. Size: {os.path.getsize(file_path)} bytes")
        
        df = pd.read_excel(file_path)
        row_count = len(df)
        columns = df.columns.tolist()
        
        print(f"Excel loaded: {row_count} rows, columns: {columns}")
        
        expected_cols = ['letter', 'word', 'definition', 'dialect', 'sentiment']
        missing_cols = [col for col in expected_cols if col not in columns]
        
        if missing_cols:
            print(f"Warning: Missing expected columns: {missing_cols}")
            return {
                "message": "Dictionary uploaded with warnings",
                "file_path": file_path,
                "rows": row_count,
                "columns": columns,
                "warning": f"Missing expected columns: {missing_cols}",
                "preview": df.head(5).to_dict(orient="records")
            }
        
        return {
            "message": "Dictionary uploaded successfully",
            "file_path": file_path,
            "rows": row_count,
            "columns": columns,
            "preview": df.head(5).to_dict(orient="records")
        }
        
    except Exception as e:
        print(f"Error in upload_dictionary: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.get("/load-climate-keywords")
async def load_climate_keywords():
    """Load climate keywords from a fixed location"""
    keywords_path = "data/weatherkeywords.csv"
    
    try:
        print(f"Loading keywords from: {keywords_path}")
        
        if not os.path.exists(keywords_path):
            raise HTTPException(
                status_code=404, 
                detail=f"Climate keywords file not found at {keywords_path}. Please ensure the file exists."
            )
        
        df = pd.read_csv(keywords_path)
        keyword_count = len(df)
        
        print(f"Keywords loaded: {keyword_count} keywords")
        
        return {
            "message": "Climate keywords loaded successfully",
            "file_path": keywords_path,
            "keyword_count": keyword_count,
            "preview": df.head(10).to_dict(orient="records")
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error loading keywords: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error loading file: {str(e)}")


@router.post("/upload-climate-keywords")
async def upload_climate_keywords(file: UploadFile = File(...)):
    """Upload the climate keywords CSV file"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(UPLOAD_DIR, f"climate_keywords_{timestamp}.csv")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        df = pd.read_csv(file_path)
        keyword_count = len(df)
        
        return {
            "message": "Climate keywords uploaded successfully",
            "file_path": file_path,
            "keyword_count": keyword_count,
            "preview": df.head(10).to_dict(orient="records")
        }
    except Exception as e:
        print(f"Error uploading keywords: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


class ProcessRequest(BaseModel):
    dictionary_path: str
    climate_keywords_path: Optional[str] = None


@router.post("/process")
async def process_lexical_dictionary(
    background_tasks: BackgroundTasks,
    request: ProcessRequest
):
    """Start processing the lexical dictionary"""
    global processing_status
    
    dictionary_path = request.dictionary_path
    climate_keywords_path = request.climate_keywords_path
    
    try:
        if processing_status["status"] == "processing":
            raise HTTPException(status_code=400, detail="Processing already in progress")
        
        if not os.path.exists(dictionary_path):
            raise HTTPException(status_code=404, detail=f"Dictionary file not found: {dictionary_path}")
        
        if not climate_keywords_path:
            climate_keywords_path = "data/weatherkeywords.csv"
            if not os.path.exists(climate_keywords_path):
                raise HTTPException(
                    status_code=404,
                    detail="Default climate keywords file not found. Please ensure data/weatherkeywords.csv exists."
                )
        
        if not os.path.exists(climate_keywords_path):
            raise HTTPException(status_code=404, detail=f"Keywords file not found: {climate_keywords_path}")
        
        processing_status = {
            "status": "processing",
            "progress": 0,
            "message": "Starting processing...",
            "result_file": None,
            "stats": {}
        }
        
        background_tasks.add_task(
            process_dictionary_task,
            dictionary_path,
            climate_keywords_path
        )
        
        return {"message": "Processing started", "status": processing_status}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error starting process: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error starting process: {str(e)}")


def process_dictionary_task(dictionary_path: str, climate_keywords_path: str):
    """Background task for processing the dictionary"""
    global processing_status
    
    try:
        print(f"\n{'='*60}")
        print(f"Starting lexical processing...")
        print(f"Dictionary: {dictionary_path}")
        print(f"Keywords: {climate_keywords_path}")
        print(f"{'='*60}\n")
        
        from services.lexical_processor import LexicalProcessor
        
        processor = LexicalProcessor(
            dictionary_path=dictionary_path,
            climate_keywords_path=climate_keywords_path
        )
        
        def update_progress(progress, message):
            processing_status["progress"] = progress
            processing_status["message"] = message
            print(f"Progress: {progress}% - {message}")
        
        processor.set_progress_callback(update_progress)
        
        print("Starting processing...")
        result = processor.process()
        
        print(f"\nProcessing completed!")
        print(f"Result keys: {result.keys()}")
        print(f"Stats: {result.get('stats', {})}")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(OUTPUT_DIR, f"lexical_dictionary_{timestamp}.csv")
        result["dataframe"].to_csv(output_file, index=False)
        
        print(f"Saved to: {output_file}")
        
        # Store processor temporarily for saving
        dict_manager._temp_processor = result.get("processor")
        
        processing_status["status"] = "completed"
        processing_status["progress"] = 100
        processing_status["message"] = "Processing completed successfully"
        processing_status["result_file"] = output_file
        processing_status["stats"] = result["stats"]
        
        print(f"Final status: {processing_status}")
        print("\n💡 TIP: Call POST /lexical/dictionary/save to save this as your main cached dictionary")
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"\n{'='*60}")
        print(f"ERROR during processing:")
        print(error_msg)
        traceback.print_exc()
        print(f"{'='*60}\n")
        
        processing_status["status"] = "error"
        processing_status["message"] = error_msg


@router.get("/status")
async def get_processing_status():
    """Get current processing status"""
    return processing_status


@router.get("/results")
async def get_results():
    """Get processing results as JSON"""
    try:
        if processing_status["status"] != "completed":
            raise HTTPException(status_code=400, detail="Processing not completed yet")
        
        if not processing_status["result_file"] or not os.path.exists(processing_status["result_file"]):
            raise HTTPException(status_code=404, detail="Result file not found")
        
        df = pd.read_csv(processing_status["result_file"])
        
        return {
            "stats": processing_status["stats"],
            "preview": df.head(50).to_dict(orient="records"),
            "total_rows": len(df),
            "download_url": f"/lexical/download"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting results: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error reading results: {str(e)}")


@router.get("/download")
async def download_result():
    """Download the processed lexical dictionary CSV"""
    try:
        if processing_status["status"] != "completed":
            raise HTTPException(status_code=400, detail="Processing not completed yet")
        
        if not processing_status["result_file"] or not os.path.exists(processing_status["result_file"]):
            raise HTTPException(status_code=404, detail="Result file not found")
        
        return FileResponse(
            processing_status["result_file"],
            media_type="text/csv",
            filename="lexical_dictionary_vader.csv"
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error downloading: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")


class SearchRequest(BaseModel):
    word: str


@router.post("/search")
async def search_word(request: SearchRequest):
    """
    Search for a word in the lexical dictionary
    Automatically loads from cache if not in memory
    Returns the word's sentiment score and detailed breakdown
    """
    try:
        word = request.word.strip().lower()
        
        if not word:
            raise HTTPException(status_code=400, detail="Word cannot be empty")
        
        # Try to ensure dictionary is loaded
        if dict_manager.lexicon_df is None:
            # Try loading from cache first
            success = dict_manager.load()
            
            if not success:
                # No cache, check if there's a completed processing
                if processing_status["status"] == "completed" and processing_status["result_file"]:
                    if os.path.exists(processing_status["result_file"]):
                        # Load from the processed file temporarily
                        temp_df = pd.read_csv(processing_status["result_file"])
                        result = temp_df[temp_df['word'] == word]
                        
                        if result.empty:
                            return {
                                "found": False,
                                "word": word,
                                "message": f"Word '{word}' not found in processed dictionary",
                                "note": "Dictionary not saved to cache yet. Use POST /lexical/dictionary/save"
                            }
                        
                        score = float(result.iloc[0]['sentiment_score'])
                        
                        # Determine polarity
                        if score > 0:
                            polarity = "positive"
                            intensity_label = get_intensity_label(score, positive=True)
                        elif score < 0:
                            polarity = "negative"
                            intensity_label = get_intensity_label(score, positive=False)
                        else:
                            polarity = "neutral"
                            intensity_label = "neutral"
                        
                        return {
                            "found": True,
                            "word": word,
                            "sentiment_score": round(score, 3),
                            "polarity": polarity,
                            "intensity": intensity_label,
                            "breakdown": {
                                "score": round(score, 3),
                                "polarity": polarity,
                                "strength": abs(score)
                            },
                            "interpretation": get_score_interpretation(score),
                            "note": "Result from temporary processing. Save to cache for persistence."
                        }
                    else:
                        raise HTTPException(
                            status_code=404,
                            detail="No dictionary available. Please load cache or process a dictionary."
                        )
                else:
                    raise HTTPException(
                        status_code=404,
                        detail="No dictionary available. Please load cache or process a dictionary."
                    )
        
        # Dictionary is loaded - search in it
        word_info = dict_manager.get_word_info(word)
        
        if word_info is None:
            return {
                "found": False,
                "word": word,
                "message": f"Word '{word}' not found in lexical dictionary",
                "suggestions": get_similar_words(word, dict_manager.lexicon_df)
            }
        
        score = word_info['sentiment_score']
        
        # Determine polarity and intensity
        if score > 0:
            polarity = "positive"
            intensity_label = get_intensity_label(score, positive=True)
        elif score < 0:
            polarity = "negative"
            intensity_label = get_intensity_label(score, positive=False)
        else:
            polarity = "neutral"
            intensity_label = "neutral"
        
        # Get detailed breakdown from processor if available
        breakdown = None
        if dict_manager.processor:
            breakdown = dict_manager.processor.get_word_breakdown(word)
            if breakdown:
                breakdown = convert_numpy_types(breakdown)
        
        response = {
            "found": True,
            "word": word,
            "sentiment_score": round(score, 3),
            "sentiment_label": word_info['sentiment_label'],
            "polarity": polarity,
            "intensity": intensity_label,
            "is_climate": word_info['is_climate'],
            "dialect": word_info['dialect'],
            "definition": word_info['definition'],
            "breakdown": {
                "score": round(score, 3),
                "polarity": polarity,
                "strength": abs(score)
            },
            "interpretation": get_score_interpretation(score),
            "editable": True  # Flag to indicate frontend can edit this
        }
        
        # Add detailed calculation breakdown if available
        if breakdown:
            response["detailed_breakdown"] = breakdown
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error searching word: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")


def get_intensity_label(score: float, positive: bool) -> str:
    """Get intensity label based on score"""
    abs_score = abs(score)
    
    if abs_score >= 3.5:
        return "very strong"
    elif abs_score >= 2.5:
        return "strong"
    elif abs_score >= 1.5:
        return "moderate"
    elif abs_score >= 0.5:
        return "weak"
    else:
        return "very weak"


def get_score_interpretation(score: float) -> str:
    """Get human-readable interpretation of the score"""
    abs_score = abs(score)
    
    if score > 0:
        if abs_score >= 3.5:
            return "This word has a very strong positive sentiment, likely climate-related"
        elif abs_score >= 2.5:
            return "This word has a strong positive sentiment"
        elif abs_score >= 1.5:
            return "This word has a moderate positive sentiment"
        elif abs_score >= 0.5:
            return "This word has a weak positive sentiment"
        else:
            return "This word has a very weak positive sentiment"
    elif score < 0:
        if abs_score >= 3.5:
            return "This word has a very strong negative sentiment, likely climate-related"
        elif abs_score >= 2.5:
            return "This word has a strong negative sentiment"
        elif abs_score >= 1.5:
            return "This word has a moderate negative sentiment"
        elif abs_score >= 0.5:
            return "This word has a weak negative sentiment"
        else:
            return "This word has a very weak negative sentiment"
    else:
        return "This word is neutral (no sentiment)"


def get_similar_words(word: str, lexicon: pd.DataFrame, limit: int = 5) -> list:
    """Get similar words using basic string matching"""
    if lexicon is None or lexicon.empty:
        return []
    
    # Find words that start with the same letters
    similar = lexicon[lexicon['word'].str.startswith(word[:2])]
    
    if similar.empty:
        # If no matches, just return first few words
        return lexicon.head(limit)['word'].tolist()
    
    return similar.head(limit)['word'].tolist()


@router.post("/reset")
async def reset_processing():
    """Reset processing status (does not affect cached dictionary)"""
    global processing_status
    
    processing_status = {
        "status": "idle",
        "progress": 0,
        "message": "",
        "result_file": None,
        "stats": {}
    }
    
    return {"message": "Processing status reset successfully"}