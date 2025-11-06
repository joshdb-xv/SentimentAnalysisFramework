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

router = APIRouter(prefix="/lexical", tags=["Lexical Dictionary"])

# Directory setup
UPLOAD_DIR = "uploads/lexical"
OUTPUT_DIR = "outputs/lexical"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Store processing status
processing_status = {
    "status": "idle",  # idle, processing, completed, error
    "progress": 0,
    "message": "",
    "result_file": None,
    "stats": {}
}


@router.post("/upload-dictionary")
async def upload_dictionary(file: UploadFile = File(...)):
    """Upload the main dictionary Excel file"""
    try:
        print(f"Received file: {file.filename}")
        
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are allowed")
        
        # Create unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(UPLOAD_DIR, f"dictionary_{timestamp}.xlsx")
        
        print(f"Saving to: {file_path}")
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"File saved successfully. Size: {os.path.getsize(file_path)} bytes")
        
        # Quick validation - read Excel
        print("Reading Excel file...")
        df = pd.read_excel(file_path)
        
        row_count = len(df)
        columns = df.columns.tolist()
        
        print(f"Excel loaded: {row_count} rows, columns: {columns}")
        
        # Check for expected columns
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
    """Upload the climate keywords CSV file (alternative to loading from fixed location)"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(UPLOAD_DIR, f"climate_keywords_{timestamp}.csv")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Quick validation
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


from pydantic import BaseModel

class ProcessRequest(BaseModel):
    dictionary_path: str
    climate_keywords_path: Optional[str] = None

@router.post("/process")
async def process_lexical_dictionary(
    background_tasks: BackgroundTasks,
    request: ProcessRequest
):
    """
    Start processing the lexical dictionary
    Note: FastText is always used (models pre-loaded at startup)
    
    Parameters:
    - dictionary_path: Path to uploaded Excel file
    - climate_keywords_path: Path to climate keywords CSV (optional, uses default if not provided)
    """
    global processing_status
    
    dictionary_path = request.dictionary_path
    climate_keywords_path = request.climate_keywords_path
    
    try:
        if processing_status["status"] == "processing":
            raise HTTPException(status_code=400, detail="Processing already in progress")
        
        # Verify dictionary file exists
        if not os.path.exists(dictionary_path):
            raise HTTPException(status_code=404, detail=f"Dictionary file not found: {dictionary_path}")
        
        # Use default climate keywords path if not provided
        if not climate_keywords_path:
            climate_keywords_path = "data/weatherkeywords.csv"
            if not os.path.exists(climate_keywords_path):
                raise HTTPException(
                    status_code=404,
                    detail="Default climate keywords file not found. Please ensure data/weatherkeywords.csv exists."
                )
        
        # Verify keywords file exists
        if not os.path.exists(climate_keywords_path):
            raise HTTPException(status_code=404, detail=f"Keywords file not found: {climate_keywords_path}")
        
        # Reset status
        processing_status = {
            "status": "processing",
            "progress": 0,
            "message": "Starting processing...",
            "result_file": None,
            "stats": {}
        }
        
        # Start background processing
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
        
        # Update progress callbacks
        def update_progress(progress, message):
            processing_status["progress"] = progress
            processing_status["message"] = message
            print(f"Progress: {progress}% - {message}")
        
        processor.set_progress_callback(update_progress)
        
        # Process
        print("Starting processing...")
        result = processor.process()
        
        print(f"\nProcessing completed!")
        print(f"Result keys: {result.keys()}")
        print(f"Stats: {result.get('stats', {})}")
        
        # Save result
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(OUTPUT_DIR, f"lexical_dictionary_{timestamp}.csv")
        result["dataframe"].to_csv(output_file, index=False)
        
        print(f"Saved to: {output_file}")
        
        # Update status
        processing_status["status"] = "completed"
        processing_status["progress"] = 100
        processing_status["message"] = "Processing completed successfully"
        processing_status["result_file"] = output_file
        processing_status["stats"] = result["stats"]
        
        print(f"Final status: {processing_status}")
        
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


@router.post("/reset")
async def reset_processing():
    """Reset processing status"""
    global processing_status
    processing_status = {
        "status": "idle",
        "progress": 0,
        "message": "",
        "result_file": None,
        "stats": {}
    }
    return {"message": "Status reset successfully"}