from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel
from services.domain_classifier_service import (
    get_domain_classifier_service,
    DomainInitialTrainingRequest,
    DomainRetrainRequest,
    DomainPseudoLabelRequest,
    DomainPredictionRequest
)

router = APIRouter(prefix="/domain-classifier", tags=["Domain Classifier"])

# -----------------------------
# Response Models
# -----------------------------
class DomainStatusResponse(BaseModel):
    has_model: bool
    current_model: Optional[str]
    total_models: int
    staged_training_samples: int
    staged_unlabeled_files: int
    low_confidence_files: int
    training_batches: int

# -----------------------------
# System Status
# -----------------------------
@router.get("/status", response_model=DomainStatusResponse)
async def get_status():
    """Get current domain classifier system status"""
    try:
        service = get_domain_classifier_service()
        status = service.get_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# File Upload Endpoints
# -----------------------------
@router.post("/upload/training")
async def upload_training_data(file: UploadFile = File(...)):
    """
    Upload labeled training data CSV
    Required columns: text, label (0=not climate, 1=climate)
    """
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are accepted")
        
        service = get_domain_classifier_service()
        result = service.upload_training_data(file)
        
        return {
            "success": True,
            "message": f"Uploaded {result['rows']} training samples",
            "filename": result['filename'],
            "rows": result['rows'],
            "total_staged_samples": result['total_staged_samples']
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload/unlabeled")
async def upload_unlabeled_data(file: UploadFile = File(...)):
    """
    Upload unlabeled data CSV for pseudo-labeling
    Required columns: text
    """
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are accepted")
        
        service = get_domain_classifier_service()
        result = service.upload_unlabeled_data(file)
        
        return {
            "success": True,
            "message": f"Uploaded {result['rows']} unlabeled samples",
            "filename": result['filename'],
            "rows": result['rows']
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# 1. Initial Training
# -----------------------------
@router.post("/train/initial")
async def train_initial_model(request: DomainInitialTrainingRequest):
    """
    Train initial domain classifier model (binary: climate/not climate)
    
    - If no model exists: trains a new model
    - If model exists and replace_existing=True: archives old model and trains new
    - If model exists and replace_existing=False: returns error
    
    Always performs 5 runs with different seeds (default behavior)
    """
    try:
        service = get_domain_classifier_service()
        result = service.train_initial_model(replace_existing=request.replace_existing)
        
        if not result['success']:
            return JSONResponse(status_code=409, content=result)
        
        return {
            "success": True,
            "message": "Initial domain model trained successfully",
            "model_name": result['model_name'],
            "training_samples": result['training_samples'],
            "benchmarks": result['benchmarks'],
            "batch_info": result['batch_info']
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# 2. Retrain with Additional Data
# -----------------------------
@router.post("/retrain")
async def retrain_model(request: DomainRetrainRequest):
    """
    Retrain domain classifier with additional labeled data
    
    - Archives current model
    - Trains new model with ALL staged training data
    - Tracks improvement from previous batch
    - Always performs 5 runs with different seeds
    
    Use this after:
    - Uploading new manually labeled data
    - Pseudo-labeling unlabeled tweets (high-confidence ones auto-added to training)
    """
    try:
        service = get_domain_classifier_service()
        result = service.retrain_model(batch_name=request.batch_name)
        
        return {
            "success": True,
            "message": f"Domain model retrained successfully: {request.batch_name}",
            "model_name": result['model_name'],
            "training_samples": result['training_samples'],
            "benchmarks": result['benchmarks'],
            "batch_info": result['batch_info'],
            "improvement": result['improvement']
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# 3. Pseudo-Labeling
# -----------------------------
@router.post("/pseudo-label")
async def pseudo_label_data(request: DomainPseudoLabelRequest):
    """
    Pseudo-label all staged unlabeled data (domain classification)
    
    Process:
    1. Loads all unlabeled CSVs from staged_unlabeled_domain directory
    2. Predicts labels using current model (0=not climate, 1=climate)
    3. High confidence (≥ threshold) → added to training data automatically
    4. Low confidence (< threshold) → saved for manual labeling
    5. Processed files moved to archive
    
    After pseudo-labeling, use /retrain to improve model with new data
    """
    try:
        service = get_domain_classifier_service()
        result = service.pseudo_label_unlabeled_data(
            confidence_threshold=request.confidence_threshold,
            save_low_confidence=request.save_low_confidence
        )
        
        return {
            "success": True,
            "message": f"Pseudo-labeled {result['total_processed']} tweets",
            "total_processed": result['total_processed'],
            "high_confidence_count": result['high_confidence_count'],
            "low_confidence_count": result['low_confidence_count'],
            "high_confidence_file": result['high_confidence_file'],
            "low_confidence_file": result['low_confidence_file'],
            "class_distribution": result['class_distribution'],
            "confidence_threshold": result['confidence_threshold'],
            "staged_training_samples": result['staged_training_samples']
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# 4. Training History & Improvement
# -----------------------------
@router.get("/history")
async def get_training_history():
    """
    Get complete training history and improvement statistics
    
    Shows:
    - All training batches with their metrics
    - Improvement percentages between batches
    - Dataset growth over time
    - Overall improvement from first to latest batch
    
    Perfect for thesis analysis and visualization
    """
    try:
        service = get_domain_classifier_service()
        history = service.get_training_history()
        
        return {
            "success": True,
            "history": history['history'],
            "improvement_stats": history['improvement_stats']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# 5. Predictions
# -----------------------------
@router.post("/predict")
async def predict_single(request: DomainPredictionRequest):
    """
    Make a prediction on a single text using the current domain model
    Returns: 0 (not climate) or 1 (climate-related)
    """
    try:
        service = get_domain_classifier_service()
        result = service.predict_single(request.text)
        
        return {
            "success": True,
            "prediction": result['prediction'],
            "is_climate_related": result['is_climate_related'],
            "confidence": result['confidence'],
            "probabilities": result['probabilities'],
            "processed_text": result['processed_text']
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# Helper Endpoints
# -----------------------------
@router.get("/info")
async def get_system_info():
    """Get detailed domain classifier system information"""
    try:
        service = get_domain_classifier_service()
        status = service.get_status()
        
        # Get current model benchmarks if available
        current_benchmarks = None
        if status['has_model']:
            history = service.get_training_history()
            if history['history']:
                latest_batch = history['history'][-1]
                current_benchmarks = latest_batch['benchmarks']
        
        return {
            "status": status,
            "current_benchmarks": current_benchmarks,
            "classifier_type": "binary",
            "classes": {
                "0": "Not Climate-Related",
                "1": "Climate-Related"
            },
            "workflow": {
                "step_1": "Upload training CSV files (text, label columns where label is 0 or 1)",
                "step_2": "Train initial model or retrain with new data",
                "step_3": "Upload unlabeled CSV files (text column)",
                "step_4": "Run pseudo-labeling to auto-label high-confidence tweets",
                "step_5": "Retrain model with expanded dataset",
                "step_6": "Check /history for improvement metrics"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))