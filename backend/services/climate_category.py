# services/climate_category_classifier.py

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# -----------------------------
# Climate Categories
# -----------------------------
CLIMATE_CATEGORIES = [
    "Storms, Typhoons, and Wind Events",
    "Coastal & Flooding Hazards",
    "Extreme Heat / Heatwaves",
    "Pollution",
    "Cold Weather / Temperature Drops",
    "Geological Events",
    "General Weather"
]

# -----------------------------
# Model loading
# -----------------------------
# Allow override via env var; fallback to default name
CATEGORY_MODEL_FILENAME = os.getenv("CLIMATE_CATEGORY_MODEL_FILENAME", "climate_category_classifier.joblib")
CATEGORY_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / CATEGORY_MODEL_FILENAME

_category_classifier: Optional[Pipeline] = None
_category_load_error: Optional[str] = None
_category_metadata: Optional[Dict] = None

def _load_category_model() -> Tuple[Optional[Pipeline], Optional[str], Optional[Dict]]:
    global _category_classifier, _category_load_error, _category_metadata
    if _category_classifier is not None:
        return _category_classifier, None, _category_metadata
    
    try:
        _category_classifier = joblib.load(CATEGORY_MODEL_PATH)
        _category_load_error = None
        
        # Try to load metadata
        metadata_path = CATEGORY_MODEL_PATH.parent / f"{CATEGORY_MODEL_PATH.stem}_metadata.joblib"
        if metadata_path.exists():
            _category_metadata = joblib.load(metadata_path)
        else:
            _category_metadata = None
            
        print(f"[INFO] Climate category classifier loaded from {CATEGORY_MODEL_PATH}")
    except Exception as e:
        _category_classifier = None
        _category_load_error = f"Failed to load category model from {CATEGORY_MODEL_PATH}: {e}"
        _category_metadata = None
        print(f"[ERROR] {_category_load_error}")
    
    return _category_classifier, _category_load_error, _category_metadata

def category_model_status() -> Dict[str, Union[bool, str, List, Dict]]:
    clf, err, metadata = _load_category_model()
    status = {
        "loaded": clf is not None,
        "detail": "loaded" if err is None else err,
        "categories": CLIMATE_CATEGORIES
    }
    
    if clf is not None and hasattr(clf, 'classes_'):
        status["model_classes"] = clf.classes_.tolist()
    
    if metadata:
        status["metadata"] = {
            "save_timestamp": metadata.get("save_timestamp"),
            "model_type": metadata.get("model_type"),
            "n_features": metadata.get("n_features")
        }
    
    return status

# -----------------------------
# Preprocessing (mirror the training script)
# -----------------------------
def preprocess_text_for_category(text: str) -> str:
    """
    Preprocess text for category classification - mirrors the training script
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove mentions but keep hashtag content
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag text
    
    # Preserve climate-related terms and numbers (temperatures, measurements)
    # Remove most punctuation but keep hyphens in compound words and decimal points
    text = re.sub(r'[^\w\s\-\.]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# -----------------------------
# Category classification functions
# -----------------------------
def categorize_tweet(raw_text: str) -> Dict:
    """
    Categorize a climate-related tweet into one of the 9 climate categories.
    
    Returns:
      {
        "prediction": str,  # The predicted category
        "confidence": float,
        "probabilities": Dict[str, float]  # All category probabilities sorted by descending probability
      }
      or { "error": "..." }
    """
    clf, err, metadata = _load_category_model()
    if clf is None:
        return {"error": err or "Category model not loaded"}
    
    processed = preprocess_text_for_category(raw_text)
    if not processed:
        return {"error": "Empty text after preprocessing"}
    
    try:
        probs = clf.predict_proba([processed])[0]
        pred_idx = int(np.argmax(probs))
        predicted_category = clf.classes_[pred_idx]
        confidence = float(np.max(probs))
        
        # Create probability dictionary for all categories and sort by descending probability
        prob_list = [(clf.classes_[i], float(probs[i])) for i in range(len(clf.classes_))]
        prob_list.sort(key=lambda x: x[1], reverse=True)  # Sort by probability descending
        prob_dict = dict(prob_list)
        
        return {
            "prediction": predicted_category,
            "confidence": confidence,
            "probabilities": prob_dict
        }
        
    except Exception as e:
        return {"error": f"Category classification failed: {e}"}

def categorize_many_tweets(raw_texts: List[str]) -> List[Dict]:
    """
    Batch categorize list of climate-related tweets.
    """
    clf, err, metadata = _load_category_model()
    if clf is None:
        return [{"error": err or "Category model not loaded"} for _ in raw_texts]
    
    processed = [preprocess_text_for_category(t) for t in raw_texts]
    empty_mask = [len(t) == 0 for t in processed]
    
    # Prepare output
    outputs: List[Dict] = [None] * len(raw_texts)
    
    # Indices with non-empty processed text
    valid_idx = [i for i, is_empty in enumerate(empty_mask) if not is_empty]
    
    if valid_idx:
        try:
            subset = [processed[i] for i in valid_idx]
            proba = clf.predict_proba(subset)
            preds = np.argmax(proba, axis=1)
            
            for j, i in enumerate(valid_idx):
                predicted_category = clf.classes_[preds[j]]
                confidence = float(np.max(proba[j]))
                
                # Create probability dictionary and sort by descending probability
                prob_list = [(clf.classes_[k], float(proba[j][k])) for k in range(len(clf.classes_))]
                prob_list.sort(key=lambda x: x[1], reverse=True)  # Sort by probability descending
                prob_dict = dict(prob_list)
                
                outputs[i] = {
                    "prediction": predicted_category,
                    "confidence": confidence,
                    "probabilities": prob_dict
                }
                
        except Exception as e:
            # If batch fails, mark all as error
            return [{"error": f"Category classification failed: {e}"} for _ in raw_texts]
    
    # Fill empties with error
    for i in range(len(outputs)):
        if outputs[i] is None:
            outputs[i] = {"error": "Empty text after preprocessing"}
    
    return outputs

def get_category_insights(results: List[Dict]) -> Dict:
    """
    Analyze categorization results and provide insights
    """
    if not results:
        return {"error": "No results to analyze"}
    
    # Count categories
    category_counts = {}
    total_processed = 0
    total_errors = 0
    confidence_scores = []
    
    for result in results:
        if "error" in result:
            total_errors += 1
        else:
            total_processed += 1
            category = result.get("prediction")
            if category:
                category_counts[category] = category_counts.get(category, 0) + 1
                confidence_scores.append(result.get("confidence", 0))
    
    insights = {
        "total_tweets": len(results),
        "successfully_categorized": total_processed,
        "errors": total_errors,
        "category_distribution": category_counts
    }
    
    if confidence_scores:
        insights["confidence_stats"] = {
            "mean": float(np.mean(confidence_scores)),
            "std": float(np.std(confidence_scores)),
            "min": float(np.min(confidence_scores)),
            "max": float(np.max(confidence_scores)),
            "median": float(np.median(confidence_scores))
        }
    
    # Most common categories
    if category_counts:
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        insights["top_categories"] = sorted_categories[:5]
    
    return insights