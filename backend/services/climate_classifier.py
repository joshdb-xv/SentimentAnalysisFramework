# services/climate_classifier.py

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

# -----------------------------
# Model loading
# -----------------------------
# Allow override via env var; fallback to default name
MODEL_FILENAME = os.getenv("CLIMATE_MODEL_FILENAME", "climate_classifier.joblib")
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / MODEL_FILENAME

_classifier: Optional[Pipeline] = None
_load_error: Optional[str] = None

def _load_model() -> Tuple[Optional[Pipeline], Optional[str]]:
    global _classifier, _load_error
    if _classifier is not None:
        return _classifier, None
    try:
        _classifier = joblib.load(MODEL_PATH)
        _load_error = None
        print(f"[INFO] Climate classifier loaded from {MODEL_PATH}")
    except Exception as e:
        _classifier = None
        _load_error = f"Failed to load model from {MODEL_PATH}: {e}"
        print(f"[ERROR] {_load_error}")
    return _classifier, _load_error

def model_status() -> Dict[str, Union[bool, str]]:
    clf, err = _load_model()
    return {
        "loaded": clf is not None,
        "detail": "loaded" if err is None else err
    }

# -----------------------------
# Preprocessing (mirror training)
# Your trainer preprocessed before feeding TF-IDF.
# We mirror that here for inference consistency.
# -----------------------------
_URL_RE = re.compile(
    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
)
_MENTION_RE = re.compile(r'@\w+')
_HASHTAG_RE = re.compile(r'#(\w+)')
_NON_ALPHA_RE = re.compile(r'[^a-zA-Z\s\u00C0-\u017F\u1E00-\u1EFF]')
_MULTI_SPACE_RE = re.compile(r'\s+')

def preprocess_text(text: str) -> str:
    if text is None or not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub("", text)
    text = _MENTION_RE.sub("", text)
    text = _HASHTAG_RE.sub(r"\1", text)  # keep hashtag token
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text

def _format_classification_result(prediction: int, confidence: float, note: str = None) -> Dict:
    """Helper function to format classification results with simplified output"""
    # Assuming class 1 means climate-related, class 0 means not climate-related
    is_climate_related = bool(prediction == 1)
    
    result = {
        "prediction": prediction,
        "is_climate_related": is_climate_related,
        "confidence": confidence
    }
    
    if note:
        result["note"] = note
    
    return result

# -----------------------------
# Inference helpers
# -----------------------------
def classify_tweet(raw_text: str) -> Dict:
    """
    Returns:
      {
        "prediction": 0|1,
        "is_climate_related": bool,
        "confidence": float
      }
      or { "error": "..." }
    """
    clf, err = _load_model()
    if clf is None:
        return {"error": err or "Model not loaded"}

    processed = preprocess_text(raw_text)
    if not processed:
        return _format_classification_result(
            prediction=0,
            confidence=1.0,
            note="Empty after preprocessing; defaulting to not climate-related."
        )

    try:
        probs = clf.predict_proba([processed])[0]
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
        
        return _format_classification_result(
            prediction=pred,
            confidence=conf
        )
    except Exception as e:
        return {"error": f"Inference failed: {e}"}

def classify_many(raw_texts: List[str]) -> List[Dict]:
    """
    Batch classify list of texts. Keeps input order.
    """
    clf, err = _load_model()
    if clf is None:
        return [{"error": err or "Model not loaded"} for _ in raw_texts]

    processed = [preprocess_text(t) for t in raw_texts]
    empty_mask = [len(t) == 0 for t in processed]
    
    # Fallback for empty rows
    def default_record():
        return _format_classification_result(
            prediction=0,
            confidence=1.0,
            note="Empty after preprocessing; defaulting to not climate-related."
        )

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
                outputs[i] = _format_classification_result(
                    prediction=int(preds[j]),
                    confidence=float(np.max(proba[j]))
                )
        except Exception as e:
            # If batch fails, mark all as error
            return [{"error": f"Inference failed: {e}"} for _ in raw_texts]

    # Fill empties and any None
    for i in range(len(outputs)):
        if outputs[i] is None:
            outputs[i] = default_record()
    return outputs