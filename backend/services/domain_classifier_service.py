from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel
from fastapi import UploadFile
import shutil
import logging
import json

# Import the domain classifier trainer
import sys
backend_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(backend_dir / "scripts"))

from train_domain_classifier import ClimateDomainClassifier

logger = logging.getLogger(__name__)

# -----------------------------
# Request Models
# -----------------------------
class DomainInitialTrainingRequest(BaseModel):
    replace_existing: bool = False

class DomainRetrainRequest(BaseModel):
    batch_name: str

class DomainPseudoLabelRequest(BaseModel):
    confidence_threshold: float = 0.9
    save_low_confidence: bool = True

class DomainPredictionRequest(BaseModel):
    text: str

# -----------------------------
# Training History Manager (Same pattern as category classifier)
# -----------------------------
class DomainTrainingHistoryManager:
    """Tracks training history for domain classifier"""
    
    def __init__(self, history_file: Path):
        self.history_file = history_file
        self.history = self._load_history()
        self._migrate_old_format()
    
    def _load_history(self) -> List[Dict[str, Any]]:
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading history: {e}")
                return []
        return []
    
    def _migrate_old_format(self):
        """Migrate old flat benchmark format to nested format"""
        migrated = False
        for batch in self.history:
            benchmarks = batch.get('benchmarks', {})
            
            if 'accuracy' in benchmarks and 'overall_metrics' not in benchmarks:
                batch['benchmarks'] = {
                    'overall_metrics': {
                        'accuracy': benchmarks.get('accuracy', 0),
                        'precision_weighted': benchmarks.get('precision_weighted', 0),
                        'recall_weighted': benchmarks.get('recall_weighted', 0),
                        'f1_weighted': benchmarks.get('f1_weighted', 0),
                        'precision_macro': benchmarks.get('precision_macro', 0),
                        'recall_macro': benchmarks.get('recall_macro', 0),
                        'f1_macro': benchmarks.get('f1_macro', 0)
                    }
                }
                migrated = True
        
        if migrated:
            self._save_history()
    
    def _save_history(self):
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def add_batch(self, batch_info: Dict[str, Any]):
        self.history.append(batch_info)
        self._save_history()
    
    def get_history(self) -> List[Dict[str, Any]]:
        return self.history
    
    def get_improvement_stats(self) -> Dict[str, Any]:
        if len(self.history) < 2:
            return {
                "total_batches": len(self.history),
                "improvements": [],
                "overall_improvement": 0,
                "message": "Need at least 2 batches to calculate improvement"
            }
        
        improvements = []
        for i in range(1, len(self.history)):
            prev_batch = self.history[i-1]
            curr_batch = self.history[i]
            
            prev_acc = prev_batch['benchmarks']['overall_metrics']['accuracy']
            curr_acc = curr_batch['benchmarks']['overall_metrics']['accuracy']
            
            improvement = {
                'from_batch': prev_batch['batch_name'],
                'to_batch': curr_batch['batch_name'],
                'prev_accuracy': prev_acc * 100,
                'curr_accuracy': curr_acc * 100,
                'improvement_percent': (curr_acc - prev_acc) * 100,
                'prev_samples': prev_batch['training_samples'],
                'curr_samples': curr_batch['training_samples'],
                'added_samples': curr_batch['training_samples'] - prev_batch['training_samples']
            }
            improvements.append(improvement)
        
        final_acc = self.history[-1]['benchmarks']['overall_metrics']['accuracy']
        initial_acc = self.history[0]['benchmarks']['overall_metrics']['accuracy']
        
        return {
            'total_batches': len(self.history),
            'improvements': improvements,
            'overall_improvement': (final_acc - initial_acc) * 100
        }

# -----------------------------
# Domain Classifier Service
# -----------------------------
class DomainClassifierService:
    def __init__(self):
        self.trainer = ClimateDomainClassifier()
        self.current_model_name = None
        
        # Paths for staged data (domain-specific directories)
        self.staged_training_dir = self.trainer.data_dir / "staged_training_domain"
        self.staged_unlabeled_dir = self.trainer.data_dir / "staged_unlabeled_domain"
        self.low_confidence_dir = self.trainer.data_dir / "low_confidence_domain"
        
        # Create directories
        self.staged_training_dir.mkdir(parents=True, exist_ok=True)
        self.staged_unlabeled_dir.mkdir(parents=True, exist_ok=True)
        self.low_confidence_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        history_file = self.trainer.data_dir / "training_history_domain.json"
        self.history_manager = DomainTrainingHistoryManager(history_file)
        
        # Auto-load latest model
        self._auto_load_latest_model()
    
    def _auto_load_latest_model(self):
        """Automatically load the most recent domain model on startup"""
        models = self._list_models()
        if models:
            latest_model = models[0]
            try:
                self.trainer.load_model(latest_model["name"])
                self.current_model_name = latest_model["name"]
                logger.info(f"Auto-loaded latest domain model: {latest_model['name']}")
            except Exception as e:
                logger.warning(f"Could not auto-load domain model: {e}")
    
    def _list_models(self) -> List[Dict[str, Any]]:
        """List domain classifier models"""
        models = []
        # Look for domain-specific model files
        model_files = [f for f in self.trainer.model_dir.glob("*domain*.joblib") if "_metadata" not in f.name]
        
        for model_file in model_files:
            metadata_file = self.trainer.model_dir / f"{model_file.stem}_metadata.joblib"
            models.append({
                "name": model_file.name,
                "size_kb": model_file.stat().st_size / 1024,
                "modified": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
                "has_metadata": metadata_file.exists()
            })
        
        models.sort(key=lambda x: x["modified"], reverse=True)
        return models
    
    def _archive_current_model(self):
        """Archive current model before creating new one"""
        if not self.current_model_name:
            return
        
        archive_dir = self.trainer.model_dir / "archived"
        archive_dir.mkdir(exist_ok=True)
        
        model_path = self.trainer.model_dir / self.current_model_name
        metadata_path = self.trainer.model_dir / f"{model_path.stem}_metadata.joblib"
        
        if model_path.exists():
            shutil.move(str(model_path), str(archive_dir / self.current_model_name))
            logger.info(f"Archived domain model: {self.current_model_name}")
        
        if metadata_path.exists():
            shutil.move(str(metadata_path), str(archive_dir / metadata_path.name))
    
    def _get_all_staged_training_files(self) -> List[str]:
        """Get all CSV files from staged training directory"""
        files = list(self.staged_training_dir.glob("*.csv"))
        return [str(f) for f in files]
    
    def _count_staged_samples(self) -> int:
        """Count total samples in staged training data"""
        total = 0
        for csv_file in self.staged_training_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                total += len(df)
            except:
                pass
        return total
    
    # -----------------------------
    # File Upload
    # -----------------------------
    def upload_training_data(self, file: UploadFile) -> Dict[str, Any]:
        """Upload labeled training data (text, label columns)"""
        filename = Path(file.filename).name
        filepath = self.staged_training_dir / filename
        
        try:
            with open(filepath, "wb") as f:
                shutil.copyfileobj(file.file, f)
        except Exception as e:
            raise ValueError(f"Failed to save file: {str(e)}")
        
        try:
            df = pd.read_csv(filepath)
            if 'text' not in df.columns or 'label' not in df.columns:
                filepath.unlink()
                raise ValueError("CSV must contain 'text' and 'label' columns (0=not climate, 1=climate)")
            
            row_count = len(df)
            logger.info(f"Uploaded domain training data: {filename} ({row_count} rows)")
            
            return {
                "success": True,
                "filename": filename,
                "rows": row_count,
                "total_staged_samples": self._count_staged_samples()
            }
        except Exception as e:
            if filepath.exists():
                filepath.unlink()
            raise e
    
    def upload_unlabeled_data(self, file: UploadFile) -> Dict[str, Any]:
        """Upload unlabeled data for pseudo-labeling"""
        filename = Path(file.filename).name
        filepath = self.staged_unlabeled_dir / filename
        
        try:
            with open(filepath, "wb") as f:
                shutil.copyfileobj(file.file, f)
        except Exception as e:
            raise ValueError(f"Failed to save file: {str(e)}")
        
        try:
            df = pd.read_csv(filepath)
            if 'text' not in df.columns:
                filepath.unlink()
                raise ValueError("CSV must contain 'text' column")
            
            row_count = len(df)
            logger.info(f"Uploaded unlabeled data for domain: {filename} ({row_count} rows)")
            
            return {
                "success": True,
                "filename": filename,
                "rows": row_count
            }
        except Exception as e:
            if filepath.exists():
                filepath.unlink()
            raise e
    
    # -----------------------------
    # Model Status
    # -----------------------------
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        models = self._list_models()
        
        return {
            "has_model": self.current_model_name is not None,
            "current_model": self.current_model_name,
            "total_models": len(models),
            "staged_training_samples": self._count_staged_samples(),
            "staged_unlabeled_files": len(list(self.staged_unlabeled_dir.glob("*.csv"))),
            "low_confidence_files": len(list(self.low_confidence_dir.glob("*.csv"))),
            "training_batches": len(self.history_manager.get_history())
        }
    
    # -----------------------------
    # Training Methods (similar pattern to category classifier)
    # -----------------------------
    def train_initial_model(self, replace_existing: bool = False) -> Dict[str, Any]:
        """Train initial domain classifier model"""
        
        if self.current_model_name and not replace_existing:
            return {
                "success": False,
                "message": "Model already exists. Set replace_existing=True to replace it."
            }
        
        training_files = self._get_all_staged_training_files()
        
        if not training_files:
            raise ValueError("No training data found. Please upload training CSV files first.")
        
        logger.info(f"Training initial domain model with {len(training_files)} file(s)")
        
        if replace_existing and self.current_model_name:
            self._archive_current_model()
        
        # Train with 5 runs
        summary = self.trainer.evaluate_with_multiple_runs(training_files, n_runs=5)
        
        best_run = max(summary['runs'], key=lambda x: x['accuracy'])
        logger.info(f"Best run: seed={best_run['seed']}, accuracy={best_run['accuracy']:.4f}")
        
        # Train final model with best seed
        df, X, y = self.trainer.load_data(training_files)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=best_run['seed'], stratify=y
        )
        
        self.trainer.pipeline = self.trainer.create_pipeline()
        self.trainer.pipeline.fit(X_train, y_train)
        
        evaluation_results = self.trainer.evaluate_model_detailed(X_test, y_test)
        
        results = {
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'evaluation': evaluation_results,
            'training_timestamp': datetime.now().isoformat(),
            'multiple_runs': summary,
            'best_run_seed': best_run['seed'],
            'test_accuracy': evaluation_results['overall_metrics']['accuracy'],
            'classification_report': evaluation_results['classification_report'],
            'confusion_matrix': evaluation_results['confusion_matrix'],
            'classes': evaluation_results['classes']
        }
        
        # Save model
        model_name = f"climate_domain_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        self.trainer.save_model(model_name, results)
        self.current_model_name = model_name
        
        # Export benchmarks
        from scripts.train_domain_classifier import export_domain_benchmarks_to_json
        metadata = self.trainer.load_model(model_name)
        if metadata:
            export_domain_benchmarks_to_json(metadata)
        
        # Add to history
        batch_info = {
            'batch_name': 'initial_training',
            'batch_number': 1,
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'benchmarks': {
                'overall_metrics': evaluation_results['overall_metrics']
            },
            'multiple_runs_stats': summary['statistics']
        }
        self.history_manager.add_batch(batch_info)
        
        return {
            "success": True,
            "model_name": model_name,
            "training_samples": len(X_train),
            "benchmarks": results['multiple_runs']['statistics'],
            "batch_info": batch_info
        }
    
    def retrain_model(self, batch_name: str) -> Dict[str, Any]:
        """Retrain domain model with new labeled data"""
        
        if not self.current_model_name:
            raise ValueError("No existing model found. Train an initial model first.")
        
        training_files = self._get_all_staged_training_files()
        
        if not training_files:
            raise ValueError("No training data found.")
        
        logger.info(f"Retraining domain model with batch: {batch_name}")
        
        # Train with 5 runs
        summary = self.trainer.evaluate_with_multiple_runs(training_files, n_runs=5)
        
        best_run = max(summary['runs'], key=lambda x: x['accuracy'])
        
        # Train final model
        df, X, y = self.trainer.load_data(training_files)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=best_run['seed'], stratify=y
        )
        
        self.trainer.pipeline = self.trainer.create_pipeline()
        self.trainer.pipeline.fit(X_train, y_train)
        
        evaluation_results = self.trainer.evaluate_model_detailed(X_test, y_test)
        
        results = {
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'evaluation': evaluation_results,
            'training_timestamp': datetime.now().isoformat(),
            'multiple_runs': summary,
            'best_run_seed': best_run['seed'],
            'test_accuracy': evaluation_results['overall_metrics']['accuracy'],
            'classification_report': evaluation_results['classification_report'],
            'confusion_matrix': evaluation_results['confusion_matrix'],
            'classes': evaluation_results['classes']
        }
        
        # Archive old model
        self._archive_current_model()
        
        # Save new model
        model_name = f"climate_domain_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        self.trainer.save_model(model_name, results)
        self.current_model_name = model_name
        
        # Export benchmarks
        from scripts.train_domain_classifier import export_domain_benchmarks_to_json
        metadata = self.trainer.load_model(model_name)
        if metadata:
            export_domain_benchmarks_to_json(metadata)
        
        # Calculate improvement
        history = self.history_manager.get_history()
        prev_batch = history[-1] if history else None
        
        improvement = None
        if prev_batch:
            if 'overall_metrics' in prev_batch.get('benchmarks', {}):
                prev_acc = prev_batch['benchmarks']['overall_metrics']['accuracy']
            else:
                prev_acc = prev_batch['benchmarks'].get('accuracy', 0)
            curr_acc = evaluation_results['overall_metrics']['accuracy']
            improvement = (curr_acc - prev_acc) * 100
        
        # Add to history
        batch_info = {
            'batch_name': batch_name,
            'batch_number': len(history) + 1,
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'benchmarks': {
                'overall_metrics': evaluation_results['overall_metrics']
            },
            'multiple_runs_stats': summary['statistics'],
            'improvement_from_previous': improvement
        }
        self.history_manager.add_batch(batch_info)
        
        return {
            "success": True,
            "model_name": model_name,
            "training_samples": len(X_train),
            "benchmarks": results['multiple_runs']['statistics'],
            "batch_info": batch_info,
            "improvement": improvement
        }
    
    def pseudo_label_unlabeled_data(
        self,
        confidence_threshold: float = 0.9,
        save_low_confidence: bool = True
    ) -> Dict[str, Any]:
        """Pseudo-label all unlabeled data"""
        
        if not self.current_model_name:
            raise ValueError("No model loaded. Train a model first.")
        
        unlabeled_files = list(self.staged_unlabeled_dir.glob("*.csv"))
        
        if not unlabeled_files:
            raise ValueError("No unlabeled data found. Upload unlabeled CSV files first.")
        
        logger.info(f"Pseudo-labeling {len(unlabeled_files)} file(s)")
        
        all_high_confidence = []
        all_low_confidence = []
        total_processed = 0
        
        for unlabeled_file in unlabeled_files:
            df = pd.read_csv(unlabeled_file)
            if 'text' not in df.columns:
                logger.warning(f"Skipping {unlabeled_file.name}: no 'text' column")
                continue
            
            df['processed_text'] = df['text'].apply(self.trainer.preprocess_text)
            df = df[df['processed_text'].str.len() > 0].reset_index(drop=True)
            
            # Predict
            predictions = self.trainer.pipeline.predict(df['processed_text'].values)
            probabilities = self.trainer.pipeline.predict_proba(df['processed_text'].values)
            
            df['label'] = predictions
            df['confidence'] = probabilities.max(axis=1)
            
            # Split by confidence
            high_conf = df[df['confidence'] >= confidence_threshold].copy()
            low_conf = df[df['confidence'] < confidence_threshold].copy()
            
            all_high_confidence.append(high_conf)
            all_low_confidence.append(low_conf)
            total_processed += len(df)
        
        # Combine results
        high_conf_df = pd.concat(all_high_confidence, ignore_index=True) if all_high_confidence else pd.DataFrame()
        low_conf_df = pd.concat(all_low_confidence, ignore_index=True) if all_low_confidence else pd.DataFrame()
        
        # Save high confidence to training directory
        if len(high_conf_df) > 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            high_conf_filename = f"pseudo_labeled_domain_{timestamp}.csv"
            high_conf_path = self.staged_training_dir / high_conf_filename
            high_conf_df[['text', 'label']].to_csv(high_conf_path, index=False)
            logger.info(f"Saved {len(high_conf_df)} high-confidence labels to {high_conf_filename}")
        
        # Save low confidence for manual labeling
        low_conf_filename = None
        if save_low_confidence and len(low_conf_df) > 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            low_conf_filename = f"low_confidence_domain_{timestamp}.csv"
            low_conf_path = self.low_confidence_dir / low_conf_filename
            low_conf_df[['text', 'label', 'confidence']].to_csv(low_conf_path, index=False)
            logger.info(f"Saved {len(low_conf_df)} low-confidence samples to {low_conf_filename}")
        
        # Move processed unlabeled files to archive
        archive_dir = self.staged_unlabeled_dir / "processed"
        archive_dir.mkdir(exist_ok=True)
        for unlabeled_file in unlabeled_files:
            shutil.move(str(unlabeled_file), str(archive_dir / unlabeled_file.name))
        
        # Class distribution
        class_dist = {}
        if len(high_conf_df) > 0:
            class_counts = high_conf_df['label'].value_counts()
            class_dist = {str(k): int(v) for k, v in class_counts.items()}
        
        return {
            "success": True,
            "total_processed": total_processed,
            "high_confidence_count": len(high_conf_df),
            "low_confidence_count": len(low_conf_df),
            "high_confidence_file": high_conf_filename if len(high_conf_df) > 0 else None,
            "low_confidence_file": low_conf_filename,
            "class_distribution": class_dist,
            "confidence_threshold": confidence_threshold,
            "staged_training_samples": self._count_staged_samples()
        }
    
    def get_training_history(self) -> Dict[str, Any]:
        """Get complete training history"""
        return {
            "history": self.history_manager.get_history(),
            "improvement_stats": self.history_manager.get_improvement_stats()
        }
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """Predict a single text"""
        if not self.trainer.pipeline:
            raise ValueError("No model loaded")
        
        processed = self.trainer.preprocess_text(text)
        if not processed:
            raise ValueError("Text became empty after preprocessing")
        
        prediction = self.trainer.pipeline.predict([processed])[0]
        probabilities = self.trainer.pipeline.predict_proba([processed])[0]
        classes = self.trainer.pipeline.classes_
        
        prob_dict = {str(c): float(p) for c, p in zip(classes, probabilities)}
        
        return {
            "prediction": int(prediction),
            "is_climate_related": bool(prediction == 1),
            "confidence": float(probabilities.max()),
            "probabilities": prob_dict,
            "processed_text": processed
        }

# -----------------------------
# Singleton Instance
# -----------------------------
_domain_classifier_service = None

def get_domain_classifier_service() -> DomainClassifierService:
    global _domain_classifier_service
    if _domain_classifier_service is None:
        _domain_classifier_service = DomainClassifierService()
    return _domain_classifier_service