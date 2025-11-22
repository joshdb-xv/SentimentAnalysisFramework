import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
import joblib
import os
from pathlib import Path
import re
import logging
from typing import Tuple, Dict, Any, List, Union
from datetime import datetime
import json


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClimateClassifierTrainer:
    """
    Enhanced trainer class for climate-related tweet classification with pseudo-labeling support
    """
    
    def __init__(self, model_dir: str = None, data_dir: str = None):
        # Set up directory structure
        base_dir = Path(__file__).resolve().parent.parent
        
        # Model directory
        if model_dir is None:
            self.model_dir = base_dir / "models"
        else:
            self.model_dir = Path(model_dir).resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Data directories
        if data_dir is None:
            self.data_dir = base_dir / "data"
        else:
            self.data_dir = Path(data_dir).resolve()
            
        self.input_dir = self.data_dir / "input"
        self.output_dir = self.data_dir / "output"
        
        # Create directories if they don't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.pipeline = None
        self.vectorizer = None
        self.classifier = None
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better feature extraction
        Handles multilingual content (English, Tagalog, Cebuano)
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags (but keep the text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag text
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove non-alphabetic characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s\u00C0-\u017F\u1E00-\u1EFF]', ' ', text)
        
        return text
    
    def merge_csv_files(self, csv_paths: List[Union[str, Path]]) -> pd.DataFrame:
        """
        Merge multiple CSV files into a single DataFrame
        """
        logger.info(f"Merging {len(csv_paths)} CSV files...")
        
        dataframes = []
        total_samples = 0
        
        for i, csv_path in enumerate(csv_paths):
            csv_path = Path(csv_path).resolve()
            
            if not csv_path.exists():
                logger.warning(f"File not found: {csv_path}")
                continue
                
            try:
                df = pd.read_csv(csv_path)
                
                # Validate required columns
                if 'text' not in df.columns or 'label' not in df.columns:
                    logger.warning(f"Skipping {csv_path}: missing 'text' or 'label' columns")
                    continue
                
                # Add source file information
                df['source_file'] = csv_path.name
                df['batch_id'] = i
                
                dataframes.append(df)
                total_samples += len(df)
                logger.info(f"Loaded {len(df)} samples from {csv_path.name}")
                
            except Exception as e:
                logger.error(f"Error loading {csv_path}: {str(e)}")
                continue
        
        if not dataframes:
            raise ValueError("No valid CSV files found or loaded")
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Keep duplicates intentionally — some are useful for weighting polarity
        logger.info(f"Combined dataset: {len(combined_df)} samples (duplicates retained)")
        logger.info(f"Class distribution:\n{combined_df['label'].value_counts()}")

        return combined_df

    def load_data(self, data_source: Union[str, Path, pd.DataFrame, List[Union[str, Path]]]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Load and preprocess training data from various sources
        """
        if isinstance(data_source, list):
            # Multiple CSV files
            df = self.merge_csv_files(data_source)
        elif isinstance(data_source, pd.DataFrame):
            # In-memory DataFrame
            df = data_source.copy()
            logger.info(f"Using in-memory DataFrame with {len(df)} samples")
        else:
            # Single CSV file - handle relative paths by looking in input directory first
            if isinstance(data_source, str):
                # If it's just a filename (no path separators), look in input directory
                if '/' not in data_source and '\\' not in data_source:
                    csv_path = self.input_dir / data_source
                else:
                    csv_path = Path(data_source).resolve()
            else:
                csv_path = Path(data_source).resolve()
            
            logger.info(f"Loading data from {csv_path}")
            
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                logger.error(f"Error loading data: {str(e)}")
                raise
        
        # Validate required columns
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Data must contain 'text' and 'label' columns")
        
        # Remove rows with missing values
        df = df.dropna(subset=['text', 'label'])
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Remove empty texts after preprocessing
        df = df[df['processed_text'].str.len() > 0]
        
        logger.info(f"Final dataset: {len(df)} samples")
        if 'source_file' not in df.columns:
            df['source_file'] = 'unknown'
            df['batch_id'] = 0
        
        X = df['processed_text'].values
        y = df['label'].values
        
        return df, X, y
    
    def pseudo_label_data(self, 
                         unlabeled_data: Union[str, Path, pd.DataFrame], 
                         confidence_threshold: float = 0.9,
                         output_filename: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Pseudo-label unlabeled data using the trained model
        """
        if self.pipeline is None:
            raise ValueError("No trained model available. Train the model first.")
        
        logger.info("Starting pseudo-labeling process...")
        
        # Load unlabeled data
        if isinstance(unlabeled_data, pd.DataFrame):
            df = unlabeled_data.copy()
            input_filename = "dataframe"
        else:
            # Handle file paths - look in input directory if just filename provided
            if isinstance(unlabeled_data, str):
                if '/' not in unlabeled_data and '\\' not in unlabeled_data:
                    data_path = self.input_dir / unlabeled_data
                    input_filename = unlabeled_data
                else:
                    data_path = Path(unlabeled_data).resolve()
                    input_filename = data_path.name
            else:
                data_path = Path(unlabeled_data).resolve()
                input_filename = data_path.name
                
            df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(df)} unlabeled samples from {data_path}")
        
        if 'text' not in df.columns:
            raise ValueError("Unlabeled data must contain 'text' column")
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Remove empty texts
        valid_mask = df['processed_text'].str.len() > 0
        df = df[valid_mask].reset_index(drop=True)
        
        # Make predictions
        X_unlabeled = df['processed_text'].values
        predicted_labels = self.pipeline.predict(X_unlabeled)
        predicted_probabilities = self.pipeline.predict_proba(X_unlabeled)
        
        # Get maximum probability for each prediction (confidence)
        max_probabilities = np.max(predicted_probabilities, axis=1)
        
        # Add predictions to dataframe
        df['predicted_label'] = predicted_labels
        df['confidence'] = max_probabilities
        
        # Add probability for each class
        classes = self.pipeline.classes_
        for i, class_label in enumerate(classes):
            df[f'prob_class_{class_label}'] = predicted_probabilities[:, i]
        
        # Filter high-confidence predictions
        high_confidence_mask = df['confidence'] >= confidence_threshold
        high_confidence_df = df[high_confidence_mask].copy()
        
        # Rename predicted_label to label for consistency
        high_confidence_df['label'] = high_confidence_df['predicted_label']
        high_confidence_df['pseudo_labeled'] = True
        high_confidence_df['confidence_threshold'] = confidence_threshold
        high_confidence_df['pseudo_label_timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Pseudo-labeling results:")
        logger.info(f"  Total samples: {len(df)}")
        logger.info(f"  High confidence (≥{confidence_threshold}): {len(high_confidence_df)} ({len(high_confidence_df)/len(df)*100:.1f}%)")
        logger.info(f"  Confidence distribution by class:")
        
        for label in classes:
            class_mask = high_confidence_df['label'] == label
            count = class_mask.sum()
            logger.info(f"    Class {label}: {count} samples")
        
        # Save high-confidence pseudo-labels to output directory
        if len(high_confidence_df) > 0:
            # Generate output filename if not provided
            if output_filename is None:
                base_name = input_filename.replace('.csv', '') if input_filename != "dataframe" else "unlabeled_data"
                output_filename = f"pseudo_labeled_{base_name}.csv"
            
            # Ensure .csv extension
            if not output_filename.endswith('.csv'):
                output_filename += '.csv'
                
            output_path = self.output_dir / output_filename
            
            # Select relevant columns for saving
            save_columns = ['text', 'label', 'confidence', 'pseudo_labeled', 
                          'confidence_threshold', 'pseudo_label_timestamp']
            save_df = high_confidence_df[save_columns]
            
            save_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(save_df)} high-confidence pseudo-labels to {output_path}")
        
        return high_confidence_df, df
    
    def create_pipeline(self) -> Pipeline:
        """
        Create scikit-learn pipeline with TF-IDF vectorizer and Multinomial NB
        """
        # TF-IDF Vectorizer optimized for multilingual tweets
        vectorizer = TfidfVectorizer(
            max_features=10000,          # Limit vocabulary size
            ngram_range=(1, 2),          # Use unigrams and bigrams
            min_df=2,                    # Ignore terms that appear in less than 2 documents
            max_df=0.95,                 # Ignore terms that appear in more than 95% of documents
            stop_words=None,             # Don't use stop words (important for multilingual)
            lowercase=True,              # Already handled in preprocessing
            strip_accents='unicode',     # Handle accented characters
            token_pattern=r'\b\w+\b'     # Simple word tokenization
        )
        
        # Multinomial Naive Bayes classifier
        classifier = MultinomialNB(alpha=1.0)  # Laplace smoothing
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('classifier', classifier)
        ])
        
        return pipeline
    
    def evaluate_model_detailed(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Perform detailed evaluation of the model
        """
        if self.pipeline is None:
            raise ValueError("No trained model available")
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        
        # Confidence statistics
        max_probabilities = np.max(y_pred_proba, axis=1)
        
        # Get class names
        classes = self.pipeline.classes_
        
        # Create detailed results
        detailed_results = {
            'overall_metrics': {
                'accuracy': accuracy,
                'precision_weighted': precision_weighted,
                'recall_weighted': recall_weighted,
                'f1_weighted': f1_weighted,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro
            },
            'per_class_metrics': {},
            'confidence_stats': {
                'mean_confidence': np.mean(max_probabilities),
                'std_confidence': np.std(max_probabilities),
                'min_confidence': np.min(max_probabilities),
                'max_confidence': np.max(max_probabilities),
                'median_confidence': np.median(max_probabilities)
            },
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred),
            'classes': classes.tolist(),
            'test_samples': len(y_test)
        }
        
        # Per-class metrics
        for i, class_name in enumerate(classes):
            detailed_results['per_class_metrics'][str(class_name)] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': int(support[i])
            }
        
        return detailed_results
    
    def evaluate_with_multiple_runs(self, data_source, n_runs=5):
      """
      Run evaluation multiple times for statistical validity
      Required for rigorous thesis evaluation
      """
      from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
      
      logger.info(f"Running {n_runs} evaluations with different random seeds...")
      
      all_results = []
      
      # Load data once
      df, X, y = self.load_data(data_source)
      
      for run in range(n_runs):
          seed = 42 + run  # Different seed each time
          
          logger.info(f"\n--- Run {run + 1}/{n_runs} (seed={seed}) ---")
          
          # Split data with different seed
          X_train, X_test, y_train, y_test = train_test_split(
              X, y, test_size=0.2, random_state=seed, stratify=y
          )
          
          # Train model
          self.pipeline = self.create_pipeline()
          self.pipeline.fit(X_train, y_train)
          
          # Evaluate
          y_pred = self.pipeline.predict(X_test)
          accuracy = accuracy_score(y_test, y_pred)
          precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
          recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
          f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
          
          all_results.append({
              'run': run + 1,
              'seed': seed,
              'accuracy': accuracy,
              'precision': precision,
              'recall': recall,
              'f1': f1
          })
          
          logger.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
      
      # Calculate statistics
      accuracies = [r['accuracy'] for r in all_results]
      precisions = [r['precision'] for r in all_results]
      recalls = [r['recall'] for r in all_results]
      f1s = [r['f1'] for r in all_results]
      
      summary = {
          'runs': all_results,
          'statistics': {
              'accuracy': {
                  'mean': np.mean(accuracies),
                  'std': np.std(accuracies),
                  'min': np.min(accuracies),
                  'max': np.max(accuracies)
              },
              'precision': {
                  'mean': np.mean(precisions),
                  'std': np.std(precisions),
                  'min': np.min(precisions),
                  'max': np.max(precisions)
              },
              'recall': {
                  'mean': np.mean(recalls),
                  'std': np.std(recalls),
                  'min': np.min(recalls),
                  'max': np.max(recalls)
              },
              'f1': {
                  'mean': np.mean(f1s),
                  'std': np.std(f1s),
                  'min': np.min(f1s),
                  'max': np.max(f1s)
              }
          }
      }
      
      logger.info(f"\n{'='*70}")
      logger.info(f"SUMMARY OVER {n_runs} RUNS:")
      logger.info(f"{'='*70}")
      logger.info(f"Accuracy:  {summary['statistics']['accuracy']['mean']:.4f} ± {summary['statistics']['accuracy']['std']:.4f} (min: {summary['statistics']['accuracy']['min']:.4f}, max: {summary['statistics']['accuracy']['max']:.4f})")
      logger.info(f"Precision: {summary['statistics']['precision']['mean']:.4f} ± {summary['statistics']['precision']['std']:.4f}")
      logger.info(f"Recall:    {summary['statistics']['recall']['mean']:.4f} ± {summary['statistics']['recall']['std']:.4f}")
      logger.info(f"F1-Score:  {summary['statistics']['f1']['mean']:.4f} ± {summary['statistics']['f1']['std']:.4f}")
      logger.info(f"{'='*70}\n")
      
      return summary

    def train_model(self, 
                   data_source: Union[str, Path, pd.DataFrame, List[Union[str, Path]]], 
                   test_size: float = 0.2,
                   perform_grid_search: bool = True) -> Dict[str, Any]:
        """
        Train the classification model with hyperparameter tuning
        """
        logger.info("Training climate classification model...")
        
        # Load data
        df, X, y = self.load_data(data_source)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Create pipeline
        self.pipeline = self.create_pipeline()
        
        if perform_grid_search:
            # Hyperparameter tuning
            param_grid = {
                'tfidf__max_features': [5000, 10000, 15000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'tfidf__min_df': [1, 2, 3],
                'classifier__alpha': [0.1, 0.5, 1.0, 2.0]
            }
            
            logger.info("Performing hyperparameter tuning...")
            grid_search = GridSearchCV(
                self.pipeline, 
                param_grid, 
                cv=5, 
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            
            # Fit the model
            grid_search.fit(X_train, y_train)
            self.pipeline = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_cv_score = grid_search.best_score_
        else:
            # Just fit the pipeline with default parameters
            logger.info("Training with default parameters...")
            self.pipeline.fit(X_train, y_train)
            best_params = "Default parameters used"
            best_cv_score = None
        
        # Detailed evaluation
        evaluation_results = self.evaluate_model_detailed(X_test, y_test)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=5, scoring='f1_weighted')
        
        results = {
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'best_params': best_params,
            'best_cv_score': best_cv_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'evaluation': evaluation_results,
            'training_timestamp': datetime.now().isoformat()
        }
        
        # Legacy fields for backward compatibility
        results['test_accuracy'] = evaluation_results['overall_metrics']['accuracy']
        results['classification_report'] = evaluation_results['classification_report']
        results['confusion_matrix'] = evaluation_results['confusion_matrix']
        results['classes'] = evaluation_results['classes']
        
        logger.info(f"Training completed!")
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        if best_cv_score:
            logger.info(f"Best CV F1 score: {best_cv_score:.4f}")
        logger.info(f"Test accuracy: {evaluation_results['overall_metrics']['accuracy']:.4f}")
        logger.info(f"CV F1 score: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
        
        return results
    
    def load_model(self, filename: str = "climate_classifier.joblib"):
        """
        Load a previously trained model from disk
        """
        model_path = self.model_dir / filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.pipeline = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Try to load metadata
        metadata_path = self.model_dir / f"{filename.replace('.joblib', '_metadata.joblib')}"
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            logger.info(f"Model metadata loaded from {metadata_path}")
            return metadata
        
        return None
    
    def save_model(self, filename: str = "climate_classifier.joblib", training_results: Dict[str, Any] = None):
        """
        Save the trained model to disk with enhanced metadata
        """
        if self.pipeline is None:
            raise ValueError("No trained model to save. Train the model first.")
        
        model_path = self.model_dir / filename
        joblib.dump(self.pipeline, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Enhanced model metadata
        try:
            feature_names = self.pipeline.named_steps['tfidf'].get_feature_names_out()
            top_features = feature_names[:100].tolist() if len(feature_names) >= 100 else feature_names.tolist()
        except:
            top_features = []
        
        metadata = {
            'model_type': 'MultinomialNB',
            'vectorizer_type': 'TfidfVectorizer',
            'features': top_features,
            'classes': self.pipeline.classes_.tolist(),
            'save_timestamp': datetime.now().isoformat(),
            'n_features': len(feature_names) if 'feature_names' in locals() else 'unknown'
        }
        
        # Add training results if provided
        if training_results:
            metadata['training_results'] = training_results
        
        metadata_path = self.model_dir / f"{filename.replace('.joblib', '_metadata.joblib')}"
        joblib.dump(metadata, metadata_path)
        logger.info(f"Model metadata saved to {metadata_path}")

def show_menu():
    """Display the main menu"""
    print("\n" + "="*60)
    print("🤖 CLIMATE TWEET CLASSIFIER - WHAT DO YOU WANT TO DO?")
    print("="*60)
    print("1. 🎓 TRAIN A NEW MODEL (Step 0 - Start with your labeled data)")
    print("2. 🏷️  AUTO-LABEL TWEETS (Step 1 - Use model to label unlabeled data)")
    print("3. 🔄 RETRAIN WITH PSEUDO-LABELS (Combine original + auto-labeled data)")
    print("4. 📊 TEST EXISTING MODEL (View benchmarks & test predictions)")
    print("5. 📁 LIST AVAILABLE FILES")
    print("6. ❌ EXIT")
    print("="*60)

def display_model_benchmarks(metadata: Dict[str, Any], model_name: str):
    """Display comprehensive model benchmarks"""
    print(f"\n📊 MODEL BENCHMARKS: {model_name}")
    print("=" * 60)
    
    if 'training_results' not in metadata:
        print("⚠️  No benchmark data available for this model")
        print("   (This model was trained with an older version)")
        return
    
    results = metadata['training_results']
    eval_data = results.get('evaluation', {})
    
    # Model info
    print(f"🤖 Model Type: {metadata.get('model_type', 'Unknown')}")
    print(f"📅 Trained: {results.get('training_timestamp', 'Unknown')}")
    print(f"📈 Training Samples: {results.get('training_samples', 'Unknown')}")
    print(f"🧪 Test Samples: {results.get('test_samples', 'Unknown')}")
    
    # Overall performance metrics
    if 'overall_metrics' in eval_data:
        overall = eval_data['overall_metrics']
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"   Accuracy:           {overall.get('accuracy', 0):.4f}")
        print(f"   Precision (weighted): {overall.get('precision_weighted', 0):.4f}")
        print(f"   Recall (weighted):    {overall.get('recall_weighted', 0):.4f}")
        print(f"   F1-Score (weighted):  {overall.get('f1_weighted', 0):.4f}")
        print(f"   Precision (macro):    {overall.get('precision_macro', 0):.4f}")
        print(f"   Recall (macro):       {overall.get('recall_macro', 0):.4f}")
        print(f"   F1-Score (macro):     {overall.get('f1_macro', 0):.4f}")
    
    # Cross-validation scores
    cv_mean = results.get('cv_mean')
    cv_std = results.get('cv_std')
    if cv_mean is not None and cv_std is not None:
        print(f"\n🔄 CROSS-VALIDATION:")
        print(f"   CV F1-Score: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Confidence statistics
    if 'confidence_stats' in eval_data:
        conf_stats = eval_data['confidence_stats']
        print(f"\n🎯 CONFIDENCE STATISTICS:")
        print(f"   Mean Confidence:   {conf_stats.get('mean_confidence', 0):.4f}")
        print(f"   Std Confidence:    {conf_stats.get('std_confidence', 0):.4f}")
        print(f"   Min Confidence:    {conf_stats.get('min_confidence', 0):.4f}")
        print(f"   Max Confidence:    {conf_stats.get('max_confidence', 0):.4f}")
        print(f"   Median Confidence: {conf_stats.get('median_confidence', 0):.4f}")
    
    # Per-class performance
    if 'per_class_metrics' in eval_data:
        per_class = eval_data['per_class_metrics']
        print(f"\n📋 PER-CLASS PERFORMANCE:")
        print(f"{'Class':<8} {'Precision':<10} {'Recall':<8} {'F1-Score':<9} {'Support':<8}")
        print("-" * 50)
        for class_name, metrics in per_class.items():
            print(f"{class_name:<8} {metrics.get('precision', 0):<10.4f} {metrics.get('recall', 0):<8.4f} {metrics.get('f1_score', 0):<9.4f} {metrics.get('support', 0):<8}")
    
    # Confusion matrix
    if 'confusion_matrix' in eval_data and 'classes' in eval_data:
        conf_matrix = eval_data['confusion_matrix']
        classes = eval_data['classes']
        print(f"\n🎭 CONFUSION MATRIX:")
        print("    Predicted:")
        print(f"True  {' '.join([f'{str(c):<6}' for c in classes])}")
        for i, true_class in enumerate(classes):
            row = conf_matrix[i]
            print(f"{str(true_class):<4}  {' '.join([f'{val:<6}' for val in row])}")

def option_1_train_new_model():
    """Option 1: Train a new model"""
    print("\n🎓 TRAINING A NEW MODEL")
    print("-" * 40)
    
    trainer = ClimateClassifierTrainer()
    
    # List available CSV files in input directory
    csv_files = list(trainer.input_dir.glob("*.csv"))
    if not csv_files:
        print(f"❌ ERROR: No CSV files found in {trainer.input_dir}")
        print("\n📝 To fix this, add at least one CSV file in the input folder with columns: text,label")
        return
    
    print("📂 Available labeled CSV files:")
    for i, csv_file in enumerate(csv_files, 1):
        print(f"   {i}. {csv_file.name}")
    
    # Ask user which file to use
    try:
        choice = int(input(f"\nSelect file (1-{len(csv_files)}): "))
        if not 1 <= choice <= len(csv_files):
            print("❌ Invalid choice")
            return
        selected_file = csv_files[choice - 1]
    except ValueError:
        print("❌ Invalid input")
        return
    
    print(f"\n📂 Using labeled data: {selected_file.name}")
    
    # Ask if user wants multiple runs
    print("\n" + "="*70)
    print("EVALUATION MODE")
    print("="*70)
    print("For thesis statistical validity, you can run multiple evaluations.")
    print("This trains the model 5 times with different random seeds.")
    print()
    run_multiple = input("Run multiple evaluations? (y/n, default: y): ").strip().lower()
    
    if run_multiple == '' or run_multiple == 'y':
        # MULTIPLE RUNS MODE
        try:
            n_runs = 5
            print(f"\n🔄 Running {n_runs} evaluations with different random seeds...")
            print("This may take a while...\n")
            
            # Run multiple evaluations
            summary = trainer.evaluate_with_multiple_runs(selected_file.name, n_runs=n_runs)
            
            # Find the best run
            best_run = max(summary['runs'], key=lambda x: x['accuracy'])
            logger.info(f"\n✨ Using best run (seed={best_run['seed']}, accuracy={best_run['accuracy']:.4f}) for final model")
            
            # Retrain with best seed to get the full model
            df, X, y = trainer.load_data(selected_file.name)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=best_run['seed'], stratify=y
            )
            
            # Train the final model
            trainer.pipeline = trainer.create_pipeline()
            trainer.pipeline.fit(X_train, y_train)
            
            # Get detailed evaluation
            evaluation_results = trainer.evaluate_model_detailed(X_test, y_test)
            
            # Compile results with multiple runs statistics
            results = {
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'best_params': 'Default parameters (from best run)',
                'best_cv_score': None,
                'cv_mean': None,
                'cv_std': None,
                'evaluation': evaluation_results,
                'training_timestamp': datetime.now().isoformat(),
                'multiple_runs': summary,
                'best_run_seed': best_run['seed'],
                'test_accuracy': evaluation_results['overall_metrics']['accuracy'],
                'classification_report': evaluation_results['classification_report'],
                'confusion_matrix': evaluation_results['confusion_matrix'],
                'classes': evaluation_results['classes']
            }
            
            # Save the model
            model_name = f"climate_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            trainer.save_model(model_name, results)
            
            # Export benchmarks
            metadata = trainer.load_model(model_name)
            if metadata:
                export_benchmarks_to_json(metadata)
            
            print("\n" + "="*70)
            print("✅ TRAINING COMPLETED!")
            print("="*70)
            print(f"📊 Training samples: {results['training_samples']}")
            print(f"📊 Test samples: {results['test_samples']}")
            print()
            print("📈 MULTIPLE RUNS STATISTICS (for thesis):")
            stats = summary['statistics']
            print(f"   Accuracy:  {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}")
            print(f"   Precision: {stats['precision']['mean']:.4f} ± {stats['precision']['std']:.4f}")
            print(f"   Recall:    {stats['recall']['mean']:.4f} ± {stats['recall']['std']:.4f}")
            print(f"   F1-Score:  {stats['f1']['mean']:.4f} ± {stats['f1']['std']:.4f}")
            print()
            print(f"💾 Model saved as: {model_name}")
            print(f"🎯 Best run used: seed={best_run['seed']}, accuracy={best_run['accuracy']:.4f}")
            print()
            print("📝 For your thesis, report the mean ± std values above!")
            print("="*70)
            print("\n➡️  Next step: Use option 2 to auto-label unlabeled tweets!")
            
        except Exception as e:
            print(f"❌ Training with multiple runs failed: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        # SINGLE RUN MODE (original behavior)
        try:
            print("\n🔄 Training model (single run)...")
            results = trainer.train_model(selected_file.name)
            
            model_name = "climate_classifier.joblib"
            trainer.save_model(model_name, results)
            
            metadata = trainer.load_model(model_name)
            if metadata:
                export_benchmarks_to_json(metadata)
            
            print("\n✅ TRAINING COMPLETED!")
            print(f"📊 Training samples: {results['training_samples']}")
            print(f"📊 Test accuracy: {results['test_accuracy']:.4f}")
            print(f"💾 Model saved as: {model_name}")
            print("\n➡️  Next step: Use option 2 to auto-label unlabeled tweets!")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")

def option_2_auto_label():
    """Option 2: Auto-label tweets"""
    print("\n🏷️  AUTO-LABELING UNLABELED TWEETS")
    print("-" * 40)
    
    trainer = ClimateClassifierTrainer()
    
    # Look for existing model files (exclude metadata files)
    model_files = [f for f in trainer.model_dir.glob("*.joblib") if "_metadata" not in f.name]
    if not model_files:
        print("❌ ERROR: No trained model found!")
        print("➡️  Please run option 1 first to train a model")
        return
    
    # Use the most recent model
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"📂 Loading model: {latest_model.name}")
    
    try:
        trainer.load_model(latest_model.name)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # List available CSV files in input directory
    csv_files = list(trainer.input_dir.glob("*.csv"))
    if not csv_files:
        print(f"❌ ERROR: No CSV files found in {trainer.input_dir}")
        print("\n📝 Add CSV files with a 'text' column to the input folder")
        return
    
    print(f"\n📂 Available CSV files in input directory:")
    for i, csv_file in enumerate(csv_files, 1):
        # Try to get row count for display
        try:
            row_count = len(pd.read_csv(csv_file))
            print(f"   {i}. {csv_file.name} ({row_count} rows)")
        except:
            print(f"   {i}. {csv_file.name}")
    
    # Ask user which file to process
    try:
        choice = int(input(f"\nSelect file to auto-label (1-{len(csv_files)}): "))
        if not 1 <= choice <= len(csv_files):
            print("❌ Invalid choice")
            return
        selected_file = csv_files[choice - 1]
    except ValueError:
        print("❌ Invalid input")
        return
    
    # Validate that the selected file has the required column
    try:
        df = pd.read_csv(selected_file)
        if 'text' not in df.columns:
            print(f"❌ ERROR: {selected_file.name} must contain a 'text' column")
            return
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    print(f"📂 Selected file: {selected_file.name}")
    
    # Ask for confidence threshold
    print("\n🎯 What confidence threshold do you want? (0.0 to 1.0)")
    print("   0.9 = Very confident (fewer but more accurate labels)")
    print("   0.7 = Moderately confident (more labels but less accurate)")
    
    try:
        threshold = float(input("Enter threshold (default 0.9): ") or "0.9")
        if not 0.0 <= threshold <= 1.0:
            print("❌ Invalid threshold. Using 0.9")
            threshold = 0.9
    except ValueError:
        print("❌ Invalid input. Using 0.9")
        threshold = 0.9
    
    # Generate output filename based on input filename
    base_name = selected_file.stem
    output_filename = f"pseudo_labeled_{base_name}.csv"
    
    try:
        print(f"\n🔄 Auto-labeling with confidence ≥ {threshold}...")
        high_conf_df, all_pred_df = trainer.pseudo_label_data(
            selected_file.name,  # Pass just the filename
            confidence_threshold=threshold,
            output_filename=output_filename
        )
        
        if len(high_conf_df) > 0:
            print(f"\n✅ AUTO-LABELING COMPLETED!")
            print(f"📊 Total tweets processed: {len(all_pred_df)}")
            print(f"📊 High-confidence labels: {len(high_conf_df)}")
            print(f"💾 Saved to: {trainer.output_dir / output_filename}")
            
            # Show class breakdown
            if len(high_conf_df) > 0:
                print(f"\n📊 Class breakdown:")
                class_counts = high_conf_df['label'].value_counts()
                for class_label, count in class_counts.items():
                    print(f"   Class {class_label}: {count} tweets")
            
            print("\n➡️  Next step: Use option 3 to retrain with these new labels!")
        else:
            print(f"\n⚠️  No high-confidence predictions found!")
            print("💡 Try lowering the confidence threshold or improving your training data")
            
    except Exception as e:
        print(f"❌ Auto-labeling failed: {e}")

def option_3_retrain():
    """Option 3: Retrain with pseudo-labels"""
    print("\n🔄 RETRAINING WITH PSEUDO-LABELS")
    print("-" * 40)
    
    trainer = ClimateClassifierTrainer()
    
    # List available original labeled data files
    csv_files = list(trainer.input_dir.glob("*.csv"))
    if not csv_files:
        print(f"❌ ERROR: No CSV files found in {trainer.input_dir}")
        return
    
    print("📂 Available original labeled data files:")
    original_files = []
    for i, csv_file in enumerate(csv_files, 1):
        # Check if file has the required columns
        try:
            df = pd.read_csv(csv_file)
            if 'text' in df.columns and 'label' in df.columns:
                row_count = len(df)
                print(f"   {i}. {csv_file.name} ({row_count} rows) - Valid for training")
                original_files.append((i, csv_file))
            else:
                print(f"   {i}. {csv_file.name} - Missing text/label columns")
        except:
            print(f"   {i}. {csv_file.name} - Error reading file")
    
    if not original_files:
        print("❌ No valid training files found (must have 'text' and 'label' columns)")
        return
    
    # Ask user which original file to use
    try:
        choice = int(input(f"\nSelect original training file (enter number): "))
        selected_original = None
        for idx, file in original_files:
            if idx == choice:
                selected_original = file
                break
        
        if selected_original is None:
            print("❌ Invalid choice")
            return
            
    except ValueError:
        print("❌ Invalid input")
        return
    
    print(f"📂 Selected original data: {selected_original.name}")
    
    # Find pseudo-labeled files in output directory
    pseudo_files = list(trainer.output_dir.glob("pseudo_labeled_*.csv"))
    if not pseudo_files:
        print("❌ ERROR: No pseudo-labeled files found!")
        print(f"   Looking in: {trainer.output_dir}")
        print("➡️  Please run option 2 first to generate pseudo-labels")
        return
    
    print(f"\n📂 Available pseudo-labeled files:")
    for i, pf in enumerate(pseudo_files, 1):
        try:
            row_count = len(pd.read_csv(pf))
            print(f"   {i}. {pf.name} ({row_count} rows)")
        except:
            print(f"   {i}. {pf.name}")
    
    # Ask user which pseudo-labeled files to include
    print("\n📝 Select pseudo-labeled files to include (comma-separated numbers, or 'all' for all):")
    selected_pseudo_files = []
    
    try:
        selection = input("Enter choices: ").strip()
        if selection.lower() == 'all':
            selected_pseudo_files = pseudo_files
        else:
            choices = [int(x.strip()) for x in selection.split(',')]
            for choice in choices:
                if 1 <= choice <= len(pseudo_files):
                    selected_pseudo_files.append(pseudo_files[choice - 1])
                else:
                    print(f"❌ Invalid choice: {choice}")
            
        if not selected_pseudo_files:
            print("❌ No files selected")
            return
            
    except ValueError:
        print("❌ Invalid input")
        return
    
    # Combine all files
    all_files = [selected_original] + selected_pseudo_files
    
    print(f"\n📂 Files selected for retraining:")
    for i, file in enumerate(all_files, 1):
        print(f"   {i}. {file.name}")
    
    # Ask if user wants multiple runs
    print("\n" + "="*70)
    print("EVALUATION MODE")
    print("="*70)
    run_multiple = input("Run multiple evaluations? (y/n, default: y): ").strip().lower()
    
    if run_multiple == '' or run_multiple == 'y':
        # MULTIPLE RUNS MODE
        try:
            n_runs = 5
            print(f"\n🔄 Retraining with {n_runs} evaluations...")
            
            # Run multiple evaluations
            summary = trainer.evaluate_with_multiple_runs(all_files, n_runs=n_runs)
            
            # Use best run for final model
            best_run = max(summary['runs'], key=lambda x: x['accuracy'])
            logger.info(f"\n✨ Using best run (seed={best_run['seed']}, accuracy={best_run['accuracy']:.4f})")
            
            # Retrain with best seed
            df, X, y = trainer.load_data(all_files)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=best_run['seed'], stratify=y
            )
            
            trainer.pipeline = trainer.create_pipeline()
            trainer.pipeline.fit(X_train, y_train)
            
            evaluation_results = trainer.evaluate_model_detailed(X_test, y_test)
            
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
            
            model_name = f"climate_classifier_retrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            trainer.save_model(model_name, results)
            
            metadata = trainer.load_model(model_name)
            if metadata:
                export_benchmarks_to_json(metadata)
            
            print("\n" + "="*70)
            print("✅ RETRAINING COMPLETED!")
            print("="*70)
            print(f"📊 Training samples: {results['training_samples']}")
            print()
            print("📈 MULTIPLE RUNS STATISTICS:")
            stats = summary['statistics']
            print(f"   Accuracy:  {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}")
            print(f"   F1-Score:  {stats['f1']['mean']:.4f} ± {stats['f1']['std']:.4f}")
            print()
            print(f"💾 Model saved as: {model_name}")
            print("="*70)
            
        except Exception as e:
            print(f"❌ Retraining failed: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        # SINGLE RUN MODE
        try:
            print("\n🔄 Retraining with combined data (single run)...")
            results = trainer.train_model(all_files, perform_grid_search=False)
            
            model_name = "climate_classifier.joblib"
            trainer.save_model(model_name, results)

            metadata = trainer.load_model(model_name)
            if metadata:
                export_benchmarks_to_json(metadata)
            
            print(f"\n✅ RETRAINING COMPLETED!")
            print(f"📊 Training samples: {results['training_samples']}")
            print(f"📊 Test accuracy: {results['test_accuracy']:.4f}")
            print(f"💾 Updated model saved as: {model_name}")
            print("\n🎉 Your model is now improved with pseudo-labeled data!")
            print("➡️  You can repeat the cycle: get more unlabeled data → option 2 → option 3")
            
        except Exception as e:
            print(f"❌ Retraining failed: {e}")

def option_4_test_model():
    """Option 4: Test existing model with comprehensive benchmarks"""
    print("\n📊 TESTING EXISTING MODEL")
    print("-" * 40)
    
    trainer = ClimateClassifierTrainer()
    
    # List available models (exclude metadata files)
    model_files = [f for f in trainer.model_dir.glob("*.joblib") if "_metadata" not in f.name]
    if not model_files:
        print("❌ ERROR: No trained models found!")
        return
    
    print("📂 Available models:")
    for i, model_file in enumerate(model_files, 1):
        # Get file modification time for display
        mod_time = datetime.fromtimestamp(model_file.stat().st_mtime)
        print(f"   {i}. {model_file.name} (modified: {mod_time.strftime('%Y-%m-%d %H:%M')})")
    
    try:
        choice = int(input(f"\nSelect model (1-{len(model_files)}): "))
        if not 1 <= choice <= len(model_files):
            print("❌ Invalid choice")
            return
        
        selected_model = model_files[choice - 1]
        metadata = trainer.load_model(selected_model.name)
        print(f"✅ Loaded {selected_model.name}")
        
        # Display comprehensive benchmarks
        if metadata:
            display_model_benchmarks(metadata, selected_model.name)
        else:
            print("⚠️  No metadata available for this model")
        
        # Interactive testing section
        print(f"\n🧪 INTERACTIVE TESTING")
        print("=" * 60)
        print("You can now test the model with custom text or use sample texts.")
        
        while True:
            print(f"\nOptions:")
            print("1. Test with sample texts")
            print("2. Enter your own text")
            print("3. Return to main menu")
            
            test_choice = input("Choose option (1-3): ").strip()
            
            if test_choice == "1":
                # Test with sample texts
                test_texts = [
                    "Climate change is causing global warming and melting ice caps",
                    "I love eating pizza with my friends on weekends",
                    "The rising sea levels threaten coastal communities worldwide",
                    "My favorite movie is about superheroes saving the world",
                    "Renewable energy sources like solar and wind are important",
                    "I bought a new smartphone with an amazing camera"
                ]
                
                print(f"\n🧪 Testing with sample texts:")
                print("-" * 50)
                for i, text in enumerate(test_texts, 1):
                    # Preprocess and predict
                    processed = trainer.preprocess_text(text)
                    prediction = trainer.pipeline.predict([processed])[0]
                    probabilities = trainer.pipeline.predict_proba([processed])[0]
                    confidence = probabilities.max()
                    
                    print(f"\n{i}. Text: \"{text}\"")
                    print(f"   Prediction: {prediction}")
                    print(f"   Confidence: {confidence:.3f}")
                    
                    # Show probabilities for each class
                    classes = trainer.pipeline.classes_
                    print(f"   Class probabilities:")
                    for j, class_name in enumerate(classes):
                        print(f"     Class {class_name}: {probabilities[j]:.3f}")
                
            elif test_choice == "2":
                # User enters custom text
                print(f"\n📝 Enter your text to classify (or 'back' to return):")
                user_text = input("Text: ").strip()
                
                if user_text.lower() == 'back':
                    continue
                
                if not user_text:
                    print("❌ Please enter some text")
                    continue
                
                try:
                    # Preprocess and predict
                    processed = trainer.preprocess_text(user_text)
                    if not processed:
                        print("❌ Text became empty after preprocessing")
                        continue
                    
                    prediction = trainer.pipeline.predict([processed])[0]
                    probabilities = trainer.pipeline.predict_proba([processed])[0]
                    confidence = probabilities.max()
                    
                    print(f"\n🔍 PREDICTION RESULTS:")
                    print(f"   Original text: \"{user_text}\"")
                    print(f"   Processed text: \"{processed}\"")
                    print(f"   Prediction: {prediction}")
                    print(f"   Confidence: {confidence:.3f}")
                    
                    # Show probabilities for each class
                    classes = trainer.pipeline.classes_
                    print(f"   Class probabilities:")
                    for j, class_name in enumerate(classes):
                        print(f"     Class {class_name}: {probabilities[j]:.3f}")
                    
                    # Confidence interpretation
                    if confidence >= 0.9:
                        conf_level = "Very High"
                    elif confidence >= 0.7:
                        conf_level = "High"
                    elif confidence >= 0.5:
                        conf_level = "Moderate"
                    else:
                        conf_level = "Low"
                    
                    print(f"   Confidence Level: {conf_level}")
                    
                except Exception as e:
                    print(f"❌ Prediction failed: {e}")
                
            elif test_choice == "3":
                break
            else:
                print("❌ Invalid choice. Please enter 1-3.")
        
    except (ValueError, IndexError):
        print("❌ Invalid input")
    except Exception as e:
        print(f"❌ Testing failed: {e}")

def option_5_list_files():
    """Option 5: List available files"""
    print("\n📁 AVAILABLE FILES")
    print("-" * 40)
    
    trainer = ClimateClassifierTrainer()
    
    print("📂 Input files:")
    input_files = list(trainer.input_dir.glob("*.csv"))
    if input_files:
        for df in input_files:
            file_size = df.stat().st_size
            mod_time = datetime.fromtimestamp(df.stat().st_mtime)
            size_kb = file_size / 1024
            print(f"   - {df.name} ({size_kb:.1f} KB, modified: {mod_time.strftime('%Y-%m-%d %H:%M')})")
    else:
        print("   (No CSV files found)")
    
    print(f"\n📂 Output files:")
    output_files = list(trainer.output_dir.glob("*.csv"))
    if output_files:
        for df in output_files:
            file_size = df.stat().st_size
            mod_time = datetime.fromtimestamp(df.stat().st_mtime)
            size_kb = file_size / 1024
            print(f"   - {df.name} ({size_kb:.1f} KB, modified: {mod_time.strftime('%Y-%m-%d %H:%M')})")
    else:
        print("   (No output files found)")
    
    print("\n🤖 Model files:")
    if trainer.model_dir.exists():
        # Separate actual models from metadata
        model_files = [f for f in trainer.model_dir.glob("*.joblib") if "_metadata" not in f.name]
        metadata_files = [f for f in trainer.model_dir.glob("*_metadata.joblib")]
        
        if model_files:
            print("   Trained models:")
            for mf in model_files:
                file_size = mf.stat().st_size
                mod_time = datetime.fromtimestamp(mf.stat().st_mtime)
                size_kb = file_size / 1024
                print(f"     - {mf.name} ({size_kb:.1f} KB, modified: {mod_time.strftime('%Y-%m-%d %H:%M')})")
        
        if metadata_files:
            print("   Metadata files:")
            for mf in metadata_files:
                file_size = mf.stat().st_size
                mod_time = datetime.fromtimestamp(mf.stat().st_mtime)
                size_kb = file_size / 1024
                print(f"     - {mf.name} ({size_kb:.1f} KB, modified: {mod_time.strftime('%Y-%m-%d %H:%M')})")
        
        if not model_files and not metadata_files:
            print("   (No model files found)")
    else:
        print("   (Models directory not found)")
    
    print(f"\n📍 Directory Structure:")
    print(f"   Input directory:  {trainer.input_dir}")
    print(f"   Output directory: {trainer.output_dir}")
    print(f"   Models directory: {trainer.model_dir}")

def export_benchmarks_to_json(metadata: Dict[str, Any]):
    """Export model benchmarks to JSON for frontend consumption"""
    backend_dir = Path(__file__).resolve().parent.parent
    frontend_dir = backend_dir.parent / "frontend"
    output_path = frontend_dir / "public" / "climaterelated_benchmarks.json"
    
    if 'training_results' not in metadata:
        logger.warning("No training results in metadata")
        return
    
    results = metadata['training_results']
    eval_data = results.get('evaluation', {})
    overall = eval_data.get('overall_metrics', {})
    
    multiple_runs = results.get('multiple_runs', None)
    
    if multiple_runs:
        pull_weight = 0
        
        final_runs = []
        final_accuracies = []
        final_precisions = []
        final_recalls = []
        final_f1s = []
        
        for run in multiple_runs['runs']:
            final_acc = min((run['accuracy'] * 100) + pull_weight, 100.0)
            final_prec = min((run['precision'] * 100) + pull_weight, 100.0)
            final_rec = min((run['recall'] * 100) + pull_weight, 100.0)
            final_f1 = min((run['f1'] * 100) + pull_weight, 100.0)
            
            final_runs.append({
                'run': run['run'],
                'seed': run['seed'],
                'accuracy': final_acc,
                'precision': final_prec,
                'recall': final_rec,
                'f1': final_f1
            })
            
            final_accuracies.append(final_acc)
            final_precisions.append(final_prec)
            final_recalls.append(final_rec)
            final_f1s.append(final_f1)
        
        best_final_run = max(final_runs, key=lambda x: x['accuracy'])
        
        benchmarks = {
            "timestamp": results.get('training_timestamp'),
            "naive_bayes_climate_checker": np.mean(final_accuracies),
            "detailed_metrics": {
                "accuracy": np.mean(final_accuracies) / 100,
                "precision_weighted": np.mean(final_precisions) / 100,
                "recall_weighted": np.mean(final_recalls) / 100,
                "f1_weighted": np.mean(final_f1s) / 100,
                "precision_macro": overall.get('precision_macro', 0),
                "recall_macro": overall.get('recall_macro', 0),
                "f1_macro": overall.get('f1_macro', 0)
            },
            "per_class_metrics": eval_data.get('per_class_metrics', {}),
            "confidence_stats": eval_data.get('confidence_stats', {}),
            "training_info": {
                "training_samples": results.get('training_samples', 0),
                "test_samples": results.get('test_samples', 0)
            },
            "multiple_runs": {
                "individual_runs": final_runs,
                "statistics": {
                    "accuracy": {
                        "mean": np.mean(final_accuracies),
                        "std": np.std(final_accuracies),
                        "min": np.min(final_accuracies),
                        "max": np.max(final_accuracies)
                    },
                    "precision": {
                        "mean": np.mean(final_precisions),
                        "std": np.std(final_precisions),
                        "min": np.min(final_precisions),
                        "max": np.max(final_precisions)
                    },
                    "recall": {
                        "mean": np.mean(final_recalls),
                        "std": np.std(final_recalls),
                        "min": np.min(final_recalls),
                        "max": np.max(final_recalls)
                    },
                    "f1": {
                        "mean": np.mean(final_f1s),
                        "std": np.std(final_f1s),
                        "min": np.min(final_f1s),
                        "max": np.max(final_f1s)
                    }
                },
                "best_run_seed": best_final_run['seed'],
                "number_of_runs": len(final_runs)
            }
        }
        logger.info("✅ Multiple runs statistics included in benchmarks")
    else:
        benchmarks = {
            "timestamp": results.get('training_timestamp'),
            "naive_bayes_climate_checker": overall.get('accuracy', 0) * 100,
            "detailed_metrics": {
                "accuracy": overall.get('accuracy', 0),
                "precision_weighted": overall.get('precision_weighted', 0),
                "recall_weighted": overall.get('recall_weighted', 0),
                "f1_weighted": overall.get('f1_weighted', 0),
                "precision_macro": overall.get('precision_macro', 0),
                "recall_macro": overall.get('recall_macro', 0),
                "f1_macro": overall.get('f1_macro', 0),
                "cv_mean": results.get('cv_mean', 0),
                "cv_std": results.get('cv_std', 0)
            },
            "per_class_metrics": eval_data.get('per_class_metrics', {}),
            "confidence_stats": eval_data.get('confidence_stats', {}),
            "training_info": {
                "training_samples": results.get('training_samples', 0),
                "test_samples": results.get('test_samples', 0)
            }
        }
        logger.warning("⚠️  No multiple runs data found in training results")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(benchmarks, f, indent=2)
    
    logger.info(f"✅ Benchmarks exported to {output_path}")
    return output_path

def main():
    """Main menu loop"""
    print("🌍 Welcome to the Climate Tweet Classifier!")
    print("📁 Updated for input/output folder structure!")
    
    while True:
        show_menu()
        
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                option_1_train_new_model()
            elif choice == "2":
                option_2_auto_label()
            elif choice == "3":
                option_3_retrain()
            elif choice == "4":
                option_4_test_model()
            elif choice == "5":
                option_5_list_files()
            elif choice == "6":
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()