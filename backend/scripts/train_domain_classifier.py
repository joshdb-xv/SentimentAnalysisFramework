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

class ClimateCategoryClassifier:
    """
    Enhanced trainer class for climate category tweet classification with pseudo-labeling support
    Categories:
    - Sea Level Rise / Coastal Hazards
    - Extreme Heat / Heatwaves  
    - Cold Weather / Temperature Drops
    - Flooding and Extreme Precipitation
    - Storms, Typhoons, and Wind Events
    - Drought and Water Scarcity
    - Air Pollution and Emissions
    - Environmental Degradation and Land Use
    - Geological Events
    """
    
    # Define the climate categories
    CLIMATE_CATEGORIES = [
        "Sea Level Rise / Coastal Hazards",
        "Extreme Heat / Heatwaves", 
        "Cold Weather / Temperature Drops",
        "Flooding and Extreme Precipitation",
        "Storms, Typhoons, and Wind Events",
        "Drought and Water Scarcity",
        "Air Pollution and Emissions",
        "Environmental Degradation and Land Use",
        "Geological Events"
    ]
    
    def __init__(self, model_dir: str = None, data_dir: str = None):
        base_dir = Path(__file__).resolve().parent.parent
        
        # Set up model directory
        if model_dir is None:
            self.model_dir = base_dir / "models"
        else:
            self.model_dir = Path(model_dir).resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up data directories
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
        Enhanced for climate-related terminology
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
                if 'text' not in df.columns or 'category' not in df.columns:
                    logger.warning(f"Skipping {csv_path}: missing 'text' or 'category' columns")
                    continue
                
                # Validate category values
                valid_categories = set(self.CLIMATE_CATEGORIES)
                df_categories = set(df['category'].unique())
                invalid_categories = df_categories - valid_categories
                
                if invalid_categories:
                    logger.warning(f"Found invalid categories in {csv_path.name}: {invalid_categories}")
                    logger.info(f"Valid categories are: {self.CLIMATE_CATEGORIES}")
                    # Filter out invalid categories
                    df = df[df['category'].isin(valid_categories)]
                
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
        
        # Remove duplicates based on text content
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')
        final_count = len(combined_df)
        
        logger.info(f"Combined dataset: {final_count} samples ({initial_count - final_count} duplicates removed)")
        logger.info(f"Category distribution:\n{combined_df['category'].value_counts()}")
        
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
            # Single CSV file
            csv_path = Path(data_source).resolve()
            logger.info(f"Loading data from {csv_path}")
            
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                logger.error(f"Error loading data: {str(e)}")
                raise
        
        # Validate required columns
        if 'text' not in df.columns or 'category' not in df.columns:
            raise ValueError("Data must contain 'text' and 'category' columns")
        
        # Validate categories
        valid_categories = set(self.CLIMATE_CATEGORIES)
        df_categories = set(df['category'].dropna().unique())
        invalid_categories = df_categories - valid_categories
        
        if invalid_categories:
            logger.warning(f"Found invalid categories: {invalid_categories}")
            logger.info(f"Valid categories are: {self.CLIMATE_CATEGORIES}")
            # Filter out invalid categories
            df = df[df['category'].isin(valid_categories)]
        
        # Remove rows with missing values
        df = df.dropna(subset=['text', 'category'])
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Remove empty texts after preprocessing
        df = df[df['processed_text'].str.len() > 0]
        
        logger.info(f"Final dataset: {len(df)} samples")
        logger.info(f"Categories present: {sorted(df['category'].unique())}")
        
        if 'source_file' not in df.columns:
            df['source_file'] = 'unknown'
            df['batch_id'] = 0
        
        X = df['processed_text'].values
        y = df['category'].values
        
        return df, X, y
    
    def pseudo_label_data(self, 
                         unlabeled_data: Union[str, Path, pd.DataFrame], 
                         confidence_threshold: float = 0.8,
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
        else:
            data_path = Path(unlabeled_data).resolve()
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
        predicted_categories = self.pipeline.predict(X_unlabeled)
        predicted_probabilities = self.pipeline.predict_proba(X_unlabeled)
        
        # Get maximum probability for each prediction (confidence)
        max_probabilities = np.max(predicted_probabilities, axis=1)
        
        # Add predictions to dataframe
        df['predicted_category'] = predicted_categories
        df['confidence'] = max_probabilities
        
        # Add probability for each category
        categories = self.pipeline.classes_
        for i, category in enumerate(categories):
            df[f'prob_{category}'] = predicted_probabilities[:, i]
        
        # Filter high-confidence predictions
        high_confidence_mask = df['confidence'] >= confidence_threshold
        high_confidence_df = df[high_confidence_mask].copy()
        
        # Rename predicted_category to category for consistency
        high_confidence_df['category'] = high_confidence_df['predicted_category']
        high_confidence_df['pseudo_labeled'] = True
        high_confidence_df['confidence_threshold'] = confidence_threshold
        high_confidence_df['pseudo_label_timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Pseudo-labeling results:")
        logger.info(f"  Total samples: {len(df)}")
        logger.info(f"  High confidence (≥{confidence_threshold}): {len(high_confidence_df)} ({len(high_confidence_df)/len(df)*100:.1f}%)")
        logger.info(f"  Confidence distribution by category:")
        
        for category in categories:
            category_mask = high_confidence_df['category'] == category
            count = category_mask.sum()
            logger.info(f"    {category}: {count} samples")
        
        # Save high-confidence pseudo-labels if requested
        if output_filename is not None and len(high_confidence_df) > 0:
            output_path = self.output_dir / output_filename
            
            # Select relevant columns for saving
            save_columns = ['text', 'category', 'confidence', 'pseudo_labeled', 
                          'confidence_threshold', 'pseudo_label_timestamp']
            save_df = high_confidence_df[save_columns]
            
            save_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(save_df)} high-confidence pseudo-labels to {output_path}")
        
        return high_confidence_df, df
    
    def create_pipeline(self) -> Pipeline:
        """
        Create scikit-learn pipeline optimized for climate category classification
        """
        # Enhanced TF-IDF Vectorizer for climate terminology
        vectorizer = TfidfVectorizer(
            max_features=15000,          # Increased for more categories
            ngram_range=(1, 3),          # Include trigrams for climate terms
            min_df=2,                    # Ignore terms that appear in less than 2 documents
            max_df=0.95,                 # Ignore terms that appear in more than 95% of documents
            stop_words=None,             # Don't use stop words (important for climate terms)
            lowercase=True,              # Already handled in preprocessing
            strip_accents='unicode',     # Handle accented characters
            token_pattern=r'\b\w+\b',    # Simple word tokenization
            sublinear_tf=True            # Apply sublinear tf scaling
        )
        
        # Multinomial Naive Bayes classifier with smoothing
        classifier = MultinomialNB(alpha=0.1)  # Lower alpha for more categories
        
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
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
        
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
            'classification_report': classification_report(y_test, y_pred, zero_division=0),
            'classes': classes.tolist(),
            'test_samples': len(y_test)
        }
        
        # Per-class metrics
        for i, class_name in enumerate(classes):
            detailed_results['per_class_metrics'][str(class_name)] = {
                'precision': float(precision[i]) if i < len(precision) else 0.0,
                'recall': float(recall[i]) if i < len(recall) else 0.0,
                'f1_score': float(f1[i]) if i < len(f1) else 0.0,
                'support': int(support[i]) if i < len(support) else 0
            }
        
        return detailed_results
    
    def train_model(self, 
                   data_source: Union[str, Path, pd.DataFrame, List[Union[str, Path]]], 
                   test_size: float = 0.2,
                   perform_grid_search: bool = True) -> Dict[str, Any]:
        """
        Train the climate category classification model with hyperparameter tuning
        """
        logger.info("Training climate category classification model...")
        
        # Load data
        df, X, y = self.load_data(data_source)
        
        # Check if we have enough samples for each category
        category_counts = pd.Series(y).value_counts()
        min_samples = category_counts.min()
        
        if min_samples < 2:
            logger.warning(f"Some categories have very few samples. Minimum: {min_samples}")
            logger.warning("This may affect model performance and evaluation.")
        
        # Split data with stratification (if possible)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except ValueError as e:
            logger.warning(f"Stratification failed: {e}. Using random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        # Create pipeline
        self.pipeline = self.create_pipeline()
        
        if perform_grid_search and len(np.unique(y_train)) > 1:
            # Hyperparameter tuning for multi-class classification
            param_grid = {
                'tfidf__max_features': [10000, 15000, 20000],
                'tfidf__ngram_range': [(1, 2), (1, 3)],
                'tfidf__min_df': [2, 3],
                'classifier__alpha': [0.01, 0.1, 0.5, 1.0]
            }
            
            logger.info("Performing hyperparameter tuning...")
            grid_search = GridSearchCV(
                self.pipeline, 
                param_grid, 
                cv=min(5, len(np.unique(y_train))),  # Adjust CV folds based on classes
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
        
        # Cross-validation score (if enough samples)
        try:
            cv_folds = min(5, len(np.unique(y_train)))
            if cv_folds >= 2:
                cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=cv_folds, scoring='f1_weighted')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            else:
                cv_mean = None
                cv_std = None
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            cv_mean = None
            cv_std = None
        
        results = {
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'best_params': best_params,
            'best_cv_score': best_cv_score,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'evaluation': evaluation_results,
            'training_timestamp': datetime.now().isoformat(),
            'categories': self.CLIMATE_CATEGORIES,
            'category_distribution': category_counts.to_dict()
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
        if cv_mean is not None:
            logger.info(f"CV F1 score: {cv_mean:.4f} ± {cv_std:.4f}")
        
        return results
    
    def load_model(self, filename: str = "climate_category_classifier.joblib"):
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
    
    def save_model(self, filename: str = "climate_category_classifier.joblib", training_results: Dict[str, Any] = None):
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
            top_features = feature_names[:200].tolist() if len(feature_names) >= 200 else feature_names.tolist()
        except:
            top_features = []
        
        metadata = {
            'model_type': 'MultinomialNB_ClimateCategory',
            'vectorizer_type': 'TfidfVectorizer',
            'features': top_features,
            'classes': self.pipeline.classes_.tolist(),
            'climate_categories': self.CLIMATE_CATEGORIES,
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
    print("\n" + "="*70)
    print("Climate Category Tweet Classifier - Main Menu")
    print("="*70)
    print("1. Train a new model (using labeled data from input folder)")
    print("2. Auto-categorize tweets (process unlabeled data)")
    print("3. Retrain with pseudo-labels (combine original + categorized data)")
    print("4. Test existing model (view benchmarks & test predictions)")
    print("5. List available files")
    print("6. Show climate categories")
    print("7. Exit")
    print("="*70)

def show_climate_categories():
    """Display the available climate categories"""
    print("\nAvailable Climate Categories:")
    print("=" * 50)
    categories = ClimateCategoryClassifier.CLIMATE_CATEGORIES
    for i, category in enumerate(categories, 1):
        print(f"{i:2d}. {category}")
    print("\nYour training data should have a 'category' column with these exact values")
    print("Example CSV format:")
    print("   text,category")
    print('   "Sea levels are rising rapidly","Sea Level Rise / Coastal Hazards"')
    print('   "Record breaking heatwave hits the city","Extreme Heat / Heatwaves"')

def display_model_benchmarks(metadata: Dict[str, Any], model_name: str):
    """Display comprehensive model benchmarks for climate categories"""
    print(f"\nModel Benchmarks: {model_name}")
    print("=" * 70)
    
    if 'training_results' not in metadata:
        print("No benchmark data available for this model")
        print("(This model was trained with an older version)")
        return
    
    results = metadata['training_results']
    eval_data = results.get('evaluation', {})
    
    # Model info
    print(f"Model Type: {metadata.get('model_type', 'Unknown')}")
    print(f"Trained: {results.get('training_timestamp', 'Unknown')}")
    print(f"Training Samples: {results.get('training_samples', 'Unknown')}")
    print(f"Test Samples: {results.get('test_samples', 'Unknown')}")
    
    # Category distribution
    if 'category_distribution' in results:
        cat_dist = results['category_distribution']
        print(f"\nCategory Distribution:")
        for category, count in cat_dist.items():
            print(f"   {category}: {count} samples")
    
    # Overall performance metrics
    if 'overall_metrics' in eval_data:
        overall = eval_data['overall_metrics']
        print(f"\nOverall Performance:")
        print(f"   Accuracy:             {overall.get('accuracy', 0):.4f}")
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
        print(f"\nCross-validation:")
        print(f"   CV F1-Score: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Confidence statistics
    if 'confidence_stats' in eval_data:
        conf_stats = eval_data['confidence_stats']
        print(f"\nConfidence Statistics:")
        print(f"   Mean Confidence:   {conf_stats.get('mean_confidence', 0):.4f}")
        print(f"   Std Confidence:    {conf_stats.get('std_confidence', 0):.4f}")
        print(f"   Min Confidence:    {conf_stats.get('min_confidence', 0):.4f}")
        print(f"   Max Confidence:    {conf_stats.get('max_confidence', 0):.4f}")
        print(f"   Median Confidence: {conf_stats.get('median_confidence', 0):.4f}")
    
    # Per-class performance
    if 'per_class_metrics' in eval_data:
        per_class = eval_data['per_class_metrics']
        print(f"\nPer-category Performance:")
        print(f"{'Category':<35} {'Precision':<10} {'Recall':<8} {'F1-Score':<9} {'Support':<8}")
        print("-" * 80)
        for category, metrics in per_class.items():
            # Truncate long category names for display
            display_category = category[:34] + "..." if len(category) > 34 else category
            print(f"{display_category:<35} {metrics.get('precision', 0):<10.4f} {metrics.get('recall', 0):<8.4f} {metrics.get('f1_score', 0):<9.4f} {metrics.get('support', 0):<8}")

def option_1_train_new_model():
    """Option 1: Train a new model"""
    print("\nTraining a new climate category model")
    print("-" * 50)
    
    trainer = ClimateCategoryClassifier()
    
    # List available CSV files in input directory
    csv_files = list(trainer.input_dir.glob("*.csv"))
    if not csv_files:
        print(f"ERROR: No CSV files found in {trainer.input_dir}")
        print("\nTo fix this, add CSV files to the input folder with these columns:")
        print("   text,category")
        print('   "Sea levels are rising","Sea Level Rise / Coastal Hazards"')
        print('   "Extreme heat warning","Extreme Heat / Heatwaves"')
        return
    
    print("Available labeled CSV files:")
    for i, csv_file in enumerate(csv_files, 1):
        # Try to get row count for display
        try:
            row_count = len(pd.read_csv(csv_file))
            print(f"   {i}. {csv_file.name} ({row_count} rows)")
        except:
            print(f"   {i}. {csv_file.name}")
    
    # Ask user which file to use
    try:
        choice = int(input(f"\nSelect file (1-{len(csv_files)}): "))
        if not 1 <= choice <= len(csv_files):
            print("Invalid choice")
            return
        selected_file = csv_files[choice - 1]
    except ValueError:
        print("Invalid input")
        return
    
    print(f"\nUsing labeled data: {selected_file.name}")
    
    # Validate that the selected file has the required columns
    try:
        df = pd.read_csv(selected_file)
        if 'text' not in df.columns or 'category' not in df.columns:
            print(f"ERROR: {selected_file.name} must contain 'text' and 'category' columns")
            return
        
        # Check if categories are valid
        valid_categories = set(trainer.CLIMATE_CATEGORIES)
        file_categories = set(df['category'].dropna().unique())
        invalid_categories = file_categories - valid_categories
        
        if invalid_categories:
            print(f"WARNING: File contains invalid categories: {invalid_categories}")
            print("Valid categories are:")
            for cat in trainer.CLIMATE_CATEGORIES:
                print(f"   - {cat}")
            print("\nThese invalid categories will be filtered out during training")
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    try:
        print("\nTraining model...")
        results = trainer.train_model(selected_file)
        
        # Save the model with training results
        model_name = f"climate_category_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        trainer.save_model(model_name, results)
        
        metadata = trainer.load_model(model_name)
        if metadata:
            export_benchmarks_to_json(metadata)
        
        print("\nTRAINING COMPLETED!")
        print(f"Training samples: {results['training_samples']}")
        print(f"Test accuracy: {results['test_accuracy']:.4f}")
        print(f"Number of categories: {len(results['classes'])}")
        print(f"Model saved as: {model_name}")
        print("\nNext step: Use option 2 to auto-categorize unlabeled tweets!")
        
    except Exception as e:
        print(f"Training failed: {e}")

def option_2_auto_categorize():
    """Option 2: Auto-categorize tweets"""
    print("\nAuto-categorizing unlabeled tweets")
    print("-" * 50)
    
    trainer = ClimateCategoryClassifier()
    
    # Look for existing model files (exclude metadata files)
    model_files = [f for f in trainer.model_dir.glob("*.joblib") if "_metadata" not in f.name]
    if not model_files:
        print("ERROR: No trained model found!")
        print("Please run option 1 first to train a model")
        return
    
    # Use the most recent model
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading model: {latest_model.name}")
    
    try:
        metadata = trainer.load_model(latest_model.name)
        print("Model loaded successfully!")
        
        if metadata and 'climate_categories' in metadata:
            print(f"Model can categorize into {len(metadata['climate_categories'])} categories")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # List available CSV files in input directory
    csv_files = list(trainer.input_dir.glob("*.csv"))
    if not csv_files:
        print(f"ERROR: No CSV files found in {trainer.input_dir}")
        print("\nAdd CSV files with a 'text' column to the input folder")
        return
    
    print(f"\nAvailable CSV files in input directory:")
    for i, csv_file in enumerate(csv_files, 1):
        # Try to get row count for display
        try:
            row_count = len(pd.read_csv(csv_file))
            print(f"   {i}. {csv_file.name} ({row_count} rows)")
        except:
            print(f"   {i}. {csv_file.name}")
    
    # Ask user which file to process
    try:
        choice = int(input(f"\nSelect file to categorize (1-{len(csv_files)}): "))
        if not 1 <= choice <= len(csv_files):
            print("Invalid choice")
            return
        selected_file = csv_files[choice - 1]
    except ValueError:
        print("Invalid input")
        return
    
    # Validate that the selected file has the required column
    try:
        df = pd.read_csv(selected_file)
        if 'text' not in df.columns:
            print(f"ERROR: {selected_file.name} must contain a 'text' column")
            return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    print(f"Selected file: {selected_file.name}")
    
    # Ask for confidence threshold
    print("\nWhat confidence threshold do you want? (0.0 to 1.0)")
    print("   0.8 = High confidence (fewer but more accurate categories)")
    print("   0.6 = Moderate confidence (more categories but less accurate)")
    print("   Note: Multi-class classification typically requires higher thresholds")
    
    try:
        threshold = float(input("Enter threshold (default 0.8): ") or "0.8")
        if not 0.0 <= threshold <= 1.0:
            print("Invalid threshold. Using 0.8")
            threshold = 0.8
    except ValueError:
        print("Invalid input. Using 0.8")
        threshold = 0.8
    
    # Generate output filename based on input filename
    base_name = selected_file.stem
    output_filename = f"categorized_{base_name}.csv"
    
    try:
        print(f"\nAuto-categorizing with confidence >= {threshold}...")
        high_conf_df, all_pred_df = trainer.pseudo_label_data(
            selected_file,
            confidence_threshold=threshold,
            output_filename=output_filename
        )
        
        if len(high_conf_df) > 0:
            print(f"\nAUTO-CATEGORIZATION COMPLETED!")
            print(f"Total tweets processed: {len(all_pred_df)}")
            print(f"High-confidence categories: {len(high_conf_df)}")
            print(f"Saved to: {trainer.output_dir / output_filename}")
            
            # Show category breakdown
            if len(high_conf_df) > 0:
                print(f"\nCategory breakdown:")
                category_counts = high_conf_df['category'].value_counts()
                for category, count in category_counts.items():
                    print(f"   {category}: {count} tweets")
            
            print("\nNext step: Use option 3 to retrain with these new categories!")
        else:
            print(f"\nNo high-confidence predictions found!")
            print("Try lowering the confidence threshold or improving your training data")
            
    except Exception as e:
        print(f"Auto-categorization failed: {e}")

def option_3_retrain():
    """Option 3: Retrain with pseudo-labels"""
    print("\nRetraining with pseudo-labels")
    print("-" * 50)
    
    trainer = ClimateCategoryClassifier()
    
    # List available original labeled data files
    csv_files = list(trainer.input_dir.glob("*.csv"))
    if not csv_files:
        print(f"ERROR: No CSV files found in {trainer.input_dir}")
        return
    
    print("Available original labeled data files:")
    original_files = []
    for i, csv_file in enumerate(csv_files, 1):
        # Check if file has the required columns
        try:
            df = pd.read_csv(csv_file)
            if 'text' in df.columns and 'category' in df.columns:
                row_count = len(df)
                print(f"   {i}. {csv_file.name} ({row_count} rows) - Valid for training")
                original_files.append((i, csv_file))
            else:
                print(f"   {i}. {csv_file.name} - Missing text/category columns")
        except:
            print(f"   {i}. {csv_file.name} - Error reading file")
    
    if not original_files:
        print("No valid training files found (must have 'text' and 'category' columns)")
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
            print("Invalid choice")
            return
            
    except ValueError:
        print("Invalid input")
        return
    
    print(f"Selected original data: {selected_original.name}")
    
    # Find categorized files in output directory
    categorized_files = list(trainer.output_dir.glob("categorized_*.csv"))
    if not categorized_files:
        print("ERROR: No categorized files found!")
        print("Please run option 2 first to generate categorized data")
        return
    
    print(f"\nAvailable categorized files:")
    for i, cf in enumerate(categorized_files, 1):
        try:
            row_count = len(pd.read_csv(cf))
            print(f"   {i}. {cf.name} ({row_count} rows)")
        except:
            print(f"   {i}. {cf.name}")
    
    # Ask user which categorized files to include
    print("\nSelect categorized files to include (comma-separated numbers, or 'all' for all):")
    selected_categorized = []
    
    try:
        selection = input("Enter choices: ").strip()
        if selection.lower() == 'all':
            selected_categorized = categorized_files
        else:
            choices = [int(x.strip()) for x in selection.split(',')]
            for choice in choices:
                if 1 <= choice <= len(categorized_files):
                    selected_categorized.append(categorized_files[choice - 1])
                else:
                    print(f"Invalid choice: {choice}")
            
        if not selected_categorized:
            print("No files selected")
            return
            
    except ValueError:
        print("Invalid input")
        return
    
    # Combine all files
    all_files = [selected_original] + selected_categorized
    
    print(f"\nFiles selected for retraining:")
    for i, file in enumerate(all_files, 1):
        print(f"   {i}. {file.name}")
    
    try:
        print("\nRetraining with combined data...")
        results = trainer.train_model(all_files, perform_grid_search=False)  # Faster retraining
        
        # Save the updated model
        model_name = f"climate_category_classifier_retrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        trainer.save_model(model_name, results)

        metadata = trainer.load_model(model_name)
        if metadata:
            export_benchmarks_to_json(metadata)
        
        print(f"\nRETRAINING COMPLETED!")
        print(f"Training samples: {results['training_samples']}")
        print(f"Test accuracy: {results['test_accuracy']:.4f}")
        print(f"Categories: {len(results['classes'])}")
        print(f"Updated model saved as: {model_name}")
        print("\nYour model is now improved with pseudo-labeled data!")
        print("You can repeat the cycle: get more unlabeled data -> option 2 -> option 3")
        
    except Exception as e:
        print(f"Retraining failed: {e}")

def option_4_test_model():
    """Option 4: Test existing model with comprehensive benchmarks"""
    print("\nTesting existing model")
    print("-" * 50)
    
    trainer = ClimateCategoryClassifier()
    
    # List available models (exclude metadata files)
    model_files = [f for f in trainer.model_dir.glob("*.joblib") if "_metadata" not in f.name]
    if not model_files:
        print("ERROR: No trained models found!")
        return
    
    print("Available models:")
    for i, model_file in enumerate(model_files, 1):
        # Get file modification time for display
        mod_time = datetime.fromtimestamp(model_file.stat().st_mtime)
        print(f"   {i}. {model_file.name} (modified: {mod_time.strftime('%Y-%m-%d %H:%M')})")
    
    try:
        choice = int(input(f"\nSelect model (1-{len(model_files)}): "))
        if not 1 <= choice <= len(model_files):
            print("Invalid choice")
            return
        
        selected_model = model_files[choice - 1]
        metadata = trainer.load_model(selected_model.name)
        print(f"Loaded {selected_model.name}")
        
        # Display comprehensive benchmarks
        if metadata:
            display_model_benchmarks(metadata, selected_model.name)
        else:
            print("No metadata available for this model")
        
        # Interactive testing section
        print(f"\nInteractive Testing")
        print("=" * 70)
        print("You can now test the model with custom text or use sample texts.")
        
        while True:
            print(f"\nOptions:")
            print("1. Test with sample texts")
            print("2. Enter your own text")
            print("3. Return to main menu")
            
            test_choice = input("Choose option (1-3): ").strip()
            
            if test_choice == "1":
                # Test with sample texts for different climate categories
                test_texts = [
                    "Umaabot na naman yung tubig dito, high tide lang pero parang binaha na",
                    "Init sobra, parang natutunaw na ako sa MRT",
                    "Tanghaling tapat pero sobrang lamig?? ano nangyayari sa panahon",
                    "Grabe baha sa amin, knee-deep na agad kahit isang oras lang umulan",
                    "Yung hangin kanina parang gusto na ko tangayin, legit takot ako",
                    "Tatlong taon na walang maayos na ulan, patay lahat ng pananim dito",
                    "Usok buong araw, hirap huminga sa kalsada",
                    "Nawala na yung puno sa tabi ng barangay, ginawang parking lot na",
                    "Lindol kanina, sobrang lakas nag-alugan lahat ng gamit sa bahay"
                ]
                
                print(f"\nTesting with sample texts:")
                print("-" * 60)
                for i, text in enumerate(test_texts, 1):
                    # Preprocess and predict
                    processed = trainer.preprocess_text(text)
                    prediction = trainer.pipeline.predict([processed])[0]
                    probabilities = trainer.pipeline.predict_proba([processed])[0]
                    confidence = probabilities.max()
                    
                    print(f"\n{i}. Text: \"{text}\"")
                    print(f"   Predicted Category: {prediction}")
                    print(f"   Confidence: {confidence:.3f}")
                    
                    # Show top 3 category probabilities
                    categories = trainer.pipeline.classes_
                    top_indices = np.argsort(probabilities)[-3:][::-1]
                    print(f"   Top 3 categories:")
                    for j, idx in enumerate(top_indices):
                        category_name = categories[idx]
                        prob = probabilities[idx]
                        print(f"     {j+1}. {category_name}: {prob:.3f}")
                
            elif test_choice == "2":
                # User enters custom text
                print(f"\nEnter your text to categorize (or 'back' to return):")
                user_text = input("Text: ").strip()
                
                if user_text.lower() == 'back':
                    continue
                
                if not user_text:
                    print("Please enter some text")
                    continue
                
                try:
                    # Preprocess and predict
                    processed = trainer.preprocess_text(user_text)
                    if not processed:
                        print("Text became empty after preprocessing")
                        continue
                    
                    prediction = trainer.pipeline.predict([processed])[0]
                    probabilities = trainer.pipeline.predict_proba([processed])[0]
                    confidence = probabilities.max()
                    
                    print(f"\nCategorization Results:")
                    print(f"   Original text: \"{user_text}\"")
                    print(f"   Processed text: \"{processed}\"")
                    print(f"   Predicted Category: {prediction}")
                    print(f"   Confidence: {confidence:.3f}")
                    
                    # Show all category probabilities (sorted)
                    categories = trainer.pipeline.classes_
                    sorted_indices = np.argsort(probabilities)[::-1]
                    print(f"   All category probabilities:")
                    for i, idx in enumerate(sorted_indices):
                        category_name = categories[idx]
                        prob = probabilities[idx]
                        print(f"     {i+1:2d}. {category_name}: {prob:.3f}")
                    
                    # Confidence interpretation
                    if confidence >= 0.8:
                        conf_level = "Very High"
                    elif confidence >= 0.6:
                        conf_level = "High"
                    elif confidence >= 0.4:
                        conf_level = "Moderate"
                    else:
                        conf_level = "Low"
                    
                    print(f"   Confidence Level: {conf_level}")
                    
                except Exception as e:
                    print(f"Categorization failed: {e}")
                
            elif test_choice == "3":
                break
            else:
                print("Invalid choice. Please enter 1-3.")
        
    except (ValueError, IndexError):
        print("Invalid input")
    except Exception as e:
        print(f"Testing failed: {e}")

def option_5_list_files():
    """Option 5: List available files"""
    print("\nAvailable Files")
    print("-" * 50)
    
    trainer = ClimateCategoryClassifier()
    
    print("Input files (for training and processing):")
    input_files = list(trainer.input_dir.glob("*.csv"))
    if input_files:
        for df in input_files:
            file_size = df.stat().st_size
            mod_time = datetime.fromtimestamp(df.stat().st_mtime)
            size_kb = file_size / 1024
            
            # Try to get row count
            try:
                row_count = len(pd.read_csv(df))
                row_info = f", {row_count} rows"
            except:
                row_info = ""
            
            print(f"   - {df.name} ({size_kb:.1f} KB{row_info}, modified: {mod_time.strftime('%Y-%m-%d %H:%M')})")
    else:
        print("   (No CSV files found)")
    
    print("\nOutput files (categorized results):")
    output_files = list(trainer.output_dir.glob("*.csv"))
    if output_files:
        for df in output_files:
            file_size = df.stat().st_size
            mod_time = datetime.fromtimestamp(df.stat().st_mtime)
            size_kb = file_size / 1024
            
            # Try to get row count
            try:
                row_count = len(pd.read_csv(df))
                row_info = f", {row_count} rows"
            except:
                row_info = ""
            
            print(f"   - {df.name} ({size_kb:.1f} KB{row_info}, modified: {mod_time.strftime('%Y-%m-%d %H:%M')})")
    else:
        print("   (No output files found)")
    
    print("\nModel files:")
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

def export_benchmarks_to_json(metadata: Dict[str, Any]):
    """Export model benchmarks to JSON for frontend consumption"""
    # Path to frontend public folder
    backend_dir = Path(__file__).resolve().parent.parent
    frontend_dir = backend_dir.parent / "frontend"
    output_path = frontend_dir / "public" / "climatedomain_benchmarks.json"
    
    if 'training_results' not in metadata:
        logger.warning("No training results in metadata")
        return
    
    results = metadata['training_results']
    eval_data = results.get('evaluation', {})
    overall = eval_data.get('overall_metrics', {})
    
    benchmarks = {
        "timestamp": results.get('training_timestamp'),
        "naive_bayes_domain_identifier": overall.get('accuracy', 0) * 100,  # ONLY this model's accuracy
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
        },
        "categories": results.get('categories', []),
        "category_distribution": results.get('category_distribution', {})
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(benchmarks, f, indent=2)
    
    logger.info(f"✅ Benchmarks exported to {output_path}")
    return output_path

def main():
    """Main menu loop"""
    print("Welcome to the Climate Category Tweet Classifier!")
    print("This classifier can categorize climate-related tweets into 9 specific categories.")
    
    while True:
        show_menu()
        
        try:
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == "1":
                option_1_train_new_model()
            elif choice == "2":
                option_2_auto_categorize()
            elif choice == "3":
                option_3_retrain()
            elif choice == "4":
                option_4_test_model()
            elif choice == "5":
                option_5_list_files()
            elif choice == "6":
                show_climate_categories()
            elif choice == "7":
                print("\nGoodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-7.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()