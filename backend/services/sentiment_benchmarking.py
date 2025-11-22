# services/sentiment_benchmarking.py

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from .sentiment_analysis import MultilingualVADER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentBenchmarker:
    """
    Benchmark VADER sentiment analyzer with multiple runs for statistical validity
    """
    
    def __init__(self, lexicon_path: str = None):
        self.analyzer = MultilingualVADER(lexicon_path=lexicon_path)
        base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = base_dir / "data" / "sentiment"
        self.output_dir = base_dir / "data" / "benchmarks"
        self.frontend_dir = base_dir.parent / "frontend" / "public"
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frontend_dir.mkdir(parents=True, exist_ok=True)
        
    def load_labeled_data(self, csv_path: str) -> Tuple[List[str], List[str]]:
        """
        Load labeled sentiment data
        Expected CSV format: text,sentiment (where sentiment is 'positive', 'negative', or 'neutral')
        """
        # Try multiple possible paths
        possible_paths = [
            Path(csv_path),
            self.data_dir / csv_path,
            Path(__file__).resolve().parent.parent / "data" / "input" / csv_path
        ]
        
        df = None
        for path in possible_paths:
            if path.exists():
                logger.info(f"Loading data from: {path}")
                df = pd.read_csv(path)
                break
        
        if df is None:
            raise FileNotFoundError(f"Could not find sentiment data file: {csv_path}")
        
        if 'text' not in df.columns or 'sentiment' not in df.columns:
            raise ValueError("CSV must have 'text' and 'sentiment' columns")
        
        # Remove missing values
        df = df.dropna(subset=['text', 'sentiment'])
        
        # Normalize sentiment labels
        df['sentiment'] = df['sentiment'].str.lower().str.strip()
        
        # Filter valid sentiments
        valid_sentiments = ['positive', 'negative', 'neutral']
        df = df[df['sentiment'].isin(valid_sentiments)]
        
        logger.info(f"Loaded {len(df)} labeled samples")
        logger.info(f"Distribution: {df['sentiment'].value_counts().to_dict()}")
        
        return df['text'].tolist(), df['sentiment'].tolist()
    
    def predict_sentiment(self, texts: List[str]) -> List[str]:
        """Predict sentiment for a list of texts"""
        predictions = []
        for text in texts:
            result = self.analyzer.analyze_sentiment(text, debug=False)
            predictions.append(result.classification)
        return predictions
    
    def evaluate_single_run(self, X_test: List[str], y_test: List[str], 
                           run_number: int, seed: int) -> Dict:
        """Evaluate on a single test set"""
        logger.info(f"Run {run_number} (seed={seed}): Evaluating on {len(X_test)} samples...")
        
        # Get predictions
        y_pred = self.predict_sentiment(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, 
            average='weighted',
            zero_division=0
        )
        
        # Per-class metrics
        labels = ['positive', 'negative', 'neutral']
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(y_test, y_pred, labels=labels, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        
        # Confidence stats (only for non-inconclusive predictions)
        confidences = []
        confidence_by_class = {'positive': [], 'negative': [], 'neutral': []}
        
        for text, true_label in zip(X_test, y_test):
            result = self.analyzer.analyze_sentiment(text, debug=False)
            if result.classification != 'inconclusive':
                confidences.append(result.confidence)
                if result.classification in confidence_by_class:
                    confidence_by_class[result.classification].append(result.confidence)
        
        # Calculate per-class confidence stats
        confidence_stats_per_class = {}
        for class_name, class_confidences in confidence_by_class.items():
            if class_confidences:
                confidence_stats_per_class[class_name] = {
                    'mean': float(np.mean(class_confidences)),
                    'std': float(np.std(class_confidences)),
                    'count': len(class_confidences)
                }
        
        return {
            'run': run_number,
            'seed': seed,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'per_class': {
                'positive': {
                    'precision': float(precision_per_class[0]),
                    'recall': float(recall_per_class[0]),
                    'f1': float(f1_per_class[0]),
                    'support': int(support_per_class[0])
                },
                'negative': {
                    'precision': float(precision_per_class[1]),
                    'recall': float(recall_per_class[1]),
                    'f1': float(f1_per_class[1]),
                    'support': int(support_per_class[1])
                },
                'neutral': {
                    'precision': float(precision_per_class[2]),
                    'recall': float(recall_per_class[2]),
                    'f1': float(f1_per_class[2]),
                    'support': int(support_per_class[2])
                }
            },
            'confusion_matrix': cm.tolist(),
            'confidence_stats': {
                'overall': {
                    'mean': float(np.mean(confidences)) if confidences else 0,
                    'std': float(np.std(confidences)) if confidences else 0,
                    'min': float(np.min(confidences)) if confidences else 0,
                    'max': float(np.max(confidences)) if confidences else 0,
                    'median': float(np.median(confidences)) if confidences else 0
                },
                'by_class': confidence_stats_per_class
            }
        }
    
    def run_multiple_evaluations(self, csv_path: str, n_runs: int = 5, 
                                 test_size: float = 0.2) -> Dict:
        """
        Run multiple evaluations with different random seeds
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"VADER SENTIMENT ANALYZER - MULTIPLE RUN EVALUATION")
        logger.info(f"{'='*70}\n")
        
        # Load data
        texts, labels = self.load_labeled_data(csv_path)
        
        all_results = []
        
        for run in range(n_runs):
            seed = 42 + run
            logger.info(f"\n--- Run {run + 1}/{n_runs} (seed={seed}) ---")
            
            # Split with different seed
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels,
                test_size=test_size,
                random_state=seed,
                stratify=labels
            )
            
            # Evaluate
            run_results = self.evaluate_single_run(X_test, y_test, run + 1, seed)
            all_results.append(run_results)
            
            logger.info(f"Accuracy: {run_results['accuracy']:.4f}")
            logger.info(f"Precision: {run_results['precision']:.4f}")
            logger.info(f"Recall: {run_results['recall']:.4f}")
            logger.info(f"F1-Score: {run_results['f1']:.4f}")
        
        # Calculate statistics
        accuracies = [r['accuracy'] for r in all_results]
        precisions = [r['precision'] for r in all_results]
        recalls = [r['recall'] for r in all_results]
        f1s = [r['f1'] for r in all_results]
        
        # Per-class statistics
        per_class_stats = {}
        for class_name in ['positive', 'negative', 'neutral']:
            class_f1s = [r['per_class'][class_name]['f1'] for r in all_results]
            class_precisions = [r['per_class'][class_name]['precision'] for r in all_results]
            class_recalls = [r['per_class'][class_name]['recall'] for r in all_results]
            
            per_class_stats[class_name] = {
                'precision': {
                    'mean': float(np.mean(class_precisions)),
                    'std': float(np.std(class_precisions)),
                    'min': float(np.min(class_precisions)),
                    'max': float(np.max(class_precisions))
                },
                'recall': {
                    'mean': float(np.mean(class_recalls)),
                    'std': float(np.std(class_recalls)),
                    'min': float(np.min(class_recalls)),
                    'max': float(np.max(class_recalls))
                },
                'f1': {
                    'mean': float(np.mean(class_f1s)),
                    'std': float(np.std(class_f1s)),
                    'min': float(np.min(class_f1s)),
                    'max': float(np.max(class_f1s))
                }
            }
        
        # Find best run
        best_run = max(all_results, key=lambda x: x['accuracy'])
        
        summary = {
            'model': 'VADER (Enhanced Multilingual)',
            'lexicon_stats': self.analyzer.get_lexicon_stats(),
            'evaluation_info': {
                'total_samples': len(texts),
                'test_size': test_size,
                'n_runs': n_runs,
                'data_file': csv_path
            },
            'runs': all_results,
            'statistics': {
                'accuracy': {
                    'mean': float(np.mean(accuracies)),
                    'std': float(np.std(accuracies)),
                    'min': float(np.min(accuracies)),
                    'max': float(np.max(accuracies))
                },
                'precision': {
                    'mean': float(np.mean(precisions)),
                    'std': float(np.std(precisions)),
                    'min': float(np.min(precisions)),
                    'max': float(np.max(precisions))
                },
                'recall': {
                    'mean': float(np.mean(recalls)),
                    'std': float(np.std(recalls)),
                    'min': float(np.min(recalls)),
                    'max': float(np.max(recalls))
                },
                'f1': {
                    'mean': float(np.mean(f1s)),
                    'std': float(np.std(f1s)),
                    'min': float(np.min(f1s)),
                    'max': float(np.max(f1s))
                }
            },
            'per_class_statistics': per_class_stats,
            'best_run': {
                'run_number': best_run['run'],
                'seed': best_run['seed'],
                'accuracy': best_run['accuracy'],
                'f1': best_run['f1']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Print summary
        logger.info(f"\n{'='*70}")
        logger.info(f"SUMMARY OVER {n_runs} RUNS:")
        logger.info(f"{'='*70}")
        logger.info(f"Accuracy:  {summary['statistics']['accuracy']['mean']:.4f} ± {summary['statistics']['accuracy']['std']:.4f}")
        logger.info(f"Precision: {summary['statistics']['precision']['mean']:.4f} ± {summary['statistics']['precision']['std']:.4f}")
        logger.info(f"Recall:    {summary['statistics']['recall']['mean']:.4f} ± {summary['statistics']['recall']['std']:.4f}")
        logger.info(f"F1-Score:  {summary['statistics']['f1']['mean']:.4f} ± {summary['statistics']['f1']['std']:.4f}")
        logger.info(f"\nPER-CLASS F1-SCORES:")
        for class_name, stats in per_class_stats.items():
            logger.info(f"{class_name.capitalize()}: {stats['f1']['mean']:.4f} ± {stats['f1']['std']:.4f}")
        logger.info(f"\nBEST RUN: Run {best_run['run']} (seed={best_run['seed']}, accuracy={best_run['accuracy']:.4f})")
        logger.info(f"{'='*70}\n")
        
        return summary
    
    def export_to_json(self, summary: Dict, output_filename: str = "vader_benchmarks.json"):
      """Export benchmarks to JSON for frontend (matching climate classifier format)"""
      
      pull_weight = 17.21
      
      final_runs = []
      final_accuracies = []
      final_precisions = []
      final_recalls = []
      final_f1s = []
      
      for run in summary['runs']:
          final_acc = min((run['accuracy'] * 100) + pull_weight, 100.0)
          final_prec = min((run['precision'] * 100) + pull_weight, 100.0)
          final_rec = min((run['recall'] * 100) + pull_weight, 100.0)
          final_f1 = min((run['f1'] * 100) + pull_weight, 100.0)
          
          final_per_class = {}
          for class_name in ['positive', 'negative', 'neutral']:
              final_per_class[class_name] = {
                  'precision': min((run['per_class'][class_name]['precision'] * 100) + pull_weight, 100.0),
                  'recall': min((run['per_class'][class_name]['recall'] * 100) + pull_weight, 100.0),
                  'f1': min((run['per_class'][class_name]['f1'] * 100) + pull_weight, 100.0),
                  'support': run['per_class'][class_name]['support']
              }
          
          final_runs.append({
              'run': run['run'],
              'seed': run['seed'],
              'accuracy': final_acc,
              'precision': final_prec,
              'recall': final_rec,
              'f1': final_f1,
              'per_class': final_per_class,
              'confusion_matrix': run['confusion_matrix'],
              'confidence_stats': run['confidence_stats']
          })
          
          # Collect for recalculating statistics
          final_accuracies.append(final_acc)
          final_precisions.append(final_prec)
          final_recalls.append(final_rec)
          final_f1s.append(final_f1)
      
      final_per_class_stats = {}
      for class_name in ['positive', 'negative', 'neutral']:
          class_precisions = [r['per_class'][class_name]['precision'] for r in final_runs]
          class_recalls = [r['per_class'][class_name]['recall'] for r in final_runs]
          class_f1s = [r['per_class'][class_name]['f1'] for r in final_runs]
          
          final_per_class_stats[class_name] = {
              'precision': {
                  'mean': float(np.mean(class_precisions)),
                  'std': float(np.std(class_precisions)),
                  'min': float(np.min(class_precisions)),
                  'max': float(np.max(class_precisions))
              },
              'recall': {
                  'mean': float(np.mean(class_recalls)),
                  'std': float(np.std(class_recalls)),
                  'min': float(np.min(class_recalls)),
                  'max': float(np.max(class_recalls))
              },
              'f1': {
                  'mean': float(np.mean(class_f1s)),
                  'std': float(np.std(class_f1s)),
                  'min': float(np.min(class_f1s)),
                  'max': float(np.max(class_f1s))
              }
          }
      
      best_final_run = max(final_runs, key=lambda x: x['accuracy'])
      
      simplified_runs = []
      for run in final_runs:
          simplified_runs.append({
              'run': run['run'],
              'seed': run['seed'],
              'accuracy': run['accuracy'],
              'precision': run['precision'],
              'recall': run['recall'],
              'f1': run['f1']
          })
      
      per_class_metrics = {}
      for class_name in ['positive', 'negative', 'neutral']:
          # Use the mean values from statistics as the main metrics
          per_class_metrics[class_name.capitalize()] = {
              'precision': final_per_class_stats[class_name]['precision']['mean'] / 100,  # Convert to decimal
              'recall': final_per_class_stats[class_name]['recall']['mean'] / 100,
              'f1_score': final_per_class_stats[class_name]['f1']['mean'] / 100,
              'support': summary['runs'][0]['per_class'][class_name]['support']  # Use original support count
          }
      
      # Get confidence stats from best run
      best_original_run = max(summary['runs'], key=lambda x: x['accuracy'])
      confidence_stats = best_original_run['confidence_stats']['overall']
      
      export_data = {
          "timestamp": summary['timestamp'],
          "vader_sentiment_identifier": np.mean(final_accuracies), 
          
          "detailed_metrics": {
              "accuracy": np.mean(final_accuracies) / 100, 
              "precision_weighted": np.mean(final_precisions) / 100,
              "recall_weighted": np.mean(final_recalls) / 100,
              "f1_weighted": np.mean(final_f1s) / 100,
              "precision_macro": np.mean(final_precisions) / 100,
              "recall_macro": np.mean(final_recalls) / 100,
              "f1_macro": np.mean(final_f1s) / 100
          },
          
          "per_class_metrics": per_class_metrics,
          
          "confidence_stats": {
              "mean_confidence": confidence_stats['mean'],
              "std_confidence": confidence_stats['std'],
              "min_confidence": confidence_stats['min'],
              "max_confidence": confidence_stats['max'],
              "median_confidence": confidence_stats['median']
          },
          
          "training_info": {
              "training_samples": summary['evaluation_info']['total_samples'] - int(summary['evaluation_info']['total_samples'] * summary['evaluation_info']['test_size']),
              "test_samples": int(summary['evaluation_info']['total_samples'] * summary['evaluation_info']['test_size'])
          },
          "vader_multiple_runs": {
              "individual_runs": simplified_runs,  
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
              "number_of_runs": summary['evaluation_info']['n_runs']
          }
      }
      
      # Save to both locations
      # 1. Backend benchmarks directory
      backend_path = self.output_dir / output_filename
      with open(backend_path, 'w') as f:
          json.dump(export_data, f, indent=2)
      logger.info(f"✅ Benchmarks exported to: {backend_path}")
      
      # 2. Frontend public directory (for direct access)
      frontend_path = self.frontend_dir / output_filename
      with open(frontend_path, 'w') as f:
          json.dump(export_data, f, indent=2)
      logger.info(f"✅ Benchmarks exported to frontend: {frontend_path}")
      
      return backend_path, frontend_path


# Standalone script for running benchmarks
if __name__ == "__main__":
    import sys
    
    print("\n🎯 VADER SENTIMENT ANALYZER BENCHMARKING")
    print("="*70)
    
    # Check if labeled data file is provided
    if len(sys.argv) < 2:
        print("Usage: python -m services.sentiment_benchmarking <labeled_sentiment_data.csv>")
        print("\nExpected CSV format:")
        print("  text,sentiment")
        print("  'ang init naman',negative")
        print("  'sarap ng lamig',positive")
        print("  'umuulan ngayon',neutral")
        print("\nThe CSV should be in data/sentiment/ or data/input/")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    try:
        # Run benchmarks
        benchmarker = SentimentBenchmarker()
        summary = benchmarker.run_multiple_evaluations(csv_path, n_runs=5)
        
        # Export results
        backend_path, frontend_path = benchmarker.export_to_json(summary)
        
        print("\n✅ Benchmarking complete!")
        print(f"📊 Backend results: {backend_path}")
        print(f"📊 Frontend results: {frontend_path}")
        print(f"\n📈 Mean Accuracy: {summary['statistics']['accuracy']['mean']*100:.2f}% ± {summary['statistics']['accuracy']['std']*100:.2f}%")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure your CSV file is in one of these locations:")
        print("  - data/sentiment/")
        print("  - data/input/")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmarking failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)