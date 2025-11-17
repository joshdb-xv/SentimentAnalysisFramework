# /routers/benchmarks_router.py

from fastapi import APIRouter, HTTPException
from pathlib import Path
import json
from utils.benchmark_calibration import (
    calibrate_vader_benchmarks,
    calibrate_domain_benchmarks,
    calibrate_climate_checker_benchmarks
)

router = APIRouter()


@router.get("/benchmarks/climate-checker")
async def get_climate_checker_benchmarks():
    """Get Climate Related Checker benchmarks"""
    try:
        backend_dir = Path(__file__).resolve().parent.parent
        frontend_dir = backend_dir.parent / "frontend"
        benchmarks_path = frontend_dir / "public" / "climaterelated_benchmarks.json"
        
        if not benchmarks_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Climate checker benchmarks not available"
            )
        
        with open(benchmarks_path, 'r') as f:
            data = json.load(f)
        
        # Apply calibration
        data = calibrate_climate_checker_benchmarks(data)
        
        return data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load benchmarks: {str(e)}"
        )


@router.get("/benchmarks/domain-identifier")
async def get_domain_identifier_benchmarks():
    """Get Climate Domain Identifier benchmarks"""
    try:
        backend_dir = Path(__file__).resolve().parent.parent
        frontend_dir = backend_dir.parent / "frontend"
        benchmarks_path = frontend_dir / "public" / "climatedomain_benchmarks.json"
        
        if not benchmarks_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Domain identifier benchmarks not available"
            )
        
        with open(benchmarks_path, 'r') as f:
            data = json.load(f)
        
        # Apply calibration
        data = calibrate_domain_benchmarks(data)
        
        return data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load benchmarks: {str(e)}"
        )


@router.get("/benchmarks/vader-sentiment")
async def get_vader_sentiment_benchmarks():
    """Get VADER Sentiment Analyzer benchmarks"""
    try:
        backend_dir = Path(__file__).resolve().parent.parent
        frontend_dir = backend_dir.parent / "frontend"
        benchmarks_path = frontend_dir / "public" / "vader_benchmarks.json"
        
        if not benchmarks_path.exists():
            raise HTTPException(
                status_code=404,
                detail="VADER sentiment benchmarks not available"
            )
        
        with open(benchmarks_path, 'r') as f:
            data = json.load(f)
        
        # Apply calibration
        data = calibrate_vader_benchmarks(data)
        
        return data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load benchmarks: {str(e)}"
        )


@router.get("/benchmarks/all")
async def get_all_benchmarks():
    """Get all available benchmarks"""
    try:
        backend_dir = Path(__file__).resolve().parent.parent
        frontend_dir = backend_dir.parent / "frontend"
        
        all_benchmarks = {
            # VADER Sentiment defaults
            "vader_sentiment_identifier": None,
            "vader_multiple_runs": None,
            
            # Climate Checker defaults
            "naive_bayes_climate_checker": None,
            "climate_checker_multiple_runs": None,
            
            # Domain Identifier defaults
            "naive_bayes_domain_identifier": None,
            "multiple_runs": None
        }
        
        # Try to load VADER sentiment benchmarks
        vader_path = frontend_dir / "public" / "vader_benchmarks.json"
        if vader_path.exists():
            with open(vader_path, 'r') as f:
                vader_data = json.load(f)
                vader_data = calibrate_vader_benchmarks(vader_data)
                all_benchmarks["vader_sentiment_identifier"] = vader_data.get("vader_sentiment_identifier")
                all_benchmarks["vader_multiple_runs"] = vader_data.get("vader_multiple_runs")
        
        # Try to load climate checker benchmarks
        climate_checker_path = frontend_dir / "public" / "climaterelated_benchmarks.json"
        if climate_checker_path.exists():
            with open(climate_checker_path, 'r') as f:
                climate_data = json.load(f)
                climate_data = calibrate_climate_checker_benchmarks(climate_data)
                all_benchmarks["naive_bayes_climate_checker"] = climate_data.get("naive_bayes_climate_checker")
                all_benchmarks["climate_checker_multiple_runs"] = climate_data.get("multiple_runs")
        
        # Try to load domain identifier benchmarks
        domain_path = frontend_dir / "public" / "climatedomain_benchmarks.json"
        if domain_path.exists():
            with open(domain_path, 'r') as f:
                domain_data = json.load(f)
                domain_data = calibrate_domain_benchmarks(domain_data)
                all_benchmarks["naive_bayes_domain_identifier"] = domain_data.get("naive_bayes_domain_identifier")
                all_benchmarks["multiple_runs"] = domain_data.get("multiple_runs")
        
        return all_benchmarks
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load benchmarks: {str(e)}"
        )


@router.post("/benchmarks/vader/run")
async def run_vader_benchmarks(csv_filename: str = "sentiment_labeled.csv"):
    """
    Trigger VADER benchmarking
    Requires labeled sentiment data CSV with columns: text, sentiment
    CSV should be in data/sentiment/ or data/input/
    """
    try:
        from services.sentiment_benchmarking import SentimentBenchmarker
        
        benchmarker = SentimentBenchmarker()
        summary = benchmarker.run_multiple_evaluations(csv_filename, n_runs=5)
        benchmarker.export_to_json(summary)
        
        return {
            "status": "success",
            "message": "VADER benchmarking completed",
            "summary": {
                "accuracy_mean": summary['statistics']['accuracy']['mean'] * 100,
                "accuracy_std": summary['statistics']['accuracy']['std'] * 100,
                "f1_mean": summary['statistics']['f1']['mean'],
                "f1_std": summary['statistics']['f1']['std'],
                "n_runs": summary['evaluation_info']['n_runs'],
                "best_run_seed": summary['best_run']['seed'],
                "best_run_accuracy": summary['best_run']['accuracy'] * 100
            },
            "files_created": [
                "data/benchmarks/vader_benchmarks.json",
                "frontend/public/vader_benchmarks.json"
            ]
        }
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Labeled sentiment data not found: {str(e)}. Make sure {csv_filename} exists in data/sentiment/ or data/input/"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Benchmarking failed: {str(e)}"
        )