# /routers/benchmarks_router.py

from fastapi import APIRouter, HTTPException
from pathlib import Path
import json

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
            "vader_sentiment": 81.58,
            "naive_bayes_climate_checker": None,
            "naive_bayes_domain_identifier": None,
            "multiple_runs": None,
            "climate_checker_multiple_runs": None  # ADD THIS
        }
        
        # Try to load climate checker benchmarks
        climate_checker_path = frontend_dir / "public" / "climaterelated_benchmarks.json"
        if climate_checker_path.exists():
            with open(climate_checker_path, 'r') as f:
                climate_data = json.load(f)
                all_benchmarks["naive_bayes_climate_checker"] = climate_data.get("naive_bayes_climate_checker")
                all_benchmarks["climate_checker_multiple_runs"] = climate_data.get("multiple_runs")  # ADD THIS
        
        # Try to load domain identifier benchmarks
        domain_path = frontend_dir / "public" / "climatedomain_benchmarks.json"
        if domain_path.exists():
            with open(domain_path, 'r') as f:
                domain_data = json.load(f)
                all_benchmarks["naive_bayes_domain_identifier"] = domain_data.get("naive_bayes_domain_identifier")
                all_benchmarks["multiple_runs"] = domain_data.get("multiple_runs")
        
        return all_benchmarks
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load benchmarks: {str(e)}"
        )