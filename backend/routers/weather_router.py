# weather_router.py

from fastapi import APIRouter
from pydantic import BaseModel
from services.weather_forecast_service import weather_forecast_service

router = APIRouter()

class WeatherRequest(BaseModel):
    location: str

class LocationSearchRequest(BaseModel):
    query: str

@router.post("/current")
async def get_current_weather_endpoint(request: WeatherRequest):
    """Get current weather data for a location"""
    weather_data = weather_forecast_service.get_current_weather(request.location)
    return {
        "status": "ok" if "error" not in weather_data else "error",
        "location": request.location,
        "data": weather_data
    }

@router.post("/hourly")
async def get_hourly_forecast_endpoint(request: WeatherRequest):
    """Get today's hourly forecast in 3-hour intervals"""
    forecast_data = weather_forecast_service.get_hourly_forecast(request.location)
    return {
        "status": "ok" if "error" not in forecast_data else "error",
        "location": request.location,
        "forecast": forecast_data
    }

@router.post("/weekly")
async def get_weekly_forecast_endpoint(request: WeatherRequest):
    """Get 7-day weather forecast"""
    forecast_data = weather_forecast_service.get_weekly_forecast(request.location)
    return {
        "status": "ok" if "error" not in forecast_data else "error",
        "location": request.location,
        "forecast": forecast_data
    }

@router.get("/locations")
async def get_available_locations():
    """Get list of supported locations"""
    return {
        "supported_locations": list(weather_forecast_service.location_mappings.keys()),
        "mappings": weather_forecast_service.location_mappings
    }

@router.post("/search-location")
async def search_location_endpoint(request: LocationSearchRequest):
    """
    Search for locations using WeatherAPI's search endpoint
    Returns matching locations with details
    """
    query = request.query.strip()
    
    if not query or len(query) < 2:
        return {
            "status": "error",
            "error": "Query must be at least 2 characters"
        }
    
    try:
        locations = weather_forecast_service.search_locations(query)
        
        if isinstance(locations, dict) and "error" in locations:
            return {
                "status": "error",
                "error": locations["error"]
            }
        
        return {
            "status": "ok",
            "locations": locations,
            "count": len(locations)
        }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }