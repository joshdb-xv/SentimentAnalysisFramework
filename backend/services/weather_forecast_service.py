# services/weather_forecast_service.py

import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

class WeatherService:
    def __init__(self):
        self.api_key = os.getenv('WEATHER_API_KEY')
        self.base_url = os.getenv('WEATHER_API_BASE_URL', 'http://api.weatherapi.com/v1')
        
        # Keep location mappings as fallback for backward compatibility
        self.location_mappings = {
            "indang": "Indang, Cavite, Philippines",
            "manila": "Manila, Philippines",
            "quezon city": "Quezon City, Philippines",
            "cebu": "Cebu City, Philippines",
            "davao": "Davao City, Philippines"
        }
    
    def search_locations(self, query: str) -> List[Dict[str, Any]]:
        """Search for locations using WeatherAPI (Philippines only)"""
        try:
            url = f"{self.base_url}/search.json"
            params = {
                "key": self.api_key,
                "q": query
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            locations_data = response.json()
            
            # Filter to only include Philippines locations
            ph_locations = [
                {
                    "name": loc.get("name"),
                    "region": loc.get("region"),
                    "country": loc.get("country"),
                    "lat": loc.get("lat"),
                    "lon": loc.get("lon")
                }
                for loc in locations_data
                if loc.get("country", "").lower() in ["philippines", "ph"]
            ]
            
            return ph_locations
            
        except requests.exceptions.Timeout:
            return {"error": "Request timed out"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Connection error: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    
    def get_current_weather(self, location: str) -> Dict[str, Any]:
        """Get current weather data for a location"""
        try:
            # Use location directly (no more forced mapping)
            # Only use mapping as fallback if it exists
            mapped_location = self.location_mappings.get(location.lower(), location)
            
            url = f"{self.base_url}/current.json"
            params = {
                "key": self.api_key,
                "q": mapped_location,
                "aqi": "yes"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract relevant information
            current = data["current"]
            location_info = data["location"]
            
            return {
                "location": {
                    "name": location_info["name"],
                    "region": location_info["region"],
                    "country": location_info["country"],
                    "lat": location_info["lat"],
                    "lon": location_info["lon"],
                    "localtime": location_info["localtime"]
                },
                "current": {
                    "temp_c": current["temp_c"],
                    "condition": current["condition"]["text"],
                    "condition_icon": current["condition"]["icon"],
                    "wind_mph": current["wind_mph"],
                    "wind_kph": current["wind_kph"],
                    "wind_dir": current["wind_dir"],
                    "pressure_mb": current["pressure_mb"],
                    "precip_mm": current["precip_mm"],
                    "humidity": current["humidity"],
                    "cloud": current["cloud"],
                    "feelslike_c": current["feelslike_c"],
                    "windchill_c": current.get("windchill_c", current["feelslike_c"]),
                    "heatindex_c": current.get("heatindex_c", current["feelslike_c"]),
                    "uv": current["uv"],
                    "vis_km": current["vis_km"],
                    "is_day": current["is_day"]
                }
            }
            
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Weather data processing failed: {str(e)}"}
    
    def get_hourly_forecast(self, location: str, hours: int = 24) -> Dict[str, Any]:
        """Get hourly forecast for today (12 AM to 9 PM in 3-hour intervals)"""
        try:
            mapped_location = self.location_mappings.get(location.lower(), location)
            
            url = f"{self.base_url}/forecast.json"
            params = {
                "key": self.api_key,
                "q": mapped_location,
                "days": 1,
                "aqi": "no",
                "alerts": "no"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Get today's hourly forecast
            today_forecast = data["forecast"]["forecastday"][0]
            hourly_data = today_forecast["hour"]
            
            # Extract 3-hour intervals (12 AM, 3 AM, 6 AM, 9 AM, 12 PM, 3 PM, 6 PM, 9 PM)
            intervals = []
            for hour in [0, 3, 6, 9, 12, 15, 18, 21]:  # 0 = 12 AM, 12 = 12 PM, 21 = 9 PM
                if hour < len(hourly_data):
                    hour_data = hourly_data[hour]
                    intervals.append({
                        "time": hour_data["time"],
                        "time_epoch": hour_data["time_epoch"],
                        "temp_c": hour_data["temp_c"],
                        "condition": hour_data["condition"]["text"],
                        "condition_icon": hour_data["condition"]["icon"],
                        "humidity": hour_data["humidity"],
                        "chance_of_rain": hour_data["chance_of_rain"],
                        "wind_kph": hour_data["wind_kph"],
                        "wind_dir": hour_data["wind_dir"],
                        "hour_12": self._convert_to_12_hour(hour)
                    })
            
            return {
                "location": data["location"]["name"],
                "date": today_forecast["date"],
                "intervals": intervals,
                "sunrise": today_forecast["astro"]["sunrise"],
                "sunset": today_forecast["astro"]["sunset"]
            }
            
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Hourly forecast processing failed: {str(e)}"}
    
    def get_weekly_forecast(self, location: str) -> Dict[str, Any]:
        """Get weekly forecast (WeatherAPI free tier provides 3 days, we'll extend to 7 with logic)"""
        try:
            mapped_location = self.location_mappings.get(location.lower(), location)
            
            url = f"{self.base_url}/forecast.json"
            params = {
                "key": self.api_key,
                "q": mapped_location,
                "days": 3,  # Free tier limit
                "aqi": "no",
                "alerts": "no"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Process the 3 days we get from API
            forecast_days = []
            for day_data in data["forecast"]["forecastday"]:
                day_info = day_data["day"]
                forecast_days.append({
                    "date": day_data["date"],
                    "date_epoch": day_data["date_epoch"],
                    "day_name": self._get_day_name(day_data["date"]),
                    "max_temp_c": day_info["maxtemp_c"],
                    "min_temp_c": day_info["mintemp_c"],
                    "avg_temp_c": day_info["avgtemp_c"],
                    "condition": day_info["condition"]["text"],
                    "condition_icon": day_info["condition"]["icon"],
                    "chance_of_rain": day_info["daily_chance_of_rain"],
                    "max_wind_kph": day_info["maxwind_kph"],
                    "avg_humidity": day_info["avghumidity"],
                    "uv": day_info["uv"],
                    "is_estimated": False
                })
            
            # For days 4-7, we'll use trend analysis and historical patterns
            # This is a simplified approach for the free tier limitation
            extended_forecast = self._extend_forecast(forecast_days)
            
            return {
                "location": data["location"]["name"],
                "forecast_days": extended_forecast,
                "note": "Days 1-3 are actual forecasts, days 4-7 are trend-based estimates"
            }
            
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Weekly forecast processing failed: {str(e)}"}
    
    def _convert_to_12_hour(self, hour_24: int) -> str:
        """Convert 24-hour format to 12-hour format"""
        if hour_24 == 0:
            return "12AM"
        elif hour_24 < 12:
            return f"{hour_24}AM"
        elif hour_24 == 12:
            return "12PM"
        else:
            return f"{hour_24 - 12}PM"
    
    def _get_day_name(self, date_str: str) -> str:
        """Get day name from date string"""
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return date_obj.strftime("%A")
        except:
            return "Unknown"
    
    def _extend_forecast(self, actual_forecast: List[Dict]) -> List[Dict]:
        """Extend 3-day forecast to 7 days using trend analysis"""
        if len(actual_forecast) < 2:
            return actual_forecast
        
        extended = actual_forecast.copy()
        
        # Calculate trends from available data
        temp_trend = (actual_forecast[-1]["avg_temp_c"] - actual_forecast[0]["avg_temp_c"]) / len(actual_forecast)
        
        # Generate next 4 days
        for i in range(4):
            last_day = extended[-1]
            next_date = datetime.strptime(last_day["date"], "%Y-%m-%d") + timedelta(days=1)
            
            # Simple trend-based prediction
            estimated_temp = last_day["avg_temp_c"] + temp_trend
            
            extended_day = {
                "date": next_date.strftime("%Y-%m-%d"),
                "date_epoch": int(next_date.timestamp()),
                "day_name": next_date.strftime("%A"),
                "max_temp_c": round(estimated_temp + 2, 1),
                "min_temp_c": round(estimated_temp - 2, 1),
                "avg_temp_c": round(estimated_temp, 1),
                "condition": "Partly Cloudy",  # Default condition for extended forecast
                "condition_icon": "//cdn.weatherapi.com/weather/64x64/day/116.png",
                "chance_of_rain": 30,  # Default moderate chance
                "max_wind_kph": last_day.get("max_wind_kph", 15),
                "avg_humidity": last_day.get("avg_humidity", 70),
                "uv": last_day.get("uv", 5),
                "is_estimated": True  # Flag to indicate this is estimated
            }
            
            extended.append(extended_day)
        
        return extended
    
    def analyze_tweet_weather_context(self, tweet: str, location: str) -> Dict[str, Any]:
        """Analyze if tweet weather sentiment matches actual weather"""
        try:
            weather_data = self.get_current_weather(location)
            
            if "error" in weather_data:
                return {"error": f"Could not get weather data: {weather_data['error']}"}
            
            current_weather = weather_data["current"]
            
            # Simple keyword-based analysis
            tweet_lower = tweet.lower()
            
            # Weather keywords
            hot_keywords = ["hot", "warm", "sweltering", "scorching", "burning"]
            cold_keywords = ["cold", "chilly", "freezing", "frigid", "icy"]
            rain_keywords = ["rain", "raining", "wet", "drizzle", "downpour", "storm"]
            sunny_keywords = ["sunny", "bright", "clear", "sunshine"]
            
            # Analyze tweet sentiment about weather
            has_hot = any(word in tweet_lower for word in hot_keywords)
            has_cold = any(word in tweet_lower for word in cold_keywords)
            has_rain = any(word in tweet_lower for word in rain_keywords)
            has_sunny = any(word in tweet_lower for word in sunny_keywords)
            
            # Check consistency with actual weather
            temp = current_weather["temp_c"]
            condition = current_weather["condition"].lower()
            humidity = current_weather["humidity"]
            
            consistency_checks = []
            
            # Temperature consistency
            if has_hot and temp > 30:
                consistency_checks.append("Temperature perception matches (hot)")
            elif has_cold and temp < 20:
                consistency_checks.append("Temperature perception matches (cold)")
            elif has_hot and temp < 25:
                consistency_checks.append("Temperature mismatch (claims hot but actually cool)")
            elif has_cold and temp > 25:
                consistency_checks.append("Temperature mismatch (claims cold but actually warm)")
            
            # Rain consistency
            if has_rain and ("rain" in condition or humidity > 80):
                consistency_checks.append("Rain perception matches")
            elif has_rain and humidity < 60 and "rain" not in condition:
                consistency_checks.append("Rain mismatch (claims rain but dry conditions)")
            
            # Sunny consistency
            if has_sunny and ("sunny" in condition or "clear" in condition):
                consistency_checks.append("Sunny perception matches")
            elif has_sunny and ("rain" in condition or "cloud" in condition):
                consistency_checks.append("Sunny mismatch (claims sunny but cloudy/rainy)")
            
            overall_consistency = "Consistent" if len([c for c in consistency_checks if "matches" in c]) > 0 else "Unknown"
            if any("mismatch" in c for c in consistency_checks):
                overall_consistency = "Inconsistent"
            
            return {
                "tweet": tweet,
                "location": location,
                "weather_analysis": {
                    "current_temp": temp,
                    "current_condition": current_weather["condition"],
                    "humidity": humidity,
                    "tweet_weather_keywords": {
                        "hot_keywords": has_hot,
                        "cold_keywords": has_cold,
                        "rain_keywords": has_rain,
                        "sunny_keywords": has_sunny
                    },
                    "consistency_checks": consistency_checks,
                    "overall_consistency": overall_consistency
                }
            }
            
        except Exception as e:
            return {"error": f"Weather context analysis failed: {str(e)}"}

# Create global instance
weather_forecast_service = WeatherService()