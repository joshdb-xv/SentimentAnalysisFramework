# services/weatherapi.py

import os
import requests
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import re

class WeatherValidator:
    def __init__(self):
        self.api_key = os.getenv('WEATHER_API_KEY')
        self.base_url = os.getenv('WEATHER_API_BASE_URL', 'http://api.weatherapi.com/v1')
        
        # Location mappings - you can expand this
        self.location_mappings = {
            "Indang": "Indang, Cavite, Philippines",
            "Trece Martires": "Trece Martires, Cavite, Philippines"
        }
        
        # Enhanced climate/disaster patterns aligned with your categories
        # Multi-language support: Tagalog, Cebuano, Taglish, English
        self.weather_patterns = {
            "sea_level_rise_coastal": {
                "exact": ["sea level", "coastal", "erosion", "storm surge", "high tide", "pagbaha sa baybayin", 
                         "pagtaas ng dagat", "dalampasigan", "erosyon", "storm surge", "mataas na agos"],
                "cebuano": ["baybay", "dagat", "balod", "taob sa dagat", "kusog nga balod"],
                "patterns": [
                    r"\bsea\s+level\b",
                    r"\bcoastal\s+(erosion|flooding|hazard)\b",
                    r"\bstorm\s+surge\b",
                    r"\bhigh\s+tide\b",
                    r"\b(pag)?baha.*baybayin\b",
                    r"\b(pag)?taas.*dagat\b",
                    r"\berosion\b",
                    r"\berosyon\b",
                    r"\bdalampasigan\b",
                    r"\bbaybay\b",
                    r"\b(mataas|kusog).*balod\b",
                    r"\b(taob|luloy).*dagat\b"
                ]
            },
            "extreme_heat": {
                "exact": ["init", "mainit", "hot", "warm", "heatwave", "extreme heat", "sobrang init", "grabe ang init", 
                         "napakainit", "heat index", "el nino", "tag-init", "heat stroke"],
                "cebuano": ["init kaayo", "hilabihan ka init", "grabe ka init", "hapdi kaayo"],
                "patterns": [
                    r"\b(ma|napaka|sobra|grabe).*init\b",
                    r"\bhot\b",
                    r"\bwarm\b",
                    r"\bheatwave\b",
                    r"\bextreme\s+heat\b",
                    r"\bheat\s+(index|stroke|exhaustion)\b",
                    r"\bel\s+ni[ñn]o\b",
                    r"\btag.?init\b",
                    r"\b(nag|mag).*init\b",
                    r"\binit\s+(kaayo|banay)\b",
                    r"\bhilabihan.*init\b",
                    r"\bhapdi\s+kaayo\b"
                ]
            },
            "cold_weather": {
                "exact": ["malamig", "cold", "ginaw", "presko", "cool", "sobrang lamig", "nakakalamig", "la nina",
                         "tag-lamig", "winter", "freeze", "frost"],
                "cebuano": ["bugnaw", "tugnaw", "mabugnaw", "grabe ka bugnaw"],
                "patterns": [
                    r"\b(ma|nakaka|sobra|napaka).*lamig\b",
                    r"\bginaw\b",
                    r"\bcold\b",
                    r"\bcool\b",
                    r"\bpresko\b",
                    r"\bla\s+ni[ñn]a\b",
                    r"\btag.?lamig\b",
                    r"\bwinter\b",
                    r"\bfreez(e|ing)\b",
                    r"\bfrost\b",
                    r"\b(ma)?bugnaw\b",
                    r"\btugnaw\b",
                    r"\bgrabe.*bugnaw\b"
                ]
            },
            "flooding_precipitation": {
                "exact": ["ulan", "rain", "baha", "flood", "flashflood", "umulan", "maulan", "ambon", "uulan",
                         "pagbaha", "tubig-baha", "overflowing", "spillway", "dam", "monsoon", "habagat"],
                "cebuano": ["baha", "ulan", "lunop", "taob", "dalayegon", "kusog nga ulan"],
                "patterns": [
                    r"\b(u+|a+)mbon\b",
                    r"\b(u+)(u+)?lan\b",
                    r"\b(um|mag|nag|in|ni).*ulan\b",
                    r"\b(ma|pag).*ulan\b",
                    r"\brain(ing|y|ed)?\b",
                    r"\b(pag)?baha\b",
                    r"\bflood(ing)?\b",
                    r"\bflashflood\b",
                    r"\boverflow(ing)?\b",
                    r"\bspillway\b",
                    r"\bdam\s+(overflow|release)\b",
                    r"\bmonsoon\b",
                    r"\bhabagat\b",
                    r"\blunop\b",
                    r"\btaob\b",
                    r"\bdalayegon\b",
                    r"\bkusog.*ulan\b"
                ]
            },
            "storms_typhoons": {
                "exact": ["bagyo", "typhoon", "storm", "cyclone", "hurricane", "tornado", "bagyong", "signal",
                         "landfall", "eyewall", "winds", "hangin", "malakas na hangin", "unos"],
                "cebuano": ["bagyo", "bagio", "kusog nga hangin", "storm", "unos"],
                "patterns": [
                    r"\bbagyo\b",
                    r"\btyphoon\b",
                    r"\bstorm\b",
                    r"\bcyclone\b",
                    r"\bhurricane\b",
                    r"\btornado\b",
                    r"\bsignal\s+(no|#)?\s*[1-5]\b",
                    r"\blandfall\b",
                    r"\beyewall\b",
                    r"\b(malakas|strong).*hangin\b",
                    r"\bwinds?\b",
                    r"\bunos\b",
                    r"\bbagio\b",
                    r"\bkusog.*hangin\b"
                ]
            },
            "drought_water_scarcity": {
                "exact": ["drought", "tagtuyot", "water shortage", "kakulang sa tubig", "dry", "tuyo", "walang ulan",
                         "water crisis", "reservoir", "dam level", "irrigation", "patubig"],
                "cebuano": ["hulaw", "uga", "walay ulan", "kakulang sa tubig"],
                "patterns": [
                    r"\bdrought\b",
                    r"\btag.?tuyot\b",
                    r"\bwater\s+(shortage|crisis|scarcity)\b",
                    r"\bkakulang.*tubig\b",
                    r"\bdry\b",
                    r"\btuyo\b",
                    r"\bwala.*ulan\b",
                    r"\breservoir\s+(low|empty)\b",
                    r"\bdam\s+level\b",
                    r"\birrigation\b",
                    r"\bpatubig\b",
                    r"\bhulaw\b",
                    r"\buga\b",
                    r"\bwalay\s+ulan\b"
                ]
            },
            "air_pollution": {
                "exact": ["pollution", "smog", "haze", "air quality", "emissions", "smoke", "usok", "polusyon",
                         "pm2.5", "aqi", "particulates", "toxic", "lason sa hangin"],
                "cebuano": ["hugaw nga hangin", "aso", "usok", "polusyon"],
                "patterns": [
                    r"\bpollution\b",
                    r"\bsmog\b",
                    r"\bhaze\b",
                    r"\bair\s+quality\b",
                    r"\bemissions?\b",
                    r"\bsmoke\b",
                    r"\busok\b",
                    r"\bpolusyon\b",
                    r"\bpm\s*2\.5\b",
                    r"\baqi\b",
                    r"\bparticulates?\b",
                    r"\btoxic\b",
                    r"\blason.*hangin\b",
                    r"\bhugaw.*hangin\b",
                    r"\baso\b"
                ]
            },
            "environmental_degradation": {
                "exact": ["deforestation", "logging", "mining", "land conversion", "habitat loss", "pagputol ng puno",
                         "pagmimina", "degradation", "ecosystem", "biodiversity", "kalikasan"],
                "cebuano": ["pagputol sa kahoy", "pagmina", "guba sa kalikasan"],
                "patterns": [
                    r"\bdeforestation\b",
                    r"\blogging\b",
                    r"\bmining\b",
                    r"\bland\s+(conversion|use|change)\b",
                    r"\bhabitat\s+loss\b",
                    r"\bpagputol.*puno\b",
                    r"\bpagmimina\b",
                    r"\bdegradation\b",
                    r"\becosystem\b",
                    r"\bbiodiversity\b",
                    r"\bkalikasan\b",
                    r"\bpagputol.*kahoy\b",
                    r"\bpagmina\b",
                    r"\bguba.*kalikasan\b"
                ]
            },
            "geological_events": {
                "exact": ["earthquake", "lindol", "tremor", "magnitude", "richter", "epicenter", "fault", "tsunami",
                         "volcanic", "eruption", "lahar", "ash", "landslide", "guho", "pagguho"],
                "cebuano": ["linog", "kulog sa yuta", "pagguho", "volcanic"],
                "patterns": [
                    r"\bearthquake\b",
                    r"\blindol\b",
                    r"\btremor\b",
                    r"\bmagnitude\s+\d+\b",
                    r"\brichter\b",
                    r"\bepicenter\b",
                    r"\bfaults?\b",
                    r"\btsunami\b",
                    r"\bvolcanic\b",
                    r"\beruption\b",
                    r"\blahar\b",
                    r"\bash\s+(fall|cloud)\b",
                    r"\blandslides?\b",
                    r"\bguho\b",
                    r"\bpagguho\b",
                    r"\blinog\b",
                    r"\bkulog.*yuta\b"
                ]
            }
        }

    def get_current_weather(self, location: str) -> Optional[Dict]:
        """Get current weather data from WeatherAPI"""
        if not self.api_key:
            return {"error": "Weather API key not configured"}
        
        # Map location to full address if available
        full_location = self.location_mappings.get(location, location)
        
        try:
            url = f"{self.base_url}/current.json"
            params = {
                "key": self.api_key,
                "q": full_location,
                "aqi": "no"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return {
                "location": data["location"]["name"],
                "region": data["location"]["region"],
                "country": data["location"]["country"],
                "temperature_c": data["current"]["temp_c"],
                "temperature_f": data["current"]["temp_f"],
                "condition": data["current"]["condition"]["text"],
                "humidity": data["current"]["humidity"],
                "feels_like_c": data["current"]["feelslike_c"],
                "feels_like_f": data["current"]["feelslike_f"],
                "wind_kph": data["current"]["wind_kph"],
                "uv": data["current"]["uv"],
                "last_updated": data["current"]["last_updated"]
            }
            
        except requests.RequestException as e:
            return {"error": f"Weather API request failed: {str(e)}"}
        except KeyError as e:
            return {"error": f"Unexpected weather API response format: {str(e)}"}
        except Exception as e:
            return {"error": f"Weather data retrieval failed: {str(e)}"}

    def find_weather_matches(self, text: str, climate_category: str) -> Dict:
        """Find climate/disaster-related matches using multiple strategies"""
        text_lower = text.lower()
        patterns = self.weather_patterns.get(climate_category, {})
        
        matches = []
        match_details = {
            "exact_matches": [],
            "cebuano_matches": [],
            "pattern_matches": [],
            "total_matches": 0,
            "confidence": 0.0
        }
        
        # 1. Check exact matches (Tagalog/English)
        for exact_word in patterns.get("exact", []):
            if exact_word.lower() in text_lower:
                matches.append(exact_word)
                match_details["exact_matches"].append(exact_word)
        
        # 2. Check Cebuano matches
        for cebuano_word in patterns.get("cebuano", []):
            if cebuano_word.lower() in text_lower:
                matches.append(cebuano_word)
                match_details["cebuano_matches"].append(cebuano_word)
        
        # 3. Check regex patterns
        for pattern in patterns.get("patterns", []):
            regex_matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if regex_matches:
                for match in regex_matches:
                    if isinstance(match, tuple):
                        match = ''.join(match)  # Join tuple elements
                    matches.append(match)
                    match_details["pattern_matches"].append({
                        "pattern": pattern,
                        "match": match
                    })
        
        # 4. Calculate confidence based on matches
        total_matches = len(matches)
        match_details["total_matches"] = total_matches
        
        if total_matches > 0:
            # Base confidence on number of matches and type
            base_confidence = min(total_matches * 0.3, 1.0)
            
            # Boost confidence for exact matches
            exact_boost = len(match_details["exact_matches"]) * 0.25
            
            # Boost confidence for Cebuano matches (shows local knowledge)
            cebuano_boost = len(match_details["cebuano_matches"]) * 0.25
            
            # Boost confidence for pattern matches
            pattern_boost = len(match_details["pattern_matches"]) * 0.15
            
            match_details["confidence"] = min(base_confidence + exact_boost + cebuano_boost + pattern_boost, 1.0)
        
        return match_details

    def extract_weather_sentiment(self, tweet_text: str) -> Dict:
        """Extract climate/disaster sentiment from tweet text using enhanced matching"""
        detected_categories = []
        confidence_scores = {}
        all_matches = {}
        
        for climate_category in self.weather_patterns.keys():
            match_result = self.find_weather_matches(tweet_text, climate_category)
            
            if match_result["total_matches"] > 0:
                detected_categories.append(climate_category)
                confidence_scores[climate_category] = {
                    "confidence": match_result["confidence"],
                    "exact_matches": match_result["exact_matches"],
                    "cebuano_matches": match_result["cebuano_matches"],
                    "pattern_matches": match_result["pattern_matches"],
                    "total_matches": match_result["total_matches"]
                }
                all_matches[climate_category] = match_result
        
        return {
            "detected_categories": detected_categories,
            "confidence_scores": confidence_scores,
            "has_climate_content": len(detected_categories) > 0,
            "detailed_matches": all_matches
        }

    def validate_weather_consistency(self, tweet_text: str, weather_data: Dict) -> Dict:
        """Validate if tweet climate/disaster sentiment matches actual weather conditions"""
        if "error" in weather_data:
            return {
                "validation_status": "error",
                "error": weather_data["error"],
                "consistency": "unknown"
            }
        
        sentiment_analysis = self.extract_weather_sentiment(tweet_text)
        
        if not sentiment_analysis["has_climate_content"]:
            return {
                "validation_status": "no_climate_content",
                "consistency": "not_applicable",
                "message": "Tweet does not contain climate/disaster-related content",
                "debug_info": {
                    "original_text": tweet_text,
                    "processed_text": tweet_text.lower(),
                    "categories_tested": list(self.weather_patterns.keys())
                }
            }
        
        # Weather thresholds (adjusted for better accuracy)
        temp_c = weather_data["temperature_c"]
        feels_like_c = weather_data["feels_like_c"]
        humidity = weather_data["humidity"]
        condition = weather_data["condition"].lower()
        wind_kph = weather_data["wind_kph"]
        
        # Define climate condition thresholds with more nuanced heat detection
        # For Philippines climate context:
        # - Extreme heat: feels like >= 37°C OR actual temp >= 34°C 
        # - Hot: feels like >= 32°C OR actual temp >= 30°C
        # - Consider both actual and feels-like temperature for heat assessment
        is_extreme_heat = feels_like_c >= 37 or temp_c >= 34
        is_hot = feels_like_c >= 32 or temp_c >= 30
        is_very_hot = feels_like_c >= 35 or temp_c >= 32  # Added intermediate level
        
        is_cold = temp_c <= 20 or feels_like_c <= 18
        is_humid = humidity >= 75
        is_very_humid = humidity >= 85
        is_rainy = any(word in condition for word in ["rain", "drizzle", "shower", "storm", "thunderstorm"])
        is_stormy = any(word in condition for word in ["storm", "thunderstorm"]) or wind_kph >= 50
        is_windy = wind_kph >= 25
        is_sunny = any(word in condition for word in ["sunny", "clear", "bright"])
        is_cloudy = any(word in condition for word in ["cloud", "overcast", "partly"])
        
        # Check consistency for each detected category
        consistency_checks = []
        overall_consistent = True
        
        detected_categories = sentiment_analysis["detected_categories"]
        
        for category in detected_categories:
            consistent = False
            reason = ""
            confidence = sentiment_analysis["confidence_scores"][category]["confidence"]
            
            if category == "extreme_heat":
                # More flexible heat detection: consider various heat levels
                # Also consider high humidity as amplifying heat perception
                heat_condition = is_extreme_heat or is_very_hot or (is_hot and is_humid)
                consistent = heat_condition
                
                # Provide detailed reason based on conditions
                heat_factors = []
                if feels_like_c >= 37:
                    heat_factors.append(f"feels extremely hot ({feels_like_c}°C)")
                elif feels_like_c >= 35:
                    heat_factors.append(f"feels very hot ({feels_like_c}°C)")
                elif feels_like_c >= 32:
                    heat_factors.append(f"feels hot ({feels_like_c}°C)")
                
                if temp_c >= 34:
                    heat_factors.append(f"high temperature ({temp_c}°C)")
                elif temp_c >= 30:
                    heat_factors.append(f"warm temperature ({temp_c}°C)")
                
                if is_humid:
                    heat_factors.append(f"high humidity ({humidity}%)")
                
                if heat_factors:
                    reason = f"Heat conditions present: {', '.join(heat_factors)}"
                else:
                    reason = f"Temperature: {temp_c}°C (feels like {feels_like_c}°C), Humidity: {humidity}%"
                
            elif category == "cold_weather":
                consistent = is_cold
                reason = f"Temperature: {temp_c}°C (feels like {feels_like_c}°C)"
                
            elif category == "flooding_precipitation":
                consistent = is_rainy or is_very_humid
                reason = f"Condition: {weather_data['condition']}, Humidity: {humidity}%"
                
            elif category == "storms_typhoons":
                consistent = is_stormy or is_windy
                reason = f"Condition: {weather_data['condition']}, Wind: {wind_kph} kph"
                
            elif category == "air_pollution":
                # Air pollution might not correlate directly with basic weather
                consistent = True  # Assume consistent unless we have AQI data
                reason = "Air quality data not available in weather API"
                
            elif category in ["sea_level_rise_coastal", "drought_water_scarcity", 
                             "environmental_degradation", "geological_events"]:
                # These categories don't directly correlate with current weather
                consistent = True  # Consider these as informational
                reason = "Long-term climate/disaster category - not weather dependent"
            
            consistency_checks.append({
                "category": category,
                "consistent": consistent,
                "reason": reason,
                "confidence": confidence,
                "matches_found": sentiment_analysis["confidence_scores"][category]
            })
            
            # Only mark as inconsistent for weather-related categories
            if category in ["extreme_heat", "cold_weather", "flooding_precipitation", "storms_typhoons"] and not consistent:
                overall_consistent = False
        
        # Calculate overall consistency score
        if consistency_checks:
            weather_dependent_checks = [c for c in consistency_checks 
                                      if c["category"] in ["extreme_heat", "cold_weather", "flooding_precipitation", "storms_typhoons"]]
            if weather_dependent_checks:
                consistent_count = sum(1 for check in weather_dependent_checks if check["consistent"])
                consistency_score = consistent_count / len(weather_dependent_checks)
            else:
                consistency_score = 1.0  # All categories are non-weather dependent
        else:
            consistency_score = 0
        
        return {
            "validation_status": "completed",
            "consistency": "consistent" if overall_consistent else "inconsistent",
            "consistency_score": consistency_score,
            "sentiment_analysis": sentiment_analysis,
            "weather_conditions": {
                "temperature_c": temp_c,
                "feels_like_c": feels_like_c,
                "condition": weather_data["condition"],
                "humidity": humidity,
                "wind_kph": wind_kph,
                "is_extreme_heat": is_extreme_heat,
                "is_very_hot": is_very_hot,  # Added this new level
                "is_hot": is_hot,
                "is_cold": is_cold,
                "is_rainy": is_rainy,
                "is_stormy": is_stormy,
                "is_humid": is_humid,
                "is_very_humid": is_very_humid,
                "is_windy": is_windy,
                "is_sunny": is_sunny,
                "is_cloudy": is_cloudy
            },
            "detailed_checks": consistency_checks,
            "timestamp": datetime.now().isoformat()
        }

    def analyze_tweet_weather_context(self, tweet_text: str, location: str) -> Dict:
        """Main function to analyze tweet against weather data"""
        print(f"🌤️  Analyzing weather context for location: {location}")
        print(f"📝 Tweet: {tweet_text[:100]}...")
        
        # Get current weather data
        weather_data = self.get_current_weather(location)
        
        if "error" in weather_data:
            print(f"❌ Weather API Error: {weather_data['error']}")
            return {
                "status": "error",
                "error": weather_data["error"],
                "location": location,
                "tweet": tweet_text
            }
        
        # Validate consistency
        validation_result = self.validate_weather_consistency(tweet_text, weather_data)
        
        print(f"🔍 Climate/Disaster validation: {validation_result['consistency']}")
        if validation_result["validation_status"] == "completed":
            print(f"📊 Consistency score: {validation_result['consistency_score']:.2f}")
            print(f"🎯 Detected categories: {validation_result['sentiment_analysis']['detected_categories']}")
            
            # Print language breakdown
            for category, scores in validation_result['sentiment_analysis']['confidence_scores'].items():
                langs = []
                if scores['exact_matches']: langs.append("Tagalog/English")
                if scores['cebuano_matches']: langs.append("Cebuano")
                if scores['pattern_matches']: langs.append("Pattern-based")
                print(f"   📝 {category}: {', '.join(langs)} (confidence: {scores['confidence']:.2f})")
        
        return {
            "status": "success",
            "location": location,
            "tweet": tweet_text,
            "weather_data": weather_data,
            "validation": validation_result
        }

# Global instance
weather_validator = WeatherValidator()