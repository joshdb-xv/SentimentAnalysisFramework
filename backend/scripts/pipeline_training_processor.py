# pipeline_training_processor.py
import pandas as pd
import requests
import json
import time
from typing import Dict, List
import asyncio
import aiohttp

class TrainingDataProcessor:
    """Process training data through your existing pipeline to get proper tags"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.processed_count = 0
        self.error_count = 0
    
    def clean_location_for_weather(self, location: str) -> str:
        """Clean location and handle Filipino Twitter shenanigans based on actual data patterns"""
        if not location or pd.isna(location) or str(location).strip() == "":
            return ""
        
        location = str(location).strip()
        location_lower = location.lower()
        
        # Obvious fake/joke locations from your actual data
        fake_patterns = [
            # Heart/love related
            "sa heart", "sa puso", "puso ni", "heart ni", "heart of",
            # Joke locations
            "somewhere", "anywhere", "dito lang", "dyan lang", "sa tabi",
            "sa mundo", "earth", "mars", "saturn", "jupiter", "kepler",
            "sa langit", "heaven", "nasa dreams", "wonderland", "neverland",
            # Relationship jokes
            "crush", "jowa", "syota", "baby", "mahal", "babe", "honey",
            # Sarcastic/attitude
            "hindi mo business", "secret", "di mo kailangan malaman",
            # Meme locations
            "biringan", "wakanda", "asgard", "atlantis", "hogwarts",
            "sa isip", "sa utak", "delulu", "sa imagination",
            # Generic/vague
            "dito", "doon", "jan", "dun", "diha", "dira",
            "kahit saan", "kung saan", "pwede matulog kung saan"
        ]
        
        # Check for fake patterns
        for fake in fake_patterns:
            if fake in location_lower:
                return ""
        
        # Real Philippine locations (based on your data)
        valid_locations = [
            # Major cities
            "manila", "cebu", "davao", "quezon", "makati", "taguig", "pasig",
            "marikina", "antipolo", "las pinas", "paranaque", "muntinlupa",
            "caloocan", "malabon", "navotas", "valenzuela", "pasay", "mandaluyong",
            # Regions and provinces
            "metro manila", "ncr", "laguna", "cavite", "bulacan", "rizal", 
            "batangas", "pampanga", "zambales", "tarlac", "nueva ecija",
            "baguio", "ilocos", "cagayan", "isabela", "pangasinan",
            "albay", "bicol", "camarines", "sorsogon", "masbate",
            "leyte", "samar", "bohol", "negros", "iloilo", "capiz",
            "aklan", "antique", "palawan", "mindoro", "romblon",
            "surigao", "agusan", "bukidnon", "misamis", "camiguin",
            "lanao", "maguindanao", "cotabato", "zamboanga", "basilan",
            # Specific areas from your data
            "cagayan de oro", "bacolod", "dumaguete", "tacloban", "butuan",
            "general santos", "puerto princesa", "naga", "iloilo city",
            "lapu-lapu", "mandaue", "imus", "dasmarinas", "bacoor",
            "antipolo", "marikina", "san juan", "paranaque", "muntinlupa",
            "taguig", "pasig", "makati", "manila", "quezon city"
        ]
        
        # Check if it's a valid Philippine location
        for valid_loc in valid_locations:
            if valid_loc in location_lower:
                return location
        
        # Check for location indicators
        location_indicators = [
            "city", "province", "region", "philippines", "ph", "pilipinas",
            "luzon", "visayas", "mindanao", "republic of the philippines",
            "calabarzon", "central luzon", "central visayas", "western visayas",
            "eastern visayas", "northern mindanao", "southern mindanao",
            "caraga", "davao region", "cordillera"
        ]
        
        for indicator in location_indicators:
            if indicator in location_lower:
                return location
        
        # If we can't classify it as Philippine location, treat as no location
        return ""
    
    async def process_single_tweet(self, session: aiohttp.ClientSession, tweet_data: Dict) -> Dict:
        """Process a single tweet through your pipeline"""
        try:
            # Clean location first
            cleaned_location = self.clean_location_for_weather(tweet_data.get("location", ""))
            
            # Prepare the payload for your analysis endpoint
            payload = {
                "tweet": tweet_data["tweet"],
                "location": cleaned_location,  # Use cleaned location
            }
            
            # Call your existing analysis endpoint - UPDATE THIS to match your actual endpoint
            async with session.post(
                f"{self.api_base_url}/analyze-single-tweet",  # Update this to your actual endpoint path
                json=payload,
                timeout=30
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    self.processed_count += 1
                    
                    if self.processed_count % 10 == 0:
                        print(f"✅ Processed {self.processed_count} tweets...")
                    
                    return {
                        "original_data": tweet_data,
                        "pipeline_result": result,
                        "status": "success"
                    }
                else:
                    error_text = await response.text()
                    self.error_count += 1
                    return {
                        "original_data": tweet_data,
                        "error": f"HTTP {response.status}: {error_text}",
                        "status": "error"
                    }
                    
        except Exception as e:
            self.error_count += 1
            return {
                "original_data": tweet_data,
                "error": str(e),
                "status": "error"
            }
    
    async def process_batch(self, tweets: List[Dict], batch_size: int = 5) -> List[Dict]:
        """Process tweets in batches to avoid overwhelming your API"""
        results = []
        
        async with aiohttp.ClientSession() as session:
            # Process in small batches to avoid rate limiting
            for i in range(0, len(tweets), batch_size):
                batch = tweets[i:i + batch_size]
                
                # Process batch concurrently
                tasks = [self.process_single_tweet(session, tweet) for tweet in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                results.extend(batch_results)
                
                # Small delay between batches
                if i + batch_size < len(tweets):
                    await asyncio.sleep(1)  # 1 second delay between batches
        
        return results
    
    def load_and_prepare_csv(self, csv_path: str) -> List[Dict]:
        """Load CSV and prepare tweet data"""
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            print(f"📊 Loaded {len(df)} rows from CSV")
            print(f"🗂️  Columns: {list(df.columns)}")
            
            # Prepare tweet data - adjust column names as needed
            tweets = []
            for _, row in df.iterrows():
                tweet_data = {
                    "tweet": str(row.get("tweet", row.get("text", ""))).strip(),
                    "location": str(row.get("location", row.get("user_location", ""))).strip(),
                    "original_climate_flag": row.get("is_climate_related", None),
                    "original_domains": str(row.get("climate_domain", "")).strip(),
                    "row_index": _
                }
                
                # Skip empty tweets
                if tweet_data["tweet"] and tweet_data["tweet"] != "nan":
                    tweets.append(tweet_data)
            
            print(f"✅ Prepared {len(tweets)} valid tweets for processing")
            return tweets
            
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return []

def save_results_to_csv(results: List[Dict], output_path: str):
    """Save processed results to CSV with proper location handling"""
    processed_data = []
    
    for result in results:
        if result["status"] == "success":
            original = result["original_data"]
            pipeline = result["pipeline_result"]
            
            # Handle location display logic
            original_location = original.get("location", "")
            display_location = "No Location" if not original_location or original_location.strip() == "" else original_location
            
            # Extract all the tags from your pipeline result
            row_data = {
                # Original data
                "tweet": original["tweet"],
                "location": display_location,  # Use display-friendly location
                "original_location": original.get("location", ""),  # Keep raw location for reference
                "original_climate_flag": original.get("original_climate_flag"),
                "original_domains": original.get("original_domains"),
                
                # Pipeline results - adjust based on your actual response structure
                "is_climate_related": pipeline.get("climate_classification", {}).get("is_climate_related", False),
                "climate_confidence": pipeline.get("climate_classification", {}).get("confidence", 0.0),
                
                "category": pipeline.get("category_classification", {}).get("prediction"),
                "category_confidence": pipeline.get("category_classification", {}).get("confidence"),
                
                # Weather validation - expect many "No Location" or "Unavailable"
                "weather_flag": pipeline.get("weather_flag", "No Location"),
                "weather_consistency_score": pipeline.get("weather_validation", {}).get("validation", {}).get("consistency_score"),
                
                "sentiment_flag": pipeline.get("sentiment_flag", "Unknown"),
                "sentiment_compound": pipeline.get("sentiment_analysis", {}).get("sentiment", {}).get("compound"),
                "sentiment_positive": pipeline.get("sentiment_analysis", {}).get("sentiment", {}).get("positive"),
                "sentiment_negative": pipeline.get("sentiment_analysis", {}).get("sentiment", {}).get("negative"),
                "sentiment_neutral": pipeline.get("sentiment_analysis", {}).get("sentiment", {}).get("neutral"),
                
                # Data quality flags for your thesis analysis
                "has_valid_location": "Yes" if display_location != "No Location" else "No",
                "weather_validation_attempted": "Yes" if pipeline.get("weather_validation", {}).get("status") != "skipped" else "No",
                
                # Full response for reference
                "full_pipeline_response": json.dumps(pipeline)
            }
            processed_data.append(row_data)
        else:
            # Keep failed rows for debugging
            original = result["original_data"]
            row_data = {
                "tweet": original["tweet"],
                "location": original.get("location", "No Location"),
                "error": result.get("error", "Unknown error"),
                "status": "failed"
            }
            processed_data.append(row_data)
    
    # Save to CSV
    df = pd.DataFrame(processed_data)
    df.to_csv(output_path, index=False)
    
    # Print useful stats
    total_tweets = len(processed_data)
    valid_locations = sum(1 for row in processed_data if row.get("has_valid_location") == "Yes")
    weather_attempted = sum(1 for row in processed_data if row.get("weather_validation_attempted") == "Yes")
    
    print(f"💾 Saved results to {output_path}")
    print(f"📊 Data Quality Stats:")
    print(f"   Total tweets: {total_tweets}")
    print(f"   Valid locations: {valid_locations} ({valid_locations/total_tweets*100:.1f}%)")
    print(f"   Weather validation attempted: {weather_attempted} ({weather_attempted/total_tweets*100:.1f}%)")
    print(f"   No location/fake location: {total_tweets - valid_locations} tweets")

async def main():
    """Main processing function"""
    
    # Configuration
    INPUT_CSV = "your_training_data.csv"  # Update this path
    OUTPUT_CSV = "processed_training_data.csv"
    API_BASE_URL = "http://localhost:8000"
    
    print("🚀 Processing Training Data as Current Tweets")
    print("=" * 60)
    print("💡 Strategy: Treating training tweets as 'current' for weather validation")
    print("🌤️  Weather data will be fetched from TODAY for consistency checking")
    print("📍 Focus: Proper location handling and goofy location detection")
    print("=" * 60)
    
    processor = TrainingDataProcessor(API_BASE_URL)
    
    # Load CSV data
    tweets = processor.load_and_prepare_csv(INPUT_CSV)
    
    if not tweets:
        print("❌ No tweets to process!")
        return
    
    print(f"🎯 Processing {len(tweets)} tweets through your pipeline...")
    print(f"📡 API Base URL: {API_BASE_URL}")
    print("⚡ This will be much faster since we're using current weather data!")
    
    # Process tweets with smaller batches for reliability
    results = await processor.process_batch(tweets, batch_size=5)
    
    print("\n" + "=" * 60)
    print(f"✅ Processing completed!")
    print(f"✅ Successful: {processor.processed_count}")
    print(f"❌ Errors: {processor.error_count}")
    
    # Save results
    save_results_to_csv(results, OUTPUT_CSV)
    
    # Calculate expected distribution
    successful_results = [r for r in results if r["status"] == "success"]
    if successful_results:
        print(f"\n📊 Expected Results Distribution:")
        print(f"   🔍 This shows how your pipeline handles real-world messy data!")
        
    print(f"🎉 Done! Your training data is now fully processed with:")
    print(f"   ✅ Climate classification validation")  
    print(f"   ✅ Category classification")
    print(f"   ✅ Current weather consistency checking")
    print(f"   ✅ VADER sentiment analysis") 
    print(f"   ✅ Proper handling of goofy Filipino Twitter locations")
    print(f"\n💾 Check {OUTPUT_CSV} for your thesis-ready dataset!")

if __name__ == "__main__":
    # Run the async processing
    asyncio.run(main())