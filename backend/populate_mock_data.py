# populate_mock_data.py
import random
from datetime import datetime, timedelta
from database.models import SessionLocal, Tweet, init_db

# Initialize database
init_db()

# Climate categories with realistic probabilities
CLIMATE_CATEGORIES = [
    "Extreme Heat / Heatwaves",
    "Flooding and Extreme Precipitation",
    "Storms, Typhoons, and Wind Events",
    "Drought and Water Scarcity",
    "Environmental Degradation and Land Use",
    "Air Pollution and Emissions",
    "Cold Weather / Temperature Drops",
    "Sea Level Rise / Coastal Hazards",
    "Geological Events"
]

# Sample tweets for each category
SAMPLE_TWEETS = {
    "Extreme Heat / Heatwaves": [
        "Grabe yung init ngayon sobrang tagaktak ng pawis",
        "ang init naman dito sa cavite",
        "Sobrang init today! Climate change is real",
        "40 degrees feels like, di na matiis",
        "Nakakapaso ang init ng panahon ngayon"
    ],
    "Flooding and Extreme Precipitation": [
        "Baha na naman dito sa amin",
        "Heavy rainfall causes flood in our area",
        "Umuulan ng malakas, baha everywhere",
        "Flash flood warning issued today",
        "Grabe ang lakas ng ulan, baha agad"
    ],
    "Storms, Typhoons, and Wind Events": [
        "Malakas ang hangin dahil sa bagyo",
        "Typhoon approaching, stay safe everyone",
        "Strong winds due to tropical storm",
        "Hanging Amihan brings cool weather",
        "Napunit yung bubong dahil sa malakas na hangin"
    ],
    "Drought and Water Scarcity": [
        "Water shortage sa aming barangay",
        "Walang tubig for 3 days na",
        "Drought conditions worsen this summer",
        "Dry season is getting worse every year",
        "Water crisis in our town"
    ],
    "Environmental Degradation and Land Use": [
        "Ang daming basura sa ilog",
        "Deforestation is destroying our mountains",
        "Illegal logging continues in our area",
        "Plastic pollution everywhere",
        "Our environment is getting worse"
    ],
    "Air Pollution and Emissions": [
        "Ang dumi ng hangin dito sa Manila",
        "Smog everywhere, can't breathe properly",
        "Air quality index is very unhealthy",
        "Traffic pollution is getting worse",
        "Nakakapanghina ng loob ang air pollution"
    ],
    "Cold Weather / Temperature Drops": [
        "Sobrang lamig ngayong umaga",
        "Cold weather alert issued",
        "Freezing temperature this morning",
        "Nagsuot na kami ng jacket, ang lamig",
        "Ber months na talaga, malamig na"
    ],
    "Sea Level Rise / Coastal Hazards": [
        "High tide flooding in coastal areas",
        "Sea level is rising, coastal erosion visible",
        "Storm surge warning issued",
        "Our shoreline is disappearing",
        "Coastal flooding during high tide"
    ],
    "Geological Events": [
        "Lindol ngayong umaga",
        "Earthquake felt in our area",
        "Volcanic activity detected",
        "Ground shaking due to tremor",
        "Magnitude 4.5 earthquake recorded"
    ]
}

# Locations in Cavite
LOCATIONS = ["Indang", "Imus", "Trece Martires"]

# Weather conditions
WEATHER_CONDITIONS = [
    "Partly cloudy",
    "Sunny",
    "Overcast",
    "Light rain",
    "Heavy rain",
    "Cloudy"
]

def generate_sentiment():
    """Generate realistic sentiment scores"""
    sentiment_type = random.choices(
        ["positive", "negative", "neutral"],
        weights=[0.30, 0.45, 0.25]  # 30% positive, 45% negative, 25% neutral
    )[0]
    
    if sentiment_type == "positive":
        compound = random.uniform(0.05, 0.95)
        positive = random.uniform(0.4, 0.9)
        negative = random.uniform(0.0, 0.2)
        neutral = 1.0 - positive - negative
    elif sentiment_type == "negative":
        compound = random.uniform(-0.95, -0.05)
        negative = random.uniform(0.4, 0.9)
        positive = random.uniform(0.0, 0.2)
        neutral = 1.0 - positive - negative
    else:  # neutral
        compound = random.uniform(-0.05, 0.05)
        neutral = random.uniform(0.5, 0.8)
        positive = random.uniform(0.1, 0.3)
        negative = 1.0 - positive - neutral
    
    return {
        "classification": sentiment_type,
        "compound": round(compound, 3),
        "positive": round(positive, 3),
        "negative": round(negative, 3),
        "neutral": round(neutral, 3)
    }

def generate_mock_tweets(num_tweets=150, days_back=60):
    """Generate mock tweets with realistic data"""
    db = SessionLocal()
    tweets_created = 0
    
    try:
        # Generate tweets with varied distribution across time
        for i in range(num_tweets):
            # Random date within the last 'days_back' days
            days_ago = random.randint(0, days_back)
            tweet_date = datetime.now() - timedelta(days=days_ago)
            
            # 85% climate-related, 15% non-climate
            is_climate = random.random() < 0.85
            
            if is_climate:
                # Select category with weighted probabilities
                # Make some categories more common (Heat, Flooding, Storms)
                category_weights = [
                    0.25,  # Extreme Heat (most common in PH)
                    0.20,  # Flooding
                    0.18,  # Storms/Typhoons
                    0.10,  # Drought
                    0.08,  # Environmental Degradation
                    0.07,  # Air Pollution
                    0.05,  # Cold Weather
                    0.04,  # Sea Level Rise
                    0.03   # Geological Events
                ]
                category = random.choices(CLIMATE_CATEGORIES, weights=category_weights)[0]
                tweet_text = random.choice(SAMPLE_TWEETS[category])
                
                # Generate category probabilities
                probs = {}
                remaining = 1.0 - random.uniform(0.35, 0.70)  # Main category gets 35-70%
                for cat in CLIMATE_CATEGORIES:
                    if cat == category:
                        probs[cat] = 1.0 - remaining
                    else:
                        probs[cat] = random.uniform(0, remaining / len(CLIMATE_CATEGORIES))
                
                # Normalize probabilities
                total = sum(probs.values())
                probs = {k: round(v / total, 4) for k, v in probs.items()}
                
                category_confidence = probs[category]
            else:
                category = None
                category_confidence = None
                probs = None
                tweet_text = random.choice([
                    "Kumain ako ng masarap today",
                    "Traffic as usual",
                    "Good morning everyone!",
                    "Meeting later at 3pm",
                    "Nakakapagod ang work today"
                ])
            
            location = random.choice(LOCATIONS)
            sentiment = generate_sentiment()
            
            # Weather consistency (80% consistent, 15% inconsistent, 5% unknown)
            weather_flag = random.choices(
                ["Consistent", "Inconsistent", "Unknown"],
                weights=[0.80, 0.15, 0.05]
            )[0]
            
            # Create tweet record
            tweet = Tweet(
                tweet_text=tweet_text,
                location=location,
                length=len(tweet_text),
                
                # Climate classification
                is_climate_related=is_climate,
                climate_confidence=round(random.uniform(0.65, 0.95), 4) if is_climate else round(random.uniform(0.10, 0.35), 4),
                climate_prediction=1 if is_climate else 0,
                
                # Category classification
                category=category,
                category_confidence=category_confidence,
                category_probabilities=probs,
                
                # Weather validation
                weather_flag=weather_flag,
                weather_consistency_score=1 if weather_flag == "Consistent" else 0,
                weather_data={
                    "temperature_c": round(random.uniform(25, 35), 1),
                    "condition": random.choice(WEATHER_CONDITIONS),
                    "humidity": random.randint(50, 90)
                },
                
                # Sentiment
                sentiment_flag=sentiment["classification"].capitalize(),
                sentiment_compound=sentiment["compound"],
                sentiment_positive=sentiment["positive"],
                sentiment_negative=sentiment["negative"],
                sentiment_neutral=sentiment["neutral"],
                
                # Metadata
                analyzed_at=tweet_date
            )
            
            db.add(tweet)
            tweets_created += 1
            
            # Commit in batches
            if tweets_created % 50 == 0:
                db.commit()
                print(f"Created {tweets_created} tweets...")
        
        # Final commit
        db.commit()
        print(f"\n✅ Successfully created {tweets_created} mock tweets!")
        
        # Print summary statistics
        climate_count = db.query(Tweet).filter(Tweet.is_climate_related == True).count()
        print(f"\n📊 Summary:")
        print(f"   Total tweets: {tweets_created}")
        print(f"   Climate-related: {climate_count} ({climate_count/tweets_created*100:.1f}%)")
        print(f"   Non-climate: {tweets_created - climate_count}")
        print(f"   Date range: Last {days_back} days")
        
        # Category distribution
        print(f"\n🏷️ Category Distribution:")
        for cat in CLIMATE_CATEGORIES:
            count = db.query(Tweet).filter(Tweet.category == cat).count()
            if count > 0:
                print(f"   {cat}: {count} tweets")
        
        # Sentiment distribution
        pos = db.query(Tweet).filter(Tweet.sentiment_flag == "Positive").count()
        neg = db.query(Tweet).filter(Tweet.sentiment_flag == "Negative").count()
        neu = db.query(Tweet).filter(Tweet.sentiment_flag == "Neutral").count()
        print(f"\n😊 Sentiment Distribution:")
        print(f"   Positive: {pos} ({pos/tweets_created*100:.1f}%)")
        print(f"   Negative: {neg} ({neg/tweets_created*100:.1f}%)")
        print(f"   Neutral: {neu} ({neu/tweets_created*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("🚀 Starting mock data population...\n")
    
    # Generate 150 tweets over the last 60 days
    generate_mock_tweets(num_tweets=150, days_back=60)
    
    print("\n✨ Done! Your database is now populated with mock data.")
    print("   You can now test your observations page with dynamic charts!")