# migrate_add_columns.py
from sqlalchemy import create_engine, inspect, text

# adjust your DB URL if needed
DATABASE_URL = "postgresql://saf:0000@localhost:5432/saf_database"

engine = create_engine(DATABASE_URL)

# Define required columns and their SQL types
required_columns = {
    "tweet_text": "TEXT",
    "location": "VARCHAR(255)",
    "length": "INTEGER",
    "is_climate_related": "BOOLEAN DEFAULT FALSE NOT NULL",
    "climate_confidence": "FLOAT",
    "climate_prediction": "INTEGER",
    "category": "VARCHAR(100)",
    "category_confidence": "FLOAT",
    "category_probabilities": "JSON",
    "weather_flag": "VARCHAR(50)",
    "weather_consistency_score": "FLOAT",
    "weather_data": "JSON",
    "sentiment_flag": "VARCHAR(20)",
    "sentiment_positive": "FLOAT",
    "sentiment_negative": "FLOAT",
    "sentiment_neutral": "FLOAT",
    "sentiment_compound": "FLOAT",
    "full_response": "JSON",
    "batch_id": "INTEGER"
}

with engine.connect() as conn:
    inspector = inspect(engine)
    existing_columns = [col["name"] for col in inspector.get_columns("tweets")]

    for col, col_type in required_columns.items():
        if col not in existing_columns:
            print(f"Adding missing column: {col}")
            conn.execute(text(f'ALTER TABLE tweets ADD COLUMN "{col}" {col_type};'))
        else:
            print(f"Column already exists: {col}")

    print("Migration complete! ✅")
