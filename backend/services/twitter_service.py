import tweepy
import pandas as pd
import os
import uuid
from datetime import datetime
from difflib import SequenceMatcher
import re
from typing import Optional, Dict, List
from fastapi import BackgroundTasks

# Twitter API credentials
consumer_key = os.getenv('TWITTER_CONSUMER_KEY', "XET9KpGU3J6nN48S6qXFCUtLV")
consumer_secret = os.getenv('TWITTER_CONSUMER_SECRET', "PBFMiBqRO0DFdXgck7JuvTcG7Rglpd7NoPBnZavD8Nho3n40wx")
access_token = os.getenv('TWITTER_ACCESS_TOKEN', "1194686821956849664-a99N52yzbqdmlolpKP9GLqNVR1kKp3")
access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET', "NgkNaeoEhUMWaRgGTFD4V4R18I79XaRlyRsHZru8WqLyB")
bearer_token = os.getenv('TWITTER_BEARER_TOKEN', "AAAAAAAAAAAAAAAAAAAAAHXk1gEAAAAAO%2BNDKFpsCUMmgMc4Eix27hH2EUs%3D84Gyu9Jxss1QQ28z315d4DDhGhppnAUTpOxjHE4U7njLEXlpsl")

# In-memory task storage
scraping_tasks: Dict[str, Dict] = {}

def clean_text_for_comparison(text: str) -> str:
    """Clean text for duplicate comparison"""
    if not text:
        return ""
    
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)
    text = ' '.join(text.split()).lower().strip()
    
    return text

def has_inappropriate_content(text: str, additional_terms: Optional[List[str]] = None) -> bool:
    """Secondary check for inappropriate content"""
    inappropriate_terms = [
        "kantot", "kantutan", "burat", "titi", "puke", "bilat",
        "tamod", "jakol", "chupa", "libog", "fubu", "kantotero",
        "sex", "porn", "nude", "naked", "horny", "pussy", "dick",
        "fuck", "shit", "bitch", "slut", "whore"
    ]
    
    if additional_terms:
        inappropriate_terms.extend(additional_terms)
    
    text_lower = text.lower()
    return any(term in text_lower for term in inappropriate_terms)

def is_duplicate(new_text: str, existing_texts: List[str], threshold: float = 0.85) -> bool:
    """Check if new text is similar to existing texts"""
    cleaned_new = clean_text_for_comparison(new_text)
    
    if len(cleaned_new) < 10:
        return False
    
    for existing_text in existing_texts:
        cleaned_existing = clean_text_for_comparison(existing_text)
        if len(cleaned_existing) < 10:
            continue
            
        similarity = SequenceMatcher(None, cleaned_new, cleaned_existing).ratio()
        if similarity >= threshold:
            return True
    
    return False

def generate_word_variations(word: str) -> List[str]:
    """Generate common Filipino word variations"""
    variations = [word]
    
    prefixes = ['ma', 'nag', 'mag', 'pag', 'ka', 'um', 'in', 'ni']
    suffixes = ['an', 'in', 'ng', 'han', 'hin', 'on']
    
    for prefix in prefixes:
        variations.append(f"{prefix}{word}")
    
    for suffix in suffixes:
        variations.append(f"{word}{suffix}")
    
    seen = set()
    unique_variations = []
    for var in variations:
        if var not in seen:
            seen.add(var)
            unique_variations.append(var)
    
    return unique_variations

def build_expanded_query(base_query: str, max_variations: int = 5) -> str:
    """Build expanded query with word variations"""
    words = base_query.split()
    expanded_terms = []
    
    for word in words:
        if len(word) >= 3:
            variations = generate_word_variations(word)[:max_variations]
            if len(variations) > 1:
                variations_str = " OR ".join(variations)
                expanded_terms.append(f"({variations_str})")
            else:
                expanded_terms.append(word)
        else:
            expanded_terms.append(word)
    
    return " ".join(expanded_terms)

def scrape_tweets_task(task_id: str, query: str, limit: int, similarity_threshold: float, use_expansion: bool):
    """Background task to scrape tweets"""
    try:
        scraping_tasks[task_id]["status"] = "in_progress"
        scraping_tasks[task_id]["progress"] = 0
        
        client = tweepy.Client(
            bearer_token=bearer_token,
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_token_secret=access_token_secret
        )

        tweets_list = []
        existing_texts = []
        seen_tweet_ids = set()
        
        exclude_terms = [
            "kantot", "kantutan", "burat", "titi", "puke", "bilat", "matambok", 
            "tamod", "jakol", "chupa", "libog", "fubu", "binabayo", "totnak", 
            "kantotero", "sex", "porn", "nude", "naked", "horny", "pussy", 
            "dick", "nigga", "nigger"
        ]
        
        exclusions = " ".join([f"-{term}" for term in exclude_terms])
        
        search_query = query
        if use_expansion:
            search_query = build_expanded_query(query)
            scraping_tasks[task_id]["expanded_query"] = search_query
        
        philippines_queries = [
            f"{search_query} (Philippines OR Manila OR Cebu OR Davao OR Quezon OR PH OR Pinas OR Pilipinas) {exclusions}",
            f"{search_query} lang:tl",
            f"{search_query} (Metro Manila OR NCR OR Luzon OR Visayas OR Mindanao) {exclusions}",
            f"{search_query} -is:retweet (Philippines OR Manila OR Cebu) {exclusions}",
        ]
        
        for i, search_query_final in enumerate(philippines_queries):
            if len(tweets_list) >= limit:
                break
            
            # Check if task was cancelled
            if scraping_tasks[task_id].get("cancelled", False):
                scraping_tasks[task_id]["status"] = "cancelled"
                return
            
            try:
                remaining_needed = limit - len(tweets_list)
                fetch_count = min(100, max(10, remaining_needed + 10))
                
                tweets = client.search_recent_tweets(
                    query=search_query_final,
                    max_results=fetch_count,
                    tweet_fields=['created_at', 'text', 'public_metrics', 'geo', 'context_annotations', 'author_id', 'lang'],
                    expansions=['geo.place_id', 'author_id'],
                    place_fields=['full_name', 'country', 'geo', 'country_code'],
                    user_fields=['location']
                )
                
                if not tweets or not tweets.data:
                    continue
                
                place_lookup = {}
                user_lookup = {}
                
                if tweets.includes:
                    if 'places' in tweets.includes:
                        for place in tweets.includes['places']:
                            place_lookup[place.id] = {
                                'name': place.full_name,
                                'country': getattr(place, 'country', None),
                                'country_code': getattr(place, 'country_code', None)
                            }
                    
                    if 'users' in tweets.includes:
                        for user in tweets.includes['users']:
                            user_lookup[user.id] = {
                                'location': getattr(user, 'location', None)
                            }

                for tweet in tweets.data:
                    if len(tweets_list) >= limit:
                        break
                    
                    if tweet.id in seen_tweet_ids:
                        continue
                    seen_tweet_ids.add(tweet.id)
                    
                    if has_inappropriate_content(tweet.text):
                        continue
                    
                    if is_duplicate(tweet.text, existing_texts, similarity_threshold):
                        continue
                    
                    location = None
                    country = None
                    country_code = None
                    user_location = None
                    
                    if tweet.geo and 'place_id' in tweet.geo:
                        place_id = tweet.geo['place_id']
                        place_info = place_lookup.get(place_id, {})
                        location = place_info.get('name', None)
                        country = place_info.get('country', None)
                        country_code = place_info.get('country_code', None)
                    
                    if tweet.author_id in user_lookup:
                        user_location = user_lookup[tweet.author_id].get('location', None)
                    
                    tweets_list.append({
                        'date': tweet.created_at.isoformat() if tweet.created_at else None,
                        'id': str(tweet.id),
                        'text': tweet.text,
                        'retweet_count': tweet.public_metrics['retweet_count'],
                        'reply_count': tweet.public_metrics['reply_count'],
                        'like_count': tweet.public_metrics['like_count'],
                        'location': location,
                        'country': country,
                        'country_code': country_code,
                        'user_location': user_location,
                        'author_id': str(tweet.author_id),
                        'language': getattr(tweet, 'lang', None),
                        'search_strategy': i + 1
                    })
                    
                    existing_texts.append(tweet.text)
                
                # Update progress
                progress = min(100, int((len(tweets_list) / limit) * 100))
                scraping_tasks[task_id]["progress"] = progress
                scraping_tasks[task_id]["tweets_collected"] = len(tweets_list)
                
            except tweepy.TooManyRequests:
                scraping_tasks[task_id]["error"] = "Rate limit reached"
                break
            except Exception as e:
                scraping_tasks[task_id]["error"] = str(e)
                continue
        
        # Save results
        scraping_tasks[task_id]["tweets"] = tweets_list
        scraping_tasks[task_id]["status"] = "completed"
        scraping_tasks[task_id]["progress"] = 100
        scraping_tasks[task_id]["completed_at"] = datetime.now().isoformat()
        
        # Calculate statistics
        if tweets_list:
            df = pd.DataFrame(tweets_list)
            scraping_tasks[task_id]["statistics"] = {
                "total_tweets": len(tweets_list),
                "total_likes": int(df['like_count'].sum()),
                "total_retweets": int(df['retweet_count'].sum()),
                "total_replies": int(df['reply_count'].sum()),
                "languages": df['language'].value_counts().to_dict() if 'language' in df else {},
                "with_location": int(df['location'].notna().sum()),
                "with_user_location": int(df['user_location'].notna().sum())
            }
        
    except Exception as e:
        scraping_tasks[task_id]["status"] = "failed"
        scraping_tasks[task_id]["error"] = str(e)
        scraping_tasks[task_id]["completed_at"] = datetime.now().isoformat()

async def scrape_tweets(
    query: str, 
    limit: int, 
    similarity_threshold: float, 
    use_expansion: bool,
    background_tasks: BackgroundTasks
) -> str:
    """Start scraping tweets in background"""
    task_id = str(uuid.uuid4())
    
    scraping_tasks[task_id] = {
        "task_id": task_id,
        "query": query,
        "limit": limit,
        "similarity_threshold": similarity_threshold,
        "use_expansion": use_expansion,
        "status": "pending",
        "progress": 0,
        "tweets_collected": 0,
        "tweets": [],
        "created_at": datetime.now().isoformat(),
        "cancelled": False
    }
    
    background_tasks.add_task(
        scrape_tweets_task,
        task_id,
        query,
        limit,
        similarity_threshold,
        use_expansion
    )
    
    return task_id

def get_scraping_status(task_id: str) -> Optional[Dict]:
    """Get the status of a scraping task"""
    return scraping_tasks.get(task_id)

def cancel_scraping(task_id: str) -> bool:
    """Cancel a scraping task"""
    if task_id in scraping_tasks:
        scraping_tasks[task_id]["cancelled"] = True
        return True
    return False

def get_scraping_history(limit: int = 10) -> List[Dict]:
    """Get recent scraping history"""
    tasks = list(scraping_tasks.values())
    tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    
    # Return summary without full tweet data
    history = []
    for task in tasks[:limit]:
        summary = {k: v for k, v in task.items() if k != "tweets"}
        history.append(summary)
    
    return history