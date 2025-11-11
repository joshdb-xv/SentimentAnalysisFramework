import tweepy
import pandas as pd
import os
from datetime import datetime
from difflib import SequenceMatcher
import re

# Twitter API credentials
consumer_key = os.getenv('TWITTER_CONSUMER_KEY', "XET9KpGU3J6nN48S6qXFCUtLV")
consumer_secret = os.getenv('TWITTER_CONSUMER_SECRET', "PBFMiBqRO0DFdXgck7JuvTcG7Rglpd7NoPBnZavD8Nho3n40wx")
access_token = os.getenv('TWITTER_ACCESS_TOKEN', "1194686821956849664-a99N52yzbqdmlolpKP9GLqNVR1kKp3")
access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET', "NgkNaeoEhUMWaRgGTFD4V4R18I79XaRlyRsHZru8WqLyB")
bearer_token = os.getenv('TWITTER_BEARER_TOKEN', "AAAAAAAAAAAAAAAAAAAAAHXk1gEAAAAAO%2BNDKFpsCUMmgMc4Eix27hH2EUs%3D84Gyu9Jxss1QQ28z315d4DDhGhppnAUTpOxjHE4U7njLEXlpsl")

def clean_text_for_comparison(text):
    """Clean text for duplicate comparison"""
    if not text:
        return ""
    
    # Remove URLs, mentions, hashtags, extra whitespace
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)
    text = ' '.join(text.split()).lower().strip()
    
    return text

def has_inappropriate_content(text, additional_terms=None):
    """
    Secondary check for inappropriate content that might slip through API filtering
    """
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

def is_duplicate(new_text, existing_texts, threshold=0.85):
    """Check if new text is similar to existing texts"""
    cleaned_new = clean_text_for_comparison(new_text)
    
    # Skip very short texts
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

def generate_word_variations(word):
    """
    Generate common Filipino word variations for better search coverage
    """
    variations = [word]
    
    # Common Filipino prefixes
    prefixes = ['ma', 'nag', 'mag', 'pag', 'ka', 'um', 'in', 'ni']
    
    # Common Filipino suffixes  
    suffixes = ['an', 'in', 'ng', 'han', 'hin', 'on']
    
    # Add prefix variations
    for prefix in prefixes:
        variations.append(f"{prefix}{word}")
    
    # Add suffix variations
    for suffix in suffixes:
        variations.append(f"{word}{suffix}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for var in variations:
        if var not in seen:
            seen.add(var)
            unique_variations.append(var)
    
    return unique_variations

def build_expanded_query(base_query, max_variations=5):
    """
    Build expanded query with word variations
    """
    words = base_query.split()
    expanded_terms = []
    
    for word in words:
        if len(word) >= 3:  # Only expand meaningful words
            variations = generate_word_variations(word)[:max_variations]
            if len(variations) > 1:
                # Create OR group for this word's variations
                variations_str = " OR ".join(variations)
                expanded_terms.append(f"({variations_str})")
            else:
                expanded_terms.append(word)
        else:
            expanded_terms.append(word)
    
    return " ".join(expanded_terms)

def scrape_tweets_philippines_efficient(query, limit=50, similarity_threshold=0.85, use_expansion=True):
    """
    Efficient Philippines tweet scraper that minimizes API usage with content filtering
    """
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
    
    # Common inappropriate/sexual Filipino terms to exclude
    exclude_terms = [
        "kantot", "kantutan", "burat", "titi", "puke", "bilat", "matambok", "tamod", "jakol", "chupa", "libog", "fubu", "binabayo", "totnak", "kantotero", "sex", "porn", "nude", "naked", "horny", "pussy", "dick", "nigga", "nigger"
    ]
    
    # Build exclusion string for Twitter query
    exclusions = " ".join([f"-{term}" for term in exclude_terms])
    
    # Expand query if requested
    search_query = query
    if use_expansion:
        expanded_query = build_expanded_query(query)
        print(f"Expanded query: {expanded_query}")
        search_query = expanded_query
    
    # Philippines-specific search queries that work with Twitter API v2
    philippines_queries = [
        f"{search_query} (Philippines OR Manila OR Cebu OR Davao OR Quezon OR PH OR Pinas OR Pilipinas) {exclusions}",
        f"{search_query} lang:tl",  # Tagalog language
        f"{search_query} (Metro Manila OR NCR OR Luzon OR Visayas OR Mindanao) {exclusions}",
        f"{search_query} -is:retweet (Philippines OR Manila OR Cebu) {exclusions}",  # Exclude retweets for less duplicates
    ]
    
    print(f"Trying {len(philippines_queries)} Philippines-specific search strategies...")
    
    for i, search_query_final in enumerate(philippines_queries):
        if len(tweets_list) >= limit:
            break
            
        print(f"Strategy {i+1}: {search_query_final[:100]}...")  # Truncate for display
        
        try:
            # Calculate how many more tweets we need
            remaining_needed = limit - len(tweets_list)
            # Fetch a bit more to account for duplicates, but not too much
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
                print(f"  No tweets found with this strategy")
                continue
                
            print(f"  Fetched {len(tweets.data)} tweets, processing...")
            
            # Build lookup dictionaries
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

            new_tweets_added = 0
            duplicates_skipped = 0
            inappropriate_skipped = 0
            
            for tweet in tweets.data:
                # Skip if we already have enough tweets
                if len(tweets_list) >= limit:
                    break
                    
                # Skip if we've seen this tweet before
                if tweet.id in seen_tweet_ids:
                    continue
                seen_tweet_ids.add(tweet.id)
                
                # Secondary check for inappropriate content
                if has_inappropriate_content(tweet.text):
                    inappropriate_skipped += 1
                    continue
                
                # Check for duplicates
                if is_duplicate(tweet.text, existing_texts, similarity_threshold):
                    duplicates_skipped += 1
                    continue
                
                # Get location info
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
                
                # Add the tweet (since we're already filtering by Philippines-specific queries)
                tweets_list.append({
                    'date': tweet.created_at,
                    'id': tweet.id,
                    'text': tweet.text,
                    'retweet_count': tweet.public_metrics['retweet_count'],
                    'reply_count': tweet.public_metrics['reply_count'],
                    'like_count': tweet.public_metrics['like_count'],
                    'location': location,
                    'country': country,
                    'country_code': country_code,
                    'user_location': user_location,
                    'author_id': tweet.author_id,
                    'language': getattr(tweet, 'lang', None),
                    'search_strategy': i + 1
                })
                
                existing_texts.append(tweet.text)
                new_tweets_added += 1
            
            print(f"  Added {new_tweets_added} unique tweets, skipped {duplicates_skipped} duplicates, {inappropriate_skipped} inappropriate")
            
            # If we got good results from this strategy, we might not need to try others
            if new_tweets_added >= remaining_needed * 0.5:  # If we got at least 50% of what we need
                print(f"  Good results from this strategy, continuing...")
            
        except tweepy.TooManyRequests:
            print(f"  Rate limit reached. Got {len(tweets_list)} tweets so far.")
            break
        except tweepy.BadRequest as e:
            print(f"  Query failed: {e}")
            continue
        except Exception as e:
            print(f"  Error: {str(e)}")
            continue

    df = pd.DataFrame(tweets_list)
    print(f"\nFinal: {len(df)} unique tweets collected")
    return df.head(limit)

def get_unique_filename(base_filename):
    """Generate unique filename"""
    if not os.path.exists(base_filename):
        return base_filename
    
    name, ext = os.path.splitext(base_filename)
    counter = 1
    
    while True:
        new_filename = f"{name} ({counter}){ext}"
        if not os.path.exists(new_filename):
            return new_filename
        counter += 1

def get_user_input():
    """Get user input"""
    print("=== TWITTER SCRAPER ===")
    print("Searches with Philippines-specific keywords while filtering inappropriate content")
    print()

    while True:
        query = input("Enter search keyword or phrase: ").strip()
        if query:
            break
        print("Please enter a valid search term.")

    while True:
        try:
            limit_input = input("Enter number of tweets (default: 10, max: 1000): ").strip()
            if not limit_input:
                limit = 10
                break
            limit = int(limit_input)
            if 10 <= limit <= 1000:
                break
            else:
                print("Please enter a number between 10 and 1000.")
        except ValueError:
            print("Please enter a valid number.")
    
    while True:
        try:
            threshold_input = input("Duplicate threshold (0.8-0.95, default: 0.85): ").strip()
            if not threshold_input:
                threshold = 0.85
                break
            threshold = float(threshold_input)
            if 0.8 <= threshold <= 0.95:
                break
            else:
                print("Please enter a number between 0.8 and 0.95.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Ask about word expansion
    while True:
        expand_input = input("Expand search with word variations? (y/n, default: y): ").strip().lower()
        if expand_input in ['', 'y', 'yes']:
            use_expansion = True
            break
        elif expand_input in ['n', 'no']:
            use_expansion = False
            break
        else:
            print("Please enter 'y' for yes or 'n' for no.")

    return query, limit, threshold, use_expansion

if __name__ == "__main__":
    try:
        query, limit, threshold, use_expansion = get_user_input()

        print(f"\nSearching for: '{query}' with Philippines context")
        print(f"Target: {limit} unique tweets")
        print(f"Duplicate threshold: {threshold}")
        print(f"Word expansion: {'Enabled' if use_expansion else 'Disabled'}")
        print("=" * 50)

        tweets_df = scrape_tweets_philippines_efficient(query, limit, threshold, use_expansion)

        if len(tweets_df) > 0:
            base_filename = f"tweets_{query.replace(' ', '_').replace('#', '').replace('@', '')}.csv"
            filename = get_unique_filename(base_filename)
            tweets_df.to_csv(filename, index=False)

            print(f"\n✅ SUCCESS! Collected {len(tweets_df)} tweets")
            print(f"📁 Saved to: {filename}")

            print(f"\n📊 SUMMARY:")
            print(f"• Unique tweets: {len(tweets_df)}")
            print(f"• Total likes: {tweets_df['like_count'].sum():,}")
            print(f"• Total retweets: {tweets_df['retweet_count'].sum():,}")
            print(f"• Total replies: {tweets_df['reply_count'].sum():,}")
            
            # Language breakdown
            lang_counts = tweets_df['language'].value_counts()
            if not lang_counts.empty:
                print(f"• Top languages: {dict(lang_counts.head(3))}")
            
            # Strategy breakdown
            strategy_counts = tweets_df['search_strategy'].value_counts()
            print(f"• Best search strategies: {dict(strategy_counts.head(3))}")
            
            # Location info
            with_location = tweets_df['location'].notna().sum()
            with_user_location = tweets_df['user_location'].notna().sum()
            print(f"• Tweets with location: {with_location}")
            print(f"• Users with location: {with_user_location}")
            
            # Sample first few tweets
            print(f"\n📝 SAMPLE TWEETS:")
            for i, row in tweets_df.head(3).iterrows():
                print(f"  {i+1}. {row['text'][:100]}...")
                
        else:
            print("\n❌ No tweets found matching your criteria.")
            print("Try:")
            print("• A more general search term")
            print("• Different spelling or related terms")
            print("• Lower duplicate threshold")

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Check your API credentials and internet connection.")