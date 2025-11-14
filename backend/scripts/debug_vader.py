# debug_vader.py - Run this to diagnose VADER sentiment analysis issues with joblib

import sys
import os
import joblib
import pandas as pd

def debug_vader_issue():
    """Debug the specific issue with 'sobrang init nanaman'"""
    
    print("=" * 80)
    print("VADER SENTIMENT ANALYSIS DEBUG - JOBLIB VERSION")
    print("=" * 80)
    
    # Test tweet
    test_tweet = "sobrang init nanaman"
    
    print(f"\n📝 Test Tweet: '{test_tweet}'")
    print(f"   Translation: 'It's extremely hot again'\n")
    
    # First, check if the joblib file exists and what it contains
    print("-" * 80)
    print("1. CHECKING JOBLIB FILE")
    print("-" * 80)
    
    joblib_path = "data/lexical_dictionary/lexical_dictionary.joblib"
    
    if not os.path.exists(joblib_path):
        print(f"❌ ERROR: File not found at {joblib_path}")
        print("\nPossible locations to check:")
        
        # Check common alternative paths
        alternatives = [
            "lexical_dictionary.joblib",
            "data/lexical_dictionary.joblib",
            "../data/lexical_dictionary/lexical_dictionary.joblib",
        ]
        
        for alt in alternatives:
            if os.path.exists(alt):
                print(f"   ✅ Found at: {alt}")
            else:
                print(f"   ❌ Not at: {alt}")
        
        print("\n⚠️ Cannot proceed without the lexicon file!")
        return
    
    print(f"✅ File exists at: {joblib_path}")
    print(f"   File size: {os.path.getsize(joblib_path) / 1024:.2f} KB")
    
    # Try to load and inspect the joblib file
    try:
        data = joblib.load(joblib_path)
        print(f"✅ Successfully loaded joblib file")
        print(f"   Data type: {type(data)}")
        
        if isinstance(data, pd.DataFrame):
            print(f"   Shape: {data.shape}")
            print(f"   Columns: {list(data.columns)}")
            print(f"\n📋 First 5 rows:")
            print(data.head().to_string(index=False))
            
            # Check for required columns
            if 'word' in data.columns and 'sentivalue' in data.columns:
                print(f"\n✅ Required columns present")
                
                # Check for Tagalog words
                tagalog_test_words = ['sobrang', 'init', 'mainit', 'nanaman']
                print(f"\n🔍 Checking for key Tagalog words:")
                for word in tagalog_test_words:
                    if word in data['word'].str.lower().values:
                        row = data[data['word'].str.lower() == word].iloc[0]
                        print(f"   ✅ '{word}': sentivalue = {row['sentivalue']}")
                    else:
                        print(f"   ❌ '{word}': NOT FOUND")
            else:
                print(f"\n❌ Missing required columns!")
                print(f"   Expected: ['word', 'sentivalue']")
                print(f"   Found: {list(data.columns)}")
        
        elif isinstance(data, dict):
            print(f"   Dictionary keys: {list(data.keys())}")
        else:
            print(f"   ⚠️ Unexpected data structure")
        
    except Exception as e:
        print(f"❌ Error loading joblib file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Now test with the actual sentiment analyzer
    print("\n" + "-" * 80)
    print("2. LOADING SENTIMENT ANALYZER")
    print("-" * 80)
    
    try:
        from services.sentiment_analysis import sentiment_analyzer
        print("✅ Sentiment analyzer imported")
    except Exception as e:
        print(f"❌ Could not import sentiment analyzer: {e}")
        return
    
    # Check lexicon loading status
    print("\n" + "-" * 80)
    print("3. LEXICON STATUS")
    print("-" * 80)
    
    stats = sentiment_analyzer.get_lexicon_stats()
    print(f"Total words in lexicon: {stats.get('total_words', 0)}")
    print(f"Custom words loaded: {stats.get('custom_words', 0)}")
    print(f"Base VADER words: {stats.get('base_vader_words', 0)}")
    print(f"Custom positive words: {stats.get('custom_positive', 0)}")
    print(f"Custom negative words: {stats.get('custom_negative', 0)}")
    print(f"Custom neutral words: {stats.get('custom_neutral', 0)}")
    print(f"Load errors: {stats.get('load_errors_count', 0)}")
    
    if stats.get('custom_words', 0) == 0:
        print("\n⚠️ WARNING: No custom words loaded!")
        print("   Possible causes:")
        print("   - Lexicon file empty or invalid")
        print("   - DataFrame columns incorrect")
        print("   - All words were skipped due to errors")
    
    # Check specific words
    print("\n" + "-" * 80)
    print("4. WORD-BY-WORD ANALYSIS")
    print("-" * 80)
    
    words_to_check = ["sobrang", "init", "nanaman", "sobra", "mainit", "napakainit"]
    
    for word in words_to_check:
        result = sentiment_analyzer.test_word(word)
        if result['found']:
            print(f"✅ '{word}': {result['sentiment']:+.4f} ({result['source']}) - {result['interpretation']}")
        else:
            print(f"❌ '{word}': NOT FOUND")
            if result.get('similar_words'):
                print(f"   Similar: {', '.join(result['similar_words'][:3])}")
    
    # Test sentiment analysis with debug
    print("\n" + "-" * 80)
    print("5. SENTIMENT ANALYSIS WITH DEBUG")
    print("-" * 80)
    
    result = sentiment_analyzer.analyze_sentiment(test_tweet, debug=True)
    
    print("\n📊 FINAL RESULT:")
    print(f"   Classification: {result.classification.upper()}")
    print(f"   Compound: {result.compound:+.4f}")
    print(f"   Positive: {result.positive:.3f}")
    print(f"   Negative: {result.negative:.3f}")
    print(f"   Neutral: {result.neutral:.3f}")
    
    # Diagnosis
    print("\n" + "=" * 80)
    print("6. DIAGNOSIS")
    print("=" * 80)
    
    if result.compound == 0.0:
        print("🔴 PROBLEM CONFIRMED: Compound score is 0.0")
        print("\nRoot cause analysis:")
        
        if stats.get('custom_words', 0) == 0:
            print("\n❌ ROOT CAUSE: No custom lexicon loaded")
            print("   → The joblib file exists but no words were loaded")
            print("   → Check DataFrame structure and column names")
        else:
            # Check if words exist in lexicon
            processed = sentiment_analyzer.preprocess_text(test_tweet)
            words_in_text = processed.split()
            words_found = []
            for word in words_in_text:
                if word in sentiment_analyzer.lexicon:
                    words_found.append((word, sentiment_analyzer.lexicon[word]))
            
            if not words_found:
                print("\n❌ ROOT CAUSE: Words not in lexicon")
                print(f"   → Text: '{test_tweet}'")
                print(f"   → Processed: '{processed}'")
                print(f"   → Words: {words_in_text}")
                print(f"   → None of these words exist in the lexicon")
            else:
                print("\n⚠️ MYSTERY: Words found but score is still 0.0")
                print(f"   → Found words: {words_found}")
                print(f"   → This might be a VADER processing issue")
    else:
        print("✅ Sentiment analysis working correctly!")
        print(f"   Compound score: {result.compound:+.4f}")
    
    # Action items
    print("\n" + "=" * 80)
    print("7. RECOMMENDED ACTIONS")
    print("=" * 80)
    
    if stats.get('custom_words', 0) == 0:
        print("\n📝 ACTION REQUIRED: Regenerate lexical dictionary")
        print("   1. Check your lexical_dictionary_manager")
        print("   2. Make sure it saves with correct column names: 'word', 'sentivalue', 'weight'")
        print("   3. Regenerate and save the lexicon")
        print("   4. Restart your FastAPI server")
    elif result.compound == 0.0:
        print("\n📝 ACTION REQUIRED: Add missing words to lexicon")
        print("   Missing words that need sentiment values:")
        for word in ["sobrang", "init", "nanaman"]:
            test = sentiment_analyzer.test_word(word)
            if not test['found']:
                print(f"   - '{word}' (should be negative for 'hot/excessive')")
    else:
        print("\n✅ Everything looks good! The sentiment analysis is working.")
    
    # Quick test fix
    print("\n" + "=" * 80)
    print("8. QUICK TEST FIX (TEMPORARY)")
    print("=" * 80)
    
    print("\nYou can manually add words for testing:")
    print("```python")
    print("from services.sentiment_analysis import sentiment_analyzer")
    print("")
    print("# Add Tagalog heat-related words")
    print("sentiment_analyzer.lexicon['sobrang'] = -2.5  # extremely (negative context)")
    print("sentiment_analyzer.lexicon['init'] = -2.0     # hot (uncomfortable)")
    print("sentiment_analyzer.lexicon['mainit'] = -2.0   # hot")
    print("sentiment_analyzer.lexicon['nanaman'] = -0.5  # again (frustration)")
    print("")
    print("# Test again")
    print("result = sentiment_analyzer.analyze_sentiment('sobrang init nanaman', debug=True)")
    print("print(f'Compound: {result.compound}')  # Should be negative now")
    print("```")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    debug_vader_issue()