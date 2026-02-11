import pandas as pd
import glob
import os
import re
from sklearn.feature_extraction.text import CountVectorizer

# --- CONFIGURATION ---
TARGET_FOLDER = 'ritc_data_folder'  # Change this to your folder path
PRICE_CHANGE_THRESHOLD = 0.05       # Minimum price move ($) to count as a signal
LOOKAHEAD_TICKS = 15                # How many ticks after news to check price

def robust_read_ritc_csv(filename):
    """
    Reads RITC CSVs that may have inconsistent column counts (news vs price rows).
    """
    # Define all possible columns based on RITC format
    col_names = [
        'ticker', 'last_price', 'bid_price', 'ask_price', 'BUY_ANON', 'SELL_ANON', 
        'tick', 'timestamp', 'news_news_id', 'news_period', 'news_tick', 
        'news_ticker', 'news_headline', 'news_body'
    ]
    try:
        # Use python engine and skip header to handle ragged lines
        df = pd.read_csv(filename, names=col_names, header=None, skiprows=1, engine='python')
        
        # Clean numeric columns
        df['tick'] = pd.to_numeric(df['tick'], errors='coerce')
        df['last_price'] = pd.to_numeric(df['last_price'], errors='coerce')
        df['news_tick'] = pd.to_numeric(df['news_tick'], errors='coerce')
        
        # Drop rows with no useful data (keep news or valid price ticks)
        df = df.dropna(subset=['tick'])
        return df
    except Exception as e:
        print(f"Skipping {filename}: {e}")
        return pd.DataFrame()

def parse_tickers(ticker_str):
    """Extracts tickers from strings like 'D1-TGX/PHR'"""
    if pd.isna(ticker_str): return []
    clean_str = re.sub(r'D\d+-', '', str(ticker_str)) # Remove 'D1-'
    tickers = re.split(r'[/]', clean_str)             # Split 'TGX/PHR'
    return [t.strip() for t in tickers]

def build_sentiment_library(folder_path):
    all_labeled_news = []
    
    # Get all CSV files
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    print(f"Scanning {len(files)} files in '{folder_path}'...")
    
    for filename in files:
        df = robust_read_ritc_csv(filename)
        if df.empty: continue
            
        # 1. Pivot Price Data (Index=Tick, Columns=Ticker)
        # We need a clean time series to look up prices
        price_df = df.pivot_table(index='tick', columns='ticker', values='last_price')
        price_df = price_df.ffill() # Forward fill missing prices
        
        # 2. Extract News Rows
        news_df = df[df['news_headline'].notna()][['news_tick', 'news_ticker', 'news_headline']]
        
        # 3. Analyze Each News Item
        for _, row in news_df.iterrows():
            try:
                tickers = parse_tickers(row['news_ticker'])
                if not tickers: continue
                
                # Time window
                start_tick = int(row['news_tick'])
                end_tick = start_tick + LOOKAHEAD_TICKS
                
                # Find valid ticks in our data (nearest available)
                valid_starts = price_df.index[price_df.index >= start_tick]
                valid_ends = price_df.index[price_df.index <= end_tick]
                
                if len(valid_starts) == 0 or len(valid_ends) == 0: continue
                
                t0 = valid_starts[0]
                t1 = valid_ends[-1] # Use the latest available tick within window
                
                # Find which ticker moved the most
                max_move = 0
                sentiment = 'Neutral'
                
                for t in tickers:
                    if t in price_df.columns:
                        p_start = price_df.loc[t0, t]
                        p_end = price_df.loc[t1, t]
                        change = p_end - p_start
                        
                        if abs(change) > abs(max_move):
                            max_move = change
                
                # Label Logic
                if max_move > PRICE_CHANGE_THRESHOLD:
                    sentiment = 'Bullish'
                elif max_move < -PRICE_CHANGE_THRESHOLD:
                    sentiment = 'Bearish'
                
                if sentiment != 'Neutral':
                    all_labeled_news.append({
                        'Headline': row['news_headline'],
                        'Sentiment': sentiment,
                        'Impact': max_move
                    })
                    
            except Exception as e:
                continue

    return pd.DataFrame(all_labeled_news)

def extract_top_phrases(df, sentiment, n=2):
    """Uses NLP to find top 1-word or 2-word phrases"""
    subset = df[df['Sentiment'] == sentiment]
    if subset.empty: return []
    
    # CountVectorizer finds words and phrases (ngrams)
    vec = CountVectorizer(stop_words='english', ngram_range=(1, n))
    X = vec.fit_transform(subset['Headline'])
    
    # Sum counts
    counts = X.sum(axis=0)
    words = [(word, counts[0, idx]) for word, idx in vec.vocabulary_.items()]
    
    # Sort by frequency
    return sorted(words, key=lambda x: x[1], reverse=True)[:15]

# --- EXECUTION ---
if __name__ == "__main__":
    # 1. Build the Dataset
    # Replace with your actual folder path
    # For this demo, I assume the files are in the current directory
    labeled_df = build_sentiment_library('.') 
    
    if not labeled_df.empty:
        print(f"\nSuccessfully analyzed {len(labeled_df)} news events.")
        
        # 2. Extract & Print Insights
        print("\n--- TOP BULLISH TRIGGERS ---")
        for word, count in extract_top_phrases(labeled_df, 'Bullish'):
            print(f"{word}: {count}")
            
        print("\n--- TOP BEARISH TRIGGERS ---")
        for word, count in extract_top_phrases(labeled_df, 'Bearish'):
            print(f"{word}: {count}")
            
        # 3. Save for future training
        labeled_df.to_csv('ritc_training_data_labeled.csv', index=False)
        print("\nLabeled data saved to 'ritc_training_data_labeled.csv'")
    else:
        print("No significant news events found in the CSV files.")