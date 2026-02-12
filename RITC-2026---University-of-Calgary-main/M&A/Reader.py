import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# CONFIG
# =========================
DATA_DIR = Path(r"C:\Users\rylan\Desktop\M&A\M&A_CSV")
FILE_GLOB = "market_data_*.csv"

TICKS_FORWARD = 20  # Look 20 ticks ahead after news


def extract_tickers_from_news_ticker(news_ticker_str):
    """
    Extract individual tickers from news_ticker field.
    Examples: 
      "D1-TGX/PHR" -> ["TGX", "PHR"]
      "D1-TGXPHR" -> ["TGX", "PHR"]
      "D4-FSR/ATB" -> ["FSR", "ATB"]
    """
    if pd.isna(news_ticker_str):
        return []
    
    s = str(news_ticker_str).strip()
    
    # Skip if it's "ALL" or doesn't contain a dash
    if s.upper().startswith("ALL") or "-" not in s:
        return []
    
    # Split by dash: "D1-TGX/PHR"
    parts = s.split("-", 1)
    if len(parts) < 2:
        return []
    
    ticker_part = parts[1].strip()
    
    # Case 1: Contains slash like "TGX/PHR"
    if "/" in ticker_part:
        tickers = [t.strip() for t in ticker_part.split("/") if t.strip()]
    else:
        # Case 2: No slash, like "TGXPHR" - split every 3 chars
        tickers = [ticker_part[i:i+3] for i in range(0, len(ticker_part), 3)]
        tickers = [t for t in tickers if len(t) == 3]  # Only valid 3-letter tickers
    
    return tickers


def process_csv(csv_path, all_ticker_impacts):
    """Process one CSV file and collect individual ticker impacts."""
    print(f"Processing {csv_path.name}...")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  ERROR reading {csv_path.name}: {e}")
        return
    
    # Check required columns
    required = {"ticker", "last_price", "tick", "news_news_id", "news_tick", 
                "news_ticker", "news_headline", "news_body"}
    if not required.issubset(df.columns):
        print(f"  SKIP: Missing required columns")
        return
    
    # Convert to numeric
    df["tick"] = pd.to_numeric(df["tick"], errors="coerce")
    df["last_price"] = pd.to_numeric(df["last_price"], errors="coerce")
    df["news_tick"] = pd.to_numeric(df["news_tick"], errors="coerce")
    df["news_news_id"] = pd.to_numeric(df["news_news_id"], errors="coerce")
    
    # Get unique news events
    news_df = df[df["news_news_id"].notna()][
        ["news_news_id", "news_tick", "news_ticker", "news_headline", "news_body"]
    ].drop_duplicates().copy()
    
    if news_df.empty:
        print(f"  No news events found")
        return
    
    print(f"  Found {len(news_df)} news events")
    
    # Build price lookup: (ticker, tick) -> price
    price_lookup = df.set_index(["ticker", "tick"])["last_price"].to_dict()
    max_tick = df["tick"].max()
    
    events_processed = 0
    
    for _, news_row in news_df.iterrows():
        news_tick = news_row["news_tick"]
        news_id = news_row["news_news_id"]
        
        if pd.isna(news_tick):
            continue
        news_tick = int(news_tick)
        
        # Extract affected tickers from news_ticker column
        affected_tickers = extract_tickers_from_news_ticker(news_row["news_ticker"])
        if not affected_tickers:
            continue
        
        # Get full news text
        headline = str(news_row["news_headline"]) if pd.notna(news_row["news_headline"]) else ""
        body = str(news_row["news_body"]) if pd.notna(news_row["news_body"]) else ""
        deal = str(news_row["news_ticker"]) if pd.notna(news_row["news_ticker"]) else ""
        
        # Process EACH ticker individually
        for ticker in affected_tickers:
            # Find price at news_tick (baseline)
            baseline_price = None
            baseline_tick_used = None
            for t in range(news_tick, max(0, news_tick - 5), -1):
                baseline_price = price_lookup.get((ticker, t))
                if baseline_price is not None:
                    baseline_tick_used = t
                    break
            
            if baseline_price is None or baseline_price <= 0:
                continue
            
            # Find price 20 ticks later (future)
            target_tick = min(news_tick + TICKS_FORWARD, max_tick)
            future_price = None
            future_tick_used = None
            for t in range(target_tick, news_tick, -1):
                future_price = price_lookup.get((ticker, t))
                if future_price is not None:
                    future_tick_used = t
                    break
            
            if future_price is None or future_price <= 0:
                continue
            
            # Calculate metrics
            price_change = future_price - baseline_price
            pct_change = price_change / baseline_price
            ticks_elapsed = future_tick_used - baseline_tick_used
            
            # Store individual ticker impact
            all_ticker_impacts.append({
                "file": csv_path.name,
                "news_id": int(news_id),
                "news_tick": news_tick,
                "deal": deal,
                "ticker": ticker,
                "baseline_tick": baseline_tick_used,
                "baseline_price": round(baseline_price, 2),
                "future_tick": future_tick_used,
                "future_price": round(future_price, 2),
                "price_change": round(price_change, 2),
                "pct_change": pct_change,
                "ticks_elapsed": ticks_elapsed,
                "headline": headline,
                "body": body
            })
        
        events_processed += 1
    
    print(f"  Processed {events_processed} events")


def main():
    csv_files = sorted(DATA_DIR.glob(FILE_GLOB))
    
    if not csv_files:
        print(f"No CSV files found matching {FILE_GLOB} in {DATA_DIR}")
        return
    
    print(f"Found {len(csv_files)} CSV files\n")
    
    all_ticker_impacts = []
    
    for csv_path in csv_files:
        process_csv(csv_path, all_ticker_impacts)
    
    if not all_ticker_impacts:
        print("\nNo ticker impacts collected")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_ticker_impacts)
    
    # Add absolute change column for ranking
    results_df["abs_pct_change"] = results_df["pct_change"].abs()
    
    # Sort by percentage change (descending = most bullish first)
    results_sorted = results_df.sort_values("pct_change", ascending=False)
    
    # Save full results
    output_path = DATA_DIR / "ticker_by_ticker_impacts.csv"
    results_sorted.to_csv(output_path, index=False)
    
    print(f"\n{'='*100}")
    print(f"RESULTS SAVED TO: {output_path}")
    print(f"Total ticker-event pairs analyzed: {len(results_df)}")
    print(f"{'='*100}\n")
    
    # Display top gainers
    print("="*100)
    print("TOP 30 BIGGEST GAINERS (Individual Ticker + News Event)")
    print("="*100)
    top_gainers = results_sorted.head(30)
    for idx, row in top_gainers.iterrows():
        print(f"\n{row['ticker']} | {row['pct_change']:+.2%} ({row['price_change']:+.2f}) | Deal: {row['deal']}")
        print(f"  Price: ${row['baseline_price']} → ${row['future_price']} over {row['ticks_elapsed']} ticks")
        print(f"  News: {row['headline'][:100]}")
    
    print("\n" + "="*100)
    print("TOP 30 BIGGEST LOSERS (Individual Ticker + News Event)")
    print("="*100)
    top_losers = results_sorted.tail(30).iloc[::-1]  # Reverse to show worst first
    for idx, row in top_losers.iterrows():
        print(f"\n{row['ticker']} | {row['pct_change']:+.2%} ({row['price_change']:+.2f}) | Deal: {row['deal']}")
        print(f"  Price: ${row['baseline_price']} → ${row['future_price']} over {row['ticks_elapsed']} ticks")
        print(f"  News: {row['headline'][:100]}")
    
    print("\n" + "="*100)
    print("TOP 30 LARGEST ABSOLUTE MOVES (Regardless of Direction)")
    print("="*100)
    largest_moves = results_df.nlargest(30, "abs_pct_change")
    for idx, row in largest_moves.iterrows():
        print(f"\n{row['ticker']} | {row['pct_change']:+.2%} ({row['price_change']:+.2f}) | Deal: {row['deal']}")
        print(f"  Price: ${row['baseline_price']} → ${row['future_price']} over {row['ticks_elapsed']} ticks")
        print(f"  News: {row['headline'][:100]}")
    
    # Summary statistics by ticker
    print("\n" + "="*100)
    print("SUMMARY STATISTICS BY TICKER")
    print("="*100)
    ticker_summary = results_df.groupby("ticker").agg({
        "pct_change": ["mean", "std", "min", "max", "count"]
    }).round(4)
    ticker_summary.columns = ["avg_impact", "std_impact", "worst_loss", "best_gain", "num_events"]
    ticker_summary = ticker_summary.sort_values("avg_impact", ascending=False)
    print(ticker_summary.to_string())


if __name__ == "__main__":
    main()
