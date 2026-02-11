import pandas as pd
import glob
import numpy as np

def analyze_basket_theory():
    print("--- 1. LOADING DATA ---")
    all_files = glob.glob("order_book_data_*.csv")
    
    df_list = []
    for filename in all_files:
        try:
            # Load and handle bad lines
            temp_df = pd.read_csv(filename, on_bad_lines='skip', low_memory=False)
            df_list.append(temp_df)
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    if not df_list:
        print("No CSV files found.")
        return

    df = pd.concat(df_list, ignore_index=True)
    
    # Filter for the 4 Season tickers
    target_tickers = ['SPNG', 'SMMR', 'ATMN', 'WNTR']
    df = df[df['ticker'].isin(target_tickers)]
    
    # Ensure numeric data
    df['last_price'] = pd.to_numeric(df['last_price'], errors='coerce')
    df['tick'] = pd.to_numeric(df['tick'], errors='coerce')
    df = df.dropna(subset=['last_price', 'tick'])

    # Pivot to align prices by tick
    # Index=tick, Columns=ticker, Values=last_price
    pivot = df.pivot_table(index='tick', columns='ticker', values='last_price')
    
    # Forward fill missing data (if SPNG trades at tick 5 but WNTR doesn't, use WNTR's price from tick 4)
    pivot = pivot.ffill().dropna()

    if pivot.empty:
        print("Insufficient overlapping data.")
        return

    # CALCULATE THE BASKET
    pivot['Basket_Sum'] = pivot[target_tickers].sum(axis=1)
    
    mean_val = pivot['Basket_Sum'].mean()
    std_val = pivot['Basket_Sum'].std()
    
    print("\n--- STATISTICAL ARBITRAGE METRICS ---")
    print(f"Mean Basket Value: ${mean_val:.2f}")
    print(f"Std Deviation:      ${std_val:.4f}")
    print(f"Min Basket Value:  ${pivot['Basket_Sum'].min():.2f}")
    print(f"Max Basket Value:  ${pivot['Basket_Sum'].max():.2f}")
    
    print("\n--- CORRELATION MATRIX (Do they move opposite?) ---")
    print(pivot[target_tickers].corr())

    # Z-Score Analysis
    pivot['Z_Score'] = (pivot['Basket_Sum'] - mean_val) / std_val
    
    print("\n--- THEORETICAL PROFITS ---")
    print("If you sold the basket when Z-Score > 1.5 and bought when Z-Score < -1.5:")
    opportunities = len(pivot[abs(pivot['Z_Score']) > 1.5])
    print(f"Potential Trade Signals found: {opportunities}")
    
    if std_val < 2.0:
        print("\n✅ VERDICT: EXCELLENT CANDIDATE FOR BASKET ARB")
        print("The sum is stable. Deviations are temporary alpha.")
    else:
        print("\n⚠️ VERDICT: VOLATILE BASKET")
        print("The sum trends. Use a Moving Average instead of a fixed mean.")

if __name__ == "__main__":
    analyze_basket_theory()