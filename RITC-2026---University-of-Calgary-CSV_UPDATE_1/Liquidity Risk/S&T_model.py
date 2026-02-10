import signal
import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque, defaultdict

# Global API URL – adjust if needed.
API_URL = "http://localhost:9999/v1"

# Global shutdown flag.
shutdown = False

def signal_handler(signum, frame):
    """Gracefully handle CTRL+C."""
    global shutdown
    shutdown = True

class ApiException(Exception):
    pass

# --------------------------------------------------------------------------
# API Helper Functions
# --------------------------------------------------------------------------
def get_securities(session):
    """
    Retrieve all securities dynamically.
    Each security is expected to have at least a 'ticker' field and (optionally) a 'last' price.
    """
    resp = session.get(f"{API_URL}/securities")
    if not resp.ok:
        raise ApiException(f"get_securities failed: {resp.text}")
    return resp.json()

def get_order_book(session, ticker, limit=1000):
    """
    Retrieve the order book for a given ticker.
    """
    url = f"{API_URL}/securities/book?ticker={ticker}&limit={limit}"
    resp = session.get(url)
    if not resp.ok:
        raise ApiException(f"get_order_book({ticker}) failed: {resp.text}")
    return resp.json()

def get_tenders(session):
    """
    Retrieve the list of current tender offers.
    Each tender is expected to include at least:
      - 'id'
      - 'ticker' (which may include a market suffix, e.g. "ABC_M")
      - 'side': "BUY" or "SELL"
      - 'price': the tender price
      - 'volume': volume associated with the tender
      - 'taken': a boolean flag indicating whether you have accepted the tender
    """
    resp = session.get(f"{API_URL}/tenders")
    if not resp.ok:
        raise ApiException(f"get_tenders failed: {resp.text}")
    return resp.json()

# --------------------------------------------------------------------------
# Volume History Helpers
# --------------------------------------------------------------------------
# For each security ticker, we keep a deque of (timestamp, buy_volume, sell_volume)
volume_history = defaultdict(lambda: deque())

def update_volume_history(ticker, buy_volume, sell_volume, timestamp):
    """Append a new volume snapshot and remove entries older than 10 seconds."""
    volume_history[ticker].append((timestamp, buy_volume, sell_volume))
    while volume_history[ticker] and (timestamp - volume_history[ticker][0][0]) > 10:
        volume_history[ticker].popleft()

def get_volume_delta(ticker):
    """
    Compute the change in ANON buy and sell volume over the last 10 seconds,
    comparing the oldest available snapshot with the latest.
    Returns (delta_buy, delta_sell).
    """
    if volume_history[ticker]:
        oldest = volume_history[ticker][0]
        latest = volume_history[ticker][-1]
        delta_buy = latest[1] - oldest[1]
        delta_sell = latest[2] - oldest[2]
        return delta_buy, delta_sell
    else:
        return 0, 0

# --------------------------------------------------------------------------
# Plot Setup for ANON Volume
# --------------------------------------------------------------------------
def setup_volume_plot():
    """
    Set up a grouped bar chart that will display, for each security,
    the current ANON buy (green) and sell (red) volumes.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("ANON Volume on Each Side of the Order Book")
    ax.set_xlabel("Security")
    ax.set_ylabel("Volume")
    plt.ion()
    plt.show(block=False)
    return fig, ax

# --------------------------------------------------------------------------
# Realtime Monitoring Loop
# --------------------------------------------------------------------------
def realtime_monitoring():
    """
    Realtime loop that:
      - Retrieves the list of securities and, for each, gets the order book to sum ANON volumes.
      - Stores a 10‑second history of volumes and computes deltas (to infer market sentiment).
      - Graphs the current ANON buy and sell volumes (grouped by security).
      - For each security, prints the volume change (delta) and inferred market sentiment.
        Additionally, if one side’s volume has been increasing (e.g. buy side >> sell side),
        a warning is printed that the sentiment may reverse.
      - Retrieves accepted tenders and—for each—checks available markets (from dynamic securities)
        to see if the tender is profitable. For a tender to be considered profitable:
          (1) Enough volume is available at favorable prices such that the computed profit per share is positive.
          (2) The arbitrage per share is at least $0.10.
        In addition, if the order book for the market shows a warning (i.e. one side is rapidly increasing),
        the recommendation is overridden to “get out.”
    """
    global shutdown
    signal.signal(signal.SIGINT, signal_handler)

    # Create a session with your API key.
    API_KEY = {'X-API-Key': '2'}
    session = requests.Session()
    session.headers.update(API_KEY)

    fig, ax = setup_volume_plot()

    # We'll cache order book data here for tender analysis.
    order_books = {}

    # Arbitrage threshold is now $0.10.
    arb_threshold = 0.10

    while not shutdown:
        try:
            current_time = time.time()

            # Get all available securities.
            securities = get_securities(session)
            tickers = [sec["ticker"] for sec in securities]

            # For each ticker, get the order book and sum ANON volumes.
            anon_volumes = {}  # ticker -> (anon_buy, anon_sell)
            for ticker in tickers:
                ob = get_order_book(session, ticker, limit=1000)
                order_books[ticker] = ob  # store for later use in tender analysis

                bids = ob.get("bids", [])
                asks = ob.get("asks", [])
                anon_buy = sum(bid.get("quantity", 0) for bid in bids 
                               if bid.get("trader_id") == "ANON" and bid.get("status") == "OPEN")
                anon_sell = sum(ask.get("quantity", 0) for ask in asks 
                                if ask.get("trader_id") == "ANON" and ask.get("status") == "OPEN")
                anon_volumes[ticker] = (anon_buy, anon_sell)
                update_volume_history(ticker, anon_buy, anon_sell, current_time)

            # Update the volume graph (grouped bar chart).
            ax.cla()
            ax.set_title("ANON Volume on Each Side of the Order Book")
            ax.set_xlabel("Security")
            ax.set_ylabel("Volume")
            indices = range(len(tickers))
            buy_vols = [anon_volumes[t][0] for t in tickers]
            sell_vols = [anon_volumes[t][1] for t in tickers]
            width = 0.35
            ax.bar([i - width/2 for i in indices], buy_vols, width=width,
                   label="Buy Volume", color="green", alpha=0.7)
            ax.bar([i + width/2 for i in indices], sell_vols, width=width,
                   label="Sell Volume", color="red", alpha=0.7)
            ax.set_xticks(indices)
            ax.set_xticklabels(tickers, rotation=45)
            ax.legend()
            fig.canvas.draw()
            fig.canvas.flush_events()

            # For each security, compute volume deltas over the last 10 seconds and deduce market sentiment.
            sentiments = {}  # ticker -> sentiment ("positive", "negative", "neutral")
            for ticker in tickers:
                delta_buy, delta_sell = get_volume_delta(ticker)
                if delta_buy > delta_sell:
                    sentiment = "positive"
                elif delta_sell > delta_buy:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                sentiments[ticker] = sentiment
                print(f"[{ticker}] ΔBuy: {delta_buy}, ΔSell: {delta_sell} → Sentiment: {sentiment}")
                # If one side is increasing relative to the other, warn that sentiment may reverse.
                if delta_buy > delta_sell and delta_buy > 0:
                    print(f"Warning: For {ticker}, BUY side volume is increasing. Sentiment may reverse; if in tender, get out!")
                elif delta_sell > delta_buy and delta_sell > 0:
                    print(f"Warning: For {ticker}, SELL side volume is increasing. Sentiment may reverse; if in tender, get out!")

            # Check tenders – if accepted/taken – and analyze profitability.
            tenders = get_tenders(session)
            for tender in tenders:
                if not tender.get("taken", False):
                    continue  # only process tenders you have accepted
                tender_id     = tender.get("id")
                tender_ticker = tender.get("ticker")    # may include a market suffix, e.g., "ABC_M"
                tender_side   = tender.get("side").upper() # "BUY" or "SELL"
                tender_price  = tender.get("price")
                tender_vol    = tender.get("volume", 0)

                # Determine the base asset (e.g., if ticker is "ABC_M", base is "ABC").
                if "_" in tender_ticker:
                    base = tender_ticker.split("_")[0]
                else:
                    base = tender_ticker

                # Find available markets for this base asset from the dynamic securities list.
                available_markets = [t for t in tickers if t.startswith(base)]
                best_market = None
                best_profit = None
                best_recommendation = None

                # For each available market, use its order book to check if exit orders are available.
                for market in available_markets:
                    ob = order_books.get(market)
                    if not ob:
                        continue
                    profit = None
                    if tender_side == "BUY":
                        # For a BUY tender (you sold at tender_price, so you need to buy to cover):
                        # Look at the bid side orders with prices below the tender price.
                        bids = ob.get("bids", [])
                        profitable_bids = [bid for bid in bids if bid.get("price", 0) < tender_price]
                        cum_volume = sum(bid.get("quantity", 0) for bid in profitable_bids)
                        if cum_volume >= tender_vol and profitable_bids:
                            # Choose the highest bid below the tender price.
                            best_bid = max(bid.get("price", 0) for bid in profitable_bids)
                            profit = tender_price - best_bid
                    elif tender_side == "SELL":
                        # For a SELL tender (you bought at tender_price, so you need to sell to exit):
                        # Look at the ask side orders with prices above the tender price.
                        asks = ob.get("asks", [])
                        profitable_asks = [ask for ask in asks if ask.get("price", 0) > tender_price]
                        cum_volume = sum(ask.get("quantity", 0) for ask in profitable_asks)
                        if cum_volume >= tender_vol and profitable_asks:
                            best_ask = min(ask.get("price", 0) for ask in profitable_asks)
                            profit = best_ask - tender_price
                    else:
                        profit = None

                    # Only consider if profit is at least $0.10.
                    if profit is None or profit < arb_threshold:
                        continue

                    # Get volume delta for this market.
                    delta_buy, delta_sell = get_volume_delta(market)
                    # Start with a default recommendation based on overall sentiment.
                    if tender_side == "BUY":
                        recommendation = "BUY and hold"  # default for covering a short
                        if sentiments.get(market, "neutral") == "positive":
                            recommendation = "BUY and get out"
                    elif tender_side == "SELL":
                        recommendation = "SELL and hold"  # default for exiting a long
                        if sentiments.get(market, "neutral") == "negative":
                            recommendation = "SELL and get out"
                    else:
                        recommendation = "No recommendation"

                    # If the order book on this market shows one side is increasing significantly,
                    # warn that sentiment may reverse and override recommendation to "get out."
                    if delta_buy > delta_sell and delta_buy > 0:
                        print(f"Warning: For {market}, BUY side volume is increasing (ΔBuy: {delta_buy}, ΔSell: {delta_sell}); sentiment may reverse!")
                        if tender_side == "SELL":
                            recommendation = "SELL and get out"
                    elif delta_sell > delta_buy and delta_sell > 0:
                        print(f"Warning: For {market}, SELL side volume is increasing (ΔSell: {delta_sell}, ΔBuy: {delta_buy}); sentiment may reverse!")
                        if tender_side == "BUY":
                            recommendation = "BUY and get out"

                    # Choose the market with the highest profit per share that meets the $0.10 threshold.
                    if best_profit is None or profit > best_profit:
                        best_profit = profit
                        best_market = market
                        best_recommendation = recommendation

                if best_market and best_profit is not None and best_profit >= arb_threshold:
                    print(f"Tender {tender_id} ({tender_side} {tender_ticker} @ {tender_price:.2f}) is profitable on {best_market}:")
                    print(f"  Profit per share: {best_profit:.2f} (≥ ${arb_threshold:.2f} arbitrage)")
                    print(f"  Recommendation: {best_recommendation}")
                else:
                    print(f"Tender {tender_id} is not currently profitable in any market (arbitrage < ${arb_threshold:.2f}).")

            time.sleep(1)
        except ApiException as e:
            print(f"[API Error] {e}")
            time.sleep(2)
        except Exception as e:
            print(f"[Error] {e}")
            time.sleep(2)

    print("Shutting down monitoring.")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    realtime_monitoring()