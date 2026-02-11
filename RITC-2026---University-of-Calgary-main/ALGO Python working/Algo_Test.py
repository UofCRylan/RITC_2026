import requests
import time
import threading
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass, field

# ==============================================================================
# CONFIGURATION
# ==============================================================================
API_KEY = 'YOUR_API_KEY_HERE'  # REPLACE THIS
API_URL = 'http://localhost:9999/v1'
TICKERS = ["SPNG","SMMR","ATMN","WNTR"]

# STRATEGY PARAMETERS
MAX_AGGREGATE_POSITION = 12500  # Buffer below the hard 13,000 limit
MOMENTUM_THRESHOLD = 0.60       # OBI Threshold to trigger momentum trade (0.6 = 60% buy pressure)
MOMENTUM_SIZE = 3000            # Size for directional trades
MM_SIZE = 500                   # Size for passive limit orders
MM_MIN_SPREAD = 0.03            # Minimum spread (cents)
MM_VOL_MULTIPLIER = 2.0         # How much to widen spread per unit of volatility
MM_SKEW_FACTOR = 0.0001         # Price adjustment per share of inventory
OBI_WINDOW = 5                  # 5-tick rolling window for depth tracking

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================
@dataclass
class MarketState:
    tick: int = 0
    prices: dict = field(default_factory=lambda: {t: deque(maxlen=20) for t in TICKERS})
    obi_history: dict = field(default_factory=lambda: {t: deque(maxlen=OBI_WINDOW) for t in TICKERS})
    positions: dict = field(default_factory=lambda: {t: 0 for t in TICKERS})
    book: dict = field(default_factory=lambda: {t: {'bid': 0, 'ask': 0, 'bid_vol': 0, 'ask_vol': 0} for t in TICKERS})
    cash: float = 0.0
    
    def update_price(self, ticker, price):
        self.prices[ticker].append(price)
        
    def update_obi(self, ticker, bid_vol, ask_vol):
        # Calculate Order Book Imbalance: (Bid - Ask) / (Bid + Ask)
        # Range: -1 (Sell Pressure) to +1 (Buy Pressure)
        if (bid_vol + ask_vol) > 0:
            obi = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        else:
            obi = 0
        self.obi_history[ticker].append(obi)

    def get_volatility(self, ticker):
        if len(self.prices[ticker]) < 5:
            return 0.01 # Default low vol
        return np.std(self.prices[ticker])

    def get_avg_obi(self, ticker):
        if len(self.obi_history[ticker]) == 0:
            return 0
        return np.mean(self.obi_history[ticker])

    def get_gross_position(self):
        return sum(abs(p) for p in self.positions.values())

# ==============================================================================
# API HANDLER
# ==============================================================================
class RITSession:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'X-API-Key': API_KEY})
    
    def get(self, endpoint):
        try:
            resp = self.session.get(f"{API_URL}/{endpoint}", timeout=0.5)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"API Read Error: {e}")
        return None

    def post(self, endpoint, params=None):
        try:
            resp = self.session.post(f"{API_URL}/{endpoint}", params=params, timeout=0.5)
            return resp.ok
        except Exception as e:
            print(f"API Write Error: {e}")
        return False

    def cancel_all(self):
        self.post('commands/cancel', {'all': 1})

# ==============================================================================
# TRADING ENGINE
# ==============================================================================
class AlgoEngine:
    def __init__(self):
        self.api = RITSession()
        self.state = MarketState()
        self.running = True
        self.lock = threading.Lock()
        
    def data_feed_loop(self):
        """High-frequency poll of market data"""
        while self.running:
            # 1. Get Case Status (Tick)
            case = self.api.get('case')
            if not case or case['status']!= 'ACTIVE':
                time.sleep(1)
                continue
            self.state.tick = case['tick']

            # 2. Get Positions
            securities = self.api.get('securities')
            if securities:
                with self.lock:
                    for s in securities:
                        if s['ticker'] in TICKERS:
                            self.state.positions[s['ticker']] = s['position']

            # 3. Get Book Data (The Heavy Lifting)
            for ticker in TICKERS:
                book = self.api.get(f'securities/book?ticker={ticker}&limit=1')
                if book and book['bids'] and book['asks']:
                    bid = book['bids']['price']
                    ask = book['asks']['price']
                    bid_vol = book['bids']['quantity']
                    ask_vol = book['asks']['quantity']
                    
                    with self.lock:
                        self.state.book[ticker] = {
                            'bid': bid, 'ask': ask, 
                            'bid_vol': bid_vol, 'ask_vol': ask_vol
                        }
                        # Analytics Updates
                        mid_price = (bid + ask) / 2
                        self.state.update_price(ticker, mid_price)
                        self.state.update_obi(ticker, bid_vol, ask_vol)
            
            time.sleep(0.1) # Poll every 100ms

    def trading_loop(self):
        """Strategy Execution Logic"""
        print("Engine Started. Waiting for data...")
        time.sleep(2) # Warmup
        
        while self.running:
            gross_pos = self.state.get_gross_position()
            remaining_limit = MAX_AGGREGATE_POSITION - gross_pos
            
            # --- STRATEGY 1: MOMENTUM (High Priority) ---
            # Checks depth every iteration (rolling window logic handles the "every 5 ticks" smoothing)
            for ticker in TICKERS:
                with self.lock:
                    avg_obi = self.state.get_avg_obi(ticker)
                    curr_pos = self.state.positions[ticker]
                    book = self.state.book[ticker]
                
                # Signal: Liquidity Vacuum (Strong Imbalance)
                # If OBI > 0.6, Buy side is huge, Sell side is thin -> Price goes UP
                if avg_obi > MOMENTUM_THRESHOLD and curr_pos < MOMENTUM_SIZE:
                    if remaining_limit > MOMENTUM_SIZE:
                        print(f"MOMENTUM BUY {ticker} | OBI: {avg_obi:.2f}")
                        self.api.post('orders', {'ticker': ticker, 'type': 'MARKET', 'action': 'BUY', 'quantity': MOMENTUM_SIZE})
                        # Hold the ticker (do not MM on this one while momentum is active)
                        continue 

                # If OBI < -0.6, Sell side is huge, Buy side is thin -> Price goes DOWN
                elif avg_obi < -MOMENTUM_THRESHOLD and curr_pos > -MOMENTUM_SIZE:
                    if remaining_limit > MOMENTUM_SIZE:
                        print(f"MOMENTUM SELL {ticker} | OBI: {avg_obi:.2f}")
                        self.api.post('orders', {'ticker': ticker, 'type': 'MARKET', 'action': 'SELL', 'quantity': MOMENTUM_SIZE})
                        continue

                # Exit Logic: If we hold a position but OBI neutralizes, get out
                if abs(curr_pos) > 100 and abs(avg_obi) < 0.2:
                    action = 'SELL' if curr_pos > 0 else 'BUY'
                    print(f"MOMENTUM EXIT {ticker} | Closing {curr_pos}")
                    self.api.post('orders', {'ticker': ticker, 'type': 'MARKET', 'action': action, 'quantity': abs(curr_pos)})

            # --- STRATEGY 2: MARKET MAKING (Passive) ---
            # Only market make if we are not in a massive momentum trade for that ticker
            # and we have risk budget
            if gross_pos < MAX_AGGREGATE_POSITION:
                for ticker in TICKERS:
                    with self.lock:
                        pos = self.state.positions[ticker]
                        vol = self.state.get_volatility(ticker)
                        book = self.state.book[ticker]
                        mid = (book['bid'] + book['ask']) / 2
                    
                    # Skip MM if we are holding a directional momentum trade
                    if abs(pos) > MM_SIZE * 2: 
                        continue

                    # 1. Calculate Fair Value (Inventory Skew)
                    # If Long, lower price to sell. If Short, raise price to buy.
                    reservation_price = mid - (pos * MM_SKEW_FACTOR)

                    # 2. Calculate Spread (Volatility Adjusted)
                    spread = MM_MIN_SPREAD + (vol * MM_VOL_MULTIPLIER)
                    half_spread = spread / 2

                    my_bid = round(reservation_price - half_spread, 2)
                    my_ask = round(reservation_price + half_spread, 2)

                    # 3. Safety Check: Don't cross the market (take liquidity)
                    # We want rebates (Limit orders), so ensure Bid < Best Ask and Ask > Best Bid
                    if my_bid >= book['ask']: my_bid = book['ask'] - 0.01
                    if my_ask <= book['bid']: my_ask = book['bid'] + 0.01

                    # 4. Execute (Cancel/Replace logic roughly approximated by just sending orders)
                    # In a real bot, we would track open order IDs to modify them. 
                    # For RITC simple API, we often just place new ones and let the old ones get filled or cancelled periodically.
                    # Here we simply place orders if we aren't "full"
                    
                    # Note: To prevent spamming, only order if we don't have active orders (simplified)
                    # A robust implementation requires GET /orders status check.
                    
                    self.api.post('orders', {'ticker': ticker, 'type': 'LIMIT', 'action': 'BUY', 'quantity': MM_SIZE, 'price': my_bid})
                    self.api.post('orders', {'ticker': ticker, 'type': 'LIMIT', 'action': 'SELL', 'quantity': MM_SIZE, 'price': my_ask})

            # Cleanup / Rate Limit Sleep
            time.sleep(0.5) 

            # Periodically clear stale orders to prevent "Ghost Inventory"
            if self.state.tick % 10 == 0:
                self.api.cancel_all()

    def run(self):
        # Start Data Thread
        t_data = threading.Thread(target=self.data_feed_loop)
        t_data.daemon = True
        t_data.start()

        # Start Trading Thread
        try:
            self.trading_loop()
        except KeyboardInterrupt:
            print("Stopping Engine...")
            self.running = False
            self.api.cancel_all()

# ==============================================================================
# MAIN ENTRY
# ==============================================================================
if __name__ == "__main__":
    bot = AlgoEngine()
    bot.run()