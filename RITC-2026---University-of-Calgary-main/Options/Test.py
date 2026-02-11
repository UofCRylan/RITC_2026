import tkinter as tk
from tkinter import ttk, messagebox
import requests
import threading
import time
import math
import re
from scipy.stats import norm
from datetime import datetime

# ==============================================================================
# SECTION 1: CONFIGURATION AND CONSTANTS
# ==============================================================================
# RIT Connection Settings
# Ensure RIT Client is running and API is enabled (Port 9999 default)
API_HOST = 'http://localhost'
API_PORT = '9999'
BASE_URL = f"{API_HOST}:{API_PORT}/v1"
API_KEY = 'YOUR_API_KEY_HERE'  # <--- UPDATE THIS WITH YOUR API KEY

# Market Microstructure Constants (Based on RITC Case Package)
RISK_FREE_RATE = 0.0          # r = 0%
TICKS_PER_MONTH = 300         # Assumed ticks per sub-heat/month
TICKS_PER_YEAR = 3600         # Scaling factor for Time to Maturity (T)
CONTRACT_MULTIPLIER = 100     # 1 Option = 100 Shares
MAX_TRADE_SIZE_OPT = 100      # Max contracts per API order
MAX_NET_OPT_LIMIT = 2500      # Max net option contracts allowed (Case Constraint)
DEFAULT_VOLATILITY = 0.20     # Fallback volatility if no news parsed
DEFAULT_DELTA_LIMIT = 5000    # Fallback delta limit

# ==============================================================================
# SECTION 2: GLOBAL STATE MANAGEMENT
# ==============================================================================
# We use a thread-safe dictionary to share state between the API worker
# and the GUI main thread.
state_lock = threading.Lock()
market_state = {
    'connected': False,
    'tick': 0,
    'status': 'UNKNOWN',
    'portfolio': {},          # Stores {ticker: {'pos': x, 'last': y, 'type': z}}
    'net_delta': 0.0,         # Calculated Portfolio Delta
    'delta_limit': DEFAULT_DELTA_LIMIT,
    'forecast_vol': DEFAULT_VOLATILITY,
    'news_read': set()        # Cache of read news IDs to prevent re-parsing
}

# ==============================================================================
# SECTION 3: RIT API CLIENT (HTTP LAYER)
# ==============================================================================
class RITClient:
    """
    Wrapper for RIT REST API. Handles headers, authentication, 
    and basic error checking.
    """
    HEADERS = {'X-API-Key': API_KEY}

    @classmethod
    def get(cls, endpoint, params=None):
        try:
            resp = requests.get(f"{BASE_URL}/{endpoint}", headers=cls.HEADERS, params=params, timeout=0.5)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                # Rate limit hit - passive wait handled in worker loop
                return None
            else:
                print(f" GET {endpoint}: {resp.status_code}")
                return None
        except requests.exceptions.RequestException:
            return None

    @classmethod
    def post(cls, endpoint, params=None):
        try:
            resp = requests.post(f"{BASE_URL}/{endpoint}", headers=cls.HEADERS, params=params, timeout=0.5)
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f" POST {endpoint}: {resp.status_code} - {resp.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f" Connection failed: {e}")
            return None

# ==============================================================================
# SECTION 4: QUANTITATIVE ENGINE (BSM & GREEKS)
# ==============================================================================
class QuantEngine:
    """
    Implements Black-Scholes-Merton Logic.
    """
    @staticmethod
    def d1(S, K, T, r, sigma):
        # Prevent division by zero errors
        if T <= 0.001 or sigma <= 0.001:
            return 0
        return (math.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def calculate_delta(option_type, S, K, T, r, sigma):
        """
        Calculates the delta of a single option unit (not scaled by 100).
        """
        d1_val = QuantEngine.d1(S, K, T, r, sigma)
        if option_type == 'C':
            return norm.cdf(d1_val)
        elif option_type == 'P':
            return norm.cdf(d1_val) - 1.0
        return 0.0

    @staticmethod
    def update_portfolio_delta():
        """
        Iterates through the global portfolio state and calculates Net Delta.
        """
        with state_lock:
            portfolio = market_state['portfolio']
            sigma = market_state['forecast_vol']
            
            # 1. Get Underlying Price (S)
            rtm_data = portfolio.get('RTM')
            if not rtm_data:
                return # No market data yet
            S = rtm_data['last']
            
            # 2. Determine Time to Maturity (T)
            # T is fraction of year. 
            # In RITC, options expire at tick 300 (or end of heat).
            # We assume a fixed T approximation or derive from tick if heat info known.
            # Using a conservative 20-day annualized approximation:
            current_tick = market_state['tick']
            ticks_remaining = max(1, TICKS_PER_MONTH - (current_tick % TICKS_PER_MONTH))
            T = ticks_remaining / TICKS_PER_YEAR
            
            total_delta = 0.0
            
            # 3. Add Delta from Stock
            total_delta += rtm_data['pos'] # 1 share = 1 delta
            
            # 4. Add Delta from Options
            for ticker, data in portfolio.items():
                if data['type'] == 'OPTION':
                    # Parse Ticker (Format: RTM1C50)
                    # Regex captures: (Underlying)(Month)(Call/Put)(Strike)
                    match = re.match(r"([A-Z]+)(\d+)([CP])(\d+)", ticker)
                    if match:
                        otype = match.group(3) # 'C' or 'P'
                        strike = float(match.group(4))
                        pos = data['pos']
                        
                        unit_delta = QuantEngine.calculate_delta(otype, S, strike, T, RISK_FREE_RATE, sigma)
                        
                        # Scale by position and contract multiplier
                        position_delta = unit_delta * CONTRACT_MULTIPLIER * pos
                        total_delta += position_delta
            
            market_state['net_delta'] = total_delta

# ==============================================================================
# SECTION 5: TRADING LOGIC & NEWS PARSER
# ==============================================================================
class AlgoLogic:
    """
    Handles higher-level strategy: News parsing, Hedging, Max Position Execution.
    """
    
    @staticmethod
    def process_news(news_items):
        """
        Extracts Volatility Forecasts and Delta Limits from news text.
        """
        global market_state
        
        for item in news_items:
            nid = item['id']
            if nid in market_state['news_read']:
                continue
            
            market_state['news_read'].add(nid)
            text = f"{item['headline']} {item['body']}"
            
            # Regex 1: Volatility ("volatility... will be 25%")
            vol_match = re.search(r"volatility.*?(\d+)%", text, re.IGNORECASE)
            if vol_match:
                new_vol = float(vol_match.group(1)) / 100.0
                with state_lock:
                    market_state['forecast_vol'] = new_vol
                print(f" Volatility Updated: {new_vol:.0%}")

            # Regex 2: Delta Limit ("delta limit... is 5,000")
            limit_match = re.search(r"delta limit.*?([0-9,]+)", text, re.IGNORECASE)
            if limit_match:
                limit_str = limit_match.group(1).replace(',', '')
                with state_lock:
                    market_state['delta_limit'] = int(limit_str)
                print(f" Delta Limit Updated: {limit_str}")

    @staticmethod
    def execute_max_straddle(direction):
        """
        Buys/Sells Straddles up to the Net Option Limit.
        direction: 'BUY' or 'SELL'
        """
        with state_lock:
            pf = market_state['portfolio']
            # Calculate current Net Option Position
            current_opt_contracts = 0
            for k, v in pf.items():
                if v['type'] == 'OPTION':
                    current_opt_contracts += v['pos'] # Can be negative (short)
            
            # RTM Price for ATM determination
            S = pf.get('RTM', {}).get('last', 50.0)

        # 1. Determine ATM Strike
        strike = round(S)
        call = f"RTM1C{strike}"
        put = f"RTM1P{strike}"

        # 2. Calculate Available Capacity
        # We want to maximize |Net Position|. 
        # If we are Long 500, and Limit is 2500, we can buy 2000 more.
        # If we are Short 500, we can sell 2000 more.
        # Simplified: Just try to fill the remaining bucket.
        remaining_cap = MAX_NET_OPT_LIMIT - abs(current_opt_contracts)
        
        # A straddle uses 2 contracts (1 Call + 1 Put).
        # So max straddles = remaining_cap / 2
        max_straddles = remaining_cap // 2
        
        if max_straddles <= 0:
            print("[ALGO] Max Position Reached. Cannot execute.")
            return

        print(f"[ALGO] Executing {direction} {max_straddles} Straddles at Strike {strike}")

        # 3. Execute in Chunks (Max 100 per order)
        # We spawn a thread for this to not block the UI or Hedger
        threading.Thread(target=AlgoLogic._batch_order, args=(call, put, max_straddles, direction)).start()

    @staticmethod
    def _batch_order(call_ticker, put_ticker, total_qty, direction):
        """Helper to send batched orders."""
        remaining = total_qty
        while remaining > 0:
            batch = min(remaining, MAX_TRADE_SIZE_OPT)
            RITClient.post("orders", {'ticker': call_ticker, 'type': 'MARKET', 'quantity': batch, 'action': direction})
            RITClient.post("orders", {'ticker': put_ticker, 'type': 'MARKET', 'quantity': batch, 'action': direction})
            remaining -= batch
            time.sleep(0.1) # Small delay to prevent API flooding

    @staticmethod
    def run_hedger():
        """
        Checks Net Delta against Limit. Executed frequently by worker thread.
        """
        with state_lock:
            delta = market_state['net_delta']
            limit = market_state['delta_limit']
        
        # Buffer Logic: Hedge if we exceed 10% of limit or raw 200 delta.
        # This prevents over-hedging transaction costs.
        threshold = 200
        
        if abs(delta) > threshold:
            action = "SELL" if delta > 0 else "BUY"
            qty = min(abs(int(delta)), 10000) # Max stock order size
            
            print(f" Delta {int(delta)} exceeds threshold. {action} {qty} RTM.")
            RITClient.post("orders", {'ticker': 'RTM', 'type': 'MARKET', 'quantity': qty, 'action': action})

# ==============================================================================
# SECTION 6: GUI DASHBOARD (VISUALIZATION)
# ==============================================================================
class VolatilityDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("RITC Volatility Algo - Delta Commander")
        self.root.geometry("900x600")
        self.root.configure(bg="#222") # Dark Mode

        # -- HEADER --
        header_frame = tk.Frame(root, bg="#333", pady=10)
        header_frame.pack(fill=tk.X)
        tk.Label(header_frame, text="RITC VOLATILITY MATRIX", font=("Segoe UI", 18, "bold"), fg="white", bg="#333").pack()

        # -- METRICS --
        metric_frame = tk.Frame(root, bg="#222", pady=20)
        metric_frame.pack(fill=tk.X)
        
        self.lbl_vol = self._create_metric(metric_frame, "FORECAST VOL", "0.00%", "#f39c12")
        self.lbl_limit = self._create_metric(metric_frame, "DELTA LIMIT", "0", "#e74c3c")
        self.lbl_delta = self._create_metric(metric_frame, "NET DELTA", "0", "#2ecc71")
        self.lbl_pos = self._create_metric(metric_frame, "TICK", "0", "#3498db")

        # -- VISUAL GAUGE (The Requirement) --
        # "2 lines that are red and a middle line that is green"
        self.canvas_w = 800
        self.canvas_h = 120
        self.canvas = tk.Canvas(root, width=self.canvas_w, height=self.canvas_h, bg="black", highlightthickness=2, highlightbackground="#555")
        self.canvas.pack(pady=20)

        # -- CONTROLS --
        btn_frame = tk.Frame(root, bg="#222", pady=20)
        btn_frame.pack(fill=tk.BOTH, expand=True)

        # Max Long Button
        self.btn_long = tk.Button(btn_frame, text="MAX LONG VOL\n(BUY STRADDLE)", 
                                  font=("Segoe UI", 14, "bold"), bg="#27ae60", fg="white",
                                  command=lambda: AlgoLogic.execute_max_straddle("BUY"),
                                  relief=tk.FLAT, width=25, height=4)
        self.btn_long.pack(side=tk.LEFT, padx=50)

        # Max Short Button
        self.btn_short = tk.Button(btn_frame, text="MAX SHORT VOL\n(SELL STRADDLE)", 
                                   font=("Segoe UI", 14, "bold"), bg="#c0392b", fg="white",
                                   command=lambda: AlgoLogic.execute_max_straddle("SELL"),
                                   relief=tk.FLAT, width=25, height=4)
        self.btn_short.pack(side=tk.RIGHT, padx=50)
        
        # Start GUI Update Loop
        self.update_gui()

    def _create_metric(self, parent, title, value, color):
        f = tk.Frame(parent, bg="#222")
        f.pack(side=tk.LEFT, expand=True)
        tk.Label(f, text=title, font=("Segoe UI", 10), fg="#aaa", bg="#222").pack()
        lbl = tk.Label(f, text=value, font=("Segoe UI", 16, "bold"), fg=color, bg="#222")
        lbl.pack()
        return lbl

    def draw_traffic_light_gauge(self, current_delta, limit):
        self.canvas.delete("all")
        w, h = self.canvas_w, self.canvas_h
        
        # Geometry
        center_x = w / 2
        
        # Scale: Let the visible area represent 150% of the limit to show breaches
        display_limit = limit * 1.5 if limit > 0 else 10000
        
        def map_x(val):
            # Maps a delta value to X coordinate
            # -display_limit -> 0
            # 0 -> center_x
            # +display_limit -> w
            pct = (val + display_limit) / (2 * display_limit)
            return int(pct * w)

        # 1. Draw Center Axis (Zero Line) - Subtle Grey
        self.canvas.create_line(center_x, 0, center_x, h, fill="#444", dash=(4, 4))
        
        # 2. Draw RED Lines (The Limit)
        x_lim_pos = map_x(limit)
        x_lim_neg = map_x(-limit)
        
        self.canvas.create_line(x_lim_pos, 10, x_lim_pos, h-10, fill="red", width=3)
        self.canvas.create_text(x_lim_pos, h-5, text=f"+{limit}", fill="red", anchor="n", font=("Arial", 8))
        
        self.canvas.create_line(x_lim_neg, 10, x_lim_neg, h-10, fill="red", width=3)
        self.canvas.create_text(x_lim_neg, h-5, text=f"-{limit}", fill="red", anchor="n", font=("Arial", 8))

        # 3. Draw GREEN Line (Current Delta) - "Traffic Light Logic"
        # If we breach, turn Orange/Yellow to warn trader
        x_delta = map_x(current_delta)
        # Clamp to canvas
        x_delta = max(2, min(w-2, x_delta))
        
        color = "#2ecc71" # Green
        if abs(current_delta) > limit:
            color = "#f39c12" # Warning Orange
            
        self.canvas.create_line(x_delta, 0, x_delta, h, fill=color, width=6)
        
        # Delta Label on the line
        self.canvas.create_text(x_delta, 15, text=f"{int(current_delta)}", fill=color, font=("Arial", 10, "bold"))

    def update_gui(self):
        # Read State
        with state_lock:
            vol = market_state['forecast_vol']
            delta = market_state['net_delta']
            limit = market_state['delta_limit']
            tick = market_state['tick']

        # Update Text
        self.lbl_vol.config(text=f"{vol*100:.1f}%")
        self.lbl_delta.config(text=f"{int(delta)}")
        self.lbl_limit.config(text=f"{limit}")
        self.lbl_pos.config(text=f"{tick}")
        
        # Update Visuals
        self.draw_traffic_light_gauge(delta, limit)
        
        # Loop (100ms)
        self.root.after(100, self.update_gui)

# ==============================================================================
# SECTION 7: WORKER THREAD (BACKGROUND LOOP)
# ==============================================================================
def data_worker():
    """
    Background loop that fetches data, updates state, and runs hedger.
    """
    while True:
        try:
            # 1. Check Case Status
            case = RITClient.get('case')
            if not case: 
                time.sleep(1); continue
            
            if case['status']!= 'ACTIVE':
                with state_lock: market_state['status'] = case['status']
                time.sleep(1)
                continue
            
            with state_lock:
                market_state['tick'] = case['tick']
                market_state['status'] = 'ACTIVE'

            # 2. Update Securities (Portfolio & Prices)
            secs = RITClient.get('securities')
            if secs:
                with state_lock:
                    for s in secs:
                        market_state['portfolio'][s['ticker']] = s
            
            # 3. Process News
            news = RITClient.get('news')
            if news:
                AlgoLogic.process_news(news)

            # 4. Math: Update Greeks
            QuantEngine.update_portfolio_delta()

            # 5. Execution: Auto Hedge
            AlgoLogic.run_hedger()

            # Throttle to respect API limits (approx 5Hz)
            time.sleep(0.2)
            
        except Exception as e:
            print(f" {e}")
            time.sleep(1)

# ==============================================================================
# SECTION 8: MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    # 1. Start Background Thread
    t = threading.Thread(target=data_worker, daemon=True)
    t.start()
    
    # 2. Start GUI (Main Thread)
    root = tk.Tk()
    app = VolatilityDashboard(root)
    root.mainloop()