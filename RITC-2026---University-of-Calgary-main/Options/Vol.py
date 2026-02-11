import tkinter as tk
from tkinter import ttk, messagebox
import requests
import threading
import time
import math
import re
from scipy.stats import norm

# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================
API_KEY = 'YOUR_API_KEY_HERE'  # <--- UPDATE THIS
API_HOST = 'http://localhost'
API_PORT = '9999'
BASE_URL = f"{API_HOST}:{API_PORT}/v1"

# Simulation Time Constants
TICKS_PER_MONTH = 300         # 1 Sub-heat
TICKS_PER_WEEK = 75           # News updates occur weekly
TICKS_PER_YEAR = 3600         # 12 months * 300 ticks
RISK_FREE_RATE = 0.00         # Case assumption r=0

# Risk Management Constants
DEFAULT_DELTA_LIMIT = 25000   # Fallback if no news
SOFT_LIMIT_BUFFER = 0.80      # Hedge when we hit 80% of the limit
HEDGE_TARGET = 0.0            # When hedging, aim for 0 (neutral)
MAX_NET_OPTIONS = 2000        # Exchange limit on net option contracts

# ==============================================================================
# 2. API CLIENT (Thread-Safe & Persistent)
# ==============================================================================
class RITClient:
    _session = requests.Session()
    _session.headers.update({'X-API-Key': API_KEY})
    
    @classmethod
    def get(cls, endpoint, params=None):
        try:
            t0 = time.time()
            resp = cls._session.get(f"{BASE_URL}/{endpoint}", params=params, timeout=1.0)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                time.sleep(0.5) # Backoff
        except Exception as e:
            print(f"API Error (GET {endpoint}): {e}")
        return None

    @classmethod
    def post(cls, endpoint, params=None):
        try:
            resp = cls._session.post(f"{BASE_URL}/{endpoint}", params=params, timeout=1.0)
            return resp.ok
        except Exception as e:
            print(f"API Error (POST {endpoint}): {e}")
            return False

# ==============================================================================
# 3. MATHEMATICAL ENGINE (BSM & RMS)
# ==============================================================================
class QuantEngine:
    @staticmethod
    def calculate_rms_volatility(current_tick, vol_schedule):
        """
        Calculates Root Mean Square volatility based on time-weighted variance
        of the remaining weeks.
        """
        # Determine current week (0-3)
        # RITC ticks are 1-based usually, usually running 300 ticks/heat.
        # Week 1: 0-75, Week 2: 76-150, etc.
        eff_tick = max(0, min(current_tick, TICKS_PER_MONTH - 1))
        
        current_week_idx = eff_tick // TICKS_PER_WEEK
        ticks_in_current_week = TICKS_PER_WEEK - (eff_tick % TICKS_PER_WEEK)
        
        total_variance = 0.0
        total_ticks = 0.0
        
        # 1. Variance from rest of current week
        if current_week_idx < 4:
            vol = vol_schedule[current_week_idx]
            total_variance += (vol ** 2) * ticks_in_current_week
            total_ticks += ticks_in_current_week
        
        # 2. Variance from future weeks
        for w in range(current_week_idx + 1, 4):
            vol = vol_schedule[w]
            total_variance += (vol ** 2) * TICKS_PER_WEEK
            total_ticks += TICKS_PER_WEEK
            
        if total_ticks <= 0:
            return vol_schedule[-1] # End of heat fallback
            
        return math.sqrt(total_variance / total_ticks)

    @staticmethod
    def d1(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0: return 0
        return (math.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def get_delta(option_type, S, K, T, r, sigma):
        _d1 = QuantEngine.d1(S, K, T, r, sigma)
        if option_type == 'C':
            return norm.cdf(_d1)
        else:
            return norm.cdf(_d1) - 1.0

    @staticmethod
    def get_price(option_type, S, K, T, r, sigma):
        _d1 = QuantEngine.d1(S, K, T, r, sigma)
        _d2 = _d1 - sigma * math.sqrt(T)
        if option_type == 'C':
            return S * norm.cdf(_d1) - K * math.exp(-r*T) * norm.cdf(_d2)
        else:
            return K * math.exp(-r*T) * norm.cdf(-_d2) - S * norm.cdf(-_d1)

# ==============================================================================
# 4. STRATEGY CONTROLLER
# ==============================================================================
class Strategy:
    def __init__(self):
        # State
        self.tick = 0
        self.rtm_price = 50.00
        self.portfolio = {}
        self.net_delta = 0.0
        self.delta_limit = DEFAULT_DELTA_LIMIT
        
        # Volatility Schedule (Forecasts for Week 1, 2, 3, 4)
        # Initialize with 20% (0.20) as baseline
        self.vol_schedule = [0.20, 0.20, 0.20, 0.20] 
        self.rms_vol = 0.20
        self.processed_news_ids = set()
        
        self.lock = threading.Lock()

    def update_market_data(self):
        """Main update loop called by worker thread"""
        
        # 1. Get Case Status
        case = RITClient.get('case')
        if not case: return
        self.tick = case['tick']
        
        # 2. Get Securities (Prices & Positions)
        secs = RITClient.get('securities')
        if not secs: return
        
        with self.lock:
            # Update Portfolio & RTM Price
            self.net_delta = 0.0 # Will recalculate from API data if available, or manually
            
            for s in secs:
                self.portfolio[s['ticker']] = s
                if s['ticker'] == 'RTM':
                    self.rtm_price = s['last']
                    self.net_delta += s['position'] # RTM Delta = 1.0 * shares

            # 3. Calculate Option Deltas (Manual Calculation is safer/faster than waiting for API Greeks)
            # T in years
            ticks_left = TICKS_PER_MONTH - self.tick
            T = max(ticks_left, 1) / TICKS_PER_YEAR
            
            # Recalculate Greeks based on OUR RMS vol (better than market maker's)
            self.rms_vol = QuantEngine.calculate_rms_volatility(self.tick, self.vol_schedule)
            
            for ticker, data in self.portfolio.items():
                if data['type'] == 'OPTION':
                    # Parse Ticker RTM1C50
                    match = re.match(r"([A-Z]+)(\d+)([CP])(\d+)", ticker)
                    if match:
                        otype = match.group(3)
                        strike = float(match.group(4))
                        pos = data['position']
                        
                        # Calculate Delta
                        unit_delta = QuantEngine.get_delta(otype, self.rtm_price, strike, T, RISK_FREE_RATE, self.rms_vol)
                        total_pos_delta = unit_delta * 100 * pos
                        self.net_delta += total_pos_delta

        # 4. Process News
        self.parse_news()
        
        # 5. Hedge if necessary
        self.check_hedging()

    def parse_news(self):
        news = RITClient.get('news')
        if not news: return
        
        with self.lock:
            for item in news:
                if item['id'] in self.processed_news_ids: continue
                self.processed_news_ids.add(item['id'])
                
                text = (item['headline'] + " " + item['body']).lower()
                
                # A. Parse Volatility Updates
                # "Realized volatility... for this week will be 18%"
                # "Realized volatility... next week will be between 28% and 33%"
                
                # Logic: Current Week
                if "this week" in text and "volatility" in text:
                    pct = re.search(r"(\d+)%", text)
                    if pct:
                        val = float(pct.group(1)) / 100.0
                        week_idx = (self.tick // TICKS_PER_WEEK)
                        if week_idx < 4:
                            self.vol_schedule[week_idx] = val
                            print(f" Updated Week {week_idx+1} Vol to {val:.2%}")

                # Logic: Next Week (Forward Guidance)
                if "next week" in text and "volatility" in text:
                    # Look for range "between 28% and 33%"
                    range_match = re.search(r"between (\d+)% and (\d+)%", text)
                    if range_match:
                        low = float(range_match.group(1))
                        high = float(range_match.group(2))
                        avg = (low + high) / 200.0
                        week_idx = (self.tick // TICKS_PER_WEEK) + 1
                        if week_idx < 4:
                            self.vol_schedule[week_idx] = avg
                            print(f" Updated Week {week_idx+1} Forecast to {avg:.2%}")

                # B. Parse Delta Limits
                # "The delta limit... is 5,000"
                if "delta limit" in text:
                    lim = re.search(r"is ([\d,]+)", text)
                    if lim:
                        val_str = lim.group(1).replace(',', '')
                        self.delta_limit = int(val_str)
                        print(f" Delta Limit Updated to {self.delta_limit}")

    def check_hedging(self):
        """
        Implements the Dynamic Delta Corridor.
        Only hedge if we breach SOFT_LIMIT_BUFFER * DeltaLimit.
        """
        with self.lock:
            current = self.net_delta
            limit = self.delta_limit
            
        soft_limit = limit * SOFT_LIMIT_BUFFER
        
        action = None
        qty = 0
        
        # Hedge Logic
        if current > soft_limit:
            # Positive Delta -> Sell Underlying to reduce
            # Target: Get back to 0 (or safe zone)
            diff = current - HEDGE_TARGET
            qty = abs(int(diff))
            action = "SELL"
            
        elif current < -soft_limit:
            # Negative Delta -> Buy Underlying
            diff = HEDGE_TARGET - current
            qty = abs(int(diff))
            action = "BUY"
            
        if action and qty > 0:
            # Cap order size to 10k (RIT Limit)
            qty = min(qty, 10000)
            print(f" Delta: {current:.0f} | Limit: {limit} | Action: {action} {qty} RTM")
            RITClient.post('orders', {
                'ticker': 'RTM',
                'type': 'MARKET',
                'quantity': qty,
                'action': action
            })

    def execute_max_position(self, direction):
        """
        Calculates remaining capacity in Net Option Limits and fires orders.
        Direction: 'BUY' (Long Vol) or 'SELL' (Short Vol)
        """
        with self.lock:
            # 1. Calculate Current Net Options
            net_opts = 0
            for t, data in self.portfolio.items():
                if data['type'] == 'OPTION':
                    net_opts += data['position']
            
            # 2. Calculate Capacity
            # If Limit is 2000.
            # If we have +500, we can buy 1500 more (to +2000).
            # If we have +500, we can sell 2500 more (to -2000).
            
            if direction == 'BUY':
                capacity = MAX_NET_OPTIONS - net_opts
            else:
                capacity = MAX_NET_OPTIONS + net_opts # e.g. 2000 + 500 = 2500 room to sell
            
            if capacity <= 0:
                print("Max Position Reached.")
                return

            # 3. Find ATM Strike
            S = self.rtm_price
            strike = round(S)
            call_ticker = f"RTM1C{strike}"
            put_ticker = f"RTM1P{strike}"
            
            # A Straddle is 2 contracts (1C + 1P).
            # Capacity is in contracts.
            num_straddles = capacity // 2
            
            print(f"[EXEC] {direction} {num_straddles} Straddles at {strike}")
            
            # Execute in background to keep GUI smooth
            threading.Thread(target=self._send_batch, args=(call_ticker, put_ticker, num_straddles, direction)).start()

    def _send_batch(self, call, put, count, action):
        remaining = count
        while remaining > 0:
            batch = min(remaining, 50) # Small batches
            RITClient.post('orders', {'ticker': call, 'type': 'MARKET', 'quantity': batch, 'action': action})
            RITClient.post('orders', {'ticker': put, 'type': 'MARKET', 'quantity': batch, 'action': action})
            remaining -= batch
            time.sleep(0.05)

# ==============================================================================
# 5. GUI DASHBOARD (Traffic Light Gauge)
# ==============================================================================
class Dashboard:
    def __init__(self, root, strategy):
        self.strategy = strategy
        self.root = root
        self.root.title("RITC Volatility - Delta Corridor")
        self.root.geometry("600x450")
        self.root.configure(bg="#222")

        # -- HEADER --
        lbl_title = tk.Label(root, text="VOLATILITY MATRIX", font=("Arial", 16, "bold"), bg="#222", fg="white")
        lbl_title.pack(pady=10)

        # -- METRICS GRID --
        f_grid = tk.Frame(root, bg="#222")
        f_grid.pack(pady=10)
        
        self.lbl_rms = self._make_metric(f_grid, "RMS Vol", "0.0%", 0, 0, "#f39c12")
        self.lbl_tick = self._make_metric(f_grid, "Tick", "0", 0, 1, "#3498db")
        self.lbl_delta = self._make_metric(f_grid, "Net Delta", "0", 0, 2, "#2ecc71")

        # -- TRAFFIC LIGHT GAUGE --
        tk.Label(root, text="DELTA LIMIT MONITOR", font=("Arial", 10), bg="#222", fg="#aaa").pack(pady=(20,0))
        self.canvas = tk.Canvas(root, width=500, height=100, bg="black", highlightthickness=0)
        self.canvas.pack(pady=5)
        
        # -- CONTROLS --
        f_btn = tk.Frame(root, bg="#222")
        f_btn.pack(pady=20, fill=tk.X)
        
        btn_buy = tk.Button(f_btn, text="MAX LONG VOL\n(BUY)", bg="#27ae60", fg="white", font=("Arial", 12, "bold"),
                            command=lambda: self.strategy.execute_max_position("BUY"), height=3, width=15)
        btn_buy.pack(side=tk.LEFT, padx=30)

        btn_sell = tk.Button(f_btn, text="MAX SHORT VOL\n(SELL)", bg="#c0392b", fg="white", font=("Arial", 12, "bold"),
                             command=lambda: self.strategy.execute_max_position("SELL"), height=3, width=15)
        btn_sell.pack(side=tk.RIGHT, padx=30)

        # Start Update Loop
        self.update_ui()

    def _make_metric(self, parent, title, val, r, c, color):
        f = tk.Frame(parent, bg="#222")
        f.grid(row=r, column=c, padx=20)
        tk.Label(f, text=title, font=("Arial", 9), fg="#888", bg="#222").pack()
        l = tk.Label(f, text=val, font=("Arial", 14, "bold"), fg=color, bg="#222")
        l.pack()
        return l

    def draw_gauge(self, delta, limit):
        self.canvas.delete("all")
        w, h = 500, 100
        cx = w / 2
        
        # Mapping: +/- Limit maps to 80% of canvas width
        # This leaves 20% margin for "breach" visualization
        safe_zone_px = w * 0.4 
        
        # 1. Draw Center Line
        self.canvas.create_line(cx, 10, cx, h-10, fill="#555", dash=(2,2))
        
        # 2. Draw Red Limit Lines
        limit_px = safe_zone_px 
        self.canvas.create_line(cx - limit_px, 0, cx - limit_px, h, fill="red", width=3)
        self.canvas.create_line(cx + limit_px, 0, cx + limit_px, h, fill="red", width=3)
        self.canvas.create_text(cx - limit_px, h-10, text=f"-{limit}", fill="red", anchor="se")
        self.canvas.create_text(cx + limit_px, h-10, text=f"+{limit}", fill="red", anchor="sw")
        
        # 3. Draw Green Delta Line
        # Calculate x pos
        pct = delta / (limit if limit > 0 else 1)
        x_pos = cx + (pct * limit_px)
        
        # Clamp visual
        x_pos = max(5, min(w-5, x_pos))
        
        # Color logic: Green if safe, Orange if close, Red if breach
        col = "#2ecc71"
        if abs(delta) > limit * 0.8: col = "orange"
        if abs(delta) > limit: col = "#e74c3c"
        
        self.canvas.create_line(x_pos, 0, x_pos, h, fill=col, width=5)
        self.canvas.create_text(x_pos, 15, text=f"{int(delta)}", fill=col, font=("Arial", 10, "bold"))

    def update_ui(self):
        # Retrieve data safely
        with self.strategy.lock:
            vol = self.strategy.rms_vol
            tick = self.strategy.tick
            delta = self.strategy.net_delta
            limit = self.strategy.delta_limit
        
        # Update Labels
        self.lbl_rms.config(text=f"{vol:.1%}")
        self.lbl_tick.config(text=str(tick))
        self.lbl_delta.config(text=f"{int(delta)}")
        
        # Redraw Gauge
        self.draw_gauge(delta, limit)
        
        self.root.after(100, self.update_ui)

# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================
def worker_thread(strategy):
    while True:
        try:
            strategy.update_market_data()
            time.sleep(0.2) # 5 Hz Refresh Rate
        except Exception as e:
            print(f"Worker Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    # Initialize Strategy
    strat = Strategy()
    
    # Start Background Thread
    t = threading.Thread(target=worker_thread, args=(strat,), daemon=True)
    t.start()
    
    # Launch GUI
    root = tk.Tk()
    app = Dashboard(root, strat)
    root.mainloop()