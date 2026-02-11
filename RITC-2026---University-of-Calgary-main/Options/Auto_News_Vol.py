import tkinter as tk
from tkinter import ttk
import requests
import threading
import time
import math
import re
from scipy.stats import norm

# ==============================================================================
# CONFIGURATION
# ==============================================================================
API_KEY = 'YOUR_API_KEY_HERE'  # <--- REPLACE WITH YOUR API KEY
API_HOST = 'http://localhost'
API_PORT = '9999'
BASE_URL = f"{API_HOST}:{API_PORT}/v1"

# Case Parameters (Based on RITC Case Package)
RISK_FREE_RATE = 0.0          # r = 0%
TICKS_PER_PERIOD = 300        # Ticks per month/sub-heat
TICKS_PER_YEAR = 5760         # 240 days * minutes/day scaling
CONTRACT_MULTIPLIER = 100     # 1 Option = 100 Shares
MAX_NET_OPT_LIMIT = 1000      # Strict Case Limit (Net contracts)
DEFAULT_VOLATILITY = 0.20     # Starting volatility (20%)
DEFAULT_DELTA_LIMIT = 5000    # Starting delta limit

# ==============================================================================
# SHARED STATE
# ==============================================================================
state_lock = threading.Lock()
market_state = {
    'status': 'UNKNOWN',
    'tick': 0,
    'portfolio': {},
    'rtm_price': 50.0,
    'net_delta': 0.0,
    'delta_limit': DEFAULT_DELTA_LIMIT,
    'forecast_vol': DEFAULT_VOLATILITY,
    'news_read': set(),
    'recommendation': "NEUTRAL",  # BUY or SELL Volatility
    'max_long_qty': 0,
    'max_short_qty': 0
}

# ==============================================================================
# API CLIENT
# ==============================================================================
class RITClient:
    HEADERS = {'X-API-Key': API_KEY}

    @classmethod
    def get(cls, endpoint, params=None):
        try:
            resp = requests.get(f"{BASE_URL}/{endpoint}", headers=cls.HEADERS, params=params, timeout=0.5)
            if resp.status_code == 200: return resp.json()
        except: pass
        return None

    @classmethod
    def post(cls, endpoint, params=None):
        try:
            requests.post(f"{BASE_URL}/{endpoint}", headers=cls.HEADERS, params=params, timeout=0.5)
        except: pass

# ==============================================================================
# QUANT ENGINE (BSM & HEDGING)
# ==============================================================================
class QuantEngine:
    @staticmethod
    def d1(S, K, T, r, sigma):
        if T <= 0.001 or sigma <= 0.001: return 0
        return (math.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def bs_price(option_type, S, K, T, r, sigma):
        d1 = QuantEngine.d1(S, K, T, r, sigma)
        d2 = d1 - sigma * math.sqrt(T)
        if option_type == 'C':
            return S * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2)
        else:
            return K * math.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def calculate_delta(option_type, S, K, T, r, sigma):
        d1 = QuantEngine.d1(S, K, T, r, sigma)
        if option_type == 'C': return norm.cdf(d1)
        if option_type == 'P': return norm.cdf(d1) - 1.0
        return 0.0

    @staticmethod
    def update_greeks():
        with state_lock:
            pf = market_state['portfolio']
            S = market_state['rtm_price']
            sigma = market_state['forecast_vol']
            
            # Time to Maturity (Annualized)
            # Assuming T resets every 300 ticks (one month)
            ticks_left = TICKS_PER_PERIOD - (market_state['tick'] % TICKS_PER_PERIOD)
            T = max(ticks_left, 1) / TICKS_PER_YEAR

            # Calculate Net Delta
            total_delta = 0.0
            
            # 1. Underlying Delta
            if 'RTM' in pf:
                total_delta += pf['pos']

            # 2. Options Delta
            net_opt_contracts = 0
            for ticker, data in pf.items():
                if data['type'] == 'OPTION':
                    net_opt_contracts += data['pos']
                    # Parse Ticker RTM1C50
                    match = re.match(r"([A-Z]+)(\d+)([CP])(\d+)", ticker)
                    if match:
                        otype = match.group(3)
                        strike = float(match.group(4))
                        unit_delta = QuantEngine.calculate_delta(otype, S, strike, T, RISK_FREE_RATE, sigma)
                        total_delta += unit_delta * CONTRACT_MULTIPLIER * data['pos']

            market_state['net_delta'] = total_delta

            # 3. Update Max Position Logic (Smart Limits)
            # We can only hold +/- 1000 Net Contracts
            # Buying increases net contracts, Selling decreases
            cap = MAX_NET_OPT_LIMIT
            current = net_opt_contracts
            
            # Max we can BUY (Long Straddle adds +2 contracts)
            # Limit: current + 2*x <= cap  => 2x <= cap - current
            buy_capacity = (cap - current) // 2
            market_state['max_long_qty'] = max(0, int(buy_capacity))

            # Max we can SELL (Short Straddle subtracts 2 contracts)
            # Limit: current - 2*x >= -cap => -2x >= -cap - current => 2x <= cap + current
            sell_capacity = (cap + current) // 2
            market_state['max_short_qty'] = max(0, int(sell_capacity))

            # 4. Set Direction (Fair Value vs Market)
            # Compare ATM Call Theoretical Price vs Market Ask
            atm_strike = round(S)
            ticker = f"RTM1C{atm_strike}"
            if ticker in pf:
                market_ask = pf[ticker]['ask']
                market_bid = pf[ticker]['bid']
                
                fair_price = QuantEngine.bs_price('C', S, atm_strike, T, RISK_FREE_RATE, sigma)
                
                # Signal Generation
                if fair_price > market_ask + 0.05: # Buffer
                    market_state['recommendation'] = "BUY VOL (Undervalued)"
                elif fair_price < market_bid - 0.05:
                    market_state['recommendation'] = "SELL VOL (Overvalued)"
                else:
                    market_state['recommendation'] = "NEUTRAL"

# ==============================================================================
# ALGORITHM LOGIC (NEWS & EXECUTION)
# ==============================================================================
class AlgoLogic:
    @staticmethod
    def process_news(news_items):
        for item in news_items:
            nid = item['id']
            if nid in market_state['news_read']: continue
            market_state['news_read'].add(nid)
            
            text = f"{item['headline']} {item['body']}"
            
            # 1. Volatility Updates
            # "The realized volatility of RTM for this week will be 20%"
            vol_match = re.search(r"volatility.*?this week.*?(\d+)%", text, re.IGNORECASE)
            if vol_match:
                new_vol = float(vol_match.group(1)) / 100.0
                with state_lock:
                    market_state['forecast_vol'] = new_vol
                print(f" Volatility Updated to {new_vol:.0%}")
                
            # 2. Delta Limit Updates
            # "The delta limit for this sub-heat is 5,000"
            limit_match = re.search(r"delta limit.*?([0-9,]+)", text, re.IGNORECASE)
            if limit_match:
                new_limit = int(limit_match.group(1).replace(',', ''))
                with state_lock:
                    market_state['delta_limit'] = new_limit
                print(f" Delta Limit Updated to {new_limit}")

    @staticmethod
    def run_hedger():
        with state_lock:
            delta = market_state['net_delta']
            limit = market_state['delta_limit']
        
        # Hedge if we are using > 20% of our Delta Limit
        # or if absolute delta > 500 (safety)
        threshold = min(limit * 0.2, 500)
        
        if abs(delta) > threshold:
            # If Delta is Positive (Long), we Sell Stock to hedge
            action = "SELL" if delta > 0 else "BUY"
            qty = min(abs(int(delta)), 5000) # Cap hedge size
            
            if qty > 0:
                print(f" Delta: {int(delta)} | Action: {action} {qty} RTM")
                RITClient.post("orders", {
                    'ticker': 'RTM', 'type': 'MARKET', 
                    'quantity': qty, 'action': action
                })

    @staticmethod
    def execute_straddle(direction):
        """
        Executes MAX allowed position based on calculating capacity
        """
        with state_lock:
            if direction == "BUY":
                qty_straddles = market_state['max_long_qty']
            else:
                qty_straddles = market_state['max_short_qty']
            
            S = market_state['rtm_price']
            
        if qty_straddles <= 0:
            print(" [EXEC] Max Position Reached. No Trade.")
            return

        strike = round(S)
        call = f"RTM1C{strike}"
        put = f"RTM1P{strike}"
        
        print(f" [EXEC] {direction} {qty_straddles} Straddles ({call}/{put})")
        
        # Spawn thread to fill orders without blocking GUI
        t = threading.Thread(target=AlgoLogic._batch_orders, args=(call, put, qty_straddles, direction))
        t.start()

    @staticmethod
    def _batch_orders(call, put, total_qty, direction):
        remaining = total_qty
        # Batch size 100 (Max per order)
        while remaining > 0:
            size = min(remaining, 100)
            RITClient.post("orders", {'ticker': call, 'type': 'MARKET', 'quantity': size, 'action': direction})
            RITClient.post("orders", {'ticker': put, 'type': 'MARKET', 'quantity': size, 'action': direction})
            remaining -= size
            time.sleep(0.05) # Rate limit protection

# ==============================================================================
# GUI DASHBOARD
# ==============================================================================
class VolDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("RITC Volatility Algo")
        self.root.geometry("800x550")
        self.root.configure(bg="#1e1e1e")

        # Styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background="#1e1e1e", foreground="white")
        style.configure("TButton", font=("Arial", 12, "bold"))

        # --- Header ---
        tk.Label(root, text="VOLATILITY COMMANDER", font=("Arial", 20, "bold"), bg="#1e1e1e", fg="white").pack(pady=10)
        
        # --- Info Grid ---
        f_info = tk.Frame(root, bg="#1e1e1e")
        f_info.pack(pady=10, fill=tk.X)
        
        self.lbl_vol = self._metric(f_info, "FORECAST VOL", "0%", "#f1c40f", 0)
        self.lbl_rec = self._metric(f_info, "DIRECTION", "WAIT", "white", 1)
        self.lbl_limit = self._metric(f_info, "DELTA LIMIT", "0", "#e74c3c", 2)

        # --- Visual Traffic Light (Delta Gauge) ---
        tk.Label(root, text="DELTA HEDGING MONITOR", font=("Arial", 10), bg="#1e1e1e", fg="#999").pack(pady=(20, 5))
        self.canvas = tk.Canvas(root, width=700, height=100, bg="black", highlightthickness=0)
        self.canvas.pack()
        
        # --- Controls ---
        f_btn = tk.Frame(root, bg="#1e1e1e")
        f_btn.pack(pady=30, fill=tk.BOTH, expand=True)

        self.btn_buy = tk.Button(f_btn, text="MAX LONG VOL\n(BUY STRADDLE)", bg="#27ae60", fg="white",
                                 font=("Arial", 14, "bold"), relief="flat",
                                 command=lambda: AlgoLogic.execute_straddle("BUY"))
        self.btn_buy.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=20, pady=20)

        self.btn_sell = tk.Button(f_btn, text="MAX SHORT VOL\n(SELL STRADDLE)", bg="#c0392b", fg="white",
                                  font=("Arial", 14, "bold"), relief="flat",
                                  command=lambda: AlgoLogic.execute_straddle("SELL"))
        self.btn_sell.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=20, pady=20)

        self.update_ui()

    def _metric(self, parent, title, val, color, col):
        f = tk.Frame(parent, bg="#1e1e1e")
        f.grid(row=0, column=col, sticky="ew", padx=20)
        tk.Label(f, text=title, font=("Arial", 9), fg="#888", bg="#1e1e1e").pack()
        l = tk.Label(f, text=val, font=("Arial", 18, "bold"), fg=color, bg="#1e1e1e")
        l.pack()
        parent.grid_columnconfigure(col, weight=1)
        return l

    def draw_gauge(self, delta, limit):
        self.canvas.delete("all")
        w, h = 700, 100
        cx = w / 2
        
        # Scaling: Limit is at 80% of width from center
        scale_limit = limit if limit > 0 else 1
        px_per_unit = (w * 0.4) / scale_limit
        
        # 1. Draw Axis
        self.canvas.create_line(0, h/2, w, h/2, fill="#333", width=2)
        self.canvas.create_line(cx, 10, cx, h-10, fill="#555", dash=(2,2)) # Center
        
        # 2. Draw RED Lines (Limits)
        lim_pos_x = cx + limit * px_per_unit
        lim_neg_x = cx - limit * px_per_unit
        
        self.canvas.create_line(lim_pos_x, 0, lim_pos_x, h, fill="red", width=3)
        self.canvas.create_line(lim_neg_x, 0, lim_neg_x, h, fill="red", width=3)
        self.canvas.create_text(lim_pos_x, h-10, text=f"+{limit}", fill="red", anchor="e")
        self.canvas.create_text(lim_neg_x, h-10, text=f"-{limit}", fill="red", anchor="w")

        # 3. Draw GREEN Line (Current Delta)
        delta_x = cx + delta * px_per_unit
        # Clamp visual
        delta_x = max(5, min(w-5, delta_x))
        
        line_col = "#2ecc71" # Green
        if abs(delta) > limit: line_col = "#e74c3c" # Red if breaching
        
        self.canvas.create_line(delta_x, 0, delta_x, h, fill=line_col, width=5)
        self.canvas.create_text(delta_x, 15, text=f"{int(delta)}", fill=line_col, font=("Arial", 10, "bold"))

    def update_ui(self):
        with state_lock:
            vol = market_state['forecast_vol']
            rec = market_state['recommendation']
            d_lim = market_state['delta_limit']
            delta = market_state['net_delta']
            max_long = market_state['max_long_qty']
            max_short = market_state['max_short_qty']

        self.lbl_vol.config(text=f"{vol:.0%}")
        self.lbl_rec.config(text=rec)
        self.lbl_limit.config(text=str(d_lim))
        
        # Color code recommendation
        if "BUY" in rec: self.lbl_rec.config(fg="#2ecc71")
        elif "SELL" in rec: self.lbl_rec.config(fg="#e74c3c")
        else: self.lbl_rec.config(fg="white")
        
        # Update buttons text with dynamic capacity
        self.btn_buy.config(text=f"MAX LONG VOL\n(+{max_long} Straddles)")
        self.btn_sell.config(text=f"MAX SHORT VOL\n(-{max_short} Straddles)")

        self.draw_gauge(delta, d_lim)
        self.root.after(100, self.update_ui)

# ==============================================================================
# WORKER THREAD
# ==============================================================================
def worker():
    while True:
        try:
            # 1. Get Case Status
            case = RITClient.get("case")
            if not case or case['status']!= 'ACTIVE':
                time.sleep(1)
                continue
            
            with state_lock: market_state['tick'] = case['tick']

            # 2. Get Securities (Price & Positions)
            secs = RITClient.get("securities")
            if secs:
                with state_lock:
                    for s in secs:
                        market_state['portfolio'][s['ticker']] = s
                        if s['ticker'] == 'RTM':
                            market_state['rtm_price'] = s['last']

            # 3. Process News (Extract Volatility & Limits)
            news = RITClient.get("news")
            if news:
                AlgoLogic.process_news(news)

            # 4. Math & Hedging
            QuantEngine.update_greeks()
            AlgoLogic.run_hedger()

            time.sleep(0.2)
        except Exception as e:
            print(e)
            time.sleep(1)

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    
    root = tk.Tk()
    app = VolDashboard(root)
    root.mainloop()