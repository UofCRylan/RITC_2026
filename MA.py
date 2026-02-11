import tkinter as tk
from tkinter import messagebox
import requests
import threading
import time

# ==========================================
# CONFIGURATION & CASE LIMITS
# ==========================================
API_URL = "http://localhost:9999/v1"
API_KEY = {'X-API-Key': 'YOUR_API_KEY'} 

# Tickers from Merger Arbitrage Case [cite: 603]
TICKERS = ['PHR', 'TGX', 'CLD', 'BYL', 'PNR', 'GGD', 'ATB', 'FSR', 'SPK', 'EEC']

# Case Constraints
MAX_ORDER_SIZE = 5000        # Max order size per API call 
HARD_POSITION_LIMIT = 50000  # Hard cap per ticker (Net Limit is 50k) 

class MergerArbGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RITC 2026 - Merger Arb Execution")
        self.root.geometry("450x400")
        
        # Title
        title = tk.Label(root, text="Merger Arbitrage Algo", font=("Helvetica", 14, "bold"))
        title.pack(pady=10)

        # Safety Info
        info_text = (f"Hard Limit Per Ticker: {HARD_POSITION_LIMIT} shares\n"
                     f"Case Net Limit: 50k | Case Gross Limit: 100k")
        lbl_info = tk.Label(root, text=info_text, fg="blue", font=("Arial", 10))
        lbl_info.pack(pady=5)

        # === CONTROLS ===
        
        # Max Long Button
        btn_long = tk.Button(root, text=f"GO MAX LONG (+{HARD_POSITION_LIMIT})", 
                             bg="#4caf50", fg="white", font=("Arial", 12, "bold"),
                             command=lambda: self.run_threaded(self.go_max_long_all))
        btn_long.pack(fill="x", padx=20, pady=10)

        # Max Short Button
        btn_short = tk.Button(root, text=f"GO MAX SHORT (-{HARD_POSITION_LIMIT})", 
                              bg="#f44336", fg="white", font=("Arial", 12, "bold"),
                              command=lambda: self.run_threaded(self.go_max_short_all))
        btn_short.pack(fill="x", padx=20, pady=10)

        # Flatten Button
        btn_flat = tk.Button(root, text="FLATTEN ALL POSITIONS (0)", 
                             bg="#2196f3", fg="white", font=("Arial", 12, "bold"),
                             command=lambda: self.run_threaded(self.flatten_all))
        btn_flat.pack(fill="x", padx=20, pady=10)

        # Status Log
        self.status_var = tk.StringVar()
        self.status_var.set("System Ready")
        lbl_status = tk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        lbl_status.pack(side=tk.BOTTOM, fill="x")

    def run_threaded(self, func):
        threading.Thread(target=func).start()

    def get_position(self, ticker):
        """Gets current position for a ticker from the RIT API."""
        try:
            resp = requests.get(f"{API_URL}/securities", params={'ticker': ticker}, headers=API_KEY)
            if resp.status_code == 200:
                data = resp.json()
                if len(data) > 0:
                    return data[0]['position']
            return 0
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return 0

    def execute_order_flow(self, ticker, quantity):
        """
        Executes a trade for 'quantity' shares, breaking it down into 
        chunks of 5,000 to respect case order limits.
        """
        if quantity == 0:
            return

        action = "BUY" if quantity > 0 else "SELL"
        qty_remaining = abs(quantity)
        
        self.status_var.set(f"Processing {ticker}: {action} {qty_remaining}...")
        
        while qty_remaining > 0:
            # Chunk size is min(remaining, 5000)
            batch_size = min(qty_remaining, MAX_ORDER_SIZE)
            
            params = {
                'ticker': ticker,
                'type': 'MARKET',
                'quantity': batch_size,
                'action': action
            }
            
            try:
                resp = requests.post(f"{API_URL}/orders", params=params, headers=API_KEY)
                if resp.status_code == 200:
                    qty_remaining -= batch_size
                    time.sleep(0.02) # Micro-sleep to prevent rate limiting
                else:
                    self.status_var.set(f"API Error on {ticker}: {resp.text}")
                    print(f"Failed to place order: {resp.text}")
                    break # Stop if API rejects (likely hit global limits)
            except Exception as e:
                print(f"Request failed: {e}")
                break

    def go_max_long_all(self):
        """Attempts to buy up to exactly 50,000 shares for every ticker."""
        self.status_var.set("Starting Max Long execution...")
        for ticker in TICKERS:
            current_pos = self.get_position(ticker)
            
            # HARD LIMIT CHECK: Calculate strictly what is needed to reach 50,000
            if current_pos < HARD_POSITION_LIMIT:
                qty_needed = HARD_POSITION_LIMIT - current_pos
                self.execute_order_flow(ticker, qty_needed)
            else:
                print(f"{ticker} is already at or above limit ({current_pos}). Skipping.")
        
        self.status_var.set("Max Long execution finished.")

    def go_max_short_all(self):
        """Attempts to sell down to exactly -50,000 shares for every ticker."""
        self.status_var.set("Starting Max Short execution...")
        for ticker in TICKERS:
            current_pos = self.get_position(ticker)
            
            # HARD LIMIT CHECK: Calculate strictly what is needed to reach -50,000
            target = -HARD_POSITION_LIMIT
            if current_pos > target:
                qty_needed = target - current_pos # This will be negative (Sell)
                self.execute_order_flow(ticker, qty_needed)
            else:
                print(f"{ticker} is already at or below limit ({current_pos}). Skipping.")
        
        self.status_var.set("Max Short execution finished.")

    def flatten_all(self):
        """Liquidates all positions to 0."""
        self.status_var.set("Flattening all positions...")
        for ticker in TICKERS:
            current_pos = self.get_position(ticker)
            
            if current_pos != 0:
                # Trade opposite to current position
                qty_needed = -current_pos 
                self.execute_order_flow(ticker, qty_needed)
        
        self.status_var.set("All positions flattened.")

if __name__ == "__main__":
    root = tk.Tk()
    app = MergerArbGUI(root)
    root.mainloop()