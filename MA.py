import tkinter as tk
from tkinter import ttk
import requests
import threading
import concurrent.futures
import time

# ==========================================
# CONFIGURATION
# ==========================================
API_URL = "http://localhost:10000/v1"
API_KEY = {'X-API-Key': 'RYL2000'} 

# Tickers from Merger Arbitrage Case [cite: 603]
TICKERS = ['PHR', 'TGX', 'CLD', 'BYL', 'PNR', 'GGD', 'ATB', 'FSR', 'SPK', 'EEC']

# Case Limits
MAX_ORDER_SIZE = 5000       # 
TARGET_LIMIT = 50000        # User requested max limit

class TradeExecutor:
    """
    Handles the actual logic of calculating trade sizes, chunking orders,
    and sending them to the RIT API.
    """
    def __init__(self, update_status_callback):
        self.update_status = update_status_callback

    def get_position(self, ticker):
        """Fetches the current position for a specific ticker."""
        try:
            resp = requests.get(f"{API_URL}/securities", params={'ticker': ticker}, headers=API_KEY, timeout=0.5)
            if resp.status_code == 200:
                data = resp.json()
                # Find the specific ticker in the list
                for item in data:
                    if item['ticker'] == ticker:
                        return item['position']
            return 0
        except Exception as e:
            print(f"Error fetching position for {ticker}: {e}")
            return 0

    def post_order(self, ticker, action, quantity):
        """Sends a single order to the API."""
        params = {
            'ticker': ticker,
            'type': 'MARKET',
            'quantity': quantity,
            'action': action
        }
        try:
            resp = requests.post(f"{API_URL}/orders", params=params, headers=API_KEY)
            if resp.status_code != 200:
                print(f"Order Rejected ({ticker}): {resp.text}")
        except Exception as e:
            print(f"API Request Failed: {e}")

    def execute_to_target(self, ticker, target_position):
        """
        The CORE LOGIC:
        1. Checks current position.
        2. Calculates shares needed to reach target.
        3. Splits needed shares into 5,000 chunks.
        4. Fires all chunks in parallel.
        """
        current_pos = self.get_position(ticker)
        needed = target_position - current_pos

        if needed == 0:
            self.update_status(f"{ticker}: Position correct at {current_pos}.")
            return

        action = "BUY" if needed > 0 else "SELL"
        total_qty = abs(needed)
        
        self.update_status(f"{ticker}: {action} {total_qty} shares...")

        # --- LOGIC: CHUNK ORDERS ---
        orders = []
        remaining = total_qty
        while remaining > 0:
            # We can only send max 5,000 at a time 
            chunk_size = min(remaining, MAX_ORDER_SIZE)
            orders.append(chunk_size)
            remaining -= chunk_size

        # --- LOGIC: PARALLEL EXECUTION ---
        # We use a ThreadPool to send all API requests at the exact same time
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for qty in orders:
                futures.append(executor.submit(self.post_order, ticker, action, qty))
            
            # Wait for all orders to be sent
            concurrent.futures.wait(futures)
        
        self.update_status(f"{ticker}: Execution of {total_qty} shares complete.")

    def flatten_portfolio(self):
        """Logic to flatten ALL tickers simultaneously."""
        self.update_status("FLATTENING ALL POSITIONS...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(TICKERS)) as executor:
            futures = []
            for ticker in TICKERS:
                # We execute to target 0 for every ticker
                futures.append(executor.submit(self.execute_to_target, ticker, 0))
            
            concurrent.futures.wait(futures)
        
        self.update_status("ALL POSITIONS FLATTENED.")


class MergerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RITC 2026 - Merger Execution")
        self.root.geometry("600x500")
        
        # Initialize the logic engine
        self.executor = TradeExecutor(self.update_status_label)

        # GUI Layout
        self.create_ui()

    def update_status_label(self, message):
        # Helper to update GUI from the logic thread
        self.status_var.set(message)

    def run_async(self, func, *args):
        # Runs the logic in a background thread so GUI doesn't freeze
        threading.Thread(target=func, args=args).start()

    def create_ui(self):
        # Styles
        style = ttk.Style()
        style.configure("Header.TLabel", font=("Arial", 10, "bold"))
        
        # Header
        header_frame = tk.Frame(self.root)
        header_frame.pack(fill="x", pady=10)
        tk.Label(header_frame, text="Ticker", width=10, font=("Arial", 10, "bold")).grid(row=0, column=0)
        tk.Label(header_frame, text="Max Buy (+50k)", width=20, font=("Arial", 10, "bold")).grid(row=0, column=1)
        tk.Label(header_frame, text="Max Sell (-50k)", width=20, font=("Arial", 10, "bold")).grid(row=0, column=2)

        # Scrollable Frame for Tickers
        container = tk.Frame(self.root)
        container.pack(fill="both", expand=True)
        
        # Grid of Tickers
        for i, ticker in enumerate(TICKERS):
            row_frame = tk.Frame(container, pady=2)
            row_frame.pack(fill="x")
            
            # Ticker Name
            tk.Label(row_frame, text=ticker, width=10, font=("Arial", 11, "bold"), bg="#ddd", relief="raised").pack(side="left", padx=5)
            
            # Buy Button
            btn_buy = tk.Button(row_frame, text="BUY MAX", bg="#4caf50", fg="white", width=18,
                                command=lambda t=ticker: self.run_async(self.executor.execute_to_target, t, TARGET_LIMIT))
            btn_buy.pack(side="left", padx=5)
            
            # Sell Button
            btn_sell = tk.Button(row_frame, text="SELL MAX", bg="#f44336", fg="white", width=18,
                                 command=lambda t=ticker: self.run_async(self.executor.execute_to_target, t, -TARGET_LIMIT))
            btn_sell.pack(side="left", padx=5)

        # Global Flatten Button
        bottom_frame = tk.Frame(self.root, pady=10, bg="#333")
        bottom_frame.pack(fill="x", side="bottom")
        
        btn_flat = tk.Button(bottom_frame, text="FLATTEN ALL TICKERS", bg="black", fg="white", font=("Arial", 12, "bold"),
                             command=lambda: self.run_async(self.executor.flatten_portfolio))
        btn_flat.pack(fill="x", padx=20, pady=5)

        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        tk.Label(self.root, textvariable=self.status_var, anchor="w", relief="sunken").pack(side="bottom", fill="x")

if __name__ == "__main__":
    root = tk.Tk()
    app = MergerGUI(root)
    root.mainloop()