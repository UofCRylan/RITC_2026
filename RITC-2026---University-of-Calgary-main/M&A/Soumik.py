import pandas as pd
import numpy as np
from pathlib import Path
import requests
import time
import string
import threading
import tkinter as tk
from tkinter import ttk, font as tkfont
from collections import defaultdict


# =========================
# CONFIG
# =========================
WORD_SENTIMENT_CSV = Path(r"C:\Users\rylan\Desktop\M&A\word_sentiment_scores.csv")

# RIT API Configuration
RIT_BASE_URL = "http://localhost:9995/v1"

# Position sizes
STANDARD_SIZE = 50000  # For regular Buy/Sell buttons
MAX_SIZE = 75000       # For Max Long/Max Short buttons
HEDGE_SIZE = 25000
MAX_ORDER_SIZE = 5000

# Prediction Thresholds
SELL_THRESHOLD = -0.001    # Below -0.1% = SELL
HOLD_MIN = -0.001          # -0.1% to +0.2% = HOLD
BUY_THRESHOLD = 0.002      # +0.2% to +0.6% = BUY
TOO_HIGH_THRESHOLD = 0.006 # Above +0.6% = TOO HIGH (RED - Not a buy)

# Word sentiment scoring
MIN_WORD_MATCHES = 2
WORD_WEIGHT_BY_COUNT = True

STOPWORDS = {
    "the","and","of","for","to","a","an","in","on","with","has","have","had",
    "is","are","be","been","this","that","as","at","by","from","over","into",
    "will","would","should","could","may","might","can","its","it","their",
    "them","they","we","our","your","you","he","she","him","her","not","but",
    "or","no","so","if","do","did","does","was","were","being","been","also",
    "than","more","most","some","any","all","each","both","few","other","such",
    "only","own","same","very","just","about","between","through","during"
}
PUNCT_TRANS = str.maketrans({c: " " for c in string.punctuation})

ALL_TICKERS = ["TGX", "PHR", "BYL", "CLD", "GGD", "PNR", "FSR", "ATB", "SPK", "EEC"]

# ALL TICKERS CAN NOW BE BOUGHT/LONGED
ALLOWED_LONG_TICKERS = set(ALL_TICKERS)  # All tickers enabled

# PREFERRED: Prioritize shorting these tickers as hedges over TGX
PREFERRED_SHORT_TICKERS = {"PNR", "PHR", "EEC"}

DEAL_PAIRS = {
    "TGX": "PHR", "PHR": "TGX",
    "BYL": "CLD", "CLD": "BYL",
    "GGD": "PNR", "PNR": "GGD",
    "FSR": "ATB", "ATB": "FSR",
    "SPK": "EEC", "EEC": "SPK"
}

# UI Colors
COL_BG = "#0d1117"
COL_CARD = "#161b22"
COL_BORDER = "#21262d"
COL_TEXT = "#c9d1d9"
COL_DIM = "#484f58"
COL_WHITE = "#ffffff"
COL_YELLOW = "#d29922"
COL_NEUTRAL = "#30363d"
COL_ORANGE = "#ff9500"  # For HOLD signal

# NEON Button Colors - Vibrant and high contrast with white text
COL_NEON_GREEN = "#00ff41"      # Bright neon green for BUY
COL_NEON_RED = "#ff0055"        # Bright neon red for SELL
COL_NEON_GOLD = "#ffd700"       # Bright gold/yellow for MAX LONG
COL_NEON_BLUE = "#00bfff"       # Bright neon blue for MAX SHORT
COL_NEON_MAGENTA = "#ff00ff"    # Neon magenta for FLATTEN

# Hover/Active colors (slightly brighter)
COL_NEON_GREEN_HOVER = "#33ff66"
COL_NEON_RED_HOVER = "#ff3377"
COL_NEON_GOLD_HOVER = "#ffed4e"
COL_NEON_BLUE_HOVER = "#33ccff"
COL_NEON_MAGENTA_HOVER = "#ff33ff"


class WordSentimentScorer:
    """Loads word sentiment scores and predicts news impact via word matching."""
    
    def __init__(self, csv_path):
        print(f"Loading word sentiment scores from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        print(f"  Loaded {len(self.df)} word-sentiment mappings")
        
        self.word_sentiments = {}
        for _, row in self.df.iterrows():
            word = str(row['word']).lower().strip()
            self.word_sentiments[word] = {
                'avg_effect': row['avg_effect'],
                'std_effect': row['std_effect'],
                'count': row['count'],
                'total_impact': row['total_impact']
            }
        
        print(f"  Indexed {len(self.word_sentiments)} sentiment words")
    
    def _tokenize(self, text):
        if pd.isna(text):
            return []
        return [
            w for w in str(text).lower().translate(PUNCT_TRANS).split()
            if w and w not in STOPWORDS and not w.isdigit() and len(w) > 2
        ]
    
    def predict_impact(self, ticker, headline, body=""):
        """Predict impact for a ticker based on word sentiments in the news text."""
        text = f"{headline} {body}"
        tokens = self._tokenize(text)
        
        if not tokens:
            return 0.0
        
        matched_words = []
        for word in tokens:
            if word in self.word_sentiments:
                s = self.word_sentiments[word]
                matched_words.append({
                    'word': word,
                    'effect': s['avg_effect'],
                    'count': s['count'],
                    'std': s['std_effect']
                })
        
        if len(matched_words) < MIN_WORD_MATCHES:
            return 0.0
        
        if WORD_WEIGHT_BY_COUNT:
            total_w = sum(w['count'] for w in matched_words)
            if total_w > 0:
                impact = sum(w['effect'] * w['count'] for w in matched_words) / total_w
            else:
                impact = 0.0
        else:
            impact = np.mean([w['effect'] for w in matched_words])
        
        return impact


class RITClient:
    def __init__(self, apikey, base="http://localhost:9999/v1"):
        self.base = base
        self.s = requests.Session()
        self.s.headers.update({"X-API-Key": apikey})
    
    def get(self, ep, params=None):
        try:
            r = self.s.get(f"{self.base}{ep}", params=params, timeout=2)
            r.raise_for_status()
            return r.json()
        except:
            return None
    
    def post(self, ep, params=None):
        try:
            r = self.s.post(f"{self.base}{ep}", params=params, timeout=2)
            r.raise_for_status()
            return r.json()
        except:
            return None
    
    def case(self):
        return self.get("/case") or {}
    
    def prices(self):
        data = self.get("/securities")
        if not data:
            return {}
        return {s.get("ticker"): {
            "last": s.get("last", 0),
            "bid": s.get("bid", 0),
            "ask": s.get("ask", 0),
            "position": s.get("position", 0)
        } for s in data if "ticker" in s}
    
    def news(self, since=0):
        data = self.get("/news", {"since": since, "limit": 20})
        return data if isinstance(data, list) else []
    
    def submit_single_order(self, ticker, quantity, action):
        """Submit a single order chunk."""
        params = {
            "ticker": ticker,
            "type": "MARKET",
            "quantity": int(abs(quantity)),
            "action": action
        }
        
        result = self.post("/orders", params=params)
        return result is not None
    
    def submit_order_fast(self, ticker, quantity):
        """Submit order with parallel execution - ALL chunks fire simultaneously."""
        if quantity == 0:
            return 0
            
        action = "BUY" if quantity > 0 else "SELL"
        total_qty = int(abs(quantity))
        
        # Calculate number of chunks needed
        num_chunks = (total_qty + MAX_ORDER_SIZE - 1) // MAX_ORDER_SIZE
        
        # Create threads for each chunk
        threads = []
        for i in range(num_chunks):
            chunk_size = min(MAX_ORDER_SIZE, total_qty - i * MAX_ORDER_SIZE)
            t = threading.Thread(
                target=self.submit_single_order,
                args=(ticker, chunk_size, action),
                daemon=True
            )
            threads.append(t)
        
        # Fire all orders simultaneously
        for t in threads:
            t.start()
        
        # Wait for all to complete (but don't block UI)
        for t in threads:
            t.join(timeout=1)
        
        return quantity


class DashboardApp:
    def __init__(self, root, apikey, scorer):
        self.root = root
        self.root.title("RITC 2026 - Manual Merger Arb Trading")
        self.root.configure(bg=COL_BG)
        self.root.geometry("1200x850")
        
        self.scorer = scorer
        self.newslog = []
        self.tick = 0
        self.polling = False
        self.last_nid = 0
        self.seen_nids = set()
        self.rit = None
        
        # Ticker predictions
        self.predictions = {ticker: 0.0 for ticker in ALL_TICKERS}
        self.previous_predictions = {ticker: 0.0 for ticker in ALL_TICKERS}  # Track previous for delta
        self.positions = {ticker: 0 for ticker in ALL_TICKERS}
        
        # Fonts
        self.fn_mono = tkfont.Font(family="Consolas", size=10)
        self.fn_mono_sm = tkfont.Font(family="Consolas", size=9)
        self.fn_mono_lg = tkfont.Font(family="Consolas", size=13, weight="bold")
        self.fn_title = tkfont.Font(family="Consolas", size=11, weight="bold")
        self.fn_btn = tkfont.Font(family="Consolas", size=10, weight="bold")
        self.fn_header = tkfont.Font(family="Consolas", size=16, weight="bold")
        
        self.build_ui(apikey)
    
    def get_signal(self, prediction):
        """Determine BUY/HOLD/SELL signal based on prediction thresholds."""
        if prediction < SELL_THRESHOLD:
            # Below -0.1%
            return "SELL", COL_NEON_RED
        elif prediction > TOO_HIGH_THRESHOLD:
            # Above +0.6% - TOO HIGH, not a buy
            return "HIGH", COL_NEON_RED
        elif prediction > BUY_THRESHOLD:
            # Between +0.2% and +0.6%
            return "BUY", COL_NEON_GREEN
        else:
            # Between -0.1% and +0.2%
            return "HOLD", COL_ORANGE
    
    def build_ui(self, apikey):
        # Header
        hdr = tk.Frame(self.root, bg=COL_BG)
        hdr.pack(fill="x", padx=12, pady=(10, 4))
        
        tk.Label(hdr, text="MERGER ARB TRADER", font=self.fn_header, fg=COL_WHITE, bg=COL_BG).pack(side="left")
        tk.Label(hdr, text="Manual Trading", font=self.fn_mono_sm, fg=COL_DIM, bg=COL_BG).pack(side="left", padx=(10, 0))
        
        self.lbl_status = tk.Label(hdr, text="⚫ DISCONNECTED", font=self.fn_mono_sm, fg=COL_NEON_RED, bg=COL_BG)
        self.lbl_status.pack(side="right", padx=(10, 0))
        
        self.lbl_tick = tk.Label(hdr, text="TICK: 0", font=self.fn_mono_sm, fg=COL_DIM, bg=COL_BG)
        self.lbl_tick.pack(side="right")
        
        # Connection bar
        conn = tk.Frame(self.root, bg=COL_BG)
        conn.pack(fill="x", padx=12, pady=(4, 8))
        
        tk.Label(conn, text="API Key:", font=self.fn_mono_sm, fg=COL_DIM, bg=COL_BG).pack(side="left")
        
        self.ent_key = tk.Entry(conn, font=self.fn_mono_sm, bg=COL_CARD, fg=COL_TEXT, 
                                insertbackground=COL_TEXT, relief="flat", width=25)
        self.ent_key.insert(0, apikey)
        self.ent_key.pack(side="left", padx=(6, 6))
        
        self.btn_connect = tk.Button(conn, text="CONNECT", font=self.fn_mono_sm, fg=COL_WHITE, 
                                      bg=COL_NEON_GREEN, activebackground=COL_NEON_GREEN_HOVER, relief="flat", 
                                      padx=12, command=self.toggle_connect)
        self.btn_connect.pack(side="left")
        
        # Info label - now with all thresholds
        tk.Label(conn, text=f"⚙️ SELL<-0.1% | HOLD:-0.1%~+0.2% | BUY:+0.2%~+0.6% | HIGH>+0.6% | Hedge:PNR,PHR,EEC>TGX", 
                font=self.fn_mono_sm, fg=COL_YELLOW, bg=COL_BG).pack(side="left", padx=(20, 0))
        
        # Flatten All button - INSTANT 1-CLICK WITH VERIFICATION LOOP
        self.btn_flatten = tk.Button(conn, text="🔴 FLATTEN ALL", font=self.fn_title, fg=COL_WHITE, 
                                      bg=COL_NEON_MAGENTA, activebackground=COL_NEON_MAGENTA_HOVER, relief="flat", 
                                      padx=20, pady=8, command=self.flatten_all, state="disabled")
        self.btn_flatten.pack(side="right")
        
        tk.Frame(self.root, bg=COL_BORDER, height=1).pack(fill="x", padx=12)
        
        # Ticker cards
        self.cards_frame = tk.Frame(self.root, bg=COL_BG)
        self.cards_frame.pack(fill="both", expand=True, padx=12, pady=10)
        
        self.ticker_widgets = {}
        
        for i, ticker in enumerate(ALL_TICKERS):
            row = i // 5
            col = i % 5
            
            card = tk.Frame(self.cards_frame, bg=COL_CARD, highlightbackground=COL_BORDER, 
                            highlightthickness=1, padx=10, pady=8)
            card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
            
            self.cards_frame.rowconfigure(row, weight=1)
            self.cards_frame.columnconfigure(col, weight=1)
            
            # Header
            hdr_f = tk.Frame(card, bg=COL_CARD)
            hdr_f.pack(fill="x")
            
            tk.Label(hdr_f, text=ticker, font=self.fn_title, fg=COL_WHITE, bg=COL_CARD).pack(side="left")
            
            pair = DEAL_PAIRS.get(ticker, "")
            tk.Label(hdr_f, text=f"↔ {pair}", font=self.fn_mono_sm, fg=COL_DIM, bg=COL_CARD).pack(side="right")
            
            # Signal (BUY/HOLD/SELL/HIGH)
            signal_label = tk.Label(card, text="HOLD", font=self.fn_mono_lg, fg=COL_ORANGE, bg=COL_CARD)
            signal_label.pack(pady=(8, 2))
            
            # Prediction percentage
            pred_label = tk.Label(card, text="0.00%", font=self.fn_mono, fg=COL_DIM, bg=COL_CARD)
            pred_label.pack(pady=(0, 4))
            
            # Position
            pos_label = tk.Label(card, text="Position: 0", font=self.fn_mono, fg=COL_TEXT, bg=COL_CARD)
            pos_label.pack(pady=(0, 4))
            
            # Price
            price_label = tk.Label(card, text="Price: $0.00", font=self.fn_mono, fg=COL_TEXT, bg=COL_CARD)
            price_label.pack(pady=(0, 8))
            
            # Buy/Sell buttons (50k each) - NEON COLORS with WHITE TEXT
            btn_frame = tk.Frame(card, bg=COL_CARD)
            btn_frame.pack(fill="x", pady=(0, 4))
            
            # Buy button - NEON GREEN with WHITE TEXT
            btn_buy = tk.Button(btn_frame, text=f"🟢 BUY 50k", font=self.fn_btn, fg=COL_WHITE, 
                                bg=COL_NEON_GREEN, activebackground=COL_NEON_GREEN_HOVER, relief="flat", 
                                width=10, pady=6, state="disabled",
                                command=lambda t=ticker: self.trade_buy(t, STANDARD_SIZE))
            btn_buy.pack(side="left", fill="x", expand=True, padx=(0, 2))
            
            # Sell button - NEON RED with WHITE TEXT
            btn_sell = tk.Button(btn_frame, text=f"🔴 SELL 50k", font=self.fn_btn, fg=COL_WHITE, 
                                 bg=COL_NEON_RED, activebackground=COL_NEON_RED_HOVER, relief="flat", 
                                 width=10, pady=6, state="disabled",
                                 command=lambda t=ticker: self.trade_sell(t, STANDARD_SIZE))
            btn_sell.pack(side="right", fill="x", expand=True, padx=(2, 0))
            
            # Max Long / Max Short buttons (75k each) - NEON COLORS with WHITE TEXT
            max_frame = tk.Frame(card, bg=COL_CARD)
            max_frame.pack(fill="x")
            
            # Max Long button - NEON GOLD with WHITE TEXT
            btn_max_long = tk.Button(max_frame, text="📈 MAX LONG\n75k", font=self.fn_btn, fg=COL_WHITE, 
                                     bg=COL_NEON_GOLD, activebackground=COL_NEON_GOLD_HOVER, relief="flat", 
                                     width=10, pady=4, state="disabled",
                                     command=lambda t=ticker: self.trade_buy(t, MAX_SIZE))
            btn_max_long.pack(side="left", fill="x", expand=True, padx=(0, 2))
            
            # Max Short button - NEON BLUE with WHITE TEXT
            btn_max_short = tk.Button(max_frame, text="📉 MAX SHORT\n75k", font=self.fn_btn, fg=COL_WHITE, 
                                      bg=COL_NEON_BLUE, activebackground=COL_NEON_BLUE_HOVER, relief="flat", 
                                      width=10, pady=4, state="disabled",
                                      command=lambda t=ticker: self.trade_sell(t, MAX_SIZE))
            btn_max_short.pack(side="right", fill="x", expand=True, padx=(2, 0))
            
            self.ticker_widgets[ticker] = {
                "card": card,
                "signal_label": signal_label,
                "pred_label": pred_label,
                "pos_label": pos_label,
                "price_label": price_label,
                "btn_buy": btn_buy,
                "btn_sell": btn_sell,
                "btn_max_long": btn_max_long,
                "btn_max_short": btn_max_short
            }
        
        tk.Frame(self.root, bg=COL_BORDER, height=1).pack(fill="x", padx=12, pady=(0, 4))
        
        # News feed
        nf_hdr = tk.Frame(self.root, bg=COL_BG)
        nf_hdr.pack(fill="x", padx=12)
        
        tk.Label(nf_hdr, text="NEWS FEED", font=self.fn_title, fg=COL_DIM, bg=COL_BG).pack(side="left")
        
        self.lbl_news_count = tk.Label(nf_hdr, text="0 events", font=self.fn_mono_sm, fg=COL_DIM, bg=COL_BG)
        self.lbl_news_count.pack(side="right")
        
        nf = tk.Frame(self.root, bg=COL_BG)
        nf.pack(fill="both", expand=True, padx=12, pady=(4, 10))
        
        self.news_text = tk.Text(nf, font=self.fn_mono_sm, bg=COL_CARD, fg=COL_TEXT, relief="flat", 
                                 wrap="word", height=6, padx=8, pady=6, state="disabled", cursor="arrow")
        
        sb = tk.Scrollbar(nf, command=self.news_text.yview)
        self.news_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.news_text.pack(side="left", fill="both", expand=True)
        
        self.news_text.tag_configure("green", foreground=COL_NEON_GREEN)
        self.news_text.tag_configure("red", foreground=COL_NEON_RED)
        self.news_text.tag_configure("dim", foreground=COL_DIM)
        self.news_text.tag_configure("white", foreground=COL_WHITE)
        self.news_text.tag_configure("yellow", foreground=COL_YELLOW)
    
    def choose_hedge_ticker(self, primary_ticker):
        """
        Choose optimal hedge ticker with preference for PNR, PHR, EEC over TGX.
        """
        # Exclude the primary ticker and its deal pair
        pair = DEAL_PAIRS.get(primary_ticker)
        exclude = {primary_ticker}
        if pair:
            exclude.add(pair)
        
        available = [t for t in ALL_TICKERS if t not in exclude]
        
        if not available:
            return None
        
        # Separate into preferred and other tickers
        preferred_available = [t for t in available if t in PREFERRED_SHORT_TICKERS]
        other_available = [t for t in available if t not in PREFERRED_SHORT_TICKERS]
        
        # Prioritize preferred tickers
        if preferred_available:
            hedge_ticker = min(preferred_available, key=lambda t: self.predictions.get(t, 0))
            reason = "PREFERRED"
        else:
            hedge_ticker = min(other_available, key=lambda t: self.predictions.get(t, 0))
            reason = "FALLBACK"
        
        return hedge_ticker, reason
    
    def toggle_connect(self):
        if self.polling:
            self.polling = False
            self.btn_connect.config(text="CONNECT", bg=COL_NEON_GREEN)
            self.lbl_status.config(text="⚫ DISCONNECTED", fg=COL_NEON_RED)
            self.disable_trading_buttons()
        else:
            key = self.ent_key.get().strip()
            if not key:
                self.lbl_status.config(text="⚠ NO API KEY", fg=COL_YELLOW)
                return
            
            self.rit = RITClient(key)
            self.polling = True
            self.btn_connect.config(text="DISCONNECT", bg=COL_NEON_RED)
            self.lbl_status.config(text="🟡 CONNECTING...", fg=COL_YELLOW)
            
            t = threading.Thread(target=self.poll_loop, daemon=True)
            t.start()
    
    def poll_loop(self):
        while self.polling:
            try:
                case = self.rit.case()
                status = case.get("status", "?")
                self.tick = case.get("tick", 0)
                
                self.root.after(0, self.lbl_tick.config, {"text": f"TICK: {self.tick}"})
                
                if status != "ACTIVE":
                    self.root.after(0, self.lbl_status.config, {"text": f"🟡 {status}", "fg": COL_YELLOW})
                    time.sleep(1)
                    continue
                
                self.root.after(0, self.lbl_status.config, {"text": "🟢 LIVE", "fg": COL_NEON_GREEN})
                self.root.after(0, self.enable_trading_buttons)
                
                # Update prices and positions
                prices = self.rit.prices()
                for ticker in ALL_TICKERS:
                    if ticker in prices:
                        price_data = prices[ticker]
                        price = price_data["last"]
                        position = price_data["position"]
                        
                        self.positions[ticker] = position
                        
                        self.root.after(0, self.update_ticker_display, ticker, price)
                
                # Check for new news
                items = self.rit.news(since=self.last_nid)
                for item in items:
                    nid = item.get("news_id", 0)
                    if nid in self.seen_nids:
                        continue
                    
                    self.seen_nids.add(nid)
                    if nid > self.last_nid:
                        self.last_nid = nid
                    
                    h = (item.get("headline") or item.get("body") or "").strip()
                    t = (item.get("ticker") or "").strip()
                    
                    if h and "Welcome" not in h:
                        self.root.after(0, self.process_news, h, t)
                
            except Exception as e:
                self.root.after(0, self.lbl_status.config, {"text": "❌ ERROR", "fg": COL_NEON_RED})
            
            time.sleep(0.5)
    
    def process_news(self, headline, news_ticker):
        # Parse tickers
        affected_tickers = []
        if news_ticker and "-" in news_ticker:
            pair = news_ticker.split("-")[1]
            if "/" in pair:
                affected_tickers = pair.split("/")
            else:
                affected_tickers = [pair[i:i+3] for i in range(0, len(pair), 3)]
        
        if not affected_tickers:
            # General news
            self.add_news_line(f"[TICK {self.tick:4d}] MARKET | {headline[:60]}...", "dim")
            return
        
        # Predict impact for each ticker
        for ticker in affected_tickers:
            if ticker not in ALL_TICKERS:
                continue
            
            # Store old prediction before updating
            old_prediction = self.predictions.get(ticker, 0.0)
            
            # Calculate new prediction
            impact = self.scorer.predict_impact(ticker, headline)
            self.predictions[ticker] = impact
            
            # Calculate delta
            delta = impact - old_prediction
            
            # Update display
            self.update_ticker_display(ticker)
            
            # Format delta: () for negative, normal for positive
            if delta >= 0:
                delta_str = f"{delta:+.2%}"
            else:
                delta_str = f"({abs(delta):.2%})"
            
            # Log news with signal and delta
            signal, _ = self.get_signal(impact)
            impact_str = f"{impact:+.2%}"
            
            if signal == "BUY":
                color = "green"
            elif signal in ["SELL", "HIGH"]:
                color = "red"
            else:
                color = "yellow"
            
            self.add_news_line(
                f"[TICK {self.tick:4d}] {ticker:4s} | {signal:4s} {impact_str:8s} {delta_str:8s} | {headline[:32]}...",
                color
            )
            
            # Update previous prediction for next delta calculation
            self.previous_predictions[ticker] = impact
        
        self.lbl_news_count.config(text=f"{len(self.newslog)} events")
    
    def update_ticker_display(self, ticker, price=None):
        if ticker not in self.ticker_widgets:
            return
        
        widgets = self.ticker_widgets[ticker]
        
        # Get prediction and signal
        pred = self.predictions.get(ticker, 0.0)
        signal, signal_color = self.get_signal(pred)
        
        # Update signal (BUY/HOLD/SELL/HIGH)
        widgets["signal_label"].config(text=signal, fg=signal_color)
        
        # Update prediction percentage
        pred_text = f"{pred:+.2%}"
        widgets["pred_label"].config(text=pred_text, fg=signal_color)
        
        # Update position
        pos = self.positions.get(ticker, 0)
        pos_text = f"Position: {pos:+,}"
        pos_color = COL_NEON_GREEN if pos > 0 else COL_NEON_RED if pos < 0 else COL_TEXT
        widgets["pos_label"].config(text=pos_text, fg=pos_color)
        
        # Update price if provided
        if price is not None:
            widgets["price_label"].config(text=f"Price: ${price:.2f}")
    
    def add_news_line(self, text, tag="white"):
        """Add line to news feed only."""
        self.newslog.append(text)
        
        self.news_text.config(state="normal")
        self.news_text.insert("1.0", "\n")
        self.news_text.insert("1.0", text, tag)
        self.news_text.config(state="disabled")
        self.news_text.see("1.0")
    
    def enable_trading_buttons(self):
        """Enable ALL buttons for ALL tickers."""
        self.btn_flatten.config(state="normal")
        for ticker, widgets in self.ticker_widgets.items():
            # All tickers can now buy and sell
            widgets["btn_buy"].config(state="normal")
            widgets["btn_sell"].config(state="normal")
            widgets["btn_max_long"].config(state="normal")
            widgets["btn_max_short"].config(state="normal")
    
    def disable_trading_buttons(self):
        self.btn_flatten.config(state="disabled")
        for widgets in self.ticker_widgets.values():
            widgets["btn_buy"].config(state="disabled")
            widgets["btn_sell"].config(state="disabled")
            widgets["btn_max_long"].config(state="disabled")
            widgets["btn_max_short"].config(state="disabled")
    
    def trade_buy(self, ticker, primary_size):
        """BUY = LONG primary + SHORT hedge - INSTANT PARALLEL EXECUTION"""
        if not self.rit:
            return
        
        # Choose hedge ticker with preference
        result = self.choose_hedge_ticker(ticker)
        if not result:
            return
        
        hedge_ticker, reason = result
        
        # Execute BOTH orders in parallel threads - INSTANT
        def execute():
            # Create threads for both orders
            hedge_thread = threading.Thread(
                target=self.rit.submit_order_fast,
                args=(hedge_ticker, -HEDGE_SIZE),
                daemon=True
            )
            primary_thread = threading.Thread(
                target=self.rit.submit_order_fast,
                args=(ticker, primary_size),
                daemon=True
            )
            
            # Fire both simultaneously
            hedge_thread.start()
            primary_thread.start()
            
            # Wait for both to complete
            hedge_thread.join()
            primary_thread.join()
            
            self.root.bell()
        
        t = threading.Thread(target=execute, daemon=True)
        t.start()
    
    def trade_sell(self, ticker, primary_size):
        """SELL = SHORT primary + LONG hedge - INSTANT PARALLEL EXECUTION"""
        if not self.rit:
            return
        
        # Choose hedge ticker with preference
        result = self.choose_hedge_ticker(ticker)
        if not result:
            return
        
        hedge_ticker, reason = result
        
        # Execute BOTH orders in parallel threads - INSTANT
        def execute():
            # Create threads for both orders
            hedge_thread = threading.Thread(
                target=self.rit.submit_order_fast,
                args=(hedge_ticker, HEDGE_SIZE),
                daemon=True
            )
            primary_thread = threading.Thread(
                target=self.rit.submit_order_fast,
                args=(ticker, -primary_size),
                daemon=True
            )
            
            # Fire both simultaneously
            hedge_thread.start()
            primary_thread.start()
            
            # Wait for both to complete
            hedge_thread.join()
            primary_thread.join()
            
            self.root.bell()
        
        t = threading.Thread(target=execute, daemon=True)
        t.start()
    
    def flatten_all(self):
        """
        INSTANT 1-CLICK FLATTEN with VERIFICATION LOOP
        Closes ALL positions and loops until EVERYTHING is confirmed ZERO
        """
        if not self.rit:
            return
        
        def execute():
            max_attempts = 5
            attempt = 0
            
            print("=" * 60)
            print("🔴 FLATTEN ALL - STARTING")
            print("=" * 60)
            
            while attempt < max_attempts:
                attempt += 1
                print(f"\n🔄 ATTEMPT {attempt}/{max_attempts}")
                
                # Fetch fresh positions from API
                prices = self.rit.prices()
                if not prices:
                    print("❌ Failed to fetch positions")
                    time.sleep(0.5)
                    continue
                
                # Update local positions and check for non-zero
                current_positions = {}
                has_positions = False
                
                for ticker in ALL_TICKERS:
                    if ticker in prices:
                        pos = prices[ticker]["position"]
                        current_positions[ticker] = pos
                        self.positions[ticker] = pos
                        
                        if pos != 0:
                            has_positions = True
                            print(f"  📊 {ticker}: {pos:+,} shares")
                
                # If all positions are zero, we're done!
                if not has_positions:
                    print("\n" + "=" * 60)
                    print("✅ SUCCESS! ALL POSITIONS ARE ZERO!")
                    print("=" * 60)
                    self.root.bell()
                    return
                
                # Submit close orders for all non-zero positions
                print(f"\n🚀 Closing {sum(1 for p in current_positions.values() if p != 0)} positions...")
                threads = []
                
                for ticker, pos in current_positions.items():
                    if pos != 0:
                        close_qty = -pos
                        print(f"  🔴 {ticker}: {pos:+,} → Close with {close_qty:+,}")
                        t = threading.Thread(
                            target=self.rit.submit_order_fast,
                            args=(ticker, close_qty),
                            daemon=True
                        )
                        threads.append(t)
                
                # Fire all close orders simultaneously
                for t in threads:
                    t.start()
                
                # Wait for all to complete
                for t in threads:
                    t.join()
                
                print(f"  ⏳ Orders submitted, waiting for fills...")
                
                # Wait for orders to fill before checking again
                time.sleep(1.5)
            
            # If we hit max attempts, report final state
            print("\n" + "=" * 60)
            print(f"⚠️ FLATTEN COMPLETE after {max_attempts} attempts")
            print("📊 Final positions:")
            for ticker in ALL_TICKERS:
                pos = self.positions.get(ticker, 0)
                if pos != 0:
                    print(f"  {ticker}: {pos:+,} shares remaining")
            print("=" * 60)
            self.root.bell()
        
        # Execute immediately - no confirmation
        t = threading.Thread(target=execute, daemon=True)
        t.start()


def main():
    if not WORD_SENTIMENT_CSV.exists():
        print(f"ERROR: Word sentiment CSV not found at {WORD_SENTIMENT_CSV}")
        return
    
    scorer = WordSentimentScorer(WORD_SENTIMENT_CSV)
    
    root = tk.Tk()
    app = DashboardApp(root, apikey="", scorer=scorer)
    root.mainloop()


if __name__ == "__main__":
    main()
