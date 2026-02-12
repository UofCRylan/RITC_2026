import pandas as pd
import numpy as np
from pathlib import Path
import requests
import time
import string
from collections import defaultdict

# =========================
# CONFIG
# =========================
IMPACT_CSV = Path(r"C:\Users\rylan\Desktop\M&A\ticker_by_ticker_impacts.csv")

# RIT API Configuration
RIT_API_KEY = "RYL2000"  # Replace with your actual API key
RIT_BASE_URL = "http://localhost:9999/v1"

# Trading thresholds
BULLISH_THRESHOLD = 0.013        # 15% - normal bullish signal
BEARISH_THRESHOLD = -0.080       # -10% - normal bearish signal

HIGH_CONVICTION_BULLISH = 0.0180  # 20%+ - HUGE upside, EXIT ALL & ENTER NEW
HIGH_CONVICTION_BEARISH = -0.013 # -15%+ - HUGE downside, EXIT ALL & ENTER NEW

# Position sizes - ASYMMETRIC for high conviction
PRIMARY_SIZE = 75000   # Large position in high-conviction ticker
HEDGE_SIZE = 25000     # Smaller hedge position in different ticker

# Order execution
MAX_ORDER_SIZE = 5000  # Max shares per order (from case rules)
MAX_CLOSE_RETRIES = 3  # Max attempts to close a position
MAX_FLATTEN_RETRIES = 5  # Max attempts to flatten all positions
POSITION_CHECK_DELAY = 0.3  # Seconds to wait before checking if position closed

# Exit parameters
EXIT_AFTER_TICKS = 30  # Close positions after this many ticks

# Risk limits from Merger Arb case
GROSS_LIMIT = 100000  # Max gross position across all stocks
NET_LIMIT = 50000     # Max net position across all stocks

# News matching
MIN_WORD_MATCHES = 3  # Minimum matching words to consider news similar
STOPWORDS = {
    "the","and","of","for","to","a","an","in","on","with","has","have","had",
    "is","are","be","been","this","that","as","at","by","from","over","into"
}
PUNCT_TRANS = str.maketrans({c: " " for c in string.punctuation})

# All tickers in the market
ALL_TICKERS = ["TGX", "PHR", "BYL", "CLD", "GGD", "PNR", "FSR", "ATB", "SPK", "EEC"]

# Deal pairs (target / acquirer) - used to identify which NOT to short
DEAL_PAIRS = {
    "TGX": "PHR",  # D1
    "PHR": "TGX",
    "BYL": "CLD",  # D2
    "CLD": "BYL",
    "GGD": "PNR",  # D3
    "PNR": "GGD",
    "FSR": "ATB",  # D4
    "ATB": "FSR",
    "SPK": "EEC",  # D5
    "EEC": "SPK"
}


class NewsImpactLearner:
    """Loads historical news impacts and predicts future impacts."""
    
    def __init__(self, csv_path):
        print(f"Loading historical impact data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        print(f"  Loaded {len(self.df)} historical ticker-news events")
        
        # Build word-ticker impact lookup
        self.word_ticker_impacts = defaultdict(lambda: defaultdict(list))
        self._build_word_impacts()
        
        # Build headline pattern lookup
        self.headline_impacts = []
        self._build_headline_patterns()
    
    def _tokenize(self, text):
        """Tokenize text into words."""
        if pd.isna(text):
            return []
        return [
            w for w in str(text).lower().translate(PUNCT_TRANS).split()
            if w and w not in STOPWORDS and not w.isdigit()
        ]
    
    def _build_word_impacts(self):
        """Build word -> ticker -> [impacts] mapping."""
        print("  Building word-ticker impact database...")
        for _, row in self.df.iterrows():
            ticker = row["ticker"]
            impact = row["pct_change"]
            text = f"{row['headline']} {row['body']}"
            words = self._tokenize(text)
            
            for word in words:
                self.word_ticker_impacts[word][ticker].append(impact)
        
        print(f"  Indexed {len(self.word_ticker_impacts)} unique words")
    
    def _build_headline_patterns(self):
        """Store historical headlines with their impacts."""
        print("  Building headline pattern database...")
        for _, row in self.df.iterrows():
            self.headline_impacts.append({
                "ticker": row["ticker"],
                "impact": row["pct_change"],
                "headline": row["headline"],
                "headline_tokens": set(self._tokenize(row["headline"]))
            })
        print(f"  Indexed {len(self.headline_impacts)} headline patterns")
    
    def predict_impact(self, ticker, headline, body=""):
        """
        Predict impact for a ticker given news headline/body.
        
        Returns:
            float: predicted percentage impact (e.g., 0.15 = +15%)
        """
        text = f"{headline} {body}"
        tokens = self._tokenize(text)
        token_set = set(tokens)
        
        if not tokens:
            return 0.0
        
        # Method 1: Word-based prediction
        word_impacts = []
        for word in tokens:
            if word in self.word_ticker_impacts:
                ticker_impacts = self.word_ticker_impacts[word].get(ticker, [])
                if ticker_impacts:
                    word_impacts.extend(ticker_impacts)
        
        # Method 2: Similar headline matching
        similar_impacts = []
        for hist in self.headline_impacts:
            if hist["ticker"] != ticker:
                continue
            
            # Calculate word overlap
            overlap = len(token_set & hist["headline_tokens"])
            if overlap >= MIN_WORD_MATCHES:
                # Weight by similarity
                similarity = overlap / len(token_set | hist["headline_tokens"])
                similar_impacts.append((hist["impact"], similarity))
        
        # Combine predictions
        predictions = []
        
        if word_impacts:
            predictions.append(np.mean(word_impacts))
        
        if similar_impacts:
            # Weighted average by similarity
            total_weight = sum(w for _, w in similar_impacts)
            weighted_impact = sum(imp * w for imp, w in similar_impacts) / total_weight
            predictions.append(weighted_impact)
        
        if not predictions:
            return 0.0
        
        return np.mean(predictions)


class MergerArbTrader:
    """Real-time M&A trading bot using RIT API."""
    
    def __init__(self, learner):
        self.learner = learner
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": RIT_API_KEY})
        
        self.last_news_id = 0
        self.active_trades = []
        self.closed_trades_count = 0
        
        # Track current high conviction trade dollar move
        self.current_high_conviction_dollar_move = 0.0
        
        print("\nInitializing RIT connection...")
        self._check_connection()
    
    def _check_connection(self):
        """Test RIT API connection."""
        try:
            resp = self.session.get(f"{RIT_BASE_URL}/case")
            resp.raise_for_status()
            case = resp.json()
            print(f"  Connected to case: {case.get('name', 'Unknown')}")
            print(f"  Tick: {case.get('tick', 0)} | Period: {case.get('period', 0)}")
            print(f"  Status: {case.get('status', 'Unknown')}")
            return True
        except Exception as e:
            print(f"  ERROR: Cannot connect to RIT API: {e}")
            return False
    
    def get_case_status(self):
        """Get current case status from RIT."""
        try:
            resp = self.session.get(f"{RIT_BASE_URL}/case")
            resp.raise_for_status()
            case = resp.json()
            return case.get("status", "UNKNOWN")
        except Exception as e:
            print(f"  ERROR getting case status: {e}")
            return "UNKNOWN"
    
    def wait_for_case_start(self):
        """Wait until the case starts running."""
        print(f"\n{'='*80}")
        print("⏳ WAITING FOR CASE TO START...")
        print(f"{'='*80}")
        print("The bot will automatically begin trading once the case starts.")
        print("Press Ctrl+C to cancel.\n")
        
        check_count = 0
        while True:
            try:
                status = self.get_case_status()
                
                # Case is running - start trading!
                if status in ["ACTIVE", "RUNNING"]:
                    print(f"\n✅ Case is now {status}! Starting trading bot...\n")
                    time.sleep(1)  # Brief pause before starting
                    return True
                
                # Case not started yet
                elif status in ["STOPPED", "PAUSED", "NOT_STARTED"]:
                    check_count += 1
                    if check_count % 5 == 0:  # Print every 5 checks
                        print(f"   Status: {status} | Still waiting... (checked {check_count} times)")
                    time.sleep(2)  # Check every 2 seconds
                
                # Unknown status
                else:
                    print(f"   Unknown status: {status} | Continuing to wait...")
                    time.sleep(2)
                    
            except KeyboardInterrupt:
                print("\n\n⚠ Wait cancelled by user")
                return False
            except Exception as e:
                print(f"   Error checking status: {e}")
                time.sleep(2)
    
    def get_current_tick(self):
        """Get current tick from RIT."""
        try:
            resp = self.session.get(f"{RIT_BASE_URL}/case")
            resp.raise_for_status()
            case = resp.json()
            return case.get("tick", 0)
        except Exception as e:
            print(f"  ERROR getting tick: {e}")
            return 0
    
    def get_positions(self):
        """Get current positions from RIT."""
        try:
            resp = self.session.get(f"{RIT_BASE_URL}/securities")
            resp.raise_for_status()
            securities = resp.json()
            
            positions = {}
            for sec in securities:
                ticker = sec.get("ticker")
                position = sec.get("position", 0)
                if ticker:
                    positions[ticker] = position
            
            return positions
        except Exception as e:
            print(f"  ERROR getting positions: {e}")
            return {}
    
    def get_ticker_price(self, ticker):
        """Get current bid/ask midpoint price for a ticker."""
        try:
            resp = self.session.get(f"{RIT_BASE_URL}/securities")
            resp.raise_for_status()
            securities = resp.json()
            
            for sec in securities:
                if sec.get("ticker") == ticker:
                    bid = sec.get("bid", 0)
                    ask = sec.get("ask", 0)
                    if bid > 0 and ask > 0:
                        return (bid + ask) / 2
                    elif sec.get("last", 0) > 0:
                        return sec.get("last")
            
            return 0
        except Exception as e:
            print(f"  ERROR getting price for {ticker}: {e}")
            return 0
    
    def calculate_position_limits(self, positions):
        """Calculate current gross and net positions."""
        gross = sum(abs(pos) for pos in positions.values())
        net = sum(positions.values())
        return gross, net
    
    def submit_order(self, ticker, quantity, order_type="MARKET"):
        """Submit order to RIT - INSTANT EXECUTION."""
        action = "BUY" if quantity > 0 else "SELL"
        total_qty = abs(quantity)
        
        if total_qty <= MAX_ORDER_SIZE:
            try:
                payload = {
                    "ticker": ticker,
                    "type": order_type,
                    "quantity": total_qty,
                    "action": action
                }
                
                resp = self.session.post(f"{RIT_BASE_URL}/orders", params=payload)
                resp.raise_for_status()
                
                print(f"    ✓ {action} {total_qty} {ticker}")
                return total_qty if quantity > 0 else -total_qty
                
            except Exception as e:
                print(f"    ✗ Order failed: {e}")
                return 0
        
        else:
            executed_qty = 0
            remaining = total_qty
            
            while remaining > 0:
                order_qty = min(remaining, MAX_ORDER_SIZE)
                
                try:
                    payload = {
                        "ticker": ticker,
                        "type": order_type,
                        "quantity": order_qty,
                        "action": action
                    }
                    
                    resp = self.session.post(f"{RIT_BASE_URL}/orders", params=payload)
                    resp.raise_for_status()
                    
                    executed_qty += order_qty
                    remaining -= order_qty
                    
                except Exception as e:
                    print(f"    ✗ Failed after {executed_qty}/{total_qty}: {e}")
                    break
            
            if executed_qty > 0:
                print(f"    ✓ {action} {executed_qty} {ticker}")
            
            return executed_qty if quantity > 0 else -executed_qty
    
    def verify_position_closed(self, ticker, expected_position=0):
        """Verify that a position has been closed to the expected level."""
        positions = self.get_positions()
        current_pos = positions.get(ticker, 0)
        is_closed = (current_pos == expected_position)
        return is_closed, current_pos
    
    def close_position_with_verification(self, ticker, quantity_to_close):
        """Close a position and verify it actually closed."""
        for attempt in range(1, MAX_CLOSE_RETRIES + 1):
            executed = self.submit_order(ticker, quantity_to_close)
            
            if executed == 0:
                print(f"    ⚠ Close order failed for {ticker}, attempt {attempt}/{MAX_CLOSE_RETRIES}")
                continue
            
            time.sleep(POSITION_CHECK_DELAY)
            
            is_closed, current_pos = self.verify_position_closed(ticker, expected_position=0)
            
            if is_closed:
                print(f"    ✓ Verified: {ticker} position closed")
                return True
            else:
                print(f"    ⚠ {ticker} still has position {current_pos:+,}, retrying... ({attempt}/{MAX_CLOSE_RETRIES})")
                quantity_to_close = -current_pos
        
        print(f"    ✗ FAILED to close {ticker} after {MAX_CLOSE_RETRIES} attempts")
        return False
    
    def force_flatten_all_positions(self):
        """SAFETY CHECK: Ensure ALL tickers have 0 position."""
        print(f"\n   🔍 FORCE FLATTEN CHECK: Ensuring all positions = 0")
        
        for attempt in range(1, MAX_FLATTEN_RETRIES + 1):
            positions = self.get_positions()
            non_zero_positions = {ticker: pos for ticker, pos in positions.items() if pos != 0}
            
            if not non_zero_positions:
                print(f"   ✅ All positions verified FLAT")
                return True
            
            print(f"   ⚠ Attempt {attempt}/{MAX_FLATTEN_RETRIES}: Found {len(non_zero_positions)} non-zero positions")
            for ticker, position in non_zero_positions.items():
                print(f"      Flattening {ticker}: {position:+,} → 0")
                close_qty = -position
                self.submit_order(ticker, close_qty)
            
            time.sleep(POSITION_CHECK_DELAY * 2)
        
        print(f"   ❌ FAILED TO FLATTEN after {MAX_FLATTEN_RETRIES} attempts")
        return False
    
    def choose_hedge_ticker(self, primary_ticker, headline, body):
        """Choose a ticker to short that is NOT part of the primary deal."""
        pair_ticker = DEAL_PAIRS.get(primary_ticker)
        exclude_tickers = {primary_ticker}
        if pair_ticker:
            exclude_tickers.add(pair_ticker)
        
        available = [t for t in ALL_TICKERS if t not in exclude_tickers]
        
        if not available:
            return None
        
        predictions = {}
        for ticker in available:
            impact = self.learner.predict_impact(ticker, headline, body)
            predictions[ticker] = impact
        
        hedge_ticker = min(predictions.items(), key=lambda x: x[1])[0]
        predicted_impact = predictions[hedge_ticker]
        
        print(f"      Hedge: {hedge_ticker} (predicted: {predicted_impact:+.2%})")
        
        return hedge_ticker
    
    def close_all_positions(self, reason="High conviction opportunity"):
        """Immediately close ALL open positions - WITH VERIFICATION + FORCE FLATTEN."""
        if not self.active_trades:
            self.force_flatten_all_positions()
            return
        
        current_tick = self.get_current_tick()
        
        print(f"\n🚨 CLOSING ALL POSITIONS - {reason}")
        print(f"   Closing {len(self.active_trades)} trade(s)...")
        
        trades_to_close = list(self.active_trades)
        
        for trade in trades_to_close:
            primary_ticker = trade.get("primary_ticker")
            hedge_ticker = trade.get("hedge_ticker")
            primary_qty = trade.get("primary_qty")
            hedge_qty = trade.get("hedge_qty")
            
            print(f"\n   📤 Closing Trade #{trade.get('trade_id')}")
            
            if primary_qty != 0:
                close_qty = -primary_qty
                print(f"      Closing {primary_ticker}: {close_qty:+,}")
                self.close_position_with_verification(primary_ticker, close_qty)
            
            if hedge_qty != 0:
                close_qty = -hedge_qty
                print(f"      Closing {hedge_ticker}: {close_qty:+,}")
                self.close_position_with_verification(hedge_ticker, close_qty)
            
            self.active_trades.remove(trade)
            self.closed_trades_count += 1
        
        print(f"\n   ✓ All tracked positions closed")
        
        # Reset high conviction tracking
        self.current_high_conviction_dollar_move = 0.0
        
        self.force_flatten_all_positions()
    
    def check_exit_conditions(self):
        """Check if any open positions should be closed based on tick count."""
        current_tick = self.get_current_tick()
        
        if not self.active_trades:
            return
        
        trades_to_close = []
        
        for trade in self.active_trades:
            entry_tick = trade.get("entry_tick", 0)
            ticks_held = current_tick - entry_tick
            
            if ticks_held >= EXIT_AFTER_TICKS:
                trades_to_close.append(trade)
        
        for trade in trades_to_close:
            self.close_position(trade, current_tick)
            self.active_trades.remove(trade)
            self.closed_trades_count += 1
    
    def close_position(self, trade, current_tick):
        """Close both legs of a trade - WITH VERIFICATION + FORCE FLATTEN."""
        primary_ticker = trade.get("primary_ticker")
        hedge_ticker = trade.get("hedge_ticker")
        primary_qty = trade.get("primary_qty")
        hedge_qty = trade.get("hedge_qty")
        entry_tick = trade.get("entry_tick")
        
        ticks_held = current_tick - entry_tick
        
        print(f"\n⏰ AUTO-CLOSE Trade #{trade.get('trade_id')} (held {ticks_held} ticks)")
        
        if primary_qty != 0:
            close_qty = -primary_qty
            print(f"   Closing {primary_ticker}: {close_qty:+,}")
            self.close_position_with_verification(primary_ticker, close_qty)
        
        if hedge_qty != 0:
            close_qty = -hedge_qty
            print(f"   Closing {hedge_ticker}: {close_qty:+,}")
            self.close_position_with_verification(hedge_ticker, close_qty)
        
        print(f"   ✓ Position closed")
        
        # Reset high conviction tracking if this was the high conviction trade
        if trade.get("strategy") in ["high_conviction_bullish", "high_conviction_bearish"]:
            self.current_high_conviction_dollar_move = 0.0
        
        self.force_flatten_all_positions()
    
    def check_news(self):
        """Check for new news and analyze for trading signals."""
        try:
            resp = self.session.get(f"{RIT_BASE_URL}/news")
            resp.raise_for_status()
            news_items = resp.json()
            
            new_items = [
                n for n in news_items
                if n.get("news_id", 0) > self.last_news_id
            ]
            
            if not new_items:
                return
            
            self.last_news_id = max(n.get("news_id", 0) for n in new_items)
            
            for news in new_items:
                self.analyze_and_trade(news)
        
        except Exception as e:
            print(f"  ERROR checking news: {e}")
    
    def analyze_and_trade(self, news):
        """Analyze news and execute trades if signal is strong."""
        headline = news.get("headline", "")
        body = news.get("body", "")
        ticker_field = news.get("ticker", "")
        
        print(f"\n📰 NEWS {news.get('news_id')} | {ticker_field}")
        print(f"   {headline[:80]}")
        
        affected_tickers = []
        if ticker_field and "-" in ticker_field:
            pair = ticker_field.split("-")[1]
            if "/" in pair:
                affected_tickers = pair.split("/")
            else:
                affected_tickers = [pair[i:i+3] for i in range(0, len(pair), 3)]
        
        if not affected_tickers:
            print("   ⚠ Cannot parse tickers")
            return
        
        current_tick = self.get_current_tick()
        
        # Predict impact for each ticker AND get current prices
        predictions = {}
        for ticker in affected_tickers:
            impact = self.learner.predict_impact(ticker, headline, body)
            price = self.get_ticker_price(ticker)
            dollar_move = abs(impact * price)  # Absolute dollar move expected
            
            predictions[ticker] = {
                "impact": impact,
                "price": price,
                "dollar_move": dollar_move
            }
            
            print(f"   Predicted {ticker}: {impact:+.2%} | Price: ${price:.2f} | Dollar move: ${dollar_move:.2f}")
        
        # Separate opportunities by conviction level
        high_conviction_opportunities = []
        normal_opportunities = []
        
        for ticker, pred in predictions.items():
            impact = pred["impact"]
            dollar_move = pred["dollar_move"]
            
            if impact >= HIGH_CONVICTION_BULLISH:
                distance = impact - HIGH_CONVICTION_BULLISH
                high_conviction_opportunities.append({
                    "ticker": ticker,
                    "impact": impact,
                    "direction": "bullish",
                    "distance": distance,
                    "price": pred["price"],
                    "dollar_move": dollar_move
                })
            elif impact <= HIGH_CONVICTION_BEARISH:
                distance = abs(impact - HIGH_CONVICTION_BEARISH)
                high_conviction_opportunities.append({
                    "ticker": ticker,
                    "impact": impact,
                    "direction": "bearish",
                    "distance": distance,
                    "price": pred["price"],
                    "dollar_move": dollar_move
                })
            elif impact >= BULLISH_THRESHOLD:
                normal_opportunities.append({
                    "ticker": ticker,
                    "impact": impact,
                    "direction": "bullish",
                    "price": pred["price"],
                    "dollar_move": dollar_move
                })
            elif impact <= BEARISH_THRESHOLD:
                normal_opportunities.append({
                    "ticker": ticker,
                    "impact": impact,
                    "direction": "bearish",
                    "price": pred["price"],
                    "dollar_move": dollar_move
                })
        
        # Select best high conviction opportunity - PRIORITIZE DOLLAR MOVE
        if len(high_conviction_opportunities) > 1:
            print(f"\n   🎯 MULTIPLE HIGH CONVICTION:")
            for opp in high_conviction_opportunities:
                print(f"      {opp['ticker']}: {opp['impact']:+.2%} | ${opp['price']:.2f} | Dollar move: ${opp['dollar_move']:.2f}")
            
            # Sort by DOLLAR MOVE (highest first)
            high_conviction_opportunities.sort(key=lambda x: x['dollar_move'], reverse=True)
            best_opp = high_conviction_opportunities[0]
            
            print(f"   ✅ SELECTED: {best_opp['ticker']} with ${best_opp['dollar_move']:.2f} dollar move (highest)")
            high_conviction_opportunities = [best_opp]
        
        # NEW LOGIC: Check if we should hold current high conviction trade
        if high_conviction_opportunities:
            best_opp = high_conviction_opportunities[0]
            new_dollar_move = best_opp['dollar_move']
            
            # If we have an active high conviction trade
            if self.current_high_conviction_dollar_move > 0:
                print(f"\n   📊 COMPARISON:")
                print(f"      Current high conviction: ${self.current_high_conviction_dollar_move:.2f} dollar move")
                print(f"      New opportunity: ${new_dollar_move:.2f} dollar move")
                
                # If current position has higher dollar move, KEEP IT
                if self.current_high_conviction_dollar_move >= new_dollar_move:
                    print(f"   🛡️ HOLDING CURRENT POSITION (larger dollar move)")
                    print(f"   ⏭️ Skipping new opportunity")
                    high_conviction_opportunities = []  # Clear to skip execution
                else:
                    print(f"   🔄 NEW OPPORTUNITY IS BETTER (higher dollar move)")
                    print(f"   📤 Will exit current and enter new position")
        
        # Process high conviction opportunities (if not blocked by hold logic)
        for opp in high_conviction_opportunities:
            ticker = opp['ticker']
            impact = opp['impact']
            direction = opp['direction']
            dollar_move = opp['dollar_move']
            
            if direction == "bullish":
                print(f"\n   🟢🟢 HIGH CONVICTION BULLISH: {ticker} ({impact:+.2%}, ${dollar_move:.2f} move)")
                self.close_all_positions(reason=f"High conviction on {ticker}")
                
                hedge_ticker = self.choose_hedge_ticker(ticker, headline, body)
                if not hedge_ticker:
                    print(f"      ✗ No hedge available")
                    continue
                
                print(f"      Strategy: SHORT {HEDGE_SIZE:,} hedge FIRST → LONG {PRIMARY_SIZE:,} primary")
                
                print(f"      → [1/2] SHORT {HEDGE_SIZE:,} {hedge_ticker} (HEDGE)")
                hedge_qty = self.submit_order(hedge_ticker, -HEDGE_SIZE)
                
                if hedge_qty == 0:
                    print(f"      ✗ Hedge failed")
                    continue
                
                print(f"      → [2/2] LONG {PRIMARY_SIZE:,} {ticker} (PRIMARY)")
                primary_qty = self.submit_order(ticker, PRIMARY_SIZE)
                
                if primary_qty != 0:
                    trade_id = len(self.active_trades) + 1 + self.closed_trades_count
                    self.active_trades.append({
                        "trade_id": trade_id,
                        "news_id": news.get("news_id"),
                        "entry_tick": current_tick,
                        "primary_ticker": ticker,
                        "hedge_ticker": hedge_ticker,
                        "primary_qty": primary_qty,
                        "hedge_qty": hedge_qty,
                        "predicted_impact": impact,
                        "strategy": "high_conviction_bullish"
                    })
                    
                    # Track this high conviction trade
                    self.current_high_conviction_dollar_move = dollar_move
                    
                    final_positions = self.get_positions()
                    final_gross, final_net = self.calculate_position_limits(final_positions)
                    
                    print(f"      ✓ Trade #{trade_id} @ tick {current_tick} | Gross={final_gross:,} Net={final_net:+,}")
                    print(f"      💰 Tracking: ${dollar_move:.2f} dollar move (will hold if larger opportunity appears)")
                    print(f"      ⏰ Exit @ tick {current_tick + EXIT_AFTER_TICKS}")
            
            elif direction == "bearish":
                print(f"\n   🔴🔴 HIGH CONVICTION BEARISH: {ticker} ({impact:+.2%}, ${dollar_move:.2f} move)")
                self.close_all_positions(reason=f"High conviction on {ticker}")
                
                hedge_ticker = self.choose_hedge_ticker(ticker, headline, body)
                if not hedge_ticker:
                    print(f"      ✗ No hedge available")
                    continue
                
                print(f"      Strategy: LONG {HEDGE_SIZE:,} hedge FIRST → SHORT {PRIMARY_SIZE:,} primary")
                
                print(f"      → [1/2] LONG {HEDGE_SIZE:,} {hedge_ticker} (HEDGE)")
                hedge_qty = self.submit_order(hedge_ticker, HEDGE_SIZE)
                
                if hedge_qty == 0:
                    print(f"      ✗ Hedge failed")
                    continue
                
                print(f"      → [2/2] SHORT {PRIMARY_SIZE:,} {ticker} (PRIMARY)")
                primary_qty = self.submit_order(ticker, -PRIMARY_SIZE)
                
                if primary_qty != 0:
                    trade_id = len(self.active_trades) + 1 + self.closed_trades_count
                    self.active_trades.append({
                        "trade_id": trade_id,
                        "news_id": news.get("news_id"),
                        "entry_tick": current_tick,
                        "primary_ticker": ticker,
                        "hedge_ticker": hedge_ticker,
                        "primary_qty": primary_qty,
                        "hedge_qty": hedge_qty,
                        "predicted_impact": impact,
                        "strategy": "high_conviction_bearish"
                    })
                    
                    # Track this high conviction trade
                    self.current_high_conviction_dollar_move = dollar_move
                    
                    final_positions = self.get_positions()
                    final_gross, final_net = self.calculate_position_limits(final_positions)
                    
                    print(f"      ✓ Trade #{trade_id} @ tick {current_tick} | Gross={final_gross:,} Net={final_net:+,}")
                    print(f"      💰 Tracking: ${dollar_move:.2f} dollar move (will hold if larger opportunity appears)")
                    print(f"      ⏰ Exit @ tick {current_tick + EXIT_AFTER_TICKS}")
        
        # Process normal opportunities - ALSO prioritize by dollar move
        if not high_conviction_opportunities:
            # Sort normal opportunities by dollar move
            if len(normal_opportunities) > 1:
                print(f"\n   💰 Multiple opportunities - prioritizing by dollar move:")
                for opp in normal_opportunities:
                    print(f"      {opp['ticker']}: {opp['impact']:+.2%} | ${opp['dollar_move']:.2f} move")
                
                normal_opportunities.sort(key=lambda x: x['dollar_move'], reverse=True)
                print(f"   ✅ SELECTED: {normal_opportunities[0]['ticker']} with ${normal_opportunities[0]['dollar_move']:.2f} dollar move")
                normal_opportunities = [normal_opportunities[0]]
            
            for opp in normal_opportunities:
                ticker = opp['ticker']
                impact = opp['impact']
                direction = opp['direction']
                
                if direction == "bullish":
                    print(f"\n   🟢 BULLISH: {ticker} ({impact:+.2%}, ${opp['dollar_move']:.2f} move)")
                    
                    hedge_ticker = self.choose_hedge_ticker(ticker, headline, body)
                    if not hedge_ticker:
                        print(f"      ✗ No hedge available")
                        continue
                    
                    print(f"      Strategy: SHORT {HEDGE_SIZE:,} hedge FIRST → LONG {PRIMARY_SIZE:,} primary")
                    
                    print(f"      → [1/2] SHORT {HEDGE_SIZE:,} {hedge_ticker} (HEDGE)")
                    short_qty = self.submit_order(hedge_ticker, -HEDGE_SIZE)
                    
                    if short_qty == 0:
                        continue
                    
                    print(f"      → [2/2] LONG {PRIMARY_SIZE:,} {ticker} (PRIMARY)")
                    long_qty = self.submit_order(ticker, PRIMARY_SIZE)
                    
                    if long_qty != 0:
                        trade_id = len(self.active_trades) + 1 + self.closed_trades_count
                        self.active_trades.append({
                            "trade_id": trade_id,
                            "news_id": news.get("news_id"),
                            "entry_tick": current_tick,
                            "primary_ticker": ticker,
                            "hedge_ticker": hedge_ticker,
                            "primary_qty": long_qty,
                            "hedge_qty": short_qty,
                            "predicted_impact": impact,
                            "strategy": "bullish_75k_25k"
                        })
                        
                        final_positions = self.get_positions()
                        final_gross, final_net = self.calculate_position_limits(final_positions)
                        
                        print(f"      ✓ Trade #{trade_id} @ tick {current_tick} | Gross={final_gross:,} Net={final_net:+,}")
                
                elif direction == "bearish":
                    print(f"\n   🔴 BEARISH: {ticker} ({impact:+.2%}, ${opp['dollar_move']:.2f} move)")
                    
                    hedge_ticker = self.choose_hedge_ticker(ticker, headline, body)
                    if not hedge_ticker:
                        print(f"      ✗ No hedge available")
                        continue
                    
                    print(f"      Strategy: LONG {HEDGE_SIZE:,} hedge FIRST → SHORT {PRIMARY_SIZE:,} primary")
                    
                    print(f"      → [1/2] LONG {HEDGE_SIZE:,} {hedge_ticker} (HEDGE)")
                    long_qty = self.submit_order(hedge_ticker, HEDGE_SIZE)
                    
                    if long_qty == 0:
                        continue
                    
                    print(f"      → [2/2] SHORT {PRIMARY_SIZE:,} {ticker} (PRIMARY)")
                    short_qty = self.submit_order(ticker, -PRIMARY_SIZE)
                    
                    if short_qty != 0:
                        trade_id = len(self.active_trades) + 1 + self.closed_trades_count
                        self.active_trades.append({
                            "trade_id": trade_id,
                            "news_id": news.get("news_id"),
                            "entry_tick": current_tick,
                            "primary_ticker": ticker,
                            "hedge_ticker": hedge_ticker,
                            "primary_qty": short_qty,
                            "hedge_qty": long_qty,
                            "predicted_impact": impact,
                            "strategy": "bearish_75k_25k"
                        })
                        
                        final_positions = self.get_positions()
                        final_gross, final_net = self.calculate_position_limits(final_positions)
                        
                        print(f"      ✓ Trade #{trade_id} @ tick {current_tick} | Gross={final_gross:,} Net={final_net:+,}")
    
    def run(self, check_interval=1):
        """Main trading loop."""
        print(f"\n{'='*80}")
        print("🤖 M&A TRADING BOT - DOLLAR PRIORITY + HOLD LOGIC")
        print(f"{'='*80}")
        print(f"Normal: ≥{BULLISH_THRESHOLD:+.0%} bull | ≤{BEARISH_THRESHOLD:+.0%} bear")
        print(f"HIGH CONVICTION: ≥{HIGH_CONVICTION_BULLISH:+.0%} bull | ≤{HIGH_CONVICTION_BEARISH:+.0%} bear")
        print(f"\nSelection Priority:")
        print(f"  • Multiple opportunities → Select HIGHEST DOLLAR MOVE")
        print(f"  • Formula: |impact %| × price = dollar move")
        print(f"\nHold Logic:")
        print(f"  • If current high conviction trade has LARGER dollar move than new")
        print(f"  • KEEP current position, skip new opportunity")
        print(f"  • Only exit if new opportunity has HIGHER dollar move")
        print(f"\nExecution Order (ALL TRADES):")
        print(f"  • BULLISH: SHORT hedge FIRST → LONG primary")
        print(f"  • BEARISH: LONG hedge FIRST → SHORT primary")
        print(f"\nPosition: Primary={PRIMARY_SIZE:,} | Hedge={HEDGE_SIZE:,} | Net={PRIMARY_SIZE-HEDGE_SIZE:,}")
        print(f"Limits: Gross={GROSS_LIMIT:,} | Net={NET_LIMIT:,}")
        print(f"Exit: {EXIT_AFTER_TICKS} ticks (with forced flatten)")
        print(f"{'='*80}\n")
        
        try:
            while True:
                self.check_news()
                self.check_exit_conditions()
                time.sleep(check_interval)
        
        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped")
            print(f"Open: {len(self.active_trades)} | Closed: {self.closed_trades_count}")
            print("\n🔒 Final safety check - flattening all positions...")
            self.force_flatten_all_positions()


def main():
    if not IMPACT_CSV.exists():
        print(f"ERROR: Impact CSV not found at {IMPACT_CSV}")
        return
    
    learner = NewsImpactLearner(IMPACT_CSV)
    trader = MergerArbTrader(learner)
    
    # Wait for case to start before trading
    if not trader.wait_for_case_start():
        print("Exiting...")
        return
    
    # Start trading
    trader.run(check_interval=1)


if __name__ == "__main__":
    main()
