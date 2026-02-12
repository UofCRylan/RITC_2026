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
WORD_SENTIMENT_CSV = Path(r"C:\Users\rylan\Desktop\M&A\word_sentiment_scores.csv")


# RIT API Configuration
RIT_API_KEY = "RYL2000"
RIT_BASE_URL = "http://localhost:9999/v1"


# Trading thresholds (in percentages)
BULLISH_THRESHOLD_PCT = 0.27          # 0.9% - normal bullish signal
BEARISH_THRESHOLD_PCT = -0.23         # -0.5% - normal bearish signal

HIGH_CONVICTION_BULLISH_PCT = .6    # 1.1% - HUGE upside, EXIT ALL & ENTER NEW
HIGH_CONVICTION_BEARISH_PCT = -0.4   # -0.9% - HUGE downside, EXIT ALL & ENTER NEW

# Convert to decimal for calculations
BULLISH_THRESHOLD = BULLISH_THRESHOLD_PCT / 100
BEARISH_THRESHOLD = BEARISH_THRESHOLD_PCT / 100

HIGH_CONVICTION_BULLISH = HIGH_CONVICTION_BULLISH_PCT / 100
HIGH_CONVICTION_BEARISH = HIGH_CONVICTION_BEARISH_PCT / 100


# Position sizes
PRIMARY_SIZE = 75000
HEDGE_SIZE = 25000

# Order execution
MAX_ORDER_SIZE = 5000
MAX_CLOSE_RETRIES = 3
MAX_FLATTEN_RETRIES = 5
POSITION_CHECK_DELAY = 0.3

# Exit parameters
EXIT_AFTER_TICKS = 30

# Risk limits
GROSS_LIMIT = 100000
NET_LIMIT = 50000

# News aggregation parameters
NEWS_AGGREGATION_WINDOW = 5

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

# RESTRICTED: Can only BUY/LONG these tickers
ALLOWED_LONG_TICKERS = {"FSR", "TGX", "BYL", "GGD"}

DEAL_PAIRS = {
    "TGX": "PHR", "PHR": "TGX",
    "BYL": "CLD", "CLD": "BYL",
    "GGD": "PNR", "PNR": "GGD",
    "FSR": "ATB", "ATB": "FSR",
    "SPK": "EEC", "EEC": "SPK"
}


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
        
        top_pos = self.df.nlargest(5, 'avg_effect')[['word', 'avg_effect', 'count']]
        top_neg = self.df.nsmallest(5, 'avg_effect')[['word', 'avg_effect', 'count']]
        
        print(f"\n  Top positive:")
        for _, r in top_pos.iterrows():
            print(f"    {r['word']}: {r['avg_effect']:+.4f} ({int(r['count'])})")
        
        print(f"\n  Top negative:")
        for _, r in top_neg.iterrows():
            print(f"    {r['word']}: {r['avg_effect']:+.4f} ({int(r['count'])})")
    
    def _tokenize(self, text):
        if pd.isna(text):
            return []
        return [
            w for w in str(text).lower().translate(PUNCT_TRANS).split()
            if w and w not in STOPWORDS and not w.isdigit() and len(w) > 2
        ]
    
    def predict_impact(self, ticker, headline, body=""):
        """
        Predict impact for a ticker based on word sentiments in the news text.
        The ticker parameter is accepted for API compatibility with the original 
        NewsImpactLearner, but word sentiments are ticker-agnostic.
        """
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


class MergerArbTrader:
    """Real-time M&A trading bot using RIT API."""
    
    def __init__(self, learner):
        self.learner = learner
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": RIT_API_KEY})
        
        self.last_news_id = 0
        self.active_trades = []
        self.closed_trades_count = 0
        
        self.current_high_conviction_net_pct = 0.0
        self.recent_news_predictions = []
        
        print("\n✨ Initializing RIT connection...")
        self._check_connection()
    
    def _check_connection(self):
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
        try:
            resp = self.session.get(f"{RIT_BASE_URL}/case")
            resp.raise_for_status()
            case = resp.json()
            return case.get("status", "UNKNOWN")
        except Exception as e:
            print(f"  ERROR getting case status: {e}")
            return "UNKNOWN"
    
    def wait_for_case_start(self):
        print(f"\n{'='*80}")
        print("⏳ WAITING FOR CASE TO START...")
        print(f"{'='*80}")
        print("The bot will automatically begin trading once the case starts.")
        print("Press Ctrl+C to cancel.\n")
        
        check_count = 0
        while True:
            try:
                status = self.get_case_status()
                
                if status in ["ACTIVE", "RUNNING"]:
                    print(f"\n✅ Case is now {status}! Starting trading bot...\n")
                    time.sleep(1)
                    return True
                
                elif status in ["STOPPED", "PAUSED", "NOT_STARTED"]:
                    check_count += 1
                    if check_count % 5 == 0:
                        print(f"   Status: {status} | Still waiting... (checked {check_count} times)")
                    time.sleep(2)
                
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
        try:
            resp = self.session.get(f"{RIT_BASE_URL}/case")
            resp.raise_for_status()
            case = resp.json()
            return case.get("tick", 0)
        except Exception as e:
            print(f"  ERROR getting tick: {e}")
            return 0
    
    def get_positions(self):
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
        gross = sum(abs(pos) for pos in positions.values())
        net = sum(positions.values())
        return gross, net
    
    def get_aggregated_impact(self, ticker, current_impact, current_tick):
        """
        Check for recent news on the same ticker with same direction and aggregate impacts.
        
        Returns:
            tuple: (aggregated_impact, num_combined_news)
        """
        if current_impact > 0:
            current_direction = "bullish"
        elif current_impact < 0:
            current_direction = "bearish"
        else:
            return current_impact, 1
        
        matching_news = []
        for news in self.recent_news_predictions:
            if news["ticker"] == ticker and news["direction"] == current_direction:
                tick_diff = current_tick - news["tick"]
                if 0 < tick_diff <= NEWS_AGGREGATION_WINDOW:
                    matching_news.append(news)
        
        if matching_news:
            total_impact = current_impact
            for news in matching_news:
                total_impact += news["impact"]
            
            print(f"   📊 IMPACT AGGREGATION for {ticker}:")
            print(f"      Current news: {current_impact:+.2%}")
            for i, news in enumerate(matching_news, 1):
                print(f"      Previous news #{i} (tick {news['tick']}): {news['impact']:+.2%}")
            print(f"      ➕ COMBINED IMPACT: {total_impact:+.2%} ({len(matching_news) + 1} news items)")
            
            return total_impact, len(matching_news) + 1
        
        return current_impact, 1
    
    def record_news_prediction(self, ticker, impact, tick):
        """Record a news prediction for potential future aggregation."""
        if impact > 0:
            direction = "bullish"
        elif impact < 0:
            direction = "bearish"
        else:
            return
        
        self.recent_news_predictions.append({
            "ticker": ticker,
            "impact": impact,
            "tick": tick,
            "direction": direction
        })
        
        cutoff_tick = tick - NEWS_AGGREGATION_WINDOW
        self.recent_news_predictions = [
            n for n in self.recent_news_predictions
            if n["tick"] > cutoff_tick
        ]
    
    def submit_order(self, ticker, quantity, order_type="MARKET"):
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
        positions = self.get_positions()
        current_pos = positions.get(ticker, 0)
        is_closed = (current_pos == expected_position)
        return is_closed, current_pos
    
    def close_position_with_verification(self, ticker, quantity_to_close):
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
        
        self.current_high_conviction_net_pct = 0.0
        
        self.force_flatten_all_positions()
    
    def check_exit_conditions(self):
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
        
        if trade.get("strategy") in ["high_conviction_bullish", "high_conviction_bearish"]:
            self.current_high_conviction_net_pct = 0.0
        
        self.force_flatten_all_positions()
    
    def check_news(self):
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
        headline = news.get("headline", "")
        body = news.get("body", "")
        ticker_field = news.get("ticker", "")
        
        print(f"\n{'─'*70}")
        print(f"📰 NEWS {news.get('news_id')} | {ticker_field}")
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
        
        # Predict impact for each ticker AND aggregate with recent news
        predictions = {}
        for ticker in affected_tickers:
            base_impact = self.learner.predict_impact(ticker, headline, body)
            
            aggregated_impact, num_combined = self.get_aggregated_impact(ticker, base_impact, current_tick)
            
            self.record_news_prediction(ticker, base_impact, current_tick)
            
            price = self.get_ticker_price(ticker)
            dollar_move = abs(aggregated_impact * price)
            
            predictions[ticker] = {
                "impact": aggregated_impact,
                "base_impact": base_impact,
                "num_combined": num_combined,
                "price": price,
                "dollar_move": dollar_move
            }
            
            combined_str = f" (COMBINED ×{num_combined})" if num_combined > 1 else ""
            print(f"   Predicted {ticker}: {aggregated_impact:+.2%}{combined_str} | Price: ${price:.2f} | Dollar: ${dollar_move:.2f}")
        
        # Categorize opportunities using AGGREGATED impacts
        high_conviction_opportunities = []
        normal_opportunities = []
        
        for ticker, pred in predictions.items():
            impact = pred["impact"]
            dollar_move = pred["dollar_move"]
            
            # HIGH CONVICTION BULLISH: Only if ticker is in ALLOWED list
            if impact >= HIGH_CONVICTION_BULLISH:
                if ticker in ALLOWED_LONG_TICKERS:
                    net_pct = impact - HIGH_CONVICTION_BULLISH
                    high_conviction_opportunities.append({
                        "ticker": ticker,
                        "impact": impact,
                        "direction": "bullish",
                        "net_pct": net_pct,
                        "price": pred["price"],
                        "dollar_move": dollar_move,
                        "num_combined": pred["num_combined"]
                    })
                else:
                    print(f"   🚫 SKIPPING BULLISH {ticker}: Not in allowed LONG list (only {', '.join(ALLOWED_LONG_TICKERS)})")
            
            # HIGH CONVICTION BEARISH: Any ticker allowed for shorting
            elif impact <= HIGH_CONVICTION_BEARISH:
                net_pct = abs(impact - HIGH_CONVICTION_BEARISH)
                high_conviction_opportunities.append({
                    "ticker": ticker,
                    "impact": impact,
                    "direction": "bearish",
                    "net_pct": net_pct,
                    "price": pred["price"],
                    "dollar_move": dollar_move,
                    "num_combined": pred["num_combined"]
                })
            
            # NORMAL BULLISH: Only if ticker is in ALLOWED list
            elif impact >= BULLISH_THRESHOLD:
                if ticker in ALLOWED_LONG_TICKERS:
                    normal_opportunities.append({
                        "ticker": ticker,
                        "impact": impact,
                        "direction": "bullish",
                        "price": pred["price"],
                        "dollar_move": dollar_move,
                        "num_combined": pred["num_combined"]
                    })
                else:
                    print(f"   🚫 SKIPPING BULLISH {ticker}: Not in allowed LONG list (only {', '.join(ALLOWED_LONG_TICKERS)})")
            
            # NORMAL BEARISH: Any ticker allowed for shorting
            elif impact <= BEARISH_THRESHOLD:
                normal_opportunities.append({
                    "ticker": ticker,
                    "impact": impact,
                    "direction": "bearish",
                    "price": pred["price"],
                    "dollar_move": dollar_move,
                    "num_combined": pred["num_combined"]
                })
        
        # Select best high conviction opportunity by NET PERCENTAGE
        if len(high_conviction_opportunities) > 1:
            print(f"\n   🎯 MULTIPLE HIGH CONVICTION:")
            for opp in high_conviction_opportunities:
                combined_str = f" (COMBINED ×{opp['num_combined']})" if opp['num_combined'] > 1 else ""
                action_str = "BUY/LONG" if opp['direction'] == "bullish" else "SELL/SHORT"
                print(f"      {opp['ticker']}: {opp['impact']:+.2%}{combined_str} | Net: {opp['net_pct']:+.2%} | ${opp['dollar_move']:.2f} → {action_str}")
            
            high_conviction_opportunities.sort(key=lambda x: x['net_pct'], reverse=True)
            best_opp = high_conviction_opportunities[0]
            
            print(f"   ✅ SELECTED: {best_opp['ticker']} with {best_opp['net_pct']:+.2%} net percentage (highest)")
            high_conviction_opportunities = [best_opp]
        
        # Compare with current position (if any)
        if high_conviction_opportunities:
            best_opp = high_conviction_opportunities[0]
            new_net_pct = best_opp['net_pct']
            
            if self.current_high_conviction_net_pct > 0:
                print(f"\n   📊 COMPARISON:")
                print(f"      Current high conviction: {self.current_high_conviction_net_pct:+.2%} net percentage")
                combined_str = f" (COMBINED ×{best_opp['num_combined']})" if best_opp['num_combined'] > 1 else ""
                print(f"      New opportunity: {new_net_pct:+.2%} net percentage{combined_str}")
                
                if self.current_high_conviction_net_pct >= new_net_pct:
                    print(f"   🛡️ HOLDING CURRENT POSITION (larger net percentage)")
                    print(f"   ⏭️ Skipping new opportunity")
                    high_conviction_opportunities = []
                else:
                    print(f"   🔄 NEW OPPORTUNITY IS BETTER (higher net percentage)")
                    print(f"   📤 Will exit current and enter new position")
        
        # Execute high conviction trades
        for opp in high_conviction_opportunities:
            ticker = opp['ticker']
            impact = opp['impact']
            direction = opp['direction']
            net_pct = opp['net_pct']
            dollar_move = opp['dollar_move']
            num_combined = opp['num_combined']
            
            combined_str = f" [COMBINED ×{num_combined}]" if num_combined > 1 else ""
            
            # BULLISH = BUY/LONG the ticker
            if direction == "bullish":
                print(f"\n   🟢🟢 HIGH CONVICTION BULLISH: {ticker} ({impact:+.2%}{combined_str}, net: {net_pct:+.2%})")
                print(f"   💡 TRADING DECISION: BUY/LONG {ticker} (predicted price increase)")
                print(f"   ✅ {ticker} is in ALLOWED LONG list")
                print(f"   📊 Using AGGREGATED impact: {impact:+.2%} from {num_combined} news item(s)")
                
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
                
                print(f"      → [2/2] LONG {PRIMARY_SIZE:,} {ticker} (PRIMARY - BUYING)")
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
                    
                    self.current_high_conviction_net_pct = net_pct
                    
                    final_positions = self.get_positions()
                    final_gross, final_net = self.calculate_position_limits(final_positions)
                    
                    print(f"      ✓ Trade #{trade_id} @ tick {current_tick} | Gross={final_gross:,} Net={final_net:+,}")
                    print(f"      📈 Tracking: {net_pct:+.2%} net % | LONG {ticker} on +{impact:+.2%} combined prediction")
                    print(f"      ⏰ Exit @ tick {current_tick + EXIT_AFTER_TICKS}")
            
            # BEARISH = SELL/SHORT the ticker (any ticker allowed)
            elif direction == "bearish":
                print(f"\n   🔴🔴 HIGH CONVICTION BEARISH: {ticker} ({impact:+.2%}{combined_str}, net: {net_pct:+.2%})")
                print(f"   💡 TRADING DECISION: SELL/SHORT {ticker} (predicted price decrease)")
                print(f"   📊 Using AGGREGATED impact: {impact:+.2%} from {num_combined} news item(s)")
                
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
                
                print(f"      → [2/2] SHORT {PRIMARY_SIZE:,} {ticker} (PRIMARY - SELLING)")
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
                    
                    self.current_high_conviction_net_pct = net_pct
                    
                    final_positions = self.get_positions()
                    final_gross, final_net = self.calculate_position_limits(final_positions)
                    
                    print(f"      ✓ Trade #{trade_id} @ tick {current_tick} | Gross={final_gross:,} Net={final_net:+,}")
                    print(f"      📉 Tracking: {net_pct:+.2%} net % | SHORT {ticker} on {impact:+.2%} combined prediction")
                    print(f"      ⏰ Exit @ tick {current_tick + EXIT_AFTER_TICKS}")
        
        # Process normal opportunities (if no high conviction)
        if not high_conviction_opportunities:
            if len(normal_opportunities) > 1:
                print(f"\n   💰 Multiple opportunities - prioritizing by dollar move:")
                for opp in normal_opportunities:
                    combined_str = f" (COMBINED ×{opp['num_combined']})" if opp['num_combined'] > 1 else ""
                    print(f"      {opp['ticker']}: {opp['impact']:+.2%}{combined_str} | ${opp['dollar_move']:.2f} move")
                
                normal_opportunities.sort(key=lambda x: x['dollar_move'], reverse=True)
                print(f"   ✅ SELECTED: {normal_opportunities[0]['ticker']} with ${normal_opportunities[0]['dollar_move']:.2f} dollar move")
                normal_opportunities = [normal_opportunities[0]]
            
            for opp in normal_opportunities:
                ticker = opp['ticker']
                impact = opp['impact']
                direction = opp['direction']
                num_combined = opp['num_combined']
                
                combined_str = f" [COMBINED ×{num_combined}]" if num_combined > 1 else ""
                
                if direction == "bullish":
                    print(f"\n   🟢 BULLISH: {ticker} ({impact:+.2%}{combined_str}, ${opp['dollar_move']:.2f} move)")
                    print(f"   💡 DECISION: BUY/LONG {ticker} (predicted +{impact:+.2%})")
                    print(f"   ✅ {ticker} is in ALLOWED LONG list")
                    
                    hedge_ticker = self.choose_hedge_ticker(ticker, headline, body)
                    if not hedge_ticker:
                        print(f"      ✗ No hedge available")
                        continue
                    
                    print(f"      Strategy: SHORT {HEDGE_SIZE:,} hedge FIRST → LONG {PRIMARY_SIZE:,} primary")
                    
                    print(f"      → [1/2] SHORT {HEDGE_SIZE:,} {hedge_ticker} (HEDGE)")
                    short_qty = self.submit_order(hedge_ticker, -HEDGE_SIZE)
                    
                    if short_qty == 0:
                        continue
                    
                    print(f"      → [2/2] LONG {PRIMARY_SIZE:,} {ticker} (PRIMARY - BUYING)")
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
                    print(f"\n   🔴 BEARISH: {ticker} ({impact:+.2%}{combined_str}, ${opp['dollar_move']:.2f} move)")
                    print(f"   💡 DECISION: SELL/SHORT {ticker} (predicted {impact:+.2%})")
                    
                    hedge_ticker = self.choose_hedge_ticker(ticker, headline, body)
                    if not hedge_ticker:
                        print(f"      ✗ No hedge available")
                        continue
                    
                    print(f"      Strategy: LONG {HEDGE_SIZE:,} hedge FIRST → SHORT {PRIMARY_SIZE:,} primary")
                    
                    print(f"      → [1/2] LONG {HEDGE_SIZE:,} {hedge_ticker} (HEDGE)")
                    long_qty = self.submit_order(hedge_ticker, HEDGE_SIZE)
                    
                    if long_qty == 0:
                        continue
                    
                    print(f"      → [2/2] SHORT {PRIMARY_SIZE:,} {ticker} (PRIMARY - SELLING)")
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
        print(f"\n{'='*80}")
        print("🤖 M&A TRADING BOT - WORD SENTIMENT + NET % PRIORITY + AGGREGATION")
        print(f"{'='*80}")
        print(f"Normal: ≥{BULLISH_THRESHOLD:+.2%} bull | ≤{BEARISH_THRESHOLD:+.2%} bear")
        print(f"HIGH CONVICTION: ≥{HIGH_CONVICTION_BULLISH:+.2%} bull | ≤{HIGH_CONVICTION_BEARISH:+.2%} bear")
        print(f"\n🔒 LONG RESTRICTIONS: Can only BUY/LONG → {', '.join(sorted(ALLOWED_LONG_TICKERS))}")
        print(f"   SHORT: Any ticker allowed")
        print(f"\nSentiments: {len(self.learner.word_sentiments)} words loaded")
        print(f"Min word matches: {MIN_WORD_MATCHES}")
        print(f"Weighted by count: {WORD_WEIGHT_BY_COUNT}")
        print(f"\n📊 News Aggregation:")
        print(f"  • Window: {NEWS_AGGREGATION_WINDOW} ticks")
        print(f"  • Same ticker + same direction → ADD impacts together")
        print(f"  • Example: +0.8% + +0.5% = +1.3% combined impact")
        print(f"\n🎯 Trading Logic:")
        print(f"  • POSITIVE combined prediction → BUY/LONG that ticker (if in allowed list)")
        print(f"  • NEGATIVE combined prediction → SELL/SHORT that ticker (any ticker)")
        print(f"  • Uses AGGREGATED (combined) impact for all trading decisions")
        print(f"\n🏆 Priority:")
        print(f"  • Multiple opportunities → Select HIGHEST NET PERCENTAGE")
        print(f"  • Net %: Distance from threshold (e.g., 2.5% - 1.1% = 1.4% net)")
        print(f"\n🛡️ Hold Logic:")
        print(f"  • If current high conviction has LARGER net % → KEEP position")
        print(f"  • Only exit if new opportunity has HIGHER net %")
        print(f"\n⚙️ Execution:")
        print(f"  • BULLISH: SHORT hedge → LONG primary")
        print(f"  • BEARISH: LONG hedge → SHORT primary")
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
    if not WORD_SENTIMENT_CSV.exists():
        print(f"ERROR: Word sentiment CSV not found at {WORD_SENTIMENT_CSV}")
        return
    
    learner = WordSentimentScorer(WORD_SENTIMENT_CSV)
    trader = MergerArbTrader(learner)
    
    if not trader.wait_for_case_start():
        print("Exiting...")
        return
    
    trader.run(check_interval=1)


if __name__ == "__main__":
    main()
