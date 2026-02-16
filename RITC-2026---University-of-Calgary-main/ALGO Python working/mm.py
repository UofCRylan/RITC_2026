# ==========================================
# SEASONAL MARKET MAKER - 4 SECURITIES
# ==========================================

import asyncio
import aiohttp
import ujson
import time
import signal
import sys
import gc
import re
from typing import Dict, Optional
from dataclasses import dataclass, field

# ==========================================
# CORE PARAMETERS
# ==========================================

# Risk Limits (will be set dynamically from news)
GROSS_LIMIT_TOTAL = 0  # Will be parsed from news
LIMIT_PER_SECURITY = 0  # Will be calculated as GROSS_LIMIT_TOTAL / 4

# Base Order Sizes 
BASE_SIZE = 2500

# Pricing Constants - Simplified strategy
SPREAD_THRESHOLD = 0.05  # 5 cents
PRICE_IMPROVEMENT = 0.01  # 1 cent

# Performance Constants
TOKEN_BUCKET_CAPACITY = 50
TOKEN_REFILL_RATE_SEC = 0.001
FAST_CYCLE_MS = 0.01
REQUEST_TIMEOUT = 0.15

# Recovery Constants
MAX_CONSECUTIVE_FAILURES = 5
NO_TRADE_TIMEOUT = 30.0
MIN_TOKENS_TO_TRADE = 1

MIN_ORDER_SIZE = 300  

# Global Configuration
API_KEY = 'ZVVCI8DU'
BASE_URL = 'http://localhost:9999/v1'
TICKERS = ['SPNG', 'SMMR', 'ATMN', 'WNTR']
SHUTDOWN = False

# ==========================================
# DATA STRUCTURES
# ==========================================

@dataclass
class HeartbeatTracker:
    last_order_attempt: float = field(default_factory=time.time)
    last_order_success: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    total_attempts: int = 0
    total_successes: int = 0
    
    def record_attempt(self, success: bool):
        self.last_order_attempt = time.time()
        self.total_attempts += 1
        
        if success:
            self.last_order_success = time.time()
            self.consecutive_failures = 0
            self.total_successes += 1
        else:
            self.consecutive_failures += 1
    
    def is_stuck(self) -> bool:
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            return True
        
        now = time.time()
        time_since_success = now - self.last_order_success
        
        if time_since_success > NO_TRADE_TIMEOUT:
            time_since_attempt = now - self.last_order_attempt
            if time_since_attempt < 10.0:
                return True
        
        return False

@dataclass
class TokenBucket:
    tokens: int = TOKEN_BUCKET_CAPACITY
    last_refill: float = field(default_factory=time.monotonic)
    last_checked: float = field(default_factory=time.monotonic)
    empty_streak: int = 0
    
    def consume(self, tokens: int) -> bool:
        now = time.monotonic()
        
        if now - self.last_checked > 0.1:
            elapsed = now - self.last_refill
            refill = int(elapsed / TOKEN_REFILL_RATE_SEC)
            if refill > 0:
                self.tokens = min(TOKEN_BUCKET_CAPACITY, self.tokens + refill)
                self.last_refill = now
            self.last_checked = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            self.empty_streak = 0
            return True
        else:
            self.empty_streak += 1
            return False
    
    def force_refill(self):
        self.tokens = TOKEN_BUCKET_CAPACITY
        self.last_refill = time.monotonic()
        self.empty_streak = 0

@dataclass
class TickerState:
    ticker: str
    
    # Server state
    server_position: int = 0
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread: float = 0.0
    
    # Shadow ledger
    shadow_position: int = 0
    pending_buy_qty: int = 0
    pending_sell_qty: int = 0
    
    # Active orders
    active_buy_id: Optional[int] = None
    active_sell_id: Optional[int] = None
    active_buy_price: float = 0.0
    active_sell_price: float = 0.0
    
    # Pessimistic accounting
    worst_case_long: int = 0
    worst_case_short: int = 0
    
    # Flight control
    buy_in_flight: bool = False
    sell_in_flight: bool = False
    
    # Performance
    token_bucket: TokenBucket = field(default_factory=TokenBucket)
    
    # Heartbeat tracking
    heartbeat: HeartbeatTracker = field(default_factory=HeartbeatTracker)
    
    @property
    def effective_position(self) -> int:
        return self.server_position + self.pending_buy_qty - self.pending_sell_qty
    
    def update_pessimistic_exposure(self):
        current_exposure = abs(self.server_position)
        self.worst_case_long = current_exposure + self.pending_buy_qty
        self.worst_case_short = current_exposure + self.pending_sell_qty
        return max(self.worst_case_long, self.worst_case_short)
    
    def get_base_size(self) -> int:
        return BASE_SIZE
    
    def get_ticker_limit(self) -> int:
        return LIMIT_PER_SECURITY

# ==========================================
# TICKER ENGINE - SIMPLIFIED
# ==========================================

class TickerEngine:
    def __init__(self, ticker: str):
        self.state = TickerState(ticker=ticker)
        self.last_cycle = 0.0
    
    async def update_market_data(self, session: aiohttp.ClientSession) -> bool:
        """Fetch current market state"""
        try:
            url = f"{BASE_URL}/securities/book?ticker={self.state.ticker}&limit=1&key={API_KEY}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
                if resp.status != 200:
                    return False
                
                data = await resp.json(loads=ujson.loads)
                
                if not data or 'bids' not in data or 'asks' not in data:
                    return False
                
                bids = data.get('bids', [])
                asks = data.get('asks', [])
                
                if not bids or not asks:
                    return False
                
                self.state.best_bid = float(bids[0]['price'])
                self.state.best_ask = float(asks[0]['price'])
                self.state.spread = self.state.best_ask - self.state.best_bid
                
                return True
                
        except Exception as e:
            return False
    
    async def update_position(self, session: aiohttp.ClientSession) -> bool:
        """Fetch current position"""
        try:
            url = f"{BASE_URL}/securities?ticker={self.state.ticker}&key={API_KEY}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
                if resp.status != 200:
                    return False
                
                data = await resp.json(loads=ujson.loads)
                
                if isinstance(data, list) and len(data) > 0:
                    self.state.server_position = data[0].get('position', 0)
                    return True
                
                return False
                
        except Exception:
            return False
    
    async def sync_orders(self, session: aiohttp.ClientSession) -> bool:
        """Sync active orders with server"""
        try:
            url = f"{BASE_URL}/orders?status=OPEN&key={API_KEY}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
                if resp.status != 200:
                    return False
                
                orders = await resp.json(loads=ujson.loads)
                
                # Reset tracking
                found_buy = False
                found_sell = False
                self.state.pending_buy_qty = 0
                self.state.pending_sell_qty = 0
                
                for order in orders:
                    if order.get('ticker') != self.state.ticker:
                        continue
                    
                    order_id = order.get('order_id')
                    action = order.get('action')
                    qty = order.get('quantity', 0)
                    qty_left = order.get('quantity_left', qty)
                    price = float(order.get('price', 0))
                    
                    if action == 'BUY':
                        if not found_buy:
                            self.state.active_buy_id = order_id
                            self.state.active_buy_price = price
                            self.state.pending_buy_qty = qty_left
                            found_buy = True
                    elif action == 'SELL':
                        if not found_sell:
                            self.state.active_sell_id = order_id
                            self.state.active_sell_price = price
                            self.state.pending_sell_qty = qty_left
                            found_sell = True
                
                if not found_buy:
                    self.state.active_buy_id = None
                    self.state.active_buy_price = 0.0
                    self.state.buy_in_flight = False
                
                if not found_sell:
                    self.state.active_sell_id = None
                    self.state.active_sell_price = 0.0
                    self.state.sell_in_flight = False
                
                return True
                
        except Exception:
            return False
    
    def calculate_target_prices(self) -> tuple[float, float]:
        """
        Calculate target bid/ask prices based on spread
        
        Strategy:
        - If spread <= 5 cents: bid at best bid, ask at best ask
        - If spread > 5 cents: bid at best bid + 1 cent, ask at best ask - 1 cent
        """
        if self.state.spread <= SPREAD_THRESHOLD:
            # Tight spread: match the market
            target_bid = self.state.best_bid
            target_ask = self.state.best_ask
        else:
            # Wide spread: improve inside
            target_bid = self.state.best_bid + PRICE_IMPROVEMENT
            target_ask = self.state.best_ask - PRICE_IMPROVEMENT
        
        return target_bid, target_ask
    
    def should_place_order(self, side: str, target_price: float) -> tuple[bool, int]:
        """Determine if we should place an order and what size"""
        
        # Check exposure limits
        current_exposure = abs(self.state.server_position)
        ticker_limit = self.state.get_ticker_limit()
        
        if current_exposure >= ticker_limit:
            return False, 0
        
        # Calculate available size
        available = ticker_limit - current_exposure
        base_size = self.state.get_base_size()
        order_size = min(base_size, available)
        
        if order_size < MIN_ORDER_SIZE:
            return False, 0
        
        # Check if price needs updating
        if side == 'BUY':
            if self.state.active_buy_id is not None:
                if abs(self.state.active_buy_price - target_price) < 0.005:
                    return False, 0
            return True, order_size
        else:  # SELL
            if self.state.active_sell_id is not None:
                if abs(self.state.active_sell_price - target_price) < 0.005:
                    return False, 0
            return True, order_size
    
    async def place_order(self, session: aiohttp.ClientSession, side: str, 
                         price: float, quantity: int) -> bool:
        """Place a limit order"""
        
        if self.state.buy_in_flight and side == 'BUY':
            return False
        if self.state.sell_in_flight and side == 'SELL':
            return False
        
        if not self.state.token_bucket.consume(1):
            return False
        
        try:
            # Set flight control
            if side == 'BUY':
                self.state.buy_in_flight = True
            else:
                self.state.sell_in_flight = True
            
            payload = {
                'ticker': self.state.ticker,
                'type': 'LIMIT',
                'action': side,
                'quantity': quantity,
                'price': round(price, 2)
            }
            
            url = f"{BASE_URL}/orders?key={API_KEY}"
            async with session.post(url, params=payload, 
                                   timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
                
                if resp.status == 200:
                    self.state.heartbeat.record_attempt(True)
                    return True
                else:
                    self.state.heartbeat.record_attempt(False)
                    return False
                    
        except Exception:
            self.state.heartbeat.record_attempt(False)
            return False
        finally:
            if side == 'BUY':
                self.state.buy_in_flight = False
            else:
                self.state.sell_in_flight = False
    
    async def cancel_order(self, session: aiohttp.ClientSession, order_id: int) -> bool:
        """Cancel a specific order"""
        try:
            url = f"{BASE_URL}/commands/cancel?order_id={order_id}&key={API_KEY}"
            async with session.post(url, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
                return resp.status == 200
        except Exception:
            return False
    
    async def run_cycle(self, session: aiohttp.ClientSession, now: float) -> bool:
        """Execute one trading cycle - simplified"""
        
        # Update pessimistic exposure
        self.state.update_pessimistic_exposure()
        
        # Calculate target prices
        target_bid, target_ask = self.calculate_target_prices()
        
        # Decide on buy order
        should_buy, buy_size = self.should_place_order('BUY', target_bid)
        if should_buy:
            # Cancel existing buy if needed
            if self.state.active_buy_id is not None:
                await self.cancel_order(session, self.state.active_buy_id)
                await asyncio.sleep(0.01)
            
            # Place new buy
            await self.place_order(session, 'BUY', target_bid, buy_size)
        
        # Decide on sell order
        should_sell, sell_size = self.should_place_order('SELL', target_ask)
        if should_sell:
            # Cancel existing sell if needed
            if self.state.active_sell_id is not None:
                await self.cancel_order(session, self.state.active_sell_id)
                await asyncio.sleep(0.01)
            
            # Place new sell
            await self.place_order(session, 'SELL', target_ask, sell_size)
        
        self.last_cycle = now
        return should_buy or should_sell

# ==========================================
# GLOBAL ORCHESTRATOR
# ==========================================

class SeasonalMarketMaker:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.engines: Dict[str, TickerEngine] = {}
        
        self.cycles = 0
        self.last_status_print = 0.0
        self.endgame_active = False
        
        for ticker in TICKERS:
            self.engines[ticker] = TickerEngine(ticker)
    
    async def initialize(self) -> bool:
        """Initialize the bot"""
        print("="*80)
        print("🚀 SEASONAL MARKET MAKER - INITIALIZING")
        print("="*80)
        
        # Fetch limits from news
        if not await self.fetch_limits_from_news():
            print("❌ Failed to fetch limits from news")
            return False
        
        print(f"\n✅ Limits configured:")
        print(f"   Total Gross Limit: {GROSS_LIMIT_TOTAL:,}")
        print(f"   Limit per Security: {LIMIT_PER_SECURITY:,}")
        
        # Create session
        connector = aiohttp.TCPConnector(
            limit=100,
            ttl_dns_cache=300,
            force_close=False,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            json_serialize=ujson.dumps
        )
        
        print(f"\n📊 Trading {len(TICKERS)} securities: {', '.join(TICKERS)}")
        print("="*80)
        
        return True
    
    async def fetch_limits_from_news(self) -> bool:
        """
        Fetch news and parse for 'shares' keyword to extract gross limit
        
        Looks for patterns like:
        - "10000 shares"
        - "25,000 shares"
        - "25000 shares"
        """
        global GROSS_LIMIT_TOTAL, LIMIT_PER_SECURITY
        
        try:
            # Create temporary session for news fetch
            async with aiohttp.ClientSession() as session:
                url = f"{BASE_URL}/news?key={API_KEY}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2.0)) as resp:
                    if resp.status != 200:
                        print(f"⚠️  News fetch failed with status {resp.status}")
                        return False
                    
                    news_data = await resp.json()
                    
                    # Parse news for share limits
                    share_numbers = []
                    
                    for news_item in news_data:
                        body = news_item.get('body', '')
                        headline = news_item.get('headline', '')
                        text = f"{headline} {body}"
                        
                        # Look for patterns like "NUMBER shares"
                        # Match patterns: "10000 shares", "25,000 shares", etc.
                        pattern = r'(\d{1,3}(?:,\d{3})*|\d+)\s+shares'
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        
                        for match in matches:
                            # Remove commas and convert to int
                            number = int(match.replace(',', ''))
                            share_numbers.append(number)
                            print(f"   Found: {number:,} shares in news")
                    
                    if not share_numbers:
                        print("⚠️  No share limits found in news, using default")
                        GROSS_LIMIT_TOTAL = 25000
                    else:
                        # Use the maximum share number found
                        GROSS_LIMIT_TOTAL = max(share_numbers)
                        print(f"\n✅ Extracted gross limit: {GROSS_LIMIT_TOTAL:,} shares")
                    
                    # Divide evenly among 4 securities
                    LIMIT_PER_SECURITY = GROSS_LIMIT_TOTAL // 4
                    
                    return True
                    
        except Exception as e:
            print(f"⚠️  Error fetching news: {e}")
            # Use default if news fetch fails
            GROSS_LIMIT_TOTAL = 25000
            LIMIT_PER_SECURITY = GROSS_LIMIT_TOTAL // 4
            return True
    
    async def parallel_data_fetch(self):
        """Fetch all market data in parallel"""
        tasks = []
        
        for engine in self.engines.values():
            tasks.append(engine.update_market_data(self.session))
            tasks.append(engine.update_position(self.session))
            tasks.append(engine.sync_orders(self.session))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def calculate_current_gross(self) -> int:
        """Calculate total gross exposure across all tickers"""
        total = 0
        for engine in self.engines.values():
            total += abs(engine.state.server_position)
        return total
    
    def print_status(self):
        """Print current status"""
        now = time.time()
        if now - self.last_status_print < 1.0:
            return
        
        print("\033[2J\033[H")
        print("="*100)
        print(f"SEASONAL MARKET MAKER | Cycle: {self.cycles} | Limit: {GROSS_LIMIT_TOTAL:,} ({LIMIT_PER_SECURITY:,}/security)")
        print("="*100)
        
        total_exposure = self.calculate_current_gross()
        utilization = (total_exposure / GROSS_LIMIT_TOTAL * 100) if GROSS_LIMIT_TOTAL > 0 else 0
        health_color = "🟢" if utilization < 60 else "🟡" if utilization < 80 else "🔴"
        
        print(f"{health_color} EXPOSURE: {total_exposure:>5}/{GROSS_LIMIT_TOTAL} ({utilization:>3.0f}%)")
        
        print("-"*100)
        print(f"{'Ticker':<6} {'Pos':>6} {'Limit':>6} {'Bid/Ask':>12} {'Spread':>7} {'Active':>7} {'Success%':>9}")
        print("-"*100)
        
        for ticker, engine in self.engines.items():
            state = engine.state
            
            success_rate = 0
            if state.heartbeat.total_attempts > 0:
                success_rate = (state.heartbeat.total_successes / state.heartbeat.total_attempts) * 100
            
            active_ind = ""
            if state.active_buy_id:
                active_ind += "B"
            if state.active_sell_id:
                active_ind += "S"
            
            print(f"{ticker:<6} {state.server_position:>6} {LIMIT_PER_SECURITY:>6} "
                  f"{state.best_bid:.3f}/{state.best_ask:.3f} {state.spread*100:>6.1f}c "
                  f"{active_ind:>7} {success_rate:>8.1f}%")
        
        print("="*100)
        self.last_status_print = now
    
    async def _check_endgame(self):
        """Check if we're approaching end of trading"""
        try:
            url = f"{BASE_URL}/case?key={API_KEY}"
            async with self.session.get(url, timeout=0.5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('ticks_remaining', 999) <= 1:
                        await self.endgame_protocol()
                        return True
        except Exception:
            pass
        return False
    
    async def endgame_protocol(self):
        """Close all positions at end of game"""
        self.endgame_active = True
        
        print("\n" + "="*80)
        print("🎯 ENDGAME: Closing all positions")
        print("="*80)
        
        # Cancel all orders
        cancel_url = f"{BASE_URL}/commands/cancel?all=1&key={API_KEY}"
        try:
            async with self.session.post(cancel_url, timeout=1.0) as resp:
                await resp.read()
        except Exception:
            pass
        
        # Close positions with market orders
        for ticker, engine in self.engines.items():
            state = engine.state
            if abs(state.server_position) > 100:
                action = 'SELL' if state.server_position > 0 else 'BUY'
                
                payload = {
                    'ticker': ticker,
                    'type': 'MARKET',
                    'action': action,
                    'quantity': abs(state.server_position)
                }
                
                url = f"{BASE_URL}/orders?key={API_KEY}"
                try:
                    async with self.session.post(url, params=payload, timeout=1.0) as resp:
                        await resp.read()
                except Exception:
                    pass
        
        await asyncio.sleep(0.5)
        
        global SHUTDOWN
        SHUTDOWN = True
    
    async def main_loop(self):
        """Main trading loop"""
        print("✅ Starting main trading loop...")
        
        while not SHUTDOWN:
            now = time.monotonic()
            
            # Check for endgame
            if self.cycles % 100 == 0:
                if await self._check_endgame():
                    break
            
            # Fetch market data
            await self.parallel_data_fetch()
            
            # Execute trading for each ticker
            for engine in self.engines.values():
                await engine.run_cycle(self.session, now)
            
            # Print status
            if self.cycles % 100 == 0:
                self.print_status()
            
            self.cycles += 1
            
            # Sleep
            sleep_time = max(0, FAST_CYCLE_MS - (time.monotonic() - now))
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    
    async def cleanup(self):
        """Clean shutdown"""
        print("\n" + "="*80)
        print("🛑 SHUTTING DOWN")
        print("="*80)
        
        # Cancel all orders
        try:
            cancel_url = f"{BASE_URL}/commands/cancel?all=1&key={API_KEY}"
            async with self.session.post(cancel_url, timeout=1.0) as resp:
                await resp.read()
        except Exception:
            pass
        
        # Close session
        if self.session:
            await self.session.close()
        
        gc.enable()
        
        # Print final stats
        print(f"\n📊 FINAL STATISTICS")
        print(f"   Total Cycles: {self.cycles}")
        print(f"   Total Exposure: {self.calculate_current_gross():,}/{GROSS_LIMIT_TOTAL:,}")
        
        print(f"\n📈 PER-TICKER PERFORMANCE:")
        for ticker, engine in self.engines.items():
            heartbeat = engine.state.heartbeat
            success_rate = heartbeat.total_successes / max(1, heartbeat.total_attempts) * 100
            print(f"   {ticker}:")
            print(f"      Position: {engine.state.server_position:,}")
            print(f"      Attempts: {heartbeat.total_attempts}")
            print(f"      Successes: {heartbeat.total_successes}")
            print(f"      Success Rate: {success_rate:.1f}%")
        
        print("="*80)
        print("✅ Clean shutdown complete")

# ==========================================
# MAIN ENTRY POINT
# ==========================================

def signal_handler(signum, frame):
    global SHUTDOWN
    SHUTDOWN = True
    print("\n🛑 Shutdown signal received")

async def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    bot = SeasonalMarketMaker()
    
    try:
        if not await bot.initialize():
            print("❌ Failed to initialize. Exiting.")
            return
        
        await asyncio.sleep(0.5)
        await bot.main_loop()
        
    except KeyboardInterrupt:
        print("\n🛑 Manual shutdown")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await bot.cleanup()

if __name__ == "__main__":
    asyncio.run(main())