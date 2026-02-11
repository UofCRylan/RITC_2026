import asyncio
import aiohttp
import sys

# --- CONFIGURATION ---
API_KEY = "RYL2000"
API_URL = "http://localhost:10000/v1"
SYMBOLS = ["SPNG", "SMMR", "ATMN", "WNTR"] 

# --- STRATEGY SETTINGS ---
QTY = 5000
SPREAD = 0.08
MAX_POS = 20000

# The aggression of the mean reversion
# Higher = We adjust prices more drastically when basket deviates
# 0.25 means if Basket is off by $1.00, we shift quotes by $0.25 per asset
REVERSION_FACTOR = 0.25 

class SharedState:
    """Stores the latest price of every ticker to calculate global basket value"""
    def __init__(self):
        self.prices = {s: 0.0 for s in SYMBOLS}
        self.positions = {s: 0 for s in SYMBOLS}
        self.initial_basket_sum = None # Will set on first run
        self.ready = False

class RITC_Basket_Bot:
    def __init__(self):
        self.session = None
        self.headers = {'X-API-Key': API_KEY}
        self.state = SharedState()

    async def update_market_data(self, ticker):
        """Dedicated loop just to keep price data fresh"""
        while True:
            try:
                async with self.session.get(f"{API_URL}/securities?ticker={ticker}", headers=self.headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data:
                            t = data[0]
                            # Update Shared State
                            mid = (t['bid'] + t['ask']) / 2
                            self.state.prices[ticker] = mid
                            self.state.positions[ticker] = t['position']
            except:
                pass
            
            # Update frequently
            await asyncio.sleep(0.05)

    async def strategy_loop(self):
        """The Master Brain Loop"""
        print("⏳ CALIBRATING BASKET...")
        await asyncio.sleep(1) # Wait for data to populate
        
        # Set the "Fair Value" anchor based on current market
        current_sum = sum(self.state.prices.values())
        if current_sum == 0:
            print("❌ ERROR: No price data. Check API.")
            return

        self.state.initial_basket_sum = current_sum
        self.state.ready = True
        print(f"✅ BASKET LOCKED: {current_sum:.2f}")
        print("🚀 RUNNING ARBITRAGE...")

        while True:
            # 1. Calculate Global Deviation
            # e.g., if prices sum to 102.00 and Anchor is 100.00, Deviation is +2.00
            live_sum = sum(self.state.prices.values())
            deviation = live_sum - self.state.initial_basket_sum
            
            # 2. Correction per asset
            # If Deviation is +2.00 (Expensive), correction is -0.50 per asset (to bring sum down)
            correction = -(deviation * REVERSION_FACTOR)

            # 3. Execute Trades for ALL tickers based on Global Signal
            tasks = []
            for ticker in SYMBOLS:
                tasks.append(self.execute_ticker(ticker, correction))
            
            await asyncio.gather(*tasks)
            await asyncio.sleep(0.1) # Loop speed

    async def execute_ticker(self, ticker, global_correction):
        """Calculates individual quotes adding the Global Correction + Inventory Hedge"""
        
        mid = self.state.prices[ticker]
        pos = self.state.positions[ticker]
        
        # A. Inventory Management (Local Risk)
        # If we are Long, lower price to sell.
        inventory_skew = -(pos / MAX_POS) * 0.05
        
        # B. Fair Value Calculation
        # Fair Value = Current Mid + Global Correction + Inventory Skew
        # Example: Mid 25.00. Basket is Expensive (Correction -0.10). Long Position (Skew -0.02)
        # Target = 24.88
        target_price = mid + global_correction + inventory_skew
        
        my_bid = target_price - (SPREAD / 2)
        my_ask = target_price + (SPREAD / 2)
        
        # Format
        b_str = f"{my_bid:.2f}"
        s_str = f"{my_ask:.2f}"
        
        # API Calls
        # Cancel old
        await self.session.post(f"{API_URL}/commands/cancel?ticker={ticker}", headers=self.headers)
        
        # Place new
        if pos < MAX_POS:
            url = f"{API_URL}/orders?ticker={ticker}&type=LIMIT&quantity={QTY}&action=BUY&price={b_str}"
            asyncio.create_task(self.session.post(url, headers=self.headers))
            
        if pos > -MAX_POS:
            url = f"{API_URL}/orders?ticker={ticker}&type=LIMIT&quantity={QTY}&action=SELL&price={s_str}"
            asyncio.create_task(self.session.post(url, headers=self.headers))

    async def run(self):
        connector = aiohttp.TCPConnector(limit=0, family=2)
        async with aiohttp.ClientSession(connector=connector) as session:
            self.session = session
            
            # Start data collectors
            collectors = [asyncio.create_task(self.update_market_data(s)) for s in SYMBOLS]
            
            # Start Strategy
            await self.strategy_loop()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    bot = RITC_Basket_Bot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        pass