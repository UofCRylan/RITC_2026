import asyncio
import aiohttp
import sys

# --- USER CONFIGURATION ---
API_KEY = "RYL2000"
API_URL = "http://localhost:10000/v1"
# UPDATED SYMBOLS based on CSVs (Change to ["RY", "CNR"] if using old case)
SYMBOLS = ["SPNG", "SMMR", "ATMN", "WNTR"] 

# --- STRATEGY SETTINGS ---
SPREAD_TARGET = 0.08      # Base spread width
QTY = 5000                # Order Quantity
MAX_POSITION = 25000      # Max absolute position size
SLEEP_TIME = 0.0          # ⚡ MAX SPEED

# Momentum Settings
MOMENTUM_THRESHOLD = 0.60 # If 60% of book is on one side, trigger momentum
TICKS_TO_WAIT = 5         # Check momentum every 5 ticks

class RITC_Velocity_Bot:
    def __init__(self):
        self.session = None
        self.headers = {'X-API-Key': API_KEY}

    async def check_connection(self):
        try:
            async with self.session.get(f"{API_URL}/case", headers=self.headers) as resp:
                if resp.status == 200:
                    case_data = await resp.json()
                    tick = case_data.get('tick', 0)
                    print(f"✅ CONNECTED. Tick: {tick} | Speed: MAXIMUM")
                    return True
        except:
            print("❌ ERROR: RIT Client not found on Port 10000.")
        return False

    async def ticker_strategy(self, ticker):
        """
        Hybrid Strategy:
        1. Dynamic Spread Market Making (Inventory Skewing)
        2. Momentum (Book Depth Imbalance)
        """
        # Pre-compute URL parts
        base_cancel = f"{API_URL}/commands/cancel?ticker={ticker}"
        base_order = f"{API_URL}/orders?ticker={ticker}&type=LIMIT&quantity={QTY}&action="
        
        # Localize variables for speed
        session_post = self.session.post
        session_get = self.session.get
        headers = self.headers
        
        # Strategy State Variables
        last_check_tick = 0
        momentum_state = "NEUTRAL" # NEUTRAL, BULLISH, BEARISH
        
        print(f"⚡ LAUNCHING THREAD: {ticker}")
        
        try:
            while True:
                # 1. GET DATA
                async with session_get(f"{API_URL}/securities?ticker={ticker}", headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data:
                            t = data[0]
                            
                            current_tick = t['tick']
                            bid = t['bid']
                            ask = t['ask']
                            pos = t['position']
                            
                            # Safely get volumes (RIT API standard keys are bid_size/ask_size)
                            # Fallback to 0 if keys missing to prevent crash
                            bid_vol = t.get('bid_size', 0)
                            ask_vol = t.get('ask_size', 0)

                            # 2. MOMENTUM CHECK (Every 5 Ticks)
                            if current_tick - last_check_tick >= TICKS_TO_WAIT:
                                last_check_tick = current_tick
                                
                                total_vol = bid_vol + ask_vol
                                if total_vol > 0:
                                    buy_ratio = bid_vol / total_vol
                                    
                                    if buy_ratio > MOMENTUM_THRESHOLD:
                                        momentum_state = "BULLISH"
                                        # print(f"🚀 {ticker} MOMENTUM: BULLISH (Buy Vol: {buy_ratio:.2f})")
                                    elif buy_ratio < (1 - MOMENTUM_THRESHOLD):
                                        momentum_state = "BEARISH"
                                        # print(f"🔻 {ticker} MOMENTUM: BEARISH (Buy Vol: {buy_ratio:.2f})")
                                    else:
                                        momentum_state = "NEUTRAL"

                            # 3. CALCULATE PRICES
                            # Cancel existing orders first (simplest way to update)
                            asyncio.create_task(session_post(base_cancel, headers=headers))
                            
                            mid = (bid + ask) / 2
                            
                            # A. Dynamic Spread Logic (Inventory Skew)
                            # If we are long, lower prices to sell. If short, raise prices to buy.
                            skew = 0
                            if pos > 5000: skew = -0.02
                            if pos > 15000: skew = -0.04
                            if pos < -5000: skew = 0.02
                            if pos < -15000: skew = 0.04
                            
                            # Base prices
                            my_bid = mid - (SPREAD_TARGET / 2) + skew
                            my_ask = mid + (SPREAD_TARGET / 2) + skew

                            # B. Apply Momentum Override
                            # If Bullish: Buy Aggressively, Don't Sell (Hold)
                            # If Bearish: Sell Aggressively, Don't Buy (Hold Short)
                            
                            if momentum_state == "BULLISH":
                                # Strategy: Hold Longs, Buy More
                                # Shift bid up to capture fill, remove ask or set it very high
                                my_bid = mid + 0.01 # Aggressive bid
                                my_ask = mid + 0.50 # Defensive ask (unlikely to fill, holds position)
                                
                            elif momentum_state == "BEARISH":
                                # Strategy: Hold Shorts, Sell More
                                my_bid = mid - 0.50 # Defensive bid
                                my_ask = mid - 0.01 # Aggressive ask

                            # Format strings
                            b_str = f"{my_bid:.2f}"
                            s_str = f"{my_ask:.2f}"

                            # 4. EXECUTE ORDERS
                            # Only Buy if we aren't max long
                            if pos < MAX_POSITION:
                                # If Bearish momentum, we might skip buying entirely, but setting low price handles it
                                if momentum_state != "BEARISH": 
                                    asyncio.create_task(session_post(f"{base_order}BUY&price={b_str}", headers=headers))

                            # Only Sell if we aren't max short
                            if pos > -MAX_POSITION:
                                # If Bullish momentum, we might skip selling
                                if momentum_state != "BULLISH":
                                    asyncio.create_task(session_post(f"{base_order}SELL&price={s_str}", headers=headers))

                # 5. SPEED CONTROL
                if SLEEP_TIME > 0:
                    await asyncio.sleep(SLEEP_TIME)
                else:
                    await asyncio.sleep(0)

        except asyncio.CancelledError:
            pass
        except Exception:
            # Keep loop alive despite errors
            await asyncio.sleep(0.1)

    async def run(self):
        connector = aiohttp.TCPConnector(limit=0, family=2, ttl_dns_cache=300)
        async with aiohttp.ClientSession(connector=connector) as session:
            self.session = session
            if not await self.check_connection():
                return
            
            tasks = [self.ticker_strategy(sym) for sym in SYMBOLS]
            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                pass
            finally:
                print("\n🛑 STOPPING. Cancelling all...")
                for sym in SYMBOLS:
                    await self.session.post(f"{API_URL}/commands/cancel?ticker={sym}", headers=self.headers)

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    bot = RITC_Velocity_Bot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        pass