import signal
import sys
import logging
import requests
from time import sleep

# --- CONFIGURATION ---
API_HOST = 'http://localhost'
API_PORT = '9999'  
BASE_URL = f"{API_HOST}:{API_PORT}/v1"
API_KEY = {'X-API-Key': 'ZXH9VEYS'} 

# The 4 Companies for the new case
STOCKS = ['SPNG', 'SMMR', 'ATMN', 'WNTR']

# Strategy Settings
MAX_VOLUME = 1000        
MIN_SPREAD = 0.05      
POSITION_LIMIT = 5000    

SHUTDOWN = False

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')

def signal_handler(signum, frame):
    """ Graceful Shutdown on Ctrl+C """
    global SHUTDOWN
    print("\n[STOP] Shutdown signal received. Closing out...")
    SHUTDOWN = True

def get_session():
    """ Creates a persistent session """
    s = requests.Session()
    s.headers.update(API_KEY)
    return s

# --- API WRAPPERS ---

def get_tick(session):
    try:
        resp = session.get(f'{BASE_URL}/case')
        if resp.ok:
            return resp.json()['tick']
    except Exception as e:
        logging.error(f"Tick Error: {e}")
    return 0

def get_total_position(session, ticker):
    try:
        resp = session.get(f'{BASE_URL}/securities', params={'ticker': ticker})
        if resp.ok:
            return resp.json()[0]['position']
    except Exception:
        pass
    return 0

def get_limit_price(session, ticker):
    try:
        resp = session.get(f'{BASE_URL}/securities/book', params={'ticker': ticker, 'limit': 1})
        if resp.ok:
            book = resp.json()
            if book['bids'] and book['asks']:
                return book['bids'][0]['price'], book['asks'][0]['price']
    except Exception:
        pass
    return None, None

def send_order(session, ticker, quantity, action, type="MARKET", price=None):
    params = {
        'ticker': ticker,
        'type': type,
        'quantity': int(quantity),
        'action': action
    }
    if price:
        params['price'] = price
    
    try:
        session.post(f'{BASE_URL}/orders', params=params)
    except Exception as e:
        logging.error(f"Order Error: {e}")

def cancel_order(session, order_id):
    try:
        session.delete(f'{BASE_URL}/orders/{order_id}')
        return True
    except:
        return False

def cancel_all(session, query=None):
    params = {'all': 1}
    if query:
        params['query'] = query
        params['all'] = 0
    try:
        session.post(f'{BASE_URL}/commands/cancel', params=params)
    except:
        pass

# --- STRATEGY LOGIC ---

def buy_sell_new(session, ticker):
    bid, ask = get_limit_price(session, ticker)
    if bid is None or ask is None:
        return

    spread = ask - bid

    if spread >= MIN_SPREAD:
        my_bid = round(bid + 0.01, 2)
        my_ask = round(ask - 0.01, 2)
        
        send_order(session, ticker, MAX_VOLUME, "BUY", "LIMIT", my_bid)
        send_order(session, ticker, MAX_VOLUME, "SELL", "LIMIT", my_ask)

def re_order(session, tick):
    for ticker in STOCKS:
        try:
            resp = session.get(f'{BASE_URL}/orders', params={'ticker': ticker})
            orders = resp.json()
            
            for order in orders:
                if order['tick'] < tick - 3:
                    cancel_order(session, order['order_id'])
                    
                    bid, ask = get_limit_price(session, ticker)
                    if bid and ask:
                        mid_price = round((bid + ask) / 2, 2)
                        send_order(session, ticker, order['quantity'], order['action'], "LIMIT", mid_price)
        except Exception as e:
            logging.error(f"Re-order error: {e}")

def close_open(session):
    for ticker in STOCKS:
        position = get_total_position(session, ticker)
        
        if abs(position) > 1000:
            direction = "BUY" if position < 0 else "SELL"
            qty = abs(position)
            
            mkt_qty = int(qty / 2)
            lmt_qty = qty - mkt_qty
            
            logging.info(f"Reducing inventory on {ticker}: {position}")
            
            send_order(session, ticker, mkt_qty, direction, "MARKET")
            
            bid, ask = get_limit_price(session, ticker)
            if bid and ask:
                mid_price = round((bid + ask) / 2, 2)
                send_order(session, ticker, lmt_qty, direction, "LIMIT", mid_price)

def close_end(session):
    logging.info("Final Closeout...")
    for ticker in STOCKS:
        position = get_total_position(session, ticker)
        if position != 0:
            direction = "BUY" if position < 0 else "SELL"
            send_order(session, ticker, abs(position), direction, "MARKET")

# --- MAIN LOOP ---

def main():
    s = get_session()
    
    # FIXED: Replaced 'signal(SIGINT...)' with 'signal.signal(signal.SIGINT...)'
    signal.signal(signal.SIGINT, signal_handler)

    logging.info("Algorithm Started. Waiting for first tick...")
    
    while not SHUTDOWN:
        tick = get_tick(s)
        
        if tick is None:
            sleep(.25)
            continue

        if 0 < tick < 300:
            for stock in STOCKS:
                buy_sell_new(s, stock)
            
            if tick % 5 == 0:
                re_order(s, tick)
                cancel_all(s, query=f'tick<{tick-5}')

            if tick % 15 == 0:
                cancel_all(s)
                sleep(0.1) 
                close_open(s)

            sleep(0.25)
        
        elif tick > 300:
            logging.info("Case Ended.")
            break
        else:
            sleep(.25)

    cancel_all(s)
    close_end(s)

if __name__ == '__main__':
    main()