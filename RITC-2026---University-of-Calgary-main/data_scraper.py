import signal
import requests
import time
import pandas as pd

shutdown = False

def signal_handler(signum, frame):
    global shutdown
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    shutdown = True

class ApiException(Exception):
    pass

def get_case_status(session):
    resp = session.get("http://localhost:9999/v1/case")
    if not resp.ok:
        raise ApiException(f"get_case_status error: {resp.text}")
    return resp.json()['status']

def get_tick(session):
    resp = session.get("http://localhost:9999/v1/case")
    if not resp.ok:
        raise ApiException(f"get_tick error: {resp.text}")
    return resp.json()['tick']

def get_securities(session):
    resp = session.get("http://localhost:9999/v1/securities")
    if not resp.ok:
        raise ApiException(f"get_securities error: {resp.text}")
    return resp.json()

def get_order_book(session, ticker):
    url = f"http://localhost:9999/v1/securities/book?ticker={ticker}"
    resp = session.get(url)
    if not resp.ok:
        raise ApiException(f"get_order_book({ticker}) error: {resp.text}")
    return resp.json()

def get_latest_news(session):
    url = f"http://localhost:9999/v1/news"
    resp = session.get(url)
    if not resp.ok:
        raise ApiException(f"get_news error : {resp.text}")
    return pd.DataFrame(resp.json())

def collect_order_book_data(session, existing_columns):
    securities = get_securities(session)
    all_data = []
    new_columns = set(existing_columns)

    for sec in securities:
        ticker = sec['ticker']
        order_book = get_order_book(session, ticker)

        row = {
            'ticker': ticker,
            'last_price': sec.get('last', None),
            'bid_price': sec.get('bid', None),
            'ask_price': sec.get('ask', None)
        }

        for bid in order_book.get('bids', []):
            if bid['status'] == 'OPEN':
                trader = bid['trader_id']
                key = f"BUY_{trader}"
                row[key] = row.get(key, 0) + bid['quantity']
                new_columns.add(key)

        for ask in order_book.get('asks', []):
            if ask['status'] == 'OPEN':
                trader = ask['trader_id']
                key = f"SELL_{trader}"
                row[key] = row.get(key, 0) + ask['quantity']
                new_columns.add(key)

        all_data.append(row)

    return all_data, new_columns, order_book

def get_enemy_positions(row, old_book, current_book):
    # bid position
        old_bids = old_book.get('bids', [])
        old_bids = old_bids[['price','trader_id','quantity']]

        current_bids = current_book.get('bids',[])
        current_bids = current_bids[['price','trader_id','quantity']]

        current_bid = row['bid_price']
        if len(old_bids) > 0:
            old_bid = old_bids[0]
            old_bids = pd.DataFrame(olds_bids)
            volume_change = pd.DataFrame()

            if old_bid['price'] >= current_bid:
                bids_taken = old_bids['price' >= current_bid]
                bids_taken = bids_taken.groupby(['trader_id','price']).sum('quantity')
                volume_change = bids_taken

                if old_bid['price'] == current_bid:
                    bids_remain = current_bids['price' == current_bid]
                    bids_remain = bids_remain.groupby(['trader_id','price']).sum('quantity')

                    bid_compare = pd.merge(bids_remain, bids_taken, on='trader_id', suffixes=('remain_','taken_'))
                    bid_compare = bid_compare.fillna(0)
                    bid_compare['quantity'] = bid_compare['taken_quantity'] - bid_compare['remain_quantity']
                    bid_compare = bid_compare['quantity' > 0]
                    volume_change = pd.concat([volume_change, bid_compare], axis=0, ignore_index=True)
                    volume_change = volume_change.groupby(['trader_id','price']).sum('quantity')        
        bid_vol = volume_change

        current_ask = row['ask_price']
        if len(old_bids) > 0:
            old_bid = old_bids[0]
            old_bids = pd.DataFrame(olds_bids)
            volume_change = pd.DataFrame()

            if old_bid['price'] <= current_ask:
                bids_taken = old_bids['price' <= current_bid]
                bids_taken = bids_taken.groupby(['trader_id','price']).sum('quantity')
                volume_change = bids_taken

                if old_bid['price'] == current_bid:
                    bids_remain = current_bids['price' == current_bid]
                    bids_remain = bids_remain.groupby(['trader_id','price']).sum('quantity')

                    bid_compare = pd.merge(bids_remain, bids_taken, on='trader_id', suffixes=('remain_','taken_'))
                    bid_compare = bid_compare.fillna(0)
                    bid_compare['quantity'] = bid_compare['taken_quantity'] - bid_compare['remain_quantity']
                    bid_compare = bid_compare['quantity' > 0]
                    volume_change = pd.concat([volume_change, bid_compare], axis=0, ignore_index=True)
                    volume_change = volume_change.groupby(['trader_id','price']).sum('quantity')        
                    ask_vol['quantity'] = -ask_vol['quantity']
        ask_vol = volume_change

        volume_change = pd.concat([ask_vol, bid_vol], axis = 0, ignore_index=True)
        volume_change = volume_change.groupby(['trader_id']).sum('quantity')
        return volume_change

def save_to_csv(data, filename, tick, timestamp, columns, news):
    df = pd.DataFrame(data)

    # Ensure all columns (old + new) are included
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    df['tick'] = tick
    df['timestamp'] = timestamp
    ordered_columns = ['tick', 'timestamp'] + [col for col in df.columns if col not in ['tick', 'timestamp']]
    df = df[ordered_columns]
    if not news.empty:
        df = pd.concat([news.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

    header = not pd.io.common.file_exists(filename)
    df.to_csv(filename, mode='a', header=header, index=False)

def realtime_analysis():
    global shutdown
    signal.signal(signal.SIGINT, signal_handler)

    API_KEY = {'X-API-Key': '2'}
    session = requests.Session()
    session.headers.update(API_KEY)

    filename = None
    case_active = False
    columns = set(['ticker', 'last_price', 'bid_price', 'ask_price'])  # Initial known columns

    while not shutdown:
        try:
            status = get_case_status(session)

            if status == "ACTIVE":
                if not case_active:
                    case_active = True
                    timestamp_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"order_book_data_{timestamp_str}.csv"
                    print(f"New case detected. Starting new file: {filename}")

                tick = get_tick(session)
                timestamp = pd.Timestamp.now()

                # Collect data and dynamically update columns
                data, updated_columns, book = collect_order_book_data(session, columns)
                columns.update(updated_columns)

                latest_news = get_latest_news(session)

                try:
                    old_book = current_book
                    current_book = book
                    enemy_position = get_enemy_positions(data, old_book, current_book)
                    print(enemy_position)
                except:
                    current_book = book                    

                save_to_csv(data, filename, tick, timestamp, columns, latest_news)

                print(f"Tick {tick}: Data collected and saved to {filename}.")
                time.sleep(1)
            else:
                if case_active:
                    print("Case has gone INACTIVE. Waiting for next active session...")
                    case_active = False
                    del old_book, current_book
                time.sleep(3)
        except ApiException as e:
            print(f"[API Error] {e}")
            time.sleep(2)
        except Exception as e:
            print(f"[Error] {e}")
            time.sleep(2)

    print("Shutting down real-time analysis...")

if __name__ == "__main__":
    realtime_analysis()