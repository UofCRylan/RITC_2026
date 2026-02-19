from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import inf
from multiprocessing import Process
from requests import HTTPError
from time import sleep
from typing import Any
from warnings import warn

import numpy as np
from RIT import Case, Order, RIT, Security
# from RITDMA import Case, Order, RIT, Security
from sklearn.linear_model import LinearRegression

X_API_KEY: str = 'UUHHS8AS'
API_ENDPOINT: str = "http://flserver.rotman.utoronto.ca:16585"
AUTHORIZATION: str = "Basic Q0FMRy0xOmdyb3d0aA=="


@dataclass
class Algorithm:
    rit: RIT

    def call_safely(
            self,
            function: Callable[..., Any],
            *args: Any,
            **kwargs: Any,
    ) -> Any:
        try:
            return function(*args, **kwargs)
        except HTTPError as error:
            warn(f'error response received \'{error.response.json()}\'')

        return None

    def call_silently(
            self,
            function: Callable[..., Any],
            *args: Any,
            **kwargs: Any,
    ) -> Any:
        try:
            return function(*args, **kwargs)
        except HTTPError:
            pass

        return None

    def close(self, ticker: str) -> None:

            case = self.rit.get_case()
            if case.status == Case.Status.ACTIVE:
                security = self.rit.get_securities(ticker=ticker)[0]
                bid_quantity = min(security.max_trade_size, -security.position)
                ask_quantity = min(security.max_trade_size, security.position)
                self.call_safely(self.rit.post_commands_cancel, ticker=ticker)

                if bid_quantity > 0:
                    self.call_safely(
                        self.rit.post_orders,
                        ticker=ticker,
                        type=Order.Type.MARKET,
                        quantity=bid_quantity,
                        action=Order.Action.BUY,
                    )

                if ask_quantity > 0:
                    self.call_safely(
                        self.rit.post_orders,
                        ticker=ticker,
                        type=Order.Type.MARKET,
                        quantity=ask_quantity,
                        action=Order.Action.SELL,
                    )

                



    def make_market(self, ticker: str) -> None:
        def get_spread() -> tuple[float, float, float]:
            bid_price = book.bids[0].price
            ask_price = book.asks[0].price
            competing_bid_prices = set()
            competing_ask_prices = set()
            quantity = 5000

            for order in book.bids:
                if order.trader_id == trader.trader_id:
                    continue

                competing_bid_prices.add(order.price)
                quantity -= order.quantity - order.quantity_filled

                if quantity <= 0:
                    bid_price = order.price
                    break

            quantity = 5000

            for order in book.asks:
                if order.trader_id == trader.trader_id:
                    continue

                competing_ask_prices.add(order.price)
                quantity -= order.quantity - order.quantity_filled

                if quantity <= 0:
                    ask_price = order.price
                    break

            while bid_price in competing_bid_prices:
                bid_price += 0.01

            while ask_price in competing_ask_prices:
                ask_price -= 0.01

            spread = ask_price - bid_price

            return spread, round(bid_price, 2), round(ask_price, 2)

        while True:
            case = self.rit.get_case()
            trader = self.rit.get_trader()

            if case.status == Case.Status.ACTIVE:
            
                if case.tick % 60 == 58 or case.tick % 60 == 59:
                    self.call_safely(self.close, ticker)
                    sleep(2)
                else:    
                    security = self.rit.get_securities(ticker=ticker)[0]
                    bid_quantity = 5000 - security.position
                    ask_quantity = 5000 + security.position
                    book = self.rit.get_securities_book(ticker=ticker)

                    if not book.bids or not book.asks:
                        continue

                    spread, bid_price, ask_price = get_spread()
                    quantity = bid_quantity

                    for order in book.bids:
                        if order.trader_id != trader.trader_id \
                                or order.price != bid_price:
                            continue

                        quantity -= order.quantity - order.quantity_filled

                    if quantity > 0 and spread > security.trading_fee:
                        self.call_safely(
                            self.rit.post_orders,
                            ticker=ticker,
                            type=Order.Type.LIMIT,
                            quantity=min(
                                5000,
                                max(security.min_trade_size, quantity),
                            ),
                            action=Order.Action.BUY,
                            price=bid_price,
                        )

                    quantity = ask_quantity

                    for order in book.asks:
                        if order.trader_id != trader.trader_id \
                                or order.price != ask_price:
                            continue

                        quantity -= order.quantity - order.quantity_filled

                    if quantity > 0 and spread > security.trading_fee:
                        self.call_safely(
                            self.rit.post_orders,
                            ticker=ticker,
                            type=Order.Type.LIMIT,
                            quantity=min(
                                5000,
                                max(security.min_trade_size, quantity),
                            ),
                            action=Order.Action.SELL,
                            price=ask_price,
                        )

                    if bid_quantity <= 0:
                        self.call_safely(
                            self.rit.post_commands_cancel,
                            query=f'Ticker=\'{ticker}\' AND Volume>0',
                        )
                    else:
                        self.call_safely(
                            self.rit.post_commands_cancel,
                            query=f'Ticker=\'{ticker}\' AND Volume>0 '
                                f'AND Price<>{bid_price}',
                        )

                    if ask_quantity <= 0:
                        self.call_safely(
                            self.rit.post_commands_cancel,
                            query=f'Ticker=\'{ticker}\' AND Volume<0',
                        )
                    else:
                        self.call_safely(
                            self.rit.post_commands_cancel,
                            query=f'Ticker=\'{ticker}\' AND Volume<0 '
                                f'AND Price<>{ask_price}',
                        )

    def run(self) -> None:
        processes = []

        for security in self.rit.get_securities():
      
                if security.type == Security.Type.STOCK:
                    processes.append(
                        Process(
                            target=self.make_market,
                            args=(security.ticker,),
                        ),
                    
                    )

         



        for process in processes:
            process.start()

        for process in processes:
            process.join()


def main() -> None:
    rit = RIT(X_API_KEY)
    algorithm = Algorithm(rit)

    algorithm.run()


if __name__ == '__main__':
    main()
