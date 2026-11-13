"""Day 13 rebuild: a tiny order book with price-time priority.

Run:  python3 day13_tiny_book.py
It replays the ten-command session of Part 7.6 and checks the book after each command.

Everything is written except the match loop inside submit(): look for TODO(human).
The C++ it mirrors is src/order_book.cpp:126-166 (Part 10.5 of the Day 13 note).
"""
from collections import deque

LIMIT, IOC, MARKET = 'limit', 'ioc', 'market'
BUY, SELL = 'buy', 'sell'
MAX_PRICE = 65536


class Order:
    def __init__(self, oid, side, price, qty):
        self.id, self.side, self.price, self.qty = oid, side, price, qty


class Book:
    def __init__(self):
        self.bids = {}    # price -> deque of Orders, oldest on the left (C++: bids_[price], head = oldest)
        self.asks = {}    # price -> deque of Orders, oldest on the left
        self.index = {}   # order id -> Order                            (C++: IdTable)
        self.trades = []  # (buy id, sell id, price, qty)
        self.filled = self.resting = self.cancelled = self.expired = 0

    def best_ask(self):  # lowest price with sellers, or None
        return min(self.asks) if self.asks else None

    def best_bid(self):  # highest price with buyers, or None
        return max(self.bids) if self.bids else None

    def erase(self, o):
        """Take a resting order out of its level and out of the index (C++: erase_order)."""
        side = self.bids if o.side == BUY else self.asks
        side[o.price].remove(o)
        if not side[o.price]:
            del side[o.price]        # an empty level disappears, so best_ask()/best_bid() stay right
        del self.index[o.id]

    def rest(self, oid, side, price, qty):
        """Append the leftover at the TAIL of its price level (C++: rest)."""
        o = Order(oid, side, price, qty)
        (self.bids if side == BUY else self.asks).setdefault(price, deque()).append(o)
        self.index[oid] = o
        self.resting += qty

    def cancel(self, oid):
        o = self.index.get(oid)
        if o is None:
            return False
        self.resting -= o.qty
        self.cancelled += o.qty
        self.erase(o)
        return True

    def submit(self, oid, side, price, qty, kind=LIMIT):
        """Returns how much was filled."""
        if kind == MARKET:
            price = MAX_PRICE - 1 if side == BUY else 1
        if kind == LIMIT and oid in self.index:   # duplicate id: rejected
            self.expired += qty
            return 0
        filled = 0

        # TODO(human): the match loop.
        # While the incoming order still wants shares (qty > 0) and the best opposite
        # price is good enough, trade with the OLDEST order at that best price.
        # For each trade of x shares:
        #   - lower qty and the maker's qty by x; raise `filled` by x
        #   - self.resting -= x ; self.filled += 2 * x
        #   - append (buy id, sell id, maker's price, x) to self.trades
        #   - if the maker has nothing left, self.erase(maker)
        # Handle both sides: a BUY looks at self.asks / best_ask(), a SELL at self.bids / best_bid().
        raise NotImplementedError('write the match loop')

        if qty > 0:
            if kind == LIMIT:
                self.rest(oid, side, price, qty)
            else:
                self.expired += qty
        return filled


def view(side):
    return {p: [(o.id, o.qty) for o in q] for p, q in sorted(side.items())}


if __name__ == '__main__':
    b = Book()
    submitted = 0
    # (command, asks after, bids after, counters after = filled, resting, cancelled, expired)
    session = [
        (('new', 1, SELL, 101, 40, LIMIT), {101: [(1, 40)]}, {}, (0, 40, 0, 0)),
        (('new', 2, SELL, 101, 20, LIMIT), {101: [(1, 40), (2, 20)]}, {}, (0, 60, 0, 0)),
        (('new', 3, SELL, 103, 50, LIMIT), {101: [(1, 40), (2, 20)], 103: [(3, 50)]}, {}, (0, 110, 0, 0)),
        (('new', 4, BUY, 99, 30, LIMIT), {101: [(1, 40), (2, 20)], 103: [(3, 50)]}, {99: [(4, 30)]}, (0, 140, 0, 0)),
        (('new', 1, SELL, 105, 5, LIMIT), {101: [(1, 40), (2, 20)], 103: [(3, 50)]}, {99: [(4, 30)]}, (0, 140, 0, 5)),
        (('new', 9, BUY, 102, 70, LIMIT), {103: [(3, 50)]}, {99: [(4, 30)], 102: [(9, 10)]}, (120, 90, 0, 5)),
        (('cancel', 9), {103: [(3, 50)]}, {99: [(4, 30)]}, (120, 80, 10, 5)),
        (('new', 10, BUY, 103, 80, IOC), {}, {99: [(4, 30)]}, (220, 30, 10, 35)),
        (('new', 11, SELL, 0, 50, MARKET), {}, {}, (280, 0, 10, 55)),
        (('cancel', 2), {}, {}, (280, 0, 10, 55)),
    ]
    for n, (cmd, asks, bids, counters) in enumerate(session, 1):
        if cmd[0] == 'new':
            _, oid, side, price, qty, kind = cmd
            submitted += qty
            b.submit(oid, side, price, qty, kind)
        else:
            b.cancel(cmd[1])
        got = (b.filled, b.resting, b.cancelled, b.expired)
        assert view(b.asks) == asks, f'command {n}: asks are {view(b.asks)}, expected {asks}'
        assert view(b.bids) == bids, f'command {n}: bids are {view(b.bids)}, expected {bids}'
        assert got == counters, f'command {n}: counters are {got}, expected {counters}'
        assert sum(got) == submitted, f'command {n}: conservation broken'
        print(f'command {n:>2} ok   asks {view(b.asks)}   bids {view(b.bids)}')
    assert b.trades[:2] == [(9, 1, 101, 40), (9, 2, 101, 20)], b.trades
    print('all ten commands match Part 7.6; trades:', b.trades)
