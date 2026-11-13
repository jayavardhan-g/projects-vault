"""Concept animations for the Micro Exchange part of Day 13 (the 'from zero' primers). Run: python3 mxfigs3.py <output dir>"""
from mxfigs2 import *

def chk(fr):
    for cap, _ in fr: assert len(cap) <= 132, (len(cap), cap)
    return fr

def hb(x, y, w, h, label, fill, hot=False, size=13, sub=None):
    return B(x, y, w, h, label, fill, RED if hot else MUTE, 2.4 if hot else 1.2, size, sub=sub)

# ───────── order types ─────────
def types():
    W, H = 960, 400
    def f(a105, a103, b103, cmd, out, hot=()):
        b = T(480, 84, 'price', 12, 700, MUTE, 'middle') + T(250, 84, 'buyers (bids)', 12, 700, BT, 'middle') + T(710, 84, 'sellers (asks)', 12, 700, RED, 'middle')
        for k, (p, bid, ask) in enumerate(((105, '', a105), (103, b103, a103))):
            y = 96 + k * 62
            b += hb(430, y, 100, 50, str(p), GREY, size=15)
            b += hb(560, y, 300, 50, ask or '—', '#ffd8d3' if ask else WHITE, ('a', p) in hot)
            b += hb(100, y, 300, 50, bid or '—', BLUE if bid else WHITE, ('b', p) in hot)
        b += hb(100, 240, 400, 46, cmd, YEL, size=14) + T(520, 268, out, 13, 700, INK)
        return b
    fr = chk([
        ('The book: sellers offer 50 shares at 103 and 40 shares at 105. A buyer wants 80 shares.', f('40 waiting', '50 waiting', '', 'incoming: buy 80', '')),
        ('LIMIT buy 80 @ 103: fills 50 at 103. 105 is above the limit. The other 30 REST in the book and wait.', f('40 waiting', '', 'new order: 30 resting', 'Limit  buy 80 @ 103', 'filled 50, resting 30', (('b', 103),))),
        ('IOC buy 80 @ 103: fills 50 at 103, same as before. But the other 30 EXPIRE at once. Nothing rests.', f('40 waiting', '', '', 'IOC  buy 80 @ 103', 'filled 50, expired 30', (('a', 103),))),
        ('MARKET buy 80: no price limit. Fills 50 at 103, then 30 more at 105. It never rests either.', f('10 waiting', '', '', 'Market  buy 80', 'filled 80 (50 @ 103 + 30 @ 105)', (('a', 105), ('a', 103)))),
    ])
    save('d13y-types.svg', W, H, 'Limit, IOC and Market: the same buyer, three order types', 'the difference is only what happens to the part that cannot trade now', '', fr, 5.0)

# ───────── race on a shared order ─────────
def race():
    W, H = 960, 400
    def f(qty, a, b, note='', hot='', bad=False, fix=False):
        o = hb(310, 76, 340, 64, f'S1.qty = {qty}', '#ffd8d3', hot == 'q', 17, sub='one resting sell order (shared memory)')
        if fix:
            o += hb(40, 190, 150, 50, 'buy 30', YEL) + hb(200, 190, 150, 50, 'buy 20', YEL) + A(354, 215, 416, 215)
            o += hb(420, 180, 260, 70, 'the ONE matching thread', GREEN, True, 13, sub='runs one command at a time')
            o += A(550, 178, 500, 144, RED) + T(40, 176, 'commands wait in a queue', 12, 700, MUTE)
            o += T(700, 206, '40 → 10 → 0', 16, 700, GT) + T(700, 228, 'second buyer gets the 10 left', 12, 600, GT)
        else:
            o += T(40, 176, 'thread A: a buyer wants 30', 12, 700, BT) + T(560, 176, 'thread B: another buyer wants 20', 12, 700, GT)
            o += hb(40, 186, 360, 62, a[0], BLUE, hot == 'a', 14, sub=a[1]) + hb(560, 186, 360, 62, b[0], GREEN, hot == 'b', 14, sub=b[1])
        o += T(40, 300, note, 13, 700, RED if bad else INK)
        return o
    n = ('has read nothing yet', '')
    fr = chk([
        ('Imagine TWO matching threads. Both want to trade against the same resting order S1, which has 40 shares.', f(40, n, n)),
        ('Thread A reads S1.qty and gets 40. It keeps that number in its own register.', f(40, ('my copy: 40', 'x = S1.qty'), n, hot='a')),
        ('Before A writes anything back, thread B reads S1.qty too. It also gets 40.', f(40, ('my copy: 40', ''), ('my copy: 40', 'x = S1.qty'), hot='b')),
        ('A fills its 30 and writes back 40 − 30 = 10.', f(10, ('writes 10', 'S1.qty = 40 − 30'), ('my copy: 40 (now out of date)', ''), hot='q')),
        ('B fills its 20 and writes back 40 − 20 = 20. A\'s write is gone.', f(20, ('thinks 10 are left', ''), ('writes 20', 'S1.qty = 40 − 20'), '50 shares were sold from an order of 40, and the book says 20 remain ✗', 'q', True)),
        ('The design here: only ONE thread ever touches the book. Other threads hand it commands through a queue.', f('40 → 10 → 0', n, n, 'no lock is needed on the book, and the order of commands is always the same ✓', fix=True)),
    ])
    save('d13y-race.svg', W, H, 'Why only one thread may change the book', 'a "lost update": two threads read, change and write the same number', '', fr, 5.0)

# ───────── sleeping vs spinning ─────────
def wake():
    W, H = 960, 430
    def lane(y, name, p, q, c, chot=False, col=BLUE):
        o = T(40, y - 8, name, 12, 700, MUTE)
        o += hb(40, y, 250, 62, p[0], YEL, False, 12, sub=p[1]) + A(294, y + 31, 346, y + 31)
        o += hb(350, y, 230, 62, q, GREY, False, 13) + A(584, y + 31, 636, y + 31)
        o += hb(640, y, 280, 62, c[0], col, chot, 12, sub=c[1])
        return o
    def f(t, b, note=''):
        return lane(96, 'naive engine: mutex + condition variable queue', *t) + lane(226, 'this engine: SPSC ring, the consumer spins', *b) + T(40, 334, note, 13, 700, INK)
    fr = chk([
        ('No commands yet. The naive consumer goes to SLEEP (the OS takes it off its core). The ring\'s consumer keeps looping.',
         f((('producer: idle', ''), 'queue: empty', ('consumer: asleep', 'waiting on a condition variable'), False, GREY),
           (('producer: idle', ''), 'ring: empty', ('consumer: running', 'pop() → false, pause, try again'), False, GREEN))),
        ('A command arrives on both. Naive: lock, push, unlock, notify. Ring: write the slot, store head_.',
         f((('producer: lock, push, notify', 'a system call wakes the sleeper'), 'queue: 1 command', ('consumer: still asleep', ''), False, GREY),
           (('producer: slot = c; head_ = 1', 'two plain memory writes'), 'ring: 1 command', ('consumer: running', 'its next pop() sees head_ = 1'), True, GREEN))),
        ('The OS must now wake the sleeper: mark it runnable, pick a core, switch to it. The spinner is already working.',
         f((('producer: done', ''), 'queue: 1 command', ('consumer: waking up…', 'scheduler + context switch'), True, YEL),
           (('producer: done', ''), 'ring: empty', ('consumer: applying the command', 'log, then match'), False, GREEN), 'a wake-up costs microseconds; one more trip round a spin loop costs nanoseconds')),
        ('Only now does the naive consumer run. The docs blame most of the naive engine\'s 2.2 ms p99 on this sleep and wake.',
         f((('producer: done', ''), 'queue: empty', ('consumer: applying the command', 'much later'), False, BLUE),
           (('producer: done', ''), 'ring: empty', ('consumer: spinning again', 'ready for the next one'), False, GREEN), 'the price of spinning: one core runs at 100% even when there is no work')),
    ])
    save('d13y-wake.svg', W, H, 'Sleeping on a lock vs spinning on a ring', 'the same handoff of one command, done two ways', '', fr, 5.5)

# ───────── pointers ─────────
def pointer():
    W, H = 960, 440
    def order(x, name, addr, qty, nxt, hot=()):
        o = R(x, 96, 250, 132, '#ffd8d3', RED if 'box' in hot else MUTE, 2.4 if 'box' in hot else 1.2, 6)
        o += T(x + 125, 118, f'{name}: an Order at address {addr}', 12, 700, INK, 'middle')
        for k, (fn, fv) in enumerate((('id', name[1:]), ('price', '101'), ('qty', qty), ('next', nxt))):
            h = fn in hot
            o += C(x + 18, 144 + k * 22, f'{fn:<6}= {fv}', 13, RED if h else INK, 700 if h else 400)
        return o
    def f(q1='40', maker=None, hot1=(), hot2=(), extra='', arrow=False, link=False):
        o = T(40, 84, 'memory: a long row of numbered bytes', 12, 700, MUTE)
        o += order(330, 'S1', 1000, q1, '1048', hot1) + order(660, 'S2', 1048, '20', 'nullptr', hot2)
        if link: o += A(560, 212, 656, 150, RED)
        if maker is not None:
            o += hb(30, 126, 270, 70, f'maker = {maker}', YEL, arrow, 15, sub='Order* : 8 bytes, holds an address')
            if arrow and maker != 'nullptr': o += A(302, 160, 326, 160, RED)
        return o + extra
    ref = (R(40, 250, 880, 104, '#f6f8fa', '#d0d7de', 1, 8)
           + C(56, 276, 'Level& lvl = asks_[101];   lvl.qty -= 40;', 12, GT, 700) + T(480, 276, '& = a second NAME for asks_[101]. The book changes ✓', 12, 600, GT)
           + C(56, 304, 'Level  lvl = asks_[101];   lvl.qty -= 40;', 12, RED, 700) + T(480, 304, 'no & = a COPY. Only the copy changes; the book does not ✗', 12, 600, RED)
           + T(56, 336, 'pointer: can be null, can be re-pointed, uses ->', 12, 600, MUTE) + T(480, 336, 'reference: never null, fixed for life, uses a dot', 12, 600, MUTE))
    fr = chk([
        ('Every object lives at an address. An Order is 48 bytes: S1 starts at byte 1000, S2 right after it at 1048.', f()),
        ('Order* maker = lvl.head;  A POINTER is a variable that holds an address. maker holds 1000. Nothing was copied.', f(maker='1000', arrow=True)),
        ('maker->qty means: go to the address in maker, then take the field qty. It reads 40 from S1.', f(maker='1000', arrow=True, hot1=('qty',))),
        ('maker->qty -= 40;  This changes S1 ITSELF, in place. Everyone who points at S1 now sees qty = 0.', f('0', '1000', ('qty', 'box'), arrow=True)),
        ('S1.next holds 1048, the address of S2. That number is the "link" of a linked list.', f('0', '1000', ('next',), ('box',), arrow=True, link=True)),
        ('nullptr (address 0) means "points at nothing". if (lvl.head) asks: is there any order here at all?', f('0', 'nullptr', (), ('next',), arrow=True)),
        ('A reference (the & in Level& lvl) is a second name for an object that already exists. No copy is made.', f('0', extra=ref)),
    ])
    save('d13y-pointer.svg', W, H, 'Pointers and references, with real addresses', 'the three symbols the order book code is full of:  *  ->  &', '', fr, 5.5)

# ───────── tree vs array ─────────
def levels():
    W, H = 960, 440
    pos = {100: (480, 96), 98: (280, 176), 102: (680, 176), 97: (180, 256), 99: (380, 256), 101: (580, 256), 103: (780, 256)}
    edges = ((100, 98), (100, 102), (98, 97), (98, 99), (102, 101), (102, 103))
    def tree(hot=(), path=()):
        o = ''
        for a, b in edges:
            (x1, y1), (x2, y2) = pos[a], pos[b]
            o += L(x1, y1 + 44, x2, y2, RED if (a, b) in path else MUTE, 2.6 if (a, b) in path else 1.2)
        for p, (x, y) in pos.items():
            o += hb(x - 55, y, 110, 44, f'price {p}', YEL if p in hot else GREY, p in hot, 13)
        return o
    def arr(hot=None, note=''):
        o = T(60, 110, 'asks_ : one array, the index is the price', 12, 700, MUTE)
        for k, p in enumerate(range(97, 104)):
            x = 60 + k * 120
            o += hb(x, 124, 112, 60, f'asks_[{p}]', YEL if p == hot else GREY, p == hot, 13, sub='24 bytes')
        o += C(60, 230, 'address of asks_[101]  =  start of the array + 101 × 24 bytes', 13, INK, 700)
        o += T(60, 270, note, 13, 700, INK)
        return o
    q = lambda s: T(40, 340, s, 13, 700, INK)
    fr = chk([
        ('The usual way, std::map: prices in a balanced tree. Every node is its own small block somewhere on the heap.', tree()),
        ('Find the level for price 101. Start at the root: 101 > 100, so go right.', tree((100,)) + q('hop 1')),
        ('101 < 102, so go left.', tree((100, 102), ((100, 102),)) + q('hop 2: follow a pointer to another block of memory (likely a cache miss)')),
        ('Found, after 3 hops. With n price levels that is about log2(n) hops: O(log n).', tree((100, 102, 101), ((100, 102), (102, 101))) + q('hop 3; and a new price means allocating a new node')),
        ('This engine: a flat array. The price IS the index, so the address is one multiplication and one addition.', arr(101, 'one step, however many prices there are: O(1)')),
        ('The cost: 65,536 levels × 24 bytes ≈ 1.5 MB per side, mostly empty. Fine when prices are bounded integers.', arr(None, 'two sides → about 3 MB per symbol, whatever is in the book')),
    ])
    save('d13y-levels.svg', W, H, 'Finding a price level: a tree vs a flat array', 'std::map (the naive engine) against std::vector<Level> indexed by price (this engine)', '', fr, 5.0)

# ───────── rest() debugger ─────────
def rest_dbg():
    code = [(90, 'bool OrderBook::rest(id, side, price, qty) {'),
            (91, '  Order* o = pool_.acquire();'),
            (92, '  if (!o) return false;'),
            (93, '  *o = Order{id, price, qty, side, nullptr, nullptr};'),
            (95, '  Level& lvl = (side == Buy ? bids_ : asks_)[price];'),
            (96, '  o->prev = lvl.tail;'),
            (97, '  if (lvl.tail) lvl.tail->next = o; else lvl.head = o;'),
            (98, '  lvl.tail = o;'),
            (100, '  lvl.qty += qty;'),
            (101, '  resting_qty_ += qty;'),
            (102, '  index_.insert(id, o);'),
            (107, '  if (price > best_bid_) best_bid_ = price;'),
            (111, '  return true;'),
            (112, '}')]
    V = ['id, price, qty', 'o', 'node N', 'lvl', 'lvl.head', 'lvl.tail', 'lvl.qty', 'id table', 'best_bid_']
    steps = [
        ((90,), {'id, price, qty': '9, 102, 10', 'o': '–', 'node N': 'free, old junk inside', 'lvl': '–', 'lvl.head': '–', 'lvl.tail': '–', 'lvl.qty': '–', 'id table': '4 → B1', 'best_bid_': '99'},
         'rest(9, Buy, 102, 10): the buy matched 60 of its 70. The 10 left must wait in the book.'),
        ((91,), {'o': '→ N'}, 'Line 91: take a ready-made node from the pool. o is a pointer to it. No call to new.'),
        ((92,), {}, 'Line 92: a null o would mean the pool is empty. It is not, so go on.'),
        ((93,), {'node N': '{9, 102, 10, Buy, null, null}'}, 'Line 93: *o = "the node o points at". Overwrite ALL its fields; it still held its last order\'s data.'),
        ((95,), {'lvl': 'bids_[102]', 'lvl.head': 'null', 'lvl.tail': 'null', 'lvl.qty': '0'}, 'Line 95: lvl is a second name for the level at price 102 on the buy side. It is empty.'),
        ((96,), {}, 'Line 96: my "older neighbour" is the current tail. The level is empty, so o->prev = null.'),
        ((97,), {'lvl.head': 'N'}, 'Line 97: no tail exists, so the else branch runs: I am also the head (the oldest).'),
        ((98,), {'lvl.tail': 'N'}, 'Line 98: I am the newest, so I am the tail. Appending at the tail IS time priority.'),
        ((100, 101), {'lvl.qty': '10'}, 'Lines 100–101: add 10 to the level\'s total and to the book-wide resting counter.'),
        ((102,), {'id table': '4 → B1, 9 → N'}, 'Line 102: remember id 9 → this node, so a later cancel(9) finds it in one step.'),
        ((107,), {'best_bid_': '102'}, 'Line 107: 102 is higher than the best bid so far (99). Update the cached best bid.'),
        ((111,), {}, 'Line 111: true = the order is resting. (false on line 92 would have counted the 10 as expired.)'),
        ((96, 97, 98), {'id, price, qty': '12, 102, 5', 'o': '→ M', 'lvl.tail': 'M', 'lvl.qty': '15', 'node N': '{9, …, next → M}'}, 'If order 12 came to 102 later: M.prev = N, N.next = M (the if branch), tail = M. N stays the head.'),
    ]
    dbg('d13y-rest.svg', 'rest(), one line at a time: the leftover 10 joins the book', 'order_book.cpp:90–112 (blank and hi/lo lines left out); the buy side is shown', code, steps, V, sec=4.6)

# ───────── cancel() debugger ─────────
def cancel_dbg():
    code = [(169, 'bool OrderBook::cancel(uint64_t id) {'),
            (170, '  Order* o = index_.find(id);'),
            (171, '  if (!o) return false;'),
            (173, '  const Side side = o->side;'),
            (174, '  const int64_t price = o->price;'),
            (175, '  Level& lvl = (side == Buy ? bids_ : asks_)[price];'),
            (176, '  lvl.qty -= o->qty;'),
            (177, '  resting_qty_ -= o->qty;'),
            (178, '  cancelled_qty_ += o->qty;'),
            (179, '  erase_order(o);'),
            (181, '  if (lvl.head == nullptr) {'),
            (182, '    if (side == Buy && price == best_bid_) lower_best_bid();'),
            (183, '    if (side == Sell && price == best_ask_) raise_best_ask();'),
            (184, '  }'),
            (185, '  return true;'),
            (186, '}')]
    V = ['id', 'o', 'side, price (my copies)', 'lvl.head', 'lvl.qty', 'cancelled_qty_', 'id table', 'node N', 'best_bid_']
    steps = [
        ((169,), {'id': '9', 'o': '–', 'side, price (my copies)': '–', 'lvl.head': '–', 'lvl.qty': '–', 'cancelled_qty_': '0', 'id table': '4 → B1, 9 → N', 'node N': '{9, 102, 10, Buy}', 'best_bid_': '102'},
         'cancel(9): the user takes back the 10 shares resting at 102. All we are given is the id.'),
        ((170,), {'o': '→ N'}, 'Line 170: ask the id table: where is order 9? It answers with the node\'s address. No list is walked.'),
        ((171,), {}, 'Line 171: null would mean "not resting" (filled, cancelled already, or never existed). Not our case.'),
        ((173, 174), {'side, price (my copies)': 'Buy, 102'}, 'Lines 173–174: COPY side and price out of the node now. After line 179 the node is no longer ours.'),
        ((175,), {'lvl.head': 'N', 'lvl.qty': '10'}, 'Line 175: the level this order sits in: buy side, price 102.'),
        ((176, 177, 178), {'lvl.qty': '0', 'cancelled_qty_': '10'}, 'Lines 176–178: move its 10 shares from "resting" to "cancelled". The totals still add up.'),
        ((179,), {'lvl.head': 'null', 'id table': '4 → B1', 'node N': 'back in the pool'}, 'Line 179: the same unlink a fill uses: fix two pointers, forget the id, give the node back.'),
        ((181,), {}, 'Line 181: was that the last order at this price? head is null, so yes.'),
        ((182,), {'best_bid_': '99'}, 'Line 182: it was the BEST bid, so scan down for the next price with buyers: 101, 100, 99 ✓'),
        ((185,), {}, 'Line 185: true = something was cancelled. A second cancel(9) would now stop at line 171.'),
    ]
    dbg('d13y-cancel.svg', 'cancel(), one line at a time', 'order_book.cpp:169–186; right after the worked example, order 9 rests alone at bid 102', code, steps, V, sec=4.6)

# ───────── naive list walk ─────────
def naive():
    W, H = 960, 400
    def f(seen=0, found=False, ours=False, note=''):
        o = T(40, 96, 'price level 101: five resting orders, oldest on the left', 12, 700, MUTE)
        for k in range(5):
            x = 40 + k * 180
            hot = (k < seen) or (found and k == 3)
            fill = PINK if (k < seen and not (found and k == 3)) else (GREEN if (found and k == 3) else '#ffd8d3')
            o += hb(x, 108, 150, 60, f'order id {k + 1}', fill, hot, 14, sub=('prev / next inside' if ours else 'a std::list node'))
            if k < 4: o += A(x + 152, 138, x + 178, 138)
        o += hb(40, 214, 330, 56, 'cancel id 4', YEL, False, 14)
        o += T(400, 248, note, 13, 700, INK)
        return o
    fr = chk([
        ('The naive engine keeps each price level as a std::list of orders. A "cancel id 4" arrives.', f()),
        ('Its id map finds the order OBJECT in one step. But the map does not know where that order sits in the list.', f(note='id → the order: O(1). Its place in the list: unknown.')),
        ('So erase_from walks the list from the front, comparing ids: 1? no. 2? no. 3? no.', f(3, note='3 comparisons so far, each one a pointer hop')),
        ('Found at the 4th node. With 10,000 orders at one price this walk can take 10,000 steps: O(n).', f(3, True, note='this walk is the real "O(n) cancellation"')),
        ('This engine: the id table returns the Order*, and the Order itself holds prev and next. No walk at all.', f(0, True, True, 'two pointer writes: 3.next = 5 and 5.prev = 3. O(1).')),
    ])
    save('d13y-naive.svg', W, H, 'Cancel in the naive engine vs this one', 'both find the order by id in O(1); the difference is unlinking it from its price level', '', fr, 5.0)

# ───────── mmap and the page cache ─────────
def pagecache():
    W, H = 960, 440
    def f(proc, page, disk, note='', hot='', pdead=False, kdead=False, good=None):
        o = R(30, 78, 290, 190, '#f6f8fa' if not pdead else GREY, '#d0d7de', 1, 8, '6 4' if pdead else None) + T(46, 100, 'the engine process' + ('  (killed)' if pdead else ''), 12, 700, MUTE)
        o += R(335, 78, 290, 190, '#f6f8fa' if not kdead else GREY, '#d0d7de', 1, 8, '6 4' if kdead else None) + T(351, 100, 'the kernel: page cache (RAM)' + ('  (lost)' if kdead else ''), 12, 700, MUTE)
        o += R(640, 78, 290, 190, '#f6f8fa', '#d0d7de', 1, 8) + T(656, 100, 'the disk', 12, 700, MUTE)
        if not pdead: o += hb(50, 120, 250, 60, proc[0], YEL, hot == 'p', 12, sub=proc[1]) + hb(50, 196, 250, 50, 'the order book', '#ffd8d3', False, 12, sub='ordinary process memory')
        if not kdead: o += hb(355, 120, 250, 60, page[0], BLUE, hot == 'k', 12, sub=page[1])
        o += hb(660, 120, 250, 60, disk[0], GREEN, hot == 'd', 12, sub=disk[1])
        if not pdead and not kdead: o += L(302, 150, 353, 150, BT, 2, '4 3') + T(328, 142, 'same page', 9, 600, BT, 'middle')
        if hot == 'd': o += A(607, 150, 658, 150, RED)
        o += T(40, 312, note, 13, 700, GT if good else (RED if good is False else INK))
        return o
    fr = chk([
        ('mmap: the kernel makes a range of the process\'s addresses refer to the pages of the log file.',
         f(('records_ → the mapped range', 'looks like a normal array'), ('file page: [r0, r1]', 'clean: same as the disk'), ('wal file: [r0, r1]', '')) ),
        ('records_[2] = c is an ordinary 40-byte memory write. It lands straight in the kernel\'s page. No system call.',
         f(('records_[2] = c', 'a plain memory store'), ('file page: [r0, r1, c]', 'DIRTY: newer than the disk'), ('wal file: [r0, r1]', 'not written yet'), 'compare: write() would copy the bytes into the kernel through a system call, every record', 'k')),
        ('The kernel writes dirty pages to disk later, on its own schedule (typically within seconds).',
         f(('records_ → the mapped range', ''), ('file page: [r0, r1, c]', 'clean again'), ('wal file: [r0, r1, c]', 'flushed by the kernel'), '', 'd')),
        ('kill -9 before the flush: the process and its book vanish. The dirty page belongs to the KERNEL, so it survives.',
         f(('', ''), ('file page: [r0, r1, c]', 'still dirty, still there'), ('wal file: [r0, r1, c]', 'the kernel flushes it anyway'), 'the record reaches the disk → recovery can replay it ✓', 'd', pdead=True, good=True)),
        ('Power cut before the flush: all RAM is gone, the page cache with it. The disk never got the record.',
         f(('', ''), ('', ''), ('wal file: [r0, r1]', 'c was never written'), 'the record is lost ✗   only msync / fsync (force the flush and wait) protect against this', '', True, True, False)),
        ('This engine calls msync only at the end of a run. So the honest claim is: recovery from PROCESS crashes.',
         f(('records_[n] = c', 'fast: no syscall per record'), ('dirty pages', 'flushed later by the kernel'), ('wal file', 'may lag behind by seconds'), 'survives kill -9 and segfaults; does not promise anything after a power cut or kernel crash')),
    ])
    save('d13y-pagecache.svg', W, H, 'mmap, the page cache, and what each kind of crash destroys', 'where the bytes of a log record actually are', '', fr, 5.5)

# ───────── follower / replica ─────────
def follower():
    W, H = 960, 440
    def f(m, log, fol, snap, note='', hot='', old=False, good=None):
        o = hb(30, 86, 280, 74, m[0], GREEN, hot == 'm', 13, sub=m[1]) + T(30, 78, 'matching thread (the hot path)', 12, 700, MUTE)
        o += hb(30, 176, 280, 50, 'the LIVE book', '#ffd8d3', False, 13)
        if not old:
            o += T(350, 78, 'the log (mmap)', 12, 700, MUTE) + hb(350, 86, 250, 74, log[0], BLUE, hot == 'l', 13, sub=log[1])
            o += A(312, 122, 348, 122) + A(602, 122, 638, 122)
            o += T(640, 78, 'follower thread (background)', 12, 700, MUTE) + hb(640, 86, 290, 74, fol[0], YEL, hot == 'f', 13, sub=fol[1])
            o += hb(640, 176, 290, 50, 'the REPLICA book', '#ffd8d3', hot == 'r', 13)
        o += hb(350 if not old else 420, 250, 250, 50, snap, GREY, hot == 's', 12)
        if hot == 's' and not old: o += A(700, 228, 604, 268, RED)
        if old: o += A(170, 228, 416, 268, RED)
        o += T(40, 336, note, 13, 700, GT if good else (RED if good is False else INK))
        return o
    fr = chk([
        ('First design: every 100,000 commands the MATCHING thread stopped and copied its whole book into a snapshot.',
         f(('copying the book…', 'not matching anything meanwhile'), None, None, 'snapshot file on disk', '1.6 ms stall at 88,390 resting orders: thousands of commands wait in the ring ✗', 'm', True, False)),
        ('New design. The matching thread does two things per command: append it to the log, apply it. It never copies.',
         f(('log(c), then apply(c)', 'count = 52,345'), ('records 0 … 52,344', 'count published with release'), ('idle', ''), 'snapshot file (as of 40,000)', '', 'm')),
        ('The follower reads the log\'s count with acquire, every 200 µs, and sees the new complete records.',
         f(('keeps matching', ''), ('records 0 … 52,344', ''), ('count = 52,345: new records!', 'I have replayed up to 52,000'), 'snapshot file (as of 40,000)', '', 'f')),
        ('It replays them into its OWN book, with the same apply code. Same commands, same order → the same book.',
         f(('keeps matching', ''), ('records 0 … 52,344', ''), ('replaying 52,000 … 52,344', 'ex.apply(record)'), 'snapshot file (as of 40,000)', 'the replica equals the live book, a moment behind', 'r')),
        ('The follower writes the snapshot from the REPLICA. Matching is never paused: about 1.02× cost in throughput.',
         f(('keeps matching', 'never copies, never waits'), ('records 0 … 52,344', ''), ('snapshotting my replica', 'tmp file → fsync → rename'), 'snapshot file (as of 52,345)', 'then it frees the log segments the snapshot covers ✓', 's', False, True)),
    ])
    save('d13y-follower.svg', W, H, 'Taking snapshots without stopping the matching thread', 'move the slow work to a second thread that keeps its own copy of the book', '', fr, 5.5)

if __name__ == '__main__':
    for fn in (types, race, wake, pointer, levels, rest_dbg, cancel_dbg, naive, pagecache, follower): fn()
