"""Line-by-line ('debugger view') animations for the Micro Exchange code. Run: python3 mxfigs2.py <output dir>"""
from mxfigs import *

MONO = "font-family='DejaVu Sans Mono, Menlo, Consolas, monospace'"
def C(x, y, s, size=12, fill=INK, w=400):
    return f"<text x='{x}' y='{y}' {MONO} font-size='{size}' font-weight='{w}' fill='{fill}' xml:space='preserve'>{esc(s)}</text>"

def dbg(name, title, sub, code, steps, vars_order, codew=560, sec=4.2, lh=21):
    """code: list of (lineno, text). steps: (lines to highlight, {var: value}, changed set, caption)."""
    H = max(110 + lh * len(code), 120 + 34 * len(vars_order)) + 70
    W = 960
    st = R(20, 62, codew, H - 124, '#f6f8fa', '#d0d7de', 1, 8)
    for i, (no, txt) in enumerate(code):
        y = 86 + i * lh
        st += C(30, y, f'{no:>3}', 11, '#8c959f') + C(64, y, txt, 12)
    vx = 20 + codew + 16; vw = W - vx - 20
    st += T(vx, 78, 'the variables right now', 12, 700, MUTE)
    fr = []
    state = {}
    for hl, upd, cap in steps:
        state.update(upd)
        b = ''
        for i, (no, _) in enumerate(code):
            if no in hl:
                y = 86 + i * lh
                b += f"<rect x='24' y='{y - 15}' width='{codew - 8}' height='{lh}' rx='3' fill='#fff1a8' fill-opacity='0.75' stroke='{RED}' stroke-width='1.6'/>"
                b += C(30, y, f'{no:>3}', 11, RED, 700) + C(64, y, dict(code)[no], 12, INK, 700)
        for k, v in enumerate(vars_order):
            y = 88 + k * 34
            hot = v in upd
            b += R(vx, y, vw, 28, WHITE, RED if hot else '#d0d7de', 2.2 if hot else 1, 4)
            b += C(vx + 8, y + 19, v, 11, MUTE, 600) + T(vx + vw - 8, y + 19, state.get(v, '–'), 12, 700, RED if hot else INK, 'end')
        fr.append((cap, b))
    save(name, W, H, title, sub, st, fr, sec)

def push_dbg():
    code = [(27, 'bool push(const T& item) {'),
            (28, '  const size_t head = head_.load(relaxed);'),
            (29, '  if (head - tail_cache_ >= Capacity) {'),
            (30, '    tail_cache_ = tail_.load(acquire);'),
            (31, '    if (head - tail_cache_ >= Capacity) return false;'),
            (32, '  }'),
            (33, '  slots_[head & (Capacity - 1)] = item;'),
            (34, '  head_.store(head + 1, release);'),
            (35, '  return true;'),
            (36, '}')]
    V = ['item', 'head (local copy)', 'head_ (shared)', 'tail_cache_ (mine)', 'tail_ (shared)', 'slots_[0..3]', 'consumer can see']
    steps = [
        ((27,), {'item': 'E', 'head (local copy)': '–', 'head_ (shared)': '4', 'tail_cache_ (mine)': '0', 'tail_ (shared)': '1', 'slots_[0..3]': 'a B C D', 'consumer can see': 'B C D'},
         'push(E) is called. 4 items were pushed, 1 popped (A, shown small). The producer still believes tail is 0.'),
        ((28,), {'head (local copy)': '4'}, 'Line 28: read my own counter ONCE and keep a private copy. Nobody else writes head_, so relaxed is enough.'),
        ((29,), {}, 'Line 29: items I think are inside = head − tail_cache_ = 4 − 0 = 4 ≥ 4. It LOOKS full (my copy of tail is old).'),
        ((30,), {'tail_cache_ (mine)': '1'}, 'Line 30: only now read the consumer\'s real tail_. acquire: its pop of slot 0 is finished and visible.'),
        ((31,), {}, 'Line 31: check again with the fresh value: 4 − 1 = 3 ≥ 4? No. There is room, go on.'),
        ((33,), {'slots_[0..3]': 'E B C D'}, 'Line 33: copy the 40 bytes into slot 4 & 3 = 0. head_ is still 4, so the consumer cannot see E yet.'),
        ((34,), {'head_ (shared)': '5', 'consumer can see': 'B C D E'}, 'Line 34: publish. head_ = 5 with release: whoever sees 5 also sees the bytes written on line 33.'),
        ((35,), {}, 'Line 35: tell the caller it worked. (false on line 31 would mean: full, try again.)'),
    ]
    dbg('d13x-push.svg', 'push(E), one line at a time', 'ring_buffer.hpp:27–36, Capacity 4; memory orders shortened (relaxed = std::memory_order_relaxed)', code, steps, V)

def pop_dbg():
    code = [(38, 'bool pop(T& out) {'),
            (39, '  const size_t tail = tail_.load(relaxed);'),
            (40, '  if (tail == head_cache_) {'),
            (41, '    head_cache_ = head_.load(acquire);'),
            (42, '    if (tail == head_cache_) return false;'),
            (43, '  }'),
            (44, '  out = slots_[tail & (Capacity - 1)];'),
            (45, '  tail_.store(tail + 1, release);'),
            (46, '  return true;'),
            (47, '}')]
    V = ['out', 'tail (local copy)', 'tail_ (shared)', 'head_cache_ (mine)', 'head_ (shared)', 'slots_[0..3]', 'producer may reuse']
    steps = [
        ((38,), {'out': '–', 'tail (local copy)': '–', 'tail_ (shared)': '0', 'head_cache_ (mine)': '0', 'head_ (shared)': '4', 'slots_[0..3]': 'A B C D', 'producer may reuse': 'no slot'},
         'pop(out) is called. The producer has pushed A B C D. The consumer still believes head is 0.'),
        ((39,), {'tail (local copy)': '0'}, 'Line 39: read my own counter once into a private copy. Only the consumer writes tail_, so relaxed.'),
        ((40,), {}, 'Line 40: tail == head_cache_ → 0 == 0. It LOOKS empty (my copy of head is old).'),
        ((41,), {'head_cache_ (mine)': '4'}, 'Line 41: only now read the producer\'s real head_. acquire: the slots it wrote before publishing are visible.'),
        ((42,), {}, 'Line 42: check again: 0 == 4? No. There is something to read, go on.'),
        ((44,), {'out': 'A'}, 'Line 44: copy slot 0 & 3 = 0 into out. tail_ is still 0, so the producer may not touch slot 0 yet.'),
        ((45,), {'tail_ (shared)': '1', 'producer may reuse': 'slot 0'}, 'Line 45: publish. tail_ = 1 with release: the copy on line 44 is finished before the slot is given back.'),
        ((46,), {}, 'Line 46: tell the caller out holds a command. (false on line 42 would mean: really empty.)'),
    ]
    dbg('d13x-pop.svg', 'pop(out), one line at a time', 'ring_buffer.hpp:38–47, Capacity 4; the mirror image of push', code, steps, V)

def match_dbg():
    code = [(127, 'while (qty > 0 && best_ask_ <= price) {'),
            (128, '  Level& lvl = asks_[best_ask_];'),
            (129, '  while (qty > 0 && lvl.head) {'),
            (130, '    Order* maker = lvl.head;'),
            (131, '    const uint64_t x = std::min(qty, maker->qty);'),
            (132, '    qty -= x;'),
            (133, '    maker->qty -= x;'),
            (134, '    lvl.qty -= x;'),
            (135, '    resting_qty_ -= x;'),
            (136, '    filled_qty_ += 2 * x;'),
            (137, '    filled += x;'),
            (138, '    if (record_trades_)'),
            (139, '      trades_.push_back(Trade{id, maker->id, ...});'),
            (140, '    if (maker->qty == 0) erase_order(maker);'),
            (141, '  }'),
            (142, '  if (lvl.head == nullptr) raise_best_ask();'),
            (143, '}')]
    V = ['price (my limit)', 'qty (still unfilled)', 'best_ask_', 'lvl.head', 'lvl.qty', 'maker', 'maker->qty', 'x (this fill)', 'filled', 'last trade']
    steps = [
        ((127,), {'price (my limit)': '102', 'qty (still unfilled)': '70', 'best_ask_': '101', 'lvl.head': '–', 'lvl.qty': '–', 'maker': '–', 'maker->qty': '–', 'x (this fill)': '–', 'filled': '0', 'last trade': '–'},
         'Line 127: do I still want something (70 > 0) AND is the cheapest seller cheap enough (101 ≤ 102)? Yes.'),
        ((128,), {'lvl.head': 'S1', 'lvl.qty': '60'}, 'Line 128: lvl is another name for the Level at price 101. No copy: changes to lvl change the book.'),
        ((129,), {}, 'Line 129: still want some (70 > 0) AND someone is waiting at this price (head = S1)? Yes.'),
        ((130,), {'maker': 'S1', 'maker->qty': '40'}, 'Line 130: take the HEAD of the level: the order that has waited longest. This line is time priority.'),
        ((131,), {'x (this fill)': '40'}, 'Line 131: trade the smaller of what I want (70) and what S1 has (40): x = 40.'),
        ((132, 133, 134), {'qty (still unfilled)': '30', 'maker->qty': '0', 'lvl.qty': '20'}, 'Lines 132–134: take x off my order, off S1, and off the level\'s total.'),
        ((135, 136, 137), {'filled': '40'}, 'Lines 135–137: the book-wide counters. filled_qty_ gets 2·x: one share for the buyer, one for the seller.'),
        ((138, 139), {'last trade': '9 × S1, 40 @ 101'}, 'Lines 138–139: write down the trade. Its price is maker->price (101), not my 102.'),
        ((140,), {'lvl.head': 'S2', 'maker': 'S1 (erased)'}, 'Line 140: S1 has 0 left, so unlink it. The level\'s head is now S2.'),
        ((129,), {}, 'Back to line 129: 30 > 0 and head = S2. Go round again.'),
        ((130, 131), {'maker': 'S2', 'maker->qty': '20', 'x (this fill)': '20'}, 'Lines 130–131: maker = S2. x = min(30, 20) = 20.'),
        ((132, 133, 134, 135, 136, 137), {'qty (still unfilled)': '10', 'maker->qty': '0', 'lvl.qty': '0', 'filled': '60'}, 'Lines 132–137: I still want 10; S2 is at 0; the level holds 0; filled = 60.'),
        ((138, 139, 140), {'last trade': '9 × S2, 20 @ 101', 'lvl.head': 'null', 'maker': 'S2 (erased)'}, 'Lines 139–140: record the trade, erase S2. Nobody is left at 101: head = null.'),
        ((129,), {}, 'Line 129: 10 > 0 but head is null. Leave the inner loop.'),
        ((142,), {'best_ask_': '103', 'lvl.head': '– (level 101 empty)'}, 'Line 142: this level is empty, so find the next price that has sellers: 102 no, 103 yes.'),
        ((127,), {}, 'Line 127: 10 > 0 but 103 ≤ 102 is false. Leave the loop. The 10 left over goes to rest() on line 164.'),
    ]
    dbg('d13x-match.svg', 'The match loop, one line at a time: buy 70 @ 102', 'order_book.cpp:127–143 (comments left out); asks at 101 = [S1 40, S2 20], at 103 = [S3 50]', code, steps, V, sec=4.4)

def snapshot():
    W, H = 960, 330
    def f(shared, local, uses='', hot=''):
        b = B(40, 100, 300, 70, f'head_ = {shared}', BLUE, RED if hot == 's' else MUTE, 2.4 if hot == 's' else 1.2, 14, sub='shared: std::atomic<size_t>, in memory')
        b += B(620, 100, 300, 70, f'head = {local}', YEL, RED if hot == 'l' else MUTE, 2.4 if hot == 'l' else 1.2, 14, sub='local: const size_t, in a register')
        if hot == 'l': b += A(342, 135, 618, 135, RED) + T(480, 124, '.load() copies the value', 12, 700, RED, 'middle')
        if hot == 's': b += A(618, 150, 342, 150, RED) + T(480, 172, '.store(head + 1) writes back', 12, 700, RED, 'middle')
        b += T(40, 220, uses, 13, 600)
        return b
    fr = [
        ('Before line 28. head_ lives in memory where both threads can reach it. The local does not exist yet.', f(4, '?')),
        ('Line 28: head_.load(...) reads memory once and the result is kept in the local head.', f(4, 4, hot='l')),
        ('Lines 29, 31, 33, 34 all use the LOCAL head. Same value every time, and no more trips to shared memory.', f(4, 4, 'used by: head − tail_cache_ (29, 31)   head & (Capacity − 1) (33)   head + 1 (34)')),
        ('Line 34: head_.store(head + 1) writes 5 into the shared variable. The local is const: it stays 4.', f(5, 4, hot='s')),
    ]
    save('d13x-load.svg', W, H, 'load = take a snapshot; store = write it back', 'const size_t head = head_.load(...): one read of the shared counter, then work on the private copy', '', fr, 4.4)

def stale():
    W, H = 960, 380
    def f(mem, reg, code, verdict, bad=True, hot=''):
        b = R(40, 80, 430, 60, '#f6f8fa', '#d0d7de', 1, 8) + C(56, 104, 'size_t head_;          // plain, NOT atomic', 12) + C(56, 126, code, 12, RED if hot == 'c' else INK, 700)
        b += B(520, 80, 190, 60, f'head_ in memory: {mem}', BLUE, RED if hot == 'm' else MUTE, 2.4 if hot == 'm' else 1.2, 12)
        b += B(730, 80, 190, 60, f'consumer register: {reg}', YEL, RED if hot == 'r' else MUTE, 2.4 if hot == 'r' else 1.2, 12)
        b += T(40, 190, verdict, 13, 700, RED if bad else GT)
        return b
    good = lambda: (R(40, 230, 880, 76, WHITE, GT, 1.6, 8) + C(56, 256, 'std::atomic<size_t> head_;   while (tail == head_.load(acquire)) { }', 12, GT, 700)
                    + T(56, 284, 'atomic: every .load() must really read memory, so the next trip round the loop sees 5 and exits ✓', 12, 600, GT))
    fr = [
        ('The consumer waits for work: it loops while the ring looks empty. head_ here is a plain variable.', f(4, '–', 'while (tail == head_) { }   // tail is 4', 'the source says: read head_ every time round the loop')),
        ('The compiler sees no code in THIS thread that changes head_. It may read it once and keep it in a register.', f(4, 4, 'r = head_;  while (tail == r) { }', 'legal for a plain variable: a second thread writing it is a data race, which C++ leaves undefined', hot='r')),
        ('The producer pushes: memory now holds 5. The consumer\'s register still holds 4.', f(5, 4, 'r = head_;  while (tail == r) { }', 'the new value never reaches the loop', hot='m')),
        ('4 == 4 forever. The consumer spins and never sees the command. With std::atomic this cannot happen.', f(5, 4, 'r = head_;  while (tail == r) { }', 'the consumer hangs ✗', hot='c') + good()),
    ]
    save('d13x-stale.svg', W, H, 'Why head_ must be std::atomic at all', 'what a compiler is allowed to do with a plain shared variable', '', fr, 5.0)

def swap():
    W, H = 960, 420
    def f(head_, slots, cons, verdict='', hot=None, bad=False):
        b = R(40, 74, 400, 60, '#f6f8fa', '#d0d7de', 1, 8) + C(56, 98, 'head_.store(head + 1, release);  // first', 12, RED, 700) + C(56, 120, 'slots_[head & 3] = item;         // second', 12, RED, 700)
        b += T(470, 96, 'lines 33 and 34 swapped', 13, 700, RED)
        for i, sv in enumerate(slots):
            b += B(60 + i * 90, 170, 80, 56, sv, YEL if i else '#ffe8a3', RED if hot == i else MUTE, 2.4 if hot == i else 1.2, 16)
            b += T(100 + i * 90, 244, f'slot {i}', 10, 400, MUTE, 'middle')
        b += B(470, 170, 180, 56, f'head_ = {head_}', BLUE, RED if hot == 'h' else MUTE, 2.4 if hot == 'h' else 1.2, 14)
        b += B(680, 170, 240, 56, cons, PINK if bad else GREEN, RED if bad else MUTE, 2.4 if bad else 1.2, 12)
        b += T(60, 290, verdict, 13, 700, RED if bad else INK)
        return b
    fr = [
        ('Start: A B C D were pushed, A was popped (tail = 1). Slot 0 still holds the old bytes of A. The producer pushes E.', f(4, ['A (old)', 'B', 'C', 'D'], 'consumer: tail = 1 … later 4')),
        ('The producer publishes FIRST: head_ = 5. Slot 0 has not been written yet.', f(5, ['A (old)', 'B', 'C', 'D'], 'consumer: sees head = 5', 'index 4 is now "ready" according to head_', 'h')),
        ('The consumer has popped B, C, D (tail = 4). It sees 4 ≠ 5 and reads slot 4 & 3 = 0.', f(5, ['A (old)', 'B', 'C', 'D'], 'out = slot 0 = A', 'it gets A a second time (or half of E, if the copy is under way) ✗', 0, True)),
        ('The producer writes E into slot 0. Too late: the consumer has already moved past index 4.', f(5, ['E', 'B', 'C', 'D'], 'tail = 5: slot 0 counts as read', 'result: command A applied twice, command E never applied ✗', 0, True)),
        ('The real order: write the slot (33), then release-store head_ (34). A reader that sees 5 always finds E.', f(5, ['E', 'B', 'C', 'D'], 'out = slot 0 = E ✓', 'write first, publish second')),
    ]
    save('d13x-swap.svg', W, H, 'What breaks if head is published before the slot is written', 'the two lines of push in the wrong order', '', fr, 5.0)

def session():
    W, H = 960, 600
    prices = [103, 102, 101, 100, 99]
    def draw(cmd, result, asks, bids, ghosts, hot, ba, bb, cnt, chg, bad=False):
        b = B(30, 62, 430, 40, cmd, YEL, RED, 2, 13) + T(476, 87, result, 13, 700, RED if bad else INK)
        b += T(250, 128, 'bids (buys)', 12, 700, GT, 'middle') + T(480, 128, 'price', 12, 700, MUTE, 'middle') + T(710, 128, 'asks (sells)', 12, 700, RED, 'middle')
        for r, p in enumerate(prices):
            y = 138 + r * 50
            b += R(440, y, 80, 42, GREY, MUTE, 1) + T(480, y + 27, p, 15, 700, INK, 'middle')
            k = 0
            for name, q in asks.get(p, []):
                h = name in hot
                b += B(540 + k * 124, y + 2, 116, 38, f'{name}  qty {q}', '#ffd8d3', RED if h else MUTE, 2.6 if h else 1.2, 12); k += 1
            for side, gp, lab in ghosts:
                if side == 'A' and gp == p:
                    b += R(540 + k * 124, y + 2, 116, 38, WHITE, MUTE, 1.2, 4, '5 4') + T(598 + k * 124, y + 26, lab, 10, 600, MUTE, 'middle'); k += 1
            k = 0
            for name, q in bids.get(p, []):
                h = name in hot
                b += B(304 - k * 124, y + 2, 116, 38, f'{name}  qty {q}', GREEN, RED if h else MUTE, 2.6 if h else 1.2, 12); k += 1
            for side, gp, lab in ghosts:
                if side == 'B' and gp == p:
                    b += R(304 - k * 124, y + 2, 116, 38, WHITE, MUTE, 1.2, 4, '5 4') + T(362 - k * 124, y + 26, lab, 10, 600, MUTE, 'middle'); k += 1
            if p == ba: b += T(930, y + 27, '← best ask', 11, 700, RED, 'end')
            if p == bb: b += T(30, y + 27, 'best bid →', 11, 700, GT)
        b += T(30, 404, f"best ask: {ba or 'none'}     best bid: {bb or 'none'}", 12, 600, MUTE)
        names = ['submitted', 'filled', 'resting', 'cancelled', 'expired']
        for i, nme in enumerate(names):
            x = 30 + i * 150 + (30 if i else 0)
            h = nme in chg
            b += R(x, 424, 140, 54, BLUE if i == 0 else WHITE, RED if h else MUTE, 2.6 if h else 1.2)
            b += T(x + 70, 444, nme, 11, 600, MUTE, 'middle') + T(x + 70, 468, cnt[i], 18, 700, RED if h else INK, 'middle')
        b += T(178, 458, '=', 20, 700, INK, 'middle')
        for i in (2, 3, 4): b += T(60 + i * 150 - 5, 456, '+', 13, 700, INK, 'middle')
        ok = cnt[0] == sum(cnt[1:])
        b += T(820, 458, '✓ balances' if ok else '✗', 13, 700, GT, 'start')
        b += T(30, 500, 'dashed box = an order that just left the book, and why      red border = new or changed in this step', 11, 400, MUTE)
        return b
    A1 = {101: [('#1', 40)]}; A2 = {101: [('#1', 40), ('#2', 20)]}; A3 = {101: [('#1', 40), ('#2', 20)], 103: [('#3', 50)]}
    B4 = {99: [('#4', 30)]}; A6 = {103: [('#3', 50)]}
    fr = [
        ('Start: an empty book. Nothing submitted yet.', draw('(no command yet)', '', {}, {}, [], (), None, None, [0, 0, 0, 0, 0], ())),
        ('1. Nobody is buying, so the sell cannot trade. It rests at 101 and becomes the best ask.',
         draw('1  New  Sell  Limit  id 1   40 @ 101', 'rests', A1, {}, [], ('#1',), 101, None, [40, 0, 40, 0, 0], ('submitted', 'resting'))),
        ('2. Same price as #1, but it arrived later: it queues BEHIND #1. That queue order is time priority.',
         draw('2  New  Sell  Limit  id 2   20 @ 101', 'rests behind #1', A2, {}, [], ('#2',), 101, None, [60, 0, 60, 0, 0], ('submitted', 'resting'))),
        ('3. A worse (higher) sell price. It rests on its own level; the best ask stays 101.',
         draw('3  New  Sell  Limit  id 3   50 @ 103', 'rests', A3, {}, [], ('#3',), 101, None, [110, 0, 110, 0, 0], ('submitted', 'resting'))),
        ('4. A buy at 99. The cheapest seller wants 101 > 99, so no trade. It rests and becomes the best bid.',
         draw('4  New  Buy  Limit  id 4   30 @ 99', 'no cross (101 > 99): rests', A3, B4, [], ('#4',), 101, 99, [140, 0, 140, 0, 0], ('submitted', 'resting'))),
        ('5. Id 1 is already resting in the book. A Limit order with a duplicate id is rejected: its 5 expires.',
         draw('5  New  Sell  Limit  id 1   5 @ 105', 'REJECTED: duplicate id', A3, B4, [], (), 101, 99, [145, 0, 140, 0, 5], ('submitted', 'expired'), True)),
        ('6. Crosses: fills 40 with #1, then 20 with #2, both at 101. Level 101 empties; the last 10 rests at 102.',
         draw('6  New  Buy  Limit  id 9   70 @ 102', 'filled 60 @ 101, rests 10', A6, {102: [('#9', 10)], 99: [('#4', 30)]}, [('A', 101, '#1 filled 40'), ('A', 101, '#2 filled 20')], ('#9',), 103, 102, [215, 120, 90, 0, 5], ('submitted', 'filled', 'resting'))),
        ('7. Found through the id table and unlinked. Its 10 moves from resting to cancelled; best bid falls back to 99.',
         draw('7  Cancel  id 9', 'cancelled 10', A6, B4, [('B', 102, '#9 cancelled 10')], (), 103, 99, [215, 120, 80, 10, 5], ('resting', 'cancelled'))),
        ('8. IOC: fills 50 with #3 at 103. It wanted 80, but an IOC never rests: the other 30 expire. No sellers remain.',
         draw('8  New  Buy  IOC  id 10   80 @ 103', 'filled 50 @ 103, 30 expire', {}, B4, [('A', 103, '#3 filled 50')], (), None, 99, [295, 220, 30, 10, 35], ('submitted', 'filled', 'resting', 'expired'))),
        ('9. Market: takes any price. Fills 30 with #4 at 99; nobody else is buying, so the other 20 expire.',
         draw('9  New  Sell  Market  id 11   qty 50', 'filled 30 @ 99, 20 expire', {}, {}, [('B', 99, '#4 filled 30')], (), None, None, [345, 280, 0, 10, 55], ('submitted', 'filled', 'resting', 'expired'))),
        ('10. Id 2 was filled in step 6, so it is no longer in the id table. cancel returns false; nothing changes.',
         draw('10  Cancel  id 2', 'returns false: not in the book', {}, {}, [], (), None, None, [345, 280, 0, 10, 55], (), True)),
    ]
    save('d13x-session.svg', W, H, 'Ten commands, and the book after each one', 'every order type and every way a quantity can end; a trade of x adds 2x to filled (buyer and seller)', '', fr, 5.0)

if __name__ == '__main__':
    for fn in (push_dbg, pop_dbg, match_dbg, snapshot, stale, swap, session): fn()
