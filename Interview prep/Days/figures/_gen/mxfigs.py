"""Animated step figures for the Micro Exchange part of Day 13 (SMIL, same look as the d05s-* figures)."""
import sys, os
OUT = sys.argv[1]
INK, MUTE, RED = '#1f2328', '#57606a', '#cf222e'
BLUE, GREEN, YEL, PINK, GREY, WHITE = '#c8e1ff', '#c7f0d2', '#fff1a8', '#f5c6e9', '#eaeef2', '#ffffff'
BT, GT = '#0969da', '#1a7f37'

def esc(s): return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
def T(x, y, s, size=12, w=400, fill=INK, anchor='start'):
    return f"<text x='{x}' y='{y}' text-anchor='{anchor}' font-size='{size}' font-weight='{w}' fill='{fill}'>{esc(s)}</text>"
def R(x, y, w, h, fill=WHITE, stroke=MUTE, sw=1.2, rx=4, dash=None):
    d = f" stroke-dasharray='{dash}'" if dash else ''
    return f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{rx}' fill='{fill}' stroke='{stroke}' stroke-width='{sw}'{d}/>"
def B(x, y, w, h, label, fill=WHITE, stroke=MUTE, sw=1.2, size=12, wt=600, tfill=INK, sub=None):
    """box with a centred label (and optional second line)"""
    o = R(x, y, w, h, fill, stroke, sw)
    if sub is None:
        o += T(x + w / 2, y + h / 2 + size * 0.36, label, size, wt, tfill, 'middle')
    else:
        o += T(x + w / 2, y + h / 2 - 2, label, size, wt, tfill, 'middle')
        o += T(x + w / 2, y + h / 2 + size + 1, sub, size - 2, 400, MUTE, 'middle')
    return o
def A(x1, y1, x2, y2, color=MUTE, sw=1.6):
    m = 'ahr' if color == RED else 'ah'
    return f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-width='{sw}' marker-end='url(#{m})'/>"
def L(x1, y1, x2, y2, color=MUTE, sw=1.2, dash=None):
    d = f" stroke-dasharray='{dash}'" if dash else ''
    return f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-width='{sw}'{d}/>"

def save(name, W, H, title, sub, static, frames, sec=3.2):
    """frames: list of (caption, svg string). One frame visible at a time, looping."""
    n = len(frames); dur = n * sec
    o = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}' font-family='Inter, Helvetica, Arial, sans-serif'><title>{esc(title)}</title>",
         "<defs><marker id='ah' markerWidth='10' markerHeight='10' refX='8' refY='5' orient='auto'><path d='M0,0 L10,5 L0,10 z' fill='#57606a'/></marker>"
         "<marker id='ahr' markerWidth='10' markerHeight='10' refX='8' refY='5' orient='auto'><path d='M0,0 L10,5 L0,10 z' fill='#cf222e'/></marker></defs>",
         R(0, 0, W, H, WHITE, '#d0d7de', 1, 12), T(20, 28, title, 15, 700), T(20, 48, sub, 12, 400, MUTE), static]
    for i, (cap, body) in enumerate(frames):
        a, b = i / n, (i + 1) / n
        if i == 0: vals, keys = '1;0;0', f'0;{b:.4f};1'
        elif i == n - 1: vals, keys = '0;1;1', f'0;{a:.4f};1'
        else: vals, keys = '0;1;0;0', f'0;{a:.4f};{b:.4f};1'
        op = '1' if i == 0 else '0'
        o.append(f"<g opacity='{op}'><animate attributeName='opacity' dur='{dur:.1f}s' repeatCount='indefinite' calcMode='discrete' values='{vals}' keyTimes='{keys}'/>")
        o.append(R(20, H - 46, W - 40, 30, '#f6f8fa', '#d0d7de', 1, 6))
        o.append(T(32, H - 26, f"{i + 1}/{n}", 12, 700, RED))
        o.append(T(72, H - 26, cap, 12, 600))
        o.append(body); o.append('</g>')
    o.append('</svg>')
    open(os.path.join(OUT, name), 'w', encoding='utf-8').write('\n'.join(o))
    print(name, n, 'frames')

# ───────────────────────── 1. pipeline ─────────────────────────
def pipeline():
    W, H = 960, 330
    xs = [30, 220, 410, 600, 790]; names = ['producer thread', 'SPSC ring', 'matching thread', 'WAL (mmap file)', 'order book']
    subs = ['core A', '65,536 slots × 40 B', 'core B', 'records_[n], 40 B each', 'levels, id table, pool']
    st = ''
    for x, n, s in zip(xs, names, subs):
        st += B(x, 90, 140, 70, n, GREY, sub=s)
    for i in range(4): st += A(xs[i] + 140, 125, xs[i + 1], 125)
    tok = lambda x, lab='Command 40 B': B(x + 15, 185, 110, 34, lab, YEL, RED, 2, 11)
    fr = [
        ('The producer builds a Command: New, Buy, Limit, id 9, price 102, qty 70, tsc.', tok(xs[0])),
        ('push: the 40 bytes are copied into slots_[head & mask]; then head is release-stored.', tok(xs[1]) + T(xs[1] + 70, 240, 'copy 1', 11, 600, RED, 'middle')),
        ('pop: the matching thread acquire-loads head, copies the slot out, release-stores tail.', tok(xs[2]) + T(xs[2] + 70, 240, 'copy 2', 11, 600, RED, 'middle')),
        ('Log first: append copies the same 40 bytes into the mmap, then release-stores count + 1.', tok(xs[3]) + T(xs[3] + 70, 240, 'copy 3', 11, 600, RED, 'middle')),
        ('Only then apply: submit(id 9, Buy, 102, 70) matches against the asks and rests the remainder.', tok(xs[4], 'id, side, price, qty') + T(xs[4] + 70, 240, 'fields passed', 11, 600, RED, 'middle')),
        ('Latency = rdtsc() − c.tsc goes into the histogram. The follower thread replays the log on its replica.', B(xs[2] + 5, 185, 130, 34, 'rdtsc() − c.tsc', PINK, RED, 2, 11)),
    ]
    save('d13m-pipeline.svg', W, H, 'One command through the exchange', 'the same 40 bytes are copied three times before the book changes', st, fr, 3.6)

# ───────────────────────── 2. ring ─────────────────────────
def ring():
    W, H = 960, 430
    steps = [  # caption, head, tail, tail_cache, head_cache, slots, changed set, slot touched
        ('start: empty ring. size = head − tail = 0.', 0, 0, 0, 0, '----', set(), None),
        ('push A: 0 − 0 < 4, room. Write slot 0 & 3 = 0, then head = 1.', 1, 0, 0, 0, 'A---', {'head'}, 0),
        ('push B: write slot 1, then head = 2.', 2, 0, 0, 0, 'AB--', {'head'}, 1),
        ('push C: write slot 2, then head = 3.', 3, 0, 0, 0, 'ABC-', {'head'}, 2),
        ('push D: write slot 3, then head = 4. size = 4: the ring is full.', 4, 0, 0, 0, 'ABCD', {'head'}, 3),
        ('pop: tail == head_cache (0 == 0) looks empty → reload head_cache = 4. Read slot 0 (A), tail = 1.', 4, 1, 0, 4, 'ABCD', {'tail', 'hc'}, 0),
        ('push E: 4 − 0 ≥ 4 looks full → reload tail_cache = 1. Room: write slot 4 & 3 = 0, head = 5.', 5, 1, 1, 4, 'EBCD', {'head', 'tc'}, 0),
        ('pop: tail 1 ≠ head_cache 4 → no reload at all. Read slot 1 (B), tail = 2.', 5, 2, 1, 4, 'EBCD', {'tail'}, 1),
        ('push F: 5 − 1 ≥ 4 → reload tail_cache = 2. Write slot 5 & 3 = 1, head = 6.', 6, 2, 2, 4, 'EFCD', {'head', 'tc'}, 1),
        ('push G: 6 − 2 ≥ 4 → reload: still 2 → 6 − 2 = 4 ≥ 4 → return false. Nothing written.', 6, 2, 2, 4, 'EFCD', set(), None),
    ]
    st = T(120, 92, 'slots_ (Capacity 4)', 12, 600, MUTE)
    st += B(560, 80, 170, 26, 'producer writes', BLUE, size=11) + B(760, 80, 170, 26, 'consumer writes', GREEN, size=11)
    fr = []
    for cap, h, t, tc, hc, sl, ch, touched in steps:
        b = ''
        for i in range(4):
            x = 120 + i * 90
            live = (t <= h - 1) and any((k & 3) == i for k in range(t, h))
            fill = YEL if live else GREY
            b += R(x, 110, 80, 70, fill, RED if touched == i else MUTE, 2.4 if touched == i else 1.2)
            b += T(x + 40, 153, sl[i] if sl[i] != '-' else '', 22, 700, INK if live else '#8c959f', 'middle')
            b += T(x + 40, 198, f'slot {i}', 11, 400, MUTE, 'middle')
        # pointers
        hx = 120 + (h & 3) * 90 + 25; tx = 120 + (t & 3) * 90 + 55
        b += A(hx, 250, hx, 208, BT) + T(hx - 6, 254, f'head & 3 = {h & 3}', 11, 600, BT, 'end')
        b += A(tx, 300, tx, 208, GT) + T(tx + 6, 304, f'tail & 3 = {t & 3}', 11, 600, GT, 'start')
        b += T(120, 350, f'size = head − tail = {h} − {t} = {h - t}', 13, 700)
        def cell(x, y, name, val, key, col):
            hot = key in ch
            return R(x, y, 170, 40, WHITE, RED if hot else MUTE, 2.4 if hot else 1.2) + T(x + 12, y + 25, name, 12, 600, col) + T(x + 158, y + 26, val, 16, 700, RED if hot else INK, 'end')
        b += cell(560, 116, 'head_', h, 'head', BT) + cell(560, 166, 'tail_cache_', tc, 'tc', MUTE)
        b += cell(760, 116, 'tail_', t, 'tail', GT) + cell(760, 166, 'head_cache_', hc, 'hc', MUTE)
        b += T(560, 232, 'yellow = holds an item not yet popped', 11, 400, MUTE) + T(560, 250, 'grey letter = already popped, not erased', 11, 400, MUTE)
        b += T(560, 268, 'red = changed or touched in this step', 11, 400, RED)
        fr.append((cap, b))
    save('d13m-ring.svg', W, H, 'SPSC ring, Capacity 4: nine calls', 'indices only grow; slot = index & 3; each side reloads the other index only when its cached copy says full or empty', st, fr, 4.0)

# ───────────────────────── 3. ordering ─────────────────────────
def ordering():
    W, H = 960, 440
    def lane(y, title, col):
        return R(20, y, 920, 150, WHITE, '#d0d7de', 1, 8) + T(34, y + 22, title, 13, 700, col)
    st = lane(64, 'relaxed stores and loads: nothing forbids this order', RED) + lane(224, 'write slot → release-store head ⇒ acquire-load head → read slot', GT)
    def row(y, filled, head, cons_head, got, bad=False):
        b = T(40, y + 52, 'slot 0', 11, 600, MUTE)
        for i in range(5):
            b += R(90 + i * 34, y + 36, 30, 26, YEL if i < filled else GREY, MUTE, 1)
        b += T(90, y + 80, f'{filled * 8} of 40 bytes written', 11, 400, MUTE)
        b += B(300, y + 34, 130, 32, f'head_ = {head}', BLUE, size=12)
        b += T(300, y + 82, 'what the producer has stored', 10, 400, MUTE)
        b += B(520, y + 34, 150, 32, f'consumer sees head {cons_head}', GREEN, size=11)
        if got:
            b += B(710, y + 34, 200, 32, got, PINK if bad else GREEN, RED if bad else GT, 2.2, 11)
        return b
    fr = [
        ('The producer is in the middle of copying the 40-byte command into slot 0.', row(64, 2, 4, 4, None) + row(224, 2, 4, 4, None)),
        ('Top: the head store becomes visible early (reordered). Bottom: head cannot pass the slot write.', row(64, 2, 5, 5, None) + row(224, 5, 4, 4, None)),
        ('Top: the consumer sees head = 5 and copies a half-written slot. Bottom: head is release-stored.', row(64, 2, 5, 5, 'reads a torn command ✗', True) + row(224, 5, 5, 4, None)),
        ('Bottom: the consumer acquire-loads head = 5. Everything written before the release is visible.', row(64, 2, 5, 5, 'reads a torn command ✗', True) + row(224, 5, 5, 5, None)),
        ('Bottom: only now it reads the slot: all 40 bytes are there.', row(64, 2, 5, 5, 'reads a torn command ✗', True) + row(224, 5, 5, 5, 'reads the whole command ✓')),
    ]
    save('d13m-ordering.svg', W, H, 'Why head is published with release and read with acquire', 'the same two steps (write the slot, publish head) with and without an ordering rule', st, fr, 4.2)

# ───────────────────────── 4. false sharing ─────────────────────────
def falseshare():
    W, H = 960, 400
    st = T(40, 84, 'without alignas: head_ and tail_ on ONE 64-byte cache line', 13, 700, RED) + T(520, 84, 'with alignas(64): one line each', 13, 700, GT)
    st += L(490, 66, 490, 340, '#d0d7de')
    def core(x, y, name, has, col):
        return B(x, y, 120, 50, name, col if has else GREY, sub='owns the line' if has else 'copy invalid')
    def line1(owner):
        b = R(90, 200, 320, 44, YEL, MUTE) + T(110, 227, 'head_', 13, 700, BT) + T(250, 227, 'tail_', 13, 700, GT) + T(250, 262, 'one cache line', 11, 400, MUTE, 'middle')
        b += core(60, 110, 'core A (producer)', owner == 'A', BLUE) + core(320, 110, 'core B (consumer)', owner == 'B', GREEN)
        return b
    right = (B(540, 110, 150, 50, 'core A (producer)', BLUE, sub='owns its line') + B(770, 110, 150, 50, 'core B (consumer)', GREEN, sub='owns its line')
             + R(530, 200, 170, 44, YEL, MUTE) + T(545, 227, 'head_  tail_cache_', 12, 700, BT)
             + R(760, 200, 170, 44, YEL, MUTE) + T(775, 227, 'tail_  head_cache_', 12, 700, GT)
             + T(730, 290, 'no line ever changes owner', 12, 600, GT, 'middle'))
    def xfer(n): return T(250, 300, f'line transfers so far: {n}', 13, 700, RED, 'middle')
    fr = [
        ('push writes head_: core A must own the whole line. Core B\'s copy is invalidated.', line1('A') + xfer(1) + A(120, 162, 150, 198, RED) + right),
        ('pop writes tail_: a different variable, but the same line. It moves to core B.', line1('B') + xfer(2) + A(380, 162, 350, 198, RED) + right),
        ('next push: the line moves back to core A.', line1('A') + xfer(3) + A(120, 162, 150, 198, RED) + right),
        ('next pop: back to core B. Every operation pays a cross-core transfer. That is false sharing.', line1('B') + xfer(4) + A(380, 162, 350, 198, RED) + right),
    ]
    save('d13m-falseshare.svg', W, H, 'False sharing, and what alignas(64) fixes', 'a core must own a whole cache line to write any byte of it', st, fr, 3.6)

# ───────────────────────── 5. matching ─────────────────────────
def match():
    W, H = 960, 520
    prices = [103, 102, 101, 100, 99]
    def ladder(asks, bids, best_ask, best_bid, hot=()):
        b = T(250, 84, 'bids (buys)', 12, 700, GT, 'middle') + T(480, 84, 'price', 12, 700, MUTE, 'middle') + T(710, 84, 'asks (sells)', 12, 700, RED, 'middle')
        for r, p in enumerate(prices):
            y = 96 + r * 56
            b += R(440, y, 80, 46, GREY, MUTE, 1) + T(480, y + 29, p, 15, 700, INK, 'middle')
            for k, (name, q) in enumerate(asks.get(p, [])):
                x = 540 + k * 108
                b += B(x, y + 3, 100, 40, f'{name}  qty {q}', '#ffd8d3', RED if name in hot else MUTE, 2.6 if name in hot else 1.2, 12)
            for k, (name, q) in enumerate(bids.get(p, [])):
                x = 320 - k * 108
                b += B(x, y + 3, 100, 40, f'{name}  qty {q}', GREEN, RED if name in hot else MUTE, 2.6 if name in hot else 1.2, 12)
            if p == best_ask: b += T(930, y + 29, '← best_ask_', 11, 700, RED, 'end')
            if p == best_bid: b += T(30, y + 29, 'best_bid_ →', 11, 700, GT)
        b += T(540, 392, 'oldest order is next to the price column (the level head)', 11, 400, MUTE)
        return b
    def inc(q, note=''): return B(30, 400, 250, 44, f'incoming: buy id 9 @ 102, qty left {q}', YEL, RED, 2, 12) + T(300, 427, note, 12, 600, RED)
    A0 = {101: [('S1', 40), ('S2', 20)], 103: [('S3', 50)]}; B0 = {99: [('B1', 30)]}
    fr = [
        ('Before: asks 101 = [S1 40, S2 20] (S1 older), 103 = [S3 50]; bids 99 = [B1 30].', ladder(A0, B0, 101, 99)),
        ('A buy 70 @ 102 arrives. best_ask_ = 101 ≤ 102, so it can trade.', ladder(A0, B0, 101, 99) + inc(70)),
        ('maker = level head = S1 (the oldest). x = min(70, 40) = 40. Trade 40 @ 101.', ladder(A0, B0, 101, 99, ('S1',)) + inc(30, 'filled 40')),
        ('S1 is at 0: erase_order(S1) unlinks it. The head of 101 is now S2.', ladder({101: [('S2', 20)], 103: [('S3', 50)]}, B0, 101, 99) + inc(30, 'filled 40')),
        ('maker = S2. x = min(30, 20) = 20. Trade 20 @ 101. S2 is erased; level 101 is empty.', ladder({101: [('S2', 20)], 103: [('S3', 50)]}, B0, 101, 99, ('S2',)) + inc(10, 'filled 60')),
        ('raise_best_ask scans up: 101 empty, 102 empty, 103 has S3. best_ask_ = 103 > 102: stop.', ladder({103: [('S3', 50)]}, B0, 103, 99) + inc(10, 'filled 60')),
        ('rest: the remaining 10 goes to the tail of bid level 102. best_bid_ = 102. submit returns 60.', ladder({103: [('S3', 50)]}, {99: [('B1', 30)], 102: [('#9', 10)]}, 103, 102, ('#9',))),
        ('cancel(9): the id table finds the node, erase_order unlinks it; lower_best_bid scans 102 → 99.', ladder({103: [('S3', 50)]}, B0, 103, 99)),
    ]
    save('d13m-match.svg', W, H, 'Price-time priority: buy 70 @ 102, then cancel the remainder', 'best price first; at one price the oldest order first; trades execute at the resting order\'s price', '', fr, 4.2)

# ───────────────────────── 6. unlink ─────────────────────────
def unlink():
    W, H = 960, 400
    def node(x, name, prev, nxt, hot=(), gone=False):
        f = GREY if gone else '#ffd8d3'
        b = R(x, 150, 190, 96, f, MUTE, 1.2, 6, '5 4' if gone else None) + T(x + 95, 174, name + ('  (back in the pool)' if gone else ''), 13, 700, MUTE if gone else INK, 'middle')
        b += T(x + 16, 204, f'prev: {prev}', 13, 600, RED if 'p' in hot else INK) + T(x + 16, 228, f'next: {nxt}', 13, 600, RED if 'n' in hot else INK)
        return b
    def lvl(head, tail, hot=()):
        return (R(330, 76, 300, 44, YEL, MUTE) + T(346, 103, 'Level 101', 12, 700) + T(440, 103, f'head: {head}', 13, 600, RED if 'h' in hot else INK) + T(540, 103, f'tail: {tail}', 13, 600, RED if 't' in hot else INK))
    xs = [70, 385, 700]
    fr = [
        ('A level with three orders. head = oldest (S1), tail = newest (S3). The links live inside each Order.',
         lvl('S1', 'S3') + node(xs[0], 'S1', 'null', 'S2') + node(xs[1], 'S2', 'S1', 'S3') + node(xs[2], 'S3', 'S2', 'null')),
        ('erase_order(S2), line 84: S2->prev exists, so S2->prev->next = S2->next. S1.next becomes S3.',
         lvl('S1', 'S3') + node(xs[0], 'S1', 'null', 'S3', 'n') + node(xs[1], 'S2', 'S1', 'S3') + node(xs[2], 'S3', 'S2', 'null')),
        ('line 85: S2->next exists, so S2->next->prev = S2->prev. S3.prev becomes S1. S2 is out of the list.',
         lvl('S1', 'S3') + node(xs[0], 'S1', 'null', 'S3') + node(xs[1], 'S2', 'S1', 'S3', gone=True) + node(xs[2], 'S3', 'S1', 'null', 'p')),
        ('lines 86–87: index_.erase(S2.id), pool_.release(S2). Two pointer writes, no search, no free().',
         lvl('S1', 'S3') + node(xs[0], 'S1', 'null', 'S3') + node(xs[2], 'S3', 'S1', 'null')),
        ('erase_order(S1), line 84: S1->prev is null, so the else branch runs: lvl.head = S1->next = S3.',
         lvl('S3', 'S3', 'h') + node(xs[0], 'S1', 'null', 'S3', gone=True) + node(xs[2], 'S3', 'S1', 'null')),
        ('line 85: S1->next exists, so S3.prev = S1->prev = null. One order left: head = tail = S3.',
         lvl('S3', 'S3') + node(xs[2], 'S3', 'null', 'null', 'p')),
    ]
    save('d13m-unlink.svg', W, H, 'erase_order: unlinking from an intrusive list', 'removing the middle order, then the head order; each needs two pointer fixes', '', fr, 4.2)

# ───────────────────────── 7. id table ─────────────────────────
def idtable():
    W, H = 960, 430
    homes = {'A': 2, 'B': 2, 'C': 3, 'D': 5}
    def tab(keys, y=110, hole=None, hot=None, probe=(), label=''):
        b = T(60, y - 12, label, 12, 700, MUTE)
        for i in range(8):
            x = 60 + i * 104
            k = keys[i]
            b += R(x, y, 96, 58, YEL if k else GREY, RED if i == hot or i == hole else MUTE, 2.6 if i == hot or i == hole else 1.2, 4, '5 4' if i == hole else None)
            b += T(x + 48, y + 30, k or ('hole' if i == hole else 'empty'), 16 if k else 11, 700 if k else 400, INK if k else MUTE, 'middle')
            if k: b += T(x + 48, y + 48, f'home {homes[k]}', 10, 400, MUTE, 'middle')
            b += T(x + 48, y + 76, f'slot {i}', 10, 400, MUTE, 'middle')
            if i in probe: b += A(x + 48, y - 30, x + 48, y - 4, RED)
        return b
    K0 = [None, None, 'A', 'B', 'C', 'D', None, None]
    note = lambda s, y=250: T(60, y, s, 13, 600, INK)
    fr = [
        ('Linear probing. A and B both hash to slot 2: B sits one further. C (home 3) was pushed to 4.', tab(K0) + note('find(B): start at home 2 → not B → slot 3 → found.')),
        ('find(C): start at home 3 (holds B) → slot 4 → found. A lookup stops at the first EMPTY slot.', tab(K0, probe=(3, 4))),
        ('The wrong way to erase A: just empty slot 2. Now find(B) starts at 2, sees empty, and stops.', tab([None, None, None, 'B', 'C', 'D', None, None], hole=2, probe=(2,)) + T(60, 250, 'B is still in the table but can no longer be found ✗', 13, 700, RED)),
        ('Backward shift. Hole i = 2. Look at j = 3 (B): distance from home 3 − 2 = 1 ≥ distance to hole 1 → move.', tab([None, None, None, 'B', 'C', 'D', None, None], hole=2, hot=3) + note('(j − home) & mask ≥ (j − i) & mask   →   1 ≥ 1 ✓')),
        ('B is now at its home. Hole i = 3. j = 4 (C, home 3): 4 − 3 = 1 ≥ 4 − 3 = 1 → move.', tab([None, None, 'B', None, 'C', 'D', None, None], hole=3, hot=4) + note('1 ≥ 1 ✓')),
        ('Hole i = 4. j = 5 (D, home 5): distance from home 0 ≥ 1? No. D stays: moving it would put it before its home.', tab([None, None, 'B', 'C', None, 'D', None, None], hole=4, hot=5) + note('0 ≥ 1 ✗')),
        ('j = 6 is empty: the run ends. Slot 4 is set to empty. No tombstone is left behind.', tab([None, None, 'B', 'C', None, 'D', None, None]) + note('find(B): slot 2 ✓   find(C): slot 3 ✓   find(D): slot 5 ✓   find(A): slot 2, 3, 4 → empty → not found ✓')),
    ]
    save('d13m-idtable.svg', W, H, 'IdTable::erase(A): backward-shift deletion', 'open addressing with linear probing, 8 slots; each key shows its home slot = mix(id) & mask', '', fr, 4.6)

# ───────────────────────── 8. pool ─────────────────────────
def pool():
    W, H = 960, 380
    def draw(free, top, out, hot=None, note=''):
        b = T(60, 92, 'storage_: 4 Orders, allocated once', 12, 700, MUTE)
        for i in range(4):
            used = f'N{i}' in out
            b += B(60 + i * 96, 104, 88, 50, f'N{i}', '#ffd8d3' if used else GREY, sub='in the book' if used else 'free')
        b += T(520, 92, 'free_: a stack of pointers', 12, 700, MUTE)
        for i in range(4):
            x = 520 + i * 96
            live = i < top
            b += R(x, 104, 88, 50, YEL if live else WHITE, RED if hot == i else MUTE, 2.6 if hot == i else 1.2, 4, None if live else '5 4')
            b += T(x + 44, 134, free[i] if live else (free[i] + ' (stale)' if free[i] else ''), 14 if live else 10, 700 if live else 400, INK if live else MUTE, 'middle')
            b += T(x + 44, 172, f'free_[{i}]', 10, 400, MUTE, 'middle')
        tx = 520 + top * 96
        b += A(tx, 226, tx, 186, BT) + T(tx, 244, f'top_ = {top}', 13, 700, BT, 'middle')
        b += T(60, 250, note, 13, 600, INK)
        return b
    fr = [
        ('Constructor: every node is free. top_ = 4 counts the free ones.', draw(['N0', 'N1', 'N2', 'N3'], 4, [])),
        ('acquire(): return free_[--top_] = free_[3] = N3. top_ = 3. No call to new.', draw(['N0', 'N1', 'N2', 'N3'], 3, ['N3'], 3, 'rest() fills N3 with the order')),
        ('acquire() again: free_[2] = N2. top_ = 2.', draw(['N0', 'N1', 'N2', 'N3'], 2, ['N3', 'N2'], 2)),
        ('The order in N3 is filled or cancelled. release(N3): free_[top_++] = N3. top_ = 3.', draw(['N0', 'N1', 'N3', 'N3'], 3, ['N2'], 2, 'N3 lands on top of the stack')),
        ('The next acquire() returns N3 again: last in, first out. That memory is still in the CPU cache.', draw(['N0', 'N1', 'N3', 'N3'], 2, ['N2', 'N3'], 2)),
        ('With top_ = 0, acquire() returns nullptr and rest() refuses the order. It never allocates.', draw(['N0', 'N1', 'N3', 'N3'], 0, ['N0', 'N1', 'N2', 'N3'], None, 'pool empty → the order\'s qty is counted as expired')),
    ]
    save('d13m-pool.svg', W, H, 'ObjectPool: a free stack of pre-allocated Orders', 'acquire = free_[--top_]   release = free_[top_++] = p', '', fr, 4.0)

# ───────────────────────── 9. wal append ─────────────────────────
def wal():
    W, H = 960, 420
    def f(count, recs, hot=None, ghost=None, book='', seen=None, crash=None):
        b = T(60, 92, 'the mmap\'d segment file', 12, 700, MUTE)
        b += B(60, 104, 120, 56, f'count = {count}', BLUE, RED if hot == 'count' else MUTE, 2.6 if hot == 'count' else 1.2, 14)
        for i in range(5):
            x = 200 + i * 104
            if i < len(recs):
                g = (ghost == i)
                b += R(x, 104, 96, 56, '#ffe8a3' if g else YEL, RED if hot == i else MUTE, 2.6 if hot == i else 1.2, 4, '5 4' if g else None)
                b += T(x + 48, 130, recs[i], 14, 700, INK, 'middle') + T(x + 48, 148, 'not counted yet' if g else '40 B', 10, 400, RED if g else MUTE, 'middle')
            else:
                b += R(x, 104, 96, 56, WHITE, MUTE, 1, 4, '5 4')
            b += T(x + 48, 178, f'records_[{i}]', 10, 400, MUTE, 'middle')
        b += B(60, 220, 330, 50, f'a reader (follower, or recover) sees {seen} records', GREEN, size=12)
        b += B(430, 220, 290, 50, book, PINK, size=12)
        if crash: b += T(60, 304, crash, 13, 700, RED)
        return b
    fr = [
        ('The segment holds 2 complete records. Command c arrives in apply().', f(2, ['r0', 'r1'], seen=2, book='book: has r0, r1')),
        ('append line 77: n = count = 2. Line 78: 2 < max_records_, so there is room.', f(2, ['r0', 'r1'], hot='count', seen=2, book='book: has r0, r1')),
        ('line 79: records_[2] = c. The 40 bytes are in the file, but count is still 2.', f(2, ['r0', 'r1', 'c'], hot=2, ghost=2, seen=2, book='book: has r0, r1', crash='crash here (point A): the reader sees 2 records. c never happened. Consistent.')),
        ('line 80: count = 3, stored with release. The record becomes visible, whole.', f(3, ['r0', 'r1', 'c'], hot='count', seen=3, book='book: has r0, r1', crash='crash here (point B): logged but not applied. Recovery replays c. ✓')),
        ('Only now ex_.apply(c) changes the book.', f(3, ['r0', 'r1', 'c'], seen=3, book='book: has r0, r1, c', crash='crash here (point C): logged and applied. Replay rebuilds the same book. ✓')),
    ]
    save('d13m-wal.svg', W, H, 'Write-ahead: copy the record, publish the count, then apply', 'WalLogger::append (wal_logger.hpp:76–82) followed by ex_.apply (matching_engine.hpp:26)', '', fr, 4.6)

# ───────────────────────── 10. recover ─────────────────────────
def recover():
    W, H = 960, 430
    x0, sc = 80, 0.04  # px per command, origin at seq 48,000
    X = lambda s: x0 + (s - 48000) * sc
    def axis():
        b = L(X(48000), 300, X(68500), 300, MUTE)
        for s in (50000, 55000, 60000, 65000, 67000):
            b += L(X(s), 296, X(s), 304, MUTE) + T(X(s), 320, f'{s:,}', 10, 400, MUTE, 'middle')
        b += T(X(68500) + 6, 304, 'seq', 10, 400, MUTE)
        return b
    def seg(y, a, e, name, done_to=None, skip_to=None):
        b = R(X(a), y, (e - a) * sc, 40, GREY, MUTE) + (T(X(e), y - 6, name, 11, 600, MUTE, 'end') if a >= 65000 else T(X(a) + 6, y - 6, name, 11, 600, MUTE))
        if skip_to: b += R(X(a), y, (skip_to - a) * sc, 40, '#d0d7de', MUTE) + T(X(a) + (skip_to - a) * sc / 2, y + 25, 'in the snapshot: skip', 10, 600, MUTE, 'middle')
        if done_to: b += R(X(skip_to or a), y, (done_to - (skip_to or a)) * sc, 40, GREEN, GT, 2) + T((X(skip_to or a) + X(done_to)) / 2, y + 25, 'replay', 11, 700, GT, 'middle')
        return b
    snap = lambda: R(X(48000), 90, (60000 - 48000) * sc, 34, BLUE, MUTE) + T(X(48000) + 8, 112, 'snapshot: the book after 60,000 commands', 11, 600)
    cur = lambda s: L(X(s), 80, X(s), 300, RED, 2, '6 4') + T(X(s), 76, f'r.seq = {s:,}', 12, 700, RED, 'middle')
    SA = lambda **k: seg(150, 50000, 65000, 'segment A: base 50,000, 15,000 records', **k)
    SB = lambda **k: seg(220, 65000, 67000, 'segment B: base 65,000, 2,000 records', **k)
    st = axis()
    fr = [
        ('After a crash the book is gone. On disk: one snapshot and two log segments.', snap() + SA() + SB()),
        ('Step 1: load the snapshot. The book now contains commands 0 … 59,999. r.seq = 60,000.', snap() + SA() + SB() + cur(60000)),
        ('Segment A: end 65,000 > 60,000 and base 50,000 ≤ 60,000. Skip indices 0 … 9,999.', snap() + SA(skip_to=60000) + SB() + cur(60000)),
        ('Replay A from index 60,000 − 50,000 = 10,000 to 14,999: 5,000 commands through ex.apply. r.seq = 65,000.', snap() + SA(skip_to=60000, done_to=65000) + SB() + cur(65000)),
        ('Segment B: base 65,000 ≤ 65,000. Replay all 2,000. r.seq = 67,000: the book is exactly as before the crash.', snap() + SA(skip_to=60000, done_to=65000) + SB(done_to=67000) + cur(67000)),
        ('If B started at 66,000 instead, commands 65,000–65,999 would be missing: base > r.seq → ok = false. Refuse.',
         snap() + SA(skip_to=60000, done_to=65000) + seg(220, 66000, 67000, 'segment B: base 66,000') + R(X(65000), 220, 1000 * sc, 40, '#ffd8d3', RED, 2, 4, '5 4') + T(X(65500), 276, 'gap', 11, 700, RED, 'middle') + cur(65000)),
    ]
    save('d13m-recover.svg', W, H, 'recover: snapshot + replay, and refusing a gap', 'Journal::recover (journal.hpp:234); the book is deterministic, so replaying the same commands rebuilds it', st, fr, 4.6)

# ───────────────────────── 11. segments ─────────────────────────
def segments():
    W, H = 960, 420
    col = {'Active': YEL, 'Sealed': '#ffd8d3', 'Free': GREY}
    def f(s0, s1, fill0, fill1, m, fo, hot=None, snap=None):
        b = ''
        for i, (s, fl) in enumerate(((s0, fill0), (s1, fill1))):
            x = 80 + i * 300
            b += R(x, 150, 250, 60, WHITE, RED if hot == i else MUTE, 2.6 if hot == i else 1.2)
            b += R(x, 150, 250 * fl, 60, col[s], MUTE, 1) + T(x + 125, 186, f'segment {i}: {s}', 13, 700, INK, 'middle')
        mx = 80 + m * 300 + 125
        b += B(mx - 90, 84, 180, 34, 'matching thread appends', BLUE, size=11) + A(mx, 118, mx, 148, BT)
        if fo is not None:
            fx = 80 + fo * 300 + 125
            b += B(fx - 90, 250, 180, 34, 'follower replays', GREEN, size=11) + A(fx, 250, fx, 212, GT)
        b += B(700, 150, 220, 60, 'replica book', PINK, sub=snap or 'kept by the follower')
        return b
    fr = [
        ('Matching appends to segment 0 (Active). The follower trails it, replaying into its own replica book.', f('Active', 'Free', 0.55, 0, 0, 0)),
        ('Segment 0 fills up. append returns false.', f('Active', 'Free', 1.0, 0, 0, 0, hot=0)),
        ('Rotate, done by MATCHING: segment 0 Active → Sealed, segment 1 Free → Active. Matching goes on at once.', f('Sealed', 'Active', 1.0, 0.1, 1, 0, hot=1)),
        ('The follower finishes replaying segment 0 and snapshots its REPLICA. Matching copies nothing.', f('Sealed', 'Active', 1.0, 0.35, 1, 0, snap='snapshot written (tmp → fsync → rename)')),
        ('Done by the FOLLOWER: the snapshot covers segment 0, so Sealed → Free. It now follows segment 1.', f('Free', 'Active', 0, 0.6, 1, 1, hot=0)),
        ('Segment 1 fills. Matching seals it and takes segment 0 (Free → Active). Each state change has one owner.', f('Active', 'Sealed', 0.1, 1.0, 0, 1, hot=0)),
    ]
    save('d13m-segments.svg', W, H, 'Two log segments, one follower: who changes which state', 'Active → Sealed (matching)   Sealed → Free (follower)   Free → Active (matching)', '', fr, 4.6)

for fn in (pipeline, ring, ordering, falseshare, match, unlink, idtable, pool, wal, recover, segments): fn()
