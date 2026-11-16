"""Step animations for the Day 9 interview edition (CS728 retrieval heads, LeNet-5 compression).
Run: python3 d09figs.py <output dir>   (then python3 make_players.py)
Same look and helpers as mxfigs.py / mxfigs2.py. Toy numbers are the ones in the note; real numbers come from
LongContext_RAG (bfb2ec9), interview_materials/lenet (3b2584b) and its fml_report.pdf."""
from mxfigs import *
from mxfigs2 import C, dbg
import mxfigs as _mx
import math

def T(x, y, s, size=12, w=400, fill=INK, anchor='start'):
    return _mx.T(x, y, str(s), size, w, fill, anchor)

ORANGE = '#ffd8b5'
TOK = ['<s>', 'weather', 'api', 'flight', 'search', 'stock', 'price', 'calc', 'math', 'fly', 'Paris']
TOOLCOL = {1: BLUE, 2: BLUE, 3: GREEN, 4: GREEN, 5: YEL, 6: YEL, 7: PINK, 8: PINK}
SPANS = {'A': (1, 3), 'B': (3, 5), 'C': (5, 7), 'D': (7, 9)}

def tokens(x0, y, cw=72, hl=None, lab=True):
    o = ''
    for i, t in enumerate(TOK):
        f = TOOLCOL.get(i, GREY)
        s, sw = (RED, 2.4) if hl is not None and i == hl else (MUTE, 1)
        o += R(x0 + i * cw, y, cw - 4, 30, f, s, sw, 4) + T(x0 + i * cw + (cw - 4) / 2, y + 20, t, 11, 600, INK, 'middle')
        if lab: o += T(x0 + i * cw + (cw - 4) / 2, y + 46, str(i), 10, 400, MUTE, 'middle')
    return o

# ───────────────────────── CS728 1: one attention row ─────────────────────────
def attn_row():
    W, H, x0, cw = 960, 470, 70, 76
    sc = [2.2, 0.2, 0.1, 3.0, 1.9, 0.0, 0.3, 0.1, 0.0, 1.0, 1.2]
    ex = [math.exp(s) for s in sc]; Z = sum(ex); wt = [e / Z for e in ex]
    st = tokens(x0, 74, cw)
    st += T(x0 + 1 * cw, 140, 'tool A', 11, 700, BT) + T(x0 + 3 * cw, 140, 'tool B', 11, 700, GT) + T(x0 + 5 * cw, 140, 'tool C', 11, 700, '#9a6700') + T(x0 + 7 * cw, 140, 'tool D', 11, 700, '#bf3989') + T(x0 + 9 * cw, 140, 'query', 11, 700, MUTE)
    def row(y, label, vals, fmt, hot=None):
        o = T(20, y + 18, label, 11, 700, MUTE)
        for i, v in enumerate(vals):
            o += R(x0 + i * cw, y, cw - 4, 26, WHITE, RED if hot == i else '#d0d7de', 2.2 if hot == i else 1, 4)
            o += T(x0 + i * cw + (cw - 4) / 2, y + 18, fmt(v), 12, 700 if hot == i else 400, RED if hot == i else INK, 'middle')
        return o
    def bars(hot=None):
        o = T(20, 330, 'weight', 11, 700, MUTE)
        for i, v in enumerate(wt):
            h = v * 260
            o += R(x0 + i * cw + 14, 380 - h, cw - 32, h, RED if hot == i else BT, 'none', 0, 2)
            o += T(x0 + i * cw + (cw - 4) / 2, 374 - h, f'{v:.3f}', 11, 700, RED if hot == i else INK, 'middle')
        return o
    s1 = row(160, 'score', sc, lambda v: f'{v:.1f}')
    s2 = s1 + row(196, 'e^score', ex, lambda v: f'{v:.2f}')
    fr = [
        ('The prompt: 4 tools (2 tokens each), then the query "fly Paris". The model is about to write its answer.', ''),
        ('Score per position = (query vector · key vector) / √d. One number per earlier token, for this one head.', s1),
        ('Softmax, step 1: e^score makes every number positive and stretches the gaps (3.0 → 20.09, 0.0 → 1).', s2),
        (f'Softmax, step 2: divide each by the total Z = {Z:.2f}. The weights now sum to 1: where this head "looks".', s2 + bars()),
        ('argmax = position 3, "flight", inside tool B. That index is the only thing the copy rule and Hit Count read.', s2 + bars(3)),
        ('Note <s> (position 0) gets 0.186: the "attention sink". It is not a tool, so retrieval code must ignore it.', s2 + bars(0)),
        ('Every head of every layer has its own row like this at each decode step (32 × 32 = 1,024 rows in Llama-3.1-8B).', s2 + bars(3)),
    ]
    save('d09i-attn-row.svg', W, H, 'One head, one decode step: scores → softmax → where it looks', 'toy prompt of 11 tokens; real rows are thousands of tokens long', st, fr, 4.2)

# ───────────────────────── CS728 2: copy_hits, debugger view ─────────────────────────
def copy_dbg():
    code = [(88, 'attn = self.capture.stack()   # [layers, heads, keys]'),
            (89, 'nxt = int(out.logits[0, -1].argmax())'),
            (93, 'if nxt in needle_tokens:'),
            (94, '    copyable_steps += 1'),
            (96, 'j, hit = copy_hits(attn, prompt_ids, nxt, ns, ne)'),
            (97, 'hit_steps += hit.to(torch.int32)'),
            (98, 'rel = (j - ns).clamp(0, klen - 1)'),
            (99, 'covered |= zeros_like(covered).scatter_(2, rel, hit)'),
            (138, '  j = attn.argmax(dim=-1)'),
            (139, '  in_span = (j >= needle_start) & (j < needle_end)'),
            (142, '  tok_at_j = prompt_ids[j.clamp(max=len - 1)]'),
            (143, '  return j, in_span & (tok_at_j == generated_token)'),
            (106, 'scores = covered.sum(-1).float() / klen')]
    V = ['nxt (token emitted)', 'copyable_steps', 'j = argmax  [L0H0 L0H1 L1H0 L1H1]', 'in_span', 'tok_at_j', 'hit', 'hit_steps', 'rel (needle-relative)', 'covered (needle positions)', 'scores']
    steps = [
        ((88,), {'nxt (token emitted)': '–', 'copyable_steps': '0', 'hit_steps': '[0 0 0 0]', 'covered (needle positions)': 'none yet'},
         'Toy: prompt ids [11 12 13 14 15 16 47 21 17 18], needle = positions 4–7 ("code is 47 21"). 2 layers × 2 heads.'),
        ((89,), {'nxt (token emitted)': '47'}, 'Line 89: greedy decoding picks token 47 ("47"). Every head\'s attention row for this step is already in attn.'),
        ((93, 94), {'copyable_steps': '1'}, 'Lines 93–94: 47 is one of the needle\'s tokens, so this step is a copy opportunity (denominator of copy-precision).'),
        ((96, 138), {'j = argmax  [L0H0 L0H1 L1H0 L1H1]': '[0 6 7 6]'}, 'Line 138: argmax over keys, for all 4 heads at once. Each head\'s answer to "where do I look most?"'),
        ((139,), {'in_span': '[F T T T]'}, 'Line 139: is that position inside the needle (4 ≤ j < 8)? L0H0 looks at position 0: no.'),
        ((142,), {'tok_at_j': '[11 47 21 47]'}, 'Line 142: which token sits there? Gather by index. clamp only guards positions past the prompt.'),
        ((143,), {'hit': '[F T F T]'}, 'Line 143: a hit needs BOTH: inside the needle AND the same token as the one emitted. L1H0 looks at "21": no.'),
        ((97,), {'hit_steps': '[0 1 0 1]'}, 'Line 97: count hits per head (booleans → int32, then add).'),
        ((98, 99), {'rel (needle-relative)': '[0 2 3 2]', 'covered (needle positions)': 'L0H1 {6}, L1H1 {6}'}, 'Lines 98–99: mark WHICH needle position each hitting head copied. |= keeps old marks.'),
        ((89, 93, 94), {'nxt (token emitted)': '21', 'copyable_steps': '2'}, 'Next decode step: the model emits 21, also a needle token.'),
        ((138, 139), {'j = argmax  [L0H0 L0H1 L1H0 L1H1]': '[9 7 7 6]', 'in_span': '[F T T T]'}, 'New argmaxes. L0H0 looks at position 9, outside the needle.'),
        ((142, 143), {'tok_at_j': '[18 21 21 47]', 'hit': '[F T T F]'}, 'L1H1 still looks at "47", but "21" is being emitted: not a copy this time.'),
        ((97, 98, 99), {'hit_steps': '[0 2 1 1]', 'rel (needle-relative)': '[3 3 3 2]', 'covered (needle positions)': 'L0H1 {6,7}, L1H0 {7}, L1H1 {6}'}, 'Counts and coverage updated. The model then emits end-of-text and the loop stops.'),
        ((106,), {'scores': '[0  0.5  0.25  0.25]'}, 'Line 106: score = copied needle positions ÷ needle length 4. Max possible here is 0.5 (only 2 tokens copied).'),
    ]
    dbg('d09i-copy-dbg.svg', 'The copy rule in your code, one line at a time', 'LongContext_RAG heads/scorer.py:88–106 and copy_hits :138–143 (bfb2ec9); toy prompt and 4 heads', code, steps, V, codew=540, sec=4.4)

# ───────────────────────── CS728 3: Hit Count selection ─────────────────────────
def hitcount():
    W, H = 960, 470
    heads = ['L0H0', 'L0H1', 'L0H2', 'L1H0', 'L1H1', 'L1H2']
    am = {'L0H0': [0, 0, 0, 0, 0], 'L0H1': [9, 1, 9, 9, 9], 'L0H2': [3, 9, 5, 0, 9], 'L1H0': [3, 2, 6, 7, 4], 'L1H1': [7, 9, 9, 8, 0], 'L1H2': [4, 1, 5, 9, 3]}
    probes = [('P1 "fly to Paris"', 'B'), ('P2 "rain tomorrow?"', 'A'), ('P3 "AAPL today"', 'C'), ('P4 "17 × 23"', 'D'), ('P5 "cheap tickets"', 'B')]
    st = T(20, 82, 'probe queries (correct tool known)', 12, 700, MUTE)
    for k, (q, t) in enumerate(probes):
        st += T(20, 108 + k * 26, f'{q} → tool {t}', 12)
    st += T(20, 262, 'tool spans: A = 1–2, B = 3–4, C = 5–6, D = 7–8', 11, 400, MUTE)
    st += T(330, 82, 'per head: argmax position on each probe, and the hit counter', 12, 700, MUTE)
    def board(upto, sel=None):
        o = ''
        cnt = {h: 0 for h in heads}
        for i, h in enumerate(heads):
            x, y = 330 + (i % 3) * 205, 96 + (i // 3) * 150
            marks = ''
            for k in range(upto):
                j = am[h][k]; s, e = SPANS[probes[k][1]]
                ok = s <= j < e; cnt[h] += ok
                marks += B(x + 8 + k * 37, y + 40, 33, 26, str(j), GREEN if ok else GREY, GT if ok else MUTE, 1.2, 12)
            chosen = sel and h in sel
            o += R(x, y, 198, 130, YEL if chosen else WHITE, RED if chosen else '#d0d7de', 2.4 if chosen else 1, 8)
            o += T(x + 10, y + 24, h, 13, 700) + marks
            o += T(x + 10, y + 108, 'hits', 11, 600, MUTE) + T(x + 186, y + 110, str(cnt[h]), 22, 700, RED if chosen else INK, 'end')
        if upto:
            o += R(18, 94 + (upto - 1) * 26, 290, 22, 'none', RED, 2, 4)
        return o
    fr = [('Hit rule for CS728 (the tool version of the copy rule): a head hits if its argmax lands inside the CORRECT tool\'s span.', board(0))]
    cap = ['P1 wants B (3–4). L0H2, L1H0 look at 3 and L1H2 at 4: three hits. L0H0 looks at <s>, L0H1 and L1H1 at the query.',
           'P2 wants A (1–2). L0H1 (1), L1H0 (2), L1H2 (1) hit.',
           'P3 wants C (5–6). L0H2 (5), L1H0 (6), L1H2 (5) hit.',
           'P4 wants D (7–8). L1H0 (7) and L1H1 (8) hit; L1H2 looks at the query (9).',
           'P5 wants B again. Only L1H0 (4) and L1H2 (3) hit.']
    for k in range(5): fr.append((cap[k], board(k + 1)))
    fr.append(('Sort by hits: L1H0 = 5, L1H2 = 4, L0H2 = 2, … Keep the top k (k = 2 here; your project kept the top 10).', board(5, {'L1H0', 'L1H2'})))
    fr.append(('L0H0 never hits: it always looks at <s> (the sink). Counting hits filters such heads out without any training.', board(5, {'L1H0', 'L1H2'})))
    save('d09i-hitcount.svg', W, H, 'Hit Count: probe, count, keep the top heads', 'toy: 6 heads, 5 probes, the 11-token prompt of the attention-row figure', st, fr, 4.4)

# ───────────────────────── CS728 4: heads as a retriever (score_all) ─────────────────────────
def score_dbg():
    code = [(102, 'attn, spans, _ = self.attention_for(question, chunks)'),
            (103, 'selected = attn[self._l_idx, self._h_idx].float()'),
            (107, 'seg = torch.full((T,), -1, dtype=torch.long)'),
            (109, 'for i, (s, e) in enumerate(spans):'),
            (111, '    seg[s:e] = i'),
            (114, 'valid = seg >= 0'),
            (115, 'seg_v = seg[valid]'),
            (116, 'p = selected[:, valid]'),
            (117, 'p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)'),
            (120, 'sums = zeros(K, n_chunks).index_add_(1, seg_v, p)'),
            (129, 'out[key] = mat[:k].mean(dim=0)')]
    V = ['spans (A B C D)', 'selected[L1H0]  (11 keys)', 'selected[L1H2]', 'seg', 'seg_v', 'p[L1H0]  (8 tool tokens)', 'p[L1H2]', 'sums[L1H0]  A B C D', 'sums[L1H2]  A B C D', 'score = mean over heads', 'ranking']
    steps = [
        ((102,), {'spans (A B C D)': '(1,3) (3,5) (5,7) (7,9)'}, 'New query "flight to Rome" (correct tool B). One decode step gives attn [layers, heads, 11] and each tool\'s token span.'),
        ((103,), {'selected[L1H0]  (11 keys)': '.40 .03 .02 .25 .15 .02 .03 .02 .02 .03 .03', 'selected[L1H2]': '.30 .10 .08 .12 .10 .04 .04 .06 .06 .05 .05'}, 'Line 103: keep only the heads Hit Count chose (L1H0, L1H2): fancy indexing → shape [K = 2, T = 11].'),
        ((107, 109, 111), {'seg': '-1 0 0 1 1 2 2 3 3 -1 -1'}, 'Lines 107–111: label every position with its tool number; -1 = not a tool (<s>, the query).'),
        ((114, 115), {'seg_v': '0 0 1 1 2 2 3 3'}, 'Lines 114–115: keep tool positions only. The sink (.40 and .30!) and the query drop out here.'),
        ((116, 117), {'p[L1H0]  (8 tool tokens)': '.056 .037 .463 .278 .037 .056 .037 .037', 'p[L1H2]': '.167 .133 .200 .167 .067 .067 .100 .100'}, 'Line 117: renormalise each head over tool tokens (÷ .54 and ÷ .60): "share of this head\'s attention that went to tools".'),
        ((120,), {'sums[L1H0]  A B C D': '.093 .741 .093 .074', 'sums[L1H2]  A B C D': '.300 .367 .133 .200'}, 'Line 120: index_add_ adds each token\'s share into its tool\'s bucket: one number per (head, tool).'),
        ((129,), {'score = mean over heads': 'A .196  B .554  C .113  D .137'}, 'Line 129: average the K heads, so each head counts equally (a diffuse head can\'t dominate).'),
        ((129,), {'ranking': 'B > A > D > C   → B is rank 1 ✓'}, 'Rank tools by score. The correct tool B is first: this query counts as a Recall@1 hit.'),
    ]
    dbg('d09i-score-dbg.svg', 'Heads as a retriever: attention mass per tool, then rank', 'LongContext_RAG compress/head_scorer.py:102–129 (bfb2ec9), the same reduction CS728 needs; toy numbers', code, steps, V, codew=500, sec=4.6)

# ───────────────────────── CS728 5: lost in the middle (schematic) ─────────────────────────
def lost_middle():
    W, H = 960, 450
    x0, x1, y0, y1 = 110, 640, 340, 90
    def X(p): return x0 + (x1 - x0) * (p - 1) / 19
    def Y(a): return y0 - (y0 - y1) * a / 100
    st = L(x0, y0, x1, y0, INK, 1.2) + L(x0, y0, x0, y1, INK, 1.2)
    st += T((x0 + x1) / 2, y0 + 36, 'position of the correct tool in a list of 20', 12, 600, MUTE, 'middle')
    st += T(x0, y1 - 14, 'accuracy', 12, 600, MUTE)
    for p in (1, 5, 10, 15, 20): st += T(X(p), y0 + 18, str(p), 11, 400, MUTE, 'middle')
    for a in (0, 50, 100): st += T(x0 - 8, Y(a) + 4, f'{a}%', 11, 400, MUTE, 'end')
    U = [75 - 22 * math.sin(math.pi * (p - 1) / 19) - (3 * (p - 1) / 19) for p in range(1, 21)]
    F = [62 - 4 * math.sin(math.pi * (p - 1) / 19) for p in range(1, 21)]
    def curve(vals, col, dash=None):
        pts = ' '.join(f'{X(p)},{Y(v)}' for p, v in zip(range(1, 21), vals))
        d = f" stroke-dasharray='{dash}'" if dash else ''
        return f"<polyline points='{pts}' fill='none' stroke='{col}' stroke-width='3'{d}/>"
    def tools(pos):
        o = T(680, 84, 'the prompt', 12, 700, MUTE)
        for k in range(20):
            y = 94 + k * 12
            o += R(680, y, 120, 10, GREEN if k + 1 == pos else GREY, RED if k + 1 == pos else '#d0d7de', 1.6 if k + 1 == pos else 0.6, 2)
        o += T(810, 94 + (pos - 1) * 12 + 9, '← correct tool', 11, 700, GT)
        o += B(680, 344, 260, 30, 'query + answer come after', WHITE, MUTE, 1, 11)
        return o
    dot = lambda p, v, c: f"<circle cx='{X(p)}' cy='{Y(v)}' r='7' fill='{c}' stroke='white' stroke-width='2'/>"
    fr = [
        ('Setup of Liu et al. (2023): the same question, with the one useful item moved from first to last place in a long list.', tools(1)),
        ('Start of the list: models do well (primacy: early tokens get extra attention).', tools(1) + curve(U[:1] + U[:1], BT) + dot(1, U[0], BT)),
        ('Middle: accuracy sags. Some models then do worse than with no list at all.', tools(10) + curve(U[:10], BT) + dot(10, U[9], BT)),
        ('End: it recovers (recency: the last items sit next to the question). The U shape is "lost in the middle".', tools(20) + curve(U, BT) + dot(20, U[-1], BT) + T(X(10), Y(U[9]) + 22, 'plain model: U-shaped (shape from Liu et al.)', 11, 700, BT, 'middle')),
        ('Why: a lot of attention is POSITIONAL (sink at the start, recency at the end). It is not about what the item says.', tools(20) + curve(U, BT)),
        ('Heads picked because they land on the RIGHT item by content should depend less on position: a flatter curve.', tools(10) + curve(U, BT) + curve(F, GT, '7 5') + T(X(10), Y(F[9]) - 12, 'selected heads: flatter (schematic, not your numbers)', 11, 700, GT, 'middle')),
        ('Your claim needs your own per-position Recall@1 (not on the server). Without it, say "aimed to mitigate", not "fixed".', tools(10) + curve(U, BT) + curve(F, GT, '7 5')),
    ]
    save('d09i-lost-middle.svg', W, H, '"Lost in the middle", and why content-driven heads should help', 'schematic: curve shapes only, no measured values', st, fr, 4.4)

# ───────────────────────── LeNet 1: convolution ─────────────────────────
def conv():
    W, H = 960, 420
    img = [[0, 0, 1, 0, 0] for _ in range(5)]
    ker = [[-1, 0, 1] for _ in range(3)]
    out = [[3, 0, -3] for _ in range(3)]
    cs = 44
    def g(x, y, M, hl=None, fill=WHITE, tag=''):
        o = T(x, y - 10, tag, 12, 700)
        for i, r in enumerate(M):
            for j, v in enumerate(r):
                f = WHITE if v in ('', None) else (BLUE if (fill == WHITE and v == 1) else fill)
                s, sw = ((RED, 2.4) if hl and (i, j) in hl else ('#8c959f', 1))
                o += R(x + j * cs, y + i * cs, cs - 2, cs - 2, f, s, sw, 3)
                if v not in ('', None): o += T(x + j * cs + cs / 2 - 1, y + i * cs + cs / 2 + 5, v, 14, 600, INK, 'middle')
        return o
    st = g(40, 90, img, tag='image (5 × 5): a vertical stroke') + g(330, 134, ker, fill=YEL, tag='kernel (3 × 3), 9 weights')
    st += T(300, 200, '⊛', 22, 700, MUTE, 'middle') + T(510, 200, '=', 22, 700, MUTE, 'middle')
    def frame(k):
        o = ''
        if isinstance(k, int):
            r, c = divmod(k, 3)
            o += f"<rect x='{40 + c * cs - 3}' y='{90 + r * cs - 3}' width='{3 * cs + 4}' height='{3 * cs + 4}' fill='none' stroke='{RED}' stroke-width='3' rx='4'/>"
        if k == 'all': shown = out
        else: shown = [[out[i][j] if (k is not None and i * 3 + j <= k) else '' for j in range(3)] for i in range(3)]
        hl = {divmod(k, 3)} if isinstance(k, int) else None
        o += g(540, 134, shown, hl=hl, fill=GREEN, tag='output map (3 × 3)')
        return o
    calc = ['(−1·0 + 0·0 + 1·1) × 3 rows = 3: the window\'s right column sits on the stroke.',
            '(−1·0 + 0·1 + 1·0) × 3 = 0: the stroke is under the kernel\'s middle (weight 0).',
            '(−1·1 + 0·0 + 1·0) × 3 = −3: the stroke is under the left column (weight −1).']
    fr = [('A convolution slides a small grid of weights (the kernel) over the image. Each stop gives one output number.', frame(None))]
    for k in range(3): fr.append(('Window at column ' + str(k) + ': multiply 9 pairs and add. ' + calc[k], frame(k)))
    fr.append(('The next rows give the same numbers. The map says "dark-to-light edge here" (+3) and "light-to-dark" (−3).', frame('all')))
    fr.append(('The SAME 9 weights are used at every position. That sharing is why conv1 needs only 6 filters × 5 × 5 = 150 weights.', frame('all') + B(40, 330, 880, 32, 'LeNet conv1: 6 kernels of 5 × 5 on a 32 × 32 image → 6 maps of 28 × 28 (32 − 5 + 1 = 28). Learned, not hand-set.', ORANGE, MUTE, 1, 12)))
    save('d09i-conv.svg', W, H, 'What a convolution computes', 'toy 5 × 5 image, one 3 × 3 vertical-edge kernel, no padding, stride 1', st, fr, 4.2)

# ───────────────────────── LeNet 2: shapes and parameters ─────────────────────────
def lenet_shapes():
    W, H = 960, 500
    stages = [('input', '1 × 28 × 28', 0, 'raw pixels in [0, 1]'),
              ('normalize', '1 × 32 × 32', 0, 'pad 2 on each side, (x − 0.1307) / 0.3081'),
              ('conv1 + tanh', '6 × 28 × 28', 156, '6 filters × (1 × 5 × 5) = 150 weights + 6 biases'),
              ('pool2 (avg)', '6 × 14 × 14', 0, 'average each 2 × 2 block: halves height and width'),
              ('conv3 + tanh', '16 × 10 × 10', 2416, '16 filters × (6 × 5 × 5): every filter sees all 6 maps ("dense C3")'),
              ('pool4 (avg)', '16 × 5 × 5', 0, 'halves again'),
              ('conv5 + tanh', '120 × 1 × 1', 48120, '120 filters × (16 × 5 × 5) = 48,000 weights: 78% of the model'),
              ('flatten', '120', 0, 'drop the 1 × 1 spatial part'),
              ('fc6 + tanh', '84', 10164, '120 × 84 = 10,080 weights + 84 biases'),
              ('fc7', '10', 850, '84 × 10 = 840 weights + 10 biases → 10 logits, one per digit')]
    total = 61706
    def body(k):
        o = ''
        for i, (n, sh, p, why) in enumerate(stages[:k + 1]):
            y = 70 + i * 32
            hot = i == k
            o += B(20, y, 150, 26, n, YEL if hot else (BLUE if p else GREY), RED if hot else MUTE, 2 if hot else 1, 12)
            o += T(185, y + 18, sh, 12, 700 if hot else 400, RED if hot else INK)
            if p:
                w = 300 * p / 48120
                o += R(330, y + 4, max(w, 2), 18, GREEN, 'none', 0, 2) + T(338 + max(w, 2), y + 18, f'{p:,} params', 11, 600)
        n, sh, p, why = stages[k]
        o += B(20, 402, 920, 30, f'{n}: {why}', '#f6f8fa', MUTE, 1, 12)
        return o
    fr = [(f'Stage {k + 1}/{len(stages)}: {stages[k][0]} → {stages[k][1]}', body(k)) for k in range(len(stages))]
    fr.append(('Total 61,706 parameters (61,470 weights + 236 biases) × 4 bytes = 241.0 KiB; the saved file is 245.6 KiB.', body(len(stages) - 1)))
    save('d09i-lenet-shapes.svg', W, H, 'LeNet-5 in your repo, layer by layer (shapes checked by running it)', 'src/models.py:58–79 (3b2584b); one image, batch dimension left out; bars = parameters per layer', '', fr, 3.8)

# ───────────────────────── LeNet 3: IMP rounds, layer-wise ─────────────────────────
def imp_rounds():
    W, H = 960, 470
    Wt = {'conv1': 150, 'conv3': 2400, 'conv5': 48000, 'fc6': 10080, 'fc7': 840}
    cur = dict(Wt); hist = {0: dict(cur)}
    for r in range(1, 21):
        for k in cur: cur[k] = round(cur[k] * (0.9 if k == 'fc7' else 0.8))
        hist[r] = dict(cur)
    def body(r, glob=False):
        o = T(20, 78, f'round {r}', 18, 700, RED)
        tot = sum(hist[r].values())
        o += T(160, 78, f'{tot:,} of 61,470 weights left = {100 * tot / 61470:.2f}%', 14, 700)
        for i, k in enumerate(Wt):
            y = 104 + i * 54
            frac = hist[r][k] / Wt[k]
            o += T(20, y + 22, k, 13, 700) + R(100, y, 600, 30, GREY, 'none', 0, 4)
            o += R(100, y, max(600 * frac, 2), 30, PINK if k == 'conv1' else BLUE, 'none', 0, 4)
            o += T(710, y + 20, f'{hist[r][k]:,} / {Wt[k]:,}  ({100 * frac:.1f}%)', 12, 700 if k == 'conv1' else 400, RED if k == 'conv1' else INK)
        if glob:
            o += B(100, 384, 830, 34, 'GLOBAL pruning, same round 20 (1.15% overall): conv1 keeps 26.7% (≈ 40 of 150); a later conv layer drops to 0.3%', YEL, RED, 2, 12)
        return o
    cap = {0: 'Start: a trained dense LeNet-5. Each round removes 20% of each layer\'s SURVIVING weights (fc7: 10%, half rate).',
           1: 'Round 1: 80% left everywhere, 90% in fc7. Then rewind the survivors and retrain 20 epochs with the mask fixed.',
           3: 'Round 3: 0.8³ = 51.2%. Clean accuracy is still flat: the lottery-ticket result reproduces.',
           5: 'Round 5: about one third left.',
           9: 'Round 9: 13.8% overall. conv1 is down to 21 weights for 6 filters of 25. (The PGD +30 points was measured here.)',
           12: 'Round 12: 7.2%. conv1 has 11 weights.',
           15: 'Round 15: 3.75%, your headline model. conv1 keeps 6 of 150; 3 of its 6 filters are entirely empty.',
           20: 'Round 20: 1.30%. conv1 keeps 2 weights; 4 of 6 filters empty. Layer-wise = the same fraction everywhere.'}
    fr = [(cap[r], body(r)) for r in (0, 1, 3, 5, 9, 12, 15, 20)]
    fr.append(('Global pruning ranks all weights together. conv1\'s weights are large, so it survives; the cost moves elsewhere.', body(20, True)))
    save('d09i-imp-rounds.svg', W, H, 'Iterative magnitude pruning, layer-wise: who is left after each round', 'counts computed from the schedule (20%/round, fc7 10%/round); they match the report\'s 13.8%, 3.75%, 1.30% and conv1 21/6/2', '', fr, 4.4)

# ───────────────────────── LeNet 4: prune_step, debugger view ─────────────────────────
def prune_dbg():
    code = [(1, 'def prune_step(model, scope, rate=0.20):'),
            (3, '  if scope == "layerwise":'),
            (4, '    for layer in prunable_layers(model):'),
            (5, '      threshold = magnitude_quantile(layer.weight, rate)'),
            (6, '      layer.mask &= (layer.weight.abs() >= threshold)'),
            (7, '  elif scope == "global":'),
            (8, '    all_weights = concat_magnitudes(model)'),
            (9, '    threshold = magnitude_quantile(all_weights, rate)'),
            (10, '    for layer in prunable_layers(model):'),
            (11, '      layer.mask &= (layer.weight.abs() >= threshold)'),
            (12, '  apply_masks(model)')]
    V = ['scope, rate', 'layer 1 |w|  (4 weights)', 'layer 2 |w|  (12 weights)', 'threshold', 'layer 1 mask', 'layer 2 mask', 'kept per layer']
    l1 = '.9 .8 .7 .1'; l2 = '.3 .05 .2 .02 .25 .15 .01 .4 .12 .06 .03 .35'
    steps = [
        ((1,), {'scope, rate': '"layerwise", 0.5', 'layer 1 |w|  (4 weights)': l1, 'layer 2 |w|  (12 weights)': l2, 'layer 1 mask': '1 1 1 1', 'layer 2 mask': 'all 1'}, 'Toy: a small layer with big weights and a big layer with small ones. rate 0.5 so one round is visible (yours: 0.2).'),
        ((3, 4, 5), {'threshold': 'median of layer 1 = 0.75'}, 'Line 5: a threshold PER LAYER. Half of layer 1\'s magnitudes lie below 0.75.'),
        ((6,), {'layer 1 mask': '1 1 0 0', 'kept per layer': 'L1: 2 of 4'}, 'Line 6: keep |w| ≥ 0.75, so .9 and .8. The .7 is dropped although it beats every weight in layer 2.'),
        ((4, 5), {'threshold': 'median of layer 2 = 0.135'}, 'Next layer, its own threshold: 0.135.'),
        ((6,), {'layer 2 mask': 'keeps .3 .2 .25 .15 .4 .35', 'kept per layer': 'L1: 2 of 4,  L2: 6 of 12'}, 'Layer-wise: exactly half of EACH layer survives, whatever the weights look like.'),
        ((1, 7, 8), {'scope, rate': '"global", 0.5', 'layer 1 mask': '1 1 1 1', 'layer 2 mask': 'all 1', 'kept per layer': '–'}, 'Same weights, scope "global": line 8 puts all 16 magnitudes in one list.'),
        ((9,), {'threshold': 'median of all 16 = 0.175'}, 'Line 9: ONE threshold for the whole network.'),
        ((10, 11), {'layer 1 mask': '1 1 1 0', 'layer 2 mask': 'keeps .3 .2 .25 .4 .35', 'kept per layer': 'L1: 3 of 4,  L2: 5 of 12'}, 'Line 11: the big-weight layer keeps 3, the small-weight layer only 5. Same total (8), different split.'),
        ((12,), {}, 'Line 12: weight ← weight × mask. &= on line 6/11 means a pruned weight stays pruned in later rounds.'),
    ]
    dbg('d09i-prune-dbg.svg', 'One pruning round, layer-wise then global', 'the report\'s Listing 1 (fml_report.pdf §3.1, pseudo-code); quantile helper assumed linear like torch.quantile', code, steps, V, codew=500, sec=4.6)

# ───────────────────────── LeNet 5: Welch's t-test ─────────────────────────
def welch():
    W, H = 960, 450
    lo, hi, x0, x1 = 64, 78, 120, 840
    def X(v): return x0 + (x1 - x0) * (v - lo) / (hi - lo)
    st = L(x0, 300, x1, 300, INK, 1.2)
    for v in range(lo, hi + 1, 2): st += L(X(v), 296, X(v), 304, INK, 1) + T(X(v), 320, str(v), 11, 400, MUTE, 'middle')
    st += T(x0, 340, 'MNIST-C mean accuracy (%), 5 seeds each: dot = mean, bar = ± 1 standard deviation', 11, 400, MUTE)
    def eb(y, m, s, col, lab):
        return L(X(m - s), y, X(m + s), y, col, 3) + L(X(m - s), y - 7, X(m - s), y + 7, col, 2) + L(X(m + s), y - 7, X(m + s), y + 7, col, 2) + f"<circle cx='{X(m)}' cy='{y}' r='6' fill='{col}'/>" + T(X(m + s) + 10, y + 4, f'{lab}: {m} ± {s}', 12, 700, col)
    D = eb(120, 73.72, 0.92, BT, 'dense'); G = eb(170, 73.59, 0.98, GT, 'global 3.52%'); Lw = eb(220, 69.83, 2.41, RED, 'layer-wise 3.75%')
    def box(t): return B(40, 360, 880, 34, t, '#f6f8fa', MUTE, 1, 12)
    fr = [
        ('Question: does pruning change corruption accuracy, or is the gap just seed noise? 5 training seeds per model.', D + G + Lw),
        ('Global vs dense. Difference of means = 73.59 − 73.72 = −0.13 points. Is that bigger than the noise?', D + G + box('difference d = −0.13')),
        ('Noise of a mean of 5 numbers = s² / n. Dense: 0.92² / 5 = 0.169. Global: 0.98² / 5 = 0.192.', D + G + box('var of each mean: 0.169 and 0.192')),
        ('Standard error of the difference = √(0.169 + 0.192) = 0.601. Welch does NOT assume the two spreads are equal.', D + G + box('SE = 0.601')),
        ('t = d / SE = −0.13 / 0.601 = −0.22. The difference is a fifth of its own noise.', D + G + box('t = −0.22')),
        ('Degrees of freedom (Welch–Satterthwaite) = (0.169 + 0.192)² / (0.169²/4 + 0.192²/4) = 7.97.', D + G + box('df = 7.97')),
        ('p = chance of |t| ≥ 0.22 if there were truly no difference = 0.83. "No evidence of a difference", NOT "proved equal".', D + G + box('p = 0.83  (the report\'s number, recomputed from its own table)')),
        ('Layer-wise vs dense: d = −3.89, SE = √(2.41²/5 + 0.92²/5) = 1.15, t = −3.37, df = 5.1 → p = 0.019. A real drop.', D + Lw + box('p = 0.019')),
        ('Global vs layer-wise head to head: d = +3.76, SE = 1.16, t = 3.23, df = 5.3 → p = 0.021 (the "3.76 pp gain").', G + Lw + box('p = 0.021;  59 such tests in the report, Holm-corrected within each family')),
    ]
    save('d09i-welch.svg', W, H, "Welch's t-test on your MNIST-C numbers", 'means and standard deviations from fml_report.pdf Table 2; every p-value recomputed with scipy', st, fr, 4.6)

# ───────────────────────── LeNet 6: PGD, debugger view ─────────────────────────
def pgd_dbg():
    code = [(62, 'model.eval()'),
            (64, 'x_orig = x.clone().detach()'),
            (65, 'if random_start:'),
            (66, '    adv = x_orig + empty_like(x_orig).uniform_(-eps, eps)'),
            (67, '    adv = adv.clamp(0.0, 1.0)'),
            (68, 'else:'),
            (69, '    adv = x_orig.clone()'),
            (71, 'for _ in range(steps):'),
            (72, '    grad = _grad_wrt_input(model, adv, y, loss_fn)'),
            (73, '    adv = adv.detach() + alpha * grad.sign()'),
            (75, '    adv = x_orig + (adv - x_orig).clamp(-eps, eps)'),
            (76, '    adv = adv.clamp(0.0, 1.0)'),
            (79, 'return adv.detach()')]
    V = ['x_orig', 'adv', 'score s = 2·x1 − x2', 'loss = log(1 + e^(−s))', 'grad = dloss/dadv', 'grad.sign()', 'adv − x_orig', 'prediction']
    steps = [
        ((62,), {'x_orig': '–', 'adv': '–', 'prediction': '–'}, 'Toy "model": logits [s, 0] with s = 2·x1 − x2; class 0 wins while s > 0. True label y = 0. eps 0.15, alpha 0.04.'),
        ((64,), {'x_orig': '(0.30, 0.20)', 'score s = 2·x1 − x2': '0.40', 'loss = log(1 + e^(−s))': '0.513', 'prediction': '0 ✓'}, 'Line 64: keep an untouched copy of the clean input. Every projection measures distance from it.'),
        ((65, 68, 69), {'adv': '(0.30, 0.20)'}, 'Lines 65–69: your default is a RANDOM start inside the box (line 66). Off here, so the trace is reproducible.'),
        ((71, 72), {'grad = dloss/dadv': '(−0.803, +0.401)'}, 'Line 72: gradient of the loss w.r.t. the INPUT (weights frozen). Lowering x1 and raising x2 raises the loss.'),
        ((73,), {'grad.sign()': '(−1, +1)', 'adv': '(0.26, 0.24)'}, 'Line 73: one step of alpha = 0.04 in the SIGN direction: every coordinate moves the full step.'),
        ((75, 76), {'adv − x_orig': '(−0.04, +0.04)  ok', 'score s = 2·x1 − x2': '0.28', 'loss = log(1 + e^(−s))': '0.563'}, 'Lines 75–76: project. Still inside the ±0.15 box and inside [0, 1], so nothing changes yet.'),
        ((72, 73, 75), {'adv': '(0.22, 0.28)', 'adv − x_orig': '(−0.08, +0.08)', 'score s = 2·x1 − x2': '0.16', 'loss = log(1 + e^(−s))': '0.616', 'grad = dloss/dadv': '(−0.861, +0.430)'}, 'Step 2. The gradient grows as the loss rises, but the sign (and so the step) stays the same.'),
        ((72, 73, 75), {'adv': '(0.18, 0.32)', 'adv − x_orig': '(−0.12, +0.12)', 'score s = 2·x1 − x2': '0.04', 'loss = log(1 + e^(−s))': '0.673', 'grad = dloss/dadv': '(−0.920, +0.460)'}, 'Step 3. Almost at the boundary s = 0.'),
        ((73,), {'adv': '(0.14, 0.36)  before projecting', 'grad = dloss/dadv': '(−0.980, +0.490)'}, 'Step 4, line 73: the raw step leaves the box: x1 moved 0.16 > eps.'),
        ((75,), {'adv − x_orig': '(−0.15, +0.15)  clipped', 'adv': '(0.15, 0.35)', 'score s = 2·x1 − x2': '−0.05', 'loss = log(1 + e^(−s))': '0.719', 'prediction': '1 ✗ fooled'}, 'Line 75: clip the CHANGE to ±eps and add it back. s = −0.05 < 0: the prediction flips. Attack succeeded.'),
        ((79,), {}, 'With eps = 0.10 the box stops it at (0.20, 0.30), s = 0.10: still correct. Robust accuracy = share of images that survive.'),
    ]
    dbg('d09i-pgd-dbg.svg', 'pgd() in your repo, one line at a time', 'interview_materials/lenet src/attacks.py:62–79 (3b2584b); 2-pixel toy model', code, steps, V, codew=500, sec=4.6)

# ───────────────────────── LeNet 7: where the 75.0 KB go ─────────────────────────
def int8_bytes():
    W, H = 960, 420
    sc = 2.9
    def bar(y, parts, lab):
        o = T(20, y + 22, lab, 13, 700)
        x = 160
        for v, col, t in parts:
            w = v * sc
            o += R(x, y, w, 34, col, 'white', 1.5, 3)
            if w > 34: o += T(x + w / 2, y + 22, t, 11, 600, INK, 'middle')
            x += w
        return o + T(x + 8, y + 22, f'{sum(p[0] for p in parts):.1f} KiB', 13, 700)
    fp = bar(80, [(240.12, BLUE, 'weights fp32 240.1'), (0.92, GREEN, ''), (4.53, GREY, '')], 'FP32 file')
    hyp = bar(150, [(61.4, ORANGE, '÷ 4 = 61.4')], 'if all ÷ 4')
    i1 = [(60.03, BLUE, 'int8 weights 60.0')]
    i2 = i1 + [(3.69, YEL, '')]
    i3 = i2 + [(1.0, GREEN, '')]
    i4 = i3 + [(10.31, GREY, 'container 10.3')]
    def note(t): return B(40, 330, 880, 34, t, '#f6f8fa', MUTE, 1, 12)
    fr = [
        ('FP32 checkpoint: 61,706 numbers × 4 B = 241.0 KiB (weights 240.1 + biases 0.9) + 4.5 KiB of file container = 245.6.', fp),
        ('If every byte shrank 4×, INT8 would be 61.4 KiB. It is 75.0. Where do the other 13.6 KiB come from?', fp + hyp),
        ('Weights: 61,470 int8 numbers × 1 B = 60.0 KiB. This part really is 4× smaller.', fp + bar(220, i1, 'INT8 file')),
        ('Per-channel scales and zero points: 236 output channels × (8 B scale + 8 B zero point) = 3.7 KiB.', fp + bar(220, i2, 'INT8 file') + note('yellow = scales + zero points (one pair per output channel)')),
        ('Biases stay float32, plus the normalisation buffers and the per-layer output scales: about 1.0 KiB.', fp + bar(220, i3, 'INT8 file') + note('green = biases, buffers, activation scales')),
        ('File container: a separate record for every tensor, and the INT8 model has many more small tensors: 10.3 KiB.', fp + bar(220, i4, 'INT8 file') + note('245.6 / 75.0 = 3.27× (the report rounds to 3.28×). Weights alone: 4×.')),
        ('Pruning does NOT shrink the FP32 file: the zeros are still stored (245.6 KiB at 3.75% weights). Only a sparse format would.', fp + bar(220, i4, 'INT8 file') + note('Breakdown from a stand-in INT8 LeNet-5 (fbgemm, per-channel QAT), whose state_dict is 75.03 KiB')),
    ]
    save('d09i-int8-bytes.svg', W, H, 'Why INT8 gives 3.27×, not 4×', 'KiB = 1,024 bytes (the report\'s "KB"); stand-in converted with torch.ao.quantization, not your branch', '', fr, 4.6)

if __name__ == '__main__':
    for fn in (attn_row, copy_dbg, hitcount, score_dbg, lost_middle, conv, lenet_shapes, imp_rounds, prune_dbg, welch, pgd_dbg, int8_bytes): fn()
