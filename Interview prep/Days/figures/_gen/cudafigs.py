"""Step animations for the CUDA interview notes (Day 05 / Day 06 interview editions).
Run: python3 cudafigs.py <output dir>   (then python3 make_players.py)
Same look and helpers as mxfigs.py / mxfigs2.py. Every number comes from the Day 5 / Day 6 notes or the repo."""
from mxfigs import *
from mxfigs2 import C, dbg
import math
import mxfigs as _mx
def T(x, y, s, size=12, w=400, fill=INK, anchor='start'):
    return _mx.T(x, y, str(s).replace('  ', '\u2003 '), size, w, fill, anchor)

ORANGE = '#ffd8b5'
SC = [BLUE, GREEN, YEL, PINK]            # stream / warp colours

A4 = [[1, 2, 0, 1], [0, 1, 2, 1], [2, 0, 1, 0], [1, 1, 0, 2]]
B4 = [[1, 0, 2, 1], [2, 1, 0, 0], [0, 1, 1, 2], [1, 2, 0, 1]]
C4 = [[sum(A4[i][k] * B4[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
assert C4[0][3] == 2 and C4[1][3] == 5

def grid(x, y, M, cs, fill=GREY, hl=None, tag=None, size=12, stroke='#ffffff'):
    """draw matrix M (list of lists, '' allowed) with cell size cs. hl: {(i,j): (fill, stroke)}"""
    o = ''
    for i, row in enumerate(M):
        for j, v in enumerate(row):
            f, s, sw = fill, stroke, 1
            if hl and (i, j) in hl:
                f, s = hl[(i, j)]; sw = 2.4
            o += R(x + j * cs, y + i * cs, cs - 2, cs - 2, f, s, sw, 3)
            o += T(x + j * cs + (cs - 2) / 2, y + i * cs + cs / 2 + 4, v, size, 600, INK, 'middle')
    if tag: o += T(x, y - 8, tag, 12, 700)
    return o

# ═════════════════════════ DAY 5 ═════════════════════════

def loadmap():
    """mm_tiled_1to1<2>, toy N = 4, block (bx=1, by=0): which thread loads which element"""
    W, H = 960, 470; cs = 40
    ax, sax, bx_, sbx, cx, y0 = 40, 222, 360, 542, 700, 120
    bxI, byI, TILE = 1, 0, 2
    st = T(ax, 100, 'A (global, row-major)', 12, 700) + T(bx_, 100, 'B (global)', 12, 700)
    st += T(sax, 100, 'sA (shared)', 12, 700, BT) + T(sbx, 100, 'sB (shared)', 12, 700, GT) + T(cx, 100, 'C (block owns dashed 2×2)', 12, 700)
    st += grid(cx, y0, [[str(v) for v in r] for r in C4], cs, GREY)
    st += R(cx + 2 * cs - 3, y0 - 3, 2 * cs + 2, 2 * cs + 2, 'none', RED, 2, 4, '5 4')
    fr = []
    order = [(0, 0), (1, 0), (0, 1), (1, 1)]       # (tx, ty), flat id = ty*2 + tx
    for t in range(2):
        sA = [['', ''], ['', '']]; sB = [['', ''], ['', '']]
        for n, (tx, ty) in enumerate(order):
            row, col = byI * TILE + ty, bxI * TILE + tx
            aCol, bRow = t * TILE + tx, t * TILE + ty
            sA[ty][tx] = str(A4[row][aCol]); sB[ty][tx] = str(B4[bRow][col])
            hlA = {(r, c): (BLUE, '#ffffff') for r in range(2) for c in range(t * 2, t * 2 + 2)}
            hlB = {(r, c): (GREEN, '#ffffff') for r in range(t * 2, t * 2 + 2) for c in range(2, 4)}
            hlA[(row, aCol)] = (YEL, RED); hlB[(bRow, col)] = (YEL, RED)
            b = grid(ax, y0, [[str(v) for v in r] for r in A4], cs, GREY, hlA)
            b += grid(bx_, y0, [[str(v) for v in r] for r in B4], cs, GREY, hlB)
            b += grid(sax, y0, sA, cs + 10, '#f6f8fa', {(ty, tx): (YEL, RED)}, size=14, stroke='#d0d7de')
            b += grid(sbx, y0, sB, cs + 10, '#f6f8fa', {(ty, tx): (YEL, RED)}, size=14, stroke='#d0d7de')
            b += A(ax + aCol * cs + 19, y0 + row * cs + 19, sax + tx * 50 + 24, y0 + ty * 50 + 24, RED, 1.8)
            b += A(bx_ + col * cs + 19, y0 + bRow * cs + 19, sbx + tx * 50 + 24, y0 + ty * 50 + 24, RED, 1.8)
            info = [f'step t = {t}, thread (tx = {tx}, ty = {ty})   [{n + 1} of the block\'s 4 threads]',
                    f'row = by·TILE + ty = 0·2 + {ty} = {row}      col = bx·TILE + tx = 1·2 + {tx} = {col}',
                    f'aCol = t·TILE + tx = {aCol}      bRow = t·TILE + ty = {bRow}',
                    f'line 91: sA[{ty}][{tx}] = A[{row}][{aCol}] = {A4[row][aCol]}      line 92: sB[{ty}][{tx}] = B[{bRow}][{col}] = {B4[bRow][col]}']
            for k, s in enumerate(info):
                b += T(40, 330 + k * 22, s, 12 if k else 13, 700 if k == 0 else 400, RED if k == 0 else INK)
            fr.append((f't = {t}: thread ({tx},{ty}) copies ONE element of A and ONE of B. Blue/green = the tiles this step needs.', b))
        # after barrier
        b = grid(ax, y0, [[str(v) for v in r] for r in A4], cs, GREY, {(r, c): (BLUE, '#ffffff') for r in range(2) for c in range(t * 2, t * 2 + 2)})
        b += grid(bx_, y0, [[str(v) for v in r] for r in B4], cs, GREY, {(r, c): (GREEN, '#ffffff') for r in range(t * 2, t * 2 + 2) for c in range(2, 4)})
        b += grid(sax, y0, sA, cs + 10, BLUE, {(0, 0): (YEL, RED), (0, 1): (YEL, RED)}, size=14)
        b += grid(sbx, y0, sB, cs + 10, GREEN, {(0, 1): (YEL, RED), (1, 1): (YEL, RED)}, size=14)
        acc = [1, 2][t]
        b += T(40, 330, f'line 93 __syncthreads(): all 4 cells of both tiles are written. Now thread (1,0) reads row 0 of sA and column 1 of sB:', 13, 700, RED)
        if t == 0:
            b += T(40, 354, 'sA[0][0] = 1 was loaded by thread (0,0); sB[1][1] = 0 by thread (1,1). Without the barrier it could read them before they exist.', 12)
            b += T(40, 376, 'k = 0: 1·1 = 1;  k = 1: 2·0 = 0  →  acc = 1.   8 HBM loads served 8 FMAs (2 per thread): each loaded float is used TILE = 2 times.', 12)
        else:
            b += T(40, 354, 'k = 0: sA[0][0]·sB[0][1] = 0·2 = 0;  k = 1: sA[0][1]·sB[1][1] = 1·1 = 1  →  acc = 1 + 1 = 2 = C[0][3] ✓', 12)
            b += T(40, 376, 'Whole block: 2 steps × 8 loads = 16 HBM floats for 4 outputs = 4 per output. Naive: 2N = 8 per output. Formula 2N / TILE = 4 ✓', 12)
        fr.append((f'After the barrier of step t = {t}: every thread reads cells that OTHER threads loaded. That is why cooperation needs a barrier.', b))
    save('d05i-loadmap.svg', W, H, 'mm_tiled_1to1<2>: which thread loads which element', 'toy N = 4, TILE = 2, block (bx = 1, by = 0), 2 × 2 threads; matmul_bench.cu:83–97', st, fr, 4.0)

def tiled_dbg():
    code = [(83, 'const int tx = threadIdx.x, ty = threadIdx.y;'),
            (84, 'const int row = blockIdx.y * TILE + ty;'),
            (85, 'const int col = blockIdx.x * TILE + tx;'),
            (87, 'float acc = 0.0f;'),
            (88, 'for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {'),
            (89, '  int aCol = t * TILE + tx;'),
            (90, '  int bRow = t * TILE + ty;'),
            (91, '  sA[ty][tx] = (row<N && aCol<N) ? A[row*N+aCol] : 0;'),
            (92, '  sB[ty][tx] = (bRow<N && col<N) ? B[bRow*N+col] : 0;'),
            (93, '  __syncthreads();'),
            (96, '  for (int k = 0; k < TILE; ++k)'),
            (97, '    acc = fmaf(sA[ty][k], sB[k][tx], acc);'),
            (98, '  __syncthreads();'),
            (99, '}'),
            (100, 'if (row < N && col < N) C[row*N + col] = acc;')]
    V = ['tx, ty', 'row', 'col', 't  (trips = (4+1)/2 = 2)', 'aCol, bRow', 'sA (whole block)', 'sB (whole block)', 'k', 'acc', 'C[0][3] in HBM']
    steps = [
        ((83,), {'tx, ty': '1, 0', 'row': '–', 'col': '–', 't  (trips = (4+1)/2 = 2)': '–', 'aCol, bRow': '–', 'sA (whole block)': '? ? / ? ?', 'sB (whole block)': '? ? / ? ?', 'k': '–', 'acc': '–', 'C[0][3] in HBM': 'not written'},
         'Thread (tx = 1, ty = 0) of block (bx = 1, by = 0), toy N = 4, TILE = 2. tx, ty live in registers.'),
        ((84,), {'row': '0·2 + 0 = 0'}, 'Line 84: the output row it owns. by·TILE jumps to the block\'s first row; ty picks the row inside the block.'),
        ((85,), {'col': '1·2 + 1 = 3'}, 'Line 85: col from tx, so neighbouring lanes own neighbouring columns: the store and the B loads coalesce.'),
        ((87,), {'acc': '0'}, 'Line 87: one float in a register. It never touches memory until line 100.'),
        ((88,), {'t  (trips = (4+1)/2 = 2)': '0'}, 'Line 88: walk along K one tile at a time. (N + TILE − 1)/TILE rounds up, so a ragged last tile is still visited.'),
        ((89, 90), {'aCol, bRow': '1, 0'}, 'Lines 89–90: this thread fetches column t·2 + tx of A\'s tile and row t·2 + ty of B\'s tile.'),
        ((91,), {'sA (whole block)': '? 2 / ? ?'}, 'Line 91: sA[0][1] = A[0][1] = 2. The other 3 threads fill the other 3 cells at the same time.'),
        ((92,), {'sB (whole block)': '? 1 / ? ?'}, 'Line 92: sB[0][1] = B[0][3] = 1. Out-of-range cells would get 0.0f (adds nothing to acc).'),
        ((93,), {'sA (whole block)': '1 2 / 0 1', 'sB (whole block)': '2 1 / 0 0'}, 'Line 93: barrier. Nobody passes until all 4 threads wrote. Now both tiles are complete.'),
        ((96, 97), {'k': '0', 'acc': '1·1 + 0 = 1'}, 'k = 0: sA[0][0] (loaded by thread 0,0) × sB[0][1] (own) = 1·1. acc = 1.'),
        ((96, 97), {'k': '1', 'acc': '2·0 + 1 = 1'}, 'k = 1: sA[0][1] × sB[1][1] (loaded by thread 1,1) = 2·0. acc stays 1.'),
        ((98,), {}, 'Line 98: second barrier. A fast thread may not overwrite sA/sB with tile 1 while a slow one still reads tile 0.'),
        ((88, 89, 90), {'t  (trips = (4+1)/2 = 2)': '1', 'aCol, bRow': '3, 2'}, 't = 1: aCol = 1·2 + 1 = 3, bRow = 1·2 + 0 = 2.'),
        ((91, 92), {'sA (whole block)': '? 1 / ? ?', 'sB (whole block)': '? 2 / ? ?'}, 'Lines 91–92: sA[0][1] = A[0][3] = 1, sB[0][1] = B[2][3] = 2 (overwriting tile 0\'s values).'),
        ((93,), {'sA (whole block)': '0 1 / 2 1', 'sB (whole block)': '1 2 / 0 1'}, 'Barrier: tile 1 complete: A rows 0–1 × cols 2–3, B rows 2–3 × cols 2–3.'),
        ((96, 97), {'k': '0', 'acc': '0·2 + 1 = 1'}, 'k = 0: 0·2 = 0, acc = 1.'),
        ((96, 97), {'k': '1', 'acc': '1·1 + 1 = 2'}, 'k = 1: 1·1 = 1, acc = 2. Same four products as the naive trace; only WHERE they were read from changed.'),
        ((98, 99), {'t  (trips = (4+1)/2 = 2)': '2 → exit'}, 'Barrier, then t = 2 is not < 2: the loop ends.'),
        ((100,), {'C[0][3] in HBM': '2 ✓'}, 'Line 100: guarded store. C[0·4 + 3] = 2. This thread read 4 floats from HBM (not 8) for its output.'),
    ]
    dbg('d05i-tiled-dbg.svg', 'mm_tiled_1to1<2>, one thread, one line at a time', 'toy N = 4, TILE = 2, block (1, 0), thread (tx 1, ty 0) → C[0][3]; matmul_bench.cu:83–100 (casts shortened)', code, steps, V, codew=560, sec=4.0)

def race():
    W, H = 960, 440
    x0, x1 = 150, 600
    def X(t): return x0 + (x1 - x0) * t / 10
    st = T(40, 104, 'warp 0', 13, 700) + T(40, 164, 'warp 1 (slow)', 13, 700)
    st += T(660, 88, 'shared tile sA (2 halves)', 12, 700)
    st += B(40, 380 - 70, 100, 22, 'load', BLUE, size=11) + B(150, 380 - 70, 100, 22, 'wait', GREY, size=11) + B(260, 380 - 70, 100, 22, 'compute', GREEN, size=11)
    def lanes(segs0, segs1, bars, now, tile, verdict, good):
        b = ''
        for y, segs in ((86, segs0), (146, segs1)):
            for ph, a, c in segs:
                col = {'load': BLUE, 'wait': GREY, 'comp': GREEN, 'bad': PINK}[ph]
                b += R(X(a), y, X(c) - X(a), 30, col, RED if ph == 'bad' else MUTE, 2 if ph == 'bad' else 1, 3)
                b += T((X(a) + X(c)) / 2, y + 20, {'load': 'load t', 'wait': 'wait', 'comp': 'compute t', 'bad': 'reads wrong data'}[ph] if not (ph == 'load' and a > 5) else 'load t+1', 10, 600, INK, 'middle')
        for t in bars:
            b += L(X(t), 74, X(t), 190, RED, 2, '5 3') + T(X(t), 206, '__syncthreads', 10, 700, RED, 'middle')
        b += L(X(now), 70, X(now), 196, BT, 1.5) + T(X(now), 66, 'now', 10, 700, BT, 'middle')
        for k, (lab, f) in enumerate(tile):
            b += B(660, 100 + k * 50, 260, 40, lab, f, size=12)
        b += B(40, 236, 880, 40, verdict, GREEN if good else PINK, GT if good else RED, 2, 12)
        return b
    T0 = ('rows from warp 0: tile t', BLUE); T1 = ('rows from warp 1: tile t', BLUE)
    OLD = ('rows from warp 1: OLD (tile t−1)', PINK); NEW0 = ('rows from warp 0: tile t+1', YEL)
    fr = [
        ('Both barriers present. Each warp loads its half of tile t; warp 1 is slower (its loads missed in cache).',
         lanes([('load', 0, 2)], [('load', 0, 3.2)], [], 2, [T0, ('rows from warp 1: loading…', GREY)], 'Each warp loads only its own cells: cooperation means nobody has the whole tile alone.', True)),
        ('Warp 0 reaches barrier 1 (line 93) and WAITS. It cannot start the inner loop on a half-filled tile.',
         lanes([('load', 0, 2), ('wait', 2, 3.2)], [('load', 0, 3.2)], [3.2], 3.2, [T0, T1], 'Barrier 1 = "everyone has WRITTEN the tile". Released only when the last thread arrives.', True)),
        ('Both compute on the complete tile. Warp 0 finishes first and waits at barrier 2 (line 98).',
         lanes([('load', 0, 2), ('wait', 2, 3.2), ('comp', 3.2, 5.2), ('wait', 5.2, 6.4)], [('load', 0, 3.2), ('comp', 3.2, 6.4)], [3.2, 6.4], 6.4, [T0, T1], 'Barrier 2 = "everyone has finished READING the tile". Only then may anyone overwrite it.', True)),
        ('Then both load tile t+1. Correct every time, whatever the timing.',
         lanes([('load', 0, 2), ('wait', 2, 3.2), ('comp', 3.2, 5.2), ('wait', 5.2, 6.4), ('load', 6.4, 8)], [('load', 0, 3.2), ('comp', 3.2, 6.4), ('load', 6.4, 9)], [3.2, 6.4], 8.5, [NEW0, ('rows from warp 1: tile t+1', YEL)], 'Cost of the two barriers: the fast warp idles (grey). That idle time is why later kernels double-buffer.', True)),
        ('REMOVE barrier 1: warp 0 starts its inner loop while warp 1 is still writing its half.',
         lanes([('load', 0, 2), ('bad', 2, 4)], [('load', 0, 3.2), ('comp', 3.2, 6.4)], [], 2.6, [T0, OLD], 'Warp 0 reads warp 1\'s cells before they are written: it multiplies tile t−1 (or garbage) values. Wrong C.', False)),
        ('REMOVE barrier 2: warp 0 finishes tile t and immediately overwrites its half with tile t+1.',
         lanes([('load', 0, 2), ('wait', 2, 3.2), ('comp', 3.2, 5.2), ('load', 5.2, 6.6)], [('load', 0, 3.2), ('comp', 3.2, 5.0), ('bad', 5.0, 7.6)], [3.2], 6, [NEW0, T1], 'Warp 1 is still reading row values from warp 0\'s half: it now mixes tile t and t+1. A race: wrong only SOMETIMES.', False)),
        ('Early return before a barrier: a guard thread that returns never arrives at __syncthreads.',
         lanes([('load', 0, 2), ('wait', 2, 10)], [('load', 0, 1)], [], 5, [T0, ('warp 1 lanes returned early', PINK)], 'Undefined behaviour (hang or wrong). So the tiled kernels load 0.0f out of range and guard only the store (line 100).', False)),
    ]
    save('d05i-race.svg', W, H, 'Why the tiled kernel needs two __syncthreads per tile', 'two warps of one block sharing one tile; time runs left to right (schematic, not measured)', st, fr, 4.4)

def edge():
    W, H = 960, 470; cs = 34; gx, gy = 40, 96
    N, TILE = 5, 2
    st = T(gx, 86, 'C index space covered by the grid: 3 × 3 blocks of 2 × 2 threads = 6 × 6 threads for a 5 × 5 C', 12, 700)
    def cgrid(hblock=None, hcells=()):
        o = ''
        for i in range(6):
            for j in range(6):
                ok = i < N and j < N
                f = BLUE if ok else GREY
                if (i, j) in hcells: f = YEL
                o += R(gx + j * cs, gy + i * cs, cs - 2, cs - 2, f, RED if (i, j) in hcells else '#ffffff', 2 if (i, j) in hcells else 1, 3)
                o += T(gx + j * cs + 16, gy + i * cs + 21, f'{i},{j}' if ok else '×', 9, 400, INK if ok else MUTE, 'middle')
        for k in range(4):
            o += L(gx + k * 2 * cs - 1, gy - 1, gx + k * 2 * cs - 1, gy + 6 * cs, INK, 2) + L(gx - 1, gy + k * 2 * cs - 1, gx + 6 * cs, gy + k * 2 * cs - 1, INK, 2)
        if hblock:
            bx_, by_ = hblock
            o += R(gx + bx_ * 2 * cs - 3, gy + by_ * 2 * cs - 3, 2 * cs + 4, 2 * cs + 4, 'none', RED, 2.6, 4)
        return o
    sx = 330
    def tiles(sA, sB, note):
        o = T(sx, 120, 'block (2, 2), last step t = 2 (k = 4, 5):', 12, 700)
        o += grid(sx, 150, sA, 64, BLUE, {(r, c): (PINK, RED) for r in range(2) for c in range(2) if sA[r][c] == '0'}, 'sA', 11)
        o += grid(sx + 170, 150, sB, 64, GREEN, {(r, c): (PINK, RED) for r in range(2) for c in range(2) if sB[r][c] == '0'}, 'sB', 11)
        for k, s in enumerate(note):
            o += T(sx, 300 + k * 22, s, 12)
        return o
    sA = [['A[4][4]', '0'], ['0', '0']]; sB = [['B[4][4]', '0'], ['0', '0']]
    fr = [
        ('N = 5 is not a multiple of TILE = 2. Grid = (5 + 1)/2 = 3 blocks per side, so 36 threads for 25 outputs.',
         cgrid() + T(sx, 140, '(N + TILE − 1) / TILE rounds UP: no output is left without a thread.', 13, 700) + T(sx, 164, 'The 11 threads marked × have row ≥ 5 or col ≥ 5: they own nothing.', 12) + T(sx, 186, 'They still exist, still load and still reach both barriers.', 12)),
        ('The K loop also rounds up: t = 0, 1, 2 covers k = 0…5, but k = 5 does not exist.',
         cgrid() + T(sx, 140, 'tiles along K: t = 0 → k 0–1,  t = 1 → k 2–3,  t = 2 → k 4–5', 13, 700) + T(sx, 164, 'k = 5 is past the edge of A\'s columns and B\'s rows.', 12)),
        ('Block (2, 2) at t = 2: only thread (0,0) maps to real data. Every other load hits the guard and stores 0.0f.',
         cgrid((2, 2), {(4, 4)}) + tiles(sA, sB, ['thread (0,0): row 4, aCol 4 → A[4][4] ✓; bRow 4, col 4 → B[4][4] ✓', 'thread (1,0): col 5 ≥ N → sB = 0; aCol 5 ≥ N → sA = 0', 'thread (0,1): row 5 ≥ N → sA = 0; bRow 5 ≥ N → sB = 0', 'thread (1,1): everything out of range → 0, 0'])),
        ('Why zero is safe: the padding only ever multiplies into a sum. 0 × anything adds nothing.',
         cgrid((2, 2), {(4, 4)}) + tiles(sA, sB, ['thread (0,0), k = 0: A[4][4]·B[4][4]   (real term)', 'thread (0,0), k = 1: sA[0][1]·sB[1][0] = 0·0 = 0', 'so acc gets exactly the terms k = 0…4 ✓', 'the ternary (line 91–92) also stops OUT-OF-BOUNDS reads'])),
        ('Only the STORE is skipped (line 100). Threads never `return` early, because they must reach __syncthreads.',
         cgrid((2, 2), {(4, 4)}) + T(sx, 140, 'if (row < N && col < N) C[row*N + col] = acc;   ← 1 of 4 threads stores', 13, 700) + T(sx, 166, 'mm_naive uses an early return (line 48): fine there, it has no barrier.', 12) + T(sx, 188, 'A `return` in the tiled kernel would leave a barrier waiting for missing threads: undefined.', 12)),
        ('Cost: idle threads. Toy: 11 of 36 threads (31%) idle. At N = 32768 with TILE 32 or 64 there is NO waste.',
         cgrid() + T(sx, 140, '32768 / 32 = 1024 and 32768 / 64 = 512 exactly: the measured sizes never hit the guards.', 13, 700) + T(sx, 164, 'So the guards are correctness insurance, not a performance factor, in every benchmark run.', 12) + T(sx, 186, '(At awkward small N they cost divergence in the last block row/column only.)', 12)),
    ]
    save('d05i-edge.svg', W, H, 'When N is not a multiple of TILE', 'mm_tiled_1to1<2> on a 5 × 5 problem: grid round-up, zero padding, store guard, no early return', st, fr, 4.6)

def occupancy():
    W, H = 960, 440
    bx0, bw = 260, 620
    rows = [('registers', 65536, 'regs'), ('shared memory', 228, 'KB'), ('threads', 2048, 'threads')]
    st = ''
    for k, (n, cap, u) in enumerate(rows):
        y = 96 + k * 70
        st += T(40, y + 24, n, 13, 700) + T(40, y + 42, f'{cap:,} {u} per SM', 11, 400, MUTE)
        st += R(bx0, y, bw, 40, '#f6f8fa', MUTE, 1, 4)
    def bars(per_block, fits, lim, regs_note, verdict, hypo=False):
        b = ''
        for k, (n, cap, u) in enumerate(rows):
            y = 96 + k * 70
            use = per_block[k]
            nb = min(fits[k], 40)
            for i in range(min(nb, int(cap // use))):
                x = bx0 + bw * i * use / cap
                inb = i < lim
                b += R(x + 1, y + 2, bw * use / cap - 2, 36, SC[i % 4] if inb else WHITE, MUTE if inb else '#d0d7de', 1, 3, None if inb else '3 3')
            b += T(bx0 + bw + 8, y + 18, f'→ {fits[k]}', 13, 700, RED if fits[k] == lim else INK)
            b += T(bx0 + bw + 8, y + 34, 'blocks', 10, 400, MUTE)
            if fits[k] == lim:
                b += R(bx0 - 3, y - 3, bw + 6, 46, 'none', RED, 2.4, 5)
        b += T(40, 316, regs_note, 12)
        b += B(40, 336, 880, 40, verdict, PINK if hypo else YEL, RED, 2, 12)
        return b
    fr = [
        ('An SM holds blocks until ONE resource runs out. Solid = blocks that fit together; the red box is the limiter.',
         T(260, 316, 'occupancy = resident warps ÷ 64 (an SM holds at most 2,048 threads = 64 warps)', 13, 700) + T(260, 340, 'register counts: cuobjdump of the committed binaries; occupancy is THEORETICAL (ncu needed root)', 12, 400, MUTE)),
        ('mm_tiled_strided<64,32>: 1,024 threads × 32 registers; two 16 KB tiles.',
         bars([32768, 32, 1024], [2, 7, 2], 2, 'regs: 32 × 1,024 = 32,768 per block → 2 blocks;  threads: 2,048 / 1,024 = 2;  shared: 228 / 32 ≈ 7',
              '2 blocks × 32 warps = 64 warps → 100% theoretical occupancy (measured 3,635 ms, 19.36 TFLOP/s)')),
        ('mm_block2d<64,2>: 1,024 threads × 40 registers: the register file fits only one block.',
         bars([40960, 32, 1024], [1, 7, 2], 1, 'regs: 40 × 1,024 = 40,960 → 65,536 / 40,960 = 1.6 → 1 block;  threads would allow 2',
              '1 block = 32 warps → 50%. Slower than strided (4,263 vs 3,635 ms): same 4 outputs/thread, half the warps.')),
        ('mm_block2d<64,4>: 256 threads × 62 registers (allocated in chunks → 64).',
         bars([16384, 32, 256], [4, 7, 8], 4, 'regs: 64 × 256 = 16,384 → 4 blocks;  threads would allow 8;  shared ≈ 7',
              '4 blocks × 8 warps = 32 warps → 50%, yet the FASTEST kernel (2,737 ms): more work per thread beats more threads.')),
        ('flash_forward_kernel (Day 6): 256 threads × 62 registers, 28,800 B dynamic shared memory.',
         bars([16384, 28.8, 256], [4, 7, 8], 4, 'regs: 64 × 256 = 16,384 → 4 blocks;  shared: 228 KB / 28.8 KB ≈ 7;  threads 8',
              '50% theoretical (occupancy API). Registers limit it, not shared memory (prep pack Q40).')),
        ('Hypothetical (the check question): 128 registers × 256 threads.',
         bars([32768, 32, 256], [2, 7, 8], 2, 'regs: 128 × 256 = 32,768 → 2 blocks = 512 threads = 16 warps',
              'HYPOTHETICAL, not a project kernel: 25% occupancy. Raising R further trades warps for reuse.', True)),
    ]
    save('d05i-occupancy.svg', W, H, 'Occupancy: which resource runs out first', 'one SM of the H200; each coloured box is one resident block of the named kernel', st, fr, 4.6)

def timing():
    W, H = 960, 430
    st = T(40, 104, 'CPU (host)', 13, 700) + T(40, 184, 'GPU stream', 13, 700)
    st += L(150, 96, 920, 96, '#d0d7de', 1) + L(150, 176, 920, 176, '#d0d7de', 1)
    def ev(x, y, lab, col=RED): return f"<circle cx='{x}' cy='{y}' r='7' fill='{col}'/>" + T(x, y - 12, lab, 10, 700, col, 'middle')
    def cpu(segs):
        o = ''
        for lab, a, c, f in segs: o += B(a, 82, c - a, 28, lab, f, size=10)
        return o
    def gpu(segs):
        o = ''
        for lab, a, c, f in segs: o += B(a, 162, c - a, 28, lab, f, size=10)
        return o
    def note(lines, bad=False):
        o = ''
        for k, s in enumerate(lines):
            o += T(40, 246 + k * 22, s, 12, 700 if k == 0 else 400, RED if (bad and k == 0) else INK)
        return o
    fr = [
        ('Warm-up: time_kernel first runs untimed launches (1 for the custom kernels, 2 for cuBLAS) and waits for them.',
         cpu([('launch ×1', 160, 230, GREY), ('cudaDeviceSynchronize', 230, 420, PINK)]) + gpu([('warm-up kernel (untimed)', 170, 410, GREY)])
         + note(['Why warm up [theory]: the first launch can pay one-off costs that are not the kernel:', 'module loading, clocks ramping up, cold caches and TLBs, cuBLAS picking its algorithm.', 'The project does this at matmul_bench.cu:318–319 (and 3 warm-ups in mlp_bench and bench_attention).'])),
        ('cudaEventRecord(beg) only ENQUEUES a marker. The CPU returns at once.',
         cpu([('Record(beg)', 160, 250, YEL)]) + gpu([]) + ev(200, 176, 'beg')
         + note(['The timestamp is taken when the GPU reaches the marker, on the GPU\'s own clock,', 'not when the CPU makes the call. So host code before or after does not leak into the time.'])),
        ('The launch is asynchronous: the CPU queues the kernel in microseconds; the GPU runs it later.',
         cpu([('Record(beg)', 160, 250, YEL), ('launch()', 250, 320, BLUE), ('Record(end)', 320, 410, YEL)]) + gpu([('kernel: block2d_t64r4 ≈ 2.7 s', 210, 860, GREEN)]) + ev(200, 176, 'beg') + ev(870, 176, 'end')
         + note(['A CPU timer around launch() alone would measure only the enqueue (µs): the classic timing bug.', 'Events bracket exactly the GPU work between them: no copies, no Python, no host code.'], True)),
        ('cudaEventSynchronize(end): the CPU sleeps until the GPU reaches end. Then ElapsedTime(beg, end) is valid.',
         cpu([('Record(beg)', 160, 250, YEL), ('launch()', 250, 320, BLUE), ('Record(end)', 320, 410, YEL), ('EventSynchronize(end) … waits', 410, 870, PINK)]) + gpu([('kernel', 210, 860, GREEN)]) + ev(200, 176, 'beg') + ev(870, 176, 'end')
         + note(['first = 2,737 ms for block2d_t64r4 at N = 32768.', 'Budget test (line 331): first / 1000 × iters < 30 s?  2.737 × 10 = 27.4 s < 30 → YES.'])),
        ('Fits the budget: 9 more launches back to back between ONE event pair; the mean of 10 is reported.',
         cpu([('Record', 160, 210, YEL), ('9 launches', 210, 300, BLUE), ('Record', 300, 350, YEL), ('Synchronize … waits', 350, 900, PINK)]) + gpu([(f'k{i + 2}', 200 + i * 77, 275 + i * 77, GREEN) for i in range(9)])
         + note(['returns {total / 10, 10}: 2,736.6 ms, n = 10 (the CSV iters column).', 'Back-to-back launches also hide the per-launch gap, which only matters for tiny kernels.'])),
        ('Does not fit: mm_naive first = 140,486 ms → 140.5 × 10 = 1,405 s > 30 s → it stops at n = 1.',
         cpu([('Record(beg)', 160, 250, YEL), ('launch()', 250, 320, BLUE), ('Record(end)', 320, 410, YEL), ('Synchronize (140 s)', 410, 900, PINK)]) + gpu([('mm_naive, one run, 140,486 ms', 210, 890, GREEN)])
         + note(['Every kernel except cuBLAS and block2d_t64r4 has n = 1 at N = 32768 (strided: 3.635 × 10 = 36 s > 30).', 'Fine for a 140 s run (launch noise is negligible), but say "single run, no error bars" if asked.'], True)),
    ]
    save('d05i-timing.svg', W, H, 'How time_kernel measures a kernel', 'matmul_bench.cu:312–344: warm-up, CUDA events, and the 30-second budget', st, fr, 4.6)

def ladder():
    W, H = 960, 470
    ks = [('mm_naive', 140486, 0.50, 'lanes walk ROWS: 32 lines per warp load of A'),
          ('mm_coalesced', 12927, 5.44, '10.9×: lanes walk COLUMNS: 2 lines per step'),
          ('tiled_1to1<32>', 8148, 8.64, '1.59×: shared tiles; now 2 shared reads per FMA → ~8.4 roof'),
          ('tiled_contiguous<64,32>', 3790, 18.57, 'controlled twin of strided: lanes F apart, 2 lines/instr'),
          ('tiled_strided<64,32>', 3635, 19.36, '2.24× over 1-to-1: 2×2 outputs/thread, 100% occupancy'),
          ('block2d<64,2>', 4263, 16.51, 'slower than strided: 1 block/SM (50%), contiguous loader'),
          ('block2d<64,4>', 2737, 25.71, '1.33× over strided: 4×4 in registers, 0.5 shared reads/FMA'),
          ('cuBLAS SGEMM', 1318, 53.37, 'the ceiling: FP32 CUDA cores, no TF32; 80% of peak')]
    x0, xw, y0, rh = 230, 600, 104, 32
    sx = lambda v: x0 + xw * v / 70
    st = ''
    for v in (0, 10, 20, 30, 40, 50, 60, 70):
        st += L(sx(v), y0 - 6, sx(v), y0 + rh * 8 + 4, '#eaeef2', 1) + T(sx(v), y0 + rh * 8 + 20, str(v), 10, 400, MUTE, 'middle')
    st += T(sx(35), y0 + rh * 8 + 38, 'TFLOP/s (FP32), N = 32768, H200', 11, 600, MUTE, 'middle')
    for v, lab in ((66.9, 'peak 66.9'), (8.4, 'shared roof, 2 reads/FMA ≈ 8.4'), (33.5, 'shared roof, 0.5 reads/FMA ≈ 33.5')):
        st += L(sx(v), y0 - 10, sx(v), y0 + rh * 8, RED, 1.4, '5 3') + T(sx(v) + 3, y0 - 14 if v != 8.4 else y0 - 30, lab, 10, 600, RED)
    fr = []
    for n in range(len(ks)):
        b = ''
        for i, (k, ms, tf, why) in enumerate(ks[:n + 1]):
            y = y0 + i * rh
            hot = i == n
            b += T(x0 - 8, y + 21, k, 11, 700 if hot else 500, INK, 'end')
            b += R(x0, y + 4, max(2, sx(tf) - x0), rh - 10, YEL if hot else (BLUE if k != 'cuBLAS SGEMM' else GREY), RED if hot else MUTE, 2 if hot else 1, 3)
            b += T(sx(tf) + 6, y + 21, f'{tf:.2f}  ({ms:,} ms)', 11, 700 if hot else 400, RED if hot else INK)
        k, ms, tf, why = ks[n]
        fr.append((f'{k}: {why}', b))
    b = ''
    for i, (k, ms, tf, why) in enumerate(ks):
        y = y0 + i * rh
        b += T(x0 - 8, y + 21, k, 11, 500, INK, 'end') + R(x0, y + 4, max(2, sx(tf) - x0), rh - 10, BLUE if i != 7 else GREY, MUTE, 1, 3) + T(sx(tf) + 6, y + 21, f'{tf:.2f}', 11)
    fr.append(('Headline: 140,486 / 2,737 = 51.3× over naive (4.7× over coalesced); 25.71 / 53.37 = 48.2% of cuBLAS; 38.4% of peak.', b))
    save('d05i-ladder.svg', W, H, 'The measured ladder, one kernel at a time', 'results/pa2_matmul.csv at N = 32768; red dashed lines are THEORY (hand-derived roofs), bars are MEASURED', st, fr, 4.2)

# ═════════════════════════ DAY 6 ═════════════════════════

def split():
    W, H = 960, 450
    def mats(xo, xcols, wcols, title, note):
        """xcols: list of (row0,row1,colour) bands over X; wcols: list of (c0,c1,colour, alpha)"""
        o = T(xo, 90, title, 13, 700)
        o += R(xo, 104, 120, 30, GREY, MUTE, 1, 3) + T(xo + 60, 124, 'X  32 × N', 10, 600, INK, 'middle')
        for r0, r1, f in xcols:
            o += R(xo + 1, 105 + 28 * r0 / 32, 118, 28 * (r1 - r0) / 32, f, 'none', 0, 0)
        o += T(xo + 60, 124, 'X  32 × N', 10, 700, INK, 'middle')
        o += R(xo, 150, 200, 170, GREY, MUTE, 1, 3)
        for c0, c1, f, op in wcols:
            o += f"<rect x='{xo + 200 * c0}' y='{150 + 2}' width='{200 * (c1 - c0)}' height='166' fill='{f}' fill-opacity='{op}'/>"
        o += T(xo + 100, 240, 'W1   N × N', 12, 700, INK, 'middle')
        for k, s in enumerate(note): o += T(xo, 342 + k * 18, s, 11)
        return o
    st = ''
    fr = [
        ('The bytes: X is 32 × N (4.2 MB at N = 32768); W1 is N × N (4.29 GB). Time is about how often W is read.',
         mats(40, [], [], 'shapes at N = 32768', ['W1 + W2 = 8.6 GB;  activations ≈ 4 MB each', 'reading both W once at 4.8 TB/s ≥ 1.8 ms', 'measured baseline: 44.4 ms → W is re-read many times'])),
        ('(a) Baseline: one kernel per layer, grid (128, 32) = 4,096 blocks, default stream, all rows × all of W.',
         mats(40, [(0, 32, BLUE)], [(0, 1, BLUE, 0.6)], '(a) baseline', ['2 launches / iteration, no host sync', '44.43 ms at N = 32768'])),
        ('(b) Batch split: stream s gets rows 8s … 8s+7, but W1 is passed WHOLE: four kernels each sweep all of W.',
         mats(40, [(0, 32, BLUE)], [(0, 1, BLUE, 0.6)], '(a) baseline', ['44.43 ms'])
         + mats(370, [(8 * s, 8 * s + 8, SC[s]) for s in range(4)], [(0, 1, SC[s], 0.35) for s in range(4)], '(b) batch split, 4 streams', ['each stream: 8 rows × ALL of W1, W2', '4× launches + 4 cudaStreamSynchronize', '52.28 ms: slower at every width'])),
        ('(c) Column split: stream s gets columns 8,192·s … +8,191 of W1 and ALL 32 rows. W is partitioned, not duplicated.',
         mats(40, [(0, 32, BLUE)], [(0, 1, BLUE, 0.6)], '(a) baseline', ['44.43 ms'])
         + mats(370, [(8 * s, 8 * s + 8, SC[s]) for s in range(4)], [(0, 1, SC[s], 0.35) for s in range(4)], '(b) batch split', ['52.28 ms'])
         + mats(700, [(0, 32, GREY)], [(s / 4, (s + 1) / 4, SC[s], 0.75) for s in range(4)], '(c) column split', ['stream s: W1 + c0, b1 + c0, A1 + c0', 'grid (32, 32) = 1,024 blocks per kernel', '25.65 ms: 1.73× at N = 32768 only'])),
        ('Layer 2 needs ALL of A1, so the column split must sync all 4 streams between the layers (line 289).',
         mats(370, [(0, 32, GREY)], [(s / 4, (s + 1) / 4, SC[s], 0.75) for s in range(4)], 'layer 1 → barrier → layer 2', ['A1 column j comes from stream j·4/N', 'every Y column needs every A1 column', '→ cudaStreamSynchronize × 4, then W2 split'])),
        ('Measured (3 warm-up + 20 timed, one run): the split only pays where the baseline needs several waves.',
         T(80, 110, 'N', 12, 700) + T(220, 110, 'baseline', 12, 700) + T(400, 110, 'batch split', 12, 700) + T(600, 110, 'column split', 12, 700)
         + ''.join(T(80, 140 + i * 30, n, 12) + T(220, 140 + i * 30, a, 12) + T(400, 140 + i * 30, b, 12) + T(600, 140 + i * 30, c, 12, 700 if i == 3 else 400, RED if i == 3 else INK)
                   for i, (n, a, b, c) in enumerate([('32', '0.0069 ms', '0.0278 (4.1× slower)', '0.0339 (4.9× slower)'), ('1,024', '0.099', '0.120', '0.125'), ('16,384', '6.29', '6.85', '6.19 (1.02×)'), ('32,768', '44.43', '52.28', '25.65 (1.73×)')]))
         + T(80, 290, 'No correctness check in Part 1; the column split is wrong for N ≤ 512 (guard uses full N). The 32K result is unaffected.', 12, 600, RED)),
    ]
    save('d06i-split.svg', W, H, 'Three ways to run the batch-32 MLP forward pass', 'mlp_bench.cu:254–299; colours = the 4 streams; W shading = which part of W1 each stream reads', st, fr, 4.6)

def waves():
    W, H = 960, 470
    gx, gy, cw, rh = 40, 96, 4.6, 7          # 128 column chunks × 32 rows
    st = T(gx, 86, 'baseline grid: 128 column chunks (blockIdx.x) × 32 batch rows (blockIdx.y) = 4,096 blocks', 12, 700)
    st += R(gx, gy, 128 * cw, 32 * rh, GREY, MUTE, 1, 0)
    for r in range(0, 33, 8): st += L(gx, gy + r * rh, gx + 128 * cw, gy + r * rh, '#ffffff', 0.8)
    for c in range(0, 129, 32): st += L(gx + c * cw, gy, gx + c * cw, gy + 32 * rh, '#ffffff', 0.8)
    st += T(gx - 6, gy + 10, 'row 0', 9, 400, MUTE, 'end') + T(gx - 6, gy + 32 * rh, 'row 31', 9, 400, MUTE, 'end')
    def band(w, f, op=0.85):
        o = ''
        a, b = 1056 * w, min(4096, 1056 * (w + 1))
        r0, c0 = divmod(a, 128); r1, c1 = divmod(b, 128)
        if r0 == r1:
            return f"<rect x='{gx + c0 * cw}' y='{gy + r0 * rh}' width='{(c1 - c0) * cw}' height='{rh}' fill='{f}' fill-opacity='{op}'/>"
        o += f"<rect x='{gx + c0 * cw}' y='{gy + r0 * rh}' width='{(128 - c0) * cw}' height='{rh}' fill='{f}' fill-opacity='{op}'/>"
        if r1 > r0 + 1: o += f"<rect x='{gx}' y='{gy + (r0 + 1) * rh}' width='{128 * cw}' height='{(r1 - r0 - 1) * rh}' fill='{f}' fill-opacity='{op}'/>"
        if c1: o += f"<rect x='{gx}' y='{gy + r1 * rh}' width='{c1 * cw}' height='{rh}' fill='{f}' fill-opacity='{op}'/>"
        return o
    tx = 650
    def txt(lines, top=110):
        return ''.join(T(tx, top + k * 21, s, 12, 700 if k == 0 else 400, RED if k == 0 else INK) for k, s in enumerate(lines))
    def colsplit(hi=False):
        o = T(gx, 352, 'one column-split kernel: 32 chunks × 32 rows = 1,024 blocks ≤ 1,056 → ONE wave', 12, 700)
        o += R(gx, 360, 32 * cw * 2, 48, GREEN if hi else GREY, MUTE, 1, 0)
        o += T(gx + 32 * cw * 2 + 10, 388, 'all 32 rows of each W chunk are resident together → W lines shared via L2', 11)
        return o
    fr = [
        ('A wave = blocks resident at once: up to 8 mm_thin blocks per SM × 132 SMs = 1,056. Baseline needs 4,096 / 1,056 ≈ 3.9 waves.',
         txt(['dispatch order (typical, not guaranteed):', 'blockIdx.x first, then the next blockIdx.y.', 'So a wave = a band of ~8 batch rows', 'across ALL 128 column chunks.'])),
        ('Wave 0: batch rows 0–8 × all 128 column chunks → it streams the WHOLE of W1 (4.29 GB) through a ~50 MB L2.',
         band(0, BLUE) + txt(['wave 0 touches every column of W1', '4.29 GB ≫ 50 MB L2:', 'the early chunks are evicted', 'before anyone else needs them.'])),
        ('Wave 1: the next ~8 rows need the same W columns again, but those lines left L2 long ago → HBM again.',
         band(0, BLUE, 0.35) + band(1, GREEN) + txt(['W1 is read from HBM again', 'for every wave: about 4 times per layer.', '(reconstructed hypothesis; not profiled,', 'ncu needed root)'])),
        ('Waves 2 and 3: two more sweeps. Reuse of W across the batch is left to cache luck.',
         band(0, BLUE, 0.35) + band(1, GREEN, 0.35) + band(2, YEL) + band(3, PINK) + txt(['≈ 3.9 sweeps of W per layer', 'baseline time scales 7.1× from', 'N = 16K → 32K for 4× the work', '(6.29 → 44.43 ms): a large-N pathology'])),
        ('Column split: each kernel is 32 chunks wide and fits in ONE wave, so its 32 batch rows run together.',
         band(0, BLUE, 0.2) + band(1, GREEN, 0.2) + band(2, YEL, 0.2) + band(3, PINK, 0.2) + colsplit(True) + txt(['each W1 line is needed by 32 blocks', 'that are resident at the same time', '→ read from HBM ~once, then L2 hits', 'column split scales 4.1× (6.19 → 25.65)'])),
        ('Verdict: the 1.73× is most likely LOCALITY, not concurrency. One experiment settles it: all 4 chunks on ONE stream.',
         colsplit(True) + txt(['test (prep pack exercise 2): st[s] → st[0]', '≈ 25–27 ms still → locality', '≈ 44 ms again → overlap mattered', 'not run: say "likely", never "proved"'])),
    ]
    save('d06i-waves.svg', W, H, 'Why the column split was 1.73× faster: waves and L2', 'mm_thin at N = 32768; each tiny cell is one block (256 threads); coloured bands = blocks resident in the same wave', st, fr, 4.8)

def ckpt_life():
    W, H = 960, 480
    names = ['Z1', 'A1', 'Z2', 'A2', 'Z3', 'A3', 'Z4']
    size = {'Z1': 64, 'A1': 64, 'Z2': 64, 'A2': 64, 'Z3': 64, 'A3': 64, 'Z4': 16}
    cols = ['F1', 'F2', 'F3', 'F4', 'rebuild B', 'grad 4', 'grad 3', 'rebuild A', 'grad 2', 'grad 1']
    gx, gy, cw, rh = 150, 108, 62, 26
    st = T(gx, 78, 'time →  (forward F1–F4, then backward layer 4 → 1)', 11, 700, MUTE)
    for j, c in enumerate(cols): st += T(gx + j * cw + cw / 2, gy - 4, c, 10, 600, INK, 'middle')
    for i, n in enumerate(names): st += T(gx - 10, gy + i * rh + 18, f'{n} ({size[n]} MiB)', 11, 600, INK, 'end')
    st += B(40, 300, 110, 22, 'stored', BLUE, size=10) + B(160, 300, 110, 22, 'checkpoint', GREEN, size=10) + B(280, 300, 130, 22, 'in arena (temp)', YEL, size=10) + B(420, 300, 130, 22, 'recomputed', ORANGE, size=10) + B(560, 300, 160, 22, 'discarded / overwritten', PINK, size=10)
    # life tables: name -> list of (col, state)
    base = {'Z1': range(0, 10), 'A1': range(0, 9), 'Z2': range(1, 9), 'A2': range(1, 7), 'Z3': range(2, 7), 'A3': range(2, 6), 'Z4': range(3, 6)}
    def cell(i, j, f, lab=''):
        return R(gx + j * cw + 1, gy + i * rh + 2, cw - 2, rh - 4, f, '#ffffff', 1, 3) + (T(gx + j * cw + cw / 2, gy + i * rh + 17, lab, 9, 700, RED, 'middle') if lab else '')
    def table(upto, mode):
        o = ''
        if mode == 'base':
            for i, n in enumerate(names):
                for j in base[n]:
                    if j <= upto: o += cell(i, j, GREEN if n in ('A2', 'Z4') else BLUE)
            return o
        ck = {  # checkpointed life: (row, col) -> state
        }
        def put(n, j, s): ck[(names.index(n), j)] = s
        for j in range(1, 7): put('A2', j, 'ck')
        for j in range(3, 6): put('Z4', j, 'ck')
        put('Z1', 0, 'tmp'); put('A1', 0, 'tmp'); put('Z1', 1, 'tmp'); put('A1', 1, 'tmp'); put('Z2', 1, 'tmp')
        put('Z1', 2, 'x'); put('A1', 2, 'x'); put('Z2', 2, 'x')
        put('Z3', 2, 'tmp'); put('A3', 2, 'tmp'); put('Z3', 3, 'tmp'); put('A3', 3, 'tmp')
        put('Z3', 4, 're'); put('A3', 4, 're'); put('Z3', 5, 're'); put('A3', 5, 're'); put('Z3', 6, 're')
        put('Z3', 7, 'x'); put('A3', 7, 'x')
        for n in ('Z1', 'A1', 'Z2'): put(n, 7, 're')
        for n in ('Z1', 'A1', 'Z2'): put(n, 8, 're')
        put('Z1', 9, 're')
        for (i, j), s in ck.items():
            if j <= upto:
                o += cell(i, j, {'ck': GREEN, 'tmp': YEL, 're': ORANGE, 'x': PINK}[s], '✗' if s == 'x' else '')
        return o
    def meter(mode, upto, lines):
        o = R(740, 108, 190, 182, '#f6f8fa', '#d0d7de', 1, 8)
        for k, s in enumerate(lines): o += T(752, 132 + k * 22, s, 11, 700 if k == 0 else 400, RED if k == 0 else INK)
        return o
    fr = [
        ('BASELINE: every Z and A is kept from the moment it is made until its last use in backward.',
         table(9, 'base') + meter('base', 9, ['baseline', 'Z1 A1 Z2 A2 Z3 A3: 6 × 64', '+ Z4: 16', '= 400 MiB of activations', 'all alive at the loss', '(the peak)'])),
        ('CHECKPOINTED, forward layers 1–2: Z1, A1, Z2 go into the arena; A2 into its own buffer (checkpoint).',
         table(1, 'ck') + meter('ck', 1, ['checkpointed', 'A2 (64) + Z4 (16)', '+ arena max(128, 192)', '= 272 MiB', 'arena holds segment A', 'Z1 · A1 · Z2'])),
        ('Forward layer 3: Z3, A3 are written at the START of the same arena: segment A is overwritten (discarded).',
         table(2, 'ck') + meter('ck', 2, ['rZ3 = arena = rZ1', 'the two layouts ALIAS:', 'writing Z3 destroys Z1', 'that is the point:', 'segment A is no longer', 'needed until much later'])),
        ('Forward layer 4: Z4 is a checkpoint (the MSE loss needs it first). Forward ends holding A2, Z4, arena.',
         table(3, 'ck') + meter('ck', 3, ['live now:', 'A2  (checkpoint, 64)', 'Z4  (checkpoint, 16)', 'arena: Z3, A3 (128 of 192)', '272 MiB vs 400 MiB', '−32% activations'])),
        ('Backward starts: rebuild segment B (Z3, A3) from A2 (lines 495–497), then the layer-4 and layer-3 gradients.',
         table(6, 'ck') + meter('ck', 6, ['recompute = layer 3 GEMM', 'grad 4 uses a3 (dW4)', 'grad 3 uses z3 (ReLU mask)', 'and A2 (dW3)', 'note: the arena still', 'held Z3, A3 from forward'])),
        ('Line 513: z3, a3 are dead. Rebuild segment A (Z1, A1, Z2) from X over the same arena (lines 515–519).',
         table(7, 'ck') + meter('ck', 7, ['segment B overwritten ✗', 'recompute = layers 1–2', 'reading z3 after here', 'would see Z1\'s numbers:', 'silently wrong gradients', 'nothing enforces the order'])),
        ('Layer-2 and layer-1 gradients use z2, a1, z1 from the arena; then SGD. The step is done.',
         table(9, 'ck') + meter('ck', 9, ['measured (N4096 config)', 'activations 400 → 272 MiB', 'whole step 962 → 834 MiB', '(−13.3%)', 'backward 136.1 → 171.7 ms', '+35.5 ms = 0.9 × forward'])),
    ]
    save('d06i-ckpt-life.svg', W, H, 'Checkpointing: what is stored, discarded and recomputed', 'mlp_bench.cu:464–534, N4096 config (batch 4096, widths 1024 → 4096 → 4096 → 4096 → 1024); one 4096 × 4096 float tensor = 64 MiB', st, fr, 4.8)

def flash_tiles():
    W, H = 960, 470
    st = T(40, 88, 'HBM (global, 4.8 TB/s)', 13, 700) + T(500, 88, 'one SM: shared memory (28,800 B) + registers', 13, 700, BT)
    st += R(30, 96, 420, 250, '#f6f8fa', '#d0d7de', 1, 8) + R(490, 96, 440, 250, '#eef6ff', '#c8e1ff', 1, 8)
    def hbm(hot=(), kj=None):
        o = B(50, 112, 120, 40, 'Q  [8192 × 64]', GREY if 'Q' not in hot else YEL, RED if 'Q' in hot else MUTE, 2 if 'Q' in hot else 1, 11)
        o += R(190, 112, 60, 130, GREY, MUTE, 1, 3) + T(220, 104, 'K', 12, 700, INK, 'middle')
        o += R(270, 112, 60, 130, GREY, MUTE, 1, 3) + T(300, 104, 'V', 12, 700, INK, 'middle')
        if kj is not None:
            y = 112 + 124 * kj / 255
            o += R(190, y, 60, 6, YEL, RED, 2, 1) + R(270, y, 60, 6, YEL, RED, 2, 1)
        o += B(350, 112, 90, 40, 'O', GREEN if 'O' in hot else GREY, RED if 'O' in hot else MUTE, 2 if 'O' in hot else 1, 11)
        o += B(50, 266, 380, 60, 'N × N scores: never written', WHITE, RED, 1.6, 11, sub='standard path: 34.4 GB at B 8, 16 heads')
        return o
    def sram(q=False, k=False, v=False, s=False, o_=False, lbl=''):
        o = B(510, 112, 130, 44, 'Qi 32 × 64', BLUE if q else WHITE, BT, 1.6 if q else 1, 11, sub='8 KB, staged once')
        o += B(650, 112, 130, 44, 'Kj 32 × 65', BLUE if k else WHITE, BT, 1.6 if k else 1, 11, sub='stride 65 (padding)')
        o += B(790, 112, 130, 44, 'Vj 32 × 64', BLUE if v else WHITE, BT, 1.6 if v else 1, 11, sub='8 KB')
        o += B(510, 176, 200, 44, 'S 32 × 32 (this tile only)', YEL if s else WHITE, RED if s else BT, 1.6 if s else 1, 11, sub='4 KB, overwritten every tile')
        o += B(720, 176, 200, 44, 'registers per row: m, l, o', GREEN if o_ else WHITE, GT, 1.6 if o_ else 1, 11, sub='o_acc[4][4] (local mem, prep Q38)')
        if lbl: o += T(510, 248, lbl, 12, 700, RED)
        return o
    def cnt(j, lines):
        return ''.join(T(510, 276 + k * 20, s, 11) for k, s in enumerate(lines))
    fr = [
        ('Block (qt, h, b) owns 32 query rows of one head. Grid (256, 16, 8) = 32,768 such blocks of 256 threads.',
         hbm() + sram() + cnt(0, ['8 warps × 4 rows = 32 query rows', 'loop over Tc = 8192 / 32 = 256 key tiles', 'nothing N × N anywhere'])),
        ('Stage Qi once (lines 81–85): 256 threads copy 32 × 64 floats. Initialise m = −∞, l = 0, o = 0 (lines 88–96).',
         hbm(('Q',)) + sram(q=True, o_=True, lbl='m_run = −INFINITY, l_run = 0, o_acc = 0') + cnt(0, ['HBM read so far: Q tile 8 KB'])),
        ('Tile j = 0: load Kj and Vj (lines 103–110), between two __syncthreads.',
         hbm(kj=0) + sram(q=True, k=True, v=True, o_=True, lbl='K, V rows 0–31 → shared') + cnt(0, ['HBM read: + 8 KB K + 8 KB V', 'barrier 1: nobody overwrites a tile still being read', 'barrier 2: nobody reads a half-loaded tile'])),
        ('Scores (lines 118–126): lane c owns key c; S = Qi·Kjᵀ for the warp\'s 4 rows, kept in REGISTERS (a[4]).',
         hbm(kj=0) + sram(q=True, k=True, v=True, o_=True, lbl='a[r] = Σx Qi[row][x] · Kj[c][x]   (64 FMAs × 4 rows per lane)') + cnt(0, ['one Kj read feeds 4 FMAs (4 query rows)', 'Qi read is the same address for all lanes: broadcast'])),
        ('Online softmax (lines 130–152): tile max and sum by 5 shuffles each; p written into S; m, l rescaled.',
         hbm(kj=0) + sram(q=True, k=True, v=True, s=True, o_=True, lbl='m_new = max(m, m_tile);  l = e_run·l + e_tile·l_tile') + cnt(0, ['S holds p for THIS tile only (4 KB)', '__syncwarp: each warp reads only its own rows of S'])),
        ('PV (lines 158–172): lane x owns features x and x + 32; o = o·e_run + (P·V)·e_tile.',
         hbm(kj=0) + sram(q=True, k=True, v=True, s=True, o_=True, lbl='o_acc[r][jj] = o_acc[r][jj]·e_run[r] + pv[r]·e_tile[r]') + cnt(0, ['one Vj read feeds 4 FMAs (4 rows)', 'Vj[c·64 + x]: adjacent x → adjacent banks, no padding needed'])),
        ('Tile j = 1 overwrites Kj, Vj and S. The old scores are GONE: only m, l, o carry the history.',
         hbm(kj=1) + sram(q=True, k=True, v=True, s=True, o_=True, lbl='tile 0\'s S was discarded, never written to HBM') + cnt(1, ['HBM read so far: 8 KB + 2 × 16 KB', 'if the new max is bigger, e_run < 1 shrinks the old l and o'])),
        ('… tile j = 255. Each block has read ALL of its head\'s K and V once (2 MB + 2 MB), and Q once.',
         hbm(kj=255) + sram(q=True, k=True, v=True, s=True, o_=True, lbl='256 tiles done') + cnt(255, ['per block: 8 KB Q + 4 MB K, V read; 0 B of scores written'])),
        ('Finish (lines 175–183): O[row][x] = o / l, written once to HBM. Exact softmax(QKᵀ/√D)·V.',
         hbm(('O',), kj=None) + sram(q=True, o_=True, lbl='divide by l once, at the very end') + cnt(255, ['max abs error vs PyTorch: 6.7e-7 (CSV ≤ 7.2e-7)', 'measured 160 ms at N = 8192 vs 105 (explicit) and 63 (fused SDPA)', 'its win is MEMORY, not speed'])),
    ]
    save('d06i-flash-tiles.svg', W, H, 'FlashAttention forward: one block walks over the key tiles', 'flash_lib.cu:47–184 at N = 8192, D = 64, BR = BC = 32; the N × N score matrix never exists', st, fr, 4.8)

def flash_dbg():
    code = [(131, 's = (lane < kv_len) ? a[r] * scale : -INFINITY;'),
            (136, 'm_tile = fmaxf(m_tile, shfl_xor(m_tile, off)); //×5'),
            (138, 'p = (lane < kv_len) ? __expf(s - m_tile) : 0;'),
            (143, 'l_tile += shfl_xor(l_tile, off);            //×5'),
            (145, 'S[(r0 + r) * BC + lane] = p;'),
            (147, 'const float m_new = fmaxf(m_run[r], m_tile);'),
            (148, 'e_run[r]  = __expf(m_run[r] - m_new);'),
            (149, 'e_tile[r] = __expf(m_tile   - m_new);'),
            (150, 'l_run[r]  = e_run[r]*l_run[r] + e_tile[r]*l_tile;'),
            (151, 'm_run[r]  = m_new;'),
            (167, 'pv[r] = fmaf(S[(r0+r)*BC + c], vv, pv[r]);  // over c'),
            (171, 'o_acc[r][jj] = o_acc[r][jj]*e_run[r] + pv[r]*e_tile[r];'),
            (182, 'O[...] = o_acc[r][jj] * (1.0f / l_run[r]);')]
    V = ['tile j (keys)', 's  (lanes 0, 1)', 'm_tile', 'p  (lanes 0, 1)', 'l_tile', 'm_run', 'e_run, e_tile', 'l_run', 'pv (feature 0)', 'o_acc (feature 0)', 'O (feature 0)']
    steps = [
        ((131,), {'tile j (keys)': '0: keys 0, 1', 's  (lanes 0, 1)': '16/8, 8/8 = 2, 1', 'm_tile': '–', 'p  (lanes 0, 1)': '–', 'l_tile': '–', 'm_run': '−∞', 'e_run, e_tile': '–', 'l_run': '0', 'pv (feature 0)': '–', 'o_acc (feature 0)': '0', 'O (feature 0)': '–'},
         'Toy: one query row, 2 keys per tile (kv_len = 2), D = 64 → scale 1/8. a = [16, 8] gives s = [2, 1]. Lanes ≥ 2 get −∞.'),
        ((136,), {'m_tile': '2 (every lane)'}, 'Line 136: 5 butterfly shuffles (offsets 16, 8, 4, 2, 1). Every lane ends with the tile max, no shared memory.'),
        ((138,), {'p  (lanes 0, 1)': 'e⁰, e⁻¹ = 1, 0.368'}, 'Line 138: exponentials relative to the TILE max, so every exponent ≤ 0 (no overflow). Lanes ≥ 2 get 0.'),
        ((143,), {'l_tile': '1.368'}, 'Line 143: the tile\'s sum, by the same 5-step shuffle with +.'),
        ((145,), {}, 'Line 145: p goes into this warp\'s own row of S (shared). The PV phase needs it with lanes in a different role.'),
        ((147,), {}, 'Line 147: m_new = max(−∞, 2) = 2.'),
        ((148, 149), {'e_run, e_tile': 'e^(−∞) = 0, e⁰ = 1'}, 'Lines 148–149: e_run = 0 wipes the empty initial state; e_tile = 1 (this tile\'s max IS the new max).'),
        ((150, 151), {'l_run': '0·0 + 1·1.368 = 1.368', 'm_run': '2'}, 'Lines 150–151: running sum and max now describe keys 0–1.'),
        ((167,), {'pv (feature 0)': '1·1 + 0.368·3 = 2.104'}, 'Line 167: values v = [1, 3] for feature 0. pv = Σc p[c]·v[c] over this tile\'s keys.'),
        ((171,), {'o_acc (feature 0)': '0·0 + 2.104·1 = 2.104'}, 'Line 171: o = o·e_run + pv·e_tile. Unnormalised: no division yet.'),
        ((131,), {'tile j (keys)': '1: keys 2, 3', 's  (lanes 0, 1)': '24/8, 0/8 = 3, 0'}, 'Tile 1: the old Kj, Vj, S are overwritten. a = [24, 0] → s = [3, 0]; values v = [2, 0].'),
        ((136, 138, 143), {'m_tile': '3', 'p  (lanes 0, 1)': '1, e⁻³ = 0.0498', 'l_tile': '1.0498'}, 'Same three steps: tile max 3, p relative to 3, tile sum 1.0498.'),
        ((147, 148, 149), {'e_run, e_tile': 'e^(2−3) = 0.368, e⁰ = 1'}, 'New max 3 > 2: the OLD state was relative to 2, so it must shrink by e^(2−3) = 0.368.'),
        ((150, 151), {'l_run': '0.368·1.368 + 1.0498 = 1.553', 'm_run': '3'}, 'Line 150: l = 0.503 + 1.050 = 1.553 = Σ over all 4 keys of e^(s − 3). Exact.'),
        ((167,), {'pv (feature 0)': '1·2 + 0.0498·0 = 2.0'}, 'Line 167: this tile\'s P·V for feature 0.'),
        ((171,), {'o_acc (feature 0)': '2.104·0.368 + 2.0 = 2.774'}, 'Line 171: the old output is rescaled by the same factor as the old sum. That is what keeps it exact.'),
        ((182,), {'O (feature 0)': '2.774 / 1.553 = 1.786 ✓'}, 'Line 182: divide once at the end. Full softmax of [2,1,3,0] · v = 0.237·1 + 0.087·3 + 0.644·2 + 0.032·0 = 1.786.'),
        ((147, 148, 150), {'m_run': 'NaN (0xFF bytes)', 'e_run, e_tile': 'e^(NaN − 2) = NaN', 'l_run': 'NaN·0 + … = NaN', 'O (feature 0)': 'NaN ✗'}, 'WHAT BREAKS (story S5, quiz kernel): seed m with memset 0xFF = NaN, not −∞. fmaxf hides it in m_new; exp does not.'),
    ]
    dbg('d06i-flash-dbg.svg', 'Online softmax inside the kernel, one row, two tiles', 'flash_lib.cu:131–182 (shortened; shfl_xor = __shfl_xor_sync(0xffffffff, …)); toy numbers from Day 6 §7.3', code, steps, V, codew=560, sec=4.2)

if __name__ == '__main__':
    for fn in (loadmap, tiled_dbg, race, edge, occupancy, timing, ladder, split, waves, ckpt_life, flash_tiles, flash_dbg): fn()
