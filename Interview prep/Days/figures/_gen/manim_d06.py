"""Day 6 (interview edition) animations made with Manim Community (env `manim`, see Infrastructure/Environments).

Render (from this folder), then copy the MP4s next to the other figures:
    eval "$(mamba shell hook --shell bash)"; mamba activate manim
    manim -qm --media_dir <scratch>/media manim_d06.py OnlineSoftmax BankConflict StreamSplits CheckpointArena
    cp <scratch>/media/videos/manim_d06/720p30/*.mp4 ../manim/   (renamed d06m-*.mp4)

No LaTeX is installed, so everything is Text (Pango) — never MathTex, DecimalNumber or axis labels.
Numbers match Day 06 - CUDA II (interview).md: §4.2 toy, §4.6 banks, §2.3 waves, §3.3 arena.
"""
from manim import *
import math

FONT, MONO = "Noto Sans", "Noto Sans Mono"
config.background_color = "#0f1117"
MUTE, HL, BAD, GOOD = "#8b949e", "#ffd33d", "#ff7b72", "#3fb950"
S_COL = ["#58a6ff", "#3fb950", "#d29922", "#f778ba"]          # stream colours
SEG_A, SEG_B, CKPT = "#58a6ff", "#f0883e", "#3fb950"           # checkpoint colours


def T(s, size=24, color=WHITE, mono=False, bold=False):
    return Text(s, font=MONO if mono else FONT, font_size=size, color=color,
                weight=BOLD if bold else NORMAL)


class Base(Scene):
    cap = None

    def heading(self, s):
        t = T(s, 32, bold=True).to_edge(UP, buff=0.3)
        self.play(FadeIn(t), run_time=0.6)
        self.head = t
        return t

    def say(self, s, wait=2.5, color=WHITE):
        new = T(s, 22, color)
        if new.width > 13.2:
            new.scale_to_fit_width(13.2)
        new.to_edge(DOWN, buff=0.3)
        if self.cap is None:
            self.play(FadeIn(new, shift=0.1 * UP), run_time=0.5)
        else:
            self.play(FadeOut(self.cap), FadeIn(new), run_time=0.5)
        self.cap = new
        self.wait(wait)

    def swap(self, old, new_text, **kw):
        """Replace a Text in place (keeps its left edge)."""
        new = T(new_text, **kw).move_to(old, aligned_edge=LEFT)
        self.play(Transform(old, new), run_time=0.6)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Online softmax (§4.2): s = [2, 1 | 3, 0], v = [1, 3 | 2, 0], answer 1.786
# ─────────────────────────────────────────────────────────────────────────────
class OnlineSoftmax(Base):
    def construct(self):
        self.heading("Online softmax: one row, two key tiles")
        s, v = [2, 1, 3, 0], [1, 3, 2, 0]

        cells = VGroup()
        for i in range(4):
            sq = Square(0.95, color=MUTE, stroke_width=2)
            cells.add(VGroup(sq, T(f"s = {s[i]}", 22).move_to(sq).shift(0.16 * UP),
                             T(f"v = {v[i]}", 18, MUTE).move_to(sq).shift(0.22 * DOWN)))
        cells.arrange(RIGHT, buff=0.12)
        cells[2:].shift(0.45 * RIGHT)
        cells.move_to([-3.2, 2.0, 0])
        tl = VGroup(T("tile 1", 18, MUTE).next_to(cells[:2], UP, 0.1),
                    T("tile 2", 18, MUTE).next_to(cells[2:], UP, 0.1))

        base_y, H = -2.2, 2.9
        axis = Line([cells.get_left()[0] - 0.2, base_y, 0], [cells.get_right()[0] + 0.2, base_y, 0], color=MUTE)
        ylab = T("weight  e^(s − m)", 18, MUTE).next_to(axis, DOWN, 0.12)

        def bar(i, w, col):
            r = Rectangle(width=0.62, height=w * H, fill_color=col, fill_opacity=0.85, stroke_width=0)
            return r.move_to([cells[i].get_x(), base_y + r.height / 2, 0])

        def wlab(r, txt):
            return T(txt, 18).next_to(r, UP, 0.08)

        # state panel
        names = ["m  (running max)", "l  = Σ e^(s − m)", "o  = Σ e^(s − m) · v", "o / l"]
        rows = VGroup(*[T(n, 22, MUTE) for n in names]).arrange(DOWN, aligned_edge=LEFT, buff=0.42)
        rows.move_to([0.4, 0.9, 0], aligned_edge=LEFT)
        vals = [T(x, 26, HL) for x in ["−∞", "0", "0", "?"]]
        for k in range(4):
            vals[k].move_to([3.2, rows[k].get_y(), 0], aligned_edge=LEFT)
        panel = Rectangle(width=6.6, height=rows.height + 0.7, color=MUTE).move_to([3.45, rows.get_y(), 0])
        ptitle = T("state kept per query row", 18, MUTE).next_to(panel, UP, 0.1)

        self.play(FadeIn(cells), FadeIn(tl), Create(axis), FadeIn(ylab))
        self.play(FadeIn(panel), FadeIn(ptitle), FadeIn(rows), *[FadeIn(x) for x in vals])
        self.say("Scores arrive one tile at a time. We never hold the whole row: only m, l and o.")

        box = SurroundingRectangle(cells[:2], color=HL, buff=0.08)
        self.play(Create(box))
        self.say("Tile 1: its max is 2, so each key gets weight e^(s − 2), at most 1.", 1.5)
        b0, b1 = bar(0, 1.0, S_COL[0]), bar(1, math.exp(-1), S_COL[0])
        l0, l1 = wlab(b0, "1.000"), wlab(b1, "0.368")
        self.play(GrowFromEdge(b0, DOWN), GrowFromEdge(b1, DOWN), FadeIn(l0), FadeIn(l1))
        self.swap(vals[0], "2", size=26, color=HL)
        self.swap(vals[1], "1.368", size=26, color=HL)
        self.swap(vals[2], "2.104", size=26, color=HL)
        self.say("l = 1 + 0.368 = 1.368      o = 1·1 + 0.368·3 = 2.104")

        self.play(box.animate.become(SurroundingRectangle(cells[2:], color=HL, buff=0.08)))
        self.say("Tile 2: score 3 beats the old max 2. Its own weights: e^(3 − 3) = 1 and e^(0 − 3) = 0.050.", 1.5)
        b2, b3 = bar(2, 1.0, S_COL[2]), bar(3, math.exp(-3), S_COL[2])
        l2, l3 = wlab(b2, "1.000"), wlab(b3, "0.050")
        self.play(GrowFromEdge(b2, DOWN), GrowFromEdge(b3, DOWN), FadeIn(l2), FadeIn(l3))
        self.wait(0.5)

        fac = T("× e^(m − m′)\n= × e^(2 − 3) = × 0.368", 20, BAD).move_to([VGroup(b0, b1).get_x(), 1.0, 0])
        self.say("Tile 1's weights were measured against m = 2. To use the new max m′ = 3, multiply them by e^(2 − 3).", 1.0)
        self.play(FadeIn(fac, shift=0.2 * DOWN))
        self.play(b0.animate.stretch_to_fit_height(math.exp(-1) * H, about_edge=DOWN),
                  b1.animate.stretch_to_fit_height(math.exp(-2) * H, about_edge=DOWN),
                  l0.animate.move_to([b0.get_x(), base_y + math.exp(-1) * H + 0.2, 0]),
                  l1.animate.move_to([b1.get_x(), base_y + math.exp(-2) * H + 0.2, 0]),
                  run_time=2.0)
        self.play(Transform(l0, T("0.368", 18).move_to(l0)), Transform(l1, T("0.135", 18).move_to(l1)))
        self.swap(vals[1], "1.368 × 0.368 → 0.503", size=22, color=BAD)
        self.swap(vals[2], "2.104 × 0.368 → 0.774", size=22, color=BAD)
        self.swap(vals[0], "3", size=26, color=HL)
        self.say("The same factor rescales the running l and o: one multiply each, the old scores are not needed.")

        self.play(FadeOut(fac))
        self.swap(vals[1], "0.503 + 1.050 → 1.553", size=22, color=HL)
        self.swap(vals[2], "0.774 + 2.000 → 2.774", size=22, color=HL)
        self.say("Add the new tile (its weights × e^(mₜ − m′) = × 1 here): l = 1.553, o = 2.774.")

        self.swap(vals[3], "2.774 / 1.553 → 1.786", size=22, color=GOOD)
        self.play(Indicate(vals[3], color=GOOD))
        self.say("Divide once at the end: 1.786. Direct softmax: [0.237, 0.087, 0.644, 0.032] · [1, 3, 2, 0] = 1.786 ✓", 3.0)

        code = VGroup(T("e_run = e^(m_run − m_new): rescales the old l and o", 18),
                      T("e_tile = e^(m_tile − m_new): scales the new tile", 18)
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        code.next_to(panel, DOWN, 0.35).align_to(panel, LEFT)
        self.play(FadeIn(code))
        self.say("That is all flash_lib.cu keeps per row: 3 numbers instead of N scores. Exact, not approximate.", 3.5)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bank conflict (§4.6): Kj stride 64 vs 65, lanes read feature x = 5
# ─────────────────────────────────────────────────────────────────────────────
class BankConflict(Base):
    def construct(self):
        self.heading("The Kj bank conflict: stride 64 vs stride 65")
        CW, CH, NK, X = 0.38, 0.2, 8, 5
        left, top = -6.08, 2.45
        key_fill = ["#1f3a5f", "#2d2f5e"]

        def pos(word):
            r, c = divmod(word, 32)
            return [left + CW * (c + 0.5), top - CH * (r + 0.5), 0]

        hdr = VGroup(*[T(str(b), 11, MUTE).move_to([left + CW * (b + 0.5), top + 0.15, 0]) for b in range(32)])
        bank_lbl = T("bank", 16, MUTE).next_to(hdr, LEFT, 0.15)
        cells, hot = VGroup(), VGroup()
        for c in range(NK):
            for x in range(64):
                sq = Rectangle(width=CW * 0.92, height=CH * 0.85, stroke_width=0,
                               fill_color=HL if x == X else key_fill[c % 2], fill_opacity=1)
                sq.move_to(pos(c * 64 + x))
                sq.cx = (c, x)
                cells.add(sq)
                if x == X:
                    hot.add(sq)
        keylbl = VGroup(*[T(f"key {c}", 14, MUTE).move_to([left - 0.55, top - CH * (2 * c + 1), 0])
                          for c in range(NK)])

        self.play(FadeIn(hdr), FadeIn(bank_lbl))
        self.play(FadeIn(cells, lag_ratio=0.002), FadeIn(keylbl), run_time=2)
        self.say("Shared memory has 32 banks; float word w lives in bank w mod 32. Kj stores 32 keys × 64 features (8 keys shown).")
        self.say("Stride 64: Kj[c][x] is word 64c + x. Each key fills exactly two rows of banks.", 2.0)

        col = SurroundingRectangle(VGroup(*hot), color=BAD, buff=0.04)
        self.play(Indicate(hot, color=HL, scale_factor=1.15), run_time=1.2)
        self.play(Create(col))
        self.say("Score loop, feature x = 5: lane c reads Kj[c][5]. Bank (64c + 5) mod 32 = 5 for every c.", 2.0)
        self.say("32 lanes, 32 different addresses, ONE bank → the hardware serves them one after another: 32 passes.", 2.5, BAD)

        # restride to 65
        self.play(FadeOut(col), FadeOut(keylbl))
        self.say("The fix (one line, KS = D + 1): pad every row by one unused float. Stride 65.", 1.0)
        moves = [sq.animate.move_to(pos(sq.cx[0] * 65 + sq.cx[1])) for sq in cells]
        pads = VGroup(*[Rectangle(width=CW * 0.92, height=CH * 0.85, stroke_width=0, fill_color="#444c56",
                                  fill_opacity=1).move_to(pos(c * 65 + 64)) for c in range(NK)])
        self.play(*moves, run_time=3.0)
        self.play(FadeIn(pads))
        keylbl2 = VGroup(*[T(f"key {c}", 14, MUTE).move_to([left - 0.55, pos(c * 65)[1], 0]) for c in range(NK)])
        self.play(FadeIn(keylbl2))
        rings = VGroup(*[SurroundingRectangle(sq, color=GOOD, buff=0.03) for sq in hot])
        self.play(Create(rings), Indicate(hot, color=HL, scale_factor=1.15), run_time=1.5)
        self.say("Now Kj[c][5] is word 65c + 5 → bank (c + 5) mod 32. Every key shifts one bank: a diagonal.", 2.5, GOOD)

        # bank load histogram
        self.play(FadeOut(VGroup(cells, pads, rings, keylbl2, hdr, bank_lbl)))
        bw, by0, sh = 0.36, -2.4, 0.12
        slots = VGroup(*[Line([left + bw * b + 0.03, by0, 0], [left + bw * (b + 1) - 0.03, by0, 0], color=MUTE)
                         for b in range(32)])
        nums = VGroup(*[T(str(b), 11, MUTE).move_to([left + bw * (b + 0.5), by0 - 0.15, 0]) for b in range(32)])
        title = T("requests per bank for one warp load of Kj[c][5], c = 0 … 31", 20, MUTE).move_to([0, 2.4, 0])
        reqs = VGroup()
        for c in range(32):
            r = Square(sh * 0.9, stroke_width=0, fill_color=interpolate_color(ManimColor("#58a6ff"), ManimColor("#f778ba"), c / 31),
                       fill_opacity=1)
            r.move_to([left + bw * 5.5, by0 + sh * (c + 0.5), 0])
            reqs.add(r)
        self.play(FadeIn(title), Create(slots), FadeIn(nums))
        self.play(LaggedStart(*[FadeIn(r, shift=0.3 * DOWN) for r in reqs], lag_ratio=0.05), run_time=2)
        tower = T("bank 5: 32 requests\n= 32 serial passes", 22, BAD).next_to(reqs, RIGHT, 0.4)
        self.play(FadeIn(tower))
        self.say("Stride 64: the whole warp queues on bank 5, on every one of the 64 feature steps of the hottest loop.", 2.5)
        self.play(FadeOut(tower))
        self.play(*[r.animate.move_to([left + bw * ((c + 5) % 32 + 0.5), by0 + sh * 0.5, 0]) for c, r in enumerate(reqs)],
                  run_time=2.5)
        flat = T("stride 65: one request per bank = 1 pass", 22, GOOD).move_to([0, 0.2, 0])
        self.play(FadeIn(flat))
        self.say("Stride 65: lane c → bank (c + 5) mod 32. All 32 banks work in parallel.", 2.0)
        res = T("Measured: 2,377 → 339 ms (7.0×) from that one line.  Not 32×: this loop was the worst cost, not the only one.", 20, HL)
        res.move_to([0, 1.2, 0])
        self.play(FadeIn(res))
        self.say("Same stride for store and load, so results are unchanged. Alternative: an XOR swizzle (keeps 16-byte alignment).", 3.5)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Stream splits (§2): baseline vs batch split vs column split, waves and W reads
# ─────────────────────────────────────────────────────────────────────────────
class StreamSplits(Base):
    def construct(self):
        self.heading("Batch split vs column split: who re-reads W?")
        Wc, Ww, Wh = [1.9, 1.15, 0], 3.6, 2.4
        Yc, Yh = [1.9, -1.55, 0], 1.6
        Xc, Xw = [-2.4, -1.55, 0], 2.4
        W = Rectangle(width=Ww, height=Wh, color=WHITE, stroke_width=2).move_to(Wc)
        Y = Rectangle(width=Ww, height=Yh, color=WHITE, stroke_width=2).move_to(Yc)
        X = Rectangle(width=Xw, height=Yh, color=WHITE, stroke_width=2).move_to(Xc)
        Wl = T("W   N × N\n4.29 GB at\nN = 32768", 18, MUTE).next_to(W, RIGHT, 0.25)
        Xl = T("X   32 × N   (4 MB)", 18, MUTE).next_to(X, DOWN, 0.12)
        Yl = T("Y = X · W   32 × N", 18, MUTE).next_to(Y, DOWN, 0.12)
        self.play(Create(W), Create(X), Create(Y), FadeIn(Wl), FadeIn(Xl), FadeIn(Yl))

        # one block
        row = Rectangle(width=Xw, height=Yh / 32, fill_color=HL, fill_opacity=0.9, stroke_width=0)
        row.move_to([Xc[0], Yc[1] + Yh / 2 - Yh / 64 - 7 * Yh / 32, 0])
        colslab = Rectangle(width=Ww / 16, height=Wh, fill_color=HL, fill_opacity=0.5, stroke_width=0)
        colslab.move_to([Wc[0] - Ww / 2 + Ww / 32 + 5 * Ww / 16, Wc[1], 0])
        blk = Rectangle(width=Ww / 16, height=Yh / 32, fill_color=HL, fill_opacity=1, stroke_width=0)
        blk.move_to([colslab.get_x(), row.get_y(), 0])
        self.play(FadeIn(row), FadeIn(colslab))
        self.play(FadeIn(blk, scale=3))
        self.say("One mm_thin block = one row m of X × a 256-column slice of W → 256 outputs of Y.", 2.0)
        self.say("All 32 rows need the SAME W slice. If their blocks run together, L2 serves it; if not, HBM again.", 2.5)
        self.play(FadeOut(row), FadeOut(colslab), FadeOut(blk))

        # block grid over Y: 4 row bands (≈ one wave each) × 16 column chunks
        def grid_cell(rb, cc, col, op=0.75):
            r = Rectangle(width=Ww / 16 * 0.92, height=Yh / 4 * 0.9, fill_color=col, fill_opacity=op, stroke_width=0)
            return r.move_to([Yc[0] - Ww / 2 + Ww / 32 + cc * Ww / 16, Yc[1] + Yh / 2 - Yh / 8 - rb * Yh / 4, 0])

        counter = T("W streamed from HBM: 0×", 22, HL).to_edge(LEFT, buff=0.4).set_y(2.3)
        hdr = T("baseline: one launch, 4,096 blocks ≈ 3.9 waves", 22, bold=True).to_edge(LEFT, buff=0.4).set_y(2.85)
        self.play(FadeIn(hdr), FadeIn(counter))
        self.say("Baseline: blocks are dispatched column-chunk first, so a wave (~1,056 blocks) = ~8 rows × ALL columns.", 1.0)
        for wv in range(4):
            band = VGroup(*[grid_cell(wv, cc, S_COL[wv]) for cc in range(16)])
            wglow = Rectangle(width=Ww, height=Wh, fill_color=S_COL[wv], fill_opacity=0.35, stroke_width=0).move_to(Wc)
            self.play(LaggedStart(*[FadeIn(b) for b in band], lag_ratio=0.06), FadeIn(wglow), run_time=1.2)
            self.play(FadeOut(wglow), Transform(counter, T(f"W streamed from HBM: ~{wv + 1}×", 22, HL).move_to(counter, aligned_edge=LEFT)),
                      run_time=0.6)
        self.say("Each wave sweeps all 4.29 GB of W (L2 is ~50 MB), so by the next wave it's evicted: W comes from HBM ~4 times.", 2.5)
        self.play(FadeIn(T("measured 44.43 ms  [MEAS]", 20).to_edge(LEFT, buff=0.4).set_y(1.75)))
        self.wait(1)
        self.play(*[FadeOut(m) for m in self.mobjects if m not in (W, X, Y, Wl, Xl, Yl, self.cap, self.head)])

        # batch split
        hdr = T("batch split: stream s takes 8 rows, all of W", 22, bold=True).to_edge(LEFT, buff=0.4).set_y(2.85)
        counter = T("W streamed from HBM: 0×", 22, HL).to_edge(LEFT, buff=0.4).set_y(2.3)
        self.play(FadeIn(hdr), FadeIn(counter))
        for s in range(4):
            xs = Rectangle(width=Xw, height=Yh / 4 * 0.9, fill_color=S_COL[s], fill_opacity=0.75, stroke_width=0)
            xs.move_to([Xc[0], Yc[1] + Yh / 2 - Yh / 8 - s * Yh / 4, 0])
            band = VGroup(*[grid_cell(s, cc, S_COL[s]) for cc in range(16)])
            wglow = Rectangle(width=Ww, height=Wh, fill_color=S_COL[s], fill_opacity=0.35, stroke_width=0).move_to(Wc)
            self.play(FadeIn(xs), FadeIn(wglow), LaggedStart(*[FadeIn(b) for b in band], lag_ratio=0.05), run_time=1.1)
            self.play(FadeOut(wglow), Transform(counter, T(f"W streamed from HBM: ~{s + 1}×", 22, HL).move_to(counter, aligned_edge=LEFT)),
                      run_time=0.5)
        self.say("Each stream's kernel is 8 rows × all columns: W is swept 4 times, same as the baseline, plus 4 host syncs.", 2.5)
        self.play(FadeIn(T("measured 52.28 ms (slower)  [MEAS]", 20).to_edge(LEFT, buff=0.4).set_y(1.75)))
        self.wait(1)
        self.play(*[FadeOut(m) for m in self.mobjects if m not in (W, X, Y, Wl, Xl, Yl, self.cap, self.head)])

        # column split
        hdr = T("column split: stream s takes ¼ of W, all 32 rows", 22, bold=True).to_edge(LEFT, buff=0.4).set_y(2.85)
        counter = T("W streamed from HBM: 0", 22, HL).to_edge(LEFT, buff=0.4).set_y(2.3)
        self.play(FadeIn(hdr), FadeIn(counter))
        parts = ["¼", "½", "¾", "1×"]
        xall = Rectangle(width=Xw, height=Yh, fill_color=MUTE, fill_opacity=0.35, stroke_width=0).move_to(Xc)
        self.play(FadeIn(xall))
        for s in range(4):
            wslab = Rectangle(width=Ww / 4, height=Wh, fill_color=S_COL[s], fill_opacity=0.45, stroke_width=0)
            wslab.move_to([Wc[0] - Ww / 2 + Ww / 8 + s * Ww / 4, Wc[1], 0])
            slab = VGroup(*[grid_cell(rb, 4 * s + k, S_COL[s]) for rb in range(4) for k in range(4)])
            self.play(FadeIn(wslab), LaggedStart(*[FadeIn(b) for b in slab], lag_ratio=0.03), run_time=1.1)
            self.play(Transform(counter, T(f"W streamed from HBM: ~{parts[s]}", 22, HL).move_to(counter, aligned_edge=LEFT)), run_time=0.5)
        self.say("One column-split kernel = 32 rows × ¼ of the columns = 1,024 blocks = ONE wave: all 32 rows of a W slice run together.", 2.5)
        self.say("So each W line comes from HBM about once and the other 31 rows hit L2. Layer 2 waits at a barrier: it needs all of A1.", 2.5)
        self.play(FadeIn(T("measured 25.65 ms = 1.73× faster  [MEAS]", 20, GOOD).to_edge(LEFT, buff=0.4).set_y(1.75)))
        note = T("The locality story is a hypothesis [RECON] (no profiler).\nTest: run the 4 chunks on ONE stream.\n~26 ms → locality.   ~44 ms → concurrency.", 19, MUTE)
        note.to_edge(LEFT, buff=0.4).set_y(0.55)
        self.play(FadeIn(note))
        self.say("Interview line: \"a decomposition win, most likely L2 locality; I couldn't prove it's concurrency.\"", 3.5)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Checkpoint arena (§3.3): keep-all 400 MiB vs checkpointed 272 MiB
# ─────────────────────────────────────────────────────────────────────────────
class CheckpointArena(Base):
    def construct(self):
        self.heading("Gradient checkpointing: one arena, two segments")
        k = 0.0105                        # units per MiB
        bot = -2.65
        bx, cx, w = 1.6, 4.8, 2.2         # x of the two stacks

        def blk(name, mib, col, x, y0):
            r = Rectangle(width=w, height=mib * k, fill_color=col, fill_opacity=0.8, stroke_color=WHITE, stroke_width=1)
            r.move_to([x, y0 + mib * k / 2, 0])
            lab = T(f"{name}  {mib}", 18 if mib >= 64 else 11).move_to(r)
            return VGroup(r, lab)

        hb = T("keep everything", 22, bold=True).move_to([bx, 2.75, 0])
        hc = T("checkpointed", 22, bold=True).move_to([cx, 2.75, 0])
        unit = T("MiB, config N4096", 16, MUTE).move_to([3.2, 2.4, 0])
        floor = Line([bx - w / 2 - 0.1, bot, 0], [cx + w / 2 + 0.1, bot, 0], color=MUTE)

        steps = ["forward",
                 "  L1   Z1, A1  ←  X",
                 "  L2   Z2, A2  ←  A1",
                 "  L3   Z3, A3  ←  A2",
                 "  L4   Z4  ←  A3",
                 "backward",
                 "  recompute Z3, A3  ←  A2",
                 "  L4   uses Z4, A3",
                 "  L3   uses Z3, A2",
                 "  rebuild Z1, A1, Z2  ←  X",
                 "  L2   uses Z2, A1",
                 "  L1   uses Z1, X"]
        lst = VGroup(*[T(s.strip(), 20, MUTE if s[0] != " " else WHITE) for s in steps])
        lst.arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([-4.4, 0.0, 0])
        for i, st in enumerate(steps):
            if st[0] == " ":
                lst[i].shift(0.35 * RIGHT)
        cur = SurroundingRectangle(lst[1], color=HL, buff=0.06)

        self.play(FadeIn(hb), FadeIn(hc), FadeIn(unit), Create(floor), FadeIn(lst))
        self.say("4-layer MLP training step. Backward needs each layer's Z (ReLU mask) and its input A. Sizes: 64 MiB each, Z4 16.", 2.5)

        # checkpointed skeleton: Z4, A2 own buffers, arena of 3 × 64
        cz4 = blk("Z4", 16, CKPT, cx, bot)
        ca2 = blk("A2", 64, CKPT, cx, bot + 16 * k)
        a0 = bot + 80 * k
        arena = DashedVMobject(Rectangle(width=w + 0.12, height=192 * k, color=WHITE).move_to([cx, a0 + 96 * k, 0]))
        alab = T("arena = max(192, 128)", 16, MUTE).next_to(arena, UP, 0.08)
        self.play(Create(arena), FadeIn(alab))
        self.say("Checkpointed: only A2 and Z4 get their own buffers. Everything else shares one arena of 192 MiB.", 2.0)

        def slot(i, name, col):
            return blk(name, 64, col, cx, a0 + i * 64 * k)

        base_order = [("Z1", 64, SEG_A), ("A1", 64, SEG_A), ("Z2", 64, SEG_A), ("A2", 64, CKPT),
                      ("Z3", 64, SEG_B), ("A3", 64, SEG_B), ("Z4", 16, CKPT)]
        y, bstack = bot, []
        for name, mib, col in base_order:
            bstack.append(blk(name, mib, col, bx, y))
            y += mib * k
        cs = {}

        self.play(Create(cur))
        cs["s0"], cs["s1"] = slot(0, "Z1", SEG_A), slot(1, "A1", SEG_A)
        self.play(FadeIn(bstack[0], shift=DOWN), FadeIn(bstack[1], shift=DOWN), FadeIn(cs["s0"], shift=DOWN),
                  FadeIn(cs["s1"], shift=DOWN))
        self.play(cur.animate.become(SurroundingRectangle(lst[2], color=HL, buff=0.06)))
        cs["s2"] = slot(2, "Z2", SEG_A)
        self.play(FadeIn(bstack[2], shift=DOWN), FadeIn(bstack[3], shift=DOWN), FadeIn(cs["s2"], shift=DOWN),
                  FadeIn(ca2, shift=DOWN))
        self.say("Forward, segment A: Z1, A1, Z2 go into the arena; A2 is a checkpoint.", 1.5)
        self.play(cur.animate.become(SurroundingRectangle(lst[3], color=HL, buff=0.06)))
        n0, n1 = slot(0, "Z3", SEG_B), slot(1, "A3", SEG_B)
        self.play(FadeIn(bstack[4], shift=DOWN), FadeIn(bstack[5], shift=DOWN),
                  Transform(cs["s0"], n0), Transform(cs["s1"], n1), run_time=1.5)
        self.say("Segment B: Z3, A3 are written OVER Z1, A1 (same addresses: rZ3 = rZ1 = arena). Segment A is gone.", 2.5, SEG_B)
        self.play(cur.animate.become(SurroundingRectangle(lst[4], color=HL, buff=0.06)))
        self.play(FadeIn(bstack[6], shift=DOWN), FadeIn(cz4, shift=DOWN))
        tb = T("400", 26, BAD, bold=True).next_to(VGroup(*bstack), UP, 0.18)
        tc = T("272", 26, GOOD, bold=True).next_to(arena, UP, 0.42)
        self.play(FadeIn(tb), FadeIn(tc))
        self.say("Peak activations: 7 buffers = 400 MiB  vs  64 + 16 + 192 = 272 MiB (−32%).", 2.5)

        self.play(cur.animate.become(SurroundingRectangle(lst[6], color=HL, buff=0.06)))
        self.play(Indicate(cs["s0"], color=SEG_B), Indicate(cs["s1"], color=SEG_B), Indicate(ca2, color=CKPT), run_time=1.2)
        self.say("Backward recomputes Z3, A3 from A2 into the arena. (Redundant in this harness: the arena still holds them — §3.6.)", 2.5)
        for i in (7, 8):
            self.play(cur.animate.become(SurroundingRectangle(lst[i], color=HL, buff=0.06)), run_time=0.6)
            self.wait(0.6)
        self.say("Layers 4 and 3 consume Z3, A3. After that line they are dead, so the arena can be reused.", 2.0)

        self.play(cur.animate.become(SurroundingRectangle(lst[9], color=HL, buff=0.06)))
        self.play(Transform(cs["s0"], slot(0, "Z1", SEG_A)), Transform(cs["s1"], slot(1, "A1", SEG_A)),
                  Indicate(cs["s2"], color=SEG_A), run_time=1.5)
        self.say("Rebuild segment A from X over the same arena. The order is enforced only by source order (comment at :513).", 2.5)
        for i in (10, 11):
            self.play(cur.animate.become(SurroundingRectangle(lst[i], color=HL, buff=0.06)), run_time=0.6)
            self.wait(0.5)

        # the 0 MB first version
        self.say("The first version gave each recomputed tensor its own buffer…", 1.0, BAD)
        segB = blk("Z3+A3", 128, SEG_B, cx, a0 + 192 * k)
        self.play(FadeOut(alab), FadeIn(segB, shift=DOWN), Transform(tc, T("400", 26, BAD, bold=True).next_to(segB, UP, 0.1)))
        self.say("…192 + 128 + 64 + 16 = 400 MiB: zero saved. Memory is saved by lifetimes not overlapping, not by recomputing.", 3.0)
        self.play(FadeOut(segB), FadeIn(alab), Transform(tc, T("272", 26, GOOD, bold=True).next_to(arena, UP, 0.42)))

        summ = VGroup(T("activations 400 → 272 MiB  (−32%)", 20, GOOD),
                      T("whole step 962 → 834 MiB  (−13.3%)", 20, WHITE),
                      T("backward 136.15 → 171.69 ms  (+26%)", 20, BAD),
                      T("= recomputed forward work: 0.9 × 39.5 ms", 18, MUTE)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        self.play(FadeOut(lst), FadeOut(cur))
        summ.move_to([-4.0, 0.3, 0])
        self.play(FadeIn(summ, lag_ratio=0.3), run_time=2)
        self.say("Say \"32% of activation memory, 13.3% of total\", and give the price: +26% backward time.  [MEAS]", 3.5)
