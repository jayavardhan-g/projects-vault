"""Day 5 (interview edition) animations made with Manim Community (env `manim`, see Infrastructure/Environments).

Render (from this folder), then copy the MP4s next to the other figures:
    eval "$(mamba shell hook --shell bash)"; mamba activate manim
    manim -qm --media_dir <scratch>/media manim_d05.py Coalescing TilingBlocks RegisterBlocking Ladder
    cp <scratch>/media/videos/manim_d05/720p30/*.mp4 ../manim/   (renamed d05m-*.mp4)

No LaTeX is installed, so everything is Text (Pango) — never MathTex, DecimalNumber or axis labels.
Numbers match Day 05 - CUDA I (interview).md: §3.3 warp addresses, §4.4 block sum, §6.3 outer products, §0.2 ladder.
"""
from manim import *

FONT, MONO = "Noto Sans", "Noto Sans Mono"
BG = "#0f1117"
config.background_color = BG
MUTE, HL, BAD, GOOD, BLUE_ = "#8b949e", "#ffd33d", "#ff7b72", "#3fb950", "#58a6ff"
ORANGE_ = "#f0883e"

A = [[1, 2, 0, 1], [0, 1, 2, 1], [2, 0, 1, 0], [1, 1, 0, 2]]
B = [[1, 0, 2, 1], [2, 1, 0, 0], [0, 1, 1, 2], [1, 2, 0, 1]]
C = [[6, 4, 2, 2], [3, 5, 2, 5], [2, 1, 5, 4], [5, 5, 2, 3]]


def T(s, size=24, color=WHITE, mono=False, bold=False):
    return Text(s, font=MONO if mono else FONT, font_size=size, color=color,
                weight=BOLD if bold else NORMAL)


def grid(vals, cell=0.5, size=22, color=WHITE, fill=None):
    """Number grid; cell (i, j) is g[i * cols + j]."""
    g = VGroup()
    cols = len(vals[0])
    for i, row in enumerate(vals):
        for j, v in enumerate(row):
            sq = Square(cell, stroke_color=MUTE, stroke_width=1.5,
                        fill_color=fill or BG, fill_opacity=1)
            sq.move_to([j * cell, -i * cell, 0])
            g.add(VGroup(sq) if v is None else VGroup(sq, T(str(v), size, color).move_to(sq)))
    return g.center()


def cells(g, cols, idx):
    return VGroup(*[g[i * cols + j] for i, j in idx])


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


# ─────────────────────────────────────────────────────────────────────────────
# 1. Coalescing (§3.3): block (2, 1), warp threadIdx.y = 3, step k = 0
# ─────────────────────────────────────────────────────────────────────────────
class Coalescing(Base):
    def construct(self):
        self.heading("Coalescing: same maths, different lanes")
        code = T("mm_naive:   row = blockIdx.x·32 + threadIdx.x     col = blockIdx.y·32 + threadIdx.y", 18, mono=True)
        code.move_to([0, 2.85, 0])
        self.play(FadeIn(code))

        y0, dy, x_dot = 1.95, 0.12, -4.5
        lane_col = [interpolate_color(ManimColor(BLUE_), ManimColor("#f778ba"), l / 31) for l in range(32)]
        dots = VGroup(*[Dot([x_dot, y0 - l * dy, 0], radius=0.04, color=lane_col[l]) for l in range(32)])
        ltop = T("lane 0", 16, MUTE).next_to(dots[0], LEFT, 0.12)
        lbot = T("lane 31", 16, MUTE).next_to(dots[31], LEFT, 0.12)
        warp = T("one warp = 32 lanes, one instruction", 15, MUTE).move_to([-4.5, 2.35, 0])

        LW, x_line = 4.0, -1.6                    # a 128-byte line = 32 floats
        cw = LW / 32
        lines = VGroup(*[Rectangle(width=LW, height=0.09, stroke_color=MUTE, stroke_width=1).move_to(
            [x_line + LW / 2, y0 - l * dy, 0]) for l in range(32)])
        lit = VGroup(*[Rectangle(width=cw, height=0.09, stroke_width=0, fill_color=BAD, fill_opacity=1).move_to(
            [x_line + cw / 2, y0 - l * dy, 0]) for l in range(32)])
        conn = VGroup(*[Line(dots[l].get_center(), lit[l].get_left(), stroke_width=1.2, color=lane_col[l])
                        for l in range(32)])
        lab = T("A[64 + lane][k]: 32 rows of A,\nN floats apart → 32 different\n128-byte lines, 4 B used of each", 17, BAD)
        lab.next_to(lines, RIGHT, 0.3).shift(0.6 * UP)
        bl = T("B[k][35]: same address for all lanes\n→ 1 line (broadcast)", 17, MUTE).next_to(lab, DOWN, 0.5, aligned_edge=LEFT)
        cnt = T("lines per warp step: 32 + 1 = 33", 22, BAD).next_to(bl, DOWN, 0.5, aligned_edge=LEFT)

        self.play(FadeIn(dots), FadeIn(ltop), FadeIn(lbot), FadeIn(warp))
        self.say("A warp = 32 lanes issuing one load together. The hardware fetches every 128-byte line their 32 addresses touch.", 2.0)
        self.play(Create(lines), run_time=1.2)
        self.play(LaggedStart(*[Create(c) for c in conn], lag_ratio=0.03), FadeIn(lit), run_time=1.8)
        self.play(FadeIn(lab))
        self.say("Naive: lane ℓ owns row 64 + ℓ. At step k each lane reads A[row][k]: one float from each of 32 lines.", 2.5)
        self.play(FadeIn(bl), FadeIn(cnt))
        self.say("33 lines for 64 useful floats. Most of every fetched line is wasted.", 2.0, BAD)

        shift = [l.animate.shift(cw * RIGHT) for l in lit]
        self.play(*shift, *[c.animate.put_start_and_end_on(dots[i].get_center(), lit[i].get_left() + cw * RIGHT)
                            for i, c in enumerate(conn)], run_time=1.0)
        self.say("Step k + 1 reads the NEXT float of the same 32 lines: some stay in L1. That's why 16.5× shrinks to a measured 10.9×.", 3.0)

        # coalesced
        code2 = T("mm_coalesced:   col = blockIdx.x·32 + threadIdx.x     row = blockIdx.y·32 + threadIdx.y", 18, mono=True, color=GOOD)
        code2.move_to(code)
        self.play(Transform(code, code2), FadeOut(lab), FadeOut(bl), FadeOut(cnt))
        self.say("Swap the two index lines: now lane ℓ owns column 64 + ℓ and reads B[k][64 + ℓ].", 1.5)
        big = Rectangle(width=LW, height=0.4, stroke_color=WHITE, stroke_width=1.5).move_to([x_line + LW / 2, 0.1, 0])
        bcells = VGroup(*[Rectangle(width=cw, height=0.4, stroke_width=0.5, stroke_color=BG, fill_color=lane_col[l],
                                    fill_opacity=1).move_to([x_line + cw * (l + 0.5), 0.1, 0]) for l in range(32)])
        newconn = VGroup(*[Line(dots[l].get_center(), bcells[l].get_left() + [cw / 2, 0, 0] + 0.2 * UP * 0,
                                stroke_width=1.2, color=lane_col[l]) for l in range(32)])
        self.play(ReplacementTransform(lines, big), ReplacementTransform(lit, bcells),
                  *[Transform(conn[l], newconn[l]) for l in range(32)], run_time=2.5)
        lab2 = T("B[k][64 … 95]: 32 adjacent floats\n= bytes 256 … 383 of row k\n= exactly ONE line, every byte used", 17, GOOD)
        lab2.next_to(big, RIGHT, 0.3).shift(1.1 * UP)
        bl2 = T("A[35][k]: same address for all lanes\n→ 1 line (broadcast)", 17, MUTE).next_to(lab2, DOWN, 0.5, aligned_edge=LEFT)
        cnt2 = T("lines per warp step: 1 + 1 = 2", 22, GOOD).next_to(bl2, DOWN, 0.5, aligned_edge=LEFT)
        self.play(FadeIn(lab2))
        self.play(FadeIn(bl2), FadeIn(cnt2))
        self.say("Same threads, same dot products, same instructions. Only which lane owns which output changed.", 2.5)
        res = T("33 → 2 lines: ≤ 16.5× predicted [THEORY]    140,486 → 12,927 ms = 10.9× measured [MEAS]", 20, HL)
        res.move_to([0.4, -2.55, 0])
        self.play(FadeIn(res))
        self.say("Rule: whatever varies with threadIdx.x must walk along contiguous memory.", 3.5)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tiling = block matrix multiplication (§4.4): TILE = 2, block (bx = 1, by = 0)
# ─────────────────────────────────────────────────────────────────────────────
class TilingBlocks(Base):
    def construct(self):
        self.heading("Tiling = block matrix multiplication")
        cs = 0.5
        gA = grid(A, cs).move_to([-3.8, -0.75, 0])
        gB = grid(B, cs).move_to([-1.2, 1.6, 0])
        gC = grid([[None] * 4 for _ in range(4)], cs).move_to([-1.2, -0.75, 0])
        lA = T("A", 24, bold=True).next_to(gA, UP, 0.12)
        lB = T("B", 24, bold=True).next_to(gB, LEFT, 0.15)
        lC = T("C = A · B", 22, bold=True).next_to(gC, DOWN, 0.12)

        def splits(g):
            c = g.get_center()
            return VGroup(Line(c + [-1, 0, 0], c + [1, 0, 0], color=WHITE, stroke_width=3),
                          Line(c + [0, -1, 0], c + [0, 1, 0], color=WHITE, stroke_width=3))

        self.play(FadeIn(gA), FadeIn(gB), FadeIn(gC), FadeIn(lA), FadeIn(lB), FadeIn(lC))
        self.play(Create(splits(gA)), Create(splits(gB)), Create(splits(gC)))
        self.say("TILE = 2: cut every 4 × 4 matrix into 2 × 2 tiles. Call: mm_tiled_1to1<2><<<dim3(2,2), dim3(2,2)>>>(dA, dB, dC, 4).", 2.0)

        tags = VGroup()
        for I in range(2):
            for J in range(2):
                tags.add(T(f"block\n({J},{I})", 13, MUTE).move_to(gC.get_corner(UL) + [cs * (2 * J + 1), -cs * (2 * I + 1), 0]))
        self.play(FadeIn(tags))
        self.say("One CUDA block per tile of C: block (bx, by) owns tile (by, bx). Its 2 × 2 threads own one entry each.", 2.0)
        own = SurroundingRectangle(cells(gC, 4, [(0, 2), (0, 3), (1, 2), (1, 3)]), color=HL, buff=0.03, stroke_width=4)
        self.play(Create(own), FadeOut(tags[2]))
        formula = T("C₀₁ = A₀₀ · B₀₁ + A₀₁ · B₁₁", 26, HL).move_to([3.5, 2.75, 0])
        self.play(FadeIn(formula))
        self.say("Follow block (1, 0). Block matrix multiplication: the same row-times-column rule, with tiles instead of numbers.", 2.5)

        # shared memory + accumulators panel
        sc = 0.55
        pos = {"sA": [2.2, 1.35, 0], "sB": [4.8, 1.35, 0], "P": [2.2, -0.95, 0], "acc": [4.8, -0.95, 0]}
        heads = VGroup(T("sA  (shared)", 18, MUTE).move_to(pos["sA"] + np.array([0, 0.85, 0])),
                       T("sB  (shared)", 18, MUTE).move_to(pos["sB"] + np.array([0, 0.85, 0])),
                       T("sA · sB, this step", 16, MUTE).move_to(pos["P"] + np.array([0, 0.85, 0])),
                       T("acc (registers)", 16, MUTE).move_to(pos["acc"] + np.array([0, 0.85, 0])))
        acc = grid([[0, 0], [0, 0]], sc, 24, HL).move_to(pos["acc"])
        self.play(FadeIn(heads), FadeIn(acc))

        steps = [  # (A tile cells, B tile cells, sA, sB, partial, acc after)
            ([(0, 0), (0, 1), (1, 0), (1, 1)], [(0, 2), (0, 3), (1, 2), (1, 3)],
             [[1, 2], [0, 1]], [[2, 1], [0, 0]], [[2, 1], [0, 0]], [[2, 1], [0, 0]], "A₀₀", "B₀₁"),
            ([(0, 2), (0, 3), (1, 2), (1, 3)], [(2, 2), (2, 3), (3, 2), (3, 3)],
             [[0, 1], [2, 1]], [[1, 2], [0, 1]], [[0, 1], [2, 5]], [[2, 2], [2, 5]], "A₀₁", "B₁₁"),
        ]
        sA_d = sB_d = P_d = None
        for t, (ia, ib, va, vb, vp, vacc, na, nb) in enumerate(steps):
            ha = SurroundingRectangle(cells(gA, 4, ia), color=BLUE_, buff=0.03, stroke_width=4)
            hb = SurroundingRectangle(cells(gB, 4, ib), color=ORANGE_, buff=0.03, stroke_width=4)
            self.play(Create(ha), Create(hb))
            cpA, cpB = cells(gA, 4, ia).copy(), cells(gB, 4, ib).copy()
            tA = grid(va, sc, 24, BLUE_).move_to(pos["sA"])
            tB = grid(vb, sc, 24, ORANGE_).move_to(pos["sB"])
            anims = [Transform(cpA, tA), Transform(cpB, tB)]
            if sA_d is not None:
                anims += [FadeOut(sA_d), FadeOut(sB_d), FadeOut(P_d)]
            self.play(*anims, run_time=1.5)
            sA_d, sB_d = cpA, cpB
            if t == 0:
                self.say(f"t = 0: each thread copies ONE element of {na} into sA and of {nb} into sB, then __syncthreads().", 2.0)
                r0 = SurroundingRectangle(VGroup(sA_d[0], sA_d[1]), color=GOOD, buff=0.02)
                c1 = SurroundingRectangle(VGroup(sB_d[1], sB_d[3]), color=GOOD, buff=0.02)
                self.play(Create(r0), Create(c1))
                self.say("Thread (tx = 1, ty = 0) owns C[0][3]: row 0 of sA · column 1 of sB = 1·1 + 2·0 = 1.", 2.5, GOOD)
                self.play(FadeOut(r0), FadeOut(c1))
            else:
                self.say(f"Barrier 2 first (nobody still reads the old tiles), then t = 1 loads {na} and {nb}.", 2.0)
            P_d = grid(vp, sc, 24, WHITE).move_to(pos["P"])
            self.play(FadeIn(P_d, shift=0.2 * DOWN))
            newacc = grid(vacc, sc, 24, HL).move_to(pos["acc"])
            self.play(Transform(acc, newacc), Indicate(P_d, color=HL), run_time=1.2)
            if t == 0:
                self.say("acc += sA · sB, each thread adding its own entry: [[2, 1], [0, 0]].", 1.5)
            else:
                self.say("acc = [[2, 1], [0, 0]] + [[0, 1], [2, 5]] = [[2, 2], [2, 5]]. Thread (1, 0): 1 + (0·2 + 1·1) = 2.", 2.0)
            self.play(FadeOut(ha), FadeOut(hb))

        fin = grid([[2, 2], [2, 5]], cs, 22, HL).move_to(cells(gC, 4, [(0, 2), (0, 3), (1, 2), (1, 3)]).get_center())
        self.play(TransformFromCopy(acc, fin), run_time=1.5)
        self.say("Guarded store: rows 0–1, cols 2–3 of C = [[2, 2], [2, 5]] ✓. The loop over t IS the sum over tiles.", 2.5, GOOD)
        self.say("Each element of a loaded tile is used by TILE threads: global loads per output fall from 2N to 2N / TILE.", 2.5)
        self.say("Real call: mm_tiled_1to1<32><<<(1024, 1024), (32, 32)>>>: each block sums 1,024 products of 32 × 32 tiles.", 3.5)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Register blocking (§6.3): mm_block2d<4,2>, thread (1, 0) owns C[0..1][2..3]
# ─────────────────────────────────────────────────────────────────────────────
class RegisterBlocking(Base):
    def construct(self):
        self.heading("Register blocking: a sum of outer products")
        cs = 0.5
        gA = grid(A, cs).move_to([-4.6, 0.4, 0])
        gB = grid(B, cs).move_to([-1.6, 0.4, 0])
        self.play(FadeIn(gA), FadeIn(gB), FadeIn(T("A", 24, bold=True).next_to(gA, UP, 0.12)),
                  FadeIn(T("B", 24, bold=True).next_to(gB, UP, 0.12)))
        self.say("mm_block2d<4,2><<<dim3(1,1), dim3(2,2)>>>: thread (tx = 1, ty = 0) owns a 2 × 2 patch, C rows 0–1, cols 2–3.", 2.0)

        sc = 0.62
        bvp, avp, prp, rgp = [3.0, 1.75, 0], [1.75, 0.5, 0], [3.0, 0.5, 0], [5.6, 0.5, 0]
        hb = T("bv = B[k][2…3]", 18, ORANGE_).move_to(np.array(bvp) + [0, 0.6, 0])
        ha = T("av =\nA[0…1][k]", 18, BLUE_).move_to(np.array(avp) + [-1.15, 0, 0])
        hp = T("av · bv", 18, MUTE).move_to(np.array(prp) + [0, -0.95, 0])
        hr = T("reg (4 registers)", 18, HL).move_to(np.array(rgp) + [0, 0.95, 0])
        reg = grid([[0, 0], [0, 0]], sc, 24, HL).move_to(rgp)
        kcnt = T("k = 0", 26, bold=True).move_to([-3.1, 2.45, 0])
        self.play(FadeIn(VGroup(hb, ha, hp, hr, reg)))
        av_d = bv_d = pr_d = None
        cur = [[0, 0], [0, 0]]
        for k in range(4):
            av = [A[0][k], A[1][k]]
            bv = [B[k][2], B[k][3]]
            pr = [[av[i] * bv[j] for j in range(2)] for i in range(2)]
            newk = T(f"k = {k}", 24, bold=True).move_to(kcnt)
            hA = SurroundingRectangle(cells(gA, 4, [(0, k), (1, k)]), color=BLUE_, buff=0.03, stroke_width=4)
            hB = SurroundingRectangle(cells(gB, 4, [(k, 2), (k, 3)]), color=ORANGE_, buff=0.03, stroke_width=4)
            anims = [Create(hA), Create(hB)] + ([FadeIn(kcnt)] if k == 0 else [Transform(kcnt, newk)])
            if av_d is not None:
                anims += [FadeOut(av_d), FadeOut(bv_d), FadeOut(pr_d)]
            self.play(*anims, run_time=0.8)
            cA = cells(gA, 4, [(0, k), (1, k)]).copy()
            cB = cells(gB, 4, [(k, 2), (k, 3)]).copy()
            av_d = grid([[av[0]], [av[1]]], sc, 24, BLUE_).move_to(avp)
            bv_d = grid([bv], sc, 24, ORANGE_).move_to(bvp)
            self.play(Transform(cA, av_d), Transform(cB, bv_d), run_time=1.0)
            av_d, bv_d = cA, cB
            pr_d = grid(pr, sc, 24).move_to(prp)
            self.play(LaggedStart(*[FadeIn(c, scale=0.6) for c in pr_d], lag_ratio=0.2), run_time=0.9)
            if k == 0:
                self.say("Per k: load 2 values of A (a column piece) and 2 of B (a row piece) into registers…", 1.5)
                self.say("…and multiply every av[i] with every bv[j]: an outer product, 4 FMAs from 4 loads.", 1.5)
            cur = [[cur[i][j] + pr[i][j] for j in range(2)] for i in range(2)]
            self.play(Transform(reg, grid(cur, sc, 24, HL).move_to(rgp)), Indicate(pr_d, color=HL), run_time=1.0)
            self.play(FadeOut(hA), FadeOut(hB), run_time=0.4)
            if k == 3:
                self.say("After k = 3: reg = [[2, 2], [2, 5]] = C rows 0–1, cols 2–3 ✓ (the same answer the tiled block produced).", 2.5, GOOD)

        self.play(*[FadeOut(m) for m in self.mobjects if m not in (self.head, self.cap)])
        rows = VGroup(
            T("tiling (§4):   for i, j:   for k:   acc += sA[i][k] · sB[k][j]", 22, mono=True),
            T("→ 2 shared loads per FMA  (roof ≈ 8.4 TFLOP/s)", 20, BAD),
            T("register blocking (§6):   for k:   load av[R], bv[R];   for i, j:   reg[i][j] += av[i] · bv[j]", 22, mono=True),
            T("→ 2R loads per R² FMAs", 20, GOOD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        rows.scale_to_fit_width(12.8).move_to([0, 1.75, 0])
        self.play(FadeIn(rows, lag_ratio=0.3), run_time=2)
        self.say("Same terms, same sum. Only the loop order changed: k moved outside, so each loaded value is reused R times.", 2.5)

        # R = 4 picture
        sc4 = 0.42
        av4 = VGroup(*[Square(sc4, stroke_color=BLUE_, fill_color=BLUE_, fill_opacity=0.5) for _ in range(4)]).arrange(DOWN, buff=0)
        bv4 = VGroup(*[Square(sc4, stroke_color=ORANGE_, fill_color=ORANGE_, fill_opacity=0.5) for _ in range(4)]).arrange(RIGHT, buff=0)
        pr4 = VGroup(*[Square(sc4, stroke_color=MUTE, fill_color=HL, fill_opacity=0) for _ in range(16)]).arrange_in_grid(4, 4, buff=0)
        pr4.move_to([-3.2, -1.25, 0])
        av4.next_to(pr4, LEFT, 0.12)
        bv4.next_to(pr4, UP, 0.12)
        self.play(FadeIn(av4), FadeIn(bv4), Create(pr4))
        self.play(LaggedStart(*[s.animate.set_fill(HL, 0.75) for s in pr4], lag_ratio=0.08), run_time=1.6)
        info = T("R = 4 (the real kernel, mm_block2d<64,4>):\n8 shared loads → 16 FMAs = 0.5 loads per FMA\nshared roof 8.4 → ≈ 33.5 TFLOP/s [THEORY]\nmeasured 25.71 TFLOP/s, 1.33× over strided [MEAS]", 20)
        info.next_to(pr4, RIGHT, 0.8)
        self.play(FadeIn(info))
        self.say("Cost: 16 accumulators + 8 operands → 62 registers per thread → 50% occupancy. It still wins: ILP replaces warps.", 4.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. The measured ladder (§0.2), N = 32768, FP32, H200
# ─────────────────────────────────────────────────────────────────────────────
class Ladder(Base):
    def construct(self):
        self.heading("The measured ladder: N = 32768, FP32, H200")
        rows = [("1  mm_naive", 0.50, "140,486 ms", BAD),
                ("2  mm_coalesced", 5.44, "12,927 ms", "#d29922"),
                ("3  mm_tiled_1to1<32>", 8.64, "8,148 ms", "#d29922"),
                ("4  mm_tiled_strided<64,32>", 19.36, "3,635 ms", BLUE_),
                ("5  mm_tiled_contiguous<64,32>", 18.57, "3,790 ms", MUTE),
                ("6  mm_block2d<64,2>", 16.51, "4,263 ms", MUTE),
                ("7  mm_block2d<64,4>", 25.71, "2,736.6 ms", GOOD),
                ("cuBLAS SGEMM", 53.37, "1,318.5 ms", "#a371f7")]
        x0, W, PEAK = -2.0, 8.0, 66.9
        s = W / PEAK
        ys = [2.45 - 0.6 * i for i in range(len(rows))]
        names = VGroup(*[T(n, 18, mono=True).move_to([x0 - 0.15, y, 0], aligned_edge=RIGHT) for (n, *_), y in zip(rows, ys)])
        axis = Line([x0, -2.15, 0], [x0 + W + 0.1, -2.15, 0], color=MUTE)
        ticks = VGroup(*[VGroup(Line([x0 + v * s, -2.15, 0], [x0 + v * s, -2.25, 0], color=MUTE),
                                T(str(v), 15, MUTE).move_to([x0 + v * s, -2.42, 0])) for v in range(0, 70, 10)])
        atitle = T("TFLOP/s", 16, MUTE).next_to(axis, RIGHT, 0.15)
        peak = DashedLine([x0 + PEAK * s, 2.85, 0], [x0 + PEAK * s, -2.15, 0], color=MUTE)
        plab = T("FP32 peak 66.9", 15, MUTE).next_to(peak, UP, 0.05)
        self.play(FadeIn(names), Create(axis), FadeIn(ticks), FadeIn(atitle), Create(peak), FadeIn(plab))

        caps = ["Naive: lanes walk rows of A → 33 cache lines per warp step. The baseline.",
                "Coalesced: swap two index lines → 2 lines per step. 10.9×.",
                "Shared 32 × 32 tiles: HBM loads ÷ 32, but 2 shared reads per FMA → a new roof ≈ 8.4. Measured 8.64. 1.59×.",
                "64-tile, 2 × 2 outputs per thread: fewer shared reads per FMA, HBM intensity 16 FLOP/B. 2.24×.",
                "Control: identical maths, stride-2 loader → only 4% slower. Global access barely matters at this point.",
                "block2d R = 2: 1,024 threads × 40 registers → 1 block per SM, contiguous loader. Slower.",
                "block2d R = 4: 4 × 4 outer products, 0.5 shared reads per FMA (roof ≈ 33.5). 25.71 TFLOP/s, 1.33×.",
                "cuBLAS: 53.37 TFLOP/s = 80% of peak (double buffering, vector loads, bigger tiles)."]
        bars = []
        for i, ((n, v, ms, col), y) in enumerate(zip(rows, ys)):
            bar = Rectangle(width=max(v * s, 0.02), height=0.36, stroke_width=0, fill_color=col, fill_opacity=0.9)
            bar.move_to([x0 + v * s / 2, y, 0])
            val = T(f"{v:.2f}   ({ms})", 16).next_to(bar, RIGHT, 0.12)
            self.play(GrowFromEdge(bar, LEFT), FadeIn(val), run_time=1.0)
            bars.append(bar)
            if i == 2:
                roof = Line([x0 + 8.4 * s, y + 0.19, 0], [x0 + 8.4 * s, y + 0.42, 0], color=BAD, stroke_width=5)
                rl = T("shared roof 8.4 [THEORY]", 14, BAD).next_to(val, RIGHT, 0.25)
                self.play(Create(roof), FadeIn(rl))
            if i == 6:
                roof = Line([x0 + 33.5 * s, y + 0.19, 0], [x0 + 33.5 * s, y + 0.42, 0], color=GOOD, stroke_width=5)
                rl = T("shared roof ≈ 33.5 [THEORY]", 14, GOOD).next_to(roof, RIGHT, 0.1)
                self.play(Create(roof), FadeIn(rl))
            self.say(caps[i], 2.2)

        bt = T("→ 48.2% of cuBLAS", 17, HL).move_to([3.4, ys[6], 0], aligned_edge=LEFT)
        self.play(FadeIn(bt), Indicate(bars[6], color=HL))
        self.say("Quote the chain, not one big number: 10.9× → 1.59× → 2.24× → 1.33×.", 2.5, HL)
        self.say("51.3× vs naive is against a strawman; 4.7× vs coalesced; the honest headline is 48.2% of cuBLAS.  [MEAS]", 4.0)
