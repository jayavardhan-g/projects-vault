"""Day 9 (interview edition) animations made with Manim Community (env `manim`, see Infrastructure/Environments).

Render (from this folder), then copy the MP4s next to the other figures:
    eval "$(mamba shell hook --shell bash)"; mamba activate manim
    manim -qm --media_dir <scratch>/media manim_d09.py RetrievalHeads ConvLeNet PruneRounds PGDPlane
    cp <scratch>/media/videos/manim_d09/720p30/*.mp4 ../manim/   (renamed d09m-*.mp4)

No LaTeX is installed, so everything is Text (Pango) — never MathTex, DecimalNumber or axis labels.
Numbers match Day 09 - CS728 and LeNet-5 (interview).md: the 11-token toy prompt (§2–§4), the conv toy and
LeNet shapes (§8), the layer-wise schedule and report Table 5 (§10–§11), the PGD toy and report Table 3 (§13).
"""
from manim import *

FONT, MONO = "Noto Sans", "Noto Sans Mono"
BG = "#0f1117"
config.background_color = BG
MUTE, HL, BAD, GOOD, BLUE_ = "#8b949e", "#ffd33d", "#ff7b72", "#3fb950", "#58a6ff"
PINK_, ORANGE_ = "#f778ba", "#f0883e"

TOK = ["<s>", "weather", "api", "flight", "search", "stock", "price", "calc", "math", "fly", "Paris"]
TCOL = [MUTE, BLUE_, BLUE_, GOOD, GOOD, HL, HL, PINK_, PINK_, MUTE, MUTE]


def T(s, size=24, color=WHITE, mono=False, bold=False):
    return Text(s, font=MONO if mono else FONT, font_size=size, color=color,
                weight=BOLD if bold else NORMAL)


def caption(s):
    t = T(s, 22, WHITE)
    if t.width > 13.4:
        t.scale_to_fit_width(13.4)
    return t.to_edge(DOWN, buff=0.35)


class RetrievalHeads(Scene):
    def construct(self):
        title = T("Retrieval heads → a tool retriever (no fine-tuning)", 30, bold=True).to_edge(UP, buff=0.3)
        self.play(FadeIn(title))
        boxes = VGroup()
        for t, c in zip(TOK, TCOL):
            r = RoundedRectangle(corner_radius=0.06, width=0.98, height=0.45, stroke_color=c, stroke_width=2,
                                 fill_color=c, fill_opacity=0.18)
            boxes.add(VGroup(r, T(t, 15).move_to(r)))
        boxes.arrange(RIGHT, buff=0.07).shift(UP * 2.2 + RIGHT * 0.55)
        tags = VGroup(*[T(x, 15, c).next_to(boxes[i:i + 2], UP, buff=0.08)
                        for x, c, i in (("tool A", BLUE_, 1), ("tool B", GOOD, 3), ("tool C", HL, 5), ("tool D", PINK_, 7), ("query", MUTE, 9))])
        cap = caption("A prompt that lists 4 tools, then the user's query 'fly Paris'. One forward pass reads it all.")
        self.play(FadeIn(boxes), FadeIn(tags), FadeIn(cap))
        self.wait(1.2)

        rows = {"L0H0": [.62, .03, .02, .05, .04, .02, .03, .02, .02, .07, .08],
                "L1H0": [.19, .02, .02, .41, .14, .02, .03, .02, .02, .06, .07],
                "L1H2": [.20, .07, .06, .10, .30, .03, .03, .05, .05, .05, .06]}
        bars_all = VGroup()
        for k, (name, w) in enumerate(rows.items()):
            y = 0.9 - k * 1.15
            lab = T(name, 18, MUTE, mono=True).move_to([boxes[0].get_left()[0] - 0.5, y, 0])
            bars = VGroup()
            for i, v in enumerate(w):
                b = Rectangle(width=0.7, height=max(v * 1.6, 0.02), stroke_width=0, fill_color=BLUE_, fill_opacity=0.85)
                b.move_to([boxes[i].get_x(), y - 0.4 + b.height / 2, 0])
                bars.add(b)
            am = max(range(11), key=lambda i: w[i])
            bars[am].set_fill(BAD if am == 0 else GOOD)
            bars_all.add(VGroup(lab, bars))
        self.play(Transform(cap, caption("Each head: one row of weights per query token (rows sum to 1). Shown: the row of 'Paris'.")))
        for g in bars_all:
            self.play(FadeIn(g), run_time=0.7)
        self.wait(1.5)
        self.play(Transform(cap, caption("L0H0 parks on <s> (the sink, in no tool). L1H0 puts most of its weight on tool B.")))
        self.wait(2.2)

        # Hit Count
        self.play(FadeOut(bars_all))
        heads = ["L0H0", "L0H1", "L0H2", "L1H0", "L1H1", "L1H2"]
        hits = [[0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [1, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 1, 0], [1, 1, 1, 0, 1]]
        cards = VGroup()
        for h in heads:
            r = RoundedRectangle(corner_radius=0.08, width=1.8, height=1.3, stroke_color=MUTE, stroke_width=1.5)
            cards.add(VGroup(r, T(h, 18, mono=True).move_to(r.get_top() + DOWN * 0.3)))
        cards.arrange_in_grid(2, 3, buff=(0.35, 0.3)).shift(DOWN * 0.4)
        nums = VGroup(*[T("0", 34, bold=True).move_to(c[0].get_center() + DOWN * 0.18) for c in cards])
        self.play(Transform(cap, caption("Hit Count: per training query, each head scores every tool (mean attention from the query). Hit = correct tool wins.")),
                  FadeIn(cards), FadeIn(nums))
        cnt = [0] * 6
        for p in range(5):
            new = VGroup()
            for i in range(6):
                cnt[i] += hits[i][p]
                new.add(T(str(cnt[i]), 34, GOOD if hits[i][p] else WHITE, bold=True).move_to(nums[i]))
            plab = T(f"probe {p + 1} of 5", 20, HL).next_to(cards, UP, buff=0.15)
            self.play(Transform(nums, new), FadeIn(plab), run_time=0.8)
            self.remove(plab)
        self.wait(0.6)
        self.play(*[cards[i][0].animate.set_stroke(HL, 4) for i in (3, 5)],
                  Transform(cap, caption("Keep the heads with the most hits (here the top 2: L1H0 = 5, L1H2 = 4). You kept 10 of 512.")))
        self.wait(2.2)

        # retrieval on a new query
        self.play(FadeOut(cards), FadeOut(nums))
        self.play(Transform(cap, caption("Test query: score each tool = mean attention, query tokens → tool tokens, averaged over the chosen heads.")))
        names = ["A", "B", "C", "D"]; vals = [.055, .158, .031, .041]; cols = [BLUE_, GOOD, HL, PINK_]
        chart = VGroup()
        for k, (n, v, c) in enumerate(zip(names, vals, cols)):
            x = -3 + k * 2
            base = Line([x - 0.6, -1.9, 0], [x + 0.6, -1.9, 0], color=MUTE)
            bar = Rectangle(width=1.0, height=0.01, stroke_width=0, fill_color=c, fill_opacity=0.9).move_to([x, -1.9, 0], aligned_edge=DOWN)
            chart.add(VGroup(base, bar, T("tool " + n, 20, c).next_to(base, DOWN, buff=0.12)))
        self.play(FadeIn(chart))
        self.play(*[chart[k][1].animate.stretch_to_fit_height(vals[k] * 16).move_to([-3 + k * 2, -1.9, 0], aligned_edge=DOWN) for k in range(4)], run_time=1.5)
        labs = VGroup(*[T(f"{vals[k]:.3f}", 18).next_to(chart[k][1], UP, buff=0.08) for k in range(4)])
        self.play(FadeIn(labs))
        crown = T("rank 1 = B ✓", 24, GOOD, bold=True).next_to(labs[1], UP, buff=0.15)
        self.play(FadeIn(crown), Transform(cap, caption("Rank tools by that score. The correct tool is first: a Recall@1 hit.")))
        self.wait(2)
        self.play(FadeOut(chart), FadeOut(labs), FadeOut(crown))
        rk = T("6 queries; rank of the correct tool:  1   3   1   7   2   1", 24, mono=True).shift(UP * 0.3)
        r1 = T("Recall@1 = share ranked 1st = 3 / 6 = 50%", 24, GOOD).next_to(rk, DOWN, buff=0.4)
        r5 = T("Recall@5 = share in the top 5 = 5 / 6 = 83%", 24, GOOD).next_to(r1, DOWN, buff=0.25)
        self.play(FadeIn(rk), Transform(cap, caption("Recall@k: the fraction of queries whose correct tool is in the top k.")))
        self.play(FadeIn(r1)); self.play(FadeIn(r5))
        res = T("Yours (5,000 queries, 100 tools): R@1 1.3% all heads → 40.7% top 10", 24, HL, bold=True).next_to(r5, DOWN, buff=0.5)
        self.play(FadeIn(res)); self.wait(3)


class ConvLeNet(Scene):
    def construct(self):
        title = T("Convolution, then LeNet-5 layer by layer", 30, bold=True).to_edge(UP, buff=0.3)
        self.play(FadeIn(title))
        cs = 0.6
        def grid(vals, fill=None, color=WHITE):
            g = VGroup()
            for i, row in enumerate(vals):
                for j, v in enumerate(row):
                    sq = Square(cs, stroke_color=MUTE, stroke_width=1.5, fill_color=fill or BG, fill_opacity=0.35 if fill else 1).move_to([j * cs, -i * cs, 0])
                    g.add(VGroup(sq, T("" if v is None else str(v), 22, color).move_to(sq)))
            return g
        img = grid([[0, 0, 1, 0, 0] for _ in range(5)]).move_to(LEFT * 4.2 + DOWN * 0.2)
        for i in range(5): img[i * 5 + 2][0].set_fill(BLUE_, 0.5)
        ker = grid([[-1, 0, 1] for _ in range(3)], fill=HL).move_to(LEFT * 0.4 + DOWN * 0.2)
        out = grid([[None] * 3 for _ in range(3)], fill=GOOD).move_to(RIGHT * 3.6 + DOWN * 0.2)
        labs = VGroup(T("image 5 × 5", 20, MUTE).next_to(img, UP), T("kernel 3 × 3 (9 weights)", 20, MUTE).next_to(ker, UP), T("output 3 × 3", 20, MUTE).next_to(out, UP))
        cap = caption("A convolution slides a small grid of weights over the image.")
        self.play(FadeIn(img), FadeIn(ker), FadeIn(out), FadeIn(labs), FadeIn(cap))
        win = Square(3 * cs, stroke_color=BAD, stroke_width=5).move_to(img[6])
        self.play(Create(win))
        vals = [3, 0, -3]
        texts = ["(−1·0 + 0·0 + 1·1) × 3 = 3", "(−1·0 + 0·1 + 1·0) × 3 = 0", "(−1·1 + 0·0 + 1·0) × 3 = −3"]
        for r in range(3):
            for c in range(3):
                self.play(win.animate.move_to(img[(r + 1) * 5 + c + 1]), run_time=0.45 if r else 0.8)
                v = T(str(vals[c]), 24, bold=True).move_to(out[r * 3 + c])
                if r == 0:
                    self.play(Transform(cap, caption("Multiply the 9 pairs and add: " + texts[c])), FadeIn(v), run_time=0.8)
                    self.wait(1.1)
                else:
                    self.play(FadeIn(v), run_time=0.3)
        self.play(Transform(cap, caption("+3 = dark-to-light edge, −3 = light-to-dark. The same 9 weights are reused everywhere.")))
        self.wait(2.2)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not title and m is not cap])

        stages = [("input", "1×28×28", 0), ("pad+norm", "1×32×32", 0), ("conv1", "6×28×28", 150), ("pool", "6×14×14", 0),
                  ("conv3", "16×10×10", 2400), ("pool", "16×5×5", 0), ("conv5", "120×1×1", 48000), ("fc6", "84", 10080), ("fc7", "10", 840)]
        col = VGroup()
        for n, sh, w in stages:
            r = RoundedRectangle(corner_radius=0.06, width=1.35, height=0.9, stroke_color=GOOD if w else MUTE, stroke_width=2)
            col.add(VGroup(r, T(n, 17, bold=True).move_to(r.get_center() + UP * 0.18), T(sh, 14, MUTE, mono=True).move_to(r.get_center() + DOWN * 0.2)))
        col.arrange(RIGHT, buff=0.12).shift(UP * 1.4)
        self.play(Transform(cap, caption("Your LeNet-5 (src/models.py): tanh after each conv/fc, average pooling, 10 logits out.")))
        for g in col: self.play(FadeIn(g), run_time=0.35)
        self.wait(1)
        mx = 48000
        wb = VGroup()
        for k, (n, sh, w) in enumerate(stages):
            if not w: continue
            bar = Rectangle(width=max(6.5 * w / mx, 0.04), height=0.38, stroke_width=0, fill_color=BAD if n == "conv1" else BLUE_, fill_opacity=0.9)
            row = VGroup(T(n, 18, mono=True), bar, T(f"{w:,} weights", 18))
            row.arrange(RIGHT, buff=0.2)
            wb.add(row)
        wb.arrange(DOWN, aligned_edge=LEFT, buff=0.2).shift(DOWN * 1.3 + LEFT * 1.5)
        self.play(FadeIn(wb), Transform(cap, caption("Weights per layer: conv1 has only 150 (0.24%); conv5 has 48,000 (78%). Total 61,470 + 236 biases.")))
        self.wait(3.5)
        self.play(Transform(cap, caption("Remember the 150. Removing the same FRACTION from every layer hits conv1 hardest.")))
        self.wait(2.5)


class PruneRounds(Scene):
    def construct(self):
        title = T("Iterative magnitude pruning: layer-wise vs global", 30, bold=True).to_edge(UP, buff=0.3)
        self.play(FadeIn(title))
        W = {"conv1": 150, "conv3": 2400, "conv5": 48000, "fc6": 10080, "fc7": 840}
        cur = dict(W); hist = [dict(cur)]
        for r in range(20):
            for k in cur: cur[k] = round(cur[k] * (0.9 if k == "fc7" else 0.8))
            hist.append(dict(cur))
        L = 7.0
        rows = VGroup(); fills = []
        for i, k in enumerate(W):
            y = 1.6 - i * 0.8
            lab = T(k, 22, mono=True).move_to([-5.6, y, 0])
            bg = Rectangle(width=L, height=0.45, stroke_width=0, fill_color="#30363d", fill_opacity=1).move_to([-1.0, y, 0])
            fg = Rectangle(width=L, height=0.45, stroke_width=0, fill_color=BAD if k == "conv1" else BLUE_, fill_opacity=0.9).move_to(bg, aligned_edge=LEFT)
            rows.add(VGroup(lab, bg, fg)); fills.append(fg)
        def txt(r):
            g = VGroup()
            for i, k in enumerate(W):
                g.add(T(f"{hist[r][k]:,} / {W[k]:,}", 18, BAD if k == "conv1" else WHITE).move_to([4.2, 1.6 - i * 0.8, 0], aligned_edge=LEFT))
            tot = sum(hist[r].values())
            g.add(T(f"round {r}:  {100 * tot / 61470:.2f}% of weights left", 26, HL, bold=True).move_to([0, 2.45, 0]))
            return g
        cap = caption("Each round: prune 20% of each layer's surviving weights (fc7: 10%), rewind, retrain 20 epochs.")
        lab = txt(0)
        self.play(FadeIn(rows), FadeIn(lab), FadeIn(cap))
        self.wait(1.5)
        stops = {9: "Round 9 (13.8%): conv1 has 21 weights. Clean accuracy is still flat here.",
                 15: "Round 15 (3.75%), the headline model: conv1 keeps 6 of 150; 3 of its 6 filters are empty.",
                 20: "Round 20 (1.30%): conv1 keeps 2 weights. Same fraction everywhere = the tiny layer starves."}
        for r in range(1, 21):
            anims = [fills[i].animate.stretch_to_fit_width(max(L * hist[r][k] / W[k], 0.02)).move_to(rows[i][1], aligned_edge=LEFT) for i, k in enumerate(W)]
            new = txt(r)
            self.play(*anims, Transform(lab, new), run_time=0.35)
            if r in stops:
                self.play(Transform(cap, caption(stops[r])))
                self.wait(2.2)
        self.play(FadeOut(rows), FadeOut(lab))
        hdr = T("at ~3.5% of weights, change from the dense model (report Table 5)", 22, MUTE).shift(UP * 2.3)
        cats = [("clean", -0.56, -0.18), ("MNIST-C", -3.89, -0.13)]
        g = VGroup()
        for k, (n, lw, gl) in enumerate(cats):
            x = -3 + k * 6
            base = Line([x - 2, 1.4, 0], [x + 2, 1.4, 0], color=MUTE)
            b1 = Rectangle(width=1.2, height=abs(lw) * 0.8, stroke_width=0, fill_color=BAD, fill_opacity=0.9).move_to([x - 0.8, 1.4, 0], aligned_edge=UP)
            b2 = Rectangle(width=1.2, height=max(abs(gl) * 0.8, 0.03), stroke_width=0, fill_color=GOOD, fill_opacity=0.9).move_to([x + 0.8, 1.4, 0], aligned_edge=UP)
            g.add(VGroup(base, b1, b2, T(n, 24, bold=True).next_to(base, UP, buff=0.15),
                         T(f"{lw:+.2f}", 20, BAD).next_to(b1, DOWN, buff=0.1), T(f"{gl:+.2f}", 20, GOOD).next_to(b2, DOWN, buff=0.1)))
        leg = VGroup(T("layer-wise 3.75%", 20, BAD), T("global 3.52%", 20, GOOD)).arrange(RIGHT, buff=0.8).shift(DOWN * 2.5)
        self.play(FadeIn(hdr), FadeIn(g), FadeIn(leg), Transform(cap, caption("Layer-wise: −0.56 clean but −3.89 MNIST-C, a 7× hidden cost (p = 0.019).")))
        self.wait(3)
        self.play(Transform(cap, caption("Global ranks all weights together; conv1 keeps 26.7%. The MNIST-C cost vanishes: −0.13, p = 0.83.")))
        self.wait(3.5)
        self.play(Transform(cap, caption("Same sparsity, same seeds: the recipe, not the sparsity, caused most of the damage.")))
        self.wait(3)


class PGDPlane(Scene):
    def construct(self):
        title = T("PGD on a 2-pixel toy, then your ε-sweep", 30, bold=True).to_edge(UP, buff=0.3)
        self.play(FadeIn(title))
        S, O = 9.0, np.array([-5.6, -3.0, 0])
        P = lambda x, y: O + np.array([x * S, y * S, 0])
        ax = VGroup(Line(P(0, 0), P(0.62, 0), color=MUTE), Line(P(0, 0), P(0, 0.6), color=MUTE),
                    T("x1", 20, MUTE).next_to(P(0.62, 0), RIGHT, buff=0.1), T("x2", 20, MUTE).next_to(P(0, 0.6), UP, buff=0.1))
        for v in (0.1, 0.2, 0.3, 0.4, 0.5):
            ax.add(T(f"{v}", 14, MUTE).next_to(P(v, 0), DOWN, buff=0.08), T(f"{v}", 14, MUTE).next_to(P(0, v), LEFT, buff=0.08))
        bound = Line(P(0, 0), P(0.3, 0.6), color=HL, stroke_width=4)
        blab = T("boundary 2·x1 − x2 = 0", 18, HL).next_to(P(0.28, 0.56), LEFT, buff=0.1)
        c0 = T("class 0 (correct)", 20, GOOD).move_to(P(0.53, 0.04)); c1 = T("class 1", 20, BAD).move_to(P(0.08, 0.45))
        cap = caption("Toy model: class 0 if 2·x1 − x2 > 0. The clean input (0.30, 0.20) is correct, with margin.")
        x0 = Dot(P(0.3, 0.2), color=WHITE, radius=0.09)
        self.play(FadeIn(ax), Create(bound), FadeIn(blab), FadeIn(c0), FadeIn(c1), FadeIn(x0), FadeIn(cap))
        self.wait(1.5)
        box = Rectangle(width=0.3 * S, height=0.3 * S, stroke_color=BLUE_, stroke_width=3).move_to(P(0.3, 0.2))
        self.play(Create(box), Transform(cap, caption("The attacker may move each pixel by at most ε = 0.15: a box around the input.")))
        self.wait(1.5)
        pts = [(0.3, 0.2), (0.26, 0.24), (0.22, 0.28), (0.18, 0.32)]
        cur = x0
        self.play(Transform(cap, caption("Each step: α = 0.04 in the sign of the input gradient, i.e. (−1, +1): raise the loss.")))
        for a, b in zip(pts, pts[1:]):
            arr = Arrow(P(*a), P(*b), buff=0, color=BAD, stroke_width=4, max_tip_length_to_length_ratio=0.35)
            d = Dot(P(*b), color=BAD, radius=0.07)
            self.play(GrowArrow(arr), FadeIn(d), run_time=0.7)
        raw = Dot(P(0.14, 0.36), color=ORANGE_, radius=0.07)
        arr = DashedLine(P(0.18, 0.32), P(0.14, 0.36), color=ORANGE_)
        self.play(Create(arr), FadeIn(raw), Transform(cap, caption("Step 4 would leave the box (x1 moved 0.16 > ε)…")))
        self.wait(1)
        proj = Dot(P(0.15, 0.35), color=BAD, radius=0.1)
        self.play(Transform(raw, proj), Transform(cap, caption("…so project it back onto the box: (0.15, 0.35). Score −0.05 < 0: fooled.")))
        self.wait(2.2)
        box2 = Rectangle(width=0.2 * S, height=0.2 * S, stroke_color=GOOD, stroke_width=3).move_to(P(0.3, 0.2))
        self.play(Create(box2), Transform(cap, caption("With ε = 0.10 the box ends at (0.20, 0.30): score 0.10, still correct. Bigger ε, more damage.")))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not title])

        cap = caption("Your PGD sweep (worst case of PGD-100 × 5 restarts), report Table 3: accuracy under attack.")
        eps = ["0.05", "0.1", "0.15", "0.3"]; dn = [51.98, 2.33, 0.01, 0.0]; pr = [82.18, 14.74, 0.13, 0.0]
        base = Line([-5.5, -1.8, 0], [5.5, -1.8, 0], color=MUTE)
        g = VGroup(base)
        for k in range(4):
            x = -4 + k * 2.7
            b1 = Rectangle(width=0.8, height=max(dn[k] * 0.045, 0.02), stroke_width=0, fill_color=BLUE_, fill_opacity=0.9).move_to([x - 0.45, -1.8, 0], aligned_edge=DOWN)
            b2 = Rectangle(width=0.8, height=max(pr[k] * 0.045, 0.02), stroke_width=0, fill_color=GOOD, fill_opacity=0.9).move_to([x + 0.45, -1.8, 0], aligned_edge=DOWN)
            g.add(b1, b2, T("ε = " + eps[k], 20).next_to(base, DOWN, buff=0.15).set_x(x),
                  T(f"{dn[k]:g}", 17, BLUE_).next_to(b1, UP, buff=0.06), T(f"{pr[k]:g}", 17, GOOD).next_to(b2, UP, buff=0.06))
        leg = VGroup(T("dense", 20, BLUE_), T("13.8% weights (layer-wise)", 20, GOOD)).arrange(RIGHT, buff=0.8).shift(UP * 2.2)
        self.play(FadeIn(g), FadeIn(leg), FadeIn(cap))
        self.wait(2.5)
        self.play(Transform(cap, caption("At ε = 0.05: 51.98% → 82.18% (p = 0.001). That is the +30 points on your resume.")))
        self.wait(2.5)
        self.play(Transform(cap, caption("But both fall to ~0 by ε = 0.2. Not a defence; adversarial training gives 90.55% at ε = 0.3.")))
        self.wait(2.5)
        self.play(Transform(cap, caption("Gradient masking ruled out: transfer from dense = 68.15% vs white-box 14.10% at ε = 0.1.")))
        self.wait(3)
