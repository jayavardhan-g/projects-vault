> Prep pack for **Speculative decoding**, written 2026-11-05 from `speculative_decoding/INTERVIEW_PREP.md`. The vault copy is the one to study and edit.

# Interview Prep — Speculative Decoding (Qwen2.5-0.5B drafting Qwen2.5-14B)

Repo: `/data/jayavardhan/projects/speculative_decoding`, branch `main`, HEAD `99ed1c2` (25 commits, 2026-08-13 to 2026-08-16).
Remote: `github.com/jayavardhan-g/speculative_decoding` (public).
Sources read: README.md, RESULTS.md, SETUP.md, CLAUDE.md (brief, untracked), PROGRESS.md (untracked), the full `git log`, all of `src/`, `scripts/`, and the key tests.
Nothing was run. No code was changed. This file is listed in `.git/info/exclude`, so git does not track it.

---

## 0. Read this first: five things that change how you talk about this project

1. **The "three draft models" are 0.5B, 1.5B and Coder-1.5B, not 7B.** The 7B is a *target*. It is the small-target comparison point (`configs/models.yaml:23-31`). If you say "0.5B, 1.5B and 7B drafts" in an interview, you will be wrong.
2. **0.88x → 2.25x is not one change.** The 0.88x was measured on the **7B target, eager** (commit `b08e59d`). The 2.25x is the **14B target, compiled**. Two fixes went in together (`a6ce931`). One was StaticCache plus torch.compile. The other was moving the target from 7B to 14B, which doubled the target/draft depth ratio. The 14B *eager* number was already 1.33x. Say "two fixes", not "compile fixed it".
3. **The 2.25x headline is one prompt, and it is a code prompt.** The prompt is "Write a Python function that checks whether a string is a palindrome." (`scripts/profile_overhead.py:37`). At the same depth, chat is 1.42x and summarisation is 1.41x. The mean across the three domains is 1.69x.
4. **The "bursty acceptance" story in the docs is probably a measurement artifact.** The cost model is fed `n_accepted / n_proposed`. That statistic counts proposals made *after* a rejection, which were never checked. It is not the per-position acceptance probability the formula needs. If you plug in the right statistic, the independence model fits the measurements to within about 5%. See §5.4. Do not claim "acceptance is bursty" as a finding.
5. **The repo cannot be re-run today.** The `speculative_decoding` mamba env no longer exists. `mamba env list` does not show it. There is no record of it in `/data/jayavardhan/env-registry/`. The exact versions (torch 2.13.0+cu130, transformers 5.15.0) appear only in the untracked PROGRESS.md. `pyproject.toml` has only floors (`torch>=2.4`, `transformers>=4.44`). The code depends on transformers-5 StaticCache internals (`layer.cumulative_length`, `cache_utils.py:163-166`). The weights are still on warm storage. Raw benchmark logs were never committed: `results/`, `logs/` and `*.log` are all gitignored, and none exist on disk.

---

## 1. Pitch

### 60-second pitch

I built speculative decoding from scratch in PyTorch and HuggingFace. A small Qwen2.5-0.5B model drafts four tokens. Then Qwen2.5-14B checks all four in one forward pass. A rejection-sampling rule keeps the output distribution exactly the same as running the 14B alone. I proved that rule and tested it statistically.

The first correct version was *slower* than the baseline, at 0.88x. Profiling showed why. At batch size 1, every model cost about 0.8 ms per layer, whatever its size. The GPU was idle between kernel launches. So a small draft was no cheaper than the big target. I made the decode step capturable as a CUDA graph. That meant a static KV cache, rollback by moving a counter, and fixed input shapes. I also moved to a deeper target. The result was 2.25x over a compiled baseline, and 2.60x on code at depth 8.

I also tested the original plan, which was switching between draft models by domain. The data showed the drafts never separate. So I cut that feature instead of shipping it.

### 3-minute deep explanation

**The problem.** A large model generates one token per forward pass. At batch size 1 that pass is slow and barely uses the GPU's compute. Speculative decoding uses a cheap draft model to guess γ tokens. The target scores all γ guesses in one pass. You keep the longest prefix the target agrees with, plus one extra token.

**Exactness.** At each position the draft proposes x from its distribution q. The engine accepts x with probability min(1, p(x)/q(x)), where p is the target's distribution. On rejection, it samples from the leftover mass (p − q)₊, renormalised. The emitted token is then distributed exactly as p, for any q. A bad draft costs speed, never correctness. I tested this three ways. First, a chi-square test on the accept/reject kernel with 60,000 samples. Second, a two-token joint distribution test on tiny random models. Third, greedy token-for-token equality on real weights.

**The performance bug.** My first working engine ran at 0.88x on a 7B target. Acceptance was healthy, at 0.84 to 0.92. Target passes dropped 4.3x. But wall-clock time got worse. I divided each model's forward time by its layer count. Eager mode gave 0.81 ms per layer for every model, across a 28x range in parameter count. That is Python dispatch and kernel-launch overhead, not compute. Two fixes followed. First, I moved the target to the 48-layer 14B, against the 24-layer 0.5B. Second, I made the step capturable. CUDA graphs need fixed tensor addresses, so I used a preallocated StaticCache. Rejected tokens are undone by moving the write offset back, with no data copy. That is sound because causal masking hides anything past the current position, and I tested it by poisoning the cache. Graphs also need fixed shapes. So I held back the last prompt token at prefill. That makes every verify exactly γ+1 wide and every draft step 1 wide, or 2 after a fully accepted round. Then `torch.compile(mode="reduce-overhead")` replays each step as one graph. I compiled the baseline too. Compilation alone doubles it, from 43.5 to 86.2 tok/s. Against that compiled baseline the 0.5B draft gives 2.25x. Work outside the forward passes is about 1.4% of an iteration.

**The numerics finding.** On the 14B, greedy output was not bit-equal to the baseline. I traced it to the GEMM batch width. Row 0 of a 5-wide forward differs from a 1-wide forward by 0.25 in the logits, about two bf16 ulps. On the 0.5B it is exactly zero. bf16 logits often tie, so two ulps can flip an argmax. The tests therefore measure this noise floor and allow a divergence only where the target's own top-1/top-2 margin falls inside it.

**The negative result.** The plan was to switch drafts by domain. I swept three drafts over code, chat and summarisation first. The 1.5B had the highest acceptance everywhere. The 0.5B had the highest throughput everywhere. The code-specialised draft tied the general one on code, 0.804 against 0.805. Acceptance tracks closeness to the *target's* distribution, not skill at the task. Depth did matter. Code peaked at γ=8, while chat and summarisation were best at γ=4 or lower. At γ=16, chat and summarisation ran slower than no speculation. So I shipped one draft at a fixed γ=4, the best single depth on an even mix, and cut the adaptive layer.

---

## 2. Architecture

### 2.1 Diagram

```
                    configs/models.yaml  (default_target: qwen2.5-14b; drafts: 1.5b, 0.5b, coder-1.5b)
                              |
                    src/specdec/models.py:103 load_model()      bf16, sdpa attention (models.py:113)
                              |                                  vocab_size = lm_head width (models.py:118)
                              v
 scripts/profile_overhead.py:234 main()  /  scripts/domain_sweep.py:204 main()  /  scripts/demo.py:34
        |   CachePool(max_length=2048)          cache_utils.py:70
        |   compile_decode(model, {1,2,γ+1})    compile_utils.py:45  -> torch.compile(reduce-overhead, fullgraph, dynamic=False)
        |   warmup_decode(...)                  compile_utils.py:110 -> 3 passes per shape before timing
        v
 +-------------------------------- SpeculativeDecoder (speculative.py:147) --------------------------------+
 |  __init__  vocab_size = min(target, draft) widths              speculative.py:178                       |
 |  generate()                                                    speculative.py:246                       |
 |    guard: every γ+1 width captured, else ValueError            speculative.py:280-293                   |
 |    acquire caches per (model, role)                            speculative.py:303-304 -> cache_utils:87 |
 |    PREFILL prompt[:-1] on target and draft (eager)             speculative.py:321-326                   |
 |    loop until max_new_tokens or EOS:                           speculative.py:337                       |
 |      1 DRAFT  γ x forward_with_cache(width 1 or 2)             speculative.py:341-356                   |
 |               logits_to_probs(align_vocab(.))  -> q, x~q       sampling.py:73, :42, :113                |
 |      2 VERIFY one target forward, width γ+1                    speculative.py:361-371                   |
 |               rows lo..lo+γ -> p_probs (γ+1, V)                                                         |
 |      3 ACCEPT verify_and_correct(proposals, q, p)              speculative.py:84-144                    |
 |               greedy: accept iff x == argmax p                 speculative.py:119-127                   |
 |               sampled: u < min(1,p/q) else draw (p-q)+         speculative.py:129-140                   |
 |               all accepted: bonus token from p[γ]              speculative.py:144                       |
 |      4 COMMIT cur_ids += accepted + 1                          speculative.py:388-393                   |
 |               crop_cache(target/draft, seq_len + n_acc)        speculative.py:394-396 -> cache_utils:124|
 |      (5 optional DepthController episode — parked, unverified) speculative.py:407-418, controller.py:79 |
 +---------------------------------------------------------------------------------------------------------+
        |
        v
 forward_with_cache (cache_utils.py:170): checks cache.length == start_pos, builds cache_position,
        _select_forward (cache_utils.py:215): compiled graph if width captured, else eager
        clones logits if graph-owned (cache_utils.py:212)
 crop_cache (cache_utils.py:124): static -> _rewind_static fills each layer's cumulative_length (cache_utils.py:150)

 Reference path: baseline.py:39 generate_baseline()  — width-1 target decode, same logits_to_probs, same vocab_size
 Parked (not shipped): stats.py:378 AcceptanceTracker, stats.py:88 AcceptanceEMA, controller.py:79 DepthController
```

### 2.2 One request, end to end

The input is the palindrome prompt at γ=4, greedy, compiled. Let L be the prompt length after the chat template.

1. `profile_overhead.py:257-261` applies the chat template and tokenises. `load_model` loads the 14B and the three drafts in bf16 (`profile_overhead.py:264-269`).
2. `CachePool(max_length=2048)` (`profile_overhead.py:272`). `compile_decode` captures widths {1, 2, 5} for every model. `warmup_decode` runs each shape three times so cudagraph trees finish recording (`compile_utils.py:132-138`).
3. `end_to_end` (`profile_overhead.py:165`) runs a discarded warm-up rep, then 3 reps round-robin: baseline, then each draft.
4. `SpeculativeDecoder.generate` (`speculative.py:246`):
   - Guard: target captured widths include 5 = γ+1, so it passes (`:280-293`).
   - `acquire` resets and returns the same StaticCache objects the graphs were recorded on (`cache_utils.py:87-102`).
   - **Prefill** L−1 tokens on both models, eager, because the width varies (`:321-326`). Both caches now hold L−1.
   - **Draft**: `pending = cur_ids[L-1:]`, one token. Four width-1 forwards. Each returns logits → `align_vocab` to 151936 → `logits_to_probs`, a one-hot argmax under greedy → `sample_from_probs` (`:341-353`). The draft cache now holds L+3. It never saw x4.
   - **Verify**: `t_len = L-1`. `feed = [last prompt token, x1, x2, x3, x4]`, width 5 (`:361-363`). Row j predicts the token after position L−1+j. `lo = 0`. `p_probs` has 5 rows (`:368-371`).
   - **Accept**: say x1 and x2 match the argmax and x3 does not. `verify_and_correct` returns `(2, argmax_p[2])` (`:119-127`).
   - **Commit**: emit [x1, x2, c]. `cur_ids` has length L+3. `keep = L+2`. The target cache is cropped from L+4 to L+2 and the draft cache from L+3 to L+2 (`:393-396`). For the static cache this is one `fill_` per layer (`cache_utils.py:163-166`).
   - **Next iteration**: the target is one position behind (L+2 against L+3). So the feed is again [c, 4 proposals], width 5. The draft feeds [c], width 1. If all 4 had been accepted, the draft would be two behind ([x4, bonus], width 2). That is why width 2 is captured (`speculative.py:195-197`).
   - The loop ends at 128 tokens. The output is trimmed to `max_new_tokens` (`:424`). Timing runs from just before prefill to the final sync (`:310-312`, `:420-422`).
5. The speedup is `max(spec tok/s over reps) / max(baseline tok/s over reps)` (`profile_overhead.py:229-231`).

---

## 3. Design decisions

Each entry gives the choice, the alternatives, the reason, the cost, and the source.

| # | Decision | Alternatives | Why | Cost | Source |
|---|---|---|---|---|---|
| D1 | Qwen2.5 family for target and drafts | Llama 3.1 8B + SmolLM2-135M (the brief's plan) | SmolLM2's tokenizer (49k) is unrelated to Llama's (128k). p(x)/q(x) over different id spaces is meaningless. Qwen2.5 gives a size ladder *and* a code sibling on one tokenizer. | Tied to one family. The code sibling later proved useless. | Documented: `e6332ce`, `configs/models.yaml:1-15`, PROGRESS §3 |
| D2 | Canonical vocab = **narrowest** embedding width (151936) | Widest (152064); per-model width | The target's token is fed back into the draft. A wider target can emit an id the draft has no row for, which raises IndexError. "The first engine run hit exactly this." | Truncates the target's distribution by 128 padding ids. The baseline must apply the same truncation, or comparisons are void. | Documented: `e6332ce`, `sampling.py:42-70`, README "canonical vocabulary" |
| D3 | One `logits_to_probs` shared by baseline, draft and verify | Separate code per path; HF logits processors | If p is not bit-for-bit what the baseline samples from, exactness is void. Sharing makes it structural. | None really. Warper order is hand-matched to HF (temp → top-k → top-p). | Documented: `e6332ce`, `sampling.py:1-10` |
| D4 | Standard single-chain Leviathan/Chen accept-reject | Tree drafting (SpecInfer), Medusa heads, EAGLE | Simplest exact rule, testable as a pure function. Uses off-the-shelf drafts, with no training. The brief made trees a Phase-4 stretch "only if agreed". The scope cut cancelled it. | Leaves speedup on the table. Published tree and EAGLE methods report higher speedups on their own setups. | Documented: brief Phase 4, PROGRESS §0a/§1 (cancelled). Technical tradeoffs in §7 are my reasoning. |
| D5 | Batch size 1 only | Continuous batching | Get single-stream exactness right first. Batch interacts with per-request draft and depth choice. | Results do not transfer to serving. The launch-bound analysis is batch-1 specific. | Documented: `speculative.py:150-153`, RESULTS caveats |
| D6 | Custom decode loop on HF Transformers | vLLM / SGLang internals; HF `assistant_model` | The brief: "build on a lightweight framework rather than fighting vLLM internals". You need full control of the cache and verification. | Slower baseline than a production server. HF assisted generation was never compared. | Documented: CLAUDE.md "Tech Stack". The HF-assisted comparison gap is my observation. |
| D7 | `StaticCache` behind a `KVCache` handle, length tracked host-side | DynamicCache (grows with `torch.cat`); reading length from the device | CUDA graphs need fixed addresses. Reading length back from the GPU costs a sync per step, which is the overhead being removed. | Fixed `max_length` (2048). Overflow raises (`cache_utils.py:192-196`). Couples to transformers-5 internals. | Documented: `a6ce931`, `cache_utils.py:33-46` |
| D8 | Rollback = move the write counter, leave stale K/V in place | Zero the rejected slots; copy | Free, and needs no data movement. The next forward overwrites the slots, and causal masking hides anything beyond. | Correctness rests on the masking argument, so it is tested by poisoning (`test_static_cache.py:49`). | Documented: `a6ce931`, `cache_utils.py:124-166` |
| D9 | Prefill **prompt minus last token**, eager | Fold prefill into the first verify | Makes every verify exactly γ+1 wide and every draft step 1 or 2 wide, a small fixed set to capture. | One extra target forward per request (counted in `test_engine_correctness.py:209`). | Documented: `a6ce931`, `speculative.py:314-326` |
| D10 | `torch.compile(mode="reduce-overhead", fullgraph=True, dynamic=False)` | Hand-written CUDA graphs; Triton kernels; `mode="default"` | One graph replay per step collapses hundreds of launches. `fullgraph` refuses graph breaks, which would put Python back on the path. The brief says Triton only if profiling demands it, and it did not (1.4% overhead). | Minutes of compile time. One graph per width. Shape-specialised kernels change bf16 numerics (floor 0.281 vs 0.250). | Documented: `compile_utils.py:1-31, 45-91`, brief "Tech Stack" |
| D11 | Default target moved 7B → 14B | Keep the 7B | The eager cost was n_layers × constant. The 7B (28 layers) vs 1.5B (28 layers) gave draft cost ≈ target cost. The 14B (48) vs 0.5B (24) gives 2:1. | More VRAM and a slower load (20–40 min cold from NFS). The 7B stays only as a comparison. | Documented: `a6ce931`, `configs/models.yaml:16-31` |
| D12 | Compile **both** sides before quoting speedup | Compiled spec vs eager baseline | Otherwise you report the compiler's 1.98x as speculation's win. | A smaller headline number. | Documented: `40c2c00`, `speculative.py:215-220`, RESULTS "Setup" |
| D13 | `CachePool` keyed by `(id(model), role)` | One cache per decoder | Graphs are bound to cache addresses. Switching drafts must not force the target to re-capture. The role key lets a model draft for itself with two caches. | `id()`-keyed. A little indirection. | Documented: `a6ce931`, `cache_utils.py:70-102` |
| D14 | Fail loudly if a requested γ was not captured | Silently fall back to eager | A fallback is correct but several times slower, with nothing visible to explain it. | One more check per call. | Documented: `09e8bbe`, `speculative.py:274-293` |
| D15 | Distribution tests on tiny random Qwen2 models (vocab 8 and 6, head weights ×6) | Test on real 7B/14B | Chi-square needs tens of thousands of samples. A 14B gives a few dozen. The ×6 head scale makes p and q peaked and different, so a wrong rule actually shows. | Real-weight bf16 behaviour is covered only by greedy tests. | Documented: `e6332ce`, `tests/conftest.py:26-49` |
| D16 | Greedy correctness = **derived tolerance**: diverge only where the margin is ≤ the measured noise floor | Bit equality; "mostly agrees" / token error rate | Bit equality is impossible on the 14B (0.25 batch-width shift). A token error rate would hide real bugs. A misindexed verify or stale cache is wrong by whole units, not ulps. | More complex tests. The floor is measured on one prompt, row 0 only, widths {2, 5, 9} (`conftest.py:137-169`). | Documented: `942b9e6`, `6c9dbf1`, `217b8cd` |
| D17 | Greedy special case inside `verify_and_correct` | Run the general rule on one-hot p | Clarity, and no coin flip whose outcome is fixed. A test checks both give the same answer (`test_verify_kernel.py:139`). | None. | Documented in a code comment, `speculative.py:119-127` |
| D18 | Accept/reject loop in Python with per-token host syncs | Vectorised on-GPU verification | Measured 0.036 ms per iteration. Not on the critical path. | Would matter at larger γ or batch. | Reconstructed from the code (`speculative.py:115-140`) plus measured cost (README "Where the rest of the time goes") |
| D19 | Benchmark protocol: discarded warm-up rep, round-robin, 3 reps; best-of for headline, median for the sweep | Single pass; average including warm-up | A cold pass reported 0.49x and 7.44x for drafts that settle near 2x. An early 2.15x was entirely warm-up. | Best-of-3 is optimistic for both sides. Two different aggregators across documents. | Documented: `b08e59d`, `cc77005`, `profile_overhead.py:187-231`, `domain_sweep.py:115-152, 169` |
| D20 | Domain tag is an evaluation key only; the controller never sees it | Tag selects the draft; infer the domain with a classifier | A tag-driven controller is a lookup table. A classifier adds a second error source. Held out, the tag becomes free ground truth. | Parked with Phase 1. | Documented: `7fd1ff6`, `stats.py:1-20` |
| D21 | Track accepted-per-iteration with a bias-corrected EMA, two timescales (α 0.15 / 0.02), prior counted in observations | Per-token rate; Beta-Bernoulli posterior | Maps linearly to tokens per target pass. The docs reject Beta-Bernoulli because "acceptance is bursty". | **The burstiness evidence is weak (§5.4).** The choice is still fine, but the stated reason is shaky. | Documented: `7fd1ff6`, `stats.py:36-48, 88-187` |
| D22 | Depth controller: episodes of `dwell=16`, switch `margin=5%`, explore every 8, one sync per episode | Per-iteration timing; pure EMA; bandit | A per-iteration timer needs a sync each step. The margin stops slow drift between near-equal depths. Counts *preference* changes, not probes, as switches. | Never run on real weights. No test that changing γ mid-request preserves the distribution. | Documented: `00d6dd5`, `controller.py:1-75` |
| D23 | **Cut the adaptive layer**. Ship one draft (0.5B), fixed γ=4 | Finish the depth controller; keep draft switching | No separation by draft. Depth-adaptivity is worth ~7% over the best fixed depth on an even mix, not worth the complexity and unverified exactness. | Loses the "adaptive" story. Chat/summarisation optimum below γ=4 never measured. | Documented: `1f9d1ad`, PROGRESS §0a (user's decision, 2026-08-16) |
| D24 | Keep the parked controller and stats code in the tree | Delete | Record of the measurement that changed the design. Deletion was offered and declined. | The README opening and `__init__.py` docstring still read as an adaptive engine. The demo still defaults to the 1.5B draft (`demo.py:37`). | Documented: PROGRESS §0a |
| D25 | Benchmarks always generate 128 tokens with **no EOS stop** | Stop at `<|im_end|>` | Not stated anywhere. It gives equal token counts on both sides. | Short answers (e.g. "summarise in two sentences") likely run past EOS. Part of what gets timed may be post-EOS text. | **Reconstructed from the code**: `profile_overhead.py:175-176, 195-202`; `domain_sweep.py:108-111, 119-126` pass no `eos_token_ids` |

---

## 4. Hardest problems, as STAR stories

### S1. Correct, but slower than doing nothing (0.88x)
- **Situation.** Phase 0 passed every correctness test. On the 7B target, speculation ran at 0.88x with the 1.5B draft, 0.92x with the 0.5B and 0.82x with Coder. Baseline: 54.2 tok/s (`b08e59d`). Acceptance was 0.84–0.92 and target forwards dropped 4.3x (129 → 30).
- **Task.** Find why fewer target passes gave slower wall-clock time.
- **Action.** I wrote `scripts/profile_overhead.py` to time each model's forward pass and divide by layer count. Eager gave about 0.64 ms per layer on the 7B pool, and later 0.81 ms per layer for the whole 14B pool (0.813, 0.808, 0.820, 0.805). It was flat across a 28x range in size. The cost model predicted 0.87x and I measured 0.88x, so it was understood, not just observed. Fix: (a) a static KV cache with a host-side length and counter rollback, fixed shapes by holding back the last prompt token, and `torch.compile(reduce-overhead)` (`a6ce931`); (b) a 14B target for a 48:24 layer ratio.
- **Result.** One 0.5B decode step went from 16.9 ms to 1.59 ms (10.6x, `a6ce931`). End to end: 2.25x over a compiled baseline at γ=4 (`40c2c00`). Compiled per-layer cost now tracks size: 0.223 / 0.098 / 0.071 ms.

### S2. Greedy outputs stopped matching on the 14B
- **Situation.** After the move to the 14B, every greedy comparison reported MISMATCH, including eager ones that used to pass. On the 7B, 27 out of 27 had matched (`cc77005`, `8709c3c`).
- **Task.** Decide whether this was an engine bug or numerics, without loosening the test until it meant nothing.
- **Action.** First I fixed a real bug: the baseline was built without `vocab_size`, so it sampled over 152064 ids while the engine used 151936 (`cc77005`). Mismatches remained. I traced one rejection with the 0.5B drafting for itself and found an exact tie, 19.25 vs 19.25 (`c3104bb`). I tested the underlying property directly: does width-1 equal row j of width-(γ+1)? On the 14B it differs by exactly 0.25 at widths 2, 5 and 9. On the 0.5B it is 0.0 (`6c9dbf1`). I poisoned the static cache with ±100 values to rule out stale K/V. Then I made the tolerance derived: measure the floor, and allow divergence only at margins ≤ floor.
- **Result.** Eager floor 0.250, compiled floor 0.281. Compiled γ=1 diverged 0/9 times. Compiled γ=4 diverged 3/9 times, all at margin exactly 0.25, none above (`217b8cd`). 21/21 GPU tests green.

### S3. The warm-up that looked like a 2.15x result
- **Situation.** An early single-pass run reported Coder-1.5B at 2.15x on the 7B (`b08e59d`). Later a compiled run reported 0.49x and 7.44x for two drafts in one rep (`cc77005`).
- **Task.** Make the benchmark trustworthy.
- **Action.** I found that a cold baseline read 17.5 tok/s against a steady 54.2. I added round-robin ordering with repeats, a warm-up phase in `warmup_decode`, and a labelled, discarded first end-to-end rep.
- **Result.** The 2.15x was entirely warm-up. Steady state was 0.82x. The later compiled drafts settled at 2.03x and 1.91x.

### S4. A test failing on correct hardware: the wrong denominator
- **Situation.** The GPU suite came back 20/21. The self-drafting test read 0.932 against a 0.95 floor. The per-iteration counts were `[4,4,4,4,4,4,4,1,4,4,4]` (`66482df`).
- **Task.** Decide: engine bug, hardware noise, or test bug?
- **Action.** `acceptance_rate = n_accepted / n_proposed` adds γ every iteration. But verification stops at the first mismatch. So one flip at index 1 counts as three disagreements. I wrote `shape_agreement` (`conftest.py:108`): count γ for a full accept and n+1 otherwise. That gives 41 agreed out of 42 verified = 0.976. I added a CPU test pinning the exact failing counts. I also moved the assert out of the per-domain loop so all three domains report.
- **Result.** 21/21 on re-run. Lesson: the bias grows with γ, so a fixed floor on the old statistic was really a floor on depth. (The same bias still sits inside the cost model, see §5.4.)

### S5. The hypothesis the data killed
- **Situation.** The whole project brief was adaptive draft *switching* by domain.
- **Task.** Check the premise before building the controller.
- **Action.** I noticed that on code, Coder-1.5B accepted *less* than its general twin (0.825 vs 0.911). I wrote `scripts/domain_sweep.py`: 3 prompts × 3 domains × 3 drafts, compiled, warm-up discarded, median of 3 reps (`f0d3a86`). I also made "no data" and "no separation" distinct verdicts after a CPU test caught the script reporting "n/a wins everywhere".
- **Result.** No separation by draft at any depth. A large separation by depth: code 2.23x → 2.60x from γ=4 to 8, chat 1.47x → 1.26x, summarisation 1.41x → 1.13x (`863f0cc`). I redirected to a depth controller, then the scope was cut to fixed-draft (`1f9d1ad`).

### S6. A test that had been red for a day, unnoticed
- **Situation.** `test_registry_has_one_target_and_several_drafts` had failed since `a6ce931` added the 7B as a second target (`a35c413`).
- **Action.** I fixed the assertion. I also found the fixture resolved "the first target in the YAML", not `default_target`. So the tokenizer checks may have checked the 7B while the engine loaded the 14B. I made it check every draft against every target.
- **Result.** It hid because hub-marked tests show as *errors* without the tokenizer cache, not failures. Lesson: a test that was edited is not a test that passes.

### S7. Smaller ones worth knowing
- **RNG device mismatch** (`40fca50`). A CPU `torch.Generator` cannot sample CUDA tensors. `sample_from_probs` now draws on the generator's device, and `uniform_scalar` returns a Python float.
- **Controller "thrashing"** (`00d6dd5`). The naive switch count read 61 in 240 episodes. Almost all were deliberate probes. I split `preferred` from `current` and counted only preference changes.
- **Prior unit bug** (`7fd1ff6`). "Prior weight 0.5" actually meant about four observations. It is now set as `1-(1-α)^n` so the knob means n real observations.
- **Shared-server hazards** (PROGRESS §2). The NFS mount reads at about 26 MB/s, so a cold 14B load takes 20–40 minutes. The quota gave ENOSPC at 12 GB while `df` showed 1.7 TB free. A broad `pkill` matched its own wrapper shell. Co-tenancy skews batch-1 timings, because they are CPU-bound.

---

## 5. Numbers sheet

**Common setup for every speed number:** 1× H200, bf16, batch 1, greedy (temperature 0), 128 new tokens, no EOS stop, both sides compiled with `reduce-overhead`, discarded warm-up rep, 3 round-robin reps. Run on a shared server (GPU 6 for the sweeps, 2026-08-14). **No raw logs are committed.** The numbers live only in README.md, RESULTS.md and commit messages. **The env is deleted**, so nothing below is re-runnable today without rebuilding it.

### 5.1 Resume bullet 1

| Claim | How it was measured | Caveats | Repo supports? |
|---|---|---|---|
| **2.25×** | `profile_overhead.py --gamma 4`, 0.5B → 14B. Baseline 86.2 tok/s, spec 194.1 tok/s. **One prompt** (palindrome, a code task). Speedup = best spec rep / best baseline rep (`profile_overhead.py:229-231`). Commit `40c2c00`. | Single prompt, and a code prompt. Best-of-3. At γ=4, chat is 1.42–1.47x and summarisation 1.41x. Mean over the three domains is 1.69x. | **Documented, yes.** Supported as "2.25x on a code prompt". **Not supported as a general "LLM inference" speedup.** |
| **2.60× on code** | `domain_sweep.py --gammas 4 8 12 16`, 0.5B, γ=8, 3 code prompts (palindrome, binary search, LRU cache). Median of 9 paired per-prompt ratios (`domain_sweep.py:142-144, 169`). The same 2.60x appeared in two separate sweep runs. | 3 short prompts, not HumanEval. 128 tokens. The γ=8 single-prompt run gave 2.56x. | **Documented, yes** (README, RESULTS, `863f0cc`). Say "three code prompts", not "code workloads". |
| **"fixed-depth"** | γ is a constructor argument. The default is 4 (`speculative.py:160`). | The 2.25x is at γ=4 and the 2.60x is at γ=8. Both are fixed depths, but they are *different* fixed depths. | Yes. Do not imply one config gives both numbers. |
| **"preserving the target distribution"** | Proof (`speculative.py:7-30`). Kernel chi-square at 60k samples, k ∈ {1,2,4}, adversarial q, p(x)=0 (`test_verify_kernel.py`). Exact two-token joint vs 12k samples, γ ∈ {1,3}, temperature 1 and 0.7/top-p 0.9 (`test_engine_correctness.py:98-176`). Greedy exact on tiny models. Real 14B within the measured noise floor. | Distribution tests run on tiny float models, not the 14B. **All speed numbers are greedy.** No sampled-mode speedup was ever measured. "Preserved" means relative to the canonical-vocab target (128 padding ids dropped, on both sides). | **Yes**, with those caveats. The 152 tests are documented, not re-runnable today. |

### 5.2 Resume bullet 2: verified against the repo

**Question: is there a committed comparison across three drafts and several depths?**
**Answer: yes, partly. The three drafts are 0.5B, 1.5B and Coder-1.5B, not 7B.**

What is committed (in README.md / RESULTS.md, with the scripts that produce them):

| Sweep | Drafts | Depths | Prompts | Where |
|---|---|---|---|---|
| A. Single prompt | all 3 | γ = 2, 4, 8 | 1 (palindrome) | README "Draft depth", commit `40c2c00` |
| B. Per-domain | all 3 | γ = 4, 8 | 3 per domain × 3 domains | README "Do the drafts separate", commit `863f0cc` (γ=8 table in full only in untracked PROGRESS §5) |
| C. Per-domain, extended | the run covered all 3 (`domain_sweep.py:223-227`) | γ = 4, 8, 12, 16 | 3 × 3 | README / RESULTS. **Only the 0.5B row is reported**, plus one Coder number (0.48x, γ=16, summarisation) |

Key results:

| | 1.5B | **0.5B** | Coder-1.5B |
|---|---|---|---|
| Single prompt, γ=4 | 2.03x (acc 0.911) | **2.25x** (0.798) | 1.91x (0.825) |
| Code γ=4 (acc / speed) | 0.805 / 1.89x | 0.734 / **2.23x** | 0.804 / 1.89x |
| Chat γ=4 | 0.463 / 1.31x | 0.401 / **1.47x** | 0.391 / 1.17x |
| Summ. γ=4 | 0.411 / 1.23x | 0.338 / **1.41x** | 0.355 / 1.18x |

0.5B depth curve (sweep C): code 2.25 / **2.60** / 2.44 / 2.31; chat **1.42** / 1.25 / 1.03 / 0.82; summ. **1.41** / 1.10 / 0.94 / 0.66 for γ = 4 / 8 / 12 / 16.

How each phrase holds up:
- "three draft models": **supported** (0.5B, 1.5B, Coder-1.5B).
- "a range of depths": **supported** (2–16, though not every draft at every depth, and γ=2 on one prompt only).
- "identifying an optimal draft model": **supported.** The 0.5B wins throughput in every domain at every reported depth. It beats the 1.5B by 12–18% at γ=4.
- "and depth": **partly.** γ=8 is bracketed as the optimum for code. γ=4 is the best *fixed* depth on an even mix (1.69x vs 1.65x at γ=8). For chat and summarisation the optimum is **below 4 and was never measured**.
- "ablations": these are sweeps, not ablations. Nothing was removed from a system. Minor, but an interviewer may pick at it.
- The README claims "identical ordering at γ = 2, 4, 8, 12, 16". Only γ=4 and 8 are committed per domain for all three drafts. γ=12 and 16 orderings for the 1.5B and Coder are **not in any tracked file**.

### 5.3 Other numbers you might be asked about

| Number | What it is | Source | Note |
|---|---|---|---|
| 0.88× | 7B target, **eager**, 1.5B draft, baseline 54.2 tok/s | `b08e59d` | Different target from the 2.25x. 0.5B was 0.92x, Coder 0.82x. |
| 1.33× | 14B target, eager, 0.5B, γ=4 (43.5 tok/s baseline) | README Results | Shows the target change alone already beat 1.0x. |
| 43.5 → 86.2 tok/s | Baseline eager → compiled (1.98×) | `40c2c00` | Compile alone ≈ 2×. |
| "4.5x end to end" | 194.1 / 43.5 | README | README calls this "against where Phase 0 stood". Phase 0 was the 7B, so the label is loose. Avoid it. |
| 0.813 / 0.808 / 0.820 / 0.805 ms/layer | Eager 14B / 1.5B / 0.5B / Coder | README table | **RESULTS.md has the compiled 0.5B wrong (0.098; README has 0.071 = 1.69/24).** PROGRESS lists them in the wrong order. |
| 10.72 / 2.74 / 1.69 ms | Compiled 1-token forward, 14B / 1.5B / 0.5B | README | |
| 1.4% (0.33 of 23.6 ms) | Non-forward work per γ=4 iteration | `40c2c00` | Computed for the **1.5B** draft (first in the registry, `profile_overhead.py:304`), by adding component timings. Not a timed iteration. For the 0.5B, forwards are ~19 ms, so ~1.7%. |
| 0.25 / 0.281 | Eager / compiled batch-width logit shift on the 14B | `6c9dbf1`, `217b8cd` | 0.0 on the 0.5B. |
| 0.125, 1/41 | Smallest nonzero top-1/top-2 gap; fraction of exact ties | `6c9dbf1` | |
| 2.18 µs, 0.0092% | `observe()` cost per iteration | `94e5b17` | CPU timeit, parked code. |
| 1.69 / 1.65 / 1.26 / 1.81 | Mean speedup at fixed γ=4 / 8 / 16 / per-domain oracle | README | The arithmetic checks out. PROGRESS says **do not quote the 7%**. |
| 152 tests (131 CPU, 21 GPU) | Suite size | RESULTS, PROGRESS §6 | Documented. Not re-runnable now (env gone). |
| 4.13 vs 3.35 tokens/iter | "Bursty" claim | `40c2c00` | See 5.4. 1 + 4×0.798 = 4.19, so 4.13 is roughly one rep's value. |

### 5.4 The cost-model claim, re-checked (my arithmetic, using only the documented numbers)

`predict_speedup` (`profile_overhead.py:150-162`) uses `tokens = (1 − a^(γ+1)) / (1 − a)` with `a = SpecStats.acceptance_rate = n_accepted / n_proposed`.

That formula needs `α`, the probability of accepting a position *given that it is reached*. But `n_accepted / n_proposed` = E[accepted] / γ, which is a different quantity. By definition, tokens per iteration = 1 + γ·a. Feeding `a` into the formula **always** under-predicts for a < 1. So "under-predicts every draft" is guaranteed by construction. It is not evidence of burstiness.

Solving for the α that matches the measured E[accepted] under independence:

| Draft | a (measured) | E[acc] = 4a | implied α | tokens = 1 + 4a | documented prediction → corrected | measured |
|---|---|---|---|---|---|---|
| 0.5B | 0.798 | 3.19 | ≈ 0.91 | 4.19 | 1.89x → ≈ 2.36x | 2.25x |
| 1.5B | 0.911 | 3.64 | ≈ 0.96 | 4.64 | 1.93x → ≈ 2.14x | 2.03x |
| Coder | 0.825 | 3.30 | ≈ 0.92 | 4.30 | 1.63x → ≈ 1.99x | 1.91x |

The corrected prediction is *over* by about 4–5% each time. That is what you would expect from the ~1.4–1.7% overhead plus prefill and host syncs. **So the honest finding is that the independence model fits once you feed it the right statistic.** This is the same denominator bug as S4, sitting in a second place. To test burstiness properly, you would compare the histogram of `accepted_per_iteration` against a truncated geometric with that α.

### 5.5 Honest resume wording

- Bullet 1: *"Built distribution-preserving speculative decoding (Qwen2.5-0.5B drafting Qwen2.5-14B). Diagnosed a launch-bound batch-1 decode loop (0.88× at first) and fixed it with a static KV cache, zero-copy rollback and CUDA-graph capture, reaching 2.25× over a compiled target-only baseline on an H200 (1.4–2.6× across code, chat and summarisation prompts; 2.60× on code at depth 8)."*
- Bullet 2: *"Swept three same-tokenizer drafts (Qwen2.5-0.5B, 1.5B, Coder-1.5B) across depths 2–16 and three domains. The 0.5B was fastest everywhere despite the lowest acceptance, the code-specialised draft gave no gain on code (0.804 vs 0.805 acceptance), and the best depth moved with domain (γ=8 for code, γ≤4 for chat and summarisation). Cut a planned draft-switching controller on this evidence."*

---

## 6. Prerequisite concepts

**Autoregressive decoding and the KV cache.** Each new token needs one forward pass. Keys and values of past tokens are cached, so each step processes one new token. At batch 1 each step reads all the weights for very little compute. *Sources:* Vaswani et al. 2017, "Attention Is All You Need"; Pope et al. 2022, "Efficiently Scaling Transformer Inference".

**Memory-bound, compute-bound, launch-bound.** Roofline thinking. A 14B bf16 model is ~29 GB. At ~4.8 TB/s that is a ~6 ms floor per token. The compiled baseline's 10.7 ms is within 2× of that. Eager decode was far slower again, because each tiny kernel launch has fixed CPU cost. *Sources:* Williams et al. 2009, "Roofline"; "Making Deep Learning Go Brrrr From First Principles" (Horace He, 2022).

**Speculative sampling (rejection rule).** Draft x ~ q. Accept with min(1, p/q). Otherwise draw from norm((p − q)₊). Emitted ~ p exactly. *Sources:* Leviathan, Kalman, Matias 2023, "Fast Inference from Transformers via Speculative Decoding"; Chen et al. 2023, "Accelerating LLM Decoding with Speculative Sampling".

**Proof sketch to say aloud.** P(emit x) = q(x)·min(1, p(x)/q(x)) + P(reject)·(p(x) − q(x))₊ / Z. The first term is min(p, q). P(reject) = 1 − Σ min(p, q) = Σ (p − q)₊ = Z. So the second term is (p − q)₊. min(p, q) + (p − q)₊ = p. It chains across positions because position i is only reached when every earlier one was accepted. At that point the target's p_i is conditioned on exactly the committed prefix. The bonus token is a plain draw from p_{γ}. Acceptance probability per position is α = Σ min(p, q) = 1 − TV(p, q).

**Expected accepted tokens and the cost model.** With i.i.d. per-position acceptance α, P(at least i accepted) = α^i. Tokens per iteration = 1 + Σ_{i=1..γ} α^i = (1 − α^(γ+1)) / (1 − α). Speedup ≈ tokens × T_target(1) / (γ·T_draft + T_target(γ+1)). The optimum γ grows with α and shrinks with draft cost. *Source:* Leviathan et al. 2023, §3.

**CUDA graphs and `torch.compile(mode="reduce-overhead")`.** Record a sequence of kernels once, then replay it with one launch. This needs fixed tensor addresses and fixed shapes. Inductor also fuses kernels. *Sources:* PyTorch blog, "Accelerating PyTorch with CUDA Graphs" (2021); PyTorch `torch.compile` docs; HF "StaticCache" docs.

**Floating-point non-associativity and batch invariance.** A different GEMM tiling, split, or batch width changes summation order, so bf16 results differ by ulps. Greedy argmax amplifies near-ties into different sequences. *Sources:* Goldberg 1991, "What Every Computer Scientist Should Know About Floating-Point"; Thinking Machines (He et al.) 2025, "Defeating Nondeterminism in LLM Inference".

**Sampling warpers.** Temperature, top-k and top-p change p. The rejection rule is exact with respect to whatever p you actually use, so both paths must warp identically. *Source:* Holtzman et al. 2020, "The Curious Case of Neural Text Degeneration".

**Tree and head-based drafting.** SpecInfer verifies a token tree in one pass with a tree attention mask. Medusa adds trained heads to the target. EAGLE trains a light draft on the target's hidden features. EAGLE-2 makes the tree dynamic. *Sources:* Miao et al. 2024, "SpecInfer"; Cai et al. 2024, "Medusa"; Li et al. 2024, "EAGLE" and "EAGLE-2".

**Draft–target alignment and distillation.** Acceptance depends on how close q is to p, not on task skill. Distilling the draft on the target's own outputs raises acceptance. *Source:* Zhou et al. 2024, "DistillSpec".

**Statistical testing of samplers.** Chi-square goodness-of-fit and total variation distance against an exact enumerated distribution. You need enough samples per cell (expected count > 5). *Source:* any statistics text; scipy `chisquare` docs.

**EMA bias correction.** A zero-initialised EMA reads low early. Divide by the accumulated weight 1 − (1 − α)^t. *Source:* Kingma & Ba 2015, "Adam", §3.

**Serving context.** Continuous batching and paged KV. At larger batch, decode becomes compute-bound and speculation's gain shrinks. *Sources:* Yu et al. 2022, "Orca"; Kwon et al. 2023, "vLLM / PagedAttention".

---

## 7. Fifty likely questions

**Core algorithm**

1. **Why is the output distribution unchanged?** Accept with min(1, p/q), else resample from (p − q)₊. The two parts sum to p. This holds for any q. See §6.
2. **What happens when all γ are accepted?** The verify pass already gave p at position γ+1. I sample a bonus token from it for free (`speculative.py:144`). Each iteration always emits n_accepted + 1 tokens.
3. **Why does the residual need normalising, and what is Z?** Z = Σ (p − q)₊ = 1 − Σ min(p, q) = P(reject). Without it you would not have a distribution.
4. **What if Z is 0?** Only if p = q, which accepts with probability 1. So it is float noise only. The code falls back to p (`speculative.py:135-139`).
5. **How does greedy differ?** p is one-hot. Accept iff the draft guessed the argmax. Otherwise emit the argmax. A test checks this matches the general rule (`test_verify_kernel.py:139`).
6. **Does it stay exact under top-p?** Yes, if both p and q use the same warped distributions, and the baseline warps identically. One shared `logits_to_probs` guarantees it. Tested at temperature 0.7 / top-p 0.9 (`test_engine_correctness.py:158`).
7. **Expected tokens per iteration?** (1 − α^(γ+1)) / (1 − α) under i.i.d. acceptance. With α ≈ 0.91 and γ=4, that is about 4.2.
8. **How do you choose γ?** Maximise tokens(α, γ) × T_target(1) / (γ·T_draft + T_verify(γ+1)). The optimum rises with α. Measured: γ=8 for code, γ≤4 for chat and summarisation. I shipped γ=4 as the best single value on an even mix.
9. **Why is acceptance rate a bad input to that formula?** Hostile follow-up to 7. `n_accepted / n_proposed` counts never-verified proposals. It is E[accepted]/γ, not α. My README's "bursty" claim came from this mix-up. With the right α the model fits within ~5%. *Say this before they do.*

**Systems and performance**

10. **Why was the first version 0.88x?** Batch-1 eager decode was launch-bound: about 0.8 ms per layer for every model. The draft cost the same as the target. 7B (28 layers) vs 1.5B (28 layers).
11. **How did you prove it was launch overhead and not compute?** ms per layer was flat across a 28x size range. After compiling, it scaled with size (0.223 vs 0.071 ms). A 0.5B step went from 16.9 ms to 1.59 ms.
12. **Hostile: "So torch.compile did it?"** No, two changes. The 14B target alone took eager speculation to 1.33x. Compiling took it to 2.25x. The 0.88x and 2.25x are on different targets.
13. **Why a static cache?** CUDA graphs replay against fixed addresses. A DynamicCache reallocates every step.
14. **How does rollback work, and why is leaving stale K/V safe?** Move each layer's write counter back. The next forward overwrites those slots first, and causal masking hides anything past the query position. Tested by poisoning the cache and requiring identical logits (`test_static_cache.py:49`).
15. **Why hold back the last prompt token?** So the target always verifies exactly γ+1 tokens and the draft always feeds 1 (or 2). That gives a fixed set of shapes to capture.
16. **When does the draft see width 2?** After a fully accepted round. The draft never ran on its own last proposal, so it feeds [x_γ, bonus].
17. **What if γ changes at runtime?** That width must be captured, or the step silently runs eager. The engine raises instead (`speculative.py:280-293`).
18. **Why compile the baseline too?** Compiling alone is 1.98x. Without it, I would credit the compiler's win to speculation.
19. **Hostile: "Your baseline is HF Transformers. vLLM would be faster. Is the 2.25x inflated?"** Possibly somewhat. The compiled baseline is 10.7 ms per token against a ~6 ms bandwidth floor. A tighter baseline leaves less overhead to hide, but speculation still cuts target passes ~4x. I did not benchmark vLLM or HF assisted generation. That is the first thing I would add.
20. **Where does the remaining time go?** Forwards are about 98.6% of an iteration. Rollback, accept/reject and softmax total about 0.33 ms.
21. **Why is the 0.5B faster than the 1.5B with lower acceptance?** 1.69 ms vs 2.74 ms per draft step. Four steps saves ~4 ms out of ~23 ms per iteration. That outweighs 0.80 vs 0.91 acceptance.
22. **Why does depth help code but hurt chat?** Code has higher α (~0.88–0.91 vs ~0.67 implied for chat). Deep drafts only pay when long runs get accepted. At γ=16, chat ran at 0.82x and summarisation at 0.66x.
23. **What would change at batch 32?** Decode moves toward compute-bound. Verifying γ+1 tokens per sequence multiplies compute, so the gain shrinks. Per-sequence rollback and ragged acceptance need paged KV. My batch-1 analysis does not transfer.
24. **Long context?** KV reads dominate. The draft also needs its own cache. The static cache is capped at 2048 and raises beyond that.
25. **Memory cost?** All three drafts and the 14B stayed resident (~36 GB of weights), plus a preallocated cache per model. The shipped config needs only the 0.5B.

**Correctness and testing**

26. **How did you test distribution preservation?** Kernel chi-square at 60k samples, including an adversarial q and p(x)=0. Exact two-token joint vs 12k samples through the full engine. Greedy exactness. Real-weight greedy within the noise floor.
27. **Why tiny random models?** A chi-square needs many samples. The tiny models run the identical engine code. I scaled the head weights ×6 so p and q are peaked and different.
28. **Hostile: "Your kernel test only checks the first emitted token."** True for `test_verify_kernel.py`. The two-token engine test checks the joint across an iteration boundary, against exact enumeration.
29. **Why aren't greedy outputs bit-equal on the 14B?** Batch-width bf16 noise: 0.25 between width 1 and width γ+1. bf16 has exact ties, so an argmax flips.
30. **Hostile: "Isn't 'within noise floor' just a loose test?"** No. A divergence is allowed only where the target's own margin is ≤ the measured floor. A cache or indexing bug gives a divergence at a decisive margin, which still fails.
31. **Hostile: "You say the baseline disagrees with itself at position 118. Same width, so that's not batch-width noise."** Correct. It is a different noise source: run-to-run nondeterminism, or the first rep taking a different kernel path during graph warm-up. The reference is taken from the warm-up rep (`profile_overhead.py:205`). It shows bf16 greedy is fragile. It does not prove the batch-width mechanism. I did not isolate which it was.
32. **What bug did the vocab alignment prevent?** The target pads to 152064 and the drafts to 151936. A target id past 151936 fed into the draft gives IndexError. It happened on the first run.
33. **What is the self-drafting probe?** The target drafts for itself under greedy. Then acceptance measures only width-1 vs width-(γ+1) agreement, independent of draft quality.
34. **What bugs did tests catch in your own tests?** The wrong denominator (S4). A stale registry assertion and a fixture checking the wrong target (S6). A sweep report announcing "no separation" with no data.

**Design choices: why X, not Y**

35. **Why Qwen2.5, not Llama + SmolLM2?** Tokenizers must be byte-identical for p(x)/q(x) to mean anything. SmolLM2's is unrelated to Llama 3's. A test asserts tokenizer identity across the pool.
36. **Why not Medusa?** It needs trained heads on the target, which changes the deployment artifact. Its default "typical acceptance" is not distribution-preserving. Tree verification adds a tree attention mask and branch rollback. The project goal was off-the-shelf drafts with provable exactness.
37. **Why not EAGLE?** It needs a draft trained on the target's hidden features, so the draft becomes tied to that target's internals. EAGLE-style training is also the one way domain drafts could work (distil from the target per domain). Out of scope: it is training, not inference engineering.
38. **Why not tree drafting?** It raises tokens per pass, but verification width becomes the tree size. You need a custom tree mask, KV compaction for the accepted branch, a fixed tree shape for CUDA graphs, and a multi-candidate rejection rule with a harder proof. The brief made it a stretch "to discuss first". It was cancelled with the scope cut.
39. **Why not prompt-lookup or n-gram drafting?** Not considered in the repo. It costs almost nothing per draft and does well on code and summarisation, which copy from the input. It would be a cheap comparison.
40. **Why a Python accept loop?** Measured at 0.036 ms. Not worth vectorising at batch 1.
41. **Why did you cut the adaptive controller?** The drafts never separate by domain. Depth-adaptivity was worth ~7% over the best fixed depth on an even mix. The controller had no test that changing γ mid-request keeps exactness, and had never run on real weights.
42. **Why does a code-specialised draft not help on code?** Acceptance measures closeness to the *target's* distribution. The target is a general instruct model. Specialising the draft moves it away from p about as much as it gains on the domain. 0.804 vs 0.805.

**Failure modes, scaling, what next**

43. **When does speculation lose?** Low acceptance with deep drafts. At γ=16, summarisation hit 0.66x with the 0.5B and 0.48x with Coder. A co-running job on the GPU also skews batch-1 timings.
44. **What breaks silently if someone edits this?** An uncaptured width (guarded). A baseline without `vocab_size` (documented trap). Removing the target crop (exercise 1). A different transformers version changing StaticCache internals.
45. **Hostile: "Best of 3 reps on one prompt is cherry-picking."** For the headline, yes, it is optimistic. Both sides got best-of-3 and round-robin. The per-domain sweep used the median of 9 paired ratios and gave 2.23x on code at γ=4, consistent with it. I would report the median and spread next time.
46. **Hostile: "Did you stop at EOS in benchmarks?"** No. Both sides generate 128 tokens. For short answers, part of the timed text is past `<|im_end|>`. I did not measure how much. I would stop at EOS and report per-prompt token counts.
47. **Hostile: "Is your 2.25x a sampling-mode result?"** No. All speed numbers are greedy. Exactness under sampling is tested on tiny models, but sampled-mode speed was never measured. Expect lower acceptance at temperature 1.
48. **What would you do differently?** Commit raw logs and a pinned env file. Use the right acceptance statistic from day one. Benchmark on HumanEval / ShareGPT / CNN-DM with EOS stopping. Compare against HF assisted generation and vLLM spec decode. Sweep γ = 1–3 for chat.
49. **What comes next?** A per-request depth choice from the first few iterations' acceptance, with an exactness test for γ changing mid-request. Prompt-lookup drafting for copy-heavy tasks. Draft distillation from the target. Batch > 1 with paged KV.
50. **Hostile: "This is just HF assisted generation reimplemented."** The algorithm is standard, yes. The work was making it actually faster at batch 1, proving where time went, building tests that can tell numerics from bugs, and testing the adaptive premise before building it.

---

## 8. "It didn't work" or "I didn't do that", and how to say it

Say these plainly and early. Each one has a short honest line plus why it still says something good.

- **The adaptive draft-switching controller.** "I designed for it, measured the premise first, and it was false. No draft won any domain. I cut it rather than ship a feature the data didn't support." A depth controller exists in `controller.py`, but it is unverified. Say "prototype, never validated", never "built an adaptive controller".
- **The code-specialised draft.** "It bought nothing: 0.804 vs 0.805 acceptance on code. That taught me acceptance is about closeness to the target, not task skill."
- **Real benchmarks.** "I did not run HumanEval, ShareGPT or CNN-DailyMail. The per-domain results are three prompts each, 128 tokens."
- **Sampled-mode speed.** "Exactness under sampling is tested. Speed was only measured greedy."
- **Batch > 1, tree drafting, Medusa, EAGLE.** "Out of scope. Batch-1 only. I can explain what each would change."
- **The "bursty acceptance" claim.** "My write-up said acceptance was bursty. Re-checking, the cost model was fed the wrong statistic, the same denominator bug I fixed in a test. With the right α, independence fits within about 5%." Saying this yourself is a strength.
- **Optimal depth for chat and summarisation.** "Below 4, and I never measured 1–3."
- **No baseline against vLLM or HF assisted generation.** "My baseline is compiled HF at about 86 tok/s. A production server would be faster, which would shrink the relative gain."
- **Reproducibility.** "Numbers are in the README and commits. Raw logs were not committed, and the pinned environment is not in the repo." (Fix this before interviews if you can: rebuild the env by name and commit an `environment.yml`.)
- **Greedy equality on the 14B.** "Not bit-equal, and it can't be. I measured why and wrote tests that still catch real bugs."
- **AI-assisted commits.** The public commits carry `Co-Authored-By: Claude` trailers. If asked, say plainly how you used the assistant and which decisions were yours. PROGRESS records the scope cut, the model choice and the tag-as-evaluation-key as your decisions. Do not deny it.

---

## 9. Self-test exercises

Predict the effect before you read the answer.

**E1. Delete the target crop.** Comment out `crop_cache(target_cache, keep)` at `speculative.py:395`. Run `pytest tests/test_engine_correctness.py -m "not gpu"`.
*Answer.* After the first iteration with a rejection, the target cache stays at seq_len+γ while the sequence is shorter. Next iteration, `cur_ids[t_len:]` is empty, so the feed is only γ wide and `lo = n_acc − γ` is negative. The slice gives fewer than γ+1 rows, so `verify_and_correct` raises `ValueError: shape mismatch`. The tiny-draft greedy tests fail. `test_greedy_is_correct_with_the_draft_as_its_own_target` **still passes**: with every proposal accepted, `keep` equals the cache length and the crop was a no-op anyway. That is why the self-draft test alone cannot guard rollback.

**E2. Resample from p instead of the residual.** At `speculative.py:133`, replace `residual = torch.clamp(p_i - q_i, min=0.0)` with `residual = p_i.clone()`.
*Answer.* The emitted distribution becomes min(p, q) + P(reject)·p, which is not p. It over-weights tokens where q ≥ p. `test_emitted_token_is_distributed_as_the_target` and `test_correct_even_when_the_draft_is_adversarial` fail the chi-square. The sampled two-token engine test fails. All greedy tests still pass, because the greedy branch returns earlier. That is why greedy tests alone cannot prove exactness.

**E3. Use the widest vocabulary.** At `speculative.py:178`, change `min` to `max`.
*Answer.* On the tiny pair (target 8, draft 6), the draft's logits get padded with −inf to width 8, which is harmless. But whenever the target emits id 6 or 7 (a correction or bonus token), the next draft forward indexes past its embedding and raises `IndexError` (a device-side assert on CUDA). Greedy tests fail if any argmax lands on 6 or 7. Sampled tests fail almost surely. This is the bug from the first real run.

**E4. Stop capturing width 2 for the draft.** In `decode_shapes` (`speculative.py:202`), return `"draft": frozenset({1})`.
*Answer.* Correctness is unchanged. After every fully accepted round, the width-2 draft step silently runs eager, so throughput drops by an amount that grows with acceptance (worst on code). `test_declared_decode_shapes_are_the_ones_actually_used` fails, because width 2 is used but not declared. Nothing else catches it. That is the reason that test exists.

**E5. Feed the cost model the right α.** In `predict_speedup` (`profile_overhead.py:159-160`), instead of `a = accept_rate`, solve for α such that α(1 − α^γ)/(1 − α) = γ·accept_rate (or use `shape_agreement`-style verified-position counts).
*Answer.* For the 0.5B at γ=4, the prediction rises from 1.89x to about 2.36x, slightly *above* the measured 2.25x. The same holds for the 1.5B (≈ 2.14x vs 2.03x) and Coder (≈ 1.99x vs 1.91x). The "consistently under-predicts, so acceptance is bursty" conclusion disappears. The leftover ~5% over-prediction is roughly the unmodelled overhead.

**Bonus E6 (no code).** Using only README numbers, predict chat speedup at γ=2 for the 0.5B. Hint: chat acceptance 0.401 at γ=4 implies α ≈ 0.67. Assume verify cost ≈ 12 ms at width 3.
*Answer.* tokens = 1 + 0.67 + 0.45 ≈ 2.11. Cost ≈ 2×1.69 + 12.1 ≈ 15.5 ms. Speedup ≈ 2.11 × 10.72 / 15.5 ≈ 1.46x. That is about equal to γ=4 (1.42–1.47x measured). So "chat's optimum is below 4" may be a very flat optimum, not a big win. Unmeasured: treat this as a hypothesis.
