---
type: prerequisite
tags: [interview, day]
---
# Prerequisite: LLM inference basics (done at the start of Day 1)

Part of [[Interview plan (Nov 2026)]]. Read sections 1–6 and answer the self-check at the start of [[Day 01 - mini-vLLM I]] (about 60 min). Then do the `llama.py` block on Day 1 too, in place of the README skim. Every number here comes from your own repos, so you can quote them.

## Blocks

- [ ] **T, 60 min.** Sections 1–6 below, plus Horace He, *Making Deep Learning Go Brrrr* (compute- vs memory- vs overhead-bound work).
- [ ] **C, 60 min.** `vLLMmini/src/mini_vllm/model/llama.py` (about 200 lines of real code). Trace one token through one layer and write down every tensor shape.
- [ ] **T, 45 min.** Pope et al., *Efficiently Scaling Transformer Inference* (2022), §2–3. Then the metrics in section 6.
- [ ] **P, 30 min.** The self-check questions at the bottom, said out loud.

## 1. One token's forward pass

A decoder-only model runs each token through L layers, and each layer does attention then an MLP. A forward pass costs about
**2·P FLOPs per token** (P = parameter count), because each weight does one multiply and one add.

Your mini-vLLM model, **Llama-3.2-1B**, has:
- 16 layers and hidden size 2048;
- 32 query heads and 8 KV heads (GQA, 4:1), head_dim 64;
- about 1.24B parameters, which is about 2.5 GB in bf16.

## 2. Why a KV cache

Attention at step t needs the keys and values of every earlier token. Without a cache you recompute them at every step,
so the work per step grows with the whole history. With a cache you compute only the new token's K and V, and read the
old ones from memory. The cost is memory:

`KV bytes per token = 2 (K and V) × layers × kv_heads × head_dim × bytes per value`

For Llama-3.2-1B: 2 × 16 × 8 × 64 × 2 B = **32 KiB per token**, so a 4k-token sequence holds 128 MiB. GQA cuts this
4× compared with having 32 KV heads. **KV memory, not compute, limits how many requests you can batch.** That is the
problem mini-vLLM's paging solves.

## 3. Prefill and decode are two different workloads

- **Prefill** handles all N prompt tokens at once. The work is large matrix–matrix multiplies, so it is **compute-bound**. It sets the TTFT.
- **Decode** makes one token per sequence per step. The work is matrix–vector multiplies, and each step reads all the weights plus that sequence's KV cache, so it is **memory-bandwidth-bound**. It sets the TPOT.

## 4. The roofline

Arithmetic intensity is FLOPs per byte moved. An H200 does about 989 TFLOP/s in bf16 and moves 4.8 TB/s. The ridge point is about
989/4.8 ≈ **206 FLOP/byte**; below it you are memory-bound.
- Decode at batch B does about 2·P·B FLOPs while reading 2·P bytes of weights, so its intensity is about **B**. Decode stays
  memory-bound until B is about 200. **Batching is nearly free throughput**, which is why schedulers and KV memory management matter.
- At batch 1 the floor is the weight read: 2.5 GB ÷ 4.8 TB/s ≈ **0.5 ms per token** for Llama-3.2-1B.

## 5. A third regime: overhead-bound

Small kernels at batch 1 can be limited by Python and the CPU launching kernels, not by the GPU.
- **Speculative decoding:** eager per-layer cost was **0.81 ms for the 0.5B, 1.5B and 14B models alike**, a 28× range of
  model sizes (`speculative_decoding/RESULTS.md`, about lines 88–104). `torch.compile` plus a StaticCache brought it to 0.10–0.22 ms.
- **mini-vLLM:** the prefix cache saves **0% of the time below about 2k tokens**. Short prefill is overhead-bound, so skipping it saves almost nothing.

## 6. The serving metrics

- **TTFT** (time to first token) = queueing + prefill.
- **TPOT** (time per output token) = one decode step.
- **Throughput** = tokens per second across all requests.
- Static batching makes new requests wait for the batch to finish. Continuous batching lets them join at the next step.
  That is your mini-vLLM result of TTFT p50 **2448 → 179 ms**, with **3.82×** throughput.

**Links to the next days:** paging frees KV memory, so batches get bigger and decode climbs the roofline (Days 2–3).
Speculative decoding (Day 4) spends idle compute in memory-bound decode to verify γ draft tokens in one pass. Day 5
applies the roofline to single matmul kernels.

## Self-check (answer out loud, then check against the text above)

1. Why does decode get almost free throughput from a bigger batch, while prefill doesn't?
2. Give the KV-cache formula, and the bytes per token for Qwen2.5-0.5B (24 layers, 2 KV heads, head_dim 64, bf16).
3. "Your prefix cache hits 91% by turn 4. Why doesn't it speed things up?" Answer in two sentences.
4. Why did speculative decoding first run at 0.88×?
5. At batch 1 on an H200, what is the lower bound on time per token for Llama-3.2-1B, and what sets it?
6. When does continuous batching improve TTFT, and when can it make TPOT worse?

## My notes and fumbles

- 
