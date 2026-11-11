# Function glossary

Every library and language function that appears in the day notes' code, explained once, with small runnable examples. The **New functions in this code** boxes in the day notes link here.

> [!info] How to read the figures
> - A tensor is drawn as **nested frames, one frame per pair of brackets**. `[3]` has one frame; `[1, 3]` has a frame inside a frame; `[3, 1]` has one outer frame holding three. Count the frames to count the dimensions.
> - **Colour = identity.** A cell keeps its colour wherever it moves, so you can follow it.
> - The figures are **animated** in Obsidian: cells slide from input to output, or a highlight walks through the cells that produce each result. The still picture shows the start and the end.
> - Example outputs were produced by running the code (PyTorch 2.11 on a CPU), not typed by hand.

## Contents

- **Python built-ins:** [[#enumerate|enumerate]], [[#zip|zip]], [[#range|range]], [[#len|len]], [[#sum|sum]], [[#max|max]], [[#min|min]], [[#abs|abs]], [[#round|round]], [[#float|float]], [[#int|int]], [[#bool|bool]], [[#list|list]], [[#tuple|tuple]], [[#set|set]], [[#frozenset|frozenset]], [[#sorted|sorted]], [[#reversed|reversed]], [[#divmod|divmod]], [[#getattr|getattr]], [[#isinstance|isinstance]], [[#id|id]], [[#type|type]], [[#all|all]], [[#any|any]]
- **Lists, dicts and sets:** [[#list.append|list.append]], [[#list.extend|list.extend]], [[#list.pop|list.pop]], [[#list.insert|list.insert]], [[#list.remove|list.remove]], [[#list.sort|list.sort]], [[#list.index|list.index]], [[#dict.get|dict.get]], [[#dict.items|dict.items]], [[#dict.values|dict.values]], [[#dict.keys|dict.keys]], [[#set.add|set.add]], [[#list.clear|list.clear]]
- **math, random, time, statistics:** [[#math.sqrt|math.sqrt]], [[#math.cos|math.cos]], [[#math.sin|math.sin]], [[#math.exp|math.exp]], [[#random.seed|random.seed]], [[#random.uniform|random.uniform]], [[#statistics.median|statistics.median]], [[#time.perf_counter|time.perf_counter]], [[#time.monotonic|time.monotonic]]
- **asyncio:** [[#asyncio.get_running_loop|asyncio.get_running_loop]], [[#asyncio.Queue|asyncio.Queue]], [[#queue.put_nowait|queue.put_nowait]], [[#asyncio.sleep|asyncio.sleep]], [[#event.set|event.set]], [[#event.wait|event.wait]], [[#future.done|future.done]], [[#future.set_result|future.set_result]]
- **Making tensors:** [[#torch.arange|torch.arange]], [[#torch.zeros|torch.zeros]], [[#torch.zeros_like|torch.zeros_like]], [[#torch.empty_like|torch.empty_like]], [[#torch.as_tensor|torch.as_tensor]], [[#torch.randint|torch.randint]], [[#x.new_zeros|x.new_zeros]], [[#x.new_full|x.new_full]]
- **Shapes: reading and changing them:** [[#x.shape|x.shape]], [[#x.dtype|x.dtype]], [[#x.device|x.device]], [[#x.numel|x.numel]], [[#x.unsqueeze|x.unsqueeze]], [[#x.view|x.view]], [[#x.reshape|x.reshape]], [[#x.transpose|x.transpose]], [[#torch.cat|torch.cat]], [[#torch.stack|torch.stack]], [[#x.repeat_interleave|x.repeat_interleave]], [[#x.stride|x.stride]]
- **Element-wise maths:** [[#x.to|x.to]], [[#x.float|x.float]], [[#x.pow|x.pow]], [[#torch.rsqrt|torch.rsqrt]], [[#x.cos|x.cos]], [[#x.sin|x.sin]], [[#torch.clamp|torch.clamp]], [[#nn.functional.silu|nn.functional.silu]]
- **Reductions and picking:** [[#x.mean|x.mean]], [[#x.sum|x.sum]], [[#x.argmax|x.argmax]], [[#torch.softmax|torch.softmax]], [[#x.cumsum|x.cumsum]], [[#torch.topk|torch.topk]], [[#torch.sort|torch.sort]], [[#torch.matmul|torch.matmul]]
- **Writing into tensors:** [[#x.masked_fill|x.masked_fill]], [[#x.scatter_|x.scatter_]], [[#x.scatter|x.scatter]], [[#x.fill_|x.fill_]], [[#x.tolist|x.tolist]], [[#x.item|x.item]]
- **Running models and the GPU:** [[#torch.inference_mode|torch.inference_mode]], [[#torch.cuda.synchronize|torch.cuda.synchronize]], [[#torch.compile|torch.compile]], [[#model.generate|model.generate]], [[#cache.crop|cache.crop]]
- **Triton (GPU kernels):** [[#tl.program_id|tl.program_id]], [[#tl.arange|tl.arange]], [[#tl.load|tl.load]], [[#tl.store|tl.store]], [[#tl.zeros|tl.zeros]], [[#tl.sum|tl.sum]], [[#tl.max|tl.max]], [[#tl.maximum|tl.maximum]], [[#tl.exp|tl.exp]], [[#tl.where|tl.where]]
- **Triton kernels and asyncio futures (Day 3: decode kernel, server):** [[#triton.jit|triton.jit]], [[#tl.constexpr|tl.constexpr]], [[#x.create_future|x.create_future]], [[#x.set_exception|x.set_exception]], [[#ValueError|ValueError]], [[#RuntimeError|RuntimeError]], [[#tokenizer|tokenizer]]
- **torch.compile settings (Day 4: speculative decoding):** [[#torch._dynamo.config.cache_size_limit|torch._dynamo.config.cache_size_limit]]
- **CUDA C++ basics (Day 5: matmul kernels):** [[#CUDA global|CUDA global]], [[#threadIdx|threadIdx]], [[#blockIdx|blockIdx]], [[#blockDim|blockDim]], [[#fmaf|fmaf]], [[#CUDA syncthreads|CUDA syncthreads]], [[#CUDA shared|CUDA shared]], [[#template|template]], [[#size_t|size_t]], [[#pragma unroll|pragma unroll]], [[#cudaMemcpy|cudaMemcpy]], [[#cudaEventCreate|cudaEventCreate]], [[#cudaEventRecord|cudaEventRecord]], [[#cudaEventSynchronize|cudaEventSynchronize]], [[#cudaEventElapsedTime|cudaEventElapsedTime]], [[#cudaEventDestroy|cudaEventDestroy]], [[#cublasSgemm|cublasSgemm]], [[#CUDA_CHECK|CUDA_CHECK]], [[#fprintf|fprintf]], [[#std::min|std::min]]
- **CUDA runtime, streams and warp intrinsics (Day 6):** [[#CUDA kernel launch|CUDA kernel launch]], [[#dim3|dim3]], [[#cudaStreamCreate|cudaStreamCreate]], [[#cudaStreamSynchronize|cudaStreamSynchronize]], [[#cudaStreamDestroy|cudaStreamDestroy]], [[#cudaDeviceSynchronize|cudaDeviceSynchronize]], [[#cudaMalloc|cudaMalloc]], [[#cudaFree|cudaFree]], [[#cudaMemGetInfo|cudaMemGetInfo]], [[#sizeof|sizeof]], [[#CUDA extern shared|CUDA extern shared]], [[#CUDA restrict|CUDA restrict]], [[#CUDA launch_bounds|CUDA launch_bounds]], [[#fmaxf|fmaxf]], [[#CUDA expf|CUDA expf]], [[#INFINITY|INFINITY]], [[#CUDA shfl_xor_sync|CUDA shfl_xor_sync]], [[#CUDA syncwarp|CUDA syncwarp]], [[#cudaFuncSetAttribute|cudaFuncSetAttribute]], [[#cudaGetLastError|cudaGetLastError]]
- **NumPy basics (Day 7: toy quantisers):** [[#np.array|np.array]], [[#np.random.randn|np.random.randn]], [[#np.argmin|np.argmin]], [[#np.round|np.round]], [[#np.tanh|np.tanh]], [[#x.copy|x.copy]], [[#log2|log2]]
- **Training loops, files and evaluation (Day 8: CosyVoice LoRA):** [[#torch.tensor|torch.tensor]], [[#x.dim|x.dim]], [[#x.squeeze|x.squeeze]], [[#x.cpu|x.cpu]], [[#torch.concat|torch.concat]], [[#x.ceil|x.ceil]], [[#x.int|x.int]], [[#x.copy_|x.copy_]], [[#x.backward|x.backward]], [[#x.zero_grad|x.zero_grad]], [[#torch.nn.utils.clip_grad_norm_|torch.nn.utils.clip_grad_norm_]], [[#torch.autocast|torch.autocast]], [[#x.parameters|x.parameters]], [[#x.named_parameters|x.named_parameters]], [[#LoraConfig|LoraConfig]], [[#inject_adapter_in_model|inject_adapter_in_model]], [[#torch.save|torch.save]], [[#x.save|x.save]], [[#Path|Path]], [[#x.mkdir|x.mkdir]], [[#x.unlink|x.unlink]], [[#dict|dict]], [[#random.random|random.random]], [[#wer|wer]], [[#cer|cer]], [[#x.DataFrame|x.DataFrame]], [[#x.unique|x.unique]]
- **Indexing, masks and input gradients (Day 9: retrieval heads, pruning, PGD):** [[#x.clamp|x.clamp]], [[#x.abs|x.abs]], [[#x.flatten|x.flatten]], [[#x.kthvalue|x.kthvalue]], [[#x.endswith|x.endswith]], [[#x.clone|x.clone]], [[#x.requires_grad_|x.requires_grad_]], [[#F.cross_entropy|F.cross_entropy]], [[#torch.autograd.grad|torch.autograd.grad]], [[#x.sign|x.sign]], [[#torch.min|torch.min]], [[#torch.max|torch.max]]
- **Day 10 code: Photo-Seek search: regex, dates, SQLite, CLIP, FAISS:** [[#x.search|x.search]], [[#x.group|x.group]], [[#x.span|x.span]], [[#timedelta|timedelta]], [[#x.isoformat|x.isoformat]], [[#replace|replace]], [[#x.fromkeys|x.fromkeys]], [[#x.warning|x.warning]], [[#x.info|x.info]], [[#x.execute|x.execute]], [[#x.fetchall|x.fetchall]], [[#torch.nn.functional.normalize|torch.nn.functional.normalize]], [[#F.normalize|F.normalize]], [[#x.encode_text|x.encode_text]], [[#x.numpy|x.numpy]], [[#np.fromiter|np.fromiter]], [[#faiss.IDSelectorBatch|faiss.IDSelectorBatch]], [[#faiss.SearchParameters|faiss.SearchParameters]], [[#x.add_with_ids|x.add_with_ids]]
- **Day 11 code: Faces and steering: numpy indexing, hooks, log-probabilities:** [[#print|print]], [[#math.isfinite|math.isfinite]], [[#np.asarray|np.asarray]], [[#np.arange|np.arange]], [[#np.where|np.where]], [[#np.full|np.full]], [[#np.argsort|np.argsort]], [[#np.random.default_rng|np.random.default_rng]], [[#x.permutation|x.permutation]], [[#np.stack|np.stack]], [[#np.isfinite|np.isfinite]], [[#x.argmin|x.argmin]], [[#x.min|x.min]], [[#x.register_forward_hook|x.register_forward_hook]], [[#torch.manual_seed|torch.manual_seed]], [[#torch.no_grad|torch.no_grad]], [[#torch.log_softmax|torch.log_softmax]], [[#x.gather|x.gather]], [[#torch.acos|torch.acos]], [[#torch.cos|torch.cos]], [[#torch.where|torch.where]], [[#x.bool|x.bool]]
- **Day 12 code: BountyBrain: strings, regex, SQLAlchemy, BM25, rerankers:** [[#str|str]], [[#x.strip|x.strip]], [[#x.upper|x.upper]], [[#x.lower|x.lower]], [[#x.startswith|x.startswith]], [[#x.isdigit|x.isdigit]], [[#x.join|x.join]], [[#x.split|x.split]], [[#x.compile|x.compile]], [[#x.findall|x.findall]], [[#json.dumps|json.dumps]], [[#math.log2|math.log2]], [[#math.log1p|math.log1p]], [[#Counter|Counter]], [[#lru_cache|lru_cache]], [[#np.zeros|np.zeros]], [[#select|select]], [[#x.where|x.where]], [[#x.scalar_one_or_none|x.scalar_one_or_none]], [[#x.scalars|x.scalars]], [[#x.all|x.all]], [[#x.ilike|x.ilike]], [[#x.group_by|x.group_by]], [[#x.count|x.count]], [[#BM25Okapi|BM25Okapi]], [[#x.get_scores|x.get_scores]], [[#x.encode|x.encode]], [[#CrossEncoder|CrossEncoder]], [[#x.predict|x.predict]], [[#x.create_collection|x.create_collection]], [[#x.VectorParams|x.VectorParams]]
- **C++ standard library, atomics and libpqxx (Day 13: DECS, Micro Exchange):** [[#std::list|std::list]], [[#std::pair|std::pair]], [[#std::unordered_map|std::unordered_map]], [[#x.find|x.find]], [[#x.end|x.end]], [[#x.begin|x.begin]], [[#x.splice|x.splice]], [[#x.erase|x.erase]], [[#x.size|x.size]], [[#x.back|x.back]], [[#x.front|x.front]], [[#x.pop_back|x.pop_back]], [[#x.emplace_front|x.emplace_front]], [[#x.push_back|x.push_back]], [[#x.push|x.push]], [[#x.empty|x.empty]], [[#std::mutex|std::mutex]], [[#std::lock_guard|std::lock_guard]], [[#std::unique_lock|std::unique_lock]], [[#x.unlock|x.unlock]], [[#x.notify_one|x.notify_one]], [[#std::atomic|std::atomic]], [[#x.load|x.load]], [[#x.store|x.store]], [[#std::memory_order_relaxed|std::memory_order_relaxed]], [[#std::memory_order_acquire|std::memory_order_acquire]], [[#std::memory_order_release|std::memory_order_release]], [[#std::atomic_ref|std::atomic_ref]], [[#alignas|alignas]], [[#rdtsc|rdtsc]], [[#pqxx::connection|pqxx::connection]], [[#db::work|db::work]], [[#x.exec_params|x.exec_params]], [[#x.commit|x.commit]], [[#x.set_content|x.set_content]], [[#std::exception|std::exception]], [[#x.what|x.what]]
- **C++ standard library (Day 14: Raft):** [[#std::to_string|std::to_string]], [[#x.resize|x.resize]], [[#x.notify_all|x.notify_all]], [[#std::make_shared|std::make_shared]], [[#std::thread|std::thread]], [[#x.detach|x.detach]], [[#x.data|x.data]], [[#std::cerr|std::cerr]]

## 1. Python built-ins

### enumerate

`enumerate(xs)`: loops over a list and gives you each item **with its index**. `for i, x in enumerate(['a', 'b'])` gives `(0, 'a')`, then `(1, 'b')`.

```python
>>> for i, tok in enumerate(['the', 'cat', 'sat']):
...     print(i, tok)
0 the
1 cat
2 sat
>>> list(enumerate('ab', start=1))
[(1, 'a'), (2, 'b')]
```

> [!warning] Watch out
> It gives (index, item), in that order; `for tok, i in enumerate(...)` swaps their names silently.

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### zip

`zip(a, b)`: walks several lists **in step**, pairing their items. `zip([1, 2], ['x', 'y'])` gives `(1, 'x')`, `(2, 'y')`.

```python
>>> list(zip([5, 2, 7], ['the', 'cat', 'sat']))
[(5, 'the'), (2, 'cat'), (7, 'sat')]
>>> list(zip([1, 2, 3], ['x', 'y']))
[(1, 'x'), (2, 'y')]
```

> [!warning] Watch out
> It stops at the **shortest** list without an error: the 3 above is silently dropped.

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### range

`range(start, stop, step)`: the integers from `start` up to (not including) `stop`. `range(0, 10, 4)` gives 0, 4, 8. `range(3)` gives 0, 1, 2.

```python
>>> list(range(4))
[0, 1, 2, 3]
>>> list(range(2, 10, 3))
[2, 5, 8]
>>> list(range(3, 0, -1))
[3, 2, 1]
```

> [!warning] Watch out
> The end is excluded: `range(4)` stops at 3.

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 04 - Speculative decoding#Code, explained|Day 4]], [[Day 07 - Seminar and speech basics#Code, explained|Day 7]], [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### len

`len(xs)`: how many items. `len([7, 8, 9])` = 3.

```python
>>> len([5, 2, 7])
3
>>> len('hello')
5
>>> len({'a': 1, 'b': 2})
2
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### sum

`sum(xs)`: adds up the items. `sum([1, 2, 3])` = 6.

```python
>>> sum([1, 2, 3])
6
>>> sum(len(r) for r in [[1, 2], [3]])
3
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 04 - Speculative decoding#Code, explained|Day 4]], [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### max

`max(xs) / max(a, b)`: the largest. `max([3, 9, 2])` = 9. With `key=f` it compares `f(x)` instead: `max(range(3), key=lambda i: s[i])` is the index of the largest `s[i]`.

```python
>>> max([3, 9, 2])
9
>>> scores = [0.1, 2.1, 0.6]
>>> max(range(3), key=lambda i: scores[i])
1
>>> max(4, 7)
7
```

> [!warning] Watch out
> `key=` changes what is compared, not what is returned: you get the item (here the index), not its score.

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### min

`min(xs) / min(a, b)`: the smallest. `min(5, 2)` = 2.

```python
>>> min(5, 2)
2
>>> min([3, 1, 4])
1
```

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 04 - Speculative decoding#Code, explained|Day 4]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### abs

`abs(x)`: the absolute value. `abs(-3)` = 3.

```python
>>> abs(-3)
3
>>> abs(2.5 - 4)
1.5
```

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### round

`round(x, n)`: rounds to `n` decimal places. `round(0.7183, 1)` = 0.7.

```python
>>> round(0.7183, 1)
0.7
>>> round(2.5)
2
>>> round(3.5)
4
```

> [!warning] Watch out
> Python rounds halves to the nearest **even** number: `round(2.5)` is 2.

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 07 - Seminar and speech basics#Code, explained|Day 7]], [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### float

`float(x)`: converts to a floating-point number. `float('-inf')` is minus infinity, a value smaller than every number (used to mask scores).

```python
>>> float('3.5')
3.5
>>> float('-inf') < -1e300
True
>>> float('-inf') + 5
-inf
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 04 - Speculative decoding#Code, explained|Day 4]], [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### int

`int(x)`: converts to an integer, dropping the fraction. `int(3.9)` = 3.

```python
>>> int(3.9)
3
>>> int(-3.9)
-3
>>> int('42')
42
```

> [!warning] Watch out
> It truncates toward zero, so `int(-3.9)` is −3, not −4.

*Used in:* [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 04 - Speculative decoding#Code, explained|Day 4]], [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### bool

`bool(x)`: converts to True / False. On a one-element tensor, `bool(t)` reads its value.

```python
>>> bool(0)
False
>>> bool([])
False
>>> bool([0])
True
```

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 04 - Speculative decoding#Code, explained|Day 4]]

### list

`list(xs)`: makes a new list from any iterable. `list(reversed(range(3)))` = `[2, 1, 0]`. `list(x)` on a list makes a **copy**.

```python
>>> list(range(3))
[0, 1, 2]
>>> a = [1, 2]
>>> b = list(a)
>>> b.append(3)
>>> a
[1, 2]
```

> [!warning] Watch out
> `b = a` would NOT copy: both names would point at one list.

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### tuple

`tuple(xs)`: an immutable list. Tuples can be dictionary keys (lists can't), which is why token chunks are turned into tuples.

```python
>>> t = tuple([5, 2, 7])
>>> t
(5, 2, 7)
>>> {t: 'block 0'}[(5, 2, 7)]
'block 0'
```

> [!warning] Watch out
> Lists can't be dict keys (they can change); that's why token chunks become tuples.

*Used in:* [[Day 02 - mini-vLLM II#Code, explained|Day 2]]

### set

`set(xs)`: a collection with no duplicates and fast membership tests: `3 in {1, 3}` is True, in constant time.

```python
>>> s = {1, 3, 3, 5}
>>> s
{1, 3, 5}
>>> 3 in s
True
```

*Used in:* [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### frozenset

`frozenset(xs)`: an immutable set (can be a dict key or compared for equality).

```python
>>> frozenset([1, 2]) == frozenset([2, 1])
True
>>> {frozenset([1, 2]): 'ok'}
{frozenset({1, 2}): 'ok'}
```

*Used in:* [[Day 04 - Speculative decoding#Code, explained|Day 4]]

### sorted

`sorted(xs, key=f)`: a new list in increasing order (by `f(x)` if `key` is given). `sorted([3, 1, 2])` = `[1, 2, 3]`.

```python
>>> sorted([3, 1, 2])
[1, 2, 3]
>>> sorted(['bb', 'a', 'ccc'], key=len)
['a', 'bb', 'ccc']
>>> sorted([3, 1, 2], reverse=True)
[3, 2, 1]
```

*Used in:* [[Day 04 - Speculative decoding#Code, explained|Day 4]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### reversed

`reversed(xs)`: the items in reverse order. `list(reversed([0, 1, 2]))` = `[2, 1, 0]`.

```python
>>> list(reversed([0, 1, 2]))
[2, 1, 0]
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]]

### divmod

`divmod(a, b)`: the quotient **and** the remainder at once: `divmod(9, 4)` = `(2, 1)`, because 9 = 2 × 4 + 1.

```python
>>> divmod(9, 4)
(2, 1)
>>> block, offset = divmod(19, 16)
>>> (block, offset)
(1, 3)
```

> [!warning] Watch out
> Day 1's slot formula is exactly this: token 19 with B = 16 lives in block 1, offset 3.

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]]

### getattr

`getattr(obj, 'name', default)`: reads an attribute by name, returning `default` if it doesn't exist. `getattr(config, 'rope_theta', 10000)` is the exact pattern behind Day 1's RoPE bug: a missing attribute silently became 10000.

```python
>>> class Cfg: hidden_size = 2048
>>> getattr(Cfg, 'hidden_size', 0)
2048
>>> getattr(Cfg, 'rope_theta', 10000.0)
10000.0
```

> [!warning] Watch out
> A typo or missing field returns the default with **no error**, as in Day 1's RoPE bug.

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 04 - Speculative decoding#Code, explained|Day 4]]

### isinstance

`isinstance(x, T)`: True if `x` is of type `T`. `isinstance(out, tuple)` checks whether a layer returned a tuple.

```python
>>> isinstance((1, 2), tuple)
True
>>> isinstance(3, (int, float))
True
```

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### id

`id(obj)`: a unique number for an object while it lives; used to put objects in a set without comparing their contents.

```python
>>> a = [1]; b = a; c = [1]
>>> a = [1]; b = a; c = [1]
>>> a = [1]; b = a; c = [1]
>>> id(a) == id(b)
True
>>> id(a) == id(c)
False
```

*Used in:* [[Day 02 - mini-vLLM II#Code, explained|Day 2]]

### type

`type(x)`: the class of `x`. `type(x).__name__` is its class name as text.

```python
>>> type(3).__name__
'int'
>>> type([1]).__name__
'list'
```

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 08 - CosyVoice#Code, explained|Day 8]]

### all

`all(xs)`: True if **every** item is true. `all(q == 1 for q in [1, 1])` = True.

```python
>>> all(x > 0 for x in [1, 2, 3])
True
>>> all([])
True
```

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### any

`any(xs)`: True if **at least one** item is true.

```python
>>> any(x > 2 for x in [1, 2, 3])
True
>>> any([])
False
```

## 2. Lists, dicts and sets

### list.append

`.append(x)`: adds `x` at the end of a list. `[1].append(2)` → `[1, 2]`.

```python
>>> ids = [5, 2]
>>> ids.append(7)
>>> ids
[5, 2, 7]
>>> ids.append([1, 2])
>>> ids
[5, 2, 7, [1, 2]]
```

> [!warning] Watch out
> Appending a list adds ONE item (a nested list); use `.extend` to add its elements.

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 04 - Speculative decoding#Code, explained|Day 4]], [[Day 07 - Seminar and speech basics#Code, explained|Day 7]], [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### list.extend

`.extend(xs)`: adds **all** items of `xs` at the end. `a = [1]; a.extend([2, 3])` → `[1, 2, 3]`.

```python
>>> ids = [5]
>>> ids.extend([2, 7])
>>> ids
[5, 2, 7]
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### list.pop

`.pop() / .pop(i) / .pop(key, default)`: removes and returns an item: on a list the last one (or index `i`); on a dict, the entry for `key` (or `default` if missing). `[0, 1, 2].pop()` returns 2.

```python
>>> q = [10, 11, 12]
>>> q.pop()
12
>>> q.pop(0)
10
>>> q
[11]
>>> d = {'a': 1}
>>> d.pop('a')
1
>>> d.pop('zz', None)
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### list.insert

`.insert(i, x)`: puts `x` at index `i` of a list, shifting the rest right. `waiting.insert(0, r)` puts `r` at the **front**.

```python
>>> waiting = ['R']
>>> waiting.insert(0, 'Q')
>>> waiting
['Q', 'R']
```

*Used in:* [[Day 02 - mini-vLLM II#Code, explained|Day 2]]

### list.remove

`.remove(x)`: deletes the first item equal to `x` from a list (an error if it isn't there).

```python
>>> running = ['P', 'Q', 'R']
>>> running.remove('Q')
>>> running
['P', 'R']
```

*Used in:* [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### list.sort

`.sort(key=f)`: sorts a list **in place** (by `f(x)` if given). `nodes.sort(key=lambda n: n.last_accessed)` puts the least recently used first.

```python
>>> ages = [3.0, 1.0, 2.0]
>>> ages.sort()
>>> ages
[1.0, 2.0, 3.0]
>>> print(ages.sort())
None
```

> [!warning] Watch out
> It sorts in place and returns None; `x = xs.sort()` leaves x = None.

*Used in:* [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 04 - Speculative decoding#Code, explained|Day 4]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### list.index

`.index(x)`: the position of the first item equal to `x`. `[5, 7].index(7)` = 1.

```python
>>> [5, 7, 7].index(7)
1
```

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### dict.get

`.get(key, default)`: a dict lookup that returns `default` (or None) instead of failing when `key` is missing.

```python
>>> children = {(1, 2): 'node A'}
>>> children.get((1, 2))
'node A'
>>> children.get((9, 9))
>>> children.get((9, 9), 'none')
'none'
```

*Used in:* [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### dict.items

`.items()`: a dict's `(key, value)` pairs, for looping over both.

```python
>>> for k, v in {'a': 1, 'b': 2}.items():
...     print(k, v)
a 1
b 2
```

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### dict.values

`.values()`: all the values of a dict, without the keys.

```python
>>> list({'a': 1, 'b': 2}.values())
[1, 2]
```

*Used in:* [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### dict.keys

`.keys()`: a dict's keys.

```python
>>> list({'a': 1, 'b': 2}.keys())
['a', 'b']
```

### set.add

`.add(x)`: puts `x` into a set (no effect if it's already there).

```python
>>> seen = set()
>>> seen.add(3); seen.add(3)
>>> seen.add(3); seen.add(3)
>>> seen
{3}
```

*Used in:* [[Day 02 - mini-vLLM II#Code, explained|Day 2]]

### list.clear

`.clear()`: empties a list, dict or set. On an `asyncio.Event`, it resets the event to "not set", so the next `wait()` blocks.

```python
>>> xs = [1, 2]
>>> xs.clear()
>>> xs
[]
```

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

## 3. math, random, time, statistics

### math.sqrt

`math.sqrt(x)`: the square root. `math.sqrt(16)` = 4.

```python
>>> import math
>>> math.sqrt(16)
4.0
>>> 1 / math.sqrt(64)
0.125
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]]

### math.cos

`math.cos(x)`: cosine of an angle in **radians**. `math.cos(1)` = 0.540.

```python
>>> import math
>>> round(math.cos(1), 3)
0.54
>>> math.cos(0)
1.0
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]]

### math.sin

`math.sin(x)`: sine of an angle in radians. `math.sin(1)` = 0.841.

```python
>>> import math
>>> round(math.sin(1), 3)
0.841
>>> math.sin(0)
0.0
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]]

### math.exp

`math.exp(x)`: e to the power x. `math.exp(0)` = 1, `math.exp(-2.85)` = 0.058.

```python
>>> import math
>>> math.exp(0)
1.0
>>> round(math.exp(1), 3)
2.718
>>> round(math.exp(-2.85), 3)
0.058
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### random.seed

`random.seed(n)`: fixes Python's random generator, so the same `n` always produces the same "random" numbers. That's how the toy numbers are reproducible.

```python
>>> import random
>>> random.seed(7); a = random.random()
>>> random.seed(7); a = random.random()
>>> random.seed(7); b = random.random()
>>> random.seed(7); b = random.random()
>>> a == b
True
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]]

### random.uniform

`random.uniform(a, b)`: a random float between `a` and `b`.

```python
>>> import random
>>> random.seed(7)
>>> round(random.uniform(-1, 1), 4)
-0.3523
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]]

### statistics.median

`statistics.median(xs)`: the middle value after sorting. `median([7.6, 4.8, 7.5])` = 7.5. Robust to one odd round.

```python
>>> import statistics
>>> statistics.median([7.6, 4.8, 7.5])
7.5
>>> statistics.median([1, 2, 3, 100])
2.5
```

> [!warning] Watch out
> With an even count it averages the middle two: median([1, 2, 3, 100]) = 2.5.

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### time.perf_counter

`time.perf_counter()`: a high-resolution clock in seconds, for measuring how long something took (subtract two readings).

```python
>>> import time
>>> t0 = time.perf_counter()
>>> sum(range(10**6)) > 0
True
>>> elapsed = time.perf_counter() - t0
>>> elapsed > 0
True
```

> [!warning] Watch out
> On a GPU, call torch.cuda.synchronize() before each reading, or you time only the launch.

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### time.monotonic

`time.monotonic()`: a clock that never goes backwards (unaffected by system time changes); used for "last used" timestamps.

```python
>>> import time
>>> a = time.monotonic(); b = time.monotonic()
>>> a = time.monotonic(); b = time.monotonic()
>>> b >= a
True
```

*Used in:* [[Day 02 - mini-vLLM II#Code, explained|Day 2]]

## 4. asyncio

### asyncio.get_running_loop

`asyncio.get_running_loop()`: the event loop running the current async code; `.create_future()` on it makes a Future a handler can `await`.

```python
>>> import asyncio
>>> async def main():
...     fut = asyncio.get_running_loop().create_future()
...     fut.set_result('done')
...     return await fut
>>> asyncio.run(main())
'done'
```

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### asyncio.Queue

`asyncio.Queue()`: a queue for async code: one side `put`s items, another `await`s `get()` and sleeps until one arrives. Used to stream tokens to a client.

```python
>>> import asyncio
>>> async def main():
...     q = asyncio.Queue()
...     q.put_nowait('tok1'); q.put_nowait('tok2')
...     return [await q.get(), await q.get()]
>>> asyncio.run(main())
['tok1', 'tok2']
```

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### queue.put_nowait

`.put_nowait(x)`: puts `x` into an `asyncio.Queue` immediately, without waiting.

```python
>>> import asyncio
>>> async def main():
...     q = asyncio.Queue()
...     q.put_nowait(5)
...     return q.qsize()
>>> asyncio.run(main())
1
```

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### asyncio.sleep

`asyncio.sleep(s)`: pauses this async task for `s` seconds, letting others run. `await asyncio.sleep(0)` means "give every other task a turn right now".

```python
>>> import asyncio
>>> async def a(log):
...     log.append('a1'); await asyncio.sleep(0); log.append('a2')
>>> async def b(log):
...     log.append('b1')
>>> async def main():
...     log = []
...     await asyncio.gather(a(log), b(log))
...     return log
>>> asyncio.run(main())
['a1', 'b1', 'a2']
```

> [!warning] Watch out
> `sleep(0)` lets b run between a1 and a2. Without an await, a long loop starves every other task.

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### event.set

`.set()`: on an `asyncio.Event`: marks it set, waking every task waiting on it.

```python
>>> import asyncio
>>> async def main():
...     ev = asyncio.Event()
...     asyncio.get_running_loop().call_soon(ev.set)
...     await ev.wait()
...     return ev.is_set()
>>> asyncio.run(main())
True
```

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### event.wait

`.wait()`: on an `asyncio.Event`: `await event.wait()` sleeps until someone calls `.set()`.

```python
>>> import asyncio
>>> async def main():
...     ev = asyncio.Event(); ev.set()
...     await ev.wait()
...     return 'woke up'
>>> asyncio.run(main())
'woke up'
```

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### future.done

`.done()`: on a Future: True once it has a result (or an error).

```python
>>> import asyncio
>>> async def main():
...     f = asyncio.get_running_loop().create_future()
...     before = f.done(); f.set_result(1)
...     return before, f.done()
>>> asyncio.run(main())
(False, True)
```

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### future.set_result

`.set_result(x)`: on a Future: gives it its value, waking whoever is `await`ing it.

```python
>>> import asyncio
>>> async def main():
...     f = asyncio.get_running_loop().create_future()
...     f.set_result([5, 2, 7])
...     return await f
>>> asyncio.run(main())
[5, 2, 7]
```

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

## 5. Making tensors

### torch.arange

`torch.arange(start, end, step)`: a 1-D tensor counting from `start` up to (not including) `end` in steps of `step`. `torch.arange(0, 8, 2)` → `tensor([0, 2, 4, 6])`. With `head_dim = 64` it gives 0, 2, …, 62: one entry per feature pair.

![[g-arange.svg]]

```python
>>> import torch
>>> torch.arange(5)
tensor([0, 1, 2, 3, 4])
>>> torch.arange(0, 8, 2)
tensor([0, 2, 4, 6])
>>> torch.arange(0, 8, 2, dtype=torch.float32) / 8
tensor([0.0000, 0.2500, 0.5000, 0.7500])
>>> torch.arange(0, 64, 2).shape
torch.Size([32])
```

> [!warning] Watch out
> The end is excluded, like range. Without dtype it gives integers; RoPE asks for float32 so the division gives fractions.

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]]

### torch.zeros

`torch.zeros(shape, dtype, device)`: a tensor of the given shape filled with 0. `torch.zeros((2, 3))` → a 2 × 3 grid of zeros. The KV cache is pre-allocated this way.

```python
>>> import torch
>>> torch.zeros(2, 3)
tensor([[0., 0., 0.],
        [0., 0., 0.]])
>>> torch.zeros((4, 16, 8, 64), dtype=torch.bfloat16).shape
torch.Size([4, 16, 8, 64])
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 08 - CosyVoice#Code, explained|Day 8]]

### torch.zeros_like

`torch.zeros_like(t)`: zeros with the same shape, dtype and device as `t`.

```python
>>> import torch
>>> x = torch.ones(2, 2, dtype=torch.long)
>>> torch.zeros_like(x)
tensor([[0, 0],
        [0, 0]])
```

*Used in:* [[Day 04 - Speculative decoding#Code, explained|Day 4]], [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]]

### torch.empty_like

`torch.empty_like(t)`: a tensor shaped like `t` **without** initialising its values (faster; every element must be written before use).

```python
>>> import torch
>>> x = torch.ones(2, 3)
>>> y = torch.empty_like(x)
>>> y.shape, y.dtype
(torch.Size([2, 3]), torch.float32)
```

> [!warning] Watch out
> The values are whatever was in that memory: never read them before writing.

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### torch.as_tensor

`torch.as_tensor(xs, dtype, device)`: turns a Python list into a tensor (without copying if it already is one on the right device).

```python
>>> import torch
>>> torch.as_tensor([5, 2, 7])
tensor([5, 2, 7])
>>> torch.as_tensor([0, 1, 2], dtype=torch.long).dtype
torch.int64
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]]

### torch.randint

`torch.randint(low, high, shape)`: random integers in [low, high). Used to make dummy token ids for warm-up.

```python
>>> import torch
>>> torch.manual_seed(0)
<torch._C.Generator object at 0x7a9ab35ee6f0>
>>> torch.randint(0, 10, (2, 4))
tensor([[4, 9, 3, 0],
        [3, 9, 7, 3]])
```

*Used in:* [[Day 04 - Speculative decoding#Code, explained|Day 4]]

### x.new_zeros

`.new_zeros(shape)`: zeros with the same dtype and device as the tensor it's called on.

```python
>>> import torch
>>> x = torch.ones(3, dtype=torch.bfloat16)
>>> x.new_zeros(2)
tensor([0., 0.], dtype=torch.bfloat16)
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]]

### x.new_full

`.new_full(shape, value)`: a tensor of `shape` filled with `value`, same dtype and device as the original. Used to pad logits with −∞.

```python
>>> import torch
>>> logits = torch.tensor([1.0, 2.0])
>>> logits.new_full((3,), float('-inf'))
tensor([-inf, -inf, -inf])
```

*Used in:* [[Day 04 - Speculative decoding#Code, explained|Day 4]]

## 6. Shapes: reading and changing them

### x.shape

`.shape`: the tensor's size along each axis. `x.shape == (6, 2048)` means 6 rows of 2048.

```python
>>> import torch
>>> x = torch.zeros(6, 2048)
>>> x.shape
torch.Size([6, 2048])
>>> x.shape[-1]
2048
>>> n, d = x.shape
>>> n
6
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 04 - Speculative decoding#Code, explained|Day 4]], [[Day 08 - CosyVoice#Code, explained|Day 8]]

### x.dtype

`.dtype`: the number type stored: `torch.bfloat16`, `torch.float32`, `torch.long` (64-bit integers), …

```python
>>> import torch
>>> torch.tensor([1, 2]).dtype
torch.int64
>>> torch.tensor([1.0]).dtype
torch.float32
>>> torch.zeros(1, dtype=torch.bfloat16).dtype
torch.bfloat16
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### x.device

`.device`: where the tensor lives: `cpu` or `cuda:0` (a GPU).

```python
>>> import torch
>>> torch.zeros(2).device
device(type='cpu')
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 04 - Speculative decoding#Code, explained|Day 4]], [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### x.numel

`.numel()`: the total number of elements. A `[3, 4]` tensor has 12.

```python
>>> import torch
>>> torch.zeros(3, 4).numel()
12
>>> torch.zeros(16, 8, 64).numel()
8192
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 04 - Speculative decoding#Code, explained|Day 4]], [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]]

### x.unsqueeze

`.unsqueeze(d)`: inserts a new axis of size 1 at position `d`. A `[3]` tensor → `.unsqueeze(1)` → `[3, 1]` (a column); `.unsqueeze(0)` → `[1, 3]` (a row). Needed so tensors broadcast against each other.

![[g-unsqueeze.svg]]

![[g-broadcast.svg]]

```python
>>> import torch
>>> x = torch.tensor([3, 5, 7])
>>> x.unsqueeze(0)
tensor([[3, 5, 7]])
>>> x.unsqueeze(0).shape
torch.Size([1, 3])
>>> x.unsqueeze(1)
tensor([[3],
        [5],
        [7]])
>>> x.unsqueeze(1).shape
torch.Size([3, 1])
>>> x.unsqueeze(-1).shape
torch.Size([3, 1])
```

> [!warning] Watch out
> `unsqueeze(-1)` means "at the end": for a 1-D tensor it's the same as `unsqueeze(1)`.

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### x.view

`.view(shape)`: reinterprets the same numbers with a new shape, **no copy**. A `[6, 2048]` tensor `.view(6, 32, 64)` reads each row of 2048 as 32 heads × 64.

![[g-view.svg]]

```python
>>> import torch
>>> x = torch.arange(12).view(2, 6)
>>> x
tensor([[ 0,  1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10, 11]])
>>> x.view(2, 2, 3)
tensor([[[ 0,  1,  2],
         [ 3,  4,  5]],

        [[ 6,  7,  8],
         [ 9, 10, 11]]])
>>> x.view(2, -1, 3).shape
torch.Size([2, 2, 3])
>>> x.view(4, 3)
tensor([[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11]])
```

> [!warning] Watch out
> `-1` means "work this size out". view needs contiguous memory; after a transpose, use .reshape.

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 04 - Speculative decoding#Code, explained|Day 4]], [[Day 08 - CosyVoice#Code, explained|Day 8]]

### x.reshape

`.reshape(shape)`: like `.view`, but copies if the memory layout requires it. `[6, 32, 64]` → `.reshape(6, 2048)` puts the heads side by side again.

![[g-view.svg]]

```python
>>> import torch
>>> x = torch.arange(6).view(2, 3)
>>> x.t().reshape(6)
tensor([0, 3, 1, 4, 2, 5])
>>> x.reshape(3, 2)
tensor([[0, 1],
        [2, 3],
        [4, 5]])
```

> [!warning] Watch out
> `x.t().view(6)` would raise an error (memory no longer in row order); reshape copies instead.

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]]

### x.transpose

`.transpose(a, b)`: swaps two axes. `[13, 32, 64].transpose(0, 1)` → `[32, 13, 64]` (heads first).

![[g-transpose.svg]]

```python
>>> import torch
>>> x = torch.arange(6).view(3, 2, 1)
>>> x.transpose(0, 1).shape
torch.Size([2, 3, 1])
>>> x.transpose(0, 1)[:, :, 0]
tensor([[0, 2, 4],
        [1, 3, 5]])
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]]

### torch.cat

`torch.cat((a, b), dim=d)`: joins tensors **along an existing axis**. `torch.cat(([1, 2], [3, 4]), dim=-1)` → `[1, 2, 3, 4]`; on `[3, 2]` tensors with `dim=-1` the result is `[3, 4]`.

![[g-cat.svg]]

```python
>>> import torch
>>> a = torch.tensor([[1, 2], [3, 4]])
>>> b = torch.tensor([[5, 6], [7, 8]])
>>> torch.cat((a, b), dim=-1)
tensor([[1, 2, 5, 6],
        [3, 4, 7, 8]])
>>> torch.cat((a, b), dim=0)
tensor([[1, 2],
        [3, 4],
        [5, 6],
        [7, 8]])
>>> torch.cat((a, b), dim=0).shape
torch.Size([4, 2])
```

> [!warning] Watch out
> Every axis except the joined one must match: [2, 2] and [2, 3] join on dim -1 but not on dim 0.

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 04 - Speculative decoding#Code, explained|Day 4]], [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]]

### torch.stack

`torch.stack(xs)`: joins tensors along a **new** axis: four `(V,)` vectors → one `(4, V)` tensor.

![[g-stack.svg]]

```python
>>> import torch
>>> vs = [torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([5, 6])]
>>> torch.stack(vs)
tensor([[1, 2],
        [3, 4],
        [5, 6]])
>>> torch.stack(vs).shape
torch.Size([3, 2])
>>> torch.stack(vs, dim=1).shape
torch.Size([2, 3])
>>> torch.cat(vs).shape
torch.Size([6])
```

*Used in:* [[Day 04 - Speculative decoding#Code, explained|Day 4]], [[Day 08 - CosyVoice#Code, explained|Day 8]]

### x.repeat_interleave

`.repeat_interleave(n, dim)`: repeats **each** element `n` times in place: `[a, b].repeat_interleave(2)` → `[a, a, b, b]`. (Plain `.repeat(2)` would give `[a, b, a, b]`.)

![[g-repeat.svg]]

```python
>>> import torch
>>> k = torch.tensor([10, 20])
>>> k.repeat_interleave(2)
tensor([10, 10, 20, 20])
>>> k.repeat(2)
tensor([10, 20, 10, 20])
>>> kv = torch.zeros(8, 13, 64)
>>> kv.repeat_interleave(4, dim=0).shape
torch.Size([32, 13, 64])
```

> [!warning] Watch out
> For GQA it must be repeat_interleave: query heads 0–3 use KV head 0, not heads 0, 8, 16, 24.

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]]

### x.stride

`.stride(d)`: how many elements you skip in memory to move one step along axis `d`. For a contiguous `[4, 16, 8, 64]` tensor, `stride(0)` = 16 × 8 × 64 = 8192. Kernels use strides to compute addresses.

![[g-stride.svg]]

```python
>>> import torch
>>> x = torch.zeros(2, 3)
>>> x.stride()
(3, 1)
>>> torch.zeros(4, 16, 8, 64).stride()
(8192, 512, 64, 1)
>>> x.t().stride()
(1, 3)
```

> [!warning] Watch out
> A transpose just swaps strides (no data moves), which is why it isn't contiguous afterwards.

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

## 7. Element-wise maths

### x.to

`.to(dtype or device)`: converts a tensor's number type or moves it. `x.to(torch.float32)` makes a float32 copy (from bfloat16); `x.to('cuda')` moves it to the GPU. In Triton code, `.to(tl.float32)` converts a loaded value.

![[g-elementwise.svg]]

```python
>>> import torch
>>> x = torch.tensor([1.5, 2.25], dtype=torch.bfloat16)
>>> x.to(torch.float32)
tensor([1.5000, 2.2500])
>>> torch.tensor([1/3]).to(torch.bfloat16)
tensor([0.3340], dtype=torch.bfloat16)
```

> [!warning] Watch out
> bfloat16 keeps only ~3 significant digits (1/3 → 0.3340); that's why RMSNorm squares in float32.

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### x.float

`.float()`: a float32 copy of the tensor.

```python
>>> import torch
>>> torch.tensor([1, 2]).float()
tensor([1., 2.])
```

*Used in:* [[Day 04 - Speculative decoding#Code, explained|Day 4]], [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### x.pow

`.pow(p)`: raises every element to the power `p`. `x.pow(2)` squares each element.

![[g-elementwise.svg]]

```python
>>> import torch
>>> torch.tensor([1., -2., 3.]).pow(2)
tensor([1., 4., 9.])
>>> torch.tensor([4., 9.]).pow(0.5)
tensor([2., 3.])
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]]

### torch.rsqrt

`torch.rsqrt(x)`: **reciprocal** square root, 1 / √x, element by element. `torch.rsqrt(tensor([4.]))` → 0.5.

![[g-elementwise.svg]]

```python
>>> import torch
>>> torch.rsqrt(torch.tensor([1., 4., 16.]))
tensor([1.0000, 0.5000, 0.2500])
>>> torch.rsqrt(torch.tensor([0.]))
tensor([inf])
```

> [!warning] Watch out
> rsqrt(0) is infinity: that's why RMSNorm adds eps (1e-5) before it.

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]]

### x.cos

`.cos()`: cosine of every element.

![[g-elementwise.svg]]

```python
>>> import torch
>>> torch.tensor([0., 1., 3.1416]).cos()
tensor([ 1.0000,  0.5403, -1.0000])
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### x.sin

`.sin()`: sine of every element.

```python
>>> import torch
>>> torch.tensor([0., 1., 3.1416]).sin()
tensor([ 0.0000,  0.8415, -0.0000])
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]]

### torch.clamp

`torch.clamp(x, min=0)`: clips values: anything below `min` becomes `min`. `clamp([-1, 2], min=0)` → `[0, 2]`.

```python
>>> import torch
>>> torch.clamp(torch.tensor([-1., 0.5, 2.]), min=0)
tensor([0.0000, 0.5000, 2.0000])
>>> torch.clamp(torch.tensor([-1., 0.5, 2.]), min=0, max=1)
tensor([0.0000, 0.5000, 1.0000])
```

*Used in:* [[Day 04 - Speculative decoding#Code, explained|Day 4]]

### nn.functional.silu

`nn.functional.silu(x)`: SiLU, x · sigmoid(x), element by element: smooth ReLU. `silu(1)` = 0.73, `silu(-1)` = −0.27.

```python
>>> import torch
>>> from torch import nn
>>> nn.functional.silu(torch.tensor([-2., -1., 0., 1., 2.]))
tensor([-0.2384, -0.2689,  0.0000,  0.7311,  1.7616])
```

> [!warning] Watch out
> It isn't zero for negative inputs (silu(−1) = −0.27): small negatives pass through, unlike ReLU.

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]]

## 8. Reductions and picking

### x.mean

`.mean(dim, keepdim=True)`: the average along an axis. On `[3, 8]` with `dim=-1` it averages each row's 8 numbers → `[3]`, or `[3, 1]` with `keepdim=True` (keeps the axis so the result broadcasts back over the row).

![[g-mean.svg]]

```python
>>> import torch
>>> x = torch.tensor([[1., 3., 5., 7.], [2., 2., 2., 2.]])
>>> x.mean(-1)
tensor([4., 2.])
>>> x.mean(-1, keepdim=True)
tensor([[4.],
        [2.]])
>>> x.mean(0)
tensor([1.5000, 2.5000, 3.5000, 4.5000])
>>> x / x.mean(-1, keepdim=True)
tensor([[0.2500, 0.7500, 1.2500, 1.7500],
        [1.0000, 1.0000, 1.0000, 1.0000]])
```

> [!warning] Watch out
> Without keepdim, `x / x.mean(-1)` would try [2, 4] / [2] and fail (or broadcast wrongly when sizes happen to match).

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### x.sum

`.sum(dim)`: adds the elements (along `dim` if given).

![[g-sum.svg]]

```python
>>> import torch
>>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])
>>> x.sum()
tensor(21)
>>> x.sum(dim=0)
tensor([5, 7, 9])
>>> x.sum(dim=1)
tensor([ 6, 15])
```

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 04 - Speculative decoding#Code, explained|Day 4]], [[Day 07 - Seminar and speech basics#Code, explained|Day 7]]

### x.argmax

`.argmax(dim=-1)`: the **index** of the largest value along an axis. `tensor([0.1, 2.1, 0.6]).argmax()` = 1. On logits it picks the most likely token (greedy decoding).

![[g-argmax.svg]]

```python
>>> import torch
>>> logits = torch.tensor([0.1, 2.1, 0.6, 1.9])
>>> logits.argmax()
tensor(1)
>>> logits.max()
tensor(2.1000)
>>> batch = torch.tensor([[0.1, 2.1], [3.0, 0.2]])
>>> batch.argmax(dim=-1)
tensor([1, 0])
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 04 - Speculative decoding#Code, explained|Day 4]], [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]]

### torch.softmax

`torch.softmax(x, dim=-1)`: softmax along an axis: each row becomes positive weights that sum to 1. `softmax([0, 0])` = `[0.5, 0.5]`.

![[g-softmax.svg]]

```python
>>> import torch
>>> torch.softmax(torch.tensor([2., 1., 0.]), dim=-1)
tensor([0.6652, 0.2447, 0.0900])
>>> torch.softmax(torch.tensor([0.5, float('-inf')]), dim=-1)
tensor([1., 0.])
>>> torch.softmax(torch.tensor([[2., 1.], [0., 0.]]), dim=-1).sum(-1)
tensor([1., 1.])
```

> [!warning] Watch out
> Use the right dim: over keys (dim=-1) for attention. A row that is all −∞ gives NaN.

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 04 - Speculative decoding#Code, explained|Day 4]]

### x.cumsum

`.cumsum(dim)`: running total. `[0.5, 0.3, 0.2].cumsum()` = `[0.5, 0.8, 1.0]`.

![[g-cumsum.svg]]

```python
>>> import torch
>>> torch.tensor([0.5, 0.3, 0.2]).cumsum(-1)
tensor([0.5000, 0.8000, 1.0000])
>>> torch.tensor([1, 1, 1, 1]).cumsum(0)
tensor([1, 2, 3, 4])
```

*Used in:* [[Day 04 - Speculative decoding#Code, explained|Day 4]]

### torch.topk

`torch.topk(x, k)`: the `k` largest values (and their indices). `.values[..., -1:]` is the k-th largest, used as a cut-off.

![[g-topk.svg]]

```python
>>> import torch
>>> x = torch.tensor([0.1, 0.9, 0.4, 0.7, 0.2])
>>> r = torch.topk(x, 2)
>>> r.values
tensor([0.9000, 0.7000])
>>> r.indices
tensor([1, 3])
```

*Used in:* [[Day 04 - Speculative decoding#Code, explained|Day 4]]

### torch.sort

`torch.sort(x, descending=True)`: sorted values and the indices they came from.

![[g-topk.svg]]

```python
>>> import torch
>>> x = torch.tensor([0.1, 0.9, 0.4])
>>> v, i = torch.sort(x, descending=True)
>>> v
tensor([0.9000, 0.4000, 0.1000])
>>> i
tensor([1, 2, 0])
```

*Used in:* [[Day 04 - Speculative decoding#Code, explained|Day 4]]

### torch.matmul

`torch.matmul(a, b) (also a @ b)`: matrix multiplication. For `[32, 1, 64] @ [32, 64, 13]`, it does 32 separate `[1, 64] × [64, 13]` products (one per head) → `[32, 1, 13]`.

![[g-matmul.svg]]

```python
>>> import torch
>>> A = torch.tensor([[1, 2, 0], [0, 1, 3]])
>>> B = torch.tensor([[1, 0], [2, 1], [0, 2]])
>>> A @ B
tensor([[5, 2],
        [2, 7]])
>>> (torch.zeros(32, 1, 64) @ torch.zeros(32, 64, 13)).shape
torch.Size([32, 1, 13])
```

> [!warning] Watch out
> Inner sizes must match: [2, 3] @ [3, 2] works, [2, 3] @ [2, 3] fails. Use k.transpose(-2, -1) for q @ kᵀ.

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]]

## 9. Writing into tensors

### x.masked_fill

`.masked_fill(mask, value)`: replaces elements where `mask` is True with `value`. `scores.masked_fill(~allowed, -inf)` hides disallowed keys; `~` flips True/False.

![[g-masked-fill.svg]]

```python
>>> import torch
>>> s = torch.tensor([[0.5, 0.1], [0.2, 0.4]])
>>> mask = torch.tensor([[False, True], [False, False]])
>>> s.masked_fill(mask, float('-inf'))
tensor([[0.5000,   -inf],
        [0.2000, 0.4000]])
>>> s.masked_fill(~mask, 0.0)
tensor([[0.0000, 0.1000],
        [0.0000, 0.0000]])
```

> [!warning] Watch out
> It returns a new tensor; `masked_fill_` (with _) changes s in place.

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 04 - Speculative decoding#Code, explained|Day 4]]

### x.scatter_

`.scatter_(dim, index, value)`: writes `value` at the positions given by `index`, in place. `zeros.scatter_(-1, argmax, 1.0)` makes a one-hot vector.

![[g-scatter.svg]]

```python
>>> import torch
>>> idx = torch.tensor([[1], [3]])
>>> torch.zeros(2, 4).scatter_(-1, idx, 1.0)
tensor([[0., 1., 0., 0.],
        [0., 0., 0., 1.]])
```

*Used in:* [[Day 04 - Speculative decoding#Code, explained|Day 4]], [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]]

### x.scatter

`.scatter(dim, index, src)`: like `scatter_` but returns a new tensor: puts `src` values back at the `index` positions (undoing a sort).

```python
>>> import torch
>>> vals = torch.tensor([0.9, 0.4, 0.1])
>>> order = torch.tensor([1, 2, 0])
>>> torch.zeros(3).scatter(0, order, vals)
tensor([0.1000, 0.9000, 0.4000])
```

> [!warning] Watch out
> It's the inverse of indexing by `order`: sorted values go back to their original positions.

*Used in:* [[Day 04 - Speculative decoding#Code, explained|Day 4]]

### x.fill_

`.fill_(v)`: sets every element to `v`, in place (the tensor keeps its memory address, which CUDA graphs require).

```python
>>> import torch
>>> x = torch.zeros(3)
>>> p = x.data_ptr()
>>> x.fill_(7.0)
tensor([7., 7., 7.])
>>> x.data_ptr() == p
True
```

*Used in:* [[Day 04 - Speculative decoding#Code, explained|Day 4]]

### x.tolist

`.tolist()`: converts a tensor to a plain Python list (copies to the CPU).

```python
>>> import torch
>>> torch.tensor([[5, 2], [7, 1]]).tolist()
[[5, 2], [7, 1]]
```

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### x.item

`.item()`: the Python number inside a one-element tensor.

```python
>>> import torch
>>> torch.tensor([2.5]).item()
2.5
>>> torch.tensor(3).item() + 1
4
```

> [!warning] Watch out
> On a GPU tensor, .item() waits for the GPU (a hidden sync); in a loop this costs time.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

## 10. Running models and the GPU

### torch.inference_mode

`torch.inference_mode()`: a `with` block in which PyTorch records no gradients: less memory and faster, for inference only.

```python
>>> import torch
>>> w = torch.ones(2, requires_grad=True)
>>> with torch.inference_mode():
...     y = w * 2
>>> y.requires_grad
False
```

*Used in:* [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]]

### torch.cuda.synchronize

`torch.cuda.synchronize()`: waits until the GPU has finished all queued work. GPU calls return immediately, so you must synchronize before reading a clock, or you'd time only the launch.

```python
import torch
torch.cuda.is_available()  # no GPU here; on a GPU:
# t0 = time.perf_counter(); out = model(x)
# torch.cuda.synchronize(); ms = (time.perf_counter() - t0) * 1e3
```

> [!warning] Watch out
> Without it you measure ~0.1 ms (the launch), whatever the real GPU time is.

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 04 - Speculative decoding#Code, explained|Day 4]]

### torch.compile

`torch.compile(fn, mode='reduce-overhead')`: compiles a function into fused GPU kernels; `reduce-overhead` also records **CUDA graphs**, so a whole step replays with one launch.

```python
>>> import torch
>>> def f(x):
...     return torch.relu(x) * 2
>>> cf = torch.compile(f)  # compiled on the first call
>>> f(torch.tensor([-1., 3.]))
tensor([0., 6.])
```

> [!warning] Watch out
> The first call is slow (it compiles); new input shapes can trigger a recompile.

*Used in:* [[Day 04 - Speculative decoding#Code, explained|Day 4]]

### model.generate

`.generate(**inputs, max_new_tokens=n)`: Hugging Face's built-in generation loop: runs the model repeatedly, appending tokens, until `n` new tokens (or EOS). The baseline uses it.

```python
# Hugging Face (needs a model, so not run here):
# out = model.generate(**tok('Hello', return_tensors='pt'), max_new_tokens=20, do_sample=False)
# out.shape == (1, prompt_len + 20)  # prompt + new tokens, unless EOS came first
```

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### cache.crop

`.crop(n)`: on a Hugging Face `DynamicCache`: drops cached positions (negative `n` = remove that many from the end).

```python
# Hugging Face DynamicCache (not run here):
# cache.get_seq_length()   -> 12
# cache.crop(10)            # keep the first 10 positions
# cache.get_seq_length()   -> 10
```

> [!warning] Watch out
> Day 4 uses it to throw away the KV of rejected draft tokens.

*Used in:* [[Day 04 - Speculative decoding#Code, explained|Day 4]]

## 11. Triton (GPU kernels)

### tl.program_id

`tl.program_id(axis)`: which program (a block of GPU threads) this is, along a grid axis. With `grid = (num_seqs, num_heads)`, `program_id(0)` is the sequence and `program_id(1)` the head.

Triton code only runs inside a GPU kernel, so the example shows the same idea in plain PyTorch.

![[g-program-id.svg]]

```python
>>> grid = (2, 3)  # (num_seqs, num_heads)
>>> for seq in range(grid[0]):
...     for head in range(grid[1]):
...         pass  # inside the kernel: seq = tl.program_id(0), head = tl.program_id(1)
>>> grid[0] * grid[1]
6
```

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### tl.arange

`tl.arange(0, N)`: a vector `[0, 1, …, N−1]` inside a Triton program (N must be a power of two). Used to build many addresses at once.

Triton code only runs inside a GPU kernel, so the example shows the same idea in plain PyTorch.

```python
>>> import torch
>>> BLOCK = 4
>>> offs = torch.arange(0, BLOCK)  # tl.arange(0, BLOCK)
>>> 8 + offs  # addresses 8..11
tensor([ 8,  9, 10, 11])
```

> [!warning] Watch out
> N must be a power of two (16, 32, 64, …); mask off the extra lanes.

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### tl.load

`tl.load(ptr + offsets, mask=m, other=0)`: reads many values from GPU memory at once (one per offset). Where `mask` is False it doesn't read and returns `other` instead, so out-of-range addresses are never touched.

Triton code only runs inside a GPU kernel, so the example shows the same idea in plain PyTorch.

![[g-tl-load.svg]]

```python
>>> import torch
>>> mem = torch.tensor([1., 2., 3., 4., 5., 6.])
>>> offs = 4 + torch.arange(4)
>>> mask = offs < 6
>>> mask
tensor([ True,  True, False, False])
>>> torch.where(mask, mem[offs.clamp(max=5)], 0.0)
tensor([5., 6., 0., 0.])
```

> [!warning] Watch out
> Without the mask, lanes 6 and 7 would read past the buffer: garbage or an illegal-address crash.

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### tl.store

`tl.store(ptr + offsets, values)`: writes values to GPU memory.

Triton code only runs inside a GPU kernel, so the example shows the same idea in plain PyTorch.

```python
>>> import torch
>>> out = torch.zeros(4)
>>> offs = torch.arange(4)
>>> out[offs] = torch.tensor([9., 8., 7., 6.])
>>> out
tensor([9., 8., 7., 6.])
```

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### tl.zeros

`tl.zeros([N], dtype)`: a vector of N zeros held in registers.

Triton code only runs inside a GPU kernel, so the example shows the same idea in plain PyTorch.

```python
>>> import torch
>>> acc = torch.zeros(4, dtype=torch.float32)  # tl.zeros([4], dtype=tl.float32)
>>> acc
tensor([0., 0., 0., 0.])
```

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### tl.sum

`tl.sum(x, axis)`: adds a vector (or a 2-D block along `axis`) inside the program.

Triton code only runs inside a GPU kernel, so the example shows the same idea in plain PyTorch.

![[g-sum.svg]]

```python
>>> import torch
>>> q = torch.tensor([1., 2.]); k = torch.tensor([[3., 4.], [0., 1.]])
>>> q = torch.tensor([1., 2.]); k = torch.tensor([[3., 4.], [0., 1.]])
>>> (q[None, :] * k).sum(1)  # tl.sum(q[None, :] * k, axis=1)
tensor([11.,  2.])
```

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### tl.max

`tl.max(x, axis)`: the largest element (along `axis`).

Triton code only runs inside a GPU kernel, so the example shows the same idea in plain PyTorch.

```python
>>> import torch
>>> torch.tensor([0.2, 1.5, -3.0]).max()
tensor(1.5000)
```

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### tl.maximum

`tl.maximum(a, b)`: element-wise larger of two values (like `max`, but for tensors).

Triton code only runs inside a GPU kernel, so the example shows the same idea in plain PyTorch.

```python
>>> import torch
>>> torch.maximum(torch.tensor([1., 5.]), torch.tensor([3., 2.]))
tensor([3., 5.])
```

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### tl.exp

`tl.exp(x)`: e^x, element by element.

Triton code only runs inside a GPU kernel, so the example shows the same idea in plain PyTorch.

```python
>>> import torch
>>> torch.exp(torch.tensor([0., 1., -1.]))
tensor([1.0000, 2.7183, 0.3679])
```

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### tl.where

`tl.where(cond, a, b)`: picks `a` where `cond` is True and `b` elsewhere. `tl.where(valid, scores, -inf)` masks the slots past the context.

Triton code only runs inside a GPU kernel, so the example shows the same idea in plain PyTorch.

![[g-where.svg]]

```python
>>> import torch
>>> valid = torch.tensor([True, True, False])
>>> scores = torch.tensor([0.3, 0.9, 0.5])
>>> torch.where(valid, scores, float('-inf'))
tensor([0.3000, 0.9000,   -inf])
```

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

## 12. Triton kernels and asyncio futures (Day 3: decode kernel, server)

### triton.jit

`@triton.jit
def kernel(...): ...
kernel[grid](args, BLOCK_N=128, num_warps=4)`: turns a Python function into a **GPU kernel** that Triton compiles on first use (one compile per new set of constexpr values). You launch it with square brackets: `kernel[grid](...)` runs one program per grid cell. `grid = (2, 32)` launches 64 programs.

```cpp
# (needs a GPU; shown for the launch syntax)
@triton.jit
def add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)

add[(3,)](x, y, out, 10, BLOCK=4)   # 3 programs cover positions 0..11; 10, 11 masked
```

> [!warning] Watch out
> The first call compiles (slow); time a kernel only after a warm-up call. The server's first request paid this (17 s total).

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### tl.constexpr

`HEAD_DIM: tl.constexpr`: marks a kernel argument as a **compile-time constant**. Triton specialises the kernel for its value, so it can size vectors: `tl.arange(0, HEAD_DIM)` needs a constant (and a power of two). Changing it triggers a new compile.

Triton code only runs inside a GPU kernel, so the example shows the same idea in plain PyTorch.

```python
# HEAD_DIM=64 and HEAD_DIM=128 compile two different kernels
>>> 64 & 63      # the wrapper's power-of-two check: 0 means OK
0
>>> 48 & 47
32
```

> [!warning] Watch out
> A constexpr that changes every call (e.g. a context length) would recompile every call. Pass varying sizes as normal arguments.

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### x.create_future

`loop.create_future()`: makes an empty asyncio **Future**: a box that will hold a result later. `await fut` sleeps until someone calls `fut.set_result(x)` (or `fut.set_exception(e)`). The server gives each request one.

```python
>>> import asyncio
>>> async def main():
...     loop = asyncio.get_running_loop()
...     fut = loop.create_future()
...     print(fut.done())
...     loop.call_later(0.01, fut.set_result, 'r7 done')
...     print(await fut)
>>> asyncio.run(main())
False
r7 done
```

> [!warning] Watch out
> Make futures from inside a running loop (`get_running_loop`); a future from another loop can't be awaited here.

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### x.set_exception

`.set_exception(exc)`: on a Future: completes it with an **error** instead of a value; whoever `await`s it gets `exc` raised. The engine loop uses it to fail every waiter when it dies, so nobody hangs.

```python
>>> async def main():
...     fut = asyncio.get_running_loop().create_future()
...     fut.set_exception(RuntimeError('engine loop died'))
...     try:
...         await fut
...     except RuntimeError as e:
...         print('caught:', e)
>>> asyncio.run(main())
caught: engine loop died
```

> [!warning] Watch out
> Calling set_result or set_exception twice raises InvalidStateError; the server checks `future.done()` first.

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### ValueError

`raise ValueError(msg)`: Python's built-in error for "right type, wrong value". `raise` stops the function and passes the error up to the caller. The kernel wrapper raises it for a bad shape (e.g. head_dim not a power of two) instead of computing a wrong answer.

```python
>>> def check(n):
...     if n > 8192:
...         raise ValueError(f'prompt of {n} tokens exceeds the 8192-token budget')
...     return 'ok'
>>> check(10)
'ok'
>>> check(9000)
Traceback (most recent call last):
  ...
ValueError: prompt of 9000 tokens exceeds the 8192-token budget
```

> [!warning] Watch out
> Raise before changing any state, so the caller can retry cleanly (the server's add-first rule).

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 04 - Speculative decoding#Code, explained|Day 4]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]]

### RuntimeError

`raise RuntimeError(msg)`: Python's built-in error for "something is wrong at run time". `submit` raises it when the engine loop is dead, so a handler fails fast instead of waiting forever.

```python
>>> try:
...     raise RuntimeError('engine loop is not running')
... except RuntimeError as e:
...     print('503:', e)
503: engine loop is not running
```

> [!warning] Watch out
> Catch specific errors; a bare `except:` would also swallow CancelledError and break shutdown.

*Used in:* [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

### tokenizer

`tokenizer(texts, return_tensors='pt', padding=True)`: a Hugging Face tokenizer, called on a list of strings: turns each into token ids and pads them to one length. Returns a dict with `input_ids` and `attention_mask` (1 = real token, 0 = padding), both `[batch, longest]` tensors. `attention_mask.sum(dim=1)` gives each prompt's real length.

```cpp
enc = tokenizer(['Hi', 'Tell me about GPUs'], return_tensors='pt', padding=True)
# enc['input_ids'].shape      -> [2, longest prompt]
# enc['attention_mask']        -> 1 for real tokens, 0 for padding
# enc['attention_mask'].sum(dim=1).tolist() -> each prompt's real length (serving.py:106)
```

> [!warning] Watch out
> Padding makes every prompt in a static batch the same length: the padded positions still cost compute in the HF baseline.

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

## 13. torch.compile settings (Day 4: speculative decoding)

### torch._dynamo.config.cache_size_limit

`torch._dynamo.config.cache_size_limit = 64`: how many compiled versions of one function `torch.compile` keeps (one per input shape or guard change). The default (8) is too few for 4 models × several decode lengths; past the limit, calls **silently run eager**, and the speedup vanishes without an error.

```cpp
torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
# shapes captured per model: 1, 2 and gamma + 1 (5 at gamma = 4)
```

> [!warning] Watch out
> Nothing warns you when the limit is hit; measure per-forward time after compiling to confirm it really dropped.

## 14. CUDA C++ basics (Day 5: matmul kernels)

### CUDA global

`__global__ void kernel(...)`: marks a function as a **kernel**: it runs on the GPU and is started from the CPU with `kernel<<<grid, block>>>(args)`. Every thread of the launch runs the same body.

```python
__global__ void add1(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.f;
}
add1<<<4, 256>>>(x, 1000);   // 1,024 threads, 1,000 do work
```

> [!warning] Watch out
> A launch returns immediately; the kernel runs later. Errors show up at the next sync or CUDA call.

*Used in:* [[Day 05 - CUDA I#Code, explained|Day 5]]

### threadIdx

`threadIdx.x / .y / .z`: this thread's position **inside its block**. In a 32 × 32 block, `threadIdx.x` and `threadIdx.y` each go 0…31. A warp = 32 consecutive `threadIdx.x` with one `threadIdx.y`.

```cpp
// block of 32 x 32 threads
// warp 0 = threadIdx.y 0, threadIdx.x 0..31
// warp 1 = threadIdx.y 1, threadIdx.x 0..31
```

> [!warning] Watch out
> x varies fastest. Whatever index is computed from threadIdx.x decides whether a warp's loads coalesce.

*Used in:* [[Day 05 - CUDA I#Code, explained|Day 5]], [[Day 06 - CUDA II#Code, explained|Day 6]]

### blockIdx

`blockIdx.x / .y / .z`: this block's position **inside the grid**. With 32 × 32 blocks at N = 32768, `blockIdx.x` goes 0…1023.

```cpp
dim3 grid(1024, 1024);   // N = 32768, 32x32 blocks
// blockIdx.x in 0..1023, blockIdx.y in 0..1023
```

> [!warning] Watch out
> Blocks run in no guaranteed order and can't wait for each other inside a kernel.

*Used in:* [[Day 05 - CUDA I#Code, explained|Day 5]], [[Day 06 - CUDA II#Code, explained|Day 6]]

### blockDim

`blockDim.x / .y`: the block's size along each axis (32 for a 32 × 32 block). So `blockIdx.x * blockDim.x + threadIdx.x` is the global index: `2 · 32 + 5 = 69`.

```cpp
// blockDim = (32, 32)
int col = blockIdx.x * blockDim.x + threadIdx.x;   // 2*32 + 5 = 69
```

> [!warning] Watch out
> Inside templated kernels the code often uses the constant TILE instead; they must match the launch.

*Used in:* [[Day 05 - CUDA I#Code, explained|Day 5]]

### fmaf

`fmaf(a, b, c)`: fused multiply-add on floats: `a · b + c` in one instruction with one rounding. `fmaf(2.f, 3.f, 1.f)` = 7. Counts as 2 FLOPs.

```cpp
float acc = 0.f;
acc = fmaf(2.f, 3.f, acc);   // 6
acc = fmaf(1.f, 4.f, acc);   // 10
```

> [!warning] Watch out
> One rounding instead of two, so results can differ from a*b+c in the last bit (cuBLAS also uses FMA).

*Used in:* [[Day 05 - CUDA I#Code, explained|Day 5]], [[Day 06 - CUDA II#Code, explained|Day 6]]

### CUDA syncthreads

`__syncthreads()`: a barrier for **one block**: no thread continues until every thread of the block has arrived. Used before reading a shared tile (everyone has written it) and before overwriting it (everyone has read it).

```cpp
sA[ty][tx] = A[...];
__syncthreads();          // tile complete
for (k...) acc += sA[ty][k] * sB[k][tx];
__syncthreads();          // everyone done reading
```

> [!warning] Watch out
> Every thread of the block must reach it; a `return` before it in some threads can hang the block.

*Used in:* [[Day 05 - CUDA I#Code, explained|Day 5]], [[Day 06 - CUDA II#Code, explained|Day 6]]

### CUDA shared

`__shared__ float sA[TILE][TILE];`: puts an array in the SM's **shared memory**: on-chip, fast, visible to all threads of the block, gone when the block ends. Two 64 × 64 float tiles = 32 KB.

```cpp
__shared__ float s[32][33];   // 33: padded to avoid bank conflicts
s[threadIdx.y][threadIdx.x] = g[...];
```

> [!warning] Watch out
> Limited (up to 228 KB per SM); more per block means fewer blocks per SM.

*Used in:* [[Day 05 - CUDA I#Code, explained|Day 5]], [[Day 06 - CUDA II#Code, explained|Day 6]]

### template

`template<int TILE> __global__ void k(...)`: makes the kernel generic over a **compile-time** number. `mm_tiled_1to1<32>` and `<64>` are compiled separately, so `TILE` can size static arrays and loops can be fully unrolled.

```python
template<int R> __global__ void k(...) { float reg[R][R]; ... }
k<4><<<grid, blk>>>(...);   // R = 4 at compile time
```

> [!warning] Watch out
> Each template value is a separate compiled kernel; the value can't come from a runtime variable.

*Used in:* [[Day 05 - CUDA I#Code, explained|Day 5]], [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### size_t

`(size_t)row * N`: an unsigned 64-bit integer type. Casting before multiplying keeps `row · N` from overflowing 32-bit `int` (2.1 × 10⁹) for big N.

```cpp
int N = 50000;
int bad = N * N;            // overflows (2.5e9 > 2.1e9)
size_t ok = (size_t)N * N;  // 2,500,000,000
```

> [!warning] Watch out
> Cast before the multiply: `(size_t)(N * N)` overflows first, then casts.

*Used in:* [[Day 05 - CUDA I#Code, explained|Day 5]], [[Day 06 - CUDA II#Code, explained|Day 6]], [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### pragma unroll

`#pragma unroll
for (int k = 0; k < TILE; ++k) ...`: asks the compiler to copy the loop body once per iteration (no loop counter, no branch). With a compile-time trip count like `TILE` or `R`, small arrays such as `reg[R][R]` can then live in registers.

```cpp
#pragma unroll
for (int i = 0; i < 4; ++i) av[i] = sA[ty*4 + i][k];
// becomes 4 straight-line loads, av[] in registers
```

> [!warning] Watch out
> With a runtime trip count it can't fully unroll; arrays indexed by a loop variable may spill to slow local memory.

*Used in:* [[Day 05 - CUDA I#Code, explained|Day 5]], [[Day 06 - CUDA II#Code, explained|Day 6]]

### cudaMemcpy

`cudaMemcpy(dst, src, bytes, kind)`: copies memory between CPU and GPU. `kind` is `cudaMemcpyHostToDevice` (upload) or `cudaMemcpyDeviceToHost` (download). It waits until the copy is done.

```cpp
cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);   // CPU -> GPU
cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);   // GPU -> CPU
```

> [!warning] Watch out
> Device-to-host waits for earlier kernels to finish, so it doubles as a sync.

*Used in:* [[Day 05 - CUDA I#Code, explained|Day 5]]

### cudaEventCreate

`cudaEventCreate(&ev)`: makes a CUDA **event**: a marker you can drop into the GPU's work queue to read the GPU's own clock.

```cpp
cudaEvent_t beg, end;
cudaEventCreate(&beg); cudaEventCreate(&end);
```

> [!warning] Watch out
> Create once, reuse; destroy when done.

### cudaEventRecord

`cudaEventRecord(ev)`: queues the event; it gets a timestamp when the GPU reaches it (not when the CPU calls this).

```python
cudaEventRecord(beg);
kernel<<<g, b>>>(...);
cudaEventRecord(end);
```

> [!warning] Watch out
> Records into the default stream unless you pass a stream; time only what's between the two in that queue.

*Used in:* [[Day 05 - CUDA I#Code, explained|Day 5]]

### cudaEventSynchronize

`cudaEventSynchronize(ev)`: makes the CPU wait until the GPU has reached the event. Needed before reading the elapsed time.

```cpp
cudaEventRecord(end);
cudaEventSynchronize(end);   // CPU waits here
```

> [!warning] Watch out
> Without it, ElapsedTime may return an error (event not yet reached).

*Used in:* [[Day 05 - CUDA I#Code, explained|Day 5]]

### cudaEventElapsedTime

`cudaEventElapsedTime(&ms, beg, end)`: milliseconds of GPU time between two recorded events. Kernel-only timing: no Python, no copies.

```cpp
float ms;
cudaEventElapsedTime(&ms, beg, end);   // e.g. 2736.6 for block2d at N = 32768
```

> [!warning] Watch out
> Resolution about 0.5 µs; for tiny kernels time many launches and divide.

*Used in:* [[Day 05 - CUDA I#Code, explained|Day 5]]

### cudaEventDestroy

`cudaEventDestroy(ev)`: frees an event.

```cpp
cudaEventDestroy(beg); cudaEventDestroy(end);
```

> [!warning] Watch out
> Leaking events is harmless in a benchmark but bad in a server.

*Used in:* [[Day 05 - CUDA I#Code, explained|Day 5]]

### cublasSgemm

`cublasSgemm(h, opA, opB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc)`: NVIDIA's single-precision matmul: C = α·op(A)·op(B) + β·C, with matrices in **column-major** order. Passing (B, A) instead of (A, B) gives a row-major A·B for free.

```cpp
// row-major C = A*B via column-major cuBLAS:
cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
            &one, dB, N, dA, N, &zero, dC, N);
```

> [!warning] Watch out
> Column-major: a row-major matrix looks transposed to cuBLAS. Swap the operands rather than transposing.

*Used in:* [[Day 05 - CUDA I#Code, explained|Day 5]]

### CUDA_CHECK

`CUDA_CHECK(call)`: a macro defined in this file: runs a CUDA call and, if it returns an error, prints the file, line and message and exits. Without it, CUDA errors are silent until something later fails.

```cpp
CUDA_CHECK(cudaMalloc(&d, bytes));
// on failure: "CUDA error file.cu:123: out of memory" and exit(1)
```

> [!warning] Watch out
> Kernel launches return nothing; check them with CUDA_CHECK(cudaGetLastError()).

*Used in:* [[Day 05 - CUDA I#Code, explained|Day 5]], [[Day 06 - CUDA II#Code, explained|Day 6]]

### fprintf

`fprintf(stream, fmt, ...)`: C's formatted print to a file or `stderr`/`stdout`. `fprintf(out, "%s,%d\n", name, N)` writes one CSV line.

```cpp
fprintf(stderr, "N=%d time=%.4f ms\n", 1024, 0.0973);
// N=1024 time=0.0973 ms
```

> [!warning] Watch out
> %zu for size_t, %d for int; a mismatch prints garbage.

*Used in:* [[Day 05 - CUDA I#Code, explained|Day 5]]

### std::min

`std::min(a, b)`: the smaller of two values (C++). `std::min(elems, (size_t)4000000)` caps the check at 4M elements; both arguments must have the same type, hence the cast.

```cpp
size_t elems = (size_t)N * N;
int n = (int)std::min(elems, (size_t)4000000);   // N = 32768 -> 4,000,000
```

> [!warning] Watch out
> Both arguments must be the same type, or it won't compile.

*Used in:* [[Day 05 - CUDA I#Code, explained|Day 5]], [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]], [[Day 14 - Raft#Code, explained|Day 14]]

## 15. CUDA runtime, streams and warp intrinsics (Day 6)

### CUDA kernel launch

`kernel<<<grid, block, shared_bytes, stream>>>(args)`: CUDA's launch syntax. `grid` and `block` are sizes (int or `dim3`); `shared_bytes` is the dynamic shared memory per block; `stream` is the queue to launch into (default stream if omitted).

```python
mm_thin<256,32><<<grd, blk>>>(...);              // default stream
mm_thin<256,32><<<g, blk, 0, st[s]>>>(...);     // stream s, no dynamic shared
flash_forward_kernel<<<grid, block, 28800>>>(...);  // 28,800 B shared
```

> [!warning] Watch out
> The launch is asynchronous: it returns before the kernel runs.

*Used in:* [[Day 06 - CUDA II#Code, explained|Day 6]]

### dim3

`dim3 grid(x, y, z)`: a 3-number size for grids and blocks; missing numbers are 1. `dim3 grd((N + 255) / 256, 32)` = 128 × 32 blocks at N = 32768.

```cpp
dim3 grid(256, 16, 8);   // Tr, heads, batch: 32,768 blocks
dim3 block(32, 8);       // 8 warps of 32 lanes
```

> [!warning] Watch out
> A block may have at most 1,024 threads in total.

*Used in:* [[Day 06 - CUDA II#Code, explained|Day 6]]

### cudaStreamCreate

`cudaStreamCreate(&s)`: makes a new **stream**: an ordered queue of GPU work. Work in different streams may overlap if SMs are free.

```cpp
cudaStream_t st[4];
for (int i = 0; i < 4; ++i) cudaStreamCreate(&st[i]);
```

> [!warning] Watch out
> Streams don't make work faster by themselves; overlap needs idle SMs.

### cudaStreamSynchronize

`cudaStreamSynchronize(s)`: the CPU waits until everything queued in stream `s` has finished.

```cpp
for (int s = 0; s < 4; ++s) cudaStreamSynchronize(st[s]);   // layer-1 barrier
```

> [!warning] Watch out
> Inside a timed loop it adds a host round-trip per call.

*Used in:* [[Day 06 - CUDA II#Code, explained|Day 6]]

### cudaStreamDestroy

`cudaStreamDestroy(s)`: frees a stream (after its work completes).

```cpp
for (int i = 0; i < 4; ++i) cudaStreamDestroy(st[i]);
```

### cudaDeviceSynchronize

`cudaDeviceSynchronize()`: the CPU waits until **all** GPU work in all streams has finished.

```cpp
for (int i = 0; i < 3; ++i) fn();   // warm-up
cudaDeviceSynchronize();          // start timing from an idle GPU
```

> [!warning] Watch out
> Heavy hammer: it waits for every stream.

*Used in:* [[Day 06 - CUDA II#Code, explained|Day 6]]

### cudaMalloc

`cudaMalloc(&ptr, bytes)`: allocates GPU global memory. `cudaMalloc(&A2, n2 * sizeof(float))` reserves 64 MiB for a 4096 × 4096 float tensor.

```cpp
float* A2;
cudaMalloc(&A2, (size_t)4096 * 4096 * sizeof(float));   // 64 MiB
```

> [!warning] Watch out
> Slow (milliseconds); allocate once up front, never inside a hot loop.

*Used in:* [[Day 06 - CUDA II#Code, explained|Day 6]]

### cudaFree

`cudaFree(ptr)`: releases memory from `cudaMalloc`.

```cpp
cudaFree(A2);
```

> [!warning] Watch out
> Implicitly synchronises the device.

### cudaMemGetInfo

`cudaMemGetInfo(&free, &total)`: free and total GPU memory in bytes. Calling it before and after allocating gives the bytes a program holds: 962 vs 834 MiB here.

```cpp
size_t f0, f1, tot;
cudaMemGetInfo(&f0, &tot);
/* ... all cudaMallocs ... */
cudaMemGetInfo(&f1, &tot);
size_t peak = f0 - f1;   // 1,008,730,112 B = 962 MiB (baseline)
```

> [!warning] Watch out
> It sees the whole device, so other processes' allocations change it too.

*Used in:* [[Day 06 - CUDA II#Code, explained|Day 6]]

### sizeof

`sizeof(float)`: size of a type in bytes: `sizeof(float)` = 4, `sizeof(double)` = 8.

```cpp
size_t n1 = (size_t)4096 * 4096;
n1 * sizeof(float)   // 67,108,864 bytes = 64 MiB
```

> [!warning] Watch out
> Multiply element counts by sizeof before cudaMalloc/cudaMemcpy, which take bytes.

*Used in:* [[Day 06 - CUDA II#Code, explained|Day 6]], [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### CUDA extern shared

`extern __shared__ float smem[];`: **dynamic** shared memory: its size isn't fixed in the code but given at launch (`<<<grid, block, 28800>>>`). The kernel carves it into Qi, Kj, Vj, S by pointer offsets.

```cpp
extern __shared__ float smem[];
float* Qi = smem;
float* Kj = Qi + 32 * 64;
float* Vj = Kj + 32 * 65;
float* S  = Vj + 32 * 64;
```

> [!warning] Watch out
> The launch must pass enough bytes, or the offsets run past the end silently.

*Used in:* [[Day 06 - CUDA II#Code, explained|Day 6]]

### CUDA restrict

`const float* __restrict__ Q`: a promise to the compiler that this pointer doesn't overlap any other; it may then cache loads more aggressively.

```cpp
void f(const float* __restrict__ a, float* __restrict__ out)
```

> [!warning] Watch out
> If the pointers do overlap, results are undefined.

### CUDA launch_bounds

`__global__ __launch_bounds__(256) void k(...)`: tells the compiler the kernel never runs with more than 256 threads per block, so it can budget registers per thread for that.

```cpp
__global__ __launch_bounds__(256) void flash_forward_kernel(...)
```

> [!warning] Watch out
> Launching with more threads than promised fails.

### fmaxf

`fmaxf(a, b)`: the larger of two floats. `fmaxf(2.f, 3.f)` = 3; `fmaxf(-INFINITY, 1.f)` = 1.

```cpp
fmaxf(-INFINITY, 2.f)   // 2
fmaxf(3.f, 2.f)         // 3
```

> [!warning] Watch out
> fmaxf(NaN, x) returns x, so a NaN can hide instead of propagating.

*Used in:* [[Day 06 - CUDA II#Code, explained|Day 6]]

### CUDA expf

`__expf(x)`: a fast, slightly less precise hardware e^x for floats. `__expf(0.f)` = 1, `__expf(-INFINITY)` = 0.

```cpp
__expf(0.f)          // 1
__expf(-1.f)         // 0.3679
__expf(-INFINITY)    // 0  (e_run on the first tile)
```

> [!warning] Watch out
> Less accurate than expf for large |x|; fine after subtracting the max.

*Used in:* [[Day 06 - CUDA II#Code, explained|Day 6]]

### INFINITY

`INFINITY / -INFINITY`: the float infinity constants. `m_run = -INFINITY` starts the running max below every score.

```cpp
float m = -INFINITY;
m = fmaxf(m, 2.f);   // 2
```

> [!warning] Watch out
> Memset to 0xFF gives NaN, not -inf (story S5).

*Used in:* [[Day 06 - CUDA II#Code, explained|Day 6]]

### CUDA shfl_xor_sync

`__shfl_xor_sync(mask, v, off)`: returns `v` from lane `lane XOR off` of the same warp, register to register. With off = 16, 8, 4, 2, 1 and `fmaxf`, every lane ends with the warp's max in 5 steps. `mask = 0xffffffff` = all 32 lanes take part.

```cpp
// 8-lane picture, values [3,1,4,1,5,9,2,6]
// off=4: [5,9,4,6,5,9,4,6]
// off=2: [5,9,5,9,5,9,5,9]
// off=1: [9,9,9,9,9,9,9,9]
```

> [!warning] Watch out
> All lanes named in mask must execute the same call, or the result is undefined.

*Used in:* [[Day 06 - CUDA II#Code, explained|Day 6]]

### CUDA syncwarp

`__syncwarp()`: a barrier for **one warp** only (cheaper than `__syncthreads`). Enough when each warp reads only the shared data it wrote itself.

```cpp
S[(r0 + r) * 32 + lane] = p;
__syncwarp();   // the warp's own rows of S are complete
... read S[(r0 + r) * 32 + c] ...
```

> [!warning] Watch out
> Only safe if no other warp reads that data.

### cudaFuncSetAttribute

`cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`: raises a kernel's dynamic shared-memory limit. Above 48 KB per block the opt-in is required; 28,800 B doesn't need it.

```cpp
cudaFuncSetAttribute(flash_forward_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
```

> [!warning] Watch out
> Needed only above 48 KB per block.

### cudaGetLastError

`cudaGetLastError()`: returns (and clears) the last CUDA error, e.g. a bad launch configuration. Kernel launches report errors only this way.

```python
kernel<<<g, b>>>(...);
int err = (int)cudaGetLastError();   // 0 = cudaSuccess
```

> [!warning] Watch out
> It reports launch errors; errors during execution appear at the next sync.

## 16. NumPy basics (Day 7: toy quantisers)

### np.array

`np.array(xs)`: turns a Python list into a NumPy array, so maths applies to every element at once. `np.array([0.9, 0.2]) * 2` → `[1.8, 0.4]`.

```python
>>> z = np.array([0.9, 0.2])
>>> print(z * 2)
[1.8 0.4]
>>> print(np.array([[1, 0], [0, 1]]).shape)
(2, 2)
```

> [!warning] Watch out
> A list times 2 repeats the list (`[1, 2] * 2` = `[1, 2, 1, 2]`); an array times 2 doubles each element.

*Used in:* [[Day 07 - Seminar and speech basics#Code, explained|Day 7]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### np.random.randn

`np.random.randn(d0, d1, ...)`: an array of the given shape filled with random numbers from a standard normal distribution (mean 0, spread 1). `np.random.randn(1024, 8)` is 1,024 random 8-d codes.

```python
>>> np.random.seed(0)
>>> print(np.random.randn(3).round(3))
[1.764 0.4   0.979]
>>> print(np.random.randn(1024, 8).shape)
(1024, 8)
```

> [!warning] Watch out
> Different numbers on every run unless you fix the seed.

*Used in:* [[Day 07 - Seminar and speech basics#Code, explained|Day 7]]

### np.argmin

`np.argmin(a)`: the **index** of the smallest element. `np.argmin([0.05, 1.45, 3.65, 2.25])` = 0. With `axis=` it works per row or column.

```python
>>> print(np.argmin([0.05, 1.45, 3.65, 2.25]))
0
>>> print(np.argmin([[3, 1], [0, 2]], axis=1))
[1 0]
```

> [!warning] Watch out
> It returns a position, not the value; use `a[np.argmin(a)]` (or `a.min()`) for the value.

*Used in:* [[Day 07 - Seminar and speech basics#Code, explained|Day 7]]

### np.round

`np.round(a)`: rounds every element to the nearest integer (still floats). `np.round([0.716, 0.197])` → `[1., 0.]`. Halves go to the nearest **even** number: 0.5 → 0, 1.5 → 2.

```python
>>> print(np.round([0.716, 0.197, -0.6]))
[ 1.  0. -1.]
>>> print(np.round([0.5, 1.5, 2.5]))
[0. 2. 2.]
```

> [!warning] Watch out
> Round-half-to-even: 2.5 → 2, not 3. Python's `round` and `torch.round` do the same.

*Used in:* [[Day 07 - Seminar and speech basics#Code, explained|Day 7]]

### np.tanh

`np.tanh(a)`: hyperbolic tangent of every element: squashes any number into (−1, 1). `tanh(0.9)` = 0.716, `tanh(0.2)` = 0.197, `tanh(5)` ≈ 0.9999.

```python
>>> print(np.tanh([0.9, 0.2]).round(3))
[0.716 0.197]
>>> print(np.tanh([-10, 10]))
[-1.  1.]
```

> [!warning] Watch out
> Saturates: big inputs all land near ±1, so their differences vanish.

*Used in:* [[Day 07 - Seminar and speech basics#Code, explained|Day 7]]

### x.copy

`.copy()`: a new, independent copy of an array (or list/dict). Changing the copy leaves the original alone: `residual = z.copy(); residual -= c` doesn't change `z`.

```python
>>> z = np.array([0.9, 0.2])
>>> r = z.copy(); r -= np.array([1, 0])
>>> print(z, r)
[0.9 0.2] [-0.1  0.2]
```

> [!warning] Watch out
> `r = z` (no copy) makes both names point at the same array, and `r -= ...` would change `z` too.

*Used in:* [[Day 07 - Seminar and speech basics#Code, explained|Day 7]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### log2

`log2(x)  (from math import log2)`: logarithm base 2: how many bits a choice among x options needs. `log2(1024)` = 10.0, `log2(6561)` = 12.68.

```python
>>> from math import log2
>>> print(log2(1024), log2(4096))
10.0 12.0
>>> print(round(log2(6561), 2))
12.68
```

> [!warning] Watch out
> Returns a float even for exact powers of two; compare with `==` only when the result is exact.

*Used in:* [[Day 07 - Seminar and speech basics#Code, explained|Day 7]]

## 17. Training loops, files and evaluation (Day 8: CosyVoice LoRA)

### torch.tensor

`torch.tensor(data, dtype=...)`: makes a new tensor from a Python list or number. `torch.tensor([5, 12, 30], dtype=torch.int32)` stores the true lengths of a batch.

```python
>>> print(torch.tensor([5, 12, 30], dtype=torch.int32))
tensor([ 5, 12, 30], dtype=torch.int32)
>>> print(torch.tensor(2.5))
tensor(2.5000)
```

> [!warning] Watch out
> `torch.tensor` always copies; `torch.as_tensor` may not.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]]

### x.dim

`.dim()`: the number of axes (dimensions). A `[1, 100]` tensor has `dim()` = 2; after `.squeeze(0)` it's 1.

```python
>>> x = torch.zeros(1, 100)
>>> print(x.dim(), x.squeeze(0).dim())
2 1
```

> [!warning] Watch out
> `.dim()` counts axes; `.numel()` counts elements (100 here).

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]]

### x.squeeze

`.squeeze(d)`: removes axis `d` if its size is 1 (the opposite of `unsqueeze`). `[1, 100]` → `.squeeze(0)` → `[100]`. Without `d` it removes every size-1 axis.

```python
>>> print(torch.zeros(1, 100).squeeze(0).shape)
torch.Size([100])
>>> print(torch.zeros(3, 100).squeeze(0).shape)
torch.Size([3, 100])
```

> [!warning] Watch out
> Squeezing an axis whose size isn't 1 silently does nothing (second example).

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### x.cpu

`.cpu()`: a copy of the tensor in CPU memory (no-op if it's already there). Needed before saving or converting to NumPy.

```python
>>> x = torch.ones(2)          # on CPU already
>>> print(x.cpu().device)
cpu
```

> [!warning] Watch out
> On a GPU tensor it copies, which forces the CPU to wait for the GPU.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### torch.concat

`torch.concat(tensors, dim=0)`: another name for `torch.cat`: joins tensors along an existing axis. `[sos (1)], [text (20)], [task (1)], [speech (100)]` along dim 0 → 122 rows.

```python
>>> a = torch.zeros(1, 4); b = torch.ones(3, 4)
>>> print(torch.concat([a, b], dim=0).shape)
torch.Size([4, 4])
```

> [!warning] Watch out
> All axes except `dim` must match.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]]

### x.ceil

`.ceil()`: rounds every element **up** to a whole number. `tensor(1.6).ceil()` → `tensor(2.)`. Upstream uses it to count 5-token text chunks: ⌈(7 + 1) / 5⌉ = 2.

```python
>>> print(((torch.tensor(7) + 1) / 5).ceil())
tensor(2.)
```

> [!warning] Watch out
> `/` on integer tensors gives floats (1.6); `//` would floor to 1.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]]

### x.int

`.int()`: converts to 32-bit integers (dropping any fraction). `tensor(2.).int()` → `tensor(2, dtype=torch.int32)`.

```python
>>> print(torch.tensor(2.7).int())
tensor(2, dtype=torch.int32)
```

> [!warning] Watch out
> It truncates (2.7 → 2); round first if you want 3.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]]

### x.copy_

`.copy_(src)`: copies `src`'s values **into** an existing tensor, in place (same shape required). Loading LoRA weights uses it so the model keeps its own tensors.

```python
>>> a = torch.zeros(2)
>>> a.copy_(torch.tensor([1., 2.]))
>>> print(a)
tensor([1., 2.])
```

> [!warning] Watch out
> Shapes must be broadcast-compatible; `a = b` would just rebind the name instead of filling `a`.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]]

### x.backward

`.backward()`: computes the gradient of a scalar (the loss) with respect to every tensor that requires grad, and **adds** it into each one's `.grad`. Adding (not overwriting) is what makes gradient accumulation work.

```python
>>> x = torch.tensor(3., requires_grad=True)
>>> (x ** 2).backward(); print(x.grad)
tensor(6.)
>>> (x ** 2).backward(); print(x.grad)      # added, not replaced
tensor(12.)
```

> [!warning] Watch out
> Gradients accumulate until you call `zero_grad()`; forgetting it silently doubles the next step.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]]

### x.zero_grad

`opt.zero_grad()`: clears the gradients of the optimiser's parameters (sets them to None), so the next `backward` starts from zero.

```python
>>> opt.zero_grad()
>>> print(p.grad)
None
```

> [!warning] Watch out
> Call it after `opt.step()` (or before the next backward), not between the backwards you want to accumulate.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]]

### torch.nn.utils.clip_grad_norm_

`torch.nn.utils.clip_grad_norm_(params, max_norm)`: treats all the gradients as one long vector; if its length is above `max_norm`, scales every gradient down so the length equals `max_norm`. A gradient (3, 4), length 5, with max_norm 1 → (0.6, 0.8). Returns the length before clipping.

```python
>>> p = torch.nn.Parameter(torch.zeros(2)); p.grad = torch.tensor([3., 4.])
>>> print(torch.nn.utils.clip_grad_norm_([p], 1.0))
tensor(5.)
>>> print(p.grad)
tensor([0.6000, 0.8000])
```

> [!warning] Watch out
> The returned value is the norm **before** clipping; log it to see how often clipping kicks in.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]]

### torch.autocast

`with torch.autocast(device_type='cuda', dtype=torch.bfloat16):`: inside this block, matrix multiplies and similar ops run in the lower precision (bf16 here); the parameters themselves stay in their own dtype (float32).

```python
>>> with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
...     y = torch.ones(2, 2) @ torch.ones(2, 2)
>>> print(y.dtype)
torch.bfloat16
```

> [!warning] Watch out
> Parameters stay float32, so a saved checkpoint is float32-sized (Day 8's lora.pt is 35 MB, not 17 MB).

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]]

### x.parameters

`.parameters()`: yields every learnable tensor of a module. `sum(p.numel() for p in layer.parameters())` counts them; `nn.Linear(896, 16, bias=False)` has 14,336.

```python
>>> layer = torch.nn.Linear(896, 16, bias=False)
>>> print(sum(p.numel() for p in layer.parameters()))
14336
```

> [!warning] Watch out
> It's a generator: it's used up after one loop.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]]

### x.named_parameters

`.named_parameters()`: like `.parameters()` but yields `(name, tensor)` pairs, e.g. `('layers.0.self_attn.q_proj.lora_A.default.weight', tensor)`. Used to save and reload tensors by name.

```python
>>> print([n for n, _ in torch.nn.Linear(4, 2).named_parameters()])
['weight', 'bias']
```

> [!warning] Watch out
> Names include the module path; save and load must use the same model structure.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]]

### LoraConfig

`LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=[...], bias='none')`: PEFT's description of the adapters: rank r, scale α (applied as α/r), dropout on the adapter's input, which `nn.Linear` names to wrap, and whether to train biases.

```python
>>> cfg = LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj', 'v_proj'])
>>> print(cfg.r, cfg.lora_alpha)
16 32
```

> [!warning] Watch out
> `target_modules` matches module **names**; a typo means no adapter is added and nothing trains.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]]

### inject_adapter_in_model

`inject_adapter_in_model(config, model)`: PEFT: wraps every matching `nn.Linear` of `model` **in place** with a LoRA layer (adds trainable A and B), without wrapping the model in a new class.

```python
>>> inject_adapter_in_model(cfg, qwen)
>>> print(sum(p.numel() for p in qwen.parameters() if p.requires_grad))
8798208   # for Day 8's config on Qwen2.5-0.5B, after freezing the rest
```

> [!warning] Watch out
> It doesn't freeze the base weights for you if they weren't frozen already; Day 8 freezes everything first.

### torch.save

`torch.save(obj, path)`: writes a tensor, or a dict of tensors, to a file (Python pickle format). `torch.load(path)` reads it back. Saved tensors keep their dtype: float32 stays 4 bytes each.

```python
>>> torch.save({'w': torch.ones(3)}, 'x.pt')
>>> print(torch.load('x.pt'))
{'w': tensor([1., 1., 1.])}
```

> [!warning] Watch out
> Files are pickles: only load files you trust.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]]

### x.save

`torchaudio.save(path, wav, sr)`: writes a waveform tensor of shape [channels, samples] to an audio file at sample rate `sr`.

```python
>>> torchaudio.save('a.wav', torch.zeros(1, 16000), 16000)   # 1 s of silence
```

> [!warning] Watch out
> The tensor must be [channels, samples]; a 1-d tensor needs `.unsqueeze(0)`.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]]

### Path

`Path('a') / 'b'`: pathlib's file path object; `/` joins parts: `Path('ckpt') / 'best' / 'lora.pt'` → `ckpt/best/lora.pt`. Has methods like `.exists()`, `.mkdir()`, `.unlink()`.

```python
>>> print(Path('ckpt') / 'best' / 'lora.pt')
ckpt/best/lora.pt
>>> print(Path('ckpt/best/lora.pt').name)
lora.pt
```

> [!warning] Watch out
> `Path` objects aren't strings; some libraries need `str(path)`.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]]

### x.mkdir

`path.mkdir(parents=True, exist_ok=True)`: creates the folder, including missing parent folders; `exist_ok=True` means no error if it already exists.

```python
>>> Path('ckpt/best').mkdir(parents=True, exist_ok=True)
```

> [!warning] Watch out
> Without `exist_ok=True` the second run crashes.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]]

### x.unlink

`path.unlink(missing_ok=True)`: deletes the file; `missing_ok=True` means no error if it's already gone.

```python
>>> Path('tmp.wav').unlink(missing_ok=True)
```

> [!warning] Watch out
> Deletes for good; there's no recycle bin.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]]

### dict

`dict(pairs)`: builds a dictionary. `dict(model.named_parameters())` maps each parameter name to its tensor, for lookup by name.

```python
>>> print(dict([('a', 1), ('b', 2)]))
{'a': 1, 'b': 2}
```

> [!warning] Watch out
> Later duplicate keys overwrite earlier ones.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### random.random

`random.random()`: a random float in [0, 1). `random.random() < 0.5` is a fair coin flip: upstream uses it to choose bistream or unistream per clip.

```python
>>> random.seed(0); print(round(random.random(), 4))
0.8444
```

> [!warning] Watch out
> Never returns exactly 1.0.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]]

### wer

`jiwer.wer(reference, hypothesis)`: word error rate: (substitutions + deletions + insertions) / reference words. `wer('the cat sat down', 'the bat sat')` = 0.5. Can exceed 1.

```python
>>> print(jiwer.wer('the cat sat down', 'the bat sat'))
0.5
>>> print(jiwer.wer('a b c d', 'a x c d p q r s t u'))
1.75
```

> [!warning] Watch out
> Normalise both texts first (spaces, punctuation such as `।`), or every mismatch counts as an error.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]]

### cer

`jiwer.cer(reference, hypothesis)`: the same as WER but on characters: `cer('abc', 'abd')` = 1/3.

```python
>>> print(round(jiwer.cer('abc', 'abd'), 3))
0.333
```

> [!warning] Watch out
> Spaces count as characters.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]]

### x.DataFrame

`pd.DataFrame(rows)`: a pandas table; from a list of dicts, each dict becomes a row and each key a column. `df['wer'].mean()` averages a column.

```python
>>> df = pd.DataFrame([{'label': 'zs', 'wer': 2.0}, {'label': 'zs', 'wer': 0.1}])
>>> print(df['wer'].mean())
1.05
```

> [!warning] Watch out
> A mean of per-row WERs is not a corpus WER.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]]

### x.unique

`series.unique()`: the distinct values of a pandas column, in order of first appearance. `df['label'].unique()` lists each system once.

```python
>>> print(pd.Series(['zs', 'lora', 'zs']).unique())
['zs' 'lora']
```

> [!warning] Watch out
> Returns a NumPy array, not a list.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]]

## 18. Indexing, masks and input gradients (Day 9: retrieval heads, pruning, PGD)

### x.clamp

`.clamp(min=..., max=...)`: clips each element into a range. `tensor([-4, 2, 12]).clamp(0, 9)` → `[0, 2, 9]`. `copy_hits` uses `clamp(max=len - 1)` so an index never runs past the end.

```python
>>> print(torch.tensor([-4, 2, 12]).clamp(0, 9))
tensor([0, 2, 9])
>>> print(torch.tensor([3, 12]).clamp(max=9))
tensor([3, 9])
```

> [!warning] Watch out
> Clamping hides out-of-range values instead of failing: only safe when those positions are ignored later (as `in_span` does).

*Used in:* [[Day 04 - Speculative decoding#Code, explained|Day 4]], [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### x.abs

`.abs()`: absolute value of every element: `tensor([-0.8, 0.1]).abs()` → `[0.8, 0.1]`. Magnitude pruning ranks weights by it.

```python
>>> print(torch.tensor([-0.8, 0.1]).abs())
tensor([0.8000, 0.1000])
```

> [!warning] Watch out
> Rank weights by `.abs()`, not by the raw value, or every negative weight looks 'small'.

*Used in:* [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]]

### x.flatten

`.flatten()`: the same numbers as one 1-d tensor. A conv1 weight `[6, 1, 5, 5]` → `.flatten()` → `[150]`.

```python
>>> print(torch.zeros(6, 1, 5, 5).flatten().shape)
torch.Size([150])
```

> [!warning] Watch out
> `flatten(1)` keeps axis 0 (e.g. the batch) and flattens the rest.

*Used in:* [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]]

### x.kthvalue

`.kthvalue(k)`: the k-th **smallest** value (and its index), counting from 1. `tensor([0.9, 0.1, 0.5, 0.3]).kthvalue(2).values` → 0.3. Used to find a pruning threshold.

```python
>>> print(torch.tensor([0.9, 0.1, 0.5, 0.3]).kthvalue(2).values)
tensor(0.3000)
```

> [!warning] Watch out
> k counts from 1 and from the **smallest**; for the k-th largest use `topk`.

*Used in:* [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]]

### x.endswith

`s.endswith(suffix)`: True if the string ends with `suffix`. `'conv1.weight'.endswith('weight')` → True; `'conv1.bias'` → False.

```python
>>> print('conv1.weight'.endswith('weight'), 'conv1.bias'.endswith('weight'))
True False
```

> [!warning] Watch out
> Case-sensitive: 'Weight' doesn't match.

*Used in:* [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]]

### x.clone

`.clone()`: a copy of a tensor with its own memory. `x0.clone().detach()` starts the attack from a copy, so `x0` stays untouched.

```python
>>> a = torch.zeros(2); b = a.clone(); b += 1
>>> print(a, b)
tensor([0., 0.]) tensor([1., 1.])
```

> [!warning] Watch out
> `b = a` shares memory; `b += 1` would change `a` too.

*Used in:* [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]]

### x.requires_grad_

`.requires_grad_(True)`: turns on gradient tracking for this tensor, in place. PGD sets it on the **input** so `autograd.grad` can return d loss / d input.

```python
>>> x = torch.tensor([0.3, 0.2]).requires_grad_(True)
>>> print(x.requires_grad)
True
```

> [!warning] Watch out
> Only works on leaf tensors; call `.detach()` first inside a loop.

*Used in:* [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]]

### F.cross_entropy

`F.cross_entropy(logits, target)`: softmax + negative log-probability of the correct class, averaged over the batch. Logits `[[2.0, 0.5, -1.0]]`, target `[0]` → 0.2413.

```python
>>> print(F.cross_entropy(torch.tensor([[2.0, 0.5, -1.0]]), torch.tensor([0])))
tensor(0.2413)
```

> [!warning] Watch out
> It takes raw **logits**, not probabilities; applying softmax first double-softmaxes.

*Used in:* [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### torch.autograd.grad

`torch.autograd.grad(output, inputs)`: returns the gradient of `output` with respect to `inputs` (a tuple), without touching any `.grad` fields. For s = 2x₁ − x₂: d(−s)/dx = (−2, 1).

```python
>>> x = torch.tensor([0.3, 0.2], requires_grad=True)
>>> s = 2 * x[0] - x[1]
>>> print(torch.autograd.grad(-s, x))
(tensor([-2.,  1.]),)
```

> [!warning] Watch out
> Returns a tuple: unpack with `grad, = ...`.

*Used in:* [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]]

### x.sign

`.sign()`: −1, 0 or +1 for each element by its sign. `tensor([-2., 1., 0.]).sign()` → `[-1., 1., 0.]`. PGD steps by α · grad.sign().

```python
>>> print(torch.tensor([-2., 1., 0.]).sign())
tensor([-1.,  1.,  0.])
```

> [!warning] Watch out
> A zero gradient gives step 0 for that pixel.

*Used in:* [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]]

### torch.min

`torch.min(a, b)`: element-wise smaller of two tensors (with one tensor and no `b`, the smallest element). Paired with `torch.max` it clips into a box.

```python
>>> print(torch.min(torch.tensor([0.5, 0.36]), torch.tensor([0.45, 0.35])))
tensor([0.4500, 0.3500])
```

> [!warning] Watch out
> With one argument plus `dim`, it returns (values, indices) instead.

*Used in:* [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]]

### torch.max

`torch.max(a, b)`: element-wise larger of two tensors (with one tensor and no `b`, the largest element). `torch.max(x, x0 - eps)` raises anything below the box's floor.

```python
>>> print(torch.max(torch.tensor([0.14, 0.36]), torch.tensor([0.15, 0.05])))
tensor([0.1500, 0.3600])
```

> [!warning] Watch out
> Same as `torch.min`: two tensors → element-wise; one tensor → reduction.

*Used in:* [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]]

## 19. Day 10 code: Photo-Seek search: regex, dates, SQLite, CLIP, FAISS

### x.search

`re.search(pattern, s) / index.search(q, k)`: two different things share this name. **`re.search`** finds the first place a regex matches anywhere in `s` and returns a match object (or None): `re.search(r'\b(last|this)\s+summer\b', 'photos last summer')` matches `'last summer'`. **A FAISS index's `.search(q, k)`** returns the k best scores and ids for each query row: two arrays of shape `[n_queries, k]`.

```python
>>> import re
>>> m = re.search(r'\b(last|this|past)\s+summer\b', 'photos of mom last summer')
>>> m.group(0), m.group(1)
('last summer', 'last')
>>> re.search(r'\bsummer\b', 'summertime') is None
True
```

> [!warning] Watch out
> `re.search` looks anywhere in the string; `re.match` only at the start.

*Used in:* [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### x.group

`match.group(n)`: the text a regex match captured. `group(0)` (or `group()`) is the whole match; `group(1)` is the first `( … )`. For `last summer` matched by `(last|this|past)\s+summer`, `group(1)` = `'last'`.

```python
>>> import re
>>> m = re.search(r'(\d+)-(\d+)', 'pages 12-15')
>>> m.group(0), m.group(1), m.group(2)
('12-15', '12', '15')
```

> [!warning] Watch out
> Groups are numbered by their opening parenthesis, left to right; group 0 is the whole match.

*Used in:* [[Day 10 - Photo-Seek I#Code, explained|Day 10]]

### x.span

`match.span()`: the (start, end) character positions of the match, end excluded. In `'photos last summer'`, `'last summer'` spans `(7, 18)`; `_cut` uses it to remove the phrase.

```python
>>> import re
>>> s = 'photos last summer'
>>> m = re.search(r'last summer', s)
>>> m.span()
(7, 18)
>>> s[:7] + s[18:]
'photos '
```

> [!warning] Watch out
> The end index is exclusive, like Python slicing.

*Used in:* [[Day 10 - Photo-Seek I#Code, explained|Day 10]]

### timedelta

`datetime.timedelta(days=n)`: a length of time you can add to a date. `date(2026, 8, 31) + timedelta(days=1)` = `date(2026, 9, 1)`: month and year roll over correctly.

```python
>>> from datetime import date, timedelta
>>> date(2026, 8, 31) + timedelta(days=1)
datetime.date(2026, 9, 1)
>>> date(2026, 12, 31) + timedelta(days=1)
datetime.date(2027, 1, 1)
```

> [!warning] Watch out
> Adding 1 to the day number by hand would give 2026-08-32.

*Used in:* [[Day 10 - Photo-Seek I#Code, explained|Day 10]]

### x.isoformat

`d.isoformat()`: a date as standard text `'YYYY-MM-DD'`. `date(2026, 9, 1).isoformat()` → `'2026-09-01'`. Such strings sort in date order, so SQLite can compare them with `<`.

```python
>>> from datetime import date
>>> date(2026, 9, 1).isoformat()
'2026-09-01'
>>> '2026-08-31 18:20:00' < '2026-09-01'
True
>>> '2026-08-31 18:20:00' <= '2026-08-31'
False
```

> [!warning] Watch out
> Text comparison works only because ISO dates put the year first and zero-pad months and days.

*Used in:* [[Day 10 - Photo-Seek I#Code, explained|Day 10]]

### replace

`dataclasses.replace(obj, field=value)`: a **copy** of a dataclass object with some fields changed; the original is untouched. `replace(q, text='beach', place=None)` moves the place into the text without mutating `q`.

```python
>>> from dataclasses import dataclass, replace
>>> @dataclass
... class Q:
...     text: str
...     place: str | None
>>> q = Q('', 'beach')
>>> replace(q, text='beach', place=None)
Q(text='beach', place=None)
>>> q
Q(text='', place='beach')
```

> [!warning] Watch out
> It works on dataclasses only; it's unrelated to `str.replace`.

*Used in:* [[Day 10 - Photo-Seek I#Code, explained|Day 10]]

### x.fromkeys

`dict.fromkeys(xs)`: a dict whose keys are `xs` in first-seen order (values None). `list(dict.fromkeys([7, 9, 7]))` → `[7, 9]`: an order-keeping way to drop duplicates (two names resolving to the same person give one SQL clause).

```python
>>> list(dict.fromkeys([7, 9, 7, 3]))
[7, 9, 3]
>>> list(set([7, 9, 7, 3])) == [7, 9, 3]  # a set may reorder
False
```

> [!warning] Watch out
> All keys share one value object: `dict.fromkeys(xs, [])` gives every key the *same* list.

*Used in:* [[Day 10 - Photo-Seek I#Code, explained|Day 10]]

### x.warning

`log.warning(msg, *args)`: writes a warning to the program's log (Python's `logging`); `%s` in `msg` is filled with `args` only if the message is actually printed. `.info` is the same at a lower level.

```python
>>> import logging
>>> log = logging.getLogger('demo')
>>> log.warning('LLM router failed (%s)', 'timeout')  # prints: LLM router failed (timeout)
```

> [!warning] Watch out
> Pass values as arguments, not with an f-string, so formatting is skipped when the level is off.

*Used in:* [[Day 10 - Photo-Seek I#Code, explained|Day 10]]

### x.info

`log.info(msg, *args)`: writes an informational message to the log (Python's `logging`), shown only if the log level is INFO or lower.

```python
>>> import logging
>>> logging.getLogger('demo').isEnabledFor(logging.INFO)  # default level is WARNING
False
```

> [!warning] Watch out
> INFO messages are hidden unless logging is configured at INFO or DEBUG.

*Used in:* [[Day 10 - Photo-Seek I#Code, explained|Day 10]]

### x.execute

`conn.execute(sql, params) / session.execute(query)`: runs a database query. **sqlite3** (Photo-Seek): `conn.execute('SELECT id FROM photos WHERE taken_at >= ?', ['2026-06-01'])`: each `?` is filled from `params` by the driver, never pasted into the text, so query text can't inject SQL; iterate the result for rows. **SQLAlchemy** (Day 12): `session.execute(select(...))` sends a built query to the database.

```python
>>> import sqlite3
>>> conn = sqlite3.connect(':memory:')
>>> _ = conn.execute('CREATE TABLE photos (id INTEGER, taken_at TEXT)')
>>> _ = conn.executemany('INSERT INTO photos VALUES (?, ?)', [(12, '2026-08-31 18:20:00'), (13, '2026-09-02 09:00:00')])
>>> [r[0] for r in conn.execute('SELECT id FROM photos WHERE taken_at < ?', ['2026-09-01'])]
[12]
```

> [!warning] Watch out
> Never build SQL with f-strings from user text; use `?` parameters.

*Used in:* [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### x.fetchall

`cursor.fetchall()`: all remaining result rows as a list. `conn.execute('SELECT id, identity_id FROM faces').fetchall()` → `[(1, 7), (2, None), …]`; `dict(...)` of that makes `{face_id: identity_id}`.

```python
>>> import sqlite3
>>> conn = sqlite3.connect(':memory:')
>>> _ = conn.execute('CREATE TABLE faces (id INTEGER, identity_id INTEGER)')
>>> _ = conn.executemany('INSERT INTO faces VALUES (?, ?)', [(1, 7), (2, None)])
>>> conn.execute('SELECT id, identity_id FROM faces').fetchall()
[(1, 7), (2, None)]
>>> dict(conn.execute('SELECT id, identity_id FROM faces').fetchall())
{1: 7, 2: None}
```

> [!warning] Watch out
> It loads every row into memory at once; iterate the cursor for big results.

*Used in:* [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### torch.nn.functional.normalize

`torch.nn.functional.normalize(x, dim=-1)`: divides each vector by its length (L2 norm) along `dim`, so every row has length 1. `normalize(tensor([[3., 4.]]), dim=-1)` → `[[0.6, 0.8]]`. After this, a dot product is the cosine similarity.

```python
>>> import torch
>>> x = torch.tensor([[3., 4.], [6., 8.]])
>>> torch.nn.functional.normalize(x, dim=-1)
tensor([[0.6000, 0.8000],
        [0.6000, 0.8000]])
```

> [!warning] Watch out
> `dim=-1` normalises each row; `dim=0` would normalise columns instead.

*Used in:* [[Day 10 - Photo-Seek I#Code, explained|Day 10]]

### F.normalize

`F.normalize(x, dim=-1)`: the same function as `torch.nn.functional.normalize` (`F` is the usual short name): each row divided by its length. `F.normalize(tensor([[3., 4.]]))` → `[[0.6, 0.8]]`.

```python
>>> import torch
>>> import torch.nn.functional as F
>>> F.normalize(torch.tensor([[1., 1.]]))
tensor([[0.7071, 0.7071]])
```

> [!warning] Watch out
> The default `dim` is 1, which is the last axis only for 2-D input; pass `dim=-1` to be explicit.

*Used in:* [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### x.encode_text

`model.encode_text(tokens)`: open_clip: runs CLIP's **text** encoder. Token ids `[n, 77]` → one embedding per text, `[n, 512]` for ViT-B-32 (not yet normalised). `encode_image` is the image twin.

```cpp
tokens = tokenizer(['beach', 'a dog'])   # [2, 77] token ids
feats = model.encode_text(tokens)       # [2, 512], not normalised
feats = F.normalize(feats, dim=-1)      # unit length
```

> [!warning] Watch out
> Its outputs aren't unit length; normalise before using dot products as cosines.

*Used in:* [[Day 10 - Photo-Seek I#Code, explained|Day 10]]

### x.numpy

`.numpy()`: the tensor as a numpy array, sharing the same memory (CPU tensors only). `torch.tensor([1., 2.]).numpy()` → `array([1., 2.], dtype=float32)`.

```python
>>> import torch
>>> t = torch.tensor([1., 2.])
>>> a = t.numpy()
>>> a
array([1., 2.], dtype=float32)
>>> t[0] = 9.
>>> a
array([9., 2.], dtype=float32)
```

> [!warning] Watch out
> The array shares memory with the tensor: changing one changes the other.

*Used in:* [[Day 10 - Photo-Seek I#Code, explained|Day 10]]

### np.fromiter

`np.fromiter(iterable, dtype)`: builds a 1-D numpy array straight from any iterable, such as a set. `np.fromiter({40, 12, 77}, dtype='int64')` → an int64 array of those 3 ids (set order is arbitrary).

```python
>>> import numpy as np
>>> a = np.fromiter({40, 12, 77}, dtype='int64')
>>> sorted(a.tolist())
[12, 40, 77]
>>> a.dtype
dtype('int64')
```

> [!warning] Watch out
> A set has no order, so don't rely on the array's order.

*Used in:* [[Day 10 - Photo-Seek I#Code, explained|Day 10]]

### faiss.IDSelectorBatch

`faiss.IDSelectorBatch(ids)`: a FAISS filter that says "only these ids may be returned". Membership is checked with a hash set; a flat index still visits every vector but scores only the allowed ones.

```cpp
sel = faiss.IDSelectorBatch(np.array([12, 40, 77], dtype='int64'))
params = faiss.SearchParameters(sel=sel)
scores, ids = index.search(q, 3, params=params)   # only 12, 40, 77 can appear
```

> [!warning] Watch out
> With a flat index, filtering doesn't speed the search up: every vector is still visited.

*Used in:* [[Day 10 - Photo-Seek I#Code, explained|Day 10]]

### faiss.SearchParameters

`faiss.SearchParameters(sel=selector)`: per-call search options for FAISS; `sel=` attaches an id filter such as `IDSelectorBatch`. Passed as `index.search(q, k, params=...)`.

```cpp
params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(ids))
index.search(q, k, params=params)
```

> [!warning] Watch out
> Index-specific options (like IVF's nprobe) use subclasses such as `SearchParametersIVF`.

*Used in:* [[Day 10 - Photo-Seek I#Code, explained|Day 10]]

### x.add_with_ids

`index.add_with_ids(vectors, ids)`: FAISS (on an `IndexIDMap`): stores each row of `vectors` (`[n, 512]` float32) under your own int64 id instead of 0, 1, 2, …. Searches then return those ids.

```cpp
index = faiss.IndexIDMap2(faiss.IndexFlatIP(512))
index.add_with_ids(vecs, np.array([12, 40, 77], dtype='int64'))   # vecs: [3, 512] float32
index.ntotal   # 3
```

> [!warning] Watch out
> Adding an id that already exists stores a second copy; Photo-Seek calls `remove_ids` first.

*Used in:* [[Day 10 - Photo-Seek I#Code, explained|Day 10]]

## 20. Day 11 code: Faces and steering: numpy indexing, hooks, log-probabilities

### print

`print(x)`: writes text to the output. `print(f'{0.5:.2f}x')` prints `0.50x`; `:.2f` = 2 decimals.

```python
>>> scale, ppl, ratio = 0.75, 8.4855, 1.5
>>> print(f'  scale {scale:.2f} → ppl {ppl:7.1f} ({ratio:.2f}x baseline)')
  scale 0.75 → ppl     8.5 (1.50x baseline)
```

> [!warning] Watch out
> `:7.1f` pads to 7 characters; that's why the numbers line up.

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 07 - Seminar and speech basics#Code, explained|Day 7]], [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### math.isfinite

`math.isfinite(x)`: True for a normal number, False for `inf`, `-inf` and `nan`. The guard's `not math.isfinite(ratio)` is True for NaN, which is why a NaN ratio is accepted.

```python
>>> import math
>>> math.isfinite(1.5), math.isfinite(float('nan')), math.isfinite(float('inf'))
(True, False, False)
>>> float('nan') <= 1.75
False
```

> [!warning] Watch out
> Every comparison with NaN is False, so test for it explicitly.

*Used in:* [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### np.asarray

`np.asarray(xs)`: turns a list into a numpy array (no copy if it already is one). `np.asarray([0, 0, 1, -1])` → `array([ 0,  0,  1, -1])`.

```python
>>> import numpy as np
>>> np.asarray([0, 0, 1, -1])
array([ 0,  0,  1, -1])
```

> [!warning] Watch out
> Unlike `np.array`, it doesn't copy an existing array: edits show through.

*Used in:* [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### np.arange

`np.arange(n)`: the integers 0 … n−1 as an array. `np.arange(4)` → `array([0, 1, 2, 3])`. With a second index array it picks one cell per row: `dist[np.arange(4), proposed]`.

```python
>>> import numpy as np
>>> dist = np.array([[0.70, 0.95], [0.40, 0.42], [0.33, 0.30]])
>>> dist[np.arange(3), np.array([0, 0, 1])]
array([0.7, 0.4, 0.3])
```

> [!warning] Watch out
> `dist[:, [0, 0, 1]]` is different: it picks whole columns, giving a 3 × 3 result.

*Used in:* [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### np.where

`np.where(cond, a, b)`: element by element: `a` where `cond` is True, else `b`. `np.where(own >= 0, own, nearest)` takes the face's own identity column, or the nearest one when it has none (−1).

```python
>>> import numpy as np
>>> own, nearest = np.array([0, 0, 1, -1]), np.array([0, 0, 1, 0])
>>> np.where(own >= 0, own, nearest)
array([0, 0, 1, 0])
```

> [!warning] Watch out
> With only a condition, `np.where(cond)` returns the indices where it's True.

*Used in:* [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### np.full

`np.full(n, value)`: an array of n copies of `value`. `np.full(3, 2.0)` → `array([2., 2., 2.])`.

```python
>>> import numpy as np
>>> np.full(3, 2.0)
array([2., 2., 2.])
```

> [!warning] Watch out
> `np.full(3, 2)` gives integers; write 2.0 for floats.

*Used in:* [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### np.argsort

`np.argsort(x, kind='stable')`: the **positions** that would sort `x` ascending, not the sorted values. `np.argsort([0.25, 0.02, 0.03])` → `[1, 2, 0]`. `argsort(-x)` sorts largest first; `kind='stable'` keeps ties in their original order.

```python
>>> import numpy as np
>>> d1 = np.array([0.70, 0.40, 0.30, 0.65])
>>> np.argsort(-d1, kind='stable')
array([0, 3, 1, 2])
>>> np.argsort(np.array([0.25, 0.02, 0.03, 0.20]), kind='stable')
array([1, 2, 3, 0])
```

> [!warning] Watch out
> It returns positions; index the array with them to get sorted values.

*Used in:* [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### np.random.default_rng

`np.random.default_rng(seed)`: a random-number generator with its own seed. The same seed always gives the same sequence; `rng.permutation(4)` is then a reproducible shuffle of 0–3.

```python
>>> import numpy as np
>>> a = np.random.default_rng(3).permutation(4)
>>> b = np.random.default_rng(3).permutation(4)
>>> (a == b).all(), sorted(a.tolist())
(np.True_, [0, 1, 2, 3])
```

> [!warning] Watch out
> A different seed per round (`seed + round`) gives a different but repeatable shuffle each round.

*Used in:* [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### x.permutation

`rng.permutation(n)`: a random ordering of 0 … n−1. `default_rng(0).permutation(4)` is always the same 4 numbers in the same order.

```python
>>> import numpy as np
>>> sorted(np.random.default_rng(0).permutation(5).tolist())
[0, 1, 2, 3, 4]
```

> [!warning] Watch out
> `rng.permutation(n)` returns a new array; `rng.shuffle(a)` shuffles in place.

*Used in:* [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### np.stack

`np.stack(arrays)`: joins equal-shape arrays along a **new** first axis: three `(512,)` means → one `(3, 512)` array (like `torch.stack`).

```python
>>> import numpy as np
>>> np.stack([np.array([1., 2.]), np.array([3., 4.]), np.array([5., 6.])]).shape
(3, 2)
```

> [!warning] Watch out
> `np.concatenate` would give shape (6,) instead.

*Used in:* [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### np.isfinite

`np.isfinite(x)`: True where `x` is a normal number, False for `inf` and `nan`. `np.isfinite(np.inf)` → False: that's how refine detects a rejected identity.

```python
>>> import numpy as np
>>> np.isfinite(np.array([0.38, np.inf, np.nan]))
array([ True, False, False])
```

> [!warning] Watch out
> NaN is not finite either, so a NaN distance would also read as 'rejected'.

*Used in:* [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### x.argmin

`.argmin(axis)`: the **index** of the smallest value (per row with `axis=1`). `np.array([[0.40, 0.38, 0.90]]).argmin(axis=1)` → `[1]`. Twin of `.argmax`.

```python
>>> import numpy as np
>>> np.array([[0.40, 0.38, 0.90], [np.inf, 0.38, 0.90]]).argmin(axis=1)
array([1, 1])
```

> [!warning] Watch out
> On ties it returns the first index.

*Used in:* [[Day 07 - Seminar and speech basics#Code, explained|Day 7]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### x.min

`.min(axis)`: the smallest value (per row with `axis=1`). `np.array([[0.70, inf, 1.0]]).min(axis=1)` → `[0.70]`.

```python
>>> import numpy as np
>>> others = np.array([[np.inf, 0.95, 1.00], [np.inf, 0.42, 0.90]])
>>> others.min(axis=1)
array([0.95, 0.42])
```

> [!warning] Watch out
> `axis=1` reduces across columns, giving one value per row.

*Used in:* [[Day 09 - CS728 and LeNet-5#Code, explained|Day 9]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### x.register_forward_hook

`module.register_forward_hook(fn)`: asks PyTorch to call `fn(module, input, output)` right after this module's forward pass on every call. If `fn` returns something, that **replaces** the output. Returns a handle; `handle.remove()` unregisters it.

```python
>>> import torch
>>> lin = torch.nn.Linear(2, 2, bias=False)
>>> _ = torch.nn.init.eye_(lin.weight)
>>> h = lin.register_forward_hook(lambda m, i, o: o + 10)
>>> lin(torch.tensor([1., 2.])).tolist()
[11.0, 12.0]
>>> h.remove(); lin(torch.tensor([1., 2.])).tolist()
[1.0, 2.0]
```

> [!warning] Watch out
> Always remove the hook (in a `finally`), or it keeps changing every later call.

*Used in:* [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### torch.manual_seed

`torch.manual_seed(s)`: fixes PyTorch's random generator. Calling it with the same `s` right before two `generate` calls makes them draw the same random numbers when sampling.

```python
>>> import torch
>>> _ = torch.manual_seed(0); a = torch.rand(2)
>>> _ = torch.manual_seed(0); b = torch.rand(2)
>>> torch.equal(a, b)
True
```

> [!warning] Watch out
> Same seed, same draws only while both runs consume random numbers identically; once their tokens differ, the streams diverge.

*Used in:* [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### torch.no_grad

`torch.no_grad()`: a `with` block in which PyTorch records no gradients: less memory, for inference. (`torch.inference_mode` is a stricter, faster variant.)

```python
>>> import torch
>>> w = torch.ones(2, requires_grad=True)
>>> with torch.no_grad():
...     y = (w * 2).sum()
>>> y.requires_grad
False
```

> [!warning] Watch out
> It doesn't change dropout or other train/eval behaviour; that's `model.eval()`.

*Used in:* [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### torch.log_softmax

`torch.log_softmax(x, dim=-1)`: the natural log of softmax, computed in one stable step. Each row becomes log-probabilities: `log_softmax([0., 0.])` → `[-0.693, -0.693]` (ln 0.5).

```python
>>> import torch
>>> torch.log_softmax(torch.tensor([0., 0.]), dim=-1)
tensor([-0.6931, -0.6931])
>>> torch.log_softmax(torch.tensor([2., 0.]), dim=-1).exp()
tensor([0.8808, 0.1192])
```

> [!warning] Watch out
> `torch.log(torch.softmax(x))` can give −inf for tiny probabilities; log_softmax doesn't.

*Used in:* [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### x.gather

`.gather(dim, index)`: picks, along `dim`, the element at the given index for every row. `logprobs.gather(-1, targets.unsqueeze(-1))` takes, at each position, the log-probability of the token that actually came next: `[1, T, V]` → `[1, T, 1]`.

```python
>>> import torch
>>> lp = torch.log(torch.tensor([[[0.5, 0.25, 0.25], [0.1, 0.8, 0.1]]]))  # [1, 2 positions, V=3]
>>> targets = torch.tensor([[0, 1]])                                          # actual next tokens
>>> lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).exp()
tensor([[0.5000, 0.8000]])
```

> [!warning] Watch out
> `index` must have the same number of dimensions as the input, hence the `.unsqueeze(-1)`.

*Used in:* [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### torch.acos

`torch.acos(x)`: arc-cosine: the angle (radians) whose cosine is x. `torch.acos(tensor(0.5))` = 1.047 (60°). Its slope is infinite at ±1, so inputs are clamped slightly inside.

```python
>>> import torch, math
>>> round(math.degrees(torch.acos(torch.tensor(0.5)).item()), 1)
60.0
```

> [!warning] Watch out
> Clamp inputs to (−1, 1): at exactly ±1 the gradient is infinite.

*Used in:* [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### torch.cos

`torch.cos(x)`: cosine of every element (radians). `torch.cos(tensor(1.047 + 0.5))` = 0.024: ArcFace's cos(θ + m) for θ = 60°, m = 0.5.

```python
>>> import torch
>>> round(torch.cos(torch.tensor(1.0472 + 0.5)).item(), 3)
0.024
```

> [!warning] Watch out
> Arguments are radians: 60° = 1.0472.

*Used in:* [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### torch.where

`torch.where(cond, a, b)`: element by element: `a` where `cond` is True, else `b` (like `np.where` and `tl.where`). ArcFace uses it to swap in cos(θ + m) only at the true class.

```python
>>> import torch
>>> onehot = torch.tensor([True, False, False])
>>> torch.where(onehot, torch.tensor([0.024, 0.0, 0.0]), torch.tensor([0.5, 0.174, 0.087]))
tensor([0.0240, 0.1740, 0.0870])
```

> [!warning] Watch out
> All three arguments broadcast together; shapes must be compatible.

*Used in:* [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

### x.bool

`.bool()`: converts to True/False: nonzero → True. A one-hot `[0., 1., 0.]` becomes `[False, True, False]`, usable as a mask.

```python
>>> import torch
>>> torch.tensor([0., 1., 0.]).bool()
tensor([False,  True, False])
```

> [!warning] Watch out
> Any nonzero value, even 0.001, becomes True.

*Used in:* [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]]

## 21. Day 12 code: BountyBrain: strings, regex, SQLAlchemy, BM25, rerankers

### str

`str(x)`: converts anything to its text form. `str(89)` → `'89'`. `find_weakness` uses `str(term)` so a number or a string both work.

```python
>>> str(89)
'89'
```

> [!warning] Watch out
> `str(None)` is `'None'`, a real string, not an error.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### x.strip

`.strip()`: removes spaces (and newlines) from both ends of a string. `'  CWE-89 '.strip()` → `'CWE-89'`.

```python
>>> '  CWE-89 '.strip()
'CWE-89'
```

> [!warning] Watch out
> it only trims the ends; inner spaces stay.

*Used in:* [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### x.upper

`.upper()`: the string in capitals. `'cwe-89'.upper()` → `'CWE-89'`, so a user typing `cwe-89` still matches `CWE-`.

```python
>>> 'cwe-89'.upper()
'CWE-89'
```

> [!warning] Watch out
> returns a new string; the original is unchanged (strings are immutable).

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

### x.lower

`.lower() / func.lower(col)`: on a string: the string in small letters, `'SQL Injection'.lower()` → `'sql injection'`. In SQLAlchemy, `func.lower(Vulnerability.name)` is the SQL function `LOWER(name)`, so the comparison happens case-insensitively inside the database.

```python
>>> 'SQL Injection'.lower()
'sql injection'
```

> [!warning] Watch out
> `func.lower(col)` builds SQL; calling `.lower()` on a column object would fail.

*Used in:* [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### x.startswith

`.startswith(prefix)`: True if the string begins with `prefix`. `'CWE-89'.startswith('CWE-')` → True.

```python
>>> 'CWE-89'.startswith('CWE-')
True
```

> [!warning] Watch out
> case-sensitive: `'cwe-89'.startswith('CWE-')` is False, hence `.upper()` first.

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

### x.isdigit

`.isdigit()`: True if every character is a digit (and there's at least one). `'89'.isdigit()` → True; `'8a'.isdigit()` → False.

```python
>>> '89'.isdigit()
True
>>> '8a'.isdigit()
False
```

> [!warning] Watch out
> `''.isdigit()` is False, and `'-1'.isdigit()` is False too.

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

### x.join

`sep.join(xs) / select(...).join(T, cond)`: on a string separator: glues a list of strings with `sep` between them, `', '.join(['a', 'b'])` → `'a, b'`. In a SQLAlchemy query (`planner.py`), `.join(Table, condition)` is an SQL JOIN: pair each row with the matching rows of another table.

```python
>>> ', '.join(['a', 'b'])
'a, b'
```

> [!warning] Watch out
> every item must already be a string: `', '.join([1, 2])` raises TypeError.

*Used in:* [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### x.split

`re.split(pattern, s) / s.split()`: cuts a string into a list at every match. `re.split(r'\s*,\s*', '2, 4–5')` → `['2', '4–5']` (commas with any spaces around them); plain `'a b'.split()` splits on whitespace.

```python
>>> import re
>>> re.split(r'\s*,\s*', '2, 4–5')
['2', '4–5']
>>> 'a b'.split()
['a', 'b']
```

> [!warning] Watch out
> `s.split(' ')` and `s.split()` differ: the first keeps empty strings for double spaces.

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

### x.compile

`re.compile(pattern)`: turns a regex pattern into a reusable object (compiled once, used many times). `_TOKEN = re.compile(r'[a-z0-9_.-]+')` matches runs of letters, digits, `_`, `.` and `-`.

```python
>>> import re
>>> re.compile(r'[a-z0-9_.-]+').findall('cwe-918: ssrf')
['cwe-918', 'ssrf']
```

> [!warning] Watch out
> write patterns as raw strings `r'...'` so backslashes reach the regex unchanged.

*Used in:* [[Day 04 - Speculative decoding#Code, explained|Day 4]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### x.findall

`pattern.findall(s)`: all non-overlapping matches, as a list of strings. `_TOKEN.findall('cwe-918: ssrf')` → `['cwe-918', 'ssrf']`. If the pattern has one group `( )`, it returns just the group's text: that's how `_CITE.findall` returns `'2, 4–5'` from `[2, 4–5]`.

```python
>>> import re
>>> re.findall(r'\[(\d+)\]', 'see [1] and [3]')
['1', '3']
```

> [!warning] Watch out
> with a group in the pattern it returns the group, not the whole match.

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

### json.dumps

`json.dumps(obj)`: converts Python dicts/lists/numbers into a JSON **string**. `json.dumps({'cwe': 639})` → `'{"cwe": 639}'`. Used to put graph facts into the prompt as text.

```python
>>> import json
>>> json.dumps({'cwe': 639, 'name': 'IDOR'})
'{"cwe": 639, "name": "IDOR"}'
```

> [!warning] Watch out
> the result is a string; `json.loads` parses it back.

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

### math.log2

`math.log2(x)`: logarithm base 2: the power you raise 2 to, to get x. `math.log2(4)` = 2.0; `1 / math.log2(3)` = 0.631 is nDCG's discount at rank 2.

```python
>>> import math
>>> math.log2(4)
2.0
>>> 1 / math.log2(3)
0.6309297535714575
```

> [!warning] Watch out
> `math.log2(0)` raises ValueError; nDCG uses `log2(i + 2)`, which starts at 1.

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

### math.log1p

`math.log1p(x)`: ln(1 + x), computed accurately even for tiny x. `math.log1p(0)` = 0; `math.log1p(25)` = 3.258. `_log_scale` divides two of these to map a count onto [0, 1].

```python
>>> import math
>>> math.log1p(0)
0.0
>>> math.log1p(25)
3.258096538021482
```

> [!warning] Watch out
> it's ln(1 + x), not ln(x): `math.log1p(1)` is 0.693.

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

### Counter

`collections.Counter(xs)`: a dict that counts how often each item appears. `Counter(['supported', 'partial', 'supported'])` → `{'supported': 2, 'partial': 1}`. A missing key reads as 0, not an error.

```python
>>> from collections import Counter
>>> Counter(['supported', 'partial', 'supported'])
Counter({'supported': 2, 'partial': 1})
>>> Counter('aab')['z']
0
```

> [!warning] Watch out
> reading a missing key gives 0 but doesn't add it.

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

### lru_cache

`@functools.lru_cache(maxsize=1)`: a decorator that remembers a function's result for given arguments. On a no-argument loader like `_model()`, the first call loads the model and every later call returns the same object instantly.

```python
>>> from functools import lru_cache
>>> @lru_cache(maxsize=1)
... def load(): print('loading'); return 'model'
>>> load()
loading
'model'
>>> load()
'model'
```

> [!warning] Watch out
> if the cached function fails and returns `None` (as `_model()` does), `None` is cached too, so it won't retry.

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

### np.zeros

`np.zeros(n)`: a numpy array of n zeros (float64). `np.zeros(3)` → `array([0., 0., 0.])`. BM25 starts every chunk's score at 0 this way.

```python
>>> np.zeros(3)
array([0., 0., 0.])
```

> [!warning] Watch out
> the default type is float64; pass `dtype=int` for integers.

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

### select

`select(Model)`: SQLAlchemy: starts building a `SELECT … FROM table` query for a mapped class. Nothing runs until `session.execute(...)`.

```python
>>> print(select(Vulnerability).where(Vulnerability.cwe_id == 89))
SELECT … FROM vulnerability WHERE vulnerability.cwe_id = :cwe_id_1
```

> [!warning] Watch out
> building the query runs nothing; `session.execute` does.

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

### x.where

`query.where(condition)`: SQLAlchemy: adds a `WHERE` clause. `select(Vulnerability).where(Vulnerability.cwe_id == 89)` means `SELECT * FROM vulnerability WHERE cwe_id = 89`. `==` on a column builds SQL; it doesn't compare in Python.

```python
>>> q = select(Vulnerability).where(Vulnerability.cwe_id == 89)
# SQL: … WHERE cwe_id = 89
```

> [!warning] Watch out
> use `==` / `.ilike` on columns; Python's `and` / `in` don't build SQL (use `and_`, `.in_`).

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

### x.scalar_one_or_none

`result.scalar_one_or_none()`: SQLAlchemy: the single object in the result, `None` if there are no rows, and an **error** if there are two or more. Safe for lookups by a unique key like `cwe_id`.

```cpp
# 1 row  → the object
# 0 rows → None
# 2 rows → MultipleResultsFound
```

> [!warning] Watch out
> use it only when the condition is unique, like `cwe_id`.

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

### x.scalars

`result.scalars()`: SQLAlchemy: yields the first column of each row (here the `Vulnerability` objects themselves) instead of row tuples.

```python
>>> s.execute(select(Vulnerability).where(...)).scalars().all()
[<Vulnerability CWE-639>, ...]
```

> [!warning] Watch out
> without `.scalars()`, `.all()` returns one-element row tuples.

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

### x.all

`result.all()`: SQLAlchemy: reads every remaining row into a Python list.

```python
>>> rows = s.execute(q).scalars().all()
>>> len(rows)
2
```

> [!warning] Watch out
> it loads every row into memory; fine for small results like these.

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

### x.ilike

`column.ilike(pattern)`: SQLAlchemy: a case-insensitive `LIKE`. `%` matches any text, so `aliases_json.ilike('%"IDOR"%')` finds rows whose alias list contains the whole string `"IDOR"`, quotes included.

```cpp
aliases_json = '["IDOR", "insecure direct object reference"]'
'%"IDOR"%'  → matches (the quoted whole alias)
'%"IDO"%'   → no match
```

> [!warning] Watch out
> the quotes make it whole-alias; without them `'%IDOR%'` would also match longer aliases containing it.

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

### x.group_by

`query.group_by(col)`: SQLAlchemy: SQL `GROUP BY`: one output row per distinct `col` value, so `func.count(...)` counts per group (signatures per CWE).

```cpp
SELECT cwe_id, COUNT(signature_id) … GROUP BY cwe_id
→ (79, 1520), (89, 610), …
```

> [!warning] Watch out
> every selected column must be grouped or aggregated.

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

### x.count

`func.count(col) / list.count(x)`: in SQLAlchemy, `func.count(col)` is SQL `COUNT(col)`, the number of rows (per group with `group_by`). On a Python list, `['a', 'b', 'a'].count('a')` = 2.

```python
>>> ['a', 'b', 'a'].count('a')
2
```

> [!warning] Watch out
> `func.count` counts non-NULL values of its column.

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

### BM25Okapi

`rank_bm25.BM25Okapi(corpus, k1=1.5, b=0.75, epsilon=0.25)`: builds a BM25 index from a list of tokenised documents: per-document word counts, per-word IDF (negative ones floored at epsilon × mean IDF) and the average length.

```python
>>> bm = BM25Okapi([['ssrf', 'server'], ['xss'], ['idor']])
>>> bm.get_scores(['ssrf'])
array([0.417, 0.   , 0.   ])   # rounded
```

> [!warning] Watch out
> it takes token lists, not strings: pass `_tokenize(text)`.

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

### x.get_scores

`bm25.get_scores(query_tokens)`: rank_bm25: a numpy array with one BM25 score per document, in corpus order. Higher = better match; 0 = no query word present.

```python
>>> bm.get_scores(['ssrf'])   # idf ln(2.5/1.5)=0.511 × 2.5/(1+2.0625)
array([0.417, 0.   , 0.   ])
```

> [!warning] Watch out
> it scores every document, so cost grows with the corpus.

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

### x.encode

`model.encode(texts, normalize_embeddings=True)`: sentence-transformers: embeds a list of texts into a `[len(texts), dim]` array (dim = 384 for bge-small). `normalize_embeddings=True` makes every row length 1. (Not `str.encode`, which turns text into bytes.)

```python
>>> model.encode(['a', 'b'], normalize_embeddings=True).shape
(2, 384)
```

> [!warning] Watch out
> BGE queries need the instruction prefix; passages don't.

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

### CrossEncoder

`sentence_transformers.CrossEncoder(model_name)`: loads a model that reads a (query, passage) pair **together** and outputs one relevance score per pair.

```python
>>> ce = CrossEncoder('BAAI/bge-reranker-base')
>>> ce.predict([('ssrf', 'SSRF makes the server fetch…'), ('ssrf', 'XSS runs script…')])
array([ high, low ])   # illustrative
```

> [!warning] Watch out
> one forward pass per pair, so keep the candidate list short.

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

### x.predict

`model.predict(inputs)`: runs a trained model on inputs. For `CrossEncoder`, `predict([(q, p1), (q, p2)])` returns one score per pair, e.g. `array([7.4, -3.2])`; only their order matters here.

```python
>>> scores = model.predict(pairs)   # 40 pairs
>>> scores.shape
(40,)
```

> [!warning] Watch out
> pair the scores with the hits by position (`zip(..., strict=True)`).

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

### x.create_collection

`client.create_collection(name, vectors_config=...)`: Qdrant: creates a named store of vectors, fixing their size and the distance used to compare them.

```cpp
c.create_collection(collection_name='bountybrain',
    vectors_config=qm.VectorParams(size=384, distance=qm.Distance.COSINE))
```

> [!warning] Watch out
> the size must equal the embedder's dimension, or inserts fail.

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

### x.VectorParams

`qm.VectorParams(size=384, distance=Distance.COSINE)`: Qdrant: the vector settings of a collection: how many numbers per vector, and which similarity ranks results (cosine here).

```cpp
qm.VectorParams(size=384, distance=qm.Distance.COSINE)
```

> [!warning] Watch out
> with unit vectors, cosine and dot product rank identically.

*Used in:* [[Day 12 - BountyBrain#Code, explained|Day 12]]

## 22. C++ standard library, atomics and libpqxx (Day 13: DECS, Micro Exchange)

### std::list

`std::list<T>`: a **doubly linked list**: every element is its own node with prev/next pointers. Insert or remove anywhere in O(1) if you hold an iterator, but no `[i]` indexing. Iterators stay valid when *other* elements move or are removed, which is why the LRU map can store them.

```cpp
std::list<int> l = {3, 2, 1};
auto it = std::next(l.begin(), 2);   // points at 1
l.splice(l.begin(), l, it);         // l = {1, 3, 2}; it still points at 1
```

> [!warning] Watch out
> No random access: reaching the k-th element walks k nodes.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### std::pair

`std::pair<A, B>`: two values in one object, read as `.first` and `.second`. The LRU list stores `std::pair<int, std::string>` = (key, value).

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### std::unordered_map

`std::unordered_map<K, V>`: a hash table from keys to values: lookup, insert and erase in expected O(1), no ordering. `index[7]` creates an entry for 7 if missing; `index.find(7)` does not.

```cpp
std::unordered_map<int, std::string> m;
m[7] = "x";              // insert
m.count(7);   // 1
m.find(8) == m.end();   // true: not there
m[8];         // creates key 8 with "" !
```

> [!warning] Watch out
> `m[key]` inserts a default value when the key is missing; use `find` to just look.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### x.find

`.find(key)`: on a map: an iterator to the entry for `key`, or `.end()` if there is none. `if (it == index.end()) return false;` is the "miss" test.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### x.end

`.end()`: an iterator **one past** the last element: not a real element, used as "not found" and as the stop point of loops.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### x.begin

`.begin()`: an iterator to the first element. On the LRU list, the front = most recently used.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### x.splice

`list.splice(pos, other, it)`: moves the node `it` points at to just before `pos`, by relinking pointers: no copy, no allocation. `items.splice(items.begin(), items, it)` moves an entry to the front, and `it` stays valid.

```cpp
std::list<int> l = {3, 2, 1};
l.splice(l.begin(), l, std::prev(l.end()));   // {1, 3, 2}
```

> [!warning] Watch out
> Moving between two different lists also works and is still O(1) for a single node.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### x.erase

`.erase(it) / .erase(key)`: removes an element: from a list by iterator (O(1)), from a map by key or iterator. `index.erase(2)` deletes key 2.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### x.size

`.size()`: the number of elements. `items.size() >= capacity` means the cache is full.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]], [[Day 14 - Raft#Code, explained|Day 14]]

### x.back

`.back()`: a reference to the **last** element. On the LRU list that's the least recently used entry, the next to evict.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]], [[Day 14 - Raft#Code, explained|Day 14]]

### x.front

`.front()`: a reference to the **first** element. On `std::queue`, the oldest item (the next connection to lend).

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### x.pop_back

`.pop_back()`: removes the last element (O(1) on a list). Used after reading `back()` to evict.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### x.emplace_front

`.emplace_front(args...)`: builds a new element **in place** at the front, from constructor arguments: `items.emplace_front(4, "d")` creates the pair (4, "d") directly in the new node.

```cpp
std::list<std::pair<int, std::string>> items;
items.emplace_front(4, "d");   // items = {(4, "d")}
```

> [!warning] Watch out
> `push_front(std::make_pair(4, "d"))` does the same with an extra temporary.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### x.push_back

`.push_back(x)`: appends a copy of `x` at the end of a vector or list. `trades_.push_back(Trade{...})`; a vector may reallocate when it outgrows its capacity (hence `reserve`).

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]], [[Day 14 - Raft#Code, explained|Day 14]]

### x.push

`.push(x)`: on `std::queue`: adds `x` at the back. `pool.push(conn)` returns a connection to the pool.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### x.empty

`.empty()`: true when there are no elements. `!pool.empty()` = a connection is available.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]], [[Day 14 - Raft#Code, explained|Day 14]]

### std::mutex

`std::mutex`: a lock: only one thread can hold it at a time; others block in `lock()` until it's free. Protects data several threads touch.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### std::lock_guard

`std::lock_guard<std::mutex> lock(mtx);`: locks `mtx` now and unlocks it automatically when `lock` goes out of scope (RAII), even if the code returns early or throws.

```cpp
void put(int k) {
    std::lock_guard<std::mutex> lock(mtx);   // locked here
    if (k < 0) return;                        // unlocked here
    ...
}                                            // or here
```

> [!warning] Watch out
> Hold the lock as briefly as possible; a lock_guard can't be released early (use unique_lock for that).

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### std::unique_lock

`std::unique_lock<std::mutex> lock(mtx);`: like `lock_guard`, but you can `unlock()` early and a condition variable can release and re-take it while waiting. Needed for `cv.wait`.

```cpp
std::unique_lock<std::mutex> lock(mtx);
cv.wait(lock, [&] { return !pool.empty(); });  // unlocks while asleep
auto c = pool.front(); pool.pop();
lock.unlock();
```

> [!warning] Watch out
> `cv.wait` needs a unique_lock, not a lock_guard: it must unlock and re-lock the mutex.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### x.unlock

`.unlock()`: releases a `unique_lock` before its scope ends. The pool unlocks before `notify_one` so the woken thread doesn't block on the mutex.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### x.notify_one

`cv.notify_one()`: wakes **one** thread waiting on the condition variable (if any). It then re-checks its predicate.

```cpp
{ std::lock_guard<std::mutex> g(mtx); pool.push(conn); }
cv.notify_one();   // wake one waiter
```

> [!warning] Watch out
> A notify with nobody waiting is lost; the predicate in `wait` is what makes that safe.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### std::atomic

`std::atomic<size_t> head_{0};`: a variable that threads can read and write **without a data race**. Each operation can carry a memory order (relaxed, acquire, release, seq_cst) that says what other memory it orders.

```cpp
std::atomic<int> n{0};
// two threads: n.fetch_add(1) 1,000 times each
// n == 2000 always; a plain int could end below 2000
```

> [!warning] Watch out
> Atomic does not mean ordered: the default is seq_cst, but relaxed operations order nothing else.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]], [[Day 14 - Raft#Code, explained|Day 14]]

### x.load

`x.load(order)`: reads an atomic. `head_.load(std::memory_order_acquire)` reads it and makes later reads see everything the writer did before its release store.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### x.store

`x.store(v, order)`: writes an atomic. `head_.store(head + 1, std::memory_order_release)` publishes every earlier write together with the new value.

```cpp
slots_[0] = cmd;                                   // plain write
head_.store(1, std::memory_order_release);         // publish
// consumer: if (head_.load(std::memory_order_acquire) == 1) read slots_[0];
```

> [!warning] Watch out
> Release only helps a reader that loads the same atomic with acquire (or stronger).

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### std::memory_order_relaxed

`std::memory_order_relaxed`: atomicity only, **no ordering** of other memory. Fine for reading your own index (only you write it).

```cpp
// producer reading its own index:
const size_t head = head_.load(std::memory_order_relaxed);
```

> [!warning] Watch out
> Relaxed on the *publishing* store would let the consumer see the new head before the slot data.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### std::memory_order_acquire

`std::memory_order_acquire`: for loads: reads after it can't move before it, and if it reads a released value it sees all writes before that release.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### std::memory_order_release

`std::memory_order_release`: for stores: writes before it can't move after it, so a reader that acquires this value sees them all.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### std::atomic_ref

`std::atomic_ref<uint64_t>(x).store(v, order)`: C++20: treats an **ordinary** variable as atomic for one operation. The WAL header lives in an mmap'd file, so `count` is a plain `uint64_t` updated through `atomic_ref`.

```cpp
uint64_t count = 0;   // lives in an mmap'd file
std::atomic_ref<uint64_t>(count).store(5, std::memory_order_release);
```

> [!warning] Watch out
> While any atomic_ref to an object is in use, all access to it should go through atomic_ref.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### alignas

`alignas(64) T x;`: forces `x` to start at an address that is a multiple of 64 bytes, i.e. at the start of a cache line. Two `alignas(64)` members can never share a line, which prevents false sharing.

```cpp
struct Ring {
    alignas(64) std::atomic<size_t> head_{0};   // bytes 0..63
    alignas(64) std::atomic<size_t> tail_{0};   // bytes 64..127
};
static_assert(sizeof(Ring) == 128);
```

> [!warning] Watch out
> It pads only around those members; data elsewhere (the 40-byte slots) can still share lines.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### rdtsc

`rdtsc()`: reads the CPU's time-stamp counter (cycles since reset) in ~20 cycles: far cheaper than `std::chrono`. Here a helper wrapping the x86 instruction; `TscClock::to_ns` converts cycles to ns.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### pqxx::connection

`pqxx::connection`: one open connection to Postgres (a server-side backend process). Opening one costs milliseconds, hence the pool. `db::` is the project's alias for `pqxx::`.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### db::work

`db::work txn{*conn};`: starts a **transaction** on a connection (sends `BEGIN`). Nothing is permanent until `txn.commit()`; if `txn` is destroyed first, it rolls back.

```cpp
db::work txn{*conn};                                     // BEGIN
auto r = txn.exec_params("SELECT value FROM kv_store WHERE id = $1", 7);
txn.commit();                                            // COMMIT
```

> [!warning] Watch out
> For a single read-only query, `pqxx::nontransaction` skips BEGIN/COMMIT: 1 round trip instead of 3.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### x.exec_params

`txn.exec_params(sql, a, b)`: runs a SQL statement with `$1`, `$2` placeholders filled by `a`, `b` (no string concatenation, so no SQL injection). Returns a `result` of rows: `r.empty()`, `r[0][0].as<string>()`.

```cpp
auto r = txn.exec_params("SELECT value FROM kv_store WHERE id = $1", 7);
if (r.empty()) { /* 404 */ }
std::string v = r[0][0].as<std::string>();
```

> [!warning] Watch out
> Never build SQL by concatenating user input; always use placeholders.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### x.commit

`txn.commit()`: sends `COMMIT`: the transaction's changes become permanent. Postgres replies only after its WAL record is flushed (unless synchronous_commit is off).

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### x.set_content

`res.set_content(body, "text/plain")`: cpp-httplib: sets the HTTP response body and its Content-Type. The status defaults to 200.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### std::exception

`catch (const std::exception &e)`: the base class of standard errors. Catching it catches libpqxx's errors too; `e.what()` is the message.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

### x.what

`e.what()`: the error message of an exception, as a C string.

*Used in:* [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

## 23. C++ standard library (Day 14: Raft)

### std::to_string

`std::to_string(x)`: turns a number into a `std::string`. `std::to_string(42)` = "42". raftkv's wire format is text fields, so every number goes through it.

```cpp
std::string f = std::to_string(7) + "/" + std::to_string(42);   // "7/42"
```

> [!warning] Watch out
> For doubles it prints 6 decimals: std::to_string(0.5) = "0.500000".

*Used in:* [[Day 14 - Raft#Code, explained|Day 14]]

### x.resize

`v.resize(n)`: makes a vector exactly `n` elements long: drops the extra ones from the end, or adds default ones. `log_.resize(3)` keeps entries 0–2 and **truncates** the rest.

```cpp
std::vector<int> log = {1, 1, 2, 2, 2};
log.resize(3);   // {1, 1, 2}
log.push_back(4);   // {1, 1, 2, 4}
```

> [!warning] Watch out
> raftkv's index → position mapping is `index - snapshot_index_ - 1`; resizing to that keeps entries before `index`.

*Used in:* [[Day 14 - Raft#Code, explained|Day 14]]

### x.notify_all

`cv.notify_all()`: wakes **every** thread waiting on the condition variable. `apply_cv_.notify_all()` wakes the apply loop and every client waiting for its entry to commit.

```cpp
{ std::lock_guard<std::mutex> g(mutex_); commit_index_ = 7; }
apply_cv_.notify_all();   // apply loop + waiting clients re-check
```

> [!warning] Watch out
> Waiters must use a predicate (or re-check in a loop): they may wake with nothing to do.

*Used in:* [[Day 14 - Raft#Code, explained|Day 14]]

### std::make_shared

`std::make_shared<T>(args...)`: allocates a `T` and returns a `shared_ptr` to it; the object lives until the last copy of the pointer is gone. The vote counter is shared by one thread per peer this way.

```cpp
auto votes = std::make_shared<std::atomic<int>>(1);
std::thread([votes] { ++(*votes); }).detach();   // the copy keeps it alive
```

> [!warning] Watch out
> Capturing a raw pointer or `this` instead would dangle once the owner is destroyed.

*Used in:* [[Day 14 - Raft#Code, explained|Day 14]]

### std::thread

`std::thread([...] { ... })`: starts a new OS thread running the lambda. `.detach()` lets it run on its own; `.join()` waits for it.

```cpp
std::thread t([] { work(); });
t.join();   // wait for it
```

> [!warning] Watch out
> Destroying a joinable thread (neither joined nor detached) calls std::terminate.

### x.detach

`t.detach()`: lets a `std::thread` run independently; you can no longer join it. The vote threads are detached (and capture `this`, a shutdown hazard).

### x.data

`s.data()`: a pointer to the first byte of a string's (or vector's) contiguous storage, for C-style APIs like `write(fd, ptr, len)`.

*Used in:* [[Day 14 - Raft#Code, explained|Day 14]]

### std::cerr

`std::cerr << msg`: the standard error stream (unbuffered). Used for start-up failures.

*Used in:* [[Day 14 - Raft#Code, explained|Day 14]]
