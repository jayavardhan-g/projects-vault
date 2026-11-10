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

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 04 - Speculative decoding#Code, explained|Day 4]], [[Day 05 - CUDA I#Code, explained|Day 5]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]], [[Day 14 - Raft#Code, explained|Day 14]]

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

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 07 - Seminar and speech basics#Code, explained|Day 7]], [[Day 08 - CosyVoice#Code, explained|Day 8]]

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

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 08 - CosyVoice#Code, explained|Day 8]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]]

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

*Used in:* [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]]

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

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]]

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

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

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

*Used in:* [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 11 - Photo-Seek II and emotion steering#Code, explained|Day 11]], [[Day 12 - BountyBrain#Code, explained|Day 12]], [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

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

*Used in:* [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 10 - Photo-Seek I#Code, explained|Day 10]], [[Day 12 - BountyBrain#Code, explained|Day 12]]

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

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]]

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

*Used in:* [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 13 - DECS and Micro Exchange#Code, explained|Day 13]]

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

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]]

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

*Used in:* [[Day 01 - mini-vLLM I#Code, explained|Day 1]], [[Day 02 - mini-vLLM II#Code, explained|Day 2]], [[Day 03 - mini-vLLM III#Code, explained|Day 3]], [[Day 08 - CosyVoice#Code, explained|Day 8]]

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
