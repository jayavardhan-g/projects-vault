---
type: study-block
day: 10
block: 3
tags: [interview, day10, clip, code]
---
# D10-03 · CLIP code: embed/clip.py and embed/base.py (40 min)

Hub: [[Day 10 - Photo-Seek I]] · Previous: [[D10-02 CLIP theory]] · Next: [[D10-04 FAISS theory]]

Files: `src/photoseek/embed/clip.py` (118 lines) and `embed/base.py` (85 lines), branch `upgrade-2026-10`. Line numbers are the real ones. The `embed_texts` loop is already walked line by line in [[Day 10 - Photo-Seek I#4.7 The code, line by line|day note §4.7]]; this note covers everything else.

> [!warning] This code only runs a trained model
> There is no loss, no optimiser and no `.backward()` anywhere in `clip.py`. It loads **pretrained** weights and does forward passes. Training is described in [[D10-02 CLIP theory#9. Why a symmetric cross-entropy]] and was done by others.

## 0. Map of the file

| lines | member | job |
|---|---|---|
| 26–31 | `__init__` | store the config, pick the device, leave the model unloaded |
| 34–41 | `dim` | read the output size from the loaded model |
| 44–45 | `model_id` | `"ViT-B-32:laion2b_s34b_b79k"`, stored with every vector |
| 47–62 | `load` | load weights, preprocessing and tokenizer |
| 64–98 | `embed_images` | paths → unit vectors, one slot per path |
| 100–114 | `embed_texts` | strings → unit vectors |
| 116–118 | `_require_loaded` | fail loudly if `load()` was skipped |

## 1. Construction and lazy loading

`clip.py:26–31`
```python
def __init__(self, config: EmbeddingConfig) -> None:
    self.config = config
    self.device = resolve_device(config.device)
    self._model = None
    self._preprocess = None
    self._tokenizer = None
```

**What:** nothing heavy happens. The three `None`s are placeholders for things `load()` fills.

**Why:** building the object must be cheap. `import open_clip` and `import torch` take seconds and are done **inside** `load`, not at the top of the file. So `photoseek search --dry-run`, the tests, and a pure-filter query never pay for PyTorch [doc: decision 3].

**What breaks if you load in `__init__`:** every command that constructs a `CLIPEmbedder` reads 578 MB of weights, even if it never embeds.

`EmbeddingConfig` [code: `config.py:62`]: `clip_model="ViT-B-32"`, `clip_pretrained="laion2b_s34b_b79k"`, `batch_size=32`, `device="auto"`, `normalize=True`.

`resolve_device("auto")` [code: `base.py:58–76`] returns `"cuda"` if a GPU is available, else `"mps"` on Apple silicon, else `"cpu"`. It imports torch lazily too, and returns `"cpu"` if torch is missing.

## 2. Model loading

`clip.py:47–62`
```python
def load(self) -> None:
    if self._model is not None:
        return
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        self.config.clip_model, pretrained=self.config.clip_pretrained, device=self.device
    )
    model.eval()
    self._model = model
    self._preprocess = preprocess
    self._tokenizer = open_clip.get_tokenizer(self.config.clip_model)
```

| line | what it does | why | what breaks without it |
|---|---|---|---|
| `if self._model is not None: return` | makes `load()` safe to call twice | callers need not track state | a second call would reload 578 MB |
| `create_model_and_transforms(...)` | builds the ViT-B-32 architecture, loads the LAION weights, moves it to the device | one call gives the model **and** its matching preprocessing | — |
| `model, _, preprocess` | the call returns three things: the model, the **training** transform and the **inference** transform [lib]. The middle one is thrown away | training transforms add random crops; search needs the same crop every time | embeddings would change from run to run |
| `model.eval()` | switches layers like dropout to inference behaviour | deterministic outputs | possible noise in the vectors |
| `get_tokenizer("ViT-B-32")` | the text tokenizer that matches this model | token ids must match the ids the model was trained with | garbage text embeddings |

**Where the weights come from** [code: docstring, line 48–50]: the Hugging Face cache (`HF_HOME`). With `HF_HUB_OFFLINE=1` nothing touches the network; that is the setting the offline audit ran under.

## 3. Image preprocessing

`self._preprocess` is a function made by open_clip. For ViT-B-32 it does [lib; not written in your repo]:

| step | result |
|---|---|
| resize so the short side is 224 (bicubic) | e.g. 4000 × 3000 → 299 × 224 |
| centre-crop 224 × 224 | the middle square |
| to tensor, scale to 0–1 | `[3, 224, 224]` float32 |
| subtract the per-channel mean, divide by the std (CLIP's own values) | values around −2 … 2 |

Your code adds one step before it: `image.convert("RGB")` (line 83). A PNG with transparency has 4 channels and a greyscale scan has 1; the model needs exactly 3.

**Consequence to know** [mine]: the centre crop cuts the edges of wide or tall photos. A person at the far edge of a panorama may not be in the vector at all.

## 4. embed_images: paths in, one slot per path out

`clip.py:64–98`
```python
self._require_loaded()
out: list[Vector | None] = [None] * len(paths)
batch_size = max(1, self.config.batch_size)

for start in range(0, len(paths), batch_size):
    tensors, slots = [], []
    for offset, path in enumerate(paths[start:start + batch_size]):
        try:
            with Image.open(path) as image:
                tensors.append(self._preprocess(image.convert("RGB")))
            slots.append(start + offset)
        except Exception as exc:  # one bad file must not stop the run
            log.warning("could not embed %s: %s", path, exc)
    if not tensors:
        continue

    with torch.inference_mode():
        batch = torch.stack(tensors).to(self.device)
        features = self._model.encode_image(batch).float()
        if self.config.normalize:
            features = torch.nn.functional.normalize(features, dim=-1)
    for slot, vector in zip(slots, features.cpu().numpy()):
        out[slot] = vector

return out
```

**Variables and shapes**

| name | type / shape | meaning | toy value (3 paths, batch 2) | real size |
|---|---|---|---|---|
| `paths` | list of N paths | images to embed | `[a.jpg, bad.jpg, c.jpg]` | up to 256 per call from `cmd_index` |
| `out` | list of N, starts all `None` | the answer, same order as `paths` | `[None, None, None]` | |
| `batch_size` | int | images per forward pass | 2 | 32 default; 64 in the evals |
| `tensors` | list of `[3, 224, 224]` | preprocessed images that opened | | |
| `slots` | list of ints | position in `out` for each tensor | | |
| `batch` | `[B, 3, 224, 224]` float32 on the device | one stacked batch | `[1, 3, 224, 224]` | `[32, 3, 224, 224]` |
| `features` | `[B, 512]` float32 | one vector per image | `[1, 512]` | `[32, 512]` |

**Traced** with `paths = [a.jpg, bad.jpg, c.jpg]`, `batch_size = 2`, where `bad.jpg` is corrupt:

| step | state |
|---|---|
| start | `out = [None, None, None]` |
| batch `start=0`, offset 0, `a.jpg` | opens → `tensors = [t_a]`, `slots = [0]` |
| batch `start=0`, offset 1, `bad.jpg` | `Image.open` raises → warning logged; **nothing appended** |
| forward pass | `batch` `[1, 3, 224, 224]` → `features` `[1, 512]` → normalised |
| write back | `out[0] = v_a` → `[v_a, None, None]` |
| batch `start=2`, offset 0, `c.jpg` | `tensors = [t_c]`, `slots = [2]` (2 + 0) |
| forward pass, write back | `out[2] = v_c` → **`[v_a, None, v_c]`** |

**Why `slots`?** A failed image leaves a hole in the batch. Without `slots`, `v_c` would be written into position 1 and every later vector would belong to the wrong photo. That is the worst possible bug for a search engine: silent and wrong.

**Why `None` and not an exception?** Indexing 10,000 photos takes minutes. One corrupt file must not throw away the run. The caller counts the `None`s [code: `cli.py:139–140`] and prints "N unreadable".

**Line by line: the forward pass**

| line | what | why | if removed |
|---|---|---|---|
| `torch.inference_mode()` | no gradient bookkeeping | less memory, faster; we never train | works, but wastes memory |
| `torch.stack(tensors)` | B tensors `[3,224,224]` → one `[B,3,224,224]` | the model takes a batch | — |
| `.to(self.device)` | moves the batch to the GPU | model and data must be on the same device | runtime error on GPU |
| `encode_image(batch)` | the ViT: 49 patches + class token → 12 layers → class token → projection | gives `[B, 512]` | — |
| `.float()` | forces float32 | if the model ran in half precision the output would be float16; FAISS wants float32 | dtype mismatch later |
| `normalize(features, dim=-1)` | divides each row by its length | so inner product = cosine | scores depend on vector length; `IndexFlatIP` no longer gives cosine |
| `.cpu().numpy()` | GPU tensor → numpy array | FAISS (CPU) and SQLite take numpy / bytes | — |

**Where the output goes** [code: `cli.py:138–144`]: each vector is written to SQLite (`catalog.put_embedding`, as a float32 blob) **and** added to FAISS (`store.add`), then the photo is marked indexed. See [[D10-05 FAISS code#1. add - custom ids and replacing an old vector]].

## 5. Tokenisation and embed_texts

`clip.py:100–114` (walked line by line in the day note). The parts worth adding:

**Tokenisation** [lib]. `self._tokenizer(["beach"])` returns a tensor of token ids, shape `[1, 77]`, dtype int64:

```
[49406,  <id of "beach">,  49407,  0, 0, 0, …, 0]      ← 77 numbers
  start      the word       end       padding
```

- 49406 and 49407 are CLIP's start and end markers; the vocabulary has 49,408 word pieces.
- Every text is padded (or **cut**) to exactly 77 tokens, so texts stack into one tensor.
- The text tower's output is read **at the end-of-text position**, then projected to 512.

The exact id of "beach" is not shown: the env is broken today, so I could not run the tokenizer.

**Batching.** `batch_size = max(1, self.config.batch_size) * 8` (line 106). Text is cheap: 77 token ids against 150,528 pixel values per image. So 32 × 8 = 256 texts per pass by default. `max(1, …)` guards against a config of 0, which would make `range(0, n, 0)` raise.

**The search call** [code: `engine.py:105`]: `[vector] = self.embedder.embed_texts([structured.text])`. One text in, a list of one array out. The `[vector] = …` form unpacks it and would raise if the list did not have exactly one item.

## 6. Output: dimensions and data types

| what | value | source |
|---|---|---|
| vector length | **512** for ViT-B-32 (768 for ViT-L-14) | `dim` property, line 34–41: reads `text_projection.shape[-1]` |
| dtype | float32 | `.float()` |
| container | a numpy array of shape `(512,)` per item, in a Python list | `.cpu().numpy()` |
| length (L2 norm) | 1.0 | `normalize` |
| bytes per vector | 512 × 4 = 2,048 | |
| a failed image | `None` in its slot | line 75 |

The type hint says `Vector = Sequence[float]` [code: `base.py:17`]. At run time the items are numpy arrays; `cmd_index` calls `vec.tolist()` before SQLite.

`model_id` (line 44) = `"ViT-B-32:laion2b_s34b_b79k"`. It is stored in `embeddings.model` beside each vector. **Why:** vectors from two different models live in two different spaces. Comparing them gives nonsense with no error. The stored id makes a model change detectable [code: `base.py:31–32`].

## 7. L2 normalisation, by hand

`normalize(x, dim=-1)` is $x / \lVert x \rVert$ per row.

$x = (3, 4)$ → length $\sqrt{9 + 16} = 5$ → $(0.6, 0.8)$ → new length $\sqrt{0.36 + 0.64} = 1$.

`base.py:79–85` has the same thing in pure Python, used where numpy is not wanted:

```python
def l2_normalize(vector):
    norm = sum(v * v for v in vector) ** 0.5
    if norm == 0.0:
        return list(vector)
    return [v / norm for v in vector]
```

The `norm == 0.0` line avoids dividing by zero (which would give NaN, and a NaN in an index poisons every score it touches).

## 8. The Protocol pattern in base.py

**The problem.** `SearchEngine` needs "something that turns text into vectors". If its code named `CLIPEmbedder` directly, every test would have to load 578 MB of weights.

**The idea: structural typing.** A `Protocol` describes a **shape**, not a family. Any object with the right methods fits, with no inheritance.

`base.py:41–50`
```python
@runtime_checkable
class TextEmbedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed_texts(self, texts: Sequence[str]) -> list[Vector]:
        ...
```

A test can then pass this, and the engine cannot tell the difference:

```python
class FakeEmbedder:
    dim = 2
    def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]
```

I did exactly this today to trace the engine without numpy ([[D10-07b Search engine code#0. How this note's values were produced]]).

| | Abstract base class (ABC) | Protocol |
|---|---|---|
| a class fits when | it **inherits** from the base | it **has the methods** |
| a fake in a test | must import and subclass the base | any small class works |
| checked by | Python at instantiation | a type checker (mypy); `isinstance` if `@runtime_checkable` |

Three protocols [code: `base.py:20–55`]: `ImageEmbedder` (`dim`, `model_id`, `embed_images`), `TextEmbedder` (`dim`, `embed_texts`), and `Embedder` = both. **CLIP satisfies `Embedder`; ArcFace satisfies only `ImageEmbedder`**, because faces have no text side.

`@runtime_checkable` allows `isinstance(x, TextEmbedder)`. It checks only that the method **names** exist, not their argument types [lib].

**Honest limit:** `CLIPEmbedder` does not *declare* that it implements the protocol (it says so only in its docstring, line 23). And `SearchEngine` types its `embedder` argument as `object | None` [code: `engine.py:64`], so the protocol is a convention here more than an enforced contract.

## Check

1. Why is `import open_clip` inside `load()` and not at the top of the file?
2. `paths` has 5 images, the 2nd and 5th are corrupt, `batch_size = 2`. What is `out`, and what are `slots` in each batch?
3. What is the shape and dtype of `batch` and of `features` for a full default batch?
4. What would go wrong, silently, if `normalize` were set to `False` in the config?
5. Which element of `create_model_and_transforms`'s return value is discarded, and why?
6. Give one reason to prefer a Protocol over an ABC here.

> [!success]- Answers
> 1. So importing the package and running non-embedding commands stays fast and needs no PyTorch.
> 2. `[v1, None, v3, v4, None]`. Batches: `start=0` → `slots=[0]`; `start=2` → `slots=[2, 3]`; `start=4` → no tensors, `continue`.
> 3. `[32, 3, 224, 224]` float32 and `[32, 512]` float32.
> 4. FAISS would rank by raw inner product, which favours long vectors; the "cosine" scores would no longer be cosines.
> 5. The training-time transform (random augmentation); search needs the deterministic inference transform.
> 6. Tests pass tiny fakes with no inheritance and no model weights.

## Five things to remember

1. **Lazy:** construct cheap, `load()` once, heavy imports inside functions.
2. **`embed_images`:** open → RGB → preprocess `[3,224,224]` → stack → `encode_image` → float32 → normalise → numpy; a bad file gives `None` in **its own slot**.
3. **`embed_texts`:** tokenise to `[n, 77]` → `encode_text` → `[n, 512]` → normalise.
4. **Output:** 512 float32 numbers of length 1; `model_id` is stored so spaces never mix.
5. **Protocol** = "has these methods"; it is what lets the engine be tested with fakes.
