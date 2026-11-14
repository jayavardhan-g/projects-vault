"""Day 10 rebuild: a tiny Photo-Seek search. SQL prefilter first, then exact vector ranking.

Run:  python3 day10_tiny_search.py
Needs only the standard library (math, sqlite3): no numpy, no FAISS, no CLIP.

It checks four steps in order and stops at the first one that is not written yet.
Do them one at a time. The guide is the vault note "D10-09 Rebuild".

  step 1  cosine(a, b)                       <- start here: the only TODO marker in the file
  step 2  flat_search(query, vectors, k, allowed)
  step 3  candidate_ids(conn, date_from, date_to, person)
  step 4  search(...)  = step 3, then step 2 on the survivors

What each step mirrors in the real repo (branch upgrade-2026-10):
  cosine / l2_normalize  -> embed/base.py:79 and the normalize call in embed/clip.py
  flat_search            -> index/faiss_store.py:111  (IndexFlatIP + IDSelectorBatch)
  candidate_ids          -> search/engine.py:109
  search                 -> search/engine.py:78
"""
import math
import sqlite3

# ----------------------------------------------------------------------------- toy data
# Six "photos". A real CLIP vector has 512 numbers; here each has 2, so it is an arrow
# on paper and only its angle matters. The query "beach" is the arrow at 0 degrees.

ANGLES = {5: 5, 9: 15, 12: 70, 40: 30, 77: 50, 81: 120}          # photo id -> angle in degrees

TAKEN_AT = {
    5: "2026-03-02 10:00:00",
    9: "2025-07-14 09:30:00",
    12: "2026-06-20 16:05:00",
    40: "2026-07-04 11:12:00",
    77: "2026-08-31 18:20:00",      # the evening of the last day of summer
    81: "2026-09-01 08:00:00",
}

FACES = [(5, "mom"), (9, "mom"), (12, "mom"), (40, "mom"), (40, "dad"), (77, "mom"), (81, "dad")]


def unit(angle_deg):
    """The 2-D unit vector at this angle."""
    r = math.radians(angle_deg)
    return [math.cos(r), math.sin(r)]


VECTORS = {pid: unit(a) for pid, a in ANGLES.items()}             # photo id -> unit vector
QUERY = unit(0)                                                   # "beach"


def make_catalog():
    """An in-memory SQLite database with two tables, like a cut-down schema.sql."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE photos (id INTEGER PRIMARY KEY, taken_at TEXT)")
    conn.execute("CREATE TABLE faces (photo_id INTEGER, person TEXT)")
    conn.executemany("INSERT INTO photos VALUES (?, ?)", TAKEN_AT.items())
    conn.executemany("INSERT INTO faces VALUES (?, ?)", FACES)
    return conn


# ----------------------------------------------------------------------------- given

def l2_normalize(vector):
    """Scale a vector to length 1 (same as embed/base.py:79)."""
    norm = sum(v * v for v in vector) ** 0.5
    if norm == 0.0:
        return list(vector)
    return [v / norm for v in vector]


# ----------------------------------------------------------------------------- step 1

def cosine(a, b):
    """Cosine similarity of two vectors of the same length (lists of floats).

    They are NOT guaranteed to be unit length, so divide by both lengths.
    Return 0.0 if either vector has length 0.
    cosine([3, 4], [4, 3]) -> 0.96      cosine([3, 4], [6, 8]) -> 1.0
    """
    # TODO(human): 4-6 lines. Dot product, the two lengths, the division, the zero guard.
    raise NotImplementedError("step 1")


# ----------------------------------------------------------------------------- step 2

def flat_search(query, vectors, k, allowed=None):
    """Exact nearest-neighbour search, like IndexFlatIP with an optional IDSelectorBatch.

    query    a unit vector
    vectors  dict: photo id -> unit vector
    k        how many results at most
    allowed  None  -> every photo may be returned
             a set -> only these ids may be returned (an EMPTY set means: return nothing)

    Returns a list of (photo id, score), best score first, at most k long.
    Score = inner product (all vectors are unit length, so it equals the cosine).
    """
    # STEP 2 (after step 1 passes): about 6 lines.
    raise NotImplementedError("step 2")


# ----------------------------------------------------------------------------- step 3

def candidate_ids(conn, date_from=None, date_to=None, person=None):
    """Photo ids that pass the metadata filters, like SearchEngine.candidate_ids.

    date_from, date_to  'YYYY-MM-DD' strings or None. Both ends are INCLUSIVE days.
    person              a name or None; the photo must contain a face of that person.

    Returns None if no filter was given at all (meaning: search everything),
    otherwise a set of ids (possibly empty, meaning: nothing matches).
    Use ? parameters, never string formatting, for the values.
    """
    # STEP 3 (after step 2 passes): about 15 lines. Mind taken_at's time part.
    raise NotImplementedError("step 3")


# ----------------------------------------------------------------------------- step 4

def search(conn, vectors, query, k, date_from=None, date_to=None, person=None):
    """Filter in SQL first, then rank only the survivors. Returns [(photo id, score), ...]."""
    # STEP 4 (after step 3 passes): 2-4 lines.
    raise NotImplementedError("step 4")


# ----------------------------------------------------------------------------- given: the wrong way

def post_filter_search(conn, vectors, query, k, date_from=None, date_to=None, person=None):
    """Rank everything, take the top k, THEN drop what fails the filters. For comparison only."""
    top = flat_search(query, vectors, k)
    ok = candidate_ids(conn, date_from, date_to, person)
    return [(pid, s) for pid, s in top if ok is None or pid in ok]


# ----------------------------------------------------------------------------- checks

def close(x, y):
    return abs(x - y) < 1e-3


def ids(hits):
    return [pid for pid, _ in hits]


def check_step_1():
    assert close(cosine([3, 4], [4, 3]), 0.96)
    assert close(cosine([3, 4], [6, 8]), 1.0), "length must not matter"
    assert close(cosine([1, 0], [0, 1]), 0.0)
    assert close(cosine([1, 0], [-1, 0]), -1.0)
    assert cosine([0, 0], [1, 0]) == 0.0, "zero vector: return 0.0, do not divide by zero"
    assert close(cosine(l2_normalize([3, 4]), l2_normalize([4, 3])), 0.96)


def check_step_2():
    hits = flat_search(QUERY, VECTORS, 2)
    assert ids(hits) == [5, 9], hits
    assert close(hits[0][1], 0.996) and close(hits[1][1], 0.966)
    assert ids(flat_search(QUERY, VECTORS, 10)) == [5, 9, 40, 77, 12, 81], "k larger than the library"
    assert ids(flat_search(QUERY, VECTORS, 2, allowed={12, 40, 77})) == [40, 77]
    assert ids(flat_search(QUERY, VECTORS, 10, allowed={12, 40, 77})) == [40, 77, 12]
    assert flat_search(QUERY, VECTORS, 2, allowed=set()) == [], "empty set = nothing, not everything"
    assert flat_search(QUERY, VECTORS, 2, allowed={999}) == []


def check_step_3():
    conn = make_catalog()
    assert candidate_ids(conn) is None, "no filters -> None (search everything)"
    assert candidate_ids(conn, "2026-06-01", "2026-08-31") == {12, 40, 77}, \
        "photo 77 was taken at 18:20 on 2026-08-31 and must be included"
    assert candidate_ids(conn, "2026-06-01", "2026-08-31", "mom") == {12, 40, 77}
    assert candidate_ids(conn, "2026-06-01", "2026-08-31", "dad") == {40}
    assert candidate_ids(conn, person="dad") == {40, 81}
    assert candidate_ids(conn, date_from="2026-09-01") == {81}
    assert candidate_ids(conn, person="ravi") == set(), "unknown person -> empty set, not None"
    assert candidate_ids(conn, person="x' OR '1'='1") == set(), "use ? parameters"


def check_step_4():
    conn = make_catalog()
    hits = search(conn, VECTORS, QUERY, 2, "2026-06-01", "2026-08-31", "mom")
    assert ids(hits) == [40, 77], hits
    assert close(hits[0][1], 0.866) and close(hits[1][1], 0.643)
    assert ids(search(conn, VECTORS, QUERY, 2)) == [5, 9], "no filters -> whole library"
    assert search(conn, VECTORS, QUERY, 2, person="ravi") == []
    # The reason for the whole design:
    wrong = post_filter_search(conn, VECTORS, QUERY, 2, "2026-06-01", "2026-08-31", "mom")
    print("   pre-filter  top 2:", ids(hits))
    print("   post-filter top 2:", ids(wrong), "<- the two best-looking photos fail the date filter")
    assert wrong == []


if __name__ == "__main__":
    for n, check in enumerate([check_step_1, check_step_2, check_step_3, check_step_4], 1):
        try:
            check()
        except NotImplementedError:
            print(f"step {n}: not written yet. Write it, then run again.")
            break
        except AssertionError as exc:
            print(f"step {n}: FAILED  {exc}")
            raise
        print(f"step {n}: ok")
    else:
        print("all four steps pass. Now answer the questions at the end of D10-09 Rebuild.")
