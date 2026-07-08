# bplus-tree-storage-engine

**Component 2 of 3 in Storage & Persistence** of the [`strata-database-engine`](https://github.com/juanmmm21/strata-database-engine) ecosystem.
Repo: [`github.com/juanmmm21/bplus-tree-storage-engine`](https://github.com/juanmmm21/bplus-tree-storage-engine)

A disk-backed B+Tree storage engine, written from scratch in Python: a bounded, LRU-backed buffer pool sits in front of a page device, and a classic B+Tree — page splits on insert, opportunistic sibling merges on delete, a linked chain of leaves for ordered range scans — persists key/value pairs as fixed-size, checksummed binary pages. It is one of the two interchangeable storage backends of the ecosystem (the other being [`lsm-tree-engine`](https://github.com/juanmmm21/lsm-tree-engine)), built on top of the durability guarantees of [`write-ahead-log-recovery`](https://github.com/juanmmm21/write-ahead-log-recovery).

---

## What it is and what problem it solves

Every relational database needs a way to keep an ordered index on disk that stays balanced under arbitrary insertions and deletions, supports fast point lookups, and — unlike a hash table — supports fast ordered range scans. The B+Tree is the data structure virtually every production relational engine (PostgreSQL, MySQL/InnoDB, SQLite) uses for exactly this reason: internal nodes only route, all the data lives in the leaves, and the leaves are chained together so a range scan never has to walk back up the tree.

This project implements that structure from first principles, on top of a real binary page format (not an in-memory toy): a slotted-page layout with a checksummed header, a buffer pool with a bounded, configurable eviction policy, and split/merge logic proven correct against thousands of randomized insert/delete sequences.

## Role in `strata-database-engine`

```text
                    write-ahead-log-recovery
                     (durability discipline)
                              │
                              ▼
                 ┌─────────────────────────────┐
                 │  bplus-tree-storage-engine   │   (this repo)
                 │  slotted pages · buffer pool  │
                 │  split/merge · leaf chain      │
                 └───────────────┬───────────────┘
                                 │ pluggable storage backend
                                 ▼
                    mvcc-transaction-manager /
                    lock-manager-deadlock-detector
                                 │
                                 ▼
                              nanosql
```

This repo does not import or depend on any other subproject: it exposes a versioned binary page format and a plain Python API (`BPlusTree.search` / `insert` / `delete` / `scan`) as its integration surface. `nanosql` is where a real engine would sequence writes through `write-ahead-log-recovery` before they land on these pages; that composition is deliberately out of scope here, same as the WAL repo not depending on any specific storage engine.

## Goal / skills demonstrated

- A realistic **slotted-page** binary layout (header + cell-pointer array + cell content growing from the end of the page), the same scheme used by SQLite and Postgres, instead of a naive fixed-slot format.
- **Byte-size-aware splitting**: the split point is chosen by cumulative serialized size, not by entry count, so a page with wildly uneven key/value sizes still always splits into two pages that both fit — proven, not assumed (see `_find_leaf_split_point` / `_find_internal_split_point`).
- Correct **cascading split propagation** up to the root (including root growth) and **opportunistic sibling merging** on delete (including root collapse when the tree shrinks by a level).
- A **bounded buffer pool** with a pluggable `ReplacementPolicy` (LRU by default) and write-back semantics: dirty pages are only persisted on eviction, `flush()`, or `close()`.
- Page-level integrity: every page carries a version and a CRC32 checksum; truncation (a page cut short by a crash mid-write) and corruption (a flipped bit in an already-written page) are distinguished and reported as typed exceptions, never a silent misread.
- Property-based tests with fixed-seed random insert/delete workloads that verify structural invariants (key order, correct range partitioning, leaf-chain completeness) after tens of thousands of operations.

## How it works

### Page format

Every page is exactly `PAGE_SIZE = 4096` bytes. All integers are big-endian:

```text
+----------+---------+-----------+-------------+--------------------+---------------+----------+
| MAGIC(4) | version | page_type | cell_count  | free_space_offset | side_pointer  | crc32(4) |
|  "BPT1"  |   (1)   |    (1)    |     (2)     |        (2)         |     (4)       |          |
+----------+---------+-----------+-------------+--------------------+---------------+----------+
|                     cell-pointer array (2 bytes per cell, sorted by key)                      |
|                                            ...                                                  |
|                        free space                                                              |
|                                            ...                                                  |
|                     cell content (grows from the end of the page backward)                    |
+-------------------------------------------------------------------------------------------------+
```

Page 0 is always a metadata page (`root_page_id`, `page_count`, `free_page_head`), never a tree node. `side_pointer` is overloaded by `page_type`: on a **leaf**, it is the `PageId` of the next leaf in key order (the range-scan chain); on an **internal** page, it is the rightmost child (the subtree for keys greater than or equal to the last separator). A leaf cell is `key_len(2) + key + value_len(2) + value`; an internal cell is `key_len(2) + key + child_page_id(4)`. The checksum covers the entire page, so any corrupted byte — header, pointer array, or cell content — is caught.

### Splitting

`MAX_ENTRY_SIZE` guarantees at least 4 entries always fit in an empty page, which in turn guarantees a valid split point always exists once a page overflows by exactly one new entry. The split point itself is chosen by scanning cumulative cell size (not entry count) so that both halves are guaranteed to fit in a page, even with adversarial size distributions. A leaf split copies the first key of the right half up as the separator (leaves hold all the data, so nothing is removed from them); an internal split **promotes** the median key to the parent instead, since internal nodes are pure routing and that key stops living at that level. Both cascade upward through the exact parent-pointer path recorded during descent, growing a new root when the cascade reaches the top.

### Merging

Deleting an entry does not enforce a strict minimum occupancy for correctness — a sparse B+Tree is still a perfectly valid, searchable one. As an optimization, a page that drops under half-full after a delete tries to merge into an immediate sibling under the same parent (left first, then right); if neither sibling has room, the page is simply left as-is. Redistribution (borrowing a single entry from a sibling instead of merging) is intentionally not implemented: it would cover a few more cases where a full merge doesn't fit, at the cost of a real extra edge case (a borrowed key could grow the parent's separator past what it had room for), for a benefit that is purely about compactness, not correctness.

## Architecture

```text
src/bplus_tree_storage_engine/
├── __init__.py     # re-exported public API
├── models.py       # page format constants, PageHeader/LeafEntry/InternalEntry/TreeMetadata, exceptions
├── protocols.py     # PageDevice, ReplacementPolicy: the two pluggable seams
├── pipeline.py       # binary codec, FilePageDevice/InMemoryPageDevice, BufferPool, BPlusTree
└── __main__.py       # demonstration CLI
```

- **`models.py`** — immutable data types (`dataclass(frozen=True, slots=True)`) and the exception hierarchy (`BplusTreeError` → `StorageIOError` / `PageCorruptionError` → `InvalidPageError` / `ChecksumMismatchError` / `TruncatedPageError`, plus `EntryTooLargeError`, `KeyNotFoundError`, `TreeInvariantError`).
- **`protocols.py`** — `PageDevice` (the raw page storage contract; `FilePageDevice` and `InMemoryPageDevice` both implement it) and `ReplacementPolicy` (the buffer pool's eviction contract; `LruReplacementPolicy` implements it).
- **`pipeline.py`** — everything else: the binary codec (`encode_leaf_page` / `decode_leaf_page` / ... ), the two `PageDevice` implementations, `BufferPool`, and `BPlusTree` (`search`, `insert`, `delete`, `scan`).
- **`__main__.py`** — a demonstration CLI (`put`, `get`, `delete`, `scan`, `info`, `benchmark`).

**Concurrency:** `BufferPool` and both `PageDevice` implementations assume a single thread of control — there is no internal locking because there is nothing to race against in this scope. A caller needing concurrent access (`nanosql`, for instance) must serialize it in its own layer before calling into this engine.

## Requirements and installation

- Python `>=3.11`

```bash
git clone https://github.com/juanmmm21/bplus-tree-storage-engine.git
cd bplus-tree-storage-engine
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"  # or: pip install -e . pytest mypy ruff
```

## Usage

### CLI

```bash
python -m bplus_tree_storage_engine put --path /tmp/demo.db --key alice --value 30
python -m bplus_tree_storage_engine put --path /tmp/demo.db --key bob --value 25
python -m bplus_tree_storage_engine put --path /tmp/demo.db --key carol --value 35

python -m bplus_tree_storage_engine get --path /tmp/demo.db --key alice
# 30

python -m bplus_tree_storage_engine scan --path /tmp/demo.db
# alice	30
# bob	25
# carol	35

python -m bplus_tree_storage_engine scan --path /tmp/demo.db --start bob --end carol
# bob	25

python -m bplus_tree_storage_engine delete --path /tmp/demo.db --key bob
python -m bplus_tree_storage_engine info --path /tmp/demo.db
# root_page_id=1
# page_count=2
# free_page_head=0
```

### Programmatic usage

```python
from bplus_tree_storage_engine import BPlusTree, BufferPool, BufferPoolConfig, FilePageDevice

with FilePageDevice("demo.db") as device:
    pool = BufferPool(device, BufferPoolConfig(capacity=64))
    with BPlusTree(pool) as tree:
        tree.insert(b"alice", b"30")
        tree.insert(b"bob", b"25")
        tree.insert(b"carol", b"35")

        assert tree.search(b"alice") == b"30"
        assert list(tree.scan(start=b"bob", end=b"carol")) == [(b"bob", b"25")]

        tree.delete(b"bob")
        assert tree.search(b"bob") is None
```

`InMemoryPageDevice` is a drop-in replacement for `FilePageDevice` when persistence to disk is not needed (tests, scratch indexes).

## Data format / interface exposed to `nanosql`

The binary page format (`MAGIC = b"BPT1"`, versioned per page) is the on-disk contract. Programmatically, `nanosql` (or any consumer) drives the tree through `BPlusTree`'s plain methods — `search`, `insert`, `delete`, `scan` — over a `BufferPool` built from any `PageDevice`. `PageDevice` and `ReplacementPolicy` are `Protocol`s specifically so `nanosql` can plug in a different backend or eviction strategy without touching this repo's internals.

## Development

```bash
pytest
ruff check .
ruff format --check .
mypy --strict src/
```

The test suite covers: binary serialization round-trips for leaf, internal and metadata pages; checksum, truncation, wrong-magic and wrong-page-type corruption detection; buffer pool LRU eviction with dirty-page write-back; file-device persistence across a real close/reopen cycle; insert/search/delete correctness including cascading splits and merges; and property-based tests with fixed-seed random workloads that verify key ordering, correct range partitioning between children, and leaf-chain completeness after thousands of interleaved operations.

## Benchmarks

Illustrative numbers from the `benchmark` CLI subcommand (pure Python, no C extensions, single machine — treat as a rough order of magnitude, not a guarantee):

```text
$ python -m bplus_tree_storage_engine benchmark --path /tmp/bench.db --count 50000 --capacity 256
writes: 50000 en 6.614s (7559 ops/s)
reads:  50000 en 9.278s (5389 ops/s)
```

Throughput is dominated by Python-level object churn (dataclasses, `struct.pack`/`unpack` per page) rather than by the algorithm itself; `vectorized-execution-engine`, which does have real performance pressure, uses NumPy for exactly that reason.

## Troubleshooting

- **`ChecksumMismatchError` when reading an existing database file:** real corruption of bytes already written (not a truncation). The page is unreadable from that point; nothing before it is affected, since a page's checksum only covers that page.
- **`TruncatedPageError` on open or on `read_page`:** the file is shorter than the requested page boundary, or its size is not an exact multiple of `PAGE_SIZE` — typically a process killed mid-`write_page`/`allocate_page`.
- **`EntryTooLargeError` on `insert`:** the key or value (or both) exceed `MAX_ENTRY_SIZE` / `MAX_KEY_SIZE`, the bounds that guarantee at least 4 entries always fit in an empty page and that a valid split point always exists.
- **A deleted key still occupies disk space:** by design — see "Merging" above. A page below half-full only shrinks the tree if a sibling under the same parent has room to absorb it.

## License

MIT — see [`LICENSE`](./LICENSE).
