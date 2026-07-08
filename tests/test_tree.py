"""Tests funcionales del árbol B+: insert/search/delete, splits, merges y persistencia."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from bplus_tree_storage_engine import (
    BPlusTree,
    BufferPool,
    BufferPoolConfig,
    EntryTooLargeError,
    FilePageDevice,
    InMemoryPageDevice,
    KeyNotFoundError,
)
from bplus_tree_storage_engine.models import MAX_ENTRY_SIZE, NULL_PAGE_ID, PageType
from bplus_tree_storage_engine.pipeline import (
    _peek_page_type,
    decode_internal_page,
    decode_leaf_page,
)


def _new_tree(capacity: int = 8) -> BPlusTree:
    return BPlusTree(BufferPool(InMemoryPageDevice(), BufferPoolConfig(capacity=capacity)))


def _leftmost_leaf_id(tree: BPlusTree) -> int:
    page_id = tree._metadata.root_page_id  # noqa: SLF001 (inspección interna deliberada del test)
    while True:
        raw = tree._pool.get_page(page_id)  # noqa: SLF001
        if _peek_page_type(raw) is PageType.LEAF:
            return page_id
        internal = decode_internal_page(raw)
        page_id = internal.entries[0].left_child


def _leaf_chain_keys(tree: BPlusTree) -> list[bytes]:
    page_id = _leftmost_leaf_id(tree)
    collected: list[bytes] = []
    while page_id != NULL_PAGE_ID:
        leaf = decode_leaf_page(tree._pool.get_page(page_id))  # noqa: SLF001
        collected.extend(entry.key for entry in leaf.entries)
        page_id = leaf.side_pointer
    return collected


def test_insert_and_search_basic() -> None:
    tree = _new_tree()
    tree.insert(b"a", b"1")
    tree.insert(b"b", b"2")

    assert tree.search(b"a") == b"1"
    assert tree.search(b"b") == b"2"
    assert tree.search(b"missing") is None


def test_upsert_overwrites_existing_value() -> None:
    tree = _new_tree()
    tree.insert(b"a", b"1")
    tree.insert(b"a", b"2")

    assert tree.search(b"a") == b"2"


def test_delete_removes_key() -> None:
    tree = _new_tree()
    tree.insert(b"a", b"1")
    tree.delete(b"a")

    assert tree.search(b"a") is None


def test_delete_missing_key_raises() -> None:
    tree = _new_tree()
    with pytest.raises(KeyNotFoundError):
        tree.delete(b"missing")


def test_delete_then_reinsert_works() -> None:
    tree = _new_tree()
    tree.insert(b"a", b"1")
    tree.delete(b"a")
    tree.insert(b"a", b"2")

    assert tree.search(b"a") == b"2"


def test_entry_too_large_raises_before_touching_storage() -> None:
    tree = _new_tree()
    oversized_value = b"x" * (MAX_ENTRY_SIZE + 1)

    with pytest.raises(EntryTooLargeError):
        tree.insert(b"k", oversized_value)

    assert tree.search(b"k") is None


def test_empty_key_rejected() -> None:
    tree = _new_tree()
    with pytest.raises(ValueError, match="clave"):
        tree.insert(b"", b"v")


def test_many_inserts_trigger_splits_and_stay_searchable() -> None:
    tree = _new_tree(capacity=4)
    keys = [f"key-{i:05d}".encode() for i in range(2000)]
    shuffled = keys[:]
    random.Random(0).shuffle(shuffled)

    for i, key in enumerate(shuffled):
        tree.insert(key, str(i).encode())

    for key in keys:
        assert tree.search(key) is not None


def test_leaf_chain_stays_sorted_and_complete_after_many_inserts() -> None:
    tree = _new_tree(capacity=4)
    keys = [f"key-{i:05d}".encode() for i in range(1500)]
    shuffled = keys[:]
    random.Random(1).shuffle(shuffled)

    for key in shuffled:
        tree.insert(key, b"v")

    chain = _leaf_chain_keys(tree)
    assert chain == sorted(keys)
    assert len(chain) == len(set(chain))


def test_leaf_chain_reflects_deletions() -> None:
    tree = _new_tree(capacity=4)
    keys = [f"key-{i:05d}".encode() for i in range(600)]
    shuffled = keys[:]
    random.Random(2).shuffle(shuffled)
    for key in shuffled:
        tree.insert(key, b"v")

    to_delete = keys[::3]
    for key in to_delete:
        tree.delete(key)

    survivors = sorted(set(keys) - set(to_delete))
    assert _leaf_chain_keys(tree) == survivors


def test_tree_reusable_after_being_fully_emptied() -> None:
    tree = _new_tree(capacity=4)
    keys = [f"k{i:04d}".encode() for i in range(300)]
    for key in keys:
        tree.insert(key, b"v")
    for key in keys:
        tree.delete(key)

    for key in keys:
        assert tree.search(key) is None

    tree.insert(b"fresh", b"data")
    assert tree.search(b"fresh") == b"data"


def test_data_survives_close_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "tree.db"
    keys = [f"k{i:04d}".encode() for i in range(500)]
    shuffled = keys[:]
    random.Random(3).shuffle(shuffled)

    with FilePageDevice(path) as device:
        pool = BufferPool(device, BufferPoolConfig(capacity=16))
        tree = BPlusTree(pool)
        for i, key in enumerate(shuffled):
            tree.insert(key, str(i).encode())
        tree.flush()

    with FilePageDevice(path) as device:
        pool = BufferPool(device, BufferPoolConfig(capacity=16))
        tree = BPlusTree(pool)
        for key in keys:
            assert tree.search(key) is not None


def test_scan_full_range_yields_sorted_pairs() -> None:
    tree = _new_tree(capacity=4)
    keys = [f"key-{i:05d}".encode() for i in range(500)]
    shuffled = keys[:]
    random.Random(4).shuffle(shuffled)
    for key in shuffled:
        tree.insert(key, key.upper())

    result = list(tree.scan())
    assert [k for k, _ in result] == sorted(keys)
    assert all(v == k.upper() for k, v in result)


def test_scan_bounded_range_is_half_open() -> None:
    tree = _new_tree(capacity=4)
    keys = [f"key-{i:05d}".encode() for i in range(200)]
    for key in keys:
        tree.insert(key, b"v")

    start, end = keys[50], keys[80]
    result = [k for k, _ in tree.scan(start=start, end=end)]
    assert result == keys[50:80]


def test_scan_with_only_start_bound() -> None:
    tree = _new_tree(capacity=4)
    keys = [f"key-{i:05d}".encode() for i in range(100)]
    for key in keys:
        tree.insert(key, b"v")

    result = [k for k, _ in tree.scan(start=keys[70])]
    assert result == keys[70:]


def test_random_insert_delete_matches_reference_dict() -> None:
    rng = random.Random(99)
    tree = _new_tree(capacity=5)
    reference: dict[bytes, bytes] = {}
    universe = [f"u-{i:04d}".encode() for i in range(500)]

    for step in range(6000):
        key = rng.choice(universe)
        if rng.random() < 0.6:
            value = str(step).encode()
            tree.insert(key, value)
            reference[key] = value
        elif key in reference:
            tree.delete(key)
            del reference[key]
        else:
            with pytest.raises(KeyNotFoundError):
                tree.delete(key)

    for key in universe:
        assert tree.search(key) == reference.get(key)

    assert _leaf_chain_keys(tree) == sorted(reference.keys())
