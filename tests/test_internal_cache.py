"""Caché de páginas internas decodificadas de `BPlusTree` (sólo para los
descensos) y metadatos que no se reescriben si no cambian: mismo árbol,
menos trabajo."""

from __future__ import annotations

import random

import pytest

from bplus_tree_storage_engine import (
    BPlusTree,
    BufferPool,
    BufferPoolConfig,
    InMemoryPageDevice,
    pipeline,
)
from bplus_tree_storage_engine.models import METADATA_PAGE_ID


def _tree(capacity: int = 32) -> BPlusTree:
    return BPlusTree(BufferPool(InMemoryPageDevice(), BufferPoolConfig(capacity)))


@pytest.mark.parametrize("cache_pages", [1, 2, pipeline.INTERNAL_CACHE_PAGES])
def test_random_workload_matches_a_dict(monkeypatch: pytest.MonkeyPatch, cache_pages: int) -> None:
    """Inserciones, sobrescrituras y borrados que provocan splits y fusiones
    en cascada; con una caché diminuta también hay desalojos constantes."""
    monkeypatch.setattr(pipeline, "INTERNAL_CACHE_PAGES", cache_pages)
    rng = random.Random(1234 + cache_pages)
    tree = _tree()
    model: dict[bytes, bytes] = {}
    for step in range(6000):
        key = f"key-{rng.randrange(1500):05d}".encode()
        if rng.random() < 0.3 and model:
            victim = rng.choice(sorted(model))
            tree.delete(victim)
            del model[victim]
        else:
            value = bytes([rng.randrange(256)]) * rng.randrange(20, 300)
            tree.insert(key, value)
            model[key] = value
        if step % 500 == 499:
            assert list(tree.scan()) == sorted(model.items())
            for probe in rng.sample(sorted(model), min(50, len(model))):
                assert tree.search(probe) == model[probe]
    assert list(tree.scan()) == sorted(model.items())
    assert tree.search(b"absent") is None


def test_inserts_stop_decoding_internal_pages_once_warm(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = _tree()
    for i in range(3000):
        tree.insert(f"k{i:06d}".encode(), b"v" * 200)
    decodes = 0
    original = pipeline.decode_internal_page

    def counting(data: bytes) -> pipeline.InternalPage:
        nonlocal decodes
        decodes += 1
        return original(data)

    monkeypatch.setattr(pipeline, "decode_internal_page", counting)
    for i in range(3000, 4000):
        tree.insert(f"k{i:06d}".encode(), b"v" * 200)
    # sin caché serían >= 1000 * (altura - 1); ahora sólo tras un split
    assert decodes < 200, decodes
    assert tree.search(b"k003999") == b"v" * 200


def test_metadata_page_is_written_only_when_it_changes() -> None:
    tree = _tree()
    tree.insert(b"a", b"1")
    writes: list[int] = []
    original = tree._pool.put_page

    def recording(page_id: int, data: bytes) -> None:
        writes.append(page_id)
        original(page_id, data)

    tree._pool.put_page = recording  # type: ignore[method-assign]
    for i in range(20):  # caben en la hoja raíz: ni splits ni páginas nuevas
        tree.insert(f"b{i}".encode(), b"x")
    assert METADATA_PAGE_ID not in writes
    assert tree.search(b"b19") == b"x"
