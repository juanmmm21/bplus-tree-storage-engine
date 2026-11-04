"""Equivalencia exacta del códec nativo (Rust) con el de referencia.

Con `bplus-tree-storage-engine-native` instalado, `decode_leaf_page` /
`decode_internal_page` usan Rust para las páginas válidas; estas
propiedades comprueban, sobre páginas generadas por Hypothesis, que:

* sobre páginas válidas, el resultado nativo es idéntico al de Python;
* sobre páginas mutadas o basura, el nativo devuelve `None` (y entonces se
  usa Python) o exactamente el resultado de Python — nunca algo distinto —,
  y el decodificador público produce el mismo resultado o el mismo error
  que el de referencia.

Sin el paquete nativo, los tests de equivalencia se saltan.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bplus_tree_storage_engine import pipeline
from bplus_tree_storage_engine.models import PAGE_SIZE, InternalEntry, LeafEntry
from bplus_tree_storage_engine.pipeline import (
    _decode_internal_page_python,
    _decode_leaf_page_python,
    decode_internal_page,
    decode_leaf_page,
    encode_internal_page,
    encode_leaf_page,
)

native = pipeline._native
needs_native = pytest.mark.skipif(native is None, reason="paquete nativo no instalado")

keys = st.binary(min_size=1, max_size=40)
leaf_entries = st.lists(
    st.tuples(keys, st.binary(max_size=120)), max_size=24, unique_by=lambda kv: kv[0]
).map(lambda pairs: [LeafEntry(k, v) for k, v in sorted(pairs)])
internal_entries = st.lists(
    st.tuples(keys, st.integers(min_value=0, max_value=2**32 - 1)),
    max_size=40,
    unique_by=lambda kv: kv[0],
).map(lambda pairs: [InternalEntry(k, c) for k, c in sorted(pairs)])
pointers = st.integers(min_value=0, max_value=2**32 - 1)


def _outcome(decode: object, data: bytes) -> tuple[str, object]:
    try:
        return "ok", decode(data)  # type: ignore[operator]
    except Exception as exc:  # noqa: BLE001 — comparamos tipo y mensaje del error
        return type(exc).__name__, str(exc)


@needs_native
@settings(max_examples=400, deadline=None)
@given(leaf_entries, pointers)
def test_valid_leaf_pages_decode_identically(entries: list[LeafEntry], next_leaf: int) -> None:
    page = encode_leaf_page(entries, next_leaf)
    assert native.decode_leaf_cells(page) is not None
    assert decode_leaf_page(page) == _decode_leaf_page_python(page)


@needs_native
@settings(max_examples=400, deadline=None)
@given(internal_entries, pointers)
def test_valid_internal_pages_decode_identically(
    entries: list[InternalEntry], rightmost: int
) -> None:
    page = encode_internal_page(entries, rightmost)
    assert native.decode_internal_cells(page) is not None
    assert decode_internal_page(page) == _decode_internal_page_python(page)


@needs_native
@settings(max_examples=600, deadline=None)
@given(
    leaf_entries,
    pointers,
    st.lists(st.tuples(st.integers(0, PAGE_SIZE - 1), st.integers(0, 255)), min_size=1, max_size=6),
    st.booleans(),
)
def test_mutated_pages_never_diverge(
    entries: list[LeafEntry], pointer: int, mutations: list[tuple[int, int]], internal: bool
) -> None:
    if internal:
        page = bytearray(
            encode_internal_page([InternalEntry(e.key, len(e.value)) for e in entries], pointer)
        )
    else:
        page = bytearray(encode_leaf_page(entries, pointer))
    for offset, value in mutations:
        page[offset] = value
    data = bytes(page)
    public, reference = (
        (decode_internal_page, _decode_internal_page_python)
        if internal
        else (decode_leaf_page, _decode_leaf_page_python)
    )
    assert _outcome(public, data) == _outcome(reference, data)
    raw = native.decode_internal_cells(data) if internal else native.decode_leaf_cells(data)
    if raw is not None:  # si el nativo acepta la página, debe coincidir con Python
        side, pairs = raw
        expected = reference(data)
        assert side == expected.side_pointer
        assert pairs == [
            (e.key, e.left_child if internal else e.value)  # type: ignore[union-attr]
            for e in expected.entries
        ]


@needs_native
@settings(max_examples=300, deadline=None)
@given(st.binary(min_size=0, max_size=PAGE_SIZE + 8))
def test_garbage_is_rejected_by_both(data: bytes) -> None:
    assert _outcome(decode_leaf_page, data) == _outcome(_decode_leaf_page_python, data)
    assert _outcome(decode_internal_page, data) == _outcome(_decode_internal_page_python, data)


def test_pure_python_fallback_is_always_available() -> None:
    page = encode_leaf_page([LeafEntry(b"k", b"v")])
    assert _decode_leaf_page_python(page).entries == [LeafEntry(b"k", b"v")]
