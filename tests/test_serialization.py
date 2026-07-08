"""Tests del codec binario de páginas: round-trip, checksum y truncamiento."""

from __future__ import annotations

import pytest

from bplus_tree_storage_engine.models import (
    HEADER_SIZE,
    PAGE_SIZE,
    ChecksumMismatchError,
    InternalEntry,
    InvalidPageError,
    LeafEntry,
    PageType,
    TreeMetadata,
    TruncatedPageError,
)
from bplus_tree_storage_engine.pipeline import (
    decode_internal_page,
    decode_leaf_page,
    decode_metadata_page,
    encode_internal_page,
    encode_leaf_page,
    encode_metadata_page,
    internal_cell_size,
    leaf_cell_size,
    page_free_space,
)


def test_leaf_page_roundtrip() -> None:
    entries = [LeafEntry(key=f"k{i:03d}".encode(), value=f"v{i}".encode() * 3) for i in range(10)]
    data = encode_leaf_page(entries, next_leaf=7)

    assert len(data) == PAGE_SIZE
    page = decode_leaf_page(data)
    assert page.side_pointer == 7
    assert page.entries == entries


def test_leaf_page_empty_roundtrip() -> None:
    data = encode_leaf_page([], next_leaf=0)
    page = decode_leaf_page(data)
    assert page.entries == []
    assert page.side_pointer == 0


def test_internal_page_roundtrip() -> None:
    entries = [InternalEntry(key=f"k{i:03d}".encode(), left_child=i + 1) for i in range(6)]
    data = encode_internal_page(entries, rightmost_child=99)

    assert len(data) == PAGE_SIZE
    page = decode_internal_page(data)
    assert page.side_pointer == 99
    assert page.entries == entries


def test_metadata_page_roundtrip() -> None:
    metadata = TreeMetadata(root_page_id=3, page_count=12, free_page_head=5)
    data = encode_metadata_page(metadata)

    assert len(data) == PAGE_SIZE
    assert decode_metadata_page(data) == metadata


def test_checksum_mismatch_detected() -> None:
    entries = [LeafEntry(key=b"k", value=b"v")]
    data = bytearray(encode_leaf_page(entries))
    data[100] ^= 0xFF  # corrompe un byte dentro del contenido de la celda

    with pytest.raises(ChecksumMismatchError):
        decode_leaf_page(bytes(data))


def test_truncated_page_detected_on_decode() -> None:
    data = encode_leaf_page([LeafEntry(key=b"k", value=b"v")])

    with pytest.raises(TruncatedPageError):
        decode_leaf_page(data[: HEADER_SIZE - 1])


def test_invalid_magic_detected() -> None:
    data = bytearray(encode_leaf_page([LeafEntry(key=b"k", value=b"v")]))
    data[0:4] = b"XXXX"

    with pytest.raises(InvalidPageError):
        decode_leaf_page(bytes(data))


def test_wrong_page_type_detected() -> None:
    internal_data = encode_internal_page([InternalEntry(key=b"k", left_child=1)], rightmost_child=2)

    with pytest.raises(InvalidPageError):
        decode_leaf_page(internal_data)


def test_wrong_page_type_detected_for_metadata() -> None:
    leaf_data = encode_leaf_page([LeafEntry(key=b"k", value=b"v")])

    with pytest.raises(InvalidPageError):
        decode_metadata_page(leaf_data)


def test_leaf_page_preserves_key_order_regardless_of_insertion_order_in_encode() -> None:
    # el codec no reordena: el orden de salida es exactamente el de entrada.
    entries = [LeafEntry(key=b"b", value=b"2"), LeafEntry(key=b"a", value=b"1")]
    page = decode_leaf_page(encode_leaf_page(entries))
    assert [e.key for e in page.entries] == [b"b", b"a"]


def test_leaf_cell_size_matches_serialized_layout() -> None:
    key, value = b"abcdef", b"0123456789"
    size = leaf_cell_size(key, value)
    # 2 bytes de longitud de clave + clave + 2 bytes de longitud de valor + valor
    assert size == 2 + len(key) + 2 + len(value)


def test_internal_cell_size_matches_serialized_layout() -> None:
    key = b"abcdef"
    size = internal_cell_size(key)
    # 2 bytes de longitud de clave + clave + 4 bytes de PageId
    assert size == 2 + len(key) + 4


def test_page_free_space_accounts_for_header_and_pointer_array() -> None:
    # página completamente vacía: todo el espacio utilizable está libre
    assert page_free_space(cell_count=0, free_space_offset=PAGE_SIZE) == PAGE_SIZE - HEADER_SIZE


def test_leaf_page_type_is_leaf() -> None:
    data = encode_leaf_page([])
    assert decode_leaf_page(data) is not None
    header_page_type = data[5]
    assert header_page_type == int(PageType.LEAF)
