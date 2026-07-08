"""Tests del dispositivo de páginas (archivo y memoria) y del buffer pool."""

from __future__ import annotations

from pathlib import Path

import pytest

from bplus_tree_storage_engine.models import PAGE_SIZE, BufferPoolConfig, TruncatedPageError
from bplus_tree_storage_engine.pipeline import BufferPool, FilePageDevice, InMemoryPageDevice


def test_in_memory_device_allocate_write_read() -> None:
    device = InMemoryPageDevice()
    first = device.allocate_page()
    second = device.allocate_page()

    assert (first, second) == (0, 1)
    assert device.page_count() == 2

    device.write_page(first, b"a" * PAGE_SIZE)
    assert device.read_page(first) == b"a" * PAGE_SIZE


def test_in_memory_device_rejects_wrong_size_write() -> None:
    device = InMemoryPageDevice()
    page_id = device.allocate_page()

    with pytest.raises(ValueError, match="4096 bytes"):
        device.write_page(page_id, b"short")


def test_in_memory_device_rejects_read_out_of_range() -> None:
    device = InMemoryPageDevice()

    with pytest.raises(TruncatedPageError):
        device.read_page(0)


def test_file_device_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "pages.db"

    with FilePageDevice(path) as device:
        page_id = device.allocate_page()
        device.write_page(page_id, b"x" * PAGE_SIZE)
        device.sync()

    with FilePageDevice(path) as reopened:
        assert reopened.page_count() == 1
        assert reopened.read_page(page_id) == b"x" * PAGE_SIZE


def test_file_device_detects_truncated_file_on_open(tmp_path: Path) -> None:
    path = tmp_path / "pages.db"
    with FilePageDevice(path) as device:
        device.allocate_page()

    with open(path, "r+b") as raw:
        raw.truncate(PAGE_SIZE - 10)

    with pytest.raises(TruncatedPageError):
        FilePageDevice(path)


def test_file_device_starts_empty_for_new_path(tmp_path: Path) -> None:
    path = tmp_path / "new.db"
    with FilePageDevice(path) as device:
        assert device.page_count() == 0


def test_buffer_pool_reads_through_to_device() -> None:
    device = InMemoryPageDevice()
    page_id = device.allocate_page()
    device.write_page(page_id, b"z" * PAGE_SIZE)

    pool = BufferPool(device, BufferPoolConfig(capacity=4))
    assert pool.get_page(page_id) == b"z" * PAGE_SIZE


def test_buffer_pool_never_exceeds_capacity() -> None:
    device = InMemoryPageDevice()
    ids = [device.allocate_page() for _ in range(10)]
    pool = BufferPool(device, BufferPoolConfig(capacity=3))

    for page_id in ids:
        pool.put_page(page_id, bytes([page_id]) * PAGE_SIZE)
        assert len(pool._cache) <= 3  # noqa: SLF001 (inspección interna deliberada del test)


def test_buffer_pool_lru_evicts_least_recently_used_and_flushes_dirty_pages() -> None:
    device = InMemoryPageDevice()
    ids = [device.allocate_page() for _ in range(3)]
    pool = BufferPool(device, BufferPoolConfig(capacity=2))

    pool.put_page(ids[0], b"0" * PAGE_SIZE)
    pool.put_page(ids[1], b"1" * PAGE_SIZE)
    pool.get_page(ids[0])  # ids[0] pasa a ser más reciente que ids[1]
    pool.put_page(ids[2], b"2" * PAGE_SIZE)  # debe desalojar ids[1], no ids[0]

    assert pool.get_page(ids[0]) == b"0" * PAGE_SIZE
    assert pool.get_page(ids[2]) == b"2" * PAGE_SIZE
    # ids[1] fue desalojado pero su contenido sucio se persistió en el device
    assert device.read_page(ids[1]) == b"1" * PAGE_SIZE


def test_buffer_pool_flush_persists_all_dirty_pages_and_syncs() -> None:
    device = InMemoryPageDevice()
    ids = [device.allocate_page() for _ in range(2)]
    pool = BufferPool(device, BufferPoolConfig(capacity=4))

    pool.put_page(ids[0], b"a" * PAGE_SIZE)
    pool.put_page(ids[1], b"b" * PAGE_SIZE)
    pool.flush()

    assert device.read_page(ids[0]) == b"a" * PAGE_SIZE
    assert device.read_page(ids[1]) == b"b" * PAGE_SIZE
