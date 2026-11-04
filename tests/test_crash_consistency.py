"""Modo `crash_consistent`: el archivo sólo pasa de un flush completo al
siguiente, se corte donde se corte.

Un crash se simula lanzando una excepción desde `write_page` tras la
k-ésima escritura en su sitio y reabriendo el archivo con objetos nuevos.
El dispositivo viejo se conserva sin cerrar (su búfer de usuario no llega
al sistema operativo, como con `kill -9`) o se vuelca antes de reabrir
(todas las escrituras emitidas llegaron): ambos extremos deben dar el
mismo resultado.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from bplus_tree_storage_engine import (
    BPlusTree,
    BufferPool,
    BufferPoolConfig,
    FilePageDevice,
    InMemoryPageDevice,
)
from bplus_tree_storage_engine.models import PAGE_SIZE, PageId


class SimulatedCrashError(Exception):
    pass


class CrashingDevice(FilePageDevice):
    """Muere (lanza) en la escritura en su sitio número `crash_after + 1`."""

    def __init__(self, path: Path, crash_after: int | None = None) -> None:
        super().__init__(path)
        self.crash_after = crash_after
        self.writes = 0

    def write_page(self, page_id: PageId, data: bytes) -> None:
        if self.crash_after is not None and self.writes >= self.crash_after:
            raise SimulatedCrashError(f"crash before in-place write #{self.writes + 1}")
        self.writes += 1
        super().write_page(page_id, data)


def _value(key: int, generation: int) -> bytes:
    return f"value-{key}-gen{generation}".encode() * 8


def _open(path: Path, device: FilePageDevice | None = None) -> BPlusTree:
    pool = BufferPool(device or FilePageDevice(path), BufferPoolConfig(8, crash_consistent=True))
    return BPlusTree(pool)


def _contents(path: Path) -> dict[bytes, bytes]:
    tree = _open(path)
    try:
        return dict(tree.scan())
    finally:
        tree.close()


def _state_a(path: Path) -> dict[bytes, bytes]:
    tree = _open(path)
    for key in range(0, 120, 2):
        tree.insert(f"k{key:04d}".encode(), _value(key, 0))
    tree.close()
    return _contents(path)


def _mutate(tree: BPlusTree) -> None:
    # splits (nuevas claves intercaladas), sobrescrituras y borrados
    for key in range(1, 240, 2):
        tree.insert(f"k{key:04d}".encode(), _value(key, 1))
    for key in range(0, 60, 2):
        tree.insert(f"k{key:04d}".encode(), _value(key, 1))
    for key in range(60, 120, 4):
        tree.delete(f"k{key:04d}".encode())


def _expected_b(path: Path, tmp_path: Path) -> dict[bytes, bytes]:
    reference = tmp_path / "reference.db"
    reference.write_bytes(path.read_bytes())
    tree = _open(reference)
    _mutate(tree)
    tree.close()
    return _contents(reference)


def _in_place_writes_of_flush(path: Path, tmp_path: Path) -> int:
    probe = tmp_path / "probe.db"
    probe.write_bytes(path.read_bytes())
    device = CrashingDevice(probe)
    tree = _open(probe, device)
    _mutate(tree)
    before = device.writes
    tree.flush()
    count = device.writes - before
    tree.close()
    return count


@pytest.mark.parametrize("buffered_writes_reach_disk", [False, True])
def test_crash_at_every_in_place_write_of_a_flush(
    tmp_path: Path, buffered_writes_reach_disk: bool
) -> None:
    base = tmp_path / "base.db"
    state_a = _state_a(base)
    state_b = _expected_b(base, tmp_path)
    total = _in_place_writes_of_flush(base, tmp_path)
    assert total > 5, "the workload must make a multi-page flush"
    assert state_a != state_b

    for crash_after in range(total):
        path = tmp_path / f"crash-{crash_after}.db"
        path.write_bytes(base.read_bytes())
        device = CrashingDevice(path)
        tree = _open(path, device)
        _mutate(tree)
        device.crash_after = device.writes + crash_after
        with pytest.raises(SimulatedCrashError):
            tree.flush()
        if buffered_writes_reach_disk:
            device.sync()
        # el diario estaba completo: al reabrir se re-aplica el lote entero
        assert _contents(path) == state_b, f"crash after {crash_after} in-place writes"
        assert not path.with_name(path.name + ".journal").exists()


def test_without_flush_nothing_reaches_the_file(tmp_path: Path) -> None:
    """No-steal: con un pool diminuto, mutar sin flush no escribe nada en el
    archivo, así que un crash ahí deja exactamente el último flush."""
    path = tmp_path / "db"
    state_a = _state_a(path)
    device = CrashingDevice(path, crash_after=0)
    tree = _open(path, device)
    _mutate(tree)  # lanzaría SimulatedCrashError si desalojase una página sucia
    assert device.writes == 0
    assert _contents(path) == state_a


def _journal(pages: list[tuple[int, bytes]]) -> bytes:
    body = struct.pack(">4sII", b"BPTJ", 1, len(pages)) + b"".join(
        struct.pack(">I", page_id) + data for page_id, data in pages
    )
    return body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def test_torn_journal_is_discarded(tmp_path: Path) -> None:
    """Un diario cortado es un crash antes de su fsync: ninguna escritura en
    su sitio empezó, así que se descarta y queda el último flush."""
    path = tmp_path / "db"
    state_a = _state_a(path)
    journal = path.with_name(path.name + ".journal")
    full = _journal([(1, bytes([7]) * PAGE_SIZE)])
    for cut in (0, 5, len(full) // 2, len(full) - 1):
        journal.write_bytes(full[:cut])
        assert _contents(path) == state_a
        assert not journal.exists()


def test_complete_journal_is_replayed_on_open(tmp_path: Path) -> None:
    path = tmp_path / "db"
    _state_a(path)
    device = FilePageDevice(path)
    count = device.page_count()
    device.close()
    marker = bytes([9]) * PAGE_SIZE
    path.with_name(path.name + ".journal").write_bytes(_journal([(count - 1, marker)]))
    device = FilePageDevice(path)
    assert device.read_page(PageId(count - 1)) == marker
    device.close()


def test_pool_grows_while_dirty_and_shrinks_after_flush(tmp_path: Path) -> None:
    pool = BufferPool(FilePageDevice(tmp_path / "db"), BufferPoolConfig(4, crash_consistent=True))
    pages = [pool.allocate_page() for _ in range(10)]
    for page_id in pages:
        pool.put_page(page_id, bytes([page_id % 256]) * PAGE_SIZE)
    assert len(pool._cache) == 10  # todas sucias: nada desalojable
    pool.flush()
    assert len(pool._cache) <= 4
    for page_id in pages:
        assert pool.get_page(page_id) == bytes([page_id % 256]) * PAGE_SIZE
    pool.close()


def test_memory_device_falls_back_to_plain_flush() -> None:
    pool = BufferPool(InMemoryPageDevice(), BufferPoolConfig(2, crash_consistent=True))
    tree = BPlusTree(pool)
    for key in range(50):
        tree.insert(f"k{key:03d}".encode(), b"v" * 300)
    tree.flush()
    assert len(list(tree.scan())) == 50


def test_file_killed_before_its_first_flush_opens_as_an_empty_tree(tmp_path: Path) -> None:
    """`allocate_page` escribe ceros: si el proceso muere antes del primer
    flush, las páginas 0 y 1 existen pero están a cero."""
    path = tmp_path / "db"
    path.write_bytes(bytes(2 * PAGE_SIZE))
    tree = _open(path)
    assert list(tree.scan()) == []
    tree.insert(b"a", b"1")
    tree.close()
    assert _contents(path) == {b"a": b"1"}
