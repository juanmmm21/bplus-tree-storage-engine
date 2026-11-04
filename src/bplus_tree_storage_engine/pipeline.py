"""Lógica core del motor: codec de páginas, dispositivo de páginas y
buffer pool.

El codec convierte listas de entradas decodificadas (`LeafEntry` /
`InternalEntry`) o metadatos (`TreeMetadata`) en los `PAGE_SIZE` bytes
exactos que se persisten en disco, y viceversa, detectando corrupción y
truncamiento al leer. `FilePageDevice` es la implementación por defecto
de `PageDevice` (respaldada por un archivo real); `InMemoryPageDevice` es
una alternativa para tests que no toca el sistema de archivos.
`BufferPool` combina un `PageDevice` con una `ReplacementPolicy` (LRU por
defecto) para acotar cuántas páginas viven en memoria a la vez.

Concurrencia: tanto `BufferPool` como los dispositivos de este módulo
asumen un único hilo de control (el motor no lanza tareas concurrentes
internamente); no hay locking porque no hay nada con lo que competir en
este alcance educativo. Si `nanosql` necesita acceso concurrente deberá
serializarlo en su propia capa antes de llamar a estas clases.
"""

from __future__ import annotations

import bisect
import importlib
import io
import os
import struct
import zlib
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from bplus_tree_storage_engine.models import (
    CELL_POINTER_FORMAT,
    CELL_POINTER_SIZE,
    HEADER_FORMAT,
    HEADER_SIZE,
    MAGIC,
    MAX_ENTRY_SIZE,
    MAX_KEY_SIZE,
    METADATA_PAGE_ID,
    METADATA_PAYLOAD_FORMAT,
    METADATA_PAYLOAD_SIZE,
    NULL_PAGE_ID,
    PAGE_FORMAT_VERSION,
    PAGE_SIZE,
    BufferPoolConfig,
    ChecksumMismatchError,
    EntryTooLargeError,
    InternalEntry,
    InvalidPageError,
    KeyNotFoundError,
    LeafEntry,
    PageHeader,
    PageId,
    PageType,
    StorageIOError,
    TreeInvariantError,
    TreeMetadata,
    TruncatedPageError,
)
from bplus_tree_storage_engine.protocols import AtomicPageDevice, PageDevice, ReplacementPolicy

_CHECKSUM_SIZE = struct.calcsize(">I")
_CHECKSUM_OFFSET = HEADER_SIZE - _CHECKSUM_SIZE
_PAGE_TYPE_OFFSET = struct.calcsize(">4sB")
_KEY_LENGTH_FORMAT = ">H"
_KEY_LENGTH_SIZE = struct.calcsize(_KEY_LENGTH_FORMAT)
_VALUE_LENGTH_FORMAT = ">H"
_VALUE_LENGTH_SIZE = struct.calcsize(_VALUE_LENGTH_FORMAT)
_CHILD_ID_FORMAT = ">I"
_CHILD_ID_SIZE = struct.calcsize(_CHILD_ID_FORMAT)

_UNDERFLOW_THRESHOLD = PAGE_SIZE // 2
"""Ocupación mínima (en bytes serializados) por debajo de la cual una
página no raíz intenta fusionarse con un hermano al borrar una entrada.
No es un invariante de corrección (el árbol es válido igualmente por
debajo de este umbral) sino una heurística de compacidad: ver el
docstring de `BPlusTree.delete`."""


@dataclass(slots=True)
class LeafPage:
    """Página hoja decodificada: entradas ordenadas por clave ascendente."""

    side_pointer: PageId
    entries: list[LeafEntry] = field(default_factory=list)


@dataclass(slots=True)
class InternalPage:
    """Página interna decodificada: `entries` ordenadas por clave
    ascendente; `side_pointer` es el hijo más a la derecha (ver
    `InternalEntry`)."""

    side_pointer: PageId
    entries: list[InternalEntry] = field(default_factory=list)


def _checksum(buffer: bytearray | bytes) -> int:
    return zlib.crc32(bytes(buffer)) & 0xFFFFFFFF


def leaf_cell_size(key: bytes, value: bytes) -> int:
    """Bytes que ocupará la celda ya serializada de esta entrada de hoja."""
    return _KEY_LENGTH_SIZE + len(key) + _VALUE_LENGTH_SIZE + len(value)


def internal_cell_size(key: bytes) -> int:
    """Bytes que ocupará la celda ya serializada de esta entrada interna."""
    return _KEY_LENGTH_SIZE + len(key) + _CHILD_ID_SIZE


def page_free_space(cell_count: int, free_space_offset: int) -> int:
    """Espacio, en bytes, disponible para una celda nueva (más su puntero)
    dado el estado actual de una página hoja o interna."""
    pointer_array_end = HEADER_SIZE + cell_count * CELL_POINTER_SIZE
    return free_space_offset - pointer_array_end


def _assemble_page(page_type: PageType, cells: list[bytes], side_pointer: PageId) -> bytes:
    """Ensambla una página hoja o interna a partir de celdas ya
    serializadas, en el orden en que deben quedar accesibles (ascendente
    por clave). Lanza `TreeInvariantError` si no caben: el llamante debe
    comprobar `page_free_space` / decidir un split antes de llegar aquí."""
    cell_count = len(cells)
    pointer_array_end = HEADER_SIZE + cell_count * CELL_POINTER_SIZE
    buffer = bytearray(PAGE_SIZE)
    offsets: list[int] = []
    offset = PAGE_SIZE
    for cell in cells:
        offset -= len(cell)
        if offset < pointer_array_end:
            raise TreeInvariantError(
                "las celdas no caben en la página: el llamante debía haber hecho split antes"
            )
        buffer[offset : offset + len(cell)] = cell
        offsets.append(offset)
    free_space_offset = offset

    header_bytes = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        PAGE_FORMAT_VERSION,
        int(page_type),
        cell_count,
        free_space_offset,
        side_pointer,
        0,
    )
    buffer[0:HEADER_SIZE] = header_bytes
    for index, cell_offset in enumerate(offsets):
        pointer_offset = HEADER_SIZE + index * CELL_POINTER_SIZE
        struct.pack_into(CELL_POINTER_FORMAT, buffer, pointer_offset, cell_offset)

    checksum = _checksum(buffer)
    struct.pack_into(">I", buffer, _CHECKSUM_OFFSET, checksum)
    return bytes(buffer)


def _verify_and_unpack_header(data: bytes, expected_type: PageType) -> PageHeader:
    if len(data) != PAGE_SIZE:
        raise TruncatedPageError(
            f"se esperaban {PAGE_SIZE} bytes de página y se recibieron {len(data)}"
        )

    magic, version, page_type_raw, cell_count, free_space_offset, side_pointer, checksum = (
        struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
    )
    if magic != MAGIC:
        raise InvalidPageError(f"magic de página inválido: {magic!r}")
    if version != PAGE_FORMAT_VERSION:
        raise InvalidPageError(f"versión de formato de página no soportada: {version}")
    try:
        page_type = PageType(page_type_raw)
    except ValueError as exc:
        raise InvalidPageError(f"page_type desconocido: {page_type_raw}") from exc
    if page_type is not expected_type:
        raise InvalidPageError(f"se esperaba una página {expected_type.name} y es {page_type.name}")

    verification_buffer = bytearray(data)
    struct.pack_into(">I", verification_buffer, _CHECKSUM_OFFSET, 0)
    computed_checksum = _checksum(verification_buffer)
    if computed_checksum != checksum:
        raise ChecksumMismatchError(
            f"checksum de página inválido: almacenado {checksum}, calculado {computed_checksum}"
        )

    return PageHeader(
        page_type=page_type,
        cell_count=cell_count,
        free_space_offset=free_space_offset,
        side_pointer=side_pointer,
    )


def _disassemble_page(data: bytes, expected_type: PageType) -> tuple[PageHeader, list[bytes]]:
    header = _verify_and_unpack_header(data, expected_type)
    pointer_array_start = HEADER_SIZE
    offsets = [
        struct.unpack_from(
            CELL_POINTER_FORMAT, data, pointer_array_start + index * CELL_POINTER_SIZE
        )[0]
        for index in range(header.cell_count)
    ]
    cells: list[bytes] = []
    for index, cell_offset in enumerate(offsets):
        cell_end = offsets[index - 1] if index > 0 else PAGE_SIZE
        cells.append(bytes(data[cell_offset:cell_end]))
    return header, cells


def encode_leaf_page(entries: list[LeafEntry], next_leaf: PageId = NULL_PAGE_ID) -> bytes:
    """Serializa una página hoja. `entries` debe venir ordenada por clave
    ascendente; este codec no reordena ni deduplica."""
    cells = [
        struct.pack(_KEY_LENGTH_FORMAT, len(entry.key))
        + entry.key
        + struct.pack(_VALUE_LENGTH_FORMAT, len(entry.value))
        + entry.value
        for entry in entries
    ]
    return _assemble_page(PageType.LEAF, cells, next_leaf)


# --- Acelerador nativo opcional ------------------------------------------------
#
# `bplus-tree-storage-engine-native` (crate PyO3 en `native/`) decodifica las
# páginas hoja e internas en Rust. Es opcional: sin él, o con
# `BPLUS_TREE_PURE_PYTHON=1`, todo funciona igual en Python puro. Contrato:
# el camino nativo sólo acepta páginas válidas y en rango; ante cualquier
# anomalía devuelve `None` y se usa la implementación Python de referencia,
# que produce el resultado o el error exacto de siempre. Los tests de
# propiedades comparan ambos caminos sobre páginas aleatorias y corruptas.


def _load_native() -> Any:
    if os.environ.get("BPLUS_TREE_PURE_PYTHON"):
        return None
    try:
        return importlib.import_module("bplus_tree_native")
    except ImportError:
        return None


_native: Any = _load_native()
NATIVE_CODEC = _native is not None


def decode_leaf_page(data: bytes) -> LeafPage:
    if _native is not None and type(data) is bytes:
        decoded = _native.decode_leaf_cells(data)
        if decoded is not None:
            side_pointer, pairs = decoded
            return LeafPage(
                side_pointer=side_pointer,
                entries=[LeafEntry.trusted(key, value) for key, value in pairs],
            )
    return _decode_leaf_page_python(data)


def decode_internal_page(data: bytes) -> InternalPage:
    if _native is not None and type(data) is bytes:
        decoded = _native.decode_internal_cells(data)
        if decoded is not None:
            side_pointer, pairs = decoded
            return InternalPage(
                side_pointer=side_pointer,
                entries=[InternalEntry.trusted(key, child) for key, child in pairs],
            )
    return _decode_internal_page_python(data)


def _decode_leaf_page_python(data: bytes) -> LeafPage:
    header, cells = _disassemble_page(data, PageType.LEAF)
    entries: list[LeafEntry] = []
    for cell in cells:
        offset = 0
        (key_len,) = struct.unpack_from(_KEY_LENGTH_FORMAT, cell, offset)
        offset += _KEY_LENGTH_SIZE
        key = cell[offset : offset + key_len]
        offset += key_len
        (value_len,) = struct.unpack_from(_VALUE_LENGTH_FORMAT, cell, offset)
        offset += _VALUE_LENGTH_SIZE
        value = cell[offset : offset + value_len]
        entries.append(LeafEntry(key=key, value=value))
    return LeafPage(side_pointer=header.side_pointer, entries=entries)


def encode_internal_page(entries: list[InternalEntry], rightmost_child: PageId) -> bytes:
    """Serializa una página interna. `entries` debe venir ordenada por
    clave separadora ascendente; `rightmost_child` es el hijo del rango
    de claves >= la última clave separadora."""
    cells = [
        struct.pack(_KEY_LENGTH_FORMAT, len(entry.key))
        + entry.key
        + struct.pack(_CHILD_ID_FORMAT, entry.left_child)
        for entry in entries
    ]
    return _assemble_page(PageType.INTERNAL, cells, rightmost_child)


def _decode_internal_page_python(data: bytes) -> InternalPage:
    header, cells = _disassemble_page(data, PageType.INTERNAL)
    entries: list[InternalEntry] = []
    for cell in cells:
        offset = 0
        (key_len,) = struct.unpack_from(_KEY_LENGTH_FORMAT, cell, offset)
        offset += _KEY_LENGTH_SIZE
        key = cell[offset : offset + key_len]
        offset += key_len
        (left_child,) = struct.unpack_from(_CHILD_ID_FORMAT, cell, offset)
        entries.append(InternalEntry(key=key, left_child=left_child))
    return InternalPage(side_pointer=header.side_pointer, entries=entries)


def encode_metadata_page(metadata: TreeMetadata) -> bytes:
    """Serializa la página de metadatos (página 0). No usa el array de
    celdas: el payload fijo va justo después de la cabecera común."""
    buffer = bytearray(PAGE_SIZE)
    used_up_to = HEADER_SIZE + METADATA_PAYLOAD_SIZE
    header_bytes = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        PAGE_FORMAT_VERSION,
        int(PageType.META),
        0,
        used_up_to,
        NULL_PAGE_ID,
        0,
    )
    buffer[0:HEADER_SIZE] = header_bytes
    payload = struct.pack(
        METADATA_PAYLOAD_FORMAT, metadata.root_page_id, metadata.page_count, metadata.free_page_head
    )
    buffer[HEADER_SIZE:used_up_to] = payload

    checksum = _checksum(buffer)
    struct.pack_into(">I", buffer, _CHECKSUM_OFFSET, checksum)
    return bytes(buffer)


def decode_metadata_page(data: bytes) -> TreeMetadata:
    _verify_and_unpack_header(data, PageType.META)
    root_page_id, page_count, free_page_head = struct.unpack_from(
        METADATA_PAYLOAD_FORMAT, data, HEADER_SIZE
    )
    return TreeMetadata(
        root_page_id=root_page_id, page_count=page_count, free_page_head=free_page_head
    )


# Diario de doble escritura de `FilePageDevice.write_pages_atomically`:
# cabecera (magic, versión, nº de páginas), registros (page_id + página) y
# un CRC32 final de todo lo anterior. Un diario incompleto o con CRC
# erróneo es el rastro de un crash *antes* de empezar las escrituras en su
# sitio, así que se descarta; uno completo se re-aplica entero.
_JOURNAL_MAGIC = b"BPTJ"
_JOURNAL_VERSION = 1
_JOURNAL_HEADER = struct.Struct(">4sII")
_JOURNAL_PAGE_ID = struct.Struct(">I")
_JOURNAL_CRC = struct.Struct(">I")


def _fsync_directory(directory: Path) -> None:
    """Hace durable la creación o el borrado de una entrada del directorio
    (el diario). En Windows no hay descriptores de directorio: no aplica."""
    if os.name == "nt":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        raise StorageIOError(f"fallo de E/S en fsync del directorio {directory}: {exc}") from exc


def _parse_journal(data: bytes) -> list[tuple[PageId, bytes]] | None:
    """Páginas de un diario completo y válido, o `None` si está a medias."""
    if len(data) < _JOURNAL_HEADER.size + _JOURNAL_CRC.size:
        return None
    magic, version, count = _JOURNAL_HEADER.unpack_from(data, 0)
    record = _JOURNAL_PAGE_ID.size + PAGE_SIZE
    if magic != _JOURNAL_MAGIC or version != _JOURNAL_VERSION:
        return None
    if len(data) != _JOURNAL_HEADER.size + count * record + _JOURNAL_CRC.size:
        return None
    (stored,) = _JOURNAL_CRC.unpack_from(data, len(data) - _JOURNAL_CRC.size)
    if zlib.crc32(data[: -_JOURNAL_CRC.size]) & 0xFFFFFFFF != stored:
        return None
    pages: list[tuple[PageId, bytes]] = []
    offset = _JOURNAL_HEADER.size
    for _ in range(count):
        (page_id,) = _JOURNAL_PAGE_ID.unpack_from(data, offset)
        offset += _JOURNAL_PAGE_ID.size
        pages.append((PageId(page_id), data[offset : offset + PAGE_SIZE]))
        offset += PAGE_SIZE
    return pages


class FilePageDevice:
    """`PageDevice` respaldado por un archivo real en disco.

    El tamaño del archivo es siempre un múltiplo exacto de `PAGE_SIZE`:
    si no lo es al abrir, se interpreta como el rastro de un `write_page`
    o `allocate_page` interrumpido a mitad (proceso matado durante la
    escritura) y se reporta como `TruncatedPageError` en vez de arrancar
    con datos parciales silenciosamente.

    También es un `AtomicPageDevice`: `write_pages_atomically` escribe el
    lote primero en un diario al lado (`<archivo>.journal`, con fsync) y
    sólo después en su sitio. Al abrir, un diario completo se re-aplica y
    uno a medias se descarta, así que un crash en cualquier punto deja el
    lote entero o nada de él.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._journal_path = self._path.with_name(self._path.name + ".journal")
        mode = "r+b" if self._path.exists() else "w+b"
        try:
            self._file: IO[bytes] = open(self._path, mode)  # noqa: SIM115 (cerrado en close())
        except OSError as exc:
            raise StorageIOError(f"no se pudo abrir {self._path}: {exc}") from exc
        self._recover_journal()
        self._page_count = self._read_page_count_from_file_size()

    def _recover_journal(self) -> None:
        if not self._journal_path.exists():
            return
        try:
            pages = _parse_journal(self._journal_path.read_bytes())
            if pages is not None:
                for page_id, data in pages:
                    self._file.seek(page_id * PAGE_SIZE)
                    self._file.write(data)
                self._file.flush()
                os.fsync(self._file.fileno())
            self._journal_path.unlink()
        except OSError as exc:
            raise StorageIOError(f"fallo de E/S recuperando {self._journal_path}: {exc}") from exc
        _fsync_directory(self._path.parent)

    def write_pages_atomically(self, pages: list[tuple[PageId, bytes]]) -> None:
        if not pages:
            self.sync()
            return
        parts = [_JOURNAL_HEADER.pack(_JOURNAL_MAGIC, _JOURNAL_VERSION, len(pages))]
        for page_id, data in pages:
            if len(data) != PAGE_SIZE:
                raise ValueError(f"write_page requiere {PAGE_SIZE} bytes, recibidos {len(data)}")
            if page_id >= self._page_count:
                raise ValueError(f"write_page en página {page_id} no asignada aún")
            parts += [_JOURNAL_PAGE_ID.pack(page_id), data]
        body = b"".join(parts)
        blob = body + _JOURNAL_CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)
        try:
            with open(self._journal_path, "wb") as journal:
                journal.write(blob)
                journal.flush()
                os.fsync(journal.fileno())
        except OSError as exc:
            raise StorageIOError(f"fallo de E/S escribiendo {self._journal_path}: {exc}") from exc
        _fsync_directory(self._path.parent)
        # Desde aquí el lote sobrevive a cualquier crash: lo que no llegue a
        # escribirse en su sitio lo re-aplica `_recover_journal` al abrir.
        for page_id, data in pages:
            self.write_page(page_id, data)
        self.sync()
        try:
            self._journal_path.unlink()
        except OSError as exc:
            raise StorageIOError(f"fallo de E/S borrando {self._journal_path}: {exc}") from exc
        _fsync_directory(self._path.parent)

    def _read_page_count_from_file_size(self) -> int:
        try:
            self._file.seek(0, io.SEEK_END)
            size = self._file.tell()
        except OSError as exc:
            raise StorageIOError(f"no se pudo determinar el tamaño de {self._path}: {exc}") from exc
        if size % PAGE_SIZE != 0:
            raise TruncatedPageError(
                f"{self._path} mide {size} bytes, no es múltiplo de PAGE_SIZE={PAGE_SIZE}"
            )
        return size // PAGE_SIZE

    def read_page(self, page_id: PageId) -> bytes:
        if page_id >= self._page_count:
            raise TruncatedPageError(
                f"página {page_id} fuera de rango: el archivo tiene {self._page_count} páginas"
            )
        try:
            self._file.seek(page_id * PAGE_SIZE)
            data = self._file.read(PAGE_SIZE)
        except OSError as exc:
            raise StorageIOError(f"fallo de E/S leyendo página {page_id}: {exc}") from exc
        if len(data) != PAGE_SIZE:
            raise TruncatedPageError(f"página {page_id} truncada: se leyeron {len(data)} bytes")
        return data

    def write_page(self, page_id: PageId, data: bytes) -> None:
        if len(data) != PAGE_SIZE:
            raise ValueError(f"write_page requiere {PAGE_SIZE} bytes, recibidos {len(data)}")
        if page_id >= self._page_count:
            raise ValueError(f"write_page en página {page_id} no asignada aún")
        try:
            self._file.seek(page_id * PAGE_SIZE)
            self._file.write(data)
        except OSError as exc:
            raise StorageIOError(f"fallo de E/S escribiendo página {page_id}: {exc}") from exc

    def allocate_page(self) -> PageId:
        new_page_id = self._page_count
        try:
            self._file.seek(new_page_id * PAGE_SIZE)
            self._file.write(bytes(PAGE_SIZE))
        except OSError as exc:
            raise StorageIOError(f"fallo de E/S asignando página {new_page_id}: {exc}") from exc
        self._page_count += 1
        return new_page_id

    def page_count(self) -> int:
        return self._page_count

    def sync(self) -> None:
        try:
            self._file.flush()
            os.fsync(self._file.fileno())
        except OSError as exc:
            raise StorageIOError(f"fallo de E/S en fsync de {self._path}: {exc}") from exc

    def close(self) -> None:
        try:
            self._file.close()
        except OSError as exc:
            raise StorageIOError(f"fallo de E/S cerrando {self._path}: {exc}") from exc

    def __enter__(self) -> FilePageDevice:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class InMemoryPageDevice:
    """`PageDevice` en memoria: mismo contrato que `FilePageDevice` sin
    tocar el sistema de archivos, para tests rápidos y deterministas."""

    def __init__(self) -> None:
        self._pages: list[bytearray] = []

    def read_page(self, page_id: PageId) -> bytes:
        if page_id >= len(self._pages):
            raise TruncatedPageError(
                f"página {page_id} fuera de rango: hay {len(self._pages)} páginas"
            )
        return bytes(self._pages[page_id])

    def write_page(self, page_id: PageId, data: bytes) -> None:
        if len(data) != PAGE_SIZE:
            raise ValueError(f"write_page requiere {PAGE_SIZE} bytes, recibidos {len(data)}")
        if page_id >= len(self._pages):
            raise ValueError(f"write_page en página {page_id} no asignada aún")
        self._pages[page_id] = bytearray(data)

    def allocate_page(self) -> PageId:
        self._pages.append(bytearray(PAGE_SIZE))
        return len(self._pages) - 1

    def page_count(self) -> int:
        return len(self._pages)

    def sync(self) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> InMemoryPageDevice:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class LruReplacementPolicy:
    """Política de reemplazo LRU: desaloja siempre la página cuyo acceso
    registrado sea el menos reciente."""

    def __init__(self) -> None:
        self._order: OrderedDict[PageId, None] = OrderedDict()

    def record_access(self, page_id: PageId) -> None:
        if page_id in self._order:
            self._order.move_to_end(page_id)
        else:
            self._order[page_id] = None

    def choose_victim(self) -> PageId | None:
        if not self._order:
            return None
        return next(iter(self._order))

    def forget(self, page_id: PageId) -> None:
        self._order.pop(page_id, None)


@dataclass(slots=True)
class _CacheEntry:
    data: bytearray
    dirty: bool


class BufferPool:
    """Buffer pool de tamaño acotado sobre un `PageDevice`, con desalojo
    vía `ReplacementPolicy` (LRU por defecto). Escritura "write-back": las
    páginas modificadas se mantienen en memoria y sólo se persisten al
    desalojarlas o al llamar a `flush()` / `close()` explícitamente.
    """

    def __init__(
        self,
        device: PageDevice,
        config: BufferPoolConfig,
        replacement_policy: ReplacementPolicy | None = None,
    ) -> None:
        self._device = device
        self._capacity = config.capacity
        self._crash_consistent = config.crash_consistent
        self._policy: ReplacementPolicy = (
            replacement_policy if replacement_policy is not None else LruReplacementPolicy()
        )
        self._cache: dict[PageId, _CacheEntry] = {}

    def get_page(self, page_id: PageId) -> bytes:
        entry = self._cache.get(page_id)
        if entry is None:
            data = self._device.read_page(page_id)
            entry = _CacheEntry(data=bytearray(data), dirty=False)
            self._insert_new(page_id, entry)
        self._policy.record_access(page_id)
        return bytes(entry.data)

    def put_page(self, page_id: PageId, data: bytes) -> None:
        if len(data) != PAGE_SIZE:
            raise ValueError(f"put_page requiere {PAGE_SIZE} bytes, recibidos {len(data)}")
        entry = self._cache.get(page_id)
        if entry is None:
            self._insert_new(page_id, _CacheEntry(data=bytearray(data), dirty=True))
        else:
            entry.data = bytearray(data)
            entry.dirty = True
        self._policy.record_access(page_id)

    def allocate_page(self) -> PageId:
        return self._device.allocate_page()

    def page_count(self) -> int:
        return self._device.page_count()

    def _insert_new(self, page_id: PageId, entry: _CacheEntry) -> None:
        if len(self._cache) >= self._capacity:
            self._evict_one()
        self._cache[page_id] = entry

    def _evict_one(self) -> None:
        if self._crash_consistent:
            self._evict_one_clean()
            return
        victim = self._policy.choose_victim()
        while victim is not None and victim not in self._cache:
            self._policy.forget(victim)
            victim = self._policy.choose_victim()
        if victim is None:
            raise TreeInvariantError(
                "el buffer pool está a capacidad pero la política no tiene víctima que desalojar"
            )
        evicted = self._cache.pop(victim)
        self._policy.forget(victim)
        if evicted.dirty:
            self._device.write_page(victim, bytes(evicted.data))

    def _evict_one_clean(self) -> None:
        """Desalojo *no-steal*: la víctima es la página limpia menos
        reciente. Las sucias se rotan al final del orden sin escribirlas;
        si todas lo están, no se desaloja nada y el pool crece por encima
        de `capacity` hasta el próximo `flush()`."""
        attempts = len(self._cache)
        while attempts > 0:
            victim = self._policy.choose_victim()
            if victim is None:
                return
            if victim not in self._cache:
                self._policy.forget(victim)
                continue
            attempts -= 1
            if not self._cache[victim].dirty:
                del self._cache[victim]
                self._policy.forget(victim)
                return
            self._policy.record_access(victim)

    def flush(self) -> None:
        """Persiste todas las páginas sucias y fuerza `fsync` del
        dispositivo. Ninguna escritura se considera durable hasta que
        este método retorna. En modo `crash_consistent`, y si el
        dispositivo es un `AtomicPageDevice`, el lote es atómico."""
        dirty = [(page_id, entry) for page_id, entry in self._cache.items() if entry.dirty]
        if self._crash_consistent and isinstance(self._device, AtomicPageDevice):
            self._device.write_pages_atomically([(pid, bytes(e.data)) for pid, e in dirty])
            for _, entry in dirty:
                entry.dirty = False
            # Lo que creció por encima de `capacity` por no-steal ya está
            # limpio: se vuelve al tamaño configurado.
            while len(self._cache) > self._capacity:
                before = len(self._cache)
                self._evict_one_clean()
                if len(self._cache) == before:
                    break
            return
        for page_id, entry in dirty:
            self._device.write_page(page_id, bytes(entry.data))
            entry.dirty = False
        self._device.sync()

    def close(self) -> None:
        self.flush()
        self._device.close()

    def __enter__(self) -> BufferPool:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# --- Árbol B+ ---------------------------------------------------------
#
# Formato de las páginas internas: `entries` ordenadas por clave
# ascendente donde `entries[i].left_child` contiene las claves del rango
# [entries[i-1].key, entries[i].key) (o (-inf, entries[0].key) para i=0);
# `side_pointer` es el hijo más a la derecha, con las claves >= la última
# clave separadora. Ver `InternalEntry` en `models.py`.
#
# El split de hoja copia la clave separadora (las hojas guardan todos los
# datos); el split de nodo interno la promueve (los nodos internos sólo
# enrutan, así que la clave promovida deja de vivir en este nivel).


def _peek_page_type(data: bytes) -> PageType:
    """Lee sólo el campo `page_type` de la cabecera, sin verificar
    checksum, para decidir qué decoder invocar durante un recorrido. La
    corrupción real se detecta igualmente al llamar al decoder completo
    correspondiente (que sí verifica el checksum)."""
    if len(data) != PAGE_SIZE:
        raise TruncatedPageError(
            f"se esperaban {PAGE_SIZE} bytes de página y se recibieron {len(data)}"
        )
    raw_type = data[_PAGE_TYPE_OFFSET]
    try:
        return PageType(raw_type)
    except ValueError as exc:
        raise InvalidPageError(f"page_type desconocido: {raw_type}") from exc


def leaf_page_size(entries: list[LeafEntry]) -> int:
    """Tamaño total, en bytes, que ocuparía una página hoja con estas
    entradas: cabecera + array de punteros + contenido de celdas."""
    content = sum(leaf_cell_size(entry.key, entry.value) for entry in entries)
    return HEADER_SIZE + len(entries) * CELL_POINTER_SIZE + content


def internal_page_size(entries: list[InternalEntry]) -> int:
    """Equivalente a `leaf_page_size` para una página interna."""
    content = sum(internal_cell_size(entry.key) for entry in entries)
    return HEADER_SIZE + len(entries) * CELL_POINTER_SIZE + content


def _find_leaf_split_point(entries: list[LeafEntry]) -> int:
    """Índice a partir del cual empieza la mitad derecha tras un split.

    El corte se elige por tamaño acumulado (no por cuenta de entradas):
    dado que cada entrada individual está acotada por `MAX_ENTRY_SIZE`
    (ver `EntryTooLargeError`), y que estas `entries` sólo exceden
    `PAGE_SIZE` por, como mucho, una entrada de más sobre una página que
    antes cabía, siempre existe un corte que deja ambas mitades por
    debajo de `PAGE_SIZE`. Repartir por cuenta en vez de por tamaño podría
    dejar, en teoría, una mitad arbitrariamente grande si las entradas
    tienen tamaños muy dispares.
    """
    cumulative = HEADER_SIZE + CELL_POINTER_SIZE + leaf_cell_size(entries[0].key, entries[0].value)
    split_at = 1
    for index in range(1, len(entries)):
        entry = entries[index]
        cumulative += CELL_POINTER_SIZE + leaf_cell_size(entry.key, entry.value)
        if cumulative > PAGE_SIZE:
            break
        split_at = index + 1
    return split_at


def _find_internal_split_point(entries: list[InternalEntry]) -> int:
    """Índice de la entrada a promover al padre: la mitad izquierda queda
    en `entries[:m]` (con `side_pointer` = `entries[m].left_child`), la
    derecha en `entries[m + 1:]`. Mismo razonamiento de corte por tamaño
    acumulado que `_find_leaf_split_point`."""
    cumulative = HEADER_SIZE + CELL_POINTER_SIZE + internal_cell_size(entries[0].key)
    split_at = 0
    for index in range(1, len(entries)):
        entry = entries[index]
        cumulative += CELL_POINTER_SIZE + internal_cell_size(entry.key)
        if cumulative > PAGE_SIZE:
            break
        split_at = index
    return split_at


class BPlusTree:
    """Árbol B+ persistente: índice primario clave -> valor sobre un
    `BufferPool`. La página 0 del dispositivo subyacente son siempre los
    metadatos (`TreeMetadata`); si el dispositivo está vacío, se
    inicializa aquí un árbol nuevo (metadatos + una hoja raíz vacía).
    """

    def __init__(self, buffer_pool: BufferPool) -> None:
        self._pool = buffer_pool
        if self._pool.page_count() == 0:
            self._bootstrap()
        elif self._pool.get_page(METADATA_PAGE_ID) == bytes(PAGE_SIZE):
            # `allocate_page` escribe ceros; una página de metadatos a cero
            # significa que el primer flush del árbol nunca terminó (crash
            # durante la creación): no hay nada persistido que conservar.
            self._bootstrap(reuse_allocated=True)
        self._metadata = decode_metadata_page(self._pool.get_page(METADATA_PAGE_ID))

    def _bootstrap(self, *, reuse_allocated: bool = False) -> None:
        if reuse_allocated:
            metadata_page_id = METADATA_PAGE_ID
            root_page_id = 1 if self._pool.page_count() > 1 else self._pool.allocate_page()
        else:
            metadata_page_id = self._pool.allocate_page()
            root_page_id = self._pool.allocate_page()
        if metadata_page_id != METADATA_PAGE_ID:
            raise TreeInvariantError(
                f"la página de metadatos debía asignarse como {METADATA_PAGE_ID}, "
                f"se asignó {metadata_page_id}"
            )
        self._pool.put_page(root_page_id, encode_leaf_page([], next_leaf=NULL_PAGE_ID))
        metadata = TreeMetadata(root_page_id=root_page_id, page_count=self._pool.page_count())
        self._pool.put_page(METADATA_PAGE_ID, encode_metadata_page(metadata))
        self._pool.flush()

    def _save_metadata(self, root_page_id: PageId) -> None:
        self._metadata = TreeMetadata(
            root_page_id=root_page_id,
            page_count=self._pool.page_count(),
            free_page_head=self._metadata.free_page_head,
        )
        self._pool.put_page(METADATA_PAGE_ID, encode_metadata_page(self._metadata))

    @property
    def metadata(self) -> TreeMetadata:
        """Instantánea de sólo lectura de los metadatos actuales del árbol
        (página raíz, páginas asignadas). Para inspección/CLI; la lógica
        interna sigue usando `self._metadata` directamente."""
        return self._metadata

    def flush(self) -> None:
        self._pool.flush()

    def close(self) -> None:
        self._pool.close()

    def __enter__(self) -> BPlusTree:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def search(self, key: bytes) -> bytes | None:
        """Devuelve el valor asociado a `key`, o `None` si no existe."""
        page_id = self._metadata.root_page_id
        while True:
            raw = self._pool.get_page(page_id)
            if _peek_page_type(raw) is PageType.LEAF:
                leaf = decode_leaf_page(raw)
                index = bisect.bisect_left(leaf.entries, key, key=lambda entry: entry.key)
                if index < len(leaf.entries) and leaf.entries[index].key == key:
                    return leaf.entries[index].value
                return None
            internal = decode_internal_page(raw)
            index = bisect.bisect_right(internal.entries, key, key=lambda entry: entry.key)
            page_id = (
                internal.entries[index].left_child
                if index < len(internal.entries)
                else internal.side_pointer
            )

    def scan(
        self, start: bytes | None = None, end: bytes | None = None
    ) -> Iterator[tuple[bytes, bytes]]:
        """Recorre las entradas en orden ascendente de clave, opcionalmente
        acotado a `[start, end)`. Sigue la cadena de hojas enlazadas
        (`side_pointer`) en vez de recorrer el árbol completo: la ventaja
        clásica de un índice B+ frente a uno hash para lecturas de rango.
        """
        page_id = self._locate_starting_leaf(start)
        while page_id != NULL_PAGE_ID:
            leaf = decode_leaf_page(self._pool.get_page(page_id))
            for entry in leaf.entries:
                if start is not None and entry.key < start:
                    continue
                if end is not None and entry.key >= end:
                    return
                yield entry.key, entry.value
            page_id = leaf.side_pointer

    def _locate_starting_leaf(self, start: bytes | None) -> PageId:
        page_id = self._metadata.root_page_id
        while True:
            raw = self._pool.get_page(page_id)
            if _peek_page_type(raw) is PageType.LEAF:
                return page_id
            internal = decode_internal_page(raw)
            if start is None:
                page_id = (
                    internal.entries[0].left_child if internal.entries else internal.side_pointer
                )
            else:
                index = bisect.bisect_right(internal.entries, start, key=lambda entry: entry.key)
                page_id = (
                    internal.entries[index].left_child
                    if index < len(internal.entries)
                    else internal.side_pointer
                )

    def insert(self, key: bytes, value: bytes) -> None:
        """Inserta `key`/`value`, o sobrescribe el valor si `key` ya existía."""
        if leaf_cell_size(key, value) > MAX_ENTRY_SIZE or len(key) > MAX_KEY_SIZE:
            raise EntryTooLargeError(
                f"la entrada para la clave de {len(key)} bytes excede el máximo soportado"
            )

        path: list[tuple[PageId, int]] = []
        page_id = self._metadata.root_page_id
        while True:
            raw = self._pool.get_page(page_id)
            if _peek_page_type(raw) is PageType.LEAF:
                break
            internal = decode_internal_page(raw)
            index = bisect.bisect_right(internal.entries, key, key=lambda entry: entry.key)
            path.append((page_id, index))
            page_id = (
                internal.entries[index].left_child
                if index < len(internal.entries)
                else internal.side_pointer
            )

        leaf = decode_leaf_page(self._pool.get_page(page_id))
        index = bisect.bisect_left(leaf.entries, key, key=lambda entry: entry.key)
        if index < len(leaf.entries) and leaf.entries[index].key == key:
            leaf.entries[index] = LeafEntry(key=key, value=value)
        else:
            leaf.entries.insert(index, LeafEntry(key=key, value=value))

        if leaf_page_size(leaf.entries) <= PAGE_SIZE:
            self._pool.put_page(page_id, encode_leaf_page(leaf.entries, leaf.side_pointer))
            self._save_metadata(self._metadata.root_page_id)
            return

        split_at = _find_leaf_split_point(leaf.entries)
        left_entries = leaf.entries[:split_at]
        right_entries = leaf.entries[split_at:]
        separator_key = right_entries[0].key

        new_leaf_page_id = self._pool.allocate_page()
        self._pool.put_page(
            new_leaf_page_id, encode_leaf_page(right_entries, next_leaf=leaf.side_pointer)
        )
        self._pool.put_page(page_id, encode_leaf_page(left_entries, next_leaf=new_leaf_page_id))

        self._propagate_split(path, separator_key, page_id, new_leaf_page_id)
        self._save_metadata(self._metadata.root_page_id)

    def _propagate_split(
        self,
        path: list[tuple[PageId, int]],
        separator_key: bytes,
        left_page_id: PageId,
        right_page_id: PageId,
    ) -> None:
        """Inserta `(separator_key, left_page_id)` en el padre inmediato de
        `path` (ajustando qué apunta a `right_page_id`), haciendo split en
        cascada hacia arriba si hace falta. Si `path` se agota, crea una
        raíz nueva y actualiza `self._metadata` (sin persistirla: eso lo
        hace el llamante)."""
        while path:
            parent_page_id, descend_index = path.pop()
            parent = decode_internal_page(self._pool.get_page(parent_page_id))

            if descend_index == len(parent.entries):
                parent.side_pointer = right_page_id
                insert_position = len(parent.entries)
            else:
                original = parent.entries[descend_index]
                parent.entries[descend_index] = InternalEntry(
                    key=original.key, left_child=right_page_id
                )
                insert_position = descend_index
            parent.entries.insert(
                insert_position, InternalEntry(key=separator_key, left_child=left_page_id)
            )

            if internal_page_size(parent.entries) <= PAGE_SIZE:
                self._pool.put_page(
                    parent_page_id, encode_internal_page(parent.entries, parent.side_pointer)
                )
                return

            split_at = _find_internal_split_point(parent.entries)
            promoted = parent.entries[split_at]
            left_entries = parent.entries[:split_at]
            right_entries = parent.entries[split_at + 1 :]

            new_right_page_id = self._pool.allocate_page()
            self._pool.put_page(
                new_right_page_id, encode_internal_page(right_entries, parent.side_pointer)
            )
            self._pool.put_page(
                parent_page_id, encode_internal_page(left_entries, promoted.left_child)
            )

            separator_key = promoted.key
            left_page_id = parent_page_id
            right_page_id = new_right_page_id

        new_root_page_id = self._pool.allocate_page()
        new_root_entries = [InternalEntry(key=separator_key, left_child=left_page_id)]
        self._pool.put_page(new_root_page_id, encode_internal_page(new_root_entries, right_page_id))
        self._metadata = TreeMetadata(
            root_page_id=new_root_page_id,
            page_count=self._metadata.page_count,
            free_page_head=self._metadata.free_page_head,
        )

    def delete(self, key: bytes) -> None:
        """Borra `key`. Lanza `KeyNotFoundError` si no existe.

        Al desalojar una entrada, una página no raíz que quede por debajo
        de `_UNDERFLOW_THRESHOLD` intenta fusionarse con un hermano bajo
        el mismo padre inmediato (izquierdo primero, luego derecho). Si
        ningún hermano tiene hueco para absorberla, la página se deja tal
        cual: sigue siendo un árbol B+ válido y buscable, sólo algo menos
        compacto. No se implementa redistribución (préstamo de una única
        entrada entre hermanos sin fusionar): habría cubierto algo más de
        casos de fusión fallida a cambio de una cascada de casos límite
        adicional (el separador del padre podría crecer de tamaño al
        tomar prestada una clave más larga) que no aporta corrección,
        sólo compacidad marginal.
        """
        path: list[tuple[PageId, int]] = []
        page_id = self._metadata.root_page_id
        while True:
            raw = self._pool.get_page(page_id)
            if _peek_page_type(raw) is PageType.LEAF:
                break
            internal = decode_internal_page(raw)
            index = bisect.bisect_right(internal.entries, key, key=lambda entry: entry.key)
            path.append((page_id, index))
            page_id = (
                internal.entries[index].left_child
                if index < len(internal.entries)
                else internal.side_pointer
            )

        leaf = decode_leaf_page(self._pool.get_page(page_id))
        index = bisect.bisect_left(leaf.entries, key, key=lambda entry: entry.key)
        if index >= len(leaf.entries) or leaf.entries[index].key != key:
            raise KeyNotFoundError(f"clave no encontrada: {key!r}")
        del leaf.entries[index]

        self._rebalance_leaf(path, page_id, leaf)
        self._save_metadata(self._metadata.root_page_id)

    def _sibling_ids(
        self, parent: InternalPage, descend_index: int
    ) -> tuple[PageId | None, PageId | None]:
        """Hermanos inmediatos, bajo `parent`, del hijo alcanzado usando
        `descend_index` (ver el bookkeeping de `path` en `insert`/`delete`)."""
        entry_count = len(parent.entries)
        left = parent.entries[descend_index - 1].left_child if descend_index > 0 else None
        if descend_index < entry_count:
            right = (
                parent.entries[descend_index + 1].left_child
                if descend_index + 1 < entry_count
                else parent.side_pointer
            )
        else:
            right = None
        return left, right

    @staticmethod
    def _remove_separator_and_retarget(
        parent: InternalPage, boundary: int, surviving_page_id: PageId
    ) -> None:
        """Tras fusionar dos hermanos en una sola página (`surviving_page_id`),
        elimina de `parent` el separador que los distinguía (en el índice
        `boundary`) y redirige el puntero que quedaba justo después de ese
        separador hacia la página superviviente."""
        target_index = boundary + 1
        if target_index < len(parent.entries):
            original_key = parent.entries[target_index].key
            parent.entries[target_index] = InternalEntry(
                key=original_key, left_child=surviving_page_id
            )
        else:
            parent.side_pointer = surviving_page_id
        del parent.entries[boundary]

    def _rebalance_leaf(
        self, path: list[tuple[PageId, int]], leaf_page_id: PageId, leaf: LeafPage
    ) -> None:
        if not path or leaf_page_size(leaf.entries) >= _UNDERFLOW_THRESHOLD:
            self._pool.put_page(leaf_page_id, encode_leaf_page(leaf.entries, leaf.side_pointer))
            return

        parent_page_id, descend_index = path[-1]
        parent = decode_internal_page(self._pool.get_page(parent_page_id))
        left_sibling_id, right_sibling_id = self._sibling_ids(parent, descend_index)

        if left_sibling_id is not None:
            left_leaf = decode_leaf_page(self._pool.get_page(left_sibling_id))
            merged = left_leaf.entries + leaf.entries
            if leaf_page_size(merged) <= PAGE_SIZE:
                self._pool.put_page(
                    left_sibling_id, encode_leaf_page(merged, next_leaf=leaf.side_pointer)
                )
                self._remove_separator_and_retarget(parent, descend_index - 1, left_sibling_id)
                self._rebalance_internal(path[:-1], parent_page_id, parent)
                return

        if right_sibling_id is not None:
            right_leaf = decode_leaf_page(self._pool.get_page(right_sibling_id))
            merged = leaf.entries + right_leaf.entries
            if leaf_page_size(merged) <= PAGE_SIZE:
                self._pool.put_page(
                    leaf_page_id, encode_leaf_page(merged, next_leaf=right_leaf.side_pointer)
                )
                self._remove_separator_and_retarget(parent, descend_index, leaf_page_id)
                self._rebalance_internal(path[:-1], parent_page_id, parent)
                return

        self._pool.put_page(leaf_page_id, encode_leaf_page(leaf.entries, leaf.side_pointer))

    def _rebalance_internal(
        self, path: list[tuple[PageId, int]], node_page_id: PageId, node: InternalPage
    ) -> None:
        if not path:
            if len(node.entries) == 0:
                # El nodo raíz quedó como un simple paso intermedio hacia un
                # único hijo: la altura del árbol se reduce adoptándolo
                # directamente como nueva raíz. La página vieja de la raíz
                # queda asignada pero sin referencias (no se recicla en
                # esta versión: ver `TreeMetadata.free_page_head`).
                self._metadata = TreeMetadata(
                    root_page_id=node.side_pointer,
                    page_count=self._metadata.page_count,
                    free_page_head=self._metadata.free_page_head,
                )
                return
            self._pool.put_page(node_page_id, encode_internal_page(node.entries, node.side_pointer))
            return

        if internal_page_size(node.entries) >= _UNDERFLOW_THRESHOLD:
            self._pool.put_page(node_page_id, encode_internal_page(node.entries, node.side_pointer))
            return

        parent_page_id, descend_index = path[-1]
        parent = decode_internal_page(self._pool.get_page(parent_page_id))
        left_sibling_id, right_sibling_id = self._sibling_ids(parent, descend_index)

        if left_sibling_id is not None:
            left_node = decode_internal_page(self._pool.get_page(left_sibling_id))
            separator_key = parent.entries[descend_index - 1].key
            pulled_down = InternalEntry(key=separator_key, left_child=left_node.side_pointer)
            merged = [*left_node.entries, pulled_down, *node.entries]
            if internal_page_size(merged) <= PAGE_SIZE:
                self._pool.put_page(
                    left_sibling_id, encode_internal_page(merged, node.side_pointer)
                )
                self._remove_separator_and_retarget(parent, descend_index - 1, left_sibling_id)
                self._rebalance_internal(path[:-1], parent_page_id, parent)
                return

        if right_sibling_id is not None:
            right_node = decode_internal_page(self._pool.get_page(right_sibling_id))
            separator_key = parent.entries[descend_index].key
            pulled_down = InternalEntry(key=separator_key, left_child=node.side_pointer)
            merged = [*node.entries, pulled_down, *right_node.entries]
            if internal_page_size(merged) <= PAGE_SIZE:
                self._pool.put_page(
                    node_page_id, encode_internal_page(merged, right_node.side_pointer)
                )
                self._remove_separator_and_retarget(parent, descend_index, node_page_id)
                self._rebalance_internal(path[:-1], parent_page_id, parent)
                return

        self._pool.put_page(node_page_id, encode_internal_page(node.entries, node.side_pointer))
