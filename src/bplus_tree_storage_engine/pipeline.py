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

import io
import os
import struct
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from bplus_tree_storage_engine.models import (
    CELL_POINTER_FORMAT,
    CELL_POINTER_SIZE,
    HEADER_FORMAT,
    HEADER_SIZE,
    MAGIC,
    METADATA_PAYLOAD_FORMAT,
    METADATA_PAYLOAD_SIZE,
    NULL_PAGE_ID,
    PAGE_FORMAT_VERSION,
    PAGE_SIZE,
    BufferPoolConfig,
    ChecksumMismatchError,
    InternalEntry,
    InvalidPageError,
    LeafEntry,
    PageHeader,
    PageId,
    PageType,
    StorageIOError,
    TreeInvariantError,
    TreeMetadata,
    TruncatedPageError,
)
from bplus_tree_storage_engine.protocols import PageDevice, ReplacementPolicy

_CHECKSUM_SIZE = struct.calcsize(">I")
_CHECKSUM_OFFSET = HEADER_SIZE - _CHECKSUM_SIZE
_KEY_LENGTH_FORMAT = ">H"
_KEY_LENGTH_SIZE = struct.calcsize(_KEY_LENGTH_FORMAT)
_VALUE_LENGTH_FORMAT = ">H"
_VALUE_LENGTH_SIZE = struct.calcsize(_VALUE_LENGTH_FORMAT)
_CHILD_ID_FORMAT = ">I"
_CHILD_ID_SIZE = struct.calcsize(_CHILD_ID_FORMAT)


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
        HEADER_FORMAT, MAGIC, PAGE_FORMAT_VERSION, int(page_type), cell_count, free_space_offset,
        side_pointer, 0,
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


def decode_leaf_page(data: bytes) -> LeafPage:
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


def decode_internal_page(data: bytes) -> InternalPage:
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


class FilePageDevice:
    """`PageDevice` respaldado por un archivo real en disco.

    El tamaño del archivo es siempre un múltiplo exacto de `PAGE_SIZE`:
    si no lo es al abrir, se interpreta como el rastro de un `write_page`
    o `allocate_page` interrumpido a mitad (proceso matado durante la
    escritura) y se reporta como `TruncatedPageError` en vez de arrancar
    con datos parciales silenciosamente.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        mode = "r+b" if self._path.exists() else "w+b"
        try:
            self._file: IO[bytes] = open(self._path, mode)  # noqa: SIM115 (cerrado en close())
        except OSError as exc:
            raise StorageIOError(f"no se pudo abrir {self._path}: {exc}") from exc
        self._page_count = self._read_page_count_from_file_size()

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

    def _insert_new(self, page_id: PageId, entry: _CacheEntry) -> None:
        if len(self._cache) >= self._capacity:
            self._evict_one()
        self._cache[page_id] = entry

    def _evict_one(self) -> None:
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

    def flush(self) -> None:
        """Persiste todas las páginas sucias y fuerza `fsync` del
        dispositivo. Ninguna escritura se considera durable hasta que
        este método retorna."""
        for page_id, entry in self._cache.items():
            if entry.dirty:
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
