"""Lógica core del motor: codec binario de páginas (por ahora).

Este módulo se completará de forma incremental con el buffer pool y la
lógica del árbol B+ en commits posteriores. La responsabilidad de esta
primera pieza es el codec: convertir listas de entradas decodificadas
(`LeafEntry` / `InternalEntry`) o metadatos (`TreeMetadata`) en los
`PAGE_SIZE` bytes exactos que se persisten en disco, y viceversa,
detectando corrupción y truncamiento al leer.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

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
    ChecksumMismatchError,
    InternalEntry,
    InvalidPageError,
    LeafEntry,
    PageHeader,
    PageId,
    PageType,
    TreeInvariantError,
    TreeMetadata,
    TruncatedPageError,
)

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
