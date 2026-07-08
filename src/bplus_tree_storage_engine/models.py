"""Tipos de datos y formato de página del motor de almacenamiento B+Tree.

El motor persiste un único archivo de páginas de tamaño fijo (`PAGE_SIZE`).
La página 0 es siempre una página de metadatos (raíz del árbol, contador de
páginas, cabeza de la lista de páginas libres); el resto son nodos del
árbol (hoja o interno) usando un layout de "slotted page" clásico: cabecera
fija + array de punteros a celdas (ordenado por clave) + contenido de celdas
creciendo desde el final de la página hacia el principio.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

PageId = int
"""Alias de tipo: identificador de página, siempre >= 0. La página 0 está
reservada para los metadatos del árbol y nunca es un nodo del árbol."""

METADATA_PAGE_ID: PageId = 0
NULL_PAGE_ID: PageId = 0
"""Valor centinela para "sin página" (p.ej. hoja sin siguiente hermano).
Coincide con `METADATA_PAGE_ID` porque la página 0 nunca es alcanzable como
hijo o hermano de un nodo real, así que reutilizar el valor es seguro."""

PAGE_SIZE = 4096
"""Tamaño de página fijo del motor, en bytes. Constante explícita y
documentada por módulo, tal como exige `../CLAUDE.md`."""

MAGIC = b"BPT1"
PAGE_FORMAT_VERSION = 1

# Cabecera común a toda página (nodo o metadatos):
#   magic(4s) + version(B) + page_type(B) + cell_count(H)
#   + free_space_offset(H) + side_pointer(I) + checksum(I)
HEADER_FORMAT = ">4sBBHHII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

CELL_POINTER_FORMAT = ">H"
CELL_POINTER_SIZE = struct.calcsize(CELL_POINTER_FORMAT)

# Payload fijo de la página de metadatos, justo después de la cabecera
# común: root_page_id(I) + page_count(I) + free_page_head(I).
METADATA_PAYLOAD_FORMAT = ">III"
METADATA_PAYLOAD_SIZE = struct.calcsize(METADATA_PAYLOAD_FORMAT)

# Una celda de hoja/interna lleva al menos su cabecera de longitudes; se
# exige que al menos 4 celdas quepan en una página para acotar el fan-out
# mínimo del árbol y evitar entradas que degeneren la estructura en una
# lista enlazada de páginas.
MIN_CELLS_PER_PAGE = 4
_USABLE_SPACE = PAGE_SIZE - HEADER_SIZE
MAX_ENTRY_SIZE = (_USABLE_SPACE // MIN_CELLS_PER_PAGE) - CELL_POINTER_SIZE
"""Tamaño máximo, en bytes, de una celda ya serializada (claves + valores
incluidos sus prefijos de longitud). Ver `EntryTooLargeError`."""


class PageType(IntEnum):
    """Tipo de contenido de una página persistida."""

    META = 1
    INTERNAL = 2
    LEAF = 3


@dataclass(frozen=True, slots=True)
class PageHeader:
    """Cabecera lógica de una página, ya decodificada (sin checksum).

    `side_pointer` tiene doble propósito según `page_type`: en una hoja es
    el `PageId` de la siguiente hoja en orden de clave (para range scans
    secuenciales sin subir por el árbol); en un nodo interno es el `PageId`
    del hijo más a la derecha (el que contiene las claves >= la última
    clave separadora de la página). En una página de metadatos no se usa
    (vale `NULL_PAGE_ID`).
    """

    page_type: PageType
    cell_count: int
    free_space_offset: int
    side_pointer: PageId

    def __post_init__(self) -> None:
        if self.cell_count < 0:
            raise ValueError(f"cell_count debe ser >= 0, recibido {self.cell_count}")
        if not (HEADER_SIZE <= self.free_space_offset <= PAGE_SIZE):
            raise ValueError(
                f"free_space_offset fuera de rango: {self.free_space_offset}"
            )
        if self.side_pointer < 0:
            raise ValueError(f"side_pointer debe ser >= 0, recibido {self.side_pointer}")


@dataclass(frozen=True, slots=True)
class LeafEntry:
    """Una entrada clave/valor decodificada de una página hoja."""

    key: bytes
    value: bytes

    def __post_init__(self) -> None:
        if len(self.key) == 0:
            raise ValueError("la clave no puede estar vacía")


@dataclass(frozen=True, slots=True)
class InternalEntry:
    """Una entrada (clave separadora, hijo izquierdo) de una página interna.

    El hijo más a la derecha de la página (subárbol de claves >= la última
    clave separadora) no se modela aquí: vive en `PageHeader.side_pointer`.
    """

    key: bytes
    left_child: PageId

    def __post_init__(self) -> None:
        if len(self.key) == 0:
            raise ValueError("la clave separadora no puede estar vacía")
        if self.left_child < 0:
            raise ValueError(f"left_child debe ser >= 0, recibido {self.left_child}")


@dataclass(frozen=True, slots=True)
class TreeMetadata:
    """Contenido de la página de metadatos (página 0)."""

    root_page_id: PageId
    page_count: int
    free_page_head: PageId = NULL_PAGE_ID
    """Cabeza de la lista enlazada de páginas libres reclamadas por merges
    de borrado; `NULL_PAGE_ID` mientras el árbol no haya liberado ninguna."""

    def __post_init__(self) -> None:
        if self.root_page_id < 0:
            raise ValueError(f"root_page_id debe ser >= 0, recibido {self.root_page_id}")
        if self.page_count < 1:
            raise ValueError(f"page_count debe ser >= 1, recibido {self.page_count}")
        if self.free_page_head < 0:
            raise ValueError(f"free_page_head debe ser >= 0, recibido {self.free_page_head}")


@dataclass(frozen=True, slots=True)
class BufferPoolConfig:
    """Configuración del buffer pool: tamaño máximo acotado y explícito."""

    capacity: int

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError(f"capacity debe ser >= 1, recibido {self.capacity}")


class BplusTreeError(Exception):
    """Error base de todo el módulo."""


class StorageIOError(BplusTreeError):
    """Fallo real de E/S al leer, escribir o hacer `fsync` del archivo de
    páginas (disco lleno, permisos, dispositivo desconectado...). Envuelve
    el `OSError` original sin perder contexto, tal como exige `../CLAUDE.md`:
    nunca un `except Exception: pass` genérico ante errores de disco."""


class PageCorruptionError(BplusTreeError):
    """Clase base para cualquier problema de integridad detectado al leer
    una página."""


class InvalidPageError(PageCorruptionError):
    """La cabecera de la página no empieza por `MAGIC` o su `page_type`
    no es uno de los valores reconocidos de `PageType`."""


class ChecksumMismatchError(PageCorruptionError):
    """La página tiene el tamaño esperado pero su checksum no coincide:
    corrupción real de bytes ya escritos, distinta de un truncamiento."""


class TruncatedPageError(PageCorruptionError):
    """El archivo de páginas es más corto de lo que exige el `PageId`
    solicitado: la última escritura quedó a mitad, típicamente por un
    proceso matado durante un `write_page`."""


class EntryTooLargeError(BplusTreeError):
    """Una clave o valor (ya serializados) exceden `MAX_ENTRY_SIZE` y no
    podrían caber nunca en una página, sin importar cuántas veces se
    intente hacer split."""


class KeyNotFoundError(BplusTreeError):
    """`delete` invocado sobre una clave que no existe en el árbol."""


class TreeInvariantError(BplusTreeError):
    """Comprobación defensiva de consistencia interna fallida (p.ej. un
    hijo referenciado tiene un `page_type` incompatible con el esperado).
    No debería ocurrir nunca en operación normal; indica un bug del propio
    motor, no un error del usuario de la API."""
