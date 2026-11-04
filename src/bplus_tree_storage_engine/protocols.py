"""Interfaces internas del motor: dispositivo de páginas y política de reemplazo.

Ninguna de las dos abstracciones se comparte por import directo con otros
repositorios del ecosistema (la integración real ocurre dentro de
`nanosql`): existen para permitir sustituir la implementación por defecto
(dispositivo respaldado por archivo, política LRU) por otra en tests o en
`nanosql` sin tocar la lógica del árbol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from bplus_tree_storage_engine.models import PageId


@runtime_checkable
class PageDevice(Protocol):
    """Dispositivo de almacenamiento de páginas de tamaño fijo.

    No sabe nada de árboles ni de claves: sólo lee, escribe y asigna
    páginas identificadas por `PageId`. La implementación por defecto es
    un archivo en disco; en tests se puede sustituir por una en memoria
    para no depender del sistema de archivos.
    """

    def read_page(self, page_id: PageId) -> bytes:
        """Devuelve los `PAGE_SIZE` bytes crudos de `page_id`.

        Lanza `TruncatedPageError` si el dispositivo es más corto de lo
        que exige `page_id` (crash a mitad de escritura de la última
        página) y `StorageIOError` ante cualquier otro fallo de E/S.
        """
        ...

    def write_page(self, page_id: PageId, data: bytes) -> None:
        """Escribe `data` (debe medir exactamente `PAGE_SIZE` bytes) en
        `page_id`, sobrescribiendo su contenido anterior."""
        ...

    def allocate_page(self) -> PageId:
        """Reserva y devuelve un `PageId` nuevo, sin inicializar su
        contenido todavía (el llamante debe escribirlo antes de leerlo)."""
        ...

    def page_count(self) -> int:
        """Número total de páginas asignadas, incluida la de metadatos."""
        ...

    def sync(self) -> None:
        """Fuerza `fsync` de todo lo escrito hasta ahora. Ninguna página
        se considera durable hasta que esta llamada retorna, siguiendo el
        invariante de durabilidad de `../CLAUDE.md`."""
        ...

    def close(self) -> None:
        """Libera el descriptor de archivo subyacente de forma explícita."""
        ...


@runtime_checkable
class AtomicPageDevice(PageDevice, Protocol):
    """Dispositivo capaz de escribir un lote de páginas de forma atómica
    frente a un crash: tras reabrir se ve el lote entero o nada de él."""

    def write_pages_atomically(self, pages: list[tuple[PageId, bytes]]) -> None:
        """Escribe y hace durable `pages` como una unidad. Al volver, todas
        las páginas están en su sitio y sincronizadas."""
        ...


@runtime_checkable
class ReplacementPolicy(Protocol):
    """Política de reemplazo del buffer pool (LRU u otra), desacoplada de
    la mecánica de fijar/liberar páginas del propio buffer pool."""

    def record_access(self, page_id: PageId) -> None:
        """Registra un acceso (lectura o escritura) a `page_id`, para que
        la política lo tenga en cuenta al elegir la próxima víctima."""
        ...

    def choose_victim(self) -> PageId | None:
        """Devuelve el `PageId` que debería desalojarse a continuación, o
        `None` si la política no tiene ninguna página candidata (p.ej.
        todas están ausentes de su seguimiento todavía)."""
        ...

    def forget(self, page_id: PageId) -> None:
        """Elimina `page_id` del seguimiento de la política, típicamente
        tras desalojarla del buffer pool."""
        ...
