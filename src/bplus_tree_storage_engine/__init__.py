"""bplus-tree-storage-engine: disk-backed B+Tree con buffer pool e indexación por páginas."""

from bplus_tree_storage_engine.models import (
    BplusTreeError,
    BufferPoolConfig,
    ChecksumMismatchError,
    EntryTooLargeError,
    InternalEntry,
    InvalidPageError,
    KeyNotFoundError,
    LeafEntry,
    PageCorruptionError,
    PageHeader,
    PageId,
    PageType,
    StorageIOError,
    TreeInvariantError,
    TreeMetadata,
    TruncatedPageError,
)

__all__ = [
    "BplusTreeError",
    "BufferPoolConfig",
    "ChecksumMismatchError",
    "EntryTooLargeError",
    "InternalEntry",
    "InvalidPageError",
    "KeyNotFoundError",
    "LeafEntry",
    "PageCorruptionError",
    "PageHeader",
    "PageId",
    "PageType",
    "StorageIOError",
    "TreeInvariantError",
    "TreeMetadata",
    "TruncatedPageError",
]

__version__ = "0.1.0"
