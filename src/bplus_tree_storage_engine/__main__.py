"""CLI de demostración para bplus-tree-storage-engine.

Subcomandos:

    put         --path <db> --key <k> --value <v> [--capacity N]
    get         --path <db> --key <k> [--capacity N]
    delete      --path <db> --key <k> [--capacity N]
    scan        --path <db> [--start <k>] [--end <k>] [--capacity N]
    info        --path <db> [--capacity N]
    benchmark   --path <db> --count N [--capacity N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from bplus_tree_storage_engine.models import BplusTreeError, BufferPoolConfig
from bplus_tree_storage_engine.pipeline import BPlusTree, BufferPool, FilePageDevice

_DEFAULT_CAPACITY = 64


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bplus-tree-storage-engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    put = subparsers.add_parser("put", help="inserta o sobrescribe una clave")
    put.add_argument("--path", required=True, type=Path)
    put.add_argument("--key", required=True)
    put.add_argument("--value", required=True)
    put.add_argument("--capacity", type=int, default=_DEFAULT_CAPACITY)

    get = subparsers.add_parser("get", help="busca el valor de una clave")
    get.add_argument("--path", required=True, type=Path)
    get.add_argument("--key", required=True)
    get.add_argument("--capacity", type=int, default=_DEFAULT_CAPACITY)

    delete = subparsers.add_parser("delete", help="borra una clave")
    delete.add_argument("--path", required=True, type=Path)
    delete.add_argument("--key", required=True)
    delete.add_argument("--capacity", type=int, default=_DEFAULT_CAPACITY)

    scan = subparsers.add_parser("scan", help="recorre claves en orden ascendente")
    scan.add_argument("--path", required=True, type=Path)
    scan.add_argument("--start", default=None)
    scan.add_argument("--end", default=None)
    scan.add_argument("--capacity", type=int, default=_DEFAULT_CAPACITY)

    info = subparsers.add_parser("info", help="metadatos del árbol")
    info.add_argument("--path", required=True, type=Path)
    info.add_argument("--capacity", type=int, default=_DEFAULT_CAPACITY)

    benchmark = subparsers.add_parser("benchmark", help="benchmark básico de lectura/escritura")
    benchmark.add_argument("--path", required=True, type=Path)
    benchmark.add_argument("--count", type=int, default=10_000)
    benchmark.add_argument("--capacity", type=int, default=_DEFAULT_CAPACITY)

    return parser


def _open_tree(path: Path, capacity: int) -> BPlusTree:
    device = FilePageDevice(path)
    pool = BufferPool(device, BufferPoolConfig(capacity=capacity))
    return BPlusTree(pool)


def _cmd_put(args: argparse.Namespace) -> int:
    with _open_tree(args.path, args.capacity) as tree:
        tree.insert(args.key.encode(), args.value.encode())
    return 0


def _cmd_get(args: argparse.Namespace) -> int:
    with _open_tree(args.path, args.capacity) as tree:
        value = tree.search(args.key.encode())
    if value is None:
        print("not found")
        return 1
    print(value.decode(errors="replace"))
    return 0


def _cmd_delete(args: argparse.Namespace) -> int:
    with _open_tree(args.path, args.capacity) as tree:
        tree.delete(args.key.encode())
    return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    start = args.start.encode() if args.start is not None else None
    end = args.end.encode() if args.end is not None else None
    with _open_tree(args.path, args.capacity) as tree:
        for key, value in tree.scan(start=start, end=end):
            print(f"{key.decode(errors='replace')}\t{value.decode(errors='replace')}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    with _open_tree(args.path, args.capacity) as tree:
        metadata = tree.metadata
        print(f"root_page_id={metadata.root_page_id}")
        print(f"page_count={metadata.page_count}")
        print(f"free_page_head={metadata.free_page_head}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    keys = [f"bench-{i:08d}".encode() for i in range(args.count)]

    with _open_tree(args.path, args.capacity) as tree:
        started = time.perf_counter()
        for i, key in enumerate(keys):
            tree.insert(key, str(i).encode())
        tree.flush()
        write_elapsed = time.perf_counter() - started

        started = time.perf_counter()
        for key in keys:
            tree.search(key)
        read_elapsed = time.perf_counter() - started

    print(f"writes: {args.count} en {write_elapsed:.3f}s ({args.count / write_elapsed:.0f} ops/s)")
    print(f"reads:  {args.count} en {read_elapsed:.3f}s ({args.count / read_elapsed:.0f} ops/s)")
    return 0


_HANDLERS = {
    "put": _cmd_put,
    "get": _cmd_get,
    "delete": _cmd_delete,
    "scan": _cmd_scan,
    "info": _cmd_info,
    "benchmark": _cmd_benchmark,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = _HANDLERS[args.command]
    try:
        return handler(args)
    except BplusTreeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
