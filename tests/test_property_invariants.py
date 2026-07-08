"""Tests de propiedades: secuencias aleatorias de operaciones (semilla fija)
verificando los invariantes estructurales del árbol B+ tras cada carga:
orden de claves dentro de cada página, particionamiento correcto de rangos
entre hijos, y que la cadena de hojas refleje exactamente las claves vivas.
"""

from __future__ import annotations

import random

from bplus_tree_storage_engine import BPlusTree, BufferPool, BufferPoolConfig, InMemoryPageDevice
from bplus_tree_storage_engine.models import NULL_PAGE_ID, PageId, PageType
from bplus_tree_storage_engine.pipeline import (
    _peek_page_type,
    decode_internal_page,
    decode_leaf_page,
)

_SEEDS = [1, 2, 3, 17, 42, 1000]
_UNIVERSE_SIZE = 400
_STEPS = 4000


def _assert_valid_subtree(
    tree: BPlusTree,
    page_id: PageId,
    lower_bound: bytes | None,
    upper_bound: bytes | None,
) -> None:
    """Recorre recursivamente el subárbol en `page_id` verificando que:

    - las entradas de cada página están en orden estrictamente ascendente
      por clave (sin duplicados dentro de la misma página);
    - toda clave del subárbol cae dentro de [`lower_bound`, `upper_bound`),
      límites que se van estrechando al descender, reflejando el
      particionamiento de rangos que las páginas internas codifican.
    """
    raw = tree._pool.get_page(page_id)  # noqa: SLF001 (inspección interna deliberada del test)

    if _peek_page_type(raw) is PageType.LEAF:
        leaf = decode_leaf_page(raw)
        keys = [entry.key for entry in leaf.entries]
        assert keys == sorted(keys), f"hoja {page_id} desordenada: {keys}"
        assert len(set(keys)) == len(keys), f"hoja {page_id} con claves duplicadas: {keys}"
        for key in keys:
            if lower_bound is not None:
                assert key >= lower_bound, f"clave {key!r} < límite inferior {lower_bound!r}"
            if upper_bound is not None:
                assert key < upper_bound, f"clave {key!r} >= límite superior {upper_bound!r}"
        return

    internal = decode_internal_page(raw)
    keys = [entry.key for entry in internal.entries]
    assert keys == sorted(keys), f"página interna {page_id} desordenada: {keys}"
    assert len(set(keys)) == len(keys), f"página interna {page_id} con separadores duplicados"

    previous_bound = lower_bound
    for entry in internal.entries:
        _assert_valid_subtree(tree, entry.left_child, previous_bound, entry.key)
        previous_bound = entry.key
    _assert_valid_subtree(tree, internal.side_pointer, previous_bound, upper_bound)


def _leaf_chain_keys(tree: BPlusTree) -> list[bytes]:
    page_id = tree._metadata.root_page_id  # noqa: SLF001
    while True:
        raw = tree._pool.get_page(page_id)  # noqa: SLF001
        if _peek_page_type(raw) is PageType.LEAF:
            break
        internal = decode_internal_page(raw)
        page_id = internal.entries[0].left_child if internal.entries else internal.side_pointer

    collected: list[bytes] = []
    while page_id != NULL_PAGE_ID:
        leaf = decode_leaf_page(tree._pool.get_page(page_id))  # noqa: SLF001
        collected.extend(entry.key for entry in leaf.entries)
        page_id = leaf.side_pointer
    return collected


def _run_random_workload(seed: int) -> tuple[BPlusTree, dict[bytes, bytes]]:
    rng = random.Random(seed)
    tree = BPlusTree(BufferPool(InMemoryPageDevice(), BufferPoolConfig(capacity=6)))
    reference: dict[bytes, bytes] = {}
    universe = [f"u-{i:05d}".encode() for i in range(_UNIVERSE_SIZE)]

    for step in range(_STEPS):
        key = rng.choice(universe)
        if rng.random() < 0.6:
            value = f"v-{step}".encode() * (1 + step % 3)
            tree.insert(key, value)
            reference[key] = value
        elif key in reference:
            tree.delete(key)
            del reference[key]

    return tree, reference


def test_structural_invariants_hold_after_random_workloads() -> None:
    for seed in _SEEDS:
        tree, reference = _run_random_workload(seed)
        root = tree._metadata.root_page_id  # noqa: SLF001
        _assert_valid_subtree(tree, root, lower_bound=None, upper_bound=None)

        for key, value in reference.items():
            assert tree.search(key) == value, f"seed={seed}"


def test_leaf_chain_matches_reference_after_random_workloads() -> None:
    for seed in _SEEDS:
        tree, reference = _run_random_workload(seed)
        assert _leaf_chain_keys(tree) == sorted(reference.keys()), f"seed={seed}"


def test_deleted_keys_are_truly_absent_after_random_workloads() -> None:
    for seed in _SEEDS:
        tree, reference = _run_random_workload(seed)
        universe = [f"u-{i:05d}".encode() for i in range(_UNIVERSE_SIZE)]
        for key in universe:
            if key not in reference:
                assert tree.search(key) is None, f"seed={seed}, clave {key!r}"
