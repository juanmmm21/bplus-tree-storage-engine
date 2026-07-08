"""Tests de humo de la CLI: cada subcomando invocado de punta a punta."""

from __future__ import annotations

from pathlib import Path

import pytest

from bplus_tree_storage_engine.__main__ import main


def test_put_get_roundtrip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "demo.db"

    assert main(["put", "--path", str(db), "--key", "alice", "--value", "30"]) == 0
    assert main(["get", "--path", str(db), "--key", "alice"]) == 0
    assert capsys.readouterr().out.strip() == "30"


def test_get_missing_key_prints_not_found_and_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "demo.db"
    main(["put", "--path", str(db), "--key", "alice", "--value", "30"])

    exit_code = main(["get", "--path", str(db), "--key", "nobody"])

    assert exit_code == 1
    assert "not found" in capsys.readouterr().out


def test_delete_missing_key_reports_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "demo.db"
    main(["put", "--path", str(db), "--key", "alice", "--value", "30"])

    exit_code = main(["delete", "--path", str(db), "--key", "nobody"])

    assert exit_code == 1
    assert "error:" in capsys.readouterr().err


def test_scan_prints_all_pairs_in_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "demo.db"
    for key, value in [("bob", "25"), ("alice", "30"), ("carol", "35")]:
        main(["put", "--path", str(db), "--key", key, "--value", value])

    assert main(["scan", "--path", str(db)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == ["alice\t30", "bob\t25", "carol\t35"]


def test_scan_bounded_range(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "demo.db"
    for i in range(10):
        main(["put", "--path", str(db), "--key", f"k{i}", "--value", str(i)])

    assert main(["scan", "--path", str(db), "--start", "k3", "--end", "k6"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == ["k3\t3", "k4\t4", "k5\t5"]


def test_info_reports_metadata(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "demo.db"
    main(["put", "--path", str(db), "--key", "alice", "--value", "30"])

    assert main(["info", "--path", str(db)]) == 0
    output = capsys.readouterr().out
    assert "root_page_id=" in output
    assert "page_count=" in output


def test_benchmark_runs_and_reports_throughput(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "bench.db"

    assert main(["benchmark", "--path", str(db), "--count", "200"]) == 0
    output = capsys.readouterr().out
    assert "writes:" in output
    assert "reads:" in output


def test_delete_then_get_reports_not_found(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "demo.db"
    main(["put", "--path", str(db), "--key", "alice", "--value", "30"])
    main(["delete", "--path", str(db), "--key", "alice"])

    exit_code = main(["get", "--path", str(db), "--key", "alice"])

    assert exit_code == 1
    assert "not found" in capsys.readouterr().out
