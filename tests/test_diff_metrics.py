"""CLI regression tests for metric validation; no model dependencies needed."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diff_metrics.py"


def _metrics(value):
    return {"fog": {"bbox": {"AP": value}}}


def _run(tmp_path, clean, reference, *args):
    clean_path = tmp_path / "clean.json"
    reference_path = tmp_path / "reference.json"
    clean_path.write_text(json.dumps(clean), encoding="utf-8")
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(clean_path), str(reference_path), *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "metrics",
    [
        {"fog": {"bbox": {"AP": 31.25, "AP50": 45.0, "AP75": 21.0}}},
        {"fog": {"segm": {"AP": 28.0, "AP50": 42.0}}},
        {"fog": {"sem_seg": {"mIoU": 61.0, "fwIoU": 72.0, "pACC": 89.0}}},
        {"fog": {"bbox": {"AP": "31.25"}}},
    ],
    ids=["detection", "instance-segmentation", "semantic-segmentation", "numeric-string"],
)
def test_matching_metrics_pass_without_optional_heads(tmp_path, metrics):
    result = _run(tmp_path, metrics, metrics)

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert "+0.000" in result.stdout


def test_preserves_table_and_signed_three_decimal_deltas(tmp_path):
    clean = {"rain": {"bbox": {"AP": 10.125}}, "fog": {"bbox": {"AP": 9.875}}}
    reference = {domain: {"bbox": {"AP": 10.0}} for domain in clean}

    result = _run(tmp_path, clean, reference, "--atol", "0.125", "--rtol", "0")

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        f"{'domain':30s} {'metric':16s} {'clean':>8s} {'ref':>8s} {'Δ':>8s}",
        "-" * 74,
        f"{'fog':30s} {'bbox.AP':16s} {'9.875':>8s} {'10.000':>8s} {'-0.125':>8s}",
        f"{'rain':30s} {'bbox.AP':16s} {'10.125':>8s} {'10.000':>8s} {'+0.125':>8s}",
    ]


@pytest.mark.parametrize("clean", [9.0, 11.0])
def test_out_of_tolerance_metrics_fail(tmp_path, clean):
    result = _run(tmp_path, _metrics(clean), _metrics(10.0))

    assert result.returncode == 1
    assert "fog" in result.stderr
    assert "bbox.AP" in result.stderr
    assert "toleran" in result.stderr.lower()
    assert "bbox.AP" in result.stdout


def test_overflowing_difference_cannot_pass_with_overflowing_tolerance(tmp_path):
    result = _run(
        tmp_path, _metrics(1e308), _metrics(-1e308), "--atol", "8e307", "--rtol", "1.1"
    )

    assert result.returncode == 1
    assert "bbox.AP" in result.stderr
    assert "difference" in result.stderr.lower()


@pytest.mark.parametrize(
    "clean, reference, args, expected_status",
    [
        (10.00005, 10.0, (), 0),
        (0.0000005, 0.0, (), 0),
        (0.000002, 0.0, (), 1),
        (1.125, 1.0, ("--atol", "0.125", "--rtol", "0"), 0),
        (1.126, 1.0, ("--atol", "0.125", "--rtol", "0"), 1),
        (9.0, 8.0, ("--atol", "0", "--rtol", "0.125"), 0),
        (9.125, 8.0, ("--atol", "0", "--rtol", "0.125"), 1),
        (9.5, 8.0, ("--atol", "0.5", "--rtol", "0.125"), 0),
        (9.625, 8.0, ("--atol", "0.5", "--rtol", "0.125"), 1),
        (8.0, 9.125, ("--atol", "0", "--rtol", "0.125"), 0),
        (-9.0, -8.0, ("--atol", "0", "--rtol", "0.125"), 0),
        (10.0, 10.0, ("--atol", "0", "--rtol", "0"), 0),
        (10.00000001, 10.0, ("--atol", "0", "--rtol", "0"), 1),
    ],
)
def test_tolerance_configuration(tmp_path, clean, reference, args, expected_status):
    result = _run(tmp_path, _metrics(clean), _metrics(reference), *args)

    assert result.returncode == expected_status, result.stderr


@pytest.mark.parametrize("missing_side", ["clean", "reference"])
@pytest.mark.parametrize("missing_kind", ["metric", "head", "domain"])
def test_missing_metrics_fail(tmp_path, missing_side, missing_kind):
    complete = {"fog": {"bbox": {"AP": 10.0, "AP50": 20.0}, "sem_seg": {"mIoU": 30.0}}}
    incomplete = {
        "metric": {"fog": {"bbox": {"AP50": 20.0}, "sem_seg": {"mIoU": 30.0}}},
        "head": {"fog": {"sem_seg": {"mIoU": 30.0}}},
        "domain": {},
    }[missing_kind]
    clean, reference = (
        (incomplete, complete) if missing_side == "clean" else (complete, incomplete)
    )

    result = _run(tmp_path, clean, reference)

    assert result.returncode == 1
    assert "fog" in result.stderr
    assert "bbox.AP" in result.stderr
    assert "missing" in result.stderr.lower()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("invalid_side", ["clean", "reference", "both"])
def test_nonfinite_metrics_fail(tmp_path, invalid, invalid_side):
    clean = invalid if invalid_side in {"clean", "both"} else 10.0
    reference = invalid if invalid_side in {"reference", "both"} else 10.0

    result = _run(tmp_path, _metrics(clean), _metrics(reference))

    assert result.returncode == 1
    assert "fog" in result.stderr
    assert "bbox.AP" in result.stderr
    assert "finite" in result.stderr.lower()


@pytest.mark.parametrize("invalid", [None, "not-a-number", [], {}, True])
@pytest.mark.parametrize("invalid_side", ["clean", "reference", "both"])
def test_present_invalid_metrics_fail(tmp_path, invalid, invalid_side):
    clean = invalid if invalid_side in {"clean", "both"} else 10.0
    reference = invalid if invalid_side in {"reference", "both"} else 10.0

    result = _run(tmp_path, _metrics(clean), _metrics(reference))

    assert result.returncode == 1
    assert "fog" in result.stderr
    assert "bbox.AP" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "metrics",
    [
        {},
        {"fog": {}},
        {"fog": {"bbox": {}}},
        {"fog": {"bbox": {"unknown": 10.0}}},
        {"fog": {"unknown": {"AP": 10.0}}},
        {"fog": {}, "rain": {"bbox": {"AP": 10.0}}},
    ],
    ids=["no-domains", "empty-domain", "empty-head", "unknown-metric", "unknown-head", "one-empty-domain"],
)
def test_empty_comparisons_fail(tmp_path, metrics):
    result = _run(tmp_path, metrics, metrics)

    assert result.returncode == 1
    assert result.stderr
    assert "Traceback" not in result.stderr
    if metrics:
        assert "fog" in result.stderr


@pytest.mark.parametrize("empty_side", ["clean", "reference"])
def test_blank_json_file_fails_without_traceback(tmp_path, empty_side):
    paths = {side: tmp_path / f"{side}.json" for side in ("clean", "reference")}
    for side, path in paths.items():
        path.write_text("" if side == empty_side else json.dumps(_metrics(10.0)), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(paths["clean"]), str(paths["reference"])],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("option", ["--atol", "--rtol"])
@pytest.mark.parametrize("invalid", ["-1", "nan", "inf", "-inf", "text"])
def test_invalid_tolerances_are_usage_errors(tmp_path, option, invalid):
    result = _run(tmp_path, _metrics(10.0), _metrics(10.0), f"{option}={invalid}")

    assert result.returncode == 2
    assert option in result.stderr
    assert "Traceback" not in result.stderr


def test_missing_positional_arguments_are_usage_errors():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, check=False
    )

    assert result.returncode == 2
    assert "usage:" in result.stderr.lower()
