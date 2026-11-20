# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for capture_baseline_sbom.py's --prune-superseded with --image-family.

lmsysorg/sglang serves the -cu130-runtime, -xpu and -xeon baselines from one
repository. The xpu and xeon rows carry an image_family, so pruning on one
family's tag bump must not delete the others' rows.

Run from the repo root:

    python3 -m pytest container/compliance/tests/test_capture_prune.py -q
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from compliance.base_sboms import capture_baseline_sbom as cap

# CPU-only unit tests; markers are required by .ai/pytest-guidelines.md
# (lifecycle / test-type / hardware categories).
pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

SGLANG = "lmsysorg/sglang"


def _row(image: str, tag: str, sbom: str, family: str | None = None) -> dict:
    row = {
        "from_image": image,
        "from_tag": tag,
        "baseline_sbom": sbom,
        "platform": "linux/amd64",
    }
    if family:
        row["image_family"] = family
    return row


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[dict, Path]:
    """The lmsysorg/sglang rows as they are in manifest.json."""
    rows = [
        _row(SGLANG, "v0.5.23-cu130-runtime", "cuda@0230b7f2-amd64.cdx.json"),
        _row(SGLANG, "v0.5.23-xpu", "sglang@f5627c5e-amd64.cdx.json", "sglang-xpu"),
        _row(SGLANG, "v0.5.23-xeon", "sglang@f22c18d1-amd64.cdx.json", "sglang-xeon"),
    ]
    for r in rows:
        (tmp_path / r["baseline_sbom"]).write_text("{}")
    return {"entries": rows}, tmp_path


def _capture(manifest: dict, corpus_dir: Path, entry: dict) -> list[str]:
    cap.upsert_entry(manifest, entry)
    return cap.prune_superseded(manifest, entry, corpus_dir)


def _tags(manifest: dict) -> list[str]:
    return sorted(e["from_tag"] for e in manifest["entries"])


def test_family_bump_prunes_only_its_own_family(corpus):
    manifest, d = corpus
    new = _row(SGLANG, "v0.5.24-xpu", "sglang@0badc0de-amd64.cdx.json", "sglang-xpu")

    removed = _capture(manifest, d, new)

    assert removed == ["sglang@f5627c5e-amd64.cdx.json"]
    assert _tags(manifest) == ["v0.5.23-cu130-runtime", "v0.5.23-xeon", "v0.5.24-xpu"]
    assert (d / "cuda@0230b7f2-amd64.cdx.json").is_file()
    assert (d / "sglang@f22c18d1-amd64.cdx.json").is_file()
    assert not (d / "sglang@f5627c5e-amd64.cdx.json").exists()


def test_cuda_bump_without_family_keeps_family_rows(corpus, caplog):
    manifest, d = corpus
    new = _row(SGLANG, "v0.5.24-cu130-runtime", "cuda@12345678-amd64.cdx.json")

    with caplog.at_level(logging.WARNING):
        removed = _capture(manifest, d, new)

    assert removed == ["cuda@0230b7f2-amd64.cdx.json"]
    assert _tags(manifest) == ["v0.5.23-xeon", "v0.5.23-xpu", "v0.5.24-cu130-runtime"]
    assert "sglang-xeon, sglang-xpu" in caplog.text


def test_xeon_recapture_keeps_xpu_and_cuda(corpus):
    """The case that motivated image_family: adding the Xeon baseline beside XPU."""
    manifest, d = corpus
    xeon = manifest["entries"].pop()

    assert _capture(manifest, d, xeon) == []
    assert len(manifest["entries"]) == 3


def test_single_family_image_without_family_is_unchanged(tmp_path):
    """TRT-LLM's auto-upgrade CI passes no --image-family; that must keep working."""
    image = "nvcr.io/nvidia/tensorrt-llm/release"
    old = _row(image, "1.4.0rc3", "release@aa9c10c5-amd64.cdx.json")
    (tmp_path / old["baseline_sbom"]).write_text("{}")
    manifest = {"entries": [old]}
    new = _row(image, "1.4.0rc4", "release@12345678-amd64.cdx.json")

    assert _capture(manifest, tmp_path, new) == ["release@aa9c10c5-amd64.cdx.json"]
    assert _tags(manifest) == ["1.4.0rc4"]
    assert "image_family" not in manifest["entries"][0]


def test_cli_passes_image_family_to_capture(monkeypatch):
    seen: dict = {}

    def _fake_capture(**kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(cap, "capture", _fake_capture)
    argv = ["--from", "a:1", "--baseline", "a:1", "--image-family", "sglang-xpu"]

    assert cap.main(argv) == 0
    assert seen["image_family"] == "sglang-xpu"
