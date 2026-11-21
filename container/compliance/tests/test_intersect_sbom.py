# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the generators' --intersect-sbom (base-image attribution) mode.

--intersect-sbom is the inverse of --subtract-sbom: it keeps only the components
the baseline image already ships, so the delta NOTICES (/legal) and the
base-image NOTICES (/legal/base) together list the whole image without overlap.

Run from the repo root:

    python3 -m pytest container/compliance/tests/test_intersect_sbom.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from compliance.generators import __main__ as gen_main
from compliance.generators import common
from compliance.generators.common import Component

# CPU-only unit tests; markers are required by .ai/pytest-guidelines.md
# (lifecycle / test-type / hardware categories).
pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _comp(name: str, version: str) -> Component:
    return Component(ecosystem="python", name=name, version=version, spdx="MIT")


IMAGE = [
    _comp("torch", "2.14.0+cpu"),  # in the base image
    _comp("sglang_cpu", "0.5.23"),  # in the base image, underscore spelling
    _comp("ai-dynamo", "1.8.0"),  # added by Dynamo
    _comp("certifi", "2026.10.23"),  # upgraded by Dynamo: new version
]
BASELINE = {("torch", "2.14.0+cpu"), ("sglang-cpu", "0.5.23"), ("certifi", "2022.12.7")}


def _names(components: list[Component]) -> list[str]:
    return [c.name for c in components]


def test_subtract_and_intersect_partition_the_image():
    delta = common.subtract_baseline(IMAGE, set(BASELINE))
    base = common.subtract_baseline(IMAGE, common.IntersectKeys(BASELINE))

    assert _names(delta) == ["ai-dynamo", "certifi"]
    assert _names(base) == ["torch", "sglang_cpu"]
    assert sorted(_names(delta) + _names(base)) == sorted(_names(IMAGE))


def test_upgraded_package_is_attributed_to_dynamo_not_the_base():
    """A version Dynamo installed over the base's is in the delta, never in both."""
    base = common.subtract_baseline(IMAGE, common.IntersectKeys(BASELINE))
    assert "certifi" not in _names(base)


def test_empty_intersect_keeps_nothing():
    assert common.subtract_baseline(IMAGE, common.IntersectKeys()) == []
    assert common.subtract_baseline(IMAGE, set()) == IMAGE


def _sbom(tmp_path: Path, components: list[dict]) -> Path:
    path = tmp_path / "base.cdx.json"
    path.write_text(json.dumps({"components": components}))
    return path


def test_cli_rejects_subtract_and_intersect_together(tmp_path):
    sbom = _sbom(tmp_path, [{"name": "torch", "version": "1"}])
    with pytest.raises(SystemExit):
        gen_main.main(
            [
                "--ecosystem",
                "dpkg",
                "--output-dir",
                str(tmp_path),
                "--subtract-sbom",
                str(sbom),
                "--intersect-sbom",
                str(sbom),
            ]
        )


def test_cli_fails_on_an_empty_intersect_sbom(tmp_path):
    sbom = _sbom(tmp_path, [])
    rc = gen_main.main(
        [
            "--ecosystem",
            "dpkg",
            "--output-dir",
            str(tmp_path),
            "--intersect-sbom",
            str(sbom),
        ]
    )
    assert rc == 1


def test_cli_skips_native_under_intersect(tmp_path, monkeypatch):
    """Native entries are Dynamo's own builds; they must never land in /legal/base."""
    from compliance.generators import native

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("native generator ran under --intersect-sbom")

    monkeypatch.setattr(native, "generate", _must_not_run)
    sbom = _sbom(tmp_path, [{"name": "torch", "version": "1"}])
    native_yaml = tmp_path / "native.yaml"
    native_yaml.write_text("packages: []\n")

    rc = gen_main.main(
        [
            "--ecosystem",
            "native",
            "--native-yaml",
            str(native_yaml),
            "--native-image",
            "vllm-runtime",
            "--output-dir",
            str(tmp_path),
            "--intersect-sbom",
            str(sbom),
        ]
    )
    assert rc == 0
