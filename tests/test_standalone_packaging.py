"""E3/T-110 regression guard: standalone installable dependency declarations.

scholar-agent-kit shipped NO ``[build-system]`` at all, so ``uv build --wheel .``
could not succeed, and it declared its sibling kits as BARE names resolvable only
through relative editable ``[tool.uv.sources]`` paths. Two required siblings
were not declared at all: ``scholar-verify-kit`` and ``nexus-scholar-harness``,
even though ``scholar_agent/server.py`` imports both at module level.

This test locks in the fix: the build backend and wheel package are declared,
every sibling is a PEP 508 direct git reference pinned to a full 40-hex
canonical SHA, and no relative path source may come back. It is hermetic -- it
reads the checked-in ``pyproject.toml`` only and never touches the network.
When a wheel has already been built into ``dist/``, the built METADATA is
additionally checked.
"""

from __future__ import annotations

import re
import tomllib
import zipfile
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# scholar-<name>[@ git+https://github.com/nexus-scholar-org/<repo>@<40-hex>]
# with an optional PEP 508 extras suffix, e.g. scholar-pdf-kit[extract].
DIRECT_REF = re.compile(
    r"^[a-z0-9-]+(\[[a-z0-9,.-]+\])? @ git\+https://github\.com/nexus-scholar-org/[a-z0-9-]+@[0-9a-f]{40}$"
)

# The canonical main SHAs recorded at E3/T-110 dispatch time.
EXPECTED_SHA = {
    "scholar-protocol-kit": "4e10f25c25a1b150ce518348d211c7771683a9b7",
    "scholar-search-kit": "911d864fcb6a706d4c0339f80524a46f591e2cad",
    "scholar-bib-kit": "0bf3cdd2d0ace6dc626f033871747a78681c6ddf",
    "scholar-pdf-kit": "0430ee40c491edbb055af4ab068637275aada476",
    "scholar-rag-kit": "d95469f6791c6cf6f608b5fdcf71483af45a3caa",
    "scholar-graph-kit": "4f0f4382ad4c940b2a3da838a12f1b47affd2b31",
    "scholar-verify-kit": "44a8d63cc0a11694bbcb7a355a53f73c3f93145c",
    "nexus-scholar-harness": "ddcefe5ed3208c9b65a4d65616da8093b02110f5",
}


def _load() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_wheel_build_is_configured() -> None:
    data = _load()
    assert data["build-system"]["build-backend"] == "hatchling.build"
    assert data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/scholar_agent"
    ]


def test_every_sibling_is_a_sha_pinned_direct_git_reference() -> None:
    deps = _load()["project"]["dependencies"]
    siblings = [d for d in deps if " git+" in d]
    assert siblings, "expected declared sibling requirements"

    for dep in siblings:
        assert DIRECT_REF.match(dep), f"not a SHA-pinned direct git reference: {dep!r}"

    declared = {re.split(r"[ @\[]", d, maxsplit=1)[0]: d for d in siblings}
    assert set(declared) == set(EXPECTED_SHA), (
        f"unexpected sibling set: {sorted(declared)}"
    )
    for name, sha in EXPECTED_SHA.items():
        assert declared[name].endswith(sha), (
            f"{name} is not pinned to canonical main {sha}: {declared[name]!r}"
        )


def test_no_bare_sibling_name_remains() -> None:
    """A bare ``scholar-pdf-kit`` cannot be resolved: no sibling is on PyPI."""
    deps = _load()["project"]["dependencies"]
    bare = [
        d
        for d in deps
        if re.match(r"^(nexus-)?scholar-[a-z0-9-]+(\[[a-z0-9,.-]+\])?$", d)
    ]
    assert not bare, f"bare, unresolvable sibling requirements: {bare}"


def test_no_relative_editable_sibling_source_can_come_back() -> None:
    sources = _load().get("tool", {}).get("uv", {}).get("sources", {})
    relative = {
        name: src
        for name, src in sources.items()
        if isinstance(src, dict) and "path" in src
    }
    assert not relative, (
        f"relative sibling sources break standalone installs: {relative}"
    )


def test_built_wheel_metadata_carries_the_direct_refs() -> None:
    wheels = sorted((PYPROJECT.parent / "dist").glob("*.whl"))
    if not wheels:
        import pytest

        pytest.skip("no built wheel in dist/; run `uv build --wheel .` first")

    archive = zipfile.ZipFile(wheels[-1])
    metadata_name = next(
        n for n in archive.namelist() if n.endswith(".dist-info/METADATA")
    )
    requires = [
        line.removeprefix("Requires-Dist: ")
        for line in archive.read(metadata_name).decode().splitlines()
        if line.startswith("Requires-Dist: ")
    ]
    git_requires = [r for r in requires if "git+" in r]
    assert len(git_requires) == len(EXPECTED_SHA), (
        f"expected {len(EXPECTED_SHA)} git direct refs, got {git_requires}"
    )
    for name, sha in EXPECTED_SHA.items():
        assert any(name in r and r.endswith(sha) for r in git_requires), (
            f"missing {name}@{sha} in METADATA"
        )
