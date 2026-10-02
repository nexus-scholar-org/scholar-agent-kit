"""E3/T-110 regression guard: standalone installable dependency declarations.

scholar-agent-kit shipped NO ``[build-system]`` at all, so ``uv build --wheel .``
could not succeed, and it declared its sibling kits as BARE names resolvable only
through relative editable ``[tool.uv.sources]`` paths. Two required siblings
were not declared at all: ``scholar-verify-kit`` and ``nexus-scholar-harness``,
even though ``scholar_agent/server.py`` imported both at module level.

E3/T-110b / F-AGT-01 changed the second half of that: ``nexus-scholar-harness``
is no longer a runtime dependency at all. ``server.py`` now imports
``scholar_harness.recon`` and ``scholar_harness.recon.gates`` only INSIDE
``_ensure_harness_recon_loaded()``, so the recon names are module-level ``None``
until that function runs, and ``scholar_harness.orchestrator`` is imported
function-locally. The harness is therefore NOT declared, and this module guards
that absence explicitly -- see ``test_harness_is_not_a_runtime_dependency``.

This test locks in the fix: the build backend and wheel package are declared,
every declared sibling is a PEP 508 direct git reference pinned to a full 40-hex
canonical SHA, no relative path source may come back, and the harness must not
reappear as a runtime dependency. It is hermetic -- it reads the checked-in
``pyproject.toml`` and ``uv.lock`` only and never touches the network. When a
wheel has already been built into ``dist/``, the built METADATA is additionally
checked.
"""

from __future__ import annotations

import re
import tomllib
import zipfile
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
LOCKFILE = PYPROJECT.parent / "uv.lock"

# The name that must NOT be a runtime dependency (E3/T-110b / F-AGT-01).
HARNESS = "nexus-scholar-harness"

# scholar-<name>[@ git+https://github.com/nexus-scholar-org/<repo>@<40-hex>]
# with an optional PEP 508 extras suffix, e.g. scholar-pdf-kit[extract].
DIRECT_REF = re.compile(
    r"^[a-z0-9-]+(\[[a-z0-9,.-]+\])? @ git\+https://github\.com/nexus-scholar-org/[a-z0-9-]+@[0-9a-f]{40}$"
)

# The canonical main SHAs these siblings are pinned to. Tracked to the merged
# canonical mains after bib#1 / pdf#4 / graph#1 / rag#11 merged (T-110-REPIN);
# the structural DIRECT_REF regex and the full-40-hex requirement are unchanged,
# so this mirror tracks the pins rather than relaxing what the guard enforces.
# nexus-scholar-harness is deliberately ABSENT: it is lazy-guarded, not declared.
EXPECTED_SHA = {
    "scholar-protocol-kit": "4e10f25c25a1b150ce518348d211c7771683a9b7",
    "scholar-search-kit": "911d864fcb6a706d4c0339f80524a46f591e2cad",
    "scholar-bib-kit": "fbdd38ba25301e613621f18119623f04d2673dca",
    "scholar-pdf-kit": "3c024c37071b49265cfea6e713c1c9065e2a2cc0",
    "scholar-rag-kit": "033191eff967abf19023b258539c9a1422c8747f",
    "scholar-graph-kit": "646b84cec215ebd9b7b449448bca879492e78492",
    "scholar-verify-kit": "44a8d63cc0a11694bbcb7a355a53f73c3f93145c",
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


def test_harness_is_not_a_runtime_dependency() -> None:
    """The lazy-guard removed the need for a harness distribution.

    ``server.py`` imports ``scholar_harness.recon`` / ``.gates`` only inside
    ``_ensure_harness_recon_loaded()`` and ``scholar_harness.orchestrator`` only
    function-locally, so ``import scholar_agent.server`` must work with no harness
    installed. Re-declaring it would force every standalone consumer to pin the
    whole monorepo for a feature the kit does not need to import.
    """
    deps = _load()["project"]["dependencies"]
    declared = [d for d in deps if HARNESS in d]
    assert not declared, (
        f"{HARNESS} must not be a runtime dependency (lazy-guarded): {declared}"
    )

    extras = _load()["project"].get("optional-dependencies", {})
    extra_hits = [d for group in extras.values() for d in group if HARNESS in d]
    assert not extra_hits, f"{HARNESS} must not appear in any extra: {extra_hits}"

    assert HARNESS not in LOCKFILE.read_text(encoding="utf-8"), (
        f"{HARNESS} still present in uv.lock"
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

    # The wheel must not drag the harness distribution in with it.
    assert not [r for r in requires if HARNESS in r], (
        f"{HARNESS} must not be a wheel Requires-Dist: "
        f"{[r for r in requires if HARNESS in r]}"
    )

    git_requires = [r for r in requires if "git+" in r]
    assert len(git_requires) == len(EXPECTED_SHA), (
        f"expected {len(EXPECTED_SHA)} git direct refs, got {git_requires}"
    )
    for name, sha in EXPECTED_SHA.items():
        assert any(name in r and r.endswith(sha) for r in git_requires), (
            f"missing {name}@{sha} in METADATA"
        )
