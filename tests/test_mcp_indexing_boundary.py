"""WP01-E3 declared MCP capability boundary for RAG indexing (E3-008/NEG-040/041).

Packet E3 section 9.1 decides that the **indexing** surface is declared
unsupported over MCP and answers every request with one deterministic
``UNSUPPORTED_CAPABILITY`` envelope. This module is the executable form of that
decision, and it asserts the properties the decision rests on:

* the capability registry declares ``rag_indexing`` with
  ``mcp_supported=False``, owning surfaces ``("API", "CLI")``, owner
  ``nexus-scholar-org/scholar-rag-kit``, the stable non-retryable code
  ``UNSUPPORTED_CAPABILITY``, and ``operation="rag_index"``;
* the boundary is *observable*: the tool stays registered, is discoverable
  through the public ``list_tools`` API, is reachable through
  ``MCPServer.call_tool``, and is listed in ``scholar-agent --help`` -- so tool
  discovery, help text, and the capability registry agree (required
  behavior 5);
* an indexing-shaped request returns the standard operation envelope,
  unconditionally: the envelope is byte-identical across a payload-mutation
  matrix (absent/empty/garbage/hostile args, and args that *claim* a store
  path, workspace identity, accepted parent, or journal), across working
  directories, and across varying local workspace contents including a decoy
  ``chroma_db`` directory that must never be opened (``E3-NEG-040`` and
  ``E3-NEG-041``: a client cannot discover a different answer by switching
  transport);
* the refusal performs **no** indexing, store opening, journal discovery, or
  filesystem mutation, and never a free-text-only error;
* the legacy ``ScholarIndexer`` invocation path is gone from the adapter rather
  than forwarded to, and mutation tests fail loudly if it is ever reintroduced;
* agent-kit contains no second indexing implementation -- this change adds a
  *declaration*, never a second indexer;
* the advertised ``scholar-rag index`` alternative is checked against the
  canonical T-90 command's own signature, so a caller cannot be redirected to a
  flag the owning command does not accept;
* ``E1``/``E2`` declarations are unchanged by this addition.

Every test is offline, hermetic, and deterministic: no network, no real vector
store, no indexing, and no writing outside ``tmp_path``. Zero-I/O evidence is
layered the way the E1/E2 boundary suites layer it -- runtime thread-scoped
tripwires over network entry points and filesystem mutators, tripwires on the
vector-store clients and on the legacy indexer, a sandboxed CWD/workspace
snapshot, and a static reachability walk over the rejection path's own bytecode
(including nested code objects). The static walk is the substantive zero-I/O
proof; it shows no I/O-capable or indexing symbol is even *reachable*.
"""

from __future__ import annotations

import ast
import asyncio
import builtins
import contextlib
import dis
import importlib
import inspect
import json
import textwrap
import threading
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any
from unittest.mock import patch

import pytest

from scholar_agent import capabilities as caps
from scholar_agent import server as server_module
from scholar_agent.capabilities import (
    PDF_ACQUISITION,
    PDF_EXTRACTION,
    RAG_INDEXING,
    UNSUPPORTED_CAPABILITY,
    CapabilityDeclaration,
    UnknownCapabilityError,
    get_capability,
    unsupported_capability_envelope,
    unsupported_capability_envelope_json,
)
from scholar_agent.server import main, mcp, nexus_rag_index

# --------------------------------------------------------------------------- #
# Module-wide CWD sandbox
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _sandbox_cwd(tmp_path, monkeypatch):
    """Run *every* test with a sandboxed working directory.

    Without this, a relative-path read or write escaping the tripwires would
    land in the process CWD -- i.e. the repo root -- instead of a tmp dir. The
    CWD-independence tests below override this with their own ``chdir``.
    """
    monkeypatch.chdir(tmp_path)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

TOOL_NAME = "nexus_rag_index"
REJECTION_CODE = "UNSUPPORTED_CAPABILITY"
REJECTION_OPERATION = "rag_index"

#: The 25 MCP tools that existed before E3, i.e. the same set E2 pinned. The E3
#: boundary must *convert* an existing tool, not add or drop one: discovery has
#: to keep showing it, so the count must stay 25.
PRE_EXISTING_TOOLS = frozenset(
    {
        "nexus_protocol_compile",
        "nexus_protocol_validate",
        "nexus_protocol_render_criteria",
        "nexus_discover",
        "nexus_dedup",
        "nexus_screen",
        "nexus_screen_llm",
        "nexus_pipeline_run",
        "nexus_extract_pdf",
        "nexus_rag_index",
        "nexus_rag_query",
        "nexus_rag_synthesize",
        "nexus_matrix_extract",
        "nexus_graph_build",
        "nexus_bib_clean",
        "nexus_screen_reconcile",
        "nexus_verify_claims",
        "nexus_verify_phase4",
        "nexus_critique_methodology",
        "nexus_graph_narrative",
        "recon_probe",
        "recon_distill",
        "recon_delta",
        "nexus_pdf_acquire",
        "nexus_pdf_extraction",
    }
)

#: Exact envelope shape. "No artifacts" is explicit, and the rejection carries no
#: index/manifest/audit/success bookkeeping.
ENVELOPE_KEYS = {"operation", "status", "artifacts", "warnings", "errors"}
ERROR_KEYS = {"code", "message", "retryable", "details"}
DETAIL_KEYS = {
    "capability",
    "mcp_supported",
    "owning_surfaces",
    "owner",
    "alternatives",
    "reference",
}

#: Network entry points that would constitute "provider transport".
_NETWORK_TARGETS = (
    ("socket", "create_connection"),
    ("socket", "getaddrinfo"),
    ("socket", "gethostbyname"),
    ("urllib.request", "urlopen"),
)

#: Filesystem mutators: any of these on the rejection path would mean the
#: adapter staged, wrote, replaced, or removed something.
_FS_TARGETS = (
    ("pathlib", "Path.mkdir"),
    ("pathlib", "Path.touch"),
    ("pathlib", "Path.write_text"),
    ("pathlib", "Path.write_bytes"),
    ("pathlib", "Path.unlink"),
    ("pathlib", "Path.rename"),
    ("pathlib", "Path.replace"),
    ("os", "mkdir"),
    ("os", "makedirs"),
    ("os", "remove"),
    ("os", "rename"),
    ("os", "replace"),
    ("shutil", "copyfile"),
    ("shutil", "copy"),
    ("shutil", "move"),
    ("shutil", "rmtree"),
    ("tempfile", "mkstemp"),
    ("tempfile", "mkdtemp"),
    ("tempfile", "NamedTemporaryFile"),
)

#: Vector-store entry points. Reaching one of these means a backend was opened,
#: which the refusal must never do.
_STORE_TARGETS = (
    ("chromadb", "PersistentClient"),
    ("chromadb", "HttpClient"),
    ("chromadb", "Client"),
)

#: I/O-capable modules that must not be reachable from the rejection path.
_IO_MODULES = frozenset(
    {
        "aiohttp",
        "http",
        "httpx",
        "io",
        "os",
        "pathlib",
        "requests",
        "shutil",
        "socket",
        "sqlite3",
        "subprocess",
        "tempfile",
        "urllib",
    }
)

#: I/O-capable callables that must not be reachable either.
_IO_CALLABLES = frozenset(
    {
        "connect",
        "connect_ex",
        "create_connection",
        "getaddrinfo",
        "makedirs",
        "mkdir",
        "mkdtemp",
        "mkstemp",
        "open",
        "read_bytes",
        "read_text",
        "remove",
        "rename",
        "replace",
        "touch",
        "unlink",
        "urlopen",
        "write_bytes",
        "write_text",
    }
)

#: Pure stdlib modules the envelope builder may legitimately reach.
_ALLOWED_MODULES = frozenset({"json"})

#: Builtins the pure envelope builder may legitimately reach (pure by
#: definition: no filesystem, network, or process effect).
_ALLOWED_BUILTINS = frozenset(
    {
        "KeyError",
        "ValueError",
        "dict",
        "enumerate",
        "isinstance",
        "list",
        "set",
        "sorted",
        "str",
        "tuple",
    }
)

_MISSING = object()

#: Symbols that would mean a second indexing implementation is reachable.
_INDEXING_SYMBOLS = (
    "ScholarIndexer",
    "index_directory",
    "index_workspace",
    "IndexServiceRequest",
    "IndexServiceResult",
    "ScholarRetriever",
    "GroundedSynthesisEngine",
)

#: E1's and E2's declarations, hardcoded so that adding E3 fails if a future
#: edit re-interprets an existing capability instead of adding a third one
#: (the E2 "no silent broaden" rule, re-applied to E3).
E1_EXPECTED_ACQUISITION = {
    "name": "pdf_acquisition",
    "owner": "nexus-scholar-org/scholar-pdf-kit",
    "owning_surfaces": ("API", "CLI"),
    "mcp_supported": False,
    "rejection_code": "UNSUPPORTED_CAPABILITY",
    "rejection_operation": "acquire_pdf",
    "reference": "docs/architecture/wp01_packet_e1_acquired_document_handoff.md#4.7",
    "alternatives": (
        "`scholar-pdf acquire` CLI (e.g. `uv run scholar-pdf acquire <config.json>`)",
        "Python `scholar_pdf.acquisition` API (acquisition request/outcome models)",
    ),
}
E2_EXPECTED_EXTRACTION = {
    "name": "pdf_extraction",
    "owner": "nexus-scholar-org/scholar-pdf-kit",
    "owning_surfaces": ("API", "CLI"),
    "mcp_supported": False,
    "rejection_code": "UNSUPPORTED_CAPABILITY",
    "rejection_operation": "extract_pdf",
    "reference": "docs/architecture/wp01_packet_e2_extracted_text_handoff.md#9",
    "alternatives": (
        (
            "`scholar-pdf extract-run` CLI (e.g. `uv run scholar-pdf extract-run "
            "<config.json> --audit-logger <path-to-log_event.py>`)"
        ),
        (
            "Python `scholar_pdf.extraction.PDFExtractionService` API "
            "(parent-bound extraction request/outcome service)"
        ),
    ),
}

#: Markers belonging to the E3 limb alone. Their absence from the acquisition
#: and extraction envelopes is direct textual proof that E3 did not rewrite them.
_E3_ONLY_MARKERS = (
    "rag_index",
    "rag_indexing",
    "E3 RAG indexing",
    "scholar-rag index",
    "scholar_rag.index_service",
    "wp01_packet_e3_implementation_handoff",
)

#: Strings that must never appear as the tool's answer. The retired tool returned
#: exactly these, so their continued presence would mean a free-text-only
#: boundary has crept back in.
_FREE_TEXT_MARKERS = (
    "Successfully indexed",
    "Error during indexing",
    "structural chunks",
)

# --------------------------------------------------------------------------- #
# Payload-mutation matrix (required behavior 2 / E3-NEG-040, E3-NEG-041)
#
# None of these values is read: the rejection is unconditional. The matrix
# exists to prove that, and to prove that a payload *claiming* a store path,
# workspace identity, accepted parent, or journal changes nothing.
# --------------------------------------------------------------------------- #

#: The shape the retired tool advertised.
INDEXING_REQUEST_ARGS: dict[str, Any] = {
    "docs_dir": "extracted",
    "db_path": "./chroma_db",
    "workspace_id": "SCI-000001",
}

#: A payload that aggressively *claims* authority-bearing state: a real-looking
#: store path, a verified workspace identity, an accepted parent record, and a
#: journal path. None may be honoured, echoed, or discovered.
AUTHORITY_CLAIMING_ARGS: dict[str, Any] = {
    "docs_dir": "/srv/workspaces/demo/extracted",
    "db_path": "/srv/workspaces/demo/chroma_db",
    "bib_file": "/srv/workspaces/demo/references.bib",
    "workspace_id": "SCI-000001",
}

HOSTILE_ARGS: dict[str, Any] = {
    "docs_dir": "../../../outside-the-workspace",
    "db_path": "\\\\evil\\share\\chroma",
    "bib_file": "; DROP TABLE index; --",
    "workspace_id": "",
}

GARBAGE_ARGS: dict[str, Any] = {
    "docs_dir": "\x00\x01 garbage \ud83d\ude80",
    "db_path": "chroma_db",
}

ARGUMENT_CASES: dict[str, dict[str, Any]] = {
    "no_optional_arguments": {"docs_dir": "extracted"},
    "indexing_request_shape": INDEXING_REQUEST_ARGS,
    "authority_claiming_payload": AUTHORITY_CLAIMING_ARGS,
    "invalid_and_hostile_arguments": HOSTILE_ARGS,
    "garbage_payload": GARBAGE_ARGS,
    "empty_docs_dir": {"docs_dir": ""},
    "none_valued_optionals": {"docs_dir": "x", "db_path": None, "workspace_id": None},
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _envelope(raw: str) -> dict[str, Any]:
    """Parse a tool result as the standard JSON operation envelope."""
    parsed = json.loads(raw)
    assert isinstance(parsed, dict), "envelope must be a JSON object"
    return parsed


def _declaration() -> CapabilityDeclaration:
    return get_capability(RAG_INDEXING)


def _registered_tool_names() -> set[str]:
    return {tool.name for tool in mcp._tool_manager.list_tools()}


def _body_of(func: Callable[..., Any]) -> str:
    """Executable body of a function, with its docstring removed."""
    return textwrap.dedent(inspect.getsource(func).split('"""', 2)[-1]).strip()


def _source_files() -> list[Path]:
    """Every module in this kit's ``scholar_agent`` package."""
    package_dir = Path(server_module.__file__).parent
    return sorted(package_dir.glob("*.py"))


class _Tripwire:
    """Records I/O attempts and raises for the calling thread only.

    Background threads (e.g. vendor telemetry started by an earlier test) are
    recorded and ignored, so the assertion is precisely "the rejection path
    performed no I/O" rather than "the process performed no I/O anywhere".
    """

    def __init__(self) -> None:
        self.events: list[tuple[int, str]] = []
        self._owner_thread = threading.get_ident()

    def hook(self, label: str) -> Callable[..., Any]:
        def _record(*args: Any, **kwargs: Any) -> Any:
            current = threading.get_ident()
            self.events.append((current, label))
            if current == self._owner_thread:
                raise AssertionError(f"rejection path performed I/O via {label}")
            return None

        return _record

    def owner_events(self) -> list[str]:
        return [label for tid, label in self.events if tid == self._owner_thread]


@contextlib.contextmanager
def _forbid_io() -> Iterator[_Tripwire]:
    """Tripwire network entry points, filesystem mutators, and store clients."""
    wire = _Tripwire()
    with contextlib.ExitStack() as stack:
        for module_name, attr in _NETWORK_TARGETS + _FS_TARGETS + _STORE_TARGETS:
            target = importlib.import_module(module_name)
            owner, _, leaf = attr.rpartition(".")
            holder = getattr(target, owner) if owner else target
            stack.enter_context(
                patch.object(holder, leaf, wire.hook(f"{module_name}.{attr}"))
            )
        yield wire


def _snapshot(root: Path) -> list[tuple[str, bool, int]]:
    """Sorted (relative path, is_dir, size) snapshot of a directory tree."""
    if not root.exists():
        return []
    entries: list[tuple[str, bool, int]] = []
    for path in sorted(root.rglob("*")):
        size = 0 if path.is_dir() else path.stat().st_size
        entries.append((str(path.relative_to(root)), path.is_dir(), size))
    return entries


def _instructions(code: Any) -> list[Any]:
    """All bytecode instructions of ``code`` including nested code objects."""
    found = list(dis.get_instructions(code))
    for const in code.co_consts:
        if hasattr(const, "co_code"):
            found.extend(_instructions(const))
    return found


def _referenced_globals(func: Callable[..., Any]) -> set[str]:
    """Global/local names referenced by a function's bytecode."""
    return {
        instruction.argval
        for instruction in _instructions(func.__code__)
        if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
        and isinstance(instruction.argval, str)
    }


def _referenced_imports(func: Callable[..., Any]) -> set[str]:
    """Modules imported anywhere on a function's path (incl. nested scopes).

    Scanning ``IMPORT_NAME`` closes the hole where a function-local import
    (``from pathlib import Path``) would otherwise evade a LOAD_GLOBAL walk.
    """
    modules: set[str] = set()
    pending: list[Any] = [func.__code__]
    while pending:
        code = pending.pop()
        for instruction in dis.get_instructions(code):
            if instruction.opname == "IMPORT_NAME" and isinstance(
                instruction.argval, str
            ):
                modules.add(instruction.argval.split(".")[0])
        for const in code.co_consts:
            if hasattr(const, "co_code"):
                pending.append(const)
    return modules


def _is_kit_function(obj: Any) -> bool:
    return inspect.isfunction(obj) and str(getattr(obj, "__module__", "")).startswith(
        "scholar_agent"
    )


def _reachable_globals(func: Callable[..., Any]) -> dict[str, Any]:
    """Resolve every symbol reachable from ``func`` through kit functions.

    Walks kit-local functions transitively (the rejection path delegates to the
    pure envelope builder) and returns ``name -> resolved object``.
    """
    resolved: dict[str, Any] = {}
    seen: set[int] = set()
    pending: list[Callable[..., Any]] = [func]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for name in _referenced_globals(current):
            if name in resolved:
                continue
            if name in current.__globals__:
                target = current.__globals__[name]
            else:
                target = getattr(builtins, name, _MISSING)
            assert target is not _MISSING, f"{name} is referenced but not bound"
            resolved[name] = target
            if _is_kit_function(target):
                pending.append(target)
    return resolved


class _ExplodingIndexer:
    """Stand-in for the legacy indexer that fails loudly if it is reached.

    Both construction and ``index_directory`` raise, so a test that patches the
    real ``scholar_rag.indexer.ScholarIndexer`` with this class proves the
    refusal never forwards to it: any forwarding attempt surfaces as an error
    rather than as a passing test.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "legacy ScholarIndexer was constructed: the E3 refusal must not "
            "forward to a legacy indexing path"
        )

    def index_directory(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError(
            "legacy ScholarIndexer.index_directory was called: the E3 refusal "
            "must not perform indexing"
        )


# --------------------------------------------------------------------------- #
# (a) Capability registry declaration
# --------------------------------------------------------------------------- #


def test_e3_neg_040_registry_declares_rag_indexing_mcp_unsupported():
    """E3-008: the registry declares rag_indexing unsupported on MCP."""
    declaration = _declaration()
    assert declaration.name == "rag_indexing"
    assert declaration.mcp_supported is False
    assert RAG_INDEXING in caps.CAPABILITIES
    assert caps.RAG_INDEXING == "rag_indexing"
    assert sorted(caps.CAPABILITIES) == [
        "pdf_acquisition",
        "pdf_extraction",
        "rag_indexing",
    ]


def test_e3_neg_040_registry_owner_is_the_rag_kit_over_api_and_cli():
    """scholar-rag-kit owns indexing; this MCP surface is not an owner."""
    declaration = _declaration()
    assert set(declaration.owning_surfaces) == {"API", "CLI"}
    assert caps.MCP_SURFACE == "MCP"
    assert caps.MCP_SURFACE not in declaration.owning_surfaces
    assert declaration.owner == caps.RAG_OWNER == "nexus-scholar-org/scholar-rag-kit"
    # A different canonical owner from the two PDF capabilities: this is a third
    # domain service, not a re-labelling of either PDF one.
    assert declaration.owner != caps.ACQUISITION_OWNER
    assert declaration.owner != caps.EXTRACTION_OWNER


def test_e3_neg_040_registry_rejection_code_reuses_the_one_constant():
    """One vocabulary: E3 reuses UNSUPPORTED_CAPABILITY, it does not fork it."""
    declaration = _declaration()
    assert declaration.rejection_code == REJECTION_CODE == UNSUPPORTED_CAPABILITY
    assert declaration.rejection_operation == REJECTION_OPERATION == "rag_index"
    # Distinguishable from the E1/E2 operations from the envelope alone.
    assert declaration.rejection_operation not in {
        caps.ACQUIRE_PDF_OPERATION,
        caps.EXTRACT_PDF_OPERATION,
    }
    for other in (get_capability(PDF_ACQUISITION), get_capability(PDF_EXTRACTION)):
        assert other.rejection_code == UNSUPPORTED_CAPABILITY


def test_e3_neg_040_registry_names_the_t90_api_and_cli_alternatives():
    """Required behavior 4: the T-90 surfaces are named, CLI first then API."""
    declaration = _declaration()
    assert len(declaration.alternatives) == 2
    cli, api = declaration.alternatives
    assert "scholar-rag index" in cli
    assert "scholar_rag" not in cli
    assert "scholar_rag.index_service" in api
    assert "index_workspace" in api
    assert "IndexServiceRequest" in api
    assert "IndexServiceResult" in api
    # The alternatives must be the *supported* surfaces, not the legacy helper
    # Packet E3 section 9.1 found non-authoritative.
    assert "indexer" not in cli
    assert "ScholarIndexer" not in cli + api
    assert declaration.reference == (
        "docs/architecture/wp01_packet_e3_implementation_handoff.md#9.1"
    )


def test_e3_neg_040_registry_projection_is_json_serialisable():
    """as_dict() renders the third declaration without extra fields."""
    projection = _declaration().as_dict()
    assert json.loads(json.dumps(projection)) == projection
    assert projection["mcp_supported"] is False
    assert projection["owning_surfaces"] == ["API", "CLI"]
    assert projection["alternatives"][0].index("scholar-rag index") >= 0


def test_e3_neg_040_registry_is_immutable_and_rejects_unknown_capability():
    """The registry cannot be mutated, and an unknown name raises -- never a
    silent "unsupported" answer (a misspelled capability is a client bug)."""
    with pytest.raises(TypeError):
        caps.CAPABILITIES["rag_indexing_v2"] = _declaration()  # type: ignore[index]
    for misspelling in (
        "rag_index",
        "RAG_INDEXING",
        "rag-indexing",
        "indexing",
        "rag_indexing ",
        "",
    ):
        with pytest.raises(UnknownCapabilityError):
            get_capability(misspelling)


def test_e3_neg_040_no_rejection_envelope_for_a_supported_capability():
    """A capability MCP actually serves has no rejection envelope (fail-closed)."""
    declaration = _declaration()
    served = replace(declaration, mcp_supported=True)
    assert served.mcp_supported is True
    with _scratch_registry(served):
        with pytest.raises(ValueError, match="declared unsupported"):
            caps.unsupported_capability_envelope(RAG_INDEXING)
        with pytest.raises(ValueError, match="declared unsupported"):
            caps.unsupported_capability_envelope_json(RAG_INDEXING)
    # The real registry is untouched by the scratch substitution.
    assert caps.CAPABILITIES[RAG_INDEXING] is declaration
    assert caps.CAPABILITIES[RAG_INDEXING].mcp_supported is False


@contextlib.contextmanager
def _scratch_registry(declaration: CapabilityDeclaration) -> Iterator[None]:
    """Temporarily substitute the declaration registry the builder consults.

    ``unsupported_capability_envelope`` reads ``capabilities.CAPABILITIES``, so
    the fail-closed guard is exercised by registering a *copy* that claims MCP
    support. The substitution is undone on exit, including on failure.
    """
    original = caps.CAPABILITIES
    caps.CAPABILITIES = MappingProxyType({declaration.name: declaration})
    try:
        yield
    finally:
        caps.CAPABILITIES = original


# --------------------------------------------------------------------------- #
# (b) The boundary is registered, discoverable, and documented
# --------------------------------------------------------------------------- #


def test_e3_neg_040_indexing_tool_is_registered_on_mcp_server():
    """E3 does not remove the tool: an absent tool is not a boundary."""
    assert server_module.mcp is mcp
    assert TOOL_NAME in _registered_tool_names()
    assert callable(nexus_rag_index)
    assert nexus_rag_index.__module__ == "scholar_agent.server"


def test_e3_neg_041_registry_discovery_and_help_all_agree():
    """Required behavior 5: registry, discovery, and help describe one boundary."""
    help_text = _help_text()
    tools = asyncio.run(mcp.list_tools())
    tool = next(t for t in tools if t.name == TOOL_NAME)

    # Registry says: indexing, owned by scholar-rag-kit, not MCP-served.
    declaration = _declaration()
    assert declaration.name == "rag_indexing"
    assert declaration.mcp_supported is False
    assert declaration.owner == "nexus-scholar-org/scholar-rag-kit"

    # Discovery says: present, and its description states the same boundary.
    assert tool.description
    flat_description = " ".join(tool.description.lower().split())
    assert "declared unsupported" in flat_description
    assert "not available through mcp" in flat_description
    assert "rag_indexing" in tool.description

    # Help says: present, and names the same code and the same alternatives.
    assert TOOL_NAME in help_text
    assert "DECLARED UNSUPPORTED" in help_text
    assert REJECTION_CODE in help_text
    assert "rag_index" in help_text
    assert "scholar-rag index" in help_text
    assert "scholar_rag.index_service" in help_text


def test_e3_neg_040_indexing_tool_is_discoverable_with_indexing_shaped_fields():
    """The request shape stays exposed for discoverability, as with E1/E2."""
    tools = asyncio.run(mcp.list_tools())
    tool = next(t for t in tools if t.name == TOOL_NAME)
    properties = tool.input_schema.get("properties", {})
    for field in ("docs_dir", "db_path", "bib_file", "workspace_id"):
        assert field in properties, f"indexing-shaped field {field} not exposed"


def test_e3_neg_040_indexing_tool_docstring_declares_the_boundary():
    """The tool's own docs state the boundary, the alternatives, and why."""
    doc = nexus_rag_index.__doc__ or ""
    flat = " ".join(doc.lower().split())
    assert "declared unsupported" in flat
    assert "not available through mcp" in flat
    assert "unsupported_capability" in flat
    assert "rag_index" in doc
    assert "rag_indexing" in doc
    assert "scholar-rag index" in doc
    assert "scholar_rag.index_service" in doc
    assert "index_workspace" in doc
    assert "before any" in flat
    assert "no store path, workspace identity, accepted parent, or journal" in flat
    # The retired behaviour is named as removed, not as still available.
    assert "non-authoritative" in flat
    assert "no partial mcp indexing" in flat


def test_e3_neg_040_tool_count_is_unchanged_at_twenty_five():
    """E3 converts a tool; it adds none and drops none."""
    registered = _registered_tool_names()
    assert PRE_EXISTING_TOOLS <= registered
    assert registered == PRE_EXISTING_TOOLS
    assert len(registered) == 25


def _help_text() -> str:
    """Capture ``scholar-agent --help`` output."""
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with contextlib.suppress(SystemExit), redirect_stdout(buffer):
        main(["--help"])
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# (c) The rejection envelope
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", sorted(ARGUMENT_CASES))
def test_e3_neg_041_indexing_shaped_call_returns_failed_envelope(case: str):
    """Any indexing-shaped request -- valid, empty, or garbage -- is rejected."""
    envelope = _envelope(nexus_rag_index(**ARGUMENT_CASES[case]))
    assert envelope["operation"] == REJECTION_OPERATION
    assert envelope["status"] == "FAILED"
    assert envelope["artifacts"] == []
    assert envelope["warnings"] == []
    assert len(envelope["errors"]) == 1
    error = envelope["errors"][0]
    assert error["code"] == REJECTION_CODE
    assert error["retryable"] is False
    assert error["message"].strip()


def test_e3_neg_040_envelope_shape_is_exact_and_carries_no_success_bookkeeping():
    """Exact vocabulary; no index/manifest/audit/success/lineage fabrication."""
    envelope = _envelope(nexus_rag_index(**INDEXING_REQUEST_ARGS))
    assert set(envelope) == ENVELOPE_KEYS
    error = envelope["errors"][0]
    assert set(error) == ERROR_KEYS
    assert set(error["details"]) == DETAIL_KEYS
    serialized = json.dumps(envelope)
    assert "SUCCESS" not in serialized
    assert "indexed_files" not in serialized
    assert "total_chunks" not in serialized
    # Pre-lineage rejection: no run identity or contract claim is invented.
    assert "run_id" not in serialized
    assert "contract_version" not in serialized


def test_e3_neg_040_envelope_is_deterministic_and_argument_independent():
    """Required behavior 2: one envelope for every payload (E3-NEG-040)."""
    results = [nexus_rag_index(**ARGUMENT_CASES[case]) for case in ARGUMENT_CASES]
    assert len(set(results)) == 1
    assert results[0] == unsupported_capability_envelope_json(RAG_INDEXING)
    assert json.loads(results[0]) == unsupported_capability_envelope(RAG_INDEXING)


def test_e3_neg_041_envelope_details_name_the_supported_api_and_cli_surfaces():
    """API/CLI note: the T-90 surfaces are named explicitly and correctly."""
    envelope = _envelope(nexus_rag_index(**INDEXING_REQUEST_ARGS))
    details = envelope["errors"][0]["details"]
    assert details["capability"] == "rag_indexing"
    assert details["mcp_supported"] is False
    assert details["owning_surfaces"] == ["API", "CLI"]
    assert details["owner"] == caps.RAG_OWNER
    assert details["reference"] == (
        "docs/architecture/wp01_packet_e3_implementation_handoff.md#9.1"
    )
    alternatives = details["alternatives"]
    assert len(alternatives) == 2
    assert "scholar-rag index" in alternatives[0]
    assert "scholar_rag.index_service" in alternatives[1]
    assert "index_workspace" in alternatives[1]
    # Message *and* details both name both alternatives.
    message = envelope["errors"][0]["message"]
    for alternative in alternatives:
        assert alternative in message


def test_e3_neg_041_advertised_cli_matches_canonical_index_signature():
    """API/CLI note: the advertised CLI is executable, not merely named.

    Packet E3 section 9.1 requires the rejection to be *actionable*, which is a
    property of the text itself: an alternative that names a flag the owning
    command does not have is worse than no alternative, because the caller
    discovers the lie only after pasting it into a shell. The E3 alternative
    previously advertised an audit logger option that ``scholar-rag index`` has
    never accepted.

    The real flag set is *derived* from the canonical T-90 command rather than
    hand-frozen here, so the assertion keeps working as the command evolves and
    fails loudly the moment the two drift apart in either direction.
    """
    # The canonical command is imported function-locally, not at module scope:
    # this test states a version floor on scholar-rag-kit (the T-90
    # ``scholar_rag.cli index`` command). A rag-kit older than that floor must
    # fail as this one test rather than as a collection error that hides every
    # other test in this module -- and the rest of the E3 boundary holds
    # independently of the rag-kit's CLI.
    import scholar_rag.cli as rag_cli
    import typer

    cli_command = typer.main.get_command(rag_cli.app)
    # typer collapses an app that declares a single command into that command
    # itself, so the sub-command map only exists for multi-command apps. Both
    # shapes must resolve to the same answer, and neither may raise here.
    sub_commands = getattr(cli_command, "commands", None)
    if sub_commands is None:
        sub_commands = (
            {"index": cli_command}
            if getattr(cli_command, "name", None) == "index"
            else {}
        )
    assert "index" in sub_commands, (
        "scholar_rag.cli has no `index` command: this rag-kit predates the "
        "T-90 index service the E3 alternative points at"
    )
    params = sub_commands["index"].params
    real_flags = {
        option for param in params for option in param.opts if option.startswith("--")
    }
    real_positionals = {
        option
        for param in params
        for option in param.opts
        if not option.startswith("-")
    }

    advertised = caps.INDEX_CLI_ALTERNATIVE
    assert "scholar-rag index" in advertised
    advertised_flags = {
        token.strip('`<>"').rstrip(",;)")
        for token in advertised.split()
        if token.strip('`<>"').startswith("--")
    }
    assert advertised_flags, "the advertised CLI alternative names no options"
    unknown = advertised_flags - real_flags
    assert not unknown, (
        f"the advertised CLI names options the canonical `scholar-rag index` "
        f"command does not accept: {sorted(unknown)}"
    )

    # The same scrutiny applies to the positional argument: the command takes
    # exactly one, and it must be named correctly. The old advertisement passed
    # `<workspace>` where `docs_path` is required, which the flag checks above
    # cannot see because a placeholder carries no leading dashes.
    assert real_positionals, (
        "the canonical `index` command takes no positional argument, which is a defect"
    )
    invocation = advertised.split("uv run scholar-rag index", 1)
    assert len(invocation) == 2, (
        "the advertised CLI alternative does not contain a `uv run scholar-rag "
        f"index` invocation to check: {advertised!r}"
    )
    positional_token = invocation[1].strip(" `").split()[0]
    assert positional_token.startswith("<") and positional_token.endswith(">"), (
        f"the advertised CLI example passes no positional `docs_path`: "
        f"{positional_token!r}"
    )
    advertised_positional = positional_token.strip("`<>")
    assert advertised_positional in real_positionals, (
        f"the advertised CLI names a positional argument the canonical "
        f"`scholar-rag index` command does not accept: {advertised_positional!r}"
    )

    # The example is only runnable if every required parameter is supplied, so
    # a partial example is a defect too (and would silently pass the checks
    # above, which only police invented names).
    required = {
        option
        for param in params
        if getattr(param, "required", False)
        for option in param.opts
    }
    assert required, "the canonical `index` command requires nothing, which is a defect"
    missing = required - advertised_flags - {advertised_positional}
    assert not missing, (
        f"the advertised CLI omits required parameters of `scholar-rag index`: "
        f"{sorted(missing)}"
    )

    # And the advertisement is carried verbatim by the envelope, so the
    # signature check above covers what a caller actually reads.
    details = _envelope(nexus_rag_index(**INDEXING_REQUEST_ARGS))["errors"][0][
        "details"
    ]
    assert details["alternatives"][0] == caps.INDEX_CLI_ALTERNATIVE
    assert (
        caps.INDEX_CLI_ALTERNATIVE
        in _envelope(nexus_rag_index(**INDEXING_REQUEST_ARGS))["errors"][0]["message"]
    )


def test_e3_neg_040_envelope_is_never_free_text_only():
    """Negative case: no canned success/failure string is returned."""
    for args in ARGUMENT_CASES.values():
        raw = nexus_rag_index(**args)
        assert raw == unsupported_capability_envelope_json(RAG_INDEXING)
        for marker in _FREE_TEXT_MARKERS:
            assert marker not in raw, f"{marker!r} leaked into the refusal"


def test_e3_neg_041_envelope_ignores_authority_claiming_payloads():
    """E3-NEG-041: a payload claiming a store/parent/journal changes nothing."""
    claimed = nexus_rag_index(**AUTHORITY_CLAIMING_ARGS)
    assert claimed == nexus_rag_index(docs_dir="anything-else")
    serialized = claimed
    for leaked in (
        "SCI-000001",
        "/srv/workspaces/demo",
        "chroma_db",
        "references.bib",
    ):
        assert leaked not in serialized, f"refusal echoed or honoured {leaked!r}"
    # The retired tool's CWD-relative default must not surface anywhere.
    assert "./chroma_db" not in serialized
    envelope = _envelope(serialized)
    details = envelope["errors"][0]["details"]
    assert set(details) == DETAIL_KEYS


def test_e3_neg_041_envelope_carries_no_db_path_parent_or_journal():
    """No CWD-derived store path, workspace identity, accepted parent, journal."""
    envelope = _envelope(nexus_rag_index(**INDEXING_REQUEST_ARGS))
    for forbidden in (
        "db_path",
        "chroma_db",
        "docs_dir",
        "accepted",
        "parent",
        "journal",
        "audit",
        "workspace_id",
    ):
        # The message may *mention* these as things it does not do; the envelope
        # must not *carry* them as values. Checked structurally instead of by
        # substring for the value-bearing fields.
        assert forbidden not in str(envelope["artifacts"])

    # Structurally: no key anywhere in the envelope is a path or identity.
    def _keys(node: Any) -> set[str]:
        found: set[str] = set()
        if isinstance(node, dict):
            for key, value in node.items():
                found.add(key)
                found |= _keys(value)
        elif isinstance(node, list):
            for item in node:
                found |= _keys(item)
        return found

    assert _keys(envelope) == ENVELOPE_KEYS | ERROR_KEYS | DETAIL_KEYS
    # No ``PARTIAL`` outcome is *claimed*: the status is the single
    # ``FAILED`` value. The word may appear in the message only as the reason
    # the retired tool could not express one, which is a prose claim about a
    # removed behaviour, not an asserted status.
    assert envelope["status"] == "FAILED"
    statuses = {envelope["status"]}
    for artifact in envelope["artifacts"]:
        if isinstance(artifact, dict) and "status" in artifact:
            statuses.add(artifact["status"])
    assert statuses == {"FAILED"}


# --------------------------------------------------------------------------- #
# (d) Zero I/O, no store opening, no journal discovery
# --------------------------------------------------------------------------- #


def test_e3_neg_040_rejection_path_performs_no_network_or_filesystem_io():
    """Runtime tripwires: nothing is created, replaced, or removed."""
    with _forbid_io() as wire:
        nexus_rag_index(**INDEXING_REQUEST_ARGS)
        nexus_rag_index(**AUTHORITY_CLAIMING_ARGS)
        nexus_rag_index(docs_dir="x", db_path="./chroma_db")
    assert wire.owner_events() == []


def test_e3_neg_040_rejection_path_opens_no_vector_store():
    """The backend is never opened, even when a payload names one."""
    with _forbid_io() as wire:
        nexus_rag_index(**AUTHORITY_CLAIMING_ARGS)
    assert not [label for label in wire.owner_events() if label.startswith("chromadb")]
    assert wire.owner_events() == []


def test_e3_neg_041_refusal_is_independent_of_cwd(tmp_path, monkeypatch):
    """Same envelope from any working directory (no CWD-derived state)."""
    baseline = nexus_rag_index(docs_dir="extracted")
    for name in ("alpha", "beta", "chroma_db_parent", "extracted"):
        candidate = tmp_path / name
        candidate.mkdir()
        monkeypatch.chdir(candidate)
        assert nexus_rag_index(docs_dir="extracted") == baseline
        assert nexus_rag_index(**AUTHORITY_CLAIMING_ARGS) == baseline
    monkeypatch.chdir(tmp_path)
    assert nexus_rag_index(docs_dir="extracted") == baseline


def test_e3_neg_041_refusal_ignores_local_workspace_contents(tmp_path, monkeypatch):
    """A populated workspace -- incl. a decoy ``chroma_db`` -- changes nothing.

    The decoy directory exists precisely so that a CWD-relative ``db_path``
    would resolve to something real; the refusal must neither open it nor read
    it, and the tree must be byte-identical afterwards.
    """
    baseline = nexus_rag_index(docs_dir="extracted")
    for layout in ("empty", "decoy_store", "full_workspace"):
        workspace = tmp_path / layout
        if layout == "decoy_store":
            (workspace / "chroma_db").mkdir(parents=True)
            (workspace / "chroma_db" / "chroma.sqlite3").write_text("decoy")
        elif layout == "full_workspace":
            for sub in ("extracted", "chroma_db", "literature", "synthesis", ".cache"):
                (workspace / sub).mkdir(parents=True)
            (workspace / "project.json").write_text(
                json.dumps({"project_id": "decoy-project", "title": "Decoy"})
            )
            (workspace / "extracted" / "SCI-000001.md").write_text("# decoy\n")
            (workspace / "chroma_db" / "chroma.sqlite3").write_text("decoy")
            (workspace / "audit").mkdir()
            (workspace / "audit" / "journal.jsonl").write_text("{}\n")
        else:
            workspace.mkdir(parents=True)

        before = _snapshot(workspace)
        monkeypatch.chdir(workspace)
        with _forbid_io() as wire:
            assert nexus_rag_index(docs_dir="extracted") == baseline
            assert nexus_rag_index(**INDEXING_REQUEST_ARGS) == baseline
            assert nexus_rag_index(**AUTHORITY_CLAIMING_ARGS) == baseline
        assert wire.owner_events() == []
        assert _snapshot(workspace) == before, "refusal mutated the workspace"


def test_e3_neg_041_refusal_never_reaches_a_workspace_or_journal(tmp_path, monkeypatch):
    """Discovery tripwire: no existence check, stat, or read on the path layer.

    ``Path.exists``/``is_dir``/``iterdir``/``read_*`` are the operations the
    retired tool used to discover a workspace and a store. Tripwiring them
    proves the refusal does not even *look* for a workspace.
    """
    workspace = tmp_path / "ws"
    (workspace / "chroma_db").mkdir(parents=True)
    (workspace / "extracted").mkdir()
    (workspace / "project.json").write_text(json.dumps({"project_id": "decoy"}))
    monkeypatch.chdir(workspace)

    probed: list[str] = []

    def _probe(label: str) -> Callable[..., Any]:
        def _record(*args: Any, **kwargs: Any) -> Any:
            probed.append(label)
            raise AssertionError(f"refusal probed the filesystem via {label}")

        return _record

    targets = (
        "exists",
        "is_dir",
        "is_file",
        "iterdir",
        "glob",
        "rglob",
        "stat",
        "read_text",
        "read_bytes",
        "resolve",
        "absolute",
    )
    with contextlib.ExitStack() as stack:
        for attr in targets:
            stack.enter_context(patch.object(Path, attr, _probe(attr)))
        result = nexus_rag_index(**INDEXING_REQUEST_ARGS)
    assert probed == []
    assert result == unsupported_capability_envelope_json(RAG_INDEXING)
    # The decoy is intact and untouched.
    assert (workspace / "chroma_db").is_dir()
    assert (workspace / "project.json").read_text() == json.dumps(
        {"project_id": "decoy"}
    )


def test_e3_neg_040_rejection_path_cannot_reach_io_or_indexing_symbols():
    """Static proof: no I/O, store, or indexing symbol is reachable."""
    for module_name in sorted(_referenced_imports(nexus_rag_index)):
        assert module_name in _ALLOWED_MODULES, (
            f"rejection path imports {module_name!r} (only pure modules allowed)"
        )
    resolved = _reachable_globals(nexus_rag_index)
    assert resolved, "the rejection path should reference its envelope builder"
    for name, target in sorted(resolved.items()):
        if isinstance(target, ModuleType):
            assert target.__name__ in _ALLOWED_MODULES, (
                f"rejection path imports module {name}={target.__name__}"
            )
            continue
        owner = getattr(target, "__module__", None)
        if owner in _IO_MODULES:
            pytest.fail(f"rejection path reaches I/O module {name} from {owner}")
        if name in _IO_CALLABLES:
            pytest.fail(f"rejection path reaches I/O callable {name}")
        if owner == "builtins":
            assert name in _ALLOWED_BUILTINS, (
                f"rejection path reaches disallowed builtin {name!r}"
            )
            continue
        assert (
            owner in _ALLOWED_MODULES
            or owner is None
            or owner.startswith("scholar_agent")
        ), f"rejection path reaches unexpected symbol {name!r} from {owner}"
    assert "unsupported_capability_envelope_json" in resolved
    assert resolved["RAG_INDEXING"] == RAG_INDEXING
    # No indexing implementation of any kind is reachable: not the legacy
    # indexer, not the T-90 service, not the retriever.
    for forbidden in _INDEXING_SYMBOLS:
        assert forbidden not in resolved, f"rejection path must not reach {forbidden!r}"


def test_e3_neg_040_capabilities_module_binds_no_io_capable_module():
    """The declaration module itself has no I/O-capable module bound."""
    for name in _IO_MODULES:
        assert name not in vars(caps), f"{name} must not be bound in capabilities"
    assert "json" in vars(caps), "the declaration module needs json for serialisation"
    assert vars(caps)["json"].__name__ == "json"  # pure serialisation only


def test_e3_neg_040_indexing_capability_adds_no_new_envelope_builder():
    """One generic pure builder serves all three boundaries."""
    builders = sorted(
        name
        for name in vars(caps)
        if name.startswith("unsupported_capability_envelope")
    )
    assert builders == [
        "unsupported_capability_envelope",
        "unsupported_capability_envelope_json",
    ]


def test_e3_neg_040_capabilities_module_exposes_no_indexing_domain_logic():
    """The declaration module names the API; it does not implement or import it."""
    for name in vars(caps):
        lowered = name.lower()
        assert "chunk" not in lowered
        assert "embed" not in lowered
        assert "chroma" not in lowered
        assert "vectorstore" not in lowered
    # The alternative strings mention the service; no attribute *is* it.
    assert not hasattr(caps, "index_workspace")
    assert not hasattr(caps, "IndexServiceRequest")
    assert not hasattr(caps, "ScholarIndexer")


# --------------------------------------------------------------------------- #
# (e) The legacy indexing path is removed, not forwarded to
# --------------------------------------------------------------------------- #


def test_e3_neg_040_indexing_tool_body_is_the_single_pure_statement():
    """No partial indexing and no forwarding survive in the tool body."""
    body = _body_of(nexus_rag_index)
    for forbidden in (
        "ScholarIndexer",
        "index_directory",
        "index_workspace",
        "_resolve_path",
        "Path(",
        "exists",
        "mkdir",
        "open(",
        "try:",
        "except",
        'return f"Error',
    ):
        assert forbidden not in body, f"tool body must not contain {forbidden!r}"
    assert [line.strip() for line in body.strip().splitlines() if line.strip()] == [
        "return unsupported_capability_envelope_json(RAG_INDEXING)"
    ]


def test_e3_neg_040_adapter_no_longer_imports_the_legacy_indexer():
    """The ``ScholarIndexer`` import is gone; no module in the kit binds it."""
    assert not hasattr(server_module, "ScholarIndexer")
    for path in _source_files():
        source = path.read_text(encoding="utf-8")
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        for statement in (
            "from scholar_rag.indexer import",
            "import scholar_rag.indexer",
            "from scholar_rag import indexer",
        ):
            assert statement not in code, (
                f"{path.name} still imports the legacy indexer"
            )


def test_e3_neg_040_refusal_survives_an_exploding_legacy_indexer(monkeypatch):
    """Mutation test: if the legacy indexer were reached, this test fails.

    Patching the *real* ``scholar_rag.indexer.ScholarIndexer`` with a class that
    raises on construction proves the refusal never forwards to it. A passing
    run is therefore evidence of absence, not merely of a well-typed return.
    """
    indexer_module = importlib.import_module("scholar_rag.indexer")
    monkeypatch.setattr(indexer_module, "ScholarIndexer", _ExplodingIndexer)
    assert nexus_rag_index(**INDEXING_REQUEST_ARGS) == (
        unsupported_capability_envelope_json(RAG_INDEXING)
    )
    assert nexus_rag_index(**AUTHORITY_CLAIMING_ARGS) == (
        unsupported_capability_envelope_json(RAG_INDEXING)
    )


def test_e3_neg_040_refusal_survives_a_failing_legacy_index_directory(monkeypatch):
    """Mutation test: a legacy ``index_directory`` on the module is never used."""
    indexer_module = importlib.import_module("scholar_rag.indexer")

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("legacy indexing was invoked by the E3 refusal")

    monkeypatch.setattr(indexer_module, "ScholarIndexer", _ExplodingIndexer)
    monkeypatch.setattr(indexer_module, "ASTChunker", _ExplodingIndexer, raising=False)
    monkeypatch.setattr(indexer_module, "index_directory", _boom, raising=False)
    assert nexus_rag_index(docs_dir="extracted", db_path="./chroma_db") == (
        unsupported_capability_envelope_json(RAG_INDEXING)
    )


def test_e3_neg_040_refusal_reaches_clients_through_mcp_dispatch():
    """The boundary holds on the real MCP dispatch path, not just in-process."""
    result = asyncio.run(
        mcp.call_tool(
            TOOL_NAME,
            {
                "docs_dir": "extracted",
                "db_path": "./chroma_db",
                "workspace_id": "SCI-1",
            },
        )
    )
    text = result.content[0].text
    assert json.loads(text) == unsupported_capability_envelope(RAG_INDEXING)
    # Transport-switch parity: the same request through the dispatcher yields the
    # same envelope as the direct call, so a client cannot discover a different
    # answer by changing transport (E3-NEG-041).
    assert text == unsupported_capability_envelope_json(RAG_INDEXING)


def test_e3_neg_040_agent_kit_contains_no_second_indexing_implementation():
    """E3 adds a declaration, not an indexer.

    The scan is over *executable* content (docstrings excluded) across every
    module of the package, so a future chunker/store/embedder added here fails
    even if it is never wired to a tool.
    """
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        executable = "\n".join(
            line for line in _strip_docstrings(tree).splitlines() if line.strip()
        )
        for forbidden in (
            "ScholarIndexer(",
            ".index_directory(",
            ".index_workspace(",
            "PersistentClient(",
            "SentenceTransformer(",
            "ASTChunker(",
        ):
            assert forbidden not in executable, (
                f"{path.name} contains an indexing call {forbidden!r}; agent-kit "
                "must declare the boundary, never implement the capability"
            )


def _strip_docstrings(tree: ast.Module) -> str:
    """Re-render a module's AST with every docstring blanked out."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body[0].value.value = ""
    return ast.unparse(tree)


# --------------------------------------------------------------------------- #
# (f) E1 and E2 are unchanged by this addition
# --------------------------------------------------------------------------- #


def test_e3_neg_040_indexing_declaration_is_a_third_separate_declaration():
    """E3 adds a capability; it does not re-interpret either PDF one."""
    acquisition = get_capability(PDF_ACQUISITION).as_dict()
    extraction = get_capability(PDF_EXTRACTION).as_dict()
    expected_acquisition = dict(E1_EXPECTED_ACQUISITION)
    expected_extraction = dict(E2_EXPECTED_EXTRACTION)
    expected_acquisition["summary"] = acquisition["summary"]
    expected_extraction["summary"] = extraction["summary"]
    assert acquisition == {
        **expected_acquisition,
        "owning_surfaces": list(expected_acquisition["owning_surfaces"]),
        "alternatives": list(expected_acquisition["alternatives"]),
    }
    assert extraction == {
        **expected_extraction,
        "owning_surfaces": list(expected_extraction["owning_surfaces"]),
        "alternatives": list(expected_extraction["alternatives"]),
    }
    # Summary text is pinned by identity of the E1/E2 references instead.
    assert acquisition["reference"] == E1_EXPECTED_ACQUISITION["reference"]
    assert extraction["reference"] == E2_EXPECTED_EXTRACTION["reference"]


def test_e3_neg_040_pdf_envelopes_are_unchanged_and_mention_no_indexing():
    """Adding E3 did not rewrite E1/E2 messages (E2-NEG-020, re-applied)."""
    for name in (PDF_ACQUISITION, PDF_EXTRACTION):
        serialized = unsupported_capability_envelope_json(name)
        for marker in _E3_ONLY_MARKERS:
            assert marker not in serialized, f"E3 marker {marker!r} leaked into {name}"
    # Their tools still answer with their own envelopes, unchanged.
    from scholar_agent.server import nexus_pdf_acquire, nexus_pdf_extraction

    assert nexus_pdf_acquire() == unsupported_capability_envelope_json(PDF_ACQUISITION)
    assert nexus_pdf_extraction() == unsupported_capability_envelope_json(
        PDF_EXTRACTION
    )


def test_e3_neg_040_rejection_code_is_one_shared_constant():
    """A single code constant backs all three declarations and both envelopes."""
    assert caps.UNSUPPORTED_CAPABILITY == "UNSUPPORTED_CAPABILITY"
    code_values = {
        get_capability(name).rejection_code
        for name in (PDF_ACQUISITION, PDF_EXTRACTION, RAG_INDEXING)
    }
    assert code_values == {caps.UNSUPPORTED_CAPABILITY}


def test_e3_neg_040_retrieval_surface_is_not_declared_unsupported():
    """E3 declares only indexing; retrieval tools keep their own behaviour."""
    from scholar_agent.server import nexus_rag_query, nexus_rag_synthesize

    for tool in (nexus_rag_query, nexus_rag_synthesize):
        body = _body_of(tool)
        assert "unsupported_capability_envelope_json" not in body
        assert "UNSUPPORTED_CAPABILITY" not in body
        assert tool.__name__ not in caps.CAPABILITIES
    # No retrieval capability is declared by this packet (E3-003): E3 declares
    # indexing only, and adds no retrieval citation-token surface.
    assert sorted(caps.CAPABILITIES) == [
        "pdf_acquisition",
        "pdf_extraction",
        "rag_indexing",
    ]
