"""WP01-E2 declared MCP capability boundary for PDF extraction (E2-NEG-019/020).

Packet E2 section 9 / acceptance criterion E2-013 require PDF extraction to be
*declared* unsupported on the MCP surface rather than silently omitted, and
separately require the pre-existing raw-path tool to be documented as
non-authoritative:

* the capability registry declares ``pdf_extraction`` with
  ``mcp_supported=False``, owning surfaces ``("API", "CLI")``, owner
  ``nexus-scholar-org/scholar-pdf-kit``, and the stable non-retryable
  rejection code ``UNSUPPORTED_CAPABILITY``;
* the boundary is observable on the real MCP surface (the tool is registered,
  discoverable through the public ``list_tools`` API, reachable through
  ``MCPServer.call_tool``, and listed in ``scholar-agent --help``);
* an extraction-shaped request returns the standard operation envelope
  (``operation="extract_pdf"``, ``status="FAILED"``, no artifacts, no warnings,
  exactly one non-retryable ``UNSUPPORTED_CAPABILITY`` error naming the API/CLI
  alternatives), unconditionally and argument-independently;
* the rejection happens before any engine import, provider transport,
  temporary/final file creation, sidecar construction, or audit append;
* ``E2-NEG-019`` -- the raw-path ``nexus_extract_pdf`` is declared
  non-authoritative: it verifies nothing, identifies nothing, and cannot
  produce a Contract artifact, a sidecar, or an identity-addressed output;
* ``E2-NEG-020`` -- no silent broaden: ``pdf_acquisition`` keeps its E1
  declaration and its ``acquire_pdf`` envelope, and E2 does not emit an
  authoritative artifact from MCP;
* the canonical ``scholar-pdf-kit`` API/CLI remain the supported E2 surfaces
  (this kit only *names* them; it does not reimplement extraction).

Every test here is offline, hermetic, and deterministic: no network, no PDF
fixtures, no engine, no writing outside ``tmp_path``. Zero-I/O evidence is
layered on purpose -- runtime thread-scoped tripwires over network entry points
and filesystem mutators, a sandboxed CWD/workspace snapshot, a declaration
module that binds no I/O-capable module at all, and a static reachability walk
over the rejection path's own bytecode (including nested code objects) showing
that no I/O-capable symbol is even reachable from it. That static walk (layer d)
is the substantive zero-I/O proof.

One layer is deliberately *weaker* and is labelled as such: the injected audit
sink (layer c) is a naming-convention guard, not independent evidence. Neither
``server_module`` nor ``caps`` has ever bound ``log_event``/``audit_sink``/
``AUDIT_SINK``/``append_audit``, so patching those names onto them (with
``raising=False``) only attaches attributes nothing reads, and the resulting
``sink.events == []`` assertion cannot fail. It guards against a future rename
adopting one of those conventional names, nothing more. The filesystem
assertions in that same test *are* real -- they are backed by the tripwires and
the before/after snapshot comparison.

Finally, this module deliberately *does not* claim to fix E2's remaining debt.
The raw-path heuristic in ``_pdf_metadata`` and the blind ``except Exception``
in the legacy tool body are the declared E1-status-quo (finding 9); the tests
below pin them so they cannot drift silently, and assert the rejection path
cannot reach them. Pinning is not fixing: E2-NEG-043 bars those values from
authoritative output, which is a PDF-kit and harness-adapter obligation, not
something this adapter can enforce by deleting a legacy tool.
"""

from __future__ import annotations

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
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest

from scholar_agent import capabilities as caps
from scholar_agent import server as server_module
from scholar_agent.capabilities import (
    PDF_ACQUISITION,
    PDF_EXTRACTION,
    UNSUPPORTED_CAPABILITY,
    CapabilityDeclaration,
    get_capability,
    unsupported_capability_envelope,
    unsupported_capability_envelope_json,
)
from scholar_agent.server import main, mcp, nexus_extract_pdf, nexus_pdf_extraction

# --------------------------------------------------------------------------- #
# Module-wide CWD sandbox
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _sandbox_cwd(tmp_path, monkeypatch):
    """Run *every* test in this module with a sandboxed working directory.

    Without this, a relative-path write that escaped the tripwires below would
    land in the process CWD -- i.e. the repo root -- instead of a tmp dir. The
    sandbox makes the module's ``no writing outside tmp_path`` claim hold even
    for the direct-call tests that install no tripwires of their own.

    CWD-sensitive bootstrap work (e.g. the asyncio loop) is absorbed by the
    sandbox rather than being a reason to skip it. Tests that need a more
    specific CWD (``mcp-cwd-sandbox`` / ``workspace``) still call
    ``monkeypatch.chdir`` themselves and override this default; the autouse
    fixture only guarantees a non-repo-root starting point, and ``monkeypatch``
    unwinds both chdirs at teardown.
    """
    monkeypatch.chdir(tmp_path)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

TOOL_NAME = "nexus_pdf_extraction"
LEGACY_TOOL_NAME = "nexus_extract_pdf"
REJECTION_CODE = "UNSUPPORTED_CAPABILITY"
REJECTION_OPERATION = "extract_pdf"

#: The 23 MCP tools that existed before either boundary was added -- i.e. the
#: same set E1 pinned. Asserted individually so an unrelated tool cannot be
#: renamed or dropped here.
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
    }
)

#: Exact envelope shape. "No artifacts" is explicit (never merely absent), and
#: the rejection carries no manifest/sidecar/audit/success bookkeeping.
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

#: The legacy raw-path tool's exact executable statements, pinned verbatim.
#: E2 changed its docstring only (Packet E2 section 8.2); this list is the
#: evidence that no statement of its behaviour moved. The pre-existing blind
#: ``except Exception`` is included deliberately: it is declared status quo
#: (finding 9), and silently "fixing" it here would be an unrequested behaviour
#: change disguised as tidiness.
LEGACY_BODY_STATEMENTS = [
    "pdf_path = _resolve_path(pdf_path) or pdf_path",
    "output_dir = _resolve_path(output_dir) or output_dir",
    "pdf = Path(pdf_path)",
    "out_dir = Path(output_dir)",
    "if not pdf.exists():",
    'return f"Error: PDF {pdf_path} not found."',
    "try:",
    "out_dir.mkdir(parents=True, exist_ok=True)",
    "metadata = _pdf_metadata(pdf)",
    'if engine.lower() == "grobid":',
    "res_file = GrobidEngine.extract_markdown(pdf, out_dir)",
    "else:",
    'engine_cls = DoclingEngine if engine.lower() == "docling" else PyMuPDFEngine',
    "res_file = engine_cls.extract_markdown(pdf, out_dir, metadata=metadata)",
    'return f"Extracted {pdf.name} to {res_file}"',
    "except Exception as e:",
    'return f"Error during PDF extraction: {e}"',
]

#: E1's declaration values, hardcoded so E2-NEG-020 ("no silent broaden") fails
#: if a future edit re-interprets acquisition instead of adding extraction.
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

#: Markers that belong to the E2 limb alone. Their total absence from the
#: acquisition envelope is the direct, textual proof that adding extraction did
#: not rewrite acquisition's message.
_E2_ONLY_MARKERS = (
    "extract_pdf",
    "E2 PDF extraction",
    "scholar-pdf extract-run",
    "scholar_pdf.extraction.PDFExtractionService",
    "wp01_packet_e2_extracted_text_handoff",
)


# --------------------------------------------------------------------------- #
# Representative extraction-shaped arguments. None of them is read: the
# rejection is unconditional. ``pdf_path``/``output_dir``/``engine`` mirror the
# raw-path tool's public shape, which is exactly the request a client would
# send after E2-NEG-019 declared that tool non-authoritative.
# --------------------------------------------------------------------------- #

EXTRACTION_REQUEST_ARGS: dict[str, Any] = {
    "pdf_path": "extracted/SCI-000001/10.1038_s41586-024-07000-0.pdf",
    "output_dir": "extracted/SCI-000001",
    "engine": "docling",
}

HOSTILE_ARGS: dict[str, Any] = {
    "pdf_path": "C:\\Windows\\Temp\\..\\..\\etc\\passwd",
    "output_dir": "../../../outside-the-workspace",
    "engine": "; DROP TABLE audit; --",
}

ARGUMENT_CASES: dict[str, dict[str, Any]] = {
    "no_arguments": {},
    "extraction_request_shape": EXTRACTION_REQUEST_ARGS,
    "invalid_and_hostile_arguments": HOSTILE_ARGS,
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
    return get_capability(PDF_EXTRACTION)


def _registered_tool_names() -> set[str]:
    return {tool.name for tool in mcp._tool_manager.list_tools()}


def _body_of(func: Callable[..., Any]) -> str:
    """Executable body of a function, with its docstring removed."""
    return textwrap.dedent(inspect.getsource(func).split('"""', 2)[-1]).strip()


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
    """Tripwire every network entry point and filesystem mutator."""
    wire = _Tripwire()
    with contextlib.ExitStack() as stack:
        for module_name, attr in _NETWORK_TARGETS + _FS_TARGETS:
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
                root = instruction.argval.split(".")[0]
                modules.add(root)
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

    Walks kit-local functions transitively (the rejection path delegates to
    the pure envelope builder) and returns ``name -> resolved object``.
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


class _RecordingAuditSink:
    """Fake audit sink; the rejection path must never append to it.

    Naming-convention guard only: the attribute names patched onto
    ``server_module``/``caps`` by the layer-(c) test do not exist on those
    modules, so this sink cannot observe the rejection path. See that test's
    docstring for the full scope statement.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def append(self, event: dict[str, Any]) -> None:  # pragma: no cover
        self.events.append(event)

    def log_event(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        self.events.append({"args": args, "kwargs": kwargs})

    def log(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        self.events.append({"args": args, "kwargs": kwargs})


# --------------------------------------------------------------------------- #
# (a) Capability registry declaration
# --------------------------------------------------------------------------- #


def test_e2_neg_019_registry_declares_pdf_extraction_mcp_unsupported():
    """E2-NEG-019: the registry declares pdf_extraction unsupported on MCP."""
    declaration = _declaration()
    assert declaration.name == "pdf_extraction"
    assert declaration.mcp_supported is False
    assert "pdf_extraction" in caps.CAPABILITIES
    assert sorted(caps.CAPABILITIES) == ["pdf_acquisition", "pdf_extraction"]


def test_e2_neg_019_registry_owning_surface_is_api_and_cli_owned_by_pdf_kit():
    """The canonical PDF kit owns extraction through API/CLI, not this MCP kit."""
    declaration = _declaration()
    assert set(declaration.owning_surfaces) == {"API", "CLI"}
    assert caps.MCP_SURFACE == "MCP"
    assert caps.MCP_SURFACE not in declaration.owning_surfaces
    assert declaration.owner == caps.EXTRACTION_OWNER
    assert "scholar-pdf-kit" in declaration.owner


def test_e2_neg_019_registry_rejection_code_is_stable_unsupported_capability():
    """The declared rejection code is the stable, shared UNSUPPORTED_CAPABILITY."""
    declaration = _declaration()
    assert declaration.rejection_code == REJECTION_CODE == UNSUPPORTED_CAPABILITY
    assert caps.UNSUPPORTED_CAPABILITY == "UNSUPPORTED_CAPABILITY"
    assert declaration.rejection_operation == REJECTION_OPERATION == "extract_pdf"
    assert caps.EXTRACT_PDF_OPERATION == "extract_pdf"
    assert caps.SUPPORTED_OWNING_SURFACES == ("API", "CLI")
    # The code is shared, not forked: one vocabulary entry for "this surface
    # does not serve this capability".
    assert caps.PDF_ACQUISITION_DECLARATION.rejection_code == (
        caps.PDF_EXTRACTION_DECLARATION.rejection_code
    )
    # ...but the two capabilities keep distinct, non-interchangeable operations.
    assert (
        caps.PDF_EXTRACTION_DECLARATION.rejection_operation
        != caps.PDF_ACQUISITION_DECLARATION.rejection_operation
    )


def test_e2_neg_019_registry_projection_is_json_serialisable():
    """The declaration projects to a plain JSON-ready dict for parity fixtures."""
    projection = _declaration().as_dict()
    assert json.loads(json.dumps(projection))["mcp_supported"] is False
    assert projection["owning_surfaces"] == ["API", "CLI"]
    assert projection["rejection_code"] == REJECTION_CODE
    assert projection["alternatives"]
    assert projection["reference"] == caps.E2_REFERENCE


def test_e2_neg_019_registry_is_immutable_and_rejects_unknown_capability():
    """Declarations are immutable facts; an undeclared capability is a bug."""
    with pytest.raises(TypeError):
        caps.CAPABILITIES["pdf_extraction_v2"] = _declaration()  # type: ignore[index]
    with pytest.raises(TypeError):
        del caps.CAPABILITIES[PDF_EXTRACTION]  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        _declaration().mcp_supported = True  # type: ignore[misc]
    with pytest.raises(caps.UnknownCapabilityError):
        get_capability("pdf_extraction_v2")


def test_e2_neg_019_no_rejection_envelope_for_a_capability_mcp_actually_serves(
    monkeypatch,
):
    """A supported extraction capability must not be answered with a rejection.

    The builder is fail-closed: it refuses to build a rejection for a
    declaration that claims MCP support. Modelling that requires replacing the
    registry, because the real ``pdf_extraction`` declaration is unsupported.
    """
    served = CapabilityDeclaration(
        name="pdf_extraction",
        summary="Extract fulltext from committed bytes.",
        owner="nexus-scholar-org/scholar-pdf-kit",
        owning_surfaces=("API", "CLI", "MCP"),
        mcp_supported=True,
        rejection_code=None,
        rejection_operation="extract_pdf",
        rejection_message="",
        alternatives=(),
        reference=caps.E2_REFERENCE,
    )
    monkeypatch.setattr(caps, "CAPABILITIES", {"pdf_extraction": served})
    with pytest.raises(ValueError, match="not declared unsupported"):
        unsupported_capability_envelope("pdf_extraction")


# --------------------------------------------------------------------------- #
# (b) The boundary is registered and discoverable (not silently omitted)
# --------------------------------------------------------------------------- #


def test_e2_neg_019_extraction_rejection_tool_is_registered_on_mcp_server():
    """The extraction-shaped tool exists on the module-level MCPServer."""
    assert server_module.mcp is mcp
    assert TOOL_NAME in _registered_tool_names()
    assert callable(nexus_pdf_extraction)
    assert nexus_pdf_extraction.__module__ == "scholar_agent.server"


def test_e2_neg_019_extraction_rejection_tool_is_discoverable_via_list_tools():
    """Public discovery exposes the tool with a boundary-describing schema."""
    tools = asyncio.run(mcp.list_tools())
    tool = next(t for t in tools if t.name == TOOL_NAME)
    assert tool.description
    assert "UNSUPPORTED" in tool.description
    properties = tool.input_schema.get("properties", {})
    for field in ("pdf_path", "output_dir", "engine"):
        assert field in properties, f"extraction-shaped field {field} not exposed"


def test_e2_neg_019_extraction_rejection_tool_docstring_declares_the_boundary():
    """The tool's own documentation states the boundary and the alternatives."""
    doc = nexus_pdf_extraction.__doc__ or ""
    # Identifier checks use the raw text; phrase checks use ``flat`` so a
    # reflowed line break cannot silently weaken (or fake) a statement.
    flat = " ".join(doc.lower().split())
    assert "declared unsupported" in flat
    assert "not available through mcp" in flat
    assert "unsupported_capability" in flat
    assert "extract_pdf" in doc
    assert "scholar-pdf" in doc
    assert "scholar_pdf" in doc
    assert "before any" in flat
    # E2-NEG-019: the tool must not be read as an extraction implementation.
    assert "non-authoritative" in flat
    assert "before any engine import" in flat


def test_e2_neg_019_extraction_rejection_tool_is_listed_in_server_help(capsys):
    """`scholar-agent --help` enumerates the tool (MCP surface parity gate)."""
    with contextlib.suppress(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "Exposed MCP Tools" in out
    assert TOOL_NAME in out
    assert REJECTION_CODE in out
    assert "extract_pdf" in out
    # E2-NEG-019: the same help must not advertise the legacy tool as if it
    # produced authoritative extracted text.
    assert "non-authoritative" in out.lower()


def test_e2_neg_019_pre_existing_tools_are_untouched():
    """The 23 pre-existing tools plus two boundaries are exactly 25 tools."""
    registered = _registered_tool_names()
    assert PRE_EXISTING_TOOLS <= registered
    assert sorted(registered - PRE_EXISTING_TOOLS) == [
        "nexus_pdf_acquire",
        "nexus_pdf_extraction",
    ]
    assert len(registered) == len(PRE_EXISTING_TOOLS) + 2 == 25


# --------------------------------------------------------------------------- #
# (c) The rejection envelope
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", sorted(ARGUMENT_CASES))
def test_e2_neg_019_extraction_shaped_call_returns_failed_envelope(case: str):
    """Any extraction-shaped request -- valid, empty, or invalid -- is rejected."""
    envelope = _envelope(nexus_pdf_extraction(**ARGUMENT_CASES[case]))
    assert envelope["operation"] == REJECTION_OPERATION
    assert envelope["status"] == "FAILED"
    assert envelope["artifacts"] == []
    assert envelope["warnings"] == []
    assert len(envelope["errors"]) == 1
    error = envelope["errors"][0]
    assert error["code"] == REJECTION_CODE
    assert error["retryable"] is False
    assert error["message"].strip()


def test_e2_neg_019_envelope_shape_is_exact_and_carries_no_success_bookkeeping():
    """Exact envelope vocabulary; no sidecar/audit/success/lineage fabrication."""
    envelope = _envelope(nexus_pdf_extraction(**EXTRACTION_REQUEST_ARGS))
    assert set(envelope) == ENVELOPE_KEYS
    assert envelope["warnings"] == []
    assert envelope["artifacts"] == []
    error = envelope["errors"][0]
    assert set(error) == ERROR_KEYS
    assert set(error["details"]) == DETAIL_KEYS
    serialized = json.dumps(envelope)
    assert "SUCCESS" not in serialized
    # Pre-lineage rejection: no run identity or contract claim is invented.
    assert "run_id" not in serialized
    assert "contract_version" not in serialized


def test_e2_neg_019_envelope_is_deterministic_and_argument_independent():
    """The rejection is unconditional: same envelope for every argument set."""
    results = [nexus_pdf_extraction(**ARGUMENT_CASES[case]) for case in ARGUMENT_CASES]
    assert len(set(results)) == 1
    assert results[0] == unsupported_capability_envelope_json(PDF_EXTRACTION)
    assert json.loads(results[0]) == unsupported_capability_envelope(PDF_EXTRACTION)


def test_e2_neg_019_envelope_details_name_the_supported_api_and_cli_surfaces():
    """API/CLI note: both supported alternatives are named explicitly."""
    envelope = _envelope(nexus_pdf_extraction(**EXTRACTION_REQUEST_ARGS))
    details = envelope["errors"][0]["details"]
    assert details["capability"] == "pdf_extraction"
    assert details["mcp_supported"] is False
    assert details["owning_surfaces"] == ["API", "CLI"]
    assert details["owner"] == caps.EXTRACTION_OWNER
    assert details["reference"] == caps.E2_REFERENCE
    assert len(details["alternatives"]) == 2
    joined = " ".join(details["alternatives"])
    assert "scholar-pdf extract-run" in joined
    assert "scholar_pdf.extraction.PDFExtractionService" in joined
    message = envelope["errors"][0]["message"]
    assert "not available through MCP" in message
    assert "scholar-pdf extract-run" in message
    assert "scholar_pdf.extraction.PDFExtractionService" in message
    assert "`scholar-pdf extract` CLI" not in message
    assert "Python `scholar_pdf.extract` API" not in message
    assert "API" in message and "CLI" in message


def test_e2_neg_019_rejection_message_states_the_pre_io_guarantee():
    """The message says the rejection precedes every I/O effect, engine first."""
    message = _envelope(nexus_pdf_extraction(**EXTRACTION_REQUEST_ARGS))["errors"][0][
        "message"
    ]
    lowered = message.lower()
    assert "rejected before any engine import" in lowered
    assert "transport" in lowered
    assert "sidecar" in lowered
    assert "audit" in lowered
    assert "not a parity claim" in lowered
    # E2-NEG-043 honesty is stated at the boundary, where a caller reads it.
    assert "filename" in lowered
    assert "regex" in lowered


def test_e2_neg_019_envelope_error_object_is_contract_error_shape_compatible():
    """The frozen Contract v1 error shape accepts this code: no contract change."""
    from scholar_harness.contracts.models import ContractError, OperationStatus

    envelope = _envelope(nexus_pdf_extraction(**EXTRACTION_REQUEST_ARGS))
    validated = ContractError.model_validate(envelope["errors"][0])
    assert validated.code == REJECTION_CODE
    assert validated.retryable is False
    assert OperationStatus(envelope["status"]) is OperationStatus.FAILED
    # Hard-failure coherence: a FAILED outcome must carry at least one error.
    assert envelope["errors"]


def test_e2_neg_019_rejection_envelope_reaches_clients_through_mcp_dispatch():
    """End-to-end through MCPServer.call_tool, not only via direct import."""
    result = asyncio.run(mcp.call_tool(TOOL_NAME, dict(EXTRACTION_REQUEST_ARGS)))
    payload = json.loads(result.content[0].text)
    assert payload["operation"] == REJECTION_OPERATION
    assert payload["status"] == "FAILED"
    assert payload["artifacts"] == []
    assert payload["warnings"] == []
    assert payload["errors"][0]["code"] == REJECTION_CODE
    assert payload["errors"][0]["retryable"] is False


# --------------------------------------------------------------------------- #
# (d) Zero I/O on the rejection path
# --------------------------------------------------------------------------- #


def test_e2_neg_019_rejection_path_performs_no_provider_transport_or_filesystem_io(
    tmp_path, monkeypatch
):
    """No network, no staging, no writes: the tripwires stay silent."""
    sandbox = tmp_path / "mcp-cwd-sandbox"
    workspace = tmp_path / "workspace"
    sandbox.mkdir()
    workspace.mkdir()
    monkeypatch.chdir(sandbox)
    before = _snapshot(tmp_path)

    with _forbid_io() as wire:
        for args in ARGUMENT_CASES.values():
            _envelope(nexus_pdf_extraction(**args))

    assert wire.owner_events() == []
    assert _snapshot(tmp_path) == before
    assert list(sandbox.iterdir()) == []
    assert list(workspace.iterdir()) == []
    assert Path.cwd() == sandbox.resolve()


def test_e2_neg_019_rejection_path_creates_no_sidecar_and_no_audit_success_event(
    tmp_path, monkeypatch
):
    """No sidecar and no staged/final file; the audit-sink layer is a guard.

    Honest scope: ``sink.events == []`` is a *naming-convention guard*, not
    independent evidence. ``server_module`` and ``caps`` have never bound
    ``log_event``/``audit_sink``/``AUDIT_SINK``/``append_audit``, so
    ``monkeypatch.setattr(..., raising=False)`` only injects names that nothing
    reads and the assertion cannot fail. It exists to catch a future rename
    adopting one of those conventional names.

    The substantive zero-I/O proof is layer (d), the static reachability walk
    over the rejection path's own bytecode, in
    ``test_e2_neg_019_rejection_path_cannot_reach_io_or_network_symbols``.

    The filesystem half of this test *is* substantive: the tripwires plus the
    before/after snapshot comparison, the empty workspace, and the absence of
    any sidecar/journal/PDF are all backed by real observation and are
    unaffected by the caveat above.
    """
    sink = _RecordingAuditSink()
    for module in (server_module, caps):
        for attribute in ("log_event", "audit_sink", "AUDIT_SINK", "append_audit"):
            monkeypatch.setattr(module, attribute, sink, raising=False)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    before = _snapshot(tmp_path)

    with _forbid_io():
        envelope = _envelope(nexus_pdf_extraction(**EXTRACTION_REQUEST_ARGS))

    assert sink.events == []
    assert _snapshot(tmp_path) == before
    assert list(workspace.iterdir()) == []
    assert not list(workspace.rglob("*.json"))
    assert not list(workspace.rglob("*sidecar*"))
    assert not list(workspace.rglob("*manifest*"))
    assert not list(workspace.rglob("*journal*"))
    assert not list(workspace.rglob("*.pdf"))
    # No success-shaped bookkeeping in the payload either.
    assert envelope["status"] == "FAILED"
    assert envelope["artifacts"] == []


def test_e2_neg_019_capabilities_module_binds_no_io_capable_module():
    """The declaration module itself has no I/O-capable module bound."""
    for name in _IO_MODULES:
        assert name not in vars(caps), f"{name} must not be bound in capabilities"
    assert "json" in vars(caps), "the declaration module needs json for serialisation"
    assert vars(caps)["json"].__name__ == "json"  # pure serialisation only


def test_e2_neg_019_rejection_path_cannot_reach_io_or_network_symbols():
    """Static proof: no I/O or network symbol is reachable from the tool body."""
    for module_name in sorted(_referenced_imports(nexus_pdf_extraction)):
        assert module_name in _ALLOWED_MODULES, (
            f"rejection path imports {module_name!r} (only pure modules allowed)"
        )
    resolved = _reachable_globals(nexus_pdf_extraction)
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
    assert resolved["PDF_EXTRACTION"] == PDF_EXTRACTION
    # No extraction engine, and none of the path-heuristic helpers the legacy
    # tool uses, is reachable from the rejection path.
    for forbidden in (
        "DoclingEngine",
        "GrobidEngine",
        "PyMuPDFEngine",
        "_pdf_metadata",
        "_resolve_path",
    ):
        assert forbidden not in resolved, f"rejection path must not reach {forbidden!r}"
    # The whole reachable surface is the two pure envelope builders plus the
    # declaration module's own names -- no adapter-local state, no I/O.
    assert all(
        getattr(target, "__module__", "scholar_agent.capabilities").startswith(
            ("scholar_agent", "json", "builtins")
        )
        for target in resolved.values()
    )


# --------------------------------------------------------------------------- #
# (e) Adapter scope guards
# --------------------------------------------------------------------------- #


def test_e2_neg_019_adapter_contains_no_pdf_extraction_domain_logic():
    """The adapter declares/rejects only: no engine, sidecar, or I/O logic."""
    body = _body_of(nexus_pdf_extraction)
    for forbidden in (
        "DoclingEngine",
        "GrobidEngine",
        "PyMuPDFEngine",
        "extract_markdown",
        "_pdf_metadata",
        "sidecar",
        "Path(",
        "mkdir",
        "write_text",
        "write_bytes",
        "open(",
    ):
        assert forbidden not in body, f"adapter body must not contain {forbidden!r}"
    # The one statement executed by the tool is the pure envelope builder.
    assert [line.strip() for line in body.strip().splitlines() if line.strip()] == [
        "return unsupported_capability_envelope_json(PDF_EXTRACTION)"
    ]


def test_e2_neg_019_capabilities_module_exposes_only_declaration_symbols():
    """No PDF domain symbols leaked into the declaration module."""
    for name in vars(caps):
        assert "download" not in name.lower()
        assert "ingest" not in name.lower()
        assert "manifest_builder" not in name.lower()
        assert "sidecar_builder" not in name.lower()
    assert not hasattr(caps, "extract_pdf")
    assert not hasattr(caps, "sidecar")


def test_e2_neg_019_extraction_capability_adds_no_new_envelope_builder():
    """One generic pure builder serves both boundaries; no per-capability fork.

    A capability-specific builder is how a rejection path would quietly grow
    I/O, so the count is pinned rather than merely observed.
    """
    builders = sorted(
        name
        for name in vars(caps)
        if name.startswith("unsupported_capability_envelope")
    )
    assert builders == [
        "unsupported_capability_envelope",
        "unsupported_capability_envelope_json",
    ]


# --------------------------------------------------------------------------- #
# (f) E2-NEG-019: the raw-path tool is declared non-authoritative
# --------------------------------------------------------------------------- #


def test_e2_neg_019_legacy_raw_path_tool_is_declared_non_authoritative():
    """Its own docstring says: verifies nothing, identifies nothing, no Contract."""
    doc = nexus_extract_pdf.__doc__ or ""
    lowered = " ".join(doc.lower().split())
    assert "non-authoritative" in lowered
    assert "verifies nothing" in lowered
    assert "identifies nothing" in lowered
    assert "heuristic" in lowered
    assert "not a verified" in lowered
    # The prohibited claims are named as prohibited, not merely omitted.
    assert "contract v1 artifact" in lowered
    assert "sidecar" in lowered
    assert "identity" in lowered
    # ...and the authoritative alternative is named.
    assert "scholar-pdf-kit" in lowered
    assert "scholar-pdf extract-run" in doc
    assert "scholar_pdf" in doc


def test_e2_neg_019_legacy_raw_path_tool_behaviour_is_unchanged():
    """Docstring-only change: every executable statement is pinned verbatim.

    This is the evidence for "non-authoritative, not removed or rewritten": the
    convenience path still works for exploration, and the pre-existing blind
    ``except Exception`` and path-derived metadata are retained as declared
    status quo (finding 9) rather than quietly changed here.
    """
    body = _body_of(nexus_extract_pdf)
    assert [line.strip() for line in body.splitlines() if line.strip()] == (
        LEGACY_BODY_STATEMENTS
    )
    signature = inspect.signature(nexus_extract_pdf)
    assert list(signature.parameters) == ["pdf_path", "output_dir", "engine"]
    # The legacy tool keeps its required positional path; only the new boundary
    # tool defaults every argument, because it never reads them.
    assert signature.parameters["pdf_path"].default is inspect.Parameter.empty
    assert signature.parameters["output_dir"].default == "./extracted"
    assert signature.parameters["engine"].default == "pymupdf"
    assert LEGACY_TOOL_NAME in _registered_tool_names()


def test_e2_neg_043_legacy_path_heuristic_metadata_is_status_quo_and_unreachable():
    """The path heuristic still exists; the rejection path cannot reach it.

    ``_pdf_metadata`` is pinned as declared status quo: a filename stem is not a
    title and a regex over a path is not a verified DOI. E2-NEG-043 bars those
    values from authoritative output, which is enforced where authority is
    created (the PDF kit and the harness adapter) -- not by silently changing
    this kit's legacy convenience behaviour.
    """
    source = inspect.getsource(server_module._pdf_metadata)
    assert 'pdf.stem.replace("_", " ")' in source
    assert 're.search(r"10\\.\\d{4,9}' in source
    assert 're.search(r"SCI-\\d+"' in source
    # Unreachable from the boundary: layer (d) asserts this, restated as an
    # explicit capability so a future refactor cannot quietly couple them.
    assert "_pdf_metadata" not in _reachable_globals(nexus_pdf_extraction)


# --------------------------------------------------------------------------- #
# (g) E2-NEG-020: no silent broaden of E1's acquisition boundary
# --------------------------------------------------------------------------- #


def test_e2_neg_020_pdf_acquisition_declaration_is_byte_identical_to_e1():
    """E2 added a key; it did not mutate, broaden, or re-interpret E1's fact."""
    declaration = get_capability(PDF_ACQUISITION)
    assert declaration.name == E1_EXPECTED_ACQUISITION["name"]
    assert declaration.owner == E1_EXPECTED_ACQUISITION["owner"]
    assert declaration.owning_surfaces == E1_EXPECTED_ACQUISITION["owning_surfaces"]
    assert declaration.mcp_supported is E1_EXPECTED_ACQUISITION["mcp_supported"]
    assert declaration.rejection_code == E1_EXPECTED_ACQUISITION["rejection_code"]
    assert (
        declaration.rejection_operation
        == E1_EXPECTED_ACQUISITION["rejection_operation"]
    )
    assert declaration.reference == E1_EXPECTED_ACQUISITION["reference"]
    assert declaration.alternatives == E1_EXPECTED_ACQUISITION["alternatives"]
    # Not merely equivalent to the hardcoded E1 facts: the same object E1 pinned.
    assert declaration is caps.PDF_ACQUISITION_DECLARATION


def test_e2_neg_020_acquire_pdf_rejection_envelope_is_unchanged_and_mentions_no_extraction():
    """E1's envelope still answers ``acquire_pdf`` and says nothing about E2."""
    envelope = unsupported_capability_envelope(PDF_ACQUISITION)
    assert envelope["operation"] == "acquire_pdf"
    assert envelope["status"] == "FAILED"
    assert envelope["artifacts"] == []
    assert envelope["warnings"] == []
    assert len(envelope["errors"]) == 1
    error = envelope["errors"][0]
    assert error["code"] == REJECTION_CODE
    assert error["retryable"] is False
    assert error["details"]["capability"] == "pdf_acquisition"
    assert error["details"]["reference"] == E1_EXPECTED_ACQUISITION["reference"]
    # The strongest form of "no silent broaden": E2's vocabulary is entirely
    # absent from the acquisition envelope.
    serialized = json.dumps(envelope)
    for marker in _E2_ONLY_MARKERS:
        assert marker not in serialized, (
            f"acquisition envelope must not mention E2 marker {marker!r}"
        )


def test_e2_neg_020_acquire_rejection_tool_still_returns_its_own_envelope():
    """End-to-end: the E1 tool and the E2 tool answer with different operations."""
    from scholar_agent.server import nexus_pdf_acquire

    acquisition = _envelope(nexus_pdf_acquire(workspace_id="SCI-000001"))
    extraction = _envelope(nexus_pdf_extraction(pdf_path="x.pdf"))
    assert acquisition["operation"] == "acquire_pdf"
    assert extraction["operation"] == "extract_pdf"
    assert acquisition["errors"][0]["details"]["capability"] == "pdf_acquisition"
    assert extraction["errors"][0]["details"]["capability"] == "pdf_extraction"
    assert acquisition != extraction
    # Neither envelope claims to be the other capability's success.
    assert acquisition["artifacts"] == extraction["artifacts"] == []
