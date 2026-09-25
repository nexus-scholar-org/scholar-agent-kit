"""WP01-E1 declared MCP capability boundary for PDF acquisition (E1-NEG-047).

Packet E1 section 4.7 / acceptance criterion E1-016 require PDF acquisition to
be *declared* unsupported on the MCP surface rather than silently omitted:

* the capability registry declares ``pdf_acquisition`` with
  ``mcp_supported=False``, owning surfaces ``("API", "CLI")``, and the stable
  non-retryable rejection code ``UNSUPPORTED_CAPABILITY``;
* the boundary is observable on the real MCP surface (the tool is registered,
  discoverable through the public ``list_tools`` API, reachable through
  ``MCPServer.call_tool``, and listed in ``scholar-agent --help``);
* an acquisition-shaped request returns the standard operation envelope
  (``operation="acquire_pdf"``, ``status="FAILED"``, no artifacts, exactly one
  non-retryable ``UNSUPPORTED_CAPABILITY`` error naming the API/CLI
  alternatives), unconditionally and argument-independently;
* the rejection happens before any provider transport, temporary/final file
  creation, manifest creation, or audit-success append -- zero I/O;
* the canonical ``scholar-pdf-kit`` API/CLI remain the supported E1 surfaces
  (this kit only *names* them; it does not reimplement acquisition).

Every test here is offline, hermetic, and deterministic: no network, no PDF
fixtures, no provider, no writing outside ``tmp_path``. Zero-I/O evidence is
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
"""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import dis
import importlib
import inspect
import json
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
    UNSUPPORTED_CAPABILITY,
    CapabilityDeclaration,
    get_capability,
    unsupported_capability_envelope,
    unsupported_capability_envelope_json,
)
from scholar_agent.server import main, mcp, nexus_pdf_acquire

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

TOOL_NAME = "nexus_pdf_acquire"
REJECTION_CODE = "UNSUPPORTED_CAPABILITY"
REJECTION_OPERATION = "acquire_pdf"

#: The 23 MCP tools that existed before the E1 boundary was added. Asserted
#: individually so an unrelated tool cannot be renamed or dropped here.
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
#: the rejection carries no manifest/audit/success bookkeeping of any kind.
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


# --------------------------------------------------------------------------- #
# Representative acquisition-shaped arguments (Packet E1 section 3 shape).
# None of them is read: the rejection is unconditional.
# --------------------------------------------------------------------------- #

DISCOVERY_ARGS: dict[str, Any] = {
    "workspace_id": "SCI-000001",
    "workspace_root": "/tmp/does-not-exist/sci-review",
    "run_id": "RUN-000001",
    "study_id": "STU-000001",
    "protocol_fingerprint": "sha256:" + "1" * 64,
    "corpus_fingerprint": "sha256:" + "2" * 64,
    "inputs_json": json.dumps(
        [
            {
                "artifact_id": "SCR-000001",
                "sha256": "sha256:" + "3" * 64,
                "path": "literature/screening/decisions.json",
            }
        ]
    ),
    "doi": "10.1038/s41586-024-07000-0",
    "source_mode": "DISCOVERY",
    "source_path": "",
    "access_assertion_json": "",
    "validation_profile": "pdf_default",
    "validation_profile_version": "1.0.0",
}

USER_PATH_ARGS: dict[str, Any] = {
    **DISCOVERY_ARGS,
    "doi": "",
    "source_mode": "USER_PATH",
    "source_path": "incoming/author-copy.pdf",
    "access_assertion_json": json.dumps(
        {"supplied_by": "lead researcher", "permission_basis": "AUTHOR_COPY"}
    ),
}

HOSTILE_ARGS: dict[str, Any] = {
    "workspace_id": "SCI-000001\n../../etc/passwd",
    "workspace_root": "C:\\Windows\\Temp\\..",
    "run_id": "RUN-000001'; DROP TABLE audit; --",
    "study_id": "",
    "protocol_fingerprint": "not-a-fingerprint",
    "corpus_fingerprint": "not-a-fingerprint",
    "inputs_json": "{not json at all",
    "doi": "10.0000/" + "x" * 512,
    "source_mode": "SOMETHING_ELSE",
    "source_path": "/does/not/exist.pdf",
    "access_assertion_json": "null",
    "validation_profile": "",
    "validation_profile_version": "",
}

ARGUMENT_CASES: dict[str, dict[str, Any]] = {
    "no_arguments": {},
    "discovery_request_shape": DISCOVERY_ARGS,
    "user_path_request_shape": USER_PATH_ARGS,
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
    return get_capability(PDF_ACQUISITION)


def _registered_tool_names() -> set[str]:
    return {tool.name for tool in mcp._tool_manager.list_tools()}


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


def test_e1_neg_047_registry_declares_pdf_acquisition_mcp_unsupported():
    """E1-NEG-047: the registry declares pdf_acquisition unsupported on MCP."""
    declaration = _declaration()
    assert declaration.name == "pdf_acquisition"
    assert declaration.mcp_supported is False
    assert "pdf_acquisition" in caps.CAPABILITIES
    assert sorted(caps.CAPABILITIES) == ["pdf_acquisition"]


def test_e1_neg_047_registry_owning_surface_is_api_and_cli_owned_by_pdf_kit():
    """The canonical PDF kit owns acquisition through API/CLI, not this MCP kit."""
    declaration = _declaration()
    assert set(declaration.owning_surfaces) == {"API", "CLI"}
    assert caps.MCP_SURFACE == "MCP"
    assert caps.MCP_SURFACE not in declaration.owning_surfaces
    assert declaration.owner == caps.ACQUISITION_OWNER
    assert "scholar-pdf-kit" in declaration.owner


def test_e1_neg_047_registry_rejection_code_is_stable_unsupported_capability():
    """The declared rejection code is the stable UNSUPPORTED_CAPABILITY value."""
    declaration = _declaration()
    assert declaration.rejection_code == REJECTION_CODE == UNSUPPORTED_CAPABILITY
    assert caps.UNSUPPORTED_CAPABILITY == "UNSUPPORTED_CAPABILITY"
    assert declaration.rejection_operation == REJECTION_OPERATION == "acquire_pdf"
    assert caps.ACQUIRE_PDF_OPERATION == "acquire_pdf"
    assert caps.SUPPORTED_OWNING_SURFACES == ("API", "CLI")


def test_e1_neg_047_registry_projection_is_json_serialisable():
    """The declaration projects to a plain JSON-ready dict for parity fixtures."""
    projection = _declaration().as_dict()
    assert json.loads(json.dumps(projection))["mcp_supported"] is False
    assert projection["owning_surfaces"] == ["API", "CLI"]
    assert projection["rejection_code"] == REJECTION_CODE
    assert projection["alternatives"]


def test_e1_neg_047_registry_is_immutable_and_rejects_unknown_capability():
    """Declarations are immutable facts; an undeclared capability is a bug."""
    with pytest.raises(TypeError):
        caps.CAPABILITIES["pdf_extraction"] = _declaration()  # type: ignore[index]
    with pytest.raises(AttributeError):
        _declaration().mcp_supported = True  # type: ignore[misc]
    with pytest.raises(caps.UnknownCapabilityError):
        get_capability("pdf_acquisition_v2")


def test_e1_neg_047_no_rejection_envelope_for_a_capability_mcp_actually_serves(
    monkeypatch,
):
    """A supported capability must not be answered with a rejection envelope."""
    served = CapabilityDeclaration(
        name="pdf_extraction",
        summary="Extract fulltext from a local PDF.",
        owner="nexus-scholar-org/scholar-pdf-kit",
        owning_surfaces=("API", "CLI", "MCP"),
        mcp_supported=True,
        rejection_code=None,
        rejection_operation="extract_pdf",
        rejection_message="",
        alternatives=(),
        reference=caps.E1_REFERENCE,
    )
    monkeypatch.setattr(caps, "CAPABILITIES", {"pdf_extraction": served})
    with pytest.raises(ValueError, match="not declared unsupported"):
        unsupported_capability_envelope("pdf_extraction")


# --------------------------------------------------------------------------- #
# (b) The boundary is registered and discoverable (not silently omitted)
# --------------------------------------------------------------------------- #


def test_e1_neg_047_acquisition_rejection_tool_is_registered_on_mcp_server():
    """The acquisition-shaped tool exists on the module-level MCPServer."""
    assert server_module.mcp is mcp
    assert TOOL_NAME in _registered_tool_names()
    assert callable(nexus_pdf_acquire)
    assert nexus_pdf_acquire.__module__ == "scholar_agent.server"


def test_e1_neg_047_acquisition_rejection_tool_is_discoverable_via_list_tools():
    """Public discovery exposes the tool with a boundary-describing schema."""
    tools = asyncio.run(mcp.list_tools())
    tool = next(t for t in tools if t.name == TOOL_NAME)
    assert tool.description
    assert "UNSUPPORTED" in tool.description
    properties = tool.input_schema.get("properties", {})
    for field in (
        "workspace_id",
        "workspace_root",
        "run_id",
        "study_id",
        "source_mode",
        "source_path",
        "validation_profile",
        "validation_profile_version",
    ):
        assert field in properties, f"acquisition-shaped field {field} not exposed"


def test_e1_neg_047_acquisition_rejection_tool_docstring_declares_the_boundary():
    """The tool's own documentation states the boundary and the alternatives."""
    doc = nexus_pdf_acquire.__doc__ or ""
    lowered = doc.lower()
    assert "declared unsupported" in lowered
    assert "not available through mcp" in lowered
    assert "unsupported_capability" in lowered
    assert "acquire_pdf" in lowered
    assert "scholar-pdf" in doc
    assert "scholar_pdf" in doc
    assert "before any" in lowered


def test_e1_neg_047_acquisition_rejection_tool_is_listed_in_server_help(capsys):
    """`scholar-agent --help` enumerates the tool (MCP surface parity gate)."""
    with contextlib.suppress(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "Exposed MCP Tools" in out
    assert TOOL_NAME in out
    assert REJECTION_CODE in out


def test_e1_neg_047_pre_existing_tools_are_untouched():
    """The 23 pre-existing tools survive; the boundary is the 24th registration."""
    registered = _registered_tool_names()
    assert PRE_EXISTING_TOOLS <= registered
    assert sorted(registered - PRE_EXISTING_TOOLS) == [TOOL_NAME]
    assert len(registered) == len(PRE_EXISTING_TOOLS) + 1 == 24


# --------------------------------------------------------------------------- #
# (c) The rejection envelope
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", sorted(ARGUMENT_CASES))
def test_e1_neg_047_acquisition_shaped_call_returns_failed_envelope(case: str):
    """Any acquisition-shaped request -- valid, empty, or invalid -- is rejected."""
    envelope = _envelope(nexus_pdf_acquire(**ARGUMENT_CASES[case]))
    assert envelope["operation"] == REJECTION_OPERATION
    assert envelope["status"] == "FAILED"
    assert envelope["artifacts"] == []
    assert len(envelope["errors"]) == 1
    error = envelope["errors"][0]
    assert error["code"] == REJECTION_CODE
    assert error["retryable"] is False
    assert error["message"].strip()


def test_e1_neg_047_envelope_shape_is_exact_and_carries_no_success_bookkeeping():
    """Exact envelope vocabulary; no manifest/audit/success/lineage fabrication."""
    envelope = _envelope(nexus_pdf_acquire(**DISCOVERY_ARGS))
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


def test_e1_neg_047_envelope_is_deterministic_and_argument_independent():
    """The rejection is unconditional: same envelope for every argument set."""
    results = [nexus_pdf_acquire(**ARGUMENT_CASES[case]) for case in ARGUMENT_CASES]
    assert len(set(results)) == 1
    assert results[0] == unsupported_capability_envelope_json(PDF_ACQUISITION)
    assert json.loads(results[0]) == unsupported_capability_envelope(PDF_ACQUISITION)


def test_e1_neg_047_envelope_details_name_the_supported_api_and_cli_surfaces():
    """API/CLI parity note: both supported alternatives are named explicitly."""
    envelope = _envelope(nexus_pdf_acquire(**DISCOVERY_ARGS))
    details = envelope["errors"][0]["details"]
    assert details["capability"] == "pdf_acquisition"
    assert details["mcp_supported"] is False
    assert details["owning_surfaces"] == ["API", "CLI"]
    assert details["owner"] == caps.ACQUISITION_OWNER
    assert len(details["alternatives"]) == 2
    joined = " ".join(details["alternatives"])
    assert "scholar-pdf acquire" in joined
    assert "scholar_pdf.acquisition" in joined
    message = envelope["errors"][0]["message"]
    assert "not available through MCP" in message
    assert "scholar-pdf acquire" in message
    assert "scholar_pdf.acquisition" in message
    assert "API" in message and "CLI" in message


def test_e1_neg_047_rejection_message_states_the_pre_io_guarantee():
    """The message says the rejection precedes every I/O effect."""
    message = _envelope(nexus_pdf_acquire(**DISCOVERY_ARGS))["errors"][0]["message"]
    lowered = message.lower()
    assert "rejected before any provider transport" in lowered
    assert "manifest" in lowered
    assert "audit" in lowered
    assert "not a parity claim" in lowered


def test_e1_neg_047_envelope_error_object_is_contract_error_shape_compatible():
    """The frozen Contract v1 error shape accepts this code: no contract change."""
    from scholar_harness.contracts.models import ContractError, OperationStatus

    envelope = _envelope(nexus_pdf_acquire(**DISCOVERY_ARGS))
    validated = ContractError.model_validate(envelope["errors"][0])
    assert validated.code == REJECTION_CODE
    assert validated.retryable is False
    assert OperationStatus(envelope["status"]) is OperationStatus.FAILED
    # Hard-failure coherence: a FAILED outcome must carry at least one error.
    assert envelope["errors"]


def test_e1_neg_047_rejection_envelope_reaches_clients_through_mcp_dispatch():
    """End-to-end through MCPServer.call_tool, not only via direct import."""
    result = asyncio.run(mcp.call_tool(TOOL_NAME, dict(DISCOVERY_ARGS)))
    payload = json.loads(result.content[0].text)
    assert payload["operation"] == REJECTION_OPERATION
    assert payload["status"] == "FAILED"
    assert payload["artifacts"] == []
    assert payload["errors"][0]["code"] == REJECTION_CODE
    assert payload["errors"][0]["retryable"] is False


# --------------------------------------------------------------------------- #
# (d) Zero I/O on the rejection path
# --------------------------------------------------------------------------- #


def test_e1_neg_047_rejection_path_performs_no_provider_transport_or_filesystem_io(
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
            _envelope(nexus_pdf_acquire(**args))

    assert wire.owner_events() == []
    assert _snapshot(tmp_path) == before
    assert list(sandbox.iterdir()) == []
    assert list(workspace.iterdir()) == []
    assert Path.cwd() == sandbox.resolve()


def test_e1_neg_047_rejection_path_creates_no_manifest_and_no_audit_success_event(
    tmp_path, monkeypatch
):
    """No manifest and no staged/final file; the audit-sink layer is a guard.

    Honest scope: ``sink.events == []`` is a *naming-convention guard*, not
    independent evidence. ``server_module`` and ``caps`` have never bound
    ``log_event``/``audit_sink``/``AUDIT_SINK``/``append_audit``, so
    ``monkeypatch.setattr(..., raising=False)`` only injects names that nothing
    reads and the assertion cannot fail. It exists to catch a future rename
    adopting one of those conventional names.

    The substantive zero-I/O proof is layer (d), the static reachability walk
    over the rejection path's own bytecode, in
    ``test_e1_neg_047_rejection_path_cannot_reach_io_or_network_symbols``.

    The filesystem half of this test *is* substantive: the tripwires plus the
    before/after snapshot comparison, the empty workspace, and the absence of
    any manifest/journal/PDF are all backed by real observation and are
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
        envelope = _envelope(nexus_pdf_acquire(**DISCOVERY_ARGS))

    assert sink.events == []
    assert _snapshot(tmp_path) == before
    assert list(workspace.iterdir()) == []
    assert not list(workspace.rglob("*.json"))
    assert not list(workspace.rglob("*manifest*"))
    assert not list(workspace.rglob("*journal*"))
    assert not list(workspace.rglob("*.pdf"))
    # No success-shaped bookkeeping in the payload either.
    assert envelope["status"] == "FAILED"
    assert envelope["artifacts"] == []


def test_e1_neg_047_capabilities_module_binds_no_io_capable_module():
    """The declaration module itself has no I/O-capable module bound."""
    for name in _IO_MODULES:
        assert name not in vars(caps), f"{name} must not be bound in capabilities"
    assert "json" in vars(caps), "the declaration module needs json for serialisation"
    assert vars(caps)["json"].__name__ == "json"  # pure serialisation only


def test_e1_neg_047_rejection_path_cannot_reach_io_or_network_symbols():
    """Static proof: no I/O or network symbol is reachable from the tool body."""
    for module_name in sorted(_referenced_imports(nexus_pdf_acquire)):
        assert module_name in _ALLOWED_MODULES, (
            f"rejection path imports {module_name!r} (only pure modules allowed)"
        )
    resolved = _reachable_globals(nexus_pdf_acquire)
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
    assert resolved["PDF_ACQUISITION"] == PDF_ACQUISITION
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


def test_e1_neg_047_adapter_contains_no_pdf_domain_logic():
    """The adapter declares/rejects only: no PDF download/ingest/validation."""
    source = inspect.getsource(nexus_pdf_acquire)
    body = source.split('"""', 2)[-1]
    for forbidden in (
        "AsyncPDFDownloader",
        "download_batch",
        "ingest_pdf",
        "is_valid_pdf",
        "validate_pdf_structure",
        "aiohttp",
        "httpx",
        "requests",
        "urlopen",
        "mkdir",
        "write_text",
        "open(",
    ):
        assert forbidden not in body, f"adapter body must not contain {forbidden!r}"
    # The one statement executed by the tool is the pure envelope builder.
    assert [line.strip() for line in body.strip().splitlines() if line.strip()] == [
        "return unsupported_capability_envelope_json(PDF_ACQUISITION)"
    ]


def test_e1_neg_047_capabilities_module_exposes_only_declaration_symbols():
    """No PDF domain symbols leaked into the declaration module."""
    for name in vars(caps):
        assert "download" not in name.lower()
        assert "ingest" not in name.lower()
        assert "manifest_builder" not in name.lower()
    assert not hasattr(caps, "acquire_pdf")
