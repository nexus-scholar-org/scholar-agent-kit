"""Declared cross-surface capability registry for the Nexus Scholar MCP server.

This module is the **declaration half** of the PDF-acquisition MCP boundary
required by WP01-E1 (Packet E1 -- Acquired-Document Boundary) section 4.7 and
acceptance criterion E1-016. It contains no PDF domain logic whatsoever: no
download, ingest, validation, storage, or manifest construction. The canonical
PDF kit (``nexus-scholar-org/scholar-pdf-kit``) remains the domain-service
owner, and API/CLI are the supported E1 acquisition surfaces.

Why a declaration instead of a silent omission
----------------------------------------------
An acquisition-shaped request that is merely *absent* from the MCP surface is
indistinguishable from a client bug, a schema mistake, or a tool that vanished
in a refactor. E1 therefore requires the boundary to be **observable**: the
capability is declared as unsupported, and an acquisition-shaped MCP call
returns a structured, non-retryable rejection instead of an implicit 404.

The declared facts are:

* capability ``pdf_acquisition`` -> ``mcp_supported=False``;
* owning surfaces -> ``("API", "CLI")`` (both route through the same public
  PDF-kit domain service);
* stable rejection code -> ``UNSUPPORTED_CAPABILITY``;
* rejection envelope -> the standard operation-envelope vocabulary
  (``operation``/``status``/``artifacts``/``warnings``/``errors``) with
  ``operation="acquire_pdf"`` and ``status="FAILED"``.

Envelope contract notes (deliberate, and asserted by the conformance suite)
---------------------------------------------------------------------------
* ``UNSUPPORTED_CAPABILITY`` is carried as a plain string code. The frozen
  Contract v1 ``contract-error`` schema types ``code`` as
  ``anyOf[ErrorCode, string]``, so a non-enum code is a supported extension
  point: no Contract v1 change, and no edit to the frozen registries, is
  required to express this rejection.
* The envelope intentionally omits ``contract_version`` and ``run_id``. The
  rejection is unconditional and happens *before* any request validation, so
  echoing lineage would mean fabricating a run identity. The envelope uses the
  operation-envelope field vocabulary; it is not a Contract v1
  ``OperationOutcome`` and does not claim to be one.
* ``artifacts`` is always present and always empty: "no artifacts" must be
  explicit, not merely absent.
* Exactly one error is emitted, with ``retryable=False``. Retrying a declared
  capability boundary cannot succeed, so it is never retryable.

Purity / zero-I/O guarantee
---------------------------
``acquisition_rejection_envelope`` and ``acquisition_rejection_envelope_json``
are pure functions: they read an in-memory immutable mapping and build a fresh
``dict``/``str``. They perform no provider transport, no temporary or final file
creation, no manifest creation, and no audit append. This is what makes the
zero-I/O property of the rejection path structurally provable rather than
merely asserted (see ``tests/test_mcp_acquisition_capability.py``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

# --------------------------------------------------------------------------- #
# Stable identifiers
# --------------------------------------------------------------------------- #

#: Capability name used by the registry, the skill, and the surface matrix.
PDF_ACQUISITION = "pdf_acquisition"

#: Stable, non-retryable rejection code. Not a planner ``BLOCKED_*`` label and
#: not a Contract v1 ``ErrorCode`` member; the frozen contract-error schema
#: permits any string code, so this needs no contract change.
UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"

#: Operation name carried by the rejection envelope.
ACQUIRE_PDF_OPERATION = "acquire_pdf"

#: Surfaces that own (and support) PDF acquisition in E1.
SUPPORTED_OWNING_SURFACES = ("API", "CLI")

#: Surfaces that are served by this kit's MCP server.
MCP_SURFACE = "MCP"

#: Canonical owner of the acquisition domain service.
ACQUISITION_OWNER = "nexus-scholar-org/scholar-pdf-kit"

#: Normative reference for the declaration (harness-side architecture doc).
E1_REFERENCE = "docs/architecture/wp01_packet_e1_acquired_document_handoff.md#4.7"

#: The E1 alternatives a caller must be redirected to, named verbatim so the
#: rejection message is actionable rather than a bare refusal.
ACQUIRE_CLI_ALTERNATIVE = (
    "`scholar-pdf acquire` CLI "
    "(e.g. `uv run scholar-pdf acquire --request <request.json>`)"
)
ACQUIRE_API_ALTERNATIVE = (
    "Python `scholar_pdf.acquisition` API (acquisition request/outcome models)"
)

ACQUISITION_REJECTION_MESSAGE = (
    "E1 PDF acquisition is not available through MCP: capability "
    "'pdf_acquisition' is declared unsupported on the MCP surface, and this "
    "request was rejected before any provider transport, file, manifest, or "
    "audit I/O. Use the owning scholar-pdf-kit surfaces instead -- the "
    f"{ACQUIRE_CLI_ALTERNATIVE} or the {ACQUIRE_API_ALTERNATIVE}. API and CLI "
    "are the supported E1 acquisition surfaces and route through the same "
    "public PDF-kit domain service; this is a declared unsupported difference, "
    "not a parity claim."
)


@dataclass(frozen=True)
class CapabilityDeclaration:
    """One immutable cross-surface capability declaration.

    Attributes
    ----------
    name:
        Registry key (e.g. ``pdf_acquisition``).
    summary:
        One-line statement of what the capability is.
    owner:
        Canonical repository/surface that owns the domain service.
    owning_surfaces:
        Surfaces that own and support the capability (the PDF kit is ``API``
        and ``CLI``; this kit's MCP server is not one of them).
    mcp_supported:
        Whether this kit's MCP server serves the capability.
    rejection_code:
        Stable non-retryable code returned when the MCP surface is asked for
        the capability; ``None`` for capabilities MCP does support.
    rejection_operation:
        ``operation`` value of the rejection envelope.
    rejection_message:
        Human-readable rejection naming the supported alternatives.
    alternatives:
        Ordered, human-readable names of the supported surfaces.
    reference:
        Normative reference for the declaration.
    """

    name: str
    summary: str
    owner: str
    owning_surfaces: tuple[str, ...]
    mcp_supported: bool
    rejection_code: str | None
    rejection_operation: str
    rejection_message: str
    alternatives: tuple[str, ...]
    reference: str

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready projection of the declaration (used by parity fixtures)."""
        return {
            "name": self.name,
            "summary": self.summary,
            "owner": self.owner,
            "owning_surfaces": list(self.owning_surfaces),
            "mcp_supported": self.mcp_supported,
            "rejection_code": self.rejection_code,
            "rejection_operation": self.rejection_operation,
            "alternatives": list(self.alternatives),
            "reference": self.reference,
        }


PDF_ACQUISITION_DECLARATION = CapabilityDeclaration(
    name=PDF_ACQUISITION,
    summary=(
        "Deterministic acquired-document boundary: validate and commit exact "
        "PDF bytes for an accepted study (download or USER_PATH ingest)."
    ),
    owner=ACQUISITION_OWNER,
    owning_surfaces=SUPPORTED_OWNING_SURFACES,
    mcp_supported=False,
    rejection_code=UNSUPPORTED_CAPABILITY,
    rejection_operation=ACQUIRE_PDF_OPERATION,
    rejection_message=ACQUISITION_REJECTION_MESSAGE,
    alternatives=(ACQUIRE_CLI_ALTERNATIVE, ACQUIRE_API_ALTERNATIVE),
    reference=E1_REFERENCE,
)

#: The registry. Immutable: declarations are observable facts, not mutable
#: configuration, and must not be flipped at runtime.
CAPABILITIES: Mapping[str, CapabilityDeclaration] = MappingProxyType(
    {PDF_ACQUISITION: PDF_ACQUISITION_DECLARATION}
)


class UnknownCapabilityError(KeyError):
    """Raised when a capability is not present in the declaration registry."""


def get_capability(name: str) -> CapabilityDeclaration:
    """Return the declaration for ``name``.

    Raises
    ------
    UnknownCapabilityError
        If the capability was never declared. A missing declaration is a bug
        in the registry, never a silent "unsupported" answer.
    """
    try:
        return CAPABILITIES[name]
    except KeyError as exc:
        raise UnknownCapabilityError(
            f"capability {name!r} is not declared in the agent-kit capability "
            f"registry; declared capabilities: {sorted(CAPABILITIES)}"
        ) from exc


def unsupported_capability_envelope(name: str) -> dict[str, Any]:
    """Build the standard rejection envelope for a capability (pure).

    The returned mapping mirrors the operation-envelope field vocabulary and
    is fail-closed: ``status`` is always ``FAILED``, ``artifacts`` is always
    empty, and exactly one non-retryable error is present. It performs no I/O.
    """
    declaration = get_capability(name)
    if declaration.mcp_supported or declaration.rejection_code is None:
        raise ValueError(
            f"capability {name!r} is not declared unsupported on the MCP "
            "surface; no rejection envelope is defined for it"
        )
    error: dict[str, Any] = {
        "code": declaration.rejection_code,
        "message": declaration.rejection_message,
        "retryable": False,
        "details": {
            "capability": declaration.name,
            "mcp_supported": False,
            "owning_surfaces": list(declaration.owning_surfaces),
            "owner": declaration.owner,
            "alternatives": list(declaration.alternatives),
            "reference": declaration.reference,
        },
    }
    return {
        "operation": declaration.rejection_operation,
        "status": "FAILED",
        "artifacts": [],
        "warnings": [],
        "errors": [error],
    }


def unsupported_capability_envelope_json(name: str) -> str:
    """JSON-serialised :func:`unsupported_capability_envelope` (pure)."""
    return json.dumps(unsupported_capability_envelope(name), sort_keys=True)
