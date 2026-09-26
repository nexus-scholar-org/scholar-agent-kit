"""Declared cross-surface capability registry for the Nexus Scholar MCP server.

This module is the **declaration half** of the two declared-unsupported MCP
boundaries required by WP01-E1 (Packet E1 -- Acquired-Document Boundary)
section 4.7 / E1-016 and WP01-E2 (Packet E2 -- Extracted-Text Handoff)
section 9 / E2-013. It contains no PDF domain logic whatsoever: no download,
ingest, validation, extraction, storage, or manifest construction. The
canonical PDF kit (``nexus-scholar-org/scholar-pdf-kit``) remains the
domain-service owner, and API/CLI are the supported E1 acquisition and E2
extraction surfaces.

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

The E2 declared facts (PDF extraction, Packet E2 section 9)
-----------------------------------------------------------
E2 adds a **separate** declaration for extraction and leaves ``pdf_acquisition``
byte-identical (E2-NEG-020 -- no silent broaden). The facts are parallel but not
interchangeable, because acquisition and extraction are different domain
services in the same canonical owner:

* capability ``pdf_extraction`` -> ``mcp_supported=False``;
* owning surfaces -> ``("API", "CLI")`` (the same public PDF-kit domain
  service, exercised through a different entry point);
* stable rejection code -> ``UNSUPPORTED_CAPABILITY`` (the *same* code constant
  as E1 -- the boundary is "this surface does not serve this capability", not
  "a different kind of failure", so a second code would be a needless
  vocabulary fork);
* rejection envelope -> the same operation-envelope vocabulary with
  ``operation="extract_pdf"`` and ``status="FAILED"``.

Why extraction is declared rather than served
---------------------------------------------
The MCP surface does have a raw-path extraction tool (``nexus_extract_pdf``).
That tool is a **non-authoritative convenience**, not the E2 extraction
service: it verifies nothing, identifies nothing, derives ``title``/``doi``/
``workspace_id`` by heuristic from the path, and therefore cannot produce a
Contract artifact, a sidecar, or an identity-addressed output. E2-NEG-019 and
E2-NEG-043 require that status to be stated rather than implied, so the
authoritative boundary is made *observable* here: an extraction-shaped request
that reaches the declared-unsupported tool is rejected before any engine import
rather than answered with heuristic output that a caller could mistake for a
verified result.

Envelope contract notes (deliberate, and asserted by the conformance suite)
---------------------------------------------------------------------------
* ``UNSUPPORTED_CAPABILITY`` is carried as a plain string code. The frozen
  Contract v1 ``contract-error`` schema types ``code`` as
  ``anyOf[ErrorCode, string]``, so a non-enum code is a supported extension
  point: no Contract v1 change, and no edit to the frozen registries, is
  required to express this rejection. Both E1 and E2 reuse the one constant.
* The envelope intentionally omits ``contract_version`` and ``run_id``. The
  rejection is unconditional and happens *before* any request validation, so
  echoing lineage would mean fabricating a run identity. The envelope uses the
  operation-envelope field vocabulary; it is not a Contract v1
  ``OperationOutcome`` and does not claim to be one. This holds for the E2
  envelope too: a rejected extraction request has no run to belong to.
* ``artifacts`` is always present and always empty: "no artifacts" must be
  explicit, not merely absent.
* Exactly one error is emitted, with ``retryable=False``. Retrying a declared
  capability boundary cannot succeed, so it is never retryable.

Purity / zero-I/O guarantee
---------------------------
``unsupported_capability_envelope`` and ``unsupported_capability_envelope_json``
are the single, generic pair of pure builders behind *both* declarations. They
read an in-memory immutable mapping and build a fresh ``dict``/``str``. They
perform no engine import, no provider transport, no temporary or final file
creation, no manifest or sidecar construction, and no audit append. This is
what makes the zero-I/O property of the rejection path structurally provable
rather than merely asserted (see ``tests/test_mcp_acquisition_capability.py``
and ``tests/test_mcp_extraction_capability.py``). No capability-specific
builder was added, and none may be: one generic pure path is what keeps
"zero I/O" a property of the code rather than of each call site's discipline.
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

#: E2's capability name. A separate registry key from :data:`PDF_ACQUISITION`:
#: acquisition (validate and commit exact PDF bytes) and extraction (fulltext
#: production from committed bytes) are distinct domain services, so E2 must not
#: broaden, narrow, or re-interpret the E1 declaration to cover extraction.
PDF_EXTRACTION = "pdf_extraction"

#: Stable, non-retryable rejection code. Not a planner ``BLOCKED_*`` label and
#: not a Contract v1 ``ErrorCode`` member; the frozen contract-error schema
#: permits any string code, so this needs no contract change. Shared by E1 and
#: E2: the meaning is "this surface does not serve this capability", not "a
#: different kind of failure", so a second code would only fork the vocabulary.
UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"

#: Operation name carried by the rejection envelope.
ACQUIRE_PDF_OPERATION = "acquire_pdf"

#: Operation name carried by the E2 rejection envelope. Deliberately distinct
#: from :data:`ACQUIRE_PDF_OPERATION` so the two declared boundaries remain
#: distinguishable from the envelope alone.
EXTRACT_PDF_OPERATION = "extract_pdf"

#: Surfaces that own (and support) PDF acquisition in E1.
SUPPORTED_OWNING_SURFACES = ("API", "CLI")

#: Surfaces that are served by this kit's MCP server.
MCP_SURFACE = "MCP"

#: Canonical owner of the acquisition domain service.
ACQUISITION_OWNER = "nexus-scholar-org/scholar-pdf-kit"

#: Canonical owner of the E2 extraction domain service. The same canonical kit
#: as acquisition, on purpose: one domain owner, two capabilities.
EXTRACTION_OWNER = "nexus-scholar-org/scholar-pdf-kit"

#: Normative reference for the declaration (harness-side architecture doc).
E1_REFERENCE = "docs/architecture/wp01_packet_e1_acquired_document_handoff.md#4.7"

#: Normative reference for the E2 declaration (Packet E2 section 9 is the MCP
#: surface boundary this declaration implements).
E2_REFERENCE = "docs/architecture/wp01_packet_e2_extracted_text_handoff.md#9"

#: The E1 alternatives a caller must be redirected to, named verbatim so the
#: rejection message is actionable rather than a bare refusal.
ACQUIRE_CLI_ALTERNATIVE = (
    "`scholar-pdf acquire` CLI (e.g. `uv run scholar-pdf acquire <config.json>`)"
)
ACQUIRE_API_ALTERNATIVE = (
    "Python `scholar_pdf.acquisition` API (acquisition request/outcome models)"
)

#: The authoritative E2 extraction alternatives, named verbatim for the same
#: reason. Stage 1 deliberately retained ``scholar-pdf extract <path>`` and
#: ``scholar_pdf.extract`` as non-authoritative conveniences; redirecting an MCP
#: rejection to either would erase the very boundary this declaration exists to
#: expose. The parent-bound service is ``extract-run`` /
#: :class:`scholar_pdf.extraction.PDFExtractionService`.
EXTRACT_CLI_ALTERNATIVE = (
    "`scholar-pdf extract-run` CLI (e.g. `uv run scholar-pdf extract-run "
    "<config.json> --audit-logger <path-to-log_event.py>`)"
)
EXTRACT_API_ALTERNATIVE = (
    "Python `scholar_pdf.extraction.PDFExtractionService` API "
    "(parent-bound extraction request/outcome service)"
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

EXTRACTION_REJECTION_MESSAGE = (
    "E2 PDF extraction is not available through MCP: capability "
    "'pdf_extraction' is declared unsupported on the MCP surface, and this "
    "request was rejected before any engine import, provider transport, file, "
    "manifest/sidecar, or audit I/O. Use the owning scholar-pdf-kit surfaces "
    f"instead -- the {EXTRACT_CLI_ALTERNATIVE} or the {EXTRACT_API_ALTERNATIVE}"
    ". API and CLI are the supported E2 extraction surfaces and route through "
    "the same public PDF-kit domain service; this is a declared unsupported "
    "difference, not a parity claim. In particular, this rejection asserts no "
    "parity between MCP output and the authoritative PDF-kit result, and no "
    "metadata, identity, or status may be derived from a filename, URL, or a "
    "regex over a path."
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

PDF_EXTRACTION_DECLARATION = CapabilityDeclaration(
    name=PDF_EXTRACTION,
    summary=(
        "Deterministic extracted-text boundary: produce fulltext for a study "
        "from bytes already committed by the E1 acquisition manifest "
        "(manifest-bound, identity-addressed, sidecar-committed)."
    ),
    owner=EXTRACTION_OWNER,
    owning_surfaces=SUPPORTED_OWNING_SURFACES,
    mcp_supported=False,
    rejection_code=UNSUPPORTED_CAPABILITY,
    rejection_operation=EXTRACT_PDF_OPERATION,
    rejection_message=EXTRACTION_REJECTION_MESSAGE,
    alternatives=(EXTRACT_CLI_ALTERNATIVE, EXTRACT_API_ALTERNATIVE),
    reference=E2_REFERENCE,
)

#: The registry. Immutable: declarations are observable facts, not mutable
#: configuration, and must not be flipped at runtime. E2 adds a second key and
#: changes nothing about the first one (E2-NEG-020).
CAPABILITIES: Mapping[str, CapabilityDeclaration] = MappingProxyType(
    {
        PDF_ACQUISITION: PDF_ACQUISITION_DECLARATION,
        PDF_EXTRACTION: PDF_EXTRACTION_DECLARATION,
    }
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
