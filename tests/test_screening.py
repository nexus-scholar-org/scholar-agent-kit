"""Tests for nexus_screen_llm MCP tool (Phase D: LLM-enhanced screening)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scholar_agent.server import nexus_screen_llm, mcp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_papers_json(tmp_path: Path) -> str:
    """Write a minimal papers JSON file and return its path as a string."""
    papers = [
        {
            "title": "Agent Benchmark Evaluation",
            "authors": ["Alice"],
            "year": 2024,
            "doi": "10.1038/s1",
            "abstract": "Reports benchmark pass rates of 95%.",
        },
        {
            "title": "Non-English Editorial",
            "authors": ["Bob"],
            "year": 2024,
            "doi": "10.1038/s2",
            "abstract": "Non-English commentary on unrelated topics.",
        },
    ]
    p = tmp_path / "papers.json"
    p.write_text(json.dumps(papers), encoding="utf-8")
    return str(p)


@pytest.fixture
def sample_protocol_json(tmp_path: Path) -> str:
    """Write a minimal protocol JSON file and return its path as a string."""
    protocol = {
        "protocol_id": "test-proto",
        "metadata": {"title": "Agent Test Review"},
        "research_questions": [{"id": "RQ1", "text": "How do agents perform?"}],
        "screening_criteria": {
            "inclusion": [{"id": "INC-01", "criterion": "Reports benchmark results"}],
            "exclusion": [{"id": "EXC-01", "criterion": "Non-English language"}],
        },
    }
    p = tmp_path / "protocol.json"
    p.write_text(json.dumps(protocol), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# Conformance
# ---------------------------------------------------------------------------


def test_screen_llm_tool_registered():
    """nexus_screen_llm must be in the MCP tool registry."""
    registered = {t.name for t in mcp._tool_manager.list_tools()}
    assert "nexus_screen_llm" in registered


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_screen_llm_missing_input():
    """Calling with a nonexistent input file returns an error JSON."""
    result = nexus_screen_llm(
        input_path="/nonexistent/papers.json",
        protocol_path="/nonexistent/protocol.json",
    )
    data = json.loads(result)
    assert data["status"] == "ERROR"
    assert "not found" in data["error"].lower()


def test_screen_llm_missing_protocol(sample_papers_json):
    """Calling with a valid input but missing protocol returns an error JSON."""
    result = nexus_screen_llm(
        input_path=sample_papers_json,
        protocol_path="/nonexistent/protocol.json",
    )
    data = json.loads(result)
    assert data["status"] == "ERROR"
    assert "not found" in data["error"].lower()


# ---------------------------------------------------------------------------
# Happy path (mocked LLM)
# ---------------------------------------------------------------------------


def _make_screening_decision(ws_id: str, decision: str):
    """Create a real ScreeningDecision object."""
    from scholar_search.screening import ScreeningDecision

    return ScreeningDecision(
        workspace_id=ws_id,
        decision=decision,
        confidence=0.90 if decision == "INCLUDE" else 0.75,
        matched_inclusion_criteria=["INC-01"] if decision == "INCLUDE" else [],
        violated_exclusion_criteria=["EXC-01"] if decision == "EXCLUDE" else [],
        relevant_rqs=["RQ1"],
        screening_reasoning="Mock decision",
        document_title=f"Paper {ws_id}",
        doi=f"10.1000/{ws_id}",
    )


def test_screen_llm_happy_path_mock_llm(
    tmp_path, sample_papers_json, sample_protocol_json
):
    """Mock LLM returns 1 INCLUDE + 1 EXCLUDE; verify 5 output files written."""
    mock_decisions = [
        _make_screening_decision("SCI-000001", "INCLUDE"),
        _make_screening_decision("SCI-000002", "EXCLUDE"),
    ]

    mock_screener = MagicMock()
    mock_screener.screen = AsyncMock(return_value=mock_decisions)

    # Patch at the source module since imports are deferred inside function body
    with patch("scholar_search.screening.LLMBatchScreener", return_value=mock_screener):
        result = nexus_screen_llm(
            input_path=sample_papers_json,
            protocol_path=sample_protocol_json,
            output_dir=str(tmp_path / "literature"),
            api_key="test-key",
        )

    data = json.loads(result)
    assert data["status"] == "SUCCESS"
    assert data["included"] >= 1
    assert data["excluded"] >= 1

    out = tmp_path / "literature"
    assert (out / "included.json").exists()
    assert (out / "excluded.json").exists()
    assert (out / "conflicts.json").exists()
    assert (out / "prisma_report.json").exists()
    assert (out / "prisma_screening_report.md").exists()


# ---------------------------------------------------------------------------
# Heuristic fallback on LLM failure
# ---------------------------------------------------------------------------


def test_screen_llm_fallback_on_llm_failure(
    tmp_path, sample_papers_json, sample_protocol_json
):
    """When LLM raises, heuristic fallback still produces valid output files."""
    mock_screener = MagicMock()
    mock_screener.screen = AsyncMock(side_effect=ConnectionError("LLM timeout"))

    with patch("scholar_search.screening.LLMBatchScreener", return_value=mock_screener):
        result = nexus_screen_llm(
            input_path=sample_papers_json,
            protocol_path=sample_protocol_json,
            output_dir=str(tmp_path / "literature"),
            api_key="test-key",
        )

    data = json.loads(result)
    assert data["status"] == "SUCCESS"

    out = tmp_path / "literature"
    assert (out / "included.json").exists()
    assert (out / "excluded.json").exists()
    assert (out / "conflicts.json").exists()
    assert (out / "prisma_report.json").exists()
    assert (out / "prisma_screening_report.md").exists()


def test_screen_llm_fallback_on_constructor_error(
    tmp_path, sample_papers_json, sample_protocol_json
):
    """When LLMBatchScreener constructor raises, heuristic fallback still works."""
    with patch(
        "scholar_search.screening.LLMBatchScreener",
        side_effect=ValueError("No API key"),
    ):
        result = nexus_screen_llm(
            input_path=sample_papers_json,
            protocol_path=sample_protocol_json,
            output_dir=str(tmp_path / "literature"),
        )

    data = json.loads(result)
    assert data["status"] == "SUCCESS"

    out = tmp_path / "literature"
    assert (out / "included.json").exists()
    assert (out / "excluded.json").exists()


# ---------------------------------------------------------------------------
# Checklist schema integration
# ---------------------------------------------------------------------------


def test_screen_llm_checklist_schema_used(
    tmp_path, sample_papers_json, sample_protocol_json
):
    """Verify build_checklist_schema is invoked with protocol data."""
    from scholar_agent.calibration import build_checklist_schema

    mock_screener = MagicMock()
    mock_screener.screen = AsyncMock(
        return_value=[
            _make_screening_decision("SCI-000001", "EXCLUDE"),
        ]
    )

    with (
        patch("scholar_search.screening.LLMBatchScreener", return_value=mock_screener),
        patch(
            "scholar_agent.calibration.build_checklist_schema",
            side_effect=build_checklist_schema,
        ) as mock_schema,
    ):
        result = nexus_screen_llm(
            input_path=sample_papers_json,
            protocol_path=sample_protocol_json,
            output_dir=str(tmp_path / "literature"),
            api_key="test-key",
        )

    data = json.loads(result)
    assert data["status"] == "SUCCESS"
    mock_schema.assert_called_once()
