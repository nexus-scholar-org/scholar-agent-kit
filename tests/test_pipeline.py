"""Tests for nexus_pipeline_run MCP tool (Phase D: pipeline orchestration)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scholar_agent.server import nexus_pipeline_run, mcp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace_dir(tmp_path: Path) -> str:
    """Create a minimal workspace directory and return its path."""
    ws = tmp_path / "my-review"
    ws.mkdir()
    (ws / "protocol.json").write_text(
        json.dumps({"protocol_id": "test-proto", "metadata": {"title": "Test"}}),
        encoding="utf-8",
    )
    return str(ws)


# ---------------------------------------------------------------------------
# Conformance
# ---------------------------------------------------------------------------


def test_pipeline_tool_registered():
    """nexus_pipeline_run must be in the MCP tool registry."""
    registered = {t.name for t in mcp._tool_manager.list_tools()}
    assert "nexus_pipeline_run" in registered


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_pipeline_missing_workspace():
    """Calling with a nonexistent workspace dir returns an error JSON."""
    result = nexus_pipeline_run(workspace_dir="/nonexistent/workspace")
    data = json.loads(result)
    assert data["status"] == "ERROR"
    assert "not found" in data["error"].lower()


# ---------------------------------------------------------------------------
# Happy path (mocked orchestrator)
# ---------------------------------------------------------------------------


def test_pipeline_happy_path_mock(tmp_path, workspace_dir):
    """Mock ResearchOrchestrator returns success; verify JSON response."""
    mock_orch = MagicMock()
    mock_orch.run_pipeline_async = AsyncMock(
        return_value={
            "status": "SUCCESS",
            "stages": {"discovery": 10, "dedup": 8, "screening": 6},
        }
    )

    # Patch at source module since import is deferred inside function body
    with patch(
        "scholar_harness.orchestrator.ResearchOrchestrator", return_value=mock_orch
    ):
        result = nexus_pipeline_run(workspace_dir=workspace_dir)

    data = json.loads(result)
    assert data["status"] == "SUCCESS"
    assert data["stages"]["discovery"] == 10
    assert data["workspace"] == str(Path(workspace_dir).resolve())


# ---------------------------------------------------------------------------
# Skip stages
# ---------------------------------------------------------------------------


def test_pipeline_skip_stages(tmp_path, workspace_dir):
    """Pass skip_stages='discovery,dedup'; verify those keys are removed."""
    mock_orch = MagicMock()
    mock_orch.run_pipeline_async = AsyncMock(
        return_value={
            "status": "SUCCESS",
            "stages": {"discovery": 10, "dedup": 8, "screening": 6},
        }
    )

    with patch(
        "scholar_harness.orchestrator.ResearchOrchestrator", return_value=mock_orch
    ):
        result = nexus_pipeline_run(
            workspace_dir=workspace_dir,
            skip_stages="discovery,dedup",
        )

    data = json.loads(result)
    assert data["status"] == "SUCCESS"
    assert "discovery" not in data["stages"]
    assert "dedup" not in data["stages"]
    assert "screening" in data["stages"]


# ---------------------------------------------------------------------------
# Orchestrator error
# ---------------------------------------------------------------------------


def test_pipeline_orchestrator_error(tmp_path, workspace_dir):
    """When orchestrator raises, verify error JSON is returned."""
    mock_orch = MagicMock()
    mock_orch.run_pipeline_async = AsyncMock(
        side_effect=FileNotFoundError("Protocol file not found")
    )

    with patch(
        "scholar_harness.orchestrator.ResearchOrchestrator", return_value=mock_orch
    ):
        result = nexus_pipeline_run(workspace_dir=workspace_dir)

    data = json.loads(result)
    assert data["status"] == "ERROR"
    assert "Protocol file not found" in data["error"]


# ---------------------------------------------------------------------------
# With query parameter
# ---------------------------------------------------------------------------


def test_pipeline_with_query(tmp_path, workspace_dir):
    """Pass a query parameter; verify orchestrator is called with protocol_path."""
    mock_orch = MagicMock()
    mock_orch.run_pipeline_async = AsyncMock(
        return_value={
            "status": "SUCCESS",
            "stages": {},
        }
    )

    with patch(
        "scholar_harness.orchestrator.ResearchOrchestrator", return_value=mock_orch
    ):
        result = nexus_pipeline_run(
            workspace_dir=workspace_dir,
            query="autonomous agents",
        )

    data = json.loads(result)
    assert data["status"] == "SUCCESS"
    # Verify the orchestrator was called
    mock_orch.run_pipeline_async.assert_called_once()
    call_kwargs = mock_orch.run_pipeline_async.call_args
    assert "protocol_path" in call_kwargs.kwargs or len(call_kwargs.args) >= 1
