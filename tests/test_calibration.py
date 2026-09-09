"""Unit tests for the screener calibration module in scholar-agent-kit."""

import json

import pytest
from scholar_agent.calibration import (
    build_checklist_schema,
    build_preflight_calibration,
    checklist_to_decision,
    evaluate_calibration,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_protocol():
    return {
        "metadata": {"title": "UAV-CV Calibration Test"},
        "research_questions": [{"id": "RQ1", "text": "How accurate is DL crop segmentation?"}],
        "screening_criteria": {
            "inclusion": [
                {"id": "INC-01", "criterion": "Studies evaluating DL segmentation on crop or weed images captured from UAVs"},
                {"id": "INC-02", "criterion": "Reports quantitative accuracy metrics (mIoU, F1, etc.)"},
            ],
            "exclusion": [
                {"id": "EXC-01", "criterion": "Satellite-only studies without UAV imagery", "reason_category": "OUT_OF_SCOPE_RESOLUTION"},
                {"id": "EXC-02", "criterion": "Pure review papers with no empirical evaluation", "reason_category": "NO_EMPIRICAL_EVALUATION"},
            ],
        },
    }


@pytest.fixture
def gold_papers():
    return [
        {
            "workspace_id": f"CAL-{i+1:04d}",
            "title": f"Test Paper {i+1}",
            "year": 2023,
            "abstract": "Some abstract text",
            "doi": f"10.1000/test.{i+1}",
            "gold_decision": "INCLUDE" if i in (0, 2, 4) else "EXCLUDE",
            "gold_matched_inclusion": ["INC-01", "INC-02"] if i in (0, 2, 4) else [],
            "gold_violated_exclusion": [] if i in (0, 2, 4) else ["EXC-01"],
        }
        for i in range(20)
    ]


# ---------------------------------------------------------------------------
# Checklist schema tests
# ---------------------------------------------------------------------------

def test_build_checklist_schema(simple_protocol):
    schema = build_checklist_schema(simple_protocol)
    assert len(schema) == 4
    inc_fields = [s for s in schema if s["criterion_type"] == "inclusion"]
    exc_fields = [s for s in schema if s["criterion_type"] == "exclusion"]
    assert len(inc_fields) == 2
    assert len(exc_fields) == 2
    assert inc_fields[0]["field_name"] == "inc_01"
    assert inc_fields[1]["field_name"] == "inc_02"
    assert exc_fields[0]["field_name"] == "exc_01"
    assert exc_fields[1]["field_name"] == "exc_02"


def test_build_checklist_schema_empty_protocol():
    schema = build_checklist_schema({})
    assert schema == []


# ---------------------------------------------------------------------------
# checklist_to_decision tests
# ---------------------------------------------------------------------------

def test_checklist_all_true_inclusions_all_false_exclusions(simple_protocol):
    schema = build_checklist_schema(simple_protocol)
    decision = checklist_to_decision(
        workspace_id="SCI-0001",
        checklist={"inc_01": True, "inc_02": True, "exc_01": False, "exc_02": False},
        schema=schema,
        document_title="Perfect UAV Crop Paper",
    )
    assert decision.decision == "INCLUDE"
    assert decision.confidence == 0.90
    assert decision.matched_inclusion_criteria == ["INC-01", "INC-02"]
    assert decision.violated_exclusion_criteria == []


def test_checklist_exclusion_triggered(simple_protocol):
    schema = build_checklist_schema(simple_protocol)
    decision = checklist_to_decision(
        workspace_id="SCI-0002",
        checklist={"inc_01": True, "inc_02": True, "exc_01": True, "exc_02": False},
        schema=schema,
    )
    assert decision.decision == "EXCLUDE"
    assert decision.confidence == 0.55
    assert decision.violated_exclusion_criteria == ["EXC-01"]


def test_checklist_inclusion_missing(simple_protocol):
    schema = build_checklist_schema(simple_protocol)
    decision = checklist_to_decision(
        workspace_id="SCI-0003",
        checklist={"inc_01": False, "inc_02": True, "exc_01": False, "exc_02": False},
        schema=schema,
    )
    assert decision.decision == "EXCLUDE"
    assert decision.confidence == 0.75
    assert decision.matched_inclusion_criteria == ["INC-02"]


# ---------------------------------------------------------------------------
# Pre-flight calibration builder tests
# ---------------------------------------------------------------------------

def test_build_preflight_calibration(gold_papers, simple_protocol):
    batch = build_preflight_calibration(gold_papers, simple_protocol, sample_size=20, seed=7)
    assert len(batch.batch_data["papers"]) == 20
    assert batch.batch_data["is_calibration"] is True
    assert len(batch.batch_data["checklist_schema"]) == 4
    assert batch.gold_data["is_calibration_gold"] is True
    assert len(batch.gold_data["papers"]) == 20


def test_build_preflight_calibration_writes_files(gold_papers, simple_protocol, tmp_path):
    batch = build_preflight_calibration(gold_papers, simple_protocol, sample_size=10, seed=99)
    screening_dir = tmp_path / "literature" / "screening"
    batch.write(screening_dir)
    assert (screening_dir / "calibration_batch_000.json").exists()
    assert (screening_dir / "calibration_gold.json").exists()
    batch_file = json.loads((screening_dir / "calibration_batch_000.json").read_text(encoding="utf-8"))
    gold_file = json.loads((screening_dir / "calibration_gold.json").read_text(encoding="utf-8"))
    assert batch_file["is_calibration"] is True
    assert gold_file["is_calibration_gold"] is True
    assert len(batch_file["papers"]) == 10
    assert len(gold_file["papers"]) == 10


def test_build_preflight_calibration_sample_size_clamp(gold_papers, simple_protocol):
    batch = build_preflight_calibration(gold_papers, simple_protocol, sample_size=200, seed=0)
    assert len(batch.batch_data["papers"]) == 20
    assert len(batch.gold_data["papers"]) == 20


# ---------------------------------------------------------------------------
# Calibration evaluator tests
# ---------------------------------------------------------------------------

def test_evaluate_calibration_perfect(simple_protocol):
    gold = [
        {"workspace_id": "CAL-0001", "gold_decision": "INCLUDE"},
        {"workspace_id": "CAL-0002", "gold_decision": "EXCLUDE"},
        {"workspace_id": "CAL-0003", "gold_decision": "INCLUDE"},
        {"workspace_id": "CAL-0004", "gold_decision": "EXCLUDE"},
    ]
    decisions = [
        {"workspace_id": "CAL-0001", "inc_01": True, "inc_02": True, "exc_01": False, "exc_02": False},
        {"workspace_id": "CAL-0002", "inc_01": False, "inc_02": False, "exc_01": True, "exc_02": False},
        {"workspace_id": "CAL-0003", "inc_01": True, "inc_02": True, "exc_01": False, "exc_02": False},
        {"workspace_id": "CAL-0004", "inc_01": False, "inc_02": False, "exc_01": True, "exc_02": False},
    ]
    report = evaluate_calibration(decisions, gold)
    assert report.verdict == "PASS"
    assert report.sensitivity == 1.0
    assert report.specificity == 1.0
    assert report.accuracy == 1.0
    assert report.inclusion_rate == 0.50
    assert report.gold_inclusion_rate == 0.50
    assert report.inclusion_rate_delta == 0.0


def test_evaluate_calibration_flag_on_drift():
    gold = [
        {"workspace_id": f"CAL-{i:04d}", "gold_decision": "INCLUDE" if i < 3 else "EXCLUDE"}
        for i in range(10)
    ]
    decisions = [
        {"workspace_id": f"CAL-{i:04d}", "inc_01": True, "exc_01": False}
        for i in range(10)
    ]
    report = evaluate_calibration(decisions, gold, inclusion_rate_tolerance=0.15)
    assert report.verdict == "FLAG"
    assert any("drift" in r.lower() for r in report.reasons)


def test_evaluate_calibration_flag_on_low_sensitivity():
    gold = [
        {"workspace_id": "CAL-0001", "gold_decision": "INCLUDE"},
        {"workspace_id": "CAL-0002", "gold_decision": "INCLUDE"},
    ]
    decisions = [
        {"workspace_id": "CAL-0001", "inc_01": False, "exc_01": False},
        {"workspace_id": "CAL-0002", "inc_01": False, "exc_01": False},
    ]
    report = evaluate_calibration(decisions, gold, sensitivity_threshold=0.80)
    assert report.verdict == "FLAG"
    assert report.sensitivity == 0.0


def test_evaluate_calibration_empty_gold():
    report = evaluate_calibration([], [])
    assert report.verdict == "FLAG"
    assert "Empty gold set" in report.reasons


def test_evaluate_calibration_missing_decisions():
    gold = [
        {"workspace_id": "CAL-0001", "gold_decision": "INCLUDE"},
        {"workspace_id": "CAL-0002", "gold_decision": "EXCLUDE"},
    ]
    report = evaluate_calibration([], gold)
    assert report.sensitivity == 0.0
    assert report.specificity == 1.0
    assert report.accuracy == 0.5
