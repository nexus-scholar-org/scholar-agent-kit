"""Unit tests for FastMCP Server in scholar-agent-kit."""

import json
from pathlib import Path
import pytest

from scholar_agent.server import (
    nexus_protocol_compile,
    nexus_protocol_validate,
    nexus_protocol_render_criteria,
    nexus_discover,
    nexus_dedup,
    nexus_screen,
    nexus_rag_index,
    nexus_rag_query,
    nexus_rag_synthesize,
    nexus_matrix_extract,
    nexus_graph_build,
)


@pytest.fixture
def sample_intent_json():
    return json.dumps({
        "protocol_id": "proto-agent-test",
        "genesis_timestamp": "2026-09-01T00:00:00+00:00",
        "project_slug": "agent-test-review",
        "playbook_type": "DESIGN_SCIENCE",
        "title": "Agent Test Evaluation",
        "lead_researcher": "Agent Tester",
        "unit_of_analysis": "Autonomous Agents",
        "epistemological_rationale": "Empirical Benchmark",
        "research_questions": [
            {
                "text": "How do agents perform on benchmark X?",
                "target_facet": "evaluation_metrics",
                "required_evidence_type": "Quantitative Benchmark"
            }
        ],
        "core_concepts": [
            {"concept": "Agent", "synonyms": ["autonomous assistant"]}
        ],
        "inclusion_criteria": [
            {"criterion": "Reports benchmark pass rates", "maps_to_rqs": ["RQ1"]}
        ],
        "exclusion_criteria": [
            {"criterion": "Non-English", "reason_category": "LANGUAGE", "maps_to_rqs": ["RQ1"]}
        ],
        "matrix_dimensions": [
            {"id": "sample_size", "name": "Sample Size", "description": "Number of runs"}
        ]
    })


def test_nexus_protocol_compile_and_validate(tmp_path, sample_intent_json):
    # 1. Compile
    compile_res_raw = nexus_protocol_compile(sample_intent_json)
    compile_res = json.loads(compile_res_raw)
    assert compile_res["status"] == "SUCCESS"
    assert compile_res["protocol_id"] == "proto-agent-test"
    assert "fingerprint" in compile_res

    # Save to temp protocol file
    proto_file = tmp_path / "protocol.json"
    proto_file.write_text(json.dumps(compile_res["protocol"]), encoding="utf-8")

    # 2. Validate
    validate_res_raw = nexus_protocol_validate(str(proto_file))
    validate_res = json.loads(validate_res_raw)
    assert validate_res["status"] == "VALID"
    assert validate_res["title"] == "Agent Test Evaluation"

    # 3. Render Criteria
    criteria_md = nexus_protocol_render_criteria(str(proto_file))
    assert "Screening Criteria" in criteria_md
    assert "Agent Test Evaluation" in criteria_md


def test_nexus_dedup_and_screen(tmp_path, sample_intent_json):
    # Setup protocol
    compile_res = json.loads(nexus_protocol_compile(sample_intent_json))
    proto_file = tmp_path / "protocol.json"
    proto_file.write_text(json.dumps(compile_res["protocol"]), encoding="utf-8")

    # Setup raw docs
    raw_docs = [
        {"title": "Agent Benchmark Evaluation", "authors": ["Alice"], "year": 2024, "doi": "10.1038/s1", "abstract": "Reports benchmark pass rates of 95%."},
        {"title": "Agent Benchmark Evaluation", "authors": ["Alice"], "year": 2024, "doi": "10.1038/s1", "abstract": "Duplicate copy."},
        {"title": "Non-English Editorial", "authors": ["Bob"], "year": 2024, "doi": "10.1038/s2", "abstract": "Non-English commentary."}
    ]
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(raw_docs), encoding="utf-8")

    # Deduplicate
    dedup_file = tmp_path / "deduped.json"
    dedup_res = nexus_dedup(str(raw_file), str(dedup_file))
    assert "Successfully deduplicated 3 papers into 2 unique" in dedup_res
    assert dedup_file.exists()

    # Screen
    lit_dir = tmp_path / "literature"
    screen_res = nexus_screen(str(dedup_file), str(proto_file), str(lit_dir))
    assert "Screening complete" in screen_res
    assert (lit_dir / "included.json").exists()
    assert (lit_dir / "excluded.json").exists()
    assert (lit_dir / "prisma_screening_report.md").exists()


def test_nexus_rag_and_matrix_tools(tmp_path, sample_intent_json):
    # Setup protocol
    compile_res = json.loads(nexus_protocol_compile(sample_intent_json))
    proto_file = tmp_path / "protocol.json"
    proto_file.write_text(json.dumps(compile_res["protocol"]), encoding="utf-8")

    # Create extracted markdown
    docs_dir = tmp_path / "extracted"
    docs_dir.mkdir()
    md_file = docs_dir / "SCI-000001.md"
    md_file.write_text(
        "---\n"
        "workspace_id: \"SCI-000001\"\n"
        "doi: \"10.1038/agent\"\n"
        "title: \"Autonomous Agent Benchmark Study\"\n"
        "authors: [\"Alice\"]\n"
        "year: 2024\n"
        "---\n\n"
        "# Autonomous Agent Benchmark Study\n\n"
        "## Abstract\nEvaluation of agents on complex coding benchmarks.\n\n"
        "## Results\nEmpirical accuracy reached 98.4% on standard benchmarks.\n",
        encoding="utf-8"
    )

    db_path = str(tmp_path / "chroma_db")

    # 1. Index
    idx_res = nexus_rag_index(docs_dir=str(docs_dir), db_path=db_path, workspace_id="agent-test")
    assert "Successfully indexed 1 files" in idx_res

    # 2. Query
    query_res = nexus_rag_query(query="empirical accuracy", db_path=db_path)
    assert "Result 1" in query_res

    # 3. Synthesize
    synth_res = nexus_rag_synthesize(query="What is the empirical accuracy?", db_path=db_path)
    assert "Grounded Synthesis" in synth_res

    # 4. Matrix Extract
    lit_dir = tmp_path / "literature"
    matrix_res = nexus_matrix_extract(workspace_dir=str(tmp_path), protocol_path=str(proto_file), output_dir=str(lit_dir))
    assert "Successfully extracted dynamic matrix" in matrix_res
    assert (lit_dir / "synthesis_matrix.csv").exists()
