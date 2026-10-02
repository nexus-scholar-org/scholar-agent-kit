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
    # The typed index service is imported here, not at module scope: this test
    # states a version floor on scholar-rag-kit (the T-90 index service), and a
    # rag-kit older than that floor must fail as this one test rather than as a
    # collection error that hides every other test in this module.
    from scholar_rag.chunker import text_fingerprint
    from scholar_rag.cli import DEFAULT_CHUNKER_CONFIGURATION
    from scholar_rag.embedder import get_embedder
    from scholar_rag.index_models import IndexDocumentRequest
    from scholar_rag.index_service import (
        IndexedSource,
        IndexServiceRequest,
        index_workspace,
    )
    from scholar_rag.index_verifier import ChromaVisibleSetReader
    from scholar_rag.replacement import ChromaReplacementView

    # Setup protocol
    compile_res = json.loads(nexus_protocol_compile(sample_intent_json))
    proto_file = tmp_path / "protocol.json"
    proto_file.write_text(json.dumps(compile_res["protocol"]), encoding="utf-8")

    # Create extracted markdown
    docs_dir = tmp_path / "extracted"
    docs_dir.mkdir()
    md_file = docs_dir / "SCI-000001.md"
    extracted_text = (
        "---\n"
        "workspace_id: \"SCI-000001\"\n"
        "doi: \"10.1038/agent\"\n"
        "title: \"Autonomous Agent Benchmark Study\"\n"
        "authors: [\"Alice\"]\n"
        "year: 2024\n"
        "---\n\n"
        "# Autonomous Agent Benchmark Study\n\n"
        "## Abstract\nEvaluation of agents on complex coding benchmarks.\n\n"
        "## Results\nEmpirical accuracy reached 98.4% on standard benchmarks.\n"
    )
    md_file.write_text(extracted_text, encoding="utf-8")

    db_path = str(tmp_path / "chroma_db")

    # 1. Index -- through the supported indexing surface, not through MCP.
    #
    # E3 declares RAG indexing unsupported over MCP: nexus_rag_index can only
    # answer with the UNSUPPORTED_CAPABILITY envelope pinned in
    # tests/test_mcp_indexing_boundary.py, so it cannot seed anything. The
    # alternative that boundary names is scholar-rag-kit's typed index service,
    # and that is what runs here: the same IndexServiceRequest /
    # index_workspace pair the `scholar-rag index` CLI calls, over the same two
    # store views (section 9.1, E3-NEG-040/041). Nothing below is a second
    # indexing implementation -- the request, the run, the commit and the
    # check-6 verification are scholar-rag-kit's own.
    workspace_root = tmp_path
    # scholar-rag-kit writes its *own* run report, never the adapter-owned
    # canonical ledger audit/journal.jsonl: the service refuses that
    # destination by construction (G-9), so the request states a kit path
    # beneath the workspace root and everything stays under tmp_path.
    journal_relative = "run-reports/rag-index.jsonl"
    (tmp_path / "run-reports").mkdir()
    # Retrieval below reads the store through ScholarRetriever, whose defaults
    # are the `scholar_docs` collection and the all-MiniLM-L6-v2 embedding
    # identity. Seeding a different collection would leave steps 2-4 querying
    # an empty store, and seeding with a different embedding identity would put
    # two vector spaces in one result set -- exactly what T-60's R1 refuses --
    # so the seed declares the identity the reader uses.
    collection = "scholar_docs"
    embedder_model = "all-MiniLM-L6-v2"
    embedder_dimension = 384
    hnsw_space = "cosine"
    declared_embedder = get_embedder(provider="sentence-transformers", model_name=embedder_model)

    def embedder(texts):
        # chromadb 1.5 refuses numpy scalars inside add(embeddings=[[...]]),
        # and the replacement view hands the embedder's own rows straight to it,
        # so the vectors are cast to plain floats. A dtype adapter, not a second
        # embedder: these are the vectors the declared model produced.
        return [[float(value) for value in vector] for vector in declared_embedder(list(texts))]

    assert len(embedder(["dimension probe"])[0]) == embedder_dimension, (
        "the declared dimension must be the one the declared embedder produces (R3)"
    )

    # Identity is stated, never inferred: no filename, title, DOI or
    # project.json field is an identity channel, so the identities below are
    # literal constants and not the frontmatter's SCI-000001.
    run_id = "RUN-" + "a" * 32
    workspace_id = "WSP-" + "0" * 32
    study_id = "STU-" + "4" * 32
    document_id = "DOC-" + "1" * 32
    parent_artifact_id = "ART-" + "1" * 32
    parent_sha256 = "sha256:" + "1" * 64
    extracted_relative = "extracted/SCI-000001.md"
    content_fingerprint = text_fingerprint(extracted_text)

    source = IndexedSource(
        request=IndexDocumentRequest(
            workspace_id=workspace_id,
            study_id=study_id,
            document_id=document_id,
            parent_artifact_id=parent_artifact_id,
            parent_artifact_sha256=parent_sha256,
            extracted_content_sha256=content_fingerprint,
            backend_provider="sentence-transformers",
            collection=collection,
            run_id=run_id,
        ),
        extracted_text=extracted_text,
        extracted_path=extracted_relative,
        extraction_method="DETERMINISTIC_RULE",
    )
    # The accepted parent view is the adapter's, supplied rather than
    # discovered: it admits exactly the one document above and states the
    # extracted artifact the eligibility join proves was accepted.
    parent_view = {
        "artifact_id": parent_artifact_id,
        "artifact_type": "document_manifest",
        "sha256": parent_sha256,
        "workspace_id": workspace_id,
        "protocol_fingerprint": "sha256:" + "2" * 64,
        "corpus_fingerprint": "sha256:" + "3" * 64,
        "documents": [
            {
                "document_id": document_id,
                "study_id": study_id,
                "extracted_path": extracted_relative,
                "extracted_content_sha256": content_fingerprint,
                "extraction_method": "DETERMINISTIC_RULE",
            }
        ],
    }
    request = IndexServiceRequest(
        run_id=run_id,
        created_at="2026-09-29T00:00:00Z",
        sources=(source,),
        parent_view=parent_view,
        chunker_configuration=dict(DEFAULT_CHUNKER_CONFIGURATION),
        backend_type="chroma",
        collection_name=collection,
        storage_schema_version="chroma-2",
        hnsw_space=hnsw_space,
        embedder_provider="sentence-transformers",
        embedder_model=embedder_model,
        embedder_dimension=embedder_dimension,
        embedder_normalize_embeddings=True,
        embedder_distance_metric="cosine",
        producer_version="0.2.0",
        producer_commit="c89b68f0d35173082a03b8c6b228e84381271185",
        journal_path=journal_relative,
        docs_path="extracted",
    )
    index_result = index_workspace(
        request,
        backend=ChromaReplacementView(
            db_path=db_path, collection_name=collection, embedder=embedder, hnsw_space=hnsw_space
        ),
        reader=ChromaVisibleSetReader(db_path=db_path, collection_name=collection),
        embedder=embedder,
        workspace_root=workspace_root,
    )

    # A typed, complete run -- never a free-text success string, and never a
    # partial run read as a complete one (C-29).
    assert index_result.outcome == "SUCCESS", index_result.envelope()
    assert index_result.complete is True
    assert index_result.codes == ()
    assert index_result.live_set_matches is True
    assert index_result.journaled is True
    assert index_result.counts.accepted_documents == 1
    assert index_result.counts.rejected_documents == 0
    assert index_result.counts.visible_chunks >= 1
    # Read the store back through the reader the service verified it with, so
    # the seed proves an index exists instead of claiming one.
    stored = ChromaVisibleSetReader(db_path=db_path, collection_name=collection)
    assert stored.visible_count() == index_result.counts.visible_chunks
    assert stored.visible_count() >= 1
    # The sidecar, the commit intent and the kit's own run report are all
    # written beneath the stated workspace root, and the adapter-owned ledger is
    # never created.
    assert (tmp_path / index_result.sidecar_path).exists()
    assert (tmp_path / index_result.intent_path).exists()
    assert (tmp_path / journal_relative).exists()
    assert not (tmp_path / "audit" / "journal.jsonl").exists()

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
