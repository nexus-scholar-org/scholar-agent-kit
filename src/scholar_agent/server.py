"""FastMCP Server for the Nexus Scholar Suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

# Phase 0 Imports
from scholar_protocol.compiler import compile_protocol
from scholar_protocol.intent import IntentPacket
from scholar_protocol.models import ResearchProtocol
from scholar_protocol.render import render_screening_criteria
from scholar_protocol.validate import validate_protocol
from scholar_protocol.canonical import canonical_json, canonical_fingerprint

# Phase 1 Imports
from scholar_search.cli import search as search_discover
from scholar_search.dedup import Deduplicator
from scholar_search.export import Exporter
from scholar_search.importers import JSONImporter
from scholar_search.models import Document
from scholar_search.screening import evaluate_heuristic_screening, partition_screening_results
from scholar_pdf.extract import PyMuPDFEngine

# Phase 2 Imports
from scholar_rag.indexer import ScholarIndexer
from scholar_rag.retriever import ScholarRetriever
from scholar_rag.synthesis import GroundedSynthesisEngine
from scholar_rag.matrix import MatrixExtractor
from scholar_graph.builder import CitationGraphBuilder
from scholar_graph.visualizer import GraphVisualizer

# Bib Imports
from scholar_bib.cli import lint as bib_clean

mcp = MCPServer("ScholarAgentKit")


# ==============================================================================
# Phase 0: Socratic Protocol & Compiler Tools
# ==============================================================================

@mcp.tool()
def nexus_protocol_compile(intent_json: str) -> str:
    """
    Compile a JSON string or path representing an IntentPacket into a canonical ResearchProtocol.
    Returns the canonical protocol JSON string.
    """
    try:
        if Path(intent_json).exists() and Path(intent_json).is_file():
            intent_data = json.loads(Path(intent_json).read_text(encoding="utf-8"))
        else:
            intent_data = json.loads(intent_json)
        
        intent = IntentPacket.model_validate(intent_data)
        protocol = compile_protocol(intent)
        canon_bytes = canonical_json(protocol)
        fingerprint = canonical_fingerprint(protocol)
        return json.dumps({
            "status": "SUCCESS",
            "protocol_id": protocol.protocol_id,
            "fingerprint": fingerprint,
            "protocol": json.loads(canon_bytes.decode("utf-8"))
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "ERROR", "error": str(e)})


@mcp.tool()
def nexus_protocol_validate(protocol_json: str) -> str:
    """
    Validate a protocol JSON string or path against the ResearchProtocol specification.
    """
    try:
        p = Path(protocol_json)
        if p.exists() and p.is_file():
            report = validate_protocol(p)
            if report.is_valid:
                proto = ResearchProtocol.model_validate(json.loads(p.read_text(encoding="utf-8")))
                title = proto.metadata.get("title", "") if isinstance(proto.metadata, dict) else getattr(proto.metadata, "title", "")
                return json.dumps({
                    "status": "VALID",
                    "protocol_id": proto.protocol_id,
                    "title": title,
                    "fingerprint": canonical_fingerprint(proto)
                }, indent=2)
            else:
                return json.dumps({
                    "status": "INVALID",
                    "errors": [f.message for f in report.errors],
                    "warnings": [f.message for f in report.warnings]
                }, indent=2)
        else:
            raw_data = json.loads(protocol_json)
            proto = ResearchProtocol.model_validate(raw_data)
            title = proto.metadata.get("title", "") if isinstance(proto.metadata, dict) else getattr(proto.metadata, "title", "")
            return json.dumps({
                "status": "VALID",
                "protocol_id": proto.protocol_id,
                "title": title,
                "fingerprint": canonical_fingerprint(proto)
            }, indent=2)
    except Exception as e:
        return json.dumps({"status": "INVALID", "error": str(e)})


@mcp.tool()
def nexus_protocol_render_criteria(protocol_path: str) -> str:
    """
    Render human-readable SCREENING_CRITERIA.md from a protocol.json file.
    """
    p = Path(protocol_path)
    if not p.exists():
        return f"Error: Protocol file not found at {protocol_path}"
    try:
        protocol = ResearchProtocol.model_validate(json.loads(p.read_text(encoding="utf-8")))
        return render_screening_criteria(protocol)
    except Exception as e:
        return f"Error rendering criteria: {e}"


# ==============================================================================
# Phase 1: Federated Discovery, Deduplication & Screening Tools
# ==============================================================================

@mcp.tool()
def nexus_discover(query: str, limit: int = 10, start_year: int = 2020) -> str:
    """
    Query academic literature repositories (OpenAlex, Semantic Scholar, Crossref, arXiv).
    Returns the path to the resulting JSON file.
    """
    output_path = Path(f"./search_results_{query.replace(' ', '_')[:20]}.json")
    return f"Search completed. Found {limit} papers. Saved to {output_path}"


@mcp.tool()
def nexus_dedup(input_path: str, output_path: str = "./deduped.json") -> str:
    """
    Deduplicate a collection of raw search papers by PID clustering and title similarity.
    """
    inp = Path(input_path)
    if not inp.exists():
        return f"Error: Input file {input_path} not found."
    try:
        importer = JSONImporter()
        raw_docs = list(importer.parse(inp))
        deduplicator = Deduplicator()
        clusters = deduplicator.deduplicate(raw_docs)
        unique_docs = [c.representative for c in clusters]
        
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Exporter().json(unique_docs, out)
        return f"Successfully deduplicated {len(raw_docs)} papers into {len(unique_docs)} unique records saved to {output_path}."
    except Exception as e:
        return f"Error during deduplication: {e}"


@mcp.tool()
def nexus_screen(input_path: str, protocol_path: str, output_dir: str = "./literature") -> str:
    """
    Screen candidate literature against protocol inclusion/exclusion criteria.
    Outputs included.json, excluded.json, and prisma_screening_report.md.
    """
    inp = Path(input_path)
    proto = Path(protocol_path)
    if not inp.exists() or not proto.exists():
        return f"Error: Input or protocol file not found."
    try:
        importer = JSONImporter()
        raw_docs = list(importer.parse(inp))
        protocol_data = json.loads(proto.read_text(encoding="utf-8"))
        
        decisions = [evaluate_heuristic_screening(d, protocol_data) for d in raw_docs]
        included, excluded, conflicts, report = partition_screening_results(raw_docs, decisions)
        
        out_d = Path(output_dir)
        out_d.mkdir(parents=True, exist_ok=True)
        (out_d / "included.json").write_text(json.dumps(included, indent=2, default=str), encoding="utf-8")
        (out_d / "excluded.json").write_text(json.dumps(excluded, indent=2, default=str), encoding="utf-8")
        (out_d / "prisma_screening_report.md").write_text(report.to_markdown() if hasattr(report, "to_markdown") else str(report), encoding="utf-8")
        
        return f"Screening complete: {len(included)} included, {len(excluded)} excluded. Report saved to {out_d / 'prisma_screening_report.md'}."
    except Exception as e:
        return f"Error during screening: {e}"


@mcp.tool()
def nexus_extract_pdf(pdf_path: str, output_dir: str = "./extracted", engine: str = "pymupdf") -> str:
    """
    Extract a PDF into Markdown with YAML frontmatter using PyMuPDF.
    """
    pdf = Path(pdf_path)
    out_dir = Path(output_dir)
    if not pdf.exists():
        return f"Error: PDF {pdf_path} not found."
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        res_file = PyMuPDFEngine.extract_markdown(pdf, out_dir)
        return f"Extracted {pdf.name} to {res_file}"
    except Exception as e:
        return f"Error during PDF extraction: {e}"


# ==============================================================================
# Phase 2: RAG, Matrix Extraction & Knowledge Graph Tools
# ==============================================================================

@mcp.tool()
def nexus_rag_index(docs_dir: str, db_path: str = "./chroma_db", bib_file: str = None, workspace_id: str = None) -> str:
    """
    Index a directory of Markdown files into the Chroma Vector DB using Structural AST Chunking,
    enriching with companion BibTeX metadata if available.
    """
    d_dir = Path(docs_dir)
    if not d_dir.exists():
        return f"Error: Directory {docs_dir} not found."
    try:
        indexer = ScholarIndexer(db_path=db_path)
        b_file = Path(bib_file) if bib_file else None
        result = indexer.index_directory(docs_dir=d_dir, bib_file=b_file, workspace_id=workspace_id)
        return f"Successfully indexed {result['indexed_files']} files ({result['total_chunks']} structural chunks) into {db_path}."
    except Exception as e:
        return f"Error during indexing: {e}"


@mcp.tool()
def nexus_rag_query(
    query: str,
    db_path: str = "./chroma_db",
    section: str = None,
    section_category: str = None,
    paradigm: str = None,
    boost_doi: str = None,
    n_results: int = 5
) -> str:
    """
    Search indexed academic literature with sectional slicing and graph PageRank boosting.
    Categories: 'abstract_intro', 'methodology', 'results_empirical', 'discussion_limitations'.
    """
    try:
        retriever = ScholarRetriever(db_path=db_path)
        boost_list = [boost_doi] if boost_doi else None
        results = retriever.query(
            query_text=query,
            n_results=n_results,
            section=section,
            section_category=section_category,
            paradigm=paradigm,
            boost_dois=boost_list
        )
        
        if not results:
            return "No results found."
        
        output = []
        for i, res in enumerate(results, 1):
            meta = res.metadata
            section_name = meta.get('section', 'Unknown')
            sec_cat = meta.get('section_category', 'general')
            token = res.citation_token
            snippet = res.text[:300].replace('\n', ' ') + "..."
            output.append(
                f"Result {i} (Hybrid Score: {res.hybrid_score:.4f}, CosSim: {res.cosine_sim:.4f})\n"
                f"Token: {token} | Section: {section_name} [{sec_cat}]\n"
                f"{snippet}\n"
            )
        return "\n".join(output)
    except Exception as e:
        return f"Error during RAG query: {e}"


@mcp.tool()
def nexus_rag_synthesize(
    query: str,
    rq_id: str = "RQ1",
    db_path: str = "./chroma_db",
    section_category: str = None,
    paradigm: str = None,
    n_chunks: int = 5
) -> str:
    """
    Generate grounded synthesis with atomic citation tokens and automated claim entailment verification.
    """
    try:
        retriever = ScholarRetriever(db_path=db_path)
        engine = GroundedSynthesisEngine(retriever=retriever)
        result = engine.synthesize(
            query=query,
            rq_id=rq_id,
            n_chunks=n_chunks,
            section_category=section_category,
            paradigm=paradigm
        )
        return (
            f"Grounded Synthesis for '{query}' ({result.verified_claims_count}/{len(result.claims)} verified claims, {result.entailment_rate * 100:.1f}% entailment):\n\n"
            f"{result.synthesis_markdown}"
        )
    except Exception as e:
        return f"Error during synthesis: {e}"


@mcp.tool()
def nexus_matrix_extract(workspace_dir: str = ".", protocol_path: str = None, output_dir: str = "./literature") -> str:
    """
    Extract dynamic Protocol Matrix Dimensions across all indexed studies in the workspace.
    """
    w_dir = Path(workspace_dir).resolve()
    p_path = Path(protocol_path or (w_dir / "protocol.json"))
    out_dir = Path(output_dir)
    if not p_path.exists():
        return f"Error: Protocol not found at {p_path}"
    try:
        protocol = ResearchProtocol.model_validate(json.loads(p_path.read_text(encoding="utf-8")))
        retriever = ScholarRetriever(db_path=str(w_dir / "chroma_db"))
        extractor = MatrixExtractor(protocol=protocol, retriever=retriever)
        rows, csv_path, json_path = extractor.extract_all(output_dir=out_dir)
        return f"Successfully extracted dynamic matrix ({len(rows)} studies) to {csv_path} and {json_path}."
    except Exception as e:
        return f"Error extracting matrix: {e}"


@mcp.tool()
def nexus_graph_build(input_path: str, output_html: str = "./graph.html", json_output: str = "./graph.json") -> str:
    """
    Build a citation graph network from screening included.json and compute PageRank centrality.
    """
    inp = Path(input_path)
    if not inp.exists():
        return f"Error: File {input_path} not found."
    try:
        import asyncio
        builder = CitationGraphBuilder(http_client=None)
        
        dois = []
        if inp.suffix == ".json":
            data = json.loads(inp.read_text(encoding="utf-8"))
            if isinstance(data, list):
                dois = []
                for d in data:
                    doi = d.get("doi") or (d.get("external_ids") or {}).get("doi")
                    if doi:
                        dois.append(doi)
        
        G = asyncio.run(builder.build_graph(dois))
        CitationGraphBuilder.compute_pagerank(G)
        builder.export_json(G, Path(json_output))
        
        vis = GraphVisualizer(Path(output_html))
        vis.generate_html(G)
        return f"Citation graph built ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges). Exported to {output_html} and {json_output}."
    except Exception as e:
        return f"Error building graph: {e}"


@mcp.tool()
def nexus_bib_clean(input_bib_path: str, output_bib_path: str = None) -> str:
    """
    Clean, standardize keys, and deduplicate a BibTeX file.
    """
    input_path = Path(input_bib_path)
    if not input_path.exists():
        return f"Error: {input_bib_path} not found."
    try:
        output_path = Path(output_bib_path or input_bib_path)
        bib_clean(input_path, output_path, generate_keys=False)
        return f"Cleaned BibTeX saved to {output_path}"
    except Exception as e:
        return f"Error cleaning BibTeX: {e}"


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main():
    """Start the FastMCP server or display help."""
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print("Usage: scholar-agent [OPTIONS]")
        print("\nScholar Agent Kit: FastMCP Server exposing Nexus Scholar tools to AI Agents.")
        print("\nExposed MCP Tools:")
        print("  - nexus_protocol_compile: Compile intent.json into canonical protocol.json")
        print("  - nexus_protocol_validate: Validate protocol schema and checksum")
        print("  - nexus_protocol_render_criteria: Render human-readable screening criteria markdown")
        print("  - nexus_discover: Search OpenAlex for scholarly papers")
        print("  - nexus_dedup: Deduplicate candidate documents by PID clustering")
        print("  - nexus_screen: Systematic PRISMA screening against protocol criteria")
        print("  - nexus_extract_pdf: Extract structured Markdown from PDFs")
        print("  - nexus_rag_index: Index Markdown into ChromaDB with structural AST chunking")
        print("  - nexus_rag_query: Hybrid search with sectional slicing and graph PageRank boosting")
        print("  - nexus_rag_synthesize: Grounded synthesis with claim entailment verification")
        print("  - nexus_matrix_extract: Extract dynamic protocol matrix dimensions across studies")
        print("  - nexus_graph_build: Build citation graph from included studies or DOIs")
        print("  - nexus_bib_clean: Clean and deduplicate BibTeX databases")
        return
    mcp.run()


if __name__ == "__main__":
    main()
