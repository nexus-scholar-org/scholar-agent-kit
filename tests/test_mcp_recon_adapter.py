"""HCM-05 adapter-wiring regression guard: kit recon tools via the harness adapter.

The recon tools (``recon_probe`` / ``recon_distill`` / ``recon_delta``) drive
the harness exploratory-recon layers ONLY through the explicit adapter
interface (``scholar_harness.recon.adapter`` +
``scholar_harness.recon.lexicon.DomainLexicon``) bound by
``_ensure_harness_recon_loaded()``. There is no source discovery and no
standalone harness distribution.

This module is hermetic: no network, no harness install. The positive paths
inject a fake adapter into ``sys.modules`` with recording doubles; the
negative paths force the adapter to be unavailable and assert typed error
envelopes (never an uncaught ``TypeError``).
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from scholar_agent import server
from scholar_agent.server import mcp, recon_delta, recon_distill, recon_probe

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _typed_error(payload: dict) -> None:
    assert payload.get("status") == "error"
    assert payload.get("cache_key") is None
    message = str(payload.get("message") or "")
    assert "harness recon adapter unavailable" in message
    assert "scholar_harness.recon.adapter" in message


@pytest.fixture
def _clean_recon_state(monkeypatch):
    """Save/restore the recon globals so tests are independent."""
    saved = {
        "adapter": server._recon_adapter,
        "lexicon": server.DomainLexicon,
        "cache_root": server.RECON_CACHE_ROOT,
        "search_fn": server.RECON_SEARCH_FN,
        "loaded": server._harness_recon_loaded,
    }
    yield
    server._recon_adapter = saved["adapter"]
    server.DomainLexicon = saved["lexicon"]
    server.RECON_CACHE_ROOT = saved["cache_root"]
    server.RECON_SEARCH_FN = saved["search_fn"]
    server._harness_recon_loaded = saved["loaded"]


def _block_harness_imports(monkeypatch):
    """Force adapter-unavailable regardless of environment.

    Setting the sys.modules entries to None makes any
    ``from scholar_harness... import ...`` raise ImportError.
    """
    for name in (
        "scholar_harness",
        "scholar_harness.recon",
        "scholar_harness.recon.adapter",
        "scholar_harness.recon.lexicon",
    ):
        monkeypatch.setitem(sys.modules, name, None)


class _FakeCalls(dict):
    pass


def _install_fake_adapter(monkeypatch, tmp_root: Path, calls: dict):
    """Inject a fake adapter + lexicon exposing the exact adapter surface."""
    calls.clear()

    adapter_mod = types.ModuleType("scholar_harness.recon.adapter")
    lexicon_mod = types.ModuleType("scholar_harness.recon.lexicon")
    recon_pkg = types.ModuleType("scholar_harness.recon")
    harness_pkg = types.ModuleType("scholar_harness")
    recon_pkg.__path__ = []  # type: ignore[attr-defined]
    harness_pkg.__path__ = []  # type: ignore[attr-defined]

    def resolve_cache_root():
        calls["resolve_cache_root_n"] = calls.get("resolve_cache_root_n", 0) + 1
        return tmp_root

    def default_lexicon():
        calls["default_lexicon_n"] = calls.get("default_lexicon_n", 0) + 1
        return {"_default": True}

    def merge_lexicon(default, extra):
        calls["merge_lexicon"] = {"default": default, "extra": extra}
        return {"_merged": True}

    class FakeDomainLexicon:
        def __init__(self, metrics=None, datasets=None, schools=None):
            self.metrics = metrics or {}
            self.datasets = datasets or {}
            self.schools = schools or {}
            calls["lexicon_init"] = {
                "metrics": dict(self.metrics),
                "datasets": dict(self.datasets),
                "schools": dict(self.schools),
            }

        def __repr__(self):  # pragma: no cover - debugging aid
            return "FakeDomainLexicon(...)"

    def distill_terms(pool, lexicon=None, query_text=None):
        calls.setdefault("distill_terms", []).append(
            {"pool": pool, "lexicon": lexicon, "query_text": query_text}
        )
        return {
            "qei": 0.42,
            "metrics": {"m1": 2},
            "datasets": {"d1": 1},
            "schools": [{"label": "s1", "n": 2}],
            "topics": [{"label": "t1", "share": 0.9}],
            "micro_taxonomy": [{"term": "t1"}],
        }

    def pool_gate(n):
        calls["pool_gate"] = n
        return {"n_docs": n, "label": "ok"}

    def topic_gate(topics):
        calls["topic_gate"] = topics
        return {"label": "coherent", "top1": 0.9}

    class FakeEngine:
        def __init__(self, cache_root=None, search_fn=None):
            calls.setdefault("make_engine", []).append(
                {"cache_root": cache_root, "search_fn": search_fn}
            )
            self._cache_root = Path(cache_root) if cache_root is not None else tmp_root
            self._search_fn = search_fn

        async def probe(
            self, query, year_min=None, year_max=None, max_results=10, semantic=False
        ):
            calls["probe"] = {
                "query": query,
                "year_min": year_min,
                "year_max": year_max,
                "max_results": max_results,
                "semantic": semantic,
            }
            pool = {
                "cache_key": "ck_fake_probe",
                "docs": [{"id": "d1"}, {"id": "d2"}],
                "query": query,
            }
            self._cache_root.mkdir(parents=True, exist_ok=True)
            upstream = self._cache_root / "upstream_fake_pool.json"
            upstream.write_text(json.dumps(pool), encoding="utf-8")
            return (upstream, len(pool["docs"]))

    def make_engine(cache_root=None, search_fn=None):
        return FakeEngine(cache_root=cache_root, search_fn=search_fn)

    async def run_followups(
        pool, distilled, cache_root=None, search_fn=None, max_followups=3
    ):
        calls["run_followups"] = {
            "pool": pool,
            "distilled": distilled,
            "cache_root": cache_root,
            "search_fn": search_fn,
            "max_followups": max_followups,
        }
        merged_pool = {
            "cache_key": str(pool.get("cache_key", "")) + "+merged",
            "docs": list(pool.get("docs", [])) + [{"id": "d3"}],
        }
        merged_terms = {"topics": [{"label": "t1"}], "qei": 0.5}
        return {
            "pool": merged_pool,
            "distilled": merged_terms,
            "cache_keys_merged": ["ck_followup1"],
            "followups": [
                {
                    "term": "t-gap",
                    "reason": "2 direct hits, 1 adjacent",
                    "school_n": 1,
                    "corpus_total": 42,
                    "saturation_label": "sparse",
                }
            ],
            "dropped_n": 0,
        }

    adapter_mod.resolve_cache_root = resolve_cache_root  # type: ignore[attr-defined]
    adapter_mod.default_lexicon = default_lexicon  # type: ignore[attr-defined]
    adapter_mod.merge_lexicon = merge_lexicon  # type: ignore[attr-defined]
    adapter_mod.distill_terms = distill_terms  # type: ignore[attr-defined]
    adapter_mod.pool_gate = pool_gate  # type: ignore[attr-defined]
    adapter_mod.topic_gate = topic_gate  # type: ignore[attr-defined]
    adapter_mod.make_engine = make_engine  # type: ignore[attr-defined]
    adapter_mod.run_followups = run_followups  # type: ignore[attr-defined]
    lexicon_mod.DomainLexicon = FakeDomainLexicon  # type: ignore[attr-defined]

    recon_pkg.adapter = adapter_mod  # type: ignore[attr-defined]
    recon_pkg.lexicon = lexicon_mod  # type: ignore[attr-defined]
    harness_pkg.recon = recon_pkg  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "scholar_harness", harness_pkg)
    monkeypatch.setitem(sys.modules, "scholar_harness.recon", recon_pkg)
    monkeypatch.setitem(sys.modules, "scholar_harness.recon.adapter", adapter_mod)
    monkeypatch.setitem(sys.modules, "scholar_harness.recon.lexicon", lexicon_mod)
    return calls


def _reset_loader_state():
    server._recon_adapter = None
    server.DomainLexicon = None
    server.RECON_CACHE_ROOT = None
    server._harness_recon_loaded = False


# ---------------------------------------------------------------------------
# (a) H01-H03 ports: adapter unavailable -> typed envelopes, never TypeError
# ---------------------------------------------------------------------------


def test_h01_probe_without_adapter_returns_typed_error(monkeypatch, _clean_recon_state):
    _block_harness_imports(monkeypatch)
    _reset_loader_state()
    server.RECON_SEARCH_FN = None
    raw = recon_probe("some topic", start_year=2021, limit=5)
    payload = json.loads(raw)
    _typed_error(payload)


def test_h02_distill_without_adapter_returns_typed_error(
    monkeypatch, _clean_recon_state
):
    _block_harness_imports(monkeypatch)
    _reset_loader_state()
    server.RECON_SEARCH_FN = None
    raw = recon_distill("rec_abc123")
    payload = json.loads(raw)
    _typed_error(payload)


def test_h03_delta_without_adapter_returns_typed_error(monkeypatch, _clean_recon_state):
    _block_harness_imports(monkeypatch)
    _reset_loader_state()
    server.RECON_SEARCH_FN = None
    raw = recon_delta("rec_abc123", followups=2)
    payload = json.loads(raw)
    _typed_error(payload)


def test_h06_probe_without_adapter_never_raises_type_error(
    monkeypatch, _clean_recon_state
):
    """Even with RECON_CACHE_ROOT None, the guard precedes any path use."""
    _block_harness_imports(monkeypatch)
    _reset_loader_state()
    assert server.RECON_CACHE_ROOT is None
    try:
        raw = recon_probe("topic-never-raises")
    except TypeError:
        pytest.fail("recon_probe raised TypeError instead of typed envelope")
    _typed_error(json.loads(raw))


# ---------------------------------------------------------------------------
# (b) Absence proof: no discovery, no path mutation, no env-override read
# ---------------------------------------------------------------------------


def test_absence_no_discovery_source():
    src = Path(server.__file__).read_text(encoding="utf-8")
    assert "_harness_src" not in src
    assert "_HARNESS_SRC" not in src
    assert "NEXUS_HARNESS_SRC" not in src
    assert "sys.path" not in src
    assert "import sys" not in src
    assert "importlib.util" not in src
    assert "find_spec" not in src


def test_absence_import_does_not_mutate_sys_path():
    code = (
        "import sys, json; "
        "before = list(sys.path); "
        "mods_before = set(sys.modules); "
        "import scholar_agent.server; "
        "after = list(sys.path); "
        "print(json.dumps({'same_path': before == after, "
        "'before_len': len(before), 'after_len': len(after)}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["same_path"] is True


def test_absence_env_override_is_ignored():
    """Setting the legacy override must not change import-time path state."""
    code = (
        "import os, sys, json; "
        "os.environ['NEXUS_HARNESS_SRC'] = '/nonexistent-override'; "
        "before = list(sys.path); "
        "import scholar_agent.server; "
        "after = list(sys.path); "
        "print(json.dumps({'same': before == after, "
        "'leaked': '/nonexistent-override' in after}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["same"] is True
    assert payload["leaked"] is False


# ---------------------------------------------------------------------------
# (c) Adapter-wiring proof: fake adapter drives all three tools
# ---------------------------------------------------------------------------


def test_wiring_probe_uses_adapter_make_engine(
    tmp_path, monkeypatch, _clean_recon_state
):
    calls: dict = {}
    _install_fake_adapter(monkeypatch, tmp_path, calls)
    _reset_loader_state()

    def _fake_search(*args, **kwargs):
        raise AssertionError("fake search must not be called by the FakeEngine probe")

    monkeypatch.setattr(server, "RECON_SEARCH_FN", _fake_search)

    raw = recon_probe("wiring topic", start_year=2021, limit=5, semantic=True)
    payload = json.loads(raw)

    assert payload["status"] == "probe_ok"
    assert payload["session_id"].startswith("rec_")
    assert payload["cache_key"] == "ck_fake_probe"
    assert payload["n_docs"] == 2
    assert payload["pool_path"].endswith(".json")
    assert Path(payload["pool_path"]).is_file()

    assert calls.get("resolve_cache_root_n", 0) >= 1
    assert server.RECON_CACHE_ROOT == tmp_path
    assert len(calls.get("make_engine", [])) == 1
    assert calls["make_engine"][0]["cache_root"] == tmp_path
    assert calls["make_engine"][0]["search_fn"] is _fake_search
    assert calls["probe"]["query"] == "wiring topic"
    assert calls["probe"]["year_min"] == 2021
    assert calls["probe"]["max_results"] == 5
    assert calls["probe"]["semantic"] is True


def test_wiring_distill_uses_adapter_terms_and_gates(
    tmp_path, monkeypatch, _clean_recon_state
):
    calls: dict = {}
    _install_fake_adapter(monkeypatch, tmp_path, calls)
    _reset_loader_state()
    monkeypatch.setattr(server, "RECON_SEARCH_FN", None)

    probe_payload = json.loads(recon_probe("distill wiring"))
    sid = probe_payload["session_id"]

    lex = {"metrics": {"m.*": "M"}, "datasets": {}, "schools": {}}
    raw = recon_distill(sid, lexicon_json=dict(lex))
    payload = json.loads(raw)

    assert payload["session_id"] == sid
    assert payload["cache_key"] == "ck_fake_probe"
    assert Path(payload["terms_path"]).is_file()
    assert payload["qei"] == 0.42
    assert payload["pool"] == {"n_docs": 2, "label": "ok"}
    assert payload["purity"] == {"label": "coherent", "top1": 0.9}
    assert payload["metrics"] == [{"label": "m1", "count": 2}]
    assert payload["datasets"] == [{"label": "d1", "count": 1}]
    assert payload["schools"] == [{"label": "s1", "n": 2}]
    assert payload["topics"] == [{"label": "t1", "share": 0.9}]
    assert payload["micro_taxonomy_top"] == [{"term": "t1"}]

    assert calls["lexicon_init"]["metrics"] == {"m.*": "M"}
    assert calls["merge_lexicon"]["default"] == {"_default": True}
    assert calls["distill_terms"][-1]["query_text"] == "distill wiring"
    assert calls["distill_terms"][-1]["lexicon"] == {"_merged": True}
    assert calls["pool_gate"] == 2
    assert calls["topic_gate"] == [{"label": "t1", "share": 0.9}]


def test_wiring_distill_default_lexicon_shape(
    tmp_path, monkeypatch, _clean_recon_state
):
    calls: dict = {}
    _install_fake_adapter(monkeypatch, tmp_path, calls)
    _reset_loader_state()
    monkeypatch.setattr(server, "RECON_SEARCH_FN", None)

    sid = json.loads(recon_probe("default lexicon"))["session_id"]
    payload = json.loads(recon_distill(sid))

    assert payload["cache_key"] == "ck_fake_probe"
    assert "merge_lexicon" not in calls
    assert calls["distill_terms"][-1]["lexicon"] is None


def test_wiring_delta_uses_adapter_run_followups(
    tmp_path, monkeypatch, _clean_recon_state
):
    calls: dict = {}
    _install_fake_adapter(monkeypatch, tmp_path, calls)
    _reset_loader_state()

    sentinel_search = object()
    monkeypatch.setattr(server, "RECON_SEARCH_FN", sentinel_search)

    sid = json.loads(recon_probe("delta wiring"))["session_id"]
    calls.clear()
    # Re-install to keep adapter importable after clear (loader is cached).
    # The cached adapter stays bound; only the call records were cleared.
    raw = recon_delta(sid, followups=2)
    payload = json.loads(raw)

    assert payload["session_id"] == sid
    assert payload["merged_cache_keys"] == ["ck_fake_probe", "ck_followup1"]
    assert payload["pool_size_before"] == 2
    assert payload["pool_size_after"] == 3
    assert payload["dropped_n"] == 0
    assert payload["topics"] == [{"label": "t1"}]
    assert len(payload["followups"]) == 1
    item = payload["followups"][0]
    assert item["term"] == "t-gap"
    assert item["reason"] == "2 direct hits, 1 adjacent"
    assert item["confidence"] == {"label": "gap", "detail": "2 direct hits, 1 adjacent"}
    assert item["school_n"] == 1
    assert item["corpus_total"] == 42
    assert item["saturation_label"] == "sparse"

    assert len(calls.get("make_engine", [])) == 1
    assert calls["make_engine"][0]["cache_root"] == tmp_path
    assert calls["make_engine"][0]["search_fn"] is sentinel_search
    assert calls["run_followups"]["cache_root"] == tmp_path
    assert calls["run_followups"]["search_fn"] is sentinel_search
    assert calls["run_followups"]["max_followups"] == 2
    assert calls["run_followups"]["pool"]["cache_key"] == "ck_fake_probe"


# ---------------------------------------------------------------------------
# (d) Loader-called-by-tools proof
# ---------------------------------------------------------------------------


def test_loader_is_called_by_tools_without_preloading(
    tmp_path, monkeypatch, _clean_recon_state
):
    calls: dict = {}
    _install_fake_adapter(monkeypatch, tmp_path, calls)
    _reset_loader_state()
    monkeypatch.setattr(server, "RECON_SEARCH_FN", None)

    assert server._recon_adapter is None
    assert server._harness_recon_loaded is False

    payload = json.loads(recon_probe("no-preload"))
    assert payload["status"] == "probe_ok"
    assert server._recon_adapter is not None
    assert server.DomainLexicon is not None
    assert server._harness_recon_loaded is True


def test_ensure_loader_returns_bool(tmp_path, monkeypatch, _clean_recon_state):
    calls: dict = {}
    _install_fake_adapter(monkeypatch, tmp_path, calls)
    _reset_loader_state()
    assert server._ensure_harness_recon_loaded() is True

    _block_harness_imports(monkeypatch)
    _reset_loader_state()
    assert server._ensure_harness_recon_loaded() is False


# ---------------------------------------------------------------------------
# (e) Registration intact
# ---------------------------------------------------------------------------


def test_recon_tools_still_registered():
    names = {tool.name for tool in mcp._tool_manager.list_tools()}
    assert "recon_probe" in names
    assert "recon_distill" in names
    assert "recon_delta" in names


def test_recon_tools_reachable_via_mcp_dispatch(
    tmp_path, monkeypatch, _clean_recon_state
):
    calls: dict = {}
    _install_fake_adapter(monkeypatch, tmp_path, calls)
    _reset_loader_state()
    monkeypatch.setattr(server, "RECON_SEARCH_FN", None)

    result = asyncio.run(mcp.call_tool("recon_probe", {"topic": "dispatch check"}))
    flat: str
    if isinstance(result, tuple):
        flat = json.dumps(result, default=str)
    elif hasattr(result, "content"):
        flat = json.dumps(result.content, default=str)
    else:
        flat = json.dumps(result, default=str)
    assert "probe_ok" in flat or "ck_fake_probe" in flat
