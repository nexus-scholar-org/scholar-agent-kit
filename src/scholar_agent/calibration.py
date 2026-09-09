"""Pre-flight screener calibration and structured boolean checklist for PRISMA screening.

Addresses the P1 screener-bias problem (Kappa ~0.115 from retrospective) by:

1. **Boolean Checklist**: Replace free-form INCLUDE/EXCLUDE decisions with a
   structured schema — one boolean per inclusion/exclusion criterion — so the
   agent cannot drift on subjective framing.

2. **20-Paper Pre-flight Calibration**: Sample 20 gold-labeled papers, hand them
   to the agent *without* gold answers, collect the agent's filled checklists,
   and compare against gold to surface inclusion-rate drift and per-criterion
   misclassification before real screening begins.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from scholar_search.screening import ScreeningDecision

# ---------------------------------------------------------------------------
# Checklist schema
# ---------------------------------------------------------------------------

def build_checklist_schema(protocol_data: dict[str, Any]) -> list[dict[str, str]]:
    """
    Build the structured boolean checklist schema from protocol screening_criteria.

    Returns a list of checklist-item descriptors suitable for LLM prompts, each
    with keys: ``criterion_id``, ``criterion_type``, ``description``,
    ``field_name`` (the JSON key the agent must populate).
    """
    criteria = protocol_data.get("screening_criteria", {})
    items: list[dict[str, str]] = []

    for inc in criteria.get("inclusion", []):
        cid = inc.get("id", "INC-01")
        items.append({
            "criterion_id": cid,
            "criterion_type": "inclusion",
            "description": inc.get("criterion", ""),
            "field_name": cid.lower().replace("-", "_"),
        })

    for exc in criteria.get("exclusion", []):
        cid = exc.get("id", "EXC-01")
        items.append({
            "criterion_id": cid,
            "criterion_type": "exclusion",
            "description": exc.get("criterion", ""),
            "field_name": cid.lower().replace("-", "_"),
        })

    return items


# ---------------------------------------------------------------------------
# Checklist → deterministic ScreeningDecision
# ---------------------------------------------------------------------------

def checklist_to_decision(
    *,
    workspace_id: str,
    checklist: dict[str, bool],
    schema: list[dict[str, str]],
    document_title: str = "",
    doi: str | None = None,
    relevant_rqs: list[str] | None = None,
    screening_reasoning: str = "",
) -> ScreeningDecision:
    """
    Deterministically derive a ScreeningDecision from a filled boolean checklist.

    Rules:
    - INCLUDE iff every inclusion field is True **and** every exclusion field is False.
    - EXCLUDE with conf 0.55 (human audit flag) if any exclusion field is True.
    - EXCLUDE with conf 0.75 if any inclusion field is False.

    Parameters
    ----------
    workspace_id:
        Paper identifier (e.g. ``SCI-000001`` or ``CAL-0001``).
    checklist:
        Mapping of field_name → bool as filled by the agent.
    schema:
        Output of :func:`build_checklist_schema`.
    """
    matched_inc: list[str] = []
    triggered_exc: list[str] = []

    for item in schema:
        fid = item["field_name"]
        val = bool(checklist.get(fid, False))
        if item["criterion_type"] == "inclusion" and val:
            matched_inc.append(item["criterion_id"])
        if item["criterion_type"] == "exclusion" and val:
            triggered_exc.append(item["criterion_id"])

    all_inclusions_met = len(matched_inc) == sum(
        1 for s in schema if s["criterion_type"] == "inclusion"
    )
    no_exclusions_triggered = len(triggered_exc) == 0

    if all_inclusions_met and no_exclusions_triggered:
        decision = "INCLUDE"
        confidence = 0.90
    elif triggered_exc:
        decision = "EXCLUDE"
        confidence = 0.55
    else:
        decision = "EXCLUDE"
        confidence = 0.75

    return ScreeningDecision(
        workspace_id=workspace_id,
        decision=decision,  # type: ignore[arg-type]
        confidence=confidence,
        matched_inclusion_criteria=matched_inc,
        violated_exclusion_criteria=triggered_exc,
        relevant_rqs=relevant_rqs or [],
        screening_reasoning=screening_reasoning,
        document_title=document_title,
        doi=doi,
    )


# ---------------------------------------------------------------------------
# Pre-flight calibration builder
# ---------------------------------------------------------------------------

@dataclass
class CalibrationBatch:
    """Container for a pre-flight calibration batch and its gold standard."""

    batch_data: dict[str, Any]
    gold_data: dict[str, Any]

    def write(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "calibration_batch_000.json").write_text(
            json.dumps(self.batch_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "calibration_gold.json").write_text(
            json.dumps(self.gold_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def build_preflight_calibration(
    papers_with_gold: list[dict[str, Any]],
    protocol_data: dict[str, Any],
    *,
    sample_size: int = 20,
    seed: int = 42,
) -> CalibrationBatch:
    """
    Prepare a 20-paper pre-flight calibration batch + gold standard.

    Parameters
    ----------
    papers_with_gold:
        List of paper dicts. Each **must** include:
        - ``title``, ``abstract``
        - ``gold_decision``: ``"INCLUDE"`` | ``"EXCLUDE"``
        Optional: ``workspace_id``, ``year``, ``venue``, ``doi``,
        ``gold_matched_inclusion``, ``gold_violated_exclusion``.
    protocol_data:
        Full protocol dict with ``screening_criteria``.
    sample_size:
        Number of papers to sample for calibration (default 20).
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    CalibrationBatch
        A dataclass whose ``batch_data`` dict is the agent-facing batch and
        ``gold_data`` is the secret gold standard.
    """
    rng = random.Random(seed)
    n = min(sample_size, len(papers_with_gold))
    selected = (
        rng.sample(papers_with_gold, n)
        if n < len(papers_with_gold)
        else list(papers_with_gold)
    )

    criteria = protocol_data.get("screening_criteria", {})
    checklist_schema = build_checklist_schema(protocol_data)

    # Agent-facing instructions
    lines = [
        "# PRISMA 2020 Pre-Flight Screening Calibration",
        "",
        "You are performing a pre-flight calibration screening.",
        "For EACH paper below, fill in the structured boolean checklist.",
        "This is a calibration step to measure bias before real screening.",
        "",
        "## Checklist Schema (fill ONE boolean per criterion per paper)",
        "```json",
        json.dumps(checklist_schema, indent=2),
        "```",
        "",
        "## Papers to Screen",
    ]

    papers_for_batch: list[dict[str, Any]] = []
    for p in selected:
        papers_for_batch.append({
            "workspace_id": p.get(
                "workspace_id", f"CAL-{len(papers_for_batch) + 1:04d}"
            ),
            "title": p.get("title", "Untitled"),
            "year": p.get("year"),
            "abstract": p.get("abstract", "No abstract available."),
            "venue": p.get("venue"),
            "doi": p.get("doi"),
        })

    lines += [
        "```json",
        json.dumps(papers_for_batch, indent=2),
        "```",
        "",
        "## Required Output",
        "Write a JSON array (one object per paper, same order). Each object MUST have:",
        "```json",
        json.dumps(
            [
                {
                    "workspace_id": "CAL-XXXX",
                    **{item["field_name"]: False for item in checklist_schema},
                    "screening_reasoning": "Brief justification.",
                }
            ],
            indent=2,
        ),
"```",
        "Write your response to: `literature/screening/calibration_batch_000_decisions.json`",
    ]

    batch_data = {
        "batch_index": 0,
        "is_calibration": True,
        "total_batches": 1,
        "batch_size": len(papers_for_batch),
        "status": "PENDING",
        "protocol": {
            "title": (protocol_data.get("metadata") or {}).get(
                "title", "Calibration"
            ),
            "research_questions": protocol_data.get("research_questions", []),
            "screening_criteria": criteria,
        },
        "papers": papers_for_batch,
        "checklist_schema": checklist_schema,
        "agent_instructions": "\n".join(lines),
    }

    gold_data = {
        "batch_index": 0,
        "is_calibration_gold": True,
        "papers": [
            {
                "workspace_id": p.get(
                    "workspace_id", f"CAL-{i + 1:04d}"
                ),
                "gold_decision": p.get("gold_decision", "EXCLUDE"),
                "gold_matched_inclusion": p.get("gold_matched_inclusion", []),
                "gold_violated_exclusion": p.get(
                    "gold_violated_exclusion", []
                ),
            }
            for i, p in enumerate(selected)
        ],
    }

    return CalibrationBatch(batch_data=batch_data, gold_data=gold_data)


# ---------------------------------------------------------------------------
# Calibration evaluator
# ---------------------------------------------------------------------------

@dataclass
class CalibrationReport:
    """Results of evaluating agent calibration against gold-standard labels."""

    total: int
    inclusion_rate: float
    gold_inclusion_rate: float
    inclusion_rate_delta: float
    sensitivity: float | None
    specificity: float | None
    accuracy: float
    verdict: Literal["PASS", "FLAG"]
    reasons: list[str] = field(default_factory=list)
    per_criterion: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def _fmt(self, value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1%}"

    def to_markdown(self) -> str:
        lines = [
            "# Screener Calibration Report",
            "",
            f"- **Total papers**: {self.total}",
            f"- **Agent inclusion rate**: {self.inclusion_rate:.1%}",
            f"- **Gold inclusion rate**: {self.gold_inclusion_rate:.1%}",
            f"- **Inclusion-rate delta**: {self.inclusion_rate_delta:+.1%}",
            f"- **Sensitivity** (gold-INC caught): {self._fmt(self.sensitivity)}",
            f"- **Specificity** (gold-EXC caught): {self._fmt(self.specificity)}",
            f"- **Accuracy**: {self.accuracy:.1%}",
            f"- **Verdict**: `{self.verdict}`",
        ]
        if self.reasons:
            lines += [""] + [f"  - {r}" for r in self.reasons]
        if self.per_criterion:
            lines += [
                "",
                "### Per-Criterion Breakdown",
                "",
                "| Criterion | Sensitivity | Specificity |",
                "| :--- | :--- | :--- |",
            ]
            for cid, stats in self.per_criterion.items():
                lines.append(
                    f"| {cid} | {stats.get('sensitivity', 0):.1%} | "
                    f"{stats.get('specificity', 0):.1%} |"
                )
        return "\n".join(lines)


def evaluate_calibration(
    decisions: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    *,
    inclusion_rate_tolerance: float = 0.15,
    sensitivity_threshold: float = 0.80,
    specificity_threshold: float = 0.70,
) -> CalibrationReport:
    """
    Evaluate agent calibration against gold-standard labels.

    Parameters
    ----------
    decisions:
        Agent decision files — list of dicts each with ``workspace_id`` and
        the boolean checklist keys (e.g. ``inc_01``, ``exc_03``).
    gold:
        Gold standard — list of dicts each with ``workspace_id`` and
        ``gold_decision`` (``"INCLUDE"`` / ``"EXCLUDE"``).
    inclusion_rate_tolerance:
        Maximum absolute delta between the agent's inclusion rate and the gold
        rate before a FLAG verdict.
    sensitivity_threshold:
        Minimum sensitivity (proportion of gold-INC papers correctly identified).
    specificity_threshold:
        Minimum specificity (proportion of gold-EXC papers correctly identified).

    Returns
    -------
    CalibrationReport
    """
    gold_map: dict[str, dict[str, Any]] = {
        g["workspace_id"]: g for g in gold
    }
    total = len(gold)
    if total == 0:
        return CalibrationReport(
            total=0,
            inclusion_rate=0,
            gold_inclusion_rate=0,
            inclusion_rate_delta=0,
            sensitivity=None,
            specificity=None,
            accuracy=0,
            verdict="FLAG",
            reasons=["Empty gold set"],
        )

    tp = fp = tn = fn = 0

    for g in gold:
        wid = g["workspace_id"]
        gold_entry = gold_map[wid]
        agent_entry = next(
            (d for d in decisions if d.get("workspace_id") == wid), None
        )

        if agent_entry is None:
            agent_says_include = False
        else:
            inc_vals = [
                v for k, v in agent_entry.items() if k.startswith("inc_")
            ]
            exc_vals = [
                v for k, v in agent_entry.items() if k.startswith("exc_")
            ]
            agent_says_include = (
                (all(inc_vals) if inc_vals else True)
                and (not any(exc_vals) if exc_vals else True)
            )

        gold_inc = gold_entry.get("gold_decision") == "INCLUDE"

        if gold_inc and agent_says_include:
            tp += 1
        elif gold_inc and not agent_says_include:
            fn += 1
        elif not gold_inc and agent_says_include:
            fp += 1
        else:
            tn += 1

    sensitivity = (
        tp / (tp + fn) if (tp + fn) > 0 else None
    )  # None = no gold-INC papers
    specificity = (
        tn / (tn + fp) if (tn + fp) > 0 else None
    )  # None = no gold-EXC papers
    accuracy = (tp + tn) / total if total > 0 else 0.0
    agent_inc_rate = (tp + fp) / total
    gold_inc_rate = (tp + fn) / total
    delta = agent_inc_rate - gold_inc_rate

    reasons: list[str] = []
    verdict: Literal["PASS", "FLAG"] = "PASS"

    if abs(delta) > inclusion_rate_tolerance:
        reasons.append(
            f"Inclusion-rate drift {delta:+.1%} exceeds tolerance "
            f"±{inclusion_rate_tolerance:.0%}"
        )
        verdict = "FLAG"
    if sensitivity is not None and sensitivity < sensitivity_threshold:
        reasons.append(
            f"Sensitivity {sensitivity:.1%} below threshold "
            f"{sensitivity_threshold:.0%}"
        )
        verdict = "FLAG"
    if specificity is not None and specificity < specificity_threshold:
        reasons.append(
            f"Specificity {specificity:.1%} below threshold "
            f"{specificity_threshold:.0%}"
        )
        verdict = "FLAG"

    return CalibrationReport(
        total=total,
        inclusion_rate=agent_inc_rate,
        gold_inclusion_rate=gold_inc_rate,
        inclusion_rate_delta=delta,
        sensitivity=sensitivity,
        specificity=specificity,
        accuracy=accuracy,
        verdict=verdict,
        reasons=reasons,
    )
