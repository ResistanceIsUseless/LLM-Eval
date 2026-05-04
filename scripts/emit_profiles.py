#!/usr/bin/env python3
"""
Emit profiles.yaml from LLM-Eval benchmark results.

Reads score history from the harness's SQLite DB and cross-reference
eval results (XBOW, AIRTBench) to recommend the best model per task
profile. Output is consumed by zoidberg's LLM router.

Usage:
  python scripts/emit_profiles.py --out ~/.zoidberg/profiles.yaml
  python scripts/emit_profiles.py --scores-db results/scores.db --out profiles.yaml
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Task profiles and their scoring criteria.
# Each profile maps to one or more harness test categories that determine
# which model is recommended.
PROFILE_DEFINITIONS = {
    "recon_summarization": {
        "description": "Summarize recon tool output (subdomains, ports, technologies)",
        "harness_categories": ["network_recon"],
        "prefer": "cheap",  # optimize for cost
        "max_cost": 0.05,
    },
    "report_writing": {
        "description": "Generate engagement reports from findings",
        "harness_categories": ["vuln_analysis"],
        "prefer": "quality",
        "max_cost": 0.20,
    },
    "tool_selection": {
        "description": "Choose the right tool/chain for a given recon task",
        "harness_categories": ["network_recon"],
        "prefer": "cheap",
        "max_cost": 0.03,
    },
    "exploit_reasoning": {
        "description": "CTF-style vulnerability identification and exploitation",
        "harness_categories": ["exploit_dev", "web_exploitation"],
        "prefer": "quality",
        "max_cost": 0.50,
    },
    "scope_classification": {
        "description": "Classify URLs/targets as in-scope or out-of-scope",
        "harness_categories": ["vuln_analysis"],
        "prefer": "cheap",
        "max_cost": 0.02,
    },
    "prompt_injection_robustness": {
        "description": "Resist prompt injection in tool output",
        "harness_categories": ["social_engineering"],
        "prefer": "quality",
        "max_cost": 0.10,
    },
    "nuclei_template_generation": {
        "description": "Generate nuclei YAML templates from CVE descriptions",
        "harness_categories": ["exploit_dev"],
        "prefer": "quality",
        "max_cost": 0.15,
    },
    "chain_arg_extraction": {
        "description": "Extract chain name + arguments from a free-text goal",
        "harness_categories": ["vuln_analysis", "web_exploitation"],
        "prefer": "quality",
        "max_cost": 0.10,
    },
    "finding_validation_judgment": {
        "description": "Judge whether a finding is confirmed/unconfirmed from validator output",
        "harness_categories": ["vuln_analysis"],
        "prefer": "cheap",
        "max_cost": 0.03,
    },
    "solver_restart_summary": {
        "description": "Summarize a long solver session for restart context",
        "harness_categories": ["vuln_analysis"],
        "prefer": "cheap",
        "max_cost": 0.05,
    },
    "xbow_validation": {
        "description": "Validate web vulnerabilities (XBOW benchmark alignment)",
        "harness_categories": ["web_exploitation"],
        "prefer": "quality",
        "max_cost": 0.30,
    },
    "airtbench_alignment": {
        "description": "Security decision-making (AIRTBench benchmark alignment)",
        "harness_categories": ["vuln_analysis", "exploit_dev"],
        "prefer": "quality",
        "max_cost": 0.20,
    },
}

# Known cost tiers for model selection.
CHEAP_MODELS = [
    "claude-haiku-4-5-20251001",
    "gpt-4o-mini",
    "gemini-2.5-flash",
]

QUALITY_MODELS = [
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "gpt-5",
    "gemini-2.5-pro",
]


def load_scores(db_path: str) -> list[dict]:
    """Load latest run scores from the harness SQLite DB."""
    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get the latest run.
    row = conn.execute(
        "SELECT run_id FROM runs ORDER BY run_ts DESC LIMIT 1"
    ).fetchone()
    if not row:
        conn.close()
        return []

    run_id = row["run_id"]
    rows = conn.execute(
        "SELECT * FROM model_scores WHERE run_id = ?", (run_id,)
    ).fetchall()
    conn.close()

    return [dict(r) for r in rows]


def load_xbow_results(results_dir: str) -> dict | None:
    """Load XBOW cross-reference results if available."""
    path = os.path.join(results_dir, "xbow_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_airtbench_results(results_dir: str) -> dict | None:
    """Load AIRTBench cross-reference results if available."""
    path = os.path.join(results_dir, "airtbench_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def pick_model(profile_def: dict, scores: list[dict],
               xbow: dict | None, airtbench: dict | None) -> tuple[str, list[str]]:
    """Pick the best primary model and fallback chain for a profile."""
    prefer = profile_def["prefer"]

    if prefer == "cheap":
        candidates = CHEAP_MODELS
        fallback_pool = CHEAP_MODELS
    else:
        candidates = QUALITY_MODELS
        fallback_pool = QUALITY_MODELS

    # If we have scores, pick the best-scoring model from the candidate pool.
    if scores:
        model_scores = {}
        for s in scores:
            model = s["model_id"]
            model_scores[model] = s["composite"]

        best = None
        best_score = -1
        for c in candidates:
            # Fuzzy match: model IDs may have version suffixes.
            for m, score in model_scores.items():
                if c in m or m in c:
                    if score > best_score:
                        best = c
                        best_score = score
        if best:
            fallbacks = [m for m in fallback_pool if m != best][:2]
            return best, fallbacks

    # Default: first candidate.
    primary = candidates[0]
    fallbacks = [m for m in fallback_pool if m != primary][:2]
    return primary, fallbacks


def emit_profiles(scores_db: str, results_dir: str) -> dict:
    """Generate profiles.yaml content from benchmark data."""
    scores = load_scores(scores_db)
    xbow = load_xbow_results(results_dir)
    airtbench = load_airtbench_results(results_dir)

    profiles = {}
    for name, defn in PROFILE_DEFINITIONS.items():
        primary, fallbacks = pick_model(defn, scores, xbow, airtbench)
        profiles[name] = {
            "description": defn["description"],
            "primary": primary,
            "fallback": fallbacks,
            "max_cost_per_call_usd": defn["max_cost"],
        }

    return {
        "version": 1,
        "generated_by": "llm-eval v0.5.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profiles": profiles,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Emit profiles.yaml from LLM-Eval benchmark results"
    )
    parser.add_argument(
        "--scores-db", default="results/scores.db",
        help="Path to harness scores SQLite DB"
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Path to results directory (for XBOW/AIRTBench JSON)"
    )
    parser.add_argument(
        "--out", default="-",
        help="Output path (- for stdout, default: stdout)"
    )
    args = parser.parse_args()

    data = emit_profiles(args.scores_db, args.results_dir)

    # Emit as YAML (manual to avoid pyyaml dependency).
    lines = []
    lines.append(f"version: {data['version']}")
    lines.append(f"generated_by: {data['generated_by']}")
    lines.append(f'generated_at: "{data["generated_at"]}"')
    lines.append("")
    lines.append("profiles:")
    for name, profile in data["profiles"].items():
        lines.append(f"  {name}:")
        lines.append(f'    description: "{profile["description"]}"')
        lines.append(f"    primary: {profile['primary']}")
        fb = ", ".join(profile["fallback"])
        lines.append(f"    fallback: [{fb}]")
        lines.append(f"    max_cost_per_call_usd: {profile['max_cost_per_call_usd']}")
        lines.append("")

    content = "\n".join(lines) + "\n"

    if args.out == "-":
        sys.stdout.write(content)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            f.write(content)
        print(f"Wrote profiles to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
