#!/usr/bin/env python3
"""
Chain Argument Extraction Eval — Phase 2 LLM-Eval benchmark.

Tests whether a model can correctly extract a chain name and arguments
from a free-text engagement goal, given the set of available chains.

Example:
  Goal: "Check example.com for subdomain takeover"
  Available chains: recon-basic, subdomain-takeover, xss-reflected, ...
  Expected: chain=subdomain-takeover, args={target: example.com}

Scoring: exact chain match (0.5) + argument accuracy (0.5).

Usage:
  python -m evals.chain_arg_extraction --models claude-sonnet-4-6
  python harness.py eval-chain-args --models claude-sonnet-4-6
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class ChainArgChallenge:
    """A single chain-argument extraction challenge."""
    id: str
    goal: str                             # free-text engagement goal
    available_chains: list                # list of chain names
    expected_chain: str                   # correct chain name
    expected_args: dict                   # correct arguments
    difficulty: str = "intermediate"
    notes: str = ""


@dataclass
class ChainArgResult:
    """Result of one extraction attempt."""
    challenge_id: str
    model_id: str
    predicted_chain: str
    predicted_args: dict
    expected_chain: str
    expected_args: dict
    chain_correct: bool
    arg_score: float                      # 0.0–1.0
    composite_score: float                # 0.5 * chain + 0.5 * args
    latency_ms: float
    error: Optional[str] = None


AVAILABLE_CHAINS = [
    {"name": "recon-basic", "description": "Passive reconnaissance: subdomain enumeration, DNS, tech detection", "args": ["target"]},
    {"name": "recon-deep", "description": "Active reconnaissance: port scanning, service fingerprinting", "args": ["target", "ports"]},
    {"name": "subdomain-takeover", "description": "Check for subdomain takeover via dangling DNS", "args": ["target"]},
    {"name": "xss-reflected", "description": "Test for reflected XSS on URL parameters", "args": ["url", "params"]},
    {"name": "xss-stored", "description": "Test for stored XSS via form inputs", "args": ["url", "form_fields"]},
    {"name": "sqli-error", "description": "Test for error-based SQL injection", "args": ["url", "params"]},
    {"name": "sqli-blind", "description": "Test for blind SQL injection (boolean and time-based)", "args": ["url", "params"]},
    {"name": "ssrf-basic", "description": "Test for SSRF via URL parameters", "args": ["url", "param"]},
    {"name": "nuclei-cve", "description": "Run nuclei with CVE templates against a target", "args": ["target", "severity"]},
    {"name": "oauth-probe", "description": "Test OAuth implementation for misconfigurations", "args": ["auth_url", "client_id"]},
    {"name": "jwt-test", "description": "Test JWT implementation for common weaknesses", "args": ["url", "token"]},
    {"name": "cors-check", "description": "Test CORS configuration for overly permissive policies", "args": ["url"]},
    {"name": "header-audit", "description": "Audit security headers on a target", "args": ["url"]},
    {"name": "api-fuzz", "description": "Fuzz API endpoints for unexpected behavior", "args": ["base_url", "endpoints"]},
    {"name": "validate-xss", "description": "Validate a reported XSS finding", "args": ["url", "payload", "context"]},
    {"name": "validate-sqli", "description": "Validate a reported SQL injection finding", "args": ["url", "payload"]},
]

SYSTEM_PROMPT = """You are a security automation assistant. Given an engagement goal and a list
of available tool chains, select the most appropriate chain and extract its arguments.

Available chains:
{chains}

Respond with EXACTLY this JSON format:
{{
  "chain": "chain-name",
  "args": {{"arg1": "value1", "arg2": "value2"}},
  "reasoning": "Brief explanation of chain selection"
}}

Rules:
- Pick exactly ONE chain from the available list
- Extract argument values from the goal text where possible
- Use sensible defaults when specific values aren't mentioned
- For 'target' args, extract the domain/URL from the goal
- For 'severity' args, default to 'high,critical' unless specified"""


def _format_chains() -> str:
    lines = []
    for c in AVAILABLE_CHAINS:
        args_str = ", ".join(c["args"])
        lines.append(f"  - {c['name']}: {c['description']} (args: {args_str})")
    return "\n".join(lines)


BUILTIN_CHALLENGES = [
    ChainArgChallenge(
        id="chain-arg-001",
        goal="Check example.com for subdomain takeover vulnerabilities",
        available_chains=[c["name"] for c in AVAILABLE_CHAINS],
        expected_chain="subdomain-takeover",
        expected_args={"target": "example.com"},
        difficulty="beginner",
    ),
    ChainArgChallenge(
        id="chain-arg-002",
        goal="Run a reflected XSS test on https://app.example.com/search against the q and category parameters",
        available_chains=[c["name"] for c in AVAILABLE_CHAINS],
        expected_chain="xss-reflected",
        expected_args={"url": "https://app.example.com/search", "params": ["q", "category"]},
        difficulty="intermediate",
    ),
    ChainArgChallenge(
        id="chain-arg-003",
        goal="I need to check if shop.example.com has any critical CVEs",
        available_chains=[c["name"] for c in AVAILABLE_CHAINS],
        expected_chain="nuclei-cve",
        expected_args={"target": "shop.example.com", "severity": "critical"},
        difficulty="beginner",
    ),
    ChainArgChallenge(
        id="chain-arg-004",
        goal="Verify that the SQL injection in https://api.example.com/users?id=1 is real using the payload 1' OR '1'='1",
        available_chains=[c["name"] for c in AVAILABLE_CHAINS],
        expected_chain="validate-sqli",
        expected_args={"url": "https://api.example.com/users?id=1", "payload": "1' OR '1'='1"},
        difficulty="intermediate",
    ),
    ChainArgChallenge(
        id="chain-arg-005",
        goal="Do passive recon on megacorp.io — I want subdomains, DNS records, and tech stack",
        available_chains=[c["name"] for c in AVAILABLE_CHAINS],
        expected_chain="recon-basic",
        expected_args={"target": "megacorp.io"},
        difficulty="beginner",
    ),
    ChainArgChallenge(
        id="chain-arg-006",
        goal="The login at https://auth.example.com/oauth/authorize might have OAuth issues. The client_id is abc123.",
        available_chains=[c["name"] for c in AVAILABLE_CHAINS],
        expected_chain="oauth-probe",
        expected_args={"auth_url": "https://auth.example.com/oauth/authorize", "client_id": "abc123"},
        difficulty="intermediate",
    ),
    ChainArgChallenge(
        id="chain-arg-007",
        goal="We found a stored XSS report — need to confirm it. The URL is https://forum.example.com/post/42, the payload was <img src=x onerror=alert(1)>, and it's in the comment body context.",
        available_chains=[c["name"] for c in AVAILABLE_CHAINS],
        expected_chain="validate-xss",
        expected_args={"url": "https://forum.example.com/post/42", "payload": "<img src=x onerror=alert(1)>", "context": "comment body"},
        difficulty="advanced",
    ),
    ChainArgChallenge(
        id="chain-arg-008",
        goal="Fuzz the /api/v2/users and /api/v2/orders endpoints on https://api.shop.io",
        available_chains=[c["name"] for c in AVAILABLE_CHAINS],
        expected_chain="api-fuzz",
        expected_args={"base_url": "https://api.shop.io", "endpoints": ["/api/v2/users", "/api/v2/orders"]},
        difficulty="intermediate",
    ),
]


def load_challenges(challenges_dir: str) -> list[ChainArgChallenge]:
    """Load challenges from directory, falling back to built-ins."""
    cdir = Path(challenges_dir)
    if cdir.exists():
        challenges = []
        for f in sorted(cdir.glob("*.json")):
            with open(f) as fh:
                data = json.load(fh)
            if isinstance(data, list):
                for item in data:
                    challenges.append(ChainArgChallenge(**item))
            else:
                challenges.append(ChainArgChallenge(**data))
        if challenges:
            return challenges

    return BUILTIN_CHALLENGES


def _score_args(predicted: dict, expected: dict) -> float:
    """Score predicted arguments against expected. Returns 0.0–1.0."""
    if not expected:
        return 1.0 if not predicted else 0.5

    scores = []
    for key, exp_val in expected.items():
        pred_val = predicted.get(key)
        if pred_val is None:
            scores.append(0.0)
            continue

        # Normalize for comparison.
        if isinstance(exp_val, str) and isinstance(pred_val, str):
            # Fuzzy string match: check containment.
            exp_norm = exp_val.lower().strip().rstrip("/")
            pred_norm = pred_val.lower().strip().rstrip("/")
            if exp_norm == pred_norm:
                scores.append(1.0)
            elif exp_norm in pred_norm or pred_norm in exp_norm:
                scores.append(0.7)
            else:
                scores.append(0.0)
        elif isinstance(exp_val, list) and isinstance(pred_val, list):
            # List: check overlap.
            exp_set = set(str(v).lower() for v in exp_val)
            pred_set = set(str(v).lower() for v in pred_val)
            if exp_set == pred_set:
                scores.append(1.0)
            elif exp_set & pred_set:
                scores.append(len(exp_set & pred_set) / len(exp_set | pred_set))
            else:
                scores.append(0.0)
        elif isinstance(exp_val, list) and isinstance(pred_val, str):
            # Model returned comma-separated string instead of list.
            pred_items = set(v.strip().lower() for v in pred_val.split(","))
            exp_set = set(str(v).lower() for v in exp_val)
            overlap = exp_set & pred_items
            scores.append(len(overlap) / len(exp_set) if exp_set else 0.5)
        else:
            scores.append(1.0 if str(exp_val) == str(pred_val) else 0.0)

    return sum(scores) / len(scores) if scores else 0.0


def evaluate_challenge(challenge: ChainArgChallenge, model_id: str,
                       client, backend_config) -> ChainArgResult:
    """Evaluate a single chain-arg extraction challenge."""
    system = SYSTEM_PROMPT.format(chains=_format_chains())
    prompt = f"Engagement goal: {challenge.goal}"

    start = time.monotonic()
    try:
        response = client.send_prompt(
            config=backend_config,
            prompt=prompt,
            system_prompt=system,
        )
        latency = (time.monotonic() - start) * 1000

        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        parsed = json.loads(content)
        pred_chain = parsed.get("chain", "")
        pred_args = parsed.get("args", {})

        chain_correct = pred_chain == challenge.expected_chain
        arg_score = _score_args(pred_args, challenge.expected_args)
        composite = 0.5 * (1.0 if chain_correct else 0.0) + 0.5 * arg_score

        return ChainArgResult(
            challenge_id=challenge.id,
            model_id=model_id,
            predicted_chain=pred_chain,
            predicted_args=pred_args,
            expected_chain=challenge.expected_chain,
            expected_args=challenge.expected_args,
            chain_correct=chain_correct,
            arg_score=arg_score,
            composite_score=composite,
            latency_ms=latency,
        )
    except json.JSONDecodeError as e:
        latency = (time.monotonic() - start) * 1000
        return ChainArgResult(
            challenge_id=challenge.id, model_id=model_id,
            predicted_chain="", predicted_args={},
            expected_chain=challenge.expected_chain, expected_args=challenge.expected_args,
            chain_correct=False, arg_score=0.0, composite_score=0.0,
            latency_ms=latency, error=f"JSON parse error: {e}",
        )
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        return ChainArgResult(
            challenge_id=challenge.id, model_id=model_id,
            predicted_chain="", predicted_args={},
            expected_chain=challenge.expected_chain, expected_args=challenge.expected_args,
            chain_correct=False, arg_score=0.0, composite_score=0.0,
            latency_ms=latency, error=str(e),
        )


def run_chain_arg_eval(models: list[str], challenges_dir: str,
                       results_dir: str, backends: Optional[list] = None):
    """Run chain-arg extraction eval and save results."""
    from harness import LLMClient, build_backend_configs_for_models

    challenges = load_challenges(challenges_dir)
    print(f"Loaded {len(challenges)} chain-arg challenges")
    print(f"Models: {', '.join(models)}")

    client = LLMClient()
    all_results = []
    all_summaries = {}

    for model_id in models:
        print(f"\n--- Evaluating {model_id} ---")
        configs = build_backend_configs_for_models([model_id], backends=backends)
        if not configs:
            print(f"  Skipping {model_id}: no backend config available", file=sys.stderr)
            continue

        config = configs[0]
        model_results = []
        for i, challenge in enumerate(challenges):
            print(f"  [{i+1}/{len(challenges)}] {challenge.id}", end="")
            result = evaluate_challenge(challenge, model_id, client, config)
            model_results.append(result)
            chain_ok = "✓" if result.chain_correct else "✗"
            print(f" chain={chain_ok} args={result.arg_score:.2f} composite={result.composite_score:.2f} ({result.latency_ms:.0f}ms)")

        total = len(model_results)
        chain_acc = sum(1 for r in model_results if r.chain_correct) / total if total else 0
        avg_composite = sum(r.composite_score for r in model_results) / total if total else 0
        errors = sum(1 for r in model_results if r.error)

        summary = {
            "model_id": model_id,
            "total": total,
            "chain_accuracy": round(chain_acc, 4),
            "avg_composite": round(avg_composite, 4),
            "errors": errors,
        }
        all_summaries[model_id] = summary
        all_results.extend(model_results)

        print(f"\n  {model_id}: chain_accuracy={chain_acc:.1%} avg_composite={avg_composite:.3f}")

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    output = {
        "benchmark": "chain_arg_extraction",
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_challenges": len(challenges),
        "models": all_summaries,
        "results": [asdict(r) for r in all_results],
    }

    out_path = os.path.join(results_dir, "chain_arg_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Run chain-arg extraction eval")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--challenges-dir", default="data/chain_arg")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--backends", nargs="+")
    args = parser.parse_args()
    run_chain_arg_eval(args.models, args.challenges_dir, args.results_dir, args.backends)


if __name__ == "__main__":
    main()
