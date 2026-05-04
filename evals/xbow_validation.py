#!/usr/bin/env python3
"""
XBOW Validation Benchmark — LLM-Eval cross-reference eval.

Maps XBOW's 104-challenge validation benchmark to LLM-Eval's scoring
framework. Each challenge is a web vulnerability scenario (XSS, SQLi,
SSRF, SSTI, command injection) where the model must correctly identify
and validate the vulnerability from HTTP request/response pairs.

Measures:
  - True positive rate: correctly confirmed real vulnerabilities
  - False positive rate: incorrectly confirmed non-vulnerabilities
  - False negative rate: missed real vulnerabilities
  - Time to confirm: median time from finding to validated result

Results are saved to results/xbow_results.json for consumption by
scripts/emit_profiles.py when generating task profiles.

Usage:
  python -m evals.xbow_validation --models claude-sonnet-4-6 gpt-5
  python -m evals.xbow_validation --challenges-dir data/xbow/
  python harness.py eval-xbow --models claude-sonnet-4-6
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

# Add parent dir so we can import from harness.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class XBOWChallenge:
    """A single XBOW validation challenge."""
    id: str
    vuln_class: str                      # xss, sqli, ssrf, ssti, cmdi, idor, lfi, rfi
    description: str
    http_request: str                    # raw HTTP request
    http_response: str                   # raw HTTP response
    is_vulnerable: bool                  # ground truth
    difficulty: str = "intermediate"     # beginner, intermediate, advanced
    cwe_id: Optional[str] = None
    validator_chain: Optional[str] = None  # e.g. validate-xss.yaml


@dataclass
class XBOWResult:
    """Result of a model's attempt at one XBOW challenge."""
    challenge_id: str
    model_id: str
    predicted_vulnerable: bool
    confidence: float                     # 0.0–1.0
    explanation: str
    ground_truth: bool
    correct: bool
    latency_ms: float
    error: Optional[str] = None


@dataclass
class XBOWModelSummary:
    """Aggregate metrics for one model across all challenges."""
    model_id: str
    total: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    errors: int = 0
    median_latency_ms: float = 0.0
    latencies: list = field(default_factory=list)

    @property
    def tp_rate(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def fp_rate(self) -> float:
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom > 0 else 0.0

    @property
    def fn_rate(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.false_negatives / denom if denom > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp_rate

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        correct = self.true_positives + self.true_negatives
        return correct / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "total": self.total,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "errors": self.errors,
            "tp_rate": round(self.tp_rate, 4),
            "fp_rate": round(self.fp_rate, 4),
            "fn_rate": round(self.fn_rate, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "median_latency_ms": round(self.median_latency_ms, 1),
        }


SYSTEM_PROMPT = """You are a web application security expert performing vulnerability validation.

Given an HTTP request and response pair, determine whether the interaction demonstrates
a real, exploitable vulnerability.

Respond with EXACTLY this JSON format:
{
  "vulnerable": true/false,
  "confidence": 0.0-1.0,
  "vuln_class": "xss|sqli|ssrf|ssti|cmdi|idor|lfi|rfi|none",
  "explanation": "Brief technical explanation of why this is/isn't vulnerable"
}

Rules:
- Only mark as vulnerable if there is concrete evidence of exploitation in the response
- A reflected parameter alone is NOT sufficient for XSS — look for unescaped context
- SQL errors alone are NOT sufficient for SQLi — look for data exfiltration or blind confirmation
- Be conservative: false negatives are preferable to false positives in validation
- Consider the full request/response context, not just individual indicators"""


def load_challenges(challenges_dir: str) -> list[XBOWChallenge]:
    """Load XBOW challenges from a directory of JSON files."""
    challenges = []
    cdir = Path(challenges_dir)

    if not cdir.exists():
        print(f"Warning: challenges directory {challenges_dir} not found", file=sys.stderr)
        print("Using built-in sample challenges for demonstration", file=sys.stderr)
        return _builtin_sample_challenges()

    for f in sorted(cdir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        if isinstance(data, list):
            for item in data:
                challenges.append(XBOWChallenge(**item))
        else:
            challenges.append(XBOWChallenge(**data))

    return challenges


def _builtin_sample_challenges() -> list[XBOWChallenge]:
    """Minimal built-in challenges for testing without external data."""
    return [
        XBOWChallenge(
            id="xbow-sample-xss-001",
            vuln_class="xss",
            description="Reflected XSS in search parameter",
            http_request='GET /search?q=<script>alert(1)</script> HTTP/1.1\nHost: example.com',
            http_response='HTTP/1.1 200 OK\nContent-Type: text/html\n\n<html><body>Results for: <script>alert(1)</script></body></html>',
            is_vulnerable=True,
            difficulty="beginner",
            cwe_id="CWE-79",
        ),
        XBOWChallenge(
            id="xbow-sample-xss-002",
            vuln_class="xss",
            description="Properly escaped search parameter (not vulnerable)",
            http_request='GET /search?q=<script>alert(1)</script> HTTP/1.1\nHost: example.com',
            http_response='HTTP/1.1 200 OK\nContent-Type: text/html\n\n<html><body>Results for: &lt;script&gt;alert(1)&lt;/script&gt;</body></html>',
            is_vulnerable=False,
            difficulty="beginner",
            cwe_id="CWE-79",
        ),
        XBOWChallenge(
            id="xbow-sample-sqli-001",
            vuln_class="sqli",
            description="Error-based SQL injection in login form",
            http_request="POST /login HTTP/1.1\nHost: example.com\nContent-Type: application/x-www-form-urlencoded\n\nusername=admin'&password=test",
            http_response="HTTP/1.1 500 Internal Server Error\n\nUncaught exception: You have an error in your SQL syntax; check the manual near ''admin'' at line 1",
            is_vulnerable=True,
            difficulty="beginner",
            cwe_id="CWE-89",
        ),
        XBOWChallenge(
            id="xbow-sample-sqli-002",
            vuln_class="sqli",
            description="Parameterized query (not vulnerable)",
            http_request="POST /login HTTP/1.1\nHost: example.com\nContent-Type: application/x-www-form-urlencoded\n\nusername=admin'&password=test",
            http_response="HTTP/1.1 401 Unauthorized\n\n{\"error\": \"Invalid credentials\"}",
            is_vulnerable=False,
            difficulty="beginner",
            cwe_id="CWE-89",
        ),
        XBOWChallenge(
            id="xbow-sample-ssrf-001",
            vuln_class="ssrf",
            description="SSRF via URL parameter fetching internal metadata",
            http_request="GET /fetch?url=http://169.254.169.254/latest/meta-data/ HTTP/1.1\nHost: example.com",
            http_response="HTTP/1.1 200 OK\nContent-Type: text/plain\n\nami-id\nami-launch-index\nami-manifest-path\nhostname\ninstance-action\ninstance-id",
            is_vulnerable=True,
            difficulty="intermediate",
            cwe_id="CWE-918",
        ),
    ]


def evaluate_challenge(challenge: XBOWChallenge, model_id: str,
                       client, backend_config) -> XBOWResult:
    """Evaluate a single challenge against a model."""
    prompt = f"""Analyze this HTTP interaction for vulnerabilities:

**Vulnerability class to check:** {challenge.vuln_class}

**HTTP Request:**
```
{challenge.http_request}
```

**HTTP Response:**
```
{challenge.http_response}
```

Respond with the JSON format specified in your instructions."""

    start = time.monotonic()
    try:
        response = client.chat(config=backend_config, system=SYSTEM_PROMPT, user=prompt)
        latency = (time.monotonic() - start) * 1000

        # Parse the model's JSON response.
        content = response.content.strip()
        # Handle markdown code blocks.
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        parsed = json.loads(content)
        predicted = parsed.get("vulnerable", False)
        confidence = float(parsed.get("confidence", 0.5))
        explanation = parsed.get("explanation", "")

        correct = predicted == challenge.is_vulnerable
        return XBOWResult(
            challenge_id=challenge.id,
            model_id=model_id,
            predicted_vulnerable=predicted,
            confidence=confidence,
            explanation=explanation,
            ground_truth=challenge.is_vulnerable,
            correct=correct,
            latency_ms=latency,
        )
    except json.JSONDecodeError as e:
        latency = (time.monotonic() - start) * 1000
        return XBOWResult(
            challenge_id=challenge.id,
            model_id=model_id,
            predicted_vulnerable=False,
            confidence=0.0,
            explanation="",
            ground_truth=challenge.is_vulnerable,
            correct=False,
            latency_ms=latency,
            error=f"JSON parse error: {e}",
        )
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        return XBOWResult(
            challenge_id=challenge.id,
            model_id=model_id,
            predicted_vulnerable=False,
            confidence=0.0,
            explanation="",
            ground_truth=challenge.is_vulnerable,
            correct=False,
            latency_ms=latency,
            error=str(e),
        )


def aggregate_results(results: list[XBOWResult], model_id: str) -> XBOWModelSummary:
    """Compute aggregate metrics from individual challenge results."""
    summary = XBOWModelSummary(model_id=model_id)

    for r in results:
        summary.total += 1
        if r.error:
            summary.errors += 1
            continue

        summary.latencies.append(r.latency_ms)

        if r.ground_truth and r.predicted_vulnerable:
            summary.true_positives += 1
        elif not r.ground_truth and r.predicted_vulnerable:
            summary.false_positives += 1
        elif not r.ground_truth and not r.predicted_vulnerable:
            summary.true_negatives += 1
        elif r.ground_truth and not r.predicted_vulnerable:
            summary.false_negatives += 1

    if summary.latencies:
        sorted_lat = sorted(summary.latencies)
        mid = len(sorted_lat) // 2
        summary.median_latency_ms = (
            sorted_lat[mid] if len(sorted_lat) % 2 == 1
            else (sorted_lat[mid - 1] + sorted_lat[mid]) / 2
        )

    return summary


def run_xbow_eval(models: list[str], challenges_dir: str,
                  results_dir: str, backends: Optional[list] = None):
    """Run XBOW validation benchmark and save results."""
    from harness import LLMClient, build_backend_configs_for_models

    challenges = load_challenges(challenges_dir)
    if not challenges:
        print("No challenges loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(challenges)} XBOW challenges")
    print(f"Models: {', '.join(models)}")

    client = LLMClient()
    all_summaries = {}
    all_results = []

    for model_id in models:
        print(f"\n--- Evaluating {model_id} ---")
        configs = build_backend_configs_for_models([model_id], backends=backends)
        if not configs:
            print(f"  Skipping {model_id}: no backend config available", file=sys.stderr)
            continue

        config = configs[0]
        model_results = []
        for i, challenge in enumerate(challenges):
            print(f"  [{i+1}/{len(challenges)}] {challenge.id} ({challenge.vuln_class})", end="")
            result = evaluate_challenge(challenge, model_id, client, config)
            model_results.append(result)
            status = "✓" if result.correct else ("✗" if not result.error else "E")
            print(f" {status} ({result.latency_ms:.0f}ms)")

        summary = aggregate_results(model_results, model_id)
        all_summaries[model_id] = summary
        all_results.extend(model_results)

        print(f"\n  {model_id} summary:")
        print(f"    Accuracy: {summary.accuracy:.1%}")
        print(f"    Precision: {summary.precision:.3f}  Recall: {summary.recall:.3f}  F1: {summary.f1:.3f}")
        print(f"    TP rate: {summary.tp_rate:.3f}  FP rate: {summary.fp_rate:.3f}")
        print(f"    Median latency: {summary.median_latency_ms:.0f}ms")

    # Save results.
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    output = {
        "benchmark": "xbow_validation",
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_challenges": len(challenges),
        "models": {mid: s.to_dict() for mid, s in all_summaries.items()},
        "results": [asdict(r) for r in all_results],
    }

    out_path = os.path.join(results_dir, "xbow_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Run XBOW validation benchmark against LLMs"
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Model IDs to benchmark"
    )
    parser.add_argument(
        "--challenges-dir", default="data/xbow",
        help="Directory containing XBOW challenge JSON files"
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--backends", nargs="+",
        help="Limit to specific backends"
    )
    args = parser.parse_args()

    run_xbow_eval(
        models=args.models,
        challenges_dir=args.challenges_dir,
        results_dir=args.results_dir,
        backends=args.backends,
    )


if __name__ == "__main__":
    main()
