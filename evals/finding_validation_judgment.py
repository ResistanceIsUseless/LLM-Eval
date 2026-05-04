#!/usr/bin/env python3
"""
Finding Validation Judgment Eval — Phase 2 LLM-Eval benchmark.

Tests whether a model can correctly classify a security finding as
confirmed, unconfirmed, or needs-review based on the original finding
claim and the validator chain's output.

This directly maps to zoidberg's solver loop where Claude Code claims
a finding and a validator chain runs to verify it. The LLM must then
make the final judgment call.

Usage:
  python -m evals.finding_validation_judgment --models claude-haiku-4-5-20251001
  python harness.py eval-finding-judgment --models claude-haiku-4-5-20251001
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
class FindingChallenge:
    """A single finding validation challenge."""
    id: str
    finding_claim: str                    # what Claude Code claims to have found
    vuln_class: str                       # xss, sqli, ssrf, etc.
    validator_output: str                 # raw output from the validator chain
    ground_truth: str                     # confirmed, unconfirmed, needs_review
    difficulty: str = "intermediate"
    notes: str = ""


@dataclass
class FindingResult:
    """Result of one judgment attempt."""
    challenge_id: str
    model_id: str
    predicted: str                        # confirmed, unconfirmed, needs_review
    confidence: float
    reasoning: str
    ground_truth: str
    correct: bool
    latency_ms: float
    error: Optional[str] = None


SYSTEM_PROMPT = """You are a security finding validator. Given a vulnerability claim and the
output from an automated validation chain, classify the finding.

Respond with EXACTLY this JSON format:
{
  "classification": "confirmed|unconfirmed|needs_review",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation"
}

Classification guidelines:
- **confirmed**: Validator output shows clear evidence the vulnerability is real and exploitable
- **unconfirmed**: Validator output shows the vulnerability does NOT exist or is not exploitable
- **needs_review**: Validator output is ambiguous, incomplete, or contradictory — human review needed

Be conservative:
- Only classify as "confirmed" when evidence is unambiguous
- "needs_review" is appropriate when the validator output is partial or unclear
- False confirmations are worse than missed findings in this context"""


BUILTIN_CHALLENGES = [
    FindingChallenge(
        id="fv-001",
        finding_claim="Reflected XSS in search parameter on https://app.example.com/search?q=test",
        vuln_class="xss",
        validator_output="""validate-xss chain output:
Step 1: Sent payload <script>alert('xss')</script> in q parameter
Step 2: Response contains unescaped payload in HTML body:
  <div class="results">Results for: <script>alert('xss')</script></div>
Step 3: Content-Type is text/html, no CSP header present
Step 4: Payload renders in browser context without sanitization
Result: VULNERABLE — reflected XSS confirmed, payload executes in response""",
        ground_truth="confirmed",
        difficulty="beginner",
    ),
    FindingChallenge(
        id="fv-002",
        finding_claim="SQL injection in login form at https://app.example.com/login",
        vuln_class="sqli",
        validator_output="""validate-sqli chain output:
Step 1: Sent payload admin' OR '1'='1 in username field
Step 2: Server returned 401 Unauthorized with generic error message
Step 3: Sent payload admin'; DROP TABLE users;-- in username field
Step 4: Server returned 401 Unauthorized with same generic error message
Step 5: Time-based test: admin'; WAITFOR DELAY '0:0:5';-- — response time 43ms (no delay)
Step 6: Boolean-based test: admin' AND '1'='1 vs admin' AND '1'='2 — same response
Result: NOT VULNERABLE — all injection attempts returned identical error responses, no evidence of SQL interpretation""",
        ground_truth="unconfirmed",
        difficulty="beginner",
    ),
    FindingChallenge(
        id="fv-003",
        finding_claim="SSRF via image URL parameter at https://app.example.com/avatar?url=",
        vuln_class="ssrf",
        validator_output="""validate-ssrf chain output:
Step 1: Sent url=http://169.254.169.254/latest/meta-data/
Step 2: Server returned 200 OK but response body is: {"error": "Invalid image URL"}
Step 3: Sent url=http://burp-collaborator.example.net/ssrf-test
Step 4: Received DNS lookup on collaborator but no HTTP request
Step 5: Sent url=http://127.0.0.1:6379/
Step 6: Server returned 200 OK with body: {"error": "Invalid image format"}
Result: PARTIAL — DNS resolution occurs (Step 4) suggesting outbound resolution, but no HTTP content is returned. Possible blind SSRF.""",
        ground_truth="needs_review",
        difficulty="intermediate",
    ),
    FindingChallenge(
        id="fv-004",
        finding_claim="Stored XSS in comment field on https://forum.example.com/post/123",
        vuln_class="xss",
        validator_output="""validate-xss chain output:
Step 1: Posted comment with payload: <img src=x onerror=alert(document.cookie)>
Step 2: Comment stored successfully (201 Created)
Step 3: Retrieved post page — response contains:
  <div class="comment"><img src=x onerror=alert(document.cookie)></div>
Step 4: No Content-Security-Policy header present
Step 5: JavaScript execution confirmed in headless browser — alert triggered
Result: VULNERABLE — stored XSS confirmed, payload persists and executes on page load""",
        ground_truth="confirmed",
        difficulty="beginner",
    ),
    FindingChallenge(
        id="fv-005",
        finding_claim="Command injection in filename parameter at https://app.example.com/export?file=report.pdf",
        vuln_class="cmdi",
        validator_output="""validate-cmdi chain output:
Step 1: Sent file=report.pdf;id
Step 2: Server returned 200 with PDF content (no command output visible)
Step 3: Sent file=report.pdf|id
Step 4: Server returned 400 Bad Request: "Invalid filename"
Step 5: Sent file=report.pdf$(sleep 5)
Step 6: Response time 203ms (no delay observed)
Step 7: Sent file=report.pdf`sleep 5`
Step 8: Response time 5147ms (5-second delay!)
Step 9: Sent file=report.pdf`sleep 10`
Step 10: Response time 10089ms (10-second delay!)
Result: VULNERABLE — time-based command injection confirmed via backtick injection in filename parameter""",
        ground_truth="confirmed",
        difficulty="intermediate",
    ),
    FindingChallenge(
        id="fv-006",
        finding_claim="IDOR allowing access to other users' profiles at https://api.example.com/users/42",
        vuln_class="idor",
        validator_output="""validate-idor chain output:
Step 1: Authenticated as user 42, requested /users/42 — 200 OK with profile data
Step 2: Requested /users/43 (different user) — 200 OK with profile data
Step 3: Requested /users/1 (admin user) — 200 OK with profile data
BUT: All responses contain only public fields (name, bio, avatar_url)
Step 4: Requested /users/42/settings (private endpoint) — 200 OK with email, phone
Step 5: Requested /users/43/settings — 403 Forbidden
Result: NOT VULNERABLE — /users/{id} returns only public profile data (by design). Private endpoints properly enforce authorization.""",
        ground_truth="unconfirmed",
        difficulty="advanced",
    ),
    FindingChallenge(
        id="fv-007",
        finding_claim="Blind SQL injection in order_id parameter at https://shop.example.com/orders?order_id=100",
        vuln_class="sqli",
        validator_output="""validate-sqli chain output:
Step 1: Sent order_id=100 — 200 OK with order details
Step 2: Sent order_id=100 AND 1=1 — 200 OK with order details
Step 3: Sent order_id=100 AND 1=2 — 200 OK with empty results
Step 4: Sent order_id=100 AND (SELECT COUNT(*) FROM information_schema.tables)>0 — 200 OK with order details
Step 5: Error injection: order_id=100' — 200 OK with empty results (no SQL error)
Step 6: Time-based: order_id=100; WAITFOR DELAY '0:0:5' — response time 198ms (no delay)
Step 7: UNION: order_id=100 UNION SELECT null,null,null — 200 OK with empty results
Note: Steps 2–4 show boolean-based differential response, but could be application-level filtering not SQL.
Result: INCONCLUSIVE — boolean differential exists but time-based and UNION tests negative. Could be integer casting, not SQL injection.""",
        ground_truth="needs_review",
        difficulty="advanced",
    ),
]


def load_challenges(challenges_dir: str) -> list[FindingChallenge]:
    """Load challenges from directory, falling back to built-ins."""
    cdir = Path(challenges_dir)
    if cdir.exists():
        challenges = []
        for f in sorted(cdir.glob("*.json")):
            with open(f) as fh:
                data = json.load(fh)
            if isinstance(data, list):
                for item in data:
                    challenges.append(FindingChallenge(**item))
            else:
                challenges.append(FindingChallenge(**data))
        if challenges:
            return challenges
    return BUILTIN_CHALLENGES


def evaluate_challenge(challenge: FindingChallenge, model_id: str,
                       client, backend_config) -> FindingResult:
    """Evaluate a single finding validation challenge."""
    prompt = f"""**Finding Claim:**
{challenge.finding_claim}

**Vulnerability Class:** {challenge.vuln_class}

**Validator Chain Output:**
```
{challenge.validator_output}
```

Based on the validator output, classify this finding."""

    start = time.monotonic()
    try:
        response = client.send_prompt(
            config=backend_config,
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        )
        latency = (time.monotonic() - start) * 1000

        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        parsed = json.loads(content)
        predicted = parsed.get("classification", "").lower()
        confidence = float(parsed.get("confidence", 0.5))
        reasoning = parsed.get("reasoning", "")

        correct = predicted == challenge.ground_truth
        return FindingResult(
            challenge_id=challenge.id,
            model_id=model_id,
            predicted=predicted,
            confidence=confidence,
            reasoning=reasoning,
            ground_truth=challenge.ground_truth,
            correct=correct,
            latency_ms=latency,
        )
    except json.JSONDecodeError as e:
        latency = (time.monotonic() - start) * 1000
        return FindingResult(
            challenge_id=challenge.id, model_id=model_id,
            predicted="", confidence=0.0, reasoning="",
            ground_truth=challenge.ground_truth, correct=False,
            latency_ms=latency, error=f"JSON parse error: {e}",
        )
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        return FindingResult(
            challenge_id=challenge.id, model_id=model_id,
            predicted="", confidence=0.0, reasoning="",
            ground_truth=challenge.ground_truth, correct=False,
            latency_ms=latency, error=str(e),
        )


def run_finding_judgment_eval(models: list[str], challenges_dir: str,
                              results_dir: str, backends: Optional[list] = None):
    """Run finding validation judgment eval and save results."""
    from harness import LLMClient, build_backend_configs_for_models

    challenges = load_challenges(challenges_dir)
    print(f"Loaded {len(challenges)} finding validation challenges")
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
            print(f"  [{i+1}/{len(challenges)}] {challenge.id} ({challenge.vuln_class})", end="")
            result = evaluate_challenge(challenge, model_id, client, config)
            model_results.append(result)
            status = "✓" if result.correct else ("✗" if not result.error else "E")
            print(f" {status} predicted={result.predicted} truth={result.ground_truth} ({result.latency_ms:.0f}ms)")

        total = len(model_results)
        accuracy = sum(1 for r in model_results if r.correct) / total if total else 0
        errors = sum(1 for r in model_results if r.error)

        # Per-class metrics.
        classes = ["confirmed", "unconfirmed", "needs_review"]
        per_class = {}
        for cls in classes:
            tp = sum(1 for r in model_results if r.predicted == cls and r.ground_truth == cls)
            fp = sum(1 for r in model_results if r.predicted == cls and r.ground_truth != cls)
            fn = sum(1 for r in model_results if r.predicted != cls and r.ground_truth == cls)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            per_class[cls] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

        summary = {
            "model_id": model_id,
            "total": total,
            "accuracy": round(accuracy, 4),
            "errors": errors,
            "per_class": per_class,
        }
        all_summaries[model_id] = summary
        all_results.extend(model_results)

        print(f"\n  {model_id}: accuracy={accuracy:.1%}")
        for cls, metrics in per_class.items():
            print(f"    {cls}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    output = {
        "benchmark": "finding_validation_judgment",
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_challenges": len(challenges),
        "models": all_summaries,
        "results": [asdict(r) for r in all_results],
    }

    out_path = os.path.join(results_dir, "finding_judgment_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Run finding validation judgment eval")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--challenges-dir", default="data/finding_validation")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--backends", nargs="+")
    args = parser.parse_args()
    run_finding_judgment_eval(args.models, args.challenges_dir, args.results_dir, args.backends)


if __name__ == "__main__":
    main()
