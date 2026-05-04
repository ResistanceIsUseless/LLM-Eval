#!/usr/bin/env python3
"""
Solver Restart Summary Eval — Phase 2 LLM-Eval benchmark.

Tests whether a model can produce a faithful, concise summary of a long
solver session log that preserves:
  - What was tried (tools run, findings discovered)
  - What worked vs. what failed
  - What's left to do
  - Key context needed to resume without re-running completed steps

This maps to zoidberg's solver loop where a session may be interrupted
(budget, timeout, operator pause) and needs to be resumed with context.

Scoring uses Opus-as-Judge with a structured rubric evaluating:
  - Faithfulness (no hallucinated steps or findings)
  - Completeness (all key events mentioned)
  - Actionability (reader can resume without re-reading the full log)
  - Conciseness (no unnecessary detail)

Usage:
  python -m evals.solver_restart_summary --models claude-haiku-4-5-20251001
  python harness.py eval-solver-summary --models claude-haiku-4-5-20251001
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
class SolverSessionChallenge:
    """A solver session log to be summarized."""
    id: str
    session_log: str                      # the full session log
    goal: str                             # original engagement goal
    key_events: list                      # events the summary MUST mention
    findings: list                        # findings discovered (with status)
    remaining_tasks: list                 # what's left to do
    difficulty: str = "intermediate"
    max_summary_words: int = 300


@dataclass
class SolverSummaryResult:
    """Result of one summarization attempt."""
    challenge_id: str
    model_id: str
    summary: str
    faithfulness: float                   # 0–5 judge score
    completeness: float                   # 0–5 judge score
    actionability: float                  # 0–5 judge score
    conciseness: float                    # 0–5 judge score
    composite: float                      # weighted average
    latency_ms: float
    judge_error: Optional[str] = None
    error: Optional[str] = None


SYSTEM_PROMPT = """You are a security engagement session summarizer. Given a solver session log,
produce a concise restart summary that allows a new operator (or resumed agent) to continue
the investigation without re-reading the full log.

Your summary MUST include:
1. **Goal**: The original engagement objective (one line)
2. **Completed**: What tools were run and what they found (bulleted list)
3. **Findings**: Any confirmed, unconfirmed, or pending findings (with status)
4. **Failed/Blocked**: Steps that failed and why
5. **Remaining**: What's left to do (prioritized list)
6. **Key Context**: Any non-obvious state or configuration the next session needs

Rules:
- Be factual — only include events from the log
- Do NOT hallucinate findings or steps that didn't happen
- Keep it under {max_words} words
- Use structured format (headers + bullets)
- Preserve target names, tool names, and finding classifications exactly"""


JUDGE_PROMPT = """You are evaluating the quality of a solver session restart summary.

**Original Session Log:**
```
{session_log}
```

**Key Events That Must Be Mentioned:**
{key_events}

**Findings Discovered:**
{findings}

**Remaining Tasks:**
{remaining}

**Model's Summary:**
```
{summary}
```

Score the summary on four dimensions (0–5 scale each):

1. **Faithfulness** (0–5): Does the summary only contain information from the log? Any hallucinated steps, findings, or details that weren't in the log score 0.
2. **Completeness** (0–5): Does the summary mention all key events, findings, and remaining tasks? Missing critical items reduces the score.
3. **Actionability** (0–5): Could someone resume the engagement using only this summary? Does it clearly state what's done vs. remaining?
4. **Conciseness** (0–5): Is the summary appropriately brief? Unnecessary verbosity reduces the score. Omitting important details to be brief also reduces it.

Respond with EXACTLY this JSON:
{{
  "faithfulness": 0-5,
  "completeness": 0-5,
  "actionability": 0-5,
  "conciseness": 0-5,
  "notes": "Brief evaluation notes"
}}"""


BUILTIN_CHALLENGES = [
    SolverSessionChallenge(
        id="ss-001",
        goal="Find and validate web vulnerabilities on shop.example.com",
        session_log="""[00:00] Session start — goal: find web vulns on shop.example.com, budget: $5.00
[00:01] Running chain: recon-basic target=shop.example.com
[00:15] recon-basic complete:
  - Subdomains: shop.example.com, api.shop.example.com, admin.shop.example.com
  - Tech: Nginx 1.24, PHP 8.1, MySQL (detected via headers)
  - Ports: 80, 443, 8080 (admin panel)
[00:16] Claude Code: "admin panel on 8080 is interesting — let's check for auth issues"
[00:17] Running chain: header-audit url=https://shop.example.com
[00:22] header-audit complete:
  - Missing: X-Frame-Options, CSP, HSTS
  - Present: X-Content-Type-Options: nosniff
[00:23] Running chain: xss-reflected url=https://shop.example.com/search params=q,category
[00:35] xss-reflected complete:
  - q param: reflected unescaped in HTML body — POTENTIAL XSS
  - category param: properly escaped
[00:36] Running chain: validate-xss url=https://shop.example.com/search?q=test payload=<script>alert(1)</script>
[00:42] validate-xss: CONFIRMED — reflected XSS in q parameter, payload executes
[00:42] Finding logged: XSS-001 (confirmed) — reflected XSS in search q param
[00:43] Running chain: sqli-error url=https://api.shop.example.com/products params=id,category
[00:55] sqli-error: ERROR — connection timeout to api.shop.example.com (target may be rate-limiting)
[00:56] Claude Code: "API is blocking us, switching to admin panel"
[00:57] Running chain: cors-check url=https://admin.shop.example.com:8080
[01:02] cors-check complete:
  - Access-Control-Allow-Origin: * (overly permissive!)
  - Credentials: true
[01:02] Finding logged: CORS-001 (unconfirmed) — permissive CORS on admin panel, needs manual validation
[01:03] Budget remaining: $2.15
[01:03] Session paused by operator""",
        key_events=[
            "recon-basic found 3 subdomains and admin panel on 8080",
            "header-audit found missing security headers",
            "XSS found and confirmed in search q parameter",
            "SQLi test failed due to rate limiting on API",
            "CORS misconfiguration found on admin panel",
            "Session paused with $2.15 remaining",
        ],
        findings=[
            {"id": "XSS-001", "status": "confirmed", "desc": "reflected XSS in search q param"},
            {"id": "CORS-001", "status": "unconfirmed", "desc": "permissive CORS on admin panel"},
        ],
        remaining_tasks=[
            "Retry SQLi test on api.shop.example.com (was rate-limited)",
            "Validate CORS finding on admin panel",
            "Test admin panel at :8080 for auth bypass",
            "Check for stored XSS in other input fields",
        ],
        difficulty="beginner",
    ),
    SolverSessionChallenge(
        id="ss-002",
        goal="Investigate OAuth implementation on auth.megacorp.io for PKCE downgrade and token leakage",
        session_log="""[00:00] Session start — goal: OAuth investigation on auth.megacorp.io, budget: $8.00
[00:01] Running chain: recon-basic target=auth.megacorp.io
[00:12] recon-basic complete:
  - Tech: Node.js, Express, MongoDB
  - OAuth endpoints: /oauth/authorize, /oauth/token, /oauth/userinfo
  - OpenID configuration at /.well-known/openid-configuration
[00:13] Claude Code: "Let me analyze the OpenID config first"
[00:14] HTTP GET /.well-known/openid-configuration
[00:14] Response:
  - grant_types_supported: authorization_code, implicit, client_credentials
  - code_challenge_methods_supported: S256, plain
  - token_endpoint_auth_methods_supported: client_secret_post, client_secret_basic, none
[00:15] Claude Code: "Several concerns: implicit grant still supported, PKCE supports 'plain' method, and 'none' auth method on token endpoint"
[00:16] Running chain: oauth-probe auth_url=https://auth.megacorp.io/oauth/authorize client_id=test-app
[00:28] oauth-probe complete:
  - PKCE bypass: Sending code_challenge_method=plain with predictable verifier — token issued! VULNERABILITY
  - Implicit flow: Returns access_token in URL fragment — information exposure risk
  - Token endpoint accepts none for client auth — public client can exchange codes
  - State parameter not validated (can be omitted)
[00:29] Finding logged: OAUTH-001 (unconfirmed) — PKCE downgrade via plain method
[00:29] Finding logged: OAUTH-002 (unconfirmed) — implicit flow enabled
[00:30] Finding logged: OAUTH-003 (unconfirmed) — token endpoint accepts none auth
[00:30] Finding logged: OAUTH-004 (unconfirmed) — missing state validation
[00:31] Claude Code: "Need to validate PKCE downgrade — this is the highest impact finding"
[00:32] Running custom validation: full PKCE flow with plain challenge method
[00:45] Validation steps:
  1. Initiated auth flow with code_challenge_method=plain, code_challenge=AAAA
  2. Received authorization code after consent
  3. Exchanged code at token endpoint with code_verifier=AAAA
  4. Received valid access_token + refresh_token
  5. Used access_token to call /oauth/userinfo — got full profile
[00:45] Finding OAUTH-001 upgraded: CONFIRMED — PKCE downgrade, attacker can intercept auth code and exchange with known verifier
[00:46] Running custom validation: implicit flow token in URL
[00:52] Validation steps:
  1. Initiated implicit flow with response_type=token
  2. Token returned in URL fragment
  3. Token is valid (verified against /oauth/userinfo)
  4. BUT: token has limited scope (read:profile only), 5-minute expiry
[00:52] Finding OAUTH-002: status unchanged (unconfirmed) — implicit flow works but limited scope/expiry reduces impact
[00:53] Budget remaining: $3.80
[00:54] Claude Code: "PKCE downgrade is the critical finding. Should validate state bypass and none auth next"
[00:55] Running chain: nuclei-cve target=auth.megacorp.io severity=high,critical
[01:05] nuclei-cve: 0 findings (no known CVEs matched)
[01:06] Budget remaining: $2.95
[01:06] Session paused — operator reviewing findings""",
        key_events=[
            "Recon identified OAuth endpoints and OpenID configuration",
            "OpenID config revealed: implicit grant, plain PKCE, none auth method",
            "oauth-probe found 4 potential issues",
            "PKCE downgrade validated as confirmed",
            "Implicit flow partially validated but low impact (limited scope/expiry)",
            "nuclei-cve scan found nothing",
            "State bypass and none-auth still need validation",
        ],
        findings=[
            {"id": "OAUTH-001", "status": "confirmed", "desc": "PKCE downgrade via plain method"},
            {"id": "OAUTH-002", "status": "unconfirmed", "desc": "implicit flow enabled (limited scope)"},
            {"id": "OAUTH-003", "status": "unconfirmed", "desc": "token endpoint accepts none auth"},
            {"id": "OAUTH-004", "status": "unconfirmed", "desc": "missing state validation"},
        ],
        remaining_tasks=[
            "Validate OAUTH-003: test if none auth allows unauthorized code exchange",
            "Validate OAUTH-004: test state parameter CSRF attack",
            "Re-evaluate OAUTH-002 impact with broader scope testing",
            "Test for token fixation or replay attacks",
        ],
        difficulty="advanced",
    ),
]


def load_challenges(challenges_dir: str) -> list[SolverSessionChallenge]:
    """Load challenges from directory, falling back to built-ins."""
    cdir = Path(challenges_dir)
    if cdir.exists():
        challenges = []
        for f in sorted(cdir.glob("*.json")):
            with open(f) as fh:
                data = json.load(fh)
            if isinstance(data, list):
                for item in data:
                    challenges.append(SolverSessionChallenge(**item))
            else:
                challenges.append(SolverSessionChallenge(**data))
        if challenges:
            return challenges
    return BUILTIN_CHALLENGES


def evaluate_challenge(challenge: SolverSessionChallenge, model_id: str,
                       client, backend_config) -> SolverSummaryResult:
    """Generate and judge a summary for one session challenge."""
    system = SYSTEM_PROMPT.format(max_words=challenge.max_summary_words)
    prompt = f"""**Engagement Goal:** {challenge.goal}

**Session Log:**
```
{challenge.session_log}
```

Produce a restart summary for the next session."""

    start = time.monotonic()
    try:
        response = client.send_prompt(
            config=backend_config,
            prompt=prompt,
            system_prompt=system,
        )
        latency = (time.monotonic() - start) * 1000
        summary = response.content.strip()
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        return SolverSummaryResult(
            challenge_id=challenge.id, model_id=model_id,
            summary="", faithfulness=0, completeness=0, actionability=0, conciseness=0,
            composite=0, latency_ms=latency, error=str(e),
        )

    # Now judge the summary with the judge model.
    return SolverSummaryResult(
        challenge_id=challenge.id,
        model_id=model_id,
        summary=summary,
        faithfulness=0, completeness=0, actionability=0, conciseness=0,
        composite=0,
        latency_ms=latency,
    )


def judge_summary(result: SolverSummaryResult, challenge: SolverSessionChallenge,
                  client, judge_config) -> SolverSummaryResult:
    """Use Opus-as-Judge to score a summary."""
    prompt = JUDGE_PROMPT.format(
        session_log=challenge.session_log,
        key_events="\n".join(f"- {e}" for e in challenge.key_events),
        findings="\n".join(f"- {f['id']} ({f['status']}): {f['desc']}" for f in challenge.findings),
        remaining="\n".join(f"- {t}" for t in challenge.remaining_tasks),
        summary=result.summary,
    )

    try:
        response = client.send_prompt(
            config=judge_config,
            prompt=prompt,
            system_prompt="You are an expert evaluator of security engagement summaries.",
        )
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        parsed = json.loads(content)
        result.faithfulness = float(parsed.get("faithfulness", 0))
        result.completeness = float(parsed.get("completeness", 0))
        result.actionability = float(parsed.get("actionability", 0))
        result.conciseness = float(parsed.get("conciseness", 0))
        result.composite = (
            0.35 * result.faithfulness +
            0.25 * result.completeness +
            0.25 * result.actionability +
            0.15 * result.conciseness
        )
    except Exception as e:
        result.judge_error = str(e)

    return result


def run_solver_summary_eval(models: list[str], challenges_dir: str,
                            results_dir: str, backends: Optional[list] = None):
    """Run solver restart summary eval and save results."""
    from harness import LLMClient, build_backend_configs_for_models, build_judge_config

    challenges = load_challenges(challenges_dir)
    print(f"Loaded {len(challenges)} solver session challenges")
    print(f"Models: {', '.join(models)}")

    client = LLMClient()
    judge_config = build_judge_config()
    if not judge_config:
        print("Warning: no judge config available, scores will be 0", file=sys.stderr)

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
            if result.error:
                print(f" ERROR: {result.error}")
                model_results.append(result)
                continue

            if judge_config:
                result = judge_summary(result, challenge, client, judge_config)

            model_results.append(result)
            if result.judge_error:
                print(f" judge-error ({result.latency_ms:.0f}ms)")
            else:
                print(f" F={result.faithfulness:.1f} C={result.completeness:.1f} A={result.actionability:.1f} B={result.conciseness:.1f} composite={result.composite:.2f} ({result.latency_ms:.0f}ms)")

        total = len(model_results)
        scored = [r for r in model_results if not r.error and not r.judge_error]
        avg_composite = sum(r.composite for r in scored) / len(scored) if scored else 0
        avg_faith = sum(r.faithfulness for r in scored) / len(scored) if scored else 0

        summary = {
            "model_id": model_id,
            "total": total,
            "scored": len(scored),
            "avg_composite": round(avg_composite, 4),
            "avg_faithfulness": round(avg_faith, 4),
            "avg_completeness": round(sum(r.completeness for r in scored) / len(scored) if scored else 0, 4),
            "avg_actionability": round(sum(r.actionability for r in scored) / len(scored) if scored else 0, 4),
            "avg_conciseness": round(sum(r.conciseness for r in scored) / len(scored) if scored else 0, 4),
        }
        all_summaries[model_id] = summary
        all_results.extend(model_results)

        print(f"\n  {model_id}: composite={avg_composite:.3f} faithfulness={avg_faith:.1f}")

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    output = {
        "benchmark": "solver_restart_summary",
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_challenges": len(challenges),
        "models": all_summaries,
        "results": [asdict(r) for r in all_results],
    }

    out_path = os.path.join(results_dir, "solver_summary_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Run solver restart summary eval")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--challenges-dir", default="data/solver_sessions")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--backends", nargs="+")
    args = parser.parse_args()
    run_solver_summary_eval(args.models, args.challenges_dir, args.results_dir, args.backends)


if __name__ == "__main__":
    main()
