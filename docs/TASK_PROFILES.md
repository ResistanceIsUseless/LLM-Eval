# Task Profiles

Task profiles map security-relevant LLM capabilities to optimal model selections. Each profile defines a task type, scoring criteria, and cost constraints. zoidberg's LLM router reads `profiles.yaml` to pick the right model for each call.

## Profile Taxonomy

### Phase 0 — Foundation

| Profile | Task | Optimized For | Category |
|---------|------|---------------|----------|
| `recon_summarization` | Summarize recon tool output | Cost | network_recon |
| `report_writing` | Generate engagement reports | Quality | vuln_analysis |
| `tool_selection` | Choose tool/chain for a task | Cost | network_recon |

### Phase 1 — Operator MVP

| Profile | Task | Optimized For | Category |
|---------|------|---------------|----------|
| `exploit_reasoning` | CTF-style vuln identification | Quality | exploit_dev, web_exploitation |
| `scope_classification` | In/out of scope classification | Cost | vuln_analysis |
| `prompt_injection_robustness` | Resist prompt injection | Quality | social_engineering |
| `nuclei_template_generation` | CVE → nuclei YAML template | Quality | exploit_dev |

### Phase 2 — Community Substrate

| Profile | Task | Optimized For | Category |
|---------|------|---------------|----------|
| `chain_arg_extraction` | Goal → chain + arguments | Quality | vuln_analysis, web_exploitation |
| `finding_validation_judgment` | Confirmed/unconfirmed judgment | Cost | vuln_analysis |
| `solver_restart_summary` | Session → restart context | Cost | vuln_analysis |

### Phase 3 — Cross-Reference

| Profile | Task | Optimized For | External Benchmark |
|---------|------|---------------|--------------------|
| `xbow_validation` | Web vulnerability validation | Quality | XBOW 104-challenge |
| `airtbench_alignment` | Security decision-making | Quality | AIRTBench (Dreadnode) |

## How Profiles Are Generated

```bash
python scripts/emit_profiles.py --out ~/.zoidberg/profiles.yaml
```

The script reads:
1. **Harness scores** from `results/scores.db` — per-model composite scores across test categories
2. **XBOW results** from `results/xbow_results.json` — validation benchmark scores
3. **AIRTBench results** from `results/airtbench_results.json` — alignment benchmark scores

For each profile, it picks the best-scoring model from either the "cheap" pool (Haiku, GPT-4o-mini, Gemini Flash) or "quality" pool (Sonnet, GPT-5, Gemini Pro) based on the profile's optimization target.

## Override Precedence

1. Per-invocation flag/env (highest)
2. Engagement-level `profiles:` block in scope YAML
3. `~/.zoidberg/profiles.yaml` (operator defaults)
4. Built-in defaults (lowest)

## Cross-Reference Benchmarks

### XBOW Validation Benchmark

XBOW's 104-challenge validation benchmark tests whether an LLM-driven agent can correctly identify and validate real web vulnerabilities. LLM-Eval maps each challenge to our validator chains (`validate-xss.yaml`, `validate-sqli.yaml`, etc.) and measures:

- **True positive rate**: Correctly confirmed real vulnerabilities
- **False positive rate**: Incorrectly confirmed non-vulnerabilities
- **False negative rate**: Missed real vulnerabilities
- **Time to confirm**: Median time from finding to validated result

### AIRTBench

AIRTBench from Dreadnode evaluates AI security decision-making across categories:
- Vulnerability triage (severity classification)
- Attack path reasoning (multi-step exploitation)
- Remediation recommendation (fix quality)
- False positive identification (noise filtering)

LLM-Eval measures F1 score, precision, and recall per category against expert-labeled ground truth.
