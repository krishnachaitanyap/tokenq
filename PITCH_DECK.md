# tokenq

**The cost & context layer for enterprise LLM usage**

---

## Slide 1 — The Problem

Enterprise LLM spend is growing 3–5× YoY, and the largest line items are invisible to finance:

- **Repeated context.** The same tool_results, file reads, and system prompts get re-sent on every turn. A single coding session can re-send the same 50KB file 30 times.
- **Wrong-tier model use.** Teams default to the most capable (and most expensive) model "to be safe," even when a cheaper model would land the same answer 90%+ of the time.
- **No attribution.** Finance gets one Anthropic invoice. They cannot tell you which team, product, or feature drove the spend.
- **No governance.** Prompts containing PII, secrets, or regulated data flow to vendors with no central log, no redaction, no allowlist.

**The result:** LLM spend looks like AWS spend in 2012 — growing fast, opaque, ungoverned, and full of waste that nobody owns.

---

## Slide 2 — The Solution

**tokenq** is a transparent proxy that sits between your applications and Claude. It does four things:

1. **Compresses context** — recalls prior tool_results from a local FTS5 store instead of re-sending them.
2. **Routes models** — uses a Thompson-sampling bandit to pick the cheapest tier likely to succeed for each request, never above the user's chosen ceiling.
3. **Attributes spend** — every request is logged with team, product, model, tokens, and cost.
4. **Enforces governance** — central allowlists, PII redaction, per-team quotas.

Drop-in: applications point at the proxy URL instead of `api.anthropic.com`. No SDK changes.

---

## Slide 3 — Why now

- **Model spend is now a board-level line item.** A single team can burn $100K/month without anyone noticing until the invoice arrives.
- **Compliance is catching up.** GDPR, HIPAA, and emerging AI-specific regulation (EU AI Act, US state laws) require prompt-level audit trails that vendor consoles do not provide.
- **The proxy pattern is the obvious shape.** Just as every cloud team eventually built an internal AWS cost layer (FinOps), every AI-using enterprise will eventually build an LLM cost layer. tokenq is that layer, ready to deploy.
- **Compounding levers.** Each optimization (caching, dedup, routing, batching) stacks. The first integration is the expensive one. After tokenq is in place, every new lever costs days, not months.

---

## Slide 4 — Architecture (one diagram)

```
   ┌──────────────┐    ┌─────────────────────────────────────┐    ┌──────────┐
   │  App / IDE / │───▶│            tokenq proxy             │───▶│ Anthropic│
   │  Agent / SDK │    │                                     │    │   API    │
   └──────────────┘    │  ┌─────────────────────────────┐    │    └──────────┘
                       │  │  Pipeline (ordered stages)  │    │
                       │  │  ─────────────────────────  │    │
                       │  │  cache → dedup → compress   │    │
                       │  │  → compaction → bandit      │    │
                       │  │  → bigmemory recall         │    │
                       │  └─────────────────────────────┘    │
                       │            │           │            │
                       │            ▼           ▼            │
                       │      ┌──────────┐  ┌──────────┐     │
                       │      │ SQLite   │  │ Telemetry│     │
                       │      │ (FTS5)   │  │ Dashboard│     │
                       │      └──────────┘  └──────────┘     │
                       └─────────────────────────────────────┘
```

Single binary, single SQLite file, no external dependencies. Runs on a laptop or scales horizontally behind a load balancer with shared Postgres.

---

## Slide 5 — Strategy 1: bigmemory (FTS5 recall)

**Problem.** A coding agent re-sends the same 50KB tool_result every turn. Over a 30-turn session that's 1.5MB of redundant input tokens.

**Mechanism.**

- Every tool_result above a size threshold is captured into a local SQLite FTS5 index, keyed by content hash (dedup is automatic — same bytes never stored twice).
- On the next turn, the proxy replaces the verbatim tool_result with a short pointer ("see memory item #1710"), and exposes a `memory_search` MCP tool.
- The agent calls `memory_search` with a BM25 query when it needs the content back. Only the relevant fragment is recalled, not the whole blob.

**Why it works.**

- Most tool_results are read once, referenced many times. BM25 retrieval surfaces just the referenced span.
- Hash-based dedup means a hot file read every turn occupies one row, not thirty.
- The `OF content` trigger guard means hit-counter bookkeeping doesn't rewrite the inverted index.

**Typical savings.** 20–40% input token reduction on long agent sessions. Higher on tool-heavy workloads (code review, data analysis, research).

---

## Slide 6 — Strategy 2: Thompson-sampling bandit router

**Problem.** Teams default to Opus when Sonnet or Haiku would land the same answer. Opus is ~5× the cost of Sonnet and ~25× the cost of Haiku.

**Mechanism.**

- Every request is classified into a **context bucket** (short-prompt-no-tools, long-prompt-with-tools, code-task, summarization, etc.).
- For each `(bucket, arm)` cell, the proxy maintains a Beta(α, β) distribution of observed reward.
- On a new request, Thompson-sample from each candidate arm's Beta and pick the highest. Reward = weighted blend of success signal (no error, clean stop_reason) and cost penalty.
- **Hard ceiling**: the bandit never picks an arm above the user's chosen tier. If the user requested Haiku, the bandit is a no-op.

**Safety.** Three modes:

- `shadow` (default) — pick an arm, log the would-be decision, but call the user's chosen model. Builds evidence on day 1 with zero correctness risk.
- `live` — actually route, but only to **graduated** arms (those with enough live observations above a reward floor) plus the user's original tier.
- Demotion fires automatically when a graduated arm's rolling reward drops below floor.

**Typical savings.** 10–25% cost reduction once arms graduate, with sub-1% regression on success metrics.

---

## Slide 7 — Strategy 3: prompt-cache exploitation

**Problem.** Anthropic's prompt cache offers ~90% discount on cached input tokens, but the cache TTL is 5 minutes and applications rarely structure their prompts to hit it.

**Mechanism.**

- The proxy reorders message segments so that stable content (system prompt, tool definitions, long file contents) lands at the front of the prompt, where cache breakpoints are inserted.
- Volatile content (user message, recent turns) goes at the end.
- The proxy tracks per-conversation cache state and warns when the working set is about to exceed the 5-minute TTL — agents can decide to consolidate or accept the miss.

**Typical savings.** 30–60% input token cost reduction for any conversation lasting more than one turn.

---

## Slide 8 — Strategy 4: pipeline stages (dedup, compress, compaction)

**dedup** — within a single request, identical message blocks (same hash) are collapsed to a single block plus references. Catches the "same file pasted twice" pattern.

**compress** — long, repetitive content (logs, JSON arrays, table dumps) is run through structured compression that preserves semantically important fragments and elides repetitive ones.

**compaction** — old turns in a long conversation are summarized into a single compact turn once they fall outside a relevance window. Summarization happens **out-of-band** at lower priority, not on the critical path.

Each stage is independently toggleable, has its own metrics, and can run in shadow mode before going live.

---

## Slide 9 — Strategy 5: skill compression

**Problem.** Tool definitions and system prompts often duplicate instructions across teams. A 10KB system prompt sent on every request adds up.

**Mechanism.** A skill registry stores canonical versions of common skills (code review, summarization, extraction, etc.). The proxy substitutes a short skill reference for the verbatim prompt and expands it on the way to the model.

**Bonus.** Centralizing skills means prompt improvements roll out everywhere at once instead of per-team.

---

## Slide 10 — Strategy stack: how the savings compose

Each lever is multiplicative, not additive — they apply to *what's left* after the previous lever:

| Lever | Typical savings | Cumulative spend remaining |
|---|---|---|
| Baseline | — | 100% |
| + bigmemory (FTS5 recall) | 30% | 70% |
| + prompt-cache exploitation | 40% of remaining | 42% |
| + bandit routing | 15% of remaining | 36% |
| + dedup / compress / compaction | 10% of remaining | 32% |

**End state: ~65–70% reduction in LLM spend** for tool-heavy agentic workloads. Lighter savings (15–25%) for short, single-turn workloads.

The 3.5% headline number is what you get from any *one* lever in isolation. The whole point of building the proxy is that you get to stack them.

---

## Slide 11 — What you get beyond cost savings

- **Per-team / per-product cost attribution** — finance can finally answer "who is spending what on AI."
- **Centralized prompt audit log** — every prompt and response, searchable, with PII redaction at ingest.
- **Model allowlists & quotas** — enforce policy without trusting every team to self-police.
- **Vendor portability** — swap Anthropic for OpenAI, Bedrock, or self-hosted by changing one config. Critical leverage at contract renewal.
- **A/B experimentation** — run prompt or model experiments on real production traffic without app code changes.
- **SLO observability** — p50/p99 latency, error rate, stop-reason distribution, cost per successful task.

---

## Slide 12 — Deployment model

- **Phase 1 (week 1–2):** deploy in shadow mode behind one team's traffic. Zero behavior change. Collect baseline.
- **Phase 2 (week 3–6):** turn on bigmemory and prompt caching. Measure savings against baseline.
- **Phase 3 (week 7–12):** turn on bandit in shadow, then live for graduated cells. Roll out to additional teams.
- **Phase 4 (quarter 2):** governance features (allowlists, quotas, redaction) become hard-required.

Each phase is independently reversible. No big-bang migration.

---

## Slide 13 — Risks & mitigations

| Risk | Mitigation |
|---|---|
| Proxy becomes a single point of failure | HA deployment + circuit breaker that falls back to direct vendor calls |
| Bandit routes to a worse model and degrades quality | Shadow-first, graduation gates, automatic demotion, hard ceiling at user's chosen tier |
| Latency overhead | Pipeline stages run in <5ms p99 in production; bigmemory recall is a local SQLite query |
| Compliance review delays rollout | Start with a single low-risk team; security-review the proxy in parallel with shadow-mode data collection |
| Model updates break routing | Cost tables and feature detection are config-driven, not code; updates take hours, not days |

---

## Slide 14 — Build vs. buy

**Buy** (vendor LLM gateways) when:
- Spend is under $1M/yr.
- No regulated-data requirements.
- No platform team to own shared infra.

**Build internally with tokenq** when:
- Spend > $1M/yr and growing >50% YoY.
- Compliance requires prompt-level audit trails.
- You already have platform infrastructure (auth, observability, secrets).
- You want vendor portability as strategic leverage.

tokenq is open-source-able, no vendor lock-in, runs in your VPC, your data never leaves your perimeter.

---

## Slide 15 — The ask

- **Funding.** 1.5 FTE for 2 quarters to harden, deploy, and onboard the first 3 teams.
- **Sponsorship.** A platform-team owner who will run the service post-launch.
- **First-team partnership.** One team willing to run in shadow mode for 4 weeks to produce the baseline numbers.

**12-month target.** 30%+ reduction in company-wide LLM spend, 100% of LLM traffic flowing through the proxy, full per-team attribution available to finance, and a governance layer that satisfies the compliance team's audit requirements.

The 3.5% number is the floor. The strategic value is the ceiling.
