# sales-agents

A multi-agent sales system for one fictional B2B SaaS seller. It researches leads, qualifies them, drafts outreach, and records the outcome in a CRM. A guardrail checks every draft, and nothing counts as sent without a human approving it. Everything runs on synthetic data and is measured against gold labels.

This is a portfolio project. It does not send email, touch real leads or use real client data.

## Status

| Part | State |
|---|---|
| Schemas, qualification rubric, seeded lead generator, eval harness, token guard, rule baseline | Done |
| Qualifier, Researcher | Done, measured |
| Supervisor, Writer, guardrail, scripted approver, CRM agent, mock CRM over MCP | Done, measured |
| Reply handling, inbound routing, reactivation, RFP bid or no-bid | Done as typed pipeline functions, measured. Not wired through the Supervisor |
| Hand review of draft verdicts | Tooling done. No human labels yet, so the agreement is not measured |
| Langfuse tracing | Code only, as an optional extra. Never checked against a Langfuse server |

## Results

Every row comes from a run on 2026-11-13, recorded in [results/measurements.json](results/measurements.json). Rows that call a model used `openai:gpt-4.1-mini`. The table below is the output of `uv run python -m sales_agents.eval.report`, with the notes shortened.

| Use case | Agent | n | Metrics | Tokens | p50 s/case | Note |
|---|---|---|---|---|---|---|
| Lead qualification | Rule baseline | 60 | precision (hot) 1.000, recall (hot) 1.000 | 0 | 0.00 | ceiling, same rubric as the gold labels |
| Lead qualification | Qualifier (LLM) | 60 | precision (hot) 0.800, recall (hot) 1.000 | 26,982 | 0.94 | no enrichment |
| Researcher | Qualifier + Researcher | 60 | precision (hot) 0.952, recall (hot) 1.000 | 28,187 | 0.84 | enrichment lifts hot precision by +0.152 |
| Supervisor | Supervisor (agents as tools) | 60 | precision (hot) 0.952, recall (hot) 1.000 | 303,470 | 3.07 | research then qualify through the Deep Agent |
| Outbound outreach | Writer + guardrail + approver | 20 | guardrail pass rate 0.700, approver acceptance rate 0.700 | 9,742 | n/a | drafts for gold-hot leads, not yet hand-reviewed |
| Outbound outreach | Outbound path (Supervisor, MCP CRM, scripted approver) | 12 | CRM holds exactly the approved drafts 1.000, label accuracy in CRM 1.000 | 121,191 | 6.61 | 2 of 4 drafts approved |
| Reply handling | Reply handling | 40 | classification accuracy 1.000, correct CRM update rate 1.000, accuracy on unclear and injected 1.000 | 25,143 | 1.96 | classify, update the CRM, draft, scripted approver |
| Inbound routing | Inbound routing | 24 | correct-rep rate 1.000 | 11,313 | 0.93 | latency is form to decision |
| Lead reactivation | Reactivation selector (LLM) | 40 | precision 1.000, recall 1.000, guardrail pass rate on drafts 0.333 | 6,128 | 0.79 | code gates on label and quiet days, model reads the history |
| Lead reactivation | Reactivation baseline | 40 | precision 0.429, recall 1.000 | 0 | n/a | label and days only, ignores the history |
| RFP bid or no-bid | RFP bid or no-bid | 40 | precision (bid) 1.000, recall (bid) 1.000, accuracy 1.000 | 33,062 | 1.55 | model extracts facts, thresholds in code decide |

Read these numbers with care. The 1.000 rows are not a quality claim. The gold labels and the text of every set are written by the same person and the same templates the agents' prompts were built from, so the sets are easy and the scores are optimistic. The sets are small, and each number is one run. The rule baseline is the ceiling for the same reason.

What the runs did show:

- **Enrichment helps the Qualifier.** Hot precision went from 0.800 to 0.952 on the same 60 leads, recall stayed at 1.000, and the cost was about 1,200 more tokens. The lead notes already state the same facts, so the gain is not new information. Where it comes from was not isolated.
- **Going through the Supervisor costs 10.7 times the tokens and 3.4 times the time**, for the same accuracy as calling the Qualifier directly (303,470 tokens against 28,305 and 3.07 s against 0.89 s on the same run). The Deep Agents harness adds its own prompt and tool schemas to every model call.
- **The first Supervisor design lost accuracy.** It delegated to LLM subagents, and the Supervisor's own written task description dropped the company facts, so the Qualifier answered "size, industry or region unknown". Hot precision and recall were both 0.650 over 666,906 tokens. Passing data through code instead of through the Supervisor's words, with agents as tools that take only a lead id, brought a first fix to 0.833, and making `qualify_lead` research on demand brought it to 0.952. That first design is not in the repository, so the 0.650 run cannot be reproduced.
- **The guardrail is conservative and over-blocks.** On planted faults it catches every one (`uv run python -m sales_agents.eval.outreach`: catch rate 1.000, false-block rate 0.000), but those faults were written by the same person as the detectors. On drafts a real model wrote, 14 of 20 passed. Of the 6 blocked, 4 flagged a number that matches the numeric suffix this dataset adds to company names (such as "Quill Dynamics 12"), which was inferred from the numbers and not checked in the drafts, and 3 flagged paraphrased product claims. One draft had both. Reactivation drafts passed 4 of 12. In a separate sample of 8 (not saved), 4 were blocked: 3 for paraphrased claims and 1 for a number. A human reviewer may approve many of these. That is what the hand-review sample is for.

## What was not measured

- Any real lead, reply, RFP or CRM. All data is synthetic and the CRM is a mock.
- Agreement between the scripted approver and a human. [datasets/hand_review_sample.jsonl](datasets/hand_review_sample.jsonl) holds 20 real drafts with the checklist's verdict and a blank `human` field. Fill it in, then run `uv run python -m sales_agents.eval.hand_review`.
- Robustness to prompt injection beyond the injected cases in the reply, reactivation and RFP sets. The lead set has no injected case.
- A Langfuse trace, a different model, many seeds, or variance between runs.
- Whether the guardrail's word-overlap check holds up on a larger product corpus.

## What it does

The hero use case is lead qualification. The other use cases reuse the same agents. Each one has its own diagram, as a PNG with an interactive HTML version next to it (download the HTML and open it in a browser).

1. **Lead qualification:** decide which leads deserve a rep's time, as hot, warm or cold with a reason. [Diagram](docs/diagrams/1-qualification.png), [interactive](docs/diagrams/1-qualification.html).
2. **Outbound outreach:** draft a grounded first message for a hot lead, get approval, check it, record it. A warm or cold lead is saved to the CRM with its label and reason, and no draft is written. [Diagram](docs/diagrams/2-outbound-outreach.png), [interactive](docs/diagrams/2-outbound-outreach.html).
3. **Reply handling:** classify a prospect's reply, update the CRM, and draft a response unless the prospect declined. [Diagram](docs/diagrams/3-reply-handling.png), [interactive](docs/diagrams/3-reply-handling.html).
4. **Inbound routing:** qualify a form or demo request and route it to a rep by region and industry. [Diagram](docs/diagrams/4-inbound-routing.png), [interactive](docs/diagrams/4-inbound-routing.html).
5. **Lead reactivation:** pick quiet leads worth another touch and draft a message. [Diagram](docs/diagrams/5-reactivation.png), [interactive](docs/diagrams/5-reactivation.html).
6. **RFP bid or no-bid:** a recommendation on whether to respond to an RFP, decision only. [Diagram](docs/diagrams/6-rfp-bid-or-no-bid.png), [interactive](docs/diagrams/6-rfp-bid-or-no-bid.html).

Use cases 1 and 2 run through the Supervisor. Use cases 3 to 6 are typed pipeline functions that follow the diagram's steps and are measured on their own. The Supervisor is not wired to them yet, so those four diagrams show the target design. The diagram sources are the `*.workflow.json` files next to them.

## Architecture

![Outbound path through the agents](docs/architecture.png)

[Open the interactive diagram](docs/architecture.html) (download it and open it in a browser). It supports pan, zoom, search and light or dark theme. The source is [docs/architecture.workflow.json](docs/architecture.workflow.json).

The Supervisor sits in the middle. It calls the Researcher, Qualifier and Writer as tools that take only a lead id, and the CRM tools, and gets each result back, so the agents never call each other. The guardrail and the approval gate are not agents. They sit in front of the CRM write. `record_outreach` pauses for a human decision first, then the guardrail checks the text that was approved (an edited draft is checked too), and only then does the mock CRM record it.

Design rules:

- **Agents are plain typed functions.** Each takes a typed input and returns a typed output, so it can be evaluated alone against gold labels.
- **The Supervisor is the Deep Agent.** It is built with `create_deep_agent` from [LangChain Deep Agents](https://github.com/langchain-ai/deepagents). Deep Agents runs on LangGraph, which supplies checkpointing and human-in-the-loop. The Supervisor reaches the agents through tools and not through LLM subagents, because passing data through a model's own words lost the company facts (see the results).
- **Data moves through code, not through the Supervisor's words.** The agents share a per-run `LeadDesk`. The lead's free-text notes never reach the Supervisor model, and the label comes from the `qualify_lead` result.
- **Safety rules live in code, not in prompts.** The approval gate is `interrupt_on` and the guardrail is inside the `record_outreach` tool. A rejected draft is never recorded, an edited draft records the edited text, and a draft the guardrail blocks is never recorded even if it was approved.
- **Measure before wiring.** Each agent gets a number on its own before the Supervisor connects it.

## Agents

### Supervisor

Receives a lead id and calls the other agents in order. It owns no business rules. With the outbound tools it saves the label and, for a hot lead, asks for a draft and records it. Implemented in [src/sales_agents/agents/supervisor.py](src/sales_agents/agents/supervisor.py).

### Researcher

Enriches a lead with company facts: size, industry, region and public signals. It is a lookup in a synthetic store and makes no model call, so it cannot invent a fact. An unknown company returns nothing. Its value is the change in Qualifier accuracy, shown above.

### Qualifier

Takes a `Lead` and an optional `CompanyRecord` and returns a `QualificationResult` with a label (hot, warm or cold) and a short reason. It makes one structured-output model call and uses no tools. The prompt is built from the rubric constants, and the lead's free-text notes are wrapped in a delimited block that the prompt marks as untrusted data. Implemented in [src/sales_agents/agents/qualifier.py](src/sales_agents/agents/qualifier.py).

The rubric:

- **Fit gate:** company size 50 to 1000, a target industry, a target region.
- **Buyer role gate:** the contact holds a buyer role such as VP Sales or Head of Revenue Operations.
- **Need and Intent:** scored from signals, at most 2 points each. A total of 3 or more is hot, 1 or 2 is warm, 0 is cold.
- **Missing data:** a missing gate field makes the lead cold, with the missing field named in the reason.

The thresholds are the builder's choices, set in [src/sales_agents/rubric.py](src/sales_agents/rubric.py).

### Writer

Drafts a first outreach email, a reply to a prospect, or a re-engagement email. It sees the company facts and passages from the fictional product docs, found by a keyword search, and never the lead's free-text notes. Its prompt allows only facts from the docs. Implemented in [src/sales_agents/agents/writer.py](src/sales_agents/agents/writer.py).

### Guardrail

Checks each draft in code: an opt-out line is present, no prohibited promise such as "guarantee", every number appears in the docs, and every product claim shares most of its words with the docs. The claim check is a word-overlap heuristic and not understanding, and it over-blocks paraphrases. Implemented in [src/sales_agents/guardrail.py](src/sales_agents/guardrail.py).

### Human approval

Nothing is recorded until a human approves, edits or rejects, even when the guardrail would pass. In evals the approver is a scripted checklist: the guardrail passes, the body is at most 150 words, the subject at most 80 characters, at most one exclamation mark, and not many words in capitals. There is no LLM judge. The terminal asks you instead when you run the command line below.

### CRM agent

The mock CRM is a FastAPI service over sqlite, exposed through an MCP server. The tools are `upsert_lead` and `record_outreach`. Saving the same lead again updates its record instead of creating a duplicate.

### Reply classifier, router, selector, RFP extractor

- **Reply handling** classifies a reply as interested, not interested, question or reschedule. An unclear reply goes to question so a person follows up. The CRM status follows the classification, never the reply's words.
- **Routing** is plain code over the label, region and industry. Form text cannot choose its own rep.
- **Reactivation** keeps two code gates (not cold, quiet for at least 60 days) and asks the model only to read the history of earlier contact.
- **RFP** has the model extract the requirements, whether the docs support each, the deadline and the budget. Code then decides: a missing deadline or budget, a deadline under 14 days, a budget under $20,000, or fewer than 70% of requirements supported is a no-bid. These thresholds are the builder's, not from the spec.

## Evaluation

- **Headline metric for qualification:** precision and recall of the hot label, plus the 3x3 confusion matrix, cost and latency per lead. Each other use case has its own metric, in the table above.
- **Failures count as wrong.** A timeout or invalid output is a wrong prediction and stays in the denominator.
- **Token guard.** Each run estimates its tokens first and refuses to start when the estimate is above the cap. The cap is 2,000,000 tokens per run, set in [src/sales_agents/config.py](src/sales_agents/config.py), and live tests also stay under 1,000,000. The Supervisor's measured overhead is part of the estimate.
- **Synthetic sets.** All are seeded and written from templates, with gold labels derived from the attributes behind the text.

| Set | n | How the gold labels are made |
|---|---|---|
| Leads | 60 | The rubric applied to the attributes. 20 hot, 20 warm, 20 cold, 12 ambiguous, some with a missing gate field |
| Planted-fault drafts | 64 | The planted fault. 24 clean and 40 with one fault each |
| Replies | 40 | The category each phrase belongs to. 8 per category, 4 unclear, 4 with an injected instruction |
| Inbound routing | 24 | The inbound leads of the lead set, routed by the same rules with the gold label |
| Aged CRM records | 40 | Hidden attributes: last label, quiet days and the outcome of earlier contact. 12 worthwhile |
| RFPs | 40 | The builder's rule applied to the true facts. 12 bid, with missing budgets, missing deadlines and injected instructions |

- **Known limit.** The labels and the agents share their rules, so accuracy can look better than it would on real data.

## Getting started

Requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env          # then set OPENAI_API_KEY

uv run pytest                                  # offline, no tokens
uv run python -m sales_agents outbound lead-0010            # run one lead, you approve the draft
uv run python -m sales_agents outbound lead-0010 --scripted # the checklist approves instead
uv run python -m sales_agents.eval.report                   # the results table
```

The command line researches and qualifies the lead, saves its label to the mock CRM over MCP, and for a hot lead writes a draft and asks you to approve, edit or reject it. Nothing is sent. The mock CRM state is in `crm.sqlite3`.

To measure again with the real model, which calls it and costs tokens (about 630,000 in all):

```bash
uv run pytest -m llm -s tests/agents
```

Each live test estimates its tokens first and refuses to run above 1,000,000. Tests that call a real model carry the `llm` marker and are excluded from the default run. Each one rewrites its row in `results/measurements.json`.

## Layout

```
src/sales_agents/
  config.py        settings and the optional tracing callback
  schemas.py       Lead, CompanyRecord, Draft, Reply, CrmRecord, Rfp, ...
  rubric.py        deterministic labeller
  guardrail.py     draft checks
  approver.py      scripted checklist approver
  cli.py           the command line
  datagen.py, draftgen.py, replygen.py, reactivationgen.py, rfpgen.py   seeded synthetic sets
  agents/          qualifier, researcher, supervisor, writer, replies, routing, reactivation, rfp, baseline
  tools/           enrichment (company store), kb (product docs search), crm (MCP and guardrail wrapper)
  crm_mock/        store, FastAPI app, MCP server
  eval/            metrics, harness, runner, tokens, live, outreach, hand_review, report
datasets/          committed gold set, product docs, hand-review sample
results/           measurements.json, the source of the table
docs/              diagrams
tests/             offline tests, plus llm-marked tests
```
