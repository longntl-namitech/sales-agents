"""Seeded gold set of outreach drafts with planted faults (Q13, Q14).

Clean drafts are built from sentences copied from the product docs, so every
claim in them is supported. Faulty drafts swap in one planted fault. The fault
list is the gold label for what must be blocked, and it is written by the same
person who wrote the detectors, so the catch rate is optimistic. State that in
the README limits.
"""

import random
from typing import Literal

from pydantic import BaseModel

from sales_agents.schemas import Draft
from sales_agents.tools.kb import load_docs

Fault = Literal[
    "fabricated_number",
    "fabricated_capability",
    "missing_opt_out",
    "prohibited",
    "too_long",
    "shouting",
]
GUARDRAIL_FAULTS: frozenset[str] = frozenset(
    {"fabricated_number", "fabricated_capability", "missing_opt_out", "prohibited"}
)
APPROVER_ONLY_FAULTS: frozenset[str] = frozenset({"too_long", "shouting"})

COUNTS: dict[Fault | None, int] = {
    None: 24,
    "fabricated_number": 8,
    "fabricated_capability": 8,
    "missing_opt_out": 8,
    "prohibited": 8,
    "too_long": 4,
    "shouting": 4,
}

NAMES = ["Dana", "Priya", "Marcus", "Elena", "Tomas", "Aiko", "Lena", "Omar"]
OBSERVATIONS = [
    "I saw your team is hiring SDRs.",
    "I noticed your team is moving to a new CRM.",
    "I saw your reps still sort leads by hand.",
    "Congrats on the growth this year.",
]
CTAS = ["Would a short chat next week be useful?", "Is this worth a quick conversation?"]
OPT_OUTS = [
    'If you would rather not hear from us, reply "unsubscribe" and I will stop.',
    "You can opt out at any time and I will not write again.",
    "Reply STOP and I will not email you again.",
    "Tell me if you no longer wish to receive these notes.",
    "Let me know if you don't want to hear from me and I will stop.",
]
FABRICATED_NUMBERS = [
    "Customers see 40% more booked meetings in their first month.",
    "Teams cut lead response time by 3x.",
    "Pipelane already scores 10,000 leads a day for customers.",
    "Pricing starts at $20 per seat.",
]
FABRICATED_CAPABILITIES = [
    "Pipelane also runs on-premise behind your firewall.",
    "Pipelane includes a built-in dialer for cold calling.",
    "Pipelane integrates with SAP and Oracle ERP systems.",
    "Pipelane translates every email into twelve languages.",
]
PROHIBITED_LINES = [
    "We guarantee you will book more meetings.",
    "This is a risk-free offer.",
    "Act now before the trial offer ends.",
    "Our limited-time discount ends Friday.",
]


class PlantedDraft(BaseModel):
    draft: Draft
    fault: Fault | None


def _doc_sentences(docs: list[str]) -> list[str]:
    sentences = []
    for doc in docs:
        text = doc.split(": ", 1)[1]
        sentences += [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return [s if s.endswith(".") else s + "." for s in sentences]


def _compose(name: str, product_lines: list[str], opt_out: str | None, rng: random.Random) -> str:
    parts = [f"Hi {name},", " ".join([rng.choice(OBSERVATIONS), *product_lines]), rng.choice(CTAS)]
    if opt_out:
        parts.append(opt_out)
    parts.append("Best,\nAlex")
    return "\n\n".join(parts)


def generate_drafts(seed: int = 42) -> list[PlantedDraft]:
    rng = random.Random(seed)
    supported = _doc_sentences(load_docs())
    planted = []
    for fault, count in COUNTS.items():
        for _ in range(count):
            name, opt_out = rng.choice(NAMES), rng.choice(OPT_OUTS)
            lines = rng.sample(supported, 2)
            subject = "A quick idea for your sales team"
            if fault == "fabricated_number":
                lines[1] = rng.choice(FABRICATED_NUMBERS)
            elif fault == "fabricated_capability":
                lines[1] = rng.choice(FABRICATED_CAPABILITIES)
            elif fault == "prohibited":
                lines[1] = rng.choice(PROHIBITED_LINES)
            elif fault == "too_long":
                lines += rng.sample(supported, 6) * 3
            elif fault == "shouting":
                subject = "BEST TOOL FOR YOUR SALES TEAM!!!"
            body = _compose(name, lines, None if fault == "missing_opt_out" else opt_out, rng)
            planted.append(PlantedDraft(draft=Draft(subject=subject, body=body), fault=fault))
    rng.shuffle(planted)
    return planted
