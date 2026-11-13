"""The guardrail: checks a draft against the product docs, in code.

Blocks a draft that lacks an opt-out line, makes a prohibited promise, states a
number the docs do not contain, or makes a product claim whose words are mostly
absent from the docs. It runs inside the CRM tool, never in a prompt, so a model
cannot talk its way past it.

The claim check is a heuristic on word overlap, not understanding. It is
conservative on purpose: a paraphrase can be blocked, and the numbers in the
README measure how often.
"""

import re

from sales_agents.schemas import Draft, GuardrailResult
from sales_agents.tools.kb import words

OPT_OUT_PATTERNS = [
    r"\bunsubscribe\b",
    r"\bopt[- ]?out\b",
    r"\bstop (emailing|contacting|sending|writing)\b",
    r"\breply\W{0,3}stop\b",
    r"\b(rather|prefer) not (to )?(hear|receive|get|be contacted)\b",
    r"\bno longer wish to (receive|hear|be contacted)\b",
    r"\bdon.?t want (to )?(hear|receive|get)\b",
]
PROHIBITED_PATTERNS = [
    r"\bguarantee[sd]?\b",
    r"\b100\s?%",
    r"\brisk[- ]free\b",
    r"\bact now\b",
    r"\blimited[- ]time\b",
]
NUMBER = re.compile(r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?")
PRODUCT_TERMS = re.compile(r"\b(pipelane|we|our|it|platform|tool|product|software)\b", re.IGNORECASE)
MIN_CLAIM_COVERAGE = 0.6  # share of a claim's content words that must appear in the docs


def _sentences(text: str) -> list[str]:
    parts = []
    for line in text.replace("’", "'").splitlines():
        parts += re.split(r"(?<=[.!?])\s+", line.strip())
    return [p for p in parts if p]


def check_draft(draft: Draft, docs: list[str]) -> GuardrailResult:
    text = draft.text
    violations: list[str] = []

    if not any(re.search(p, text, re.IGNORECASE) for p in OPT_OUT_PATTERNS):
        violations.append("missing opt-out line")

    for pattern in PROHIBITED_PATTERNS:
        if match := re.search(pattern, text, re.IGNORECASE):
            violations.append(f"prohibited phrase '{match.group(0)}'")

    docs_text = " ".join(docs)
    known_numbers = set(NUMBER.findall(docs_text))
    for number in dict.fromkeys(NUMBER.findall(text)):
        if number not in known_numbers:
            violations.append(f"number {number} not supported by product docs")

    docs_vocabulary = set(words(docs_text))
    for sentence in _sentences(text):
        if sentence.endswith("?") or not PRODUCT_TERMS.search(sentence):
            continue
        content = [w for w in words(sentence) if len(w) >= 3 and w != "pipelane" and not w.isdigit()]
        if not content:
            continue
        coverage = sum(w in docs_vocabulary for w in content) / len(content)
        if coverage < MIN_CLAIM_COVERAGE:
            violations.append(f"claim not supported by product docs: '{sentence}'")

    return GuardrailResult(passed=not violations, violations=violations)
