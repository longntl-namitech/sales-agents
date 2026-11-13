"""Seeded synthetic lead generator (Q13).

Attributes are sampled to hit a target label, then the gold label is computed
from the rubric. An injected `write_text` writes the free-text notes, so tests
use a fake and a live run can use an LLM. Labels never depend on the notes.

Gold labels come from the same rubric the Qualifier is told to apply, so
accuracy on this set is optimistic. State that in the README limits.
"""

import json
import random
from collections.abc import Callable
from pathlib import Path

from sales_agents.rubric import (
    BUYER_ROLES,
    INTENT_SIGNALS,
    NEED_SIGNALS,
    TARGET_INDUSTRIES,
    TARGET_REGIONS,
    label_lead,
)
from sales_agents.schemas import LABELS, CompanyRecord, LabelledLead, Lead

AMBIGUOUS_SHARE = 0.2

BUYER_TITLES = [
    "VP Sales",
    "Head of Sales",
    "Director of Sales Operations",
    "Chief Revenue Officer",
    "Head of Revenue Operations",
]
NON_BUYER_TITLES = ["Software Engineer", "Recruiter", "Office Manager", "Marketing Intern", "Data Analyst"]
OFF_TARGET_INDUSTRIES = ["mining", "agriculture", "education"]
OFF_TARGET_REGIONS = ["asia pacific", "latin america"]
NAME_PREFIXES = ["Nimbus", "Harbor", "Lumen", "Vertex", "Cobalt", "Juniper", "Atlas", "Ember", "Quill", "Sable", "Tidal", "Orbit"]
NAME_SUFFIXES = ["Labs", "Systems", "Works", "Cloud", "Logic", "Group", "Partners", "Dynamics", "Networks", "Analytics", "Robotics", "Health"]

assert {t.lower() for t in BUYER_TITLES} == BUYER_ROLES


def template_text(attributes: dict) -> str:
    """Offline stand-in for the LLM writer. Mentions only the attributes that are known."""
    parts = []
    if attributes["source"] == "inbound":
        parts.append(f"Contact form from {attributes['company_name']}.")
    else:
        parts.append(f"Target account {attributes['company_name']}.")
    if attributes["employee_count"] is not None:
        parts.append(f"About {attributes['employee_count']} employees.")
    if attributes["industry"] is not None:
        parts.append(f"Industry: {attributes['industry']}.")
    if attributes["region"] is not None:
        parts.append(f"Region: {attributes['region']}.")
    if attributes["contact_role"] is not None:
        parts.append(f"Contact is {attributes['contact_role']}.")
    if attributes["signals"]:
        parts.append("Seen: " + ", ".join(s.replace("_", " ") for s in attributes["signals"]) + ".")
    return " ".join(parts)


def _signals(rng: random.Random, need: int, intent: int) -> list[str]:
    return rng.sample(sorted(NEED_SIGNALS), need) + rng.sample(sorted(INTENT_SIGNALS), intent)


def _good_gates(rng: random.Random) -> dict:
    return {
        "employee_count": rng.randint(60, 900),
        "industry": rng.choice(sorted(TARGET_INDUSTRIES)),
        "region": rng.choice(sorted(TARGET_REGIONS)),
        "contact_role": rng.choice(BUYER_TITLES),
    }


def _attributes_for(rng: random.Random, target: str, ambiguous: bool) -> dict:
    attrs = _good_gates(rng)
    if target == "hot":
        need, intent = rng.choice([(2, 1), (1, 2)]) if ambiguous else (2, 2)
        attrs["signals"] = _signals(rng, need, intent)
    elif target == "warm":
        need, intent = rng.choice([(1, 1), (2, 0), (0, 2)] if ambiguous else [(1, 0), (0, 1)])
        attrs["signals"] = _signals(rng, need, intent)
    elif ambiguous:
        # Just outside the size gate with strong signals: close to qualified, still cold.
        attrs["employee_count"] = rng.choice([49, 1001])
        attrs["signals"] = _signals(rng, 2, 2)
    else:
        mode = rng.choice(
            ["size", "industry", "region", "role", "missing_size", "missing_industry", "missing_region", "missing_role", "no_signals"]
        )
        attrs["signals"] = [] if mode == "no_signals" else _signals(rng, rng.randint(0, 2), rng.randint(0, 2))
        if mode == "size":
            attrs["employee_count"] = rng.choice([rng.randint(1, 40), rng.randint(1500, 20000)])
        elif mode == "industry":
            attrs["industry"] = rng.choice(OFF_TARGET_INDUSTRIES)
        elif mode == "region":
            attrs["region"] = rng.choice(OFF_TARGET_REGIONS)
        elif mode == "role":
            attrs["contact_role"] = rng.choice(NON_BUYER_TITLES)
        elif mode.startswith("missing_"):
            field = {"size": "employee_count", "industry": "industry", "region": "region", "role": "contact_role"}[
                mode.removeprefix("missing_")
            ]
            attrs[field] = None
    return attrs


def generate_leads(n: int, seed: int, write_text: Callable[[dict], str]) -> list[LabelledLead]:
    rng = random.Random(seed)
    targets = [LABELS[i % len(LABELS)] for i in range(n)]
    ambiguous_flags = [True] * round(n * AMBIGUOUS_SHARE) + [False] * (n - round(n * AMBIGUOUS_SHARE))
    rng.shuffle(targets)
    rng.shuffle(ambiguous_flags)

    cases: list[LabelledLead] = []
    used_names: set[str] = set()
    for i, (target, ambiguous) in enumerate(zip(targets, ambiguous_flags)):
        attrs = _attributes_for(rng, target, ambiguous)
        name = f"{rng.choice(NAME_PREFIXES)} {rng.choice(NAME_SUFFIXES)}"
        if name in used_names:
            name = f"{name} {i}"
        used_names.add(name)
        source = rng.choice(["outbound", "inbound"])

        notes = write_text({**attrs, "company_name": name, "source": source})
        lead = Lead(
            lead_id=f"lead-{i:04d}",
            company_name=name,
            contact_role=attrs["contact_role"],
            notes=notes,
            source=source,
        )
        company = CompanyRecord(
            company_name=name,
            employee_count=attrs["employee_count"],
            industry=attrs["industry"],
            region=attrs["region"],
            signals=attrs["signals"],
        )
        gold = label_lead(lead, company)
        if gold != target:
            raise RuntimeError(f"generator bug: wanted {target}, rubric says {gold} for {attrs}")
        cases.append(LabelledLead(lead=lead, company=company, gold=gold, ambiguous=ambiguous))
    return cases


def save_leads(cases: list[LabelledLead], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(c.model_dump_json() + "\n" for c in cases))


def load_leads(path: Path) -> list[LabelledLead]:
    return [LabelledLead.model_validate(json.loads(line)) for line in path.read_text().splitlines() if line]


if __name__ == "__main__":
    out = Path("datasets/leads_seed42.jsonl")
    save_leads(generate_leads(60, seed=42, write_text=template_text), out)
    print(f"wrote {out}")
