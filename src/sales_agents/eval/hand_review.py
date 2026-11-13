"""Hand-review of draft verdicts (Q14).

`write_sample` saves drafts with the scripted checklist's verdict and a blank
`human` field. The owner fills `human` with "approve" or "reject". `agreement`
then counts how often the checklist and the human match. Nothing is claimed
until the owner has labelled rows.

    uv run python -m sales_agents.eval.hand_review
"""

import json
from pathlib import Path

from sales_agents.approver import scripted_approve
from sales_agents.guardrail import check_draft
from sales_agents.schemas import Draft

SAMPLE_PATH = Path(__file__).parents[3] / "datasets" / "hand_review_sample.jsonl"


def write_sample(items: list[tuple[str, Draft]], docs: list[str], path: Path = SAMPLE_PATH) -> None:
    rows = [
        {
            "id": item_id,
            "subject": draft.subject,
            "body": draft.body,
            "checklist": scripted_approve(draft, docs),
            "violations": check_draft(draft, docs).violations,
            "human": None,
        }
        for item_id, draft in items
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def agreement(path: Path = SAMPLE_PATH) -> tuple[int, int] | None:
    """(rows where human and checklist match, rows the human labelled), or None if none are labelled."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    labelled = [r for r in rows if r["human"] is not None]
    if not labelled:
        return None
    return sum(r["human"] == r["checklist"] for r in labelled), len(labelled)


if __name__ == "__main__":
    result = agreement()
    print("no rows labelled yet" if result is None else f"checklist matches the human on {result[0]} of {result[1]} labelled drafts")
