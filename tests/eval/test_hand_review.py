import json

from sales_agents.eval.hand_review import agreement, write_sample
from sales_agents.schemas import Draft
from sales_agents.tools.kb import load_docs

DOCS = load_docs()
GOOD = Draft(
    subject="Hello",
    body='Hi Dana,\n\nPipelane routes each qualified lead to a rep by territory and segment rules.\n\nReply "unsubscribe" and I will stop.',
)
BAD = Draft(subject="Hello", body="Hi Dana,\n\nPipelane guarantees 40% more meetings.")


def test_sample_records_the_checklist_verdict_and_leaves_the_human_blank(tmp_path):
    path = tmp_path / "sample.jsonl"

    write_sample([("a", GOOD), ("b", BAD)], DOCS, path)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["checklist"] for r in rows] == ["approve", "reject"]
    assert all(r["human"] is None for r in rows)


def test_agreement_is_none_until_a_human_labels_something(tmp_path):
    path = tmp_path / "sample.jsonl"
    write_sample([("a", GOOD)], DOCS, path)

    assert agreement(path) is None


def test_agreement_counts_only_labelled_rows(tmp_path):
    path = tmp_path / "sample.jsonl"
    write_sample([("a", GOOD), ("b", BAD), ("c", GOOD)], DOCS, path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["human"] = "approve"
    rows[1]["human"] = "approve"  # the human disagrees with the checklist here
    path.write_text("\n".join(json.dumps(r) for r in rows))

    assert agreement(path) == (1, 2)
