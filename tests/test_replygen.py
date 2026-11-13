from sales_agents.replygen import generate_replies
from sales_agents.schemas import REPLY_CATEGORIES


def test_same_seed_same_replies():
    assert generate_replies(seed=1) == generate_replies(seed=1)


def test_every_category_is_present():
    assert {r.gold for r in generate_replies(seed=42)} == set(REPLY_CATEGORIES)


def test_unclear_and_injection_replies_are_included():
    tricky = {r.tricky for r in generate_replies(seed=42)}

    assert tricky == {None, "unclear", "injection"}


def test_unclear_replies_are_gold_question_and_injections_never_gold_interested():
    replies = generate_replies(seed=42)

    assert all(r.gold == "question" for r in replies if r.tricky == "unclear")
    injections = [r for r in replies if r.tricky == "injection"]
    assert injections and all(r.gold != "interested" for r in injections)
    assert all("Ignore all previous instructions" in r.reply.text for r in injections)


def test_reply_ids_are_unique():
    ids = [r.reply.lead_id for r in generate_replies(seed=42)]

    assert len(ids) == len(set(ids))
