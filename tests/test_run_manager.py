import asyncio

import pytest
from test_cli import DATASET, GOOD, LEAD, models

from sales_agents.crm_mock.store import CrmStore
from sales_agents.runner import RunManager
from sales_agents.tools.crm import store_tools


def manager():
    return RunManager(dataset=DATASET, raw_tools=store_tools(CrmStore()), models=models())


async def collect(stream):
    return [event async for event in stream]


async def _start(m):
    return await collect(await m.start(LEAD.lead_id))


def started_run(m):
    events = asyncio.run(_start(m))
    return events[0]["run_id"], events


def test_start_yields_started_then_steps_then_one_approval_with_the_draft():
    _, events = started_run(manager())

    assert events[0]["type"] == "started" and events[0]["lead_id"] == LEAD.lead_id
    assert [e["agent"] for e in events if e["type"] == "step"][:3] == ["researcher", "qualifier", "crm_agent"]
    assert sum(e["type"] == "approval" for e in events) == 1
    assert events[-1]["type"] == "approval" and events[-1]["draft_text"] == GOOD


def test_approve_ends_with_a_result_record_with_outreach():
    m = manager()
    run_id, _ = started_run(m)

    events = asyncio.run(collect(m.decide(run_id, {"type": "approve"})))

    assert events[-1]["type"] == "result" and events[-1]["record"]["outreach"] == GOOD


def test_edit_records_the_edited_text():
    m = manager()
    run_id, _ = started_run(m)
    edited = GOOD.replace("Hi Dana", "Hello Dana")

    events = asyncio.run(collect(m.decide(run_id, {"type": "edit", "text": edited})))

    assert events[-1]["record"]["outreach"] == edited


def test_reject_leaves_no_outreach():
    m = manager()
    run_id, _ = started_run(m)

    events = asyncio.run(collect(m.decide(run_id, {"type": "reject", "message": "no"})))

    assert events[-1]["type"] == "result" and not events[-1]["record"].get("outreach")


def test_unknown_lead_raises_key_error():
    with pytest.raises(KeyError):
        asyncio.run(manager().start("nope"))


def test_deciding_an_unknown_or_decided_run_raises_key_error():
    m = manager()
    run_id, _ = started_run(m)
    with pytest.raises(KeyError):
        m.decide("zzz", {"type": "approve"})
    asyncio.run(collect(m.decide(run_id, {"type": "approve"})))

    with pytest.raises(KeyError):
        m.decide(run_id, {"type": "approve"})


def test_drop_forgets_a_pending_run():
    m = manager()
    run_id, _ = started_run(m)

    m.drop(run_id)

    with pytest.raises(KeyError):
        m.decide(run_id, {"type": "approve"})
